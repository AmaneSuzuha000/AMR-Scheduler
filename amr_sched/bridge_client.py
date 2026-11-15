"""Windows side: bring the Ubuntu live_bridge pose stream into the workshop map.

Why it cannot be consumed directly
---------------------------------
The Ubuntu side (`~/AMR/tools/live_bridge/live_bridge.py`) speaks **proto=1**:

    {"type":"hello","vehicle":"agv1","proto":1,"frame":"odom","caps":[...]}
    {"type":"pose","vehicle":"agv1","seq":1,"t":...,"x":1.07,"y":2.29,"yaw":-2.64}
    {"type":"status","vehicle":"agv1","battery":null,"mode":"circle","error":null}

The C# `LiveClient.PoseToXY` on Windows, however, understands only **graph-semantic** poses
(`node` / `edge` / `progress`), because it interpolates points along the workshop network
in order to draw them.
The two are not describing the same thing at all: one is a **metric world coordinate**, the
other is **progress between graph nodes**.

So this layer has to exist in between. It does three things:

1. **Coordinate calibration**: map the bridge's `(x, y)` into workshop-map coordinates.
   This is not a matter of dividing by a scale factor. The bridge's origin, orientation and
   scale
   may all differ from the workshop map, so the transform must be stated explicitly and its
   residuals checked as part of calibrating it.
2. **Heartbeat keep-alive**: the bridge defaults to `--watchdog 0.5`, so if the client sends
   nothing upstream for 0.5 s it calls `stop()`,
   and **it does not recover on its own**. A read-only client must send `ping` periodically,
   otherwise what it sees is a frozen pose.
3. **Provenance tagging**: `frame=odom` is wheel odometry and drifts; `frame=sim` is
   simulation ground truth.
   The two carry completely different authority and must not be written into the same
   column without being told apart.

**What it does not do**: it makes no claim that a gazebo/circle pose is equivalent to
real-vehicle localisation,
and it never treats a drifted odom coordinate as an absolute position in the workshop.
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

#: Bridge protocol version, matching the `proto` field on the Ubuntu side.
BRIDGE_PROTO = 1


@dataclass
class Pose2D:
    """One pose frame as reported by the bridge, in metres."""

    x: float
    y: float
    yaw: float
    seq: int
    t_wall: float
    vehicle: str = "agv1"


@dataclass
class BridgeStatus:
    """Status as reported by the bridge. When ``error`` is set, stop trusting the pose
    immediately."""

    mode: str | None = None
    error: str | None = None
    battery: float | None = None
    t_wall: float = 0.0

    #: Errors after which the pose on the wire must not be mistaken for a live position.
    #: odom_stale is the odd one out: the vehicle was never told to stop, the bridge simply
    #: stopped receiving poses from its own source. On screen that is indistinguishable from
    #: a parked vehicle, which is exactly why it has to be listed here.
    UNTRUSTWORTHY = ("watchdog", "estop", "odom_stale")

    @property
    def is_stalled(self) -> bool:
        """When the watchdog or estop trips, the bridge halts the vehicle and the pose stops
        changing."""
        return self.error in ("watchdog", "estop")

    @property
    def pose_is_stale(self) -> bool:
        """True when the pose must be drawn as stale rather than as a live position."""
        return self.error in self.UNTRUSTWORTHY


@dataclass(frozen=True)
class AffineCalib:
    """A 2-D similarity transform (rotation, scale, translation) taking bridge coordinates
    into workshop-map coordinates.

    Why a similarity transform rather than an affine one: between the workshop map and the
    chassis frame there should only ever be
    translation, rotation and uniform scaling -- never shear. If a fitted matrix shows clear
    shear,
    the **correspondence points are wrong**, and that should be reported rather than forced
    through a matrix that happens to hit the points.

    The mapping is::

        [X]   [s*cos -s*sin] [x]   [tx]
        [Y] = [s*sin  s*cos] [y] + [ty]
    """

    scale: float = 1.0
    theta: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    #: `odom` and `sim` may calibrate differently; this records which one the parameters
    #: belong to.
    frame: str = "odom"

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("scale must be positive, got %r" % (self.scale,))

    def apply(self, x: float, y: float) -> tuple:
        c, s = math.cos(self.theta), math.sin(self.theta)
        X = self.scale * (c * x - s * y) + self.tx
        Y = self.scale * (s * x + c * y) + self.ty
        return X, Y

    def apply_yaw(self, yaw: float) -> float:
        """Heading has to rotate too, or the vehicle points the wrong way."""
        return _wrap(yaw + self.theta)

    @classmethod
    def from_pairs(cls, pairs, frame: str = "odom") -> "AffineCalib":
        """Fit a similarity transform from two point sets, by least squares.

        Parameters
        ----------
        pairs:
            ``[((bx, by), (mx, my)), ...]``, bridge to map, at least 2 pairs.

        Raises
        ------
        ValueError
            Too few points, or the sets are degenerate (coincident or collinear, leaving the
            scale unsolvable).
        """
        pts = list(pairs)
        if len(pts) < 2:
            raise ValueError("need at least 2 correspondence pairs, got %d" % len(pts))
        # Solve in complex form: m = a*b + c, with the least-squares solution
        # c = mean(m) - a*mean(b).
        bs = [complex(bx, by) for (bx, by), _ in pts]
        ms = [complex(mx, my) for _, (mx, my) in pts]
        nb, nm = len(bs), len(ms)
        mb = sum(bs) / nb
        mm = sum(ms) / nm
        num = sum((m - mm) * (b - mb).conjugate() for b, m in zip(bs, ms))
        den = sum(abs(b - mb) ** 2 for b in bs)
        if den < 1e-12:
            raise ValueError("all bridge points coincide, so no scale can be recovered")
        a = num / den
        if abs(a) < 1e-12:
            raise ValueError("the point sets do not correspond (fitted scale is zero)")
        c = mm - a * mb
        return cls(scale=abs(a), theta=math.atan2(a.imag, a.real),
                   tx=c.real, ty=c.imag, frame=frame)

    def residuals(self, pairs) -> list:
        """Per-point residuals, in metres in map coordinates.

        Calibration quality has to be judged on this. A transform that merely solved is not
        evidence of anything.
        """
        out = []
        for (bx, by), (mx, my) in pairs:
            X, Y = self.apply(bx, by)
            out.append(math.hypot(X - mx, Y - my))
        return out


def _wrap(a: float) -> float:
    """Normalise an angle into (-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a <= -math.pi:
        a += 2 * math.pi
    return a


class BridgeClient:
    """A read-only client for the Ubuntu `live_bridge`, with heartbeat.

    Why ping is mandatory: see the module docstring on the bridge watchdog. It is not
    optional --
    a connection that does not ping sees a frozen pose after 0.5 s.
    """

    def __init__(self, host: str, port: int = 9300, *,
                 ping_interval: float = 0.2,
                 on_pose: Callable[[Pose2D], None] | None = None,
                 on_status: Callable[[BridgeStatus], None] | None = None,
                 on_hello: Callable[[dict], None] | None = None,
                 timeout: float = 10.0) -> None:
        self.host = host
        self.port = port
        self.ping_interval = ping_interval
        self.on_pose = on_pose
        self.on_status = on_status
        self.on_hello = on_hello
        self.timeout = timeout
        self._sock = None
        self._buf = b""
        self._stop = threading.Event()
        self._ping_thread = None
        self._lock = threading.Lock()
        #: Most recent pose and status, for the visualiser to poll.
        self.last_pose: Pose2D | None = None
        self.last_status: BridgeStatus | None = None
        self.hello: dict = {}
        self.n_poses = 0
        self.n_pings = 0

    # ---------- connection ----------
    def connect(self) -> None:
        s = socket.create_connection((self.host, self.port), timeout=self.timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(1.0)
        self._sock = s
        self._stop.clear()

    def _send(self, obj: dict) -> None:
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            if self._sock is None:
                return
            self._sock.sendall(data)

    def ping(self) -> None:
        """Send one heartbeat. ``t1`` is used to compute round-trip time when `pong`
        comes back."""
        self._send({"type": "ping", "t1": time.time()})
        self.n_pings += 1

    def estop(self) -> None:
        self._send({"type": "estop"})

    def estop_release(self) -> None:
        self._send({"type": "estop_release"})

    def _ping_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.ping()
            except OSError:
                return
            self._stop.wait(self.ping_interval)

    def start(self) -> None:
        """Connect and start: ping once, start the heartbeat thread, then enter the read
        loop."""
        self.connect()
        self.ping()          # feed it immediately, so the first watchdog period does not trip
        self._ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        self._ping_thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

    # ---------- read ----------
    def _dispatch(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "hello":
            self.hello = msg
            if self.on_hello:
                self.on_hello(msg)
        elif kind == "pose":
            p = Pose2D(x=float(msg["x"]), y=float(msg["y"]),
                       yaw=float(msg["yaw"]), seq=int(msg["seq"]),
                       t_wall=float(msg.get("t", 0.0)),
                       vehicle=str(msg.get("vehicle", "")))
            self.last_pose = p
            self.n_poses += 1
            if self.on_pose:
                self.on_pose(p)
        elif kind == "status":
            st = BridgeStatus(mode=msg.get("mode"), error=msg.get("error"),
                              battery=msg.get("battery"),
                              t_wall=time.time())
            self.last_status = st
            if self.on_status:
                self.on_status(st)
        # pong / nav_result do not affect drawing today; a caller can extend them as needed.

    def poll(self, duration: float) -> int:
        """Read and dispatch messages for ``duration`` seconds, returning how many were
        handled.

        Blocking, which suits a script that wants to capture N seconds. A GUI integration
        can run it on a thread.
        """
        if self._sock is None:
            raise RuntimeError("connect() has not been called")
        deadline = time.monotonic() + duration
        n = 0
        while time.monotonic() < deadline and not self._stop.is_set():
            try:
                chunk = self._sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            self._buf += chunk
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line.decode("utf-8", "replace"))
                except (ValueError, UnicodeDecodeError):
                    continue
                self._dispatch(msg)
                n += 1
        return n


def to_map_pose(pose: Pose2D, calib: AffineCalib) -> tuple:
    """Convert a bridge pose into workshop-map coordinates ``(X, Y, yaw)``."""
    X, Y = calib.apply(pose.x, pose.y)
    return X, Y, calib.apply_yaw(pose.yaw)
