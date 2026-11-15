"""Live vehicle overlay for the console: the bridge pose, on the scheduler's own canvas.

Why this file exists
--------------------
The console could already draw the vehicles its own Simulator steps. What it could not
show was the *other* side of the join: where the vehicle in Gazebo actually was while the
scheduler believed it was somewhere on its graph. Those are two separate claims, and until
both sit on one canvas nobody can see whether the join is right -- the scheduler can be
perfectly self-consistent while the vehicle is in the wrong aisle.

Three rules this module keeps, because breaking any one of them turns a demonstration
into a misleading one:

1. **The channel label is reported, never upgraded.** The bridge declares a frame name in
   its hello (odom in the current setup). That is a *channel name*, not a claim of
   absolute localisation. What sits behind it was measured rather than assumed: on
   2026-11-05 teleporting the Gazebo model to known world points moved the reported (x, y)
   1:1 within 1e-6 m, so in *this* world the channel carries the simulator's entity pose.
   On a real chassis the same channel would carry wheel odometry, which drifts. Either way
   the snapshot carries the label verbatim and never renames it to "position" or
   "absolute" -- the number is only as good as the source, and the source is whatever the
   bridge says it is.

2. **A frozen pose is not a parked vehicle.** If no pose arrives within stale_after
   seconds the snapshot is marked stale and the console has to say so. Drawing the last
   value as though it were live is exactly how a dead bridge gets mistaken for an idle AGV.

3. **The world -> network mapping is the demo frame's inverse and nothing else.** The
   Gazebo world used for the motion demo is the image of the workshop network under
   GzFrame; inverting that same object is therefore exact for that world, and meaningless
   for any other. It is a chosen mapping, not a measurement, so it is injected rather than
   hard-coded here.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from amr_sched.bridge_client import BridgeClient


#: Seconds without a pose after which the value on screen is declared stale. The bridge
#: publishes at 20 Hz, so 1.5 s is thirty missed frames -- long enough not to flicker on a
#: scheduler hiccup, short enough that a stopped bridge is visible before anyone trusts it.
DEFAULT_STALE_AFTER = 1.5


@dataclass(frozen=True)
class LivePose:
    """One bridge pose, already in **network** coordinates."""

    vehicle: str
    x: float
    y: float
    yaw: float
    seq: int
    age_s: float
    stale: bool
    #: The bridge's own channel label, passed through unchanged.
    channel: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "vehicle": self.vehicle,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "yaw": round(self.yaw, 4),
            "seq": self.seq,
            "age_s": round(self.age_s, 3),
            "stale": self.stale,
            "channel": self.channel,
            # Spelled out for the UI so nobody has to infer it from the field name.
            "provenance": "bridge:" + self.channel,
        }


class LiveLink:
    """A background connection to the Ubuntu live_bridge, read-only.

    The console asks for :meth:`snapshot` on every frame; this object never pushes, so a
    slow browser cannot stall the bridge read loop.

    Heartbeat is not optional: the bridge watchdog freezes the pose after 0.5 s without an
    uplink (tools/mock_bridge.py pins that behaviour), so a client that only reads would
    present a permanently frozen vehicle that looks like a stopped one. The ping thread
    inside BridgeClient is what keeps the pose moving.
    """

    def __init__(self, host: str, port: int, *, frame,
                 stale_after: float = DEFAULT_STALE_AFTER,
                 ping_interval: float = 0.2,
                 timeout: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.frame = frame
        self.stale_after = stale_after
        self.client = BridgeClient(host, port, ping_interval=ping_interval,
                                   timeout=timeout)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_monotonic: float | None = None
        self._last_pose = None
        self.error: str | None = None

    # ---------- lifecycle ----------
    def start(self) -> None:
        """Connect and begin reading.

        Raises whatever the socket raises, so the caller can report a refused connection
        instead of showing an empty canvas with no reason.
        """
        self.client.start()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="live-link-%s" % self.host)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.client.poll(0.2)
            except RuntimeError:
                break
            except OSError as exc:
                self.error = str(exc)
                break
            pose = self.client.last_pose
            if pose is not None:
                self._last_pose = pose
                self._last_monotonic = time.monotonic()

    def stop(self) -> None:
        self._stop.set()
        self.client.stop()

    # ---------- read ----------
    def snapshot(self) -> LivePose | None:
        """Latest pose in network coordinates, or None before the first one arrives."""
        pose = self._last_pose
        if pose is None or self._last_monotonic is None:
            return None
        age = time.monotonic() - self._last_monotonic
        x, y = self.frame.to_network(pose.x, pose.y)
        yaw = self.frame.to_network_yaw(pose.yaw)
        status = self.client.last_status
        return LivePose(
            vehicle=pose.vehicle or "agv?",
            x=x, y=y, yaw=yaw, seq=pose.seq,
            age_s=age,
            # The bridge's own status is the stronger signal. Its watchdog means the vehicle
            # is held still; its odom_stale means the bridge stopped receiving poses at all.
            # In both cases poses keep arriving at 20 Hz with an unchanged value, so age
            # alone would look perfectly fresh -- this is the case that has to be caught
            # here, because nothing downstream can tell a frozen pose from a parked one.
            stale=bool(age > self.stale_after
                       or (status is not None and status.pose_is_stale)),
            channel=str(self.client.hello.get("frame") or "?"),
        )


class LiveRegistry:
    """At most one live link at a time -- the console has one canvas and one vehicle."""

    def __init__(self) -> None:
        self.link: LiveLink | None = None

    def attach(self, host: str, port: int, *, frame,
               stale_after: float = DEFAULT_STALE_AFTER) -> LiveLink:
        self.detach()
        link = LiveLink(host, port, frame=frame, stale_after=stale_after)
        link.start()
        self.link = link
        return link

    def detach(self) -> None:
        if self.link is not None:
            self.link.stop()
            self.link = None

    def snapshot(self) -> dict[str, Any] | None:
        if self.link is None:
            return None
        pose = self.link.snapshot()
        if pose is None:
            return {"attached": True, "host": self.link.host, "port": self.link.port,
                    "waiting": True, "error": self.link.error}
        out = pose.as_dict()
        out.update({"attached": True, "host": self.link.host, "port": self.link.port})
        return out
