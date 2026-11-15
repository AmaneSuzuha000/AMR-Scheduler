"""Local mock bridge: reproduces the proto=1 behaviour of the Ubuntu `live_bridge`.

Why write another bridge on the Windows side
--------------------------------------------
The bridge on the Ubuntu machine is the real thing, but its `--watchdog` flag keeps
being flipped (off while debugging, on during demos). If the tests connect to it directly,
the results drift with whatever the local configuration happens to be, and we can never
answer the question "does the heartbeat actually do anything".

This mock freezes the **protocol and watchdog semantics** into code:
the protocol fields match `~/AMR/tools/live_bridge/live_bridge.py`,
and after a heartbeat timeout it calls `stop()` and **does not recover automatically** —
which is exactly the behaviour observed in practice.

Differences allowed for testing: the internal implementation (threads/timers) may differ.
The contract that must not change is:

* `hello` must be the first message;
* if the client sends no uplink message for 0.5 s -> the pose freezes + `status.error="watchdog"`;
* once frozen, even sending messages afterwards **does not recover it** (as measured).
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time


class MockBridge:
    """Minimal working proto=1 bridge. Listens on a random port by default so tests can run concurrently."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, *,
                 watchdog: float = 0.5, rate: float = 20.0,
                 speed: float = 0.2, frame: str = "sim",
                 vehicle: str = "agv1") -> None:
        self.host = host
        self.port = port
        self.watchdog = watchdog
        self.rate = rate
        self.speed = speed
        self.frame = frame
        self.vehicle = vehicle
        self._srv = None
        self._client = None
        self._stop = threading.Event()
        self._thread = None
        self._last_msg = 0.0
        self._stalled = False
        self._lock = threading.Lock()
        #: counts the ping messages received, so a test can assert that the client really did send heartbeats.
        self.n_pings = 0
        self.n_poses_sent = 0

    # ---------- lifecycle ----------
    def start(self) -> int:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        self._srv = srv
        self.port = srv.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except OSError:
                    pass
                self._client = None
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass

    # ---------- internals ----------
    def _send(self, obj: dict) -> None:
        with self._lock:
            if self._client is None:
                return
            try:
                self._client.sendall(
                    (json.dumps(obj) + "\n").encode("utf-8"))
            except OSError:
                self._client = None

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                c, _addr = self._srv.accept()
            except OSError:
                break
            with self._lock:
                if self._client is not None:
                    try:
                        self._client.close()
                    except OSError:
                        pass
                self._client = c
                self._last_msg = time.monotonic()
                self._stalled = False
                try:
                    c.sendall((json.dumps({
                        "type": "hello", "vehicle": self.vehicle, "proto": 1,
                        "frame": self.frame,
                        "caps": ["ping", "estop", "nav"],
                    }) + "\n").encode("utf-8"))
                except OSError:
                    c.close()
                    self._client = None
                    continue
            threading.Thread(target=self._read, args=(c,), daemon=True).start()

    def _read(self, c) -> None:
        buf = b""
        while not self._stop.is_set():
            try:
                data = c.recv(4096)
            except OSError:
                break
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line.decode("utf-8", "replace"))
                except ValueError:
                    continue
                self._last_msg = time.monotonic()
                if msg.get("type") == "ping":
                    self.n_pings += 1
                    self._send({"type": "pong", "t1": msg.get("t1"),
                                "t2": time.time(), "t3": time.time()})

    def _serve(self) -> None:
        period = 1.0 / self.rate
        seq = 0
        # circular trajectory parameters: radius 0.9 m, angular rate 0.222 rad/s -> linear speed 0.2 m/s
        radius, omega = 0.9, self.speed / 0.9
        phase = 0.0
        last = time.perf_counter()
        next_status = 0.0
        while not self._stop.is_set():
            now = time.perf_counter()
            dt = now - last
            last = now
            if self._client is None:
                time.sleep(period)
                continue
            # watchdog: on timeout, halt and never recover
            if (self.watchdog > 0 and not self._stalled
                    and now - self._last_msg > self.watchdog):
                self._stalled = True
            if not self._stalled:
                phase += omega * dt
            x = radius * math.cos(phase)
            y = radius * math.sin(phase)
            yaw = phase + math.pi / 2
            seq += 1
            self._send({"type": "pose", "vehicle": self.vehicle, "seq": seq,
                        "t": round(time.time(), 3),
                        "x": round(x, 4), "y": round(y, 4),
                        "yaw": round(yaw, 4)})
            self.n_poses_sent += 1
            if now >= next_status:
                self._send({"type": "status", "vehicle": self.vehicle,
                            "battery": None,
                            "mode": "stalled" if self._stalled else "circle",
                            "error": "watchdog" if self._stalled else None})
                next_status = now + 0.2
            time.sleep(period)
