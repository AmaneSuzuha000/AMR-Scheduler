#!/usr/bin/env python3
"""One command that proves the console and Gazebo are looking at the same vehicle.

Why this exists
---------------
Each link of the chain had its own test: the scheduler plans a route, the exporter scales it
into Gazebo coordinates, the follower drives it, the bridge reports /odom, the console draws
it. Every one of those can pass while the *join* is wrong -- the console could be drawing a
vehicle that is nowhere near the one Gazebo is moving. That failure is invisible from any
single test, and it is exactly the failure the demo would be dishonest about.

So this script samples the console's own HTTP endpoint while Gazebo really drives a route,
and then compares the console's last position against Gazebo's ground truth
("gz model -m scout_mini -i", never /odom). It exits non-zero on any of:

  * the console never receives a pose (the bridge or the overlay is dead)
  * the route does not complete inside the follower
  * any sample is flagged stale (a frozen pose would silently pass otherwise)
  * the channel label is not reported verbatim
  * the console position and Gazebo disagree by more than TOLERANCE_M

Requires the Ubuntu host with Gazebo up and the odom bridge listening on 9300.

    python tools/verify_console_gazebo_e2e.py --host 127.0.0.1 --port 9300
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = __file__.rsplit("tools", 1)[0]
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#: How far the console's idea of the vehicle may sit from the simulator's own entity pose.
#: 5 cm is a tenth of the 0.45 m arrival tolerance used by the follower, so a wrong join
#: (a stale pose, a wrong scale, an inverted axis) cannot hide inside it.
TOLERANCE_M = 0.05

#: The follower's own report of a completed route.
WAYPOINT_TOLERANCE_M = 0.45


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def ssh(host: str, inner: str, timeout: float = 180.0) -> subprocess.CompletedProcess:
    """Run a command on the chassis host with a login shell (ROS needs its setup.bash)."""
    target = host if "@" in host else f"{os.environ.get('AMR_BRIDGE_USER', 'ubuntu')}@{host}"
    return subprocess.run(["ssh", "-o", "BatchMode=yes", target,
                           "bash -lc %s" % json.dumps(inner)],
                          capture_output=True, text=True, timeout=timeout)


def gz_truth(host: str) -> dict:
    """The simulator's own entity pose -- the only ground truth in this chain."""
    proc = ssh(host, "gz model -m scout_mini -i | grep -A2 'position {' | sed -n '2,3p'",
               timeout=60)
    out = {}
    for line in proc.stdout.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out


class Console:
    """The console under test, in its own thread, on a port nobody else is using."""

    def __init__(self) -> None:
        import uvicorn
        from web.server import app
        self.port = free_port()
        self.base = "http://127.0.0.1:%d" % self.port
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1",
                                                    port=self.port, log_level="error"))
        threading.Thread(target=self._server.run, daemon=True).start()
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            try:
                self.get("/api/live")
                return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError("console did not come up on %s" % self.base)

    def stop(self) -> None:
        self._server.should_exit = True

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read())

    def post(self, path: str, body: dict):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def delete(self, path: str) -> dict:
        req = urllib.request.Request(self.base + path, method="DELETE")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def live(self) -> dict:
        """The pose the console would paint, or {} while it is still waiting."""
        return self.get("/api/live").get("live") or {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("AMR_BRIDGE_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=9300)
    ap.add_argument("--route", default="R0->B1")
    ap.add_argument("--routes-file", default="~/AMR/tools/live_bridge/routes.json")
    ap.add_argument("--domain", default="42")
    a = ap.parse_args(argv)

    problems: list[str] = []
    console = Console()
    try:
        print("== console ==", console.base)
        geom = console.get("/api/map?layout=ring")
        print("   ring geometry: %d nodes / %d edges" % (len(geom["nodes"]), len(geom["edges"])))

        print("== attach bridge %s:%d ==" % (a.host, a.port))
        status, body = console.post("/api/live/attach", {"host": a.host, "port": a.port})
        if status != 200:
            print("   FAIL attach returned %d: %s" % (status, body))
            return 1

        live: dict = {}
        for _ in range(200):
            live = console.live()
            if live.get("x") is not None:
                break
            time.sleep(0.1)
        if live.get("x") is None:
            print("   FAIL no pose reached the console in 20 s")
            return 1
        print("   first pose:", {k: live.get(k) for k in ("x", "y", "yaw", "channel")})
        if live.get("channel") != "odom":
            problems.append("channel label %r was not reported verbatim" % live.get("channel"))

        print("== drive %s in Gazebo (real motion) ==" % a.route)
        inner = ("cd ~/AMR && source /opt/ros/humble/setup.bash && "
                 "source ~/scout_gazebo_ws/install/setup.bash && "
                 "export ROS_DOMAIN_ID=%s && "
                 "python3 tools/live_bridge/gz_follow_routes.py --routes %s --only %s"
                 # the route name contains "->", which an unquoted shell would read as a
                 # redirection and silently turn into a different (failing) command.
                 % (a.domain, a.routes_file, json.dumps(a.route)))
        target = a.host if "@" in a.host else f"{os.environ.get('AMR_BRIDGE_USER', 'ubuntu')}@{a.host}"
        proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", target,
                                 "bash -lc %s" % json.dumps(inner)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        samples: list[dict] = []
        t0 = time.monotonic()
        while proc.poll() is None and time.monotonic() - t0 < 180:
            snap = console.live()
            if snap.get("x") is not None:
                samples.append(snap)
            time.sleep(0.25)
        follower_log = proc.stdout.read() if proc.stdout else ""
        for line in follower_log.strip().splitlines()[-6:]:
            print("   " + line)

        if "PASS" not in follower_log:
            problems.append("the follower did not report a completed route")

        print("== console samples ==")
        print("   n = %d" % len(samples))
        if len(samples) < 10:
            problems.append("only %d samples arrived; the stream is not being consumed"
                            % len(samples))
        else:
            stale = [s for s in samples if s.get("stale")]
            if stale:
                problems.append("%d/%d samples were flagged stale" % (len(stale), len(samples)))
            moved = max(abs(s["x"] - samples[0]["x"]) for s in samples)
            moved = max(moved, max(abs(s["y"] - samples[0]["y"]) for s in samples))
            print("   largest excursion %.2f network m" % moved)
            if moved < 10.0:
                problems.append("the console saw the vehicle move only %.2f network m" % moved)
            print("   channels %s  provenance %s"
                  % ({s.get("channel") for s in samples},
                     {s.get("provenance") for s in samples}))

        truth = gz_truth(a.host)
        print("== Gazebo ground truth ==", truth)
        if not truth:
            problems.append("could not read the Gazebo ground truth")
        elif samples:
            last = samples[-1]
            # The demo frame scales the workshop network by 1/25 into the Gazebo world.
            wx, wy = last["x"] / 25.0, last["y"] / 25.0
            dx, dy = wx - truth.get("x", 0.0), wy - truth.get("y", 0.0)
            dist = (dx * dx + dy * dy) ** 0.5
            print("   console (%.4f, %.4f) world  vs  gz (%.4f, %.4f)"
                  % (wx, wy, truth.get("x", 0.0), truth.get("y", 0.0)))
            print("   |delta| = %.4f m (tolerance %.2f m)" % (dist, TOLERANCE_M))
            if dist > TOLERANCE_M:
                problems.append("console and Gazebo disagree by %.3f m" % dist)

        console.delete("/api/live")
    finally:
        console.stop()

    print()
    if problems:
        print("FAIL")
        for p in problems:
            print("  - " + p)
        return 1
    print("PASS  console and Gazebo are looking at the same vehicle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
