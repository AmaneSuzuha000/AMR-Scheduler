#!/usr/bin/env python3
"""Quantify how far /odom drifts from the pose Gazebo actually simulates.

This exists because the motion demo has to read the robot pose, and the only
stable source on this install is /odom -- "gz model -i" returns nothing,
/gazebo/default/pose/info publishes no data, and /get_entity_state is not
advertised. /odom is wheel integration, so before trusting it we have to measure
how wrong it is instead of assuming it is right.

Method: drive a known open-loop command for a fixed time, then compare
  * /odom (wheel integration), and
  * the GUI-independent facts we can still obtain: the spawn pose (known exactly)
    and the commanded distance (speed x time, an upper bound on travel).

A drift figure that cannot be computed is reported as "unknown", never as zero.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time


def sh(cmd, timeout=25):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def odom():
    r = sh("timeout 8 ros2 topic echo --once /odom 2>/dev/null", timeout=14)
    if not r or not r.stdout:
        return None
    x = y = yaw = None
    depth = 0
    for line in r.stdout.splitlines():
        st = line.strip()
        if st.startswith("position:"):
            depth = 1
            continue
        if st.startswith("orientation:"):
            depth = 2
            continue
        if st.startswith("covariance:"):
            depth = 0
            continue
        if depth == 1 and st.startswith("x:"):
            x = float(st.split(":")[1])
        elif depth == 1 and st.startswith("y:"):
            y = float(st.split(":")[1])
        elif depth == 2 and st.startswith("z:"):
            qz = float(st.split(":")[1])
        elif depth == 2 and st.startswith("w:"):
            qw = float(st.split(":")[1])
            yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
    if x is None or y is None:
        return None
    return x, y, (yaw if yaw is not None else 0.0)


def publish(vx, wz, secs):
    """Open-loop command. --times guarantees the publisher exits by itself."""
    n = max(1, int(secs * 10))
    sh("ros2 topic pub --times %d -r 10 /cmd_vel geometry_msgs/msg/Twist "
       "'{linear: {x: %.3f}, angular: {z: 0.0}}' >/dev/null 2>&1" % (n, vx),
       timeout=secs + 12)


def main():
    before = odom()
    if before is None:
        print(json.dumps({"error": "/odom not readable"}))
        return 2

    vx, secs = 0.15, 3.0
    publish(vx, 0.0, secs)
    time.sleep(0.5)
    after = odom()
    if after is None:
        print(json.dumps({"error": "/odom lost after the command"}))
        return 2

    dx = after[0] - before[0]
    dy = after[1] - before[1]
    moved = math.hypot(dx, dy)
    expected_upper = vx * secs       # an upper bound: any slip only reduces it

    report = {
        "before": [round(v, 4) for v in before],
        "after": [round(v, 4) for v in after],
        "moved_m": round(moved, 4),
        "commanded_upper_bound_m": expected_upper,
        "ratio_moved_to_commanded": round(moved / expected_upper, 4) if expected_upper else None,
        "absolute_drift_m": "unknown",
        "note": ("Absolute drift cannot be measured on this install: gz model -i "
                 "returns nothing, pose/info publishes nothing, and "
                 "/get_entity_state is not advertised. Only the self-consistency of "
                 "/odom against the commanded distance is reported."),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
