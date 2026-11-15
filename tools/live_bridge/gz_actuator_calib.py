#!/usr/bin/env python3
"""Measure what a /cmd_vel command actually does to the vehicle in Gazebo.

Why this exists:

The follow loop was tuned by trial and error, and every failure looked like a control
bug. Measured facts instead: the planar_move plugin ignores a pure angular command, and
with vx non-zero it delivers only about half the commanded yaw rate. Guessing the
missing factor produced a controller that orbited its waypoints.

This script drives a fixed command for a fixed time and reports the displacement and
rotation actually achieved, so the controller gains can be derived from measurement.

Uses gz_pose_node (the /get_entity_state bridge) for truth, never /odom.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


class Pose:
    def __init__(self, model: str):
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "gz_pose_node.py"), model],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        line = self.proc.stdout.readline().strip()
        if line != "ready":
            err = self.proc.stderr.read()[-400:] if self.proc.stderr else ""
            raise RuntimeError("pose node did not start: %r %s" % (line, err))

    def ask(self, cmd: str) -> str:
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        return self.proc.stdout.readline().strip()

    def get(self, model: str):
        parts = self.ask("POSE %s" % model).split()
        if len(parts) != 3:
            raise RuntimeError("bad pose reply: %r" % parts)
        return float(parts[0]), float(parts[1]), float(parts[2])

    def set(self, model: str, x: float, y: float, yaw: float) -> None:
        reply = self.ask("SET %s %.4f %.4f %.4f" % (model, x, y, yaw))
        if reply != "ok":
            raise RuntimeError("set failed: %r" % reply)

    def close(self) -> None:
        try:
            self.ask("QUIT")
        except Exception:
            pass
        self.proc.terminate()


def drive(vx: float, wz: float, secs: float, hz: float = 20.0):
    """Publish a constant command for ``secs`` and return the process."""
    cmd = ("ros2 topic pub -r %g /cmd_vel geometry_msgs/msg/Twist "
           "'{linear: {x: %.3f}, angular: {z: %.3f}}'") % (hz, vx, wz)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    return proc


def _send_zero_twist() -> None:
    """Publish one zero Twist.

    Not optional: libgazebo_ros_planar_move latches the last command it receives, so a
    publisher that is killed without a zero in front of it leaves the vehicle driving at the
    last commanded speed for ever. Measured: a route that had already printed PASS left the
    robot metres past its goal and still moving.
    """
    subprocess.run(
        "ros2 topic pub --times 3 -r 10 /cmd_vel geometry_msgs/msg/Twist "
        "'{linear: {x: 0.0}, angular: {z: 0.0}}' >/dev/null 2>&1",
        shell=True, capture_output=True, timeout=20)


def _kill_publishers() -> None:
    """Kill leftover publishers. Bracketed so pkill cannot match its own command line."""
    subprocess.run("pkill -f 'ros2 topic [p]ub'", shell=True, capture_output=True)


def stop() -> None:
    _send_zero_twist()
    _kill_publishers()


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def measure(pose: Pose, model: str, vx: float, wz: float, secs: float):
    pose.set(model, 0.0, 0.0, 0.0)
    time.sleep(0.4)
    x0, y0, yaw0 = pose.get(model)

    proc = drive(vx, wz, secs)
    t0 = time.time()
    samples = []
    while time.time() - t0 < secs:
        try:
            samples.append(pose.get(model))
        except RuntimeError:
            pass
        time.sleep(0.1)
    proc.terminate()
    stop()
    time.sleep(0.5)

    x1, y1, yaw1 = pose.get(model)
    dist = math.hypot(x1 - x0, y1 - y0)
    dyaw = abs(wrap(yaw1 - yaw0))
    return dist, dyaw, len(samples), (x1, y1, yaw1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="scout_mini")
    ap.add_argument("--secs", type=float, default=3.0)
    args = ap.parse_args(argv)

    pose = Pose(args.model)
    stop()
    print("model=%s  duration=%.1fs each" % (args.model, args.secs))
    print()
    print("%-9s %-9s %-9s %-9s %-9s %-7s" % (
        "vx_cmd", "wz_cmd", "dist_m", "vx_real", "yawdeg", "wz_real"))

    cases = [
        (0.00, 1.20),   # pure rotation at a modest rate
        (0.00, 2.50),   # pure rotation at the clamp the follower uses
        (0.00, 5.00),   # pure rotation, well past the clamp
        (0.10, 2.50),   # crawl + hard turn
        (0.10, 5.00),
        (0.12, 1.30),
        (0.10, 0.00),   # straight at a crawl
        (0.20, 0.00),
        (0.30, 0.00),
    ]
    for vx, wz in cases:
        dist, dyaw, n, end = measure(pose, args.model, vx, wz, args.secs)
        vx_real = dist / args.secs
        wz_real = dyaw / args.secs
        print("%-9.2f %-9.2f %-9.3f %-9.3f %-9.1f %-7.2f" % (
            vx, wz, dist, vx_real, math.degrees(dyaw), wz_real))

    pose.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())