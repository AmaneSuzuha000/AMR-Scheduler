#!/usr/bin/env python3
"""Measure how fast the follow loop can actually observe and command.

Written because the loop kept failing in ways that looked like controller bugs. Before
changing any more gains, the honest question is: how many control decisions per second
does this actually make, and what is the delay between deciding and observing the result?

A loop that reads the pose, decides, publishes, and then sleeps is a different system from
one that does it ten times a second, and the difference dominates at 0.3 m/s.
"""
from __future__ import annotations

import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "scout_mini"

    proc = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "gz_pose_node.py"), model],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1)
    if proc.stdout.readline().strip() != "ready":
        print("pose node failed")
        return 1

    def pose():
        proc.stdin.write("POSE %s\n" % model)
        proc.stdin.flush()
        return proc.stdout.readline().split()

    # 1. Pose read latency
    times = []
    for _ in range(30):
        t0 = time.time()
        pose()
        times.append(time.time() - t0)
    print("pose read      : median %.1f ms, max %.1f ms"
          % (statistics.median(times) * 1e3, max(times) * 1e3))

    # 2. Cost of restarting the publisher, which the loop does whenever the command
    #    changes quantisation step.
    t0 = time.time()
    p = subprocess.Popen(
        "ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "
        "'{linear: {x: 0.0}, angular: {z: 0.0}}'",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    spawn = time.time() - t0
    p.terminate()
    p.wait()
    print("publisher spawn: %.0f ms" % (spawn * 1e3))

    # 3. Achievable loop rate with the current sleep
    t0 = time.time()
    n = 0
    while time.time() - t0 < 2.0:
        pose()
        time.sleep(0.05)
        n += 1
    rate = n / (time.time() - t0)
    print("loop with 50 ms sleep: %.1f Hz" % rate)

    proc.stdin.write("QUIT\n")
    proc.stdin.flush()
    proc.terminate()
    print()
    print("At 0.30 m/s, %.1f Hz means one decision every %.2f m of travel."
          % (rate, 0.30 / rate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())