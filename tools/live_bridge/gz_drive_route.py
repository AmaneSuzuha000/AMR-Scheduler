#!/usr/bin/env python3
"""Drive a route that the *scheduler* planned, inside Gazebo.

This is the end-to-end proof for the motion demo. The waypoints are not written
down here: they are produced by amr_sched's static Dijkstra planner over the
workshop network, converted to Gazebo world coordinates through DEMO_FRAME, and
then followed by the robot. Deviation is measured against Gazebo ground truth
("gz model -m scout_mini -i"), never against /odom.

Run on the Ubuntu host where Gazebo is already up:

    python3 tools/live_bridge/gz_drive_route.py --start D --goal W3
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from tools.live_bridge.gz_frame import DEMO_FRAME


def sh(cmd, timeout=25):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def truth(model="scout_mini"):
    """Ground-truth pose from Gazebo. /odom is wheel integration and drifts."""
    r = sh("gz model -m %s -i 2>/dev/null" % model)
    if not r or r.returncode != 0:
        return None
    m = re.search(r"position \{\s*x: (-?[0-9.eE+-]+)\s*y: (-?[0-9.eE+-]+)", r.stdout)
    if not m:
        return None
    x, y = float(m.group(1)), float(m.group(2))
    ro = re.search(
        r"orientation \{\s*x: (-?[0-9.eE+-]+)\s*y: (-?[0-9.eE+-]+)"
        r"\s*z: (-?[0-9.eE+-]+)\s*w: (-?[0-9.eE+-]+)", r.stdout)
    yaw = 0.0
    if ro:
        qx, qy, qz, qw = (float(ro.group(i)) for i in (1, 2, 3, 4))
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return x, y, yaw


def drive(vx, wz, secs):
    # --times makes the publisher exit by itself. A leftover "ros2 topic pub"
    # keeps pushing /cmd_vel and the robot drives away on its own; that exact
    # accident invalidated an earlier measurement batch.
    n = max(1, int(secs * 10))
    sh("ros2 topic pub --times %d -r 10 /cmd_vel geometry_msgs/msg/Twist "
       "'{linear: {x: %.3f}, angular: {z: %.3f}}' >/dev/null 2>&1" % (n, vx, wz),
       timeout=secs + 12)


def stop_all():
    """Stop the vehicle, then remove any leftover publisher.

    The zero command is not optional and must come first.
    libgazebo_ros_planar_move latches the last Twist it received and keeps integrating it,
    so killing the publisher only stops *new* commands; the old one stays in force. Measured
    here: after a route reached its goal and stop_all() had already run, the robot was still
    moving and had travelled more than a metre past the goal, and a later run started half a
    metre away from where the previous one had finished. This function used to kill the
    publisher without ever sending a zero.
    """
    sh("ros2 topic pub --times 3 -r 10 /cmd_vel geometry_msgs/msg/Twist "
       "'{linear: {x: 0.0}, angular: {z: 0.0}}' >/dev/null 2>&1", timeout=20)
    # The bracketed pattern keeps pkill from matching the shell that runs this very command;
    # an unbracketed pattern makes pkill kill its own parent and the kill then never lands.
    sh("pkill -f 'ros2 topic [p]ub' >/dev/null 2>&1 || true")


def plan_route(start_node, goal_node, tasks=5, agvs=1, seed=0):
    """Ask the scheduler for the node sequence between two nodes.

    The planner needs a reservation table and a horizon even for the static case,
    because the interface is shared with the space-time planner. An empty table is
    correct here: the static planner ignores reservations by design.
    """
    from amr_sched.reservation import ReservationTable

    net, _tasks, _agvs = generate_scenario(ScenarioConfig(n_tasks=tasks, n_agvs=agvs, seed=seed))
    planner = StaticDijkstraPlanner(max_speed=1.0, dt=1.0)
    plan = planner.plan(
        network=net,
        start=start_node,
        goal=goal_node,
        start_step=0,
        reservations=ReservationTable(),
        agv_id="gz",
        horizon=20000,
    )
    if plan is None:
        raise SystemExit("no route from %s to %s" % (start_node, goal_node))
    # PlannedPath.nodes already excludes the start and ends at the goal.
    nodes = [start_node] + list(plan.nodes)
    return net, nodes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="D")
    ap.add_argument("--goal", default="W3")
    ap.add_argument("--model", default="scout_mini")
    ap.add_argument("--reach", type=float, default=0.35)
    ap.add_argument("--max-iters", type=int, default=400)
    args = ap.parse_args(argv)

    net, nodes = plan_route(args.start, args.goal)
    waypoints = DEMO_FRAME.path_to_world(nodes, net)

    print("planned route: %s" % " -> ".join(nodes))
    print("world waypoints: %s" % ", ".join("(%.2f,%.2f)" % p for p in waypoints))

    p = truth(args.model)
    if p is None:
        print("FAIL: cannot read ground truth for model %r" % args.model)
        return 2
    print("start pose: x=%.3f y=%.3f yaw=%.3f" % p)

    reached = 0
    for idx, (gx, gy) in enumerate(waypoints):
        ok = False
        for _ in range(args.max_iters // max(1, len(waypoints))):
            p = truth(args.model)
            if p is None:
                stop_all()
                print("FAIL: lost ground truth")
                return 2
            x, y, yaw = p
            dist = math.hypot(gx - x, gy - y)
            err = math.atan2(gy - y, gx - x) - yaw
            err = (err + math.pi) % (2.0 * math.pi) - math.pi
            if dist < args.reach:
                reached += 1
                ok = True
                print("  wp%d %-4s reached x=%.3f y=%.3f dist=%.3f"
                      % (idx, nodes[idx + 1] if idx + 1 < len(nodes) else "end", x, y, dist))
                break
            # The planar_move plugin ignores a pure angular command, so the linear
            # speed is never allowed to reach zero while turning.
            wz = max(-1.2, min(1.2, 2.2 * err))
            vx = 0.06 if abs(err) > 0.4 else min(0.20, 0.6 * dist + 0.05)
            drive(vx, wz, 0.5)
        if not ok:
            stop_all()
            print("  wp%d TIMEOUT at x=%.3f y=%.3f dist=%.3f" % (idx, x, y, dist))
            print("reached %d/%d" % (reached, len(waypoints)))
            return 1

    stop_all()
    print("reached %d/%d waypoints" % (reached, len(waypoints)))
    print("PASS" if reached == len(waypoints) else "FAIL")
    return 0 if reached == len(waypoints) else 1


if __name__ == "__main__":
    raise SystemExit(main())
