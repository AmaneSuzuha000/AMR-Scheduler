"""Drive a route planned over the *workshop scene* inside Gazebo.

The waypoints come from the scene file through the scheduler planner, then through
GzFrame at 1/25. Nothing here invents coordinates: if the scene and the world
disagree, the vehicle simply fails to reach the waypoints.

Run on the Ubuntu host with Gazebo already up:

    python3 tools/live_bridge/gz_drive_scene_route.py --start J_S_8 --goal J_S_125
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "live_bridge"))

from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.reservation import ReservationTable
from amr_sched.workshop import WorkshopScene
from gz_frame import GzFrame
from gz_drive_route import drive, stop_all, truth


def plan(scene, start: str, goal: str):
    net = scene.network()
    planner = StaticDijkstraPlanner(max_speed=1.0, dt=1.0)
    p = planner.plan(
        network=net, start=start, goal=goal, start_step=0,
        reservations=ReservationTable(), agv_id="gz", horizon=20000,
    )
    if p is None:
        raise SystemExit("no route %s -> %s" % (start, goal))
    return [start] + list(p.nodes)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="J_S_8")
    ap.add_argument("--goal", default="J_S_125")
    ap.add_argument("--model", default="scout_mini")
    ap.add_argument("--reach", type=float, default=0.20)
    ap.add_argument("--csv", default="scene_route_run.csv")
    args = ap.parse_args(argv)

    scene = WorkshopScene.load()
    nodes = plan(scene, args.start, args.goal)
    frame = GzFrame(network_origin=(0.0, 0.0), origin=(0.0, 0.0), scale=1.0 / 25.0)
    waypoints = []
    for nid in nodes:
        n = scene.nodes[nid]
        waypoints.append(frame.to_world(n.x, n.y))

    net_len = sum(
        math.hypot(scene.nodes[nodes[i + 1]].x - scene.nodes[nodes[i]].x,
                   scene.nodes[nodes[i + 1]].y - scene.nodes[nodes[i]].y)
        for i in range(len(nodes) - 1)
    )
    print("route: %s" % " -> ".join(nodes))
    print("network %.1f m -> world %.2f m" % (net_len, net_len / 25.0))

    p = truth(args.model)
    if p is None:
        print("FAIL: no ground truth for %r" % args.model)
        return 2
    print("start pose x=%.3f y=%.3f yaw=%.3f" % p)

    rows = []
    t0 = time.time()
    reached = 0
    for idx, (gx, gy) in enumerate(waypoints):
        ok = False
        for _ in range(160):
            p = truth(args.model)
            if p is None:
                stop_all()
                print("FAIL: lost ground truth")
                return 2
            x, y, yaw = p
            dist = math.hypot(gx - x, gy - y)
            rows.append([round(time.time() - t0, 2), idx, round(x, 4), round(y, 4),
                         round(yaw, 4), round(gx, 4), round(gy, 4), round(dist, 4)])
            if dist < args.reach:
                reached += 1
                ok = True
                print("  wp%d %s reached x=%.3f y=%.3f d=%.3f" % (idx, nodes[idx], x, y, dist))
                break
            err = math.atan2(gy - y, gx - x) - yaw
            err = (err + math.pi) % (2.0 * math.pi) - math.pi
            wz = max(-1.2, min(1.2, 2.2 * err))
            vx = 0.06 if abs(err) > 0.4 else min(0.20, 0.6 * dist + 0.05)
            drive(vx, wz, 0.5)
        if not ok:
            stop_all()
            print("  wp%d TIMEOUT x=%.3f y=%.3f d=%.3f" % (idx, x, y, dist))
            break
    stop_all()

    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "waypoint", "x", "y", "yaw", "goal_x", "goal_y", "dist_m"])
        w.writerows(rows)
    print("reached %d/%d waypoints, %d samples -> %s" % (reached, len(waypoints), len(rows), args.csv))
    print("PASS" if reached == len(waypoints) else "FAIL")
    return 0 if reached == len(waypoints) else 1


if __name__ == "__main__":
    raise SystemExit(main())