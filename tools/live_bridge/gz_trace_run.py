#!/usr/bin/env python3
"""Trace one route: every command issued next to the pose it produced.

Written because the controller kept stopping in ways that looked like a stall but were
not: the vehicle was still moving, just not toward anything. Aggregate pass/fail says
nothing about why, so this records the actual sequence and lets the failure be read
rather than guessed.
"""
from __future__ import annotations

import argparse
import math
import time

import rclpy
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import GetEntityState, SetEntityState
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from rclpy.node import Node


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Tracer(Node):
    def __init__(self, model):
        super().__init__("gz_trace_run")
        self.model = model
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.g = self.create_client(GetEntityState, "/get_entity_state")
        self.s = self.create_client(SetEntityState, "/set_entity_state")
        for c, n in ((self.g, "get"), (self.s, "set")):
            if not c.wait_for_service(timeout_sec=15.0):
                raise RuntimeError("%s service missing" % n)

    def pose(self):
        req = GetEntityState.Request()
        req.name = self.model
        req.reference_frame = "world"
        fut = self.g.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        r = fut.result()
        if r is None or not r.success:
            return None
        p = r.state.pose.position
        return p.x, p.y, yaw_of(r.state.pose.orientation)

    def teleport(self, x=0.0, y=0.0, yaw=0.0):
        req = SetEntityState.Request()
        st = EntityState()
        st.name = self.model
        st.pose = Pose(position=Point(x=float(x), y=float(y), z=0.05),
                       orientation=Quaternion(x=0.0, y=0.0,
                                              z=math.sin(yaw / 2), w=math.cos(yaw / 2)))
        st.reference_frame = "world"
        req.state = st
        fut = self.s.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        return fut.result() is not None

    def cmd(self, vx, wz):
        t = Twist()
        t.linear.x = float(vx)
        t.angular.z = float(wz)
        self.pub.publish(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="scout_mini")
    ap.add_argument("--goal", default="-1.0,0.8")
    ap.add_argument("--secs", type=float, default=20.0)
    args = ap.parse_args()

    gx, gy = (float(v) for v in args.goal.split(","))

    rclpy.init()
    node = Tracer(args.model)
    node.teleport()
    time.sleep(0.8)

    print("%-7s %-8s %-8s %-8s %-8s %-7s %-7s %-8s" % (
        "t", "x", "y", "yaw", "err", "vx", "wz", "dist"))

    t0 = time.time()
    try:
        while time.time() - t0 < args.secs:
            p = node.pose()
            if p is None:
                time.sleep(0.05)
                continue
            x, y, yaw = p
            dist = math.hypot(gx - x, gy - y)
            err = wrap(math.atan2(gy - y, gx - x) - yaw)

            # Deliberately simple: turn in place at a speed above the attenuation floor.
            vx = 0.12
            wz = max(-2.5, min(2.5, 1.30 * err / 0.31))
            node.cmd(vx, wz)

            print("%-7.2f %-8.3f %-8.3f %-8.2f %-8.2f %-7.2f %-7.2f %-8.3f" % (
                time.time() - t0, x, y, yaw, err, vx, wz, dist))
            time.sleep(0.1)
    finally:
        node.cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())