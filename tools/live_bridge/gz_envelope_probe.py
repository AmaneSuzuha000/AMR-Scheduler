#!/usr/bin/env python3
"""Map the command envelope in which this vehicle repeats itself.

Repeating one rotation command produced yaw results differing by up to 35 degrees, while
the same command at a different forward speed repeated to within 0.3 degrees. That is not
noise -- it means part of the command space is deterministic and part is not, and the
controller has to live in the part that is.

Each (vx, wz) pair is run twice and the spread recorded. A pair is marked USABLE only when
both the distance and the yaw repeat. The output is a table, not a tuned parameter: the
follower then restricts itself to the cells marked USABLE rather than the whole plane.
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


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Probe(Node):
    def __init__(self):
        super().__init__("envelope_probe")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.g = self.create_client(GetEntityState, "/get_entity_state")
        self.s = self.create_client(SetEntityState, "/set_entity_state")
        self.g.wait_for_service(timeout_sec=15.0)
        self.s.wait_for_service(timeout_sec=15.0)

    def pose(self):
        r = GetEntityState.Request()
        r.name = "scout_mini"
        r.reference_frame = "world"
        f = self.g.call_async(r)
        rclpy.spin_until_future_complete(self, f, timeout_sec=2.0)
        res = f.result()
        p = res.state.pose.position
        return p.x, p.y, yaw_of(res.state.pose.orientation)

    def reset(self):
        r = SetEntityState.Request()
        st = EntityState()
        st.name = "scout_mini"
        st.pose = Pose(position=Point(x=0.0, y=0.0, z=0.05),
                       orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
        st.twist = Twist()
        st.reference_frame = "world"
        r.state = st
        f = self.s.call_async(r)
        rclpy.spin_until_future_complete(self, f, timeout_sec=2.0)

    def cmd(self, vx, wz):
        t = Twist()
        t.linear.x = float(vx)
        t.angular.z = float(wz)
        self.pub.publish(t)


def run(node, vx, wz, secs):
    node.reset()
    time.sleep(0.7)
    x0, y0, yaw0 = node.pose()
    t0 = time.time()
    while time.time() - t0 < secs:
        node.cmd(vx, wz)
        time.sleep(0.05)
    node.cmd(0.0, 0.0)
    time.sleep(0.4)
    x1, y1, yaw1 = node.pose()
    return math.hypot(x1 - x0, y1 - y0), yaw1 - yaw0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    rclpy.init()
    node = Probe()

    print("%-7s %-7s %-9s %-9s %-8s %-8s %-8s" % (
        "vx", "wz", "dist_sp", "yaw_sp", "vx_real", "wz_real", "verdict"))

    usable = []
    for vx in (0.10, 0.15, 0.20, 0.25, 0.30):
        for wz in (0.0, 0.4, 0.8, 1.2):
            ds, ys = [], []
            for _ in range(args.repeats):
                d, y = run(node, vx, wz, args.secs)
                ds.append(d)
                ys.append(y)
            sp_d = max(ds) - min(ds)
            sp_y = max(ys) - min(ys)
            vx_real = (sum(ds) / len(ds)) / args.secs
            wz_real = (sum(ys) / len(ys)) / args.secs
            ok = sp_d < 0.04 and math.degrees(sp_y) < 8.0
            if ok:
                usable.append((vx, wz))
            print("%-7.2f %-7.2f %-9.3f %-9.1f %-8.3f %-8.3f %-8s" % (
                vx, wz, sp_d, math.degrees(sp_y), vx_real, wz_real,
                "USABLE" if ok else "unstable"))

    print()
    print("usable cells: %d / 20" % len(usable))
    for c in usable:
        print("    vx=%.2f  wz=%.2f" % c)

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
