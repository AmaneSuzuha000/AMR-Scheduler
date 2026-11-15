"""Three repeats of one command, to test whether this simulator is repeatable at all.

Every controller gain in this directory was derived by measuring the response to a
commanded velocity. That is only meaningful if repeating the same command produces the
same motion. After a restart, a pure straight-line command (wz = 0) produced 57 degrees
of rotation, which is either a measurement bug or a reason to trust none of the gains.

This runs the identical command three times and prints all three results. If they differ
substantially, the conclusion is that this Gazebo setup cannot support repeatable motion
measurement, and no accuracy figure should be quoted from it.
"""
from __future__ import annotations

import math
import subprocess
import sys
import time

import rclpy
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import GetEntityState, SetEntityState
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from rclpy.node import Node


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class P(Node):
    def __init__(self):
        super().__init__("repeat_probe")
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
        st.reference_frame = "world"
        r.state = st
        f = self.s.call_async(r)
        rclpy.spin_until_future_complete(self, f, timeout_sec=2.0)

    def cmd(self, vx, wz):
        t = Twist()
        t.linear.x = float(vx)
        t.angular.z = float(wz)
        self.pub.publish(t)


def run_once(node, vx, wz, secs):
    node.reset()
    time.sleep(0.8)
    x0, y0, yaw0 = node.pose()
    t0 = time.time()
    while time.time() - t0 < secs:
        node.cmd(vx, wz)
        time.sleep(0.05)
    node.cmd(0.0, 0.0)
    time.sleep(0.4)
    x1, y1, yaw1 = node.pose()
    return math.hypot(x1 - x0, y1 - y0), yaw1 - yaw0


rclpy.init()
n = P()
SECS = 2.5

# Wheel speeds for a differential drive are vx -/+ wz*L/2. With L ~ 0.5 m, a command of
# vx=0.10 with wz=0.80 puts the inner wheel at -0.10 m/s: it must reverse. A reversing wheel
# at low speed slips, and slip is not deterministic. The cases below separate "both wheels
# forward" from "one wheel reversing", which is the difference that matters.
for vx, wz in [(0.10, 0.8), (0.20, 0.8), (0.30, 0.8), (0.30, 0.4)]:
    print("command vx=%.2f wz=%.2f  (%d repeats, %.1f s each)" % (vx, wz, 3, SECS))
    ds, ys = [], []
    for _ in range(3):
        d, y = run_once(n, vx, wz, SECS)
        ds.append(d)
        ys.append(y)
        print("    dist=%.3f  dyaw=%+.3f rad (%+.0f deg)" % (d, y, math.degrees(y)))
    spread_d = max(ds) - min(ds)
    spread_y = max(ys) - min(ys)
    print("    spread: dist %.3f m, yaw %.0f deg  ->  %s"
          % (spread_d, math.degrees(spread_y),
             "repeatable" if spread_d < 0.05 and math.degrees(spread_y) < 10
             else "NOT REPEATABLE"))
    print()

n.destroy_node()
rclpy.shutdown()
