"""检查 /cmd_vel 的 angular.z 符号与世界系 yaw 增量是否一致。

如果一致：命令 +wz 应使 yaw 增大。
若不一致，说明 plugin 的机器人坐标系与 world 之间差一个变换，
那么任何基于世界系 yaw 的转向律都会系统性跑偏。
"""
from __future__ import annotations

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


class P(Node):
    def __init__(self):
        super().__init__("sign_probe")
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


rclpy.init()
n = P()

for label, vx, wz in [("+wz 0.8, vx 0.12", 0.12, 0.8),
                      ("-wz 0.8, vx 0.12", 0.12, -0.8)]:
    n.reset()
    time.sleep(0.8)
    x0, y0, yaw0 = n.pose()
    t0 = time.time()
    while time.time() - t0 < 2.5:
        n.cmd(vx, wz)
        time.sleep(0.05)
    n.cmd(0.0, 0.0)
    time.sleep(0.4)
    x1, y1, yaw1 = n.pose()
    dyaw = yaw1 - yaw0
    print("%-20s dyaw = %+.3f rad   (%.0f deg)" % (label, dyaw, math.degrees(dyaw)))

n.cmd(0.0, 0.0)
n.destroy_node()
rclpy.shutdown()
