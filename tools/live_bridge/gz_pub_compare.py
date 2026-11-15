"""Same command, three ways of publishing it. Compare the motion.

Calibration with `ros2 topic pub` measured 0.71 rad/s for a commanded 1.20.
A Python rclpy publisher issuing the same Twist measured 0.05 rad/s. A factor of
thirteen between two ways of putting the same numbers on the same topic is not a
controller problem, so this isolates which part differs.
"""
from __future__ import annotations

import math
import subprocess
import time

import rclpy
from gazebo_msgs.srv import GetEntityState, SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class P(Node):
    def __init__(self, qos=None):
        super().__init__("pub_compare")
        if qos is None:
            self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        else:
            self.pub = self.create_publisher(Twist, "/cmd_vel", qos)
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
SECS = 3.0


def measure(label, publish_fn):
    n.reset()
    time.sleep(0.6)
    x0, y0, yaw0 = n.pose()
    publish_fn()
    x1, y1, yaw1 = n.pose()
    dyaw = yaw1 - yaw0
    dist = math.hypot(x1 - x0, y1 - y0)
    print("%-32s dist=%.3f (%.3f m/s)  dyaw=%+.3f (%.2f rad/s)"
          % (label, dist, dist / SECS, dyaw, dyaw / SECS))


def rclpy_default():
    t0 = time.time()
    while time.time() - t0 < SECS:
        n.cmd(0.0, 1.2)
        time.sleep(0.05)
    n.cmd(0.0, 0.0)
    time.sleep(0.3)


def rclpy_besteffort():
    q = QoSProfile(depth=10)
    q.reliability = ReliabilityPolicy.BEST_EFFORT
    q.history = HistoryPolicy.KEEP_LAST
    p2 = n.create_publisher(Twist, "/cmd_vel", q)
    t0 = time.time()
    while time.time() - t0 < SECS:
        t = Twist()
        t.angular.z = 1.2
        p2.publish(t)
        time.sleep(0.05)
    t = Twist()
    p2.publish(t)
    time.sleep(0.3)


def _send_zero_twist() -> None:
    """Publish a zero Twist before killing the publisher.

    libgazebo_ros_planar_move latches the last command, so a killed publisher without a zero
    in front of it leaves the robot rotating at the last commanded rate for ever - which is
    exactly what a "the robot keeps turning after the script exits" report looks like.
    """
    subprocess.run(
        "ros2 topic pub --times 3 -r 10 /cmd_vel geometry_msgs/msg/Twist "
        "'{linear: {x: 0.0}, angular: {z: 0.0}}' >/dev/null 2>&1",
        shell=True, capture_output=True, timeout=20)


def external_cli():
    p = subprocess.Popen(
        "ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "
        "'{linear: {x: 0.0}, angular: {z: 1.2}}'",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(SECS)
    p.terminate()
    p.wait()
    _send_zero_twist()
    subprocess.run("pkill -f 'ros2 topic [p]ub'", shell=True, capture_output=True)
    time.sleep(0.3)


measure("rclpy default QoS", rclpy_default)
measure("rclpy best-effort QoS", rclpy_besteffort)
measure("ros2 topic pub subprocess", external_cli)

n.destroy_node()
rclpy.shutdown()
