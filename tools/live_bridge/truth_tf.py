"""Publish planar Gazebo truth TF with the observation's own timestamp.

This is an explicit integration fixture, not sensor-based localization. Earlier
odom/TF failures are retained in experiment logs; the differential-drive plugin
may use world truth or encoder integration per its configured odometry_source.
Neither source is inferred from its topic name. Cached poses are never restamped.
"""
from __future__ import annotations

import math
import time

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from tf2_ros import TransformBroadcaster

from gazebo_msgs.srv import GetEntityState

# Gazebo 的 /clock 是 best-effort，必须显式匹配，否则一条都收不到。
CLOCK_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
)


class TruthTf(Node):
    def __init__(self) -> None:
        super().__init__("truth_tf")
        self.declare_parameter("model", "scout_mini")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("rate", 20.0)
        self.model = self.get_parameter("model").value
        self.world = self.get_parameter("world_frame").value
        self.odom = self.get_parameter("odom_frame").value
        self.base = self.get_parameter("base_frame").value
        self.rate = float(self.get_parameter("rate").value)

        self.br = TransformBroadcaster(self)
        self.cli = self.create_client(GetEntityState, "/get_entity_state")
        self._sim = None
        self._pending = None
        self._generation = 0
        self._last_stamp = -math.inf
        self.sent = 0
        self.errors = 0
        self.create_subscription(Clock, "/clock", self._on_clock, CLOCK_QOS)

    def _on_clock(self, msg) -> None:
        stamp = msg.clock.sec + msg.clock.nanosec * 1e-9
        if self._sim is not None and stamp < self._sim:
            self._generation += 1
            self._last_stamp = -math.inf
        self._sim = stamp

    def broadcast(self) -> None:
        """Request fresh truth; only the response callback broadcasts its TF."""
        if (self._sim is None or not self.cli.service_is_ready()
                or self._pending is not None and not self._pending.done()):
            return
        req = GetEntityState.Request()
        req.name, req.reference_frame = self.model, self.world
        requested_at, generation = time.monotonic(), self._generation
        self._pending = self.cli.call_async(req)
        self._pending.add_done_callback(lambda f: self._on_response(f, requested_at, generation))

    def _on_response(self, future, requested_at, generation) -> None:
        try:
            response = future.result()
        except Exception:
            self.errors += 1
            return
        if response is None or not response.success:
            self.errors += 1
            return
        stamp = response.header.stamp.sec + response.header.stamp.nanosec * 1e-9
        if (generation != self._generation or self._sim is None
                or time.monotonic() - requested_at > .5
                or self._sim - stamp > .25 or stamp - self._sim > .1
                or stamp <= self._last_stamp):
            self.errors += 1
            return
        p, q = response.state.pose.position, response.state.pose.orientation
        norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if not all(math.isfinite(v) for v in (p.x, p.y, norm)) or norm < 1e-9:
            self.errors += 1
            return
        qx, qy, qz, qw = (q.x/norm, q.y/norm, q.z/norm, q.w/norm)
        yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        t = TransformStamped()
        t.header.stamp = response.header.stamp
        t.header.frame_id = self.odom
        t.child_frame_id = self.base
        t.transform.translation.x = p.x
        t.transform.translation.y = p.y
        t.transform.translation.z = 0.0
        t.transform.rotation.z = math.sin(yaw/2)
        t.transform.rotation.w = math.cos(yaw/2)
        self.br.sendTransform(t)
        self._last_stamp = stamp
        self.sent += 1


def main(argv=None) -> int:
    rclpy.init()
    node = TruthTf()
    node.create_timer(1.0 / node.rate, node.broadcast)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
