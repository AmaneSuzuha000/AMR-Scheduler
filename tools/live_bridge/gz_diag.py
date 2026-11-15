from __future__ import annotations
import math, time
import rclpy
from gazebo_msgs.srv import GetEntityState
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock

rclpy.init()
n = rclpy.create_node("diag")
g = n.create_client(GetEntityState, "/get_entity_state")
g.wait_for_service(timeout_sec=10.0)

r = GetEntityState.Request()
r.name = "scout_mini"
r.reference_frame = "world"
f = g.call_async(r)
rclpy.spin_until_future_complete(n, f, timeout_sec=5.0)
res = f.result()
print("success:", res.success)
p = res.state.pose.position
o = res.state.pose.orientation
v = res.state.twist.linear
w = res.state.twist.angular
print("pose: x=%.3f y=%.3f z=%.3f" % (p.x, p.y, p.z))
print("quat: x=%.3f y=%.3f z=%.3f w=%.3f" % (o.x, o.y, o.z, o.w))
print("lin vel: %.3f %.3f %.3f" % (v.x, v.y, v.z))
print("ang vel: %.3f %.3f %.3f" % (w.x, w.y, w.z))

# clock
samples = []
q = QoSProfile(depth=10); q.reliability = ReliabilityPolicy.BEST_EFFORT
n.create_subscription(Clock, "/clock", lambda m: samples.append(m.clock.sec + m.clock.nanosec/1e9), q)
t0 = time.time()
while time.time() - t0 < 2.0:
    rclpy.spin_once(n, timeout_sec=0.05)
print("clock advancing:", len(samples) > 5, "(%d msgs)" % len(samples))

n.destroy_node()
rclpy.shutdown()
