"""Measure the simulator real-time factor.

Every controller gain here was derived by comparing a commanded rate against motion
observed over wall-clock seconds. That comparison is only valid if simulated time
advances at the same rate as wall time. If Gazebo is running at 40% real time, then a
vehicle that appears to move at 0.12 m/s is actually moving at 0.30 m/s in the units the
physics engine uses, and every gain tuned by hand is wrong by that factor.
"""
from __future__ import annotations

import time

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock

rclpy.init()
node = rclpy.create_node("rtf_probe")
samples = []


def cb(msg):
    samples.append(msg.clock.sec + msg.clock.nanosec / 1e9)


# /clock is published with BEST_EFFORT reliability; the default (RELIABLE) subscription
# is rejected by QoS compatibility and silently receives nothing.
qos = QoSProfile(depth=10)
qos.reliability = ReliabilityPolicy.BEST_EFFORT
node.create_subscription(Clock, "/clock", cb, qos)

print("collecting 6 s of /clock ...")
t0 = time.time()
while time.time() - t0 < 6.0:
    rclpy.spin_once(node, timeout_sec=0.05)

if len(samples) < 10:
    print("only %d clock messages -- /clock may not be published" % len(samples))
else:
    sim_span = samples[-1] - samples[0]
    wall_span = 6.0
    print("messages   : %d" % len(samples))
    print("sim span   : %.3f s" % sim_span)
    print("wall span  : %.3f s" % wall_span)
    print("RTF        : %.3f" % (sim_span / wall_span))
    print()
    print("If RTF is well below 1.0, every wall-clock-derived speed is inflated by 1/RTF.")

node.destroy_node()
rclpy.shutdown()
