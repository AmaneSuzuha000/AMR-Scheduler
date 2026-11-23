"""Derive signed encoder velocity from observed wheel-position increments.

The Gazebo joint-position stream is the sole measurement source. No model
state, map pose, commanded velocity, task, guard or future data enters this
conversion. Continuous wheel angles must be unwrapped by the source.
"""
from __future__ import annotations

import math

WHEELS = ('front_left_wheel', 'front_right_wheel',
          'rear_left_wheel', 'rear_right_wheel')


class PositionEncoder:
    """Finite-interval kinematics; never turn a missing interval into zero speed."""

    def __init__(self, radius_m=.08, separation_m=.416503, max_interval_s=.2,
                 max_wheel_speed_mps=5.):
        values = (radius_m, separation_m, max_interval_s)
        if not all(type(value) in (float, int) and math.isfinite(value) and value > 0
                   for value in values):
            raise ValueError('positive finite encoder geometry and interval required')
        if (type(max_wheel_speed_mps) not in (float, int)
                or not math.isfinite(max_wheel_speed_mps) or max_wheel_speed_mps <= 0):
            raise ValueError('positive finite ambiguity threshold required')
        self.radius = radius_m
        self.separation = separation_m
        self.max_interval = max_interval_s
        self.max_wheel_speed = max_wheel_speed_mps
        self.previous = None
        self.pose = (0., 0., 0.)

    def observe(self, source_s, names, positions):
        """Return an interval measurement at its original ending source time."""
        if (type(source_s) not in (float, int) or not math.isfinite(source_s)
                or source_s < 0 or len(names) != len(positions) or len(set(names)) != len(names)):
            raise ValueError('finite source time and unique matched joints required')
        try:
            angles = tuple(positions[names.index(name)] for name in WHEELS)
        except ValueError as exc:
            raise ValueError('all four named wheel positions required') from exc
        if not all(type(value) in (float, int) and math.isfinite(value) for value in angles):
            raise ValueError('finite wheel positions required')
        if self.previous is None:
            self.previous = (source_s, angles)
            return None
        before_s, before = self.previous
        dt = source_s-before_s
        if dt == 0:
            return None
        if dt < 0:
            raise ValueError('encoder source clock moved backwards')
        if dt > self.max_interval:
            self.previous = (source_s, angles)
            return None
        delta = tuple((after-start)*self.radius for after, start in zip(angles, before))
        if any(abs(value)/dt > self.max_wheel_speed for value in delta):
            # Gazebo's 50 Hz joint-state stream interleaves independently
            # interpolated positions at near-identical source times. A jump
            # beyond the physical drive limit cannot identify which branch is
            # the continuous wheel angle, so do not publish a fabricated
            # velocity. The next valid pair can resume without inventing zero.
            self.previous = (source_s, angles)
            return None
        left, right = (delta[0]+delta[2])/2., (delta[1]+delta[3])/2.
        distance = (left+right)/2.
        turn = (right-left)/self.separation
        if not all(math.isfinite(value) for value in (*delta, distance, turn,
                                                      distance/dt, turn/dt)):
            raise ValueError('encoder increment exceeds finite arithmetic')
        x, y, yaw = self.pose
        # Exact differential-drive arc in the encoder frame. This pose is dead
        # reckoning, not a map/world measurement and is not fused by the EKF.
        chord = distance if abs(turn) < 1e-12 else distance*2.*math.sin(turn/2.)/turn
        x += chord*math.cos(yaw+turn/2.)
        y += chord*math.sin(yaw+turn/2.)
        yaw += turn
        if not all(math.isfinite(value) for value in (x, y, yaw)):
            raise ValueError('encoder pose exceeds finite arithmetic')
        self.pose = (x, y, yaw)
        self.previous = (source_s, angles)
        return dict(source_s=source_s, interval_s=dt, linear_x=distance/dt,
                    angular_z=turn/dt, pose=self.pose)


def main():
    """ROS conversion only; robot control remains in the existing Python owner."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from nav_msgs.msg import Odometry

    class EncoderNode(Node):
        def __init__(self):
            super().__init__('position_encoder')
            radius = self.declare_parameter('wheel_radius_m', .08).value
            separation = self.declare_parameter('wheel_separation_m', .416503).value
            self.encoder = PositionEncoder(radius, separation)
            self.publisher = self.create_publisher(Odometry, 'wheel/position_odometry', 10)
            self.subscription = self.create_subscription(
                JointState, 'joint_states', self.convert, qos_profile_sensor_data)

        def convert(self, message):
            stamp = message.header.stamp
            measurement = self.encoder.observe(stamp.sec+stamp.nanosec*1e-9,
                list(message.name), list(message.position))
            if measurement is None:
                return
            result = Odometry()
            result.header.stamp = message.header.stamp
            result.header.frame_id = 'odom'
            result.child_frame_id = 'base_footprint'
            x, y, yaw = measurement['pose']
            result.pose.pose.position.x, result.pose.pose.position.y = x, y
            result.pose.pose.orientation.z = math.sin(yaw/2.)
            result.pose.pose.orientation.w = math.cos(yaw/2.)
            result.twist.twist.linear.x = measurement['linear_x']
            result.twist.twist.angular.z = measurement['angular_z']
            # Preserve the stock synthetic longitudinal uncertainty; neither
            # this value nor these continuous angles are hardware calibration.
            for index in (0, 7, 14, 21, 28, 35):
                result.pose.covariance[index] = 1e12
                result.twist.covariance[index] = 1e12
            result.twist.covariance[0] = 1e-5
            result.twist.covariance[35] = 1e-3
            self.publisher.publish(result)

    rclpy.init()
    node = EncoderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
