#!/usr/bin/env python3
"""Ground truth from Gazebo over the /get_entity_state and /set_entity_state services.

Why a node rather than shelling out:

Every other way of reading the pose on this install is either unreliable or wrong.
``gz model -m <name> -i`` answers in ~0.41 s and then stops answering entirely after a
few minutes while gzserver keeps simulating. ``/gazebo/default/pose/info`` publishes
nothing on this launch. ``/odom`` is wheel integration and, measured against a bounded
command, reported 0.879 m for at most 0.45 m of travel.

The services exposed by ``libgazebo_ros_state.so`` are the only channel that returns the
simulator own pose on demand. Spawning ``ros2 service call`` per sample costs about a
second of interpreter start-up, far too slow for a control loop, so this is a persistent
node that answers request/reply over stdin/stdout.

Protocol: one request per line on stdin, one reply per line on stdout.

    POSE                      -> "x y yaw"
    POSE <name>               -> "x y yaw"
    SET <name> <x> <y> [<yaw>] -> "ok" or "err <reason>"

Replies never contain a newline, so a reader can block on readline().
"""
from __future__ import annotations

import math
import sys

import rclpy
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import GetEntityState, SetEntityState
from geometry_msgs.msg import Point, Pose, Quaternion, Twist


def yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PoseServer:
    def __init__(self, node, get_name="/get_entity_state",
                 set_name="/set_entity_state"):
        # The node must be kept and passed to spin_until_future_complete. In Humble
        # that is a module-level function taking (node, future); it is a method on
        # neither the node nor the client, and the wrong form fails at call time with
        # an AttributeError rather than at import.
        self._node = node
        self._get = node.create_client(GetEntityState, get_name)
        self._set = node.create_client(SetEntityState, set_name)
        if not self._get.wait_for_service(timeout_sec=15.0):
            raise RuntimeError("%s never appeared" % get_name)
        if not self._set.wait_for_service(timeout_sec=15.0):
            raise RuntimeError("%s never appeared" % set_name)

    def pose(self, model: str):
        req = GetEntityState.Request()
        req.name = model
        req.reference_frame = "world"
        future = self._get.call_async(req)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=5.0)
        res = future.result()
        if res is None:
            raise RuntimeError("get_entity_state timed out")
        # Checking success matters: a missing model answers with a pose of (0,0,0) and
        # success False. Treating that as a real position would put the vehicle at the
        # origin and make every measurement silently wrong.
        if not res.success:
            raise RuntimeError("get_entity_state failed: %s" % res.status_message)
        p = res.state.pose.position
        return p.x, p.y, yaw_from_quaternion(res.state.pose.orientation)

    def set_pose(self, model: str, x: float, y: float, yaw: float = 0.0) -> None:
        req = SetEntityState.Request()
        st = EntityState()
        st.name = model
        st.pose = Pose(
            position=Point(x=float(x), y=float(y), z=0.05),
            orientation=Quaternion(x=0.0, y=0.0,
                                   z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0)),
        )
        # Zero the velocity as well as the pose. set_entity_state will happily move a model
        # while leaving its twist alone, so a vehicle teleported mid-turn keeps spinning:
        # the next measurement then starts with an unknown angular velocity carried over
        # from the previous run. That is a plausible explanation for one repeated rotation
        # command producing +49, +49 and +25 degrees.
        st.twist = Twist()
        st.reference_frame = "world"
        req.state = st
        future = self._set.call_async(req)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=5.0)
        res = future.result()
        if res is None or not res.success:
            raise RuntimeError("set_entity_state failed: %s"
                               % (getattr(res, "status_message", "timeout")))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    model = argv[0] if argv else "scout_mini"

    rclpy.init()
    node = rclpy.create_node("gz_pose_node")
    try:
        server = PoseServer(node)
    except RuntimeError as exc:
        print("err %s" % exc, flush=True)
        return 1

    print("ready", flush=True)
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        try:
            if parts[0].upper() == "POSE":
                name = parts[1] if len(parts) > 1 else model
                x, y, yaw = server.pose(name)
                print("%.6f %.6f %.6f" % (x, y, yaw), flush=True)
            elif parts[0].upper() == "SET":
                name = parts[1]
                x, y = float(parts[2]), float(parts[3])
                yaw = float(parts[4]) if len(parts) > 4 else 0.0
                server.set_pose(name, x, y, yaw)
                print("ok", flush=True)
            elif parts[0].upper() == "QUIT":
                break
            else:
                print("err unknown command %s" % parts[0], flush=True)
        except Exception as exc:            # noqa: BLE001 - reported, not raised
            print("err %s" % exc, flush=True)

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
