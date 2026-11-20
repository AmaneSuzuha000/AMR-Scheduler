"""Linux Kernel SocketCAN (`vcan0` / `can0`) <-> Gazebo & Nav2 Closed-Loop Bridge.

Translates ROS 2 `/<ns>/cmd_vel` twists through AgileX SCOUT Mini 16-byte SocketCAN frames
(`0x421` motion cmd, `0x441` E-stop, `0x211` chassis status, `0x221` actuator RPMs, `0x311`
encoder odometry) over real Linux kernel `AF_CAN` (`SOCK_RAW`) interfaces or deterministic
in-memory/UDP relays for cross-platform testing. Also synthesizes standard `tf2_msgs/TFMessage`
transforms (`map -> <ns>/odom -> <ns>/base_footprint -> <ns>/laser_link`) so `nav2_costmap_2d`
and RViz2 project raw LiDAR scans without any world-truth shortcuts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import socket
import struct
import time
from typing import Any, Callable, Optional

from tools.live_bridge.scout_hardware import (
    CanFrame,
    ScoutHardwareBridge,
    ScoutMotionMode,
    ScoutProtocol,
)


def is_socketcan_interface_available(interface: str = "vcan0") -> bool:
    """Return True if Linux `AF_CAN` raw sockets can bind to `interface`."""
    if not (hasattr(socket, "AF_CAN") and hasattr(socket, "CAN_RAW")):
        return False
    sock: Optional[socket.socket] = None
    try:
        sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        sock.bind((str(interface),))
        return True
    except OSError:
        return False
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


@dataclass
class VCANBridgeStats:
    """Counters tracking raw 16-byte SocketCAN frames sent/received across bus endpoints."""

    tx_cmd_frames: int = 0
    rx_cmd_frames: int = 0
    tx_telem_frames: int = 0
    rx_telem_frames: int = 0
    estop_intercepts: int = 0

    @property
    def total_tx(self) -> int:
        """Total transmitted CAN frames across controller and chassis sides."""
        return int(self.tx_cmd_frames + self.tx_telem_frames)

    @property
    def total_rx(self) -> int:
        """Total received CAN frames across chassis and controller sides."""
        return int(self.rx_cmd_frames + self.rx_telem_frames)


class SocketCANChassisEndpoint:
    """Simulates the SCOUT Mini STM32 chassis CAN endpoint over real `AF_CAN` or loopback.

    Listens on the chassis side for `0x421` velocity commands and `0x441` E-stop frames,
    decodes them into target `(vx, wz)` commands for `/<ns>/guarded_cmd_vel`, and encodes
    observed wheel odometry `(vx, wz, dt)` back into `0x211`/`0x221`/`0x311` CAN frames.
    """

    def __init__(
        self,
        robot_id: str,
        *,
        can_channel: str = "vcan0",
        wheel_track_m: float = ScoutProtocol.WHEEL_TRACK_M,
        use_kernel_socketcan: Optional[bool] = None,
    ) -> None:
        """Initialize chassis-side CAN endpoint and paired controller `ScoutHardwareBridge`."""
        self.robot_id = str(robot_id)
        self.can_channel = str(can_channel)
        self.wheel_track_m = float(wheel_track_m)
        if use_kernel_socketcan is None:
            use_kernel_socketcan = is_socketcan_interface_available(self.can_channel)
        self.use_kernel_socketcan = bool(use_kernel_socketcan)
        self.stats = VCANBridgeStats()

        self._chassis_sock: Optional[socket.socket] = None
        self.controller_bridge: Optional[ScoutHardwareBridge] = None
        self.estop_latched: bool = False
        self.last_cmd_vx: float = 0.0
        self.last_cmd_wz: float = 0.0
        self.left_odom_mm: int = 0
        self.right_odom_mm: int = 0
        self._last_odom_wall: Optional[float] = None
        self.bound_real_vcan: bool = False

    def connect(self) -> bool:
        """Bind both controller `ScoutHardwareBridge` and chassis-side raw CAN socket."""
        if self.use_kernel_socketcan:
            self.controller_bridge = ScoutHardwareBridge(
                can_channel=self.can_channel,
                wheel_track_m=self.wheel_track_m,
                use_virtual_loopback=False,
                transport_mode="socketcan",
            )
            ok = self.controller_bridge.connect()
            if ok:
                csock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
                csock.bind((self.can_channel,))
                csock.setblocking(False)
                self._chassis_sock = csock
                self.bound_real_vcan = True
                return True

        self.controller_bridge = ScoutHardwareBridge(
            can_channel=self.can_channel,
            wheel_track_m=self.wheel_track_m,
            use_virtual_loopback=True,
            transport_mode="loopback",
        )
        self.controller_bridge.connect()
        self.bound_real_vcan = False
        return True

    def disconnect(self) -> None:
        """Close chassis SocketCAN socket and controller bridge safely."""
        if self.controller_bridge is not None:
            self.controller_bridge.disconnect()
            self.controller_bridge = None
        if self._chassis_sock is not None:
            try:
                self._chassis_sock.close()
            finally:
                self._chassis_sock = None

    def send_cmd_vel(self, linear_m_s: float, angular_rad_s: float) -> tuple[float, float]:
        """Encode `/cmd_vel` -> `0x421` SocketCAN frame and pump chassis decoder."""
        assert self.controller_bridge is not None, "Call connect() first"
        frame = self.controller_bridge.send_velocity(float(linear_m_s), float(angular_rad_s))
        self.stats.tx_cmd_frames += 1

        if self._chassis_sock is not None:
            self.poll_chassis_rx()
            return self.last_cmd_vx, self.last_cmd_wz

        vx, wz, _ = ScoutProtocol.decode_motion_cmd(frame.data)
        self.last_cmd_vx, self.last_cmd_wz = (0.0, 0.0) if self.estop_latched else (vx, wz)
        self.stats.rx_cmd_frames += 1
        return self.last_cmd_vx, self.last_cmd_wz

    def poll_chassis_rx(self, max_frames: int = 32) -> int:
        """Drain chassis-side `AF_CAN` socket for incoming `0x421`/`0x441` command frames."""
        if self._chassis_sock is None:
            return 0
        count = 0
        for _ in range(max_frames):
            try:
                raw = self._chassis_sock.recv(ScoutProtocol.CAN_FRAME_SIZE)
            except (BlockingIOError, TimeoutError, OSError):
                break
            if len(raw) < ScoutProtocol.CAN_FRAME_SIZE:
                break
            frame = ScoutProtocol.unpack_socketcan_frame(raw)
            if frame.arbitration_id == ScoutProtocol.CAN_ID_ESTOP_CMD:
                self.estop_latched = bool(len(frame.data) > 1 and frame.data[1] == 0x01)
                if self.estop_latched:
                    self.last_cmd_vx, self.last_cmd_wz = 0.0, 0.0
                    self.stats.estop_intercepts += 1
                count += 1
            elif frame.arbitration_id == ScoutProtocol.CAN_ID_MOTION_CMD:
                vx, wz, _ = ScoutProtocol.decode_motion_cmd(frame.data)
                if self.estop_latched:
                    vx, wz = 0.0, 0.0
                    self.stats.estop_intercepts += 1
                self.last_cmd_vx, self.last_cmd_wz = vx, wz
                self.stats.rx_cmd_frames += 1
                count += 1
        return count

    def publish_odometry_feedback(
        self,
        actual_vx: float,
        actual_wz: float,
        *,
        dt_s: Optional[float] = None,
        battery_voltage: float = 24.8,
        timestamp: Optional[float] = None,
    ) -> int:
        """Encode physical wheel velocity & encoder increments into `0x211`/`0x221`/`0x311` CAN frames."""
        assert self.controller_bridge is not None, "Call connect() first"
        now = time.time() if timestamp is None else float(timestamp)
        if dt_s is None:
            dt_val = 0.05 if self._last_odom_wall is None else max(0.005, min(0.5, now - self._last_odom_wall))
        else:
            dt_val = max(0.001, float(dt_s))
        self._last_odom_wall = now

        v_left = float(actual_vx) - 0.5 * float(actual_wz) * self.wheel_track_m
        v_right = float(actual_vx) + 0.5 * float(actual_wz) * self.wheel_track_m
        self.left_odom_mm += int(round(v_left * dt_val * 1000.0))
        self.right_odom_mm += int(round(v_right * dt_val * 1000.0))

        volt_byte = max(0, min(255, int(round(battery_voltage * 10.0))))
        mode_b = int(ScoutMotionMode.ESTOP if self.estop_latched else ScoutMotionMode.CAN_CONTROL)
        estop_b = 0x01 if self.estop_latched else 0x00
        status_frame = CanFrame(
            ScoutProtocol.CAN_ID_CHASSIS_STATUS,
            bytes([mode_b, volt_byte, estop_b, 0x00, 0x00, 0x00, 0x00, 0x00]),
            timestamp=now,
        )

        wheel_scale = 60.0 / (math.pi * ScoutProtocol.WHEEL_DIAMETER_M)
        rpm_l = max(-32767, min(32767, int(round(v_left * wheel_scale))))
        rpm_r = max(-32767, min(32767, int(round(v_right * wheel_scale))))
        rpm_frame = CanFrame(
            ScoutProtocol.CAN_ID_ACTUATOR_FEEDBACK,
            struct.pack(">hhhh", rpm_l, rpm_r, rpm_l, rpm_r),
            timestamp=now,
        )
        odom_frame = ScoutProtocol.encode_odometry(self.left_odom_mm, self.right_odom_mm)
        frames = (status_frame, rpm_frame, odom_frame)

        sent = 0
        if self._chassis_sock is not None:
            for fr in frames:
                self._chassis_sock.send(ScoutProtocol.pack_socketcan_frame(fr))
                sent += 1
            self.stats.tx_telem_frames += sent
            recvd = self.controller_bridge.poll_socketcan_frames(max_frames=32)
            self.stats.rx_telem_frames += recvd
            return recvd

        for fr in frames:
            self.controller_bridge.receive_can_frame(fr)
            sent += 1
        self.stats.tx_telem_frames += sent
        self.stats.rx_telem_frames += sent
        return sent


def build_robot_tf_transforms(
    namespace: str,
    x: float,
    y: float,
    yaw: float,
    *,
    origin_xy: tuple[float, float] = (0.0, 0.0),
    laser_offset_xyz: tuple[float, float, float] = (0.15, 0.0, 0.32),
) -> list[dict[str, Any]]:
    """Build standard TF chain descriptors `map -> odom -> base_footprint -> laser_link`."""
    half = 0.5 * float(yaw)
    qz = math.sin(half)
    qw = math.cos(half)
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    prefix = f"{namespace}/" if namespace else ""
    return [
        {
            "parent_frame": "map",
            "child_frame": f"{prefix}odom" if namespace else "odom",
            "translation": (ox, oy, 0.0),
            "rotation": (0.0, 0.0, 0.0, 1.0),
        },
        {
            "parent_frame": f"{prefix}odom" if namespace else "odom",
            "child_frame": f"{prefix}base_footprint" if namespace else "base_footprint",
            "translation": (float(x), float(y), 0.0),
            "rotation": (0.0, 0.0, float(qz), float(qw)),
        },
        {
            "parent_frame": f"{prefix}base_footprint" if namespace else "base_footprint",
            "child_frame": f"{prefix}laser_link" if namespace else "laser_link",
            "translation": tuple(float(v) for v in laser_offset_xyz),
            "rotation": (0.0, 0.0, 0.0, 1.0),
        },
    ]


class VCANGazeboRelayNode:
    """ROS 2 node bridging Nav2 `/nav2_cmd_vel` <-> `vcan0` SocketCAN <-> `/cmd_vel`, `/odom`, and TF."""

    def __init__(
        self,
        robot_id: str,
        namespace: str,
        *,
        origin_xy: tuple[float, float] = (0.0, 0.0),
        can_channel: str = "vcan0",
        publish_tf_frames: bool = True,
        publish_odom_base_tf: bool = True,
        use_motion_guard: bool = True,
        input_cmd_topic: str = "nav2_cmd_vel",
        output_cmd_topic: Optional[str] = None,
        max_linear: float = 0.8,
        max_angular: float = 1.2,
    ) -> None:
        """Create ROS 2 publishers/subscribers and connect the kernel SocketCAN endpoint."""
        from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Twist
        from nav2_msgs.msg import SpeedLimit
        from nav_msgs.msg import Odometry
        from rclpy.node import Node as _RosNode
        from rclpy.qos import qos_profile_sensor_data
        from std_srvs.srv import Empty
        from tf2_msgs.msg import TFMessage

        self._Twist = Twist
        self._Odometry = Odometry
        self._PoseCov = PoseWithCovarianceStamped
        self._TFMessage = TFMessage
        self._TransformStamped = TransformStamped
        self._node = _RosNode(f"vcan_bridge_{robot_id}")
        self.robot_id = str(robot_id)
        self.namespace = str(namespace)
        self.origin_xy = (float(origin_xy[0]), float(origin_xy[1]))
        self.wx, self.wy, self.wyaw = self.origin_xy[0], self.origin_xy[1], 0.0
        self.saw_odom = False
        self.saw_guard_status = False
        self.guard_id_seen = ""
        self.max_linear = float(max_linear)
        self.max_angular = float(max_angular)
        self.publish_tf_frames = bool(publish_tf_frames)
        self.publish_odom_base_tf = bool(publish_odom_base_tf)
        self.output_cmd_topic = str(output_cmd_topic or ("cmd_vel" if use_motion_guard else "guarded_cmd_vel"))
        self.input_cmd_topic = str(input_cmd_topic)

        self.endpoint = SocketCANChassisEndpoint(robot_id, can_channel=can_channel)
        self.endpoint.connect()
        self.latest_odom: Optional[tuple[float, float, float, float, float]] = None
        self.clock_msg_fn: Optional[Callable[[], Any]] = None

        self._pub_out = self._node.create_publisher(Twist, f"/{self.namespace}/{self.output_cmd_topic}", 10)
        self._pub_filt = self._node.create_publisher(Odometry, f"/{self.namespace}/odometry/filtered", 10)
        self._pub_amcl = self._node.create_publisher(PoseWithCovarianceStamped, f"/{self.namespace}/amcl_pose", 10)
        self._pub_tf = self._node.create_publisher(TFMessage, "/tf", 10)
        self._pub_tf_ns = self._node.create_publisher(TFMessage, f"/{self.namespace}/tf", 10)

        self._sub_nav_cmd = self._node.create_subscription(
            Twist, f"/{self.namespace}/{self.input_cmd_topic}", self._on_cmd_vel, 10
        )
        if self.output_cmd_topic != "cmd_vel":
            self._sub_cmd = self._node.create_subscription(Twist, f"/{self.namespace}/cmd_vel", self._on_cmd_vel, 10)
        else:
            self._sub_cmd = None
        self._sub_odom = self._node.create_subscription(
            Odometry, f"/{self.namespace}/odom", self._on_odom, qos_profile_sensor_data
        )
        self._sub_spd = self._node.create_subscription(SpeedLimit, f"/{self.namespace}/speed_limit", lambda _m: None, 10)
        self._sub_guard = None
        try:
            from amr_gazebo_guard.msg import GuardStatus

            self._sub_guard = self._node.create_subscription(
                GuardStatus, f"/{self.namespace}/motion_guard/status", self._on_guard, 10
            )
        except Exception:
            self._sub_guard = None
        self._srv_nomotion = self._node.create_service(
            Empty, f"/{self.namespace}/request_nomotion_update", lambda _req, resp: resp
        )

    def _on_guard(self, msg: Any) -> None:
        """Record motion guard lease heartbeat from `libamr_motion_guard.so`."""
        self.saw_guard_status = True
        self.guard_id_seen = str(getattr(msg, "guard_id", ""))

    @property
    def ros_node(self) -> Any:
        """Return underlying `rclpy.node.Node` handle for executor registration."""
        return self._node

    def _on_cmd_vel(self, msg: Any) -> None:
        """Relay velocity command through SocketCAN `0x421` and publish decoded twist."""
        vx_req = max(-self.max_linear, min(self.max_linear, float(msg.linear.x)))
        wz_req = max(-self.max_angular, min(self.max_angular, float(msg.angular.z)))
        vx_dec, wz_dec = self.endpoint.send_cmd_vel(vx_req, wz_req)
        out = self._Twist()
        out.linear.x = float(vx_dec)
        out.angular.z = float(wz_dec)
        self._pub_out.publish(out)

    def _on_odom(self, msg: Any) -> None:
        """Publish `0x211`/`0x221`/`0x311` CAN frames, map-frame odometry, and TF transforms."""
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        siny = 2.0 * (ori.w * ori.z + ori.x * ori.y)
        cosy = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
        yaw = math.atan2(siny, cosy)
        vx = float(msg.twist.twist.linear.x)
        wz = float(msg.twist.twist.angular.z)
        wx = self.origin_xy[0] + float(pos.x)
        wy = self.origin_xy[1] + float(pos.y)
        self.saw_odom = True
        self.wx = wx
        self.wy = wy
        self.wyaw = yaw
        self.latest_odom = (wx, wy, yaw, vx, wz)
        self.endpoint.publish_odometry_feedback(vx, wz)

        stamp = msg.header.stamp
        odom_filt = self._Odometry()
        odom_filt.header.stamp = stamp
        odom_filt.header.frame_id = "map"
        odom_filt.child_frame_id = f"{self.namespace}/base_footprint"
        odom_filt.pose.pose.position.x = wx
        odom_filt.pose.pose.position.y = wy
        odom_filt.pose.pose.orientation = ori
        odom_filt.twist = msg.twist
        self._pub_filt.publish(odom_filt)

        amcl = self._PoseCov()
        amcl.header.stamp = stamp
        amcl.header.frame_id = "map"
        amcl.pose.pose = odom_filt.pose.pose
        self._pub_amcl.publish(amcl)

        if self.publish_tf_frames:
            tf_global = self._TFMessage()
            tf_local = self._TFMessage()
            for item in build_robot_tf_transforms(
                self.namespace, float(pos.x), float(pos.y), yaw, origin_xy=self.origin_xy
            ):
                if (not self.publish_odom_base_tf) and str(item["child_frame"]).endswith("base_footprint"):
                    continue
                ts = self._TransformStamped()
                ts.header.stamp = stamp
                ts.header.frame_id = item["parent_frame"]
                ts.child_frame_id = item["child_frame"]
                ts.transform.translation.x = item["translation"][0]
                ts.transform.translation.y = item["translation"][1]
                ts.transform.translation.z = item["translation"][2]
                ts.transform.rotation.x = item["rotation"][0]
                ts.transform.rotation.y = item["rotation"][1]
                ts.transform.rotation.z = item["rotation"][2]
                ts.transform.rotation.w = item["rotation"][3]
                tf_global.transforms.append(ts)
                ts_loc = self._TransformStamped()
                ts_loc.header.stamp = stamp
                ts_loc.header.frame_id = item["parent_frame"].replace(f"{self.namespace}/", "")
                ts_loc.child_frame_id = item["child_frame"].replace(f"{self.namespace}/", "")
                ts_loc.transform = ts.transform
                tf_local.transforms.append(ts_loc)
            self._pub_tf.publish(tf_global)
            self._pub_tf_ns.publish(tf_local)

    def close(self) -> None:
        """Close SocketCAN resources and destroy ROS node cleanly."""
        self.endpoint.disconnect()
        self._node.destroy_node()
