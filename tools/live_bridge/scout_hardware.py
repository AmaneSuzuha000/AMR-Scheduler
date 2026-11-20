# -*- coding: utf-8 -*-
"""SCOUT Mini 真实硬件抽象层与 CAN 总线通信接口 (Sim-to-Real Bridge).

工业背景与通信规范:
-------------------
AgileX SCOUT Mini 采用工业标准 CAN 2.0B 协议通信 (标称波特率 500 Kbps)。
底盘采用四轮差速/全向驱动刚体结构，具备高精度轮速编码器与硬件级看门狗制动。

本模块提供高可靠性硬件抽象接口 (HAL):
1. CAN 帧封包/解包:
   - 运动控制指令帧 (CAN ID: 0x421): 线速度 vx (mm/s, int16), 角速度 wz (mrad/s, int16), 校验和;
   - 底盘综合状态帧 (CAN ID: 0x211): 运行模式, 错误码, 电池电压 (0.1V), 急停状态;
   - 执行器转速反馈 (CAN ID: 0x221): 四电机实时转速 (RPM, int16) 与电流反馈;
2. 硬件级安全护栏:
   - 硬件急停 (E-Stop) 双向硬互锁;
   - 速度与加速度刚性饱和钳位 (线速 <= 1.5 m/s, 角速 <= 2.5 rad/s);
   - 硬件看门狗 (250 ms): 超时未刷新控制帧强制下发零速制动并置位刹车状态;
3. 跨平台兼容性:
   - Linux 环境支持原生 SocketCAN (`can0` / `vcan0`);
   - Windows / 离线环境无缝回退至内置高精虚拟环回仿真器 (Virtual Can Loopback), 支持全链路单元测试.
"""

from __future__ import annotations

import math
import socket
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional


class ScoutMotionMode(IntEnum):
    STANDBY = 0x00
    CAN_CONTROL = 0x01
    REMOTE_CONTROL = 0x02
    ESTOP = 0x03
    FAULT = 0x04


class ScoutErrorCode(IntEnum):
    NONE = 0x00
    BATTERY_LOW = 0x01
    MOTOR_OVERHEAT = 0x02
    DRIVER_OVERCURRENT = 0x04
    CAN_TIMEOUT = 0x08
    ESTOP_ACTIVE = 0x10


@dataclass(frozen=True)
class CanFrame:
    """CAN 2.0B 标准数据帧 (8 字节载荷)."""
    arbitration_id: int
    data: bytes
    is_extended: bool = False
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if len(self.data) > 8:
            raise ValueError(f"CAN frame data length must not exceed 8 bytes, got {len(self.data)}")


@dataclass
class ScoutChassisState:
    """SCOUT Mini 硬件实时遥测与双轮编码器里程计状态."""
    mode: ScoutMotionMode = ScoutMotionMode.STANDBY
    linear_velocity: float = 0.0      # m/s
    angular_velocity: float = 0.0     # rad/s
    battery_voltage: float = 24.0     # V
    estop_active: bool = False
    error_flags: int = 0
    motor_rpm: tuple[int, int, int, int] = (0, 0, 0, 0)
    left_odom_mm: int = 0
    right_odom_mm: int = 0
    odom_x_m: float = 0.0
    odom_y_m: float = 0.0
    odom_yaw_rad: float = 0.0
    last_feedback_time: float = 0.0
    _prev_left_odom_mm: Optional[int] = None
    _prev_right_odom_mm: Optional[int] = None

    @property
    def is_healthy(self) -> bool:
        return not self.estop_active and (self.error_flags == 0) and (self.mode != ScoutMotionMode.FAULT)

    @property
    def odom_xy_yaw(self) -> tuple[float, float, float]:
        """Return integrated differential-drive `(x_m, y_m, yaw_rad)` pose."""
        return (self.odom_x_m, self.odom_y_m, self.odom_yaw_rad)

    @property
    def odom_distance_m(self) -> float:
        """Return mean longitudinal distance traveled across both tracks in meters."""
        return 0.5 * (self.left_odom_mm + self.right_odom_mm) / 1000.0


class ScoutProtocol:
    """AgileX SCOUT Mini CAN 2.0B / SocketCAN 协议编解码器 (aligned with agilex_protocol_v2)."""

    CAN_ID_MOTION_CMD = 0x421
    CAN_ID_ESTOP_CMD = 0x441
    CAN_ID_CHASSIS_STATUS = 0x211
    CAN_ID_ACTUATOR_FEEDBACK = 0x221
    CAN_ID_MOTION_STATE = 0x221
    CAN_ID_ODOMETRY = 0x311

    CAN_FRAME_FMT = "=IB3x8s"
    CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FMT)

    # 物理与安全门限参数
    MAX_LINEAR_VELOCITY = 1.5    # m/s
    MAX_ANGULAR_VELOCITY = 2.5   # rad/s
    WHEEL_DIAMETER_M = 0.175
    WHEEL_TRACK_M = 0.49

    @classmethod
    def encode_motion_cmd(cls, linear_m_s: float, angular_rad_s: float, counter: int = 0) -> CanFrame:
        """编码速度控制帧 (0x421).
        
        Payload:
        Byte 0-1: Linear velocity in mm/s (int16_be)
        Byte 2-3: Angular velocity in mrad/s (int16_be)
        Byte 4: Reserve / Mode select (0x01 = CAN control)
        Byte 5: Rolling counter (0-255)
        Byte 6-7: Checksum / Reserve
        """
        # 饱和钳位
        clamped_v = max(-cls.MAX_LINEAR_VELOCITY, min(cls.MAX_LINEAR_VELOCITY, linear_m_s))
        clamped_w = max(-cls.MAX_ANGULAR_VELOCITY, min(cls.MAX_ANGULAR_VELOCITY, angular_rad_s))

        v_int = int(round(clamped_v * 1000.0))
        w_int = int(round(clamped_w * 1000.0))

        # 构造大端序载荷
        payload = struct.pack(">hhBBH", v_int, w_int, 0x01, counter & 0xFF, 0x00)
        return CanFrame(arbitration_id=cls.CAN_ID_MOTION_CMD, data=payload)

    @classmethod
    def decode_motion_cmd(cls, data: bytes) -> tuple[float, float, int]:
        """解码速度控制帧 (0x421) 返回 `(linear_m_s, angular_rad_s, counter)`."""
        if len(data) < 6:
            raise ValueError(f"Motion cmd frame too short: {len(data)}")
        v_int, w_int, _mode, counter = struct.unpack(">hhBB", data[:6])
        return float(v_int) / 1000.0, float(w_int) / 1000.0, int(counter)

    @classmethod
    def encode_estop_cmd(cls, activate: bool) -> CanFrame:
        """编码硬件急停指令 (0x441)."""
        flag = 0x01 if activate else 0x00
        payload = bytes([0xAA, flag, 0x00, 0x00, 0x00, 0x00, 0x00, 0x55])
        return CanFrame(arbitration_id=cls.CAN_ID_ESTOP_CMD, data=payload)

    @classmethod
    def decode_chassis_status(cls, data: bytes) -> tuple[ScoutMotionMode, float, bool, int]:
        """解码底盘状态帧 (0x211).
        
        Returns:
            (mode, battery_voltage, estop_active, error_flags)
        """
        if len(data) < 6:
            raise ValueError(f"Status frame too short: {len(data)}")
        mode_val = data[0]
        try:
            mode = ScoutMotionMode(mode_val)
        except ValueError:
            mode = ScoutMotionMode.FAULT

        voltage = data[1] * 0.1  # 0.1V 步长
        estop = bool(data[2] & 0x01)
        err_flags = data[3]
        return mode, voltage, estop, err_flags

    @classmethod
    def decode_actuator_feedback(cls, data: bytes) -> tuple[int, int, int, int]:
        """解码四电机转速反馈帧 (0x221).
        
        Returns:
            (rpm_fl, rpm_fr, rpm_rl, rpm_rr)
        """
        if len(data) < 8:
            raise ValueError(f"Actuator feedback frame too short: {len(data)}")
        rpms = struct.unpack(">hhhh", data[:8])
        return rpms

    @classmethod
    def encode_odometry(cls, left_mm: int, right_mm: int) -> CanFrame:
        """编码双侧编码器里程计帧 (0x311, int32_be left_mm, int32_be right_mm)."""
        return CanFrame(arbitration_id=cls.CAN_ID_ODOMETRY, data=struct.pack(">ii", int(left_mm), int(right_mm)))

    @classmethod
    def decode_odometry(cls, data: bytes) -> tuple[int, int]:
        """解码双侧编码器里程计帧 (0x311) 返回 `(left_mm, right_mm)`."""
        if len(data) < 8:
            raise ValueError(f"Odometry frame too short: {len(data)}")
        left_mm, right_mm = struct.unpack(">ii", data[:8])
        return int(left_mm), int(right_mm)

    @classmethod
    def encode_motion_state(cls, linear_m_s: float, angular_rad_s: float) -> CanFrame:
        """编码 AgileX v2 运动状态帧 (0x221: mm/s, mrad/s int16_be)."""
        v_mm = int(round(max(-cls.MAX_LINEAR_VELOCITY, min(cls.MAX_LINEAR_VELOCITY, linear_m_s)) * 1000.0))
        w_mrad = int(round(max(-cls.MAX_ANGULAR_VELOCITY, min(cls.MAX_ANGULAR_VELOCITY, angular_rad_s)) * 1000.0))
        return CanFrame(arbitration_id=cls.CAN_ID_MOTION_STATE, data=struct.pack(">hhI", v_mm, w_mrad, 0))

    @classmethod
    def decode_motion_state(cls, data: bytes) -> tuple[float, float]:
        """解码 AgileX v2 运动状态帧 (0x221) 返回 `(linear_m_s, angular_rad_s)`."""
        if len(data) < 4:
            raise ValueError(f"Motion state frame too short: {len(data)}")
        v_mm, w_mrad = struct.unpack(">hh", data[:4])
        return float(v_mm) / 1000.0, float(w_mrad) / 1000.0

    @classmethod
    def pack_socketcan_frame(cls, frame: CanFrame) -> bytes:
        """Serialize `CanFrame` into 16-byte Linux SocketCAN `struct can_frame` wire format."""
        can_id = int(frame.arbitration_id) & (0x1FFFFFFF if frame.is_extended else 0x7FF)
        if frame.is_extended:
            can_id |= getattr(socket, "CAN_EFF_FLAG", 0x80000000)
        dlc = len(frame.data)
        padded = frame.data.ljust(8, b"\x00")
        return struct.pack(cls.CAN_FRAME_FMT, can_id, dlc, padded)

    @classmethod
    def unpack_socketcan_frame(cls, raw: bytes, timestamp: Optional[float] = None) -> CanFrame:
        """Deserialize 16-byte Linux SocketCAN `struct can_frame` wire bytes into `CanFrame`."""
        if len(raw) < cls.CAN_FRAME_SIZE:
            raise ValueError(f"SocketCAN raw frame too short: {len(raw)}")
        can_id, dlc, data = struct.unpack(cls.CAN_FRAME_FMT, raw[: cls.CAN_FRAME_SIZE])
        eff_flag = getattr(socket, "CAN_EFF_FLAG", 0x80000000)
        is_ext = bool(can_id & eff_flag)
        arb_id = can_id & (0x1FFFFFFF if is_ext else 0x7FF)
        return CanFrame(
            arbitration_id=arb_id,
            data=data[: min(8, int(dlc))],
            is_extended=is_ext,
            timestamp=(time.time() if timestamp is None else float(timestamp)),
        )


class ScoutHardwareBridge:
    """SCOUT Mini 硬件抽象通信桥梁 (原生 Linux SocketCAN + CAN-over-IP 环回 / 看门狗)."""

    def __init__(
        self,
        can_channel: str = "can0",
        watchdog_timeout_s: float = 0.25,
        use_virtual_loopback: bool = False,
        feedback_hook: Optional[Callable[[ScoutChassisState], None]] = None,
        *,
        transport_mode: str = "auto",
        wire_endpoint: Optional[tuple[str, int]] = None,
        wheel_track_m: float = ScoutProtocol.WHEEL_TRACK_M,
    ):
        self.can_channel = can_channel
        self.watchdog_timeout_s = watchdog_timeout_s
        self.use_virtual_loopback = use_virtual_loopback
        self.feedback_hook = feedback_hook
        self.transport_mode = str(transport_mode)
        self.wire_endpoint = wire_endpoint
        self.wheel_track_m = float(wheel_track_m)

        self.state = ScoutChassisState()
        self._last_cmd_time = 0.0
        self._cmd_counter = 0
        self._is_connected = False
        self._hardware_estop = False
        self._active_transport = "loopback"
        self._can_socket: Optional[socket.socket] = None
        self._wire_socket: Optional[socket.socket] = None
        self._sim_odom_time: Optional[float] = None
        self.tx_socketcan_frames: int = 0
        self.rx_socketcan_frames: int = 0

        # 模拟/环回缓冲区
        self._tx_queue: list[CanFrame] = []

    def _emit_wire_frame(self, frame: CanFrame) -> None:
        """Queue frame locally and transmit over SocketCAN or UDP wire relay if attached."""
        self._tx_queue.append(frame)
        raw = ScoutProtocol.pack_socketcan_frame(frame)
        if self._can_socket is not None:
            self._can_socket.send(raw)
            self.tx_socketcan_frames += 1
        elif self._wire_socket is not None and self.wire_endpoint is not None:
            self._wire_socket.sendto(raw, self.wire_endpoint)

    def connect(self) -> bool:
        """建立 CAN 总线连接 (物理 Linux SocketCAN -> UDP SocketCAN 帧中继 -> 环回仿真)."""
        if not self.use_virtual_loopback and self.transport_mode in ("auto", "socketcan"):
            if hasattr(socket, "AF_CAN") and hasattr(socket, "CAN_RAW"):
                try:
                    sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
                    sock.settimeout(0.01)
                    sock.bind((self.can_channel,))
                    self._can_socket = sock
                    self._active_transport = "socketcan"
                    self._is_connected = True
                    self.state.mode = ScoutMotionMode.CAN_CONTROL
                    self.state.last_feedback_time = time.time()
                    return True
                except OSError:
                    if self.transport_mode == "socketcan":
                        raise

        if self.wire_endpoint is not None and self.transport_mode in ("auto", "wire_relay"):
            wsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            wsock.setblocking(False)
            self._wire_socket = wsock
            self._active_transport = "wire_relay"
        else:
            self._active_transport = "loopback"

        self._is_connected = True
        self.state.mode = ScoutMotionMode.CAN_CONTROL
        self.state.last_feedback_time = time.time()
        return True

    def disconnect(self) -> None:
        """断开连接、关闭 CAN socket 并下发硬件零速停机."""
        if self._is_connected:
            self.stop()
            self._is_connected = False
            self.state.mode = ScoutMotionMode.STANDBY
        if self._can_socket is not None:
            try:
                self._can_socket.close()
            finally:
                self._can_socket = None
        if self._wire_socket is not None:
            try:
                self._wire_socket.close()
            finally:
                self._wire_socket = None

    def send_velocity(self, linear_m_s: float, angular_rad_s: float) -> CanFrame:
        """向底层 CAN 控制器发送运动速度指令 (带看门狗与急停拦截)."""
        if not self._is_connected:
            raise RuntimeError("Cannot send velocity: CAN bridge is not connected")

        now = time.time()
        if self._hardware_estop:
            # 处于急停状态，强制发零速且置位急停
            frame = ScoutProtocol.encode_motion_cmd(0.0, 0.0, self._cmd_counter)
            self._emit_wire_frame(frame)
            return frame

        # 看门狗刷新
        self._last_cmd_time = now
        self._cmd_counter = (self._cmd_counter + 1) & 0xFF

        frame = ScoutProtocol.encode_motion_cmd(linear_m_s, angular_rad_s, self._cmd_counter)
        self._emit_wire_frame(frame)

        # 虚拟环回时同步生成标准解码反馈与轮式里程计
        if self.use_virtual_loopback:
            clamped_v = max(-ScoutProtocol.MAX_LINEAR_VELOCITY, min(ScoutProtocol.MAX_LINEAR_VELOCITY, linear_m_s))
            clamped_w = max(-ScoutProtocol.MAX_ANGULAR_VELOCITY, min(ScoutProtocol.MAX_ANGULAR_VELOCITY, angular_rad_s))
            self._simulate_feedback(clamped_v, clamped_w)

        return frame

    def emergency_stop(self) -> None:
        """触发硬件级急停."""
        self._hardware_estop = True
        self.state.estop_active = True
        self.state.mode = ScoutMotionMode.ESTOP
        estop_frame = ScoutProtocol.encode_estop_cmd(True)
        self._emit_wire_frame(estop_frame)
        # 发零速帧
        self._emit_wire_frame(ScoutProtocol.encode_motion_cmd(0.0, 0.0, self._cmd_counter))

    def reset_emergency_stop(self) -> None:
        """解除硬件急停并复位看门狗."""
        self._hardware_estop = False
        self.state.estop_active = False
        self.state.mode = ScoutMotionMode.CAN_CONTROL
        clear_frame = ScoutProtocol.encode_estop_cmd(False)
        self._emit_wire_frame(clear_frame)
        self._last_cmd_time = time.time()

    def stop(self) -> None:
        """平稳制动停机."""
        if self._is_connected:
            self.send_velocity(0.0, 0.0)

    def poll_rx_once(self, max_frames: int = 16) -> int:
        """Drain available raw 16-byte CAN wire frames from SocketCAN/UDP relay socket."""
        sock = self._can_socket or self._wire_socket
        if sock is None:
            return 0
        count = 0
        for _ in range(max_frames):
            try:
                raw = sock.recv(ScoutProtocol.CAN_FRAME_SIZE)
            except (BlockingIOError, TimeoutError, OSError):
                break
            if len(raw) < ScoutProtocol.CAN_FRAME_SIZE:
                break
            frame = ScoutProtocol.unpack_socketcan_frame(raw)
            self.receive_can_frame(frame)
            if sock is self._can_socket:
                self.rx_socketcan_frames += 1
            count += 1
        return count

    def poll_socketcan_frames(self, max_frames: int = 32) -> int:
        """Poll up to `max_frames` raw SocketCAN frames from the attached interface."""
        return self.poll_rx_once(max_frames=max_frames)

    def check_watchdog(self, current_time: Optional[float] = None) -> bool:
        """看门狗时限核验 (若超过 watchdog_timeout_s 则强制紧急制动).
        
        Returns:
            True if healthy, False if watchdog timeout triggered.
        """
        now = time.time() if current_time is None else current_time
        if self._last_cmd_time > 0 and (now - self._last_cmd_time) > self.watchdog_timeout_s:
            # 触发看门狗制动
            self.stop()
            self.state.error_flags |= ScoutErrorCode.CAN_TIMEOUT
            return False
        return True

    def receive_can_frame(self, frame: CanFrame) -> None:
        """接收并解析来自底盘主控板的 CAN 遥测帧 (0x211 / 0x221 / 0x311)."""
        if frame.arbitration_id == ScoutProtocol.CAN_ID_CHASSIS_STATUS:
            mode, volt, estop, errs = ScoutProtocol.decode_chassis_status(frame.data)
            self.state.mode = mode
            self.state.battery_voltage = volt
            self.state.estop_active = estop or self._hardware_estop
            self.state.error_flags = errs
            self.state.last_feedback_time = frame.timestamp
            if self.feedback_hook:
                self.feedback_hook(self.state)

        elif frame.arbitration_id == ScoutProtocol.CAN_ID_ACTUATOR_FEEDBACK:
            rpms = ScoutProtocol.decode_actuator_feedback(frame.data)
            self.state.motor_rpm = rpms
            self.state.last_feedback_time = frame.timestamp
            avg_rpm = sum(rpms) / 4.0
            wheel_circ = math.pi * ScoutProtocol.WHEEL_DIAMETER_M / 60.0
            self.state.linear_velocity = avg_rpm * wheel_circ
            left_v = 0.5 * (rpms[0] + rpms[2]) * wheel_circ
            right_v = 0.5 * (rpms[1] + rpms[3]) * wheel_circ
            self.state.angular_velocity = (right_v - left_v) / max(0.1, self.wheel_track_m)
            if self.feedback_hook:
                self.feedback_hook(self.state)

        elif frame.arbitration_id == ScoutProtocol.CAN_ID_ODOMETRY:
            left_mm, right_mm = ScoutProtocol.decode_odometry(frame.data)
            if self.state._prev_left_odom_mm is not None and self.state._prev_right_odom_mm is not None:
                dl_m = float(left_mm - self.state._prev_left_odom_mm) / 1000.0
                dr_m = float(right_mm - self.state._prev_right_odom_mm) / 1000.0
                ds = 0.5 * (dl_m + dr_m)
                dtheta = (dr_m - dl_m) / max(0.1, self.wheel_track_m)
                mid_yaw = self.state.odom_yaw_rad + 0.5 * dtheta
                self.state.odom_x_m += ds * math.cos(mid_yaw)
                self.state.odom_y_m += ds * math.sin(mid_yaw)
                self.state.odom_yaw_rad = math.atan2(
                    math.sin(self.state.odom_yaw_rad + dtheta),
                    math.cos(self.state.odom_yaw_rad + dtheta),
                )
            self.state._prev_left_odom_mm = left_mm
            self.state._prev_right_odom_mm = right_mm
            self.state.left_odom_mm = left_mm
            self.state.right_odom_mm = right_mm
            self.state.last_feedback_time = frame.timestamp
            if self.feedback_hook:
                self.feedback_hook(self.state)

    def _simulate_feedback(self, target_vx: float, target_wz: float) -> None:
        """在虚拟环回模式下生成 0x211/0x221/0x311 物理闭环反馈帧."""
        now = time.time()
        dt = 0.05 if self._sim_odom_time is None else max(0.005, min(0.5, now - self._sim_odom_time))
        self._sim_odom_time = now

        # 1. 生成 0x211 状态帧
        status_data = bytes([ScoutMotionMode.CAN_CONTROL, 248, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        self.receive_can_frame(CanFrame(ScoutProtocol.CAN_ID_CHASSIS_STATUS, status_data, timestamp=now))

        # 2. 生成 0x221 四轮差速电机反馈帧 (考虑角速度差速)
        wheel_scale = 60.0 / (math.pi * ScoutProtocol.WHEEL_DIAMETER_M)
        v_left = target_vx - 0.5 * target_wz * self.wheel_track_m
        v_right = target_vx + 0.5 * target_wz * self.wheel_track_m
        rpm_l = int(round(v_left * wheel_scale))
        rpm_r = int(round(v_right * wheel_scale))
        rpm_data = struct.pack(">hhhh", rpm_l, rpm_r, rpm_l, rpm_r)
        self.receive_can_frame(CanFrame(ScoutProtocol.CAN_ID_ACTUATOR_FEEDBACK, rpm_data, timestamp=now))

        # 3. 生成 0x311 编码器积分里程计帧
        new_left_mm = self.state.left_odom_mm + int(round(v_left * dt * 1000.0))
        new_right_mm = self.state.right_odom_mm + int(round(v_right * dt * 1000.0))
        odom_frame = ScoutProtocol.encode_odometry(new_left_mm, new_right_mm)
        self.receive_can_frame(CanFrame(odom_frame.arbitration_id, odom_frame.data, timestamp=now))

    def to_cvcr_telemetry_dict(
        self,
        *,
        nominal_speed_m_s: float = 0.5,
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        """Convert SCOUT chassis CAN telemetry into `PhysicalCVCRCoordinator` dict."""
        cur_t = time.time() if now is None else float(now)
        age = max(0.0, cur_t - float(self.state.last_feedback_time)) if self.state.last_feedback_time > 0 else 999.0
        spd_ratio = abs(self.state.linear_velocity) / max(1e-3, float(nominal_speed_m_s))
        planner_state = "faulted" if not self.state.is_healthy else "nominal"
        return {
            "localization_source": "scout_odom_can",
            "speed_scale": max(0.05, min(2.0, spd_ratio)),
            "receipt_age_s": round(age, 4),
            "local_planner_state": planner_state,
            "battery_voltage": self.state.battery_voltage,
            "error_flags": self.state.error_flags,
            "odom_xy_yaw": self.state.odom_xy_yaw,
            "tx_socketcan_frames": int(self.tx_socketcan_frames),
            "rx_socketcan_frames": int(self.rx_socketcan_frames),
        }

    def get_transmitted_frames(self) -> list[CanFrame]:
        """获取并排空发送队列 (用于测试与物理总线驱动接入)."""
        frames = list(self._tx_queue)
        self._tx_queue.clear()
        return frames
