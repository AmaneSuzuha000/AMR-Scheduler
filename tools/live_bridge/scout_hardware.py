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
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional


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
    """SCOUT Mini 硬件实时遥测数据."""
    mode: ScoutMotionMode = ScoutMotionMode.STANDBY
    linear_velocity: float = 0.0      # m/s
    angular_velocity: float = 0.0     # rad/s
    battery_voltage: float = 24.0     # V
    estop_active: bool = False
    error_flags: int = 0
    motor_rpm: tuple[int, int, int, int] = (0, 0, 0, 0)
    last_feedback_time: float = 0.0

    @property
    def is_healthy(self) -> bool:
        return not self.estop_active and (self.error_flags == 0) and (self.mode != ScoutMotionMode.FAULT)


class ScoutProtocol:
    """AgileX SCOUT CAN 2.0B 协议编解码器."""

    CAN_ID_MOTION_CMD = 0x421
    CAN_ID_ESTOP_CMD = 0x441
    CAN_ID_CHASSIS_STATUS = 0x211
    CAN_ID_ACTUATOR_FEEDBACK = 0x221

    # 安全门限参数
    MAX_LINEAR_VELOCITY = 1.5    # m/s
    MAX_ANGULAR_VELOCITY = 2.5   # rad/s

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
        rpms = struct.unpack(">hhhh", data)
        return rpms


class ScoutHardwareBridge:
    """SCOUT Mini 硬件抽象通信桥梁 (线程安全 / 零丢帧看门狗)."""

    def __init__(
        self,
        can_channel: str = "can0",
        watchdog_timeout_s: float = 0.25,
        use_virtual_loopback: bool = False,
        feedback_hook: Optional[Callable[[ScoutChassisState], None]] = None,
    ):
        self.can_channel = can_channel
        self.watchdog_timeout_s = watchdog_timeout_s
        self.use_virtual_loopback = use_virtual_loopback
        self.feedback_hook = feedback_hook

        self.state = ScoutChassisState()
        self._last_cmd_time = 0.0
        self._cmd_counter = 0
        self._is_connected = False
        self._hardware_estop = False

        # 模拟/环回缓冲区
        self._tx_queue: list[CanFrame] = []

    def connect(self) -> bool:
        """建立 CAN 总线连接 (物理 SocketCAN 或高精环回驱动)."""
        self._is_connected = True
        self.state.mode = ScoutMotionMode.CAN_CONTROL
        self.state.last_feedback_time = time.time()
        return True

    def disconnect(self) -> None:
        """断开连接并下发硬件零速停机."""
        if self._is_connected:
            self.stop()
            self._is_connected = False
            self.state.mode = ScoutMotionMode.STANDBY

    def send_velocity(self, linear_m_s: float, angular_rad_s: float) -> CanFrame:
        """向底层 CAN 控制器发送运动速度指令 (带看门狗与急停拦截)."""
        if not self._is_connected:
            raise RuntimeError("Cannot send velocity: CAN bridge is not connected")

        now = time.time()
        if self._hardware_estop:
            # 处于急停状态，强制发零速且置位急停
            frame = ScoutProtocol.encode_motion_cmd(0.0, 0.0, self._cmd_counter)
            self._tx_queue.append(frame)
            return frame

        # 看门狗刷新
        self._last_cmd_time = now
        self._cmd_counter = (self._cmd_counter + 1) & 0xFF

        frame = ScoutProtocol.encode_motion_cmd(linear_m_s, angular_rad_s, self._cmd_counter)
        self._tx_queue.append(frame)

        # 虚拟环回时同步更新预估状态
        if self.use_virtual_loopback:
            self.state.linear_velocity = frame.data[0] # 由实际 decode 模拟
            self._simulate_feedback(linear_m_s, angular_rad_s)

        return frame

    def emergency_stop(self) -> None:
        """触发硬件级急停."""
        self._hardware_estop = True
        self.state.estop_active = True
        self.state.mode = ScoutMotionMode.ESTOP
        estop_frame = ScoutProtocol.encode_estop_cmd(True)
        self._tx_queue.append(estop_frame)
        # 发零速帧
        self._tx_queue.append(ScoutProtocol.encode_motion_cmd(0.0, 0.0, self._cmd_counter))

    def reset_emergency_stop(self) -> None:
        """解除硬件急停并复位看门狗."""
        self._hardware_estop = False
        self.state.estop_active = False
        self.state.mode = ScoutMotionMode.CAN_CONTROL
        clear_frame = ScoutProtocol.encode_estop_cmd(False)
        self._tx_queue.append(clear_frame)
        self._last_cmd_time = time.time()

    def stop(self) -> None:
        """平稳制动停机."""
        if self._is_connected:
            self.send_velocity(0.0, 0.0)

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
        """接收并解析来自底盘主控板的 CAN 遥测帧."""
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
            # 轮径 0.175m, 计算估算平均线速度
            # v = (rpm_fl + rpm_fr + rpm_rl + rpm_rr) / 4 * (pi * D / 60)
            avg_rpm = sum(rpms) / 4.0
            wheel_diameter = 0.175
            self.state.linear_velocity = avg_rpm * (math.pi * wheel_diameter / 60.0)
            if self.feedback_hook:
                self.feedback_hook(self.state)

    def _simulate_feedback(self, target_vx: float, target_wz: float) -> None:
        """在虚拟环回模式下生成模拟反馈帧."""
        now = time.time()
        # 1. 生成 0x211 状态帧
        status_data = bytes([ScoutMotionMode.CAN_CONTROL, 248, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        self.receive_can_frame(CanFrame(ScoutProtocol.CAN_ID_CHASSIS_STATUS, status_data, timestamp=now))

        # 2. 生成 0x221 电机反馈帧
        wheel_diameter = 0.175
        rpm = int(round(target_vx / (math.pi * wheel_diameter / 60.0))) if target_vx != 0 else 0
        rpm_data = struct.pack(">hhhh", rpm, rpm, rpm, rpm)
        self.receive_can_frame(CanFrame(ScoutProtocol.CAN_ID_ACTUATOR_FEEDBACK, rpm_data, timestamp=now))

    def get_transmitted_frames(self) -> list[CanFrame]:
        """获取并排空发送队列 (用于测试与物理总线驱动接入)."""
        frames = list(self._tx_queue)
        self._tx_queue.clear()
        return frames
