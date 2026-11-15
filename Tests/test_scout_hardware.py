# -*- coding: utf-8 -*-
"""SCOUT Mini 硬件抽象与 CAN 通信单元测试 (100% 覆盖)."""

import math
import struct
import time
import pytest

from tools.live_bridge.scout_hardware import (
    CanFrame,
    ScoutChassisState,
    ScoutErrorCode,
    ScoutHardwareBridge,
    ScoutMotionMode,
    ScoutProtocol,
)


def test_can_frame_invariants():
    frame = CanFrame(0x421, b"\x01\x02\x03\x04")
    assert frame.arbitration_id == 0x421
    assert len(frame.data) == 4
    assert not frame.is_extended

    with pytest.raises(ValueError, match="must not exceed 8 bytes"):
        CanFrame(0x421, b"\x00" * 9)


def test_scout_protocol_encode_motion_cmd():
    # 正常范围编码
    frame = ScoutProtocol.encode_motion_cmd(0.85, -1.20, counter=5)
    assert frame.arbitration_id == 0x421
    assert len(frame.data) == 8

    v_mm, w_mrad, mode, cnt, _ = struct.unpack(">hhBBH", frame.data)
    assert v_mm == 850
    assert w_mrad == -1200
    assert mode == 0x01
    assert cnt == 5

    # 极限饱和钳位
    frame_max = ScoutProtocol.encode_motion_cmd(5.0, -10.0, counter=0)
    v_max, w_max, _, _, _ = struct.unpack(">hhBBH", frame_max.data)
    assert v_max == 1500   # 1.5 m/s 钳位
    assert w_max == -2500  # -2.5 rad/s 钳位


def test_scout_protocol_estop_encode_decode():
    frame_on = ScoutProtocol.encode_estop_cmd(True)
    assert frame_on.arbitration_id == 0x441
    assert frame_on.data[1] == 0x01

    frame_off = ScoutProtocol.encode_estop_cmd(False)
    assert frame_off.arbitration_id == 0x441
    assert frame_off.data[1] == 0x00


def test_scout_protocol_decode_chassis_status():
    payload = bytes([0x01, 248, 0x00, 0x00, 0x00, 0x00])
    mode, volt, estop, errs = ScoutProtocol.decode_chassis_status(payload)
    assert mode == ScoutMotionMode.CAN_CONTROL
    assert volt == pytest.approx(24.8)
    assert not estop
    assert errs == 0

    # 包含急停与低电量
    payload_err = bytes([0x03, 210, 0x01, 0x01, 0x00, 0x00])
    mode_err, volt_err, estop_err, errs_err = ScoutProtocol.decode_chassis_status(payload_err)
    assert mode_err == ScoutMotionMode.ESTOP
    assert volt_err == pytest.approx(21.0)
    assert estop_err
    assert errs_err == ScoutErrorCode.BATTERY_LOW


def test_scout_protocol_decode_actuator_feedback():
    payload = struct.pack(">hhhh", 120, 120, 118, 122)
    rpms = ScoutProtocol.decode_actuator_feedback(payload)
    assert rpms == (120, 120, 118, 122)


def test_scout_hardware_bridge_lifecycle():
    bridge = ScoutHardwareBridge(can_channel="vcan0", use_virtual_loopback=True)
    assert not bridge._is_connected

    with pytest.raises(RuntimeError, match="not connected"):
        bridge.send_velocity(0.5, 0.0)

    assert bridge.connect()
    assert bridge.state.mode == ScoutMotionMode.CAN_CONTROL

    # 发送运动指令
    f = bridge.send_velocity(0.5, 0.1)
    assert f.arbitration_id == 0x421
    tx = bridge.get_transmitted_frames()
    assert len(tx) >= 1

    # 虚拟环回反馈已更新
    assert bridge.state.battery_voltage == pytest.approx(24.8)
    assert bridge.state.is_healthy

    # 断开连接
    bridge.disconnect()
    assert bridge.state.mode == ScoutMotionMode.STANDBY


def test_scout_hardware_bridge_estop():
    bridge = ScoutHardwareBridge(can_channel="vcan0")
    bridge.connect()

    bridge.send_velocity(1.0, 0.0)
    # 触发急停
    bridge.emergency_stop()
    assert bridge.state.estop_active
    assert bridge.state.mode == ScoutMotionMode.ESTOP
    assert not bridge.state.is_healthy

    # 急停状态下尝试发非零速度，强制被拦截为 0
    f = bridge.send_velocity(1.2, 0.5)
    v_mm, w_mrad, _, _, _ = struct.unpack(">hhBBH", f.data)
    assert v_mm == 0
    assert w_mrad == 0

    # 复位急停
    bridge.reset_emergency_stop()
    assert not bridge.state.estop_active
    assert bridge.state.mode == ScoutMotionMode.CAN_CONTROL


def test_scout_hardware_bridge_watchdog():
    bridge = ScoutHardwareBridge(can_channel="vcan0", watchdog_timeout_s=0.25)
    bridge.connect()

    t0 = 100.0
    bridge.send_velocity(0.5, 0.0)
    bridge._last_cmd_time = t0

    # 0.10s 正常
    assert bridge.check_watchdog(current_time=t0 + 0.10)
    assert (bridge.state.error_flags & ScoutErrorCode.CAN_TIMEOUT) == 0

    # 0.30s (> 0.25s) 超时触发制动
    assert not bridge.check_watchdog(current_time=t0 + 0.30)
    assert (bridge.state.error_flags & ScoutErrorCode.CAN_TIMEOUT) != 0
