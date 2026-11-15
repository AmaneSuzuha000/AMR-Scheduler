"""mock 桥与真实心跳行为的回归测试。

这组测试回答一个问题：**不发 ping 会怎样**。
答案是位姿冻结且不恢复——这是实测结论，必须固化成测试，
否则下次有人写个「只读客户端」又会踩同一个坑。
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from mock_bridge import MockBridge  # noqa: E402

from amr_sched.bridge_client import BridgeClient  # noqa: E402


def _collect(client, seconds):
    poses, stats = [], []
    client.on_pose = poses.append
    client.on_status = stats.append
    client.poll(seconds)
    return poses, stats


def _travel(poses):
    d = 0.0
    for a, b in zip(poses, poses[1:]):
        d += math.hypot(b.x - a.x, b.y - a.y)
    return d


@pytest.fixture
def bridge():
    b = MockBridge(watchdog=0.5)
    b.start()
    yield b
    b.stop()


def test_hello_is_first_message(bridge):
    c = BridgeClient("127.0.0.1", bridge.port)
    c.connect()
    got = []
    c.on_hello = got.append
    c.poll(0.5)
    c.stop()
    assert got, "没有收到 hello"
    assert got[0]["proto"] == 1
    assert got[0]["frame"] == "sim"


def test_without_ping_pose_freezes(bridge):
    """核心回归：不发 ping，位姿必须在 watchdog 后冻结。"""
    c = BridgeClient("127.0.0.1", bridge.port)
    c.connect()
    poses, stats = _collect(c, 2.5)
    c.stop()
    assert len(poses) > 20, "样本太少，无法判断"
    # 后半段应完全静止
    half = poses[len(poses) // 2:]
    assert _travel(half) < 1e-6, "未发 ping 位姿居然还在动，watchdog 失效"
    assert any(s.error == "watchdog" for s in stats), "没有报告 watchdog"


def test_with_ping_pose_keeps_moving(bridge):
    """发 ping 时位姿必须持续更新。"""
    c = BridgeClient("127.0.0.1", bridge.port, ping_interval=0.2)
    c.start()
    poses, stats = _collect(c, 2.5)
    c.stop()
    assert len(poses) > 20
    assert _travel(poses) > 0.3, "发了 ping 位姿却几乎不动"
    assert not any(s.is_stalled for s in stats), "发 ping 仍被判 watchdog"
    assert bridge.n_pings >= 5, "客户端没真的发出心跳"


def test_moving_client_survives_full_watchdog_period(bridge):
    """跑满多个 watchdog 周期仍不能停。"""
    c = BridgeClient("127.0.0.1", bridge.port, ping_interval=0.2)
    c.start()
    poses, _ = _collect(c, 2.0)
    c.stop()
    # 每段位移都不应为 0（不应中途冻结）
    for a, b in zip(poses, poses[1:]):
        d = math.hypot(b.x - a.x, b.y - a.y)
        assert d > 0, "中途出现静止帧，说明心跳有断档"


def test_pose_fields_and_seq_increase(bridge):
    c = BridgeClient("127.0.0.1", bridge.port, ping_interval=0.2)
    c.start()
    poses, _ = _collect(c, 1.0)
    c.stop()
    assert poses
    seqs = [p.seq for p in poses]
    assert seqs == sorted(seqs), "seq 不单调"
    for p in poses:
        assert p.vehicle == "agv1"
        assert p.t_wall > 0


def test_default_ping_interval_beats_watchdog(bridge):
    """用默认 ping_interval（不显式给 0.2）也必须活下来。"""
    c = BridgeClient("127.0.0.1", bridge.port)
    assert c.ping_interval < 0.5
    c.start()
    poses, stats = _collect(c, 2.0)
    c.stop()
    assert _travel(poses) > 0.3
    assert not any(s.is_stalled for s in stats)


def test_no_watchdog_env_always_moves():
    """watchdog 关掉时，不发 ping 也应该动——这解释了现场为何复现不出来。"""
    b = MockBridge(watchdog=0)
    b.start()
    try:
        c = BridgeClient("127.0.0.1", b.port)
        c.connect()
        poses, stats = _collect(c, 1.5)
        c.stop()
        assert _travel(poses) > 0.1
        assert not any(s.is_stalled for s in stats)
    finally:
        b.stop()
