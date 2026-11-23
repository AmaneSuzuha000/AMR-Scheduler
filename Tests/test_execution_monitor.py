"""执行层监控：上报要能被校验，决策要能被复算。

这些测试盯的是三件容易悄悄错掉的事：
1. 少一个字段就当作没收到——否则会拿半个事件去做调度判断；
2. 不认识的通道不猜——猜错等于自己改全局路线，正是本项目禁止的行为；
3. 受阻通道两个方向都要封——物理上它是一条路，不是一个方向。
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from amr_sched.monitor import (  # noqa: E402
    DeviationEvent, DeviationEventError, ExecutionMonitor, KNOWN_EVENTS)
from amr_sched.workshop import WorkshopScene  # noqa: E402


@pytest.fixture(scope="module")
def network():
    return WorkshopScene.load().network()


@pytest.fixture(scope="module")
def monitor(network):
    return ExecutionMonitor(network)


def _event(**over):
    base = {"robot_id": "AGV_01", "event": "ROUTE_DEVIATION",
            "blocked_edge": "J_S_45-J_S_65", "current_pose": [8.0, 30.0, 0.0],
            "timestamp": 10.0}
    base.update(over)
    return base


# ---------- 校验 ----------

def test_accepts_the_documented_shape():
    ev = DeviationEvent.from_dict(_event())
    assert ev.robot_id == "AGV_01"
    assert ev.event == "ROUTE_DEVIATION"
    assert ev.blocked_edge == ("J_S_45", "J_S_65")
    assert ev.current_pose == (8.0, 30.0, 0.0)


@pytest.mark.parametrize("missing", ["robot_id", "event", "blocked_edge",
                                     "current_pose", "timestamp"])
def test_rejects_an_event_missing_any_documented_field(missing):
    raw = _event()
    del raw[missing]
    with pytest.raises(DeviationEventError) as info:
        DeviationEvent.from_dict(raw)
    assert missing in str(info.value)


def test_rejects_an_unknown_event_type():
    with pytest.raises(DeviationEventError):
        DeviationEvent.from_dict(_event(event="SOMETHING_ELSE"))


def test_rejects_a_blocked_edge_it_cannot_parse():
    with pytest.raises(DeviationEventError):
        DeviationEvent.from_dict(_event(blocked_edge="J_S_45"))


def test_rejects_a_blocked_edge_with_the_same_endpoint_twice():
    with pytest.raises(DeviationEventError):
        DeviationEvent.from_dict(_event(blocked_edge="J_S_45-J_S_45"))


def test_rejects_a_pose_that_is_not_numeric():
    with pytest.raises(DeviationEventError):
        DeviationEvent.from_dict(_event(current_pose=["a", "b"]))


def test_edge_order_does_not_matter():
    """(A,B) 和 (B,A) 是同一条通道，必须归一成同一个值。"""
    a = DeviationEvent.from_dict(_event(blocked_edge="J_S_45-J_S_65"))
    b = DeviationEvent.from_dict(_event(blocked_edge=["J_S_65", "J_S_45"]))
    assert a.blocked_edge == b.blocked_edge


# ---------- 决策 ----------

def test_unknown_edge_is_ignored_not_guessed(monitor):
    """网络里没有的通道不做任何改变——猜一条边就等于自己改了全局路线。"""
    decision = monitor.handle(
        DeviationEvent.from_dict(_event(blocked_edge="NO_SUCH_A-NO_SUCH_B")),
        goal="J_S_125")
    assert decision.action == "ignore"
    assert "不在网络中" in decision.reason


def test_known_edge_produces_a_detour_that_avoids_it(monitor):
    decision = monitor.handle(DeviationEvent.from_dict(_event()), goal="J_S_125")
    assert decision.action == "reroute"
    assert len(decision.nodes) >= 2
    blocked = set(decision.blocked_edge)
    for a, b in zip(decision.nodes, decision.nodes[1:]):
        assert {a, b} != blocked, "绕行路线仍然用了受阻通道"
    assert decision.meters > 0


def test_the_detour_is_longer_than_the_straight_corridor(monitor):
    """挡掉直通道后应当变长；等长说明根本没绕开。"""
    direct = monitor.handle(
        DeviationEvent.from_dict(_event(blocked_edge="J_S_45-J_S_65")),
        goal="J_S_125")
    assert direct.meters > 117.0


def test_blocked_edge_is_directed_both_ways(network, monitor):
    """受阻通道两个方向都封：物理上它是一条路。"""
    decision = monitor.handle(
        DeviationEvent.from_dict(_event(blocked_edge="J_S_65-J_S_45")),
        goal="J_S_125")
    assert decision.action == "reroute"
    blocked = {"J_S_45", "J_S_65"}
    for a, b in zip(decision.nodes, decision.nodes[1:]):
        assert {a, b} != blocked


def test_a_dead_end_target_reports_reassign(network, monitor):
    """绕开后到不了，就把结论交回上层，而不是自己挑一辆车。

    车间里的工位是死胡同：只有一条入口（入度 1）。封掉它，从路网任何别处都进不去。
    """
    incoming: dict[str, list] = {n: [] for n in network.nodes}
    for edge in network.edges.values():
        incoming[edge.dst].append(edge)
    single = [n for n, ins in incoming.items() if len(ins) == 1]
    assert single, "车间网络里应当存在只有一条入口的工位"

    target = single[0]
    entry = incoming[target][0]
    decision = monitor.handle(
        DeviationEvent.from_dict(_event(
            blocked_edge="%s-%s" % (entry.src, entry.dst),
            current_pose=[8.0, 30.0, 0.0])),
        goal=target)
    assert decision.action == "reassign", decision.reason
    assert "不可达" in decision.reason


def test_decision_is_json_shaped_for_the_wire(network):
    import json
    monitor = ExecutionMonitor(network)
    decision = monitor.handle(DeviationEvent.from_dict(_event()), goal="J_S_125")
    back = json.loads(json.dumps(decision.to_dict(), ensure_ascii=False))
    assert back["action"] == "reroute"
    assert back["nodes"] == list(decision.nodes)
    assert back["blocked_edge"] == list(decision.blocked_edge)


def test_history_is_bounded(network):
    monitor = ExecutionMonitor(network)
    for _ in range(300):
        monitor.handle(DeviationEvent.from_dict(_event()), goal="J_S_125")
    assert len(monitor.history) <= 256


def test_only_the_documented_event_type_is_known():
    assert KNOWN_EVENTS == frozenset({"ROUTE_DEVIATION"})
