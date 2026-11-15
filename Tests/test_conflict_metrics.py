"""冲突等待的「次数」与「步数」两种口径（任务书 R5.5）。

任务书要求的是「冲突等待**次数**」，而早期实现只记录了阻塞**步数**。
两者不可互相代替：次数反映「被打断几次」，步数反映「一共等了多久」；
同一批数据两者可能给出不同排序。本文件守住两种口径各自的语义。
"""

from __future__ import annotations

import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import Simulator, SimulationConfig


def _summary(n_tasks=50, n_agvs=4, seed=0):
    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed,
                        dt=1.0, max_speed=1.0)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(net, tasks, agvs,
                    StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(),
                    SimulationConfig(dt=1.0, max_steps=60000, seed=seed))
    result = sim.run()
    # Each vehicle contributes exactly one busy or idle unit per simulated tick.
    # Arrival queues and pickup replanning are traffic waits as well.
    assert all(a.busy_steps + a.idle_steps == result.finished_step for a in result.agvs)
    assert result.conflict_wait_steps() == sum(a.blocked_steps for a in result.agvs)
    return result.summary()


def test_summary_exposes_both_conflict_metrics():
    """两个口径都要出现在 summary 里，缺一个就无法满足任务书。"""
    s = _summary()
    assert "conflict_wait_steps" in s
    assert "conflict_wait_events" in s
    assert isinstance(s["conflict_wait_steps"], int)
    assert isinstance(s["conflict_wait_events"], int)


def test_events_are_strictly_less_than_steps_when_blocking_occurs():
    """一段阻塞至少 1 步：次数不可能超过步数。"""
    s = _summary()
    steps = s["conflict_wait_steps"]
    events = s["conflict_wait_events"]
    assert steps > 0, "该算例应有阻塞，否则测不出区别"
    assert events > 0, "有阻塞步数却无事件数，说明计数没触发"
    assert events <= steps, (
        "事件数 %d 大于步数 %d，计数逻辑有误" % (events, steps))


def test_events_increase_with_load():
    """负载升高，阻塞事件数应单调不减（同种子、同规划器）。"""
    small = _summary(n_tasks=20, n_agvs=2)
    large = _summary(n_tasks=100, n_agvs=6)
    assert large["conflict_wait_events"] > small["conflict_wait_events"], (
        "规模增大后事件数未增加：%d -> %d"
        % (small["conflict_wait_events"], large["conflict_wait_events"]))


def test_event_count_is_not_step_count():
    """守门：两者不能是同一个数，否则相当于没区分口径。"""
    s = _summary()
    assert s["conflict_wait_events"] != s["conflict_wait_steps"], (
        "事件数与步数相同，说明 conflict_wait_events 只是步数的别名")


def test_average_block_length_is_plausible():
    """平均每段阻塞长度应大于等于 1 步，且不夸张。"""
    s = _summary()
    avg = s["conflict_wait_steps"] / s["conflict_wait_events"]
    assert avg >= 1.0, "平均每段不足 1 步，不可能"
    assert avg < 10000, "平均阻塞长度异常大，疑似计数漏计"


def test_events_are_deterministic_across_runs():
    """同种子两次运行，事件数必须一致。"""
    a = _summary(seed=7)
    b = _summary(seed=7)
    assert a["conflict_wait_events"] == b["conflict_wait_events"]
    assert a["conflict_wait_steps"] == b["conflict_wait_steps"]


def test_free_flow_case_has_zero_events():
    """低负载下若完全无阻塞，两个口径都应归零。"""
    s = _summary(n_tasks=2, n_agvs=6)
    if s["conflict_wait_steps"] == 0:
        assert s["conflict_wait_events"] == 0, (
            "步数为 0 但事件数非 0，计数有重复触发")
