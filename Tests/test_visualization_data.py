"""R4.4 状态监控 + R4.5 过程可视化的数据契约测试。

这几项守住的是「可视化能看到什么」，不是「画得好不好看」：
如果 StepRecord 里没有预约快照与抢占事件，界面再漂亮也解释不了冲突。
"""

from __future__ import annotations

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.reservation import Interval, ReservationTable
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import Simulator, SimulationConfig


def _run(policy="none", n_tasks=30, n_agvs=4, seed=0):
    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed,
                        dt=1.0, max_speed=1.0, urgent_ratio=0.2)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(),
                    SimulationConfig(dt=1.0, max_steps=60_000, seed=seed,
                                     reschedule_policy=ReschedulePolicy(policy),
                                     record_trajectory=True))
    return sim.run(), sim


def test_reservation_snapshot_isolated_in_window():
    """snapshot 只返回与窗口重叠的预约。"""
    r = ReservationTable()
    r.reserve_node("A", Interval(10, 20), "AGV0")
    r.reserve_node("B", Interval(100, 110), "AGV1")
    assert set(r.snapshot(12, 13)) == {"A"}
    assert set(r.snapshot(105, 106)) == {"B"}
    assert r.snapshot(50, 60) == {}


def test_reservation_snapshot_reports_owner():
    """快照必须带占用者，否则看不出「谁把位置订走了」。"""
    r = ReservationTable()
    r.reserve_node("W1", Interval(5, 15), "AGV7")
    got = r.snapshot(10, 11)
    assert got["W1"] == [(5, 15, "AGV7")]


def test_edge_snapshot_is_undirected():
    """边的快照按无向键归并：单车道两个方向占的是同一条通道。"""
    r = ReservationTable()
    r.reserve_edge("A", "B", Interval(1, 5), "AGV0")
    r.reserve_edge("B", "A", Interval(6, 9), "AGV1")
    snap = r.edge_snapshot(0, 10)
    assert list(snap) == ["A|B"]
    assert len(snap["A|B"]) == 2


def test_trajectory_records_reservation_holds():
    """轨迹必须记录节点预约占用（预约层的数据源）。"""
    result, sim = _run()
    assert result.trajectory, "没有轨迹"
    with_hold = [s for s in result.trajectory if s.node_holds]
    assert with_hold, "整段轨迹没有任何预约占用记录"
    for snap in with_hold[:20]:
        for node, owners in snap.node_holds.items():
            assert isinstance(node, str) and node
            assert isinstance(owners, list) and owners
            assert all(isinstance(o, str) for o in owners)


def test_trajectory_records_preemption_events():
    """抢占发生时，轨迹里必须有事件，且带「对方是否已装货」。"""
    result, sim = _run(policy="preempt_any")
    n_pre = result.summary()["preemption_count"]
    assert n_pre > 0, "preempt_any 竟然没有发生任何抢占，场景或策略失效"
    seen = [e for s in result.trajectory for e in s.preemptions]
    assert len(seen) == n_pre, (
        "轨迹里的抢占事件 {} 条与 preemption_count {} 不一致"
        .format(len(seen), n_pre))
    for urgent_id, agv_id, victim_task, picked in seen:
        assert urgent_id and agv_id and victim_task
        assert isinstance(picked, bool)


def test_preemption_events_carry_step_and_detail():
    """result 上的抢占明细要含 step，方便与轨迹对齐。"""
    result, sim = _run(policy="preempt_any")
    assert result.preemption_events, "没有抢占明细"
    for ev in result.preemption_events:
        assert len(ev) == 5
        step, urgent_id, agv_id, victim_task, picked = ev
        assert isinstance(step, int)
        assert urgent_id.startswith("T")
        assert agv_id.startswith("AGV")
        assert isinstance(picked, bool)


def test_picking_unpicked_has_zero_material_waste():
    """抢未取货车必须零物料浪费——这是 6.4 节结论的前提。"""
    result, sim = _run(policy="preempt_unpicked")
    s = result.summary()
    if s["preemption_count"] == 0:
        return  # 该配置没触发抢占，跳过
    assert s["material_waste_steps"] == 0, (
        "preempt_unpicked 产生了 {} s 物料浪费".format(
            s["material_waste_steps"]))


def test_node_safe_preemption_remains_active_in_busy_fleets():
    """到节点后重试的空载抢占，在繁忙车队中仍能实际发生。"""
    total = 0
    for seed in range(3):
        result, _ = _run(policy="preempt_unpicked", n_tasks=50, seed=seed)
        total += result.summary()["preemption_count"]
    assert total > 0, (
        "3 个种子里 preempt_unpicked 一次抢占都没有")
