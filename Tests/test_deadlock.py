"""单车道死锁与容量校验的测试（R5.3 压力工况）。"""

from __future__ import annotations

from collections import Counter

import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.network import build_demo_network
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator


def _run(cap=2, one_way=False, agvs=4, win=600, seed=0, max_steps=60_000,
         timeout=200):
    net = build_demo_network(aisle_capacity=cap, one_way_ring=one_way)
    sc = ScenarioConfig(n_tasks=50, n_agvs=agvs, seed=seed,
                        release_window=win, aisle_capacity=cap)
    net, tasks, ag = generate_scenario(sc, network=net)
    r = Simulator(net, tasks, ag, SpaceTimeAStarPlanner(1.0, 1.0),
                  GreedyAllocator(),
                  SimulationConfig(max_steps=max_steps,
                                   deadlock_timeout=timeout,
                                   record_trajectory=True)).run()
    return net, r


def _max_over_capacity(net, result) -> int:
    """轨迹上任意时刻、任意节点的最大超容量数（0 表示从不超容）。"""
    worst = 0
    for rec in result.trajectory:
        pos = getattr(rec, "agv_positions", None) or getattr(rec, "positions", None)
        if not pos:
            continue
        for nid, c in Counter(pos.values()).items():
            if "->" in nid:      # 在途的车不算停在节点上
                continue
            worst = max(worst, c - net.node(nid).capacity)
    return worst


def test_double_lane_never_exceeds_node_capacity():
    """双车道下节点容量必须始终不被突破（曾经一次运行最多 917 次重叠）。"""
    net, r = _run(cap=2, agvs=4)
    assert _max_over_capacity(net, r) <= 0


def test_single_lane_never_exceeds_node_capacity():
    """单车道压力工况下同样不得超容。

    这条是回归：曾误判"R2 容量 1 却停了 3 台车"，实际是末态快照把
    在途车辆算错了。逐步审计证明真实占用从不超容。
    """
    net, r = _run(cap=1, agvs=4)
    assert _max_over_capacity(net, r) <= 0


def test_single_lane_has_no_collisions():
    """无论哪种车道，"移动中相撞"都必须是 0。"""
    for cap in (1, 2):
        _net, r = _run(cap=cap, agvs=4)
        assert r.summary()["collision_events"] == 0
        assert r.summary()["hard_failures"] == 0


def test_double_lane_completes_everything():
    """双车道是主要工作工况，必须 100% 完成。"""
    for agvs in (2, 4, 6):
        _net, r = _run(cap=2, agvs=agvs, max_steps=200_000)
        assert r.summary()["tasks_completed"] == 50, "双车道 %d 车未完成全部任务" % agvs


def test_capacity_wait_marks_blocked():
    """车因目标节点满而等待时，必须被记为阻塞。

    这条守的是两件事，两者含义相反但都必须成立：

    1. **阻塞必须真的被记录**——否则 ``_break_deadlock`` 永远等不到超时，
       那个 0 是假象而不是"没有死锁"。
    2. **但打破次数不该被滥用**——第 65 轮修掉 ``_park_step()`` 的咽喉缺陷后，
       单车道 6 车已经能顺畅跑完，deadlock_breaks 降到 0。

    所以这里不再断言"必须 > 0"（那是旧缺陷的症状），
    而是断言「阻塞确实发生过」+「最终没有靠打破机制收场」。
    """
    net, r = _run(cap=1, agvs=6, timeout=200)
    s = r.summary()
    assert s["tasks_completed"] == 50
    for record in r.trajectory:
        channels = Counter(tuple(sorted(pos.split("->", 1)))
                           for pos in record.agv_positions.values() if "->" in pos)
        for channel, count in channels.items():
            capacity = min(net.edge(*edge).capacity for edge in (channel, channel[::-1])
                           if net.has_edge(*edge))
            assert count <= capacity, (record.step, channel, count, capacity)
    # 阻塞必须被真实记录：单车道 6 车不可能零等待
    assert s["conflict_wait_steps"] > 0, "阻塞完全没有被记录，阻塞标记失效"
    # 修复后这类场景应当靠正常调度跑完，而不是靠反复打破
    assert s["deadlock_breaks"] == 0, (
        "单车道 6 车仍有 %d 次死锁打破，说明还有车卡在咽喉节点上"
        % s["deadlock_breaks"])


def test_no_deadlock_break_when_disabled():
    """timeout<=0 时不触发打破，行为应退回纯等待。"""
    _net, r = _run(cap=1, agvs=4, timeout=0)
    assert r.summary()["deadlock_breaks"] == 0


def test_one_way_ring_reduces_edges_but_keeps_connectivity():
    """单向环行只删方向，不能让任何节点对变得不可达。"""
    two = build_demo_network(aisle_capacity=1, one_way_ring=False)
    one = build_demo_network(aisle_capacity=1, one_way_ring=True)
    assert len(one.edges) < len(two.edges)
    assert one.unreachable_pairs(1.0) == [], "单向环行造成了不可达节点对"


def test_one_way_ring_maintains_ring_direction():
    """主环上不得出现逆时针方向的边。"""
    net = build_demo_network(aisle_capacity=1, one_way_ring=True)
    outer = ["R0", "R1", "R2", "R5", "R4", "R3"]
    for i in range(len(outer)):
        a, b = outer[i], outer[(i + 1) % len(outer)]
        assert net.has_edge(a, b), "缺少顺时针边 %s->%s" % (a, b)
        assert not net.has_edge(b, a), "不应存在逆时针边 %s->%s" % (b, a)


@pytest.mark.parametrize("cap", [1, 2])
def test_reschedule_never_loses_tasks(cap):
    """无论车道容量，任务状态都必须收敛：要么完成，要么仍在待分配/执行中。

    不允许出现"任务凭空消失"或状态与 AGV 不匹配。
    """
    _net, r = _run(cap=cap, agvs=4)
    ids = [t.task_id for t in r.tasks]
    assert len(ids) == len(set(ids)) == 50
    done = [t for t in r.tasks if t.complete_step is not None]
    assert len(done) == r.summary()["tasks_completed"]

def test_loaded_deadlock_recovery_preserves_original_delivery(monkeypatch):
    """载货死锁恢复须保持原取货点、载荷归属，并完成原运输。"""
    from amr_sched.allocation.greedy import GreedyAllocator
    from amr_sched.entities import TaskStatus
    from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
    from amr_sched.scenarios import ScenarioConfig, generate_scenario
    from amr_sched.simulation import SimulationConfig, Simulator

    sc = ScenarioConfig(n_tasks=50, n_agvs=4, seed=0, dt=1.0, max_speed=1.0)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(
        net, tasks, agvs, StaticDijkstraPlanner(1.0, 1.0), GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=200_000, seed=0,
                         process_time=450.0))
    result = sim.run()
    s = result.summary()
    assert s["tasks_completed"] == 50, (
        "加工时间 450s 下只完成 %d/50，疑似活锁" % s["tasks_completed"])
    assert s["deadlock_breaks"] < 100, (
        "死锁打破 %d 次，远超正常水平，疑似反复解锁-再锁"
        % s["deadlock_breaks"])


    from pathlib import Path
    from amr_sched.workshop import WorkshopScene, legal_task_pairs, parking_slots

    scene = WorkshopScene.load(Path(__file__).resolve().parents[1] / 'scenes/rail_freight_workshop_v1.json')
    # Force the timeout fallback in these frozen inputs even when normal
    # coordination would let the vehicles pass before the timeout.
    monkeypatch.setattr(Simulator, "_coordinate_passing", lambda self, step: None)
    # Frozen cases 256 and 808 originally rewrote a loaded task's pickup.
    for window, seed in ((200, 2), (1800, 4)):
        config = ScenarioConfig(n_tasks=100, n_agvs=6, seed=seed, release_window=window)
        net, tasks, agvs = generate_scenario(config, network=scene.network(),
            pairs=legal_task_pairs(scene), start_pool=parking_slots(scene))
        endpoints = {t.task_id: (t.pickup, t.dropoff) for t in tasks}
        custody = {}
        sim = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(1., 1.), GreedyAllocator(),
            SimulationConfig(seed=seed, reservation_window=1, max_steps=12000))
        sim.reset()
        while sim.step_once():
            channels = Counter(tuple(sorted(tr[:2])) for tr in sim._transit.values())
            assert all(count <= 1 for count in channels.values()), (window, seed, sim._step_cursor, channels)
            for task in tasks:
                assert (task.pickup, task.dropoff) == endpoints[task.task_id]
                if task.status is TaskStatus.PICKED:
                    custody.setdefault(task.task_id, (task.assigned_agv, task.assigned_step))
                if task.task_id in custody:
                    assert task.status in (TaskStatus.PICKED, TaskStatus.COMPLETED)
                    assert (task.assigned_agv, task.assigned_step) == custody[task.task_id]
        assert all(task.status is TaskStatus.COMPLETED for task in tasks), (window, seed)
        assert sim._build_result().deadlock_breaks > 0, (window, seed)


def test_no_agv_stuck_beyond_timeout():
    """不允许任何车的累计阻塞超过总步数的一半。"""
    from amr_sched.allocation.greedy import GreedyAllocator
    from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
    from amr_sched.scenarios import ScenarioConfig, generate_scenario
    from amr_sched.simulation import SimulationConfig, Simulator

    sc = ScenarioConfig(n_tasks=50, n_agvs=4, seed=0, dt=1.0, max_speed=1.0)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(
        net, tasks, agvs, StaticDijkstraPlanner(1.0, 1.0), GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=200_000, seed=0,
                         process_time=450.0))
    result = sim.run()
    total = result.summary()["finished_step"]
    for agv in sim.agvs:
        assert agv.blocked_steps < total * 0.5, (
            "%s 阻塞 %d 步 / 总 %d 步，超过一半，疑似卡死"
            % (agv.agv_id, agv.blocked_steps, total))
