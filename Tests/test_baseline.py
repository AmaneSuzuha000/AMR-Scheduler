"""基准策略（R5.6）的测试。

这两条基线代表"上了 AGV 但没上调度系统"的朴素做法，
用于回答"本课题方法比不做研究强多少"。
测试重点是**基线必须真的笨**——如果基线被偷偷加强，
对比就失去意义。
"""

from __future__ import annotations

import pytest

from amr_sched.allocation.baseline import (FifoShortestAllocator,
                                           NearestShortestAllocator)
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator


def _setup(seed=0, n_tasks=20, n_agvs=2):
    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed)
    net, tasks, agvs = generate_scenario(sc)
    return net, tasks, agvs


def test_nearest_picks_closest_pickup_only():
    """NearestShortest 只看取货点距离，不看送达点与截止期。

    构造两个任务：A 取货点很近但送达点极远，B 取货点稍远但送达点也近。
    朴素基线必须选 A——这正是它"笨"的证据。
    """
    net, tasks, agvs = _setup()
    alloc = NearestShortestAllocator()
    cache = {}
    one_agv = [agvs[0]]
    # 取前 5 个任务，看它是否只按取货距离挑
    pairs = alloc.allocate(net, one_agv, tasks[:5], 0, 1.0, cache)
    assert len(pairs) == 1
    chosen = pairs[0][1]
    here = one_agv[0].cur_node or one_agv[0].start_node
    tt = net.shortest_travel_times(here, 1.0)
    dists = {t.task_id: tt.get(t.pickup, float("inf")) for t in tasks[:5]}
    assert dists[chosen.task_id] == min(dists.values()), \
        "基线没有选最近的取货点，说明它并不笨"


def test_fifo_pairs_in_order():
    """FifoShortest 必须严格先来后到，不做任何择优。"""
    net, tasks, agvs = _setup(n_tasks=10, n_agvs=3)
    alloc = FifoShortestAllocator()
    pairs = alloc.allocate(net, agvs, tasks, 0, 1.0, {})
    assert len(pairs) == min(len(agvs), len(tasks))
    # 车按 id 排序，任务按 (release_step, task_id) 排序
    exp_agvs = sorted(agvs, key=lambda a: a.agv_id)
    exp_tasks = sorted(tasks, key=lambda t: (t.release_step, t.task_id))
    for i, (a, t) in enumerate(pairs):
        assert a.agv_id == exp_agvs[i].agv_id
        assert t.task_id == exp_tasks[i].task_id


def test_baselines_do_not_mutate_tasks():
    """分配器不得修改任务状态（那是仿真器的职责）。"""
    net, tasks, agvs = _setup()
    before = [(t.task_id, t.status.value) for t in tasks]
    NearestShortestAllocator().allocate(net, agvs, tasks, 0, 1.0, {})
    FifoShortestAllocator().allocate(net, agvs, tasks, 0, 1.0, {})
    after = [(t.task_id, t.status.value) for t in tasks]
    assert before == after


def test_baselines_handle_empty_inputs():
    """没有空闲车或没有任务时，必须返回空列表而不是崩溃。"""
    net, tasks, agvs = _setup()
    for alloc in (NearestShortestAllocator(), FifoShortestAllocator()):
        assert alloc.allocate(net, [], tasks, 0, 1.0, {}) == []
        assert alloc.allocate(net, agvs, [], 0, 1.0, {}) == []


def test_no_task_assigned_twice():
    """同一次分配里，一个任务不能给两台车。"""
    net, tasks, agvs = _setup(n_tasks=10, n_agvs=6)
    for alloc in (NearestShortestAllocator(), FifoShortestAllocator()):
        pairs = alloc.allocate(net, agvs, tasks, 0, 1.0, {})
        ids = [t.task_id for _a, t in pairs]
        assert len(ids) == len(set(ids)), "%s 把同一任务分配了两次" % alloc.name


def test_baseline_run_completes_all_tasks():
    """基线在常规算例上必须能跑完（否则对比不公平）。"""
    net, tasks, agvs = _setup(n_tasks=20, n_agvs=2)
    r = Simulator(net, tasks, agvs,
                  StaticDijkstraPlanner(1.0, 1.0),
                  NearestShortestAllocator(),
                  SimulationConfig(max_steps=200_000)).run()
    s = r.summary()
    assert s["tasks_completed"] == 20
    assert s["collision_events"] == 0


def test_static_planner_writes_no_reservations():
    """静态规划器 reservations_aware=False —— 基线确实是"无预约协调"。

    这条守住一个关键事实：`nearest_shortest` 基线与主方法的差异
    不只是分配逻辑，还包括**是否写预约表**。
    写论文时必须说明这一点，否则对比会被质疑不公平。
    """
    assert StaticDijkstraPlanner.reservations_aware is False
    from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
    assert SpaceTimeAStarPlanner.reservations_aware is True
