"""通道容量与负荷算例的测试（任务书 R5.3）。"""

from __future__ import annotations

import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.network import NodeKind, build_demo_network
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_aisle_capacity_applied_to_nodes_and_edges(cap):
    """主环节点与边的容量必须与 aisle_capacity 一致。"""
    net = build_demo_network(aisle_capacity=cap)
    for nid in ("R0", "R1", "R2", "R3", "R4", "R5"):
        assert net.node(nid).capacity == cap
    for a, b in (("R0", "R1"), ("R1", "R4"), ("R3", "R4")):
        assert net.edge(a, b).capacity == cap


def test_aisle_capacity_does_not_change_topology():
    """容量只影响占用约束，不该改变节点数、边数或连通性。"""
    base = build_demo_network(aisle_capacity=2)
    single = build_demo_network(aisle_capacity=1)
    assert len(base.nodes) == len(single.nodes)
    assert len(base.edges) == len(single.edges)
    assert base.unreachable_pairs(1.0) == single.unreachable_pairs(1.0) == []


def test_workstations_keep_capacity_one():
    """工位容量恒为 1，不随 aisle_capacity 变化。"""
    for cap in (1, 2, 3):
        net = build_demo_network(aisle_capacity=cap)
        for n in net.nodes_of_kind(NodeKind.WORKSTATION):
            assert n.capacity == 1


def test_scenario_passes_aisle_capacity():
    """ScenarioConfig 的 aisle_capacity 必须真的传到网络里。"""
    sc = ScenarioConfig(n_tasks=10, n_agvs=2, seed=0, aisle_capacity=1)
    net, _tasks, _agvs = generate_scenario(sc)
    assert net.node("R0").capacity == 1


@pytest.mark.parametrize("window", [1800, 600, 200])
def test_release_window_bounds_task_release(window):
    """所有任务的下达步必须落在 [0, release_window) 内。"""
    sc = ScenarioConfig(n_tasks=30, n_agvs=3, seed=5, release_window=window)
    _net, tasks, _agvs = generate_scenario(sc)
    assert all(0 <= t.release_step < window for t in tasks)


def test_double_lane_completes_all_tasks():
    """双车道下系统必须能 100% 完成任务（回归：曾经会卡死在半路）。"""
    sc = ScenarioConfig(n_tasks=50, n_agvs=4, seed=0, aisle_capacity=2)
    net, tasks, agvs = generate_scenario(sc)
    r = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(1.0, 1.0),
                  GreedyAllocator(), SimulationConfig(max_steps=200_000)).run()
    s = r.summary()
    assert s["tasks_completed"] == 50
    assert s["collision_events"] == 0
    assert s["hard_failures"] == 0


def test_scenario_regenerates_fresh_tasks():
    """同一 ScenarioConfig 两次生成必须是两批独立的任务对象。

    这是实际踩过的坑：把同一批 Task 对象喂给两次仿真，第二次运行会在
    第 0 步看到"全部已完成"（因为第一次把状态改成了 COMPLETED），
    得出 makespan=0 的假数据。实验脚本必须每次重新生成。
    """
    sc = ScenarioConfig(n_tasks=15, n_agvs=2, seed=3)
    _n1, t1, _a1 = generate_scenario(sc)
    _n2, t2, _a2 = generate_scenario(sc)
    assert [x.task_id for x in t1] == [x.task_id for x in t2]
    assert all(a is not b for a, b in zip(t1, t2)), "两次生成必须是不同对象"


def test_reusing_mutated_tasks_is_now_safe():
    """复用被改过的 Task 曾经会得到"零步完成"的假数据；第 64 轮已修好。

    历史
    ----
    这个测试原本断言的是**反例**：把跑完的同一批 Task 对象再喂给一次仿真，
    第二次会在第 0 步看到"全部已完成"，得出 makespan=0 的假数据。
    当时的结论是"实验脚本必须每次重新生成任务"。

    第 64 轮为 Web 控制台做分步执行时，把 ``reset()`` 补成真正幂等
    （原先漏重置任务状态、预约表、计数器等 8 项）。副产品就是这个陷阱
    被彻底修掉了——``run()`` 现在会先把状态复位再跑。

    因此本测试改为断言**新的正确行为**，并保留 r1 的指标作为对照，
    确保复用与不复制出同一份结果。
    """
    sc = ScenarioConfig(n_tasks=10, n_agvs=2, seed=1)
    net, tasks, agvs = generate_scenario(sc)
    r1 = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(1.0, 1.0),
                   GreedyAllocator(), SimulationConfig(max_steps=100_000)).run()
    assert r1.summary()["tasks_completed"] == 10

    # 复用同一批已被改成 COMPLETED 的任务：现在应当完整重跑，而不是 0 步结束
    r2 = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(1.0, 1.0),
                   GreedyAllocator(), SimulationConfig(max_steps=100_000)).run()
    assert r2.summary()["tasks_completed"] == 10, (
        "复用任务对象后未能完整重跑，reset() 可能又有遗漏")
    assert r2.finished_step == r1.finished_step, (
        "复用与新建的 finished_step 不一致：%d vs %d"
        % (r2.finished_step, r1.finished_step))
    assert abs(r2.summary()["makespan_seconds"]
               - r1.summary()["makespan_seconds"]) < 1e-9, "复用后 makespan 漂移"
