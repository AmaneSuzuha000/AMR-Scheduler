"""仿真主循环与四项指标测试。"""

import math

import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.entities import Task, TaskPriority, TaskStatus
from amr_sched.network import build_demo_network
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator


def _run(planner_name="spacetime_astar", n_tasks=12, n_agvs=2, seed=0, max_steps=60_000):
    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed)
    net, tasks, agvs = generate_scenario(sc)
    if planner_name == "spacetime_astar":
        planner = SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0)
    else:
        planner = StaticDijkstraPlanner(max_speed=1.0, dt=1.0)
    sim = Simulator(
        net, tasks, agvs, planner, GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=max_steps),
    )
    return sim.run()


def test_all_tasks_complete():
    r = _run()
    assert r.summary()["tasks_completed"] == 12


def test_makespan_positive():
    r = _run()
    assert r.makespan_seconds() > 0


def test_on_time_rate_in_range():
    r = _run()
    v = r.on_time_rate()
    assert 0.0 <= v <= 1.0
    assert not math.isnan(v)


def test_agv_utilization_in_range():
    r = _run()
    v = r.agv_utilization()
    assert 0.0 <= v <= 1.0


def test_no_unexpected_collisions():
    """执行器的独立复核不应发现超容占用。"""
    r = _run()
    assert r.collision_events == 0


def test_determinism_same_seed():
    a = _run(seed=7)
    b = _run(seed=7)
    assert a.makespan_seconds() == b.makespan_seconds()
    assert a.summary()["conflict_wait_steps"] == b.summary()["conflict_wait_steps"]


def test_different_seed_may_differ():
    a = _run(seed=1)
    b = _run(seed=2)
    # 不保证一定不同，但至少都要完成
    assert a.summary()["tasks_completed"] == b.summary()["tasks_completed"] == 12


def test_empty_task_list_terminates():
    net = build_demo_network()
    from amr_sched.entities import Agv
    agv = Agv("a0", start_node="B1")
    sim = Simulator(
        net, [], [agv], SpaceTimeAStarPlanner(), GreedyAllocator(),
        SimulationConfig(max_steps=100),
    )
    r = sim.run()
    assert r.finished_step == 0


def test_urgent_task_gets_priority():
    """加急任务应先于普通任务被分配。"""
    net = build_demo_network()
    from amr_sched.entities import Agv
    normal = Task("N1", "D", "W1", release_step=0, due_step=10_000)
    urgent = Task("U1", "D", "W2", release_step=0, due_step=10_000, priority=TaskPriority.URGENT)
    agv = Agv("a0", start_node="B1")
    sim = Simulator(
        net, [normal, urgent], [agv], SpaceTimeAStarPlanner(), GreedyAllocator(),
        SimulationConfig(max_steps=20_000),
    )
    r = sim.run()
    assert urgent.assigned_step <= normal.assigned_step
