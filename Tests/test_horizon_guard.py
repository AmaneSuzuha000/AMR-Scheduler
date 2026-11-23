"""回归测试：不允许把车 commit 到仿真 horizon 之后。

背景
----
``_plan_and_reserve`` 里有一条校验：计划到达步 ≥ ``max_steps`` 就拒绝。
但它用的是**规划时刻**的 ``start_step``。车被阻塞后会在更晚的 step 才出发，
``_commit_move`` 用实际 step 重算到达时刻，于是越过 horizon，
撞上 ``_commit_move`` 的兜底断言直接抛 RuntimeError。

实测触发场景：hub 布局、100 任务、6 台车、``max_steps=120000``。
规划时 arrive=119980 合规，实际出发后是 120072，整个实验脚本崩溃。

这个 bug 在默认布局下永远测不出来——只有 hub 这种"中心必经、
高负载、大量未完成算例"的拓扑才会把车压到 horizon 边界。
"""

from __future__ import annotations

import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.layouts import build_layout
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator


def _run(layout: str, n_tasks: int, n_agvs: int, seed: int, max_steps: int):
    net = build_layout(layout, aisle_capacity=2)
    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed)
    net, tasks, agvs = generate_scenario(sc, network=net)
    sim = Simulator(
        net, tasks, agvs,
        StaticDijkstraPlanner(max_speed=sc.max_speed, dt=sc.dt),
        GreedyAllocator(),
        SimulationConfig(dt=sc.dt, max_steps=max_steps),
    )
    return sim.run()


def test_hub_high_load_does_not_overshoot_horizon():
    """hub 布局压到 horizon 边界时不应抛异常。

    这是「规划时合规、执行时越界」的最小复现：
    修复前必定抛 ``RuntimeError: commit_move 的到达步 ... 超过 max_steps``。
    """
    result = _run("hub", 100, 6, 0, 120_000)
    assert result.finished_step <= 120_000
    # 任务没做完是这个算例的固有结局（hub 中心车道饱和），不是 bug；
    # 这里只要求「不崩、不越界」。
    assert result.summary()["tasks_completed"] > 0


def test_no_task_completes_after_horizon():
    """任何任务的完成时刻都不得晚于 max_steps。"""
    result = _run("hub", 100, 6, 0, 120_000)
    for t in result.tasks:
        if t.complete_step is not None:
            assert t.complete_step <= 120_000


def test_makespan_never_exceeds_horizon():
    """makespan 取的是已完成任务的最大完成步，自然不能越界。

    这条看似显然，但修复前根本走不到这里——进程先崩了。
    """
    result = _run("hub", 50, 4, 1, 60_000)
    ms = result.summary()["makespan_seconds"]
    assert ms != ms or ms <= 60_000  # NaN 表示一个都没完成
