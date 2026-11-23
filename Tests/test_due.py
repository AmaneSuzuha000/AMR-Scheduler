"""截止时间口径的测试。"""

from __future__ import annotations

import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.scenarios import DueModel, ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator


def _leads(model: DueModel, **kw) -> list[int]:
    kw.setdefault("n_agvs", 4)
    sc = ScenarioConfig(n_tasks=40, seed=2, due_model=model, **kw)
    _net, tasks, _agvs = generate_scenario(sc)
    return [t.due_step - t.release_step for t in tasks]


def test_travel_only_ignores_queue():
    """仅计行驶时间的口径：容差只和路程有关，换车队规模不应改变。"""
    a = _leads(DueModel.TRAVEL_ONLY, n_agvs=2)
    b = _leads(DueModel.TRAVEL_ONLY, n_agvs=6)
    assert a == b, "仅计行驶时间的口径不该随车队规模变化"


def test_travel_queue_grows_with_fewer_agvs():
    """含排队口径：车越少，每台车分摊的任务越多，容差必须更长。"""
    many = _leads(DueModel.TRAVEL_QUEUE, n_agvs=6)
    few = _leads(DueModel.TRAVEL_QUEUE, n_agvs=2)
    assert sum(few) > sum(many), "2 台车应当比 6 台车需要更长的交货期"


def test_travel_queue_is_longer_than_travel_only():
    """含排队口径必须比仅计行驶时间更宽松（否则排队项没起作用）。"""
    only = _leads(DueModel.TRAVEL_ONLY)
    queue = _leads(DueModel.TRAVEL_QUEUE)
    assert sum(queue) > sum(only)


def test_fixed_ignores_task_geometry():
    """固定容差口径下，所有任务的容差必须完全相同。"""
    leads = _leads(DueModel.FIXED)
    assert len(set(leads)) == 1, "固定口径下容差应当一致"


def test_all_models_respect_slack():
    """slack 变大，容差必须单调变长（三个口径都要成立）。"""
    for model in DueModel:
        small = sum(_leads(model, deadline_slack=1.0))
        large = sum(_leads(model, deadline_slack=5.0))
        assert large > small, "%s 口径下 slack 未生效" % model.value


def test_due_after_release():
    """截止步必须不早于下达步（Task 自身会校验，这里守一道）。"""
    for model in DueModel:
        sc = ScenarioConfig(n_tasks=30, n_agvs=3, seed=9, due_model=model)
        _net, tasks, _agvs = generate_scenario(sc)
        assert all(t.due_step >= t.release_step for t in tasks)


def test_rate_against_due_separates_metric_from_formula():
    """同一批调度结果，换口径重判准时率必须能得到不同数值。

    这是整个敏感性实验的核心机制：不重跑仿真，只换截止公式。
    若两个口径给出完全相同的结果，说明函数没有真正按传入的 due 判定。
    """
    sc = ScenarioConfig(n_tasks=40, n_agvs=3, seed=4)
    net, tasks, agvs = generate_scenario(sc)
    r = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(1.0, 1.0),
                  GreedyAllocator(), SimulationConfig(max_steps=100_000)).run()

    tight = {t.task_id: t.release_step + 200 for t in r.tasks}
    loose = {t.task_id: t.release_step + 200_000 for t in r.tasks}
    tight_rate = r.rate_against_due(tight)["on_time_rate"]
    loose_rate = r.rate_against_due(loose)["on_time_rate"]

    assert loose_rate == 1.0, "极宽松的截止期下应当全部准时"
    assert tight_rate < loose_rate, "收紧截止期必须降低准时率"


def test_rate_against_due_counts_only_completed():
    """分母是已完成任务数，未完成的任务不参与判定。"""
    sc = ScenarioConfig(n_tasks=20, n_agvs=2, seed=6)
    net, tasks, agvs = generate_scenario(sc)
    r = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(1.0, 1.0),
                  GreedyAllocator(), SimulationConfig(max_steps=100_000)).run()
    dues = {t.task_id: t.release_step + 10**6 for t in r.tasks}
    got = r.rate_against_due(dues)
    assert got["evaluated"] == r.summary()["tasks_completed"]
