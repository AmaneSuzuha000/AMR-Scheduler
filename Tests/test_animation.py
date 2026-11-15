"""调度过程监控可视化的测试（R4.4 / R4.5）。"""

from __future__ import annotations

import os

import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.viz.animation import (
    STATE_COLORS,
    agv_state_timeline,
    render_animation,
    task_progress_curve,
)


@pytest.fixture(scope="module")
def traced():
    """跑一次带轨迹的仿真，供本模块所有用例复用。"""
    sc = ScenarioConfig(n_tasks=15, n_agvs=2, seed=0)
    net, tasks, agvs = generate_scenario(sc)
    r = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(1.0, 1.0),
                  GreedyAllocator(),
                  SimulationConfig(max_steps=100_000,
                                   record_trajectory=True)).run()
    return net, r


def test_trajectory_records_states_not_just_positions(traced):
    """轨迹必须记录 AGV 状态，否则做不了 R4.4 的状态监控。"""
    _net, r = traced
    assert r.trajectory, "轨迹为空"
    rec = r.trajectory[len(r.trajectory) // 2]
    assert rec.agv_positions, "缺少位置"
    assert rec.agv_states, "缺少状态"
    assert set(rec.agv_states) == set(rec.agv_positions)
    for st in rec.agv_states.values():
        assert st in STATE_COLORS, "出现未定义的状态名：%s" % st


def test_task_counts_sum_to_total(traced):
    """每一步的四类任务计数之和必须等于任务总数（不漏不重）。"""
    _net, r = traced
    n = len(r.tasks)
    for rec in r.trajectory:
        total = sum(rec.task_counts.values())
        assert total == n, "步 %d 任务计数和为 %d，应为 %d" % (rec.step, total, n)


def test_completed_count_is_monotonic(traced):
    """已完成任务数只能增加，不能回退。"""
    _net, r = traced
    prev = -1
    for rec in r.trajectory:
        cur = rec.task_counts.get("completed", 0)
        assert cur >= prev, "步 %d 完成数从 %d 回退到 %d" % (rec.step, prev, cur)
        prev = cur


def test_final_completed_matches_summary(traced):
    """轨迹末帧的完成数必须与 summary 一致（两处口径不能打架）。"""
    _net, r = traced
    last = r.trajectory[-1].task_counts["completed"]
    assert last == r.summary()["tasks_completed"]


def test_trajectory_empty_when_disabled():
    """未开启 record_trajectory 时不应产生轨迹（避免无谓内存开销）。"""
    sc = ScenarioConfig(n_tasks=10, n_agvs=2, seed=1)
    net, tasks, agvs = generate_scenario(sc)
    r = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(1.0, 1.0),
                  GreedyAllocator(),
                  SimulationConfig(max_steps=50_000)).run()
    assert r.trajectory == []


def test_state_timeline_writes_file(traced, tmp_path):
    _net, r = traced
    out = str(tmp_path / "states.png")
    agv_state_timeline(r, out)
    assert os.path.getsize(out) > 3000


def test_task_progress_curve_writes_file(traced, tmp_path):
    _net, r = traced
    out = str(tmp_path / "progress.png")
    task_progress_curve(r, out)
    assert os.path.getsize(out) > 3000


def test_render_animation_writes_gif(traced, tmp_path):
    net, r = traced
    out = str(tmp_path / "anim.gif")
    render_animation(net, r, out, frame_every=50, max_frames=40)
    assert os.path.getsize(out) > 5000
    from PIL import Image
    with Image.open(out) as im:
        assert im.n_frames >= 2


def test_render_animation_rejects_empty_trajectory():
    sc = ScenarioConfig(n_tasks=5, n_agvs=1, seed=0)
    net, tasks, agvs = generate_scenario(sc)
    r = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(1.0, 1.0),
                  GreedyAllocator(),
                  SimulationConfig(max_steps=50_000)).run()
    assert r.trajectory == []
    with pytest.raises(ValueError, match="轨迹为空"):
        render_animation(net, r, "/tmp/never.gif")
