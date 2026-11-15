"""调度过程可视化测试（任务书 R4.5）。"""

import os

import pytest

pytest.importorskip("matplotlib")

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.network import build_demo_network
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.viz import (corridor_load, plot_corridor_heatmap, plot_gantt,
                           plot_layout, plot_trajectories)


@pytest.fixture(scope="module")
def run():
    sc = ScenarioConfig(n_tasks=12, n_agvs=2, seed=0)
    net, tasks, agvs = generate_scenario(sc)
    res = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(1.0, 1.0),
                    GreedyAllocator(),
                    SimulationConfig(max_steps=100_000,
                                     record_trajectory=True)).run()
    return net, tasks, agvs, res


def test_plot_layout_writes_file(tmp_path, run):
    net, _t, agvs, _r = run
    p = tmp_path / "layout.png"
    plot_layout(net, agvs, out_path=str(p))
    assert p.exists() and p.stat().st_size > 5000


def test_plot_gantt_writes_file(tmp_path, run):
    _n, _t, _a, res = run
    p = tmp_path / "gantt.png"
    plot_gantt(res, out_path=str(p))
    assert p.exists() and p.stat().st_size > 5000


def test_plot_trajectories_requires_trajectory(tmp_path, run):
    net, tasks, agvs, _r = run
    res_no_traj = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(1.0, 1.0),
                            GreedyAllocator(),
                            SimulationConfig(max_steps=100_000,
                                             record_trajectory=False)).run()
    with pytest.raises(ValueError, match="轨迹为空"):
        plot_trajectories(net, res_no_traj)


def test_plot_trajectories_writes_file(tmp_path, run):
    net, _t, _a, res = run
    p = tmp_path / "traj.png"
    plot_trajectories(net, res, out_path=str(p))
    assert p.exists() and p.stat().st_size > 5000


def test_corridor_load_range(run):
    net, _t, _a, res = run
    load = corridor_load(net, res)
    assert load, "应当有通道"
    for key, info in load.items():
        assert key[0] < key[1], "通道必须归一为 u<v"
        assert 0.0 <= info["load"] <= 1.0, (key, info)
        assert info["capacity"] >= 1


def test_corridor_load_symmetric_keys(run):
    """(u,v) 与 (v,u) 必须是同一条通道，不能重复统计。"""
    net, _t, _a, res = run
    load = corridor_load(net, res)
    total_edges = len(net.edges)
    undirected = {(min(e.src, e.dst), max(e.src, e.dst)) for e in net.edges.values()}
    assert len(load) == len(undirected)
    assert len(load) < total_edges, "双向边应当合并为一条物理通道"


def test_plot_corridor_heatmap_writes_file(tmp_path, run):
    net, _t, _a, res = run
    p = tmp_path / "corr.png"
    plot_corridor_heatmap(net, res, out_path=str(p))
    assert p.exists() and p.stat().st_size > 5000
