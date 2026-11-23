"""工位加工时间建模（工位加工时间与服务驻留联合建模）。

原来的模型是「卸货即完成」，工位卸完就空闲，**低估了工位排队**。
本文件守住三点：
1. process_time=0 时行为与旧版完全一致（向后兼容，历史数据可比）；
2. process_time>0 时工位确实被占住，且**只有工位/组装区**，缓存区不算；
3. AGV 不陪等——加工期间 AGV 可以去干别的活。
"""

from __future__ import annotations

import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.network import NodeKind, build_demo_network
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import Simulator, SimulationConfig


def _run(n_tasks=20, n_agvs=2, seed=0, process_time=0.0, max_steps=60000):
    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed,
                        dt=1.0, max_speed=1.0)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(net, tasks, agvs,
                    StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(),
                    SimulationConfig(dt=1.0, max_steps=max_steps, seed=seed,
                                     process_time=process_time))
    return sim, sim.run()


def test_zero_process_time_is_backward_compatible():
    """process_time=0 时，结果必须与不传该参数完全相同。"""
    _, a = _run(process_time=0.0)
    sc = ScenarioConfig(n_tasks=20, n_agvs=2, seed=0, dt=1.0, max_speed=1.0)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(net, tasks, agvs,
                    StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(),
                    SimulationConfig(dt=1.0, max_steps=60000, seed=0))
    b = sim.run()
    assert a.summary()["tasks_completed"] == b.summary()["tasks_completed"]
    assert a.summary()["finished_step"] == b.summary()["finished_step"]
    assert a.summary()["makespan_seconds"] == b.summary()["makespan_seconds"]


def test_process_time_creates_holds():
    """process_time>0 时必须真的产生工位占用记录。"""
    sim, _ = _run(process_time=120.0)
    total = sum(len(v) for v in sim._process_holds.values())
    assert total > 0, "设了加工时间却没有占用记录"


def test_zero_process_time_creates_no_holds():
    """process_time=0 时不应有任何占用记录。"""
    sim, _ = _run(process_time=0.0)
    assert not sim._process_holds, "未设加工时间却产生了占用"


def test_holds_only_on_workstations_and_assembly():
    """缓存区不应有加工占用——缓存是中转暂存，卸下即走。"""
    sim, result = _run(n_tasks=30, n_agvs=3, process_time=120.0)
    for node_id in sim._process_holds:
        kind = sim.network.node(node_id).kind
        assert kind in (NodeKind.WORKSTATION, NodeKind.ASSEMBLY), (
            "节点 %s 是 %s，不该有加工占用" % (node_id, kind))


def test_process_time_increases_system_load():
    """加工时间越长，完成时间应越长（同种子、同配置）。"""
    _, a = _run(n_tasks=30, n_agvs=2, seed=1, process_time=0.0)
    _, b = _run(n_tasks=30, n_agvs=2, seed=1, process_time=300.0)
    assert b.summary()["finished_step"] > a.summary()["finished_step"], (
        "加了加工时间反而更快，说明占用没生效: %d -> %d"
        % (a.summary()["finished_step"], b.summary()["finished_step"]))


def test_agv_does_not_wait_during_processing():
    """AGV 不陪等：加工期间它应继续执行其他任务，而非长时间空闲。"""
    sim, result = _run(n_tasks=40, n_agvs=3, seed=2, process_time=300.0,
                       max_steps=120000)
    # 若 AGV 陪等，空闲步数会随加工时间线性膨胀
    idle = sum(a.idle_steps for a in sim.agvs)
    finished = result.summary()["finished_step"]
    assert idle < finished * len(sim.agvs) * 0.5, (
        "空闲占比过高（%.1f%%），疑似 AGV 在陪等加工"
        % (100.0 * idle / (finished * len(sim.agvs))))


def test_has_processing_by_node_kind():
    """_has_processing 只对工位与组装区返回真。"""
    sc = ScenarioConfig(n_tasks=2, n_agvs=1, seed=0, dt=1.0, max_speed=1.0)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(net, tasks, agvs,
                    StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(),
                    SimulationConfig(dt=1.0, max_steps=1000, seed=0,
                                     process_time=60.0))
    assert sim._has_processing("W1")
    assert sim._has_processing("S")
    assert not sim._has_processing("B1")
    assert not sim._has_processing("R0")


def test_process_steps_rounding():
    """加工步数向上取整，且 0 时不返回 1（不能把「关闭」变成「1 步」）。"""
    sc = ScenarioConfig(n_tasks=2, n_agvs=1, seed=0, dt=1.0, max_speed=1.0)
    net, tasks, agvs = generate_scenario(sc)
    for pt, expect in [(0.0, 0), (0.4, 1), (1.0, 1), (1.5, 2), (60.0, 60)]:
        sim = Simulator(net, tasks, agvs,
                        StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                        GreedyAllocator(),
                        SimulationConfig(dt=1.0, max_steps=1000, seed=0,
                                         process_time=pt))
        assert sim._process_steps() == expect, (
            "process_time=%.1f 应折算 %d 步" % (pt, expect))


def test_stale_holds_are_cleaned():
    """过期的加工占用会被清理，不会让工位永久锁死。"""
    sim, result = _run(n_tasks=30, n_agvs=3, seed=3, process_time=120.0,
                       max_steps=120000)
    assert result.summary()["tasks_completed"] == 30, (
        "有加工时间就跑不完，说明工位被永久锁死")
    # 运行结束后，所有占用都应已过期或即将过期
    last = result.summary()["finished_step"]
    for node_id, holds in sim._process_holds.items():
        for begin, end, _aid in holds:
            assert end <= last + 200, (
                "节点 %s 的占用延续到 %d，远超结束步 %d"
                % (node_id, end, last))