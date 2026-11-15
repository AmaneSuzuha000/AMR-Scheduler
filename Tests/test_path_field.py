"""路径字段（`AgvFrame.path`）的行为测试。

为什么单独测这个字段
---------------------
它此前**恒为空**：`AgvFrame` 定义了 `path`，`to_dict()` 也输出了它，
但仿真器从来没往里填过值。这类"字段存在但永远为空"的坑最隐蔽——
消费端拿到 `[]` 会以为"车没有路径"，而真相是"没人填"。
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from amr_sched.allocation.greedy import GreedyAllocator  # noqa: E402
from amr_sched.layouts import build_layout  # noqa: E402
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner  # noqa: E402
from amr_sched.scenarios import ScenarioConfig, generate_scenario  # noqa: E402
from amr_sched.simulation import SimulationConfig, Simulator  # noqa: E402
from amr_sched.stream import frames_from_result  # noqa: E402


def _run(n_tasks=20, n_agvs=3, seed=0):
    net = build_layout("ring", aisle_capacity=2)
    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed,
                        dt=1.0, max_speed=1.0, urgent_ratio=0.0)
    _, tasks, agvs = generate_scenario(sc, network=net)
    sim = Simulator(net, tasks, agvs,
                    StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(),
                    SimulationConfig(dt=1.0, max_steps=200_000, seed=seed,
                                     record_trajectory=True))
    return sim.run(), net


def test_path_is_actually_populated():
    """路径不能恒为空——这正是修好之前的症状。"""
    res, _ = _run()
    frames = frames_from_result(res, stride=1)
    with_path = sum(1 for f in frames for a in f.agvs if a.path)
    assert with_path > 0, "所有帧的 path 都是空的：填充逻辑没生效"
    total = sum(len(f.agvs) for f in frames)
    assert with_path / total > 0.3, \
        "只有 %.1f%% 的(帧,车)有路径，低于合理水平" % (100 * with_path / total)


def test_idle_agv_has_empty_path():
    """空闲车没有路径是正确的，不能被当成缺失。"""
    res, _ = _run()
    frames = frames_from_result(res, stride=1)
    idle_with_path = [a for f in frames for a in f.agvs
                      if a.state == "idle" and a.path]
    assert not idle_with_path, \
        "空闲车不该有剩余路径，实际有 %d 例" % len(idle_with_path)


def test_path_nodes_are_real_nodes():
    """路径里的节点必须真实存在，且不应包含当前所在节点。"""
    res, net = _run()
    frames = frames_from_result(res, stride=20)
    known = set(net.nodes)
    for f in frames:
        for a in f.agvs:
            for node in a.path:
                assert node in known, "路径里出现未知节点 %r" % node
            if a.pose.node and a.path:
                assert a.path[0] != a.pose.node, \
                    "路径首节点与当前节点相同，说明 idx 取错了一位"


def test_path_is_json_serialisable():
    """导出的 JSON 里必须真有这个字段，不能只有内存里有。"""
    res, net = _run()
    frames = frames_from_result(res, stride=50)
    d = frames[0].to_dict()
    assert "path" in d["agvs"][0]
    assert isinstance(d["agvs"][0]["path"], list)
