"""统一数据接口（``amr_sched.stream``）的契约测试。

这一层存在的理由是：Python 调度核心与 C# 可视化界面之间
此前**没有数据接口**，两边各跑各的，构不成一个系统。
接口一旦松动，那两套成果又会退回去彼此分离。
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.layouts import build_layout
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.stream import (
    SCHEMA_VERSION,
    NetworkSpec,
    Pose,
    PoseSource,
    export_stream,
    frames_from_result,
)


def _run(n_tasks=20, n_agvs=3, seed=0, record=True):
    net = build_layout("ring", aisle_capacity=2)
    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed,
                        dt=1.0, max_speed=1.0, urgent_ratio=0.15)
    _, tasks, agvs = generate_scenario(sc, network=net)
    sim = Simulator(net, tasks, agvs,
                    StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(),
                    SimulationConfig(dt=1.0, max_steps=200_000, seed=seed,
                                     record_trajectory=record))
    return sim.run(), net


def test_pose_requires_explicit_source():
    """位姿必须带来源；默认是 SIM，不允许"无来源"的位姿存在。"""
    p = Pose(agv_id="A0", node="B1")
    assert p.source is PoseSource.SIM
    assert p.to_dict()["source"] == "sim"


def test_odom_is_distinguishable_from_sim():
    """里程计位姿与仿真真值必须能区分开。

    混用二者会让「实车与仿真一致性」无法讨论，
    也会把里程计漂移误当成定位误差。
    """
    sim_p = Pose(agv_id="A0", node="B1", source=PoseSource.SIM)
    odom_p = Pose(agv_id="A0", node="B1", source=PoseSource.ODOM)
    loc_p = Pose(agv_id="A0", node="B1", source=PoseSource.LOCALIZED)
    assert sim_p.to_dict()["source"] != odom_p.to_dict()["source"]
    assert odom_p.to_dict()["source"] != loc_p.to_dict()["source"]


def test_frames_are_self_contained():
    """任一帧都能脱离前后帧单独解读。"""
    result, _ = _run()
    frames = frames_from_result(result, stride=10)
    assert frames, "应当产生帧"
    mid = frames[len(frames) // 2]
    d = mid.to_dict()
    # 自包含：步号、时间、每台车的位姿与状态、任务计数都在
    assert "step" in d and "t" in d
    assert d["agvs"], "帧里应当有车"
    for a in d["agvs"]:
        assert a["pose"]["node"] or a["pose"]["edge"], "位姿既不在节点也不在边"
        assert a["state"], "状态缺失"
    assert "task_counts" in d


def test_stride_reduces_frames():
    """抽帧必须真的减少帧数，且保持时间顺序。"""
    result, _ = _run()
    f1 = frames_from_result(result, stride=1)
    f5 = frames_from_result(result, stride=5)
    assert len(f5) < len(f1)
    steps = [f.step for f in f5]
    assert steps == sorted(steps)
    assert all(s % 5 == 0 for s in steps)


def test_export_without_trajectory_fails_loudly():
    """忘了开 record_trajectory 时必须报错，不能静默导出空流。"""
    result, net = _run(record=False)
    out = os.path.join(tempfile.gettempdir(), "should_not_exist.json")
    with pytest.raises(ValueError, match="record_trajectory"):
        export_stream(result, out, network=net)
    assert not os.path.exists(out)


def test_export_roundtrip():
    """导出的 JSON 必须能被独立读回，且结构与版本号正确。"""
    result, net = _run()
    out = os.path.join(tempfile.gettempdir(), "stream_rt.json")
    info = export_stream(result, out, dt=1.0, stride=5, network=net,
                         meta={"scenario": "ring/20/3/seed0"})
    try:
        assert info["n_frames"] > 0
        assert info["bytes"] > 0
        assert info["schema"] == SCHEMA_VERSION
        data = json.load(open(out, encoding="utf-8"))
        assert data["schema"] == SCHEMA_VERSION
        assert data["meta"]["scenario"] == "ring/20/3/seed0"
        assert len(data["frames"]) == info["n_frames"]
        assert data["network"] is not None
        assert len(data["network"]["nodes"]) == len(net.nodes)
    finally:
        if os.path.exists(out):
            os.remove(out)


def test_network_spec_preserves_capacity():
    """容量信息必须带出去——可视化端的拥堵着色全靠它。"""
    net = build_layout("ring", aisle_capacity=2)
    spec = NetworkSpec.from_network(net)
    for n in spec.nodes:
        assert n["capacity"] >= 1
        assert n["kind"]
    assert len(spec.edges) == len(net.edges)


def test_all_poses_in_export_are_sim():
    """仿真导出的每一帧每一个位姿都必须是 SIM，不许混入别的来源。"""
    result, net = _run()
    out = os.path.join(tempfile.gettempdir(), "stream_src.json")
    export_stream(result, out, stride=20, network=net)
    try:
        data = json.load(open(out, encoding="utf-8"))
        sources = {
            a["pose"]["source"]
            for f in data["frames"] for a in f["agvs"]
        }
        assert sources == {"sim"}, f"混入了非仿真来源: {sources}"
    finally:
        if os.path.exists(out):
            os.remove(out)
