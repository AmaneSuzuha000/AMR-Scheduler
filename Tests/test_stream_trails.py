"""实际轨迹：控制台要能看到车到底走过哪里，而不只是现在在哪。

这组测试钉住三件事：
1. 轨迹点的坐标与控制台投影用的是同一套（场景米），否则线会画到地图外面；
2. 轨迹会随行驶增长，但**有上限**——20 Hz 跑一小时不能把内存吃光；
3. 轨迹是生产侧算的，显示层只拿现成点，不自己插值。
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from amr_sched.live import build_sim  # noqa: E402
from amr_sched.stream import FrameBuilder  # noqa: E402


def _builder(n_tasks=8, n_agvs=3, seed=0):
    sim = build_sim(n_tasks, n_agvs, seed, layout="workshop")
    sim.reset()
    return sim, FrameBuilder(sim)


def test_every_vehicle_gets_a_trail_from_the_first_frame():
    sim, builder = _builder()
    sim.step_once()
    frame = builder.frame()
    assert set(frame["trails"]) == {a.agv_id for a in sim.agvs}
    for agv_id, points in frame["trails"].items():
        assert len(points) == 1, "%s 起步就应当有一个点" % agv_id


def test_trail_points_are_scene_metres_not_node_names():
    """点必须是场景米，和控制台投影同源；写成节点名就没法画。"""
    sim, builder = _builder()
    sim.step_once()
    frame = builder.frame()
    node = sim.network.node(sim.agvs[0].cur_node)
    first = frame["trails"][sim.agvs[0].agv_id][0]
    assert first == [node.x, node.y]


def test_trail_grows_while_the_vehicle_drives():
    sim, builder = _builder()
    for _ in range(200):
        sim.step_once()
        frame = builder.frame()
    grown = [k for k, v in frame["trails"].items() if len(v) >= 2]
    assert grown, "跑 200 步后至少应有一辆车走过不止一个位置"


def test_trail_is_bounded():
    """长跑不能无限增长。"""
    sim, builder = _builder()
    builder.TRAIL_POINTS = 20
    for _ in range(400):
        sim.step_once()
        frame = builder.frame()
    assert all(len(v) <= 20 for v in frame["trails"].values())


def test_trail_survives_a_json_round_trip():
    sim, builder = _builder()
    for _ in range(30):
        sim.step_once()
        frame = builder.frame()
    back = json.loads(json.dumps(frame))
    assert set(back["trails"]) == set(frame["trails"])
    assert back["trails"] == frame["trails"]


def test_each_vehicle_keeps_its_own_trail():
    sim, builder = _builder()
    for _ in range(60):
        sim.step_once()
        frame = builder.frame()
    a, b = sim.agvs[0].agv_id, sim.agvs[1].agv_id
    assert frame["trails"][a] is not frame["trails"][b]
