"""物理车真值：Gazebo 的那台车必须以**独立实体**出现在帧里。

这里钉住的是一条架构约束，不是普通功能：同一辆车不能同时有两个位置。
曾经的做法是把仿真车的位姿"换成"真值——引擎仍在推进这台车、Gazebo 说它原地没动，
于是同一辆 AGV 有了两个互相矛盾的坐标。这组测试就是防止那种写法回来。
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from amr_sched.live import build_sim  # noqa: E402
from amr_sched.stream import FrameBuilder  # noqa: E402


def _builder(n_tasks=6, n_agvs=2, seed=0):
    sim = build_sim(n_tasks, n_agvs, seed, layout="workshop")
    sim.reset()
    return sim, FrameBuilder(sim)


def test_no_physical_block_when_no_robot_is_connected():
    sim, builder = _builder()
    sim.step_once()
    frame = builder.frame()
    assert "physical" not in frame


def test_physical_entity_is_tagged_gazebo():
    sim, builder = _builder()
    builder.set_physical(12.0, 55.0, 0.0)
    sim.step_once()
    physical = builder.frame()["physical"]
    assert physical["source"] == "gazebo"
    assert physical["id"] == "scout_mini"
    assert physical["pose"][:2] == [12.0, 55.0]


def test_fleet_trails_stay_sim_while_a_physical_robot_reports():
    """关键回归：物理车出现后，仿真车的位姿来源不能被悄悄改写。"""
    sim, builder = _builder()
    for i in range(5):
        builder.set_physical(10.0 + i, 50.0, 0.0)
        sim.step_once()
        frame = builder.frame()
    assert set(frame["trail_sources"].values()) == {"sim"}
    for agv_id, points in frame["trails"].items():
        assert points, "%s 的轨迹不该被物理车顶掉" % agv_id


def test_repeated_identical_poses_do_not_pad_the_trail():
    """静止的车每帧都报同一个坐标，轨迹不该被同一个点撑满 300 格。"""
    sim, builder = _builder()
    for _ in range(20):
        builder.set_physical(12.0, 55.0, 0.0)
    sim.step_once()
    assert len(builder.frame()["physical"]["trail"]) == 1


def test_moving_physical_robot_builds_a_trail():
    sim, builder = _builder()
    for i in range(10):
        builder.set_physical(12.0 + i * 0.5, 55.0, 0.0)
    sim.step_once()
    trail = builder.frame()["physical"]["trail"]
    assert len(trail) == 10
    assert trail[0] == [12.0, 55.0]
    assert trail[-1] == [16.5, 55.0]


def test_losing_the_feed_clears_the_pose_but_keeps_the_history():
    sim, builder = _builder()
    for i in range(4):
        builder.set_physical(12.0 + i, 55.0, 0.0)
    builder.clear_physical()
    sim.step_once()
    physical = builder.frame()["physical"]
    assert physical["pose"] is None, "断流后不能再报一个位置"
    assert len(physical["trail"]) == 4, "已经走过的路是事实，不该被抹掉"


def test_physical_trail_is_bounded():
    sim, builder = _builder()
    builder.TRAIL_POINTS = 5
    for i in range(40):
        builder.set_physical(12.0 + i * 0.1, 55.0, 0.0)
    sim.step_once()
    assert len(builder.frame()["physical"]["trail"]) <= 5


def test_physical_block_survives_a_json_round_trip():
    sim, builder = _builder()
    builder.set_physical(12.0, 55.0, 0.3)
    sim.step_once()
    frame = builder.frame()
    back = json.loads(json.dumps(frame))
    assert back["physical"] == frame["physical"]
