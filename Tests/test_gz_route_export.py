"""The route file the vehicle drives must be the route the scheduler planned.

This is the claim the whole Gazebo link rests on, and until now it was not enforced: the
runner consumed a routes.json that had been typed by hand, so a divergence between the
vehicle's path and the scheduler's path would have been invisible. The coordinates happened
to match to 0.0000 m, but nothing checked that.

These tests check it. They also pin the coordinate scaling, because a wrong scale factor
would move every waypoint while leaving the route names and node sequences looking right.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "live_bridge"))

from amr_sched.network import build_demo_network            # noqa: E402
import gz_export_routes as ex                              # noqa: E402


def test_scale_is_the_documented_aisle_pitch_ratio():
    """The scheduler models 40 m between aisles, the Gazebo world 1.6 m."""
    net = build_demo_network()
    pos = ex.scaled_positions(net)
    # R0 -> R1 is one aisle pitch in the scheduler.
    spacing_sched = 40.0
    spacing_gazebo = abs(pos["R1"][0] - pos["R0"][0])
    assert spacing_gazebo == pytest.approx(spacing_sched * ex.SCALE)
    assert spacing_gazebo == pytest.approx(1.6)


def test_the_origin_node_lands_at_the_world_origin():
    """The network must be placed relative to a named node, not to raw coordinates."""
    pos = ex.scaled_positions(build_demo_network())
    assert pos[ex.ORIGIN_NODE] == pytest.approx((0.0, 0.0))


def test_every_demo_node_has_a_position():
    net = build_demo_network()
    pos = ex.scaled_positions(net)
    assert set(pos) == set(net.nodes)


def test_waypoints_come_from_the_network_not_from_the_route_list():
    """Rewriting a node's coordinates must move the waypoint that refers to it.

    If the route table carried its own coordinates this test would fail -- which is exactly
    the failure that went unnoticed while routes.json was handwritten.
    """
    net = build_demo_network()
    before = ex.build_routes(net)["R0->B1"]["waypoints"]

    # Node is a frozen dataclass, so it is replaced rather than mutated -- which also makes
    # the independence being tested explicit.
    net.nodes["B1"] = dataclasses.replace(net.nodes["B1"], x=net.nodes["B1"].x + 25.0)
    after = ex.build_routes(net)["R0->B1"]["waypoints"]

    assert before[-1] != after[-1]
    assert after[-1][0] - before[-1][0] == pytest.approx(25.0 * ex.SCALE)
    assert after[-1][1] == pytest.approx(before[-1][1])


def test_a_path_through_an_unknown_node_is_rejected():
    """Silently dropping a waypoint would produce a route that looks fine and is not."""
    with pytest.raises(KeyError):
        ex.path_waypoints(["R0", "NOT_A_NODE"], ex.scaled_positions(build_demo_network()))


def test_route_names_match_their_node_sequences():
    """The label and the path must not drift apart."""
    routes = ex.build_routes()
    for name, spec in routes.items():
        src, dst = name.split("->")
        assert spec["nodes"][0] == src
        assert spec["nodes"][-1] == dst


def test_waypoint_count_matches_node_count():
    routes = ex.build_routes()
    for name, spec in routes.items():
        assert len(spec["waypoints"]) == len(spec["nodes"]), name


def test_the_end_to_end_script_covers_all_three_joins():
    """Stages that pass individually can still fail at the joins between them.

    The script must export from the network, check the file still describes that network,
    and then drive it -- not merely do the first of those and report success.
    """
    path = os.path.join(ROOT, "tools", "live_bridge", "verify_scheduler_to_gazebo.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert "def stage_export" in src
    assert "def stage_plan_vs_file" in src
    assert "def stage_drive" in src
    assert "routes failed: 0/" in src, (
        "the drive stage must require every route to complete, not just report")


def test_the_exported_file_round_trips():
    """The follower reads JSON; the exporter must produce something it can read."""
    doc = ex.build_routes()
    text = json.dumps(doc)
    back = json.loads(text)
    assert back == doc
    for name, spec in back.items():
        assert isinstance(spec["nodes"], list)
        assert all(len(w) == 2 for w in spec["waypoints"])


def test_scene_export_uses_the_same_one_to_25_scale(tmp_path):
    """场景导出必须和 gz_frame 用同一个比例，否则世界里的地面和路线会对不上。

    这条守的是 MAP-01E 的前提：车间场景 -> Gazebo 世界只能靠一个比例换算，
    任何一方偷偷改了 scale，另一方的坐标就全错位，而且看起来"有地图"。
    """
    import gz_export_scene as scene_ex
    from amr_sched.workshop import WorkshopScene

    scene = WorkshopScene.load()
    out_world = tmp_path / "w.world"
    out_route = tmp_path / "r.json"
    stats = scene_ex.export(scene, 1.0 / 25.0, out_world, out_route, "J_S_8", "J_S_125")

    assert out_world.exists() and out_route.exists()
    route = json.loads(out_route.read_text(encoding="utf-8"))
    # 世界距离必须正好是路网距离的 1/25。
    assert route["world_distance_m"] == pytest.approx(route["network_distance_m"] / 25.0, rel=1e-6)
    # 同一段路：南主通道 J_S_8 -> J_S_125 在路网里 117.0 m，世界里 4.68 m。
    assert route["network_distance_m"] == pytest.approx(117.0, abs=0.05)
    assert stats["segments"] > 0


def test_scene_export_lays_every_lane_into_the_world(tmp_path):
    """每条通道都要落成地面模型；少一条，车就会走进一个没有地面的洞。"""
    import gz_export_scene as scene_ex
    from amr_sched.workshop import WorkshopScene

    scene = WorkshopScene.load()
    out_world = tmp_path / "w.world"
    stats = scene_ex.export(scene, 1.0 / 25.0, out_world, tmp_path / "r.json",
                            "J_S_8", "J_S_125")
    text = out_world.read_text(encoding="utf-8")

    assert text.count("<model name=") == stats["segments"]
    for lane in scene.lanes:
        assert ("<model name=\"%s_0\"" % lane.lane_id.lower()) in text
    # 世界包围盒不能超出 1/25 的场景范围太多（140x90 m -> 5.6x3.6 m）。
    xmin, ymin, xmax, ymax = stats["world_bbox"]
    assert xmax - xmin <= 5.6 + 0.2
    assert ymax - ymin <= 3.6 + 0.2
