"""Regression tests for wheelset_workshop_v2 layout clearances, RAWSim-O export, and mobile AGV physics model."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from amr_sched.entities import Task, TaskStatus
from amr_sched.workshop import load_and_validate
from tools.wheelset_scene.wheelset_v2_agv_model import (
    CARRIER_DECK_WHEELSET_Z,
    build_dynamic_wheelset_sdf,
    build_wheelset_agv_urdf,
)
from tools.wheelset_scene.wheelset_v2_live_helpers import (
    GROUNDED_WORKPIECE_MAX_Z_M,
    GroundedWheelsetWorkpieceSync,
    STATION_RAIL_Z_M,
    build_wheelset_v2_nav_and_loc_templates,
)
from tools.wheelset_scene.workshop_v2_layout import (
    REMOVED_STATIC_SNAPSHOT_IDS,
    compile_v2_bundle,
)


def test_v2_layout_bundle_clearances_and_workshop_validation(tmp_path: Path) -> None:
    out = tmp_path / "v2_bundle"
    summary = compile_v2_bundle(out, resolution=0.25)

    assert summary["scene_id"] == "wheelset_workshop_v2"
    audit = summary["clearance_audit"]
    assert audit["passed"] is True
    assert audit["min_clearance_m"] >= 1.15
    assert audit["connected_components"] == 1
    assert audit["failed"] == []

    sc, issues = load_and_validate(out / "wheelset_workshop_v2_nav.json")
    assert issues == []
    net = sc.network()
    assert len(net.nodes) == 50
    assert len(net.edges) == 118

    scene_json = (out / "wheelset_workshop_v2.json").read_text(encoding="utf-8")
    for removed_id in REMOVED_STATIC_SNAPSHOT_IDS:
        assert f'"{removed_id}"' not in scene_json

    assert (out / "wheelset_workshop_v2_map.xinst").is_file()
    assert (out / "wheelset_workshop_v2.world").is_file()
    assert (out / "map.pgm").is_file()
    assert (out / "map.yaml").is_file()


def test_wheelset_v2_agv_urdf_physics_sensors_and_guard() -> None:
    urdf_text = build_wheelset_agv_urdf(namespace="agv_1")
    root = ET.fromstring(urdf_text)

    mesh = root.find("./link[@name='base_link']/visual/geometry/mesh")
    assert mesh is not None
    assert mesh.attrib["filename"] == "model://wsv1_wheelset_agv/meshes/wheelset_agv.dae"

    box = root.find("./link[@name='base_link']/collision/geometry/box")
    assert box is not None
    assert box.attrib["size"] == "1.7 1.5 0.14"

    drive = root.find(".//plugin[@filename='libgazebo_ros_diff_drive.so']")
    assert drive is not None
    assert drive.findtext("odometry_source") == "0"
    assert drive.findtext("publish_odom_tf") == "false"
    assert drive.findtext("ros/namespace") == "/agv_1"
    remaps = [el.text for el in drive.findall("ros/remapping")]
    assert "cmd_vel:=guarded_cmd_vel" in remaps
    assert "odom:=wheel/odometry" in remaps
    assert "/tf:=/agv_1/tf" in remaps

    assert root.find(".//plugin[@filename='libamr_motion_guard.so']") is not None
    assert root.find(".//plugin[@filename='libgazebo_ros_imu_sensor.so']") is not None
    assert root.find(".//plugin[@filename='libgazebo_ros_bumper.so']") is not None

    ray = root.find(".//sensor[@type='ray']")
    assert ray is not None
    assert ray.findtext("update_rate") == "20"
    assert float(ray.findtext("ray/range/max")) == pytest.approx(60.0)

    loc, nav = build_wheelset_v2_nav_and_loc_templates()
    assert loc["amcl"]["ros__parameters"]["robot_model_type"] == "nav2_amcl::DifferentialMotionModel"
    assert nav["local_costmap"]["local_costmap"]["ros__parameters"]["robot_base_frame"] == "front_tracking"


def test_grounded_workpiece_sdf_and_height_invariant(tmp_path: Path) -> None:
    sdf_root = ET.fromstring(build_dynamic_wheelset_sdf("ws_wp_01"))
    model = sdf_root.find("./model[@name='ws_wp_01']")
    assert model is not None
    assert model.find(".//collision") is None
    assert model.findtext(".//mesh/uri") == "model://wsv1_wheelset/meshes/wheelset.dae"

    class _DummyNode:
        def __init__(self, x: float, y: float) -> None:
            self.x = x
            self.y = y

    class _DummyNet:
        def __init__(self) -> None:
            self.nodes = {"A": _DummyNode(10.0, 20.0), "B": _DummyNode(30.0, 20.0)}

    class _DummyExec:
        def __init__(self, x: float, y: float, yaw: float) -> None:
            self.pose = {"x": x, "y": y, "yaw": yaw}

    class _DummySim:
        def __init__(self, step_cursor: int) -> None:
            self._step_cursor = step_cursor

            class _Cfg:
                dt = 1.0

            self.config = _Cfg()

    task = Task("T1", pickup="A", dropoff="B")
    sync = GroundedWheelsetWorkpieceSync(
        {"ws1": {"chain": ["T1"], "hold_xy": (10.0, 20.0)}},
        {"T1": task},
        _DummyNet(),
        tmp_path / "transfer.jsonl",
    )
    sync._last_sync_wall = -10.0
    sync.step(_DummySim(1), {})
    task.status = TaskStatus.PICKED
    task.assigned_agv = "AGV_1"
    sync._last_sync_wall = -10.0
    sync.step(_DummySim(2), {"AGV_1": _DummyExec(18.5, 20.0, 0.25)})
    task.status = TaskStatus.COMPLETED
    sync._last_sync_wall = -10.0
    sync.step(_DummySim(3), {"AGV_1": _DummyExec(30.0, 20.0, 0.0)})

    summary = sync.finalize()
    assert summary["grounded_passed"] is True
    assert summary["min_z_m"] == pytest.approx(STATION_RAIL_Z_M)
    assert summary["max_z_m"] == pytest.approx(CARRIER_DECK_WHEELSET_Z)
    assert summary["max_z_m"] <= GROUNDED_WORKPIECE_MAX_Z_M
    assert summary["carried_updates"] == 1
    with pytest.raises(ValueError, match="height invariant violated"):
        sync._record(4.0, "ws1", "illegal_fly", None, 10.0, 20.0, 6.0, 0.0)
