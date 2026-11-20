"""Tests for AgileX SCOUT Mini v2 SocketCAN extensions and CostmapCorridorProjector."""

from __future__ import annotations

import math
import socket

from amr_sched.network import Edge, Network, Node, NodeKind
from tools.live_bridge.costmap_corridor_projector import CostmapCorridorProjector
from tools.live_bridge.scout_hardware import (
    CanFrame,
    ScoutHardwareBridge,
    ScoutMotionMode,
    ScoutProtocol,
)


def _sample_corridor_net() -> Network:
    net = Network()
    net.add_node(Node("A", 0.0, 0.0, kind=NodeKind.BUFFER, capacity=1))
    net.add_node(Node("B", 4.0, 0.0, kind=NodeKind.AISLE, capacity=1))
    net.add_node(Node("C", 4.0, 3.0, kind=NodeKind.WORKSTATION, capacity=1))
    for u, v, d in (("A", "B", 4.0), ("B", "C", 3.0)):
        net.add_edge(Edge(u, v, d, speed_limit=0.5, capacity=1))
        net.add_edge(Edge(v, u, d, speed_limit=0.5, capacity=1))
    return net


def test_scout_v2_odometry_motion_state_and_socketcan_framing() -> None:
    """Verify 0x221/0x311 encoding/decoding, 16-byte SocketCAN packing, and wheel odom pose."""
    f_ms = ScoutProtocol.encode_motion_state(0.55, -0.30)
    assert f_ms.arbitration_id == ScoutProtocol.CAN_ID_MOTION_STATE
    vx, wz = ScoutProtocol.decode_motion_state(f_ms.data)
    assert math.isclose(vx, 0.55, abs_tol=1e-3)
    assert math.isclose(wz, -0.30, abs_tol=1e-3)

    f_od = ScoutProtocol.encode_odometry(1240, -350)
    assert f_od.arbitration_id == ScoutProtocol.CAN_ID_ODOMETRY
    l_mm, r_mm = ScoutProtocol.decode_odometry(f_od.data)
    assert (l_mm, r_mm) == (1240, -350)

    raw = ScoutProtocol.pack_socketcan_frame(f_od)
    assert len(raw) == 16
    unpacked = ScoutProtocol.unpack_socketcan_frame(raw)
    assert unpacked.arbitration_id == ScoutProtocol.CAN_ID_ODOMETRY
    assert ScoutProtocol.decode_odometry(unpacked.data) == (1240, -350)


def test_scout_wire_relay_loopback_and_odom_pose_integration() -> None:
    """Verify UDP SocketCAN wire relay TX + 0x311 straight-line differential odometry integration."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0))
    srv.settimeout(1.0)
    port = srv.getsockname()[1]

    bridge = ScoutHardwareBridge(
        use_virtual_loopback=False,
        transport_mode="wire_relay",
        wire_endpoint=("127.0.0.1", port),
    )
    assert bridge.connect()
    try:
        bridge.send_velocity(0.6, 0.0)
        raw, _ = srv.recvfrom(64)
        frame = ScoutProtocol.unpack_socketcan_frame(raw)
        assert frame.arbitration_id == ScoutProtocol.CAN_ID_MOTION_CMD

        bridge.receive_can_frame(
            CanFrame(ScoutProtocol.CAN_ID_CHASSIS_STATUS, bytes([ScoutMotionMode.CAN_CONTROL, 245, 0, 0, 0, 0, 0, 0]))
        )
        bridge.receive_can_frame(ScoutProtocol.encode_odometry(1000, 1000))
        bridge.receive_can_frame(ScoutProtocol.encode_odometry(2500, 2500))
        x, y, yaw = bridge.state.odom_xy_yaw
        assert math.isclose(x, 1.5, abs_tol=1e-4)
        assert math.isclose(y, 0.0, abs_tol=1e-4)
        assert math.isclose(yaw, 0.0, abs_tol=1e-4)
        telem = bridge.to_cvcr_telemetry_dict(nominal_speed_m_s=0.5)
        assert telem["localization_source"] == "scout_odom_can"
        assert telem["local_planner_state"] == "nominal"
    finally:
        bridge.disconnect()
        srv.close()


def test_costmap_projector_teammate_filter_and_hysteresis() -> None:
    """Verify CostmapCorridorProjector ignores teammate footprints and applies clear hysteresis."""
    net = _sample_corridor_net()
    proj = CostmapCorridorProjector(
        net,
        corridor_half_width_m=0.5,
        robot_exclusion_radius_m=0.45,
        min_obstacle_hits=2,
        clear_hysteresis_scans=2,
        default_clear_horizon_steps=18,
    )
    # Teammate stationed directly on (B, C) at (4.0, 1.5) must NOT trigger road closure
    reports = proj.project_points([(4.0, 1.48), (4.0, 1.52)], robot_positions={"r1": (4.0, 1.50)})
    assert reports == []

    # True external obstacle on (A, B) at (2.0, 0.05) triggers canonical edge ("A", "B")
    reports = proj.project_points([(1.95, 0.04), (2.05, -0.03), (4.0, 1.50)], robot_positions={"r1": (4.0, 1.50)})
    assert len(reports) == 1
    assert reports[0].edge == ("A", "B")
    assert reports[0].estimated_clear_horizon_steps == 18

    # First clean scan keeps ("A", "B") held due to hysteresis=2; second clean scan clears it
    assert [r.edge for r in proj.project_points([])] == [("A", "B")]
    assert proj.project_points([]) == []


def test_costmap_projector_laserscan_and_occupancy_grid() -> None:
    """Verify LaserScan polar and OccupancyGrid projection onto network edges."""
    net = _sample_corridor_net()
    proj = CostmapCorridorProjector(net, corridor_half_width_m=0.55, min_obstacle_hits=2, clear_hysteresis_scans=1)
    # Robot at (0.0, 0.0, yaw=0.0) sees obstacle ~2.0m ahead along (A, B)
    r_grid = proj.project_occupancy_grid(
        origin_xy=(0.0, -1.0),
        resolution=0.25,
        width=20,
        height=10,
        data=[100 if (r == 4 and c in (7, 8)) else 0 for r in range(10) for c in range(20)],
    )
    assert [r.edge for r in r_grid] == [("A", "B")]

    r_scan = proj.project_laserscan((4.0, 0.0, math.pi / 2.0), [1.4, 1.5, 1.6], angle_min=-0.05, angle_increment=0.05)
    assert [r.edge for r in r_scan] == [("B", "C")]
