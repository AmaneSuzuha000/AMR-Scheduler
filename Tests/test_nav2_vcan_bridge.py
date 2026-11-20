"""Tests for L1 Nav2 controller config, SocketCAN `vcan0` chassis bridge, Costmap grid/scan projection, and RViz2 overlay."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from amr_sched.cvcr.execution_feedback import CorridorFeedback, ExecutionFeedbackPacket
from amr_sched.cvcr.p4_belief_choke import CorridorDurationBeliefModel
from amr_sched.cvcr.production import OperationSnapshot
from amr_sched.network import Edge, Network, Node, NodeKind
from tools.live_bridge.costmap_corridor_projector import CostmapCorridorProjector
from tools.live_bridge.rviz_cvcr_overlay import build_rviz_overlay_payloads
from tools.live_bridge.scout_hardware import (
    CanFrame,
    ScoutHardwareBridge,
    ScoutMotionMode,
    ScoutProtocol,
)
from tools.live_bridge.vcan_gazebo_bridge import (
    SocketCANChassisEndpoint,
    build_robot_tf_transforms,
)


class _DummyCanSocketPair:
    """In-memory mock SocketCAN stream testing `tx_socketcan_frames` and `poll_socketcan_frames`."""

    def __init__(self) -> None:
        self.tx_buf: list[bytes] = []
        self.rx_buf: list[bytes] = []

    def send(self, raw: bytes) -> int:
        """Record transmitted 16-byte raw SocketCAN bytes."""
        self.tx_buf.append(bytes(raw))
        return len(raw)

    def recv(self, bufsize: int) -> bytes:
        """Pop queued incoming 16-byte SocketCAN frame or raise BlockingIOError."""
        if not self.rx_buf:
            raise BlockingIOError()
        return self.rx_buf.pop(0)[:bufsize]

    def close(self) -> None:
        """Clear buffers cleanly on disconnect."""
        self.tx_buf.clear()
        self.rx_buf.clear()


def _sample_corridor_net() -> Network:
    """Create minimal two-corridor workshop topology for L1 projection and overlay tests."""
    net = Network()
    net.add_node(Node("S0", 0.0, 0.0, kind=NodeKind.BUFFER, capacity=1))
    net.add_node(Node("J0", 2.0, 0.0, kind=NodeKind.AISLE, capacity=1))
    net.add_node(Node("M_FAST", 4.0, 0.0, kind=NodeKind.AISLE, capacity=1))
    for u, v, d in (("S0", "J0", 2.0), ("J0", "M_FAST", 2.0)):
        net.add_edge(Edge(u, v, d, speed_limit=0.5, capacity=1))
        net.add_edge(Edge(v, u, d, speed_limit=0.5, capacity=1))
    return net


def test_scout_socketcan_counters_and_decode_motion_cmd() -> None:
    """Verify `decode_motion_cmd`, SocketCAN frame TX/RX counters, and `poll_socketcan_frames`."""
    cmd_fr = ScoutProtocol.encode_motion_cmd(0.42, -0.31, counter=11)
    vx, wz, cnt = ScoutProtocol.decode_motion_cmd(cmd_fr.data)
    assert math.isclose(vx, 0.42, abs_tol=1e-3)
    assert math.isclose(wz, -0.31, abs_tol=1e-3)
    assert cnt == 11

    mock_sock = _DummyCanSocketPair()
    bridge = ScoutHardwareBridge(can_channel="vcan0", use_virtual_loopback=False)
    bridge._is_connected = True
    bridge._can_socket = mock_sock  # type: ignore[assignment]
    bridge.state.mode = ScoutMotionMode.CAN_CONTROL

    bridge.send_velocity(0.42, -0.31)
    assert bridge.tx_socketcan_frames == 1
    assert len(mock_sock.tx_buf) == 1
    unpacked = ScoutProtocol.unpack_socketcan_frame(mock_sock.tx_buf[0])
    assert unpacked.arbitration_id == ScoutProtocol.CAN_ID_MOTION_CMD

    # Inject 0x211 status + 0x311 odom back through `_can_socket`
    st = CanFrame(ScoutProtocol.CAN_ID_CHASSIS_STATUS, bytes([ScoutMotionMode.CAN_CONTROL, 248, 0, 0, 0, 0, 0, 0]))
    od1 = ScoutProtocol.encode_odometry(0, 0)
    od2 = ScoutProtocol.encode_odometry(500, 500)
    mock_sock.rx_buf.extend(
        ScoutProtocol.pack_socketcan_frame(f) for f in (st, od1, od2)
    )
    n_rx = bridge.poll_socketcan_frames()
    assert n_rx == 3
    assert bridge.rx_socketcan_frames == 3
    assert math.isclose(bridge.state.odom_x_m, 0.5, abs_tol=1e-4)
    telem = bridge.to_cvcr_telemetry_dict()
    assert telem["tx_socketcan_frames"] == 1
    assert telem["rx_socketcan_frames"] == 3


def test_vcan_chassis_endpoint_loopback_estop_and_tf() -> None:
    """Verify `SocketCANChassisEndpoint` roundtrip, E-stop clamping, and `map->odom->base` TF chain."""
    ep = SocketCANChassisEndpoint("scout_a", can_channel="vcan0", use_kernel_socketcan=False)
    assert ep.connect()
    try:
        vx, wz = ep.send_cmd_vel(0.55, -0.25)
        assert math.isclose(vx, 0.55, abs_tol=1e-3)
        assert math.isclose(wz, -0.25, abs_tol=1e-3)
        rec = ep.publish_odometry_feedback(0.55, 0.0, dt_s=0.2)
        assert rec == 3
        assert ep.stats.total_tx > 0 and ep.stats.total_rx > 0

        ep.estop_latched = True
        assert ep.controller_bridge is not None
        ep.controller_bridge.emergency_stop()
        vx_e, wz_e = ep.send_cmd_vel(0.6, 0.4)
        assert (vx_e, wz_e) == (0.0, 0.0)
    finally:
        ep.disconnect()

    tfs = build_robot_tf_transforms("scout_a", 1.2, 0.3, math.pi / 2.0, origin_xy=(0.5, 2.5))
    assert len(tfs) == 3
    assert tfs[0]["parent_frame"] == "map" and tfs[0]["child_frame"] == "scout_a/odom"
    assert tfs[0]["translation"] == (0.5, 2.5, 0.0)
    assert tfs[1]["child_frame"] == "scout_a/base_footprint"
    assert tfs[2]["child_frame"] == "scout_a/laser_link"


def test_costmap_occupancy_grid_and_laserscan_teammate_filtering() -> None:
    """Verify zero-hardcoding `OccupancyGrid` + `LaserScan` projection with teammate masking & reopen."""
    net = _sample_corridor_net()
    proj = CostmapCorridorProjector(
        net,
        corridor_half_width_m=0.50,
        robot_exclusion_radius_m=0.45,
        min_obstacle_hits=2,
        clear_hysteresis_scans=2,
        default_clear_horizon_steps=16,
    )
    # 1. OccupancyGrid (20x10 @ 0.25m, origin (0.0, -1.25)): cell (12, 5) = world (3.125, 0.125) on J0<->M_FAST
    w, h, res = 20, 10, 0.25
    ox, oy = 0.0, -1.25
    grid = [0] * (w * h)
    # Place teammate directly on (0.875, 0.125) -> masked out by robot_positions
    grid[5 * w + 3] = 100
    grid[5 * w + 4] = 100
    assert proj.project_occupancy_grid((ox, oy), res, w, h, grid, robot_positions={"scout_b": (1.0, 0.0)}) == []

    # Place real box cells on ("J0", "M_FAST") at (3.125, 0.125) & (3.375, 0.125) while scout_b stays on (S0, J0)
    grid[5 * w + 12] = 100
    grid[5 * w + 13] = 100
    rep1 = proj.project_occupancy_grid((ox, oy), res, w, h, grid, robot_positions={"scout_b": (1.0, 0.0)})
    assert len(rep1) == 1 and rep1[0].edge == ("J0", "M_FAST")

    # Empty grid triggers hysteresis reopen after 2 clear scans
    empty = [0] * (w * h)
    assert len(proj.project_occupancy_grid((ox, oy), res, w, h, empty)) == 1
    assert proj.project_occupancy_grid((ox, oy), res, w, h, empty) == []

    # 2. Polar LaserScan projection from scout_a at (1.5, 0.0, 0.0) hitting obstacle at (3.0, 0.0) & (3.1, 0.05)
    ranges = [math.inf] * 11
    ranges[5] = 1.5  # angle 0 -> (3.0, 0.0)
    ranges[6] = 1.6  # angle +0.05 -> (3.1, 0.08)
    scan_rep = proj.project_laserscan((1.5, 0.0, 0.0), ranges, -0.25, 0.05)
    assert len(scan_rep) == 1 and scan_rep[0].edge == ("J0", "M_FAST")

    # 3. Multi-scan mark hysteresis ignores 1-scan transient and confirms 2 consecutive scans
    proj_hyst = CostmapCorridorProjector(net, mark_hysteresis_scans=2, clear_hysteresis_scans=2)
    assert proj_hyst.project_points([(3.1, 0.0), (3.3, 0.0)]) == []
    confirmed = proj_hyst.project_points([(3.1, 0.0), (3.3, 0.0)])
    assert len(confirmed) == 1 and confirmed[0].edge == ("J0", "M_FAST")


def test_rviz_overlay_and_nav2_yaml_configs() -> None:
    """Verify RViz2 L1-L2-L3 MarkerSpec payload generation and valid Nav2/RViz2 config schemas."""
    net = _sample_corridor_net()
    belief = CorridorDurationBeliefModel()
    belief.observe_step(5, {("J0", "M_FAST")})
    pkt = ExecutionFeedbackPacket(
        tick=8,
        corridor_updates={
            ("J0", "M_FAST"): CorridorFeedback(
                src="J0", dst="M_FAST", status="blocked", extra_delay_steps=14, blocked_until_step=22, q3_penalty_override=28
            )
        },
    )
    ops = [OperationSnapshot("OP_0", ("T0",), "M_FAST", 1, 1, 3, 40, resource_windows=((10, 25),))]
    out = build_rviz_overlay_payloads(
        net,
        tick=8,
        belief_model=belief,
        feedback_packet=pkt,
        planned_routes={"scout_a": [("S0", 0), ("J0", 4), ("M_FAST", 10)]},
        robot_poses={"scout_a": (0.8, 0.0)},
        operations=ops,
    )
    assert len(out["topology_markers"]) > 0
    assert len(out["corridor_belief_markers"]) >= 2
    assert "scout_a" in out["planned_paths"]
    assert len(out["kitting_hud_markers"]) == 1

    root = Path(__file__).resolve().parents[1]
    nav2_cfg = yaml.safe_load((root / "tools" / "ros2" / "nav2_scout_config" / "nav2_controller_params.yaml").read_text(encoding="utf-8"))
    cs = nav2_cfg["controller_server"]["ros__parameters"]
    assert cs["use_sim_time"] is True
    assert "RegulatedPurePursuitController" in cs["FollowPath"]["plugin"]
    lc = nav2_cfg["local_costmap"]["local_costmap"]["ros__parameters"]
    assert lc["global_frame"] == "map"
    assert lc["obstacle_layer"]["scan"]["topic"] == "scan"

    rviz_cfg = yaml.safe_load((root / "tools" / "ros2" / "nav2_scout_config" / "amr_v15_digital_twin.rviz").read_text(encoding="utf-8"))
    vm = rviz_cfg["Visualization Manager"]
    assert vm["Global Options"]["Fixed Frame"] == "map"
    topics = {
        d["Topic"]["Value"]
        for d in vm["Displays"]
        if isinstance(d.get("Topic"), dict) and "Value" in d["Topic"]
    }
    for expected in (
        "/scout_a/scan",
        "/scout_a/local_costmap/costmap",
        "/cvcr/topology_markers",
        "/cvcr/corridor_belief_markers",
        "/cvcr/fleet_plan_markers",
        "/cvcr/kitting_hud_markers",
    ):
        assert expected in topics
