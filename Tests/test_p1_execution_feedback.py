"""Tests for P1 Execution-Aware AMR State & Deviation Feedback Interface."""

from __future__ import annotations

import pytest

from amr_sched.cvcr.commitment import TIER_EXECUTING_HARD, classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.execution_feedback import (
    EXEC_FEEDBACK_SCHEMA,
    AmrExecutionStatus,
    CorridorFeedback,
    ExecutionFeedbackPacket,
)
from amr_sched.cvcr.policy import LGCVCRPolicy
from amr_sched.cvcr.snapshot import (
    NativeSnapshot,
    NativeSnapshotCapture,
    OperationSnapshot,
    RobotSnapshot,
    TaskSnapshot,
)
from amr_sched.external_execution import ExternalExecution, ExternalHandling
from amr_sched.monitor import DeviationEvent
from amr_sched.network import Edge, Network, Node, NodeKind
from amr_sched.rl import AMRReschedulingEnv


def _make_diamond_network() -> Network:
    """Build small metric depot->station diamond with an alternate detour corridor."""
    net = Network()
    net.add_node(Node("N_DEPOT", 0.0, 0.0, kind=NodeKind.BUFFER, capacity=4))
    net.add_node(Node("N_SHORT", 5.0, 0.0, kind=NodeKind.AISLE, capacity=1))
    net.add_node(Node("N_ALT1", 2.0, 5.0, kind=NodeKind.AISLE, capacity=2))
    net.add_node(Node("N_ALT2", 8.0, 5.0, kind=NodeKind.AISLE, capacity=2))
    net.add_node(Node("N_BAY", 10.0, 0.0, kind=NodeKind.WORKSTATION, capacity=1))
    net.add_node(Node("N_PARK", 12.0, 0.0, kind=NodeKind.BUFFER, capacity=4))
    for u, v, d in (
        ("N_DEPOT", "N_SHORT", 5.0),
        ("N_SHORT", "N_BAY", 5.0),
        ("N_DEPOT", "N_ALT1", 5.5),
        ("N_ALT1", "N_ALT2", 6.0),
        ("N_ALT2", "N_BAY", 5.5),
        ("N_BAY", "N_PARK", 2.0),
    ):
        net.add_edge(Edge(u, v, length=d, speed_limit=1.0, capacity=1))
        net.add_edge(Edge(v, u, length=d, speed_limit=1.0, capacity=1))
    return net


def test_p1_feedback_schema_validation_and_roundtrip() -> None:
    """Verify strict schema validation and lossless wire dictionary serialization."""
    st = AmrExecutionStatus(
        robot_id="agv_0",
        pose_xy_yaw=(1.25, 3.5, 0.785),
        localization_source="amcl",
        active_leg=("N_DEPOT", "N_SHORT"),
        leg_progress=0.45,
        eta_drift_steps=6,
        speed_scale=0.65,
        local_planner_state="detouring",
        receipt_age_s=0.35,
        handling_delay_steps=2,
        route_id="run1:0",
    )
    cf = CorridorFeedback(
        src="N_SHORT",
        dst="N_BAY",
        status="blocked",
        blocked_until_step=80,
        source="deviation_event",
        reason="ROUTE_DEVIATION",
    )
    pkt = ExecutionFeedbackPacket(
        tick=14,
        robot_states={"agv_0": st},
        corridor_updates={("N_SHORT", "N_BAY"): cf},
        source_events=({"event": "ROUTE_DEVIATION", "robot_id": "agv_0"},),
    )
    assert pkt.schema == EXEC_FEEDBACK_SCHEMA
    assert not pkt.is_empty()
    assert st.is_degraded and not st.is_hard_faulted
    assert pkt.robot_ready_offset_steps("agv_0", dt=1.0) == 8

    wire = pkt.to_dict()
    rebuilt = ExecutionFeedbackPacket.from_dict(wire)
    assert rebuilt.canonical_payload() == pkt.canonical_payload()
    assert rebuilt.signature() == pkt.signature()

    with pytest.raises(ValueError):
        AmrExecutionStatus(robot_id="agv_0", leg_progress=1.5)
    with pytest.raises(ValueError):
        AmrExecutionStatus(robot_id="agv_0", speed_scale=-0.1)
    with pytest.raises(ValueError):
        AmrExecutionStatus(robot_id="agv_0", local_planner_state="unknown_mode")
    with pytest.raises(ValueError):
        CorridorFeedback(src="A", dst="A", status="open")


def test_p1_nominal_snapshot_digest_backward_compatible() -> None:
    """Nominal benchmark snapshots remain zero-diff when no physical deviations exist."""
    env = AMRReschedulingEnv(
        topology="ring",
        seed=6001,
        max_steps=40,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        real_compute_window_coupling=True,
    )
    env.reset(seed=6001, options={"topology": "ring"})
    assert env.sim is not None and env.tracker is not None

    snap_plain = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
    assert snap_plain.execution_feedback is None

    empty_pkt = ExecutionFeedbackPacket(tick=snap_plain.tick)
    assert empty_pkt.is_empty()
    snap_empty_fb = NativeSnapshotCapture.capture(
        env.sim, env.tracker.operations, hide_future_releases=True, execution_feedback=empty_pkt
    )
    assert snap_empty_fb.compute_digest() == snap_plain.compute_digest()


def test_p1_adapter_bridges_external_execution_handling_and_deviation() -> None:
    """Bridge ExternalExecution, ExternalHandling, and DeviationEvent into CVCR snapshot."""
    net = _make_diamond_network()
    gate = ExternalExecution("scene_p1", "agv_0", (0.0, 0.0), source="gazebo", arrival_tolerance=0.2, timeout=2.0)
    gate.feedback(
        {
            **gate.envelope(),
            "seq": 1,
            "timestamp": 10.0,
            "pose": [0.0, 0.0, 0.0],
            "event": "POSE",
        },
        received_at=10.0,
    )
    hdl = ExternalHandling(gate, source="contact_sensor")
    hdl.require("T1", "LOAD", "N_DEPOT", (0.0, 0.0), source_now=10.0)

    class _SimStub:
        _step_cursor = 12
        agvs = [type("Agv", (), {"agv_id": "agv_0"})()]
        executions = {"agv_0": gate}
        handlings = {"agv_0": hdl}
        _closed_corridors = set()
        _transit = {"agv_0": ("N_DEPOT", "N_SHORT", 5)}
        _physical_detours = {}

    _SimStub.network = net
    dev = DeviationEvent(
        robot_id="agv_0",
        event="ROUTE_DEVIATION",
        blocked_edge=("N_SHORT", "N_DEPOT"),
        current_pose=(0.5, 0.0, 0.0),
        timestamp=10.5,
    )
    pkt = ExecutionFeedbackPacket.from_simulator_and_events(
        _SimStub,
        deviations=[dev],
        robot_telemetry={"agv_0": {"eta_drift_steps": 4, "handling_delay_steps": 3}},
        now_wall=10.8,
    )
    assert pkt is not None
    st = pkt.robot_states["agv_0"]
    assert st.localization_source == "gazebo"
    assert st.pose_xy_yaw == (0.0, 0.0, 0.0)
    assert pkt.robot_ready_offset_steps("agv_0") == 7
    assert ("N_DEPOT", "N_SHORT") in pkt.corridor_updates
    assert ("N_SHORT", "N_DEPOT") in pkt.corridor_updates
    assert pkt.corridor_updates[("N_DEPOT", "N_SHORT")].status == "blocked"


def test_p1_evaluator_and_policy_adapt_to_drift_and_blocked_corridor() -> None:
    """Execution-aware CVCR detours around blocked edges & avoids delayed/degraded AMRs."""
    net = _make_diamond_network()
    base_snap = NativeSnapshot(
        schema="amr-lgcvcr.snapshot.v1",
        tick=10,
        version={"world": 1, "reservation": 1, "production": 1, "fleet": 1, "admitted_sig": -1},
        world_fingerprint="p1_test",
        robots=[
            RobotSnapshot("agv_0", "N_DEPOT", "to_dropoff", cur_task="T_CARGO", loaded_task_id="T_CARGO", route=("N_SHORT", "N_BAY")),
            RobotSnapshot("agv_1", "N_DEPOT", "idle"),
        ],
        tasks=[
            TaskSnapshot("T_CARGO", "N_DEPOT", "N_BAY", release_step=0, due_step=60, status="picked", assigned_agv="agv_0", kit_op_id="OP_1", pickup_step=5),
            TaskSnapshot("T_SOFT", "N_DEPOT", "N_BAY", release_step=8, due_step=60, status="assigned", assigned_agv="agv_0", kit_op_id="OP_1"),
        ],
        operations=[
            OperationSnapshot("OP_1", ("T_CARGO", "T_SOFT"), "N_BAY", crew_id=1, tool_id=1, proc_steps=10, due_step=80),
        ],
        reservations=[],
        now_monotonic=10.0,
    )

    ev_nom = CVCREvaluator(net, base_snap, mode="B2", use_rust=False)
    res_nom = ev_nom.evaluate({"T_CARGO": "agv_0", "T_SOFT": "agv_1"})
    assert res_nom.valid
    short_path_nodes = [n for n, _ in res_nom.robot_paths["agv_0"]]
    assert "N_SHORT" in short_path_nodes and "N_ALT1" not in short_path_nodes

    # Inject L1 execution deviations: N_SHORT blocked by dynamic obstacle + agv_0 severely delayed
    fb = ExecutionFeedbackPacket(
        tick=10,
        robot_states={
            "agv_0": AmrExecutionStatus(
                robot_id="agv_0",
                localization_source="amcl",
                local_planner_state="waiting_obstacle",
                eta_drift_steps=20,
                speed_scale=0.5,
            )
        },
        corridor_updates={
            ("N_DEPOT", "N_SHORT"): CorridorFeedback("N_DEPOT", "N_SHORT", status="blocked", blocked_until_step=120),
            ("N_SHORT", "N_DEPOT"): CorridorFeedback("N_SHORT", "N_DEPOT", status="blocked", blocked_until_step=120),
        },
    )
    snap_dev = NativeSnapshot(
        schema=base_snap.schema,
        tick=base_snap.tick,
        version=base_snap.version,
        world_fingerprint=base_snap.world_fingerprint,
        robots=base_snap.robots,
        tasks=base_snap.tasks,
        operations=base_snap.operations,
        reservations=base_snap.reservations,
        now_monotonic=base_snap.now_monotonic,
        execution_feedback=fb,
    )

    clf = classify_commitments(snap_dev)
    assert clf.task_tiers["T_CARGO"] == TIER_EXECUTING_HARD
    assert "T_CARGO" in clf.frozen_tasks
    assert "T_SOFT" in clf.disturbed_tasks and "T_SOFT" in clf.releasable_tasks

    ev_b1 = CVCREvaluator(net, snap_dev, mode="B1", use_rust=True)
    ev_b2 = CVCREvaluator(net, snap_dev, mode="B2", use_rust=True)
    res_b1 = ev_b1.evaluate({"T_CARGO": "agv_0", "T_SOFT": "agv_1"})
    res_b2 = ev_b2.evaluate({"T_CARGO": "agv_0", "T_SOFT": "agv_1"})
    assert res_b1.valid and res_b2.valid
    assert res_b1.objective == res_b2.objective
    assert "N_ALT1" in [n for n, _ in res_b2.robot_paths["agv_0"]]
    assert "N_SHORT" not in [n for n, _ in res_b2.robot_paths["agv_0"]]
    assert res_b2.order_unload_ticks["T_CARGO"] > res_nom.order_unload_ticks["T_CARGO"]

    # Policy must preserve onboard T_CARGO on agv_0 while reassigning T_SOFT to healthy agv_1
    pol = LGCVCRPolicy("V13_KITTING_DET", net, use_rust=False)
    prop = pol.propose(snap_dev, budget_ms=60.0)
    assert prop.proposed_assignment["T_CARGO"] == "agv_0"
    assert prop.proposed_assignment["T_SOFT"] == "agv_1"
