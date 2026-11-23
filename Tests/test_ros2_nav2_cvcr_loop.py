"""Tests for P3 ROS 2 / Nav2 / Gazebo Physical ExternalExecution <-> CVCR Closed Loop."""

from __future__ import annotations

import math

import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.physical_bridge import (
    CorridorObstacleReport,
    Nav2RobotTelemetry,
    PhysicalCVCRCoordinator,
)
from amr_sched.cvcr.policy import LGCVCRPolicy
from amr_sched.cvcr.production import OperationSnapshot
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.cvcr.transaction import NativeTransactionOwner
from amr_sched.entities import Agv, Task, TaskStatus
from amr_sched.external_execution import ExternalExecution, ExternalHandling
from amr_sched.network import Edge, Network, Node, NodeKind
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.simulation import SimulationConfig, Simulator
from tools.wheelset_scene.navigation_fleet import build_sensor_fleet


def _build_physical_bypass_network() -> tuple[Network, dict[str, tuple[float, float]]]:
    """Build capacity-1 metric network with distinct starts, junction obstacles, and alt corridor."""
    coords = {
        "S0": (0.0, 0.0, NodeKind.BUFFER),
        "S1": (0.0, 2.0, NodeKind.BUFFER),
        "J0": (2.0, 1.0, NodeKind.AISLE),
        "M_FAST": (4.0, 1.0, NodeKind.AISLE),
        "B1": (2.0, 4.0, NodeKind.AISLE),
        "B2": (4.0, 4.0, NodeKind.AISLE),
        "J1": (6.0, 1.0, NodeKind.AISLE),
        "W0": (8.0, 0.0, NodeKind.WORKSTATION),
        "W1": (8.0, 2.0, NodeKind.WORKSTATION),
    }
    net = Network()
    for nid, (x, y, kind) in coords.items():
        net.add_node(Node(nid, x, y, kind=kind, capacity=1))
    edges = [
        ("S0", "J0", 2.2),
        ("S1", "J0", 2.2),
        ("J0", "M_FAST", 2.0),
        ("M_FAST", "J1", 2.0),
        ("J0", "B1", 3.0),
        ("B1", "B2", 2.0),
        ("B2", "J1", 3.6),
        ("S1", "B1", 2.8),
        ("J1", "W0", 2.2),
        ("J1", "W1", 2.2),
    ]
    for u, v, d in edges:
        net.add_edge(Edge(u, v, length=d, speed_limit=0.5, capacity=1))
        net.add_edge(Edge(v, u, length=d, speed_limit=0.5, capacity=1))
    return net, {k: (v[0], v[1]) for k, v in coords.items()}


def _advance_physical_motion(
    sim: Simulator,
    positions: dict[str, tuple[float, float]],
    motion: dict[str, dict],
    now: float,
    *,
    fail_stop_robots: set[str] = (),
    degraded_robots: set[str] = (),
) -> None:
    """Drive realistic Nav2/AMCL POSE, ARRIVED, and STOPPED messages for all bound physical gates."""
    for robot, gate in sim.executions.items():
        if robot in degraded_robots and gate.bound:
            # Simulate stalled AMCL / scan timeout: cease sending fresh receipts
            gate.drain_commands()
            continue
        # Drain newly queued MOVE commands and emit STOPPED if faulted cleanly at node/after deceleration
        gate.drain_commands()
        if gate.fault and gate.bound:
            xy = positions[robot]
            if not gate.stopped and (robot in fail_stop_robots or gate.command is None):
                msg = {
                    **gate.envelope(),
                    "event": "STOPPED",
                    "pose": [xy[0], xy[1], 0.0],
                    "seq": gate.last_seq + 1,
                    "timestamp": now,
                }
                sim.execution_feedback(msg, now)
                continue
            msg = {
                **gate.envelope(),
                "event": "POSE",
                "pose": [xy[0], xy[1], 0.0],
                "seq": gate.last_seq + 1,
                "timestamp": now,
            }
            sim.execution_feedback(msg, now)
            continue

        command = gate.command
        event = "POSE"
        xy = positions[robot]
        if command is not None:
            token = command["route_id"]
            if token not in motion:
                dist = math.dist(xy, command["target_xy"])
                dur = max(0.4, (dist / max(0.05, command["speed"])) * 1.15)
                motion[token] = dict(start=now, source=xy, target=tuple(command["target_xy"]), duration=dur)
            act = motion[token]
            frac = min(1.0, (now - act["start"]) / max(1e-6, act["duration"]))
            xy = tuple(a + (b - a) * frac for a, b in zip(act["source"], act["target"]))
            positions[robot] = xy
            if frac >= 1.0:
                event = "ARRIVED"
        msg = {
            **gate.envelope(),
            "event": event,
            "pose": [xy[0], xy[1], 0.0],
            "seq": gate.last_seq + 1,
            "timestamp": now,
        }
        res = sim.execution_feedback(msg, now)
        assert res["accepted"], f"Rejected feedback for {robot} at t={now}: {res}"


def test_p3_default_transaction_owner_still_blocks_physical_without_opt_in() -> None:
    """Verify legacy SW TransactionOwner rejects physical fleets while opt-in mode guards inflight pins."""
    net, coords = _build_physical_bypass_network()
    gates = {
        "r0": ExternalExecution("sc", "r0", coords["S0"], run_id="p3_guard", source="localized", frame_id="map", namespace_routes=True),
        "r1": ExternalExecution("sc", "r1", coords["S1"], run_id="p3_guard", source="localized", frame_id="map", namespace_routes=True),
    }
    sim = Simulator(
        net,
        [Task("T0", "J0", "W0"), Task("T1", "S1", "W1")],
        [Agv("r0", max_speed=0.5, start_node="S0"), Agv("r1", max_speed=0.5, start_node="S1")],
        SpaceTimeAStarPlanner(max_speed=0.5, dt=0.5),
        GreedyAllocator(),
        SimulationConfig(dt=0.5, max_steps=600, load_time=0.5, unload_time=0.5, idle_retreat=False, deadlock_timeout=0, reschedule_policy=ReschedulePolicy.NONE),
        executions=gates,
    )
    sim.reset()
    # Bootstrap initial poses and run until r0 launches physical transit on S0->J0 (T0 is ASSIGNED and in-transit)
    for rid, g in gates.items():
        sim.execution_feedback({**g.envelope(), "event": "POSE", "pose": [*g.start_xy, 0.0], "seq": 1, "timestamp": 0.0}, 0.0)
    sim.step_once()
    assert "r0" in sim._transit or gates["r0"].command is not None
    assert sim.tasks[0].status == TaskStatus.ASSIGNED
    snap = NativeSnapshotCapture.capture(sim, [], hide_future_releases=True)
    prop = LGCVCRPolicy("V13_KITTING_DET", net, use_rust=False).propose(snap, budget_ms=20.0)

    # Default SW owner MUST still forbid unconstrained SW mutation on physical fleets
    sw_owner = NativeTransactionOwner(sim, [])
    rcpt_sw = sw_owner.validate_and_commit(prop)
    assert not rcpt_sw.accepted
    assert "PHYSICAL_MODE_FORBIDDEN_FOR_SW_TRANSACTION" in rcpt_sw.reasons

    # Physical owner MUST forbid mutating an in-flight / commanded robot away from its assigned task
    phys_owner = NativeTransactionOwner(sim, [], allow_physical_mode=True)
    # r0 is now holding T0 in transit / commanded
    assert "r0" in sim._transit or gates["r0"].command is not None
    bad_assign = dict(prop.proposed_assignment)
    bad_assign["T0"] = "r1"
    bad_prop = prop.__class__(**{**prop.__dict__, "proposed_assignment": bad_assign})
    rcpt_bad = phys_owner.validate_and_commit(bad_prop)
    assert not rcpt_bad.accepted
    assert any("VIOLATION_PHYSICAL_INFLIGHT_MUTATION:T0" in r for r in rcpt_bad.reasons)


def test_p3_physical_cvcr_dynamic_obstacle_detour_and_pin_safety() -> None:
    """Verify PhysicalCVCRCoordinator closes/reopens obstacles, detours safely, and passes audit."""
    net, coords = _build_physical_bypass_network()
    gates = {
        "r0": ExternalExecution("sc", "r0", coords["S0"], run_id="p3_detour", source="localized", frame_id="map", namespace_routes=True, timeout=6.0),
        "r1": ExternalExecution("sc", "r1", coords["S1"], run_id="p3_detour", source="localized", frame_id="map", namespace_routes=True, timeout=6.0),
    }
    tasks = [
        Task("T0", "S0", "W0", release_step=0, due_step=250),
        Task("T1", "S1", "W1", release_step=0, due_step=250),
    ]
    ops = [
        OperationSnapshot("OP0", ("T0",), "W0", crew_id=1, tool_id=1, proc_steps=4, due_step=280, eligible_bays=("W0",), eligible_crews=(1,)),
        OperationSnapshot("OP1", ("T1",), "W1", crew_id=2, tool_id=2, proc_steps=4, due_step=280, eligible_bays=("W1",), eligible_crews=(2,)),
    ]
    sim = Simulator(
        net,
        tasks,
        [Agv("r0", max_speed=0.5, start_node="S0"), Agv("r1", max_speed=0.5, start_node="S1")],
        SpaceTimeAStarPlanner(max_speed=0.5, dt=0.5),
        GreedyAllocator(),
        SimulationConfig(dt=0.5, max_steps=1200, planning_horizon=600, load_time=0.5, unload_time=0.5, idle_retreat=False, deadlock_timeout=0, reschedule_policy=ReschedulePolicy.NONE, record_trajectory=True),
        executions=gates,
    )
    sim.reset()
    coord = PhysicalCVCRCoordinator(sim, operations=ops, policy_method="V13_KITTING_DET", budget_ms=25.0, epoch_interval=10, use_rust=False)
    positions = {r: g.start_xy for r, g in gates.items()}
    motion: dict[str, dict] = {}
    saw_detour_replan = False
    saw_reopen = False

    for step in range(450):
        now = float(step) * 0.5
        _advance_physical_motion(sim, positions, motion, now, fail_stop_robots=set(gates))
        # Nav2 local costmap blocks J0<->M_FAST from step 2..75 -> r0 detours around closed link via B1->B2
        blk = [CorridorObstacleReport(("J0", "M_FAST"), estimated_clear_horizon_steps=20)] if 2 <= step < 75 else []
        rep = coord.step_coordination(now, blocked_corridors=blk)
        if rep.replanned_detours:
            saw_detour_replan = True
        if rep.reopened_corridors:
            saw_reopen = True
        sim.step_once()
        # Every pin invariant must hold at all steps
        assert all(len(owners) >= 1 for group in sim._res.pins().values() for owners in group.values())
        if sim._all_done():
            while not coord.tracker.all_done(sim._step_cursor) and sim._step_cursor < sim.config.max_steps:
                sim._step_cursor += 1
                coord.tracker.step(sim._step_cursor)
            break

    assert sim._all_done()
    assert coord.tracker.all_done(sim._step_cursor)
    assert saw_detour_replan
    assert saw_reopen
    assert not sim._closed_corridors
    aud = IndependentAuditor.audit_simulation_run(sim, ops, require_trajectory=True)
    assert aud.pass_audit, f"P3 detour run failed audit: {aud.issues}"


def test_p3_physical_cvcr_nav2_degraded_soft_reassignment() -> None:
    """Verify PhysicalCVCRCoordinator reassigns unpicked task away from Nav2-degraded stationary AMR."""
    net, coords = _build_physical_bypass_network()
    gates = {
        "r0": ExternalExecution("sc", "r0", coords["S0"], run_id="p3_reassign", source="localized", frame_id="map", namespace_routes=True, timeout=1.0),
        "r1": ExternalExecution("sc", "r1", coords["S1"], run_id="p3_reassign", source="localized", frame_id="map", namespace_routes=True, timeout=6.0),
    }
    # Single task spawned near J0 (or S1) initially assigned to r0 at S0; r0 fails scan/receipt at step 1 -> r1 rescues
    tasks = [Task("T_RESCUE", "J0", "W1", release_step=0, due_step=200)]
    sim = Simulator(
        net,
        tasks,
        [Agv("r0", max_speed=0.5, start_node="S0"), Agv("r1", max_speed=0.5, start_node="S1")],
        SpaceTimeAStarPlanner(max_speed=0.5, dt=0.5),
        GreedyAllocator(),
        SimulationConfig(dt=0.5, max_steps=600, planning_horizon=300, load_time=0.5, unload_time=0.5, idle_retreat=False, deadlock_timeout=0, reschedule_policy=ReschedulePolicy.NONE, record_trajectory=True),
        executions=gates,
    )
    sim.reset()
    # Bootstrap both robots at t=0.0 cleanly (bound, idle)
    for rid, g in gates.items():
        sim.execution_feedback({**g.envelope(), "event": "POSE", "pose": [*g.start_xy, 0.0], "seq": 1, "timestamp": 0.0}, 0.0)

    # First let r0 claim T_RESCUE as stationary ASSIGNED (without dispatching hop yet via blocked gate test or immediate scan timeout after assignment)
    assert sim._try_assign(sim.agvs[0], tasks[0], 0)
    assert tasks[0].status == TaskStatus.ASSIGNED and tasks[0].assigned_agv == "r0"
    # Before r0 begins moving on edge, r0 scan/AMCL becomes stale -> timeout watchdog triggers fault (stationary, zero command)
    now_stale = 1.6
    sim.poll_execution(now_stale)
    assert gates["r0"].fault is not None and gates["r0"].command is None

    coord = PhysicalCVCRCoordinator(sim, operations=[], policy_method="V13_KITTING_DET", budget_ms=25.0, epoch_interval=6, use_rust=False)
    positions = {r: g.start_xy for r, g in gates.items()}
    motion: dict[str, dict] = {}
    saw_reassign = False
    for step in range(2, 260):
        now = float(step) * 0.5 + 1.0
        _advance_physical_motion(sim, positions, motion, now, degraded_robots={"r0"})
        telem = {
            "r0": Nav2RobotTelemetry("r0", speed_scale=0.1, local_planner_state="local_failed", receipt_age_s=2.5, fault_reason="amcl_scan_stale"),
            "r1": Nav2RobotTelemetry("r1", speed_scale=1.0, local_planner_state="nominal", receipt_age_s=0.05),
        }
        rep = coord.step_coordination(now, nav2_telemetry=telem)
        if "T_RESCUE" in rep.reassigned_tasks:
            saw_reassign = True
        sim.step_once()
        if sim._all_done():
            break

    assert saw_reassign
    assert sim._all_done()
    assert tasks[0].status == TaskStatus.COMPLETED
    assert tasks[0].assigned_agv == "r1"
    aud = IndependentAuditor.audit_simulation_run(sim, require_trajectory=True)
    assert aud.pass_audit, f"P3 soft reassignment audit failed: {aud.issues}"


def test_p3_build_sensor_fleet_and_external_handling_cvcr_closed_loop() -> None:
    """Drive the wheelset workshop build_sensor_fleet + ExternalHandling with PhysicalCVCRCoordinator."""
    sim = build_sensor_fleet(run_id="p3-workshop-fleet", case="delivery", handling_source="p3_contract")
    coord = PhysicalCVCRCoordinator(sim, operations=[], policy_method="V13_KITTING_DET", budget_ms=20.0, epoch_interval=15, use_rust=False)
    positions = {r: g.start_xy for r, g in sim.executions.items()}
    motion: dict[str, dict] = {}
    handling_done: set[str] = set()

    for step in range(3000):
        now = float(step) * 0.2
        _advance_physical_motion(sim, positions, motion, now)
        # Acknowledge equipment handoffs cleanly using strict ExternalHandling contract
        for rid, h in (sim.handlings or {}).items():
            if h.command is not None and not h.completed and h.command["handling_id"] not in handling_done:
                hid = h.command["handling_id"]
                handling_done.add(hid)
                cmd = h.command
                sup = rid if cmd["operation"] == "LOAD" else cmd["node_id"]
                ok = sim.handling_feedback(
                    {
                        **cmd,
                        "event": "HANDLING_COMPLETED",
                        "timestamp": now,
                        "robot_xy": list(positions[rid]),
                        "stopped": True,
                        "supported_by": sup,
                        "evidence_id": f"ev-{hid}",
                    },
                    now,
                    now,
                )
                assert ok["accepted"], f"Handling feedback rejected: {ok}"
        coord.step_coordination(now)
        sim.step_once()
        if sim._all_done():
            break

    assert sim._all_done()
    assert any(h.triggered for h in coord.history)
    assert any(h.receipt_accepted for h in coord.history)


def test_p3_recovery_backend_clears_nav2_active_state_on_detour() -> None:
    """Verify PhysicalCVCRCoordinator delegates resume/replan via recovery_backends so adapter state resets."""
    net, coords = _build_physical_bypass_network()
    gates = {
        "r0": ExternalExecution("sc", "r0", coords["S0"], run_id="p3_be", source="localized", frame_id="map", namespace_routes=True, timeout=6.0),
        "r1": ExternalExecution("sc", "r1", coords["S1"], run_id="p3_be", source="localized", frame_id="map", namespace_routes=True, timeout=6.0),
    }
    sim = Simulator(
        net,
        [Task("T0", "S0", "W0", release_step=0, due_step=250)],
        [Agv("r0", max_speed=0.5, start_node="S0"), Agv("r1", max_speed=0.5, start_node="S1")],
        SpaceTimeAStarPlanner(max_speed=0.5, dt=0.5),
        GreedyAllocator(),
        SimulationConfig(dt=0.5, max_steps=400, planning_horizon=300, load_time=0.5, unload_time=0.5, idle_retreat=False, deadlock_timeout=0, reschedule_policy=ReschedulePolicy.NONE),
        executions=gates,
    )
    sim.reset()

    class _MockNav2Backend:
        """Minimal stub mirroring FleetRobotNode.resume_execution/replan_execution contract."""

        def __init__(self, robot_id: str) -> None:
            self.robot_id = robot_id
            self.active = None
            self.resume_calls = 0
            self.replan_calls = 0

        def resume_execution(self, expected_route_id: str) -> dict:
            """Delegate resume_execution and clear adapter active action state."""
            self.resume_calls += 1
            ans = sim.resume_execution(expected_route_id, 10.0, robot_id=self.robot_id)
            if ans.get("accepted"):
                self.active = None
            return ans

        def replan_execution(self, expected_route_id: str) -> dict:
            """Delegate replan_execution and clear adapter active action state."""
            self.replan_calls += 1
            ans = sim.replan_execution(expected_route_id, 10.0, robot_id=self.robot_id)
            if ans.get("accepted"):
                self.active = None
            return ans

    backend = _MockNav2Backend("r0")
    coord = PhysicalCVCRCoordinator(sim, operations=[], policy_method="V13_KITTING_DET", budget_ms=20.0, epoch_interval=10, use_rust=False)
    coord.set_recovery_backends({"r0": backend})

    # Step until r0 receives MOVE along S0->J0, mark backend active, then close J0<->M_FAST -> resume/replan must clear active
    positions = {r: g.start_xy for r, g in gates.items()}
    motion: dict[str, dict] = {}
    saw_active = False
    for step in range(25):
        now = float(step) * 0.5
        _advance_physical_motion(sim, positions, motion, now, fail_stop_robots={"r0"})
        if not saw_active and gates["r0"].command is not None:
            backend.active = dict(gates["r0"].command)
            saw_active = True
        blk = [CorridorObstacleReport(("J0", "M_FAST"), estimated_clear_horizon_steps=20)] if (saw_active and step >= 3) else []
        coord.step_coordination(now, blocked_corridors=blk)
        sim.step_once()
        if (backend.resume_calls >= 1 or backend.replan_calls >= 1) and backend.active is None:
            break

    assert saw_active
    assert backend.resume_calls >= 1 or backend.replan_calls >= 1
    assert backend.active is None
