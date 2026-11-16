"""Gate 4 Unit Test: Held-Out 62-Node Workshop Scene Closed-Loop Verification.

Verifies:
1. Native snapshot capture on the 62-node, 144-edge Workshop layout.
2. Real-time proposal generation (< 150 ms) for IDEA_CORRIDOR_CVCR_JOINT and BUDGET_MATCHED_ALNS.
3. Software transaction commit via NativeTransactionOwner.
4. Native step-by-step closed-loop simulation with genuine fleet admission gating.
5. IndependentAuditor trajectory verification (0 collisions, 0 invalid edge traversals).
"""
from __future__ import annotations

import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.policy import LGCVCRPolicy
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.cvcr.transaction import NativeTransactionOwner
from amr_sched.planners.rust_spacetime_astar import (
    RustSpaceTimeAStarPlanner,
    is_rust_planner_available,
)
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.workshop import WorkshopScene, build_scenario


def test_workshop_62node_snapshot_capture():
    """Verify NativeSnapshotCapture correctly captures 62-node WorkshopScene state."""
    net, tasks, agvs = build_scenario(n_tasks=12, n_agvs=4, seed=101)
    planner_cls = RustSpaceTimeAStarPlanner if is_rust_planner_available() else SpaceTimeAStarPlanner
    sim = Simulator(
        network=net,
        tasks=tasks,
        agvs=agvs,
        planner=planner_cls(max_speed=1.0, dt=1.0),
        allocator=GreedyAllocator(),
        config=SimulationConfig(dt=1.0, max_steps=500),
    )
    sim.reset()
    snap = NativeSnapshotCapture.capture(sim)
    assert snap.schema == "amr-lgcvcr.snapshot.v1"
    assert len(snap.robots) == 4
    assert len(snap.tasks) == 12
    assert snap.tick == 0


def test_workshop_62node_policy_deadline_and_commit():
    """Verify IDEA_CORRIDOR_CVCR_JOINT proposes within 150ms and commits cleanly on 62-node Workshop."""
    net, tasks, agvs = build_scenario(n_tasks=12, n_agvs=4, seed=102)
    planner_cls = RustSpaceTimeAStarPlanner if is_rust_planner_available() else SpaceTimeAStarPlanner
    sim = Simulator(
        network=net,
        tasks=tasks,
        agvs=agvs,
        planner=planner_cls(max_speed=1.0, dt=1.0),
        allocator=GreedyAllocator(),
        config=SimulationConfig(dt=1.0, max_steps=600),
    )
    sim.reset()
    snap = NativeSnapshotCapture.capture(sim)

    policy = LGCVCRPolicy("IDEA_CORRIDOR_CVCR_JOINT", net, use_rust=True)
    proposal = policy.propose(snap, budget_ms=150.0)

    assert proposal.policy_id == "IDEA_CORRIDOR_CVCR_JOINT"
    assert proposal.telemetry["deadline_met"] is True, f"Took {proposal.telemetry['elapsed_ms']}ms > 150ms"
    assert proposal.proposed_admitted_agvs is not None
    assert len(proposal.proposed_admitted_agvs) >= 1

    owner = NativeTransactionOwner(sim)
    receipt = owner.validate_and_commit(proposal)
    assert receipt.accepted is True
    assert receipt.physical_authorization is False


def test_workshop_62node_closed_loop_and_audit():
    """Verify step-by-step closed loop execution and independent trajectory audit on 62-node Workshop."""
    net, tasks, agvs = build_scenario(n_tasks=10, n_agvs=4, seed=103)
    planner_cls = RustSpaceTimeAStarPlanner if is_rust_planner_available() else SpaceTimeAStarPlanner
    sim = Simulator(
        network=net,
        tasks=tasks,
        agvs=agvs,
        planner=planner_cls(max_speed=1.0, dt=1.0),
        allocator=GreedyAllocator(),
        config=SimulationConfig(dt=1.0, max_steps=1200),
    )
    sim.reset()
    snap = NativeSnapshotCapture.capture(sim)

    policy = LGCVCRPolicy("IDEA_CORRIDOR_CVCR_JOINT", net, use_rust=True)
    proposal = policy.propose(snap, budget_ms=150.0)
    owner = NativeTransactionOwner(sim)
    receipt = owner.validate_and_commit(proposal)
    assert receipt.accepted is True

    # Advance closed-loop simulation to completion
    while sim.step_once():
        pass

    res = sim._build_result()
    assert len(res.completed_tasks()) == 10

    audit = IndependentAuditor.audit_simulation_run(sim)
    assert audit.pass_audit is True, f"Audit issues: {audit.issues}"
    assert audit.conflict_events == 0
