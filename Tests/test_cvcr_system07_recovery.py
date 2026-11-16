"""Gate 3 Witness 1: SYSTEM07 Online Stagnation Recovery in Native Execution.

Verifies with live NativeProductionTracker & Rust-B2:
- Baseline (no recovery): fault on agv_2 leaves T3 stranded (5/6 tasks, 1/3 ops).
- Recovery (`SYSTEM07_RECOVERY`): fault detected at step 10, T3 reassigned to healthy AGV,
  operations proceed respecting kitting & predecessors to 6/6 tasks & 3/3 ops,
  strictly within 150 ms budget and passing step-by-step IndependentAuditor.
"""
from __future__ import annotations

import time
import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.policy import LGCVCRPolicy
from amr_sched.cvcr.production import NativeProductionTracker
from amr_sched.cvcr.snapshot import NativeSnapshotCapture, OperationSnapshot
from amr_sched.cvcr.transaction import NativeTransactionOwner
from amr_sched.entities import Agv, AgvState, Task, TaskStatus
from amr_sched.layouts import build_layout
from amr_sched.planners.rust_spacetime_astar import RustSpaceTimeAStarPlanner, is_rust_planner_available
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.simulation import SimulationConfig, Simulator


def build_system07_scenario():
    """Build calibrated 4-AMR, 6-task, 3-operation maintenance scenario on ring layout."""
    net = build_layout("ring")
    tasks = [
        Task("T1", pickup="W1", dropoff="W2", release_step=0, due_step=120),
        Task("T2", pickup="W2", dropoff="W2", release_step=0, due_step=120),
        Task("T3", pickup="W3", dropoff="W4", release_step=0, due_step=240),
        Task("T4", pickup="W4", dropoff="W4", release_step=0, due_step=240),
        Task("T5", pickup="W5", dropoff="W6", release_step=0, due_step=360),
        Task("T6", pickup="W6", dropoff="W6", release_step=0, due_step=360),
    ]
    agvs = [
        Agv("agv_0", start_node="W1", cur_node="W1"),
        Agv("agv_1", start_node="W2", cur_node="W2"),
        Agv("agv_2", start_node="W3", cur_node="W3"),
        Agv("agv_3", start_node="W5", cur_node="W5"),
    ]
    operations = [
        OperationSnapshot("OP_1", kit_task_ids=("T1", "T2"), bay_id="W2", crew_id=1, tool_id=1, proc_steps=20, due_step=150),
        OperationSnapshot("OP_2", kit_task_ids=("T3", "T4"), bay_id="W4", crew_id=2, tool_id=1, proc_steps=20, due_step=380, predecessor_op_id="OP_1"),
        OperationSnapshot("OP_3", kit_task_ids=("T5", "T6"), bay_id="W6", crew_id=1, tool_id=2, proc_steps=25, due_step=520, predecessor_op_id="OP_2"),
    ]
    for i, t in enumerate(tasks):
        t.kit_op_id = f"OP_{i // 2 + 1}"
    return net, tasks, agvs, operations


def test_system07_stagnation_recovery_witness():
    """Compare Baseline (stagnation/starvation) vs LGCVCR Online Recovery."""
    planner_cls = RustSpaceTimeAStarPlanner if is_rust_planner_available() else SpaceTimeAStarPlanner

    # 1. BASELINE RUN
    net1, tasks1, agvs1, ops1 = build_system07_scenario()
    sim1 = Simulator(
        network=net1,
        tasks=tasks1,
        agvs=agvs1,
        planner=planner_cls(max_speed=1.0, dt=1.0),
        allocator=GreedyAllocator(),
        config=SimulationConfig(dt=1.0, max_steps=600),
    )
    prod1 = NativeProductionTracker(ops1, tasks1)
    sim1.reset()
    tasks1[2].assigned_agv = "agv_2"
    tasks1[2].status = TaskStatus.ASSIGNED

    fault_step = 10
    while sim1.step_once():
        step = sim1._step_cursor
        prod1.step(step)
        if step == fault_step:
            agvs1[2].state = AgvState.BLOCKED
            sim1.set_agv_faulted("agv_2", True)

    completed_baseline = sum(1 for t in sim1.tasks if t.status == TaskStatus.COMPLETED)
    assert completed_baseline < 6, f"Expected incomplete tasks in baseline, got {completed_baseline}/6"
    assert prod1.metrics(sim1._step_cursor).completed_ops < 3  # downstream ops starved waiting for T3

    # 2. ONLINE RECOVERY RUN
    net2, tasks2, agvs2, ops2 = build_system07_scenario()
    sim2 = Simulator(
        network=net2,
        tasks=tasks2,
        agvs=agvs2,
        planner=planner_cls(max_speed=1.0, dt=1.0),
        allocator=GreedyAllocator(),
        config=SimulationConfig(dt=1.0, max_steps=700),
    )
    prod2 = NativeProductionTracker(ops2, tasks2)
    sim2.reset()
    tasks2[2].assigned_agv = "agv_2"
    tasks2[2].status = TaskStatus.ASSIGNED

    owner2 = NativeTransactionOwner(sim2, ops2)
    recovery_executed = False
    recovery_elapsed_ms = 9999.0

    while sim2.step_once():
        step = sim2._step_cursor
        prod2.step(step)
        if step == fault_step:
            agvs2[2].state = AgvState.BLOCKED
            sim2.set_agv_faulted("agv_2", True)

            snap = NativeSnapshotCapture.capture(sim2, ops2)
            policy = LGCVCRPolicy("SYSTEM07_RECOVERY", net2, use_rust=True)
            proposal = policy.propose(snap, budget_ms=150.0)
            recovery_elapsed_ms = proposal.telemetry["elapsed_ms"]

            assert proposal.proposed_assignment["T3"] != "agv_2"
            receipt = owner2.validate_and_commit(proposal)
            assert receipt.accepted is True, f"Rejected: {receipt.reasons}"
            recovery_executed = True

    # Finish remaining production ops after last transport task completes
    end_tick = sim2._step_cursor
    for _ in range(100):
        if prod2.all_done(end_tick):
            break
        end_tick += 1
        prod2.step(end_tick)

    assert recovery_executed is True
    assert recovery_elapsed_ms <= 150.0, f"Exceeded 150ms budget: {recovery_elapsed_ms}ms"
    assert sum(1 for t in sim2.tasks if t.status == TaskStatus.COMPLETED) == 6
    assert prod2.metrics(end_tick).completed_ops == 3

    audit = IndependentAuditor.audit_simulation_run(sim2, ops2)
    assert audit.pass_audit is True, f"Audit issues: {audit.issues}"
    assert audit.conflict_events == 0
    assert audit.invalid_edge_moves == 0
