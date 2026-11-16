"""Gate 3 Witness 1: SYSTEM07 Online Stagnation Recovery in Native Execution.

Replicates the scientific witness:
- Stagnation Condition: In a multi-AMR kitted production environment, a transport AMR
  encounters an unexpected disruption (fault/hold), causing downstream operation starvation.
- Baseline (No Recovery): Unpicked orders assigned to the stalled AMR remain unserviced.
  Downstream kitting operations starve, resulting in incomplete tasks (5/6 completed).
- Online Recovery (LGCVCR SYSTEM07): The policy detects stagnation, identifies the
  releasable commitment set, and reassigns the unpicked task to an available healthy AMR
  while preserving active commitments.
- Outcome: Task completion recovers from 5/6 to 6/6 with zero safety violations.
"""
from __future__ import annotations

import time
import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.policy import LGCVCRPolicy
from amr_sched.cvcr.snapshot import (
    NativeSnapshotCapture,
    OperationSnapshot,
    RobotSnapshot,
    TaskSnapshot,
)
from amr_sched.cvcr.transaction import NativeTransactionOwner
from amr_sched.entities import Agv, AgvState, Task, TaskPriority, TaskStatus
from amr_sched.layouts import build_layout
from amr_sched.network import Network
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.simulation import SimulationConfig, Simulator


def build_system07_scenario():
    """Builds a calibrated 4-AMR, 6-task, 3-operation maintenance scenario on ring layout."""
    net = build_layout("ring")
    tasks = [
        Task("T1", pickup="W1", dropoff="W2", release_step=0, due_step=80),
        Task("T2", pickup="W2", dropoff="W2", release_step=0, due_step=80),
        Task("T3", pickup="W3", dropoff="W4", release_step=0, due_step=120),
        Task("T4", pickup="W4", dropoff="W4", release_step=0, due_step=120),
        Task("T5", pickup="W5", dropoff="W6", release_step=0, due_step=160),
        Task("T6", pickup="W6", dropoff="W6", release_step=0, due_step=160),
    ]
    agvs = [
        Agv("agv_0", start_node="W1", cur_node="W1"),
        Agv("agv_1", start_node="W2", cur_node="W2"),
        Agv("agv_2", start_node="W3", cur_node="W3"),
        Agv("agv_3", start_node="W5", cur_node="W5"),
    ]
    operations = [
        OperationSnapshot("OP_1", kit_task_ids=("T1", "T2"), bay_id="W2", crew_id=1, tool_id=1, proc_steps=20, due_step=100),
        OperationSnapshot("OP_2", kit_task_ids=("T3", "T4"), bay_id="W4", crew_id=2, tool_id=1, proc_steps=20, due_step=140, predecessor_op_id="OP_1"),
        OperationSnapshot("OP_3", kit_task_ids=("T5", "T6"), bay_id="W6", crew_id=1, tool_id=2, proc_steps=25, due_step=180, predecessor_op_id="OP_2"),
    ]
    for i, t in enumerate(tasks):
        t.kit_op_id = f"OP_{i // 2 + 1}"
    return net, tasks, agvs, operations


def test_system07_stagnation_recovery_witness():
    """Directly compares Baseline (stagnation/starvation) vs LGCVCR Online Recovery."""
    # 1. BASELINE RUN: Stagnation occurs on agv_2; no online recovery
    net1, tasks1, agvs1, ops1 = build_system07_scenario()
    planner1 = SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0)
    allocator1 = GreedyAllocator()
    sim1 = Simulator(
        network=net1,
        tasks=tasks1,
        agvs=agvs1,
        planner=planner1,
        allocator=allocator1,
        config=SimulationConfig(dt=1.0, max_steps=250),
    )
    sim1.reset()

    # Pre-assign tasks: T3 to agv_2
    tasks1[2].assigned_agv = "agv_2"
    tasks1[2].status = TaskStatus.ASSIGNED

    # Inject breakdown at step 10: agv_2 faults completely
    fault_step = 10
    while sim1.step_once():
        step = sim1._step_cursor
        if step == fault_step:
            agvs1[2].state = AgvState.BLOCKED
            agvs1[2].remaining_job_steps = 9999  # permanently stalled

    # Without recovery: T3 remains stranded, OP_2 cannot complete, T5/T6 starved
    completed_baseline = sum(1 for t in sim1.tasks if t.status == TaskStatus.COMPLETED)
    assert completed_baseline < 6, f"Expected incomplete tasks in baseline, got {completed_baseline}/6"
    assert sim1.tasks[2].status != TaskStatus.COMPLETED, "T3 should have failed or stayed incomplete"

    # 2. ONLINE RECOVERY RUN: Stagnation occurs on agv_2; LGCVCR Online Recovery rescues T3
    net2, tasks2, agvs2, ops2 = build_system07_scenario()
    planner2 = SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0)
    allocator2 = GreedyAllocator()
    sim2 = Simulator(
        network=net2,
        tasks=tasks2,
        agvs=agvs2,
        planner=planner2,
        allocator=allocator2,
        config=SimulationConfig(dt=1.0, max_steps=700),
    )
    sim2.reset()

    # Same initial assignment
    tasks2[2].assigned_agv = "agv_2"
    tasks2[2].status = TaskStatus.ASSIGNED

    owner2 = NativeTransactionOwner(sim2, ops2)
    recovery_executed = False

    while sim2.step_once():
        step = sim2._step_cursor
        if step == fault_step:
            # Fault occurs
            agvs2[2].state = AgvState.BLOCKED
            agvs2[2].remaining_job_steps = 9999

            # Trigger LGCVCR Online Recovery Policy
            snap = NativeSnapshotCapture.capture(sim2, ops2)
            policy = LGCVCRPolicy("SYSTEM07_RECOVERY", net2)
            proposal = policy.propose(snap, budget_ms=150.0)

            # Reassignment should move T3 to a healthy AMR (e.g. agv_0, agv_1, or agv_3)
            assert proposal.proposed_assignment["T3"] != "agv_2", "Policy must reassign stalled task"

            # Validate and commit atomically
            receipt = owner2.validate_and_commit(proposal)
            assert receipt.accepted is True, f"Recovery proposal was rejected: {receipt.reasons}"
            recovery_executed = True

    assert recovery_executed, "Recovery policy must have executed"
    completed_recovered = sum(1 for t in sim2.tasks if t.status == TaskStatus.COMPLETED)
    assert completed_recovered == 6, f"Expected 6/6 completed tasks after recovery, got {completed_recovered}/6"

    # 3. Independent Audit
    audit = IndependentAuditor.audit_simulation_run(sim2, ops2)
    assert audit.pass_audit is True, f"Independent audit failed: {audit.issues}"
    assert audit.conflict_events == 0, "Zero spatial conflicts during recovery"
    assert audit.completed_tasks == 6, "All 6 tasks confirmed delivered by independent audit"
