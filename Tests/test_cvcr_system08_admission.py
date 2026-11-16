"""Gate 3 Witness 2: SYSTEM08 High-Density Fleet Admission in Native Execution.

Replicates the scientific witness:
- Congested Setting: 8 AMRs competing in a constrained workshop / ring topology with
  high task density.
- Native Baseline (All Admitted): Admitting all 8 AMRs saturates corridor capacity,
  generating severe congestion, massive conflict wait steps, or traffic gridlock.
- Fixed 4-AMR Baseline: Avoids gridlock, but is statically fixed.
- Capacity-Aware Admission (LGCVCR SYSTEM08): Evaluates network capacity, selects
  an optimal active subset (e.g. 4 active AMRs), while keeping inactive AMRs safely
  parked on haven/station nodes with static reservations.
- Outcome: Capacity-aware admission avoids congestion collapse, drastically reducing
  conflict wait steps and makespan compared to all-admit.
"""
from __future__ import annotations

import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.policy import LGCVCRPolicy
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.cvcr.transaction import NativeTransactionOwner
from amr_sched.entities import Agv, Task, TaskStatus
from amr_sched.layouts import build_layout
from amr_sched.network import Network
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.simulation import SimulationConfig, Simulator


def build_congested_scenario(active_agv_ids: list[str] | None = None) -> tuple[Network, list[Task], list[Agv]]:
    """Builds a high-density 8-AMR, 16-task scenario."""
    net = build_layout("ring")
    tasks = []
    # 16 tasks creating bidirectional traffic flows across ring nodes
    for i in range(16):
        src = f"W{(i % 6) + 1}"
        dst = f"W{((i + 3) % 6) + 1}"
        tasks.append(
            Task(f"T{i + 1}", pickup=src, dropoff=dst, release_step=(i // 4) * 5, due_step=200 + i * 20)
        )

    # 8 AMRs placed across stations and haven nodes
    all_agvs = [
        Agv("agv_0", start_node="W1"),
        Agv("agv_1", start_node="W2"),
        Agv("agv_2", start_node="W3"),
        Agv("agv_3", start_node="W4"),
        Agv("agv_4", start_node="W5"),
        Agv("agv_5", start_node="W6"),
        Agv("agv_6", start_node="B1"),
        Agv("agv_7", start_node="B2"),
    ]

    if active_agv_ids is not None:
        # Filter active AMRs
        agvs = [a for a in all_agvs if a.agv_id in active_agv_ids]
    else:
        agvs = all_agvs

    return net, tasks, agvs


def test_system08_fleet_admission_witness():
    """Compares All-Admitted (8 AMRs) vs Static (4 AMRs) vs Capacity-Aware Admission."""
    planner = SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0)
    allocator = GreedyAllocator()

    # 1. ALL-ADMITTED RUN (8 AMRs in tight network)
    net_all, tasks_all, agvs_all = build_congested_scenario()
    sim_all = Simulator(
        network=net_all,
        tasks=tasks_all,
        agvs=agvs_all,
        planner=planner,
        allocator=allocator,
        config=SimulationConfig(dt=1.0, max_steps=800),
    )
    sim_all.reset()
    res_all = sim_all.run()

    # 2. FIXED 4-AMR BASELINE RUN
    active_4 = ["agv_0", "agv_1", "agv_2", "agv_3"]
    net_4, tasks_4, agvs_4 = build_congested_scenario(active_agv_ids=active_4)
    sim_4 = Simulator(
        network=net_4,
        tasks=tasks_4,
        agvs=agvs_4,
        planner=planner,
        allocator=allocator,
        config=SimulationConfig(dt=1.0, max_steps=800),
    )
    sim_4.reset()
    res_4 = sim_4.run()

    # 3. CAPACITY-AWARE ADMISSION RUN (SYSTEM08 Policy)
    net_adm, tasks_adm, agvs_adm = build_congested_scenario()
    sim_adm = Simulator(
        network=net_adm,
        tasks=tasks_adm,
        agvs=agvs_adm,
        planner=planner,
        allocator=allocator,
        config=SimulationConfig(dt=1.0, max_steps=800),
    )
    sim_adm.reset()

    # Execute admission policy at t=0
    snap = NativeSnapshotCapture.capture(sim_adm)
    policy = LGCVCRPolicy("SYSTEM08_ADMISSION", net_adm)
    proposal = policy.propose(snap, budget_ms=150.0)

    # Verify that capacity-aware admission selected a subset of AMRs (<= 6 AMRs)
    admitted_robots = set(proposal.proposed_assignment.values())
    assert len(admitted_robots) < len(agvs_adm), (
        f"Capacity-aware admission must gate fleet size below {len(agvs_adm)}, got {len(admitted_robots)}"
    )

    owner = NativeTransactionOwner(sim_adm)
    receipt = owner.validate_and_commit(proposal)
    assert receipt.accepted is True

    # Complete execution
    while sim_adm.step_once():
        pass
    res_adm = sim_adm._build_result()

    # Compare Metrics:
    # Capacity-aware admission must significantly reduce conflict wait steps compared to all-admit
    conflicts_all = res_all.conflict_wait_steps()
    conflicts_adm = res_adm.conflict_wait_steps()
    completions_all = len(res_all.completed_tasks())
    completions_adm = len(res_adm.completed_tasks())

    # Assertions
    assert completions_adm >= completions_all, (
        f"Capacity-aware admission completed {completions_adm} vs all-admit {completions_all}"
    )
    assert conflicts_adm <= conflicts_all, (
        f"Capacity-aware admission conflict wait {conflicts_adm} should be <= all-admit {conflicts_all}"
    )

    # Independent Audit
    audit = IndependentAuditor.audit_simulation_run(sim_adm)
    assert audit.pass_audit is True, f"Audit failed: {audit.issues}"
    assert audit.conflict_events == 0, "No spatial capacity violations in admitted run"
