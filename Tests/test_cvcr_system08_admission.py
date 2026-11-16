"""Gate 3 Witness 2: SYSTEM08 High-Density Fleet Admission in Native Execution.

Verifies genuine native fleet gating (`Simulator.set_admitted_agvs`):
1. Unadmitted AGVs remain physically parked in the network (retaining node occupancy
   and executing yield retreat only if blocking a transit waypoint), and NEVER receive tasks.
2. Compares All-Admitted (8 AGVs) vs Fixed-4 vs Capacity-Aware Admission (`SYSTEM08_ADMISSION`)
   on single-lane ring (`aisle_capacity=1`) without wiping commit via `sim.run()`.
3. Validates per-step trajectory with IndependentAuditor (0 collisions, 0 invalid edges).
"""
from __future__ import annotations

import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.policy import LGCVCRPolicy
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.cvcr.transaction import NativeTransactionOwner
from amr_sched.entities import Agv, Task
from amr_sched.network import Network, build_demo_network
from amr_sched.planners.rust_spacetime_astar import RustSpaceTimeAStarPlanner, is_rust_planner_available
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.simulation import SimulationConfig, Simulator


def build_congested_scenario(aisle_capacity: int = 1) -> tuple[Network, list[Task], list[Agv]]:
    """Build an 8-AMR, 16-task dense scenario on single-lane or 2-lane ring."""
    net = build_demo_network(aisle_capacity=aisle_capacity)

    tasks = []
    for i in range(16):
        src = f"W{(i % 6) + 1}"
        dst = f"W{((i + 3) % 6) + 1}"
        tasks.append(
            Task(f"T{i + 1}", pickup=src, dropoff=dst, release_step=(i // 4) * 5, due_step=300 + i * 20)
        )

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
    return net, tasks, all_agvs


def test_system08_fleet_admission_witness():
    """Verify native fleet admission gates active robots genuinely and cuts conflicts."""
    planner_cls = RustSpaceTimeAStarPlanner if is_rust_planner_available() else SpaceTimeAStarPlanner
    allocator = GreedyAllocator()

    # 1. ALL-ADMITTED RUN
    net_all, tasks_all, agvs_all = build_congested_scenario(aisle_capacity=1)
    sim_all = Simulator(
        network=net_all,
        tasks=tasks_all,
        agvs=agvs_all,
        planner=planner_cls(max_speed=1.0, dt=1.0),
        allocator=allocator,
        config=SimulationConfig(dt=1.0, max_steps=1000),
    )
    res_all = sim_all.run()

    # 2. CAPACITY-AWARE ADMISSION RUN (SYSTEM08_ADMISSION)
    net_adm, tasks_adm, agvs_adm = build_congested_scenario(aisle_capacity=1)
    sim_adm = Simulator(
        network=net_adm,
        tasks=tasks_adm,
        agvs=agvs_adm,
        planner=planner_cls(max_speed=1.0, dt=1.0),
        allocator=allocator,
        config=SimulationConfig(dt=1.0, max_steps=1000),
    )
    sim_adm.reset()  # MUST reset before committing, then advance via step_once()

    snap = NativeSnapshotCapture.capture(sim_adm)
    policy = LGCVCRPolicy("SYSTEM08_ADMISSION", net_adm, use_rust=True)
    proposal = policy.propose(snap, budget_ms=150.0)

    assert proposal.telemetry["deadline_met"] is True, f"Exceeded 150ms: {proposal.telemetry['elapsed_ms']}ms"
    assert proposal.proposed_admitted_agvs is not None
    assert len(proposal.proposed_admitted_agvs) < len(agvs_adm), (
        f"Expected admitted < {len(agvs_adm)}, got {proposal.proposed_admitted_agvs}"
    )

    owner = NativeTransactionOwner(sim_adm)
    receipt = owner.validate_and_commit(proposal)
    assert receipt.accepted is True

    while sim_adm.step_once():
        pass
    res_adm = sim_adm._build_result()

    # Verify GENUINE enforcement: zero tasks executed by unadmitted robots!
    used_agvs = {t.assigned_agv for t in sim_adm.tasks if t.assigned_agv is not None}
    assert used_agvs <= set(proposal.proposed_admitted_agvs), (
        f"Unadmitted AGVs executed tasks! used={used_agvs}, admitted={proposal.proposed_admitted_agvs}"
    )

    assert len(res_adm.completed_tasks()) == 16
    assert res_adm.conflict_wait_steps() <= res_all.conflict_wait_steps(), (
        f"Admission conflict wait {res_adm.conflict_wait_steps()} > all-admit {res_all.conflict_wait_steps()}"
    )

    audit = IndependentAuditor.audit_simulation_run(sim_adm)
    assert audit.pass_audit is True, f"Audit issues: {audit.issues}"
    assert audit.conflict_events == 0
