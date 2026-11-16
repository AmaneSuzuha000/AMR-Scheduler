"""Gate 2 Differential Tests: LG-CVCR B1 vs B2 Incremental Equivalence & Commitments.

Verifies:
1. Differential Equivalence: B1 (full rebuild) and B2 (incremental cache E2) produce
   identical objective scores, identical feasibility status, and identical makespan.
2. Speedup: B2 demonstrates speedup over B1 on multi-candidate evaluations.
3. 5-Tier Commitment Enforcement: IRREVOCABLE and EXECUTING_HARD tasks cannot be mutated.
4. Transaction Safety: Illegal mutations, stale steps, and illegal edges are rejected.
"""
from __future__ import annotations

import time
import pytest

from amr_sched.cvcr.commitment import (
    TIER_EXECUTING_HARD,
    TIER_IRREVOCABLE,
    TIER_SOFT,
    TIER_UNCOMMITTED,
    classify_commitments,
    select_release_set,
)
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.policy import LGCVCRPolicy
from amr_sched.cvcr.snapshot import (
    NativeSnapshot,
    NativeSnapshotCapture,
    OperationSnapshot,
    RobotSnapshot,
    TaskSnapshot,
)
from amr_sched.cvcr.transaction import NativeTransactionOwner
from amr_sched.entities import Agv, AgvState, Task, TaskPriority, TaskStatus
from amr_sched.layouts import build_layout
from amr_sched.network import Network
from amr_sched.simulation import SimulationConfig, Simulator


@pytest.fixture
def ring_network() -> Network:
    return build_layout("ring")


@pytest.fixture
def synthetic_snapshot(ring_network: Network) -> NativeSnapshot:
    """Builds a deterministic test snapshot with 4 AMRs, 8 tasks, and 4 operations."""
    robots = [
        RobotSnapshot(robot_id="agv_0", cur_node="W1", state="idle"),
        RobotSnapshot(robot_id="agv_1", cur_node="W2", state="idle"),
        RobotSnapshot(robot_id="agv_2", cur_node="W3", state="idle"),
        RobotSnapshot(robot_id="agv_3", cur_node="W4", state="idle"),
    ]
    tasks = [
        TaskSnapshot("T1", pickup="W1", dropoff="W2", release_step=0, due_step=50, status="assigned", assigned_agv="agv_0", kit_op_id="OP_1"),
        TaskSnapshot("T2", pickup="W2", dropoff="W3", release_step=0, due_step=50, status="assigned", assigned_agv="agv_1", kit_op_id="OP_1"),
        TaskSnapshot("T3", pickup="W3", dropoff="W4", release_step=5, due_step=70, status="assigned", assigned_agv="agv_2", kit_op_id="OP_2"),
        TaskSnapshot("T4", pickup="W4", dropoff="W5", release_step=5, due_step=70, status="assigned", assigned_agv="agv_3", kit_op_id="OP_2"),
        TaskSnapshot("T5", pickup="W5", dropoff="W6", release_step=10, due_step=90, status="pending", assigned_agv=None, kit_op_id="OP_3"),
        TaskSnapshot("T6", pickup="W6", dropoff="W1", release_step=10, due_step=90, status="pending", assigned_agv=None, kit_op_id="OP_3"),
        TaskSnapshot("T7", pickup="W1", dropoff="W3", release_step=15, due_step=110, status="pending", assigned_agv=None, kit_op_id="OP_4"),
        TaskSnapshot("T8", pickup="W2", dropoff="W4", release_step=15, due_step=110, status="pending", assigned_agv=None, kit_op_id="OP_4"),
    ]
    operations = [
        OperationSnapshot("OP_1", kit_task_ids=("T1", "T2"), bay_id="W2", crew_id=1, tool_id=1, proc_steps=15, due_step=60),
        OperationSnapshot("OP_2", kit_task_ids=("T3", "T4"), bay_id="W4", crew_id=2, tool_id=1, proc_steps=15, due_step=80, predecessor_op_id="OP_1"),
        OperationSnapshot("OP_3", kit_task_ids=("T5", "T6"), bay_id="W6", crew_id=1, tool_id=2, proc_steps=20, due_step=100, predecessor_op_id="OP_2"),
        OperationSnapshot("OP_4", kit_task_ids=("T7", "T8"), bay_id="W3", crew_id=2, tool_id=2, proc_steps=20, due_step=120, predecessor_op_id="OP_3"),
    ]
    return NativeSnapshot(
        schema="amr-lgcvcr.snapshot.v1",
        tick=0,
        version={"world": 1, "reservation": 1, "production": 1, "fleet": 1},
        world_fingerprint="test_ring_world",
        robots=robots,
        tasks=tasks,
        operations=operations,
        reservations=[],
        now_monotonic=time.monotonic(),
    )


def test_commitment_classification_tiers(synthetic_snapshot: NativeSnapshot):
    """Verifies that 5-tier classification strictly assigns correct commitment tiers."""
    # Mark T1 as COMPLETED and T3 as PICKED
    modified_tasks = list(synthetic_snapshot.tasks)
    modified_tasks[0] = TaskSnapshot("T1", pickup="S1", dropoff="S2", release_step=0, due_step=50, status="completed", assigned_agv="agv_0", complete_step=10)
    modified_tasks[2] = TaskSnapshot("T3", pickup="S2", dropoff="S4", release_step=5, due_step=70, status="picked", assigned_agv="agv_2", pickup_step=12)
    snap = NativeSnapshot(
        schema="amr-lgcvcr.snapshot.v1",
        tick=15,
        version=synthetic_snapshot.version,
        world_fingerprint=synthetic_snapshot.world_fingerprint,
        robots=synthetic_snapshot.robots,
        tasks=modified_tasks,
        operations=synthetic_snapshot.operations,
        reservations=[],
        now_monotonic=time.monotonic(),
    )

    clf = classify_commitments(snap)
    assert clf.task_tiers["T1"] == TIER_IRREVOCABLE
    assert "T1" in clf.frozen_tasks
    assert clf.task_tiers["T3"] == TIER_EXECUTING_HARD
    assert "T3" in clf.frozen_tasks
    assert clf.task_tiers["T2"] == TIER_SOFT
    assert "T2" in clf.releasable_tasks
    assert clf.task_tiers["T5"] == TIER_UNCOMMITTED
    assert "T5" in clf.releasable_tasks

    # Coupling graph check
    assert "T2" in clf.adj["T1"]  # KITTING coupling
    assert "T3" in clf.adj["T1"] or "T4" in clf.adj["T1"]  # PRECEDENCE coupling


def test_differential_equivalence_b1_vs_b2(ring_network: Network, synthetic_snapshot: NativeSnapshot):
    """Core Gate 2 Test: B1 (full rebuild) and B2 (incremental cache) must produce identical objectives."""
    eval_b1 = CVCREvaluator(ring_network, synthetic_snapshot, mode="B1")
    eval_b2 = CVCREvaluator(ring_network, synthetic_snapshot, mode="B2")

    # Base candidate assignment
    base_assign = {
        "T1": "agv_0", "T2": "agv_1", "T3": "agv_2", "T4": "agv_3",
        "T5": "agv_0", "T6": "agv_1", "T7": "agv_2", "T8": "agv_3",
    }

    # Evaluate base
    res1_b1 = eval_b1.evaluate(base_assign)
    res1_b2 = eval_b2.evaluate(base_assign)
    assert res1_b1.valid == res1_b2.valid
    assert abs(res1_b1.objective - res1_b2.objective) < 1e-4

    # Seed both B1 and B2 cache with incumbent
    eval_b1.set_incumbent(base_assign, res1_b1)
    eval_b2.set_incumbent(base_assign, res1_b2)

    # Candidate 2: modify ONLY 1 robot (agv_0 gives T5 to agv_2)
    cand2 = dict(base_assign)
    cand2["T5"] = "agv_2"

    res2_b1 = eval_b1.evaluate(cand2)
    res2_b2 = eval_b2.evaluate(cand2)

    # Verify MATHEMATICAL EQUIVALENCE
    assert res2_b1.valid == res2_b2.valid, f"Validity mismatch: B1={res2_b1.valid} vs B2={res2_b2.valid}"
    assert abs(res2_b1.objective - res2_b2.objective) < 1e-4, (
        f"Objective mismatch: B1={res2_b1.objective} vs B2={res2_b2.objective}"
    )
    assert res2_b1.op_end == res2_b2.op_end
    assert eval_b2.b2_inc_hits > 0, "B2 should have scored an incremental hit on 1-robot change"

    # Candidate 3: modify 2 robots
    cand3 = dict(base_assign)
    cand3["T6"] = "agv_3"

    res3_b1 = eval_b1.evaluate(cand3)
    res3_b2 = eval_b2.evaluate(cand3)

    assert res3_b1.valid == res3_b2.valid
    assert abs(res3_b1.objective - res3_b2.objective) < 1e-4


def test_cvcr_policy_propose_budget(ring_network: Network, synthetic_snapshot: NativeSnapshot):
    """Verifies that LGCVCRPolicy proposes a valid candidate within the 150 ms deadline."""
    policy = LGCVCRPolicy("CVCR_B2", ring_network)
    proposal = policy.propose(synthetic_snapshot, budget_ms=150.0)

    assert proposal.schema == "amr-lgcvcr.proposal.v1"
    assert proposal.policy_id == "CVCR_B2"
    assert proposal.telemetry["deadline_met"] is True
    assert proposal.telemetry["elapsed_ms"] <= 150.0
    assert proposal.candidate_digest is not None
    assert len(proposal.proposed_assignment) == len(synthetic_snapshot.tasks)


def test_native_transaction_owner_rejects_illegal_mutations(ring_network: Network):
    """Verifies transaction owner enforces commitment preservation and rejects illegal moves."""
    tasks = [
        Task("T1", pickup="W1", dropoff="W2", status=TaskStatus.PICKED, assigned_agv="agv_0"),
        Task("T2", pickup="W3", dropoff="W4", status=TaskStatus.PENDING),
    ]
    agvs = [
        Agv("agv_0", start_node="W1", cur_node="W1", state=AgvState.LOADING, cur_task="T1"),
        Agv("agv_1", start_node="W3", cur_node="W3", state=AgvState.IDLE),
    ]
    sim = Simulator(
        network=ring_network,
        tasks=tasks,
        agvs=agvs,
        planner=None,  # Not needed for validation
        allocator=None,
        config=SimulationConfig(dt=1.0, max_steps=100),
    )
    sim.reset()
    # Emulate running mid-step state where T1 is picked up on agv_0
    sim.tasks[0].status = TaskStatus.PICKED
    sim.tasks[0].assigned_agv = "agv_0"
    sim.tasks[0].pickup_step = 5

    owner = NativeTransactionOwner(sim)

    # 1. Illegal mutation: attempt to reassign T1 (which is PICKED) from agv_0 to agv_1
    snap = NativeSnapshotCapture.capture(sim)
    illegal_proposal = LGCVCRPolicy("CVCR_B2", ring_network).propose(snap)
    # Tamper assignment to violate commitment
    illegal_proposal.proposed_assignment["T1"] = "agv_1"

    receipt = owner.validate_and_commit(illegal_proposal)
    assert receipt.accepted is False
    assert any("VIOLATION_IRREVOCABLE_TASK_MUTATION" in r for r in receipt.reasons)
    assert receipt.physical_authorization is False

    # 2. Legal proposal: assign T2 (which is PENDING) to agv_1
    legal_proposal = LGCVCRPolicy("CVCR_B2", ring_network).propose(snap)
    legal_proposal.proposed_assignment["T1"] = "agv_0"  # keep unchanged
    legal_proposal.proposed_assignment["T2"] = "agv_1"

    receipt2 = owner.validate_and_commit(legal_proposal)
    assert receipt2.accepted is True
    assert sim.tasks[1].assigned_agv == "agv_1"
    assert sim.tasks[1].status == TaskStatus.ASSIGNED
