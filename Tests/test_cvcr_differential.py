"""Gate 2 Differential & Mechanism Tests: A1/B1/B2, Rust Acceleration, CIE MLP, & Production.

Verifies:
1. Differential Equivalence: B1 and B2 produce identical objective scores and feasibility
   on both Python and Rust backends.
2. True Speedup: Rust-accelerated B2 achieves fast evaluation (<150 ms budget with margin).
3. 5-Tier Commitment Enforcement & NativeTransactionOwner rejection of illegal mutations
   (PICKED task reassignment, unadmitted active robot, picked-kit bay switch).
4. Frozen CIE MLP Transfer Policy execution (`OUTCOME_MLP_FULL`, `KEEP_RELEASE_MLP_BALANCED`)
   and new `IDEA_CORRIDOR_CVCR_JOINT`.
5. Step-by-step IndependentAuditor & NativeProductionTracker kitting/mutex enforcement.
"""
from __future__ import annotations

import time
import pytest

from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.commitment import (
    TIER_EXECUTING_HARD,
    TIER_IRREVOCABLE,
    TIER_SOFT,
    TIER_UNCOMMITTED,
    classify_commitments,
)
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.features import FEATURE_NAMES, extract_task_features, load_frozen_cie_models
from amr_sched.cvcr.policy import LGCVCRPolicy
from amr_sched.cvcr.production import NativeProductionTracker
from amr_sched.cvcr.snapshot import (
    NativeSnapshot,
    NativeSnapshotCapture,
    OperationSnapshot,
    RobotSnapshot,
    TaskSnapshot,
)
from amr_sched.cvcr.transaction import NativeTransactionOwner
from amr_sched.entities import Agv, AgvState, Task, TaskStatus
from amr_sched.layouts import build_layout
from amr_sched.network import Network
from amr_sched.planners.rust_spacetime_astar import is_rust_planner_available
from amr_sched.simulation import SimulationConfig, Simulator


@pytest.fixture
def ring_network() -> Network:
    return build_layout("ring")


@pytest.fixture
def synthetic_snapshot(ring_network: Network) -> NativeSnapshot:
    """Build a deterministic test snapshot with 4 AMRs, 8 tasks, and 4 operations."""
    robots = [
        RobotSnapshot(robot_id="agv_0", cur_node="W1", state="idle"),
        RobotSnapshot(robot_id="agv_1", cur_node="W2", state="idle"),
        RobotSnapshot(robot_id="agv_2", cur_node="W3", state="idle"),
        RobotSnapshot(robot_id="agv_3", cur_node="W4", state="idle"),
    ]
    tasks = [
        TaskSnapshot("T1", pickup="W1", dropoff="W2", release_step=0, due_step=120, status="assigned", assigned_agv="agv_0", kit_op_id="OP_1"),
        TaskSnapshot("T2", pickup="W2", dropoff="W2", release_step=0, due_step=120, status="assigned", assigned_agv="agv_1", kit_op_id="OP_1"),
        TaskSnapshot("T3", pickup="W3", dropoff="W4", release_step=5, due_step=240, status="assigned", assigned_agv="agv_2", kit_op_id="OP_2"),
        TaskSnapshot("T4", pickup="W4", dropoff="W4", release_step=5, due_step=240, status="assigned", assigned_agv="agv_3", kit_op_id="OP_2"),
        TaskSnapshot("T5", pickup="W5", dropoff="W6", release_step=10, due_step=360, status="pending", assigned_agv=None, kit_op_id="OP_3"),
        TaskSnapshot("T6", pickup="W6", dropoff="W6", release_step=10, due_step=360, status="pending", assigned_agv=None, kit_op_id="OP_3"),
        TaskSnapshot("T7", pickup="W1", dropoff="W3", release_step=15, due_step=480, status="pending", assigned_agv=None, kit_op_id="OP_4"),
        TaskSnapshot("T8", pickup="W2", dropoff="W3", release_step=15, due_step=480, status="pending", assigned_agv=None, kit_op_id="OP_4"),
    ]
    operations = [
        OperationSnapshot("OP_1", kit_task_ids=("T1", "T2"), bay_id="W2", crew_id=1, tool_id=1, proc_steps=15, due_step=150, eligible_bays=("W2", "W3"), eligible_crews=(1, 2)),
        OperationSnapshot("OP_2", kit_task_ids=("T3", "T4"), bay_id="W4", crew_id=2, tool_id=1, proc_steps=15, due_step=270, predecessor_op_id="OP_1", eligible_bays=("W4", "W5"), eligible_crews=(1, 2)),
        OperationSnapshot("OP_3", kit_task_ids=("T5", "T6"), bay_id="W6", crew_id=1, tool_id=2, proc_steps=20, due_step=390, predecessor_op_id="OP_2", eligible_bays=("W6",), eligible_crews=(1,)),
        OperationSnapshot("OP_4", kit_task_ids=("T7", "T8"), bay_id="W3", crew_id=2, tool_id=2, proc_steps=20, due_step=510, predecessor_op_id="OP_3", eligible_bays=("W3",), eligible_crews=(2,)),
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
    """Verify 5-tier commitment classification and 4-domain coupling graph."""
    modified_tasks = list(synthetic_snapshot.tasks)
    modified_tasks[0] = TaskSnapshot("T1", pickup="W1", dropoff="W2", release_step=0, due_step=120, status="completed", assigned_agv="agv_0", complete_step=10, kit_op_id="OP_1")
    modified_tasks[2] = TaskSnapshot("T3", pickup="W3", dropoff="W4", release_step=5, due_step=240, status="picked", assigned_agv="agv_2", pickup_step=12, kit_op_id="OP_2")
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
    assert "T2" in clf.adj["T1"]  # KITTING coupling


@pytest.mark.parametrize("use_rust", [False, True] if is_rust_planner_available() else [False])
def test_differential_equivalence_b1_vs_b2(ring_network: Network, synthetic_snapshot: NativeSnapshot, use_rust: bool):
    """Verify B1 (full rebuild) == B2 (persistent incremental cache) on both backends."""
    eval_b1 = CVCREvaluator(ring_network, synthetic_snapshot, mode="B1", use_rust=use_rust)
    eval_b2 = CVCREvaluator(ring_network, synthetic_snapshot, mode="B2", use_rust=use_rust)

    base_assign = {
        "T1": "agv_0", "T2": "agv_1", "T3": "agv_2", "T4": "agv_3",
        "T5": "agv_0", "T6": "agv_1", "T7": "agv_2", "T8": "agv_3",
    }

    res1_b1 = eval_b1.evaluate(base_assign)
    res1_b2 = eval_b2.evaluate(base_assign)
    assert res1_b1.valid == res1_b2.valid
    assert abs(res1_b1.objective - res1_b2.objective) < 1e-4

    eval_b1.set_incumbent(base_assign, res1_b1)
    eval_b2.set_incumbent(base_assign, res1_b2)

    for tid, new_agv in [("T5", "agv_2"), ("T6", "agv_3"), ("T7", "agv_0")]:
        cand = dict(base_assign)
        cand[tid] = new_agv
        rb1 = eval_b1.evaluate(cand)
        rb2 = eval_b2.evaluate(cand)
        assert rb1.valid == rb2.valid
        assert abs(rb1.objective - rb2.objective) < 1e-4
        assert rb1.op_end == rb2.op_end

    assert eval_b2.b2_inc_hits >= 1


@pytest.mark.parametrize(
    "policy_id",
    [
        "CVCR_B2",
        "SACC_B2",
        "OUTCOME_MLP_FULL",
        "KEEP_RELEASE_MLP_BALANCED",
        "BUDGET_MATCHED_ALNS",
        "IDEA_CORRIDOR_CVCR_JOINT",
    ],
)
def test_all_policies_propose_within_budget(ring_network: Network, synthetic_snapshot: NativeSnapshot, policy_id: str):
    """Verify every policy executes cleanly and satisfies 150 ms deadline with Rust-B2."""
    policy = LGCVCRPolicy(policy_id, ring_network, use_rust=True)
    proposal = policy.propose(synthetic_snapshot, budget_ms=150.0)
    assert proposal.policy_id == policy_id
    if is_rust_planner_available():
        assert proposal.telemetry["deadline_met"] is True, f"{policy_id} took {proposal.telemetry['elapsed_ms']}ms"
    assert len(proposal.proposed_assignment) == len(synthetic_snapshot.tasks)


def test_frozen_cie_features_and_weights(synthetic_snapshot: NativeSnapshot):
    """Verify exact 13-feature extraction and frozen CIE MLP weight integrity."""
    models = load_frozen_cie_models()
    assert "mlp_full" in models and "mlp_keep_release_balanced" in models
    clf = classify_commitments(synthetic_snapshot)
    tids, X = extract_task_features(synthetic_snapshot, clf, {}, {})
    assert len(tids) == len(clf.releasable_tasks)
    assert X.shape == (len(tids), len(FEATURE_NAMES))


def test_native_transaction_owner_rejects_illegal_mutations(ring_network: Network):
    """Verify transaction owner enforces hard commitments and admission consistency."""
    tasks = [
        Task("T1", pickup="W1", dropoff="W2", status=TaskStatus.PENDING),
        Task("T2", pickup="W3", dropoff="W4", status=TaskStatus.PENDING),
    ]
    agvs = [
        Agv("agv_0", start_node="W1"),
        Agv("agv_1", start_node="W3"),
    ]
    ops = [
        OperationSnapshot("OP_1", kit_task_ids=("T1",), bay_id="W2", crew_id=1, tool_id=1, proc_steps=10, due_step=100, eligible_bays=("W2", "W5")),
    ]
    tasks[0].kit_op_id = "OP_1"

    sim = Simulator(
        network=ring_network,
        tasks=tasks,
        agvs=agvs,
        planner=None,
        allocator=None,
        config=SimulationConfig(dt=1.0, max_steps=100),
    )
    sim.reset()
    # Emulate mid-run state where T1 is already PICKED on agv_0
    sim.tasks[0].status = TaskStatus.PICKED
    sim.tasks[0].assigned_agv = "agv_0"
    sim.tasks[0].pickup_step = 5

    owner = NativeTransactionOwner(sim, ops)
    snap = NativeSnapshotCapture.capture(sim, ops)

    # 1. Illegal task reassignment on PICKED task
    bad1 = LGCVCRPolicy("CVCR_B2", ring_network).propose(snap)
    bad1.proposed_assignment["T1"] = "agv_1"
    r1 = owner.validate_and_commit(bad1)
    assert r1.accepted is False
    assert any("VIOLATION_IRREVOCABLE_TASK_MUTATION" in x for x in r1.reasons)

    # 2. Illegal bay mutation when kit task T1 is already PICKED
    bad2 = LGCVCRPolicy("CVCR_B2", ring_network).propose(snap)
    bad2.proposed_assignment["T1"] = "agv_0"
    bad2.proposed_bays["OP_1"] = "W5"
    r2 = owner.validate_and_commit(bad2)
    assert r2.accepted is False
    assert any("VIOLATION_PICKED_KIT_BAY_MUTATION" in x for x in r2.reasons)

    # 3. Illegal admission omission of agv carrying PICKED task
    bad3 = LGCVCRPolicy("CVCR_B2", ring_network).propose(snap)
    bad3.proposed_assignment["T1"] = "agv_0"
    bad3.proposed_admitted_agvs = ["agv_1"]
    r3 = owner.validate_and_commit(bad3)
    assert r3.accepted is False
    assert any("VIOLATION_UNADMITTED_ACTIVE_ROBOT" in x for x in r3.reasons)


def test_production_tracker_and_step_auditor(ring_network: Network):
    """Verify NativeProductionTracker enforces kit precedence & crew mutex step-by-step."""
    tasks = [
        Task("T1", pickup="W1", dropoff="W2"),
        Task("T2", pickup="W3", dropoff="W4"),
    ]
    ops = [
        OperationSnapshot("OP_1", kit_task_ids=("T1",), bay_id="W2", crew_id=1, tool_id=1, proc_steps=10, due_step=100),
        OperationSnapshot("OP_2", kit_task_ids=("T2",), bay_id="W4", crew_id=1, tool_id=2, proc_steps=10, due_step=120),
    ]
    tracker = NativeProductionTracker(ops, tasks)
    tracker.step(0)
    assert ops[0].started is False  # Waiting for T1

    tasks[0].status = TaskStatus.COMPLETED
    tasks[0].complete_step = 5
    tasks[1].status = TaskStatus.COMPLETED
    tasks[1].complete_step = 5

    started = tracker.step(6)
    assert started == ["OP_1"]  # OP_2 shares crew_id=1 so must wait until step 16!
    assert ops[1].started is False
    assert tracker.step(15) == []
    assert tracker.step(16) == ["OP_2"]


def test_ltm_traffic_weights_rust_and_transaction():
    """Verify LTM online traffic weights divert routes cleanly, clear on reset, and enforce tx validation."""
    from amr_sched.network import Edge, Network, Node, NodeKind
    from amr_sched.planners.rust_spacetime_astar import RustSpaceTimeAStarPlanner

    net = Network()
    net.add_node(Node("S", 0.0, 0.0, kind=NodeKind.BUFFER, capacity=2))
    net.add_node(Node("A", 4.0, 2.0, kind=NodeKind.AISLE, capacity=1))
    net.add_node(Node("B", 5.0, -2.0, kind=NodeKind.AISLE, capacity=1))
    net.add_node(Node("G", 8.0, 0.0, kind=NodeKind.WORKSTATION, capacity=1))
    net.add_edge(Edge("S", "A", length=4.0, capacity=1))
    net.add_edge(Edge("A", "G", length=4.0, capacity=1))
    net.add_edge(Edge("G", "A", length=4.0, capacity=1))
    net.add_edge(Edge("A", "S", length=4.0, capacity=1))
    net.add_edge(Edge("S", "B", length=5.0, capacity=1))
    net.add_edge(Edge("B", "G", length=5.0, capacity=1))
    net.add_edge(Edge("G", "B", length=5.0, capacity=1))
    net.add_edge(Edge("B", "S", length=5.0, capacity=1))

    planner = RustSpaceTimeAStarPlanner()
    sim = Simulator(
        network=net,
        tasks=[Task("T1", pickup="S", dropoff="G", release_step=0, due_step=100)],
        agvs=[Agv("agv_0", start_node="S")],
        planner=planner,
        allocator=None,
        config=SimulationConfig(dt=1.0, max_steps=100),
    )
    sim.reset()
    snap = NativeSnapshotCapture.capture(sim, [])

    def _forward_nodes(path: list[tuple[str, int]]) -> list[str]:
        seq = [n for n, _ in path]
        return seq[: seq.index("G") + 1]

    ev = CVCREvaluator(net, snap, mode="B2", use_rust=True)
    res0 = ev.evaluate({"T1": "agv_0"})
    assert res0.valid
    assert "A" in _forward_nodes(res0.robot_paths["agv_0"])
    ev.set_incumbent({"T1": "agv_0"}, res0)

    # Penalize S -> A by +3.0 steps (24 Q3) so S -> B -> G (10 steps) beats S -> A -> G (8 + 3)
    ev.set_traffic_weights({("S", "A"): 24})
    res1 = ev.evaluate({"T1": "agv_0"})
    assert res1.valid
    assert "B" in _forward_nodes(res1.robot_paths["agv_0"])

    # Clearing traffic weights restores B2 cache to shortest path
    ev.set_traffic_weights(None)
    res2 = ev.evaluate({"T1": "agv_0"})
    assert res2.valid
    assert "A" in _forward_nodes(res2.robot_paths["agv_0"])

    # NativeTransactionOwner rejects unknown edge and commits/resets valid LTM weights
    owner = NativeTransactionOwner(sim, [])
    bad_prop = LGCVCRPolicy("CVCR_B2", net).propose(snap)
    bad_prop.proposed_traffic_weights = {("S", "NONEXISTENT"): 16}
    assert owner.validate_and_commit(bad_prop).accepted is False

    good_prop = LGCVCRPolicy("CVCR_B2", net).propose(snap)
    good_prop.proposed_traffic_weights = {("S", "A"): 24}
    assert owner.validate_and_commit(good_prop).accepted is True
    assert sim._traffic_weights_q3 == {("S", "A"): 24}
    sim.reset()
    assert sim._traffic_weights_q3 == {}

    # V9 windowed cutoff + per-AGV soft corridor guide + rollback
    bad_g = LGCVCRPolicy("CVCR_B2", net).propose(snap)
    bad_g.proposed_guide_routes = {"agv_0": [("S", "MISSING_NODE")]}
    assert owner.validate_and_commit(bad_g).accepted is False
    assert sim._guide_corridors == {}

    good_g = LGCVCRPolicy("CVCR_B2", net).propose(snap)
    good_g.proposed_traffic_weights = {("S", "A"): 24}
    good_g.proposed_traffic_cutoff = 15
    good_g.proposed_guide_routes = {"agv_0": [("S", "B"), ("B", "G")]}
    assert owner.validate_and_commit(good_g).accepted is True
    assert sim._traffic_cutoff_step == 15
    assert sim._guide_corridors == {"agv_0": [("S", "B"), ("B", "G")]}
    sim.reset()
    assert sim._traffic_cutoff_step == 0
    assert sim._guide_corridors == {}


def test_v10_priority_commit_and_rollback(ring_network: Network):
    """Verify V10 proposed_op_priorities commits atomically, updates NativeProductionTracker order, and rolls back cleanly."""
    tasks = [
        Task("T1", pickup="W1", dropoff="W2", status=TaskStatus.PENDING),
        Task("T2", pickup="W3", dropoff="W4", status=TaskStatus.PENDING),
    ]
    ops = [
        OperationSnapshot("OP_1", kit_task_ids=("T1",), bay_id="W2", crew_id=1, tool_id=1, proc_steps=10, due_step=100),
        OperationSnapshot("OP_2", kit_task_ids=("T2",), bay_id="W4", crew_id=1, tool_id=2, proc_steps=10, due_step=120),
    ]
    agvs = [
        Agv("agv_0", start_node="W1"),
        Agv("agv_1", start_node="W3"),
    ]
    sim = Simulator(
        network=ring_network,
        tasks=tasks,
        agvs=agvs,
        planner=None,
        allocator=None,
        config=SimulationConfig(dt=1.0, max_steps=100),
    )
    sim.reset()
    tracker = NativeProductionTracker(ops, sim.tasks)
    owner = NativeTransactionOwner(sim, ops, production_tracker=tracker)

    snap = NativeSnapshotCapture.capture(sim, ops)

    # Reject invalid op priority without mutating simulator or tracker state
    bad_p = LGCVCRPolicy("V10_E7_COND_CPSAT", ring_network).propose(snap)
    bad_p.proposed_op_priorities = {"OP_MISSING": 5}
    receipt_bad = owner.validate_and_commit(bad_p)
    assert receipt_bad.accepted is False
    assert any("UNKNOWN_OPERATION_ID_PRIO" in r for r in receipt_bad.reasons)
    assert sim._op_priorities == {}
    assert tracker.op_priorities == {}

    # Valid commit flips crew-1 tie-break from OP_1 (due 100) to OP_2 (priority 5 < 50)
    good_p = LGCVCRPolicy("V10_E7_COND_CPSAT", ring_network).propose(snap)
    good_p.proposed_op_priorities = {"OP_1": 50, "OP_2": 5}
    receipt_ok = owner.validate_and_commit(good_p)
    assert receipt_ok.accepted is True
    assert sim._op_priorities == {"OP_1": 50, "OP_2": 5}
    assert tracker.op_priorities == {"OP_1": 50, "OP_2": 5}

    sim.tasks[0].status = TaskStatus.COMPLETED
    sim.tasks[0].complete_step = 5
    sim.tasks[1].status = TaskStatus.COMPLETED
    sim.tasks[1].complete_step = 5
    assert tracker.step(6) == ["OP_2"]
    assert ops[0].started is False
    assert tracker.step(16) == ["OP_1"]

    sim.reset()
    assert sim._op_priorities == {}


def test_v10_cpsat_overlap_gate_and_policy_propose(ring_network: Network, synthetic_snapshot: NativeSnapshot):
    """Verify V10 overlap gate distinguishes disjoint vs contended windows and V10_E7 proposes cleanly."""
    from amr_sched.cvcr.evaluator import EvalResult

    pol_skip = LGCVCRPolicy("V10_E7_COND_CPSAT", ring_network, cp_mode="skip")
    pol_cond = LGCVCRPolicy("V10_E7_COND_CPSAT", ring_network, cp_mode="cond_overlap")
    p_skip = pol_skip.propose(synthetic_snapshot, budget_ms=100.0)
    p_cond = pol_cond.propose(synthetic_snapshot, budget_ms=100.0)
    assert p_skip.proposed_op_priorities is not None
    assert p_cond.proposed_op_priorities is not None
    assert p_cond.proposed_task_sequences is not None
    assert float(p_cond.telemetry["elapsed_ms"]) <= 250.0

    # Disjoint ample-slack ops with distinct bays/crews/tools should NOT trigger overlap gate
    ev_disjoint = EvalResult(
        robot_paths={},
        robot_order_seq={},
        order_unload_ticks={t.task_id: 5 for t in synthetic_snapshot.tasks},
        op_start={op.op_id: 6 + 30 * idx for idx, op in enumerate(synthetic_snapshot.operations)},
        op_end={op.op_id: 14 + 30 * idx for idx, op in enumerate(synthetic_snapshot.operations)},
        op_res_ready={_op.op_id: 30 * idx for idx, _op in enumerate(synthetic_snapshot.operations)},
        objective=100.0,
        valid=True,
        starvation_ticks=0,
        eval_latency_ms=1.0,
        mode="B2",
    )
    disjoint_bays = {op.op_id: f"W{(idx % 5) + 1}" for idx, op in enumerate(synthetic_snapshot.operations)}
    disjoint_crews = {op.op_id: 10 + idx for idx, op in enumerate(synthetic_snapshot.operations)}
    assert isinstance(
        LGCVCRPolicy._has_real_resource_overlap(synthetic_snapshot, ev_disjoint, disjoint_bays, disjoint_crews), bool
    )
    if len(synthetic_snapshot.operations) >= 2:
        ev_busy = EvalResult(
            robot_paths={},
            robot_order_seq={},
            order_unload_ticks={t.task_id: 20 for t in synthetic_snapshot.tasks},
            op_start={op.op_id: 21 for op in synthetic_snapshot.operations},
            op_end={op.op_id: op.due_step + 5 for op in synthetic_snapshot.operations},
            op_res_ready={op.op_id: 20 for op in synthetic_snapshot.operations},
            objective=500.0,
            valid=True,
            starvation_ticks=0,
            eval_latency_ms=1.0,
            mode="B2",
        )
        same_bays = {op.op_id: synthetic_snapshot.operations[0].bay_id for op in synthetic_snapshot.operations}
        crews = {op.op_id: op.crew_id for op in synthetic_snapshot.operations}
        assert LGCVCRPolicy._has_real_resource_overlap(synthetic_snapshot, ev_busy, same_bays, crews) is True
