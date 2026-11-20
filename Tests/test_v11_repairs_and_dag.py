"""V11 unit and integration tests for multi-predecessor DAGs, scenario determinism,
live loaded-corridor repairs, shared env execution parity, and CP-SAT telemetry.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.loaded_corridor_repair import (
    detect_stalled_picked_agvs,
    plan_loaded_corridor_repair,
)
from amr_sched.cvcr.policy import CandidateProposal, LGCVCRPolicy
from amr_sched.cvcr.production import NativeProductionTracker
from amr_sched.cvcr.snapshot import NativeSnapshotCapture, OperationSnapshot, op_predecessors
from amr_sched.cvcr.transaction import NativeTransactionOwner
from amr_sched.entities import Agv, AgvState, Task, TaskPriority, TaskStatus
from amr_sched.network import Edge, Network, Node, NodeKind
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.rl import (
    AMRReschedulingEnv,
    EXPERT_V10_ACTION_NAMES,
    RuleV10PortfolioPolicy,
    compute_expert_v10_action_mask,
    compute_expert_v9_state,
    should_query_v11_controller,
)
from amr_sched.scenarios_v10 import (
    OPENING_FEATURE_DIM,
    build_v10_stress_scenario,
    extract_opening_structural_features,
)
from amr_sched.simulation import Hop, SimulationConfig, Simulator


def _build_minimal_line_sim() -> tuple[Simulator, list[OperationSnapshot], NativeProductionTracker]:
    """Build a small deterministic simulator for unit-testing DAG and live repairs."""
    net = Network()
    net.add_node(Node("P1", 0.0, 0.0, NodeKind.BUFFER, capacity=2))
    net.add_node(Node("J1", 6.0, 0.0, NodeKind.AISLE, capacity=2))
    net.add_node(Node("C1", 12.0, 0.0, NodeKind.AISLE, capacity=1))
    net.add_node(Node("W1", 18.0, 0.0, NodeKind.WORKSTATION, capacity=2))
    net.add_node(Node("B1", 12.0, 6.0, NodeKind.AISLE, capacity=2))
    for u, v in (("P1", "J1"), ("J1", "C1"), ("C1", "W1"), ("J1", "B1"), ("B1", "W1")):
        net.add_bidirectional(u, v, length=6.0, speed_limit=3.0)

    t1 = Task("T1", "P1", "W1", release_step=0, due_step=80, priority=TaskPriority.NORMAL)
    t2 = Task("T2", "P1", "W1", release_step=0, due_step=80, priority=TaskPriority.NORMAL)
    t5 = Task("T5", "P1", "W1", release_step=0, due_step=90, priority=TaskPriority.NORMAL)
    ops = [
        OperationSnapshot(
            op_id="OP_1",
            kit_task_ids=("T1",),
            bay_id="W1",
            crew_id=0,
            tool_id=0,
            proc_steps=4,
            due_step=40,
            eligible_bays=("W1",),
        ),
        OperationSnapshot(
            op_id="OP_2",
            kit_task_ids=("T2",),
            bay_id="W1",
            crew_id=1,
            tool_id=1,
            proc_steps=6,
            due_step=50,
            eligible_bays=("W1",),
        ),
        OperationSnapshot(
            op_id="OP_5",
            kit_task_ids=("T5",),
            bay_id="W1",
            crew_id=0,
            tool_id=0,
            proc_steps=4,
            due_step=80,
            predecessor_op_id="OP_1",
            predecessor_op_ids=("OP_1", "OP_2"),
            eligible_bays=("W1",),
        ),
    ]
    agvs = [Agv("agv_1", max_speed=3.0, start_node="P1"), Agv("agv_2", max_speed=3.0, start_node="J1")]
    sim = Simulator(
        network=net,
        tasks=[t1, t2, t5],
        agvs=agvs,
        planner=SpaceTimeAStarPlanner(max_speed=3.0, dt=1.0),
        allocator=GreedyAllocator(),
        config=SimulationConfig(dt=1.0, max_steps=200, record_trajectory=True),
    )
    sim.reset()
    tracker = NativeProductionTracker(ops, sim.tasks)
    return sim, ops, tracker


def test_multi_predecessor_dag_enforcement_tracker_eval_auditor() -> None:
    """Ensure multi-predecessor joins enforce ALL predecessors across tracker, evaluator, and auditor."""
    sim, ops, tracker = _build_minimal_line_sim()
    t1, t2, t5 = sim.tasks

    # Complete kit for OP_1 and OP_5 early, while keeping OP_2 uncompleted until tick 25
    t1.status = TaskStatus.COMPLETED
    t1.pickup_step = 2
    t1.complete_step = 6
    t5.status = TaskStatus.COMPLETED
    t5.pickup_step = 2
    t5.complete_step = 8

    for step in range(1, 35):
        if step == 25:
            t2.status = TaskStatus.COMPLETED
            t2.pickup_step = 18
            t2.complete_step = 25
        tracker.step(step)

    op1, op2, op5 = ops
    assert op_predecessors(op5) == ("OP_1", "OP_2")
    assert op1.started and op1.end_step is not None and op1.end_step <= 15
    assert op2.started and op2.end_step == 31
    assert op5.started and op5.start_step is not None and op5.start_step >= int(op2.end_step)

    # Evaluator check on multi-predecessor snapshot
    sim2, ops2, _ = _build_minimal_line_sim()
    sim2._task_by_id["T1"].status = TaskStatus.COMPLETED
    sim2._task_by_id["T1"].complete_step = 5
    sim2._task_by_id["T5"].status = TaskStatus.COMPLETED
    sim2._task_by_id["T5"].complete_step = 6
    sim2._step_cursor = 6
    snap2 = NativeSnapshotCapture.capture(sim2, ops2, hide_future_releases=True)
    ev = CVCREvaluator(sim2.network, snap2, mode="B1", max_speed=3.0, use_rust=False)
    res = ev.evaluate({"T2": "agv_1"})
    assert res.valid
    assert res.op_start["OP_5"] >= res.op_end["OP_2"]

    # Auditor check must catch any artificial start before secondary predecessor OP_2
    op5.start_step = int(op1.end_step)
    op5.end_step = op5.start_step + 4
    sim._step_cursor = 35
    aud = IndependentAuditor.audit_simulation_run(sim, ops, require_trajectory=False)
    assert not aud.pass_audit
    assert any("PREDECESSOR_VIOLATION:OP_5" in issue for issue in aud.issues)


def test_scenario_hash_determinism_across_pythonhashseed() -> None:
    """Verify stress scenario generation is invariant across PYTHONHASHSEED values."""
    script = (
        "import json; from amr_sched.scenarios_v10 import build_v10_stress_scenario; "
        "net, tasks, agvs, ops, dis = build_v10_stress_scenario('unseen_choke_yard', 7104, stress_axis='flow_inversion'); "
        "out = {'tasks': [(t.task_id, t.pickup, t.dropoff, t.release_step, t.due_step) for t in tasks], "
        "'agvs': [(a.agv_id, a.start_node) for a in agvs], "
        "'ops': [(o.op_id, o.bay_id, o.predecessor_op_id, list(o.predecessor_op_ids)) for o in ops], "
        "'dis': dis}; print(json.dumps(out, sort_keys=True))"
    )
    outputs = []
    for hseed in ("11", "97"):
        env_sub = dict(os.environ)
        env_sub["PYTHONHASHSEED"] = hseed
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env=env_sub,
            stdin=subprocess.DEVNULL,
        )
        outputs.append(json.loads(proc.stdout.strip()))
    assert outputs[0] == outputs[1]


def test_proposed_live_repairs_validation_rollback_and_unlock() -> None:
    """Validate proposed_live_repairs integrity, rollback, and end-to-end live-lock clearance."""
    sim, ops, tracker = _build_minimal_line_sim()
    owner = NativeTransactionOwner(sim, ops, production_tracker=tracker)
    t1 = sim._task_by_id["T1"]
    t1.status = TaskStatus.PICKED
    t1.assigned_agv = "agv_1"
    t1.assigned_step = 0
    t1.pickup_step = 4
    agv1 = sim.agvs[0]
    agv1.cur_task = "T1"
    agv1.cur_node = "J1"
    agv1.state = AgvState.TO_DROPOFF
    sim._step_cursor = 12
    snap = NativeSnapshotCapture.capture(sim, ops, hide_future_releases=True)
    snap_digest = snap.compute_digest()

    # Invalid repair attempting to change goal away from picked dropoff must reject
    bad_prop = CandidateProposal(
        schema="amr-lgcvcr.proposal.v1",
        policy_id="TEST",
        snapshot_digest=snap_digest,
        affected_robots=["agv_1"],
        affected_tasks=["T1"],
        affected_operations=[],
        proposed_assignment={},
        proposed_routes={},
        proposed_bays={},
        proposed_crews={},
        predicted_obj=0.0,
        predicted_makespan=50,
        candidate_digest="bad_goal",
        telemetry={},
        proposed_live_repairs={
            "agv_1": {
                "kind": "picked_reroute",
                "task_id": "T1",
                "goal": "B1",
                "hops": [("J1", "B1", 12, 14)],
            }
        },
    )
    rcpt_bad = owner.validate_and_commit(bad_prop)
    assert not rcpt_bad.accepted
    assert any("VIOLATION_LIVE_REPAIR_COMMITMENT" in r for r in rcpt_bad.reasons)

    # Atomic rollback check: inject commit-time failure after live repair application
    sim._blocked_since["agv_1"] = 4
    sim._drop_end["agv_1"] = 99
    snap_ok = NativeSnapshotCapture.capture(sim, ops, hide_future_releases=True)
    orig_set_tw = sim.set_traffic_weights

    def _boom_tw(*args, **kwargs):
        raise RuntimeError("injected rollback failure")

    sim.set_traffic_weights = _boom_tw
    rollback_prop = CandidateProposal(
        schema="amr-lgcvcr.proposal.v1",
        policy_id="TEST",
        snapshot_digest=snap_ok.compute_digest(),
        affected_robots=["agv_1"],
        affected_tasks=["T1"],
        affected_operations=[],
        proposed_assignment={},
        proposed_routes={},
        proposed_bays={},
        proposed_crews={},
        predicted_obj=0.0,
        predicted_makespan=50,
        candidate_digest="ok_until_boom",
        telemetry={},
        proposed_live_repairs={
            "agv_1": {
                "kind": "picked_reroute",
                "task_id": "T1",
                "goal": "W1",
                "hops": [("J1", "B1", 12, 14), ("B1", "W1", 14, 16)],
                "traffic_weights": {("J1", "C1"): 50},
                "traffic_cutoff": 60,
            }
        },
    )
    try:
        try:
            owner.validate_and_commit(rollback_prop)
            raise AssertionError("Expected RuntimeError during atomic rollback test")
        except RuntimeError:
            pass
    finally:
        sim.set_traffic_weights = orig_set_tw
    assert sim._blocked_since.get("agv_1") == 4
    assert sim._drop_end.get("agv_1") == 99

    # Full end-to-end live-lock unlock on Codex's card 7104 (flow_inversion on unseen_choke_yard)
    env = AMRReschedulingEnv(
        topology="unseen_choke_yard",
        seed=7104,
        max_steps=1000,
        expert_v10_portfolio=True,
        record_trajectory=True,
        loaded_repair_mode="coordinated_neighbor_repair",
        loaded_repair_threshold=8,
    )
    obs, _ = env.reset(seed=7104, options={"topology": "unseen_choke_yard", "stress_axis": "flow_inversion"})
    rule = RuleV10PortfolioPolicy(enable_v11_stall_override=True, stall_threshold=8)
    done = False
    while not done:
        assert env.sim is not None and env.tracker is not None
        tick = int(env.sim._step_cursor)
        obs9 = compute_expert_v9_state(env.sim, env.tracker, last_replan_tick=env.last_replan_tick, recent_conflict_ticks=env.recent_conflict_ticks)
        mask = compute_expert_v10_action_mask(env.sim, env.last_replan_tick)
        act = rule.select_action(obs9, {"action_mask": mask, "step": tick}, sim=env.sim)
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc

    metrics = env.compute_episode_metrics()
    audit = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
    assert audit.pass_audit, f"Audit failed: {audit.issues}"
    assert metrics["task_completion_rate"] == 1.0
    assert metrics["production_completion_rate"] == 1.0
    assert metrics["cmax"] < 600


def test_shared_env_execution_parity_external_vs_internal_and_visibility() -> None:
    """Confirm external_proposer follows identical env accounting and visibility modes isolate future tasks."""
    e7_idx = EXPERT_V10_ACTION_NAMES.index("V10_E7_COND_CPSAT_100MS")
    ext_proposer = lambda snap, net, budget_ms=100.0: LGCVCRPolicy(
        "V10_E7_COND_CPSAT", net, use_rust=True
    ).propose(snap, budget_ms=budget_ms)

    def _run_fixed_e7(ext_fn=None) -> tuple[dict[str, object], list[dict[str, object]]]:
        with virtual_clock_scope(poll_ms=0.002):
            env = AMRReschedulingEnv(
                topology="grid",
                seed=2001,
                max_steps=400,
                expert_v10_portfolio=True,
                external_proposer=ext_fn,
                record_trajectory=True,
            )
            env.reset(seed=2001)
            last_dec = -1
            prev_em: tuple[bool, bool] = (False, False)
            done = False
            while not done:
                assert env.sim is not None and env.tracker is not None
                tick = int(env.sim._step_cursor)
                obs9 = compute_expert_v9_state(env.sim, env.tracker, last_replan_tick=env.last_replan_tick)
                mask = compute_expert_v10_action_mask(env.sim, env.last_replan_tick)
                do_q, prev_em = should_query_v11_controller(obs9, mask, tick, last_dec, prev_emerg=prev_em)
                act = e7_idx if (do_q and mask[e7_idx] == 1) else 0
                if act != 0:
                    last_dec = tick
                _, _, term, trunc, _ = env.step(act)
                done = term or trunc
            aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
            assert aud.pass_audit, f"Audit failed: {aud.issues}"
            return env.compute_episode_metrics(), env.decision_logs

    m_int, l_int = _run_fixed_e7(None)
    m_ext, l_ext = _run_fixed_e7(ext_proposer)

    assert m_int["cmax"] == m_ext["cmax"]
    assert m_int["total_churn"] == m_ext["total_churn"]
    assert m_int["total_obj"] == m_ext["total_obj"]
    assert [d["step_churn"] for d in l_int] == [d["step_churn"] for d in l_ext]
    assert [d["window_expired"] for d in l_int] == [d["window_expired"] for d in l_ext]
    assert "cp_stats_totals" in m_int

    # Check dual selector visibility at t=0 when tasks have future release steps
    env_vis = AMRReschedulingEnv(topology="unseen_choke_yard", seed=7104, expert_v10_portfolio=True)
    env_vis.reset(seed=7104, options={"topology": "unseen_choke_yard", "stress_axis": "standard"})
    assert env_vis.sim is not None and env_vis.tracker is not None
    snap_vis = NativeSnapshotCapture.capture(env_vis.sim, env_vis.tracker.operations, hide_future_releases=False)
    f_plan = extract_opening_structural_features(env_vis.sim.network, snap_vis, visibility="planned")
    f_rel = extract_opening_structural_features(env_vis.sim.network, snap_vis, visibility="released_only")
    assert f_plan.shape == (OPENING_FEATURE_DIM,)
    assert f_rel.shape == (OPENING_FEATURE_DIM,)
    assert not np.allclose(f_plan, f_rel)


def test_v11_selectors_pricing_and_two_point_handoff() -> None:
    """Verify MultiHeadRegretSelector pricing invariance, CEWinnerClassifier, and TwoPointCausalSelector serialization."""
    from amr_sched.rl import (
        CEWinnerClassifier,
        HANDOFF_FEATURE_DIM,
        MultiHeadRegretSelector,
        TwoPointCausalSelector,
        extract_handoff_features,
    )

    cands = ["rl_v7", "rule_v10", "rule_v11_solo"]
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, OPENING_FEATURE_DIM))
    # Synthetic: rl_v7 has lowest churn; rule_v11_solo has lowest op_obj when X[:, 0] > 0
    Y_op = np.zeros((24, 3), dtype=np.float64)
    Y_ch = np.zeros((24, 3), dtype=np.float64)
    Y_ms = np.zeros((24, 3), dtype=np.float64)
    for i in range(24):
        Y_op[i] = [1800.0, 1750.0, 1500.0 if X[i, 0] > 0 else 1900.0]
        Y_ch[i] = [2.0, 18.0, 18.0]
        Y_ms[i] = [300.0, 900.0, 950.0]

    mh = MultiHeadRegretSelector(cands, anchor="rl_v7", reg_lam=1.0).fit(X, Y_op, Y_ch, Y_ms)
    mh2 = MultiHeadRegretSelector.from_dict(mh.to_dict())
    feat_pos = np.array([2.0] + [0.0] * (OPENING_FEATURE_DIM - 1))
    # Heavy churn weight must pick low-churn rl_v7; heavy op weight must pick rule_v11_solo
    assert mh2.predict(feat_pos, w_op=1.0, w_churn=100.0, w_ms=0.05) == "rl_v7"
    assert mh2.predict(feat_pos, w_op=5.0, w_churn=1.0, w_ms=0.0) == "rule_v11_solo"

    ce = CEWinnerClassifier(cands).fit(X, Y_op + 15.0 * Y_ch + 0.05 * Y_ms)
    ce2 = CEWinnerClassifier.from_dict(ce.to_dict())
    assert ce2.predict(feat_pos) in cands

    obs38 = np.zeros(38, dtype=np.float64)
    obs38[9] = 0.4
    obs38[32] = 0.5
    H = np.asarray(
        [
            extract_handoff_features(X[i], obs38, tick=40, is_stalled=(X[i, 0] > 0), opening_policy="rl_v7")
            for i in range(24)
        ]
    )
    assert H.shape == (24, HANDOFF_FEATURE_DIM)
    tp = TwoPointCausalSelector(mh2, cands, reg_lam=1.0, min_switch_margin=10.0).fit_handoff(
        H, Y_op, Y_ch, Y_ms, ["rl_v7"] * 24
    )
    tp2 = TwoPointCausalSelector.from_dict(tp.to_dict())
    assert tp2.select_opening(feat_pos) in cands
    assert tp2.select_handoff(H[np.argmax(X[:, 0])], "rl_v7", w_op=5.0, w_churn=1.0, w_ms=0.0) == "rule_v11_solo"
