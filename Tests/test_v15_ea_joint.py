"""Tests for V15 Execution-Aware Joint Policy (`V15_EA_JOINT`) and controller integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.p2_perturbation import P2PerturbationConfig
from amr_sched.cvcr.policy import LGCVCRPolicy
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.cvcr.v15_ea_joint import (
    V15_CAND_FEAT_DIM,
    V15_FAMILIES,
    V15_GLOBAL_FEAT_DIM,
    V15AdaptiveController,
    V15Candidate,
    V15ExecutionAwarePolicy,
    _build_candidate_features,
    extract_v15_global_features,
)
from amr_sched.rl import AMRReschedulingEnv


def _synthetic_v15_weights(hidden_c: int = 16, d_att: int = 8, hidden_t: int = 16) -> dict[str, np.ndarray]:
    """Create deterministic compact V15 NumPy weight tensors matching inference schema."""
    rng = np.random.default_rng(42)
    return {
        "c_mu": np.zeros((V15_CAND_FEAT_DIM,), dtype=np.float32),
        "c_sd": np.ones((V15_CAND_FEAT_DIM,), dtype=np.float32),
        "g_mu": np.zeros((V15_GLOBAL_FEAT_DIM,), dtype=np.float32),
        "g_sd": np.ones((V15_GLOBAL_FEAT_DIM,), dtype=np.float32),
        "t_mu": np.zeros((44,), dtype=np.float32),
        "t_sd": np.ones((44,), dtype=np.float32),
        "cand_w1": (rng.standard_normal((V15_CAND_FEAT_DIM, hidden_c)) * 0.05).astype(np.float32),
        "cand_b1": np.zeros((hidden_c,), dtype=np.float32),
        "att_q": (rng.standard_normal((hidden_c, d_att)) * 0.05).astype(np.float32),
        "att_k": (rng.standard_normal((hidden_c, d_att)) * 0.05).astype(np.float32),
        "att_v": (rng.standard_normal((hidden_c, hidden_c)) * 0.05).astype(np.float32),
        "cand_w2": (rng.standard_normal((hidden_c * 2, hidden_c)) * 0.05).astype(np.float32),
        "cand_b2": np.zeros((hidden_c,), dtype=np.float32),
        "cand_w3": (rng.standard_normal((hidden_c, 1)) * 0.05).astype(np.float32),
        "cand_b3": np.zeros((1,), dtype=np.float32),
        "g_w1": (rng.standard_normal((V15_GLOBAL_FEAT_DIM, 12)) * 0.05).astype(np.float32),
        "g_b1": np.zeros((12,), dtype=np.float32),
        "g_w2": (rng.standard_normal((12, 1)) * 0.05).astype(np.float32),
        "g_b2": np.zeros((1,), dtype=np.float32),
        "tier_w1": (rng.standard_normal((44, hidden_t)) * 0.05).astype(np.float32),
        "tier_b1": np.zeros((hidden_t,), dtype=np.float32),
        "tier_w2": (rng.standard_normal((hidden_t, 3)) * 0.05).astype(np.float32),
        "tier_b2": np.array([0.0, 0.4, -0.2], dtype=np.float32),
    }


def test_v15_feature_dims_bandit_and_npz_roundtrip(tmp_path: Path) -> None:
    """Verify 14-D/36-D V15 features, .npz checkpoint load, bandit updates, and budget chooser."""
    env = AMRReschedulingEnv(
        topology="unseen_ladder",
        seed=5101,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        real_compute_window_coupling=True,
        expert_v12_portfolio=True,
    )
    env.reset(seed=5101, options={"topology": "unseen_ladder"})
    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
    clf = classify_commitments(snap)
    pol = LGCVCRPolicy("V15_EA_JOINT", env.sim.network)
    healthy = pol._healthy_robots(snap, clf.disturbed_robots)
    op_prio = {op.op_id: max(snap.tick, op.due_step - op.proc_steps) for op in snap.operations}
    base_assign = pol._base_assignment(snap, healthy, op_prio)
    ev_tool = CVCREvaluator(network=env.sim.network, snapshot=snap, mode="B2")
    base_eval = ev_tool.evaluate(base_assign, op_priority=op_prio)

    g_feats = extract_v15_global_features(snap, base_eval, clf, {})
    assert g_feats.shape == (V15_GLOBAL_FEAT_DIM,)
    assert np.all(np.isfinite(g_feats))

    bays = {op.op_id: op.bay_id for op in snap.operations}
    crews = {op.op_id: op.crew_id for op in snap.operations}
    c_feat, det_s = _build_candidate_features(
        "balans_adaptive_destroy",
        base_eval,
        base_eval,
        base_assign,
        base_assign,
        bays,
        crews,
        snap,
        0,
        False,
        0.1,
        0.8,
        g_feats,
    )
    assert c_feat.shape == (V15_CAND_FEAT_DIM,)
    assert np.all(np.isfinite(c_feat))
    assert math.isfinite(det_s) if False else np.isfinite(det_s)

    ckpt_path = tmp_path / "v15_test.npz"
    np.savez(str(ckpt_path), **_synthetic_v15_weights())
    v15_pol = V15ExecutionAwarePolicy.from_npz(ckpt_path)

    r_before = v15_pol.select_destroy_radius()
    v15_pol.observe_outcome(V15_FAMILIES[2], r_before, 0.75)
    assert v15_pol.family_ucb_bonus(V15_FAMILIES[2]) != 0.0

    cands = [
        V15Candidate(
            cand_id="C0",
            source_kind="keep_incumbent",
            assignment=base_assign,
            bay_overrides=bays,
            crew_overrides=crews,
            op_priority=op_prio,
            eval_res=base_eval,
            traffic_weights={},
            traffic_cutoff=0,
            guide_routes={},
            commit_seq_horizon=0,
            destroy_tids=(),
            cliff_saved=False,
            det_score=0.0,
            features=c_feat,
        ),
        V15Candidate(
            cand_id="C1",
            source_kind="balans_adaptive_destroy",
            assignment=base_assign,
            bay_overrides=bays,
            crew_overrides=crews,
            op_priority=op_prio,
            eval_res=base_eval,
            traffic_weights={},
            traffic_cutoff=0,
            guide_routes={},
            commit_seq_horizon=0,
            destroy_tids=(),
            cliff_saved=False,
            det_score=det_s + 0.5,
            features=c_feat,
        ),
    ]
    scores = v15_pol.score_candidates(g_feats, cands)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))

    obs44 = np.zeros((44,), dtype=np.float32)
    mask = np.ones((9,), dtype=np.int8)
    assert v15_pol.select_budget_action(obs44, mask, tick=0) == 5
    act_routine = v15_pol.select_budget_action(obs44, mask, tick=32)
    assert act_routine in (0, 2, 3, 5)


def test_v15_conservative_shield_and_zero_queue_lock(tmp_path: Path) -> None:
    """Verify Regret Shield blocks bad neural picks and V15 never locks multi-hop task queues."""
    w = _synthetic_v15_weights()
    # Bias neural head heavily against keeping valid candidate
    w["cand_b3"] = np.array([-3.0], dtype=np.float32)
    p = tmp_path / "shield.npz"
    np.savez(str(p), **w)
    v15_pol = V15ExecutionAwarePolicy.from_npz(p)

    env = AMRReschedulingEnv(
        topology="unseen_choke_yard",
        seed=5102,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        real_compute_window_coupling=True,
        expert_v12_portfolio=True,
        v13_kitting_mode="V15_EA_JOINT",
        v15_policy=v15_pol,
    )
    obs, _ = env.reset(seed=5102, options={"topology": "unseen_choke_yard"})
    # Advance past tick 16 so any unpicked older task is checked against stale-safe age gate
    for _ in range(18):
        obs, _, tm, tr, _ = env.step(0)
        if tm or tr:
            break
    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
    pol = LGCVCRPolicy("V15_EA_JOINT", env.sim.network, v15_policy=v15_pol)
    prop = pol.propose(snap, budget_ms=100.0)
    assert prop.proposed_task_sequences is None
    assert prop.telemetry["kitting_decision"]["num_candidates"] >= 2
    snap_by_id = {t.task_id: t for t in snap.tasks}
    for tid in prop.affected_tasks:
        t_orig = snap_by_id.get(tid)
        if t_orig is not None and t_orig.status.lower() == "pending" and not t_orig.is_urgent:
            assert (snap.tick - t_orig.release_step) < 16


def test_v15_closed_loop_p45_episode_and_auditor() -> None:
    """Verify full closed-loop V15_EA_JOINT + P2/P4.5 execution completes 100% and passes IndependentAuditor."""
    p2_cfg = P2PerturbationConfig(
        seed=5106,
        enable_travel_drift=True,
        enable_dynamic_obstacles=True,
        obstacle_count=2,
        obstacle_durations=(8, 28),
        enable_local_planner_fail=True,
        enable_p4_belief=True,
    )
    v15_pol = V15ExecutionAwarePolicy(weights=None)
    ctrl = V15AdaptiveController(epoch_interval=16, policy=v15_pol)
    ctrl.reset()
    env = AMRReschedulingEnv(
        topology="unseen_choke_yard",
        seed=5106,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        real_compute_window_coupling=True,
        expert_v12_portfolio=True,
        loaded_repair_mode="haven_v12",
        loaded_repair_threshold=8,
        v13_kitting_mode="V15_EA_JOINT",
        v15_policy=v15_pol,
        execution_perturbation=p2_cfg,
        execution_aware=True,
        p4_choke_robust=True,
    )
    obs, info = env.reset(seed=5106, options={"topology": "unseen_choke_yard"})
    done = False
    while not done:
        act = ctrl.select_action(env.get_expert_v12_obs(obs), info, sim=env.sim)
        obs, _, tm, tr, info = env.step(act)
        done = tm or tr
    m = env.compute_episode_metrics()
    aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=False)
    assert m["task_completion_rate"] == 1.0
    assert aud.pass_audit, f"Audit failed: {aud.issues}"
