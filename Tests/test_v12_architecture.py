"""Unit and closed-loop integration tests for V12 Top-Venue Synergistic Rescheduler."""

from __future__ import annotations

import numpy as np

from amr_sched.baselines.top_venue_baselines import BalansIJCAI25PlainProposer, DetourDeltaRAL26Proposer
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.loaded_corridor_repair import detect_stalled_picked_agvs
from amr_sched.cvcr.policy import LGCVCRPolicy
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.cvcr.v12_destroy_selector import (
    DESTROY_TASK_FEAT_DIM,
    HeteroDestroySelector,
    extract_hetero_destroy_features,
)
from amr_sched.cvcr.v12_proactive_guard import (
    analyze_network_havens,
    compute_detour_delta_signals,
    compute_kitting_marginal_start_value,
    plan_haven_safe_corridor_repair,
)
from amr_sched.rl import (
    ACTION_KEEP,
    AMRReschedulingEnv,
    BalansOnlineController,
    EXPERT_V12_ACTION_NAMES,
    EXPERT_V12_STATE_DIM,
    NUM_EXPERT_V12_ACTIONS,
    RuleV12ProactivePolicy,
    WhenToReplanTrigger,
    build_mpasil_v12_model,
    compute_expert_v12_action_mask,
    compute_expert_v12_dual_graph,
    compute_expert_v12_state,
)


def test_haven_profile_and_detour_delta_causality() -> None:
    """Verify dead-end leaf spur extraction and Detour-Delta causal bay congestion estimation."""
    env = AMRReschedulingEnv(topology="unseen_ladder", seed=7001, expert_v12_portfolio=True)
    obs, info = env.reset(seed=7001, options={"topology": "unseen_ladder"})
    assert env.sim is not None and env.tracker is not None
    prof = analyze_network_havens(env.sim.network)
    assert len(prof.leaf_spurs) >= 6
    assert prof.leaf_spurs.get("W5") == "LN_4"
    assert len(prof.safe_havens) >= 4

    for _ in range(18):
        obs, _, term, trunc, info = env.step(ACTION_KEEP)
        if term or trunc:
            break

    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
    sig = compute_detour_delta_signals(env.sim.network, snap, simulator=env.sim)
    assert isinstance(sig["weights_q3"], dict)
    assert isinstance(sig["bay_detour_delta"], dict)
    for val in sig["bay_detour_delta"].values():
        assert 0.0 <= val <= 45.0
    for w in sig["weights_q3"].values():
        assert 3 <= w <= 48


def test_edge_gate_stall_detection_backward_compatible() -> None:
    """Verify `include_edge_gate_stalls` catches elapsed-transit stalls while keeping v11 unchanged."""
    env = AMRReschedulingEnv(topology="unseen_ladder", seed=7102, expert_v10_portfolio=True)
    env.reset(seed=7102, options={"topology": "unseen_ladder", "stress_axis": "fleet_density", "n_agvs_override": 8})
    assert env.sim is not None
    for _ in range(32):
        _, _, term, trunc, _ = env.step(ACTION_KEEP)
        if term or trunc:
            break
    # Pick first loaded AGV if available, simulate elapsed transit + stall
    agv = env.sim.agvs[0]
    for t in env.sim.tasks:
        from amr_sched.entities import TaskStatus
        if t.assigned_agv == agv.agv_id:
            t.status = TaskStatus.PICKED
            env.sim._jobs[agv.agv_id] = env.sim._jobs.get(agv.agv_id) or type("_TmpJob", (), {"task_id": t.task_id, "hops": [], "idx": 0})()
            if hasattr(env.sim._jobs[agv.agv_id], "task_id"):
                env.sim._jobs[agv.agv_id].task_id = t.task_id
            env.sim._transit[agv.agv_id] = ("LN_4", "W5", int(env.sim._step_cursor) - 10, int(env.sim._step_cursor) - 4)
            env.sim._blocked_since[agv.agv_id] = int(env.sim._step_cursor) - 12
            break
    st_v11 = detect_stalled_picked_agvs(env.sim, stall_threshold=8, include_edge_gate_stalls=False)
    st_v12 = detect_stalled_picked_agvs(env.sim, stall_threshold=8, include_edge_gate_stalls=True)
    assert agv.agv_id not in st_v11
    assert agv.agv_id in st_v12


def test_hetero_destroy_selector_and_v12_spaces() -> None:
    """Verify V12 44-D state, 9 actions, HeteroDestroy multi-radius pools, and MP-ASIL net forward."""
    env = AMRReschedulingEnv(topology="ring", seed=6001, expert_v12_portfolio=True, loaded_repair_mode="haven_v12")
    obs, info = env.reset(seed=6001, options={"topology": "ring"})
    assert env.sim is not None and env.tracker is not None
    assert NUM_EXPERT_V12_ACTIONS == len(EXPERT_V12_ACTION_NAMES) == 9
    assert info["action_mask"].shape == (9,)

    o12 = compute_expert_v12_state(env.sim, env.tracker, obs)
    assert o12.shape == (EXPERT_V12_STATE_DIM,)
    assert np.all(np.isfinite(o12))
    dg12 = compute_expert_v12_dual_graph(env.sim, env.tracker, obs)
    assert dg12["global_vec"].shape == (EXPERT_V12_STATE_DIM,)

    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
    clf = classify_commitments(snap)
    ev = CVCREvaluator(env.sim.network, snap, use_rust=env.use_rust)
    base_res = ev.evaluate(
        {t.task_id: t.assigned_agv for t in snap.tasks if t.assigned_agv is not None},
        op_priority={op.op_id: op.due_step for op in snap.operations},
    )
    sel = HeteroDestroySelector()
    bays = {op.op_id: op.bay_id for op in snap.operations}
    crews = {op.op_id: op.crew_id for op in snap.operations}
    tids, tx, ax, ox, _, _, _ = extract_hetero_destroy_features(env.sim.network, snap, clf, base_res, bays, crews)
    assert tx.shape[1] == DESTROY_TASK_FEAT_DIM
    if len(tids) >= 3:
        d_s, f_s, _ = sel.select_destroy_tasks(env.sim.network, snap, clf, base_res, bays, crews, radius="small")
        d_m, f_m, _ = sel.select_destroy_tasks(env.sim.network, snap, clf, base_res, bays, crews, radius="medium")
        assert len(d_s) <= len(d_m)
        assert not (set(d_s) & f_s) and not (set(d_m) & f_m)

    import torch
    net = build_mpasil_v12_model()
    logits, val, gates = net(
        torch.from_numpy(dg12["global_vec"]).unsqueeze(0),
        torch.from_numpy(dg12["op_nodes"]).unsqueeze(0),
        torch.from_numpy(dg12["op_mask"]).unsqueeze(0),
        torch.from_numpy(dg12["agv_nodes"]).unsqueeze(0),
        torch.from_numpy(dg12["agv_mask"]).unsqueeze(0),
        torch.from_numpy(info["action_mask"].astype(np.float32)).unsqueeze(0),
    )
    assert logits.shape == (1, NUM_EXPERT_V12_ACTIONS)
    assert val.shape == (1,)
    assert gates.shape == (1, 3)


def test_v12_rule_and_balans_solve_unseen_7102_with_full_audit() -> None:
    """Verify V12 Rule and BALANS resolve the notorious v11 fleet_density trap `7102` cleanly."""
    for pol_cls in (RuleV12ProactivePolicy, BalansOnlineController):
        env = AMRReschedulingEnv(
            topology="unseen_ladder",
            seed=7102,
            allow_kitting_shortcut=False,
            pure_physical_reward=True,
            real_compute_window_coupling=True,
            expert_v12_portfolio=True,
            loaded_repair_mode="haven_v12",
            loaded_repair_threshold=8,
            record_trajectory=True,
        )
        obs, info = env.reset(
            seed=7102,
            options={"topology": "unseen_ladder", "stress_axis": "fleet_density", "n_agvs_override": 8},
        )
        pol = pol_cls(epoch_interval=16)
        pol.reset()
        done = False
        while not done:
            o12 = env.get_expert_v12_obs(obs)
            act = pol.select_action(o12, info, sim=env.sim)
            obs, _, term, trunc, info = env.step(act)
            if isinstance(pol, BalansOnlineController) and act >= 2:
                pol.observe_outcome(info, env.get_expert_v12_obs(obs))
            done = term or trunc
        m = env.compute_episode_metrics()
        aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
        assert aud.pass_audit, f"Audit failed: {aud.issues}"
        assert m["task_completion_rate"] == 1.0, f"{pol_cls.__name__} left incomplete tasks on 7102: {m}"
        assert m["production_completion_rate"] == 1.0


def test_external_top_venue_baselines_unified_g1_contract() -> None:
    """Verify RA-L'26 Detour-Delta and IJCAI'25 Plain BALANS run under unified G1 contract."""
    for proposer in (DetourDeltaRAL26Proposer(use_rust=True), BalansIJCAI25PlainProposer(seed=42, use_rust=True)):
        env = AMRReschedulingEnv(
            topology="ring",
            seed=6001,
            allow_kitting_shortcut=False,
            pure_physical_reward=True,
            real_compute_window_coupling=True,
            expert_v10_portfolio=True,
            external_proposer=proposer,
            record_trajectory=True,
        )
        obs, info = env.reset(seed=6001, options={"topology": "ring"})
        done = False
        last_dec = -999
        while not done:
            t = int(info.get("step", 0))
            mask = info["action_mask"]
            act = 7 if ((last_dec < 0 or t - last_dec >= 16) and mask[7] == 1) else ACTION_KEEP
            if act != ACTION_KEEP:
                last_dec = t
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
        aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
        m = env.compute_episode_metrics()
        assert aud.pass_audit
        assert m["task_completion_rate"] == 1.0


def test_when_to_replan_tick0_kickoff_guard() -> None:
    """Confirm all-zero action mask at tick 0 does not consume opening episode kickoff."""
    trig = WhenToReplanTrigger(epoch_interval=16)
    trig.reset()
    obs = np.zeros((EXPERT_V12_STATE_DIM,), dtype=np.float32)
    mask0 = np.zeros((NUM_EXPERT_V12_ACTIONS,), dtype=np.int64)
    mask0[0] = 1
    qm0, _ = trig.should_query(obs, mask0, tick=0, sim=None)
    assert qm0 == "none"
    assert trig.last_dec < 0

    mask1 = np.ones((NUM_EXPERT_V12_ACTIONS,), dtype=np.int64)
    qm1, _ = trig.should_query(obs, mask1, tick=1, sim=None)
    assert qm1 == "global"
    assert trig.last_dec == 1


def test_causal_opening_tree_and_smdp_v12_primitives() -> None:
    """Verify CausalOpeningTreeSelector serialization and V12 SMDP Q-net/query primitives."""
    from amr_sched.rl import CausalOpeningTreeSelector, build_smdp_v12_qnet, classify_v12_regime, should_query_v12_smdp
    import torch

    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 16))
    cands = ["rl_v7", "rule_v10", "rl_v12_smdp"]
    Y = np.column_stack([
        1500.0 + 100.0 * (X[:, 0] > 0.0),
        1650.0 + np.zeros(40),
        1300.0 + 300.0 * (X[:, 0] <= 0.0),
    ])
    tree = CausalOpeningTreeSelector(cands, max_depth=2, min_samples_leaf=8).fit(X, Y)
    restored = CausalOpeningTreeSelector.from_dict(tree.to_dict())
    for i in range(len(X)):
        assert restored.predict(X[i]) == tree.predict(X[i])

    obs = np.zeros((EXPERT_V12_STATE_DIM,), dtype=np.float32)
    assert classify_v12_regime(obs) == 2
    obs[18] = 1.0
    assert classify_v12_regime(obs) == 0
    obs[18] = 0.0
    obs[19] = 1.0
    assert classify_v12_regime(obs) == 1
    obs[19] = 0.0

    mask = np.ones((NUM_EXPERT_V12_ACTIONS,), dtype=np.int64)
    qm, _, exec_m = should_query_v12_smdp(obs, mask, 1, -999, epoch_interval=12, prev_emerg=(False, False))
    assert qm == "global"
    assert exec_m[0] == 1 and exec_m[8] == 1
    obs_f = obs.copy()
    obs_f[18] = 1.0
    obs_f[36] = 0.40
    qmf, _, exec_mf = should_query_v12_smdp(obs_f, mask, 10, 1, epoch_interval=12, prev_emerg=(False, False))
    assert qmf == "global" and exec_mf[2] == 1 and exec_mf[0] == 0

    qnet = build_smdp_v12_qnet()
    with torch.no_grad():
        qv = qnet.forward_one(
            torch.from_numpy(obs).float().unsqueeze(0),
            torch.from_numpy(exec_m).float().unsqueeze(0),
            classify_v12_regime(obs),
        )
    assert qv.shape == (1, NUM_EXPERT_V12_ACTIONS)


def test_kitting_marginal_start_value_and_switchable_precedence() -> None:
    """Verify marginal operation-start kitting valuation and BTPG/SES corridor arbitration."""
    env = AMRReschedulingEnv(topology="unseen_ladder", seed=7001, expert_v12_portfolio=True)
    env.reset(seed=7001, options={"topology": "unseen_ladder"})
    assert env.sim is not None and env.tracker is not None

    for _ in range(15):
        _, _, term, trunc, _ = env.step(ACTION_KEEP)
        if term or trunc:
            break

    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
    kval = compute_kitting_marginal_start_value(snap, simulator=env.sim)
    assert isinstance(kval, dict)
    for info in kval.values():
        assert isinstance(info, tuple) and len(info) == 6
        assert int(info[0]) in (0, 1, 2, 3)
        assert int(info[1]) in (0, 1)

    rep_sw, st_sw = plan_haven_safe_corridor_repair(
        env.sim, snap, stall_threshold=8, allow_precedence_swap=True, use_kitting_value=True
    )
    rep_no, st_no = plan_haven_safe_corridor_repair(
        env.sim, snap, stall_threshold=8, allow_precedence_swap=False, use_kitting_value=False
    )
    assert isinstance(rep_sw, dict) and isinstance(rep_no, dict)
    assert "corridor_arbitrations" in st_sw and "corridor_arbitrations" in st_no


def test_v10_selector_from_dict_and_causal_tree_feature_subset() -> None:
    """Verify HistoricalTopoSelector unseen_choke_fallback and CausalOpeningTreeSelector feature subset."""
    from amr_sched.rl.v10_selector import CausalOpeningTreeSelector, HistoricalTopoSelector

    hist = HistoricalTopoSelector.from_dict(
        {
            "candidates": ["rl_v7", "rl_v10", "rl_v12_smdp", "rule_v11_coord"],
            "topo_winners": {"ring": "rl_v7", "grid": "rl_v12_smdp", "workshop": "rule_v11_coord"},
            "unseen_choke_fallback": "rl_v10",
            "fallback": "rule_v11_coord",
        }
    )
    feat_choke_ratio = np.array([3.35] + [0.0] * 15, dtype=np.float64)
    feat_choke_deg = np.array([2.40, 0.0, 0.0, 0.30] + [0.0] * 12, dtype=np.float64)
    assert hist.predict("ring", feat_choke_ratio) == "rl_v7"
    assert hist.predict("unseen_choke_yard", feat_choke_ratio) == "rl_v10"
    assert hist.predict("unseen_choke_yard", feat_choke_deg) == "rl_v10"

    feats = np.zeros((18, 16), dtype=np.float64)
    feats[:9, 0] = 2.0
    feats[9:, 0] = 3.2
    feats[:9, 15] = 1.0
    feats[9:, 15] = -1.0
    costs = np.zeros((18, 2), dtype=np.float64)
    costs[:9, 0] = 1.0
    costs[:9, 1] = 10.0
    costs[9:, 0] = 10.0
    costs[9:, 1] = 1.0
    tree = CausalOpeningTreeSelector(
        ["rl_v7", "rl_v10"], max_depth=2, min_samples_leaf=4, allowed_features=tuple(range(15))
    ).fit(feats, costs)
    tree2 = CausalOpeningTreeSelector.from_dict(tree.to_dict())
    assert tuple(tree2.allowed_features) == tuple(range(15))
    assert tree2.predict(feats[0]) == "rl_v7"
    assert tree2.predict(feats[-1]) == "rl_v10"


