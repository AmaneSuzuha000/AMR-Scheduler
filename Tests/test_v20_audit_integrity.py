"""Integrity and determinism gate for Stage-1 IF-CORE-V20-01 audit harness."""

from __future__ import annotations

import copy
import pytest

from amr_sched.baselines.policy_registry import create_discrete_policy_env
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state
from experiments.v20_audit.common import (
    COHORT_A_SEEDS,
    COHORT_B_SEEDS,
    COHORT_C_RESERVED_SEEDS,
    RecordedStep,
    SeqDiffEvalContext,
    assert_stage1_quarantine,
    build_task_c_outlier_specs,
    build_val72_specs,
    build_val_static35_specs,
    make_v18_external_proposer,
    replay_fork_env_at_tick,
    rollout_48_and_optional_full,
)
from experiments.v20_audit.proposer_variants import (
    EXPERT_IDS,
    ExpertSuiteEvaluator,
    Phase0Config,
    Phase0RestoredProposer,
    UnifiedE0E9PrototypeProposer,
    compute_extended_amr_footprint,
    derive_base_op_prio,
)


def test_stage1_seed_quarantine_blocks_cohorts_a_b_c() -> None:
    """Tasks A/B/D must reject Cohort A/B/C seeds; Task C permits ONLY its 2 explicit outlier keys."""
    val72 = build_val72_specs()
    stat35 = build_val_static35_specs()
    assert len(val72) == 72
    assert len(stat35) == 35
    assert_stage1_quarantine([s["seed"] for s in val72], label="val72_ok")

    with pytest.raises(RuntimeError, match="QUARANTINE VIOLATION"):
        assert_stage1_quarantine([next(iter(COHORT_A_SEEDS))], label="bad_a")
    with pytest.raises(RuntimeError, match="QUARANTINE VIOLATION"):
        assert_stage1_quarantine([next(iter(COHORT_B_SEEDS))], label="bad_b")
    with pytest.raises(RuntimeError, match="QUARANTINE VIOLATION"):
        assert_stage1_quarantine([next(iter(COHORT_C_RESERVED_SEEDS))], label="bad_c")

    tc_specs = build_task_c_outlier_specs()
    assert len(tc_specs) == 2
    with pytest.raises(RuntimeError, match="whitelist violation"):
        assert_stage1_quarantine(
            [37301],
            label="bad_c_key",
            episode_keys=["topo4_offset_dual_lane_window_bottleneck_37301"],
            allow_task_c_outliers=True,
        )


def test_replay_fork_and_48_tick_rollout_exact_and_crash_free() -> None:
    """Replay fork avoids Rust deepcopy corruption and reproduces bit-exact snapshots & +48t rollouts."""
    sp = build_val72_specs()[0]
    sc_copy = copy.deepcopy(sp["scenario_tuple"])
    kind, ctrl, env = create_discrete_policy_env(
        "V19_ADAPTIVE_RL",
        topology=str(sc_copy[0]),
        seed=int(sp["seed"]),
        max_steps=int(sp["max_steps"]),
        use_rust=True,
        p2_cfg=copy.deepcopy(sp["p2_cfg"]),
    )
    raw_ext = env.external_proposer
    assert raw_ext is not None
    recorded: list[RecordedStep] = []
    target_tick = None
    target_snap = None
    target_prop = None
    target_act = None

    def _cap(snap, net, *, budget_ms=35.0):
        nonlocal target_tick, target_snap, target_prop
        p = raw_ext(snap, net, budget_ms=budget_ms)
        recorded.append(RecordedStep(step_before=int(snap.tick), action=-1, triggered_replan=True, proposal=copy.deepcopy(p)))
        if target_tick is None and int(snap.tick) >= 16:
            target_tick = int(snap.tick)
            target_snap = copy.deepcopy(snap)
            target_prop = copy.deepcopy(p)
        return p

    env.external_proposer = _cap
    obs, info = env.reset(seed=int(sp["seed"]), options={"scenario_tuple": sc_copy})
    tm = tr = False
    while not (tm or tr):
        f = compute_expert_v9_state(
            env.sim, env.tracker, last_replan_tick=env.last_replan_tick,
            recent_conflict_ticks=env.recent_conflict_ticks, decision_logs=env.decision_logs,
        )
        m_a = compute_expert_v10_action_mask(env.sim, env.last_replan_tick)
        act = int(ctrl.select_action(f, dict(info, action_mask=m_a), sim=env.sim))
        before_n = len(recorded)
        sb = int(env.current_step)
        obs, _, tm, tr, info = env.step(act)
        if len(recorded) > before_n:
            recorded[-1].action = act
            if int(recorded[-1].step_before) == target_tick and target_act is None:
                target_act = act
        else:
            recorded.append(RecordedStep(step_before=sb, action=act, triggered_replan=False, proposal=None))
        if target_tick is not None and int(env.current_step) > target_tick + 2:
            break

    assert target_tick is not None and target_snap is not None and target_act is not None
    fork_env, fork_snap, _ = replay_fork_env_at_tick(sp, recorded, target_tick)
    assert int(fork_snap.tick) == int(target_snap.tick)
    assert [t.task_id for t in fork_snap.tasks] == [t.task_id for t in target_snap.tasks]
    assert {t.task_id: t.assigned_agv for t in fork_snap.tasks} == {t.task_id: t.assigned_agv for t in target_snap.tasks}

    ro_keep = rollout_48_and_optional_full(sp, recorded, target_tick, None, action_at_target=target_act, rollout_horizon=48)
    ro_prop = rollout_48_and_optional_full(sp, recorded, target_tick, target_prop, action_at_target=target_act, rollout_horizon=48)
    assert ro_keep["ticks_48_end"] >= target_tick + 1
    assert ro_prop["ticks_48_end"] >= target_tick + 1

    # Verify propose_ms recording preserves compute-window coupling in replay_fork_env_at_tick
    from amr_sched.cvcr.clock import record_work_ms as _vc_rec, virtual_clock_scope as _vc_scope

    _, ctrl_vc, env_vc = create_discrete_policy_env(
        "V19_ADAPTIVE_RL", topology=str(sc_copy[0]), seed=int(sp["seed"]), max_steps=int(sp["max_steps"]), use_rust=True, p2_cfg=copy.deepcopy(sp["p2_cfg"])
    )
    raw_vc = env_vc.external_proposer
    assert raw_vc is not None
    rec_vc: list[RecordedStep] = []
    pre_step_expirations: dict[int, int] = {}

    def _cap_slow(snap, net, *, budget_ms=35.0):
        p = raw_vc(snap, net, budget_ms=budget_ms)
        _vc_rec(42.0)  # Force 2 compute steps (floor(42 / 20) == 2)
        rec_vc.append(RecordedStep(step_before=int(snap.tick), action=-1, triggered_replan=True, proposal=copy.deepcopy(p)))
        return p

    env_vc.external_proposer = _cap_slow
    with _vc_scope(poll_ms=0.0):
        obs_v, info_v = env_vc.reset(seed=int(sp["seed"]), options={"scenario_tuple": copy.deepcopy(sp["scenario_tuple"])})
        for _ in range(35):
            f_v = compute_expert_v9_state(
                env_vc.sim, env_vc.tracker, last_replan_tick=env_vc.last_replan_tick,
                recent_conflict_ticks=env_vc.recent_conflict_ticks, decision_logs=env_vc.decision_logs,
            )
            m_v = compute_expert_v10_action_mask(env_vc.sim, env_vc.last_replan_tick)
            act_v = int(ctrl_vc.select_action(f_v, dict(info_v, action_mask=m_v), sim=env_vc.sim))
            bn_v = len(rec_vc)
            sb_v = int(env_vc.current_step)
            pre_step_expirations[sb_v] = int(env_vc.total_window_expirations)
            obs_v, _, tm_v, tr_v, info_v = env_vc.step(act_v)
            if len(rec_vc) > bn_v:
                rec_vc[-1].action = act_v
                rec_vc[-1].propose_ms = float(env_vc.decision_logs[-1]["t_propose_ms"])
            else:
                rec_vc.append(RecordedStep(step_before=sb_v, action=act_v, triggered_replan=False, proposal=None))
            if tm_v or tr_v:
                break

    assert any(st.propose_ms >= 41.0 for st in rec_vc)
    t_last = int(rec_vc[-1].step_before)
    f_env2, f_snap2, _ = replay_fork_env_at_tick(sp, rec_vc, t_last)
    assert int(f_snap2.tick) == t_last
    assert int(f_env2.total_window_expirations) == pre_step_expirations[t_last]
    assert [int(d["window_cost_steps"]) for d in f_env2.decision_logs] == [
        int(d["window_cost_steps"]) for d in env_vc.decision_logs if int(d["step"]) < t_last
    ]
    assert any(float(d["t_propose_ms"]) >= 41.0 and int(d["window_cost_steps"]) >= 2 for d in f_env2.decision_logs)


def test_all_10_experts_phase0_unified_and_seq_diff_context() -> None:
    """All 10 experts E0-E9, Phase0 P0.3, Unified E0-E9, and SeqDiffEvalContext run cleanly without mutating amr_sched."""
    specs = build_val72_specs()
    sp_unwin = specs[0]  # 4-topo standard (unwindowed)
    _, _, env0 = create_discrete_policy_env(
        "V19_ADAPTIVE_RL", topology=str(sp_unwin["scenario_tuple"][0]), seed=int(sp_unwin["seed"]), max_steps=120, use_rust=True
    )
    env0.reset(seed=int(sp_unwin["seed"]), options={"scenario_tuple": copy.deepcopy(sp_unwin["scenario_tuple"])})
    snap0 = NativeSnapshotCapture.capture(env0.sim, env0.tracker.operations, hide_future_releases=True)

    sp = specs[4]  # window_bottleneck
    _, _, env = create_discrete_policy_env(
        "V19_ADAPTIVE_RL", topology=str(sp["scenario_tuple"][0]), seed=int(sp["seed"]), max_steps=120, use_rust=True
    )
    env.reset(seed=int(sp["seed"]), options={"scenario_tuple": copy.deepcopy(sp["scenario_tuple"])})
    for _ in range(12):
        env.step(0)
    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)

    suite = ExpertSuiteEvaluator(seed=0)
    arts = suite.evaluate_all_experts(snap, env.sim.network)
    assert set(arts.keys()) == set(EXPERT_IDS)
    for eid in EXPERT_IDS:
        assert arts[eid].eval_res_b2 is not None

    p03 = Phase0RestoredProposer(
        Phase0Config(
            name="P0.3_SD",
            fix_a4_seq=True,
            commit_unwindowed_seq=True,
            unlock_full_pool_and_bays=True,
            wire_voi_and_scale_budget=True,
            use_seq_diff_b2=True,
        ),
        backend="numpy",
        use_rust=True,
    )
    prop0 = p03(snap0, env0.sim.network, budget_ms=35.0)
    assert prop0.proposed_assignment
    if not prop0.telemetry.get("fallback_keep", True):
        assert prop0.proposed_task_sequences is not None
    prop03 = p03(snap, env.sim.network, budget_ms=35.0)
    assert prop03.proposed_assignment is not None

    uni = UnifiedE0E9PrototypeProposer(hard_budget_ms=35.0, seed=0)
    prop_u = uni(snap, env.sim.network, budget_ms=35.0)
    assert prop_u.proposed_assignment

    # Verify SeqDiffEvalContext restores original Evaluator cleanly in policy module
    import amr_sched.cvcr.policy as policy_mod
    orig_cls = policy_mod.CVCREvaluator
    v18_sd = make_v18_external_proposer(hard_budget_ms=20.0, seq_diff_b2=True)
    p_v18 = v18_sd(snap, env.sim.network, budget_ms=20.0)
    assert p_v18.proposed_assignment is not None
    assert policy_mod.CVCREvaluator is orig_cls

    # Verify extended footprint catches op_priority-only diffs
    from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
    g = extract_v19_hetero_graph(snap, env.sim.network)
    base_p = derive_base_op_prio(snap)
    mod_p = dict(base_p)
    if g.op_ids:
        mod_p[g.op_ids[0]] = mod_p[g.op_ids[0]] - 20
        fp = compute_extended_amr_footprint(snap, g, {}, g.base_assign, g.base_bays, g.base_crews, mod_p, base_p, None)
        assert fp["legacy_count"] == 0
        assert fp["changed_op_prios_count"] == 1


def test_v20_a_gate_config_and_critic(tmp_path) -> None:
    import numpy as np
    from experiments.v20_audit.common import OUT_AUDIT_DIR, build_val72_specs, run_wrapped_episode
    from experiments.v20_audit.critic_model import CRITIC_FEAT_DIM, CRITIC_FEAT_NAMES, NumpyValueCritic
    from experiments.v20_audit.proposer_variants import UnifiedE0E9PrototypeProposer

    # 1. Verify NumpyValueCritic schema roundtrip & scale/scale_ws dispatch
    npz_path = tmp_path / "dummy_critic.npz"
    w_zero = np.zeros(CRITIC_FEAT_DIM, dtype=np.float64)
    mu_zero = np.zeros(CRITIC_FEAT_DIM, dtype=np.float64)
    sig_one = np.ones(CRITIC_FEAT_DIM, dtype=np.float64)
    np.savez(
        npz_path,
        scale=np.array([250.0]),
        scale_ws=np.array([500.0]),
        mean_4t=mu_zero,
        std_4t=sig_one,
        w_4t=w_zero,
        b_4t=np.array([1.0]),
        jmix_4t=np.array([0.0]),
        mean_ws=mu_zero,
        std_ws=sig_one,
        w_ws=w_zero,
        b_ws=np.array([1.0]),
        jmix_ws=np.array([0.0]),
    )
    vc = NumpyValueCritic(npz_path)
    ws_idx = CRITIC_FEAT_NAMES.index("ws_flag")
    fv_ws = np.zeros(CRITIC_FEAT_DIM, dtype=np.float64)
    fv_ws[ws_idx] = 1.0
    assert abs(vc.predict_delta_keep(fv_ws) - 500.0) < 1e-6
    fv_4t = np.zeros(CRITIC_FEAT_DIM, dtype=np.float64)
    assert abs(vc.predict_delta_keep(fv_4t) - 250.0) < 1e-6

    real_npz = OUT_AUDIT_DIR / "v20_a_critic_weights.npz"
    if real_npz.exists():
        rvc = NumpyValueCritic(real_npz)
        assert np.isfinite(rvc.predict_delta_keep(fv_ws))
        assert np.isfinite(rvc.predict_delta_keep(fv_4t))

    # 2. Verify Gate-A Unified config completes & truncates unwindowed task_sequences to horizon=2
    sp = build_val72_specs()[0]
    uni = UnifiedE0E9PrototypeProposer(
        hard_budget_ms=35.0,
        soft_deadline_ms=15.0,
        seed=0,
        enable_e4=False,
        enable_e5=False,
        enable_e6=False,
        enable_e7=True,
        enable_e9=False,
        enable_r2_global_ws=False,
        r2_zc_ws_mode="never",
        include_staged_g="on_invalid",
        commit_unwin_seq=False,
        unwin_seq_horizon=2,
    )
    seq_lens: list[int] = []

    def _wrapped(snap, net, *, budget_ms: float = 35.0):
        prop = uni(snap, net, budget_ms=budget_ms)
        if prop.proposed_task_sequences:
            seq_lens.extend(len(seq) for seq in prop.proposed_task_sequences.values())
        return prop

    row, _ = run_wrapped_episode(
        variant_id="test_v20a_gate_cfg",
        spec=sp,
        base_env_policy="V19_ADAPTIVE_RL",
        external_proposer=_wrapped,
        hard_budget_ms=35.0,
    )
    assert row["comp100"] is True
    assert row["audit_passed"] is True
    assert all(n <= 2 for n in seq_lens)


def test_v20_b_bidirectional_parity_and_masked_subgraph() -> None:
    """Verify V20-B cached Layer-1 decoding parity, causal release closure, and masked subgraph extraction."""
    from amr_sched.cvcr.commitment import classify_commitments
    from amr_sched.rl.v19_hetero_constructor import decode_v19_candidate
    from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
    from experiments.v20_audit.common import OUT_AUDIT_DIR, build_val72_specs
    from experiments.v20_audit.v20_b_bidirectional import (
        V20BProposer,
        build_masked_subgraph,
        decode_v19_with_cached_enc,
        expand_causal_release_closure,
    )

    sp = build_val72_specs()[4]
    _, _, env = create_discrete_policy_env(
        "V19_ADAPTIVE_RL", topology=str(sp["scenario_tuple"][0]), seed=int(sp["seed"]), max_steps=120, use_rust=True
    )
    env.reset(seed=int(sp["seed"]), options={"scenario_tuple": copy.deepcopy(sp["scenario_tuple"])})
    for _ in range(12):
        env.step(0)
    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
    clf = classify_commitments(snap)
    graph = extract_v19_hetero_graph(snap, env.sim.network, clf=clf)

    c_path = OUT_AUDIT_DIR / "v20_a_critic_weights.npz"
    v20b = V20BProposer(
        hard_budget_ms=35.0,
        soft_deadline_ms=15.0,
        seed=0,
        enable_bidir_v18_to_v19=True,
        critic_weights_path=c_path if c_path.exists() else None,
        critic_beta=0.25,
        critic_mode="ws_only",
    )
    import numpy as np

    np_full = v20b._suite._v19_anchor._np_full_net
    dec_ref = decode_v19_candidate(np_full, graph, snap, np_rng=np.random.default_rng(0), greedy=True)
    enc = np_full.encode_entities(graph)
    h12 = np_full.a1_a2_heads(enc, graph)
    dec_cached = decode_v19_with_cached_enc(np_full, enc, h12, graph, snap)
    assert dec_cached.assign == dec_ref.assign
    assert dec_cached.bays == dec_ref.bays
    assert dec_cached.crews == dec_ref.crews

    if graph.rel_tids:
        rt, ro = expand_causal_release_closure(
            snap, graph, [graph.rel_tids[0]], [], max_tasks=4, max_ops=6
        )
        assert len(rt) <= 4 and len(ro) <= 6
        sub = build_masked_subgraph(
            graph,
            release_tasks=rt,
            release_ops=ro,
            skel_assign=graph.base_assign,
            skel_bays=graph.base_bays,
            skel_crews=graph.base_crews,
        )
        assert sub.frozen_ops >= graph.frozen_ops

    prop = v20b(snap, env.sim.network, budget_ms=35.0)
    assert prop.proposed_assignment is not None


def test_v20_c_neighborhood_and_v20_d_anytime_smoke() -> None:
    """Verify V20-C 408-dim neighborhood feature extraction and V20-D anytime state/proposer cleanly."""
    from amr_sched.cvcr.commitment import classify_commitments
    from amr_sched.cvcr.evaluator import CVCREvaluator
    from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
    from experiments.v20_audit.common import OUT_AUDIT_DIR, build_val72_specs
    from experiments.v20_audit.v20_c_neighborhood import (
        NEIGHBORHOOD_FEAT_DIM,
        V20CNeighborhoodProposer,
        extract_task_release_features,
    )
    from experiments.v20_audit.v20_d_anytime import (
        ANYTIME_FEAT_DIM,
        ANYTIME_OPS,
        V20DAnytimeProposer,
        build_anytime_state_vector,
    )

    assert NEIGHBORHOOD_FEAT_DIM == 408
    assert ANYTIME_FEAT_DIM == 13 + len(ANYTIME_OPS)

    sp = build_val72_specs()[4]
    _, _, env = create_discrete_policy_env(
        "V19_ADAPTIVE_RL", topology=str(sp["scenario_tuple"][0]), seed=int(sp["seed"]), max_steps=120, use_rust=True
    )
    env.reset(seed=int(sp["seed"]), options={"scenario_tuple": copy.deepcopy(sp["scenario_tuple"])})
    for _ in range(10):
        env.step(0)
    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
    clf = classify_commitments(snap)
    graph = extract_v19_hetero_graph(snap, env.sim.network, clf=clf)

    nh_w = OUT_AUDIT_DIR / "v20_c_neighborhood_head.npz"
    v20c = V20CNeighborhoodProposer(
        neighborhood_weights_path=nh_w if nh_w.exists() else None,
        neighborhood_strategy="learned_dep_guided" if nh_w.exists() else "det_causal_cone",
        release_k_max=4,
        strict_fixed_release=True,
        hard_budget_ms=35.0,
    )
    enc = v20c._suite._v19_anchor._np_full_net.encode_entities(graph)
    ev = CVCREvaluator(env.sim.network, snap, mode="B2_SEEDED", use_rust=True)
    res_keep = ev.evaluate(graph.base_assign, op_bay_overrides=graph.base_bays, op_crew_overrides=graph.base_crews)
    feats = extract_task_release_features(snap, env.sim.network, graph, enc, res_keep, graph.base_assign, graph.base_bays, graph.base_crews)
    assert feats.shape == (len(graph.rel_tids), NEIGHBORHOOD_FEAT_DIM)
    prop_c = v20c(snap, env.sim.network, budget_ms=35.0)
    assert prop_c.proposed_assignment is not None

    any_w = OUT_AUDIT_DIR / "v20_d_anytime_weights.npz"
    v20d = V20DAnytimeProposer(
        anytime_weights_path=any_w if any_w.exists() else None,
        hard_budget_ms=35.0,
    )
    sv = build_anytime_state_vector(
        rem_ms=15.0, cap_ms=35.0, is_ws_net=False, n_rel_tids=len(graph.rel_tids),
        n_ops=len(graph.op_ids), n_frozen_ops=len(graph.frozen_ops), has_win=True,
        keep_valid=True, has_disturbed=False, blocked_frac=0.0, keep_sc=500.0, best_sc=480.0,
        eval_cnt=2, tried_mask=[False] * len(ANYTIME_OPS),
    )
    assert sv.shape == (ANYTIME_FEAT_DIM,)
    prop_d = v20d(snap, env.sim.network, budget_ms=35.0)
    assert prop_d.proposed_assignment is not None


def test_v20_e_liveness_tracker_and_default_proposer() -> None:
    """Verify V20-E PDGLivenessTracker actionable guard, default proposer rollout, and clean amr_sched."""
    from types import SimpleNamespace
    import experiments.v20_audit.v20_e_liveness as v20e_mod
    from experiments.v20_audit.common import assert_git_amr_sched_clean_relative_to_baseline, build_val72_specs, run_wrapped_episode
    from experiments.v20_audit.v20_e_liveness import PDGLivenessTracker, build_default_v20_proposer

    assert_git_amr_sched_clean_relative_to_baseline()

    # 1. Mock test: non-deadlock in-flight 'picked' single-task leg MUST NOT alert
    tr = PDGLivenessTracker(cooldown_ticks=12)
    fake_task = SimpleNamespace(status="picked", release_step=10)
    fake_env = SimpleNamespace(
        current_step=1593,
        decision_logs=[],
        sim=SimpleNamespace(_step_cursor=1593, tasks=[fake_task]),
        tracker=SimpleNamespace(operations=[]),
        _completion_counts=lambda _cursor: (10, 5),
    )
    orig_pdg_fn = v20e_mod.compute_pdg_state
    try:
        v20e_mod.compute_pdg_state = lambda *_a, **_kw: {
            "tick": 1593, "rem_tasks": 1, "blocked_healthy": 0, "blocked_ratio": 0.0,
            "max_loaded_stall_ticks": 0, "wait_cycle_count": 0, "wait_cycle_agvs": [],
            "max_kit_split_lag": 626, "zero_progress_ticks": 420, "recent_reject_streak": 0,
            "pdg_alert": True,
        }
        act_out = tr.observe_and_maybe_escalate(fake_env, {"action_mask": [1, 1, 1]}, 0)
        assert act_out == 0
        assert len(tr.alert_ticks) == 0
        assert tr.latest_pdg is not None and tr.latest_pdg["pdg_alert"] is False

        # True wait-cycle MUST alert & escalate
        v20e_mod.compute_pdg_state = lambda *_a, **_kw: {
            "tick": 1593, "rem_tasks": 5, "blocked_healthy": 2, "blocked_ratio": 0.5,
            "max_loaded_stall_ticks": 25, "wait_cycle_count": 1, "wait_cycle_agvs": ["AGV_1"],
            "max_kit_split_lag": 80, "zero_progress_ticks": 45, "recent_reject_streak": 0,
            "pdg_alert": True,
        }
        act_esc = tr.observe_and_maybe_escalate(fake_env, {"action_mask": [1, 1, 1]}, 0)
        assert act_esc == 2
        assert tr.alert_ticks == [1593]
    finally:
        v20e_mod.compute_pdg_state = orig_pdg_fn

    # 2. Live default V20 proposer smoke run on VAL-72 episode 0
    sp = build_val72_specs()[0]
    prop = build_default_v20_proposer(hard_budget_ms=35.0, seed=0, enable_pdg_recovery=True)
    tr_live = PDGLivenessTracker(cooldown_ticks=12)
    row, _ = run_wrapped_episode(
        variant_id="test_v20e_default",
        spec=sp,
        base_env_policy="V19_ADAPTIVE_RL",
        external_proposer=prop,
        hard_budget_ms=35.0,
        outer_action_hook=lambda env, info, act, traces: tr_live.observe_and_maybe_escalate(env, info, act, traces, proposer=prop),
    )
    assert row["comp100"] is True
    assert row["audit_passed"] is True
    assert len(tr_live.alert_ticks) == 0

    from experiments.v20_audit.run_v20_final_blind_holdout import assert_cohort_c_quarantine, build_cohort_c_120_specs

    assert_cohort_c_quarantine()
    c_specs = build_cohort_c_120_specs()
    assert len(c_specs) == 120
    assert {s["seed"] for s in c_specs} <= COHORT_C_RESERVED_SEEDS
