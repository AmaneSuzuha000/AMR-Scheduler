"""Phase V20-D: Train Contextual Anytime VOI+STOP Controller on TRAIN & Benchmark 10/20/35/50ms on VAL."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.baselines.policy_registry import create_discrete_policy_env
from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.policy import LGCVCRPolicy
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
from amr_sched.cvcr.v18_frontier_synthesis import compute_semicircle_kitting_bays
from amr_sched.rl.causal_repair_cone import (
    FastJAWKittingSurrogate,
    _compute_fresh_semicircle_bays,
    _regret2_insert_tasks,
    build_causal_repair_cone,
    generate_causal_cone_candidates,
    observable_op_closure,
    project_zero_churn_candidate,
)
from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
from experiments.v20_audit.common import (
    OUT_AUDIT_DIR,
    _select_ctrl_action,
    assert_stage1_quarantine,
    build_stratified_train_val_specs,
    build_val72_specs,
    build_val_static35_specs,
    run_wrapped_episode,
    summarize_episode_rows,
)
from experiments.v20_audit.proposer_variants import (
    _LazyKittingSurrogate,
    _default_task_order_from_prio,
    _healthy_robots_from_graph,
    derive_base_op_prio,
    WindowAwareKittingSurrogate,
)
from experiments.v20_audit.v20_b_bidirectional import (
    V20BProposer,
    build_masked_subgraph,
    decode_v19_with_cached_enc,
    expand_causal_release_closure,
)
from experiments.v20_audit.v20_d_anytime import (
    ANYTIME_FEAT_DIM,
    ANYTIME_OPS,
    V20DAnytimeProposer,
    build_anytime_state_vector,
)

ANYTIME_DATASET_NPZ = OUT_AUDIT_DIR / "v20_d_anytime_dataset.npz"
ANYTIME_WEIGHTS_NPZ = OUT_AUDIT_DIR / "v20_d_anytime_weights.npz"
CRITIC_NPZ = OUT_AUDIT_DIR / "v20_a_critic_weights.npz"


def _collect_episode_anytime_steps(job: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    """Worker: collect unbiased per-operator VOI (delta_J / dt_ms) trajectories on TRAIN."""
    ep_idx, spec = job
    seed = int(spec["seed"])
    max_steps = int(spec["max_steps"])
    sc_copy = copy.deepcopy(spec["scenario_tuple"])
    p2_copy = copy.deepcopy(spec["p2_cfg"])

    kind, ctrl, env = create_discrete_policy_env(
        "V19_ADAPTIVE_RL",
        topology=str(sc_copy[0]),
        seed=seed,
        max_steps=max_steps,
        use_rust=True,
        p2_cfg=p2_copy,
        allow_missing_pgmpo=False,
    )
    prop_worker = V20BProposer(
        hard_budget_ms=35.0, soft_deadline_ms=15.0, seed=0,
        enable_e4=True, e4_selector="fast_surr", e4_include_zc=False,
        e4_mode="ws_or_invalid", staged_mode="ws_or_invalid",
        enable_bidir_v18_to_v19=True, enable_e6_static_or_headroom=False,
        enable_e7=True, enable_e9=False, enable_early_stop=False, compact_calm_ms=None,
        enable_r2_global_ws=False, r2_zc_ws_mode="never", include_staged_g="on_invalid",
        unwin_seq_horizon=2, seq_diff_b2=True,
    )
    env.external_proposer = prop_worker
    np_full = prop_worker._suite._v19_anchor._np_full_net
    np_staged = prop_worker._suite._v19_anchor._np_staged_net

    obs, info = env.reset(seed=seed, options={"scenario_tuple": sc_copy})
    tm = False
    tr = False
    rng = np.random.default_rng(20261125 + ep_idx * 79 + seed)

    feat_list: list[np.ndarray] = []
    op_list: list[int] = []
    voi_list: list[float] = []
    ep_list: list[int] = []

    collected = 0
    cursor = 0
    while not (tm or tr) and collected < 5:
        act = _select_ctrl_action(kind, ctrl, env, obs, info)
        if act != 0 and (cursor % 2 == 0):
            snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
            clf = classify_commitments(snap)
            graph = extract_v19_hetero_graph(snap, env.sim.network, clf=clf)
            healthy = _healthy_robots_from_graph(graph)
            if healthy and graph.rel_tids:
                base_prio = derive_base_op_prio(snap)
                ev_b2 = CVCREvaluator(
                    env.sim.network, snap, mode="B2_SEEDED", alpha_makespan=3.0, lambda_change=15.0,
                    use_rust=True, seq_diff_b2=True,
                )
                dd_sig = compute_detour_delta_signals(env.sim.network, snap)
                bay_delta = dd_sig["bay_detour_delta"] if dd_sig else {}
                lazy_surr = _LazyKittingSurrogate(
                    env.sim.network, snap, base_prio, alpha_makespan=3.0, lambda_change=15.0, bay_detour_delta=bay_delta
                )
                w_surr = WindowAwareKittingSurrogate(lazy_surr, ev_b2, snap)
                fast_surr = FastJAWKittingSurrogate(env.sim.network, snap, ev_b2, graph, lambda_change=15.0)
                fresh_ops = observable_op_closure(snap)
                rob_cur_h = {
                    r.robot_id: r.cur_task for r in snap.robots if r.cur_task and r.robot_id not in clf.disturbed_robots
                }
                bays_semi, crews_semi = compute_semicircle_kitting_bays(snap, env.sim.network, clf, lazy_surr, graph.base_bays, bay_delta)
                _, _, km_prio = _compute_fresh_semicircle_bays(snap, env.sim.network, graph, fast_surr, fresh_ops)
                eff_km_prio = km_prio or base_prio
                pending_or_dist = prop_worker._compute_pending_or_disturbed(snap, graph, clf)
                has_win = any(getattr(op, "resource_windows", ()) for op in snap.operations if not getattr(op, "completed", False))
                is_ws_net = len(env.sim.network.nodes) >= 30
                blocked_frac = float(
                    sum(1 for r in snap.robots if str(getattr(r, "state", "")).lower() in ("blocked", "faulted"))
                ) / float(max(1, len(snap.robots)))

                from experiments.v20_audit.common import CandidateArtifact
                keep_art = CandidateArtifact(
                    "E0_KEEP", "keep", dict(graph.base_assign), dict(graph.base_bays), dict(graph.base_crews),
                    dict(base_prio), None, False,
                )
                prop_worker._suite._finish_artifact(
                    keep_art, snapshot=snap, graph=graph, ev_b2=ev_b2, w_surr=w_surr,
                    rob_cur_h=rob_cur_h, obs_ops=fresh_ops, fast_surr=fast_surr, compute_details=False,
                )
                keep_eval = keep_art.eval_res_b2
                keep_sc = float(keep_art.ja_unified_score or 1e9)
                keep_seq = {k: list(v) for k, v in keep_eval.robot_order_seq.items() if v} if keep_eval.robot_order_seq else None
                ev_b2.set_incumbent(
                    graph.base_assign, keep_eval, op_priority=base_prio, op_bay_overrides=graph.base_bays, task_order_override=keep_seq
                )

                enc_f = np_full.encode_entities(graph)
                h12_f = np_full.a1_a2_heads(enc_f, graph)
                enc_s = np_staged.encode_entities(graph)
                h12_s = np_staged.a1_a2_heads(enc_s, graph)

                best_sc = keep_sc
                best_art = keep_art
                tried = [False] * len(ANYTIME_OPS)
                nn_pool: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Any, bool]] = []
                cap_ms = float(rng.choice([10.0, 20.0, 35.0, 50.0]))
                rem_ms = cap_ms

                # Permute operator trajectory to collect unbiased state-action VOI samples
                perm = rng.permutation(len(ANYTIME_OPS)).tolist()
                for m_idx in perm:
                    op_nm = ANYTIME_OPS[m_idx]
                    fvec = build_anytime_state_vector(
                        rem_ms=rem_ms, cap_ms=cap_ms, is_ws_net=is_ws_net,
                        n_rel_tids=len(graph.rel_tids), n_ops=len(graph.op_ids), n_frozen_ops=len(graph.frozen_ops),
                        has_win=has_win, keep_valid=bool(keep_eval.valid), has_disturbed=bool(clf.disturbed_robots),
                        blocked_frac=blocked_frac, keep_sc=keep_sc, best_sc=best_sc,
                        eval_cnt=len(nn_pool) + 1, tried_mask=tried,
                    )
                    t_s0 = _cvcr_now_s()
                    before = best_sc
                    cands: list[CandidateArtifact] = []
                    if op_nm == "op_regret2_local":
                        if not is_ws_net:
                            a_g, sq_g = _regret2_insert_tasks(snap, graph, fast_surr, dict(graph.fixed_assign), list(graph.rel_tids), eff_km_prio, bays_semi)
                            cands.append(CandidateArtifact("E1", "r2_g", a_g, dict(bays_semi), dict(crews_semi), dict(eff_km_prio), sq_g, True))
                    elif op_nm == "op_ahg_full":
                        df = decode_v19_with_cached_enc(np_full, enc_f, h12_f, graph, snap)
                        cands.append(CandidateArtifact("E2", "ahg_f", dict(df.assign), dict(df.bays), dict(df.crews), dict(df.op_prio), df.task_sequences, True))
                    elif op_nm == "op_ahg_staged":
                        ds = decode_v19_with_cached_enc(np_staged, enc_s, h12_s, graph, snap)
                        za, zb, zc, zp, zsq = project_zero_churn_candidate(
                            snap, graph, ds.assign, ds.bays, ds.crews, ds.op_prio, ds.task_sequences, fresh_ops, pending_or_dist, rob_cur_h
                        )
                        cands.append(CandidateArtifact("E3", "ahg_sz", za, zb, zc, zp, zsq, False))
                    elif op_nm == "op_e4_pgmpo" and prop_worker._suite._v18_pol.pgmpo is not None:
                        asns = prop_worker._suite._v18_pol.pgmpo.generate_assignments(snap, env.sim.network, clf, healthy, graph.fixed_assign, bays_semi, lazy_surr)
                        if asns:
                            top_p = min(asns, key=lambda a: float(fast_surr.score(a, eff_km_prio, bays_semi, crews_semi, _default_task_order_from_prio(snap, graph, a, eff_km_prio))))
                            cands.append(CandidateArtifact("E4", "pgmpo", dict(top_p), dict(bays_semi), dict(crews_semi), dict(eff_km_prio), _default_task_order_from_prio(snap, graph, top_p, eff_km_prio), True))
                    elif op_nm == "op_bidir_v18_to_v19" and best_art.eval_res_b2 and best_art.eval_res_b2.valid:
                        cone = build_causal_repair_cone(snap, graph, best_art.eval_res_b2, best_art.assign, best_art.bays, best_art.crews, max_cone_amrs=3, max_cone_tasks=5, max_cone_ops=3)
                        rt_b, ro_b = expand_causal_release_closure(snap, graph, cone.cone_tasks, cone.cone_ops, max_tasks=6, max_ops=4)
                        if rt_b or ro_b:
                            sg = build_masked_subgraph(graph, release_tasks=rt_b, release_ops=ro_b, skel_assign=best_art.assign, skel_bays=best_art.bays, skel_crews=best_art.crews, restrict_agvs=set(cone.cone_amrs) if cone.cone_amrs else None)
                            db = decode_v19_with_cached_enc(np_full, enc_f, h12_f, sg, snap)
                            cands.append(CandidateArtifact("Bidir", "bidir", dict(db.assign), dict(db.bays), dict(db.crews), dict(db.op_prio), db.task_sequences, False))
                    elif op_nm == "op_e7_causal_cone":
                        nnt = nn_pool[:3] if nn_pool else [(best_art.tag, best_art.assign, best_art.bays, best_art.crews, best_art.op_prio, best_art.task_orders, best_art.is_full)]
                        for tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc in generate_causal_cone_candidates(snap, env.sim.network, graph, ev_b2, fast_surr, keep_eval, base_prio, nnt, fresh_ops, pending_or_dist, budget_ms_limit=1.2)[:2]:
                            cands.append(CandidateArtifact("E7", tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc))
                    elif op_nm == "op_e9_cpsat_focal" and rem_ms >= 7.0 and best_art.eval_res_b2 and best_art.eval_res_b2.valid:
                        cpo = LGCVCRPolicy("V18_EA_JOINT", env.sim.network, use_rust=True)
                        a_cp, b_cp, cr_cp, pr_cp, ord_cp, res_cp, _ = cpo._search_local_cpsat_window(
                            snap, clf, ev_b2, dict(best_art.assign), dict(best_art.bays), dict(best_art.crews),
                            best_art.eval_res_b2, healthy, dict(best_art.op_prio), _cvcr_now_s() + 0.005, lazy_surr,
                            allow_order_only=True, min_improve=0.1, solve_time_cap_s=0.004,
                        )
                        if res_cp.valid:
                            cands.append(CandidateArtifact("E9", "cpsat", dict(a_cp), dict(b_cp), dict(cr_cp), dict(pr_cp), {k: list(v) for k, v in (ord_cp or res_cp.robot_order_seq).items() if v} or None, False))

                    for c_art in cands:
                        prop_worker._suite._finish_artifact(
                            c_art, snapshot=snap, graph=graph, ev_b2=ev_b2, w_surr=w_surr,
                            rob_cur_h=rob_cur_h, obs_ops=fresh_ops, fast_surr=fast_surr, compute_details=False,
                        )
                        if c_art.eval_res_b2 and c_art.eval_res_b2.valid:
                            nn_pool.append((c_art.tag, c_art.assign, c_art.bays, c_art.crews, c_art.op_prio, c_art.task_orders, c_art.is_full))
                            sc_c = float(c_art.ja_unified_score or 1e9)
                            if sc_c + 1e-4 < best_sc:
                                best_sc = sc_c
                                best_art = c_art
                                ev_b2.set_incumbent(best_art.assign, best_art.eval_res_b2, op_priority=best_art.op_prio, op_bay_overrides=best_art.bays, task_order_override=best_art.task_orders)

                    dt_ms = max(0.25, (_cvcr_now_s() - t_s0) * 1000.0)
                    gain = max(0.0, before - best_sc)
                    voi_val = float(np.arcsinh(gain / 25.0) / dt_ms)
                    feat_list.append(fvec)
                    op_list.append(int(m_idx))
                    voi_list.append(voi_val)
                    ep_list.append(int(ep_idx))
                    tried[m_idx] = True
                    rem_ms = max(0.0, rem_ms - dt_ms)
                collected += 1
        obs, _, tm, tr, info = env.step(act)
        cursor += 1

    if not feat_list:
        return {"n_rows": 0}
    return {
        "n_rows": len(feat_list),
        "X": np.asarray(feat_list, dtype=np.float32),
        "op_idx": np.asarray(op_list, dtype=np.int64),
        "voi": np.asarray(voi_list, dtype=np.float32),
        "ep_id": np.asarray(ep_list, dtype=np.int64),
    }


def train_anytime_controller() -> dict[str, Any]:
    """Collect multi-budget anytime VOI trajectories strictly on TRAIN and fit Ridge VOI + STOP controller."""
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    all_specs = build_stratified_train_val_specs()
    train_specs = [sp for sp in all_specs if sp.get("split") == "train"]
    assert_stage1_quarantine([sp["seed"] for sp in train_specs], label="V20D_Train_Anytime")
    print(f"[V20-D] Collecting anytime VOI trajectories across {len(train_specs)} TRAIN episodes...", flush=True)

    xs, ops, vois, eids = [], [], [], []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for res in ex.map(_collect_episode_anytime_steps, [(i, sp) for i, sp in enumerate(train_specs)]):
            if res.get("n_rows", 0) > 0:
                xs.append(res["X"])
                ops.append(res["op_idx"])
                vois.append(res["voi"])
                eids.append(res["ep_id"])

    X = np.concatenate(xs, axis=0).astype(np.float32)
    op_idx = np.concatenate(ops, axis=0)
    voi = np.concatenate(vois, axis=0)
    ep_id = np.concatenate(eids, axis=0)
    np.savez_compressed(ANYTIME_DATASET_NPZ, X=X, op_idx=op_idx, voi=voi, ep_id=ep_id)

    def _fit(X_tr: np.ndarray, op_tr: np.ndarray, v_tr: np.ndarray):
        mu = X_tr.mean(axis=0).astype(np.float32)
        sig = np.maximum(1e-4, X_tr.std(axis=0)).astype(np.float32)
        Xn = np.clip((X_tr - mu[None, :]) / sig[None, :], -5.0, 5.0)
        W = np.zeros((len(ANYTIME_OPS), ANYTIME_FEAT_DIM), dtype=np.float32)
        b = np.zeros((len(ANYTIME_OPS),), dtype=np.float32)
        for m in range(len(ANYTIME_OPS)):
            mask = op_tr == m
            if int(mask.sum()) >= 10:
                reg = Ridge(alpha=50.0).fit(Xn[mask], v_tr[mask])
                W[m] = reg.coef_.astype(np.float32)
                b[m] = np.float32(reg.intercept_)
        return mu, sig, W, b

    uniq_eps = np.unique(ep_id)
    folds = np.array_split(np.random.default_rng(20261125).permutation(uniq_eps), 5)
    cv_r2s: list[float] = []
    for f_eps in folds:
        vm = np.isin(ep_id, f_eps)
        tm = ~vm
        mu, sig, W, b = _fit(X[tm], op_idx[tm], voi[tm])
        Xv = np.clip((X[vm] - mu[None, :]) / sig[None, :], -5.0, 5.0)
        preds = np.array([float(W[m] @ x + b[m]) for x, m in zip(Xv, op_idx[vm])], dtype=np.float32)
        yv = voi[vm]
        if float(np.std(preds)) > 1e-6 and float(np.std(yv)) > 1e-6:
            cv_r2s.append(float(np.corrcoef(preds, yv)[0, 1]))

    mu, sig, W, b = _fit(X, op_idx, voi)
    Xn_all = np.clip((X - mu[None, :]) / sig[None, :], -5.0, 5.0)
    preds_all = np.array([float(W[m] @ x + b[m]) for x, m in zip(Xn_all, op_idx)], dtype=np.float32)
    # Calibrate STOP threshold safely on expected predicted VOI distribution (20th percentile)
    stop_thr = float(np.clip(np.quantile(preds_all, 0.20), 0.015, 0.08))
    np.savez(ANYTIME_WEIGHTS_NPZ, mu=mu, sig=sig, W=W, b=b, stop_threshold=np.array([stop_thr], dtype=np.float32))
    info = {
        "n_steps": int(len(X)),
        "n_train_eps": int(len(uniq_eps)),
        "cv_5fold_pearson_r": round(float(np.mean(cv_r2s)) if cv_r2s else 0.0, 4),
        "calibrated_stop_threshold": round(stop_thr, 5),
        "per_op_mean_voi": {
            op_nm: round(float(np.mean(voi[op_idx == m])) if np.any(op_idx == m) else 0.0, 5)
            for m, op_nm in enumerate(ANYTIME_OPS)
        },
    }
    print(f"[V20-D] TRAIN 5-fold VOI r={info['cv_5fold_pearson_r']} stop_thr={stop_thr:.5f} steps={len(X)}", flush=True)
    return info


def run_multi_budget_pareto_and_gate_d() -> dict[str, Any]:
    """Benchmark BALANS_IJCAI25 vs V20B_Det_Rule vs V20D_Learned_Anytime at 10/20/35/50 ms on VAL-72."""
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    specs72 = build_val72_specs()
    specs35 = build_val_static35_specs()
    assert_stage1_quarantine([sp["seed"] for sp in specs72 + specs35], label="V20D_GateD_Eval")

    base_kw: dict[str, Any] = dict(
        enable_e4=True, e4_selector="fast_surr", e4_include_zc=False,
        e4_mode="ws_or_invalid", staged_mode="ws_or_invalid",
        enable_bidir_v18_to_v19=True, enable_e6_static_or_headroom=False,
        enable_e7=True, enable_e9=False, enable_early_stop=False, compact_calm_ms=None,
        enable_r2_global_ws=False, r2_zc_ws_mode="never", include_staged_g="on_invalid",
        unwin_seq_horizon=2, seq_diff_b2=True,
        critic_weights_path=CRITIC_NPZ, critic_beta=0.25, critic_mode="ws_only",
    )

    pareto: dict[str, Any] = {}
    raw_at_35: dict[str, list[dict[str, Any]]] = {}

    for b_ms in (10.0, 20.0, 35.0, 50.0):
        b_key = f"{int(b_ms)}ms"
        soft_ms = min(15.5, max(5.5, b_ms * 0.65)) if b_ms <= 35.0 else 17.5
        print(f"=== [V20-D] Multi-Budget Anytime Pareto Check @ {b_key} (soft={soft_ms}ms) ===", flush=True)

        # 1. BALANS_IJCAI25 baseline
        r_bal = [
            run_wrapped_episode(variant_id=f"bal_{b_key}", spec=sp, base_env_policy="BALANS_IJCAI25", hard_budget_ms=b_ms)[0]
            for sp in specs72
        ]
        ov_bal = summarize_episode_rows(r_bal)

        # 2. V20-B Strong Deterministic Rule Controller
        p_b = V20BProposer(hard_budget_ms=b_ms, soft_deadline_ms=soft_ms, seed=0, **base_kw)
        r_b = [
            run_wrapped_episode(variant_id=f"v20b_{b_key}", spec=sp, base_env_policy="V19_ADAPTIVE_RL", external_proposer=p_b, hard_budget_ms=b_ms)[0]
            for sp in specs72
        ]
        ov_b = summarize_episode_rows(r_b)

        # 3. V20-D Learned Contextual Anytime Controller
        p_d = V20DAnytimeProposer(
            hard_budget_ms=b_ms, soft_deadline_ms=soft_ms, seed=0,
            anytime_weights_path=ANYTIME_WEIGHTS_NPZ, **base_kw,
        )
        r_d = [
            run_wrapped_episode(variant_id=f"v20d_{b_key}", spec=sp, base_env_policy="V19_ADAPTIVE_RL", external_proposer=p_d, hard_budget_ms=b_ms)[0]
            for sp in specs72
        ]
        ov_d = summarize_episode_rows(r_d)

        diffs_db = np.array([float(x["net_obj"]) - float(y["net_obj"]) for x, y in zip(r_d, r_b)])
        p_db = float(wilcoxon(diffs_db, zero_method="wilcox", alternative="less").pvalue) if np.any(np.abs(diffs_db) > 1e-9) else 1.0
        diffs_dbal = np.array([float(x["net_obj"]) - float(y["net_obj"]) for x, y in zip(r_d, r_bal)])
        p_dbal = float(wilcoxon(diffs_dbal, zero_method="wilcox", alternative="less").pvalue) if np.any(np.abs(diffs_dbal) > 1e-9) else 1.0

        for nm, ov in (("BALANS_IJCAI25", ov_bal), ("V20B_Det_Rule", ov_b), ("V20D_Learned_Anytime", ov_d)):
            print(
                f"  {b_key} {nm:20s}: net={ov['mean_net_obj']:7.2f} comp={ov['comp100_count']}/72 "
                f"mean_ms={ov['mean_decision_ms']:6.3f} p95={ov['pooled_p95_ms']:6.3f} p99={ov['pooled_p99_ms']:6.3f}",
                flush=True,
            )
        print(f"  Wilcoxon p_less: V20D<V20B={p_db:.6f}, V20D<BALANS={p_dbal:.6f}", flush=True)
        pareto[b_key] = {
            "BALANS_IJCAI25": ov_bal,
            "V20B_Det_Rule": ov_b,
            "V20D_Learned_Anytime": ov_d,
            "wilcoxon_v20d_lt_v20b_p": round(p_db, 6),
            "wilcoxon_v20d_lt_balans_p": round(p_dbal, 6),
        }
        if abs(b_ms - 35.0) < 0.1:
            raw_at_35["b"] = r_b
            raw_at_35["d"] = r_d

    # Static-35 check on V20-D @ 35ms
    p_d35 = V20DAnytimeProposer(
        hard_budget_ms=35.0, soft_deadline_ms=15.0, seed=0, anytime_weights_path=ANYTIME_WEIGHTS_NPZ, **base_kw
    )
    r_d35 = [
        run_wrapped_episode(variant_id="v20d_s35", spec=sp, base_env_policy="V19_ADAPTIVE_RL", external_proposer=p_d35, hard_budget_ms=35.0)[0]
        for sp in specs35
    ]
    ov_d35 = summarize_episode_rows(r_d35)

    p35 = pareto["35ms"]
    v20d_promoted = bool(
        p35["V20D_Learned_Anytime"]["comp100_count"] == 72
        and p35["V20D_Learned_Anytime"]["pooled_p95_ms"] <= 28.0
        and ov_d35["comp100_count"] == 35
        and ov_d35["mean_h1_all"] <= 400.0
        and p35["V20D_Learned_Anytime"]["mean_net_obj"] < p35["V20B_Det_Rule"]["mean_net_obj"]
        and p35["wilcoxon_v20d_lt_v20b_p"] < 0.05
    )
    print(
        f"=== [V20-D / Gate D] Static-35 H1={ov_d35['mean_h1_all']} ({ov_d35['comp100_count']}/35) | "
        f"V20D_Promoted={v20d_promoted} ===",
        flush=True,
    )
    return {
        "schema": "v20_d.pareto_gate_eval.v1",
        "pareto_by_budget": pareto,
        "v20d_static35": ov_d35,
        "v20d_promoted": v20d_promoted,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()

    train_info = None
    eval_info = None
    if not args.eval_only:
        train_info = train_anytime_controller()
    if not args.train_only:
        eval_info = run_multi_budget_pareto_and_gate_d()

    rep = {}
    rep_path = OUT_AUDIT_DIR / "v20_d_gate_eval.json"
    if rep_path.exists():
        try:
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
        except Exception:
            rep = {}
    if train_info is not None:
        rep["train_summary"] = train_info
    if eval_info is not None:
        rep.update(eval_info)
    rep_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
