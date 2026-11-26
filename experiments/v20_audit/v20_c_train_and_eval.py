"""Phase V20-C: Train Commitment-Aware Neural Neighborhood Head on TRAIN + Run RQ1/RQ2 & Gate C on VAL."""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.baselines.policy_registry import create_discrete_policy_env
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.rl.causal_repair_cone import (
    FastJAWKittingSurrogate,
    _regret2_insert_tasks,
    build_causal_repair_cone,
    observable_op_closure,
    score_candidate_ja_unified,
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
from experiments.v20_audit.proposer_variants import _healthy_robots_from_graph, derive_base_op_prio
from experiments.v20_audit.v20_b_bidirectional import (
    V20BProposer,
    build_masked_subgraph,
    decode_v19_with_cached_enc,
    expand_causal_release_closure,
)
from experiments.v20_audit.v20_c_neighborhood import (
    NEIGHBORHOOD_FEAT_DIM,
    NumpyNeighborhoodSelector,
    V20CNeighborhoodProposer,
    extract_task_release_features,
    select_release_neighborhood,
)

TRAIN_DATASET_NPZ = OUT_AUDIT_DIR / "v20_c_neighborhood_dataset.npz"
WEIGHTS_NPZ = OUT_AUDIT_DIR / "v20_c_neighborhood_head.npz"
CRITIC_NPZ = OUT_AUDIT_DIR / "v20_a_critic_weights.npz"


def _collect_episode_counterfactuals(job: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    """Worker: collect counterfactual neighborhood release credit labels strictly on one TRAIN episode."""
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
        hard_budget_ms=35.0,
        soft_deadline_ms=15.0,
        seed=0,
        enable_e4=True,
        e4_selector="fast_surr",
        e4_include_zc=False,
        e4_mode="ws_or_invalid",
        staged_mode="ws_or_invalid",
        enable_bidir_v18_to_v19=True,
        enable_e6_static_or_headroom=False,
        enable_e7=True,
        enable_e9=False,
        enable_early_stop=False,
        compact_calm_ms=None,
        enable_r2_global_ws=False,
        r2_zc_ws_mode="never",
        include_staged_g="on_invalid",
        unwin_seq_horizon=2,
        seq_diff_b2=True,
    )
    np_full = prop_worker._suite._v19_anchor._np_full_net
    env.external_proposer = prop_worker

    obs, info = env.reset(seed=seed, options={"scenario_tuple": sc_copy})
    tm = False
    tr = False
    rng = np.random.default_rng(20261125 + ep_idx * 131 + seed)

    feat_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    snap_ids: list[np.ndarray] = []
    ep_ids: list[np.ndarray] = []

    collected_snaps = 0
    tick_cursor = 0
    while not (tm or tr) and collected_snaps < 8:
        act = _select_ctrl_action(kind, ctrl, env, obs, info)
        if act != 0 and (tick_cursor % 2 == 0):
            snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
            clf = classify_commitments(snap)
            graph = extract_v19_hetero_graph(snap, env.sim.network, clf=clf)
            healthy = _healthy_robots_from_graph(graph)
            if healthy and len(graph.rel_tids) >= 3:
                base_prio = derive_base_op_prio(snap)
                ev_b2 = CVCREvaluator(
                    env.sim.network, snap, mode="B2_SEEDED", alpha_makespan=3.0, lambda_change=15.0,
                    use_rust=True, seq_diff_b2=True,
                )
                fast_surr = FastJAWKittingSurrogate(env.sim.network, snap, ev_b2, graph, lambda_change=15.0)
                fresh_ops = observable_op_closure(snap)
                keep_eval = ev_b2.evaluate(
                    graph.base_assign, op_priority=base_prio, op_bay_overrides=graph.base_bays,
                    op_crew_overrides=graph.base_crews, task_order_override=None,
                )
                sim_h = int(ev_b2.sim_horizon)
                keep_ja, _ = score_candidate_ja_unified(
                    snap, env.sim.network, graph, keep_eval, graph.base_assign, graph.base_bays, graph.base_crews,
                    fresh_ops, fast_surr, lambda_change=15.0, sim_horizon=sim_h,
                )
                keep_seq = {k: list(v) for k, v in keep_eval.robot_order_seq.items() if v} if keep_eval.robot_order_seq else None
                ev_b2.set_incumbent(
                    graph.base_assign, keep_eval, op_priority=base_prio, op_bay_overrides=graph.base_bays, task_order_override=keep_seq
                )

                enc = np_full.encode_entities(graph)
                h12 = np_full.a1_a2_heads(enc, graph)
                feats = extract_task_release_features(
                    snap, env.sim.network, graph, enc, keep_eval, graph.base_assign, graph.base_bays, graph.base_crews
                )

                # Paired counterfactual do-intervention per task: singleton {t_i}+kit_op AND LOO (U \ {t_i} vs U | {t_i})
                N_t = len(graph.rel_tids)
                t_by_id = {t.task_id: t for t in snap.tasks}

                def _eval_subset(rt_set: set[str], ro_set: set[str]) -> float:
                    if not rt_set:
                        return float(keep_ja)
                    best_j = float(keep_ja)
                    sub_g = build_masked_subgraph(
                        graph, release_tasks=rt_set, release_ops=ro_set,
                        skel_assign=graph.base_assign, skel_bays=graph.base_bays, skel_crews=graph.base_crews,
                    )
                    dec = decode_v19_with_cached_enc(np_full, enc, h12, sub_g, snap)
                    res_v = ev_b2.evaluate(
                        dec.assign, op_priority=dec.op_prio, op_bay_overrides=dec.bays,
                        op_crew_overrides=dec.crews, task_order_override=dec.task_sequences,
                    )
                    if res_v.valid:
                        ja_v, _ = score_candidate_ja_unified(
                            snap, env.sim.network, graph, res_v, dec.assign, dec.bays, dec.crews,
                            fresh_ops, fast_surr, lambda_change=15.0, sim_horizon=sim_h,
                        )
                        best_j = min(best_j, float(ja_v))
                    anc = {tid: aid for tid, aid in graph.base_assign.items() if tid not in rt_set}
                    a_r, sq_r = _regret2_insert_tasks(snap, graph, fast_surr, anc, sorted(rt_set), base_prio, graph.base_bays)
                    res_r = ev_b2.evaluate(
                        a_r, op_priority=base_prio, op_bay_overrides=graph.base_bays,
                        op_crew_overrides=graph.base_crews, task_order_override=sq_r,
                    )
                    if res_r.valid:
                        ja_r, _ = score_candidate_ja_unified(
                            snap, env.sim.network, graph, res_r, a_r, graph.base_bays, graph.base_crews,
                            fresh_ops, fast_surr, lambda_change=15.0, sim_horizon=sim_h,
                        )
                        best_j = min(best_j, float(ja_r))
                    return best_j

                k_base = min(N_t, max(3, min(6, N_t // 2 + 1)))
                rt_base, ro_base, _ = select_release_neighborhood(
                    "det_causal_cone", k_max=k_base, snapshot=snap, network=env.sim.network, graph=graph,
                    enc=enc, inc_eval=keep_eval, inc_assign=graph.base_assign,
                    inc_bays=graph.base_bays, inc_crews=graph.base_crews, rng=rng,
                )
                j_base = _eval_subset(rt_base, ro_base)

                raw_credit = np.zeros((N_t,), dtype=np.float32)
                for i, tid in enumerate(graph.rel_tids):
                    oid = t_by_id[tid].kit_op_id
                    ro_i = {oid} if (oid and oid not in graph.frozen_ops) else set()
                    # Intervention 1: singleton unlock gain of task t_i (+ its op bay/crew)
                    j_single = _eval_subset({tid}, ro_i)
                    single_gain = max(0.0, float(keep_ja - j_single))
                    # Intervention 2: exact leave-one-out / add-one-in counterfactual delta around rt_base
                    if tid in rt_base:
                        rt_wo = rt_base - {tid}
                        j_wo = _eval_subset(rt_wo, ro_base)
                        marg = max(0.0, float(j_wo - j_base))
                    else:
                        rt_wi = rt_base | {tid}
                        j_wi = _eval_subset(rt_wi, ro_base | ro_i)
                        marg = max(0.0, float(j_base - j_wi))
                    raw_credit[i] = float(np.arcsinh((0.6 * marg + 0.4 * single_gain) / 20.0))

                if float(np.std(raw_credit)) > 1e-4:
                    z_cred = (raw_credit - float(np.mean(raw_credit))) / max(1e-4, float(np.std(raw_credit)))
                    global_sid = ep_idx * 100 + collected_snaps
                    feat_chunks.append(feats)
                    label_chunks.append(z_cred.astype(np.float32))
                    snap_ids.append(np.full((N_t,), global_sid, dtype=np.int64))
                    ep_ids.append(np.full((N_t,), ep_idx, dtype=np.int64))
                    collected_snaps += 1
        obs, _, tm, tr, info = env.step(act)
        tick_cursor += 1

    if not feat_chunks:
        return {"n_snaps": 0}
    return {
        "n_snaps": collected_snaps,
        "X": np.concatenate(feat_chunks, axis=0),
        "y": np.concatenate(label_chunks, axis=0),
        "snap_id": np.concatenate(snap_ids, axis=0),
        "ep_id": np.concatenate(ep_ids, axis=0),
    }


def train_neighborhood_head() -> dict[str, Any]:
    """Collect counterfactual TRAIN labels and train+validate MLP selector cleanly."""
    import torch
    import torch.nn as nn

    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    all_specs = build_stratified_train_val_specs()
    train_specs = [sp for sp in all_specs if sp.get("split") == "train"]
    assert_stage1_quarantine([sp["seed"] for sp in train_specs], label="V20C_Train_Neighborhood")
    print(f"[V20-C] Collecting counterfactual release labels across {len(train_specs)} TRAIN episodes...", flush=True)

    jobs = [(i, sp) for i, sp in enumerate(train_specs)]
    xs, ys, sids, eids = [], [], [], []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for res in ex.map(_collect_episode_counterfactuals, jobs):
            if res.get("n_snaps", 0) > 0:
                xs.append(res["X"])
                ys.append(res["y"])
                sids.append(res["snap_id"])
                eids.append(res["ep_id"])

    X = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    snap_id = np.concatenate(sids, axis=0)
    ep_id = np.concatenate(eids, axis=0)
    np.savez_compressed(TRAIN_DATASET_NPZ, X=X, y=y, snap_id=snap_id, ep_id=ep_id)
    n_snaps = int(len(np.unique(snap_id)))
    n_eps = int(len(np.unique(ep_id)))
    print(f"[V20-C] Collected {len(X)} task rows across {n_snaps} snapshots / {n_eps} TRAIN episodes.", flush=True)

    # 5-fold GroupKFold by ep_id on TRAIN
    uniq_eps = np.unique(ep_id)
    rng = np.random.default_rng(20261125)
    perm_eps = rng.permutation(uniq_eps)
    folds = np.array_split(perm_eps, 5)

    from sklearn.linear_model import Ridge

    def _fit_mlp(X_tr: np.ndarray, y_tr: np.ndarray, sid_tr: np.ndarray, *, epochs: int = 45, wd: float = 2e-1):
        torch.manual_seed(0)
        mu = X_tr.mean(axis=0).astype(np.float32)
        sig = np.maximum(1e-4, X_tr.std(axis=0)).astype(np.float32)
        # Scale down 384 high-dim Layer-1 channels relative to 24 cross-domain phys dims to prevent overfitting
        sig[:-24] = sig[:-24] / 0.15
        Xn = np.clip((X_tr - mu[None, :]) / sig[None, :], -6.0, 6.0)
        w_lin = Ridge(alpha=400.0).fit(Xn, y_tr).coef_.astype(np.float32).reshape(-1)
        xt = torch.from_numpy(Xn)
        yt = torch.from_numpy(y_tr)

        pair_i, pair_j = [], []
        for s in np.unique(sid_tr):
            idx = np.where(sid_tr == s)[0]
            if len(idx) < 2:
                continue
            ys_loc = y_tr[idx]
            for a_i in range(len(idx)):
                for b_i in range(len(idx)):
                    if ys_loc[a_i] > ys_loc[b_i] + 0.25:
                        pair_i.append(int(idx[a_i]))
                        pair_j.append(int(idx[b_i]))
        pi_t = torch.tensor(pair_i, dtype=torch.long)
        pj_t = torch.tensor(pair_j, dtype=torch.long)

        net = nn.Sequential(
            nn.Linear(NEIGHBORHOOD_FEAT_DIM, 32),
            nn.ELU(),
            nn.Dropout(p=0.25),
            nn.Linear(32, 1),
        )
        opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=wd)
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            pred = net(xt).squeeze(-1)
            loss = torch.mean((pred - yt) ** 2)
            if len(pi_t) > 0:
                loss = loss + 0.5 * torch.mean(torch.nn.functional.softplus(-(pred[pi_t] - pred[pj_t])))
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            w1 = net[0].weight.detach().numpy().T.copy()
            b1 = net[0].bias.detach().numpy().copy()
            w2 = net[3].weight.detach().numpy().T.copy()
            b2 = net[3].bias.detach().numpy().reshape(()).copy()
        return mu, sig, w1, b1, w2, b2, w_lin

    def _eval_fold(mu, sig, w1, b1, w2, b2, w_lin, X_v, y_v, sid_v) -> tuple[float, float]:
        Xn = np.clip((X_v - mu[None, :]) / sig[None, :], -6.0, 6.0)
        h = np.where(Xn @ w1 + b1[None, :] >= 0.0, Xn @ w1 + b1[None, :], np.expm1(np.clip(Xn @ w1 + b1[None, :], -20.0, 0.0)))
        sc = 0.25 * (h @ w2 + b2).reshape(-1) + 0.75 * (Xn @ w_lin).reshape(-1)
        conc_num, conc_den = 0, 0
        top1_hits, top1_cnt = 0, 0
        for s in np.unique(sid_v):
            idx = np.where(sid_v == s)[0]
            if len(idx) < 2:
                continue
            p_loc, y_loc = sc[idx], y_v[idx]
            if int(np.argmax(p_loc)) in np.where(y_loc >= float(np.max(y_loc)) - 0.15)[0]:
                top1_hits += 1
            top1_cnt += 1
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    if abs(y_loc[i] - y_loc[j]) > 0.15:
                        conc_den += 1
                        if (p_loc[i] - p_loc[j]) * (y_loc[i] - y_loc[j]) > 0:
                            conc_num += 1
        return conc_num / max(1, conc_den), top1_hits / max(1, top1_cnt)

    concs, top1s = [], []
    for f_eps in folds:
        val_mask = np.isin(ep_id, f_eps)
        tr_mask = ~val_mask
        mu, sig, w1, b1, w2, b2, w_lin = _fit_mlp(X[tr_mask], y[tr_mask], snap_id[tr_mask])
        c_f, t_f = _eval_fold(mu, sig, w1, b1, w2, b2, w_lin, X[val_mask], y[val_mask], snap_id[val_mask])
        concs.append(c_f)
        top1s.append(t_f)

    # Fit final export weights on 100% of TRAIN dataset
    mu, sig, w1, b1, w2, b2, w_lin = _fit_mlp(X, y, snap_id)
    np.savez(
        WEIGHTS_NPZ,
        mu=mu, sig=sig, w1=w1, b1=b1, w2=w2, b2=b2, w_lin=w_lin, mlp_mix=np.array([0.25], dtype=np.float32),
    )
    train_summary = {
        "n_rows": int(len(X)),
        "n_snaps": n_snaps,
        "n_train_eps": n_eps,
        "cv_5fold_concordance": round(float(np.mean(concs)), 4),
        "cv_5fold_top1_unlock_acc": round(float(np.mean(top1s)), 4),
    }
    print(f"[V20-C] TRAIN 5-fold GroupKFold: conc={train_summary['cv_5fold_concordance']} top1_acc={train_summary['cv_5fold_top1_unlock_acc']}", flush=True)
    return train_summary


def run_rq1_rq2_and_gate_c(*, gate_c_only: bool = False) -> dict[str, Any]:
    """Run RQ1 Equal-Size Ablation, RQ2 Release Pareto Curve, and Gate C Verdict on VAL-72 & VAL-Static-35."""
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    specs72 = build_val72_specs()
    specs35 = build_val_static35_specs()
    assert_stage1_quarantine([sp["seed"] for sp in specs72 + specs35], label="V20C_VAL_Eval")

    base_v20b_kw: dict[str, Any] = dict(
        soft_deadline_ms=15.0,
        enable_e4=True,
        e4_selector="fast_surr",
        e4_include_zc=False,
        e4_mode="ws_or_invalid",
        staged_mode="ws_or_invalid",
        enable_bidir_v18_to_v19=True,
        enable_e6_static_or_headroom=False,
        enable_e7=True,
        enable_e9=False,
        enable_early_stop=False,
        compact_calm_ms=None,
        enable_r2_global_ws=False,
        r2_zc_ws_mode="never",
        include_staged_g="on_invalid",
        unwin_seq_horizon=2,
        seq_diff_b2=True,
        critic_weights_path=CRITIC_NPZ,
        critic_beta=0.25,
        critic_mode="ws_only",
    )

    rq1_strategies = ("learned_dep_guided", "det_causal_cone", "transport_local", "critical_path_only", "random_subset")
    rq1_summary: dict[str, Any] = {}
    rq1_raw: dict[str, list[dict[str, Any]]] = {}
    rq1_wilcoxon: dict[str, float] = {}
    rq2_points: dict[str, Any] = {}

    if not gate_c_only:
        print("=== [V20-C / RQ1] Equal-Size |U_t| <= k (k=4) & Equal-Budget 35ms Controlled Ablation on VAL-72 ===", flush=True)
        for strat in rq1_strategies:
            prop = V20CNeighborhoodProposer(
                hard_budget_ms=35.0,
                seed=0,
                neighborhood_weights_path=WEIGHTS_NPZ if strat == "learned_dep_guided" else None,
                neighborhood_strategy=strat,
                release_k_max=4,
                strict_fixed_release=True,
                **base_v20b_kw,
            )
            rows = [
                run_wrapped_episode(
                    variant_id=f"rq1_{strat}", spec=sp, base_env_policy="V19_ADAPTIVE_RL", external_proposer=prop, hard_budget_ms=35.0
                )[0]
                for sp in specs72
            ]
            rq1_raw[strat] = rows
            ov = summarize_episode_rows(rows)
            rq1_summary[strat] = ov
            print(
                f"  RQ1 k=4 {strat:20s}: net={ov['mean_net_obj']:7.2f} comp={ov['comp100_count']}/72 "
                f"churn={ov['mean_churn']:5.2f} p95={ov['pooled_p95_ms']:6.3f}ms",
                flush=True,
            )

        learned_vals = np.array([float(r["net_obj"]) for r in rq1_raw["learned_dep_guided"]])
        det_vals = np.array([float(r["net_obj"]) for r in rq1_raw["det_causal_cone"]])
        for strat in rq1_strategies[1:]:
            b_vals = np.array([float(r["net_obj"]) for r in rq1_raw[strat]])
            diff = learned_vals - b_vals
            p_val = float(wilcoxon(diff, zero_method="wilcox", alternative="less").pvalue) if np.any(np.abs(diff) > 1e-9) else 1.0
            rq1_wilcoxon[f"learned_lt_{strat}_p"] = round(p_val, 6)
            if strat != "det_causal_cone":
                d2 = det_vals - b_vals
                p2 = float(wilcoxon(d2, zero_method="wilcox", alternative="less").pvalue) if np.any(np.abs(d2) > 1e-9) else 1.0
                rq1_wilcoxon[f"det_cone_lt_{strat}_p"] = round(p2, 6)
        print("  RQ1 Wilcoxon vs baselines:", rq1_wilcoxon, flush=True)

        print("=== [V20-C / RQ2] Release Fraction Pareto Curve on VAL-72 ===", flush=True)
        for label, strat, frac in [
            ("zero_release", "zero_release", 0.0),
            ("rule_frozen", "rule_frozen", None),
            ("selective_25pct", "learned_dep_guided", 0.25),
            ("selective_50pct", "learned_dep_guided", 0.50),
            ("selective_75pct", "learned_dep_guided", 0.75),
            ("soft_all_100pct", "soft_all", 1.0),
            ("global_unanchored", "global_unanchored", 1.0),
        ]:
            prop = V20CNeighborhoodProposer(
                hard_budget_ms=35.0,
                seed=0,
                neighborhood_weights_path=WEIGHTS_NPZ if strat == "learned_dep_guided" else None,
                neighborhood_strategy=strat,
                release_frac=frac,
                strict_fixed_release=True,
                **base_v20b_kw,
            )
            rows = [
                run_wrapped_episode(
                    variant_id=f"rq2_{label}", spec=sp, base_env_policy="V19_ADAPTIVE_RL", external_proposer=prop, hard_budget_ms=35.0
                )[0]
                for sp in specs72
            ]
            ov = summarize_episode_rows(rows)
            rq2_points[label] = ov
            print(
                f"  RQ2 {label:20s}: net={ov['mean_net_obj']:7.2f} comp={ov['comp100_count']}/72 churn={ov['mean_churn']:5.2f} p95={ov['pooled_p95_ms']:6.3f}ms",
                flush=True,
            )

    print("=== [V20-C / Gate C] V20-B Baseline vs V20-B + Learned N_k (VAL-72 & VAL-Static-35) ===", flush=True)
    prop_b = V20BProposer(hard_budget_ms=35.0, seed=0, **base_v20b_kw)
    prop_c = V20CNeighborhoodProposer(
        hard_budget_ms=35.0,
        seed=0,
        neighborhood_weights_path=WEIGHTS_NPZ,
        neighborhood_strategy="learned_dep_guided",
        release_frac=0.45,
        strict_fixed_release=False,
        **base_v20b_kw,
    )
    rows_b72 = [run_wrapped_episode(variant_id="gatec_v20b", spec=sp, base_env_policy="V19_ADAPTIVE_RL", external_proposer=prop_b, hard_budget_ms=35.0)[0] for sp in specs72]
    rows_c72 = [run_wrapped_episode(variant_id="gatec_v20c", spec=sp, base_env_policy="V19_ADAPTIVE_RL", external_proposer=prop_c, hard_budget_ms=35.0)[0] for sp in specs72]
    ov_b = summarize_episode_rows(rows_b72)
    ov_c = summarize_episode_rows(rows_c72)
    diff_cb = np.array([float(c["net_obj"]) - float(b["net_obj"]) for c, b in zip(rows_c72, rows_b72)])
    p_cb = float(wilcoxon(diff_cb, zero_method="wilcox", alternative="less").pvalue) if np.any(np.abs(diff_cb) > 1e-9) else 1.0

    rows_c35 = [run_wrapped_episode(variant_id="gatec_v20c_s35", spec=sp, base_env_policy="V19_ADAPTIVE_RL", external_proposer=prop_c, hard_budget_ms=35.0)[0] for sp in specs35]
    ov_c35 = summarize_episode_rows(rows_c35)

    learned_beats_det = bool(
        ov_c["comp100_count"] == 72
        and ov_c["pooled_p95_ms"] <= 28.0
        and ov_c35["comp100_count"] == 35
        and ov_c["mean_net_obj"] < ov_b["mean_net_obj"]
        and p_cb < 0.05
    )
    print(
        f"  V20-B VAL-72={ov_b['mean_net_obj']} ({ov_b['comp100_count']}/72, p95={ov_b['pooled_p95_ms']}ms)\n"
        f"  V20-C VAL-72={ov_c['mean_net_obj']} ({ov_c['comp100_count']}/72, p95={ov_c['pooled_p95_ms']}ms) | "
        f"Wilcoxon p_less={p_cb:.6f} | Static-35 H1={ov_c35['mean_h1_all']} ({ov_c35['comp100_count']}/35) | "
        f"GateC_Learned_Promoted={learned_beats_det}",
        flush=True,
    )
    out = {
        "schema": "v20_c.rq1_rq2_gate_eval.v1",
        "rq1_equal_size_k4": rq1_summary,
        "rq1_wilcoxon_p": rq1_wilcoxon,
        "rq2_release_pareto": rq2_points,
        "gate_c": {
            "v20b_val72": ov_b,
            "v20c_val72": ov_c,
            "v20c_by_suite": {
                g: summarize_episode_rows([r for r in rows_c72 if r["suite_group"] == g])
                for g in ("4topo", "track1", "track2")
            },
            "v20c_static35": ov_c35,
            "wilcoxon_v20c_lt_v20b_p": round(p_cb, 6),
            "learned_nk_promoted": learned_beats_det,
        },
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--gate-c-only", action="store_true")
    args = p.parse_args()

    train_info = None
    eval_info = None
    if not args.eval_only and not args.gate_c_only:
        train_info = train_neighborhood_head()
    if not args.train_only:
        eval_info = run_rq1_rq2_and_gate_c(gate_c_only=args.gate_c_only)

    prev = {}
    rep_path = OUT_AUDIT_DIR / "v20_c_gate_eval.json"
    if rep_path.exists():
        try:
            prev = json.loads(rep_path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    if train_info is not None:
        prev["train_summary"] = train_info
    if eval_info is not None:
        if args.gate_c_only:
            prev["gate_c"] = eval_info["gate_c"]
        else:
            prev.update(eval_info)
    rep_path.write_text(json.dumps(prev, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
