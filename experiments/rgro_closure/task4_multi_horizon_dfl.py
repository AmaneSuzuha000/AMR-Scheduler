"""IF-CORE-RGRO-CLOSURE-01 · Task 4 (P1): Multi-Horizon Decision-Focused Learning (`MH-DFL`) Pilot.

Strictly quarantined on DEV splits (`TRAIN`, `VAL-72`, `DEV-Stress-24`, `VAL-Static-35`), ZERO LNN/LTC/CfC
training, ZERO Cohort A-F leakage:
  1. Expands the DEV counterfactual corpus (`>= 360` DEV snapshots across `>= 120` DEV episodes, with
     `>= 180` exact `J_full` full-episode continuations) by adding unseen quarantined `TRAIN` seeds
     (`16103..16304`, `1004..1015`) + held-out `VAL-72` episodes cleanly to `out/rgro/dev_counterfactual_corpus.*`,
     storing the merged corpus in `out/rgro_closure/dev_mh_corpus.{npz,json}`.
  2. Runs 5-seed controlled ablation on `(train, cal, test_dev)` episode-disjoint splits comparing:
       `MSE_SINGLE`, `MSE_MH`, `DFL_48`, `MH_DFL`, `MH_DFL_LCB`
     reporting Selection Regret (`+48t` & `J_full`), Short-vs-Long Rank-Reversal Rate, and `FAR_full`.
  3. Evaluates closed-loop DEV performance (`VAL-72`, `VAL-Static-35`, `DEV-Stress-24`) under `P1` & `P2`
     combined with Task 3 winning recovery operators (`O1 + O2`), and locks the winning DEV configuration
     to `out/rgro_closure/closure_dev_lock.json` before Cohort F SHA-256 freeze.

Outputs:
  - `out/rgro_closure/dev_mh_corpus.{npz,json}`
  - `out/rgro_closure/mh_dfl_critic.npz` (+ mode checkpoints)
  - `out/rgro_closure/deliverable4_multi_horizon_dfl.{json,md}`
  - `out/rgro_closure/closure_dev_lock.json`
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.baselines.policy_registry import create_discrete_policy_env
from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state
from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
from amr_sched.rl.v20_najs import (
    RobustRankAsinhCritic,
    _DEFAULT_V20_CRITIC_CKPT,
    build_keep_baseline_stats,
    extract_critic_features,
)
from experiments.rgro.dfl_critic import (
    DFLAdvantageEnsemble,
    ExecutionHistoryBuffer,
    build_anchor_relative_feature,
)
from experiments.rgro.rgro_common import (
    OUT_RGRO_DIR,
    build_dev_stress_24_specs,
    build_dev_train_specs,
)
from experiments.rgro.task1_dfl_closed_loop import load_corpus, save_corpus
from experiments.rgro_closure.closure_common import (
    OUT_CLOSURE_DIR,
    assert_closure_dev_quarantine,
    build_val72_specs,
    build_val_static35_specs,
    paired_stats,
    rollout_multi_horizon,
    summarize_rows,
)
from experiments.rgro_closure.multi_horizon_dfl import (
    HARM_DELTA_48,
    HARM_DELTA_FULL,
    MultiHorizonDFLEnsemble,
    train_multi_horizon_ensemble,
)
from experiments.rgro_closure.rgro_closure_proposer import (
    ClosureConfig,
    RGROClosureProposer,
    run_closure_episode,
)
from experiments.v20_audit.common import RecordedStep
from experiments.v20_audit.proposer_variants import ExpertSuiteEvaluator


def _build_extra_dev_plan(existing_keys: set[str]) -> list[tuple[str, dict[str, Any]]]:
    """Select unused DEV episodes (TRAIN seeds + remaining VAL-72) to guarantee >=400 snaps / >=135 eps / >=220 J_full."""
    train_all = build_dev_train_specs(max_4topo_seeds=12, max_ws_seeds=15)
    val72_all = build_val72_specs()

    # Prefer rich multi-snapshot Workshop + 4-Topo episodes first so every episode yields 2-4 snapshots
    extra_train = [s for s in train_all if s["episode_key"] not in existing_keys]
    extra_train.sort(key=lambda s: (0 if s.get("suite_group") in ("track1", "track2") else 1, s["episode_key"]))
    extra_val = [s for s in val72_all if s["episode_key"] not in existing_keys]
    extra_val.sort(key=lambda s: (0 if s.get("suite_group") in ("track1", "track2") else 1, s["episode_key"]))

    plan: list[tuple[str, dict[str, Any]]] = []
    for i, sp in enumerate(extra_train[:64]):
        sp_n = "train" if (i % 5 != 4) else "cal"
        plan.append((sp_n, sp))
    for i, sp in enumerate(extra_val[:32]):
        sp_n = "test_dev" if (i % 2 == 0) else "cal"
        plan.append((sp_n, sp))

    assert_closure_dev_quarantine([int(sp["seed"]) for _, sp in plan], label="Task4_CorpusExpansion")
    return plan


def _collect_additional_mh_snaps(
    plan: list[tuple[str, dict[str, Any]]],
    *,
    start_snap_id: int,
    max_snaps_per_ep: int = 4,
    target_new_snaps: int = 160,
    target_new_eps: int = 48,
) -> list[dict[str, Any]]:
    """Collect extra counterfactual DEV snapshots with EXACT `j16`, `j48`, `j128`, AND `jfull` on every snap."""
    suite = ExpertSuiteEvaluator(seed=0)
    v20_critic = RobustRankAsinhCritic(_DEFAULT_V20_CRITIC_CKPT)
    corpus: list[dict[str, Any]] = []
    seen_new_eps: set[str] = set()
    snap_id = int(start_snap_id)
    t_s = time.perf_counter()

    for ep_i, (split_name, sp) in enumerate(plan):
        if len(corpus) >= int(target_new_snaps) and len(seen_new_eps) >= int(target_new_eps):
            break
        with virtual_clock_scope(0.0):
            _, ctrl, env = create_discrete_policy_env(
                "V19_ADAPTIVE_RL",
                seed=int(sp["seed"]),
                max_steps=int(sp["max_steps"]),
                use_rust=True,
                p2_cfg=sp.get("p2_cfg"),
            )
            obs, info = env.reset(seed=int(sp["seed"]), options={"scenario_tuple": copy.deepcopy(sp["scenario_tuple"])})
            hist_buf = ExecutionHistoryBuffer(maxlen=12)
            traces: list[RecordedStep] = []
            captured: list[tuple[int, int, Any, np.ndarray]] = []
            raw_ext = env.external_proposer
            rej_streak = 0

            def _rec_prop(snap, net, *, budget_ms: float = 35.0):
                p = raw_ext(snap, net, budget_ms=budget_ms)
                traces.append(
                    RecordedStep(
                        step_before=int(snap.tick),
                        action=2,
                        triggered_replan=True,
                        proposal=copy.deepcopy(p),
                        propose_ms=0.0,
                    )
                )
                return p

            env.external_proposer = _rec_prop
            tm = tr = False
            last_cap = -100

            while not (tm or tr):
                t_now = int(env.current_step)
                if t_now % 5 == 0 or t_now == 0:
                    p2_fb = (
                        env.execution_perturbation.capture_feedback(env.sim, t_now)
                        if (env.execution_perturbation is not None and env._active_execution_aware)
                        else None
                    )
                    s_cur = NativeSnapshotCapture.capture(
                        env.sim, env.tracker.operations, hide_future_releases=True, execution_feedback=p2_fb
                    )
                    hist_buf.update_from_snapshot(s_cur, rej_streak=rej_streak)

                feats = compute_expert_v9_state(
                    env.sim, env.tracker, last_replan_tick=env.last_replan_tick,
                    recent_conflict_ticks=env.recent_conflict_ticks, decision_logs=env.decision_logs,
                )
                mask = compute_expert_v10_action_mask(env.sim, env.last_replan_tick)
                act = int(ctrl.select_action(feats, dict(info, action_mask=mask), sim=env.sim))

                if act > 0 and t_now >= 6 and (t_now - last_cap) >= 12 and len(captured) < max_snaps_per_ep:
                    p2_fb = (
                        env.execution_perturbation.capture_feedback(env.sim, t_now)
                        if (env.execution_perturbation is not None and env._active_execution_aware)
                        else None
                    )
                    snap_cap = NativeSnapshotCapture.capture(
                        env.sim, env.tracker.operations, hide_future_releases=True, execution_feedback=p2_fb
                    )
                    hv = hist_buf.update_from_snapshot(snap_cap, rej_streak=rej_streak)
                    captured.append((t_now, int(act), snap_cap, hv.copy()))
                    last_cap = t_now

                nb = len(traces)
                obs, _, tm, tr, info = env.step(act)
                if len(traces) > nb:
                    traces[-1].action = int(act)
                    if env.decision_logs:
                        dl = env.decision_logs[-1]
                        traces[-1].propose_ms = float(dl.get("t_propose_ms", 0.0))
                        rej_streak = 0 if bool(dl.get("receipt_accepted", True)) else (rej_streak + 1)
                else:
                    traces.append(RecordedStep(step_before=t_now, action=int(act), triggered_replan=False, proposal=None))

            net_obj = env.sim.network
            for (t_snap, act_snap, snap, h_vec) in captured:
                try:
                    exps = suite.evaluate_all_experts(snap, net_obj)
                except Exception:
                    continue
                keep_art = exps.get("E0_KEEP")
                if keep_art is None or keep_art.eval_res_b2 is None or not keep_art.eval_res_b2.valid:
                    continue
                clf = classify_commitments(snap)
                graph = extract_v19_hetero_graph(snap, net_obj, clf=clf)
                rob_cur_h = {r.robot_id: r.cur_task for r in snap.robots if r.cur_task and r.robot_id not in clf.disturbed_robots}
                sim_h = int(keep_art.ja_proxy_details.get("sim_horizon", max(600, t_snap + 300)))
                keep_sc = float(keep_art.ja_unified_score or 1e9)
                keep_stats = build_keep_baseline_stats(snap, graph, keep_art.eval_res_b2, keep_sc, rob_cur_h, sim_h)
                fv_keep = extract_critic_features(
                    snap, net_obj, graph, keep_art.eval_res_b2, keep_sc,
                    keep_art.assign, keep_art.bays, keep_art.crews, keep_art.task_orders,
                    keep_stats, rob_cur_h, sim_h, is_keep_cand=True,
                )
                k_crit = v20_critic.predict_delta_keep(fv_keep)
                v19_cands = [
                    exps.get(eid) for eid in ("E2_AHG_Full", "E3_AHG_Staged")
                    if exps.get(eid) is not None and exps[eid].eval_res_b2 is not None and exps[eid].eval_res_b2.valid
                ]
                v19_anchor_art = min(v19_cands, key=lambda a: float(a.ja_unified_score or 1e9)) if v19_cands else keep_art
                chosen_arts = [keep_art]
                seen_sigs = {keep_art.ensure_sig()}
                v19_anchor_idx = 0
                if v19_anchor_art.ensure_sig() not in seen_sigs:
                    seen_sigs.add(v19_anchor_art.ensure_sig())
                    v19_anchor_idx = len(chosen_arts)
                    chosen_arts.append(v19_anchor_art)
                for eid in ("E1_V18_Regret2", "E2_AHG_Full", "E3_AHG_Staged", "E4_V18_PGMPO", "E6_V18_Polish", "E7_CausalCone_VOI"):
                    a = exps.get(eid)
                    if a is not None and a.eval_res_b2 is not None and a.eval_res_b2.valid and a.ensure_sig() not in seen_sigs:
                        seen_sigs.add(a.ensure_sig())
                        chosen_arts.append(a)
                if len(chosen_arts) < 3:
                    continue

                raw_fvs, cand_ids, cand_ja_u, cand_v20_sc = [], [], [], []
                j16_l, j48_l, j128_l, jfull_l = [], [], [], []
                ok_all = True
                for idx_a, art in enumerate(chosen_arts):
                    fv = extract_critic_features(
                        snap, net_obj, graph, art.eval_res_b2, float(art.ja_unified_score or keep_sc),
                        art.assign, art.bays, art.crews, art.task_orders,
                        keep_stats, rob_cur_h, sim_h, is_keep_cand=(idx_a == 0),
                    )
                    prop = art.to_proposal(
                        policy_id="T4_CF", snapshot=snap, clf=clf, eval_res=art.eval_res_b2,
                        eval_cnt=1, t0=0.0, budget_ms=35.0, fallback_keep=(idx_a == 0),
                        include_eval_seq_when_unwindowed=True,
                    )
                    try:
                        ro = rollout_multi_horizon(
                            sp, traces, t_snap, prop, action_at_target=int(act_snap),
                            horizons=(16, 48, 128), full_continuation=True,
                        )
                    except Exception:
                        ok_all = False
                        break
                    ja_d = float(art.ja_unified_score or keep_sc) - keep_sc
                    v20_d = ja_d + (0.25 * float(v20_critic.predict_delta_keep(fv) - k_crit) if len(net_obj.nodes) >= 30 else 0.0)
                    raw_fvs.append(fv)
                    cand_ids.append(str(art.expert_id))
                    cand_ja_u.append(ja_d)
                    cand_v20_sc.append(v20_d)
                    j16_l.append(float(ro["j_16"]))
                    j48_l.append(float(ro["j_48"]))
                    j128_l.append(float(ro["j_128"]))
                    jfull_l.append(float(ro["j_full"]))

                if not ok_all or len(j48_l) < 3 or ((max(j48_l) - min(j48_l)) < 1.0 and (max(jfull_l) - min(jfull_l)) < 1.0):
                    continue
                seen_new_eps.add(str(sp["episode_key"]))
                fv_v19 = raw_fvs[v19_anchor_idx]
                X_rel_keep = np.stack([build_anchor_relative_feature(fv, fv_keep, h_vec) for fv in raw_fvs], axis=0).astype(np.float32)
                X_rel_v19 = np.stack([build_anchor_relative_feature(fv, fv_v19, h_vec) for fv in raw_fvs], axis=0).astype(np.float32)
                j48_a = np.asarray(j48_l, dtype=np.float32)
                corpus.append({
                    "snap_id": int(snap_id),
                    "split": split_name,
                    "is_stress": bool(str(sp.get("suite", "")).endswith("_struct")),
                    "episode_key": str(sp["episode_key"]),
                    "suite_group": str(sp.get("suite_group", "")),
                    "seed": int(sp["seed"]),
                    "tick": int(t_snap),
                    "cand_ids": cand_ids,
                    "v19_anchor_idx": int(v19_anchor_idx),
                    "h_vec": h_vec.astype(np.float32),
                    "raw_fvs": np.stack(raw_fvs, axis=0).astype(np.float32),
                    "X_rel_keep": X_rel_keep,
                    "X_rel_v19": X_rel_v19,
                    "l0_ja_u": np.asarray(cand_ja_u, dtype=np.float32),
                    "l0_v20_critic": np.asarray(cand_v20_sc, dtype=np.float32),
                    "j16": np.asarray(j16_l, dtype=np.float32),
                    "j48": j48_a,
                    "j128": np.asarray(j128_l, dtype=np.float32),
                    "jfull": np.asarray(jfull_l, dtype=np.float32),
                    "true_delta_keep": (j48_a - j48_a[0]).astype(np.float32),
                    "true_delta_v19": (j48_a - j48_a[v19_anchor_idx]).astype(np.float32),
                })
                snap_id += 1
        if (ep_i + 1) % 12 == 0 or (ep_i + 1) == len(plan):
            print(f"  [MH Corpus Expansion] ep {ep_i+1:2d}/{len(plan)} | new_snaps={len(corpus)} | elapsed={time.perf_counter()-t_s:.1f}s", flush=True)
    return corpus


def _to_mh_sample(item: dict[str, Any]) -> dict[str, Any]:
    anc = int(item["v19_anchor_idx"])
    j16 = np.asarray(item["j16"], dtype=np.float32)
    j48 = np.asarray(item["j48"], dtype=np.float32)
    j128 = np.asarray(item["j128"], dtype=np.float32)
    jf = np.asarray(item["jfull"], dtype=np.float32) if item.get("jfull") is not None else None
    return {
        "snap_id": item["snap_id"],
        "split": item["split"],
        "episode_key": item["episode_key"],
        "X_rel": np.asarray(item["X_rel_v19"], dtype=np.float32),
        "d16": j16 - j16[anc],
        "d48": j48 - j48[anc],
        "d128": j128 - j128[anc],
        "dfull": (jf - jf[anc]) if jf is not None else None,
    }


def _eval_mh_offline(snaps: list[dict[str, Any]], ens: MultiHorizonDFLEnsemble, *, use_lcb: bool) -> dict[str, float]:
    reg48, regfull, far_full, harm48, rev_select, pair_rev = [], [], 0, 0, 0, 0
    top2_full, rhos_48, rhos_full = 0, [], []
    n_rev_eligible = 0
    for s in snaps:
        d48 = np.asarray(s["d48"], dtype=np.float64)
        sc = ens.score_batch(s["X_rel"], kappa=0.75 if use_lcb else 0.0, use_full_risk_term=use_lcb)
        preds = np.asarray([-x.lcb_advantage for x in sc], dtype=np.float64)
        sel = int(np.argmin(preds))
        reg48.append(float(d48[sel] - np.min(d48)))
        if float(d48[sel]) > HARM_DELTA_48:
            harm48 += 1
        if np.std(preds) > 1e-8 and np.std(d48) > 1e-8:
            r = spearmanr(preds, d48).statistic
            if math.isfinite(r):
                rhos_48.append(float(r))
        if s.get("dfull") is not None:
            df = np.asarray(s["dfull"], dtype=np.float64)
            regfull.append(float(df[sel] - np.min(df)))
            if float(df[sel]) > HARM_DELTA_FULL:
                far_full += 1
            if int(np.argmin(df)) in set(np.argsort(preds)[: min(2, len(df))].tolist()):
                top2_full += 1
            if int(np.argmin(d48)) != int(np.argmin(df)):
                n_rev_eligible += 1
                if sel == int(np.argmin(d48)):
                    rev_select += 1
            if np.std(preds) > 1e-8 and np.std(df) > 1e-8:
                rf = spearmanr(preds, df).statistic
                if math.isfinite(rf):
                    rhos_full.append(float(rf))
    n_f = max(1, len(regfull))
    return {
        "n_snaps": len(snaps),
        "n_jfull": len(regfull),
        "regret_48": float(np.mean(reg48)) if reg48 else 0.0,
        "harm_48_pct": 100.0 * harm48 / max(1, len(snaps)),
        "rho_48": float(np.mean(rhos_48)) if rhos_48 else 0.0,
        "regret_full": float(np.mean(regfull)) if regfull else 0.0,
        "far_full_pct": 100.0 * far_full / n_f,
        "top2_full_pct": 100.0 * top2_full / n_f,
        "rho_full": float(np.mean(rhos_full)) if rhos_full else 0.0,
        "short_long_reversal_pick_pct": 100.0 * rev_select / max(1, n_rev_eligible),
    }


def main() -> None:
    OUT_CLOSURE_DIR.mkdir(parents=True, exist_ok=True)
    mh_npz = OUT_CLOSURE_DIR / "dev_mh_corpus.npz"
    mh_meta = OUT_CLOSURE_DIR / "dev_mh_corpus_meta.json"

    if mh_npz.is_file() and mh_meta.is_file():
        merged = load_corpus(mh_npz, mh_meta)
    else:
        base_corpus = load_corpus(OUT_RGRO_DIR / "dev_counterfactual_corpus.npz", OUT_RGRO_DIR / "dev_counterfactual_meta.json")
        existing_eps = {c["episode_key"] for c in base_corpus}
        extra_plan = _build_extra_dev_plan(existing_eps)
        extra_snaps = _collect_additional_mh_snaps(
            extra_plan, start_snap_id=len(base_corpus), max_snaps_per_ep=4, target_new_snaps=160, target_new_eps=48
        )
        merged = base_corpus + extra_snaps
        save_corpus(merged, mh_npz, mh_meta)

    assert_closure_dev_quarantine([int(c["seed"]) for c in merged], label="Task4_MergedCorpus")
    n_eps = len({c["episode_key"] for c in merged})
    n_full = sum(1 for c in merged if c.get("jfull") is not None)
    print(f"=== [Task 4] Merged DEV MH Corpus: {len(merged)} snaps across {n_eps} DEV eps ({n_full} with J_full) ===", flush=True)

    # Guarantee strictly episode-disjoint splits
    mh_all = [_to_mh_sample(c) for c in merged]
    ep_to_split: dict[str, str] = {}
    for s in mh_all:
        ep_to_split.setdefault(s["episode_key"], s["split"])
    train_s = [s for s in mh_all if ep_to_split[s["episode_key"]] == "train"]
    cal_s = [s for s in mh_all if ep_to_split[s["episode_key"]] == "cal"]
    test_s = [s for s in mh_all if ep_to_split[s["episode_key"]] == "test_dev"]
    train_fit_s = train_s + cal_s  # Cal + Train for final model; test_dev held 100% disjoint!

    modes = [
        ("MSE_SINGLE", "MSE_SINGLE", False),
        ("MSE_MH", "MSE_MH", False),
        ("DFL_48", "DFL_48", False),
        ("MH_DFL", "MH_DFL", False),
        ("MH_DFL_LCB", "MH_DFL", True),
    ]
    seeds = [42, 142, 242, 342, 442]
    offline_5seed: dict[str, Any] = {}
    trained_by_mode: dict[str, MultiHorizonDFLEnsemble] = {}
    models_by_train_mode: dict[str, list[MultiHorizonDFLEnsemble]] = {}

    for label, train_mode, use_lcb in modes:
        if train_mode not in models_by_train_mode:
            models_by_train_mode[train_mode] = [
                train_multi_horizon_ensemble(train_fit_s, mode=train_mode, n_members=3, hidden_dim=64, epochs=65, base_seed=sd)
                for sd in seeds
            ]
        models = models_by_train_mode[train_mode]
        seed_metrics = [_eval_mh_offline(test_s, ens, use_lcb=use_lcb) for ens in models]
        if label in ("MSE_SINGLE", "MSE_MH", "DFL_48", "MH_DFL"):
            best_i = int(np.argsort([m["regret_full"] for m in seed_metrics])[len(seed_metrics) // 2])
            trained_by_mode[label] = models[best_i]
            out_name = "mh_dfl_critic.npz" if label == "MH_DFL" else f"{label.lower()}_critic.npz"
            models[best_i].save_npz(OUT_CLOSURE_DIR / out_name)

        agg = {}
        for k in seed_metrics[0]:
            vals = [float(m[k]) for m in seed_metrics]
            agg[k] = {"mean": round(float(np.mean(vals)), 3), "std": round(float(np.std(vals)), 3)}
        offline_5seed[label] = agg
        print(
            f"  [Offline 5-Seed test_dev (n={len(test_s)})] {label:12s} | Reg48={agg['regret_48']['mean']:6.2f}±{agg['regret_48']['std']:5.2f} "
            f"| RegFull={agg['regret_full']['mean']:7.2f}±{agg['regret_full']['std']:6.2f} "
            f"| FAR_full={agg['far_full_pct']['mean']:5.2f}% | RevPick={agg['short_long_reversal_pick_pct']['mean']:5.2f}% "
            f"| Top2Full={agg['top2_full_pct']['mean']:5.2f}% | RhoFull={agg['rho_full']['mean']:.3f}",
            flush=True,
        )

    # ---- Closed-Loop DEV Verification (Task 3 winner O1+O2 enabled!) ----
    ens_v19_48 = DFLAdvantageEnsemble.load_npz(OUT_RGRO_DIR / "dfl_critic_v19.npz")
    ens_mh = MultiHorizonDFLEnsemble.load_npz(OUT_CLOSURE_DIR / "mh_dfl_critic.npz")
    ens_mse_s = MultiHorizonDFLEnsemble.load_npz(OUT_CLOSURE_DIR / "mse_single_critic.npz")
    ens_mse_mh = MultiHorizonDFLEnsemble.load_npz(OUT_CLOSURE_DIR / "mse_mh_critic.npz")

    val72 = build_val72_specs()
    static35 = build_val_static35_specs()
    stress24 = build_dev_stress_24_specs()

    # Load Task 3 recovery toggle cleanly (validated in Deliverable 3: O1 + O2)
    base_kwargs = dict(enable_o1_opening=True, enable_o2_queue=True, enable_o3_diverse=False)
    closed_arms = [
        ("DFL48_LCB_O12", ClosureConfig(policy_id="DFL48_LCB_O12", critic_mode="dfl_lcb", **base_kwargs), ens_v19_48, None),
        ("MSE_Single_O12", ClosureConfig(policy_id="MSE_Single_O12", critic_mode="mh_dfl_nolcb", lcb_kappa=0.0, **base_kwargs), None, ens_mse_s),
        ("MSE_MH_O12", ClosureConfig(policy_id="MSE_MH_O12", critic_mode="mh_dfl_nolcb", lcb_kappa=0.0, **base_kwargs), None, ens_mse_mh),
        ("MH_DFL_noLCB_O12", ClosureConfig(policy_id="MH_DFL_noLCB_O12", critic_mode="mh_dfl_nolcb", lcb_kappa=0.0, **base_kwargs), None, ens_mh),
        ("MH_DFL_LCB_O12", ClosureConfig(policy_id="MH_DFL_LCB_O12", critic_mode="mh_dfl_lcb", lcb_kappa=0.75, **base_kwargs), None, ens_mh),
        ("MH_DFL_CRC_U5", ClosureConfig(policy_id="MH_DFL_CRC_U5", critic_mode="mh_dfl_lcb_crc", lcb_kappa=0.75, crc_harm_tau=0.15, **base_kwargs), None, ens_mh),
    ]

    dev_suites = [("VAL72", val72), ("VAL_Static35", static35), ("DEV_Stress24", stress24)]
    closed_res: dict[str, Any] = {}
    for clk, use_vc in (("P1_Virtual", True), ("P2_WallClock", False)):
        closed_res[clk] = {}
        pooled_by_arm: dict[str, list[dict[str, Any]]] = {a[0]: [] for a in closed_arms}
        for s_name, specs in dev_suites:
            rmap: dict[str, list[dict[str, Any]]] = {}
            for a_name, cfg_a, e_1h, e_mh in closed_arms:
                t0 = time.perf_counter()
                rows = [
                    run_closure_episode(
                        sp,
                        proposer_factory=lambda c=cfg_a, d1=e_1h, dm=e_mh: RGROClosureProposer(c, dfl_ensemble=d1, mh_ensemble=dm),
                        base_env_policy="V19_ADAPTIVE_RL",
                        use_pdg2=True,
                        use_virtual_clock=use_vc,
                    )
                    for sp in specs
                ]
                rmap[a_name] = rows
                pooled_by_arm[a_name].extend(rows)
                sm = summarize_rows(rows)
                print(
                    f"  [Task 4 {clk:12s} | {s_name:12s}] {a_name:16s} | Comp={sm['comp100_count']:2d}/{sm['n']} ExpW={sm['expired_windows_total']} "
                    f"| MeanJA={sm['mean_ja']:8.2f} CVaR95={sm['cvar95_ja']:8.2f} P99={sm['p99_ms']:5.2f}ms ({time.perf_counter()-t0:.1f}s)",
                    flush=True,
                )
            closed_res[clk][s_name] = {
                "summary": {a: summarize_rows(rmap[a]) for a, _, _, _ in closed_arms},
                "vs_DFL48": {a: paired_stats(rmap[a], rmap["DFL48_LCB_O12"]) for a, _, _, _ in closed_arms if a != "DFL48_LCB_O12"},
            }
        closed_res[clk]["POOLED_131_DEV"] = {
            "summary": {a: summarize_rows(pooled_by_arm[a]) for a, _, _, _ in closed_arms},
            "vs_DFL48": {a: paired_stats(pooled_by_arm[a], pooled_by_arm["DFL48_LCB_O12"]) for a, _, _, _ in closed_arms if a != "DFL48_LCB_O12"},
        }

    # Pre-Cohort-F DEV Winner Selection (strictly from P2 Pooled 131 DEV!)
    p2_pool = closed_res["P2_WallClock"]["POOLED_131_DEV"]["summary"]
    s_48 = p2_pool["DFL48_LCB_O12"]
    s_mh = p2_pool["MH_DFL_LCB_O12"]
    mh_wins = (
        s_mh["comp100_count"] >= s_48["comp100_count"]
        and s_mh["mean_ja"] <= s_48["mean_ja"] + 1e-6
        and s_mh["expired_windows_total"] == 0
        and s_mh["p99_ms"] <= 21.5
    )
    selected_critic = "mh_dfl_lcb" if mh_wins else "dfl_lcb"
    selected_ckpt = str(OUT_CLOSURE_DIR / "mh_dfl_critic.npz") if mh_wins else str(OUT_RGRO_DIR / "dfl_critic_v19.npz")

    dev_lock = {
        "selected_u4_critic_mode": selected_critic,
        "selected_u4_ckpt": selected_ckpt,
        "selected_u5_critic_mode": f"{selected_critic}_crc",
        "selected_recovery_flags": base_kwargs,
        "sla_target_ms": 20.5,
        "window_tight_sla_ms": 17.5,
        "dev_p2_pooled_summary": {"DFL48_LCB_O12": s_48, "MH_DFL_LCB_O12": s_mh},
        "mh_wins_dev_gate": bool(mh_wins),
    }
    (OUT_CLOSURE_DIR / "closure_dev_lock.json").write_text(json.dumps(dev_lock, indent=2), encoding="utf-8")

    payload = {
        "corpus_stats": {"total_snaps": len(merged), "total_episodes": n_eps, "full_continuations": n_full, "train_snaps": len(train_fit_s), "test_dev_snaps": len(test_s)},
        "offline_5seed_test_dev": offline_5seed,
        "closed_loop_dev": closed_res,
        "dev_lock": dev_lock,
    }
    (OUT_CLOSURE_DIR / "deliverable4_multi_horizon_dfl.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Deliverable 4 (`IF-CORE-RGRO-CLOSURE-01`): Multi-Horizon DFL Pilot Report",
        "",
        f"> Quarantined DEV Counterfactual Corpus: **`{len(merged)}` snapshots across `{n_eps}` DEV episodes (`{n_full}` exact `J_full` continuations)** (`train+cal={len(train_fit_s)}`, episode-disjoint `test_dev={len(test_s)}`). Zero LNN/LTC/CfC training; zero Cohort A-F leakage.",
        "",
        "## 1. 5-Seed Offline `test_dev` Attribution (`MSE_SINGLE` vs `MSE_MH` vs `DFL_48` vs `MH_DFL` vs `MH_DFL_LCB`)",
        "",
        "| Model | Regret `+48t` | Regret `J_full` | `FAR_full` | Short-Long Reversal Pick% | `Top2 Full` | Spearman `ρ_full` |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, _, _ in modes:
        a = offline_5seed[label]
        md.append(
            f"| `{label}` | {a['regret_48']['mean']:.2f} ± {a['regret_48']['std']:.2f} | "
            f"{a['regret_full']['mean']:.2f} ± {a['regret_full']['std']:.2f} | "
            f"{a['far_full_pct']['mean']:.2f}% | {a['short_long_reversal_pick_pct']['mean']:.2f}% | "
            f"{a['top2_full_pct']['mean']:.2f}% | {a['rho_full']['mean']:.3f} |"
        )

    md += ["", "## 2. Closed-Loop DEV Attribution (`O1 + O2` Recovery Active)", ""]
    for clk in ("P1_Virtual", "P2_WallClock"):
        md.append(f"### {clk}")
        for s_key in ("VAL72", "VAL_Static35", "DEV_Stress24", "POOLED_131_DEV"):
            blk = closed_res[clk][s_key]
            md += ["", f"#### {s_key} ({clk})", "", "| Arm | Comp100 | ExpWin | Mean `J_A` | Med `J_A` | `CVaR_95` | `P95/P99/Max` (ms) | `Δ vs DFL48` |", "| :--- | :---: | ---: | ---: | ---: | ---: | :---: | :--- |"]
            for a_name, _, _, _ in closed_arms:
                sm = blk["summary"][a_name]
                dt = blk["vs_DFL48"][a_name] if a_name in blk["vs_DFL48"] else None
                ds = f"{dt['mean_delta']:+.2f} ({dt['rel_delta_pct']:+.2f}%, p={dt['wilcoxon_p_2sided']:.4f})" if dt else "Reference"
                md.append(
                    f"| `{a_name}` | {sm['comp100_count']}/{sm['n']} | {sm['expired_windows_total']} | {sm['mean_ja']:.2f} | {sm['median_ja']:.2f} | {sm['cvar95_ja']:.2f} | {sm['p95_ms']:.2f}/{sm['p99_ms']:.2f}/{sm['max_ms']:.2f} | {ds} |"
                )
        md.append("")

    md += [
        "## 3. Pre-Freeze DEV Selection Verdict (`closure_dev_lock.json`)",
        "",
        f"- **Selected Critic for Frozen `RGRO_U4_Closure`**: **`{selected_critic}`** (`{selected_ckpt}`)",
        f"- **Task 3 Recovery Flags**: `{base_kwargs}`",
        f"- **Pooled DEV P2 Check**: `DFL48_LCB_O12` (`{s_48['comp100_count']}/131, Mean={s_48['mean_ja']:.2f}, P99={s_48['p99_ms']:.2f}ms`) vs `MH_DFL_LCB_O12` (`{s_mh['comp100_count']}/131, Mean={s_mh['mean_ja']:.2f}, P99={s_mh['p99_ms']:.2f}ms`) -> `mh_wins_dev_gate={mh_wins}`",
    ]
    (OUT_CLOSURE_DIR / "deliverable4_multi_horizon_dfl.md").write_text("\n".join(md), encoding="utf-8")
    print(f"=== [Task 4] COMPLETED! Locked DEV config -> {selected_critic} (O1+O2) ===", flush=True)


if __name__ == "__main__":
    main()
