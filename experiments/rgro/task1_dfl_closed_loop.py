"""IF-CORE-RGRO-01 · Task 1: Phase 0 Oracle Headroom, Expanded DEV Counterfactual Corpus,
5-Seed DFL vs MSE Training, and Controlled Closed-Loop Causal Attribution (`C0..C3`).

Strictly isolated on quarantined DEV splits (`TRAIN`, `VAL-72`, `DEV-Stress-24`).
ZERO overlap with Cohort A/B/C/D/D-Struct or locked Cohort E/E-Struct/E-Compute.

Outputs:
  - `out/rgro/dev_counterfactual_corpus.npz` + `out/rgro/dev_counterfactual_meta.json`
  - `out/rgro/mse_critic_keep.npz`
  - `out/rgro/dfl_critic_keep.npz`
  - `out/rgro/dfl_critic_v19.npz`
  - `out/rgro/deliverable1_dfl_causal_attribution.{json,md}` (Deliverable 1)
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Optional

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.baselines.policy_registry import create_discrete_policy_env
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.pdg_liveness import PDG2Config, PDG2SCCLivenessMonitor
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
    train_dfl_advantage_ensemble,
)
from experiments.rgro.rgro_common import (
    OUT_RGRO_DIR,
    assert_dev_quarantine,
    build_dev_stress_24_specs,
    build_dev_train_specs,
    paired_stats,
    rollout_multi_horizon,
    summarize_rows,
)
from experiments.v20_audit.common import RecordedStep, build_val72_specs
from experiments.v20_audit.proposer_variants import ExpertSuiteEvaluator

HARM_DELTA_JA = 50.0


def _build_corpus_episode_plan() -> list[tuple[str, dict[str, Any]]]:
    """Build 84 strictly quarantined DEV episodes tagged with episode-disjoint split (`train`/`cal`/`test_dev`)."""
    train_raw = build_dev_train_specs(max_4topo_seeds=2, max_ws_seeds=3)
    # Stratified 36 TRAIN episodes: 16 4-Topo + 10 Track1 + 10 Track2
    tr_4 = [s for s in train_raw if s.get("suite_group") == "4topo"][:16]
    tr_1 = [s for s in train_raw if s.get("suite_group") == "track1"][:10]
    tr_2 = [s for s in train_raw if s.get("suite_group") == "track2"][:10]
    train_std = tr_4 + tr_1 + tr_2

    val72 = build_val72_specs()
    # Stratified 24 VAL episodes split 12 cal / 12 test_dev (episode-disjoint!)
    v_4 = [s for s in val72 if s.get("suite_group") == "4topo"]
    v_1 = [s for s in val72 if s.get("suite_group") == "track1"]
    v_2 = [s for s in val72 if s.get("suite_group") == "track2"]
    val_sub = v_4[:12] + v_1[:6] + v_2[:6]  # 24 eps

    dev_stress = build_dev_stress_24_specs()  # 24 eps (6 seeds x 4 families)

    plan: list[tuple[str, dict[str, Any]]] = []
    for sp in train_std:
        plan.append(("train", sp))
    for i, sp in enumerate(val_sub):
        plan.append(("cal" if (i % 2 == 0) else "test_dev", sp))
    for idx, sp in enumerate(dev_stress):
        # 6 seeds -> seeds 0,1,2 train (12 eps), seed 3,4? Better: seeds 0..2 train (12), seed 3 cal (4)+seed 4 half,
        # cleanly group by seed index (seed_idx = idx // 4): 0,1,2 -> train (12), 3 -> cal (4), 4 -> cal/test ->
        # Actually 6 seeds: 0,1,2 -> train (12 eps), 3,4 -> cal (8 eps? or 3 -> cal, 4,5 -> test? Use 0,1,2 train, 3,4 cal (6? no 4+4=8), 5 test (4))
        # Better balanced: seeds 0,1,2 -> train (12 eps), seed 3 + half of 4? Keep seed-disjoint:
        # 0,1,2 -> train (12 eps), 3 -> cal (4 eps), 4 -> test_dev (4 eps), 5 -> alternating? No, 0,1 -> train (8), 2,3 -> cal (8), 4,5 -> test_dev (8)!
        sd_i = idx // 4
        if sd_i in (0, 1):
            plan.append(("train", sp))
        elif sd_i in (2, 3):
            plan.append(("cal", sp))
        else:
            plan.append(("test_dev", sp))

    assert_dev_quarantine([int(sp["seed"]) for _, sp in plan], label="Task1_DEV_Corpus")
    return plan


def collect_expanded_dev_corpus(
    *,
    max_snaps_per_ep: int = 4,
    full_cont_stride: int = 3,
) -> list[dict[str, Any]]:
    """Collect >= 240 quarantined DEV counterfactual snapshots across horizons 16/48/128 + full."""
    plan = _build_corpus_episode_plan()
    suite = ExpertSuiteEvaluator(seed=0)
    v20_critic = RobustRankAsinhCritic(_DEFAULT_V20_CRITIC_CKPT)
    corpus: list[dict[str, Any]] = []
    snap_global_idx = 0
    t_start = time.perf_counter()

    for ep_i, (split_name, sp) in enumerate(plan):
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
            last_cap_tick = -100

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
                    env.sim,
                    env.tracker,
                    last_replan_tick=env.last_replan_tick,
                    recent_conflict_ticks=env.recent_conflict_ticks,
                    decision_logs=env.decision_logs,
                )
                mask = compute_expert_v10_action_mask(env.sim, env.last_replan_tick)
                act = int(ctrl.select_action(feats, dict(info, action_mask=mask), sim=env.sim))

                if (
                    act > 0
                    and t_now >= 12
                    and (t_now - last_cap_tick) >= 18
                    and len(captured) < max_snaps_per_ep
                ):
                    p2_fb = (
                        env.execution_perturbation.capture_feedback(env.sim, t_now)
                        if (env.execution_perturbation is not None and env._active_execution_aware)
                        else None
                    )
                    snap_cap = NativeSnapshotCapture.capture(
                        env.sim, env.tracker.operations, hide_future_releases=True, execution_feedback=p2_fb
                    )
                    h_vec = hist_buf.update_from_snapshot(snap_cap, rej_streak=rej_streak)
                    captured.append((t_now, int(act), snap_cap, h_vec.copy()))
                    last_cap_tick = t_now

                before_len = len(traces)
                obs, _, tm, tr, info = env.step(act)
                if len(traces) > before_len:
                    traces[-1].action = int(act)
                    if env.decision_logs:
                        dl = env.decision_logs[-1]
                        traces[-1].propose_ms = float(dl.get("t_propose_ms", 0.0))
                        rej_streak = 0 if bool(dl.get("receipt_accepted", True)) else (rej_streak + 1)
                else:
                    traces.append(
                        RecordedStep(step_before=t_now, action=int(act), triggered_replan=False, proposal=None)
                    )

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
                rob_cur_h = {
                    r.robot_id: r.cur_task
                    for r in snap.robots
                    if r.cur_task and r.robot_id not in clf.disturbed_robots
                }
                sim_h = int(keep_art.ja_proxy_details.get("sim_horizon", max(600, t_snap + 300)))
                keep_sc = float(keep_art.ja_unified_score or 1e9)
                keep_stats = build_keep_baseline_stats(snap, graph, keep_art.eval_res_b2, keep_sc, rob_cur_h, sim_h)
                fv_keep = extract_critic_features(
                    snap, net_obj, graph, keep_art.eval_res_b2, keep_sc,
                    keep_art.assign, keep_art.bays, keep_art.crews, keep_art.task_orders,
                    keep_stats, rob_cur_h, sim_h, is_keep_cand=True,
                )
                k_crit = v20_critic.predict_delta_keep(fv_keep)

                # Identify the V19 Reference Anchor (E2_AHG_Full, or E3_AHG_Staged if valid & lower JA, else E0_KEEP)
                v19_cands = [
                    exps.get(eid)
                    for eid in ("E2_AHG_Full", "E3_AHG_Staged")
                    if exps.get(eid) is not None
                    and exps[eid].eval_res_b2 is not None
                    and exps[eid].eval_res_b2.valid
                ]
                v19_anchor_art = (
                    min(v19_cands, key=lambda a: float(a.ja_unified_score or 1e9))
                    if v19_cands
                    else keep_art
                )

                # Deduplicate candidates while guaranteeing E0_KEEP (index 0) and V19_Anchor are present
                chosen_arts = [keep_art]
                seen_sigs = {keep_art.ensure_sig()}
                v19_anchor_idx = 0
                if v19_anchor_art.ensure_sig() not in seen_sigs:
                    seen_sigs.add(v19_anchor_art.ensure_sig())
                    v19_anchor_idx = len(chosen_arts)
                    chosen_arts.append(v19_anchor_art)

                for eid in ("E1_V18_Regret2", "E2_AHG_Full", "E3_AHG_Staged", "E4_V18_PGMPO", "E6_V18_Polish", "E7_CausalCone_VOI"):
                    a = exps.get(eid)
                    if a is not None and a.eval_res_b2 is not None and a.eval_res_b2.valid:
                        sg = a.ensure_sig()
                        if sg not in seen_sigs:
                            seen_sigs.add(sg)
                            chosen_arts.append(a)
                if len(chosen_arts) < 3:
                    continue

                do_full = (snap_global_idx % max(1, int(full_cont_stride)) == 0)
                raw_fvs: list[np.ndarray] = []
                cand_ids: list[str] = []
                cand_ja_u: list[float] = []
                cand_v20_sc: list[float] = []
                j16_list: list[float] = []
                j48_list: list[float] = []
                j128_list: list[float] = []
                jfull_list: list[float] = []
                ok_all = True

                for idx_a, art in enumerate(chosen_arts):
                    fv = extract_critic_features(
                        snap, net_obj, graph, art.eval_res_b2, float(art.ja_unified_score or keep_sc),
                        art.assign, art.bays, art.crews, art.task_orders,
                        keep_stats, rob_cur_h, sim_h, is_keep_cand=(idx_a == 0),
                    )
                    prop = art.to_proposal(
                        policy_id="T1_CF", snapshot=snap, clf=clf, eval_res=art.eval_res_b2,
                        eval_cnt=1, t0=0.0, budget_ms=35.0, fallback_keep=(idx_a == 0),
                        include_eval_seq_when_unwindowed=True,
                    )
                    try:
                        ro = rollout_multi_horizon(
                            sp, traces, t_snap, prop,
                            action_at_target=int(act_snap),
                            horizons=(16, 48, 128),
                            full_continuation=do_full,
                        )
                    except Exception:
                        ok_all = False
                        break

                    ja_d = float(art.ja_unified_score or keep_sc) - keep_sc
                    c_d = float(v20_critic.predict_delta_keep(fv) - k_crit)
                    v20_d = ja_d + (0.25 * c_d if len(net_obj.nodes) >= 30 else 0.0)

                    raw_fvs.append(fv)
                    cand_ids.append(str(art.expert_id))
                    cand_ja_u.append(ja_d)
                    cand_v20_sc.append(v20_d)
                    j16_list.append(float(ro["j_16"]))
                    j48_list.append(float(ro["j_48"]))
                    j128_list.append(float(ro["j_128"]))
                    if do_full:
                        jfull_list.append(float(ro["j_full"]))

                if not ok_all or len(j48_list) < 3:
                    continue
                if (max(j48_list) - min(j48_list)) < 1.0:
                    continue

                fv_v19 = raw_fvs[v19_anchor_idx]
                X_rel_keep = np.stack(
                    [build_anchor_relative_feature(fv, fv_keep, h_vec) for fv in raw_fvs], axis=0
                ).astype(np.float32)
                X_rel_v19 = np.stack(
                    [build_anchor_relative_feature(fv, fv_v19, h_vec) for fv in raw_fvs], axis=0
                ).astype(np.float32)

                j48_arr = np.asarray(j48_list, dtype=np.float32)
                j16_arr = np.asarray(j16_list, dtype=np.float32)
                j128_arr = np.asarray(j128_list, dtype=np.float32)
                jfull_arr = np.asarray(jfull_list, dtype=np.float32) if do_full else None

                corpus.append(
                    {
                        "snap_id": int(snap_global_idx),
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
                        "j16": j16_arr,
                        "j48": j48_arr,
                        "j128": j128_arr,
                        "jfull": jfull_arr,
                        "true_delta_keep": (j48_arr - j48_arr[0]).astype(np.float32),
                        "true_delta_v19": (j48_arr - j48_arr[v19_anchor_idx]).astype(np.float32),
                    }
                )
                snap_global_idx += 1

        if (ep_i + 1) % 12 == 0 or (ep_i + 1) == len(plan):
            el = time.perf_counter() - t_start
            print(
                f"  [Corpus] ep {ep_i+1:2d}/{len(plan)} | snaps={len(corpus):3d} "
                f"(full_cont={sum(1 for c in corpus if c['jfull'] is not None)}) | elapsed={el:.1f}s",
                flush=True,
            )

    return corpus


def save_corpus(corpus: list[dict[str, Any]], npz_path: Path, meta_path: Path) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    metas: list[dict[str, Any]] = []
    for i, item in enumerate(corpus):
        arrays[f"s{i}_h_vec"] = item["h_vec"]
        arrays[f"s{i}_raw_fvs"] = item["raw_fvs"]
        arrays[f"s{i}_X_rel_keep"] = item["X_rel_keep"]
        arrays[f"s{i}_X_rel_v19"] = item["X_rel_v19"]
        arrays[f"s{i}_l0_ja_u"] = item["l0_ja_u"]
        arrays[f"s{i}_l0_v20_critic"] = item["l0_v20_critic"]
        arrays[f"s{i}_j16"] = item["j16"]
        arrays[f"s{i}_j48"] = item["j48"]
        arrays[f"s{i}_j128"] = item["j128"]
        if item["jfull"] is not None:
            arrays[f"s{i}_jfull"] = item["jfull"]
        arrays[f"s{i}_true_delta_keep"] = item["true_delta_keep"]
        arrays[f"s{i}_true_delta_v19"] = item["true_delta_v19"]
        metas.append(
            {
                "idx": i,
                "snap_id": item["snap_id"],
                "split": item["split"],
                "is_stress": item["is_stress"],
                "episode_key": item["episode_key"],
                "suite_group": item["suite_group"],
                "seed": item["seed"],
                "tick": item["tick"],
                "cand_ids": item["cand_ids"],
                "v19_anchor_idx": item["v19_anchor_idx"],
                "has_jfull": item["jfull"] is not None,
            }
        )
    np.savez_compressed(str(npz_path), **arrays)
    meta_path.write_text(json.dumps(metas, indent=2), encoding="utf-8")


def load_corpus(npz_path: Path, meta_path: Path) -> list[dict[str, Any]]:
    metas = json.loads(meta_path.read_text(encoding="utf-8"))
    data = np.load(str(npz_path), allow_pickle=False)
    out: list[dict[str, Any]] = []
    for m in metas:
        i = int(m["idx"])
        out.append(
            {
                "snap_id": int(m["snap_id"]),
                "split": str(m["split"]),
                "is_stress": bool(m["is_stress"]),
                "episode_key": str(m["episode_key"]),
                "suite_group": str(m["suite_group"]),
                "seed": int(m["seed"]),
                "tick": int(m["tick"]),
                "cand_ids": list(m["cand_ids"]),
                "v19_anchor_idx": int(m["v19_anchor_idx"]),
                "h_vec": np.asarray(data[f"s{i}_h_vec"], dtype=np.float32),
                "raw_fvs": np.asarray(data[f"s{i}_raw_fvs"], dtype=np.float32),
                "X_rel_keep": np.asarray(data[f"s{i}_X_rel_keep"], dtype=np.float32),
                "X_rel_v19": np.asarray(data[f"s{i}_X_rel_v19"], dtype=np.float32),
                "l0_ja_u": np.asarray(data[f"s{i}_l0_ja_u"], dtype=np.float32),
                "l0_v20_critic": np.asarray(data[f"s{i}_l0_v20_critic"], dtype=np.float32),
                "j16": np.asarray(data[f"s{i}_j16"], dtype=np.float32),
                "j48": np.asarray(data[f"s{i}_j48"], dtype=np.float32),
                "j128": np.asarray(data[f"s{i}_j128"], dtype=np.float32),
                "jfull": np.asarray(data[f"s{i}_jfull"], dtype=np.float32) if m["has_jfull"] else None,
                "true_delta_keep": np.asarray(data[f"s{i}_true_delta_keep"], dtype=np.float32),
                "true_delta_v19": np.asarray(data[f"s{i}_true_delta_v19"], dtype=np.float32),
            }
        )
    return out


def compute_phase0_headroom_audit(corpus: list[dict[str, Any]]) -> dict[str, Any]:
    """Phase 0 Empirical Headroom & Multi-Horizon Alignment Audit."""
    def _audit_subset(snaps: list[dict[str, Any]], horizon_key: str) -> dict[str, Any]:
        valid = [s for s in snaps if s.get(horizon_key) is not None]
        if not valid:
            return {}
        keep_costs, v19_costs, jau_costs, v20c_costs, oracle_costs = [], [], [], [], []
        jau_harm_vs_v19 = 0
        jau_gain_vs_v19 = 0
        v19_beats_jau_headroom = []
        for s in valid:
            j = np.asarray(s[horizon_key], dtype=np.float64)
            v_idx = int(s["v19_anchor_idx"])
            jau_idx = int(np.argmin(s["l0_ja_u"]))
            v20_idx = int(np.argmin(s["l0_v20_critic"]))
            keep_costs.append(float(j[0]))
            v19_costs.append(float(j[v_idx]))
            jau_costs.append(float(j[jau_idx]))
            v20c_costs.append(float(j[v20_idx]))
            oracle_costs.append(float(np.min(j)))
            diff_jau_v19 = float(j[jau_idx] - j[v_idx])
            if diff_jau_v19 > HARM_DELTA_JA:
                jau_harm_vs_v19 += 1
            elif diff_jau_v19 < -HARM_DELTA_JA:
                jau_gain_vs_v19 += 1
            v19_beats_jau_headroom.append(float(j[v_idx] - np.min(j)))
        mean_v19 = float(np.mean(v19_costs))
        mean_oracle = float(np.mean(oracle_costs))
        return {
            "n_snapshots": len(valid),
            "mean_keep": round(float(np.mean(keep_costs)), 2),
            "mean_v19_anchor": round(mean_v19, 2),
            "mean_ja_unified_sel": round(float(np.mean(jau_costs)), 2),
            "mean_v20_critic_sel": round(float(np.mean(v20c_costs)), 2),
            "mean_pool_oracle": round(mean_oracle, 2),
            "oracle_headroom_over_v19_ja": round(mean_v19 - mean_oracle, 2),
            "oracle_headroom_over_v19_pct": round(100.0 * (mean_v19 - mean_oracle) / max(1e-6, mean_v19), 2),
            "naive_jau_regret_vs_oracle": round(float(np.mean(jau_costs)) - mean_oracle, 2),
            "naive_jau_harm_rate_vs_v19_pct": round(100.0 * jau_harm_vs_v19 / len(valid), 2),
            "naive_jau_gain_rate_vs_v19_pct": round(100.0 * jau_gain_vs_v19 / len(valid), 2),
        }

    # Multi-horizon ranking alignment vs exact full-episode continuation `jfull`
    full_snaps = [s for s in corpus if s["jfull"] is not None]
    alignment: dict[str, Any] = {}
    for hk in ("l0_ja_u", "j16", "j48", "j128"):
        rhos, regs_full, top1_hits = [], [], []
        for s in full_snaps:
            proxy = np.asarray(s[hk], dtype=np.float64)
            jf = np.asarray(s["jfull"], dtype=np.float64)
            if np.std(proxy) > 1e-8 and np.std(jf) > 1e-8:
                r = spearmanr(proxy, jf).statistic
                if math.isfinite(r):
                    rhos.append(float(r))
            sel = int(np.argmin(proxy))
            regs_full.append(float(jf[sel] - np.min(jf)))
            top1_hits.append(1.0 if sel == int(np.argmin(jf)) else 0.0)
        alignment[hk] = {
            "n_full_snaps": len(full_snaps),
            "spearman_vs_jfull": round(float(np.mean(rhos)), 4) if rhos else 0.0,
            "mean_full_regret": round(float(np.mean(regs_full)), 2) if regs_full else 0.0,
            "top1_match_jfull_pct": round(100.0 * float(np.mean(top1_hits)), 2) if top1_hits else 0.0,
        }

    std_snaps = [s for s in corpus if not s["is_stress"]]
    str_snaps = [s for s in corpus if s["is_stress"]]
    return {
        "total_snapshots": len(corpus),
        "n_episodes": len({s["episode_key"] for s in corpus}),
        "n_full_continuation": len(full_snaps),
        "by_horizon_all": {hk: _audit_subset(corpus, hk) for hk in ("j16", "j48", "j128", "jfull")},
        "j48_standard_dev": _audit_subset(std_snaps, "j48"),
        "j48_stress_dev": _audit_subset(str_snaps, "j48"),
        "jfull_standard_dev": _audit_subset(std_snaps, "jfull"),
        "jfull_stress_dev": _audit_subset(str_snaps, "jfull"),
        "horizon_alignment_vs_jfull": alignment,
    }


def _eval_offline_selector(
    snaps: list[dict[str, Any]],
    score_fn,
    *,
    anchor_mode: str = "keep",
    horizon_key: str = "j48",
) -> dict[str, float]:
    rhos, regrets, harm_hits, pair_ok, pair_tot, top2_hits = [], [], 0, 0, 0, 0
    valid = [s for s in snaps if s.get(horizon_key) is not None]
    for s in valid:
        preds = np.asarray(score_fn(s), dtype=np.float64)
        j = np.asarray(s[horizon_key], dtype=np.float64)
        a_idx = 0 if anchor_mode == "keep" else int(s["v19_anchor_idx"])
        tr = j - j[a_idx]
        sel = int(np.argmin(preds))
        regrets.append(float(tr[sel] - np.min(tr)))
        if float(tr[sel]) > HARM_DELTA_JA:
            harm_hits += 1
        if int(np.argmin(tr)) in set(np.argsort(preds)[: min(2, len(tr))].tolist()):
            top2_hits += 1
        if np.std(preds) > 1e-8 and np.std(tr) > 1e-8:
            r = spearmanr(preds, tr).statistic
            if math.isfinite(r):
                rhos.append(float(r))
        for i in range(len(tr)):
            for k in range(i + 1, len(tr)):
                if abs(tr[i] - tr[k]) > 1.0:
                    pair_tot += 1
                    if (preds[i] - preds[k]) * (tr[i] - tr[k]) > 0.0:
                        pair_ok += 1
    return {
        "n_snaps": len(valid),
        "spearman_rho": round(float(np.mean(rhos)), 4) if rhos else 0.0,
        "mean_regret": round(float(np.mean(regrets)), 2) if regrets else 0.0,
        "harm_rate_pct": round(100.0 * harm_hits / max(1, len(valid)), 2),
        "pairwise_acc": round(pair_ok / max(1, pair_tot), 4),
        "top2_recall": round(top2_hits / max(1, len(valid)), 4),
    }


class _DFLV20CriticAdapter:
    """Thin read-only adapter plugging `DFLAdvantageEnsemble` into frozen `V20_1_NAJS` (`C1/C2/C3`)."""

    def __init__(
        self,
        ensemble: DFLAdvantageEnsemble,
        hist_buf: ExecutionHistoryBuffer,
        *,
        risk_gate: bool = False,
        harm_tau: float = 0.25,
        lcb_kappa: float = 0.5,
    ) -> None:
        self.ensemble = ensemble
        self.hist_buf = hist_buf
        self.risk_gate = bool(risk_gate)
        self.harm_tau = float(harm_tau)
        self.lcb_kappa = float(lcb_kappa)
        self._cur_tick: int = -1
        self._fv_keep: Optional[np.ndarray] = None

    def begin_tick(self, tick: int) -> None:
        self._cur_tick = int(tick)
        self._fv_keep = None

    def predict_delta_keep(self, fv: np.ndarray) -> float:
        fv_arr = np.asarray(fv, dtype=np.float32).reshape(-1)
        if self._fv_keep is None:
            self._fv_keep = fv_arr.copy()
            return 0.0
        h_vec = self.hist_buf.get_vector_readonly()
        x_rel = build_anchor_relative_feature(fv_arr, self._fv_keep, h_vec)
        sc = self.ensemble.score_batch(x_rel[None, :], kappa=self.lcb_kappa)[0]
        if self.risk_gate and (sc.harm_prob > self.harm_tau or sc.lcb_advantage < -25.0):
            return 10000.0
        return float(sc.pred_delta_ja)


def _run_c_arm_episode(
    spec: dict[str, Any],
    arm_id: str,
    *,
    mse_ens: DFLAdvantageEnsemble,
    dfl_ens: DFLAdvantageEnsemble,
    harm_tau_c3: float,
    use_virtual_clock: bool,
) -> dict[str, Any]:
    """Run one controlled episode under identical frozen `V20_1_NAJS` backbone, swapping ONLY the critic (`C0..C3`)."""
    def _exec() -> dict[str, Any]:
        _, ctrl, env = create_discrete_policy_env(
            "V20_1_NAJS",
            seed=int(spec["seed"]),
            max_steps=int(spec["max_steps"]),
            use_rust=True,
            p2_cfg=spec.get("p2_cfg"),
        )
        prop = env._raw_external_proposer
        mon: PDG2SCCLivenessMonitor = env._pdg_monitor
        hist_buf = ExecutionHistoryBuffer(maxlen=12)
        adapter: Optional[_DFLV20CriticAdapter] = None

        if arm_id == "C1_MSE":
            adapter = _DFLV20CriticAdapter(mse_ens, hist_buf, risk_gate=False)
            prop._critic = adapter
            prop.critic_mode = "all"
            prop.critic_beta = 0.35
        elif arm_id == "C2_DFL":
            adapter = _DFLV20CriticAdapter(dfl_ens, hist_buf, risk_gate=False)
            prop._critic = adapter
            prop.critic_mode = "all"
            prop.critic_beta = 0.35
        elif arm_id == "C3_DFL_Gate":
            adapter = _DFLV20CriticAdapter(dfl_ens, hist_buf, risk_gate=True, harm_tau=harm_tau_c3, lcb_kappa=0.5)
            prop._critic = adapter
            prop.critic_mode = "all"
            prop.critic_beta = 0.35

        obs, info = env.reset(seed=int(spec["seed"]), options={"scenario_tuple": copy.deepcopy(spec["scenario_tuple"])})
        tm = tr = False
        rej_streak = 0
        while not (tm or tr):
            t_now = int(env.current_step)
            if adapter is not None and (t_now % 5 == 0 or t_now == 0):
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
            s_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
            act = int(ctrl.select_action(feats, s_info, sim=env.sim))
            act = int(mon.observe_and_maybe_escalate(env, s_info, act, proposer=prop))

            if adapter is not None and act > 0:
                p2_fb = (
                    env.execution_perturbation.capture_feedback(env.sim, t_now)
                    if (env.execution_perturbation is not None and env._active_execution_aware)
                    else None
                )
                s_cur = NativeSnapshotCapture.capture(
                    env.sim, env.tracker.operations, hide_future_releases=True, execution_feedback=p2_fb
                )
                hist_buf.update_from_snapshot(s_cur, rej_streak=rej_streak)
                adapter.begin_tick(t_now)

            nb = len(env.decision_logs)
            obs, _, tm, tr, info = env.step(act)
            if len(env.decision_logs) > nb:
                dl = env.decision_logs[-1]
                rej_streak = 0 if bool(dl.get("receipt_accepted", True)) else (rej_streak + 1)

        m = env.compute_episode_metrics()
        aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
        dec_ms = [float(r["t_propose_ms"]) + float(r["t_validate_commit_ms"]) for r in env.decision_logs]
        return {
            "arm": arm_id,
            "episode_key": str(spec["episode_key"]),
            "suite_group": str(spec.get("suite_group", "")),
            "seed": int(spec["seed"]),
            "net_obj": round(float(m["net_obj"]), 4),
            "h1": round(float(m["cmax"]) + float(m.get("task_tardiness_cost", 0.0)) + float(m.get("op_tardiness_cost", 0.0)), 4),
            "churn": int(m["total_churn"]),
            "comp100": bool(float(m["task_completion_rate"]) >= 0.9999 and float(m["production_completion_rate"]) >= 0.9999),
            "audit_passed": bool(aud.pass_audit),
            "expired_windows": int(env.total_window_expirations),
            "decision_latencies_ms": [round(float(x), 4) for x in dec_ms],
        }

    if use_virtual_clock:
        with virtual_clock_scope(0.0):
            return _exec()
    return _exec()


def main() -> None:
    OUT_RGRO_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUT_RGRO_DIR / "dev_counterfactual_corpus.npz"
    meta_path = OUT_RGRO_DIR / "dev_counterfactual_meta.json"

    print("=== [Task 1A] Collecting Expanded Quarantined DEV Counterfactual Corpus (Horizons 16/48/128/Full) ===", flush=True)
    if npz_path.is_file() and meta_path.is_file():
        corpus = load_corpus(npz_path, meta_path)
        print(f"  Loaded cached corpus: {len(corpus)} snapshots", flush=True)
    else:
        corpus = collect_expanded_dev_corpus(max_snaps_per_ep=4, full_cont_stride=3)
        save_corpus(corpus, npz_path, meta_path)
        print(f"  Saved corpus: {len(corpus)} snapshots -> {npz_path}", flush=True)

    phase0 = compute_phase0_headroom_audit(corpus)
    print(
        f"  [Phase 0 Headroom +48t ALL] V19={phase0['by_horizon_all']['j48']['mean_v19_anchor']:.2f} "
        f"| PoolOracle={phase0['by_horizon_all']['j48']['mean_pool_oracle']:.2f} "
        f"| Headroom={phase0['by_horizon_all']['j48']['oracle_headroom_over_v19_ja']:.2f} "
        f"({phase0['by_horizon_all']['j48']['oracle_headroom_over_v19_pct']:.2f}%) "
        f"| Naive JAU HarmRate vs V19={phase0['by_horizon_all']['j48']['naive_jau_harm_rate_vs_v19_pct']:.1f}%",
        flush=True,
    )
    print(f"  [Horizon Alignment vs Full J_A] {phase0['horizon_alignment_vs_jfull']}", flush=True)

    print("=== [Task 1B] 5-Seed Offline Training (`C1: MSE` vs `C2: DFL`, KEEP-Anchored & V19-Anchored) ===", flush=True)
    tr_snaps = [s for s in corpus if s["split"] == "train"]
    cal_snaps = [s for s in corpus if s["split"] == "cal"]
    te_snaps = [s for s in corpus if s["split"] == "test_dev"]
    print(f"  Episode-disjoint splits: train={len(tr_snaps)}, cal={len(cal_snaps)}, test_dev={len(te_snaps)}", flush=True)

    # Prepare train/eval views for KEEP-anchored and V19-anchored
    def _view(snaps: list[dict[str, Any]], x_key: str, y_key: str) -> list[dict[str, Any]]:
        return [{"X_rel": s[x_key], "true_delta_anchor": s[y_key], **s} for s in snaps]

    tr_keep, cal_keep, te_keep = _view(tr_snaps, "X_rel_keep", "true_delta_keep"), _view(cal_snaps, "X_rel_keep", "true_delta_keep"), _view(te_snaps, "X_rel_keep", "true_delta_keep")
    tr_v19, cal_v19, te_v19 = _view(tr_snaps, "X_rel_v19", "true_delta_v19"), _view(cal_snaps, "X_rel_v19", "true_delta_v19"), _view(te_snaps, "X_rel_v19", "true_delta_v19")

    offline_5seed: dict[str, Any] = {}
    chosen_models: dict[str, DFLAdvantageEnsemble] = {}

    # Baselines on test_dev (KEEP-anchored & V19-anchored)
    offline_5seed["L0_JA_Unified_Keep"] = _eval_offline_selector(te_keep, lambda s: s["l0_ja_u"], anchor_mode="keep")
    offline_5seed["L0_V20_StaticCritic_Keep"] = _eval_offline_selector(te_keep, lambda s: s["l0_v20_critic"], anchor_mode="keep")
    offline_5seed["L0_JA_Unified_V19"] = _eval_offline_selector(te_v19, lambda s: s["l0_ja_u"], anchor_mode="v19")

    for tag, loss_m, tr_v, cal_v, te_v, anc_m in (
        ("C1_MSE_Keep", "MSE", tr_keep, cal_keep, te_keep, "keep"),
        ("C2_DFL_Keep", "DFL", tr_keep, cal_keep, te_keep, "keep"),
        ("C1_MSE_V19", "MSE", tr_v19, cal_v19, te_v19, "v19"),
        ("C2_DFL_V19", "DFL", tr_v19, cal_v19, te_v19, "v19"),
    ):
        seed_models = []
        seed_cal_regs = []
        te_metrics = []
        for sd in (42, 43, 44, 45, 46):
            ens = train_dfl_advantage_ensemble(tr_v, loss_mode=loss_m, n_members=3, hidden_dim=48, epochs=55, base_seed=sd, harm_delta_ja=HARM_DELTA_JA)
            cal_m = _eval_offline_selector(cal_v, lambda s, e=ens: [x.pred_delta_ja for x in e.score_batch(s["X_rel"])], anchor_mode=anc_m)
            te_m = _eval_offline_selector(te_v, lambda s, e=ens: [x.pred_delta_ja for x in e.score_batch(s["X_rel"])], anchor_mode=anc_m)
            seed_models.append(ens)
            seed_cal_regs.append(float(cal_m["mean_regret"]))
            te_metrics.append(te_m)
        # Pick median-cal-regret checkpoint (zero test leakage!)
        med_idx = int(np.argsort(seed_cal_regs)[len(seed_cal_regs) // 2])
        chosen_models[tag] = seed_models[med_idx]
        # Also evaluate median checkpoint on full-episode jfull subset of test_dev+cal
        full_te = [s for s in te_v if s["jfull"] is not None]
        full_m = _eval_offline_selector(
            full_te, lambda s, e=seed_models[med_idx]: [x.pred_delta_ja for x in e.score_batch(s["X_rel"])],
            anchor_mode=anc_m, horizon_key="jfull",
        )
        offline_5seed[tag] = {
            "n_test_snaps": len(te_v),
            "spearman_mean": round(float(np.mean([m["spearman_rho"] for m in te_metrics])), 4),
            "spearman_std": round(float(np.std([m["spearman_rho"] for m in te_metrics])), 4),
            "regret48_mean": round(float(np.mean([m["mean_regret"] for m in te_metrics])), 2),
            "regret48_std": round(float(np.std([m["mean_regret"] for m in te_metrics])), 2),
            "harm_rate_mean_pct": round(float(np.mean([m["harm_rate_pct"] for m in te_metrics])), 2),
            "pair_acc_mean": round(float(np.mean([m["pairwise_acc"] for m in te_metrics])), 4),
            "top2_mean": round(float(np.mean([m["top2_recall"] for m in te_metrics])), 4),
            "median_ckpt_jfull": full_m,
        }
        print(
            f"  {tag:14s} | rho={offline_5seed[tag]['spearman_mean']:.4f}±{offline_5seed[tag]['spearman_std']:.4f} "
            f"| Regret48={offline_5seed[tag]['regret48_mean']:6.2f}±{offline_5seed[tag]['regret48_std']:4.2f} "
            f"| Harm%={offline_5seed[tag]['harm_rate_mean_pct']:4.1f}% | Top2={offline_5seed[tag]['top2_mean']:.4f}",
            flush=True,
        )

    # Save pure-NumPy checkpoints
    chosen_models["C1_MSE_Keep"].save_npz(OUT_RGRO_DIR / "mse_critic_keep.npz")
    chosen_models["C2_DFL_Keep"].save_npz(OUT_RGRO_DIR / "dfl_critic_keep.npz")
    chosen_models["C2_DFL_V19"].save_npz(OUT_RGRO_DIR / "dfl_critic_v19.npz")

    # Verify search-order invariance & pure-NumPy latency
    ens_check = DFLAdvantageEnsemble.load_npz(OUT_RGRO_DIR / "dfl_critic_v19.npz")
    sample_X = te_v19[0]["X_rel"]
    fwd = [x.pred_delta_ja for x in ens_check.score_batch(sample_X)]
    rev = [x.pred_delta_ja for x in ens_check.score_batch(sample_X[::-1])][::-1]
    order_inv_err = float(np.max(np.abs(np.asarray(fwd) - np.asarray(rev))))
    t_inf0 = time.perf_counter()
    for _ in range(500):
        ens_check.score_batch(sample_X)
    inf_ms = ((time.perf_counter() - t_inf0) * 1000.0) / 500.0
    print(f"  [Verify] Pure-NumPy DFL Ensemble: order_inv_max_diff={order_inv_err:.2e}, lat={inf_ms:.4f} ms/snap", flush=True)

    # Select C3 DEV harm threshold cleanly on CAL split (alpha=0.10 target harm rate on cal_keep)
    c2_keep = chosen_models["C2_DFL_Keep"]
    cal_probs_harm = []
    for s in cal_keep:
        scs = c2_keep.score_batch(s["X_rel"])
        for k, sc in enumerate(scs):
            if k == 0:
                continue
            cal_probs_harm.append((float(sc.harm_prob), 1.0 if float(s["true_delta_keep"][k]) > HARM_DELTA_JA else 0.0))
    tau_c3 = 0.25
    for cand_tau in np.linspace(0.05, 0.55, 51):
        sel_h = [h for p, h in cal_probs_harm if p <= cand_tau]
        if sel_h and (sum(sel_h) + 1.0) / (len(sel_h) + 1.0) <= 0.15:
            tau_c3 = float(cand_tau)
    print(f"  [CAL] Selected C3 DEV harm_tau={tau_c3:.3f}", flush=True)

    print("=== [Task 1C] Controlled 4-Arm Closed-Loop Attribution (`C0..C3`) on VAL-72 + DEV-Stress-24 (P1 & P2) ===", flush=True)
    val72_specs = build_val72_specs()
    dev_stress_specs = build_dev_stress_24_specs()
    assert_dev_quarantine([int(s["seed"]) for s in val72_specs + dev_stress_specs], label="Task1C_ClosedLoop")

    arms = ("C0_V20_Static", "C1_MSE", "C2_DFL", "C3_DFL_Gate")
    cl_results: dict[str, Any] = {}
    for clock_name, use_vc in (("P1_Virtual", True), ("P2_WallClock", False)):
        cl_results[clock_name] = {}
        for suite_name, specs in (("VAL72", val72_specs), ("DEV_Stress24", dev_stress_specs)):
            rows_by_arm: dict[str, list[dict[str, Any]]] = {}
            for arm in arms:
                t_a0 = time.perf_counter()
                arm_rows = [
                    _run_c_arm_episode(
                        sp, arm,
                        mse_ens=chosen_models["C1_MSE_Keep"],
                        dfl_ens=chosen_models["C2_DFL_Keep"],
                        harm_tau_c3=tau_c3,
                        use_virtual_clock=use_vc,
                    )
                    for sp in specs
                ]
                rows_by_arm[arm] = arm_rows
                sm = summarize_rows(arm_rows)
                print(
                    f"  [{clock_name:12s} | {suite_name:12s}] {arm:14s} | Comp={sm['comp100_count']:2d}/{sm['n']} "
                    f"| MeanJA={sm['mean_ja']:8.2f} Med={sm['median_ja']:7.2f} CVaR95={sm['cvar95_ja']:8.2f} "
                    f"| P99={sm['p99_ms']:5.2f}ms Max={sm['max_ms']:5.2f}ms ({time.perf_counter()-t_a0:.1f}s)",
                    flush=True,
                )
            cl_results[clock_name][suite_name] = {
                "summary": {arm: summarize_rows(rows_by_arm[arm]) for arm in arms},
                "paired_vs_C0": {arm: paired_stats(rows_by_arm[arm], rows_by_arm["C0_V20_Static"]) for arm in arms[1:]},
                "paired_C2_vs_C1": paired_stats(rows_by_arm["C2_DFL"], rows_by_arm["C1_MSE"]),
                "paired_C3_vs_C2": paired_stats(rows_by_arm["C3_DFL_Gate"], rows_by_arm["C2_DFL"]),
            }

    payload = {
        "phase0_headroom": phase0,
        "offline_5seed": offline_5seed,
        "verification": {
            "order_invariance_max_diff": order_inv_err,
            "pure_numpy_inference_ms_per_snap": round(inf_ms, 5),
            "c3_cal_harm_tau": round(tau_c3, 4),
        },
        "closed_loop_c0_to_c3": cl_results,
    }
    (OUT_RGRO_DIR / "deliverable1_dfl_causal_attribution.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Write Markdown Deliverable 1
    lines: list[str] = [
        "# Deliverable 1 (`IF-CORE-RGRO-01`): Phase 0 Oracle Headroom & `DFL` 4-Level Causal Attribution",
        "",
        f"> Quarantined DEV Counterfactual Corpus: **{phase0['total_snapshots']} snapshots** across **{phase0['n_episodes']} DEV episodes** (`train={len(tr_snaps)} / cal={len(cal_snaps)} / test_dev={len(te_snaps)}`, including **{phase0['n_full_continuation']}** full-episode continuation snapshots). Pure-NumPy 3-member `DFLAdvantageEnsemble`: `{inf_ms:.4f} ms/snap`, search-order invariance max diff `{order_inv_err:.2e}`.",
        "",
        "## 1. Phase 0 Empirical Oracle Headroom & Multi-Horizon Alignment",
        "",
        "| Horizon / Subset | Snaps | `KEEP` | `V19` Anchor | Naive `JA_Unified` | Static V20 Critic | Pool Oracle | Headroom over `V19` | Naive `JA_U` Harm Rate vs `V19` (`>50`) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lbl, d in (
        ("ALL (+16t)", phase0["by_horizon_all"].get("j16", {})),
        ("ALL (+48t)", phase0["by_horizon_all"].get("j48", {})),
        ("ALL (+128t)", phase0["by_horizon_all"].get("j128", {})),
        ("ALL (Full J_A)", phase0["by_horizon_all"].get("jfull", {})),
        ("Standard DEV (+48t)", phase0["j48_standard_dev"]),
        ("DEV-Stress-24 (+48t)", phase0["j48_stress_dev"]),
        ("Standard DEV (Full J_A)", phase0["jfull_standard_dev"]),
        ("DEV-Stress-24 (Full J_A)", phase0["jfull_stress_dev"]),
    ):
        if d:
            lines.append(
                f"| {lbl} | {d['n_snapshots']} | {d['mean_keep']:.2f} | {d['mean_v19_anchor']:.2f} | "
                f"{d['mean_ja_unified_sel']:.2f} | {d['mean_v20_critic_sel']:.2f} | **{d['mean_pool_oracle']:.2f}** | "
                f"**{d['oracle_headroom_over_v19_ja']:.2f} ({d['oracle_headroom_over_v19_pct']:.2f}%)** | {d['naive_jau_harm_rate_vs_v19_pct']:.1f}% |"
            )

    lines += [
        "",
        "### Multi-Horizon Ranking Stability vs. Exact Full-Episode `J_A`",
        "",
        "| Surrogate / Horizon | N Full Snaps | Spearman $\\rho$ vs `J_full` | Mean Full-Episode Regret | Top-1 Match `J_full` |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for hk, ad in phase0["horizon_alignment_vs_jfull"].items():
        lines.append(
            f"| `{hk}` | {ad['n_full_snaps']} | {ad['spearman_vs_jfull']:.4f} | {ad['mean_full_regret']:.2f} | {ad['top1_match_jfull_pct']:.1f}% |"
        )

    lines += [
        "",
        "## 2. Level 1 & Level 2 — Offline 5-Seed Counterfactual Attribution (`test_dev` Episode-Disjoint)",
        "",
        "| Model | N Test | Spearman $\\rho$ | Mean `+48t` Regret | Harm Rate (`>50`) | Pairwise Acc | Top-2 Recall |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in ("L0_JA_Unified_Keep", "L0_V20_StaticCritic_Keep", "L0_JA_Unified_V19"):
        d = offline_5seed[k]
        lines.append(
            f"| `{k}` | {d['n_snaps']} | {d['spearman_rho']:.4f} | {d['mean_regret']:.2f} | {d['harm_rate_pct']:.1f}% | {d['pairwise_acc']:.4f} | {d['top2_recall']:.4f} |"
        )
    for k in ("C1_MSE_Keep", "C2_DFL_Keep", "C1_MSE_V19", "C2_DFL_V19"):
        d = offline_5seed[k]
        lines.append(
            f"| `{k}` (5-seed) | {d['n_test_snaps']} | {d['spearman_mean']:.4f}±{d['spearman_std']:.4f} | "
            f"**{d['regret48_mean']:.2f}±{d['regret48_std']:.2f}** | {d['harm_rate_mean_pct']:.1f}% | {d['pair_acc_mean']:.4f} | {d['top2_mean']:.4f} |"
        )

    lines += ["", "## 3. Level 4 — Full Closed-Loop Attribution (`C0..C3`, Identical `V20.1` Backbone)", ""]
    for clock_name in ("P1_Virtual", "P2_WallClock"):
        lines.append(f"### {clock_name}")
        lines.append("")
        lines.append("| Suite | Arm | Comp100 | Mean `J_A` | Med `J_A` | `CVaR_95` | `P95/P99/Max` (ms) | Δ vs `C0` (W/T/L, p_2s) |")
        lines.append("| :--- | :--- | :---: | ---: | ---: | ---: | :---: | :--- |")
        for suite_name in ("VAL72", "DEV_Stress24"):
            blk = cl_results[clock_name][suite_name]
            for arm in arms:
                sm = blk["summary"][arm]
                if arm == "C0_V20_Static":
                    vs_str = "Reference"
                else:
                    ps = blk["paired_vs_C0"][arm]
                    vs_str = f"{ps['mean_delta']:+.2f} ({ps['rel_delta_pct']:+.2f}%, {ps['wins']}/{ps['ties']}/{ps['losses']}, p={ps['wilcoxon_p_2sided']:.4f})"
                lines.append(
                    f"| {suite_name} | `{arm}` | {sm['comp100_count']}/{sm['n']} | {sm['mean_ja']:.2f} | {sm['median_ja']:.2f} | "
                    f"{sm['cvar95_ja']:.2f} | {sm['p95_ms']:.2f}/{sm['p99_ms']:.2f}/{sm['max_ms']:.2f} | {vs_str} |"
                )
            p21 = blk["paired_C2_vs_C1"]
            lines.append(
                f"| *{suite_name} DFL vs MSE* | `C2 vs C1` | — | — | — | — | — | "
                f"{p21['mean_delta']:+.2f} ({p21['rel_delta_pct']:+.2f}%, {p21['wins']}/{p21['ties']}/{p21['losses']}, p={p21['wilcoxon_p_2sided']:.4f}) |"
            )
        lines.append("")

    (OUT_RGRO_DIR / "deliverable1_dfl_causal_attribution.md").write_text("\n".join(lines), encoding="utf-8")
    print("=== [Task 1] COMPLETED! Wrote Deliverable 1 -> out/rgro/deliverable1_dfl_causal_attribution.{json,md} ===", flush=True)


if __name__ == "__main__":
    main()
