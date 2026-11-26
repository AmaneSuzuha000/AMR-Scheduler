"""Task B (`IF-CORE-V20-01`): V18-V19 Same-Snapshot Expert Complementarity, Oracle Headroom & 48-Tick Rollout Calibration."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.baselines.policy_registry import create_discrete_policy_env
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.cvcr.snapshot import NativeSnapshot, NativeSnapshotCapture
from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state
from experiments.train_v19_bc import (
    TRAIN_4TOPO_SEEDS,
    TRAIN_WORKSHOP_SEEDS,
    VAL_4TOPO_SEEDS,
    VAL_WORKSHOP_SEEDS,
    _build_4topo_episode_spec,
    _build_workshop_episode_spec,
)
from experiments.v20_audit.common import (
    OUT_AUDIT_DIR,
    RecordedStep,
    assert_stage1_quarantine,
    make_v18_external_proposer,
    rollout_48_and_optional_full,
)
from experiments.v20_audit.proposer_variants import (
    EXPERT_IDS,
    ExpertSuiteEvaluator,
)
from tools.benchmarks.compare_workshop_algorithms import REGIMES as WORKSHOP_REGIMES
from experiments.v18_confirmation_scenarios import LAYOUTS as TOPO4_LAYOUTS, REGIMES as TOPO4_REGIMES


def _lifecycle_bucket(snapshot: NativeSnapshot) -> str:
    n_t = max(1, len(snapshot.tasks))
    done_ratio = sum(1 for t in snapshot.tasks if t.status.lower() == "completed") / float(n_t)
    has_fault = any(bool(getattr(r, "faulted", False)) for r in snapshot.robots)
    has_urgent = any(bool(getattr(t, "is_urgent", False)) and t.status.lower() in ("pending", "assigned") for t in snapshot.tasks)
    if has_fault or has_urgent:
        return "disruption_active"
    if done_ratio < 0.25:
        return "early_wave"
    if done_ratio < 0.70:
        return "mid_flow"
    return "tail_cleanup"


def _collect_stratified_snapshots(target_count: int = 640) -> list[dict[str, Any]]:
    """Collect >= 600 diverse stratified decision snapshots across quarantined TRAIN+VAL trajectories."""
    tr_4t = TRAIN_4TOPO_SEEDS[:4]
    val_4t = VAL_4TOPO_SEEDS
    tr_ws = TRAIN_WORKSHOP_SEEDS[:7]
    val_ws = VAL_WORKSHOP_SEEDS
    assert_stage1_quarantine(
        list(tr_4t) + list(val_4t) + list(tr_ws) + list(val_ws),
        label="TaskB_SnapshotPool",
    )

    specs: list[dict[str, Any]] = []
    # Mix 4-Topo + Track 1 + Track 2 across all regimes cleanly
    for s in list(tr_4t) + list(val_4t):
        for lay in ("offset_dual_lane", "perimeter_bypass", "narrow_spine", "workshop_cells"):
            for reg in TOPO4_REGIMES:
                sp = _build_4topo_episode_spec(lay, s, reg)
                sp["suite_group"] = "4topo"
                specs.append(sp)
    for tr in ("track1", "track2"):
        for s in list(tr_ws) + list(val_ws):
            for reg in WORKSHOP_REGIMES:
                sp = _build_workshop_episode_spec(tr, s, reg)
                sp["suite_group"] = tr
                specs.append(sp)

    # Interleave trajectories across V19_ADAPTIVE_RL and V18_EA_JOINT for unbiased state distribution
    rng = np.random.default_rng(20261124)
    per_group_target = {
        "4topo": max(260, int(target_count * 0.40)),
        "track1": max(200, int(target_count * 0.31)),
        "track2": max(200, int(target_count * 0.31)),
    }
    collected_by_group: dict[str, list[dict[str, Any]]] = {"4topo": [], "track1": [], "track2": []}

    # Shuffle order deterministically per suite group so both V19 and V18 states populate every regime
    by_grp: dict[str, list[dict[str, Any]]] = {"4topo": [], "track1": [], "track2": []}
    for sp in specs:
        by_grp[str(sp["suite_group"])].append(sp)
    for grp, lst in by_grp.items():
        idx_perm = rng.permutation(len(lst))
        by_grp[grp] = [lst[i] for i in idx_perm]

    for grp, grp_specs in by_grp.items():
        cap_n = per_group_target[grp]
        for ep_i, sp in enumerate(grp_specs):
            if len(collected_by_group[grp]) >= cap_n:
                break
            src_pol = "V19_ADAPTIVE_RL" if (ep_i % 2 == 0) else "V18_EA_JOINT"
            sc_copy = copy.deepcopy(sp["scenario_tuple"])
            p2_copy = copy.deepcopy(sp["p2_cfg"])
            seed = int(sp["seed"])
            max_steps = int(sp["max_steps"])
            kind, ctrl, env = create_discrete_policy_env(
                src_pol,
                topology=str(sc_copy[0]),
                seed=seed,
                max_steps=max_steps,
                use_rust=True,
                p2_cfg=p2_copy,
                allow_missing_pgmpo=False,
            )
            if src_pol == "V18_EA_JOINT":
                raw_ext = make_v18_external_proposer(seq_diff_b2=False, env_ref=env)
            else:
                raw_ext = env.external_proposer
            assert raw_ext is not None

            prefix_steps: list[RecordedStep] = []
            snaps_this_ep: list[dict[str, Any]] = []

            def _hook(snap: NativeSnapshot, net: Any, *, budget_ms: float = 35.0) -> CandidateProposal:
                prop = raw_ext(snap, net, budget_ms=budget_ms)
                prefix_steps.append(
                    RecordedStep(
                        step_before=int(snap.tick),
                        action=-1,
                        triggered_replan=True,
                        proposal=copy.deepcopy(prop),
                    )
                )
                snaps_this_ep.append(
                    {
                        "spec": sp,
                        "source_policy": src_pol,
                        "suite_group": grp,
                        "regime": str(sp["regime"]),
                        "episode_key": str(sp["episode_key"]),
                        "tick": int(snap.tick),
                        "lifecycle": _lifecycle_bucket(snap),
                        "network": copy.deepcopy(env.sim.network),
                        "snapshot": copy.deepcopy(snap),
                        "source_proposal": copy.deepcopy(prop),
                        "prefix_steps": [copy.copy(x) for x in prefix_steps[:-1]],
                        "action_at_tick": -1,  # filled right after step()
                    }
                )
                return prop

            env.external_proposer = _hook
            obs, info = env.reset(seed=seed, options={"scenario_tuple": sc_copy})
            tm = False
            tr = False
            while not (tm or tr):
                if kind == "rule_v10":
                    f = compute_expert_v9_state(
                        env.sim, env.tracker, last_replan_tick=env.last_replan_tick,
                        recent_conflict_ticks=env.recent_conflict_ticks, decision_logs=env.decision_logs,
                    )
                    si = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
                elif kind == "v18":
                    f = ctrl.observe(env, obs)
                    si = info
                else:
                    f = env.get_expert_v12_obs(obs)
                    si = info
                act = int(ctrl.select_action(f, si, sim=env.sim))
                s_before = int(env.current_step)
                before_n = len(prefix_steps)
                obs, _, tm, tr, info = env.step(act)
                if len(prefix_steps) > before_n:
                    prefix_steps[-1].action = act
                    snaps_this_ep[-1]["action_at_tick"] = act
                else:
                    prefix_steps.append(
                        RecordedStep(step_before=s_before, action=act, triggered_replan=False, proposal=None)
                    )

            # Sample up to 8 diverse snapshots per episode across lifecycle buckets
            if len(snaps_this_ep) > 8:
                pick_idx = sorted({int(round(q * (len(snaps_this_ep) - 1))) for q in np.linspace(0.0, 1.0, 8)})
                snaps_this_ep = [snaps_this_ep[k] for k in pick_idx]
            for item in snaps_this_ep:
                # Truncate proposal payloads on old steps? They are pure Python; fast copy
                if len(collected_by_group[grp]) < cap_n:
                    collected_by_group[grp].append(item)

    all_snaps = collected_by_group["4topo"] + collected_by_group["track1"] + collected_by_group["track2"]
    return all_snaps


def run_task_b(*, target_snapshots: int = 640, rollout_cal_snapshots: int = 180) -> dict[str, Any]:
    """Execute Task B across N >= 600 snapshots (`E0-E9` Oracle + LOO + 48-tick calibration)."""
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Task B] Collecting >= {target_snapshots} stratified TRAIN+VAL snapshots...", flush=True)
    snaps = _collect_stratified_snapshots(target_count=target_snapshots)
    n_snaps = len(snaps)
    print(f"[Task B] Collected {n_snaps} snapshots; evaluating all 10 experts (E0-E9)...", flush=True)

    evaluator = ExpertSuiteEvaluator(seed=0)
    oracle_gaps_vs_v121: list[float] = []
    oracle_gaps_vs_keep: list[float] = []
    v121_valid_refs: list[float] = []
    oracle_valid_refs: list[float] = []
    feasibility_rescue_count = 0
    v121_infeasible_count = 0
    gaps_by_suite: dict[str, list[float]] = {"4topo": [], "track1": [], "track2": []}
    gaps_by_regime: dict[str, list[float]] = {}
    win_counts: dict[str, int] = {e: 0 for e in EXPERT_IDS}
    near_oracle_counts: dict[str, int] = {e: 0 for e in EXPERT_IDS}
    oracle_beats_v121_by_suite: dict[str, dict[str, int]] = {}
    expert_scores_mat: list[dict[str, Optional[float]]] = []
    expert_valid_counts: dict[str, int] = {e: 0 for e in EXPERT_IDS}
    expert_gen_lats: dict[str, list[float]] = {e: [] for e in EXPERT_IDS}
    expert_b2_lats: dict[str, list[float]] = {e: [] for e in EXPERT_IDS}
    oracle_win_when_v121_loses: dict[str, int] = {e: 0 for e in EXPERT_IDS}

    # Select balanced indices for 48-tick rollout calibration
    cal_step = max(1, n_snaps // max(1, rollout_cal_snapshots))
    cal_indices = set(range(0, n_snaps, cal_step)[:rollout_cal_snapshots])
    full_cont_indices = set(list(sorted(cal_indices))[:: max(1, len(cal_indices) // 30)][:30])

    cal_rows: list[dict[str, Any]] = []
    rho_b2_list: list[float] = []
    rho_ja_list: list[float] = []
    rho_wsurr_list: list[float] = []
    concord_b2: list[float] = []
    concord_ja: list[float] = []
    concord_ws: list[float] = []
    regret_b2: list[float] = []
    regret_ja: list[float] = []
    regret_ws: list[float] = []
    full_rho_ja: list[float] = []
    full_regret_ja: list[float] = []

    for idx, item in enumerate(snaps):
        snap: NativeSnapshot = item["snapshot"]
        net = item["network"]
        grp = str(item["suite_group"])
        reg = str(item["regime"])

        art_map = evaluator.evaluate_all_experts(snap, net)
        e_scores: dict[str, Optional[float]] = {}
        for eid in EXPERT_IDS:
            art = art_map[eid]
            ok = bool(art.eval_res_b2 is not None and art.eval_res_b2.valid and art.ja_unified_score is not None)
            if ok:
                expert_valid_counts[eid] += 1
                e_scores[eid] = float(art.ja_unified_score)
            else:
                e_scores[eid] = None
            expert_gen_lats[eid].append(float(art.gen_ms))
            expert_b2_lats[eid].append(float(art.b2_ms))
        expert_scores_mat.append(e_scores)

        valid_eids = [e for e in EXPERT_IDS if e_scores[e] is not None]
        if valid_eids:
            oracle_eid = min(valid_eids, key=lambda e: float(e_scores[e]))
            oracle_sc = float(e_scores[oracle_eid])
            win_counts[oracle_eid] += 1
            for eid in valid_eids:
                if float(e_scores[eid]) <= oracle_sc + max(5.0, 0.01 * max(1.0, abs(oracle_sc))):
                    near_oracle_counts[eid] += 1

            v121_cands = [float(e_scores[e]) for e in ("E0_KEEP", "E2_AHG_Full", "E3_AHG_Staged") if e_scores[e] is not None]
            if not v121_cands:
                v121_infeasible_count += 1
                feasibility_rescue_count += 1
                oracle_win_when_v121_loses[oracle_eid] += 1
            else:
                v121_ref = min(v121_cands)
                v121_valid_refs.append(v121_ref)
                oracle_valid_refs.append(oracle_sc)
                gap_v121 = max(0.0, v121_ref - oracle_sc)
                oracle_gaps_vs_v121.append(gap_v121)
                gaps_by_suite[grp].append(gap_v121)
                gaps_by_regime.setdefault(reg, []).append(gap_v121)
                if e_scores["E0_KEEP"] is not None:
                    oracle_gaps_vs_keep.append(max(0.0, float(e_scores["E0_KEEP"]) - oracle_sc))
                if gap_v121 > 1.0:
                    oracle_win_when_v121_loses[oracle_eid] += 1
                    oracle_beats_v121_by_suite.setdefault(grp, {}).setdefault(oracle_eid, 0)
                    oracle_beats_v121_by_suite[grp][oracle_eid] += 1
        else:
            v121_infeasible_count += 1

        # Cloned 48-tick forward rollout proxy calibration
        if idx in cal_indices:
            from amr_sched.cvcr.commitment import classify_commitments

            clf_i = classify_commitments(snap)
            dedup_sigs: dict[str, dict[str, Any]] = {}
            do_full = idx in full_cont_indices
            for eid in EXPERT_IDS:
                art = art_map[eid]
                if not (art.eval_res_b2 and art.eval_res_b2.valid):
                    continue
                sig = art.ensure_sig()
                if sig not in dedup_sigs:
                    if eid == "E0_KEEP":
                        prop_opt = None
                    else:
                        prop_opt = art.to_proposal(
                            policy_id=f"CAL_{eid}",
                            snapshot=snap,
                            clf=clf_i,
                            eval_res=art.eval_res_b2,
                            eval_cnt=1,
                            t0=0.0,
                            budget_ms=35.0,
                            fallback_keep=False,
                            include_eval_seq_when_unwindowed=True,
                        )
                        if art.live_repairs:
                            prop_opt.proposed_live_repairs = dict(art.live_repairs)
                        if art.bottleneck_locks:
                            prop_opt.proposed_bottleneck_locks = dict(art.bottleneck_locks)
                    ro = rollout_48_and_optional_full(
                        item["spec"],
                        item["prefix_steps"],
                        int(item["tick"]),
                        prop_opt,
                        action_at_target=int(item["action_at_tick"]),
                        base_env_policy=str(item["source_policy"]),
                        rollout_horizon=48,
                        full_continuation=do_full,
                    )
                    dedup_sigs[sig] = ro
                ro_res = dedup_sigs[sig]
                cal_rows.append(
                    {
                        "snap_idx": idx,
                        "suite_group": grp,
                        "regime": reg,
                        "expert_id": eid,
                        "b2_raw_obj": round(float(art.eval_res_b2.objective), 4),
                        "ja_unified": round(float(art.ja_unified_score or 1e9), 4),
                        "w_surr": round(float(art.w_surr_score if art.w_surr_score is not None else 1e9), 4),
                        "j48_with_residual": float(ro_res["j48_with_residual"]),
                        "j48_incurred_delta": float(ro_res["j48_incurred_delta"]),
                        "receipt_accepted": bool(ro_res["receipt_accepted"]),
                        "full_net_obj": ro_res.get("full_net_obj"),
                    }
                )

            # Snapshot rank correlation across distinct rollout candidates
            per_s = [r for r in cal_rows if r["snap_idx"] == idx]
            y_j48 = np.array([r["j48_with_residual"] for r in per_s], dtype=np.float64)
            if len(per_s) >= 3 and np.std(y_j48) > 1e-6:
                x_b2 = np.array([r["b2_raw_obj"] for r in per_s], dtype=np.float64)
                x_ja = np.array([r["ja_unified"] for r in per_s], dtype=np.float64)
                x_ws = np.array([r["w_surr"] for r in per_s], dtype=np.float64)
                min_y = float(np.min(y_j48))

                for x_vec, rho_dst, conc_dst, reg_dst in (
                    (x_b2, rho_b2_list, concord_b2, regret_b2),
                    (x_ja, rho_ja_list, concord_ja, regret_ja),
                    (x_ws, rho_wsurr_list, concord_ws, regret_ws),
                ):
                    if np.std(x_vec) > 1e-6:
                        rho_v = spearmanr(x_vec, y_j48).statistic
                        if rho_v is not None and not np.isnan(rho_v):
                            rho_dst.append(float(rho_v))
                    pick_i = int(np.argmin(x_vec))
                    reg_dst.append(float(y_j48[pick_i] - min_y))
                    # Pairwise concordance
                    con = 0
                    tot_p = 0
                    for a_i in range(len(per_s)):
                        for b_i in range(a_i + 1, len(per_s)):
                            if abs(y_j48[a_i] - y_j48[b_i]) > 1e-6 and abs(x_vec[a_i] - x_vec[b_i]) > 1e-6:
                                tot_p += 1
                                if (y_j48[a_i] < y_j48[b_i]) == (x_vec[a_i] < x_vec[b_i]):
                                    con += 1
                    if tot_p > 0:
                        conc_dst.append(con / float(tot_p))

                if do_full:
                    y_full = np.array([float(r["full_net_obj"]) for r in per_s], dtype=np.float64)
                    if np.std(y_full) > 1e-6 and np.std(x_ja) > 1e-6:
                        rv = spearmanr(x_ja, y_full).statistic
                        if rv is not None and not np.isnan(rv):
                            full_rho_ja.append(float(rv))
                        full_regret_ja.append(float(y_full[int(np.argmin(x_ja))] - np.min(y_full)))

        if (idx + 1) % 100 == 0:
            print(f"  [Task B] Processed {idx + 1}/{n_snaps} snapshots...", flush=True)

    # Leave-One-Out expert marginal value separated cleanly into Feasibility Loss vs Cost Degradation
    loo_marginal: dict[str, dict[str, Any]] = {}
    for eid in EXPERT_IDS:
        cost_drops: list[float] = []
        feas_losses = 0
        for m in expert_scores_mat:
            all_vals = [float(v) for v in m.values() if v is not None]
            if not all_vals:
                continue
            best_all = min(all_vals)
            rem_vals = [float(v) for k, v in m.items() if k != eid and v is not None]
            if not rem_vals:
                feas_losses += 1
            else:
                cost_drops.append(max(0.0, min(rem_vals) - best_all))
        gl = sorted(expert_gen_lats[eid])
        bl = sorted(expert_b2_lats[eid])
        loo_marginal[eid] = {
            "strict_oracle_wins": win_counts[eid],
            "win_pct": round(100.0 * win_counts[eid] / max(1, n_snaps), 2),
            "near_oracle_1pct_count": near_oracle_counts[eid],
            "near_oracle_pct": round(100.0 * near_oracle_counts[eid] / max(1, n_snaps), 2),
            "oracle_wins_when_v121_suboptimal": oracle_win_when_v121_loses[eid],
            "loo_feasibility_rescue_exclusive": feas_losses,
            "loo_mean_oracle_degradation": round(float(np.mean(cost_drops)), 3) if cost_drops else 0.0,
            "loo_p95_degradation": round(float(np.percentile(cost_drops, 95)), 3) if cost_drops else 0.0,
            "valid_rate": round(expert_valid_counts[eid] / max(1, n_snaps), 4),
            "mean_gen_ms": round(float(np.mean(gl)), 3) if gl else 0.0,
            "p95_gen_ms": round(float(np.percentile(gl, 95)), 3) if gl else 0.0,
            "mean_b2_ms": round(float(np.mean(bl)), 3) if bl else 0.0,
        }

    proxy_summary = {
        "rollout48_snapshots": len(cal_indices),
        "full_terminal_snapshots": len(full_cont_indices),
        "proxies": {
            "CVCR_B2_raw": {
                "mean_spearman_rho_48": round(float(np.mean(rho_b2_list)), 4) if rho_b2_list else 0.0,
                "pairwise_concordance_48": round(float(np.mean(concord_b2)), 4) if concord_b2 else 0.0,
                "mean_top1_regret_48": round(float(np.mean(regret_b2)), 3) if regret_b2 else 0.0,
            },
            "JA_Unified": {
                "mean_spearman_rho_48": round(float(np.mean(rho_ja_list)), 4) if rho_ja_list else 0.0,
                "pairwise_concordance_48": round(float(np.mean(concord_ja)), 4) if concord_ja else 0.0,
                "mean_top1_regret_48": round(float(np.mean(regret_ja)), 3) if regret_ja else 0.0,
                "mean_spearman_rho_full_terminal": round(float(np.mean(full_rho_ja)), 4) if full_rho_ja else 0.0,
                "mean_top1_regret_full_terminal": round(float(np.mean(full_regret_ja)), 3) if full_regret_ja else 0.0,
            },
            "WindowAwareKittingSurrogate": {
                "mean_spearman_rho_48": round(float(np.mean(rho_wsurr_list)), 4) if rho_wsurr_list else 0.0,
                "pairwise_concordance_48": round(float(np.mean(concord_ws)), 4) if concord_ws else 0.0,
                "mean_top1_regret_48": round(float(np.mean(regret_ws)), 3) if regret_ws else 0.0,
            },
        },
    }

    mean_v121_ref = float(np.mean(v121_valid_refs)) if v121_valid_refs else 0.0
    mean_oracle_valid = float(np.mean(oracle_valid_refs)) if oracle_valid_refs else 0.0
    mean_gap_valid = float(np.mean(oracle_gaps_vs_v121)) if oracle_gaps_vs_v121 else 0.0
    rel_headroom_pct = 100.0 * mean_gap_valid / max(1e-6, mean_v121_ref)

    payload = {
        "schema": "if-core-v20-01.task_b.v1",
        "num_snapshots": n_snaps,
        "headroom": {
            "mutually_valid_snapshots": len(oracle_gaps_vs_v121),
            "v121_b2_infeasible_snapshots": v121_infeasible_count,
            "oracle_feasibility_rescue_count": feasibility_rescue_count,
            "oracle_feasibility_rescue_pct": round(100.0 * feasibility_rescue_count / max(1, v121_infeasible_count), 2) if v121_infeasible_count else 0.0,
            "mean_v121_best_ref": round(mean_v121_ref, 3),
            "mean_oracle_all_e0_e9": round(mean_oracle_valid, 3),
            "mean_oracle_gap_vs_v121": round(mean_gap_valid, 3),
            "rel_headroom_vs_v121_pct": round(rel_headroom_pct, 2),
            "mean_oracle_gap_vs_keep": round(float(np.mean(oracle_gaps_vs_keep)), 3) if oracle_gaps_vs_keep else 0.0,
            "pct_snapshots_beating_v121_gt10": round(
                100.0 * (sum(1 for g in oracle_gaps_vs_v121 if g > 10.0) + feasibility_rescue_count) / max(1, n_snaps), 2
            ),
            "by_suite": {
                k: {
                    "n": len(v),
                    "mean_gap_vs_v121": round(float(np.mean(v)), 3) if v else 0.0,
                    "p90_gap": round(float(np.percentile(v, 90)), 3) if v else 0.0,
                }
                for k, v in gaps_by_suite.items()
            },
            "by_regime": {
                k: {
                    "n": len(v),
                    "mean_gap_vs_v121": round(float(np.mean(v)), 3) if v else 0.0,
                }
                for k, v in gaps_by_regime.items()
            },
        },
        "expert_complementarity_and_loo": loo_marginal,
        "proxy_calibration": proxy_summary,
    }

    (OUT_AUDIT_DIR / "task_b_expert_oracle.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Export complementarity matrix CSV and proxy calibration CSV
    with (OUT_AUDIT_DIR / "task_b_complementarity_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = [
            "expert_id", "strict_oracle_wins", "win_pct", "near_oracle_pct",
            "oracle_wins_when_v121_suboptimal", "loo_feasibility_rescue_exclusive",
            "loo_mean_oracle_degradation", "loo_p95_degradation",
            "valid_rate", "mean_gen_ms", "p95_gen_ms", "mean_b2_ms",
        ]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for eid in EXPERT_IDS:
            row = dict(loo_marginal[eid])
            row["expert_id"] = eid
            w.writerow({c: row[c] for c in cols})

    if cal_rows:
        with (OUT_AUDIT_DIR / "task_b_proxy_calibration.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cal_rows[0].keys()))
            w.writeheader()
            w.writerows(cal_rows)

    md = [
        "# Task B: `V18`-`V19` Same-Snapshot Expert Complementarity & Oracle Headroom\n",
        f"- **Stratified Quarantined Snapshots Evaluated**: `{n_snaps}` (`>= 600` required; `{payload['headroom']['mutually_valid_snapshots']}` mutually B2-feasible)",
        f"- **Infeasible B2 Rescue by E0-E9**: `{feasibility_rescue_count}/{v121_infeasible_count}` (`{payload['headroom']['oracle_feasibility_rescue_pct']}%`)",
        f"- **Mean Mutually-Feasible Cost Headroom vs v1.2.1 Anchor**: `ΔJ = {payload['headroom']['mean_oracle_gap_vs_v121']:.2f}` (`{rel_headroom_pct:.2f}%`, `{payload['headroom']['pct_snapshots_beating_v121_gt10']}%` snapshots gain > 10 J or rescue infeasibility)",
        "\n## 1. Expert Complementarity & Leave-One-Out Value (`E0-E9`)\n",
        "| Expert | Strict Wins (%) | Near-Oracle <=1% (%) | Wins when v1.2.1 Subopt | LOO Feas Loss | LOO Cost Drop (Mean / P95) | Valid Rate | Gen Mean / P95 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for eid in EXPERT_IDS:
        d = loo_marginal[eid]
        md.append(
            f"| {eid} | {d['strict_oracle_wins']} ({d['win_pct']:.1f}%) | {d['near_oracle_pct']:.1f}% | "
            f"{d['oracle_wins_when_v121_suboptimal']} | {d['loo_feasibility_rescue_exclusive']} | "
            f"{d['loo_mean_oracle_degradation']:.2f} / {d['loo_p95_degradation']:.2f} | "
            f"{d['valid_rate']:.3f} | {d['mean_gen_ms']:.2f} / {d['p95_gen_ms']:.2f} |"
        )
    md.append("\n## 2. Cloned Closed-Loop Rollout Proxy Calibration (`48-tick` & `Terminal`)\n")
    for p_name, pd in proxy_summary["proxies"].items():
        md.append(f"- **{p_name}**: Spearman ρ(48t) = `{pd['mean_spearman_rho_48']}`, Concordance = `{pd['pairwise_concordance_48']}`, Mean Top-1 Regret(48t) = `{pd['mean_top1_regret_48']}`")
    (OUT_AUDIT_DIR / "task_b_expert_oracle.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    run_task_b()
