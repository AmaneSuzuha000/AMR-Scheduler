"""EXP-08 (Revised Fair Audit): Dual-Engine Synergy (`AHG-RL` x `CVCR-B2`) & 3-Level Isolated Cache Audit.

Addresses all strict fairness & semantic equivalence requirements:
  1. Exact Candidate Semantic Parity: `A1`, `B1`, `Seeded Raw-Context B2`, and `Seeded Seq-Diff B2`
     all evaluate the EXACT same `(cand.assign, cand.op_prio, cand.bays, cand.crews, cand.task_sequences)`,
     matching the committed `best_op_prio` and `best_seq` 100%.
  2. Clean 3-Level Isolation of B2 Cache Invalidation (Part A):
     - Level 1 (`Unseeded B2`): `set_incumbent()` NOT called after `keep_eval`.
     - Level 2 (`Seeded Raw-Context B2`): `set_incumbent()` IS called, strict `prio_key` & raw dict match (`seq_diff_b2=False`).
     - Level 3 (`Seeded Seq-Diff B2`): `set_incumbent()` IS called, per-AMR effective task-sequence diffing (`seq_diff_b2=True`).
     Also compares Standard Random Sampling (`focus_frac=1.0`) vs `A1`-Focused Sampling (`focus_frac=0.60`).
  3. Zero Hardcoded Telemetry Defaults (Part B):
     All policies (`rule_v10`, `V18_EA_JOINT`, `V19_Full`, `V19_Staged`, `AHG-RL`) report 100% runtime-measured
     counters from `CVCREvaluator.get_global_telemetry()` (or `None` / `"N/A"` when inapplicable).
  4. Full 120-Episode Unseen Blind Holdout Verification of Default Registered `V19_ADAPTIVE_RL` (`AHG-RL + CVCR-B2`).

Writes outputs to `out/v19_scientific_audit/exp08_learning_b2_synergy.json` and `.md`.
"""

from __future__ import annotations

import os
import platform

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

from scipy.stats import wilcoxon

if sys.platform == "win32" and getattr(platform, "_wmi", None) is not None:
    platform._wmi = None  # type: ignore[attr-defined]

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.baselines.policy_registry import run_discrete_episode
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.rl import (
    AMRReschedulingEnv,
    RuleV10PortfolioPolicy,
    compute_expert_v10_action_mask,
    compute_expert_v9_state,
)
from amr_sched.rl.v19_hetero_constructor import (
    V19AdaptiveCouplingProposer,
    V19HeteroConstructorProposer,
    _load_v19_checkpoint,
    compute_candidate_amr_footprint,
    decode_v19_candidate,
)
from amr_sched.rl.v19_hetero_graph import apply_graph_ablation, extract_v19_hetero_graph
from experiments.train_v19_bc import (
    CKPT_DIR,
    _build_4topo_episode_spec,
    _build_workshop_episode_spec,
)
from tools.benchmarks.compare_workshop_algorithms import (
    REGIMES as WORKSHOP_REGIMES,
    build_track1_scenario,
    build_track2_scenario,
)
from experiments.v18_confirmation_scenarios import LAYOUTS as TOPO4_LAYOUTS, REGIMES as TOPO4_REGIMES

OUT_DIR = ROOT / "out" / "v19_scientific_audit"
WORKSHOP_SEEDS: tuple[int, ...] = (901, 902, 903)
BLIND_4TOPO_SEEDS: tuple[int, ...] = (17101, 17102, 17201, 17202, 17301, 17302)
BLIND_WORKSHOP_SEEDS: tuple[int, ...] = (901, 902, 903, 904, 905, 906)


def _build_ws_spec(track: str, seed: int, regime: str) -> tuple[Any, Any, int]:
    return build_track1_scenario(seed, regime) if track == "track1" else build_track2_scenario(seed, regime)


def _run_part_a_3level_cache_isolation() -> dict[str, Any]:
    """Audit 3-level isolated B2 cache hit causes under 100% identical candidate semantics."""
    full_net, _, _, _ = _load_v19_checkpoint(
        CKPT_DIR / "v19_hetero_rl.npz", backend="numpy", action_stage="A1234", ablation_mode="full"
    )
    staged_net, _, _, _ = _load_v19_checkpoint(
        CKPT_DIR / "v19_staged_rl.npz", backend="numpy", action_stage="A1234", ablation_mode="staged"
    )
    a123_net, _, _, _ = _load_v19_checkpoint(
        CKPT_DIR / "v19_hetero_bc_a123.npz", backend="numpy", action_stage="A123", ablation_mode="full"
    )

    snapshots: list[tuple[str, Any, Any]] = []
    for track in ("track1", "track2"):
        for regime in ("kitting_windows", "execution_blockage", "multi_wave_surge"):
            sc_tuple, p2_cfg, max_steps = _build_ws_spec(track, 901, regime)
            env = AMRReschedulingEnv(
                seed=901,
                max_steps=max_steps,
                use_rust=True,
                loaded_repair_mode="haven_v12",
                loaded_repair_threshold=20,
                execution_perturbation=p2_cfg,
                execution_aware=True,
                p4_choke_robust=True,
                expert_v10_portfolio=True,
            )
            ctrl = RuleV10PortfolioPolicy(epoch_interval=16)
            ctrl.reset()
            _, info = env.reset(seed=901, options={"scenario_tuple": sc_tuple})
            for _ in range(80):
                feats = compute_expert_v9_state(
                    env.sim,
                    env.tracker,
                    last_replan_tick=env.last_replan_tick,
                    recent_conflict_ticks=env.recent_conflict_ticks,
                    decision_logs=env.decision_logs,
                )
                s_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
                act = ctrl.select_action(feats, s_info, sim=env.sim)
                _, _, tm, tr, info = env.step(act)
                if env.current_step in (12, 24, 36, 48):
                    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
                    clf = classify_commitments(snap)
                    g = extract_v19_hetero_graph(snap, env.sim.network, clf=clf)
                    if len(g.rel_tids) >= 2:
                        snapshots.append((f"{track}:{regime}:t{env.current_step}", snap, env.sim.network))
                if tm or tr:
                    break

    modes_audit: dict[str, dict[str, Any]] = {}
    configs = [
        ("V19_Full_A1234_StdSample", full_net, "full", 1.0),
        ("V19_Full_A1234_A1Focused", full_net, "full", 0.60),
        ("V19_Staged_A1234_A1Focused", staged_net, "staged", 0.60),
        ("V19_Full_A123_NoA4", a123_net, "full", 1.0),
    ]
    for mode_label, net_obj, abl_mode, focus_frac in configs:
        changed_ratios: list[float] = []
        reused_ratios: list[float] = []
        assign_ratios: list[float] = []
        bay_ratios: list[float] = []
        order_ratios: list[float] = []
        unseeded_b2_hits = 0
        seeded_raw_b2_hits = 0
        seeded_seqdiff_b2_hits = 0
        b2_zero_hits = 0
        b2_fallbacks = 0
        total_cands = 0
        lats_a1: list[float] = []
        lats_b1: list[float] = []
        lats_b2_seqdiff: list[float] = []
        obj_abs_diffs: list[float] = []

        for idx, (_tag, snap, net) in enumerate(snapshots):
            clf = classify_commitments(snap)
            raw_g = extract_v19_hetero_graph(snap, net, clf=clf)
            g = apply_graph_ablation(raw_g, abl_mode)
            base_op_prio = {
                op.op_id: max(int(snap.tick), int(op.due_step) - int(op.proc_steps))
                for op in snap.operations
            }

            # Level 1: Unseeded B2 (missing set_incumbent)
            ev_unseeded = CVCREvaluator(net, snap, use_rust=True, mode="B2", seq_diff_b2=False)
            ev_unseeded.evaluate(g.base_assign, op_priority=base_op_prio, op_bay_overrides=g.base_bays, op_crew_overrides=g.base_crews)

            # Level 2: Seeded Raw-Context B2 (set_incumbent IS called, seq_diff_b2=False)
            ev_seeded_raw = CVCREvaluator(net, snap, use_rust=True, mode="B2", seq_diff_b2=False)
            keep_raw = ev_seeded_raw.evaluate(g.base_assign, op_priority=base_op_prio, op_bay_overrides=g.base_bays, op_crew_overrides=g.base_crews)
            ev_seeded_raw.set_incumbent(g.base_assign, keep_raw, op_priority=base_op_prio, op_bay_overrides=g.base_bays)

            # Level 3: Seeded Seq-Diff B2 (set_incumbent IS called, seq_diff_b2=True)
            ev_seeded_seq = CVCREvaluator(net, snap, use_rust=True, mode="B2", seq_diff_b2=True)
            keep_seq = ev_seeded_seq.evaluate(g.base_assign, op_priority=base_op_prio, op_bay_overrides=g.base_bays, op_crew_overrides=g.base_crews)
            ev_seeded_seq.set_incumbent(g.base_assign, keep_seq, op_priority=base_op_prio, op_bay_overrides=g.base_bays)

            # Reference B1 and A1 (all evaluating IDENTICAL cand.op_prio and cand.task_sequences)
            ev_b1 = CVCREvaluator(net, snap, use_rust=True, mode="B1")
            ev_b1.evaluate(g.base_assign, op_priority=base_op_prio, op_bay_overrides=g.base_bays, op_crew_overrides=g.base_crews)

            ev_a1 = CVCREvaluator(net, snap, use_rust=True, mode="A1")
            ev_a1.evaluate(g.base_assign, op_priority=base_op_prio, op_bay_overrides=g.base_bays, op_crew_overrides=g.base_crews)

            np_rng = np.random.default_rng(700 + idx)
            for k in range(3):
                cand = decode_v19_candidate(
                    net_obj,
                    g,
                    snap,
                    greedy=(k == 0),
                    np_rng=np_rng,
                    tau=0.45,
                    focus_sample_frac=(1.0 if k == 0 else focus_frac),
                )
                fp = compute_candidate_amr_footprint(
                    snap,
                    g.base_assign,
                    g.base_bays,
                    keep_seq.robot_order_seq,
                    cand.assign,
                    cand.bays,
                    cand.task_sequences,
                )
                n_r = float(fp["n_robots"])
                total_cands += 1
                changed_ratios.append(float(fp["changed_amr_ratio"]))
                reused_ratios.append(float(fp["reused_amr_ratio"]))
                assign_ratios.append(len(fp["assign_changed_rids"]) / n_r)
                bay_ratios.append(len(fp["bay_changed_rids"]) / n_r)
                order_ratios.append(len(fp["order_changed_rids"]) / n_r)
                if fp["is_b2_zero_route_hit"]:
                    b2_zero_hits += 1
                if fp["is_b2_full_fallback"]:
                    b2_fallbacks += 1

                # IDENTICAL candidate payload across all 5 evaluators!
                eval_kwargs = dict(
                    op_priority=cand.op_prio,
                    op_bay_overrides=cand.bays,
                    op_crew_overrides=cand.crews,
                    task_order_override=cand.task_sequences,
                )

                h1 = ev_unseeded.b2_inc_hits
                ev_unseeded.evaluate(cand.assign, **eval_kwargs)
                if ev_unseeded.b2_inc_hits > h1:
                    unseeded_b2_hits += 1

                h2 = ev_seeded_raw.b2_inc_hits
                ev_seeded_raw.evaluate(cand.assign, **eval_kwargs)
                if ev_seeded_raw.b2_inc_hits > h2:
                    seeded_raw_b2_hits += 1

                h3 = ev_seeded_seq.b2_inc_hits
                t_s0 = time.perf_counter()
                res_b2 = ev_seeded_seq.evaluate(cand.assign, **eval_kwargs)
                lats_b2_seqdiff.append((time.perf_counter() - t_s0) * 1000.0)
                if ev_seeded_seq.b2_inc_hits > h3:
                    seeded_seqdiff_b2_hits += 1

                t_b1 = time.perf_counter()
                res_b1 = ev_b1.evaluate(cand.assign, **eval_kwargs)
                lats_b1.append((time.perf_counter() - t_b1) * 1000.0)
                if res_b1.valid and res_b2.valid:
                    obj_abs_diffs.append(abs(float(res_b1.objective) - float(res_b2.objective)))

                t_a1 = time.perf_counter()
                ev_a1.evaluate(cand.assign, **eval_kwargs)
                lats_a1.append((time.perf_counter() - t_a1) * 1000.0)

        modes_audit[mode_label] = {
            "total_candidates_audited": total_cands,
            "mean_changed_amr_ratio_pct": round(100.0 * float(np.mean(changed_ratios)), 2),
            "mean_reused_amr_ratio_pct": round(100.0 * float(np.mean(reused_ratios)), 2),
            "mean_a3_assign_changed_pct": round(100.0 * float(np.mean(assign_ratios)), 2),
            "mean_a2_bay_changed_pct": round(100.0 * float(np.mean(bay_ratios)), 2),
            "mean_a4_order_changed_pct": round(100.0 * float(np.mean(order_ratios)), 2),
            "level1_unseeded_b2_hit_pct": round(100.0 * unseeded_b2_hits / max(1, total_cands), 2),
            "level2_seeded_raw_ctx_b2_hit_pct": round(100.0 * seeded_raw_b2_hits / max(1, total_cands), 2),
            "level3_seeded_seqdiff_b2_hit_pct": round(100.0 * seeded_seqdiff_b2_hits / max(1, total_cands), 2),
            "b2_zero_route_hit_rate_pct": round(100.0 * b2_zero_hits / max(1, total_cands), 2),
            "b2_full_fleet_fallback_rate_pct": round(100.0 * b2_fallbacks / max(1, total_cands), 2),
            "mean_eval_lat_a1_ms": round(float(np.mean(lats_a1)), 4),
            "mean_eval_lat_b1_ms": round(float(np.mean(lats_b1)), 4),
            "mean_eval_lat_b2_seqdiff_ms": round(float(np.mean(lats_b2_seqdiff)), 4),
            "speedup_b2_vs_a1": round(float(np.mean(lats_a1)) / max(1e-6, float(np.mean(lats_b2_seqdiff))), 2),
            "speedup_b2_vs_b1": round(float(np.mean(lats_b1)) / max(1e-6, float(np.mean(lats_b2_seqdiff))), 2),
            "median_b1_vs_b2_obj_diff": round(float(np.median(obj_abs_diffs)) if obj_abs_diffs else 0.0, 2),
        }

    # Local 1-opt/swap baseline on the exact same snapshots
    loc_changed: list[float] = []
    loc_reused: list[float] = []
    loc_l1 = 0
    loc_l2 = 0
    loc_l3 = 0
    loc_total = 0
    loc_lats_b1: list[float] = []
    loc_lats_b2: list[float] = []
    for _tag, snap, net in snapshots:
        clf = classify_commitments(snap)
        g = extract_v19_hetero_graph(snap, net, clf=clf)
        base_op_prio = {op.op_id: max(int(snap.tick), int(op.due_step) - int(op.proc_steps)) for op in snap.operations}
        ev_u = CVCREvaluator(net, snap, use_rust=True, mode="B2", seq_diff_b2=False)
        ev_u.evaluate(g.base_assign, op_priority=base_op_prio, op_bay_overrides=g.base_bays, op_crew_overrides=g.base_crews)
        ev_r = CVCREvaluator(net, snap, use_rust=True, mode="B2", seq_diff_b2=False)
        kr = ev_r.evaluate(g.base_assign, op_priority=base_op_prio, op_bay_overrides=g.base_bays, op_crew_overrides=g.base_crews)
        ev_r.set_incumbent(g.base_assign, kr, op_priority=base_op_prio, op_bay_overrides=g.base_bays)
        ev_s = CVCREvaluator(net, snap, use_rust=True, mode="B2", seq_diff_b2=True)
        ks = ev_s.evaluate(g.base_assign, op_priority=base_op_prio, op_bay_overrides=g.base_bays, op_crew_overrides=g.base_crews)
        ev_s.set_incumbent(g.base_assign, ks, op_priority=base_op_prio, op_bay_overrides=g.base_bays)
        ev_b1 = CVCREvaluator(net, snap, use_rust=True, mode="B1")
        ev_b1.evaluate(g.base_assign, op_priority=base_op_prio, op_bay_overrides=g.base_bays, op_crew_overrides=g.base_crews)
        rel, agvs = list(g.rel_tids), list(g.agv_ids)
        if not rel or len(agvs) < 2:
            continue
        for k in range(min(3, len(rel))):
            tid = rel[k]
            cur_a = g.base_assign.get(tid, agvs[0])
            alt_a = next((a for a in agvs if a != cur_a), cur_a)
            c_assign = dict(g.base_assign)
            c_assign[tid] = alt_a
            fp = compute_candidate_amr_footprint(snap, g.base_assign, g.base_bays, ks.robot_order_seq, c_assign, g.base_bays, None)
            loc_total += 1
            loc_changed.append(float(fp["changed_amr_ratio"]))
            loc_reused.append(float(fp["reused_amr_ratio"]))
            kw = dict(op_priority=base_op_prio, op_bay_overrides=g.base_bays, op_crew_overrides=g.base_crews)
            hu = ev_u.b2_inc_hits
            ev_u.evaluate(c_assign, **kw)
            loc_l1 += int(ev_u.b2_inc_hits > hu)
            hr = ev_r.b2_inc_hits
            ev_r.evaluate(c_assign, **kw)
            loc_l2 += int(ev_r.b2_inc_hits > hr)
            hs = ev_s.b2_inc_hits
            t0 = time.perf_counter()
            ev_s.evaluate(c_assign, **kw)
            loc_lats_b2.append((time.perf_counter() - t0) * 1000.0)
            loc_l3 += int(ev_s.b2_inc_hits > hs)
            tb1 = time.perf_counter()
            ev_b1.evaluate(c_assign, **kw)
            loc_lats_b1.append((time.perf_counter() - tb1) * 1000.0)

    modes_audit["Local_1Opt_Swap_CVCR_B2"] = {
        "total_candidates_audited": loc_total,
        "mean_changed_amr_ratio_pct": round(100.0 * float(np.mean(loc_changed)), 2),
        "mean_reused_amr_ratio_pct": round(100.0 * float(np.mean(loc_reused)), 2),
        "mean_a3_assign_changed_pct": round(100.0 * float(np.mean(loc_changed)), 2),
        "mean_a2_bay_changed_pct": 0.0,
        "mean_a4_order_changed_pct": 0.0,
        "level1_unseeded_b2_hit_pct": round(100.0 * loc_l1 / max(1, loc_total), 2),
        "level2_seeded_raw_ctx_b2_hit_pct": round(100.0 * loc_l2 / max(1, loc_total), 2),
        "level3_seeded_seqdiff_b2_hit_pct": round(100.0 * loc_l3 / max(1, loc_total), 2),
        "b2_zero_route_hit_rate_pct": 0.0,
        "b2_full_fleet_fallback_rate_pct": round(100.0 * (loc_total - loc_l3) / max(1, loc_total), 2),
        "mean_eval_lat_b1_ms": round(float(np.mean(loc_lats_b1)), 4),
        "mean_eval_lat_b2_seqdiff_ms": round(float(np.mean(loc_lats_b2)), 4),
        "speedup_b2_vs_b1": round(float(np.mean(loc_lats_b1)) / max(1e-6, float(np.mean(loc_lats_b2))), 2),
    }
    return modes_audit


def _run_part_b_fair_closed_loop_synergy() -> dict[str, Any]:
    """Run fair 4-way closed-loop synergy with identical candidate semantics and 100% real telemetry."""
    episodes: list[tuple[str, Any, Any, int, int]] = []
    for track in ("track1", "track2"):
        for regime in WORKSHOP_REGIMES:
            for seed in WORKSHOP_SEEDS:
                sc_tuple, p2_cfg, max_steps = _build_ws_spec(track, seed, regime)
                episodes.append((f"{track}:{regime}:{seed}", sc_tuple, p2_cfg, max_steps, seed))

    combos = [
        ("Rule_V10_CVCR_B2", "rule_v10", "B2"),
        ("V19_Full_B1_FullRebuild", "v19_full", "B1"),
        ("V19_Full_CVCR_B2_SeqDiff", "v19_full", "B2_SEEDED"),
        ("V19_Staged_CVCR_B2_SeqDiff", "v19_staged", "B2_SEEDED"),
        ("AHG_RL_B1_FullRebuild", "ahg_rl", "B1"),
        ("AHG_RL_CVCR_B2_SeqDiff", "ahg_rl", "B2_SEEDED"),
    ]

    results: dict[str, Any] = {}
    for label, kind, ev_mode in combos:
        CVCREvaluator.reset_global_telemetry()
        net_objs: list[float] = []
        cmaxs: list[float] = []
        tards: list[float] = []
        task_comps: list[float] = []
        op_comps: list[float] = []
        proposer_changed_ratios: list[float] = []
        cands_total = 0

        for ep_key, sc_tuple, p2_cfg, max_steps, seed in episodes:
            if kind == "rule_v10":
                proposer = None
            elif kind == "v19_full":
                proposer = V19HeteroConstructorProposer(
                    CKPT_DIR / "v19_hetero_rl.npz",
                    policy_id=label,
                    action_stage="A1234",
                    ablation_mode="full",
                    backend="numpy",
                    use_rust=True,
                    seed=seed,
                    sample_candidates=3,
                    evaluator_mode=ev_mode,
                )
            elif kind == "v19_staged":
                proposer = V19HeteroConstructorProposer(
                    CKPT_DIR / "v19_staged_rl.npz",
                    policy_id=label,
                    action_stage="A1234",
                    ablation_mode="staged",
                    backend="numpy",
                    use_rust=True,
                    seed=seed,
                    sample_candidates=3,
                    evaluator_mode=ev_mode,
                )
            else:
                proposer = V19AdaptiveCouplingProposer(
                    CKPT_DIR / "v19_hetero_rl.npz",
                    CKPT_DIR / "v19_staged_rl.npz",
                    policy_id=label,
                    mode="bandit",
                    backend="numpy",
                    use_rust=True,
                    seed=seed,
                    sample_candidates=3,
                    evaluator_mode=ev_mode,
                )

            sc_copy = copy.deepcopy(sc_tuple)
            p2_copy = copy.deepcopy(p2_cfg)
            env = AMRReschedulingEnv(
                seed=seed,
                max_steps=max_steps,
                use_rust=True,
                allow_kitting_shortcut=False,
                pure_physical_reward=True,
                real_compute_window_coupling=False,
                record_trajectory=False,
                loaded_repair_mode="haven_v12",
                loaded_repair_threshold=20,
                execution_perturbation=p2_copy,
                execution_aware=True,
                p4_choke_robust=True,
                expert_v10_portfolio=True,
                external_proposer=proposer,
                reward_mode="net_obj",
            )
            ctrl = RuleV10PortfolioPolicy(epoch_interval=16)
            ctrl.reset()
            _, info = env.reset(seed=seed, options={"scenario_tuple": sc_copy})
            tm = False
            tr = False
            while not (tm or tr):
                feats = compute_expert_v9_state(
                    env.sim,
                    env.tracker,
                    last_replan_tick=env.last_replan_tick,
                    recent_conflict_ticks=env.recent_conflict_ticks,
                    decision_logs=env.decision_logs,
                )
                s_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
                act = ctrl.select_action(feats, s_info, sim=env.sim)
                _, _, tm, tr, info = env.step(act)

            m = env.compute_episode_metrics()
            aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=False)
            assert aud.pass_audit, f"Audit failed on {label} {ep_key}: {aud.issues}"
            net_objs.append(float(m["net_obj"]))
            cmaxs.append(float(m["cmax"]))
            tards.append(float(m["total_tardiness"]))
            task_comps.append(float(m["task_completion_rate"]))
            op_comps.append(float(m["production_completion_rate"]))
            if proposer is not None:
                cands_total += int(proposer.total_cands_evaluated)
                proposer_changed_ratios.extend(proposer.total_changed_ratios)

        gt = CVCREvaluator.get_global_telemetry()
        b2_cands = int(gt["b2_cand_evals"])
        b2_hits = int(gt["b2_inc_hits"])
        b2_hit_pct = round(100.0 * b2_hits / b2_cands, 2) if b2_cands > 0 else None
        if gt["mean_changed_amr_ratio"] is not None:
            reused_pct = round(100.0 * (1.0 - float(gt["mean_changed_amr_ratio"])), 2)
        elif proposer_changed_ratios:
            reused_pct = round(100.0 * (1.0 - float(np.mean(proposer_changed_ratios))), 2)
        else:
            reused_pct = None

        results[label] = {
            "n_episodes": len(episodes),
            "mean_net_obj": round(float(np.mean(net_objs)), 2),
            "median_net_obj": round(float(np.median(net_objs)), 2),
            "mean_cmax": round(float(np.mean(cmaxs)), 2),
            "mean_tardiness": round(float(np.mean(tards)), 2),
            "task_comp_pct": round(100.0 * float(np.mean(task_comps)), 2),
            "op_comp_pct": round(100.0 * float(np.mean(op_comps)), 2),
            "total_eval_calls": int(gt["calls"]),
            "b2_cand_evals": b2_cands,
            "b2_inc_hits": b2_hits,
            "b2_zero_route_hits": int(gt["b2_zero_route_hits"]),
            "b2_fallbacks": int(gt["b2_fallbacks"]),
            "b2_hit_rate_pct": b2_hit_pct,
            "mean_reused_amr_ratio_pct": reused_pct,
            "mean_eval_lat_ms": round(float(gt["mean_eval_lat_ms"]), 4) if gt["mean_eval_lat_ms"] is not None else None,
        }
        b2_s = f"{b2_hit_pct:5.1f}%" if b2_hit_pct is not None else "  N/A "
        reu_s = f"{reused_pct:5.1f}%" if reused_pct is not None else "  N/A "
        lat_s = f"{results[label]['mean_eval_lat_ms']:.3f}ms" if results[label]["mean_eval_lat_ms"] is not None else "N/A"
        print(
            f"[EXP-08 Part B] {label:28s} | NetObj={results[label]['mean_net_obj']:8.2f} "
            f"| Med={results[label]['median_net_obj']:7.1f} | B2Hit={b2_s} | Reused={reu_s} | Lat={lat_s}",
            flush=True,
        )

    return results


def _paired_bootstrap_ci(deltas: np.ndarray, n_boot: int = 10_000, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


def _compute_paired_stats_vs_blind_holdout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute exact 120-episode paired Wilcoxon p-values & 95% Bootstrap CIs vs all blind holdout baselines."""
    bh_path = ROOT / "out" / "g4_blind_holdout" / "blind_holdout_report.json"
    if not bh_path.is_file():
        return {}
    bh = json.loads(bh_path.read_text(encoding="utf-8"))
    core_by_key = {str(r["episode_key"]): r for r in rows}
    paired_out: dict[str, Any] = {}
    for pol, bl_rows in bh.get("rows", {}).items():
        bl_by_key = {str(r["episode_key"]): r for r in bl_rows}
        common_keys = [k for k in core_by_key if k in bl_by_key]
        if not common_keys:
            continue
        core_vals = np.array([float(core_by_key[k]["net_obj"]) for k in common_keys], dtype=np.float64)
        bl_vals = np.array([float(bl_by_key[k]["net_obj"]) for k in common_keys], dtype=np.float64)
        deltas = core_vals - bl_vals  # negative = CoRe-Sched lower cost (better)
        wins = int(np.sum(deltas < -1e-6))
        losses = int(np.sum(deltas > 1e-6))
        ties = int(len(deltas) - wins - losses)
        nonzero = deltas[np.abs(deltas) > 1e-9]
        if len(nonzero) >= 1:
            stat, pval = wilcoxon(nonzero, alternative="two-sided")
            w_stat, w_p = float(stat), float(pval)
        else:
            w_stat, w_p = 0.0, 1.0
        bl_mean = float(np.mean(bl_vals))
        core_mean = float(np.mean(core_vals))
        pct_change = 100.0 * (core_mean - bl_mean) / max(1e-9, bl_mean)
        # Also compute paired delta on mutually 100%-completed episodes (H2 efficiency vs H1 reliability)
        both_comp_mask = np.array(
            [
                (core_by_key[k]["task_comp"] >= 0.9999 and core_by_key[k]["prod_comp"] >= 0.9999)
                and (bl_by_key[k]["task_comp"] >= 0.9999 and bl_by_key[k]["prod_comp"] >= 0.9999)
                for k in common_keys
            ],
            dtype=bool,
        )
        h2_deltas = deltas[both_comp_mask]
        h2_p = float(wilcoxon(h2_deltas[np.abs(h2_deltas) > 1e-9], alternative="two-sided")[1]) if np.sum(np.abs(h2_deltas) > 1e-9) >= 1 else 1.0
        paired_out[pol] = {
            "n_paired": len(common_keys),
            "core_mean_net_obj": round(core_mean, 2),
            "baseline_mean_net_obj": round(bl_mean, 2),
            "paired_delta_mean": round(float(np.mean(deltas)), 2),
            "paired_delta_median": round(float(np.median(deltas)), 2),
            "pct_change_vs_baseline": round(float(pct_change), 2),
            "bootstrap_95ci": _paired_bootstrap_ci(deltas),
            "wilcoxon_stat": round(w_stat, 2),
            "wilcoxon_p": w_p,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "n_both_100pct_completed": int(np.sum(both_comp_mask)),
            "h2_paired_delta_mean": round(float(np.mean(h2_deltas)), 2) if len(h2_deltas) > 0 else None,
            "h2_wilcoxon_p": h2_p if len(h2_deltas) > 0 else None,
        }
    return paired_out


def _run_part_c_120ep_registered_ahg_rl() -> dict[str, Any]:
    """Verify the default registered `V19_ADAPTIVE_RL` (`CoRe-Sched: AHG-RL + CVCR-IR Seq-Diff`) on all 120 blind holdout episodes."""
    specs: list[dict[str, Any]] = []
    for s in BLIND_4TOPO_SEEDS:
        for lay in TOPO4_LAYOUTS:
            for reg in TOPO4_REGIMES:
                sp = _build_4topo_episode_spec(lay, s, reg)
                sp["split"] = "blind_4topo"
                specs.append(sp)
    for s in BLIND_WORKSHOP_SEEDS:
        for tr in ("track1", "track2"):
            for reg in WORKSHOP_REGIMES:
                sp = _build_workshop_episode_spec(tr, s, reg)
                sp["split"] = f"blind_{tr}"
                specs.append(sp)

    CVCREvaluator.reset_global_telemetry()
    rows: list[dict[str, Any]] = []
    for sp in specs:
        rec = run_discrete_episode(
            "V19_ADAPTIVE_RL",
            copy.deepcopy(sp["scenario_tuple"]),
            seed=int(sp["seed"]),
            max_steps=int(sp["max_steps"]),
            use_rust=True,
            p2_cfg=copy.deepcopy(sp["p2_cfg"]),
            clock_mode="wall",
            episode_key=sp["episode_key"],
            backend="numpy",
        )
        rec["split"] = sp["split"]
        assert rec["audit_passed"], f"Audit failed on {sp['episode_key']}"
        rows.append(rec)

    gt = CVCREvaluator.get_global_telemetry()
    b2_cands = int(gt["b2_cand_evals"])
    b2_hits = int(gt["b2_inc_hits"])
    by_split = {}
    for sp_k in ("blind_4topo", "blind_track1", "blind_track2"):
        sub = [r for r in rows if r["split"] == sp_k]
        by_split[sp_k] = round(float(np.mean([r["net_obj"] for r in sub])), 2)

    n_comp = sum(1 for r in rows if r["task_comp"] >= 0.9999 and r["prod_comp"] >= 0.9999)
    paired_stats = _compute_paired_stats_vs_blind_holdout(rows)
    compact_rows = [
        {
            "episode_key": str(r["episode_key"]),
            "split": str(r["split"]),
            "net_obj": round(float(r["net_obj"]), 4),
            "cmax": round(float(r["cmax"]), 4),
            "task_tardiness_cost": round(float(r["task_tardiness_cost"]), 4),
            "op_tardiness_cost": round(float(r["op_tardiness_cost"]), 4),
            "unfinished_task_penalty": round(float(r["unfinished_task_penalty"]), 4),
            "unfinished_op_penalty": round(float(r["unfinished_op_penalty"]), 4),
            "churn_cost": round(float(r["churn_cost"]), 4),
            "compute_cost": round(float(r["compute_cost"]), 4),
            "task_comp": round(float(r["task_comp"]), 6),
            "prod_comp": round(float(r["prod_comp"]), 6),
            "mean_decision_ms": round(float(r["mean_decision_ms"]), 3),
            "decision_ms_p95": round(float(r["decision_ms_p95"]), 3),
            "audit_passed": bool(r["audit_passed"]),
        }
        for r in rows
    ]
    summary = {
        "policy": "CoRe-Sched (V19_ADAPTIVE_RL: AHG-RL + CVCR-IR Seq-Diff)",
        "num_episodes": len(rows),
        "num_100pct_completed": n_comp,
        "audit_pass_rate": 1.0,
        "mean_net_obj": round(float(np.mean([r["net_obj"] for r in rows])), 2),
        "median_net_obj": round(float(np.median([r["net_obj"] for r in rows])), 2),
        "iqr_net_obj": round(float(np.percentile([r["net_obj"] for r in rows], 75) - np.percentile([r["net_obj"] for r in rows], 25)), 2),
        "p90_net_obj": round(float(np.percentile([r["net_obj"] for r in rows], 90)), 2),
        "cvar10_worst_net_obj": round(float(np.mean(sorted([r["net_obj"] for r in rows], reverse=True)[:12])), 2),
        "mean_cmax": round(float(np.mean([r["cmax"] for r in rows])), 2),
        "mean_task_tardiness_cost": round(float(np.mean([r["task_tardiness_cost"] for r in rows])), 2),
        "mean_op_tardiness_cost": round(float(np.mean([r["op_tardiness_cost"] for r in rows])), 2),
        "mean_tardiness": round(float(np.mean([0.5 * (r["task_tardiness_cost"] + r["op_tardiness_cost"]) for r in rows])), 2),
        "mean_churn_cost": round(float(np.mean([r["churn_cost"] for r in rows])), 2),
        "mean_compute_cost": round(float(np.mean([r["compute_cost"] for r in rows])), 2),
        "mean_task_comp_pct": round(100.0 * float(np.mean([r["task_comp"] for r in rows])), 2),
        "mean_prod_comp_pct": round(100.0 * float(np.mean([r["prod_comp"] for r in rows])), 2),
        "unfinished_penalty_total": round(float(np.mean([r["unfinished_task_penalty"] + r["unfinished_op_penalty"] for r in rows])), 2),
        "split_mean_net_obj": by_split,
        "b2_cand_evals": b2_cands,
        "b2_inc_hits": b2_hits,
        "b2_hit_rate_pct": round(100.0 * b2_hits / b2_cands, 2) if b2_cands > 0 else None,
        "mean_reused_amr_ratio_pct": round(100.0 * (1.0 - float(gt["mean_changed_amr_ratio"])), 2) if gt["mean_changed_amr_ratio"] is not None else None,
        "mean_eval_lat_ms": round(float(gt["mean_eval_lat_ms"]), 4) if gt["mean_eval_lat_ms"] is not None else None,
        "paired_vs_blind_holdout": paired_stats,
        "rows": compact_rows,
    }
    print(
        f"[EXP-08 Part C 120-Ep] CoRe-Sched | NetObj={summary['mean_net_obj']:.2f} "
        f"| Comp={n_comp}/120 ({summary['mean_task_comp_pct']:.2f}%/{summary['mean_prod_comp_pct']:.2f}%) "
        f"| B2Hit={summary['b2_hit_rate_pct']}% | Reused={summary['mean_reused_amr_ratio_pct']}%",
        flush=True,
    )
    for pol, ps in paired_stats.items():
        print(
            f"  vs {pol:20s}: delta={ps['paired_delta_mean']:+8.2f} ({ps['pct_change_vs_baseline']:+6.2f}%) "
            f"95%CI={ps['bootstrap_95ci']} W/T/L={ps['wins']}/{ps['ties']}/{ps['losses']} p={ps['wilcoxon_p']:.4g}",
            flush=True,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXP-08 dual-engine synergy & 120-ep CoRe-Sched verification.")
    parser.add_argument("--part-c-only", action="store_true", help="Reuse existing Part A & Part B JSON and refresh Part C + paired Wilcoxon stats.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "exp08_learning_b2_synergy.json"
    t0 = time.perf_counter()
    if args.part_c_only and json_path.is_file():
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        part_a = existing["part_a_3level_isolated_cache_audit"]
        part_b = existing["part_b_fair_closed_loop_synergy"]
    else:
        print("[EXP-08] Running Part A: 3-Level Isolated B2 Cache Audit (Identical Semantics)...", flush=True)
        part_a = _run_part_a_3level_cache_isolation()
        print("[EXP-08] Running Part B: Fair Closed-Loop Synergy (100% Real Telemetry)...", flush=True)
        part_b = _run_part_b_fair_closed_loop_synergy()
    print("[EXP-08] Running Part C: Full 120-Episode Blind Holdout Check & Paired Wilcoxon Stats of CoRe-Sched...", flush=True)
    part_c = _run_part_c_120ep_registered_ahg_rl()

    payload = {
        "experiment": "EXP-08_Revised_Fair_Dual_Engine_Synergy_and_3Level_B2_Audit",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "part_a_3level_isolated_cache_audit": part_a,
        "part_b_fair_closed_loop_synergy": part_b,
        "part_c_120ep_registered_ahg_rl": part_c,
    }
    json_path = OUT_DIR / "exp08_learning_b2_synergy.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# EXP-08（公平修正版）: 学习引擎 (`AHG-RL`) × 优化引擎 (`CVCR-B2`) 双引擎协同与三层隔离缓存审计",
        "",
        "## 1. Part A: 严格相同候选语义下的三层隔离 B2 缓存命中归因审计",
        "",
        "所有 5 个评价器（`A1`、`B1`、`L1 Unseeded B2`、`L2 Seeded Raw-Context B2`、`L3 Seeded Seq-Diff B2`）均对**完全相同的 `(cand.assign, cand.op_prio, cand.bays, cand.crews, cand.task_sequences)`** 开展评价：",
        "",
        "| 候选生成模式 | 审计候选数 | 受影响车队比例 (`|ΔR|/|R|`) | 可复用未变车队比例 | L1 未初始化 B2 命中率 | **L2 已种子化+原始上下文 B2 命中率** | **L3 已种子化+序列差分 B2 命中率** | 零重路由命中率 | 全车队回退 B1 比例 | 单候选耗时 (`B1 / B2_SeqDiff` ms, 提速) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k, v in part_a.items():
        lat_str = f"`{v.get('mean_eval_lat_b1_ms', '-')} / {v['mean_eval_lat_b2_seqdiff_ms']}` (**{v.get('speedup_b2_vs_b1', '-')}x**)"
        md_lines.append(
            f"| `{k}` | {v['total_candidates_audited']} | **{v['mean_changed_amr_ratio_pct']:.2f}%** | "
            f"**{v['mean_reused_amr_ratio_pct']:.2f}%** | {v['level1_unseeded_b2_hit_pct']:.2f}% | "
            f"**{v['level2_seeded_raw_ctx_b2_hit_pct']:.2f}%** | **{v['level3_seeded_seqdiff_b2_hit_pct']:.2f}%** | "
            f"{v['b2_zero_route_hit_rate_pct']:.2f}% | {v['b2_full_fleet_fallback_rate_pct']:.2f}% | {lat_str} |"
        )

    md_lines.extend(
        [
            "",
            "## 2. Part B: 严格一致候选语义与 100% 实测遥测的闭环双引擎组合对比 (24 场高负载车间)",
            "",
            "| 组合方案 | Mean Net Obj ($J_A$) | Median Net Obj | Mean $C_{\\max}$ | Mean Tardiness | 任务完工率 | 工序完工率 | 实测 B2 增量命中率 | 实测复用未变车队比例 | 平均单次评价耗时 (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for k, v in part_b.items():
        b2_s = f"**{v['b2_hit_rate_pct']:.1f}%**" if v["b2_hit_rate_pct"] is not None else "N/A"
        reu_s = f"**{v['mean_reused_amr_ratio_pct']:.1f}%**" if v["mean_reused_amr_ratio_pct"] is not None else "N/A"
        lat_s = f"{v['mean_eval_lat_ms']:.3f}" if v["mean_eval_lat_ms"] is not None else "N/A"
        md_lines.append(
            f"| `{k}` | **{v['mean_net_obj']:.2f}** | {v['median_net_obj']:.2f} | {v['mean_cmax']:.1f} | "
            f"{v['mean_tardiness']:.1f} | {v['task_comp_pct']:.2f}% | {v['op_comp_pct']:.2f}% | "
            f"{b2_s} | {reu_s} | {lat_s} |"
        )

    md_lines.extend(
        [
            "",
            "## 3. Part C: 正式注册默认 `V19_ADAPTIVE_RL` (`AHG-RL + CVCR-B2`) 120 场未见盲测完整核验",
            "",
            f"- **Mean `net_obj`**: **`{part_c['mean_net_obj']:.2f}`** (Median `{part_c['median_net_obj']:.2f}`, `4-Topo = {part_c['split_mean_net_obj']['blind_4topo']}`, `Track 1 = {part_c['split_mean_net_obj']['blind_track1']}`, `Track 2 = {part_c['split_mean_net_obj']['blind_track2']}`)",
            f"- **完工可靠性**: **`{part_c['num_100pct_completed']} / 120` 场 100% 完工**（任务完工率 **`{part_c['mean_task_comp_pct']:.2f}%`**，工序完工率 **`{part_c['mean_prod_comp_pct']:.2f}%`**，未完工惩罚 `{part_c['unfinished_penalty_total']:.2f}`，独立轨迹审计 `100.0%` 通过）",
            f"- **实测 `CVCR-B2` 增量效率**: B2 增量命中率 **`{part_c['b2_hit_rate_pct']}%`**（`{part_c['b2_inc_hits']} / {part_c['b2_cand_evals']}`），平均复用未变车队比例 **`{part_c['mean_reused_amr_ratio_pct']}%`**，平均单次评价耗时 **`{part_c['mean_eval_lat_ms']} ms`**。",
        ]
    )

    md_path = OUT_DIR / "exp08_learning_b2_synergy.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[EXP-08] Saved {json_path} and {md_path}", flush=True)


if __name__ == "__main__":
    main()
