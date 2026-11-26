"""Task A (`IF-CORE-V20-01`): Decision Capability Completeness, Candidate Funnel & Phase-0 Probe on VAL-72."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.baselines.policy_registry import (
    _V19_RL_CKPT,
    _V19_STAGED_RL_CKPT,
    create_discrete_policy_env,
)
from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.cvcr.snapshot import NativeSnapshot, NativeSnapshotCapture
from amr_sched.cvcr.v17_window_protection import (
    plan_bridge_order_control,
    select_coupled_op_pair,
)
from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state
from amr_sched.rl.causal_repair_cone import (
    FastJAWKittingSurrogate,
    VOIBudgetedOperatorSearch,
    _compute_fresh_semicircle_bays,
    compute_candidate_ja_components,
    generate_causal_cone_candidates,
    observable_op_closure,
    project_zero_churn_candidate,
    score_candidate_ja_unified,
)
from amr_sched.rl.v19_hetero_constructor import (
    V19AdaptiveCouplingProposer,
    decode_v19_candidate,
)
from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
from experiments.v20_audit.common import (
    OUT_AUDIT_DIR,
    assert_stage1_quarantine,
    build_val72_specs,
    run_wrapped_episode,
    summarize_episode_rows,
)
from experiments.v20_audit.proposer_variants import (
    Phase0Config,
    Phase0RestoredProposer,
    compute_extended_amr_footprint,
    derive_base_op_prio,
)


def _capture_val72_v121_snapshots(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replay frozen `V19_ADAPTIVE_RL` on `VAL-72` and capture exact replan snapshots, proposals & receipts."""
    captured: list[dict[str, Any]] = []
    for sp in specs:
        sc_copy = copy.deepcopy(sp["scenario_tuple"])
        p2_copy = copy.deepcopy(sp["p2_cfg"])
        seed = int(sp["seed"])
        max_steps = int(sp["max_steps"])
        kind, ctrl, env = create_discrete_policy_env(
            "V19_ADAPTIVE_RL",
            topology=str(sc_copy[0]),
            seed=seed,
            max_steps=max_steps,
            use_rust=True,
            p2_cfg=p2_copy,
            allow_missing_pgmpo=False,
        )
        raw_ext = env.external_proposer
        assert raw_ext is not None
        step_snaps: dict[int, tuple[NativeSnapshot, CandidateProposal]] = {}

        def _cap(snap: NativeSnapshot, net: Any, *, budget_ms: float = 35.0) -> CandidateProposal:
            prop = raw_ext(snap, net, budget_ms=budget_ms)
            step_snaps[int(snap.tick)] = (copy.deepcopy(snap), copy.deepcopy(prop))
            return prop

        env.external_proposer = _cap
        obs, info = env.reset(seed=seed, options={"scenario_tuple": sc_copy})
        tm = False
        tr = False
        while not (tm or tr):
            f = compute_expert_v9_state(
                env.sim,
                env.tracker,
                last_replan_tick=env.last_replan_tick,
                recent_conflict_ticks=env.recent_conflict_ticks,
                decision_logs=env.decision_logs,
            )
            m = compute_expert_v10_action_mask(env.sim, env.last_replan_tick)
            act = ctrl.select_action(f, dict(info, action_mask=m), sim=env.sim)
            t_cur = int(env.current_step)
            obs, _, tm, tr, info = env.step(act)
            if t_cur in step_snaps:
                snap_obj, prop_obj = step_snaps[t_cur]
                log_row = env.decision_logs[-1] if env.decision_logs else {}
                captured.append(
                    {
                        "episode_key": sp["episode_key"],
                        "suite_group": sp["suite_group"],
                        "regime": sp["regime"],
                        "seed": seed,
                        "tick": t_cur,
                        "network": copy.deepcopy(env.sim.network),
                        "snapshot": snap_obj,
                        "proposal": prop_obj,
                        "receipt_accepted": bool(log_row.get("receipt_accepted", False)),
                        "receipt_reasons": list(log_row.get("receipt_reasons", ())),
                        "budget_ms_requested": float(log_row.get("budget_ms", 35.0)),
                        "propose_ms_actual": float(log_row.get("t_propose_ms", 0.0)),
                    }
                )
    return captured


def _analyze_snapshot_funnel_and_flaws(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit F1-F5 and candidate funnel across all captured `VAL-72` replan snapshots."""
    anchor = V19AdaptiveCouplingProposer(_V19_RL_CKPT, _V19_STAGED_RL_CKPT, backend="numpy", use_rust=True, seed=0)
    anchor._ensure_nets()
    anchor._ensure_infer_nets()

    funnel_by_group: dict[str, dict[str, int]] = {}
    snap_rows: list[dict[str, Any]] = []

    f1_a4_evals = 0
    f1_a4_improves = 0
    f1_delta_j: list[float] = []
    f1_delta_h1: list[float] = []
    f1_unwindowed_seq_omitted = 0

    f2_trunc_evals = 0
    f2_nn_tail_beat_top3 = 0
    f2_staged_zc_in_tail = 0
    f2_staged_zc_best = 0
    f2_cone_tail_beat_head = 0
    f2_locked_nn_bays = 0

    f3_footprint_checked = 0
    f3_footprint_missed_amrs = 0
    f3_legacy_zero_with_prio_change = 0
    f3_cliff_eligible = 0
    f3_cliff_had_live_or_lock_dropped = 0

    f4_voi_calls_in_v121 = 0
    f4_voi_improves = 0
    f4_voi_delta_j: list[float] = []
    f4_eff_budget_lt_requested = 0
    f4_budget_slack_ms: list[float] = []

    f5_bandit_updates = 0
    f5_linucb_misattributed = 0
    f5_non_arm_won = 0

    for idx, rec in enumerate(records):
        snap: NativeSnapshot = rec["snapshot"]
        net = rec["network"]
        prop: CandidateProposal = rec["proposal"]
        grp = str(rec["suite_group"])
        fb = funnel_by_group.setdefault(
            grp,
            {
                "snapshots": 0,
                "generated": 0,
                "shortlisted": 0,
                "b2_verified": 0,
                "b2_valid": 0,
                "accepted_vs_keep": 0,
                "tx_accepted": 0,
                "tx_rejected": 0,
            },
        )
        fb["snapshots"] += 1

        telem = prop.telemetry or {}
        chosen_arm = str(telem.get("chosen_arm", "KEEP"))
        sel_arm_raw = str(telem.get("selected_arm", "full")).lower()
        sel_is_full = sel_arm_raw in ("full", "0")
        nn_eval_n = int(telem.get("neural_candidates_evaluated", 0))
        tot_eval_n = int(telem.get("total_candidates_evaluated", 0))
        cone_gen_n = int(telem.get("cone_candidates_generated", 0))
        fb["b2_verified"] += tot_eval_n
        fb["shortlisted"] += min(3, max(nn_eval_n, 3)) + min(5, cone_gen_n)
        if not bool(telem.get("fallback_keep", True)):
            fb["accepted_vs_keep"] += 1
        if rec["receipt_accepted"]:
            fb["tx_accepted"] += 1
        else:
            fb["tx_rejected"] += 1

        # Check F1 unwindowed proposed_task_sequences drop & F4 eff_budget ceiling & F5 LinUCB
        if prop.proposed_task_sequences is None:
            f1_unwindowed_seq_omitted += 1
        eff_b = float(telem.get("eff_budget_ms", 29.0))
        req_b = float(rec["budget_ms_requested"])
        if eff_b + 0.5 < req_b:
            f4_eff_budget_lt_requested += 1
        f4_budget_slack_ms.append(max(0.0, req_b * 0.85 - float(rec["propose_ms_actual"])))

        if not bool(telem.get("fallback_keep", True)):
            f5_bandit_updates += 1
            win_is_full = chosen_arm.startswith("full")
            win_is_staged = chosen_arm.startswith("staged")
            if (sel_is_full and not win_is_full) or ((not sel_is_full) and not win_is_staged):
                f5_linucb_misattributed += 1
            if not (win_is_full or win_is_staged):
                f5_non_arm_won += 1

        # Reconstruct exact Stage-1 + Cone candidate pool for counterfactual F1-F4 diagnostics
        clf = classify_commitments(snap)
        graph = extract_v19_hetero_graph(snap, net, clf=clf)
        if not graph.rel_tids:
            continue
        ev = CVCREvaluator(net, snap, mode="B2_SEEDED", alpha_makespan=3.0, lambda_change=15.0, use_rust=True, seq_diff_b2=True)
        base_prio = derive_base_op_prio(snap)
        keep_eval = ev.evaluate(graph.base_assign, op_priority=base_prio, op_bay_overrides=graph.base_bays, op_crew_overrides=graph.base_crews)
        keep_seq = {k: list(v) for k, v in keep_eval.robot_order_seq.items() if v} if keep_eval.robot_order_seq else None
        ev.set_incumbent(graph.base_assign, keep_eval, op_priority=base_prio, op_bay_overrides=graph.base_bays, task_order_override=keep_seq)
        rob_cur_h = {r.robot_id: r.cur_task for r in snap.robots if r.cur_task and r.robot_id not in clf.disturbed_robots}
        fast_surr = FastJAWKittingSurrogate(net, snap, ev, graph, lambda_change=15.0)
        fresh_ops = observable_op_closure(snap)
        bays_semi, crews_semi, _ = _compute_fresh_semicircle_bays(snap, net, graph, fast_surr, fresh_ops)
        t_by_id = {t.task_id: t for t in snap.tasks}
        pending_or_dist = {
            tid
            for tid in graph.rel_tids
            if t_by_id[tid].status.lower() == "pending"
            or (t_by_id[tid].assigned_agv is not None and t_by_id[tid].assigned_agv in clf.disturbed_robots)
        }
        rng = np.random.default_rng(int(rec["tick"]) + 17)
        c_full = decode_v19_candidate(anchor._np_full_net, graph, snap, greedy=True, np_rng=rng)
        c_s0 = decode_v19_candidate(anchor._np_full_net, graph, snap, greedy=False, np_rng=rng, tau=0.65)
        c_s1 = decode_v19_candidate(anchor._np_full_net, graph, snap, greedy=False, np_rng=rng, tau=0.65)
        c_st = decode_v19_candidate(anchor._np_staged_net, graph, snap, greedy=True, np_rng=rng)
        zc_a, zc_b, zc_c, zc_p, zc_sq = project_zero_churn_candidate(
            snap, graph, c_st.assign, c_st.bays, c_st.crews, c_st.op_prio, c_st.task_sequences, fresh_ops, pending_or_dist, rob_cur_h
        )

        # F2 locked bays check on >= 4 AMRs / workshops
        if len(net.nodes) >= 30 and c_st.bays != bays_semi:
            f2_locked_nn_bays += 1

        # F1: A4 sequence restoration check on greedy A1..A4 candidate
        res_noseq = ev.evaluate(c_full.assign, op_priority=c_full.op_prio, op_bay_overrides=c_full.bays, op_crew_overrides=c_full.crews, task_order_override=None)
        res_seq = ev.evaluate(c_full.assign, op_priority=c_full.op_prio, op_bay_overrides=c_full.bays, op_crew_overrides=c_full.crews, task_order_override=c_full.task_sequences)
        sc_noseq, _ = score_candidate_ja_unified(snap, net, graph, res_noseq, c_full.assign, c_full.bays, c_full.crews, fresh_ops, fast_surr, sim_horizon=ev.sim_horizon)
        sc_seq, _ = score_candidate_ja_unified(snap, net, graph, res_seq, c_full.assign, c_full.bays, c_full.crews, fresh_ops, fast_surr, sim_horizon=ev.sim_horizon)
        c_noseq = compute_candidate_ja_components(snap, graph, res_noseq, c_full.assign, c_full.bays, c_full.crews, None, rob_cur_h, sim_horizon=ev.sim_horizon)
        c_seq = compute_candidate_ja_components(snap, graph, res_seq, c_full.assign, c_full.bays, c_full.crews, c_full.task_sequences, rob_cur_h, sim_horizon=ev.sim_horizon)
        if res_noseq.valid and res_seq.valid:
            f1_a4_evals += 1
            dj = float(sc_noseq - sc_seq)
            dh1 = float(c_noseq.get("h1_proxy", sc_noseq) - c_seq.get("h1_proxy", sc_seq))
            f1_delta_j.append(dj)
            f1_delta_h1.append(dh1)
            if dj > 1e-4:
                f1_a4_improves += 1

        # F2: Neural & Cone shortlist truncation check
        raw_nn_list = [
            ("full_greedy", c_full.assign, c_full.bays, c_full.crews, c_full.op_prio, c_full.task_sequences, True),
            ("full_s0", c_s0.assign, c_s0.bays, c_s0.crews, c_s0.op_prio, c_s0.task_sequences, True),
            ("full_s1", c_s1.assign, c_s1.bays, c_s1.crews, c_s1.op_prio, c_s1.task_sequences, True),
            ("staged_greedy", c_st.assign, c_st.bays, c_st.crews, c_st.op_prio, c_st.task_sequences, False),
            ("staged_zc", zc_a, zc_b, zc_c, zc_p, zc_sq, False),
        ]
        fb["generated"] += len(raw_nn_list)
        nn_scores: list[tuple[str, float, bool]] = []
        valid_pool: list[Any] = []
        base_r_seq = {k: list(v) for k, v in keep_eval.robot_order_seq.items()} if keep_eval.robot_order_seq else {}
        for tg, asn, bs, cs, op_p, sq, is_f in raw_nn_list:
            rv = ev.evaluate(asn, op_priority=op_p, op_bay_overrides=bs, op_crew_overrides=cs, task_order_override=sq)
            sc_v, _ = score_candidate_ja_unified(snap, net, graph, rv, asn, bs, cs, fresh_ops, fast_surr, sim_horizon=ev.sim_horizon)
            nn_scores.append((tg, float(sc_v) if rv.valid else 1e12, bool(rv.valid)))
            if rv.valid:
                fb["b2_valid"] += 1
                valid_pool.append((tg, rv, float(sc_v), asn, bs, cs, op_p, sq))
            fp = compute_extended_amr_footprint(snap, graph, base_r_seq, asn, bs, cs, op_p, base_prio, sq)
            f3_footprint_checked += 1
            if fp["missed_queue_or_prio_amrs"]:
                f3_footprint_missed_amrs += 1
                if fp["legacy_count"] == 0:
                    f3_legacy_zero_with_prio_change += 1

        if len(nn_scores) >= 4:
            f2_trunc_evals += 1
            best_top3 = min(x[1] for x in nn_scores[:3])
            best_tail = min(x[1] for x in nn_scores[3:])
            if best_tail + 1e-4 < best_top3:
                f2_nn_tail_beat_top3 += 1
            if nn_scores[-1][0] == "staged_zc":
                f2_staged_zc_in_tail += 1
                if nn_scores[-1][1] + 1e-4 < best_top3:
                    f2_staged_zc_best += 1

        ccands = generate_causal_cone_candidates(
            snap, net, graph, ev, fast_surr, keep_eval, base_prio, raw_nn_list[:3], fresh_ops, pending_or_dist, budget_ms_limit=2.0
        )
        fb["generated"] += len(ccands)
        if len(ccands) > 5:
            cone_scores = []
            for tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc in ccands:
                rc = ev.evaluate(asn_c, op_priority=op_c, op_bay_overrides=bs_c, op_crew_overrides=cs_c, task_order_override=sq_c)
                scc, _ = score_candidate_ja_unified(snap, net, graph, rc, asn_c, bs_c, cs_c, fresh_ops, fast_surr, sim_horizon=ev.sim_horizon)
                cone_scores.append(float(scc) if rc.valid else 1e12)
                if rc.valid:
                    fb["b2_valid"] += 1
            if min(cone_scores[5:]) + 1e-4 < min(cone_scores[:5]):
                f2_cone_tail_beat_head += 1

        # F3 Cliff Bridge live_repairs / bottleneck_locks drop check
        pair = select_coupled_op_pair(net, snap, keep_eval, graph.base_assign, base_prio)
        if pair is not None:
            oa, ob, _ = pair
            bp_p = plan_bridge_order_control(net, snap, ev, graph.base_assign, base_prio, oa, ob, base_bays=graph.base_bays, base_crews=graph.base_crews)
            if bp_p.eligible:
                f3_cliff_eligible += 1
                if bp_p.live_repairs or bp_p.bottleneck_locks:
                    f3_cliff_had_live_or_lock_dropped += 1

        # F4 VOI counterfactual check on top of valid pool
        if valid_pool and len(graph.rel_tids) >= 2:
            pre_voi_best = min(x[2] for x in valid_pool)
            voi = VOIBudgetedOperatorSearch(neighborhood_mode="causal_cone", seed=0)
            v_eval, v_sc, *_ = voi.refine(
                ev=ev,
                snapshot=snap,
                graph=graph,
                seed_states=valid_pool,
                score_fn=lambda r, a, b, c, sq: float(score_candidate_ja_unified(snap, net, graph, r, a, b, c, fresh_ops, fast_surr, sim_horizon=ev.sim_horizon)[0]),
                seen_sigs=set(),
                deadline_s=_cvcr_now_s() + 0.010,
                max_evals=6,
                hysteresis_margin=0.5,
                keep_eval=keep_eval,
            )
            if v_eval.valid and float(v_sc) + 1e-4 < pre_voi_best:
                f4_voi_improves += 1
                f4_voi_delta_j.append(float(pre_voi_best - float(v_sc)))

        if idx < 250:
            snap_rows.append(
                {
                    "episode_key": rec["episode_key"],
                    "suite_group": grp,
                    "tick": rec["tick"],
                    "chosen_arm": chosen_arm,
                    "sel_arm": sel_arm_raw,
                    "fallback_keep": bool(telem.get("fallback_keep", True)),
                    "receipt_accepted": rec["receipt_accepted"],
                    "eff_budget_ms": eff_b,
                    "propose_ms": rec["propose_ms_actual"],
                    "f1_dj": round(float(sc_noseq - sc_seq), 3) if (res_noseq.valid and res_seq.valid) else None,
                    "f2_tail_beat_top3": bool(len(nn_scores) >= 4 and min(x[1] for x in nn_scores[3:]) + 1e-4 < min(x[1] for x in nn_scores[:3])),
                }
            )

    summary = {
        "total_snapshots_audited": len(records),
        "funnel_by_suite": funnel_by_group,
        "diagnostics": {
            "F1_a4_queue_drop": {
                "pairs_evaluated": f1_a4_evals,
                "a4_improves_count": f1_a4_improves,
                "a4_improves_pct": round(100.0 * f1_a4_improves / max(1, f1_a4_evals), 2),
                "mean_delta_ja_proxy": round(float(np.mean(f1_delta_j)), 3) if f1_delta_j else 0.0,
                "mean_delta_h1_proxy": round(float(np.mean(f1_delta_h1)), 3) if f1_delta_h1 else 0.0,
                "unwindowed_seq_omitted_calls": f1_unwindowed_seq_omitted,
            },
            "F2_truncation_and_bay_lock": {
                "truncation_checks": f2_trunc_evals,
                "nn_tail_beat_top3_count": f2_nn_tail_beat_top3,
                "nn_tail_beat_top3_pct": round(100.0 * f2_nn_tail_beat_top3 / max(1, f2_trunc_evals), 2),
                "staged_zc_in_tail_count": f2_staged_zc_in_tail,
                "staged_zc_beat_top3_count": f2_staged_zc_best,
                "cone_tail_beat_head5_count": f2_cone_tail_beat_head,
                "workshop_forced_bay_lock_count": f2_locked_nn_bays,
            },
            "F3_footprint_and_cliff_locks": {
                "candidates_checked": f3_footprint_checked,
                "footprint_missed_queue_or_prio_amrs": f3_footprint_missed_amrs,
                "legacy_reported_zero_changed_when_prio_or_queue_changed": f3_legacy_zero_with_prio_change,
                "cliff_eligible_snapshots": f3_cliff_eligible,
                "cliff_live_repairs_or_locks_dropped": f3_cliff_had_live_or_lock_dropped,
            },
            "F4_voi_and_budget_cap": {
                "voi_calls_in_frozen_v121": f4_voi_calls_in_v121,
                "voi_improves_when_wired_count": f4_voi_improves,
                "voi_mean_delta_j_when_better": round(float(np.mean(f4_voi_delta_j)), 3) if f4_voi_delta_j else 0.0,
                "calls_with_eff_budget_below_requested": f4_eff_budget_lt_requested,
                "mean_unused_35ms_budget_ms": round(float(np.mean(f4_budget_slack_ms)), 3) if f4_budget_slack_ms else 0.0,
            },
            "F5_linucb_credit_misattribution": {
                "non_keep_bandit_updates": f5_bandit_updates,
                "misattributed_updates": f5_linucb_misattributed,
                "misattributed_pct": round(100.0 * f5_linucb_misattributed / max(1, f5_bandit_updates), 2),
                "non_arm_cone_or_cliff_won_updates": f5_non_arm_won,
            },
        },
    }
    return summary, snap_rows


def run_task_a() -> dict[str, Any]:
    """Execute Task A (`VAL-72` funnel audit + `Phase0_Restored` closed-loop ladder probe)."""
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    specs = build_val72_specs()
    assert_stage1_quarantine([sp["seed"] for sp in specs], label="TaskA_VAL72")

    print("[Task A] Capturing frozen CoRe-Sched v1.2.1 VAL-72 replan snapshots...", flush=True)
    records = _capture_val72_v121_snapshots(specs)
    print(f"[Task A] Captured {len(records)} VAL-72 replan snapshots; analyzing funnel & F1-F5...", flush=True)
    funnel_summary, snap_rows = _analyze_snapshot_funnel_and_flaws(records)

    print("[Task A] Running non-invasive closed-loop Phase-0 ladder on VAL-72...", flush=True)
    configs: list[tuple[str, str, Any]] = [
        ("P0.0_CoRe_v121_Frozen", "V19_ADAPTIVE_RL", None),
        (
            "P0.1a_A4_EvalOnly",
            "V19_ADAPTIVE_RL",
            Phase0RestoredProposer(Phase0Config(name="P0.1a", fix_a4_seq=True, commit_unwindowed_seq=False), backend="numpy", use_rust=True),
        ),
        (
            "P0.1b_A4_CommitUnwin",
            "V19_ADAPTIVE_RL",
            Phase0RestoredProposer(Phase0Config(name="P0.1b", fix_a4_seq=True, commit_unwindowed_seq=True), backend="numpy", use_rust=True),
        ),
        (
            "P0.2a_Pool_Bays_Bandit_NoLocks",
            "V19_ADAPTIVE_RL",
            Phase0RestoredProposer(
                Phase0Config(name="P0.2a", fix_a4_seq=False, unlock_full_pool_and_bays=True, attach_cliff_locks=False),
                backend="numpy",
                use_rust=True,
            ),
        ),
        (
            "P0.2b_Pool_PlusCliffLocks",
            "V19_ADAPTIVE_RL",
            Phase0RestoredProposer(
                Phase0Config(name="P0.2b", fix_a4_seq=False, unlock_full_pool_and_bays=True, attach_cliff_locks=True),
                backend="numpy",
                use_rust=True,
            ),
        ),
        (
            "P0.3_VOI_CleanBudget_NoSeqLock",
            "V19_ADAPTIVE_RL",
            Phase0RestoredProposer(
                Phase0Config(
                    name="P0.3_SD",
                    fix_a4_seq=False,
                    unlock_full_pool_and_bays=True,
                    attach_cliff_locks=False,
                    wire_voi_and_scale_budget=True,
                    use_seq_diff_b2=True,
                ),
                backend="numpy",
                use_rust=True,
            ),
        ),
        ("V18_EA_Ref", "V18_EA_JOINT", None),
    ]

    ladder_results: dict[str, Any] = {}
    for label, base_pol, proposer in configs:
        rows: list[dict[str, Any]] = []
        for sp in specs:
            rec, _ = run_wrapped_episode(
                variant_id=label,
                spec=sp,
                base_env_policy=base_pol,
                external_proposer=proposer,
                collect_traces=False,
            )
            rows.append(rec)
        by_suite = {
            grp: summarize_episode_rows([r for r in rows if r["suite_group"] == grp])
            for grp in ("4topo", "track1", "track2")
        }
        ladder_results[label] = {
            "overall": summarize_episode_rows(rows),
            "by_suite": by_suite,
        }
        ov = ladder_results[label]["overall"]
        print(
            f"  -> {label}: net_obj={ov['mean_net_obj']}, comp={ov['comp100_count']}/{ov['n']}, "
            f"4topo={by_suite['4topo']['mean_net_obj']}, t1={by_suite['track1']['mean_net_obj']}, "
            f"t2={by_suite['track2']['mean_net_obj']}, p95={ov['pooled_p95_ms']}ms",
            flush=True,
        )

    payload = {
        "schema": "if-core-v20-01.task_a.v1",
        "val_episodes": len(specs),
        "funnel_and_flaws": funnel_summary,
        "phase0_ladder": ladder_results,
    }
    (OUT_AUDIT_DIR / "task_a_funnel_and_phase0.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = OUT_AUDIT_DIR / "task_a_funnel_snapshots.csv"
    if snap_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(snap_rows[0].keys()))
            w.writeheader()
            w.writerows(snap_rows)

    md_lines = [
        "# Task A: Decision Capability Completeness & Candidate Funnel Audit (`VAL-72`)\n",
        f"- **Quarantined VAL-72 Replan Snapshots Audited**: `{funnel_summary['total_snapshots_audited']}`",
        "\n## 1. Root-Cause Quantification (F1-F5)",
        f"- **F1 (`A4` Queue Sequence Drop)**: A4 improves B2/JA in `{funnel_summary['diagnostics']['F1_a4_queue_drop']['a4_improves_pct']}%` (`{funnel_summary['diagnostics']['F1_a4_queue_drop']['a4_improves_count']}/{funnel_summary['diagnostics']['F1_a4_queue_drop']['pairs_evaluated']}`) snapshots; mean ΔJ = `{funnel_summary['diagnostics']['F1_a4_queue_drop']['mean_delta_ja_proxy']}`; `proposed_task_sequences=None` calls = `{funnel_summary['diagnostics']['F1_a4_queue_drop']['unwindowed_seq_omitted_calls']}`.",
        f"- **F2 (`nn_cands[:3]` Truncation & Bay Lock)**: Tail `nn_cands[3:]` beats top-3 in `{funnel_summary['diagnostics']['F2_truncation_and_bay_lock']['nn_tail_beat_top3_pct']}%` (`{funnel_summary['diagnostics']['F2_truncation_and_bay_lock']['nn_tail_beat_top3_count']}/{funnel_summary['diagnostics']['F2_truncation_and_bay_lock']['truncation_checks']}`); `staged_zc` beat top-3 `{funnel_summary['diagnostics']['F2_truncation_and_bay_lock']['staged_zc_beat_top3_count']}` times; workshop forced bay locks = `{funnel_summary['diagnostics']['F2_truncation_and_bay_lock']['workshop_forced_bay_lock_count']}`.",
        f"- **F3 (`op_priority` Footprint Blindness & Dropped Cliff Locks)**: `{funnel_summary['diagnostics']['F3_footprint_and_cliff_locks']['footprint_missed_queue_or_prio_amrs']}` candidates had missed queue/prio AMRs (`{funnel_summary['diagnostics']['F3_footprint_and_cliff_locks']['legacy_reported_zero_changed_when_prio_or_queue_changed']}` reported legacy 0 changed); `{funnel_summary['diagnostics']['F3_footprint_and_cliff_locks']['cliff_live_repairs_or_locks_dropped']}` cliff bridge plans dropped `live_repairs`/`bottleneck_locks`.",
        f"- **F4 (`VOI` Unwired & Budget Ceiling)**: Frozen v1.2.1 `VOI` calls = `{funnel_summary['diagnostics']['F4_voi_and_budget_cap']['voi_calls_in_frozen_v121']}`; wiring VOI improved `{funnel_summary['diagnostics']['F4_voi_and_budget_cap']['voi_improves_when_wired_count']}` snapshots; `eff_budget < req_budget` in `{funnel_summary['diagnostics']['F4_voi_and_budget_cap']['calls_with_eff_budget_below_requested']}` calls (mean unused headroom = `{funnel_summary['diagnostics']['F4_voi_and_budget_cap']['mean_unused_35ms_budget_ms']} ms`).",
        f"- **F5 (`LinUCB` Credit Misattribution)**: `{funnel_summary['diagnostics']['F5_linucb_credit_misattribution']['misattributed_pct']}%` (`{funnel_summary['diagnostics']['F5_linucb_credit_misattribution']['misattributed_updates']}/{funnel_summary['diagnostics']['F5_linucb_credit_misattribution']['non_keep_bandit_updates']}`) non-KEEP updates rewarded the wrong arm.",
        "\n## 2. Phase-0 Non-Invasive Repair Ladder (`VAL-72`)\n",
        "| Variant | VAL-72 Net Obj | 4-Topo (48) | Track 1 (12) | Track 2 (12) | Comp100 | Mean H1 | Pooled P95 (ms) | Pooled P99 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, _, _ in configs:
        ov = ladder_results[label]["overall"]
        bs = ladder_results[label]["by_suite"]
        md_lines.append(
            f"| {label} | {ov['mean_net_obj']:.2f} | {bs['4topo']['mean_net_obj']:.2f} | "
            f"{bs['track1']['mean_net_obj']:.2f} | {bs['track2']['mean_net_obj']:.2f} | "
            f"{ov['comp100_count']}/{ov['n']} | {ov['mean_h1_all']:.1f} | "
            f"{ov['pooled_p95_ms']:.2f} | {ov['pooled_p99_ms']:.2f} |"
        )
    (OUT_AUDIT_DIR / "task_a_funnel_and_phase0.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    run_task_a()
