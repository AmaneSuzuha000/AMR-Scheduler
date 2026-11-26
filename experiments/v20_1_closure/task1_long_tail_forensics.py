"""Task 1 (F1): Long-Tail Causal Forensics for 2906, 3905, and 1163.

Strictly diagnostic & read-only relative to frozen V20.0 / amr_sched/.
Produces Deliverable 1 (`out/v20_1_closure/deliverable1_failure_root_cause_report.{json,md}`):
  1. `track1_nominal_2906`: First divergence tick via counterfactual replay fork vs V18/V19 + exhaustive
     classification of all inner PDG escape calls into 6 structural failure modes.
  2. `track2_kitting_windows_3905`: Tick-by-tick critical-chain op progress, window-miss accumulation,
     and global-progress-reset masking count + first divergence tick vs V18/V19.
  3. `track2_execution_blockage_1163`: Multi-trial wall-clock vs deterministic virtual-clock dispersion
     to isolate budget truncation jitter from structural blockage.
"""

from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.baselines.policy_registry import create_discrete_policy_env, run_discrete_episode
from amr_sched.cvcr.clock import now_s as _cvcr_now_s, virtual_clock_scope
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.cvcr.snapshot import NativeSnapshot, NativeSnapshotCapture, align_start_to_windows, op_predecessors
from amr_sched.cvcr.v12_proactive_guard import plan_haven_safe_corridor_repair
from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state
from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
from experiments.run_g3_fresh_frozen_holdout import build_fresh_120_specs
from experiments.run_g5_pristine_blind_holdout import build_pristine_120_specs
from experiments.v20_audit.common import (
    RecordedStep,
    build_val72_specs,
    make_v18_external_proposer,
    replay_fork_env_at_tick,
    rollout_48_and_optional_full,
)
from experiments.v20_audit.v20_e_liveness import PDGLivenessTracker, build_default_v20_proposer

OUT_CLOSURE_DIR = ROOT / "out" / "v20_1_closure"


def _stat_str(x: Any) -> str:
    return str(getattr(x, "value", x)).lower()


def _record_v20_trace(sp: dict[str, Any], *, use_virtual_clock: bool = False) -> dict[str, Any]:
    """Run frozen V20_NAJS recording exact step traces, per-decision telemetry, and per-tick op/agv snapshots."""
    sc_copy = copy.deepcopy(sp["scenario_tuple"])
    p2_copy = copy.deepcopy(sp["p2_cfg"])
    seed = int(sp["seed"])
    max_steps = int(sp["max_steps"])
    topo = str(sc_copy[0])

    _, ctrl, env = create_discrete_policy_env(
        "V19_ADAPTIVE_RL", topology=topo, seed=seed, max_steps=max_steps, use_rust=True, p2_cfg=p2_copy
    )
    prop = build_default_v20_proposer(hard_budget_ms=35.0, seed=0, enable_pdg_recovery=True)
    tracker = PDGLivenessTracker(cooldown_ticks=12)

    recorded: list[RecordedStep] = []
    decision_events: list[dict[str, Any]] = []
    escape_events: list[dict[str, Any]] = []
    op_timeline: list[dict[str, Any]] = []
    global_progress_resets: list[dict[str, Any]] = []

    def _wrapped_prop(snap: NativeSnapshot, net: Any, *, budget_ms: float = 35.0) -> CandidateProposal:
        is_pdg = bool(prop._active_pdg and prop._active_pdg.get("pdg_alert"))
        t0 = _cvcr_now_s()
        p = prop(snap, net, budget_ms=budget_ms)
        dt_ms = (_cvcr_now_s() - t0) * 1000.0
        recorded.append(
            RecordedStep(step_before=int(snap.tick), action=-1, triggered_replan=True, proposal=copy.deepcopy(p), propose_ms=dt_ms)
        )
        telem = dict(p.telemetry or {})
        clf = classify_commitments(snap)
        graph = extract_v19_hetero_graph(snap, net, clf=clf)
        n_changed_a = sum(1 for tid, aid in p.proposed_assignment.items() if graph.base_assign.get(tid) != aid)
        n_changed_b = sum(1 for oid, bid in p.proposed_bays.items() if graph.base_bays.get(oid) != bid)
        ev_rec = {
            "tick": int(snap.tick),
            "pdg_mode": is_pdg,
            "budget_ms": float(budget_ms),
            "propose_ms": round(dt_ms, 3),
            "fallback_keep": bool(telem.get("fallback_keep", False)),
            "best_expert": telem.get("best_expert"),
            "best_tag": telem.get("best_tag"),
            "eval_cnt": int(telem.get("eval_cnt", 0)),
            "keep_obj": telem.get("keep_obj"),
            "best_obj": telem.get("best_obj"),
            "n_rel_tids": len(graph.rel_tids),
            "n_frozen_ops": len(graph.frozen_ops),
            "n_changed_assign": n_changed_a,
            "n_changed_bays": n_changed_b,
            "has_live_repairs": bool(getattr(p, "proposed_live_repairs", None)),
            "has_bottleneck_locks": bool(getattr(p, "proposed_bottleneck_locks", None)),
            "has_task_sequences": bool(getattr(p, "proposed_task_sequences", None)),
            "pdg_snapshot": copy.deepcopy(prop._active_pdg) if is_pdg else None,
        }
        decision_events.append(ev_rec)
        if is_pdg:
            # Check whether haven corridor repair or alternative V18 proposal exists at this exact stalled tick
            haven_reps, _ = plan_haven_safe_corridor_repair(env.sim, snap, stall_threshold=6, enable_choke_oscillation_guard=True)
            agv_states = [
                {
                    "agv": r.robot_id, "node": r.cur_node, "state": r.state,
                    "cur_task": r.cur_task, "loaded": r.loaded_task_id, "fault_until": r.faulted_until,
                }
                for r in snap.robots
            ]
            unfin_tasks = [
                {"tid": t.task_id, "status": t.status, "agv": t.assigned_agv, "kit_op": t.kit_op_id, "rel": t.release_step}
                for t in snap.tasks
                if t.status.lower() not in ("completed", "failed")
            ]
            unfin_ops = [
                {
                    "op": op.op_id, "bay": op.bay_id, "started": op.started,
                    "kits": list(op.kit_task_ids), "windows": list(op.resource_windows),
                }
                for op in snap.operations
                if op.end_step is None
            ]
            escape_events.append(
                {
                    **ev_rec,
                    "haven_repairs_available": len(haven_reps),
                    "haven_repair_agvs": sorted(haven_reps.keys()),
                    "agv_states": agv_states,
                    "unfin_tasks": unfin_tasks,
                    "unfin_ops": unfin_ops,
                }
            )
        return p

    env.external_proposer = _wrapped_prop
    scope = virtual_clock_scope(poll_ms=0.002) if use_virtual_clock else _null_ctx()
    with scope:
        obs, info = env.reset(seed=seed, options={"scenario_tuple": sc_copy})
        tm = tr = False
        prev_ct, prev_co = 0, 0
        op_last_adv: dict[str, int] = {op.op_id: 0 for op in env.tracker.operations}
        prev_op_kdone: dict[str, int] = {op.op_id: 0 for op in env.tracker.operations}
        prev_op_state: dict[str, tuple[bool, bool]] = {op.op_id: (False, False) for op in env.tracker.operations}

        while not (tm or tr):
            tick = int(env.current_step)
            t_by_id = {t.task_id: t for t in env.sim.tasks}
            # Per-op progress tracking
            starving_ops_now: list[str] = []
            for op in env.tracker.operations:
                started = bool(getattr(op, "started", False))
                done = bool(op.end_step is not None and int(op.end_step) <= tick)
                k_done = sum(
                    1 for kt in op.kit_task_ids
                    if kt in t_by_id and _stat_str(t_by_id[kt].status) == "completed"
                )
                if k_done > prev_op_kdone[op.op_id] or (started, done) != prev_op_state[op.op_id]:
                    op_last_adv[op.op_id] = tick
                    prev_op_kdone[op.op_id] = k_done
                    prev_op_state[op.op_id] = (started, done)
                if not done and (tick - op_last_adv[op.op_id]) >= 180:
                    starving_ops_now.append(op.op_id)

            ct, co = env._completion_counts(int(env.sim._step_cursor))
            if (ct > prev_ct or co > prev_co) and starving_ops_now:
                global_progress_resets.append(
                    {
                        "tick": tick,
                        "ct": ct, "co": co,
                        "starving_ops_masked": list(starving_ops_now),
                        "max_op_stall_ticks": max(tick - op_last_adv[o] for o in starving_ops_now),
                    }
                )
            prev_ct, prev_co = ct, co

            if tick % 50 == 0 or tick == max_steps - 1:
                op_rows = []
                for op in env.tracker.operations:
                    done = bool(op.end_step is not None and int(op.end_step) <= tick)
                    if done:
                        continue
                    k_done = sum(
                        1 for kt in op.kit_task_ids
                        if kt in t_by_id and _stat_str(t_by_id[kt].status) == "completed"
                    )
                    wins = tuple(getattr(op, "resource_windows", ()) or ())
                    missed_wins = sum(1 for _, w_end in wins if tick > int(w_end))
                    aligned = align_start_to_windows(tick, wins) if wins else tick
                    op_rows.append(
                        {
                            "op": op.op_id,
                            "bay": op.bay_id,
                            "started": bool(op.started),
                            "k_done": f"{k_done}/{len(op.kit_task_ids)}",
                            "due": int(op.due_step),
                            "stall_age": tick - op_last_adv[op.op_id],
                            "missed_wins": missed_wins,
                            "total_wins": len(wins),
                            "win_align_delay": int(aligned - tick),
                        }
                    )
                op_timeline.append(
                    {
                        "tick": tick,
                        "ct": ct, "co": co,
                        "global_zpt": tick - tracker.last_progress_tick,
                        "active_ops": op_rows,
                    }
                )

            feats = compute_expert_v9_state(
                env.sim, env.tracker,
                last_replan_tick=env.last_replan_tick,
                recent_conflict_ticks=env.recent_conflict_ticks,
                decision_logs=env.decision_logs,
            )
            mask = compute_expert_v10_action_mask(env.sim, env.last_replan_tick)
            step_info = dict(info, action_mask=mask)
            base_act = int(ctrl.select_action(feats, step_info, sim=env.sim))
            act = tracker.observe_and_maybe_escalate(env, step_info, base_act, proposer=prop)

            before_n = len(recorded)
            before_logs = len(env.decision_logs)
            obs, _, tm, tr, info = env.step(act)
            if len(recorded) > before_n:
                recorded[-1].action = act
                if len(env.decision_logs) > before_logs:
                    dl = env.decision_logs[-1]
                    recorded[-1].propose_ms = float(dl.get("t_propose_ms", recorded[-1].propose_ms))
                    decision_events[-1]["base_action"] = base_act
                    decision_events[-1]["executed_action"] = act
                    decision_events[-1]["receipt_accepted"] = bool(dl.get("receipt_accepted", False))
                    decision_events[-1]["receipt_reasons"] = list(dl.get("receipt_reasons", []))
                    decision_events[-1]["window_expired"] = bool(dl.get("window_expired", False))
                    decision_events[-1]["window_cost_steps"] = int(dl.get("window_cost_steps", 0))
                    decision_events[-1]["live_repair_count"] = int(dl.get("live_repair_count", 0))
                    if escape_events and escape_events[-1]["tick"] == tick:
                        escape_events[-1]["base_action"] = base_act
                        escape_events[-1]["executed_action"] = act
                        escape_events[-1]["receipt_accepted"] = bool(dl.get("receipt_accepted", False))
                        escape_events[-1]["receipt_reasons"] = list(dl.get("receipt_reasons", []))
                        escape_events[-1]["window_expired"] = bool(dl.get("window_expired", False))
                        escape_events[-1]["live_repair_count"] = int(dl.get("live_repair_count", 0))
            else:
                recorded.append(RecordedStep(step_before=tick, action=act, triggered_replan=False, proposal=None))

    m = env.compute_episode_metrics()
    return {
        "episode_key": sp["episode_key"],
        "net_obj": round(float(m["net_obj"]), 2),
        "cmax": round(float(m["cmax"]), 2),
        "task_comp": round(float(m["task_completion_rate"]), 4),
        "prod_comp": round(float(m["production_completion_rate"]), 4),
        "comp100": bool(float(m["task_completion_rate"]) >= 0.9999 and float(m["production_completion_rate"]) >= 0.9999),
        "expired_windows": int(env.total_window_expirations),
        "pdg_alert_ticks": len(tracker.alert_ticks),
        "pdg_first_alert": tracker.alert_ticks[0] if tracker.alert_ticks else None,
        "pdg_escalations": len(tracker.escalation_ticks),
        "pdg_inner_recoveries": int(prop.pdg_recoveries_triggered),
        "recorded_steps": recorded,
        "decision_events": decision_events,
        "escape_events": escape_events,
        "op_timeline": op_timeline,
        "global_progress_resets": global_progress_resets,
    }


class _null_ctx:
    def __enter__(self):
        return None
    def __exit__(self, *args):
        return False


def _counterfactual_divergence_scan(
    sp: dict[str, Any],
    trace: dict[str, Any],
    *,
    max_ticks_to_probe: int = 12,
) -> list[dict[str, Any]]:
    """Fork `V20_NAJS` trajectory at early decision ticks and compare substituting `V19` or `V18` proposals."""
    recorded: list[RecordedStep] = trace["recorded_steps"]
    dec_ticks = [d["tick"] for d in trace["decision_events"] if not d.get("pdg_mode")]
    if not dec_ticks:
        return []
    # Sample evenly across early-to-mid decisions
    if len(dec_ticks) > max_ticks_to_probe:
        idxs = np.linspace(0, len(dec_ticks) - 1, max_ticks_to_probe, dtype=int)
        probe_ticks = sorted({dec_ticks[i] for i in idxs})
    else:
        probe_ticks = dec_ticks

    _, _, tmp_v19 = create_discrete_policy_env(
        "V19_ADAPTIVE_RL", topology=str(sp["scenario_tuple"][0]), seed=int(sp["seed"]), max_steps=int(sp["max_steps"]), use_rust=True
    )
    v19_prop = tmp_v19.external_proposer
    assert v19_prop is not None
    v18_prop = make_v18_external_proposer(hard_budget_ms=35.0, seq_diff_b2=True)

    results: list[dict[str, Any]] = []
    for tk in probe_ticks:
        try:
            f_env, f_snap, f_net = replay_fork_env_at_tick(sp, recorded, tk)
        except Exception as exc:
            results.append({"tick": tk, "error": str(exc)})
            continue
        p_v19 = v19_prop(f_snap, f_net, budget_ms=35.0)
        p_v18 = v18_prop(f_snap, f_net, budget_ms=35.0)
        rec_step = next((r for r in recorded if r.step_before == tk and r.proposal is not None), None)
        p_v20 = rec_step.proposal if rec_step else None
        act_tk = int(rec_step.action) if rec_step and rec_step.action > 0 else 4

        ro_v20 = rollout_48_and_optional_full(sp, recorded, tk, p_v20, action_at_target=act_tk, rollout_horizon=48, full_continuation=True)
        ro_v19 = rollout_48_and_optional_full(sp, recorded, tk, p_v19, action_at_target=act_tk, rollout_horizon=48, full_continuation=True)
        ro_v18 = rollout_48_and_optional_full(sp, recorded, tk, p_v18, action_at_target=act_tk, rollout_horizon=48, full_continuation=True)

        v20_a = dict(p_v20.proposed_assignment) if p_v20 else {}
        v19_a = dict(p_v19.proposed_assignment)
        v18_a = dict(p_v18.proposed_assignment)
        results.append(
            {
                "tick": tk,
                "v20_diff_v19_assigns": sum(1 for k in set(v20_a) | set(v19_a) if v20_a.get(k) != v19_a.get(k)),
                "v20_diff_v18_assigns": sum(1 for k in set(v20_a) | set(v18_a) if v20_a.get(k) != v18_a.get(k)),
                "v20_full_net": ro_v20.get("full_net_obj"),
                "v20_full_comp": ro_v20.get("full_comp100"),
                "v19_full_net": ro_v19.get("full_net_obj"),
                "v19_full_comp": ro_v19.get("full_comp100"),
                "v18_full_net": ro_v18.get("full_net_obj"),
                "v18_full_comp": ro_v18.get("full_comp100"),
                "v20_j48": ro_v20.get("j48_with_residual"),
                "v19_j48": ro_v19.get("j48_with_residual"),
                "v18_j48": ro_v18.get("j48_with_residual"),
            }
        )
    return results


def _classify_2906_escapes(escape_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify every inner PDG escape call on 2906 into mutually exclusive root-cause buckets."""
    counts: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    for idx, ev in enumerate(escape_events, 1):
        fb = bool(ev.get("fallback_keep", False))
        acc = bool(ev.get("receipt_accepted", True))
        wexp = bool(ev.get("window_expired", False))
        act = int(ev.get("executed_action", 0))
        haven_n = int(ev.get("haven_repairs_available", 0))
        live_cnt = int(ev.get("live_repair_count", 0))
        n_rel = int(ev.get("n_rel_tids", 0))
        n_chg_a = int(ev.get("n_changed_assign", 0))
        n_chg_b = int(ev.get("n_changed_bays", 0))
        pms = float(ev.get("propose_ms", 0.0))

        if wexp:
            mode = "M5_OUTER_ACTION_OR_WINDOW_EXPIRED"
        elif not acc and not fb:
            mode = "M2_EVAL_INVALID_OR_TX_REJECTED"
        elif n_rel == 0:
            mode = "M6_IRREVERSIBLE_NO_RELEASABLE_TASKS"
        elif haven_n > 0 and live_cnt == 0 and act != 1:
            mode = "M5_OUTER_ACTION_SUPPRESSED_CORRIDOR_REPAIR"
        elif pms >= 22.5 and int(ev.get("eval_cnt", 0)) <= 3:
            mode = "M4_BUDGET_TRUNCATED"
        elif fb:
            # Distinguishes whether zero changed candidate improved vs structural absence
            if ev.get("keep_obj") is not None and ev.get("best_obj") is not None and float(ev["best_obj"]) >= float(ev["keep_obj"]) - 1e-3:
                mode = "M3_SHORT_TERM_SCORE_FAVORED_KEEP"
            else:
                mode = "M1_CANDIDATE_ABSENT_OR_INFEASIBLE"
        elif n_chg_a == 0 and n_chg_b == 0:
            mode = "M1_NOOP_CANDIDATE_SELECTED"
        else:
            mode = "M1_TASK_REASSIGN_DID_NOT_BREAK_STALL"
        counts[mode] += 1
        if idx <= 10 or idx % 10 == 0 or idx == len(escape_events):
            details.append(
                {
                    "call_idx": idx,
                    "tick": ev["tick"],
                    "mode": mode,
                    "base_act": ev.get("base_action"),
                    "exec_act": act,
                    "fallback_keep": fb,
                    "accepted": acc,
                    "reasons": ev.get("receipt_reasons", []),
                    "propose_ms": pms,
                    "eval_cnt": ev.get("eval_cnt"),
                    "n_rel": n_rel,
                    "chg_a": n_chg_a,
                    "chg_b": n_chg_b,
                    "haven_avail": haven_n,
                    "live_cnt": live_cnt,
                    "pdg": ev.get("pdg_snapshot"),
                    "agv_states": ev.get("agv_states"),
                }
            )
    return {
        "total_inner_escapes": len(escape_events),
        "taxonomy_counts": dict(counts),
        "sample_calls": details,
    }


def run_task1_forensics() -> dict[str, Any]:
    OUT_CLOSURE_DIR.mkdir(parents=True, exist_ok=True)
    sp_2906 = next(s for s in build_fresh_120_specs() if s["episode_key"] == "track1_nominal_2906")
    sp_3905 = next(s for s in build_pristine_120_specs() if s["episode_key"] == "track2_kitting_windows_3905")
    sp_1163 = next(s for s in build_val72_specs() if s["episode_key"] == "track2_execution_blockage_1163")

    print("=== [Task 1.1] Forensics on track1_nominal_2906 (Cohort A) ===", flush=True)
    tr_2906 = _record_v20_trace(sp_2906, use_virtual_clock=False)
    # If wall-clock happened to pass on this run, also check deterministic virtual clock to capture exact failure path
    if tr_2906["comp100"]:
        tr_2906_vc = _record_v20_trace(sp_2906, use_virtual_clock=True)
        print(f"  2906 wall comp={tr_2906['comp100']} ({tr_2906['net_obj']}), virtual comp={tr_2906_vc['comp100']} ({tr_2906_vc['net_obj']})", flush=True)
        if not tr_2906_vc["comp100"]:
            tr_2906 = tr_2906_vc
    else:
        print(f"  2906 captured failure: net={tr_2906['net_obj']} comp={tr_2906['comp100']} alerts={tr_2906['pdg_alert_ticks']} inner={tr_2906['pdg_inner_recoveries']}", flush=True)
    esc_tax_2906 = _classify_2906_escapes(tr_2906["escape_events"])
    div_2906 = _counterfactual_divergence_scan(sp_2906, tr_2906, max_ticks_to_probe=10)
    print(f"  2906 escape taxonomy: {esc_tax_2906['taxonomy_counts']}", flush=True)

    print("=== [Task 1.2] Forensics on track2_kitting_windows_3905 (Cohort B) ===", flush=True)
    tr_3905 = _record_v20_trace(sp_3905, use_virtual_clock=False)
    if tr_3905["comp100"]:
        tr_3905_vc = _record_v20_trace(sp_3905, use_virtual_clock=True)
        print(f"  3905 wall comp={tr_3905['comp100']} ({tr_3905['net_obj']}), virtual comp={tr_3905_vc['comp100']} ({tr_3905_vc['net_obj']})", flush=True)
        if not tr_3905_vc["comp100"]:
            tr_3905 = tr_3905_vc
    else:
        print(f"  3905 captured failure: net={tr_3905['net_obj']} comp={tr_3905['comp100']} alerts={tr_3905['pdg_alert_ticks']} resets_masking={len(tr_3905['global_progress_resets'])}", flush=True)
    div_3905 = _counterfactual_divergence_scan(sp_3905, tr_3905, max_ticks_to_probe=10)

    print("=== [Task 1.3] Forensics on track2_execution_blockage_1163 (VAL-72 Wall vs Virtual Clock) ===", flush=True)
    trials_1163_wall = []
    for i in range(6):
        r = _record_v20_trace(sp_1163, use_virtual_clock=False)
        trials_1163_wall.append(
            {
                "trial": i, "net_obj": r["net_obj"], "cmax": r["cmax"], "comp100": r["comp100"],
                "alerts": r["pdg_alert_ticks"], "inner": r["pdg_inner_recoveries"], "expired": r["expired_windows"],
            }
        )
    trials_1163_virt = []
    for i in range(3):
        r = _record_v20_trace(sp_1163, use_virtual_clock=True)
        trials_1163_virt.append(
            {
                "trial": i, "net_obj": r["net_obj"], "cmax": r["cmax"], "comp100": r["comp100"],
                "alerts": r["pdg_alert_ticks"], "inner": r["pdg_inner_recoveries"], "expired": r["expired_windows"],
            }
        )
    print(f"  1163 wall trials: {[ (x['net_obj'], x['comp100']) for x in trials_1163_wall ]}", flush=True)
    print(f"  1163 virt trials: {[ (x['net_obj'], x['comp100']) for x in trials_1163_virt ]}", flush=True)

    # Baselines reference on all 3 episodes
    refs = {}
    for sp in (sp_2906, sp_3905, sp_1163):
        refs[sp["episode_key"]] = {}
        for pol in ("V19_ADAPTIVE_RL", "V18_EA_JOINT"):
            res = run_discrete_episode(
                pol, copy.deepcopy(sp["scenario_tuple"]), seed=int(sp["seed"]), max_steps=int(sp["max_steps"]),
                use_rust=True, p2_cfg=copy.deepcopy(sp["p2_cfg"]), episode_key=sp["episode_key"], clock_mode="wall",
            )
            refs[sp["episode_key"]][pol] = {
                "net_obj": res["net_obj"], "cmax": res["cmax"],
                "comp100": bool(res["task_comp"] >= 0.9999 and res["prod_comp"] >= 0.9999),
                "task_comp": res["task_comp"], "prod_comp": res["prod_comp"],
            }

    def _strip_recorded(tr: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in tr.items() if k != "recorded_steps"}

    report = {
        "schema": "v20_1_closure.deliverable1_forensics.v1",
        "baseline_refs": refs,
        "case_2906": {
            "trace_summary": {k: v for k, v in _strip_recorded(tr_2906).items() if k not in ("decision_events", "escape_events")},
            "escape_taxonomy": esc_tax_2906,
            "counterfactual_divergence": div_2906,
            "first_10_decisions": tr_2906["decision_events"][:10],
            "last_10_decisions": tr_2906["decision_events"][-10:],
        },
        "case_3905": {
            "trace_summary": {k: v for k, v in _strip_recorded(tr_3905).items() if k not in ("decision_events", "escape_events")},
            "counterfactual_divergence": div_3905,
            "global_progress_resets_masking_starvation": tr_3905["global_progress_resets"],
            "op_timeline_samples": tr_3905["op_timeline"],
        },
        "case_1163": {
            "wall_clock_6_trials": trials_1163_wall,
            "virtual_clock_3_trials": trials_1163_virt,
            "virtual_clock_bit_identical": len({(x["net_obj"], x["cmax"], x["comp100"]) for x in trials_1163_virt}) == 1,
        },
    }
    (OUT_CLOSURE_DIR / "deliverable1_failure_root_cause_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Build clean Markdown Deliverable 1
    md = [
        "# Deliverable 1: Long-Tail Failure Root-Cause Forensics (`2906`, `3905`, `1163`)",
        "",
        "## 1. `track1_nominal_2906` (Cohort A — Detected Alert, Failed Recovery)",
        f"- **V20.0 Trace**: `net_obj={tr_2906['net_obj']}`, `comp100={tr_2906['comp100']}`, `cmax={tr_2906['cmax']}`, `task_comp={tr_2906['task_comp']}`, `prod_comp={tr_2906['prod_comp']}`, `pdg_first_alert={tr_2906['pdg_first_alert']}`, `alerts={tr_2906['pdg_alert_ticks']}`, `escalations={tr_2906['pdg_escalations']}`, `inner_recoveries={tr_2906['pdg_inner_recoveries']}`",
        f"- **Refs**: `V19={refs['track1_nominal_2906']['V19_ADAPTIVE_RL']}`, `V18={refs['track1_nominal_2906']['V18_EA_JOINT']}`",
        f"- **Inner Escape Failure Taxonomy ({esc_tax_2906['total_inner_escapes']} calls)**: `{esc_tax_2906['taxonomy_counts']}`",
        "",
        "| Fork Tick | V20 vs V19 ΔAssign | V20 Full Net (Comp) | V19 Full Net (Comp) | V18 Full Net (Comp) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for d in div_2906:
        if "error" not in d:
            md.append(f"| {d['tick']} | {d['v20_diff_v19_assigns']} | {d['v20_full_net']} ({d['v20_full_comp']}) | {d['v19_full_net']} ({d['v19_full_comp']}) | {d['v18_full_net']} ({d['v18_full_comp']}) |")

    md.extend([
        "",
        "## 2. `track2_kitting_windows_3905` (Cohort B — Undetected Progressive Starvation)",
        f"- **V20.0 Trace**: `net_obj={tr_3905['net_obj']}`, `comp100={tr_3905['comp100']}`, `cmax={tr_3905['cmax']}`, `task_comp={tr_3905['task_comp']}`, `prod_comp={tr_3905['prod_comp']}`, `alerts={tr_3905['pdg_alert_ticks']}`, `global_resets_masking_starvation={len(tr_3905['global_progress_resets'])}`",
        f"- **Refs**: `V19={refs['track2_kitting_windows_3905']['V19_ADAPTIVE_RL']}`, `V18={refs['track2_kitting_windows_3905']['V18_EA_JOINT']}`",
        "",
        "| Fork Tick | V20 vs V19 ΔAssign | V20 Full Net (Comp) | V19 Full Net (Comp) | V18 Full Net (Comp) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for d in div_3905:
        if "error" not in d:
            md.append(f"| {d['tick']} | {d['v20_diff_v19_assigns']} | {d['v20_full_net']} ({d['v20_full_comp']}) | {d['v19_full_net']} ({d['v19_full_comp']}) | {d['v18_full_net']} ({d['v18_full_comp']}) |")

    md.extend([
        "",
        "### 3905 Global-Progress Reset Masking Events (Unrelated Completions Zeroing `zpt` While Critical Ops Starved >= 180t)",
    ])
    for ev in tr_3905["global_progress_resets"][:15]:
        md.append(f"- `t={ev['tick']}` (`ct={ev['ct']}, co={ev['co']}`): masked starving ops `{ev['starving_ops_masked']}` (max stall `{ev['max_op_stall_ticks']}` ticks)")

    md.extend([
        "",
        "## 3. `track2_execution_blockage_1163` (VAL-72 — Wall-Clock Jitter vs Deterministic Work Clock)",
        f"- **Wall-Clock (6 trials)**: `{[(x['net_obj'], x['comp100'], x['alerts']) for x in trials_1163_wall]}`",
        f"- **Deterministic Virtual Clock (3 trials)**: `{[(x['net_obj'], x['comp100'], x['alerts']) for x in trials_1163_virt]}` (bit-identical=`{report['case_1163']['virtual_clock_bit_identical']}`)",
    ])
    (OUT_CLOSURE_DIR / "deliverable1_failure_root_cause_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[TASK 1 DONE] Wrote Deliverable 1 -> {OUT_CLOSURE_DIR / 'deliverable1_failure_root_cause_report.md'}", flush=True)
    return report


if __name__ == "__main__":
    run_task1_forensics()
