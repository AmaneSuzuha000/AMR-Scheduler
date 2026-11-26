"""Task C (`IF-CORE-V20-01`): Liveness Failure Forensics & Topology-Agnostic PDG Early Warning."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.baselines.policy_registry import create_discrete_policy_env
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.cvcr.snapshot import NativeSnapshot, NativeSnapshotCapture
from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state
from experiments.v20_audit.common import (
    OUT_AUDIT_DIR,
    RecordedStep,
    assert_stage1_quarantine,
    build_task_c_outlier_specs,
    build_val72_specs,
    make_v18_external_proposer,
    rollout_48_and_optional_full,
)
from experiments.v20_audit.proposer_variants import (
    EXPERT_IDS,
    ExpertSuiteEvaluator,
    Phase0Config,
    Phase0RestoredProposer,
)


def compute_pdg_state(
    sim: Any,
    tracker: Any,
    *,
    last_progress_tick: int,
    op_first_kit_done_tick: dict[str, int],
    recent_reject_streak: int,
) -> dict[str, Any]:
    """Compute topology-agnostic Progress Dependency Graph (PDG) structural early-warning metrics."""
    tick = int(getattr(sim, "_step_cursor", 0))
    faulted = set(getattr(sim, "_faulted_agvs", set()))
    blocked_since = dict(getattr(sim, "_blocked_since", {}))
    tasks_by_id = getattr(sim, "_task_by_id", {t.task_id: t for t in sim.tasks})

    healthy_agvs = [a for a in sim.agvs if a.agv_id not in faulted and not getattr(a, "faulted", False)]
    node_occupant: dict[str, str] = {}
    for a in sim.agvs:
        cur_n = getattr(a, "cur_node", None) or getattr(a, "start_node", None)
        if cur_n:
            node_occupant[str(cur_n)] = a.agv_id

    # Directed wait-for graph: AGV u waits on AGV v if u is blocked and u's planned/next hop targets v's node
    wait_adj: dict[str, set[str]] = {a.agv_id: set() for a in healthy_agvs}
    max_loaded_stall = 0
    blocked_healthy = 0
    for a in healthy_agvs:
        aid = a.agv_id
        st_name = getattr(getattr(a, "state", ""), "value", str(getattr(a, "state", ""))).lower()
        is_blocked = (aid in blocked_since) or st_name in ("blocked", "parking")
        if is_blocked:
            blocked_healthy += 1
            stall_len = max(1, tick - int(blocked_since.get(aid, tick)))
            t_cur = tasks_by_id.get(getattr(a, "cur_task", None) or "")
            t_stat = getattr(getattr(t_cur, "status", ""), "value", str(getattr(t_cur, "status", ""))).lower() if t_cur else ""
            if t_stat == "picked":
                max_loaded_stall = max(max_loaded_stall, stall_len)
            # Check active planned path or dropoff target node
            nxt_nodes: list[str] = []
            plan = getattr(sim, "_active_plans", {}).get(aid)
            if plan and getattr(plan, "nodes", None):
                cur_n = getattr(a, "cur_node", None)
                for n_step in plan.nodes:
                    if str(n_step) != str(cur_n):
                        nxt_nodes.append(str(n_step))
                        break
            elif t_cur is not None:
                tgt = str(t_cur.dropoff if t_stat == "picked" else t_cur.pickup)
                if tgt:
                    nxt_nodes.append(tgt)
            for nxt in nxt_nodes:
                occ = node_occupant.get(nxt)
                if occ and occ != aid and occ in wait_adj:
                    wait_adj[aid].add(occ)

    # Detect directed cycles in wait-for graph (pure structural cycle, independent of topology names)
    cycle_nodes: set[str] = set()
    cycle_count = 0
    color: dict[str, int] = {}
    stack: list[str] = []

    def _dfs(u: str) -> None:
        nonlocal cycle_count
        color[u] = 1
        stack.append(u)
        for v in sorted(wait_adj.get(u, ())):
            c = color.get(v, 0)
            if c == 1 and v in stack:
                cycle_count += 1
                cycle_nodes.update(stack[stack.index(v) :])
            elif c == 0:
                _dfs(v)
        stack.pop()
        color[u] = 2

    for u in sorted(wait_adj):
        if color.get(u, 0) == 0:
            _dfs(u)

    # Kitting split-brain skew: max waiting lag for unstarted ops where at least 1 kit task already arrived
    kit_split_lags: list[int] = []
    for op in getattr(tracker, "operations", ()):
        if getattr(op, "started", False) or getattr(op, "completed", False):
            continue
        k_done = sum(
            1
            for kt in getattr(op, "kit_task_ids", ())
            if kt in tasks_by_id
            and getattr(getattr(tasks_by_id[kt], "status", ""), "value", str(getattr(tasks_by_id[kt], "status", ""))).lower()
            == "completed"
        )
        if 0 < k_done < len(getattr(op, "kit_task_ids", ())):
            first_t = op_first_kit_done_tick.setdefault(op.op_id, tick)
            kit_split_lags.append(tick - first_t)
        elif op.op_id in op_first_kit_done_tick and k_done == len(getattr(op, "kit_task_ids", ())):
            op_first_kit_done_tick.pop(op.op_id, None)

    max_kit_split_lag = max(kit_split_lags) if kit_split_lags else 0
    rem_tasks = sum(
        1
        for t in sim.tasks
        if getattr(getattr(t, "status", ""), "value", str(getattr(t, "status", ""))).lower() not in ("completed", "failed")
    )
    zero_progress_ticks = (tick - last_progress_tick) if rem_tasks > 0 else 0
    blocked_ratio = blocked_healthy / max(1, len(healthy_agvs))

    alert = bool(
        rem_tasks > 0
        and (
            (cycle_count >= 1 and zero_progress_ticks >= 40)
            or (blocked_ratio >= 0.75 and max_loaded_stall >= 25 and zero_progress_ticks >= 40)
            or (zero_progress_ticks >= 420 and max_kit_split_lag >= 200)
            or zero_progress_ticks >= 500
        )
    )
    return {
        "tick": tick,
        "rem_tasks": rem_tasks,
        "blocked_healthy": blocked_healthy,
        "blocked_ratio": round(blocked_ratio, 3),
        "max_loaded_stall_ticks": max_loaded_stall,
        "wait_cycle_count": cycle_count,
        "wait_cycle_agvs": sorted(cycle_nodes),
        "max_kit_split_lag": max_kit_split_lag,
        "zero_progress_ticks": zero_progress_ticks,
        "recent_reject_streak": recent_reject_streak,
        "pdg_alert": alert,
    }


def _trace_policy_on_outlier(
    policy_name: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Run one full policy trajectory on a Task-C outlier episode recording tick-by-tick PDG & decision logs."""
    sc_copy = copy.deepcopy(spec["scenario_tuple"])
    p2_copy = copy.deepcopy(spec["p2_cfg"])
    seed = int(spec["seed"])
    max_steps = int(spec["max_steps"])
    kind, ctrl, env = create_discrete_policy_env(
        policy_name,
        topology=str(sc_copy[0]),
        seed=seed,
        max_steps=max_steps,
        use_rust=True,
        p2_cfg=p2_copy,
        allow_missing_pgmpo=False,
    )
    if policy_name == "V18_EA_JOINT":
        raw_ext = make_v18_external_proposer(seq_diff_b2=False, env_ref=env)
    else:
        raw_ext = env.external_proposer

    prefix_steps: list[RecordedStep] = []
    snap_by_tick: dict[int, tuple[NativeSnapshot, CandidateProposal]] = {}

    if raw_ext is not None:
        def _hook(snap: NativeSnapshot, net: Any, *, budget_ms: float = 35.0) -> CandidateProposal:
            prop = raw_ext(snap, net, budget_ms=budget_ms)
            snap_by_tick[int(snap.tick)] = (copy.deepcopy(snap), copy.deepcopy(prop))
            prefix_steps.append(
                RecordedStep(
                    step_before=int(snap.tick),
                    action=-1,
                    triggered_replan=True,
                    proposal=copy.deepcopy(prop),
                )
            )
            return prop

        env.external_proposer = _hook

    obs, info = env.reset(seed=seed, options={"scenario_tuple": sc_copy})
    tm = False
    tr = False
    tick_trace: list[dict[str, Any]] = []
    last_done_t, last_done_o = 0, 0
    last_progress_tick = 0
    op_kit_first_done: dict[str, int] = {}
    reject_streak = 0
    first_alert_tick: Optional[int] = None

    while not (tm or tr):
        t_cur = int(env.current_step)
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
        before_n = len(prefix_steps)
        obs, _, tm, tr, info = env.step(act)
        if len(prefix_steps) > before_n:
            prefix_steps[-1].action = act
            if env.decision_logs:
                last_log = env.decision_logs[-1]
                if not bool(last_log.get("receipt_accepted", False)):
                    reject_streak += 1
                else:
                    reject_streak = 0
        else:
            prefix_steps.append(
                RecordedStep(step_before=t_cur, action=act, triggered_replan=False, proposal=None)
            )

        c_t, c_o = env._completion_counts(int(env.sim._step_cursor))
        if c_t > last_done_t or c_o > last_done_o:
            last_done_t, last_done_o = c_t, c_o
            last_progress_tick = int(env.current_step)

        pdg = compute_pdg_state(
            env.sim,
            env.tracker,
            last_progress_tick=last_progress_tick,
            op_first_kit_done_tick=op_kit_first_done,
            recent_reject_streak=reject_streak,
        )
        if pdg["pdg_alert"] and first_alert_tick is None:
            first_alert_tick = int(pdg["tick"])

        tick_trace.append(
            {
                "tick": int(env.current_step),
                "completed_tasks": int(c_t),
                "completed_ops": int(c_o),
                "total_tasks": len(env.sim.tasks),
                "total_ops": len(env.tracker.operations),
                "blocked_healthy": pdg["blocked_healthy"],
                "max_loaded_stall": pdg["max_loaded_stall_ticks"],
                "wait_cycle_count": pdg["wait_cycle_count"],
                "max_kit_split_lag": pdg["max_kit_split_lag"],
                "zero_progress_ticks": pdg["zero_progress_ticks"],
                "pdg_alert": pdg["pdg_alert"],
                "replan": bool(t_cur in snap_by_tick),
                "action": act,
                "chosen_arm": (snap_by_tick[t_cur][1].telemetry or {}).get("chosen_arm", "") if t_cur in snap_by_tick else "",
                "fallback_keep": (snap_by_tick[t_cur][1].telemetry or {}).get("fallback_keep", None) if t_cur in snap_by_tick else None,
                "receipt_accepted": (env.decision_logs[-1]["receipt_accepted"] if (t_cur in snap_by_tick and env.decision_logs) else None),
                "receipt_reasons": (list(env.decision_logs[-1].get("receipt_reasons", ())) if (t_cur in snap_by_tick and env.decision_logs) else []),
            }
        )

    m = env.compute_episode_metrics()
    aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
    return {
        "policy": policy_name,
        "episode_key": str(spec["episode_key"]),
        "net_obj": round(float(m["net_obj"]), 4),
        "cmax": int(m["cmax"]),
        "task_comp": round(float(m["task_completion_rate"]), 6),
        "prod_comp": round(float(m["production_completion_rate"]), 6),
        "comp100": bool(float(m["task_completion_rate"]) >= 0.9999 and float(m["production_completion_rate"]) >= 0.9999),
        "audit_passed": bool(aud.pass_audit),
        "audit_issues": list(aud.issues),
        "starvation_steps": int(env.tracker.starvation_steps),
        "first_pdg_alert_tick": first_alert_tick,
        "pdg_lead_before_terminal": (int(env.current_step) - first_alert_tick) if first_alert_tick is not None else 0,
        "tick_trace": tick_trace,
        "prefix_steps": prefix_steps,
        "snap_by_tick": snap_by_tick,
        "network": copy.deepcopy(env.sim.network),
    }


def _evaluate_pdg_false_positives_on_val72(specs_val72: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify PDG alert False-Positive Rate on healthy `VAL-72` CoRe-Sched runs."""
    healthy_eps = 0
    healthy_fp_eps = 0
    incomplete_eps = 0
    incomplete_detected_eps = 0
    lead_ticks_incomplete: list[int] = []

    for sp in specs_val72:
        sc_copy = copy.deepcopy(sp["scenario_tuple"])
        p2_copy = copy.deepcopy(sp["p2_cfg"])
        seed = int(sp["seed"])
        max_steps = int(sp["max_steps"])
        kind, ctrl, env = create_discrete_policy_env(
            "V19_ADAPTIVE_RL", topology=str(sc_copy[0]), seed=seed, max_steps=max_steps, use_rust=True, p2_cfg=p2_copy
        )
        obs, info = env.reset(seed=seed, options={"scenario_tuple": sc_copy})
        tm = False
        tr = False
        last_t, last_o, last_prog = 0, 0, 0
        kit_done: dict[str, int] = {}
        first_alert: Optional[int] = None
        while not (tm or tr):
            f = compute_expert_v9_state(
                env.sim, env.tracker, last_replan_tick=env.last_replan_tick,
                recent_conflict_ticks=env.recent_conflict_ticks, decision_logs=env.decision_logs,
            )
            m_a = compute_expert_v10_action_mask(env.sim, env.last_replan_tick)
            act = int(ctrl.select_action(f, dict(info, action_mask=m_a), sim=env.sim))
            obs, _, tm, tr, info = env.step(act)
            ct, co = env._completion_counts(int(env.sim._step_cursor))
            if ct > last_t or co > last_o:
                last_t, last_o = ct, co
                last_prog = int(env.current_step)
            pdg = compute_pdg_state(
                env.sim, env.tracker, last_progress_tick=last_prog, op_first_kit_done_tick=kit_done, recent_reject_streak=0
            )
            if pdg["pdg_alert"] and first_alert is None:
                first_alert = int(pdg["tick"])

        m = env.compute_episode_metrics()
        comp100 = bool(float(m["task_completion_rate"]) >= 0.9999 and float(m["production_completion_rate"]) >= 0.9999)
        if comp100:
            healthy_eps += 1
            if first_alert is not None:
                healthy_fp_eps += 1
        else:
            incomplete_eps += 1
            if first_alert is not None:
                incomplete_detected_eps += 1
                lead_ticks_incomplete.append(int(env.current_step) - first_alert)

    return {
        "val72_clean_healthy_episodes": healthy_eps,
        "val72_healthy_false_positive_episodes": healthy_fp_eps,
        "val72_healthy_fpr": round(healthy_fp_eps / max(1, healthy_eps), 4),
        "val72_incomplete_episodes": incomplete_eps,
        "val72_incomplete_detected": incomplete_detected_eps,
        "val72_incomplete_mean_lead_ticks": round(float(np.mean(lead_ticks_incomplete)), 1) if lead_ticks_incomplete else None,
    }


def run_task_c() -> dict[str, Any]:
    """Execute Task C forensics strictly on Cohort B outliers (`37302`, `3906`) + `VAL-72` PDG FPR."""
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    specs_c = build_task_c_outlier_specs()
    policies_c = ("V19_ADAPTIVE_RL", "V18_EA_JOINT", "IJCAI26_PGMPO", "FGCS26_REN_GNN")

    expert_eval = ExpertSuiteEvaluator(seed=0)
    phase0_p03 = Phase0RestoredProposer(
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
    )

    episode_reports: dict[str, Any] = {}
    divergence_csv_rows: list[dict[str, Any]] = []

    for sp in specs_c:
        ek = str(sp["episode_key"])
        print(f"[Task C] Tracing 4 policies on {ek}...", flush=True)
        traces_by_pol: dict[str, dict[str, Any]] = {}
        for pol in policies_c:
            traces_by_pol[pol] = _trace_policy_on_outlier(pol, sp)

        core_t = traces_by_pol["V19_ADAPTIVE_RL"]
        v18_t = traces_by_pol["V18_EA_JOINT"]
        core_rows = core_t["tick_trace"]
        v18_rows = v18_t["tick_trace"]
        min_len = min(len(core_rows), len(v18_rows))

        # Locate first divergence epoch t*
        t_star = 0
        div_reason = "none"
        for i in range(min_len):
            cr = core_rows[i]
            vr = v18_rows[i]
            if (vr["completed_tasks"] - cr["completed_tasks"]) >= 2 or (
                cr["blocked_healthy"] >= 2 and vr["blocked_healthy"] == 0 and cr["tick"] >= 16
            ):
                t_star = int(cr["tick"])
                div_reason = (
                    f"task_progress_lag(v18={vr['completed_tasks']},core={cr['completed_tasks']})"
                    if (vr["completed_tasks"] - cr["completed_tasks"]) >= 2
                    else "multi_amr_stall"
                )
                break
        if t_star == 0:
            replan_ticks = sorted(core_t["snap_by_tick"].keys())
            t_star = int(replan_ticks[len(replan_ticks) // 3]) if replan_ticks else 32
            div_reason = "mid_episode_fallback"

        alert_tick = core_t.get("first_pdg_alert_tick")
        all_rt = sorted(core_t["snap_by_tick"].keys())
        win_lo = max(0, t_star - 16)
        win_hi = t_star + 32
        probe_ticks = [tk for tk in all_rt if win_lo <= tk <= win_hi]
        if alert_tick is not None:
            late_probes = [tk for tk in all_rt if (int(alert_tick) - 25) <= tk <= (int(alert_tick) + 5)]
            probe_ticks = sorted(set(probe_ticks[:2]) | set(late_probes[:2]))
        if not probe_ticks:
            probe_ticks = [min(all_rt, key=lambda x: abs(x - t_star))] if all_rt else []

        counterfactual_probes: list[dict[str, Any]] = []
        for tk in probe_ticks[:4]:
            snap_tk, prop_core_tk = core_t["snap_by_tick"][tk]
            net_tk = core_t["network"]
            clf_tk = classify_commitments(snap_tk)

            # Match recorded step action at tk
            act_tk = next((st.action for st in core_t["prefix_steps"] if st.step_before == tk and st.triggered_replan), 4)
            arts = expert_eval.evaluate_all_experts(snap_tk, net_tk)
            p03_prop = phase0_p03(snap_tk, net_tk, budget_ms=35.0)

            expert_b2_table: dict[str, Any] = {}
            for eid in EXPERT_IDS:
                a = arts[eid]
                expert_b2_table[eid] = {
                    "valid": bool(a.eval_res_b2 and a.eval_res_b2.valid),
                    "ja_unified": round(float(a.ja_unified_score), 2) if a.ja_unified_score is not None else None,
                    "b2_obj": round(float(a.eval_res_b2.objective), 2) if a.eval_res_b2 else None,
                    "starvation": int(a.eval_res_b2.starvation_ticks) if a.eval_res_b2 else None,
                }

            # Run counterfactual FULL episode rescue probe using best non-KEEP oracle expert + Phase0 P0.3
            best_rescue_eid = min(
                [e for e in EXPERT_IDS if e != "E0_KEEP"],
                key=lambda e: float(arts[e].ja_unified_score or 1e12),
            )
            best_art = arts[best_rescue_eid]
            rescue_prop = best_art.to_proposal(
                policy_id=f"CF_{best_rescue_eid}",
                snapshot=snap_tk,
                clf=clf_tk,
                eval_res=best_art.eval_res_b2,
                eval_cnt=1,
                t0=0.0,
                budget_ms=35.0,
                fallback_keep=False,
                include_eval_seq_when_unwindowed=True,
            )
            if best_art.live_repairs:
                rescue_prop.proposed_live_repairs = dict(best_art.live_repairs)
            if best_art.bottleneck_locks:
                rescue_prop.proposed_bottleneck_locks = dict(best_art.bottleneck_locks)

            cf_core = rollout_48_and_optional_full(
                sp, core_t["prefix_steps"], tk, prop_core_tk, action_at_target=act_tk, rollout_horizon=48, full_continuation=True
            )
            cf_oracle = rollout_48_and_optional_full(
                sp, core_t["prefix_steps"], tk, rescue_prop, action_at_target=act_tk, rollout_horizon=48, full_continuation=True
            )
            cf_p03 = rollout_48_and_optional_full(
                sp, core_t["prefix_steps"], tk, p03_prop, action_at_target=act_tk, rollout_horizon=48, full_continuation=True
            )

            counterfactual_probes.append(
                {
                    "tick": tk,
                    "core_chosen_arm": (prop_core_tk.telemetry or {}).get("chosen_arm"),
                    "core_fallback_keep": (prop_core_tk.telemetry or {}).get("fallback_keep"),
                    "best_oracle_expert": best_rescue_eid,
                    "expert_scores": expert_b2_table,
                    "full_rescue_core_v121": cf_core,
                    "full_rescue_oracle_expert": cf_oracle,
                    "full_rescue_phase0_p03": cf_p03,
                }
            )

        for cr in core_rows:
            if win_lo <= cr["tick"] <= max(win_hi, t_star + 60) or cr["replan"] or cr["pdg_alert"]:
                divergence_csv_rows.append(
                    {
                        "episode_key": ek,
                        "tick": cr["tick"],
                        "completed_tasks": cr["completed_tasks"],
                        "completed_ops": cr["completed_ops"],
                        "blocked_healthy": cr["blocked_healthy"],
                        "max_loaded_stall": cr["max_loaded_stall"],
                        "wait_cycle_count": cr["wait_cycle_count"],
                        "max_kit_split_lag": cr["max_kit_split_lag"],
                        "zero_progress_ticks": cr["zero_progress_ticks"],
                        "pdg_alert": cr["pdg_alert"],
                        "replan": cr["replan"],
                        "chosen_arm": cr["chosen_arm"],
                        "receipt_accepted": cr["receipt_accepted"],
                    }
                )

        episode_reports[ek] = {
            "divergence_tick_t_star": t_star,
            "divergence_reason": div_reason,
            "policy_summaries": {
                pol: {
                    "net_obj": d["net_obj"],
                    "cmax": d["cmax"],
                    "task_comp": d["task_comp"],
                    "prod_comp": d["prod_comp"],
                    "comp100": d["comp100"],
                    "audit_passed": d["audit_passed"],
                    "audit_issues": d["audit_issues"],
                    "starvation_steps": d["starvation_steps"],
                    "first_pdg_alert_tick": d["first_pdg_alert_tick"],
                    "pdg_lead_before_terminal": d["pdg_lead_before_terminal"],
                }
                for pol, d in traces_by_pol.items()
            },
            "counterfactual_probes_around_t_star": counterfactual_probes,
        }

    print("[Task C] Measuring PDG false-positive rate on healthy VAL-72 episodes...", flush=True)
    specs_val = build_val72_specs()
    assert_stage1_quarantine([sp["seed"] for sp in specs_val], label="TaskC_VAL72_FPR")
    pdg_val72 = _evaluate_pdg_false_positives_on_val72(specs_val)

    payload = {
        "schema": "if-core-v20-01.task_c.v1",
        "episodes": episode_reports,
        "pdg_generalization_on_val72": pdg_val72,
    }
    (OUT_AUDIT_DIR / "task_c_liveness_forensics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if divergence_csv_rows:
        with (OUT_AUDIT_DIR / "task_c_divergence_trace.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(divergence_csv_rows[0].keys()))
            w.writeheader()
            w.writerows(divergence_csv_rows)

    md = ["# Task C: Liveness Failure Forensics & Topology-Agnostic PDG Indicators\n"]
    for ek, er in episode_reports.items():
        md.append(f"## Episode `{ek}` (`t* = {er['divergence_tick_t_star']}`, `{er['divergence_reason']}`)\n")
        md.append("| Policy | Net Obj | Cmax | Comp100 | Audit Pass | Starvation | First PDG Alert (Lead) |")
        md.append("|---|---:|---:|---:|---:|---:|---:|")
        for pol, ps in er["policy_summaries"].items():
            md.append(
                f"| {pol} | {ps['net_obj']:.2f} | {ps['cmax']} | {ps['comp100']} ({ps['task_comp']:.2f}/{ps['prod_comp']:.2f}) | "
                f"{ps['audit_passed']} | {ps['starvation_steps']} | {ps['first_pdg_alert_tick']} (+{ps['pdg_lead_before_terminal']}t) |"
            )
        md.append("\n### Counterfactual Frozen-State Probes around `t*`")
        for pr in er["counterfactual_probes_around_t_star"]:
            md.append(
                f"- **Tick `{pr['tick']}`** (`CoRe` chose `{pr['core_chosen_arm']}`, keep=`{pr['core_fallback_keep']}`): "
                f"Best Oracle = `{pr['best_oracle_expert']}` -> Full rescue: "
                f"`v1.2.1` comp=`{pr['full_rescue_core_v121'].get('full_comp100')}` (`net_obj={pr['full_rescue_core_v121'].get('full_net_obj')}`), "
                f"`Oracle` comp=`{pr['full_rescue_oracle_expert'].get('full_comp100')}` (`net_obj={pr['full_rescue_oracle_expert'].get('full_net_obj')}`), "
                f"`Phase0 P0.3` comp=`{pr['full_rescue_phase0_p03'].get('full_comp100')}` (`net_obj={pr['full_rescue_phase0_p03'].get('full_net_obj')}`)"
            )
        md.append("")
    md.append("## PDG Generalization on Healthy `VAL-72` Episodes\n")
    md.append(
        f"- Clean Healthy `VAL-72` False Positive Rate: `{pdg_val72['val72_healthy_false_positive_episodes']}/{pdg_val72['val72_clean_healthy_episodes']}` (`FPR = {pdg_val72['val72_healthy_fpr']}`)"
    )
    (OUT_AUDIT_DIR / "task_c_liveness_forensics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    run_task_c()
