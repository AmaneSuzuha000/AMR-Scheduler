"""Task 2: Planning-Execution Mismatch Audit (`U0_Orig_CPSA` + fixed `R1_Robust`) on DEV 6001-6003."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from experiments.res_guided_avoidance.gate1_headroom_audit import m2_res_margin_selector
from experiments.res_guided_avoidance.metric_world import (
    MultiRobotEpisodeResult,
    build_grid_metric_map,
    build_occluded_junction_map,
    build_workshop_metric_map,
)
from experiments.stp_research_reset.audit_env import ExecutionAuditSim, LegTelemetry

OUT_DIR = REPO_ROOT / "out" / "stp_research_reset"
DEV_SEEDS = (6001, 6002, 6003)


def _stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(len(arr)),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    a = np.asarray(xs, dtype=float)
    b = np.asarray(ys, dtype=float)
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


def run_suite() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    om = build_occluded_junction_map(cell_m=2.6)
    ma = build_grid_metric_map("Map_A", cell_m=2.6)
    mb = build_grid_metric_map("Map_B", cell_m=2.6)
    wm = build_workshop_metric_map()

    specs = [
        # Main DEV scenarios matching robust_ramp_study
        ("Occluded_Jct_6R_Clean", "main", om, 6, 12, 70.0, 2, 0.0, 0.0, 0.0, 0.0, 0.15),
        ("MapA_8R_Clean", "main", ma, 8, 16, 80.0, 2, 0.0, 0.0, 0.0, 0.0, 0.15),
        ("MapB_12R_Clean", "main", mb, 12, 20, 85.0, 2, 0.0, 0.0, 0.0, 0.0, 0.15),
        ("Workshop_8R_Clean", "main", wm, 8, 12, 160.0, 1, 0.0, 0.0, 0.0, 0.0, 0.15),
        ("Occluded_Jct_6R_Noise0.12", "main", om, 6, 12, 70.0, 2, 0.12, 0.20, 0.0, 0.0, 0.15),
        ("Occluded_Jct_6R_Compound", "main", om, 6, 12, 70.0, 2, 0.12, 0.20, 2.0, 0.10, 0.15),
        # Zero-disturbance & intruder-free structural mismatch contrasts
        ("Occluded_Jct_6R_NoSlow", "contrast", om, 6, 12, 70.0, 2, 0.0, 0.0, 0.0, 0.0, 0.0),
        ("Occluded_Jct_6R_PureStruct", "contrast", om, 6, 12, 70.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ("MapA_8R_PureStruct", "contrast", ma, 8, 16, 80.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ("MapB_12R_PureStruct", "contrast", mb, 12, 20, 85.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ("Workshop_8R_PureStruct", "contrast", wm, 8, 12, 160.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]

    ep_rows: list[dict[str, Any]] = []
    sc_summaries: dict[str, Any] = {}
    all_main_legs: list[tuple[float, LegTelemetry]] = []
    all_struct_legs: list[tuple[float, LegTelemetry]] = []

    for label, group, mmap, nr, nt, hz, ni, np_m, nv_m, lag_s, drop_p, slow_p in specs:
        sc_ep_rows: list[dict[str, Any]] = []
        sc_comp_legs: list[LegTelemetry] = []
        sc_all_legs: list[LegTelemetry] = []
        sc_wp_deltas: list[float] = []

        for sd in DEV_SEEDS:
            sim = ExecutionAuditSim(
                mmap,
                upper_mode="U0_Orig_CPSA",
                n_robots=nr,
                n_tasks=nt,
                horizon_s=hz,
                seed=sd,
                n_intruders=ni,
                obs_noise_pos_m=np_m,
                obs_noise_vel_mps=nv_m,
                reservation_lag_s=lag_s,
                reservation_dropout=drop_p,
                robust_safety=True,
                disturb_slow_prob=slow_p,
            )
            res: MultiRobotEpisodeResult = sim.run_episode(m2_res_margin_selector, controller_name="U0+R1_Robust")
            legs = list(sim.closed_legs)
            comp_legs = [lg for lg in legs if lg.completed and lg.end_t_s is not None]
            sc_all_legs.extend(legs)
            sc_comp_legs.extend(comp_legs)
            for lg in comp_legs:
                if group == "main":
                    all_main_legs.append((mmap.macro_step_s, lg))
                else:
                    all_struct_legs.append((mmap.macro_step_s, lg))
            for lg in legs:
                for p_s, a_s in zip(lg.waypoint_plan_s[1:], lg.waypoint_actual_s[1:]):
                    if a_s is not None:
                        sc_wp_deltas.append(float(a_s - p_s))

            sh_tot = res.shield_interventions
            row = {
                "scenario": label,
                "group": group,
                "seed": sd,
                "macro_step_s": mmap.macro_step_s,
                "completed_tasks": res.completed_tasks,
                "n_tasks": nt,
                "makespan_s": res.makespan_s,
                "zero_collisions": res.zero_collisions,
                "independent_collision_events": sim.independent_collision_events,
                "wall_col_ticks": res.wall_collisions,
                "peer_col_ticks": res.peer_collisions,
                "intr_col_ticks": res.intruder_collisions,
                "wait_ratio": res.wait_ratio,
                "replans": res.replans,
                "eta_drift_replans": sim.eta_drift_replans,
                "eta_drift_pct": round(100.0 * sim.eta_drift_replans / max(1, res.replans), 2),
                "init_plan_failures": sim.init_plan_failures,
                "row_gate_events": sim.row_gate_events,
                "shield_interventions": sh_tot,
                "shield_wall_events": sim.shield_wall_events,
                "shield_peer_events": sim.shield_peer_events,
                "shield_intr_events": sim.shield_intr_events,
                "shield_other_events": sim.shield_other_events,
                "spillover_peer_waits": sim.spillover_peer_waits,
                "spillover_peer_replans": sim.spillover_peer_replans,
                "total_legs": len(legs),
                "completed_legs": len(comp_legs),
                "truncated_legs": sum(1 for lg in legs if lg.truncated),
            }
            ep_rows.append(row)
            sc_ep_rows.append(row)
            print(
                f"[Mismatch | {label:26s} | s={sd}] comp={res.completed_tasks}/{nt} "
                f"repl={res.replans}(drift={sim.eta_drift_replans}) row={sim.row_gate_events} "
                f"sh={sh_tot}(W={sim.shield_wall_events}/P={sim.shield_peer_events}/I={sim.shield_intr_events}) "
                f"spill_r={sim.spillover_peer_replans}",
                flush=True,
            )

        dt_arr = [float(lg.end_t_s - lg.planned_eta_s) for lg in sc_comp_legs if lg.end_t_s is not None]
        dur_ratios = [
            float((lg.end_t_s - lg.plan_t_s) / lg.planned_dur_s)
            for lg in sc_comp_legs
            if lg.end_t_s is not None and lg.planned_dur_s > 0.25
        ]
        len_ratios = [
            float(lg.actual_dist_m / lg.planned_len_m)
            for lg in sc_comp_legs
            if lg.planned_len_m > 0.25
        ]
        straight_dt = [
            float(lg.end_t_s - lg.planned_eta_s)
            for lg in sc_comp_legs
            if lg.end_t_s is not None and lg.sharp_turns == 0 and lg.init_hdg_err_rad < math.pi / 4
        ]
        turning_dt = [
            float(lg.end_t_s - lg.planned_eta_s)
            for lg in sc_comp_legs
            if lg.end_t_s is not None and (lg.sharp_turns > 0 or lg.init_hdg_err_rad >= math.pi / 4)
        ]
        straight_ratio = [
            float((lg.end_t_s - lg.plan_t_s) / lg.planned_dur_s)
            for lg in sc_comp_legs
            if lg.end_t_s is not None and lg.planned_dur_s > 0.25 and lg.sharp_turns == 0 and lg.init_hdg_err_rad < math.pi / 4
        ]
        turning_ratio = [
            float((lg.end_t_s - lg.plan_t_s) / lg.planned_dur_s)
            for lg in sc_comp_legs
            if lg.end_t_s is not None and lg.planned_dur_s > 0.25 and (lg.sharp_turns > 0 or lg.init_hdg_err_rad >= math.pi / 4)
        ]
        slow_fracs = [
            float(lg.slow_steps) / max(1.0, float((lg.end_t_s - lg.plan_t_s) / 0.25))
            for lg in sc_comp_legs
            if lg.end_t_s is not None and lg.end_t_s > lg.plan_t_s
        ]

        macro_s = mmap.macro_step_s
        n_comp = max(1, len(dt_arr))
        n_wp = max(1, len(sc_wp_deltas))
        tot_sh = sum(r["shield_interventions"] for r in sc_ep_rows)
        sc_summaries[label] = {
            "group": group,
            "macro_step_s": macro_s,
            "episodes": len(sc_ep_rows),
            "mean_completed_tasks": round(float(np.mean([r["completed_tasks"] for r in sc_ep_rows])), 3),
            "n_tasks": nt,
            "zero_col_episodes": f"{sum(1 for r in sc_ep_rows if r['zero_collisions'])}/{len(sc_ep_rows)}",
            "mean_wait_ratio": round(float(np.mean([r["wait_ratio"] for r in sc_ep_rows])), 4),
            "replans_tot": int(sum(r["replans"] for r in sc_ep_rows)),
            "eta_drift_replans_tot": int(sum(r["eta_drift_replans"] for r in sc_ep_rows)),
            "eta_drift_share_pct": round(100.0 * sum(r["eta_drift_replans"] for r in sc_ep_rows) / max(1, sum(r["replans"] for r in sc_ep_rows)), 2),
            "row_gate_events_tot": int(sum(r["row_gate_events"] for r in sc_ep_rows)),
            "spillover_peer_waits_tot": int(sum(r["spillover_peer_waits"] for r in sc_ep_rows)),
            "spillover_peer_replans_tot": int(sum(r["spillover_peer_replans"] for r in sc_ep_rows)),
            "spillover_wait_share_pct": round(100.0 * sum(r["spillover_peer_waits"] for r in sc_ep_rows) / max(1, sum(r["row_gate_events"] for r in sc_ep_rows)), 2),
            "spillover_replan_share_pct": round(100.0 * sum(r["spillover_peer_replans"] for r in sc_ep_rows) / max(1, sum(r["replans"] for r in sc_ep_rows)), 2),
            "shield_interventions_tot": int(tot_sh),
            "shield_wall_pct": round(100.0 * sum(r["shield_wall_events"] for r in sc_ep_rows) / max(1, tot_sh), 2),
            "shield_peer_pct": round(100.0 * sum(r["shield_peer_events"] for r in sc_ep_rows) / max(1, tot_sh), 2),
            "shield_intr_pct": round(100.0 * sum(r["shield_intr_events"] for r in sc_ep_rows) / max(1, tot_sh), 2),
            "leg_eta_error_s": _stats(dt_arr),
            "leg_pct_early_gt_1s": round(100.0 * sum(1 for x in dt_arr if x < -1.0) / n_comp, 2),
            "leg_pct_late_gt_1s": round(100.0 * sum(1 for x in dt_arr if x > 1.0) / n_comp, 2),
            "leg_pct_late_gt_1macro": round(100.0 * sum(1 for x in dt_arr if x > macro_s) / n_comp, 2),
            "leg_pct_late_gt_3macro": round(100.0 * sum(1 for x in dt_arr if x > 3.0 * macro_s) / n_comp, 2),
            "wp_eta_error_s": _stats(sc_wp_deltas),
            "wp_pct_late_gt_1macro": round(100.0 * sum(1 for x in sc_wp_deltas if x > macro_s) / n_wp, 2),
            "wp_pct_early_gt_1s": round(100.0 * sum(1 for x in sc_wp_deltas if x < -1.0) / n_wp, 2),
            "actual_vs_plan_dur_ratio": _stats(dur_ratios),
            "actual_vs_plan_len_ratio": _stats(len_ratios),
            "straight_vs_turn_leg_eta": {
                "straight_n": len(straight_dt),
                "straight_dt_mean_s": round(float(np.mean(straight_dt)), 4) if straight_dt else 0.0,
                "straight_dur_ratio_mean": round(float(np.mean(straight_ratio)), 4) if straight_ratio else 0.0,
                "turning_n": len(turning_dt),
                "turning_dt_mean_s": round(float(np.mean(turning_dt)), 4) if turning_dt else 0.0,
                "turning_dur_ratio_mean": round(float(np.mean(turning_ratio)), 4) if turning_ratio else 0.0,
            },
            "mean_sharp_turns_per_leg": round(float(np.mean([lg.sharp_turns for lg in sc_comp_legs])) if sc_comp_legs else 0.0, 3),
            "mean_init_hdg_err_deg": round(float(np.mean([math.degrees(lg.init_hdg_err_rad) for lg in sc_comp_legs])) if sc_comp_legs else 0.0, 2),
            "slow_step_fraction": round(float(np.mean(slow_fracs)) if slow_fracs else 0.0, 4),
            "correlations_with_dt_arr": {
                "r_sharp_turns": _pearson([float(lg.sharp_turns) for lg in sc_comp_legs], dt_arr),
                "r_init_hdg_err": _pearson([float(lg.init_hdg_err_rad) for lg in sc_comp_legs], dt_arr),
                "r_planned_len_m": _pearson([float(lg.planned_len_m) for lg in sc_comp_legs], dt_arr),
                "r_row_pause_steps": _pearson([float(lg.row_pause_steps) for lg in sc_comp_legs], dt_arr),
            },
        }

    # Global correlation summary across main DEV suite vs pure-structural contrast
    def _pool_summary(items: list[tuple[float, LegTelemetry]]) -> dict[str, Any]:
        dts = [float(lg.end_t_s - lg.planned_eta_s) for _, lg in items if lg.end_t_s is not None]
        if not items:
            return {}
        return {
            "n_completed_legs": len(dts),
            "dt_arr_s": _stats(dts),
            "pct_late_gt_1s": round(100.0 * sum(1 for x in dts if x > 1.0) / max(1, len(dts)), 2),
            "pct_early_gt_1s": round(100.0 * sum(1 for x in dts if x < -1.0) / max(1, len(dts)), 2),
            "pct_late_gt_1macro": round(100.0 * sum(1 for (ms, lg), x in zip(items, dts) if x > ms) / max(1, len(dts)), 2),
            "r_sharp_turns": _pearson([float(lg.sharp_turns) for _, lg in items], dts),
            "r_init_hdg_err": _pearson([float(lg.init_hdg_err_rad) for _, lg in items], dts),
            "r_planned_len_m": _pearson([float(lg.planned_len_m) for _, lg in items], dts),
            "r_row_pause_steps": _pearson([float(lg.row_pause_steps) for _, lg in items], dts),
        }

    payload = {
        "protocol": "Execution-Aware-STP-Audit Task 2 (DEV 6001-6003, Zero RL, Fixed R1_Robust)",
        "seeds": list(DEV_SEEDS),
        "global_pool": {
            "main_dev": _pool_summary(all_main_legs),
            "pure_struct_contrast": _pool_summary(all_struct_legs),
        },
        "scenarios": sc_summaries,
        "episode_rows": ep_rows,
    }
    json_path = OUT_DIR / "planning_execution_mismatch_report.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Build concise Markdown report
    lines = [
        "# Task 2 — Planning–Execution Mismatch Audit (`U0_Orig_CPSA` + Fixed `R1_Robust`)",
        "",
        f"- DEV Seeds: `{list(DEV_SEEDS)}` (blind `8001–8010` untouched; zero RL training)",
        "",
        "## 1. Per-Scenario Structural & Disturbance Breakdown",
        "",
        "| Scenario | Comp | Wait | Replans (Drift%) | ROW Stops (Spill%) | Shield (W/P/I%) | Leg dETA Mean / P90 (s) | Late>1s / Early<-1s (%) | Straight vs Turn dETA (s) | Dur Ratio (Act/Plan) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, _, *_ in specs:
        s = sc_summaries[label]
        sh_str = f"{s['shield_interventions_tot']} ({s['shield_wall_pct']:.0f}/{s['shield_peer_pct']:.0f}/{s['shield_intr_pct']:.0f})"
        eta_s = f"{s['leg_eta_error_s']['mean']:+.2f} / {s['leg_eta_error_s']['p90']:+.2f}"
        le_s = f"{s['leg_pct_late_gt_1s']:.1f}% / {s['leg_pct_early_gt_1s']:.1f}%"
        st_s = f"{s['straight_vs_turn_leg_eta']['straight_dt_mean_s']:+.2f} vs {s['straight_vs_turn_leg_eta']['turning_dt_mean_s']:+.2f}"
        lines.append(
            f"| {label} | {s['mean_completed_tasks']:.2f}/{s['n_tasks']} | {s['mean_wait_ratio']:.3f} | "
            f"{s['replans_tot']} ({s['eta_drift_share_pct']:.1f}%) | {s['row_gate_events_tot']} ({s['spillover_wait_share_pct']:.1f}%) | "
            f"{sh_str} | {eta_s} | {le_s} | {st_s} | {s['actual_vs_plan_dur_ratio']['mean']:.2f}x |"
        )

    lines.extend([
        "",
        "## 2. Structural vs Stochastic Mismatch Decomposition",
        "",
    ])
    for pool_name, pk in [("Main DEV Suite", "main_dev"), ("Pure-Structural Contrast (disturb=0, intr=0)", "pure_struct_contrast")]:
        ps = payload["global_pool"][pk]
        if ps:
            lines.append(
                f"- **{pool_name}** (`N={ps['n_completed_legs']}` completed legs): "
                f"`dETA mean={ps['dt_arr_s']['mean']:+.2f}s` (`p50={ps['dt_arr_s']['p50']:+.2f}s`, `p90={ps['dt_arr_s']['p90']:+.2f}s`, `max={ps['dt_arr_s']['max']:+.2f}s`), "
                f"`late>1s={ps['pct_late_gt_1s']:.1f}%`, `early<-1s={ps['pct_early_gt_1s']:.1f}%`, "
                f"`r(turns)={ps['r_sharp_turns']:+.3f}`, `r(init_hdg)={ps['r_init_hdg_err']:+.3f}`, "
                f"`r(len)={ps['r_planned_len_m']:+.3f}`, `r(row)={ps['r_row_pause_steps']:+.3f}`."
            )

    md_path = OUT_DIR / "planning_execution_mismatch_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Task2] Wrote {md_path} and {json_path}", flush=True)
    return payload


if __name__ == "__main__":
    run_suite()
