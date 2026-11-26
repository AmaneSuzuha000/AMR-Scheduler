"""Task 3: Upper-Level Candidate Headroom Audit (`U0..U4` + fixed `R1_Robust`) on DEV 6001-6003."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.stats import wilcoxon

from experiments.res_guided_avoidance.gate1_headroom_audit import m2_res_margin_selector
from experiments.res_guided_avoidance.metric_world import (
    MultiRobotEpisodeResult,
    build_grid_metric_map,
    build_occluded_junction_map,
    build_workshop_metric_map,
)
from experiments.stp_research_reset.audit_env import ExecutionAuditSim
from experiments.stp_research_reset.upper_planners import (
    U4CounterfactualOracle,
    make_u1b_plan_fn,
    make_u2_fixed_buffer_fn,
    u0_buf1_plan_fn,
    u1a_plan_fn,
    u3_state_dep_buffer_fn,
)

OUT_DIR = REPO_ROOT / "out" / "stp_research_reset"
DEV_SEEDS = (6001, 6002, 6003)


def _paired_wilcoxon(a: list[float], b: list[float]) -> float:
    diffs = [x - y for x, y in zip(a, b) if abs(x - y) > 1e-9]
    if len(diffs) < 2:
        return 1.0
    try:
        _, p = wilcoxon(diffs)
        return round(float(p), 4)
    except Exception:
        return 1.0


def _summarize_variant(rows: list[dict[str, Any]], u0_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(rows))
    comps = [float(r["completed_tasks"]) for r in rows]
    u0_comps = [float(r["completed_tasks"]) for r in u0_rows]
    waits = [float(r["wait_ratio"]) for r in rows]
    u0_waits = [float(r["wait_ratio"]) for r in u0_rows]
    detas = [abs(float(r["mean_deta_s"])) for r in rows if r["mean_deta_s"] is not None]
    return {
        "episodes": len(rows),
        "zero_col_cnt": f"{sum(1 for r in rows if r['zero_collisions'])}/{len(rows)}",
        "zero_col_pct": round(100.0 * sum(1 for r in rows if r["zero_collisions"]) / n, 1),
        "completed_tasks_mean": round(float(np.mean(comps)), 3),
        "completed_vs_u0_pct": round(100.0 * (float(np.mean(comps)) - float(np.mean(u0_comps))) / max(1e-6, float(np.mean(u0_comps))), 2),
        "p_comp_vs_u0": _paired_wilcoxon(comps, u0_comps),
        "wait_ratio_mean": round(float(np.mean(waits)), 4),
        "p_wait_vs_u0": _paired_wilcoxon(waits, u0_waits),
        "replans_tot": int(sum(r["replans"] for r in rows)),
        "eta_drift_replans_tot": int(sum(r["eta_drift_replans"] for r in rows)),
        "row_gate_events_tot": int(sum(r["row_gate_events"] for r in rows)),
        "shield_interventions_tot": int(sum(r["shield_interventions"] for r in rows)),
        "abs_deta_mean_s": round(float(np.mean(detas)) if detas else 0.0, 3),
        "plan_ms_mean": round(float(np.mean([r["plan_ms_mean"] for r in rows])), 3),
    }


def run_suite() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    om = build_occluded_junction_map(cell_m=2.6)
    ma = build_grid_metric_map("Map_A", cell_m=2.6)
    mb = build_grid_metric_map("Map_B", cell_m=2.6)
    wm = build_workshop_metric_map()

    specs = [
        ("Occluded_Jct_6R_Clean", om, 6, 12, 70.0, 2, 0.0, 0.0, 0.0, 0.0),
        ("MapA_8R_Clean", ma, 8, 16, 80.0, 2, 0.0, 0.0, 0.0, 0.0),
        ("MapB_12R_Clean", mb, 12, 20, 85.0, 2, 0.0, 0.0, 0.0, 0.0),
        ("Workshop_8R_Clean", wm, 8, 12, 160.0, 1, 0.0, 0.0, 0.0, 0.0),
        ("Occluded_Jct_6R_Noise0.12", om, 6, 12, 70.0, 2, 0.12, 0.20, 0.0, 0.0),
        ("Occluded_Jct_6R_Compound", om, 6, 12, 70.0, 2, 0.12, 0.20, 2.0, 0.10),
    ]

    variants = [
        ("U0_Orig_CPSA", 1, None),
        ("U0_Buf1_Macro", 1, u0_buf1_plan_fn),
        ("U1a_EdgeSpan_Fix", 1, u1a_plan_fn),
        ("U1b_KinoTime_CPSA", 4, make_u1b_plan_fn(gamma=4.0, turn_cost_s=0.5)),
        ("U2_FixedBuf_1.0s", 4, make_u2_fixed_buffer_fn(1.0)),
        ("U2_FixedBuf_2.0s", 4, make_u2_fixed_buffer_fn(2.0)),
        ("U3_StateDep_Buf", 4, u3_state_dep_buffer_fn),
    ]

    ep_rows: list[dict[str, Any]] = []
    by_var: dict[str, list[dict[str, Any]]] = {v[0]: [] for v in variants}

    t_start = time.perf_counter()
    for sc_label, mmap, nr, nt, hz, ni, np_m, nv_m, lag_s, drop_p in specs:
        for vname, div, cfn in variants:
            for sd in DEV_SEEDS:
                sim = ExecutionAuditSim(
                    mmap,
                    upper_mode=vname,
                    sub_tick_div=div,
                    custom_plan_fn=cfn,
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
                )
                res: MultiRobotEpisodeResult = sim.run_episode(m2_res_margin_selector, controller_name=f"{vname}+R1_Robust")
                comp_legs = [lg for lg in sim.closed_legs if lg.completed and lg.end_t_s is not None]
                detas = [float(lg.end_t_s - lg.planned_eta_s) for lg in comp_legs if lg.end_t_s is not None]
                row = {
                    "scenario": sc_label,
                    "variant": vname,
                    "seed": sd,
                    "completed_tasks": res.completed_tasks,
                    "n_tasks": nt,
                    "makespan_s": res.makespan_s,
                    "zero_collisions": res.zero_collisions,
                    "col_ticks": res.wall_collisions + res.peer_collisions + res.intruder_collisions,
                    "wait_ratio": res.wait_ratio,
                    "replans": res.replans,
                    "eta_drift_replans": sim.eta_drift_replans,
                    "row_gate_events": sim.row_gate_events,
                    "shield_interventions": res.shield_interventions,
                    "mean_deta_s": round(float(np.mean(detas)), 3) if detas else None,
                    "completed_legs": len(comp_legs),
                    "plan_ms_mean": round(float(np.mean(sim._plan_wall_ms_samples)) if sim._plan_wall_ms_samples else 0.0, 3),
                }
                ep_rows.append(row)
                by_var[vname].append(row)
                print(
                    f"[Task3 | {sc_label:24s} | {vname:18s} | s={sd}] "
                    f"zcol={res.zero_collisions} comp={res.completed_tasks}/{nt} wait={res.wait_ratio:.3f} "
                    f"drift={sim.eta_drift_replans} dETA={row['mean_deta_s']}",
                    flush=True,
                )

    u0_rows = by_var["U0_Orig_CPSA"]
    main_summary = {vname: _summarize_variant(by_var[vname], u0_rows) for vname, _, _ in variants}
    by_scenario: dict[str, dict[str, Any]] = {}
    for sc_label, *_ in specs:
        by_scenario[sc_label] = {}
        sc_u0 = [r for r in u0_rows if r["scenario"] == sc_label]
        for vname, _, _ in variants:
            sc_r = [r for r in by_var[vname] if r["scenario"] == sc_label]
            by_scenario[sc_label][vname] = _summarize_variant(sc_r, sc_u0)

    # Identify best deterministic upper variant on full DEV
    det_cands = [v for v in variants if v[0] != "U0_Orig_CPSA"]
    best_det_name = max(
        [v[0] for v in variants],
        key=lambda vn: (main_summary[vn]["zero_col_pct"], main_summary[vn]["completed_tasks_mean"], -main_summary[vn]["wait_ratio_mean"]),
    )

    # Matched counterfactual U4 Finite-Pool Oracle Slice (seed=6001 across 4 representative scenarios)
    cf_specs = [
        ("Occluded_Jct_6R_Clean", om, 6, 12, 70.0, 2, 0.0, 0.0, 0.0, 0.0),
        ("MapA_8R_Clean", ma, 8, 16, 80.0, 2, 0.0, 0.0, 0.0, 0.0),
        ("Workshop_8R_Clean", wm, 8, 12, 160.0, 1, 0.0, 0.0, 0.0, 0.0),
        ("Occluded_Jct_6R_Noise0.12", om, 6, 12, 70.0, 2, 0.12, 0.20, 0.0, 0.0),
    ]
    u4_rows: list[dict[str, Any]] = []
    u4_oracle = U4CounterfactualOracle(rollout_steps=80)
    for sc_label, mmap, nr, nt, hz, ni, np_m, nv_m, lag_s, drop_p in cf_specs:
        sim = ExecutionAuditSim(
            mmap,
            upper_mode="U4_FinitePool_CF_Oracle",
            sub_tick_div=4,
            custom_plan_fn=u4_oracle,
            n_robots=nr,
            n_tasks=nt,
            horizon_s=hz,
            seed=6001,
            n_intruders=ni,
            obs_noise_pos_m=np_m,
            obs_noise_vel_mps=nv_m,
            reservation_lag_s=lag_s,
            reservation_dropout=drop_p,
            robust_safety=True,
        )
        res = sim.run_episode(m2_res_margin_selector, controller_name="U4_CF_Oracle+R1_Robust")
        comp_legs = [lg for lg in sim.closed_legs if lg.completed and lg.end_t_s is not None]
        detas = [float(lg.end_t_s - lg.planned_eta_s) for lg in comp_legs if lg.end_t_s is not None]
        u4_r = {
            "scenario": sc_label,
            "variant": "U4_FinitePool_CF_Oracle",
            "seed": 6001,
            "completed_tasks": res.completed_tasks,
            "n_tasks": nt,
            "makespan_s": res.makespan_s,
            "zero_collisions": res.zero_collisions,
            "col_ticks": res.wall_collisions + res.peer_collisions + res.intruder_collisions,
            "wait_ratio": res.wait_ratio,
            "replans": res.replans,
            "eta_drift_replans": sim.eta_drift_replans,
            "row_gate_events": sim.row_gate_events,
            "shield_interventions": res.shield_interventions,
            "mean_deta_s": round(float(np.mean(detas)), 3) if detas else None,
        }
        u4_rows.append(u4_r)
        print(
            f"[Task3-U4 | {sc_label:24s} | s=6001] zcol={res.zero_collisions} "
            f"comp={res.completed_tasks}/{nt} wait={res.wait_ratio:.3f} drift={sim.eta_drift_replans}",
            flush=True,
        )

    # Matched slice comparison against U0, U1b, U3, and best deterministic
    cf_scenarios = {s[0] for s in cf_specs}
    slice_summary: dict[str, Any] = {}
    for vname in ["U0_Orig_CPSA", "U1a_EdgeSpan_Fix", "U1b_KinoTime_CPSA", "U2_FixedBuf_1.0s", "U2_FixedBuf_2.0s", "U3_StateDep_Buf"]:
        srows = [r for r in by_var[vname] if r["seed"] == 6001 and r["scenario"] in cf_scenarios]
        slice_summary[vname] = {
            "completed_tasks_sum": int(sum(r["completed_tasks"] for r in srows)),
            "completed_tasks_mean": round(float(np.mean([r["completed_tasks"] for r in srows])), 3),
            "zero_col_cnt": f"{sum(1 for r in srows if r['zero_collisions'])}/{len(srows)}",
            "wait_ratio_mean": round(float(np.mean([r["wait_ratio"] for r in srows])), 4),
            "eta_drift_tot": int(sum(r["eta_drift_replans"] for r in srows)),
        }
    slice_summary["U4_FinitePool_CF_Oracle"] = {
        "completed_tasks_sum": int(sum(r["completed_tasks"] for r in u4_rows)),
        "completed_tasks_mean": round(float(np.mean([r["completed_tasks"] for r in u4_rows])), 3),
        "zero_col_cnt": f"{sum(1 for r in u4_rows if r['zero_collisions'])}/{len(u4_rows)}",
        "wait_ratio_mean": round(float(np.mean([r["wait_ratio"] for r in u4_rows])), 4),
        "eta_drift_tot": int(sum(r["eta_drift_replans"] for r in u4_rows)),
    }

    best_det_slice_sum = max(
        slice_summary[v]["completed_tasks_sum"]
        for v in ["U0_Orig_CPSA", "U1a_EdgeSpan_Fix", "U1b_KinoTime_CPSA", "U2_FixedBuf_1.0s", "U2_FixedBuf_2.0s", "U3_StateDep_Buf"]
    )
    u4_sum = slice_summary["U4_FinitePool_CF_Oracle"]["completed_tasks_sum"]
    oracle_over_best_det_pct = round(100.0 * (u4_sum - best_det_slice_sum) / max(1, best_det_slice_sum), 2)
    oracle_over_u1b_pct = round(
        100.0 * (u4_sum - slice_summary["U1b_KinoTime_CPSA"]["completed_tasks_sum"]) / max(1, slice_summary["U1b_KinoTime_CPSA"]["completed_tasks_sum"]),
        2,
    )
    oracle_over_u0_pct = round(
        100.0 * (u4_sum - slice_summary["U0_Orig_CPSA"]["completed_tasks_sum"]) / max(1, slice_summary["U0_Orig_CPSA"]["completed_tasks_sum"]),
        2,
    )

    payload = {
        "protocol": "Execution-Aware-STP-Audit Task 3 (DEV 6001-6003, Fixed R1_Robust, Zero RL)",
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "full_dev_summary": main_summary,
        "best_deterministic_variant": best_det_name,
        "by_scenario": by_scenario,
        "episode_rows": ep_rows,
        "u4_counterfactual_slice": {
            "scenarios_seed_6001": [s[0] for s in cf_specs],
            "summary": slice_summary,
            "u4_rows": u4_rows,
            "oracle_over_best_det_pct": oracle_over_best_det_pct,
            "oracle_over_u1b_pct": oracle_over_u1b_pct,
            "oracle_over_u0_pct": oracle_over_u0_pct,
            "oracle_decision_stats": {
                "decision_count": u4_oracle.decision_count,
                "differed_from_u1b_count": u4_oracle.differed_from_u1b_count,
                "differed_from_u3_count": u4_oracle.differed_from_u3_count,
                "chosen_tag_histogram": dict(u4_oracle.chosen_tags),
                "mean_20s_rollout_score_gap_vs_u1b": round(
                    float(np.mean(u4_oracle.oracle_vs_u1b_score_gaps)) if u4_oracle.oracle_vs_u1b_score_gaps else 0.0,
                    4,
                ),
            },
        },
    }
    json_path = OUT_DIR / "upper_level_headroom_report.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown report
    md_lines = [
        "# Task 3 — Upper-Level Candidate Headroom Report (`U0..U4` + Fixed `R1_Robust`)",
        "",
        f"- Full DEV Evaluation: `6 scenarios x 3 seeds (6001-6003) = 18 episodes` per variant (`8001-8010` untouched; zero RL).",
        f"- Best Deterministic Upper Variant on DEV: **`{best_det_name}`**",
        "",
        "## 1. Full 18-Episode DEV Comparison",
        "",
        "| Variant | ZeroCol | Comp Mean | vs U0 (%) | p (Comp) | Wait Mean | ETA Drift Replans | ROW Stops | Shield Intv | |dETA| Mean (s) | Plan ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for vname, _, _ in variants:
        s = main_summary[vname]
        md_lines.append(
            f"| {vname} | {s['zero_col_cnt']} ({s['zero_col_pct']:.1f}%) | {s['completed_tasks_mean']:.3f} | "
            f"{s['completed_vs_u0_pct']:+.2f}% | {s['p_comp_vs_u0']:.4f} | {s['wait_ratio_mean']:.4f} | "
            f"{s['eta_drift_replans_tot']} | {s['row_gate_events_tot']} | {s['shield_interventions_tot']} | "
            f"{s['abs_deta_mean_s']:.2f} | {s['plan_ms_mean']:.2f} |"
        )

    md_lines.extend([
        "",
        "## 2. Counterfactual Finite-Pool Oracle Slice (`seed=6001`, 4 Scenarios)",
        "",
        f"- **`U4_FinitePool_CF_Oracle` vs Best Deterministic (`{best_det_slice_sum}` tasks)**: **`{oracle_over_best_det_pct:+.2f}%`** (`{u4_sum}` tasks)",
        f"- **`U4_FinitePool_CF_Oracle` vs `U1b_KinoTime_CPSA`**: **`{oracle_over_u1b_pct:+.2f}%`**",
        f"- **`U4_FinitePool_CF_Oracle` vs `U0_Orig_CPSA`**: **`{oracle_over_u0_pct:+.2f}%`**",
        f"- Oracle Triggers: `{u4_oracle.decision_count}` plans (`differed_from_u1b={u4_oracle.differed_from_u1b_count}`, `differed_from_u3={u4_oracle.differed_from_u3_count}`, `tags={dict(u4_oracle.chosen_tags)}`)",
        "",
        "| Variant (seed=6001 slice) | ZeroCol | Comp Sum | Comp Mean | Wait Mean | ETA Drift |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for vn, sv in slice_summary.items():
        md_lines.append(
            f"| {vn} | {sv['zero_col_cnt']} | {sv['completed_tasks_sum']} | {sv['completed_tasks_mean']:.3f} | "
            f"{sv['wait_ratio_mean']:.4f} | {sv['eta_drift_tot']} |"
        )

    md_path = OUT_DIR / "upper_level_headroom_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[Task3] Wrote {md_path} and {json_path}", flush=True)
    return payload


if __name__ == "__main__":
    run_suite()
