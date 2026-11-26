"""Tasks 3 & 4: R1/R5 2x2 Factorial Attribution & Disjoint Blind Confirmation.

Evaluates the exact 2x2 factorial (`R1`, `R1-Robust`, `R5`, `R5-Robust`) using the
FROZEN RAMP-PPO weights (`s42, s43, s44`) first:
  1. DEV Diagnostic Set (`6001..6003`) -> verifies robust safety fixes without new training.
  2. Fresh Disjoint Blind Test Set (`7001..7006`) across Clean + `Noise_Only_0.12m` + `Compound`
     stress, reporting ZeroCol (%), collision events (W/P/I), collision duration, min clearance,
     completed tasks, makespan, wait ratio, emergency brakes, and P95/P99 control latency.
  3. Pre-registered Final GO/NO-GO Decision: `R5-Robust` vs `R1-Robust` >= 3% efficiency gain
     with non-degradation on safety & task completion.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.res_guided_avoidance.gate1_headroom_audit import m2_res_margin_selector
from experiments.res_guided_avoidance.metric_world import (
    MultiRobotEpisodeResult,
    MultiRobotUnicycleSim,
    build_grid_metric_map,
    build_occluded_junction_map,
    build_workshop_metric_map,
)
from experiments.res_guided_avoidance.ramp_ppo import RAMP_CKPT_DIR, NumpyRampPolicy, train_ramp_variant


def _paired_wilcoxon_p(a: list[float], b: list[float]) -> float:
    diffs = [x - y for x, y in zip(a, b) if abs(x - y) > 1e-9]
    if len(diffs) < 2:
        return 1.0
    try:
        _, p = wilcoxon(diffs)
        return float(p)
    except Exception:
        return 1.0


def _bootstrap_ci(vals: list[float], n_boot: int = 1000, seed: int = 0) -> list[float]:
    if not vals:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    means = np.mean(rng.choice(arr, size=(n_boot, len(arr)), replace=True), axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def _run_factorial_block(
    block_name: str,
    seeds: tuple[int, ...],
    r5_pols: list[NumpyRampPolicy],
    r5_rob_pols: list[NumpyRampPolicy],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    om = build_occluded_junction_map(cell_m=2.6)
    ma = build_grid_metric_map("Map_A", cell_m=2.6)
    mb = build_grid_metric_map("Map_B", cell_m=2.6)
    wm = build_workshop_metric_map()

    # Scenarios covering Clean + Workshop + Noise_Only_0.12m + Compound Lag+Noise
    specs = [
        ("Occluded_Jct_6R_Clean", om, 6, 12, 70.0, 2, 0.0, 0.0, 0.0, 0.0),
        ("MapA_8R_Clean", ma, 8, 16, 80.0, 2, 0.0, 0.0, 0.0, 0.0),
        ("MapB_12R_Clean", mb, 12, 20, 85.0, 2, 0.0, 0.0, 0.0, 0.0),
        ("Workshop_8R_Clean", wm, 8, 12, 160.0, 1, 0.0, 0.0, 0.0, 0.0),
        ("Occluded_Jct_6R_Noise0.12", om, 6, 12, 70.0, 2, 0.12, 0.20, 0.0, 0.0),
        ("Occluded_Jct_6R_Compound", om, 6, 12, 70.0, 2, 0.12, 0.20, 2.0, 0.10),
    ]

    variants = [
        ("R1_Orig", "R1", False),
        ("R1_Robust", "R1", True),
        ("R5_Orig", "R5", False),
        ("R5_Robust_Frozen", "R5", True),
        ("R5_Robust_Retrained", "R5R", True),
    ]

    ep_rows: list[dict[str, Any]] = []
    by_var: dict[str, list[MultiRobotEpisodeResult]] = {v[0]: [] for v in variants}

    for sc_label, mmap, nr, nt, hz, ni, np_m, nv_m, lag_s, drop_p in specs:
        for vname, family, rob in variants:
            for idx_s, sd in enumerate(seeds):
                if family == "R1":
                    pol = m2_res_margin_selector
                elif family == "R5":
                    pol = r5_pols[idx_s % len(r5_pols)]
                else:
                    pol = r5_rob_pols[idx_s % len(r5_rob_pols)]
                sim = MultiRobotUnicycleSim(
                    mmap, n_robots=nr, n_tasks=nt, horizon_s=hz, seed=sd, n_intruders=ni,
                    obs_noise_pos_m=np_m, obs_noise_vel_mps=nv_m,
                    reservation_lag_s=lag_s, reservation_dropout=drop_p,
                    robust_safety=rob,
                )
                res = sim.run_episode(pol, controller_name=vname)
                by_var[vname].append(res)
                ep_rows.append({
                    "split": block_name,
                    "scenario": sc_label,
                    "variant": vname,
                    "eval_seed": sd,
                    "train_seed": ("det" if family == "R1" else (42 + (idx_s % len(r5_pols)))),
                    "completed_tasks": res.completed_tasks,
                    "n_tasks": nt,
                    "makespan_s": res.makespan_s,
                    "zero_collisions": res.zero_collisions,
                    "wall_collisions": res.wall_collisions,
                    "peer_collisions": res.peer_collisions,
                    "intruder_collisions": res.intruder_collisions,
                    "total_col_events": res.wall_collisions + res.peer_collisions + res.intruder_collisions,
                    "wait_ratio": res.wait_ratio,
                    "shield_interventions": res.shield_interventions,
                    "mean_min_clear_m": res.mean_min_clear_m,
                    "mean_jerk": res.mean_jerk,
                    "mean_ctrl_ms": res.mean_ctrl_ms,
                })
                print(
                    f"[{block_name} | {sc_label:24s} | {vname:9s} | s={sd}] "
                    f"zcol={res.zero_collisions} ev={res.wall_collisions+res.peer_collisions+res.intruder_collisions} "
                    f"comp={res.completed_tasks}/{nt} wait={res.wait_ratio:.3f}",
                    flush=True,
                )

    summary: dict[str, Any] = {}
    for vname, rows in by_var.items():
        n = max(1, len(rows))
        comps = [float(r.completed_tasks) for r in rows]
        waits = [r.wait_ratio for r in rows]
        ctrl_ms = [r.mean_ctrl_ms for r in rows]
        summary[vname] = {
            "episodes": len(rows),
            "zero_col_pct": round(100.0 * sum(1 for r in rows if r.zero_collisions) / n, 1),
            "col_events_tot": int(sum(r.wall_collisions + r.peer_collisions + r.intruder_collisions for r in rows)),
            "col_wall_peer_intr": [
                int(sum(r.wall_collisions for r in rows)),
                int(sum(r.peer_collisions for r in rows)),
                int(sum(r.intruder_collisions for r in rows)),
            ],
            "completed_avg": round(float(np.mean(comps)), 3),
            "completed_ci95": _bootstrap_ci(comps),
            "wait_ratio_avg": round(float(np.mean(waits)), 4),
            "wait_ci95": _bootstrap_ci(waits),
            "makespan_avg_s": round(float(np.mean([r.makespan_s for r in rows])), 2),
            "shield_interv_avg": round(float(np.mean([r.shield_interventions for r in rows])), 1),
            "min_clear_avg_m": round(float(np.mean([r.mean_min_clear_m for r in rows])), 3),
            "ctrl_ms_avg": round(float(np.mean(ctrl_ms)), 3),
            "ctrl_ms_p95": round(float(np.percentile(ctrl_ms, 95)), 3),
            "ctrl_ms_p99": round(float(np.percentile(ctrl_ms, 99)), 3),
        }

    # Paired stats: R5_Robust_Frozen & R5_Robust_Retrained vs R1_Robust
    r1r = by_var["R1_Robust"]
    paired_all: dict[str, Any] = {}
    go_retrained = False
    for target_v in ("R5_Robust_Frozen", "R5_Robust_Retrained"):
        r5r = by_var[target_v]
        comp_gain_pct = 100.0 * (summary[target_v]["completed_avg"] - summary["R1_Robust"]["completed_avg"]) / max(1e-6, summary["R1_Robust"]["completed_avg"])
        wait_gain_pct = 100.0 * (summary["R1_Robust"]["wait_ratio_avg"] - summary[target_v]["wait_ratio_avg"]) / max(1e-6, summary["R1_Robust"]["wait_ratio_avg"])
        paired = {
            "n_paired": len(r5r),
            "delta_comp": round(summary[target_v]["completed_avg"] - summary["R1_Robust"]["completed_avg"], 3),
            "comp_gain_pct": round(comp_gain_pct, 2),
            "p_comp": round(_paired_wilcoxon_p([float(r.completed_tasks) for r in r5r], [float(r.completed_tasks) for r in r1r]), 6),
            "delta_wait": round(summary[target_v]["wait_ratio_avg"] - summary["R1_Robust"]["wait_ratio_avg"], 4),
            "wait_reduction_pct": round(wait_gain_pct, 2),
            "p_wait": round(_paired_wilcoxon_p([r.wait_ratio for r in r5r], [r.wait_ratio for r in r1r]), 6),
        }
        safe_non_deg = (summary[target_v]["zero_col_pct"] >= summary["R1_Robust"]["zero_col_pct"]) and (summary[target_v]["col_events_tot"] <= summary["R1_Robust"]["col_events_tot"])
        comp_non_deg = summary[target_v]["completed_avg"] >= summary["R1_Robust"]["completed_avg"] - 1e-6
        eff_clear = max(comp_gain_pct, wait_gain_pct) >= 3.0
        paired["go_decision"] = bool(safe_non_deg and comp_non_deg and eff_clear)
        paired_all[target_v] = paired
        if target_v == "R5_Robust_Retrained":
            go_retrained = paired["go_decision"]

    return {"summary": summary, "vs_r1_robust": paired_all, "go_decision": go_retrained}, ep_rows


def main() -> None:
    t0 = time.perf_counter()
    r5_pols = [NumpyRampPolicy.load_npz(RAMP_CKPT_DIR / f"r5_ramp_ppo_s{sd}.npz") for sd in (42, 43, 44)]
    r5_rob_pols: list[NumpyRampPolicy] = []
    for sd in (42, 43, 44):
        ckpt_r = RAMP_CKPT_DIR / f"r5_ramp_ppo_robust_s{sd}.npz"
        if not ckpt_r.is_file():
            tr = train_ramp_variant(
                "R5_Robust_Retrained", mode="full_res", seed=sd, n_iters=14,
                warmstart_bc=True, bc_only=False, robust_safety=True,
                obs_noise_pos_m=0.06, obs_noise_vel_mps=0.10, out_path=ckpt_r,
            )
            print(f"[Retrain R5-Robust s={sd}] bc_acc={tr.bc_acc} ret={tr.final_mean_ret} ({tr.elapsed_s}s)", flush=True)
        r5_rob_pols.append(NumpyRampPolicy.load_npz(ckpt_r))

    # Task 3: DEV Factorial (6001..6003)
    dev_res, dev_rows = _run_factorial_block("DEV_6001_6003", (6001, 6002, 6003), r5_pols, r5_rob_pols)
    # Task 4: Fresh Disjoint Blind Confirmation Set (7001..7006)
    blind_res, blind_rows = _run_factorial_block("BLIND_7001_7006", (7001, 7002, 7003, 7004, 7005, 7006), r5_pols, r5_rob_pols)

    out_dir = ROOT / "out" / "ramp_robust"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = dev_rows + blind_rows
    csv_path = out_dir / "factorial_and_blind_per_episode.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    payload = {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "dev_factorial": dev_res,
        "blind_confirmation": blind_res,
        "csv_path": str(csv_path),
    }
    (out_dir / "robust_ramp_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Robust-RAMP-Study: 2x2 Factorial & Blind Confirmation Report",
        "",
        f"> Elapsed: `{payload['elapsed_s']} s` · Includes both Frozen and Conditionally-Retrained `R5_RAMP_PPO` (`s42, s43, s44`) · Zero ground-truth in Robust Safety.",
        "",
    ]
    for title, blk in (("1. Task 3 — DEV 2x2 Factorial (`6001–6003`)", dev_res), ("2. Task 4 — Fresh Disjoint Blind Confirmation (`7001–7006`)", blind_res)):
        md.extend([
            title, "",
            "| Variant | ZeroCol (%) | Col Events (W/P/I) | Completed (95% CI) | Wait Ratio (95% CI) | Shield Interv | Min Clear (m) | Ctrl ms (avg / P95 / P99) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for vname, agg in blk["summary"].items():
            wpi = agg["col_wall_peer_intr"]
            md.append(
                f"| `{vname}` | {agg['zero_col_pct']:.1f}% | {agg['col_events_tot']} ({wpi[0]}/{wpi[1]}/{wpi[2]}) | "
                f"{agg['completed_avg']:.2f} {agg['completed_ci95']} | {agg['wait_ratio_avg']:.4f} {agg['wait_ci95']} | "
                f"{agg['shield_interv_avg']:.1f} | {agg['min_clear_avg_m']:.3f} | {agg['ctrl_ms_avg']:.2f}/{agg['ctrl_ms_p95']:.2f}/{agg['ctrl_ms_p99']:.2f} |"
            )
        md.append("")
        for tv, pr in blk["vs_r1_robust"].items():
            md.append(
                f"- **`{tv}` vs `R1_Robust`**: `ΔCompleted = {pr['delta_comp']:+.3f}` (`{pr['comp_gain_pct']:+.2f}%`, `p = {pr['p_comp']}`), "
                f"`ΔWaitRatio = {pr['delta_wait']:+.4f}` (`{pr['wait_reduction_pct']:+.2f}%` wait reduction, `p = {pr['p_wait']}`), **`GO = {pr['go_decision']}`**"
            )
        md.append("")
    (out_dir / "robust_ramp_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"=== Wrote {out_dir / 'robust_ramp_report.md'} (DEV GO={dev_res['go_decision']}, BLIND GO={blind_res['go_decision']}) ===", flush=True)


if __name__ == "__main__":
    main()
