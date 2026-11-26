"""P4 Multi-Robot Generalization, Density Scaling, Latency & Stale-Reservation Evaluation Suite.

Evaluates all 5 controlled P2 RL variants (`BC_Only`, `Pure_PPO_NoBC`, `Plain_Kin_PPO`,
`Intent_NoTime_PPO`, `RG_PPO`) alongside 5 P3 continuous baselines (`ORCA_Uni`, `DWA`,
`Res_DWA`, `Uni_MPC`, `Res_Yield`) on quarantined held-out seeds across:
  - Bench A: Occluded Intersections & Narrow Corridors
  - Bench B: Fleet Density Scaling (`N = 4..24` simultaneous robots)
  - Bench C: Sensor Noise & Stale/Lagged Reservation Stress (`lag = 0..4.0 s`)
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.res_guided_avoidance.continuous_baselines import (
    dwa_controller,
    orca_uni_controller,
    res_dwa_controller,
    res_yield_controller,
    uni_mpc_controller,
)
from experiments.res_guided_avoidance.metric_world import (
    MultiRobotEpisodeResult,
    MultiRobotUnicycleSim,
    build_grid_metric_map,
    build_occluded_junction_map,
    build_workshop_metric_map,
)
from experiments.res_guided_avoidance.rg_ppo import (
    CKPT_DIR,
    NumpyMultiRobotPolicy,
    train_all_p2_variants,
)


def _paired_wilcoxon_p(a: list[float], b: list[float]) -> float:
    diffs = [x - y for x, y in zip(a, b) if abs(x - y) > 1e-9]
    if len(diffs) < 2:
        return 1.0
    try:
        _, p = wilcoxon(diffs)
        return float(p)
    except Exception:
        return 1.0


def _load_controllers(train_seed: int = 42) -> dict[str, Callable[[MultiRobotUnicycleSim, str], tuple[float, float]]]:
    ctrls: dict[str, Callable[[MultiRobotUnicycleSim, str], tuple[float, float]]] = {
        "ORCA_Uni": orca_uni_controller,
        "DWA": dwa_controller,
        "Res_DWA": res_dwa_controller,
        "Uni_MPC": uni_mpc_controller,
        "Res_Yield": res_yield_controller,
    }
    for name in ("BC_Only", "Pure_PPO_NoBC", "Plain_Kin_PPO", "Intent_NoTime_PPO", "RG_PPO"):
        ckpt = CKPT_DIR / f"{name.lower()}_s{train_seed}.npz"
        if ckpt.is_file():
            ctrls[name] = NumpyMultiRobotPolicy.load_npz(ckpt)
    return ctrls


def _aggregate(rows: list[MultiRobotEpisodeResult]) -> dict[str, Any]:
    n = max(1, len(rows))
    return {
        "episodes": len(rows),
        "zero_col_pct": round(100.0 * sum(1 for r in rows if r.zero_collisions) / n, 1),
        "completed_avg": round(float(np.mean([r.completed_tasks for r in rows])), 2),
        "makespan_avg_s": round(float(np.mean([r.makespan_s for r in rows])), 2),
        "wait_ratio_avg": round(float(np.mean([r.wait_ratio for r in rows])), 4),
        "jerk_avg": round(float(np.mean([r.mean_jerk for r in rows])), 3),
        "shield_interv_avg": round(float(np.mean([r.shield_interventions for r in rows])), 1),
        "shield_infeas_tot": int(sum(r.shield_infeasible_count for r in rows)),
        "replans_avg": round(float(np.mean([r.replans for r in rows])), 1),
        "ctrl_ms_avg": round(float(np.mean([r.mean_ctrl_ms for r in rows])), 3),
    }


def run_eval_suite(
    *,
    train_seed: int = 42,
    eval_seeds: tuple[int, ...] = (5001, 5002, 5003),
    out_dir: Path = ROOT / "out" / "res_guided_avoidance",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    ctrls = _load_controllers(train_seed=train_seed)
    wm = build_workshop_metric_map()
    om = build_occluded_junction_map(cell_m=2.6)
    ma = build_grid_metric_map("Map_A", cell_m=2.6)
    mb = build_grid_metric_map("Map_B", cell_m=2.6)

    # Bench A: Occluded Intersections & Narrow Corridors
    bench_a_specs = [
        ("Occluded_Jct_6R", om, 6, 12, 70.0, 2),
        ("MapA_Metric_8R", ma, 8, 16, 80.0, 2),
        ("Workshop_Metric_8R", wm, 8, 16, 150.0, 2),
    ]
    bench_a_rows: list[MultiRobotEpisodeResult] = []
    for sc_label, mmap, nr, nt, hz, ni in bench_a_specs:
        for cname, cfn in ctrls.items():
            for sd in eval_seeds:
                sim = MultiRobotUnicycleSim(mmap, n_robots=nr, n_tasks=nt, horizon_s=hz, seed=sd, n_intruders=ni)
                res = sim.run_episode(cfn, controller_name=cname)
                bench_a_rows.append(res)
                print(f"[BenchA | {sc_label:18s} | {cname:18s} | s={sd}] comp={res.completed_tasks}/{nt} ms={res.makespan_s}s zcol={res.zero_collisions} wait={res.wait_ratio:.3f} sh={res.shield_interventions}", flush=True)

    # Bench B: Density Scaling on Workshop_v1 (N=4, 8, 12, 16, 24) & Map_B (N=16)
    bench_b_specs = [
        ("Workshop_N4", wm, 4, 8, 140.0, 1),
        ("Workshop_N8", wm, 8, 14, 150.0, 2),
        ("Workshop_N12", wm, 12, 18, 150.0, 2),
        ("Workshop_N16", wm, 16, 22, 160.0, 2),
        ("Workshop_N24", wm, 24, 28, 160.0, 2),
        ("MapB_N16", mb, 16, 24, 90.0, 2),
    ]
    focus_ctrls = [c for c in ("ORCA_Uni", "Res_DWA", "Uni_MPC", "Plain_Kin_PPO", "Intent_NoTime_PPO", "RG_PPO") if c in ctrls]
    bench_b_rows: list[MultiRobotEpisodeResult] = []
    for sc_label, mmap, nr, nt, hz, ni in bench_b_specs:
        for cname in focus_ctrls:
            for sd in eval_seeds:
                sim = MultiRobotUnicycleSim(mmap, n_robots=nr, n_tasks=nt, horizon_s=hz, seed=sd, n_intruders=ni)
                res = sim.run_episode(ctrls[cname], controller_name=cname)
                bench_b_rows.append(res)
                print(f"[BenchB | {sc_label:18s} | {cname:18s} | s={sd}] comp={res.completed_tasks}/{nt} ms={res.makespan_s}s zcol={res.zero_collisions} wait={res.wait_ratio:.3f}", flush=True)

    # Bench C: Sensor Noise & Stale Reservation Lag Stress (lag = 0.0, 1.0, 2.0, 4.0 s)
    bench_c_rows: list[dict[str, Any]] = []
    for lag_s, drop_p, noise_p, noise_v in (
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.08, 0.15),
        (2.0, 0.10, 0.12, 0.22),
        (4.0, 0.20, 0.15, 0.25),
    ):
        for cname in [c for c in ("Res_DWA", "Plain_Kin_PPO", "RG_PPO") if c in ctrls]:
            sub: list[MultiRobotEpisodeResult] = []
            for sd in eval_seeds[:3]:
                sim = MultiRobotUnicycleSim(
                    om, n_robots=6, n_tasks=12, horizon_s=70.0, seed=sd, n_intruders=2,
                    obs_noise_pos_m=noise_p, obs_noise_vel_mps=noise_v,
                    reservation_lag_s=lag_s, reservation_dropout=drop_p,
                )
                sub.append(sim.run_episode(ctrls[cname], controller_name=cname))
            agg = _aggregate(sub)
            agg.update({"lag_s": lag_s, "dropout": drop_p, "noise_pos": noise_p, "controller": cname})
            bench_c_rows.append(agg)

    # Summarize Bench A by controller + paired tests vs RG_PPO
    summary_a: dict[str, Any] = {}
    for cname in ctrls:
        summary_a[cname] = _aggregate([r for r in bench_a_rows if r.controller == cname])

    paired_vs_rg: dict[str, Any] = {}
    if "RG_PPO" in ctrls:
        rg_rows = [r for r in bench_a_rows if r.controller == "RG_PPO"]
        rg_waits = [r.wait_ratio for r in rg_rows]
        rg_ms = [r.makespan_s for r in rg_rows]
        for cname in ctrls:
            if cname == "RG_PPO":
                continue
            ot_rows = [r for r in bench_a_rows if r.controller == cname]
            paired_vs_rg[cname] = {
                "delta_wait_ratio": round(float(np.mean(rg_waits) - np.mean([r.wait_ratio for r in ot_rows])), 4),
                "p_wait": round(_paired_wilcoxon_p(rg_waits, [r.wait_ratio for r in ot_rows]), 6),
                "delta_makespan_s": round(float(np.mean(rg_ms) - np.mean([r.makespan_s for r in ot_rows])), 2),
                "p_makespan": round(_paired_wilcoxon_p(rg_ms, [r.makespan_s for r in ot_rows]), 6),
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "bench_a_summary": summary_a,
        "paired_vs_rg_ppo": paired_vs_rg,
        "bench_a_rows": [asdict(r) for r in bench_a_rows],
        "bench_b_rows": [asdict(r) for r in bench_b_rows],
        "bench_c_staleness": bench_c_rows,
    }
    (out_dir / "res_guided_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Spatiotemporal Reservation-Guided RL Multi-Robot Dynamic Avoidance Report (`P0–P4`)",
        "",
        f"> Elapsed: `{payload['elapsed_s']} s` · Unified capsule footprint (`1.20 m × 0.90 m`, `r = 0.45 m`) + `M_sub = 5` swept collision check + Braking-Invariant Shield + Squashed-Gaussian PPO.",
        "",
        "## 1. Bench A — Occluded Intersections & Narrow Corridors (Aggregated Across Maps & Held-Out Seeds)",
        "",
        "| Controller | ZeroCol (%) | Completed | Makespan (s) | Wait Ratio | Mean Jerk | Shield Interv | Infeasible | Replans | Ctrl (ms/bot) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for cname, agg in summary_a.items():
        md.append(
            f"| `{cname}` | {agg['zero_col_pct']:.1f}% | {agg['completed_avg']:.2f} | {agg['makespan_avg_s']:.2f} | "
            f"{agg['wait_ratio_avg']:.4f} | {agg['jerk_avg']:.3f} | {agg['shield_interv_avg']:.1f} | "
            f"{agg['shield_infeas_tot']} | {agg['replans_avg']:.1f} | {agg['ctrl_ms_avg']:.3f} |"
        )
    if paired_vs_rg:
        md.extend(["", "## 2. Paired Wilcoxon Comparisons (`RG_PPO` vs Comparators on Bench A)", "", "| Comparator | `ΔWaitRatio` (`RG - Base`) | `p_wait` | `ΔMakespan` (s) | `p_makespan` |", "| :--- | :---: | :---: | :---: | :---: |"])
        for cname, pr in paired_vs_rg.items():
            md.append(f"| `{cname}` | {pr['delta_wait_ratio']:+.4f} | {pr['p_wait']:.4f} | {pr['delta_makespan_s']:+.2f} | {pr['p_makespan']:.4f} |")

    md.extend(["", "## 3. Bench B — Fleet Density Scaling (`N = 4..24` Simultaneous Robots)", "", "| Map & Fleet | Controller | ZeroCol (%) | Completed | Makespan (s) | Wait Ratio | Ctrl (ms/bot) |", "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"])
    for sc_label, mmap, nr, nt, hz, ni in bench_b_specs:
        for cname in focus_ctrls:
            sub = [r for r in bench_b_rows if r.map_name == mmap.name and r.n_robots == nr and r.controller == cname]
            if not sub:
                continue
            agg = _aggregate(sub)
            md.append(f"| `{sc_label}` | `{cname}` | {agg['zero_col_pct']:.1f}% | {agg['completed_avg']:.2f}/{nt} | {agg['makespan_avg_s']:.2f} | {agg['wait_ratio_avg']:.4f} | {agg['ctrl_ms_avg']:.3f} |")

    md.extend(["", "## 4. Bench C — Stale Reservation Lag & Sensor Noise Stress (`Occluded_Junction`)", "", "| Lag (s) / Dropout / Noise | Controller | ZeroCol (%) | Completed | Makespan (s) | Wait Ratio | Shield Interv |", "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"])
    for r in bench_c_rows:
        md.append(f"| `{r['lag_s']}s / drop={r['dropout']} / σ={r['noise_pos']}m` | `{r['controller']}` | {r['zero_col_pct']:.1f}% | {r['completed_avg']:.2f}/12 | {r['makespan_avg_s']:.2f} | {r['wait_ratio_avg']:.4f} | {r['shield_interv_avg']:.1f} |")

    (out_dir / "res_guided_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"=== Wrote {out_dir / 'res_guided_report.md'} ({payload['elapsed_s']}s) ===", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="P0-P4 Spatiotemporal Reservation-Guided RL Avoidance Suite")
    parser.add_argument("--train-all", action="store_true", help="Train all 5 controlled P2 variants")
    parser.add_argument("--eval-all", action="store_true", help="Run full P4 Bench A/B/C evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out" / "res_guided_avoidance")
    args = parser.parse_args()

    if args.train_all:
        res = train_all_p2_variants(seed=args.seed, n_iters=args.iters)
        print(json.dumps(res, indent=2), flush=True)
    if args.eval_all:
        run_eval_suite(train_seed=args.seed, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
