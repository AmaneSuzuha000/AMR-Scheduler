"""Gate 3: Independent Confirmation, Strong Baselines, Single-Variable Stress & GO/NO-GO Verdict.

Evaluates all trained RAMP variants (`R2_BC_Only`, `R3_Primitive_PPO`, `R4_Spatial_Intent_PPO`,
`R5_RAMP_PPO`) across seeds (42, 43, 44) alongside:
  - Non-learned primitive selectors (`R0_Det_Primitive`, `R1_Det_ResMargin`)
  - Original continuous RG-PPO (`R6_Orig_Cont_RGPPO`)
  - Strong baselines (`CPSA_Follower`, `ORCA_Uni`, `DWA`, `Res_DWA`, `Uni_MPC`, `TPG_ADG_Follower`, `MAWPF_Kin_Ref`)

Outputs clean metrics (Collision %, Completion, Arrival/Makespan, Wait Ratio, Shield Interv,
Jerk, Ctrl ms/bot) + Paired Wilcoxon + Single-Variable vs Compound Bench C + GO/NO-GO verdict.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
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
    cpsa_follower_controller,
    dwa_controller,
    mawpf_kin_controller,
    orca_uni_controller,
    res_dwa_controller,
    tpg_adg_controller,
    uni_mpc_controller,
)
from experiments.res_guided_avoidance.gate1_headroom_audit import (
    m0_perception_selector,
    m2_res_margin_selector,
)
from experiments.res_guided_avoidance.metric_world import (
    MultiRobotEpisodeResult,
    MultiRobotUnicycleSim,
    build_grid_metric_map,
    build_occluded_junction_map,
    build_workshop_metric_map,
)
from experiments.res_guided_avoidance.ramp_ppo import (
    RAMP_CKPT_DIR,
    NumpyRampPolicy,
    train_all_ramp_variants,
)
from experiments.res_guided_avoidance.rg_ppo import CKPT_DIR as RG_CKPT_DIR, NumpyMultiRobotPolicy


def _paired_wilcoxon_p(a: list[float], b: list[float]) -> float:
    diffs = [x - y for x, y in zip(a, b) if abs(x - y) > 1e-9]
    if len(diffs) < 2:
        return 1.0
    try:
        _, p = wilcoxon(diffs)
        return float(p)
    except Exception:
        return 1.0


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
        "min_clear_avg_m": round(float(np.mean([r.mean_min_clear_m for r in rows])), 3),
        "ctrl_ms_avg": round(float(np.mean([r.mean_ctrl_ms for r in rows])), 3),
    }


def _load_controllers(train_seeds: tuple[int, ...] = (42, 43, 44)) -> dict[str, list[Callable[[MultiRobotUnicycleSim, str], tuple[float, float]]]]:
    ctrls: dict[str, list[Callable[[MultiRobotUnicycleSim, str], tuple[float, float]]]] = {
        "R0_Det_Primitive": [m0_perception_selector],
        "R1_Det_ResMargin": [m2_res_margin_selector],
        "CPSA_Follower": [cpsa_follower_controller],
        "ORCA_Uni": [orca_uni_controller],
        "DWA": [dwa_controller],
        "Res_DWA": [res_dwa_controller],
        "Uni_MPC": [uni_mpc_controller],
        "TPG_ADG_Follower": [tpg_adg_controller],
        "MAWPF_Kin_Ref": [mawpf_kin_controller],
    }
    for vname in ("R2_BC_Only", "R3_Primitive_PPO", "R4_Spatial_Intent_PPO", "R5_RAMP_PPO"):
        pols = []
        for sd in train_seeds:
            p = RAMP_CKPT_DIR / f"{vname.lower()}_s{sd}.npz"
            if p.is_file():
                pols.append(NumpyRampPolicy.load_npz(p))
        if pols:
            ctrls[vname] = pols
    rg_p = RG_CKPT_DIR / "rg_ppo_s42.npz"
    if rg_p.is_file():
        ctrls["R6_Orig_Cont_RGPPO"] = [NumpyMultiRobotPolicy.load_npz(rg_p)]
    return ctrls


def run_ramp_eval(
    *,
    train_seeds: tuple[int, ...] = (42, 43, 44),
    eval_seeds: tuple[int, ...] = (6001, 6002, 6003),
    out_dir: Path = ROOT / "out" / "ramp_study",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    ctrls = _load_controllers(train_seeds=train_seeds)

    om = build_occluded_junction_map(cell_m=2.6)
    ma = build_grid_metric_map("Map_A", cell_m=2.6)
    mb = build_grid_metric_map("Map_B", cell_m=2.6)
    wm = build_workshop_metric_map()

    # Held-out Confirmation Suite across 4 regimes
    conf_specs = [
        ("Occluded_Jct_6R", om, 6, 12, 70.0, 2),
        ("MapA_Metric_8R", ma, 8, 16, 80.0, 2),
        ("MapB_Metric_12R", mb, 12, 20, 85.0, 2),
        ("Workshop_8R", wm, 8, 12, 160.0, 1),
    ]
    conf_rows: list[MultiRobotEpisodeResult] = []
    for sc_label, mmap, nr, nt, hz, ni in conf_specs:
        for cname, pol_list in ctrls.items():
            for idx_s, sd in enumerate(eval_seeds):
                pol = pol_list[idx_s % len(pol_list)]
                sim = MultiRobotUnicycleSim(mmap, n_robots=nr, n_tasks=nt, horizon_s=hz, seed=sd, n_intruders=ni)
                res = sim.run_episode(pol, controller_name=cname)
                conf_rows.append(res)
                print(
                    f"[Conf | {sc_label:15s} | {cname:21s} | s={sd}] comp={res.completed_tasks}/{nt} "
                    f"zcol={res.zero_collisions} wait={res.wait_ratio:.3f} sh={res.shield_interventions}",
                    flush=True,
                )

    summary: dict[str, Any] = {
        cname: _aggregate([r for r in conf_rows if r.controller == cname])
        for cname in ctrls
    }

    paired_vs_ramp: dict[str, Any] = {}
    if "R5_RAMP_PPO" in ctrls:
        rg_rows = [r for r in conf_rows if r.controller == "R5_RAMP_PPO"]
        rg_w = [r.wait_ratio for r in rg_rows]
        rg_c = [float(r.completed_tasks) for r in rg_rows]
        for cname in ctrls:
            if cname == "R5_RAMP_PPO":
                continue
            ot = [r for r in conf_rows if r.controller == cname]
            paired_vs_ramp[cname] = {
                "delta_comp": round(float(np.mean(rg_c) - np.mean([r.completed_tasks for r in ot])), 2),
                "p_comp": round(_paired_wilcoxon_p(rg_c, [float(r.completed_tasks) for r in ot]), 4),
                "delta_wait": round(float(np.mean(rg_w) - np.mean([r.wait_ratio for r in ot])), 4),
                "p_wait": round(_paired_wilcoxon_p(rg_w, [r.wait_ratio for r in ot]), 4),
            }

    # Single-Variable vs Compound Stress (Bench C) on Occluded_Junction
    stress_specs = [
        ("Clean_0s", 0.0, 0.0, 0.0, 0.0),
        ("Lag_Only_1s", 1.0, 0.0, 0.0, 0.0),
        ("Lag_Only_2s", 2.0, 0.0, 0.0, 0.0),
        ("Drop_Only_0.2", 0.0, 0.20, 0.0, 0.0),
        ("Noise_Only_0.12m", 0.0, 0.0, 0.12, 0.20),
        ("Compound_2s_0.1_0.12", 2.0, 0.10, 0.12, 0.20),
    ]
    stress_rows: list[dict[str, Any]] = []
    stress_ctrls = [c for c in ("R1_Det_ResMargin", "R4_Spatial_Intent_PPO", "R5_RAMP_PPO", "Res_DWA") if c in ctrls]
    for s_label, lag_s, drop_p, np_m, nv_m in stress_specs:
        for cname in stress_ctrls:
            sub: list[MultiRobotEpisodeResult] = []
            for idx_s, sd in enumerate(eval_seeds):
                pol = ctrls[cname][idx_s % len(ctrls[cname])]
                sim = MultiRobotUnicycleSim(
                    om, n_robots=6, n_tasks=12, horizon_s=70.0, seed=sd, n_intruders=2,
                    obs_noise_pos_m=np_m, obs_noise_vel_mps=nv_m,
                    reservation_lag_s=lag_s, reservation_dropout=drop_p,
                )
                sub.append(sim.run_episode(pol, controller_name=cname))
            agg = _aggregate(sub)
            agg.update({"stress": s_label, "controller": cname})
            stress_rows.append(agg)

    # Pre-registered GO/NO-GO Verdict
    verdict_notes: list[str] = []
    go_decision = False
    if "R5_RAMP_PPO" in summary and "R4_Spatial_Intent_PPO" in summary and "R1_Det_ResMargin" in summary:
        r5 = summary["R5_RAMP_PPO"]
        r4 = summary["R4_Spatial_Intent_PPO"]
        r1 = summary["R1_Det_ResMargin"]
        gain_vs_r4_comp = 100.0 * (r5["completed_avg"] - r4["completed_avg"]) / max(1e-6, r4["completed_avg"])
        gain_vs_r4_wait = 100.0 * (r4["wait_ratio_avg"] - r5["wait_ratio_avg"]) / max(1e-6, r4["wait_ratio_avg"])
        gain_vs_r1_comp = 100.0 * (r5["completed_avg"] - r1["completed_avg"]) / max(1e-6, r1["completed_avg"])
        gain_vs_r1_wait = 100.0 * (r1["wait_ratio_avg"] - r5["wait_ratio_avg"]) / max(1e-6, r1["wait_ratio_avg"])
        non_deg_safe = (r5["zero_col_pct"] >= r4["zero_col_pct"]) and (r5["zero_col_pct"] >= r1["zero_col_pct"])
        beats_spatial = max(gain_vs_r4_comp, gain_vs_r4_wait) >= 3.0
        beats_det = max(gain_vs_r1_comp, gain_vs_r1_wait) >= 3.0
        go_decision = bool(non_deg_safe and beats_spatial and beats_det)
        verdict_notes.append(f"R5_RAMP_PPO vs R4_Spatial_Intent_PPO: comp={gain_vs_r4_comp:+.1f}%, wait={gain_vs_r4_wait:+.1f}%")
        verdict_notes.append(f"R5_RAMP_PPO vs R1_Det_ResMargin: comp={gain_vs_r1_comp:+.1f}%, wait={gain_vs_r1_wait:+.1f}%")
        verdict_notes.append(f"Safety non-degradation: {non_deg_safe} (R5={r5['zero_col_pct']}%, R4={r4['zero_col_pct']}%, R1={r1['zero_col_pct']}%)")
        verdict_notes.append(f"Verdict: {'GO (RAMP-PPO provides independent >=3% value over both spatial-intent RL and deterministic reservation selector)' if go_decision else 'NO-GO / PARTIAL (RAMP-PPO did not clear >=3% over both R4 and R1; report honestly without inflating complexity)'}")

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "summary": summary,
        "paired_vs_ramp_ppo": paired_vs_ramp,
        "stress_single_var_and_compound": stress_rows,
        "go_decision": go_decision,
        "verdict_notes": verdict_notes,
    }
    (out_dir / "ramp_study_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# RAMP-Primitive-Study Confirmation Report (`Gate 2–3`)",
        "",
        f"> Elapsed: `{payload['elapsed_s']} s` · Held-out seeds: `{eval_seeds}` · Training seeds: `{train_seeds}`",
        "",
        "## 1. Independent Confirmation Across 4 Regimes (`Occluded_Jct_6R`, `MapA_8R`, `MapB_12R`, `Workshop_8R`)",
        "",
        "| Controller | ZeroCol (%) | Completed | Makespan (s) | Wait Ratio | Shield Interv | Mean Clear (m) | Mean Jerk | Ctrl (ms/bot) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for cname, agg in summary.items():
        md.append(
            f"| `{cname}` | {agg['zero_col_pct']:.1f}% | {agg['completed_avg']:.2f} | {agg['makespan_avg_s']:.2f} | "
            f"{agg['wait_ratio_avg']:.4f} | {agg['shield_interv_avg']:.1f} | {agg['min_clear_avg_m']:.3f} | "
            f"{agg['jerk_avg']:.3f} | {agg['ctrl_ms_avg']:.3f} |"
        )
    if paired_vs_ramp:
        md.extend(["", "## 2. Paired Wilcoxon (`R5_RAMP_PPO` vs Comparators)", "", "| Comparator | `ΔCompleted` | `p_comp` | `ΔWaitRatio` | `p_wait` |", "| :--- | :---: | :---: | :---: | :---: |"])
        for cname, pr in paired_vs_ramp.items():
            md.append(f"| `{cname}` | {pr['delta_comp']:+.2f} | {pr['p_comp']:.4f} | {pr['delta_wait']:+.4f} | {pr['p_wait']:.4f} |")

    md.extend(["", "## 3. Single-Variable vs Compound Stress (`Occluded_Junction`)", "", "| Stress Condition | Controller | ZeroCol (%) | Completed | Wait Ratio | Shield Interv |", "| :--- | :--- | :---: | :---: | :---: | :---: |"])
    for r in stress_rows:
        md.append(f"| `{r['stress']}` | `{r['controller']}` | {r['zero_col_pct']:.1f}% | {r['completed_avg']:.2f}/12 | {r['wait_ratio_avg']:.4f} | {r['shield_interv_avg']:.1f} |")

    md.extend(["", "## 4. GO / NO-GO Verdict", ""])
    for note in verdict_notes:
        md.append(f"- {note}")

    (out_dir / "ramp_study_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"=== Wrote {out_dir / 'ramp_study_report.md'} (go={go_decision}) ===", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 2-3 RAMP-PPO Training & Independent Confirmation")
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--eval-all", action="store_true")
    parser.add_argument("--iters", type=int, default=12)
    args = parser.parse_args()
    if args.train_all:
        train_all_ramp_variants(n_iters=args.iters)
    if args.eval_all:
        run_ramp_eval()


if __name__ == "__main__":
    main()
