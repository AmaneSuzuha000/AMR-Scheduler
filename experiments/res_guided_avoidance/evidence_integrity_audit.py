"""Task 0: Evidence Integrity Audit.

1. Compares Gate 0 under original test conditions (`start_sep_m=1.8`, Workshop 130s)
   vs modified conditions (`start_sep_m=2.2`, Workshop 300s) so code fixes vs test
   condition changes are transparently separated.
2. Exports full per-episode CSV (`out/ramp_robust/gate3_per_episode.csv`) and
   per-training-seed breakdown (`s42, s43, s44`) across all R0..R6 and external baselines,
   with paired Wilcoxon + bootstrap 95% CI on BOTH `completed_tasks` and `wait_ratio`.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

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
from experiments.res_guided_avoidance.gate0_execution_audit import (
    _deterministic_follower,
    _make_headon_bay_map,
    _make_straight_corridor_map,
    _make_turn90_map,
)
from experiments.res_guided_avoidance.gate1_headroom_audit import (
    m0_perception_selector,
    m2_res_margin_selector,
)
from experiments.res_guided_avoidance.metric_world import (
    ContTask,
    DynamicIntruder,
    MultiRobotEpisodeResult,
    MultiRobotUnicycleSim,
    build_grid_metric_map,
    build_occluded_junction_map,
    build_workshop_metric_map,
)
from experiments.res_guided_avoidance.ramp_ppo import RAMP_CKPT_DIR, NumpyRampPolicy
from experiments.res_guided_avoidance.rg_ppo import CKPT_DIR as RG_CKPT_DIR, NumpyMultiRobotPolicy


def _bootstrap_ci(vals: list[float], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    means = np.mean(rng.choice(arr, size=(n_boot, len(arr)), replace=True), axis=1)
    return round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)


def _paired_wilcoxon_p(a: list[float], b: list[float]) -> float:
    diffs = [x - y for x, y in zip(a, b) if abs(x - y) > 1e-9]
    if len(diffs) < 2:
        return 1.0
    try:
        _, p = wilcoxon(diffs)
        return float(p)
    except Exception:
        return 1.0


def audit_gate0_conditions() -> list[dict[str, Any]]:
    """Compare Gate 0 fixed code under original conditions vs updated conditions."""
    wm = build_workshop_metric_map()
    om = build_occluded_junction_map(cell_m=2.6)
    out: list[dict[str, Any]] = []
    for cond_name, sep_m, ws_hz in (
        ("OriginalCond_sep1.8_ws130s", 1.8, 130.0),
        ("UpdatedCond_sep2.2_ws300s", 2.2, 300.0),
    ):
        specs = [
            ("1R_Straight", _make_straight_corridor_map(), 1, 1, 35.0, 1, None),
            ("1R_Turn90", _make_turn90_map(), 1, 1, 45.0, 1, None),
            ("1R_StaticBypass", _make_straight_corridor_map(half_width_m=1.65), 1, 1, 40.0, 1, "static"),
            ("1R_DynamicAvoid", _make_straight_corridor_map(half_width_m=1.65), 1, 1, 45.0, 1, "dyn"),
            ("2R_CrossJunction", om, 2, 2, 50.0, 7, None),
            ("2R_HeadOnYield", _make_headon_bay_map(), 2, 2, 60.0, 1, "headon"),
            ("4R_LifelongWorkshop", wm, 4, 4, ws_hz, 42, None),
        ]
        for sc_name, mmap, nr, nt, hz, sd, extra in specs:
            sim = MultiRobotUnicycleSim(mmap, n_robots=nr, n_tasks=nt, horizon_s=hz, seed=sd, n_intruders=0, start_sep_m=sep_m)
            if extra == "static":
                sim.intruders.append(DynamicIntruder(x=8.0, y=0.25, vx=0.0, vy=0.0, radius=0.35, x_lo=7.5, x_hi=8.5, y_lo=-0.5, y_hi=0.5))
            elif extra == "dyn":
                sim.intruders.append(DynamicIntruder(x=10.5, y=0.0, vx=-0.35, vy=0.0, radius=0.32, x_lo=4.0, x_hi=13.0, y_lo=-0.5, y_hi=0.5))
            elif extra == "headon":
                sim.pending.clear()
                sim.ready.clear()
                sim.ready.extend([ContTask(0, "W0", "E0", release_s=0.0, due_s=60.0), ContTask(1, "E0", "W0", release_s=0.0, due_s=60.0)])
                r0, r1 = sim.robots["R_00"], sim.robots["R_01"]
                nw, ne = sim.mmap.network.node("W0"), sim.mmap.network.node("E0")
                r0.x, r0.y, r0.theta, r0.cur_node, r0.route_nodes = nw.x, nw.y, 0.0, "W0", ["W0"]
                r1.x, r1.y, r1.theta, r1.cur_node, r1.route_nodes = ne.x, ne.y, math.pi, "E0", ["E0"]
            res = sim.run_episode(_deterministic_follower, controller_name=cond_name)
            row = {
                "condition": cond_name,
                "scenario": sc_name,
                "passed": bool(res.zero_collisions and res.completed_tasks >= nt),
                "completed": f"{res.completed_tasks}/{nt}",
                "makespan_s": res.makespan_s,
                "zero_col": res.zero_collisions,
                "wait_ratio": res.wait_ratio,
                "shield_interv": res.shield_interventions,
            }
            out.append(row)
            print(f"[G0-Audit | {cond_name:24s} | {sc_name:19s}] pass={row['passed']} comp={row['completed']} ms={row['makespan_s']}s zcol={row['zero_col']}", flush=True)
    return out


def audit_per_seed_and_export_csv(out_dir: Path) -> dict[str, Any]:
    """Export per-episode CSV + per-training-seed breakdown for Gate 3 DEV (6001..6003)."""
    om = build_occluded_junction_map(cell_m=2.6)
    ma = build_grid_metric_map("Map_A", cell_m=2.6)
    mb = build_grid_metric_map("Map_B", cell_m=2.6)
    wm = build_workshop_metric_map()
    conf_specs = [
        ("Occluded_Jct_6R", om, 6, 12, 70.0, 2),
        ("MapA_Metric_8R", ma, 8, 16, 80.0, 2),
        ("MapB_Metric_12R", mb, 12, 20, 85.0, 2),
        ("Workshop_8R", wm, 8, 12, 160.0, 1),
    ]
    eval_seeds = (6001, 6002, 6003)

    ctrls: dict[str, Any] = {
        "R0_Det_Primitive": m0_perception_selector,
        "R1_Det_ResMargin": m2_res_margin_selector,
        "CPSA_Follower": cpsa_follower_controller,
        "ORCA_Uni": orca_uni_controller,
        "DWA": dwa_controller,
        "Res_DWA": res_dwa_controller,
        "Uni_MPC": uni_mpc_controller,
        "TPG_ADG_Follower": tpg_adg_controller,
        "MAWPF_Kin_Ref": mawpf_kin_controller,
    }
    for vname in ("R2_BC_Only", "R3_Primitive_PPO", "R4_Spatial_Intent_PPO", "R5_RAMP_PPO"):
        for sd in (42, 43, 44):
            p = RAMP_CKPT_DIR / f"{vname.lower()}_s{sd}.npz"
            if p.is_file():
                ctrls[f"{vname}_s{sd}"] = NumpyRampPolicy.load_npz(p)
    rg_p = RG_CKPT_DIR / "rg_ppo_s42.npz"
    if rg_p.is_file():
        ctrls["R6_Orig_Cont_RGPPO_s42"] = NumpyMultiRobotPolicy.load_npz(rg_p)

    ep_rows: list[dict[str, Any]] = []
    by_ctrl: dict[str, list[MultiRobotEpisodeResult]] = {c: [] for c in ctrls}
    for sc_label, mmap, nr, nt, hz, ni in conf_specs:
        for cname, pol in ctrls.items():
            for esd in eval_seeds:
                sim = MultiRobotUnicycleSim(mmap, n_robots=nr, n_tasks=nt, horizon_s=hz, seed=esd, n_intruders=ni)
                res = sim.run_episode(pol, controller_name=cname)
                by_ctrl[cname].append(res)
                ep_rows.append({
                    "scenario": sc_label,
                    "controller": cname,
                    "eval_seed": esd,
                    "n_robots": nr,
                    "n_tasks": nt,
                    "completed_tasks": res.completed_tasks,
                    "makespan_s": res.makespan_s,
                    "zero_collisions": res.zero_collisions,
                    "wall_collisions": res.wall_collisions,
                    "peer_collisions": res.peer_collisions,
                    "intruder_collisions": res.intruder_collisions,
                    "total_col_events": res.wall_collisions + res.peer_collisions + res.intruder_collisions,
                    "wait_ratio": res.wait_ratio,
                    "shield_interventions": res.shield_interventions,
                    "shield_infeasible_count": res.shield_infeasible_count,
                    "mean_min_clear_m": res.mean_min_clear_m,
                    "mean_jerk": res.mean_jerk,
                    "mean_ctrl_ms": res.mean_ctrl_ms,
                })

    csv_path = out_dir / "gate3_per_episode.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ep_rows[0].keys()))
        w.writeheader()
        w.writerows(ep_rows)

    per_seed_summary: dict[str, Any] = {}
    for cname, rows in by_ctrl.items():
        n = max(1, len(rows))
        comps = [float(r.completed_tasks) for r in rows]
        waits = [r.wait_ratio for r in rows]
        per_seed_summary[cname] = {
            "episodes": len(rows),
            "zero_col_pct": round(100.0 * sum(1 for r in rows if r.zero_collisions) / n, 1),
            "col_events_tot": int(sum(r.wall_collisions + r.peer_collisions + r.intruder_collisions for r in rows)),
            "col_wall_peer_intr": [
                int(sum(r.wall_collisions for r in rows)),
                int(sum(r.peer_collisions for r in rows)),
                int(sum(r.intruder_collisions for r in rows)),
            ],
            "completed_avg": round(float(np.mean(comps)), 2),
            "completed_ci95": _bootstrap_ci(comps),
            "wait_ratio_avg": round(float(np.mean(waits)), 4),
            "wait_ci95": _bootstrap_ci(waits),
            "shield_interv_avg": round(float(np.mean([r.shield_interventions for r in rows])), 1),
        }

    # Pool across 3 training seeds for R5_RAMP_PPO vs R4 & R1 (36 paired episodes)
    r5_pool = [r for sd in (42, 43, 44) for r in by_ctrl.get(f"R5_RAMP_PPO_s{sd}", [])]
    r4_pool = [r for sd in (42, 43, 44) for r in by_ctrl.get(f"R4_Spatial_Intent_PPO_s{sd}", [])]
    r1_pool = by_ctrl["R1_Det_ResMargin"] * 3
    pooled_stats = {}
    if len(r5_pool) == len(r4_pool) == len(r1_pool) and r5_pool:
        for lbl, ref in (("R4_Spatial_Intent_PPO_3seed", r4_pool), ("R1_Det_ResMargin_3x", r1_pool)):
            c_a = [float(r.completed_tasks) for r in r5_pool]
            c_b = [float(r.completed_tasks) for r in ref]
            w_a = [r.wait_ratio for r in r5_pool]
            w_b = [r.wait_ratio for r in ref]
            pooled_stats[lbl] = {
                "n_paired": len(r5_pool),
                "delta_comp": round(float(np.mean(c_a) - np.mean(c_b)), 3),
                "p_comp_wilcoxon": round(_paired_wilcoxon_p(c_a, c_b), 6),
                "delta_wait": round(float(np.mean(w_a) - np.mean(w_b)), 4),
                "p_wait_wilcoxon": round(_paired_wilcoxon_p(w_a, w_b), 6),
            }

    return {"per_seed_summary": per_seed_summary, "pooled_3seed_stats": pooled_stats, "csv_path": str(csv_path)}


def main() -> None:
    out_dir = ROOT / "out" / "ramp_robust"
    out_dir.mkdir(parents=True, exist_ok=True)
    g0 = audit_gate0_conditions()
    per_sd = audit_per_seed_and_export_csv(out_dir)
    payload = {"gate0_condition_attribution": g0, **per_sd}
    (out_dir / "evidence_integrity_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"=== Wrote {out_dir / 'evidence_integrity_report.json'} & {per_sd['csv_path']} ===", flush=True)


if __name__ == "__main__":
    main()
