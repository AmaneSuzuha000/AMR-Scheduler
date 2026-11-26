"""Task 3: Matched Multi-Horizon Counterfactual Oracle (`Route A` Headroom Audit).

From each snapshot `s`, clones `MultiRobotUnicycleSim` (bitwise-identical RNG states
and reservations), applies each safe primitive `a in A_safe(s)` at t=0, and rolls
ALL robots forward using identical `R1_Robust` (`m2_res_margin_selector` + `robust_safety=True`)
across H in {0.5s, 2.0s, 5.0s, 10.0s, H_clear <= 15.0s}.
Measures exact J(s, a, H), Delta(s, H) >= 0, candidate coverage loss, and selection headroom.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.res_guided_avoidance.audit_primitive_freedom import (
    _classify_regime,
    _is_prim_shield_safe,
    _pick_id_from_v,
)
from experiments.res_guided_avoidance.gate1_headroom_audit import m2_res_margin_selector
from experiments.res_guided_avoidance.geometry_and_shield import DT_MICRO
from experiments.res_guided_avoidance.metric_world import (
    MultiRobotUnicycleSim,
    build_grid_metric_map,
    build_occluded_junction_map,
    build_workshop_metric_map,
)
from experiments.res_guided_avoidance.motion_primitives import generate_primitive_rollouts


HORIZONS_STEPS = (
    ("0.5s", 2),
    ("2.0s", 8),
    ("5.0s", 20),
    ("10.0s", 40),
    ("clear_15s", 60),
)


def _rollout_candidate_horizons(
    sim: MultiRobotUnicycleSim,
    target_rid: str,
    v_first: float,
    w_first: float,
    max_steps: int = 60,
) -> dict[str, dict[str, float]]:
    cl = sim.clone_snapshot()
    rb0 = cl.robots[target_rid]
    start_comp = cl.completed_tasks
    start_sh = cl.shield_interventions
    start_rem = cl.remaining_route_dist_m(rb0)
    # Track total forward motion + task completions across fleet
    fleet_start_rem = {r: cl.remaining_route_dist_m(cl.robots[r]) for r in cl.robot_ids}
    wait_i = 0

    def _pol_step0(s: MultiRobotUnicycleSim, rid: str) -> tuple[float, float]:
        if rid == target_rid:
            return v_first, w_first
        return m2_res_margin_selector(s, rid)

    checkpoints = {h_steps: lbl for lbl, h_steps in HORIZONS_STEPS}
    out: dict[str, dict[str, float]] = {}

    for st in range(1, max_steps + 1):
        fn = _pol_step0 if st == 1 else m2_res_margin_selector
        done = cl.step(fn, controller_name="CF_Branch")
        if cl.robots[target_rid].v < 0.15 and cl.robots[target_rid].phase != "idle":
            wait_i += 1
        if st in checkpoints or done:
            # Progress = reduction in remaining distance + 20m bonus per completed task
            d_comp = cl.completed_tasks - start_comp
            fleet_prog = sum(
                max(-2.0, fleet_start_rem[r] - cl.remaining_route_dist_m(cl.robots[r]))
                for r in cl.robot_ids
                if cl.robots[r].phase != "idle"
            ) + 18.0 * d_comp
            ego_prog = max(-2.0, start_rem - cl.remaining_route_dist_m(cl.robots[target_rid])) + 18.0 * d_comp
            d_sh = cl.shield_interventions - start_sh
            col = cl.wall_collisions + cl.peer_collisions + cl.intruder_collisions
            # Cost J to minimize (penalize collisions overwhelmingly)
            J_ego = (-ego_prog + 0.12 * wait_i + 0.08 * d_sh) if col == 0 else (1e6 + col)
            J_fleet = (-fleet_prog + 0.12 * wait_i + 0.04 * d_sh) if col == 0 else (1e6 + col)
            rec = {
                "J_ego": float(J_ego),
                "J_fleet": float(J_fleet),
                "ego_prog_m": float(ego_prog),
                "fleet_prog_m": float(fleet_prog),
                "wait_i": float(wait_i),
                "col": float(col),
            }
            for lbl, hs in HORIZONS_STEPS:
                if hs == st or (done and lbl not in out):
                    out[lbl] = rec
        if done:
            break
    return out


def main() -> None:
    out_dir = ROOT / "out" / "decision_headroom"
    out_dir.mkdir(parents=True, exist_ok=True)

    om = build_occluded_junction_map(cell_m=2.6)
    ma = build_grid_metric_map("Map_A", cell_m=2.6)
    mb = build_grid_metric_map("Map_B", cell_m=2.6)
    wm = build_workshop_metric_map()

    scenarios = [
        ("Occluded_Jct_6R_Clean", om, 6, 12, 70.0, 2, 0.0, 0.0),
        ("MapA_8R_Clean", ma, 8, 16, 80.0, 2, 0.0, 0.0),
        ("MapB_12R_Clean", mb, 12, 20, 85.0, 2, 0.0, 0.0),
        ("Workshop_8R_Clean", wm, 8, 12, 160.0, 1, 0.0, 0.0),
        ("Occluded_Jct_6R_Noise0.12", om, 6, 12, 70.0, 2, 0.12, 0.20),
    ]

    snapshots: list[dict[str, Any]] = []
    max_snaps_per_sc = 12

    for sc_name, mmap, nr, nt, hz, ni, np_m, nv_m in scenarios:
        sim = MultiRobotUnicycleSim(mmap, n_robots=nr, n_tasks=nt, horizon_s=hz, seed=6001, n_intruders=ni, obs_noise_pos_m=np_m, obs_noise_vel_mps=nv_m, robust_safety=True)
        sc_count = 0
        for step_i in range(int((hz - 16.0) / DT_MICRO)):
            if step_i >= 12 and step_i % 10 == 0 and sc_count < max_snaps_per_sc:
                # Prefer conflict/non-trivial snapshots with K_safe >= 2
                for rid in sim.robot_ids:
                    rb = sim.robots[rid]
                    if rb.phase == "idle" or rb.service_rem_s > 0.0:
                        continue
                    rolls = generate_primitive_rollouts(sim, rid)
                    obs_vis = sim.get_perceived_circles(rid)
                    safe_ids = [r.prim_id for r in rolls if r.legal and _is_prim_shield_safe(sim, rid, r.v_cmd, r.w_cmd, obs_vis)]
                    if len(safe_ids) < 2:
                        continue
                    reg = _classify_regime(sim, rid)
                    v1, _ = m2_res_margin_selector(sim, rid)
                    p_r1 = _pick_id_from_v(rolls, v1)
                    if p_r1 not in safe_ids:
                        p_r1 = safe_ids[0]

                    geom_ids = [r.prim_id for r in rolls if r.legal]
                    eval_by_id: dict[int, dict[str, dict[str, float]]] = {}
                    for pid in geom_ids:
                        eval_by_id[pid] = _rollout_candidate_horizons(sim, rid, rolls[pid].v_cmd, rolls[pid].w_cmd)

                    snap_res: dict[str, Any] = {"scenario": sc_name, "regime": reg, "k_safe": len(safe_ids), "k_geom": len(geom_ids), "horizons": {}}
                    for lbl, _ in HORIZONS_STEPS:
                        j_r1 = eval_by_id[p_r1][lbl]["J_ego"]
                        j_safe_best = min(eval_by_id[pid][lbl]["J_ego"] for pid in safe_ids)
                        j_geom_best = min(eval_by_id[pid][lbl]["J_ego"] for pid in geom_ids)
                        delta_sel = max(0.0, j_r1 - j_safe_best)
                        cov_loss = max(0.0, j_safe_best - j_geom_best)
                        prog_ref = max(0.5, abs(eval_by_id[p_r1][lbl]["ego_prog_m"]))

                        jf_r1 = eval_by_id[p_r1][lbl]["J_fleet"]
                        jf_safe_best = min(eval_by_id[pid][lbl]["J_fleet"] for pid in safe_ids)
                        jf_geom_best = min(eval_by_id[pid][lbl]["J_fleet"] for pid in geom_ids)
                        delta_fleet = max(0.0, jf_r1 - jf_safe_best)
                        cov_loss_fleet = max(0.0, jf_safe_best - jf_geom_best)
                        fprog_ref = max(1.0, abs(eval_by_id[p_r1][lbl]["fleet_prog_m"]))

                        snap_res["horizons"][lbl] = {
                            "delta_sel": round(delta_sel, 4),
                            "rel_gain_pct": round(100.0 * delta_sel / prog_ref, 2),
                            "cov_loss": round(cov_loss, 4),
                            "j_r1": round(j_r1, 4),
                            "j_safe_best": round(j_safe_best, 4),
                            "delta_fleet": round(delta_fleet, 4),
                            "fleet_rel_gain_pct": round(100.0 * delta_fleet / fprog_ref, 2),
                            "cov_loss_fleet": round(cov_loss_fleet, 4),
                        }
                    snapshots.append(snap_res)
                    sc_count += 1
                    break
            if sim.step(m2_res_margin_selector, controller_name="R1_Robust"):
                break

    by_horizon: dict[str, Any] = {}
    for lbl, _ in HORIZONS_STEPS:
        ds = [s["horizons"][lbl]["delta_sel"] for s in snapshots]
        rg = [s["horizons"][lbl]["rel_gain_pct"] for s in snapshots]
        cl = [s["horizons"][lbl]["cov_loss"] for s in snapshots]
        df = [s["horizons"][lbl]["delta_fleet"] for s in snapshots]
        frg = [s["horizons"][lbl]["fleet_rel_gain_pct"] for s in snapshots]
        clf = [s["horizons"][lbl]["cov_loss_fleet"] for s in snapshots]
        nonzero_pct = 100.0 * sum(1 for x in ds if x > 0.05) / max(1, len(ds))
        ge3_pct = 100.0 * sum(1 for x in rg if x >= 3.0) / max(1, len(rg))
        fge3_pct = 100.0 * sum(1 for x in frg if x >= 3.0) / max(1, len(frg))
        by_horizon[lbl] = {
            "n_snapshots": len(snapshots),
            "mean_delta_sel": round(float(np.mean(ds)), 4) if ds else 0.0,
            "median_delta_sel": round(float(np.median(ds)), 4) if ds else 0.0,
            "mean_rel_gain_pct": round(float(np.mean(rg)), 2) if rg else 0.0,
            "pct_delta_gt_0.05": round(nonzero_pct, 1),
            "pct_gain_ge_3pct": round(ge3_pct, 1),
            "mean_cov_loss": round(float(np.mean(cl)), 4) if cl else 0.0,
            "mean_delta_fleet": round(float(np.mean(df)), 4) if df else 0.0,
            "mean_fleet_rel_gain_pct": round(float(np.mean(frg)), 2) if frg else 0.0,
            "fleet_pct_gain_ge_3pct": round(fge3_pct, 1),
            "mean_cov_loss_fleet": round(float(np.mean(clf)), 4) if clf else 0.0,
        }
        print(f"[Task3 CF-Oracle | H={lbl:9s}] ego_rel={by_horizon[lbl]['mean_rel_gain_pct']:+.2f}% (>=3%: {ge3_pct:.1f}%) | fleet_rel={by_horizon[lbl]['mean_fleet_rel_gain_pct']:+.2f}% (>=3%: {fge3_pct:.1f}%) cov_loss={by_horizon[lbl]['mean_cov_loss']:.3f}", flush=True)

    # Also summarize by regime at H=5.0s and H=clear_15s
    by_regime: dict[str, Any] = {}
    for reg in sorted({s["regime"] for s in snapshots}):
        sub = [s for s in snapshots if s["regime"] == reg]
        by_regime[reg] = {
            "count": len(sub),
            "H_2s_rel_gain_pct": round(float(np.mean([s["horizons"]["2.0s"]["rel_gain_pct"] for s in sub])), 2),
            "H_5s_rel_gain_pct": round(float(np.mean([s["horizons"]["5.0s"]["rel_gain_pct"] for s in sub])), 2),
            "H_clear_rel_gain_pct": round(float(np.mean([s["horizons"]["clear_15s"]["rel_gain_pct"] for s in sub])), 2),
        }

    report = {"n_snapshots": len(snapshots), "by_horizon": by_horizon, "by_regime": by_regime}
    (out_dir / "task3_counterfactual_headroom.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
