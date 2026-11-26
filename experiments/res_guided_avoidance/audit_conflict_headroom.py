"""Task 4: Conflict-Level Passing-Order Decision Headroom (`Route B / STCP-RL` Audit).

Keeps CPS-A* global paths and `R1_Robust` as the FIXED lower-level executor.
Enforces strict execution-commitment safety:
  - Already-entered robots (d < 0.9 m) retain NON-REVOCABLE priority.
  - Compares passing-order strategies over uncommitted conflict entrants:
      1. `R1_Default_Order`
      2. `Orig_Res_Order` (CPS-A* reservation timestamp order)
      3. `FCFS` (physical distance/ETA order)
      4. `SPT_Clear` (shortest remaining path to clear)
      5. `Queue_Downstream_Det` (waiting queue + downstream slot congestion)
  - Evaluates `Conflict_Order_Oracle` via matched counterfactual rollouts over safe
    legal passing permutations at real multi-robot conflicts (`N_comp >= 2`).
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.res_guided_avoidance.gate1_headroom_audit import m2_res_margin_selector
from experiments.res_guided_avoidance.geometry_and_shield import DT_MICRO
from experiments.res_guided_avoidance.metric_world import (
    MultiRobotEpisodeResult,
    MultiRobotUnicycleSim,
    build_grid_metric_map,
    build_occluded_junction_map,
    build_workshop_metric_map,
)


def _compute_dynamic_ranks(sim: MultiRobotUnicycleSim, strategy: str) -> dict[str, int]:
    """Compute per-robot priority rank `(lower = higher priority)` for `strategy`."""
    if strategy == "R1_Default_Order":
        return {}

    scores: list[tuple[float, str]] = []
    res_view = sim.get_reservation_view()
    for rid in sim.robot_ids:
        rb = sim.robots[rid]
        if rb.phase == "idle":
            scores.append((1e9, rid))
            continue
        rem = sim.remaining_route_dist_m(rb)
        up = rb.route_nodes[rb.route_idx : rb.route_idx + 3]

        if strategy == "Orig_Res_Order":
            t_en = 1e6
            segs = res_view.get(rid, [])
            for nd_o, te, _, _ in segs:
                if nd_o in up:
                    t_en = float(te)
                    break
            scores.append((t_en, rid))
        elif strategy == "FCFS":
            nxt = sim.mmap.network.node(up[1] if len(up) > 1 else up[0])
            d_nxt = math.hypot(nxt.x - rb.x, nxt.y - rb.y)
            scores.append((d_nxt / max(0.3, rb.v + 0.5), rid))
        elif strategy == "SPT_Clear":
            scores.append((rem, rid))
        elif strategy == "Queue_Downstream_Det":
            # Count followers behind ego + downstream congestion on ego next-2 hops
            behind_cnt = 0
            for oid in sim.robot_ids:
                if oid == rid or sim.robots[oid].phase == "idle":
                    continue
                ob = sim.robots[oid]
                if math.hypot(ob.x - rb.x, ob.y - rb.y) < 5.0 and ob.route_nodes and any(n in up for n in ob.route_nodes[ob.route_idx : ob.route_idx + 3]):
                    behind_cnt += 1
            cong = 0.0
            for nid in up[1:]:
                cong += float(sim.field.node_congestion(nid, sim.macro_tick)) if hasattr(sim.field, "node_congestion") else 0.0
            # Prefer clearing robots with followers behind and clear downstream exits
            sc = -1.5 * behind_cnt + 0.8 * cong + 0.1 * rem
            scores.append((sc, rid))
        else:
            scores.append((0.0, rid))

    scores.sort(key=lambda t: (t[0], t[1]))
    return {rid: rank for rank, (_, rid) in enumerate(scores)}


def _detect_conflict_cluster(sim: MultiRobotUnicycleSim) -> list[str]:
    """Identify >=2 uncommitted robots sharing an upcoming conflict zone within 4.5 m."""
    cands: dict[str, list[str]] = {}
    for rid in sim.robot_ids:
        rb = sim.robots[rid]
        if rb.phase == "idle" or not rb.route_nodes:
            continue
        for nid in rb.route_nodes[rb.route_idx : rb.route_idx + 3]:
            nd = sim.mmap.network.node(nid)
            d = math.hypot(nd.x - rb.x, nd.y - rb.y)
            if 0.9 <= d <= 4.5:
                cands.setdefault(nid, []).append(rid)
    for _, rids in cands.items():
        if len(rids) >= 2:
            return rids[:4]
    return []


def run_conflict_strategy_episode(sim: MultiRobotUnicycleSim, strategy: str) -> MultiRobotEpisodeResult:
    while True:
        sim.row_priority_ranks = _compute_dynamic_ranks(sim, strategy)
        done = sim.step(m2_res_margin_selector, controller_name=strategy)
        if done:
            break
    tot_col = sim.wall_collisions + sim.peer_collisions + sim.intruder_collisions
    return MultiRobotEpisodeResult(
        map_name=sim.mmap.name,
        controller=strategy,
        n_robots=sim.n_robots,
        n_tasks=sim.n_tasks,
        seed=sim.seed,
        completed_tasks=sim.completed_tasks,
        makespan_s=round(sim.t_s, 2),
        zero_collisions=(tot_col == 0),
        wall_collisions=sim.wall_collisions,
        peer_collisions=sim.peer_collisions,
        intruder_collisions=sim.intruder_collisions,
        shield_interventions=sim.shield_interventions,
        shield_infeasible_count=sim.shield_infeasible_count,
        wait_ratio=round(sim._wait_steps / max(1, sim._total_robot_steps), 4),
        mean_jerk=round(sim._jerk_sum / max(1, sim._total_robot_steps), 3),
        replans=sim.replans,
        mean_ctrl_ms=round(float(np.mean(sim._ctrl_ms_samples)) if sim._ctrl_ms_samples else 0.0, 4),
        mean_min_clear_m=round(float(np.mean(sim._min_clear_samples)) if sim._min_clear_samples else 9.0, 3),
    )


def audit_conflict_snapshot_oracle() -> dict[str, Any]:
    """Sample real multi-robot conflict snapshots and evaluate ALL legal passing orders via clone_snapshot()."""
    om = build_occluded_junction_map(cell_m=2.6)
    ma = build_grid_metric_map("Map_A", cell_m=2.6)
    mb = build_grid_metric_map("Map_B", cell_m=2.6)
    wm = build_workshop_metric_map()

    conflict_snaps: list[dict[str, Any]] = []
    for sc_name, mp, nr, nt, hz, ni in (
        ("Occluded_Jct_6R", om, 6, 12, 70.0, 2),
        ("MapA_8R", ma, 8, 16, 80.0, 2),
        ("MapB_12R", mb, 12, 20, 85.0, 2),
        ("Workshop_8R", wm, 8, 12, 160.0, 1),
    ):
        sim = MultiRobotUnicycleSim(mp, n_robots=nr, n_tasks=nt, horizon_s=hz, seed=6001, n_intruders=ni, robust_safety=True)
        last_t = -10.0
        for st in range(int((hz - 14.0) / DT_MICRO)):
            cluster = _detect_conflict_cluster(sim)
            if len(cluster) >= 2 and (sim.t_s - last_t) >= 5.0 and len(conflict_snaps) < 36:
                last_t = sim.t_s
                # Compare Orig_Res_Order vs Queue_Downstream_Det vs all safe permutations over 10.0s (40 steps)
                perms = list(itertools.permutations(cluster[:3]))
                base_ranks_orig = _compute_dynamic_ranks(sim, "Orig_Res_Order")
                base_ranks_qdet = _compute_dynamic_ranks(sim, "Queue_Downstream_Det")

                def _eval_ranks(ranks: dict[str, int]) -> tuple[float, float, int]:
                    cl = sim.clone_snapshot()
                    cl.row_priority_ranks = dict(ranks)
                    s_comp = cl.completed_tasks
                    s_rem = {r: cl.remaining_route_dist_m(cl.robots[r]) for r in cl.robot_ids}
                    w_tot = 0
                    for _ in range(40):  # 10.0s clearing horizon
                        done = cl.step(m2_res_margin_selector, controller_name="OrderEval")
                        w_tot += sum(1 for r in cluster if cl.robots[r].v < 0.15 and cl.robots[r].phase != "idle")
                        if done:
                            break
                    col = cl.wall_collisions + cl.peer_collisions + cl.intruder_collisions
                    prog = sum(max(-2.0, s_rem[r] - cl.remaining_route_dist_m(cl.robots[r])) for r in cl.robot_ids if cl.robots[r].phase != "idle") + 18.0 * (cl.completed_tasks - s_comp)
                    cost = (-prog + 0.10 * w_tot) if col == 0 else 1e6
                    return cost, prog, col

                c_r1def, p_r1def, col_r1def = _eval_ranks({})
                c_orig, p_orig, col_orig = _eval_ranks(base_ranks_orig)
                c_qdet, p_qdet, col_qdet = _eval_ranks(base_ranks_qdet)
                best_perm_c = min(c_r1def, c_orig)
                best_perm_p = p_r1def if c_r1def <= c_orig else p_orig
                safe_perms = 0
                for pm in perms:
                    rk = dict(base_ranks_orig)
                    for idx_r, rid_p in enumerate(pm):
                        rk[rid_p] = idx_r - 100
                    cp, pp, colp = _eval_ranks(rk)
                    if colp == 0:
                        safe_perms += 1
                        if cp < best_perm_c:
                            best_perm_c = cp
                            best_perm_p = pp
                gain_over_orig_pct = 100.0 * max(0.0, c_orig - best_perm_c) / max(1.0, abs(p_orig))
                gain_over_r1def_pct = 100.0 * max(0.0, c_r1def - best_perm_c) / max(1.0, abs(p_r1def))
                best_det_c = min(c_r1def, c_orig, c_qdet)
                best_det_p = max(p_r1def, p_orig, p_qdet)
                gain_over_bestdet_pct = 100.0 * max(0.0, best_det_c - best_perm_c) / max(1.0, abs(best_det_p))
                conflict_snaps.append({
                    "scenario": sc_name,
                    "t_s": sim.t_s,
                    "cluster_size": len(cluster),
                    "safe_perms": safe_perms,
                    "cost_r1def": round(c_r1def, 3),
                    "cost_orig": round(c_orig, 3),
                    "cost_qdet": round(c_qdet, 3),
                    "cost_oracle": round(best_perm_c, 3),
                    "oracle_gain_pct": round(gain_over_orig_pct, 2),
                    "oracle_gain_over_r1def_pct": round(gain_over_r1def_pct, 2),
                    "oracle_gain_over_bestdet_pct": round(gain_over_bestdet_pct, 2),
                })
            if sim.step(m2_res_margin_selector, controller_name="R1_Robust"):
                break
    return {
        "n_conflicts_audited": len(conflict_snaps),
        "mean_oracle_gain_over_orig_pct": round(float(np.mean([c["oracle_gain_pct"] for c in conflict_snaps])), 2) if conflict_snaps else 0.0,
        "mean_oracle_gain_over_r1def_pct": round(float(np.mean([c["oracle_gain_over_r1def_pct"] for c in conflict_snaps])), 2) if conflict_snaps else 0.0,
        "mean_oracle_gain_over_bestdet_pct": round(float(np.mean([c["oracle_gain_over_bestdet_pct"] for c in conflict_snaps])), 2) if conflict_snaps else 0.0,
        "pct_conflicts_gain_ge_3pct": round(100.0 * sum(1 for c in conflict_snaps if c["oracle_gain_over_bestdet_pct"] >= 3.0) / max(1, len(conflict_snaps)), 1),
        "conflicts": conflict_snaps,
    }


def main() -> None:
    out_dir = ROOT / "out" / "decision_headroom"
    out_dir.mkdir(parents=True, exist_ok=True)

    om = build_occluded_junction_map(cell_m=2.6)
    ma = build_grid_metric_map("Map_A", cell_m=2.6)
    mb = build_grid_metric_map("Map_B", cell_m=2.6)
    wm = build_workshop_metric_map()

    strategies = ("R1_Default_Order", "Orig_Res_Order", "FCFS", "SPT_Clear", "Queue_Downstream_Det")
    specs = [
        ("Occluded_Jct_6R", om, 6, 12, 70.0, 2, 0.0, 0.0),
        ("MapA_8R", ma, 8, 16, 80.0, 2, 0.0, 0.0),
        ("MapB_12R", mb, 12, 20, 85.0, 2, 0.0, 0.0),
        ("Workshop_8R", wm, 8, 12, 160.0, 1, 0.0, 0.0),
        ("Occluded_Jct_6R_Noise", om, 6, 12, 70.0, 2, 0.12, 0.20),
    ]
    seeds = (6001, 6002, 6003)

    summary: dict[str, Any] = {}
    for st in strategies:
        rows: list[MultiRobotEpisodeResult] = []
        for _, mmap, nr, nt, hz, ni, np_m, nv_m in specs:
            for sd in seeds:
                sim = MultiRobotUnicycleSim(mmap, n_robots=nr, n_tasks=nt, horizon_s=hz, seed=sd, n_intruders=ni, obs_noise_pos_m=np_m, obs_noise_vel_mps=nv_m, robust_safety=True)
                rows.append(run_conflict_strategy_episode(sim, st))
        n = max(1, len(rows))
        summary[st] = {
            "episodes": len(rows),
            "zero_col_pct": round(100.0 * sum(1 for r in rows if r.zero_collisions) / n, 1),
            "completed_avg": round(float(np.mean([r.completed_tasks for r in rows])), 3),
            "wait_ratio_avg": round(float(np.mean([r.wait_ratio for r in rows])), 4),
            "shield_interv_avg": round(float(np.mean([r.shield_interventions for r in rows])), 1),
        }
        print(f"[Task4 ConflictOrder | {st:21s}] zcol={summary[st]['zero_col_pct']:5.1f}% comp={summary[st]['completed_avg']:.3f} wait={summary[st]['wait_ratio_avg']:.4f} sh={summary[st]['shield_interv_avg']:.1f}", flush=True)

    oracle_audit = audit_conflict_snapshot_oracle()
    print(f"[Task4 Snapshot Oracle | N={oracle_audit['n_conflicts_audited']}] mean_gain={oracle_audit['mean_oracle_gain_over_orig_pct']:+.2f}% (>=3%: {oracle_audit['pct_conflicts_gain_ge_3pct']:.1f}%)", flush=True)

    (out_dir / "task4_conflict_headroom.json").write_text(
        json.dumps({"episode_strategies": summary, "conflict_snapshot_oracle": oracle_audit}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
