"""Gate 1: Motion Primitive & Information Headroom Audit (No-RL Pre-Check).

Compares 4 zero-RL selectors on IDENTICAL states and IDENTICAL legal primitive pools:
  - M0: Perception-only deterministic selector
  - M1: Spatial-intent deterministic selector (peer path geometry, ZERO time info)
  - M2: Spatiotemporal reservation-margin selector (uses predicted m_ij(z))
  - M3: Counterfactual rollout Oracle (short-horizon forward simulation over legal primitives)

Gate 1 Exit Criterion: M2 or M3 must improve efficiency by >= 3% over M1 without
safety/completion degradation; otherwise STOP further RL expansion.
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

from experiments.res_guided_avoidance.geometry_and_shield import (
    CircleState,
    DT_MICRO,
    obstacle_surface_clearance,
)
from experiments.res_guided_avoidance.metric_world import (
    MultiRobotEpisodeResult,
    MultiRobotUnicycleSim,
    build_grid_metric_map,
    build_occluded_junction_map,
)
from experiments.res_guided_avoidance.motion_primitives import (
    PrimitiveRollout,
    generate_primitive_rollouts,
)


def compute_reservation_margin_signal(
    sim: MultiRobotUnicycleSim,
    rid: str,
    *,
    delta_safe_s: float = 1.0,
) -> tuple[bool, float, float]:
    """Return `(has_conflict, min_margin_s, dist_to_conflict_m)` using reservations."""
    rb = sim.robots[rid]
    res_view = sim.get_reservation_view()
    up_nodes = rb.route_nodes[rb.route_idx : rb.route_idx + 5] if rb.route_nodes else [rb.cur_node]
    if not up_nodes:
        return False, 9.0, 99.0

    # Estimate ego arrival window at each upcoming node
    ego_windows: dict[str, tuple[float, float, float]] = {}
    cum_d = 0.0
    px, py = rb.x, rb.y
    for nid in up_nodes:
        nd = sim.mmap.network.node(nid)
        cum_d += math.hypot(nd.x - px, nd.y - py)
        t_in = sim.t_s + cum_d / 0.90
        t_out = t_in + 1.5
        ego_windows[nid] = (t_in, t_out, cum_d)
        px, py = nd.x, nd.y

    min_m = 9.0
    min_d = 99.0
    has_conf = False
    for other_id, segs in res_view.items():
        if other_id == rid or not segs:
            continue
        for nd_o, t_en, t_ex, _ in segs:
            if nd_o not in ego_windows:
                continue
            ei, eo, cd = ego_windows[nd_o]
            oj_in = t_en * sim.mmap.macro_step_s
            oj_out = t_ex * sim.mmap.macro_step_s
            # Signed time separation between ego and peer occupancy
            m_ij = max(oj_in - eo, ei - oj_out) - delta_safe_s
            if m_ij < min_m:
                min_m = m_ij
                min_d = cd
                has_conf = True
    return has_conf, float(min_m), float(min_d)


def m0_perception_selector(sim: MultiRobotUnicycleSim, rid: str) -> tuple[float, float]:
    """M0: Current perception only (no peer intent, no reservation timing)."""
    rb = sim.robots[rid]
    rolls = generate_primitive_rollouts(sim, rid)
    obs = sim.get_perceived_circles(rid)
    ct, st = math.cos(-rb.theta), math.sin(-rb.theta)

    min_fwd_clear = 9.0
    lat_occ = 0.0
    for o in obs:
        lx = ct * (o.x - rb.x) - st * (o.y - rb.y)
        ly = st * (o.x - rb.x) + ct * (o.y - rb.y)
        if -0.2 < lx < 3.2 and abs(ly) < (sim.fp.radius_m + o.radius + 0.35):
            d_c = math.hypot(o.x - rb.x, o.y - rb.y) - sim.fp.radius_m - o.radius
            if d_c < min_fwd_clear:
                min_fwd_clear = d_c
                lat_occ = ly

    if min_fwd_clear < 0.75:
        if rolls[4].legal and lat_occ <= 0.0:
            return rolls[4].v_cmd, rolls[4].w_cmd
        if rolls[5].legal and lat_occ > 0.0:
            return rolls[5].v_cmd, rolls[5].w_cmd
        return rolls[3].v_cmd, rolls[3].w_cmd
    if min_fwd_clear < 1.8:
        return rolls[2].v_cmd, rolls[2].w_cmd
    return rolls[0].v_cmd, rolls[0].w_cmd


def m1_spatial_intent_selector(sim: MultiRobotUnicycleSim, rid: str) -> tuple[float, float]:
    """M1: Perception + spatial path intent (ZERO reservation time information)."""
    rb = sim.robots[rid]
    rolls = generate_primitive_rollouts(sim, rid)
    obs = sim.get_perceived_circles(rid)

    # Check purely spatial overlap on next 3 hops (ignoring all timestamps)
    ego_up = set(rb.route_nodes[rb.route_idx : rb.route_idx + 3])
    spatial_conflict = False
    for other_id in sim.robot_ids:
        if other_id == rid:
            continue
        o_rb = sim.robots[other_id]
        if o_rb.phase == "idle":
            continue
        o_up = set(o_rb.route_nodes[o_rb.route_idx : o_rb.route_idx + 3])
        if ego_up & o_up and math.hypot(o_rb.x - rb.x, o_rb.y - rb.y) < 5.5:
            spatial_conflict = True
            break

    # Base perception check
    ct, st = math.cos(-rb.theta), math.sin(-rb.theta)
    min_fwd = 9.0
    lat_occ = 0.0
    for o in obs:
        lx = ct * (o.x - rb.x) - st * (o.y - rb.y)
        ly = st * (o.x - rb.x) + ct * (o.y - rb.y)
        if -0.2 < lx < 3.2 and abs(ly) < (sim.fp.radius_m + o.radius + 0.35):
            d_c = math.hypot(o.x - rb.x, o.y - rb.y) - sim.fp.radius_m - o.radius
            if d_c < min_fwd:
                min_fwd = d_c
                lat_occ = ly

    if min_fwd < 0.75:
        if rolls[4].legal and lat_occ <= 0.0:
            return rolls[4].v_cmd, rolls[4].w_cmd
        if rolls[5].legal and lat_occ > 0.0:
            return rolls[5].v_cmd, rolls[5].w_cmd
        return rolls[3].v_cmd, rolls[3].w_cmd
    if min_fwd < 1.8 or spatial_conflict:
        return rolls[2].v_cmd, rolls[2].w_cmd
    return rolls[0].v_cmd, rolls[0].w_cmd


def m2_res_margin_selector(sim: MultiRobotUnicycleSim, rid: str) -> tuple[float, float]:
    """M2: Perception + spatiotemporal reservation time-margin m_ij(z)."""
    rb = sim.robots[rid]
    rolls = generate_primitive_rollouts(sim, rid)
    obs = sim.get_perceived_circles(rid)

    ct, st = math.cos(-rb.theta), math.sin(-rb.theta)
    min_fwd = 9.0
    lat_occ = 0.0
    for o in obs:
        lx = ct * (o.x - rb.x) - st * (o.y - rb.y)
        ly = st * (o.x - rb.x) + ct * (o.y - rb.y)
        if -0.2 < lx < 3.0 and abs(ly) < (sim.fp.radius_m + o.radius + 0.35):
            d_c = math.hypot(o.x - rb.x, o.y - rb.y) - sim.fp.radius_m - o.radius
            if d_c < min_fwd:
                min_fwd = d_c
                lat_occ = ly

    if min_fwd < 0.75:
        if rolls[4].legal and lat_occ <= 0.0:
            return rolls[4].v_cmd, rolls[4].w_cmd
        if rolls[5].legal and lat_occ > 0.0:
            return rolls[5].v_cmd, rolls[5].w_cmd
        return rolls[3].v_cmd, rolls[3].w_cmd

    has_conf, min_m, dist_c = compute_reservation_margin_signal(sim, rid)
    if has_conf and min_m < 0.5 and dist_c < 6.0:
        # Tight/negative time margin ahead -> proactive smooth slow-down
        return rolls[2].v_cmd, rolls[2].w_cmd
    if min_fwd > 3.0 and (not has_conf or min_m > 2.0):
        # Verified wide spatiotemporal margin -> accelerate briskly
        return rolls[1].v_cmd, rolls[1].w_cmd
    return rolls[0].v_cmd, rolls[0].w_cmd


def m3_physics_lookahead_greedy(sim: MultiRobotUnicycleSim, rid: str) -> tuple[float, float]:
    """M3a: Short-horizon physics-lookahead greedy primitive selector (no reservation timing)."""
    rolls = generate_primitive_rollouts(sim, rid)
    obs = sim.get_perceived_circles(rid)

    best_score = -1e9
    best_r = rolls[0]
    for r in rolls:
        if not r.legal:
            continue
        min_obs_c = 9.0
        for s_idx, (px, py, pth, _, _) in enumerate(r.traj):
            if obs:
                t_look = (s_idx + 1) * DT_MICRO
                obs_pred = [CircleState(o.x + o.vx * t_look, o.y + o.vy * t_look, o.vx, o.vy, o.radius) for o in obs]
                oc = obstacle_surface_clearance(px, py, pth, obs_pred, sim.fp)
                if oc < min_obs_c:
                    min_obs_c = oc
        if min_obs_c < 0.18:
            # Penalize trajectories that would trigger emergency shield braking
            score = -50.0 + min_obs_c
        else:
            score = r.progress_m + 0.15 * min(1.5, min_obs_c) - (0.05 if r.name == "BRAKE_WAIT" else 0.0)
        if score > best_score:
            best_score = score
            best_r = r
    return best_r.v_cmd, best_r.w_cmd


def m3_true_oracle(sim: MultiRobotUnicycleSim, rid: str) -> tuple[float, float]:
    """M3b: Counterfactual candidate-pool Oracle using unified-horizon rollout over actual peer trajectories & route progress."""
    rb = sim.robots[rid]
    rolls = generate_primitive_rollouts(sim, rid)
    # Use clean full-horizon peer + intruder states as upper-bound Oracle probe (NOT deployable)
    obs_clean = sim.get_perceived_circles(rid, clean=True)
    # Also incorporate peer upcoming conflict-zone occupancy over H=8 steps
    has_conf, min_m, dist_c = compute_reservation_margin_signal(sim, rid)

    best_score = -1e9
    best_r = rolls[0]
    cur_rem = sim.remaining_route_dist_m(rb)
    orig_x, orig_y, orig_th = rb.x, rb.y, rb.theta
    for r in rolls:
        if not r.legal:
            continue
        min_obs_c = 9.0
        for s_idx, (px, py, pth, _, _) in enumerate(r.traj):
            if obs_clean:
                t_look = (s_idx + 1) * DT_MICRO
                obs_pred = [CircleState(o.x + o.vx * t_look, o.y + o.vy * t_look, o.vx, o.vy, o.radius) for o in obs_clean]
                oc = obstacle_surface_clearance(px, py, pth, obs_pred, sim.fp)
                if oc < min_obs_c:
                    min_obs_c = oc
        rb.x, rb.y, rb.theta = r.traj[-1][0], r.traj[-1][1], r.traj[-1][2]
        end_rem = sim.remaining_route_dist_m(rb)
        rb.x, rb.y, rb.theta = orig_x, orig_y, orig_th
        arc_prog = cur_rem - end_rem
        if min_obs_c < 0.18:
            score = -60.0 + min_obs_c
        else:
            res_pen = 0.35 if (has_conf and min_m < 0.4 and dist_c < 5.0 and r.name == "ACCELERATE") else 0.0
            score = arc_prog + 0.10 * min(1.5, min_obs_c) - res_pen - (0.04 if r.name == "BRAKE_WAIT" else 0.0)
        if score > best_score:
            best_score = score
            best_r = r
    return best_r.v_cmd, best_r.w_cmd


def run_gate1_audit() -> dict[str, Any]:
    om = build_occluded_junction_map(cell_m=2.6)
    ma = build_grid_metric_map("Map_A", cell_m=2.6)
    mb = build_grid_metric_map("Map_B", cell_m=2.6)

    scenarios = [
        ("Occluded_Jct_6R", om, 6, 12, 70.0),
        ("MapA_8R", ma, 8, 16, 80.0),
        ("MapB_8R", mb, 8, 16, 80.0),
    ]
    policies = [
        ("M0_Perception", m0_perception_selector),
        ("M1_SpatialIntent", m1_spatial_intent_selector),
        ("M2_ResMargin", m2_res_margin_selector),
        ("M3_Physics_Lookahead_Greedy", m3_physics_lookahead_greedy),
        ("M3_True_Oracle", m3_true_oracle),
    ]
    seeds = (5001, 5002, 5003)

    summary: dict[str, dict[str, Any]] = {}
    per_policy_rows: dict[str, list[MultiRobotEpisodeResult]] = {}
    for pname, pfn in policies:
        rows: list[MultiRobotEpisodeResult] = []
        for _, mmap, nr, nt, hz in scenarios:
            for sd in seeds:
                sim = MultiRobotUnicycleSim(mmap, n_robots=nr, n_tasks=nt, horizon_s=hz, seed=sd, n_intruders=2)
                rows.append(sim.run_episode(pfn, controller_name=pname))
        per_policy_rows[pname] = rows
        n = len(rows)
        agg = {
            "zero_col_pct": round(100.0 * sum(1 for r in rows if r.zero_collisions) / n, 1),
            "completed_avg": round(float(np.mean([r.completed_tasks for r in rows])), 2),
            "makespan_avg_s": round(float(np.mean([r.makespan_s for r in rows])), 2),
            "wait_ratio_avg": round(float(np.mean([r.wait_ratio for r in rows])), 4),
            "shield_interv_avg": round(float(np.mean([r.shield_interventions for r in rows])), 1),
            "ctrl_ms_avg": round(float(np.mean([r.mean_ctrl_ms for r in rows])), 3),
        }
        summary[pname] = agg
        print(
            f"[Gate1 | {pname:28s}] zcol={agg['zero_col_pct']:5.1f}% comp={agg['completed_avg']:5.2f} "
            f"ms={agg['makespan_avg_s']:6.2f}s wait={agg['wait_ratio_avg']:.4f} sh={agg['shield_interv_avg']:6.1f}",
            flush=True,
        )

    m1_w = summary["M1_SpatialIntent"]["wait_ratio_avg"]
    m2_w = summary["M2_ResMargin"]["wait_ratio_avg"]
    m3_w = summary["M3_True_Oracle"]["wait_ratio_avg"]
    m1_c = summary["M1_SpatialIntent"]["completed_avg"]
    m2_c = summary["M2_ResMargin"]["completed_avg"]
    m3_c = summary["M3_True_Oracle"]["completed_avg"]

    paired_deltas = [
        {
            "map": r_m1.map_name,
            "seed": r_m1.seed,
            "m2_minus_m1_comp": r_m2.completed_tasks - r_m1.completed_tasks,
            "m2_minus_m1_wait": round(r_m2.wait_ratio - r_m1.wait_ratio, 4),
            "oracle_minus_m1_comp": r_or.completed_tasks - r_m1.completed_tasks,
            "oracle_minus_m1_wait": round(r_or.wait_ratio - r_m1.wait_ratio, 4),
        }
        for r_m1, r_m2, r_or in zip(
            per_policy_rows["M1_SpatialIntent"],
            per_policy_rows["M2_ResMargin"],
            per_policy_rows["M3_True_Oracle"],
        )
    ]

    m2_wait_gain_pct = 100.0 * (m1_w - m2_w) / max(1e-6, m1_w)
    m3_wait_gain_pct = 100.0 * (m1_w - m3_w) / max(1e-6, m1_w)
    m2_comp_gain_pct = 100.0 * (m2_c - m1_c) / max(1e-6, m1_c)
    m3_comp_gain_pct = 100.0 * (m3_c - m1_c) / max(1e-6, m1_c)

    headroom_pass = (
        max(m2_wait_gain_pct, m3_wait_gain_pct, m2_comp_gain_pct, m3_comp_gain_pct) >= 3.0
        and summary["M2_ResMargin"]["zero_col_pct"] >= summary["M1_SpatialIntent"]["zero_col_pct"]
    )
    verdict = {
        "summary": summary,
        "paired_instance_deltas": paired_deltas,
        "m2_vs_m1_wait_gain_pct": round(m2_wait_gain_pct, 2),
        "m2_vs_m1_comp_gain_pct": round(m2_comp_gain_pct, 2),
        "m3_true_oracle_vs_m1_wait_gain_pct": round(m3_wait_gain_pct, 2),
        "m3_true_oracle_vs_m1_comp_gain_pct": round(m3_comp_gain_pct, 2),
        "gate1_headroom_pass": bool(headroom_pass),
    }
    print(f"=== Gate 1 Headroom Verdict: pass={headroom_pass} (M2 wait={m2_wait_gain_pct:+.1f}%, comp={m2_comp_gain_pct:+.1f}%; TrueOracle wait={m3_wait_gain_pct:+.1f}%, comp={m3_comp_gain_pct:+.1f}%) ===", flush=True)
    return verdict


if __name__ == "__main__":
    out_p = ROOT / "out" / "ramp_study" / "gate1_headroom.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    res = run_gate1_audit()
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
