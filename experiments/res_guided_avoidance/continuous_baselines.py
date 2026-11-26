"""P3 Strong Continuous Multi-Robot Baselines (`ORCA_Uni`, `DWA`, `Res_DWA`, `Uni_MPC`, `Res_Yield`).

All baselines operate inside `MultiRobotUnicycleSim` under identical unicycle acceleration
constraints (`MAX_V_MPS`, `MAX_OMEGA_RADPS`, `MAX_DV_DT`, `MAX_DW_DT`) and perception.
"""

from __future__ import annotations

import math
from typing import Sequence

from experiments.res_guided_avoidance.geometry_and_shield import (
    CircleState,
    DT_MICRO,
    MAX_DV_DT,
    MAX_DW_DT,
    MAX_OMEGA_RADPS,
    MAX_V_MPS,
    clip_val,
    obstacle_surface_clearance,
    wrap_pi,
)
from experiments.res_guided_avoidance.metric_world import MultiRobotUnicycleSim
from experiments.res_guided_avoidance.reservation_features import extract_multi_robot_obs


def _preferred_velocity(sim: MultiRobotUnicycleSim, rid: str, speed: float = 1.05) -> tuple[float, float, float, float]:
    rb = sim.robots[rid]
    wx, wy = sim._advance_route_cursor(rb)
    dx, dy = wx - rb.x, wy - rb.y
    dist = math.hypot(dx, dy)
    if dist <= 1e-6:
        return 0.0, 0.0, wx, wy
    v_pref = min(speed, max(0.25, dist / 1.0))
    return v_pref * dx / dist, v_pref * dy / dist, wx, wy


def orca_uni_controller(sim: MultiRobotUnicycleSim, rid: str) -> tuple[float, float]:
    """Geometric ORCA half-plane solver in R^2 projected onto unicycle `(v, omega)`."""
    rb = sim.robots[rid]
    vx_pref, vy_pref, _, _ = _preferred_velocity(sim, rid, speed=1.05)
    vx_cur = rb.v * math.cos(rb.theta)
    vy_cur = rb.v * math.sin(rb.theta)
    tau = 2.2

    # Build ORCA linear half-planes: n . v >= b
    planes: list[tuple[float, float, float]] = []
    obs_vis = sim.get_perceived_circles(rid)
    for o in obs_vis:
        rel_p_x = o.x - rb.x
        rel_p_y = o.y - rb.y
        rel_v_x = vx_cur - o.vx
        rel_v_y = vy_cur - o.vy
        dist_sq = rel_p_x * rel_p_x + rel_p_y * rel_p_y
        comb_r = sim.fp.max_extent_m + o.radius + 0.10
        if dist_sq > comb_r * comb_r:
            w_x = rel_v_x - rel_p_x / tau
            w_y = rel_v_y - rel_p_y / tau
            w_len = math.hypot(w_x, w_y)
            if w_len > 1e-6:
                unit_w_x, unit_w_y = w_x / w_len, w_y / w_len
                dot = w_x * rel_p_x + w_y * rel_p_y
                if dot < 0.0 and dot * dot > comb_r * comb_r * w_len * w_len / (tau * tau):
                    u_x = (comb_r / tau - w_len) * unit_w_x
                    u_y = (comb_r / tau - w_len) * unit_w_y
                    nx, ny = unit_w_x, unit_w_y
                else:
                    # Project onto leg of VO
                    leg = math.sqrt(max(1e-6, dist_sq - comb_r * comb_r))
                    if rel_p_x * w_y - rel_p_y * w_x > 0.0:
                        dir_x = (rel_p_x * leg - rel_p_y * comb_r) / dist_sq
                        dir_y = (rel_p_x * comb_r + rel_p_y * leg) / dist_sq
                    else:
                        dir_x = -(rel_p_x * leg + rel_p_y * comb_r) / dist_sq
                        dir_y = -(-rel_p_x * comb_r + rel_p_y * leg) / dist_sq
                    nx, ny = -dir_y, dir_x
                    proj = rel_v_x * nx + rel_v_y * ny
                    u_x, u_y = max(0.0, -proj) * nx, max(0.0, -proj) * ny
                px = vx_cur + 0.5 * u_x
                py = vy_cur + 0.5 * u_y
                planes.append((nx, ny, px * nx + py * ny))
        else:
            # Already closer than comb_r: push radially out
            dist = max(1e-4, math.sqrt(dist_sq))
            nx, ny = -rel_p_x / dist, -rel_p_y / dist
            push = (comb_r - dist) / DT_MICRO
            planes.append((nx, ny, push))

    # Project v_pref iteratively onto ORCA half-planes
    vx_opt, vy_opt = vx_pref, vy_pref
    for nx, ny, b in planes:
        if vx_opt * nx + vy_opt * ny < b:
            diff = b - (vx_opt * nx + vy_opt * ny)
            vx_opt += diff * nx
            vy_opt += diff * ny

    spd = math.hypot(vx_opt, vy_opt)
    if spd > MAX_V_MPS:
        vx_opt *= MAX_V_MPS / spd
        vy_opt *= MAX_V_MPS / spd
        spd = MAX_V_MPS

    if spd < 0.08:
        return 0.0, 0.0
    tgt_th = math.atan2(vy_opt, vx_opt)
    err = wrap_pi(tgt_th - rb.theta)
    v_cmd = spd * max(0.0, math.cos(err))
    w_cmd = clip_val(2.2 * err, -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS)
    return v_cmd, w_cmd


def _dwa_rollout_score(
    sim: MultiRobotUnicycleSim,
    rid: str,
    vt: float,
    wt: float,
    wx: float,
    wy: float,
    obs_vis: Sequence[CircleState],
    *,
    res_penalty: float = 0.0,
) -> float:
    rb = sim.robots[rid]
    px, py, pth, pv, pw = rb.x, rb.y, rb.theta, rb.v, rb.omega
    dv_max = MAX_DV_DT * DT_MICRO
    dw_max = MAX_DW_DT * DT_MICRO
    min_c = 9.0
    for s in range(1, 7):
        pv = clip_val(pv + clip_val(vt - pv, -dv_max, dv_max), 0.0, MAX_V_MPS)
        pw = clip_val(pw + clip_val(wt - pw, -dw_max, dw_max), -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS)
        px += pv * math.cos(pth) * DT_MICRO
        py += pv * math.sin(pth) * DT_MICRO
        pth = wrap_pi(pth + pw * DT_MICRO)
        wc = sim.mmap.wall_clearance(px, py, pth, sim.fp)
        min_c = min(min_c, wc)
        if obs_vis:
            t_lk = s * DT_MICRO
            obs_s = [CircleState(o.x + o.vx * t_lk, o.y + o.vy * t_lk, o.vx, o.vy, o.radius) for o in obs_vis]
            min_c = min(min_c, obstacle_surface_clearance(px, py, pth, obs_s, sim.fp))
        if min_c <= 0.0:
            return -1e6
    d_goal = math.hypot(wx - px, wy - py)
    hdg = abs(wrap_pi(math.atan2(wy - py, wx - px) - pth))
    return -1.4 * d_goal - 0.35 * hdg + 0.55 * min(1.0, min_c) + 0.35 * (vt / MAX_V_MPS) - res_penalty


def dwa_controller(sim: MultiRobotUnicycleSim, rid: str, *, use_res: bool = False) -> tuple[float, float]:
    """Dynamic Window Approach (`7 x 7` reachable velocity grid), optionally reservation-guided."""
    rb = sim.robots[rid]
    _, _, wx, wy = _preferred_velocity(sim, rid)
    obs_vis = sim.get_perceived_circles(rid)
    dv_max = MAX_DV_DT * DT_MICRO
    dw_max = MAX_DW_DT * DT_MICRO

    v_lo = max(0.0, rb.v - dv_max)
    v_hi = min(MAX_V_MPS, rb.v + dv_max)
    w_lo = max(-MAX_OMEGA_RADPS, rb.omega - dw_max)
    w_hi = min(MAX_OMEGA_RADPS, rb.omega + dw_max)

    res_gate = 0.0
    if use_res:
        obs_r = extract_multi_robot_obs(sim, rid, mode="full_res")
        # Slot 0 conflict penalty proportional to speed if another higher-prio robot reserves upcoming node simultaneously
        if obs_r[24] < 0.60 and abs(obs_r[25]) < 0.30 and obs_r[27] >= 0.0:
            res_gate = 1.6

    best_s = -1e9
    best_pair = (0.0, 0.0)
    for vi in range(5):
        vt = v_lo + (v_hi - v_lo) * (vi / 4.0)
        for wi in range(7):
            wt = w_lo + (w_hi - w_lo) * (wi / 6.0)
            sc = _dwa_rollout_score(sim, rid, vt, wt, wx, wy, obs_vis, res_penalty=res_gate * (vt / MAX_V_MPS))
            if sc > best_s:
                best_s = sc
                best_pair = (vt, wt)
    return best_pair


def res_dwa_controller(sim: MultiRobotUnicycleSim, rid: str) -> tuple[float, float]:
    return dwa_controller(sim, rid, use_res=True)


def uni_mpc_controller(sim: MultiRobotUnicycleSim, rid: str) -> tuple[float, float]:
    """Receding-horizon unicycle sampling MPC (`N = 8`, log-barrier clearance + jerk penalty)."""
    rb = sim.robots[rid]
    _, _, wx, wy = _preferred_velocity(sim, rid)
    obs_vis = sim.get_perceived_circles(rid)
    err_nom = wrap_pi(math.atan2(wy - rb.y, wx - rb.x) - rb.theta)
    w_nom = clip_val(2.0 * err_nom, -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS)

    v_cands = (0.0, 0.35, 0.70, 1.05, 1.35)
    w_cands = (w_nom, 0.0, clip_val(w_nom + 0.65, -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS), clip_val(w_nom - 0.65, -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS), MAX_OMEGA_RADPS, -MAX_OMEGA_RADPS)

    best_cost = 1e18
    best_pair = (0.0, 0.0)
    dv_max = MAX_DV_DT * DT_MICRO
    dw_max = MAX_DW_DT * DT_MICRO
    for vt in v_cands:
        for wt in w_cands:
            px, py, pth, pv, pw = rb.x, rb.y, rb.theta, rb.v, rb.omega
            cost = 0.25 * ((vt - rb.v) ** 2 + 0.3 * (wt - rb.omega) ** 2)
            for s in range(1, 9):
                pv = clip_val(pv + clip_val(vt - pv, -dv_max, dv_max), 0.0, MAX_V_MPS)
                pw = clip_val(pw + clip_val(wt - pw, -dw_max, dw_max), -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS)
                px += pv * math.cos(pth) * DT_MICRO
                py += pv * math.sin(pth) * DT_MICRO
                pth = wrap_pi(pth + pw * DT_MICRO)
                wc = sim.mmap.wall_clearance(px, py, pth, sim.fp)
                mc = wc
                if obs_vis:
                    t_lk = s * DT_MICRO
                    obs_s = [CircleState(o.x + o.vx * t_lk, o.y + o.vy * t_lk, o.vx, o.vy, o.radius) for o in obs_vis]
                    mc = min(mc, obstacle_surface_clearance(px, py, pth, obs_s, sim.fp))
                if mc <= 0.02:
                    cost += 500.0
                elif mc < 0.45:
                    cost += 2.5 * (0.45 - mc) ** 2
                cost += 0.30 * math.hypot(wx - px, wy - py)
            if cost < best_cost:
                best_cost = cost
                best_pair = (vt, wt)
    return best_pair


def res_yield_controller(sim: MultiRobotUnicycleSim, rid: str) -> tuple[float, float]:
    """Pure discrete-reservation gating follower (stops smoothly if upcoming reservation is occupied)."""
    rb = sim.robots[rid]
    _, _, wx, wy = _preferred_velocity(sim, rid)
    err = wrap_pi(math.atan2(wy - rb.y, wx - rb.x) - rb.theta)
    v_cmd = 1.00 if abs(err) < 0.65 else 0.28
    w_cmd = clip_val(2.0 * err, -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS)

    obs_r = extract_multi_robot_obs(sim, rid, mode="full_res")
    if obs_r[24] < 0.55 and abs(obs_r[25]) < 0.35 and obs_r[27] >= 0.0:
        v_cmd = 0.0
    return v_cmd, w_cmd
