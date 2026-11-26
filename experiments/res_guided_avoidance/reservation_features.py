"""P2 Modular Observation Extractor: Kinematic/Local Perception (`24-D`) + Spatiotemporal Reservation Tensor (`26-D`).

Supports controlled masking for rigorous ablation:
  - `mode="kin_only"`: zeros out `obs_res` (`26-D`).
  - `mode="intent_notime"`: keeps spatial conflict point & approach tangent `(s_int, cos_psi, sin_psi)`
    while zeroing out temporal arrival offset, overlap duration, priority, and field density.
  - `mode="full_res"`: full `50-D` (`obs_kin` + `obs_res`) spatiotemporal reservation context.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np

from experiments.res_guided_avoidance.geometry_and_shield import (
    MAX_OMEGA_RADPS,
    MAX_V_MPS,
    clip_val,
    obstacle_surface_clearance,
    wrap_pi,
)
from experiments.res_guided_avoidance.metric_world import MultiRobotUnicycleSim

KIN_DIM: int = 24
K_RES_SLOTS: int = 4
RES_SLOT_DIM: int = 6
RES_DIM: int = K_RES_SLOTS * RES_SLOT_DIM + 2  # 26-D
OBS_TOTAL_DIM: int = KIN_DIM + RES_DIM  # 50-D

ObsMode = Literal["kin_only", "intent_notime", "full_res"]


def extract_multi_robot_obs(
    sim: MultiRobotUnicycleSim,
    rid: str,
    *,
    mode: ObsMode = "full_res",
    t_hor_s: float = 8.0,
) -> np.ndarray:
    """Extract fixed-size `50-D` observation `(obs_kin[24], obs_res[26])` for robot `rid`."""
    rb = sim.robots[rid]
    feats = np.zeros(OBS_TOTAL_DIM, dtype=np.float32)

    # Active segment & lookahead waypoints
    wx, wy = sim._advance_route_cursor(rb)
    u_nd = sim.mmap.network.node(rb.route_nodes[rb.route_idx]) if rb.route_nodes else sim.mmap.network.node(rb.cur_node)
    if rb.route_idx + 1 < len(rb.route_nodes):
        v_nd = sim.mmap.network.node(rb.route_nodes[rb.route_idx + 1])
        hw = sim.mmap.local_half_width(u_nd.node_id, v_nd.node_id)
        seg_dx, seg_dy = v_nd.x - u_nd.x, v_nd.y - u_nd.y
        seg_len = max(1e-6, math.hypot(seg_dx, seg_dy))
        # Signed lateral offset from current centerline segment
        y_lat = ((rb.x - u_nd.x) * (-seg_dy) + (rb.y - u_nd.y) * seg_dx) / seg_len
    else:
        hw = sim.mmap.local_half_width(u_nd.node_id, None)
        y_lat = 0.0

    # Next-2 waypoint heading error
    if rb.route_idx + 2 < len(rb.route_nodes):
        w2_nd = sim.mmap.network.node(rb.route_nodes[rb.route_idx + 2])
        wx2, wy2 = float(w2_nd.x), float(w2_nd.y)
    else:
        wx2, wy2 = wx, wy
    g_nd = sim.mmap.network.node(rb.route_nodes[-1]) if rb.route_nodes else u_nd
    d_wp = math.hypot(wx - rb.x, wy - rb.y)
    d_goal = math.hypot(g_nd.x - rb.x, g_nd.y - rb.y)
    err1 = wrap_pi(math.atan2(wy - rb.y, wx - rb.x) - rb.theta)
    err2 = wrap_pi(math.atan2(wy2 - rb.y, wx2 - rb.x) - rb.theta)

    feats[0] = clip_val(y_lat / max(0.8, hw), -1.5, 1.5)
    feats[1] = math.sin(err1)
    feats[2] = math.cos(err1)
    feats[3] = math.sin(err2)
    feats[4] = clip_val(rb.v / MAX_V_MPS, 0.0, 1.2)
    feats[5] = clip_val(rb.omega / MAX_OMEGA_RADPS, -1.2, 1.2)
    feats[6] = clip_val(hw / 1.5, 0.5, 1.5)
    feats[7] = clip_val(d_wp / 6.0, 0.0, 1.5)
    feats[8] = clip_val(d_goal / 20.0, 0.0, 1.5)

    # Perceived obstacles / peers (subject to LOS occlusion and sensor noise)
    obs_vis = sim.get_perceived_circles(rid)
    ct, st = math.cos(-rb.theta), math.sin(-rb.theta)
    obs_ranked: list[tuple[float, float, float, float, float, float]] = []
    for o in obs_vis:
        d_surf = obstacle_surface_clearance(rb.x, rb.y, rb.theta, (o,), sim.fp)
        dx_w, dy_w = o.x - rb.x, o.y - rb.y
        lx = ct * dx_w - st * dy_w
        ly = st * dx_w + ct * dy_w
        lvx = ct * o.vx - st * o.vy
        lvy = st * o.vx + ct * o.vy
        fwd_prio = 0.0 if lx >= -0.35 else 1.0
        obs_ranked.append((fwd_prio * 100.0 + d_surf, lx, ly, lvx, lvy, d_surf))
    obs_ranked.sort(key=lambda t: t[0])

    if obs_ranked:
        _, lx, ly, lvx, lvy, ds = obs_ranked[0]
        feats[9] = clip_val(lx / 8.0, -1.5, 1.5)
        feats[10] = clip_val(ly / max(0.8, hw), -2.0, 2.0)
        feats[11] = clip_val(lvx / MAX_V_MPS, -1.5, 1.5)
        feats[12] = clip_val(lvy / MAX_V_MPS, -1.5, 1.5)
        feats[13] = clip_val(max(-0.5, ds) / 4.0, -0.5, 1.5)
    else:
        feats[9], feats[10], feats[11], feats[12], feats[13] = 1.5, 0.0, 0.0, 0.0, 1.5

    if len(obs_ranked) >= 2:
        _, lx2, ly2, _, _, ds2 = obs_ranked[1]
        feats[14] = clip_val(lx2 / 8.0, -1.5, 1.5)
        feats[15] = clip_val(ly2 / max(0.8, hw), -2.0, 2.0)
        feats[16] = clip_val(max(-0.5, ds2) / 4.0, -0.5, 1.5)
    else:
        feats[14], feats[15], feats[16] = 1.5, 0.0, 1.5

    # 7 forward-fan range probes (-60..+60 deg) up to 6.0 m
    fan_angles = (-1.05, -0.70, -0.35, 0.0, 0.35, 0.70, 1.05)
    for fi, ang_off in enumerate(fan_angles):
        ang = rb.theta + ang_off
        ux, uy = math.cos(ang), math.sin(ang)
        d_hit = 6.0
        # Coarse raymarch for wall clearance (fast 6-step check)
        for step_m in (0.6, 1.2, 2.0, 3.0, 4.5, 6.0):
            if step_m >= d_hit:
                break
            wc = sim.mmap.wall_clearance(rb.x + ux * step_m, rb.y + uy * step_m, ang, sim.fp)
            if wc <= 0.0:
                d_hit = min(d_hit, max(0.0, step_m - 0.3))
                break
        # Analytical circle intersection for perceived peers/intruders
        for o in obs_vis:
            proj = (o.x - rb.x) * ux + (o.y - rb.y) * uy
            if proj <= -0.2 or proj > 6.5:
                continue
            perp = abs((o.x - rb.x) * (-uy) + (o.y - rb.y) * ux)
            rad = sim.fp.radius_m + o.radius + 0.05
            if perp < rad:
                hs = max(0.0, proj - math.sqrt(max(0.0, rad * rad - perp * perp)))
                d_hit = min(d_hit, hs)
        feats[17 + fi] = clip_val(d_hit / 6.0, 0.0, 1.0)

    if mode == "kin_only":
        return feats

    # ------------------------------------------------------------------
    # Branch 2: Spatiotemporal Reservation Context (`obs_res`, 26-D)
    # ------------------------------------------------------------------
    res_view = sim.get_reservation_view()
    ego_prio = 2 if rb.phase == "to_dropoff" else (1 if rb.phase == "to_pickup" else 0)
    hor_ticks = max(1.0, t_hor_s / max(0.2, sim.mmap.macro_step_s))

    # Build ego upcoming nodes & estimated arrival times over next <= 5 hops
    ego_future: list[tuple[str, float, float, float, float]] = []
    # Tuple: (node_id, cum_dist_m, est_t_s, dir_x, dir_y)
    cum_d = 0.0
    prev_x, prev_y = rb.x, rb.y
    up_nodes = rb.route_nodes[rb.route_idx :] if rb.route_nodes else [rb.cur_node]
    for idx_u, nid_u in enumerate(up_nodes[:6]):
        nd_u = sim.mmap.network.node(nid_u)
        seg_d = math.hypot(nd_u.x - prev_x, nd_u.y - prev_y)
        cum_d += seg_d
        est_t = sim.t_s + cum_d / max(0.4, MAX_V_MPS * 0.75)
        if idx_u + 1 < len(up_nodes):
            nd_v = sim.mmap.network.node(up_nodes[idx_u + 1])
            dx_u, dy_u = nd_v.x - nd_u.x, nd_v.y - nd_u.y
        else:
            dx_u, dy_u = nd_u.x - prev_x, nd_u.y - prev_y
        ln_u = max(1e-6, math.hypot(dx_u, dy_u))
        ego_future.append((nid_u, cum_d, est_t, dx_u / ln_u, dy_u / ln_u))
        prev_x, prev_y = nd_u.x, nd_u.y

    ego_node_map = {nid: (cd, et, ux, uy) for nid, cd, et, ux, uy in ego_future}
    l_hor_m = max(4.0, min(14.0, cum_d if cum_d > 1.0 else 8.0))

    # Scan peer future reservations intersecting ego upcoming nodes.
    # Strict time isolation: `spatial_intent` / `intent_notime` MUST NOT filter or sort by time!
    use_time = (mode == "full_res")
    conflicts: list[tuple[float, float, float, float, float, float, float]] = []
    for other_id, segs in res_view.items():
        if other_id == rid or not segs:
            continue
        for k_idx, (nd_o, t_en, t_ex, o_prio) in enumerate(segs):
            if nd_o not in ego_node_map:
                continue
            t_en_s = t_en * sim.mmap.macro_step_s
            t_ex_s = t_ex * sim.mmap.macro_step_s
            if use_time and (t_ex_s < sim.t_s - 1.0 or t_en_s > sim.t_s + t_hor_s):
                continue
            cd, est_t, ego_ux, ego_uy = ego_node_map[nd_o]
            # Approach tangent of peer robot at this node
            if k_idx + 1 < len(segs):
                nd_next = sim.mmap.network.node(segs[k_idx + 1][0])
                nd_cur = sim.mmap.network.node(nd_o)
                odx, ody = nd_next.x - nd_cur.x, nd_next.y - nd_cur.y
            elif k_idx > 0:
                nd_prev = sim.mmap.network.node(segs[k_idx - 1][0])
                nd_cur = sim.mmap.network.node(nd_o)
                odx, ody = nd_cur.x - nd_prev.x, nd_cur.y - nd_prev.y
            else:
                odx, ody = 1.0, 0.0
            oln = max(1e-6, math.hypot(odx, ody))
            oux, ouy = odx / oln, ody / oln
            # Relative angle in ego path frame: cos_psi = -1 -> head-on, 0 -> cross, +1 -> same dir
            cos_psi = clip_val(ego_ux * oux + ego_uy * ouy, -1.0, 1.0)
            sin_psi = clip_val(ego_ux * ouy - ego_uy * oux, -1.0, 1.0)

            s_norm = clip_val(cd / l_hor_m, 0.0, 1.0)
            dt_arr = clip_val((t_en_s - est_t) / t_hor_s, -1.0, 1.0) if use_time else 0.0
            ovl = clip_val((min(t_ex_s, est_t + 2.0) - max(t_en_s, est_t - 2.0)) / t_hor_s, 0.0, 1.0) if use_time else 0.0
            prio_diff = clip_val((o_prio - ego_prio) / 2.0, -1.0, 1.0) if use_time else 0.0
            score_sort = (s_norm + 0.5 * abs(dt_arr)) if use_time else s_norm
            conflicts.append((score_sort, s_norm, dt_arr, ovl, cos_psi, sin_psi, prio_diff))

    conflicts.sort(key=lambda t: t[0])
    base = KIN_DIM
    for slot_i in range(K_RES_SLOTS):
        off = base + slot_i * RES_SLOT_DIM
        if slot_i < len(conflicts):
            _, s_norm, dt_arr, ovl, cos_psi, sin_psi, prio_diff = conflicts[slot_i]
            feats[off + 0] = s_norm
            feats[off + 4] = cos_psi
            feats[off + 5] = sin_psi
            if mode == "full_res":
                feats[off + 1] = dt_arr
                feats[off + 2] = ovl
                feats[off + 3] = prio_diff
        else:
            feats[off + 0] = 1.0  # far / no conflict

    # 2 global congestion field lookaheads at next-1 and next-2 hops
    if mode == "full_res":
        hop1 = up_nodes[1] if len(up_nodes) > 1 else up_nodes[0]
        hop2 = up_nodes[2] if len(up_nodes) > 2 else hop1
        feats[base + K_RES_SLOTS * RES_SLOT_DIM + 0] = clip_val(
            sim.field.phi(hop1, sim.macro_tick + 1) / 3.0, 0.0, 1.0
        )
        feats[base + K_RES_SLOTS * RES_SLOT_DIM + 1] = clip_val(
            sim.field.phi(hop2, sim.macro_tick + 3) / 3.0, 0.0, 1.0
        )
    return feats


def extract_centralized_critic_state(sim: MultiRobotUnicycleSim, rid: str, obs_vec: np.ndarray) -> np.ndarray:
    """CTDE critic state: decentralized `50-D` obs + `6-D` global fleet congestion/progress summary."""
    extra = np.zeros(6, dtype=np.float32)
    active = [r for r in sim.robot_ids if sim.robots[r].phase != "idle"]
    extra[0] = len(active) / max(1.0, float(sim.n_robots))
    extra[1] = sim.completed_tasks / max(1.0, float(sim.n_tasks))
    extra[2] = clip_val(sim.t_s / max(1.0, sim.horizon_s), 0.0, 1.0)
    ego = sim.robots[rid]
    near_cnt = sum(
        1 for o_id in sim.robot_ids
        if o_id != rid and math.hypot(sim.robots[o_id].x - ego.x, sim.robots[o_id].y - ego.y) < 4.5
    )
    extra[3] = clip_val(near_cnt / 4.0, 0.0, 1.0)
    extra[4] = clip_val(sum(sim.robots[r].v for r in sim.robot_ids) / max(1.0, sim.n_robots * MAX_V_MPS), 0.0, 1.0)
    extra[5] = clip_val(ego.eta_drift_ticks / 10.0, 0.0, 1.0)
    return np.concatenate([obs_vec, extra], axis=0)
