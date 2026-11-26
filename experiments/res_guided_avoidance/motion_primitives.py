"""Gate 1: Kinodynamically Feasible Motion Primitive Library (`K = 6`).

Generates `H = 8` micro-step (`2.0 s` at `dt = 0.25 s`) unicycle trajectory
rollouts respecting `MAX_V_MPS`, `MAX_OMEGA_RADPS`, `MAX_DV_DT`, `MAX_DW_DT`
with geometric legality masking (wall clearance & narrow-lane bypass exclusion).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from experiments.res_guided_avoidance.geometry_and_shield import (
    DT_MICRO,
    MAX_DV_DT,
    MAX_DW_DT,
    MAX_OMEGA_RADPS,
    MAX_V_MPS,
    clip_val,
    wrap_pi,
)

if TYPE_CHECKING:
    from experiments.res_guided_avoidance.metric_world import MultiRobotUnicycleSim


PRIMITIVE_NAMES: tuple[str, ...] = (
    "FOLLOW",
    "ACCELERATE",
    "DECELERATE",
    "BRAKE_WAIT",
    "BYPASS_LEFT",
    "BYPASS_RIGHT",
)
N_PRIMITIVES = len(PRIMITIVE_NAMES)


@dataclass(frozen=True)
class PrimitiveRollout:
    prim_id: int
    name: str
    v_cmd: float
    w_cmd: float
    traj: tuple[tuple[float, float, float, float, float], ...]
    legal: bool
    min_wall_clear_m: float
    progress_m: float


def _route_tangent(sim: "MultiRobotUnicycleSim", rid: str) -> tuple[float, float]:
    rb = sim.robots[rid]
    if rb.route_idx + 1 < len(rb.route_nodes):
        u_n = sim.mmap.network.node(rb.route_nodes[rb.route_idx])
        v_n = sim.mmap.network.node(rb.route_nodes[rb.route_idx + 1])
        dx, dy = v_n.x - u_n.x, v_n.y - u_n.y
        ln = max(1e-6, math.hypot(dx, dy))
        return dx / ln, dy / ln
    return math.cos(rb.theta), math.sin(rb.theta)


def generate_primitive_rollouts(
    sim: "MultiRobotUnicycleSim",
    rid: str,
    *,
    horizon_steps: int = 8,
) -> list[PrimitiveRollout]:
    """Generate `K = 6` kinodynamic trajectory primitives and geometric legal masks."""
    rb = sim.robots[rid]
    wx, wy = sim._advance_route_cursor(rb)
    tx, ty = _route_tangent(sim, rid)

    u_id = rb.route_nodes[rb.route_idx] if rb.route_idx < len(rb.route_nodes) else rb.cur_node
    v_id = rb.route_nodes[rb.route_idx + 1] if (rb.route_idx + 1) < len(rb.route_nodes) else None
    hw = sim.mmap.local_half_width(u_id, v_id)

    # Specs: (name, v_nom, lat_offset_m)
    specs = (
        ("FOLLOW", 0.95, 0.0),
        ("ACCELERATE", 1.25, 0.0),
        ("DECELERATE", 0.45, 0.0),
        ("BRAKE_WAIT", 0.00, 0.0),
        ("BYPASS_LEFT", 0.72, +0.65),
        ("BYPASS_RIGHT", 0.72, -0.65),
    )

    dv_max = MAX_DV_DT * DT_MICRO
    dw_max = MAX_DW_DT * DT_MICRO
    rollouts: list[PrimitiveRollout] = []

    for pid, (pname, v_nom, lat_off) in enumerate(specs):
        # Narrow lanes (< 2.65 m wide -> hw < 1.325 m) physically disallow lateral bypass
        geom_allowed = True
        if abs(lat_off) > 1e-3 and hw < 1.32:
            geom_allowed = False

        tgt_x = wx - ty * lat_off
        tgt_y = wy + tx * lat_off
        ang0 = math.atan2(tgt_y - rb.y, tgt_x - rb.x)
        err0 = wrap_pi(ang0 - rb.theta)
        v_cmd0 = v_nom if abs(err0) < 0.75 else min(v_nom, 0.35)
        w_cmd0 = clip_val(2.2 * err0, -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS) if v_nom > 1e-3 else 0.0

        px, py, pth, pv, pw = rb.x, rb.y, rb.theta, rb.v, rb.omega
        traj: list[tuple[float, float, float, float, float]] = []
        min_wc = 9.0
        for s in range(horizon_steps):
            ang_s = math.atan2(tgt_y - py, tgt_x - px)
            err_s = wrap_pi(ang_s - pth)
            vt = v_nom if abs(err_s) < 0.75 else min(v_nom, 0.35)
            wt = clip_val(2.2 * err_s, -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS) if v_nom > 1e-3 else 0.0
            pv = clip_val(pv + clip_val(vt - pv, -dv_max, dv_max), 0.0, MAX_V_MPS)
            pw = clip_val(pw + clip_val(wt - pw, -dw_max, dw_max), -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS)
            px += pv * math.cos(pth) * DT_MICRO
            py += pv * math.sin(pth) * DT_MICRO
            pth = wrap_pi(pth + pw * DT_MICRO)
            traj.append((px, py, pth, pv, pw))
            wc = sim.mmap.wall_clearance(px, py, pth, sim.fp)
            if wc < min_wc:
                min_wc = wc

        legal = geom_allowed and (min_wc >= 0.03)
        # Progress along route tangent
        prog_m = (px - rb.x) * tx + (py - rb.y) * ty
        rollouts.append(
            PrimitiveRollout(
                prim_id=pid,
                name=pname,
                v_cmd=float(v_cmd0),
                w_cmd=float(w_cmd0),
                traj=tuple(traj),
                legal=bool(legal),
                min_wall_clear_m=float(min_wc),
                progress_m=float(prog_m),
            )
        )

    # Ensure at least BRAKE_WAIT (id=3) or FOLLOW (id=0) is always legal fallback
    if not any(r.legal for r in rollouts):
        r_bw = rollouts[3]
        rollouts[3] = PrimitiveRollout(
            r_bw.prim_id, r_bw.name, r_bw.v_cmd, r_bw.w_cmd,
            r_bw.traj, True, r_bw.min_wall_clear_m, r_bw.progress_m,
        )
    return rollouts
