"""Task 2: Primitive Decision Freedom & Shield-Override Audit.

Quantifies on identical snapshots across 5 regimes (`Nominal`, `Junction`, `HeadOn`,
`Intruder`, `CongestionWait`):
  - K_geom(s): geometrically legal primitives before safety filtering
  - K_safe(s) = |A_safe(s)|: primitives surviving BrakingInvariantShield + robust_safety
  - Pre-shield & Post-shield action disagreement between R1_Robust, R5_Orig, and R5_Robust_Retrained
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.res_guided_avoidance.gate1_headroom_audit import (
    compute_reservation_margin_signal,
    m2_res_margin_selector,
)
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
from experiments.res_guided_avoidance.metric_world import (
    MultiRobotUnicycleSim,
    build_grid_metric_map,
    build_occluded_junction_map,
    build_workshop_metric_map,
)
from experiments.res_guided_avoidance.motion_primitives import generate_primitive_rollouts
from experiments.res_guided_avoidance.ramp_ppo import RAMP_CKPT_DIR, NumpyRampPolicy, extract_ramp_obs


def _classify_regime(sim: MultiRobotUnicycleSim, rid: str) -> str:
    rb = sim.robots[rid]
    obs = sim.get_perceived_circles(rid)
    has_c, min_m, dist_c = compute_reservation_margin_signal(sim, rid)
    for intr in sim.intruders:
        if math.hypot(intr.x - rb.x, intr.y - rb.y) < 4.0:
            return "Dynamic_Intruder"
    for oid in sim.robot_ids:
        if oid == rid:
            continue
        o = sim.robots[oid]
        if math.hypot(o.x - rb.x, o.y - rb.y) < 4.0 and math.cos(rb.theta - o.theta) < -0.45:
            return "HeadOn"
    if has_c and dist_c < 5.5 and min_m < 1.2:
        return "Junction_Conflict"
    if rb.v < 0.25 or (obs and min(math.hypot(o.x - rb.x, o.y - rb.y) for o in obs) < 2.8):
        return "Congestion_Wait"
    return "Nominal_Cruise"


def _is_prim_shield_safe(sim: MultiRobotUnicycleSim, rid: str, v_cmd: float, w_cmd: float, obs_vis: list[CircleState]) -> bool:
    rb = sim.robots[rid]
    wx, wy = sim._advance_route_cursor(rb)
    sh = sim.shield.filter(
        x=rb.x, y=rb.y, theta=rb.theta, v=rb.v, omega=rb.omega,
        v_cmd=v_cmd, omega_cmd=w_cmd, obstacles=obs_vis,
        wall_fn=lambda px, py, pth: sim.mmap.wall_clearance(px, py, pth, sim.fp),
        center_w=clip_val(1.6 * wrap_pi(math.atan2(wy - rb.y, wx - rb.x) - rb.theta), -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS),
        ref_heading_err_fn=lambda px, py, pth, _wx=wx, _wy=wy: wrap_pi(math.atan2(_wy - py, _wx - px) - pth),
        obs_pos_std_m=(sim.obs_noise_pos_m if sim.robust_safety else 0.0),
        obs_vel_std_mps=(sim.obs_noise_vel_mps if sim.robust_safety else 0.0),
    )
    if sh.intervened:
        return False
    if sim.robust_safety and v_cmd > 0.05 and obs_vis:
        dv_r = MAX_DV_DT * DT_MICRO
        dw_r = MAX_DW_DT * DT_MICRO
        rx, ry, rth, rv, rw = rb.x, rb.y, rb.theta, rb.v, rb.omega
        fwd_obs = [o for o in obs_vis if ((o.x - rb.x) * math.cos(rb.theta) + (o.y - rb.y) * math.sin(rb.theta)) > -0.25 and math.hypot(o.x - rb.x, o.y - rb.y) < 3.2]
        if fwd_obs:
            for s_j in (1, 2):
                rv = clip_val(rv + clip_val(v_cmd - rv, -dv_r, dv_r), 0.0, MAX_V_MPS)
                rw = clip_val(rw + clip_val(w_cmd - rw, -dw_r, dw_r), -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS)
                rx += rv * math.cos(rth) * DT_MICRO
                ry += rv * math.sin(rth) * DT_MICRO
                rth = wrap_pi(rth + rw * DT_MICRO)
                t_j = s_j * DT_MICRO
                eps_j = 1.96 * math.hypot(sim.obs_noise_pos_m, min(t_j, 1.0) * sim.obs_noise_vel_mps) if (sim.obs_noise_pos_m > 0 or sim.obs_noise_vel_mps > 0) else 0.0
                obs_j = [CircleState(o.x + o.vx * t_j, o.y + o.vy * t_j, o.vx, o.vy, o.radius) for o in fwd_obs]
                if obstacle_surface_clearance(rx, ry, rth, obs_j, sim.fp) < (0.18 + eps_j):
                    return False
    return True


def _pick_id_from_v(rolls, v_cmd: float) -> int:
    return min(range(len(rolls)), key=lambda i: (0 if rolls[i].legal else 1, abs(rolls[i].v_cmd - v_cmd)))


def main() -> None:
    out_dir = ROOT / "out" / "decision_headroom"
    out_dir.mkdir(parents=True, exist_ok=True)

    om = build_occluded_junction_map(cell_m=2.6)
    ma = build_grid_metric_map("Map_A", cell_m=2.6)
    wm = build_workshop_metric_map()

    r5_orig = NumpyRampPolicy.load_npz(RAMP_CKPT_DIR / "r5_ramp_ppo_s42.npz")
    r5_rob = NumpyRampPolicy.load_npz(RAMP_CKPT_DIR / "r5_ramp_ppo_robust_s42.npz")

    scenarios = [
        ("Occluded_Jct_6R_Clean", om, 6, 12, 70.0, 2, 0.0, 0.0),
        ("MapA_8R_Clean", ma, 8, 16, 80.0, 2, 0.0, 0.0),
        ("Workshop_8R_Clean", wm, 8, 12, 120.0, 1, 0.0, 0.0),
        ("Occluded_Jct_6R_Noise0.12", om, 6, 12, 70.0, 2, 0.12, 0.20),
    ]

    k_geom_hist: Counter[int] = Counter()
    k_safe_hist: Counter[int] = Counter()
    regime_stats: dict[str, dict[str, Any]] = {}
    tot_samples = 0
    disagree_r1_vs_r5orig = 0
    disagree_r1_vs_r5rob = 0
    r5orig_unsafe_overridden = 0
    r5rob_unsafe_overridden = 0
    r1_unsafe_overridden = 0

    for sc_name, mmap, nr, nt, hz, ni, np_m, nv_m in scenarios:
        sim = MultiRobotUnicycleSim(mmap, n_robots=nr, n_tasks=nt, horizon_s=hz, seed=6001, n_intruders=ni, obs_noise_pos_m=np_m, obs_noise_vel_mps=nv_m, robust_safety=True)
        for step_i in range(int(hz / DT_MICRO)):
            if step_i % 4 == 0:
                for rid in sim.robot_ids:
                    rb = sim.robots[rid]
                    if rb.phase == "idle" or rb.service_rem_s > 0.0:
                        continue
                    rolls = generate_primitive_rollouts(sim, rid)
                    obs_vis = sim.get_perceived_circles(rid)
                    reg = _classify_regime(sim, rid)

                    k_g = sum(1 for r in rolls if r.legal)
                    safe_ids = [r.prim_id for r in rolls if r.legal and _is_prim_shield_safe(sim, rid, r.v_cmd, r.w_cmd, obs_vis)]
                    # Always count BRAKE_WAIT as fallback safe action
                    k_s = max(1, len(safe_ids))

                    k_geom_hist[k_g] += 1
                    k_safe_hist[k_s] += 1
                    tot_samples += 1

                    # Compare R1_Robust vs R5_Orig vs R5_Robust choices on exact same snapshot
                    v1, _ = m2_res_margin_selector(sim, rid)
                    p1 = _pick_id_from_v(rolls, v1)
                    ob_o, msk_o, _ = extract_ramp_obs(sim, rid, mode=r5_orig.mode)
                    p_orig = int(np.argmax(r5_orig._logits(ob_o, msk_o)))
                    ob_r, msk_r, _ = extract_ramp_obs(sim, rid, mode=r5_rob.mode)
                    p_rob = int(np.argmax(r5_rob._logits(ob_r, msk_r)))

                    if p1 != p_orig:
                        disagree_r1_vs_r5orig += 1
                    if p1 != p_rob:
                        disagree_r1_vs_r5rob += 1
                    if p1 not in safe_ids:
                        r1_unsafe_overridden += 1
                    if p_orig not in safe_ids:
                        r5orig_unsafe_overridden += 1
                    if p_rob not in safe_ids:
                        r5rob_unsafe_overridden += 1

                    rs = regime_stats.setdefault(reg, {"count": 0, "k_safe_sum": 0, "k_safe_eq1": 0, "disagree_r1_r5rob": 0, "r5orig_unsafe": 0, "r5rob_unsafe": 0, "r1_unsafe": 0})
                    rs["count"] += 1
                    rs["k_safe_sum"] += k_s
                    if k_s == 1:
                        rs["k_safe_eq1"] += 1
                    if p1 != p_rob:
                        rs["disagree_r1_r5rob"] += 1
                    if p_orig not in safe_ids:
                        rs["r5orig_unsafe"] += 1
                    if p_rob not in safe_ids:
                        rs["r5rob_unsafe"] += 1
                    if p1 not in safe_ids:
                        rs["r1_unsafe"] += 1

            if sim.step(m2_res_margin_selector, controller_name="R1_Robust"):
                break

    for reg, d in regime_stats.items():
        n = max(1, d["count"])
        d["k_safe_avg"] = round(d["k_safe_sum"] / n, 2)
        d["k_safe_eq1_pct"] = round(100.0 * d["k_safe_eq1"] / n, 1)
        d["disagree_r1_r5rob_pct"] = round(100.0 * d["disagree_r1_r5rob"] / n, 1)
        d["r1_override_pct"] = round(100.0 * d["r1_unsafe"] / n, 1)
        d["r5orig_override_pct"] = round(100.0 * d["r5orig_unsafe"] / n, 1)
        d["r5rob_override_pct"] = round(100.0 * d["r5rob_unsafe"] / n, 1)

    report = {
        "total_decision_snapshots": tot_samples,
        "k_geom_pct": {str(k): round(100.0 * v / max(1, tot_samples), 1) for k, v in sorted(k_geom_hist.items())},
        "k_safe_pct": {str(k): round(100.0 * v / max(1, tot_samples), 1) for k, v in sorted(k_safe_hist.items())},
        "overall": {
            "disagree_r1_vs_r5orig_pct": round(100.0 * disagree_r1_vs_r5orig / max(1, tot_samples), 1),
            "disagree_r1_vs_r5rob_pct": round(100.0 * disagree_r1_vs_r5rob / max(1, tot_samples), 1),
            "r1_shield_override_pct": round(100.0 * r1_unsafe_overridden / max(1, tot_samples), 1),
            "r5orig_shield_override_pct": round(100.0 * r5orig_unsafe_overridden / max(1, tot_samples), 1),
            "r5rob_shield_override_pct": round(100.0 * r5rob_unsafe_overridden / max(1, tot_samples), 1),
        },
        "by_regime": regime_stats,
    }
    (out_dir / "task2_primitive_freedom.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[Task2 Freedom | N={tot_samples}] K_safe={report['k_safe_pct']} overrides: R1={report['overall']['r1_shield_override_pct']}% R5Orig={report['overall']['r5orig_shield_override_pct']}% R5Rob={report['overall']['r5rob_shield_override_pct']}%", flush=True)


if __name__ == "__main__":
    main()
