"""Task 1: Collision Forensics & D1–D4 Single-Variable Diagnostics.

Replays every colliding episode in `Workshop_8R` and `Noise_Only_0.12m`, logs the exact
pre-collision trajectory & state window (`[t_col-6 .. t_col]`), classifies Wall / Peer / Intruder
events, and executes D1–D4 single-variable causal tests:
  - D1: Actor-only 0.12 m noise (Shield receives clean state)
  - D2: Safety-check-only 0.12 m noise (Actor receives clean state)
  - D3: Reservation ETA-only 2.0 s lag (zero perception noise)
  - D4: Independent vs Local Joint Action Check (`diag_joint_action_check=True`)
"""

from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.res_guided_avoidance.continuous_baselines import (
    cpsa_follower_controller,
    dwa_controller,
    orca_uni_controller,
    res_dwa_controller,
    tpg_adg_controller,
)
from experiments.res_guided_avoidance.gate1_headroom_audit import (
    compute_reservation_margin_signal,
    m0_perception_selector,
    m2_res_margin_selector,
)
from experiments.res_guided_avoidance.geometry_and_shield import (
    SUBSTEPS_DEFAULT,
    obstacle_surface_clearance,
    pair_robot_clearance,
    wrap_pi,
)
from experiments.res_guided_avoidance.metric_world import (
    MultiRobotUnicycleSim,
    build_occluded_junction_map,
    build_workshop_metric_map,
)
from experiments.res_guided_avoidance.motion_primitives import generate_primitive_rollouts
from experiments.res_guided_avoidance.ramp_ppo import RAMP_CKPT_DIR, NumpyRampPolicy, extract_ramp_obs


def _replay_first_collision_window(
    sim: MultiRobotUnicycleSim,
    pol,
    controller_name: str,
) -> dict[str, Any] | None:
    """Step episode and capture exact [t_col-6 .. t_col] forensic trace on first collision."""
    ring: deque[dict[str, Any]] = deque(maxlen=7)
    first_col_trace: dict[str, Any] | None = None

    while True:
        snap_robots: dict[str, dict[str, Any]] = {}
        for rid in sim.robot_ids:
            rb = sim.robots[rid]
            if rb.phase == "idle":
                continue
            rolls = generate_primitive_rollouts(sim, rid)
            has_c, min_m, dist_c = compute_reservation_margin_signal(sim, rid)
            obs_clean = sim.get_perceived_circles(rid, clean=True)
            obs_noisy = sim.get_perceived_circles(rid, clean=False)
            min_true_peer = 9.0
            for o_id in sim.robot_ids:
                if o_id == rid:
                    continue
                orb = sim.robots[o_id]
                d_pr = pair_robot_clearance(rb.x, rb.y, rb.theta, sim.fp, orb.x, orb.y, orb.theta, sim.fp)
                if d_pr < min_true_peer:
                    min_true_peer = d_pr
            min_noisy_obs = min((math.hypot(o.x - rb.x, o.y - rb.y) - sim.fp.radius_m - o.radius for o in obs_noisy), default=9.0)
            chosen_name = "EXT_CTRL"
            if isinstance(pol, NumpyRampPolicy):
                ob, msk, rls = extract_ramp_obs(sim, rid, mode=pol.mode)
                pid = int(np.argmax(pol._logits(ob, msk)))
                chosen_name = rls[pid].name
            snap_robots[rid] = {
                "x": round(rb.x, 3),
                "y": round(rb.y, 3),
                "th": round(rb.theta, 3),
                "v": round(rb.v, 3),
                "w": round(rb.omega, 3),
                "wall_clr": round(sim.mmap.wall_clearance(rb.x, rb.y, rb.theta, sim.fp), 3),
                "true_peer_clr": round(min_true_peer, 3),
                "noisy_obs_clr": round(min_noisy_obs, 3),
                "res_margin": round(min_m, 3) if has_c else None,
                "legal_mask": [int(r.legal) for r in rolls],
                "chosen_prim": chosen_name,
            }

        pw, pp, pi = sim.wall_collisions, sim.peer_collisions, sim.intruder_collisions
        psh = sim.shield_interventions
        pinf = sim.shield_infeasible_count
        done = sim.step(pol, controller_name=controller_name)

        dw = sim.wall_collisions - pw
        dp = sim.peer_collisions - pp
        di = sim.intruder_collisions - pi
        ring.append({
            "t_s": sim.t_s,
            "dw": dw,
            "dp": dp,
            "di": di,
            "d_shield": sim.shield_interventions - psh,
            "d_infeas": sim.shield_infeasible_count - pinf,
            "robots": snap_robots,
        })

        if first_col_trace is None and (dw + dp + di) > 0:
            col_kind = "wall" if dw > 0 else ("peer" if dp > 0 else "intruder")
            # Identify closest colliding robots after step
            culprits = []
            for r1 in sim.robot_ids:
                rb1 = sim.robots[r1]
                wc1 = sim.mmap.wall_clearance(rb1.x, rb1.y, rb1.theta, sim.fp)
                if wc1 <= 0.05:
                    culprits.append({"rid": r1, "type": "wall", "clear_m": round(wc1, 4), "v": round(rb1.v, 3)})
                for r2 in sim.robot_ids:
                    if r2 <= r1:
                        continue
                    rb2 = sim.robots[r2]
                    pc = pair_robot_clearance(rb1.x, rb1.y, rb1.theta, sim.fp, rb2.x, rb2.y, rb2.theta, sim.fp)
                    if pc <= 0.05:
                        culprits.append({"pair": [r1, r2], "type": "peer", "clear_m": round(pc, 4), "vs": [round(rb1.v, 3), round(rb2.v, 3)]})
            first_col_trace = {
                "first_col_t_s": sim.t_s,
                "first_col_kind": col_kind,
                "culprits_at_t_col": culprits,
                "pre_collision_window": list(ring)[-4:],
            }
        if done:
            break

    if (sim.wall_collisions + sim.peer_collisions + sim.intruder_collisions) == 0:
        return None
    return {
        "map": sim.mmap.name,
        "controller": controller_name,
        "seed": sim.seed,
        "wall_events": sim.wall_collisions,
        "peer_events": sim.peer_collisions,
        "intruder_events": sim.intruder_collisions,
        "completed": sim.completed_tasks,
        **(first_col_trace or {}),
    }


def run_forensics_and_d1_d4() -> dict[str, Any]:
    om = build_occluded_junction_map(cell_m=2.6)
    wm = build_workshop_metric_map()

    r5_s42 = NumpyRampPolicy.load_npz(RAMP_CKPT_DIR / "r5_ramp_ppo_s42.npz")
    r5_s43 = NumpyRampPolicy.load_npz(RAMP_CKPT_DIR / "r5_ramp_ppo_s43.npz")
    r5_s44 = NumpyRampPolicy.load_npz(RAMP_CKPT_DIR / "r5_ramp_ppo_s44.npz")
    r4_s42 = NumpyRampPolicy.load_npz(RAMP_CKPT_DIR / "r4_spatial_intent_ppo_s42.npz")
    r4_s43 = NumpyRampPolicy.load_npz(RAMP_CKPT_DIR / "r4_spatial_intent_ppo_s43.npz")
    r4_s44 = NumpyRampPolicy.load_npz(RAMP_CKPT_DIR / "r4_spatial_intent_ppo_s44.npz")

    # 1. Exact replay of colliding cases in Workshop_8R and Noise_Only_0.12m
    replays: list[dict[str, Any]] = []
    ws_targets = [
        ("R0_Det_Primitive", m0_perception_selector),
        ("R5_RAMP_PPO", (r5_s42, r5_s43, r5_s44)),
        ("R4_Spatial_Intent_PPO", (r4_s42, r4_s43, r4_s44)),
        ("CPSA_Follower", cpsa_follower_controller),
        ("ORCA_Uni", orca_uni_controller),
        ("TPG_ADG_Follower", tpg_adg_controller),
    ]
    for cname, pol_or_tuple in ws_targets:
        for idx, sd in enumerate((6001, 6002, 6003)):
            pol = pol_or_tuple[idx] if isinstance(pol_or_tuple, tuple) else pol_or_tuple
            sim = MultiRobotUnicycleSim(wm, n_robots=8, n_tasks=12, horizon_s=160.0, seed=sd, n_intruders=1)
            rep = _replay_first_collision_window(sim, pol, cname)
            if rep is not None:
                replays.append(rep)
                print(
                    f"[Forensics | Workshop_8R | {cname:21s} | s={sd}] kind={rep.get('first_col_kind')} "
                    f"events=({rep['wall_events']},{rep['peer_events']},{rep['intruder_events']}) t={rep.get('first_col_t_s')}s culprits={rep.get('culprits_at_t_col')}",
                    flush=True,
                )

    for cname, pol_or_tuple in (("R4_Spatial_Intent_PPO", (r4_s42, r4_s43, r4_s44)), ("R5_RAMP_PPO", (r5_s42, r5_s43, r5_s44)), ("Res_DWA", res_dwa_controller)):
        for idx, sd in enumerate((6001, 6002, 6003)):
            pol = pol_or_tuple[idx] if isinstance(pol_or_tuple, tuple) else pol_or_tuple
            sim = MultiRobotUnicycleSim(om, n_robots=6, n_tasks=12, horizon_s=70.0, seed=sd, n_intruders=2, obs_noise_pos_m=0.12, obs_noise_vel_mps=0.20)
            rep = _replay_first_collision_window(sim, pol, cname)
            if rep is not None:
                replays.append(rep)
                print(
                    f"[Forensics | Noise_0.12m | {cname:21s} | s={sd}] kind={rep.get('first_col_kind')} "
                    f"events=({rep['wall_events']},{rep['peer_events']},{rep['intruder_events']}) t={rep.get('first_col_t_s')}s",
                    flush=True,
                )

    # 2. D1-D4 Single-Variable Diagnostics
    d_configs = [
        ("Baseline_BothNoisy", om, 6, 12, 70.0, 0.12, 0.20, 0.0, False, False, False),
        ("D1_ActorOnlyNoise", om, 6, 12, 70.0, 0.12, 0.20, 0.0, True, False, False),
        ("D2_ShieldOnlyNoise", om, 6, 12, 70.0, 0.12, 0.20, 0.0, False, True, False),
        ("D3_ETA_Lag2sOnly", om, 6, 12, 70.0, 0.0, 0.0, 2.0, False, False, False),
        ("D4_JointCheck_Noise", om, 6, 12, 70.0, 0.12, 0.20, 0.0, False, False, True),
        ("WS_Baseline", wm, 8, 12, 160.0, 0.0, 0.0, 0.0, False, False, False),
        ("WS_D4_JointCheck", wm, 8, 12, 160.0, 0.0, 0.0, 0.0, False, False, True),
    ]
    d_rows: list[dict[str, Any]] = []
    diag_ctrls = [
        ("R1_Det_ResMargin", (m2_res_margin_selector, m2_res_margin_selector, m2_res_margin_selector)),
        ("R4_Spatial_Intent_PPO", (r4_s42, r4_s43, r4_s44)),
        ("R5_RAMP_PPO", (r5_s42, r5_s43, r5_s44)),
    ]
    for d_name, mmap, nr, nt, hz, np_m, nv_m, lag_s, c_sh, c_act, j_chk in d_configs:
        for cname, pols in diag_ctrls:
            res_list = []
            for idx, sd in enumerate((6001, 6002, 6003)):
                sim = MultiRobotUnicycleSim(
                    mmap, n_robots=nr, n_tasks=nt, horizon_s=hz, seed=sd, n_intruders=(1 if mmap is wm else 2),
                    obs_noise_pos_m=np_m, obs_noise_vel_mps=nv_m, reservation_lag_s=lag_s,
                    diag_clean_shield=c_sh, diag_clean_actor=c_act, diag_joint_action_check=j_chk,
                )
                res_list.append(sim.run_episode(pols[idx], controller_name=cname))
            zcol = round(100.0 * sum(1 for r in res_list if r.zero_collisions) / 3.0, 1)
            w_ev = sum(r.wall_collisions for r in res_list)
            p_ev = sum(r.peer_collisions for r in res_list)
            i_ev = sum(r.intruder_collisions for r in res_list)
            comp = round(float(np.mean([r.completed_tasks for r in res_list])), 2)
            wait = round(float(np.mean([r.wait_ratio for r in res_list])), 4)
            row = {"diag": d_name, "controller": cname, "zero_col_pct": zcol, "wall_ev": w_ev, "peer_ev": p_ev, "intr_ev": i_ev, "comp": comp, "wait": wait}
            d_rows.append(row)
            print(f"[{d_name:20s} | {cname:21s}] zcol={zcol:5.1f}% ev=(W:{w_ev},P:{p_ev},I:{i_ev}) comp={comp:.2f} wait={wait:.4f}", flush=True)

    return {"collision_replays": replays, "d1_d4_diagnostics": d_rows}


if __name__ == "__main__":
    out_dir = ROOT / "out" / "ramp_robust"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_forensics_and_d1_d4()
    (out_dir / "collision_forensics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"=== Wrote {out_dir / 'collision_forensics.json'} ===", flush=True)
