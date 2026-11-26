"""Unit tests for P0-P1: Unified Geometry, Braking-Invariant Shield, Squashed PPO, & Multi-Robot World."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.res_guided_avoidance.geometry_and_shield import (
    BrakingInvariantShield,
    CircleState,
    FootprintConfig,
    corridor_wall_clearance,
    swept_corridor_collision,
)
from experiments.res_guided_avoidance.metric_world import (
    MultiRobotUnicycleSim,
    build_grid_metric_map,
    build_occluded_junction_map,
    build_workshop_metric_map,
)
from experiments.track_r_cpsa.route_b_kinematic_rl import (
    DynamicObstacle,
    UnicycleCorridorEnv,
    build_corridor_scenario,
    train_route_b_ppo,
)


def test_p0_unified_wall_and_obstacle_geometry_and_swept_tunneling() -> None:
    fp_disc = FootprintConfig(mode="disc", radius_m=0.45)
    fp_cap = FootprintConfig(mode="capsule", radius_m=0.45, half_offset_m=0.15)
    hw = 1.10  # 2.20 m workshop branch aisle
    # Wall clearance must use exact 0.45 m radius (1.10 - 0.65 - 0.45 == 0.0)
    assert abs(corridor_wall_clearance(0.0, 0.65, 0.0, hw, fp_disc) - 0.0) < 1e-9
    assert corridor_wall_clearance(0.0, 0.66, 0.0, hw, fp_disc) < 0.0
    # Rotated capsule has larger lateral projection than disc: 0.15 * sin(pi/2) + 0.45 = 0.60
    assert abs(corridor_wall_clearance(0.0, 0.50, math.pi / 2.0, hw, fp_cap) - 0.0) < 1e-9

    # Swept collision must catch mid-step tunneling across a thin obstacle between t and t+dt
    obs0 = [CircleState(x=2.0, y=0.0, vx=0.0, vy=0.0, radius=0.20)]
    obs1 = [CircleState(x=2.0, y=0.0, vx=0.0, vy=0.0, radius=0.20)]
    # Start x=1.2 (clear = 0.15 > 0), End x=2.8 (clear = 0.15 > 0), midpoint x=2.0 collides!
    collided, min_o, _ = swept_corridor_collision(1.2, 0.0, 0.0, 2.8, 0.0, 0.0, 1.5, obs0, obs1, fp_disc, substeps=5)
    assert collided is True and min_o < 0.0


def test_p0_stop_at_y_bound_and_braking_shield_infeasible_flag() -> None:
    sc = build_corridor_scenario("Crossing_Narrow_2.2m", seed=7)
    env = UnicycleCorridorEnv(sc, obs_noise_std=0.0, seed=7)
    assert env.obstacles[0].stop_at_y_bound is True

    shield = BrakingInvariantShield(footprint=FootprintConfig(mode="disc", radius_m=0.45), hard_min_clear_m=0.08)
    # Feasible braking scenario: obstacle 1.8m ahead
    out_ok = shield.filter(
        x=0.0, y=0.0, theta=0.0, v=1.0, omega=0.0, v_cmd=1.35, omega_cmd=0.0,
        obstacles=[CircleState(x=1.8, y=0.0, vx=0.0, vy=0.0, radius=0.35)],
        wall_fn=lambda px, py, pth: corridor_wall_clearance(px, py, pth, 1.5),
    )
    assert out_ok.infeasible is False and out_ok.pred_min_clear_m >= 0.0

    # Impossibly boxed-in scenario must explicitly set infeasible=True
    out_bad = shield.filter(
        x=0.0, y=0.0, theta=0.0, v=1.35, omega=0.0, v_cmd=1.35, omega_cmd=0.0,
        obstacles=[CircleState(x=0.82, y=0.0, vx=-0.5, vy=0.0, radius=0.35)],
        wall_fn=lambda px, py, pth: corridor_wall_clearance(px, py, pth, 0.55),
    )
    assert out_bad.intervened is True and out_bad.infeasible is True


def test_p0_squashed_ppo_exact_logp_consistency(tmp_path: Path) -> None:
    ckpt = tmp_path / "test_ppo.npz"
    meta = train_route_b_ppo(ckpt, n_iters=2, rollout_steps=128, seed=17, warmstart_bc=False)
    assert ckpt.is_file()
    assert meta["max_lp_consistency_err"] < 1e-5


def test_p1_unified_multi_robot_sim_workshop_and_occluded_junction() -> None:
    wm = build_workshop_metric_map()
    assert "Workshop" in wm.name and len(wm.corridor_capsules) > 50

    def _simple_follower(sim: MultiRobotUnicycleSim, rid: str) -> tuple[float, float]:
        rb = sim.robots[rid]
        wx, wy = sim._advance_route_cursor(rb)
        ang = math.atan2(wy - rb.y, wx - rb.x)
        err = (ang - rb.theta + math.pi) % (2.0 * math.pi) - math.pi
        v = 0.85 if abs(err) < 0.7 else 0.25
        return v, max(-1.65, min(1.65, 2.0 * err))

    sim_w = MultiRobotUnicycleSim(wm, n_robots=8, n_tasks=10, horizon_s=60.0, seed=42, n_intruders=2)
    res_w = sim_w.run_episode(_simple_follower, controller_name="Follower_Shield")
    assert res_w.zero_collisions is True
    assert res_w.replans >= 8

    om = build_occluded_junction_map(cell_m=2.6)
    # Blind corner test: robot at (3.9, 1.3) cannot see robot at (1.3, 3.9) across wall block (1,1)-(2,2)
    assert om.has_line_of_sight(3.5 * 2.6, 0.5 * 2.6, 0.5 * 2.6, 3.5 * 2.6) is False
    sim_o = MultiRobotUnicycleSim(om, n_robots=6, n_tasks=8, horizon_s=50.0, seed=42, n_intruders=1)
    res_o = sim_o.run_episode(_simple_follower, controller_name="Follower_Shield")
    assert res_o.zero_collisions is True


def test_p2_reservation_features_ablation_masks_and_trainer(tmp_path: Path) -> None:
    from experiments.res_guided_avoidance.reservation_features import (
        KIN_DIM,
        OBS_TOTAL_DIM,
        extract_centralized_critic_state,
        extract_multi_robot_obs,
    )
    from experiments.res_guided_avoidance.rg_ppo import NumpyMultiRobotPolicy, train_variant

    om = build_occluded_junction_map(cell_m=2.6)
    sim = MultiRobotUnicycleSim(om, n_robots=4, n_tasks=6, horizon_s=30.0, seed=19, n_intruders=0)
    sim._macro_coordinate()
    rid = sim.robot_ids[0]
    obs_kin = extract_multi_robot_obs(sim, rid, mode="kin_only")
    obs_not = extract_multi_robot_obs(sim, rid, mode="intent_notime")
    obs_full = extract_multi_robot_obs(sim, rid, mode="full_res")
    assert obs_full.shape == (OBS_TOTAL_DIM,)
    assert np.allclose(obs_kin[KIN_DIM:], 0.0)
    # In intent_notime, temporal/priority/field slots (offsets 1,2,3 and last 2 dims) must be identically 0.0
    for s in range(4):
        off = KIN_DIM + s * 6
        assert obs_not[off + 1] == 0.0 and obs_not[off + 2] == 0.0 and obs_not[off + 3] == 0.0
    assert np.allclose(obs_not[-2:], 0.0)
    cst = extract_centralized_critic_state(sim, rid, obs_full)
    assert cst.shape == (OBS_TOTAL_DIM + 6,)

    ckpt = tmp_path / "rg_mini.npz"
    meta = train_variant("RG_Mini", ckpt, obs_mode="full_res", warmstart_bc=False, n_iters=2, rollout_steps_per_env=24, seed=7)
    assert ckpt.is_file() and meta["max_lp_consistency_err"] < 1e-5
    pol = NumpyMultiRobotPolicy.load_npz(ckpt)
    v, w = pol(sim, rid)
    assert np.isfinite(v) and np.isfinite(w)


def test_p3_continuous_baselines_run_safely() -> None:
    from experiments.res_guided_avoidance.continuous_baselines import (
        dwa_controller,
        orca_uni_controller,
        res_dwa_controller,
        res_yield_controller,
        uni_mpc_controller,
    )

    om = build_occluded_junction_map(cell_m=2.6)
    for name, fn in (
        ("ORCA_Uni", orca_uni_controller),
        ("DWA", dwa_controller),
        ("Res_DWA", res_dwa_controller),
        ("Uni_MPC", uni_mpc_controller),
        ("Res_Yield", res_yield_controller),
    ):
        sim = MultiRobotUnicycleSim(om, n_robots=4, n_tasks=6, horizon_s=30.0, seed=42, n_intruders=1)
        res = sim.run_episode(fn, controller_name=name)
        assert res.zero_collisions is True, f"{name} collided: {res}"

