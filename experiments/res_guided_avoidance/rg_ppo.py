"""P2 Multi-Robot Squashed-Gaussian PPO (`RG-PPO`) & Controlled Ablation Trainer.

Trains decentralized continuous unicycle policies with CTDE centralized critic inside
the unified `MultiRobotUnicycleSim` (supporting `BC_Only`, `Pure_PPO_NoBC`, `Plain_Kin_PPO`,
`Intent_NoTime_PPO`, and `RG_PPO`), exporting pure-NumPy `.npz` inference weights.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from experiments.res_guided_avoidance.geometry_and_shield import (
    MAX_OMEGA_RADPS,
    MAX_V_MPS,
    clip_val,
    wrap_pi,
)
from experiments.res_guided_avoidance.metric_world import (
    MultiRobotUnicycleSim,
    build_grid_metric_map,
    build_occluded_junction_map,
    build_workshop_metric_map,
)
from experiments.res_guided_avoidance.reservation_features import (
    OBS_TOTAL_DIM,
    ObsMode,
    extract_centralized_critic_state,
    extract_multi_robot_obs,
)

CKPT_DIR = Path(__file__).resolve().parents[2] / "amr_sched" / "checkpoints" / "res_guided"


class MultiRobotActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_TOTAL_DIM, critic_dim: int = OBS_TOTAL_DIM + 6, hidden: int = 96) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(critic_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.tensor([-1.15, -1.15], dtype=torch.float32))

    def forward_raw_mean(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.actor(x), -4.0, 4.0)

    @staticmethod
    def squashed_log_prob(dist: torch.distributions.Normal, u_raw: torch.Tensor) -> torch.Tensor:
        a = torch.tanh(u_raw)
        return (dist.log_prob(u_raw) - torch.log(torch.clamp(1.0 - a * a, min=1e-6))).sum(-1)


class NumpyMultiRobotPolicy:
    """Zero-PyTorch pure-NumPy actor for fast decentralized execution in `MultiRobotUnicycleSim`."""

    def __init__(self, weights: dict[str, np.ndarray], mode: ObsMode = "full_res", metadata: Optional[dict[str, Any]] = None) -> None:
        self.w = weights
        self.mode: ObsMode = mode
        self.metadata = dict(metadata or {})

    @classmethod
    def load_npz(cls, path: Path) -> "NumpyMultiRobotPolicy":
        raw = np.load(str(path), allow_pickle=False)
        weights = {k: raw[k].astype(np.float32) for k in raw.files if k != "__meta_json__"}
        meta: dict[str, Any] = {}
        if "__meta_json__" in raw.files:
            meta = json.loads(str(raw["__meta_json__"].item()))
        mode: ObsMode = meta.get("obs_mode", "full_res")
        return cls(weights, mode=mode, metadata=meta)

    def act_obs(self, obs: np.ndarray) -> tuple[float, float]:
        h = np.tanh(obs @ self.w["actor.0.weight"].T + self.w["actor.0.bias"])
        h = np.tanh(h @ self.w["actor.2.weight"].T + self.w["actor.2.bias"])
        u_raw = np.clip(h @ self.w["actor.4.weight"].T + self.w["actor.4.bias"], -4.0, 4.0)
        a = np.tanh(u_raw)
        v_cmd = (0.5 * (float(a[0]) + 1.0)) * (MAX_V_MPS - 0.05) + 0.05
        w_cmd = float(a[1]) * MAX_OMEGA_RADPS
        return v_cmd, w_cmd

    def __call__(self, sim: MultiRobotUnicycleSim, rid: str) -> tuple[float, float]:
        obs = extract_multi_robot_obs(sim, rid, mode=self.mode)
        return self.act_obs(obs)


def _teacher_avoidance_cmd(sim: MultiRobotUnicycleSim, rid: str, *, use_res: bool = True) -> tuple[float, float]:
    """Heuristic teacher used ONLY for BC warm-start & `BC_Only` baseline."""
    rb = sim.robots[rid]
    wx, wy = sim._advance_route_cursor(rb)
    ang = math.atan2(wy - rb.y, wx - rb.x)
    err = wrap_pi(ang - rb.theta)
    v = 1.05 if abs(err) < 0.6 else 0.35
    w = clip_val(2.2 * err, -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS)

    obs_vis = sim.get_perceived_circles(rid)
    if obs_vis:
        nearest = min(obs_vis, key=lambda o: math.hypot(o.x - rb.x, o.y - rb.y) - o.radius)
        d = math.hypot(nearest.x - rb.x, nearest.y - rb.y) - (sim.fp.max_extent_m + nearest.radius)
        rel_ang = wrap_pi(math.atan2(nearest.y - rb.y, nearest.x - rb.x) - rb.theta)
        if abs(rel_ang) < 1.1 and d < 1.8:
            v = min(v, 0.45 if d > 0.7 else 0.15)
            w = clip_val(w + (-1.1 if rel_ang >= 0.0 else 1.1), -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS)

    if use_res:
        obs_full = extract_multi_robot_obs(sim, rid, mode="full_res")
        # Slot 0: (s_norm, dt_arr, ovl, prio_diff, cos_psi, sin_psi)
        s0, dt0, _, prio0 = float(obs_full[24]), float(obs_full[25]), float(obs_full[26]), float(obs_full[27])
        if s0 < 0.65 and abs(dt0) < 0.30 and prio0 >= 0.0:
            v = min(v, 0.40)
    return v, w


def _build_train_maps() -> list:
    return [
        build_occluded_junction_map(cell_m=2.6),
        build_grid_metric_map("Map_A", cell_m=2.6),
        build_workshop_metric_map(),
    ]


def train_variant(
    variant_name: str,
    out_npz: Path,
    *,
    obs_mode: ObsMode,
    warmstart_bc: bool = True,
    bc_only: bool = False,
    n_iters: int = 16,
    rollout_steps_per_env: int = 180,
    seed: int = 42,
) -> dict[str, Any]:
    """Train one controlled variant and export `.npz` weights."""
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    model = MultiRobotActorCritic()
    t0 = time.perf_counter()
    maps = _build_train_maps()

    if warmstart_bc or bc_only:
        bc_obs: list[np.ndarray] = []
        bc_act: list[np.ndarray] = []
        for k, mmap in enumerate(maps):
            sim = MultiRobotUnicycleSim(
                mmap,
                n_robots=6 if "Workshop" not in mmap.name else 8,
                n_tasks=10,
                horizon_s=45.0,
                seed=100 + k + seed,
                reservation_lag_s=0.5 if obs_mode == "full_res" else 0.0,
                n_intruders=2,
            )
            for _ in range(140):
                for rid in sim.robot_ids:
                    if sim.robots[rid].phase == "idle" or sim.robots[rid].service_rem_s > 0.0:
                        continue
                    ob = extract_multi_robot_obs(sim, rid, mode=obs_mode)
                    vc, wc = _teacher_avoidance_cmd(sim, rid, use_res=(obs_mode == "full_res"))
                    a0 = clip_val((vc - 0.05) / (MAX_V_MPS - 0.05) * 2.0 - 1.0, -0.95, 0.95)
                    a1 = clip_val(wc / MAX_OMEGA_RADPS, -0.95, 0.95)
                    bc_obs.append(ob)
                    bc_act.append(np.array([a0, a1], dtype=np.float32))
                if sim.step(lambda s, r: _teacher_avoidance_cmd(s, r, use_res=(obs_mode == "full_res"))):
                    break

        X_bc = torch.from_numpy(np.asarray(bc_obs, dtype=np.float32))
        Y_bc = torch.from_numpy(np.asarray(bc_act, dtype=np.float32))
        opt_bc = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(280):
            opt_bc.zero_grad()
            pred = torch.tanh(model.forward_raw_mean(X_bc))
            loss_bc = torch.mean((pred - Y_bc) ** 2)
            loss_bc.backward()
            opt_bc.step()

    opt = torch.optim.Adam(model.parameters(), lr=1.2e-4 if warmstart_bc else 3e-4)
    gamma, gae_lam = 0.98, 0.95
    clip_eps = 0.18
    max_lp_err = 0.0
    mean_rets: list[float] = []

    for it in range(0 if bc_only else int(n_iters)):
        mmap = maps[it % len(maps)]
        # Mild stochastic reservation lag during training (0.0 - 0.75 s) for P2 robustness
        train_lag = 0.5 if (obs_mode == "full_res" and (it % 2 == 1)) else 0.0
        sim = MultiRobotUnicycleSim(
            mmap,
            n_robots=6 if "Workshop" not in mmap.name else 8,
            n_tasks=10,
            horizon_s=45.0,
            seed=1000 + seed * 31 + it,
            reservation_lag_s=train_lag,
            n_intruders=2,
        )

        trajs: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]]] = {
            rid: [] for rid in sim.robot_ids
        }
        prev_goals = {rid: 0.0 for rid in sim.robot_ids}
        ep_ret_sum = 0.0

        for _ in range(int(rollout_steps_per_env)):
            # Step once with sampled actions for all active robots
            step_records: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]] = {}

            def _ppo_ctrl(s_ref: MultiRobotUnicycleSim, rid: str) -> tuple[float, float]:
                rb = s_ref.robots[rid]
                ob = extract_multi_robot_obs(s_ref, rid, mode=obs_mode)
                c_st = extract_centralized_critic_state(s_ref, rid, ob)
                with torch.no_grad():
                    ob_t = torch.from_numpy(ob[None, :])
                    cst_t = torch.from_numpy(c_st[None, :])
                    mu_raw = model.forward_raw_mean(ob_t)
                    std = torch.exp(torch.clamp(model.log_std, -2.2, -0.7))
                    dist = torch.distributions.Normal(mu_raw, std)
                    u_raw = dist.sample()
                    a_sq = torch.tanh(u_raw)
                    lp = float(MultiRobotActorCritic.squashed_log_prob(dist, u_raw).item())
                    val = float(model.critic(cst_t).squeeze().item())
                a_np = a_sq[0].numpy()
                v_cmd = (0.5 * (float(a_np[0]) + 1.0)) * (MAX_V_MPS - 0.05) + 0.05
                w_cmd = float(a_np[1]) * MAX_OMEGA_RADPS
                wx, wy = s_ref._advance_route_cursor(rb)
                d_before = math.hypot(wx - rb.x, wy - rb.y)
                step_records[rid] = (ob, c_st, u_raw[0].numpy().astype(np.float32), lp, val, d_before, float(s_ref.completed_tasks))
                return v_cmd, w_cmd

            col_before = sim.wall_collisions + sim.peer_collisions + sim.intruder_collisions
            sh_before = sim.shield_interventions
            done = sim.step(_ppo_ctrl, controller_name=variant_name)
            col_after = sim.wall_collisions + sim.peer_collisions + sim.intruder_collisions
            sh_after = sim.shield_interventions

            for rid, (ob, c_st, u_np, lp, val, d_before, comp_before) in step_records.items():
                rb = sim.robots[rid]
                wx, wy = sim._advance_route_cursor(rb)
                d_after = math.hypot(wx - rb.x, wy - rb.y)
                prog = d_before - d_after
                rew = (
                    2.5 * prog
                    + 0.35 * (rb.v / MAX_V_MPS)
                    - 0.03 * (rb.omega ** 2)
                    - 0.04
                )
                if sim.completed_tasks > comp_before:
                    rew += 3.0
                if sh_after > sh_before:
                    rew -= 0.15 / max(1.0, float(len(step_records)))
                if col_after > col_before:
                    rew -= 12.0 / max(1.0, float(len(step_records)))
                trajs[rid].append((ob, c_st, u_np, lp, val, rew))
                ep_ret_sum += rew
            if done:
                break

        mean_rets.append(ep_ret_sum / max(1, sim.n_robots))

        # Assemble GAE across all active robot trajectories
        X_list, C_list, U_list, LP_list, ADV_list, RET_list = [], [], [], [], [], []
        for rid, recs in trajs.items():
            if len(recs) < 2:
                continue
            T_r = len(recs)
            adv = np.zeros(T_r, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(T_r)):
                nxt_v = 0.0 if t == T_r - 1 else recs[t + 1][4]
                delta = recs[t][5] + gamma * nxt_v - recs[t][4]
                gae = delta + gamma * gae_lam * gae
                adv[t] = gae
            vals = np.array([r[4] for r in recs], dtype=np.float32)
            ret = adv + vals
            for t in range(T_r):
                X_list.append(recs[t][0])
                C_list.append(recs[t][1])
                U_list.append(recs[t][2])
                LP_list.append(recs[t][3])
                ADV_list.append(adv[t])
                RET_list.append(ret[t])

        if len(X_list) < 32:
            continue

        X_tr = torch.from_numpy(np.asarray(X_list, dtype=np.float32))
        C_tr = torch.from_numpy(np.asarray(C_list, dtype=np.float32))
        U_tr = torch.from_numpy(np.asarray(U_list, dtype=np.float32))
        LP_old = torch.from_numpy(np.asarray(LP_list, dtype=np.float32))
        ADV_t = torch.from_numpy(np.asarray(ADV_list, dtype=np.float32))
        ADV_t = (ADV_t - ADV_t.mean()) / (ADV_t.std() + 1e-6)
        RET_t = torch.from_numpy(np.asarray(RET_list, dtype=np.float32))

        with torch.no_grad():
            mu_chk = model.forward_raw_mean(X_tr)
            std_chk = torch.exp(torch.clamp(model.log_std, -2.2, -0.7))
            lp_chk = MultiRobotActorCritic.squashed_log_prob(torch.distributions.Normal(mu_chk, std_chk), U_tr)
            max_lp_err = max(max_lp_err, float((lp_chk - LP_old).abs().max().item()))

        T_tot = X_tr.shape[0]
        for _ in range(3):
            perm = rng.permutation(T_tot)
            for s0 in range(0, T_tot, 256):
                mb = perm[s0 : s0 + 256]
                mu_raw = model.forward_raw_mean(X_tr[mb])
                std = torch.exp(torch.clamp(model.log_std, -2.2, -0.7))
                dist = torch.distributions.Normal(mu_raw, std)
                lp = MultiRobotActorCritic.squashed_log_prob(dist, U_tr[mb])
                if float((LP_old[mb] - lp).mean().item()) > 0.03:
                    break
                ratio = torch.exp(lp - LP_old[mb])
                surr1 = ratio * ADV_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * ADV_t[mb]
                p_loss = -torch.min(surr1, surr2).mean()
                v_loss = 0.5 * ((model.critic(C_tr[mb]).squeeze(-1) - RET_t[mb]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = p_loss + 0.5 * v_loss - 0.005 * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
                opt.step()

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "variant": variant_name,
        "obs_mode": obs_mode,
        "warmstart_bc": bool(warmstart_bc),
        "bc_only": bool(bc_only),
        "n_iters": 0 if bc_only else int(n_iters),
        "max_lp_consistency_err": round(max_lp_err, 8),
        "mean_ret_init": round(mean_rets[0], 2) if mean_rets else 0.0,
        "mean_ret_final": round(mean_rets[-1], 2) if mean_rets else 0.0,
        "train_s": round(time.perf_counter() - t0, 2),
    }
    np.savez(
        str(out_npz),
        **{k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()},
        __meta_json__=np.asarray(json.dumps(meta)),
    )
    return meta


def train_all_p2_variants(*, seed: int = 42, n_iters: int = 16) -> dict[str, dict[str, Any]]:
    """Train all 5 controlled P2 variants cleanly under identical seed/budget."""
    specs: list[tuple[str, ObsMode, bool, bool]] = [
        ("BC_Only", "full_res", True, True),
        ("Pure_PPO_NoBC", "kin_only", False, False),
        ("Plain_Kin_PPO", "kin_only", True, False),
        ("Intent_NoTime_PPO", "intent_notime", True, False),
        ("RG_PPO", "full_res", True, False),
    ]
    out: dict[str, dict[str, Any]] = {}
    for name, mode, w_bc, bco in specs:
        ckpt = CKPT_DIR / f"{name.lower()}_s{seed}.npz"
        out[name] = train_variant(
            name, ckpt, obs_mode=mode, warmstart_bc=w_bc, bc_only=bco, n_iters=n_iters, seed=seed
        )
    return out
