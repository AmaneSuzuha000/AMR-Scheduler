"""Gate 2: RAMP-PPO (Reservation-Aware Motion Primitive Masked Categorical PPO).

Replaces direct continuous `(v, w)` output with a masked categorical policy over
`K = 6` kinodynamic trajectory primitives, trained with clean 5-term path-progress
rewards and strict time-information isolation across controlled variants:
  - R2_BC_Only
  - R3_Primitive_PPO (kin_only)
  - R4_Spatial_Intent_PPO (spatial_intent, zero time info)
  - R5_RAMP_PPO (full_res + reservation time-margin & reliability features)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.res_guided_avoidance.gate1_headroom_audit import (
    compute_reservation_margin_signal,
    m0_perception_selector,
    m2_res_margin_selector,
)
from experiments.res_guided_avoidance.geometry_and_shield import DT_MICRO, clip_val
from experiments.res_guided_avoidance.metric_world import (
    MultiRobotUnicycleSim,
    build_grid_metric_map,
    build_occluded_junction_map,
    build_workshop_metric_map,
)
from experiments.res_guided_avoidance.motion_primitives import (
    N_PRIMITIVES,
    PrimitiveRollout,
    generate_primitive_rollouts,
)
from experiments.res_guided_avoidance.reservation_features import (
    OBS_TOTAL_DIM,
    extract_centralized_critic_state,
    extract_multi_robot_obs,
)

CRITIC_GLOBAL_DIM = OBS_TOTAL_DIM + 6  # 56-D

ROOT = Path(__file__).resolve().parents[2]
RAMP_CKPT_DIR = ROOT / "amr_sched" / "checkpoints" / "ramp"
RAMP_OBS_DIM = OBS_TOTAL_DIM + 4  # 50 + (has_conf, min_margin_norm, dist_conf_norm, eta_drift_norm)


def extract_ramp_obs(
    sim: MultiRobotUnicycleSim,
    rid: str,
    *,
    mode: str = "full_res",
) -> tuple[np.ndarray, np.ndarray, list[PrimitiveRollout]]:
    """Return `(obs_54, legal_mask_6, rollouts)` for RAMP policies."""
    base = extract_multi_robot_obs(sim, rid, mode=mode)
    out = np.zeros(RAMP_OBS_DIM, dtype=np.float32)
    out[:OBS_TOTAL_DIM] = base
    if mode == "full_res":
        rb = sim.robots[rid]
        has_c, min_m, dist_c = compute_reservation_margin_signal(sim, rid)
        out[OBS_TOTAL_DIM + 0] = 1.0 if has_c else 0.0
        out[OBS_TOTAL_DIM + 1] = clip_val(min_m / 4.0, -1.0, 2.0)
        out[OBS_TOTAL_DIM + 2] = clip_val(dist_c / 10.0, 0.0, 1.5)
        out[OBS_TOTAL_DIM + 3] = clip_val(rb.eta_drift_ticks / 5.0, 0.0, 1.5)
    rolls = generate_primitive_rollouts(sim, rid)
    mask = np.array([1.0 if r.legal else 0.0 for r in rolls], dtype=np.float32)
    return out, mask, rolls


class RampActorCritic(nn.Module):
    def __init__(self, obs_dim: int = RAMP_OBS_DIM, critic_dim: int = CRITIC_GLOBAL_DIM, n_prims: int = N_PRIMITIVES) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 96),
            nn.Tanh(),
            nn.Linear(96, 64),
            nn.Tanh(),
            nn.Linear(64, n_prims),
        )
        self.critic = nn.Sequential(
            nn.Linear(critic_dim, 96),
            nn.Tanh(),
            nn.Linear(96, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def masked_logits(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.actor(obs)
        return logits.masked_fill(mask <= 0.0, -1e9)

    def value(self, c_state: torch.Tensor) -> torch.Tensor:
        return self.critic(c_state).squeeze(-1)


class NumpyRampPolicy:
    """Zero-PyTorch runtime inference wrapper for `.npz` RAMP checkpoints."""

    def __init__(self, weights: dict[str, np.ndarray], *, mode: str = "full_res") -> None:
        self.w = weights
        self.mode = mode

    @classmethod
    def load_npz(cls, path: Path) -> "NumpyRampPolicy":
        data = np.load(path, allow_pickle=False)
        mode = str(data["mode"].item()) if "mode" in data else "full_res"
        return cls({k: data[k] for k in data.files if k != "mode"}, mode=mode)

    def _logits(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        x = np.tanh(obs @ self.w["actor.0.weight"].T + self.w["actor.0.bias"])
        x = np.tanh(x @ self.w["actor.2.weight"].T + self.w["actor.2.bias"])
        logits = x @ self.w["actor.4.weight"].T + self.w["actor.4.bias"]
        return np.where(mask > 0.0, logits, -1e9)

    def __call__(self, sim: MultiRobotUnicycleSim, rid: str) -> tuple[float, float]:
        obs, mask, rolls = extract_ramp_obs(sim, rid, mode=self.mode)
        logits = self._logits(obs, mask)
        pid = int(np.argmax(logits))
        return rolls[pid].v_cmd, rolls[pid].w_cmd


def _teacher_primitive_id(sim: MultiRobotUnicycleSim, rid: str, rolls: list[PrimitiveRollout], mode: str) -> int:
    """Match M2_ResMargin (full_res) or M0_Perception (kin_only/spatial_intent) to discrete primitive ID."""
    v_t, _ = m2_res_margin_selector(sim, rid) if mode == "full_res" else m0_perception_selector(sim, rid)
    best_id = 0
    best_d = 1e9
    for r in rolls:
        if not r.legal:
            continue
        d = abs(r.v_cmd - v_t)
        if d < best_d:
            best_d = d
            best_id = r.prim_id
    return best_id


@dataclass
class RampTrainResult:
    variant: str
    mode: str
    seed: int
    ckpt_path: Path
    elapsed_s: float
    bc_acc: float
    final_mean_ret: float


def train_ramp_variant(
    variant: str,
    *,
    mode: str,
    seed: int = 42,
    n_iters: int = 14,
    warmstart_bc: bool = True,
    bc_only: bool = False,
    robust_safety: bool = False,
    obs_noise_pos_m: float = 0.0,
    obs_noise_vel_mps: float = 0.0,
    out_path: Path | None = None,
) -> RampTrainResult:
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)

    om = build_occluded_junction_map(cell_m=2.6)
    ma = build_grid_metric_map("Map_A", cell_m=2.6)
    wm = build_workshop_metric_map()
    train_maps = (om, ma, wm)

    net = RampActorCritic()
    bc_acc = 0.0

    # 1. Matched Behavior Cloning warm-start
    if warmstart_bc or bc_only:
        bc_obs: list[np.ndarray] = []
        bc_masks: list[np.ndarray] = []
        bc_acts: list[int] = []
        for ep in range(6):
            mmap = train_maps[ep % len(train_maps)]
            nr = 6 if mmap is om else 8
            sim = MultiRobotUnicycleSim(
                mmap, n_robots=nr, n_tasks=10, horizon_s=50.0, seed=seed * 100 + ep,
                reservation_lag_s=0.5, n_intruders=1,
                robust_safety=robust_safety, obs_noise_pos_m=obs_noise_pos_m, obs_noise_vel_mps=obs_noise_vel_mps,
            )
            for _ in range(140):
                def _t_fn(s: MultiRobotUnicycleSim, r_id: str) -> tuple[float, float]:
                    ob, msk, rls = extract_ramp_obs(s, r_id, mode=mode)
                    pid = _teacher_primitive_id(s, r_id, rls, mode)
                    bc_obs.append(ob)
                    bc_masks.append(msk)
                    bc_acts.append(pid)
                    return rls[pid].v_cmd, rls[pid].w_cmd
                if sim.step(_t_fn, controller_name="Teacher"):
                    break
        opt_bc = torch.optim.Adam(net.actor.parameters(), lr=1.5e-3)
        x_t = torch.tensor(np.asarray(bc_obs), dtype=torch.float32)
        m_t = torch.tensor(np.asarray(bc_masks), dtype=torch.float32)
        y_t = torch.tensor(np.asarray(bc_acts), dtype=torch.long)
        for _ in range(25):
            logits = net.masked_logits(x_t, m_t)
            loss = F.cross_entropy(logits, y_t)
            opt_bc.zero_grad()
            loss.backward()
            opt_bc.step()
        with torch.no_grad():
            preds = net.masked_logits(x_t, m_t).argmax(dim=-1)
            bc_acc = float((preds == y_t).float().mean().item())

    final_ret = 0.0
    if not bc_only and n_iters > 0:
        opt = torch.optim.Adam(net.parameters(), lr=3e-4)
        for it in range(n_iters):
            mmap = train_maps[it % len(train_maps)]
            nr = 6 if mmap is om else 8
            sim = MultiRobotUnicycleSim(
                mmap, n_robots=nr, n_tasks=10, horizon_s=45.0,
                seed=seed * 1000 + it, reservation_lag_s=0.5, n_intruders=1,
                robust_safety=robust_safety, obs_noise_pos_m=obs_noise_pos_m, obs_noise_vel_mps=obs_noise_vel_mps,
            )
            ep_obs: list[np.ndarray] = []
            ep_masks: list[np.ndarray] = []
            ep_cs: list[np.ndarray] = []
            ep_acts: list[int] = []
            ep_lps: list[float] = []
            ep_vals: list[float] = []
            ep_rews: list[float] = []

            prev_comp = 0
            for _ in range(160):
                step_recs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float, float]] = {}
                prev_sh = sim.shield_interventions

                def _pol_fn(s: MultiRobotUnicycleSim, r_id: str) -> tuple[float, float]:
                    rb = s.robots[r_id]
                    ob, msk, rls = extract_ramp_obs(s, r_id, mode=mode)
                    cs = extract_centralized_critic_state(s, r_id, ob[:OBS_TOTAL_DIM])
                    with torch.no_grad():
                        ot = torch.from_numpy(ob).unsqueeze(0)
                        mt = torch.from_numpy(msk).unsqueeze(0)
                        ct = torch.from_numpy(cs).unsqueeze(0)
                        logits = net.masked_logits(ot, mt)
                        dist = torch.distributions.Categorical(logits=logits)
                        act = dist.sample()
                        lp = float(dist.log_prob(act).item())
                        val = float(net.value(ct).item())
                    pid = int(act.item())
                    rem_before = s.remaining_route_dist_m(rb)
                    step_recs[r_id] = (ob, msk, cs, pid, lp, val, rem_before)
                    return rls[pid].v_cmd, rls[pid].w_cmd

                done = sim.step(_pol_fn, controller_name=variant)
                d_comp = sim.completed_tasks - prev_comp
                prev_comp = sim.completed_tasks
                sh_step = sim.shield_interventions - prev_sh

                for r_id, (ob, msk, cs, pid, lp, val, rem_before) in step_recs.items():
                    rb = sim.robots[r_id]
                    rem_after = sim.remaining_route_dist_m(rb) if rb.phase != "idle" else 0.0
                    ds = clip_val(rem_before - rem_after, -0.5, 0.8)
                    # Clean 5-term reward: progress, time, smoothness, intervention, arrival
                    rew = 1.8 * ds - 0.04 * DT_MICRO - 0.02 * abs(rb.omega) - (0.35 if sh_step > 0 else 0.0) + (3.0 * d_comp / max(1, len(step_recs)))
                    ep_obs.append(ob)
                    ep_masks.append(msk)
                    ep_cs.append(cs)
                    ep_acts.append(pid)
                    ep_lps.append(lp)
                    ep_vals.append(val)
                    ep_rews.append(float(rew))
                if done:
                    break

            if not ep_rews:
                continue
            final_ret = float(sum(ep_rews) / max(1, sim.n_robots))
            # Compute discounted returns & normalized advantages
            advs = np.zeros(len(ep_rews), dtype=np.float32)
            rets = np.zeros(len(ep_rews), dtype=np.float32)
            running = 0.0
            for k in range(len(ep_rews) - 1, -1, -1):
                running = ep_rews[k] + 0.96 * running
                rets[k] = running
                advs[k] = running - ep_vals[k]
            advs = (advs - advs.mean()) / (advs.std() + 1e-6)

            x_b = torch.tensor(np.asarray(ep_obs), dtype=torch.float32)
            m_b = torch.tensor(np.asarray(ep_masks), dtype=torch.float32)
            c_b = torch.tensor(np.asarray(ep_cs), dtype=torch.float32)
            a_b = torch.tensor(np.asarray(ep_acts), dtype=torch.long)
            olp_b = torch.tensor(np.asarray(ep_lps), dtype=torch.float32)
            ret_b = torch.tensor(rets, dtype=torch.float32)
            adv_b = torch.tensor(advs, dtype=torch.float32)

            for _ in range(4):
                logits = net.masked_logits(x_b, m_b)
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(a_b)
                ratio = torch.exp(lp - olp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_b
                pi_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist.entropy().mean()
                v_loss = F.mse_loss(net.value(c_b), ret_b)
                loss = pi_loss + 0.5 * v_loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

    if out_path is None:
        RAMP_CKPT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RAMP_CKPT_DIR / f"{variant.lower()}_s{seed}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    save_dict["mode"] = np.array(mode)
    np.savez_compressed(out_path, **save_dict)

    return RampTrainResult(
        variant=variant,
        mode=mode,
        seed=seed,
        ckpt_path=out_path,
        elapsed_s=round(time.perf_counter() - t0, 2),
        bc_acc=round(bc_acc, 4),
        final_mean_ret=round(final_ret, 3),
    )


def train_all_ramp_variants(*, seeds: tuple[int, ...] = (42, 43, 44), n_iters: int = 12) -> list[dict[str, Any]]:
    specs = (
        ("R2_BC_Only", "full_res", True, True),
        ("R3_Primitive_PPO", "kin_only", True, False),
        ("R4_Spatial_Intent_PPO", "spatial_intent", True, False),
        ("R5_RAMP_PPO", "full_res", True, False),
    )
    out: list[dict[str, Any]] = []
    for sd in seeds:
        for vname, mode, wbc, bco in specs:
            res = train_ramp_variant(vname, mode=mode, seed=sd, n_iters=n_iters, warmstart_bc=wbc, bc_only=bco)
            d = {
                "variant": res.variant,
                "mode": res.mode,
                "seed": res.seed,
                "elapsed_s": res.elapsed_s,
                "bc_acc": res.bc_acc,
                "final_mean_ret": res.final_mean_ret,
                "ckpt": str(res.ckpt_path),
            }
            out.append(d)
            print(f"[Train | {vname:22s} | s={sd}] bc_acc={res.bc_acc:.3f} ret={res.final_mean_ret:.2f} ({res.elapsed_s}s)", flush=True)
    return out


if __name__ == "__main__":
    train_all_ramp_variants()
