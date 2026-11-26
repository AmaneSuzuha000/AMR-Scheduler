"""Route B (Layer 1/2): 2D Continuous Kinematic Avoidance RL + Analytic Safety Shield.

Integrates global Layer 2/3 Rust CPS-A* route plans with a continuous-space
differential-drive/unicycle local controller (v, omega) at dt_micro = 0.25s inside
workshop-scaled corridors (SCOUT Mini envelope: 1.2m x 0.9m -> lateral collision
radius R_rob = 0.45m, safety clearance target >= 0.30m, aisle width 2.2m - 3.0m).

Honesty & Modelling Scope:
  - Simulates 2D continuous unicycle kinematics + dynamic circular obstacles +
    bounded Gaussian range/position jitter inside static corridor walls.
  - This is a physics-kinematic 2D simulation bridge and is NOT real onboard
    Gazebo/ROS 2 LiDAR perception validation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from amr_sched.cvcr.physical_bridge import CorridorObstacleReport, Nav2RobotTelemetry
from amr_sched.network import Network
from experiments.res_guided_avoidance.geometry_and_shield import (
    BrakingInvariantShield,
    CircleState,
    FootprintConfig,
    corridor_wall_clearance,
    obstacle_surface_clearance,
    swept_corridor_collision,
)
from experiments.rgro.dfl_critic import DFLAdvantageEnsemble

ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH = ROOT / "amr_sched" / "checkpoints" / "track_r" / "route_b_kinematic_ppo.npz"
OUT_DIR = ROOT / "out" / "track_r_cpsa"

# Grounded physical parameters matching rail_freight_workshop_v1 SCOUT Mini width (0.9m -> r_lat=0.45m)
OBS_DIM = 20
ACT_DIM = 2
DT_MICRO = 0.25
ROBOT_RADIUS_M = 0.45
HARD_MIN_CLEAR_M = 0.20  # Hard shield intervention margin exceeding single 0.25s chord displacement
SAFE_MARGIN_TARGET_M = 0.30
MAX_V_MPS = 1.35
MAX_OMEGA_RADPS = 1.65
MAX_DV_DT = 1.6
MAX_DW_DT = 3.2


def _wrap_pi(ang: float) -> float:
    return (ang + math.pi) % (2.0 * math.pi) - math.pi


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


@dataclass
class DynamicObstacle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    x_lo: float
    x_hi: float
    y_lo: float
    y_hi: float
    stop_at_y_bound: bool = False

    def step(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.x < self.x_lo:
            self.x = self.x_lo
            self.vx = abs(self.vx)
        elif self.x > self.x_hi:
            self.x = self.x_hi
            self.vx = -abs(self.vx)
        if self.y < self.y_lo:
            self.y = self.y_lo
            self.vy = 0.0 if self.stop_at_y_bound else abs(self.vy)
        elif self.y > self.y_hi:
            self.y = self.y_hi
            self.vy = 0.0 if self.stop_at_y_bound else -abs(self.vy)


@dataclass
class CorridorScenarioSpec:
    name: str
    length_m: float
    half_width_m: float
    obstacles: tuple[DynamicObstacle, ...]
    nominal_v_mps: float = 1.0


def build_corridor_scenario(name: str, seed: int) -> CorridorScenarioSpec:
    """Create reproducible 2D continuous aisle-crossing & dynamic-intruder scenarios."""
    rng = np.random.default_rng(int(seed))
    if name == "Bypass_MainAisle_3.0m":
        hw = 1.50
        length = 18.0
        jitter = float(rng.uniform(-0.22, 0.22))
        s_lat = float(rng.choice([-1.0, 1.0]))
        obs = (
            DynamicObstacle(
                x=6.8 + jitter, y=0.22 * s_lat,
                vx=0.0, vy=0.0, radius=0.35, x_lo=4.0, x_hi=10.0, y_lo=-hw + 0.35, y_hi=hw - 0.35,
            ),
            DynamicObstacle(
                x=12.8, y=-0.30 * s_lat,
                vx=-0.10, vy=0.0, radius=0.34, x_lo=9.0, x_hi=16.5, y_lo=-hw + 0.35, y_hi=hw - 0.35,
            ),
        )
        return CorridorScenarioSpec(name=name, length_m=length, half_width_m=hw, obstacles=obs)

    if name == "Crossing_Narrow_2.2m":
        hw = 1.15
        length = 14.0
        s_dir = float(rng.choice([-1.0, 1.0]))
        vy_cross = float(rng.uniform(0.50, 0.70)) * s_dir
        obs = (
            DynamicObstacle(
                x=6.2 + float(rng.uniform(-0.35, 0.35)),
                y=-1.20 * s_dir,
                vx=0.0, vy=vy_cross, radius=0.34,
                x_lo=3.0, x_hi=10.0, y_lo=-2.20, y_hi=2.20, stop_at_y_bound=True,
            ),
            DynamicObstacle(
                x=10.8, y=0.26 * (-s_dir),
                vx=0.0, vy=0.0, radius=0.32,
                x_lo=7.5, x_hi=12.8, y_lo=-hw + 0.32, y_hi=hw - 0.32,
            ),
        )
        return CorridorScenarioSpec(name=name, length_m=length, half_width_m=hw, obstacles=obs)

    # HeadOn_Yield_2.2m: tight 2.2m doorway / slow approaching pallet tug with offset passing bay
    hw = 1.15
    length = 14.0
    s_lat = float(rng.choice([-1.0, 1.0]))
    obs = (
        DynamicObstacle(
            x=9.8, y=0.28 * s_lat,
            vx=-0.26, vy=0.08 * s_lat,
            radius=0.34, x_lo=2.5, x_hi=12.5, y_lo=-hw + 0.34, y_hi=hw - 0.34,
        ),
    )
    return CorridorScenarioSpec(name="HeadOn_Yield_2.2m", length_m=length, half_width_m=hw, obstacles=obs)


class UnicycleCorridorEnv:
    """2D continuous differential-drive robot navigating a global CPS-A* corridor segment."""

    def __init__(
        self,
        spec: CorridorScenarioSpec,
        *,
        obs_noise_std: float = 0.02,
        seed: int = 0,
        footprint: FootprintConfig = FootprintConfig(mode="disc", radius_m=ROBOT_RADIUS_M),
    ) -> None:
        self.spec = spec
        self.obs_noise_std = float(obs_noise_std)
        self.footprint = footprint
        self.shield = BrakingInvariantShield(
            footprint=footprint,
            hard_min_clear_m=HARD_MIN_CLEAR_M,
            wall_min_clear_m=0.04,
            horizon_steps=14,
            dt=DT_MICRO,
            max_v=MAX_V_MPS,
            max_omega=MAX_OMEGA_RADPS,
            max_dv_dt=MAX_DV_DT,
            max_dw_dt=MAX_DW_DT,
        )
        self.rng = np.random.default_rng(int(seed))
        self.obstacles: list[DynamicObstacle] = []
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.omega = 0.0
        self.t_step = 0
        self.max_steps = int(round((spec.length_m / max(0.3, spec.nominal_v_mps) + 12.0) / DT_MICRO))
        self.reset(spec)

    def reset(self, spec: Optional[CorridorScenarioSpec] = None) -> np.ndarray:
        if spec is not None:
            self.spec = spec
        self.obstacles = [
            DynamicObstacle(
                o.x, o.y, o.vx, o.vy, o.radius, o.x_lo, o.x_hi, o.y_lo, o.y_hi,
                stop_at_y_bound=o.stop_at_y_bound,
            )
            for o in self.spec.obstacles
        ]
        self.x = 0.0
        self.y = float(self.rng.uniform(-0.12, 0.12))
        self.theta = float(self.rng.uniform(-0.08, 0.08))
        self.v = 0.35
        self.omega = 0.0
        self.t_step = 0
        self.max_steps = int(round((self.spec.length_m / max(0.3, self.spec.nominal_v_mps) + 12.0) / DT_MICRO))
        return self.observe()

    def _surface_clearance(self, rx: float, ry: float, rth: Optional[float] = None) -> tuple[float, float, list[tuple[float, DynamicObstacle]]]:
        """Return (min_obstacle_surface_clearance, wall_surface_clearance, sorted_obstacles)."""
        th = self.theta if rth is None else rth
        wall_clear = corridor_wall_clearance(rx, ry, th, self.spec.half_width_m, self.footprint)
        items: list[tuple[float, DynamicObstacle]] = []
        for o in self.obstacles:
            d_surf = obstacle_surface_clearance(
                rx, ry, th, (CircleState(o.x, o.y, o.vx, o.vy, o.radius),), self.footprint
            )
            items.append((d_surf, o))
        items.sort(key=lambda t: t[0])
        min_obs = items[0][0] if items else 9.0
        return min_obs, wall_clear, items

    def observe(self) -> np.ndarray:
        """Build normalized 20-D Markovian kinematic + corridor-frame obstacle feature vector."""
        hw = max(0.8, float(self.spec.half_width_m))
        rem_x = max(0.0, self.spec.length_m - self.x)
        look_x = min(self.spec.length_m, self.x + 2.5)
        hdg_err = _wrap_pi(math.atan2(-self.y, max(0.4, look_x - self.x)) - self.theta)

        _, _, obs_sorted = self._surface_clearance(self.x, self.y)
        obs_forward = sorted(obs_sorted, key=lambda pair: (0 if (pair[1].x - self.x) >= -0.35 else 1, pair[0]))
        feats = np.zeros(OBS_DIM, dtype=np.float32)
        feats[0] = _clip(self.y / hw, -1.5, 1.5)
        feats[1] = math.sin(self.theta)
        feats[2] = math.cos(self.theta)
        feats[3] = math.sin(hdg_err)
        feats[4] = _clip(rem_x / 18.0, 0.0, 1.5)
        feats[5] = _clip(self.v / MAX_V_MPS, 0.0, 1.2)
        feats[6] = _clip(self.omega / MAX_OMEGA_RADPS, -1.2, 1.2)
        feats[7] = _clip(hw / 1.5, 0.5, 1.5)

        noise = self.rng.normal(0.0, self.obs_noise_std, size=8) if self.obs_noise_std > 0 else np.zeros(8)
        ct, st = math.cos(-self.theta), math.sin(-self.theta)
        if obs_forward:
            d_surf, o = obs_forward[0]
            dx_w, dy_w = o.x - self.x, o.y - self.y
            lx = ct * dx_w - st * dy_w + float(noise[0])
            ly = st * dx_w + ct * dy_w + float(noise[1])
            lvx = ct * o.vx - st * o.vy + float(noise[2])
            lvy = st * o.vx + ct * o.vy + float(noise[3])
            feats[8] = _clip(lx / 8.0, -1.5, 1.5)
            feats[9] = _clip(ly / hw, -2.0, 2.0)
            feats[10] = _clip(lvx / MAX_V_MPS, -1.5, 1.5)
            feats[11] = _clip(lvy / MAX_V_MPS, -1.5, 1.5)
            feats[12] = _clip(max(-0.5, d_surf) / 4.0, -0.5, 1.5)
            # Corridor-relative obstacle lateral center (+ wall-gap asymmetry) derived from wall & obstacle distance
            feats[13] = _clip((o.y + float(noise[4])) / hw, -2.0, 2.0)
        else:
            feats[8], feats[9], feats[10], feats[11], feats[12], feats[13] = 1.5, 0.0, 0.0, 0.0, 1.5, 0.0

        if len(obs_forward) >= 2:
            d_surf2, o2 = obs_forward[1]
            dx_w2, dy_w2 = o2.x - self.x, o2.y - self.y
            lx2 = ct * dx_w2 - st * dy_w2 + float(noise[5])
            ly2 = st * dx_w2 + ct * dy_w2 + float(noise[6])
            feats[14] = _clip(lx2 / 8.0, -1.5, 1.5)
            feats[15] = _clip(ly2 / hw, -2.0, 2.0)
            feats[16] = _clip(max(-0.5, d_surf2) / 4.0, -0.5, 1.5)
        else:
            feats[14], feats[15], feats[16] = 1.5, 0.0, 1.5

        # 3 forward-fan ray clearance probes (-25 deg, 0 deg, +25 deg) capped at 6.0m
        for r_i, fan in enumerate((-0.44, 0.0, 0.44)):
            ang = self.theta + fan
            ux, uy = math.cos(ang), math.sin(ang)
            d_hit = 6.0
            if abs(uy) > 1e-4:
                d_wall = ((hw - abs(self.y)) if (uy * self.y >= 0.0) else (hw + abs(self.y))) / abs(uy)
                d_hit = min(d_hit, max(0.0, d_wall - ROBOT_RADIUS_M))
            for _, o in obs_sorted:
                proj = (o.x - self.x) * ux + (o.y - self.y) * uy
                if proj <= -0.2 or proj > 6.5:
                    continue
                perp = abs((o.x - self.x) * (-uy) + (o.y - self.y) * ux)
                rad = ROBOT_RADIUS_M + o.radius + 0.05
                if perp < rad:
                    hit_s = max(0.0, proj - math.sqrt(max(0.0, rad * rad - perp * perp)))
                    d_hit = min(d_hit, hit_s)
            feats[17 + r_i] = _clip(d_hit / 6.0, 0.0, 1.0)
        return feats

    def apply_safety_shield(self, v_cmd: float, omega_cmd: float) -> tuple[float, float, bool]:
        """Unified-geometry inductive braking-invariant safety barrier filter."""
        hw = float(self.spec.half_width_m)
        center_w = _clip(-1.4 * self.y - 0.6 * self.theta, -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS)
        obs_c = [CircleState(o.x, o.y, o.vx, o.vy, o.radius) for o in self.obstacles]
        out = self.shield.filter(
            x=self.x,
            y=self.y,
            theta=self.theta,
            v=self.v,
            omega=self.omega,
            v_cmd=v_cmd,
            omega_cmd=omega_cmd,
            obstacles=obs_c,
            wall_fn=lambda px, py, pth: corridor_wall_clearance(px, py, pth, hw, self.footprint),
            center_w=center_w,
        )
        self._last_shield_infeasible = bool(out.infeasible)
        return out.v_cmd, out.omega_cmd, out.intervened

    def step(
        self,
        v_cmd: float,
        omega_cmd: float,
        *,
        use_shield: bool = True,
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        shielded = False
        self._last_shield_infeasible = False
        if use_shield:
            v_cmd, omega_cmd, shielded = self.apply_safety_shield(v_cmd, omega_cmd)

        dv = _clip(v_cmd - self.v, -MAX_DV_DT * DT_MICRO, MAX_DV_DT * DT_MICRO)
        dw = _clip(omega_cmd - self.omega, -MAX_DW_DT * DT_MICRO, MAX_DW_DT * DT_MICRO)
        self.v = _clip(self.v + dv, 0.0, MAX_V_MPS)
        self.omega = _clip(self.omega + dw, -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS)

        prev_x, prev_y, prev_th = self.x, self.y, self.theta
        obs_start = [CircleState(o.x, o.y, o.vx, o.vy, o.radius) for o in self.obstacles]
        self.x += self.v * math.cos(self.theta) * DT_MICRO
        self.y += self.v * math.sin(self.theta) * DT_MICRO
        self.theta = _wrap_pi(self.theta + self.omega * DT_MICRO)
        for o in self.obstacles:
            o.step(DT_MICRO)
        obs_end = [CircleState(o.x, o.y, o.vx, o.vy, o.radius) for o in self.obstacles]
        self.t_step += 1

        collided, min_obs, wall_clear = swept_corridor_collision(
            prev_x, prev_y, prev_th, self.x, self.y, self.theta,
            self.spec.half_width_m, obs_start, obs_end, self.footprint,
        )
        reached = self.x >= self.spec.length_m
        timeout = self.t_step >= self.max_steps
        done = reached or timeout

        # Shaped PPO reward: progress + centered smoothness + proactive margin + collision guard
        prog = self.x - prev_x
        r = (
            4.0 * prog
            - 0.08 * (self.y / max(1.0, self.spec.half_width_m)) ** 2
            - 0.03 * (self.omega ** 2)
            - 0.03
        )
        if min_obs < 0.70:
            r -= 1.2 * (0.70 - max(0.0, min_obs))
        if shielded:
            r -= 0.45
        if collided:
            r -= 28.0
        if reached:
            r += 6.0

        info = {
            "reached": reached,
            "collided": collided,
            "shielded": shielded,
            "shield_infeasible": bool(self._last_shield_infeasible),
            "min_obs_clear_m": float(min_obs),
            "wall_clear_m": float(wall_clear),
            "v": float(self.v),
        }
        return self.observe(), float(r), done, info


class _PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 96) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.tensor([-1.25, -1.25], dtype=torch.float32))

    def forward_mean(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(x))

    def forward_raw_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Unconstrained pre-tanh Gaussian mean (atanh-consistent with squashed actions)."""
        return torch.clamp(self.actor(x), -4.0, 4.0)

    @staticmethod
    def squashed_log_prob(dist: torch.distributions.Normal, u_raw: torch.Tensor) -> torch.Tensor:
        """Exact tanh-squashed Gaussian log-probability with Jacobian determinant correction."""
        a = torch.tanh(u_raw)
        return (dist.log_prob(u_raw) - torch.log(torch.clamp(1.0 - a * a, min=1e-6))).sum(-1)


class NumpyRouteBPPO:
    """Zero-PyTorch pure-NumPy inference runtime for Route B continuous kinematic PPO."""

    def __init__(self, weights: dict[str, np.ndarray], metadata: Optional[dict[str, Any]] = None) -> None:
        self.w = weights
        self.metadata = dict(metadata or {})

    @classmethod
    def load_npz(cls, path: Path = CKPT_PATH) -> "NumpyRouteBPPO":
        raw = np.load(str(path), allow_pickle=False)
        weights = {k: raw[k].astype(np.float32) for k in raw.files if k != "__meta_json__"}
        meta: dict[str, Any] = {}
        if "__meta_json__" in raw.files:
            meta = json.loads(str(raw["__meta_json__"].item()))
        return cls(weights, meta)

    def act_raw(self, obs: np.ndarray) -> tuple[float, float]:
        h = np.tanh(obs @ self.w["actor.0.weight"].T + self.w["actor.0.bias"])
        h = np.tanh(h @ self.w["actor.2.weight"].T + self.w["actor.2.bias"])
        mu = np.tanh(h @ self.w["actor.4.weight"].T + self.w["actor.4.bias"])
        # Map [-1, 1]^2 -> v_cmd in [0.10, MAX_V_MPS], omega_cmd in [-MAX_OMEGA, +MAX_OMEGA]
        v_cmd = (0.5 * (float(mu[0]) + 1.0)) * (MAX_V_MPS - 0.08) + 0.08
        w_cmd = float(mu[1]) * MAX_OMEGA_RADPS
        return v_cmd, w_cmd


def _rigid_follower_cmd(env: UnicycleCorridorEnv) -> tuple[float, float]:
    """Pure pursuit centerline tracker (no obstacle perception)."""
    look_x = min(env.spec.length_m, env.x + 2.2)
    hdg = _wrap_pi(math.atan2(-env.y, max(0.5, look_x - env.x)) - env.theta)
    return env.spec.nominal_v_mps, _clip(2.0 * hdg, -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS)


def train_route_b_ppo(
    out_npz: Path = CKPT_PATH,
    *,
    n_iters: int = 24,
    rollout_steps: int = 1024,
    seed: int = 42,
    warmstart_bc: bool = True,
    bc_only: bool = False,
) -> dict[str, Any]:
    """Train continuous 2D unicycle avoidance via Squashed-Gaussian PPO and export pure-NumPy checkpoint."""
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    model = _PPOActorCritic()
    t0 = time.perf_counter()
    scenarios = ("Bypass_MainAisle_3.0m", "Crossing_Narrow_2.2m", "HeadOn_Yield_2.2m")

    if warmstart_bc or bc_only:
        bc_obs: list[np.ndarray] = []
        bc_act: list[np.ndarray] = []
        for k in range(24):
            spec = build_corridor_scenario(scenarios[k % len(scenarios)], seed=100 + k)
            env = UnicycleCorridorEnv(spec, obs_noise_std=0.02, seed=100 + k)
            ob = env.reset()
            for _ in range(env.max_steps):
                vc, wc = _rigid_follower_cmd(env)
                _, _, obs_sorted = env._surface_clearance(env.x, env.y)
                obs_fwd = sorted(obs_sorted, key=lambda pair: (0 if (pair[1].x - env.x) >= -0.35 else 1, pair[0]))
                if obs_fwd:
                    d_surf_o, o = obs_fwd[0]
                    dx = o.x - env.x
                    if -0.25 <= dx <= 6.8:
                        hw_safe = env.spec.half_width_m - ROBOT_RADIUS_M - 0.05
                        gap_pos = hw_safe - (o.y + o.radius)
                        gap_neg = (o.y - o.radius) - (-hw_safe)
                        y_plus = _clip(o.y + o.radius + ROBOT_RADIUS_M + 0.12, -hw_safe, hw_safe)
                        y_minus = _clip(o.y - o.radius - ROBOT_RADIUS_M - 0.12, -hw_safe, hw_safe)
                        if abs(o.vy) > 0.25 and abs(o.y) < 1.10 and dx < 3.6:
                            vc = 0.08
                            target_y = 0.0
                        else:
                            target_y = y_plus if gap_pos >= gap_neg else y_minus
                        look_dx = max(1.2, min(1.8, dx + 0.6))
                        wc = _clip(3.2 * _wrap_pi(math.atan2(target_y - env.y, look_dx) - env.theta), -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS)
                        if d_surf_o < 0.70 and abs(o.vx) < 0.05 and abs(env.y - target_y) > 0.18:
                            vc = 0.65
                        elif d_surf_o < 0.50 and abs(o.vy) < 0.20:
                            vc = min(vc, 0.45)
                mu0 = (vc - 0.08) / (MAX_V_MPS - 0.08) * 2.0 - 1.0
                mu1 = wc / MAX_OMEGA_RADPS
                bc_obs.append(ob)
                bc_act.append(np.array([_clip(mu0, -0.95, 0.95), _clip(mu1, -0.95, 0.95)], dtype=np.float32))
                ob, _, done, _ = env.step(vc, wc, use_shield=True)
                if done:
                    break

        X_bc = torch.from_numpy(np.asarray(bc_obs, dtype=np.float32))
        Y_bc = torch.from_numpy(np.asarray(bc_act, dtype=np.float32))
        bc_weights = torch.where(torch.abs(Y_bc[:, 1]) > 0.10, 6.0, 1.0)
        opt_bc = torch.optim.Adam(model.parameters(), lr=4e-3)
        for ep_bc in range(520):
            opt_bc.zero_grad()
            pred = model.forward_mean(X_bc)
            loss_bc = torch.mean(bc_weights[:, None] * (pred - Y_bc) ** 2)
            loss_bc.backward()
            opt_bc.step()
            if ep_bc in (260, 400):
                for g in opt_bc.param_groups:
                    g["lr"] *= 0.45

    def _eval_det_score(cur_model: _PPOActorCritic) -> float:
        cur_w = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in cur_model.state_dict().items()}
        ag = NumpyRouteBPPO(cur_w)
        tot_sc = 0.0
        for sc_n in scenarios:
            for sd_v in (5, 6):
                e_v = UnicycleCorridorEnv(build_corridor_scenario(sc_n, sd_v), obs_noise_std=0.0, seed=sd_v)
                ob_v = e_v.reset()
                ep_c = 0
                reached_v = False
                for _ in range(e_v.max_steps):
                    vc, wc = ag.act_raw(ob_v)
                    ob_v, _, done_v, info_v = e_v.step(vc, wc, use_shield=True)
                    ep_c += int(info_v["collided"])
                    if info_v["reached"]:
                        reached_v = True
                        break
                    if done_v:
                        break
                tot_sc += (120.0 if (reached_v and ep_c == 0) else 0.0) - 40.0 * ep_c - 0.25 * e_v.t_step
        return tot_sc

    best_score = _eval_det_score(model)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    opt = torch.optim.Adam(model.parameters(), lr=8e-5 if warmstart_bc else 3e-4)
    gamma, gae_lam = 0.99, 0.95
    clip_eps = 0.18
    mean_ret_hist: list[float] = []
    max_lp_consistency_err = 0.0

    for it in range(0 if bc_only else int(n_iters)):
        obs_buf: list[np.ndarray] = []
        u_raw_buf: list[np.ndarray] = []
        logp_buf: list[float] = []
        rew_buf: list[float] = []
        val_buf: list[float] = []
        done_buf: list[float] = []
        ep_rets: list[float] = []

        sc_name = scenarios[it % len(scenarios)]
        env = UnicycleCorridorEnv(build_corridor_scenario(sc_name, seed=1000 + it), obs_noise_std=0.02, seed=1000 + it)
        ob = env.reset()
        ep_ret = 0.0

        for step_idx in range(int(rollout_steps)):
            with torch.no_grad():
                ob_t = torch.from_numpy(ob[None, :])
                mu_raw = model.forward_raw_mean(ob_t)
                std = torch.exp(torch.clamp(model.log_std, -2.2, -0.7))
                dist = torch.distributions.Normal(mu_raw, std)
                u_raw = dist.sample()
                a_squashed = torch.tanh(u_raw)
                logp = float(_PPOActorCritic.squashed_log_prob(dist, u_raw).item())
                val = float(model.critic(ob_t).squeeze().item())
            a_np = a_squashed[0].numpy()
            v_cmd = (0.5 * (float(a_np[0]) + 1.0)) * (MAX_V_MPS - 0.08) + 0.08
            w_cmd = float(a_np[1]) * MAX_OMEGA_RADPS
            nxt, rew, done, _ = env.step(v_cmd, w_cmd, use_shield=True)

            obs_buf.append(ob)
            u_raw_buf.append(u_raw[0].numpy().astype(np.float32))
            logp_buf.append(logp)
            rew_buf.append(rew)
            val_buf.append(val)
            done_buf.append(1.0 if done else 0.0)
            ep_ret += rew
            ob = nxt
            if done:
                ep_rets.append(ep_ret)
                ep_ret = 0.0
                sc_next = scenarios[(it + step_idx + 1) % len(scenarios)]
                ob = env.reset(build_corridor_scenario(sc_next, seed=2000 + it * 97 + step_idx))

        with torch.no_grad():
            last_v = float(model.critic(torch.from_numpy(ob[None, :])).squeeze().item()) if not done_buf[-1] else 0.0

        T = len(rew_buf)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nxt_v = last_v if t == T - 1 else val_buf[t + 1]
            non_term = 1.0 - done_buf[t]
            delta = rew_buf[t] + gamma * nxt_v * non_term - val_buf[t]
            gae = delta + gamma * gae_lam * non_term * gae
            adv[t] = gae
        ret = adv + np.asarray(val_buf, dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        X_tr = torch.from_numpy(np.asarray(obs_buf, dtype=np.float32))
        U_tr = torch.from_numpy(np.asarray(u_raw_buf, dtype=np.float32))
        LP_old = torch.from_numpy(np.asarray(logp_buf, dtype=np.float32))
        ADV_t = torch.from_numpy(adv)
        RET_t = torch.from_numpy(ret)

        with torch.no_grad():
            mu_chk = model.forward_raw_mean(X_tr)
            std_chk = torch.exp(torch.clamp(model.log_std, -2.2, -0.7))
            lp_chk = _PPOActorCritic.squashed_log_prob(torch.distributions.Normal(mu_chk, std_chk), U_tr)
            max_lp_consistency_err = max(max_lp_consistency_err, float((lp_chk - LP_old).abs().max().item()))

        for _ in range(3):
            perm = rng.permutation(T)
            for s0 in range(0, T, 256):
                mb = perm[s0 : s0 + 256]
                mu_raw = model.forward_raw_mean(X_tr[mb])
                std = torch.exp(torch.clamp(model.log_std, -2.2, -0.7))
                dist = torch.distributions.Normal(mu_raw, std)
                lp = _PPOActorCritic.squashed_log_prob(dist, U_tr[mb])
                approx_kl = float((LP_old[mb] - lp).mean().item())
                if approx_kl > 0.03:
                    break
                ratio = torch.exp(lp - LP_old[mb])
                surr1 = ratio * ADV_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * ADV_t[mb]
                p_loss = -torch.min(surr1, surr2).mean()
                v_loss = 0.5 * ((model.critic(X_tr[mb]).squeeze(-1) - RET_t[mb]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = p_loss + 0.5 * v_loss - 0.005 * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
                opt.step()

        if (it + 1) % 4 == 0:
            sc_val = _eval_det_score(model)
            if sc_val >= best_score:
                best_score = sc_val
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if ep_rets:
            mean_ret_hist.append(float(np.mean(ep_rets)))

    model.load_state_dict(best_state, strict=True)
    final_eval = _eval_det_score(model)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "domain": "route_b_2d_kinematic_corridor",
        "robot_radius_m": ROBOT_RADIUS_M,
        "dt_micro_s": DT_MICRO,
        "n_iters": 0 if bc_only else int(n_iters),
        "warmstart_bc": bool(warmstart_bc),
        "bc_only": bool(bc_only),
        "max_lp_consistency_err": round(max_lp_consistency_err, 8),
        "mean_return_init": round(mean_ret_hist[0], 2) if mean_ret_hist else 0.0,
        "mean_return_final": round(mean_ret_hist[-1], 2) if mean_ret_hist else 0.0,
        "best_val_score": round(final_eval, 2),
        "train_s": round(time.perf_counter() - t0, 2),
    }
    np.savez(
        str(out_npz),
        **{k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()},
        __meta_json__=np.asarray(json.dumps(meta)),
    )
    _ = NumpyRouteBPPO.load_npz(out_npz)
    return meta


@dataclass(frozen=True)
class CorridorBenchRow:
    scenario: str
    controller: str
    success_rate: float
    collisions_total: int
    mean_min_clear_m: float
    mean_shield_stops: float
    mean_transit_s: float
    mean_lat_err_m: float


def run_isolated_corridor_benchmark(
    ppo_agent: NumpyRouteBPPO,
    *,
    seeds: tuple[int, ...] = (11, 12, 13, 14, 15, 16, 17, 18),
) -> list[CorridorBenchRow]:
    """Compare Rigid Follower vs Braking Shield vs Continuous PPO+Shield across 2D scenarios."""
    scenarios = ("Bypass_MainAisle_3.0m", "Crossing_Narrow_2.2m", "HeadOn_Yield_2.2m")
    controllers = ("Rigid_Follower_NoShield", "Braking_Shield_Only", "PPO_RL_SafetyShield")
    rows: list[CorridorBenchRow] = []

    for sc_name in scenarios:
        for ctrl in controllers:
            succ = 0
            cols = 0
            clears: list[float] = []
            sh_cnts: list[int] = []
            times: list[float] = []
            lats: list[float] = []
            for sd in seeds:
                env = UnicycleCorridorEnv(build_corridor_scenario(sc_name, sd), obs_noise_std=0.02, seed=sd)
                ob = env.reset()
                min_c = 9.0
                sh_hits = 0
                ep_col = 0
                ep_lat: list[float] = []
                reached = False
                for _ in range(env.max_steps):
                    if ctrl == "PPO_RL_SafetyShield":
                        vc, wc = ppo_agent.act_raw(ob)
                        use_sh = True
                    elif ctrl == "Braking_Shield_Only":
                        vc, wc = _rigid_follower_cmd(env)
                        use_sh = True
                    else:
                        vc, wc = _rigid_follower_cmd(env)
                        use_sh = False
                    ob, _, done, info = env.step(vc, wc, use_shield=use_sh)
                    min_c = min(min_c, float(info["min_obs_clear_m"]))
                    ep_lat.append(abs(env.y))
                    if info["shielded"]:
                        sh_hits += 1
                    if info["collided"]:
                        ep_col += 1
                    if info["reached"]:
                        reached = True
                        break
                    if done:
                        break
                succ += int(reached and ep_col == 0)
                cols += ep_col
                clears.append(max(-0.5, min_c))
                sh_cnts.append(sh_hits)
                times.append(env.t_step * DT_MICRO)
                lats.append(float(np.mean(ep_lat)) if ep_lat else 0.0)
            n = float(len(seeds))
            rows.append(
                CorridorBenchRow(
                    scenario=sc_name,
                    controller=ctrl,
                    success_rate=round(succ / n, 4),
                    collisions_total=int(cols),
                    mean_min_clear_m=round(float(np.mean(clears)), 3),
                    mean_shield_stops=round(float(np.mean(sh_cnts)), 2),
                    mean_transit_s=round(float(np.mean(times)), 2),
                    mean_lat_err_m=round(float(np.mean(lats)), 3),
                )
            )
    return rows


def make_route_b_kinematic_hook(
    *,
    mode: str = "ppo_shield",
    ppo_agent: Optional[NumpyRouteBPPO] = None,
    seed: int = 42,
    corridor_disturb_prob: float = 0.25,
) -> tuple[Callable[..., dict[str, Any]], dict[str, Any]]:
    """Create a closed-loop `route_b_telemetry_hook` binding micro-level 2D avoidance to global Rust CPS-A*."""
    rng = np.random.default_rng(int(seed))
    stats: dict[str, Any] = {
        "mode": mode,
        "micro_rollouts": 0,
        "physical_collisions": 0,
        "shield_interventions": 0,
        "min_surface_clear_m": 9.0,
        "eta_drift_events": 0,
        "corridor_reports": 0,
    }
    agent = ppo_agent or (NumpyRouteBPPO.load_npz() if (mode == "ppo_shield" and CKPT_PATH.exists()) else None)
    edge_obs_cache: dict[tuple[int, tuple[str, str]], float] = {}

    seen_edge_starts: set[tuple[str, str, str, int]] = set()

    def _hook(
        tick: int,
        network: Network,
        pos: dict[str, str],
        in_edge: dict[str, Optional[tuple[str, str, int]]],
        phase: dict[str, str],
    ) -> dict[str, Any]:
        telemetry: dict[str, Nav2RobotTelemetry] = {}
        obstacles: list[CorridorObstacleReport] = []

        # Periodically inject continuous intruder inside an active busy corridor for physical testing
        active_edges = [
            (aid, ie)
            for aid, ie in sorted(in_edge.items())
            if ie is not None and phase.get(aid) in ("to_pickup", "to_dropoff")
        ]
        if tick > 0 and active_edges and (tick % 6 == 0):
            for aid, (u, v, rem_steps) in active_edges:
                edge_key = (aid, u, v, int(tick) - int(rem_steps))
                if edge_key in seen_edge_starts or not network.has_edge(u, v):
                    continue
                if float(rng.random()) >= corridor_disturb_prob:
                    continue
                e = network.edge(u, v)
                if e.length < 5.5:
                    continue
                seen_edge_starts.add(edge_key)
                planned_ticks = max(1, int(math.ceil(e.length / 5.0)))
                half_w = 1.50 if e.capacity >= 2 else 1.25
                sc = CorridorScenarioSpec(
                    name=f"Edge_{u}_{v}",
                    length_m=float(e.length),
                    half_width_m=half_w,
                    obstacles=(
                        DynamicObstacle(
                            x=min(e.length - 1.5, max(3.6, 0.56 * e.length)),
                            y=float(rng.choice([-0.24, 0.24])),
                            vx=0.0,
                            vy=0.0,
                            radius=0.33,
                            x_lo=2.0, x_hi=max(2.5, e.length - 1.0),
                            y_lo=-half_w + 0.34, y_hi=half_w - 0.34,
                        ),
                    ),
                )
                env = UnicycleCorridorEnv(sc, obs_noise_std=0.02, seed=int(seed) + tick * 17 + abs(hash(aid)) % 97)
                ob = env.reset()
                sh_hits = 0
                ep_col = 0
                min_c = 9.0
                for _ in range(env.max_steps):
                    if mode == "ppo_shield" and agent is not None:
                        vc, wc = agent.act_raw(ob)
                        use_sh = True
                    elif mode == "shield_only":
                        vc, wc = _rigid_follower_cmd(env)
                        use_sh = True
                    else:
                        vc, wc = _rigid_follower_cmd(env)
                        use_sh = False
                    ob, _, done, info = env.step(vc, wc, use_shield=use_sh)
                    min_c = min(min_c, float(info["min_obs_clear_m"]))
                    if info["shielded"]:
                        sh_hits += 1
                    if info["collided"]:
                        ep_col += 1
                    if info["reached"] or done:
                        break

                realized_ticks = int(math.ceil((env.t_step * DT_MICRO) / 5.0))
                drift_ticks = max(0, realized_ticks - planned_ticks)
                stats["micro_rollouts"] += 1
                stats["physical_collisions"] += ep_col
                stats["shield_interventions"] += sh_hits
                stats["min_surface_clear_m"] = round(min(float(stats["min_surface_clear_m"]), max(-0.5, min_c)), 3)
                if drift_ticks > 0:
                    stats["eta_drift_events"] += 1
                    nu = network.node(u)
                    telemetry[aid] = Nav2RobotTelemetry(
                        robot_id=aid,
                        pose_xy_yaw=(nu.x, nu.y, 0.0),
                        speed_scale=max(0.25, float(planned_ticks) / float(max(1, realized_ticks))),
                        eta_drift_steps=min(4, drift_ticks),
                        local_planner_state="shield_avoid" if sh_hits > 0 else "normal",
                        handling_delay_steps=0,
                        fault_reason="route_b_local_avoidance",
                    )
                if drift_ticks >= 2 and half_w <= 1.3:
                    obstacles.append(
                        CorridorObstacleReport(
                            edge=(u, v),
                            status="partially_blocked",
                            estimated_clear_horizon_steps=min(6, drift_ticks + 2),
                            source="route_b_kinematic_monitor",
                            reason=f"clear={min_c:.2f}m_drift={drift_ticks}",
                        )
                    )
                    stats["corridor_reports"] += 1
                break

        return {"telemetry": telemetry, "obstacles": obstacles}

    return _hook, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate Route B 2D Kinematic RL + Safety Shield.")
    parser.add_argument("--retrain", action="store_true", help="Train/retrain Route B PPO checkpoint.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.retrain or not CKPT_PATH.exists():
        print("=== Training Route B Continuous 2D Unicycle PPO Controller ===", flush=True)
        train_meta = train_route_b_ppo(CKPT_PATH, n_iters=44, seed=42)
        print(f"  [Train] {train_meta}", flush=True)
    else:
        train_meta = NumpyRouteBPPO.load_npz(CKPT_PATH).metadata

    ppo_agent = NumpyRouteBPPO.load_npz(CKPT_PATH)
    print("\n=== [Suite B1] Isolated 2D Corridor Avoidance Benchmark (3 Scenarios x 8 Seeds) ===", flush=True)
    iso_rows = run_isolated_corridor_benchmark(ppo_agent)
    for r in iso_rows:
        print(
            f"  [{r.scenario:22s} | {r.controller:23s}] Succ={r.success_rate*100:5.1f}% Col={r.collisions_total:2d} "
            f"MinClr={r.mean_min_clear_m:5.3f}m ShStops={r.mean_shield_stops:4.2f} Transit={r.mean_transit_s:5.2f}s",
            flush=True,
        )

    from experiments.track_r_cpsa.workshop_bridge_eval import _load_wu26_net, run_workshop_episode

    wu_ckpt = ROOT / "amr_sched" / "checkpoints" / "track_r" / "ejor26_wu_gru_workshop.npz"
    dfl_ckpt = ROOT / "amr_sched" / "checkpoints" / "track_r" / "mh_dfl_critic_workshop.npz"
    wu_net = _load_wu26_net(wu_ckpt) if wu_ckpt.exists() else _load_wu26_net()
    dfl_critic = DFLAdvantageEnsemble.load_npz(dfl_ckpt) if dfl_ckpt.exists() else None

    print("\n=== [Suite B2] Closed-Loop Global Rust CPS-A* + Route B Hook (`workshop_v1` & `demo17`) ===", flush=True)
    closed_rows: list[dict[str, Any]] = []
    for lay, nt, na, dp in (("workshop_v1", 40, 6, 0.12), ("demo17", 30, 4, 0.10)):
        for mode, label in (
            ("rigid_no_shield", "Global_CPS + Rigid_Follower"),
            ("shield_only", "Global_CPS + Analytic_Shield"),
            ("ppo_shield", "Global_CPS + PPO_RL_Shield"),
        ):
            done_l, ms_l, col_l, sh_l, clr_l, rep_l = [], [], 0, 0, [], 0
            for sd in (42, 43):
                hook, st = make_route_b_kinematic_hook(mode=mode, ppo_agent=ppo_agent, seed=sd, corridor_disturb_prob=0.30)
                res = run_workshop_episode(
                    layout=lay, aisle_capacity=1, n_tasks=nt, n_agvs=na, seed=sd,
                    dispatcher="Wu26_SemiCircle_GRU", router="CPS_CVCR_DFL", delay_prob=dp,
                    wu_net=wu_net, dfl_critic=dfl_critic, use_rust_e2e=True,
                    route_b_telemetry_hook=hook,
                )
                done_l.append(res.completed_tasks)
                ms_l.append(res.makespan)
                col_l += int(st["physical_collisions"])
                sh_l += int(st["shield_interventions"])
                clr_l.append(float(st["min_surface_clear_m"]))
                rep_l += int(st["corridor_reports"])
            rec = {
                "layout": lay,
                "controller": label,
                "n_tasks": nt,
                "completed_avg": round(float(np.mean(done_l)), 1),
                "makespan_avg": round(float(np.mean(ms_l)), 1),
                "phys_collisions": col_l,
                "min_clear_m": round(float(min(clr_l)), 3),
                "shield_interventions": sh_l,
                "corridor_escalations": rep_l,
            }
            closed_rows.append(rec)
            print(
                f"  [{lay:11s} | {label:30s}] Done={rec['completed_avg']:4.1f}/{nt} MS={rec['makespan_avg']:5.1f} "
                f"PhysCol={rec['phys_collisions']:2d} MinClr={rec['min_clear_m']:5.3f}m Shield={rec['shield_interventions']:3d}",
                flush=True,
            )

    payload = {
        "train_meta": train_meta,
        "isolated_corridors": [asdict(r) for r in iso_rows],
        "closed_loop_workshop": closed_rows,
        "scope_disclaimer": (
            "2D continuous unicycle kinematic + bounded Gaussian obstacle-geometry simulation; "
            "not real onboard SCOUT Mini Gazebo/ROS2 LiDAR point-cloud perception."
        ),
    }
    (OUT_DIR / "route_b_kinematic_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Route B 实验报告：全局 Rust CPS-A* + 2D 连续运动学局部避障 RL 与安全护盾",
        "",
        "> **严谨口径声明**：本模块在连续二维平面与 `rail_freight_workshop_v1` 实景走廊宽度（`2.2–3.0 m`，SCOUT Mini 横向等效避障半径 `r_rob = 0.45 m`，对应车体 `1.2 m × 0.9 m`）下，基于二维差速/单轮车连续运动学 `(x, y, yaw, v, omega)` 与解析动态障碍物几何（含观测高斯扰动）进行闭环联合验证；**尚未接入真实 Gazebo 激光点云传感或实机 SCOUT Mini 硬件测试**。",
        "",
        f"- 策略检查点：[`route_b_kinematic_ppo.npz`](file:///{str(CKPT_PATH).replace(chr(92), '/')})（训练元信息：`{train_meta}`）",
        "",
        "## 一、连续走廊独立避障对比（3 场景 × 8 种子）",
        "",
        "| 走廊场景 | 局部控制策略 | 无碰撞完成率 | 累计物理碰撞 | 平均最小表面净距 (m) | 安全护盾介入次 | 平均通过耗时 (s) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in iso_rows:
        md.append(
            f"| `{r.scenario}` | `{r.controller}` | {r.success_rate*100:.1f}% | {r.collisions_total} | "
            f"{r.mean_min_clear_m:.3f} | {r.mean_shield_stops:.2f} | {r.mean_transit_s:.2f} |"
        )
    md.extend([
        "",
        "## 二、上下层闭环联动：全局 Rust `CPS-A*` + `route_b_telemetry_hook`（2 种子均值）",
        "",
        "| 车间场景 | 分层控制组合 | 完工数 | Makespan | 连续物理碰撞 | 最小安全净距 (m) | 护盾紧急减速 | L1→L2 走廊上报 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for rec in closed_rows:
        md.append(
            f"| `{rec['layout']}` | `{rec['controller']}` | {rec['completed_avg']:.1f}/{rec['n_tasks']} | {rec['makespan_avg']:.1f} | "
            f"{rec['phys_collisions']} | {rec['min_clear_m']:.3f} | {rec['shield_interventions']} | {rec['corridor_escalations']} |"
        )
    (OUT_DIR / "route_b_kinematic_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"=== Wrote {OUT_DIR / 'route_b_kinematic_report.md'} ===", flush=True)


if __name__ == "__main__":
    main()
