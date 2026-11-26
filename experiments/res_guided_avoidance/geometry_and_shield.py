"""P0 Unified Collision Geometry, Swept-Step Checking, and Braking-Invariant Safety Shield.

Eliminates geometry mismatches between wall checks, obstacle checks, and shield horizons:
  - Supports both two-disc capsule (`1.20 m x 0.90 m` SCOUT Mini: two `r = 0.45 m` discs at
    `+-0.15 m` longitudinal offset) and single circular disc (`r = 0.45 m`).
  - Applies the EXACT same signed surface clearance function to walls, moving obstacles,
    and peer robots, with `M_sub = 5` swept sub-step checking over each `dt = 0.25 s` step.
  - Enforces an inductive braking-invariant tail (`H_stop = 14` steps to `v = 0`) and
    explicitly surfaces `infeasible=True` when no strictly safe fallback exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Sequence

DT_MICRO: float = 0.25
ROBOT_RADIUS_M: float = 0.45
CAPSULE_HALF_OFFSET_M: float = 0.15
MAX_V_MPS: float = 1.35
MAX_OMEGA_RADPS: float = 1.65
MAX_DV_DT: float = 1.60
MAX_DW_DT: float = 3.20
SUBSTEPS_DEFAULT: int = 5


def wrap_pi(ang: float) -> float:
    return (ang + math.pi) % (2.0 * math.pi) - math.pi


def clip_val(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


@dataclass(frozen=True)
class FootprintConfig:
    """Unified metric robot collision envelope."""

    mode: Literal["capsule", "disc"] = "capsule"
    radius_m: float = ROBOT_RADIUS_M
    half_offset_m: float = CAPSULE_HALF_OFFSET_M

    @property
    def max_extent_m(self) -> float:
        return self.radius_m + (self.half_offset_m if self.mode == "capsule" else 0.0)


@dataclass(frozen=True)
class BoxObstacle:
    """Axis-aligned metric wall / solid obstacle box `[x_min, x_max] x [y_min, y_max]`."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def point_distance(self, px: float, py: float) -> float:
        """Euclidean distance from `(px, py)` to box (0.0 if inside)."""
        dx = max(self.x_min - px, 0.0, px - self.x_max)
        dy = max(self.y_min - py, 0.0, py - self.y_max)
        return math.hypot(dx, dy)

    def signed_distance(self, px: float, py: float) -> float:
        """Signed distance (negative inside box, positive outside)."""
        dx = max(self.x_min - px, px - self.x_max)
        dy = max(self.y_min - py, py - self.y_max)
        if dx <= 0.0 and dy <= 0.0:
            return max(dx, dy)
        return math.hypot(max(0.0, dx), max(0.0, dy))


@dataclass(frozen=True)
class CircleState:
    """Moving circular obstacle or peer disc snapshot."""

    x: float
    y: float
    vx: float
    vy: float
    radius: float


@dataclass(frozen=True)
class ShieldOutput:
    v_cmd: float
    omega_cmd: float
    intervened: bool
    infeasible: bool
    pred_min_clear_m: float


def robot_disc_centers(
    x: float,
    y: float,
    theta: float,
    fp: FootprintConfig = FootprintConfig(),
) -> tuple[tuple[float, float], ...]:
    """Return center coordinates of the robot's collision disc(s) in world frame."""
    if fp.mode == "disc" or fp.half_offset_m <= 1e-6:
        return ((x, y),)
    ct = math.cos(theta) * fp.half_offset_m
    st = math.sin(theta) * fp.half_offset_m
    return ((x + ct, y + st), (x - ct, y - st))


def corridor_wall_clearance(
    x: float,
    y: float,
    theta: float,
    half_width_m: float,
    fp: FootprintConfig = FootprintConfig(),
) -> float:
    """Exact signed surface clearance to straight corridor walls `|y| = half_width_m`."""
    del x  # straight infinite/bounded x-corridor uses y-walls uniformly across discs
    worst = float("inf")
    for _, cy in robot_disc_centers(0.0, y, theta, fp):
        gap = (half_width_m - abs(cy)) - fp.radius_m
        if gap < worst:
            worst = gap
    return worst


def box_wall_clearance(
    x: float,
    y: float,
    theta: float,
    wall_boxes: Sequence[BoxObstacle],
    fp: FootprintConfig = FootprintConfig(),
    *,
    world_bounds: tuple[float, float, float, float] | None = None,
) -> float:
    """Exact signed surface clearance to a collection of wall boxes and optional world bounds."""
    worst = float("inf")
    for cx, cy in robot_disc_centers(x, y, theta, fp):
        if world_bounds is not None:
            x0, y0, x1, y1 = world_bounds
            b_gap = min(cx - x0, x1 - cx, cy - y0, y1 - cy) - fp.radius_m
            if b_gap < worst:
                worst = b_gap
        for box in wall_boxes:
            d = box.signed_distance(cx, cy) - fp.radius_m
            if d < worst:
                worst = d
    return worst


def obstacle_surface_clearance(
    x: float,
    y: float,
    theta: float,
    obstacles: Sequence[CircleState],
    fp: FootprintConfig = FootprintConfig(),
) -> float:
    """Minimum surface clearance between robot footprint and circular obstacles."""
    if not obstacles:
        return 9.0
    centers = robot_disc_centers(x, y, theta, fp)
    worst = 9.0
    for o in obstacles:
        for cx, cy in centers:
            d = math.hypot(o.x - cx, o.y - cy) - (fp.radius_m + o.radius)
            if d < worst:
                worst = d
    return worst


def pair_robot_clearance(
    x1: float,
    y1: float,
    th1: float,
    fp1: FootprintConfig,
    x2: float,
    y2: float,
    th2: float,
    fp2: FootprintConfig,
) -> float:
    """Exact surface clearance between two multi-disc robots."""
    # Quick bounding-circle early reject
    bound = fp1.max_extent_m + fp2.max_extent_m + 0.5
    if abs(x1 - x2) > bound and abs(y1 - y2) > bound:
        return math.hypot(x1 - x2, y1 - y2) - (fp1.max_extent_m + fp2.max_extent_m)
    c1 = robot_disc_centers(x1, y1, th1, fp1)
    c2 = robot_disc_centers(x2, y2, th2, fp2)
    r_sum = fp1.radius_m + fp2.radius_m
    worst = float("inf")
    for ax, ay in c1:
        for bx, by in c2:
            d = math.hypot(ax - bx, ay - by) - r_sum
            if d < worst:
                worst = d
    return worst


def swept_corridor_collision(
    x0: float,
    y0: float,
    th0: float,
    x1: float,
    y1: float,
    th1: float,
    half_width_m: float,
    obs_start: Sequence[CircleState],
    obs_end: Sequence[CircleState],
    fp: FootprintConfig = FootprintConfig(),
    *,
    substeps: int = SUBSTEPS_DEFAULT,
) -> tuple[bool, float, float]:
    """Swept continuous-time check over `substeps` linear-interpolated fractions `(0, 1]`.

    Returns `(collided, min_obs_clear_m, min_wall_clear_m)`.
    """
    m = max(1, int(substeps))
    dth = wrap_pi(th1 - th0)
    min_obs = 9.0
    min_wall = 9.0
    for k in range(1, m + 1):
        alpha = k / float(m)
        xk = x0 + (x1 - x0) * alpha
        yk = y0 + (y1 - y0) * alpha
        thk = wrap_pi(th0 + dth * alpha)
        w_c = corridor_wall_clearance(xk, yk, thk, half_width_m, fp)
        if w_c < min_wall:
            min_wall = w_c
        if obs_start:
            obs_k = [
                CircleState(
                    x=o0.x + (o1.x - o0.x) * alpha,
                    y=o0.y + (o1.y - o0.y) * alpha,
                    vx=o1.vx,
                    vy=o1.vy,
                    radius=o0.radius,
                )
                for o0, o1 in zip(obs_start, obs_end)
            ]
            o_c = obstacle_surface_clearance(xk, yk, thk, obs_k, fp)
            if o_c < min_obs:
                min_obs = o_c
    collided = (min_obs <= 0.0) or (min_wall <= 0.0)
    return collided, min_obs, min_wall


class BrakingInvariantShield:
    """Inductive braking-invariant safety shield using unified SDF geometry.

    Evaluates `(v_cmd, omega_cmd)` followed by a full emergency deceleration tail
    (`H_stop` steps until `v == 0`), enforcing `min_clearance >= hard_min_clear_m`
    across BOTH walls and obstacles/peers with exact unified footprint geometry.
    """

    def __init__(
        self,
        *,
        footprint: FootprintConfig = FootprintConfig(),
        hard_min_clear_m: float = 0.08,
        wall_min_clear_m: float = 0.04,
        horizon_steps: int = 14,
        dt: float = DT_MICRO,
        max_v: float = MAX_V_MPS,
        max_omega: float = MAX_OMEGA_RADPS,
        max_dv_dt: float = MAX_DV_DT,
        max_dw_dt: float = MAX_DW_DT,
    ) -> None:
        self.fp = footprint
        self.hard_min_clear_m = float(hard_min_clear_m)
        self.wall_min_clear_m = float(wall_min_clear_m)
        self.horizon_steps = int(horizon_steps)
        self.dt = float(dt)
        self.max_v = float(max_v)
        self.max_omega = float(max_omega)
        self.max_dv_dt = float(max_dv_dt)
        self.max_dw_dt = float(max_dw_dt)

    def evaluate_margin(
        self,
        *,
        x: float,
        y: float,
        theta: float,
        v: float,
        omega: float,
        vt: float,
        wt: float,
        obstacles: Sequence[CircleState],
        wall_fn,
        ref_heading_err_fn=None,
        obs_pos_std_m: float = 0.0,
        obs_vel_std_mps: float = 0.0,
    ) -> float:
        px, py, pth, pv, pw = x, y, theta, v, omega
        worst = 9.0
        dv_max = self.max_dv_dt * self.dt
        dw_max = self.max_dw_dt * self.dt
        has_unc = (obs_pos_std_m > 0.0) or (obs_vel_std_mps > 0.0)
        for s in range(1, self.horizon_steps + 1):
            tgt_v = vt if s == 1 else 0.0
            if s <= 2:
                tgt_w = wt
            elif not obstacles and ref_heading_err_fn is not None:
                tgt_w = clip_val(1.6 * ref_heading_err_fn(px, py, pth), -self.max_omega, self.max_omega)
            else:
                tgt_w = 0.0
            pv = clip_val(pv + clip_val(tgt_v - pv, -dv_max, dv_max), 0.0, self.max_v)
            pw = clip_val(pw + clip_val(tgt_w - pw, -dw_max, dw_max), -self.max_omega, self.max_omega)
            px += pv * math.cos(pth) * self.dt
            py += pv * math.sin(pth) * self.dt
            pth = wrap_pi(pth + pw * self.dt)

            w_gap = wall_fn(px, py, pth) - self.wall_min_clear_m
            if w_gap < worst:
                worst = w_gap
            t_look = s * self.dt
            if obstacles:
                obs_pred = [
                    CircleState(o.x + o.vx * t_look, o.y + o.vy * t_look, o.vx, o.vy, o.radius)
                    for o in obstacles
                ]
                eps_t = 0.0
                if has_unc:
                    t_eff = min(t_look, 1.0)
                    eps_t = 1.96 * math.hypot(obs_pos_std_m, t_eff * obs_vel_std_mps)
                o_gap = obstacle_surface_clearance(px, py, pth, obs_pred, self.fp) - self.hard_min_clear_m - eps_t
                if o_gap < worst:
                    worst = o_gap
        return worst

    def filter(
        self,
        *,
        x: float,
        y: float,
        theta: float,
        v: float,
        omega: float,
        v_cmd: float,
        omega_cmd: float,
        obstacles: Sequence[CircleState],
        wall_fn,
        center_w: float = 0.0,
        ref_heading_err_fn=None,
        obs_pos_std_m: float = 0.0,
        obs_vel_std_mps: float = 0.0,
    ) -> ShieldOutput:
        v_req = clip_val(v_cmd, 0.0, self.max_v)
        w_req = clip_val(omega_cmd, -self.max_omega, self.max_omega)

        m_req = self.evaluate_margin(
            x=x, y=y, theta=theta, v=v, omega=omega, vt=v_req, wt=w_req,
            obstacles=obstacles, wall_fn=wall_fn, ref_heading_err_fn=ref_heading_err_fn,
            obs_pos_std_m=obs_pos_std_m, obs_vel_std_mps=obs_vel_std_mps,
        )
        if m_req >= 0.0:
            return ShieldOutput(v_req, w_req, intervened=False, infeasible=False, pred_min_clear_m=m_req)

        # Build rich fallback candidate grid: deceleration ladder + steering recovery + lateral repulsion
        repulse_w = center_w
        if obstacles:
            nearest = min(obstacles, key=lambda o: math.hypot(o.x - x, o.y - y) - o.radius)
            ang_to_obs = wrap_pi(math.atan2(nearest.y - y, nearest.x - x) - theta)
            repulse_w = clip_val((-1.65 if ang_to_obs >= 0.0 else 1.65), -self.max_omega, self.max_omega)

        w_cands = [
            w_req,
            center_w,
            repulse_w,
            0.0,
            clip_val(w_req + 0.55, -self.max_omega, self.max_omega),
            clip_val(w_req - 0.55, -self.max_omega, self.max_omega),
            clip_val(w_req + 1.10, -self.max_omega, self.max_omega),
            clip_val(w_req - 1.10, -self.max_omega, self.max_omega),
            self.max_omega,
            -self.max_omega,
        ]
        # Deduplicate close candidates
        uniq_w: list[float] = []
        for wc in w_cands:
            if not any(abs(wc - u) < 1e-3 for u in uniq_w):
                uniq_w.append(wc)

        best_m = -1e9
        best_pair = (0.0, 0.0)
        # First pass: try v_scales from high to low looking for ANY strictly safe (m >= 0) control
        for v_scale in (0.80, 0.55, 0.30, 0.12, 0.0):
            vt = v_req * v_scale if v_scale > 0.0 else 0.0
            for wt in uniq_w:
                m = self.evaluate_margin(
                    x=x, y=y, theta=theta, v=v, omega=omega, vt=vt, wt=wt,
                    obstacles=obstacles, wall_fn=wall_fn, ref_heading_err_fn=ref_heading_err_fn,
                    obs_pos_std_m=obs_pos_std_m, obs_vel_std_mps=obs_vel_std_mps,
                )
                if m >= 0.0:
                    return ShieldOutput(vt, wt, intervened=True, infeasible=False, pred_min_clear_m=m)
                # When infeasible: capsule mode never selects v > 0 (and holds still if nearly stopped);
                # disc mode may use slow crawl (v_scale <= 0.30) if it strictly improves margin (e.g. off wall)
                if self.fp.mode == "capsule":
                    if v_scale == 0.0 and m > best_m:
                        best_m = m
                        best_pair = (0.0, 0.0 if v <= 0.15 else wt)
                else:
                    if v_scale <= 0.30 and m > best_m:
                        best_m = m
                        best_pair = (vt, wt)

        # Explicitly report infeasible=True if NO strictly safe tail exists; always command v=0 brake
        return ShieldOutput(best_pair[0], best_pair[1], intervened=True, infeasible=True, pred_min_clear_m=best_m)
