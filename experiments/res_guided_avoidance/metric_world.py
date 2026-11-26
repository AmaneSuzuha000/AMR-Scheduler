"""P1 Unified Continuous Multi-Robot World + Spatiotemporal Reservation Coupling.

Co-simulates ALL `N` fleet robots simultaneously in one shared metric coordinate space
(`dt = 0.25 s`) over:
  1. `rail_freight_workshop_v1` (`140 m x 90 m`, 62 nodes, 78 lanes, exact `2.2-3.0 m` widths).
  2. Metric-lifted grid/corridor arenas (`Map_A_Metric`, `Map_B_Metric`, `Blind_Corner_T`,
     `Bottleneck_Merge`) with explicit wall boxes and occluded line-of-sight geometry.

Upstream task assignment uses a fixed lightweight background dispatcher (`Nearest_Idle` /
`Wu26_SemiCircle`) and `CPS-A*` + `EALR` reservation maintenance; all `N` robots interact
physically in continuous space with `M_sub = 5` swept collision checks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import random
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from amr_sched.network import Network, NodeKind
from amr_sched.planners.rust_spacetime_astar import RustCPSAStarGraphPlanner
from amr_sched.reservation import Interval
from amr_sched.rust_reservation import RustReservationTable
from amr_sched.workshop import WorkshopScene, legal_task_pairs, parking_slots
from experiments.cps_astar_routing.cps_astar_graph import CPSAStarConfig
from experiments.res_guided_avoidance.geometry_and_shield import (
    BoxObstacle,
    BrakingInvariantShield,
    CircleState,
    DT_MICRO,
    FootprintConfig,
    MAX_DV_DT,
    MAX_DW_DT,
    MAX_OMEGA_RADPS,
    MAX_V_MPS,
    SUBSTEPS_DEFAULT,
    box_wall_clearance,
    clip_val,
    obstacle_surface_clearance,
    pair_robot_clearance,
    robot_disc_centers,
    wrap_pi,
)
from experiments.track_r_cpsa.cpsa_pure_runner import expand_planned_to_step_positions, shortest_dist_steps
from experiments.track_r_cpsa.maps_and_workloads import build_map_a, build_map_b


def _point_to_segment_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * vx + (py - ay) * vy) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx = ax + t * vx
    cy = ay + t * vy
    return math.hypot(px - cx, py - cy)


def _segments_intersect(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, dx: float, dy: float) -> bool:
    def _ccw(p1x: float, p1y: float, p2x: float, p2y: float, p3x: float, p3y: float) -> float:
        return (p3y - p1y) * (p2x - p1x) - (p2y - p1y) * (p3x - p1x)

    if max(ax, bx) < min(cx, dx) or max(cx, dx) < min(ax, bx):
        return False
    if max(ay, by) < min(cy, dy) or max(cy, dy) < min(ay, by):
        return False
    d1 = _ccw(ax, ay, bx, by, cx, cy)
    d2 = _ccw(ax, ay, bx, by, dx, dy)
    d3 = _ccw(cx, cy, dx, dy, ax, ay)
    d4 = _ccw(cx, cy, dx, dy, bx, by)
    return ((d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0)) and (
        (d3 > 0.0 and d4 < 0.0) or (d3 < 0.0 and d4 > 0.0)
    )


def _segment_hits_box(ax: float, ay: float, bx: float, by: float, box: BoxObstacle) -> bool:
    if box.x_min <= ax <= box.x_max and box.y_min <= ay <= box.y_max:
        return True
    if box.x_min <= bx <= box.x_max and box.y_min <= by <= box.y_max:
        return True
    edges = (
        (box.x_min, box.y_min, box.x_max, box.y_min),
        (box.x_max, box.y_min, box.x_max, box.y_max),
        (box.x_max, box.y_max, box.x_min, box.y_max),
        (box.x_min, box.y_max, box.x_min, box.y_min),
    )
    for ex0, ey0, ex1, ey1 in edges:
        if _segments_intersect(ax, ay, bx, by, ex0, ey0, ex1, ey1):
            return True
    return False


@dataclass(frozen=True)
class CorridorCapsule:
    ax: float
    ay: float
    bx: float
    by: float
    half_width_m: float


@dataclass
class MetricMap:
    """Continuous 2D metric map paired with its topological routing graph."""

    name: str
    network: Network
    world_bounds: tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    wall_boxes: tuple[BoxObstacle, ...] = ()
    corridor_capsules: tuple[CorridorCapsule, ...] = ()
    edge_half_widths: dict[tuple[str, str], float] = field(default_factory=dict)
    pickup_nodes: tuple[str, ...] = ()
    dropoff_nodes: tuple[str, ...] = ()
    start_pool: tuple[str, ...] = ()
    task_pairs: tuple[tuple[str, str], ...] = ()
    macro_step_s: float = 1.0
    _cell_size: float = field(default=4.0, init=False, repr=False)
    _cap_grid: dict[tuple[int, int], tuple[CorridorCapsule, ...]] = field(default_factory=dict, init=False, repr=False)
    _box_grid: dict[tuple[int, int], tuple[BoxObstacle, ...]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        cs = self._cell_size
        if self.corridor_capsules:
            tmp: dict[tuple[int, int], list[CorridorCapsule]] = {}
            for cap in self.corridor_capsules:
                pad = cap.half_width_m + 1.5
                gx0 = int(math.floor((min(cap.ax, cap.bx) - pad) / cs))
                gx1 = int(math.floor((max(cap.ax, cap.bx) + pad) / cs))
                gy0 = int(math.floor((min(cap.ay, cap.by) - pad) / cs))
                gy1 = int(math.floor((max(cap.ay, cap.by) + pad) / cs))
                for gx in range(gx0, gx1 + 1):
                    for gy in range(gy0, gy1 + 1):
                        tmp.setdefault((gx, gy), []).append(cap)
            self._cap_grid = {k: tuple(v) for k, v in tmp.items()}
        if self.wall_boxes:
            btmp: dict[tuple[int, int], list[BoxObstacle]] = {}
            pad = 2.0
            for box in self.wall_boxes:
                gx0 = int(math.floor((box.x_min - pad) / cs))
                gx1 = int(math.floor((box.x_max + pad) / cs))
                gy0 = int(math.floor((box.y_min - pad) / cs))
                gy1 = int(math.floor((box.y_max + pad) / cs))
                for gx in range(gx0, gx1 + 1):
                    for gy in range(gy0, gy1 + 1):
                        btmp.setdefault((gx, gy), []).append(box)
            self._box_grid = {k: tuple(v) for k, v in btmp.items()}

    def wall_clearance(
        self,
        x: float,
        y: float,
        theta: float,
        fp: FootprintConfig = FootprintConfig(),
    ) -> float:
        """Signed surface clearance to nearest wall (`> 0` is strictly inside free space)."""
        cs = self._cell_size
        if self.corridor_capsules:
            worst = float("inf")
            for cx, cy in robot_disc_centers(x, y, theta, fp):
                best_cap = -1e9
                cands = self._cap_grid.get((int(math.floor(cx / cs)), int(math.floor(cy / cs))))
                if cands:
                    for cap in cands:
                        d_seg = _point_to_segment_dist(cx, cy, cap.ax, cap.ay, cap.bx, cap.by)
                        margin = (cap.half_width_m - d_seg) - fp.radius_m
                        if margin > best_cap:
                            best_cap = margin
                if best_cap < -1e8:
                    # Fallback full scan if outside spatial hash padding
                    for cap in self.corridor_capsules:
                        d_seg = _point_to_segment_dist(cx, cy, cap.ax, cap.ay, cap.bx, cap.by)
                        margin = (cap.half_width_m - d_seg) - fp.radius_m
                        if margin > best_cap:
                            best_cap = margin
                if best_cap < worst:
                    worst = best_cap
            return worst

        cands_box = self._box_grid.get((int(math.floor(x / cs)), int(math.floor(y / cs))), ())
        return box_wall_clearance(x, y, theta, cands_box, fp, world_bounds=self.world_bounds)

    def has_line_of_sight(self, ax: float, ay: float, bx: float, by: float) -> bool:
        """Return False if an opaque wall box or non-walkable wall blocks visibility."""
        if self.wall_boxes:
            for box in self.wall_boxes:
                if _segment_hits_box(ax, ay, bx, by, box):
                    return False
            return True
        if self.corridor_capsules:
            # Sample along sightline; if midpoint leaves walkable corridor union, sight is occluded by corner walls
            cs = self._cell_size
            dist = math.hypot(bx - ax, by - ay)
            n_chk = max(2, min(16, int(math.ceil(dist / 0.75))))
            for k in range(1, n_chk):
                al = k / float(n_chk)
                px = ax + (bx - ax) * al
                py = ay + (by - ay) * al
                inside = False
                cands = self._cap_grid.get((int(math.floor(px / cs)), int(math.floor(py / cs))))
                if cands:
                    for cap in cands:
                        if _point_to_segment_dist(px, py, cap.ax, cap.ay, cap.bx, cap.by) <= cap.half_width_m + 0.15:
                            inside = True
                            break
                if not inside:
                    return False
            return True
        return True

    def local_half_width(self, u: Optional[str], v: Optional[str]) -> float:
        if u is not None and v is not None:
            if (u, v) in self.edge_half_widths:
                return self.edge_half_widths[(u, v)]
            if (v, u) in self.edge_half_widths:
                return self.edge_half_widths[(v, u)]
        return 1.35


def build_workshop_metric_map() -> MetricMap:
    """Build exact continuous `MetricMap` from `scenes/rail_freight_workshop_v1.json`."""
    sc = WorkshopScene.load()
    net = sc.network()
    capsules: list[CorridorCapsule] = []
    ehw: dict[tuple[str, str], float] = {}
    for lane in sc.lanes:
        hw = max(1.10, 0.5 * float(lane.width_m))
        pts = lane.centerline
        if len(pts) < 2:
            nu, nv = net.node(lane.src), net.node(lane.dst)
            pts = ((nu.x, nu.y), (nv.x, nv.y))
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
            capsules.append(CorridorCapsule(ax, ay, bx, by, hw))
        ehw[(lane.src, lane.dst)] = hw
        ehw[(lane.dst, lane.src)] = hw
    # Ensure every node has a local junction pad >= max incident half-width
    for nid, nd in net.nodes.items():
        inc = [ehw.get((nid, e.dst), 1.35) for e in net.out_edges(nid)]
        pad_hw = max(inc) if inc else 1.35
        capsules.append(CorridorCapsule(nd.x, nd.y, nd.x, nd.y, pad_hw))

    pairs = tuple(legal_task_pairs(sc))
    sp = tuple(parking_slots(sc))
    p_nodes = tuple(sorted({p for p, _ in pairs}))
    d_nodes = tuple(sorted({d for _, d in pairs}))
    bx, by = sc.bounds
    return MetricMap(
        name="Workshop_v1_Metric",
        network=net,
        world_bounds=(0.0, 0.0, float(bx), float(by)),
        corridor_capsules=tuple(capsules),
        edge_half_widths=ehw,
        pickup_nodes=p_nodes,
        dropoff_nodes=d_nodes,
        start_pool=sp,
        task_pairs=pairs,
        macro_step_s=2.0,
    )


def build_grid_metric_map(
    which: str = "Map_A",
    *,
    cell_m: float = 2.6,
) -> MetricMap:
    """Lift a discrete grid MAPF/MAPD map into exact metric boxes (`cell_m >= 2.2 m`)."""
    spec = build_map_a() if which == "Map_A" else build_map_b()
    net = Network()
    for nid, nd in spec.network.nodes.items():
        net.add_node(
            type(nd)(
                node_id=nid,
                x=nd.x * cell_m + 0.5 * cell_m,
                y=nd.y * cell_m + 0.5 * cell_m,
                kind=nd.kind,
                capacity=nd.capacity,
            )
        )
    ehw: dict[tuple[str, str], float] = {}
    hw = 0.5 * cell_m
    for (u, v), e in spec.network.edges.items():
        net.add_edge(type(e)(src=u, dst=v, length=cell_m, speed_limit=1.0, capacity=e.capacity))
        ehw[(u, v)] = hw

    walls = tuple(
        BoxObstacle(x * cell_m, y * cell_m, (x + 1) * cell_m, (y + 1) * cell_m)
        for (x, y) in sorted(spec.obstacles)
    )
    p_nodes = tuple(spec.cell_id(x, y) for x, y in spec.pickup_cells)
    d_nodes = tuple(spec.cell_id(x, y) for x, y in spec.dropoff_cells)
    all_nodes = tuple(sorted(net.nodes.keys()))
    return MetricMap(
        name=f"{spec.name}_Metric_{cell_m:.1f}m",
        network=net,
        world_bounds=(0.0, 0.0, spec.width * cell_m, spec.height * cell_m),
        wall_boxes=walls,
        edge_half_widths=ehw,
        pickup_nodes=p_nodes,
        dropoff_nodes=d_nodes,
        start_pool=all_nodes,
        macro_step_s=max(1.0, round(cell_m / 1.0, 1)),
    )


def build_occluded_junction_map(*, cell_m: float = 2.6) -> MetricMap:
    """Build a 7x7 metric 4-way T/cross blind-corner arena where walls occlude cross-traffic LOS."""
    w, h = 7, 7
    obs: set[tuple[int, int]] = set()
    # 4 opaque corner blocks creating blind cross-intersections at x=3 and y=3
    for x in (0, 1, 2, 4, 5, 6):
        for y in (0, 1, 2, 4, 5, 6):
            if (x, y) in ((1, 1), (1, 2), (2, 1), (2, 2), (4, 1), (5, 1), (4, 2), (5, 2),
                          (1, 4), (2, 4), (1, 5), (2, 5), (4, 4), (5, 4), (4, 5), (5, 5)):
                obs.add((x, y))
    net = Network()
    trav: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if (x, y) in obs:
                continue
            trav.append((x, y))
            net.add_node(
                type(next(iter(build_map_a().network.nodes.values())))(
                    node_id=f"C_{x}_{y}",
                    x=x * cell_m + 0.5 * cell_m,
                    y=y * cell_m + 0.5 * cell_m,
                    kind=NodeKind.AISLE,
                    capacity=1,
                )
            )
    ehw: dict[tuple[str, str], float] = {}
    hw = 0.5 * cell_m
    for x, y in trav:
        u = f"C_{x}_{y}"
        for dx, dy in ((1, 0), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in obs:
                v = f"C_{nx}_{ny}"
                net.add_bidirectional(u, v, length=cell_m, speed_limit=1.0, capacity=1)
                ehw[(u, v)] = hw
                ehw[(v, u)] = hw

    walls = tuple(
        BoxObstacle(x * cell_m, y * cell_m, (x + 1) * cell_m, (y + 1) * cell_m)
        for (x, y) in sorted(obs)
    )
    perimeter = tuple(
        f"C_{x}_{y}" for (x, y) in trav if x in (0, w - 1) or y in (0, h - 1)
    )
    return MetricMap(
        name=f"Occluded_Junction_{cell_m:.1f}m",
        network=net,
        world_bounds=(0.0, 0.0, w * cell_m, h * cell_m),
        wall_boxes=walls,
        edge_half_widths=ehw,
        pickup_nodes=perimeter,
        dropoff_nodes=perimeter,
        start_pool=perimeter,
        macro_step_s=max(1.0, round(cell_m / 1.0, 1)),
    )


@dataclass
class ContTask:
    task_id: int
    pickup: str
    dropoff: str
    release_s: float
    due_s: float
    service_s: float = 1.0


@dataclass
class RobotContinuousState:
    robot_id: str
    x: float
    y: float
    theta: float
    v: float = 0.0
    omega: float = 0.0
    cur_node: str = ""
    phase: str = "idle"  # idle | to_pickup | to_dropoff
    cur_task: Optional[ContTask] = None
    service_rem_s: float = 0.0
    route_nodes: list[str] = field(default_factory=list)
    route_times: list[int] = field(default_factory=list)
    route_idx: int = 0
    speed_factor: float = 1.0
    eta_drift_ticks: int = 0


@dataclass
class DynamicIntruder:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    x_lo: float
    x_hi: float
    y_lo: float
    y_hi: float

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
            self.vy = abs(self.vy)
        elif self.y > self.y_hi:
            self.y = self.y_hi
            self.vy = -abs(self.vy)


@dataclass(frozen=True)
class MultiRobotEpisodeResult:
    map_name: str
    controller: str
    n_robots: int
    n_tasks: int
    seed: int
    completed_tasks: int
    makespan_s: float
    zero_collisions: bool
    wall_collisions: int
    peer_collisions: int
    intruder_collisions: int
    shield_interventions: int
    shield_infeasible_count: int
    wait_ratio: float
    mean_jerk: float
    replans: int
    mean_ctrl_ms: float
    mean_min_clear_m: float


class MultiRobotUnicycleSim:
    """Unified `dt = 0.25 s` continuous multi-robot world coupled with `CPS-A*` + `EALR` reservations."""

    def __init__(
        self,
        metric_map: MetricMap,
        *,
        n_robots: int = 8,
        n_tasks: int = 24,
        horizon_s: float = 180.0,
        seed: int = 0,
        footprint: FootprintConfig = FootprintConfig(mode="capsule"),
        use_shield: bool = True,
        obs_noise_pos_m: float = 0.0,
        obs_noise_vel_mps: float = 0.0,
        reservation_lag_s: float = 0.0,
        reservation_dropout: float = 0.0,
        disturb_slow_prob: float = 0.08,
        n_intruders: int = 2,
        occluded_los: bool = True,
    ) -> None:
        self.mmap = metric_map
        self.n_robots = int(n_robots)
        self.n_tasks = int(n_tasks)
        self.horizon_s = float(horizon_s)
        self.seed = int(seed)
        self.fp = footprint
        self.use_shield = bool(use_shield)
        self.obs_noise_pos_m = float(obs_noise_pos_m)
        self.obs_noise_vel_mps = float(obs_noise_vel_mps)
        self.reservation_lag_s = float(reservation_lag_s)
        self.reservation_dropout = float(reservation_dropout)
        self.disturb_slow_prob = float(disturb_slow_prob)
        self.n_intruders = int(n_intruders)
        self.occluded_los = bool(occluded_los)

        self.rng = np.random.default_rng(self.seed)
        self.py_rng = random.Random(self.seed)
        self.shield = BrakingInvariantShield(
            footprint=self.fp,
            hard_min_clear_m=0.18,
            wall_min_clear_m=0.03,
            horizon_steps=14,
            dt=DT_MICRO,
        )

        # Background Rust CPS-A* + ReservationTable
        self.cfg = CPSAStarConfig(mode="lifelong", beta=0.5, gamma=4.0)
        self.planner = RustCPSAStarGraphPlanner(
            self.mmap.network, self.cfg, max_speed=1.0, dt=self.mmap.macro_step_s
        )
        self.res_table = RustReservationTable.fast()
        self.field = self.planner.create_field()
        self.macro_steps_per_tick = max(1, int(round(self.mmap.macro_step_s / DT_MICRO)))
        self.plan_horizon_ticks = int(math.ceil(self.horizon_s / self.mmap.macro_step_s)) + 80

        self.robots: dict[str, RobotContinuousState] = {}
        self.robot_ids: list[str] = []
        self.intruders: list[DynamicIntruder] = []
        self.pending: deque[ContTask] = deque()
        self.ready: list[ContTask] = []
        self.completed_tasks = 0
        self.t_s = 0.0
        self.micro_step = 0
        self.macro_tick = 0

        # Reservation snapshot history for staleness/lag injection
        self._res_history: deque[tuple[float, dict[str, list[tuple[str, int, int, int]]]]] = deque(maxlen=64)
        self._current_res_view: dict[str, list[tuple[str, int, int, int]]] = {}
        self._lagged_res_view: dict[str, list[tuple[str, int, int, int]]] = {}

        # Metrics
        self.wall_collisions = 0
        self.peer_collisions = 0
        self.intruder_collisions = 0
        self.shield_interventions = 0
        self.shield_infeasible_count = 0
        self.replans = 0
        self._wait_steps = 0
        self._total_robot_steps = 0
        self._jerk_sum = 0.0
        self._prev_v: dict[str, float] = {}
        self._prev_a: dict[str, float] = {}
        self._min_clear_samples: list[float] = []
        self._ctrl_ms_samples: list[float] = []

        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.py_rng = random.Random(self.seed)
        self.res_table = RustReservationTable.fast()
        self.field = self.planner.create_field()
        self.robots.clear()
        self.robot_ids.clear()
        self.intruders.clear()
        self.pending.clear()
        self.ready.clear()
        self.completed_tasks = 0
        self.t_s = 0.0
        self.micro_step = 0
        self.macro_tick = 0
        self._res_history.clear()
        self._current_res_view.clear()
        self._lagged_res_view.clear()

        self.wall_collisions = 0
        self.peer_collisions = 0
        self.intruder_collisions = 0
        self.shield_interventions = 0
        self.shield_infeasible_count = 0
        self.replans = 0
        self._wait_steps = 0
        self._total_robot_steps = 0
        self._jerk_sum = 0.0
        self._prev_v.clear()
        self._prev_a.clear()
        self._min_clear_samples.clear()
        self._ctrl_ms_samples.clear()

        # Place N robots cleanly with >= 1.8 m mutual separation
        pool = list(self.mmap.start_pool or sorted(self.mmap.network.nodes.keys()))
        self.py_rng.shuffle(pool)
        all_nodes = list(sorted(self.mmap.network.nodes.keys()))
        self.py_rng.shuffle(all_nodes)
        candidates = pool + [n for n in all_nodes if n not in pool]

        placed: list[tuple[float, float]] = []
        for idx in range(self.n_robots):
            rid = f"R_{idx:02d}"
            chosen_nid = candidates[idx % len(candidates)]
            for cand in candidates:
                nc = self.mmap.network.node(cand)
                if all(math.hypot(nc.x - px, nc.y - py) >= 1.8 for px, py in placed):
                    chosen_nid = cand
                    break
            nd = self.mmap.network.node(chosen_nid)
            placed.append((nd.x, nd.y))
            outs = self.mmap.network.out_edges(chosen_nid)
            if outs:
                nb = self.mmap.network.node(outs[0].dst)
                yaw = math.atan2(nb.y - nd.y, nb.x - nd.x)
            else:
                yaw = 0.0
            self.robots[rid] = RobotContinuousState(
                robot_id=rid,
                x=float(nd.x),
                y=float(nd.y),
                theta=yaw,
                cur_node=chosen_nid,
                route_nodes=[chosen_nid],
                route_times=[0],
                route_idx=0,
            )
            self.robot_ids.append(rid)
            self.res_table.reserve_node(chosen_nid, Interval(0, 2), rid)
            self._prev_v[rid] = 0.0
            self._prev_a[rid] = 0.0

        # Spawn tasks cleanly
        p_pool = list(self.mmap.pickup_nodes or all_nodes)
        d_pool = list(self.mmap.dropoff_nodes or all_nodes)
        rel = 0.0
        dist_cache: dict[tuple[str, str], int] = {}
        for tid in range(self.n_tasks):
            if self.mmap.task_pairs:
                pu, do = self.py_rng.choice(self.mmap.task_pairs)
            else:
                pu = self.py_rng.choice(p_pool)
                do = self.py_rng.choice(d_pool)
                while do == pu and len(d_pool) > 1:
                    do = self.py_rng.choice(d_pool)
            if tid >= max(2, int(0.25 * self.n_tasks)):
                rel += self.py_rng.expovariate(0.45)
            d_m = shortest_dist_steps(self.mmap.network, pu, do, dist_cache) * self.mmap.macro_step_s
            self.pending.append(ContTask(tid, pu, do, release_s=round(rel, 2), due_s=round(rel + d_m + 35.0, 2)))

        # Spawn dynamic intruders on wider corridors (hw >= 1.25 m) away from robot starts
        edges = list(self.mmap.network.edges.values())
        self.py_rng.shuffle(edges)
        for e in edges:
            if len(self.intruders) >= self.n_intruders:
                break
            if e.length < 3.0:
                continue
            if self.mmap.local_half_width(e.src, e.dst) < 1.25:
                continue
            nu, nv = self.mmap.network.node(e.src), self.mmap.network.node(e.dst)
            mx, my = 0.5 * (nu.x + nv.x), 0.5 * (nu.y + nv.y)
            if any(math.hypot(mx - px, my - py) < 2.8 for px, py in placed):
                continue
            dx, dy = nv.x - nu.x, nv.y - nu.y
            ln = max(1e-6, math.hypot(dx, dy))
            ux, uy = dx / ln, dy / ln
            # Offset intruder slightly to one side of the wide corridor and move slowly
            lat_sign = 1.0 if (len(self.intruders) % 2 == 0) else -1.0
            ox = mx - uy * 0.35 * lat_sign
            oy = my + ux * 0.35 * lat_sign
            spd = float(self.rng.uniform(0.15, 0.30))
            self.intruders.append(
                DynamicIntruder(
                    x=ox,
                    y=oy,
                    vx=ux * spd,
                    vy=uy * spd,
                    radius=0.30,
                    x_lo=min(nu.x, nv.x) - 0.2,
                    x_hi=max(nu.x, nv.x) + 0.2,
                    y_lo=min(nu.y, nv.y) - 0.2,
                    y_hi=max(nu.y, nv.y) + 0.2,
                )
            )
        self._refresh_res_views()

    def _nearest_node(self, x: float, y: float) -> str:
        best_nid = self.robot_ids[0]
        best_d = 1e18
        for nid, nd in self.mmap.network.nodes.items():
            d = (nd.x - x) ** 2 + (nd.y - y) ** 2
            if d < best_d:
                best_d = d
                best_nid = nid
        return best_nid

    def _plan_robot(self, rid: str, goal_node: str, t_tick: int) -> bool:
        rb = self.robots[rid]
        start_node = self._nearest_node(rb.x, rb.y)
        rb.cur_node = start_node
        self.res_table.release_all(rid)
        self.field.remove_agv(rid)
        self.res_table.reserve_node(start_node, Interval(t_tick, t_tick + 1), rid)
        pl, _ = self.planner.plan_leg(
            start_node,
            goal_node,
            t_tick,
            self.res_table,
            self.field,
            rid,
            min(self.plan_horizon_ticks, t_tick + 90),
            goal_park_steps=1,
        )
        self.replans += 1
        if pl is None or not pl.nodes:
            self.res_table.reserve_node(start_node, Interval(t_tick, t_tick + 3), rid)
            rb.route_nodes = [start_node]
            rb.route_times = [t_tick]
            rb.route_idx = 0
            return False

        pts = expand_planned_to_step_positions(start_node, t_tick, pl)
        self.planner.reserve_step_points_indexed(self.res_table, rid, pts, park_until=None)
        self.field.add_agv_path(rid, pts)
        # Compress repeated wait nodes into route waypoints + expected arrival ticks
        w_nodes: list[str] = []
        w_times: list[int] = []
        for nd_id, step_t in zip(pl.nodes, pl.arrival_steps):
            if not w_nodes or w_nodes[-1] != nd_id:
                w_nodes.append(nd_id)
                w_times.append(int(step_t))
        rb.route_nodes = w_nodes
        rb.route_times = w_times
        rb.route_idx = 0
        return True

    def _refresh_res_views(self) -> None:
        view: dict[str, list[tuple[str, int, int, int]]] = {}
        for rid in self.robot_ids:
            rb = self.robots[rid]
            segs: list[tuple[str, int, int, int]] = []
            # Tuple: (node_id, enter_step, exit_step, prio)
            prio = 2 if rb.phase == "to_dropoff" else (1 if rb.phase == "to_pickup" else 0)
            for idx, nd_id in enumerate(rb.route_nodes):
                t_en = rb.route_times[idx] if idx < len(rb.route_times) else self.macro_tick
                t_ex = rb.route_times[idx + 1] if (idx + 1) < len(rb.route_times) else (t_en + 2)
                segs.append((nd_id, t_en, t_ex, prio))
            view[rid] = segs
        self._current_res_view = view
        self._res_history.append((self.t_s, view))
        if self.reservation_lag_s <= 1e-6:
            self._lagged_res_view = view
        else:
            target_t = self.t_s - self.reservation_lag_s
            chosen = self._res_history[0][1]
            for ts, vw in self._res_history:
                if ts <= target_t:
                    chosen = vw
                else:
                    break
            self._lagged_res_view = chosen

    def _macro_coordinate(self) -> None:
        t_tick = self.macro_tick
        while self.pending and self.pending[0].release_s <= self.t_s + 1e-6:
            self.ready.append(self.pending.popleft())

        # Check pickup / dropoff arrival within 0.65 m of target node
        for rid in self.robot_ids:
            rb = self.robots[rid]
            if rb.cur_task is None or rb.service_rem_s > 0.0:
                continue
            tgt_id = rb.cur_task.pickup if rb.phase == "to_pickup" else rb.cur_task.dropoff
            tgt = self.mmap.network.node(tgt_id)
            if math.hypot(rb.x - tgt.x, rb.y - tgt.y) <= 0.70:
                if rb.phase == "to_pickup":
                    rb.service_rem_s = rb.cur_task.service_s
                    rb.phase = "to_dropoff"
                    self._plan_robot(rid, rb.cur_task.dropoff, t_tick + 1)
                elif rb.phase == "to_dropoff":
                    self.completed_tasks += 1
                    rb.cur_task = None
                    rb.phase = "idle"
                    rb.service_rem_s = 0.0
                    rb.route_nodes = [tgt_id]
                    rb.route_times = [t_tick]
                    rb.route_idx = 0

        # Background Nearest-Idle dispatcher (fixed, zero learning)
        idle_rids = [rid for rid in self.robot_ids if self.robots[rid].phase == "idle" and self.robots[rid].service_rem_s <= 0.0]
        if idle_rids and self.ready:
            used_tasks: set[int] = set()
            for rid in idle_rids:
                rb = self.robots[rid]
                best_tk: Optional[ContTask] = None
                best_score = 1e18
                for tk in self.ready:
                    if tk.task_id in used_tasks:
                        continue
                    pu = self.mmap.network.node(tk.pickup)
                    d = math.hypot(rb.x - pu.x, rb.y - pu.y)
                    if d < best_score:
                        best_score = d
                        best_tk = tk
                if best_tk is not None:
                    used_tasks.add(best_tk.task_id)
                    rb.cur_task = best_tk
                    rb.phase = "to_pickup"
                    self._plan_robot(rid, best_tk.pickup, t_tick)
            if used_tasks:
                self.ready = [tk for tk in self.ready if tk.task_id not in used_tasks]

        # EALR-style ETA drift detection: replan robots whose progress drifted behind reservation
        for rid in self.robot_ids:
            rb = self.robots[rid]
            if rb.phase == "idle" or rb.cur_task is None or rb.service_rem_s > 0.0:
                continue
            if rb.route_idx + 1 < len(rb.route_times):
                exp_t = rb.route_times[rb.route_idx + 1]
                if t_tick > exp_t + 2:
                    rb.eta_drift_ticks += 1
                    goal = rb.cur_task.pickup if rb.phase == "to_pickup" else rb.cur_task.dropoff
                    self._plan_robot(rid, goal, t_tick)

        # Random mild speed perturbation injection
        for rid in self.robot_ids:
            rb = self.robots[rid]
            if self.rng.random() < self.disturb_slow_prob:
                rb.speed_factor = float(self.rng.uniform(0.65, 0.90))
            else:
                rb.speed_factor = 1.0

        self._refresh_res_views()

    def _advance_route_cursor(self, rb: RobotContinuousState) -> tuple[float, float]:
        """Return active continuous lookahead waypoint `(wx, wy)` projected onto robot's planned polyline."""
        while rb.route_idx + 1 < len(rb.route_nodes):
            nxt_nd = self.mmap.network.node(rb.route_nodes[rb.route_idx + 1])
            if math.hypot(nxt_nd.x - rb.x, nxt_nd.y - rb.y) <= 0.75:
                rb.route_idx += 1
            else:
                break
        if rb.route_idx + 1 < len(rb.route_nodes):
            u_nd = self.mmap.network.node(rb.route_nodes[rb.route_idx])
            v_nd = self.mmap.network.node(rb.route_nodes[rb.route_idx + 1])
            dx, dy = v_nd.x - u_nd.x, v_nd.y - u_nd.y
            seg_len = math.hypot(dx, dy)
            if seg_len > 1e-6:
                t_proj = clip_val(((rb.x - u_nd.x) * dx + (rb.y - u_nd.y) * dy) / (seg_len * seg_len), 0.0, 1.0)
                look_s = min(seg_len, t_proj * seg_len + 1.6)
                al = look_s / seg_len
                return float(u_nd.x + al * dx), float(u_nd.y + al * dy)
            return float(v_nd.x), float(v_nd.y)
        cur_nd = self.mmap.network.node(rb.route_nodes[-1] if rb.route_nodes else rb.cur_node)
        return float(cur_nd.x), float(cur_nd.y)

    def get_perceived_circles(self, rid: str) -> list[CircleState]:
        """Return peer robots + dynamic intruders visible to `rid` (respecting LOS occlusion & noise)."""
        ego = self.robots[rid]
        out: list[CircleState] = []
        for other_id in self.robot_ids:
            if other_id == rid:
                continue
            o = self.robots[other_id]
            if math.hypot(o.x - ego.x, o.y - ego.y) > 9.0:
                continue
            if self.occluded_los and not self.mmap.has_line_of_sight(ego.x, ego.y, o.x, o.y):
                continue
            nx = float(self.rng.normal(0.0, self.obs_noise_pos_m)) if self.obs_noise_pos_m > 0 else 0.0
            ny = float(self.rng.normal(0.0, self.obs_noise_pos_m)) if self.obs_noise_pos_m > 0 else 0.0
            nvx = float(self.rng.normal(0.0, self.obs_noise_vel_mps)) if self.obs_noise_vel_mps > 0 else 0.0
            nvy = float(self.rng.normal(0.0, self.obs_noise_vel_mps)) if self.obs_noise_vel_mps > 0 else 0.0
            out.append(
                CircleState(
                    x=o.x + nx,
                    y=o.y + ny,
                    vx=o.v * math.cos(o.theta) + nvx,
                    vy=o.v * math.sin(o.theta) + nvy,
                    radius=self.fp.max_extent_m,
                )
            )
        for intr in self.intruders:
            if math.hypot(intr.x - ego.x, intr.y - ego.y) > 9.0:
                continue
            if self.occluded_los and not self.mmap.has_line_of_sight(ego.x, ego.y, intr.x, intr.y):
                continue
            out.append(CircleState(intr.x, intr.y, intr.vx, intr.vy, intr.radius))
        return out

    def get_reservation_view(self) -> dict[str, list[tuple[str, int, int, int]]]:
        """Return (possibly lagged / dropout-corrupted) reservation snapshot for local policies."""
        if self.reservation_dropout <= 1e-6:
            return self._lagged_res_view
        return {
            rid: segs
            for rid, segs in self._lagged_res_view.items()
            if self.rng.random() >= self.reservation_dropout
        }

    def step(
        self,
        controller_fn: Callable[["MultiRobotUnicycleSim", str], tuple[float, float]],
        *,
        controller_name: str = "controller",
    ) -> bool:
        """Advance all `N` robots simultaneously by `DT_MICRO = 0.25 s`. Returns `done`."""
        if self.micro_step % self.macro_steps_per_tick == 0:
            self._macro_coordinate()
            self.macro_tick += 1

        cmds: dict[str, tuple[float, float]] = {}
        t0 = time.perf_counter()
        for rid in self.robot_ids:
            rb = self.robots[rid]
            if rb.service_rem_s > 0.0 or rb.phase == "idle":
                cmds[rid] = (0.0, 0.0)
                continue
            v_req, w_req = controller_fn(self, rid)
            if self.use_shield:
                wx, wy = self._advance_route_cursor(rb)
                obs_vis = self.get_perceived_circles(rid)
                sh_out = self.shield.filter(
                    x=rb.x,
                    y=rb.y,
                    theta=rb.theta,
                    v=rb.v,
                    omega=rb.omega,
                    v_cmd=v_req,
                    omega_cmd=w_req,
                    obstacles=obs_vis,
                    wall_fn=lambda px, py, pth: self.mmap.wall_clearance(px, py, pth, self.fp),
                    center_w=clip_val(1.6 * wrap_pi(math.atan2(wy - rb.y, wx - rb.x) - rb.theta), -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS),
                    ref_heading_err_fn=lambda px, py, pth, _wx=wx, _wy=wy: wrap_pi(math.atan2(_wy - py, _wx - px) - pth),
                )
                if sh_out.intervened:
                    self.shield_interventions += 1
                if sh_out.infeasible:
                    self.shield_infeasible_count += 1
                v_req, w_req = sh_out.v_cmd, sh_out.omega_cmd
            cmds[rid] = (v_req, w_req)
        self._ctrl_ms_samples.append(((time.perf_counter() - t0) * 1000.0) / max(1, self.n_robots))

        # Record start poses for swept M_sub = 5 collision check
        prev_poses = {rid: (rb.x, rb.y, rb.theta) for rid, rb in self.robots.items()}
        prev_intruders = [(intr.x, intr.y, intr.vx, intr.vy, intr.radius) for intr in self.intruders]

        dv_max = MAX_DV_DT * DT_MICRO
        dw_max = MAX_DW_DT * DT_MICRO
        for rid in self.robot_ids:
            rb = self.robots[rid]
            if rb.service_rem_s > 0.0:
                rb.service_rem_s = max(0.0, rb.service_rem_s - DT_MICRO)
                rb.v = 0.0
                rb.omega = 0.0
                continue
            vc, wc = cmds[rid]
            vc = clip_val(vc, 0.0, MAX_V_MPS * rb.speed_factor)
            wc = clip_val(wc, -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS)
            rb.v = clip_val(rb.v + clip_val(vc - rb.v, -dv_max, dv_max), 0.0, MAX_V_MPS)
            rb.omega = clip_val(rb.omega + clip_val(wc - rb.omega, -dw_max, dw_max), -MAX_OMEGA_RADPS, MAX_OMEGA_RADPS)
            rb.x += rb.v * math.cos(rb.theta) * DT_MICRO
            rb.y += rb.v * math.sin(rb.theta) * DT_MICRO
            rb.theta = wrap_pi(rb.theta + rb.omega * DT_MICRO)

            # Update kinematic smoothness / wait metrics
            if rb.phase != "idle":
                self._total_robot_steps += 1
                if rb.v < 0.15:
                    self._wait_steps += 1
                acc = (rb.v - self._prev_v[rid]) / DT_MICRO
                self._jerk_sum += abs(acc - self._prev_a[rid]) / DT_MICRO
                self._prev_v[rid] = rb.v
                self._prev_a[rid] = acc

        for intr in self.intruders:
            nxt_ix = intr.x + intr.vx * DT_MICRO
            nxt_iy = intr.y + intr.vy * DT_MICRO
            blocked_by_stopped = any(
                math.hypot(nxt_ix - rb.x, nxt_iy - rb.y) < (self.fp.max_extent_m + intr.radius + 0.08)
                for rb in self.robots.values()
            )
            if blocked_by_stopped:
                intr.vx = -intr.vx
                intr.vy = -intr.vy
            else:
                intr.step(DT_MICRO)

        # Swept continuous-time collision check across ALL robots, walls, and intruders
        for k in range(1, SUBSTEPS_DEFAULT + 1):
            al = k / float(SUBSTEPS_DEFAULT)
            interp_robots: dict[str, tuple[float, float, float]] = {}
            for rid, rb in self.robots.items():
                px, py, pth = prev_poses[rid]
                interp_robots[rid] = (
                    px + (rb.x - px) * al,
                    py + (rb.y - py) * al,
                    wrap_pi(pth + wrap_pi(rb.theta - pth) * al),
                )
            interp_intruders = [
                CircleState(
                    x=ix0 + (intr.x - ix0) * al,
                    y=iy0 + (intr.y - iy0) * al,
                    vx=intr.vx,
                    vy=intr.vy,
                    radius=intr.radius,
                )
                for (ix0, iy0, _, _, _), intr in zip(prev_intruders, self.intruders)
            ]
            for rid in self.robot_ids:
                rx, ry, rth = interp_robots[rid]
                w_c = self.mmap.wall_clearance(rx, ry, rth, self.fp)
                if w_c <= 0.0:
                    self.wall_collisions += 1
                if interp_intruders:
                    o_c = obstacle_surface_clearance(rx, ry, rth, interp_intruders, self.fp)
                    if o_c <= 0.0:
                        self.intruder_collisions += 1
                    self._min_clear_samples.append(min(w_c, o_c))
                else:
                    self._min_clear_samples.append(w_c)

            for i in range(len(self.robot_ids)):
                r1 = self.robot_ids[i]
                x1, y1, th1 = interp_robots[r1]
                for j in range(i + 1, len(self.robot_ids)):
                    r2 = self.robot_ids[j]
                    x2, y2, th2 = interp_robots[r2]
                    pc = pair_robot_clearance(x1, y1, th1, self.fp, x2, y2, th2, self.fp)
                    if pc <= 0.0:
                        self.peer_collisions += 1
                    self._min_clear_samples.append(pc)

        self.micro_step += 1
        self.t_s = round(self.micro_step * DT_MICRO, 4)
        done = (self.completed_tasks >= self.n_tasks) or (self.t_s >= self.horizon_s)
        return done

    def run_episode(
        self,
        controller_fn: Callable[["MultiRobotUnicycleSim", str], tuple[float, float]],
        *,
        controller_name: str = "controller",
    ) -> MultiRobotEpisodeResult:
        while not self.step(controller_fn, controller_name=controller_name):
            pass
        tot_col = self.wall_collisions + self.peer_collisions + self.intruder_collisions
        return MultiRobotEpisodeResult(
            map_name=self.mmap.name,
            controller=controller_name,
            n_robots=self.n_robots,
            n_tasks=self.n_tasks,
            seed=self.seed,
            completed_tasks=self.completed_tasks,
            makespan_s=round(self.t_s, 2),
            zero_collisions=(tot_col == 0),
            wall_collisions=self.wall_collisions,
            peer_collisions=self.peer_collisions,
            intruder_collisions=self.intruder_collisions,
            shield_interventions=self.shield_interventions,
            shield_infeasible_count=self.shield_infeasible_count,
            wait_ratio=round(self._wait_steps / max(1, self._total_robot_steps), 4),
            mean_jerk=round(self._jerk_sum / max(1, self._total_robot_steps), 3),
            replans=self.replans,
            mean_ctrl_ms=round(float(np.mean(self._ctrl_ms_samples)) if self._ctrl_ms_samples else 0.0, 4),
            mean_min_clear_m=round(float(np.mean(self._min_clear_samples)) if self._min_clear_samples else 9.0, 3),
        )
