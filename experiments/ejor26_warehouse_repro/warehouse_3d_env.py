"""3D automated warehouse geometry, Zipf SKU catalog, and batch-picking VRP evaluator.

Standalone synthetic reproduction of Wu et al. (EJOR 2026) 3D storage + path-planning VRP.
Explicitly isolated from AMR-Scheduler factory production datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Warehouse3DConfig:
    """Configuration for a 3D multi-tier aisle-based automated warehouse."""

    n_aisles: int = 4
    n_bays: int = 8
    n_tiers: int = 4
    aisle_pitch: float = 3.0
    bay_pitch: float = 2.0
    tier_height: float = 1.5
    v_xy: float = 1.5
    v_z: float = 0.75
    vehicle_capacity: float = 20.0


@dataclass(frozen=True)
class SKUItem:
    """Synthetic SKU descriptor with turnover frequency and demand weight."""

    sku_id: int
    turnover: float
    demand_weight: int
    turnover_norm: float
    demand_norm: float
    priority_pm: float


@dataclass(frozen=True)
class WarehouseVRPInstance:
    """Single 3D batch-picking VRP instance with precomputed exact 3D travel times."""

    instance_id: str
    storage_policy: str
    coords_3d: np.ndarray     # [N+1, 3], node 0 is depot (0, 0, 0)
    demands: np.ndarray       # [N+1], depot demand is 0
    priorities: np.ndarray    # [N+1], priority p_m in [0, 1]
    turnovers: np.ndarray     # [N+1]
    dist_matrix: np.ndarray   # [N+1, N+1], exact 3D travel time (seconds)
    capacity: float
    slots: tuple[tuple[int, int, int], ...]


class Warehouse3DEnv:
    """3D rack-aisle warehouse environment supporting exact travel times and tour checks."""

    def __init__(self, config: Warehouse3DConfig | None = None) -> None:
        self.cfg = config or Warehouse3DConfig()
        self.depot_coord = (0.0, 0.0, 0.0)
        self.all_slots: tuple[tuple[int, int, int], ...] = tuple(
            (a, b, t)
            for a in range(self.cfg.n_aisles)
            for b in range(self.cfg.n_bays)
            for t in range(self.cfg.n_tiers)
        )
        self.y_rear = float(self.cfg.n_bays + 1) * float(self.cfg.bay_pitch)

    def slot_to_xyz(self, slot: tuple[int, int, int]) -> tuple[float, float, float]:
        """Convert 0-based `(aisle, bay, tier)` index into continuous 3D coordinates."""
        a, b, t = slot
        return (
            float(a + 1) * float(self.cfg.aisle_pitch),
            float(b + 1) * float(self.cfg.bay_pitch),
            float(t) * float(self.cfg.tier_height),
        )

    def travel_time_3d(
        self,
        xyz_a: tuple[float, float, float],
        xyz_b: tuple[float, float, float],
    ) -> float:
        """Exact symmetric 3D aisle-constrained Manhattan horizontal + hoist vertical time."""
        xa, ya, za = float(xyz_a[0]), float(xyz_a[1]), float(xyz_a[2])
        xb, yb, zb = float(xyz_b[0]), float(xyz_b[1]), float(xyz_b[2])
        if abs(xa - xb) < 1e-9:
            horiz = abs(ya - yb)
        else:
            front = ya + abs(xa - xb) + yb
            rear = (self.y_rear - ya) + abs(xa - xb) + (self.y_rear - yb)
            # Depot sits on y=0 and x=0 (outside rack aisles), so front/rear cross-aisle still holds cleanly.
            horiz = min(front, rear)
        vert = abs(za - zb)
        return float(horiz / float(self.cfg.v_xy) + vert / float(self.cfg.v_z))

    def build_sku_catalog(self, seed: int) -> list[SKUItem]:
        """Create synthetic SKU catalog with Zipf turnover frequencies and discrete weights."""
        rng = random.Random(int(seed) * 1009 + 17)
        n_skus = len(self.all_slots)
        ranks = list(range(n_skus))
        rng.shuffle(ranks)
        raw_turnovers = [1.0 / ((r + 1.0) ** 0.85) for r in ranks]
        tot_t = sum(raw_turnovers)
        turnovers = [t / tot_t for t in raw_turnovers]
        weights = [rng.randint(1, 5) for _ in range(n_skus)]

        log_t = [math.log(max(1e-12, t)) for t in turnovers]
        min_lt, max_lt = min(log_t), max(log_t)
        span_lt = max(1e-9, max_lt - min_lt)
        min_w, max_w = min(weights), max(weights)
        span_w = max(1e-9, float(max_w - min_w))

        catalog: list[SKUItem] = []
        for idx in range(n_skus):
            t_norm = (log_t[idx] - min_lt) / span_lt
            w_norm = float(weights[idx] - min_w) / span_w
            pm = 0.85 * t_norm + 0.15 * w_norm
            catalog.append(
                SKUItem(
                    sku_id=idx,
                    turnover=float(turnovers[idx]),
                    demand_weight=int(weights[idx]),
                    turnover_norm=float(t_norm),
                    demand_norm=float(w_norm),
                    priority_pm=float(pm),
                )
            )
        return catalog

    def generate_instance(
        self,
        catalog: list[SKUItem],
        storage_map: dict[int, tuple[int, int, int]],
        n_stops: int,
        seed: int,
        storage_policy: str,
        instance_id: str,
    ) -> WarehouseVRPInstance:
        """Sample a batch-picking VRP instance weighted by SKU turnover."""
        rng = np.random.default_rng(int(seed))
        probs = np.array([item.turnover for item in catalog], dtype=np.float64)
        probs = probs / probs.sum()
        chosen_skus = rng.choice(len(catalog), size=int(n_stops), replace=False, p=probs)

        coords: list[tuple[float, float, float]] = [self.depot_coord]
        demands: list[float] = [0.0]
        prios: list[float] = [0.0]
        turns: list[float] = [0.0]
        slots: list[tuple[int, int, int]] = [(-1, -1, -1)]
        for sku_idx in chosen_skus.tolist():
            sku = catalog[int(sku_idx)]
            slot = storage_map[sku.sku_id]
            coords.append(self.slot_to_xyz(slot))
            demands.append(float(sku.demand_weight))
            prios.append(float(sku.priority_pm))
            turns.append(float(sku.turnover_norm))
            slots.append(slot)

        coords_arr = np.asarray(coords, dtype=np.float32)
        xa = coords_arr[:, 0:1]
        ya = coords_arr[:, 1:2]
        za = coords_arr[:, 2:3]
        xb = coords_arr[None, :, 0]
        yb = coords_arr[None, :, 1]
        zb = coords_arr[None, :, 2]
        same_aisle = np.abs(xa - xb) < 1e-9
        dx_abs = np.abs(xa - xb)
        front = ya + dx_abs + yb
        rear = (self.y_rear - ya) + dx_abs + (self.y_rear - yb)
        horiz = np.where(same_aisle, np.abs(ya - yb), np.minimum(front, rear))
        vert = np.abs(za - zb)
        dist = (horiz / float(self.cfg.v_xy) + vert / float(self.cfg.v_z)).astype(np.float32)

        return WarehouseVRPInstance(
            instance_id=str(instance_id),
            storage_policy=str(storage_policy),
            coords_3d=coords_arr,
            demands=np.asarray(demands, dtype=np.float32),
            priorities=np.asarray(prios, dtype=np.float32),
            turnovers=np.asarray(turns, dtype=np.float32),
            dist_matrix=dist,
            capacity=float(self.cfg.vehicle_capacity),
            slots=tuple(slots),
        )

    @staticmethod
    def evaluate_tour(inst: WarehouseVRPInstance, route: list[int]) -> dict[str, Any]:
        """Strictly validate a depot-to-depot picking route and compute exact 3D travel time."""
        n_stops = int(inst.coords_3d.shape[0] - 1)
        if not route or int(route[0]) != 0 or int(route[-1]) != 0:
            raise ValueError("Route must start and end at depot 0.")

        visited: list[int] = []
        rem_cap = float(inst.capacity)
        reloads = 0
        total_time = 0.0
        for step_idx in range(1, len(route)):
            prev_nd = int(route[step_idx - 1])
            cur_nd = int(route[step_idx])
            if cur_nd < 0 or cur_nd > n_stops:
                raise ValueError(f"Out-of-range node {cur_nd}")
            total_time += float(inst.dist_matrix[prev_nd, cur_nd])
            if cur_nd == 0:
                rem_cap = float(inst.capacity)
                if step_idx < len(route) - 1:
                    reloads += 1
            else:
                dem = float(inst.demands[cur_nd])
                if dem > rem_cap + 1e-6:
                    raise ValueError(
                        f"Capacity exceeded at stop {cur_nd}: demand {dem} > rem_cap {rem_cap}"
                    )
                rem_cap -= dem
                visited.append(cur_nd)

        if len(visited) != n_stops or set(visited) != set(range(1, n_stops + 1)):
            raise ValueError(
                f"Route must visit each stop 1..{n_stops} exactly once (got {len(visited)}, unique {len(set(visited))})."
            )
        if not math.isfinite(total_time) or total_time <= 0.0:
            raise ValueError(f"Invalid total_time {total_time}")
        return {
            "valid": True,
            "total_time": float(total_time),
            "reloads": int(reloads),
            "n_stops": n_stops,
        }
