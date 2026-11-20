"""Project continuous L1 LiDAR / 2D Nav2 Costmap obstacles onto CVCR topological corridors.

Bridges metric perception (`/scan` LaserScan polar returns or `/local_costmap/costmap`
OccupancyGrid cells) into typed `CorridorObstacleReport` alerts for L2/L3 `PhysicalCVCRCoordinator`.
Includes self-and-teammate footprint exclusion plus multi-frame clear hysteresis so
multi-AMR teammates inside LiDAR field-of-view never trigger spurious road closures.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

from amr_sched.cvcr.physical_bridge import CorridorObstacleReport
from amr_sched.network import Network


def _point_to_segment_distance_and_t(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, float]:
    """Return `(perpendicular_distance_m, clamped_projection_t)` from `P` to segment `A->B`."""
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-12:
        return math.hypot(px - ax, py - ay), 0.0
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy), t


class CostmapCorridorProjector:
    """Metric-to-topological obstacle projector with teammate filtering and clear hysteresis."""

    def __init__(
        self,
        network: Network,
        *,
        corridor_half_width_m: float = 0.65,
        endpoint_exclusion_t: float = 0.10,
        robot_exclusion_radius_m: float = 0.45,
        min_obstacle_hits: int = 2,
        mark_hysteresis_scans: int = 1,
        clear_hysteresis_scans: int = 3,
        default_clear_horizon_steps: int = 20,
    ) -> None:
        self.network = network
        self.corridor_half_width_m = float(corridor_half_width_m)
        self.endpoint_exclusion_t = float(endpoint_exclusion_t)
        self.robot_exclusion_radius_m = float(robot_exclusion_radius_m)
        self.min_obstacle_hits = int(max(1, min_obstacle_hits))
        self.mark_hysteresis_scans = int(max(1, mark_hysteresis_scans))
        self.clear_hysteresis_scans = int(max(1, clear_hysteresis_scans))
        self.default_clear_horizon_steps = int(max(1, default_clear_horizon_steps))

        # Deduplicate undirected corridors (canonical lexicographic `(u, v)`)
        self._undirected_edges: list[tuple[str, str, tuple[float, float], tuple[float, float]]] = []
        seen: set[tuple[str, str]] = set()
        for edge in network.edges.values():
            key = (edge.src, edge.dst) if edge.src <= edge.dst else (edge.dst, edge.src)
            if key in seen or edge.src == edge.dst:
                continue
            seen.add(key)
            na = network.node(key[0])
            nb = network.node(key[1])
            self._undirected_edges.append((key[0], key[1], (float(na.x), float(na.y)), (float(nb.x), float(nb.y))))
        self._undirected_edges.sort(key=lambda item: (item[0], item[1]))

        # Active hysteresis state per canonical corridor key -> remaining clear scans before reopen
        self._active_ttl: dict[tuple[str, str], int] = {}
        self._mark_streak: dict[tuple[str, str], int] = {}
        self._last_hit_counts: dict[tuple[str, str], int] = {}
        self.last_filtered_points: list[tuple[float, float]] = []

    def _filter_teammates(
        self,
        occupied_xy: Sequence[tuple[float, float]],
        robot_positions: Optional[Mapping[str, tuple[float, float]]],
    ) -> list[tuple[float, float]]:
        """Exclude obstacle returns falling inside known tracked AMR robot footprints."""
        if not robot_positions:
            return [(float(x), float(y)) for x, y in occupied_xy]
        r_excl = self.robot_exclusion_radius_m
        r_positions = [(float(rx), float(ry)) for rx, ry in robot_positions.values()]
        filtered: list[tuple[float, float]] = []
        for ox, oy in occupied_xy:
            fx, fy = float(ox), float(oy)
            if any(math.hypot(fx - rx, fy - ry) <= r_excl for rx, ry in r_positions):
                continue
            filtered.append((fx, fy))
        return filtered

    def project_points(
        self,
        occupied_xy: Sequence[tuple[float, float]],
        *,
        robot_positions: Optional[Mapping[str, tuple[float, float]]] = None,
        clear_horizon_steps: Optional[int] = None,
    ) -> list[CorridorObstacleReport]:
        """Project world-frame occupied `(x, y)` obstacle points onto canonical obstructed corridors."""
        pts = self._filter_teammates(occupied_xy, robot_positions)
        self.last_filtered_points = list(pts)
        hit_counts: dict[tuple[str, str], int] = {}

        t_lo = self.endpoint_exclusion_t
        t_hi = 1.0 - self.endpoint_exclusion_t
        half_w = self.corridor_half_width_m

        for u, v, (ax, ay), (bx, by) in self._undirected_edges:
            hits = 0
            for px, py in pts:
                dist, t = _point_to_segment_distance_and_t(px, py, ax, ay, bx, by)
                if t_lo <= t <= t_hi and dist <= half_w:
                    hits += 1
            if hits >= self.min_obstacle_hits:
                hit_counts[(u, v)] = hits

        next_streak: dict[tuple[str, str], int] = {}
        for key in hit_counts:
            next_streak[key] = self._mark_streak.get(key, 0) + 1
        self._mark_streak = next_streak

        next_ttl: dict[tuple[str, str], int] = {}
        for key in set(hit_counts.keys()) | set(self._active_ttl.keys()):
            if key in hit_counts:
                if key in self._active_ttl or self._mark_streak.get(key, 0) >= self.mark_hysteresis_scans:
                    next_ttl[key] = self.clear_hysteresis_scans
                    self._last_hit_counts[key] = hit_counts[key]
            else:
                rem = self._active_ttl.get(key, 0) - 1
                if rem > 0:
                    next_ttl[key] = rem
                else:
                    self._last_hit_counts.pop(key, None)
        self._active_ttl = next_ttl

        hz = int(clear_horizon_steps) if clear_horizon_steps is not None else self.default_clear_horizon_steps
        reports: list[CorridorObstacleReport] = []
        for key in sorted(self._active_ttl.keys()):
            hits = self._last_hit_counts.get(key, self.min_obstacle_hits)
            reports.append(
                CorridorObstacleReport(
                    edge=key,
                    status="blocked",
                    estimated_clear_horizon_steps=hz,
                    source="nav2_costmap",
                    reason=f"costmap_hits:{hits}",
                )
            )
        return reports

    def project_laserscan(
        self,
        robot_pose_xyyaw: tuple[float, float, float],
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
        *,
        range_min: float = 0.12,
        range_max: float = 12.0,
        robot_positions: Optional[Mapping[str, tuple[float, float]]] = None,
        clear_horizon_steps: Optional[int] = None,
    ) -> list[CorridorObstacleReport]:
        """Project a `sensor_msgs/LaserScan` sweep from robot frame into world-frame corridor alerts."""
        rx, ry, ryaw = (float(robot_pose_xyyaw[0]), float(robot_pose_xyyaw[1]), float(robot_pose_xyyaw[2]))
        pts: list[tuple[float, float]] = []
        ang = float(angle_min)
        inc = float(angle_increment)
        for r in ranges:
            rf = float(r)
            if math.isfinite(rf) and range_min <= rf <= range_max:
                world_ang = ryaw + ang
                pts.append((rx + rf * math.cos(world_ang), ry + rf * math.sin(world_ang)))
            ang += inc
        return self.project_points(pts, robot_positions=robot_positions, clear_horizon_steps=clear_horizon_steps)

    def project_occupancy_grid(
        self,
        origin_xy: tuple[float, float],
        resolution: float,
        width: int,
        height: int,
        data: Sequence[int],
        *,
        occ_threshold: int = 65,
        robot_positions: Optional[Mapping[str, tuple[float, float]]] = None,
        clear_horizon_steps: Optional[int] = None,
    ) -> list[CorridorObstacleReport]:
        """Project a `nav_msgs/OccupancyGrid` costmap array into world-frame corridor alerts."""
        ox0, oy0 = float(origin_xy[0]), float(origin_xy[1])
        res = float(resolution)
        pts: list[tuple[float, float]] = []
        limit = min(len(data), int(width) * int(height))
        w = int(width)
        for idx in range(limit):
            if int(data[idx]) >= occ_threshold:
                row = idx // w
                col = idx % w
                pts.append((ox0 + (col + 0.5) * res, oy0 + (row + 0.5) * res))
        return self.project_points(pts, robot_positions=robot_positions, clear_horizon_steps=clear_horizon_steps)

    def active_edges(self) -> list[tuple[str, str]]:
        """Return currently debounced obstructed corridor keys sorted deterministically."""
        return sorted(self._active_ttl.keys())
