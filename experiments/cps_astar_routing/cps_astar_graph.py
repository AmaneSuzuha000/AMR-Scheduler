"""Faithful graph adaptation of CPS-A* (Guo, Huang, Jiang, Xin & D'Ariano, Nov 2026)
for controlled lower-layer routing comparison and cross-layer diagnostic evaluation.

Implements on general directed/bidirectional workshop graphs (`amr_sched.network.Network`)
with bidirectional physical single-lane exclusion (`physical_edge_id`):
  1. `GraphCongestionField`: raw reservation occupancy psi(v, t), graph-hop + temporal
     Gaussian congestion field phi(v, t) [Eqs. 4.2-4.4], look-ahead phi_bar(v, t) [Eqs. 4.7-4.8],
     and delta-aggregated slot congestion field Phi(v, floor(t/delta)) [Section 4.4].
  2. `CPSAStarGraphPlanner`: both `oneshot` [Eqs. 4.5-4.8] and `lifelong` [Eq. 4.10] single-AGV
     best-first / admissible-heuristic A* planners tracking hard-conflict blockers B(i, U).
  3. `CPSAStarGraphEvaluator`: drop-in `CVCREvaluator` subclass running Algorithm 1 bounded
     local priority repair (`R=5`) & incremental K-path field updates inside fixed upper schedulers.
  4. `CPSAStarGreedyMAPDProposer`: Algorithm 2 idle greedy dispatch + lifelong CPS-A* routing,
     strictly reserved for cross-layer diagnostic contrast against coupled FMS-AMR solvers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import heapq
import itertools
import json
import math
import time
from typing import Any, Optional

from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult, physical_edge_id
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.cvcr.snapshot import NativeSnapshot, TaskSnapshot
from amr_sched.network import Network, NodeKind
from amr_sched.planners.base import PlannedPath
from amr_sched.reservation import Interval, ReservationTable
from amr_sched.rust_reservation import RustReservationTable


@dataclass(frozen=True)
class CPSGraphFieldConfig:
    """Kernel and look-ahead hyperparameters mirroring CPS-A* Section 5.2."""

    spatial_radius_L: int = 1
    temporal_radius_T: int = 1
    sigma_l: float = 2.0
    sigma_t: float = 2.0
    lookahead_ks: int = 1
    lookahead_kt: int = 3
    slot_delta: int = 3
    max_track_horizon: int = 220


@dataclass(frozen=True)
class CPSAStarConfig:
    """Algorithmic hyperparameters mirroring CPS-A* One-shot vs. Lifelong settings."""

    mode: str = "lifelong"  # "oneshot" | "lifelong"
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 5.0
    lookahead_omega: float = 50.0  # Active only when mode == "oneshot"
    batch_update_K: int = 20
    retry_limit_R: int = 5
    prio_age_tau: float = 0.25
    prio_cong_kappa: float = 4.0
    max_expansions: int = 12_000
    field: CPSGraphFieldConfig = CPSGraphFieldConfig()


def _probe_blocker_hits(
    reservations: ReservationTable | RustReservationTable,
    node: str,
    span: Interval,
    *,
    exclude_agv: str,
    blocker_hits: dict[str, int],
) -> None:
    for own in reservations.node_conflicts(node, span, exclude_agv=exclude_agv):
        blocker_hits[own] = blocker_hits.get(own, 0) + 1


def _probe_edge_blocker_hits(
    reservations: ReservationTable | RustReservationTable,
    network: Network,
    src: str,
    dst: str,
    span: Interval,
    *,
    exclude_agv: str,
    blocker_hits: dict[str, int],
) -> None:
    phys = physical_edge_id(src, dst)
    occ = list(reservations.edge_conflicts(phys[0], phys[1], span, exclude_agv=exclude_agv))
    if network.has_edge(phys[1], phys[0]) and phys[1] != phys[0]:
        occ.extend(reservations.edge_conflicts(phys[1], phys[0], span, exclude_agv=exclude_agv))
    for own in set(occ):
        blocker_hits[own] = blocker_hits.get(own, 0) + 1


class GraphCongestionField:
    """Reservation-driven 3D graph-spatiotemporal Gaussian & slot congestion field.

    Constructs raw occupancy `psi(v, t)` from scheduled AGV trajectories in the planner's
    reservation state and applies Eq. (4.3)-(4.4) Gaussian smoothing over graph hop radius
    `L` and temporal radius `T`, plus slot aggregation `floor(t / delta)` per Eq. (4.9)-(4.10).
    """

    def __init__(
        self,
        network: Network,
        cfg: CPSGraphFieldConfig = CPSGraphFieldConfig(),
        *,
        t_start: int = 0,
    ) -> None:
        self.network = network
        self.cfg = cfg
        self.t_start = int(t_start)
        self._agv_points: dict[str, set[tuple[str, int]]] = {}
        self._agv_slots: dict[str, dict[tuple[str, int], float]] = {}
        self._psi: dict[tuple[str, int], int] = {}
        self._psi_slot: dict[tuple[str, int], float] = {}
        self._phi_cache: dict[tuple[str, int], float] = {}
        self._slot_phi_cache: dict[tuple[str, int], float] = {}
        self._L: int = max(0, int(cfg.spatial_radius_L))
        self._T: int = max(0, int(cfg.temporal_radius_T))
        self._t_slot_rad: int = max(0, int(math.ceil(float(self._T) / float(max(1, int(cfg.slot_delta))))))

        max_hop = max(int(cfg.spatial_radius_L), int(cfg.lookahead_ks), 0)
        adj: dict[str, set[str]] = {nid: set() for nid in network.nodes}
        for (u, v) in network.edges:
            adj[u].add(v)
            adj[v].add(u)

        self._hop_neighbors: dict[str, list[tuple[str, int]]] = {}
        for src in network.nodes:
            dists: dict[str, int] = {src: 0}
            dq: deque[str] = deque([src])
            while dq:
                cur = dq.popleft()
                cd = dists[cur]
                if cd >= max_hop:
                    continue
                for nxt in adj[cur]:
                    if nxt not in dists:
                        dists[nxt] = cd + 1
                        dq.append(nxt)
            self._hop_neighbors[src] = sorted(dists.items(), key=lambda kv: (kv[1], kv[0]))

        # Precompute 3D Gaussian weights (Eq. 4.3-4.4) per center node
        s_l2 = max(1e-6, 2.0 * float(cfg.sigma_l) * float(cfg.sigma_l))
        s_t2 = max(1e-6, 2.0 * float(cfg.sigma_t) * float(cfg.sigma_t))
        L = max(0, int(cfg.spatial_radius_L))
        T = max(0, int(cfg.temporal_radius_T))
        dt_range = range(-T, T + 1)

        self._gauss_kernel: dict[str, list[tuple[str, int, float]]] = {}
        self._gauss_denom: dict[str, float] = {}
        for v in network.nodes:
            items: list[tuple[str, int, float]] = []
            denom = 0.0
            for u, d_hop in self._hop_neighbors[v]:
                if d_hop > L:
                    continue
                sp_exp = -(float(d_hop * d_hop)) / s_l2
                for dt in dt_range:
                    w = math.exp(sp_exp - float(dt * dt) / s_t2)
                    items.append((u, dt, w))
                    denom += w
            self._gauss_kernel[v] = items
            self._gauss_denom[v] = max(1e-9, denom)

        # Precompute normalized forward look-ahead weights (Eq. 4.7-4.8)
        Ks = max(0, int(cfg.lookahead_ks))
        Kt = max(1, int(cfg.lookahead_kt))
        self._lookahead_kernel: dict[str, list[tuple[str, int, float]]] = {}
        self._lookahead_denom: dict[str, float] = {}
        for v in network.nodes:
            items = []
            denom = 0.0
            for u, d_hop in self._hop_neighbors[v]:
                if d_hop > Ks:
                    continue
                for dt in range(1, Kt + 1):
                    w = math.exp(-0.5 * float(d_hop + dt))
                    items.append((u, dt, w))
                    denom += w
            self._lookahead_kernel[v] = items
            self._lookahead_denom[v] = max(1e-9, denom)

        # Slot-level Gaussian kernel for delta-aggregated lifelong field Phi(v, slot)
        self._slot_kernel: dict[str, list[tuple[str, int, float]]] = {}
        self._slot_denom: dict[str, float] = {}
        delta = max(1, int(cfg.slot_delta))
        t_slot_rad = max(0, int(math.ceil(float(T) / float(delta))))
        for v in network.nodes:
            items = []
            denom = 0.0
            for u, d_hop in self._hop_neighbors[v]:
                if d_hop > L:
                    continue
                sp_exp = -(float(d_hop * d_hop)) / s_l2
                for ds in range(-t_slot_rad, t_slot_rad + 1):
                    dt_eff = float(ds * delta)
                    w = math.exp(sp_exp - (dt_eff * dt_eff) / s_t2)
                    items.append((u, ds, w))
                    denom += w
            self._slot_kernel[v] = items
            self._slot_denom[v] = max(1e-9, denom)

    def clear(self) -> None:
        self._agv_points.clear()
        self._agv_slots.clear()
        self._psi.clear()
        self._psi_slot.clear()
        self._phi_cache.clear()
        self._slot_phi_cache.clear()

    def _invalidate_affected_cache(
        self,
        pts: set[tuple[str, int]] | None,
        slots: dict[tuple[str, int], float] | None,
    ) -> None:
        """Invalidate only Gaussian & slot cache entries within (L, T) of modified occupancy."""
        if pts and self._phi_cache:
            T = self._T
            L = self._L
            phi_c = self._phi_cache
            hops = self._hop_neighbors
            for u_pt, step_pt in pts:
                for v_n, d_hop in hops.get(u_pt, ()):
                    if d_hop > L:
                        break
                    for dt in range(-T, T + 1):
                        phi_c.pop((v_n, step_pt + dt), None)
        if slots and self._slot_phi_cache:
            L = self._L
            tr = self._t_slot_rad
            sphi_c = self._slot_phi_cache
            hops = self._hop_neighbors
            for u_pt, sl_pt in slots:
                for v_n, d_hop in hops.get(u_pt, ()):
                    if d_hop > L:
                        break
                    for ds in range(-tr, tr + 1):
                        sphi_c.pop((v_n, sl_pt + ds), None)

    def remove_agv(self, agv_id: str) -> None:
        pts = self._agv_points.pop(agv_id, None)
        if pts:
            for key in pts:
                nv = self._psi.get(key, 0) - 1
                if nv > 0:
                    self._psi[key] = nv
                else:
                    self._psi.pop(key, None)
        slots = self._agv_slots.pop(agv_id, None)
        if slots:
            for skey, amt in slots.items():
                nv = self._psi_slot.get(skey, 0.0) - amt
                if nv > 1e-9:
                    self._psi_slot[skey] = nv
                else:
                    self._psi_slot.pop(skey, None)
        if pts or slots:
            self._invalidate_affected_cache(pts, slots)

    def add_agv_path(self, agv_id: str, path: list[tuple[str, int]], *, horizon: int | None = None) -> None:
        self.remove_agv(agv_id)
        if not path:
            return
        delta = max(1, int(self.cfg.slot_delta))
        t_cap = self.t_start + int(self.cfg.max_track_horizon)
        if horizon is not None:
            t_cap = min(t_cap, int(horizon))

        pts: set[tuple[str, int]] = set()
        for idx in range(len(path) - 1):
            u_n, u_t = path[idx]
            v_n, v_t = path[idx + 1]
            u_ti = max(self.t_start, int(u_t))
            v_ti = min(t_cap, int(v_t))
            if v_ti < u_ti:
                continue
            if u_n == v_n:
                for step in range(u_ti, min(t_cap + 1, v_ti + 1)):
                    pts.add((u_n, step))
            else:
                pts.add((u_n, u_ti))
                # Intermediate transit dwell contributes light half-way occupancy at corridor head
                if v_ti <= t_cap:
                    pts.add((v_n, v_ti))
        last_n, last_t = path[-1]
        if int(last_t) <= t_cap:
            pts.add((last_n, max(self.t_start, int(last_t))))

        slots: dict[tuple[str, int], float] = {}
        inv_d = 1.0 / float(delta)
        for node, step in pts:
            self._psi[(node, step)] = self._psi.get((node, step), 0) + 1
            skey = (node, int(step) // delta)
            slots[skey] = slots.get(skey, 0.0) + inv_d
        for skey, amt in slots.items():
            self._psi_slot[skey] = self._psi_slot.get(skey, 0.0) + amt

        self._agv_points[agv_id] = pts
        self._agv_slots[agv_id] = slots
        self._invalidate_affected_cache(pts, slots)

    def phi(self, node: str, step: int) -> float:
        """Exact Gaussian-smoothed congestion potential phi(node, step) [Eq. 4.3]."""
        key = (node, int(step))
        cached = self._phi_cache.get(key)
        if cached is not None:
            return cached
        psi = self._psi
        num = 0.0
        for u, dt, w in self._gauss_kernel.get(node, ()):
            occ = psi.get((u, int(step) + dt), 0)
            if occ:
                num += float(occ) * w
        val = num / self._gauss_denom.get(node, 1.0)
        self._phi_cache[key] = val
        return val

    def phi_bar(self, node: str, step: int) -> float:
        """Forward spatiotemporal look-ahead congestion exposure phi_bar(node, step) [Eq. 4.7]."""
        num = 0.0
        for u, dt, w in self._lookahead_kernel.get(node, ()):
            p_val = self.phi(u, int(step) + dt)
            if p_val > 0.0:
                num += p_val * w
        return num / self._lookahead_denom.get(node, 1.0)

    def slot_Phi(self, node: str, step: int) -> float:
        """Gaussian-smoothed slot-level congestion Phi(node, floor(step/delta)) [Section 4.4]."""
        delta = max(1, int(self.cfg.slot_delta))
        slot = int(step) // delta
        key = (node, slot)
        cached = self._slot_phi_cache.get(key)
        if cached is not None:
            return cached
        psi_s = self._psi_slot
        num = 0.0
        for u, ds, w in self._slot_kernel.get(node, ()):
            occ = psi_s.get((u, slot + ds), 0.0)
            if occ > 0.0:
                num += occ * w
        val = num / self._slot_denom.get(node, 1.0)
        self._slot_phi_cache[key] = val
        return val


class CPSAStarGraphPlanner:
    """Faithful CPS-A* single-AGV search on general graphs with blocker tracking."""

    def __init__(
        self,
        network: Network,
        cfg: CPSAStarConfig = CPSAStarConfig(),
        *,
        max_speed: float = 3.0,
        dt: float = 1.0,
    ) -> None:
        self.network = network
        self.cfg = cfg
        self.max_speed = float(max_speed)
        self.dt = float(dt)
        self.last_expansions: int = 0
        self._dist_cache: dict[str, dict[str, float]] = {}
        self._edge_steps_map: dict[tuple[str, str], int] = {
            (u, v): max(1, int(math.ceil(e.traversal_time(self.max_speed) / self.dt)))
            for (u, v), e in network.edges.items()
        }

    def _heuristic_table(self, goal: str) -> dict[str, float]:
        if goal in self._dist_cache:
            return self._dist_cache[goal]
        rev: dict[str, list[tuple[str, float]]] = {n: [] for n in self.network.nodes}
        for (_, _), e in self.network.edges.items():
            rev[e.dst].append((e.src, e.traversal_time(self.max_speed)))
        dist: dict[str, float] = {goal: 0.0}
        heap: list[tuple[float, str]] = [(0.0, goal)]
        done: set[str] = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for v, w in rev[u]:
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self._dist_cache[goal] = dist
        return dist

    def _turn_penalty(self, prev_node: Optional[str], cur_node: str, nxt_node: str) -> float:
        if prev_node is None or prev_node == cur_node or cur_node == nxt_node:
            return 0.0
        if nxt_node == prev_node:
            return 1.0
        n0 = self.network.nodes.get(prev_node)
        n1 = self.network.nodes.get(cur_node)
        n2 = self.network.nodes.get(nxt_node)
        if n0 is None or n1 is None or n2 is None:
            return 0.0
        dx1, dy1 = float(n1.x) - float(n0.x), float(n1.y) - float(n0.y)
        dx2, dy2 = float(n2.x) - float(n1.x), float(n2.y) - float(n1.y)
        l1 = math.hypot(dx1, dy1)
        l2 = math.hypot(dx2, dy2)
        if l1 <= 1e-6 or l2 <= 1e-6:
            return 0.0
        cos_ang = (dx1 * dx2 + dy1 * dy2) / (l1 * l2)
        return 1.0 if cos_ang < 0.707 else 0.0

    def plan_leg(
        self,
        start: str,
        goal: str,
        start_step: int,
        reservations: ReservationTable | RustReservationTable,
        field: GraphCongestionField,
        agv_id: str,
        horizon: int,
        *,
        prev_arrival_node: Optional[str] = None,
        goal_park_until: Optional[int] = None,
        goal_park_steps: int = 0,
        max_expansions_override: Optional[int] = None,
    ) -> tuple[PlannedPath | None, dict[str, int]]:
        """Plan one spatiotemporal leg, returning `(PlannedPath | None, blocker_hits)`."""
        if start == goal:
            eff_park = int(goal_park_until) if goal_park_until is not None else min(
                int(horizon), int(start_step) + 1 + max(0, int(goal_park_steps))
            )
            if eff_park <= int(start_step) + 1:
                self.last_expansions = 0
                return PlannedPath(nodes=[], arrival_steps=[], wait_steps=0), {}
            cap_g = self.network.node(goal).capacity
            if reservations.is_node_free(
                goal, Interval(int(start_step) + 1, eff_park), cap_g, exclude_agv=agv_id
            ):
                self.last_expansions = 0
                return PlannedPath(nodes=[], arrival_steps=[], wait_steps=0), {}
        h_tbl = self._heuristic_table(goal)
        if start not in h_tbl:
            self.last_expansions = 0
            return None, {}

        mode = str(self.cfg.mode).lower()
        alpha = float(self.cfg.alpha)
        beta = float(self.cfg.beta)
        gamma = float(self.cfg.gamma)
        omega = float(self.cfg.lookahead_omega) if mode == "oneshot" else 0.0

        def h_base(node: str) -> float:
            v = h_tbl.get(node)
            return 1e9 if v is None else alpha * float(math.floor(v / self.dt))

        def h_total(node: str, step: int) -> float:
            hb = h_base(node)
            if omega > 0.0:
                return hb + omega * field.phi_bar(node, step)
            return hb

        counter = itertools.count()
        # State key preserves arrival heading parent for C_t turning awareness
        init_state = (start, int(start_step), prev_arrival_node)
        frontier: list[tuple[float, float, int, int, str, int, Optional[str]]] = [
            (h_total(start, int(start_step)), 0.0, 0, next(counter), start, int(start_step), prev_arrival_node)
        ]
        best_cost: dict[tuple[str, int, Optional[str]], float] = {init_state: 0.0}
        parent: dict[tuple[str, int, Optional[str]], tuple[str, int, Optional[str]] | None] = {init_state: None}
        wait_cnt: dict[tuple[str, int, Optional[str]], int] = {init_state: 0}
        blocker_hits: dict[str, int] = {}

        goal_state: tuple[str, int, Optional[str]] | None = None
        expansions = 0
        max_exp_lim = int(max_expansions_override) if max_expansions_override is not None else int(self.cfg.max_expansions)

        while frontier:
            expansions += 1
            if expansions > max_exp_lim:
                break
            _, cur_g, w_c, _, node, t, p_node = heapq.heappop(frontier)
            if t > horizon:
                continue
            state = (node, t, p_node)
            if best_cost.get(state, 1e18) + 1e-9 < cur_g:
                continue
            if node == goal:
                eff_park = int(goal_park_until) if goal_park_until is not None else min(
                    int(horizon), t + 1 + max(0, int(goal_park_steps))
                )
                if eff_park > t + 1:
                    park_span = Interval(t + 1, eff_park)
                    cap_g = self.network.node(goal).capacity
                    if reservations.is_node_free(goal, park_span, cap_g, exclude_agv=agv_id):
                        goal_state = state
                        break
                    _probe_blocker_hits(reservations, goal, park_span, exclude_agv=agv_id, blocker_hits=blocker_hits)
                else:
                    goal_state = state
                    break

            # Action 1: Wait (u == v) occupies `node` at t + 1 -> Interval(t + 1, t + 2)
            if t + 1 <= horizon:
                span = Interval(t + 1, t + 2)
                cap = self.network.node(node).capacity
                if reservations.is_node_free(node, span, cap, exclude_agv=agv_id):
                    cong_pen = (
                        gamma * field.phi(node, t + 1)
                        if mode == "oneshot"
                        else gamma * field.slot_Phi(node, t + 1)
                    )
                    ng = cur_g + alpha * 1.0 + cong_pen
                    ns = (node, t + 1, p_node)
                    if ng + 1e-9 < best_cost.get(ns, 1e18):
                        best_cost[ns] = ng
                        parent[ns] = state
                        wait_cnt[ns] = w_c + 1
                        heapq.heappush(frontier, (ng + h_total(node, t + 1), ng, w_c + 1, next(counter), node, t + 1, p_node))
                else:
                    _probe_blocker_hits(reservations, node, span, exclude_agv=agv_id, blocker_hits=blocker_hits)

            # Action 2: Edge move (u -> v)
            for e in self.network.out_edges(node):
                n_steps = self._edge_steps_map[(e.src, e.dst)]
                arr = t + n_steps
                if arr > horizon:
                    continue
                edge_span = Interval(t, arr)
                phys = physical_edge_id(e.src, e.dst)
                edge_cap = (
                    min(e.capacity, self.network.edge(e.dst, e.src).capacity)
                    if self.network.has_edge(e.dst, e.src)
                    else e.capacity
                )
                occ = reservations.edge_conflicts(phys[0], phys[1], edge_span, exclude_agv=agv_id)
                if self.network.has_edge(phys[1], phys[0]) and phys[1] != phys[0]:
                    occ = list(occ) + list(reservations.edge_conflicts(phys[1], phys[0], edge_span, exclude_agv=agv_id))
                if len(set(occ)) >= edge_cap:
                    for own in set(occ):
                        blocker_hits[own] = blocker_hits.get(own, 0) + 1
                    continue
                arr_span = Interval(arr, arr + 1)
                dst_cap = self.network.node(e.dst).capacity
                if not reservations.is_node_free(e.dst, arr_span, dst_cap, exclude_agv=agv_id):
                    _probe_blocker_hits(reservations, e.dst, arr_span, exclude_agv=agv_id, blocker_hits=blocker_hits)
                    continue

                turn_c = beta * self._turn_penalty(p_node, node, e.dst)
                cong_move = (gamma * field.phi(e.dst, arr)) if mode == "oneshot" else 0.0
                ng = cur_g + alpha * float(n_steps) + turn_c + cong_move
                ns = (e.dst, arr, node)
                if ng + 1e-9 < best_cost.get(ns, 1e18):
                    best_cost[ns] = ng
                    parent[ns] = state
                    wait_cnt[ns] = w_c
                    heapq.heappush(frontier, (ng + h_total(e.dst, arr), ng, w_c, next(counter), e.dst, arr, node))

        self.last_expansions = expansions
        if goal_state is None:
            return None, blocker_hits

        chain: list[tuple[str, int]] = []
        cur: tuple[str, int, Optional[str]] | None = goal_state
        while cur is not None:
            chain.append((cur[0], cur[1]))
            cur = parent[cur]
        chain.reverse()

        collapsed: list[tuple[str, int]] = []
        for nd, st in chain:
            if not collapsed or collapsed[-1][0] != nd:
                collapsed.append((nd, st))
        return PlannedPath(
            nodes=[c[0] for c in collapsed[1:]],
            arrival_steps=[c[1] for c in collapsed[1:]],
            wait_steps=wait_cnt[goal_state],
        ), blocker_hits


class CPSAStarGraphEvaluator(CVCREvaluator):
    """Controlled lower-layer CPS-A* routing evaluator inside fixed CoRe-Sched schedulers.

    Keeps the exact same upper-layer objective (`_compute_objective`), kitting propagation,
    and B2/B1/A1 signature while replacing hard-only `SpaceTimeAStar` with faithful CPS-A*
    congestion field search (`oneshot` or `lifelong`) + Algorithm 1 local priority repair.
    """

    def __init__(
        self,
        network: Network,
        snapshot: NativeSnapshot,
        cps_cfg: CPSAStarConfig = CPSAStarConfig(),
        **kwargs: Any,
    ) -> None:
        # Force Python reservation visibility cleanly unless Rust shadow isn't needed;
        # we still accept RustReservationTable seamlessly if caller didn't force False.
        kwargs.setdefault("use_rust", False)
        super().__init__(network=network, snapshot=snapshot, **kwargs)
        self.cps_cfg = cps_cfg
        self._cps_planner = CPSAStarGraphPlanner(
            network=network,
            cfg=cps_cfg,
            max_speed=self.max_speed,
            dt=self.dt,
        )
        self._last_blocker_hits: dict[str, int] = {}
        self.cps_repairs_attempted: int = 0
        self.cps_repairs_succeeded: int = 0

    def _route_robot_with_field(
        self,
        agv_id: str,
        tasks: list[TaskSnapshot],
        res_table: ReservationTable | RustReservationTable,
        field: GraphCongestionField,
        op_priority: dict[str, int],
        op_res_ready_hint: dict[str, int] | None,
        op_bay_overrides: Optional[dict[str, str]],
        task_order_override: Optional[dict[str, list[str]]],
        *,
        collect_res: bool,
    ) -> tuple[
        list[tuple[str, int]],
        list[str],
        dict[str, int],
        list[tuple[str, Interval]],
        list[tuple[tuple[str, str], Interval]],
        bool,
        dict[str, int],
    ]:
        self._active_field = field
        self._last_blocker_hits = {}
        try:
            res = self._route_single_robot(
                agv_id,
                tasks,
                res_table,
                op_priority,
                op_res_ready_hint,
                op_bay_overrides,
                task_order_override=task_order_override,
                collect_res=collect_res,
            )
            return (*res, dict(self._last_blocker_hits))
        finally:
            self._active_field = None  # type: ignore[assignment]

    def _space_time_route(
        self,
        start_node: str,
        goal_node: str,
        start_step: int,
        reservations: ReservationTable | RustReservationTable,
        agv_id: str,
        max_horizon: Optional[int] = None,
    ) -> list[tuple[str, int]] | None:
        if start_node == goal_node:
            return [(start_node, start_step)]
        field = getattr(self, "_active_field", None)
        if field is None:
            return super()._space_time_route(start_node, goal_node, start_step, reservations, agv_id, max_horizon)
        horizon = (
            max_horizon
            if max_horizon is not None
            else min(self.sim_horizon - start_step, max(450, self.sim_horizon // 2))
        )
        limit_step = min(self.sim_horizon, start_step + max(1, horizon))
        planned, hits = self._cps_planner.plan_leg(
            start_node,
            goal_node,
            start_step,
            reservations,
            field,
            agv_id,
            limit_step,
        )
        for own, c in hits.items():
            self._last_blocker_hits[own] = self._last_blocker_hits.get(own, 0) + c
        if planned is None:
            return None
        chain: list[tuple[str, int]] = [(start_node, start_step)]
        cur_n = start_node
        cur_t = start_step
        for nxt_n, arr_t in zip(planned.nodes, planned.arrival_steps):
            move_steps = self._edge_steps(cur_n, nxt_n)
            dep_t = arr_t - move_steps
            if dep_t > cur_t:
                chain.append((cur_n, dep_t))
            chain.append((nxt_n, arr_t))
            cur_n = nxt_n
            cur_t = arr_t
        return chain

    def _dynamic_priority_sort(
        self,
        rids: list[str],
        by_robot: dict[str, list[TaskSnapshot]],
        op_priority: dict[str, int],
        field: GraphCongestionField,
    ) -> list[str]:
        if str(self.cps_cfg.mode).lower() != "lifelong":
            return sorted(rids, key=lambda r: self._robot_plan_sort_key(r, by_robot[r], op_priority))

        tau = float(self.cps_cfg.prio_age_tau)
        kappa = float(self.cps_cfg.prio_cong_kappa)

        def _key(rid: str) -> tuple[int, float, str]:
            tasks = by_robot[rid]
            if not tasks:
                return (0, 0.0, rid)
            base_k = self._robot_plan_sort_key(rid, tasks, op_priority)
            max_age = max(max(0, self.t_commit - int(t.release_step)) for t in tasks)
            cur_n = self.robots_by_id[rid].cur_node
            local_cong = field.slot_Phi(cur_n, self.t_commit)
            # Lower score plans earlier (Eq. 4.9 inverted sign for min-sort)
            score = float(base_k[1]) - tau * float(max_age) - kappa * float(local_cong)
            return (1, score, rid)

        return sorted(rids, key=_key)

    def _solve_cps_batch(
        self,
        rids_to_route: list[str],
        by_robot: dict[str, list[TaskSnapshot]],
        res_table: ReservationTable | RustReservationTable,
        field: GraphCongestionField,
        op_priority: dict[str, int],
        op_res_ready_hint: dict[str, int] | None,
        op_bay_overrides: Optional[dict[str, str]],
        task_order_override: Optional[dict[str, list[str]]],
        *,
        collect_res: bool,
    ) -> tuple[
        dict[str, list[tuple[str, int]]],
        dict[str, list[str]],
        dict[str, int],
        dict[str, list[tuple[str, Interval]]],
        dict[str, list[tuple[tuple[str, str], Interval]]],
        bool,
    ]:
        """Algorithm 1: sequential CPS-A* + incremental K-batch congestion update + bounded rollback."""
        q: deque[str] = deque(self._dynamic_priority_sort(rids_to_route, by_robot, op_priority, field))
        uncommitted: dict[str, tuple[list[tuple[str, int]], list[str], dict[str, int], list[tuple[str, Interval]], list[tuple[tuple[str, str], Interval]]]] = {}
        committed_order: list[str] = []
        retries: dict[str, int] = {r: 0 for r in rids_to_route}
        R_lim = max(0, int(self.cps_cfg.retry_limit_R))
        K_batch = max(1, int(self.cps_cfg.batch_update_K))

        out_paths: dict[str, list[tuple[str, int]]] = {}
        out_seqs: dict[str, list[str]] = {}
        out_unloads: dict[str, int] = {}
        out_nres: dict[str, list[tuple[str, Interval]]] = {}
        out_eres: dict[str, list[tuple[tuple[str, str], Interval]]] = {}

        while q:
            rid = q.popleft()
            path, seq, unloads, n_res, e_res, ok, hits = self._route_robot_with_field(
                rid,
                by_robot[rid],
                res_table,
                field,
                op_priority,
                op_res_ready_hint,
                op_bay_overrides,
                task_order_override,
                collect_res=collect_res,
            )
            if ok:
                uncommitted[rid] = (path, seq, unloads, n_res, e_res)
                if len(uncommitted) >= K_batch:
                    for crid, (cp, cseq, cun, cnr, cer) in uncommitted.items():
                        out_paths[crid] = cp
                        out_seqs[crid] = cseq
                        out_unloads.update(cun)
                        out_nres[crid] = cnr
                        out_eres[crid] = cer
                        committed_order.append(crid)
                        field.add_agv_path(crid, cp, horizon=self.sim_horizon)
                    uncommitted.clear()
                continue

            if retries[rid] >= R_lim or not uncommitted:
                out_paths[rid] = path
                out_seqs[rid] = seq
                out_unloads.update(unloads)
                out_nres[rid] = n_res
                out_eres[rid] = e_res
                return out_paths, out_seqs, out_unloads, out_nres, out_eres, False

            # Select main blocker j in U maximizing hard conflicts encountered during rid's failed search
            self.cps_repairs_attempted += 1
            active_blockers = [(hits.get(j, 0), j) for j in uncommitted]
            _, blocker_j = max(active_blockers, key=lambda kv: (kv[0], kv[1]))
            uncommitted.pop(blocker_j, None)
            res_table.release_all(blocker_j)
            field.remove_agv(blocker_j)
            retries[rid] += 1
            q.appendleft(rid)
            q.append(blocker_j)

        for crid, (cp, cseq, cun, cnr, cer) in uncommitted.items():
            out_paths[crid] = cp
            out_seqs[crid] = cseq
            out_unloads.update(cun)
            out_nres[crid] = cnr
            out_eres[crid] = cer
            field.add_agv_path(crid, cp, horizon=self.sim_horizon)
        uncommitted.clear()
        if self.cps_repairs_attempted > 0:
            self.cps_repairs_succeeded += 1
        return out_paths, out_seqs, out_unloads, out_nres, out_eres, True

    def _eval_a1(
        self,
        assignment: dict[str, str],
        op_priority: dict[str, int],
        op_res_ready_hint: Optional[dict[str, int]] = None,
        op_bay_overrides: Optional[dict[str, str]] = None,
        op_crew_overrides: Optional[dict[str, int]] = None,
        task_order_override: Optional[dict[str, list[str]]] = None,
    ) -> EvalResult:
        self.a1_calls += 1
        self.__class__._GLOBAL_TELEMETRY["a1_calls"] = int(self.__class__._GLOBAL_TELEMETRY["a1_calls"]) + 1
        res_table = self._new_res_table()
        field = GraphCongestionField(self.network, self.cps_cfg.field, t_start=self.t_commit)
        rids = sorted(r.robot_id for r in self.snapshot.robots)
        by_robot: dict[str, list[TaskSnapshot]] = {rid: [] for rid in rids}
        for tid, t in self.tasks_by_id.items():
            if t.status.lower() != "completed":
                aid = assignment.get(tid)
                if aid in by_robot:
                    by_robot[aid].append(t)

        paths, seqs, unloads, _, _, all_ok = self._solve_cps_batch(
            rids, by_robot, res_table, field, op_priority, op_res_ready_hint, op_bay_overrides, task_order_override, collect_res=False
        )
        all_unloads = dict(self.delivered_unloads)
        all_unloads.update(unloads)
        op_s, op_e, op_rr, obj, starv = self._compute_objective(
            assignment, all_unloads, op_priority, all_ok, op_bay_overrides, op_crew_overrides
        )
        return EvalResult(paths, seqs, all_unloads, op_s, op_e, op_rr, obj, all_ok, starv, 0.0, "A1")

    def _eval_b1(
        self,
        assignment: dict[str, str],
        op_priority: dict[str, int],
        op_res_ready_hint: Optional[dict[str, int]] = None,
        op_bay_overrides: Optional[dict[str, str]] = None,
        op_crew_overrides: Optional[dict[str, int]] = None,
        task_order_override: Optional[dict[str, list[str]]] = None,
    ) -> EvalResult:
        self.b1_calls += 1
        self.__class__._GLOBAL_TELEMETRY["b1_calls"] = int(self.__class__._GLOBAL_TELEMETRY["b1_calls"]) + 1
        res_table = self._new_res_table()
        field = GraphCongestionField(self.network, self.cps_cfg.field, t_start=self.t_commit)
        rids_to_route = sorted(r.robot_id for r in self.snapshot.robots)
        by_robot: dict[str, list[TaskSnapshot]] = {rid: [] for rid in rids_to_route}
        for tid, t in self.tasks_by_id.items():
            if t.status.lower() != "completed":
                aid = assignment.get(tid)
                if aid in by_robot:
                    by_robot[aid].append(t)

        paths, seqs, unloads, nres, eres, all_ok = self._solve_cps_batch(
            rids_to_route, by_robot, res_table, field, op_priority, op_res_ready_hint, op_bay_overrides, task_order_override, collect_res=True
        )
        all_unloads = dict(self.delivered_unloads)
        all_unloads.update(unloads)
        op_s, op_e, op_rr, obj, starv = self._compute_objective(
            assignment, all_unloads, op_priority, all_ok, op_bay_overrides, op_crew_overrides
        )
        res_obj = EvalResult(paths, seqs, all_unloads, op_s, op_e, op_rr, obj, all_ok, starv, 0.0, "B1")
        if all_ok and self.mode == "B2":
            self._pending_b1_seed = (res_obj, res_table, nres, eres)
        return res_obj

    def _eval_b2(
        self,
        assignment: dict[str, str],
        op_priority: dict[str, int],
        op_res_ready_hint: Optional[dict[str, int]] = None,
        op_bay_overrides: Optional[dict[str, str]] = None,
        op_crew_overrides: Optional[dict[str, int]] = None,
        task_order_override: Optional[dict[str, list[str]]] = None,
    ) -> EvalResult:
        # Delegate through _eval_b2 context matching by reusing base checks, then CPS-A* incremental batch
        prio_key = tuple(sorted(op_priority.items()))
        hint_key = tuple(sorted(op_res_ready_hint.items())) if op_res_ready_hint is not None else None
        ctx_invalid = (
            self.base_ctx_key is None
            or (self.base_ctx_key[1] != hint_key if self.seq_diff_b2 else self.base_ctx_key[:2] != (prio_key, hint_key))
            or self.base_traffic_sig != self._traffic_sig
            or not self.robot_cache
            or self._inc_res_table is None
        )
        if ctx_invalid:
            return self._eval_b1(assignment, op_priority, op_res_ready_hint, op_bay_overrides, op_crew_overrides, task_order_override)

        # Compute changed_rids exactly as CVCREvaluator._eval_b2
        bay_key = tuple(sorted(op_bay_overrides.items())) if op_bay_overrides else None
        order_key = (
            tuple((k, tuple(v)) for k, v in sorted(task_order_override.items()))
            if task_order_override
            else None
        )
        changed_set = (
            {assignment[tid] for tid in assignment if assignment[tid] != self.base_assign.get(tid)}
            | {self.base_assign[tid] for tid in self.base_assign if assignment.get(tid) != self.base_assign[tid]}
        )
        if bay_key != self.base_ctx_key[2]:
            new_bays = op_bay_overrides or {}
            changed_ops = {
                op_id
                for op_id in (set(self.base_bay_overrides) | set(new_bays))
                if self.base_bay_overrides.get(op_id, self.ops_by_id[op_id].bay_id)
                != new_bays.get(op_id, self.ops_by_id[op_id].bay_id)
            }
            changed_set |= {
                assignment.get(t.task_id)
                for t in self.tasks_by_id.values()
                if t.kit_op_id in changed_ops and t.status.lower() in ("pending", "assigned") and assignment.get(t.task_id)
            }
        base_order_key = self.base_ctx_key[3] if len(self.base_ctx_key) > 3 else None
        if not self.seq_diff_b2 and order_key != base_order_key:
            new_orders = task_order_override or {}
            changed_set |= {
                rid
                for rid in (set(self.base_task_order) | set(new_orders))
                if self.base_task_order.get(rid) != new_orders.get(rid)
            }

        changed_rids = sorted(changed_set)
        if not changed_rids or len(changed_rids) >= len(self.snapshot.robots):
            return self._eval_b1(assignment, op_priority, op_res_ready_hint, op_bay_overrides, op_crew_overrides, task_order_override)

        res_table = self._inc_res_table
        assert res_table is not None
        for rid in changed_rids:
            res_table.release_all(rid)

        field = GraphCongestionField(self.network, self.cps_cfg.field, t_start=self.t_commit)
        order_unload_ticks = dict(self.delivered_unloads)
        robot_paths: dict[str, list[tuple[str, int]]] = {}
        robot_order_seq: dict[str, list[str]] = {}
        for r in self.snapshot.robots:
            rid = r.robot_id
            if rid not in changed_rids and rid in self.robot_cache:
                rc = self.robot_cache[rid]
                robot_paths[rid] = rc["path"]
                robot_order_seq[rid] = rc["seq"]
                order_unload_ticks.update(rc["unloads"])
                field.add_agv_path(rid, rc["path"], horizon=self.sim_horizon)

        by_robot: dict[str, list[TaskSnapshot]] = {rid: [] for rid in changed_rids}
        for tid, t in self.tasks_by_id.items():
            if t.status.lower() != "completed":
                aid = assignment.get(tid)
                if aid in by_robot:
                    by_robot[aid].append(t)

        c_paths, c_seqs, c_unloads, _, _, all_ok = self._solve_cps_batch(
            changed_rids, by_robot, res_table, field, op_priority, op_res_ready_hint, op_bay_overrides, task_order_override, collect_res=False
        )
        robot_paths.update(c_paths)
        robot_order_seq.update(c_seqs)
        order_unload_ticks.update(c_unloads)
        self._restore_inc_table(changed_rids)

        op_s, op_e, op_rr, obj, starv = self._compute_objective(
            assignment, order_unload_ticks, op_priority, all_ok, op_bay_overrides, op_crew_overrides
        )
        return EvalResult(robot_paths, robot_order_seq, order_unload_ticks, op_s, op_e, op_rr, obj, all_ok, starv, 0.0, "B2")


class CPSAStarGreedyMAPDProposer:
    """Supplementary Cross-Layer Diagnostic Baseline (CPS-A* Algorithm 2 greedy dispatch).

    Strictly NOT for Table 1 full-stack strawman comparison; isolates the degradation of pure
    lower-layer congestion-aware MAPD without kitting-DAG/workstation/churn coupling.
    """

    def __init__(self, cps_cfg: CPSAStarConfig = CPSAStarConfig(mode="lifelong")) -> None:
        self.cps_cfg = cps_cfg

    def __call__(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        budget_ms: float = 35.0,
    ) -> CandidateProposal:
        t0 = _cvcr_now_s()
        clf = classify_commitments(snapshot)
        healthy = sorted(
            r.robot_id for r in snapshot.robots
            if r.robot_id not in clf.disturbed_robots and r.state not in ("faulted", "blocked")
        ) or sorted(r.robot_id for r in snapshot.robots if r.state != "faulted")

        # Pure MAPD greedy nearest-idle dispatch (Algorithm 2: AssignTasks(D, I^idle))
        ev = CPSAStarGraphEvaluator(network, snapshot, cps_cfg=self.cps_cfg, mode="B1", use_rust=False)
        rob_pos = {r.robot_id: r.cur_node for r in snapshot.robots}
        load = {r: 0 for r in healthy}
        assign: dict[str, str] = {}
        for t in snapshot.tasks:
            if t.status.lower() == "completed":
                continue
            if t.assigned_agv is not None and (t.assigned_agv in healthy or t.status.lower() in ("picked", "loading", "to_dropoff")):
                assign[t.task_id] = t.assigned_agv
                if t.assigned_agv in load:
                    load[t.assigned_agv] += 1

        unassigned = [t for t in snapshot.tasks if t.status.lower() != "completed" and t.task_id not in assign]
        unassigned.sort(key=lambda t: (int(t.release_step), str(t.task_id)))
        for t in unassigned:
            h_tbl = ev._heuristic_table(t.pickup)
            best_r = min(
                healthy,
                key=lambda r: (load.get(r, 0), float(h_tbl.get(rob_pos[r], 1e6)), r),
            )
            assign[t.task_id] = best_r
            load[best_r] = load.get(best_r, 0) + 1

        op_prio = {op.op_id: max(int(snapshot.tick), int(op.due_step) - int(op.proc_steps)) for op in snapshot.operations}
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
        best_eval = ev.evaluate(assign, op_priority=op_prio, op_bay_overrides=best_bays, op_crew_overrides=best_crews)

        snap_digest = snapshot.compute_digest()
        snap_by_id = {t.task_id: t for t in snapshot.tasks}
        commit_assign: dict[str, str] = {}
        affected_tasks: list[str] = []
        for tid, aid in assign.items():
            t0_obj = snap_by_id.get(tid)
            if t0_obj is None:
                continue
            orig = t0_obj.assigned_agv
            is_dist = orig is not None and orig in clf.disturbed_robots
            is_rel = orig is not None and orig != aid and tid in clf.releasable_tasks
            is_urg = orig is None and getattr(t0_obj, "is_urgent", False)
            if orig is not None or is_urg:
                commit_assign[tid] = orig if (orig is not None and not is_rel and not is_dist) else aid
                if commit_assign[tid] != orig:
                    affected_tasks.append(tid)
        affected_tasks.sort()
        elapsed_ms = (_cvcr_now_s() - t0) * 1000.0
        digest = hashlib.sha256(
            json.dumps({"policy": "CPSAStar_GreedyMAPD", "snap": snap_digest, "a": sorted(commit_assign.items())}, sort_keys=True).encode()
        ).hexdigest()
        return CandidateProposal(
            schema="amr-lgcvcr.proposal.v1",
            policy_id="CPSAStar_GreedyMAPD",
            snapshot_digest=snap_digest,
            affected_robots=sorted({commit_assign[t] for t in affected_tasks}),
            affected_tasks=affected_tasks,
            affected_operations=[],
            proposed_assignment=commit_assign,
            proposed_routes=best_eval.robot_paths if best_eval.valid else {},
            proposed_bays=best_bays,
            proposed_crews=best_crews,
            predicted_obj=float(best_eval.objective),
            predicted_makespan=max(best_eval.op_end.values(), default=snapshot.tick),
            candidate_digest=digest,
            telemetry={"elapsed_ms": round(elapsed_ms, 3), "deadline_met": elapsed_ms <= budget_ms},
            proposed_op_priorities=op_prio,
        )
