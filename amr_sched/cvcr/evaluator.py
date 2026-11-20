"""CVCR Incremental Evaluator (A1 / B1 / B2) for AMR-Scheduler.

Implements the exact factorial decomposition from RAIL-CVCR with optional native
Rust acceleration (`RustSpaceTimeAStarPlanner` + `RustReservationTable`):
- `A1`: Full replanning of ALL robots + Full reservation build (pre-CVCR baseline)
- `B1`: Preserve unchanged robots' incumbent paths + Full reservation rebuild from paths
- `B2`: Preserve unchanged robots' incumbent paths + Persistent Incremental Reservation Cache (E2)
       (releases and re-routes ONLY changed_rids in-place, then restores/promotes)

Supports:
- Production kitting, predecessor, bay/crew/tool mutex propagation
- Joint production bay/crew override evaluation for uncommitted operations
- Active fleet admission gating (`admitted_agvs`) while retaining parked-AGV occupancy
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Any, Optional

from amr_sched.cvcr.clock import (
    is_virtual_clock_active,
    now_s as _cvcr_now_s,
    record_work_ms as _cvcr_record_work_ms,
)
from amr_sched.cvcr.snapshot import (
    NativeSnapshot,
    TaskSnapshot,
    align_start_to_windows,
    op_predecessors,
)
from amr_sched.network import Network, NodeKind
from amr_sched.planners.rust_spacetime_astar import (
    RustSpaceTimeAStarPlanner,
    is_rust_planner_available,
)
from amr_sched.reservation import Interval, ReservationTable
from amr_sched.rust_reservation import RustReservationTable, is_rust_core_available


def physical_edge_id(src: str, dst: str) -> tuple[str, str]:
    """Canonical unordered physical corridor identifier."""
    return (src, dst) if src <= dst else (dst, src)


@dataclass
class EvalResult:
    """Result returned by CVCREvaluator."""

    robot_paths: dict[str, list[tuple[str, int]]]
    robot_order_seq: dict[str, list[str]]
    order_unload_ticks: dict[str, int]
    op_start: dict[str, int]
    op_end: dict[str, int]
    op_res_ready: dict[str, int]
    objective: float
    valid: bool
    starvation_ticks: int
    eval_latency_ms: float
    mode: str  # "A1" | "B1" | "B2"


class CVCREvaluator:
    """Spatiotemporal Evaluator with Rust acceleration and Incremental Reservation Caching."""

    # Bounded static graph/planner cache per (network_id, max_speed, dt) to avoid rebuilding Rust graphs across ticks.
    _STATIC_NET_CACHE: dict[tuple[int, float, float], dict[str, Any]] = {}

    @classmethod
    def _get_static_cache(
        cls,
        network: Network,
        max_speed: float,
        dt: float,
        use_rust: bool,
    ) -> dict[str, Any]:
        key = (id(network), float(max_speed), float(dt))
        entry = cls._STATIC_NET_CACHE.get(key)
        if entry is not None and entry.get("network") is network:
            if use_rust and entry.get("planner") is None and is_rust_planner_available():
                entry["planner"] = RustSpaceTimeAStarPlanner(max_speed=max_speed, dt=dt, max_expansions=3_000)
            return entry

        if len(cls._STATIC_NET_CACHE) >= 8:
            cls._STATIC_NET_CACHE.clear()

        rev: dict[str, list[tuple[str, float]]] = {n: [] for n in network.nodes}
        edge_steps_map: dict[tuple[str, str], int] = {}
        for (src, dst), e in network.edges.items():
            rev[e.dst].append((e.src, e.traversal_time(max_speed)))
            edge_steps_map[(src, dst)] = max(1, int(math.ceil(e.length / min(max_speed, e.speed_limit) / dt)))

        haven_nodes = [
            nd.node_id
            for nd in sorted(network.nodes.values(), key=lambda x: (0 if x.kind == NodeKind.BUFFER else 1, x.node_id))
            if nd.kind in (NodeKind.BUFFER, NodeKind.AISLE)
        ]
        planner = (
            RustSpaceTimeAStarPlanner(max_speed=max_speed, dt=dt, max_expansions=3_000)
            if (use_rust and is_rust_planner_available())
            else None
        )
        entry = {
            "network": network,
            "planner": planner,
            "rev_graph": rev,
            "dist_cache": {},
            "edge_steps": edge_steps_map,
            "haven_nodes": haven_nodes,
        }
        cls._STATIC_NET_CACHE[key] = entry
        return entry

    def __init__(
        self,
        network: Network,
        snapshot: NativeSnapshot,
        mode: str = "B2",  # "A1" | "B1" | "B2"
        max_speed: float = 1.0,
        dt: float = 1.0,
        sim_horizon: Optional[int] = None,
        lambda_change: float = 15.0,
        alpha_makespan: float = 3.0,
        use_rust: bool = True,
    ) -> None:
        self.network = network
        self.snapshot = snapshot
        self.mode = mode
        self.max_speed = max_speed
        self.dt = dt

        max_due = max(
            [snapshot.tick + 400]
            + [t.due_step + 200 for t in snapshot.tasks if t.due_step is not None]
            + [op.due_step + 200 for op in snapshot.operations]
        )
        max_edge_steps = max(
            (
                max(1, int(math.ceil(e.length / min(max_speed, e.speed_limit) / dt)))
                for e in network.edges.values()
            ),
            default=1,
        )
        n_active_tasks = sum(1 for t in snapshot.tasks if t.status.lower() != "completed")
        eff_robots = max(2, min(len(snapshot.robots), 4) if len(snapshot.robots) > 3 else len(snapshot.robots))
        est_work = max_edge_steps * 8 * max(1, (n_active_tasks + eff_robots - 1) // eff_robots)
        self.sim_horizon = (
            sim_horizon if sim_horizon is not None else max(600, max_due, snapshot.tick + est_work + 200)
        )
        self.lambda_change = lambda_change
        self.alpha_makespan = alpha_makespan

        self.use_rust = bool(use_rust and is_rust_planner_available() and is_rust_core_available())
        static_entry = self._get_static_cache(network, max_speed, dt, self.use_rust)
        self._rust_planner = static_entry["planner"] if self.use_rust else None
        self._haven_nodes = static_entry["haven_nodes"]
        self._edge_steps_map: dict[tuple[str, str], int] = static_entry["edge_steps"]
        self._rev_graph: dict[str, list[tuple[str, float]]] = static_entry["rev_graph"]

        self.t_commit = snapshot.tick
        self.tasks_by_id = {t.task_id: t for t in snapshot.tasks}
        self.robots_by_id = {r.robot_id: r for r in snapshot.robots}
        self.ops_by_id = {op.op_id: op for op in snapshot.operations}

        self.snapshot_assign = {
            t.task_id: t.assigned_agv
            for t in snapshot.tasks
            if t.assigned_agv is not None
        }
        self._dist_cache: dict[str, dict[str, float]] = static_entry["dist_cache"]

        self._task_to_op: dict[str, str] = {}
        for op in snapshot.operations:
            for ktid in op.kit_task_ids:
                self._task_to_op[str(ktid)] = str(op.op_id)
        for t in snapshot.tasks:
            if t.kit_op_id:
                self._task_to_op[str(t.task_id)] = str(t.kit_op_id)

        # E2 / B2 persistent incremental cache
        self.base_assign: dict[str, str] = {}
        self.base_ctx_key: tuple[Any, ...] | None = None
        self.base_bay_overrides: dict[str, str] = {}
        self.base_task_order: dict[str, list[str]] = {}
        self.base_traffic_sig: tuple[Any, ...] = ()
        self._traffic_weights_q3: dict[tuple[str, str], int] = {}
        self._traffic_cutoff_step: int = 0
        self._traffic_sig: tuple[Any, ...] = ()
        self.robot_cache: dict[str, dict[str, Any]] = {}
        self._inc_res_table: ReservationTable | RustReservationTable | None = None
        self._inc_backup_table: ReservationTable | RustReservationTable | None = None
        self._pending_b1_seed: Optional[tuple[Any, ...]] = None

        self.delivered_unloads: dict[str, int] = {
            t.task_id: int(t.complete_step)
            for t in snapshot.tasks
            if t.status.lower() == "completed" and t.complete_step is not None
        }

        fb = getattr(snapshot, "execution_feedback", None)
        self._exec_blocked_until: dict[tuple[str, str], int] = (
            fb.blocked_intervals(self.t_commit, default_horizon=max(30, self.sim_horizon - self.t_commit))
            if fb is not None and not fb.is_empty()
            else {}
        )
        self._exec_q3_weights: dict[tuple[str, str], int] = (
            fb.corridor_q3_penalties()
            if fb is not None and not fb.is_empty()
            else {}
        )
        if self._exec_q3_weights:
            self.set_traffic_weights(None)

        # Telemetry
        self.calls = 0
        self.a1_calls = 0
        self.b1_calls = 0
        self.b2_inc_hits = 0
        self.b2_fallbacks = 0
        self.eval_lats_ms: list[float] = []

    def _new_res_table(self) -> ReservationTable | RustReservationTable:
        rt: ReservationTable | RustReservationTable = (
            RustReservationTable(track_shadow=False) if self.use_rust else ReservationTable()
        )
        if self._exec_blocked_until:
            for (src, dst), end_step in self._exec_blocked_until.items():
                if not self.network.has_edge(src, dst) or end_step <= self.t_commit:
                    continue
                e = self.network.edge(src, dst)
                cap = e.capacity
                if self.network.has_edge(dst, src):
                    cap = min(cap, self.network.edge(dst, src).capacity)
                span = Interval(self.t_commit, min(self.sim_horizon, int(end_step)))
                if len(span) <= 0:
                    continue
                for k in range(max(1, int(cap))):
                    rt.reserve_edge(src, dst, span, f"__L1_OBSTACLE_{k}__")
        return rt

    def set_traffic_weights(
        self,
        weights: dict[tuple[str, str], int] | None,
        cutoff_step: int = 0,
    ) -> None:
        """Install Lightweight Traffic Map Q3 edge penalties and optional horizon cutoff."""
        merged: dict[tuple[str, str], int] = dict(self._exec_q3_weights)
        if weights:
            for (s, d), w in weights.items():
                iv = max(0, min(64, int(w)))
                if iv > 0:
                    ek = (str(s), str(d))
                    merged[ek] = max(merged.get(ek, 0), iv)
        c_step = max(0, int(cutoff_step)) if weights else 0
        if not merged:
            self._traffic_cutoff_step = 0
            self._traffic_weights_q3 = {}
            self._traffic_sig = ()
            return
        self._traffic_cutoff_step = c_step
        self._traffic_weights_q3 = merged
        self._traffic_sig = (
            c_step,
            tuple(sorted((s, d, w) for (s, d), w in merged.items())),
        )

    def _sync_rust_traffic(self) -> None:
        if self._rust_planner is not None:
            if hasattr(self._rust_planner, "clear_agv_guides"):
                self._rust_planner.clear_agv_guides()
            if hasattr(self._rust_planner, "sync_traffic_weights"):
                self._rust_planner.sync_traffic_weights(
                    self.network,
                    self._traffic_weights_q3,
                    cutoff_step=self._traffic_cutoff_step,
                )

    @staticmethod
    def extract_guide_corridors(
        robot_paths: dict[str, list[tuple[str, int]]],
        t_start: int,
        horizon_steps: int = 60,
    ) -> dict[str, list[tuple[str, str]]]:
        """Extract near-horizon directed corridor transitions per AGV from CVCR routes."""
        t_end = int(t_start) + max(10, int(horizon_steps))
        out: dict[str, list[tuple[str, str]]] = {}
        for rid, path in robot_paths.items():
            if not path:
                continue
            edges: list[tuple[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for idx in range(len(path) - 1):
                u_n, u_t = path[idx]
                v_n, _v_t = path[idx + 1]
                if u_n != v_n and int(u_t) <= t_end:
                    key = (str(u_n), str(v_n))
                    if key not in seen:
                        seen.add(key)
                        edges.append(key)
            if edges:
                out[str(rid)] = edges
        return out

    def compute_ltm_weights_from_paths(
        self,
        robot_paths: dict[str, list[tuple[str, int]]],
        prev_weights: Optional[dict[tuple[str, str], int]] = None,
    ) -> dict[tuple[str, str], int]:
        """Extract bounded IJCAI-26 LTM edge penalties (Q3 steps) from spatiotemporal congestion."""
        edge_robots: dict[tuple[str, str], set[str]] = {}
        dir_robots: dict[tuple[str, str], set[str]] = {}
        wait_by_node: dict[str, int] = {}

        for r in self.snapshot.robots:
            st_low = str(r.state).lower()
            if "block" in st_low or "yield" in st_low:
                wait_by_node[r.cur_node] = wait_by_node.get(r.cur_node, 0) + 6

        for rid, path in robot_paths.items():
            if not path:
                continue
            for idx in range(len(path) - 1):
                u_n, u_t = path[idx]
                v_n, v_t = path[idx + 1]
                dt_seg = max(0, int(v_t) - int(u_t))
                if dt_seg <= 0:
                    continue
                if u_n == v_n:
                    nd = self.network.nodes.get(u_n)
                    if idx < len(path) - 1:
                        wait_pen = (
                            dt_seg
                            if (nd is None or nd.kind in (NodeKind.AISLE, NodeKind.BUFFER))
                            else max(0, dt_seg - 2)
                        )
                        if wait_pen > 0:
                            wait_by_node[u_n] = wait_by_node.get(u_n, 0) + wait_pen
                else:
                    edge_robots.setdefault(physical_edge_id(u_n, v_n), set()).add(rid)
                    dir_robots.setdefault((u_n, v_n), set()).add(rid)

        raw_w: dict[tuple[str, str], float] = {}
        if prev_weights:
            for e_key, pw in prev_weights.items():
                if pw > 1:
                    raw_w[e_key] = float(pw) * 0.45

        for (src, dst) in self.network.edges:
            phys = physical_edge_id(src, dst)
            fwd = len(dir_robots.get((src, dst), ()))
            rev = len(dir_robots.get((dst, src), ()))
            tot_phys = len(edge_robots.get(phys, ()))
            w_q3 = 0.0
            if fwd >= 1 and rev >= 1:
                w_q3 += 8.0 + 4.0 * float(min(4, fwd + rev - 2))
            elif tot_phys >= 2:
                w_q3 += 4.0 * float(min(4, tot_phys - 1))
            w_node = wait_by_node.get(dst, 0) + 0.5 * wait_by_node.get(src, 0)
            if w_node >= 2.0:
                w_q3 += min(16.0, 2.5 * math.log1p(w_node))
            if w_q3 > 0.0:
                raw_w[(src, dst)] = raw_w.get((src, dst), 0.0) + w_q3

        out: dict[tuple[str, str], int] = {}
        for k, val in raw_w.items():
            iv = min(32, int(round(val)))
            if iv >= 3:
                out[k] = iv
        return out

    @staticmethod
    def _reserve_path_steps(
        res_table: ReservationTable | RustReservationTable,
        agv_id: str,
        path: list[tuple[str, int]],
        node_res_added: list[tuple[str, Interval]],
        edge_res_added: list[tuple[tuple[str, str], Interval]],
        collect_res: bool,
        extra_node_wait: Optional[tuple[str, int, int]] = None,
        *,
        rust_planner: Any = None,
        network: Optional[Network] = None,
    ) -> None:
        """Reserve transitions along `path` (+ optional dwell wait), zero-copy indexed for Rust tables."""
        if isinstance(res_table, RustReservationTable):
            if collect_res:
                for i in range(len(path) - 1):
                    u_n, u_t = path[i]
                    v_n, v_t = path[i + 1]
                    if v_t <= u_t:
                        continue
                    if u_n == v_n:
                        node_res_added.append((u_n, Interval(u_t, v_t)))
                    else:
                        edge_res_added.append(((u_n, v_n), Interval(u_t, v_t)))
                if extra_node_wait is not None and extra_node_wait[2] > extra_node_wait[1]:
                    node_res_added.append((extra_node_wait[0], Interval(extra_node_wait[1], extra_node_wait[2])))
            if (
                rust_planner is not None
                and network is not None
                and hasattr(rust_planner, "reserve_chain_indexed")
                and rust_planner.reserve_chain_indexed(network, res_table, agv_id, path, extra_node_wait)
            ):
                return
            lines: list[str] = []
            for i in range(len(path) - 1):
                u_n, u_t = path[i]
                v_n, v_t = path[i + 1]
                if v_t <= u_t:
                    continue
                if u_n == v_n:
                    lines.append(f"N\t{u_n}\t{u_t}\t{v_t}")
                else:
                    lines.append(f"E\t{u_n}\t{v_n}\t{u_t}\t{v_t}")
            if extra_node_wait is not None:
                w_n, w_s, w_e = extra_node_wait
                if w_e > w_s:
                    lines.append(f"N\t{w_n}\t{w_s}\t{w_e}")
            if lines:
                res_table.reserve_batch_script(agv_id, "\n".join(lines))
            return

        for i in range(len(path) - 1):
            u_n, u_t = path[i]
            v_n, v_t = path[i + 1]
            if v_t <= u_t:
                continue
            span = Interval(u_t, v_t)
            if u_n == v_n:
                res_table.reserve_node(u_n, span, agv_id)
                if collect_res:
                    node_res_added.append((u_n, span))
            else:
                res_table.reserve_edge(u_n, v_n, span, agv_id)
                if collect_res:
                    edge_res_added.append(((u_n, v_n), span))
        if extra_node_wait is not None:
            w_n, w_s, w_e = extra_node_wait
            if w_e > w_s:
                w_span = Interval(w_s, w_e)
                res_table.reserve_node(w_n, w_span, agv_id)
                if collect_res:
                    node_res_added.append((w_n, w_span))

    def _heuristic_table(self, goal: str) -> dict[str, float]:
        if goal in self._dist_cache:
            return self._dist_cache[goal]
        rev = self._rev_graph
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

    def _edge_steps(self, src: str, dst: str) -> int:
        steps = self._edge_steps_map.get((src, dst))
        if steps is not None:
            return steps
        edge = self.network.edge(src, dst)
        return max(1, int(math.ceil(edge.length / min(self.max_speed, edge.speed_limit) / self.dt)))

    def _space_time_route(
        self,
        start_node: str,
        goal_node: str,
        start_step: int,
        reservations: ReservationTable | RustReservationTable,
        agv_id: str,
        max_horizon: Optional[int] = None,
    ) -> list[tuple[str, int]] | None:
        """Find collision-free path from start_node at start_step to goal_node."""
        if start_node == goal_node:
            return [(start_node, start_step)]

        horizon = (
            max_horizon
            if max_horizon is not None
            else min(self.sim_horizon - start_step, max(450, self.sim_horizon // 2))
        )
        limit_step = min(self.sim_horizon, start_step + max(1, horizon))

        if self._rust_planner is not None and isinstance(reservations, RustReservationTable):
            planned = self._rust_planner.plan(
                self.network,
                start_node,
                goal_node,
                start_step,
                reservations,
                agv_id,
                limit_step,
            )
            if planned is None:
                return None
            # Convert PlannedPath (collapsed wait+move hops) into step-preserving transitions
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

        # Pure Python fallback A*
        h_tbl = self._heuristic_table(goal_node)
        if goal_node not in h_tbl:
            return None

        def h(node: str) -> int:
            return int(math.floor(h_tbl.get(node, 1000.0) / self.dt))

        tw_q3 = self._traffic_weights_q3
        counter = itertools.count()
        frontier_q3: list[tuple[int, int, int, int, str]] = [
            (h(start_node) * 8, 0, 0, next(counter), start_node)
        ]
        best_score: dict[tuple[str, int], int] = {(start_node, start_step): 0}
        parent: dict[tuple[str, int], tuple[str, int] | None] = {(start_node, start_step): None}
        goal_state: tuple[str, int] | None = None
        expansions = 0
        max_exp = 20_000

        while frontier_q3:
            expansions += 1
            if expansions > max_exp:
                return None
            _, cur_sc, g, _, node = heapq.heappop(frontier_q3)
            t = start_step + g
            if t > limit_step:
                continue
            state = (node, t)
            if best_score.get(state, 10**18) < cur_sc:
                continue
            if node == goal_node:
                goal_state = state
                break

            if t + 1 <= limit_step:
                span = Interval(t, t + 1)
                cap = self.network.node(node).capacity
                if reservations.is_node_free(node, span, cap, exclude_agv=agv_id):
                    ns = (node, t + 1)
                    n_sc = cur_sc + 8
                    if n_sc < best_score.get(ns, 10**18):
                        best_score[ns] = n_sc
                        parent[ns] = state
                        heapq.heappush(frontier_q3, (n_sc + h(node) * 8, n_sc, g + 1, next(counter), node))

            for e in self.network.out_edges(node):
                n_steps = self._edge_steps(e.src, e.dst)
                arr = t + n_steps
                if arr > limit_step:
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
                    occ += reservations.edge_conflicts(phys[1], phys[0], edge_span, exclude_agv=agv_id)
                if len(set(occ)) >= edge_cap:
                    continue
                arr_span = Interval(arr, arr + 1)
                if not reservations.is_node_free(e.dst, arr_span, self.network.node(e.dst).capacity, exclude_agv=agv_id):
                    continue
                ns = (e.dst, arr)
                ng = g + n_steps
                cut_s = self._traffic_cutoff_step
                tw_pen = tw_q3.get((e.src, e.dst), 0) if (tw_q3 and (cut_s == 0 or t < cut_s)) else 0
                n_sc = cur_sc + n_steps * 8 + tw_pen
                if n_sc < best_score.get(ns, 10**18):
                    best_score[ns] = n_sc
                    parent[ns] = state
                    heapq.heappush(frontier_q3, (n_sc + h(e.dst) * 8, n_sc, ng, next(counter), e.dst))

        if goal_state is None:
            return None

        chain_py: list[tuple[str, int]] = []
        cur: tuple[str, int] | None = goal_state
        while cur is not None:
            chain_py.append(cur)
            cur = parent[cur]
        chain_py.reverse()
        return chain_py

    def _propagate_production(
        self,
        order_unload_ticks: dict[str, int],
        op_priority: dict[str, int],
        op_bay_overrides: Optional[dict[str, str]] = None,
        op_crew_overrides: Optional[dict[str, int]] = None,
    ) -> tuple[dict[str, int], dict[str, int], dict[str, int], int, int, int]:
        """Propagate kitting, predecessor, bay, crew, and tool mutexes."""
        op_start: dict[str, int] = {}
        op_end: dict[str, int] = {}
        op_res_ready: dict[str, int] = {}
        bay_free: dict[str, int] = {}
        crew_free: dict[int, int] = {}
        tool_free: dict[int, int] = {}

        ops = sorted(
            self.snapshot.operations,
            key=lambda op: (0 if op.started else 1, op_priority.get(op.op_id, 0), op.due_step),
        )
        resolved: set[str] = set()
        starvation_idle = 0

        for _ in range(len(ops) + 2):
            for op in ops:
                if op.op_id in resolved:
                    continue
                preds = op_predecessors(op)
                if any(pid not in resolved for pid in preds):
                    continue
                bay = op_bay_overrides.get(op.op_id, op.bay_id) if op_bay_overrides else op.bay_id
                crew = op_crew_overrides.get(op.op_id, op.crew_id) if op_crew_overrides else op.crew_id
                if op.started and op.start_step is not None and op.end_step is not None:
                    op_start[op.op_id] = op.start_step
                    op_end[op.op_id] = op.end_step
                    op_res_ready[op.op_id] = op.start_step
                    bay_free[bay] = max(bay_free.get(bay, 0), op.end_step)
                    crew_free[crew] = max(crew_free.get(crew, 0), op.end_step)
                    tool_free[op.tool_id] = max(tool_free.get(op.tool_id, 0), op.end_step)
                else:
                    mat_t = max(
                        (order_unload_ticks.get(tid, self.sim_horizon) for tid in op.kit_task_ids),
                        default=self.t_commit,
                    )
                    pred_t = max((op_end.get(pid, 0) for pid in preds), default=0)
                    r_ready = max(
                        self.t_commit,
                        pred_t,
                        bay_free.get(bay, self.t_commit),
                        crew_free.get(crew, self.t_commit),
                        tool_free.get(op.tool_id, self.t_commit),
                    )
                    op_res_ready[op.op_id] = r_ready
                    st_t = align_start_to_windows(
                        max(r_ready, mat_t),
                        getattr(op, "resource_windows", ()),
                    )
                    win_ready = align_start_to_windows(r_ready, getattr(op, "resource_windows", ()))
                    starvation_idle += max(0, st_t - win_ready)
                    en_t = st_t + op.proc_steps
                    op_start[op.op_id] = st_t
                    op_end[op.op_id] = en_t
                    bay_free[bay] = en_t
                    crew_free[crew] = en_t
                    tool_free[op.tool_id] = en_t
                resolved.add(op.op_id)

        makespan = max(op_end.values()) if op_end else self.t_commit
        tard = sum(
            (12 if op.is_urgent else 4) * max(0, op_end.get(op.op_id, self.sim_horizon) - op.due_step)
            for op in self.snapshot.operations
        )
        return op_start, op_end, op_res_ready, makespan, tard, starvation_idle

    def _route_single_robot(
        self,
        agv_id: str,
        tasks: list[TaskSnapshot],
        res_table: ReservationTable | RustReservationTable,
        op_priority: dict[str, int],
        op_res_ready_hint: dict[str, int] | None,
        op_bay_overrides: Optional[dict[str, str]] = None,
        task_order_override: Optional[dict[str, list[str]]] = None,
        collect_res: bool = True,
    ) -> tuple[
        list[tuple[str, int]],
        list[str],
        dict[str, int],
        list[tuple[str, Interval]],
        list[tuple[tuple[str, str], Interval]],
        bool,
    ]:
        """Route one AMR sequentially through its assigned tasks."""
        rob = self.robots_by_id[agv_id]
        cur_node = rob.cur_node
        fb = getattr(self.snapshot, "execution_feedback", None)
        exec_delay = fb.robot_ready_offset_steps(agv_id, self.dt) if fb is not None else 0
        cur_t = max(self.t_commit, rob.faulted_until, self.t_commit + max(0, int(exec_delay)))

        order_rank = (
            {tid: idx for idx, tid in enumerate(task_order_override[agv_id])}
            if task_order_override and agv_id in task_order_override
            else None
        )

        if order_rank is not None:
            tasks_sorted = sorted(
                tasks,
                key=lambda t: (
                    0 if t.status.lower() in ("picked", "loading", "to_dropoff", "unloading") else 1,
                    order_rank.get(t.task_id, 100_000),
                    op_priority.get(t.kit_op_id or "", 0),
                    t.release_step,
                    t.task_id,
                ),
            )
        elif op_res_ready_hint is not None:
            tasks_sorted = sorted(
                tasks,
                key=lambda t: (
                    0 if t.status.lower() in ("picked", "loading", "to_dropoff", "unloading") else 1,
                    max(op_res_ready_hint.get(t.kit_op_id or "", 0), t.release_step),
                    0
                    if (
                        t.kit_op_id
                        and self.ops_by_id.get(t.kit_op_id)
                        and self.ops_by_id[t.kit_op_id].is_urgent
                    )
                    else 1,
                    t.task_id,
                ),
            )
        else:
            tasks_sorted = sorted(
                tasks,
                key=lambda t: (
                    0 if t.status.lower() in ("picked", "loading", "to_dropoff", "unloading") else 1,
                    op_priority.get(t.kit_op_id or "", 0),
                    t.release_step,
                    t.task_id,
                ),
            )

        task_seq = [t.task_id for t in tasks_sorted]
        full_path: list[tuple[str, int]] = []
        unload_ticks: dict[str, int] = {}
        node_res_added: list[tuple[str, Interval]] = []
        edge_res_added: list[tuple[tuple[str, str], Interval]] = []

        # Parked/faulted idle robot retreats off capacity=1 stations if needed and holds haven
        if not tasks_sorted:
            nd_obj = self.network.node(cur_node)
            if (
                rob.state not in ("blocked", "faulted")
                and nd_obj.kind not in (NodeKind.BUFFER, NodeKind.AISLE)
            ):
                for hv in self._haven_nodes:
                    hv_cap = self.network.node(hv).capacity
                    if res_table.is_node_free(hv, Interval(cur_t, self.sim_horizon), hv_cap, exclude_agv=agv_id):
                        p_park = self._space_time_route(cur_node, hv, cur_t, res_table, agv_id, max_horizon=200)
                        if p_park is not None:
                            self._reserve_path_steps(
                                res_table, agv_id, p_park, node_res_added, edge_res_added, collect_res,
                                rust_planner=self._rust_planner, network=self.network,
                            )
                            cur_t = p_park[-1][1]
                            cur_node = hv
                            full_path.extend(p_park)
                            break
            if cur_t < self.sim_horizon:
                rest_span = Interval(cur_t, self.sim_horizon)
                res_table.reserve_node(cur_node, rest_span, agv_id)
                if collect_res:
                    node_res_added.append((cur_node, rest_span))
            return full_path or [(cur_node, self.t_commit)], [], {}, node_res_added, edge_res_added, True

        all_ok = True
        for idx_t, t in enumerate(tasks_sorted):
            dropoff = (
                op_bay_overrides[t.kit_op_id]
                if (
                    op_bay_overrides
                    and t.kit_op_id in op_bay_overrides
                    and t.status.lower() in ("pending", "assigned")
                )
                else t.dropoff
            )

            if t.status.lower() not in ("picked", "loading", "to_dropoff", "unloading"):
                p1 = self._space_time_route(cur_node, t.pickup, cur_t, res_table, agv_id)
                if p1 is None:
                    all_ok = False
                    for rem_t in tasks_sorted[idx_t:]:
                        unload_ticks[rem_t.task_id] = self.sim_horizon
                    break
                cur_t_arr = p1[-1][1]
                cur_node = t.pickup
                drift_buf1 = 0
                if fb is not None:
                    raw_buf1 = fb.robot_leg_step_buffer(agv_id, cur_t_arr - cur_t)
                    drift_buf1 = raw_buf1 if idx_t == 0 else min(2, max(0, raw_buf1 - idx_t))
                eff_arr1 = min(self.sim_horizon, cur_t_arr + drift_buf1)
                load_time = 2
                wait_until = max(eff_arr1, t.release_step) + load_time
                extra_wait = (
                    (cur_node, cur_t_arr, wait_until)
                    if (wait_until > cur_t_arr and wait_until <= self.sim_horizon)
                    else None
                )
                self._reserve_path_steps(
                    res_table, agv_id, p1, node_res_added, edge_res_added, collect_res, extra_wait,
                    rust_planner=self._rust_planner, network=self.network,
                )
                for pt in p1:
                    if not full_path or full_path[-1] != pt:
                        full_path.append(pt)
                cur_t = min(wait_until, self.sim_horizon)
                full_path.append((cur_node, cur_t))

            p2 = self._space_time_route(cur_node, dropoff, cur_t, res_table, agv_id)
            if p2 is None:
                all_ok = False
                for rem_t in tasks_sorted[idx_t:]:
                    unload_ticks[rem_t.task_id] = self.sim_horizon
                break
            cur_t_arr2 = p2[-1][1]
            cur_node = dropoff
            drift_buf2 = 0
            if fb is not None:
                raw_buf2 = fb.robot_leg_step_buffer(agv_id, cur_t_arr2 - cur_t)
                drift_buf2 = raw_buf2 if idx_t == 0 else min(2, max(0, raw_buf2 - idx_t))
            eff_arr2 = min(self.sim_horizon, cur_t_arr2 + drift_buf2)
            unload_time = 2
            end_u = min(eff_arr2 + unload_time, self.sim_horizon)
            extra_u = (cur_node, cur_t_arr2, end_u) if end_u > cur_t_arr2 else None
            self._reserve_path_steps(
                res_table, agv_id, p2, node_res_added, edge_res_added, collect_res, extra_u,
                rust_planner=self._rust_planner, network=self.network,
            )
            for pt in p2:
                if not full_path or full_path[-1] != pt:
                    full_path.append(pt)
            cur_t = end_u
            full_path.append((cur_node, cur_t))
            unload_ticks[t.task_id] = cur_t

        # Haven parking at end of active route (mirrors railway_prod_model.py: never squat on capacity=1 workstation)
        tail_reserved = False
        if all_ok and self.network.node(cur_node).kind not in (NodeKind.BUFFER, NodeKind.AISLE):
            for hv in self._haven_nodes:
                hv_cap = self.network.node(hv).capacity
                if res_table.is_node_free(hv, Interval(cur_t, self.sim_horizon), hv_cap, exclude_agv=agv_id):
                    p_park = self._space_time_route(cur_node, hv, cur_t, res_table, agv_id, max_horizon=200)
                    if p_park is not None:
                        cur_t_pk = p_park[-1][1]
                        extra_rest = (hv, cur_t_pk, self.sim_horizon) if cur_t_pk < self.sim_horizon else None
                        self._reserve_path_steps(
                            res_table, agv_id, p_park, node_res_added, edge_res_added, collect_res, extra_rest,
                            rust_planner=self._rust_planner, network=self.network,
                        )
                        for pt in p_park:
                            if not full_path or full_path[-1] != pt:
                                full_path.append(pt)
                        cur_t = cur_t_pk
                        cur_node = hv
                        tail_reserved = extra_rest is not None
                        break

        if not tail_reserved and cur_t < self.sim_horizon:
            rest_span = Interval(cur_t, self.sim_horizon)
            res_table.reserve_node(cur_node, rest_span, agv_id)
            if collect_res:
                node_res_added.append((cur_node, rest_span))

        return full_path, task_seq, unload_ticks, node_res_added, edge_res_added, all_ok

    def _compute_objective(
        self,
        assignment: dict[str, str],
        order_unload_ticks: dict[str, int],
        op_priority: dict[str, int],
        all_ok: bool,
        op_bay_overrides: Optional[dict[str, str]] = None,
        op_crew_overrides: Optional[dict[str, int]] = None,
    ) -> tuple[dict[str, int], dict[str, int], dict[str, int], float, int]:
        op_s, op_e, op_rr, mk, tard, starv = self._propagate_production(
            order_unload_ticks, op_priority, op_bay_overrides, op_crew_overrides
        )
        tot_deliv = sum(
            order_unload_ticks.get(t.task_id, self.sim_horizon)
            for t in self.snapshot.tasks
            if t.status.lower() != "completed"
        )
        chg = sum(
            1
            for tid, agv_id in assignment.items()
            if tid in self.snapshot_assign
            and self.snapshot_assign[tid] != agv_id
            and self.tasks_by_id[tid].status.lower() == "assigned"
        )
        obj = float(tard + self.alpha_makespan * mk + 0.5 * tot_deliv + self.lambda_change * chg)
        if not all_ok:
            obj += 5000.0
        return op_s, op_e, op_rr, obj, starv

    def evaluate(
        self,
        assignment: dict[str, str],
        op_priority: Optional[dict[str, int]] = None,
        op_res_ready_hint: Optional[dict[str, int]] = None,
        op_bay_overrides: Optional[dict[str, str]] = None,
        op_crew_overrides: Optional[dict[str, int]] = None,
        task_order_override: Optional[dict[str, list[str]]] = None,
    ) -> EvalResult:
        """Evaluate candidate assignment under configured variant (A1, B1, or B2)."""
        t0 = _cvcr_now_s()
        self.calls += 1
        c_full_before = self.a1_calls + self.b1_calls
        self._sync_rust_traffic()

        prio = op_priority or {
            op.op_id: max(self.t_commit, op.due_step - op.proc_steps)
            for op in self.snapshot.operations
        }

        if self.mode == "A1":
            res = self._eval_a1(
                assignment, prio, op_res_ready_hint, op_bay_overrides, op_crew_overrides, task_order_override
            )
        elif self.mode == "B1":
            res = self._eval_b1(
                assignment, prio, op_res_ready_hint, op_bay_overrides, op_crew_overrides, task_order_override
            )
        else:
            res = self._eval_b2(
                assignment, prio, op_res_ready_hint, op_bay_overrides, op_crew_overrides, task_order_override
            )

        if is_virtual_clock_active():
            ran_full = (self.a1_calls + self.b1_calls) > c_full_before
            if ran_full:
                _cvcr_record_work_ms(0.28 + 0.024 * len(self.tasks_by_id) + 0.008 * len(self.network.nodes))
            else:
                _cvcr_record_work_ms(0.04 + 0.007 * len(self.network.nodes))

        lat = (_cvcr_now_s() - t0) * 1000.0
        res.eval_latency_ms = lat
        self.eval_lats_ms.append(lat)
        return res

    def _robot_plan_sort_key(
        self,
        rid: str,
        rob_tasks: list[TaskSnapshot],
        op_priority: dict[str, int],
    ) -> tuple[int, int, str]:
        """Order idle robots first to vacate spurs, then active robots by op_priority (tie by id)."""
        if not rob_tasks:
            return (0, 0, rid)
        best_p = min(
            (
                int(op_priority.get(t.kit_op_id or self._task_to_op.get(t.task_id, ""), 999999))
                for t in rob_tasks
            ),
            default=999999,
        )
        return (1, best_p, rid)

    def _eval_a1(
        self,
        assignment: dict[str, str],
        op_priority: dict[str, int],
        op_res_ready_hint: Optional[dict[str, int]] = None,
        op_bay_overrides: Optional[dict[str, str]] = None,
        op_crew_overrides: Optional[dict[str, int]] = None,
        task_order_override: Optional[dict[str, list[str]]] = None,
    ) -> EvalResult:
        """A1: Full replan of ALL robots from scratch + full reservation build."""
        self.a1_calls += 1
        res_table = self._new_res_table()
        by_robot: dict[str, list[TaskSnapshot]] = {r.robot_id: [] for r in self.snapshot.robots}
        for tid, t in self.tasks_by_id.items():
            if t.status.lower() == "completed":
                continue
            agv_id = assignment.get(tid)
            if agv_id in by_robot:
                by_robot[agv_id].append(t)

        order_unload_ticks: dict[str, int] = dict(self.delivered_unloads)
        robot_paths: dict[str, list[tuple[str, int]]] = {}
        robot_order_seq: dict[str, list[str]] = {}
        all_ok = True

        for agv_id in sorted(by_robot.keys(), key=lambda r: self._robot_plan_sort_key(r, by_robot[r], op_priority)):
            path, seq, unloads, _, _, ok = self._route_single_robot(
                agv_id,
                by_robot[agv_id],
                res_table,
                op_priority,
                op_res_ready_hint,
                op_bay_overrides,
                task_order_override=task_order_override,
                collect_res=False,
            )
            robot_paths[agv_id] = path
            robot_order_seq[agv_id] = seq
            order_unload_ticks.update(unloads)
            if not ok:
                all_ok = False
                break

        op_s, op_e, op_rr, obj, starv = self._compute_objective(
            assignment, order_unload_ticks, op_priority, all_ok, op_bay_overrides, op_crew_overrides
        )
        return EvalResult(
            robot_paths=robot_paths,
            robot_order_seq=robot_order_seq,
            order_unload_ticks=order_unload_ticks,
            op_start=op_s,
            op_end=op_e,
            op_res_ready=op_rr,
            objective=obj,
            valid=all_ok,
            starvation_ticks=starv,
            eval_latency_ms=0.0,
            mode="A1",
        )

    def _eval_b1(
        self,
        assignment: dict[str, str],
        op_priority: dict[str, int],
        op_res_ready_hint: Optional[dict[str, int]] = None,
        op_bay_overrides: Optional[dict[str, str]] = None,
        op_crew_overrides: Optional[dict[str, int]] = None,
        task_order_override: Optional[dict[str, list[str]]] = None,
    ) -> EvalResult:
        """B1: Preserve unchanged robots' paths, reconstruct full reservation table from scratch."""
        self.b1_calls += 1
        res_table = self._new_res_table()

        prio_key = tuple(sorted(op_priority.items()))
        hint_key = tuple(sorted(op_res_ready_hint.items())) if op_res_ready_hint is not None else None
        ctx_matches_base = (
            self.base_ctx_key is not None
            and self.base_ctx_key[:2] == (prio_key, hint_key)
            and self.base_traffic_sig == self._traffic_sig
        )
        changed_rids = None
        if (
            ctx_matches_base
            and self.base_assign
            and self.robot_cache
            and not op_bay_overrides
            and not self.base_bay_overrides
            and not task_order_override
            and not self.base_task_order
        ):
            changed_rids = sorted(
                {assignment[tid] for tid in assignment if assignment[tid] != self.base_assign.get(tid)}
                | {self.base_assign[tid] for tid in self.base_assign if assignment.get(tid) != self.base_assign[tid]}
            )

        order_unload_ticks: dict[str, int] = dict(self.delivered_unloads)
        robot_paths: dict[str, list[tuple[str, int]]] = {}
        robot_order_seq: dict[str, list[str]] = {}
        node_res_by_rid: dict[str, list[tuple[str, Interval]]] = {}
        edge_res_by_rid: dict[str, list[tuple[tuple[str, str], Interval]]] = {}
        collect_res = True

        if changed_rids is not None and 0 < len(changed_rids) < len(self.snapshot.robots):
            for r in self.snapshot.robots:
                rid = r.robot_id
                if rid not in changed_rids and rid in self.robot_cache:
                    rc = self.robot_cache[rid]
                    robot_paths[rid] = rc["path"]
                    robot_order_seq[rid] = rc["seq"]
                    order_unload_ticks.update(rc["unloads"])
                    n_res = list(rc["node_res"])
                    e_res = list(rc["edge_res"])
                    node_res_by_rid[rid] = n_res
                    edge_res_by_rid[rid] = e_res
                    if isinstance(res_table, RustReservationTable):
                        if "batch_script" in rc:
                            res_table.reserve_batch_script(rid, rc["batch_script"])
                        else:
                            lines = [f"N\t{n}\t{sp.start}\t{sp.end}" for n, sp in n_res]
                            lines.extend(f"E\t{e[0]}\t{e[1]}\t{sp.start}\t{sp.end}" for e, sp in e_res)
                            if lines:
                                res_table.reserve_batch_script(rid, "\n".join(lines))
                    else:
                        for node, span in n_res:
                            res_table.reserve_node(node, span, rid)
                        for edge, span in e_res:
                            res_table.reserve_edge(edge[0], edge[1], span, rid)
            rids_to_route = changed_rids
        else:
            rids_to_route = sorted(r.robot_id for r in self.snapshot.robots)

        by_robot: dict[str, list[TaskSnapshot]] = {rid: [] for rid in rids_to_route}
        for tid, t in self.tasks_by_id.items():
            if t.status.lower() == "completed":
                continue
            agv_id = assignment.get(tid)
            if agv_id in by_robot:
                by_robot[agv_id].append(t)

        all_ok = True
        for agv_id in sorted(rids_to_route, key=lambda r: self._robot_plan_sort_key(r, by_robot[r], op_priority)):
            path, seq, unloads, n_res, e_res, ok = self._route_single_robot(
                agv_id,
                by_robot[agv_id],
                res_table,
                op_priority,
                op_res_ready_hint,
                op_bay_overrides,
                task_order_override=task_order_override,
                collect_res=collect_res,
            )
            robot_paths[agv_id] = path
            robot_order_seq[agv_id] = seq
            node_res_by_rid[agv_id] = n_res
            edge_res_by_rid[agv_id] = e_res
            order_unload_ticks.update(unloads)
            if not ok:
                all_ok = False
                break

        op_s, op_e, op_rr, obj, starv = self._compute_objective(
            assignment, order_unload_ticks, op_priority, all_ok, op_bay_overrides, op_crew_overrides
        )
        res_obj = EvalResult(
            robot_paths=robot_paths,
            robot_order_seq=robot_order_seq,
            order_unload_ticks=order_unload_ticks,
            op_start=op_s,
            op_end=op_e,
            op_res_ready=op_rr,
            objective=obj,
            valid=all_ok,
            starvation_ticks=starv,
            eval_latency_ms=0.0,
            mode="B1",
        )
        if all_ok and self.mode == "B2":
            self._pending_b1_seed = (res_obj, res_table, node_res_by_rid, edge_res_by_rid)
        return res_obj

    def _restore_inc_table(self, changed_rids: list[str]) -> None:
        """Restore persistent incremental reservation table back to incumbent state."""
        assert self._inc_res_table is not None
        if (
            self.use_rust
            and isinstance(self._inc_res_table, RustReservationTable)
            and isinstance(self._inc_backup_table, RustReservationTable)
        ):
            self._inc_res_table.restore_from(self._inc_backup_table)
            return
        for rid in changed_rids:
            self._inc_res_table.release_all(rid)
            rc = self.robot_cache.get(rid)
            if rc:
                for node, span in rc["node_res"]:
                    self._inc_res_table.reserve_node(node, span, rid)
                for edge, span in rc["edge_res"]:
                    self._inc_res_table.reserve_edge(edge[0], edge[1], span, rid)

    def _eval_b2(
        self,
        assignment: dict[str, str],
        op_priority: dict[str, int],
        op_res_ready_hint: Optional[dict[str, int]] = None,
        op_bay_overrides: Optional[dict[str, str]] = None,
        op_crew_overrides: Optional[dict[str, int]] = None,
        task_order_override: Optional[dict[str, list[str]]] = None,
    ) -> EvalResult:
        """B2: Persistent Incremental Reservation-Cached Evaluator (E2)."""
        prio_key = tuple(sorted(op_priority.items()))
        hint_key = tuple(sorted(op_res_ready_hint.items())) if op_res_ready_hint is not None else None
        bay_key = tuple(sorted(op_bay_overrides.items())) if op_bay_overrides else None
        order_key = (
            tuple((k, tuple(v)) for k, v in sorted(task_order_override.items()))
            if task_order_override
            else None
        )

        if (
            self.base_ctx_key is None
            or self.base_ctx_key[:2] != (prio_key, hint_key)
            or self.base_traffic_sig != self._traffic_sig
            or not self.robot_cache
            or self._inc_res_table is None
        ):
            return self._eval_b1(
                assignment,
                op_priority,
                op_res_ready_hint,
                op_bay_overrides,
                op_crew_overrides,
                task_order_override,
            )

        assign_changed_rids = (
            {assignment[tid] for tid in assignment if assignment[tid] != self.base_assign.get(tid)}
            | {self.base_assign[tid] for tid in self.base_assign if assignment.get(tid) != self.base_assign[tid]}
        )
        changed_set = set(assign_changed_rids)
        if bay_key != self.base_ctx_key[2]:
            new_bays = op_bay_overrides or {}
            changed_ops = {
                op_id
                for op_id in (set(self.base_bay_overrides) | set(new_bays))
                if self.base_bay_overrides.get(op_id, self.ops_by_id[op_id].bay_id)
                != new_bays.get(op_id, self.ops_by_id[op_id].bay_id)
            }
            bay_rids = {
                assignment.get(t.task_id)
                for t in self.tasks_by_id.values()
                if t.kit_op_id in changed_ops
                and t.status.lower() in ("pending", "assigned")
                and assignment.get(t.task_id)
            }
            changed_set |= bay_rids
        base_order_key = self.base_ctx_key[3] if len(self.base_ctx_key) > 3 else None
        if order_key != base_order_key:
            new_orders = task_order_override or {}
            order_rids = {
                rid
                for rid in (set(self.base_task_order) | set(new_orders))
                if self.base_task_order.get(rid) != new_orders.get(rid)
            }
            changed_set |= order_rids

        changed_rids = sorted(changed_set)

        if not changed_rids:
            # Pure production-resource (crew) or already-picked bay change: robot routes unchanged
            order_unload_ticks: dict[str, int] = dict(self.delivered_unloads)
            robot_paths: dict[str, list[tuple[str, int]]] = {}
            robot_order_seq: dict[str, list[str]] = {}
            for r in self.snapshot.robots:
                rid = r.robot_id
                rc = self.robot_cache.get(rid)
                if rc is not None:
                    robot_paths[rid] = rc["path"]
                    robot_order_seq[rid] = rc["seq"]
                    order_unload_ticks.update(rc["unloads"])
            self.b2_inc_hits += 1
            op_s, op_e, op_rr, obj, starv = self._compute_objective(
                assignment, order_unload_ticks, op_priority, True, op_bay_overrides, op_crew_overrides
            )
            return EvalResult(
                robot_paths=robot_paths,
                robot_order_seq=robot_order_seq,
                order_unload_ticks=order_unload_ticks,
                op_start=op_s,
                op_end=op_e,
                op_res_ready=op_rr,
                objective=obj,
                valid=True,
                starvation_ticks=starv,
                eval_latency_ms=0.0,
                mode="B2",
            )

        if len(changed_rids) >= len(self.snapshot.robots):
            self.b2_fallbacks += 1
            return self._eval_b1(
                assignment,
                op_priority,
                op_res_ready_hint,
                op_bay_overrides,
                op_crew_overrides,
                task_order_override,
            )

        res_table = self._inc_res_table

        # Release ONLY changed robots in-place from persistent reservation table
        for rid in changed_rids:
            res_table.release_all(rid)

        order_unload_ticks = dict(self.delivered_unloads)
        robot_paths = {}
        robot_order_seq = {}

        for r in self.snapshot.robots:
            rid = r.robot_id
            if rid not in changed_rids and rid in self.robot_cache:
                rc = self.robot_cache[rid]
                robot_paths[rid] = rc["path"]
                robot_order_seq[rid] = rc["seq"]
                order_unload_ticks.update(rc["unloads"])

        by_robot: dict[str, list[TaskSnapshot]] = {rid: [] for rid in changed_rids}
        for tid, t in self.tasks_by_id.items():
            if t.status.lower() == "completed":
                continue
            agv_id = assignment.get(tid)
            if agv_id in by_robot:
                by_robot[agv_id].append(t)

        all_ok = True
        for rid in sorted(changed_rids, key=lambda r: self._robot_plan_sort_key(r, by_robot[r], op_priority)):
            path, seq, unloads, _, _, ok = self._route_single_robot(
                rid,
                by_robot[rid],
                res_table,
                op_priority,
                op_res_ready_hint,
                op_bay_overrides,
                task_order_override=task_order_override,
                collect_res=False,
            )
            robot_paths[rid] = path
            robot_order_seq[rid] = seq
            order_unload_ticks.update(unloads)
            if not ok:
                all_ok = False
                break

        # Restore persistent table back to incumbent before returning
        self._restore_inc_table(changed_rids)

        if not all_ok:
            self.b2_fallbacks += 1
        else:
            self.b2_inc_hits += 1

        op_s, op_e, op_rr, obj, starv = self._compute_objective(
            assignment, order_unload_ticks, op_priority, all_ok, op_bay_overrides, op_crew_overrides
        )
        return EvalResult(
            robot_paths=robot_paths,
            robot_order_seq=robot_order_seq,
            order_unload_ticks=order_unload_ticks,
            op_start=op_s,
            op_end=op_e,
            op_res_ready=op_rr,
            objective=obj,
            valid=all_ok,
            starvation_ticks=starv,
            eval_latency_ms=0.0,
            mode="B2",
        )

    def _seed_cache(
        self,
        assignment: dict[str, str],
        eval_result: EvalResult,
        ctx_key: tuple[Any, ...],
        op_bay_overrides: Optional[dict[str, str]] = None,
        task_order_override: Optional[dict[str, list[str]]] = None,
    ) -> None:
        """Seed the incremental robot cache and persistent reservation table."""
        self.base_assign = dict(assignment)
        self.base_ctx_key = ctx_key
        self.base_bay_overrides = dict(op_bay_overrides) if op_bay_overrides else {}
        self.base_task_order = {k: list(v) for k, v in task_order_override.items()} if task_order_override else {}
        self.base_traffic_sig = self._traffic_sig
        self.robot_cache = {}

        pending = self._pending_b1_seed
        if pending is not None and pending[0] is eval_result:
            self._pending_b1_seed = None
            _, res_table, node_res_by_rid, edge_res_by_rid = pending
            self._inc_res_table = res_table
            if self.use_rust and isinstance(res_table, RustReservationTable):
                if self._inc_backup_table is None or not isinstance(self._inc_backup_table, RustReservationTable):
                    self._inc_backup_table = self._new_res_table()
                self._inc_backup_table.restore_from(res_table)
            for agv_id, path in eval_result.robot_paths.items():
                unloads = {
                    t.task_id: eval_result.order_unload_ticks[t.task_id]
                    for t in self.snapshot.tasks
                    if assignment.get(t.task_id) == agv_id and t.task_id in eval_result.order_unload_ticks
                }
                self.robot_cache[agv_id] = {
                    "path": path,
                    "seq": eval_result.robot_order_seq.get(agv_id, []),
                    "unloads": unloads,
                    "node_res": list(node_res_by_rid.get(agv_id, [])),
                    "edge_res": list(edge_res_by_rid.get(agv_id, [])),
                }
            return

        self._pending_b1_seed = None
        self._inc_res_table = self._new_res_table()

        for agv_id, path in eval_result.robot_paths.items():
            node_res: list[tuple[str, Interval]] = []
            edge_res: list[tuple[tuple[str, str], Interval]] = []
            batch_lines: list[str] = []
            use_batch = isinstance(self._inc_res_table, RustReservationTable)
            for i in range(len(path) - 1):
                u_n, u_t = path[i]
                v_n, v_t = path[i + 1]
                if v_t <= u_t:
                    continue
                if use_batch:
                    batch_lines.append(f"N\t{u_n}\t{u_t}\t{v_t}" if u_n == v_n else f"E\t{u_n}\t{v_n}\t{u_t}\t{v_t}")
                else:
                    span = Interval(u_t, v_t)
                    if u_n == v_n:
                        node_res.append((u_n, span))
                        self._inc_res_table.reserve_node(u_n, span, agv_id)
                    else:
                        edge_res.append(((u_n, v_n), span))
                        self._inc_res_table.reserve_edge(u_n, v_n, span, agv_id)
            if path and path[-1][1] < self.sim_horizon:
                if use_batch:
                    batch_lines.append(f"N\t{path[-1][0]}\t{path[-1][1]}\t{self.sim_horizon}")
                else:
                    rest_iv = Interval(path[-1][1], self.sim_horizon)
                    node_res.append((path[-1][0], rest_iv))
                    self._inc_res_table.reserve_node(path[-1][0], rest_iv, agv_id)
            if use_batch and batch_lines:
                assert isinstance(self._inc_res_table, RustReservationTable)
                self._inc_res_table.reserve_batch_script(agv_id, "\n".join(batch_lines))

            unloads = {
                t.task_id: eval_result.order_unload_ticks[t.task_id]
                for t in self.snapshot.tasks
                if assignment.get(t.task_id) == agv_id and t.task_id in eval_result.order_unload_ticks
            }
            self.robot_cache[agv_id] = {
                "path": path,
                "seq": eval_result.robot_order_seq.get(agv_id, []),
                "unloads": unloads,
                "node_res": node_res,
                "edge_res": edge_res,
            }

        if self.use_rust and isinstance(self._inc_res_table, RustReservationTable):
            if self._inc_backup_table is None or not isinstance(self._inc_backup_table, RustReservationTable):
                self._inc_backup_table = self._new_res_table()
            self._inc_backup_table.restore_from(self._inc_res_table)

    def set_incumbent(
        self,
        assignment: dict[str, str],
        eval_result: EvalResult,
        op_priority: Optional[dict[str, int]] = None,
        op_res_ready_hint: Optional[dict[str, int]] = None,
        op_bay_overrides: Optional[dict[str, str]] = None,
        task_order_override: Optional[dict[str, list[str]]] = None,
    ) -> None:
        """Promote an evaluated valid candidate to the new incumbent base."""
        if eval_result.valid:
            prio = op_priority or {
                op.op_id: max(self.t_commit, op.due_step - op.proc_steps)
                for op in self.snapshot.operations
            }
            ctx_key = (
                tuple(sorted(prio.items())),
                tuple(sorted(op_res_ready_hint.items())) if op_res_ready_hint is not None else None,
                tuple(sorted(op_bay_overrides.items())) if op_bay_overrides else None,
                tuple((k, tuple(v)) for k, v in sorted(task_order_override.items()))
                if task_order_override
                else None,
            )
            self._seed_cache(
                assignment,
                eval_result,
                ctx_key,
                op_bay_overrides=op_bay_overrides,
                task_order_override=task_order_override,
            )
