"""CVCR Incremental Evaluator (E2 / B2) for AMR-Scheduler.

Faithfully implements the 2x2 factorial decomposition from RAIL-CVCR:
- B1: Preserve unchanged robot assignments + Full reservation rebuild
- B2: Preserve unchanged robot paths + Incremental Spacetime Reservation Cache (E2)

Guarantees:
1. Zero unsafe cache reuse: every output is strictly collision-free by construction.
2. Differential equivalence: On any candidate assignment, B2 outputs identical
   feasibility and objective score to B1.
3. Telemetry tracking: call counts, incremental hits, fallbacks, search latencies.
"""
from __future__ import annotations

import copy
import heapq
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from amr_sched.cvcr.snapshot import NativeSnapshot, OperationSnapshot, TaskSnapshot
from amr_sched.network import Network
from amr_sched.reservation import Interval, ReservationTable


def physical_edge_id(src: str, dst: str) -> tuple[str, str]:
    return (src, dst) if src <= dst else (dst, src)


@dataclass
class EvalResult:
    robot_paths: dict[str, list[tuple[str, int]]]  # agv_id -> [(node, step), ...]
    robot_order_seq: dict[str, list[str]]
    order_unload_ticks: dict[str, int]
    op_start: dict[str, int]
    op_end: dict[str, int]
    op_res_ready: dict[str, int]
    objective: float
    valid: bool
    starvation_ticks: int
    eval_latency_ms: float
    mode: str  # "B1" | "B2"


class CVCREvaluator:
    """Spatiotemporal Evaluator with Incremental Reservation Caching (E2/B2)."""

    def __init__(
        self,
        network: Network,
        snapshot: NativeSnapshot,
        mode: str = "B2",  # "B1" or "B2"
        max_speed: float = 1.0,
        dt: float = 1.0,
        sim_horizon: int = 600,
        lambda_change: float = 15.0,
        alpha_makespan: float = 3.0,
    ) -> None:
        self.network = network
        self.snapshot = snapshot
        self.mode = mode
        self.max_speed = max_speed
        self.dt = dt
        self.sim_horizon = sim_horizon
        self.lambda_change = lambda_change
        self.alpha_makespan = alpha_makespan

        self.t_commit = snapshot.tick
        self.tasks_by_id = {t.task_id: t for t in snapshot.tasks}
        self.robots_by_id = {r.robot_id: r for r in snapshot.robots}
        self.ops_by_id = {op.op_id: op for op in snapshot.operations}

        # Base snapshot assignment (to compute reassignment penalty)
        self.snapshot_assign = {
            t.task_id: t.assigned_agv
            for t in snapshot.tasks
            if t.assigned_agv is not None
        }

        # Heuristic distance cache for A*
        self._dist_cache: dict[str, dict[str, float]] = {}

        # E2 Incumbent Cache
        self.base_assign: dict[str, str] = {}
        self.base_ctx_key: tuple[Any, ...] | None = None
        self.robot_cache: dict[str, dict[str, Any]] = {}
        # Stores delivered unloads from snapshot
        self.delivered_unloads: dict[str, int] = {
            t.task_id: t.complete_step
            for t in snapshot.tasks
            if t.status.lower() == "completed" and t.complete_step is not None
        }

        # Telemetry
        self.calls = 0
        self.b1_calls = 0
        self.b2_inc_hits = 0
        self.b2_fallbacks = 0
        self.eval_lats_ms: list[float] = []

    def _heuristic_table(self, goal: str) -> dict[str, float]:
        if goal in self._dist_cache:
            return self._dist_cache[goal]
        rev: dict[str, list[tuple[str, float]]] = {n: [] for n in self.network.nodes}
        for (src, dst), e in self.network.edges.items():
            rev[dst].append((src, e.traversal_time(self.max_speed)))
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
        edge = self.network.edge(src, dst)
        return max(1, int(math.ceil(edge.length / min(self.max_speed, edge.speed_limit) / self.dt)))

    def _space_time_route(
        self,
        start_node: str,
        goal_node: str,
        start_step: int,
        reservations: ReservationTable,
        agv_id: str,
        max_horizon: Optional[int] = None,
    ) -> list[tuple[str, int]] | None:
        """Finds collision-free path from start_node at start_step to goal_node."""
        if start_node == goal_node:
            return [(start_node, start_step)]

        h_tbl = self._heuristic_table(goal_node)
        if goal_node not in h_tbl:
            return None

        def h(node: str) -> int:
            return int(math.floor(h_tbl.get(node, 1000.0) / self.dt))

        counter = itertools.count()
        frontier: list[tuple[int, int, int, str]] = [
            (h(start_node), 0, next(counter), start_node)
        ]
        best_g: dict[tuple[str, int], int] = {(start_node, start_step): 0}
        parent: dict[tuple[str, int], tuple[str, int] | None] = {(start_node, start_step): None}

        goal_state: tuple[str, int] | None = None
        expansions = 0
        max_exp = 15000
        horizon = max_horizon if max_horizon is not None else self.sim_horizon
        limit_step = start_step + horizon

        while frontier:
            expansions += 1
            if expansions > max_exp:
                return None
            f, g, _, node = heapq.heappop(frontier)
            t = start_step + g
            if t > limit_step:
                continue
            state = (node, t)
            if best_g.get(state, 10**18) < g:
                continue
            if node == goal_node:
                goal_state = state
                break

            # 1. Wait action
            if t + 1 <= limit_step:
                span = Interval(t, t + 1)
                cap = self.network.node(node).capacity
                if reservations.is_node_free(node, span, cap, exclude_agv=agv_id):
                    ns = (node, t + 1)
                    if g + 1 < best_g.get(ns, 10**18):
                        best_g[ns] = g + 1
                        parent[ns] = state
                        heapq.heappush(frontier, (g + 1 + h(node), g + 1, next(counter), node))

            # 2. Move along edges
            for e in self.network.out_edges(node):
                n_steps = self._edge_steps(e.src, e.dst)
                arr = t + n_steps
                if arr > limit_step:
                    continue
                edge_span = Interval(t, arr)
                phys = physical_edge_id(e.src, e.dst)
                edge_cap = min(e.capacity, self.network.edge(e.dst, e.src).capacity) if self.network.has_edge(e.dst, e.src) else e.capacity
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
                if ng < best_g.get(ns, 10**18):
                    best_g[ns] = ng
                    parent[ns] = state
                    heapq.heappush(frontier, (ng + h(e.dst), ng, next(counter), e.dst))

        if goal_state is None:
            return None

        # Reconstruct path
        chain: list[tuple[str, int]] = []
        cur = goal_state
        while cur is not None:
            chain.append(cur)
            cur = parent[cur]
        chain.reverse()
        return chain

    def _propagate_production(
        self,
        order_unload_ticks: dict[str, int],
        op_priority: dict[str, int],
    ) -> tuple[dict[str, int], dict[str, int], dict[str, int], int, int, int]:
        """Propagates production schedule over ops, crews, and tools."""
        op_start: dict[str, int] = {}
        op_end: dict[str, int] = {}
        op_res_ready: dict[str, int] = {}
        bay_free: dict[str, int] = {}
        crew_free: dict[int, int] = {}
        tool_free: dict[int, int] = {}

        ops = sorted(
            self.snapshot.operations,
            key=lambda op: (
                0 if op.started else 1,
                op_priority.get(op.op_id, 0),
                op.due_step,
            ),
        )
        resolved: set[str] = set()
        starvation_idle = 0

        for _ in range(len(ops) + 2):
            for op in ops:
                if op.op_id in resolved:
                    continue
                if op.predecessor_op_id and op.predecessor_op_id not in resolved:
                    continue
                if op.started and op.start_step is not None and op.end_step is not None:
                    op_start[op.op_id] = op.start_step
                    op_end[op.op_id] = op.end_step
                    op_res_ready[op.op_id] = op.start_step
                    bay_free[op.bay_id] = max(bay_free.get(op.bay_id, 0), op.end_step)
                    crew_free[op.crew_id] = max(crew_free.get(op.crew_id, 0), op.end_step)
                    tool_free[op.tool_id] = max(tool_free.get(op.tool_id, 0), op.end_step)
                else:
                    mat_t = max(
                        (order_unload_ticks.get(tid, 250) for tid in op.kit_task_ids),
                        default=self.t_commit,
                    )
                    pred_t = op_end[op.predecessor_op_id] if op.predecessor_op_id else 0
                    r_ready = max(
                        self.t_commit,
                        pred_t,
                        bay_free.get(op.bay_id, self.t_commit),
                        crew_free.get(op.crew_id, self.t_commit),
                        tool_free.get(op.tool_id, self.t_commit),
                    )
                    op_res_ready[op.op_id] = r_ready
                    st_t = max(r_ready, mat_t)
                    starvation_idle += max(0, mat_t - r_ready)
                    en_t = st_t + op.proc_steps
                    op_start[op.op_id] = st_t
                    op_end[op.op_id] = en_t
                    bay_free[op.bay_id] = en_t
                    crew_free[op.crew_id] = en_t
                    tool_free[op.tool_id] = en_t
                resolved.add(op.op_id)

        makespan = max(op_end.values()) if op_end else self.t_commit
        tard = sum(
            (12 if op.is_urgent else 4) * max(0, op_end.get(op.op_id, 250) - op.due_step)
            for op in self.snapshot.operations
        )
        return op_start, op_end, op_res_ready, makespan, tard, starvation_idle

    def _route_single_robot(
        self,
        agv_id: str,
        tasks: list[TaskSnapshot],
        res_table: ReservationTable,
        op_priority: dict[str, int],
        op_res_ready_hint: dict[str, int] | None,
    ) -> tuple[list[tuple[str, int]], list[str], dict[str, int], list[tuple[str, Interval]], list[tuple[tuple[str, str], Interval]], bool]:
        """Routes a single AMR through its assigned tasks."""
        rob = self.robots_by_id[agv_id]
        cur_node = rob.cur_node
        cur_t = max(self.t_commit, rob.faulted_until)

        # Sort tasks
        if op_res_ready_hint is not None:
            tasks_sorted = sorted(
                tasks,
                key=lambda t: (
                    0 if t.status.lower() in ("picked", "loading", "unloading") else 1,
                    max(op_res_ready_hint.get(t.kit_op_id or "", 0), t.release_step),
                    0 if (t.kit_op_id and self.ops_by_id.get(t.kit_op_id, None) and self.ops_by_id[t.kit_op_id].is_urgent) else 1,
                ),
            )
        else:
            tasks_sorted = sorted(
                tasks,
                key=lambda t: (
                    0 if t.status.lower() in ("picked", "loading", "unloading") else 1,
                    op_priority.get(t.kit_op_id or "", 0),
                    t.release_step,
                ),
            )

        task_seq = [t.task_id for t in tasks_sorted]
        full_path: list[tuple[str, int]] = []
        unload_ticks: dict[str, int] = {}
        node_res_added: list[tuple[str, Interval]] = []
        edge_res_added: list[tuple[tuple[str, str], Interval]] = []

        all_ok = True
        for t in tasks_sorted:
            # 1. Move to pickup if not already picked up
            if t.status.lower() not in ("picked", "loading", "to_dropoff", "unloading"):
                p1 = self._space_time_route(cur_node, t.pickup, cur_t, res_table, agv_id)
                if p1 is None:
                    all_ok = False
                    unload_ticks[t.task_id] = 250
                    continue
                # Add reservations
                for i in range(len(p1) - 1):
                    u_n, u_t = p1[i]
                    v_n, v_t = p1[i + 1]
                    span = Interval(u_t, v_t)
                    if u_n == v_n:
                        res_table.reserve_node(u_n, span, agv_id)
                        node_res_added.append((u_n, span))
                    else:
                        res_table.reserve_edge(u_n, v_n, span, agv_id)
                        edge_res_added.append(((u_n, v_n), span))
                # Append to full path
                for pt in p1:
                    if not full_path or full_path[-1] != pt:
                        full_path.append(pt)
                cur_t = p1[-1][1]
                cur_node = t.pickup

                # Loading time + release time wait
                load_time = 2
                arr_t = cur_t
                if arr_t < t.release_step:
                    wait_until = t.release_step + load_time
                else:
                    wait_until = arr_t + load_time
                wait_span = Interval(cur_t, wait_until)
                res_table.reserve_node(cur_node, wait_span, agv_id)
                node_res_added.append((cur_node, wait_span))
                cur_t = wait_until
                full_path.append((cur_node, cur_t))

            # 2. Move to dropoff
            p2 = self._space_time_route(cur_node, t.dropoff, cur_t, res_table, agv_id)
            if p2 is None:
                all_ok = False
                unload_ticks[t.task_id] = 250
                continue
            for i in range(len(p2) - 1):
                u_n, u_t = p2[i]
                v_n, v_t = p2[i + 1]
                span = Interval(u_t, v_t)
                if u_n == v_n:
                    res_table.reserve_node(u_n, span, agv_id)
                    node_res_added.append((u_n, span))
                else:
                    res_table.reserve_edge(u_n, v_n, span, agv_id)
                    edge_res_added.append(((u_n, v_n), span))
            for pt in p2:
                if not full_path or full_path[-1] != pt:
                    full_path.append(pt)
            cur_t = p2[-1][1]
            cur_node = t.dropoff

            # Unloading dwell
            unload_time = 2
            unload_span = Interval(cur_t, cur_t + unload_time)
            res_table.reserve_node(cur_node, unload_span, agv_id)
            node_res_added.append((cur_node, unload_span))
            cur_t += unload_time
            full_path.append((cur_node, cur_t))
            unload_ticks[t.task_id] = cur_t

        # Pin final position to sim_horizon
        if cur_t < self.sim_horizon:
            rest_span = Interval(cur_t, self.sim_horizon)
            res_table.reserve_node(cur_node, rest_span, agv_id)
            node_res_added.append((cur_node, rest_span))

        return full_path, task_seq, unload_ticks, node_res_added, edge_res_added, all_ok

    def evaluate(
        self,
        assignment: dict[str, str],
        op_priority: Optional[dict[str, int]] = None,
        op_res_ready_hint: Optional[dict[str, int]] = None,
    ) -> EvalResult:
        """Evaluates assignment under configured mode (B1 or B2)."""
        t0 = time.perf_counter()
        self.calls += 1

        prio = op_priority or {
            op.op_id: max(self.t_commit, op.due_step - op.proc_steps)
            for op in self.snapshot.operations
        }

        if self.mode == "B1":
            res = self._eval_b1(assignment, prio, op_res_ready_hint)
        else:  # B2
            res = self._eval_b2(assignment, prio, op_res_ready_hint)

        lat = (time.perf_counter() - t0) * 1000.0
        self.eval_lats_ms.append(lat)
        return res

    def _apply_preserved_reservations(
        self,
        res_table: ReservationTable,
        changed_rids: list[str],
    ) -> tuple[dict[str, list[tuple[str, int]]], dict[str, list[str]], dict[str, int]]:
        order_unload_ticks: dict[str, int] = dict(self.delivered_unloads)
        robot_paths: dict[str, list[tuple[str, int]]] = {}
        robot_order_seq: dict[str, list[str]] = {}

        for r in self.snapshot.robots:
            rid = r.robot_id
            if rid not in changed_rids and rid in self.robot_cache:
                rc = self.robot_cache[rid]
                robot_paths[rid] = rc["path"]
                robot_order_seq[rid] = rc["seq"]
                order_unload_ticks.update(rc["unloads"])
                for node, span in rc["node_res"]:
                    res_table.reserve_node(node, span, rid)
                for edge, span in rc["edge_res"]:
                    res_table.reserve_edge(edge[0], edge[1], span, rid)
        return robot_paths, robot_order_seq, order_unload_ticks

    def _eval_b1(
        self,
        assignment: dict[str, str],
        op_priority: dict[str, int],
        op_res_ready_hint: Optional[dict[str, int]] = None,
    ) -> EvalResult:
        """B1: Preserves unchanged robots' incumbent paths, fully rebuilds reservation table."""
        self.b1_calls += 1
        t0 = time.perf_counter()

        res_table = ReservationTable()

        # Determine changed robots relative to incumbent base
        changed_rids = None
        if self.base_assign and self.robot_cache:
            changed_rids = sorted(
                {assignment[tid] for tid in assignment if assignment[tid] != self.base_assign.get(tid)}
                | {self.base_assign[tid] for tid in self.base_assign if assignment.get(tid) != self.base_assign[tid]}
            )

        if changed_rids is not None and 0 < len(changed_rids) < len(self.snapshot.robots):
            robot_paths, robot_order_seq, order_unload_ticks = self._apply_preserved_reservations(
                res_table, changed_rids
            )
            rids_to_route = changed_rids
        else:
            robot_paths = {}
            robot_order_seq = {}
            order_unload_ticks = dict(self.delivered_unloads)
            rids_to_route = sorted(r.robot_id for r in self.snapshot.robots)

        by_robot: dict[str, list[TaskSnapshot]] = {rid: [] for rid in rids_to_route}
        for tid, t in self.tasks_by_id.items():
            if t.status.lower() == "completed":
                continue
            agv_id = assignment.get(tid)
            if agv_id in by_robot:
                by_robot[agv_id].append(t)

        all_ok = True
        for agv_id in rids_to_route:
            r_tasks = by_robot[agv_id]
            path, seq, unloads, n_res, e_res, ok = self._route_single_robot(
                agv_id, r_tasks, res_table, op_priority, op_res_ready_hint
            )
            robot_paths[agv_id] = path
            robot_order_seq[agv_id] = seq
            order_unload_ticks.update(unloads)
            if not ok:
                all_ok = False

        op_s, op_e, op_rr, mk, tard, starv = self._propagate_production(order_unload_ticks, op_priority)
        tot_deliv = sum(order_unload_ticks.get(t.task_id, 250) for t in self.snapshot.tasks if t.status.lower() != "completed")

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

        lat = (time.perf_counter() - t0) * 1000.0
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
            eval_latency_ms=lat,
            mode="B1",
        )

    def _eval_b2(
        self,
        assignment: dict[str, str],
        op_priority: dict[str, int],
        op_res_ready_hint: Optional[dict[str, int]] = None,
    ) -> EvalResult:
        """B2: Incremental Reservation-Cached Evaluator (E2).

        Preserves unchanged robots' reservations and re-routes ONLY changed robots.
        Falls back to B1 if context changes or incremental route encounters conflict.
        """
        ctx_key = (
            tuple(sorted(op_priority.items())),
            tuple(sorted(op_res_ready_hint.items())) if op_res_ready_hint is not None else None,
        )

        # If base is missing or context changed, run full B1
        if self.base_ctx_key != ctx_key or not self.robot_cache:
            return self._eval_b1(assignment, op_priority, op_res_ready_hint)

        # Identify changed robots
        changed_rids = sorted(
            {assignment[tid] for tid in assignment if assignment[tid] != self.base_assign.get(tid)}
            | {self.base_assign[tid] for tid in self.base_assign if assignment.get(tid) != self.base_assign[tid]}
        )

        # If all robots changed or no robots changed, fall back to B1 without overwriting base
        if not changed_rids or len(changed_rids) >= len(self.snapshot.robots):
            self.b2_fallbacks += 1
            return self._eval_b1(assignment, op_priority, op_res_ready_hint)

        # Incremental evaluation: preserve UNCHANGED robots' reservations
        t0 = time.perf_counter()
        res_table = ReservationTable()
        robot_paths, robot_order_seq, order_unload_ticks = self._apply_preserved_reservations(
            res_table, changed_rids
        )

        # Route ONLY changed robots
        by_robot: dict[str, list[TaskSnapshot]] = {rid: [] for rid in changed_rids}
        for tid, t in self.tasks_by_id.items():
            if t.status.lower() == "completed":
                continue
            agv_id = assignment.get(tid)
            if agv_id in by_robot:
                by_robot[agv_id].append(t)

        all_ok = True
        for rid in changed_rids:
            r_tasks = by_robot[rid]
            path, seq, unloads, n_res, e_res, ok = self._route_single_robot(
                rid, r_tasks, res_table, op_priority, op_res_ready_hint
            )
            robot_paths[rid] = path
            robot_order_seq[rid] = seq
            order_unload_ticks.update(unloads)
            if not ok:
                all_ok = False
                break

        # If incremental routing failed, fall back to B1
        if not all_ok:
            self.b2_fallbacks += 1
            return self._eval_b1(assignment, op_priority, op_res_ready_hint)

        self.b2_inc_hits += 1
        op_s, op_e, op_rr, mk, tard, starv = self._propagate_production(order_unload_ticks, op_priority)
        tot_deliv = sum(order_unload_ticks.get(t.task_id, 250) for t in self.snapshot.tasks if t.status.lower() != "completed")

        chg = sum(
            1
            for tid, agv_id in assignment.items()
            if tid in self.snapshot_assign
            and self.snapshot_assign[tid] != agv_id
            and self.tasks_by_id[tid].status.lower() == "assigned"
        )

        obj = float(tard + self.alpha_makespan * mk + 0.5 * tot_deliv + self.lambda_change * chg)
        lat = (time.perf_counter() - t0) * 1000.0

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
            eval_latency_ms=lat,
            mode="B2",
        )

    def _seed_cache(
        self,
        assignment: dict[str, str],
        eval_result: EvalResult,
        ctx_key: tuple[Any, ...],
    ) -> None:
        """Seeds the incremental robot cache with the incumbent solution."""
        self.base_assign = dict(assignment)
        self.base_ctx_key = ctx_key
        self.robot_cache = {}

        # Re-derive node and edge reservations per robot from evaluated paths
        for agv_id, path in eval_result.robot_paths.items():
            node_res: list[tuple[str, Interval]] = []
            edge_res: list[tuple[tuple[str, str], Interval]] = []
            for i in range(len(path) - 1):
                u_n, u_t = path[i]
                v_n, v_t = path[i + 1]
                span = Interval(u_t, v_t)
                if u_n == v_n:
                    node_res.append((u_n, span))
                else:
                    edge_res.append(((u_n, v_n), span))
            if path and path[-1][1] < self.sim_horizon:
                node_res.append((path[-1][0], Interval(path[-1][1], self.sim_horizon)))

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

    def set_incumbent(self, assignment: dict[str, str], eval_result: EvalResult) -> None:
        """Manually promote an evaluated candidate to the new incumbent base."""
        if eval_result.valid:
            prio = {
                op.op_id: max(self.t_commit, op.due_step - op.proc_steps)
                for op in self.snapshot.operations
            }
            ctx_key = (
                tuple(sorted(prio.items())),
                None,
            )
            self._seed_cache(assignment, eval_result, ctx_key)
