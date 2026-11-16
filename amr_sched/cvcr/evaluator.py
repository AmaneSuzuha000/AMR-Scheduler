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
import time
from dataclasses import dataclass
from typing import Any, Optional

from amr_sched.cvcr.snapshot import NativeSnapshot, TaskSnapshot
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
        self._rust_planner = (
            RustSpaceTimeAStarPlanner(max_speed=max_speed, dt=dt, max_expansions=6_000)
            if self.use_rust
            else None
        )
        self._haven_nodes = [
            nd.node_id
            for nd in sorted(network.nodes.values(), key=lambda x: (0 if x.kind == NodeKind.BUFFER else 1, x.node_id))
            if nd.kind in (NodeKind.BUFFER, NodeKind.AISLE)
        ]

        self.t_commit = snapshot.tick
        self.tasks_by_id = {t.task_id: t for t in snapshot.tasks}
        self.robots_by_id = {r.robot_id: r for r in snapshot.robots}
        self.ops_by_id = {op.op_id: op for op in snapshot.operations}

        self.snapshot_assign = {
            t.task_id: t.assigned_agv
            for t in snapshot.tasks
            if t.assigned_agv is not None
        }
        self._dist_cache: dict[str, dict[str, float]] = {}

        # E2 / B2 persistent incremental cache
        self.base_assign: dict[str, str] = {}
        self.base_ctx_key: tuple[Any, ...] | None = None
        self.robot_cache: dict[str, dict[str, Any]] = {}
        self._inc_res_table: ReservationTable | RustReservationTable | None = None

        self.delivered_unloads: dict[str, int] = {
            t.task_id: int(t.complete_step)
            for t in snapshot.tasks
            if t.status.lower() == "completed" and t.complete_step is not None
        }

        # Telemetry
        self.calls = 0
        self.a1_calls = 0
        self.b1_calls = 0
        self.b2_inc_hits = 0
        self.b2_fallbacks = 0
        self.eval_lats_ms: list[float] = []

    def _new_res_table(self) -> ReservationTable | RustReservationTable:
        return RustReservationTable() if self.use_rust else ReservationTable()

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

    def _edge_steps(self, src: str, dst: str) -> int:
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

        counter = itertools.count()
        frontier: list[tuple[int, int, int, str]] = [
            (h(start_node), 0, next(counter), start_node)
        ]
        best_g: dict[tuple[str, int], int] = {(start_node, start_step): 0}
        parent: dict[tuple[str, int], tuple[str, int] | None] = {(start_node, start_step): None}

        goal_state: tuple[str, int] | None = None
        expansions = 0
        max_exp = 20_000

        while frontier:
            expansions += 1
            if expansions > max_exp:
                return None
            _, g, _, node = heapq.heappop(frontier)
            t = start_step + g
            if t > limit_step:
                continue
            state = (node, t)
            if best_g.get(state, 10**18) < g:
                continue
            if node == goal_node:
                goal_state = state
                break

            if t + 1 <= limit_step:
                span = Interval(t, t + 1)
                cap = self.network.node(node).capacity
                if reservations.is_node_free(node, span, cap, exclude_agv=agv_id):
                    ns = (node, t + 1)
                    if g + 1 < best_g.get(ns, 10**18):
                        best_g[ns] = g + 1
                        parent[ns] = state
                        heapq.heappush(frontier, (g + 1 + h(node), g + 1, next(counter), node))

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
                if ng < best_g.get(ns, 10**18):
                    best_g[ns] = ng
                    parent[ns] = state
                    heapq.heappush(frontier, (ng + h(e.dst), ng, next(counter), e.dst))

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
                if op.predecessor_op_id and op.predecessor_op_id not in resolved:
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
                    pred_t = op_end[op.predecessor_op_id] if op.predecessor_op_id else 0
                    r_ready = max(
                        self.t_commit,
                        pred_t,
                        bay_free.get(bay, self.t_commit),
                        crew_free.get(crew, self.t_commit),
                        tool_free.get(op.tool_id, self.t_commit),
                    )
                    op_res_ready[op.op_id] = r_ready
                    st_t = max(r_ready, mat_t)
                    starvation_idle += max(0, mat_t - r_ready)
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
        cur_t = max(self.t_commit, rob.faulted_until)

        if op_res_ready_hint is not None:
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
                            for i in range(len(p_park) - 1):
                                u_n, u_t = p_park[i]
                                v_n, v_t = p_park[i + 1]
                                if v_t <= u_t:
                                    continue
                                span = Interval(u_t, v_t)
                                if u_n == v_n:
                                    res_table.reserve_node(u_n, span, agv_id)
                                    node_res_added.append((u_n, span))
                                else:
                                    res_table.reserve_edge(u_n, v_n, span, agv_id)
                                    edge_res_added.append(((u_n, v_n), span))
                            cur_t = p_park[-1][1]
                            cur_node = hv
                            full_path.extend(p_park)
                            break
            if cur_t < self.sim_horizon:
                rest_span = Interval(cur_t, self.sim_horizon)
                res_table.reserve_node(cur_node, rest_span, agv_id)
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
                for i in range(len(p1) - 1):
                    u_n, u_t = p1[i]
                    v_n, v_t = p1[i + 1]
                    if v_t <= u_t:
                        continue
                    span = Interval(u_t, v_t)
                    if u_n == v_n:
                        res_table.reserve_node(u_n, span, agv_id)
                        node_res_added.append((u_n, span))
                    else:
                        res_table.reserve_edge(u_n, v_n, span, agv_id)
                        edge_res_added.append(((u_n, v_n), span))
                for pt in p1:
                    if not full_path or full_path[-1] != pt:
                        full_path.append(pt)
                cur_t = p1[-1][1]
                cur_node = t.pickup

                load_time = 2
                wait_until = max(cur_t, t.release_step) + load_time
                if wait_until > cur_t and wait_until <= self.sim_horizon:
                    wait_span = Interval(cur_t, wait_until)
                    res_table.reserve_node(cur_node, wait_span, agv_id)
                    node_res_added.append((cur_node, wait_span))
                cur_t = min(wait_until, self.sim_horizon)
                full_path.append((cur_node, cur_t))

            p2 = self._space_time_route(cur_node, dropoff, cur_t, res_table, agv_id)
            if p2 is None:
                all_ok = False
                for rem_t in tasks_sorted[idx_t:]:
                    unload_ticks[rem_t.task_id] = self.sim_horizon
                break
            for i in range(len(p2) - 1):
                u_n, u_t = p2[i]
                v_n, v_t = p2[i + 1]
                if v_t <= u_t:
                    continue
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
            cur_node = dropoff

            unload_time = 2
            end_u = min(cur_t + unload_time, self.sim_horizon)
            if end_u > cur_t:
                unload_span = Interval(cur_t, end_u)
                res_table.reserve_node(cur_node, unload_span, agv_id)
                node_res_added.append((cur_node, unload_span))
            cur_t = end_u
            full_path.append((cur_node, cur_t))
            unload_ticks[t.task_id] = cur_t

        # Haven parking at end of active route (mirrors railway_prod_model.py: never squat on capacity=1 workstation)
        if all_ok and self.network.node(cur_node).kind not in (NodeKind.BUFFER, NodeKind.AISLE):
            for hv in self._haven_nodes:
                hv_cap = self.network.node(hv).capacity
                if res_table.is_node_free(hv, Interval(cur_t, self.sim_horizon), hv_cap, exclude_agv=agv_id):
                    p_park = self._space_time_route(cur_node, hv, cur_t, res_table, agv_id, max_horizon=200)
                    if p_park is not None:
                        for i in range(len(p_park) - 1):
                            u_n, u_t = p_park[i]
                            v_n, v_t = p_park[i + 1]
                            if v_t <= u_t:
                                continue
                            span = Interval(u_t, v_t)
                            if u_n == v_n:
                                res_table.reserve_node(u_n, span, agv_id)
                                node_res_added.append((u_n, span))
                            else:
                                res_table.reserve_edge(u_n, v_n, span, agv_id)
                                edge_res_added.append(((u_n, v_n), span))
                        for pt in p_park:
                            if not full_path or full_path[-1] != pt:
                                full_path.append(pt)
                        cur_t = p_park[-1][1]
                        cur_node = hv
                        break

        if cur_t < self.sim_horizon:
            rest_span = Interval(cur_t, self.sim_horizon)
            res_table.reserve_node(cur_node, rest_span, agv_id)
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
    ) -> EvalResult:
        """Evaluate candidate assignment under configured variant (A1, B1, or B2)."""
        t0 = time.perf_counter()
        self.calls += 1

        prio = op_priority or {
            op.op_id: max(self.t_commit, op.due_step - op.proc_steps)
            for op in self.snapshot.operations
        }

        if self.mode == "A1":
            res = self._eval_a1(assignment, prio, op_res_ready_hint, op_bay_overrides, op_crew_overrides)
        elif self.mode == "B1":
            res = self._eval_b1(assignment, prio, op_res_ready_hint, op_bay_overrides, op_crew_overrides)
        else:
            res = self._eval_b2(assignment, prio, op_res_ready_hint, op_bay_overrides, op_crew_overrides)

        lat = (time.perf_counter() - t0) * 1000.0
        self.eval_lats_ms.append(lat)
        return res

    def _eval_a1(
        self,
        assignment: dict[str, str],
        op_priority: dict[str, int],
        op_res_ready_hint: Optional[dict[str, int]] = None,
        op_bay_overrides: Optional[dict[str, str]] = None,
        op_crew_overrides: Optional[dict[str, int]] = None,
    ) -> EvalResult:
        """A1: Full replan of ALL robots from scratch + full reservation build."""
        self.a1_calls += 1
        t0 = time.perf_counter()
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

        for agv_id in sorted(by_robot.keys(), key=lambda r: (0 if not by_robot[r] else 1, r)):
            path, seq, unloads, _, _, ok = self._route_single_robot(
                agv_id, by_robot[agv_id], res_table, op_priority, op_res_ready_hint, op_bay_overrides
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
            eval_latency_ms=(time.perf_counter() - t0) * 1000.0,
            mode="A1",
        )

    def _eval_b1(
        self,
        assignment: dict[str, str],
        op_priority: dict[str, int],
        op_res_ready_hint: Optional[dict[str, int]] = None,
        op_bay_overrides: Optional[dict[str, str]] = None,
        op_crew_overrides: Optional[dict[str, int]] = None,
    ) -> EvalResult:
        """B1: Preserve unchanged robots' paths, reconstruct full reservation table from scratch."""
        self.b1_calls += 1
        t0 = time.perf_counter()
        res_table = self._new_res_table()

        changed_rids = None
        if self.base_assign and self.robot_cache and not op_bay_overrides:
            changed_rids = sorted(
                {assignment[tid] for tid in assignment if assignment[tid] != self.base_assign.get(tid)}
                | {self.base_assign[tid] for tid in self.base_assign if assignment.get(tid) != self.base_assign[tid]}
            )

        order_unload_ticks: dict[str, int] = dict(self.delivered_unloads)
        robot_paths: dict[str, list[tuple[str, int]]] = {}
        robot_order_seq: dict[str, list[str]] = {}

        if changed_rids is not None and 0 < len(changed_rids) < len(self.snapshot.robots):
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
        for agv_id in sorted(rids_to_route, key=lambda r: (0 if not by_robot[r] else 1, r)):
            path, seq, unloads, _, _, ok = self._route_single_robot(
                agv_id, by_robot[agv_id], res_table, op_priority, op_res_ready_hint, op_bay_overrides
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
            eval_latency_ms=(time.perf_counter() - t0) * 1000.0,
            mode="B1",
        )

    def _restore_inc_table(self, changed_rids: list[str]) -> None:
        """Restore persistent incremental reservation table back to incumbent state."""
        assert self._inc_res_table is not None
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
    ) -> EvalResult:
        """B2: Persistent Incremental Reservation-Cached Evaluator (E2)."""
        ctx_key = (
            tuple(sorted(op_priority.items())),
            tuple(sorted(op_res_ready_hint.items())) if op_res_ready_hint is not None else None,
            tuple(sorted(op_bay_overrides.items())) if op_bay_overrides else None,
        )

        if self.base_ctx_key != ctx_key or not self.robot_cache or self._inc_res_table is None:
            return self._eval_b1(assignment, op_priority, op_res_ready_hint, op_bay_overrides, op_crew_overrides)

        changed_rids = sorted(
            {assignment[tid] for tid in assignment if assignment[tid] != self.base_assign.get(tid)}
            | {self.base_assign[tid] for tid in self.base_assign if assignment.get(tid) != self.base_assign[tid]}
        )

        if not changed_rids or len(changed_rids) >= len(self.snapshot.robots):
            self.b2_fallbacks += 1
            return self._eval_b1(assignment, op_priority, op_res_ready_hint, op_bay_overrides, op_crew_overrides)

        t0 = time.perf_counter()
        res_table = self._inc_res_table

        # Release ONLY changed robots in-place from persistent reservation table
        for rid in changed_rids:
            res_table.release_all(rid)

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

        by_robot: dict[str, list[TaskSnapshot]] = {rid: [] for rid in changed_rids}
        for tid, t in self.tasks_by_id.items():
            if t.status.lower() == "completed":
                continue
            agv_id = assignment.get(tid)
            if agv_id in by_robot:
                by_robot[agv_id].append(t)

        all_ok = True
        for rid in sorted(changed_rids, key=lambda r: (0 if not by_robot[r] else 1, r)):
            path, seq, unloads, _, _, ok = self._route_single_robot(
                rid, by_robot[rid], res_table, op_priority, op_res_ready_hint, op_bay_overrides
            )
            robot_paths[rid] = path
            robot_order_seq[rid] = seq
            order_unload_ticks.update(unloads)
            if not ok:
                all_ok = False
                break
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
            eval_latency_ms=(time.perf_counter() - t0) * 1000.0,
            mode="B2",
        )

    def _seed_cache(
        self,
        assignment: dict[str, str],
        eval_result: EvalResult,
        ctx_key: tuple[Any, ...],
    ) -> None:
        """Seed the incremental robot cache and persistent reservation table."""
        self.base_assign = dict(assignment)
        self.base_ctx_key = ctx_key
        self.robot_cache = {}
        self._inc_res_table = self._new_res_table()

        for agv_id, path in eval_result.robot_paths.items():
            node_res: list[tuple[str, Interval]] = []
            edge_res: list[tuple[tuple[str, str], Interval]] = []
            for i in range(len(path) - 1):
                u_n, u_t = path[i]
                v_n, v_t = path[i + 1]
                if v_t <= u_t:
                    continue
                span = Interval(u_t, v_t)
                if u_n == v_n:
                    node_res.append((u_n, span))
                    self._inc_res_table.reserve_node(u_n, span, agv_id)
                else:
                    edge_res.append(((u_n, v_n), span))
                    self._inc_res_table.reserve_edge(u_n, v_n, span, agv_id)
            if path and path[-1][1] < self.sim_horizon:
                rest_iv = Interval(path[-1][1], self.sim_horizon)
                node_res.append((path[-1][0], rest_iv))
                self._inc_res_table.reserve_node(path[-1][0], rest_iv, agv_id)

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

    def set_incumbent(
        self,
        assignment: dict[str, str],
        eval_result: EvalResult,
        op_priority: Optional[dict[str, int]] = None,
        op_res_ready_hint: Optional[dict[str, int]] = None,
        op_bay_overrides: Optional[dict[str, str]] = None,
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
            )
            self._seed_cache(assignment, eval_result, ctx_key)
