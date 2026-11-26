"""Distilled CPS-A* Mechanisms (`W + P + R`) over Incremental `CVCREvaluator` (`CPR-B2`).

Implements our selective "drop 2, absorb 2" engineering synthesis without altering frozen
production `amr_sched/` files:
  - Dropped: inadmissible look-ahead heuristic (`L=1`, Fig. 12 hurt throughput) and heavy 3D
    Gaussian convolution (`S | L=0` gain only -0.005..+0.040 in Table D.2).
  - Absorbed Factor W (`SlotReservationPressureTable`): O(1) delta-slot occupancy Psi(v, floor(t/delta))
    built from exact scheduled trajectories in fixed background & prior batch paths; penalizes
    wait transitions (when `u != goal`) & congested single-lane corridor entries, keeping h(v) admissible.
  - Absorbed Factor P (`enable_congestion_priority`): sorts active uncommitted robots using task
    urgency, starvation age u_i(t), and local node congestion pressure Psi(cur_node, t_commit) so
    chokepoint occupants vacate early.
  - Absorbed Factor R (`enable_bounded_rollback`): inside active uncommitted `rids_to_route` ONLY
    (never rolling back pinned custody owners, `__L1_OBSTACLE_*__`, or unchanged cached robots),
    when robot i fails and eval deadline headroom >= `min_headroom_ms_for_repair`, releases top
    blocker j in active batch, plans i first then re-plans j, restoring atomically on failure.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
from typing import Any, Optional

from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult, physical_edge_id
from amr_sched.cvcr.snapshot import NativeSnapshot, TaskSnapshot
from amr_sched.network import Network, NodeKind
from amr_sched.reservation import Interval, ReservationTable
from amr_sched.rust_reservation import RustReservationTable


@dataclass(frozen=True)
class CPRRoutingConfig:
    """Orthogonal configuration switches for 2^3 factorial validation."""

    enable_slot_wait_pressure: bool = False   # Factor W
    enable_congestion_priority: bool = False  # Factor P
    enable_bounded_rollback: bool = False     # Factor R
    slot_delta: int = 3
    wait_gamma_q3: int = 6
    wait_cap_q3: int = 24
    edge_choke_gamma_q3: int = 4
    edge_cap_q3: int = 16
    prio_age_tau: float = 0.25
    prio_congestion_kappa: float = 4.0
    max_repairs_per_eval: int = 2
    min_headroom_ms_for_repair: float = 5.0
    hybrid_rust_slot_projection: bool = True
    allow_exact_slot_retry_python: bool = True


class SlotReservationPressureTable:
    """O(1) sparse slot-level occupancy table derived directly from committed trajectories."""

    def __init__(
        self,
        network: Network,
        *,
        slot_delta: int = 3,
        t_start: int = 0,
        track_horizon: int = 160,
    ) -> None:
        self.network = network
        self.slot_delta = max(1, int(slot_delta))
        self.t_start = int(t_start)
        self.track_horizon = max(16, int(track_horizon))
        self._node_slot: dict[tuple[str, int], int] = {}
        self._phys_edge_slot: dict[tuple[tuple[str, str], int], int] = {}
        self._agv_node_slots: dict[str, set[tuple[str, int]]] = {}
        self._agv_edge_slots: dict[str, set[tuple[tuple[str, str], int]]] = {}

    def remove_agv(self, agv_id: str) -> None:
        nkeys = self._agv_node_slots.pop(agv_id, None)
        if nkeys:
            for k in nkeys:
                nv = self._node_slot.get(k, 0) - 1
                if nv > 0:
                    self._node_slot[k] = nv
                else:
                    self._node_slot.pop(k, None)
        ekeys = self._agv_edge_slots.pop(agv_id, None)
        if ekeys:
            for k in ekeys:
                nv = self._phys_edge_slot.get(k, 0) - 1
                if nv > 0:
                    self._phys_edge_slot[k] = nv
                else:
                    self._phys_edge_slot.pop(k, None)

    def add_agv_path(self, agv_id: str, path: list[tuple[str, int]]) -> None:
        self.remove_agv(agv_id)
        if not path:
            return
        delta = self.slot_delta
        t_lo = self.t_start
        t_hi = self.t_start + self.track_horizon
        nslots: set[tuple[str, int]] = set()
        eslots: set[tuple[tuple[str, str], int]] = set()

        for i in range(len(path) - 1):
            u_n, u_t = path[i]
            v_n, v_t = path[i + 1]
            a = max(t_lo, int(u_t))
            b = min(t_hi, int(v_t))
            if b < a:
                continue
            s_a, s_b = a // delta, b // delta
            if u_n == v_n:
                for sl in range(s_a, s_b + 1):
                    nslots.add((u_n, sl))
            else:
                phys = physical_edge_id(u_n, v_n)
                for sl in range(s_a, s_b + 1):
                    eslots.add((phys, sl))
                nslots.add((u_n, s_a))
                nslots.add((v_n, s_b))
        ln, lt = path[-1]
        if t_lo <= int(lt) <= t_hi:
            nslots.add((ln, int(lt) // delta))

        for k in nslots:
            self._node_slot[k] = self._node_slot.get(k, 0) + 1
        for k in eslots:
            self._phys_edge_slot[k] = self._phys_edge_slot.get(k, 0) + 1
        self._agv_node_slots[agv_id] = nslots
        self._agv_edge_slots[agv_id] = eslots

    def node_slot_count(self, node: str, step: int) -> int:
        return int(self._node_slot.get((node, int(step) // self.slot_delta), 0))

    def edge_slot_count(self, src: str, dst: str, step: int) -> int:
        return int(self._phys_edge_slot.get((physical_edge_id(src, dst), int(step) // self.slot_delta), 0))

    def wait_penalty_q3(
        self,
        node: str,
        step: int,
        *,
        is_goal: bool,
        gamma_q3: int,
        cap_q3: int,
    ) -> int:
        if is_goal:
            return 0
        nd = self.network.nodes.get(node)
        cnt = self.node_slot_count(node, step)
        if cnt <= 0:
            return 0
        # Extra pressure on capacity=1 aisle/junction bottlenecks vs buffered sidings
        mult = 2 if (nd is not None and nd.kind == NodeKind.AISLE and nd.capacity <= 1) else 1
        return min(int(cap_q3), int(gamma_q3) * cnt * mult)

    def edge_entry_penalty_q3(
        self,
        src: str,
        dst: str,
        step: int,
        *,
        edge_cap: int,
        edge_gamma_q3: int,
        edge_cap_q3: int,
    ) -> int:
        e_cnt = self.edge_slot_count(src, dst, step)
        n_cnt = self.node_slot_count(dst, step)
        if e_cnt <= 0 and n_cnt <= 0:
            return 0
        narrow_boost = 1 if edge_cap <= 1 else 0
        raw = int(edge_gamma_q3) * (e_cnt + (n_cnt if narrow_boost else 0))
        return min(int(edge_cap_q3), raw)

    def dynamic_corridor_q3_weights(
        self,
        t_commit: int,
        *,
        horizon_steps: int = 36,
        edge_gamma_q3: int = 4,
        edge_cap_q3: int = 16,
    ) -> dict[tuple[str, str], int]:
        """Project near-horizon slot congestion onto directed edge Q3 penalties (Rust compatible)."""
        s_lo = int(t_commit) // self.slot_delta
        s_hi = (int(t_commit) + max(3, int(horizon_steps))) // self.slot_delta
        out: dict[tuple[str, str], int] = {}
        for (src, dst), e in self.network.edges.items():
            phys = physical_edge_id(src, dst)
            rev_e = self.network.edge(dst, src) if self.network.has_edge(dst, src) else None
            pcap = min(e.capacity, rev_e.capacity) if rev_e is not None else e.capacity
            peak_e = max((self._phys_edge_slot.get((phys, sl), 0) for sl in range(s_lo, s_hi + 1)), default=0)
            peak_n = max((self._node_slot.get((dst, sl), 0) for sl in range(s_lo, s_hi + 1)), default=0)
            score = peak_e * edge_gamma_q3 + (peak_n * (edge_gamma_q3 // 2) if pcap <= 1 else 0)
            if score >= 3:
                out[(src, dst)] = min(int(edge_cap_q3), int(score))
        return out


def select_active_blocker(
    failed_agv: str,
    conflict_hits: dict[str, int],
    active_routed: dict[str, Any],
    *,
    pinned_owners: set[str],
) -> Optional[str]:
    """Select the highest-conflict uncommitted robot j in active_batch (never pinned/cached)."""
    eligible = [
        j for j in active_routed
        if j != failed_agv and j not in pinned_owners and not str(j).startswith("__")
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda j: (int(conflict_hits.get(j, 0)), j))


class CPRIncrementalEvaluator(CVCREvaluator):
    """CVCREvaluator augmented with toggleable CPR factors W, P, and R."""

    def __init__(
        self,
        network: Network,
        snapshot: NativeSnapshot,
        cpr_cfg: CPRRoutingConfig = CPRRoutingConfig(),
        **kwargs: Any,
    ) -> None:
        super().__init__(network=network, snapshot=snapshot, **kwargs)
        self.cpr_cfg = cpr_cfg
        self._caller_tw_q3: dict[tuple[str, str], int] = {}
        self._caller_cutoff: int = 0
        self._active_slot_table: SlotReservationPressureTable | None = None
        self._force_python_slot_route: bool = False
        self._eval_deadline_s: float | None = None
        # Telemetry counters
        self.cpr_repairs_attempted: int = 0
        self.cpr_repairs_succeeded: int = 0
        self.cpr_repairs_failed: int = 0
        self.cpr_candidates_rescued: int = 0
        self.cpr_blocker_cache_misses: int = 0

    def set_eval_deadline(self, deadline_s: float | None) -> None:
        """Bind caller deadline (seconds from clock) so repair respects SLA headroom."""
        self._eval_deadline_s = deadline_s

    def set_traffic_weights(
        self,
        weights: dict[tuple[str, str], int] | None,
        cutoff_step: int = 0,
    ) -> None:
        self._caller_tw_q3 = dict(weights) if weights else {}
        self._caller_cutoff = int(cutoff_step) if weights else 0
        super().set_traffic_weights(weights, cutoff_step=cutoff_step)

    def _pinned_owners_set(self, res_table: ReservationTable | RustReservationTable) -> set[str]:
        owners: set[str] = set()
        for ows in getattr(res_table, "_pinned_nodes", {}).values():
            owners.update(ows)
        for ows in getattr(res_table, "_pinned_edges", {}).values():
            owners.update(ows)
        for r in self.snapshot.robots:
            if str(r.state).lower() in ("faulted", "blocked"):
                owners.add(r.robot_id)
        return owners

    def _probe_corridor_conflicts(
        self,
        failed_rid: str,
        tasks: list[TaskSnapshot],
        res_table: ReservationTable | RustReservationTable,
        active_set: set[str],
    ) -> dict[str, int]:
        """Fast conflict probe along static shortest legs + congested neighborhoods of `failed_rid`."""
        hits: dict[str, int] = {}
        rob = self.robots_by_id[failed_rid]
        cur_n = rob.cur_node
        t_cur = max(self.t_commit, rob.faulted_until)

        def _scan_node_band(nid: str, lo: int, hi: int) -> None:
            sp = Interval(max(self.t_commit, lo), min(self.sim_horizon, max(lo + 1, hi)))
            if len(sp) <= 0:
                return
            for ow in res_table.node_conflicts(nid, sp, exclude_agv=failed_rid):
                if ow in active_set:
                    hits[ow] = hits.get(ow, 0) + 1

        _scan_node_band(cur_n, t_cur, t_cur + 24)
        for e in self.network.out_edges(cur_n):
            phys = physical_edge_id(e.src, e.dst)
            sp = Interval(t_cur, min(self.sim_horizon, t_cur + 18))
            if len(sp) > 0:
                for ow in res_table.edge_conflicts(phys[0], phys[1], sp, exclude_agv=failed_rid):
                    if ow in active_set:
                        hits[ow] = hits.get(ow, 0) + 2
                if self.network.has_edge(phys[1], phys[0]) and phys[1] != phys[0]:
                    for ow in res_table.edge_conflicts(phys[1], phys[0], sp, exclude_agv=failed_rid):
                        if ow in active_set:
                            hits[ow] = hits.get(ow, 0) + 2
            _scan_node_band(e.dst, t_cur, t_cur + 24)

        for t in tasks:
            _scan_node_band(t.pickup, t_cur, t_cur + 40)
            _scan_node_band(t.dropoff, t_cur, t_cur + 60)
        return hits

    def _space_time_route(
        self,
        start_node: str,
        goal_node: str,
        start_step: int,
        reservations: ReservationTable | RustReservationTable,
        agv_id: str,
        max_horizon: Optional[int] = None,
    ) -> list[tuple[str, int]] | None:
        use_slot_py = (
            self.cpr_cfg.enable_slot_wait_pressure
            and self._active_slot_table is not None
            and (self._force_python_slot_route or (not self.use_rust) or (not self.cpr_cfg.hybrid_rust_slot_projection))
        )
        if not use_slot_py:
            return super()._space_time_route(start_node, goal_node, start_step, reservations, agv_id, max_horizon)

        if start_node == goal_node:
            return [(start_node, start_step)]
        h_tbl = self._heuristic_table(goal_node)
        if goal_node not in h_tbl:
            return None

        horizon = (
            max_horizon
            if max_horizon is not None
            else min(self.sim_horizon - start_step, max(450, self.sim_horizon // 2))
        )
        limit_step = min(self.sim_horizon, start_step + max(1, horizon))
        slot_tbl = self._active_slot_table
        assert slot_tbl is not None
        g_w = int(self.cpr_cfg.wait_gamma_q3)
        c_w = int(self.cpr_cfg.wait_cap_q3)
        g_e = int(self.cpr_cfg.edge_choke_gamma_q3)
        c_e = int(self.cpr_cfg.edge_cap_q3)
        tw_q3 = self._traffic_weights_q3
        cut_s = self._traffic_cutoff_step

        def h(node: str) -> int:
            return int(math.floor(h_tbl.get(node, 1000.0) / self.dt))

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
                    w_pen = slot_tbl.wait_penalty_q3(
                        node, t + 1, is_goal=(node == goal_node), gamma_q3=g_w, cap_q3=c_w
                    )
                    ns = (node, t + 1)
                    n_sc = cur_sc + 8 + w_pen
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
                tw_pen = tw_q3.get((e.src, e.dst), 0) if (tw_q3 and (cut_s == 0 or t < cut_s)) else 0
                slot_ep = slot_tbl.edge_entry_penalty_q3(
                    e.src, e.dst, t, edge_cap=edge_cap, edge_gamma_q3=g_e, edge_cap_q3=c_e
                )
                n_sc = cur_sc + n_steps * 8 + tw_pen + slot_ep
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

    def _cpr_sort_key(
        self,
        rid: str,
        rob_tasks: list[TaskSnapshot],
        op_priority: dict[str, int],
        slot_tbl: Optional[SlotReservationPressureTable],
    ) -> tuple[int, float, str]:
        base_k = self._robot_plan_sort_key(rid, rob_tasks, op_priority)
        if (not self.cpr_cfg.enable_congestion_priority) or not rob_tasks:
            return (int(base_k[0]), float(base_k[1]), rid)
        max_age = max(max(0, self.t_commit - int(t.release_step)) for t in rob_tasks)
        local_p = slot_tbl.node_slot_count(self.robots_by_id[rid].cur_node, self.t_commit) if slot_tbl is not None else 0
        score = float(base_k[1]) - float(self.cpr_cfg.prio_age_tau) * float(max_age) - float(self.cpr_cfg.prio_congestion_kappa) * float(local_p)
        return (1, score, rid)

    def _sync_slot_projection(self, slot_tbl: Optional[SlotReservationPressureTable]) -> None:
        if (
            self.cpr_cfg.enable_slot_wait_pressure
            and self.use_rust
            and self.cpr_cfg.hybrid_rust_slot_projection
            and slot_tbl is not None
        ):
            dyn_w = slot_tbl.dynamic_corridor_q3_weights(
                self.t_commit,
                edge_gamma_q3=int(self.cpr_cfg.edge_choke_gamma_q3),
                edge_cap_q3=int(self.cpr_cfg.edge_cap_q3),
            )
            if dyn_w:
                merged = dict(self._caller_tw_q3)
                for ek, w in dyn_w.items():
                    merged[ek] = max(merged.get(ek, 0), w)
                cut = max(self._caller_cutoff, self.t_commit + 36) if self._caller_cutoff > 0 else (self.t_commit + 36)
                super().set_traffic_weights(merged, cutoff_step=cut)
                self._sync_rust_traffic()
                return
        if self._traffic_weights_q3 != self._caller_tw_q3 or self._traffic_cutoff_step != self._caller_cutoff:
            super().set_traffic_weights(self._caller_tw_q3 or None, cutoff_step=self._caller_cutoff)
            self._sync_rust_traffic()

    def _restore_robot_reservations(
        self,
        res_table: ReservationTable | RustReservationTable,
        agv_id: str,
        node_res: list[tuple[str, Interval]],
        edge_res: list[tuple[tuple[str, str], Interval]],
    ) -> None:
        res_table.release_all(agv_id)
        if isinstance(res_table, RustReservationTable):
            lines = [f"N\t{n}\t{sp.start}\t{sp.end}" for n, sp in node_res]
            lines.extend(f"E\t{e[0]}\t{e[1]}\t{sp.start}\t{sp.end}" for e, sp in edge_res)
            if lines:
                res_table.reserve_batch_script(agv_id, "\n".join(lines))
        else:
            for n, sp in node_res:
                res_table.reserve_node(n, sp, agv_id)
            for e, sp in edge_res:
                res_table.reserve_edge(e[0], e[1], sp, agv_id)

    def _has_repair_headroom(self) -> bool:
        if self._eval_deadline_s is None:
            return True
        rem_ms = (self._eval_deadline_s - _cvcr_now_s()) * 1000.0
        return rem_ms >= float(self.cpr_cfg.min_headroom_ms_for_repair)

    def _route_batch_cpr(
        self,
        rids_to_route: list[str],
        by_robot: dict[str, list[TaskSnapshot]],
        res_table: ReservationTable | RustReservationTable,
        fixed_paths: dict[str, list[tuple[str, int]]],
        op_priority: dict[str, int],
        op_res_ready_hint: Optional[dict[str, int]],
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
        need_slot = bool(self.cpr_cfg.enable_slot_wait_pressure or self.cpr_cfg.enable_congestion_priority)
        slot_tbl = (
            SlotReservationPressureTable(self.network, slot_delta=self.cpr_cfg.slot_delta, t_start=self.t_commit)
            if need_slot
            else None
        )
        if slot_tbl is not None:
            for frid, fpath in fixed_paths.items():
                slot_tbl.add_agv_path(frid, fpath)

        self._active_slot_table = slot_tbl
        ordered = sorted(rids_to_route, key=lambda r: self._cpr_sort_key(r, by_robot[r], op_priority, slot_tbl))

        out_paths: dict[str, list[tuple[str, int]]] = {}
        out_seqs: dict[str, list[str]] = {}
        out_unloads: dict[str, int] = {}
        out_nres: dict[str, list[tuple[str, Interval]]] = {}
        out_eres: dict[str, list[tuple[tuple[str, str], Interval]]] = {}

        active_routed: dict[str, tuple[list[tuple[str, int]], list[str], dict[str, int], list[tuple[str, Interval]], list[tuple[tuple[str, str], Interval]]]] = {}
        pinned_owners = self._pinned_owners_set(res_table)
        repairs_used = 0
        had_initial_fail = False
        need_res = bool(collect_res or self.cpr_cfg.enable_bounded_rollback)

        try:
            idx = 0
            while idx < len(ordered):
                rid = ordered[idx]
                self._sync_slot_projection(slot_tbl)
                path, seq, unloads, n_res, e_res, ok = self._route_single_robot(
                    rid,
                    by_robot[rid],
                    res_table,
                    op_priority,
                    op_res_ready_hint,
                    op_bay_overrides,
                    task_order_override=task_order_override,
                    collect_res=need_res,
                )
                if ok:
                    active_routed[rid] = (path, seq, unloads, n_res, e_res)
                    if slot_tbl is not None:
                        slot_tbl.add_agv_path(rid, path)
                    idx += 1
                    continue

                had_initial_fail = True
                # Optional exact Python slot-wait retry before rollback if configured
                if (
                    self.cpr_cfg.enable_slot_wait_pressure
                    and self.cpr_cfg.allow_exact_slot_retry_python
                    and self.use_rust
                    and self.cpr_cfg.hybrid_rust_slot_projection
                    and self._has_repair_headroom()
                ):
                    res_table.release_all(rid)
                    self._force_python_slot_route = True
                    try:
                        p_py, s_py, u_py, nr_py, er_py, ok_py = self._route_single_robot(
                            rid, by_robot[rid], res_table, op_priority, op_res_ready_hint,
                            op_bay_overrides, task_order_override=task_order_override, collect_res=need_res,
                        )
                    finally:
                        self._force_python_slot_route = False
                    if ok_py:
                        active_routed[rid] = (p_py, s_py, u_py, nr_py, er_py)
                        if slot_tbl is not None:
                            slot_tbl.add_agv_path(rid, p_py)
                        idx += 1
                        continue
                    res_table.release_all(rid)

                # Factor R: Bounded active-batch rollback
                if (
                    not self.cpr_cfg.enable_bounded_rollback
                    or repairs_used >= int(self.cpr_cfg.max_repairs_per_eval)
                    or not active_routed
                    or not self._has_repair_headroom()
                ):
                    out_paths[rid] = path
                    out_seqs[rid] = seq
                    out_unloads.update(unloads)
                    out_nres[rid] = n_res
                    out_eres[rid] = e_res
                    for crid, (cp, cseq, cun, cnr, cer) in active_routed.items():
                        out_paths[crid] = cp
                        out_seqs[crid] = cseq
                        out_unloads.update(cun)
                        out_nres[crid] = cnr
                        out_eres[crid] = cer
                    return out_paths, out_seqs, out_unloads, out_nres, out_eres, False

                repairs_used += 1
                self.cpr_repairs_attempted += 1
                res_table.release_all(rid)
                hits = self._probe_corridor_conflicts(rid, by_robot[rid], res_table, set(active_routed))
                blocker_j = select_active_blocker(rid, hits, active_routed, pinned_owners=pinned_owners)
                if blocker_j is None:
                    self.cpr_repairs_failed += 1
                    out_paths[rid] = path
                    out_seqs[rid] = seq
                    out_unloads.update(unloads)
                    return out_paths, out_seqs, out_unloads, out_nres, out_eres, False

                saved_j = active_routed[blocker_j]
                res_table.release_all(blocker_j)
                if slot_tbl is not None:
                    slot_tbl.remove_agv(blocker_j)
                self._sync_slot_projection(slot_tbl)

                p_i, s_i, u_i, nr_i, er_i, ok_i = self._route_single_robot(
                    rid, by_robot[rid], res_table, op_priority, op_res_ready_hint,
                    op_bay_overrides, task_order_override=task_order_override, collect_res=need_res,
                )
                if not ok_i:
                    self._restore_robot_reservations(res_table, blocker_j, saved_j[3], saved_j[4])
                    if slot_tbl is not None:
                        slot_tbl.add_agv_path(blocker_j, saved_j[0])
                    self.cpr_repairs_failed += 1
                    out_paths[rid] = p_i
                    out_seqs[rid] = s_i
                    out_unloads.update(u_i)
                    return out_paths, out_seqs, out_unloads, out_nres, out_eres, False

                if slot_tbl is not None:
                    slot_tbl.add_agv_path(rid, p_i)
                self._sync_slot_projection(slot_tbl)
                p_j, s_j, u_j, nr_j, er_j, ok_j = self._route_single_robot(
                    blocker_j, by_robot[blocker_j], res_table, op_priority, op_res_ready_hint,
                    op_bay_overrides, task_order_override=task_order_override, collect_res=need_res,
                )
                if not ok_j:
                    res_table.release_all(rid)
                    self._restore_robot_reservations(res_table, blocker_j, saved_j[3], saved_j[4])
                    if slot_tbl is not None:
                        slot_tbl.remove_agv(rid)
                        slot_tbl.add_agv_path(blocker_j, saved_j[0])
                    self.cpr_repairs_failed += 1
                    out_paths[rid] = p_i
                    out_seqs[rid] = s_i
                    out_unloads.update(u_i)
                    return out_paths, out_seqs, out_unloads, out_nres, out_eres, False

                active_routed[rid] = (p_i, s_i, u_i, nr_i, er_i)
                active_routed[blocker_j] = (p_j, s_j, u_j, nr_j, er_j)
                if slot_tbl is not None:
                    slot_tbl.add_agv_path(blocker_j, p_j)
                self.cpr_repairs_succeeded += 1
                idx += 1

            if had_initial_fail:
                self.cpr_candidates_rescued += 1
            for crid, (cp, cseq, cun, cnr, cer) in active_routed.items():
                out_paths[crid] = cp
                out_seqs[crid] = cseq
                out_unloads.update(cun)
                out_nres[crid] = cnr
                out_eres[crid] = cer
            return out_paths, out_seqs, out_unloads, out_nres, out_eres, True
        finally:
            self._active_slot_table = None
            super().set_traffic_weights(self._caller_tw_q3 or None, cutoff_step=self._caller_cutoff)
            self._sync_rust_traffic()

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
        rids = sorted(r.robot_id for r in self.snapshot.robots)
        by_robot: dict[str, list[TaskSnapshot]] = {rid: [] for rid in rids}
        for tid, t in self.tasks_by_id.items():
            if t.status.lower() != "completed":
                aid = assignment.get(tid)
                if aid in by_robot:
                    by_robot[aid].append(t)
        paths, seqs, unloads, _, _, all_ok = self._route_batch_cpr(
            rids, by_robot, res_table, {}, op_priority, op_res_ready_hint, op_bay_overrides, task_order_override, collect_res=False
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
        rids_to_route = sorted(r.robot_id for r in self.snapshot.robots)
        by_robot: dict[str, list[TaskSnapshot]] = {rid: [] for rid in rids_to_route}
        for tid, t in self.tasks_by_id.items():
            if t.status.lower() != "completed":
                aid = assignment.get(tid)
                if aid in by_robot:
                    by_robot[aid].append(t)
        paths, seqs, unloads, nres, eres, all_ok = self._route_batch_cpr(
            rids_to_route, by_robot, res_table, {}, op_priority, op_res_ready_hint, op_bay_overrides, task_order_override, collect_res=True
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
        # Temporarily evaluate with base signature matching; remember caller weights before comparing
        prio_key = tuple(sorted(op_priority.items()))
        hint_key = tuple(sorted(op_res_ready_hint.items())) if op_res_ready_hint is not None else None
        bay_key = tuple(sorted(op_bay_overrides.items())) if op_bay_overrides else None
        order_key = (
            tuple((k, tuple(v)) for k, v in sorted(task_order_override.items()))
            if task_order_override
            else None
        )
        ctx_invalid = (
            self.base_ctx_key is None
            or (self.base_ctx_key[1] != hint_key if self.seq_diff_b2 else self.base_ctx_key[:2] != (prio_key, hint_key))
            or self.base_traffic_sig != self._traffic_sig
            or not self.robot_cache
            or self._inc_res_table is None
        )
        if ctx_invalid:
            if self.base_ctx_key is not None:
                self.b2_fallbacks += 1
            return self._eval_b1(
                assignment, op_priority, op_res_ready_hint, op_bay_overrides, op_crew_overrides, task_order_override
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
        if self.seq_diff_b2:
            for r in self.snapshot.robots:
                rid = r.robot_id
                if rid in changed_set:
                    continue
                rc_seq = self.robot_cache.get(rid, {}).get("seq", [])
                if len(rc_seq) <= 1:
                    continue
                r_tasks = [self.tasks_by_id[tid] for tid in rc_seq if tid in self.tasks_by_id]
                order_rank = (
                    {tid: idx for idx, tid in enumerate(task_order_override[rid])}
                    if task_order_override and rid in task_order_override
                    else None
                )
                r_sorted = sorted(
                    r_tasks,
                    key=lambda t: (
                        0 if t.status.lower() in ("picked", "loading", "to_dropoff", "unloading") else 1,
                        order_rank.get(t.task_id, 100_000) if order_rank is not None else 0,
                        op_priority.get(t.kit_op_id or "", 0),
                        t.release_step,
                        t.task_id,
                    ),
                )
                if [t.task_id for t in r_sorted] != rc_seq:
                    changed_set.add(rid)
        elif order_key != base_order_key:
            new_orders = task_order_override or {}
            changed_set |= {
                rid
                for rid in (set(self.base_task_order) | set(new_orders))
                if self.base_task_order.get(rid) != new_orders.get(rid)
            }

        changed_rids = sorted(changed_set)
        gt = self.__class__._GLOBAL_TELEMETRY
        gt["b2_cand_evals"] = int(gt["b2_cand_evals"]) + 1
        gt["b2_changed_amr_ratio_sum"] = float(gt["b2_changed_amr_ratio_sum"]) + float(len(changed_rids)) / float(max(1, len(self.snapshot.robots)))
        gt["b2_changed_amr_ratio_cnt"] = int(gt["b2_changed_amr_ratio_cnt"]) + 1

        if not changed_rids:
            order_unload_ticks = dict(self.delivered_unloads)
            robot_paths: dict[str, list[tuple[str, int]]] = {}
            robot_order_seq: dict[str, list[str]] = {}
            for r in self.snapshot.robots:
                rc = self.robot_cache.get(r.robot_id)
                if rc is not None:
                    robot_paths[r.robot_id] = rc["path"]
                    robot_order_seq[r.robot_id] = rc["seq"]
                    order_unload_ticks.update(rc["unloads"])
            self.b2_inc_hits += 1
            gt["b2_inc_hits"] = int(gt["b2_inc_hits"]) + 1
            gt["b2_zero_route_hits"] = int(gt["b2_zero_route_hits"]) + 1
            op_s, op_e, op_rr, obj, starv = self._compute_objective(
                assignment, order_unload_ticks, op_priority, True, op_bay_overrides, op_crew_overrides
            )
            return EvalResult(robot_paths, robot_order_seq, order_unload_ticks, op_s, op_e, op_rr, obj, True, starv, 0.0, "B2")

        if len(changed_rids) >= len(self.snapshot.robots):
            self.b2_fallbacks += 1
            gt["b2_fallbacks"] = int(gt["b2_fallbacks"]) + 1
            return self._eval_b1(
                assignment, op_priority, op_res_ready_hint, op_bay_overrides, op_crew_overrides, task_order_override
            )

        res_table = self._inc_res_table
        assert res_table is not None
        for rid in changed_rids:
            res_table.release_all(rid)

        fixed_paths: dict[str, list[tuple[str, int]]] = {}
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
                fixed_paths[rid] = rc["path"]

        by_robot: dict[str, list[TaskSnapshot]] = {rid: [] for rid in changed_rids}
        for tid, t in self.tasks_by_id.items():
            if t.status.lower() != "completed":
                aid = assignment.get(tid)
                if aid in by_robot:
                    by_robot[aid].append(t)

        c_paths, c_seqs, c_unloads, _, _, all_ok = self._route_batch_cpr(
            changed_rids, by_robot, res_table, fixed_paths, op_priority, op_res_ready_hint,
            op_bay_overrides, task_order_override, collect_res=False,
        )
        robot_paths.update(c_paths)
        robot_order_seq.update(c_seqs)
        order_unload_ticks.update(c_unloads)
        self._restore_inc_table(changed_rids)

        if not all_ok:
            self.b2_fallbacks += 1
            gt["b2_fallbacks"] = int(gt["b2_fallbacks"]) + 1
        else:
            self.b2_inc_hits += 1
            gt["b2_inc_hits"] = int(gt["b2_inc_hits"]) + 1

        op_s, op_e, op_rr, obj, starv = self._compute_objective(
            assignment, order_unload_ticks, op_priority, all_ok, op_bay_overrides, op_crew_overrides
        )
        return EvalResult(robot_paths, robot_order_seq, order_unload_ticks, op_s, op_e, op_rr, obj, all_ok, starv, 0.0, "B2")
