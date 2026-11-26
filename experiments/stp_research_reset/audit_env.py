"""Isolated ExecutionAuditSim wrapper over frozen MultiRobotUnicycleSim.

Zero edits to `metric_world.py`, `amr_sched/`, or `crates/`.
Supports:
  - Fine-grained planning-vs-execution mismatch logging (Task 2).
  - Pluggable upper-level planning modes (`U0..U4`) with shared fixed `R1_Robust` executor (Task 3).
  - Bitwise-faithful `clone_snapshot()` preserving audit state for matched counterfactual rollouts.
"""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass, field
import math
import random
import time
from typing import Any, Callable, Optional

import numpy as np

from amr_sched.cvcr.evaluator import physical_edge_id
from amr_sched.planners.base import PlannedPath
from amr_sched.reservation import Interval
from amr_sched.rust_reservation import RustReservationTable
from experiments.res_guided_avoidance.gate1_headroom_audit import m2_res_margin_selector
from experiments.res_guided_avoidance.geometry_and_shield import (
    DT_MICRO,
    MAX_DV_DT,
    MAX_DW_DT,
    MAX_OMEGA_RADPS,
    MAX_V_MPS,
    CircleState,
    clip_val,
    obstacle_surface_clearance,
    pair_robot_clearance,
    wrap_pi,
)
from experiments.res_guided_avoidance.metric_world import (
    MetricMap,
    MultiRobotUnicycleSim,
    SUBSTEPS_DEFAULT,
    expand_planned_to_step_positions,
)


@dataclass
class LegTelemetry:
    """Tracks one planned-to-executed robot navigation leg."""

    leg_id: int
    robot_id: str
    phase: str
    start_node: str
    goal_node: str
    is_replan: bool
    is_eta_drift_replan: bool
    plan_t_s: float
    planned_eta_s: float
    planned_dur_s: float
    planned_len_m: float
    planned_waits: int
    sharp_turns: int
    init_hdg_err_rad: float
    plan_failed: bool
    actual_dist_m: float = 0.0
    row_pause_steps: int = 0
    shield_interv_steps: int = 0
    slow_steps: int = 0
    wait_steps: int = 0
    end_t_s: Optional[float] = None
    completed: bool = False
    truncated: bool = False
    waypoint_nodes: list[str] = field(default_factory=list)
    waypoint_plan_s: list[float] = field(default_factory=list)
    waypoint_actual_s: list[Optional[float]] = field(default_factory=list)
    caused_late_spillover: int = 0


def _route_len_and_turns(mmap: MetricMap, nodes: list[str], init_theta: float) -> tuple[float, int, float]:
    if len(nodes) < 2:
        return 0.0, 0, 0.0
    net = mmap.network
    tot_len = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        if net.has_edge(u, v):
            tot_len += net.edge(u, v).length
        else:
            nu, nv = net.node(u), net.node(v)
            tot_len += math.hypot(nv.x - nu.x, nv.y - nu.y)
    n0, n1 = net.node(nodes[0]), net.node(nodes[1])
    dir0 = math.atan2(n1.y - n0.y, n1.x - n0.x)
    init_err = abs(wrap_pi(dir0 - init_theta))
    turns = 0
    for a, b, c in zip(nodes[:-2], nodes[1:-1], nodes[2:]):
        na, nb, nc = net.node(a), net.node(b), net.node(c)
        dx1, dy1 = nb.x - na.x, nb.y - na.y
        dx2, dy2 = nc.x - nb.x, nc.y - nb.y
        l1, l2 = math.hypot(dx1, dy1), math.hypot(dx2, dy2)
        if l1 > 1e-6 and l2 > 1e-6:
            if ((dx1 * dx2 + dy1 * dy2) / (l1 * l2)) < 0.707:
                turns += 1
    return float(tot_len), turns, float(init_err)


def _reserve_exact_span(
    res_table: RustReservationTable,
    mmap: MetricMap,
    agv_id: str,
    start_node: str,
    start_tick: int,
    planned: PlannedPath,
    *,
    buffer_ticks: int = 0,
    sub_tick_div: int = 1,
    park_until: int | None = None,
) -> list[tuple[str, int]]:
    """Reserve accurate multi-step node AND edge occupancy `[dep_t, arr_t + buf)` without `arr_i - 1` bug."""
    pts: list[tuple[str, int]] = [(start_node, int(start_tick))]
    cur_n = start_node
    cur_t = int(start_tick)
    net = mmap.network
    buf = max(0, int(buffer_ticks))
    div = max(1, int(sub_tick_div))
    hold_ticks = max(1, (div + 1) // 2)
    for nxt_n, arr_t in zip(planned.nodes, planned.arrival_steps):
        arr_i = int(arr_t)
        if nxt_n == cur_n:
            for tk in range(cur_t + 1, arr_i + 1):
                res_table.reserve_node(cur_n, Interval(tk, tk + 1), agv_id)
                pts.append((cur_n, tk))
            cur_t = arr_i
            continue
        dep_t = cur_t
        dep_hold = max(dep_t + 1, min(arr_i, dep_t + hold_ticks))
        res_table.reserve_node(cur_n, Interval(dep_t, dep_hold), agv_id)
        e_end = max(dep_t + 1, arr_i + buf)
        res_table.reserve_edge(cur_n, nxt_n, Interval(dep_t, e_end), agv_id)
        # Keep intermediate slot field sample points across the multi-step edge
        for tk in range(dep_t + 1, arr_i):
            pts.append((cur_n if (tk - dep_t) * 2 < (arr_i - dep_t) else nxt_n, tk))
        res_table.reserve_node(nxt_n, Interval(arr_i, arr_i + hold_ticks + buf), agv_id)
        pts.append((nxt_n, arr_i))
        cur_n = nxt_n
        cur_t = arr_i
    if park_until is not None and park_until > cur_t + hold_ticks + buf:
        res_table.reserve_node(cur_n, Interval(cur_t + hold_ticks + buf, park_until), agv_id)
    return pts


class ExecutionAuditSim(MultiRobotUnicycleSim):
    """Instrumented simulation & pluggable upper-planner testbed preserving R1_Robust invariants."""

    def __init__(
        self,
        metric_map: MetricMap,
        *,
        upper_mode: str = "U0_Orig_CPSA",
        sub_tick_div: int = 1,
        fixed_buffer_s: float = 1.0,
        custom_plan_fn: Optional[Callable[["ExecutionAuditSim", str, str, int], tuple[PlannedPath | None, int]]] = None,
        record_telemetry: bool = True,
        **kwargs: Any,
    ) -> None:
        self.upper_mode = str(upper_mode)
        self._sub_tick_div: int = 1 if self.upper_mode in ("U0_Orig_CPSA", "U1a_EdgeSpan_Fix") else max(1, int(sub_tick_div))
        self.fixed_buffer_s = float(fixed_buffer_s)
        self.custom_plan_fn = custom_plan_fn
        self.record_telemetry = bool(record_telemetry)

        # Telemetry containers initialized prior to super().__init__ -> reset()
        self._leg_seq: int = 0
        self.closed_legs: list[LegTelemetry] = []
        self.active_legs: dict[str, LegTelemetry] = {}
        self.eta_drift_replans: int = 0
        self.init_plan_failures: int = 0
        self.row_pause_steps_total: int = 0
        self.row_gate_events: int = 0
        self.shield_wall_events: int = 0
        self.shield_peer_events: int = 0
        self.shield_intr_events: int = 0
        self.shield_other_events: int = 0
        self.spillover_peer_waits: int = 0
        self.spillover_peer_replans: int = 0
        self._recent_late_nodes: dict[str, tuple[str, float]] = {}  # node_id -> (late_rid, expires_t_s)
        self._step_row_blocked: set[str] = set()
        self._next_replan_reason_drift: set[str] = set()
        self._plan_wall_ms_samples: list[float] = []
        self._route_times_sec: dict[str, list[float]] = {}
        self._active_buf_s: dict[str, float] = {}
        self._active_leg_plans: dict[str, tuple[str, int, PlannedPath, int]] = {}
        self._sub_res_table: RustReservationTable = RustReservationTable.fast()
        self._sub_phys_locks: dict[str, tuple[str, int, int]] = {}
        super().__init__(metric_map, **kwargs)

    @property
    def plan_dt_s(self) -> float:
        return self.mmap.macro_step_s / float(self._sub_tick_div)

    @property
    def active_res_table(self) -> RustReservationTable:
        return self._sub_res_table if self._sub_tick_div > 1 else self.res_table

    def reset(self) -> None:
        self._leg_seq = 0
        self.closed_legs = []
        self.active_legs = {}
        self.eta_drift_replans = 0
        self.init_plan_failures = 0
        self.row_pause_steps_total = 0
        self.row_gate_events = 0
        self.shield_wall_events = 0
        self.shield_peer_events = 0
        self.shield_intr_events = 0
        self.shield_other_events = 0
        self.spillover_peer_waits = 0
        self.spillover_peer_replans = 0
        self._recent_late_nodes = {}
        self._step_row_blocked = set()
        self._next_replan_reason_drift = set()
        self._plan_wall_ms_samples = []
        self._route_times_sec = {}
        self._active_buf_s = {}
        self._active_leg_plans = {}
        self._sub_res_table = RustReservationTable.fast()
        self._sub_phys_locks = {}
        super().reset()

    def _plan_robot(self, rid: str, goal_node: str, t_tick: int) -> bool:
        t0 = time.perf_counter()
        rb = self.robots[rid]
        start_node = self._nearest_node(rb.x, rb.y)
        rb.cur_node = start_node

        is_replan = rid in self.active_legs
        is_drift = rid in self._next_replan_reason_drift
        self._next_replan_reason_drift.discard(rid)

        if self.upper_mode == "U0_Orig_CPSA":
            ok = super()._plan_robot(rid, goal_node, t_tick)
            self._plan_wall_ms_samples.append((time.perf_counter() - t0) * 1000.0)
            self._route_times_sec[rid] = [float(tk) * self.mmap.macro_step_s for tk in rb.route_times]
            self._active_buf_s[rid] = 0.0
            if not ok:
                self.init_plan_failures += 1
            elif self.record_telemetry and rb.phase != "idle":
                self._open_leg_record(rid, goal_node, t_tick, is_replan=is_replan, is_drift=is_drift, waited=0)
                if is_drift:
                    self.eta_drift_replans += 1
            return ok

        # Non-U0 upper planning modes (U1a, U1b, U2, U3, custom U4)
        div = self._sub_tick_div
        plan_rt = self.active_res_table
        if div > 1 and t_tick > self.macro_tick and rb.service_rem_s > 0.0:
            t_plan_tick = self.macro_tick * div + int(math.ceil(rb.service_rem_s / self.plan_dt_s - 1e-9))
        else:
            t_plan_tick = int(t_tick) * div

        plan_rt.release_all(rid)
        self.field.remove_agv(rid)
        self._active_pts.pop(rid, None)
        self._active_leg_plans.pop(rid, None)
        lock_len = 2 * div
        plan_rt.reserve_node(start_node, Interval(t_plan_tick, t_plan_tick + lock_len), rid)
        if div > 1:
            self._sub_phys_locks[rid] = (start_node, t_plan_tick, t_plan_tick + lock_len)
        else:
            self._phys_locks[rid] = (start_node, t_plan_tick, t_plan_tick + lock_len)

        buf_ticks = 0
        pl: PlannedPath | None = None
        if self.custom_plan_fn is not None:
            pl, buf_ticks = self.custom_plan_fn(self, rid, goal_node, t_plan_tick)
            self.replans += 1
        elif div == 1:
            pl, _ = self.planner.plan_leg(
                start_node,
                goal_node,
                t_plan_tick,
                plan_rt,
                self.field,
                rid,
                min(self.plan_horizon_ticks, t_plan_tick + 90),
                goal_park_steps=1,
            )
            self.replans += 1

        if (pl is None or not pl.nodes) and div == 1:
            empty_rt = RustReservationTable.fast()
            pl, _ = self.planner.plan_leg(
                start_node,
                goal_node,
                t_plan_tick,
                empty_rt,
                self.field,
                rid,
                max(self.plan_horizon_ticks, t_plan_tick + 140),
                goal_park_steps=1,
            )
            buf_ticks = 0
        elif (pl is None or not pl.nodes) and div > 1:
            empty_rt = RustReservationTable.fast()
            raw_pl, _ = self.planner.plan_leg(
                start_node,
                goal_node,
                int(t_tick),
                empty_rt,
                self.planner.create_field(),
                rid,
                max(self.plan_horizon_ticks, int(t_tick) + 140),
                goal_park_steps=1,
            )
            if raw_pl is not None and raw_pl.nodes:
                pl = PlannedPath(
                    nodes=list(raw_pl.nodes),
                    arrival_steps=[t_plan_tick + max(1, (int(s) - int(t_tick)) * div) for s in raw_pl.arrival_steps],
                    wait_steps=int(raw_pl.wait_steps) * div,
                )
            buf_ticks = 0

        self._plan_wall_ms_samples.append((time.perf_counter() - t0) * 1000.0)

        if pl is None or not pl.nodes:
            plan_rt.reserve_node(start_node, Interval(t_plan_tick, t_plan_tick + 3 * div), rid)
            rb.route_nodes = [start_node]
            rb.route_times = [t_tick]
            rb.route_idx = 0
            rb.last_plan_fail_tick = t_tick
            rb.prev_rem_dist_m = 0.0
            self._route_times_sec[rid] = [float(t_plan_tick) * self.plan_dt_s]
            self._active_buf_s[rid] = 0.0
            self.init_plan_failures += 1
            return False

        pts = _reserve_exact_span(
            plan_rt,
            self.mmap,
            rid,
            start_node,
            t_plan_tick,
            pl,
            buffer_ticks=buf_ticks,
            sub_tick_div=div,
            park_until=None,
        )
        self.field.add_agv_path(rid, pts)
        self._active_pts[rid] = list(pts)
        self._active_leg_plans[rid] = (start_node, t_plan_tick, pl, int(buf_ticks))
        self._active_buf_s[rid] = float(max(0, int(buf_ticks))) * self.plan_dt_s

        w_nodes: list[str] = [start_node]
        w_times_sub: list[int] = [t_plan_tick]
        for nd_id, step_t in zip(pl.nodes, pl.arrival_steps):
            if w_nodes[-1] != nd_id:
                w_nodes.append(nd_id)
                w_times_sub.append(int(step_t))
        plan_s_list = [float(tk) * self.plan_dt_s for tk in w_times_sub]
        rb.route_nodes = w_nodes
        rb.route_times = [int(math.ceil(s / self.mmap.macro_step_s - 1e-9)) for s in plan_s_list]
        rb.route_idx = 0
        rb.prev_rem_dist_m = self.remaining_route_dist_m(rb)
        self._route_times_sec[rid] = plan_s_list
        if self.record_telemetry and rb.phase != "idle":
            self._open_leg_record(rid, goal_node, t_tick, is_replan=is_replan, is_drift=is_drift, waited=pl.wait_steps)
            if is_drift:
                self.eta_drift_replans += 1
        return True

    def _refresh_res_views(self) -> None:
        if self.upper_mode == "U0_Orig_CPSA":
            super()._refresh_res_views()
            return
        view: dict[str, list[tuple[str, float, float, int]]] = {}
        macro = self.mmap.macro_step_s
        for rid in self.robot_ids:
            rb = self.robots[rid]
            buf_s = self._active_buf_s.get(rid, 0.0)
            times_s = self._route_times_sec.get(rid)
            segs: list[tuple[str, float, float, int]] = []
            prio = 2 if rb.phase == "to_dropoff" else (1 if rb.phase == "to_pickup" else 0)
            for idx, nd_id in enumerate(rb.route_nodes):
                if times_s and idx < len(times_s):
                    t_en_s = times_s[idx]
                    t_ex_s = (times_s[idx + 1] if (idx + 1) < len(times_s) else (t_en_s + macro)) + buf_s
                else:
                    t_en_tk = rb.route_times[idx] if idx < len(rb.route_times) else self.macro_tick
                    t_ex_tk = rb.route_times[idx + 1] if (idx + 1) < len(rb.route_times) else (t_en_tk + 2)
                    t_en_s = float(t_en_tk) * macro
                    t_ex_s = float(t_ex_tk) * macro + buf_s
                segs.append((nd_id, t_en_s / macro, t_ex_s / macro, prio))
            view[rid] = segs
        self._current_res_view = view  # type: ignore[assignment]
        self._res_history.append((self.t_s, view))  # type: ignore[arg-type]
        if self.reservation_lag_s <= 1e-6:
            self._lagged_res_view = view  # type: ignore[assignment]
        else:
            target_t = self.t_s - self.reservation_lag_s
            chosen = self._res_history[0][1]
            for ts, vw in self._res_history:
                if ts <= target_t:
                    chosen = vw
                else:
                    break
            self._lagged_res_view = chosen

    def _open_leg_record(
        self,
        rid: str,
        goal_node: str,
        t_tick: int,
        *,
        is_replan: bool,
        is_drift: bool,
        waited: int,
    ) -> None:
        # Close any previous unfinished leg on this robot first as replan-truncated
        if rid in self.active_legs:
            old = self.active_legs.pop(rid)
            old.end_t_s = self.t_s
            old.truncated = True
            self.closed_legs.append(old)

        rb = self.robots[rid]
        p_len, n_turns, init_err = _route_len_and_turns(self.mmap, rb.route_nodes, rb.theta)
        plan_s_list = list(self._route_times_sec.get(rid) or [float(tk) * self.mmap.macro_step_s for tk in rb.route_times])
        t_now = plan_s_list[0] if plan_s_list else float(t_tick) * self.mmap.macro_step_s
        eta_s = plan_s_list[-1] if plan_s_list else t_now
        self._leg_seq += 1
        wp_act: list[Optional[float]] = [t_now] + [None] * max(0, len(rb.route_nodes) - 1)
        self.active_legs[rid] = LegTelemetry(
            leg_id=self._leg_seq,
            robot_id=rid,
            phase=rb.phase,
            start_node=rb.route_nodes[0] if rb.route_nodes else rb.cur_node,
            goal_node=goal_node,
            is_replan=is_replan,
            is_eta_drift_replan=is_drift,
            plan_t_s=t_now,
            planned_eta_s=eta_s,
            planned_dur_s=max(0.0, eta_s - t_now),
            planned_len_m=p_len,
            planned_waits=waited,
            sharp_turns=n_turns,
            init_hdg_err_rad=init_err,
            plan_failed=False,
            waypoint_nodes=list(rb.route_nodes),
            waypoint_plan_s=plan_s_list,
            waypoint_actual_s=wp_act,
        )

    def _macro_coordinate(self) -> None:
        t_tick = self.macro_tick
        if self.upper_mode != "U0_Orig_CPSA" and self._sub_tick_div > 1:
            t_sub_phys = t_tick * self._sub_tick_div
            lock_sub = 2 * self._sub_tick_div
            for rid_p in self.robot_ids:
                rb_p = self.robots[rid_p]
                phys_nd = self._nearest_node(rb_p.x, rb_p.y)
                self._sub_res_table.reserve_node(phys_nd, Interval(t_sub_phys, t_sub_phys + lock_sub), rid_p)
                self._sub_phys_locks[rid_p] = (phys_nd, t_sub_phys, t_sub_phys + lock_sub)

        if self.record_telemetry:
            # Pre-check which robots are about to finish leg or suffer eta_drift before calling super
            for rid in self.robot_ids:
                rb = self.robots[rid]
                if rb.cur_task is not None and rb.service_rem_s <= 0.0:
                    tgt_id = rb.cur_task.pickup if rb.phase == "to_pickup" else rb.cur_task.dropoff
                    tgt = self.mmap.network.node(tgt_id)
                    if math.hypot(rb.x - tgt.x, rb.y - tgt.y) <= 0.85:
                        if rid in self.active_legs:
                            leg = self.active_legs.pop(rid)
                            if leg.waypoint_actual_s:
                                leg.waypoint_actual_s[-1] = self.t_s
                            leg.end_t_s = self.t_s
                            leg.completed = True
                            self.closed_legs.append(leg)

            # Detect upcoming eta_drift triggers (`t_tick > exp_t + 3`) before super()._macro_coordinate() fires _plan_robot
            for rid in self.robot_ids:
                rb = self.robots[rid]
                if rb.phase == "idle" or rb.cur_task is None or rb.service_rem_s > 0.0:
                    continue
                if len(rb.route_nodes) > 1 and rb.route_idx + 1 < len(rb.route_times):
                    exp_t = rb.route_times[rb.route_idx + 1]
                    if t_tick > exp_t + 3:
                        self._next_replan_reason_drift.add(rid)
                        nxt_nd = rb.route_nodes[rb.route_idx + 1]
                        for ck_nd in (rb.cur_node, nxt_nd):
                            rec = self._recent_late_nodes.get(ck_nd)
                            if rec is not None and rec[0] != rid and rec[1] >= self.t_s:
                                self.spillover_peer_replans += 1
                                break

        super()._macro_coordinate()

    def apply_right_of_way_gate(self, rid: str, v_req: float, w_req: float) -> tuple[float, float]:
        v_out, w_out = super().apply_right_of_way_gate(rid, v_req, w_req)
        if self.record_telemetry and v_out < v_req - 1e-6:
            self.row_pause_steps_total += 1
            self.row_gate_events += 1
            self._step_row_blocked.add(rid)
            if rid in self.active_legs:
                self.active_legs[rid].row_pause_steps += 1
            rb = self.robots[rid]
            if rb.route_nodes:
                for nid in rb.route_nodes[rb.route_idx : rb.route_idx + 3]:
                    rec = self._recent_late_nodes.get(nid)
                    if rec is not None and rec[0] != rid and rec[1] >= self.t_s:
                        self.spillover_peer_waits += 1
                        break
        return v_out, w_out

    def step(
        self,
        controller_fn: Callable[[MultiRobotUnicycleSim, str], tuple[float, float]],
        *,
        controller_name: str = "controller",
    ) -> bool:
        prev_sh = self.shield_interventions
        prev_pos = {rid: (rb.x, rb.y) for rid, rb in self.robots.items()} if self.record_telemetry else {}
        if self.record_telemetry:
            self._step_row_blocked.clear()

        done = super().step(controller_fn, controller_name=controller_name)
        if not self.record_telemetry:
            return done

        delta_sh = self.shield_interventions - prev_sh
        if delta_sh > 0:
            hazards: list[tuple[float, str, str]] = []
            for rid in self.robot_ids:
                rb = self.robots[rid]
                if rb.phase == "idle":
                    continue
                wc = self.mmap.wall_clearance(rb.x, rb.y, rb.theta, self.fp)
                min_p = 9.0
                for oid in self.robot_ids:
                    if oid != rid:
                        o = self.robots[oid]
                        d_p = pair_robot_clearance(rb.x, rb.y, rb.theta, self.fp, o.x, o.y, o.theta, self.fp)
                        if d_p < min_p:
                            min_p = d_p
                min_i = 9.0
                for intr in self.intruders:
                    di = obstacle_surface_clearance(
                        rb.x, rb.y, rb.theta, (CircleState(intr.x, intr.y, intr.vx, intr.vy, intr.radius),), self.fp
                    )
                    if di < min_i:
                        min_i = di
                if wc < 0.30 and wc <= min(min_p, min_i):
                    hazards.append((wc, "wall", rid))
                elif min_i < 0.65 and min_i <= min_p:
                    hazards.append((min_i, "intr", rid))
                elif min_p < 0.90:
                    hazards.append((min_p, "peer", rid))
                else:
                    hazards.append((min(wc, min_p, min_i), "other", rid))
            hazards.sort(key=lambda item: item[0])
            for _, kind, rid in hazards[:delta_sh]:
                if kind == "wall":
                    self.shield_wall_events += 1
                elif kind == "intr":
                    self.shield_intr_events += 1
                elif kind == "peer":
                    self.shield_peer_events += 1
                else:
                    self.shield_other_events += 1
                if rid in self.active_legs:
                    self.active_legs[rid].shield_interv_steps += 1
            if len(hazards) < delta_sh:
                self.shield_other_events += delta_sh - len(hazards)

        for rid in self.robot_ids:
            rb = self.robots[rid]
            if rid in self.active_legs:
                leg = self.active_legs[rid]
                px0, py0 = prev_pos[rid]
                leg.actual_dist_m += math.hypot(rb.x - px0, rb.y - py0)
                if rb.v < 0.50:
                    leg.slow_steps += 1
                if rb.v < 0.15:
                    leg.wait_steps += 1
                for k, nd_id in enumerate(leg.waypoint_nodes):
                    if k < len(leg.waypoint_actual_s) and leg.waypoint_actual_s[k] is None:
                        if k <= rb.route_idx:
                            leg.waypoint_actual_s[k] = self.t_s
                            plan_t = leg.waypoint_plan_s[k]
                            if (self.t_s - plan_t) > max(1.5, self.mmap.macro_step_s):
                                self._recent_late_nodes[nd_id] = (rid, self.t_s + 8.0)
                                leg.caused_late_spillover += 1
                if self.t_s > leg.planned_eta_s + 2.0 and rb.route_nodes:
                    for nd_id in rb.route_nodes[rb.route_idx : rb.route_idx + 2]:
                        self._recent_late_nodes[nd_id] = (rid, self.t_s + 6.0)

        if done and self.active_legs:
            for rid, leg in list(self.active_legs.items()):
                leg.end_t_s = self.t_s
                leg.truncated = True
                self.closed_legs.append(leg)
            self.active_legs.clear()

        return done

    def clone_snapshot(self) -> "ExecutionAuditSim":
        """Clone preserving subclass planner hooks & exact reservations without touching base class."""
        base_cl = super().clone_snapshot()
        cl = ExecutionAuditSim.__new__(ExecutionAuditSim)
        cl.__dict__.update(base_cl.__dict__)
        cl.upper_mode = self.upper_mode
        cl._sub_tick_div = self._sub_tick_div
        cl.fixed_buffer_s = self.fixed_buffer_s
        cl.custom_plan_fn = self.custom_plan_fn
        cl.record_telemetry = False  # fast branch rollouts skip heavy leg logs
        cl._leg_seq = self._leg_seq
        cl.closed_legs = []
        cl.active_legs = {}
        cl.eta_drift_replans = self.eta_drift_replans
        cl.init_plan_failures = self.init_plan_failures
        cl.row_pause_steps_total = self.row_pause_steps_total
        cl.row_gate_events = self.row_gate_events
        cl.shield_wall_events = self.shield_wall_events
        cl.shield_peer_events = self.shield_peer_events
        cl.shield_intr_events = self.shield_intr_events
        cl.shield_other_events = self.shield_other_events
        cl.spillover_peer_waits = self.spillover_peer_waits
        cl.spillover_peer_replans = self.spillover_peer_replans
        cl._recent_late_nodes = dict(self._recent_late_nodes)
        cl._step_row_blocked = set()
        cl._next_replan_reason_drift = set(self._next_replan_reason_drift)
        cl._plan_wall_ms_samples = []
        cl._route_times_sec = copy.deepcopy(self._route_times_sec)
        cl._active_buf_s = dict(self._active_buf_s)
        cl._active_leg_plans = {
            rid: (sn, st, PlannedPath(list(pl.nodes), list(pl.arrival_steps), pl.wait_steps), bt)
            for rid, (sn, st, pl, bt) in self._active_leg_plans.items()
        }
        cl._sub_phys_locks = dict(self._sub_phys_locks)
        cl._sub_res_table = RustReservationTable.fast()

        if self.upper_mode != "U0_Orig_CPSA":
            div = cl._sub_tick_div
            target_rt = cl._sub_res_table if div > 1 else cl.res_table
            if div == 1:
                cl.res_table = RustReservationTable.fast()
                target_rt = cl.res_table
            cl.field = cl.planner.create_field()
            for rid_a, (sn, st, pl, bt) in cl._active_leg_plans.items():
                pts = _reserve_exact_span(
                    target_rt,
                    cl.mmap,
                    rid_a,
                    sn,
                    st,
                    pl,
                    buffer_ticks=bt,
                    sub_tick_div=div,
                    park_until=None,
                )
                cl.field.add_agv_path(rid_a, pts)
            locks = cl._sub_phys_locks if div > 1 else cl._phys_locks
            for rid_l, (nd_l, t0_l, t1_l) in locks.items():
                target_rt.reserve_node(nd_l, Interval(t0_l, t1_l), rid_l)
        return cl
