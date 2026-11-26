"""Track A Attribution Closure (A1-A2).

A1: Disentangles `U1a_EdgeSpan_Fix`'s pure `_reserve_exact_span` fix from its
`+120` planning horizon change via a 2x2 factorial design over all 18 DEV episodes
(`6 scenarios x 3 seeds 6001-6003`):
  1. `U0_H90_ArrMinus1`        (`MetricWorldSim` baseline: `+90` + `arr_i - 1`)
  2. `U0_Pure_H90_EdgeSpanFix` (`+90` + retry `+2/+92` + `_reserve_exact_span`)
  3. `U0_H120_ArrMinus1`       (`+120` + retry `+2/+122` + `arr_i - 1`)
  4. `U1a_H120_EdgeSpanFix`    (`u1a_plan_fn`: `+120` + retry `+2/+122` + `_reserve_exact_span`)

A2: Evaluates `eta_drift` replanning trigger calibration (`t_tick > exp_t + thresh`
for `thresh in {3, 6, 12, None}`) across all 4 clean DEV scenarios (`12` episodes
per variant x threshold) for:
  - `U0_Orig_CPSA` (`thresh=3` reproduces `ExecutionAuditSim` bit-for-bit)
  - `U0_Pure_H90_EdgeSpanFix`
  - `U1b_KinoTime_CPSA`

Outputs `out/foundation_audit/track_a_attribution.{json,md}`.

Usage:
    python -B -m experiments.foundation_audit.run_track_a_attribution
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import time
from typing import Any, Callable

import numpy as np
from scipy.stats import wilcoxon

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.rust_reservation import Interval, RustReservationTable
from amr_sched.planners.spacetime_astar import PlannedPath
from experiments.res_guided_avoidance.gate1_headroom_audit import m2_res_margin_selector
from experiments.res_guided_avoidance.metric_world import (
    ContTask,
    MultiRobotEpisodeResult,
    build_grid_metric_map,
    build_occluded_junction_map,
    build_workshop_metric_map,
)
from experiments.stp_research_reset.audit_env import ExecutionAuditSim
from experiments.stp_research_reset.upper_planners import make_u1b_plan_fn, u1a_plan_fn
from experiments.track_r_cpsa.cpsa_pure_runner import expand_planned_to_step_positions

OUT_DIR = ROOT / "out" / "foundation_audit"
DEV_SEEDS = (6001, 6002, 6003)


def u0_pure_h90_plan_fn(
    sim: ExecutionAuditSim, rid: str, goal_node: str, t_plan_tick: int
) -> tuple[PlannedPath | None, int]:
    """`U0` horizon (`+90` + retry `+2/+92`) paired strictly with `_reserve_exact_span`."""
    pl, _ = sim.planner.plan_leg(
        sim.robots[rid].cur_node,
        goal_node,
        t_plan_tick,
        sim.active_res_table,
        sim.field,
        rid,
        min(sim.plan_horizon_ticks, t_plan_tick + 90),
        goal_park_steps=1,
    )
    if pl is None or not pl.nodes:
        pl, _ = sim.planner.plan_leg(
            sim.robots[rid].cur_node,
            goal_node,
            t_plan_tick + 2,
            sim.active_res_table,
            sim.field,
            rid,
            min(sim.plan_horizon_ticks, t_plan_tick + 92),
            goal_park_steps=1,
        )
    return pl, 0


class ConfigurableDriftAuditSim(ExecutionAuditSim):
    """ExecutionAuditSim variant supporting `U0_H120_ArrMinus1` and parameterized `eta_drift` threshold."""

    def __init__(
        self,
        *args: Any,
        drift_thresh_ticks: int | None = 3,
        force_unified_macro: bool = False,
        **kwargs: Any,
    ) -> None:
        self.drift_thresh_ticks = drift_thresh_ticks
        self.force_unified_macro = bool(force_unified_macro)
        super().__init__(*args, **kwargs)

    def _plan_robot(self, rid: str, goal_node: str, t_tick: int) -> bool:
        if self.upper_mode != "U0_H120_ArrMinus1":
            return super()._plan_robot(rid, goal_node, t_tick)

        t0 = time.perf_counter()
        rb = self.robots[rid]
        start_node = self._nearest_node(rb.x, rb.y)
        rb.cur_node = start_node
        is_replan = rid in self.active_legs
        is_drift = rid in self._next_replan_reason_drift
        self._next_replan_reason_drift.discard(rid)

        self.res_table.release_all(rid)
        self.field.remove_agv(rid)
        self._active_pts.pop(rid, None)
        self.res_table.reserve_node(start_node, Interval(t_tick, t_tick + 2), rid)
        self._phys_locks[rid] = (start_node, t_tick, t_tick + 2)
        pl, _ = self.planner.plan_leg(
            start_node,
            goal_node,
            t_tick,
            self.res_table,
            self.field,
            rid,
            min(self.plan_horizon_ticks, t_tick + 120),
            goal_park_steps=1,
        )
        self.replans += 1
        if pl is None or not pl.nodes:
            pl, _ = self.planner.plan_leg(
                start_node,
                goal_node,
                t_tick + 2,
                self.res_table,
                self.field,
                rid,
                min(self.plan_horizon_ticks, t_tick + 122),
                goal_park_steps=1,
            )
        if pl is None or not pl.nodes:
            empty_rt = RustReservationTable.fast()
            pl, _ = self.planner.plan_leg(
                start_node,
                goal_node,
                t_tick,
                empty_rt,
                self.field,
                rid,
                max(self.plan_horizon_ticks, t_tick + 140),
                goal_park_steps=1,
            )

        self._plan_wall_ms_samples.append((time.perf_counter() - t0) * 1000.0)
        if pl is None or not pl.nodes:
            self.res_table.reserve_node(start_node, Interval(t_tick, t_tick + 3), rid)
            rb.route_nodes = [start_node]
            rb.route_times = [t_tick]
            rb.route_idx = 0
            rb.last_plan_fail_tick = t_tick
            rb.prev_rem_dist_m = 0.0
            self._route_times_sec[rid] = [float(t_tick) * self.mmap.macro_step_s]
            self._active_buf_s[rid] = 0.0
            self.init_plan_failures += 1
            return False

        pts = expand_planned_to_step_positions(start_node, t_tick, pl)
        self.planner.reserve_step_points_indexed(self.res_table, rid, pts, park_until=None)
        self.field.add_agv_path(rid, pts)
        self._active_pts[rid] = list(pts)
        w_nodes: list[str] = [start_node]
        w_times: list[int] = [t_tick]
        for nd_id, step_t in zip(pl.nodes, pl.arrival_steps):
            if w_nodes[-1] != nd_id:
                w_nodes.append(nd_id)
                w_times.append(int(step_t))
        rb.route_nodes = w_nodes
        rb.route_times = w_times
        rb.route_idx = 0
        rb.prev_rem_dist_m = self.remaining_route_dist_m(rb)
        self._route_times_sec[rid] = [float(tk) * self.mmap.macro_step_s for tk in rb.route_times]
        self._active_buf_s[rid] = 0.0
        if self.record_telemetry and rb.phase != "idle":
            self._open_leg_record(rid, goal_node, t_tick, is_replan=is_replan, is_drift=is_drift, waited=0)
            if is_drift:
                self.eta_drift_replans += 1
        return True

    def _macro_coordinate(self) -> None:
        if self.drift_thresh_ticks == 3 and not self.force_unified_macro:
            super()._macro_coordinate()
            return

        t_tick = self.macro_tick
        is_u0_family = self.upper_mode in ("U0_Orig_CPSA", "U0_H120_ArrMinus1")
        if not is_u0_family and self._sub_tick_div > 1:
            t_sub_phys = t_tick * self._sub_tick_div
            lock_sub = 2 * self._sub_tick_div
            for rid_p in self.robot_ids:
                rb_p = self.robots[rid_p]
                phys_nd = self._nearest_node(rb_p.x, rb_p.y)
                self._sub_res_table.reserve_node(phys_nd, Interval(t_sub_phys, t_sub_phys + lock_sub), rid_p)
                self._sub_phys_locks[rid_p] = (phys_nd, t_sub_phys, t_sub_phys + lock_sub)

        if self.record_telemetry:
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

            if self.drift_thresh_ticks is not None:
                for rid in self.robot_ids:
                    rb = self.robots[rid]
                    if rb.phase == "idle" or rb.cur_task is None or rb.service_rem_s > 0.0:
                        continue
                    if len(rb.route_nodes) > 1 and rb.route_idx + 1 < len(rb.route_times):
                        exp_t = rb.route_times[rb.route_idx + 1]
                        if t_tick > exp_t + self.drift_thresh_ticks:
                            self._next_replan_reason_drift.add(rid)
                            nxt_nd = rb.route_nodes[rb.route_idx + 1]
                            for ck_nd in (rb.cur_node, nxt_nd):
                                rec = self._recent_late_nodes.get(ck_nd)
                                if rec is not None and rec[0] != rid and rec[1] >= self.t_s:
                                    self.spillover_peer_replans += 1
                                    break

        # Exact MetricWorldSim._macro_coordinate loop with parameterized `drift_thresh_ticks`
        while self.pending and self.pending[0].release_s <= self.t_s + 1e-6:
            self.ready.append(self.pending.popleft())

        for rid in self.robot_ids:
            rb = self.robots[rid]
            if rb.cur_task is None or rb.service_rem_s > 0.0:
                continue
            tgt_id = rb.cur_task.pickup if rb.phase == "to_pickup" else rb.cur_task.dropoff
            tgt = self.mmap.network.node(tgt_id)
            if math.hypot(rb.x - tgt.x, rb.y - tgt.y) <= 0.85:
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

        idle_rids = [rid for rid in self.robot_ids if self.robots[rid].phase == "idle" and self.robots[rid].service_rem_s <= 0.0]
        if idle_rids and self.ready:
            used_tasks: set[int] = set()
            for rid in idle_rids:
                rb = self.robots[rid]
                best_tk: ContTask | None = None
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

        active_route_nodes: set[str] = set()
        for rid_a in self.robot_ids:
            rb_a = self.robots[rid_a]
            if rb_a.phase != "idle" and rb_a.route_nodes:
                active_route_nodes.update(rb_a.route_nodes[rb_a.route_idx :])
        occupied_nodes = {self.robots[r].cur_node for r in self.robot_ids}
        bay_nodes = set(self.mmap.pickup_nodes) | set(self.mmap.dropoff_nodes)
        for rid_i in self.robot_ids:
            rb_i = self.robots[rid_i]
            if rb_i.phase == "idle" and rb_i.service_rem_s <= 0.0 and len(rb_i.route_nodes) <= 1:
                if rb_i.cur_node in active_route_nodes or (
                    self.mmap.start_pool and rb_i.cur_node in bay_nodes and rb_i.cur_node not in self.mmap.start_pool
                ):
                    cands_p = [
                        n
                        for n in (self.mmap.start_pool or ())
                        if n not in active_route_nodes and n not in occupied_nodes and n != rb_i.cur_node
                    ]
                    if not cands_p:
                        cands_p = [
                            e.dst
                            for e in self.mmap.network.out_edges(rb_i.cur_node)
                            if e.dst not in active_route_nodes and e.dst not in occupied_nodes
                        ]
                    if cands_p:
                        cands_p.sort(
                            key=lambda nid: math.hypot(
                                self.mmap.network.node(nid).x - rb_i.x,
                                self.mmap.network.node(nid).y - rb_i.y,
                            )
                        )
                        self._plan_robot(rid_i, cands_p[0], t_tick)

        for rid_p in self.robot_ids:
            rb_p = self.robots[rid_p]
            phys_nd = self._nearest_node(rb_p.x, rb_p.y)
            rb_p.cur_node = phys_nd
            self.res_table.reserve_node(phys_nd, Interval(t_tick, t_tick + 2), rid_p)
            self._phys_locks[rid_p] = (phys_nd, t_tick, t_tick + 2)

        for rid in self.robot_ids:
            rb = self.robots[rid]
            if rb.phase == "idle" or rb.cur_task is None or rb.service_rem_s > 0.0:
                continue
            goal = rb.cur_task.pickup if rb.phase == "to_pickup" else rb.cur_task.dropoff
            if len(rb.route_nodes) <= 1 and rb.cur_node != goal:
                if (t_tick - rb.last_plan_fail_tick) >= 2:
                    self._plan_robot(rid, goal, t_tick)
                continue
            if self.drift_thresh_ticks is not None and rb.route_idx + 1 < len(rb.route_times):
                exp_t = rb.route_times[rb.route_idx + 1]
                if t_tick > exp_t + self.drift_thresh_ticks:
                    rb.eta_drift_ticks += 1
                    self._plan_robot(rid, goal, t_tick)

        for rid in self.robot_ids:
            rb = self.robots[rid]
            if self.rng.random() < self.disturb_slow_prob:
                rb.speed_factor = float(self.rng.uniform(0.65, 0.90))
            else:
                rb.speed_factor = 1.0

        self._refresh_res_views()


def _paired_wilcoxon(a: list[float], b: list[float]) -> float:
    diffs = [x - y for x, y in zip(a, b) if abs(x - y) > 1e-9]
    if len(diffs) < 2:
        return 1.0
    try:
        _, p = wilcoxon(diffs)
        return round(float(p), 4)
    except Exception:
        return 1.0


def _summarize(rows: list[dict[str, Any]], base_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(rows))
    comps = [float(r["completed_tasks"]) for r in rows]
    base_comps = [float(r["completed_tasks"]) for r in base_rows]
    waits = [float(r["wait_ratio"]) for r in rows]
    base_waits = [float(r["wait_ratio"]) for r in base_rows]
    detas = [abs(float(r["mean_deta_s"])) for r in rows if r["mean_deta_s"] is not None]
    return {
        "episodes": len(rows),
        "zero_col_cnt": f"{sum(1 for r in rows if r['zero_collisions'])}/{len(rows)}",
        "zero_col_pct": round(100.0 * sum(1 for r in rows if r["zero_collisions"]) / n, 1),
        "completed_tasks_mean": round(float(np.mean(comps)), 3),
        "completed_vs_base_pct": round(
            100.0 * (float(np.mean(comps)) - float(np.mean(base_comps))) / max(1e-6, float(np.mean(base_comps))),
            2,
        ),
        "p_comp_vs_base": _paired_wilcoxon(comps, base_comps),
        "wait_ratio_mean": round(float(np.mean(waits)), 4),
        "p_wait_vs_base": _paired_wilcoxon(waits, base_waits),
        "replans_tot": int(sum(r["replans"] for r in rows)),
        "eta_drift_replans_tot": int(sum(r["eta_drift_replans"] for r in rows)),
        "row_gate_events_tot": int(sum(r["row_gate_events"] for r in rows)),
        "shield_interventions_tot": int(sum(r["shield_interventions"] for r in rows)),
        "abs_deta_mean_s": round(float(np.mean(detas)) if detas else 0.0, 3),
        "plan_ms_mean": round(float(np.mean([r["plan_ms_mean"] for r in rows])), 3),
    }


def _eval_one(
    sc_label: str,
    mmap: Any,
    nr: int,
    nt: int,
    hz: float,
    ni: int,
    np_m: float,
    nv_m: float,
    lag_s: float,
    drop_p: float,
    upper_mode: str,
    sub_tick_div: int,
    custom_fn: Callable[..., tuple[PlannedPath | None, int]] | None,
    seed: int,
    drift_thresh_ticks: int | None = 3,
    force_unified_macro: bool = False,
) -> dict[str, Any]:
    sim = ConfigurableDriftAuditSim(
        mmap,
        upper_mode=upper_mode,
        sub_tick_div=sub_tick_div,
        custom_plan_fn=custom_fn,
        drift_thresh_ticks=drift_thresh_ticks,
        force_unified_macro=force_unified_macro,
        n_robots=nr,
        n_tasks=nt,
        horizon_s=hz,
        seed=seed,
        n_intruders=ni,
        obs_noise_pos_m=np_m,
        obs_noise_vel_mps=nv_m,
        reservation_lag_s=lag_s,
        reservation_dropout=drop_p,
        robust_safety=True,
    )
    res: MultiRobotEpisodeResult = sim.run_episode(
        m2_res_margin_selector, controller_name=f"{upper_mode}_th{drift_thresh_ticks}"
    )
    comp_legs = [lg for lg in sim.closed_legs if lg.completed and lg.end_t_s is not None]
    detas = [float(lg.end_t_s - lg.planned_eta_s) for lg in comp_legs if lg.end_t_s is not None]
    return {
        "scenario": sc_label,
        "upper_mode": upper_mode,
        "drift_thresh_ticks": drift_thresh_ticks,
        "seed": seed,
        "completed_tasks": int(res.completed_tasks),
        "n_tasks": int(nt),
        "makespan_s": float(res.makespan_s),
        "zero_collisions": bool(res.zero_collisions),
        "col_ticks": int(res.wall_collisions + res.peer_collisions + res.intruder_collisions),
        "wait_ratio": float(res.wait_ratio),
        "replans": int(res.replans),
        "eta_drift_replans": int(sim.eta_drift_replans),
        "row_gate_events": int(sim.row_gate_events),
        "shield_interventions": int(res.shield_interventions),
        "mean_deta_s": round(float(np.mean(detas)), 3) if detas else None,
        "plan_ms_mean": round(float(np.mean(sim._plan_wall_ms_samples)) if sim._plan_wall_ms_samples else 0.0, 3),
    }


def run_a1_factorial(maps: dict[str, Any]) -> dict[str, Any]:
    """Run A1 2x2 factorial (`Horizon x Reservation`) across all 18 DEV episodes."""
    specs = [
        ("Occluded_Jct_6R_Clean", maps["om"], 6, 12, 70.0, 2, 0.0, 0.0, 0.0, 0.0),
        ("MapA_8R_Clean", maps["ma"], 8, 16, 80.0, 2, 0.0, 0.0, 0.0, 0.0),
        ("MapB_12R_Clean", maps["mb"], 12, 20, 85.0, 2, 0.0, 0.0, 0.0, 0.0),
        ("Workshop_8R_Clean", maps["wm"], 8, 12, 160.0, 1, 0.0, 0.0, 0.0, 0.0),
        ("Occluded_Jct_6R_Noise0.12", maps["om"], 6, 12, 70.0, 2, 0.12, 0.20, 0.0, 0.0),
        ("Occluded_Jct_6R_Compound", maps["om"], 6, 12, 70.0, 2, 0.12, 0.20, 2.0, 0.10),
    ]
    variants = [
        ("U0_H90_ArrMinus1", "U0_Orig_CPSA", 1, None),
        ("U0_Pure_H90_EdgeSpanFix", "U0_Pure_H90_EdgeSpanFix", 1, u0_pure_h90_plan_fn),
        ("U0_H120_ArrMinus1", "U0_H120_ArrMinus1", 1, None),
        ("U1a_H120_EdgeSpanFix", "U1a_EdgeSpan_Fix", 1, u1a_plan_fn),
    ]
    rows: list[dict[str, Any]] = []
    by_var: dict[str, list[dict[str, Any]]] = {v[0]: [] for v in variants}
    for sc_label, mmap, nr, nt, hz, ni, np_m, nv_m, lag_s, drop_p in specs:
        for vlabel, umode, div, cfn in variants:
            for sd in DEV_SEEDS:
                r = _eval_one(
                    sc_label, mmap, nr, nt, hz, ni, np_m, nv_m, lag_s, drop_p, umode, div, cfn, sd
                )
                r["variant"] = vlabel
                rows.append(r)
                by_var[vlabel].append(r)
                print(
                    f"[A1 | {sc_label:24s} | {vlabel:24s} | s={sd}] "
                    f"zcol={r['zero_collisions']} comp={r['completed_tasks']}/{nt} drift={r['eta_drift_replans']}",
                    flush=True,
                )

    base = by_var["U0_H90_ArrMinus1"]
    summary = {v[0]: _summarize(by_var[v[0]], base) for v in variants}
    by_sc: dict[str, dict[str, Any]] = {}
    for sc_label, *_ in specs:
        by_sc[sc_label] = {}
        sc_base = [r for r in base if r["scenario"] == sc_label]
        for vlabel, *_ in variants:
            sc_rows = [r for r in by_var[vlabel] if r["scenario"] == sc_label]
            by_sc[sc_label][vlabel] = _summarize(sc_rows, sc_base)

    # Pure effects
    eff_span_h90 = summary["U0_Pure_H90_EdgeSpanFix"]["completed_tasks_mean"] - summary["U0_H90_ArrMinus1"]["completed_tasks_mean"]
    eff_span_h120 = summary["U1a_H120_EdgeSpanFix"]["completed_tasks_mean"] - summary["U0_H120_ArrMinus1"]["completed_tasks_mean"]
    eff_h_arr = summary["U0_H120_ArrMinus1"]["completed_tasks_mean"] - summary["U0_H90_ArrMinus1"]["completed_tasks_mean"]
    eff_h_span = summary["U1a_H120_EdgeSpanFix"]["completed_tasks_mean"] - summary["U0_Pure_H90_EdgeSpanFix"]["completed_tasks_mean"]

    return {
        "episodes": rows,
        "summary": summary,
        "by_scenario": by_sc,
        "factorial_effects": {
            "pure_edge_span_at_h90": round(eff_span_h90, 4),
            "pure_edge_span_at_h120": round(eff_span_h120, 4),
            "pure_horizon_120_at_arr_minus_1": round(eff_h_arr, 4),
            "pure_horizon_120_at_exact_span": round(eff_h_span, 4),
        },
    }


def run_a2_drift_sweep(maps: dict[str, Any]) -> dict[str, Any]:
    """Run A2 `eta_drift` threshold sweep (`{3, 6, 12, None}`) on clean DEV scenarios."""
    clean_specs = [
        ("Occluded_Jct_6R_Clean", maps["om"], 6, 12, 70.0, 2, 0.0, 0.0, 0.0, 0.0),
        ("MapA_8R_Clean", maps["ma"], 8, 16, 80.0, 2, 0.0, 0.0, 0.0, 0.0),
        ("MapB_12R_Clean", maps["mb"], 12, 20, 85.0, 2, 0.0, 0.0, 0.0, 0.0),
        ("Workshop_8R_Clean", maps["wm"], 8, 12, 160.0, 1, 0.0, 0.0, 0.0, 0.0),
    ]
    variants = [
        ("U0_Orig_CPSA", "U0_Orig_CPSA", 1, None),
        ("U0_Pure_H90_EdgeSpanFix", "U0_Pure_H90_EdgeSpanFix", 1, u0_pure_h90_plan_fn),
        ("U1b_KinoTime_CPSA", "U1b_KinoTime_CPSA", 4, make_u1b_plan_fn(gamma=4.0, turn_cost_s=0.5)),
    ]
    thresholds: list[int | None] = [3, 6, 12, None]

    rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {v[0]: {str(th): [] for th in thresholds} for v in variants}

    for sc_label, mmap, nr, nt, hz, ni, np_m, nv_m, lag_s, drop_p in clean_specs:
        for vlabel, umode, div, cfn in variants:
            for th in thresholds:
                for sd in DEV_SEEDS:
                    r = _eval_one(
                        sc_label,
                        mmap,
                        nr,
                        nt,
                        hz,
                        ni,
                        np_m,
                        nv_m,
                        lag_s,
                        drop_p,
                        umode,
                        div,
                        cfn,
                        sd,
                        drift_thresh_ticks=th,
                    )
                    r["variant"] = vlabel
                    rows.append(r)
                    grouped[vlabel][str(th)].append(r)
                    print(
                        f"[A2 | {sc_label:20s} | {vlabel:24s} | th={str(th):4s} | s={sd}] "
                        f"zcol={r['zero_collisions']} comp={r['completed_tasks']}/{nt} replans={r['replans']} drift={r['eta_drift_replans']}",
                        flush=True,
                    )

    u0_th3_base = grouped["U0_Orig_CPSA"]["3"]
    summary_matrix: dict[str, dict[str, Any]] = {}
    summary_vs_own_th3: dict[str, dict[str, Any]] = {}
    by_sc_u1b: dict[str, dict[str, Any]] = {}
    for vlabel, *_ in variants:
        summary_matrix[vlabel] = {}
        summary_vs_own_th3[vlabel] = {}
        for th in thresholds:
            key = str(th)
            summary_matrix[vlabel][key] = _summarize(grouped[vlabel][key], u0_th3_base)
            summary_vs_own_th3[vlabel][key] = _summarize(grouped[vlabel][key], grouped[vlabel]["3"])
    for sc_label, *_ in clean_specs:
        by_sc_u1b[sc_label] = {}
        sc_u0_3 = [r for r in u0_th3_base if r["scenario"] == sc_label]
        for th in thresholds:
            sc_rows = [r for r in grouped["U1b_KinoTime_CPSA"][str(th)] if r["scenario"] == sc_label]
            by_sc_u1b[sc_label][str(th)] = _summarize(sc_rows, sc_u0_3)

    return {
        "episodes": rows,
        "summary_vs_u0_th3": summary_matrix,
        "summary_vs_own_th3": summary_vs_own_th3,
        "u1b_by_scenario": by_sc_u1b,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render concise Markdown report for A1-A2 Track A attribution closure."""
    a1 = payload["a1_factorial"]
    a2 = payload["a2_drift_sweep"]
    fe = a1["factorial_effects"]
    lines = [
        "# Track A Attribution Closure (`A1-A2`)",
        "",
        "## A1: 2x2 Factorial Disentanglement (`Horizon x Reservation Span`, 18 DEV Episodes)",
        "",
        "| Variant | Horizon | Reservation | ZeroCol | Completed | vs U0 | p_comp | WaitRatio | Replans | ETA_Drift | RowGates | Plan_ms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    meta = {
        "U0_H90_ArrMinus1": ("+90", "`arr_i - 1`"),
        "U0_Pure_H90_EdgeSpanFix": ("+90", "`_reserve_exact_span`"),
        "U0_H120_ArrMinus1": ("+120", "`arr_i - 1`"),
        "U1a_H120_EdgeSpanFix": ("+120", "`_reserve_exact_span`"),
    }
    for vn, (hz_s, rs_s) in meta.items():
        s = a1["summary"][vn]
        lines.append(
            f"| `{vn}` | {hz_s} | {rs_s} | {s['zero_col_cnt']} ({s['zero_col_pct']:.1f}%) | "
            f"{s['completed_tasks_mean']:.3f} | {s['completed_vs_base_pct']:+.2f}% | {s['p_comp_vs_base']:.4f} | "
            f"{s['wait_ratio_mean']:.4f} | {s['replans_tot']} | {s['eta_drift_replans_tot']} | "
            f"{s['row_gate_events_tot']} | {s['plan_ms_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"- **Pure `_reserve_exact_span` effect at H90**: `{fe['pure_edge_span_at_h90']:+.4f}` tasks/ep",
            f"- **Pure `_reserve_exact_span` effect at H120**: `{fe['pure_edge_span_at_h120']:+.4f}` tasks/ep",
            f"- **Pure `+120` horizon effect under `arr_i - 1`**: `{fe['pure_horizon_120_at_arr_minus_1']:+.4f}` tasks/ep",
            f"- **Pure `+120` horizon effect under `_reserve_exact_span`**: `{fe['pure_horizon_120_at_exact_span']:+.4f}` tasks/ep",
            "",
            "### A1 Per-Scenario Completed Tasks Mean",
            "",
            "| Scenario | U0_H90_ArrMinus1 | U0_Pure_H90_EdgeSpanFix | U0_H120_ArrMinus1 | U1a_H120_EdgeSpanFix |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for sc, sc_d in a1["by_scenario"].items():
        lines.append(
            f"| `{sc}` | {sc_d['U0_H90_ArrMinus1']['completed_tasks_mean']:.3f} | "
            f"{sc_d['U0_Pure_H90_EdgeSpanFix']['completed_tasks_mean']:.3f} | "
            f"{sc_d['U0_H120_ArrMinus1']['completed_tasks_mean']:.3f} | "
            f"{sc_d['U1a_H120_EdgeSpanFix']['completed_tasks_mean']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## A2: `eta_drift` Trigger Threshold Sweep (12 Clean DEV Episodes)",
            "",
            "| Variant | Drift Thresh | ZeroCol | Completed | vs U0(th=3) | vs Own(th=3) | WaitRatio | Replans | ETA_Drift | |dETA| (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for vn in ("U0_Orig_CPSA", "U0_Pure_H90_EdgeSpanFix", "U1b_KinoTime_CPSA"):
        for th in ("3", "6", "12", "None"):
            s = a2["summary_vs_u0_th3"][vn][th]
            own = a2["summary_vs_own_th3"][vn][th]
            lines.append(
                f"| `{vn}` | `{th}` | {s['zero_col_cnt']} ({s['zero_col_pct']:.1f}%) | "
                f"{s['completed_tasks_mean']:.3f} | {s['completed_vs_base_pct']:+.2f}% | "
                f"{own['completed_vs_base_pct']:+.2f}% | {s['wait_ratio_mean']:.4f} | "
                f"{s['replans_tot']} | {s['eta_drift_replans_tot']} | {s['abs_deta_mean_s']:.3f} |"
            )

    lines.extend(
        [
            "",
            "### A2 `U1b_KinoTime_CPSA` Completed Tasks Mean by Clean Scenario & Threshold",
            "",
            "| Clean Scenario | th=3 | th=6 | th=12 | th=None |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for sc, sc_d in a2["u1b_by_scenario"].items():
        lines.append(
            f"| `{sc}` | {sc_d['3']['completed_tasks_mean']:.3f} | {sc_d['6']['completed_tasks_mean']:.3f} | "
            f"{sc_d['12']['completed_tasks_mean']:.3f} | {sc_d['None']['completed_tasks_mean']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Execute Track A attribution experiments A1-A2 and save JSON/Markdown outputs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    maps = {
        "om": build_occluded_junction_map(cell_m=2.6),
        "ma": build_grid_metric_map("Map_A", cell_m=2.6),
        "mb": build_grid_metric_map("Map_B", cell_m=2.6),
        "wm": build_workshop_metric_map(),
    }
    t0 = time.perf_counter()
    a1 = run_a1_factorial(maps)
    a2 = run_a2_drift_sweep(maps)
    elapsed_s = round(time.perf_counter() - t0, 2)
    payload = {
        "elapsed_s": elapsed_s,
        "a1_factorial": a1,
        "a2_drift_sweep": a2,
    }
    json_path = OUT_DIR / "track_a_attribution.json"
    md_path = OUT_DIR / "track_a_attribution.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Finished A1-A2 in {elapsed_s}s -> {md_path}", flush=True)


if __name__ == "__main__":
    main()
