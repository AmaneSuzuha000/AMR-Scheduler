"""Migrated Fast Accelerator for Track R (`RustReservationTable` + `RustSpaceTimeAStar(use_sipp=True)` + `SlotReservationPressureTable`).

Connects directly to CPS-A* Section 6 Future Work ("reducing planning latency") by migrating:
  1. `RustReservationTable` (native Rust interval sets + `clone_fast`/`restore_from`)
  2. `RustSpaceTimeAStarPlanner(use_sipp=True|False)` (native Rust Safe-Interval Path Planning)
  3. `SlotReservationPressureTable` (`O(1)` corridor pressure Q3 weights without Gaussian cache wipes)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Optional

from amr_sched.cvcr.evaluator import physical_edge_id
from amr_sched.network import Network
from amr_sched.planners.rust_spacetime_astar import RustSlotField, RustSpaceTimeAStarPlanner
from amr_sched.reservation import Interval
from amr_sched.rust_reservation import RustReservationTable
from experiments.track_r_cpsa.cpsa_pure_runner import (
    enforce_step_mutex_and_sync,
    expand_planned_to_step_positions,
    run_lifelong_mapd,
    shortest_dist_steps,
    verify_mapf_trace_safety,
)
from experiments.track_r_cpsa.maps_and_workloads import GridMapSpec


@dataclass(frozen=True)
class AccelLifelongResult:
    """Comparison record for Python CPS-A* vs Rust A* / Rust SIPP + SlotPressure Lifelong MAPD."""

    backend: str
    map_name: str
    seed: int
    n_agvs: int
    horizon: int
    completed_tasks: int
    throughput: float
    mean_waiting_time: float
    wall_ms: float
    plan_calls: int
    zero_conflicts: bool


def _reserve_rust_path(
    res_table: RustReservationTable,
    agv_id: str,
    step_positions: list[tuple[str, int]],
    *,
    park_until: Optional[int] = None,
) -> None:
    if not step_positions:
        return
    for i, (u_n, u_t) in enumerate(step_positions):
        res_table.reserve_node(u_n, Interval(u_t, u_t + 1), agv_id)
        if i + 1 < len(step_positions):
            v_n, v_t = step_positions[i + 1]
            if u_n != v_n and v_t > u_t:
                res_table.reserve_edge(u_n, v_n, Interval(u_t, v_t), agv_id)
    last_n, last_t = step_positions[-1]
    if park_until is not None and int(park_until) > last_t + 1:
        res_table.reserve_node(last_n, Interval(last_t + 1, int(park_until)), agv_id)


def _rust_hold_or_evade(
    aid: str,
    cur_node: str,
    t: int,
    plan_h: int,
    res_table: RustReservationTable,
    network: Network,
    plan_steps: dict[str, dict[int, str]],
    plan_end_t: dict[str, int],
) -> None:
    res_table.release_all(aid)
    res_table.reserve_node(cur_node, Interval(t, min(plan_h, t + 1)), aid)
    if t + 1 >= plan_h:
        plan_end_t[aid] = t
        return
    next_span = Interval(t + 1, min(plan_h, t + 2))
    safe2_span = Interval(t + 1, min(plan_h, t + 3))
    cap_cur = network.node(cur_node).capacity
    if res_table.is_node_free(cur_node, safe2_span, cap_cur, exclude_agv=aid):
        res_table.reserve_node(cur_node, safe2_span, aid)
        plan_steps[aid][t + 1] = cur_node
        plan_end_t[aid] = t + 1
        return
    edge_span = Interval(t, t + 1)
    for target_span in (safe2_span, next_span):
        for e in network.out_edges(cur_node):
            phys = physical_edge_id(e.src, e.dst)
            ecap = (
                min(e.capacity, network.edge(e.dst, e.src).capacity)
                if network.has_edge(e.dst, e.src)
                else e.capacity
            )
            occ = set(res_table.edge_conflicts(phys[0], phys[1], edge_span, exclude_agv=aid))
            if network.has_edge(phys[1], phys[0]) and phys[1] != phys[0]:
                occ |= set(res_table.edge_conflicts(phys[1], phys[0], edge_span, exclude_agv=aid))
            if len(occ) >= ecap:
                continue
            dst_cap = network.node(e.dst).capacity
            if res_table.is_node_free(e.dst, target_span, dst_cap, exclude_agv=aid):
                res_table.reserve_edge(cur_node, e.dst, edge_span, aid)
                res_table.reserve_node(e.dst, target_span, aid)
                plan_steps[aid][t + 1] = e.dst
                plan_end_t[aid] = t + 1
                return
    if res_table.is_node_free(cur_node, next_span, cap_cur, exclude_agv=aid):
        res_table.reserve_node(cur_node, next_span, aid)
    plan_steps[aid][t + 1] = cur_node
    plan_end_t[aid] = t + 1


def run_accelerated_lifelong_mapd(
    grid: GridMapSpec,
    init_positions: dict[str, str],
    request_stream: list[tuple[str, str, str]],
    *,
    backend: str = "rust_sipp_slot",
    horizon: int = 200,
    seed: int = 39,
) -> AccelLifelongResult:
    """Benchmark Lifelong MAPD across `python_cpsa` vs `rust_astar_slot` vs `rust_sipp_slot`."""
    if backend in ("python_cps", "python_cpsa"):
        t0 = time.perf_counter()
        py_res = run_lifelong_mapd(grid, init_positions, request_stream, algo="cps_lifelong", horizon=horizon, seed=seed)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return AccelLifelongResult(
            backend=backend,
            map_name=grid.name,
            seed=int(seed),
            n_agvs=len(init_positions),
            horizon=horizon,
            completed_tasks=py_res.completed_tasks,
            throughput=py_res.throughput,
            mean_waiting_time=py_res.mean_waiting_time,
            wall_ms=round(dt_ms, 2),
            plan_calls=py_res.total_expansions,
            zero_conflicts=py_res.zero_conflicts,
        )

    use_sipp = (backend == "rust_sipp_slot")
    planner = RustSpaceTimeAStarPlanner(max_speed=1.0, dt=1.0, max_expansions=120_000, use_sipp=use_sipp)
    res_table = RustReservationTable.fast()
    plan_h = int(horizon) + 120
    rust_field: RustSlotField = planner.create_slot_field(
        grid.network,
        spatial_radius_L=0,
        temporal_radius_T=0,
        slot_delta=3,
        max_track_horizon=plan_h + 20,
        t_start=0,
        time_aware=True,
    )
    dist_cache: dict[tuple[str, str], int] = {}

    agv_ids = sorted(init_positions.keys())
    pos: dict[str, str] = dict(init_positions)
    phase: dict[str, str] = {a: "idle" for a in agv_ids}
    cur_req: dict[str, Optional[tuple[str, str, str, int]]] = {a: None for a in agv_ids}
    plan_steps: dict[str, dict[int, str]] = {a: {0: pos[a]} for a in agv_ids}
    plan_end_t: dict[str, int] = {a: 0 for a in agv_ids}
    traces: dict[str, dict[int, str]] = {a: {0: pos[a]} for a in agv_ids}

    for aid in agv_ids:
        res_table.reserve_node(pos[aid], Interval(0, plan_h), aid)

    task_q: deque[tuple[str, str, str]] = deque(request_stream)
    completed = 0
    wait_sum = 0
    plan_calls = 0

    t0 = time.perf_counter()
    for t in range(horizon):
        if t > 0 and (t % 25 == 0):
            res_table.purge_before(t)

        if t > 0:
            for aid in agv_ids:
                nxt = plan_steps[aid].get(t, pos[aid])
                pos[aid] = nxt
                traces[aid][t] = nxt
                req = cur_req[aid]
                if req is not None:
                    _, pu, do, rel_t = req
                    if phase[aid] == "to_pickup" and nxt == pu and t >= plan_end_t[aid]:
                        phase[aid] = "need_dropoff"
                    elif phase[aid] == "to_dropoff" and nxt == do and t >= plan_end_t[aid]:
                        completed += 1
                        wait_sum += max(0, t - rel_t)
                        phase[aid] = "idle"
                        cur_req[aid] = None

        need_plan: list[tuple[str, str]] = []
        for aid in agv_ids:
            if phase[aid] == "need_dropoff" and cur_req[aid] is not None:
                need_plan.append((aid, cur_req[aid][2]))
                phase[aid] = "to_dropoff"
            elif phase[aid] == "to_pickup" and cur_req[aid] is not None and t >= plan_end_t[aid]:
                need_plan.append((aid, cur_req[aid][1]))
            elif phase[aid] == "to_dropoff" and cur_req[aid] is not None and t >= plan_end_t[aid]:
                need_plan.append((aid, cur_req[aid][2]))
            elif phase[aid] == "idle" and task_q:
                req_id, pu, do = task_q.popleft()
                cur_req[aid] = (req_id, pu, do, t)
                if pos[aid] == pu:
                    phase[aid] = "to_dropoff"
                    need_plan.append((aid, do))
                else:
                    phase[aid] = "to_pickup"
                    need_plan.append((aid, pu))

        need_set = {x[0] for x in need_plan}
        for aid in agv_ids:
            if aid not in need_set and phase[aid] == "idle" and t >= plan_end_t[aid]:
                cap_p = grid.network.node(pos[aid]).capacity
                if res_table.is_node_free(pos[aid], Interval(t, plan_h), cap_p, exclude_agv=aid):
                    res_table.release_all(aid)
                    res_table.reserve_node(pos[aid], Interval(t, plan_h), aid)
                else:
                    _rust_hold_or_evade(aid, pos[aid], t, plan_h, res_table, grid.network, plan_steps, plan_end_t)

        if need_plan:
            for aid, _ in need_plan:
                res_table.release_all(aid)
                res_table.reserve_node(pos[aid], Interval(t, min(plan_h, t + 1)), aid)
                cap_p = grid.network.node(pos[aid]).capacity
                if t + 3 <= plan_h and res_table.is_node_free(pos[aid], Interval(t + 1, t + 3), cap_p, exclude_agv=aid):
                    res_table.reserve_node(pos[aid], Interval(t + 1, t + 3), aid)
                elif t + 2 <= plan_h and res_table.is_node_free(pos[aid], Interval(t + 1, t + 2), cap_p, exclude_agv=aid):
                    res_table.reserve_node(pos[aid], Interval(t + 1, t + 2), aid)
                rust_field.remove_agv(aid)

            goal_map = dict(need_plan)

            def _dyn_prio(aid: str) -> tuple[float, str]:
                req = cur_req[aid]
                age = float(t - req[3]) if req is not None else 0.0
                base_d = float(shortest_dist_steps(grid.network, pos[aid], goal_map[aid], dist_cache))
                cong = float(rust_field.slot_Phi(pos[aid], t))
                return (base_d - 0.25 * age - 4.0 * cong, aid)

            pq_aids: deque[str] = deque(sorted(goal_map.keys(), key=_dyn_prio))
            uncommitted: dict[str, list[tuple[str, int]]] = {}
            retries: dict[str, int] = {a: 0 for a in goal_map}
            R_lim = 3
            leg_h = min(plan_h, t + 120)

            def _try_sipp_with_park(aid_x: str, gl_x: str) -> tuple[list[tuple[str, int]] | None, int, dict[str, int]]:
                nonlocal plan_calls
                p1, hits1 = planner.plan_leg(
                    grid.network,
                    pos[aid_x],
                    gl_x,
                    t,
                    res_table,
                    rust_field,
                    aid_x,
                    leg_h,
                    goal_park_steps=2,
                    alpha=1.0,
                    beta=0.0 if use_sipp else 0.5,
                    gamma=2.5,
                    collect_blockers=True,
                )
                plan_calls += 1
                if p1 is not None:
                    pts1 = expand_planned_to_step_positions(pos[aid_x], t, p1)
                    pe1 = min(plan_h, pts1[-1][1] + 3)
                    return pts1, pe1, hits1
                return None, t, hits1

            while pq_aids:
                aid = pq_aids.popleft()
                gl = goal_map[aid]
                res_table.release_all(aid)
                res_table.reserve_node(pos[aid], Interval(t, min(plan_h, t + 1)), aid)
                pts, park_end, hits = _try_sipp_with_park(aid, gl)
                if pts is not None:
                    planner.reserve_step_points_indexed(grid.network, res_table, aid, pts, park_until=park_end)
                    for node_s, step_s in pts:
                        plan_steps[aid][step_s] = node_s
                    plan_end_t[aid] = pts[-1][1]
                    uncommitted[aid] = pts
                    rust_field.add_agv_path(aid, pts, horizon=plan_h)
                    continue

                if retries[aid] < R_lim and uncommitted:
                    cand_blockers = {
                        ow: int(cnt) for ow, cnt in hits.items() if ow in uncommitted and ow != aid
                    }
                    if cand_blockers:
                        blocker_j = max(cand_blockers, key=lambda j: (cand_blockers[j], -len(uncommitted[j]), j))
                        saved_j = uncommitted.pop(blocker_j)
                        res_table.release_all(blocker_j)
                        res_table.reserve_node(pos[blocker_j], Interval(t, min(plan_h, t + 1)), blocker_j)
                        rust_field.remove_agv(blocker_j)
                        retries[aid] += 1
                        pts_a, park_a, _ = _try_sipp_with_park(aid, gl)
                        if pts_a is not None:
                            planner.reserve_step_points_indexed(grid.network, res_table, aid, pts_a, park_until=park_a)
                            for node_s, step_s in pts_a:
                                plan_steps[aid][step_s] = node_s
                            plan_end_t[aid] = pts_a[-1][1]
                            uncommitted[aid] = pts_a
                            rust_field.add_agv_path(aid, pts_a, horizon=plan_h)
                            pq_aids.append(blocker_j)
                            continue
                        planner.reserve_step_points_indexed(
                            grid.network,
                            res_table,
                            blocker_j,
                            saved_j,
                            park_until=min(plan_h, saved_j[-1][1] + 3),
                        )
                        rust_field.add_agv_path(blocker_j, saved_j, horizon=plan_h)
                        uncommitted[blocker_j] = saved_j

                _rust_hold_or_evade(aid, pos[aid], t, plan_h, res_table, grid.network, plan_steps, plan_end_t)

        if t + 1 < horizon:
            enforce_step_mutex_and_sync(agv_ids, pos, grid.network, res_table, plan_steps, plan_end_t, t, plan_h)

    dt_ms = (time.perf_counter() - t0) * 1000.0
    safe_ok, _ = verify_mapf_trace_safety(grid.network, traces)
    return AccelLifelongResult(
        backend=backend,
        map_name=grid.name,
        seed=int(seed),
        n_agvs=len(agv_ids),
        horizon=horizon,
        completed_tasks=completed,
        throughput=round(float(completed) / max(1, horizon), 4),
        mean_waiting_time=round(float(wait_sum) / max(1, completed), 2),
        wall_ms=round(dt_ms, 2),
        plan_calls=plan_calls,
        zero_conflicts=safe_ok,
    )
