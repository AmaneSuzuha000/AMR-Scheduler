"""Pure MAPF / Lifelong-MAPD baseline reproduction runner (`CA*` vs `CPS-A*`) for Track R.

Maintains strict MAPF safety invariants:
  - Per-step vertex occupancy intervals `[t, t+1)` prevent simultaneous occupancy (`Eq. A.8`).
  - Bidirectional `physical_edge_id` edge intervals prevent head-on swaps (`Eq. A.9`).
  - Independent trace verifier confirms ZERO collisions across all steps.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import math
from typing import Any, Optional

from amr_sched.cvcr.evaluator import physical_edge_id
from amr_sched.network import Network
from amr_sched.planners.base import PlannedPath
from amr_sched.reservation import Interval, ReservationTable
from experiments.cps_astar_routing.cps_astar_graph import (
    CPSGraphFieldConfig,
    CPSAStarConfig,
    CPSAStarGraphPlanner,
    GraphCongestionField,
)
from experiments.track_r_cpsa.maps_and_workloads import GridMapSpec


@dataclass(frozen=True)
class OneShotResult:
    algo: str
    map_name: str
    seed: int
    n_total: int
    n_success: int
    success_rate: float
    apl: float
    avg_congestion: float
    acc: float
    total_expansions: int
    rollback_attempts: int
    rollback_wasted: int
    zero_conflicts: bool


@dataclass(frozen=True)
class LifelongResult:
    algo: str
    map_name: str
    seed: int
    n_agvs: int
    horizon: int
    completed_tasks: int
    throughput: float
    mean_waiting_time: float
    mean_queue_len: float
    avg_congestion: float
    total_expansions: int
    rollback_attempts: int
    rollback_wasted: int
    zero_conflicts: bool


def expand_planned_to_step_positions(
    start_node: str,
    start_step: int,
    planned: PlannedPath,
) -> list[tuple[str, int]]:
    """Expand collapsed `PlannedPath` waits/moves into explicit step-by-step `(node, t)` list."""
    out: list[tuple[str, int]] = [(start_node, int(start_step))]
    cur_n = start_node
    cur_t = int(start_step)
    for nxt_n, arr_t in zip(planned.nodes, planned.arrival_steps):
        arr_i = int(arr_t)
        # In unit-step grid graphs (move takes 1 step), dep_t = arr_i - 1; waits happen on cur_n
        dep_t = arr_i - 1
        while cur_t < dep_t:
            cur_t += 1
            out.append((cur_n, cur_t))
        while cur_t < arr_i:
            cur_t += 1
            out.append((nxt_n, cur_t))
        cur_n = nxt_n
    return out


def reserve_mapf_path(
    res_table: ReservationTable,
    agv_id: str,
    step_positions: list[tuple[str, int]],
    *,
    park_until: int | None = None,
) -> None:
    """Reserve per-step node intervals `[t, t+1)` AND bidirectional edges for exact MAPF safety."""
    if not step_positions:
        return
    for i, (u_n, u_t) in enumerate(step_positions):
        res_table.reserve_node(u_n, Interval(u_t, u_t + 1), agv_id)
        if i + 1 < len(step_positions):
            v_n, v_t = step_positions[i + 1]
            if u_n != v_n and v_t > u_t:
                res_table.reserve_edge(u_n, v_n, Interval(u_t, v_t), agv_id)
    last_n, last_t = step_positions[-1]
    if park_until is not None and park_until > last_t + 1:
        res_table.reserve_node(last_n, Interval(last_t + 1, int(park_until)), agv_id)


def verify_mapf_trace_safety(
    network: Network,
    traces: dict[str, dict[int, str]],
) -> tuple[bool, list[str]]:
    """Independently verify vertex capacity, edge validity, and zero reverse-edge collisions."""
    if not traces:
        return True, []
    all_steps = sorted({t for tr in traces.values() for t in tr})
    issues: list[str] = []
    for t in all_steps:
        by_node: dict[str, list[str]] = {}
        edge_dir: dict[tuple[str, str], list[str]] = {}
        for aid, tr in traces.items():
            if t not in tr:
                continue
            pos = tr[t]
            by_node.setdefault(pos, []).append(aid)
            if (t - 1) in tr:
                prev = tr[t - 1]
                if prev != pos:
                    if not network.has_edge(prev, pos):
                        issues.append(f"t={t}:{aid}:invalid_edge:{prev}->{pos}")
                    edge_dir.setdefault((prev, pos), []).append(aid)

        for nid, owners in by_node.items():
            cap = network.node(nid).capacity
            if len(owners) > cap:
                issues.append(f"t={t}:vertex_collision:{nid}:{owners}")

        seen_phys: set[tuple[str, str]] = set()
        for (u, v), fwd_owners in edge_dir.items():
            phys = physical_edge_id(u, v)
            if phys in seen_phys:
                continue
            seen_phys.add(phys)
            rev_owners = edge_dir.get((v, u), [])
            pcap = (
                min(network.edge(u, v).capacity, network.edge(v, u).capacity)
                if network.has_edge(v, u)
                else network.edge(u, v).capacity
            )
            if fwd_owners and rev_owners and pcap <= 1:
                issues.append(f"t={t}:swap_collision:{u}<->{v}:{fwd_owners} vs {rev_owners}")
            elif len(fwd_owners) + len(rev_owners) > pcap:
                issues.append(f"t={t}:edge_cap:{phys}:{fwd_owners + rev_owners}")
    return (len(issues) == 0), issues


def shortest_dist_steps(network: Network, start: str, goal: str, cache: dict[tuple[str, str], int]) -> int:
    key = (start, goal)
    if key in cache:
        return cache[key]
    if start == goal:
        cache[key] = 0
        return 0
    dist = {start: 0}
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            cache[key] = d
            return d
        if d > dist.get(u, 10**9):
            continue
        for e in network.out_edges(u):
            nd = d + 1
            if nd < dist.get(e.dst, 10**9):
                dist[e.dst] = nd
                heapq.heappush(pq, (nd, e.dst))
    cache[key] = 10**9
    return 10**9


def impact_aware_blocker_score(
    failed_agv: str,
    blocker_j: str,
    conflict_hits: dict[str, int],
    remaining_len_j: int,
    local_cong_j: float,
    unblock_probe_ok: Optional[bool] = None,
) -> float:
    """Direction B diagnostic utility score S_j = expected unblock gain / (replan cost + eps)."""
    gain = float(conflict_hits.get(blocker_j, 0))
    if unblock_probe_ok is True:
        gain += 50.0
    elif unblock_probe_ok is False:
        gain *= 0.05
    cost = max(1.0, float(remaining_len_j) + 4.0 * float(local_cong_j))
    return gain / (cost + 1e-6)


def _eval_field_metrics(
    field_eval: GraphCongestionField,
    paths_by_agv: dict[str, list[tuple[str, int]]],
    *,
    high_threshold: float = 0.25,
) -> tuple[float, float]:
    tot_states = 0
    sum_ac = 0.0
    hi_cnt = 0
    for pts in paths_by_agv.values():
        for nd, t in pts:
            val = field_eval.phi(nd, t)
            sum_ac += val
            tot_states += 1
            if val >= high_threshold:
                hi_cnt += 1
    if tot_states == 0:
        return 0.0, 0.0
    return sum_ac / float(tot_states), float(hi_cnt) / float(tot_states)


def run_oneshot_mapf(
    grid: GridMapSpec,
    requests: list[tuple[str, str, str]],
    *,
    algo: str = "cps_oneshot",
    horizon: int = 120,
    fixed_bg_requests: list[tuple[str, str, str]] = (),
    seed: int = 0,
    use_impact_blocker: bool = False,
) -> OneShotResult:
    """Run one-shot MAPF (`ca_star` vs `cps_oneshot`) cleanly on pure grid `Network`."""
    is_ca = (algo == "ca_star")
    cfg = (
        CPSAStarConfig(mode="oneshot", beta=0.0 if is_ca else 0.5, gamma=0.0 if is_ca else 5.0, lookahead_omega=0.0 if is_ca else 50.0, batch_update_K=20, retry_limit_R=0 if is_ca else 5)
    )
    planner = CPSAStarGraphPlanner(grid.network, cfg, max_speed=1.0, dt=1.0)
    res_table = ReservationTable()
    field = GraphCongestionField(grid.network, cfg.field, t_start=0)

    # Plan fixed background schedules first if Table 5 mode
    bg_paths: dict[str, list[tuple[str, int]]] = {}
    for aid, st, gl in fixed_bg_requests:
        planned, _ = planner.plan_leg(
            st, gl, 0, res_table, field, aid, horizon, goal_park_until=horizon
        )
        if planned is not None:
            pts = expand_planned_to_step_positions(st, 0, planned)
            reserve_mapf_path(res_table, aid, pts, park_until=horizon)
            field.add_agv_path(aid, pts, horizon=horizon)
            bg_paths[aid] = pts

    # Reserve stationary anchors for ALL active test starts BEFORE sequential planning?
    # Standard CA* / CPS-A* reserves paths sequentially; starts are distinct.
    req_map = {aid: (st, gl) for aid, st, gl in requests}
    q: deque[str] = deque([aid for aid, _, _ in requests])
    uncommitted: dict[str, list[tuple[str, int]]] = {}
    accepted: dict[str, list[tuple[str, int]]] = {}
    retries: dict[str, int] = {aid: 0 for aid in req_map}
    total_exp = 0
    rb_attempts = 0
    rb_wasted = 0
    R_lim = 0 if is_ca else int(cfg.retry_limit_R)
    K_batch = max(1, int(cfg.batch_update_K))

    while q:
        aid = q.popleft()
        st, gl = req_map[aid]
        planned, hits = planner.plan_leg(
            st, gl, 0, res_table, field, aid, horizon, goal_park_until=horizon
        )
        total_exp += int(getattr(planner, "last_expansions", 0))
        if planned is not None:
            pts = expand_planned_to_step_positions(st, 0, planned)
            reserve_mapf_path(res_table, aid, pts, park_until=horizon)
            uncommitted[aid] = pts
            if len(uncommitted) >= K_batch:
                for caid, cpts in uncommitted.items():
                    accepted[caid] = cpts
                    if not is_ca:
                        field.add_agv_path(caid, cpts, horizon=horizon)
                uncommitted.clear()
            continue

        if retries[aid] >= R_lim or not uncommitted:
            continue

        if use_impact_blocker:
            hit_blockers = [j for j in uncommitted if int(hits.get(j, 0)) > 0]
            if not hit_blockers:
                # Direction B: No uncommitted agent blocked `aid` (only fixed background / committed) -> skip futile rollback
                continue
            def _sc(j: str) -> float:
                pts_j = uncommitted[j]
                return impact_aware_blocker_score(aid, j, hits, len(pts_j), field.phi(pts_j[0][0], 0))
            ranked = sorted(hit_blockers, key=lambda j: (_sc(j), j), reverse=True)
            candidates = ranked[: min(2, len(ranked))]
            rb_attempts += 1
            retries[aid] += 1
            chosen_j: str | None = None
            planned_chk = None
            for cand_j in candidates:
                saved_j = uncommitted.pop(cand_j)
                res_table.release_all(cand_j)
                field.remove_agv(cand_j)
                p_try, _ = planner.plan_leg(
                    st, gl, 0, res_table, field, aid, horizon, goal_park_until=horizon
                )
                total_exp += int(getattr(planner, "last_expansions", 0))
                if p_try is not None:
                    chosen_j = cand_j
                    planned_chk = p_try
                    break
                reserve_mapf_path(res_table, cand_j, saved_j, park_until=horizon)
                uncommitted[cand_j] = saved_j
            if chosen_j is None or planned_chk is None:
                rb_wasted += 1
                continue
            pts_aid = expand_planned_to_step_positions(st, 0, planned_chk)
            reserve_mapf_path(res_table, aid, pts_aid, park_until=horizon)
            uncommitted[aid] = pts_aid
            q.append(chosen_j)
            continue

        rb_attempts += 1
        active_blockers = [j for j in uncommitted]
        blocker_j = max(active_blockers, key=lambda j: (int(hits.get(j, 0)), j))

        saved_j = uncommitted.pop(blocker_j)
        res_table.release_all(blocker_j)
        field.remove_agv(blocker_j)

        # Quick check whether releasing blocker_j actually enabled aid
        retries[aid] += 1
        planned_chk, _ = planner.plan_leg(
            st, gl, 0, res_table, field, aid, horizon, goal_park_until=horizon
        )
        total_exp += int(getattr(planner, "last_expansions", 0))
        if planned_chk is None:
            # Blocker selection wasted! Restore blocker_j atomically
            rb_wasted += 1
            reserve_mapf_path(res_table, blocker_j, saved_j, park_until=horizon)
            uncommitted[blocker_j] = saved_j
            continue
        pts_aid = expand_planned_to_step_positions(st, 0, planned_chk)
        reserve_mapf_path(res_table, aid, pts_aid, park_until=horizon)
        uncommitted[aid] = pts_aid
        q.append(blocker_j)

    for caid, cpts in uncommitted.items():
        accepted[caid] = cpts
        if not is_ca:
            field.add_agv_path(caid, cpts, horizon=horizon)

    # Verify trace safety across all accepted + bg
    traces: dict[str, dict[int, str]] = {}
    all_solved = {**bg_paths, **accepted}
    for aid, pts in all_solved.items():
        tr = {t: n for n, t in pts}
        ln, lt = pts[-1]
        for t in range(lt + 1, horizon):
            tr[t] = ln
        traces[aid] = tr
    safe_ok, _ = verify_mapf_trace_safety(grid.network, traces)

    # Reconstruct neutral reference field L=T=1, sigma=2 per Section 5.1
    eval_field = GraphCongestionField(grid.network, CPSGraphFieldConfig(), t_start=0)
    for aid, pts in all_solved.items():
        eval_field.add_agv_path(aid, pts, horizon=horizon)
    ac, acc = _eval_field_metrics(eval_field, accepted)
    apl = sum(max(0, len(pts) - 1) for pts in accepted.values()) / max(1, len(accepted))
    n_tot = len(requests)
    n_ok = len(accepted)
    return OneShotResult(
        algo=algo,
        map_name=grid.name,
        seed=int(seed),
        n_total=n_tot,
        n_success=n_ok,
        success_rate=round(float(n_ok) / max(1, n_tot), 4),
        apl=round(apl, 3),
        avg_congestion=round(ac, 4),
        acc=round(acc, 4),
        total_expansions=total_exp,
        rollback_attempts=rb_attempts,
        rollback_wasted=rb_wasted,
        zero_conflicts=safe_ok,
    )


def _hold_or_evade_step(
    aid: str,
    cur_node: str,
    t: int,
    horizon: int,
    res_table: ReservationTable,
    network: Network,
    plan_steps: dict[str, dict[int, str]],
    plan_end_t: dict[str, int],
) -> None:
    """Safely hold at `cur_node` for `t+1` if free, else step aside onto a free neighbor."""
    res_table.release_all(aid)
    res_table.reserve_node(cur_node, Interval(t, min(horizon, t + 1)), aid)
    if t + 1 >= horizon:
        plan_end_t[aid] = t
        return
    next_span = Interval(t + 1, min(horizon, t + 2))
    safe2_end = min(horizon, t + 3)
    safe2_span = Interval(t + 1, safe2_end)
    cap_cur = network.node(cur_node).capacity
    if res_table.is_node_free(cur_node, safe2_span, cap_cur, exclude_agv=aid):
        res_table.reserve_node(cur_node, safe2_span, aid)
        plan_steps[aid][t + 1] = cur_node
        plan_end_t[aid] = t + 1
        return

    edge_span = Interval(t, t + 1)
    # Prefer stepping aside to a neighbor free for [t+1, t+3) so we don't get boxed in at t+1
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
        return

    # Hold in place without invalidating existing reservation table when fully boxed in
    plan_steps[aid][t + 1] = cur_node
    plan_end_t[aid] = t + 1


def enforce_step_mutex_and_sync(
    agv_ids: list[str],
    cur_pos: dict[str, str],
    network: Network,
    res_table: Any,
    plan_steps: dict[str, dict[int, str]],
    plan_end_t: dict[str, int],
    t: int,
    plan_h: int,
) -> dict[str, str]:
    """Enforce deterministic single-step capacity & bidirectional edge safety at `t -> t+1`."""
    desired = {aid: plan_steps[aid].get(t + 1, cur_pos[aid]) for aid in agv_ids}
    nxt = dict(desired)
    for _ in range(len(agv_ids) + 2):
        changed = False
        stay_counts: dict[str, int] = {}
        claimers: dict[str, list[str]] = {}
        dir_movers: dict[tuple[str, str], list[str]] = {}
        for a in agv_ids:
            u, v = cur_pos[a], nxt[a]
            if u == v:
                stay_counts[u] = stay_counts.get(u, 0) + 1
            else:
                claimers.setdefault(v, []).append(a)
                dir_movers.setdefault((u, v), []).append(a)

        for a in agv_ids:
            u, v = cur_pos[a], nxt[a]
            if u == v:
                continue
            if not network.has_edge(u, v):
                nxt[a] = u
                changed = True
                continue
            e = network.edge(u, v)
            ecap = min(e.capacity, network.edge(v, u).capacity) if network.has_edge(v, u) else e.capacity
            if dir_movers.get((v, u)):
                opp_aids = dir_movers[(v, u)]
                if len(dir_movers[(u, v)]) + len(opp_aids) > ecap or ecap <= 1 or a > min(opp_aids):
                    nxt[a] = u
                    changed = True
                    continue
            if len(dir_movers.get((u, v), [])) > ecap and a not in sorted(dir_movers[(u, v)])[:ecap]:
                nxt[a] = u
                changed = True
                continue
            v_cap = network.node(v).capacity
            avail_v = v_cap - stay_counts.get(v, 0)
            if avail_v <= 0 or a not in sorted(claimers.get(v, []))[:avail_v]:
                nxt[a] = u
                changed = True
                continue
        if not changed:
            break

    # Sync deviations safely so `res_table` never carries stale ghost paths after a freeze
    for a in agv_ids:
        if nxt[a] != desired[a]:
            plan_steps[a][t + 1] = nxt[a]
            plan_end_t[a] = min(int(plan_end_t[a]), t + 1)
            res_table.release_all(a)
            res_table.reserve_node(cur_pos[a], Interval(t, min(plan_h, t + 1)), a)
            if t + 2 <= plan_h and res_table.is_node_free(
                nxt[a], Interval(t + 1, t + 2), network.node(nxt[a]).capacity, exclude_agv=a
            ):
                res_table.reserve_node(nxt[a], Interval(t + 1, t + 2), a)
    return nxt


def run_lifelong_mapd(
    grid: GridMapSpec,
    init_positions: dict[str, str],
    request_stream: list[tuple[str, str, str]],
    *,
    algo: str = "cps_lifelong",
    horizon: int = 500,
    seed: int = 0,
    use_impact_blocker: bool = False,
) -> LifelongResult:
    """Run Algorithm 2 Lifelong two-stage pickup->dropoff MAPD (`ca_star` vs `cps_lifelong`)."""
    is_ca = (algo == "ca_star")
    cfg = CPSAStarConfig(
        mode="lifelong",
        beta=0.0 if is_ca else 0.5,
        gamma=0.0 if is_ca else 5.0,
        lookahead_omega=0.0,
        batch_update_K=20,
        retry_limit_R=0 if is_ca else 5,
    )
    planner = CPSAStarGraphPlanner(grid.network, cfg, max_speed=1.0, dt=1.0)
    plan_h = int(horizon) + 120
    res_table = ReservationTable()
    field = GraphCongestionField(grid.network, CPSGraphFieldConfig(max_track_horizon=plan_h + 20), t_start=0)
    dist_cache: dict[tuple[str, str], int] = {}

    agv_ids = sorted(init_positions.keys())
    pos: dict[str, str] = dict(init_positions)
    # Phase: "idle" | "to_pickup" | "to_dropoff"
    phase: dict[str, str] = {a: "idle" for a in agv_ids}
    cur_req: dict[str, Optional[tuple[str, str, str, int]]] = {a: None for a in agv_ids}  # (req_id, pu, do, start_step)
    plan_steps: dict[str, dict[int, str]] = {a: {0: pos[a]} for a in agv_ids}
    plan_end_t: dict[str, int] = {a: 0 for a in agv_ids}
    traces: dict[str, dict[int, str]] = {a: {0: pos[a]} for a in agv_ids}

    for aid in agv_ids:
        res_table.reserve_node(pos[aid], Interval(0, plan_h), aid)

    task_q: deque[tuple[str, str, str]] = deque(request_stream)
    completed = 0
    wait_sum = 0
    q_len_sum = 0
    total_exp = 0
    rb_attempts = 0
    rb_wasted = 0

    for t in range(horizon):
        if t > 0 and (t % 25 == 0):
            res_table.purge_before(t)

        # 1. Advance active executions along plan_steps from t -> t+1
        if t > 0:
            for aid in agv_ids:
                nxt = plan_steps[aid].get(t, pos[aid])
                pos[aid] = nxt
                traces[aid][t] = nxt
                req = cur_req[aid]
                if req is not None:
                    rid, pu, do, rel_t = req
                    if phase[aid] == "to_pickup" and nxt == pu and t >= plan_end_t[aid]:
                        phase[aid] = "need_dropoff"
                    elif phase[aid] == "to_dropoff" and nxt == do and t >= plan_end_t[aid]:
                        completed += 1
                        wait_sum += max(0, t - rel_t)
                        phase[aid] = "idle"
                        cur_req[aid] = None

        # 2. Construct active planning set Q_t (idle assigned new tasks + pickup completed + unfinished holds)
        need_plan: list[tuple[str, str]] = []  # (aid, target_goal)
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

        q_len_sum += len(task_q)

        # Keep idle AGVs safely reserved as far ahead as conflict-free, else side-step
        need_set = {x[0] for x in need_plan}
        for aid in agv_ids:
            if aid not in need_set and phase[aid] == "idle" and t >= plan_end_t[aid]:
                cap_p = grid.network.node(pos[aid]).capacity
                if res_table.is_node_free(pos[aid], Interval(t, plan_h), cap_p, exclude_agv=aid):
                    res_table.release_all(aid)
                    res_table.reserve_node(pos[aid], Interval(t, plan_h), aid)
                else:
                    _hold_or_evade_step(aid, pos[aid], t, plan_h, res_table, grid.network, plan_steps, plan_end_t)

        if need_plan:
            # Protect `pos[aid]` at `[t, t+2)` (when free) before planning so higher-priority AGVs don't trap it
            for aid, _ in need_plan:
                res_table.release_all(aid)
                res_table.reserve_node(pos[aid], Interval(t, min(plan_h, t + 1)), aid)
                cap_p = grid.network.node(pos[aid]).capacity
                if t + 2 <= plan_h and res_table.is_node_free(pos[aid], Interval(t + 1, t + 2), cap_p, exclude_agv=aid):
                    res_table.reserve_node(pos[aid], Interval(t + 1, t + 2), aid)
                field.remove_agv(aid)

            goal_map = dict(need_plan)

            def _dyn_prio(aid: str) -> tuple[float, str]:
                req = cur_req[aid]
                age = float(t - req[3]) if req is not None else 0.0
                base_d = float(shortest_dist_steps(grid.network, pos[aid], goal_map[aid], dist_cache))
                cong = 0.0 if is_ca else field.slot_Phi(pos[aid], t)
                score = base_d - float(cfg.prio_age_tau) * age - float(cfg.prio_cong_kappa) * cong
                return (score, aid)

            pq_aids: deque[str] = deque(sorted(goal_map.keys(), key=_dyn_prio))
            uncommitted: dict[str, list[tuple[str, int]]] = {}
            retries: dict[str, int] = {a: 0 for a in goal_map}
            R_lim = 0 if is_ca else int(cfg.retry_limit_R)
            leg_h = min(plan_h, t + 120)

            while pq_aids:
                aid = pq_aids.popleft()
                gl = goal_map[aid]
                res_table.release_all(aid)
                res_table.reserve_node(pos[aid], Interval(t, min(plan_h, t + 1)), aid)
                planned, hits = planner.plan_leg(
                    pos[aid], gl, t, res_table, field, aid, leg_h, goal_park_steps=2
                )
                total_exp += int(getattr(planner, "last_expansions", 0))
                if planned is not None:
                    pts = expand_planned_to_step_positions(pos[aid], t, planned)
                    reserve_mapf_path(res_table, aid, pts, park_until=min(plan_h, pts[-1][1] + 3))
                    for node_s, step_s in pts:
                        plan_steps[aid][step_s] = node_s
                    plan_end_t[aid] = pts[-1][1]
                    uncommitted[aid] = pts
                    if not is_ca and len(uncommitted) >= cfg.batch_update_K:
                        for caid, cpts in uncommitted.items():
                            field.add_agv_path(caid, cpts, horizon=plan_h)
                        uncommitted.clear()
                    continue

                if retries[aid] >= R_lim or not uncommitted:
                    _hold_or_evade_step(aid, pos[aid], t, plan_h, res_table, grid.network, plan_steps, plan_end_t)
                    continue

                rb_attempts += 1
                active_b = [j for j in uncommitted]
                if use_impact_blocker:
                    blocker_j = max(
                        active_b,
                        key=lambda j: (impact_aware_blocker_score(aid, j, hits, len(uncommitted[j]), field.slot_Phi(pos[j], t)), j),
                    )
                else:
                    blocker_j = max(active_b, key=lambda j: (int(hits.get(j, 0)), j))
                saved_j = uncommitted.pop(blocker_j)
                res_table.release_all(blocker_j)
                res_table.reserve_node(pos[blocker_j], Interval(t, min(plan_h, t + 1)), blocker_j)
                field.remove_agv(blocker_j)

                retries[aid] += 1
                p_chk, _ = planner.plan_leg(
                    pos[aid], gl, t, res_table, field, aid, leg_h, goal_park_steps=2
                )
                total_exp += int(getattr(planner, "last_expansions", 0))
                if p_chk is None:
                    rb_wasted += 1
                    reserve_mapf_path(res_table, blocker_j, saved_j, park_until=min(plan_h, saved_j[-1][1] + 3))
                    uncommitted[blocker_j] = saved_j
                    _hold_or_evade_step(aid, pos[aid], t, plan_h, res_table, grid.network, plan_steps, plan_end_t)
                else:
                    pts_a = expand_planned_to_step_positions(pos[aid], t, p_chk)
                    reserve_mapf_path(res_table, aid, pts_a, park_until=min(plan_h, pts_a[-1][1] + 3))
                    for node_s, step_s in pts_a:
                        plan_steps[aid][step_s] = node_s
                    plan_end_t[aid] = pts_a[-1][1]
                    uncommitted[aid] = pts_a
                    pq_aids.append(blocker_j)

            if not is_ca:
                for caid, cpts in uncommitted.items():
                    field.add_agv_path(caid, cpts, horizon=horizon)

        if t + 1 < horizon:
            enforce_step_mutex_and_sync(agv_ids, pos, grid.network, res_table, plan_steps, plan_end_t, t, plan_h)

    safe_ok, _ = verify_mapf_trace_safety(grid.network, traces)
    eval_f = GraphCongestionField(grid.network, CPSGraphFieldConfig(max_track_horizon=horizon + 5), t_start=0)
    ac_pts = {aid: sorted((t, n) for t, n in tr.items()) for aid, tr in traces.items()}
    eval_paths = {aid: [(n, t) for t, n in pairs] for aid, pairs in ac_pts.items()}
    for aid, pts in eval_paths.items():
        eval_f.add_agv_path(aid, pts, horizon=horizon)
    ac, _ = _eval_field_metrics(eval_f, eval_paths)

    return LifelongResult(
        algo=algo,
        map_name=grid.name,
        seed=int(seed),
        n_agvs=len(agv_ids),
        horizon=horizon,
        completed_tasks=completed,
        throughput=round(float(completed) / max(1, horizon), 4),
        mean_waiting_time=round(float(wait_sum) / max(1, completed), 2),
        mean_queue_len=round(float(q_len_sum) / max(1, horizon), 2),
        avg_congestion=round(ac, 4),
        total_expansions=total_exp,
        rollback_attempts=rb_attempts,
        rollback_wasted=rb_wasted,
        zero_conflicts=safe_ok,
    )
