"""Commitment-Preserving Post-Pickup Stalled-Corridor & Bay Repair for AMR-Scheduler.

When one or more AGVs are physically loaded (`TaskStatus.PICKED`, irrevocable
vehicle-cargo binding) and continuously stalled along single-lane bottleneck bridges,
cross-rungs, or congested destination bays, greedy shortest-path waiting loops
(`aisle_hold_steps=6`) repeatedly steer opposing loaded vehicles back into nose-to-nose
livelock until `max_steps`.

Provides two strictly causal, commitment-preserving repair modes evaluated alongside the
unmodified baseline (`"none"`):
  - `"solo_reroute"`: Penalizes stalled outgoing choke segments in Space-Time A* and
    re-routes only the stalled PICKED vehicle(s) individually while locking `(agv_id, task_id)`.
  - `"coordinated_neighbor_repair"`: Repairs the stalled PICKED cluster jointly with
    blocking idle/queued/oncoming neighbors — clearing parked idle blockers, arbitrating
    head-on 1-lane corridor priority (yielder retreats/bypasses onto a `cap >= 2` junction
    first, leader advances cleanly), and projecting non-blocking detour guides within the
    shared transaction budget.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from amr_sched.cvcr.snapshot import NativeSnapshot, op_predecessors
from amr_sched.entities import TaskStatus
from amr_sched.network import Network, NodeKind
from amr_sched.reservation import Interval, ReservationTable
from amr_sched.rust_reservation import RustReservationTable


def _clone_res_from(src: ReservationTable | RustReservationTable) -> ReservationTable | RustReservationTable:
    """Create a shallow copy of an existing reservation table (native Rust clone when available)."""
    if isinstance(src, RustReservationTable):
        return src.clone_fast()
    dst = ReservationTable()
    dst._node = {k: list(v) for k, v in getattr(src, "_node", {}).items()}
    dst._edge = {k: list(v) for k, v in getattr(src, "_edge", {}).items()}
    dst._pinned_nodes = {k: set(v) for k, v in getattr(src, "_pinned_nodes", {}).items()}
    dst._pinned_edges = {k: set(v) for k, v in getattr(src, "_pinned_edges", {}).items()}
    return dst


def _clone_res_table(simulator: Any) -> ReservationTable:
    """Create a fast isolated clone of `simulator._res` so repair planning never mutates `_res`."""
    return _clone_res_from(simulator._res)


def _release_future_on_clone(simulator: Any, res_tbl: ReservationTable, owner: str) -> None:
    """Release `owner` future reservations on `res_tbl` while keeping actual node hold/claim."""
    res_tbl.release_all(owner)
    node_hold = getattr(simulator, "_node_hold", {})
    node_claim = getattr(simulator, "_node_claim", {})
    for hold in (node_hold.get(owner), node_claim.get(owner)):
        if hold is not None:
            res_tbl.reserve_node(*hold, owner)


def _is_choke_node(network: Network, nid: str) -> bool:
    """Return True when `nid` sits inside a narrow capacity-1 passage/workstation spur."""
    if nid not in network.nodes:
        return False
    return int(network.node(nid).capacity) <= 1


def detect_stalled_picked_agvs(
    simulator: Any,
    *,
    stall_threshold: int = 8,
    min_retry_cooldown: int = 24,
    include_edge_gate_stalls: bool = False,
) -> list[str]:
    """Identify non-faulted loaded (`PICKED`) AGVs stalled `>= stall_threshold` steps.

    When `include_edge_gate_stalls=True`, also includes loaded AGVs whose edge
    transit time has already elapsed (`step >= arr`) yet remain blocked at entry.
    """
    step = int(getattr(simulator, "_step_cursor", 0))
    faulted = set(getattr(simulator, "_faulted_agvs", set()))
    transit = getattr(simulator, "_transit", {})
    blocked_since = getattr(simulator, "_blocked_since", {})
    jobs = getattr(simulator, "_jobs", {})
    tmap = getattr(simulator, "_task_by_id", {})
    last_rep = getattr(simulator, "_last_loaded_repair_at", {})
    stalled: list[str] = []

    for agv in getattr(simulator, "agvs", []):
        aid = agv.agv_id
        if aid in faulted:
            continue
        tr = transit.get(aid)
        if tr is not None:
            if not include_edge_gate_stalls or step < int(tr[3]):
                continue
        cur_nd = agv.cur_node or agv.start_node
        prev_rep = last_rep.get(aid)
        if prev_rep is not None and prev_rep[1] == cur_nd and (step - int(prev_rep[0])) < int(min_retry_cooldown):
            continue
        job = jobs.get(aid)
        if job is None or not job.task_id:
            continue
        t = tmap.get(job.task_id)
        if t is None or t.status is not TaskStatus.PICKED:
            continue
        since = blocked_since.get(aid)
        if since is not None and (step - int(since)) >= int(stall_threshold):
            stalled.append(aid)
    return sorted(stalled)


def _find_safe_siding_target(
    simulator: Any,
    agv_id: str,
    here: str,
    avoid_nodes: set[str],
    exclude_next: Optional[str],
) -> Optional[str]:
    """Find nearest free `cap >= 2` (or buffer/branch) node off `avoid_nodes` for retreating."""
    network: Network = simulator.network
    transit = getattr(simulator, "_transit", {})
    faulted = getattr(simulator, "_faulted_agvs", set())
    occ_cnt: dict[str, int] = {}
    for other in getattr(simulator, "agvs", []):
        if other.agv_id == agv_id or other.agv_id in faulted:
            continue
        tr = transit.get(other.agv_id)
        n = tr[1] if tr else other.cur_node
        if n:
            occ_cnt[n] = occ_cnt.get(n, 0) + 1

    try:
        dists = network.shortest_travel_times(here, 3.0)
    except Exception:
        return None

    cands: list[tuple[int, float, str]] = []
    for nid, nobj in network.nodes.items():
        if nid == here or nid == exclude_next:
            continue
        if occ_cnt.get(nid, 0) >= int(nobj.capacity):
            continue
        d = float(dists.get(nid, float("inf")))
        if not math.isfinite(d):
            continue
        in_avoid = 1 if nid in avoid_nodes else 0
        # Prefer junctions with cap >= 2 outside the stalled corridor and shorter retreat distance
        if nobj.capacity >= 2 or nobj.kind in (NodeKind.BUFFER, NodeKind.AISLE):
            tier = (in_avoid, 0 if nobj.capacity >= 2 else 1)
            cands.append((tier[0] * 10 + tier[1], d, nid))
    if not cands:
        return None
    cands.sort(key=lambda x: (x[0], x[1], x[2]))
    return cands[0][2]


def _extract_blocked_corridor_penalties(
    simulator: Any,
    stalled_agvs: list[str],
    penalty_q3: int = 48,
) -> dict[tuple[str, str], int]:
    """Build directed Q3 edge penalties around currently jammed `cap=1` edges for detour planning."""
    network: Network = simulator.network
    jobs = getattr(simulator, "_jobs", {})
    agv_by_id = {a.agv_id: a for a in getattr(simulator, "agvs", [])}
    penalties: dict[tuple[str, str], int] = dict(getattr(simulator, "_traffic_weights_q3", {}))

    for aid in stalled_agvs:
        agv = agv_by_id.get(aid)
        job = jobs.get(aid)
        if agv is None or job is None:
            continue
        here = agv.cur_node or agv.start_node
        next_n = job.hops[job.idx].dst if job.idx < len(job.hops) else None
        if here and next_n and network.has_edge(here, next_n):
            penalties[(here, next_n)] = max(penalties.get((here, next_n), 0), penalty_q3)
            if network.has_edge(next_n, here):
                penalties[(next_n, here)] = max(penalties.get((next_n, here), 0), penalty_q3)
            # Extend along consecutive cap=1 single-lane spine neighbors so planner discovers bypass
            frontier = [next_n]
            visited = {here, next_n}
            for _ in range(4):
                nxt_f: list[str] = []
                for u in frontier:
                    for e in network.out_edges(u):
                        v = e.dst
                        if v in visited:
                            continue
                        if int(e.capacity) <= 1 or _is_choke_node(network, u) or _is_choke_node(network, v):
                            penalties[(u, v)] = max(penalties.get((u, v), 0), penalty_q3)
                            if network.has_edge(v, u):
                                penalties[(v, u)] = max(penalties.get((v, u), 0), penalty_q3)
                            visited.add(v)
                            nxt_f.append(v)
                frontier = nxt_f
                if not frontier:
                    break
    return penalties


def _plan_on_sim_with_weights(
    simulator: Any,
    agv: Any,
    src: str,
    goal: str,
    start_step: int,
    weights_q3: dict[tuple[str, str], int],
    *,
    res_table: Optional[ReservationTable] = None,
    extra_blocked_edges: Optional[list[tuple[str, str, int, int]]] = None,
) -> Optional[list[tuple[str, str, int, int]]]:
    """Plan conflict-free Space-Time A* hop list `(src, dst, dep, arr)` on an isolated `ReservationTable`."""
    if src == goal:
        return []
    routing_net = simulator._routing_network()
    planner = simulator.planner
    if hasattr(planner, "sync_traffic_weights"):
        planner.sync_traffic_weights(routing_net, weights_q3, cutoff_step=start_step + 90)
    if hasattr(planner, "clear_agv_guides"):
        planner.clear_agv_guides()

    work_res = res_table if res_table is not None else _clone_res_table(simulator)
    # Prevent Space-Time A* from assuming a stalled/faulted stationary AGV clears in node_hold_steps=6 ticks
    blocked_since = getattr(simulator, "_blocked_since", {})
    faulted_now = set(getattr(simulator, "_faulted_agvs", set()))
    transit_now = getattr(simulator, "_transit", {})
    use_batch = isinstance(work_res, RustReservationTable)
    batch_lines: list[str] = []
    for other in getattr(simulator, "agvs", []):
        oid = other.agv_id
        if oid == agv.agv_id or oid in transit_now:
            continue
        ond = other.cur_node or other.start_node
        if not ond or ond == src or ond == goal:
            continue
        is_stuck_other = (oid in faulted_now) or (
            oid in blocked_since and (start_step - int(blocked_since[oid])) >= 4
        )
        if is_stuck_other:
            if use_batch:
                batch_lines.append(f"N\t{ond}\t{start_step}\t{start_step + 32}")
            else:
                work_res.reserve_node(ond, Interval(start_step, start_step + 32), "__REPAIR_PEER__")

    if extra_blocked_edges:
        for (u, v, dep, arr) in extra_blocked_edges:
            if arr > dep and simulator.network.has_edge(u, v):
                dep_i, arr_i = int(dep), int(arr)
                if use_batch:
                    batch_lines.append(f"E\t{u}\t{v}\t{dep_i}\t{arr_i}")
                    if simulator.network.has_edge(v, u):
                        batch_lines.append(f"E\t{v}\t{u}\t{dep_i}\t{arr_i}")
                    batch_lines.append(f"N\t{v}\t{arr_i}\t{arr_i + 2}")
                else:
                    iv = Interval(dep_i, arr_i)
                    work_res.reserve_edge(u, v, iv, "__REPAIR_PEER__")
                    if simulator.network.has_edge(v, u):
                        work_res.reserve_edge(v, u, iv, "__REPAIR_PEER__")
                    work_res.reserve_node(v, Interval(arr_i, arr_i + 2), "__REPAIR_PEER__")
    if use_batch and batch_lines:
        work_res.reserve_batch_script("__REPAIR_PEER__", "\n".join(batch_lines))

    horizon = min(start_step + simulator.config.planning_horizon, simulator.config.max_steps)
    path = planner.plan(
        network=routing_net,
        start=src,
        goal=goal,
        start_step=start_step,
        reservations=work_res,
        agv_id=agv.agv_id,
        horizon=horizon,
    )
    if path is None or not path.nodes or (path.arrival_steps and path.arrival_steps[-1] >= simulator.config.max_steps):
        return None
    hops: list[tuple[str, str, int, int]] = []
    cur = src
    t = start_step
    for node, arrive in zip(path.nodes, path.arrival_steps):
        if node == cur:
            continue
        hops.append((cur, node, int(t), int(arrive)))
        cur = node
        t = int(arrive)
    return hops


def plan_loaded_corridor_repair(
    simulator: Any,
    snapshot: NativeSnapshot,
    mode: str = "coordinated_neighbor_repair",
    *,
    stall_threshold: int = 8,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Compute legal, commitment-preserving local repair actions for stalled `PICKED` AGVs.

    Returns `(repairs_by_agv, stats_dict)` where `repairs_by_agv` may be attached directly to
    `CandidateProposal.proposed_live_repairs` and committed atomically by `NativeTransactionOwner`.
    """
    stats = {
        "mode": str(mode),
        "stalled_picked_count": 0,
        "solo_reroutes": 0,
        "neighbor_yields": 0,
        "corridor_arbitrations": 0,
        "bay_unblocks": 0,
    }
    if mode in ("none", "", "off"):
        return {}, stats

    stalled = detect_stalled_picked_agvs(simulator, stall_threshold=stall_threshold)
    stats["stalled_picked_count"] = len(stalled)
    if not stalled:
        return {}, stats

    step = int(getattr(simulator, "_step_cursor", 0))
    network: Network = simulator.network
    agv_by_id = {a.agv_id: a for a in getattr(simulator, "agvs", [])}
    jobs = getattr(simulator, "_jobs", {})
    tmap = getattr(simulator, "_task_by_id", {})
    faulted = set(getattr(simulator, "_faulted_agvs", set()))
    transit = getattr(simulator, "_transit", {})

    base_weights = _extract_blocked_corridor_penalties(simulator, stalled, penalty_q3=48)

    if mode == "solo_reroute":
        repairs: dict[str, dict[str, Any]] = {}
        for aid in stalled:
            agv = agv_by_id[aid]
            job = jobs[aid]
            t = tmap[job.task_id]
            here = agv.cur_node or agv.start_node
            old_next = job.hops[job.idx].dst if job.idx < len(job.hops) else None
            # Only penalize the currently failing immediate hop + choke cluster while keeping alternatives open
            w_local = dict(base_weights)
            if old_next and network.has_edge(here, old_next):
                w_local[(here, old_next)] = 200
            # Try replanning on an isolated copy of _res so simulator state & digest stay immutable
            tmp_res = _clone_res_table(simulator)
            _release_future_on_clone(simulator, tmp_res, aid)
            hops = _plan_on_sim_with_weights(simulator, agv, here, t.dropoff, step, w_local, res_table=tmp_res)
            if hops:
                repairs[aid] = {
                    "kind": "picked_reroute",
                    "task_id": t.task_id,
                    "goal": t.dropoff,
                    "hops": hops,
                    "traffic_weights": w_local,
                    "traffic_cutoff": step + 60,
                }
                stats["solo_reroutes"] += 1
        return repairs, stats

    # mode == "coordinated_neighbor_repair":
    # Identify all blocking neighbors (on stalled AGVs' outgoing edges, choke corridors, or destination bays)
    occ_node_to_agv: dict[str, list[str]] = {}
    for other in getattr(simulator, "agvs", []):
        if other.agv_id in faulted or other.agv_id in transit:
            continue
        nd = other.cur_node or other.start_node
        if nd:
            occ_node_to_agv.setdefault(nd, []).append(other.agv_id)

    op_due_by_tid: dict[str, int] = {}
    urg_by_tid: dict[str, bool] = {}
    for op in snapshot.operations:
        for ktid in op.kit_task_ids:
            op_due_by_tid[ktid] = int(op.due_step)
            urg_by_tid[ktid] = bool(op.is_urgent)

    repairs = {}
    avoid_corridor: set[str] = set()
    blockers_to_yield: dict[str, str] = {}  # nbr_aid -> avoid_next_target

    for aid in stalled:
        agv = agv_by_id[aid]
        job = jobs[aid]
        t = tmap[job.task_id]
        here = agv.cur_node or agv.start_node
        avoid_corridor.add(here)
        next_n = job.hops[job.idx].dst if job.idx < len(job.hops) else None
        if next_n:
            avoid_corridor.add(next_n)
            for nbr_id in occ_node_to_agv.get(next_n, ()):
                if nbr_id != aid:
                    blockers_to_yield[nbr_id] = here
        # Check destination bay blocker
        for nbr_id in occ_node_to_agv.get(t.dropoff, ()):
            if nbr_id != aid:
                nbr_job = jobs.get(nbr_id)
                nbr_t = tmap.get(nbr_job.task_id) if (nbr_job and nbr_job.task_id) else None
                if nbr_t is None or nbr_t.status is not TaskStatus.PICKED:
                    blockers_to_yield[nbr_id] = t.dropoff
                    stats["bay_unblocks"] += 1

    # Also locate opposing vehicles inside cap=1 neighbors adjacent to stalled vehicles
    cluster_picked: list[str] = list(stalled)
    for nbr_id in list(blockers_to_yield.keys()):
        nbr_job = jobs.get(nbr_id)
        nbr_t = tmap.get(nbr_job.task_id) if (nbr_job and nbr_job.task_id) else None
        if nbr_t is not None and nbr_t.status is TaskStatus.PICKED and nbr_id not in cluster_picked:
            cluster_picked.append(nbr_id)

    def _picked_leader_key(rid: str) -> tuple[int, int, int, str]:
        agv = agv_by_id[rid]
        job = jobs[rid]
        t = tmap[job.task_id]
        here = agv.cur_node or agv.start_node
        # Prefer vehicles ALREADY deep inside cap=1 bridge or holding urgent cargo as leader,
        # while a vehicle standing at a multi-lane junction can easily yield/take the bypass!
        has_bypass_or_multilane = sum(
            1 for e in network.out_edges(here) if int(network.node(e.dst).capacity) >= 2
        )
        inside_choke_bonus = 0 if int(network.node(here).capacity) <= 1 and has_bypass_or_multilane <= 1 else 1
        is_urg = 0 if (t.is_urgent or urg_by_tid.get(t.task_id, False)) else 1
        due = int(op_due_by_tid.get(t.task_id, t.due_step or 999))
        return (is_urg, inside_choke_bonus, due, rid)

    # Arbitrate order: yielders/bypassers move or evacuate FIRST, clearing the path for leader
    ordered_picked = sorted(cluster_picked, key=_picked_leader_key)
    staged_edges: list[tuple[str, str, int, int]] = []
    coord_res = _clone_res_table(simulator)

    # 1. Evacuate idle / non-picked blocking neighbors first
    for nbr_id, from_avoid in blockers_to_yield.items():
        if nbr_id in cluster_picked:
            continue
        nbr_agv = agv_by_id.get(nbr_id)
        if nbr_agv is None or nbr_id in faulted or nbr_id in transit:
            continue
        nbr_job = jobs.get(nbr_id)
        nbr_t = tmap.get(nbr_job.task_id) if (nbr_job and nbr_job.task_id) else None
        if nbr_t is not None and nbr_t.status is TaskStatus.PICKED:
            continue
        if nbr_job is not None and nbr_job.task_id is not None:
            continue
        siding = _find_safe_siding_target(simulator, nbr_id, nbr_agv.cur_node or nbr_agv.start_node, avoid_corridor, from_avoid)
        if siding:
            trial_res = _clone_res_from(coord_res)
            _release_future_on_clone(simulator, trial_res, nbr_id)
            hops = _plan_on_sim_with_weights(
                simulator, nbr_agv, nbr_agv.cur_node or nbr_agv.start_node, siding, step, base_weights,
                res_table=trial_res, extra_blocked_edges=staged_edges,
            )
            if hops:
                _release_future_on_clone(simulator, coord_res, nbr_id)
                repairs[nbr_id] = {
                    "kind": "idle_park",
                    "task_id": None,
                    "goal": siding,
                    "hops": hops,
                    "traffic_weights": base_weights,
                    "traffic_cutoff": step + 45,
                }
                for u, v, dep, arr in hops[:4]:
                    staged_edges.append((u, v, dep, arr))
                stats["neighbor_yields"] += 1

    # 2. Plan picked yielders (reverse order = junction/bypass side first!) taking detour off choke
    if len(ordered_picked) >= 2:
        stats["corridor_arbitrations"] += 1
    yielders_first = list(reversed(ordered_picked[1:]))
    leaders_last = ordered_picked[:1]

    for aid in yielders_first + leaders_last:
        _release_future_on_clone(simulator, coord_res, aid)

    for aid in yielders_first:
        agv = agv_by_id[aid]
        job = jobs[aid]
        t = tmap[job.task_id]
        here = agv.cur_node or agv.start_node
        w_detour = dict(base_weights)
        for av in avoid_corridor:
            for e in network.out_edges(av):
                w_detour[(e.src, e.dst)] = 64
                if network.has_edge(e.dst, e.src):
                    w_detour[(e.dst, e.src)] = 64
        hops = _plan_on_sim_with_weights(
            simulator, agv, here, t.dropoff, step, w_detour, res_table=_clone_res_from(coord_res), extra_blocked_edges=staged_edges
        )
        if not hops:
            siding = _find_safe_siding_target(simulator, aid, here, avoid_corridor, None)
            if siding and siding != here:
                hops = _plan_on_sim_with_weights(
                    simulator, agv, here, siding, step, w_detour, res_table=_clone_res_from(coord_res), extra_blocked_edges=staged_edges
                )
                if hops:
                    arr_siding = hops[-1][3]
                    tail = _plan_on_sim_with_weights(
                        simulator, agv, siding, t.dropoff, arr_siding + 4, w_detour,
                        res_table=_clone_res_from(coord_res), extra_blocked_edges=staged_edges,
                    )
                    hops = (hops + tail) if tail else None
        if not hops:
            hops = _plan_on_sim_with_weights(
                simulator, agv, here, t.dropoff, step, base_weights, res_table=_clone_res_from(coord_res), extra_blocked_edges=staged_edges
            )
        if hops:
            repairs[aid] = {
                "kind": "picked_reroute",
                "task_id": t.task_id,
                "goal": t.dropoff,
                "hops": hops,
                "traffic_weights": base_weights,
                "traffic_cutoff": step + 50,
            }
            for u, v, dep, arr in hops[:6]:
                staged_edges.append((u, v, dep, arr))
            stats["neighbor_yields"] += 1

    # 3. Route leader along cleared corridor / shortest unblocked path
    for aid in leaders_last:
        agv = agv_by_id[aid]
        job = jobs[aid]
        t = tmap[job.task_id]
        here = agv.cur_node or agv.start_node
        old_next = job.hops[job.idx].dst if job.idx < len(job.hops) else None
        if len(ordered_picked) >= 2 and yielders_first:
            w_leader = dict(getattr(simulator, "_traffic_weights_q3", {}))
        else:
            w_leader = dict(base_weights)
            if old_next and network.has_edge(here, old_next):
                w_leader[(here, old_next)] = 200
        hops = _plan_on_sim_with_weights(
            simulator, agv, here, t.dropoff, step, w_leader, res_table=_clone_res_from(coord_res), extra_blocked_edges=staged_edges
        )
        if not hops:
            hops = _plan_on_sim_with_weights(
                simulator, agv, here, t.dropoff, step, base_weights, res_table=_clone_res_from(coord_res), extra_blocked_edges=staged_edges
            )
        if hops:
            repairs[aid] = {
                "kind": "picked_reroute",
                "task_id": t.task_id,
                "goal": t.dropoff,
                "hops": hops,
                "traffic_weights": w_leader,
                "traffic_cutoff": step + 50,
            }
            stats["solo_reroutes"] += 1

    return repairs, stats
