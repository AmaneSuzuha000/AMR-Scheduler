"""Proactive Anti-Congestion, Detour-Delta, Haven-Invariant & Switchable-TPG Guard (V12).

Synthesizes three orthogonal top-venue MAPD mechanisms cleanly over standard `CandidateProposal`
contracts without altering simulator baseline physics:
  1. **RA-L 2026 Detour-Delta (`compute_detour_delta_signals`)**:
     Estimates causal per-edge congestion Q3 penalties and per-bay detour/queue wait deltas
     (`Delta = eta - d_bar`) from the active `NativeSnapshot` (+ optional live simulator queues)
     so assignment, KM matching, Regret-k, and focal CP-SAT avoid feeding congested 1-lane spurs.
  2. **IJCAI-ECAI 2026 `A#` Haven Exclusivity & AAMAS 2024 Deadlock Prevention**:
     Analyzes degree-1 / `cap=1` dead-end leaf stations (`W -> entry`), proactively evacuates
     idle blockers occupying occupied leaf bays to `cap >= 2` Havens before incoming loaded AGVs
     enter the sole spur edge, and enforces safe Haven holds during loaded corridor repair.
  3. **AAAI 2025 Switchable TPG / PPSS Low-Churn Precedence Guides**:
     Resolves corridor conflicts via conflict-free local precedence guides and bounded edge
     weights rather than churning task assignments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from amr_sched.cvcr.loaded_corridor_repair import (
    _clone_res_from,
    _clone_res_table,
    _extract_blocked_corridor_penalties,
    _find_safe_siding_target,
    _plan_on_sim_with_weights,
    _release_future_on_clone,
    detect_stalled_picked_agvs,
)
from amr_sched.cvcr.snapshot import NativeSnapshot, op_predecessors
from amr_sched.entities import TaskStatus
from amr_sched.network import Network, NodeKind
from amr_sched.reservation import Interval


@dataclass(frozen=True)
class NetworkHavenProfile:
    """Cached topological classification of dead-end leaf spurs, chokes, and safe Havens."""

    network_id: int
    leaf_spurs: dict[str, str]       # spur_node -> sole_entry_node (cap<=1 dead-end spurs)
    entry_to_spurs: dict[str, tuple[str, ...]]
    choke_nodes: frozenset[str]
    safe_havens: tuple[str, ...]      # cap>=2 junctions / buffers ordered by safety headroom


_HAVEN_CACHE: dict[int, tuple[Network, NetworkHavenProfile]] = {}


def analyze_network_havens(network: Network) -> NetworkHavenProfile:
    """Identify `cap=1` leaf spurs, choke passages, and `cap>=2` Haven staging nodes."""
    key = id(network)
    cached = _HAVEN_CACHE.get(key)
    if cached is not None and cached[0] is network:
        return cached[1]
    if len(_HAVEN_CACHE) >= 16:
        _HAVEN_CACHE.clear()

    undir_nbrs: dict[str, set[str]] = {nid: set() for nid in network.nodes}
    for e in network.edges.values():
        if e.src in undir_nbrs:
            undir_nbrs[e.src].add(e.dst)
        if e.dst in undir_nbrs:
            undir_nbrs[e.dst].add(e.src)

    leaf_spurs: dict[str, str] = {}
    entry_map: dict[str, list[str]] = {}
    choke_nodes: set[str] = set()
    haven_ranks: list[tuple[int, int, str]] = []

    for nid, nobj in network.nodes.items():
        nbrs = sorted(undir_nbrs.get(nid, set()))
        if int(nobj.capacity) <= 1:
            choke_nodes.add(nid)
            if len(nbrs) == 1:
                entry = nbrs[0]
                leaf_spurs[nid] = entry
                entry_map.setdefault(entry, []).append(nid)
        else:
            is_buf = 0 if nobj.kind == NodeKind.BUFFER else (1 if nobj.kind == NodeKind.AISLE else 2)
            haven_ranks.append((is_buf, -int(nobj.capacity), nid))

    haven_ranks.sort()
    prof = NetworkHavenProfile(
        network_id=key,
        leaf_spurs=leaf_spurs,
        entry_to_spurs={k: tuple(sorted(v)) for k, v in entry_map.items()},
        choke_nodes=frozenset(choke_nodes),
        safe_havens=tuple(x[2] for x in haven_ranks),
    )
    _HAVEN_CACHE[key] = (network, prof)
    return prof


def compute_detour_delta_signals(
    network: Network,
    snapshot: NativeSnapshot,
    simulator: Any = None,
    *,
    stall_threshold: int = 8,
) -> dict[str, Any]:
    """Compute causal `Detour-Delta` edge weights, per-bay waits, and `A#` leaf-spur hazards."""
    prof = analyze_network_havens(network)
    t_now = int(snapshot.tick)

    # 1. Decay any active simulator traffic weights cleanly as starting prior
    edge_q3: dict[tuple[str, str], float] = {}
    if simulator is not None:
        for ek, wv in getattr(simulator, "_traffic_weights_q3", {}).items():
            if int(wv) > 1:
                edge_q3[(str(ek[0]), str(ek[1]))] = float(wv) * 0.45

    bay_pressure: dict[str, int] = {}
    occ_by_node: dict[str, list[str]] = {}
    rob_by_id = {r.robot_id: r for r in snapshot.robots}
    tasks_by_id = {t.task_id: t for t in snapshot.tasks}

    # Current positions, dropoff bay demand, and predicted path overlap along chokes
    dir_flow: dict[tuple[str, str], int] = {}
    for r in snapshot.robots:
        nd = r.cur_node
        occ_by_node.setdefault(nd, []).append(r.robot_id)
        if len(r.route) >= 2:
            limit = min(len(r.route) - 1, 6)
            for idx in range(limit):
                u, v = r.route[idx], r.route[idx + 1]
                if u != v and network.has_edge(u, v):
                    dir_flow[(u, v)] = dir_flow.get((u, v), 0) + 1

    # Opposing directional flow ONLY when narrow choke/capacity-1 corridor or multi-robot saturation
    for (u, v), fwd in dir_flow.items():
        rev = dir_flow.get((v, u), 0)
        e_cap = int(network.edge(u, v).capacity) if network.has_edge(u, v) else 1
        in_choke = (u in prof.choke_nodes) or (v in prof.choke_nodes) or e_cap <= 1
        if in_choke and fwd >= 1 and rev >= 1:
            pen = 14.0 + 4.0 * float(min(3, fwd + rev - 2))
            edge_q3[(u, v)] = max(edge_q3.get((u, v), 0.0), pen)
            if network.has_edge(v, u):
                edge_q3[(v, u)] = max(edge_q3.get((v, u), 0.0), pen)
        elif in_choke and fwd >= 3:
            edge_q3[(u, v)] = max(edge_q3.get((u, v), 0.0), 5.0 * float(min(4, fwd - 1)))

    # 2. Identify occupied `cap=1` leaf spurs & compute per-bay Detour-Delta (`Delta = eta - d_bar`)
    leaf_conflicts: list[dict[str, Any]] = []
    target_counts: dict[str, list[str]] = {}
    for t in snapshot.tasks:
        if t.status.lower() == "completed":
            continue
        bay_pressure[t.dropoff] = bay_pressure.get(t.dropoff, 0) + 1
        if t.assigned_agv and t.status.lower() in ("assigned", "loading", "picked", "to_dropoff"):
            target_counts.setdefault(t.dropoff, []).append(t.assigned_agv)
        if t.status.lower() in ("assigned", "loading") and t.pickup != t.dropoff:
            bay_pressure[t.pickup] = bay_pressure.get(t.pickup, 0) + 1

    for spur, entry in prof.leaf_spurs.items():
        occupants = occ_by_node.get(spur, [])
        incoming = [aid for aid in target_counts.get(spur, ()) if aid not in occupants]
        if occupants and incoming:
            # Penalize entering dead-end spur until occupant evacuates or completes
            occ_idle = [
                aid
                for aid in occupants
                if (rob_by_id.get(aid) and rob_by_id[aid].cur_task is None and rob_by_id[aid].state.lower() != "faulted")
            ]
            w_spur = 42.0 if occ_idle else 28.0
            if network.has_edge(entry, spur):
                edge_q3[(entry, spur)] = max(edge_q3.get((entry, spur), 0.0), w_spur)
            leaf_conflicts.append(
                {
                    "spur": spur,
                    "entry": entry,
                    "occupants": list(occupants),
                    "idle_occupants": occ_idle,
                    "incoming_agvs": list(incoming),
                }
            )

    # 3. Live blocked robots & elapsed-transit edge-gate stalls (catches mid-edge 7102 traps too)
    blocked_cnt = 0
    edge_gate_stalled: list[str] = []
    if simulator is not None:
        blocked_since = getattr(simulator, "_blocked_since", {})
        transit = getattr(simulator, "_transit", {})
        jobs = getattr(simulator, "_jobs", {})
        faulted = set(getattr(simulator, "_faulted_agvs", set()))
        edge_gate_stalled = detect_stalled_picked_agvs(
            simulator,
            stall_threshold=stall_threshold,
            include_edge_gate_stalls=True,
        )
        for aid, bs in blocked_since.items():
            wait_len = max(0, int(t_now) - int(bs))
            if wait_len < 3:
                continue
            blocked_cnt += 1
            r_obj = rob_by_id.get(aid)
            tr = transit.get(aid)
            if tr is not None:
                u, v = str(tr[0]), str(tr[1])
                if network.has_edge(u, v):
                    edge_q3[(u, v)] = max(edge_q3.get((u, v), 0.0), min(48.0, 12.0 + 2.5 * wait_len))
                    if network.has_edge(v, u):
                        edge_q3[(v, u)] = max(edge_q3.get((v, u), 0.0), min(48.0, 12.0 + 2.5 * wait_len))
            elif r_obj is not None:
                job = jobs.get(aid)
                nxt = job.hops[job.idx].dst if (job and job.idx < len(job.hops)) else None
                if nxt and network.has_edge(r_obj.cur_node, nxt):
                    edge_q3[(r_obj.cur_node, nxt)] = max(
                        edge_q3.get((r_obj.cur_node, nxt), 0.0),
                        min(48.0, 10.0 + 2.0 * wait_len),
                    )
            if aid in faulted and r_obj is not None:
                for e in network.out_edges(r_obj.cur_node):
                    edge_q3[(e.src, e.dst)] = max(edge_q3.get((e.src, e.dst), 0.0), 32.0)

    bay_detour_delta: dict[str, float] = {}
    for nid, nobj in network.nodes.items():
        if nobj.kind != NodeKind.WORKSTATION and nid not in prof.leaf_spurs:
            continue
        delta = 0.0
        q_cnt = bay_pressure.get(nid, 0)
        is_choke = int(nobj.capacity) <= 1 or nid in prof.leaf_spurs
        # Only add true Detour-Delta delay when a cap=1 bottleneck/leaf spur is actually contended or occupied
        if is_choke:
            if q_cnt >= 2:
                delta += 4.5 * float(min(5, q_cnt - 1))
            if occ_by_node.get(nid, ()):
                delta += 8.0
            if nid in prof.leaf_spurs:
                entry = prof.leaf_spurs[nid]
                if len(occ_by_node.get(entry, ())) >= int(network.node(entry).capacity):
                    delta += 6.0
        elif q_cnt >= 4:
            # Extreme multi-robot queue saturation on cap>=2 open bays only
            delta += 3.0 * float(min(4, q_cnt - 3))
        if delta > 0.0:
            bay_detour_delta[nid] = float(min(45.0, delta))

    weights_q3: dict[tuple[str, str], int] = {}
    for k, val in edge_q3.items():
        iv = max(0, min(48, int(round(val))))
        if iv >= 3:
            weights_q3[k] = iv

    deltas = list(bay_detour_delta.values())
    n_agv = max(1, len(snapshot.robots))
    return {
        "weights_q3": weights_q3,
        "bay_detour_delta": bay_detour_delta,
        "leaf_conflicts": leaf_conflicts,
        "edge_gate_stalled": edge_gate_stalled,
        "summary": {
            "edge_gate_stall_ratio": float(min(1.0, len(edge_gate_stalled) / float(n_agv))),
            "leaf_spur_contention": float(min(1.0, len(leaf_conflicts) / float(max(1, len(prof.leaf_spurs) or 4)))),
            "detour_delta_mean": float(min(1.0, (sum(deltas) / len(deltas)) / 25.0)) if deltas else 0.0,
            "detour_delta_p90": float(min(1.0, max(deltas) / 35.0)) if deltas else 0.0,
            "blocked_ratio": float(min(1.0, blocked_cnt / float(n_agv))),
        },
    }


def compute_kitting_marginal_start_value(
    snapshot: NativeSnapshot,
    simulator: Any = None,
) -> dict[str, tuple[int, int, int, int, int, str]]:
    """Rank tasks by marginal production start value (`unlock_tier`, urgency, DAG fanout, rem_kit, due_eff).

    Lower tuple sorts higher priority: tier 0 indicates the final missing kit item for an
    unstarted operation whose predecessors are complete and whose bay/crew are ready.
    """
    t_now = int(snapshot.tick)
    op_by_id = {op.op_id: op for op in snapshot.operations}
    tasks_by_id = {t.task_id: t for t in snapshot.tasks}

    succ_cnt: dict[str, int] = {}
    busy_bays: set[str] = set()
    busy_crews: set[int] = set()
    for op in snapshot.operations:
        for pid in op_predecessors(op):
            succ_cnt[pid] = succ_cnt.get(pid, 0) + 1
        if op.started and (op.end_step is None or int(op.end_step) > t_now + 6):
            busy_bays.add(op.bay_id)
            busy_crews.add(int(op.crew_id))

    out: dict[str, tuple[int, int, int, int, int, str]] = {}
    for t in snapshot.tasks:
        op = op_by_id.get(t.kit_op_id or "")
        urg_tier = 0 if (t.is_urgent or (op is not None and op.is_urgent)) else 1
        if op is None or op.started:
            out[t.task_id] = (2, urg_tier, 0, 9, int(t.due_step or 999), t.task_id)
            continue

        preds_done = True
        for pid in op_predecessors(op):
            p_op = op_by_id.get(pid)
            if p_op is not None and not (p_op.started and p_op.end_step is not None and int(p_op.end_step) <= t_now):
                preds_done = False
                break

        rem_kit = 0
        for ktid in op.kit_task_ids:
            ks = tasks_by_id.get(ktid)
            if ks is None or ks.status.lower() != "completed":
                rem_kit += 1

        bay_free = (op.bay_id not in busy_bays) or (len(op.eligible_bays) > 1)
        crew_free = (int(op.crew_id) not in busy_crews) or (len(op.eligible_crews) > 1)
        res_ready = bay_free and crew_free

        if preds_done and rem_kit <= 1 and res_ready:
            unlock_tier = 0
        elif preds_done and rem_kit <= 1:
            unlock_tier = 1
        elif preds_done and rem_kit == 2 and res_ready:
            unlock_tier = 2
        else:
            unlock_tier = 3

        due_eff = int(op.due_step) - int(op.proc_steps)
        out[t.task_id] = (unlock_tier, urg_tier, -succ_cnt.get(op.op_id, 0), rem_kit, due_eff, t.task_id)
    return out


def plan_haven_safe_corridor_repair(
    simulator: Any,
    snapshot: NativeSnapshot,
    *,
    stall_threshold: int = 8,
    include_proactive_haven: bool = True,
    include_edge_gate_stalls: bool = True,
    allow_precedence_swap: bool = True,
    use_kitting_value: bool = True,
    enable_choke_oscillation_guard: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build Haven-invariant `proposed_live_repairs` + proactive dead-end bay evacuations.

    Key guarantees over v11:
      1. Proactively evacuates idle occupants trapped inside `cap=1` dead-end leaf stations
         before an approaching loaded vehicle commits onto the sole entry spur.
      2. Detects elapsed-transit gate stalls (`include_edge_gate_stalls=True`) while only
         rewriting routes for non-mid-transit vehicles (`aid not in simulator._transit`) so
         atomic transaction validation stays strictly physically valid.
      3. Arbitrates contending `PICKED` corridor pairs via BTPG/SES switchable precedence
         ordered by kitting marginal operation-start value (`unlock_tier`).
    """
    sig = compute_detour_delta_signals(
        simulator.network,
        snapshot,
        simulator=simulator,
        stall_threshold=stall_threshold,
    )
    prof = analyze_network_havens(simulator.network)
    step = int(getattr(simulator, "_step_cursor", 0))
    network: Network = simulator.network
    agv_by_id = {a.agv_id: a for a in getattr(simulator, "agvs", [])}
    jobs = getattr(simulator, "_jobs", {})
    tmap = getattr(simulator, "_task_by_id", {})
    faulted = set(getattr(simulator, "_faulted_agvs", set()))
    transit = getattr(simulator, "_transit", {})
    loading_active = {
        aid for aid, end_t in getattr(simulator, "_load_end", {}).items() if int(end_t) > step
    }

    stats = {
        "mode": "haven_v12",
        "stalled_picked_count": 0,
        "edge_gate_stalled_count": len(sig["edge_gate_stalled"]),
        "proactive_haven_evacuations": 0,
        "solo_reroutes": 0,
        "neighbor_yields": 0,
        "corridor_arbitrations": 0,
        "bay_unblocks": 0,
    }

    repairs: dict[str, dict[str, Any]] = {}
    staged_edges: list[tuple[str, str, int, int]] = []
    coord_res = _clone_res_table(simulator)
    base_weights = dict(sig["weights_q3"])
    avoid_corridor: set[str] = set()

    fb = getattr(snapshot, "execution_feedback", None)
    fb_blocked: dict[tuple[str, str], int] = {}
    if fb is not None and hasattr(fb, "blocked_intervals"):
        fb_blocked = fb.blocked_intervals(t_commit=step, default_horizon=12)
    stats["fb_blocked_edges"] = len(fb_blocked)
    if fb_blocked:
        fb_pen = fb.corridor_q3_penalties() if hasattr(fb, "corridor_q3_penalties") else {}
        for (u_fb, v_fb), en_fb in fb_blocked.items():
            s_iv = step
            e_iv = max(s_iv + 1, int(en_fb))
            coord_res.reserve_edge(u_fb, v_fb, Interval(s_iv, e_iv), "__L1_OBSTACLE__")
            base_weights[(u_fb, v_fb)] = max(base_weights.get((u_fb, v_fb), 0), int(fb_pen.get((u_fb, v_fb), 64)))
        for ek_fb, wv_fb in fb_pen.items():
            base_weights[ek_fb] = max(base_weights.get(ek_fb, 0), int(wv_fb))

    # 1. Proactive A# Haven evacuation: free occupied cap=1 leaf spurs BEFORE loaded AGV enters
    if include_proactive_haven and sig["leaf_conflicts"]:
        for conf in sig["leaf_conflicts"]:
            spur = conf["spur"]
            entry = conf["entry"]
            avoid_corridor.add(spur)
            # Only evacuate idle occupant if the spur entrance path isn't already mid-edge blocked
            in_tr_conflict = any(
                (tr[0] == entry and tr[1] == spur) or (tr[0] == spur and tr[1] == entry)
                for tr in transit.values()
            )
            if in_tr_conflict:
                continue
            for occ_id in conf["idle_occupants"]:
                if occ_id in repairs or occ_id in transit or occ_id in faulted:
                    continue
                occ_job = jobs.get(occ_id)
                if occ_job is not None and getattr(occ_job, "task_id", None) not in (None, ""):
                    continue
                occ_agv = agv_by_id.get(occ_id)
                if occ_agv is None or occ_agv.cur_task is not None:
                    continue
                if getattr(occ_agv.state, "value", str(occ_agv.state)).lower() not in ("idle", "parking"):
                    continue
                if getattr(simulator, "_pinned_task_queues", {}).get(occ_id):
                    continue
                # Avoid evacuating onto another occupied/targeted leaf spur
                avoid_spurs = set(avoid_corridor) | set(prof.leaf_spurs.keys())
                siding = _find_safe_siding_target(simulator, occ_id, spur, avoid_spurs, None)
                if siding and siding != spur:
                    trial_res = _clone_res_from(coord_res)
                    _release_future_on_clone(simulator, trial_res, occ_id)
                    hops = _plan_on_sim_with_weights(
                        simulator, occ_agv, spur, siding, step, base_weights,
                        res_table=trial_res, extra_blocked_edges=staged_edges,
                    )
                    if hops:
                        _release_future_on_clone(simulator, coord_res, occ_id)
                        repairs[occ_id] = {
                            "kind": "idle_park",
                            "task_id": None,
                            "goal": siding,
                            "hops": hops,
                            "traffic_weights": base_weights,
                            "traffic_cutoff": step + 35,
                        }
                        for u, v, dep, arr in hops[:4]:
                            staged_edges.append((u, v, dep, arr))
                            coord_res.reserve_edge(u, v, Interval(dep, arr), "__REPAIR_PEER__")
                        # Hold spur entry briefly while evacuating AGV exits leaf spur
                        exit_arr = int(hops[0][3]) + 1
                        coord_res.reserve_node(spur, Interval(step, exit_arr + 1), "__HAVEN_LOCK__")
                        stats["proactive_haven_evacuations"] += 1
                        stats["bay_unblocks"] += 1

        # Hold approaching loaded AGVs safely at entry junction if their leaf target was just evac'd or still occupied
        for conf in sig["leaf_conflicts"]:
            spur = conf["spur"]
            entry = conf["entry"]
            for inc_id in conf["incoming_agvs"]:
                if inc_id in repairs or inc_id in transit or inc_id in faulted:
                    continue
                inc_job = jobs.get(inc_id)
                inc_agv = agv_by_id.get(inc_id)
                if inc_job is None or not inc_job.task_id or inc_agv is None:
                    continue
                t_inc = tmap.get(inc_job.task_id)
                if t_inc is None or t_inc.status is not TaskStatus.PICKED:
                    continue
                here = inc_agv.cur_node or inc_agv.start_node
                # Only re-stage if at entry junction or approaching within spur zone without holding leaf inside
                if here == entry or (len(inc_job.hops) > inc_job.idx and any(h.dst == spur for h in inc_job.hops[inc_job.idx : inc_job.idx + 3])):
                    trial_res = _clone_res_from(coord_res)
                    _release_future_on_clone(simulator, trial_res, inc_id)
                    trial_res.reserve_node(spur, Interval(step, step + 10), "__HAVEN_LOCK__")
                    hops = _plan_on_sim_with_weights(
                        simulator, inc_agv, here, t_inc.dropoff, step, base_weights,
                        res_table=trial_res, extra_blocked_edges=staged_edges,
                    )
                    if hops:
                        _release_future_on_clone(simulator, coord_res, inc_id)
                        repairs[inc_id] = {
                            "kind": "picked_reroute",
                            "task_id": t_inc.task_id,
                            "goal": t_inc.dropoff,
                            "hops": hops,
                            "traffic_weights": base_weights,
                            "traffic_cutoff": step + 45,
                        }
                        for u, v, dep, arr in hops[:4]:
                            staged_edges.append((u, v, dep, arr))
                        stats["neighbor_yields"] += 1

    # 1b. P4 Coupled Choke-Bridge Oscillation & Multi-Hop Haven Passage Resolution
    use_choke_guard = bool(enable_choke_oscillation_guard or getattr(simulator, "_p4_choke_robust", False))
    if use_choke_guard:
        from amr_sched.cvcr.p4_belief_choke import (
            detect_choke_oscillation_agvs,
            plan_choke_passage_repairs,
        )
        osc_agvs = detect_choke_oscillation_agvs(simulator)
        if osc_agvs:
            rep_choke, st_choke = plan_choke_passage_repairs(
                simulator,
                snapshot,
                osc_agvs,
                coord_res,
                staged_edges,
                base_weights,
            )
            for k_c, v_c in rep_choke.items():
                if k_c not in repairs:
                    repairs[k_c] = v_c
            for k_s, v_s in st_choke.items():
                stats[k_s] = int(stats.get(k_s, 0)) + int(v_s)

    # 2. Stalled PICKED corridor repair (strictly excluding active mid-transit robots for physical safety)
    raw_stalled = detect_stalled_picked_agvs(
        simulator,
        stall_threshold=stall_threshold,
        include_edge_gate_stalls=include_edge_gate_stalls,
    )
    stalled = [aid for aid in raw_stalled if aid not in transit and aid not in loading_active and aid not in repairs]
    # Also include non-transit PICKED AMRs whose near-term hops intersect a live L1-blocked corridor
    if fb_blocked:
        for cand_agv in getattr(simulator, "agvs", []):
            cid = cand_agv.agv_id
            if cid in transit or cid in loading_active or cid in repairs or cid in faulted or cid in stalled:
                continue
            c_job = jobs.get(cid)
            c_task = tmap.get(c_job.task_id) if (c_job is not None and c_job.task_id) else None
            if c_task is None or c_task.status is not TaskStatus.PICKED:
                continue
            if any(
                (h.src, h.dst) in fb_blocked or (h.dst, h.src) in fb_blocked
                for h in c_job.hops[c_job.idx : c_job.idx + 4]
            ):
                stalled.append(cid)
                stats["fb_blocked_corridor_repairs"] = int(stats.get("fb_blocked_corridor_repairs", 0)) + 1

    # Also proactively detour non-transit ASSIGNED (to_pickup) AMRs approaching a blocked corridor
    if fb_blocked:
        max_fb_cutoff = max(int(en_fb) for en_fb in fb_blocked.values())
        for cand_agv in getattr(simulator, "agvs", []):
            cid = cand_agv.agv_id
            if cid in transit or cid in repairs or cid in faulted:
                continue
            c_job = jobs.get(cid)
            if c_job is None or c_job.leg != "to_pickup" or c_job.idx >= len(c_job.hops) or not c_job.task_id:
                continue
            c_task = tmap.get(c_job.task_id)
            if c_task is None or c_task.status is not TaskStatus.ASSIGNED or c_task.assigned_agv != cid:
                continue
            here_a = cand_agv.cur_node or cand_agv.start_node
            if not here_a or here_a == c_task.pickup:
                continue
            if not any(
                (h.src, h.dst) in fb_blocked or (h.dst, h.src) in fb_blocked
                for h in c_job.hops[c_job.idx : c_job.idx + 4]
            ):
                continue
            trial_res = _clone_res_from(coord_res)
            _release_future_on_clone(simulator, trial_res, cid)
            hops_a = _plan_on_sim_with_weights(
                simulator, cand_agv, here_a, c_task.pickup, step, base_weights,
                res_table=trial_res, extra_blocked_edges=staged_edges,
            )
            if hops_a and not any(
                (u, v) in fb_blocked or (v, u) in fb_blocked for (u, v, _, _) in hops_a[:3]
            ):
                _release_future_on_clone(simulator, coord_res, cid)
                repairs[cid] = {
                    "kind": "assigned_reroute",
                    "task_id": c_task.task_id,
                    "goal": c_task.pickup,
                    "hops": hops_a,
                    "traffic_weights": base_weights,
                    "traffic_cutoff": max_fb_cutoff,
                }
                for u, v, dep, arr in hops_a[:4]:
                    staged_edges.append((u, v, dep, arr))
                    coord_res.reserve_edge(u, v, Interval(dep, arr), "__REPAIR_PEER__")
                stats["fb_blocked_corridor_repairs"] = int(stats.get("fb_blocked_corridor_repairs", 0)) + 1

    stats["stalled_picked_count"] = len(raw_stalled)
    if not stalled:
        return repairs, stats

    corridor_pen = _extract_blocked_corridor_penalties(simulator, stalled, penalty_q3=48)
    for k, v in corridor_pen.items():
        base_weights[k] = max(base_weights.get(k, 0), v)

    occ_node_to_agv: dict[str, list[str]] = {}
    for other in getattr(simulator, "agvs", []):
        if other.agv_id in faulted or other.agv_id in transit:
            continue
        nd = other.cur_node or other.start_node
        if nd:
            occ_node_to_agv.setdefault(nd, []).append(other.agv_id)

    blockers_to_yield: dict[str, str] = {}
    pre_choke_peers: list[str] = []
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
                    nbr_job = jobs.get(nbr_id)
                    nbr_t = tmap.get(nbr_job.task_id) if (nbr_job and nbr_job.task_id) else None
                    if (
                        nbr_t is not None
                        and nbr_t.status is TaskStatus.PICKED
                        and nbr_job.idx < len(nbr_job.hops)
                        and nbr_job.hops[nbr_job.idx].dst == here
                        and nbr_id not in pre_choke_peers
                    ):
                        pre_choke_peers.append(nbr_id)
        for nbr_id in occ_node_to_agv.get(t.dropoff, ()):
            if nbr_id != aid:
                blockers_to_yield[nbr_id] = t.dropoff

    avoid_spurs_all = set(avoid_corridor) | set(prof.leaf_spurs.keys())
    for nbr_id, from_avoid in blockers_to_yield.items():
        if nbr_id in repairs or nbr_id in faulted or nbr_id in transit:
            continue
        nbr_job = jobs.get(nbr_id)
        if nbr_job is not None and getattr(nbr_job, "task_id", None) not in (None, ""):
            continue
        nbr_agv = agv_by_id.get(nbr_id)
        if nbr_agv is None or nbr_agv.cur_task is not None:
            continue
        if getattr(nbr_agv.state, "value", str(nbr_agv.state)).lower() not in ("idle", "parking"):
            continue
        if getattr(simulator, "_pinned_task_queues", {}).get(nbr_id):
            continue
        here_n = nbr_agv.cur_node or nbr_agv.start_node
        siding = _find_safe_siding_target(simulator, nbr_id, here_n, avoid_spurs_all, from_avoid)
        if siding and siding != here_n:
            trial_res = _clone_res_from(coord_res)
            _release_future_on_clone(simulator, trial_res, nbr_id)
            hops = _plan_on_sim_with_weights(
                simulator, nbr_agv, here_n, siding, step, base_weights,
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
                    "traffic_cutoff": step + 40,
                }
                for u, v, dep, arr in hops[:4]:
                    staged_edges.append((u, v, dep, arr))
                    coord_res.reserve_edge(u, v, Interval(dep, arr), "__REPAIR_PEER__")
                stats["neighbor_yields"] += 1
                stats["bay_unblocks"] += 1

    # 2b. BTPG/SES Switchable Precedence Arbitration over contending PICKED AGVs ordered by kitting value
    cluster_picked: list[str] = list(stalled)
    if allow_precedence_swap:
        stalled_corridors: set[str] = set()
        for s_id in stalled:
            s_agv = agv_by_id[s_id]
            s_job = jobs[s_id]
            s_t = tmap[s_job.task_id]
            stalled_corridors.add(s_agv.cur_node or s_agv.start_node)
            stalled_corridors.add(s_t.dropoff)
            for h in s_job.hops[s_job.idx : s_job.idx + 3]:
                stalled_corridors.add(h.dst)
        for nbr_id in list(blockers_to_yield.keys()) + pre_choke_peers + [a.agv_id for a in getattr(simulator, "agvs", [])]:
            if nbr_id in transit or nbr_id in faulted or nbr_id in repairs or nbr_id in cluster_picked:
                continue
            nbr_job = jobs.get(nbr_id)
            nbr_t = tmap.get(nbr_job.task_id) if (nbr_job and nbr_job.task_id) else None
            if nbr_t is not None and nbr_t.status is TaskStatus.PICKED:
                nbr_agv = agv_by_id[nbr_id]
                nbr_here = nbr_agv.cur_node or nbr_agv.start_node
                nbr_nodes = {nbr_here, nbr_t.dropoff} | {h.dst for h in nbr_job.hops[nbr_job.idx : nbr_job.idx + 3]}
                if (nbr_id in blockers_to_yield) or (nbr_id in pre_choke_peers) or bool(nbr_nodes & stalled_corridors):
                    cluster_picked.append(nbr_id)

    if allow_precedence_swap and len(cluster_picked) >= 2:
        kit_vals = compute_kitting_marginal_start_value(snapshot, simulator) if use_kitting_value else {}

        def _kitting_precedence_key(rid: str) -> tuple[Any, ...]:
            agv = agv_by_id[rid]
            job = jobs[rid]
            t = tmap[job.task_id]
            here = agv.cur_node or agv.start_node
            has_bypass_or_multilane = sum(
                1 for e in network.out_edges(here) if int(network.node(e.dst).capacity) >= 2
            )
            inside_choke_bonus = 0 if int(network.node(here).capacity) <= 1 and has_bypass_or_multilane <= 1 else 1
            if use_kitting_value and t.task_id in kit_vals:
                kv = kit_vals[t.task_id]
                # (unlock_tier, inside_choke_bonus, urg_tier, -succ_cnt, due_eff, rid)
                return (kv[0], inside_choke_bonus, kv[1], kv[2], kv[4], rid)
            return (0 if t.is_urgent else 1, inside_choke_bonus, int(t.due_step or 999), rid)

        ordered_picked = sorted(cluster_picked, key=_kitting_precedence_key)
        stats["corridor_arbitrations"] += 1
        yielders_first = list(reversed(ordered_picked[1:]))
        leaders_last = ordered_picked[:1]
        for aid in yielders_first + leaders_last:
            _release_future_on_clone(simulator, coord_res, aid)
    else:
        yielders_first = []
        leaders_last = list(stalled)

    # Plan PICKED yielders first via bypass/detour or Haven staging (respecting staged edges)
    for aid in yielders_first:
        agv = agv_by_id[aid]
        job = jobs[aid]
        t = tmap[job.task_id]
        here = agv.cur_node or agv.start_node
        old_next = job.hops[job.idx].dst if job.idx < len(job.hops) else None
        w_detour = dict(base_weights)
        for av in avoid_corridor:
            for e in network.out_edges(av):
                w_detour[(e.src, e.dst)] = 64
                if network.has_edge(e.dst, e.src):
                    w_detour[(e.dst, e.src)] = 64
        trial_res = _clone_res_from(coord_res)
        hops = _plan_on_sim_with_weights(
            simulator, agv, here, t.dropoff, step, w_detour,
            res_table=trial_res, extra_blocked_edges=staged_edges,
        )
        if not hops:
            siding = _find_safe_siding_target(simulator, aid, here, avoid_spurs_all, old_next)
            if siding and siding != here:
                h_pre = _plan_on_sim_with_weights(
                    simulator, agv, here, siding, step, w_detour,
                    res_table=_clone_res_from(coord_res), extra_blocked_edges=staged_edges,
                )
                if h_pre:
                    arr_siding = int(h_pre[-1][3])
                    tail = _plan_on_sim_with_weights(
                        simulator, agv, siding, t.dropoff, arr_siding + 3, w_detour,
                        res_table=_clone_res_from(coord_res), extra_blocked_edges=staged_edges,
                    )
                    hops = (h_pre + tail) if tail else None
        if not hops:
            hops = _plan_on_sim_with_weights(
                simulator, agv, here, t.dropoff, step, base_weights,
                res_table=_clone_res_from(coord_res), extra_blocked_edges=staged_edges,
            )
        if hops:
            _release_future_on_clone(simulator, coord_res, aid)
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
                coord_res.reserve_edge(u, v, Interval(dep, arr), "__REPAIR_PEER__")
            stats["neighbor_yields"] += 1

    # Reroute leader(s) along cleared corridor / shortest unblocked path (with Haven fallback inside chokes)
    for aid in leaders_last:
        agv = agv_by_id[aid]
        job = jobs[aid]
        t = tmap[job.task_id]
        here = agv.cur_node or agv.start_node
        old_next = job.hops[job.idx].dst if job.idx < len(job.hops) else None
        if allow_precedence_swap and len(cluster_picked) >= 2 and yielders_first:
            w_loc = dict(getattr(simulator, "_traffic_weights_q3", {}))
            for ek_fb in fb_blocked:
                w_loc[ek_fb] = max(w_loc.get(ek_fb, 0), 64)
        else:
            w_loc = dict(base_weights)
            if old_next and network.has_edge(here, old_next):
                w_loc[(here, old_next)] = 200
        trial_res = _clone_res_from(coord_res)
        if not (allow_precedence_swap and len(cluster_picked) >= 2):
            _release_future_on_clone(simulator, trial_res, aid)
        hops = _plan_on_sim_with_weights(
            simulator, agv, here, t.dropoff, step, w_loc,
            res_table=trial_res, extra_blocked_edges=staged_edges,
        )
        if not hops:
            hops = _plan_on_sim_with_weights(
                simulator, agv, here, t.dropoff, step, base_weights,
                res_table=_clone_res_from(trial_res), extra_blocked_edges=staged_edges,
            )
        if not hops and here in prof.choke_nodes:
            haven = _find_safe_siding_target(simulator, aid, here, avoid_spurs_all, old_next)
            if haven and haven != here:
                h_pre = _plan_on_sim_with_weights(
                    simulator, agv, here, haven, step, base_weights,
                    res_table=_clone_res_from(trial_res), extra_blocked_edges=staged_edges,
                )
                if h_pre:
                    arr_h = int(h_pre[-1][3])
                    h_post = _plan_on_sim_with_weights(
                        simulator, agv, haven, t.dropoff, arr_h + 2, base_weights,
                        res_table=_clone_res_from(trial_res), extra_blocked_edges=staged_edges,
                    )
                    if h_post:
                        hops = h_pre + h_post
        if hops:
            _release_future_on_clone(simulator, coord_res, aid)
            repairs[aid] = {
                "kind": "picked_reroute",
                "task_id": t.task_id,
                "goal": t.dropoff,
                "hops": hops,
                "traffic_weights": w_loc,
                "traffic_cutoff": step + 50,
            }
            for u, v, dep, arr in hops[:6]:
                staged_edges.append((u, v, dep, arr))
            stats["solo_reroutes"] += 1

    if loading_active:
        repairs = {
            aid: r
            for aid, r in repairs.items()
            if not (aid in loading_active and str(r.get("kind", "")) == "picked_reroute")
        }
    return repairs, stats
