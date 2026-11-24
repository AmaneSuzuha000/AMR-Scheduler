"""Pure-NumPy zero-leakage heterogeneous graph & safety mask extractor for V19.

Constructs entity-level state graphs strictly from `(NativeSnapshot, Network)` without
importing `torch` or `gymnasium`, enabling identical execution on Windows CUDA/CPU and
torch-free WSL2 ROS 2 + Gazebo + Nav2 physical bridges.

Entities & feature widths:
  - `Operation [N_op, 12]`: critical-chain remaining length, due slack, remaining kit
    fraction, unfinished predecessor fraction, resource-window cliff slack, urgency,
    bay congestion, normalized proc time, and commitment tiers.
  - `Task [N_t, 12]`: pickup/dropoff coordinates, travel norm, due slack, urgency,
    kitting bottleneck unlock signal, commitment tiers, disturbed flag, and bay Detour-Delta.
  - `AMR [N_a, 10]`: position, busy/loaded flags, fixed queue load, remaining route,
    L1/L2 execution feedback delay & degradation, disturbed flag, and speed/transit scale.
  - `Bay [N_b, 8]`: position, assigned op load, earliest free time, Detour-Delta,
    degree-1 leaf spur flag, choke flag, and normalized node capacity.
  - `Crew [N_c, 4]`: assigned op load, earliest free time, urgent load share, capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.commitment import (
    TIER_CONDITIONAL,
    TIER_EXECUTING_HARD,
    TIER_IRREVOCABLE,
    TIER_SOFT,
    TIER_UNCOMMITTED,
    CommitmentClassification,
    classify_commitments,
)
from amr_sched.cvcr.snapshot import (
    NativeSnapshot,
    OperationSnapshot,
    TaskSnapshot,
    op_predecessors,
)
from amr_sched.cvcr.v12_proactive_guard import (
    analyze_network_havens,
    compute_detour_delta_signals,
    compute_kitting_marginal_start_value,
)
from amr_sched.network import Network

OP_FEAT_DIM = 12
TASK_FEAT_DIM = 12
AGV_FEAT_DIM = 10
BAY_FEAT_DIM = 8
CREW_FEAT_DIM = 4
OP_BAY_EDGE_DIM = 4
TASK_AGV_EDGE_DIM = 8

_DIST_CACHE: dict[tuple[int, int, int, str], dict[str, int]] = {}


def _clip4(x: np.ndarray) -> np.ndarray:
    """Clamp finite float32 arrays cleanly to `[-4.0, 4.0]` preventing >6sigma OOD blowups."""
    arr = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=4.0, neginf=-4.0)
    return np.clip(arr, -4.0, 4.0).astype(np.float32, copy=False)


def _norm_bounds(network: Network) -> tuple[float, float, float, float]:
    xs = [float(n.x) for n in network.nodes.values()] or [0.0]
    ys = [float(n.y) for n in network.nodes.values()] or [0.0]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    return minx, max(1.0, maxx - minx), miny, max(1.0, maxy - miny)


def _node_xy_norm(
    network: Network,
    node_id: str,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    minx, spanx, miny, spany = bounds
    nd = network.nodes.get(node_id)
    if nd is None:
        return 0.5, 0.5
    return (float(nd.x) - minx) / spanx, (float(nd.y) - miny) / spany


def _travel_dist(network: Network, src: str, dst: str) -> float:
    if src == dst:
        return 0.0
    key = (id(network), len(network.nodes), len(network.edges), src)
    tt = _DIST_CACHE.get(key)
    if tt is None:
        if len(_DIST_CACHE) >= 2048:
            _DIST_CACHE.clear()
        max_spd = max((float(e.speed_limit) for e in network.edges.values()), default=3.0)
        tt = network.shortest_travel_times(src, max(0.2, max_spd)) if src in network.nodes else {}
        _DIST_CACHE[key] = tt
    val = tt.get(dst)
    if val is not None:
        return float(val)
    ns, nd = network.nodes.get(src), network.nodes.get(dst)
    if ns is not None and nd is not None:
        return float(math.hypot(float(ns.x) - float(nd.x), float(ns.y) - float(nd.y)))
    return 60.0


def _compute_crit_rem(snapshot: NativeSnapshot) -> dict[str, float]:
    """Compute remaining DAG critical-chain processing length from each op to its sink."""
    t_now = int(snapshot.tick)
    op_map = {op.op_id: op for op in snapshot.operations}
    children: dict[str, list[str]] = {op.op_id: [] for op in snapshot.operations}
    for op in snapshot.operations:
        for pid in op_predecessors(op):
            if pid in children:
                children[pid].append(op.op_id)

    memo: dict[str, float] = {}
    visiting: set[str] = set()

    def _dfs(oid: str) -> float:
        if oid in memo:
            return memo[oid]
        if oid in visiting:
            return 0.0
        visiting.add(oid)
        op = op_map[oid]
        done = op.started and op.end_step is not None and int(op.end_step) <= t_now
        if done:
            rem = 0.0
        else:
            self_rem = (
                float(max(0, int(op.end_step) - t_now))
                if (op.started and op.end_step is not None)
                else float(op.proc_steps)
            )
            succ_rem = max((_dfs(cid) for cid in children.get(oid, ())), default=0.0)
            rem = self_rem + succ_rem
        visiting.remove(oid)
        memo[oid] = rem
        return rem

    for op in snapshot.operations:
        _dfs(op.op_id)
    return memo


def _window_cliff_slack(op: OperationSnapshot, tick: int) -> tuple[float, float]:
    """Return `(win_exists, win_close_slack_norm)` without peeking beyond static op metadata."""
    wins = tuple(getattr(op, "resource_windows", ()) or ())
    if not wins:
        return 0.0, 1.0
    parsed = sorted((int(a), int(b)) for a, b in wins)
    for wa, wb in parsed:
        if wb >= tick:
            slack = float(wb - tick) / 80.0
            return 1.0, max(-1.0, min(2.0, slack))
    return 1.0, -1.0


@dataclass
class V19HeteroGraph:
    """Zero-leakage heterogeneous graph snapshot + hard safety masks for V19 constructor."""

    tick: int
    op_ids: list[str]
    rel_tids: list[str]
    agv_ids: list[str]
    bay_ids: list[str]
    crew_ids: list[int]

    op_feats: np.ndarray            # [N_op, 12]
    task_feats: np.ndarray          # [N_t, 12]
    agv_feats: np.ndarray           # [N_a, 10]
    bay_feats: np.ndarray           # [N_b, 8]
    crew_feats: np.ndarray          # [N_c, 4]

    task_to_op: np.ndarray          # [N_t] int64 (-1 if no kitting op)
    op_dag_adj: np.ndarray          # [N_op, N_op, 3] (pred, self, succ)
    op_bay_mask: np.ndarray         # [N_op, N_b] bool
    op_bay_feats: np.ndarray        # [N_op, N_b, 4]
    op_crew_mask: np.ndarray        # [N_op, N_c] bool
    task_agv_mask: np.ndarray       # [N_t, N_a] bool
    task_agv_feats: np.ndarray      # [N_t, N_a, 8]

    healthy_agv_indices: list[int]
    fixed_assign: dict[str, str]
    base_bays: dict[str, str]
    base_crews: dict[str, int]
    base_assign: dict[str, str]
    frozen_ops: set[str]
    frozen_tasks: set[str]
    task_pickups: list[str]
    task_incumbent_agv_idx: np.ndarray     # [N_t] int64 (-1 if unassigned/non-healthy)
    task_incumbent_bay_idx: np.ndarray     # [N_t] int64 (current dropoff index in bay_ids)
    op_incumbent_bay_idx: np.ndarray       # [N_op] int64
    op_incumbent_crew_idx: np.ndarray      # [N_op] int64
    bay_xy_travel_from_pickup: np.ndarray  # [N_t, N_b] normalized travel from task pickup -> bay
    bay_detour_norm: np.ndarray            # [N_b]
    task_due_steps: np.ndarray             # [N_t] float32
    agv_est_free_steps: np.ndarray         # [N_a] float32
    agv_cur_nodes: list[str]
    clf: CommitmentClassification


def compute_simulator_aligned_base_assign(
    snapshot: NativeSnapshot,
    network: Network,
    c_clf: CommitmentClassification,
    agvs_sorted: list[Any],
    healthy_flags: list[bool],
    rel_set: set[str],
) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    """Build exact GreedyAllocator-aligned `base_assign`, `fixed_assign`, and `fixed_counts`.

    Replaces arbitrary modulo round-robin on unassigned `PENDING` tasks with the exact
    1-to-1 greedy matching used by `Simulator._assign_step` + `GreedyAllocator.allocate`
    for immediately idle healthy AMRs, plus shortest-completion sequential rollover for
    remaining unassigned releasable/disturbed tasks so `KEEP` evaluates the true incumbent.
    """
    tick = int(snapshot.tick)
    agv_ids = [r.robot_id for r in agvs_sorted]
    agv_idx_map = {aid: idx for idx, aid in enumerate(agv_ids)}
    healthy_agv_ids = [agv_ids[idx] for idx, ok in enumerate(healthy_flags) if ok]
    op_dropoff_by_id = {op.op_id: op.bay_id for op in snapshot.operations}

    fixed_assign: dict[str, str] = {}
    base_assign: dict[str, str] = {}
    fixed_counts: dict[str, int] = {aid: 0 for aid in agv_ids}

    # Pre-seed already committed / healthy incumbent tasks (matches original fixed_counts!)
    unassigned_rel: list[TaskSnapshot] = []
    for t in sorted(snapshot.tasks, key=lambda x: x.task_id):
        if t.task_id in rel_set:
            if t.assigned_agv and t.assigned_agv in agv_idx_map and healthy_flags[agv_idx_map[t.assigned_agv]]:
                base_assign[t.task_id] = t.assigned_agv
            else:
                unassigned_rel.append(t)
        elif t.assigned_agv is not None:
            fixed_assign[t.task_id] = t.assigned_agv
            base_assign[t.task_id] = t.assigned_agv
            if t.status.lower() != "completed" and t.assigned_agv in fixed_counts:
                fixed_counts[t.assigned_agv] += 1

    if not unassigned_rel or not healthy_agv_ids:
        if unassigned_rel and agv_ids:
            for t in unassigned_rel:
                base_assign[t.task_id] = agv_ids[0]
        return fixed_assign, base_assign, fixed_counts

    # Compute exact Simulator max_speed travel times (`planner.max_speed = 3.0`)
    max_spd = max(3.0, max((float(e.speed_limit) for e in network.edges.values()), default=1.0))
    sp_cache: dict[str, dict[str, float]] = {}

    def _tt(src: str, dst: str) -> float:
        if src == dst:
            return 0.0
        table = sp_cache.get(src)
        if table is None:
            table = network.shortest_travel_times(src, max_spd) if src in network.nodes else {}
            sp_cache[src] = table
        val = table.get(dst)
        if val is not None:
            return float(val)
        ns, nd = network.nodes.get(src), network.nodes.get(dst)
        if ns is not None and nd is not None:
            return float(math.hypot(float(ns.x) - float(nd.x), float(ns.y) - float(nd.y)))
        return 1e6

    # Match `Simulator._assign_step` + `GreedyAllocator.allocate` for immediately idle healthy AGVs
    idle_healthy = [
        r for idx, r in enumerate(agvs_sorted)
        if healthy_flags[idx]
        and r.cur_task is None
        and not r.in_transit
        and r.state.lower() not in ("blocked", "faulted")
        and int(r.faulted_until) <= tick
    ]
    ready_pending = [
        t for t in unassigned_rel
        if (t.assigned_agv is None or t.assigned_agv not in healthy_agv_ids)
        and t.status.lower() == "pending"
        and int(t.release_step) <= tick
    ]
    used_agv: set[str] = set()
    used_task: set[str] = set()
    if idle_healthy and ready_pending:
        pairs: list[tuple[int, float, str, str]] = []
        for r in idle_healthy:
            here = r.cur_node
            for t in ready_pending:
                to_pick = _tt(here, t.pickup)
                if to_pick >= 1e5:
                    continue
                to_drop = _tt(t.pickup, t.dropoff)
                if to_drop >= 1e5:
                    continue
                prio_rank = -1 if t.is_urgent else 0
                pairs.append((prio_rank, to_pick + to_drop, r.robot_id, t.task_id))
        pairs.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        for _, _, aid, tid in pairs:
            if aid in used_agv or tid in used_task:
                continue
            base_assign[tid] = aid
            used_agv.add(aid)
            used_task.add(tid)

    # Estimate post-commit node and readiness per healthy AGV for any remaining unassigned tasks
    est_free: dict[str, float] = {}
    est_node: dict[str, str] = {}
    tasks_by_id_loc = {t.task_id: t for t in snapshot.tasks}
    for idx, r in enumerate(agvs_sorted):
        if not healthy_flags[idx]:
            continue
        aid = r.robot_id
        rem_t = float(tick + len(r.route))
        end_n = r.route[-1] if r.route else r.cur_node
        if aid in used_agv:
            # Find the matched task to advance estimated availability
            for tid_m, aid_m in base_assign.items():
                if tid_m in used_task and aid_m == aid:
                    tm = tasks_by_id_loc[tid_m]
                    drop_m = op_dropoff_by_id.get(tm.kit_op_id or "", tm.dropoff) if tm.kit_op_id else tm.dropoff
                    rem_t = max(rem_t, float(tick)) + _tt(r.cur_node, tm.pickup) + _tt(tm.pickup, drop_m) + 4.0
                    end_n = drop_m
                    break
        # Include any already-queued assigned tasks on this AGV
        queued_t = [
            t for t in snapshot.tasks
            if t.task_id not in used_task
            and base_assign.get(t.task_id) == aid
            and t.task_id != r.cur_task
            and t.status.lower() not in ("completed",)
        ]
        for qt in sorted(queued_t, key=lambda x: (x.release_step, x.due_step or 9999, x.task_id)):
            drop_q = op_dropoff_by_id.get(qt.kit_op_id or "", qt.dropoff) if qt.kit_op_id else qt.dropoff
            rem_t = max(rem_t, float(qt.release_step)) + _tt(end_n, qt.pickup) + _tt(qt.pickup, drop_q) + 4.0
            end_n = drop_q
        est_free[aid] = rem_t
        est_node[aid] = end_n

    remaining = [t for t in unassigned_rel if t.task_id not in base_assign]
    remaining.sort(
        key=lambda t: (
            0 if t.is_urgent else 1,
            int(t.release_step),
            int(t.due_step) if t.due_step is not None else 9999,
            t.task_id,
        )
    )
    for t in remaining:
        drop_t = op_dropoff_by_id.get(t.kit_op_id or "", t.dropoff) if t.kit_op_id else t.dropoff
        best_aid = min(
            healthy_agv_ids,
            key=lambda aid: (
                max(est_free[aid], float(t.release_step)) + _tt(est_node[aid], t.pickup) + _tt(t.pickup, drop_t),
                aid,
            ),
        )
        base_assign[t.task_id] = best_aid
        est_free[best_aid] = (
            max(est_free[best_aid], float(t.release_step))
            + _tt(est_node[best_aid], t.pickup)
            + _tt(t.pickup, drop_t)
            + 4.0
        )
        est_node[best_aid] = drop_t

    return fixed_assign, base_assign, fixed_counts


def extract_v19_hetero_graph(
    snapshot: NativeSnapshot,
    network: Network,
    *,
    clf: Optional[CommitmentClassification] = None,
) -> V19HeteroGraph:
    """Extract normalized V19 heterogeneous graph and strict safety masks from `NativeSnapshot`."""
    tick = int(snapshot.tick)
    c_clf = clf if clf is not None else classify_commitments(snapshot)
    havens = analyze_network_havens(network)
    dd_sig = compute_detour_delta_signals(network, snapshot, simulator=None)
    km_val = compute_kitting_marginal_start_value(snapshot)
    bay_dd: dict[str, float] = dd_sig["bay_detour_delta"]
    bounds = _norm_bounds(network)

    tasks_by_id: dict[str, TaskSnapshot] = {t.task_id: t for t in snapshot.tasks}
    ops_sorted = sorted(snapshot.operations, key=lambda o: o.op_id)
    op_ids = [o.op_id for o in ops_sorted]
    op_idx_map = {oid: idx for idx, oid in enumerate(op_ids)}

    rel_tids = sorted(tid for tid in c_clf.releasable_tasks if tid in tasks_by_id)
    rel_set = set(rel_tids)

    agvs_sorted = sorted(snapshot.robots, key=lambda r: r.robot_id)
    agv_ids = [r.robot_id for r in agvs_sorted]
    agv_idx_map = {aid: idx for idx, aid in enumerate(agv_ids)}

    healthy_flags: list[bool] = [
        (r.robot_id not in c_clf.disturbed_robots)
        and r.state != "faulted"
        and int(r.faulted_until) <= tick
        for r in agvs_sorted
    ]
    if not any(healthy_flags):
        healthy_flags = [r.state != "faulted" for r in agvs_sorted]
    if not any(healthy_flags) and agvs_sorted:
        healthy_flags = [True for _ in agvs_sorted]
    healthy_agv_indices = [idx for idx, ok in enumerate(healthy_flags) if ok]

    bay_set: set[str] = set()
    for op in ops_sorted:
        bay_set.add(op.bay_id)
        for eb in op.eligible_bays:
            if eb in network.nodes:
                bay_set.add(eb)
    for t in snapshot.tasks:
        if t.dropoff in network.nodes or t.dropoff in bay_set:
            bay_set.add(t.dropoff)
    bay_ids = sorted(bay_set) if bay_set else sorted(list(network.nodes.keys())[:1])
    bay_idx_map = {bid: idx for idx, bid in enumerate(bay_ids)}

    crew_set: set[int] = set()
    for op in ops_sorted:
        crew_set.add(int(op.crew_id))
        for ec in op.eligible_crews:
            crew_set.add(int(ec))
    crew_ids = sorted(crew_set) if crew_set else [0]
    crew_idx_map = {cid: idx for idx, cid in enumerate(crew_ids)}

    fixed_assign, base_assign, fixed_counts = compute_simulator_aligned_base_assign(
        snapshot, network, c_clf, agvs_sorted, healthy_flags, rel_set
    )

    base_bays = {op.op_id: op.bay_id for op in ops_sorted}
    base_crews = {op.op_id: int(op.crew_id) for op in ops_sorted}

    # --- Bay & Crew current commitments ---
    bay_op_load: dict[str, int] = {bid: 0 for bid in bay_ids}
    bay_free_at: dict[str, int] = {bid: tick for bid in bay_ids}
    crew_op_load: dict[int, int] = {cid: 0 for cid in crew_ids}
    crew_urg_load: dict[int, int] = {cid: 0 for cid in crew_ids}
    crew_free_at: dict[int, int] = {cid: tick for cid in crew_ids}
    for op in ops_sorted:
        done = op.started and op.end_step is not None and int(op.end_step) <= tick
        if done:
            continue
        bay_op_load[op.bay_id] = bay_op_load.get(op.bay_id, 0) + 1
        c_id = int(op.crew_id)
        crew_op_load[c_id] = crew_op_load.get(c_id, 0) + 1
        if op.is_urgent:
            crew_urg_load[c_id] = crew_urg_load.get(c_id, 0) + 1
        if op.started and op.end_step is not None:
            bay_free_at[op.bay_id] = max(bay_free_at.get(op.bay_id, tick), int(op.end_step))
            crew_free_at[c_id] = max(crew_free_at.get(c_id, tick), int(op.end_step))

    # --- 1. Operation Features [N_op, 12], DAG Adj, Bay & Crew Masks ---
    n_op = len(ops_sorted)
    n_b = len(bay_ids)
    n_c = len(crew_ids)
    crit_rem = _compute_crit_rem(snapshot)

    op_feats = np.zeros((n_op, OP_FEAT_DIM), dtype=np.float32)
    op_dag_adj = np.zeros((n_op, n_op, 3), dtype=np.float32)
    op_bay_mask = np.zeros((n_op, n_b), dtype=bool)
    op_bay_feats = np.zeros((n_op, n_b, OP_BAY_EDGE_DIM), dtype=np.float32)
    op_crew_mask = np.zeros((n_op, n_c), dtype=bool)
    op_incumbent_bay_idx = np.zeros((n_op,), dtype=np.int64)
    op_incumbent_crew_idx = np.zeros((n_op,), dtype=np.int64)
    frozen_ops: set[str] = set()

    for oi, op in enumerate(ops_sorted):
        op_dag_adj[oi, oi, 1] = 1.0
        for pid in op_predecessors(op):
            pi = op_idx_map.get(pid)
            if pi is not None:
                op_dag_adj[oi, pi, 0] = 1.0
                op_dag_adj[pi, oi, 2] = 1.0

        kit_total = max(1, len(op.kit_task_ids))
        kit_unfin = sum(
            1
            for ktid in op.kit_task_ids
            if ktid in tasks_by_id and tasks_by_id[ktid].status.lower() != "completed"
        )
        preds = op_predecessors(op)
        pred_unfin = sum(
            1
            for pid in preds
            if pid in op_idx_map
            and not (
                ops_sorted[op_idx_map[pid]].started
                and ops_sorted[op_idx_map[pid]].end_step is not None
                and int(ops_sorted[op_idx_map[pid]].end_step) <= tick
            )
        )
        win_ex, win_slk = _window_cliff_slack(op, tick)
        c_rem = crit_rem.get(op.op_id, float(op.proc_steps))
        due_slk = (float(op.due_step) - float(tick) - c_rem) / 80.0
        bay_cong = bay_dd.get(op.bay_id, 0.0) / 15.0 + bay_op_load.get(op.bay_id, 0) / 3.0
        op_tier = c_clf.op_tiers.get(op.op_id, TIER_SOFT)
        op_feats[oi] = [
            c_rem / 80.0,
            due_slk,
            float(kit_unfin) / float(kit_total),
            float(pred_unfin) / float(max(1, len(preds))) if preds else 0.0,
            win_ex,
            win_slk,
            1.0 if op.is_urgent else 0.0,
            bay_cong,
            float(op.proc_steps) / 30.0,
            1.0 if op_tier in (TIER_IRREVOCABLE, TIER_EXECUTING_HARD) else 0.0,
            1.0 if op_tier == TIER_CONDITIONAL else 0.0,
            1.0 if op_tier == TIER_SOFT else 0.0,
        ]

        inc_bi = bay_idx_map[op.bay_id]
        inc_ci = crew_idx_map[int(op.crew_id)]
        op_incumbent_bay_idx[oi] = inc_bi
        op_incumbent_crew_idx[oi] = inc_ci

        # Hard production commitment check: frozen if started or any kit item frozen/picked/completed
        has_frozen_kit = any(
            (ktid in c_clf.frozen_tasks)
            or (
                ktid in tasks_by_id
                and tasks_by_id[ktid].status.lower() in ("picked", "loading", "unloading", "completed")
            )
            for ktid in op.kit_task_ids
        )
        if op.started or has_frozen_kit:
            frozen_ops.add(op.op_id)
            op_bay_mask[oi, inc_bi] = True
        else:
            for eb in op.eligible_bays:
                if eb in bay_idx_map:
                    op_bay_mask[oi, bay_idx_map[eb]] = True
            op_bay_mask[oi, inc_bi] = True

        if op.started:
            op_crew_mask[oi, inc_ci] = True
        else:
            for ec in op.eligible_crews:
                if int(ec) in crew_idx_map:
                    op_crew_mask[oi, crew_idx_map[int(ec)]] = True
            op_crew_mask[oi, inc_ci] = True

        kit_objs = [tasks_by_id[ktid] for ktid in op.kit_task_ids if ktid in tasks_by_id]
        for bi, bid in enumerate(bay_ids):
            if kit_objs:
                d_mean = sum(_travel_dist(network, kt.pickup, bid) for kt in kit_objs) / float(len(kit_objs))
            else:
                d_mean = 0.0
            op_bay_feats[oi, bi] = [
                min(4.0, d_mean / 40.0),
                1.0 if bid == op.bay_id else 0.0,
                min(4.0, bay_dd.get(bid, 0.0) / 15.0),
                min(4.0, bay_op_load.get(bid, 0) / 4.0),
            ]
    op_feats = _clip4(op_feats)
    op_bay_feats = _clip4(op_bay_feats)

    # --- 2. AGV Features [N_a, 10] ---
    n_a = len(agvs_sorted)
    agv_feats = np.zeros((n_a, AGV_FEAT_DIM), dtype=np.float32)
    agv_est_free_steps = np.zeros((n_a,), dtype=np.float32)
    agv_cur_nodes: list[str] = []
    fb = getattr(snapshot, "execution_feedback", None)
    for ai, r in enumerate(agvs_sorted):
        ax, ay = _node_xy_norm(network, r.cur_node, bounds)
        agv_cur_nodes.append(r.cur_node)
        is_busy = 1.0 if (r.cur_task is not None) else 0.0
        is_loaded = 1.0 if (r.loaded_task_id is not None) else 0.0
        fq_norm = float(fixed_counts.get(r.robot_id, 0)) / 5.0
        rem_steps = float(len(r.route))
        fb_delay = 0.0
        fb_deg = 0.0
        spd_scale = 1.0 if r.in_transit else 0.5
        if fb is not None:
            fb_delay = float(fb.robot_ready_offset_steps(r.robot_id))
            r_fb = fb.robot_states.get(r.robot_id)
            if r_fb is not None:
                fb_deg = 1.0 if r_fb.is_degraded else 0.0
                spd_scale = float(r_fb.speed_scale)
        is_dist = 1.0 if not healthy_flags[ai] else 0.0
        agv_est_free_steps[ai] = float(max(0.0, rem_steps + fb_delay + float(fixed_counts.get(r.robot_id, 0)) * 12.0))
        agv_feats[ai] = [
            ax,
            ay,
            is_busy,
            is_loaded,
            fq_norm,
            rem_steps / 25.0,
            fb_delay / 15.0,
            fb_deg,
            is_dist,
            spd_scale,
        ]
    agv_feats = _clip4(agv_feats)

    # --- 3. Bay Features [N_b, 8] & Crew Features [N_c, 4] ---
    bay_feats = np.zeros((n_b, BAY_FEAT_DIM), dtype=np.float32)
    bay_detour_norm = np.zeros((n_b,), dtype=np.float32)
    for bi, bid in enumerate(bay_ids):
        bx, by = _node_xy_norm(network, bid, bounds)
        nd_obj = network.nodes.get(bid)
        cap = float(nd_obj.capacity) if nd_obj is not None else 1.0
        dd_norm = float(bay_dd.get(bid, 0.0)) / 15.0
        bay_detour_norm[bi] = min(4.0, dd_norm)
        bay_feats[bi] = [
            bx,
            by,
            float(bay_op_load.get(bid, 0)) / 4.0,
            max(0.0, float(bay_free_at.get(bid, tick) - tick)) / 30.0,
            dd_norm,
            1.0 if bid in havens.leaf_spurs else 0.0,
            1.0 if bid in havens.choke_nodes else 0.0,
            cap / 4.0,
        ]
    bay_feats = _clip4(bay_feats)

    crew_feats = np.zeros((n_c, CREW_FEAT_DIM), dtype=np.float32)
    for ci, cid in enumerate(crew_ids):
        tot_l = max(1, crew_op_load.get(cid, 0))
        crew_feats[ci] = [
            float(crew_op_load.get(cid, 0)) / 4.0,
            max(0.0, float(crew_free_at.get(cid, tick) - tick)) / 30.0,
            float(crew_urg_load.get(cid, 0)) / float(tot_l),
            1.0 / 4.0,
        ]
    crew_feats = _clip4(crew_feats)

    # --- 4. Releasable Task Features [N_t, 12] & Task x AGV Edges [N_t, N_a, 8] ---
    n_t = len(rel_tids)
    task_feats = np.zeros((n_t, TASK_FEAT_DIM), dtype=np.float32)
    task_to_op = np.full((n_t,), -1, dtype=np.int64)
    task_agv_mask = np.zeros((n_t, n_a), dtype=bool)
    task_agv_feats = np.zeros((n_t, n_a, TASK_AGV_EDGE_DIM), dtype=np.float32)
    task_pickups: list[str] = []
    task_incumbent_agv_idx = np.full((n_t,), -1, dtype=np.int64)
    task_incumbent_bay_idx = np.zeros((n_t,), dtype=np.int64)
    bay_xy_travel_from_pickup = np.zeros((n_t, n_b), dtype=np.float32)
    task_due_steps = np.zeros((n_t,), dtype=np.float32)

    res_count_by_agv: dict[str, int] = {}
    for rv in snapshot.reservations:
        res_count_by_agv[rv.agv_id] = res_count_by_agv.get(rv.agv_id, 0) + 1

    # Choke node set along agv -> pickup corridor heuristic overlap
    for ti, tid in enumerate(rel_tids):
        t = tasks_by_id[tid]
        task_pickups.append(t.pickup)
        oid = t.kit_op_id or ""
        oi = op_idx_map.get(oid, -1)
        task_to_op[ti] = oi
        op_obj = ops_sorted[oi] if oi >= 0 else None

        px, py = _node_xy_norm(network, t.pickup, bounds)
        dn_id = op_obj.bay_id if op_obj is not None else t.dropoff
        dx, dy = _node_xy_norm(network, dn_id, bounds)
        if dn_id in bay_idx_map:
            task_incumbent_bay_idx[ti] = bay_idx_map[dn_id]

        for bi, bid in enumerate(bay_ids):
            bay_xy_travel_from_pickup[ti, bi] = min(4.0, _travel_dist(network, t.pickup, bid) / 40.0)

        travel_s = _travel_dist(network, t.pickup, dn_id)
        due = float(t.due_step if t.due_step is not None else (tick + 120))
        task_due_steps[ti] = due
        due_slk = (due - float(tick)) / 80.0
        urg = 1.0 if (t.is_urgent or (op_obj is not None and op_obj.is_urgent)) else 0.0
        km_tier = km_val.get(tid, (3,))[0]
        kit_bot = 1.0 if km_tier == 0 else (0.5 if km_tier == 1 else 0.0)
        t_tier = c_clf.task_tiers.get(tid, TIER_UNCOMMITTED)
        orig_disturbed = 1.0 if (
            (t.assigned_agv is not None and t.assigned_agv in c_clf.disturbed_robots)
            or tid in c_clf.disturbed_tasks
        ) else 0.0

        task_feats[ti] = [
            px,
            py,
            dx,
            dy,
            travel_s / 35.0,
            due_slk,
            urg,
            kit_bot,
            1.0 if t_tier == TIER_SOFT else 0.0,
            1.0 if t_tier == TIER_UNCOMMITTED else 0.0,
            orig_disturbed,
            bay_dd.get(dn_id, 0.0) / 15.0,
        ]

        if t.assigned_agv and t.assigned_agv in agv_idx_map and healthy_flags[agv_idx_map[t.assigned_agv]]:
            task_incumbent_agv_idx[ti] = agv_idx_map[t.assigned_agv]

        win_ex = 1.0 if (op_obj is not None and bool(getattr(op_obj, "resource_windows", ()))) else 0.0
        for ai, aid in enumerate(agv_ids):
            task_agv_mask[ti, ai] = bool(healthy_flags[ai])
            deadhead = _travel_dist(network, agvs_sorted[ai].cur_node, t.pickup)
            est_total = deadhead + travel_s + float(agv_est_free_steps[ai]) * 0.4
            margin = (due - (float(tick) + est_total)) / 60.0
            is_inc = 1.0 if t.assigned_agv == aid else 0.0
            soft_pen = 1.0 if (t_tier == TIER_SOFT and t.assigned_agv is not None and t.assigned_agv != aid) else 0.0
            agv_load = (fixed_counts.get(aid, 0) + res_count_by_agv.get(aid, 0) * 0.1) / 5.0
            choke_ov = (
                (1.0 if agvs_sorted[ai].cur_node in havens.choke_nodes else 0.0)
                + (1.0 if dn_id in havens.leaf_spurs else 0.0)
            ) * 0.5
            task_agv_feats[ti, ai] = [
                deadhead / 30.0,
                est_total / 60.0,
                margin,
                is_inc,
                soft_pen,
                agv_load,
                choke_ov,
                win_ex,
            ]

    task_feats = _clip4(task_feats)
    task_agv_feats = _clip4(task_agv_feats)

    return V19HeteroGraph(
        tick=tick,
        op_ids=op_ids,
        rel_tids=rel_tids,
        agv_ids=agv_ids,
        bay_ids=bay_ids,
        crew_ids=crew_ids,
        op_feats=op_feats,
        task_feats=task_feats,
        agv_feats=agv_feats,
        bay_feats=bay_feats,
        crew_feats=crew_feats,
        task_to_op=task_to_op,
        op_dag_adj=op_dag_adj,
        op_bay_mask=op_bay_mask,
        op_bay_feats=op_bay_feats,
        op_crew_mask=op_crew_mask,
        task_agv_mask=task_agv_mask,
        task_agv_feats=task_agv_feats,
        healthy_agv_indices=healthy_agv_indices,
        fixed_assign=fixed_assign,
        base_bays=base_bays,
        base_crews=base_crews,
        base_assign=base_assign,
        frozen_ops=frozen_ops,
        frozen_tasks=set(c_clf.frozen_tasks),
        task_pickups=task_pickups,
        task_incumbent_agv_idx=task_incumbent_agv_idx,
        task_incumbent_bay_idx=task_incumbent_bay_idx,
        op_incumbent_bay_idx=op_incumbent_bay_idx,
        op_incumbent_crew_idx=op_incumbent_crew_idx,
        bay_xy_travel_from_pickup=bay_xy_travel_from_pickup,
        bay_detour_norm=bay_detour_norm,
        task_due_steps=task_due_steps,
        agv_est_free_steps=agv_est_free_steps,
        agv_cur_nodes=agv_cur_nodes,
        clf=c_clf,
    )


ORTHOGONAL_ABLATION_MODES: tuple[str, ...] = (
    "full",
    "staged",
    "no_prod_comm",
    "no_op_dag",
    "no_kit_features",
    "no_commit_features",
    "no_cross_edge_msg",
    "no_a3_bay_cond",
)


def apply_graph_ablation(graph: V19HeteroGraph, ablation_mode: str) -> V19HeteroGraph:
    """Return a graph view with the specified ablation mask applied."""
    if ablation_mode not in ORTHOGONAL_ABLATION_MODES:
        raise ValueError(f"Unsupported ablation_mode {ablation_mode!r}")
    if ablation_mode in ("full", "staged", "no_cross_edge_msg", "no_a3_bay_cond"):
        return graph

    op_f = graph.op_feats.copy()
    task_f = graph.task_feats.copy()
    ta_f = graph.task_agv_feats.copy()
    dag = graph.op_dag_adj.copy()

    if ablation_mode in ("no_prod_comm", "no_op_dag"):
        dag = np.zeros_like(graph.op_dag_adj)
        for i in range(dag.shape[0]):
            dag[i, i, 1] = 1.0

    if ablation_mode == "no_prod_comm":
        if op_f.shape[0] > 0:
            op_f[:, [0, 1, 2, 3, 4, 5, 9, 10, 11]] = 0.0
        if task_f.shape[0] > 0:
            task_f[:, [7, 8, 9, 10]] = 0.0
        if ta_f.shape[0] > 0:
            ta_f[:, :, [3, 4, 7]] = 0.0
    elif ablation_mode == "no_kit_features":
        if op_f.shape[0] > 0:
            op_f[:, [2, 3]] = 0.0
        if task_f.shape[0] > 0:
            task_f[:, [7]] = 0.0
        if ta_f.shape[0] > 0:
            ta_f[:, :, [7]] = 0.0
    elif ablation_mode == "no_commit_features":
        if op_f.shape[0] > 0:
            op_f[:, [9, 10, 11]] = 0.0
        if task_f.shape[0] > 0:
            task_f[:, [8, 9, 10]] = 0.0
        if ta_f.shape[0] > 0:
            ta_f[:, :, [3, 4]] = 0.0

    return V19HeteroGraph(
        tick=graph.tick,
        op_ids=graph.op_ids,
        rel_tids=graph.rel_tids,
        agv_ids=graph.agv_ids,
        bay_ids=graph.bay_ids,
        crew_ids=graph.crew_ids,
        op_feats=op_f,
        task_feats=task_f,
        agv_feats=graph.agv_feats,
        bay_feats=graph.bay_feats,
        crew_feats=graph.crew_feats,
        task_to_op=graph.task_to_op,
        op_dag_adj=dag,
        op_bay_mask=graph.op_bay_mask,
        op_bay_feats=graph.op_bay_feats,
        op_crew_mask=graph.op_crew_mask,
        task_agv_mask=graph.task_agv_mask,
        task_agv_feats=ta_f,
        healthy_agv_indices=graph.healthy_agv_indices,
        fixed_assign=graph.fixed_assign,
        base_bays=graph.base_bays,
        base_crews=graph.base_crews,
        base_assign=graph.base_assign,
        frozen_ops=graph.frozen_ops,
        frozen_tasks=graph.frozen_tasks,
        task_pickups=graph.task_pickups,
        task_incumbent_agv_idx=graph.task_incumbent_agv_idx,
        task_incumbent_bay_idx=graph.task_incumbent_bay_idx,
        op_incumbent_bay_idx=graph.op_incumbent_bay_idx,
        op_incumbent_crew_idx=graph.op_incumbent_crew_idx,
        bay_xy_travel_from_pickup=graph.bay_xy_travel_from_pickup,
        bay_detour_norm=graph.bay_detour_norm,
        task_due_steps=graph.task_due_steps,
        agv_est_free_steps=graph.agv_est_free_steps,
        agv_cur_nodes=graph.agv_cur_nodes,
        clf=graph.clf,
    )


__all__ = [
    "AGV_FEAT_DIM",
    "BAY_FEAT_DIM",
    "CREW_FEAT_DIM",
    "OP_BAY_EDGE_DIM",
    "OP_FEAT_DIM",
    "TASK_AGV_EDGE_DIM",
    "TASK_FEAT_DIM",
    "V19HeteroGraph",
    "apply_graph_ablation",
    "extract_v19_hetero_graph",
]
