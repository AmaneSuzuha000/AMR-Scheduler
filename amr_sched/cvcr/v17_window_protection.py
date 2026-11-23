"""V17 Dual-Op Window Protection under Physical Execution Commitments.

Implements:
  1. Stage 1 (E0): Legal bridge-order control across `pre_pickup` (`PENDING`/`ASSIGNED`),
     `loaded_at_staging` (`PICKED` at haven/buffer), and strict rejection/masking for
     `in_edge` (`_transit`) or no-safe-haven configurations via `proposed_op_priorities`,
     `proposed_live_repairs`, and `proposed_bottleneck_locks`.
  2. Stage 2 (E1): Pairwise Window Payoff Representation `(K(o,a), P(o), R(o,a), S(o,a))`,
     window-crossing indicators, and action-diff deduplicated dual-op candidate enumeration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.checkpoints import resolve_default_checkpoint
from amr_sched.cvcr.commitment import CommitmentClassification
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.snapshot import NativeSnapshot, align_start_to_windows, op_predecessors
from amr_sched.cvcr.v15_ea_joint import (
    ACTION_KEEP,
    V15_GLOBAL_FEAT_DIM,
    V15Candidate,
    V15ExecutionAwarePolicy,
    _build_candidate_features,
    _relu,
    _sigmoid,
    _softmax,
    compute_action_diff_digest,
)
from amr_sched.cvcr.v16_frontier import (
    DEFAULT_V16_CKPT,
    FEATURE_SCHEMA_VERSION,
    V16_CTX_DIM,
    V16_MODES,
    V16AdaptiveController,
    V16FrontierPolicy,
    _load_cached_npz_weights,
    extract_v16_context,
    load_v16_frontier_policy,
)
from amr_sched.network import Network, NodeKind


@dataclass
class BridgeOrderControlPlan:
    """Verified legal crossing-order plan prioritizing `leader_op_id` over `yielder_op_id`."""

    eligible: bool
    reason: str
    leader_op_id: str
    yielder_op_id: str
    leader_agv: Optional[str]
    yielder_agv: Optional[str]
    shared_bottleneck_nodes: tuple[str, ...]
    bridge_clear_step: int
    op_priority: dict[str, int]
    eval_res: Optional[EvalResult]
    live_repairs: Optional[dict[str, dict[str, Any]]]
    bottleneck_locks: Optional[dict[str, list[tuple[str, int, int]]]]


def _can_reach_avoiding(
    network: Network,
    start: str,
    goal: str,
    avoid: set[str],
) -> bool:
    """Return True if `goal` is reachable from `start` without entering `avoid` nodes."""
    if start == goal:
        return True
    if start in avoid or goal in avoid or start not in network.nodes or goal not in network.nodes:
        return False
    q: deque[str] = deque([start])
    seen = {start}
    while q:
        u = q.popleft()
        for e in network.out_edges(u):
            v = e.dst
            if v in seen or v in avoid:
                continue
            if v == goal:
                return True
            seen.add(v)
            q.append(v)
    return False


def _find_adjacent_haven(
    network: Network,
    node_id: str,
    avoid: set[str],
) -> Optional[str]:
    """Find an adjacent buffer/aisle or multi-capacity refuge node outside `avoid`."""
    if node_id not in network.nodes:
        return None
    for e in sorted(network.out_edges(node_id), key=lambda x: x.dst):
        v = e.dst
        if v in avoid or v not in network.nodes:
            continue
        nd = network.node(v)
        if int(nd.capacity) >= 2 and nd.kind in (NodeKind.BUFFER, NodeKind.AISLE):
            return v
        if nd.kind == NodeKind.BUFFER:
            return v
    return None


def _sim_hop_steps(network: Network, sim: Any, u: str, v: str, default_speed: float = 1.0) -> int:
    """Compute discrete edge traversal steps consistent with `Simulator._hop_steps`."""
    if sim is not None and hasattr(sim, "_hop_steps"):
        return max(1, int(sim._hop_steps(u, v)))
    e = network.edge(u, v)
    spd = min(float(default_speed), float(e.speed_limit))
    return max(1, int(np.ceil(float(e.length) / max(1e-6, spd))))


def extract_timed_hops_for_leg(
    network: Network,
    path_pts: list[tuple[str, int]],
    start_node: str,
    goal_node: str,
    tick: int,
    *,
    sim: Any = None,
    default_speed: float = 1.0,
) -> list[tuple[str, str, int, int]]:
    """Extract contiguous directed edge hops `(u, v, dep, arr)` aligned cleanly to `sim._hop_steps`."""
    if not path_pts or start_node == goal_node:
        return []
    start_idx = 0
    for idx, (nd, ts) in enumerate(path_pts):
        if nd == start_node and int(ts) >= int(tick):
            start_idx = idx
            break
    # Walk until first arrival at goal_node after leaving start
    hops: list[tuple[str, str, int, int]] = []
    cur_n = start_node
    cur_t = int(tick)
    for idx in range(start_idx, len(path_pts) - 1):
        u_n, u_t = path_pts[idx]
        v_n, _v_t = path_pts[idx + 1]
        if u_n == v_n:
            cur_t = max(cur_t, int(_v_t))
            continue
        if u_n != cur_n:
            continue
        if not network.has_edge(u_n, v_n):
            break
        n_steps = _sim_hop_steps(network, sim, u_n, v_n, default_speed=default_speed)
        dep = max(cur_t, int(u_t))
        arr = dep + n_steps
        hops.append((u_n, v_n, dep, arr))
        cur_n = v_n
        cur_t = arr
        if cur_n == goal_node:
            break
    if not hops or hops[-1][1] != goal_node:
        return []
    return hops


def _identify_shared_bottleneck_nodes(
    network: Network,
    path_a: list[tuple[str, int]],
    path_b: list[tuple[str, int]],
    exclude_endpoints: set[str],
) -> list[str]:
    """Identify shared corridor nodes traversed by both robots (`cap<=1` or interior aisle)."""
    nodes_a = [n for n, _ in path_a]
    set_b = {n for n, _ in path_b}
    shared: list[str] = []
    seen: set[str] = set()
    for nd in nodes_a:
        if nd in seen or nd not in set_b or nd in exclude_endpoints or nd not in network.nodes:
            continue
        nd_obj = network.node(nd)
        if int(nd_obj.capacity) <= 1 or nd_obj.kind == NodeKind.AISLE:
            seen.add(nd)
            shared.append(nd)
    return shared


def _compute_clear_step(
    path_pts: list[tuple[str, int]],
    bottleneck_nodes: set[str],
    tick: int,
    extra_load_shift: int = 0,
) -> int:
    """Compute step when `leader` exits all shared bottleneck nodes."""
    if not path_pts or not bottleneck_nodes:
        return int(tick) + 1
    clear_t = int(tick) + 1
    for i, (nd, ts) in enumerate(path_pts):
        if nd in bottleneck_nodes:
            if i + 1 < len(path_pts):
                clear_t = max(clear_t, int(path_pts[i + 1][1]) + int(extra_load_shift))
            else:
                clear_t = max(clear_t, int(ts) + 1 + int(extra_load_shift))
    return int(clear_t)


def plan_bridge_order_control(
    network: Network,
    snapshot: NativeSnapshot,
    evaluator: CVCREvaluator,
    assign: dict[str, str],
    base_prio: dict[str, int],
    leader_op_id: str,
    yielder_op_id: str,
    *,
    sim: Any = None,
    base_bays: Optional[dict[str, str]] = None,
    base_crews: Optional[dict[str, int]] = None,
    allow_in_edge: bool = False,
) -> BridgeOrderControlPlan:
    """Construct a commitment-preserving crossing-order control plan for `(leader_op_id, yielder_op_id)`.

    Enforces:
      - Strict `in_edge` (`_transit` / active mid-edge) irrevocability masking unless bypassed for negative tests.
      - Safe waiting haven verification when the yielder sits on a capacity-1 node.
      - Priority-ordered space-time evaluation (`CVCREvaluator`) plus `NativeTransactionOwner`
        enforceable `proposed_live_repairs` and `proposed_bottleneck_locks`.
    """
    tick = int(snapshot.tick)
    ops_by_id = {op.op_id: op for op in snapshot.operations}
    robs_by_id = {r.robot_id: r for r in snapshot.robots}
    tasks_by_id = {t.task_id: t for t in snapshot.tasks}

    def _op_task(op_id: str) -> Any:
        op = ops_by_id.get(op_id)
        if op is None:
            return None
        cands = [
            tasks_by_id[tid]
            for tid in op.kit_task_ids
            if tid in tasks_by_id and tasks_by_id[tid].status.lower() != "completed"
        ]
        if not cands:
            return None
        cands.sort(key=lambda t: (0 if t.status.lower() == "picked" else 1, t.release_step, t.task_id))
        return cands[0]

    t_lead = _op_task(leader_op_id)
    t_yield = _op_task(yielder_op_id)
    if t_lead is None or t_yield is None:
        return BridgeOrderControlPlan(
            eligible=False,
            reason="missing_active_kit_task",
            leader_op_id=leader_op_id,
            yielder_op_id=yielder_op_id,
            leader_agv=None,
            yielder_agv=None,
            shared_bottleneck_nodes=(),
            bridge_clear_step=tick,
            op_priority=dict(base_prio),
            eval_res=None,
            live_repairs=None,
            bottleneck_locks=None,
        )

    lead_agv = t_lead.assigned_agv or assign.get(t_lead.task_id)
    yield_agv = t_yield.assigned_agv or assign.get(t_yield.task_id)
    if not lead_agv or not yield_agv:
        return BridgeOrderControlPlan(
            eligible=False,
            reason="unassigned_robot",
            leader_op_id=leader_op_id,
            yielder_op_id=yielder_op_id,
            leader_agv=lead_agv,
            yielder_agv=yield_agv,
            shared_bottleneck_nodes=(),
            bridge_clear_step=tick,
            op_priority=dict(base_prio),
            eval_res=None,
            live_repairs=None,
            bottleneck_locks=None,
        )

    if lead_agv == yield_agv:
        # Same robot serves both: handle via priority ordering & task sequences, no live yield lock needed
        p_lo = int(base_prio.get(leader_op_id, 0))
        p_hi = int(base_prio.get(yielder_op_id, 0))
        new_prio = dict(base_prio)
        new_prio[leader_op_id] = min(p_lo, p_hi) - 10
        new_prio[yielder_op_id] = max(p_lo, p_hi) + 10
        ev = evaluator.evaluate(
            assign,
            op_priority=new_prio,
            op_bay_overrides=base_bays,
            op_crew_overrides=base_crews,
        )
        return BridgeOrderControlPlan(
            eligible=bool(ev.valid),
            reason="same_robot_seq" if ev.valid else "eval_invalid",
            leader_op_id=leader_op_id,
            yielder_op_id=yielder_op_id,
            leader_agv=lead_agv,
            yielder_agv=yield_agv,
            shared_bottleneck_nodes=(),
            bridge_clear_step=tick,
            op_priority=new_prio,
            eval_res=ev,
            live_repairs=None,
            bottleneck_locks=None,
        )

    transit_map = getattr(sim, "_transit", {}) if sim is not None else {}
    faulted_set = getattr(sim, "_faulted_agvs", set()) if sim is not None else set()
    load_end_map = getattr(sim, "_load_end", {}) if sim is not None else {}
    r_lead = robs_by_id.get(lead_agv)
    r_yield = robs_by_id.get(yield_agv)
    if r_lead is None or r_yield is None:
        return BridgeOrderControlPlan(
            eligible=False,
            reason="unknown_robot",
            leader_op_id=leader_op_id,
            yielder_op_id=yielder_op_id,
            leader_agv=lead_agv,
            yielder_agv=yield_agv,
            shared_bottleneck_nodes=(),
            bridge_clear_step=tick,
            op_priority=dict(base_prio),
            eval_res=None,
            live_repairs=None,
            bottleneck_locks=None,
        )

    in_edge_lead = (lead_agv in transit_map) or (r_lead.state.lower() == "in_transit") or bool(getattr(r_lead, "in_transit", False))
    in_edge_yield = (yield_agv in transit_map) or (r_yield.state.lower() == "in_transit") or bool(getattr(r_yield, "in_transit", False))
    if (in_edge_lead or in_edge_yield) and not allow_in_edge:
        return BridgeOrderControlPlan(
            eligible=False,
            reason="in_edge_irrevocable",
            leader_op_id=leader_op_id,
            yielder_op_id=yielder_op_id,
            leader_agv=lead_agv,
            yielder_agv=yield_agv,
            shared_bottleneck_nodes=(),
            bridge_clear_step=tick,
            op_priority=dict(base_prio),
            eval_res=None,
            live_repairs=None,
            bottleneck_locks=None,
        )

    if (
        lead_agv in faulted_set
        or yield_agv in faulted_set
        or r_lead.state.lower() == "faulted"
        or r_yield.state.lower() == "faulted"
    ):
        return BridgeOrderControlPlan(
            eligible=False,
            reason="faulted_robot",
            leader_op_id=leader_op_id,
            yielder_op_id=yielder_op_id,
            leader_agv=lead_agv,
            yielder_agv=yield_agv,
            shared_bottleneck_nodes=(),
            bridge_clear_step=tick,
            op_priority=dict(base_prio),
            eval_res=None,
            live_repairs=None,
            bottleneck_locks=None,
        )

    if (t_lead.status.lower() == "picked" and int(load_end_map.get(lead_agv, 0)) > tick) or (
        t_yield.status.lower() == "picked" and int(load_end_map.get(yield_agv, 0)) > tick
    ):
        return BridgeOrderControlPlan(
            eligible=False,
            reason="active_loading_dwell",
            leader_op_id=leader_op_id,
            yielder_op_id=yielder_op_id,
            leader_agv=lead_agv,
            yielder_agv=yield_agv,
            shared_bottleneck_nodes=(),
            bridge_clear_step=tick,
            op_priority=dict(base_prio),
            eval_res=None,
            live_repairs=None,
            bottleneck_locks=None,
        )

    # Safe haven check: if yielder stands on a capacity-1 node blocking leader's path without haven
    l_drop = (base_bays or {}).get(t_lead.kit_op_id or "", t_lead.dropoff)
    l_first_target = t_lead.dropoff if t_lead.status.lower() == "picked" else t_lead.pickup
    if (
        r_yield.cur_node in network.nodes
        and int(network.node(r_yield.cur_node).capacity) <= 1
        and r_yield.cur_node != r_lead.cur_node
    ):
        blocked_for_lead = not _can_reach_avoiding(network, r_lead.cur_node, l_drop, {r_yield.cur_node})
        if t_lead.status.lower() != "picked":
            blocked_for_lead = blocked_for_lead or (
                not _can_reach_avoiding(network, r_lead.cur_node, l_first_target, {r_yield.cur_node})
            )
        if blocked_for_lead and _find_adjacent_haven(network, r_yield.cur_node, {r_lead.cur_node}) is None:
            return BridgeOrderControlPlan(
                eligible=False,
                reason="no_safe_yield_haven",
                leader_op_id=leader_op_id,
                yielder_op_id=yielder_op_id,
                leader_agv=lead_agv,
                yielder_agv=yield_agv,
                shared_bottleneck_nodes=(),
                bridge_clear_step=tick,
                op_priority=dict(base_prio),
                eval_res=None,
                live_repairs=None,
                bottleneck_locks=None,
            )

    p_l = int(base_prio.get(leader_op_id, 0))
    p_y = int(base_prio.get(yielder_op_id, 0))
    new_prio = dict(base_prio)
    base_min = min(new_prio.values(), default=0)
    base_max = max(new_prio.values(), default=0)
    new_prio[leader_op_id] = min(p_l, p_y, base_min) - 500
    new_prio[yielder_op_id] = max(p_l, p_y, base_max) + 500

    ev_res = evaluator.evaluate(
        assign,
        op_priority=new_prio,
        op_bay_overrides=base_bays,
        op_crew_overrides=base_crews,
    )
    if not ev_res.valid:
        return BridgeOrderControlPlan(
            eligible=False,
            reason="eval_invalid",
            leader_op_id=leader_op_id,
            yielder_op_id=yielder_op_id,
            leader_agv=lead_agv,
            yielder_agv=yield_agv,
            shared_bottleneck_nodes=(),
            bridge_clear_step=tick,
            op_priority=new_prio,
            eval_res=ev_res,
            live_repairs=None,
            bottleneck_locks=None,
        )

    path_l = ev_res.robot_paths.get(lead_agv, [])
    path_y = ev_res.robot_paths.get(yield_agv, [])
    excl_pts = {r_lead.cur_node, r_yield.cur_node, l_drop, t_yield.dropoff}
    shared_nodes = _identify_shared_bottleneck_nodes(network, path_l, path_y, excl_pts)

    extra_load_shift = 0
    if sim is not None and t_lead.status.lower() in ("pending", "assigned") and hasattr(sim, "_load_steps"):
        raw_t = sim._task_by_id.get(t_lead.task_id) if hasattr(sim, "_task_by_id") else None
        extra_load_shift = max(0, int(sim._load_steps(raw_t)) - 2)
    t_clear = _compute_clear_step(path_l, set(shared_nodes), tick, extra_load_shift=extra_load_shift)

    # Lock shared bottleneck entry/interior nodes on leader_rid to prevent early-departure overtake
    lock_spans = [(nd, tick, t_clear) for nd in shared_nodes] if t_clear > tick else []
    bottleneck_locks = {lead_agv: lock_spans} if lock_spans else None

    live_repairs: dict[str, dict[str, Any]] = {}
    lead_stat = t_lead.status.lower()
    yield_stat = t_yield.status.lower()
    if lead_stat == "picked":
        l_hops = extract_timed_hops_for_leg(
            network, path_l, r_lead.cur_node, l_drop, tick, sim=sim, default_speed=evaluator.max_speed
        )
        if l_hops:
            live_repairs[lead_agv] = {
                "kind": "picked_reroute",
                "task_id": t_lead.task_id,
                "goal": l_drop,
                "hops": l_hops,
                "min_reserve_hops": len(l_hops),
                "extra_node_locks": lock_spans,
            }
    elif lead_stat == "assigned":
        l_hops1 = extract_timed_hops_for_leg(
            network, path_l, r_lead.cur_node, t_lead.pickup, tick, sim=sim, default_speed=evaluator.max_speed
        )
        if l_hops1:
            live_repairs[lead_agv] = {
                "kind": "assigned_reroute",
                "task_id": t_lead.task_id,
                "goal": t_lead.pickup,
                "hops": l_hops1,
                "min_reserve_hops": len(l_hops1),
                "extra_node_locks": lock_spans,
            }

    if yield_stat == "picked":
        y_drop = (base_bays or {}).get(t_yield.kit_op_id or "", t_yield.dropoff)
        y_hops = extract_timed_hops_for_leg(
            network, path_y, r_yield.cur_node, y_drop, tick, sim=sim, default_speed=evaluator.max_speed
        )
        if y_hops:
            live_repairs[yield_agv] = {
                "kind": "picked_reroute",
                "task_id": t_yield.task_id,
                "goal": y_drop,
                "hops": y_hops,
                "min_reserve_hops": len(y_hops),
            }
    elif yield_stat == "assigned":
        y_hops1 = extract_timed_hops_for_leg(
            network, path_y, r_yield.cur_node, t_yield.pickup, tick, sim=sim, default_speed=evaluator.max_speed
        )
        if y_hops1:
            live_repairs[yield_agv] = {
                "kind": "assigned_reroute",
                "task_id": t_yield.task_id,
                "goal": t_yield.pickup,
                "hops": y_hops1,
                "min_reserve_hops": len(y_hops1),
            }

    return BridgeOrderControlPlan(
        eligible=True,
        reason="ok",
        leader_op_id=leader_op_id,
        yielder_op_id=yielder_op_id,
        leader_agv=lead_agv,
        yielder_agv=yield_agv,
        shared_bottleneck_nodes=tuple(shared_nodes),
        bridge_clear_step=int(t_clear),
        op_priority=new_prio,
        eval_res=ev_res,
        live_repairs=live_repairs or None,
        bottleneck_locks=bottleneck_locks,
    )


V17_PAIRWISE_DIM = 14


def _matched_window_index(earliest_step: int, windows: Any) -> int:
    """Return index of first non-expired window `[w_s, w_e]`, or `len(windows)` if all missed."""
    if not windows:
        return 0
    t = int(earliest_step)
    wins = sorted((int(a), int(b)) for a, b in windows)
    for idx, (_ws, we) in enumerate(wins):
        if t <= we:
            return idx
    return len(wins)


def compute_op_readiness_profile(
    snapshot: NativeSnapshot,
    eval_res: EvalResult,
    op_prio: Optional[dict[str, int]] = None,
    op_bay_overrides: Optional[dict[str, str]] = None,
    op_crew_overrides: Optional[dict[str, int]] = None,
) -> dict[str, dict[str, Any]]:
    """Compute `(K, P, R, S, end, win_idx, win_margin, tard)` per operation from an `EvalResult`."""
    t_now = int(snapshot.tick)
    prio = op_prio or {op.op_id: op.due_step for op in snapshot.operations}
    ops = sorted(
        snapshot.operations,
        key=lambda op: (0 if op.started else 1, prio.get(op.op_id, op.due_step), op.op_id),
    )
    unloads = eval_res.order_unload_ticks if eval_res.valid else {}
    op_end: dict[str, int] = {}
    bay_free: dict[str, int] = {}
    crew_free: dict[int, int] = {}
    tool_free: dict[int, int] = {}
    profile: dict[str, dict[str, Any]] = {}
    resolved: set[str] = set()

    for _ in range(len(ops) + 2):
        for op in ops:
            if op.op_id in resolved:
                continue
            preds = op_predecessors(op)
            if any(pid not in resolved and pid in {o.op_id for o in ops} for pid in preds):
                continue
            bay = (op_bay_overrides or {}).get(op.op_id, op.bay_id)
            crew = int((op_crew_overrides or {}).get(op.op_id, op.crew_id))
            if op.started and op.start_step is not None and op.end_step is not None:
                st_s = int(op.start_step)
                en_s = int(op.end_step)
                profile[op.op_id] = {
                    "K": st_s,
                    "P": 0,
                    "R": st_s,
                    "S": st_s,
                    "end": en_s,
                    "win_idx": 0,
                    "win_margin": 999,
                    "tard": max(0, en_s - int(op.due_step)),
                }
                op_end[op.op_id] = en_s
                bay_free[bay] = max(bay_free.get(bay, 0), en_s)
                crew_free[crew] = max(crew_free.get(crew, 0), en_s)
                tool_free[int(op.tool_id)] = max(tool_free.get(int(op.tool_id), 0), en_s)
            else:
                k_val = max((int(unloads.get(tid, t_now + 240)) for tid in op.kit_task_ids), default=t_now)
                p_val = max((op_end.get(pid, 0) for pid in preds), default=0)
                r_val = max(
                    t_now,
                    p_val,
                    bay_free.get(bay, t_now),
                    crew_free.get(crew, t_now),
                    tool_free.get(int(op.tool_id), t_now),
                )
                ready = max(k_val, r_val)
                wins = getattr(op, "resource_windows", ())
                st_s = align_start_to_windows(ready, wins)
                en_s = st_s + int(op.proc_steps)
                w_idx = _matched_window_index(ready, wins)
                if wins:
                    s_wins = sorted((int(a), int(b)) for a, b in wins)
                    w_close = s_wins[w_idx][1] if w_idx < len(s_wins) else s_wins[-1][1]
                    w_margin = w_close - ready
                else:
                    w_margin = int(op.due_step) - int(op.proc_steps) - ready
                profile[op.op_id] = {
                    "K": k_val,
                    "P": p_val,
                    "R": r_val,
                    "S": st_s,
                    "end": en_s,
                    "win_idx": w_idx,
                    "win_margin": int(w_margin),
                    "tard": max(0, en_s - int(op.due_step)),
                }
                op_end[op.op_id] = en_s
                bay_free[bay] = en_s
                crew_free[crew] = en_s
                tool_free[int(op.tool_id)] = en_s
            resolved.add(op.op_id)
    return profile


def _op_critical_task_and_agv(
    snapshot: NativeSnapshot,
    op_id: str,
    assign: dict[str, str],
) -> tuple[Any, Optional[str]]:
    """Return latest-remaining kit task snapshot and its currently assigned/candidate AGV."""
    op = next((o for o in snapshot.operations if o.op_id == op_id), None)
    if op is None:
        return None, None
    tmap = {t.task_id: t for t in snapshot.tasks}
    rem = [
        tmap[tid]
        for tid in op.kit_task_ids
        if tid in tmap and tmap[tid].status.lower() != "completed"
    ]
    if not rem:
        return None, None
    rem.sort(key=lambda t: (0 if t.status.lower() == "picked" else 1, t.release_step, t.task_id))
    t_crit = rem[0]
    aid = t_crit.assigned_agv or assign.get(t_crit.task_id)
    return t_crit, aid


def _has_alternate_route(
    network: Network,
    start: str,
    goal: str,
    avoid_nodes: set[str],
) -> bool:
    """Return True iff `goal` remains reachable from `start` when blocking `avoid_nodes`."""
    if not avoid_nodes or start == goal:
        return False
    clean_avoid = set(avoid_nodes) - {start, goal}
    if not clean_avoid:
        return False
    return _can_reach_avoiding(network, start, goal, clean_avoid)


def select_coupled_op_pair(
    network: Network,
    snapshot: NativeSnapshot,
    base_eval: EvalResult,
    assign: dict[str, str],
    base_prio: dict[str, int],
) -> Optional[tuple[str, str, dict[str, Any]]]:
    """Select the top-2 unstarted operations `(op_A, op_B)` coupled via bottleneck paths or resources."""
    if not base_eval.valid:
        return None
    prof = compute_op_readiness_profile(snapshot, base_eval, op_prio=base_prio)
    unstarted = [
        op
        for op in snapshot.operations
        if not op.started
        and _op_critical_task_and_agv(snapshot, op.op_id, assign)[0] is not None
    ]
    if len(unstarted) < 2:
        return None

    best_score: Optional[tuple[float, str, str]] = None
    best_meta: dict[str, Any] = {}
    for i in range(len(unstarted)):
        for j in range(i + 1, len(unstarted)):
            oa, ob = unstarted[i], unstarted[j]
            ta, ra = _op_critical_task_and_agv(snapshot, oa.op_id, assign)
            tb, rb = _op_critical_task_and_agv(snapshot, ob.op_id, assign)
            if ta is None or tb is None or not ra or not rb:
                continue
            pa = base_eval.robot_paths.get(ra, [])
            pb = base_eval.robot_paths.get(rb, [])
            shared_bt = (
                _identify_shared_bottleneck_nodes(
                    network, pa, pb, {ta.dropoff, tb.dropoff}
                )
                if ra != rb
                else []
            )
            same_agv = 1.0 if ra == rb else 0.0
            res_coup = 1.0 if (oa.bay_id == ob.bay_id or oa.crew_id == ob.crew_id or oa.tool_id == ob.tool_id) else 0.0
            win_flag = 1.0 if (getattr(oa, "resource_windows", ()) or getattr(ob, "resource_windows", ())) else 0.0
            m_a = abs(float(prof.get(oa.op_id, {}).get("win_margin", 999)))
            m_b = abs(float(prof.get(ob.op_id, {}).get("win_margin", 999)))
            cliff_bonus = 2.0 if (min(m_a, m_b) <= 30.0 and win_flag > 0.5) else 0.0
            urgency_term = 1.0 / (1.0 + 0.05 * min(m_a, m_b))
            coupling = (
                3.0 * min(1.0, len(shared_bt) / 2.0)
                + 2.5 * same_agv
                + 1.5 * res_coup
                + 1.5 * win_flag
                + cliff_bonus
                + urgency_term
            )
            key = (coupling, oa.op_id, ob.op_id)
            if best_score is None or key > best_score:
                best_score = key
                best_meta = {
                    "coupling_raw": float(coupling),
                    "shared_bottleneck": shared_bt,
                    "same_agv": bool(ra == rb),
                    "shared_resource": bool(res_coup > 0.5),
                    "has_windows": bool(win_flag > 0.5),
                    "agv_A": ra,
                    "agv_B": rb,
                }
    if best_score is None:
        return None
    _, op_a, op_b = best_score
    return op_a, op_b, best_meta


def compute_pairwise_payoff(
    network: Network,
    snapshot: NativeSnapshot,
    op_a: str,
    op_b: str,
    cand_eval: EvalResult,
    base_eval: EvalResult,
    base_assign: dict[str, str],
    cand_assign: dict[str, str],
    base_prio: dict[str, int],
    cand_prio: dict[str, int],
    cand_bays: dict[str, str],
    cand_crews: dict[str, int],
    *,
    pair_meta: Optional[dict[str, Any]] = None,
    reversible: bool = True,
) -> np.ndarray:
    """Compute 14-D Pairwise Window Payoff Vector comparing candidate `c` against incumbent `d`."""
    if not cand_eval.valid or not base_eval.valid:
        return np.zeros((V17_PAIRWISE_DIM,), dtype=np.float32)

    p_d = compute_op_readiness_profile(snapshot, base_eval, op_prio=base_prio)
    p_c = compute_op_readiness_profile(
        snapshot,
        cand_eval,
        op_prio=cand_prio,
        op_bay_overrides=cand_bays,
        op_crew_overrides=cand_crews,
    )
    da, ca = p_d.get(op_a, {}), p_c.get(op_a, {})
    db, cb = p_d.get(op_b, {}), p_c.get(op_b, {})

    ds_a = float(ca.get("S", 0) - da.get("S", 0)) / 40.0
    ds_b = float(cb.get("S", 0) - db.get("S", 0)) / 40.0
    dt_a = float(ca.get("tard", 0) - da.get("tard", 0)) / 40.0
    dt_b = float(cb.get("tard", 0) - db.get("tard", 0)) / 40.0

    wi_da, wi_ca = int(da.get("win_idx", 0)), int(ca.get("win_idx", 0))
    wi_db, wi_cb = int(db.get("win_idx", 0)), int(cb.get("win_idx", 0))
    cross_a = 1.0 if wi_ca > wi_da else (-1.0 if wi_ca < wi_da else 0.0)
    cross_b = 1.0 if wi_cb > wi_db else (-1.0 if wi_cb < wi_db else 0.0)

    worst_loss = max(0.0, dt_a, dt_b)

    ta, ra = _op_critical_task_and_agv(snapshot, op_a, cand_assign)
    tb, rb = _op_critical_task_and_agv(snapshot, op_b, cand_assign)
    loaded_a = 1.0 if (ta is not None and ta.status.lower() == "picked") else 0.0
    loaded_b = 1.0 if (tb is not None and tb.status.lower() == "picked") else 0.0

    shared = set((pair_meta or {}).get("shared_bottleneck") or ())
    robs_by_id = {r.robot_id: r for r in snapshot.robots}
    alt_a = 0.0
    alt_b = 0.0
    if shared and ra and ra in robs_by_id and ta is not None:
        alt_a = 1.0 if _has_alternate_route(network, robs_by_id[ra].cur_node, ta.dropoff, shared) else 0.0
    if shared and rb and rb in robs_by_id and tb is not None:
        alt_b = 1.0 if _has_alternate_route(network, robs_by_id[rb].cur_node, tb.dropoff, shared) else 0.0

    churn_d = float(sum(1 for tid, r in cand_assign.items() if base_assign.get(tid) != r)) / 4.0
    coup_norm = min(1.0, float((pair_meta or {}).get("coupling_raw", 0.0)) / 6.0)

    return np.asarray(
        [
            float(np.clip(ds_a, -5.0, 5.0)),
            float(np.clip(ds_b, -5.0, 5.0)),
            float(np.clip(dt_a, -5.0, 5.0)),
            float(np.clip(dt_b, -5.0, 5.0)),
            cross_a,
            cross_b,
            float(np.clip(worst_loss, 0.0, 5.0)),
            loaded_a,
            loaded_b,
            alt_a,
            alt_b,
            1.0 if reversible else 0.0,
            min(2.0, churn_d),
            coup_norm,
        ],
        dtype=np.float32,
    )


def generate_v17_dual_op_candidates(
    policy_owner: Any,
    network: Network,
    snapshot: NativeSnapshot,
    clf: CommitmentClassification,
    evaluator: CVCREvaluator,
    surr: Any,
    base_assign: dict[str, str],
    base_bays: dict[str, str],
    base_crews: dict[str, int],
    base_eval: EvalResult,
    base_prio: dict[str, int],
    healthy_robots: list[str],
    bay_delta: dict[str, float],
    tw_seed: dict[tuple[str, str], int],
    v15_pol: V15ExecutionAwarePolicy,
    deadline: float,
    budget_ms: float,
    now_fn: Any,
    *,
    sim: Any = None,
    include_dual_op: bool = True,
) -> tuple[list[V15Candidate], np.ndarray, dict[str, Any]]:
    """Generate shared-budget `C0..C4` + Dual-Op Window Protection candidates with Action-Diff dedup."""
    from amr_sched.cvcr.v12_proactive_guard import compute_kitting_marginal_start_value
    from amr_sched.cvcr.v16_frontier import generate_v16_unified_candidates

    cands, g_feats = generate_v16_unified_candidates(
        policy_owner,
        network,
        snapshot,
        clf,
        evaluator,
        surr,
        base_assign,
        base_bays,
        base_crews,
        base_eval,
        base_prio,
        healthy_robots,
        bay_delta,
        tw_seed,
        v15_pol,
        deadline,
        budget_ms,
        now_fn,
    )
    pair = select_coupled_op_pair(network, snapshot, base_eval, base_assign, base_prio)
    op_a = pair[0] if pair is not None else ""
    op_b = pair[1] if pair is not None else ""
    pair_meta = pair[2] if pair is not None else {}

    km_vals = compute_kitting_marginal_start_value(snapshot)
    km_top_op = ""
    if km_vals:
        best_t = min(km_vals.keys(), key=lambda tid: km_vals[tid])
        t_obj = next((t for t in snapshot.tasks if t.task_id == best_t), None)
        if t_obj is not None and t_obj.kit_op_id:
            km_top_op = str(t_obj.kit_op_id)

    cutoff_base = int(snapshot.tick) + 45 if tw_seed else 0
    tw_active = dict(tw_seed) if tw_seed else {}

    # Tag baseline candidates with pairwise payoff + action diff digest
    seen_digests: set[str] = set()
    deduped: list[V15Candidate] = []
    c0_c4_raw = len(cands)

    for c in cands:
        dig = compute_action_diff_digest(
            c,
            base_assign,
            base_bays,
            base_crews,
            base_prio,
            live_repairs=getattr(c, "live_repairs", None),
        )
        c.action_diff_digest = dig
        if op_a and op_b:
            c.pairwise_features = compute_pairwise_payoff(
                network,
                snapshot,
                op_a,
                op_b,
                c.eval_res,
                base_eval,
                base_assign,
                c.assignment,
                base_prio,
                c.op_priority,
                c.bay_overrides,
                c.crew_overrides,
                pair_meta=pair_meta,
                reversible=True,
            )
        else:
            c.pairwise_features = np.zeros((V17_PAIRWISE_DIM,), dtype=np.float32)
        if dig not in seen_digests:
            seen_digests.add(dig)
            deduped.append(c)

    dual_merged = 0
    dual_added = 0
    dual_attempted = 0
    dual_infeasible = 0

    # Anchored joint candidate to layer global kitting-marginal priority & dual-op pair protection cleanly
    valid_c04 = [c for c in deduped if c.eval_res.valid]
    anchor_c = max(valid_c04, key=lambda c: c.det_score) if valid_c04 else None
    anc_assign = dict(anchor_c.assignment) if anchor_c is not None else dict(base_assign)
    anc_bays = dict(anchor_c.bay_overrides) if anchor_c is not None else dict(base_bays)
    anc_crews = dict(anchor_c.crew_overrides) if anchor_c is not None else dict(base_crews)
    anc_tw = dict(anchor_c.traffic_weights) if (anchor_c is not None and anchor_c.traffic_weights) else tw_active
    anc_cut = int(anchor_c.traffic_cutoff) if anchor_c is not None else cutoff_base
    anc_h = int(anchor_c.commit_seq_horizon) if anchor_c is not None else 0

    # R1: Global Kitting-Marginal Start Value op_priority candidate (unifies rule/production urgency into CVCREvaluator)
    if km_vals and policy_owner._has_budget(deadline, evaluator, min_ms=2.0):
        km_prio: dict[str, int] = dict(base_prio)
        for op_obj in snapshot.operations:
            if not op_obj.started:
                w_margin_bias = -300 if getattr(op_obj, "resource_windows", ()) else 0
                best_k = min(
                    (km_vals[kt][0] * 1000 + km_vals[kt][1] * 200 + km_vals[kt][4] + w_margin_bias for kt in op_obj.kit_task_ids if kt in km_vals),
                    default=int(base_prio.get(op_obj.op_id, op_obj.due_step)),
                )
                km_prio[op_obj.op_id] = int(best_k)
        if km_prio != base_prio:
            evaluator.set_traffic_weights(anc_tw or None, cutoff_step=anc_cut)
            ev_km = evaluator.evaluate(
                anc_assign,
                op_priority=km_prio,
                op_bay_overrides=anc_bays,
                op_crew_overrides=anc_crews,
            )
            if ev_km.valid:
                rem_now = max(0.0, min(1.0, (deadline - now_fn()) / max(0.01, float(budget_ms) / 1000.0)))
                feat_km, ds_km = _build_candidate_features(
                    "v10_crit_ci_balance",
                    ev_km,
                    base_eval,
                    anc_assign,
                    base_assign,
                    anc_bays,
                    anc_crews,
                    snapshot,
                    anc_h,
                    bool(anchor_c.cliff_saved) if anchor_c is not None else False,
                    v15_pol.family_ucb_bonus("v10_crit_ci_balance"),
                    rem_now,
                    g_feats,
                )
                c_km = V15Candidate(
                    cand_id="R1_km_prio",
                    source_kind="v10_crit_ci_balance",
                    assignment=dict(anc_assign),
                    bay_overrides=dict(anc_bays),
                    crew_overrides=dict(anc_crews),
                    op_priority=km_prio,
                    eval_res=ev_km,
                    traffic_weights=dict(anc_tw),
                    traffic_cutoff=int(anc_cut),
                    guide_routes={},
                    commit_seq_horizon=anc_h,
                    destroy_tids=tuple(anchor_c.destroy_tids) if anchor_c is not None else (),
                    cliff_saved=bool(anchor_c.cliff_saved) if anchor_c is not None else False,
                    det_score=float(ds_km),
                    features=feat_km,
                    radius_used=int(anchor_c.radius_used) if anchor_c is not None else 999,
                )
                dig_km = compute_action_diff_digest(c_km, base_assign, base_bays, base_crews, base_prio)
                c_km.action_diff_digest = dig_km
                if op_a and op_b:
                    c_km.pairwise_features = compute_pairwise_payoff(
                        network, snapshot, op_a, op_b, ev_km, base_eval, base_assign, anc_assign, base_prio, km_prio, anc_bays, anc_crews, pair_meta=pair_meta, reversible=True
                    )
                else:
                    c_km.pairwise_features = np.zeros((V17_PAIRWISE_DIM,), dtype=np.float32)
                if dig_km not in seen_digests:
                    seen_digests.add(dig_km)
                    deduped.append(c_km)

    if include_dual_op and pair is not None and policy_owner._has_budget(deadline, evaluator, min_ms=2.0):
        # Enumerate {protect_A, protect_B, legal_detour} anchored on best joint assignment (preserving C1..C4 gains)
        plans_to_try: list[tuple[str, str, BridgeOrderControlPlan, dict[tuple[str, str], int]]] = []
        evaluator.set_traffic_weights(anc_tw or None, cutoff_step=anc_cut)
        for cid, ldr, yld in (("D1_protect_A", op_a, op_b), ("D2_protect_B", op_b, op_a)):
            dual_attempted += 1
            bp_plan = plan_bridge_order_control(
                network,
                snapshot,
                evaluator,
                anc_assign,
                base_prio,
                ldr,
                yld,
                sim=sim,
                base_bays=anc_bays,
                base_crews=anc_crews,
            )
            if bp_plan.eligible and bp_plan.eval_res is not None and bp_plan.eval_res.valid:
                plans_to_try.append((cid, "selective_cliff_rescue", bp_plan, anc_tw))
            else:
                dual_infeasible += 1

        # D3_legal_detour: penalize shared bottleneck edges so yielder detours onto alternate corridor if legal
        shared_bt = pair_meta.get("shared_bottleneck") or []
        if shared_bt and policy_owner._has_budget(deadline, evaluator, min_ms=2.0):
            det_tw = dict(anc_tw)
            for nd_s in shared_bt:
                for e in network.out_edges(nd_s):
                    det_tw[(e.src, e.dst)] = max(det_tw.get((e.src, e.dst), 0), 32)
                for nd_u in network.nodes:
                    if network.has_edge(nd_u, nd_s):
                        det_tw[(nd_u, nd_s)] = max(det_tw.get((nd_u, nd_s), 0), 32)
            evaluator.set_traffic_weights(det_tw, cutoff_step=int(snapshot.tick) + 60)
            prof = compute_op_readiness_profile(snapshot, base_eval, op_prio=base_prio)
            m_a = prof.get(op_a, {}).get("win_margin", 999)
            m_b = prof.get(op_b, {}).get("win_margin", 999)
            ldr, yld = (op_a, op_b) if m_a <= m_b else (op_b, op_a)
            dual_attempted += 1
            det_plan = plan_bridge_order_control(
                network,
                snapshot,
                evaluator,
                anc_assign,
                base_prio,
                ldr,
                yld,
                sim=sim,
                base_bays=anc_bays,
                base_crews=anc_crews,
            )
            evaluator.set_traffic_weights(anc_tw or None, cutoff_step=anc_cut)
            if det_plan.eligible and det_plan.eval_res is not None and det_plan.eval_res.valid:
                plans_to_try.append(("D3_legal_detour", "fast_detour_regret", det_plan, det_tw))
            else:
                dual_infeasible += 1

        for cid, skind, pl, tw_use in plans_to_try:
            assert pl.eval_res is not None
            rem_now = max(0.0, min(1.0, (deadline - now_fn()) / max(0.01, float(budget_ms) / 1000.0)))
            feat, ds = _build_candidate_features(
                skind,
                pl.eval_res,
                base_eval,
                anc_assign,
                base_assign,
                anc_bays,
                anc_crews,
                snapshot,
                anc_h,
                bool(anchor_c.cliff_saved) if anchor_c is not None else False,
                v15_pol.family_ucb_bonus(skind),
                rem_now,
                g_feats,
            )
            cand_d = V15Candidate(
                cand_id=cid,
                source_kind=skind,
                assignment=dict(anc_assign),
                bay_overrides=dict(anc_bays),
                crew_overrides=dict(anc_crews),
                op_priority=dict(pl.op_priority),
                eval_res=pl.eval_res,
                traffic_weights=dict(tw_use),
                traffic_cutoff=int(snapshot.tick) + 60 if tw_use else anc_cut,
                guide_routes={},
                commit_seq_horizon=anc_h,
                destroy_tids=tuple(anchor_c.destroy_tids) if anchor_c is not None else (),
                cliff_saved=bool(anchor_c.cliff_saved) if anchor_c is not None else False,
                det_score=float(ds),
                features=feat,
                radius_used=int(anchor_c.radius_used) if anchor_c is not None else 999,
                live_repairs=pl.live_repairs,
                bottleneck_locks=pl.bottleneck_locks,
            )
            dig = compute_action_diff_digest(
                cand_d,
                base_assign,
                base_bays,
                base_crews,
                base_prio,
                live_repairs=pl.live_repairs,
            )
            cand_d.action_diff_digest = dig
            cand_d.pairwise_features = compute_pairwise_payoff(
                network,
                snapshot,
                op_a,
                op_b,
                pl.eval_res,
                base_eval,
                base_assign,
                cand_d.assignment,
                base_prio,
                cand_d.op_priority,
                cand_d.bay_overrides,
                cand_d.crew_overrides,
                pair_meta=pair_meta,
                reversible=True,
            )
            if dig in seen_digests:
                dual_merged += 1
            else:
                seen_digests.add(dig)
                deduped.append(cand_d)
                dual_added += 1

    meta = {
        "op_a": op_a,
        "op_b": op_b,
        "km_top_op": km_top_op,
        "km_overlap_pair": bool(km_top_op and km_top_op in (op_a, op_b)),
        "c0_c4_raw": c0_c4_raw,
        "c0_c4_deduped": len(deduped) - dual_added,
        "dual_attempted": dual_attempted,
        "dual_added_distinct": dual_added,
        "dual_merged_with_existing": dual_merged,
        "dual_infeasible": dual_infeasible,
        "pair_meta": pair_meta,
    }
    return deduped, g_feats, meta


V17_CAND_FEAT_DIM = 36 + V17_PAIRWISE_DIM  # 50
DEFAULT_V17_CKPT = resolve_default_checkpoint("v17_window_protection.npz")


def build_v17_cand_matrix(cands: list[V15Candidate]) -> np.ndarray:
    """Assemble `[N, 50]` candidate feature matrix concatenating 36-D structural + 14-D pairwise payoff."""
    rows: list[np.ndarray] = []
    for c in cands:
        pw = getattr(c, "pairwise_features", None)
        pw_vec = np.asarray(pw, dtype=np.float32)[:V17_PAIRWISE_DIM] if pw is not None else np.zeros((V17_PAIRWISE_DIM,), dtype=np.float32)
        if len(pw_vec) < V17_PAIRWISE_DIM:
            pw_vec = np.pad(pw_vec, (0, V17_PAIRWISE_DIM - len(pw_vec)))
        rows.append(np.concatenate([np.asarray(c.features, dtype=np.float32), pw_vec], axis=0))
    return np.stack(rows, axis=0)


class V17WindowProtectionPolicy(V16FrontierPolicy):
    """Pure-NumPy V17 policy unifying causal router, pairwise window ranker, and v7 restraint."""

    def __init__(
        self,
        weights: Optional[dict[str, np.ndarray]] = None,
        *,
        conservative_gate: float = 1.5,
        zero_keep_residual: bool = True,
        allow_probe_override: bool = False,
        router_mode: str = "learned",
        ranker_mode: str = "pairwise",
        enable_restraint: bool = True,
    ) -> None:
        """Initialize V17 policy with explicit Stage-3 attribution toggles."""
        super().__init__(
            weights=weights,
            conservative_gate=conservative_gate,
            zero_keep_residual=zero_keep_residual,
            allow_probe_override=allow_probe_override,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
        )
        self.router_mode = str(router_mode)
        self.ranker_mode = str(ranker_mode)
        self.enable_restraint = bool(enable_restraint)

    def select_portfolio_mode(self, ctx14: np.ndarray, *, profile: str = "quality") -> str:
        """Select coherent solver mode using either causal learned router or deterministic structural rule."""
        has_res_win = float(ctx14[0]) > 0.5
        if has_res_win:
            return "V15_SOFT"
        if profile == "balanced":
            return "V15_SOFT"
        if profile == "calm":
            return "V7_CALM"

        if self.router_mode == "learned" and self.w is not None and "v17_mode_w1" in self.w:
            x = np.clip((ctx14[:V16_CTX_DIM] - self.w["v17_m_mu"]) / np.maximum(0.05, self.w["v17_m_sd"]), -6.0, 6.0)
            h = _relu(x @ self.w["v17_mode_w1"] + self.w["v17_mode_b1"])
            logits = h @ self.w["v17_mode_w2"] + self.w["v17_mode_b2"]
            return V16_MODES[int(np.argmax(logits))]

        if self.router_mode == "learned":
            return super().select_portfolio_mode(ctx14, profile=profile)

        # Deterministic structural rule router (pure heuristic without mode_w1 weights)
        n_preds_norm = float(ctx14[1])
        max_kit_norm = float(ctx14[2])
        has_drift = float(ctx14[4]) > 0.5
        fault_ratio = float(ctx14[5])
        tasks_per_agv = float(ctx14[7])
        bay_dd_p90 = float(ctx14[9])
        if n_preds_norm <= 0.05 and not has_drift:
            return "V10_GUIDED"
        if has_drift or max_kit_norm >= 0.75 or bay_dd_p90 > 0.15 or fault_ratio > 0.01 or tasks_per_agv >= 0.85:
            return "V7_CALM"
        return "V15_SOFT"

    def score_candidates(self, g_feats: np.ndarray, cands: list[V15Candidate]) -> np.ndarray:
        """Score candidates under configured `ranker_mode` (`pairwise` vs `det` vs `v15`)."""
        if not cands:
            return np.zeros((0,), dtype=np.float32)
        det_vec = np.asarray([c.det_score for c in cands], dtype=np.float32)
        if self.ranker_mode == "det":
            for i, c in enumerate(cands):
                if not c.eval_res.valid:
                    det_vec[i] = -1e9
            return det_vec

        if self.ranker_mode == "pairwise" and self.w is not None and "v17_cand_w1" in self.w:
            w = self.w
            mat = build_v17_cand_matrix(cands)
            x = np.clip((mat - w["v17_c_mu"]) / np.maximum(0.05, w["v17_c_sd"]), -6.0, 6.0)
            h = _relu(x @ w["v17_cand_w1"] + w["v17_cand_b1"])
            q = h @ w["v17_att_q"]
            k = h @ w["v17_att_k"]
            v = h @ w["v17_att_v"]
            att = _softmax((q @ k.T) / np.sqrt(float(q.shape[-1])))
            h_att = _relu(np.concatenate([h, att @ v], axis=-1) @ w["v17_cand_w2"] + w["v17_cand_b2"])
            delta = (h_att @ w["v17_cand_w3"] + w["v17_cand_b3"]).reshape(-1)

            g_norm = np.clip((g_feats[:V15_GLOBAL_FEAT_DIM] - w["v17_g_mu"]) / np.maximum(0.05, w["v17_g_sd"]), -6.0, 6.0)
            hg = _relu(g_norm @ w["v17_g_w1"] + w["v17_g_b1"])
            trust_gain = float(_sigmoid((hg @ w["v17_g_w2"] + w["v17_g_b2"]).reshape(-1)[0])) * 2.0 * self.conservative_gate

            adj = np.tanh(delta / 2.0) * trust_gain
            if getattr(self, "zero_keep_residual", True):
                for i, c in enumerate(cands):
                    if c.source_kind == "keep_incumbent":
                        adj[i] = 0.0
            scores = det_vec + adj
            for i, c in enumerate(cands):
                if not c.eval_res.valid:
                    scores[i] = -1e9
            return scores.astype(np.float32)

        return super().score_candidates(g_feats, cands)


def load_v17_window_policy(
    path: Optional[str | Path] = None,
    *,
    router_mode: str = "learned",
    ranker_mode: str = "pairwise",
    enable_restraint: bool = True,
    allow_fallback: bool = False,
) -> V17WindowProtectionPolicy:
    """Load `V17WindowProtectionPolicy` strictly from V17 checkpoint unless `allow_fallback=True`."""
    p = Path(path) if path is not None else DEFAULT_V17_CKPT
    if p.exists():
        weights = _load_cached_npz_weights(p)
        return V17WindowProtectionPolicy(
            weights=weights,
            router_mode=router_mode,
            ranker_mode=ranker_mode,
            enable_restraint=enable_restraint,
        )
    if not allow_fallback:
        raise FileNotFoundError(f"Required V17 checkpoint not found: {p}")
    if DEFAULT_V16_CKPT.exists():
        weights = _load_cached_npz_weights(DEFAULT_V16_CKPT)
        return V17WindowProtectionPolicy(
            weights=weights,
            router_mode=router_mode,
            ranker_mode=ranker_mode,
            enable_restraint=enable_restraint,
        )
    return V17WindowProtectionPolicy(
        weights=None,
        router_mode=router_mode,
        ranker_mode=ranker_mode,
        enable_restraint=enable_restraint,
    )


class V17AdaptiveController(V16AdaptiveController):
    """Stage-3 causal controller supporting explicit router/ranker/restraint attribution toggles."""

    def __init__(
        self,
        *,
        epoch_interval: int = 16,
        stall_threshold: int = 20,
        policy: Optional[V17WindowProtectionPolicy] = None,
        trigger: Optional[Any] = None,
        profile: str = "quality",
        enable_restraint: Optional[bool] = None,
    ) -> None:
        """Initialize V17 controller preserving `v7` restraint control and causal context."""
        pol = policy if policy is not None else load_v17_window_policy()
        if enable_restraint is not None:
            pol.enable_restraint = bool(enable_restraint)
        super().__init__(
            epoch_interval=epoch_interval,
            stall_threshold=stall_threshold,
            policy=pol,
            trigger=trigger,
            profile=profile,
        )

    def select_action(self, obs_v12: np.ndarray, info: dict[str, Any], sim: Any = None) -> int:
        """Select V17 portfolio action causally, preserving surgical stall safety across all modes."""
        from amr_sched.rl.spaces import is_in_secondary_hold

        mask = info["action_mask"]
        tick = int(info.get("step", 0))
        if sim is not None and is_in_secondary_hold(sim):
            return ACTION_KEEP
        surg = self.check_surgical_repair_action(mask, tick, sim)
        if surg is not None:
            if isinstance(self.policy, V16FrontierPolicy):
                self.policy.active_subscope = None
            return surg
        return super().select_action(obs_v12, info, sim=sim)


