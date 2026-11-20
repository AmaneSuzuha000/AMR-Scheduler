"""Native Transaction Owner for LG-CVCR / AMR-Scheduler.

Single-writer transaction boundary protecting simulation integrity:
1. Validates proposal freshness against simulator step version.
2. Enforces hard commitment preservation (picked/completed tasks cannot mutate AGV or dropoff).
3. Validates admitted fleet consistency (no picked task belongs to an unadmitted AGV).
4. Validates directed edge reachability along proposed routes.
5. Atomically commits admission gates, pinned assignments, production bay/crew updates,
   and task dropoff synchronization, or rolls back with an explicit audit receipt.
6. Strictly enforces software simulation mode (physical authorization is always False).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional

from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.cvcr.snapshot import NativeSnapshotCapture, OperationSnapshot
from amr_sched.entities import AgvState, TaskStatus
from amr_sched.reservation import Interval
from amr_sched.simulation import Hop, _Job


@dataclass
class TransactionReceipt:
    """Immutable commit/rejection receipt emitted by NativeTransactionOwner."""

    schema: str
    accepted: bool
    reasons: list[str]
    committed_tick: int
    snapshot_digest: str
    candidate_digest: str
    receipt_digest: str
    physical_authorization: bool = False  # Always False in software simulation


class NativeTransactionOwner:
    """Manages transactional validation & atomic commit of CVCR proposals into Simulator."""

    def __init__(
        self,
        simulator: Any,
        operations: Optional[list[OperationSnapshot]] = None,
        production_tracker: Optional[Any] = None,
        allow_physical_mode: bool = False,
    ) -> None:
        self.simulator = simulator
        self.production_tracker = production_tracker
        self.operations = operations or []
        self.allow_physical_mode = bool(allow_physical_mode)

    def _reject(self, current_tick: int, proposal: CandidateProposal, reasons: list[str]) -> TransactionReceipt:
        payload = {
            "accepted": False,
            "reasons": reasons,
            "tick": current_tick,
            "candidate_digest": proposal.candidate_digest,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return TransactionReceipt(
            schema="amr-lgcvcr.transaction-receipt.v1",
            accepted=False,
            reasons=reasons,
            committed_tick=current_tick,
            snapshot_digest=proposal.snapshot_digest,
            candidate_digest=proposal.candidate_digest,
            receipt_digest=digest,
            physical_authorization=False,
        )

    def validate_and_commit(
        self,
        proposal: CandidateProposal,
        *,
        execution_feedback: Any = None,
    ) -> TransactionReceipt:
        """Validate snapshot freshness, commitments, admission & routes, then commit atomically."""
        current_tick = int(self.simulator._step_cursor)
        reasons: list[str] = []

        # Never grant physical authorization for unconstrained SW edits unless explicit physical coordination gate enabled
        execs = getattr(self.simulator, "executions", None) or {}
        handlings = getattr(self.simulator, "handlings", None) or {}
        phys_locked: set[str] = set()
        if execs:
            if not self.allow_physical_mode:
                reasons.append("PHYSICAL_MODE_FORBIDDEN_FOR_SW_TRANSACTION")
            else:
                transit_p = getattr(self.simulator, "_transit", {})
                load_p = getattr(self.simulator, "_load_end", {})
                drop_p = getattr(self.simulator, "_drop_end", {})
                detours_p = getattr(self.simulator, "_physical_detours", {})
                preempts_p = getattr(self.simulator, "_physical_preemptions", {})
                for aid_p, gate_p in execs.items():
                    h_p = handlings.get(aid_p)
                    h_act = bool(
                        h_p is not None
                        and (getattr(h_p, "command", None) is not None or getattr(h_p, "in_progress", False))
                    )
                    if (
                        (aid_p in transit_p)
                        or (getattr(gate_p, "command", None) is not None)
                        or (aid_p in detours_p)
                        or (aid_p in preempts_p)
                        or (int(load_p.get(aid_p, 0)) > current_tick)
                        or (int(drop_p.get(aid_p, 0)) > current_tick)
                        or h_act
                    ):
                        phys_locked.add(aid_p)

        eff_fb = execution_feedback
        if (
            eff_fb is None
            and getattr(self.simulator, "_p2_execution_aware", False)
            and getattr(self.simulator, "_perturbation_model", None) is not None
        ):
            eff_fb = self.simulator._perturbation_model.capture_feedback(self.simulator, current_tick)

        snap_full = NativeSnapshotCapture.capture(
            self.simulator,
            self.operations,
            hide_future_releases=False,
            execution_feedback=eff_fb,
        )
        snap_causal = NativeSnapshotCapture.capture(
            self.simulator,
            self.operations,
            hide_future_releases=True,
            execution_feedback=eff_fb,
        )
        if snap_full.tick != current_tick or (
            proposal.snapshot_digest != snap_full.compute_digest()
            and proposal.snapshot_digest != snap_causal.compute_digest()
        ):
            reasons.append("STALE_SNAPSHOT_DIGEST_MISMATCH")

        snap = snap_full
        clf = classify_commitments(snap)
        tasks_by_id = {t.task_id: t for t in self.simulator.tasks}
        ops_by_id = {op.op_id: op for op in self.operations}
        agvs_by_id = {a.agv_id: a for a in self.simulator.agvs}
        known_agvs = set(agvs_by_id.keys())
        faulted = getattr(self.simulator, "_faulted_agvs", set())

        # Reject unknown task/robot references before any state change
        for tid, agv_id in proposal.proposed_assignment.items():
            if tid not in tasks_by_id:
                reasons.append(f"UNKNOWN_TASK_ID:{tid}")
            if agv_id not in known_agvs:
                reasons.append(f"UNKNOWN_ROBOT_ID:{agv_id}")
        if proposal.proposed_admitted_agvs is not None:
            for agv_id in proposal.proposed_admitted_agvs:
                if agv_id not in known_agvs:
                    reasons.append(f"UNKNOWN_ADMITTED_ROBOT_ID:{agv_id}")
        for agv_id in proposal.proposed_routes:
            if agv_id not in known_agvs:
                reasons.append(f"UNKNOWN_ROUTE_ROBOT_ID:{agv_id}")
        if proposal.proposed_task_sequences is not None:
            for agv_id, seq in proposal.proposed_task_sequences.items():
                if agv_id not in known_agvs:
                    reasons.append(f"UNKNOWN_QUEUE_ROBOT_ID:{agv_id}")
                for tid in seq:
                    if tid not in tasks_by_id:
                        reasons.append(f"UNKNOWN_QUEUE_TASK_ID:{tid}")
        if proposal.proposed_traffic_weights is not None:
            net_chk = self.simulator.network
            for src_n, dst_n in proposal.proposed_traffic_weights:
                if not net_chk.has_edge(src_n, dst_n):
                    reasons.append(f"UNKNOWN_TRAFFIC_EDGE:{src_n}->{dst_n}")
        if proposal.proposed_guide_routes is not None:
            net_chk = self.simulator.network
            for agv_id, elist in proposal.proposed_guide_routes.items():
                if agv_id not in known_agvs:
                    reasons.append(f"UNKNOWN_GUIDE_ROBOT_ID:{agv_id}")
                for src_n, dst_n in elist:
                    if not net_chk.has_edge(src_n, dst_n):
                        reasons.append(f"UNKNOWN_GUIDE_EDGE:{agv_id}:{src_n}->{dst_n}")

        # Hard task commitment preservation
        for tid in clf.frozen_tasks:
            orig_task = tasks_by_id.get(tid)
            if orig_task and orig_task.assigned_agv is not None:
                new_assigned = proposal.proposed_assignment.get(tid)
                if new_assigned is not None and new_assigned != orig_task.assigned_agv:
                    reasons.append(
                        f"VIOLATION_IRREVOCABLE_TASK_MUTATION:{tid}:{orig_task.assigned_agv}->{new_assigned}"
                    )

        # Admitted fleet must include every AGV currently holding a PICKED/EXECUTING_HARD task
        if proposal.proposed_admitted_agvs is not None:
            admitted_set = set(proposal.proposed_admitted_agvs)
            for tid in clf.frozen_tasks:
                t = tasks_by_id.get(tid)
                if t and t.status == TaskStatus.PICKED and t.assigned_agv and t.assigned_agv not in admitted_set:
                    reasons.append(f"VIOLATION_UNADMITTED_ACTIVE_ROBOT:{t.assigned_agv}:{tid}")

        if getattr(proposal, "proposed_op_priorities", None) is not None:
            for op_id, prio_val in proposal.proposed_op_priorities.items():
                if op_id not in ops_by_id:
                    reasons.append(f"UNKNOWN_OPERATION_ID_PRIO:{op_id}")
                elif isinstance(prio_val, bool) or not isinstance(prio_val, (int, float)):
                    reasons.append(f"INVALID_OP_PRIORITY:{op_id}")

        # Physical-mode safety boundaries: never mutate in-flight/hardware-locked robots or edge pins
        if execs and self.allow_physical_mode:
            pins_chk = self.simulator._res.pins() if hasattr(self.simulator, "_res") else {"edges": {}}
            edge_pin_owners = {o for owners in pins_chk.get("edges", {}).values() for o in owners}
            for tid, new_assigned in proposal.proposed_assignment.items():
                orig_task = tasks_by_id.get(tid)
                if orig_task and orig_task.assigned_agv is not None and orig_task.assigned_agv != new_assigned:
                    if orig_task.assigned_agv in phys_locked:
                        reasons.append(
                            f"VIOLATION_PHYSICAL_INFLIGHT_MUTATION:{tid}:{orig_task.assigned_agv}->{new_assigned}"
                        )
                    elif orig_task.assigned_agv in edge_pin_owners:
                        reasons.append(
                            f"VIOLATION_PHYSICAL_EDGE_PIN_MUTATION:{tid}:{orig_task.assigned_agv}"
                        )

        b_locks = getattr(proposal, "proposed_bottleneck_locks", None)
        if b_locks is not None:
            net_bl = self.simulator.network
            max_s_bl = int(getattr(self.simulator.config, "max_steps", 200000))
            for aid_bl, spans_bl in b_locks.items():
                if aid_bl not in agvs_by_id:
                    reasons.append(f"UNKNOWN_BOTTLENECK_LOCK_AGV:{aid_bl}")
                elif aid_bl in faulted:
                    reasons.append(f"VIOLATION_BOTTLENECK_LOCK_FAULTED_AGV:{aid_bl}")
                for lock_nd, lo_s, hi_s in spans_bl:
                    if str(lock_nd) not in net_bl.nodes:
                        reasons.append(f"UNKNOWN_BOTTLENECK_LOCK_NODE:{lock_nd}")
                    elif int(hi_s) <= int(lo_s) or int(lo_s) < current_tick or int(hi_s) > max_s_bl:
                        reasons.append(f"INVALID_BOTTLENECK_LOCK_INTERVAL:{aid_bl}:{lock_nd}:{lo_s}->{hi_s}")

        live_repairs = getattr(proposal, "proposed_live_repairs", None)
        if live_repairs and execs and self.allow_physical_mode:
            reasons.append("VIOLATION_PHYSICAL_LIVE_REPAIR_USE_REPLAN_API")
        if live_repairs is not None:
            net_chk = self.simulator.network
            transit_chk = getattr(self.simulator, "_transit", {})
            load_end_chk = getattr(self.simulator, "_load_end", {})
            max_steps_chk = int(getattr(self.simulator.config, "max_steps", 200000))
            for aid, rep in live_repairs.items():
                agv = agvs_by_id.get(aid)
                if agv is None:
                    reasons.append(f"UNKNOWN_LIVE_REPAIR_AGV:{aid}")
                    continue
                if aid in faulted:
                    reasons.append(f"VIOLATION_LIVE_REPAIR_FAULTED_AGV:{aid}")
                    continue
                if aid in transit_chk:
                    reasons.append(f"VIOLATION_LIVE_REPAIR_IN_TRANSIT:{aid}")
                    continue
                kind = str(rep.get("kind", ""))
                hops = rep.get("hops") or []
                if kind not in ("picked_reroute", "assigned_reroute", "idle_park"):
                    reasons.append(f"INVALID_LIVE_REPAIR_KIND:{aid}:{kind}")
                    continue
                if not hops:
                    reasons.append(f"EMPTY_LIVE_REPAIR_HOPS:{aid}")
                    continue
                if kind == "picked_reroute":
                    tid_rep = rep.get("task_id")
                    t_rep = tasks_by_id.get(tid_rep)
                    if (
                        t_rep is None
                        or t_rep.status != TaskStatus.PICKED
                        or t_rep.assigned_agv != aid
                        or agv.cur_task != tid_rep
                        or rep.get("goal") != t_rep.dropoff
                    ):
                        reasons.append(f"VIOLATION_LIVE_REPAIR_COMMITMENT:{aid}:{tid_rep}")
                        continue
                    if int(load_end_chk.get(aid, 0)) > current_tick:
                        reasons.append(f"VIOLATION_LIVE_REPAIR_LOADING_ACTIVE:{aid}")
                        continue
                elif kind == "assigned_reroute":
                    tid_rep = rep.get("task_id")
                    t_rep = tasks_by_id.get(tid_rep)
                    if (
                        t_rep is None
                        or t_rep.status != TaskStatus.ASSIGNED
                        or t_rep.assigned_agv != aid
                        or agv.cur_task != tid_rep
                        or rep.get("goal") != t_rep.pickup
                    ):
                        reasons.append(f"VIOLATION_ASSIGNED_REPAIR_COMMITMENT:{aid}:{tid_rep}")
                        continue
                else:
                    if agv.cur_task is not None:
                        reasons.append(f"VIOLATION_IDLE_PARK_HAS_TASK:{aid}:{agv.cur_task}")
                        continue
                    goal_park = str(rep.get("goal", ""))
                    if not goal_park or hops[-1][1] != goal_park:
                        reasons.append(f"INVALID_IDLE_PARK_GOAL:{aid}:{goal_park}")
                        continue
                cur_chk = agv.cur_node or agv.start_node
                if hops[0][0] != cur_chk:
                    reasons.append(f"DISCONTINUOUS_LIVE_REPAIR_START:{aid}:{cur_chk}!={hops[0][0]}")
                    continue
                prev_arr = current_tick
                for u_r, v_r, dep_r, arr_r in hops:
                    if u_r != cur_chk or not net_chk.has_edge(u_r, v_r):
                        reasons.append(f"ILLEGAL_LIVE_REPAIR_EDGE:{aid}:{u_r}->{v_r}")
                        break
                    if dep_r < prev_arr or arr_r <= dep_r or arr_r >= max_steps_chk:
                        reasons.append(f"INVALID_LIVE_REPAIR_TIMING:{aid}:{dep_r}->{arr_r}")
                        break
                    cur_chk = v_r
                    prev_arr = arr_r

        # Production bay mutation check: bay is irrevocable once any kit task is PICKED/COMPLETED
        for op_id, new_bay in proposal.proposed_bays.items():
            op = ops_by_id.get(op_id)
            if op and new_bay != op.bay_id:
                if op.started:
                    reasons.append(f"VIOLATION_STARTED_OP_BAY_MUTATION:{op_id}")
                else:
                    for ktid in op.kit_task_ids:
                        kt = tasks_by_id.get(ktid)
                        if kt and kt.status in (TaskStatus.PICKED, TaskStatus.COMPLETED):
                            reasons.append(f"VIOLATION_PICKED_KIT_BAY_MUTATION:{op_id}:{ktid}")
                            break
                        if execs and self.allow_physical_mode and kt and kt.assigned_agv in phys_locked:
                            reasons.append(f"VIOLATION_PHYSICAL_LOCKED_KIT_BAY_MUTATION:{op_id}:{ktid}")
                            break

        # Directed edge validity check along proposed routes
        network = self.simulator.network
        for agv_id, path in proposal.proposed_routes.items():
            if not path:
                continue
            for i in range(len(path) - 1):
                u_n, _ = path[i]
                v_n, _ = path[i + 1]
                if u_n != v_n and not network.has_edge(u_n, v_n):
                    reasons.append(f"ILLEGAL_EDGE_TRAVERSAL:{agv_id}:{u_n}->{v_n}")
                    break

        if reasons:
            return self._reject(current_tick, proposal, reasons)

        # Save full pre-commit state for atomic rollback on failure
        op_backup = [(op, op.bay_id, op.crew_id, op.tool_id) for op in self.operations]
        task_backup = [
            (t, t.dropoff, t.status, t.assigned_agv, t.assigned_step)
            for t in self.simulator.tasks
        ]
        agv_backup = [
            (a, a.cur_task, a.state, list(a.route), a.depart_step)
            for a in self.simulator.agvs
        ]
        jobs_map = getattr(self.simulator, "_jobs", None)
        jobs_backup = dict(jobs_map) if jobs_map is not None else None
        admitted_backup = (
            set(self.simulator._admitted_agvs)
            if getattr(self.simulator, "_admitted_agvs", None) is not None
            else None
        )
        pins_backup = dict(getattr(self.simulator, "_pinned_task_agv", {}))
        queues_backup = {
            k: list(v) for k, v in getattr(self.simulator, "_pinned_task_queues", {}).items()
        }
        traffic_backup = dict(getattr(self.simulator, "_traffic_weights_q3", {}))
        traffic_cutoff_backup = int(getattr(self.simulator, "_traffic_cutoff_step", 0))
        guide_backup = {
            k: list(v) for k, v in getattr(self.simulator, "_guide_corridors", {}).items()
        }
        guide_pen_backup = int(getattr(self.simulator, "_guide_pen_q3", 12))
        op_prio_backup = dict(getattr(self.simulator, "_op_priorities", {}))
        load_end_backup = dict(getattr(self.simulator, "_load_end", {}))
        drop_end_backup = dict(getattr(self.simulator, "_drop_end", {}))
        drop_arrive_backup = dict(getattr(self.simulator, "_drop_arrive", {}))
        blocked_since_backup = dict(getattr(self.simulator, "_blocked_since", {}))
        repaired_agv_ids: list[str] = list(
            set((getattr(proposal, "proposed_live_repairs", None) or {}).keys())
            | set((getattr(proposal, "proposed_bottleneck_locks", None) or {}).keys())
        )
        fleet_rev_backup = getattr(self.simulator, "_fleet_revision", 1)
        res_rev_backup = getattr(self.simulator, "_reservation_revision", 1)
        prod_rev_backup = getattr(self.simulator, "_production_revision", 1)

        try:
            # Atomic commit:
            # 1. Production bay & crew updates + sync dropoffs for unpicked kit tasks
            if self.operations:
                for op in self.operations:
                    if not op.started:
                        if op.op_id in proposal.proposed_bays:
                            op.bay_id = proposal.proposed_bays[op.op_id]
                        if op.op_id in proposal.proposed_crews:
                            op.crew_id = proposal.proposed_crews[op.op_id]
                        for ktid in op.kit_task_ids:
                            kt = tasks_by_id.get(ktid)
                            if kt is not None and kt.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED):
                                kt.dropoff = op.bay_id

            # 2. Fleet admission gating
            if proposal.proposed_admitted_agvs is not None and hasattr(self.simulator, "set_admitted_agvs"):
                self.simulator.set_admitted_agvs(proposal.proposed_admitted_agvs)

            # 3. Pin selective or full assignments in Simulator
            if hasattr(self.simulator, "set_task_assignment_pins"):
                if proposal.affected_tasks:
                    selective_pins = {
                        tid: proposal.proposed_assignment[tid]
                        for tid in proposal.affected_tasks
                        if tid in proposal.proposed_assignment
                    }
                    if selective_pins:
                        self.simulator.set_task_assignment_pins(selective_pins)
                elif proposal.proposed_admitted_agvs is not None or proposal.policy_id == "audit-probe":
                    if proposal.proposed_assignment:
                        self.simulator.set_task_assignment_pins(proposal.proposed_assignment)

            # 4. Preempt unpicked cur_task on nodes when explicitly demoted behind queue head
            res_tbl = getattr(self.simulator, "_res", None)
            agvs_by_id = {a.agv_id: a for a in self.simulator.agvs}
            faulted = getattr(self.simulator, "_faulted_agvs", set())
            admitted_gate = getattr(self.simulator, "_admitted_agvs", None)
            transit_map = getattr(self.simulator, "_transit", {})

            if proposal.proposed_task_sequences is not None:
                for agv_id, seq in proposal.proposed_task_sequences.items():
                    if not seq or agv_id in transit_map or agv_id in faulted:
                        continue
                    agv_obj = agvs_by_id.get(agv_id)
                    if agv_obj is None or not agv_obj.cur_task or agv_obj.cur_task == seq[0]:
                        continue
                    cur_t = tasks_by_id.get(agv_obj.cur_task)
                    if cur_t is not None and cur_t.status == TaskStatus.ASSIGNED:
                        if hasattr(self.simulator, "_jobs") and agv_id in self.simulator._jobs:
                            if self.simulator._jobs[agv_id].task_id == cur_t.task_id:
                                del self.simulator._jobs[agv_id]
                        if res_tbl is not None and hasattr(res_tbl, "release_all"):
                            res_tbl.release_all(agv_id)
                        agv_obj.cur_task = None
                        if agv_obj.state not in (AgvState.BLOCKED, "blocked"):
                            agv_obj.state = AgvState.IDLE
                        cur_t.status = TaskStatus.PENDING
                        cur_t.assigned_agv = None
                        cur_t.assigned_step = None

            if proposal.proposed_task_sequences is not None and hasattr(self.simulator, "set_task_queues"):
                self.simulator.set_task_queues(proposal.proposed_task_sequences)
            elif proposal.policy_id.startswith("V16_"):
                if hasattr(self.simulator, "set_task_queues"):
                    self.simulator.set_task_queues(None)
                pinned_agv = getattr(self.simulator, "_pinned_task_agv", None)
                if pinned_agv:
                    for tid_p in list(pinned_agv.keys()):
                        t_p = tasks_by_id.get(tid_p)
                        if (
                            t_p is not None
                            and t_p.status == TaskStatus.PENDING
                            and not getattr(t_p, "is_urgent", False)
                        ):
                            del pinned_agv[tid_p]

            if proposal.proposed_traffic_weights is not None and hasattr(self.simulator, "set_traffic_weights"):
                self.simulator.set_traffic_weights(
                    proposal.proposed_traffic_weights,
                    cutoff_step=int(getattr(proposal, "proposed_traffic_cutoff", 0)),
                )

            if proposal.proposed_guide_routes is not None and hasattr(self.simulator, "set_guide_corridors"):
                self.simulator.set_guide_corridors(proposal.proposed_guide_routes)
            elif proposal.policy_id.startswith("V16_") and hasattr(self.simulator, "set_guide_corridors"):
                self.simulator.set_guide_corridors(None)

            if getattr(proposal, "proposed_op_priorities", None) is not None:
                clean_prio = {str(k): int(v) for k, v in proposal.proposed_op_priorities.items()}
                if hasattr(self.simulator, "set_op_priorities"):
                    self.simulator.set_op_priorities(clean_prio)
                else:
                    self.simulator._op_priorities = clean_prio
                if self.production_tracker is not None and hasattr(self.production_tracker, "set_op_priorities"):
                    self.production_tracker.set_op_priorities(clean_prio)
                if clean_prio != op_prio_backup and hasattr(self.simulator, "_production_revision"):
                    self.simulator._production_revision = int(self.simulator._production_revision) + 1

            # 5. Update active/assigned tasks and release stale jobs on reassigned AGVs
            prio_lookup = getattr(proposal, "proposed_op_priorities", None) or {}
            ordered_assign = sorted(
                proposal.proposed_assignment.items(),
                key=lambda item: (
                    int(prio_lookup.get(getattr(tasks_by_id.get(item[0]), "kit_op_id", None) or "", 999999)),
                    str(item[0]),
                ),
            )
            for tid, new_agv_id in ordered_assign:
                t = tasks_by_id.get(tid)
                if not t or t.status in (TaskStatus.COMPLETED, TaskStatus.PICKED):
                    continue
                old_agv_id = t.assigned_agv
                if old_agv_id and old_agv_id != new_agv_id:
                    transit_now = getattr(self.simulator, "_transit", {})
                    tr_entry = transit_now.get(old_agv_id)
                    hold_map = getattr(self.simulator, "_node_hold", {})
                    claim_map = getattr(self.simulator, "_node_claim", {})
                    if tr_entry is not None:
                        tr_src, tr_dst, tr_dep, tr_arr = tr_entry
                        if hasattr(self.simulator, "_jobs") and old_agv_id in self.simulator._jobs:
                            if self.simulator._jobs[old_agv_id].task_id == tid:
                                self.simulator._jobs[old_agv_id] = _Job(
                                    task_id=None,
                                    leg="park",
                                    hops=[Hop(tr_src, tr_dst, tr_dep, tr_arr)],
                                    idx=1,
                                    goal=tr_dst,
                                    reserved_upto=1,
                                )
                        if res_tbl is not None and hasattr(res_tbl, "release_all"):
                            res_tbl.release_all(old_agv_id)
                            for h in (hold_map.get(old_agv_id), claim_map.get(old_agv_id)):
                                if h is not None:
                                    res_tbl.reserve_node(h[0], h[1], old_agv_id)
                            res_tbl.reserve_edge(
                                tr_src,
                                tr_dst,
                                Interval(int(tr_dep), max(int(tr_dep) + 1, int(tr_arr))),
                                old_agv_id,
                            )
                    else:
                        if hasattr(self.simulator, "_jobs") and old_agv_id in self.simulator._jobs:
                            if self.simulator._jobs[old_agv_id].task_id == tid:
                                del self.simulator._jobs[old_agv_id]
                        if res_tbl is not None and hasattr(res_tbl, "release_all"):
                            res_tbl.release_all(old_agv_id)
                            h = hold_map.get(old_agv_id)
                            if h is not None:
                                res_tbl.reserve_node(h[0], h[1], old_agv_id)
                        if hasattr(self.simulator, "_load_end"):
                            self.simulator._load_end.pop(old_agv_id, None)
                        if hasattr(self.simulator, "_blocked_since"):
                            self.simulator._blocked_since.pop(old_agv_id, None)
                    old_agv = agvs_by_id.get(old_agv_id)
                    gate_old = execs.get(old_agv_id) if execs else None
                    if old_agv and old_agv.cur_task == tid:
                        old_agv.cur_task = None
                        if (
                            old_agv.state not in (AgvState.BLOCKED, "blocked")
                            and old_agv_id not in faulted
                            and not (gate_old is not None and gate_old.fault)
                        ):
                            old_agv.state = AgvState.PARKING if tr_entry is not None else AgvState.IDLE
                    t.status = TaskStatus.PENDING
                    t.assigned_agv = None
                    t.assigned_step = None

                if t.status == TaskStatus.PENDING and t.release_step <= current_tick:
                    new_agv = agvs_by_id.get(new_agv_id)
                    is_admitted = admitted_gate is None or new_agv_id in admitted_gate
                    gate_new = execs.get(new_agv_id) if execs else None
                    phys_ready = (
                        (not execs)
                        or (
                            gate_new is not None
                            and gate_new.bound
                            and gate_new.command is None
                            and not gate_new.fault
                            and new_agv_id not in phys_locked
                        )
                    )
                    if (
                        new_agv is not None
                        and is_admitted
                        and phys_ready
                        and new_agv_id not in faulted
                        and new_agv.cur_task is None
                        and new_agv_id not in getattr(self.simulator, "_transit", {})
                        and new_agv.state not in (AgvState.BLOCKED, "blocked")
                        and hasattr(self.simulator, "_try_assign")
                    ):
                        self.simulator._try_assign(new_agv, t, current_tick)

            # 6. Commitment-preserving live route/siding repair for stalled AGVs (picked/idle)
            if getattr(proposal, "proposed_live_repairs", None):
                res_aware_lr = bool(
                    getattr(getattr(self.simulator, "planner", None), "reservations_aware", False)
                )
                win_lr = int(getattr(self.simulator.config, "reservation_window", 1))
                merged_tw = dict(getattr(self.simulator, "_traffic_weights_q3", {}))
                merged_cutoff = int(getattr(self.simulator, "_traffic_cutoff_step", 0))
                has_tw_update = False

                for aid_lr, rep in proposal.proposed_live_repairs.items():
                    agv_lr = agvs_by_id[aid_lr]
                    raw_hops = rep.get("hops") or []
                    kind_lr = str(rep.get("kind"))
                    if kind_lr not in ("picked_reroute", "assigned_reroute"):
                        cur_job_lr = self.simulator._jobs.get(aid_lr) if hasattr(self.simulator, "_jobs") else None
                        has_live_task = (
                            agv_lr.cur_task is not None
                            or (cur_job_lr is not None and getattr(cur_job_lr, "task_id", None) not in (None, ""))
                        )
                        if has_live_task:
                            continue
                    hop_objs = [Hop(src=str(u), dst=str(v), depart=int(d), arrive=int(a)) for (u, v, d, a) in raw_hops]
                    if hasattr(self.simulator, "_release_future_at_node"):
                        self.simulator._release_future_at_node(aid_lr)
                    if res_aware_lr:
                        min_res_hops = int(rep.get("min_reserve_hops", 0))
                        upto_lr = (
                            len(hop_objs)
                            if (win_lr <= 0 or min_res_hops >= len(hop_objs))
                            else min(len(hop_objs), max(win_lr, min_res_hops))
                        )
                        for h_lr in hop_objs[:upto_lr]:
                            self.simulator._reserve_hop(agv_lr, h_lr)
                        extra_locks = rep.get("extra_node_locks") or ()
                        if extra_locks and res_tbl is not None and hasattr(res_tbl, "reserve_node"):
                            for lock_nd, lock_lo, lock_hi in extra_locks:
                                if int(lock_hi) > int(lock_lo):
                                    res_tbl.reserve_node(str(lock_nd), Interval(int(lock_lo), int(lock_hi)), aid_lr)
                    else:
                        upto_lr = 0

                    if kind_lr == "picked_reroute":
                        tid_lr = str(rep["task_id"])
                        t_lr = tasks_by_id[tid_lr]
                        if hasattr(self.simulator, "_jobs"):
                            self.simulator._jobs[aid_lr] = _Job(
                                task_id=tid_lr,
                                leg="to_dropoff",
                                hops=hop_objs,
                                idx=0,
                                goal=t_lr.dropoff,
                                reserved_upto=upto_lr,
                            )
                        agv_lr.state = AgvState.TO_DROPOFF
                        if hasattr(self.simulator, "_load_end"):
                            self.simulator._load_end.pop(aid_lr, None)
                        if hasattr(self.simulator, "_drop_arrive"):
                            self.simulator._drop_arrive.pop(aid_lr, None)
                        if hasattr(self.simulator, "_drop_end"):
                            unload_n = (
                                self.simulator._unload_steps(t_lr)
                                if hasattr(self.simulator, "_unload_steps")
                                else 5
                            )
                            self.simulator._drop_end[aid_lr] = hop_objs[-1].arrive + unload_n
                            if res_aware_lr:
                                self.simulator._res.reserve_node(
                                    t_lr.dropoff,
                                    Interval(hop_objs[-1].arrive, hop_objs[-1].arrive + unload_n),
                                    aid_lr,
                                )
                    elif kind_lr == "assigned_reroute":
                        tid_lr = str(rep["task_id"])
                        t_lr = tasks_by_id.get(tid_lr)
                        if (
                            t_lr is None
                            or t_lr.status != TaskStatus.ASSIGNED
                            or t_lr.assigned_agv != aid_lr
                            or agv_lr.cur_task != tid_lr
                        ):
                            continue
                        if hasattr(self.simulator, "_jobs"):
                            self.simulator._jobs[aid_lr] = _Job(
                                task_id=tid_lr,
                                leg="to_pickup",
                                hops=hop_objs,
                                idx=0,
                                goal=t_lr.pickup,
                                reserved_upto=upto_lr,
                            )
                        agv_lr.state = AgvState.TO_PICKUP
                        if hasattr(self.simulator, "_load_end"):
                            load_n = (
                                self.simulator._load_steps(t_lr)
                                if hasattr(self.simulator, "_load_steps")
                                else 5
                            )
                            self.simulator._load_end[aid_lr] = hop_objs[-1].arrive + load_n
                            if res_aware_lr:
                                self.simulator._res.reserve_node(
                                    t_lr.pickup,
                                    Interval(hop_objs[-1].arrive, hop_objs[-1].arrive + load_n),
                                    aid_lr,
                                )
                    else:
                        goal_lr = str(rep["goal"])
                        if hasattr(self.simulator, "_jobs"):
                            self.simulator._jobs[aid_lr] = _Job(
                                task_id="",
                                leg="park",
                                hops=hop_objs,
                                idx=0,
                                goal=goal_lr,
                                reserved_upto=upto_lr,
                            )
                        agv_lr.state = AgvState.PARKING
                    if hasattr(self.simulator, "_blocked_since"):
                        self.simulator._blocked_since.pop(aid_lr, None)
                    agv_lr.route = [h.src for h in hop_objs] + [hop_objs[-1].dst]

                    rep_tw = rep.get("traffic_weights")
                    if isinstance(rep_tw, dict) and rep_tw:
                        has_tw_update = True
                        for ek, ev in rep_tw.items():
                            k_tup = (str(ek[0]), str(ek[1]))
                            merged_tw[k_tup] = max(int(merged_tw.get(k_tup, 0)), int(ev))
                        merged_cutoff = max(merged_cutoff, int(rep.get("traffic_cutoff", current_tick + 60)))

                if has_tw_update and hasattr(self.simulator, "set_traffic_weights"):
                    self.simulator.set_traffic_weights(merged_tw, cutoff_step=merged_cutoff)
                if not hasattr(self.simulator, "_last_loaded_repair_at"):
                    self.simulator._last_loaded_repair_at = {}
                for aid_lr_rec in proposal.proposed_live_repairs:
                    agv_rec = agvs_by_id.get(aid_lr_rec)
                    if agv_rec is not None:
                        self.simulator._last_loaded_repair_at[aid_lr_rec] = (
                            current_tick,
                            str(agv_rec.cur_node or agv_rec.start_node),
                        )

            # 6b. Commitment-preserving bottleneck crossing-order locks (applied after live_repairs release)
            b_locks_commit = getattr(proposal, "proposed_bottleneck_locks", None)
            if b_locks_commit and res_tbl is not None and hasattr(res_tbl, "reserve_node"):
                for aid_bl, spans_bl in b_locks_commit.items():
                    for lock_nd, lo_s, hi_s in spans_bl:
                        if int(hi_s) > int(lo_s):
                            res_tbl.reserve_node(str(lock_nd), Interval(int(lo_s), int(hi_s)), str(aid_bl))

            self.simulator._reservation_revision = getattr(self.simulator, "_reservation_revision", 1) + 1
            self.simulator._production_revision = getattr(self.simulator, "_production_revision", 1) + 1
        except Exception:
            for op, bay_id, crew_id, tool_id in op_backup:
                op.bay_id = bay_id
                op.crew_id = crew_id
                op.tool_id = tool_id
            for t, dropoff, status, assigned_agv, assigned_step in task_backup:
                t.dropoff = dropoff
                t.status = status
                t.assigned_agv = assigned_agv
                t.assigned_step = assigned_step
            for a, cur_task, state, route, depart_step in agv_backup:
                a.cur_task = cur_task
                a.state = state
                a.route = route
                a.depart_step = depart_step
            if jobs_backup is not None:
                self.simulator._jobs.clear()
                self.simulator._jobs.update(jobs_backup)
            for aid_lr in repaired_agv_ids:
                if hasattr(self.simulator, "_release_future_at_node"):
                    self.simulator._release_future_at_node(aid_lr)
                orig_job = jobs_backup.get(aid_lr) if jobs_backup is not None else None
                agv_lr = agvs_by_id.get(aid_lr)
                if orig_job is not None and agv_lr is not None and hasattr(self.simulator, "_reserve_hop"):
                    for h_orig in orig_job.hops[orig_job.idx : orig_job.reserved_upto]:
                        self.simulator._reserve_hop(agv_lr, h_orig)
            if hasattr(self.simulator, "_load_end"):
                self.simulator._load_end.clear()
                self.simulator._load_end.update(load_end_backup)
            if hasattr(self.simulator, "_drop_end"):
                self.simulator._drop_end.clear()
                self.simulator._drop_end.update(drop_end_backup)
            if hasattr(self.simulator, "_drop_arrive"):
                self.simulator._drop_arrive.clear()
                self.simulator._drop_arrive.update(drop_arrive_backup)
            if hasattr(self.simulator, "_blocked_since"):
                self.simulator._blocked_since.clear()
                self.simulator._blocked_since.update(blocked_since_backup)
            if hasattr(self.simulator, "_admitted_agvs"):
                self.simulator._admitted_agvs = admitted_backup
            if hasattr(self.simulator, "_pinned_task_agv"):
                self.simulator._pinned_task_agv = dict(pins_backup)
            if hasattr(self.simulator, "_pinned_task_queues"):
                self.simulator._pinned_task_queues = queues_backup
            if hasattr(self.simulator, "set_traffic_weights"):
                self.simulator.set_traffic_weights(traffic_backup, cutoff_step=traffic_cutoff_backup)
            if hasattr(self.simulator, "set_guide_corridors"):
                self.simulator.set_guide_corridors(guide_backup, off_guide_pen_q3=guide_pen_backup)
            if hasattr(self.simulator, "set_op_priorities"):
                self.simulator.set_op_priorities(op_prio_backup)
            elif hasattr(self.simulator, "_op_priorities"):
                self.simulator._op_priorities = op_prio_backup
            if self.production_tracker is not None and hasattr(self.production_tracker, "set_op_priorities"):
                self.production_tracker.set_op_priorities(op_prio_backup)
            if hasattr(self.simulator, "_fleet_revision"):
                self.simulator._fleet_revision = fleet_rev_backup
            if hasattr(self.simulator, "_reservation_revision"):
                self.simulator._reservation_revision = res_rev_backup
            if hasattr(self.simulator, "_production_revision"):
                self.simulator._production_revision = prod_rev_backup
            raise

        rcpt_payload = {
            "accepted": True,
            "reasons": ["TRANSACTION_COMMITTED_SUCCESSFULLY"],
            "tick": current_tick,
            "candidate_digest": proposal.candidate_digest,
        }
        rcpt_digest = hashlib.sha256(json.dumps(rcpt_payload, sort_keys=True).encode()).hexdigest()

        return TransactionReceipt(
            schema="amr-lgcvcr.transaction-receipt.v1",
            accepted=True,
            reasons=["TRANSACTION_COMMITTED_SUCCESSFULLY"],
            committed_tick=current_tick,
            snapshot_digest=proposal.snapshot_digest,
            candidate_digest=proposal.candidate_digest,
            receipt_digest=rcpt_digest,
            physical_authorization=False,
        )
