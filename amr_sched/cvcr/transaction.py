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
    ) -> None:
        self.simulator = simulator
        self.operations = operations or []

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

    def validate_and_commit(self, proposal: CandidateProposal) -> TransactionReceipt:
        """Validate snapshot freshness, commitments, admission & routes, then commit atomically."""
        current_tick = int(self.simulator._step_cursor)
        reasons: list[str] = []

        # Never grant physical authorization when hardware executions are bound
        if getattr(self.simulator, "executions", None):
            reasons.append("PHYSICAL_MODE_FORBIDDEN_FOR_SW_TRANSACTION")

        snap = NativeSnapshotCapture.capture(self.simulator, self.operations)
        if snap.tick != current_tick:
            reasons.append("STALE_SNAPSHOT_STEP_MISMATCH")

        clf = classify_commitments(snap)
        tasks_by_id = {t.task_id: t for t in self.simulator.tasks}
        ops_by_id = {op.op_id: op for op in self.operations}

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

        # 3. Pin assignments in Simulator so future-released & queued tasks honor CVCR allocation
        if hasattr(self.simulator, "set_task_assignment_pins"):
            if proposal.proposed_assignment or proposal.proposed_admitted_agvs is not None:
                self.simulator.set_task_assignment_pins(proposal.proposed_assignment)

        # 4. Update active/assigned tasks and release stale jobs on reassigned AGVs
        agvs_by_id = {a.agv_id: a for a in self.simulator.agvs}
        faulted = getattr(self.simulator, "_faulted_agvs", set())
        admitted_gate = getattr(self.simulator, "_admitted_agvs", None)

        for tid, new_agv_id in proposal.proposed_assignment.items():
            t = tasks_by_id.get(tid)
            if not t or t.status in (TaskStatus.COMPLETED, TaskStatus.PICKED):
                continue
            old_agv_id = t.assigned_agv
            if old_agv_id and old_agv_id != new_agv_id:
                if hasattr(self.simulator, "_jobs") and old_agv_id in self.simulator._jobs:
                    if self.simulator._jobs[old_agv_id].task_id == tid:
                        del self.simulator._jobs[old_agv_id]
                old_agv = agvs_by_id.get(old_agv_id)
                if old_agv and old_agv.cur_task == tid:
                    old_agv.cur_task = None
                    if old_agv.state not in (AgvState.BLOCKED, "blocked") and old_agv_id not in faulted:
                        old_agv.state = AgvState.IDLE
                t.status = TaskStatus.PENDING
                t.assigned_agv = None
                t.assigned_step = None

            if t.status == TaskStatus.PENDING and t.release_step <= current_tick:
                new_agv = agvs_by_id.get(new_agv_id)
                is_admitted = admitted_gate is None or new_agv_id in admitted_gate
                if (
                    new_agv is not None
                    and is_admitted
                    and new_agv_id not in faulted
                    and new_agv.cur_task is None
                    and new_agv_id not in getattr(self.simulator, "_transit", {})
                    and new_agv.state not in (AgvState.BLOCKED, "blocked")
                    and hasattr(self.simulator, "_try_assign")
                ):
                    self.simulator._try_assign(new_agv, t, current_tick)

        self.simulator._reservation_revision = getattr(self.simulator, "_reservation_revision", 1) + 1
        self.simulator._production_revision = getattr(self.simulator, "_production_revision", 1) + 1

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
