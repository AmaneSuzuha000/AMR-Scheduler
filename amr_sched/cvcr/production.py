"""Native Production & Kitting Runtime Tracker for AMR-Scheduler.

Advances railway maintenance operations step-by-step alongside Simulator.step_once():
- Tracks material kitting readiness (all kit tasks must reach TaskStatus.COMPLETED)
- Enforces operation predecessor completion
- Enforces exclusive occupancy of bay_id, crew_id, and tool_id
- Computes exact closed-loop production makespan, weighted tardiness, and completion count
- Syncs legal bay switching onto uncommitted transport tasks' dropoff nodes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from amr_sched.cvcr.snapshot import OperationSnapshot, align_start_to_windows, op_predecessors
from amr_sched.entities import Task, TaskStatus


@dataclass
class ProductionMetrics:
    completed_ops: int
    total_ops: int
    production_makespan: int
    weighted_tardiness: float
    urgent_tardiness: float
    starvation_steps: int


class NativeProductionTracker:
    """Closed-loop production operation tracker synchronized with Simulator ticks."""

    def __init__(self, operations: list[OperationSnapshot], tasks: list[Task]) -> None:
        self.operations = operations
        self.op_by_id = {op.op_id: op for op in operations}
        self.task_by_id = {t.task_id: t for t in tasks}
        self.starvation_steps = 0
        self.op_priorities: dict[str, int] = {}

    def set_op_priorities(self, priorities: dict[str, int] | None) -> None:
        """Install explicit CVCR operation priority tie-breakers (smaller runs earlier)."""
        if not priorities:
            self.op_priorities = {}
            return
        self.op_priorities = {str(k): int(v) for k, v in priorities.items()}

    def _kit_ready_step(self, op: OperationSnapshot) -> Optional[int]:
        if not op.kit_task_ids:
            return 0
        steps: list[int] = []
        for tid in op.kit_task_ids:
            t = self.task_by_id.get(tid)
            if t is None or t.status is not TaskStatus.COMPLETED or t.complete_step is None:
                return None
            steps.append(int(t.complete_step))
        return max(steps) if steps else 0

    def _resources_busy(self, step: int) -> tuple[set[str], set[int], set[int]]:
        busy_bays: set[str] = set()
        busy_crews: set[int] = set()
        busy_tools: set[int] = set()
        for op in self.operations:
            if op.started and op.end_step is not None and step < op.end_step:
                busy_bays.add(op.bay_id)
                busy_crews.add(op.crew_id)
                busy_tools.add(op.tool_id)
        return busy_bays, busy_crews, busy_tools

    def step(self, tick: int) -> list[str]:
        """Advance production state at simulation tick `tick`. Returns started op IDs."""
        started_now: list[str] = []
        busy_bays, busy_crews, busy_tools = self._resources_busy(tick)

        # Deterministic priority order: use explicit CVCR priorities when provided, else urgent/due/id
        pending_ops = [
            op for op in self.operations if not op.started
        ]
        if self.op_priorities:
            pending_ops.sort(
                key=lambda op: (
                    self.op_priorities.get(
                        op.op_id,
                        (0 if op.is_urgent else 1) * 100000 + op.due_step,
                    ),
                    0 if op.is_urgent else 1,
                    op.due_step,
                    op.op_id,
                )
            )
        else:
            pending_ops.sort(key=lambda op: (0 if op.is_urgent else 1, op.due_step, op.op_id))

        for op in pending_ops:
            pred_done = True
            for pid in op_predecessors(op):
                pred = self.op_by_id.get(pid)
                if pred is None or not pred.started or pred.end_step is None or pred.end_step > tick:
                    pred_done = False
                    break

            kit_ready = self._kit_ready_step(op)
            res_free = (
                op.bay_id not in busy_bays
                and op.crew_id not in busy_crews
                and op.tool_id not in busy_tools
            )
            r_wins = getattr(op, "resource_windows", ())
            win_open = align_start_to_windows(tick, r_wins) == tick if r_wins else True

            if pred_done and res_free and win_open and kit_ready is None:
                # Resources, predecessor, and shift window are ready, waiting solely on kitting transport
                self.starvation_steps += 1

            if pred_done and kit_ready is not None and kit_ready <= tick and res_free and win_open:
                op.started = True
                op.start_step = tick
                op.end_step = tick + op.proc_steps
                busy_bays.add(op.bay_id)
                busy_crews.add(op.crew_id)
                busy_tools.add(op.tool_id)
                started_now.append(op.op_id)

        return started_now

    def sync_task_dropoffs(self) -> None:
        """Ensure unpicked tasks mirror any legally switched operation bay_id."""
        for op in self.operations:
            for tid in op.kit_task_ids:
                t = self.task_by_id.get(tid)
                if t is not None and t.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED):
                    t.dropoff = op.bay_id

    def all_done(self, current_tick: int) -> bool:
        """Return True iff every operation has started and finished by current_tick."""
        return all(
            op.started and op.end_step is not None and op.end_step <= current_tick
            for op in self.operations
        )

    def metrics(self, horizon_step: int) -> ProductionMetrics:
        """Compute exact closed-loop production completion, makespan, and weighted tardiness."""
        completed = 0
        ends: list[int] = []
        tard = 0.0
        urg_tard = 0.0
        for op in self.operations:
            if op.started and op.end_step is not None and op.end_step <= horizon_step:
                completed += 1
                en = op.end_step
            else:
                en = horizon_step
            ends.append(en)
            lateness = max(0, en - op.due_step)
            w = 12.0 if op.is_urgent else 4.0
            tard += w * lateness
            if op.is_urgent:
                urg_tard += 12.0 * lateness
        mk = max(ends) if ends else horizon_step
        return ProductionMetrics(
            completed_ops=completed,
            total_ops=len(self.operations),
            production_makespan=mk,
            weighted_tardiness=tard,
            urgent_tardiness=urg_tard,
            starvation_steps=self.starvation_steps,
        )
