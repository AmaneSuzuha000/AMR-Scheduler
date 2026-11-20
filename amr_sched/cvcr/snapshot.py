"""Native Snapshot Capture for LG-CVCR / AMR-Scheduler integration.

Captures an immutable, frozen state snapshot from the native Simulator at a single
discrete time step (tick). Guarantees that algorithms observe only the currently
released state without future knowledge.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from amr_sched.cvcr.execution_feedback import ExecutionFeedbackPacket


@dataclass(frozen=True)
class RobotSnapshot:
    robot_id: str
    cur_node: str
    state: str
    cur_task: Optional[str] = None
    loaded_task_id: Optional[str] = None
    faulted_until: int = 0
    route: tuple[str, ...] = field(default_factory=tuple)
    in_transit: bool = False


@dataclass(frozen=True)
class TaskSnapshot:
    task_id: str
    pickup: str
    dropoff: str
    release_step: int
    due_step: Optional[int]
    status: str
    assigned_agv: Optional[str] = None
    kit_op_id: Optional[str] = None
    pickup_step: Optional[int] = None
    complete_step: Optional[int] = None
    priority: str = "normal"
    is_urgent: bool = False


@dataclass
class OperationSnapshot:
    op_id: str
    kit_task_ids: tuple[str, ...]
    bay_id: str
    crew_id: int
    tool_id: int
    proc_steps: int
    due_step: int
    predecessor_op_id: Optional[str] = None
    predecessor_op_ids: tuple[str, ...] = field(default_factory=tuple)
    started: bool = False
    start_step: Optional[int] = None
    end_step: Optional[int] = None
    is_urgent: bool = False
    eligible_bays: tuple[str, ...] = field(default_factory=tuple)
    eligible_crews: tuple[int, ...] = field(default_factory=tuple)
    eligible_tools: tuple[int, ...] = field(default_factory=tuple)
    resource_windows: tuple[tuple[int, int], ...] = field(default_factory=tuple)


def align_start_to_windows(earliest_step: int, windows: Any) -> int:
    """Align earliest candidate op start step onto discrete `[win_start, win_close]` windows.

    Returns `earliest_step` unchanged when `windows` is empty. If `earliest_step` misses all
    explicit windows (`> last_close`), shifts start to `max(earliest_step, last_close + 40)`
    reflecting waiting for the next backup crew/bay shift.
    """
    if not windows:
        return int(earliest_step)
    t = int(earliest_step)
    last_close: int | None = None
    for w_start, w_end in sorted((int(a), int(b)) for a, b in windows):
        last_close = w_end if last_close is None else max(last_close, w_end)
        if t <= w_end:
            return max(t, w_start)
    assert last_close is not None
    return max(t, last_close + 40)


def op_predecessors(op: Any) -> tuple[str, ...]:
    """Return all normalized unique predecessor operation IDs for `op`."""
    preds: list[str] = []
    seen: set[str] = set()
    for p in tuple(getattr(op, "predecessor_op_ids", ()) or ()) + (
        (op.predecessor_op_id,) if getattr(op, "predecessor_op_id", None) else ()
    ):
        if p and p not in seen:
            seen.add(p)
            preds.append(str(p))
    return tuple(preds)


@dataclass(frozen=True)
class ReservationSnapshot:
    resource_type: str  # "node" | "edge"
    resource_id: str  # "N1" or "N1->N2"
    start_step: int
    end_step: int
    agv_id: str


@dataclass
class NativeSnapshot:
    schema: str
    tick: int
    version: dict[str, int]
    world_fingerprint: str
    robots: list[RobotSnapshot]
    tasks: list[TaskSnapshot]
    operations: list[OperationSnapshot]
    reservations: list[ReservationSnapshot]
    now_monotonic: float
    execution_feedback: Optional[ExecutionFeedbackPacket] = None

    def compute_digest(self) -> str:
        """Deterministic SHA-256 fingerprint of the snapshot."""
        payload = {
            "schema": self.schema,
            "tick": self.tick,
            "version": self.version,
            "world_fingerprint": self.world_fingerprint,
            "robots": [
                {
                    "robot_id": r.robot_id,
                    "cur_node": r.cur_node,
                    "state": r.state,
                    "cur_task": r.cur_task,
                    "loaded_task_id": r.loaded_task_id,
                    "faulted_until": r.faulted_until,
                    "route": list(r.route),
                }
                for r in sorted(self.robots, key=lambda x: x.robot_id)
            ],
            "tasks": [
                {
                    "task_id": t.task_id,
                    "pickup": t.pickup,
                    "dropoff": t.dropoff,
                    "release_step": t.release_step,
                    "due_step": t.due_step,
                    "status": t.status,
                    "assigned_agv": t.assigned_agv,
                    "kit_op_id": t.kit_op_id,
                    "pickup_step": t.pickup_step,
                    "complete_step": t.complete_step,
                    "priority": t.priority,
                    "is_urgent": t.is_urgent,
                }
                for t in sorted(self.tasks, key=lambda x: x.task_id)
            ],
            "operations": [
                {
                    "op_id": op.op_id,
                    "kit_task_ids": list(op.kit_task_ids),
                    "bay_id": op.bay_id,
                    "crew_id": op.crew_id,
                    "tool_id": op.tool_id,
                    "proc_steps": op.proc_steps,
                    "due_step": op.due_step,
                    "predecessor_op_id": op.predecessor_op_id,
                    **(
                        {"predecessor_op_ids": list(op.predecessor_op_ids)}
                        if getattr(op, "predecessor_op_ids", ())
                        else {}
                    ),
                    "started": op.started,
                    "start_step": op.start_step,
                    "end_step": op.end_step,
                    "is_urgent": op.is_urgent,
                    "eligible_bays": list(op.eligible_bays),
                    "eligible_crews": list(op.eligible_crews),
                    "eligible_tools": list(op.eligible_tools),
                    **(
                        {"resource_windows": [list(w) for w in op.resource_windows]}
                        if getattr(op, "resource_windows", ())
                        else {}
                    ),
                }
                for op in sorted(self.operations, key=lambda x: x.op_id)
            ],
            "reservations": [
                {
                    "resource_type": res.resource_type,
                    "resource_id": res.resource_id,
                    "start_step": res.start_step,
                    "end_step": res.end_step,
                    "agv_id": res.agv_id,
                }
                for res in sorted(
                    self.reservations,
                    key=lambda x: (x.resource_type, x.resource_id, x.start_step, x.agv_id),
                )
            ],
        }
        if self.execution_feedback is not None and not self.execution_feedback.is_empty():
            payload["execution_feedback"] = self.execution_feedback.canonical_payload()
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


class NativeSnapshotCapture:
    """Extracts immutable NativeSnapshot from AMR-Scheduler Simulator."""

    @staticmethod
    def capture(
        simulator: Any,
        operations: list[OperationSnapshot] | None = None,
        version: dict[str, int] | None = None,
        hide_future_releases: bool = False,
        execution_feedback: Optional[ExecutionFeedbackPacket] = None,
    ) -> NativeSnapshot:
        """Capture an immutable snapshot of state strictly observable at simulator._step_cursor."""
        tick = int(simulator._step_cursor)
        now_mono = time.monotonic()

        # 1. World fingerprint
        node_ids = sorted(simulator.network.nodes.keys())
        edge_keys = sorted(f"{u}->{v}" for u, v in simulator.network.edges.keys())
        topo_sig = f"nodes:{len(node_ids)};edges:{len(edge_keys)};dt:{simulator.config.dt}"
        world_fp = hashlib.sha256(topo_sig.encode("utf-8")).hexdigest()[:16]

        # 2. Robots
        faulted_set = getattr(simulator, "_faulted_agvs", set())
        transit_set = getattr(simulator, "_transit", {})
        robot_snaps: list[RobotSnapshot] = []
        for agv in simulator.agvs:
            st_val = str(agv.state.value if hasattr(agv.state, "value") else agv.state)
            is_phys_fault = (
                agv.agv_id in faulted_set
                or getattr(agv, "faulted", False)
                or st_val == "faulted"
            )
            if is_phys_fault:
                st_val = "faulted"
            f_until = getattr(agv, "faulted_until_step", 0)
            if is_phys_fault and f_until <= tick:
                f_until = tick + 999
            r_snap = RobotSnapshot(
                robot_id=str(agv.agv_id),
                cur_node=str(agv.cur_node or agv.start_node),
                state=st_val,
                cur_task=str(agv.cur_task) if agv.cur_task else None,
                loaded_task_id=str(agv.cur_task)
                if st_val in ("loading", "to_dropoff", "unloading")
                else None,
                faulted_until=f_until,
                route=tuple(str(n) for n in agv.route),
                in_transit=bool(agv.agv_id in transit_set),
            )
            robot_snaps.append(r_snap)

        # 3. Tasks - enforce causal observability for exogenous/hidden future releases
        task_snaps: list[TaskSnapshot] = []
        for t in simulator.tasks:
            if t.release_step > tick and (
                hide_future_releases or getattr(t, "hidden_until_release", False)
            ):
                continue
            status_val = str(t.status.value if hasattr(t.status, "value") else t.status)
            prio_raw = getattr(t, "priority", "normal")
            prio_str = str(getattr(prio_raw, "value", getattr(prio_raw, "name", prio_raw))).lower()
            is_urg = bool(
                getattr(t, "is_urgent", False)
                or prio_str in ("urgent", "high", "1", "taskpriority.urgent")
            )
            t_snap = TaskSnapshot(
                task_id=str(t.task_id),
                pickup=str(t.pickup),
                dropoff=str(t.dropoff),
                release_step=int(t.release_step),
                due_step=int(t.due_step) if t.due_step is not None else None,
                status=status_val,
                assigned_agv=str(t.assigned_agv) if t.assigned_agv else None,
                kit_op_id=getattr(t, "kit_op_id", None),
                pickup_step=getattr(t, "pickup_step", None),
                complete_step=getattr(t, "complete_step", None),
                priority="urgent" if is_urg else "normal",
                is_urgent=is_urg,
            )
            task_snaps.append(t_snap)

        # 4. Reservations
        res_snaps: list[ReservationSnapshot] = []
        res_table = getattr(simulator, "_res", None)
        if res_table is not None:
            # Nodes
            for node, ivs in getattr(res_table, "_node", {}).items():
                for iv, agv_id in ivs:
                    if iv.end > tick:
                        res_snaps.append(
                            ReservationSnapshot(
                                resource_type="node",
                                resource_id=str(node),
                                start_step=int(iv.start),
                                end_step=int(iv.end),
                                agv_id=str(agv_id),
                            )
                        )
            # Edges
            for (u, v), ivs in getattr(res_table, "_edge", {}).items():
                for iv, agv_id in ivs:
                    if iv.end > tick:
                        res_snaps.append(
                            ReservationSnapshot(
                                resource_type="edge",
                                resource_id=f"{u}->{v}",
                                start_step=int(iv.start),
                                end_step=int(iv.end),
                                agv_id=str(agv_id),
                            )
                        )

        # 5. Operations
        op_snaps: list[OperationSnapshot] = []
        if operations is not None:
            for op in operations:
                op_snaps.append(copy.deepcopy(op))

        if version is not None:
            ver = version
        else:
            adm = getattr(simulator, "_admitted_agvs", None)
            adm_sig = (
                int(hashlib.sha256(",".join(sorted(adm)).encode("utf-8")).hexdigest()[:8], 16)
                if adm is not None
                else -1
            )
            ver = {
                "world": 1,
                "reservation": int(getattr(simulator, "_reservation_revision", 1)),
                "production": int(getattr(simulator, "_production_revision", 1)),
                "fleet": int(getattr(simulator, "_fleet_revision", 1)),
                "admitted_sig": adm_sig,
            }

        resolved_fb = execution_feedback
        if resolved_fb is None:
            resolved_fb = ExecutionFeedbackPacket.from_simulator_and_events(simulator)

        return NativeSnapshot(
            schema="amr-lgcvcr.snapshot.v1",
            tick=tick,
            version=ver,
            world_fingerprint=world_fp,
            robots=robot_snaps,
            tasks=task_snaps,
            operations=op_snaps,
            reservations=res_snaps,
            now_monotonic=now_mono,
            execution_feedback=resolved_fb,
        )
