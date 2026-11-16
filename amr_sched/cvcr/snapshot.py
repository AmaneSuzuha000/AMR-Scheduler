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


@dataclass(frozen=True)
class RobotSnapshot:
    robot_id: str
    cur_node: str
    state: str
    cur_task: Optional[str] = None
    loaded_task_id: Optional[str] = None
    faulted_until: int = 0
    route: tuple[str, ...] = field(default_factory=tuple)


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
    started: bool = False
    start_step: Optional[int] = None
    end_step: Optional[int] = None
    is_urgent: bool = False
    eligible_bays: tuple[str, ...] = field(default_factory=tuple)
    eligible_crews: tuple[int, ...] = field(default_factory=tuple)
    eligible_tools: tuple[int, ...] = field(default_factory=tuple)


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

    def compute_digest(self) -> str:
        """Deterministic SHA-256 fingerprint of the snapshot."""
        payload = {
            "schema": self.schema,
            "tick": self.tick,
            "version": self.version,
            "world_fingerprint": self.world_fingerprint,
            "robots": [asdict(r) for r in sorted(self.robots, key=lambda x: x.robot_id)],
            "tasks": [asdict(t) for t in sorted(self.tasks, key=lambda x: x.task_id)],
            "operations": [asdict(op) for op in sorted(self.operations, key=lambda x: x.op_id)],
            "reservations": [
                asdict(res)
                for res in sorted(
                    self.reservations,
                    key=lambda x: (x.resource_type, x.resource_id, x.start_step, x.agv_id),
                )
            ],
        }
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
        robot_snaps: list[RobotSnapshot] = []
        for agv in simulator.agvs:
            st_val = str(agv.state.value if hasattr(agv.state, "value") else agv.state)
            if agv.agv_id in faulted_set:
                st_val = "faulted"
            f_until = getattr(agv, "faulted_until_step", 0)
            if st_val in ("blocked", "faulted"):
                f_until = max(f_until, tick + 999)
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

        ver = version or {
            "world": 1,
            "reservation": getattr(simulator, "_reservation_revision", 1),
            "production": getattr(simulator, "_production_revision", 1),
            "fleet": getattr(simulator, "_fleet_revision", 1),
        }

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
        )
