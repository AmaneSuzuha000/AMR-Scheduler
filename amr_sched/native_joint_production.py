"""Opt-in software-only production/transport joint transaction for SYSTEM12.

The operation calendar is *not* inferred from delivery completion: kits, precedence,
bay, crew, and tool availability all gate real starts. Only unassigned PENDING kits
may change destination. Physical execution is deliberately unsupported.
"""
from __future__ import annotations

from dataclasses import dataclass
from amr_sched.entities import TaskStatus
from amr_sched.network import NodeKind

@dataclass(frozen=True)
class ResourceChoice:
    bay: str
    crew: str
    tool: str

@dataclass(frozen=True)
class ProductionOperation:
    op_id: str
    kits: tuple[str, ...]
    choices: tuple[ResourceChoice, ...]
    duration: int
    predecessor: str | None = None

@dataclass(frozen=True)
class PreparedJointChange:
    expected_tick: int
    production_revision: int
    op_id: str
    choice_index: int
    previous_choice: int
    fleet_proposal: object | None

class NativeJointProduction:
    """The same native Simulator is the sole owner of the production calendar."""

    def __init__(self, sim, operations, *, fleet=None):
        if sim.executions:
            raise ValueError("PHYSICAL_CROSS_DOMAIN_ATOMICITY_NOT_VERIFIED")
        if not hasattr(sim, "_load_end"):
            raise ValueError("reset simulator before binding production")
        if getattr(sim, "_native_production", None) is not None:
            raise ValueError("only one production owner per simulator")
        if fleet is not None and (fleet.sim is not sim or
                                  getattr(sim, "_fleet_admission", None) is not fleet):
            raise ValueError("fleet owner mismatch")
        ops = list(operations)
        if not ops or len({o.op_id for o in ops}) != len(ops):
            raise ValueError("nonempty unique operations required")
        self.sim = sim
        self.operations = {op.op_id: op for op in ops}
        self.choice = {op.op_id: 0 for op in ops}
        self.started = {op.op_id: None for op in ops}
        self.finished = {op.op_id: None for op in ops}
        self.fleet = fleet
        self.revision = 0
        self.events = []
        seen = set()
        for op in ops:
            if (type(op.duration) is not int or op.duration <= 0
                    or not op.kits or not op.choices):
                raise ValueError("invalid operation processing requirements")
            if op.predecessor is not None and op.predecessor not in self.operations:
                raise ValueError("unknown predecessor")
            for c in op.choices:
                if c.bay not in sim.network.nodes or sim.network.node(c.bay).kind is not NodeKind.WORKSTATION:
                    raise ValueError("qualified bay must exist and be a workstation")
                if not c.crew or not c.tool:
                    raise ValueError("qualified crew/tool identifiers required")
            for task_id in op.kits:
                if task_id in seen or task_id not in sim._task_by_id:
                    raise ValueError("each kit must be owned by exactly one operation")
                seen.add(task_id)
                if sim._task_by_id[task_id].dropoff != op.choices[0].bay:
                    raise ValueError("initial kit destination and qualified bay mismatch")
        for op in ops:
            history = set()
            current = op.op_id
            while current is not None:
                if current in history:
                    raise ValueError("cyclic production precedence")
                history.add(current)
                current = self.operations[current].predecessor
        sim._native_production = self

    def all_finished(self):
        return all(x is not None for x in self.finished.values())

    def advance(self, now):
        """Triggered *after* a native transport step; no synthetic delivery receipts."""
        for op_id, op in self.operations.items():
            start = self.started[op_id]
            if start is not None and self.finished[op_id] is None and now >= start+op.duration:
                self.finished[op_id] = now
                self.events.append(dict(event="PRODUCTION_FINISHED", op_id=op_id, tick=now))
        for op_id, op in self.operations.items():
            if self.started[op_id] is not None:
                continue
            if op.predecessor and self.finished[op.predecessor] is None:
                continue
            if any(self.sim._task_by_id[k].status is not TaskStatus.COMPLETED for k in op.kits):
                continue
            resource = op.choices[self.choice[op_id]]
            # Exact crew/tool/bay calendars. Completed operations release their resources.
            def occupied(other):
                if self.started[other] is None or self.finished[other] is not None:
                    return False
                use = self.operations[other].choices[self.choice[other]]
                return bool(use.crew == resource.crew or use.tool == resource.tool or use.bay == resource.bay)
            if any(occupied(other) for other in self.operations if other != op_id):
                continue
            self.started[op_id] = now
            self.events.append(dict(event="PRODUCTION_STARTED", op_id=op_id, tick=now,
                                    bay=resource.bay, crew=resource.crew, tool=resource.tool))

    def prepare(self, *, op_id, choice_index, expected_tick, expected_revision,
                fleet_action=None):
        sim = self.sim
        if type(expected_tick) is not int or expected_tick != sim._step_cursor:
            raise ValueError("STALE_OWNER_TICK")
        if type(expected_revision) is not int or expected_revision != self.revision:
            raise ValueError("STALE_PRODUCTION_REVISION")
        op = self.operations.get(op_id)
        if op is None:
            raise ValueError("UNKNOWN_OPERATION")
        if type(choice_index) is not int or not 0 <= choice_index < len(op.choices):
            raise ValueError("UNQUALIFIED_RESOURCE_COMBINATION")
        if self.choice[op_id] == choice_index:
            raise ValueError("PRODUCTION_CHOICE_UNCHANGED")
        if self.started[op_id] is not None or self.finished[op_id] is not None:
            raise ValueError("STARTED_OPERATION_IMMUTABLE")
        for kit in op.kits:
            task = sim._task_by_id[kit]
            if task.status is not TaskStatus.PENDING or task.assigned_agv is not None:
                raise ValueError("KIT_PICKUP_OR_ASSIGNMENT_COMMITTED")
            if any(job.task_id == kit for job in sim._jobs.values()):
                raise ValueError("KIT_HAS_EXECUTION_JOB")
        proposal = None
        if fleet_action is not None:
            if self.fleet is None:
                raise ValueError("MISSING_FLEET_OWNER")
            proposal = self.fleet.prepare(**fleet_action)
        return PreparedJointChange(expected_tick, expected_revision,
                                   op_id, choice_index, self.choice[op_id], proposal)

    def commit(self, candidate):
        if not isinstance(candidate, PreparedJointChange):
            raise ValueError("INVALID_JOINT_PROPOSAL")
        fp = candidate.fleet_proposal
        kwargs = dict(op_id=candidate.op_id, choice_index=candidate.choice_index,
                      expected_tick=candidate.expected_tick,
                      expected_revision=candidate.production_revision)
        if fp is not None:
            kwargs["fleet_action"] = dict(robot_id=fp.robot_id, action=fp.action,
                                          expected_tick=fp.expected_tick,
                                          expected_revision=fp.expected_revision)
        if self.prepare(**kwargs) != candidate or self.choice[candidate.op_id] != candidate.previous_choice:
            raise ValueError("STALE_JOINT_PROPOSAL")
        sim = self.sim
        op_id = candidate.op_id
        op = self.operations[op_id]
        old = op.choices[candidate.previous_choice]
        new = op.choices[candidate.choice_index]
        # All preflights completed. Mutations are synchronous on the one Simulator
        # executor; rollback in-memory if any unexpected exception occurs.
        old_drops = {kit:sim._task_by_id[kit].dropoff for kit in op.kits}
        prev_events = len(self.events)
        prev_revision = self.revision
        fleet_before = None
        if fp is not None:
            fleet_before = (self.fleet.active, self.fleet.revision, len(self.fleet.events))
        try:
            for kit in op.kits:
                sim._task_by_id[kit].dropoff = new.bay
            self.choice[op_id] = candidate.choice_index
            self.revision += 1
            fleet_receipt = self.fleet.commit(fp) if fp is not None else None
            receipt = dict(event="JOINT_PRODUCTION_FLEET_COMMIT", tick=sim._step_cursor,
                           op_id=op_id, production_revision=self.revision,
                           bay_before=old.bay, bay_after=new.bay,
                           crew_before=old.crew, crew_after=new.crew,
                           tool_before=old.tool, tool_after=new.tool,
                           updated_kits=sorted(op.kits),
                           future_destination_changes=sum(old_drops[k] != new.bay for k in op.kits),
                           fleet=fleet_receipt, physical_authorization=False)
            self.events.append(receipt)
            return receipt
        except BaseException:
            for kit,dst in old_drops.items():
                sim._task_by_id[kit].dropoff = dst
            self.choice[op_id] = candidate.previous_choice
            self.revision = prev_revision
            del self.events[prev_events:]
            if fleet_before is not None:
                self.fleet.active, self.fleet.revision, n = fleet_before
                del self.fleet.events[n:]
            raise
