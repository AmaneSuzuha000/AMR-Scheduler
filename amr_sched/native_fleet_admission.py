"""Opt-in fleet admission on the actual AMR-Scheduler *software* Simulator.

Changes only future allocator eligibility. Never clears existing reservations,
transfers physical leases, or grants hardware movement. Physical mode is refused.
"""
from __future__ import annotations
from dataclasses import dataclass
from amr_sched.entities import AgvState, TaskStatus
from amr_sched.network import NodeKind

@dataclass(frozen=True)
class FleetChange:
    robot_id: str
    action: str
    expected_tick: int
    expected_revision: int
    active_before: frozenset[str]

class NativeFleetAdmission:
    def __init__(self, sim, *, initial_active=None, parking_nodes=None):
        if sim.executions:
            raise ValueError("PHYSICAL_EXECUTION_REQUIRES_DURABLE_GLOBAL_COMMIT")
        if not hasattr(sim, "_load_end"):
            raise ValueError("reset simulator before binding admission")
        if getattr(sim, "_fleet_admission", None) is not None:
            raise ValueError("one admission owner per simulator")
        ids = {a.agv_id for a in sim.agvs}
        active = ids if initial_active is None else set(initial_active)
        if not active or not active <= ids:
            raise ValueError("initial active set must be a nonempty subset")
        parking = ({n.node_id for n in sim.network.nodes.values()
                    if n.kind is NodeKind.BUFFER} if parking_nodes is None
                   else set(parking_nodes))
        if not parking <= set(sim.network.nodes):
            raise ValueError("unknown parking node")
        self.sim = sim
        self.active = frozenset(active)
        self.parking = frozenset(parking)
        self.revision = 0
        self.events = []
        for a in sim.agvs:
            if a.agv_id not in self.active:
                self._check_idle_parked(a.agv_id)
        sim._fleet_admission = self

    def eligible(self, robot_id):
        return robot_id in self.active

    def _check_idle_parked(self, robot_id):
        sim = self.sim
        a = next((a for a in sim.agvs if a.agv_id == robot_id), None)
        if a is None:
            raise ValueError("UNKNOWN_ROBOT")
        if a.cur_node not in self.parking:
            raise ValueError("NOT_AT_VERIFIED_PARKING")
        if a.cur_task is not None or a.state is not AgvState.IDLE or a.route:
            raise ValueError("ROBOT_NOT_IDLE")
        if robot_id in sim._jobs or robot_id in sim._transit:
            raise ValueError("UNSETTLED_JOB_OR_TRANSIT")
        if robot_id in sim._node_claim:
            raise ValueError("FUTURE_NODE_CLAIM")
        if any(t.assigned_agv == robot_id and t.status is not TaskStatus.COMPLETED
               for t in sim.tasks):
            raise ValueError("IRREVOCABLE_TASK_OWNERSHIP")
        pins = sim._res.pins() if hasattr(sim._res, "pins") else {}
        if any(robot_id in owners for group in ("nodes", "edges")
               for owners in pins.get(group, {}).values()):
            raise ValueError("ACTIVE_PHYSICAL_PIN")
        # Never release the current occupancy. Stationary robots remain in
        # Simulator's actual occupancy and all collision/edge checks.

    def prepare(self, *, robot_id, action, expected_tick, expected_revision):
        if action not in ("ACTIVATE", "DEACTIVATE"):
            raise ValueError("UNKNOWN_FLEET_ACTION")
        if type(expected_tick) is not int or expected_tick != self.sim._step_cursor:
            raise ValueError("STALE_OWNER_TICK")
        if type(expected_revision) is not int or expected_revision != self.revision:
            raise ValueError("STALE_FLEET_REVISION")
        if robot_id not in {a.agv_id for a in self.sim.agvs}:
            raise ValueError("UNKNOWN_ROBOT")
        if action == "ACTIVATE" and robot_id in self.active:
            raise ValueError("ALREADY_ACTIVE")
        if action == "DEACTIVATE" and robot_id not in self.active:
            raise ValueError("ALREADY_INACTIVE")
        if action == "DEACTIVATE" and len(self.active) == 1:
            raise ValueError("CANNOT_DEACTIVATE_LAST_ROBOT")
        self._check_idle_parked(robot_id)
        return FleetChange(robot_id, action, expected_tick,
                           expected_revision, self.active)

    def commit(self, proposed):
        if not isinstance(proposed, FleetChange):
            raise ValueError("INVALID_PROPOSAL")
        if proposed.active_before != self.active:
            raise ValueError("STALE_FLEET_MEMBERSHIP")
        if self.prepare(robot_id=proposed.robot_id, action=proposed.action,
                        expected_tick=proposed.expected_tick,
                        expected_revision=proposed.expected_revision) != proposed:
            raise ValueError("PROPOSAL_CONTEXT_CHANGED")
        nxt = set(self.active)
        if proposed.action == "ACTIVATE":
            nxt.add(proposed.robot_id)
        else:
            nxt.remove(proposed.robot_id)
        self.active = frozenset(nxt)
        self.revision += 1
        receipt = dict(event="FLEET_MEMBERSHIP_COMMITTED",
                       tick=proposed.expected_tick, action=proposed.action,
                       robot_id=proposed.robot_id, revision=self.revision,
                       active=sorted(self.active),
                       effect="FUTURE_ALLOCATION_ONLY", physical_authorization=False)
        self.events.append(dict(receipt))
        return receipt
