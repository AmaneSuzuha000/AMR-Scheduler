"""Durable, versioned physical-runtime snapshots with an explicit type allowlist.

Loading never imports a type named by the file. A checksum detects corruption;
it is not a signature or a substitute for trusted local file permissions.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

from amr_sched.entities import Agv, AgvState, MaterialType, Task, TaskPriority, TaskStatus
from amr_sched.monitor import Decision
from amr_sched.network import Edge, Node, NodeKind
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.reservation import Interval
from amr_sched.simulation import Hop, SimulationConfig, StepRecord, _Job

_TYPES = {cls.__name__: cls for cls in (
    Agv, AgvState, MaterialType, Task, TaskPriority, TaskStatus, Decision,
    Node, Edge, NodeKind, ReschedulePolicy, Interval, Hop, SimulationConfig, StepRecord, _Job)}
_REFERENCES = {"network", "config", "planner", "allocator", "execution", "executions", "handlings", "inventory", "_res", "_monitor", "_task_by_id"}


def _encode(value):
    if isinstance(value, Enum):
        if type(value).__name__ not in _TYPES:
            raise ValueError("unsupported enum")
        return {"type": type(value).__name__, "value": value.value}
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"float": repr(value)}
    if isinstance(value, random.Random):
        return {"random": _encode(value.getstate())}
    if is_dataclass(value) and type(value).__name__ in _TYPES:
        return {"type": type(value).__name__, "fields": {f.name: _encode(getattr(value, f.name)) for f in fields(value)}}
    if type(value) in (list, tuple, set):
        return {type(value).__name__: [_encode(v) for v in value]}
    if type(value) is dict:
        return {"dict": [[_encode(k), _encode(v)] for k, v in value.items()]}
    raise ValueError("unsupported checkpoint type: " + type(value).__name__)


def _decode(value):
    if value is None or type(value) in (bool, int, str, float):
        return value
    if not isinstance(value, dict):
        raise ValueError("invalid checkpoint value")
    keys = set(value)
    if keys == {"float"} and value["float"] in {"inf", "-inf", "nan"}:
        return float(value["float"])
    if keys == {"random"}:
        rng = random.Random()
        rng.setstate(_decode(value["random"]))
        return rng
    for name, constructor in (("list", list), ("tuple", tuple), ("set", set)):
        if keys == {name}:
            return constructor(_decode(v) for v in value[name])
    if keys == {"dict"}:
        pairs = [(_decode(k), _decode(v)) for k, v in value["dict"]]
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate checkpoint key")
        return result
    cls = _TYPES.get(value.get("type"))
    if cls is not None:
        if issubclass(cls, Enum) and keys == {"type", "value"}:
            return cls(value["value"])
        if is_dataclass(cls) and keys == {"type", "fields"}:
            if set(value["fields"]) != {f.name for f in fields(cls)}:
                raise ValueError("checkpoint dataclass fields differ")
            return cls(**{k: _decode(v) for k, v in value["fields"].items()})
    raise ValueError("unknown checkpoint type or fields")


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()


def definition(sim, execution_files=()):
    """Bind recovery to the same topology, initial scenario and implementation."""
    if not sim.executions:
        raise ValueError('physical simulator required')
    task_runtime = {"status", "assigned_agv", "assigned_step", "pickup_step", "complete_step"}
    agv_runtime = {"state", "cur_node", "cur_task", "route", "depart_step", "remaining_job_steps",
                   "busy_steps", "idle_steps", "blocked_steps", "loaded_steps", "deadhead_steps"}
    def initial(items, excluded):
        return [{f.name: _encode(getattr(v, f.name)) for f in fields(v) if f.name not in excluded} for v in items]
    root = Path(__file__).parent
    code = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*.py"))}
    execution_code = {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in execution_files}
    planner = {k: v for k, v in vars(sim.planner).items() if k != "_rng"}
    gates = sim.executions
    identity = lambda gate: {k:v for k,v in gate.identity.items() if k!='run_id'}
    value = dict(nodes=_encode(list(sim.network.nodes.values())), edges=_encode(list(sim.network.edges.values())),
                 config=_encode(sim.config), tasks=initial(sim.tasks, task_runtime), agvs=initial(sim.agvs, agv_runtime),
                 planner=[type(sim.planner).__name__, _encode(planner)],
                 allocator=[type(sim.allocator).__name__, _encode(vars(sim.allocator))],
                 execution=(identity(sim.execution) if sim.execution else
                            {robot:identity(gate) for robot,gate in sorted(gates.items())}),
                 tolerance=(sim.execution.tolerance if sim.execution else
                            {robot:gate.tolerance for robot,gate in sorted(gates.items())}),
                 timeout=(sim.execution.timeout if sim.execution else
                          {robot:gate.timeout for robot,gate in sorted(gates.items())}), code=code,
                 namespace_routes={robot:gate.namespace_routes for robot,gate in sorted(gates.items())},
                 execution_code=execution_code)
    if sim.handlings:
        value['handling'] = {r:dict(source=h.source, freshness_s=h.freshness_s)
                             for r, h in sorted(sim.handlings.items())}
    if sim.inventory is not None:
        value['inventory'] = sim.inventory.definition()
    return hashlib.sha256(_json(value)).hexdigest()


def capture(sim, fingerprint, runtime):
    """Capture on the single owner executor, before any native actuation."""
    if not sim.executions or not hasattr(sim, "_load_end"):
        raise ValueError("reset physical simulator required")
    data = dict(definition=fingerprint,
                        state={k: v for k, v in vars(sim).items() if k not in _REFERENCES},
                        reservations=vars(sim._res), monitor=sim._monitor.history,
                        gate=(vars(sim.execution) if sim.execution else
                              {robot:vars(gate) for robot,gate in sim.executions.items()}),
                        planner=vars(sim.planner), runtime=runtime)
    if sim.handlings:
        data['handling'] = {r:{k:v for k,v in vars(h).items() if k != 'execution'}
                            for r,h in sim.handlings.items()}
    if sim.inventory is not None:
        data['inventory'] = sim.inventory.snapshot()
    return _encode(data)


def restore(sim, payload, fingerprint):
    """Restore resources and load, discard all old freshness and latch restart."""
    data = _decode(payload)
    schema = {"definition", "state", "reservations", "monitor", "gate", "planner", "runtime"}
    if sim.handlings:
        schema.add('handling')
    if sim.inventory is not None:
        schema.add('inventory')
    if set(data) != schema:
        raise ValueError("checkpoint schema fields differ")
    if data["definition"] != fingerprint:
        raise ValueError("checkpoint definition mismatch")
    expected = set(vars(sim)) - _REFERENCES
    recorded = ({sim.execution.identity['robot_id']:data['gate']} if sim.execution else data['gate'])
    if set(recorded) != set(sim.executions):
        raise ValueError('checkpoint execution identities differ')
    for actual, wanted in ((set(data["state"]), expected),
                           (set(data["reservations"]), set(vars(sim._res))),
                           (set(data["planner"]), set(vars(sim.planner)))):
        if actual != wanted:
            raise ValueError("checkpoint state fields differ")
    if any(set(recorded[robot])!=set(vars(gate)) for robot,gate in sim.executions.items()):
        raise ValueError('checkpoint state fields differ')
    # Validate references before mutating the live object. Strict native-state
    # reconciliation remains the adapter's responsibility.
    tasks = data["state"]["tasks"]
    agvs = data["state"]["agvs"]
    if (len({t.task_id for t in tasks}) != len(tasks) or len(agvs) != len(sim.executions)
            or {a.agv_id for a in agvs} != set(sim.executions)):
        raise ValueError("checkpoint task/vehicle identities invalid")
    for robot, gate in recorded.items():
        if gate['identity']['robot_id'] != robot:
            raise ValueError("checkpoint execution identity invalid")
    identities={tuple(g['identity'][k] for k in ('scene_id','run_id','frame_id','time_base','source'))
                for g in recorded.values()}
    if len(identities)!=1:
        raise ValueError('checkpoint fleet authority identities differ')
    if sim.handlings:
        if set(data['handling']) != set(sim.handlings):
            raise ValueError('checkpoint handling endpoints differ')
        for robot, live in sim.handlings.items():
            saved = data['handling'][robot]
            if (set(saved) != set(vars(live)) - {'execution'}
                    or saved['source'] != live.source or saved['freshness_s'] != live.freshness_s):
                raise ValueError('checkpoint handling authority differs')
            command = saved['command']
            if command is not None and (command['robot_id'] != robot
                    or command['run_id'] != recorded[robot]['identity']['run_id']):
                raise ValueError('checkpoint handling identity differs')
    if sim.inventory is not None:
        sim.inventory.validate_snapshot(data['inventory'])
        if any(not sim.inventory.validate_task(t) for t in tasks):
            raise ValueError('checkpoint inventory task recipes differ')
        task_map = {t.task_id: t for t in tasks}
        for kind in ('reservations', 'committed'):
            for task_id, item in data['inventory'][kind].items():
                task = task_map.get(task_id)
                robot = item['robot_id']
                if (task is None or task.assigned_agv != robot or robot not in sim.executions
                        or task.dropoff != item['node_id']
                        or task.status not in (TaskStatus.PICKED, TaskStatus.COMPLETED)
                        or kind == 'reservations' and task.status is not TaskStatus.PICKED):
                    raise ValueError('checkpoint inventory and task ownership differ')
                if kind == 'committed' and not any(
                        event.get('event') == 'HANDLING_COMPLETED'
                        and all(event.get('receipt', {}).get(key) == value for key, value in dict(
                            task_id=task_id, robot_id=robot, node_id=item['node_id'], operation='UNLOAD',
                            handling_id=item['handling_id'], evidence_id=item['evidence_id']).items())
                        for event in data['handling'][robot]['events']):
                    raise ValueError('checkpoint inventory lacks accepted handling provenance')
    sim.__dict__.update(data["state"])
    sim._res.__dict__.update(data["reservations"])
    sim._monitor.history = data["monitor"]
    sim.planner.__dict__.update(data["planner"])
    sim._task_by_id = {t.task_id: t for t in sim.tasks}
    if sim.inventory is not None:
        sim.inventory.restore(data['inventory'])
    for agv in sim.agvs:
        gate=sim.execution_for(agv.agv_id)
        gate.__dict__.update(recorded[agv.agv_id])
        gate.pose = None
        gate.last_receipt = None
        gate.stopped = False
        gate._outbox = []
        gate.fault = None
        gate.fail("runtime restarted; reconciliation required")
        agv.state = AgvState.BLOCKED
        if agv.agv_id in sim.handlings:
            handling = sim.handlings[agv.agv_id]
            handling.__dict__.update(data['handling'][agv.agv_id])
            job = sim._jobs.get(agv.agv_id)
            waiting = (job is not None and agv.agv_id not in sim._transit and job.idx >= len(job.hops)
                       and job.leg in {'to_pickup', 'to_dropoff'})
            command = handling.command
            current_transfer = (waiting and command is not None
                and (command['task_id'], command['operation'], command['node_id']) ==
                    (job.task_id, 'LOAD' if job.leg == 'to_pickup' else 'UNLOAD', agv.cur_node))
            if current_transfer:
                # Keep material/dock ownership; require a new support observation
                # after restart. This is reconciliation, not permission to repeat
                # an uncertain mechanical transfer. Arrival can precede issuing
                # the next request: keep the completed previous transfer as
                # history, so it cannot block that new request after recovery.
                handling.revision += 1
                handling.command = dict(handling.command, revision=handling.revision,
                    handling_id=f"{gate.identity['run_id']}:{agv.agv_id}:handling:{handling.revision}",
                    reconcile_only=True)
                handling.completed = False
                handling.last_receipt = -math.inf
                handling.events.append(dict(event='HANDLING_RECONCILIATION_REQUIRED',
                                            command=dict(handling.command)))
    return data["runtime"]


class CheckpointStore:
    """One writer per local file; atomic replace with file and directory flush."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self.path.with_suffix(self.path.suffix + ".lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self._lock.seek(0)
                if self._lock.read(1) == b"":
                    self._lock.write(b"0")
                    self._lock.flush()
                self._lock.seek(0)
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock.close()
            raise RuntimeError("checkpoint already has a live owner") from None

    def save(self, payload):
        body = _json(payload)
        data = _json(dict(schema=1, sha256=hashlib.sha256(body).hexdigest(), payload=payload))
        fd, name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(name, self.path)
            if os.name != "nt":
                directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def load(self):
        data = json.loads(self.path.read_bytes())
        if set(data) != {"schema", "sha256", "payload"} or data["schema"] != 1:
            raise ValueError("unsupported checkpoint schema")
        if hashlib.sha256(_json(data["payload"])).hexdigest() != data["sha256"]:
            raise ValueError("checkpoint checksum mismatch")
        return data["payload"]

    def close(self):
        self._lock.close()
