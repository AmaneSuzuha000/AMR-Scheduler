"""Bounded operator transport, polled only by the durable fleet owner's executor."""
from __future__ import annotations

from copy import deepcopy
import json
import socket
import time
import uuid

from amr_sched.entities import MaterialType, Task, TaskPriority


MAX_LINE = 65536


def _line(value):
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(',', ':'))+'\n').encode()


class FleetControl:
    """Validate identity and deduplicate requests before invoking the sole owner.

    Session identity changes on restart. A retry within that session returns the
    original receipt; request IDs are never evicted and then executed a second time.
    """

    def __init__(self, owner, *, max_requests=4096):
        self.owner = owner
        self.session_id = uuid.uuid4().hex
        self.max_requests = max_requests
        self.receipts = {}

    def state(self):
        sim = self.owner.sim
        first = next(iter(sim.executions.values()))
        return dict(session_id=self.session_id, scene_id=first.identity['scene_id'],
                    run_id=first.identity['run_id'], paused=sim._operator_paused,
                    operations=['submit_task', 'pause', 'resume', 'preview_fleet_membership'], 
                    routes={r:g.recovery_token() for r,g in sim.executions.items()},
                    faults={r:g.fault for r,g in sim.executions.items()},
                    task_total=len(sim.tasks),
                    tasks=[dict(task_id=t.task_id,pickup=t.pickup,dropoff=t.dropoff,
                                status=t.status.value,assigned_agv=t.assigned_agv)
                           for t in sim.tasks[-100:]])

    def handle(self, request):
        """Execute at most once, on the calling owner thread, after strict validation."""
        reply = dict(type='control_result', request_id=None, operation=None, accepted=False)
        try:
            if type(request) is not dict or set(request) != {
                'type', 'session_id', 'scene_id', 'run_id', 'request_id', 'operation', 'payload'}:
                raise ValueError('control request fields differ')
            reply.update(request_id=request['request_id'], operation=request['operation'])
            if request['type'] != 'control_request':
                raise ValueError('control_request required')
            state = self.state()
            if any(request[k] != state[k] for k in ('session_id', 'scene_id', 'run_id')):
                raise ValueError('current scene, run and owner session required')
            rid = request['request_id']
            if type(rid) is not str or not 1 <= len(rid) <= 128:
                raise ValueError('bounded request_id required')
            if type(request['operation']) is not str or type(request['payload']) is not dict:
                raise ValueError('named operation and object payload required')
            canonical = _line(request)
            if len(canonical) > MAX_LINE:
                raise ValueError('request exceeds line limit')
            if rid in self.receipts:
                previous, receipt = self.receipts[rid]
                if previous != canonical:
                    raise ValueError('request_id already used with different content')
                return deepcopy(receipt)
            if len(self.receipts) >= self.max_requests:
                raise ValueError('session receipt capacity reached; restart requires reconciliation')
            op, payload = request['operation'], request['payload']
            try:
                if self.owner.failed or self.owner.closing or not self.owner.ready:
                    raise ValueError('active durable fleet owner required')
                if op == 'preview_fleet_membership':
                    if set(payload) != {'robot_id', 'routes'} or type(payload['robot_id']) is not str:
                        raise ValueError('exact robot_id and global route tokens required')
                    from .native_physical_admission_audit import inspect_physical_membership
                    assessment=inspect_physical_membership(self.owner,payload['robot_id'],
                        expected_routes=payload['routes'],now=time.monotonic())
                    # A successful preview is NEVER acceptance of a physical
                    # membership mutation or a motor permit.
                    answer=dict(accepted=False,reason='PREVIEW_ONLY_NO_PHYSICAL_COMMIT',
                                readiness=assessment)
                elif op == 'submit_task':
                    answer = self._submit(payload)
                elif op in {'pause', 'resume'}:
                    if set(payload) != {'routes'} or payload['routes'] != state['routes']:
                        raise ValueError('current versions for all physical robots required')
                    method = self.owner.pause_fleet if op == 'pause' else self.owner.resume_fleet
                    answer = method(payload['routes'])
                else:
                    raise ValueError('unknown fleet operation')
                reply.update(answer)
            except (ValueError, TypeError, KeyError) as exc:
                reply.update(reason=str(exc))
            reply['control'] = self.state()
            # Even a refusal consumes the ID: a retry cannot become a new command
            # later when stop/road/freshness conditions have changed.
            self.receipts[rid] = canonical, deepcopy(reply)
        except (ValueError, TypeError, KeyError) as exc:
            reply.update(reason=str(exc))
        return reply

    def _submit(self, payload):
        allowed = {'task_id', 'pickup', 'dropoff', 'priority', 'due_step', 'material_type'}
        if not {'task_id', 'pickup', 'dropoff'} <= set(payload) or set(payload)-allowed:
            raise ValueError('task fields differ')
        if any(type(payload[k]) is not str or not 1 <= len(payload[k]) <= 128
               for k in ('task_id', 'pickup', 'dropoff')):
            raise ValueError('bounded task and node identifiers required')
        priority = payload.get('priority', 0)
        due = payload.get('due_step')
        if type(priority) is not int or priority not in (0, 1):
            raise ValueError('priority must be 0 or 1')
        if due is not None and type(due) is not int:
            raise ValueError('due_step must be an integer or null')
        material = payload.get('material_type')
        task = Task(payload['task_id'], payload['pickup'], payload['dropoff'],
                    release_step=self.owner.sim._step_cursor, due_step=due,
                    priority=TaskPriority(priority),
                    material_type=MaterialType(material) if material is not None else None)
        # Submitting an urgent task doesn't silently preempt physical cargo.
        # Physical preemption remains the separate stopped-node owner procedure.
        return self.owner.submit_physical_task(task)


class FleetControlServer:
    """Nonblocking loopback service with bounded work on the existing executor.

    No network thread can mutate the simulator. Slow receivers are dropped,
    without delaying leases or allowing an unbounded write queue.
    """

    def __init__(self, owner, port=0):
        self.owner = owner
        self.control = FleetControl(owner)
        self.listener = socket.socket()
        try:
            self.listener.bind(('127.0.0.1', port))
            self.listener.listen(8)
            self.listener.setblocking(False)
        except Exception:
            self.listener.close()
            raise
        self.port = self.listener.getsockname()[1]
        self.clients = {}
        self.last_frame = 0.

    def _drop(self, client):
        self.clients.pop(client, None)
        client.close()

    def _queue(self, client, value):
        data = _line(value)
        if len(data) > MAX_LINE and value.get('type') == 'frame':
            # Keep the complete owner log. The transport can carry a shorter
            # authoritative trail window; it never invents intermediate poses.
            value = {**value, 'trails':{r:points[-64:] for r,points in value.get('trails',{}).items()}}
            data = _line(value)
        state = self.clients[client]
        if len(data) > MAX_LINE or len(state['out'])+len(data) > MAX_LINE*4:
            self._drop(client)
        else:
            state['out'].extend(data)

    def poll(self, frame):
        """Accept, read bounded commands, and publish frames without blocking I/O."""
        started = time.monotonic()
        for _ in range(2):
            try:
                client, _ = self.listener.accept()
            except BlockingIOError:
                break
            if len(self.clients) >= 8:
                client.close()
                continue
            client.setblocking(False)
            self.clients[client] = dict(buffer=bytearray(), out=bytearray())
            self._queue(client, {**self.owner.builder.hello(), 'control':self.control.state()})
        broadcast = started-self.last_frame >= .1
        if broadcast:
            self.last_frame = started
        for client, state in list(self.clients.items()):
            if time.monotonic()-started > .01:
                break
            # Round-robin the bounded client set when a command uses this tick's budget.
            self.clients.pop(client)
            self.clients[client] = state
            try:
                try:
                    data = client.recv(8192)
                except BlockingIOError:
                    data = None
                if data == b'':
                    self._drop(client)
                    continue
                if data:
                    state['buffer'].extend(data)
                for _ in range(2):
                    index = state['buffer'].find(b'\n')
                    if index < 0:
                        break
                    if index+1 > MAX_LINE:
                        self._drop(client)
                        break
                    raw = bytes(state['buffer'][:index])
                    del state['buffer'][:index+1]
                    try:
                        request = json.loads(raw)
                        answer = self.control.handle(request)
                    except (ValueError, UnicodeError):
                        answer = dict(type='control_result', accepted=False, reason='invalid JSON')
                    self.owner.record('operator_request', result=answer)
                    self._queue(client, answer)
                    if client not in self.clients:
                        break
                if client not in self.clients:
                    continue
                if len(state['buffer']) > MAX_LINE:
                    self._drop(client)
                    continue
                if broadcast:
                    # Commands were processed after this tick's observation.
                    # Publish the new state only on the next tick, preventing a
                    # new control token being paired with an older frame.
                    self._queue(client, frame)
                if client in self.clients and state['out']:
                    try:
                        sent = client.send(state['out'])
                        del state['out'][:sent]
                    except BlockingIOError:
                        pass
            except OSError:
                self._drop(client)

    def close(self):
        for client in list(self.clients):
            self._drop(client)
        self.listener.close()
