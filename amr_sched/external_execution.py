"""Feedback gate for one physical vehicle; routing stays in Simulator.

Coordinates and tolerance are in the explicitly named network metric frame.
Receipt time is monotonic wall time, independent of the producer's ROS clock.
Faults latch: a stop acknowledgement is evidence of stopping, not permission to
discard a task, release a corridor, or restart motion.
"""
from __future__ import annotations

import math
import uuid


class ExternalHandling:
    """Identity and freshness gate for a measured load/unload acknowledgement.

    The equipment adapter must validate support/contact and record its raw proof.
    This gate validates the resulting receipt; it does not infer physical loading
    from vehicle arrival, a timer, or an unverified visual animation.
    """

    def __init__(self, execution: ExternalExecution, *, source: str,
                 freshness_s: float = .15) -> None:
        if not isinstance(source, str) or not source or source == 'localized':
            raise ValueError("explicit equipment observation source required")
        if not math.isfinite(freshness_s) or freshness_s <= 0:
            raise ValueError("positive finite handling freshness required")
        self.execution = execution
        self.source = source
        self.freshness_s = freshness_s
        self.command = None
        self.completed = False
        self.revision = 0
        self.last_receipt = -math.inf
        self.events = []

    def require(self, task_id, operation, node_id, target_xy, source_now):
        """Create one request and retain it while its task holds the dock."""
        if operation not in {'LOAD', 'UNLOAD'} or not math.isfinite(source_now):
            raise ValueError("finite handling time and supported operation required")
        key = (task_id, operation, node_id)
        if self.command and tuple(self.command[k] for k in ('task_id', 'operation', 'node_id')) == key:
            return self.completed
        if self.command and not self.completed:
            raise RuntimeError("unsettled handling request must retain its task")
        self.revision += 1
        identity = {**self.execution.identity, 'source': self.source}
        self.command = dict(**identity, task_id=task_id, operation=operation, node_id=node_id,
            handling_id=f"{identity['run_id']}:{identity['robot_id']}:handling:{self.revision}",
            revision=self.revision, target_xy=list(target_xy), issued_at=source_now)
        self.completed = False
        self.events.append(dict(event='HANDLING_REQUESTED', command=dict(self.command)))
        return False

    def feedback(self, message, received_at, source_now):
        """Accept a single fresh, stopped, correctly supported material receipt."""
        reject = lambda reason: dict(accepted=False, reason=reason)
        command = self.command
        if not command or self.completed or not isinstance(message, dict):
            return reject('one outstanding handling request required')
        if set(message) != set(command) | {
                'event', 'timestamp', 'robot_xy', 'stopped', 'supported_by', 'evidence_id'}:
            return reject('explicit handling receipt fields required')
        if any(message.get(k) != v for k, v in command.items()):
            return reject('handling identity mismatch')
        if (type(received_at) not in (int, float) or not math.isfinite(received_at)
                or received_at < self.last_receipt
                or type(source_now) not in (int, float) or not math.isfinite(source_now)):
            return reject('finite monotone receipt time required')
        gate = self.execution
        gate.poll(received_at)
        if (not gate.bound or gate.fault or gate.command is not None or gate.state != 'idle'
                or gate.pose is None or gate.last_receipt is None or received_at < gate.last_receipt):
            return reject('fresh healthy stationary execution required')
        stamp = message.get('timestamp')
        if (type(stamp) not in (int, float) or not math.isfinite(stamp)
                or not command['issued_at'] <= stamp <= source_now
                or source_now-stamp > self.freshness_s
                or abs(gate.pose['timestamp']-stamp) > self.freshness_s):
            return reject('fresh handling and vehicle source times required')
        xy = message.get('robot_xy')
        if (not isinstance(xy, (list, tuple)) or len(xy) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in xy)
                or math.dist(xy, command['target_xy']) > gate.tolerance
                or math.dist(xy, (gate.pose['x'], gate.pose['y'])) > gate.tolerance):
            return reject('handling must occur at the retained dock')
        expected = gate.identity['robot_id'] if command['operation'] == 'LOAD' else command['node_id']
        if (message.get('event') != 'HANDLING_COMPLETED' or message.get('stopped') is not True
                or message.get('supported_by') != expected
                or not isinstance(message.get('evidence_id'), str) or not message['evidence_id']):
            return reject('identified stopped support observation required')
        self.last_receipt = received_at
        self.completed = True
        self.events.append(dict(event='HANDLING_COMPLETED', receipt=dict(message), received_at=received_at))
        return dict(accepted=True, handling_id=command['handling_id'])


class ExternalExecution:
    """Validate action identity, actual arrival, freshness and cancellation."""

    def __init__(self, scene_id: str, robot_id: str, start_xy: tuple[float, float],
                 *, frame_id: str = "network", time_base: str = "ros_sim",
                 source: str = "gazebo", arrival_tolerance: float = 0.15,
                 timeout: float = 2.0, run_id: str | None = None,
                 namespace_routes: bool = False) -> None:
        if any(not math.isfinite(v) for v in start_xy):
            raise ValueError("start must be finite")
        if not all(math.isfinite(v) and v > 0 for v in (arrival_tolerance, timeout)):
            raise ValueError("tolerance and timeout must be finite and positive")
        self.identity = dict(scene_id=scene_id, run_id=run_id or uuid.uuid4().hex,
                             robot_id=robot_id, frame_id=frame_id,
                             time_base=time_base, source=source)
        self.start_xy = start_xy
        self.namespace_routes = namespace_routes
        self.tolerance = arrival_tolerance
        self.timeout = timeout
        self.bound = False
        self.state = "unbound"
        self.fault: str | None = None
        self.command: dict | None = None
        self.pose: dict | None = None
        self.last_receipt: float | None = None
        self.last_timestamp = -math.inf
        self.last_seq = -1
        self.revision = 0
        self.stopped = False
        self._outbox: list[dict] = []

    def envelope(self) -> dict:
        """Identity the execution source must echo with each observation."""
        return {**self.identity,
                "task_id": self.command["task_id"] if self.command else None,
                "route_id": self.command["route_id"] if self.command else None}

    def feedback(self, msg: dict, received_at: float) -> dict:
        """Reject old/misbound data before it can refresh the watchdog or pose."""
        def reject(reason):
            return {"accepted": False, "reason": reason}

        if not isinstance(msg, dict):
            return reject("message must be an object")
        if any(msg.get(k) != v for k, v in self.envelope().items()):
            return reject("identity mismatch")
        seq, stamp = msg.get("seq"), msg.get("timestamp")
        pose = msg.get("pose")
        if (isinstance(seq, bool) or not isinstance(seq, int) or seq <= self.last_seq):
            return reject("obsolete sequence")
        if (isinstance(stamp, bool) or not isinstance(stamp, (int, float))
                or not math.isfinite(stamp) or stamp < self.last_timestamp):
            return reject("invalid timestamp")
        if (not math.isfinite(received_at)
                or self.last_receipt is not None and received_at < self.last_receipt):
            return reject("invalid receipt time")
        if (not isinstance(pose, (list, tuple)) or len(pose) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in pose)):
            return reject("invalid pose")
        kind = msg.get("event")
        if kind not in {"POSE", "ARRIVED", "FAILED", "STOPPED"}:
            return reject("unknown event")
        if stamp == self.last_timestamp and kind == "POSE":
            return reject("duplicate source timestamp")
        if self.fault and kind not in {"POSE", "STOPPED"}:
            return reject("fault latched")
        # A delayed arrival must not outrun the receipt watchdog.
        self.poll(received_at)
        if self.fault and kind not in {"POSE", "STOPPED"}:
            return reject("fault latched")
        if not self.bound:
            if kind != "POSE" or math.dist(pose[:2], self.start_xy) > self.tolerance:
                return reject("bootstrap requires pose at declared start")
        if kind == "ARRIVED":
            if self.command is None or self.state != "moving":
                return reject("no moving action")
            if math.dist(pose[:2], self.command["target_xy"]) > self.tolerance:
                return reject("arrival outside tolerance")
        if kind == "FAILED" and self.command is None:
            return reject("no action to fail")
        if kind == "STOPPED" and not self.fault:
            return reject("no stop requested")
        # A second event may refer to the same physical observation, but it
        # cannot make that observation fresh again for the receipt watchdog.
        if stamp > self.last_timestamp:
            self.last_receipt = received_at
        self.last_seq, self.last_timestamp = seq, float(stamp)
        self.pose = dict(x=float(pose[0]), y=float(pose[1]), yaw=float(pose[2]),
                         timestamp=float(stamp))
        self.bound = True
        if self.state == "unbound":
            self.state = "idle"
        if kind == "ARRIVED":
            self.state = "arrived"
        elif kind == "FAILED":
            self.fail(str(msg.get("reason", "execution failed")))
        elif kind == "STOPPED":
            self.stopped = True
        return {"accepted": True, "state": self.state}

    def begin(self, task_id: str, src: str, dst: str,
              target_xy: tuple[float, float], speed: float, *, path_xy=None) -> dict:
        """Bind exactly one scheduler-approved graph hop to a unique action."""
        if not self.bound or self.fault or self.command:
            raise RuntimeError("execution is not ready for a new action")
        self.revision += 1
        self.command = {**self.identity, "task_id": task_id,
                        "route_id": self.recovery_token(),
                        "revision": self.revision,
                        "kind": "MOVE", "nodes": [src, dst],
                        "target_xy": list(target_xy), "speed": speed}
        if path_xy is not None:
            self.command['path_xy'] = [list(p) for p in path_xy]
        self.state = "moving"
        self._outbox.append(dict(self.command))
        return self.command

    def consume_arrival(self) -> bool:
        """Called by the same scheduler that owns this movement and its task."""
        if self.state != "arrived" or self.fault:
            return False
        self.command = None
        self.state = "idle"
        return True

    def recovery_token(self) -> str:
        """A version also exists while stationary, so recovery cannot be replayed."""
        prefix = self.identity['run_id']
        if self.namespace_routes:
            prefix += ':' + self.identity['robot_id']
        return self.command["route_id"] if self.command else f"{prefix}:{self.revision}"

    def resume(self, expected_route_id: str, now: float, *, resume_path_xy=None) -> dict:
        """Explicitly reauthorize the retained action; geometry is checked by its owner."""
        if not math.isfinite(now) or self.last_receipt is None or now < self.last_receipt:
            return {"accepted": False, "reason": "invalid recovery time"}
        self.poll(now)
        if (not self.bound or not self.fault or not self.stopped or self.pose is None
                ):
            return {"accepted": False, "reason": "fresh stopped faulted action required"}
        if self.recovery_token() != expected_route_id:
            return {"accepted": False, "reason": "recovery action mismatch"}
        if self._outbox:
            return {"accepted": False, "reason": "pending stop command"}
        old = self.command
        self.revision += 1
        if old is None:
            self.fault = None
            self.stopped = False
            self.state = "idle"
            return {"accepted": True, "recovery_token": self.recovery_token()}
        self.command = None
        self.command = {**old, "route_id": self.recovery_token(),
                        "revision": self.revision, "resumes_route_id": old["route_id"],
                        "resume_from": [self.pose["x"], self.pose["y"]]}
        if resume_path_xy is not None:
            self.command['resume_path_xy'] = [list(p) for p in resume_path_xy]
        self.fault = None
        self.stopped = False
        self.state = "moving"
        self._outbox.append(dict(self.command))
        return {"accepted": True, "command": dict(self.command)}

    def fail(self, reason: str) -> None:
        """Latch once, retaining identity for late cancellation acknowledgements."""
        if self.fault:
            return
        self.fault = reason
        self.stopped = False
        self.state = "fault"
        # A not-yet-dispatched MOVE must not survive the stop latch. Retain
        # command identity/occupancy, while the dispatch queue carries only stop.
        self._outbox = [c for c in self._outbox if c['kind'] == 'CANCEL']
        self._outbox.append({**self.envelope(), "kind": "CANCEL", "reason": reason})

    def poll(self, now: float) -> None:
        """Only accepted observations refresh this monotonic watchdog."""
        if self.last_receipt is not None and now - self.last_receipt > self.timeout:
            self.fail("feedback timeout")
            self.pose = None
            self.stopped = False

    def drain_commands(self) -> list[dict]:
        commands, self._outbox = self._outbox, []
        return commands

    def snapshot(self) -> dict:
        return {**self.envelope(), "state": self.state, "fault": self.fault,
                "stopped": self.stopped, "pose": self.pose,
                "last_seq": self.last_seq, "last_receipt": self.last_receipt}
