"""Run the existing Python scheduler with native Nav2 actions and Gazebo truth.

Single-car execution milestone. The shipped scene transform is 1:25, so this is
an integration fixture, not a real-scale collision or localization experiment.
No global routing lives here: each MOVE comes from Simulator's admission gate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import rclpy
from action_msgs.msg import GoalStatusArray
from action_msgs.srv import CancelGoal
from gazebo_msgs.srv import GetEntityState
from geometry_msgs.msg import Twist
from nav2_msgs.action import FollowPath
from nav2_msgs.msg import SpeedLimit
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from amr_gazebo_guard.msg import GuardStatus, MotionLease
from amr_gazebo_guard.srv import AuthorizeMotion

from nav2_route_follower import RouteFollower, load_network
from gz_frame import DEMO_FRAME
from terminal_docking import DockRegulator, DockSpec
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.entities import Agv, Task
from amr_sched.checkpoint import CheckpointStore, capture, definition, restore
from tools.wheelset_scene.navigation_records import BufferedTextLog
from amr_sched.external_execution import ExternalExecution
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.stream import FrameBuilder, frame_to_line
from unique_identifier_msgs.msg import UUID


def build_scheduler(start="J_S_8", pickup="J_S_25", dropoff="J_S_45", *,
                    network=None, frame=DEMO_FRAME, source="gazebo",scene_id="rail_freight_workshop_v1"):
    """Use production planning/allocation/reservations, with external advancement."""
    network = load_network() if network is None else network
    a = network.node(start)
    gate = ExternalExecution(scene_id, "AGV_01", (a.x, a.y), source=source,
                             arrival_tolerance=0.15 / frame.scale, timeout=2.0)
    sim = Simulator(network, [Task("DELIVERY_01", pickup, dropoff)],
                    [Agv("AGV_01", start_node=start)],
                    SpaceTimeAStarPlanner(dt=0.1), GreedyAllocator(),
                    SimulationConfig(dt=0.1, load_time=0.2, unload_time=0.2,
                                     max_steps=6000, idle_retreat=False,
                                     deadlock_timeout=0, reschedule_policy=ReschedulePolicy.NONE),
                    execution=gate)
    sim.reset()
    return sim


class SchedulerNode(RouteFollower):
    """One ROS executor owns callbacks, scheduler mutations and stream snapshots."""

    def __init__(self, sim, log_path, port=0, *, checkpoint_path=None, resume=False, recover_route=None,
                 coordinate_frame=DEMO_FRAME,frame_id="odom",feedback_source="gazebo",runtime_files=(),
                 ros_namespace='',tracking_offset_m=0.,dynamic_tf_depth=None,dock_takeover=None,
                 pose_measurement=None,dynamic_tf_reliability=None,async_logs=False):
        super().__init__(frame_id=frame_id,coordinate_frame=coordinate_frame,namespace=ros_namespace,
                         tracking_offset_m=tracking_offset_m)
        if sim.execution.identity["source"] != feedback_source:
            raise ValueError("execution pose authority mismatch")
        self.sim = sim
        self.builder = FrameBuilder(sim)
        self.closed = False
        self.clock_s = None
        self.epoch = None
        self.seq = -1
        self.truth_pending = None
        self.goal_handle = None
        self.guard_status = None
        self.guard_received_at = None
        self.guard_id = None
        self.permit = None
        self.guard_authorized = False
        self.guard_pending = None
        self.waiting_move = None
        self.waiting_since = None
        self.lease_seq = 0
        self.native_pending = False
        self.active = None
        self.cancel_requested = False
        self.result_status = None
        self.stop_since = None
        self.suppress_feedback = False  # harness-only fault injection, logged by caller
        self.done = False
        self.latest_truth = None
        self.latest_frame = None
        self.last_arrival = None
        self.display_clients = []
        self.listener = None
        self.store = CheckpointStore(checkpoint_path or Path(log_path).with_suffix(".checkpoint.json"))
        fingerprint = definition(sim, [Path(__file__), Path(__file__).with_name("gz_frame.py"),
                                            Path(__file__).with_name("nav2_local.yaml"),*runtime_files])
        runtime_definition=dict(coordinate_frame=asdict(coordinate_frame),frame_id=frame_id,feedback_source=feedback_source,
                                tracking_offset_m=tracking_offset_m,dynamic_tf_depth=dynamic_tf_depth,
                                terminal_docking=dock_takeover,pose_measurement=pose_measurement)
        runtime_definition['dynamic_tf_reliability']=dynamic_tf_reliability
        runtime_definition['async_logs']=async_logs
        self.fingerprint=hashlib.sha256((fingerprint+json.dumps(runtime_definition,sort_keys=True)).encode()).hexdigest()
        self.persistence_failed = False
        self.native_goal_id = None
        self.native_state = "none"
        self.native_sent_at = None
        self.velocity_received_at = None
        self.last_motion = None
        self.dock_spec = None if dock_takeover is None else DockSpec(**dock_takeover)
        self.dock = None
        self.dock_candidate = None
        self.dock_pub = None
        self.dock_pending = None
        self.reconcile_pending = None
        self.last_reconcile = -math.inf
        self.restart_clock = None
        self.restart_first_clock = False
        self.reconciling = False
        self.requested_recovery = recover_route
        try:
            if resume:
                context = restore(sim, self.store.load(), self.fingerprint)
                self.guard_id, self.permit = context["guard_id"], context["permit"]
                self.native_goal_id, self.native_state = context["native_goal_id"], context["native_state"]
                self.native_pending = self.native_state in {"prepared", "active", "unknown"}
                self.reconciling = self.native_pending
                if self.native_pending and self.native_goal_id is None:
                    raise ValueError("unknown native action requires its exact UUID")
                self.seq = context["seq"]
                self.restart_clock = context["clock_s"]
                self.restart_first_clock = True
            elif self.store.path.exists():
                raise RuntimeError("checkpoint exists; explicitly resume it or select a new path")
            # Acquire ownership and validate the snapshot before opening output.
            # A second writer or bad restart must not truncate existing evidence.
            if type(async_logs) is not bool:
                raise ValueError('explicit boolean asynchronous log mode required')
            if async_logs:
                from tools.wheelset_scene.navigation_records import ProcessTextLog
                self.log_file = ProcessTextLog(log_path)
            else:
                self.log_file = BufferedTextLog(log_path)
        except Exception:
            self.store.close()
            raise
        if port:
            self.listener = socket.socket()
            self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.listener.bind(("0.0.0.0", port))
            self.listener.listen(4)
            self.listener.setblocking(False)
        self.truth = self.create_client(GetEntityState, "/get_entity_state") if feedback_source=="gazebo" else None
        self.guard_client = self.create_client(AuthorizeMotion, "/motion_guard/authorize")
        self.native_cancel = self.create_client(CancelGoal, "/follow_path/_action/cancel_goal")
        self.lease_pub = self.create_publisher(MotionLease, "/motion_guard/lease", 1)
        self.speed_pub = self.create_publisher(SpeedLimit, "/speed_limit", 10)
        self.create_subscription(Clock, "/clock", self.on_clock, qos_profile_sensor_data)
        self.create_subscription(Twist, "/cmd_vel", self.on_velocity, 100)
        # Keep a one-argument ROS callback; received_at is an owner-only keyword.
        self.create_subscription(GuardStatus, "/motion_guard/status", lambda message: self.on_guard(message), 10)
        self.create_subscription(GoalStatusArray, "/follow_path/_action/status", self.on_native_status,
                                 QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                            reliability=ReliabilityPolicy.RELIABLE))
        self.timer = self.create_timer(0.05, self.tick)
        self.record("hello", value=self.builder.hello())
        self.persist()

    def persist(self):
        """A failed durable write latches a stop; no authorization or lease follows."""
        if self.persistence_failed:
            return False
        try:
            context = dict(guard_id=self.guard_id, permit=self.permit,
                           native_goal_id=self.native_goal_id, native_state=self.native_state,
                           clock_s=self.clock_s, seq=self.seq)
            self.store.save(capture(self.sim, self.fingerprint, context))
            return True
        except Exception as exc:
            self.persistence_failed = True
            self.sim.execution.fail("durable checkpoint failed: " + str(exc))
            self.record("checkpoint_failed", reason=str(exc))
            self.stop_guard()
            return False

    def on_native_status(self, msg):
        """Only an exact saved native UUID with terminal status proves completion."""
        if self.closed or not self.native_goal_id:
            return
        for status in msg.status_list:
            if bytes(status.goal_info.goal_id.uuid).hex() != self.native_goal_id:
                continue
            if status.status in (4, 5, 6):
                if self.reconciling and self.native_pending:
                    self.record("native_reconciled", goal_id=self.native_goal_id, status=int(status.status))
                    self.native_state = "terminal"
                    self.native_pending = False
                    self.reconciling = False
                    self.active = None
                    self.goal_handle = None
                    self.persist()
                elif self.active is not None and self.guard_authorized and self.dock is None:
                    # Nav2 publishes a terminal action status before the action
                    # client result callback is necessarily delivered. Close
                    # the native motion permit immediately so a stopped
                    # controller cannot trip the guard's .25 s command
                    # watchdog during that bounded handoff. Task completion
                    # still requires the exact result callback and a fresh
                    # stopped guard observation.
                    # A declared terminal take-over is the one case where the
                    # owner keeps commanding the wheels, so closing the permit
                    # here would freeze the vehicle short of the dock.
                    self.record("native_terminal_seen", goal_id=self.native_goal_id,
                                status=int(status.status))
                    self.stop_guard()
                return

    def reconcile_native(self):
        if not self.native_pending or not self.native_goal_id or not self.sim.execution.fault:
            return
        now=time.monotonic()
        if self.reconcile_pending is not None and not self.reconcile_pending.done():
            if now-self.last_reconcile<1.:
                return
            pending=self.reconcile_pending
            self.reconcile_pending=None
            self.native_cancel.remove_pending_request(pending)
            pending.cancel()
            self.record('native_cancel_timeout',goal_id=self.native_goal_id)
        if (not self.native_cancel.service_is_ready()
                or now-self.last_reconcile<.5):
            return
        req = CancelGoal.Request()
        req.goal_info.goal_id = UUID(uuid=list(bytes.fromhex(self.native_goal_id)))
        self.last_reconcile = now
        self.reconcile_pending = self.native_cancel.call_async(req)
        self.reconcile_pending.add_done_callback(self.on_reconcile_cancel)

    def on_reconcile_cancel(self, future):
        if self.closed or future.cancelled():
            return
        try:
            response = future.result()
            self.record("native_cancel_ack", goal_id=self.native_goal_id,
                        return_code=int(response.return_code), count=len(response.goals_canceling))
        except Exception as exc:
            self.record("native_cancel_error", reason=str(exc))

    def record(self, kind, **data):
        if self.closed:
            return
        self.log_file.write(json.dumps(dict(kind=kind, wall=time.monotonic(),
                                            sim=self.clock_s, **data)) + "\n")

    def on_clock(self, msg):
        stamp = msg.clock.sec + msg.clock.nanosec * 1e-9
        if self.restart_first_clock:
            self.restart_first_clock = False
            if self.restart_clock is not None and stamp < self.restart_clock:
                self.sim.execution.fail("restart ROS clock moved backwards")
                # A world reset invalidates the retained metric occupancy.
                self.persistence_failed = True
            self.epoch = stamp - self.sim._step_cursor * self.sim.config.dt
        if self.clock_s is not None and stamp < self.clock_s:
            self.sim.execution.fail("ROS clock moved backwards")
        self.clock_s = stamp

    def on_velocity(self, msg):
        self.velocity_received_at = time.monotonic()
        # Keep the owner's terminal take-over distinguishable from Nav2 in the
        # same evidence stream; the guard sees one topic either way.
        self.record("cmd_vel", vx=msg.linear.x, wz=msg.angular.z,
                    origin="dock_owner" if self.dock is not None else "nav2")

    def on_guard(self, msg, *, received_at=None):
        if self.closed:
            return
        if self.guard_id is None:
            self.guard_id = msg.guard_id
        elif self.guard_id != msg.guard_id:
            self.sim.execution.fail("execution guard restarted; reconciliation required")
            self.persistence_failed = True
            self.stop_guard()
            return
        self.guard_status = msg
        self.guard_received_at = time.monotonic() if received_at is None else received_at
        self.record("guard", guard_id=msg.guard_id, run_id=msg.run_id,
                    route_id=msg.route_id, revision=msg.revision, state=msg.state,
                    reason=msg.reason, stopped=msg.stopped,
                    actual_speed=msg.actual_speed, actual_wz=msg.actual_wz,
                    command_age=msg.command_age, lease_age=msg.lease_age,
                    output_vx=msg.output_vx, output_wz=msg.output_wz)
        # The service request can take longer to return than the guard's first
        # lease deadline. An exact, fresh ARMED status from this guard is the
        # independent acknowledgement that permits the first heartbeat; the
        # request's durable intent was saved before it was sent.
        if (not self.guard_authorized and self.guard_pending is not None
                and self.permit and msg.state == "armed"
                and msg.guard_id == self.permit["guard_id"]
                and msg.run_id == self.permit["run_id"]
                and msg.route_id == self.permit["route_id"]
                and msg.revision == self.permit["revision"]
                and not self.persistence_failed and not self.sim.execution.fault
                and self.sim.execution.command is not None
                and self.sim.execution.command.get("run_id") == self.permit["run_id"]
                and self.sim.execution.command.get("route_id") == self.permit["route_id"]
                and self.sim.execution.command.get("revision") == self.permit["revision"]
                and self.guard_fresh()):
            self.guard_authorized = True
            self.record("guard_authorization_observed", route_id=msg.route_id,
                        revision=msg.revision, source="fresh_exact_guard_status")
            self.publish_lease()
        if (self.guard_authorized and self.permit
                and msg.route_id == self.permit["route_id"] and msg.state != "armed"):
            self.sim.execution.fail("execution guard: " + msg.reason)

    def guard_fresh(self):
        """True only when the newest guard message is fresh on both clocks.

        The two conditions fail for different physical reasons: wall age means the
        receiver did not hand over a message in time, while the clock difference
        means the local simulation clock and the writer's stamp disagree. Record
        which one tripped so a stop can never be diagnosed by inference alone.
        """
        now = time.monotonic()
        if self.guard_status is None or self.guard_received_at is None or self.clock_s is None:
            self.record_guard_freshness_rejected('missing_sample', now, None, None, None)
            return False
        stamp = self.guard_status.header.stamp.sec + self.guard_status.header.stamp.nanosec * 1e-9
        wall_age = now - self.guard_received_at
        clock_delta = self.clock_s - stamp
        if wall_age < .25 and -.1 <= clock_delta < .25:
            return True
        clause = 'wall_age' if wall_age >= .25 else 'clock_delta'
        self.record_guard_freshness_rejected(clause, now, wall_age, clock_delta, stamp)
        return False

    def record_guard_freshness_rejected(self, clause, now, wall_age, clock_delta, stamp):
        """Rate-limit rejections to 5 Hz so the evidence stream stays bounded."""
        last = getattr(self, '_last_guard_freshness_rejected', None)
        if last is not None and now-last < .2:
            return
        self._last_guard_freshness_rejected = now
        try:
            self.record('guard_freshness_rejected', clause=clause,
                        wall_age_s=wall_age, clock_minus_stamp_s=clock_delta,
                        clock_s=self.clock_s, guard_stamp_s=stamp,
                        received_at=self.guard_received_at)
        except Exception:
            # Diagnostics must never become a new reason to stop a robot; the
            # freshness decision above has already been taken from raw values.
            pass

    def publish_lease(self):
        if (self.persistence_failed or not self.guard_authorized or not self.permit
                or self.sim.execution.fault or self.clock_s is None):
            return
        msg = MotionLease()
        msg.header.stamp = rclpy.time.Time(seconds=self.clock_s).to_msg()
        for key, value in self.permit.items():
            setattr(msg, key, value)
        self.lease_seq += 1
        msg.seq = self.lease_seq
        self.lease_pub.publish(msg)
        self.record("lease", route_id=msg.route_id, revision=msg.revision, seq=msg.seq,
                    stamp=self.clock_s, subscribers=self.lease_pub.get_subscription_count())

    def stop_guard(self):
        if not self.permit or not self.guard_authorized:
            return
        self.guard_authorized = False
        req = AuthorizeMotion.Request()
        for key, value in self.permit.items():
            setattr(req, key, value)
        req.enable = False
        # Even an unavailable stop service cannot keep motion alive: no further
        # leases are published, and the physics-process watchdog expires it.
        if self.guard_client.service_is_ready():
            future = self.guard_client.call_async(req)
            future.add_done_callback(self.on_guard_stopped)

    def on_guard_stopped(self, future):
        if self.closed:
            return
        try:
            response = future.result()
            self.record("guard_stop", accepted=response.accepted, reason=response.reason)
        except Exception as exc:
            self.record("guard_stop_error", reason=str(exc))

    def emit(self, kind, pose, envelope, stamp, reason=None):
        self.seq += 1
        msg = {**envelope, "seq": self.seq, "timestamp": stamp,
               "event": kind, "pose": pose}
        if reason:
            msg["reason"] = reason
        result = self.sim.execution_feedback(msg, time.monotonic())
        gate = self.sim.execution
        if result['accepted'] and gate.command and 'path_xy' in gate.command and not gate.fault:
            from amr_sched.geometry import project_polyline
            distance = project_polyline(pose[:2], gate.command['path_xy'])[0]
            if distance > .35:
                gate.fail('local execution left the approved hop capsule')
                self.stop_guard()
                self.record('local_corridor_lock', distance=distance, command=gate.command)
        if result["accepted"] and kind == "ARRIVED":
            self.last_arrival = dict(msg)
        self.record("feedback", message=msg, result=result)
        return result

    def query_truth(self):
        if (self.clock_s is None or self.truth is None or not self.truth.service_is_ready()
                or self.truth_pending is not None and not self.truth_pending.done()):
            return
        req = GetEntityState.Request()
        req.name, req.reference_frame = "scout_mini", "world"
        envelope = self.sim.execution.envelope()
        requested_at = time.monotonic()
        self.truth_pending = self.truth.call_async(req)
        self.truth_pending.add_done_callback(lambda f: self.on_truth(f, envelope, requested_at))

    def on_truth(self, future, envelope, requested_at):
        if self.closed:
            return
        try:
            response = future.result()
        except Exception as exc:
            self.record("truth_error", error=str(exc))
            return
        if not response or not response.success:
            return
        # GetEntityState stamps the returned observation at service processing.
        # The request's /clock value is not the time of the returned pose.
        stamp = response.header.stamp.sec + response.header.stamp.nanosec * 1e-9
        if (time.monotonic() - requested_at > .5 or self.clock_s is None
                or self.clock_s - stamp > .25 or stamp - self.clock_s > .1):
            self.record("truth_rejected", reason="stale or inconsistent source time", stamp=stamp)
            return
        p, q, twist = response.state.pose.position, response.state.pose.orientation, response.state.twist
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.latest_truth = dict(x=p.x, y=p.y, yaw=yaw, vx=twist.linear.x,
                                 vy=twist.linear.y, wz=twist.angular.z, timestamp=stamp)
        self.record("truth", value=self.latest_truth, suppressed=self.suppress_feedback)
        self.process_observation(p.x,p.y,yaw,twist.linear.x,twist.linear.y,twist.angular.z,stamp,envelope)

    def query_feedback(self):
        self.query_truth()

    def feedback_available(self):
        """Last known coordinates remain historical when source feedback expires."""
        gate=self.sim.execution
        return (not self.suppress_feedback and gate.pose is not None and gate.last_receipt is not None
                and self.clock_s is not None and 0<=time.monotonic()-gate.last_receipt<=.2
                and -.1<=self.clock_s-gate.pose['timestamp']<=.2)

    def process_observation(self, world_x, world_y, yaw, vx, vy, wz, stamp, envelope):
        x,y=self.coordinate_frame.to_network(world_x,world_y)
        pose=[x,y,self.coordinate_frame.to_network_yaw(yaw)]
        # The terminal regulator needs the measured body motion of the same
        # sample it regulates; a commanded twist is not evidence of movement.
        self.last_motion = dict(speed=math.hypot(vx, vy), wz=wz, timestamp=stamp)
        gate = self.sim.execution
        if self.suppress_feedback:
            return
        if gate.fault:
            # Fresh truth can reappear while the operational fault stays latched.
            answer = self.emit("POSE", pose, envelope, stamp)
            if not answer["accepted"]:
                # Sensor TF may update slower than this timer. A duplicate is
                # not new evidence, but must not erase an ongoing stop proof.
                # No stop acknowledgement is evaluated until a fresh sample.
                if answer.get('reason') != 'duplicate source timestamp':
                    self.stop_since = None
                return
            speed = math.hypot(vx,vy)
            if (not self.native_pending and self.goal_handle is None and self.guard_fresh()
                    and self.guard_status.state == "latched" and self.guard_status.stopped
                    and speed < .01 and abs(wz) < .02):
                if self.stop_since is None:
                    self.stop_since = (time.monotonic(), stamp)
                if (not gate.stopped and time.monotonic() - self.stop_since[0] >= .3
                        and stamp - self.stop_since[1] >= .3):
                    self.emit("STOPPED", pose, envelope, stamp)
            else:
                self.stop_since = None
            return
        kind = "POSE"
        if self.result_status is not None and envelope.get("route_id") is not None:
            if self.result_status != 4:
                kind = "FAILED"
            elif self.guard_fresh() and self.guard_status.state == "latched" and self.guard_status.stopped:
                kind = "ARRIVED"
        answer = self.emit(kind, pose, envelope, stamp,
                           reason=f"Nav2 action status {self.result_status}" if kind == "FAILED" else None)
        if answer["accepted"] and kind in ("ARRIVED", "FAILED"):
            self.result_status = None

    def set_speed(self, command):
        msg = SpeedLimit()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.percentage = False
        msg.speed_limit = float(command["speed"] * self.coordinate_frame.scale)
        self.speed_pub.publish(msg)

    def dispatch(self, command):
        self.record("command", value=command)
        if command["kind"] == "CANCEL":
            self.cancel_requested = True
            self.waiting_move = None
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
            self.stop_guard()
            return
        if self.persistence_failed or self.sim.execution.fault:
            return
        if self.active is not None or self.native_pending or self.goal_handle is not None:
            self.sim.execution.fail("Nav2 action busy")
            return
        # Only source-node-to-actual connector is admitted, within arrival tolerance.
        actual = self.sim.execution.pose
        src = self.sim.network.node(command["nodes"][0])
        admitted_start = command.get("resume_from", (src.x, src.y))
        if actual is None or math.dist((actual["x"], actual["y"]), admitted_start) > self.sim.execution.tolerance:
            self.sim.execution.fail("actual start outside admitted node tolerance")
            return
        self.active = command
        self.waiting_move = command
        self.waiting_since = time.monotonic()
        self.cancel_requested = False

    def resume_execution(self, expected_route_id):
        """Explicit operator/API recovery after native termination and fresh stop."""
        if (self.persistence_failed or self.native_pending or self.goal_handle is not None or self.guard_pending is not None
                or not self.guard_fresh() or self.guard_status.state != "latched"
                or not self.guard_status.stopped):
            answer = {"accepted": False, "reason": "native termination and fresh guard stop required"}
        else:
            answer = self.sim.resume_execution(expected_route_id, time.monotonic())
        if answer["accepted"]:
            self.active = self.waiting_move = None
            self.result_status = None
            self.cancel_requested = False
            self.stop_since = None
            if not self.persist():
                answer = {"accepted": False, "reason": "durable recovery write failed"}
        self.record("recovery", previous_route_id=expected_route_id, result=answer)
        return answer

    def replan_execution(self, expected_route_id):
        """Apply the owner's reserved node detour after native/guard reconciliation."""
        if (self.persistence_failed or self.native_pending or self.goal_handle is not None or self.guard_pending is not None
                or not self.guard_fresh() or self.guard_status.state != 'latched'
                or not self.guard_status.stopped):
            answer = {'accepted': False, 'reason': 'native termination and fresh guard stop required'}
        else:
            answer = self.sim.replan_execution(expected_route_id, time.monotonic())
        if answer['accepted']:
            self.active = self.waiting_move = None
            self.result_status = None
            self.cancel_requested = False
            self.stop_since = None
            if not self.persist():
                answer = {'accepted': False, 'reason': 'durable detour write failed'}
        self.record('physical_replan', previous_route_id=expected_route_id, result=answer)
        return answer

    def preempt_execution(self, expected_route_id):
        """Commit an unloaded task switch only after native termination and stop."""
        if (self.persistence_failed or self.native_pending or self.goal_handle is not None or self.guard_pending is not None
                or not self.guard_fresh() or self.guard_status.state != 'latched'
                or not self.guard_status.stopped):
            answer = {'accepted': False, 'reason': 'native termination and fresh guard stop required'}
        else:
            answer = self.sim.preempt_execution(expected_route_id, time.monotonic())
        if answer['accepted']:
            self.active = self.waiting_move = None
            self.result_status = None
            self.cancel_requested = False
            self.stop_since = None
            if not self.persist():
                answer = {'accepted': False, 'reason': 'durable urgent task write failed'}
        self.record('physical_preemption', previous_route_id=expected_route_id, result=answer)
        return answer

    def authorize_waiting_move(self):
        command = self.waiting_move
        if command is None or self.persistence_failed or self.sim.execution.fault or self.guard_pending is not None:
            return
        if time.monotonic() - self.waiting_since > 5:
            self.sim.execution.fail("execution guard not ready for a fresh action")
            return
        if (not self.guard_fresh() or not self.guard_client.service_is_ready()
                or self.guard_status.state != "latched" or not self.guard_status.stopped
                or not self._client.server_is_ready() or self.speed_pub.get_subscription_count() < 1
                or self.lease_pub.get_subscription_count() < 1):
            return
        # Start the new controller while the physics guard remains locked.
        # Idle velocity-smoother zeros must not start the guard's .25 s command
        # timeout before a delayed action acceptance/authorization response.
        if self.native_pending:
            return
        if self.goal_handle is None:
            self.start_native(command)
            return
        if (self.velocity_received_at is None or self.native_sent_at is None
                or self.velocity_received_at <= self.native_sent_at
                or time.monotonic() - self.velocity_received_at > .15):
            return
        self.permit = dict(guard_id=self.guard_id, run_id=command["run_id"],
                           route_id=command["route_id"], revision=command["revision"])
        self.lease_seq = 0
        req = AuthorizeMotion.Request()
        for key, value in self.permit.items():
            setattr(req, key, value)
        req.enable = True
        if not self.persist():
            return
        self.guard_pending = self.guard_client.call_async(req)
        self.record("guard_authorize_requested", route_id=command["route_id"],
                    revision=command["revision"])
        self.guard_pending.add_done_callback(lambda f: self.on_authorized(f, command))

    def on_authorized(self, future, command):
        if self.closed:
            return
        self.guard_pending = None
        try:
            response = future.result()
        except Exception as exc:
            self.sim.execution.fail("guard authorization failed: " + str(exc))
            return
        self.record("guard_authorize", route_id=command["route_id"],
                    accepted=response.accepted, reason=response.reason)
        if not response.accepted:
            self.sim.execution.fail("guard rejected action: " + response.reason)
            return
        self.guard_authorized = True
        self.waiting_move = None
        if (self.persistence_failed or self.sim.execution.fault or self.sim.execution.command != command
                or self.goal_handle is None or self.active is None):
            self.stop_guard()
            return
        self.publish_lease()

    def terminal_goal_checker(self, command):
        """Station terminals need the handoff tolerance; junction hops do not.

        The transit checker admits a stop anywhere inside its window, and the
        contact capture gate is measured on the true carrier pose, so a route
        that ends at a station must ask the controller for the docking checker.
        A route that ends at an ordinary junction keeps its original tolerance.
        The lookup is deliberately defensive: an unknown or missing node keeps
        the transit checker rather than inventing a tighter requirement for
        geometry nobody declared.
        """
        nodes = command.get("nodes") or []
        network = getattr(self.sim, "network", None)
        node = None
        if nodes and hasattr(network, "node"):
            try:
                node = network.node(nodes[-1])
            except (KeyError, TypeError, AttributeError):
                node = None
        if node is None or getattr(node, "kind", None) == "aisle":
            return "general_goal_checker"
        return "dock_goal_checker"

    def start_native(self, command):
        """Prepare durably and start FollowPath, with the wheel guard still locked."""
        goal_checker = self.terminal_goal_checker(command)
        self.dock_candidate = None
        actual = self.sim.execution.pose
        if actual is None:
            self.sim.execution.fail("actual pose lost before native action")
            self.stop_guard()
            return
        self.cancel_requested = False
        self.native_pending = True
        self.set_speed(command)
        goal = FollowPath.Goal()
        # The owner supplies every scene corner. Recovery supplies only the
        # approved remainder, starting at the measured stop, never a chord.
        if 'path_xy' in command:
            geometry = command.get('resume_path_xy', command['path_xy'])
            goal.path = self.build_path(command['nodes'], self.sim.network, path_xy=geometry)
        else:
            # Legacy commands have no explicit geometry and only straight hops.
            path_nodes = command['nodes'][1:] if command.get('resumes_route_id') else command['nodes']
            goal.path = self.build_path(path_nodes, self.sim.network,
                prepend_pose=(self.coordinate_frame.to_world(actual['x'],actual['y'])
                              if command.get('resumes_route_id') else None))
        # A static map route denotes fixed geometry, not a historical robot
        # observation. AMCL can correct map->odom during this action. Humble's
        # goal checker transforms the stamped endpoint, so a dispatch-time
        # stamp would freeze the old correction and admit an early arrival.
        # Zero requests the latest transform; sensor observation stamps and
        # action identity remain independently checked as before.
        transform_time = 0. if self.sim.execution.identity['source'] == 'localized' else self.clock_s
        goal.path.header.stamp = rclpy.time.Time(seconds=transform_time).to_msg()
        for pose in goal.path.poses:
            pose.header.stamp = goal.path.header.stamp
        goal.controller_id, goal.goal_checker_id = "FollowPath", goal_checker
        if goal_checker == "dock_goal_checker" and self.dock_spec is not None:
            self.dock_candidate = self.dock_target(command, goal.path)
        self.record("path", route_id=command["route_id"],
                    frame=goal.path.header.frame_id, transform_time=transform_time,
                    goal_checker=goal_checker,
                    xy=[[p.pose.position.x, p.pose.position.y] for p in goal.path.poses])
        self.native_goal_id = uuid.uuid4().hex
        self.native_state = "prepared"
        if not self.persist():
            return
        self.native_sent_at = time.monotonic()
        future = self._client.send_goal_async(goal, goal_uuid=UUID(uuid=list(bytes.fromhex(self.native_goal_id))))
        future.add_done_callback(lambda f: self.on_accepted(f, command))

    def dock_target(self, command, path):
        """The station pose this hop was actually asked for, in network coordinates.

        The position comes from the same ``target_xy`` the arrival gate checks,
        so the regulator cannot close a tolerance nobody declared. The heading
        is the orientation of the final path segment Nav2 was given, converted
        back out of the goal frame. A terminal whose sent endpoint disagrees
        with the retained target is refused rather than guessed: the two would
        otherwise steer the vehicle to different places.
        """
        poses = list(path.poses)
        if len(poses) < 2 or not command.get("target_xy"):
            return None
        first, last = poses[-2].pose.position, poses[-1].pose.position
        heading = math.atan2(last.y - first.y, last.x - first.x)
        target_xy = [float(command["target_xy"][0]), float(command["target_xy"][1])]
        endpoint = self.coordinate_frame.to_network(last.x, last.y)
        if not all(math.isfinite(v) for v in (heading, endpoint[0], endpoint[1])):
            return None
        if math.dist(endpoint, target_xy) > 1e-6 + 1e-9 * max(abs(v) for v in target_xy):
            self.record("dock_target_mismatch", route_id=command["route_id"],
                        endpoint=list(endpoint), target_xy=target_xy)
            return None
        target_yaw=self.coordinate_frame.to_network_yaw(heading)
        if self.dock_spec is not None and self.dock_spec.steering_mode=="staged":
            stations=[s for s in self.dock_spec.approach_stations
                      if math.dist(s["target_xy"],target_xy)<1e-6]
            if len(stations)!=1:
                self.record("dock_approach_undefined",route_id=command["route_id"],target_xy=target_xy)
                return None
            station=stations[0]
            return dict(route_id=command["route_id"],target_xy=target_xy,
                        target_yaw=station["yaw_rad"],approach_side=station["approach_side"])
        return dict(route_id=command["route_id"],target_xy=target_xy,target_yaw=target_yaw)

    def dock_step(self, now):
        """Close a station hop the transit controller left orbiting its endpoint.

        Only a terminal inside the declared capture window is taken over, and
        only while the owner holds the retained action and a live guard permit.
        A refusal, a fault or a deadline ends the take-over; the arrival is
        still emitted through the unchanged event path once the same action
        reports a fresh stopped guard.
        """
        if self.dock_spec is None:
            return
        gate = self.sim.execution
        if self.dock is None and self.dock_pending is not None:
            # The take-over handed the terminal back but the guard has not yet
            # proved the standstill the arrival event needs. An unbounded wait
            # would look like progress, so it is a bounded, declared deadline.
            if self.result_status is None:
                self.dock_pending = None
            elif now - self.dock_pending["since"] > self.dock_spec.arrival_deadline_s:
                self.record("dock_arrival_timeout", route_id=self.dock_pending["route_id"],
                            elapsed=now - self.dock_pending["since"])
                self.dock_pending = None
                self.sim.execution.fail("terminal docking stop proof did not follow")
                self.stop_guard()
            return
        if self.dock is None:
            if (self.dock_pending is not None or self.dock_candidate is None or self.active is None
                    or self.goal_handle is None
                    or self.dock_candidate["route_id"] != self.active["route_id"]
                    or gate.fault or self.persistence_failed or not self.guard_authorized):
                return
            pose = gate.pose
            if pose is None:
                return
            distance = math.dist((pose["x"], pose["y"]), self.dock_candidate["target_xy"])
            if distance > self.dock_spec.capture_radius_m:
                return
            if self.dock_pub is None:
                self.dock_pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.dock = dict(route_id=self.active["route_id"], started=now, handover=True,
                             regulator=DockRegulator(self.dock_spec,
                                                     self.dock_candidate["target_xy"],
                                                     self.dock_candidate["target_yaw"],
                                                     approach_side=self.dock_candidate.get("approach_side",-1)))
            self.record("dock_takeover", route_id=self.dock["route_id"],
                        target_xy=list(self.dock_candidate["target_xy"]),
                        target_yaw=self.dock_candidate["target_yaw"], distance=distance,
                        capture_radius_m=self.dock_spec.capture_radius_m)
            self.goal_handle.cancel_goal_async()
            return
        if gate.fault or self.persistence_failed or not self.guard_authorized:
            self.dock = None
            self.dock_candidate = None
            return
        if now - self.dock["started"] > self.dock_spec.deadline_s:
            self.record("dock_timeout", route_id=self.dock["route_id"],
                        elapsed=now - self.dock["started"])
            self.dock = None
            self.dock_candidate = None
            self.sim.execution.fail("terminal docking did not converge")
            self.stop_guard()
            return
        if self.dock["handover"]:
            # Nav2 may still publish for a tick after the cancel request. Hold
            # the wheels at zero until it confirms the action is gone; the
            # guard's bounded park allowance covers this window.
            if self.goal_handle is not None:
                self.dock_pub.publish(Twist())
                self.record("dock_cmd", route_id=self.dock["route_id"], phase="handover",
                            vx=0., wz=0.)
                return
            self.dock["handover"] = False
            self.record("dock_handover", route_id=self.dock["route_id"])
        pose = gate.pose
        if pose is None:
            return
        if self.last_motion is None:
            # No measured body motion yet: hold and wait. Regulating against an
            # invented speed would be exactly the fabrication this project
            # forbids, and a missing sample is not evidence of standing still.
            self.dock_pub.publish(Twist())
            self.record("dock_cmd", route_id=self.dock["route_id"], phase="waiting_motion",
                        vx=0., wz=0.)
            return
        motion = self.last_motion
        measured_at=self.docking_measurement_time(now)
        if measured_at is None:
            self.dock_pub.publish(Twist())
            self.dock["regulator"].settled_s=0.
            self.dock["regulator"].last_now=None
            self.record("dock_cmd",route_id=self.dock["route_id"],phase="waiting_station_measurement",
                        vx=0.,wz=0.)
            return
        command = self.dock["regulator"].update((pose["x"], pose["y"], pose["yaw"]),
                                                motion["speed"], motion["wz"], measured_at)
        twist = Twist()
        twist.linear.x = float(command["vx"])
        twist.angular.z = float(command["wz"])
        self.dock_pub.publish(twist)
        self.record("dock_cmd", route_id=self.dock["route_id"], phase=command["phase"],
                    vx=twist.linear.x, wz=twist.angular.z, distance=command["distance"],
                    heading_error=command["heading_error"], yaw_error=command["yaw_error"],
                    approach_side=command.get("approach_side"),
                    staging_clearance_m=command.get("staging_clearance_m"))
        if command["done"]:
            self.record("dock_aligned", route_id=self.dock["route_id"],
                        distance=command["distance"], yaw_error=command["yaw_error"],
                        hold_s=self.dock_spec.hold_s)
            self.finish_dock(now)

    def docking_measurement_time(self,now):
        """Legacy truth/transit adapter retains its existing measured control."""
        return now

    def finish_dock(self, now):
        """Hand the terminal back to the unchanged arrival and receipt path."""
        self.dock = None
        self.dock_candidate = None
        self.goal_handle = None
        self.active = None
        self.native_state = "terminal"
        self.native_pending = False
        # The same explicit stop the transit completion used: the guard latches,
        # proves the standstill, and only then does the arrival event exist.
        self.stop_guard()
        self.result_status = 4
        self.dock_pending = dict(route_id=self.permit["route_id"] if self.permit else None,
                                 since=now)
        self.persist()

    def on_accepted(self, future, command):
        if self.closed:
            return
        self.native_pending = False
        try:
            handle = future.result()
        except Exception as exc:
            # Delivery/acceptance is unknown; do not claim native termination.
            self.native_pending = True
            self.native_state = "unknown"
            self.reconciling = True
            self.sim.execution.fail("Nav2 send failed: " + str(exc))
            return
        self.record("native_accept", route_id=command["route_id"], accepted=bool(handle and handle.accepted))
        if handle is None or not handle.accepted:
            self.native_state = "rejected"
            self.active = None
            self.stop_guard()
            self.sim.execution.fail("Nav2 rejected action")
            return
        self.goal_handle = handle
        self.native_state = "active"
        self.persist()
        if self.cancel_requested or self.sim.execution.fault:
            handle.cancel_goal_async()
        handle.get_result_async().add_done_callback(lambda f: self.on_result(f, command))

    def on_result(self, future, command):
        if self.closed:
            return
        try:
            status = int(future.result().status)
        except Exception as exc:
            self.sim.execution.fail("Nav2 result failed: " + str(exc))
            return
        self.record("native_result", route_id=command["route_id"], status=status)
        if self.dock is not None and self.dock["route_id"] == command["route_id"]:
            # The owner cancelled this action on purpose: the transit
            # controller handed the terminal over inside the declared capture
            # window. Nav2's cancellation is not an execution failure, and the
            # retained action, permit and live guard must survive it.
            self.record("dock_native_cancel", route_id=command["route_id"], status=status)
            self.goal_handle = None
            self.native_state = "handover"
            self.native_pending = False
            self.persist()
            return
        if self.active is not None and command["route_id"] == self.active["route_id"]:
            self.stop_guard()
            self.goal_handle = None
            self.active = None
            self.result_status = status
            self.native_state = "terminal"
            self.native_pending = False
            self.persist()

    def broadcast(self, frame):
        if self.listener:
            try:
                conn, _ = self.listener.accept()
                conn.settimeout(.02)
                conn.sendall(frame_to_line(self.builder.hello()))
                self.display_clients.append(conn)
            except BlockingIOError:
                pass
        for conn in self.display_clients[:]:
            try:
                conn.sendall(frame_to_line(frame))
            except OSError:
                conn.close()
                self.display_clients.remove(conn)

    def tick(self):
        if self.closed:
            return
        self.sim.poll_execution(time.monotonic())
        if self.sim.execution.fault:
            self.stop_guard()
        if self.guard_authorized and not self.guard_fresh():
            self.sim.execution.fail("execution guard feedback timeout")
            self.stop_guard()
        self.reconcile_native()
        if self.active is not None and not self.sim.execution.fault:
            self.set_speed(self.active)
        self.query_feedback()
        self.dock_step(time.monotonic())
        if self.requested_recovery and self.sim.execution.stopped:
            answer = self.resume_execution(self.requested_recovery)
            if answer["accepted"]:
                self.requested_recovery = None
        if self.sim.execution.bound and self.clock_s is not None:
            if self.epoch is None:
                self.epoch = self.clock_s
            due = int((self.clock_s - self.epoch) / self.sim.config.dt)
            for _ in range(min(5, max(0, due - self.sim._step_cursor + 1))):
                if not self.sim.step_once():
                    self.done = True
                    if not self.sim._all_done():
                        self.sim.execution.fail("scheduler horizon exhausted")
                    break
        commands = self.sim.execution.drain_commands()
        # Persist scheduler mutations and the retained MOVE before dispatching.
        durable = self.persist()
        for command in commands:
            if not durable and command["kind"] != "CANCEL":
                continue
            self.dispatch(command)
        self.authorize_waiting_move()
        self.publish_lease()
        self.latest_frame = self.builder.frame()
        available=self.feedback_available()
        for agv in self.latest_frame['agvs']:
            agv['pose']['available']=available
        self.record("frame", value=self.latest_frame)
        self.broadcast(self.latest_frame)

    def clean_idle_shutdown_ready(self):
        """A completed idle task can close on fresh independent physical stop proof.

        execution.stopped is a fault-recovery acknowledgement, not an idle flag.
        Never fabricate that acknowledgement or drop retained node occupancy.
        """
        gate=self.sim.execution
        now=time.monotonic()
        if (gate.fault or not gate.bound or gate.state!='idle' or gate.command is not None
                or not self.sim._all_done() or self.persistence_failed
                or self.native_pending or self.goal_handle is not None or self.active is not None
                or self.waiting_move is not None or self.guard_pending is not None
                or self.guard_authorized or not self.guard_fresh()
                or gate.pose is None or gate.last_receipt is None
                or not 0<=now-gate.last_receipt<.25
                or not -.1<=self.clock_s-gate.pose['timestamp']<.25):
            return False
        status=self.guard_status
        if self.permit and any(getattr(status,key)!=value for key,value in self.permit.items()):
            return False
        return (status.state=='latched' and status.stopped and status.actual_speed<.01
                and abs(status.actual_wz)<.02 and status.output_vx==0 and status.output_wz==0)

    def close(self):
        clean_idle=self.clean_idle_shutdown_ready()
        if self.sim.execution.bound and not self.sim.execution.stopped and not clean_idle:
            self.suppress_feedback = False
            self.sim.execution.fail("runtime shutdown")
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline and not self.sim.execution.stopped:
                rclpy.spin_once(self, timeout_sec=.05)
            self.record("shutdown", stopped=self.sim.execution.stopped,
                        execution=self.sim.execution.snapshot(), pins=self.sim._res.pins())
        elif clean_idle:
            self.persist()
            self.record('shutdown',stopped=True,proof='fresh matching native guard; completed idle task',
                        execution=self.sim.execution.snapshot(),pins=self.sim._res.pins())
        if getattr(self,'sensor_ingress',None) is not None:
            self.sensor_ingress.close()
        self.timer.cancel()
        self.closed = True
        for conn in self.display_clients:
            conn.close()
        if self.listener:
            self.listener.close()
        self.log_file.close()
        self.store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="J_S_8")
    parser.add_argument("--pickup", default="J_S_25")
    parser.add_argument("--dropoff", default="J_S_45")
    parser.add_argument("--log", required=True)
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recover-route", help="Explicitly reauthorize this exact retained action after fresh stop")
    args = parser.parse_args()
    rclpy.init()
    node = SchedulerNode(build_scheduler(args.start, args.pickup, args.dropoff), args.log, args.port,
                         checkpoint_path=args.checkpoint, resume=args.resume, recover_route=args.recover_route)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
