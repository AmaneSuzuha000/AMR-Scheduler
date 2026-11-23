"""Run stationary readiness, short bay hops, or the full sensor-authoritative fleet.

Readiness sends no motion authorization. Moving modes reuse the existing
durable AMR fleet/Nav2 adapter. Known spawn priors initialize AMCL; separate
Gazebo state reads only measure error, never supply execution feedback.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

from .build import ROOT
from .navigation_parameters import expected_parameters, parameter_mismatches
from .navigation_records import plain_ros_value, EventWriter


class SpawnPriorHandshake:
    """Confirm the declared stationary spawn prior before motion can be enabled.

    An initialpose publish is a send attempt. A later, fresh AMCL estimate near
    the declared prior acknowledges it; independent readiness error checks
    remain responsible for measuring actual localization accuracy.
    """

    def __init__(self, x, y, yaw, *, retry_s=5., max_attempts=3):
        if (not all(math.isfinite(v) for v in (x, y, yaw, retry_s))
                or retry_s <= 0 or type(max_attempts) is not int or max_attempts < 1):
            raise ValueError('finite spawn prior and bounded retry required')
        self.prior = (x, y, yaw)
        self.retry_s = retry_s
        self.max_attempts = max_attempts
        self.attempts = 0
        self.sent_wall = self.sent_source = None
        self.confirmed = False

    def update(self, now, source_s, *, ready, motion_locked,
               estimate=None, estimate_source_s=None, estimate_received=None, estimate_valid=False):
        if self.confirmed:
            return 'confirmed'
        if not all(math.isfinite(v) for v in (now, source_s)):
            return 'wait'
        if self.sent_wall is not None and estimate_valid and estimate is not None:
            values = (*estimate, estimate_source_s, estimate_received)
            if (len(estimate) == 3 and all(type(v) in (int, float) and math.isfinite(v) for v in values)
                    and estimate_received >= self.sent_wall
                    and estimate_source_s >= self.sent_source
                    and 0 <= now-estimate_received <= 2.
                    and -.1 <= source_s-estimate_source_s <= 2.):
                x, y, yaw = self.prior
                angle = math.atan2(math.sin(estimate[2]-yaw), math.cos(estimate[2]-yaw))
                if math.hypot(estimate[0]-x, estimate[1]-y) <= .5 and abs(angle) <= .3:
                    self.confirmed = True
                    return 'acknowledged'
        if not ready or not motion_locked or source_s <= 0:
            return 'wait'
        if self.sent_wall is not None and now-self.sent_wall < self.retry_s:
            return 'wait'
        if self.attempts == self.max_attempts:
            raise RuntimeError('stationary spawn prior not acknowledged after bounded retries')
        self.sent_wall = now
        self.sent_source = source_s
        self.attempts += 1
        return 'send'


class ReadinessWindow:
    """Track a common readiness interval using the actual poll timestamps.

    The old loop used its timestamp from before callbacks and sensor polling.
    A slow input drain could therefore label a two-second interval as healthy
    even though the last readiness observation was already outside its end.
    This small state machine has no ROS or scheduling authority; it only
    returns an interval after every observer reported ready on the same loop
    and the adapter health check was evaluated for that loop.
    """

    def __init__(self, duration_s=2.0):
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError('readiness duration must be positive and finite')
        self.duration_s = duration_s
        self.started_at = None
        self.previous = None
        self.checks = 0

    def reset(self):
        self.started_at = self.previous = None
        self.checks = 0

    def update(self, states, evaluated_at, adapter_ready, *, checked_at=None):
        if len(states) != len(evaluated_at) or not evaluated_at:
            raise ValueError('one readiness timestamp is required per observer')
        if not all(math.isfinite(value) for value in evaluated_at):
            raise ValueError('readiness timestamps must be finite')
        first, last = min(evaluated_at), max(evaluated_at)
        if checked_at is None:
            checked_at = last
        if not math.isfinite(checked_at) or checked_at < last:
            raise ValueError('adapter check must follow observer evaluations')
        if not (all(states) and adapter_ready) or checked_at-first >= .05:
            self.reset()
            return None
        if self.previous is not None and (len(self.previous) != len(evaluated_at)
                or any(not 0 < new-old <= .2 for old, new in zip(self.previous, evaluated_at))):
            self.reset()
        if self.started_at is None:
            self.started_at = first
        self.previous = tuple(evaluated_at)
        self.checks += 1
        # Both endpoint rows are inside [first initial poll, last final poll].
        # Log every ready evaluation below so the independent auditor sees
        # exactly these endpoints and every intervening failure/gap.
        if last - self.started_at < self.duration_s or self.checks < 10:
            return None
        interval = (self.started_at, last)
        self.reset()
        return interval
from tools.live_bridge.sensor_health import SensorHealth, pose_quality


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def process_group_members(group):
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            values = (entry/'stat').read_text().split(') ', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(values[2]) == group and values[0] != 'Z':
            result.append(dict(pid=int(entry.name), start_ticks=int(values[19]), state=values[0]))
    return result


def stop_launch_group(process):
    """Close the session created here even when its launch leader already died."""
    group = process.pid; before = process_group_members(group); signals = []
    for sig, timeout in ((signal.SIGINT, 15.), (signal.SIGTERM, 5.), (signal.SIGKILL, 5.)):
        if not process_group_members(group): break
        try: os.killpg(group, sig)
        except ProcessLookupError: break
        signals.append(sig.name); deadline = time.monotonic()+timeout
        while process_group_members(group) and time.monotonic() < deadline:
            process.poll(); time.sleep(.05)
    process.wait(timeout=2.)
    after = process_group_members(group)
    return dict(passed=not after, group=group, before=before, signals=signals, after=after)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--guard-build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=11368)
    p.add_argument('--domain', type=int, default=82)
    p.add_argument('--mode', choices=['readiness', 'short', 'delivery', 'faults'], default='readiness')
    p.add_argument('--whole-line', action='store_true', help='Run the active 18-stage workers with localized dock receiving')
    p.add_argument('--contact-delivery', action='store_true',
                   help='Require actual SCOUT bin contacts and the same fleet-owned inventory')
    p.add_argument('--input-thread', action='store_true',
                   help='Receive localization/guard/equipment independently; decisions remain on fleet owner')
    p.add_argument('--latest-tf', action='store_true',
                   help='Use the existing depth-one dynamic TF receiver; static TF remains durable')
    p.add_argument('--best-effort-tf',action='store_true',
                   help='Use the independently traced best-effort depth100 dynamic reader; keep static TF durable')
    p.add_argument('--udp-only',action='store_true',
                   help='Use the frozen Fast DDS UDPv4 profile for this run and its children only')
    p.add_argument('--async-logs',action='store_true',
                   help='Write fleet/robot text logs in bounded FIFO peers; keep checkpoints synchronous')
    p.add_argument('--terminal-docking', action='store_true',
                   help='Let the owner close the declared station tolerance after the transit controller hands the terminal over')
    p.add_argument('--wall-timeout-s', type=float)
    p.add_argument('--startup-timeout-s', type=float, default=240.)
    p.add_argument('--contact-drain-ms', type=float, default=20.,
                   help='Wall budget for one owner equipment drain turn (max 20 ms)')
    p.add_argument('--contact-drain-rows', type=int, default=32,
                   help='Row cap for one owner equipment drain turn (1..32)')
    p.add_argument('--callback-batch', type=int, choices=(1, 16), default=1,
                   help='Bounded callback drain before equipment polling; 16 is an explicit diagnostic candidate')
    args = p.parse_args()
    if args.latest_tf and args.best_effort_tf:
        p.error('choose one explicit dynamic TF receiver')
    if args.wall_timeout_s is None:
        args.wall_timeout_s = 7500. if args.whole_line else {'readiness': 240., 'short': 1200., 'delivery': 7200., 'faults': 600.}[args.mode]
    if not 0 < args.contact_drain_ms <= 20.:
        raise SystemExit('--contact-drain-ms must be in (0, 20]')
    if not 1 <= args.contact_drain_rows <= 32:
        raise SystemExit('--contact-drain-rows must be in [1, 32]')
    if os.name == 'nt':
        raise SystemExit('Run on the Ubuntu ROS2/Gazebo host.')
    transport_profile=None
    if args.udp_only:
        transport_profile=ROOT/'tools/ros2/fastdds_udp_only.xml'
        if not transport_profile.is_file():
            raise RuntimeError('declared UDP transport profile missing')
        os.environ['RMW_IMPLEMENTATION']='rmw_fastrtps_cpp'
        os.environ['FASTRTPS_DEFAULT_PROFILES_FILE']=str(transport_profile)
    if any(not math.isfinite(value) or value <= 0
           for value in (args.wall_timeout_s, args.startup_timeout_s)):
        raise SystemExit('positive finite timeout required')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', args.port)) == 0:
            raise SystemExit('Requested Gazebo port occupied; preserved.')
    bundle = args.bundle.resolve(); guard_build = args.guard_build.resolve()
    manifest = json.loads((bundle/'manifest.json').read_text())
    if (bool(manifest.get('contact_bins')) != args.contact_delivery
            or args.contact_delivery and args.mode != 'delivery'):
        raise SystemExit('Contact geometry requires explicit contact-delivery and delivery mode; legacy receiving cannot bypass it.')
    if args.contact_delivery and manifest['contact_bins']['geometry'].get('platform_lift_m',0.)!=.02:
        raise SystemExit('Legacy contact targets overlap the chassis; rebuild with the declared 20mm raised platform.')
    if bool(manifest.get('whole_line')) != args.whole_line or args.whole_line and args.mode not in ('delivery', 'faults'):
        raise SystemExit('Whole-line execution requires an explicit matching active-world bundle and delivery/faults mode.')
    if bool(manifest.get('model_parameters', {}).get('guard_command_relay')) != (args.mode == 'faults'):
        raise SystemExit('Fault mode requires its explicit command relay bundle; normal modes require a direct bundle.')
    build = json.loads((guard_build/'build_manifest.json').read_text())
    if not build['passed']:
        raise RuntimeError('guard build not validated')
    for name, digest in build['library_sha256'].items():
        if hashlib.sha256((Path(build['install_prefix'])/'lib'/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('guard library differs from build evidence')
    for name, digest in build['source_sha256'].items():
        if name.startswith('launch/'):
            installed = Path(build['install_prefix'])/'share/amr_gazebo_guard'/name
            if not installed.is_file() or hashlib.sha256(installed.read_bytes()).hexdigest() != digest:
                raise RuntimeError('installed launch differs from guard source evidence: '+name)
    from ament_index_python.packages import get_package_prefix
    if Path(get_package_prefix('amr_gazebo_guard')).resolve() != Path(build['install_prefix']).resolve():
        raise RuntimeError('Source this exact owned guard overlay before running.')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(bundle, out/'scene_snapshot')
    native_snapshot = out/'native_guard'; native_snapshot.mkdir()
    shutil.copy2(guard_build/'build_manifest.json', native_snapshot/'build_manifest.json')
    for name in build['library_sha256']:
        shutil.copy2(Path(build['install_prefix'])/'lib'/name, native_snapshot/name)
    for name, digest in build['source_sha256'].items():
        source = guard_build/'src/amr_gazebo_guard'/name
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError('native build source snapshot changed: '+name)
        target = native_snapshot/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    os.environ['ROS_DOMAIN_ID'] = str(args.domain)
    os.environ['GAZEBO_MASTER_URI'] = f'http://127.0.0.1:{args.port}'
    os.environ['GAZEBO_MODEL_DATABASE_URI'] = ''
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ.setdefault('DISPLAY', ':0')
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu, LaserScan, JointState
    from rosgraph_msgs.msg import Clock
    from gazebo_msgs.srv import GetEntityState
    from amr_gazebo_guard.msg import GuardStatus
    from nav2_msgs.action import FollowPath
    from rclpy.action import ActionClient
    from std_srvs.srv import Empty
    from lifecycle_msgs.srv import GetState
    from rcl_interfaces.srv import GetParameters
    from rclpy.parameter import parameter_value_to_python
    from tf2_ros import Buffer, TransformListener
    from rosidl_runtime_py.convert import message_to_ordereddict

    started = time.monotonic()
    events = (EventWriter(out/'observations.jsonl',started,process=True) if args.input_thread
              else (out/'observations.jsonl').open('x'))
    def record(kind, **values):
        if args.input_thread:
            events.record(kind,**values)
            return
        row = plain_ros_value(dict(kind=kind, wall_s=time.monotonic()-started, **values))
        events.write(json.dumps(row, allow_nan=False)+'\n'); events.flush()
    def stamp(msg):
        return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
    layout = json.loads((bundle/'network.json').read_text())
    nodes = {n['id']: n for n in layout['nodes']}
    class Observer(Node):
        def __init__(self, robot, config):
            super().__init__('workshop_readiness', namespace=robot, cli_args=['--ros-args',
                '-r', '/tf:=/'+robot+'/tf', '-r', '/tf_static:=/'+robot+'/tf_static'])
            self.robot = robot; self.health = SensorHealth(); self.clock = None
            self.guard = None; self.guard_received = None; self.amcl = None
            self.tf = Buffer(); self.listener = TransformListener(self.tf, self)
            self.tf_ingress = None
            if args.input_thread:
                from tools.live_bridge.feedback_ingress import TransformIngress
                self.listener.unregister()
                self.tf_ingress = TransformIngress(self)
                self.tf = self.tf_ingress.tf
            self.nomotion = self.create_client(Empty, 'request_nomotion_update')
            self.nomotion_future = None; self.last_request = -math.inf
            self.action = ActionClient(self, FollowPath, 'follow_path')
            self.lifecycle_clients = {name: self.create_client(GetState, name+'/get_state')
                for name in ('controller_server', 'map_server', 'amcl')}
            self.lifecycle_pending = {}; self.lifecycle_states = {}; self.last_lifecycle = -math.inf
            self.expected_parameters = expected_parameters(bundle, robot, config)
            self.parameter_clients = {name: self.create_client(GetParameters, name+'/get_parameters')
                                      for name in self.expected_parameters}
            self.parameter_pending = {}; self.parameters_verified = set()
            self.initial = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 1)
            self.position = nodes[config['start_node']]
            self.spawn_yaw = next(robot.get('start_yaw_rad', 0.) for robot in layout['robots'] if robot['id'] == self.robot)
            self.spawn_prior = SpawnPriorHandshake(self.position['x'], self.position['y'], self.spawn_yaw)
            self.truth = self.create_client(GetEntityState, '/wheelset_review/get_entity_state')
            self.pending = None; self.last_check = -math.inf; self.checks = []
            self.link_pending = None; self.last_link_query = -math.inf
            self.pending_query = None; self.query_sequence = 0
            self.unpaired = deque()
            self.latest_check_wall = None
            self.latest_readiness_eval_wall = None
            self.latest_readiness_ready = None
            self.sensor_worker = None; self._input_received_wall = None
            if args.input_thread:
                from tools.live_bridge.feedback_ingress import InputWorker
                self.sensor_worker = InputWorker(self,'amr_observer_sensor_ingress',namespace=self.get_namespace())
            def subscribe(cls,topic,callback,qos):
                if self.sensor_worker is None:
                    self.create_subscription(cls,topic,callback,qos)
                    return
                def receive(message,received):
                    self._input_received_wall = received
                    callback(message)
                self.sensor_worker.subscription(cls,topic,receive,qos)
            subscribe(Clock, '/clock', lambda m: setattr(self, 'clock', m.clock.sec+m.clock.nanosec*1e-9), qos_profile_sensor_data)
            subscribe(LaserScan, 'scan', self.scan, qos_profile_sensor_data)
            subscribe(Imu, 'imu/data', self.imu, qos_profile_sensor_data)
            subscribe(Odometry, 'odometry/filtered', self.odom, qos_profile_sensor_data)
            subscribe(Odometry, 'wheel/odometry', self.wheel_odom, qos_profile_sensor_data)
            if manifest.get('encoder_source') == 'joint_positions':
                subscribe(Odometry, 'wheel/position_odometry', self.position_odom, qos_profile_sensor_data)
            subscribe(JointState, 'joint_states', self.wheel_joints, qos_profile_sensor_data)
            subscribe(PoseWithCovarianceStamped, 'amcl_pose', self.localized, qos_profile_sensor_data)
            subscribe(GuardStatus, 'motion_guard/status', self.on_guard, 10)
            if self.sensor_worker is not None:self.sensor_worker.start()

        def received_now(self):
            return time.monotonic() if self._input_received_wall is None else self._input_received_wall

        def observe(self, source, message, valid):
            received = self.received_now()
            self.health.observe(source, stamp(message), received, valid)
            return received-started

        def scan(self, m):
            useful = sum(math.isfinite(v) and m.range_min <= v < m.range_max for v in m.ranges)
            received = self.observe('scan', m, bool(m.ranges) and useful >= max(30, .1*len(m.ranges)))
            record('scan', robot=self.robot, source_s=stamp(m), beams=len(m.ranges), finite_returns=useful,
                received_wall_s=received,
                frame_id=m.header.frame_id, range_min=m.range_min, range_max=m.range_max,
                angle_min=m.angle_min, angle_max=m.angle_max, angle_increment=m.angle_increment,
                ranges=[v if math.isfinite(v) else None for v in m.ranges],
                nonfinite_ranges='null represents no finite return')

        def imu(self, m):
            received = self.observe('imu', m, math.isfinite(m.angular_velocity.z))
            record('imu', robot=self.robot, source_s=stamp(m), yaw_rate=m.angular_velocity.z,
                   received_wall_s=received)

        def odom(self, m):
            t = m.twist.twist
            received = self.observe('odom', m, all(math.isfinite(v) for v in (t.linear.x, t.linear.y, t.angular.z)))
            record('filtered_odom', robot=self.robot, source_s=stamp(m),
                   received_wall_s=received, vx=t.linear.x, vy=t.linear.y, wz=t.angular.z)

        def localized(self, m):
            a, q = m.pose.pose.position, m.pose.pose.orientation
            self.amcl = m
            received = self.observe('amcl', m, all(math.isfinite(v) for v in (a.x, a.y, q.x, q.y, q.z, q.w))
                                    and pose_quality(m.pose.covariance, (a.x, a.y)))
            record('amcl', robot=self.robot, received_wall_s=received, message=message_to_ordereddict(m))

        def wheel_odom(self, m):
            record('wheel_odometry', robot=self.robot, source_s=stamp(m),
                   received_wall_s=self.received_now()-started, message=message_to_ordereddict(m))

        def wheel_joints(self, m):
            record('wheel_joints', robot=self.robot, source_s=stamp(m),
                   received_wall_s=self.received_now()-started, message=message_to_ordereddict(m))

        def position_odom(self, m):
            record('position_encoder_odometry', robot=self.robot, source_s=stamp(m),
                   received_wall_s=self.received_now()-started, message=message_to_ordereddict(m))

        def on_guard(self, m):
            self.guard = m; self.guard_received = self.received_now()
            record('guard', robot=self.robot, received_wall_s=self.guard_received-started,
                   message=message_to_ordereddict(m))

        def poll(self, now):
            if self.sensor_worker is not None:
                self.sensor_worker.check_owner()
                with self.sensor_worker.lock:
                    if self.sensor_worker.error or not self.sensor_worker.thread.is_alive():
                        raise RuntimeError('observer sensor input: '+str(self.sensor_worker.error))
                    return self.poll_ready(time.monotonic())
            # ``now`` is sampled before callback draining in the outer loop.
            # Readiness age and the common startup window must use the actual
            # post-poll evaluation time for both ingress modes.
            return self.poll_ready(time.monotonic())

        def poll_ready(self, now):
            self.latest_readiness_eval_wall = now
            if self.tf_ingress is not None:
                self.tf_ingress.check()
            if self.clock is None:
                return False
            if self.link_pending is not None and self.link_pending.done():
                reply = self.link_pending.result(); self.link_pending = None
                record('independent_base_link', robot=self.robot, request=self.link_query,
                       received_wall_s=now-started, message=message_to_ordereddict(reply))
            if self.link_pending is None and now-self.last_link_query >= .5 and self.truth.service_is_ready():
                request = GetEntityState.Request()
                request.name = self.robot+'::base_footprint'; request.reference_frame = 'world'
                self.link_query = message_to_ordereddict(request)
                self.link_pending = self.truth.call_async(request); self.last_link_query = now
            for name, future in list(self.parameter_pending.items()):
                if not future.done():
                    continue
                reply = future.result(); del self.parameter_pending[name]
                expected = self.expected_parameters[name]
                actual = dict(zip(expected, (parameter_value_to_python(v) for v in reply.values)))
                mismatches = parameter_mismatches(expected, actual)
                record('runtime_parameters', robot=self.robot, node=name, expected=expected,
                       actual=actual, mismatches=mismatches, received_wall_s=now-started)
                if mismatches:
                    raise RuntimeError(f'{self.robot}/{name} runtime parameter mismatch: {mismatches}')
                self.parameters_verified.add(name)
            for name, client in self.parameter_clients.items():
                if name not in self.parameters_verified and name not in self.parameter_pending and client.service_is_ready():
                    request = GetParameters.Request(); request.names = list(self.expected_parameters[name])
                    self.parameter_pending[name] = client.call_async(request)
            for name, future in list(self.lifecycle_pending.items()):
                if future.done():
                    response = future.result(); del self.lifecycle_pending[name]
                    if response is not None:
                        self.lifecycle_states[name] = (response.current_state.id, now)
                        record('lifecycle', robot=self.robot, node=name,
                               received_wall_s=now-started,
                               state=response.current_state.id, label=response.current_state.label)
            if now-self.last_lifecycle >= 1.:
                for name, client in self.lifecycle_clients.items():
                    if name not in self.lifecycle_pending and client.service_is_ready():
                        self.lifecycle_pending[name] = client.call_async(GetState.Request())
                self.last_lifecycle = now
            g = self.guard
            motion_locked = bool(g is not None and self.guard_received is not None
                and 0 <= now-self.guard_received < .2 and -.1 <= self.clock-stamp(g) < .25
                and g.state == 'latched' and g.stopped
                and g.output_vx == g.output_wz == 0 and g.actual_speed < .01 and abs(g.actual_wz) < .02)
            sensor_ready = all(name in self.health.samples and self.health.samples[name]['valid']
                and 0 <= now-self.health.samples[name]['received'] <= self.health.limits[name]
                and -.1 <= self.clock-self.health.samples[name]['stamp'] <= self.health.limits[name]
                for name in ('scan', 'imu', 'odom'))
            amcl_active = ('amcl' in self.lifecycle_states and self.lifecycle_states['amcl'][0] == 3
                           and now-self.lifecycle_states['amcl'][1] < 2.)
            estimate = None
            amcl_sample = self.health.samples.get('amcl')
            if self.amcl is not None:
                a, q = self.amcl.pose.pose.position, self.amcl.pose.pose.orientation
                estimate = (a.x, a.y, math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)))
            prior_action = self.spawn_prior.update(now, self.clock,
                ready=bool(self.initial.get_subscription_count() and sensor_ready and amcl_active),
                motion_locked=motion_locked, estimate=estimate,
                estimate_source_s=None if self.amcl is None else stamp(self.amcl),
                estimate_received=None if amcl_sample is None else amcl_sample['received'],
                estimate_valid=amcl_sample is not None and amcl_sample['valid'])
            if prior_action == 'acknowledged':
                record('initial_pose_acknowledged', robot=self.robot, attempts=self.spawn_prior.attempts,
                       estimate=list(estimate), estimate_source_s=stamp(self.amcl),
                       estimate_received_wall_s=amcl_sample['received']-started,
                       source='fresh AMCL estimate near declared spawn prior; independent readiness checks retained')
            if prior_action == 'send':
                initial = PoseWithCovarianceStamped(); initial.header.frame_id = 'map'
                initial.header.stamp.sec = int(self.clock)
                initial.header.stamp.nanosec = round((self.clock-int(self.clock))*1e9)
                # A perturbed spawn prior exercises correction; no state service is used here.
                initial.pose.pose.position.x = self.position['x']+.15
                initial.pose.pose.position.y = self.position['y']-.08
                initial.pose.pose.orientation.z = math.sin((self.spawn_yaw+.05)/2)
                initial.pose.pose.orientation.w = math.cos((self.spawn_yaw+.05)/2)
                for index in (0, 7, 35): initial.pose.covariance[index] = .1**2
                self.initial.publish(initial)
                record('initial_pose', robot=self.robot, message=message_to_ordereddict(initial),
                       attempt=self.spawn_prior.attempts,
                       source='known spawn prior plus fixed perturbation; independent of state queries')
            if now-self.last_request >= .5:
                if self.nomotion_future is not None and not self.nomotion_future.done():
                    self.nomotion.remove_pending_request(self.nomotion_future)
                    self.nomotion_future.cancel()
                if self.nomotion.service_is_ready():
                    self.nomotion_future = self.nomotion.call_async(Empty.Request())
                self.last_request = now
            if self.pending is not None and self.pending.done():
                reply = self.pending.result(); self.pending = None
                if reply is not None and reply.success:
                    self.unpaired.append((reply, now, self.pending_query))
                else:
                    # A failed or missing GetEntityState reply used to be dropped
                    # with no record at all: the query id simply vanished from the
                    # stream. Record it so the audit account stays closed and a
                    # physics service that cannot answer before the model exists is
                    # visible instead of looking like a missing query.
                    # gazebo_msgs/GetEntityState_Response in Humble carries only
                    # success/state/header; read status_message defensively so
                    # recording a failed reply can never abort the owner loop.
                    status = None if reply is None else getattr(reply, 'status_message', None)
                    record('pose_check_service_failed', robot=self.robot,
                           state_query=self.pending_query,
                           reply_success=None if reply is None else bool(reply.success),
                           status_message=None if status is None else str(status),
                           reason='physics state service returned no usable reply')
                self.pending_query = None
            while self.unpaired:
                reply, received, query = self.unpaired[0]
                try:
                    tf = self.tf.lookup_transform('map', 'base_footprint', rclpy.time.Time.from_msg(reply.header.stamp))
                except Exception as exc:
                    # A physics service reply can arrive before the same-time
                    # EKF TF. Retain it for bounded pairing instead of dropping
                    # every query as an extrapolation into the future.
                    if now-received <= 1.:
                        break
                    self.unpaired.popleft()
                    record('pose_check_expired_tf', robot=self.robot, source_s=stamp(reply),
                           state_query=query, reason=str(exc))
                    continue
                self.unpaired.popleft()
                if -.1 <= self.clock-stamp(reply) <= .25:
                    a, q = reply.state.pose.position, reply.state.pose.orientation
                    b, r = tf.transform.translation, tf.transform.rotation
                    yaw = lambda q: math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
                    angle = math.atan2(math.sin(yaw(q)-yaw(r)), math.cos(yaw(q)-yaw(r)))
                    row = dict(source_s=stamp(reply), truth=[a.x, a.y, yaw(q)],
                        localized=[b.x, b.y, yaw(r)], error_m=math.hypot(a.x-b.x, a.y-b.y),
                        error_rad=abs(angle), speed_mps=math.hypot(reply.state.twist.linear.x, reply.state.twist.linear.y),
                        health=self.health.reason(self.clock, now), evaluated_wall_s=now-started,
                        state_received_wall_s=received-started, clock_s=self.clock,
                        state_query=query,
                        state_reply=message_to_ordereddict(reply), transform=message_to_ordereddict(tf))
                    self.checks.append(row); self.latest_check_wall = now
                    record('independent_pose_check', robot=self.robot, **row)
                else:
                    # A reply that paired with a TF but arrived outside the
                    # freshness window used to be dropped silently: the
                    # unpaired queue advanced, nothing was recorded, and the
                    # readiness diagnostic then showed only a growing
                    # "last_pose_check_age" with no explanation. Record the
                    # rejection so a stale physics service is visible instead
                    # of impersonating a TF outage.
                    record('pose_check_rejected_stale', robot=self.robot,
                           source_s=stamp(reply), clock_s=self.clock,
                           age_s=self.clock-stamp(reply),
                           state_query=query,
                           reason='physics reply outside [-0.1, 0.25] s of clock')
            if self.pending is None and now-self.last_check >= .5 and self.truth.service_is_ready():
                request = GetEntityState.Request(); request.name = self.robot; request.reference_frame = 'world'
                self.pending_query = dict(query_id=self.query_sequence, service=self.truth.srv_name,
                    request=message_to_ordereddict(request), issued_wall_s=now-started)
                self.query_sequence += 1
                record('state_query_issued', robot=self.robot, **self.pending_query)
                self.pending = self.truth.call_async(request); self.last_check = now
            # Service and TF work above may take time; evaluate freshness at
            # the actual decision point while the sensor snapshot is locked.
            now = time.monotonic()
            self.latest_readiness_eval_wall = now
            g = self.guard
            action_ready = self.action.server_is_ready()
            ready = bool(self.spawn_prior.confirmed and self.health.reason(self.clock, now) is None and action_ready
                and self.parameters_verified == self.expected_parameters.keys()
                and all(name in self.lifecycle_states and self.lifecycle_states[name][0] == 3
                    and now-self.lifecycle_states[name][1] < 2. for name in self.lifecycle_clients)
                and g is not None and now-self.guard_received < .2 and g.state == 'latched' and g.stopped
                and -.1 <= self.clock-stamp(g) < .25
                and g.output_vx == g.output_wz == 0 and g.actual_speed < .01 and abs(g.actual_wz) < .02
                and self.checks and self.latest_check_wall is not None and now-self.latest_check_wall < 1.
                and self.checks[-1]['health'] is None and self.checks[-1]['error_m'] <= .15
                and self.checks[-1]['error_rad'] <= .20)
            if not ready and now-getattr(self,'last_readiness_diagnostic',-math.inf)>=1.:
                record('readiness_diagnostic',robot=self.robot,source_s=self.clock,
                    health=self.health.reason(self.clock,now),
                    guard_age_s=None if self.guard_received is None else now-self.guard_received,
                    guard_source_s=None if g is None else stamp(g),
                    last_pose_check_age_s=None if self.latest_check_wall is None else now-self.latest_check_wall,
                    lifecycle=self.lifecycle_states,parameter_nodes=sorted(self.parameters_verified),
                    action_server_ready=action_ready)
                self.last_readiness_diagnostic=now
            if getattr(self, 'last_readiness_state', None) != ready or ready:
                record('readiness_state', robot=self.robot, ready=ready, source_s=self.clock,
                       evaluated_wall_s=now-started, action_server_ready=action_ready,
                       parameter_nodes_required=list(self.expected_parameters))
                self.last_readiness_record = now
            self.last_readiness_state = ready
            self.latest_readiness_ready = ready
            return ready

    launch_path = ROOT/'tools/wheelset_scene/workshop_navigation.launch.py'
    source_paths = [Path(__file__), launch_path, ROOT/'tools/wheelset_scene/workshop_navigation.py',
                    ROOT/'tools/wheelset_scene/navigation_parameters.py',
                    ROOT/'tools/wheelset_scene/navigation_model.py',
                    ROOT/'tools/wheelset_scene/navigation_records.py',
                    ROOT/'tools/live_bridge/sensor_health.py']
    if transport_profile is not None:
        source_paths.append(transport_profile)
    if args.mode != 'readiness':
        source_paths += [ROOT/'tools/wheelset_scene/navigation_fleet.py',
                        ROOT/'tools/wheelset_scene/light_material_scheduler.py']
        source_paths += [ROOT/'tools/live_bridge'/name for name in
            ['fleet_scheduler.py', 'sensor_scheduler.py', 'nav2_scheduler.py', 'nav2_route_follower.py',
             'gz_frame.py', 'nav2_path.py', 'fleet_config.py', 'terminal_docking.py', 'nav2_local.yaml',
             'sensor_localization.yaml', 'nav2_metric.yaml', 'nav2_metric_dwb.yaml', 'feedback_ingress.py']]
        if (manifest.get('contact_bins') or {}).get('terminal_docking',{}).get('measurement'):
            source_paths += [ROOT/'tools/live_bridge'/name for name in
                             ('station_refinement.py','station_fit_worker.py')]
        source_paths += sorted((ROOT/'amr_sched').rglob('*.py'))
    if args.mode == 'faults':
        source_paths += [ROOT/'tools/wheelset_scene/navigation_faults.py']
    if args.whole_line:
        source_paths += list(sorted((ROOT/'tools/wheelset_scene').glob('*.py')))
    if args.contact_delivery:
        source_paths += [ROOT/'tools/wheelset_scene'/name for name in
                         ('scout_bin_hardware.py', 'navigation_handling.py', 'navigation_bin_runtime.py', 'navigation_samples.py',
                          'navigation_equipment_ingress.py',
                          'inventory_supply.py', 'inventory_supply_ipc.py')]
    source_paths = sorted(set(source_paths))
    if manifest.get('encoder_source') == 'joint_positions':
        source_paths.append(ROOT/'tools/live_bridge/position_encoder.py')
    source_hashes = {}
    for path in source_paths:
        data = path.read_bytes(); source_hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(data).hexdigest()
        target = out/'executed_sources'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    command = ['ros2', 'launch', str(launch_path), 'bundle:='+str(bundle), 'model_output:='+str(out/'models')]
    log = (out/'launch.log').open('x')
    process = None; executor = None; observers = []; error = None; success = False
    readiness_window = ReadinessWindow(2.)
    runtime = None; fleet_started = None; fleet_summary = None; readiness_interval = None
    initialized = False; cleanup = None; campaign = None; fault_summary = None
    workshop = None; workshop_summary = None; fleet_finished = False
    contact_adapter = None; contact_summary = None
    try:
        # Resolve all child-node packages before any Gazebo action can start.
        for package in ('robot_localization', 'nav2_amcl', 'nav2_map_server', 'nav2_controller',
                        'nav2_lifecycle_manager', 'robot_state_publisher', 'gazebo_ros',
                        'scout_gazebo_description', 'amr_gazebo_guard'):
            get_package_prefix(package)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        rclpy.init(); initialized = True; executor = SingleThreadedExecutor()
        for robot, config in manifest['robot_configs'].items():
            observer = Observer(robot, config); observers.append(observer); executor.add_node(observer)
        if args.mode != 'readiness':
            sys.path.insert(0, str(ROOT/'tools/live_bridge'))
            from fleet_scheduler import FleetRuntime
            from .navigation_fleet import build_sensor_fleet
            case = 'delivery' if args.whole_line else 'short' if args.mode == 'faults' else args.mode
            inventory_supply = None
            if args.contact_delivery:
                from .navigation_fleet import build_line_side_fleet
                sim, inventory_supply = build_line_side_fleet(layout, run_id=uuid.uuid4().hex,
                    handling_source=manifest['contact_bins']['source'])
            else:
                sim = build_sensor_fleet(layout, run_id=uuid.uuid4().hex, case=case)
            dock_takeover = None
            if args.terminal_docking:
                dock_takeover = (manifest.get('contact_bins') or {}).get('terminal_docking')
                if not dock_takeover:
                    raise RuntimeError('terminal docking requires a bundle that declares it')
                dock_takeover=json.loads(json.dumps(dock_takeover))
                measurement=dock_takeover.get('measurement')
                if measurement is not None:
                    image=(bundle/measurement['map_image']).resolve()
                    if not image.is_relative_to(bundle.resolve()):
                        raise ValueError('station map must belong to the bound bundle')
                    measurement['map_image']=str(image)
            runtime = FleetRuntime(sim, out/'fleet', {r: r for r in manifest['robot_configs']},
                executor, runtime_files=[*source_paths, bundle/'manifest.json'], latest_tf=args.latest_tf,
                best_effort_tf=args.best_effort_tf,
                input_thread=args.input_thread, dock_takeover=dock_takeover,
                async_logs=args.async_logs)
            if args.input_thread:
                runtime.start_owner_pump()
            if args.contact_delivery:
                from .navigation_bin_runtime import ContactBinRuntime
                contact_adapter = ContactBinRuntime(runtime, executor, manifest['contact_bins'],
                    out/'contact_evidence', record, input_thread=args.input_thread,
                    drain_budget_s=args.contact_drain_ms/1000.,
                    drain_rows=args.contact_drain_rows)
            if args.mode == 'faults':
                from .navigation_faults import FaultCampaign
                campaign = FaultCampaign(runtime, executor, record)
            if args.whole_line:
                from .navigation_workshop import LocalizedWorkshop
                workshop = LocalizedWorkshop(runtime, executor, bundle, out, process.pid, record,
                    case=args.mode, inventory_supply=inventory_supply)
        loop_timing = dict(iterations=0, spin_attempts=0, max_spin_wall_s=0.)
        phase_timing = dict(pump_max_s=0., consume_max_s=0., poll_max_s=0., loop_max_s=0.)
        last_slow_record = -math.inf
        next_loop_timing = started+1.
        while time.monotonic()-started < args.wall_timeout_s:
            if process.poll() is not None:
                raise RuntimeError('owned navigation launch ended early')
            loop_started = time.monotonic()
            spin_started = time.monotonic()
            for attempt in range(args.callback_batch):
                executor.spin_once(timeout_sec=.01 if attempt == 0 else 0.)
                # Keep equipment polling and safety checks bounded even when
                # more callbacks are available. One slow callback is logged;
                # the drain never creates an unbounded busy loop.
                if time.monotonic()-spin_started >= .005:
                    break
            spin_elapsed = time.monotonic()-spin_started
            pump_started = time.monotonic()
            if runtime is not None and args.input_thread:
                runtime.pump_owner()
            pump_elapsed = time.monotonic()-pump_started
            now = time.monotonic()
            loop_timing['iterations'] += 1
            loop_timing['spin_attempts'] += attempt+1
            loop_timing['max_spin_wall_s'] = max(loop_timing['max_spin_wall_s'], spin_elapsed)
            if now >= next_loop_timing:
                record('event_loop_timing', callback_batch=args.callback_batch,
                    drain_budget_s=args.contact_drain_ms/1000., phase_max_s=dict(phase_timing),
                    **loop_timing)
                loop_timing = dict(iterations=0, spin_attempts=0, max_spin_wall_s=0.)
                phase_timing = dict(pump_max_s=0., consume_max_s=0., poll_max_s=0.,
                                    loop_max_s=0.)
                next_loop_timing = now+1.
            consume_elapsed = 0.
            if contact_adapter is not None and contact_adapter.ingress is not None:
                # Observe equipment while waiting for readiness; no commands or
                # handoff state advance occurs before the original fleet gate.
                consume_started = time.monotonic()
                contact_adapter.consume_inputs()
                consume_elapsed = time.monotonic()-consume_started
            poll_started = time.monotonic()
            states = [o.poll(now) for o in observers]
            readiness_evaluated = [o.latest_readiness_eval_wall for o in observers]
            if workshop is not None:
                workshop.poll_inventory()
            poll_elapsed = time.monotonic()-poll_started
            phase_timing['pump_max_s'] = max(phase_timing['pump_max_s'], pump_elapsed)
            phase_timing['consume_max_s'] = max(phase_timing['consume_max_s'], consume_elapsed)
            phase_timing['poll_max_s'] = max(phase_timing['poll_max_s'], poll_elapsed)
            loop_elapsed = time.monotonic()-loop_started
            phase_timing['loop_max_s'] = max(phase_timing['loop_max_s'], loop_elapsed)
            # The guard gate needs a fresh owner turn inside 200-250 ms. Record
            # which phase consumed a slow turn instead of inferring it later.
            guard_ages = {o.robot: (None if o.guard_received is None
                                    else time.monotonic()-o.guard_received) for o in observers}
            clock_gaps = {}
            for o in observers:
                stamp_s = (None if o.guard is None else
                           o.guard.header.stamp.sec+o.guard.header.stamp.nanosec*1e-9)
                clock_gaps[o.robot] = (None if o.clock is None or stamp_s is None
                                       else o.clock-stamp_s)
            if ((loop_elapsed >= .05 or any(v is not None and v >= .2 for v in guard_ages.values()))
                    and time.monotonic()-started-last_slow_record >= .2):
                last_slow_record = time.monotonic()-started
                record('loop_phase_slow', loop_s=loop_elapsed, spin_s=spin_elapsed,
                       pump_s=pump_elapsed, consume_s=consume_elapsed, poll_s=poll_elapsed,
                       guard_age_s=guard_ages, clock_minus_guard_s=clock_gaps,
                       contact_drain_ms=args.contact_drain_ms,
                       contact_drain_rows=args.contact_drain_rows)
            if fleet_started is None and now-started > args.startup_timeout_s:
                raise RuntimeError('stationary localization/Nav2 startup timeout')
            poll_finished = time.monotonic()
            adapter_ready = runtime is None or all(n.clock_s is not None and n.guard_fresh()
                and n.health.reason(n.clock_s, poll_finished) is None for n in runtime.nodes.values())
            interval = (readiness_window.update(states, readiness_evaluated, adapter_ready,
                                               checked_at=poll_finished)
                        if fleet_started is None else None)
            if fleet_started is None and interval is not None:
                readiness_interval = dict(start_wall_s=interval[0]-started,
                                          end_wall_s=interval[1]-started,
                                          source_s=min(o.clock for o in observers))
                record('readiness_satisfied', **readiness_interval)
                if runtime is None:
                    success = True; break
                runtime.enable(); fleet_started = min(o.clock for o in observers)
                record('fleet_enabled', source_s=fleet_started, case=args.mode,
                       authority='localized map TF and filtered encoder/IMU velocity; no truth feedback')
            if runtime is not None and fleet_started is not None:
                if contact_adapter is not None:
                    contact_adapter.poll(now)
                process_finished = workshop is None or workshop.poll(now)
                if campaign is not None:
                    campaign.poll(now)
                elif any(g.fault for g in runtime.sim.executions.values()):
                    raise RuntimeError('sensor fleet latched: '+str({r: g.fault for r, g in runtime.sim.executions.items()}))
                if runtime.sim._all_done() and all(n.clean_idle_shutdown_ready() for n in runtime.nodes.values()):
                    if campaign is not None and not campaign.summary()['passed']:
                        raise RuntimeError('fleet finished without all requested fault recoveries')
                    if not fleet_finished:
                        record('fleet_completed', source_s=min(o.clock for o in observers), case=args.mode)
                        fleet_finished = True
                    if process_finished:
                        success = True; break
        if not success:
            raise RuntimeError('two-robot sensor/localization/Nav2 '+args.mode+' timeout')
    except Exception as exc:
        error = repr(exc)
    finally:
        if workshop is not None:
            try:
                workshop_summary = workshop.close()
            except Exception as exc:
                error = '; '.join(filter(None, [error, 'workshop cleanup: '+repr(exc)]))
        if runtime is not None:
            try:
                runtime.begin_close()
            except Exception as exc:
                error = '; '.join(filter(None, [error, 'fleet stop request: '+repr(exc)]))
        if executor is not None:
            if contact_adapter is not None:
                try:
                    contact_summary = contact_adapter.close()
                except Exception as exc:
                    error = '; '.join(filter(None, [error, 'contact cleanup: '+repr(exc)]))
        if runtime is not None:
            try:
                runtime.close()
                fleet_summary = dict(case=args.mode, started_source_s=fleet_started,
                    tasks={t.task_id: t.status.value for t in runtime.sim.tasks},
                    gates={r: g.snapshot() for r, g in runtime.sim.executions.items()},
                    pins=runtime.sim._res.pins())
            except Exception as exc:
                error = '; '.join(filter(None, [error, 'fleet cleanup: '+repr(exc)]))
        if executor is not None:
            if campaign is not None:
                fault_summary = campaign.summary(); campaign.close()
            executor.shutdown(timeout_sec=3)
        for observer in observers:
            if observer.sensor_worker is not None:
                try:
                    record('observer_sensor_ingress_closed',robot=observer.robot,**observer.sensor_worker.close())
                except Exception as exc:
                    error='; '.join(filter(None,[error,'observer sensor cleanup: '+repr(exc)]))
            if observer.tf_ingress is not None:
                try:
                    record('observer_tf_ingress_closed',robot=observer.robot,**observer.tf_ingress.close())
                except Exception as exc:
                    error='; '.join(filter(None,[error,'observer input cleanup: '+repr(exc)]))
            observer.destroy_node()
        if initialized:
            rclpy.shutdown()
        if process is not None:
            # The group was created above. A dead leader can still have a live
            # gzserver child, so process.poll() must not skip group cleanup.
            try:
                cleanup = stop_launch_group(process)
                if not cleanup['passed']: raise RuntimeError('owned launch group still active')
            except Exception as exc:
                error = '; '.join(filter(None, [error, 'launch cleanup: '+repr(exc)]))
            (out/'launch_process_cleanup.json').write_text(json.dumps(cleanup, indent=2)+'\n')
        # close() can time out while the process is still serializing evidence.
        # The world is already shut down above; never hash a live child writer.
        workshop_evidence = None
        if workshop is not None:
            try:
                if cleanup is None or not cleanup['passed']:
                    raise RuntimeError('evidence closure requires confirmed owned-world shutdown')
                workshop_evidence = workshop.finish_evidence()
                if not workshop_evidence['evidence_complete']:
                    raise RuntimeError('whole-line evidence writer did not complete')
            except Exception as exc:
                error = '; '.join(filter(None, [error, 'workshop evidence: '+repr(exc)]))
        log.close()
        try:
            raw_writer_summary=events.close()
        except Exception as exc:
            raw_writer_summary=dict(passed=False,error=repr(exc))
            error='; '.join(filter(None,[error,'raw evidence cleanup: '+repr(exc)]))
    workshop_files_stable = workshop is None or (workshop_evidence is not None and workshop_evidence['writer_exited'])
    sources_preserved = all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
                            for name, digest in source_hashes.items())
    result = dict(passed=success and error is None, error=error, command=command,
        callback_batch=args.callback_batch,
        input_thread=args.input_thread,
        latest_tf=args.latest_tf,
        best_effort_tf=args.best_effort_tf,
        udp_only=args.udp_only,
        async_logs=args.async_logs,
        transport_profile_sha256=(hashlib.sha256(transport_profile.read_bytes()).hexdigest()
                                  if transport_profile is not None else None),
        raw_writer=raw_writer_summary,
        contact_delivery=contact_summary,
        mode=args.mode, fleet=fleet_summary, fault_campaign=fault_summary, executed_sources_preserved=sources_preserved,
        readiness_interval=readiness_interval, network=layout, build_fingerprint=manifest['build_fingerprint'],
        runtime_parameters_required=True,
        whole_line=args.whole_line, workshop=workshop_summary, workshop_evidence=workshop_evidence,
        elapsed_wall_s=time.monotonic()-started, guard_build=build,
        start_monotonic_s=started,
        bundle_manifest_sha256=hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest(),
        source_sha256=source_hashes,
        files_sha256={path.relative_to(out).as_posix(): file_sha256(path)
            for path in out.rglob('*') if path.is_file()
            and (workshop_files_stable or not path.is_relative_to(out/'process'))},
        files_sha256_exclusions=[] if workshop_files_stable else ['process/ (writer closure unconfirmed)'],
        checks={o.robot: o.checks for o in observers},
        owned_launch_pid=process.pid if process else None, owned_launch_ended=process is None or process.poll() is not None,
        owned_process_group_stopped=process is None or cleanup is not None and cleanup['passed'],
        claim_boundary='Two SCOUT surrogates in the derived video workshop with encoder/IMU EKF, '
            'LiDAR AMCL, Nav2 FollowPath availability and native guard locked/stopped. '
            'Separate Gazebo truth checks audit error only. Mode readiness stays stationary; short runs only '
            'private-bay hops, delivery uses Python-owned full light-material tasks and reservations. '
            'Runner success alone is not independent motion/localization or combined workshop acceptance.')
    if not sources_preserved:
        result['passed'] = False
        result['error'] = '; '.join(filter(None, [result['error'], 'execution source changed during run']))
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['passed', 'error', 'elapsed_wall_s', 'owned_launch_ended']}))
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
