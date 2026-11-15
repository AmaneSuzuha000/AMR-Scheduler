"""Input-only ROS workers; decisions and original receipt times stay with owner.

Adapted from frozen G06 sensor_ingress57/61 and shutdown69. Construction now
finishes before starting callbacks; TF, clock and sensors share one snapshot.
"""
from collections import deque
from copy import deepcopy
import threading
import time


def dynamic_tf_qos(*, latest_tf=False, best_effort_tf=False):
    """Choose a declared dynamic reader; static TF durability is untouched."""
    if latest_tf and best_effort_tf:
        raise ValueError('choose one explicit dynamic TF receiver')
    if not latest_tf and not best_effort_tf:
        return None
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    return QoSProfile(depth=100 if best_effort_tf else 1,
        reliability=ReliabilityPolicy.BEST_EFFORT if best_effort_tf else ReliabilityPolicy.RELIABLE)


class ReceiptQueue:
    """Keep transitions in arrival order; overflow is sticky, never drop-and-pass."""

    def __init__(self, limit=256):
        if not isinstance(limit, int) or limit < 1:
            raise ValueError('positive bounded input queue required')
        self.limit = limit; self.pending = deque(); self.error = None

    def append(self, value):
        if len(self.pending) >= self.limit:
            self.error = self.error or 'input queue overflow'
            return
        self.pending.append(value)

    def drain(self, limit=None):
        if limit is not None:
            if type(limit) is not int or limit < 1:
                raise ValueError('positive drain limit required')
            return [self.pending.popleft() for _ in range(min(limit, len(self.pending)))]
        values = list(self.pending); self.pending.clear()
        return values


class InputWorker:
    """One executor receives immutable observations, with no motion authority."""

    def __init__(self, owner, name, *, namespace=None, cli_args=None):
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        self.lock = threading.RLock(); self.stop = threading.Event()
        self.error = None; self.counts = {}; self.callback_threads = set()
        self.owner_thread = threading.get_ident(); self.thread = None
        self.node = Node(name, namespace=namespace, context=owner.context,
                         use_global_arguments=False, cli_args=cli_args)
        self.executor = SingleThreadedExecutor(context=owner.context)
        self.executor.add_node(self.node)

    def subscription(self, cls, topic, callback, qos):
        def receive(message):
            received = time.monotonic()
            with self.lock:
                callback(message, received)
                self.counts[topic] = self.counts.get(topic, 0)+1
                self.callback_threads.add(threading.get_ident())
        return self.node.create_subscription(cls, topic, receive, qos)

    def start(self):
        if self.thread is not None:
            raise RuntimeError('input worker already started')
        self.thread = threading.Thread(target=self.run, name=self.node.get_name())
        self.thread.start()

    def run(self):
        try:
            while not self.stop.is_set():
                self.executor.spin_once(timeout_sec=.02)
        except Exception as exc:
            with self.lock:
                self.error = type(exc).__name__+': '+str(exc)

    def check_owner(self):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError('only the decision thread may consume inputs')

    def close(self, before_destroy=None):
        self.check_owner(); self.stop.set(); self.executor.wake()
        if self.thread is not None:
            self.thread.join(timeout=2.)
        joined = self.thread is None or not self.thread.is_alive()
        stopped = False
        if joined:
            if before_destroy is not None: before_destroy()
            self.executor.remove_node(self.node)
            stopped = self.executor.shutdown(timeout_sec=1.)
            self.node.destroy_node()
        if not joined or not stopped:
            raise RuntimeError('input executor did not close within its bound')
        return dict(joined=joined, executor_stopped=stopped, error=self.error,
                    callbacks=dict(self.counts), callback_thread_ids=sorted(self.callback_threads))


class SensorIngress:
    """Receive localization and every guard transition without executing them."""

    def __init__(self, owner, *, latest_tf=False, best_effort_tf=False):
        from types import SimpleNamespace
        from amr_gazebo_guard.msg import GuardStatus
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import Imu, LaserScan
        from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
        from tf2_ros import Buffer, TransformListener
        from sensor_health import SensorHealth
        from sensor_scheduler import SensorSchedulerNode
        import rclpy
        self.owner = owner; self.clock = None; self.regression = None
        self.tf_edges = {}
        self.guard = ReceiptQueue(); self.reported_error = False
        self.state = SimpleNamespace(health=SensorHealth(), filtered=None, filtered_received=None,
                                     latest_scan=None,
                                     retain_station_scans=owner.station_refiner is not None)
        self.static_frames = set()
        self.state.note_sensor = lambda *a: SensorSchedulerNode.note_sensor(self.state, *a)
        ns = owner.get_namespace().rstrip('/')
        self.worker = InputWorker(owner, 'amr_feedback_ingress', namespace=owner.get_namespace(),
            cli_args=['--ros-args', '-r', '/tf:='+ns+'/tf', '-r', '/tf_static:='+ns+'/tf_static'])
        self.tf = Buffer(); self.time_zero = rclpy.time.Time()
        profile=dynamic_tf_qos(latest_tf=latest_tf,best_effort_tf=best_effort_tf)
        self.listener = (TransformListener(self.tf,self.worker.node,qos=profile) if profile is not None
                         else TransformListener(self.tf,self.worker.node))
        for subscription in (self.listener.tf_sub, self.listener.tf_static_sub):
            original = subscription.callback
            def bind(callback, topic):
                def receive(message):
                    received = time.monotonic()
                    with self.worker.lock:
                        callback(message)
                        self.note_transforms(message, received)
                        if topic.endswith('tf_static'):
                            self.static_frames.update(t.child_frame_id for t in message.transforms)
                        self.worker.counts[topic] = self.worker.counts.get(topic, 0)+1
                        self.worker.callback_threads.add(threading.get_ident())
                return receive
            subscription.callback = bind(original, subscription.topic_name)
        for suffix, cls, method in (
                ('scan', LaserScan, SensorSchedulerNode.on_scan),
                ('imu/data', Imu, SensorSchedulerNode.on_imu),
                ('odometry/filtered', Odometry, SensorSchedulerNode.on_filtered),
                ('amcl_pose', PoseWithCovarianceStamped, SensorSchedulerNode.on_amcl)):
            def bind(callback):
                def receive(message, received):
                    self.state._input_received_at = received
                    callback(self.state, message)
                return receive
            self.worker.subscription(cls, ns+'/'+suffix, bind(method), qos_profile_sensor_data)
        self.worker.subscription(Clock, '/clock', self.receive_clock, qos_profile_sensor_data)
        self.worker.subscription(GuardStatus, ns+'/motion_guard/status',
            lambda message, received: self.guard.append((message, received)), 10)
        self.worker.start()
        owner.record('feedback_ingress_started', thread_id=self.worker.thread.ident,
                     guard_queue_limit=self.guard.limit,
                     latest_tf=latest_tf,
                     best_effort_tf=best_effort_tf,
                     scope='Input only; owner retains task, motion, reservation and inventory authority.')

    def note_transforms(self, message, received):
        """Retain source and receipt times only, without authorizing freshness."""
        for transform in message.transforms:
            parent, child = transform.header.frame_id, transform.child_frame_id
            if (parent, child) not in (('map', 'odom'), ('odom', 'base_footprint')):
                continue
            stamp = transform.header.stamp
            self.tf_edges[parent+'->'+child] = dict(
                source_s=stamp.sec+stamp.nanosec*1e-9, received_wall_s=received)

    def transform_diagnostic(self):
        """Read both TF edges under the receiver lock at an existing rejection."""
        self.worker.check_owner()
        with self.worker.lock:
            edges = deepcopy(self.tf_edges)
            for key, parent, child in (
                    ('map->odom', 'map', 'odom'),
                    ('odom->base_footprint', 'odom', 'base_footprint')):
                item = edges.setdefault(key, {})
                try:
                    stamp = self.tf.lookup_transform(parent, child, self.time_zero).header.stamp
                    item['buffer_source_s'] = stamp.sec+stamp.nanosec*1e-9
                except Exception as exc:
                    item['buffer_error'] = str(exc)
            return edges

    def receive_clock(self, message, received):
        if self.clock is not None:
            before, after = self.clock.clock, message.clock
            if (after.sec, after.nanosec) < (before.sec, before.nanosec) and self.regression is None:
                self.regression = (self.clock, message)
        self.clock = message

    def pull(self):
        self.worker.check_owner()
        with self.worker.lock:
            clock, regression = self.clock, self.regression; self.regression = None
            health = deepcopy(self.state.health.samples)
            filtered, received = self.state.filtered, self.state.filtered_received
            scan = getattr(self.state, 'latest_scan', None)
            extrinsic = None
            if (scan is not None and getattr(self.owner, 'station_refiner', None) is not None
                    and scan['frame_id'] in self.static_frames):
                try:
                    from station_refinement import laser_pose
                    extrinsic = laser_pose(self.tf.lookup_transform(
                        'base_footprint', scan['frame_id'], self.time_zero))
                except Exception:
                    pass  # Missing/invalid TF is a rejected measurement, never a guessed extrinsic.
            pending = self.guard.drain()
            error = self.worker.error or self.guard.error
            if not self.worker.thread.is_alive(): error = error or 'input thread stopped'
            try:
                transform = self.tf.lookup_transform('map', 'base_footprint', self.time_zero)
                transform_error = None
            except Exception as exc:
                transform = None; transform_error = str(exc)
        if regression is not None:
            self.owner.on_clock(regression[0]); self.owner.on_clock(regression[1])
        if clock is not None: self.owner.on_clock(clock)
        self.owner.health.samples = {} if error else health
        self.owner.filtered, self.owner.filtered_received = filtered, received
        self.owner.input_transform, self.owner.input_transform_error = transform, transform_error
        self.owner.latest_scan, self.owner.laser_extrinsic = scan, extrinsic
        if error:
            self.owner.guard_status = self.owner.guard_received_at = None
            self.owner.sim.execution.fail('feedback input: '+error)
            if not self.reported_error:
                self.owner.record('feedback_ingress_failed', error=error); self.reported_error = True
            return
        for message, received in pending:
            self.owner.on_guard(message, received_at=received)

    def close(self):
        summary = self.worker.close(self.listener.unregister)
        self.owner.record('feedback_ingress_closed', pending_guard=len(self.guard.pending),
                          queue_error=self.guard.error, **summary)


class TransformIngress:
    """Independent observer TF reception; exact source-time lookup stays original."""

    def __init__(self, owner):
        from tf2_ros import Buffer, TransformListener
        ns = owner.get_namespace().rstrip('/')
        self.worker = InputWorker(owner, 'amr_observer_tf_ingress', namespace=owner.get_namespace(),
            cli_args=['--ros-args', '-r', '/tf:='+ns+'/tf', '-r', '/tf_static:='+ns+'/tf_static'])
        self.tf = Buffer(); self.listener = TransformListener(self.tf, self.worker.node)
        self.worker.start()

    def check(self):
        self.worker.check_owner()
        with self.worker.lock:
            error = self.worker.error
            if not self.worker.thread.is_alive(): error = error or 'observer TF input stopped'
        if error: raise RuntimeError('observer TF input: '+error)

    def close(self):
        return self.worker.close(self.listener.unregister)
