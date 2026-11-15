"""Chassis-side bridge that streams a vehicle pose to the RAWSim-O GUI.

Framing: TCP, one UTF-8 JSON object per line. The GUI connects to this bridge.
  bridge -> GUI: hello, pose, status, nav_result
  GUI -> bridge: nav, cancel
All coordinates on the wire are in the chassis frame (metres, radians); the GUI
converts them to the map frame with its calibration file.

Vehicle sources:
  circle  drive a circle forever (pose display test)
  sim     kinematic vehicle that follows nav paths (A->B test)
  replay  resend the poses of a JSONL log
  gazebo  pose from ROS 2 /odom (Gazebo SCOUT); optional /cmd_vel Pure Pursuit on nav
          with --pose-source truth it instead reads Gazebo ground truth
          (get_entity_state), applies the measured calibration, and announces
          frame="map" -- announcing that only because the transform was measured.
A real chassis driver implements the same Vehicle interface.
Standard library only, Python 3.8+.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import socket
import sys
import threading
import time

# The console asks this bridge for a route rather than planning one itself; the planner
# therefore stays on the Python side. See route_service.py for why that matters.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from route_service import RouteService  # noqa: E402

PROTO = 1
DEFAULT_PORT = 9300
#: A client that never sends a newline must not be able to grow the read buffer
#: indefinitely; one pending line larger than this drops the connection.
MAX_LINE_BYTES = 64 * 1024
log = logging.getLogger('live_bridge')


def wrap_angle(a):
    """Map an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def encode(msg):
    return (json.dumps(msg, separators=(',', ':'), ensure_ascii=False) + '\n').encode('utf-8')


def decode(line):
    msg = json.loads(line)
    if not isinstance(msg, dict) or not isinstance(msg.get('type'), str):
        raise ValueError('message must be a JSON object with a string "type"')
    return msg


def parse_path(msg):
    """Return the nav path as [(x, y), ...], or None if it is malformed."""
    path = msg.get('path')
    if not isinstance(path, list) or not path:
        return None
    out = []
    for p in path:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            return None
        x, y = p
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (x, y)):
            return None
        out.append((float(x), float(y)))
    return out


class Vehicle:
    """Interface for a pose source. step() returns (x, y, yaw) in the chassis frame."""

    on_result = None  # callback(goal_id, status, reason)

    def step(self, dt):
        raise NotImplementedError

    def nav(self, goal, path, max_speed=None):
        """Start following path; return False if this vehicle cannot navigate."""
        return False

    def cancel(self, goal):
        pass

    def stop(self):
        """Stop motion; the default has no motion to stop."""
        pass

    def pose_age(self):
        """Seconds since the pose source last produced a value, or None if unknown.

        Synthetic sources (circle / sim / replay) compute the pose on demand, so they are
        fresh by construction. A source backed by real sensors overrides this: a pose that
        stopped arriving is the one thing a downstream consumer cannot detect by itself,
        because the frozen value is still perfectly well-formed.
        """
        return 0.0

    def _report(self, goal, status, reason=None):
        if self.on_result:
            self.on_result(goal, status, reason)


class CircleVehicle(Vehicle):
    def __init__(self, cx, cy, radius, speed):
        self.cx, self.cy, self.r, self.speed = cx, cy, radius, speed
        self.phase = 0.0
        self.stopped = False
        self._pose = (cx + radius, cy, math.pi / 2.0)

    def step(self, dt):
        if self.stopped:
            return self._pose
        self.phase += self.speed / self.r * dt
        x = self.cx + self.r * math.cos(self.phase)
        y = self.cy + self.r * math.sin(self.phase)
        self._pose = (x, y, wrap_angle(self.phase + math.pi / 2.0))
        return self._pose

    def stop(self):
        self.stopped = True


class SimVehicle(Vehicle):
    """Unicycle that turns toward the next path point, then drives to it."""

    def __init__(self, x=0.0, y=0.0, yaw=0.0, max_speed=0.3, turn_rate=1.5,
                 reach_tol=0.03, heading_tol=0.15):
        self.x, self.y, self.yaw = x, y, yaw
        self.max_speed, self.turn_rate = max_speed, turn_rate
        self.reach_tol, self.heading_tol = reach_tol, heading_tol
        self._lock = threading.Lock()
        self._goal, self._path, self._speed = None, [], max_speed

    def step(self, dt):
        done = None
        with self._lock:
            while self._path:
                tx, ty = self._path[0]
                dist = math.hypot(tx - self.x, ty - self.y)
                if dist > self.reach_tol:
                    break
                self._path.pop(0)
            if self._goal is not None and not self._path:
                done, self._goal = self._goal, None
            elif self._path:
                tx, ty = self._path[0]
                dist = math.hypot(tx - self.x, ty - self.y)
                err = wrap_angle(math.atan2(ty - self.y, tx - self.x) - self.yaw)
                turn = max(-self.turn_rate * dt, min(self.turn_rate * dt, err))
                self.yaw = wrap_angle(self.yaw + turn)
                if abs(err) < self.heading_tol:
                    d = min(self._speed * dt, dist)
                    self.x += d * math.cos(self.yaw)
                    self.y += d * math.sin(self.yaw)
            pose = (self.x, self.y, self.yaw)
        if done is not None:
            self._report(done, 'succeeded')
        return pose

    def nav(self, goal, path, max_speed=None):
        with self._lock:
            old = self._goal
            self._goal, self._path = goal, list(path)
            self._speed = min(max_speed, self.max_speed) if max_speed else self.max_speed
        if old is not None and old != goal:
            self._report(old, 'canceled', 'replaced by ' + str(goal))
        return True

    def cancel(self, goal):
        with self._lock:
            hit = self._goal is not None and (goal is None or goal == self._goal)
            if hit:
                goal, self._goal, self._path = self._goal, None, []
        if hit:
            self._report(goal, 'canceled')

    def stop(self):
        with self._lock:
            self._goal, self._path = None, []


def read_replay(path):
    """Read poses from a JSONL log: bare pose messages or GUI records {"msg": {...}}."""
    poses = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msg = rec.get('msg', rec) if isinstance(rec, dict) else None
            if isinstance(msg, dict) and msg.get('type') == 'pose':
                poses.append((float(msg['t']), float(msg['x']), float(msg['y']), float(msg['yaw'])))
    poses.sort(key=lambda p: p[0])
    if not poses:
        raise ValueError('no pose messages in ' + path)
    return poses



class GazeboOdomVehicle(Vehicle):
    """Stream pose from ROS 2 /odom; optional Pure Pursuit → /cmd_vel on nav.

    Requires a sourced ROS 2 environment (rclpy). Used by --mode gazebo for
    Gazebo Classic SCOUT Mini (see AMR/tools/scout GZ-3). circle/sim/replay unchanged.
    """

    def __init__(self, odom_topic='/odom', cmd_topic='/cmd_vel', max_speed=0.35,
                 turn_rate=1.2, reach_tol=0.15, heading_tol=0.2, rate_hz=50.0):
        import threading as _threading
        self.odom_topic, self.cmd_topic = odom_topic, cmd_topic
        self.max_speed, self.turn_rate = max_speed, turn_rate
        self.reach_tol, self.heading_tol = reach_tol, heading_tol
        self._lock = _threading.Lock()
        self.x = self.y = self.yaw = 0.0
        self._have_odom = False
        #: monotonic timestamp of the last odom callback; None until the first one.
        self._last_odom_t = None
        self._goal, self._path, self._speed = None, [], max_speed
        self._stop = _threading.Event()
        self._node = None
        self._pub = None
        self._rclpy = None
        self._thread = _threading.Thread(target=self._ros_loop, name='gazebo-odom', daemon=True)
        self._thread.start()
        # Wait briefly for first odom (non-fatal if sim not up yet)
        deadline = time.time() + 2.0
        while time.time() < deadline and not self._have_odom and not self._stop.is_set():
            time.sleep(0.05)

    def _ros_loop(self):
        try:
            import rclpy
            from rclpy.node import Node
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
        except ImportError as e:
            log.error('gazebo mode needs rclpy: %s', e)
            return
        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        node = rclpy.create_node('live_bridge_gazebo_odom')
        self._node = node
        self._pub = node.create_publisher(Twist, self.cmd_topic, 10)

        def on_odom(msg):
            p = msg.pose.pose
            q = p.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            with self._lock:
                self.x, self.y, self.yaw = float(p.position.x), float(p.position.y), yaw
                self._have_odom = True
                self._last_odom_t = time.monotonic()

        node.create_subscription(Odometry, self.odom_topic, on_odom, 10)
        log.info('gazebo vehicle subscribed %s, publishing %s', self.odom_topic, self.cmd_topic)
        try:
            while not self._stop.is_set() and rclpy.ok():
                try:
                    rclpy.spin_once(node, timeout_sec=0.05)
                except Exception as e:
                    # ExternalShutdownException / context teardown while sibling ros2 CLIs exit
                    if 'Shutdown' in type(e).__name__ or 'shutdown' in str(e).lower():
                        break
                    raise
                self._control_once()
        finally:
            try:
                self._publish_cmd(0.0, 0.0)
            except Exception:
                pass
            try:
                node.destroy_node()
            except Exception:
                pass

    def pose_age(self):
        """How long ago /odom last delivered a message (None = never did).

        Also covers the case where the rclpy thread is gone entirely: no callbacks means
        the timestamp stops advancing, which is exactly what the caller must see.
        """
        t = self._last_odom_t
        if t is None:
            return None
        return time.monotonic() - t

    def _publish_cmd(self, vx, wz):
        if self._pub is None:
            return
        from geometry_msgs.msg import Twist
        tw = Twist()
        tw.linear.x = float(vx)
        tw.angular.z = float(wz)
        self._pub.publish(tw)

    def _control_once(self):
        """Turn-then-drive follower (SimVehicle-like) using live /odom."""
        with self._lock:
            # consume reached waypoints
            while self._path:
                tx, ty = self._path[0]
                if math.hypot(tx - self.x, ty - self.y) > self.reach_tol:
                    break
                self._path.pop(0)
            done = None
            if self._goal is not None and not self._path:
                done, self._goal = self._goal, None
            if done is not None:
                goal_done = done
                active = None
            elif self._path:
                goal_done = None
                tx, ty = self._path[0]
                active = (self.x, self.y, self.yaw, tx, ty, self._speed)
            else:
                goal_done = None
                active = None
        if goal_done is not None:
            self._publish_cmd(0.0, 0.0)
            self._report(goal_done, 'succeeded')
            return
        if active is None:
            return
        x, y, yaw, tx, ty, speed = active
        dist = math.hypot(tx - x, ty - y)
        err = wrap_angle(math.atan2(ty - y, tx - x) - yaw)
        # --- GZ-2 fix ---
        # planar_move 插件对纯角速度完全不响应，因此绝不允许 vx 归零。
        # 转角实测约为指令的 50%，用 TURN_GAIN 补偿。
        TURN_GAIN = 2.0
        CRAWL = 0.05          # 转向时的最小前进速度 m/s
        wz = max(-self.turn_rate, min(self.turn_rate, TURN_GAIN * 2.0 * err))
        if abs(err) > self.heading_tol:
            # 大误差：慢速前进 + 大角速度，边挪边转
            vz = CRAWL
        else:
            # 小误差：正常速度，按误差比例减速
            slow = max(0.35, 1.0 - abs(err) / 0.6)
            vz = max(CRAWL, min(speed, dist, speed * slow))
        self._publish_cmd(vz, wz)


    def step(self, dt):
        with self._lock:
            return self.x, self.y, self.yaw

    def nav(self, goal, path, max_speed=None):
        with self._lock:
            old = self._goal
            self._goal, self._path = goal, list(path)
            self._speed = min(max_speed, self.max_speed) if max_speed else self.max_speed
        if old is not None and old != goal:
            self._report(old, 'canceled', 'replaced by ' + str(goal))
        return True

    def cancel(self, goal):
        with self._lock:
            hit = self._goal is not None and (goal is None or goal == self._goal)
            if hit:
                goal, self._goal, self._path = self._goal, None, []
        self._publish_cmd(0.0, 0.0)
        if hit:
            self._report(goal, 'canceled')

    def stop(self):
        with self._lock:
            self._goal, self._path = None, []
        self._publish_cmd(0.0, 0.0)

    def shutdown(self):
        self.stop()
        self._stop.set()


def load_calibration(path):
    """Read a calibration file and return a callable (x, y, yaw) -> map (x, y, yaw).

    The file is the JSON written by tools/live_bridge/gz_calibrate_scene.py; its field names
    match RAWSim-O's CalibrationResult (camelCase) so the very same file can also be loaded
    through the GUI's own "加载标定" button. An empty path returns None (no calibration).
    """
    if not path:
        return None
    with open(path, 'r', encoding='utf-8') as fh:
        c = json.load(fh)
    theta = float(c['theta'])
    tx, ty = float(c['tx']), float(c['ty'])
    scale = float(c.get('scaleDiag', 1.0))
    mirror = bool(c.get('mirror', False))
    if not (scale > 0.0):
        raise ValueError('calibration scaleDiag must be positive, got %r' % scale)
    yaw_offset = float(c.get('yawOffset', 0.0))
    cth, sth = math.cos(theta), math.sin(theta)

    def to_map(x, y, yaw):
        wx, wy = (x, -y) if mirror else (x, y)
        wyaw = -yaw if mirror else yaw
        return (scale * (cth * wx - sth * wy) + tx,
                scale * (sth * wx + cth * wy) + ty,
                wrap_angle(theta + wyaw + yaw_offset))

    return to_map


class GazeboTruthVehicle(Vehicle):
    """Pose from Gazebo ground truth, expressed in map coordinates.

    Why this exists
    ---------------
    The GUI refuses to draw an odom-frame pose without a calibration, and that refusal is
    correct: /odom is wheel integration, and this project measured it reporting 0.879 m for
    at most 0.45 m of travel. Drawing it would place a plausible-looking marker in the wrong
    place. Ground truth comes from Gazebo's own get_entity_state service (via
    gz_pose_node.py) -- the only channel verified to answer; gz model -i hangs and
    /gazebo/.../pose/info publishes nothing.

    What the calibration is
    -----------------------
    The measured Sim(2) produced by tools/live_bridge/gz_calibrate_scene.py: several scene
    nodes visited, N ground-truth frames averaged at each, then a least-squares fit with a
    leave-one-out check. The bridge applies it here and therefore announces frame="map" --
    and only then, because applying a *measured* transform is what makes that claim true
    rather than assumed.

    This source observes; it does not drive. nav() returns False so the GUI is told the
    truth instead of silently accepting a goal that nothing will pursue.
    """

    def __init__(self, model='scout_mini', calibration=None, timeout=60.0):
        self.model = model
        self.to_map = load_calibration(calibration) if isinstance(calibration, str) else calibration
        self._lock = threading.Lock()
        self.x = self.y = self.yaw = 0.0
        self._last_ok = None
        self._error = None
        self._proc = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, args=(timeout,),
                                        name='gazebo-truth', daemon=True)
        self._thread.start()
        deadline = time.time() + min(timeout, 10.0)
        while time.time() < deadline and self._last_ok is None and not self._stop.is_set():
            time.sleep(0.05)

    def _read_loop(self, timeout):
        import subprocess
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gz_pose_node.py')
        try:
            self._proc = subprocess.Popen(
                [sys.executable, script, self.model],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1)
        except OSError as e:
            self._error = 'cannot start gz_pose_node: %s' % e
            log.error('%s', self._error)
            return
        proc = self._proc
        ready = proc.stdout.readline().strip()
        if ready != 'ready':
            self._error = 'gz_pose_node not ready: %r' % ready
            log.error('%s', self._error)
            return
        log.info('gazebo truth source live for model %s', self.model)
        while not self._stop.is_set():
            try:
                proc.stdin.write('POSE %s\n' % self.model)
                proc.stdin.flush()
                reply = proc.stdout.readline().strip()
                # A source that stops answering must remain reportable: a frozen pose on the
                # wire is indistinguishable from a parked vehicle.
                if not reply or reply.startswith('err') or len(reply.split()) < 3:
                    self._error = 'truth read failed: %r' % reply
                    time.sleep(0.2)
                    continue
                px, py, pyaw = (float(v) for v in reply.split()[:3])
                if self.to_map is not None:
                    px, py, pyaw = self.to_map(px, py, pyaw)
                with self._lock:
                    self.x, self.y, self.yaw = px, py, pyaw
                    self._last_ok = time.monotonic()
                    self._error = None
            except (OSError, ValueError) as e:
                self._error = 'truth read failed: %s' % e
                if self._stop.is_set():
                    break
                time.sleep(0.2)
        try:
            proc.stdin.write('QUIT\n')
            proc.stdin.flush()
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:                       # noqa: BLE001 - shutdown path never raises
            proc.kill()

    def step(self, dt):
        with self._lock:
            return self.x, self.y, self.yaw

    def pose_age(self):
        with self._lock:
            if self._last_ok is None:
                return None
            return time.monotonic() - self._last_ok

    def nav(self, goal, path, max_speed=None):
        """This source observes; it does not drive. Report that instead of pretending."""
        log.info('nav ignored: truth source does not command motion (goal=%s)', goal)
        return False

    def stop(self):
        """Nothing to stop: the bridge is not the thing flying the vehicle."""

    def shutdown(self):
        self._stop.set()


class ReplayVehicle(Vehicle):
    def __init__(self, poses, speed=1.0, loop=True):
        self.poses, self.speed, self.loop = poses, speed, loop
        self.t0, self.clock, self.i = poses[0][0], 0.0, 0
        self.stopped = False

    def step(self, dt):
        if self.stopped:
            _, x, y, yaw = self.poses[self.i]
            return x, y, yaw
        self.clock += dt * self.speed
        span = self.poses[-1][0] - self.t0
        if self.loop and span > 0 and self.clock > span:
            self.clock %= span
            self.i = 0
        while self.i + 1 < len(self.poses) and self.poses[self.i + 1][0] - self.t0 <= self.clock:
            self.i += 1
        _, x, y, yaw = self.poses[self.i]
        return x, y, yaw


class BridgeServer:
    """Accepts one GUI client at a time and publishes the vehicle pose at a fixed rate."""

    #: Seconds without a fresh pose before the bridge declares its own source stale.
    #: 1.0 s is ~30 missed messages at the 30 Hz /odom rate: long enough to ride out a
    #: scheduling hiccup, short enough that a dead source is obvious on screen.
    POSE_STALE_SECONDS = 1.0

    def __init__(self, vehicle, vehicle_id='agv1', host='127.0.0.1', port=DEFAULT_PORT,
                 rate=20.0, mode='sim', proto=PROTO, frame='sim', watchdog_seconds=0.0):
        self.vehicle, self.vehicle_id, self.rate, self.mode = vehicle, vehicle_id, rate, mode
        self.proto, self.frame, self.watchdog_seconds = proto, frame, watchdog_seconds
        self.estop = False
        self.last_client_message = time.monotonic()
        self._srv = socket.create_server((host, port))
        self.port = self._srv.getsockname()[1]
        self._client, self._send_lock = None, threading.Lock()
        self._stop = threading.Event()
        #: Answers "route to this waypoint?" using the scheduler planner. Lazily loads the
        #: scene on the first request, so a bridge that is never asked costs nothing.
        self.routes = RouteService()
        #: Turns a deviation reported by the execution layer into a scheduling decision.
        #: Created lazily: a bridge that never receives one loads nothing.
        self._monitor = None
        vehicle.on_result = self._on_result

    def _handle_deviation(self, msg):
        """Validates one report and returns the monitor's decision as a plain dict.

        A malformed report is answered, not ignored: the sender needs to know its message
        was rejected, otherwise it waits for a decision that will never come.
        """
        from amr_sched.monitor import DeviationEvent, DeviationEventError, ExecutionMonitor

        if self._monitor is None:
            network = self.routes.network()
            if network is None:
                return {'action': 'rejected', 'reason': '场景未加载'}
            self._monitor = ExecutionMonitor(network)
        try:
            event = DeviationEvent.from_dict(msg)
        except DeviationEventError as exc:
            return {'action': 'rejected', 'reason': str(exc)}
        goal = msg.get('goal')
        if not goal:
            return {'action': 'rejected', 'reason': '缺少 goal，无法判断要改去哪'}
        return self._monitor.handle(event, goal=str(goal)).to_dict()

    def send(self, msg):
        with self._send_lock:
            c = self._client
            if c is None:
                return False
            try:
                c.sendall(encode(msg))
                return True
            except OSError as e:
                log.info('send failed, dropping client: %s', e)
                self._drop(c)
                return False

    def _drop(self, c):
        if self._client is c:
            self._client = None
        try:
            c.close()
        except OSError:
            pass

    def _on_result(self, goal, status, reason):
        self.send({'type': 'nav_result', 'goal': goal, 'status': status, 'reason': reason})

    def handle(self, msg):
        kind = msg['type']
        self.last_client_message = time.monotonic()
        if kind == 'ping' and self.proto >= 1:
            t1 = msg.get('t1')
            if isinstance(t1, (int, float)) and not isinstance(t1, bool):
                now = time.time()
                self.send({'type': 'pong', 't1': t1, 't2': now, 't3': time.time()})
        elif kind == 'estop':
            self.estop = True
            self.vehicle.stop()
            self.send({'type': 'status', 'vehicle': self.vehicle_id, 'battery': None, 'mode': 'estop', 'error': 'estop'})
        elif kind == 'estop_release':
            self.estop = False
            self.send({'type': 'status', 'vehicle': self.vehicle_id, 'battery': None, 'mode': self.mode, 'error': None})
        elif kind == 'nav':
            path = parse_path(msg)
            goal = msg.get('goal')
            log.info('nav goal=%s points=%s', goal, None if path is None else len(path))
            speed = msg.get('max_speed')
            speed = speed if isinstance(speed, (int, float)) and speed > 0 else None
            if path is None:
                self._on_result(goal, 'failed', 'invalid path')
            elif self.estop:
                self._on_result(goal, 'failed', 'estop')
            elif not self.vehicle.nav(goal, path, speed):
                self._on_result(goal, 'failed', 'vehicle mode "%s" does not navigate' % self.mode)
        elif kind == 'route_request':
            # The console never plans. It asks, and it draws the answer.
            target = msg.get('target')
            here = msg.get('from')
            fx = fy = None
            if isinstance(here, (list, tuple)) and len(here) == 2:
                try:
                    fx, fy = float(here[0]), float(here[1])
                except (TypeError, ValueError):
                    fx = fy = None
            result = self.routes.plan(target, from_x=fx, from_y=fy,
                                      direct=bool(msg.get('direct')))
            log.info('route_request target=%s from=%s -> %s nodes%s',
                     target, here, len(result.get('nodes') or []),
                     '' if not result.get('reason') else ' reason=%s' % result['reason'])
            self.send(result)
        elif kind == 'route_deviation':
            # The local layer reports that it had to leave the global route. It must not
            # decide the new route itself: that would put a second decision maker outside
            # Python. Here the monitor decides and answers.
            decision = self._handle_deviation(msg)
            log.info('route_deviation robot=%s edge=%s -> %s',
                     msg.get('robot_id'), msg.get('blocked_edge'), decision.get('action'))
            self.send(dict(decision, type='route_decision'))
        elif kind == 'cancel':
            log.info('cancel goal=%s', msg.get('goal'))
            self.vehicle.cancel(msg.get('goal'))
        else:
            log.info('ignoring message type %r', kind)

    def _read(self, c):
        buf = b''
        try:
            while not self._stop.is_set():
                data = c.recv(4096)
                if not data:
                    break
                buf += data
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    if not line.strip():
                        continue
                    try:
                        self.handle(decode(line.decode('utf-8')))
                    except (ValueError, UnicodeDecodeError) as e:
                        log.warning('bad message skipped: %s', e)
                # A client that never sends a newline must not be able to grow
                # the buffer indefinitely; drop the connection once a single
                # pending line exceeds the limit.
                if len(buf) > MAX_LINE_BYTES:
                    log.warning('closing client: pending line exceeds %d bytes', MAX_LINE_BYTES)
                    break
        except OSError:
            pass
        finally:
            with self._send_lock:
                self._drop(c)
            log.info('client disconnected')

    def _accept(self):
        while not self._stop.is_set():
            try:
                c, addr = self._srv.accept()
            except OSError:
                break
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # Send hello before registering the client, so it is always the first message.
            with self._send_lock:
                if self._client is not None:
                    log.info('new client replaces the old one')
                    self._drop(self._client)
                try:
                    # "route" has to be advertised: the console decides from this list whether
                    # it may ask Python to plan. Leaving it out made a working capability look
                    # absent, so the console could only find out by sending a request and
                    # reading the refusal.
                    caps = ['ping', 'estop', 'nav', 'route'] if self.proto >= 1 else []
                    c.sendall(encode({'type': 'hello', 'vehicle': self.vehicle_id, 'proto': self.proto,
                                      'frame': self.frame, 'scene': getattr(self.routes, 'scene_id', None),
                                      'caps': caps}))
                except OSError:
                    c.close()
                    continue
                self._client = c
                self.last_client_message = time.monotonic()
            log.info('client connected from %s:%s', *addr[:2])
            threading.Thread(target=self._read, args=(c,), daemon=True).start()

    def serve(self):
        threading.Thread(target=self._accept, daemon=True).start()
        period, seq, last, next_status = 1.0 / self.rate, 0, time.perf_counter(), 0.0
        deadline = last
        while not self._stop.is_set():
            now = time.perf_counter()
            watchdog = (self.watchdog_seconds > 0 and self._client is not None
                        and now - self.last_client_message > self.watchdog_seconds)
            if self.estop or watchdog:
                self.vehicle.stop()
            x, y, yaw = self.vehicle.step(now - last)
            last = now
            # Ask the *source* how fresh it is. A frozen pose is indistinguishable from a
            # parked one once it is on the wire, so the only place this can be detected is
            # here, before it is sent.
            pose_age = self.vehicle.pose_age()
            pose_stale = pose_age is None or pose_age > self.POSE_STALE_SECONDS
            if pose_stale != getattr(self, '_pose_stale_last', False):
                if pose_stale:
                    log.error('pose source stalled: no fresh pose for %s s',
                              'n/a' if pose_age is None else round(pose_age, 2))
                else:
                    log.info('pose source recovered (%.2f s)', pose_age)
                self._pose_stale_last = pose_stale
            if self._client is not None:
                seq += 1
                self.send({'type': 'pose', 'vehicle': self.vehicle_id, 'seq': seq, 't': round(time.time(), 3),
                           'x': round(x, 4), 'y': round(y, 4), 'yaw': round(wrap_angle(yaw), 4)})
                if now >= next_status:
                    error = None
                    if self.estop:
                        error = 'estop'
                    elif watchdog:
                        error = 'watchdog'
                    elif pose_stale:
                        # Higher priority than nothing, lower than an operator stop: the
                        # vehicle is not being held, the bridge simply stopped knowing
                        # where it is, and every consumer must stop trusting the pose.
                        error = 'odom_stale'
                    self.send({'type': 'status', 'vehicle': self.vehicle_id, 'battery': None,
                               'mode': 'estop' if self.estop else self.mode,
                               'error': error})
                    next_status = now + 1.0
            # Absolute deadlines keep the mean rate exact (sleep overshoots); after a stall, restart the schedule.
            # perf_counter, because monotonic() ticks every 15.6 ms on Windows before Python 3.13.
            deadline += period
            delay = deadline - time.perf_counter()
            if delay < -period:
                deadline, delay = time.perf_counter(), 0.0
            time.sleep(max(0.0, delay))

    def stop(self):
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass
        with self._send_lock:
            if self._client is not None:
                self._drop(self._client)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--host', default='127.0.0.1', help='listen address; use 0.0.0.0 only on a trusted LAN')
    p.add_argument('--port', type=int, default=DEFAULT_PORT)
    p.add_argument('--vehicle', default='agv1')
    p.add_argument('--mode', choices=('circle', 'sim', 'replay', 'gazebo'), default='sim')
    p.add_argument('--rate', type=float, default=20.0, help='pose messages per second')
    p.add_argument('--center', type=float, nargs=2, default=(1.5, 1.5), metavar=('X', 'Y'))
    p.add_argument('--radius', type=float, default=0.9)
    p.add_argument('--speed', type=float, default=0.2, help='circle speed in m/s')
    p.add_argument('--start', type=float, nargs=3, default=(0.825, 1.275, 0.0), metavar=('X', 'Y', 'YAW'))
    p.add_argument('--max-speed', type=float, default=0.3, help='sim top speed in m/s')
    p.add_argument('--log', help='JSONL log for replay mode')
    p.add_argument('--replay-speed', type=float, default=1.0)
    p.add_argument('--pose-source', choices=('odom', 'truth'), default='odom',
                   help='gazebo mode: odom = ROS /odom topic; truth = Gazebo ground truth '
                        '(get_entity_state) transformed by --calibration')
    p.add_argument('--model', default=None,
                   help='gazebo truth mode: Gazebo model name (default: --vehicle)')
    p.add_argument('--calibration', default=None,
                   help='gazebo truth mode: calibration JSON (see gz_calibrate_scene.py); '
                        'without one the bridge will not claim frame=map')
    p.add_argument('--odom-topic', default='/odom', help='gazebo mode: odometry topic')
    p.add_argument('--cmd-topic', default='/cmd_vel', help='gazebo mode: Twist command topic')
    p.add_argument('--proto', type=int, choices=(0, 1), default=1,
                   help='protocol version announced in hello; 0 is the legacy compatibility mode')
    p.add_argument('--frame', choices=('chassis', 'odom', 'map', 'sim'), default='sim',
                   help='coordinate frame announced in hello')
    p.add_argument('--watchdog', type=float, default=0.5,
                   help='stop the vehicle when no message arrives for this many seconds (0 disables)')
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if a.mode == 'circle':
        vehicle = CircleVehicle(a.center[0], a.center[1], a.radius, a.speed)
    elif a.mode == 'replay':
        if not a.log:
            p.error('--mode replay needs --log')
        vehicle = ReplayVehicle(read_replay(a.log), a.replay_speed)
    elif a.mode == 'gazebo':
        if a.frame == 'sim':
            a.frame = 'odom'
        if a.pose_source == 'truth':
            # Ground truth plus a measured calibration is what makes frame="map" a fact
            # rather than a claim; without the file the frame stays honest.
            if not a.calibration:
                log.error('--pose-source truth without --calibration: the pose stays in the '
                          'Gazebo world frame, which is not the map; run gz_calibrate_scene.py')
            elif a.frame == 'odom':
                a.frame = 'map'
            vehicle = GazeboTruthVehicle(model=a.model or a.vehicle, calibration=a.calibration)
        else:
            vehicle = GazeboOdomVehicle(
                odom_topic=a.odom_topic, cmd_topic=a.cmd_topic, max_speed=a.max_speed)
    else:
        vehicle = SimVehicle(*a.start, max_speed=a.max_speed)
    # Proto 0 has no ping; a watchdog would stop it without any way to keep alive.
    watchdog = a.watchdog if a.proto >= 1 else 0.0
    server = BridgeServer(vehicle, a.vehicle, a.host, a.port, a.rate, a.mode,
                          proto=a.proto, frame=a.frame, watchdog_seconds=watchdog)
    log.info('bridge "%s" (%s mode) listening on %s:%d', a.vehicle, a.mode, a.host, server.port)
    try:
        server.serve()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        if hasattr(vehicle, 'shutdown'):
            vehicle.shutdown()


if __name__ == '__main__':
    main()
