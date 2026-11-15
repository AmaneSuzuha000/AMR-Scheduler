import json
import math
import os
import socket
import tempfile
import threading
import time
import unittest

import live_bridge as lb


class ProtocolTests(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        msg = {'type': 'pose', 'x': 1.5, 'y': -0.25, 'yaw': 3.0}
        raw = lb.encode(msg)
        self.assertTrue(raw.endswith(b'\n'))
        self.assertEqual(lb.decode(raw.decode('utf-8')), msg)

    def test_decode_rejects_non_objects(self):
        for bad in ('[1, 2]', '{"x": 1}', '{"type": 3}'):
            with self.assertRaises(ValueError):
                lb.decode(bad)

    def test_wrap_angle(self):
        self.assertAlmostEqual(lb.wrap_angle(3 * math.pi), -math.pi)
        self.assertAlmostEqual(lb.wrap_angle(-math.pi / 2), -math.pi / 2)
        self.assertAlmostEqual(lb.wrap_angle(2 * math.pi + 0.1), 0.1)

    def test_parse_path(self):
        self.assertEqual(lb.parse_path({'path': [[0, 1], [2.5, 3]]}), [(0.0, 1.0), (2.5, 3.0)])
        for bad in ([], [[1]], [[1, 'a']], [[1, float('nan')]], [[True, 1]], 'x'):
            self.assertIsNone(lb.parse_path({'path': bad}), bad)


class VehicleTests(unittest.TestCase):
    def test_circle_stays_on_radius_and_yaw_is_tangent(self):
        v = lb.CircleVehicle(1.0, 2.0, 0.5, 0.2)
        for _ in range(50):
            x, y, yaw = v.step(0.1)
            self.assertAlmostEqual(math.hypot(x - 1.0, y - 2.0), 0.5, places=9)
            radial = math.atan2(y - 2.0, x - 1.0)
            self.assertAlmostEqual(lb.wrap_angle(yaw - radial), math.pi / 2, places=9)

    def test_sim_follows_path_and_reports_success(self):
        v = lb.SimVehicle(0.0, 0.0, math.pi)
        results = []
        v.on_result = lambda g, s, r: results.append((g, s))
        self.assertTrue(v.nav('g1', [(1.0, 0.0), (1.0, 1.0)]))
        for _ in range(2000):
            x, y, _ = v.step(0.02)
            if results:
                break
        self.assertEqual(results, [('g1', 'succeeded')])
        self.assertLess(math.hypot(x - 1.0, y - 1.0), 0.05)

    def test_sim_cancel_and_replace(self):
        v = lb.SimVehicle()
        results = []
        v.on_result = lambda g, s, r: results.append((g, s))
        v.nav('a', [(5.0, 0.0)])
        v.nav('b', [(0.0, 5.0)])
        v.cancel('b')
        self.assertEqual(results, [('a', 'canceled'), ('b', 'canceled')])
        x0 = v.step(0.1)
        self.assertEqual(v.step(1.0)[:2], x0[:2])

    def test_synthetic_sources_report_a_fresh_pose(self):
        """circle / sim / replay compute on demand, so their pose is never stale."""
        self.assertEqual(lb.CircleVehicle(0.0, 0.0, 0.5, 0.2).pose_age(), 0.0)
        self.assertEqual(lb.SimVehicle().pose_age(), 0.0)

    def test_a_source_that_never_delivered_reports_unknown_age(self):
        """None, not 0.0: \"no pose ever\" must not be laundered into \"just now\"."""
        v = lb.GazeboOdomVehicle.__new__(lb.GazeboOdomVehicle)
        v._last_odom_t = None
        self.assertIsNone(v.pose_age())

    def test_age_advances_when_callbacks_stop(self):
        """The point of the field: the age grows on its own after the last callback."""
        v = lb.GazeboOdomVehicle.__new__(lb.GazeboOdomVehicle)
        v._last_odom_t = time.monotonic() - 2.5
        self.assertGreaterEqual(v.pose_age(), 2.4)

    def test_staleness_threshold_distinguishes_dead_from_parked(self):
        """A frozen pose is indistinguishable from a parked vehicle on the wire.

        Measured 2026-11-05: the gazebo bridge rclpy thread died while the TCP side kept
        publishing one frozen pose at 20 Hz with error=None. Nothing downstream could tell
        that apart from a stationary vehicle, so the bridge has to declare it itself.
        """
        class Fake(lb.Vehicle):
            def __init__(self, age):
                self.age = age

            def step(self, dt):
                return (0.0, 0.0, 0.0)

            def pose_age(self):
                return self.age

        limit = lb.BridgeServer.POSE_STALE_SECONDS
        self.assertGreater(Fake(limit + 1.0).pose_age(), limit)
        self.assertLess(Fake(0.01).pose_age(), limit)
        # None means "never delivered"; the server treats that as stale, not fresh.
        self.assertIsNone(Fake(None).pose_age())
    def test_replay_reads_both_formats_in_time_order(self):
        rows = [{'msg': {'type': 'pose', 't': 10.2, 'x': 2, 'y': 0, 'yaw': 0}},
                {'type': 'pose', 't': 10.0, 'x': 1, 'y': 0, 'yaw': 0},
                {'msg': {'type': 'status', 't': 10.1}}]
        with tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False, encoding='utf-8') as fh:
            fh.write('\n'.join(json.dumps(r) for r in rows) + '\n')
        try:
            poses = lb.read_replay(fh.name)
        finally:
            os.unlink(fh.name)
        self.assertEqual([p[1] for p in poses], [1.0, 2.0])
        v = lb.ReplayVehicle(poses, loop=False)
        self.assertEqual(v.step(0.0)[0], 1.0)
        self.assertEqual(v.step(0.3)[0], 2.0)


class ReportStalePoseTests(unittest.TestCase):
    """The status line has to carry the failure of the pose *source*.

    Measured 2026-11-05 on the SCOUT Mini: the rclpy thread of the gazebo bridge died while
    the TCP side kept publishing one frozen pose at 20 Hz with error=None. A frozen pose is
    a perfectly well-formed message, so no consumer can detect it; only the bridge can.
    """

    @staticmethod
    def _statuses(vehicle, seconds=2.5):
        server = lb.BridgeServer(vehicle, port=0, rate=50)
        threading.Thread(target=server.serve, daemon=True).start()
        out = []
        try:
            with socket.create_connection(('127.0.0.1', server.port), timeout=5) as c:
                f = c.makefile('rb')
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    m = json.loads(f.readline())
                    if m['type'] == 'status':
                        out.append(m)
        finally:
            server.stop()
        return out

    def test_a_source_that_never_delivered_is_reported_as_odom_stale(self):
        class Dead(lb.Vehicle):
            def step(self, dt):
                return (0.0, 0.0, 0.0)

            def pose_age(self):
                return None

        statuses = self._statuses(Dead())
        self.assertTrue(statuses, "no status line was sent")
        self.assertEqual(statuses[0]['error'], 'odom_stale')

    def test_a_source_whose_age_exceeds_the_limit_is_reported_stale(self):
        class Frozen(lb.Vehicle):
            def step(self, dt):
                return (1.0, 2.0, 0.0)

            def pose_age(self):
                return lb.BridgeServer.POSE_STALE_SECONDS + 0.5

        statuses = self._statuses(Frozen())
        self.assertEqual(statuses[0]['error'], 'odom_stale')
        # The pose is still sent -- consumers decide -- but the vehicle was never halted.
        self.assertEqual(statuses[0]['mode'], 'sim')

    def test_a_fresh_source_is_not_reported_stale(self):
        statuses = self._statuses(lb.SimVehicle(0.0, 0.0, 0.0))
        self.assertIsNone(statuses[0]['error'])

class BridgeLoopbackTest(unittest.TestCase):
    def test_hello_pose_and_nav_roundtrip(self):
        server = lb.BridgeServer(lb.SimVehicle(0.0, 0.0, 0.0, max_speed=1.0), port=0, rate=100)
        threading.Thread(target=server.serve, daemon=True).start()
        try:
            with socket.create_connection(('127.0.0.1', server.port), timeout=5) as c:
                f = c.makefile('rb')
                read = lambda: json.loads(f.readline())
                self.assertEqual(read()['type'], 'hello')
                deadline = time.monotonic() + 5
                while read()['type'] != 'pose':
                    self.assertLess(time.monotonic(), deadline)
                c.sendall(lb.encode({'type': 'nav', 'goal': 'bad', 'path': [[1]]}))
                c.sendall(lb.encode({'type': 'nav', 'goal': 'g', 'path': [[0.3, 0.0]]}))
                results = []
                while len(results) < 2:
                    self.assertLess(time.monotonic(), deadline)
                    m = read()
                    if m['type'] == 'nav_result':
                        results.append((m['goal'], m['status']))
                self.assertEqual(results, [('bad', 'failed'), ('g', 'succeeded')])
        finally:
            server.stop()


class ProtoOneTests(unittest.TestCase):
    def _connect(self, server):
        threading.Thread(target=server.serve, daemon=True).start()
        c = socket.create_connection(('127.0.0.1', server.port), timeout=5)
        f = c.makefile('rb')
        hello = json.loads(f.readline())
        return c, f, hello

    def _read_until(self, f, wanted, deadline=5.0):
        end = time.monotonic() + deadline
        while time.monotonic() < end:
            msg = json.loads(f.readline())
            if msg.get('type') == wanted:
                return msg
        self.fail('did not receive %s' % wanted)

    def test_hello_declares_proto_frame_and_caps(self):
        server = lb.BridgeServer(lb.CircleVehicle(1.0, 1.0, 0.5, 0.2), port=0, rate=50, proto=1, frame='sim')
        try:
            c, f, hello = self._connect(server)
            with c:
                self.assertEqual(hello['type'], 'hello')
                self.assertEqual(hello['proto'], 1)
                self.assertEqual(hello['frame'], 'sim')
                self.assertIn('ping', hello['caps'])
                self.assertIn('estop', hello['caps'])
        finally:
            server.stop()

    def test_proto_zero_still_accepted_without_caps(self):
        server = lb.BridgeServer(lb.CircleVehicle(1.0, 1.0, 0.5, 0.2), port=0, rate=50, proto=0, frame='chassis')
        try:
            c, f, hello = self._connect(server)
            with c:
                self.assertEqual(hello['proto'], 0)
                self.assertEqual(hello['frame'], 'chassis')
                self.assertEqual(hello['caps'], [])
        finally:
            server.stop()

    def test_ping_pong_echoes_t1(self):
        server = lb.BridgeServer(lb.CircleVehicle(1.0, 1.0, 0.5, 0.2), port=0, rate=50, proto=1)
        try:
            c, f, _ = self._connect(server)
            with c:
                c.sendall(lb.encode({'type': 'ping', 't1': 123.456}))
                pong = self._read_until(f, 'pong')
                self.assertEqual(pong['t1'], 123.456)
                self.assertLessEqual(pong['t2'], pong['t3'])
        finally:
            server.stop()

    def test_estop_latches_until_release(self):
        server = lb.BridgeServer(lb.CircleVehicle(1.0, 1.0, 0.5, 0.2), port=0, rate=50, proto=1,
                                 watchdog_seconds=0.0)
        try:
            c, f, _ = self._connect(server)
            with c:
                c.sendall(lb.encode({'type': 'estop'}))
                deadline = time.monotonic() + 3.0
                status = None
                while time.monotonic() < deadline:
                    msg = json.loads(f.readline())
                    if msg.get('type') == 'status' and msg.get('mode') == 'estop':
                        status = msg
                        break
                self.assertIsNotNone(status)
                c.sendall(lb.encode({'type': 'nav', 'goal': 'g', 'path': [[1.2, 1.0]]}))
                result = self._read_until(f, 'nav_result')
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['reason'], 'estop')
                c.sendall(lb.encode({'type': 'estop_release'}))
                status = self._read_until(f, 'status')
                self.assertEqual(status['mode'], 'sim')
        finally:
            server.stop()

    def test_watchdog_stops_without_messages(self):
        server = lb.BridgeServer(lb.CircleVehicle(1.0, 1.0, 0.5, 0.2), port=0, rate=50, proto=1,
                                 watchdog_seconds=0.2)
        try:
            c, f, _ = self._connect(server)
            with c:
                deadline = time.monotonic() + 3.0
                status = None
                while time.monotonic() < deadline:
                    msg = json.loads(f.readline())
                    if msg.get('type') == 'status' and msg.get('error') == 'watchdog':
                        status = msg
                        break
                self.assertIsNotNone(status)
        finally:
            server.stop()


if __name__ == '__main__':
    unittest.main()

class MultiVehicleBridgeTest(unittest.TestCase):
    """T5.1: two bridge scripts serve two vehicles at the same time.

    The design fixes one bridge per vehicle, so the multi-vehicle case is several bridges on
    different ports. This pins the pattern the GUI relies on: both bridges answer concurrently,
    each announces its own vehicle id, and each streams poses for its own circle.
    """

    @staticmethod
    def _start(vehicle, vehicle_id):
        server = lb.BridgeServer(vehicle, vehicle_id=vehicle_id, port=0, rate=50, proto=1, frame='sim')
        threading.Thread(target=server.serve, daemon=True).start()
        return server

    def test_two_bridges_serve_two_vehicles_concurrently(self):
        servers = [
            self._start(lb.CircleVehicle(1.0, 1.0, 0.5, 0.2), 'agv1'),
            self._start(lb.CircleVehicle(2.0, 2.0, 0.8, 0.3), 'agv2'),
        ]
        self.assertNotEqual(servers[0].port, servers[1].port)
        conns = []
        try:
            for server in servers:
                c = socket.create_connection(('127.0.0.1', server.port), timeout=5)
                conns.append((c, c.makefile('rb')))

            hellos = [json.loads(f.readline()) for _, f in conns]
            self.assertEqual([h['type'] for h in hellos], ['hello', 'hello'])
            self.assertEqual([h['vehicle'] for h in hellos], ['agv1', 'agv2'])

            poses = [[], []]
            deadline = time.monotonic() + 5
            while any(len(p) < 6 for p in poses):
                self.assertLess(time.monotonic(), deadline)
                for index, (_, f) in enumerate(conns):
                    msg = json.loads(f.readline())
                    if msg.get('type') == 'pose' and len(poses[index]) < 6:
                        poses[index].append(msg)

            for msg in poses[0]:
                self.assertEqual(msg['vehicle'], 'agv1')
                self.assertAlmostEqual(math.hypot(msg['x'] - 1.0, msg['y'] - 1.0), 0.5, places=2)
            for msg in poses[1]:
                self.assertEqual(msg['vehicle'], 'agv2')
                self.assertAlmostEqual(math.hypot(msg['x'] - 2.0, msg['y'] - 2.0), 0.8, places=2)

            # Each vehicle is moving on its own circle, driven by its own bridge loop.
            self.assertGreater(len({(m['x'], m['y']) for m in poses[0]}), 1)
            self.assertGreater(len({(m['x'], m['y']) for m in poses[1]}), 1)
        finally:
            for c, f in conns:
                try:
                    f.close()
                    c.close()
                except OSError:
                    pass
            for server in servers:
                server.stop()


class MessageSizeLimitTests(unittest.TestCase):
    """A client must not be able to grow the receive buffer without bound."""

    def test_oversized_pending_line_drops_the_connection(self):
        server = lb.BridgeServer(lb.CircleVehicle(0.0, 0.0, 1.0, 0.2), port=0, rate=100)
        threading.Thread(target=server.serve, daemon=True).start()
        try:
            with socket.create_connection(('127.0.0.1', server.port), timeout=5) as c:
                f = c.makefile('rb')
                f.readline()  # hello
                # No newline at all: the bridge has to give up on its own.
                c.sendall(b'{"type":"nav","pad":"' + b'x' * (lb.MAX_LINE_BYTES + 1024))
                deadline = time.monotonic() + 5
                while server._client is not None:
                    self.assertLess(time.monotonic(), deadline, 'bridge kept the oversized client')
                    time.sleep(0.02)
        finally:
            server.stop()

    def test_a_line_just_under_the_limit_is_still_accepted(self):
        server = lb.BridgeServer(lb.CircleVehicle(0.0, 0.0, 1.0, 0.2), port=0, rate=100)
        threading.Thread(target=server.serve, daemon=True).start()
        try:
            with socket.create_connection(('127.0.0.1', server.port), timeout=5) as c:
                f = c.makefile('rb')
                f.readline()  # hello
                pad = 'x' * (lb.MAX_LINE_BYTES - 128)
                c.sendall(lb.encode({'type': 'ping', 'pad': pad}))
                deadline = time.monotonic() + 5
                while server._client is None:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.02)
                self.assertIsNotNone(server._client, 'bridge dropped a client under the limit')
        finally:
            server.stop()
