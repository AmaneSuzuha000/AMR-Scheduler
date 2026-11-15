"""Tests for the console's route service and its wiring into the bridge.

The point of these tests is not that A* works - the planner has its own suite. The point is
that the *decision* stays in Python: the console asks over the wire and draws the answer,
and the answer is the scheduler's own route, not something recomputed in C#.
"""
import json
import os
import socket
import sys
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import live_bridge as lb
from route_service import RouteService


class RouteServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = RouteService()

    def test_loads_the_same_scene_as_the_scheduler(self):
        self.assertEqual(self.svc.scene_id, 'rail_freight_workshop_v1')
        self.assertEqual(len(self.svc.node_ids()), 62)

    def test_route_matches_the_scheduler_planner(self):
        # The drive script plans J_S_8 -> J_S_125 with StaticDijkstraPlanner and gets these
        # seven nodes over 117.0 m. The console must get the identical answer: one planner,
        # one scene, one answer.
        r = self.svc.plan('J_S_125', from_x=8.0, from_y=30.0)
        self.assertIsNone(r['reason'])
        self.assertEqual(r['start'], 'J_S_8')
        self.assertEqual(r['nodes'],
                         ['J_S_8', 'J_S_25', 'J_S_45', 'J_S_65', 'J_S_85', 'J_S_105', 'J_S_125'])
        self.assertAlmostEqual(r['meters'], 117.0, places=3)

    def test_start_is_snapped_from_a_pose_off_the_network(self):
        # A pose half a metre off the line still snaps to the nearest node rather than failing.
        r = self.svc.plan('J_S_125', from_x=8.4, from_y=30.5)
        self.assertEqual(r['start'], 'J_S_8')

    def test_map_path_carries_scene_coordinates_in_order(self):
        r = self.svc.plan('J_S_125', from_x=8.0, from_y=30.0)
        self.assertEqual(r['map_path'][0], [8.0, 30.0])
        self.assertEqual(r['map_path'][-1], [125.0, 30.0])
        self.assertEqual(len(r['map_path']), len(r['nodes']))

    def test_accepts_the_rawsimo_waypoint_number_the_console_sends(self):
        # The console only knows RAWSim-O's integer waypoint ids. Waypoint 7 is J_S_125 in
        # scene order. Before this translation existed the console asked for "7" and Python
        # answered "unknown node '7'" - the same map under two different names.
        self.assertEqual(self.svc.resolve_target('7'), 'J_S_125')
        r = self.svc.plan('7', from_x=8.0, from_y=30.0)
        self.assertIsNone(r['reason'])
        self.assertEqual(r['nodes'][-1], 'J_S_125')

    def test_accepts_a_scene_node_id_unchanged(self):
        self.assertEqual(self.svc.resolve_target('J_S_125'), 'J_S_125')

    def test_waypoint_numbering_comes_from_the_exporter(self):
        # Both sides must use one numbering rule. The exporter owns it; this pins the rule
        # (index+1 over the scene node order) instead of letting each side re-derive it.
        import sys as _sys
        _sys.path.insert(0, os.path.join(ROOT, 'tools'))
        from scene_to_rawsimo import waypoint_ids, node_ids_by_waypoint
        scene = self.svc._scene
        ids = waypoint_ids(scene.raw)
        self.assertEqual(ids['J_S_8'], 1)
        self.assertEqual(ids['J_S_125'], 7)
        self.assertEqual(node_ids_by_waypoint(scene.raw)[7], 'J_S_125')

    def test_refuses_without_a_position_instead_of_inventing_a_one_point_route(self):
        # There used to be a silent fallback: with no "from", the start was assumed to be the
        # target, which produced a one-point "route" that looked like a successful answer. The
        # console then drew nothing and said nothing. A missing position must be a refusal.
        r = self.svc.plan('J_S_125')
        self.assertIsNotNone(r['reason'])
        self.assertIn('no current position', r['reason'])
        self.assertEqual(r['map_path'], [])

    def test_unknown_target_reports_a_reason_instead_of_raising(self):
        r = self.svc.plan('NOT_A_NODE')
        self.assertEqual(r['nodes'], [])
        self.assertIn('unknown node', r['reason'])

    def test_unreachable_target_reports_a_reason(self):
        # Plan from a node to itself succeeds; a genuinely absent target must not.
        r = self.svc.plan('J_S_125', start='NO_SUCH_START')
        self.assertIsNotNone(r['reason'])


class RouteRequestOverTheWireTests(unittest.TestCase):
    def test_bridge_answers_route_request(self):
        server = lb.BridgeServer(lb.CircleVehicle(1.0, 1.0, 0.5, 0.2), port=0, rate=20)
        threading.Thread(target=server.serve, daemon=True).start()
        try:
            with socket.create_connection(('127.0.0.1', server.port), timeout=5) as c:
                f = c.makefile('rb')
                self.assertEqual(json.loads(f.readline())['type'], 'hello')
                c.sendall(lb.encode({'type': 'route_request', 'target': 'J_S_125',
                                     'from': [8.0, 30.0]}))
                deadline = time.monotonic() + 10
                while True:
                    self.assertLess(time.monotonic(), deadline)
                    m = json.loads(f.readline())
                    if m.get('type') == 'route_result':
                        break
                self.assertIsNone(m['reason'])
                self.assertEqual(m['nodes'][0], 'J_S_8')
                self.assertEqual(m['nodes'][-1], 'J_S_125')
                self.assertAlmostEqual(m['meters'], 117.0, places=3)
        finally:
            server.stop()

    def test_bridge_answers_a_route_deviation_with_a_decision(self):
        # The execution layer reports; Python decides. The bridge must answer, because a
        # sender waiting for a decision that never comes has to guess -- which is exactly
        # the second decision maker this architecture forbids.
        server = lb.BridgeServer(lb.CircleVehicle(1.0, 1.0, 0.5, 0.2), port=0, rate=20)
        threading.Thread(target=server.serve, daemon=True).start()
        try:
            with socket.create_connection(('127.0.0.1', server.port), timeout=5) as c:
                f = c.makefile('rb')
                self.assertEqual(json.loads(f.readline())['type'], 'hello')
                c.sendall(lb.encode({'type': 'route_deviation', 'robot_id': 'AGV_01',
                                     'event': 'ROUTE_DEVIATION',
                                     'blocked_edge': 'J_S_45-J_S_65',
                                     'current_pose': [8.0, 30.0, 0.0],
                                     'timestamp': 10.0,
                                     'goal': 'J_S_125'}))
                deadline = time.monotonic() + 10
                while True:
                    self.assertLess(time.monotonic(), deadline)
                    m = json.loads(f.readline())
                    if m.get('type') == 'route_decision':
                        break
                self.assertEqual(m['action'], 'reroute')
                self.assertGreater(m['meters'], 117.0)
                blocked = set(m['blocked_edge'])
                nodes = m['nodes']
                self.assertGreaterEqual(len(nodes), 2)
                for a, b in zip(nodes, nodes[1:]):
                    self.assertNotEqual({a, b}, blocked, '绕行仍然用了受阻通道')
        finally:
            server.stop()

    def test_bridge_rejects_a_malformed_deviation_instead_of_staying_silent(self):
        server = lb.BridgeServer(lb.CircleVehicle(1.0, 1.0, 0.5, 0.2), port=0, rate=20)
        threading.Thread(target=server.serve, daemon=True).start()
        try:
            with socket.create_connection(('127.0.0.1', server.port), timeout=5) as c:
                f = c.makefile('rb')
                self.assertEqual(json.loads(f.readline())['type'], 'hello')
                c.sendall(lb.encode({'type': 'route_deviation', 'robot_id': 'AGV_01'}))
                deadline = time.monotonic() + 10
                while True:
                    self.assertLess(time.monotonic(), deadline)
                    m = json.loads(f.readline())
                    if m.get('type') == 'route_decision':
                        break
                self.assertEqual(m['action'], 'rejected')
                self.assertIn('blocked_edge', m['reason'])
        finally:
            server.stop()

    def test_hello_advertises_route_and_the_scene_id(self):
        # The console decides from the caps list whether it may ask Python to plan, and it
        # must be able to check that both sides loaded the same map. Neither was in hello,
        # so a working capability looked absent.
        server = lb.BridgeServer(lb.CircleVehicle(1.0, 1.0, 0.5, 0.2), port=0, rate=20)
        threading.Thread(target=server.serve, daemon=True).start()
        try:
            with socket.create_connection(('127.0.0.1', server.port), timeout=5) as c:
                hello = json.loads(c.makefile('rb').readline())
                self.assertEqual(hello['type'], 'hello')
                self.assertIn('route', hello['caps'])
                self.assertEqual(hello.get('scene'), 'rail_freight_workshop_v1')
        finally:
            server.stop()

    def test_bridge_answers_even_for_a_bad_target(self):
        server = lb.BridgeServer(lb.CircleVehicle(1.0, 1.0, 0.5, 0.2), port=0, rate=20)
        threading.Thread(target=server.serve, daemon=True).start()
        try:
            with socket.create_connection(('127.0.0.1', server.port), timeout=5) as c:
                f = c.makefile('rb')
                self.assertEqual(json.loads(f.readline())['type'], 'hello')
                c.sendall(lb.encode({'type': 'route_request', 'target': 'NOPE'}))
                deadline = time.monotonic() + 10
                while True:
                    self.assertLess(time.monotonic(), deadline)
                    m = json.loads(f.readline())
                    if m.get('type') == 'route_result':
                        break
                self.assertEqual(m['nodes'], [])
                self.assertTrue(m['reason'])
        finally:
            server.stop()


if __name__ == '__main__':
    unittest.main()
