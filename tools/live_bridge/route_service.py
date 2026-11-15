"""Answer route requests from the RAWSim-O console with the scheduler planner.

Why this exists
---------------
The console used to build its own route with a C# A* (RAWSimO.Live/Planning/AStarPlanner.cs)
whenever the operator picked a target waypoint and pressed execute. That put a second
planner inside the display layer, which contradicts the agreed split: Python owns task
allocation, global routing and reservation, and the console only draws what it is told.

This module is the Python side of that split. It answers one question - "from where the
vehicle is now, what is the node sequence to this target?" - using the *same* planner the
scheduler uses (amr_sched.planners.static_dijkstra.StaticDijkstraPlanner) and the *same*
scene file (scenes/rail_freight_workshop_v1.json). Nothing here invents coordinates.

Deliberate limits
-----------------
* Reservations are empty. This is the single-vehicle display path; multi-vehicle
  reservation and conflict resolution live in the scheduler, not here.
* If the scene cannot be loaded the caller gets a "reason" instead of an exception, so the
  console can show the failure rather than silently drawing nothing.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools", "live_bridge")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class RouteService:
    """Plan over the workshop scene on demand. Loads the scene once, lazily."""

    def __init__(self, scene_path=None, max_speed=1.0, dt=1.0):
        self.scene_path = scene_path
        self.max_speed = max_speed
        self.dt = dt
        self._scene = None
        self._network = None
        self._error = None
        self._ids = None

    def _ensure(self):
        if self._scene is not None or self._error is not None:
            return
        try:
            from amr_sched.workshop import WorkshopScene

            self._scene = (WorkshopScene.load(self.scene_path)
                           if self.scene_path else WorkshopScene.load())
            self._network = self._scene.network()
        except Exception as exc:  # noqa: BLE001 - reported to the console, never raised
            self._error = "%s: %s" % (type(exc).__name__, exc)

    @property
    def scene_id(self):
        self._ensure()
        return getattr(self._scene, "scene_id", "") if self._scene else ""

    def node_ids(self):
        self._ensure()
        return sorted(self._scene.nodes) if self._scene else []

    def network(self):
        """The loaded network, for callers that need more than a route (the monitor)."""
        self._ensure()
        return self._network

    def nearest_node(self, x, y):
        """Closest node to a map-frame point. A linear scan: 62 nodes, one call per request."""
        self._ensure()
        if self._scene is None:
            return None
        best, best_d = None, None
        for nid, node in self._scene.nodes.items():
            d = (node.x - x) ** 2 + (node.y - y) ** 2
            if best_d is None or d < best_d:
                best, best_d = nid, d
        return best

    def resolve_target(self, target):
        """Accepts either a scene node id ("J_S_125") or a RAWSim-O waypoint id (7).

        The console knows waypoints only as the integers RAWSim-O assigns (index+1 over
        the scene node order), while the planner and the scene speak node ids. Returning a
        single canonical id here keeps that translation in one place: without it the console
        asked for waypoint 7 and Python answered "unknown node '7'".
        """
        self._ensure()
        if self._scene is None:
            return target
        if target in self._scene.nodes:
            return target
        try:
            index = int(str(target).strip())
        except (TypeError, ValueError):
            return target
        for node_id, waypoint_index in self._waypoint_ids().items():
            if waypoint_index == index:
                return node_id
        return target

    def _waypoint_ids(self):
        """RAWSim-O waypoint numbering, taken from the exporter so it cannot drift."""
        if self._ids is None:
            self._ids = {}
            try:
                sys.path.insert(0, os.path.join(ROOT, "tools"))
                from scene_to_rawsimo import waypoint_ids
                self._ids = waypoint_ids(self._scene.raw)
            except Exception:  # noqa: BLE001 - fall back to the same rule inline
                self._ids = {node_id: i + 1 for i, node_id in enumerate(self._scene.nodes)}
        return self._ids
    def plan(self, target, from_x=None, from_y=None, start=None, direct=False):
        """Return the route to target as a plain dict. Never raises."""
        self._ensure()
        if self._scene is None:
            return {"type": "route_result", "target": target, "nodes": [],
                    "map_path": [], "reason": "scene unavailable (%s)" % self._error}

        target = self.resolve_target(target)
        if target not in self._scene.nodes:
            return {"type": "route_result", "target": target, "nodes": [],
                    "map_path": [], "reason": "unknown node %r" % (target,)}

        # "Direct" is the operator's explicit request to drive straight at the target rather
        # than along the network. It is still answered here, so the console has no branch of
        # its own that could disagree with the routed case.
        if direct:
            node = self._scene.nodes[target]
            path = [[round(node.x, 4), round(node.y, 4)]]
            if from_x is not None and from_y is not None:
                path.insert(0, [round(float(from_x), 4), round(float(from_y), 4)])
            if len(path) < 2:
                return {"type": "route_result", "target": target, "nodes": [],
                        "map_path": [], "reason": "direct needs a current position"}
            meters = sum(
                ((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2) ** 0.5
                for i in range(1, len(path)))
            return {"type": "route_result", "target": target, "start": None,
                    "nodes": [], "map_path": path, "meters": round(meters, 3),
                    "seconds": round(meters / max(self.max_speed, 1e-6), 2), "reason": None}

        # The start is wherever the vehicle actually is, snapped to the network. There used to
        # be a silent fallback to the target when no position was supplied, which produced a
        # one-point "route" that looked like a successful answer so the console drew nothing
        # and reported nothing. A missing position is a refusal, not a route.
        if from_x is None or from_y is None:
            if start is None:
                return {"type": "route_result", "target": target, "nodes": [],
                        "map_path": [], "reason": "no current position (waiting for a pose)"}
        else:
            start = self.nearest_node(float(from_x), float(from_y))
        if start is None:
            return {"type": "route_result", "target": target, "nodes": [],
                    "map_path": [], "reason": "no start node"}

        try:
            from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
            from amr_sched.reservation import ReservationTable

            planner = StaticDijkstraPlanner(max_speed=self.max_speed, dt=self.dt)
            planned = planner.plan(
                network=self._network, start=start, goal=target, start_step=0,
                reservations=ReservationTable(), agv_id="console", horizon=20000,
            )
        except Exception as exc:  # noqa: BLE001
            return {"type": "route_result", "target": target, "nodes": [],
                    "map_path": [], "reason": "%s: %s" % (type(exc).__name__, exc)}

        if planned is None:
            return {"type": "route_result", "target": target, "nodes": [],
                    "map_path": [], "reason": "no route %s -> %s" % (start, target)}

        nodes = [start] + list(planned.nodes)
        path = []
        meters = 0.0
        for i, nid in enumerate(nodes):
            node = self._scene.nodes[nid]
            path.append([round(node.x, 4), round(node.y, 4)])
            if i:
                prev = self._scene.nodes[nodes[i - 1]]
                meters += ((node.x - prev.x) ** 2 + (node.y - prev.y) ** 2) ** 0.5
        return {
            "type": "route_result",
            # Echo the canonical scene node id *and* the RAWSim-O waypoint number, so the
            # console can match the answer to the request it made without guessing which
            # numbering the server replied in.
            "target": target,
            "waypoint": self._waypoint_ids().get(target),
            "start": start,
            "nodes": nodes,
            "map_path": path,
            "meters": round(meters, 3),
            "seconds": round(meters / max(self.max_speed, 1e-6), 2),
            "reason": None,
        }
