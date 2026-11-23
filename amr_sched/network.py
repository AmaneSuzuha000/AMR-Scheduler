"""Node/edge network model.

Models the workshop's transport lanes and stations as a graph: nodes are junctions,
workstations, and buffer slots; edges are lane segments.

Design constraints
--------
* **It is a directed graph.** Maintenance workshops often have one-way lanes, and a
  two-way lane is written as two edges,
  which is less error-prone than special-casing "may it turn around" inside the
  algorithms.
* **Lengths are metres, speeds are metres per second, times are seconds.**
* Edges carry a capacity (how many vehicles may occupy the edge at once), which is the
  "lane load" dimension of the experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class NodeKind(str, Enum):
    """Node kind. Corresponds to the workshop's functional zones in the system specification."""

    AISLE = "aisle"                 # ordinary lane node / junction
    DISASSEMBLY = "disassembly"     # disassembly area
    WORKSTATION = "workstation"     # maintenance workstation
    ASSEMBLY = "assembly"           # assembly area
    BUFFER = "buffer"               # buffer area
    DOCK = "dock"                   # loading point


@dataclass(frozen=True)
class Node:
    """A network node.

    Attributes
    ----------
    node_id:
        Unique identifier. Strings by convention, so JSON/YAML reads straight in.
    x, y:
        Planar coordinates in metres. Used only for rendering and distance estimates;
        never for reachability.
    kind:
        Functional kind; see :class:`NodeKind`.
    capacity:
        How many AGVs may sit here at once. Workstations and buffer slots are usually 1;
        lane nodes may be larger.
    """

    node_id: str
    x: float
    y: float
    kind: NodeKind = NodeKind.AISLE
    capacity: int = 1

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"node {self.node_id}: capacity must be >= 1, got {self.capacity}")


@dataclass(frozen=True)
class Edge:
    """A directed edge.

    Attributes
    ----------
    src, dst:
        Endpoint node ids.
    length:
        Length in metres; must be > 0.
    speed_limit:
        Speed limit on this lane (m/s); must be > 0.
    capacity:
        How many AGVs may be on this edge at once. ``1`` means an exclusive section.
    """

    src: str
    dst: str
    length: float
    speed_limit: float = 1.0
    capacity: int = 1
    # Empty means the legacy straight edge; explicit geometry is immutable.
    centerline: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError(f"edge {self.src}->{self.dst}: length must be > 0, got {self.length}")
        if self.speed_limit <= 0:
            raise ValueError(f"edge {self.src}->{self.dst}: speed_limit must be > 0")
        if self.capacity < 1:
            raise ValueError(f"edge {self.src}->{self.dst}: capacity must be >= 1")
        if self.src == self.dst:
            raise ValueError(f"self-loops are not allowed: {self.src}")
        if self.centerline:
            points = tuple(tuple(float(v) for v in p) for p in self.centerline)
            if (len(points) < 2 or any(len(p) != 2 or any(not math.isfinite(v) for v in p) for p in points)
                    or any(a == b for a,b in zip(points,points[1:]))):
                raise ValueError('finite nondegenerate centerline required')
            if not math.isclose(sum(math.dist(a,b) for a,b in zip(points,points[1:])), self.length,
                                rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError('centerline length differs from declared edge length')
            object.__setattr__(self, 'centerline', points)

    def traversal_time(self, max_speed: float) -> float:
        """Time in seconds to traverse this edge at the given maximum speed."""
        v = min(self.speed_limit, max_speed)
        return self.length / v


class Network:
    """The workshop transport network.

Owns graph storage, connectivity checks, and shortest-path distances.
Deliberately does **not** handle space-time reservations; those belong to the planners.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[tuple[str, str], Edge] = {}
        self._out: dict[str, list[Edge]] = {}
        self._in: dict[str, list[Edge]] = {}
        self._tt_cache: dict[tuple[str, float], dict[str, float]] = {}

    # ---------- Construction ----------

    def add_node(self, node: Node) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"duplicate node id: {node.node_id}")
        self._tt_cache.clear()
        self._nodes[node.node_id] = node
        self._out[node.node_id] = []
        self._in[node.node_id] = []

    def add_edge(self, edge: Edge) -> None:
        for n in (edge.src, edge.dst):
            if n not in self._nodes:
                raise ValueError(f"edge references a node that does not exist: {n}")
        key = (edge.src, edge.dst)
        if key in self._edges:
            raise ValueError(f"duplicate edge: {edge.src}->{edge.dst}")
        if edge.centerline:
            a,b = self.node(edge.src), self.node(edge.dst)
            if (math.dist(edge.centerline[0], (a.x,a.y)) > 1e-6
                    or math.dist(edge.centerline[-1], (b.x,b.y)) > 1e-6):
                raise ValueError('centerline endpoints differ from network nodes')
        self._tt_cache.clear()
        self._edges[key] = edge
        self._out[edge.src].append(edge)
        self._in[edge.dst].append(edge)

    def add_bidirectional(self, a: str, b: str, length: float, **kwargs) -> None:
        """Add a pair of opposing edges. Used for two-way lanes so the algorithms never
        have to special-case turning around."""
        self.add_edge(Edge(a, b, length, **kwargs))
        if kwargs.get('centerline'):
            kwargs = {**kwargs, 'centerline': tuple(reversed(kwargs['centerline']))}
        self.add_edge(Edge(b, a, length, **kwargs))

    # ---------- Queries ----------

    @property
    def nodes(self) -> dict[str, Node]:
        return self._nodes

    @property
    def edges(self) -> dict[tuple[str, str], Edge]:
        return self._edges

    def node(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def edge(self, src: str, dst: str) -> Edge:
        return self._edges[(src, dst)]

    def edge_centerline(self, src: str, dst: str) -> tuple[tuple[float, float], ...]:
        """Authoritative directed geometry; legacy edges join their two nodes."""
        edge = self.edge(src,dst)
        a,b = self.node(src),self.node(dst)
        return edge.centerline or ((a.x,a.y),(b.x,b.y))

    def route_centerline(self, nodes) -> tuple[tuple[float, float], ...]:
        """Expand connected graph hops without omitting intermediate vertices."""
        nodes = list(nodes)
        if not nodes:
            raise ValueError('route must contain at least one node')
        start = self.node(nodes[0])
        points = [(start.x,start.y)]
        for src,dst in zip(nodes,nodes[1:]):
            points.extend(self.edge_centerline(src,dst)[1:])
        return tuple(points)

    def out_edges(self, node_id: str) -> list[Edge]:
        return self._out[node_id]

    def in_edges(self, node_id: str) -> list[Edge]:
        return self._in[node_id]

    def nodes_of_kind(self, *kinds: NodeKind) -> list[Node]:
        want = set(kinds)
        return [n for n in self._nodes.values() if n.kind in want]

    def has_edge(self, src: str, dst: str) -> bool:
        return (src, dst) in self._edges

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    # ---------- Distances ----------

    def shortest_travel_times(
        self, source: str, max_speed: float
    ) -> dict[str, float]:
        """Shortest travel time from ``source`` to every node (Dijkstra).

The returned mapping contains only **reachable** nodes; unreachable ones are absent.
        """
        if source not in self._nodes:
            raise KeyError(f"no such source node: {source}")
        cache_key = (source, float(max_speed))
        cached = self._tt_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        INF = math.inf
        dist: dict[str, float] = {source: 0.0}
        visited: set[str] = set()
        # A plain binary heap: the node count is small and this avoids a dependency.
        import heapq

        heap: list[tuple[float, str]] = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            for e in self._out[u]:
                nd = d + e.traversal_time(max_speed)
                if nd < dist.get(e.dst, INF):
                    dist[e.dst] = nd
                    heapq.heappush(heap, (nd, e.dst))
        # Note: **includes the source itself** at distance 0.
        # An earlier version dropped the source, which made the most convenient case --
        # a vehicle already parked at the pick-up point --
        # look up as inf, so the whole pairing was discarded. Measured effect: two
        # vehicles sitting at the pick-up point
        # could not be given a single task. Callers therefore need no separate a == b test.
        self._tt_cache[cache_key] = dist
        return dict(dist)

    def unreachable_pairs(self, max_speed: float) -> list[tuple[str, str]]:
        """Every unreachable (source, target) pair.

Used to validate a scenario on load, rather than discovering mid-run that some task can
never be completed.
        """
        bad: list[tuple[str, str]] = []
        for src in self._nodes:
            reachable = self.shortest_travel_times(src, max_speed)
            for dst in self._nodes:
                if dst == src:
                    continue
                if dst not in reachable:
                    bad.append((src, dst))
        return bad


def build_demo_network(n_workstations: int = 6,
                       aisle_capacity: int = 2,
                       one_way_ring: bool = False) -> Network:
    r"""Build a maintenance-workshop layout that contains a **loop**, for comparisons and
    smoke tests.

    Why the loop is not optional
    --------------
If the network were a tree, any two points would have exactly one path between them,
every planner would return that same route,

the benefit of "conflict resolution" would be identically zero, and the baseline
comparison would mean nothing.

Real maintenance workshops really are laid out as a **loop main lane plus branch
workstations**,

so a loop is built here, making "go clockwise" versus "go anticlockwise" a genuine
choice.


    Layout (schematic)::

        D(disassembly)               S(assembly)
            \                          /
             +---- R0 ---- R1 ---- R2 --+
             |      |       |       |    |
             |     W1      W2      W3    |
             |      |       |       |    |
             +---- R3 ---- R4 ---- R5 --+
                   |               |
                  W4              W5
                  B1(buffer)

    * The main loop R0..R5 is two-way, with capacity set by ``aisle_capacity``.
    * Workstations W1..W6 hang off the loop with capacity 1 (a single slot each).

    Parameters
    ----------
    n_workstations:
        How many workstations to hang (1..6).
    aisle_capacity:
        Capacity of the loop nodes and edges. **Defaults to 2 (two lanes).** Setting it
        to 1 gives a single-lane loop,

        where any two opposing vehicles can deadlock -- the extreme case for testing
        conflict resolution.

        The corridor load scenario is built from this together with the
        task arrival density.

    one_way_ring:
        Whether to force the loop to run **one way** (keeping only the clockwise
        direction).


        This is the classic prevention for single-lane deadlock: with every vehicle on
        the loop heading the same way,

        a circular wait -- "I wait for you, you wait for me" -- **cannot** form, and a
        circular wait is exactly

        what causes single-lane deadlock. The cost is detours, at worst most of the loop.


        Note that it constrains only the **main loop**; the workstation and buffer
        branches stay two-way

        (a branch is a dead end, forms no cycle, and so cannot produce a circular wait).

    """

    net = Network()

    # ---- Main loop: six ring nodes forming a rectangular circuit ----
    ring = [("R0", 0.0, 0.0), ("R1", 40.0, 0.0), ("R2", 80.0, 0.0),
            ("R3", 0.0, 40.0), ("R4", 40.0, 40.0), ("R5", 80.0, 40.0)]
    for nid, x, y in ring:
        net.add_node(Node(nid, x, y, NodeKind.AISLE, capacity=aisle_capacity))

    def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    coords = {nid: (x, y) for nid, x, y in ring}

    # Outer ring: R0-R1-R2-R5-R4-R3-R0
    outer = ["R0", "R1", "R2", "R5", "R4", "R3"]
    for i in range(len(outer)):
        a, b = outer[i], outer[(i + 1) % len(outer)]
        length = dist(coords[a], coords[b])
        if one_way_ring:
            net.add_edge(Edge(a, b, length, speed_limit=1.0,
                              capacity=aisle_capacity))
        else:
            net.add_bidirectional(a, b, length, speed_limit=1.0,
                                  capacity=aisle_capacity)

    # Middle beam: R1-R4, cutting the circuit into two smaller loops so there is more
    # than one route to choose from
    # When running one way the beam may only go R1 -> R4; otherwise it becomes a
    # shortcut for anticlockwise travel

    # and "one way" would exist in name only.
    if one_way_ring:
        net.add_edge(Edge("R1", "R4", 40.0, speed_limit=1.0,
                          capacity=aisle_capacity))
    else:
        net.add_bidirectional("R1", "R4", 40.0, speed_limit=1.0,
                              capacity=aisle_capacity)

    # ---- Functional zones ----
    net.add_node(Node("D", -25.0, 20.0, NodeKind.DISASSEMBLY, capacity=2))
    net.add_node(Node("S", 105.0, 20.0, NodeKind.ASSEMBLY, capacity=2))
    net.add_bidirectional("D", "R0", dist((-25.0, 20.0), coords["R0"]), speed_limit=1.0)
    net.add_bidirectional("S", "R2", dist((105.0, 20.0), coords["R2"]), speed_limit=1.0)

    # ---- Workstations: hanging off ring nodes ----
    station_specs = [
        ("W1", "R0", 10.0, -20.0), ("W2", "R1", 40.0, -20.0),
        ("W3", "R2", 70.0, -20.0), ("W4", "R3", 10.0, 60.0),
        ("W5", "R5", 70.0, 60.0), ("W6", "R3", -20.0, 55.0),
    ]
    for i, (wid, hook, x, y) in enumerate(station_specs[:n_workstations]):
        net.add_node(Node(wid, x, y, NodeKind.WORKSTATION, capacity=1))
        net.add_bidirectional(wid, hook, dist((x, y), coords[hook]), speed_limit=1.0)

    # ---- Buffer areas: hanging off ring nodes ----
    buffer_specs = [("B1", "R4", 40.0, 70.0), ("B2", "R5", 95.0, 55.0), ("B3", "R0", -15.0, -10.0)]
    for bid, hook, x, y in buffer_specs:
        net.add_node(Node(bid, x, y, NodeKind.BUFFER, capacity=2))
        net.add_bidirectional(bid, hook, dist((x, y), coords[hook]), speed_limit=1.0)

    return net
