"""Spatiotemporal reservation table: the data structure underneath conflict resolution.

Semantics
---------
* **Node reservation**: an AGV occupies a node during the single time step ``[t, t+1)``.
* **Edge reservation**: an AGV occupies an edge over ``[t, t+n)``, where n is the number
  of steps the traversal takes.
* Both node and edge capacity may exceed 1, which is what the "corridor load" experiment
  dimension varies.
* Intervals are **half-open** ``[start, end)``, so adjacent time steps do not conflict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Interval:
    """A half-open interval ``[start, end)``, **measured in whole steps**.

    Why integers are enforced (this constraint has a history)
    ---------------------------------------------------------
    The C# RAWSim-O reservation table stores time as double, with an **absolute**
    tolerance of ``TOLERANCE = 1e-12``.
    But the spacing between adjacent doubles -- the ULP -- grows with magnitude:

    ========= ============ ==================
    sim time  ULP          tolerance / ULP
    ========= ============ ==================
    4 500 s   9.09e-13     1.10
    12 062 s  1.82e-12     0.550   <- the point of collapse
    28 800 s  3.64e-12     0.275
    ========= ============ ==================

    The two checks then contradict each other. The "meaningless interval" check decides
    that an interval one ULP wide is not noise,
    while the "overlap" check finds it overlapping because
    ``start + TOLERANCE == start``, raises, and ends the simulation.
    The full analysis is in the crash and reproducibility write-up for the upstream CBS
    planner (internal record).

    **The Python side removes the problem at the root** by keeping time in whole steps
    (dt is adjustable, but the step index is an int).
    Integer arithmetic has no ULP problem, adjacency is decidable without any care, and no
    tolerance constant is needed at all.

    The cost is that the caller must convert seconds to steps first (``round(t / dt)``).
    That conversion happens once, at the entry point, so error never accumulates
    repeatedly inside the table.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        # Non-integers are rejected outright. A value like 2.0000000001 used to be accepted
        # silently and then truncated to 1 by len(), which hid the error -- Interval(1,
        # 2.0000000001) did in fact get through at one point.
        for name, val in (("start", self.start), ("end", self.end)):
            if isinstance(val, bool) or not isinstance(val, int):
                if isinstance(val, float) and float(val).is_integer():
                    # Let an integral float such as 3.0 through, but convert it, because a
                    # float must not be left sitting in the table.
                    object.__setattr__(self, name, int(val))
                    continue
                raise TypeError(
                    "Interval %s must be a whole number of steps, got %r (%s). "
                    "Floating-point times bring ULP-level tolerance problems; "
                    "convert with round(t / dt) first."
                    % (name, val, type(val).__name__)
                )
        if self.end < self.start:
            raise ValueError(f"invalid interval [{self.start}, {self.end})")

    def overlaps(self, other: "Interval") -> bool:
        return self.start < other.end and other.start < self.end

    def __len__(self) -> int:
        return self.end - self.start


class ReservationTable:
    """Records who occupies what, and when.

    Reservations on the same resource -- node or edge -- are kept sorted by start.
    Lookups are a linear scan. At the scale of a few hundred vehicles over a few thousand
    steps that is fast enough;
    an interval tree can replace it once the instances get larger.
    """

    def __init__(self) -> None:
        self._node: dict[str, list[tuple[Interval, str]]] = {}
        self._edge: dict[tuple[str, str], list[tuple[Interval, str]]] = {}
        # Actual occupancy survives ETA expiry and plan release/clear operations.
        self._pinned_nodes: dict[str, set[str]] = {}
        self._pinned_edges: dict[tuple[str, str], set[str]] = {}

    def transfer_pins(self, agv_id: str, nodes: Iterable[str],
                      edges: Iterable[tuple[str, str]] = ()) -> None:
        """Replace physical resource ownership only after confirmed movement."""
        for table in (self._pinned_nodes, self._pinned_edges):
            for owners in table.values():
                owners.discard(agv_id)
        for node in nodes:
            self._pinned_nodes.setdefault(node, set()).add(agv_id)
        for edge in edges:
            self._pinned_edges.setdefault(edge, set()).add(agv_id)

    def pins(self) -> dict:
        """Physical occupancy, separate from predicted time windows."""
        return {"nodes": {k: sorted(v) for k, v in self._pinned_nodes.items() if v},
                "edges": {"->".join(k): sorted(v)
                          for k, v in self._pinned_edges.items() if v}}

    def pinned_node_conflicts(self, node: str, exclude_agv: str | None = None) -> set[str]:
        """Owners with no predicted expiry, excluding the requesting robot."""
        return self._pin_owners(self._pinned_nodes, node, exclude_agv)

    def pinned_edge_conflicts(self, src: str, dst: str, exclude_agv: str | None = None) -> set[str]:
        """Physical edge owners; finite reservations are deliberately separate."""
        return self._pin_owners(self._pinned_edges, (src, dst), exclude_agv)

    @staticmethod
    def _pin_owners(table, key, exclude):
        return {o for o in table.get(key, ()) if o != exclude}

    def clear(self) -> None:
        self._node.clear()
        self._edge.clear()

    # ---------- queries ----------

    def _overlapping(
        self, entries: list[tuple[Interval, str]], span: Interval, exclude: str | None
    ) -> list[tuple[Interval, str]]:
        out = []
        for iv, owner in entries:
            if exclude is not None and owner == exclude:
                continue
            if iv.overlaps(span):
                out.append((iv, owner))
        return out

    def node_conflicts(
        self, node: str, span: Interval, exclude_agv: str | None = None
    ) -> list[str]:
        """Return the ids of AGVs that conflict with ``span``, without duplicates."""
        entries = self._node.get(node, [])
        return sorted({owner for _, owner in self._overlapping(entries, span, exclude_agv)}
                      | self._pin_owners(self._pinned_nodes, node, exclude_agv))

    def edge_conflicts(
        self, src: str, dst: str, span: Interval, exclude_agv: str | None = None
    ) -> list[str]:
        entries = self._edge.get((src, dst), [])
        return sorted({owner for _, owner in self._overlapping(entries, span, exclude_agv)}
                      | self._pin_owners(self._pinned_edges, (src, dst), exclude_agv))

    def is_node_free(
        self, node: str, span: Interval, capacity: int, exclude_agv: str | None = None
    ) -> bool:
        entries = self._node.get(node, [])
        hits = self._overlapping(entries, span, exclude_agv)
        pins = self._pin_owners(self._pinned_nodes, node, exclude_agv)
        return len(hits) + len(pins - {o for _, o in hits}) < capacity

    def is_edge_free(
        self, src: str, dst: str, span: Interval, capacity: int, exclude_agv: str | None = None
    ) -> bool:
        entries = self._edge.get((src, dst), [])
        hits = self._overlapping(entries, span, exclude_agv)
        pins = self._pin_owners(self._pinned_edges, (src, dst), exclude_agv)
        return len(hits) + len(pins - {o for _, o in hits}) < capacity

    # ---------- writes ----------

    def reserve_node(self, node: str, span: Interval, agv_id: str) -> None:
        if len(span) == 0:
            return
        bucket = self._node.setdefault(node, [])
        bucket.append((span, agv_id))
        bucket.sort(key=lambda x: x[0].start)

    def reserve_edge(self, src: str, dst: str, span: Interval, agv_id: str) -> None:
        if len(span) == 0:
            return
        bucket = self._edge.setdefault((src, dst), [])
        bucket.append((span, agv_id))
        bucket.sort(key=lambda x: x[0].start)

    def release_edge(
        self, src: str, dst: str, span: Interval, agv_id: str
    ) -> None:
        """Release one vehicle's reservation on one edge over one window.

        Used to migrate a reservation when a vehicle departs early.
        """
        bucket = self._edge.get((src, dst))
        if not bucket:
            return
        self._edge[(src, dst)] = [
            (iv, o) for iv, o in bucket if not (o == agv_id and iv == span)
        ]

    def release_node(
        self, node: str, span: Interval, agv_id: str
    ) -> None:
        bucket = self._node.get(node)
        if not bucket:
            return
        self._node[node] = [
            (iv, o) for iv, o in bucket if not (o == agv_id and iv == span)
        ]

    def release_all(self, agv_id: str) -> None:
        """Release every reservation held by one vehicle, as a replan does."""
        for table in (self._node, self._edge):
            for key in list(table):
                table[key] = [(iv, o) for iv, o in table[key] if o != agv_id]

    def purge_before(self, step: int) -> None:
        """Drop reservations that ended before `step`, to bound memory growth."""
        for table in (self._node, self._edge):
            for key in list(table):
                kept = [(iv, o) for iv, o in table[key] if iv.end > step]
                if kept:
                    table[key] = kept
                else:
                    del table[key]

    # ---------- statistics ----------

    def snapshot(self, lo: int, hi: int) -> dict[str, list[tuple[int, int, str]]]:
        """Node occupancy over [lo, hi), as a snapshot for visualisation.

        Only **node** occupancy is returned. Edge occupancy adds little to understanding a
        conflict,
        whereas nodes -- stations, buffers, intersections -- are where vehicles actually
        queue.
        Returns ``{node_id: [(start, end, agv_id), ...]}``.
        """
        span = Interval(lo, hi)
        out: dict[str, list[tuple[int, int, str]]] = {}
        for node, entries in self._node.items():
            hits = [(iv.start, iv.end, owner) for iv, owner in entries
                    if iv.overlaps(span)]
            if hits:
                out[node] = sorted(hits)
        for node, owners in self._pinned_nodes.items():
            for owner in owners:
                out.setdefault(node, []).append((lo, hi, owner))
        return out

    def edge_snapshot(self, lo: int, hi: int) -> dict[str, list[tuple[int, int, str]]]:
        """Undirected edge occupancy over [lo, hi), keyed as "a|b" in sorted order."""
        span = Interval(lo, hi)
        out: dict[str, list[tuple[int, int, str]]] = {}
        for (a, b), entries in self._edge.items():
            key = "|".join(sorted((a, b)))
            hits = [(iv.start, iv.end, owner) for iv, owner in entries
                    if iv.overlaps(span)]
            if hits:
                out.setdefault(key, []).extend(sorted(hits))
        for (a, b), owners in self._pinned_edges.items():
            key = "|".join(sorted((a, b)))
            for owner in owners:
                out.setdefault(key, []).append((lo, hi, owner))
        return out
    def total_reservations(self) -> int:
        return sum(len(v) for v in self._node.values()) + sum(
            len(v) for v in self._edge.values()
        )

    def utilization(self, horizon: int, n_nodes: int, n_edges: int) -> float:
        """Resource occupancy, a coarse-grained metric used for mechanism analysis."""
        if horizon <= 0 or (n_nodes + n_edges) == 0:
            return 0.0
        occupied = sum(len(iv) for v, _ in self._node.values()) + sum(
            len(iv) for v, _ in self._edge.values()
        )
        return occupied / (horizon * (n_nodes + n_edges))
