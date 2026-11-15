"""Native Rust-accelerated Spatiotemporal A* Path Planner.

Drop-in replacement for `SpaceTimeAStarPlanner` providing orders-of-magnitude faster
search and zero-GC memory performance.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import Any

from amr_sched.network import Network
from amr_sched.planners.base import PathPlanner, PlannedPath
from amr_sched.reservation import ReservationTable
from amr_sched.rust_reservation import RustReservationTable, _find_library, _LIB as _RES_LIB


def _find_planner_library() -> str | None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidates = [
        repo_root / "crates" / "target" / "release" / "spacetime_planner.dll",
        repo_root / "crates" / "target" / "release" / "libspacetime_planner.so",
        repo_root / "crates" / "target" / "release" / "libspacetime_planner.dylib",
        repo_root / "amr_sched" / "spacetime_planner.dll",
        repo_root / "amr_sched" / "libspacetime_planner.so",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


_PLANNER_LIB_PATH = _find_planner_library()
_PLANNER_LIB = None

if _PLANNER_LIB_PATH and _RES_LIB is not None:
    try:
        _PLANNER_LIB = ctypes.CDLL(_PLANNER_LIB_PATH)

        _PLANNER_LIB.planner_graph_create.restype = ctypes.c_void_p
        _PLANNER_LIB.planner_graph_destroy.argtypes = [ctypes.c_void_p]

        _PLANNER_LIB.planner_graph_add_node.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        _PLANNER_LIB.planner_graph_add_node.restype = ctypes.c_uint16

        _PLANNER_LIB.planner_graph_add_edge.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_size_t
        ]
        _PLANNER_LIB.planner_graph_add_edge.restype = ctypes.c_bool

        _PLANNER_LIB.spacetime_planner_create.argtypes = [ctypes.c_size_t]
        _PLANNER_LIB.spacetime_planner_create.restype = ctypes.c_void_p

        _PLANNER_LIB.spacetime_planner_destroy.argtypes = [ctypes.c_void_p]

        _PLANNER_LIB.spacetime_planner_plan.argtypes = [
            ctypes.c_void_p,  # planner
            ctypes.c_void_p,  # graph
            ctypes.c_void_p,  # reservations
            ctypes.c_char_p,  # start_node
            ctypes.c_char_p,  # goal_node
            ctypes.c_uint32,  # start_step
            ctypes.c_char_p,  # agv_id
            ctypes.c_uint32,  # horizon
            ctypes.POINTER(ctypes.c_void_p),  # out_nodes_csv
            ctypes.POINTER(ctypes.c_void_p),  # out_arrivals_csv
            ctypes.POINTER(ctypes.c_uint32),  # out_wait_steps
        ]
        _PLANNER_LIB.spacetime_planner_plan.restype = ctypes.c_bool

        _PLANNER_LIB.spacetime_planner_free_string.argtypes = [ctypes.c_void_p]
    except Exception:
        _PLANNER_LIB = None


def is_rust_planner_available() -> bool:
    """Returns True if the native Rust spacetime planner library is loaded."""
    return _PLANNER_LIB is not None


class RustSpaceTimeAStarPlanner(PathPlanner):
    """High-performance Spatiotemporal A* planner implemented in Rust."""

    name = "spacetime_astar"
    reservations_aware = True

    @staticmethod
    def create_reservation_table() -> RustReservationTable:
        return RustReservationTable()

    def __init__(
        self,
        max_speed: float = 1.0,
        dt: float = 1.0,
        tie_break_noise: float = 0.0,
        seed: int = 0,
        max_expansions: int = 200_000,
    ) -> None:
        super().__init__(max_speed=max_speed, dt=dt)
        self.tie_break_noise = tie_break_noise
        self._seed = seed
        self.max_expansions = max_expansions
        if _PLANNER_LIB is None:
            raise RuntimeError("Native spacetime_planner library not loaded.")
        self._planner = _PLANNER_LIB.spacetime_planner_create(max_expansions)
        self._cached_graph_net_id: Any = None
        self._cached_graph_ptr: ctypes.c_void_p | None = None

    def __del__(self) -> None:
        if hasattr(self, "_planner") and self._planner and _PLANNER_LIB:
            _PLANNER_LIB.spacetime_planner_destroy(self._planner)
            self._planner = None
        if hasattr(self, "_cached_graph_ptr") and self._cached_graph_ptr and _PLANNER_LIB:
            _PLANNER_LIB.planner_graph_destroy(self._cached_graph_ptr)
            self._cached_graph_ptr = None

    def _get_or_build_graph(self, network: Network) -> ctypes.c_void_p:
        net_id = id(network)
        if self._cached_graph_net_id == net_id and self._cached_graph_ptr is not None:
            return self._cached_graph_ptr

        if self._cached_graph_ptr is not None:
            _PLANNER_LIB.planner_graph_destroy(self._cached_graph_ptr)
            self._cached_graph_ptr = None

        g_ptr = _PLANNER_LIB.planner_graph_create()
        for node in network.nodes.values():
            _PLANNER_LIB.planner_graph_add_node(g_ptr, node.node_id.encode("utf-8"), node.capacity)

        for (src, dst), edge in network.edges.items():
            steps = self.steps_for(edge.traversal_time(self.max_speed))
            phys_cap = min(edge.capacity, network.edge(dst, src).capacity) if network.has_edge(dst, src) else edge.capacity
            _PLANNER_LIB.planner_graph_add_edge(
                g_ptr,
                src.encode("utf-8"),
                dst.encode("utf-8"),
                steps,
                edge.capacity,
                phys_cap,
            )

        self._cached_graph_net_id = net_id
        self._cached_graph_ptr = g_ptr
        return g_ptr

    def plan(
        self,
        network: Network,
        start: str,
        goal: str,
        start_step: int,
        reservations: ReservationTable | RustReservationTable,
        agv_id: str,
        horizon: int,
    ) -> PlannedPath | None:
        if start == goal:
            return PlannedPath(nodes=[], arrival_steps=[], wait_steps=0)

        g_ptr = self._get_or_build_graph(network)

        # Get native pointer to reservation table
        owned_rust_tbl = None
        if isinstance(reservations, RustReservationTable):
            r_ptr = reservations._ptr
        else:
            # Convert Python table on the fly if needed
            owned_rust_tbl = RustReservationTable()
            for node, bucket in getattr(reservations, "_node", {}).items():
                for iv, owner in bucket:
                    owned_rust_tbl.reserve_node(node, iv, owner)
            for (s, d), bucket in getattr(reservations, "_edge", {}).items():
                for iv, owner in bucket:
                    owned_rust_tbl.reserve_edge(s, d, iv, owner)
            r_ptr = owned_rust_tbl._ptr

        nodes_ptr = ctypes.c_void_p()
        arrivals_ptr = ctypes.c_void_p()
        wait_steps = ctypes.c_uint32(0)

        ok = _PLANNER_LIB.spacetime_planner_plan(
            self._planner,
            g_ptr,
            r_ptr,
            start.encode("utf-8"),
            goal.encode("utf-8"),
            start_step,
            agv_id.encode("utf-8"),
            horizon,
            ctypes.byref(nodes_ptr),
            ctypes.byref(arrivals_ptr),
            ctypes.byref(wait_steps),
        )

        if not ok:
            return None

        # Extract output CSVs
        nodes_raw = ctypes.string_at(nodes_ptr) if nodes_ptr.value else b""
        arrivals_raw = ctypes.string_at(arrivals_ptr) if arrivals_ptr.value else b""

        if nodes_ptr.value:
            _PLANNER_LIB.spacetime_planner_free_string(nodes_ptr)
        if arrivals_ptr.value:
            _PLANNER_LIB.spacetime_planner_free_string(arrivals_ptr)

        nodes = [n for n in nodes_raw.decode("utf-8").split(",") if n]
        arrivals = [int(a) for a in arrivals_raw.decode("utf-8").split(",") if a]

        return PlannedPath(
            nodes=nodes,
            arrival_steps=arrivals,
            wait_steps=int(wait_steps.value),
        )
