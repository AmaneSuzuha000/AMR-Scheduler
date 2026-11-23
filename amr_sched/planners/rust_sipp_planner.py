"""Native Rust-accelerated Safe Interval Path Planning (SIPP) Planner.

Computes collision-free routes across continuous safe time windows rather than
discrete time steps, yielding orders-of-magnitude fewer state expansions.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from amr_sched.network import Network
from amr_sched.planners.base import PathPlanner, PlannedPath
from amr_sched.reservation import ReservationTable
from amr_sched.rust_reservation import RustReservationTable, _LIB as _RES_LIB
from amr_sched.planners.rust_spacetime_astar import (
    _HAS_INDEXED_FFI,
    _PLANNER_LIB as _ASTAR_PLANNER_LIB,
    _PLANNER_LIB_PATH,
    build_indexed_rust_graph,
    extract_rust_reservation_ptr,
)

_SIPP_LIB = _ASTAR_PLANNER_LIB


def is_rust_sipp_available() -> bool:
    """Returns True if the native Rust SIPP library is loaded."""
    return _SIPP_LIB is not None


class RustSippPlanner(PathPlanner):
    """High-performance Safe Interval Path Planner (SIPP) implemented in Rust."""

    name = "rust_sipp"
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
        max_expansions: int = 50_000,
    ) -> None:
        super().__init__(max_speed=max_speed, dt=dt)
        self.tie_break_noise = tie_break_noise
        self._seed = seed
        self.max_expansions = max_expansions
        if _SIPP_LIB is None:
            raise RuntimeError("Native SIPP library not loaded.")
        self._planner = _SIPP_LIB.sipp_planner_create(max_expansions)
        self._cached_graph_net_id: Any = None
        self._cached_graph_ptr: ctypes.c_void_p | None = None
        self._idx_map: dict[str, int] = {}
        self._names_by_idx: list[str] = []
        self._buf_cap: int = 512
        self._out_nodes_buf = (ctypes.c_uint16 * 512)()
        self._out_arr_buf = (ctypes.c_uint32 * 512)()

    def __del__(self) -> None:
        if hasattr(self, "_planner") and self._planner and _SIPP_LIB:
            _SIPP_LIB.sipp_planner_destroy(self._planner)
            self._planner = None
        if hasattr(self, "_cached_graph_ptr") and self._cached_graph_ptr and _SIPP_LIB:
            _SIPP_LIB.planner_graph_destroy(self._cached_graph_ptr)
            self._cached_graph_ptr = None

    def _get_or_build_graph(self, network: Network) -> ctypes.c_void_p:
        net_id = id(network)
        if self._cached_graph_net_id == net_id and self._cached_graph_ptr is not None:
            return self._cached_graph_ptr

        if self._cached_graph_ptr is not None:
            _SIPP_LIB.planner_graph_destroy(self._cached_graph_ptr)
            self._cached_graph_ptr = None

        g_ptr, idx_map, names_by_idx = build_indexed_rust_graph(
            network,
            lambda e: self.steps_for(e.traversal_time(self.max_speed)),
        )
        self._cached_graph_net_id = net_id
        self._cached_graph_ptr = g_ptr
        self._idx_map = idx_map
        self._names_by_idx = names_by_idx
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
        r_ptr, _owned_rust_tbl = extract_rust_reservation_ptr(reservations)

        wait_steps = ctypes.c_uint32(0)
        if _HAS_INDEXED_FFI:
            s_idx = self._idx_map.get(start)
            g_idx = self._idx_map.get(goal)
            if s_idx is None or g_idx is None:
                return None
            agv_b = agv_id.encode("utf-8")
            while True:
                n_hops = int(
                    _SIPP_LIB.sipp_planner_plan_indexed(
                        self._planner,
                        g_ptr,
                        r_ptr,
                        s_idx,
                        g_idx,
                        start_step,
                        agv_b,
                        horizon,
                        self._out_nodes_buf,
                        self._out_arr_buf,
                        self._buf_cap,
                        ctypes.byref(wait_steps),
                    )
                )
                if n_hops == -2:
                    self._buf_cap *= 2
                    self._out_nodes_buf = (ctypes.c_uint16 * self._buf_cap)()
                    self._out_arr_buf = (ctypes.c_uint32 * self._buf_cap)()
                    continue
                break
            if n_hops < 0:
                return None
            names = self._names_by_idx
            return PlannedPath(
                nodes=[names[self._out_nodes_buf[i]] for i in range(n_hops)],
                arrival_steps=[int(self._out_arr_buf[i]) for i in range(n_hops)],
                wait_steps=int(wait_steps.value),
            )

        nodes_ptr = ctypes.c_void_p()
        arrivals_ptr = ctypes.c_void_p()
        ok = _SIPP_LIB.sipp_planner_plan(
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

        nodes_raw = ctypes.string_at(nodes_ptr) if nodes_ptr.value else b""
        arrivals_raw = ctypes.string_at(arrivals_ptr) if arrivals_ptr.value else b""

        if nodes_ptr.value:
            _SIPP_LIB.spacetime_planner_free_string(nodes_ptr)
        if arrivals_ptr.value:
            _SIPP_LIB.spacetime_planner_free_string(arrivals_ptr)

        nodes = [n for n in nodes_raw.decode("utf-8").split(",") if n]
        arrivals = [int(a) for a in arrivals_raw.decode("utf-8").split(",") if a]

        return PlannedPath(
            nodes=nodes,
            arrival_steps=arrivals,
            wait_steps=int(wait_steps.value),
        )
