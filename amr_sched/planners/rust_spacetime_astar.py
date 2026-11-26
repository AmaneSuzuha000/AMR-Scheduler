"""Native Rust-accelerated Spatiotemporal A* Path Planner.

Drop-in replacement for `SpaceTimeAStarPlanner` providing orders-of-magnitude faster
search and zero-GC memory performance.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
from pathlib import Path
from typing import Any

from amr_sched.network import Network
from amr_sched.planners.base import PathPlanner, PlannedPath
from amr_sched.reservation import Interval, ReservationTable
from amr_sched.rust_reservation import RustReservationTable, _find_library, _LIB as _RES_LIB


def _find_planner_library() -> str | None:
    # Priority 0: explicit environment variable override
    env_path = os.environ.get("AMR_SPACETIME_PLANNER_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    repo_root = Path(__file__).resolve().parent.parent.parent
    lib_name = (
        "spacetime_planner.dll"
        if sys.platform == "win32"
        else ("libspacetime_planner.dylib" if sys.platform == "darwin" else "libspacetime_planner.so")
    )
    candidates = [
        repo_root / "crates" / "target" / "release" / lib_name,
        repo_root / "amr_sched" / lib_name,
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

        _PLANNER_LIB.planner_graph_clear_traffic.argtypes = [ctypes.c_void_p]
        _PLANNER_LIB.planner_graph_clear_traffic.restype = None

        _PLANNER_LIB.planner_graph_set_traffic_batch.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _PLANNER_LIB.planner_graph_set_traffic_batch.restype = ctypes.c_size_t

        _PLANNER_LIB.planner_graph_set_traffic_cutoff.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        _PLANNER_LIB.planner_graph_set_traffic_cutoff.restype = None

        _PLANNER_LIB.planner_graph_clear_guides.argtypes = [ctypes.c_void_p]
        _PLANNER_LIB.planner_graph_clear_guides.restype = None

        _PLANNER_LIB.planner_graph_set_guide_batch.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _PLANNER_LIB.planner_graph_set_guide_batch.restype = ctypes.c_size_t

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

        _PLANNER_LIB.sipp_planner_create.argtypes = [ctypes.c_size_t]
        _PLANNER_LIB.sipp_planner_create.restype = ctypes.c_void_p
        _PLANNER_LIB.sipp_planner_destroy.argtypes = [ctypes.c_void_p]
        _PLANNER_LIB.sipp_planner_plan.argtypes = _PLANNER_LIB.spacetime_planner_plan.argtypes
        _PLANNER_LIB.sipp_planner_plan.restype = ctypes.c_bool

        _PLANNER_LIB.spacetime_planner_free_string.argtypes = [ctypes.c_void_p]

        if hasattr(_PLANNER_LIB, "sipp_planner_plan_indexed"):
            _PLANNER_LIB.planner_graph_node_index.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            _PLANNER_LIB.planner_graph_node_index.restype = ctypes.c_uint16

            _PLANNER_LIB.planner_graph_set_traffic_indexed.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.c_uint32,
            ]
            _PLANNER_LIB.planner_graph_set_traffic_indexed.restype = ctypes.c_size_t

            _idx_plan_args = [
                ctypes.c_void_p,  # planner
                ctypes.c_void_p,  # graph
                ctypes.c_void_p,  # reservations
                ctypes.c_uint16,  # start_idx
                ctypes.c_uint16,  # goal_idx
                ctypes.c_uint32,  # start_step
                ctypes.c_char_p,  # agv_id
                ctypes.c_uint32,  # horizon
                ctypes.POINTER(ctypes.c_uint16),  # out_nodes
                ctypes.POINTER(ctypes.c_uint32),  # out_arrivals
                ctypes.c_size_t,  # max_len
                ctypes.POINTER(ctypes.c_uint32),  # out_wait_steps
            ]
            _PLANNER_LIB.spacetime_planner_plan_indexed.argtypes = _idx_plan_args
            _PLANNER_LIB.spacetime_planner_plan_indexed.restype = ctypes.c_int32
            _PLANNER_LIB.sipp_planner_plan_indexed.argtypes = _idx_plan_args
            _PLANNER_LIB.sipp_planner_plan_indexed.restype = ctypes.c_int32

            _PLANNER_LIB.planner_reserve_transitions_indexed.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_size_t,
            ]
            _PLANNER_LIB.planner_reserve_transitions_indexed.restype = ctypes.c_size_t

            _PLANNER_LIB.planner_batch_plan_sipp_indexed.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_uint32),
            ]
            _PLANNER_LIB.planner_batch_plan_sipp_indexed.restype = ctypes.c_size_t

        if hasattr(_PLANNER_LIB, "spacetime_planner_plan_cps_indexed"):
            _PLANNER_LIB.planner_graph_set_dt.argtypes = [ctypes.c_void_p, ctypes.c_double]
            _PLANNER_LIB.planner_graph_set_dt.restype = None

            _PLANNER_LIB.planner_graph_set_node_coords_indexed.argtypes = [
                ctypes.c_void_p, ctypes.c_uint16, ctypes.c_double, ctypes.c_double
            ]
            _PLANNER_LIB.planner_graph_set_node_coords_indexed.restype = ctypes.c_bool

            _PLANNER_LIB.planner_graph_add_edge_ex.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_uint32, ctypes.c_double, ctypes.c_size_t, ctypes.c_size_t
            ]
            _PLANNER_LIB.planner_graph_add_edge_ex.restype = ctypes.c_bool

            _PLANNER_LIB.planner_slot_field_create.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_double, ctypes.c_double, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32
            ]
            _PLANNER_LIB.planner_slot_field_create.restype = ctypes.c_void_p
            _PLANNER_LIB.planner_slot_field_destroy.argtypes = [ctypes.c_void_p]
            _PLANNER_LIB.planner_slot_field_clone.argtypes = [ctypes.c_void_p]
            _PLANNER_LIB.planner_slot_field_clone.restype = ctypes.c_void_p
            _PLANNER_LIB.planner_slot_field_restore_from.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            _PLANNER_LIB.planner_slot_field_restore_from.restype = ctypes.c_bool
            _PLANNER_LIB.planner_slot_field_clear.argtypes = [ctypes.c_void_p]
            _PLANNER_LIB.planner_slot_field_remove_agv.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            _PLANNER_LIB.planner_slot_field_add_points_indexed.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.c_int32
            ]
            _PLANNER_LIB.planner_slot_field_slot_phi_indexed.argtypes = [
                ctypes.c_void_p, ctypes.c_uint16, ctypes.c_int32
            ]
            _PLANNER_LIB.planner_slot_field_slot_phi_indexed.restype = ctypes.c_double
            _PLANNER_LIB.planner_slot_field_phi_indexed.argtypes = [
                ctypes.c_void_p, ctypes.c_uint16, ctypes.c_int32
            ]
            _PLANNER_LIB.planner_slot_field_phi_indexed.restype = ctypes.c_double

            _PLANNER_LIB.planner_reserve_points_with_park_indexed.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t,
                ctypes.c_uint16, ctypes.c_int64, ctypes.c_int64,
            ]
            _PLANNER_LIB.planner_reserve_points_with_park_indexed.restype = ctypes.c_size_t

            _PLANNER_LIB.spacetime_planner_plan_cps_indexed.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint32,
                ctypes.c_bool, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_int64, ctypes.c_uint32, ctypes.c_uint16, ctypes.c_size_t, ctypes.c_bool,
                ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_char_p, ctypes.c_size_t,
            ]
            _PLANNER_LIB.spacetime_planner_plan_cps_indexed.restype = ctypes.c_int32

            _PLANNER_LIB.sipp_planner_plan_indexed_ex.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint32,
                ctypes.c_double, ctypes.c_double, ctypes.c_int64, ctypes.c_uint32,
                ctypes.c_size_t, ctypes.c_bool,
                ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_char_p, ctypes.c_size_t,
            ]
            _PLANNER_LIB.sipp_planner_plan_indexed_ex.restype = ctypes.c_int32

            _PLANNER_LIB.planner_batch_plan_sipp_indexed_ex.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16),
                ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.c_double, ctypes.c_double, ctypes.c_uint32, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            _PLANNER_LIB.planner_batch_plan_sipp_indexed_ex.restype = ctypes.c_size_t
    except Exception:
        _PLANNER_LIB = None


_HAS_INDEXED_FFI = bool(_PLANNER_LIB is not None and hasattr(_PLANNER_LIB, "sipp_planner_plan_indexed"))
_HAS_CPS_FFI = bool(_PLANNER_LIB is not None and hasattr(_PLANNER_LIB, "spacetime_planner_plan_cps_indexed"))


class RustSlotField:
    """Native Rust incremental Gaussian & slot congestion field matching `GraphCongestionField`."""

    def __init__(
        self,
        graph_ptr: ctypes.c_void_p,
        idx_map: dict[str, int],
        *,
        spatial_radius_L: int = 1,
        temporal_radius_T: int = 1,
        sigma_l: float = 2.0,
        sigma_t: float = 2.0,
        lookahead_ks: int = 1,
        lookahead_kt: int = 3,
        slot_delta: int = 3,
        max_track_horizon: int = 220,
        t_start: int = 0,
    ) -> None:
        if not _HAS_CPS_FFI:
            raise RuntimeError("Extended CPS Rust FFI not loaded.")
        self._graph_ptr = graph_ptr
        self._idx_map = idx_map
        self._ptr = _PLANNER_LIB.planner_slot_field_create(
            graph_ptr,
            int(spatial_radius_L),
            int(temporal_radius_T),
            float(sigma_l),
            float(sigma_t),
            int(lookahead_ks),
            int(lookahead_kt),
            int(slot_delta),
            int(max_track_horizon),
            int(t_start),
        )
        if not self._ptr:
            raise RuntimeError("planner_slot_field_create failed.")

    def __del__(self) -> None:
        if hasattr(self, "_ptr") and self._ptr and _PLANNER_LIB:
            _PLANNER_LIB.planner_slot_field_destroy(self._ptr)
            self._ptr = None

    def clear(self) -> None:
        _PLANNER_LIB.planner_slot_field_clear(self._ptr)

    def remove_agv(self, agv_id: str) -> None:
        _PLANNER_LIB.planner_slot_field_remove_agv(self._ptr, agv_id.encode("utf-8"))

    def add_agv_path(
        self,
        agv_id: str,
        path: list[tuple[str, int]],
        *,
        horizon: int | None = None,
    ) -> None:
        if not path:
            self.remove_agv(agv_id)
            return
        idx_map = self._idx_map
        n_arr = (ctypes.c_uint16 * len(path))()
        s_arr = (ctypes.c_int32 * len(path))()
        cnt = 0
        for nd, st in path:
            ui = idx_map.get(nd)
            if ui is not None:
                n_arr[cnt] = ui
                s_arr[cnt] = int(st)
                cnt += 1
        hz = int(horizon) if horizon is not None else -1
        _PLANNER_LIB.planner_slot_field_add_points_indexed(
            self._ptr, agv_id.encode("utf-8"), n_arr, s_arr, cnt, hz
        )

    def slot_Phi(self, node: str, step: int) -> float:
        ui = self._idx_map.get(node)
        if ui is None:
            return 0.0
        return float(_PLANNER_LIB.planner_slot_field_slot_phi_indexed(self._ptr, ui, int(step)))

    def phi(self, node: str, step: int) -> float:
        ui = self._idx_map.get(node)
        if ui is None:
            return 0.0
        return float(_PLANNER_LIB.planner_slot_field_phi_indexed(self._ptr, ui, int(step)))

    def clone(self) -> "RustSlotField":
        cls = type(self)
        inst = cls.__new__(cls)
        inst._graph_ptr = self._graph_ptr
        inst._idx_map = self._idx_map
        inst._ptr = _PLANNER_LIB.planner_slot_field_clone(self._ptr)
        if not inst._ptr:
            raise RuntimeError("planner_slot_field_clone failed.")
        return inst

    def restore_from(self, other: "RustSlotField") -> bool:
        return bool(_PLANNER_LIB.planner_slot_field_restore_from(self._ptr, other._ptr))


def build_indexed_rust_graph(
    network: Network,
    steps_fn: Any,
    *,
    time_fn: Any = None,
    dt_s: float = 1.0,
) -> tuple[ctypes.c_void_p, dict[str, int], list[str]]:
    """Construct a native `PlannerGraph` and parallel `node_id <-> u16` lookup tables."""
    g_ptr = _PLANNER_LIB.planner_graph_create()
    use_ex = bool(_HAS_CPS_FFI and time_fn is not None)
    if use_ex:
        _PLANNER_LIB.planner_graph_set_dt(g_ptr, float(dt_s))
    idx_map: dict[str, int] = {}
    max_idx = -1
    for node in network.nodes.values():
        idx = int(_PLANNER_LIB.planner_graph_add_node(g_ptr, node.node_id.encode("utf-8"), node.capacity))
        idx_map[node.node_id] = idx
        if _HAS_CPS_FFI:
            _PLANNER_LIB.planner_graph_set_node_coords_indexed(g_ptr, idx, float(node.x), float(node.y))
        if idx > max_idx:
            max_idx = idx
    for (src, dst), edge in network.edges.items():
        steps = steps_fn(edge)
        phys_cap = min(edge.capacity, network.edge(dst, src).capacity) if network.has_edge(dst, src) else edge.capacity
        if use_ex:
            _PLANNER_LIB.planner_graph_add_edge_ex(
                g_ptr,
                src.encode("utf-8"),
                dst.encode("utf-8"),
                steps,
                float(time_fn(edge)),
                edge.capacity,
                phys_cap,
            )
        else:
            _PLANNER_LIB.planner_graph_add_edge(
                g_ptr,
                src.encode("utf-8"),
                dst.encode("utf-8"),
                steps,
                edge.capacity,
                phys_cap,
            )
    names_by_idx = [""] * max(0, max_idx + 1)
    for nid, idx in idx_map.items():
        names_by_idx[idx] = nid
    return g_ptr, idx_map, names_by_idx


def extract_rust_reservation_ptr(
    reservations: ReservationTable | RustReservationTable,
) -> tuple[ctypes.c_void_p, RustReservationTable | None]:
    """Return raw `ReservationTable*` directly or build a temporary native mirror when Python-backed."""
    if isinstance(reservations, RustReservationTable):
        return reservations._ptr, None
    owned = RustReservationTable()
    for node, bucket in getattr(reservations, "_node", {}).items():
        for iv, owner in bucket:
            owned.reserve_node(node, iv, owner)
    for (s, d), bucket in getattr(reservations, "_edge", {}).items():
        for iv, owner in bucket:
            owned.reserve_edge(s, d, iv, owner)
    owner_nodes: dict[str, list[str]] = {}
    owner_edges: dict[str, list[tuple[str, str]]] = {}
    for node, owners in getattr(reservations, "_pinned_nodes", {}).items():
        for owner in owners:
            owner_nodes.setdefault(owner, []).append(node)
    for edge, owners in getattr(reservations, "_pinned_edges", {}).items():
        for owner in owners:
            owner_edges.setdefault(owner, []).append(edge)
    for owner in set(owner_nodes) | set(owner_edges):
        owned.transfer_pins(owner, owner_nodes.get(owner, []), owner_edges.get(owner, []))
    return owned._ptr, owned


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
        use_sipp: bool = False,
    ) -> None:
        super().__init__(max_speed=max_speed, dt=dt)
        self.tie_break_noise = tie_break_noise
        self._seed = seed
        self.max_expansions = max_expansions
        self.use_sipp = bool(use_sipp)
        if _PLANNER_LIB is None:
            raise RuntimeError("Native spacetime_planner library not loaded.")
        self._planner = (
            _PLANNER_LIB.sipp_planner_create(max_expansions)
            if self.use_sipp
            else _PLANNER_LIB.spacetime_planner_create(max_expansions)
        )
        self._cached_graph_net_id: Any = None
        self._cached_graph_ptr: ctypes.c_void_p | None = None
        self._idx_map: dict[str, int] = {}
        self._names_by_idx: list[str] = []
        self._buf_cap: int = 512
        self._out_nodes_buf = (ctypes.c_uint16 * 512)()
        self._out_arr_buf = (ctypes.c_uint32 * 512)()
        self._blocker_buf = ctypes.create_string_buffer(2048)
        self.last_expansions: int = 0
        self._time_aware: bool = False
        self._applied_traffic_sig: tuple[Any, ...] = ()
        self._applied_guide_sig: tuple[Any, ...] = ()

    def __del__(self) -> None:
        if hasattr(self, "_planner") and self._planner and _PLANNER_LIB:
            if getattr(self, "use_sipp", False):
                _PLANNER_LIB.sipp_planner_destroy(self._planner)
            else:
                _PLANNER_LIB.spacetime_planner_destroy(self._planner)
            self._planner = None
        if hasattr(self, "_cached_graph_ptr") and self._cached_graph_ptr and _PLANNER_LIB:
            _PLANNER_LIB.planner_graph_destroy(self._cached_graph_ptr)
            self._cached_graph_ptr = None

    def _get_or_build_graph(self, network: Network, *, time_aware: bool = False) -> ctypes.c_void_p:
        net_id = id(network)
        if (
            self._cached_graph_net_id == net_id
            and self._cached_graph_ptr is not None
            and self._time_aware == bool(time_aware)
        ):
            return self._cached_graph_ptr

        if self._cached_graph_ptr is not None:
            _PLANNER_LIB.planner_graph_destroy(self._cached_graph_ptr)
            self._cached_graph_ptr = None

        g_ptr, idx_map, names_by_idx = build_indexed_rust_graph(
            network,
            lambda e: self.steps_for(e.traversal_time(self.max_speed)),
            time_fn=(lambda e: e.traversal_time(self.max_speed)) if time_aware else None,
            dt_s=self.dt,
        )
        self._cached_graph_net_id = net_id
        self._cached_graph_ptr = g_ptr
        self._idx_map = idx_map
        self._names_by_idx = names_by_idx
        self._time_aware = bool(time_aware)
        self._applied_traffic_sig = ()
        self._applied_guide_sig = ()
        return g_ptr

    def create_slot_field(
        self,
        network: Network,
        *,
        spatial_radius_L: int = 1,
        temporal_radius_T: int = 1,
        sigma_l: float = 2.0,
        sigma_t: float = 2.0,
        lookahead_ks: int = 1,
        lookahead_kt: int = 3,
        slot_delta: int = 3,
        max_track_horizon: int = 220,
        t_start: int = 0,
        time_aware: bool = True,
    ) -> RustSlotField:
        """Create a native `RustSlotField` sharing this planner's cached indexed graph."""
        g_ptr = self._get_or_build_graph(network, time_aware=time_aware)
        return RustSlotField(
            g_ptr,
            self._idx_map,
            spatial_radius_L=spatial_radius_L,
            temporal_radius_T=temporal_radius_T,
            sigma_l=sigma_l,
            sigma_t=sigma_t,
            lookahead_ks=lookahead_ks,
            lookahead_kt=lookahead_kt,
            slot_delta=slot_delta,
            max_track_horizon=max_track_horizon,
            t_start=t_start,
        )

    def _parse_blockers(self) -> dict[str, int]:
        raw = self._blocker_buf.value
        if not raw:
            return {}
        out: dict[str, int] = {}
        for line in raw.decode("utf-8", errors="ignore").splitlines():
            if not line:
                continue
            aid, _, cnt_s = line.partition("\t")
            if aid:
                out[aid] = int(cnt_s) if cnt_s else 1
        return out

    def plan_leg(
        self,
        network: Network,
        start: str,
        goal: str,
        start_step: int,
        reservations: ReservationTable | RustReservationTable,
        field: RustSlotField | None,
        agv_id: str,
        horizon: int,
        *,
        prev_arrival_node: str | None = None,
        goal_park_until: int | None = None,
        goal_park_steps: int = 0,
        max_expansions_override: int | None = None,
        mode_oneshot: bool = False,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 5.0,
        lookahead_omega: float = 50.0,
        collect_blockers: bool = True,
    ) -> tuple[PlannedPath | None, dict[str, int]]:
        """Drop-in native CPS-A* / SIPP-Ex leg search returning `(PlannedPath | None, blocker_hits)`."""
        if not _HAS_CPS_FFI:
            p = self.plan(network, start, goal, start_step, reservations, agv_id, horizon)
            self.last_expansions = 0
            return p, {}

        g_ptr = self._get_or_build_graph(network, time_aware=True)
        r_ptr, _owned = extract_rust_reservation_ptr(reservations)
        idx_map = self._idx_map
        s_idx = idx_map.get(start)
        g_idx = idx_map.get(goal)
        if s_idx is None or g_idx is None:
            self.last_expansions = 0
            return None, {}

        f_ptr = field._ptr if isinstance(field, RustSlotField) else None
        p_idx = idx_map.get(prev_arrival_node, 0xFFFF) if prev_arrival_node else 0xFFFF
        gpu = int(goal_park_until) if goal_park_until is not None else -1
        m_exp = int(max_expansions_override) if max_expansions_override is not None else 0
        agv_b = agv_id.encode("utf-8")
        out_wait = ctypes.c_uint32(0)
        out_exp = ctypes.c_size_t(0)
        use_sipp = bool(getattr(self, "use_sipp", False))

        while True:
            self._blocker_buf[0] = b"\x00"
            if use_sipp:
                n_hops = int(
                    _PLANNER_LIB.sipp_planner_plan_indexed_ex(
                        self._planner, g_ptr, r_ptr, f_ptr,
                        s_idx, g_idx, int(start_step), agv_b, int(horizon),
                        float(alpha), float(gamma), gpu, int(goal_park_steps), m_exp,
                        bool(collect_blockers),
                        self._out_nodes_buf, self._out_arr_buf, self._buf_cap,
                        ctypes.byref(out_wait), ctypes.byref(out_exp),
                        self._blocker_buf, len(self._blocker_buf),
                    )
                )
            else:
                n_hops = int(
                    _PLANNER_LIB.spacetime_planner_plan_cps_indexed(
                        self._planner, g_ptr, r_ptr, f_ptr,
                        s_idx, g_idx, int(start_step), agv_b, int(horizon),
                        bool(mode_oneshot), float(alpha), float(beta), float(gamma),
                        float(lookahead_omega), gpu, int(goal_park_steps),
                        p_idx, m_exp, bool(collect_blockers),
                        self._out_nodes_buf, self._out_arr_buf, self._buf_cap,
                        ctypes.byref(out_wait), ctypes.byref(out_exp),
                        self._blocker_buf, len(self._blocker_buf),
                    )
                )
            if n_hops == -2:
                self._buf_cap *= 2
                self._out_nodes_buf = (ctypes.c_uint16 * self._buf_cap)()
                self._out_arr_buf = (ctypes.c_uint32 * self._buf_cap)()
                continue
            break

        self.last_expansions = int(out_exp.value)
        hits = self._parse_blockers() if collect_blockers else {}
        if n_hops < 0:
            return None, hits
        names = self._names_by_idx
        nodes = [names[self._out_nodes_buf[i]] for i in range(n_hops)]
        arrivals = [int(self._out_arr_buf[i]) for i in range(n_hops)]
        return PlannedPath(
            nodes=nodes,
            arrival_steps=arrivals,
            wait_steps=int(out_wait.value),
        ), hits

    def reserve_step_points_indexed(
        self,
        network: Network,
        res_table: RustReservationTable,
        agv_id: str,
        node_pts: list[tuple[str, int]],
        edge_segs: list[tuple[str, str, int, int]] | None = None,
        *,
        park_until: int | None = None,
    ) -> bool:
        """Reserve explicit step nodes, edge spans, and optional parking dwell in ONE FFI call."""
        if not node_pts:
            return True
        if edge_segs is None:
            inferred_edges: list[tuple[str, str, int, int]] = []
            for i in range(len(node_pts) - 1):
                u_n, u_t = node_pts[i]
                v_n, v_t = node_pts[i + 1]
                if u_n != v_n and v_t > u_t:
                    inferred_edges.append((u_n, v_n, int(u_t), int(v_t)))
            edge_segs = inferred_edges

        last_n, last_t = node_pts[-1]
        has_park = park_until is not None and int(park_until) > int(last_t) + 1
        if not (_HAS_CPS_FFI and isinstance(res_table, RustReservationTable) and not res_table._track_shadow):
            for nd, tk in node_pts:
                res_table.reserve_node(nd, Interval(int(tk), int(tk) + 1), agv_id)
            for u_n, v_n, dt, at in edge_segs:
                if at > dt:
                    res_table.reserve_edge(u_n, v_n, Interval(int(dt), int(at)), agv_id)
            if has_park:
                res_table.reserve_node(last_n, Interval(int(last_t) + 1, int(park_until)), agv_id)
            return True

        g_ptr = self._get_or_build_graph(network, time_aware=self._time_aware)
        idx_map = self._idx_map
        n_p = len(node_pts)
        pt_n = (ctypes.c_uint16 * n_p)()
        pt_s = (ctypes.c_int32 * n_p)()
        valid_p = 0
        for nd, tk in node_pts:
            ui = idx_map.get(nd)
            if ui is not None:
                pt_n[valid_p] = ui
                pt_s[valid_p] = int(tk)
                valid_p += 1

        n_e = len(edge_segs)
        if n_e > 0:
            e_u = (ctypes.c_uint16 * n_e)()
            e_v = (ctypes.c_uint16 * n_e)()
            e_st = (ctypes.c_int64 * n_e)()
            e_en = (ctypes.c_int64 * n_e)()
            valid_e = 0
            for u_n, v_n, dt, at in edge_segs:
                if at <= dt:
                    continue
                ui = idx_map.get(u_n)
                vi = idx_map.get(v_n)
                if ui is not None and vi is not None:
                    e_u[valid_e] = ui
                    e_v[valid_e] = vi
                    e_st[valid_e] = int(dt)
                    e_en[valid_e] = int(at)
                    valid_e += 1
        else:
            e_u = e_v = ctypes.POINTER(ctypes.c_uint16)()
            e_st = e_en = ctypes.POINTER(ctypes.c_int64)()
            valid_e = 0

        pk_idx = idx_map.get(last_n, 0xFFFF) if has_park else 0xFFFF
        pk_st = int(last_t) + 1 if has_park else 0
        pk_en = int(park_until) if has_park else 0
        _PLANNER_LIB.planner_reserve_points_with_park_indexed(
            g_ptr, res_table._ptr, agv_id.encode("utf-8"),
            pt_n, pt_s, valid_p,
            e_u, e_v, e_st, e_en, valid_e,
            pk_idx, pk_st, pk_en,
        )
        return True

    def sync_traffic_weights(
        self,
        network: Network,
        weights_q3: dict[tuple[str, str], int] | None,
        cutoff_step: int = 0,
    ) -> None:
        """Synchronize online LTM Q3 edge penalties and optional horizon cutoff onto native graph."""
        c_step = max(0, int(cutoff_step))
        if not weights_q3:
            sig: tuple[Any, ...] = (0, ()) if c_step == 0 else ()
            edges_t: tuple[tuple[str, str, int], ...] = ()
        else:
            edges_t = tuple(
                sorted(
                    (str(s), str(d), max(0, min(64, int(w))))
                    for (s, d), w in weights_q3.items()
                    if int(w) > 0
                )
            )
            sig = (c_step, edges_t) if edges_t else ()
        if not edges_t:
            sig = ()
        if self._cached_graph_ptr is None or self._cached_graph_net_id != id(network):
            self._get_or_build_graph(network)
            self._applied_traffic_sig = ()
        if sig == self._applied_traffic_sig:
            return
        g_ptr = self._cached_graph_ptr
        if not edges_t:
            _PLANNER_LIB.planner_graph_clear_traffic(g_ptr)
        elif _HAS_INDEXED_FFI:
            idx_map = self._idx_map
            m = len(edges_t)
            s_arr = (ctypes.c_uint16 * m)()
            d_arr = (ctypes.c_uint16 * m)()
            p_arr = (ctypes.c_uint32 * m)()
            cnt = 0
            for s, d, w in edges_t:
                ui = idx_map.get(s)
                vi = idx_map.get(d)
                if ui is not None and vi is not None:
                    s_arr[cnt] = ui
                    d_arr[cnt] = vi
                    p_arr[cnt] = w
                    cnt += 1
            _PLANNER_LIB.planner_graph_set_traffic_indexed(g_ptr, s_arr, d_arr, p_arr, cnt, c_step)
        else:
            script = "\n".join(f"{s}\t{d}\t{w}" for s, d, w in edges_t)
            _PLANNER_LIB.planner_graph_set_traffic_batch(g_ptr, script.encode("utf-8"))
            _PLANNER_LIB.planner_graph_set_traffic_cutoff(g_ptr, c_step)
        self._applied_traffic_sig = sig

    def clear_traffic_weights(self) -> None:
        """Reset online LTM traffic penalties and time cutoff to zero on the cached native graph."""
        if self._cached_graph_ptr is not None and self._applied_traffic_sig:
            _PLANNER_LIB.planner_graph_clear_traffic(self._cached_graph_ptr)
        self._applied_traffic_sig = ()

    def sync_agv_guides(
        self,
        network: Network,
        guide_corridors: dict[str, list[tuple[str, str]]] | None,
        off_guide_pen_q3: int = 12,
    ) -> None:
        """Synchronize per-AGV CVCR coordinated soft corridor preferences onto native graph."""
        pen = max(0, min(64, int(off_guide_pen_q3)))
        if not guide_corridors or pen <= 0:
            sig: tuple[Any, ...] = ()
            items: list[tuple[str, str, str]] = []
        else:
            items = sorted(
                (str(aid), str(u), str(v))
                for aid, elist in guide_corridors.items()
                for (u, v) in elist
            )
            sig = (pen, tuple(items)) if items else ()
        if self._cached_graph_ptr is None or self._cached_graph_net_id != id(network):
            self._get_or_build_graph(network)
            self._applied_guide_sig = ()
        if sig == self._applied_guide_sig:
            return
        g_ptr = self._cached_graph_ptr
        if not items:
            _PLANNER_LIB.planner_graph_clear_guides(g_ptr)
        else:
            script = "\n".join(f"{aid}\t{pen}\t{u}\t{v}" for aid, u, v in items)
            _PLANNER_LIB.planner_graph_set_guide_batch(g_ptr, script.encode("utf-8"))
        self._applied_guide_sig = sig

    def clear_agv_guides(self) -> None:
        """Clear per-AGV CVCR soft corridor guidance filters on cached native graph."""
        if self._cached_graph_ptr is not None and self._applied_guide_sig:
            _PLANNER_LIB.planner_graph_clear_guides(self._cached_graph_ptr)
        self._applied_guide_sig = ()

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
            idx_fn = (
                _PLANNER_LIB.sipp_planner_plan_indexed
                if getattr(self, "use_sipp", False)
                else _PLANNER_LIB.spacetime_planner_plan_indexed
            )
            agv_b = agv_id.encode("utf-8")
            while True:
                n_hops = int(
                    idx_fn(
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
            nodes = [names[self._out_nodes_buf[i]] for i in range(n_hops)]
            arrivals = [int(self._out_arr_buf[i]) for i in range(n_hops)]
            return PlannedPath(
                nodes=nodes,
                arrival_steps=arrivals,
                wait_steps=int(wait_steps.value),
            )

        nodes_ptr = ctypes.c_void_p()
        arrivals_ptr = ctypes.c_void_p()
        plan_fn = (
            _PLANNER_LIB.sipp_planner_plan
            if getattr(self, "use_sipp", False)
            else _PLANNER_LIB.spacetime_planner_plan
        )
        ok = plan_fn(
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

    def reserve_chain_indexed(
        self,
        network: Network,
        res_table: RustReservationTable,
        agv_id: str,
        path: list[tuple[str, int]],
        extra_node_wait: tuple[str, int, int] | None = None,
    ) -> bool:
        """Reserve path transitions directly in Rust without ASCII script serialization."""
        if not _HAS_INDEXED_FFI:
            return False
        g_ptr = self._get_or_build_graph(network)
        idx_map = self._idx_map
        items: list[tuple[int, int, int, int]] = []
        for i in range(len(path) - 1):
            u_n, u_t = path[i]
            v_n, v_t = path[i + 1]
            if v_t <= u_t:
                continue
            ui = idx_map.get(u_n)
            vi = idx_map.get(v_n)
            if ui is None or vi is None:
                return False
            items.append((ui, vi, int(u_t), int(v_t)))
        if extra_node_wait is not None:
            w_n, w_s, w_e = extra_node_wait
            if w_e > w_s:
                wi = idx_map.get(w_n)
                if wi is None:
                    return False
                items.append((wi, wi, int(w_s), int(w_e)))
        if not items:
            return True
        m = len(items)
        s_arr = (ctypes.c_uint16 * m)()
        d_arr = (ctypes.c_uint16 * m)()
        st_arr = (ctypes.c_int64 * m)()
        en_arr = (ctypes.c_int64 * m)()
        for idx, (ui, vi, ts, te) in enumerate(items):
            s_arr[idx] = ui
            d_arr[idx] = vi
            st_arr[idx] = ts
            en_arr[idx] = te
        applied = int(
            _PLANNER_LIB.planner_reserve_transitions_indexed(
                g_ptr,
                res_table._ptr,
                agv_id.encode("utf-8"),
                s_arr,
                d_arr,
                st_arr,
                en_arr,
                m,
            )
        )
        return applied == m

    def plan_batch_sipp(
        self,
        network: Network,
        reservations: ReservationTable | RustReservationTable,
        queries: list[tuple[str, str, int, str, int]],
    ) -> list[PlannedPath | None]:
        """Solve a batch of `(start, goal, start_step, agv_id, horizon)` queries in parallel in Rust."""
        if not queries:
            return []
        if not (_HAS_INDEXED_FFI and getattr(self, "use_sipp", False)):
            return [self.plan(network, s, g, t0, reservations, aid, hz) for s, g, t0, aid, hz in queries]
        g_ptr = self._get_or_build_graph(network)
        r_ptr, _owned = extract_rust_reservation_ptr(reservations)
        n_q = len(queries)
        idx_map = self._idx_map
        stride = max(64, len(idx_map) * 2)
        s_arr = (ctypes.c_uint16 * n_q)()
        g_arr = (ctypes.c_uint16 * n_q)()
        t_arr = (ctypes.c_uint32 * n_q)()
        h_arr = (ctypes.c_uint32 * n_q)()
        a_arr = (ctypes.c_char_p * n_q)()
        valid_mask = [True] * n_q
        for i, (s, g, t0, aid, hz) in enumerate(queries):
            si = idx_map.get(s)
            gi = idx_map.get(g)
            if si is None or gi is None or s == g:
                valid_mask[i] = False
                s_arr[i] = 0
                g_arr[i] = 0
            else:
                s_arr[i] = si
                g_arr[i] = gi
            t_arr[i] = int(t0)
            h_arr[i] = int(hz)
            a_arr[i] = aid.encode("utf-8")
        out_n = (ctypes.c_uint16 * (n_q * stride))()
        out_a = (ctypes.c_uint32 * (n_q * stride))()
        out_l = (ctypes.c_int32 * n_q)()
        out_w = (ctypes.c_uint32 * n_q)()
        _PLANNER_LIB.planner_batch_plan_sipp_indexed(
            self._planner, g_ptr, r_ptr, n_q, s_arr, g_arr, t_arr, h_arr, a_arr,
            stride, out_n, out_a, out_l, out_w,
        )
        names = self._names_by_idx
        res: list[PlannedPath | None] = []
        for i, (s, g, t0, aid, hz) in enumerate(queries):
            if s == g:
                res.append(PlannedPath(nodes=[], arrival_steps=[], wait_steps=0))
            elif not valid_mask[i]:
                res.append(None)
            else:
                ln = int(out_l[i])
                if ln == -2:
                    res.append(self.plan(network, s, g, t0, reservations, aid, hz))
                elif ln < 0:
                    res.append(None)
                else:
                    base = i * stride
                    res.append(PlannedPath(
                        nodes=[names[out_n[base + k]] for k in range(ln)],
                        arrival_steps=[int(out_a[base + k]) for k in range(ln)],
                        wait_steps=int(out_w[i]),
                    ))
        return res


class RustCPSAStarGraphPlanner:
    """1:1 drop-in native Rust replacement for `CPSAStarGraphPlanner` (`plan_leg` + `last_expansions`)."""

    def __init__(
        self,
        network: Network,
        cfg: Any,
        *,
        max_speed: float = 1.0,
        dt: float = 1.0,
        use_sipp: bool = False,
    ) -> None:
        self.network = network
        self.cfg = cfg
        self.max_speed = float(max_speed)
        self.dt = float(dt)
        self.use_sipp = bool(use_sipp)
        self.last_expansions: int = 0
        self._planner = RustSpaceTimeAStarPlanner(
            max_speed=self.max_speed,
            dt=self.dt,
            max_expansions=max(1000, int(getattr(cfg, "max_expansions", 12_000))),
            use_sipp=self.use_sipp,
        )
        # Warm indexed graph with continuous edge times matching `CPSAStarGraphPlanner._heuristic_table`
        self._planner._get_or_build_graph(network, time_aware=True)

    def create_field(self, *, max_track_horizon: int = 220, t_start: int = 0) -> RustSlotField:
        f_cfg = getattr(self.cfg, "field", None)
        return self._planner.create_slot_field(
            self.network,
            spatial_radius_L=int(getattr(f_cfg, "spatial_radius_L", 1)),
            temporal_radius_T=int(getattr(f_cfg, "temporal_radius_T", 1)),
            sigma_l=float(getattr(f_cfg, "sigma_l", 2.0)),
            sigma_t=float(getattr(f_cfg, "sigma_t", 2.0)),
            lookahead_ks=int(getattr(f_cfg, "lookahead_ks", 1)),
            lookahead_kt=int(getattr(f_cfg, "lookahead_kt", 3)),
            slot_delta=int(getattr(f_cfg, "slot_delta", 3)),
            max_track_horizon=int(max_track_horizon),
            t_start=int(t_start),
            time_aware=True,
        )

    def reserve_step_points_indexed(
        self,
        res_table: RustReservationTable,
        agv_id: str,
        node_pts: list[tuple[str, int]],
        edge_segs: list[tuple[str, str, int, int]] | None = None,
        *,
        park_until: int | None = None,
    ) -> bool:
        return self._planner.reserve_step_points_indexed(
            self.network, res_table, agv_id, node_pts, edge_segs, park_until=park_until
        )

    def plan_leg(
        self,
        start: str,
        goal: str,
        start_step: int,
        reservations: ReservationTable | RustReservationTable,
        field: RustSlotField | None,
        agv_id: str,
        horizon: int,
        *,
        prev_arrival_node: str | None = None,
        goal_park_until: int | None = None,
        goal_park_steps: int = 0,
        max_expansions_override: int | None = None,
    ) -> tuple[PlannedPath | None, dict[str, int]]:
        mode_oneshot = str(getattr(self.cfg, "mode", "lifelong")).lower() == "oneshot"
        path, hits = self._planner.plan_leg(
            self.network,
            start,
            goal,
            start_step,
            reservations,
            field,
            agv_id,
            horizon,
            prev_arrival_node=prev_arrival_node,
            goal_park_until=goal_park_until,
            goal_park_steps=goal_park_steps,
            max_expansions_override=max_expansions_override,
            mode_oneshot=mode_oneshot,
            alpha=float(getattr(self.cfg, "alpha", 1.0)),
            beta=float(getattr(self.cfg, "beta", 0.5)),
            gamma=float(getattr(self.cfg, "gamma", 5.0)),
            lookahead_omega=float(getattr(self.cfg, "lookahead_omega", 50.0)),
            collect_blockers=True,
        )
        self.last_expansions = self._planner.last_expansions
        return path, hits
