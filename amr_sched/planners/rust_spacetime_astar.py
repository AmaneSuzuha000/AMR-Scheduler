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
from amr_sched.reservation import ReservationTable
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
    except Exception:
        _PLANNER_LIB = None


_HAS_INDEXED_FFI = bool(_PLANNER_LIB is not None and hasattr(_PLANNER_LIB, "sipp_planner_plan_indexed"))


def build_indexed_rust_graph(
    network: Network,
    steps_fn: Any,
) -> tuple[ctypes.c_void_p, dict[str, int], list[str]]:
    """Construct a native `PlannerGraph` and parallel `node_id <-> u16` lookup tables."""
    g_ptr = _PLANNER_LIB.planner_graph_create()
    idx_map: dict[str, int] = {}
    max_idx = -1
    for node in network.nodes.values():
        idx = int(_PLANNER_LIB.planner_graph_add_node(g_ptr, node.node_id.encode("utf-8"), node.capacity))
        idx_map[node.node_id] = idx
        if idx > max_idx:
            max_idx = idx
    for (src, dst), edge in network.edges.items():
        steps = steps_fn(edge)
        phys_cap = min(edge.capacity, network.edge(dst, src).capacity) if network.has_edge(dst, src) else edge.capacity
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

    def _get_or_build_graph(self, network: Network) -> ctypes.c_void_p:
        net_id = id(network)
        if self._cached_graph_net_id == net_id and self._cached_graph_ptr is not None:
            return self._cached_graph_ptr

        if self._cached_graph_ptr is not None:
            _PLANNER_LIB.planner_graph_destroy(self._cached_graph_ptr)
            self._cached_graph_ptr = None

        g_ptr, idx_map, names_by_idx = build_indexed_rust_graph(
            network,
            lambda e: self.steps_for(e.traversal_time(self.max_speed)),
        )
        self._cached_graph_net_id = net_id
        self._cached_graph_ptr = g_ptr
        self._idx_map = idx_map
        self._names_by_idx = names_by_idx
        self._applied_traffic_sig = ()
        self._applied_guide_sig = ()
        return g_ptr

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
