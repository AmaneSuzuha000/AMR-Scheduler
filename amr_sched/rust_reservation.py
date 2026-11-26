"""Native Rust-accelerated Spatiotemporal Reservation Table.

Provides an exact 1:1 drop-in replacement for `amr_sched.reservation.ReservationTable`
backed by the compiled `reservation_core` native binary.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Iterable

from amr_sched.reservation import Interval


def _find_library() -> str | None:
    # Priority 0: explicit environment variable override
    env_path = os.environ.get("AMR_RESERVATION_CORE_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # Priority 1: crates/target/release & package root matching host OS ABI
    repo_root = Path(__file__).resolve().parent.parent
    lib_name = (
        "reservation_core.dll"
        if sys.platform == "win32"
        else ("libreservation_core.dylib" if sys.platform == "darwin" else "libreservation_core.so")
    )
    candidates = [
        repo_root / "crates" / "target" / "release" / lib_name,
        repo_root / "amr_sched" / lib_name,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


_LIB_PATH = _find_library()
_LIB = None

if _LIB_PATH:
    try:
        _LIB = ctypes.CDLL(_LIB_PATH)

        # Set up C function prototypes
        _LIB.reservation_table_create.restype = ctypes.c_void_p

        _LIB.reservation_table_destroy.argtypes = [ctypes.c_void_p]

        _LIB.reservation_table_clear.argtypes = [ctypes.c_void_p]
        _LIB.reservation_table_reset_all.argtypes = [ctypes.c_void_p]

        _LIB.reservation_table_reserve_node.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_char_p
        ]
        _LIB.reservation_table_reserve_node.restype = ctypes.c_bool

        _LIB.reservation_table_reserve_edge.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_char_p
        ]
        _LIB.reservation_table_reserve_edge.restype = ctypes.c_bool

        _LIB.reservation_table_is_node_free.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_size_t, ctypes.c_char_p
        ]
        _LIB.reservation_table_is_node_free.restype = ctypes.c_bool

        _LIB.reservation_table_is_edge_free.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_size_t, ctypes.c_char_p
        ]
        _LIB.reservation_table_is_edge_free.restype = ctypes.c_bool

        _LIB.reservation_table_is_opposing_edge_free.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_char_p
        ]
        _LIB.reservation_table_is_opposing_edge_free.restype = ctypes.c_bool

        _LIB.reservation_table_release_node.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_char_p
        ]
        _LIB.reservation_table_release_node.restype = ctypes.c_bool

        _LIB.reservation_table_release_edge.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_char_p
        ]
        _LIB.reservation_table_release_edge.restype = ctypes.c_bool

        _LIB.reservation_table_release_all.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _LIB.reservation_table_release_all.restype = ctypes.c_bool

        _LIB.reservation_table_restore_from.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _LIB.reservation_table_restore_from.restype = ctypes.c_bool

        _LIB.reservation_table_clone.argtypes = [ctypes.c_void_p]
        _LIB.reservation_table_clone.restype = ctypes.c_void_p

        _LIB.reservation_table_reserve_batch.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p
        ]
        _LIB.reservation_table_reserve_batch.restype = ctypes.c_size_t

        _LIB.reservation_table_purge_before.argtypes = [ctypes.c_void_p, ctypes.c_int64]

        _LIB.reservation_table_transfer_pins.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p
        ]
        _LIB.reservation_table_transfer_pins.restype = ctypes.c_bool

        _LIB.reservation_table_total_reservations.argtypes = [ctypes.c_void_p]
        _LIB.reservation_table_total_reservations.restype = ctypes.c_size_t

        _LIB.reservation_table_utilization.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_size_t, ctypes.c_size_t]
        _LIB.reservation_table_utilization.restype = ctypes.c_double

        _LIB.reservation_table_node_conflicts_csv.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_char_p
        ]
        _LIB.reservation_table_node_conflicts_csv.restype = ctypes.c_void_p

        _LIB.reservation_table_edge_conflicts_csv.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_char_p
        ]
        _LIB.reservation_table_edge_conflicts_csv.restype = ctypes.c_void_p

        _LIB.reservation_table_free_string.argtypes = [ctypes.c_void_p]

        # Transaction prototypes
        _LIB.transaction_begin.argtypes = [ctypes.c_char_p]
        _LIB.transaction_begin.restype = ctypes.c_void_p

        _LIB.transaction_destroy.argtypes = [ctypes.c_void_p]

        _LIB.transaction_stage_node.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_size_t
        ]
        _LIB.transaction_stage_node.restype = ctypes.c_bool

        _LIB.transaction_stage_edge.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_size_t
        ]
        _LIB.transaction_stage_edge.restype = ctypes.c_bool

        _LIB.transaction_staged_count.argtypes = [ctypes.c_void_p]
        _LIB.transaction_staged_count.restype = ctypes.c_size_t

        _LIB.transaction_can_commit.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _LIB.transaction_can_commit.restype = ctypes.c_bool

        _LIB.transaction_commit.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _LIB.transaction_commit.restype = ctypes.c_bool

        _LIB.transaction_rollback.argtypes = [ctypes.c_void_p]

        # Ledger prototypes
        _LIB.ledger_create.restype = ctypes.c_void_p
        _LIB.ledger_destroy.argtypes = [ctypes.c_void_p]
        _LIB.ledger_commit_hop.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64
        ]
        _LIB.ledger_commit_hop.restype = ctypes.c_bool
        _LIB.ledger_complete_hop.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        _LIB.ledger_complete_hop.restype = ctypes.c_bool
        _LIB.ledger_has_active_commitment.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _LIB.ledger_has_active_commitment.restype = ctypes.c_bool
        _LIB.ledger_active_count.argtypes = [ctypes.c_void_p]
        _LIB.ledger_active_count.restype = ctypes.c_size_t
    except Exception as _e:
        _LIB = None


def is_rust_core_available() -> bool:
    """Returns True if the native Rust reservation core library is loaded."""
    return _LIB is not None


class RustReservationTable:
    """Spatiotemporal reservation table powered by the native Rust kernel.

    API is completely compatible with `amr_sched.reservation.ReservationTable`.
    """

    def __init__(self, track_shadow: bool = True) -> None:
        if _LIB is None:
            raise RuntimeError("Native reservation_core library not loaded.")
        self._ptr = _LIB.reservation_table_create()
        self._track_shadow = bool(track_shadow)
        # Keep track of python-side pins and shadow interval dicts for rapid inspection & snapshot capture
        self._pinned_nodes: dict[str, set[str]] = {}
        self._pinned_edges: dict[tuple[str, str], set[str]] = {}
        self._node: dict[str, list[tuple[Interval, str]]] = {}
        self._edge: dict[tuple[str, str], list[tuple[Interval, str]]] = {}

    @classmethod
    def fast(cls) -> "RustReservationTable":
        """Create a zero-shadow native reservation table (maximum speed, no Python shadow dicts)."""
        return cls(track_shadow=False)

    def __del__(self) -> None:
        if hasattr(self, "_ptr") and self._ptr and _LIB:
            _LIB.reservation_table_destroy(self._ptr)
            self._ptr = None

    def clear(self) -> None:
        """Drop all finite reservations (pinned physical occupancy is preserved)."""
        _LIB.reservation_table_clear(self._ptr)
        self._node.clear()
        self._edge.clear()

    def transfer_pins(self, agv_id: str, nodes: Iterable[str],
                      edges: Iterable[tuple[str, str]] = ()) -> None:
        """Replace physical resource ownership only after confirmed movement."""
        node_list = list(nodes)
        edge_list = list(edges)

        # Update python shadow
        for table in (self._pinned_nodes, self._pinned_edges):
            for owners in table.values():
                owners.discard(agv_id)
        for node in node_list:
            self._pinned_nodes.setdefault(node, set()).add(agv_id)
        for edge in edge_list:
            self._pinned_edges.setdefault(edge, set()).add(agv_id)

        # Sync to Rust
        nodes_csv = ",".join(node_list).encode("utf-8")
        edges_csv = ",".join(f"{s}->{d}" for s, d in edge_list).encode("utf-8")
        _LIB.reservation_table_transfer_pins(self._ptr, agv_id.encode("utf-8"), nodes_csv, edges_csv)

    def pins(self) -> dict:
        return {"nodes": {k: sorted(v) for k, v in self._pinned_nodes.items() if v},
                "edges": {"->".join(k): sorted(v) for k, v in self._pinned_edges.items() if v}}

    def pinned_node_conflicts(self, node: str, exclude_agv: str | None = None) -> set[str]:
        return {o for o in self._pinned_nodes.get(node, ()) if o != exclude_agv}

    def pinned_edge_conflicts(self, src: str, dst: str, exclude_agv: str | None = None) -> set[str]:
        return {o for o in self._pinned_edges.get((src, dst), ()) if o != exclude_agv}

    # ---------- queries ----------

    def node_conflicts(self, node: str, span: Interval, exclude_agv: str | None = None) -> list[str]:
        ex_bytes = exclude_agv.encode("utf-8") if exclude_agv else None
        c_ptr = _LIB.reservation_table_node_conflicts_csv(
            self._ptr, node.encode("utf-8"), span.start, span.end, ex_bytes
        )
        if not c_ptr:
            return []
        raw = ctypes.string_at(c_ptr)
        _LIB.reservation_table_free_string(c_ptr)
        if not raw:
            return []
        parts = raw.decode("utf-8").split(",")
        return sorted([p for p in parts if p])

    def edge_conflicts(self, src: str, dst: str, span: Interval, exclude_agv: str | None = None) -> list[str]:
        ex_bytes = exclude_agv.encode("utf-8") if exclude_agv else None
        c_ptr = _LIB.reservation_table_edge_conflicts_csv(
            self._ptr, src.encode("utf-8"), dst.encode("utf-8"), span.start, span.end, ex_bytes
        )
        if not c_ptr:
            return []
        raw = ctypes.string_at(c_ptr)
        _LIB.reservation_table_free_string(c_ptr)
        if not raw:
            return []
        parts = raw.decode("utf-8").split(",")
        return sorted([p for p in parts if p])

    def is_node_free(self, node: str, span: Interval, capacity: int, exclude_agv: str | None = None) -> bool:
        ex_bytes = exclude_agv.encode("utf-8") if exclude_agv else None
        return bool(_LIB.reservation_table_is_node_free(
            self._ptr, node.encode("utf-8"), span.start, span.end, capacity, ex_bytes
        ))

    def is_edge_free(self, src: str, dst: str, span: Interval, capacity: int, exclude_agv: str | None = None) -> bool:
        ex_bytes = exclude_agv.encode("utf-8") if exclude_agv else None
        return bool(_LIB.reservation_table_is_edge_free(
            self._ptr, src.encode("utf-8"), dst.encode("utf-8"), span.start, span.end, capacity, ex_bytes
        ))

    def is_opposing_edge_free(self, src: str, dst: str, span: Interval, exclude_agv: str | None = None) -> bool:
        ex_bytes = exclude_agv.encode("utf-8") if exclude_agv else None
        return bool(_LIB.reservation_table_is_opposing_edge_free(
            self._ptr, src.encode("utf-8"), dst.encode("utf-8"), span.start, span.end, ex_bytes
        ))

    # ---------- writes ----------

    def reserve_node(self, node: str, span: Interval, agv_id: str) -> None:
        if len(span) == 0:
            return
        ok = _LIB.reservation_table_reserve_node(
            self._ptr, node.encode("utf-8"), span.start, span.end, agv_id.encode("utf-8")
        )
        if ok and self._track_shadow:
            self._node.setdefault(node, []).append((span, agv_id))

    def reserve_edge(self, src: str, dst: str, span: Interval, agv_id: str) -> None:
        if len(span) == 0:
            return
        ok = _LIB.reservation_table_reserve_edge(
            self._ptr, src.encode("utf-8"), dst.encode("utf-8"), span.start, span.end, agv_id.encode("utf-8")
        )
        if ok and self._track_shadow:
            self._edge.setdefault((src, dst), []).append((span, agv_id))

    def reserve_batch_script(self, agv_id: str, script: str) -> int:
        """Reserve multiple node/edge intervals for one AGV in a single native C-ABI call."""
        if not script:
            return 0
        n = int(_LIB.reservation_table_reserve_batch(
            self._ptr, agv_id.encode("utf-8"), script.encode("utf-8")
        ))
        if self._track_shadow and n > 0:
            for line in script.split("\n"):
                if not line:
                    continue
                parts = line.split("\t")
                if parts[0] == "N" and len(parts) == 4:
                    iv = Interval(int(parts[2]), int(parts[3]))
                    if len(iv) > 0:
                        self._node.setdefault(parts[1], []).append((iv, agv_id))
                elif parts[0] == "E" and len(parts) == 5:
                    iv = Interval(int(parts[3]), int(parts[4]))
                    if len(iv) > 0:
                        self._edge.setdefault((parts[1], parts[2]), []).append((iv, agv_id))
        return n

    def restore_from(self, other: "RustReservationTable") -> bool:
        """Restore native table state directly from `other` in a single Rust clone."""
        ok = bool(_LIB.reservation_table_restore_from(self._ptr, other._ptr))
        if ok:
            self._pinned_nodes = {k: set(v) for k, v in other._pinned_nodes.items()}
            self._pinned_edges = {k: set(v) for k, v in other._pinned_edges.items()}
            if self._track_shadow:
                self._node = {k: list(v) for k, v in other._node.items()}
                self._edge = {k: list(v) for k, v in other._edge.items()}
        return ok

    def clone_fast(self) -> "RustReservationTable":
        """Create a lightweight native Rust clone (`track_shadow=False`) in a single FFI call."""
        cls = type(self)
        inst = cls.__new__(cls)
        inst._ptr = _LIB.reservation_table_clone(self._ptr)
        if not inst._ptr:
            raise RuntimeError("reservation_table_clone failed")
        inst._track_shadow = False
        inst._pinned_nodes = {k: set(v) for k, v in self._pinned_nodes.items()}
        inst._pinned_edges = {k: set(v) for k, v in self._pinned_edges.items()}
        inst._node = {}
        inst._edge = {}
        return inst

    def release_node(self, node: str, span: Interval, agv_id: str) -> None:
        _LIB.reservation_table_release_node(
            self._ptr, node.encode("utf-8"), span.start, span.end, agv_id.encode("utf-8")
        )
        if self._track_shadow and node in self._node:
            self._node[node] = [
                (iv, o) for iv, o in self._node[node]
                if not (o == agv_id and iv.start == span.start and iv.end == span.end)
            ]
            if not self._node[node]:
                del self._node[node]

    def release_edge(self, src: str, dst: str, span: Interval, agv_id: str) -> None:
        _LIB.reservation_table_release_edge(
            self._ptr, src.encode("utf-8"), dst.encode("utf-8"), span.start, span.end, agv_id.encode("utf-8")
        )
        key = (src, dst)
        if self._track_shadow and key in self._edge:
            self._edge[key] = [
                (iv, o) for iv, o in self._edge[key]
                if not (o == agv_id and iv.start == span.start and iv.end == span.end)
            ]
            if not self._edge[key]:
                del self._edge[key]

    def release_all(self, agv_id: str) -> None:
        _LIB.reservation_table_release_all(self._ptr, agv_id.encode("utf-8"))
        if not self._track_shadow:
            return
        for k in list(self._node.keys()):
            kept = [(iv, o) for iv, o in self._node[k] if o != agv_id]
            if kept:
                self._node[k] = kept
            else:
                del self._node[k]
        for k in list(self._edge.keys()):
            kept = [(iv, o) for iv, o in self._edge[k] if o != agv_id]
            if kept:
                self._edge[k] = kept
            else:
                del self._edge[k]

    def purge_before(self, step: int) -> None:
        _LIB.reservation_table_purge_before(self._ptr, step)
        if not self._track_shadow:
            return
        for k in list(self._node.keys()):
            kept = [(iv, o) for iv, o in self._node[k] if iv.end > step]
            if kept:
                self._node[k] = kept
            else:
                del self._node[k]
        for k in list(self._edge.keys()):
            kept = [(iv, o) for iv, o in self._edge[k] if iv.end > step]
            if kept:
                self._edge[k] = kept
            else:
                del self._edge[k]

    def total_reservations(self) -> int:
        return int(_LIB.reservation_table_total_reservations(self._ptr))

    def utilization(self, horizon: int, n_nodes: int, n_edges: int) -> float:
        return float(_LIB.reservation_table_utilization(self._ptr, horizon, n_nodes, n_edges))

    def snapshot(self, lo: int, hi: int) -> dict[str, list[tuple[int, int, str]]]:
        """Return node occupancy over [lo, hi) from Python shadow tables when tracking is enabled."""
        if not self._track_shadow:
            return {}
        span = Interval(lo, hi)
        out: dict[str, list[tuple[int, int, str]]] = {}
        for node, entries in self._node.items():
            hits = [(iv.start, iv.end, owner) for iv, owner in entries if iv.overlaps(span)]
            if hits:
                out[node] = sorted(hits)
        for node, owners in self._pinned_nodes.items():
            for owner in owners:
                out.setdefault(node, []).append((lo, hi, owner))
        return out

    def begin_transaction(self, agv_id: str) -> "RustReservationTransaction":
        """Begins an atomic reservation transaction for an AGV."""
        return RustReservationTransaction(agv_id)


class RustReservationTransaction:
    """Atomic multi-hop reservation transaction.

    Stages multiple node/edge reservations and validates global consistency.
    Guarantees all-or-nothing atomic commit into `RustReservationTable`.
    """

    def __init__(self, agv_id: str) -> None:
        if _LIB is None:
            raise RuntimeError("Native reservation_core library not loaded.")
        self.agv_id = agv_id
        self._ptr = _LIB.transaction_begin(agv_id.encode("utf-8"))
        self._committed = False
        self._rolled_back = False
        self._staged_nodes: list[tuple[str, Interval]] = []
        self._staged_edges: list[tuple[str, str, Interval]] = []

    def __del__(self) -> None:
        if hasattr(self, "_ptr") and self._ptr and _LIB:
            _LIB.transaction_destroy(self._ptr)
            self._ptr = None

    def stage_node(self, node: str, span: Interval, capacity: int = 1) -> None:
        """Stages a node occupancy window."""
        if len(span) == 0:
            return
        ok = _LIB.transaction_stage_node(
            self._ptr, node.encode("utf-8"), span.start, span.end, capacity
        )
        if ok:
            self._staged_nodes.append((node, span))

    def stage_edge(self, src: str, dst: str, span: Interval, capacity: int = 1) -> None:
        """Stages an edge traversal window."""
        if len(span) == 0:
            return
        ok = _LIB.transaction_stage_edge(
            self._ptr, src.encode("utf-8"), dst.encode("utf-8"), span.start, span.end, capacity
        )
        if ok:
            self._staged_edges.append((src, dst, span))

    @property
    def staged_count(self) -> int:
        return int(_LIB.transaction_staged_count(self._ptr))

    def can_commit(self, table: RustReservationTable) -> bool:
        """Verifies if all staged reservations are free in `table` without committing."""
        return bool(_LIB.transaction_can_commit(self._ptr, table._ptr))

    def commit(self, table: RustReservationTable) -> bool:
        """Atomically commits all staged reservations into `table`.

        Returns True if successful. If any conflict exists, returns False and rolls back.
        """
        ok = bool(_LIB.transaction_commit(self._ptr, table._ptr))
        if ok:
            self._committed = True
            if table._track_shadow:
                for node, span in self._staged_nodes:
                    table._node.setdefault(node, []).append((span, self.agv_id))
                for src, dst, span in self._staged_edges:
                    table._edge.setdefault((src, dst), []).append((span, self.agv_id))
        else:
            self._rolled_back = True
        return ok

    def clone_fast(self) -> "RustReservationTable":
        """Create a lightweight native Rust clone (`track_shadow=False`) in a single FFI call."""
        cls = type(self)
        inst = cls.__new__(cls)
        inst._ptr = _LIB.reservation_table_clone(self._ptr)
        if not inst._ptr:
            raise RuntimeError("reservation_table_clone failed")
        inst._track_shadow = False
        inst._pinned_nodes = {}
        inst._pinned_edges = {}
        inst._node = {}
        inst._edge = {}
        return inst

    def rollback(self) -> None:
        """Discards all staged reservations."""
        _LIB.transaction_rollback(self._ptr)
        self._staged_nodes.clear()
        self._staged_edges.clear()
        self._rolled_back = True

    def __enter__(self) -> "RustReservationTransaction":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.rollback()


class RustExecutionCommitmentLedger:
    """Ledger tracking active non-preemptible execution commitments."""

    def __init__(self) -> None:
        if _LIB is None:
            raise RuntimeError("Native reservation_core library not loaded.")
        self._ptr = _LIB.ledger_create()

    def __del__(self) -> None:
        if hasattr(self, "_ptr") and self._ptr and _LIB:
            _LIB.ledger_destroy(self._ptr)
            self._ptr = None

    def commit_hop(self, agv_id: str, src: str, dst: str, span: Interval) -> bool:
        return bool(_LIB.ledger_commit_hop(
            self._ptr,
            agv_id.encode("utf-8"),
            src.encode("utf-8"),
            dst.encode("utf-8"),
            span.start,
            span.end,
        ))

    def complete_hop(self, agv_id: str, node: str) -> bool:
        return bool(_LIB.ledger_complete_hop(
            self._ptr,
            agv_id.encode("utf-8"),
            node.encode("utf-8"),
        ))

    def has_active_commitment(self, agv_id: str) -> bool:
        return bool(_LIB.ledger_has_active_commitment(self._ptr, agv_id.encode("utf-8")))

    @property
    def active_count(self) -> int:
        return int(_LIB.ledger_active_count(self._ptr))

