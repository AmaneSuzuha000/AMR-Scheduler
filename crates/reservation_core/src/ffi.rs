//! C-ABI Foreign Function Interface for `reservation_core`.
//!
//! Exposes an ultra-fast, zero-GC native interface that can be loaded dynamically
//! by Python (`ctypes` / `cffi`), C#, and C++ without external build script requirements.
//!
//! Every exported function is protected with `std::panic::catch_unwind` to prevent
//! unexpected internal panics from crossing the FFI boundary and aborting host processes.

use crate::interval::Interval;
use crate::table::ReservationTable;
use crate::transaction::{ExecutionCommitmentLedger, ReservationTransaction};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

macro_rules! ffi_guard {
    ($fallback:expr, $body:expr) => {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body)) {
            Ok(v) => v,
            Err(_) => $fallback,
        }
    };
}

/// Helper to convert a C string to an `&str`.
#[inline(always)]
unsafe fn c_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        None
    } else {
        CStr::from_ptr(ptr).to_str().ok()
    }
}

/// Helper to convert an optional C string to an `Option<&str>`.
#[inline(always)]
unsafe fn c_to_opt_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        None
    } else {
        let s = CStr::from_ptr(ptr).to_str().ok()?;
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    }
}

/// Creates a new `ReservationTable` on the heap.
#[no_mangle]
pub extern "C" fn reservation_table_create() -> *mut ReservationTable {
    ffi_guard!(std::ptr::null_mut(), {
        Box::into_raw(Box::new(ReservationTable::new()))
    })
}

/// Destroys a `ReservationTable` created by [`reservation_table_create`].
#[no_mangle]
pub unsafe extern "C" fn reservation_table_destroy(tbl: *mut ReservationTable) {
    ffi_guard!((), {
        if !tbl.is_null() {
            drop(Box::from_raw(tbl));
        }
    });
}

/// Clears all scheduled reservations.
#[no_mangle]
pub unsafe extern "C" fn reservation_table_clear(tbl: *mut ReservationTable) {
    ffi_guard!((), {
        if let Some(table) = tbl.as_mut() {
            table.clear();
        }
    });
}

/// Clears all reservations and pinned occupancy.
#[no_mangle]
pub unsafe extern "C" fn reservation_table_reset_all(tbl: *mut ReservationTable) {
    ffi_guard!((), {
        if let Some(table) = tbl.as_mut() {
            table.reset_all();
        }
    });
}

/// Inserts a node reservation `[start, end)`.
#[no_mangle]
pub unsafe extern "C" fn reservation_table_reserve_node(
    tbl: *mut ReservationTable,
    node: *const c_char,
    start: i64,
    end: i64,
    agv_id: *const c_char,
) -> bool {
    ffi_guard!(false, {
        let table = match tbl.as_mut() {
            Some(t) => t,
            None => return false,
        };
        let node_str = match c_to_str(node) {
            Some(s) => s,
            None => return false,
        };
        let agv_str = match c_to_str(agv_id) {
            Some(s) => s,
            None => return false,
        };
        let span = match Interval::try_new(start, end) {
            Ok(s) => s,
            Err(_) => return false,
        };

        table.reserve_node(node_str, span, agv_str);
        true
    })
}

/// Inserts an edge reservation `[start, end)`.
#[no_mangle]
pub unsafe extern "C" fn reservation_table_reserve_edge(
    tbl: *mut ReservationTable,
    src: *const c_char,
    dst: *const c_char,
    start: i64,
    end: i64,
    agv_id: *const c_char,
) -> bool {
    ffi_guard!(false, {
        let table = match tbl.as_mut() {
            Some(t) => t,
            None => return false,
        };
        let src_str = match c_to_str(src) {
            Some(s) => s,
            None => return false,
        };
        let dst_str = match c_to_str(dst) {
            Some(s) => s,
            None => return false,
        };
        let agv_str = match c_to_str(agv_id) {
            Some(s) => s,
            None => return false,
        };
        let span = match Interval::try_new(start, end) {
            Ok(s) => s,
            Err(_) => return false,
        };

        table.reserve_edge(src_str, dst_str, span, agv_str);
        true
    })
}

/// Checks whether a node is free over `[start, end)`.
#[no_mangle]
pub unsafe extern "C" fn reservation_table_is_node_free(
    tbl: *const ReservationTable,
    node: *const c_char,
    start: i64,
    end: i64,
    capacity: usize,
    exclude_agv: *const c_char,
) -> bool {
    ffi_guard!(false, {
        let table = match tbl.as_ref() {
            Some(t) => t,
            None => return false,
        };
        let node_str = match c_to_str(node) {
            Some(s) => s,
            None => return false,
        };
        let span = match Interval::try_new(start, end) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let exclude = c_to_opt_str(exclude_agv);

        table.is_node_free(node_str, span, capacity, exclude)
    })
}

/// Checks whether an edge is free over `[start, end)`.
#[no_mangle]
pub unsafe extern "C" fn reservation_table_is_edge_free(
    tbl: *const ReservationTable,
    src: *const c_char,
    dst: *const c_char,
    start: i64,
    end: i64,
    capacity: usize,
    exclude_agv: *const c_char,
) -> bool {
    ffi_guard!(false, {
        let table = match tbl.as_ref() {
            Some(t) => t,
            None => return false,
        };
        let src_str = match c_to_str(src) {
            Some(s) => s,
            None => return false,
        };
        let dst_str = match c_to_str(dst) {
            Some(s) => s,
            None => return false,
        };
        let span = match Interval::try_new(start, end) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let exclude = c_to_opt_str(exclude_agv);

        table.is_edge_free(src_str, dst_str, span, capacity, exclude)
    })
}

/// Checks whether opposing edge (dst -> src) is free over `[start, end)`.
#[no_mangle]
pub unsafe extern "C" fn reservation_table_is_opposing_edge_free(
    tbl: *const ReservationTable,
    src: *const c_char,
    dst: *const c_char,
    start: i64,
    end: i64,
    exclude_agv: *const c_char,
) -> bool {
    ffi_guard!(false, {
        let table = match tbl.as_ref() {
            Some(t) => t,
            None => return false,
        };
        let src_str = match c_to_str(src) {
            Some(s) => s,
            None => return false,
        };
        let dst_str = match c_to_str(dst) {
            Some(s) => s,
            None => return false,
        };
        let span = match Interval::try_new(start, end) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let exclude = c_to_opt_str(exclude_agv);

        table.is_opposing_edge_free(src_str, dst_str, span, exclude)
    })
}

/// Releases one vehicle's reservation on a specific node.
#[no_mangle]
pub unsafe extern "C" fn reservation_table_release_node(
    tbl: *mut ReservationTable,
    node: *const c_char,
    start: i64,
    end: i64,
    agv_id: *const c_char,
) -> bool {
    ffi_guard!(false, {
        let table = match tbl.as_mut() {
            Some(t) => t,
            None => return false,
        };
        let node_str = match c_to_str(node) {
            Some(s) => s,
            None => return false,
        };
        let agv_str = match c_to_str(agv_id) {
            Some(s) => s,
            None => return false,
        };
        let span = match Interval::try_new(start, end) {
            Ok(s) => s,
            Err(_) => return false,
        };

        table.release_node(node_str, span, agv_str);
        true
    })
}

/// Releases one vehicle's reservation on a specific edge.
#[no_mangle]
pub unsafe extern "C" fn reservation_table_release_edge(
    tbl: *mut ReservationTable,
    src: *const c_char,
    dst: *const c_char,
    start: i64,
    end: i64,
    agv_id: *const c_char,
) -> bool {
    ffi_guard!(false, {
        let table = match tbl.as_mut() {
            Some(t) => t,
            None => return false,
        };
        let src_str = match c_to_str(src) {
            Some(s) => s,
            None => return false,
        };
        let dst_str = match c_to_str(dst) {
            Some(s) => s,
            None => return false,
        };
        let agv_str = match c_to_str(agv_id) {
            Some(s) => s,
            None => return false,
        };
        let span = match Interval::try_new(start, end) {
            Ok(s) => s,
            Err(_) => return false,
        };

        table.release_edge(src_str, dst_str, span, agv_str);
        true
    })
}

/// Releases all reservations belonging to `agv_id`.
#[no_mangle]
pub unsafe extern "C" fn reservation_table_release_all(
    tbl: *mut ReservationTable,
    agv_id: *const c_char,
) -> bool {
    ffi_guard!(false, {
        let table = match tbl.as_mut() {
            Some(t) => t,
            None => return false,
        };
        let agv_str = match c_to_str(agv_id) {
            Some(s) => s,
            None => return false,
        };

        table.release_all(agv_str);
        true
    })
}

/// Purges reservations ending before `step`.
#[no_mangle]
pub unsafe extern "C" fn reservation_table_purge_before(tbl: *mut ReservationTable, step: i64) {
    ffi_guard!((), {
        if let Some(table) = tbl.as_mut() {
            table.purge_before(step);
        }
    });
}

/// Transfers physical pinned occupancy for an AGV.
/// `nodes_csv` and `edges_csv` are comma-separated strings (edges formatted as `src->dst,src2->dst2`).
#[no_mangle]
pub unsafe extern "C" fn reservation_table_transfer_pins(
    tbl: *mut ReservationTable,
    agv_id: *const c_char,
    nodes_csv: *const c_char,
    edges_csv: *const c_char,
) -> bool {
    ffi_guard!(false, {
        let table = match tbl.as_mut() {
            Some(t) => t,
            None => return false,
        };
        let agv_str = match c_to_str(agv_id) {
            Some(s) => s,
            None => return false,
        };

        let mut nodes = Vec::new();
        if let Some(n_str) = c_to_opt_str(nodes_csv) {
            for n in n_str.split(',') {
                let tr = n.trim();
                if !tr.is_empty() {
                    nodes.push(tr.to_string());
                }
            }
        }

        let mut edges = Vec::new();
        if let Some(e_str) = c_to_opt_str(edges_csv) {
            for e in e_str.split(',') {
                let tr = e.trim();
                if let Some((s, d)) = tr.split_once("->") {
                    let st = s.trim();
                    let dt = d.trim();
                    if !st.is_empty() && !dt.is_empty() {
                        edges.push((st.to_string(), dt.to_string()));
                    }
                }
            }
        }

        table.transfer_pins(agv_str, &nodes, &edges);
        true
    })
}

/// Returns total number of reservations across nodes and edges.
#[no_mangle]
pub unsafe extern "C" fn reservation_table_total_reservations(tbl: *const ReservationTable) -> usize {
    ffi_guard!(0, {
        match tbl.as_ref() {
            Some(t) => t.total_reservations(),
            None => 0,
        }
    })
}

/// Computes utilization metric.
#[no_mangle]
pub unsafe extern "C" fn reservation_table_utilization(
    tbl: *const ReservationTable,
    horizon: i64,
    n_nodes: usize,
    n_edges: usize,
) -> f64 {
    ffi_guard!(0.0, {
        match tbl.as_ref() {
            Some(t) => t.utilization(horizon, n_nodes, n_edges),
            None => 0.0,
        }
    })
}

/// Returns conflicts on a node as a comma-separated C string.
/// Caller must free the returned string using [`reservation_table_free_string`].
#[no_mangle]
pub unsafe extern "C" fn reservation_table_node_conflicts_csv(
    tbl: *const ReservationTable,
    node: *const c_char,
    start: i64,
    end: i64,
    exclude_agv: *const c_char,
) -> *mut c_char {
    ffi_guard!(std::ptr::null_mut(), {
        let table = match tbl.as_ref() {
            Some(t) => t,
            None => return std::ptr::null_mut(),
        };
        let node_str = match c_to_str(node) {
            Some(s) => s,
            None => return std::ptr::null_mut(),
        };
        let span = match Interval::try_new(start, end) {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };
        let exclude = c_to_opt_str(exclude_agv);

        let conflicts = table.node_conflicts(node_str, span, exclude);
        let csv = conflicts.join(",");
        match CString::new(csv) {
            Ok(cs) => cs.into_raw(),
            Err(_) => std::ptr::null_mut(),
        }
    })
}

/// Returns conflicts on an edge as a comma-separated C string.
/// Caller must free the returned string using [`reservation_table_free_string`].
#[no_mangle]
pub unsafe extern "C" fn reservation_table_edge_conflicts_csv(
    tbl: *const ReservationTable,
    src: *const c_char,
    dst: *const c_char,
    start: i64,
    end: i64,
    exclude_agv: *const c_char,
) -> *mut c_char {
    ffi_guard!(std::ptr::null_mut(), {
        let table = match tbl.as_ref() {
            Some(t) => t,
            None => return std::ptr::null_mut(),
        };
        let src_str = match c_to_str(src) {
            Some(s) => s,
            None => return std::ptr::null_mut(),
        };
        let dst_str = match c_to_str(dst) {
            Some(s) => s,
            None => return std::ptr::null_mut(),
        };
        let span = match Interval::try_new(start, end) {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };
        let exclude = c_to_opt_str(exclude_agv);

        let conflicts = table.edge_conflicts(src_str, dst_str, span, exclude);
        let csv = conflicts.join(",");
        match CString::new(csv) {
            Ok(cs) => cs.into_raw(),
            Err(_) => std::ptr::null_mut(),
        }
    })
}

/// Frees a string allocated by Rust.
#[no_mangle]
pub unsafe extern "C" fn reservation_table_free_string(s: *mut c_char) {
    ffi_guard!((), {
        if !s.is_null() {
            drop(CString::from_raw(s));
        }
    });
}

// ---------------------------------------------------------------------------
// Transaction C-ABI
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn transaction_begin(agv_id: *const c_char) -> *mut ReservationTransaction {
    ffi_guard!(std::ptr::null_mut(), {
        let agv = match c_to_str(agv_id) {
            Some(s) => s,
            None => return std::ptr::null_mut(),
        };
        Box::into_raw(Box::new(ReservationTransaction::begin(agv)))
    })
}

#[no_mangle]
pub unsafe extern "C" fn transaction_destroy(tx: *mut ReservationTransaction) {
    ffi_guard!((), {
        if !tx.is_null() {
            drop(Box::from_raw(tx));
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn transaction_stage_node(
    tx: *mut ReservationTransaction,
    node: *const c_char,
    start: i64,
    end: i64,
    capacity: usize,
) -> bool {
    ffi_guard!(false, {
        let transaction = match tx.as_mut() {
            Some(t) => t,
            None => return false,
        };
        let node_str = match c_to_str(node) {
            Some(s) => s,
            None => return false,
        };
        let span = match Interval::try_new(start, end) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let cap = if capacity == 0 { 1 } else { capacity };
        transaction.stage_node_with_cap(node_str, span, cap);
        true
    })
}

#[no_mangle]
pub unsafe extern "C" fn transaction_stage_edge(
    tx: *mut ReservationTransaction,
    src: *const c_char,
    dst: *const c_char,
    start: i64,
    end: i64,
    capacity: usize,
) -> bool {
    ffi_guard!(false, {
        let transaction = match tx.as_mut() {
            Some(t) => t,
            None => return false,
        };
        let src_str = match c_to_str(src) {
            Some(s) => s,
            None => return false,
        };
        let dst_str = match c_to_str(dst) {
            Some(s) => s,
            None => return false,
        };
        let span = match Interval::try_new(start, end) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let cap = if capacity == 0 { 1 } else { capacity };
        transaction.stage_edge_with_cap(src_str, dst_str, span, cap);
        true
    })
}

#[no_mangle]
pub unsafe extern "C" fn transaction_staged_count(tx: *const ReservationTransaction) -> usize {
    ffi_guard!(0, {
        match tx.as_ref() {
            Some(t) => t.staged_count(),
            None => 0,
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn transaction_can_commit(
    tx: *const ReservationTransaction,
    tbl: *const ReservationTable,
) -> bool {
    ffi_guard!(false, {
        let transaction = match tx.as_ref() {
            Some(t) => t,
            None => return false,
        };
        let table = match tbl.as_ref() {
            Some(t) => t,
            None => return false,
        };
        transaction.can_commit(table)
    })
}

#[no_mangle]
pub unsafe extern "C" fn transaction_commit(
    tx: *mut ReservationTransaction,
    tbl: *mut ReservationTable,
) -> bool {
    ffi_guard!(false, {
        let transaction = match tx.as_mut() {
            Some(t) => t,
            None => return false,
        };
        let table = match tbl.as_mut() {
            Some(t) => t,
            None => return false,
        };
        transaction.commit(table).is_ok()
    })
}

#[no_mangle]
pub unsafe extern "C" fn transaction_rollback(tx: *mut ReservationTransaction) {
    ffi_guard!((), {
        if let Some(transaction) = tx.as_mut() {
            transaction.rollback();
        }
    });
}

// ---------------------------------------------------------------------------
// Execution Commitment Ledger C-ABI
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ledger_create() -> *mut ExecutionCommitmentLedger {
    ffi_guard!(std::ptr::null_mut(), {
        Box::into_raw(Box::new(ExecutionCommitmentLedger::new()))
    })
}

#[no_mangle]
pub unsafe extern "C" fn ledger_destroy(ledger: *mut ExecutionCommitmentLedger) {
    ffi_guard!((), {
        if !ledger.is_null() {
            drop(Box::from_raw(ledger));
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn ledger_commit_hop(
    ledger: *mut ExecutionCommitmentLedger,
    agv_id: *const c_char,
    src: *const c_char,
    dst: *const c_char,
    start: i64,
    end: i64,
) -> bool {
    ffi_guard!(false, {
        let l = match ledger.as_mut() {
            Some(x) => x,
            None => return false,
        };
        let agv = match c_to_str(agv_id) {
            Some(s) => s,
            None => return false,
        };
        let s_str = match c_to_str(src) {
            Some(s) => s,
            None => return false,
        };
        let d_str = match c_to_str(dst) {
            Some(s) => s,
            None => return false,
        };
        let span = match Interval::try_new(start, end) {
            Ok(s) => s,
            Err(_) => return false,
        };
        l.commit_hop(agv, s_str, d_str, span);
        true
    })
}

#[no_mangle]
pub unsafe extern "C" fn ledger_complete_hop(
    ledger: *mut ExecutionCommitmentLedger,
    agv_id: *const c_char,
    node: *const c_char,
) -> bool {
    ffi_guard!(false, {
        let l = match ledger.as_mut() {
            Some(x) => x,
            None => return false,
        };
        let agv = match c_to_str(agv_id) {
            Some(s) => s,
            None => return false,
        };
        let n_str = match c_to_str(node) {
            Some(s) => s,
            None => return false,
        };
        l.complete_hop(agv, n_str);
        true
    })
}

#[no_mangle]
pub unsafe extern "C" fn ledger_has_active_commitment(
    ledger: *const ExecutionCommitmentLedger,
    agv_id: *const c_char,
) -> bool {
    ffi_guard!(false, {
        let l = match ledger.as_ref() {
            Some(x) => x,
            None => return false,
        };
        let agv = match c_to_str(agv_id) {
            Some(s) => s,
            None => return false,
        };
        l.has_active_commitment(agv)
    })
}

#[no_mangle]
pub unsafe extern "C" fn ledger_active_count(ledger: *const ExecutionCommitmentLedger) -> usize {
    ffi_guard!(0, {
        match ledger.as_ref() {
            Some(x) => x.active_count(),
            None => 0,
        }
    })
}
