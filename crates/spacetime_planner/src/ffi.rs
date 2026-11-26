//! C-ABI Foreign Function Interface for `spacetime_planner`.

use crate::astar::{CpsSearchParams, CpsSearchResult, IndexedPath, SpacetimeAStarPlanner};
use crate::graph::PlannerGraph;
use crate::sipp::SippPlanner;
use crate::slot_field::{SlotCongestionField, SlotFieldConfig};
use reservation_core::{Interval, ReservationTable};
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

unsafe fn c_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        None
    } else {
        CStr::from_ptr(ptr).to_str().ok()
    }
}

#[no_mangle]
pub extern "C" fn planner_graph_create() -> *mut PlannerGraph {
    ffi_guard!(std::ptr::null_mut(), {
        Box::into_raw(Box::new(PlannerGraph::new()))
    })
}

#[no_mangle]
pub unsafe extern "C" fn planner_graph_destroy(graph: *mut PlannerGraph) {
    ffi_guard!((), {
        if !graph.is_null() {
            drop(Box::from_raw(graph));
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn planner_graph_add_node(
    graph: *mut PlannerGraph,
    name: *const c_char,
    capacity: usize,
) -> u16 {
    ffi_guard!(u16::MAX, {
        let g = match graph.as_mut() {
            Some(ptr) => ptr,
            None => return u16::MAX,
        };
        let name_str = match c_to_str(name) {
            Some(s) => s,
            None => return u16::MAX,
        };
        g.set_node_capacity(name_str, capacity)
    })
}

#[no_mangle]
pub unsafe extern "C" fn planner_graph_add_edge(
    graph: *mut PlannerGraph,
    src: *const c_char,
    dst: *const c_char,
    traversal_steps: u32,
    capacity: usize,
    physical_capacity: usize,
) -> bool {
    ffi_guard!(false, {
        let g = match graph.as_mut() {
            Some(ptr) => ptr,
            None => return false,
        };
        let s = match c_to_str(src) {
            Some(val) => val,
            None => return false,
        };
        let d = match c_to_str(dst) {
            Some(val) => val,
            None => return false,
        };
        g.add_edge(s, d, traversal_steps, capacity, physical_capacity);
        true
    })
}

#[no_mangle]
pub unsafe extern "C" fn planner_graph_clear_traffic(graph: *mut PlannerGraph) {
    ffi_guard!((), {
        if let Some(g) = graph.as_mut() {
            g.clear_traffic_penalties();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn planner_graph_set_traffic_batch(
    graph: *mut PlannerGraph,
    script: *const c_char,
) -> usize {
    ffi_guard!(0, {
        let g = match graph.as_mut() {
            Some(ptr) => ptr,
            None => return 0,
        };
        let s = match c_to_str(script) {
            Some(v) => v,
            None => return 0,
        };
        g.set_traffic_batch_script(s)
    })
}

#[no_mangle]
pub unsafe extern "C" fn planner_graph_set_traffic_cutoff(
    graph: *mut PlannerGraph,
    cutoff_step: u32,
) {
    ffi_guard!((), {
        if let Some(g) = graph.as_mut() {
            g.set_traffic_cutoff(cutoff_step);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn planner_graph_clear_guides(graph: *mut PlannerGraph) {
    ffi_guard!((), {
        if let Some(g) = graph.as_mut() {
            g.clear_agv_guides();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn planner_graph_set_guide_batch(
    graph: *mut PlannerGraph,
    script: *const c_char,
) -> usize {
    ffi_guard!(0, {
        let g = match graph.as_mut() {
            Some(ptr) => ptr,
            None => return 0,
        };
        let s = match c_to_str(script) {
            Some(v) => v,
            None => return 0,
        };
        g.set_agv_guide_script(s)
    })
}

#[no_mangle]
pub extern "C" fn spacetime_planner_create(max_expansions: usize) -> *mut SpacetimeAStarPlanner {
    ffi_guard!(std::ptr::null_mut(), {
        Box::into_raw(Box::new(SpacetimeAStarPlanner::new(max_expansions)))
    })
}

#[no_mangle]
pub unsafe extern "C" fn spacetime_planner_destroy(planner: *mut SpacetimeAStarPlanner) {
    ffi_guard!((), {
        if !planner.is_null() {
            drop(Box::from_raw(planner));
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn spacetime_planner_plan(
    planner: *const SpacetimeAStarPlanner,
    graph: *const PlannerGraph,
    reservations: *const ReservationTable,
    start_node: *const c_char,
    goal_node: *const c_char,
    start_step: u32,
    agv_id: *const c_char,
    horizon: u32,
    out_nodes_csv: *mut *mut c_char,
    out_arrivals_csv: *mut *mut c_char,
    out_wait_steps: *mut u32,
) -> bool {
    ffi_guard!(false, {
        let p = match planner.as_ref() {
            Some(ptr) => ptr,
            None => return false,
        };
        let g = match graph.as_ref() {
            Some(ptr) => ptr,
            None => return false,
        };
        let r = match reservations.as_ref() {
            Some(ptr) => ptr,
            None => return false,
        };
        let start = match c_to_str(start_node) {
            Some(val) => val,
            None => return false,
        };
        let goal = match c_to_str(goal_node) {
            Some(val) => val,
            None => return false,
        };
        let agv = match c_to_str(agv_id) {
            Some(val) => val,
            None => return false,
        };

        let path_opt = p.plan(g, r, start, goal, start_step, agv, horizon);
        let path = match path_opt {
            Some(res) => res,
            None => return false,
        };

        let nodes_csv = path.nodes.join(",");
        let arrivals_csv = path
            .arrival_steps
            .iter()
            .map(|a| a.to_string())
            .collect::<Vec<_>>()
            .join(",");

        if !out_wait_steps.is_null() {
            *out_wait_steps = path.wait_steps;
        }

        if !out_nodes_csv.is_null() {
            match CString::new(nodes_csv) {
                Ok(cs) => *out_nodes_csv = cs.into_raw(),
                Err(_) => return false,
            }
        }

        if !out_arrivals_csv.is_null() {
            match CString::new(arrivals_csv) {
                Ok(cs) => *out_arrivals_csv = cs.into_raw(),
                Err(_) => return false,
            }
        }

        true
    })
}

#[no_mangle]
pub extern "C" fn sipp_planner_create(max_expansions: usize) -> *mut crate::sipp::SippPlanner {
    ffi_guard!(std::ptr::null_mut(), {
        Box::into_raw(Box::new(crate::sipp::SippPlanner::new(max_expansions)))
    })
}

#[no_mangle]
pub unsafe extern "C" fn sipp_planner_destroy(planner: *mut crate::sipp::SippPlanner) {
    ffi_guard!((), {
        if !planner.is_null() {
            drop(Box::from_raw(planner));
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn sipp_planner_plan(
    planner: *const crate::sipp::SippPlanner,
    graph: *const PlannerGraph,
    reservations: *const ReservationTable,
    start_node: *const c_char,
    goal_node: *const c_char,
    start_step: u32,
    agv_id: *const c_char,
    horizon: u32,
    out_nodes_csv: *mut *mut c_char,
    out_arrivals_csv: *mut *mut c_char,
    out_wait_steps: *mut u32,
) -> bool {
    ffi_guard!(false, {
        let p = match planner.as_ref() {
            Some(ptr) => ptr,
            None => return false,
        };
        let g = match graph.as_ref() {
            Some(ptr) => ptr,
            None => return false,
        };
        let r = match reservations.as_ref() {
            Some(ptr) => ptr,
            None => return false,
        };
        let start = match c_to_str(start_node) {
            Some(val) => val,
            None => return false,
        };
        let goal = match c_to_str(goal_node) {
            Some(val) => val,
            None => return false,
        };
        let agv = match c_to_str(agv_id) {
            Some(val) => val,
            None => return false,
        };

        let path_opt = p.plan(g, r, start, goal, start_step, agv, horizon);
        let path = match path_opt {
            Some(res) => res,
            None => return false,
        };

        let nodes_csv = path.nodes.join(",");
        let arrivals_csv = path
            .arrival_steps
            .iter()
            .map(|a| a.to_string())
            .collect::<Vec<_>>()
            .join(",");

        if !out_wait_steps.is_null() {
            *out_wait_steps = path.wait_steps;
        }

        if !out_nodes_csv.is_null() {
            match CString::new(nodes_csv) {
                Ok(cs) => *out_nodes_csv = cs.into_raw(),
                Err(_) => return false,
            }
        }

        if !out_arrivals_csv.is_null() {
            match CString::new(arrivals_csv) {
                Ok(cs) => *out_arrivals_csv = cs.into_raw(),
                Err(_) => return false,
            }
        }

        true
    })
}

#[no_mangle]
pub unsafe extern "C" fn spacetime_planner_free_string(s: *mut c_char) {
    ffi_guard!((), {
        if !s.is_null() {
            drop(CString::from_raw(s));
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn planner_graph_node_index(
    graph: *const PlannerGraph,
    name: *const c_char,
) -> u16 {
    ffi_guard!(u16::MAX, {
        let g = match graph.as_ref() {
            Some(ptr) => ptr,
            None => return u16::MAX,
        };
        let n = match c_to_str(name) {
            Some(s) => s,
            None => return u16::MAX,
        };
        g.node_index(n).unwrap_or(u16::MAX)
    })
}

#[no_mangle]
pub unsafe extern "C" fn planner_graph_set_traffic_indexed(
    graph: *mut PlannerGraph,
    srcs: *const u16,
    dsts: *const u16,
    pens_q3: *const u32,
    len: usize,
    cutoff_step: u32,
) -> usize {
    ffi_guard!(0, {
        let g = match graph.as_mut() {
            Some(ptr) => ptr,
            None => return 0,
        };
        g.clear_traffic_penalties();
        if len == 0 || srcs.is_null() || dsts.is_null() || pens_q3.is_null() {
            return 0;
        }
        let s_sl = std::slice::from_raw_parts(srcs, len);
        let d_sl = std::slice::from_raw_parts(dsts, len);
        let p_sl = std::slice::from_raw_parts(pens_q3, len);
        let mut updated = 0usize;
        for i in 0..len {
            if g.set_edge_traffic_penalty_indexed(s_sl[i], d_sl[i], p_sl[i]) {
                updated += 1;
            }
        }
        g.set_traffic_cutoff(cutoff_step);
        updated
    })
}

unsafe fn write_indexed_path_buf(
    path: Option<IndexedPath>,
    out_nodes: *mut u16,
    out_arrivals: *mut u32,
    max_len: usize,
    out_wait_steps: *mut u32,
) -> i32 {
    let p = match path {
        Some(res) => res,
        None => return -1,
    };
    if !out_wait_steps.is_null() {
        *out_wait_steps = p.wait_steps;
    }
    let n = p.nodes.len();
    if n == 0 {
        return 0;
    }
    if out_nodes.is_null() || out_arrivals.is_null() || n > max_len {
        return -2;
    }
    std::ptr::copy_nonoverlapping(p.nodes.as_ptr(), out_nodes, n);
    std::ptr::copy_nonoverlapping(p.arrivals.as_ptr(), out_arrivals, n);
    n as i32
}

#[no_mangle]
pub unsafe extern "C" fn spacetime_planner_plan_indexed(
    planner: *const SpacetimeAStarPlanner,
    graph: *const PlannerGraph,
    reservations: *const ReservationTable,
    start_idx: u16,
    goal_idx: u16,
    start_step: u32,
    agv_id: *const c_char,
    horizon: u32,
    out_nodes: *mut u16,
    out_arrivals: *mut u32,
    max_len: usize,
    out_wait_steps: *mut u32,
) -> i32 {
    ffi_guard!(-1, {
        let p = match planner.as_ref() {
            Some(ptr) => ptr,
            None => return -1,
        };
        let g = match graph.as_ref() {
            Some(ptr) => ptr,
            None => return -1,
        };
        let r = match reservations.as_ref() {
            Some(ptr) => ptr,
            None => return -1,
        };
        let agv = match c_to_str(agv_id) {
            Some(val) => val,
            None => return -1,
        };
        let path_opt = p.plan_indexed_idx(g, r, start_idx, goal_idx, start_step, agv, horizon);
        write_indexed_path_buf(path_opt, out_nodes, out_arrivals, max_len, out_wait_steps)
    })
}

#[no_mangle]
pub unsafe extern "C" fn sipp_planner_plan_indexed(
    planner: *const SippPlanner,
    graph: *const PlannerGraph,
    reservations: *const ReservationTable,
    start_idx: u16,
    goal_idx: u16,
    start_step: u32,
    agv_id: *const c_char,
    horizon: u32,
    out_nodes: *mut u16,
    out_arrivals: *mut u32,
    max_len: usize,
    out_wait_steps: *mut u32,
) -> i32 {
    ffi_guard!(-1, {
        let p = match planner.as_ref() {
            Some(ptr) => ptr,
            None => return -1,
        };
        let g = match graph.as_ref() {
            Some(ptr) => ptr,
            None => return -1,
        };
        let r = match reservations.as_ref() {
            Some(ptr) => ptr,
            None => return -1,
        };
        let agv = match c_to_str(agv_id) {
            Some(val) => val,
            None => return -1,
        };
        let path_opt = p.plan_indexed_idx(g, r, start_idx, goal_idx, start_step, agv, horizon);
        write_indexed_path_buf(path_opt, out_nodes, out_arrivals, max_len, out_wait_steps)
    })
}

/// Reserves a sequence of indexed node/edge transitions in `ReservationTable` without string marshalling.
///
/// When `src_idxs[i] == dst_idxs[i]`, reserves a node dwell interval on `src_idxs[i]`;
/// otherwise reserves directed edge `src_idxs[i] -> dst_idxs[i]`.
#[no_mangle]
pub unsafe extern "C" fn planner_reserve_transitions_indexed(
    graph: *const PlannerGraph,
    tbl: *mut ReservationTable,
    agv_id: *const c_char,
    src_idxs: *const u16,
    dst_idxs: *const u16,
    starts: *const i64,
    ends: *const i64,
    count: usize,
) -> usize {
    ffi_guard!(0, {
        if count == 0
            || src_idxs.is_null()
            || dst_idxs.is_null()
            || starts.is_null()
            || ends.is_null()
        {
            return 0;
        }
        let g = match graph.as_ref() {
            Some(ptr) => ptr,
            None => return 0,
        };
        let t = match tbl.as_mut() {
            Some(ptr) => ptr,
            None => return 0,
        };
        let agv = match c_to_str(agv_id) {
            Some(val) => val,
            None => return 0,
        };
        let s_sl = std::slice::from_raw_parts(src_idxs, count);
        let d_sl = std::slice::from_raw_parts(dst_idxs, count);
        let st_sl = std::slice::from_raw_parts(starts, count);
        let en_sl = std::slice::from_raw_parts(ends, count);
        let mut applied = 0usize;
        for i in 0..count {
            let span = match Interval::try_new(st_sl[i], en_sl[i]) {
                Ok(iv) => iv,
                Err(_) => continue,
            };
            let u = s_sl[i];
            let v = d_sl[i];
            let Some(u_name) = g.node_name(u) else {
                continue;
            };
            if u == v {
                t.reserve_node(u_name, span, agv);
                applied += 1;
            } else if let Some(v_name) = g.node_name(v) {
                t.reserve_edge(u_name, v_name, span, agv);
                applied += 1;
            }
        }
        applied
    })
}

/// Solves a batch of independent SIPP queries in parallel across Rust worker threads.
///
/// Output slots use stride `max_len` per query: `out_lens[q]` receives hop count (`-1` on failure).
#[no_mangle]
pub unsafe extern "C" fn planner_batch_plan_sipp_indexed(
    planner: *const SippPlanner,
    graph: *const PlannerGraph,
    reservations: *const ReservationTable,
    n_queries: usize,
    start_idxs: *const u16,
    goal_idxs: *const u16,
    start_steps: *const u32,
    horizons: *const u32,
    agv_ids: *const *const c_char,
    max_len: usize,
    out_nodes: *mut u16,
    out_arrivals: *mut u32,
    out_lens: *mut i32,
    out_waits: *mut u32,
) -> usize {
    ffi_guard!(0, {
        if n_queries == 0
            || start_idxs.is_null()
            || goal_idxs.is_null()
            || start_steps.is_null()
            || horizons.is_null()
            || agv_ids.is_null()
            || out_lens.is_null()
            || out_waits.is_null()
        {
            return 0;
        }
        let p = match planner.as_ref() {
            Some(ptr) => ptr,
            None => return 0,
        };
        let g = match graph.as_ref() {
            Some(ptr) => ptr,
            None => return 0,
        };
        let r = match reservations.as_ref() {
            Some(ptr) => ptr,
            None => return 0,
        };
        let s_sl = std::slice::from_raw_parts(start_idxs, n_queries);
        let g_sl = std::slice::from_raw_parts(goal_idxs, n_queries);
        let t_sl = std::slice::from_raw_parts(start_steps, n_queries);
        let h_sl = std::slice::from_raw_parts(horizons, n_queries);
        let aid_ptrs = std::slice::from_raw_parts(agv_ids, n_queries);
        let mut aids: Vec<String> = Vec::with_capacity(n_queries);
        for &ptr in aid_ptrs {
            aids.push(c_to_str(ptr).unwrap_or("").to_string());
        }

        let results: Vec<Option<IndexedPath>> = if n_queries >= 4 {
            let n_workers = std::thread::available_parallelism()
                .map(|n| n.get().min(n_queries))
                .unwrap_or(1)
                .max(1);
            let chunk = (n_queries + n_workers - 1) / n_workers;
            let mut out: Vec<Option<IndexedPath>> = vec![None; n_queries];
            std::thread::scope(|scope| {
                let mut offset = 0usize;
                for slice in out.chunks_mut(chunk) {
                    let q_start = offset;
                    let q_len = slice.len();
                    offset += q_len;
                    let aids_ref = &aids;
                    scope.spawn(move || {
                        for i in 0..q_len {
                            let q = q_start + i;
                            slice[i] = p.plan_indexed_idx(
                                g,
                                r,
                                s_sl[q],
                                g_sl[q],
                                t_sl[q],
                                &aids_ref[q],
                                h_sl[q],
                            );
                        }
                    });
                }
            });
            out
        } else {
            (0..n_queries)
                .map(|q| p.plan_indexed_idx(g, r, s_sl[q], g_sl[q], t_sl[q], &aids[q], h_sl[q]))
                .collect()
        };

        let out_l = std::slice::from_raw_parts_mut(out_lens, n_queries);
        let out_w = std::slice::from_raw_parts_mut(out_waits, n_queries);
        let mut ok_count = 0usize;
        for (q, res) in results.into_iter().enumerate() {
            let dst_n = if !out_nodes.is_null() && max_len > 0 {
                out_nodes.add(q * max_len)
            } else {
                std::ptr::null_mut()
            };
            let dst_a = if !out_arrivals.is_null() && max_len > 0 {
                out_arrivals.add(q * max_len)
            } else {
                std::ptr::null_mut()
            };
            let code = write_indexed_path_buf(res, dst_n, dst_a, max_len, &mut out_w[q]);
            out_l[q] = code;
            if code >= 0 {
                ok_count += 1;
            }
        }
        ok_count
    })
}

unsafe fn write_cps_result_buf(
    res: CpsSearchResult,
    out_nodes: *mut u16,
    out_arrivals: *mut u32,
    max_len: usize,
    out_wait_steps: *mut u32,
    out_expansions: *mut usize,
    out_blocker_buf: *mut c_char,
    blocker_buf_cap: usize,
) -> i32 {
    if !out_expansions.is_null() {
        *out_expansions = res.expansions;
    }
    if !out_blocker_buf.is_null() && blocker_buf_cap > 0 {
        let dst = std::slice::from_raw_parts_mut(out_blocker_buf as *mut u8, blocker_buf_cap);
        let mut pos = 0usize;
        for (aid, cnt) in &res.blocker_hits {
            let line = format!("{}\t{}\n", aid, cnt);
            let b = line.as_bytes();
            if pos + b.len() + 1 > blocker_buf_cap {
                break;
            }
            dst[pos..pos + b.len()].copy_from_slice(b);
            pos += b.len();
        }
        dst[pos] = 0;
    }
    write_indexed_path_buf(res.path, out_nodes, out_arrivals, max_len, out_wait_steps)
}

#[no_mangle]
pub unsafe extern "C" fn planner_graph_set_dt(graph: *mut PlannerGraph, dt_s: f64) {
    ffi_guard!((), {
        if let Some(g) = graph.as_mut() {
            g.set_dt_s(dt_s);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn planner_graph_set_node_coords_indexed(
    graph: *mut PlannerGraph,
    node_idx: u16,
    x: f64,
    y: f64,
) -> bool {
    ffi_guard!(false, {
        let Some(g) = graph.as_mut() else {
            return false;
        };
        g.set_node_coords_indexed(node_idx, x, y)
    })
}

#[no_mangle]
pub unsafe extern "C" fn planner_graph_add_edge_ex(
    graph: *mut PlannerGraph,
    src: *const c_char,
    dst: *const c_char,
    traversal_steps: u32,
    traversal_time_s: f64,
    capacity: usize,
    physical_capacity: usize,
) -> bool {
    ffi_guard!(false, {
        let Some(g) = graph.as_mut() else {
            return false;
        };
        let Some(s) = c_to_str(src) else {
            return false;
        };
        let Some(d) = c_to_str(dst) else {
            return false;
        };
        g.add_edge_with_time(s, d, traversal_steps, traversal_time_s, capacity, physical_capacity);
        true
    })
}

#[no_mangle]
pub unsafe extern "C" fn planner_slot_field_create(
    graph: *const PlannerGraph,
    l_rad: u32,
    t_rad: u32,
    sigma_l: f64,
    sigma_t: f64,
    lookahead_ks: u32,
    lookahead_kt: u32,
    slot_delta: u32,
    max_track_horizon: i32,
    t_start: i32,
) -> *mut SlotCongestionField {
    ffi_guard!(std::ptr::null_mut(), {
        let Some(g) = graph.as_ref() else {
            return std::ptr::null_mut();
        };
        let cfg = SlotFieldConfig {
            spatial_radius_l: l_rad,
            temporal_radius_t: t_rad,
            sigma_l,
            sigma_t,
            lookahead_ks,
            lookahead_kt,
            slot_delta,
            max_track_horizon,
            t_start,
        };
        Box::into_raw(Box::new(SlotCongestionField::new(g, cfg)))
    })
}

#[no_mangle]
pub unsafe extern "C" fn planner_slot_field_destroy(field: *mut SlotCongestionField) {
    ffi_guard!((), {
        if !field.is_null() {
            drop(Box::from_raw(field));
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn planner_slot_field_clone(
    field: *const SlotCongestionField,
) -> *mut SlotCongestionField {
    ffi_guard!(std::ptr::null_mut(), {
        let Some(f) = field.as_ref() else {
            return std::ptr::null_mut();
        };
        Box::into_raw(Box::new(f.clone()))
    })
}

#[no_mangle]
pub unsafe extern "C" fn planner_slot_field_restore_from(
    dst: *mut SlotCongestionField,
    src: *const SlotCongestionField,
) -> bool {
    ffi_guard!(false, {
        let Some(d) = dst.as_mut() else {
            return false;
        };
        let Some(s) = src.as_ref() else {
            return false;
        };
        d.restore_from(s);
        true
    })
}

#[no_mangle]
pub unsafe extern "C" fn planner_slot_field_clear(field: *mut SlotCongestionField) {
    ffi_guard!((), {
        if let Some(f) = field.as_mut() {
            f.clear();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn planner_slot_field_remove_agv(
    field: *mut SlotCongestionField,
    agv_id: *const c_char,
) {
    ffi_guard!((), {
        if let (Some(f), Some(aid)) = (field.as_mut(), c_to_str(agv_id)) {
            f.remove_agv(aid);
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn planner_slot_field_add_points_indexed(
    field: *mut SlotCongestionField,
    agv_id: *const c_char,
    node_idxs: *const u16,
    steps: *const i32,
    len: usize,
    horizon: i32,
) {
    ffi_guard!((), {
        if len == 0 || node_idxs.is_null() || steps.is_null() {
            return;
        }
        let Some(f) = field.as_mut() else {
            return;
        };
        let Some(aid) = c_to_str(agv_id) else {
            return;
        };
        let n_sl = std::slice::from_raw_parts(node_idxs, len);
        let s_sl = std::slice::from_raw_parts(steps, len);
        let hz = if horizon >= 0 { Some(horizon) } else { None };
        f.add_agv_transitions(aid, n_sl, s_sl, hz);
    });
}

#[no_mangle]
pub unsafe extern "C" fn planner_slot_field_slot_phi_indexed(
    field: *mut SlotCongestionField,
    node_idx: u16,
    step: i32,
) -> f64 {
    ffi_guard!(0.0, {
        let Some(f) = field.as_mut() else {
            return 0.0;
        };
        f.slot_phi_cached(node_idx, step)
    })
}

#[no_mangle]
pub unsafe extern "C" fn planner_slot_field_phi_indexed(
    field: *mut SlotCongestionField,
    node_idx: u16,
    step: i32,
) -> f64 {
    ffi_guard!(0.0, {
        let Some(f) = field.as_mut() else {
            return 0.0;
        };
        f.phi_cached(node_idx, step)
    })
}

#[no_mangle]
pub unsafe extern "C" fn planner_reserve_points_with_park_indexed(
    graph: *const PlannerGraph,
    tbl: *mut ReservationTable,
    agv_id: *const c_char,
    pt_nodes: *const u16,
    pt_steps: *const i32,
    n_pts: usize,
    e_srcs: *const u16,
    e_dsts: *const u16,
    e_starts: *const i64,
    e_ends: *const i64,
    n_edges: usize,
    park_node_idx: u16,
    park_start: i64,
    park_end: i64,
) -> usize {
    ffi_guard!(0, {
        let Some(g) = graph.as_ref() else {
            return 0;
        };
        let Some(t) = tbl.as_mut() else {
            return 0;
        };
        let Some(agv) = c_to_str(agv_id) else {
            return 0;
        };
        let mut applied = 0usize;
        if n_pts > 0 && !pt_nodes.is_null() && !pt_steps.is_null() {
            let n_sl = std::slice::from_raw_parts(pt_nodes, n_pts);
            let s_sl = std::slice::from_raw_parts(pt_steps, n_pts);
            for i in 0..n_pts {
                if let Some(name) = g.node_name(n_sl[i]) {
                    let st = s_sl[i] as i64;
                    if let Ok(iv) = Interval::try_new(st, st + 1) {
                        t.reserve_node(name, iv, agv);
                        applied += 1;
                    }
                }
            }
        }
        if n_edges > 0 && !e_srcs.is_null() && !e_dsts.is_null() && !e_starts.is_null() && !e_ends.is_null() {
            let us = std::slice::from_raw_parts(e_srcs, n_edges);
            let vs = std::slice::from_raw_parts(e_dsts, n_edges);
            let sts = std::slice::from_raw_parts(e_starts, n_edges);
            let ens = std::slice::from_raw_parts(e_ends, n_edges);
            for i in 0..n_edges {
                if let (Some(un), Some(vn)) = (g.node_name(us[i]), g.node_name(vs[i])) {
                    if let Ok(iv) = Interval::try_new(sts[i], ens[i]) {
                        t.reserve_edge(un, vn, iv, agv);
                        applied += 1;
                    }
                }
            }
        }
        if park_node_idx != u16::MAX && park_end > park_start {
            if let Some(p_name) = g.node_name(park_node_idx) {
                if let Ok(iv) = Interval::try_new(park_start, park_end) {
                    t.reserve_node(p_name, iv, agv);
                    applied += 1;
                }
            }
        }
        applied
    })
}

#[no_mangle]
pub unsafe extern "C" fn spacetime_planner_plan_cps_indexed(
    planner: *const SpacetimeAStarPlanner,
    graph: *const PlannerGraph,
    reservations: *const ReservationTable,
    field: *mut SlotCongestionField,
    start_idx: u16,
    goal_idx: u16,
    start_step: u32,
    agv_id: *const c_char,
    horizon: u32,
    mode_oneshot: bool,
    alpha: f64,
    beta: f64,
    gamma: f64,
    lookahead_omega: f64,
    goal_park_until: i64,
    goal_park_steps: u32,
    prev_arrival_node: u16,
    max_expansions_override: usize,
    collect_blockers: bool,
    out_nodes: *mut u16,
    out_arrivals: *mut u32,
    max_len: usize,
    out_wait_steps: *mut u32,
    out_expansions: *mut usize,
    out_blocker_buf: *mut c_char,
    blocker_buf_cap: usize,
) -> i32 {
    ffi_guard!(-1, {
        let Some(p) = planner.as_ref() else {
            return -1;
        };
        let Some(g) = graph.as_ref() else {
            return -1;
        };
        let Some(r) = reservations.as_ref() else {
            return -1;
        };
        let Some(agv) = c_to_str(agv_id) else {
            return -1;
        };
        let f_opt = field.as_mut();
        let params = CpsSearchParams {
            mode_oneshot,
            alpha,
            beta,
            gamma,
            lookahead_omega,
            goal_park_until: if goal_park_until >= 0 { Some(goal_park_until as u32) } else { None },
            goal_park_steps,
            prev_arrival_node,
            max_expansions_override,
            collect_blockers,
        };
        let res = p.plan_cps_indexed(g, r, f_opt, start_idx, goal_idx, start_step, agv, horizon, params);
        write_cps_result_buf(
            res,
            out_nodes,
            out_arrivals,
            max_len,
            out_wait_steps,
            out_expansions,
            out_blocker_buf,
            blocker_buf_cap,
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn sipp_planner_plan_indexed_ex(
    planner: *const SippPlanner,
    graph: *const PlannerGraph,
    reservations: *const ReservationTable,
    field: *const SlotCongestionField,
    start_idx: u16,
    goal_idx: u16,
    start_step: u32,
    agv_id: *const c_char,
    horizon: u32,
    alpha: f64,
    gamma: f64,
    goal_park_until: i64,
    goal_park_steps: u32,
    max_expansions_override: usize,
    collect_blockers: bool,
    out_nodes: *mut u16,
    out_arrivals: *mut u32,
    max_len: usize,
    out_wait_steps: *mut u32,
    out_expansions: *mut usize,
    out_blocker_buf: *mut c_char,
    blocker_buf_cap: usize,
) -> i32 {
    ffi_guard!(-1, {
        let Some(p) = planner.as_ref() else {
            return -1;
        };
        let Some(g) = graph.as_ref() else {
            return -1;
        };
        let Some(r) = reservations.as_ref() else {
            return -1;
        };
        let Some(agv) = c_to_str(agv_id) else {
            return -1;
        };
        let f_opt = field.as_ref();
        let params = CpsSearchParams {
            mode_oneshot: false,
            alpha,
            beta: 0.0,
            gamma,
            lookahead_omega: 0.0,
            goal_park_until: if goal_park_until >= 0 { Some(goal_park_until as u32) } else { None },
            goal_park_steps,
            prev_arrival_node: u16::MAX,
            max_expansions_override,
            collect_blockers,
        };
        let res = p.plan_indexed_ex(g, r, f_opt, start_idx, goal_idx, start_step, agv, horizon, params);
        write_cps_result_buf(
            res,
            out_nodes,
            out_arrivals,
            max_len,
            out_wait_steps,
            out_expansions,
            out_blocker_buf,
            blocker_buf_cap,
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn planner_batch_plan_sipp_indexed_ex(
    planner: *const SippPlanner,
    graph: *const PlannerGraph,
    reservations: *const ReservationTable,
    field: *const SlotCongestionField,
    n_queries: usize,
    start_idxs: *const u16,
    goal_idxs: *const u16,
    start_steps: *const u32,
    horizons: *const u32,
    agv_ids: *const *const c_char,
    alpha: f64,
    gamma: f64,
    goal_park_steps: u32,
    max_len: usize,
    out_nodes: *mut u16,
    out_arrivals: *mut u32,
    out_lens: *mut i32,
    out_waits: *mut u32,
    out_expansions: *mut usize,
) -> usize {
    ffi_guard!(0, {
        if n_queries == 0
            || start_idxs.is_null()
            || goal_idxs.is_null()
            || start_steps.is_null()
            || horizons.is_null()
            || agv_ids.is_null()
            || out_lens.is_null()
            || out_waits.is_null()
        {
            return 0;
        }
        let Some(p) = planner.as_ref() else {
            return 0;
        };
        let Some(g) = graph.as_ref() else {
            return 0;
        };
        let Some(r) = reservations.as_ref() else {
            return 0;
        };
        let f_opt = field.as_ref();
        let s_sl = std::slice::from_raw_parts(start_idxs, n_queries);
        let g_sl = std::slice::from_raw_parts(goal_idxs, n_queries);
        let t_sl = std::slice::from_raw_parts(start_steps, n_queries);
        let h_sl = std::slice::from_raw_parts(horizons, n_queries);
        let aid_ptrs = std::slice::from_raw_parts(agv_ids, n_queries);
        let mut aids: Vec<String> = Vec::with_capacity(n_queries);
        for &ptr in aid_ptrs {
            aids.push(c_to_str(ptr).unwrap_or("").to_string());
        }
        let params = CpsSearchParams {
            mode_oneshot: false,
            alpha,
            beta: 0.0,
            gamma,
            lookahead_omega: 0.0,
            goal_park_until: None,
            goal_park_steps,
            prev_arrival_node: u16::MAX,
            max_expansions_override: 0,
            collect_blockers: false,
        };

        let results: Vec<CpsSearchResult> = if n_queries >= 4 {
            let n_workers = std::thread::available_parallelism()
                .map(|n| n.get().min(n_queries))
                .unwrap_or(1)
                .max(1);
            let chunk = (n_queries + n_workers - 1) / n_workers;
            let mut out: Vec<CpsSearchResult> = vec![CpsSearchResult::default(); n_queries];
            std::thread::scope(|scope| {
                let mut offset = 0usize;
                for slice in out.chunks_mut(chunk) {
                    let q_start = offset;
                    let q_len = slice.len();
                    offset += q_len;
                    let aids_ref = &aids;
                    scope.spawn(move || {
                        for i in 0..q_len {
                            let q = q_start + i;
                            slice[i] = p.plan_indexed_ex(
                                g,
                                r,
                                f_opt,
                                s_sl[q],
                                g_sl[q],
                                t_sl[q],
                                &aids_ref[q],
                                h_sl[q],
                                params,
                            );
                        }
                    });
                }
            });
            out
        } else {
            (0..n_queries)
                .map(|q| p.plan_indexed_ex(g, r, f_opt, s_sl[q], g_sl[q], t_sl[q], &aids[q], h_sl[q], params))
                .collect()
        };

        let out_l = std::slice::from_raw_parts_mut(out_lens, n_queries);
        let out_w = std::slice::from_raw_parts_mut(out_waits, n_queries);
        let out_e = if !out_expansions.is_null() {
            Some(std::slice::from_raw_parts_mut(out_expansions, n_queries))
        } else {
            None
        };
        let mut ok_count = 0usize;
        let mut out_e_opt = out_e;
        for (q, res) in results.into_iter().enumerate() {
            if let Some(ref mut e_sl) = out_e_opt {
                e_sl[q] = res.expansions;
            }
            let dst_n = if !out_nodes.is_null() && max_len > 0 {
                out_nodes.add(q * max_len)
            } else {
                std::ptr::null_mut()
            };
            let dst_a = if !out_arrivals.is_null() && max_len > 0 {
                out_arrivals.add(q * max_len)
            } else {
                std::ptr::null_mut()
            };
            let code = write_indexed_path_buf(res.path, dst_n, dst_a, max_len, &mut out_w[q]);
            out_l[q] = code;
            if code >= 0 {
                ok_count += 1;
            }
        }
        ok_count
    })
}
