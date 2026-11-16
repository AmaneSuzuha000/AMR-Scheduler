//! C-ABI Foreign Function Interface for `spacetime_planner`.

use crate::astar::SpacetimeAStarPlanner;
use crate::graph::PlannerGraph;
use reservation_core::ReservationTable;
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
