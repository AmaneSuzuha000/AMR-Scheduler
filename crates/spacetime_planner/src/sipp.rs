//! Safe Interval Path Planning (SIPP) implementation.
//!
//! Unlike standard Spacetime A* which expands states step-by-step (O(H)),
//! SIPP searches across continuous safe time intervals (O(intervals)),
//! drastically speeding up pathfinding under heavy congestion and long horizons.

use crate::astar::{sort_blocker_hits, CpsSearchParams, CpsSearchResult, IndexedPath, OrdF64, PlannedPath};
use crate::graph::PlannerGraph;
use crate::slot_field::SlotCongestionField;
use reservation_core::{Interval, ReservationTable};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

#[derive(Copy, Clone, Eq, PartialEq)]
struct SippSearchItem {
    f: u32,
    g: u32,
    score_g: u32,
    node: u16,
    interval_idx: u16,
    tiebreak: u64,
}

impl Ord for SippSearchItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap on f, then tiebreak
        other.f.cmp(&self.f)
            .then_with(|| other.tiebreak.cmp(&self.tiebreak))
    }
}

impl PartialOrd for SippSearchItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Default)]
pub struct SippPlanner {
    pub max_expansions: usize,
}

impl SippPlanner {
    pub fn new(max_expansions: usize) -> Self {
        Self {
            max_expansions: if max_expansions == 0 { 50_000 } else { max_expansions },
        }
    }

    pub fn plan(
        &self,
        graph: &PlannerGraph,
        reservations: &ReservationTable,
        start_node: &str,
        goal_node: &str,
        start_step: u32,
        agv_id: &str,
        horizon: u32,
    ) -> Option<PlannedPath> {
        let start_idx = graph.node_index(start_node)?;
        let goal_idx = graph.node_index(goal_node)?;
        self.plan_indexed_idx(graph, reservations, start_idx, goal_idx, start_step, agv_id, horizon)
            .map(|p| p.to_planned_path(graph))
    }

    /// Zero-string-return Safe Interval Path Planning over `u16` node indices.
    pub fn plan_indexed_idx(
        &self,
        graph: &PlannerGraph,
        reservations: &ReservationTable,
        start_idx: u16,
        goal_idx: u16,
        start_step: u32,
        agv_id: &str,
        horizon: u32,
    ) -> Option<IndexedPath> {
        if start_idx == goal_idx {
            return Some(IndexedPath {
                nodes: Vec::new(),
                arrivals: Vec::new(),
                wait_steps: 0,
            });
        }
        if (start_idx as usize) >= graph.node_count() || (goal_idx as usize) >= graph.node_count() {
            return None;
        }

        let h_table = graph.get_heuristics_arc(goal_idx);
        if h_table[start_idx as usize] == u32::MAX {
            return None;
        }
        let guide_ref = graph.agv_guide(agv_id);

        // Cache safe intervals per node
        let n = graph.node_count();
        let mut node_intervals: Vec<Vec<Interval>> = Vec::with_capacity(n);
        for i in 0..n {
            let name = graph.node_name(i as u16).unwrap();
            let cap = graph.node_capacity(i as u16);
            node_intervals.push(reservations.safe_intervals_node(name, horizon, cap, Some(agv_id)));
        }

        // Find initial safe interval on start_node that covers start_step
        let start_intervals = &node_intervals[start_idx as usize];
        let mut start_interval_idx: Option<usize> = None;
        for (idx, iv) in start_intervals.iter().enumerate() {
            if iv.start <= (start_step as i64) && (start_step as i64) < iv.end {
                start_interval_idx = Some(idx);
                break;
            }
        }
        let init_iv_idx = start_interval_idx? as u16;

        let mut counter = 0u64;
        let mut frontier = BinaryHeap::new();

        let initial_g = start_step;
        let initial_f = h_table[start_idx as usize].saturating_mul(8);

        frontier.push(SippSearchItem {
            f: initial_f,
            g: initial_g,
            score_g: 0,
            node: start_idx,
            interval_idx: init_iv_idx,
            tiebreak: counter,
        });

        // Key: (node, interval_idx) -> best Q3 score_g
        let mut best_score: HashMap<(u16, u16), u32> = HashMap::with_capacity(128);
        // Key: (node, interval_idx) -> ((parent_node, parent_interval_idx), depart_time, arrival_time)
        let mut parent: HashMap<(u16, u16), ((u16, u16), u32, u32)> = HashMap::with_capacity(128);

        best_score.insert((start_idx, init_iv_idx), 0);

        let mut goal_state: Option<(u16, u16)> = None;
        let mut expansions = 0usize;

        while let Some(SippSearchItem { f: _, g, score_g, node, interval_idx, tiebreak: _ }) = frontier.pop() {
            expansions += 1;
            if expansions > self.max_expansions {
                return None;
            }

            if let Some(&prev_score) = best_score.get(&(node, interval_idx)) {
                if prev_score < score_g {
                    continue;
                }
            }

            if node == goal_idx {
                goal_state = Some((node, interval_idx));
                break;
            }

            let curr_iv = &node_intervals[node as usize][interval_idx as usize];
            let curr_name = graph.node_name(node).unwrap();

            // Try all outgoing edges to neighbors
            for edge in graph.out_edges(node) {
                let v = edge.dst;
                let v_name = graph.node_name(v).unwrap();
                let d = edge.traversal_steps;
                let v_intervals = &node_intervals[v as usize];

                let (p0, p1) = if curr_name <= v_name {
                    (curr_name, v_name)
                } else {
                    (v_name, curr_name)
                };

                for (v_iv_idx, v_iv) in v_intervals.iter().enumerate() {
                    let min_dep_by_arrival = g;
                    let max_dep_by_curr_iv = curr_iv.end.saturating_sub(1) as u32;
                    let min_dep_by_v = (v_iv.start as u32).saturating_sub(d);
                    if (v_iv.end as u32) <= d {
                        continue;
                    }
                    let max_dep_by_v = (v_iv.end as u32) - d - 1;

                    let earliest_dep = min_dep_by_arrival.max(min_dep_by_v);
                    let latest_dep = max_dep_by_curr_iv.min(max_dep_by_v);

                    if earliest_dep > latest_dep {
                        continue;
                    }

                    let mut valid_dep: Option<u32> = None;
                    for dep in earliest_dep..=latest_dep {
                        let arrive = dep + d;
                        if arrive > horizon {
                            break;
                        }

                        let edge_span = Interval::new_unchecked(dep as i64, arrive as i64);
                        if reservations.is_undirected_edge_free(
                            p0,
                            p1,
                            edge_span,
                            edge.physical_capacity,
                            Some(agv_id),
                        ) {
                            valid_dep = Some(dep);
                            break;
                        }
                    }

                    if let Some(dep) = valid_dep {
                        let arrive = dep + d;
                        let wait_cost = (dep - g).saturating_mul(8);
                        let pen_q3 = graph.edge_penalty_q3(node, edge, dep, guide_ref);
                        let next_score = score_g
                            .saturating_add(wait_cost)
                            .saturating_add(d.saturating_mul(8))
                            .saturating_add(pen_q3);
                        let next_state = (v, v_iv_idx as u16);
                        let prev_best = *best_score.get(&next_state).unwrap_or(&u32::MAX);

                        if next_score < prev_best {
                            best_score.insert(next_state, next_score);
                            parent.insert(next_state, ((node, interval_idx), dep, arrive));
                            counter += 1;
                            let h_val = h_table[v as usize].saturating_mul(8);
                            frontier.push(SippSearchItem {
                                f: next_score.saturating_add(h_val),
                                g: arrive,
                                score_g: next_score,
                                node: v,
                                interval_idx: v_iv_idx as u16,
                                tiebreak: counter,
                            });
                        }
                    }
                }
            }
        }

        let end_state = goal_state?;

        // Reconstruct trajectory
        let mut waypoints: Vec<(u16, u32, u32)> = Vec::new(); // (node, arrive, depart)
        let mut cur = end_state;

        while let Some(&(p_state, dep, arr)) = parent.get(&cur) {
            waypoints.push((cur.0, arr, dep));
            cur = p_state;
        }
        waypoints.reverse();

        let mut nodes = Vec::with_capacity(waypoints.len());
        let mut arrivals = Vec::with_capacity(waypoints.len());
        let mut total_waits = 0u32;
        let mut last_arr = start_step;

        for &(v, arr, dep) in &waypoints {
            nodes.push(v);
            arrivals.push(arr);
            if dep > last_arr {
                total_waits += dep - last_arr;
            }
            last_arr = arr;
        }

        Some(IndexedPath {
            nodes,
            arrivals,
            wait_steps: total_waits,
        })
    }

    /// Extended Safe Interval Path Planning supporting native `SlotCongestionField`,
    /// goal parking intervals, expansion counting, and zero-overhead failure blocker attribution.
    pub fn plan_indexed_ex(
        &self,
        graph: &PlannerGraph,
        reservations: &ReservationTable,
        field: Option<&SlotCongestionField>,
        start_idx: u16,
        goal_idx: u16,
        start_step: u32,
        agv_id: &str,
        horizon: u32,
        params: CpsSearchParams,
    ) -> CpsSearchResult {
        if (start_idx as usize) >= graph.node_count() || (goal_idx as usize) >= graph.node_count() {
            return CpsSearchResult::default();
        }
        let h_table = graph.get_heuristics_time_arc(goal_idx);
        if !h_table[start_idx as usize].is_finite() {
            return CpsSearchResult::default();
        }

        let n = graph.node_count();
        // Lazy per-node safe interval cache (`O(visited)` instead of `O(|V|)` per query)
        let mut node_intervals: Vec<Option<Vec<Interval>>> = vec![None; n];
        let start_name = graph.node_name(start_idx).unwrap();
        node_intervals[start_idx as usize] = Some(reservations.safe_intervals_node(
            start_name,
            horizon,
            graph.node_capacity(start_idx),
            Some(agv_id),
        ));

        let mut init_iv_opt: Option<usize> = None;
        if let Some(ref start_intervals) = node_intervals[start_idx as usize] {
            for (idx, iv) in start_intervals.iter().enumerate() {
                if iv.start <= (start_step as i64) && (start_step as i64) < iv.end {
                    init_iv_opt = Some(idx);
                    break;
                }
            }
        }

        let collect_fail_blockers = |collect: bool| -> Vec<(String, u32)> {
            if !collect {
                return Vec::new();
            }
            let mut bmap: HashMap<String, u32> = HashMap::new();
            let cur_name = graph.node_name(start_idx).unwrap();
            let goal_name = graph.node_name(goal_idx).unwrap();
            let chk_hi = horizon.min(start_step.saturating_add(12));
            if chk_hi > start_step + 1 {
                let sp = Interval::new_unchecked((start_step + 1) as i64, chk_hi as i64);
                reservations.collect_node_conflicts_into(cur_name, sp, Some(agv_id), &mut bmap);
                reservations.collect_node_conflicts_into(cur_name, sp, Some(agv_id), &mut bmap);
                for edge in graph.out_edges(start_idx) {
                    let dst_name = graph.node_name(edge.dst).unwrap();
                    reservations.collect_node_conflicts_into(dst_name, sp, Some(agv_id), &mut bmap);
                    let (p0, p1) = if cur_name <= dst_name { (cur_name, dst_name) } else { (dst_name, cur_name) };
                    reservations.collect_undirected_edge_conflicts_into(p0, p1, sp, Some(agv_id), &mut bmap);
                }
            }
            let g_hi = horizon.min(start_step.saturating_add(60));
            if g_hi > start_step + 1 {
                let g_sp = Interval::new_unchecked((start_step + 1) as i64, g_hi as i64);
                reservations.collect_node_conflicts_into(goal_name, g_sp, Some(agv_id), &mut bmap);
                reservations.collect_node_conflicts_into(goal_name, g_sp, Some(agv_id), &mut bmap);
            }
            sort_blocker_hits(bmap)
        };

        let Some(init_iv_usize) = init_iv_opt else {
            return CpsSearchResult {
                path: None,
                expansions: 0,
                blocker_hits: collect_fail_blockers(params.collect_blockers),
            };
        };
        let init_iv_idx = init_iv_usize as u16;

        let eff_park_end = |arr_t: u32| -> u32 {
            params
                .goal_park_until
                .unwrap_or_else(|| horizon.min(arr_t.saturating_add(1).saturating_add(params.goal_park_steps)))
        };

        if start_idx == goal_idx {
            let need_end = eff_park_end(start_step);
            let s_iv_end = node_intervals[start_idx as usize].as_ref().unwrap()[init_iv_usize].end;
            if need_end <= start_step + 1 || (s_iv_end >= (need_end as i64)) {
                return CpsSearchResult {
                    path: Some(IndexedPath {
                        nodes: Vec::new(),
                        arrivals: Vec::new(),
                        wait_steps: 0,
                    }),
                    expansions: 0,
                    blocker_hits: Vec::new(),
                };
            }
        }

        let dt_s = graph.dt_s();
        let alpha = params.alpha;
        let gamma = params.gamma;
        let max_exp = if params.max_expansions_override > 0 {
            params.max_expansions_override
        } else {
            self.max_expansions
        };
        let guide_ref = graph.agv_guide(agv_id);

        #[derive(Copy, Clone, Eq, PartialEq)]
        struct SippExItem {
            f: OrdF64,
            g: u32,
            score_g: OrdF64,
            node: u16,
            interval_idx: u16,
            tiebreak: u64,
        }
        impl Ord for SippExItem {
            fn cmp(&self, other: &Self) -> Ordering {
                other
                    .f
                    .cmp(&self.f)
                    .then_with(|| other.score_g.cmp(&self.score_g))
                    .then_with(|| other.tiebreak.cmp(&self.tiebreak))
            }
        }
        impl PartialOrd for SippExItem {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut counter = 0u64;
        let mut frontier = BinaryHeap::with_capacity(128);
        let h0 = alpha * (h_table[start_idx as usize] / dt_s).floor();
        frontier.push(SippExItem {
            f: OrdF64(h0),
            g: start_step,
            score_g: OrdF64(0.0),
            node: start_idx,
            interval_idx: init_iv_idx,
            tiebreak: counter,
        });

        let mut best_score: HashMap<(u16, u16), f64> = HashMap::with_capacity(128);
        let mut parent: HashMap<(u16, u16), ((u16, u16), u32, u32)> = HashMap::with_capacity(128);
        best_score.insert((start_idx, init_iv_idx), 0.0);

        let mut goal_state: Option<(u16, u16)> = None;
        let mut expansions = 0usize;

        while let Some(SippExItem { f: _, g, score_g: score_g_w, node, interval_idx, tiebreak: _ }) = frontier.pop() {
            expansions += 1;
            if expansions > max_exp {
                break;
            }
            let score_g = score_g_w.0;
            if let Some(&prev_score) = best_score.get(&(node, interval_idx)) {
                if prev_score + 1e-9 < score_g {
                    continue;
                }
            }
            let curr_iv = node_intervals[node as usize].as_ref().unwrap()[interval_idx as usize];
            if node == goal_idx {
                let req_end = eff_park_end(g);
                if req_end <= g + 1 || (curr_iv.end >= (req_end as i64)) {
                    goal_state = Some((node, interval_idx));
                    break;
                }
            }

            let curr_name = graph.node_name(node).unwrap();

            for edge in graph.out_edges(node) {
                let v = edge.dst;
                let v_name = graph.node_name(v).unwrap();
                let d = edge.traversal_steps;
                if node_intervals[v as usize].is_none() {
                    node_intervals[v as usize] = Some(reservations.safe_intervals_node(
                        v_name,
                        horizon,
                        graph.node_capacity(v),
                        Some(agv_id),
                    ));
                }
                let v_intervals = node_intervals[v as usize].as_ref().unwrap();
                let (p0, p1) = if curr_name <= v_name {
                    (curr_name, v_name)
                } else {
                    (v_name, curr_name)
                };

                for (v_iv_idx, v_iv) in v_intervals.iter().enumerate() {
                    let min_dep_by_arrival = g;
                    let max_dep_by_curr_iv = curr_iv.end.saturating_sub(1) as u32;
                    let min_dep_by_v = (v_iv.start as u32).saturating_sub(d);
                    if (v_iv.end as u32) <= d {
                        continue;
                    }
                    let max_dep_by_v = (v_iv.end as u32) - d - 1;
                    let earliest_dep = min_dep_by_arrival.max(min_dep_by_v);
                    let latest_dep = max_dep_by_curr_iv.min(max_dep_by_v);
                    if earliest_dep > latest_dep {
                        continue;
                    }

                    let mut valid_dep: Option<u32> = None;
                    for dep in earliest_dep..=latest_dep {
                        let arrive = dep + d;
                        if arrive > horizon {
                            break;
                        }
                        if v == goal_idx {
                            let req_end = eff_park_end(arrive);
                            if (req_end as i64) > v_iv.end {
                                break;
                            }
                        }
                        let edge_span = Interval::new_unchecked(dep as i64, arrive as i64);
                        if reservations.is_undirected_edge_free(p0, p1, edge_span, edge.physical_capacity, Some(agv_id)) {
                            valid_dep = Some(dep);
                            break;
                        }
                    }

                    if let Some(dep) = valid_dep {
                        let arrive = dep + d;
                        let wait_steps = dep - g;
                        let mut wait_cong = 0.0;
                        if wait_steps > 0 && gamma > 0.0 {
                            if let Some(fld) = field {
                                for ws in 1..=wait_steps {
                                    wait_cong += gamma * fld.slot_phi_read(node, (g + ws) as i32);
                                }
                            }
                        }
                        let pen_ltm = (graph.edge_penalty_q3(node, edge, dep, guide_ref) as f64) * 0.125;
                        let next_score = score_g + alpha * ((wait_steps + d) as f64) + wait_cong + pen_ltm;
                        let next_state = (v, v_iv_idx as u16);
                        let prev_best = best_score.get(&next_state).copied().unwrap_or(1e18);
                        if next_score + 1e-9 < prev_best {
                            best_score.insert(next_state, next_score);
                            parent.insert(next_state, ((node, interval_idx), dep, arrive));
                            counter += 1;
                            let hv = h_table[v as usize];
                            let h_val = if hv.is_finite() { alpha * (hv / dt_s).floor() } else { 1e9 };
                            frontier.push(SippExItem {
                                f: OrdF64(next_score + h_val),
                                g: arrive,
                                score_g: OrdF64(next_score),
                                node: v,
                                interval_idx: v_iv_idx as u16,
                                tiebreak: counter,
                            });
                        }
                    }
                }
            }
        }

        let Some(end_state) = goal_state else {
            return CpsSearchResult {
                path: None,
                expansions,
                blocker_hits: collect_fail_blockers(params.collect_blockers),
            };
        };

        let mut waypoints: Vec<(u16, u32, u32)> = Vec::new();
        let mut cur = end_state;
        while let Some(&(p_state, dep, arr)) = parent.get(&cur) {
            waypoints.push((cur.0, arr, dep));
            cur = p_state;
        }
        waypoints.reverse();

        let mut nodes = Vec::with_capacity(waypoints.len());
        let mut arrivals = Vec::with_capacity(waypoints.len());
        let mut total_waits = 0u32;
        let mut last_arr = start_step;
        for &(v, arr, dep) in &waypoints {
            nodes.push(v);
            arrivals.push(arr);
            if dep > last_arr {
                total_waits += dep - last_arr;
            }
            last_arr = arr;
        }

        CpsSearchResult {
            path: Some(IndexedPath {
                nodes,
                arrivals,
                wait_steps: total_waits,
            }),
            expansions,
            blocker_hits: Vec::new(),
        }
    }
}
