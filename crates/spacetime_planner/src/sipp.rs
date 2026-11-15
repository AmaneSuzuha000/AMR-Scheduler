//! Safe Interval Path Planning (SIPP) implementation.
//!
//! Unlike standard Spacetime A* which expands states step-by-step (O(H)),
//! SIPP searches across continuous safe time intervals (O(intervals)),
//! drastically speeding up pathfinding under heavy congestion and long horizons.

use crate::astar::PlannedPath;
use crate::graph::PlannerGraph;
use reservation_core::{Interval, ReservationTable};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Copy, Clone, Eq, PartialEq)]
struct SippSearchItem {
    f: u32,
    g: u32,
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
        if start_node == goal_node {
            return Some(PlannedPath {
                nodes: Vec::new(),
                arrival_steps: Vec::new(),
                wait_steps: 0,
            });
        }

        let start_idx = graph.node_index(start_node)?;
        let goal_idx = graph.node_index(goal_node)?;

        let h_table = graph.compute_heuristics(goal_idx);
        if h_table[start_idx as usize] == u32::MAX {
            return None;
        }

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
        let initial_f = initial_g.saturating_add(h_table[start_idx as usize]);

        frontier.push(SippSearchItem {
            f: initial_f,
            g: initial_g,
            node: start_idx,
            interval_idx: init_iv_idx,
            tiebreak: counter,
        });

        // Key: (node, interval_idx) -> earliest arrival time g
        let mut best_g: HashMap<(u16, u16), u32> = HashMap::new();
        // Key: (node, interval_idx) -> ((parent_node, parent_interval_idx), depart_time, arrival_time)
        let mut parent: HashMap<(u16, u16), ((u16, u16), u32, u32)> = HashMap::new();

        best_g.insert((start_idx, init_iv_idx), initial_g);

        let mut goal_state: Option<(u16, u16)> = None;
        let mut expansions = 0usize;

        while let Some(SippSearchItem { f: _, g, node, interval_idx, tiebreak: _ }) = frontier.pop() {
            expansions += 1;
            if expansions > self.max_expansions {
                return None;
            }

            if let Some(&prev_g) = best_g.get(&(node, interval_idx)) {
                if prev_g < g {
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
                    // Departure cannot happen before arrival g
                    // Nor after curr_iv.end - 1
                    let min_dep_by_arrival = g;
                    let max_dep_by_curr_iv = curr_iv.end.saturating_sub(1) as u32;

                    // Arrival v_arr = dep + d must fall into v_iv:
                    // dep + d >= v_iv.start => dep >= v_iv.start - d
                    let min_dep_by_v = (v_iv.start as u32).saturating_sub(d);
                    // dep + d < v_iv.end => dep < v_iv.end - d => dep <= v_iv.end - d - 1
                    if (v_iv.end as u32) <= d {
                        continue;
                    }
                    let max_dep_by_v = (v_iv.end as u32) - d - 1;

                    let earliest_dep = min_dep_by_arrival.max(min_dep_by_v);
                    let latest_dep = max_dep_by_curr_iv.min(max_dep_by_v);

                    if earliest_dep > latest_dep {
                        continue;
                    }

                    // Find earliest departure in [earliest_dep, latest_dep] that has free edges
                    let mut valid_dep: Option<u32> = None;
                    for dep in earliest_dep..=latest_dep {
                        let arrive = dep + d;
                        if arrive > horizon {
                            break;
                        }

                        let edge_span = Interval::new_unchecked(dep as i64, arrive as i64);

                        // Opposing channel & edge capacity check
                        let occ1 = reservations.edge_conflicts(p0, p1, edge_span, Some(agv_id));
                        let occ2 = reservations.edge_conflicts(p1, p0, edge_span, Some(agv_id));
                        let mut unique_occ: HashSet<String> = occ1.into_iter().collect();
                        unique_occ.extend(occ2);

                        if unique_occ.len() < edge.physical_capacity {
                            valid_dep = Some(dep);
                            break;
                        }
                    }

                    if let Some(dep) = valid_dep {
                        let arrive = dep + d;
                        let next_state = (v, v_iv_idx as u16);
                        let prev_best = *best_g.get(&next_state).unwrap_or(&u32::MAX);

                        if arrive < prev_best {
                            best_g.insert(next_state, arrive);
                            parent.insert(next_state, ((node, interval_idx), dep, arrive));
                            counter += 1;
                            let h_val = h_table[v as usize];
                            frontier.push(SippSearchItem {
                                f: arrive.saturating_add(h_val),
                                g: arrive,
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

        let mut nodes = Vec::new();
        let mut arrivals = Vec::new();
        let mut total_waits = 0u32;
        let mut last_arr = start_step;

        for &(v, arr, dep) in &waypoints {
            nodes.push(graph.node_name(v).unwrap().to_string());
            arrivals.push(arr);
            if dep > last_arr {
                total_waits += dep - last_arr;
            }
            last_arr = arr;
        }

        Some(PlannedPath {
            nodes,
            arrival_steps: arrivals,
            wait_steps: total_waits,
        })
    }
}
