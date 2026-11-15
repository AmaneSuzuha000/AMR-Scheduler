//! Native Spatiotemporal A* Search Engine.

use crate::graph::PlannerGraph;
use reservation_core::{Interval, ReservationTable};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedPath {
    pub nodes: Vec<String>,
    pub arrival_steps: Vec<u32>,
    pub wait_steps: u32,
}

#[inline(always)]
fn pack_state(node: u16, t: u32) -> u64 {
    ((node as u64) << 32) | (t as u64)
}

#[inline(always)]
fn unpack_state(s: u64) -> (u16, u32) {
    (((s >> 32) & 0xFFFF) as u16, (s & 0xFFFFFFFF) as u32)
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct SearchItem {
    f: u32,
    g: u32,
    tiebreak: u64,
    node: u16,
}

impl Ord for SearchItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: lowest f first, tiebreak on lowest tiebreak counter
        other.f.cmp(&self.f)
            .then_with(|| other.tiebreak.cmp(&self.tiebreak))
    }
}

impl PartialOrd for SearchItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Spatiotemporal A* path planner matching Python's exact search behavior.
#[derive(Debug, Clone, Default)]
pub struct SpacetimeAStarPlanner {
    pub max_expansions: usize,
}

impl SpacetimeAStarPlanner {
    pub fn new(max_expansions: usize) -> Self {
        Self {
            max_expansions: if max_expansions == 0 { 200_000 } else { max_expansions },
        }
    }

    /// Plans a collision-free spatiotemporal trajectory from `start` to `goal`.
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

        // Precompute reverse static heuristic
        let h_table = graph.compute_heuristics(goal_idx);
        if h_table[start_idx as usize] == u32::MAX {
            // Unreachable
            return None;
        }

        let mut counter = 0u64;
        let mut frontier = BinaryHeap::new();

        let initial_state = pack_state(start_idx, start_step);
        let h_start = h_table[start_idx as usize];

        frontier.push(SearchItem {
            f: h_start,
            g: 0,
            tiebreak: counter,
            node: start_idx,
        });

        let mut best_g: HashMap<u64, u32> = HashMap::new();
        let mut parent: HashMap<u64, u64> = HashMap::new();
        let mut wait_count: HashMap<u64, u32> = HashMap::new();

        best_g.insert(initial_state, 0);
        wait_count.insert(initial_state, 0);

        let mut goal_state: Option<u64> = None;
        let mut expansions = 0usize;

        while let Some(SearchItem { f: _, g, tiebreak: _, node }) = frontier.pop() {
            expansions += 1;
            if expansions > self.max_expansions {
                return None;
            }

            let t = start_step + g;
            if t > horizon {
                continue;
            }

            let state = pack_state(node, t);
            if let Some(&prev_g) = best_g.get(&state) {
                if prev_g < g {
                    continue;
                }
            }

            if node == goal_idx {
                goal_state = Some(state);
                break;
            }

            let node_name = graph.node_name(node).unwrap();
            let current_wait = *wait_count.get(&state).unwrap_or(&0);

            // Action 1: Wait at current node
            if t + 1 <= horizon {
                let span = Interval::new_unchecked(t as i64, (t + 1) as i64);
                if reservations.is_node_free(
                    node_name,
                    span,
                    graph.node_capacity(node),
                    Some(agv_id),
                ) {
                    let ns = pack_state(node, t + 1);
                    let ng = g + 1;
                    if ng < *best_g.get(&ns).unwrap_or(&u32::MAX) {
                        best_g.insert(ns, ng);
                        parent.insert(ns, state);
                        wait_count.insert(ns, current_wait + 1);
                        counter += 1;
                        let h_val = h_table[node as usize];
                        frontier.push(SearchItem {
                            f: ng.saturating_add(h_val),
                            g: ng,
                            tiebreak: counter,
                            node,
                        });
                    }
                }
            }

            // Action 2: Move along outgoing edges
            for edge in graph.out_edges(node) {
                let n = edge.traversal_steps;
                let arrive = t + n;
                if arrive > horizon {
                    continue;
                }

                let dst_name = graph.node_name(edge.dst).unwrap();
                let edge_span = Interval::new_unchecked(t as i64, arrive as i64);

                // Physical bidirectional channel mutual exclusion
                let (p0, p1) = if node_name <= dst_name {
                    (node_name, dst_name)
                } else {
                    (dst_name, node_name)
                };

                let occ1 = reservations.edge_conflicts(p0, p1, edge_span, Some(agv_id));
                let occ2 = reservations.edge_conflicts(p1, p0, edge_span, Some(agv_id));

                let mut unique_occ: HashSet<String> = occ1.into_iter().collect();
                unique_occ.extend(occ2);

                if unique_occ.len() >= edge.physical_capacity {
                    continue;
                }

                let arrive_span = Interval::new_unchecked(arrive as i64, (arrive + 1) as i64);
                if !reservations.is_node_free(
                    dst_name,
                    arrive_span,
                    graph.node_capacity(edge.dst),
                    Some(agv_id),
                ) {
                    continue;
                }

                let ns = pack_state(edge.dst, arrive);
                let ng = g + n;
                if ng < *best_g.get(&ns).unwrap_or(&u32::MAX) {
                    best_g.insert(ns, ng);
                    parent.insert(ns, state);
                    wait_count.insert(ns, current_wait);
                    counter += 1;
                    let h_val = h_table[edge.dst as usize];
                    frontier.push(SearchItem {
                        f: ng.saturating_add(h_val),
                        g: ng,
                        tiebreak: counter,
                        node: edge.dst,
                    });
                }
            }
        }

        let end_state = goal_state?;
        let total_waits = *wait_count.get(&end_state).unwrap_or(&0);

        // Backtrack
        let mut chain: Vec<(u16, u32)> = Vec::new();
        let mut cur = end_state;
        chain.push(unpack_state(cur));

        while let Some(&p) = parent.get(&cur) {
            chain.push(unpack_state(p));
            cur = p;
        }
        chain.reverse();

        // Collapse consecutive identical nodes (waits)
        let mut collapsed: Vec<(u16, u32)> = Vec::new();
        for (n_idx, t_step) in chain {
            if let Some(last) = collapsed.last_mut() {
                if last.0 == n_idx {
                    last.1 = t_step;
                    continue;
                }
            }
            collapsed.push((n_idx, t_step));
        }

        let mut nodes = Vec::new();
        let mut arrivals = Vec::new();

        for &(n_idx, t_step) in collapsed.iter().skip(1) {
            nodes.push(graph.node_name(n_idx).unwrap().to_string());
            arrivals.push(t_step);
        }

        Some(PlannedPath {
            nodes,
            arrival_steps: arrivals,
            wait_steps: total_waits,
        })
    }
}
