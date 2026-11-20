//! Native Spatiotemporal A* Search Engine.

use crate::graph::PlannerGraph;
use reservation_core::{Interval, ReservationTable};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedPath {
    pub nodes: Vec<String>,
    pub arrival_steps: Vec<u32>,
    pub wait_steps: u32,
}

/// Compact node-index path returned by zero-copy indexed search routines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedPath {
    pub nodes: Vec<u16>,
    pub arrivals: Vec<u32>,
    pub wait_steps: u32,
}

impl IndexedPath {
    /// Materialize string node names from a `PlannerGraph`.
    pub fn to_planned_path(&self, graph: &PlannerGraph) -> PlannedPath {
        PlannedPath {
            nodes: self
                .nodes
                .iter()
                .filter_map(|&idx| graph.node_name(idx).map(|s| s.to_string()))
                .collect(),
            arrival_steps: self.arrivals.clone(),
            wait_steps: self.wait_steps,
        }
    }
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
    score_g: u32,
    tiebreak: u64,
    node: u16,
}

impl Ord for SearchItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: lowest weighted f first, then tiebreak
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

    /// Plans a collision-free spatiotemporal trajectory from `start_node` to `goal_node`.
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

    /// Zero-allocation-return indexed Spatiotemporal A* search using `u16` node handles.
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

        // Precompute/fetch cached reverse static heuristic
        let h_table = graph.get_heuristics_arc(goal_idx);
        if h_table[start_idx as usize] == u32::MAX {
            // Unreachable
            return None;
        }
        let guide_ref = graph.agv_guide(agv_id);

        let mut counter = 0u64;
        let mut frontier = BinaryHeap::new();

        let initial_state = pack_state(start_idx, start_step);
        let h_start = h_table[start_idx as usize];

        frontier.push(SearchItem {
            f: h_start.saturating_mul(8),
            g: 0,
            score_g: 0,
            tiebreak: counter,
            node: start_idx,
        });

        // Unified state table: state -> (best_score_g, parent_state_or_MAX, wait_count)
        let mut state_info: HashMap<u64, (u32, u64, u32)> = HashMap::with_capacity(256);
        state_info.insert(initial_state, (0, u64::MAX, 0));

        let mut goal_state: Option<u64> = None;
        let mut expansions = 0usize;

        while let Some(SearchItem { f: _, g, score_g, tiebreak: _, node }) = frontier.pop() {
            expansions += 1;
            if expansions > self.max_expansions {
                return None;
            }

            let t = start_step + g;
            if t > horizon {
                continue;
            }

            let state = pack_state(node, t);
            let current_wait = match state_info.get(&state) {
                Some(&(prev_score, _, prev_wait)) => {
                    if prev_score < score_g {
                        continue;
                    }
                    prev_wait
                }
                None => 0,
            };

            if node == goal_idx {
                goal_state = Some(state);
                break;
            }

            let node_name = graph.node_name(node).unwrap();

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
                    let n_score = score_g.saturating_add(8);
                    let should_update = state_info
                        .get(&ns)
                        .map_or(true, |&(old_score, _, _)| n_score < old_score);
                    if should_update {
                        state_info.insert(ns, (n_score, state, current_wait + 1));
                        counter += 1;
                        let h_val = h_table[node as usize].saturating_mul(8);
                        frontier.push(SearchItem {
                            f: n_score.saturating_add(h_val),
                            g: ng,
                            score_g: n_score,
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

                // Physical bidirectional channel mutual exclusion (zero-HashSet fast check)
                let (p0, p1) = if node_name <= dst_name {
                    (node_name, dst_name)
                } else {
                    (dst_name, node_name)
                };

                if !reservations.is_undirected_edge_free(
                    p0,
                    p1,
                    edge_span,
                    edge.physical_capacity,
                    Some(agv_id),
                ) {
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
                let pen_q3 = graph.edge_penalty_q3(node, edge, t, guide_ref);
                let step_cost_q3 = n.saturating_mul(8).saturating_add(pen_q3);
                let n_score = score_g.saturating_add(step_cost_q3);
                let should_update = state_info
                    .get(&ns)
                    .map_or(true, |&(old_score, _, _)| n_score < old_score);
                if should_update {
                    state_info.insert(ns, (n_score, state, current_wait));
                    counter += 1;
                    let h_val = h_table[edge.dst as usize].saturating_mul(8);
                    frontier.push(SearchItem {
                        f: n_score.saturating_add(h_val),
                        g: ng,
                        score_g: n_score,
                        tiebreak: counter,
                        node: edge.dst,
                    });
                }
            }
        }

        let end_state = goal_state?;
        let total_waits = state_info.get(&end_state).map_or(0, |&(_, _, w)| w);

        // Backtrack
        let mut chain: Vec<(u16, u32)> = Vec::new();
        let mut cur = end_state;
        chain.push(unpack_state(cur));

        while let Some(&(_, p, _)) = state_info.get(&cur) {
            if p == u64::MAX {
                break;
            }
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

        let mut nodes = Vec::with_capacity(collapsed.len());
        let mut arrivals = Vec::with_capacity(collapsed.len());

        for &(n_idx, t_step) in collapsed.iter().skip(1) {
            nodes.push(n_idx);
            arrivals.push(t_step);
        }

        Some(IndexedPath {
            nodes,
            arrivals,
            wait_steps: total_waits,
        })
    }
}
