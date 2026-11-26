//! Native Spatiotemporal A* Search Engine.

use crate::graph::PlannerGraph;
use crate::slot_field::SlotCongestionField;
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
        // Collapse consecutive identical nodes preserving FIRST arrival step (matches Python PlannedPath convention)
        let mut collapsed: Vec<(u16, u32)> = Vec::new();
        for (n_idx, t_step) in chain {
            if collapsed.last().map_or(false, |last| last.0 == n_idx) {
                continue;
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

#[derive(Debug, Clone, Copy)]
pub struct CpsSearchParams {
    pub mode_oneshot: bool,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub lookahead_omega: f64,
    pub goal_park_until: Option<u32>,
    pub goal_park_steps: u32,
    pub prev_arrival_node: u16,
    pub max_expansions_override: usize,
    pub collect_blockers: bool,
}

impl Default for CpsSearchParams {
    fn default() -> Self {
        Self {
            mode_oneshot: false,
            alpha: 1.0,
            beta: 0.5,
            gamma: 5.0,
            lookahead_omega: 50.0,
            goal_park_until: None,
            goal_park_steps: 0,
            prev_arrival_node: u16::MAX,
            max_expansions_override: 0,
            collect_blockers: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CpsSearchResult {
    pub path: Option<IndexedPath>,
    pub expansions: usize,
    pub blocker_hits: Vec<(String, u32)>,
}

#[derive(Copy, Clone)]
pub(crate) struct OrdF64(pub f64);

impl PartialEq for OrdF64 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == Ordering::Equal
    }
}
impl Eq for OrdF64 {}
impl Ord for OrdF64 {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}
impl PartialOrd for OrdF64 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline(always)]
fn pack_cps_state(node: u16, prev_node: u16, t: u32) -> u64 {
    ((node as u64) << 48) | ((prev_node as u64) << 32) | (t as u64)
}

#[inline(always)]
fn unpack_cps_node_t(s: u64) -> (u16, u32) {
    (((s >> 48) & 0xFFFF) as u16, (s & 0xFFFF_FFFF) as u32)
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct CpsSearchItem {
    f: OrdF64,
    score_g: OrdF64,
    wait_cnt: u32,
    tiebreak: u64,
    node: u16,
    p_node: u16,
    t: u32,
}

impl Ord for CpsSearchItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f
            .cmp(&self.f)
            .then_with(|| other.score_g.cmp(&self.score_g))
            .then_with(|| other.wait_cnt.cmp(&self.wait_cnt))
            .then_with(|| other.tiebreak.cmp(&self.tiebreak))
    }
}

impl PartialOrd for CpsSearchItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub(crate) fn sort_blocker_hits(map: HashMap<String, u32>) -> Vec<(String, u32)> {
    let mut vec: Vec<(String, u32)> = map.into_iter().collect();
    vec.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    vec
}

impl SpacetimeAStarPlanner {
    /// CPS-A* parity search matching Python `CPSAStarGraphPlanner.plan_leg` with
    /// native `SlotCongestionField`, turning penalties, goal parking clearance, and blocker hits.
    pub fn plan_cps_indexed(
        &self,
        graph: &PlannerGraph,
        reservations: &ReservationTable,
        mut field: Option<&mut SlotCongestionField>,
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
        let goal_name = match graph.node_name(goal_idx) {
            Some(s) => s,
            None => return CpsSearchResult::default(),
        };
        let eff_park_end = |arr_t: u32| -> u32 {
            params
                .goal_park_until
                .unwrap_or_else(|| horizon.min(arr_t.saturating_add(1).saturating_add(params.goal_park_steps)))
        };
        if start_idx == goal_idx {
            let eff_park = eff_park_end(start_step);
            if eff_park <= start_step + 1
                || reservations.is_node_free(
                    goal_name,
                    Interval::new_unchecked((start_step + 1) as i64, eff_park as i64),
                    graph.node_capacity(goal_idx),
                    Some(agv_id),
                )
            {
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

        let h_table = graph.get_heuristics_time_arc(goal_idx);
        if !h_table[start_idx as usize].is_finite() {
            return CpsSearchResult::default();
        }
        let dt_s = graph.dt_s();
        let alpha = params.alpha;
        let beta = params.beta;
        let gamma = params.gamma;
        let omega = if params.mode_oneshot { params.lookahead_omega } else { 0.0 };
        let use_prev = beta > 0.0;
        let init_prev = if use_prev { params.prev_arrival_node } else { u16::MAX };
        let max_exp = if params.max_expansions_override > 0 {
            params.max_expansions_override
        } else {
            self.max_expansions
        };
        let guide_ref = graph.agv_guide(agv_id);

        #[inline]
        fn h_total_fn(
            node: u16,
            step: u32,
            h_table: &[f64],
            dt_s: f64,
            alpha: f64,
            omega: f64,
            field: &Option<&mut SlotCongestionField>,
        ) -> f64 {
            let v = h_table[node as usize];
            let hb = if v.is_finite() { alpha * (v / dt_s).floor() } else { 1e9 };
            if omega > 0.0 {
                if let Some(ref fld) = field {
                    return hb + omega * fld.phi_bar_read(node, step as i32);
                }
            }
            hb
        }

        let mut counter = 0u64;
        let mut frontier = BinaryHeap::with_capacity(128);
        let init_state = pack_cps_state(start_idx, init_prev, start_step);
        let h0 = h_total_fn(start_idx, start_step, &h_table, dt_s, alpha, omega, &field);
        frontier.push(CpsSearchItem {
            f: OrdF64(h0),
            score_g: OrdF64(0.0),
            wait_cnt: 0,
            tiebreak: counter,
            node: start_idx,
            p_node: init_prev,
            t: start_step,
        });

        let mut best_cost: HashMap<u64, f64> = HashMap::with_capacity(256);
        let mut parent: HashMap<u64, u64> = HashMap::with_capacity(256);
        let mut wait_map: HashMap<u64, u32> = HashMap::with_capacity(256);
        let mut blocker_map: HashMap<String, u32> = HashMap::new();
        best_cost.insert(init_state, 0.0);
        parent.insert(init_state, u64::MAX);
        wait_map.insert(init_state, 0);

        let mut goal_state: Option<u64> = None;
        let mut expansions = 0usize;

        while let Some(CpsSearchItem { f: _, score_g: cur_g_w, wait_cnt: w_c, tiebreak: _, node, p_node, t }) = frontier.pop() {
            expansions += 1;
            if expansions > max_exp {
                break;
            }
            if t > horizon {
                continue;
            }
            let cur_g = cur_g_w.0;
            let state = pack_cps_state(node, p_node, t);
            if best_cost.get(&state).copied().unwrap_or(1e18) + 1e-9 < cur_g {
                continue;
            }

            if node == goal_idx {
                let eff_park = eff_park_end(t);
                if eff_park > t + 1 {
                    let park_span = Interval::new_unchecked((t + 1) as i64, eff_park as i64);
                    if reservations.is_node_free(goal_name, park_span, graph.node_capacity(goal_idx), Some(agv_id)) {
                        goal_state = Some(state);
                        break;
                    }
                    if params.collect_blockers {
                        reservations.collect_node_conflicts_into(goal_name, park_span, Some(agv_id), &mut blocker_map);
                    }
                } else {
                    goal_state = Some(state);
                    break;
                }
            }

            let node_name = graph.node_name(node).unwrap();

            // Action 1: Wait at `node` `[t + 1, t + 2)` matching Python CPS-A* / `reserve_mapf_path`
            if t + 1 <= horizon {
                let span = Interval::new_unchecked((t + 1) as i64, (t + 2) as i64);
                if reservations.is_node_free(node_name, span, graph.node_capacity(node), Some(agv_id)) {
                    let cong_pen = if gamma > 0.0 {
                        if let Some(ref mut fld) = field {
                            gamma * if params.mode_oneshot {
                                fld.phi_cached(node, (t + 1) as i32)
                            } else {
                                fld.slot_phi_cached(node, (t + 1) as i32)
                            }
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    let ng = cur_g + alpha + cong_pen;
                    let ns = pack_cps_state(node, p_node, t + 1);
                    if ng + 1e-9 < best_cost.get(&ns).copied().unwrap_or(1e18) {
                        best_cost.insert(ns, ng);
                        parent.insert(ns, state);
                        wait_map.insert(ns, w_c + 1);
                        counter += 1;
                        let ht = h_total_fn(node, t + 1, &h_table, dt_s, alpha, omega, &field);
                        frontier.push(CpsSearchItem {
                            f: OrdF64(ng + ht),
                            score_g: OrdF64(ng),
                            wait_cnt: w_c + 1,
                            tiebreak: counter,
                            node,
                            p_node,
                            t: t + 1,
                        });
                    }
                } else if params.collect_blockers {
                    reservations.collect_node_conflicts_into(node_name, span, Some(agv_id), &mut blocker_map);
                }
            }

            // Action 2: Edge transition
            for edge in graph.out_edges(node) {
                let n = edge.traversal_steps;
                let arrive = t + n;
                if arrive > horizon {
                    continue;
                }
                let dst_name = graph.node_name(edge.dst).unwrap();
                let edge_span = Interval::new_unchecked(t as i64, arrive as i64);
                let (p0, p1) = if node_name <= dst_name {
                    (node_name, dst_name)
                } else {
                    (dst_name, node_name)
                };

                if !reservations.is_undirected_edge_free(p0, p1, edge_span, edge.physical_capacity, Some(agv_id)) {
                    if params.collect_blockers {
                        reservations.collect_undirected_edge_conflicts_into(p0, p1, edge_span, Some(agv_id), &mut blocker_map);
                    }
                    continue;
                }

                let arrive_span = Interval::new_unchecked(arrive as i64, (arrive + 1) as i64);
                if !reservations.is_node_free(dst_name, arrive_span, graph.node_capacity(edge.dst), Some(agv_id)) {
                    if params.collect_blockers {
                        reservations.collect_node_conflicts_into(dst_name, arrive_span, Some(agv_id), &mut blocker_map);
                    }
                    continue;
                }

                let turn_c = if use_prev {
                    beta * graph.turn_penalty(p_node, node, edge.dst)
                } else {
                    0.0
                };
                let cong_move = if params.mode_oneshot && gamma > 0.0 {
                    if let Some(ref mut fld) = field {
                        gamma * fld.phi_cached(edge.dst, arrive as i32)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                let ltm_pen = (graph.edge_penalty_q3(node, edge, t, guide_ref) as f64) * 0.125;
                let ng = cur_g + alpha * (n as f64) + turn_c + cong_move + ltm_pen;
                let next_prev = if use_prev { node } else { u16::MAX };
                let ns = pack_cps_state(edge.dst, next_prev, arrive);
                if ng + 1e-9 < best_cost.get(&ns).copied().unwrap_or(1e18) {
                    best_cost.insert(ns, ng);
                    parent.insert(ns, state);
                    wait_map.insert(ns, w_c);
                    counter += 1;
                    let ht = h_total_fn(edge.dst, arrive, &h_table, dt_s, alpha, omega, &field);
                    frontier.push(CpsSearchItem {
                        f: OrdF64(ng + ht),
                        score_g: OrdF64(ng),
                        wait_cnt: w_c,
                        tiebreak: counter,
                        node: edge.dst,
                        p_node: next_prev,
                        t: arrive,
                    });
                }
            }
        }

        let blockers = sort_blocker_hits(blocker_map);
        let Some(end_state) = goal_state else {
            return CpsSearchResult {
                path: None,
                expansions,
                blocker_hits: blockers,
            };
        };

        let total_waits = wait_map.get(&end_state).copied().unwrap_or(0);
        let mut chain: Vec<(u16, u32)> = Vec::new();
        let mut cur = end_state;
        chain.push(unpack_cps_node_t(cur));
        while let Some(&p) = parent.get(&cur) {
            if p == u64::MAX {
                break;
            }
            chain.push(unpack_cps_node_t(p));
            cur = p;
        }
        chain.reverse();

        // Collapse consecutive identical nodes preserving FIRST arrival step (matches Python PlannedPath convention)
        let mut collapsed: Vec<(u16, u32)> = Vec::new();
        for (n_idx, t_step) in chain {
            if collapsed.last().map_or(false, |last| last.0 == n_idx) {
                continue;
            }
            collapsed.push((n_idx, t_step));
        }

        let mut nodes = Vec::with_capacity(collapsed.len());
        let mut arrivals = Vec::with_capacity(collapsed.len());
        for &(n_idx, t_step) in collapsed.iter().skip(1) {
            nodes.push(n_idx);
            arrivals.push(t_step);
        }

        CpsSearchResult {
            path: Some(IndexedPath {
                nodes,
                arrivals,
                wait_steps: total_waits,
            }),
            expansions,
            blocker_hits: blockers,
        }
    }
}
