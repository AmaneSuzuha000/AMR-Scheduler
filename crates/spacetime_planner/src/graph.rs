//! Topological Network Graph representation for path planning.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone)]
pub struct PlannerEdge {
    pub dst: u16,
    pub traversal_steps: u32,
    pub capacity: usize,
    /// Physical channel capacity (considering bidirectional channel)
    pub physical_capacity: usize,
    /// Lightweight Traffic Map online penalty in Q3 fixed-point steps (8 = +1 step)
    pub traffic_pen_q3: u32,
}

#[derive(Debug, Default)]
pub struct PlannerGraph {
    /// Mapping from string node name to integer index
    node_to_idx: HashMap<String, u16>,
    /// Mapping from integer index to string node name
    idx_to_node: Vec<String>,
    /// Node capacity by node index
    node_capacities: Vec<usize>,
    /// Outgoing edges per node
    outgoing: Vec<Vec<PlannerEdge>>,
    /// Incoming edges per node (for reverse Dijkstra)
    incoming: Vec<Vec<(u16, u32)>>,
    /// Cached reverse-Dijkstra shortest-path heuristic tables per goal index.
    h_cache: RwLock<Vec<Option<Arc<Vec<u32>>>>>,
    /// Time cutoff step for LTM traffic penalties (0 = entire horizon).
    traffic_cutoff_step: u32,
    /// Per-AGV soft corridor guidance: agv_id -> (packed directed edge codes (u<<16)|v, off_guide_pen_q3)
    agv_guides: HashMap<String, (Vec<u32>, u32)>,
}

impl Clone for PlannerGraph {
    fn clone(&self) -> Self {
        let cache = self
            .h_cache
            .read()
            .map(|c| c.clone())
            .unwrap_or_default();
        Self {
            node_to_idx: self.node_to_idx.clone(),
            idx_to_node: self.idx_to_node.clone(),
            node_capacities: self.node_capacities.clone(),
            outgoing: self.outgoing.clone(),
            incoming: self.incoming.clone(),
            h_cache: RwLock::new(cache),
            traffic_cutoff_step: self.traffic_cutoff_step,
            agv_guides: self.agv_guides.clone(),
        }
    }
}

impl PlannerGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a node if not already present, or updates its capacity if explicitly set.
    pub fn set_node_capacity(&mut self, name: &str, capacity: usize) -> u16 {
        if let Some(&idx) = self.node_to_idx.get(name) {
            self.node_capacities[idx as usize] = capacity;
            return idx;
        }
        let idx = self.idx_to_node.len() as u16;
        self.node_to_idx.insert(name.to_string(), idx);
        self.idx_to_node.push(name.to_string());
        self.node_capacities.push(capacity);
        self.outgoing.push(Vec::new());
        self.incoming.push(Vec::new());
        if let Ok(mut cache) = self.h_cache.write() {
            cache.clear();
        }
        idx
    }

    /// Adds a node with a default capacity if it doesn't already exist.
    /// Does NOT overwrite existing capacity if the node was already registered.
    pub fn add_node(&mut self, name: &str, default_capacity: usize) -> u16 {
        if let Some(&idx) = self.node_to_idx.get(name) {
            return idx;
        }
        let idx = self.idx_to_node.len() as u16;
        self.node_to_idx.insert(name.to_string(), idx);
        self.idx_to_node.push(name.to_string());
        self.node_capacities.push(default_capacity);
        self.outgoing.push(Vec::new());
        self.incoming.push(Vec::new());
        if let Ok(mut cache) = self.h_cache.write() {
            cache.clear();
        }
        idx
    }

    #[inline]
    pub fn node_index(&self, name: &str) -> Option<u16> {
        self.node_to_idx.get(name).copied()
    }

    #[inline]
    pub fn node_name(&self, idx: u16) -> Option<&str> {
        self.idx_to_node.get(idx as usize).map(|s| s.as_str())
    }

    #[inline]
    pub fn node_capacity(&self, idx: u16) -> usize {
        self.node_capacities[idx as usize]
    }

    #[inline]
    pub fn node_count(&self) -> usize {
        self.idx_to_node.len()
    }

    /// Adds a directed edge `src -> dst`.
    pub fn add_edge(
        &mut self,
        src: &str,
        dst: &str,
        traversal_steps: u32,
        capacity: usize,
        physical_capacity: usize,
    ) {
        let u = self.add_node(src, 1);
        let v = self.add_node(dst, 1);

        self.outgoing[u as usize].push(PlannerEdge {
            dst: v,
            traversal_steps,
            capacity,
            physical_capacity,
            traffic_pen_q3: 0,
        });

        self.incoming[v as usize].push((u, traversal_steps));
        if let Ok(mut cache) = self.h_cache.write() {
            cache.clear();
        }
    }

    /// Resets all online Lightweight Traffic Map edge penalties and window cutoff to 0.
    pub fn clear_traffic_penalties(&mut self) {
        self.traffic_cutoff_step = 0;
        for edges in &mut self.outgoing {
            for edge in edges {
                edge.traffic_pen_q3 = 0;
            }
        }
    }

    /// Sets time cutoff step for LTM penalties (0 applies over full planning horizon).
    pub fn set_traffic_cutoff(&mut self, cutoff_step: u32) {
        self.traffic_cutoff_step = cutoff_step;
    }

    /// Clears all per-AGV soft corridor guidance filters.
    pub fn clear_agv_guides(&mut self) {
        self.agv_guides.clear();
    }

    /// Clears existing guides and parses tab-delimited `agv_id\toff_pen_q3\tsrc\tdst\n` lines.
    pub fn set_agv_guide_script(&mut self, script: &str) -> usize {
        self.agv_guides.clear();
        let mut count = 0usize;
        for line in script.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let mut parts = trimmed.split('\t');
            let agv_id = match parts.next() {
                Some(v) => v,
                None => continue,
            };
            let off_pen: u32 = match parts.next().and_then(|v| v.parse().ok()) {
                Some(v) => v,
                None => continue,
            };
            let src = match parts.next() {
                Some(v) => v,
                None => continue,
            };
            let dst = match parts.next() {
                Some(v) => v,
                None => continue,
            };
            if let (Some(u), Some(v)) = (self.node_index(src), self.node_index(dst)) {
                let code = ((u as u32) << 16) | (v as u32);
                let entry = self
                    .agv_guides
                    .entry(agv_id.to_string())
                    .or_insert_with(|| (Vec::new(), off_pen));
                entry.1 = entry.1.max(off_pen);
                if !entry.0.contains(&code) {
                    entry.0.push(code);
                }
                count += 1;
            }
        }
        count
    }

    #[inline]
    pub fn agv_guide(&self, agv_id: &str) -> Option<(&[u32], u32)> {
        self.agv_guides
            .get(agv_id)
            .map(|(edges, pen)| (edges.as_slice(), *pen))
    }

    #[inline]
    pub fn edge_penalty_q3(
        &self,
        u: u16,
        edge: &PlannerEdge,
        depart_t: u32,
        guide: Option<(&[u32], u32)>,
    ) -> u32 {
        let in_window = self.traffic_cutoff_step == 0 || depart_t < self.traffic_cutoff_step;
        let mut pen = if in_window { edge.traffic_pen_q3 } else { 0 };
        if in_window {
            if let Some((edges, off_pen)) = guide {
                let code = ((u as u32) << 16) | (edge.dst as u32);
                if !edges.contains(&code) {
                    pen = pen.saturating_add(off_pen);
                }
            }
        }
        pen
    }

    /// Sets the online penalty `pen_q3` (Q3 steps: 8 == +1.0 step) by dense `u16` node indices.
    pub fn set_edge_traffic_penalty_indexed(&mut self, u: u16, v: u16, pen_q3: u32) -> bool {
        let ui = u as usize;
        if ui >= self.outgoing.len() {
            return false;
        }
        let mut updated = false;
        for edge in &mut self.outgoing[ui] {
            if edge.dst == v {
                edge.traffic_pen_q3 = pen_q3;
                updated = true;
            }
        }
        updated
    }

    /// Sets the online penalty `pen_q3` (Q3 steps: 8 == +1.0 step) on directed edge `src -> dst`.
    pub fn set_edge_traffic_penalty(&mut self, src: &str, dst: &str, pen_q3: u32) -> bool {
        let u = match self.node_index(src) {
            Some(idx) => idx,
            None => return false,
        };
        let v = match self.node_index(dst) {
            Some(idx) => idx,
            None => return false,
        };
        self.set_edge_traffic_penalty_indexed(u, v, pen_q3)
    }

    /// Clears existing traffic penalties and applies tab-delimited `src\tdst\tpen_q3\n` script.
    pub fn set_traffic_batch_script(&mut self, script: &str) -> usize {
        self.clear_traffic_penalties();
        let mut count = 0usize;
        for line in script.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let mut parts = trimmed.split('\t');
            let src = match parts.next() {
                Some(s) => s,
                None => continue,
            };
            let dst = match parts.next() {
                Some(d) => d,
                None => continue,
            };
            let pen_q3: u32 = match parts.next().and_then(|v| v.parse().ok()) {
                Some(v) => v,
                None => continue,
            };
            if self.set_edge_traffic_penalty(src, dst, pen_q3) {
                count += 1;
            }
        }
        count
    }

    #[inline]
    pub fn out_edges(&self, u: u16) -> &[PlannerEdge] {
        &self.outgoing[u as usize]
    }

    /// Returns cached reverse-Dijkstra heuristic vector from all nodes to `goal`.
    pub fn get_heuristics_arc(&self, goal: u16) -> Arc<Vec<u32>> {
        let g_idx = goal as usize;
        if let Ok(cache) = self.h_cache.read() {
            if let Some(Some(arc)) = cache.get(g_idx) {
                return Arc::clone(arc);
            }
        }
        let dist = Arc::new(self.compute_heuristics(goal));
        if g_idx < self.node_count() {
            if let Ok(mut cache) = self.h_cache.write() {
                if cache.len() < self.node_count() {
                    cache.resize(self.node_count(), None);
                }
                cache[g_idx] = Some(Arc::clone(&dist));
            }
        }
        dist
    }

    /// Computes static shortest travel time (in discrete steps) from all nodes to `goal`.
    pub fn compute_heuristics(&self, goal: u16) -> Vec<u32> {
        let n = self.node_count();
        let mut dist = vec![u32::MAX; n];

        if (goal as usize) >= n {
            return dist;
        }

        #[derive(Copy, Clone, Eq, PartialEq)]
        struct State {
            cost: u32,
            node: u16,
        }

        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                other.cost.cmp(&self.cost) // Min-heap
            }
        }

        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap = BinaryHeap::new();
        dist[goal as usize] = 0;
        heap.push(State {
            cost: 0,
            node: goal,
        });

        while let Some(State { cost, node }) = heap.pop() {
            if cost > dist[node as usize] {
                continue;
            }

            for &(prev, weight) in &self.incoming[node as usize] {
                let next_cost = cost.saturating_add(weight);
                if next_cost < dist[prev as usize] {
                    dist[prev as usize] = next_cost;
                    heap.push(State {
                        cost: next_cost,
                        node: prev,
                    });
                }
            }
        }

        dist
    }
}
