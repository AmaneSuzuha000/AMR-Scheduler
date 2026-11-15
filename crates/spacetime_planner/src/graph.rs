//! Topological Network Graph representation for path planning.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

#[derive(Debug, Clone)]
pub struct PlannerEdge {
    pub dst: u16,
    pub traversal_steps: u32,
    pub capacity: usize,
    /// Physical channel capacity (considering bidirectional channel)
    pub physical_capacity: usize,
}

#[derive(Debug, Clone, Default)]
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
        });

        self.incoming[v as usize].push((u, traversal_steps));
    }

    #[inline]
    pub fn out_edges(&self, u: u16) -> &[PlannerEdge] {
        &self.outgoing[u as usize]
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
