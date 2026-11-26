//! Native 3D Graph-Spatiotemporal Gaussian & Slot Congestion Field.
//!
//! Matches `GraphCongestionField` (CPS-A* Section 4.2-4.4) with dense `u16` node indices,
//! compact `u64` spacetime keys, and incremental localized cache invalidation (`Seq-Diff` discipline).

use crate::graph::PlannerGraph;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Copy)]
pub struct SlotFieldConfig {
    pub spatial_radius_l: u32,
    pub temporal_radius_t: u32,
    pub sigma_l: f64,
    pub sigma_t: f64,
    pub lookahead_ks: u32,
    pub lookahead_kt: u32,
    pub slot_delta: u32,
    pub max_track_horizon: i32,
    pub t_start: i32,
}

impl Default for SlotFieldConfig {
    fn default() -> Self {
        Self {
            spatial_radius_l: 1,
            temporal_radius_t: 1,
            sigma_l: 2.0,
            sigma_t: 2.0,
            lookahead_ks: 1,
            lookahead_kt: 3,
            slot_delta: 3,
            max_track_horizon: 220,
            t_start: 0,
        }
    }
}

#[inline(always)]
fn pack_key(node_idx: u16, coord: i32) -> u64 {
    ((node_idx as u64) << 32) | ((coord as u32) as u64)
}

#[derive(Debug, Clone)]
pub struct SlotCongestionField {
    pub cfg: SlotFieldConfig,
    n_nodes: usize,
    l_rad: u32,
    t_rad: i32,
    t_slot_rad: i32,
    slot_delta: i32,
    hop_neighbors: Vec<Vec<(u16, u32)>>,
    gauss_kernel: Vec<Vec<(u16, i32, f64)>>,
    gauss_denom: Vec<f64>,
    lookahead_kernel: Vec<Vec<(u16, i32, f64)>>,
    lookahead_denom: Vec<f64>,
    slot_kernel: Vec<Vec<(u16, i32, f64)>>,
    slot_denom: Vec<f64>,

    agv_points: HashMap<String, Vec<(u16, i32)>>,
    agv_slots: HashMap<String, Vec<((u16, i32), f64)>>,
    psi: HashMap<u64, u32>,
    psi_slot: HashMap<u64, f64>,
    phi_cache: HashMap<u64, f64>,
    slot_phi_cache: HashMap<u64, f64>,
}

impl SlotCongestionField {
    pub fn new(graph: &PlannerGraph, cfg: SlotFieldConfig) -> Self {
        let n = graph.node_count();
        let l_rad = cfg.spatial_radius_l;
        let t_rad = cfg.temporal_radius_t as i32;
        let ks = cfg.lookahead_ks;
        let kt = cfg.lookahead_kt.max(1) as i32;
        let max_hop = l_rad.max(ks);
        let slot_delta = cfg.slot_delta.max(1) as i32;
        let t_slot_rad = ((t_rad as f64) / (slot_delta as f64)).ceil() as i32;

        let s_l2 = (2.0 * cfg.sigma_l * cfg.sigma_l).max(1e-6);
        let s_t2 = (2.0 * cfg.sigma_t * cfg.sigma_t).max(1e-6);

        let mut hop_neighbors = Vec::with_capacity(n);
        let mut gauss_kernel = Vec::with_capacity(n);
        let mut gauss_denom = Vec::with_capacity(n);
        let mut lookahead_kernel = Vec::with_capacity(n);
        let mut lookahead_denom = Vec::with_capacity(n);
        let mut slot_kernel = Vec::with_capacity(n);
        let mut slot_denom = Vec::with_capacity(n);

        let mut dists: Vec<u32> = vec![u32::MAX; n];
        let mut touched: Vec<u16> = Vec::with_capacity(64);
        let mut dq: VecDeque<u16> = VecDeque::with_capacity(64);

        for src in 0..n {
            touched.clear();
            dq.clear();
            let s_u16 = src as u16;
            dists[src] = 0;
            touched.push(s_u16);
            dq.push_back(s_u16);

            let mut hops_vec: Vec<(u16, u32)> = Vec::new();
            while let Some(cur) = dq.pop_front() {
                let cd = dists[cur as usize];
                hops_vec.push((cur, cd));
                if cd >= max_hop {
                    continue;
                }
                for &nxt in graph.undirected_adjacent(cur) {
                    if dists[nxt as usize] == u32::MAX {
                        dists[nxt as usize] = cd + 1;
                        touched.push(nxt);
                        dq.push_back(nxt);
                    }
                }
            }
            for &nd in &touched {
                dists[nd as usize] = u32::MAX;
            }
            hops_vec.sort_by_key(|&(u, d)| (d, u));

            // 1. Exact spacetime Gaussian kernel
            let mut g_items = Vec::new();
            let mut g_den = 0.0;
            for &(u, d_hop) in &hops_vec {
                if d_hop > l_rad {
                    continue;
                }
                let sp_exp = -((d_hop * d_hop) as f64) / s_l2;
                for dt in -t_rad..=t_rad {
                    let w = (sp_exp - ((dt * dt) as f64) / s_t2).exp();
                    g_items.push((u, dt, w));
                    g_den += w;
                }
            }

            // 2. Forward look-ahead kernel
            let mut la_items = Vec::new();
            let mut la_den = 0.0;
            for &(u, d_hop) in &hops_vec {
                if d_hop > ks {
                    continue;
                }
                for dt in 1..=kt {
                    let w = (-0.5 * ((d_hop as f64) + (dt as f64))).exp();
                    la_items.push((u, dt, w));
                    la_den += w;
                }
            }

            // 3. Delta-slot smoothed kernel
            let mut sl_items = Vec::new();
            let mut sl_den = 0.0;
            for &(u, d_hop) in &hops_vec {
                if d_hop > l_rad {
                    continue;
                }
                let sp_exp = -((d_hop * d_hop) as f64) / s_l2;
                for ds in -t_slot_rad..=t_slot_rad {
                    let dt_eff = (ds * slot_delta) as f64;
                    let w = (sp_exp - (dt_eff * dt_eff) / s_t2).exp();
                    sl_items.push((u, ds, w));
                    sl_den += w;
                }
            }

            hop_neighbors.push(hops_vec);
            gauss_kernel.push(g_items);
            gauss_denom.push(g_den.max(1e-9));
            lookahead_kernel.push(la_items);
            lookahead_denom.push(la_den.max(1e-9));
            slot_kernel.push(sl_items);
            slot_denom.push(sl_den.max(1e-9));
        }

        Self {
            cfg,
            n_nodes: n,
            l_rad,
            t_rad,
            t_slot_rad,
            slot_delta,
            hop_neighbors,
            gauss_kernel,
            gauss_denom,
            lookahead_kernel,
            lookahead_denom,
            slot_kernel,
            slot_denom,
            agv_points: HashMap::new(),
            agv_slots: HashMap::new(),
            psi: HashMap::with_capacity(1024),
            psi_slot: HashMap::with_capacity(512),
            phi_cache: HashMap::with_capacity(2048),
            slot_phi_cache: HashMap::with_capacity(1024),
        }
    }

    pub fn clear(&mut self) {
        self.agv_points.clear();
        self.agv_slots.clear();
        self.psi.clear();
        self.psi_slot.clear();
        self.phi_cache.clear();
        self.slot_phi_cache.clear();
    }

    pub fn restore_from(&mut self, other: &SlotCongestionField) {
        self.agv_points = other.agv_points.clone();
        self.agv_slots = other.agv_slots.clone();
        self.psi = other.psi.clone();
        self.psi_slot = other.psi_slot.clone();
        self.phi_cache = other.phi_cache.clone();
        self.slot_phi_cache = other.slot_phi_cache.clone();
    }

    fn invalidate_affected_cache(
        &mut self,
        pts: &[(u16, i32)],
        slots: &[((u16, i32), f64)],
    ) {
        if !pts.is_empty() && !self.phi_cache.is_empty() {
            for &(u_pt, step_pt) in pts {
                if (u_pt as usize) >= self.hop_neighbors.len() {
                    continue;
                }
                for &(v_n, d_hop) in &self.hop_neighbors[u_pt as usize] {
                    if d_hop > self.l_rad {
                        break;
                    }
                    for dt in -self.t_rad..=self.t_rad {
                        self.phi_cache.remove(&pack_key(v_n, step_pt + dt));
                    }
                }
            }
        }
        if !slots.is_empty() && !self.slot_phi_cache.is_empty() {
            for &((u_pt, sl_pt), _) in slots {
                if (u_pt as usize) >= self.hop_neighbors.len() {
                    continue;
                }
                for &(v_n, d_hop) in &self.hop_neighbors[u_pt as usize] {
                    if d_hop > self.l_rad {
                        break;
                    }
                    for ds in -self.t_slot_rad..=self.t_slot_rad {
                        self.slot_phi_cache.remove(&pack_key(v_n, sl_pt + ds));
                    }
                }
            }
        }
    }

    pub fn remove_agv(&mut self, agv_id: &str) {
        let pts = self.agv_points.remove(agv_id);
        if let Some(ref pvec) = pts {
            for &(nd, st) in pvec {
                let k = pack_key(nd, st);
                if let Some(cnt) = self.psi.get_mut(&k) {
                    if *cnt > 1 {
                        *cnt -= 1;
                    } else {
                        self.psi.remove(&k);
                    }
                }
            }
        }
        let slots = self.agv_slots.remove(agv_id);
        if let Some(ref svec) = slots {
            for &((nd, sl), amt) in svec {
                let k = pack_key(nd, sl);
                if let Some(cur) = self.psi_slot.get_mut(&k) {
                    let nv = *cur - amt;
                    if nv > 1e-9 {
                        *cur = nv;
                    } else {
                        self.psi_slot.remove(&k);
                    }
                }
            }
        }
        self.invalidate_affected_cache(
            pts.as_deref().unwrap_or(&[]),
            slots.as_deref().unwrap_or(&[]),
        );
    }

    /// Registers deduplicated `(node_idx, step)` trajectory points for `agv_id`.
    /// Matches Python `GraphCongestionField.add_agv_path` endpoint semantics on transition sequences.
    pub fn add_agv_transitions(
        &mut self,
        agv_id: &str,
        node_idxs: &[u16],
        steps: &[i32],
        horizon: Option<i32>,
    ) {
        self.remove_agv(agv_id);
        let m = node_idxs.len().min(steps.len());
        if m == 0 {
            return;
        }
        let t_start = self.cfg.t_start;
        let mut t_cap = t_start + self.cfg.max_track_horizon;
        if let Some(hz) = horizon {
            t_cap = t_cap.min(hz);
        }

        let mut pt_set: HashSet<(u16, i32)> = HashSet::with_capacity(m * 2);
        for idx in 0..(m - 1) {
            let u_n = node_idxs[idx];
            let v_n = node_idxs[idx + 1];
            let u_ti = steps[idx].max(t_start);
            let v_ti = steps[idx + 1].min(t_cap);
            if v_ti < u_ti || (u_n as usize) >= self.n_nodes || (v_n as usize) >= self.n_nodes {
                continue;
            }
            if u_n == v_n {
                for st in u_ti..=v_ti.min(t_cap) {
                    pt_set.insert((u_n, st));
                }
            } else {
                pt_set.insert((u_n, u_ti));
                if v_ti <= t_cap {
                    pt_set.insert((v_n, v_ti));
                }
            }
        }
        let last_n = node_idxs[m - 1];
        let last_t = steps[m - 1];
        if last_t <= t_cap && (last_n as usize) < self.n_nodes {
            pt_set.insert((last_n, last_t.max(t_start)));
        }
        if pt_set.is_empty() {
            return;
        }

        let mut pts: Vec<(u16, i32)> = pt_set.into_iter().collect();
        pts.sort_unstable();

        let inv_d = 1.0 / (self.slot_delta as f64);
        let mut slot_map: HashMap<(u16, i32), f64> = HashMap::with_capacity(pts.len());
        for &(nd, st) in &pts {
            let pk = pack_key(nd, st);
            *self.psi.entry(pk).or_insert(0) += 1;
            let sl = st.div_euclid(self.slot_delta);
            *slot_map.entry((nd, sl)).or_insert(0.0) += inv_d;
        }
        let slots: Vec<((u16, i32), f64)> = slot_map.into_iter().collect();
        for &((nd, sl), amt) in &slots {
            let sk = pack_key(nd, sl);
            let cur = self.psi_slot.entry(sk).or_insert(0.0);
            *cur += amt;
        }

        self.invalidate_affected_cache(&pts, &slots);
        self.agv_points.insert(agv_id.to_string(), pts);
        self.agv_slots.insert(agv_id.to_string(), slots);
    }

    #[inline]
    pub fn phi_read(&self, node: u16, step: i32) -> f64 {
        let ni = node as usize;
        if ni >= self.n_nodes {
            return 0.0;
        }
        let k = pack_key(node, step);
        if let Some(&cached) = self.phi_cache.get(&k) {
            return cached;
        }
        let mut num = 0.0;
        for &(u, dt, w) in &self.gauss_kernel[ni] {
            if let Some(&occ) = self.psi.get(&pack_key(u, step + dt)) {
                if occ > 0 {
                    num += (occ as f64) * w;
                }
            }
        }
        num / self.gauss_denom[ni]
    }

    #[inline]
    pub fn phi_cached(&mut self, node: u16, step: i32) -> f64 {
        let ni = node as usize;
        if ni >= self.n_nodes {
            return 0.0;
        }
        let k = pack_key(node, step);
        if let Some(&cached) = self.phi_cache.get(&k) {
            return cached;
        }
        let mut num = 0.0;
        for &(u, dt, w) in &self.gauss_kernel[ni] {
            if let Some(&occ) = self.psi.get(&pack_key(u, step + dt)) {
                if occ > 0 {
                    num += (occ as f64) * w;
                }
            }
        }
        let val = num / self.gauss_denom[ni];
        self.phi_cache.insert(k, val);
        val
    }

    #[inline]
    pub fn phi_bar_read(&self, node: u16, step: i32) -> f64 {
        let ni = node as usize;
        if ni >= self.n_nodes {
            return 0.0;
        }
        let mut num = 0.0;
        for &(u, dt, w) in &self.lookahead_kernel[ni] {
            let p_val = self.phi_read(u, step + dt);
            if p_val > 0.0 {
                num += p_val * w;
            }
        }
        num / self.lookahead_denom[ni]
    }

    #[inline]
    pub fn slot_phi_read(&self, node: u16, step: i32) -> f64 {
        let ni = node as usize;
        if ni >= self.n_nodes {
            return 0.0;
        }
        let slot = step.div_euclid(self.slot_delta);
        let k = pack_key(node, slot);
        if let Some(&cached) = self.slot_phi_cache.get(&k) {
            return cached;
        }
        let mut num = 0.0;
        for &(u, ds, w) in &self.slot_kernel[ni] {
            if let Some(&occ) = self.psi_slot.get(&pack_key(u, slot + ds)) {
                if occ > 0.0 {
                    num += occ * w;
                }
            }
        }
        num / self.slot_denom[ni]
    }

    #[inline]
    pub fn slot_phi_cached(&mut self, node: u16, step: i32) -> f64 {
        let ni = node as usize;
        if ni >= self.n_nodes {
            return 0.0;
        }
        let slot = step.div_euclid(self.slot_delta);
        let k = pack_key(node, slot);
        if let Some(&cached) = self.slot_phi_cache.get(&k) {
            return cached;
        }
        let mut num = 0.0;
        for &(u, ds, w) in &self.slot_kernel[ni] {
            if let Some(&occ) = self.psi_slot.get(&pack_key(u, slot + ds)) {
                if occ > 0.0 {
                    num += occ * w;
                }
            }
        }
        let val = num / self.slot_denom[ni];
        self.slot_phi_cache.insert(k, val);
        val
    }
}
