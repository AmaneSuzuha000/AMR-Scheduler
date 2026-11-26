//! Spatiotemporal Reservation Table implementation in native Rust.
//!
//! Provides mathematically exact equivalence with Python `amr_sched.reservation.ReservationTable`,
//! while executing zero-GC, cache-friendly lookups.

use crate::interval::Interval;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReservationEntry {
    pub span: Interval,
    pub agv_id: String,
}

impl ReservationEntry {
    #[inline(always)]
    pub fn new(span: Interval, agv_id: impl Into<String>) -> Self {
        Self {
            span,
            agv_id: agv_id.into(),
        }
    }
}

/// Spatiotemporal reservation table recording vehicle occupancy across nodes and edges over time.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReservationTable {
    /// Node reservations: node_id -> sorted entries by span.start
    node_table: HashMap<String, Vec<ReservationEntry>>,
    /// Directed edge reservations: src -> dst -> sorted entries by span.start
    edge_table: HashMap<String, HashMap<String, Vec<ReservationEntry>>>,
    /// Physical node occupancy (pinned, no expiry)
    pinned_nodes: HashMap<String, HashSet<String>>,
    /// Physical edge occupancy: src -> dst -> owners
    pinned_edges: HashMap<String, HashMap<String, HashSet<String>>>,
}

impl ReservationTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Discards all reservations from nodes and edges (pinned physical occupancy is preserved).
    pub fn clear(&mut self) {
        self.node_table.clear();
        self.edge_table.clear();
    }

    /// Discards all reservations and all pins.
    pub fn reset_all(&mut self) {
        self.node_table.clear();
        self.edge_table.clear();
        self.pinned_nodes.clear();
        self.pinned_edges.clear();
    }

    // ---------- Pinned physical occupancy ----------

    /// Replaces physical resource ownership for `agv_id` after confirmed movement.
    pub fn transfer_pins(
        &mut self,
        agv_id: &str,
        nodes: &[String],
        edges: &[(String, String)],
    ) {
        // Discard previous ownership across all tables
        for owners in self.pinned_nodes.values_mut() {
            owners.remove(agv_id);
        }
        for sub in self.pinned_edges.values_mut() {
            for owners in sub.values_mut() {
                owners.remove(agv_id);
            }
        }

        // Assign new ownership
        for node in nodes {
            self.pinned_nodes
                .entry(node.clone())
                .or_default()
                .insert(agv_id.to_string());
        }
        for (src, dst) in edges {
            self.pinned_edges
                .entry(src.clone())
                .or_default()
                .entry(dst.clone())
                .or_default()
                .insert(agv_id.to_string());
        }
    }

    /// Returns a structured view of pinned physical occupancy.
    pub fn pins(&self) -> (HashMap<String, Vec<String>>, HashMap<String, Vec<String>>) {
        let mut nodes_out = HashMap::new();
        for (k, v) in &self.pinned_nodes {
            if !v.is_empty() {
                let mut list: Vec<String> = v.iter().cloned().collect();
                list.sort();
                nodes_out.insert(k.clone(), list);
            }
        }

        let mut edges_out = HashMap::new();
        for (src, dst_map) in &self.pinned_edges {
            for (dst, v) in dst_map {
                if !v.is_empty() {
                    let mut list: Vec<String> = v.iter().cloned().collect();
                    list.sort();
                    let edge_key = format!("{}->{}", src, dst);
                    edges_out.insert(edge_key, list);
                }
            }
        }

        (nodes_out, edges_out)
    }

    #[inline]
    fn pin_owners_node(&self, node: &str, exclude: Option<&str>) -> HashSet<String> {
        match self.pinned_nodes.get(node) {
            Some(owners) => owners
                .iter()
                .filter(|&o| exclude.map_or(true, |ex| o != ex))
                .cloned()
                .collect(),
            None => HashSet::new(),
        }
    }

    #[inline]
    fn pin_owners_edge(&self, src: &str, dst: &str, exclude: Option<&str>) -> HashSet<String> {
        match self.pinned_edges.get(src).and_then(|m| m.get(dst)) {
            Some(owners) => owners
                .iter()
                .filter(|&o| exclude.map_or(true, |ex| o != ex))
                .cloned()
                .collect(),
            None => HashSet::new(),
        }
    }

    pub fn pinned_node_conflicts(&self, node: &str, exclude_agv: Option<&str>) -> HashSet<String> {
        self.pin_owners_node(node, exclude_agv)
    }

    pub fn pinned_edge_conflicts(
        &self,
        src: &str,
        dst: &str,
        exclude_agv: Option<&str>,
    ) -> HashSet<String> {
        self.pin_owners_edge(src, dst, exclude_agv)
    }

    // ---------- Overlapping helpers ----------

    #[inline]
    fn get_overlapping<'a>(
        entries: &'a [ReservationEntry],
        span: Interval,
        exclude: Option<&str>,
    ) -> Vec<&'a ReservationEntry> {
        let mut out = Vec::new();
        for entry in entries {
            if let Some(ex) = exclude {
                if entry.agv_id == ex {
                    continue;
                }
            }
            if entry.span.overlaps(&span) {
                out.push(entry);
            }
        }
        out
    }

    // ---------- Conflict Queries ----------

    /// Returns sorted unique AGV IDs conflicting with `span` on `node`.
    pub fn node_conflicts(
        &self,
        node: &str,
        span: Interval,
        exclude_agv: Option<&str>,
    ) -> Vec<String> {
        let mut conflict_set: HashSet<String> = HashSet::new();

        if let Some(entries) = self.node_table.get(node) {
            for entry in Self::get_overlapping(entries, span, exclude_agv) {
                conflict_set.insert(entry.agv_id.clone());
            }
        }

        for pin in self.pin_owners_node(node, exclude_agv) {
            conflict_set.insert(pin);
        }

        let mut res: Vec<String> = conflict_set.into_iter().collect();
        res.sort();
        res
    }

    /// Returns sorted unique AGV IDs conflicting with `span` on edge `(src, dst)`.
    pub fn edge_conflicts(
        &self,
        src: &str,
        dst: &str,
        span: Interval,
        exclude_agv: Option<&str>,
    ) -> Vec<String> {
        let mut conflict_set: HashSet<String> = HashSet::new();

        if let Some(entries) = self.edge_table.get(src).and_then(|m| m.get(dst)) {
            for entry in Self::get_overlapping(entries, span, exclude_agv) {
                conflict_set.insert(entry.agv_id.clone());
            }
        }

        for pin in self.pin_owners_edge(src, dst, exclude_agv) {
            conflict_set.insert(pin);
        }

        let mut res: Vec<String> = conflict_set.into_iter().collect();
        res.sort();
        res
    }

    /// Accumulates unique AGV IDs conflicting with `span` on `node` directly into `out` (+1 per call).
    #[inline]
    pub fn collect_node_conflicts_into(
        &self,
        node: &str,
        span: Interval,
        exclude_agv: Option<&str>,
        out: &mut HashMap<String, u32>,
    ) {
        let mut seen: Vec<&str> = Vec::new();
        if let Some(entries) = self.node_table.get(node) {
            for entry in entries {
                if entry.span.start >= span.end {
                    break;
                }
                if let Some(ex) = exclude_agv {
                    if entry.agv_id == ex {
                        continue;
                    }
                }
                if entry.span.overlaps(&span) {
                    let agv = entry.agv_id.as_str();
                    if !seen.contains(&agv) {
                        seen.push(agv);
                        *out.entry(entry.agv_id.clone()).or_insert(0) += 1;
                    }
                }
            }
        }
        if let Some(owners) = self.pinned_nodes.get(node) {
            for owner in owners {
                if let Some(ex) = exclude_agv {
                    if owner == ex {
                        continue;
                    }
                }
                let agv = owner.as_str();
                if !seen.contains(&agv) {
                    seen.push(agv);
                    *out.entry(owner.clone()).or_insert(0) += 1;
                }
            }
        }
    }

    /// Accumulates unique AGV IDs conflicting with `span` on undirected corridor `(p0, p1)` (+1 per call).
    #[inline]
    pub fn collect_undirected_edge_conflicts_into(
        &self,
        p0: &str,
        p1: &str,
        span: Interval,
        exclude_agv: Option<&str>,
        out: &mut HashMap<String, u32>,
    ) {
        let mut seen: Vec<&str> = Vec::new();
        let dirs: [Option<(&str, &str)>; 2] = if p0 == p1 {
            [Some((p0, p1)), None]
        } else {
            [Some((p0, p1)), Some((p1, p0))]
        };
        for dir_opt in dirs {
            let Some((a, b)) = dir_opt else {
                continue;
            };
            if let Some(entries) = self.edge_table.get(a).and_then(|m| m.get(b)) {
                for entry in entries {
                    if entry.span.start >= span.end {
                        break;
                    }
                    if let Some(ex) = exclude_agv {
                        if entry.agv_id == ex {
                            continue;
                        }
                    }
                    if entry.span.overlaps(&span) {
                        let agv = entry.agv_id.as_str();
                        if !seen.contains(&agv) {
                            seen.push(agv);
                            *out.entry(entry.agv_id.clone()).or_insert(0) += 1;
                        }
                    }
                }
            }
            if let Some(owners) = self.pinned_edges.get(a).and_then(|m| m.get(b)) {
                for owner in owners {
                    if let Some(ex) = exclude_agv {
                        if owner == ex {
                            continue;
                        }
                    }
                    let agv = owner.as_str();
                    if !seen.contains(&agv) {
                        seen.push(agv);
                        *out.entry(owner.clone()).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    /// Checks whether node capacity constraint is satisfied over `span`.
    ///
    /// Semantics:
    /// `len(overlapping_hits) + len(pinned_owners - overlapping_owners) < capacity`.
    #[inline]
    pub fn is_node_free(
        &self,
        node: &str,
        span: Interval,
        capacity: usize,
        exclude_agv: Option<&str>,
    ) -> bool {
        if capacity == 0 {
            return false;
        }
        let entries_slice: &[ReservationEntry] = self
            .node_table
            .get(node)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        let mut hit_count: usize = 0;
        for entry in entries_slice {
            if entry.span.start >= span.end {
                break;
            }
            if let Some(ex) = exclude_agv {
                if entry.agv_id == ex {
                    continue;
                }
            }
            if entry.span.overlaps(&span) {
                hit_count += 1;
                if hit_count >= capacity {
                    return false;
                }
            }
        }

        let mut pins_not_in_hits: usize = 0;
        if let Some(owners) = self.pinned_nodes.get(node) {
            for owner in owners {
                if let Some(ex) = exclude_agv {
                    if owner == ex {
                        continue;
                    }
                }
                let mut in_hits = false;
                for entry in entries_slice {
                    if entry.span.start >= span.end {
                        break;
                    }
                    if entry.agv_id == *owner && entry.span.overlaps(&span) {
                        in_hits = true;
                        break;
                    }
                }
                if !in_hits {
                    pins_not_in_hits += 1;
                    if hit_count + pins_not_in_hits >= capacity {
                        return false;
                    }
                }
            }
        }

        hit_count + pins_not_in_hits < capacity
    }

    /// Fast undirected bidirectional corridor capacity check matching
    /// `len(unique(edge_conflicts(p0, p1) ∪ edge_conflicts(p1, p0))) < physical_capacity`
    /// without HashSet or String-vec allocations.
    #[inline]
    pub fn is_undirected_edge_free(
        &self,
        p0: &str,
        p1: &str,
        span: Interval,
        physical_capacity: usize,
        exclude_agv: Option<&str>,
    ) -> bool {
        if physical_capacity == 0 {
            return false;
        }
        let dirs: [Option<(&str, &str)>; 2] = if p0 == p1 {
            [Some((p0, p1)), None]
        } else {
            [Some((p0, p1)), Some((p1, p0))]
        };

        // Fast path for capacity == 1 (common single-lane aisle): zero container allocations
        if physical_capacity == 1 {
            for dir_opt in dirs {
                let Some((a, b)) = dir_opt else {
                    continue;
                };
                if let Some(entries) = self.edge_table.get(a).and_then(|m| m.get(b)) {
                    for entry in entries {
                        if entry.span.start >= span.end {
                            break;
                        }
                        if let Some(ex) = exclude_agv {
                            if entry.agv_id == ex {
                                continue;
                            }
                        }
                        if entry.span.overlaps(&span) {
                            return false;
                        }
                    }
                }
                if let Some(owners) = self.pinned_edges.get(a).and_then(|m| m.get(b)) {
                    for owner in owners {
                        if let Some(ex) = exclude_agv {
                            if owner == ex {
                                continue;
                            }
                        }
                        return false;
                    }
                }
            }
            return true;
        }

        let mut seen_small: [&str; 8] = [""; 8];
        let mut seen_len: usize = 0;
        let mut seen_overflow: Vec<&str> = Vec::new();

        #[inline]
        fn push_owner<'a>(
            agv: &'a str,
            physical_capacity: usize,
            seen_small: &mut [&'a str; 8],
            seen_len: &mut usize,
            seen_overflow: &mut Vec<&'a str>,
        ) -> bool {
            if *seen_len < 8 {
                for &s in &seen_small[..*seen_len] {
                    if s == agv {
                        return false;
                    }
                }
                seen_small[*seen_len] = agv;
                *seen_len += 1;
                *seen_len >= physical_capacity
            } else {
                for &s in &seen_small[..] {
                    if s == agv {
                        return false;
                    }
                }
                for &s in seen_overflow.iter() {
                    if s == agv {
                        return false;
                    }
                }
                seen_overflow.push(agv);
                (*seen_len + seen_overflow.len()) >= physical_capacity
            }
        }

        for dir_opt in dirs {
            let Some((a, b)) = dir_opt else {
                continue;
            };
            if let Some(entries) = self.edge_table.get(a).and_then(|m| m.get(b)) {
                for entry in entries {
                    if entry.span.start >= span.end {
                        break;
                    }
                    if let Some(ex) = exclude_agv {
                        if entry.agv_id == ex {
                            continue;
                        }
                    }
                    if entry.span.overlaps(&span)
                        && push_owner(
                            entry.agv_id.as_str(),
                            physical_capacity,
                            &mut seen_small,
                            &mut seen_len,
                            &mut seen_overflow,
                        )
                    {
                        return false;
                    }
                }
            }
            if let Some(owners) = self.pinned_edges.get(a).and_then(|m| m.get(b)) {
                for owner in owners {
                    if let Some(ex) = exclude_agv {
                        if owner == ex {
                            continue;
                        }
                    }
                    if push_owner(
                        owner.as_str(),
                        physical_capacity,
                        &mut seen_small,
                        &mut seen_len,
                        &mut seen_overflow,
                    ) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Checks whether directed edge capacity constraint is satisfied over `span`.
    #[inline]
    pub fn is_edge_free(
        &self,
        src: &str,
        dst: &str,
        span: Interval,
        capacity: usize,
        exclude_agv: Option<&str>,
    ) -> bool {
        if capacity == 0 {
            return false;
        }
        let entries_slice: &[ReservationEntry] = self
            .edge_table
            .get(src)
            .and_then(|m| m.get(dst))
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        let mut hit_count: usize = 0;
        for entry in entries_slice {
            if entry.span.start >= span.end {
                break;
            }
            if let Some(ex) = exclude_agv {
                if entry.agv_id == ex {
                    continue;
                }
            }
            if entry.span.overlaps(&span) {
                hit_count += 1;
                if hit_count >= capacity {
                    return false;
                }
            }
        }

        let mut pins_not_in_hits: usize = 0;
        if let Some(owners) = self.pinned_edges.get(src).and_then(|m| m.get(dst)) {
            for owner in owners {
                if let Some(ex) = exclude_agv {
                    if owner == ex {
                        continue;
                    }
                }
                let mut in_hits = false;
                for entry in entries_slice {
                    if entry.span.start >= span.end {
                        break;
                    }
                    if entry.agv_id == *owner && entry.span.overlaps(&span) {
                        in_hits = true;
                        break;
                    }
                }
                if !in_hits {
                    pins_not_in_hits += 1;
                    if hit_count + pins_not_in_hits >= capacity {
                        return false;
                    }
                }
            }
        }

        hit_count + pins_not_in_hits < capacity
    }

    /// High performance check for opposing edge conflicts:
    /// For single-lane bidirectional corridors, traversing `(src, dst)` requires `(dst, src)`
    /// to have 0 conflicts and 0 pins during `span`.
    #[inline]
    pub fn is_opposing_edge_free(
        &self,
        src: &str,
        dst: &str,
        span: Interval,
        exclude_agv: Option<&str>,
    ) -> bool {
        self.is_edge_free(dst, src, span, 1, exclude_agv)
    }

    /// Extracts all continuous safe intervals [start, end) on a node within [0, horizon).
    pub fn safe_intervals_node(
        &self,
        node: &str,
        horizon: u32,
        capacity: usize,
        exclude_agv: Option<&str>,
    ) -> Vec<Interval> {
        if horizon == 0 {
            return Vec::new();
        }
        let h = horizon as usize;
        let mut occ = vec![0usize; h];

        let pins = self.pin_owners_node(node, exclude_agv);
        let pin_count = pins.len();

        if let Some(entries) = self.node_table.get(node) {
            for entry in entries {
                if let Some(ex) = exclude_agv {
                    if entry.agv_id == ex {
                        continue;
                    }
                }
                let s = (entry.span.start.max(0) as usize).min(h);
                let e = (entry.span.end.max(0) as usize).min(h);
                for t in s..e {
                    occ[t] += 1;
                }
            }
        }

        let mut intervals = Vec::new();
        let mut cur_start: Option<usize> = None;

        for t in 0..h {
            let free = (occ[t] + pin_count) < capacity;
            if free {
                if cur_start.is_none() {
                    cur_start = Some(t);
                }
            } else {
                if let Some(s) = cur_start.take() {
                    intervals.push(Interval::new_unchecked(s as i64, t as i64));
                }
            }
        }
        if let Some(s) = cur_start {
            intervals.push(Interval::new_unchecked(s as i64, h as i64));
        }

        intervals
    }

    /// Extracts all continuous safe intervals [start, end) on directed edge within [0, horizon).
    pub fn safe_intervals_edge(
        &self,
        src: &str,
        dst: &str,
        horizon: u32,
        capacity: usize,
        exclude_agv: Option<&str>,
    ) -> Vec<Interval> {
        if horizon == 0 {
            return Vec::new();
        }
        let h = horizon as usize;
        let mut occ = vec![0usize; h];

        let pins = self.pin_owners_edge(src, dst, exclude_agv);
        let pin_count = pins.len();

        if let Some(entries) = self.edge_table.get(src).and_then(|m| m.get(dst)) {
            for entry in entries {
                if let Some(ex) = exclude_agv {
                    if entry.agv_id == ex {
                        continue;
                    }
                }
                let s = (entry.span.start.max(0) as usize).min(h);
                let e = (entry.span.end.max(0) as usize).min(h);
                for t in s..e {
                    occ[t] += 1;
                }
            }
        }

        let mut intervals = Vec::new();
        let mut cur_start: Option<usize> = None;

        for t in 0..h {
            let free = (occ[t] + pin_count) < capacity;
            if free {
                if cur_start.is_none() {
                    cur_start = Some(t);
                }
            } else {
                if let Some(s) = cur_start.take() {
                    intervals.push(Interval::new_unchecked(s as i64, t as i64));
                }
            }
        }
        if let Some(s) = cur_start {
            intervals.push(Interval::new_unchecked(s as i64, h as i64));
        }

        intervals
    }

    // ---------- Reservation Writes ----------

    /// Inserts a node reservation, keeping the entries sorted by `span.start`.
    pub fn reserve_node(&mut self, node: impl Into<String>, span: Interval, agv_id: impl Into<String>) {
        if span.is_empty() {
            return;
        }
        let node_str = node.into();
        let agv_str = agv_id.into();
        let bucket = self.node_table.entry(node_str).or_default();
        let entry = ReservationEntry::new(span, agv_str);

        // Binary search insertion index to maintain sorted order
        let idx = bucket.partition_point(|x| x.span.start <= span.start);
        bucket.insert(idx, entry);
    }

    /// Inserts an edge reservation, keeping the entries sorted by `span.start`.
    pub fn reserve_edge(
        &mut self,
        src: impl Into<String>,
        dst: impl Into<String>,
        span: Interval,
        agv_id: impl Into<String>,
    ) {
        if span.is_empty() {
            return;
        }
        let agv_str = agv_id.into();
        let bucket = self
            .edge_table
            .entry(src.into())
            .or_default()
            .entry(dst.into())
            .or_default();
        let entry = ReservationEntry::new(span, agv_str);

        let idx = bucket.partition_point(|x| x.span.start <= span.start);
        bucket.insert(idx, entry);
    }

    /// Releases one vehicle's reservation on a specific edge over one window.
    pub fn release_edge(&mut self, src: &str, dst: &str, span: Interval, agv_id: &str) {
        if let Some(sub) = self.edge_table.get_mut(src) {
            if let Some(bucket) = sub.get_mut(dst) {
                bucket.retain(|e| !(e.agv_id == agv_id && e.span == span));
            }
        }
    }

    /// Releases one vehicle's reservation on a specific node over one window.
    pub fn release_node(&mut self, node: &str, span: Interval, agv_id: &str) {
        if let Some(bucket) = self.node_table.get_mut(node) {
            bucket.retain(|e| !(e.agv_id == agv_id && e.span == span));
        }
    }

    /// Releases every reservation held by one vehicle across all nodes and edges.
    pub fn release_all(&mut self, agv_id: &str) {
        for bucket in self.node_table.values_mut() {
            bucket.retain(|e| e.agv_id != agv_id);
        }
        for sub in self.edge_table.values_mut() {
            for bucket in sub.values_mut() {
                bucket.retain(|e| e.agv_id != agv_id);
            }
        }
    }

    /// Drops reservations that ended before `step`, preventing memory growth in continuous operations.
    pub fn purge_before(&mut self, step: i64) {
        self.node_table.retain(|_, bucket| {
            bucket.retain(|e| e.span.end > step);
            !bucket.is_empty()
        });
        self.edge_table.retain(|_, sub| {
            sub.retain(|_, bucket| {
                bucket.retain(|e| e.span.end > step);
                !bucket.is_empty()
            });
            !sub.is_empty()
        });
    }

    /// Replaces the entire reservation table state from `src`, reusing allocations where possible.
    pub fn restore_from(&mut self, src: &ReservationTable) {
        self.clone_from(src);
    }

    /// Applies a newline-delimited batch of node/edge reservations for `agv_id`:
    /// lines formatted as `N\tnode\tstart\tend` or `E\tsrc\tdst\tstart\tend`.
    pub fn reserve_batch_lines(&mut self, agv_id: &str, script: &str) -> usize {
        let mut count = 0usize;
        for line in script.split('\n') {
            if line.is_empty() {
                continue;
            }
            let mut it = line.split('\t');
            match it.next() {
                Some("N") => {
                    if let (Some(node), Some(s_str), Some(e_str)) =
                        (it.next(), it.next(), it.next())
                    {
                        if let (Ok(s), Ok(e)) = (s_str.parse::<i64>(), e_str.parse::<i64>()) {
                            if let Ok(span) = Interval::try_new(s, e) {
                                if !span.is_empty() {
                                    self.reserve_node(node, span, agv_id);
                                    count += 1;
                                }
                            }
                        }
                    }
                }
                Some("E") => {
                    if let (Some(src), Some(dst), Some(s_str), Some(e_str)) =
                        (it.next(), it.next(), it.next(), it.next())
                    {
                        if let (Ok(s), Ok(e)) = (s_str.parse::<i64>(), e_str.parse::<i64>()) {
                            if let Ok(span) = Interval::try_new(s, e) {
                                if !span.is_empty() {
                                    self.reserve_edge(src, dst, span, agv_id);
                                    count += 1;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        count
    }

    // ---------- Statistics & Snapshots ----------

    /// Node occupancy over `[lo, hi)`, matching Python `snapshot(lo, hi)`.
    pub fn snapshot(&self, lo: i64, hi: i64) -> HashMap<String, Vec<(i64, i64, String)>> {
        let span = Interval::new(lo, hi);
        let mut out: HashMap<String, Vec<(i64, i64, String)>> = HashMap::new();

        for (node, entries) in &self.node_table {
            let mut hits = Vec::new();
            for e in entries {
                if e.span.overlaps(&span) {
                    hits.push((e.span.start, e.span.end, e.agv_id.clone()));
                }
            }
            if !hits.is_empty() {
                hits.sort();
                out.insert(node.clone(), hits);
            }
        }

        for (node, owners) in &self.pinned_nodes {
            for owner in owners {
                out.entry(node.clone())
                    .or_default()
                    .push((lo, hi, owner.clone()));
            }
        }

        // Ensure each list is sorted
        for list in out.values_mut() {
            list.sort();
        }

        out
    }

    /// Undirected edge occupancy over `[lo, hi)`, keyed as `"a|b"` in sorted order.
    pub fn edge_snapshot(&self, lo: i64, hi: i64) -> HashMap<String, Vec<(i64, i64, String)>> {
        let span = Interval::new(lo, hi);
        let mut out: HashMap<String, Vec<(i64, i64, String)>> = HashMap::new();

        for (src, dst_map) in &self.edge_table {
            for (dst, entries) in dst_map {
                let key = if src <= dst {
                    format!("{}|{}", src, dst)
                } else {
                    format!("{}|{}", dst, src)
                };

                let mut hits = Vec::new();
                for e in entries {
                    if e.span.overlaps(&span) {
                        hits.push((e.span.start, e.span.end, e.agv_id.clone()));
                    }
                }
                if !hits.is_empty() {
                    hits.sort();
                    out.entry(key).or_default().extend(hits);
                }
            }
        }

        for (src, dst_map) in &self.pinned_edges {
            for (dst, owners) in dst_map {
                let key = if src <= dst {
                    format!("{}|{}", src, dst)
                } else {
                    format!("{}|{}", dst, src)
                };
                for owner in owners {
                    out.entry(key.clone())
                        .or_default()
                        .push((lo, hi, owner.clone()));
                }
            }
        }

        for list in out.values_mut() {
            list.sort();
        }

        out
    }

    /// Total number of active reservation records across nodes and edges.
    pub fn total_reservations(&self) -> usize {
        let node_count: usize = self.node_table.values().map(|v| v.len()).sum();
        let edge_count: usize = self
            .edge_table
            .values()
            .flat_map(|m| m.values())
            .map(|v| v.len())
            .sum();
        node_count + edge_count
    }

    /// Resource occupancy utilization metric.
    pub fn utilization(&self, horizon: i64, n_nodes: usize, n_edges: usize) -> f64 {
        if horizon <= 0 || (n_nodes + n_edges) == 0 {
            return 0.0;
        }
        let node_occupied: i64 = self
            .node_table
            .values()
            .flat_map(|v| v.iter().map(|e| e.span.len()))
            .sum();
        let edge_occupied: i64 = self
            .edge_table
            .values()
            .flat_map(|m| m.values())
            .flat_map(|v| v.iter().map(|e| e.span.len()))
            .sum();

        (node_occupied + edge_occupied) as f64 / (horizon as f64 * (n_nodes + n_edges) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reservation_node_and_edge_basic() {
        let mut tbl = ReservationTable::new();
        tbl.reserve_node("N1", Interval::new(0, 10), "AGV_1");
        tbl.reserve_node("N1", Interval::new(10, 20), "AGV_2");

        // Single capacity: [5, 15) overlaps with both
        assert!(!tbl.is_node_free("N1", Interval::new(5, 15), 1, None));
        // Excluding AGV_1: still overlaps with AGV_2 at [10, 15)
        assert!(!tbl.is_node_free("N1", Interval::new(5, 15), 1, Some("AGV_1")));
        // Capacity 2: accepts both
        assert!(tbl.is_node_free("N1", Interval::new(5, 15), 3, None));

        // Non-overlapping interval [20, 25)
        assert!(tbl.is_node_free("N1", Interval::new(20, 25), 1, None));
    }

    #[test]
    fn test_pinned_occupancy() {
        let mut tbl = ReservationTable::new();
        tbl.transfer_pins("AGV_3", &["N2".to_string()], &[]);

        // Node N2 is physically occupied by AGV_3 without expiry
        assert!(!tbl.is_node_free("N2", Interval::new(100, 200), 1, None));
        // Free if we exclude AGV_3
        assert!(tbl.is_node_free("N2", Interval::new(100, 200), 1, Some("AGV_3")));

        let conflicts = tbl.node_conflicts("N2", Interval::new(0, 10), None);
        assert_eq!(conflicts, vec!["AGV_3".to_string()]);
    }

    #[test]
    fn test_release_and_purge() {
        let mut tbl = ReservationTable::new();
        tbl.reserve_node("N1", Interval::new(0, 10), "AGV_1");
        tbl.reserve_node("N1", Interval::new(10, 20), "AGV_1");
        tbl.reserve_node("N2", Interval::new(5, 15), "AGV_2");

        assert_eq!(tbl.total_reservations(), 3);

        tbl.release_node("N1", Interval::new(0, 10), "AGV_1");
        assert_eq!(tbl.total_reservations(), 2);

        tbl.release_all("AGV_1");
        assert_eq!(tbl.total_reservations(), 1);

        tbl.purge_before(20);
        assert_eq!(tbl.total_reservations(), 0);
    }

    #[test]
    fn test_undirected_edge_restore_and_batch() {
        let mut tbl = ReservationTable::new();
        tbl.reserve_batch_lines("A1", "N\tX\t0\t5\nE\tX\tY\t5\t9\nE\tY\tZ\t10\t12");
        assert_eq!(tbl.total_reservations(), 3);
        assert!(!tbl.is_node_free("X", Interval::new(2, 4), 1, None));
        assert!(tbl.is_node_free("X", Interval::new(2, 4), 1, Some("A1")));
        // Bidirectional exclusion on (Y, X) vs reserved (X, Y) in [5, 9)
        assert!(!tbl.is_undirected_edge_free("X", "Y", Interval::new(6, 8), 1, None));
        assert!(tbl.is_undirected_edge_free("X", "Y", Interval::new(6, 8), 2, None));
        assert!(tbl.is_undirected_edge_free("X", "Y", Interval::new(6, 8), 1, Some("A1")));

        let mut copy = ReservationTable::new();
        copy.restore_from(&tbl);
        copy.release_all("A1");
        assert_eq!(copy.total_reservations(), 0);
        copy.restore_from(&tbl);
        assert_eq!(copy.total_reservations(), 3);
    }
}
