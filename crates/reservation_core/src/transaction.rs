//! Atomic Commit and Multi-Hop Execution Transaction Engine.
//!
//! Enforces ACID semantics on multi-hop trajectory reservations:
//! - **Atomicity**: An entire multi-step trajectory is committed all-at-once or not at all.
//! - **Consistency**: Mutual exclusion and corridor capacities are verified atomically before write.
//! - **Isolation**: Uncommitted/tentative paths do not leak to other concurrent planners.
//! - **Durability**: Firm commitments survive replanning preemption.

use crate::interval::Interval;
use crate::table::ReservationTable;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    Staging,
    Committed,
    RolledBack,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionError {
    CapacityExceeded { resource: String, span: Interval },
    AlreadyCommitted,
    AlreadyRolledBack,
}

/// A staged multi-hop trajectory reservation transaction.
#[derive(Debug, Clone)]
pub struct ReservationTransaction {
    pub agv_id: String,
    pub state: TransactionState,
    pub staged_nodes: Vec<(String, Interval, usize)>,
    pub staged_edges: Vec<(String, String, Interval, usize)>,
}

impl ReservationTransaction {
    pub fn begin(agv_id: impl Into<String>) -> Self {
        Self {
            agv_id: agv_id.into(),
            state: TransactionState::Staging,
            staged_nodes: Vec::new(),
            staged_edges: Vec::new(),
        }
    }

    pub fn stage_node(&mut self, node: impl Into<String>, span: Interval) {
        self.stage_node_with_cap(node, span, 1);
    }

    pub fn stage_node_with_cap(&mut self, node: impl Into<String>, span: Interval, cap: usize) {
        if self.state != TransactionState::Staging {
            return;
        }
        if !span.is_empty() {
            self.staged_nodes.push((node.into(), span, cap));
        }
    }

    pub fn stage_edge(&mut self, src: impl Into<String>, dst: impl Into<String>, span: Interval) {
        self.stage_edge_with_cap(src, dst, span, 1);
    }

    pub fn stage_edge_with_cap(
        &mut self,
        src: impl Into<String>,
        dst: impl Into<String>,
        span: Interval,
        cap: usize,
    ) {
        if self.state != TransactionState::Staging {
            return;
        }
        if !span.is_empty() {
            self.staged_edges.push((src.into(), dst.into(), span, cap));
        }
    }

    pub fn staged_count(&self) -> usize {
        self.staged_nodes.len() + self.staged_edges.len()
    }

    /// Validates all staged reservations against `table`.
    /// Returns `true` if and only if EVERY staged node and edge is collision-free.
    pub fn can_commit(&self, table: &ReservationTable) -> bool {
        if self.state != TransactionState::Staging {
            return false;
        }

        // 1. Verify all staged nodes
        for (node, span, cap) in &self.staged_nodes {
            if !table.is_node_free(node, *span, *cap, Some(&self.agv_id)) {
                return false;
            }
        }

        // 2. Verify all staged edges (including opposing edge check)
        for (src, dst, span, cap) in &self.staged_edges {
            if !table.is_edge_free(src, dst, *span, *cap, Some(&self.agv_id)) {
                return false;
            }
            if !table.is_opposing_edge_free(src, dst, *span, Some(&self.agv_id)) {
                return false;
            }
        }

        true
    }

    /// Atomically commits all staged reservations into `table`.
    /// If any check fails, rolls back completely without modifying `table`.
    pub fn commit(&mut self, table: &mut ReservationTable) -> Result<(), TransactionError> {
        if self.state == TransactionState::Committed {
            return Err(TransactionError::AlreadyCommitted);
        }
        if self.state == TransactionState::RolledBack {
            return Err(TransactionError::AlreadyRolledBack);
        }

        if !self.can_commit(table) {
            self.rollback();
            return Err(TransactionError::CapacityExceeded {
                resource: "transaction_conflict".to_string(),
                span: Interval::new_unchecked(0, 0),
            });
        }

        // Atomically apply all writes
        for (node, span, _cap) in &self.staged_nodes {
            table.reserve_node(node.clone(), *span, self.agv_id.clone());
        }
        for (src, dst, span, _cap) in &self.staged_edges {
            table.reserve_edge(src.clone(), dst.clone(), *span, self.agv_id.clone());
        }

        self.state = TransactionState::Committed;
        Ok(())
    }

    /// Aborts and discards all staged reservations.
    pub fn rollback(&mut self) {
        self.staged_nodes.clear();
        self.staged_edges.clear();
        self.state = TransactionState::RolledBack;
    }
}

/// Ledger tracking non-preemptible execution commitments.
#[derive(Debug, Clone, Default)]
pub struct ExecutionCommitmentLedger {
    /// Active firm commitment per vehicle: (src, dst, Interval)
    active_commitments: HashMap<String, (String, String, Interval)>,
    /// Completed hops
    completed_hops: HashSet<String>,
}

impl ExecutionCommitmentLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks a hop as firmly committed (vehicle is actively traversing or committed to start).
    pub fn commit_hop(&mut self, agv_id: &str, src: &str, dst: &str, span: Interval) {
        self.active_commitments.insert(
            agv_id.to_string(),
            (src.to_string(), dst.to_string(), span),
        );
    }

    /// Completes the hop, releasing edge commitment.
    pub fn complete_hop(&mut self, agv_id: &str, node: &str) {
        self.active_commitments.remove(agv_id);
        self.completed_hops.insert(format!("{}:{}", agv_id, node));
    }

    /// Checks if a vehicle has an unrevocable active commitment on edge (src, dst).
    pub fn has_active_commitment(&self, agv_id: &str) -> bool {
        self.active_commitments.contains_key(agv_id)
    }

    /// Returns the active committed edge for an AGV.
    pub fn get_commitment(&self, agv_id: &str) -> Option<&(String, String, Interval)> {
        self.active_commitments.get(agv_id)
    }

    /// Returns total active commitments count.
    pub fn active_count(&self) -> usize {
        self.active_commitments.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_transaction_success() {
        let mut tbl = ReservationTable::new();

        let mut tx = ReservationTransaction::begin("AGV_1");
        tx.stage_node("N1", Interval::new(0, 5));
        tx.stage_edge("N1", "N2", Interval::new(5, 10));
        tx.stage_node("N2", Interval::new(10, 15));

        assert_eq!(tx.staged_count(), 3);
        assert!(tx.can_commit(&tbl));
        assert!(tx.commit(&mut tbl).is_ok());

        assert_eq!(tbl.total_reservations(), 3);
        assert!(!tbl.is_node_free("N1", Interval::new(2, 4), 1, None));
    }

    #[test]
    fn test_atomic_transaction_conflict_rollback() {
        let mut tbl = ReservationTable::new();

        // Pre-reserve N2 with AGV_OBSTACLE
        tbl.reserve_node("N2", Interval::new(10, 15), "AGV_OBSTACLE");

        let mut tx = ReservationTransaction::begin("AGV_1");
        tx.stage_node("N1", Interval::new(0, 5));
        tx.stage_edge("N1", "N2", Interval::new(5, 10));
        tx.stage_node("N2", Interval::new(10, 15)); // Conflicts with AGV_OBSTACLE!

        // Transaction must fail atomically
        assert!(!tx.can_commit(&tbl));
        assert!(tx.commit(&mut tbl).is_err());

        // N1 and edge must NOT be reserved!
        assert_eq!(tbl.total_reservations(), 1); // Only the pre-existing obstacle
        assert!(tbl.is_node_free("N1", Interval::new(0, 5), 1, None));
        assert!(tbl.is_edge_free("N1", "N2", Interval::new(5, 10), 1, None));
    }
}
