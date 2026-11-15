use reservation_core::{Interval, ReservationTable};

#[test]
fn test_conflicts_and_exclusion() {
    let mut table = ReservationTable::new();
    table.reserve_node("N_A", Interval::new(0, 10), "AGV_1");
    table.reserve_node("N_A", Interval::new(5, 15), "AGV_2");
    table.reserve_node("N_A", Interval::new(15, 20), "AGV_3");

    // Overlapping [4, 6) -> overlaps with AGV_1 [0, 10) and AGV_2 [5, 15)
    let c1 = table.node_conflicts("N_A", Interval::new(4, 6), None);
    assert_eq!(c1, vec!["AGV_1".to_string(), "AGV_2".to_string()]);

    // Excluding AGV_1 -> only AGV_2
    let c2 = table.node_conflicts("N_A", Interval::new(4, 6), Some("AGV_1"));
    assert_eq!(c2, vec!["AGV_2".to_string()]);

    // Overlapping [14, 16) -> overlaps with AGV_2 [5, 15) and AGV_3 [15, 20) is adjacent [15, 20) -> AGV_3 [15, 20) overlaps with [14, 16)!
    let c3 = table.node_conflicts("N_A", Interval::new(14, 16), None);
    assert_eq!(c3, vec!["AGV_2".to_string(), "AGV_3".to_string()]);

    // Adjacent [10, 15) does NOT overlap with AGV_1 [0, 10)
    let c4 = table.node_conflicts("N_A", Interval::new(0, 5), Some("AGV_1"));
    assert!(c4.is_empty());
}

#[test]
fn test_edge_reservations_and_directedness() {
    let mut table = ReservationTable::new();
    table.reserve_edge("N_A", "N_B", Interval::new(10, 20), "AGV_1");

    // Directed edge check: (N_A, N_B) has reservation, (N_B, N_A) does not
    assert!(!table.is_edge_free("N_A", "N_B", Interval::new(15, 25), 1, None));
    assert!(table.is_edge_free("N_B", "N_A", Interval::new(15, 25), 1, None));

    // Opposing edge check: checking if we can traverse (N_B, N_A) when (N_A, N_B) is occupied
    assert!(!table.is_opposing_edge_free("N_B", "N_A", Interval::new(15, 25), None));
    // Opposing edge check excluding AGV_1
    assert!(table.is_opposing_edge_free("N_B", "N_A", Interval::new(15, 25), Some("AGV_1")));
}

#[test]
fn test_capacity_regimes() {
    let mut table = ReservationTable::new();
    table.reserve_node("N_HUB", Interval::new(0, 10), "AGV_1");
    table.reserve_node("N_HUB", Interval::new(0, 10), "AGV_2");

    // Capacity 2: full
    assert!(!table.is_node_free("N_HUB", Interval::new(2, 5), 2, None));
    // Capacity 3: free
    assert!(table.is_node_free("N_HUB", Interval::new(2, 5), 3, None));

    // Exclude AGV_1: effectively 1 reservation
    assert!(table.is_node_free("N_HUB", Interval::new(2, 5), 2, Some("AGV_1")));
    assert!(!table.is_node_free("N_HUB", Interval::new(2, 5), 1, Some("AGV_1")));
}

#[test]
fn test_pinned_combined_with_scheduled() {
    let mut table = ReservationTable::new();
    // AGV_1 is scheduled on N_1 at [0, 10)
    table.reserve_node("N_1", Interval::new(0, 10), "AGV_1");
    // AGV_1 is also pinned on N_1 (physical occupancy)
    table.transfer_pins("AGV_1", &["N_1".to_string()], &[]);

    // AGV_1 appears in both scheduled and pinned.
    // Capacity check should not double-count the same vehicle!
    // Total count for capacity 1 should be 1 (full), but capacity 2 should be free.
    assert!(!table.is_node_free("N_1", Interval::new(5, 15), 1, None));
    assert!(table.is_node_free("N_1", Interval::new(5, 15), 2, None));

    // AGV_2 also pinned on N_1
    let mut table2 = ReservationTable::new();
    table2.reserve_node("N_1", Interval::new(0, 10), "AGV_1");
    table2.transfer_pins("AGV_2", &["N_1".to_string()], &[]);

    // Now 1 scheduled (AGV_1) + 1 distinct pinned (AGV_2) = 2 occupants
    assert!(!table2.is_node_free("N_1", Interval::new(5, 15), 2, None));
    assert!(table2.is_node_free("N_1", Interval::new(5, 15), 3, None));
}

#[test]
fn test_snapshots_and_utilization() {
    let mut table = ReservationTable::new();
    table.reserve_node("N_1", Interval::new(0, 10), "AGV_1");
    table.reserve_node("N_2", Interval::new(10, 20), "AGV_2");
    table.reserve_edge("N_1", "N_2", Interval::new(0, 10), "AGV_1");

    let snap = table.snapshot(5, 15);
    assert!(snap.contains_key("N_1"));
    assert!(snap.contains_key("N_2"));

    let edge_snap = table.edge_snapshot(5, 15);
    // Key is sorted: "N_1|N_2"
    assert!(edge_snap.contains_key("N_1|N_2"));

    // Total reservations: 2 node + 1 edge = 3
    assert_eq!(table.total_reservations(), 3);

    // Utilization: occupied = 10 + 10 + 10 = 30 steps.
    // horizon = 20, n_nodes = 2, n_edges = 1 -> total slots = 20 * 3 = 60.
    // utilization = 30 / 60 = 0.5.
    let util = table.utilization(20, 2, 1);
    assert!((util - 0.5).abs() < 1e-6);
}
