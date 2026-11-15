use reservation_core::{Interval, ReservationTable};
use spacetime_planner::{PlannerGraph, SpacetimeAStarPlanner};

#[test]
fn test_direct_path() {
    let mut g = PlannerGraph::new();
    // N0 -> N1 -> N2
    g.add_edge("N0", "N1", 2, 1, 1);
    g.add_edge("N1", "N2", 3, 1, 1);

    let tbl = ReservationTable::new();
    let planner = SpacetimeAStarPlanner::new(10_000);

    let path = planner.plan(&g, &tbl, "N0", "N2", 0, "AGV_1", 20);
    assert!(path.is_some());
    let res = path.unwrap();
    assert_eq!(res.nodes, vec!["N1", "N2"]);
    assert_eq!(res.arrival_steps, vec![2, 5]);
    assert_eq!(res.wait_steps, 0);
}

#[test]
fn test_forced_wait_due_to_node_reservation() {
    let mut g = PlannerGraph::new();
    // N0 -> N1 -> N2
    g.add_edge("N0", "N1", 2, 1, 1);
    g.add_edge("N1", "N2", 2, 1, 1);

    let mut tbl = ReservationTable::new();
    // Block N1 from step 2 to 5 with another vehicle
    tbl.reserve_node("N1", Interval::new(2, 5), "AGV_OBSTACLE");

    let planner = SpacetimeAStarPlanner::new(10_000);

    // AGV_1 wants to go from N0 to N2 starting at t=0
    // Arrival at N1 would normally be t=2, but N1 is blocked at [2, 5)
    // AGV_1 must wait at N0 until t=3, then move to N1 (arrive at t=5) and N2 (arrive at t=7)
    let path = planner.plan(&g, &tbl, "N0", "N2", 0, "AGV_1", 20);
    assert!(path.is_some());
    let res = path.unwrap();
    assert_eq!(res.nodes, vec!["N1", "N2"]);
    assert_eq!(res.arrival_steps, vec![5, 7]);
    assert_eq!(res.wait_steps, 3);
}

#[test]
fn test_bidirectional_opposing_edge_mutual_exclusion() {
    let mut g = PlannerGraph::new();
    // Single lane bidirectional corridor: N0 <-> N1
    g.add_edge("N0", "N1", 5, 1, 1);
    g.add_edge("N1", "N0", 5, 1, 1);

    let mut tbl = ReservationTable::new();
    // AGV_2 reserved N1 -> N0 from t=0 to t=5
    // physical edge (N0, N1)
    tbl.reserve_edge("N0", "N1", Interval::new(0, 5), "AGV_2");

    let planner = SpacetimeAStarPlanner::new(10_000);

    // AGV_1 wants to go N0 -> N1 at t=0.
    // Opposing traffic occupies [0, 5). AGV_1 must wait until t=5, then arrive at t=10.
    let path = planner.plan(&g, &tbl, "N0", "N1", 0, "AGV_1", 20);
    assert!(path.is_some());
    let res = path.unwrap();
    assert_eq!(res.nodes, vec!["N1"]);
    assert_eq!(res.arrival_steps, vec![10]);
    assert_eq!(res.wait_steps, 5);
}
