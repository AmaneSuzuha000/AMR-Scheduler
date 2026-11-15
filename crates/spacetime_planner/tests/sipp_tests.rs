use reservation_core::{Interval, ReservationTable};
use spacetime_planner::{PlannerGraph, SippPlanner};

#[test]
fn test_sipp_direct_path() {
    let mut g = PlannerGraph::new();
    g.add_edge("N0", "N1", 2, 1, 1);
    g.add_edge("N1", "N2", 3, 1, 1);

    let tbl = ReservationTable::new();
    let planner = SippPlanner::new(10_000);

    let path = planner.plan(&g, &tbl, "N0", "N2", 0, "AGV_1", 20);
    assert!(path.is_some());
    let res = path.unwrap();
    assert_eq!(res.nodes, vec!["N1", "N2"]);
    assert_eq!(res.arrival_steps, vec![2, 5]);
    assert_eq!(res.wait_steps, 0);
}

#[test]
fn test_sipp_jump_wait_obstacle() {
    let mut g = PlannerGraph::new();
    g.add_edge("N0", "N1", 2, 1, 1);
    g.add_edge("N1", "N2", 2, 1, 1);

    let mut tbl = ReservationTable::new();
    // Block N1 from step 2 to 10
    tbl.reserve_node("N1", Interval::new(2, 10), "AGV_OBSTACLE");

    let planner = SippPlanner::new(10_000);

    // AGV_1 arrives at N0 at t=0.
    // N1 safe intervals are [0, 2) and [10, 30).
    // Direct arrival at N1 would be t=2, which is not in [10, 30).
    // Earliest departure from N0 to arrive at N1 in [10, 30) is dep = 8 -> arr = 10!
    // Arrival at N2 is t = 10 + 2 = 12.
    let path = planner.plan(&g, &tbl, "N0", "N2", 0, "AGV_1", 30);
    assert!(path.is_some());
    let res = path.unwrap();
    assert_eq!(res.nodes, vec!["N1", "N2"]);
    assert_eq!(res.arrival_steps, vec![10, 12]);
    assert_eq!(res.wait_steps, 8);
}
