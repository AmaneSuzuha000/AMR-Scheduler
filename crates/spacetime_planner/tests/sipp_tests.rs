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

#[test]
fn test_sipp_ltm_cutoff_and_agv_guide() {
    let mut g = PlannerGraph::new();
    g.add_edge("N0", "N1", 4, 1, 1);
    g.add_edge("N1", "N3", 4, 1, 1);
    g.add_edge("N0", "N2", 5, 1, 1);
    g.add_edge("N2", "N3", 5, 1, 1);

    let tbl = ReservationTable::new();
    let planner = SippPlanner::new(10_000);

    g.set_traffic_batch_script("N0\tN1\t24\n");
    g.set_traffic_cutoff(10);
    assert_eq!(planner.plan(&g, &tbl, "N0", "N3", 0, "AGV_1", 30).unwrap().nodes, vec!["N2", "N3"]);
    assert_eq!(planner.plan(&g, &tbl, "N0", "N3", 10, "AGV_1", 40).unwrap().nodes, vec!["N1", "N3"]);
    g.clear_traffic_penalties();

    g.set_agv_guide_script("AGV_2\t24\tN0\tN2\nAGV_2\t24\tN2\tN3\n");
    assert_eq!(planner.plan(&g, &tbl, "N0", "N3", 0, "AGV_1", 30).unwrap().nodes, vec!["N1", "N3"]);
    assert_eq!(planner.plan(&g, &tbl, "N0", "N3", 0, "AGV_2", 30).unwrap().nodes, vec!["N2", "N3"]);
}

#[test]
fn test_sipp_and_astar_indexed_parity() {
    use spacetime_planner::SpacetimeAStarPlanner;
    let mut g = PlannerGraph::new();
    g.add_edge("N0", "N1", 3, 1, 1);
    g.add_edge("N1", "N2", 3, 1, 1);
    g.add_edge("N0", "N3", 4, 1, 1);
    g.add_edge("N3", "N2", 4, 1, 1);

    let mut tbl = ReservationTable::new();
    tbl.reserve_node("N1", Interval::new(3, 8), "BLK");

    let sipp = SippPlanner::new(10_000);
    let astar = SpacetimeAStarPlanner::new(10_000);
    let s0 = g.node_index("N0").unwrap();
    let g0 = g.node_index("N2").unwrap();

    let p_sipp_str = sipp.plan(&g, &tbl, "N0", "N2", 0, "A1", 30).unwrap();
    let p_sipp_idx = sipp.plan_indexed_idx(&g, &tbl, s0, g0, 0, "A1", 30).unwrap().to_planned_path(&g);
    assert_eq!(p_sipp_str, p_sipp_idx);

    let p_astar_str = astar.plan(&g, &tbl, "N0", "N2", 0, "A1", 30).unwrap();
    let p_astar_idx = astar.plan_indexed_idx(&g, &tbl, s0, g0, 0, "A1", 30).unwrap().to_planned_path(&g);
    assert_eq!(p_astar_str, p_astar_idx);
}
