use reservation_core::{Interval, ReservationTable};
use spacetime_planner::{
    CpsSearchParams, PlannerGraph, SippPlanner, SlotCongestionField, SlotFieldConfig,
    SpacetimeAStarPlanner,
};

#[test]
fn test_cps_slot_field_and_turn_penalty() {
    let mut g = PlannerGraph::new();
    g.set_dt_s(1.0);
    // Corridor with 90-degree turn at N1 and straight segment N0 -> N1 -> N2 vs N1 -> N3 -> N4
    let n0 = g.set_node_capacity("N0", 1);
    let n1 = g.set_node_capacity("N1", 1);
    let n2 = g.set_node_capacity("N2", 1);
    let n3 = g.set_node_capacity("N3", 1);
    let n4 = g.set_node_capacity("N4", 1);
    g.set_node_coords_indexed(n0, 0.0, 0.0);
    g.set_node_coords_indexed(n1, 1.0, 0.0);
    g.set_node_coords_indexed(n2, 1.0, 1.0); // 90 deg turn
    g.set_node_coords_indexed(n3, 2.0, 0.0); // straight continuation
    g.set_node_coords_indexed(n4, 2.0, 1.0);

    // Build diamond: N0->N1 (1), N1->N2 (1), N2->N4 (1), N1->N3 (1), N3->N4 (1)
    g.add_edge_with_time("N0", "N1", 1, 1.0, 1, 1);
    g.add_edge_with_time("N1", "N2", 1, 1.0, 1, 1);
    g.add_edge_with_time("N2", "N4", 1, 1.0, 1, 1);
    g.add_edge_with_time("N1", "N3", 1, 1.0, 1, 1);
    g.add_edge_with_time("N3", "N4", 1, 1.0, 1, 1);

    let mut field = SlotCongestionField::new(&g, SlotFieldConfig::default());
    // Reserve heavy dwell congestion along N2 to bias against waiting / congestion
    field.add_agv_transitions("BG1", &[n2, n2, n2], &[1, 2, 3], Some(20));
    assert!(field.slot_phi_cached(n2, 2) > 0.0);
    let snap = field.clone();
    field.remove_agv("BG1");
    assert_eq!(field.slot_phi_cached(n2, 2), 0.0);
    field.restore_from(&snap);
    assert!(field.slot_phi_cached(n2, 2) > 0.0);

    let tbl = ReservationTable::new();
    let planner = SpacetimeAStarPlanner::new(10_000);
    let params = CpsSearchParams {
        mode_oneshot: false,
        alpha: 1.0,
        beta: 0.5,
        gamma: 5.0,
        lookahead_omega: 0.0,
        goal_park_until: None,
        goal_park_steps: 1,
        prev_arrival_node: u16::MAX,
        max_expansions_override: 5_000,
        collect_blockers: true,
    };
    let res = planner.plan_cps_indexed(&g, &tbl, Some(&mut field), n0, n4, 0, "AGV_1", 20, params);
    let p = res.path.unwrap();
    assert_eq!(p.nodes, vec![n1, n3, n4]);
    assert!(res.expansions > 0);
}

#[test]
fn test_cps_goal_park_and_blocker_hits() {
    let mut g = PlannerGraph::new();
    let n0 = g.set_node_capacity("N0", 1);
    let n1 = g.set_node_capacity("N1", 1);
    g.add_edge_with_time("N0", "N1", 1, 1.0, 1, 1);
    g.add_edge_with_time("N1", "N0", 1, 1.0, 1, 1);

    let mut tbl = ReservationTable::new();
    // Blocker AGV_B occupies N1 on [1, 3): arrival at t=1 is blocked, arrival at t=3 passes
    // Also check goal_park_steps=1 when N1 occupied at [2, 4): AGV_1 can only park [t+1, t+2) once free!
    tbl.reserve_node("N1", Interval::new(1, 4), "AGV_B");

    let mut field = SlotCongestionField::new(&g, SlotFieldConfig::default());
    let astar = SpacetimeAStarPlanner::new(5_000);
    let params = CpsSearchParams {
        goal_park_steps: 1,
        collect_blockers: true,
        ..CpsSearchParams::default()
    };
    let res = astar.plan_cps_indexed(&g, &tbl, Some(&mut field), n0, n1, 0, "AGV_A", 12, params);
    let p = res.path.unwrap();
    assert_eq!(p.nodes, vec![n1]);
    assert_eq!(p.arrivals, vec![4]);
    assert_eq!(p.wait_steps, 3);
    assert!(res.blocker_hits.iter().any(|(aid, cnt)| aid == "AGV_B" && *cnt >= 1));

    // Verify SippPlanner plan_indexed_ex matches goal_park_steps=1 behavior and returns failure blockers cleanly
    let sipp = SippPlanner::new(5_000);
    let s_res = sipp.plan_indexed_ex(&g, &tbl, Some(&field), n0, n1, 0, "AGV_A", 12, params);
    let sp = s_res.path.unwrap();
    assert_eq!(sp.nodes, vec![n1]);
    assert_eq!(sp.arrivals, vec![4]);

    // Horizon too short for goal_park_steps: returns None with AGV_B blocker attribution
    let fail_res = sipp.plan_indexed_ex(&g, &tbl, Some(&field), n0, n1, 0, "AGV_A", 4, params);
    assert!(fail_res.path.is_none());
    assert!(fail_res.blocker_hits.iter().any(|(aid, cnt)| aid == "AGV_B" && *cnt >= 1));
}
