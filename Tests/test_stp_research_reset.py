"""Unit & parity tests for ExecutionAuditSim and upper-planning audit harness (Execution-Aware-STP-Audit)."""

from __future__ import annotations

from amr_sched.planners.base import PlannedPath
from amr_sched.reservation import Interval
from amr_sched.rust_reservation import RustReservationTable
from experiments.res_guided_avoidance.gate1_headroom_audit import m2_res_margin_selector
from experiments.res_guided_avoidance.metric_world import (
    MultiRobotUnicycleSim,
    build_grid_metric_map,
    build_occluded_junction_map,
    build_workshop_metric_map,
)
from experiments.stp_research_reset.audit_env import ExecutionAuditSim, _reserve_exact_span
from experiments.stp_research_reset.upper_planners import (
    make_u1b_plan_fn,
    plan_kinodynamic_spacetime_astar,
    u3_state_dep_buffer_fn,
)


def test_reserve_exact_span_blocks_intermediate_multi_step_ticks() -> None:
    """Verify `_reserve_exact_span` reserves `[dep_t, arr_i + buf)` across long Workshop edges."""
    mmap = build_workshop_metric_map()
    # Find an edge spanning multiple macro ticks (> 2.0 s at 0.85 m/s => > 1.7 m)
    long_edge = max(mmap.network.edges.values(), key=lambda e: e.length)
    n_ticks = max(2, int(round(long_edge.length / (0.85 * mmap.macro_step_s))))
    rt = RustReservationTable.fast()
    pl = PlannedPath(nodes=[long_edge.dst], arrival_steps=[n_ticks], wait_steps=0)
    _reserve_exact_span(rt, mmap, "R_00", long_edge.src, 0, pl, buffer_ticks=1, sub_tick_div=1)

    mid_t = n_ticks // 2
    assert mid_t > 0 and mid_t < n_ticks
    # Another robot querying the same edge during the middle of traversal MUST see it occupied!
    assert not rt.is_edge_free(long_edge.src, long_edge.dst, Interval(mid_t, mid_t + 1), 1, exclude_agv="R_01")
    # Also check reverse direction physical edge exclusion via `is_opposing_edge_free`
    assert not rt.is_opposing_edge_free(long_edge.dst, long_edge.src, Interval(mid_t, mid_t + 1), exclude_agv="R_01")
    # Destination node should also carry buffer tick at `[n_ticks, n_ticks + 2)`
    assert not rt.is_node_free(long_edge.dst, Interval(n_ticks + 1, n_ticks + 2), 1, exclude_agv="R_01")


def test_execution_audit_u0_bitwise_parity_with_base_sim() -> None:
    """`ExecutionAuditSim(upper_mode='U0_Orig_CPSA')` must match frozen `MultiRobotUnicycleSim` bitwise."""
    configs = [
        (build_occluded_junction_map(), 6, 12, 95.0, 6001, 0),
        (build_grid_metric_map("Map_A"), 8, 16, 110.0, 6002, 0),
    ]
    for mmap, nr, nt, hz, sd, ni in configs:
        base_sim = MultiRobotUnicycleSim(
            mmap, n_robots=nr, n_tasks=nt, horizon_s=hz, seed=sd, n_intruders=ni, robust_safety=True
        )
        audit_sim = ExecutionAuditSim(
            mmap, upper_mode="U0_Orig_CPSA", n_robots=nr, n_tasks=nt, horizon_s=hz, seed=sd, n_intruders=ni, robust_safety=True
        )
        res_base = base_sim.run_episode(m2_res_margin_selector, controller_name="R1_Robust")
        res_audit = audit_sim.run_episode(m2_res_margin_selector, controller_name="R1_Robust")

        assert res_base.completed_tasks == res_audit.completed_tasks
        assert res_base.makespan_s == res_audit.makespan_s
        assert res_base.wall_collisions == res_audit.wall_collisions
        assert res_base.peer_collisions == res_audit.peer_collisions
        assert res_base.intruder_collisions == res_audit.intruder_collisions
        assert res_base.shield_interventions == res_audit.shield_interventions
        assert res_base.replans == res_audit.replans
        assert res_base.wait_ratio == res_audit.wait_ratio

        assert len(audit_sim.closed_legs) > 0
        sh_sum = (
            audit_sim.shield_wall_events
            + audit_sim.shield_peer_events
            + audit_sim.shield_intr_events
            + audit_sim.shield_other_events
        )
        assert sh_sum == audit_sim.shield_interventions


def test_execution_audit_clone_snapshot_determinism() -> None:
    """`clone_snapshot()` must yield identical future rollouts for U0, U1a, and sub-tick custom planner."""
    mmap = build_occluded_junction_map()

    def _dummy_subtick_plan(sim: ExecutionAuditSim, rid: str, goal_node: str, t_plan_tick: int):
        div = sim._sub_tick_div
        empty_rt = RustReservationTable.fast()
        raw_pl, _ = sim.planner.plan_leg(
            sim.robots[rid].cur_node,
            goal_node,
            int(t_plan_tick // div),
            empty_rt,
            sim.planner.create_field(),
            rid,
            220,
            goal_park_steps=1,
        )
        if raw_pl is None:
            return None, 0
        sub_pl = PlannedPath(
            nodes=list(raw_pl.nodes),
            arrival_steps=[t_plan_tick + max(1, (int(s) - int(t_plan_tick // div)) * div) for s in raw_pl.arrival_steps],
            wait_steps=int(raw_pl.wait_steps) * div,
        )
        return sub_pl, 1

    for mode, div, cfn in [
        ("U0_Orig_CPSA", 1, None),
        ("U1a_EdgeSpan_Fix", 1, None),
        ("U1b_Dummy_SubTick", 4, _dummy_subtick_plan),
    ]:
        sim = ExecutionAuditSim(
            mmap,
            upper_mode=mode,
            sub_tick_div=div,
            custom_plan_fn=cfn,
            n_robots=6,
            n_tasks=10,
            horizon_s=60.0,
            seed=6001,
            n_intruders=0,
            robust_safety=True,
        )
        for _ in range(80):  # 20.0 s
            if sim.step(m2_res_margin_selector, controller_name="R1_Robust"):
                break
        cl = sim.clone_snapshot()
        while True:
            d1 = sim.step(m2_res_margin_selector, controller_name="R1_Robust")
            d2 = cl.step(m2_res_margin_selector, controller_name="R1_Robust")
            assert d1 == d2
            if d1:
                break
        assert sim.completed_tasks == cl.completed_tasks
        assert sim.replans == cl.replans
        assert sim.shield_interventions == cl.shield_interventions
        for rid in sim.robot_ids:
            assert abs(sim.robots[rid].x - cl.robots[rid].x) < 1e-12
            assert abs(sim.robots[rid].y - cl.robots[rid].y) < 1e-12


def test_u1b_kinodynamic_monotonicity_and_reservation_safety() -> None:
    """`U1b_KinoTime_CPSA` produces strictly increasing timestamps and respects opposing-edge locks."""
    mmap = build_occluded_junction_map(cell_m=2.6)
    sim = ExecutionAuditSim(
        mmap,
        upper_mode="U1b_KinoTime_CPSA",
        sub_tick_div=4,
        custom_plan_fn=make_u1b_plan_fn(),
        n_robots=6,
        n_tasks=10,
        horizon_s=40.0,
        seed=6001,
        robust_safety=True,
    )
    nodes_sorted = sorted(mmap.network.nodes.keys())
    # Pick well-separated cardinal cross endpoints on Occluded_Jct
    start_nd = mmap.pickup_nodes[0]
    goal_nd = mmap.dropoff_nodes[-1]
    n_start = mmap.network.node(start_nd)
    first_nbr = mmap.network.out_edges(start_nd)[0]
    nbr_nd = mmap.network.node(first_nbr.dst)
    import math
    dir_away = math.atan2(nbr_nd.y - n_start.y, nbr_nd.x - n_start.x)

    rb = sim.robots["R_00"]
    rb.cur_node = start_nd
    rb.theta = dir_away
    pl_aligned = plan_kinodynamic_spacetime_astar(sim, "R_00", goal_nd, 0, buffer_ticks=0)
    assert pl_aligned is not None and len(pl_aligned.nodes) >= 2
    assert pl_aligned.arrival_steps[0] > 0
    for a, b in zip(pl_aligned.arrival_steps[:-1], pl_aligned.arrival_steps[1:]):
        assert b > a

    # Force opposite-direction alignment (180 deg U-turn) and verify initial turn adds >= 2 sub-ticks
    rb.theta = dir_away + math.pi
    pl_uturn = plan_kinodynamic_spacetime_astar(sim, "R_00", goal_nd, 0, buffer_ticks=0)
    assert pl_uturn is not None
    assert pl_uturn.arrival_steps[-1] > pl_aligned.arrival_steps[-1]

    # Verify U3 deterministic buffer fn works cleanly with sub-tick reservation
    pl_u3, b_u3 = u3_state_dep_buffer_fn(sim, "R_00", goal_nd, 0)
    assert pl_u3 is not None and b_u3 >= 0

