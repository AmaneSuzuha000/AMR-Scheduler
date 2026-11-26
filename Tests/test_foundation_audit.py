"""Unit and invariant regression tests for foundation compatibility & Track A attribution."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.foundation_audit.execution_interface import (
    ContinuousBackendAdapter,
    DiscreteBackendAdapter,
    UnifiedRobotSpec,
    UnifiedTaskSpec,
    network_to_metric_map,
    scale_network_for_calibration,
)
from experiments.foundation_audit.f0_reservation_semantics_audit import (
    _find_workshop_corridors,
    run_executable_corridor_checks,
    scan_repo_reservation_patterns,
)
from experiments.foundation_audit.run_track_a_attribution import (
    ConfigurableDriftAuditSim,
    u0_pure_h90_plan_fn,
)
from experiments.res_guided_avoidance.gate1_headroom_audit import m2_res_margin_selector
from experiments.res_guided_avoidance.metric_world import (
    build_grid_metric_map,
    build_workshop_metric_map,
)
from experiments.stp_research_reset.audit_env import ExecutionAuditSim


def test_f0_core_free_of_arr_minus_one_and_corridor_safe() -> None:
    """Prove core `amr_sched`/`crates` never use `arr_i - 1` and block head-on multi-tick travel."""
    scan = scan_repo_reservation_patterns()
    assert scan["summary"]["expand_planned_in_core_count"] == 0
    assert scan["summary"]["arr_i_minus_1_in_core_count"] == 0

    ws = _find_workshop_corridors()
    assert ws["corridors_20_to_30m_count"] > 0
    chk = run_executable_corridor_checks(ws)
    assert chk["spacetime_astar_python"]["stagger_ok"] is True
    assert chk["rust_spacetime_astar"]["stagger_ok"] is True
    assert chk["cvcr_evaluator"]["stagger_ok"] is True
    assert chk["simulator_isolated_corridor"]["res_window_0"]["collision_events"] == 0
    assert chk["simulator_isolated_corridor"]["res_window_0"]["simultaneous_corridor_ticks"] == 0
    assert chk["simulator_isolated_corridor"]["res_window_0"]["completed_tasks"] == 2
    assert chk["simulator_isolated_corridor"]["res_window_1"]["collision_events"] == 0
    assert chk["simulator_isolated_corridor"]["res_window_1"]["simultaneous_corridor_ticks"] == 0
    assert chk["simulator_isolated_corridor"]["res_window_1"]["completed_tasks"] == 2
    assert chk["simulator_full_workshop_point_net"]["collision_events"] == 0

    bug = chk["track_a_expand_bug_repro"]
    assert bug["blind_spot_ticks"] > 0
    assert bug["buggy_r2_arr_u"] < bug["fixed_r2_arr_u"]


def test_configurable_drift_sim_matches_execution_audit_sim_bitwise() -> None:
    """Verify `ConfigurableDriftAuditSim` at `thresh=3` matches `ExecutionAuditSim` identically."""
    ma = build_grid_metric_map("Map_A", cell_m=2.6)
    base = ExecutionAuditSim(
        ma,
        upper_mode="U0_Orig_CPSA",
        sub_tick_div=1,
        custom_plan_fn=None,
        n_robots=4,
        n_tasks=6,
        horizon_s=36.0,
        seed=6001,
        n_intruders=0,
        robust_safety=True,
    )
    res_base = base.run_episode(m2_res_margin_selector, controller_name="base_u0")

    cfg = ConfigurableDriftAuditSim(
        ma,
        upper_mode="U0_Orig_CPSA",
        sub_tick_div=1,
        custom_plan_fn=None,
        drift_thresh_ticks=3,
        force_unified_macro=True,
        n_robots=4,
        n_tasks=6,
        horizon_s=36.0,
        seed=6001,
        n_intruders=0,
        robust_safety=True,
    )
    res_cfg = cfg.run_episode(m2_res_margin_selector, controller_name="unified_u0_th3")

    assert res_cfg.completed_tasks == res_base.completed_tasks
    assert res_cfg.makespan_s == res_base.makespan_s
    assert res_cfg.wait_ratio == res_base.wait_ratio
    assert res_cfg.replans == res_base.replans
    assert cfg.eta_drift_replans == base.eta_drift_replans
    assert len(cfg.closed_legs) == len(base.closed_legs)

    no_drift = ConfigurableDriftAuditSim(
        ma,
        upper_mode="U0_Pure_H90_EdgeSpanFix",
        sub_tick_div=1,
        custom_plan_fn=u0_pure_h90_plan_fn,
        drift_thresh_ticks=None,
        force_unified_macro=True,
        n_robots=4,
        n_tasks=6,
        horizon_s=36.0,
        seed=6001,
        n_intruders=0,
        robust_safety=True,
    )
    no_drift.run_episode(m2_res_margin_selector, controller_name="no_drift")
    assert no_drift.eta_drift_replans == 0


def test_f1_unified_adapters_discrete_and_continuous_parity() -> None:
    """Exercise read-only `DiscreteBackendAdapter` and `ContinuousBackendAdapter` on Workshop tasks."""
    wm = build_workshop_metric_map()
    p0 = next(p for p in wm.task_pairs if p[0].startswith("W_BRAKE"))
    p1 = next(p for p in wm.task_pairs if p[0].startswith("W_WHEELSET") and p[1] != p0[1])
    tasks = [
        UnifiedTaskSpec("T1", p0[0], p0[1], 0.0, 400.0, assigned_agv="R_00", dispatch_s=0.0, queue_order=0),
        UnifiedTaskSpec("T2", p1[0], p1[1], 0.0, 400.0, assigned_agv="R_01", dispatch_s=0.0, queue_order=0),
    ]
    robots = [
        UnifiedRobotSpec("R_00", wm.start_pool[0], max_speed=1.0, load_s=2.0, unload_s=0.0),
        UnifiedRobotSpec("R_01", wm.start_pool[1], max_speed=1.0, load_s=2.0, unload_s=0.0),
    ]

    d_res = DiscreteBackendAdapter().run(
        "unit_ws", wm.network, tasks, robots, policy="fifo+greedy", horizon_s=400.0, fixed_dispatches=True
    )
    c_res = ContinuousBackendAdapter().run(
        "unit_ws", wm, tasks, robots, policy="fifo+greedy", horizon_s=400.0, seed=1, fixed_dispatches=True
    )
    assert d_res.completed_tasks == 2
    assert d_res.zero_collisions is True
    assert len(d_res.leg_records) == 4
    assert c_res.completed_tasks == 2
    assert c_res.zero_collisions is True
    assert {lg.task_id for lg in c_res.leg_records} <= {"T1", "T2"}

    # Non-mutating calibration & metric-lifting checks
    orig_speed = wm.network.edge(p0[0], wm.network.out_edges(p0[0])[0].dst).speed_limit
    scaled_net = scale_network_for_calibration(wm.network, global_speed_scale=1.25)
    assert abs(wm.network.edge(p0[0], wm.network.out_edges(p0[0])[0].dst).speed_limit - orig_speed) < 1e-9
    lifted = network_to_metric_map("lifted_ws", scaled_net, half_width_m=1.4)
    assert len(lifted.network.nodes) == len(wm.network.nodes)
