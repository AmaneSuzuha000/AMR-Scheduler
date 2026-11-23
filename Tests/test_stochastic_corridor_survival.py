"""Unit & regression tests for P4.5 Stochastic L2 Execution-Aware Routing & Choke Arbitration."""

from __future__ import annotations

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.execution_feedback import AmrExecutionStatus, ExecutionFeedbackPacket
from amr_sched.cvcr.p4_belief_choke import (
    CorridorDurationBeliefModel,
    predict_and_arbitrate_choke_conflicts,
)
from amr_sched.cvcr.production import OperationSnapshot
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.entities import Agv, Task, TaskStatus
from amr_sched.network import Edge, Network, Node, NodeKind
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.scenarios_v10 import build_unseen_choke_yard_network
from amr_sched.simulation import Hop, SimulationConfig, Simulator, _Job


def _make_chain_network() -> Network:
    """Build small linear network with two pickup/dropoff legs to test drifted multi-leg buffering."""
    net = Network()
    net.add_node(Node("N0", 0.0, 0.0, kind=NodeKind.BUFFER, capacity=4))
    net.add_node(Node("N1", 4.0, 0.0, kind=NodeKind.WORKSTATION, capacity=2))
    net.add_node(Node("N2", 8.0, 0.0, kind=NodeKind.WORKSTATION, capacity=2))
    for u, v in (("N0", "N1"), ("N1", "N2")):
        net.add_edge(Edge(u, v, length=4.0, speed_limit=1.0, capacity=2))
        net.add_edge(Edge(v, u, length=4.0, speed_limit=1.0, capacity=2))
    return net


def test_p45_nominal_zero_diff_parity() -> None:
    """Nominal (None or empty feedback) triggers zero leg buffer and zero proactive choke overrides."""
    pkt_empty = ExecutionFeedbackPacket(tick=0)
    assert pkt_empty.robot_leg_step_buffer("agv_0", 6) == 0

    pkt_nom = ExecutionFeedbackPacket(
        tick=5,
        robot_states={
            "agv_0": AmrExecutionStatus(robot_id="agv_0", speed_scale=0.95, eta_drift_steps=0, local_planner_state="nominal")
        },
    )
    assert pkt_nom.robot_leg_step_buffer("agv_0", 6) == 0

    net = build_unseen_choke_yard_network(aisle_capacity=1, speed=3.0)
    sim = Simulator(
        net,
        [Task("T1", "WA_S0", "W4", release_step=0, due_step=200)],
        [Agv("agv_0", max_speed=3.0, start_node="WA_S0")],
        SpaceTimeAStarPlanner(max_speed=3.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=100),
    )
    sim.reset()
    assert predict_and_arbitrate_choke_conflicts(sim, None, ()) == []
    assert predict_and_arbitrate_choke_conflicts(sim, pkt_empty, ()) == []


def test_p45_continuous_cvar75_wait_vs_detour_pricing() -> None:
    """Verify continuous CVaR75 pricing adapts to short/long/late regimes without rigid lam_post walls."""
    # Short transient regime (d=5 clears): exponential forgetting lowers scale -> waits instead of big 14-step detour
    bm_short = CorridorDurationBeliefModel()
    bm_short.observe_step(10, {("A", "B")})
    bm_short.observe_step(15, set())
    bm_short.observe_step(30, {("A", "B")})
    est_short = bm_short.estimate_remaining(("A", "B"), 31, detour_overhead_steps=14.0)
    assert est_short.prefer_wait_over_detour is True
    assert 0 < est_short.recommended_q3_weight < 15

    # Long persistent regime (d=24 clears) with moderate detour -> prefers detour with Q3=32 and long horizon
    bm_long = CorridorDurationBeliefModel()
    bm_long.observe_step(10, {("A", "B")})
    bm_long.observe_step(34, set())
    bm_long.observe_step(50, {("A", "B")})
    est_long = bm_long.estimate_remaining(("A", "B"), 50, detour_overhead_steps=8.0)
    assert est_long.prefer_wait_over_detour is False
    assert est_long.recommended_q3_weight == 32
    assert est_long.barrier_horizon_steps >= 14

    # Medium obstacle late into block: early step detours, near clearance transitions smoothly to wait
    bm_med = CorridorDurationBeliefModel()
    bm_med.observe_step(10, {("A", "B")})
    est_early = bm_med.estimate_remaining(("A", "B"), 10, detour_overhead_steps=12.0)
    est_late = bm_med.estimate_remaining(("A", "B"), 22, detour_overhead_steps=12.0)
    assert est_early.prefer_wait_over_detour is False
    assert est_late.prefer_wait_over_detour is True
    assert est_late.recommended_q3_weight < 15


def test_p45_proactive_choke_arbitration_by_op_slack() -> None:
    """Opposite-direction drifted approaches onto cap=1 choke yield the higher-slack peer BEFORE collision."""
    net = build_unseen_choke_yard_network(aisle_capacity=1, speed=3.0)
    tasks = [
        Task("T_URG", "WA_S2", "W4", release_step=0, due_step=120),
        Task("T_LAX", "EA_S0", "W3", release_step=0, due_step=400),
    ]
    tasks[0].kit_op_id = "OP_URG"
    tasks[1].kit_op_id = "OP_LAX"
    agvs = [
        Agv("agv_urg", max_speed=3.0, start_node="WA_S0"),
        Agv("agv_lax", max_speed=3.0, start_node="EA_S0"),
    ]
    sim = Simulator(
        net,
        tasks,
        agvs,
        SpaceTimeAStarPlanner(max_speed=3.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=300),
    )
    sim.reset()
    tasks[0].status = TaskStatus.PICKED
    tasks[0].assigned_agv = "agv_urg"
    tasks[1].status = TaskStatus.PICKED
    tasks[1].assigned_agv = "agv_lax"
    sim.agvs[0].cur_node = "WA_S0"
    sim.agvs[0].cur_task = "T_URG"
    sim._jobs["agv_urg"] = _Job(
        task_id="T_URG",
        leg="to_dropoff",
        hops=[Hop("WA_S0", "CH_W", 10, 12), Hop("CH_W", "CH_M", 12, 15)],
        idx=0,
        goal="W4",
    )
    sim.agvs[1].cur_node = "EA_S0"
    sim.agvs[1].cur_task = "T_LAX"
    sim._jobs["agv_lax"] = _Job(
        task_id="T_LAX",
        leg="to_dropoff",
        hops=[Hop("EA_S0", "CH_E", 10, 12), Hop("CH_E", "CH_M", 12, 15)],
        idx=0,
        goal="W3",
    )
    sim._step_cursor = 10

    ops = [
        OperationSnapshot("OP_URG", ("T_URG",), "W4", 0, 0, 10, 80, ()),
        OperationSnapshot("OP_LAX", ("T_LAX",), "W3", 1, 1, 10, 350, ()),
    ]
    fb = ExecutionFeedbackPacket(
        tick=10,
        robot_states={
            "agv_urg": AmrExecutionStatus(robot_id="agv_urg", speed_scale=0.65, eta_drift_steps=2, local_planner_state="slowed"),
            "agv_lax": AmrExecutionStatus(robot_id="agv_lax", speed_scale=1.0, eta_drift_steps=0, local_planner_state="nominal"),
        },
    )
    yielders = predict_and_arbitrate_choke_conflicts(sim, fb, ops, lookahead=14)
    assert yielders == ["agv_lax"]


def test_p45_evaluator_downstream_leg_buffer_propagation() -> None:
    """Degraded robot speed propagates causal risk buffer to multi-task evaluation while nominal is unchanged."""
    net = _make_chain_network()
    tasks = [
        Task("T1", "N0", "N1", release_step=0, due_step=40),
        Task("T2", "N1", "N2", release_step=0, due_step=60),
    ]
    agvs = [Agv("R0", max_speed=1.0, start_node="N0")]
    sim = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0), GreedyAllocator(), SimulationConfig(dt=1.0, max_steps=120))
    sim.reset()
    ops = [OperationSnapshot("OP1", ("T1", "T2"), "N2", 0, 0, 5, 100, ())]

    snap_nom = NativeSnapshotCapture.capture(sim, ops, hide_future_releases=False)
    ev_nom = CVCREvaluator(net, snap_nom, mode="B2", max_speed=1.0, use_rust=True)
    r_nom = ev_nom.evaluate({"T1": "R0", "T2": "R0"})
    assert r_nom.valid

    fb_slow = ExecutionFeedbackPacket(
        tick=0,
        robot_states={"R0": AmrExecutionStatus(robot_id="R0", speed_scale=0.55, eta_drift_steps=2, local_planner_state="slowed")},
    )
    assert fb_slow.robot_leg_step_buffer("R0", 4) > 0
    snap_slow = NativeSnapshotCapture.capture(sim, ops, hide_future_releases=False, execution_feedback=fb_slow)
    ev_slow = CVCREvaluator(net, snap_slow, mode="B2", max_speed=1.0, use_rust=True)
    r_slow = ev_slow.evaluate({"T1": "R0", "T2": "R0"})
    assert r_slow.valid
    assert r_slow.order_unload_ticks["T2"] > r_nom.order_unload_ticks["T2"]


def test_p45_voi_probe_wait_observe_vs_detour() -> None:
    """Verify causal VoI probe holds wait-and-observe early before expiring to detour cleanly."""
    # Baseline without VoI probe detours immediately when ambiguous prior cost < detour overhead
    bm_default = CorridorDurationBeliefModel(enable_voi_probe=False)
    bm_default.observe_step(10, {("A", "B")})
    est_default = bm_default.estimate_remaining(("A", "B"), 12, detour_overhead_steps=8.0)
    assert est_default.prefer_wait_over_detour is False
    assert est_default.recommended_q3_weight == 32

    # With VoI probe enabled: tau_probe = min(8, max(3, round(0.75 * 8))) = 6
    bm_voi = CorridorDurationBeliefModel(enable_voi_probe=True)
    bm_voi.observe_step(10, {("A", "B")})
    est_early = bm_voi.estimate_remaining(("A", "B"), 12, detour_overhead_steps=8.0)
    assert est_early.prefer_wait_over_detour is True
    assert est_early.recommended_q3_weight == 0
    assert est_early.barrier_horizon_steps == 4

    # Once elapsed >= tau_probe (=6), switch cleanly to detour under ambiguous moderate detour cost
    est_expired = bm_voi.estimate_remaining(("A", "B"), 16, detour_overhead_steps=8.0)
    assert est_expired.prefer_wait_over_detour is False
    assert est_expired.recommended_q3_weight == 32

    # Chronic empirical prior (completed duration = 32 steps) disables VoI hold and detours at once
    bm_chronic = CorridorDurationBeliefModel(enable_voi_probe=True)
    bm_chronic.observe_step(10, {("A", "B")})
    bm_chronic.observe_step(42, set())
    bm_chronic.observe_step(50, {("A", "B")})
    est_chr = bm_chronic.estimate_remaining(("A", "B"), 51, detour_overhead_steps=8.0)
    assert est_chr.prefer_wait_over_detour is False
    assert est_chr.recommended_q3_weight == 32
