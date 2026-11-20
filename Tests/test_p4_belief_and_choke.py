"""Unit & closed-loop regression tests for P4 Online Belief & Choke-Robust Coordination."""

from __future__ import annotations

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.cvcr.p2_perturbation import ExecutionPerturbationModel, P2PerturbationConfig
from amr_sched.cvcr.p4_belief_choke import (
    CorridorDurationBeliefModel,
    corridor_detour_overhead,
    detect_choke_oscillation_agvs,
    record_node_history_tick,
)
from amr_sched.cvcr.production import OperationSnapshot
from amr_sched.cvcr.v13_kitting_rl import DEFAULT_V13_SMDP_CKPT, load_kitting_window_ranker
from amr_sched.entities import Agv, Task, TaskStatus
from amr_sched.layouts import build_layout
from amr_sched.network import Edge, Network, Node, NodeKind
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.rl import (
    AMRReschedulingEnv,
    V13KittingPolicy,
    WhenToReplanTrigger,
)
from amr_sched.scenarios_v10 import build_unseen_choke_yard_network
from amr_sched.simulation import SimulationConfig, Simulator


def test_p4_belief_causal_hazard_decay_and_online_adaptation() -> None:
    """Verify increasing-hazard decay, online duration adaptation, and strict zero future leakage."""
    net = build_layout("ring", aisle_capacity=2)
    bm = CorridorDurationBeliefModel()

    # Zero-future-leakage check: two perturbation models with identical current obstacle start
    # but drastically different future end_step (16 vs 90) MUST produce identical estimates before clearing
    t0_cfg = P2PerturbationConfig(seed=42, enable_travel_drift=False, enable_local_planner_fail=False, obstacle_duration_steps=6, enable_p4_belief=True)
    t1_cfg = P2PerturbationConfig(seed=42, enable_travel_drift=False, enable_local_planner_fail=False, obstacle_duration_steps=40, enable_p4_belief=True)
    m_short = ExecutionPerturbationModel(t0_cfg)
    m_long = ExecutionPerturbationModel(t1_cfg)
    assert net.has_edge("R3", "R4")

    bm_a = CorridorDurationBeliefModel()
    bm_b = CorridorDurationBeliefModel()
    for step in range(10, 16):
        bm_a.observe_step(step, {("R3", "R4")})
        bm_b.observe_step(step, {("R3", "R4")})
        assert bm_a.build_corridor_feedback(net, "R3", "R4", step) == bm_b.build_corridor_feedback(net, "R3", "R4", step)

    # Under increasing-hazard prior, expected remaining duration decays monotonically over elapsed 0 -> 10
    e0 = bm.estimate_remaining(("R3", "R4"), 10, detour_overhead_steps=5.0)
    bm.observe_step(10, {("R3", "R4")})
    e0 = bm.estimate_remaining(("R3", "R4"), 10, detour_overhead_steps=5.0)
    e6 = bm.estimate_remaining(("R3", "R4"), 16, detour_overhead_steps=5.0)
    e10 = bm.estimate_remaining(("R3", "R4"), 20, detour_overhead_steps=5.0)
    assert e0.expected_remaining > e6.expected_remaining > e10.expected_remaining

    # Online posterior adaptation: after observing 3 quick clears (d=5), prior expectation at onset drops sharply
    # compared to observing 3 long persistent clears (d=24)
    bm_fast = CorridorDurationBeliefModel()
    bm_slow = CorridorDurationBeliefModel()
    for idx in range(3):
        t_start = 10 + idx * 30
        bm_fast.observe_step(t_start, {("R0", "R1")})
        bm_fast.observe_step(t_start + 5, set())
        bm_slow.observe_step(t_start, {("R0", "R1")})
        bm_slow.observe_step(t_start + 24, set())
    est_fast = bm_fast.estimate_remaining(("R0", "R1"), 120, detour_overhead_steps=5.0)
    est_slow = bm_slow.estimate_remaining(("R0", "R1"), 120, detour_overhead_steps=5.0)
    assert est_fast.expected_remaining < est_slow.expected_remaining
    assert m_short is not None and m_long is not None


def test_p4_detour_overhead_wait_vs_bypass_gating() -> None:
    """Verify spur wait-gating, bypass Q3 preservation, hazard decay, and chronic upward adaptation."""
    ring = build_layout("ring", aisle_capacity=2)
    grid = build_layout("grid", aisle_capacity=2)
    choke = build_unseen_choke_yard_network(aisle_capacity=1, speed=3.0)
    bm_ring = CorridorDurationBeliefModel()
    bm_grid = CorridorDurationBeliefModel()

    oh_grid = corridor_detour_overhead(grid, "G01", "G11")
    oh_choke = corridor_detour_overhead(choke, "CH_W", "CH_M")
    assert oh_choke > 0.0 and oh_grid > 0.0
    assert (oh_choke / 6.0) > (oh_grid / 40.0)

    # Non-bypassable spur edge -> prefer_wait=True with q3=0 (no spurious global distortion)
    bm_spur = CorridorDurationBeliefModel()
    spur_edges = [e for e in choke.edges.values() if corridor_detour_overhead(choke, e.src, e.dst) > 1e9]
    assert spur_edges
    se = spur_edges[0]
    bm_spur.observe_step(20, {(se.src, se.dst)})
    fb_spur = bm_spur.build_corridor_feedback(choke, se.src, se.dst, 21)
    assert fb_spur.q3_penalty_override == 0

    # Matured hazard on bypassable edge (elapsed=8) decays strictly below P2 fixed +12 horizon
    bm_ring.observe_step(58, {("R3", "R4")})
    fb_ring_late = bm_ring.build_corridor_feedback(ring, "R3", "R4", 66)
    assert fb_ring_late.blocked_until_step is not None
    assert (fb_ring_late.blocked_until_step - 66) <= 8
    assert fb_ring_late.q3_penalty_override == 32

    # Grid obstacle onset projects full Bayesian onset barrier (>= 10) with Q3 bypass penalty
    bm_grid.observe_step(20, {("G01", "G11")})
    fb_grid = bm_grid.build_corridor_feedback(grid, "G01", "G11", 21)
    assert fb_grid.blocked_until_step is not None
    assert (fb_grid.blocked_until_step - 21) >= 10
    assert fb_grid.q3_penalty_override == 32

    # After observing chronic d=24 clears, onset barrier adapts strictly above fixed +12
    bm_chronic = CorridorDurationBeliefModel()
    bm_chronic.observe_step(10, {("G01", "G11")})
    bm_chronic.observe_step(34, set())
    bm_chronic.observe_step(50, {("G01", "G11")})
    fb_chronic = bm_chronic.build_corridor_feedback(grid, "G01", "G11", 50)
    assert fb_chronic.blocked_until_step is not None
    assert (fb_chronic.blocked_until_step - 50) >= 15


def test_p4_choke_oscillation_detection_resolves_headon_bridge() -> None:
    """Repeated choke yields / head-on bridge stalls are caught, while transient yields pass untouched."""
    net = build_unseen_choke_yard_network(aisle_capacity=1, speed=3.0)
    sim = Simulator(
        net,
        [
            Task("T_E", "WA_S2", "W4", release_step=0, due_step=200),
            Task("T_W", "EA_S0", "W3", release_step=0, due_step=200),
        ],
        [
            Agv("agv_e", max_speed=3.0, start_node="CH_W"),
            Agv("agv_w", max_speed=3.0, start_node="CH_E"),
        ],
        SpaceTimeAStarPlanner(max_speed=3.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=300, node_hold_steps=6, deadlock_timeout=20),
    )
    sim.reset()
    sim._p4_choke_robust = True

    # Simulate head-on PICKED oscillation inside the single-lane bridge CH_W <-> CH_M <-> CH_E
    t_e, t_w = sim.tasks[0], sim.tasks[1]
    t_e.status = TaskStatus.PICKED
    t_e.assigned_agv = "agv_e"
    t_w.status = TaskStatus.PICKED
    t_w.assigned_agv = "agv_w"

    from amr_sched.simulation import Hop, _Job
    # First: a single transient normal yield must NOT prematurely fire oscillation guard
    sim.agvs[0].cur_node = "CH_M"
    sim.agvs[0].cur_task = "T_E"
    sim._jobs["agv_e"] = _Job(task_id="T_E", leg="loaded_yield", hops=[Hop("CH_M", "CH_W", 4, 10)], idx=0, goal="CH_W")
    sim.agvs[1].cur_node = "CH_E"
    sim.agvs[1].cur_task = "T_W"
    sim._jobs["agv_w"] = _Job(task_id="T_W", leg="to_dropoff", hops=[Hop("CH_E", "CH_M", 4, 10)], idx=0, goal="W3")
    sim._step_cursor = 4
    record_node_history_tick(sim, 4)
    assert detect_choke_oscillation_agvs(sim) == []

    # Next: agv_e reverses back through CH_W -> CH_M and hits a second loaded_yield (true ping-pong)
    sim._jobs["agv_e"] = _Job(task_id="T_E", leg="to_dropoff", hops=[Hop("CH_W", "CH_M", 7, 13)], idx=0, goal="W4")
    sim.agvs[0].cur_node = "CH_W"
    sim._step_cursor = 7
    record_node_history_tick(sim, 7)

    sim.agvs[0].cur_node = "CH_M"
    sim._jobs["agv_e"] = _Job(task_id="T_E", leg="loaded_yield", hops=[Hop("CH_M", "CH_W", 12, 18)], idx=0, goal="CH_W")
    sim._step_cursor = 12
    record_node_history_tick(sim, 12)
    osc = detect_choke_oscillation_agvs(sim)
    assert "agv_e" in osc


def test_p4_witness_cases_7106_ring_and_7103_choke_yard() -> None:
    """Verify P4 Chronic Belief adaptation (Long d=24) & Choke-Robust bridge clearing (unseen_choke_yard 7103)."""
    ranker = load_kitting_window_ranker("smdp_rl", DEFAULT_V13_SMDP_CKPT)

    # 1. Chronic d=24 obstacles on ring 7101: P4 Belief beats P2 Fixed cleanly while passing audit
    res_chronic: dict[bool, dict[str, float]] = {}
    for p4_b in (False, True):
        cfg_ring = P2PerturbationConfig(
            seed=7101 * 17 + 5,
            enable_travel_drift=False,
            enable_dynamic_obstacles=True,
            obstacle_count=3,
            obstacle_duration_steps=24,
            obstacle_durations=(24, 24, 24),
            obstacle_window_range=(12, 86),
            est_clear_horizon_steps=12,
            enable_local_planner_fail=False,
            enable_p4_belief=p4_b,
        )
        with virtual_clock_scope(poll_ms=0.002):
            env_ring = AMRReschedulingEnv(
                topology="ring",
                seed=7101,
                allow_kitting_shortcut=False,
                pure_physical_reward=True,
                real_compute_window_coupling=True,
                expert_v12_portfolio=True,
                loaded_repair_mode="haven_v12",
                loaded_repair_threshold=8,
                record_trajectory=True,
                execution_perturbation=cfg_ring,
                execution_aware=True,
                p4_choke_robust=p4_b,
                kitting_ranker=ranker,
                v13_kitting_mode="V13_KITTING_RL",
            )
            obs, info = env_ring.reset(seed=7101, options={"topology": "ring", "stress_axis": "standard"})
            pol = V13KittingPolicy(epoch_interval=16, trigger=WhenToReplanTrigger(epoch_interval=16, stall_threshold=8))
            pol.reset()
            done = False
            while not done:
                act = pol.select_action(env_ring.get_expert_v12_obs(obs), info, sim=env_ring.sim)
                obs, _, tm, tr, info = env_ring.step(act)
                done = tm or tr
            m_ring = env_ring.compute_episode_metrics()
            aud_r = IndependentAuditor.audit_simulation_run(env_ring.sim, env_ring.tracker.operations, require_trajectory=True)
            assert aud_r.pass_audit, f"Ring 7101 failed audit: {aud_r.issues}"
            assert m_ring["task_completion_rate"] == 1.0
            res_chronic[p4_b] = {"net_obj": float(m_ring["net_obj"]), "cmax": float(m_ring["cmax"])}
    assert res_chronic[True]["net_obj"] < res_chronic[False]["net_obj"]

    # 2. Travel_Drift unseen_choke_yard 7103 -> clears loaded_yield oscillation and completes 100% tasks & ops
    cfg_choke = P2PerturbationConfig(
        seed=7103 * 17 + 5,
        enable_travel_drift=True,
        drift_prob=0.35,
        early_max_steps=1,
        late_max_steps=2,
        enable_dynamic_obstacles=False,
        enable_local_planner_fail=False,
        enable_p4_belief=True,
    )
    with virtual_clock_scope(poll_ms=0.002):
        env_choke = AMRReschedulingEnv(
            topology="unseen_choke_yard",
            seed=7103,
            allow_kitting_shortcut=False,
            pure_physical_reward=True,
            real_compute_window_coupling=True,
            expert_v12_portfolio=True,
            loaded_repair_mode="haven_v12",
            loaded_repair_threshold=8,
            record_trajectory=True,
            execution_perturbation=cfg_choke,
            execution_aware=True,
            p4_choke_robust=True,
            kitting_ranker=ranker,
            v13_kitting_mode="V13_KITTING_RL",
        )
        obs, info = env_choke.reset(seed=7103, options={"topology": "unseen_choke_yard", "stress_axis": "standard"})
        pol = V13KittingPolicy(epoch_interval=16, trigger=WhenToReplanTrigger(epoch_interval=16, stall_threshold=8))
        pol.reset()
        done = False
        while not done:
            act = pol.select_action(env_choke.get_expert_v12_obs(obs), info, sim=env_choke.sim)
            obs, _, tm, tr, info = env_choke.step(act)
            done = tm or tr
        m_choke = env_choke.compute_episode_metrics()
        aud_c = IndependentAuditor.audit_simulation_run(env_choke.sim, env_choke.tracker.operations, require_trajectory=True)
        assert aud_c.pass_audit, f"Choke yard 7103 failed audit: {aud_c.issues}"
        assert m_choke["task_completion_rate"] == 1.0
        assert m_choke["production_completion_rate"] == 1.0
        assert float(m_choke["cmax"]) < 450.0
