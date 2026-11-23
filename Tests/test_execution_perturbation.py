"""Tests for P2 L1/L2 Execution Uncertainty Model and Execution-Blind vs Execution-Aware Closed Loop."""

from __future__ import annotations

from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.p2_perturbation import (
    ExecutionPerturbationModel,
    P2PerturbationConfig,
    ScheduledLocalFail,
    ScheduledObstacle,
)
from amr_sched.cvcr.production import OperationSnapshot
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.entities import Agv, Task
from amr_sched.network import Edge, Network, Node, NodeKind
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.rl import AMRReschedulingEnv
from amr_sched.rl.spaces import ACTION_KEEP
from amr_sched.rl.v12_controllers import RuleV12ProactivePolicy
from amr_sched.rl.v13_controller import V13KittingPolicy
from amr_sched.workshop import build_scenario
from amr_sched.simulation import SimulationConfig, Simulator


def _make_diamond_corridor_network() -> Network:
    """Build deterministic depot->workstation diamond with fast narrow shortest lane & bypass."""
    net = Network()
    net.add_node(Node("N_DEPOT", 0.0, 0.0, kind=NodeKind.BUFFER, capacity=4))
    net.add_node(Node("N_MID", 4.0, 0.0, kind=NodeKind.AISLE, capacity=2))
    net.add_node(Node("N_ALT1", 2.0, 5.0, kind=NodeKind.AISLE, capacity=2))
    net.add_node(Node("N_ALT2", 6.0, 5.0, kind=NodeKind.AISLE, capacity=2))
    net.add_node(Node("N_BAY", 8.0, 0.0, kind=NodeKind.WORKSTATION, capacity=2))
    for u, v, d in (
        ("N_DEPOT", "N_MID", 4.0),
        ("N_MID", "N_BAY", 4.0),
        ("N_DEPOT", "N_ALT1", 6.0),
        ("N_ALT1", "N_ALT2", 6.0),
        ("N_ALT2", "N_BAY", 6.0),
    ):
        net.add_edge(Edge(u, v, length=d, speed_limit=2.0, capacity=1))
        net.add_edge(Edge(v, u, length=d, speed_limit=2.0, capacity=1))
    return net


def test_p2_nominal_zero_diff_and_reset_idempotence() -> None:
    """Verify P2 perturbation model preserves simulator reset() idempotence & nominal parity."""
    net, tasks_base, agvs_base = build_scenario(n_tasks=10, n_agvs=4, release_window=15, seed=77)
    tasks_a = [Task(t.task_id, t.pickup, t.dropoff, t.release_step, t.due_step) for t in tasks_base]
    agvs_a = [Agv(a.agv_id, max_speed=a.max_speed, start_node=a.start_node) for a in agvs_base]
    tasks_b = [Task(t.task_id, t.pickup, t.dropoff, t.release_step, t.due_step) for t in tasks_base]
    agvs_b = [Agv(a.agv_id, max_speed=a.max_speed, start_node=a.start_node) for a in agvs_base]

    cfg = SimulationConfig(dt=1.0, max_steps=600, node_hold_steps=6, deadlock_timeout=20, record_trajectory=True)
    sim_nom1 = Simulator(net, tasks_a, agvs_a, SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0), GreedyAllocator(), cfg)
    res_nom1 = sim_nom1.run()
    sim_nom2 = Simulator(net, tasks_b, agvs_b, SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0), GreedyAllocator(), cfg)
    res_nom2 = sim_nom2.run()
    assert res_nom1.finished_step == res_nom2.finished_step
    assert repr(res_nom1.summary()) == repr(res_nom2.summary())

    # With P2 perturbation active: two reset()+run() cycles produce identical results and pass audit
    p2_cfg = P2PerturbationConfig(seed=101, drift_prob=0.35, obstacle_count=2, local_fail_count=2)
    p2_model = ExecutionPerturbationModel(p2_cfg)
    sim_p2 = Simulator(
        net,
        [Task(t.task_id, t.pickup, t.dropoff, t.release_step, t.due_step) for t in tasks_base],
        [Agv(a.agv_id, max_speed=a.max_speed, start_node=a.start_node) for a in agvs_base],
        SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0),
        GreedyAllocator(),
        cfg,
        perturbation_model=p2_model,
    )
    r1 = sim_p2.run()
    snap_r1 = (r1.finished_step, dict(p2_model.summary()), len(r1.completed_tasks()))
    rep1 = IndependentAuditor.audit_simulation_run(sim_p2, require_trajectory=True)
    assert rep1.pass_audit, f"Auditor issues on run1: {rep1.issues}"

    r2 = sim_p2.run()
    snap_r2 = (r2.finished_step, dict(p2_model.summary()), len(r2.completed_tasks()))
    rep2 = IndependentAuditor.audit_simulation_run(sim_p2, require_trajectory=True)
    assert rep2.pass_audit, f"Auditor issues on run2: {rep2.issues}"
    assert snap_r1 == snap_r2


def test_p2_three_axes_causal_feedback_and_audit() -> None:
    """Verify all 3 perturbation axes activate causally without oracle leakage and stay auditable."""
    net = _make_diamond_corridor_network()
    tasks = [
        Task(f"T_{i}", "N_DEPOT", "N_BAY", release_step=i * 3, due_step=80 + i * 5)
        for i in range(6)
    ]
    agvs = [Agv(f"AGV_{i}", max_speed=2.0, start_node="N_DEPOT") for i in range(3)]

    p2_cfg = P2PerturbationConfig(
        seed=42,
        enable_travel_drift=True,
        drift_prob=0.55,
        early_max_steps=1,
        late_max_steps=2,
        enable_dynamic_obstacles=True,
        obstacle_count=1,
        obstacle_duration_steps=12,
        obstacle_window_range=(6, 18),
        est_clear_horizon_steps=10,
        enable_local_planner_fail=True,
        local_fail_count=1,
        local_fail_duration_steps=8,
        local_fail_window_range=(8, 24),
    )
    model = ExecutionPerturbationModel(p2_cfg)
    sim = Simulator(
        net,
        tasks,
        agvs,
        SpaceTimeAStarPlanner(max_speed=2.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=400, node_hold_steps=6, deadlock_timeout=20, record_trajectory=True),
        perturbation_model=model,
    )
    sim.reset()
    assert len(model.obstacles) == 1
    assert len(model.local_fails) == 1
    ob = model.obstacles[0]
    lf = model.local_fails[0]

    # Future disturbance must NOT be leaked before start_step
    early_fb = model.capture_feedback(sim, max(0, min(ob.start_step, lf.start_step) - 1))
    assert early_fb is None or not early_fb.corridor_updates

    # Check strict causality at obstacle start and mid-window:
    # blocked_until_step = step + est_clear_horizon_steps (never leaking true ob.end_step)
    mid_ob = min(ob.end_step - 1, ob.start_step + 2)
    fb_a = model.capture_feedback(sim, ob.start_step)
    fb_b = model.capture_feedback(sim, mid_ob)
    assert fb_a is not None and (ob.u, ob.v) in fb_a.corridor_updates
    assert fb_a.corridor_updates[(ob.u, ob.v)].blocked_until_step == ob.start_step + p2_cfg.est_clear_horizon_steps
    assert fb_b is not None and (ob.u, ob.v) in fb_b.corridor_updates
    assert fb_b.corridor_updates[(ob.u, ob.v)].blocked_until_step == mid_ob + p2_cfg.est_clear_horizon_steps
    # Past clearance must clear immediately
    fb_post = model.capture_feedback(sim, ob.end_step)
    assert fb_post is None or (ob.u, ob.v) not in fb_post.corridor_updates

    while sim.step_once():
        pass
    rep = IndependentAuditor.audit_simulation_run(sim, require_trajectory=True)
    assert rep.pass_audit, f"P2 3-axis run failed audit: {rep.issues}"
    summary = model.summary()
    assert summary["drift_events"] > 0
    assert summary["early_hops"] + summary["late_hops"] > 0
    assert summary["local_fail_ticks"] > 0
    assert len(sim._build_result().completed_tasks()) == len(tasks)


def test_p2_closed_loop_execution_blind_vs_aware() -> None:
    """Compare Execution-Blind vs Execution-Aware closed-loop V13 under identical P2 disturbances."""
    p2_cfg = P2PerturbationConfig(
        seed=2026,
        enable_travel_drift=True,
        drift_prob=0.30,
        enable_dynamic_obstacles=True,
        obstacle_count=2,
        obstacle_duration_steps=14,
        obstacle_window_range=(16, 68),
        est_clear_horizon_steps=12,
        enable_local_planner_fail=True,
        local_fail_count=2,
        local_fail_duration_steps=10,
        local_fail_window_range=(20, 72),
    )

    def _run_episode(aware: bool) -> tuple[dict, bool, list]:
        env = AMRReschedulingEnv(
            topology="workshop",
            seed=7101,
            max_steps=450,
            use_rust=False,
            expert_v12_portfolio=True,
            loaded_repair_mode="haven_v12",
            v13_kitting_mode="V13_KITTING_DET",
            record_trajectory=True,
            execution_perturbation=p2_cfg,
            execution_aware=aware,
        )
        obs, info = env.reset(seed=7101)
        assert env.sim is not None and env.tracker is not None
        assert bool(getattr(env.sim, "_p2_execution_aware", None)) is aware
        ctrl = V13KittingPolicy(epoch_interval=16, stall_threshold=8)
        ctrl.reset()
        done = False
        while not done:
            obs_v12 = env.get_expert_v12_obs(obs)
            act = ctrl.select_action(obs_v12, info, sim=env.sim)
            obs, _rew, term, trunc, info = env.step(act)
            done = term or trunc
        metrics = env.compute_episode_metrics()
        rep = IndependentAuditor.audit_simulation_run(env.sim, operations=env.tracker.operations, require_trajectory=True)
        return metrics, rep.pass_audit, rep.issues

    m_blind, ok_blind, iss_blind = _run_episode(False)
    m_aware, ok_aware, iss_aware = _run_episode(True)

    assert ok_blind, f"Blind failed audit: {iss_blind}"
    assert ok_aware, f"Aware failed audit: {iss_aware}"
    assert m_blind["execution_aware"] is False
    assert m_aware["execution_aware"] is True
    assert m_blind["task_completion_rate"] == 1.0
    assert m_aware["task_completion_rate"] == 1.0
    assert m_blind["production_completion_rate"] == 1.0
    assert m_aware["production_completion_rate"] == 1.0
    assert m_blind["p2_perturbation"]["drift_events"] > 0
    assert m_aware["p2_perturbation"]["drift_events"] > 0


def test_p2_targeted_obstacle_detour_improves_execution_aware() -> None:
    """Verify Execution-Aware surgically detours around a blocked shortest lane instead of stalling."""
    net = _make_diamond_corridor_network()
    # Two robots at N_DEPOT carrying tasks to N_BAY across a blocked central corridor N_DEPOT<->N_MID
    tasks = [
        Task("T0", "N_DEPOT", "N_BAY", release_step=0, due_step=50),
        Task("T1", "N_DEPOT", "N_BAY", release_step=0, due_step=50),
    ]
    agvs = [
        Agv("A0", max_speed=2.0, start_node="N_DEPOT"),
        Agv("A1", max_speed=2.0, start_node="N_DEPOT"),
    ]
    ops = [
        OperationSnapshot(
            op_id="OP0",
            kit_task_ids=("T0", "T1"),
            bay_id="N_BAY",
            crew_id=1,
            tool_id=1,
            proc_steps=6,
            due_step=65,
            eligible_bays=("N_BAY",),
            eligible_crews=(1,),
        )
    ]

    class _FixedObstacleModel(ExecutionPerturbationModel):
        """Deterministic obstacle on N_DEPOT<->N_MID from step 0 to 16 (keeps bypass N_ALT1->N_ALT2 open)."""

        def reset(self, network, tasks, agvs, *, excluded_agvs=()):  # type: ignore[override]
            super().reset(network, tasks, agvs, excluded_agvs=excluded_agvs)
            self.obstacles = [ScheduledObstacle(u="N_DEPOT", v="N_MID", start_step=0, end_step=16)]
            self._obs_map = {("N_DEPOT", "N_MID"): [(0, 16)]}

    def _run_case(aware: bool) -> tuple[dict, int]:
        tasks_copy = [Task(t.task_id, t.pickup, t.dropoff, t.release_step, t.due_step) for t in tasks]
        agvs_copy = [Agv(a.agv_id, max_speed=a.max_speed, start_node=a.start_node) for a in agvs]
        ops_copy = [
            OperationSnapshot(
                op_id=op.op_id,
                kit_task_ids=op.kit_task_ids,
                bay_id=op.bay_id,
                crew_id=op.crew_id,
                tool_id=op.tool_id,
                proc_steps=op.proc_steps,
                due_step=op.due_step,
                eligible_bays=op.eligible_bays,
                eligible_crews=op.eligible_crews,
            )
            for op in ops
        ]
        model = _FixedObstacleModel(
            P2PerturbationConfig(
                seed=9,
                enable_travel_drift=False,
                enable_dynamic_obstacles=False,
                enable_local_planner_fail=False,
                est_clear_horizon_steps=10,
            )
        )
        env = AMRReschedulingEnv(
            topology="workshop",
            seed=9,
            max_steps=160,
            use_rust=False,
            expert_v12_portfolio=True,
            loaded_repair_mode="haven_v12",
            record_trajectory=True,
            execution_perturbation=model,
            execution_aware=aware,
        )
        obs, info = env.reset(
            seed=9,
            options={
                "scenario_tuple": (net, tasks_copy, agvs_copy, ops_copy, {}),
                "execution_perturbation": model,
                "execution_aware": aware,
            },
        )
        ctrl = RuleV12ProactivePolicy(epoch_interval=16, stall_threshold=8)
        ctrl.reset()
        done = False
        while not done:
            obs_v12 = env.get_expert_v12_obs(obs)
            act = ctrl.select_action(obs_v12, info, sim=env.sim)
            obs, _rew, term, trunc, info = env.step(act)
            done = term or trunc
        metrics = env.compute_episode_metrics()
        rep = IndependentAuditor.audit_simulation_run(env.sim, operations=env.tracker.operations, require_trajectory=True)
        assert rep.pass_audit, f"Targeted detour aware={aware} failed audit: {rep.issues}"
        return metrics, int(model.summary()["obstacle_blocks_hit"])

    m_blind, hits_blind = _run_case(False)
    m_aware, hits_aware = _run_case(True)
    assert m_aware["task_completion_rate"] == 1.0
    assert m_aware["production_completion_rate"] == 1.0
    assert hits_blind > hits_aware
    assert m_aware["cmax"] <= m_blind["cmax"]
