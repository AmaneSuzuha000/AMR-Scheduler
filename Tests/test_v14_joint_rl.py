"""Tests for V14 3-Layer Execution-Aware Hierarchical Joint RL (`V14_JOINT_RL`)."""

from __future__ import annotations

import numpy as np

from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.execution_feedback import AmrExecutionStatus, ExecutionFeedbackPacket
from amr_sched.cvcr.p2_perturbation import P2PerturbationConfig
from amr_sched.cvcr.policy import LGCVCRPolicy, _LazyKittingSurrogate
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.cvcr.v14_joint_rl import (
    ANYTIME_OPS,
    V14_CAND_FEAT_DIM,
    V14_OP_FEAT_DIM,
    V14JointPolicy,
    augment_candidate_features_v14,
    extract_v14_op_features,
    generate_v14_joint_candidates,
)
from amr_sched.rl.env import AMRReschedulingEnv
from amr_sched.rl.v13_controller import V13KittingPolicy
from amr_sched.scenarios_v13_kitting import V13_FACTOR_COMBOS, build_v13_paired_dev_scenario


def test_v14_op_portfolio_and_candidate_feature_dimensions() -> None:
    """Verify 16-D op features, 32-D drift-augmented candidate features, and prior anytime scoring."""
    net, tasks, agvs, ops, meta = build_v13_paired_dev_scenario(
        8201,
        shift_timing="late",
        missing_count="multi",
        corridor_mode="choke",
        window_count="two",
    )
    from amr_sched.allocation.greedy import GreedyAllocator
    from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
    from amr_sched.reschedule import ReschedulePolicy
    from amr_sched.simulation import SimulationConfig, Simulator

    sim = Simulator(
        net,
        tasks,
        agvs,
        SpaceTimeAStarPlanner(max_speed=3.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=600, planning_horizon=260, reschedule_policy=ReschedulePolicy.NONE),
    )
    sim.reset()
    sim.step_once()
    pkt = ExecutionFeedbackPacket(
        tick=sim._step_cursor,
        robot_states={agvs[0].agv_id: AmrExecutionStatus(robot_id=agvs[0].agv_id, speed_scale=0.60, eta_drift_steps=3)},
    )
    snap = NativeSnapshotCapture.capture(sim, ops, hide_future_releases=True, execution_feedback=pkt)
    clf = classify_commitments(snap)
    ev = CVCREvaluator(net, snap, clf, use_rust=True)
    pol = LGCVCRPolicy("V14_JOINT_RL", net, use_rust=True)
    healthy = pol._healthy_robots(snap, clf.disturbed_robots)
    base_assign = pol._base_assignment(snap, healthy, {o.op_id: i for i, o in enumerate(ops)})
    base_bays = {o.op_id: o.bay_id for o in ops}
    base_crews = {o.op_id: o.crew_id for o in ops}
    base_ev = ev.evaluate(base_assign, op_bay_overrides=base_bays, op_crew_overrides=base_crews)

    succ = {o.op_id: 0 for o in snap.operations}
    unstarted, op_mat = extract_v14_op_features(snap, base_ev, {}, succ)
    assert len(unstarted) >= 2
    assert op_mat.shape == (len(unstarted), V14_OP_FEAT_DIM)

    surr = _LazyKittingSurrogate(net, snap, {o.op_id: i for i, o in enumerate(ops)})
    v14_pol = V14JointPolicy(weights=None)
    cands = generate_v14_joint_candidates(
        net,
        snap,
        clf,
        ev,
        surr,
        base_assign,
        base_bays,
        base_crews,
        base_ev,
        {o.op_id: i for i, o in enumerate(ops)},
        healthy,
        policy=v14_pol,
    )
    assert len(cands) >= 2
    vec32 = augment_candidate_features_v14(cands[0], snap, base_ev, {})
    assert vec32.shape == (V14_CAND_FEAT_DIM,)

    g = np.zeros((12,), dtype=np.float32)
    scores, cont_p = v14_pol.score_candidates(g, cands, snap, base_ev, {})
    assert scores.shape == (len(cands),)
    assert 0.0 <= cont_p <= 1.0
    op_choice = v14_pol.select_anytime_operator(g, np.asarray([0.8, 0.5, 0.0, 0.0, 1.0, 1.0, cont_p, 0.5], dtype=np.float32))
    assert 0 <= op_choice < len(ANYTIME_OPS)


def test_v14_joint_rl_closed_loop_with_and_without_p2_perturbation() -> None:
    """Verify full V14_JOINT_RL env closed-loop passes IndependentAuditor clean and P2-perturbed."""
    sh, ms, cm, wc = V13_FACTOR_COMBOS[-1]  # late, multi, bypass/two-like combo
    spec = {"shift_timing": sh, "missing_count": ms, "corridor_mode": cm, "window_count": wc}
    for use_p2 in (False, True):
        p2 = (
            P2PerturbationConfig(seed=8201, enable_travel_drift=True, enable_dynamic_obstacles=True, enable_p4_belief=True)
            if use_p2
            else None
        )
        env = AMRReschedulingEnv(
            topology="unseen_choke_yard",
            seed=8201,
            allow_kitting_shortcut=False,
            pure_physical_reward=True,
            real_compute_window_coupling=True,
            expert_v12_portfolio=True,
            loaded_repair_mode="haven_v12",
            loaded_repair_threshold=8,
            record_trajectory=True,
            v13_kitting_mode="V14_JOINT_RL",
            execution_perturbation=p2,
            execution_aware=True,
            p4_choke_robust=use_p2,
        )
        obs, info = env.reset(seed=8201, options={"v13_kitting_spec": spec})
        pol = V13KittingPolicy(epoch_interval=16)
        pol.reset()
        done = False
        while not done:
            o12 = env.get_expert_v12_obs(obs)
            act = pol.select_action(o12, info, sim=env.sim)
            obs, _, tm, tr, info = env.step(act)
            done = tm or tr
        m = env.compute_episode_metrics()
        aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
        assert aud.pass_audit, f"V14 closed-loop audit failed (use_p2={use_p2}): {aud.issues}"
        assert m["task_completion_rate"] == 1.0
        assert m["production_completion_rate"] == 1.0
        decs = [d["kitting_decision"] for d in env.decision_logs if d.get("kitting_decision")]
        assert decs and decs[0].get("policy_version") == "V14_JOINT_RL"
