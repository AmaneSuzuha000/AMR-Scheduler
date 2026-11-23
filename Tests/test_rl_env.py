"""Phase R0 Unit Test: AMRReschedulingEnv Verification.

Verifies:
1. Gymnasium environment registration and observation/action space shapes.
2. Reset state normalization and mask validity.
3. Step execution for KEEP and REPLAN actions.
4. Physical disruption injection and transaction validation.
5. Terminal rewards and episode completion.
"""
from __future__ import annotations

import numpy as np
import pytest

from amr_sched.rl.env import AMRReschedulingEnv
from amr_sched.rl.spaces import ACTION_KEEP, NUM_ACTIONS, STATE_DIM, decode_action


def test_rl_env_spaces():
    env = AMRReschedulingEnv(topology="workshop", seed=10, max_steps=200)
    assert env.observation_space.shape == (STATE_DIM,)
    assert env.action_space.n == NUM_ACTIONS

    obs, info = env.reset(seed=10)
    assert obs.shape == (STATE_DIM,)
    assert isinstance(obs, np.ndarray)
    assert "action_mask" in info
    assert len(info["action_mask"]) == NUM_ACTIONS
    assert info["action_mask"][ACTION_KEEP] in (0, 1)


def test_rl_env_step_keep_and_replan():
    env = AMRReschedulingEnv(topology="ring", seed=11, max_steps=100)
    obs, info = env.reset(seed=11)

    # 1. Step KEEP
    next_obs, reward, terminated, truncated, info = env.step(ACTION_KEEP)
    assert next_obs.shape == (STATE_DIM,)
    assert isinstance(reward, float)
    assert not terminated
    assert info["action_name"] == "KEEP"

    # 2. Step a legal REPLAN action (Action 9: S3 @ 20ms)
    mask = info["action_mask"]
    legal_actions = [a for a in range(NUM_ACTIONS) if mask[a] == 1]
    assert len(legal_actions) > 0

    replan_act = legal_actions[-1]  # Pick a legal action
    next_obs, reward, terminated, truncated, info = env.step(replan_act)
    assert next_obs.shape == (STATE_DIM,)
    assert info["step"] == 2
    assert "compute_ms" in info


def test_rl_env_full_episode_rollout():
    """Verify an entire episode can roll out to termination with random legal actions."""
    env = AMRReschedulingEnv(topology="grid", seed=12, max_steps=150)
    obs, info = env.reset(seed=12)

    steps = 0
    terminated = False
    truncated = False
    while not (terminated or truncated) and steps < 150:
        mask = info["action_mask"]
        legal = np.where(mask == 1)[0]
        # Choose randomly among legal actions
        action = int(np.random.choice(legal))
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    assert steps > 0
    assert info["total_replans"] >= 0
    assert env.sim is not None


def test_baseline_policies():
    """Verify Phase R1 baseline policies can drive the environment to completion."""
    from amr_sched.rl.baselines import (
        NeverReplanPolicy,
        PeriodicPolicy,
        ReactiveFullPolicy,
        SlackThresholdPolicy,
    )

    policies = [
        NeverReplanPolicy(),
        PeriodicPolicy(period=15),
        ReactiveFullPolicy(),
        SlackThresholdPolicy(threshold=0.2),
    ]

    for pol in policies:
        env = AMRReschedulingEnv(topology="ring", seed=100, max_steps=80)
        obs, info = env.reset(seed=100)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated) and steps < 80:
            act = pol.select_action(obs, info)
            obs, r, terminated, truncated, info = env.step(act)
            steps += 1
        assert steps > 0
        assert "total_replans" in info


def test_rl_agent_dueling_dqn():
    """Verify Phase R2 Double Dueling DQN network, action masking, update step, and persistence."""
    import torch
    from amr_sched.rl.agent import DuelingQNetwork, RLCVCRAgent

    agent = RLCVCRAgent(state_dim=STATE_DIM, action_dim=NUM_ACTIONS, lr=1e-3)

    # 1. Forward pass check
    state = np.zeros(STATE_DIM, dtype=np.float32)
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    mask[ACTION_KEEP] = 1
    mask[5] = 1  # Only KEEP and action 5 are legal

    act = agent.select_action(state, mask, eval_mode=True)
    assert act in (ACTION_KEEP, 5), f"Selected masked action {act}!"

    # 2. Push transitions to buffer
    for i in range(100):
        s = np.random.randn(STATE_DIM).astype(np.float32)
        ns = np.random.randn(STATE_DIM).astype(np.float32)
        m = np.ones(NUM_ACTIONS, dtype=np.int8)
        nm = np.ones(NUM_ACTIONS, dtype=np.int8)
        agent.memory.push(s, ACTION_KEEP, -1.0, ns, False, m, nm)

    # 3. Update step
    loss = agent.update(batch_size=32)
    assert loss is not None and loss >= 0.0

    # 4. Checkpoint save and load
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_agent.pt"
        agent.save(ckpt_path)
        assert ckpt_path.exists()

        new_agent = RLCVCRAgent(state_dim=STATE_DIM, action_dim=NUM_ACTIONS)
        new_agent.load(ckpt_path)
        act2 = new_agent.select_action(state, mask, eval_mode=True)
        assert act2 in (ACTION_KEEP, 5)


def test_g0_g1_and_anytime_v4_contracts():
    """Verify label-free execution, decision_logs telemetry, and Anytime 22-D helpers."""
    from amr_sched.rl.spaces import (
        ANYTIME_STATE_DIM,
        compute_schedule_potential,
        is_in_secondary_hold,
        should_query_anytime_controller,
    )

    env = AMRReschedulingEnv(
        topology="ring",
        seed=2001,
        max_steps=80,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        real_compute_window_coupling=True,
        solver_override="BEST_CVCR_FULL",
    )
    obs, info = env.reset(seed=2001, options={"topology": "ring"})
    assert obs.shape == (STATE_DIM,)
    obs_a = env.get_anytime_obs(obs)
    assert obs_a.shape == (ANYTIME_STATE_DIM,)
    assert is_in_secondary_hold(env.sim) is False
    phi0 = compute_schedule_potential(env.sim, env.tracker)
    assert isinstance(phi0, float) and phi0 >= 0.0

    # Take one replan step (action 10 = S3 @ 50ms -> mapped to BEST_CVCR_FULL) and verify decision telemetry
    obs2, _, _, _, info2 = env.step(10)
    assert len(env.decision_logs) == 1
    lg = env.decision_logs[0]
    assert "t_propose_ms" in lg and lg["t_propose_ms"] >= 0.0
    assert "t_validate_commit_ms" in lg and "deadline_met" in lg
    assert obs2.shape == (STATE_DIM,)
    queried, state_flag = should_query_anytime_controller(
        env.get_anytime_obs(obs2), info2["action_mask"], tick=int(info2["step"]), last_decision_tick=0
    )
    assert isinstance(queried, bool) and isinstance(state_flag, tuple) and len(state_flag) == 2


def test_expert_portfolio_contracts():
    """Verify Multi-Expert 9-action environment, 24-D state, EXP_E1..E5 compute deadlines, and Auditor compliance."""
    from amr_sched.cvcr.auditor import IndependentAuditor
    from amr_sched.rl.spaces import (
        EXPERT_STATE_DIM,
        NUM_EXPERT_ACTIONS,
        should_query_expert_controller,
    )

    env = AMRReschedulingEnv(
        topology="workshop",
        seed=2001,
        max_steps=80,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        real_compute_window_coupling=True,
        expert_portfolio=True,
    )
    obs, info = env.reset(seed=2001, options={"topology": "workshop"})
    assert env.action_space.n == NUM_EXPERT_ACTIONS == 9
    assert info["action_mask"].shape == (NUM_EXPERT_ACTIONS,)
    exp_obs = env.get_expert_obs(obs)
    assert exp_obs.shape == (EXPERT_STATE_DIM,)
    assert 0.5 < exp_obs[-2] < 1.0  # workshop node count norm (~62/80)

    # Trigger each expert group cleanly at distinct ticks and audit trajectory
    for act in (1, 2, 5, 6, 7):
        obs, _, term, trunc, info = env.step(act)
        if not (term or trunc):
            q, _ = should_query_expert_controller(env.get_expert_obs(obs), info["action_mask"], int(info["step"]), 0)
            assert isinstance(q, bool)
            obs, _, _, _, info = env.step(ACTION_KEEP)
    assert len(env.decision_logs) >= 4
    for lg in env.decision_logs:
        assert lg["t_propose_ms"] <= lg["t_requested_ms"] + 5.0
    audit = IndependentAuditor().audit_simulation_run(env.sim, env.tracker.operations)
    assert audit.pass_audit, audit.issues


def test_expert_v6_portfolio_and_ci_contracts():
    """Verify V6 9-action portfolio, 26-D causal state, intra-agent c_i evaluator override, and Auditor."""
    from amr_sched.cvcr.auditor import IndependentAuditor
    from amr_sched.cvcr.evaluator import CVCREvaluator
    from amr_sched.cvcr.snapshot import NativeSnapshotCapture
    from amr_sched.rl.spaces import (
        EXPERT_V6_STATE_DIM,
        NUM_EXPERT_V6_ACTIONS,
        should_query_expert_controller,
    )

    env = AMRReschedulingEnv(
        topology="workshop",
        seed=2001,
        max_steps=80,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        real_compute_window_coupling=True,
        expert_v6_portfolio=True,
    )
    obs, info = env.reset(seed=2001, options={"topology": "workshop"})
    assert env.action_space.n == NUM_EXPERT_V6_ACTIONS == 9
    assert info["action_mask"].shape == (NUM_EXPERT_V6_ACTIONS,)
    obs_v6 = env.get_expert_v6_obs(obs)
    assert obs_v6.shape == (EXPERT_V6_STATE_DIM,)
    assert 0.0 <= obs_v6[-2] <= 1.0 and 0.0 <= obs_v6[-1] <= 1.0

    # Verify single-robot task_order_override (c_i) contract in CVCREvaluator
    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
    ev = CVCREvaluator(env.sim.network, snap, mode="B2", max_speed=3.0, use_rust=True)
    assign = {t.task_id: f"AGV_{(i % len(snap.robots)) + 1}" for i, t in enumerate(snap.tasks)}
    r0 = ev.evaluate(assign)
    assert r0.valid
    ev.set_incumbent(assign, r0)
    multi_rid = next((rid for rid, seq in r0.robot_order_seq.items() if len(seq) >= 2), None)
    if multi_rid is not None:
        rev_seq = [r0.robot_order_seq[multi_rid][1], r0.robot_order_seq[multi_rid][0]] + r0.robot_order_seq[multi_rid][2:]
        r1 = ev.evaluate(assign, task_order_override={multi_rid: rev_seq})
        assert r1.valid and r1.robot_order_seq[multi_rid][:2] == rev_seq[:2]

    for act in (1, 2, 5, 6, 7):
        obs, _, term, trunc, info = env.step(act)
        if not (term or trunc):
            q, _ = should_query_expert_controller(env.get_expert_v6_obs(obs), info["action_mask"], int(info["step"]), 0)
            assert isinstance(q, bool)
            obs, _, _, _, info = env.step(ACTION_KEEP)
    assert len(env.decision_logs) >= 4
    for lg in env.decision_logs:
        assert lg["t_propose_ms"] <= lg["t_requested_ms"] + 5.0
    audit = IndependentAuditor().audit_simulation_run(env.sim, env.tracker.operations)
    assert audit.pass_audit, audit.issues


def test_expert_v7_portfolio_and_temporal_audit():
    """Verify V7 9-action portfolio, 30-D state, explicit c_i execution & rollback, and temporal Auditor."""
    from amr_sched.cvcr.auditor import IndependentAuditor
    from amr_sched.cvcr.policy import CandidateProposal
    from amr_sched.rl.spaces import (
        EXPERT_V7_STATE_DIM,
        NUM_EXPERT_V7_ACTIONS,
        should_query_expert_controller,
    )

    for topo in ("workshop", "grid", "ring"):
        env = AMRReschedulingEnv(
            topology=topo,
            seed=2001,
            max_steps=80,
            allow_kitting_shortcut=False,
            pure_physical_reward=True,
            real_compute_window_coupling=True,
            expert_v7_portfolio=True,
        )
        obs, info = env.reset(seed=2001, options={"topology": topo})
        assert env.action_space.n == NUM_EXPERT_V7_ACTIONS == 9
        assert info["action_mask"].shape == (NUM_EXPERT_V7_ACTIONS,)
        obs_v7 = env.get_expert_v7_obs(obs)
        assert obs_v7.shape == (EXPERT_V7_STATE_DIM,)
        assert np.all(np.isfinite(obs_v7))
        for idx in range(26, 30):
            assert 0.0 <= float(obs_v7[idx]) <= 1.0

    # Test closed-loop stepping with V7 experts including V7_E6_BALANCE_KM (action 8)
    env = AMRReschedulingEnv(
        topology="grid",
        seed=2001,
        max_steps=80,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        real_compute_window_coupling=True,
        expert_v7_portfolio=True,
        record_trajectory=True,
    )
    obs, info = env.reset(seed=2001, options={"topology": "grid"})
    for act in (1, 2, 5, 7, 8):
        obs, _, term, trunc, info = env.step(act)
        if not (term or trunc):
            q, _ = should_query_expert_controller(
                env.get_expert_v7_obs(obs), info["action_mask"], int(info["step"]), 0
            )
            assert isinstance(q, bool)
            obs, _, _, _, info = env.step(ACTION_KEEP)
    assert len(env.decision_logs) >= 4
    for lg in env.decision_logs:
        assert lg["t_propose_ms"] <= lg["t_requested_ms"] + 5.0
    audit = IndependentAuditor().audit_simulation_run(
        env.sim, env.tracker.operations, require_trajectory=True
    )
    assert audit.pass_audit, audit.issues

    # Verify explicit queue commit & rollback restores previous pinned queues
    from amr_sched.cvcr.snapshot import NativeSnapshotCapture

    sim = env.sim
    snap = NativeSnapshotCapture.capture(sim, env.tracker.operations, hide_future_releases=True)
    prev_queues = {rid: list(q) for rid, q in sim._pinned_task_queues.items()}
    bad_prop = CandidateProposal(
        schema="cvcr_proposal_v1",
        policy_id="TEST_ROLLBACK",
        snapshot_digest=snap.compute_digest(),
        affected_robots=[agvs_id := sim.agvs[0].agv_id],
        affected_tasks=[],
        affected_operations=[],
        proposed_assignment={},
        proposed_routes={},
        proposed_bays={},
        proposed_crews={},
        predicted_obj=0.0,
        predicted_makespan=0,
        candidate_digest="test_rollback_digest",
        telemetry={},
        proposed_task_sequences={agvs_id: ["NONEXISTENT_TASK"]},
    )
    receipt = env.owner.validate_and_commit(bad_prop)
    assert not receipt.accepted
    assert sim._pinned_task_queues == prev_queues

    # Verify temporal fault audit still catches pre-fault multi-AGV node occupancy conflict
    agvs = sim.agvs
    min_node = min(sim.network.nodes.values(), key=lambda n: int(getattr(n, "capacity", 1)))
    cap = int(getattr(min_node, "capacity", 1))
    assert len(agvs) >= cap + 1 and len(sim._trajectory) >= 2
    sim.set_agv_faulted(agvs[0].agv_id, True)
    rec = sim._trajectory[0]
    conflict_agvs = agvs[: cap + 1]
    orig_pos = {a.agv_id: rec.agv_positions[a.agv_id] for a in conflict_agvs}
    try:
        for a in conflict_agvs:
            rec.agv_positions[a.agv_id] = min_node.node_id
        bad_audit = IndependentAuditor().audit_simulation_run(
            sim, env.tracker.operations, require_trajectory=True
        )
        assert not bad_audit.pass_audit
        assert any("NODE_CAPACITY_EXCEEDED" in msg for msg in bad_audit.issues), bad_audit.issues
    finally:
        for a in conflict_agvs:
            rec.agv_positions[a.agv_id] = orig_pos[a.agv_id]


def test_expert_v8_portfolio_e2g_rcdr_ltm():
    """Verify V8 E2G dual graph, RCDR drift trigger, LTM tx rollback, and trajectory auditing."""
    from amr_sched.cvcr.auditor import IndependentAuditor
    from amr_sched.cvcr.policy import CandidateProposal
    from amr_sched.cvcr.snapshot import NativeSnapshotCapture
    from amr_sched.rl import (
        EXPERT_V8_STATE_DIM,
        NUM_EXPERT_V8_ACTIONS,
        V8_AGV_FEAT_DIM,
        V8_GRAPH_SUMMARY_DIM,
        V8_MAX_AGVS,
        V8_MAX_OPS,
        V8_OP_FEAT_DIM,
        should_query_v8_controller,
    )

    assert EXPERT_V8_STATE_DIM == 36
    assert NUM_EXPERT_V8_ACTIONS == 9
    assert V8_GRAPH_SUMMARY_DIM == 54

    for topo in ("ring", "grid", "workshop"):
        env = AMRReschedulingEnv(
            topology=topo,
            seed=2001,
            max_steps=80,
            allow_kitting_shortcut=False,
            pure_physical_reward=True,
            real_compute_window_coupling=True,
            expert_v8_portfolio=True,
        )
        obs, info = env.reset(seed=2001, options={"topology": topo})
        assert env.action_space.n == NUM_EXPERT_V8_ACTIONS == 9
        obs_v8 = env.get_expert_v8_obs(obs)
        assert obs_v8.shape == (EXPERT_V8_STATE_DIM,)
        assert np.all(np.isfinite(obs_v8))
        dg = env.get_expert_v8_dual_graph(obs)
        assert dg["global_vec"].shape == (EXPERT_V8_STATE_DIM,)
        assert dg["op_nodes"].shape == (V8_MAX_OPS, V8_OP_FEAT_DIM)
        assert dg["op_adj"].shape == (3, V8_MAX_OPS, V8_MAX_OPS)
        assert dg["op_mask"].shape == (V8_MAX_OPS,)
        assert dg["agv_nodes"].shape == (V8_MAX_AGVS, V8_AGV_FEAT_DIM)
        assert dg["agv_op_adj"].shape == (V8_MAX_AGVS, V8_MAX_OPS)
        assert dg["agv_mask"].shape == (V8_MAX_AGVS,)
        assert dg["bandit_vec"].shape == (V8_GRAPH_SUMMARY_DIM,)
        for arr in dg.values():
            assert np.all(np.isfinite(arr))
        # Initial kickoff gate fires on the first tick with legal replan mask (t <= 8)
        if np.any(info["action_mask"][1:] == 1):
            q0, _ = should_query_v8_controller(obs_v8, info["action_mask"], 0, -999, 12, False)
            assert q0 is True
        else:
            fired = False
            for st_i in range(1, 8):
                obs, _, _, _, info = env.step(ACTION_KEEP)
                if np.any(info["action_mask"][1:] == 1):
                    q0, _ = should_query_v8_controller(
                        env.get_expert_v8_obs(obs), info["action_mask"], st_i, -999, 12, False
                    )
                    assert q0 is True
                    fired = True
                    break
            assert fired

    # Closed-loop stepping with V8 experts (E1, E2, E4_LTM, E5, E6_BALANCE_KM_PC) + live audit
    env = AMRReschedulingEnv(
        topology="grid",
        seed=2001,
        max_steps=80,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        real_compute_window_coupling=True,
        expert_v8_portfolio=True,
        record_trajectory=True,
    )
    obs, info = env.reset(seed=2001, options={"topology": "grid"})
    for act in (1, 2, 5, 7, 8):
        obs, _, term, trunc, info = env.step(act)
        if not (term or trunc):
            obs, _, _, _, info = env.step(ACTION_KEEP)
    assert len(env.decision_logs) >= 4
    for lg in env.decision_logs:
        assert lg["t_propose_ms"] <= lg["t_requested_ms"] + 5.0
    aud = IndependentAuditor().audit_simulation_run(
        env.sim, env.tracker.operations, require_trajectory=True
    )
    assert aud.pass_audit, aud.issues

    # Verify rollback of traffic weights on failed transaction commit
    sim = env.sim
    snap = NativeSnapshotCapture.capture(sim, env.tracker.operations, hide_future_releases=True)
    prev_tw = dict(getattr(sim, "_traffic_weights_q3", {}))
    bad_prop = CandidateProposal(
        schema="cvcr_proposal_v1",
        policy_id="TEST_LTM_ROLLBACK",
        snapshot_digest=snap.compute_digest(),
        affected_robots=[],
        affected_tasks=[],
        affected_operations=[],
        proposed_assignment={},
        proposed_routes={},
        proposed_bays={},
        proposed_crews={},
        predicted_obj=0.0,
        predicted_makespan=0,
        candidate_digest="test_ltm_rollback_digest",
        telemetry={},
        proposed_traffic_weights={("NO_SUCH_SRC", "NO_SUCH_DST"): 16},
    )
    assert not env.owner.validate_and_commit(bad_prop).accepted
    assert getattr(sim, "_traffic_weights_q3", {}) == prev_tw


def test_expert_v9_portfolio_cpsat_guide_rcdr():
    """Verify V9 38-D causal obs, CP-SAT joint window, corridor guide rollback, and trajectory auditing."""
    from amr_sched.cvcr.auditor import IndependentAuditor
    from amr_sched.cvcr.policy import CandidateProposal
    from amr_sched.cvcr.snapshot import NativeSnapshotCapture
    from amr_sched.rl import (
        EXPERT_V9_STATE_DIM,
        NUM_EXPERT_V9_ACTIONS,
        V9_AGV_FEAT_DIM,
        V9_GRAPH_SUMMARY_DIM,
        V9_MAX_AGVS,
        V9_MAX_OPS,
        V9_OP_FEAT_DIM,
        should_query_v9_controller,
    )

    assert EXPERT_V9_STATE_DIM == 38
    assert NUM_EXPERT_V9_ACTIONS == 9
    assert V9_GRAPH_SUMMARY_DIM == 56

    for topo in ("ring", "grid", "workshop"):
        env = AMRReschedulingEnv(
            topology=topo,
            seed=2001,
            max_steps=80,
            allow_kitting_shortcut=False,
            pure_physical_reward=True,
            real_compute_window_coupling=True,
            expert_v9_portfolio=True,
        )
        obs, info = env.reset(seed=2001, options={"topology": topo})
        assert env.action_space.n == NUM_EXPERT_V9_ACTIONS == 9
        obs_v9 = env.get_expert_v9_obs(obs)
        assert obs_v9.shape == (EXPERT_V9_STATE_DIM,)
        assert np.all(np.isfinite(obs_v9))
        dg = env.get_expert_v9_dual_graph(obs)
        assert dg["global_vec"].shape == (EXPERT_V9_STATE_DIM,)
        assert dg["op_nodes"].shape == (V9_MAX_OPS, V9_OP_FEAT_DIM)
        assert dg["op_adj"].shape == (3, V9_MAX_OPS, V9_MAX_OPS)
        assert dg["op_mask"].shape == (V9_MAX_OPS,)
        assert dg["agv_nodes"].shape == (V9_MAX_AGVS, V9_AGV_FEAT_DIM)
        assert dg["agv_op_adj"].shape == (V9_MAX_AGVS, V9_MAX_OPS)
        assert dg["agv_mask"].shape == (V9_MAX_AGVS,)
        assert dg["bandit_vec"].shape == (V9_GRAPH_SUMMARY_DIM,)
        for arr in dg.values():
            assert np.all(np.isfinite(arr))
        if np.any(info["action_mask"][1:] == 1):
            q0, _ = should_query_v9_controller(obs_v9, info["action_mask"], 0, -999, 14, False)
            assert q0 is True
        else:
            fired = False
            for st_i in range(1, 8):
                obs, _, _, _, info = env.step(ACTION_KEEP)
                if np.any(info["action_mask"][1:] == 1):
                    q0, _ = should_query_v9_controller(
                        env.get_expert_v9_obs(obs), info["action_mask"], st_i, -999, 14, False
                    )
                    assert q0 is True
                    fired = True
                    break
            assert fired

    env = AMRReschedulingEnv(
        topology="grid",
        seed=2001,
        max_steps=80,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        real_compute_window_coupling=True,
        expert_v9_portfolio=True,
        record_trajectory=True,
    )
    obs, info = env.reset(seed=2001, options={"topology": "grid"})
    for act in (1, 2, 4, 5, 7, 8):
        obs, _, term, trunc, info = env.step(act)
        if not (term or trunc):
            obs, _, _, _, info = env.step(ACTION_KEEP)
    assert len(env.decision_logs) >= 5
    aud = IndependentAuditor().audit_simulation_run(
        env.sim, env.tracker.operations, require_trajectory=True
    )
    assert aud.pass_audit, aud.issues

    sim = env.sim
    snap = NativeSnapshotCapture.capture(sim, env.tracker.operations, hide_future_releases=True)
    prev_guides = {k: list(v) for k, v in getattr(sim, "_guide_corridors", {}).items()}
    bad_g = CandidateProposal(
        schema="cvcr_proposal_v1",
        policy_id="TEST_GUIDE_ROLLBACK",
        snapshot_digest=snap.compute_digest(),
        affected_robots=[],
        affected_tasks=[],
        affected_operations=[],
        proposed_assignment={},
        proposed_routes={},
        proposed_bays={},
        proposed_crews={},
        predicted_obj=0.0,
        predicted_makespan=0,
        candidate_digest="test_guide_rollback_digest",
        telemetry={},
        proposed_guide_routes={"agv_0": [("NO_SRC", "NO_DST")]},
    )
    assert not env.owner.validate_and_commit(bad_g).accepted
    assert getattr(sim, "_guide_corridors", {}) == prev_guides
