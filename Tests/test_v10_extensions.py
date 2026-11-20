"""Unit and integration tests for V10 extensions: SPGA, L-RHO, Unseen Layouts, and 5-Axis Stress."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from amr_sched.baselines.spga_cor2026 import (
    SPGAConfig,
    SPGAProblem,
    SPGASolver,
    run_spga_policy_proposal,
)
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.lrho import (
    LRHO_AGV_FEAT_DIM,
    LRHO_TASK_FEAT_DIM,
    LRHOPredictor,
    extract_lrho_features,
)
from amr_sched.cvcr.policy import LGCVCRPolicy
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.rl import ACTION_KEEP, AMRReschedulingEnv
from amr_sched.scenarios_v10 import (
    OPENING_FEATURE_DIM,
    STRESS_AXES,
    UNSEEN_LAYOUT_NAMES,
    build_unseen_choke_yard_network,
    build_unseen_ladder_network,
    build_v10_stress_scenario,
    extract_opening_structural_features,
)


def test_unseen_topologies_connectivity_and_5axis_generator():
    """Verify unseen_ladder & unseen_choke_yard graphs and all 5-axis stress builders are valid & deterministic."""
    for builder, min_nodes in ((build_unseen_ladder_network, 24), (build_unseen_choke_yard_network, 28)):
        net = builder(aisle_capacity=1)
        assert len(net.nodes) >= min_nodes
        picks = [n.node_id for n in net.nodes.values() if n.kind.value in ("disassembly", "buffer")]
        drops = [n.node_id for n in net.nodes.values() if n.kind.value in ("workstation", "assembly")]
        assert picks and drops
        for p in picks:
            dmap = net.shortest_travel_times(p, 3.0)
            for d in drops:
                assert d in dmap and np.isfinite(dmap[d])

    for axis in STRESS_AXES:
        tp = UNSEEN_LAYOUT_NAMES[0] if axis in ("standard", "flow_inversion") else UNSEEN_LAYOUT_NAMES[1]
        a1 = build_v10_stress_scenario(tp, seed=7001, stress_axis=axis)
        a2 = build_v10_stress_scenario(tp, seed=7001, stress_axis=axis)
        assert [t.task_id for t in a1[1]] == [t.task_id for t in a2[1]]
        assert len(a1[3]) == 8
        if axis == "multi_kitting":
            assert any(len(op.kit_task_ids) >= 3 for op in a1[3])
        if axis == "converging_dag":
            assert any(op.predecessor_op_id is not None for op in a1[3])
        if axis == "dual_fault":
            assert a1[4]["post_pickup_fault_step"] - a1[4]["fault_step"] <= 15

    # Run short closed-loop episode in AMRReschedulingEnv on unseen layout
    env = AMRReschedulingEnv(
        topology="unseen_ladder",
        seed=7001,
        max_steps=60,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        real_compute_window_coupling=True,
        expert_v9_portfolio=True,
        solver_override="V10_E7_COND_CPSAT",
        record_trajectory=True,
    )
    obs, info = env.reset(seed=7001, options={"topology": "unseen_ladder", "stress_axis": "multi_kitting"})
    snap0 = NativeSnapshotCapture.capture(env.sim, env.tracker.operations)
    feat0 = extract_opening_structural_features(env.sim.network, snap0)
    assert feat0.shape == (OPENING_FEATURE_DIM,)
    assert np.all(np.isfinite(feat0))

    obs, _, term, trunc, info = env.step(8 if info["action_mask"][8] == 1 else ACTION_KEEP)
    for _ in range(15):
        obs, _, term, trunc, info = env.step(ACTION_KEEP)
        if term or trunc:
            break
    aud = IndependentAuditor().audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
    assert aud.pass_audit, aud.issues


def test_spga_cor2026_relaxed_and_g1_unified():
    """Verify Jia Li (COR 2026) SPGA 3-layer encoding, Eq.12 similarity, and CVCR transaction commit."""
    env = AMRReschedulingEnv(
        topology="grid",
        seed=2001,
        max_steps=60,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        real_compute_window_coupling=True,
        record_trajectory=True,
    )
    env.reset(seed=2001, options={"topology": "grid"})
    for _ in range(5):
        env.step(ACTION_KEEP)

    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
    prob = SPGAProblem(snap, env.sim.network)
    solver = SPGASolver(prob, SPGAConfig(pop_size=16, max_iters=6, mode="relaxed", budget_ms=80.0))
    vecs = np.stack([solver._to_vec(solver._random_chromosome()) for _ in range(6)], axis=0)
    sim_m = solver._cosine_sim_matrix(vecs)
    assert sim_m.shape == (6, 6)
    assert np.allclose(sim_m, sim_m.T, atol=1e-6)
    assert np.all(np.isfinite(sim_m))

    p_rel = run_spga_policy_proposal(snap, env.sim.network, SPGAConfig(pop_size=18, max_iters=8, mode="relaxed", budget_ms=80.0))
    p_g1 = run_spga_policy_proposal(snap, env.sim.network, SPGAConfig(pop_size=18, max_iters=8, mode="g1_unified", budget_ms=100.0))
    assert p_rel.proposed_op_priorities is not None
    assert p_g1.proposed_op_priorities is not None
    assert env.owner.validate_and_commit(p_g1).accepted is True


def test_lrho_predictor_features_and_safety_guard(tmp_path: Path):
    """Verify L-RHO 19-D/6-D features, NumPy checkpoint loading, and never-freeze safety guard."""
    env = AMRReschedulingEnv(
        topology="workshop",
        seed=2001,
        max_steps=80,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        real_compute_window_coupling=True,
    )
    env.reset(seed=2001, options={"topology": "workshop"})
    for _ in range(18):
        env.step(ACTION_KEEP)

    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
    clf = classify_commitments(snap)
    ev = CVCREvaluator(env.sim.network, snap, use_rust=True)
    base = LGCVCRPolicy("V10_E6_BALANCE_KM_LRHO", env.sim.network)._base_assignment(
        snap,
        [r.robot_id for r in snap.robots if r.state != "faulted"],
        {op.op_id: op.due_step for op in snap.operations},
    )
    res = ev.evaluate(base)
    bays = {op.op_id: op.bay_id for op in snap.operations}
    crews = {op.op_id: op.crew_id for op in snap.operations}
    tids, tx, ax, a_idx = extract_lrho_features(snap, clf, res, bays, crews, network=env.sim.network)
    if tids:
        assert tx.shape == (len(tids), LRHO_TASK_FEAT_DIM)
        assert ax.shape[1] == LRHO_AGV_FEAT_DIM
        assert np.all(np.isfinite(tx)) and np.all(np.isfinite(ax))

    # Build trivial weights that output huge positive logits (p -> ~1.0) and verify safety guards
    npz_path = tmp_path / "dummy_lrho.npz"
    np.savez(
        str(npz_path),
        t_mu=np.zeros((1, LRHO_TASK_FEAT_DIM), dtype=np.float32),
        t_sd=np.ones((1, LRHO_TASK_FEAT_DIM), dtype=np.float32),
        a_mu=np.zeros((1, LRHO_AGV_FEAT_DIM), dtype=np.float32),
        a_sd=np.ones((1, LRHO_AGV_FEAT_DIM), dtype=np.float32),
        t_w1=np.zeros((LRHO_TASK_FEAT_DIM, 8), dtype=np.float32),
        t_b1=np.zeros((8,), dtype=np.float32),
        t_w2=np.zeros((8, 8), dtype=np.float32),
        t_b2=np.zeros((8,), dtype=np.float32),
        a_w1=np.zeros((LRHO_AGV_FEAT_DIM, 8), dtype=np.float32),
        a_b1=np.zeros((8,), dtype=np.float32),
        a_w2=np.zeros((8, 8), dtype=np.float32),
        a_b2=np.zeros((8,), dtype=np.float32),
        o_w1=np.zeros((24, 8), dtype=np.float32),
        o_b1=np.ones((8,), dtype=np.float32),
        o_w2=np.ones((8, 1), dtype=np.float32),
        o_b2=np.asarray([2.0], dtype=np.float32),
        threshold=np.asarray([0.5], dtype=np.float32),
    )
    pred = LRHOPredictor.from_npz(npz_path, network=env.sim.network)
    frozen = pred.predict_frozen_tasks(snap, clf, res, bays, crews)
    snap_by_id = {t.task_id: t for t in snap.tasks}
    for tid in frozen:
        t = snap_by_id[tid]
        assert t.assigned_agv is not None and t.assigned_agv not in clf.disturbed_robots
        assert not t.is_urgent

    pol = LGCVCRPolicy("V10_E6_BALANCE_KM_LRHO", env.sim.network, lrho_predictor=pred)
    prop = pol.propose(snap, budget_ms=100.0)
    assert prop.proposed_op_priorities is not None


def test_v10_portfolio_rcdr_and_causal_opening_selectors():
    """Verify V10 action space, V10 RCDR trigger, RuleV10, and shallow causal opening tree/contextual selectors."""
    from amr_sched.rl import (
        EXPERT_V10_ACTION_NAMES,
        NUM_EXPERT_V10_ACTIONS,
        CausalOpeningContextualSelector,
        CausalOpeningTreeSelector,
        HistoricalTopoSelector,
        RuleV10PortfolioPolicy,
        should_query_v10_controller,
    )

    assert len(EXPERT_V10_ACTION_NAMES) == NUM_EXPERT_V10_ACTIONS == 9
    env = AMRReschedulingEnv(
        topology="grid",
        seed=2001,
        max_steps=60,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        real_compute_window_coupling=True,
        expert_v10_portfolio=True,
        record_trajectory=True,
    )
    obs, info = env.reset(seed=2001, options={"topology": "grid"})
    assert env.action_space.n == 9
    rule = RuleV10PortfolioPolicy()
    rule.reset()
    for _ in range(25):
        o9 = env.get_expert_v9_obs(obs)
        act = rule.select_action(o9, info, sim=env.sim)
        obs, _, term, trunc, info = env.step(act)
        if term or trunc:
            break
    # Tail progress >= 0.80 with zero late ops must suppress calm queries
    fake_obs = np.zeros((38,), dtype=np.float32)
    fake_obs[11] = 0.85
    q_tail, _ = should_query_v10_controller(fake_obs, info["action_mask"], tick=200, last_decision_tick=150)
    assert q_tail is False
    aud = IndependentAuditor().audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
    assert aud.pass_audit, aud.issues

    # Verify cost-sensitive CART splits & contextual selector predictions
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 16))
    X[:20, 3] = 0.1
    X[20:, 3] = 0.9
    Y = np.zeros((40, 2), dtype=np.float64)
    Y[:20, 0], Y[:20, 1] = 100.0, 200.0
    Y[20:, 0], Y[20:, 1] = 220.0, 90.0
    tree = CausalOpeningTreeSelector(["rl_v9", "rl_v7"], max_depth=2, min_samples_leaf=5).fit(X, Y)
    assert tree.predict(X[0]) == "rl_v9"
    assert tree.predict(X[25]) == "rl_v7"
    assert tree.describe_rules()
    ctx = CausalOpeningContextualSelector(["rl_v9", "rl_v7"], reg_lam=1.0).fit(X, Y)
    assert ctx.predict(X[0]) == "rl_v9"
    assert ctx.predict(X[25]) == "rl_v7"
    hist = HistoricalTopoSelector(["rl_v9", "rl_v7"], {"ring": "rl_v9"}, fallback="rl_v7")
    assert hist.predict("ring") == "rl_v9"
    assert hist.predict("unseen_ladder") == "rl_v7"
