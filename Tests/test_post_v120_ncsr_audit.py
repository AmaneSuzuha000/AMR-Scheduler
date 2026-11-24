"""Post-v1.2.0 regression & scientific audit tests for CoRe-Sched (NCSR)."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from amr_sched.baselines.policy_registry import _V19_RL_CKPT, _V19_STAGED_RL_CKPT, build_external_proposer
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.cvcr.transaction import NativeTransactionOwner
from amr_sched.cvcr.v12_proactive_guard import plan_haven_safe_corridor_repair
from amr_sched.rl import AMRReschedulingEnv, RuleV10PortfolioPolicy, compute_expert_v10_action_mask, compute_expert_v9_state
from amr_sched.rl.causal_repair_cone import (
    FastJAWKittingSurrogate,
    compute_candidate_ja_components,
    observable_op_closure,
    score_candidate_ja_unified,
)
from amr_sched.rl.v19_hetero_constructor import (
    NumpyV19HeteroConstructorNet,
    V19AdaptiveCouplingProposer,
    decode_v19_candidate,
    graph_to_torch_tensors,
)
from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
from experiments.run_g3_fresh_frozen_holdout import FRESH_4TOPO_SEEDS, FRESH_WORKSHOP_SEEDS
from experiments.train_v19_bc import (
    FORBIDDEN_BLIND_SEEDS,
    FORBIDDEN_DEV_SEEDS,
    TRAIN_4TOPO_SEEDS,
    TRAIN_WORKSHOP_SEEDS,
    VAL_4TOPO_SEEDS,
    VAL_WORKSHOP_SEEDS,
    _build_4topo_episode_spec,
    _build_workshop_episode_spec,
)
from experiments.train_v19_rl import RL_4TOPO_SEEDS, RL_WORKSHOP_SEEDS


def _step_to_snapshot(spec: dict, *, steps: int = 20):
    sc_tuple = copy.deepcopy(spec["scenario_tuple"])
    env = AMRReschedulingEnv(
        seed=int(spec["seed"]),
        max_steps=int(spec["max_steps"]),
        use_rust=True,
        execution_perturbation=copy.deepcopy(spec["p2_cfg"]),
        execution_aware=True,
        p4_choke_robust=True,
    )
    env.reset(seed=int(spec["seed"]), options={"scenario_tuple": sc_tuple})
    ctrl = RuleV10PortfolioPolicy(epoch_interval=16)
    ctrl.reset()
    for _ in range(steps):
        obs = compute_expert_v9_state(
            env.sim,
            env.tracker,
            last_replan_tick=env.last_replan_tick,
            recent_conflict_ticks=env.recent_conflict_ticks,
            decision_logs=env.decision_logs,
        )
        msk = compute_expert_v10_action_mask(env.sim, env.last_replan_tick)
        act = ctrl.select_action(obs, {"action_mask": msk, "step": env.current_step}, sim=env.sim)
        _, _, tm, tr, _ = env.step(act)
        if tm or tr:
            break
    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
    return env, snap, env.sim.network


def test_ja_components_and_unified_scorer_include_delivery_and_exclude_phantoms() -> None:
    """Verify 0.5*delivery term, observable_op_closure, and compact-grid vs workshop regimes."""
    sp_4t = _build_4topo_episode_spec("offset_dual_lane", 16702, "window_bottleneck")
    _, snap_0, net_4t = _step_to_snapshot(sp_4t, steps=0)
    assert len(net_4t.nodes) < 30
    assert observable_op_closure(snap_0) == set()  # Unlaunched wave ops wait until all kits release

    _, snap_4t, net_4t = _step_to_snapshot(sp_4t, steps=10)
    clf_4t = classify_commitments(snap_4t)
    g_4t = extract_v19_hetero_graph(snap_4t, net_4t, clf=clf_4t)
    ev_4t = CVCREvaluator(net_4t, snap_4t, use_rust=True)
    op_prio = {op.op_id: int(op.due_step) - int(op.proc_steps) for op in snap_4t.operations}
    res_4t = ev_4t.evaluate(g_4t.base_assign, op_priority=op_prio, op_bay_overrides=g_4t.base_bays, op_crew_overrides=g_4t.base_crews)

    rob_h = {r.robot_id: r.cur_task for r in snap_4t.robots if r.cur_task and r.robot_id not in clf_4t.disturbed_robots}
    comp = compute_candidate_ja_components(
        snap_4t,
        g_4t,
        res_4t,
        g_4t.base_assign,
        g_4t.base_bays,
        g_4t.base_crews,
        None,
        rob_h,
        sim_horizon=ev_4t.sim_horizon,
    )
    expected_deliv = 0.5 * sum(
        min(ev_4t.sim_horizon, int(res_4t.order_unload_ticks.get(t.task_id, ev_4t.sim_horizon)))
        for t in snap_4t.tasks
        if t.status.lower() != "completed"
    )
    assert comp["deliv_cost"] == pytest.approx(expected_deliv)

    obs_ops = observable_op_closure(snap_4t)
    assert obs_ops  # Launched ops with visible kits are in observable closure
    assert len(obs_ops) <= len(snap_4t.operations)
    fast_surr = FastJAWKittingSurrogate(net_4t, snap_4t, ev_4t, g_4t)
    sc_u, _ = score_candidate_ja_unified(
        snap_4t,
        net_4t,
        g_4t,
        res_4t,
        g_4t.base_assign,
        g_4t.base_bays,
        g_4t.base_crews,
        obs_ops,
        fast_surr,
        sim_horizon=ev_4t.sim_horizon,
    )
    assert np.isfinite(sc_u) and sc_u > 0.0

    # FastJAWKittingSurrogate.regret_k_assignment assigns every requested healthy task legally
    assigned = fast_surr.regret_k_assignment(
        [g_4t.agv_ids[i] for i in g_4t.healthy_agv_indices],
        {},
        list(g_4t.rel_tids),
        g_4t.base_bays,
        op_prio,
    )
    assert set(assigned.keys()) == set(g_4t.rel_tids)


def test_numpy_vs_torch_v19_constructor_parity_and_prebuild() -> None:
    """Default V19_ADAPTIVE_RL proposer pre-builds NumPy inference with < 1e-5 logit parity vs PyTorch."""
    prop = build_external_proposer("V19_ADAPTIVE_RL", seed=0, use_rust=True)
    assert isinstance(prop, V19AdaptiveCouplingProposer)
    assert prop._effective_backend() == "numpy"

    sp_ws = _build_workshop_episode_spec("track1", 1161, "kitting_windows")
    _, snap, net = _step_to_snapshot(sp_ws, steps=20)
    g = extract_v19_hetero_graph(snap, net)
    f_torch, _ = prop._ensure_nets()
    f_np, _ = prop._ensure_infer_nets()
    assert isinstance(f_np, NumpyV19HeteroConstructorNet)

    with torch.no_grad():
        ten = graph_to_torch_tensors(g)
        e_t = f_torch.encode_entities(
            ten["op_x"],
            ten["task_x"],
            ten["agv_x"],
            ten["bay_x"],
            ten["crew_x"],
            ten["task_to_op"],
            ten["op_dag_adj"],
        )
        e_n = f_np.encode_entities(g)
        assert float(np.max(np.abs(e_t["h_op"].squeeze(0).cpu().numpy() - e_n["h_op"][0]))) < 1e-5

        c_t = decode_v19_candidate(f_torch, g, snap, greedy=True, np_rng=np.random.default_rng(0))
        c_n = decode_v19_candidate(f_np, g, snap, greedy=True, np_rng=np.random.default_rng(0))
        assert c_t.assign == c_n.assign
        assert c_t.bays == c_n.bays


def test_window_sla_budget_guard_and_wide_ws_ncsr_pipeline() -> None:
    """Resource-windowed workshops retain normal 29ms budget; tight budgets bound search & NCSR activates both stages."""
    sp_t1 = _build_workshop_episode_spec("track1", 1161, "kitting_windows")
    _, snap_t1, net_t1 = _step_to_snapshot(sp_t1, steps=25)
    assert len(net_t1.nodes) >= 30
    assert any(bool(getattr(op, "resource_windows", ())) for op in snap_t1.operations)

    prop = build_external_proposer("V19_ADAPTIVE_RL", seed=0, use_rust=True)
    _ = prop(snap_t1, net_t1, budget_ms=50.0)  # Warmup Rust/NumPy paths outside timing assertion
    res_norm = prop(snap_t1, net_t1, budget_ms=50.0)
    assert res_norm.telemetry["window_sensitive"] is False
    assert float(res_norm.telemetry["eff_budget_ms"]) == pytest.approx(29.0)
    assert int(res_norm.telemetry["surrogate_pool_size"]) >= 2
    assert int(res_norm.telemetry["cone_candidates_generated"]) >= 1
    assert res_norm.proposed_task_sequences is None  # Multi-zone WS avoids queue over-pinning

    # Tight RuleV10 action-1 budget (20ms) is strictly respected as an upper cap
    res_tight = prop(snap_t1, net_t1, budget_ms=20.0)
    assert float(res_tight.telemetry["eff_budget_ms"]) <= 20.0

    # Compact unwindowed 4-Topo grid executes both stages and avoids tail queue over-pinning
    sp_4t = _build_4topo_episode_spec("offset_dual_lane", 16702, "standard")
    _, snap_4t, net_4t = _step_to_snapshot(sp_4t, steps=19)
    assert len(net_4t.nodes) < 30
    res_4t = prop(snap_4t, net_4t, budget_ms=35.0)
    assert res_4t.predicted_obj > 0.0
    assert res_4t.telemetry["inference_backend"] == "numpy"
    assert res_4t.proposed_task_sequences is None


def test_haven_live_repair_excludes_active_loading_agvs() -> None:
    """`plan_haven_safe_corridor_repair` must not emit `picked_reroute` for AGVs with active `_load_end`."""
    sp_4t = _build_4topo_episode_spec("offset_dual_lane", 16701, "execution_blockage")
    env, _, _ = _step_to_snapshot(sp_4t, steps=40)
    sim = env.sim
    owner = NativeTransactionOwner(sim, env.tracker)
    checked = False
    for _ in range(40):
        obs = compute_expert_v9_state(
            sim,
            env.tracker,
            last_replan_tick=env.last_replan_tick,
            recent_conflict_ticks=env.recent_conflict_ticks,
            decision_logs=env.decision_logs,
        )
        msk = compute_expert_v10_action_mask(sim, env.last_replan_tick)
        act = 1 if len(msk) > 1 and msk[1] == 1 else 0
        _, _, tm, tr, _ = env.step(act)
        step = int(getattr(sim, "_step_cursor", 0))
        load_end = getattr(sim, "_load_end", {})
        active_loaders = {aid for aid, end_t in load_end.items() if int(end_t) > step}
        if active_loaders:
            snap = NativeSnapshotCapture.capture(sim, env.tracker.operations, hide_future_releases=True)
            reps, _ = plan_haven_safe_corridor_repair(sim, snap, stall_threshold=4)
            for aid, rep in reps.items():
                if rep.get("kind") == "picked_reroute":
                    assert aid not in active_loaders
                    checked = True
        if tm or tr:
            break
    # Even if synthetic loading state wasn't encountered in first 80 ticks, directly test active _load_end guard
    if not checked and sim.agvs:
        from amr_sched.entities import TaskStatus

        for t in sim.tasks:
            if t.status == TaskStatus.PICKED and t.assigned_agv:
                aid = t.assigned_agv
                if aid not in getattr(sim, "_transit", {}):
                    step = int(getattr(sim, "_step_cursor", 0))
                    sim._load_end[aid] = step + 5
                    snap = NativeSnapshotCapture.capture(sim, env.tracker.operations, hide_future_releases=True)
                    reps, _ = plan_haven_safe_corridor_repair(sim, snap, stall_threshold=0)
                    assert not (aid in reps and reps[aid].get("kind") == "picked_reroute")
                    break
    assert owner is not None


def test_dual_holdout_seed_quarantine() -> None:
    """Train, VAL, G3 Cohort A, and Pristine Blind Cohort B seeds must be strictly pairwise disjoint."""
    cohort_b_4t = {37101, 37102, 37201, 37202, 37301, 37302}
    cohort_b_ws = {3901, 3902, 3903, 3904, 3905, 3906}
    cohort_b = cohort_b_4t | cohort_b_ws
    prior_all = (
        set(FORBIDDEN_BLIND_SEEDS)
        | set(FORBIDDEN_DEV_SEEDS)
        | set(TRAIN_4TOPO_SEEDS)
        | set(TRAIN_WORKSHOP_SEEDS)
        | set(VAL_4TOPO_SEEDS)
        | set(VAL_WORKSHOP_SEEDS)
        | set(RL_4TOPO_SEEDS)
        | set(RL_WORKSHOP_SEEDS)
        | set(FRESH_4TOPO_SEEDS)
        | set(FRESH_WORKSHOP_SEEDS)
    )
    assert len(cohort_b) == 12
    assert not (cohort_b & prior_all)


def test_traffic_weight_edges_validated_and_rulev10_choke_wakeup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proposed traffic weights only reference existing directed network edges; RuleV10 wakes on choke oscillation."""
    sp_t1 = _build_workshop_episode_spec("track1", 2902, "multi_wave_surge")
    env, _, net_t1 = _step_to_snapshot(sp_t1, steps=40)
    p2_fb = env.execution_perturbation.capture_feedback(env.sim, env.current_step) if env.execution_perturbation else None
    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True, execution_feedback=p2_fb)
    prop = build_external_proposer("V19_ADAPTIVE_RL", seed=0, use_rust=True)
    res = prop(snap, net_t1, budget_ms=35.0)
    if res.proposed_traffic_weights:
        for u, v in res.proposed_traffic_weights:
            assert net_t1.has_edge(u, v)

    # RuleV10 surgical choke oscillation wakeup (only when _p4_choke_robust=True & calm would return KEEP)
    sp_t2 = _build_workshop_episode_spec("track2", 1162, "kitting_windows")
    env2, _, _ = _step_to_snapshot(sp_t2, steps=15)
    ctrl = RuleV10PortfolioPolicy(epoch_interval=16)
    ctrl.last_dec = env2.current_step  # force calm cooldown -> ACTION_KEEP unless surgical wakeup triggers
    obs = compute_expert_v9_state(
        env2.sim,
        env2.tracker,
        last_replan_tick=env2.last_replan_tick,
        recent_conflict_ticks=env2.recent_conflict_ticks,
        decision_logs=env2.decision_logs,
    )
    msk = compute_expert_v10_action_mask(env2.sim, env2.last_replan_tick)
    if msk[1] == 1:
        monkeypatch.setattr(
            "amr_sched.cvcr.p4_belief_choke.detect_choke_oscillation_agvs",
            lambda _sim: ["AGV_1"],
        )
        env2.sim._p4_choke_robust = False
        assert ctrl.select_action(obs, {"action_mask": msk, "step": env2.current_step}, sim=env2.sim) == 0
        env2.sim._p4_choke_robust = True
        assert ctrl.select_action(obs, {"action_mask": msk, "step": env2.current_step}, sim=env2.sim) == 1

