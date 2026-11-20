"""Unit & integration tests for V16 Unified Frontier, pure-NumPy restraint, causal router, and regressions."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.p4_belief_choke import (
    plan_choke_passage_repairs,
    predict_and_arbitrate_choke_conflicts,
)
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.cvcr.production import NativeProductionTracker, OperationSnapshot
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.cvcr.transaction import NativeTransactionOwner
from amr_sched.cvcr.v16_frontier import (
    DEFAULT_V16_CKPT,
    V16_CTX_DIM,
    V16AdaptiveController,
    V16FrontierPolicy,
    extract_v16_context,
    load_v16_frontier_policy,
)
from amr_sched.entities import Agv, Task, TaskStatus
from amr_sched.network import Edge, Network, Node, NodeKind
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.reservation import ReservationTable
from amr_sched.scenarios_v10 import build_unseen_choke_yard_network
from amr_sched.simulation import Hop, SimulationConfig, Simulator, _Job


def _synth_v16_weights(seed: int = 17) -> dict[str, np.ndarray]:
    """Create deterministic synthetic pure-NumPy V7 dueling tensors matching `TriRegimeAnytimeNet`."""
    rng = np.random.default_rng(seed)
    weights: dict[str, np.ndarray] = {}
    for prefix in ("v7.opt_net", "v7.urgent_net", "v7.fault_net"):
        weights[f"{prefix}.backbone.0.weight"] = rng.normal(0.0, 0.1, size=(16, 30)).astype(np.float32)
        weights[f"{prefix}.backbone.0.bias"] = np.zeros(16, dtype=np.float32)
        weights[f"{prefix}.backbone.1.weight"] = np.ones(16, dtype=np.float32)
        weights[f"{prefix}.backbone.1.bias"] = np.zeros(16, dtype=np.float32)
        weights[f"{prefix}.backbone.3.weight"] = rng.normal(0.0, 0.1, size=(16, 16)).astype(np.float32)
        weights[f"{prefix}.backbone.3.bias"] = np.zeros(16, dtype=np.float32)
        weights[f"{prefix}.backbone.4.weight"] = np.ones(16, dtype=np.float32)
        weights[f"{prefix}.backbone.4.bias"] = np.zeros(16, dtype=np.float32)
        weights[f"{prefix}.val_head.weight"] = rng.normal(0.0, 0.1, size=(1, 16)).astype(np.float32)
        weights[f"{prefix}.val_head.bias"] = np.zeros(1, dtype=np.float32)
        weights[f"{prefix}.adv_head.weight"] = rng.normal(0.0, 0.1, size=(9, 16)).astype(np.float32)
        weights[f"{prefix}.adv_head.bias"] = np.zeros(9, dtype=np.float32)
    return weights


def test_v16_pure_numpy_restraint_and_cached_npz(tmp_path: Path) -> None:
    """Verify pure-NumPy `eval_v7_restraint` is finite/deterministic and `.npz` caching works cleanly."""
    ckpt = tmp_path / "synth_v16_frontier.npz"
    np.savez(ckpt, **_synth_v16_weights())

    pol_a = load_v16_frontier_policy(ckpt)
    pol_b = load_v16_frontier_policy(ckpt)
    assert pol_a is not pol_b
    assert pol_a.has_v7_restraint_weights()
    assert pol_a.w is pol_b.w  # cached immutable array mapping by (path, mtime_ns)

    mask = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8)
    for idx_flag in (None, 18, 19):
        obs = np.linspace(-0.5, 0.5, 30, dtype=np.float32)
        obs[18] = 0.0
        obs[19] = 0.0
        if idx_flag is not None:
            obs[idx_flag] = 1.0
        act1, q_act1, q_keep1 = pol_a.eval_v7_restraint(obs, mask)
        act2, q_act2, q_keep2 = pol_b.eval_v7_restraint(obs, mask)
        assert act1 == act2
        assert mask[act1] == 1
        assert np.isfinite(q_act1) and np.isfinite(q_keep1)
        assert abs(q_act1 - q_act2) < 1e-12

    if DEFAULT_V16_CKPT.exists():
        real_pol = load_v16_frontier_policy(DEFAULT_V16_CKPT)
        assert real_pol.has_v7_restraint_weights()
        obs_real = np.zeros(30, dtype=np.float32)
        act_r, q_r, qk_r = real_pol.eval_v7_restraint(obs_real, mask)
        assert mask[act_r] == 1
        assert np.isfinite(q_r) and np.isfinite(qk_r)


def test_v16_causal_portfolio_mode_dispatch() -> None:
    """Verify topology-free causal mode selection across reservation, calm, balanced, and quality priors."""
    pol = V16FrontierPolicy(weights=None)
    ctx_nom = np.zeros(V16_CTX_DIM, dtype=np.float32)

    # Active resource window protection routes directly to V15_SOFT regardless of profile
    ctx_win = ctx_nom.copy()
    ctx_win[0] = 1.0
    assert pol.select_portfolio_mode(ctx_win, profile="quality") == "V15_SOFT"
    assert pol.select_portfolio_mode(ctx_win, profile="calm") == "V15_SOFT"
    assert pol.select_portfolio_mode(ctx_win, profile="balanced") == "V15_SOFT"

    # Explicit profiles
    assert pol.select_portfolio_mode(ctx_nom, profile="balanced") == "V15_SOFT"
    assert pol.select_portfolio_mode(ctx_nom, profile="calm") == "V7_CALM"

    # Quality fallback prior: low-precedence calm layout -> V10_GUIDED; drifted/high-risk -> V7_CALM; DAG -> V15_SOFT
    assert pol.select_portfolio_mode(ctx_nom, profile="quality") == "V10_GUIDED"
    ctx_drift = ctx_nom.copy()
    ctx_drift[1] = 0.5
    ctx_drift[4] = 1.0
    assert pol.select_portfolio_mode(ctx_drift, profile="quality") == "V7_CALM"
    ctx_dag = ctx_nom.copy()
    ctx_dag[1] = 0.5
    assert pol.select_portfolio_mode(ctx_dag, profile="quality") == "V15_SOFT"


def test_v16_transaction_clears_stale_queue_and_guide_corridors() -> None:
    """Verify committing V16_* with empty queues/guides clears zombie locks and pending non-urgent pins."""
    net = Network()
    net.add_node(Node("N0", 0.0, 0.0, kind=NodeKind.BUFFER, capacity=2))
    net.add_node(Node("N1", 2.0, 0.0, kind=NodeKind.WORKSTATION, capacity=2))
    net.add_edge(Edge("N0", "N1", length=2.0, speed_limit=1.0, capacity=2))
    net.add_edge(Edge("N1", "N0", length=2.0, speed_limit=1.0, capacity=2))

    tasks = [Task("T1", "N0", "N1", release_step=0, due_step=100)]
    ops = [OperationSnapshot("OP1", ("T1",), "N1", crew_id=1, tool_id=1, proc_steps=1, due_step=120)]
    sim = Simulator(
        net,
        tasks,
        [Agv("agv_0", max_speed=1.0, start_node="N0")],
        SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=120, reschedule_policy=ReschedulePolicy.NONE),
    )
    sim.reset()
    sim.set_task_queues({"agv_0": ["T1"]})
    sim._pinned_task_agv["T1"] = "agv_0"
    sim.set_guide_corridors({"agv_0": [("N0", "N1")]})
    assert sim._pinned_task_queues == {"agv_0": ["T1"]}
    assert sim._guide_corridors == {"agv_0": [("N0", "N1")]}

    owner = NativeTransactionOwner(sim, operations=ops)
    snap = NativeSnapshotCapture.capture(sim, ops, hide_future_releases=True)
    prop = CandidateProposal(
        schema="amr-lgcvcr.proposal.v1",
        policy_id="V16_BALANCED",
        snapshot_digest=snap.compute_digest(),
        affected_robots=["agv_0"],
        affected_tasks=[],
        affected_operations=[],
        proposed_assignment={},
        proposed_routes={},
        proposed_bays={},
        proposed_crews={},
        predicted_obj=0.0,
        predicted_makespan=50,
        candidate_digest="v16_clear",
        telemetry={},
        proposed_task_sequences=None,
        proposed_guide_routes={},
    )
    rcpt = owner.validate_and_commit(prop)
    assert rcpt.accepted, f"Expected accepted V16_ proposal, got {rcpt.reasons}"
    assert sim._pinned_task_queues == {}
    assert "T1" not in sim._pinned_task_agv
    assert sim._guide_corridors == {}


def test_p45_choke_regressions_missing_kit_op_id_and_inverted_lock_interval() -> None:
    """Regression guard: tasks without `.kit_op_id` and slow retreats (`t_hv > t_bridge_clear`) never raise."""
    # 1. Missing `.kit_op_id` attribute on Task inside predict_and_arbitrate_choke_conflicts
    net = build_unseen_choke_yard_network(aisle_capacity=1, speed=3.0)
    t_urg = Task("T_U", "WA_S2", "W4", release_step=0, due_step=120)
    t_lax = Task("T_L", "EA_S0", "W3", release_step=0, due_step=400)
    assert not hasattr(t_urg, "kit_op_id")
    sim = Simulator(
        net,
        [t_urg, t_lax],
        [Agv("agv_u", max_speed=3.0, start_node="WA_S0"), Agv("agv_l", max_speed=3.0, start_node="EA_S0")],
        SpaceTimeAStarPlanner(max_speed=3.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=300),
    )
    sim.reset()
    t_urg.status = TaskStatus.PICKED
    t_urg.assigned_agv = "agv_u"
    t_lax.status = TaskStatus.PICKED
    t_lax.assigned_agv = "agv_l"
    sim.agvs[0].cur_node = "WA_S0"
    sim.agvs[0].cur_task = "T_U"
    sim._jobs["agv_u"] = _Job(
        task_id="T_U",
        leg="to_dropoff",
        hops=[Hop("WA_S0", "CH_W", 10, 12), Hop("CH_W", "CH_M", 12, 15)],
        idx=0,
        goal="W4",
    )
    sim.agvs[1].cur_node = "EA_S0"
    sim.agvs[1].cur_task = "T_L"
    sim._jobs["agv_l"] = _Job(
        task_id="T_L",
        leg="to_dropoff",
        hops=[Hop("EA_S0", "CH_E", 10, 12), Hop("CH_E", "CH_M", 12, 15)],
        idx=0,
        goal="W3",
    )
    predict_and_arbitrate_choke_conflicts(sim, None, ())

    # 2. Inverted lock interval (`t_hv > t_bridge_clear`) inside plan_choke_passage_repairs
    c_net = Network()
    c_net.add_node(Node("H_L", 0.0, 0.0, kind=NodeKind.BUFFER, capacity=2))
    c_net.add_node(Node("C0", 4.0, 0.0, kind=NodeKind.AISLE, capacity=1))
    c_net.add_node(Node("C1", 5.0, 0.0, kind=NodeKind.AISLE, capacity=1))
    c_net.add_node(Node("C2", 6.0, 0.0, kind=NodeKind.AISLE, capacity=1))
    c_net.add_node(Node("H_R", 8.0, 0.0, kind=NodeKind.WORKSTATION, capacity=2))
    for u, v, d in (("H_L", "C0", 4.0), ("C0", "C1", 1.0), ("C1", "C2", 1.0), ("C2", "H_R", 2.0)):
        c_net.add_edge(Edge(u, v, length=d, speed_limit=0.25, capacity=1))
        c_net.add_edge(Edge(v, u, length=d, speed_limit=0.25, capacity=1))

    tl = Task("T_LEAD", "H_L", "H_R", release_step=0, due_step=200)
    ty = Task("T_YIELD", "H_R", "H_L", release_step=0, due_step=400)
    c_sim = Simulator(
        c_net,
        [tl, ty],
        [Agv("agv_lead", max_speed=0.25, start_node="C0"), Agv("agv_yield", max_speed=0.25, start_node="C1")],
        SpaceTimeAStarPlanner(max_speed=0.25, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=300),
    )
    c_sim.reset()
    tl.status = TaskStatus.PICKED
    tl.assigned_agv = "agv_lead"
    ty.status = TaskStatus.PICKED
    ty.assigned_agv = "agv_yield"
    c_sim.agvs[0].cur_node = "C0"
    c_sim.agvs[0].cur_task = "T_LEAD"
    c_sim._jobs["agv_lead"] = _Job(
        task_id="T_LEAD",
        leg="to_dropoff",
        hops=[Hop("C0", "C1", 0, 4), Hop("C1", "C2", 4, 8), Hop("C2", "H_R", 8, 16)],
        idx=0,
        goal="H_R",
    )
    c_sim.agvs[1].cur_node = "C1"
    c_sim.agvs[1].cur_task = "T_YIELD"
    c_sim._jobs["agv_yield"] = _Job(
        task_id="T_YIELD",
        leg="to_dropoff",
        hops=[Hop("C1", "C0", 0, 4), Hop("C0", "H_L", 4, 20)],
        idx=0,
        goal="H_L",
    )
    c_snap = NativeSnapshotCapture.capture(c_sim, (), hide_future_releases=True)
    # Slow retreat from C1 -> H_L takes ~20s > t_bridge_clear (step+12); previously raised ValueError
    repairs, _stats = plan_choke_passage_repairs(
        c_sim, c_snap, ["agv_lead", "agv_yield"], ReservationTable(), [], {}
    )
    assert isinstance(repairs, dict)


def _run_v16_closed_loop_episode(mode: str, profile: str) -> dict[str, float]:
    """Run closed-loop `V16_*` episode on standard `ring` with virtual clock and strict audit."""
    pytest.importorskip("gymnasium")
    from amr_sched.rl.env import AMRReschedulingEnv

    with virtual_clock_scope(poll_ms=0.002):
        env = AMRReschedulingEnv(
            topology="ring",
            seed=7101,
            allow_kitting_shortcut=False,
            pure_physical_reward=True,
            real_compute_window_coupling=True,
            expert_v12_portfolio=True,
            loaded_repair_mode="haven_v12",
            loaded_repair_threshold=8,
            record_trajectory=True,
            execution_aware=True,
            p4_choke_robust=True,
            v13_kitting_mode=mode,
        )
        obs, info = env.reset(seed=7101, options={"topology": "ring", "stress_axis": "standard"})
        pol = V16AdaptiveController(epoch_interval=16, profile=profile)
        pol.reset()
        done = False
        while not done:
            act = pol.select_action(env.get_expert_v12_obs(obs), info, sim=env.sim)
            obs, _, tm, tr, info = env.step(act)
            done = tm or tr
        metrics = env.compute_episode_metrics()
        aud = IndependentAuditor.audit_simulation_run(
            env.sim, env.tracker.operations, require_trajectory=True
        )
        assert aud.pass_audit, f"{mode} failed IndependentAuditor: {aud.issues}"
        assert metrics["task_completion_rate"] == 1.0
        assert metrics["production_completion_rate"] == 1.0
        return {
            "net_obj": float(metrics["net_obj"]),
            "cmax": float(metrics["cmax"]),
            "replans": float(metrics.get("replans", 0.0)),
        }


def test_v16_closed_loop_profiles_complete_and_pass_audit() -> None:
    """Verify `V16_EA_JOINT`, `V16_BALANCED`, and `V16_CALM_FAST` complete 100% and pass audit."""
    m_quality = _run_v16_closed_loop_episode("V16_EA_JOINT", "quality")
    m_balanced = _run_v16_closed_loop_episode("V16_BALANCED", "balanced")
    m_calm = _run_v16_closed_loop_episode("V16_CALM_FAST", "calm")
    assert all(m["cmax"] > 0 for m in (m_quality, m_balanced, m_calm))


def test_v16_2amr_bridge_resource_window_reversal() -> None:
    """Verify 2-AMR symmetric bridge crew-window reversal rewards prioritizing tight window `OP_B`."""
    net = Network()
    coords = {
        "S0": (0.0, 0.0, NodeKind.BUFFER, 2),
        "LA": (2.0, 1.0, NodeKind.BUFFER, 2),
        "LB": (2.0, -1.0, NodeKind.BUFFER, 2),
        "C0": (4.0, 0.0, NodeKind.AISLE, 1),
        "C1": (8.0, 0.0, NodeKind.AISLE, 1),
        "WA": (10.0, 1.0, NodeKind.WORKSTATION, 2),
        "WB": (10.0, -1.0, NodeKind.WORKSTATION, 2),
    }
    for nid, (x, y, kind, cap) in coords.items():
        net.add_node(Node(nid, x, y, kind=kind, capacity=cap))
    for u, v, d in (
        ("S0", "LA", 2.0),
        ("S0", "LB", 2.0),
        ("LA", "C0", 2.0),
        ("LB", "C0", 2.0),
        ("C0", "C1", 4.0),
        ("C1", "WA", 2.0),
        ("C1", "WB", 2.0),
    ):
        net.add_edge(Edge(u, v, length=d, speed_limit=1.0, capacity=1))
        net.add_edge(Edge(v, u, length=d, speed_limit=1.0, capacity=1))

    tasks = [
        Task("T_A", "LA", "WA", release_step=0, due_step=20),
        Task("T_B", "LB", "WB", release_step=0, due_step=25),
    ]
    ops = [
        OperationSnapshot("OP_A", ("T_A",), "WA", crew_id=1, tool_id=1, proc_steps=2, due_step=24, resource_windows=((22, 70),)),
        OperationSnapshot("OP_B", ("T_B",), "WB", crew_id=2, tool_id=2, proc_steps=2, due_step=24, resource_windows=((0, 15),)),
    ]
    sim = Simulator(
        net,
        tasks,
        [Agv("agv_0", max_speed=1.0, start_node="S0"), Agv("agv_1", max_speed=1.0, start_node="S0")],
        SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=150, load_time=1.0, unload_time=1.0, idle_retreat=False),
    )
    sim.reset()
    tracker = NativeProductionTracker(ops, tasks)
    snap = NativeSnapshotCapture.capture(sim, ops, hide_future_releases=True)
    ev = CVCREvaluator(net, snap, mode="B2", max_speed=1.0, use_rust=True)

    # Prioritizing agv_0 on T_A blocks C0->C1 and forces OP_B past [0, 15] -> +40 shift penalty; reversing preserves both
    res_a_first = ev.evaluate({"T_A": "agv_0", "T_B": "agv_1"})
    res_b_first = ev.evaluate({"T_A": "agv_1", "T_B": "agv_0"})
    assert res_a_first.valid and res_b_first.valid
    assert res_b_first.op_start["OP_B"] < res_a_first.op_start["OP_B"]
    assert res_b_first.objective < res_a_first.objective

    ctx = extract_v16_context(sim, tracker, np.zeros(14, dtype=np.float32), np.zeros(30, dtype=np.float32))
    assert float(ctx[0]) == 1.0
    assert load_v16_frontier_policy(None).select_portfolio_mode(ctx, profile="quality") == "V15_SOFT"
