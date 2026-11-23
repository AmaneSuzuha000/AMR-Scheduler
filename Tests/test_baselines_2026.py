"""Unit, Audit & Contract Verification Tests for 2026 SOTA Baselines."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from amr_sched.baselines.baselines_2026 import (
    ICRA26KittingHGNNProposer,
    IJCAI26PGMPOProposer,
    ParimiMIT26TAPFPCLNSProposer,
    SowJAIR26DyMABProposer,
    TII26E2GMoEProposer,
    WuEJOR26Proposer,
)
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.rl import AMRReschedulingEnv, RuleV10PortfolioPolicy, compute_expert_v10_action_mask, compute_expert_v9_state


def _run_closed_loop_sanity(proposer) -> None:
    env = AMRReschedulingEnv(
        topology="unseen_choke_yard",
        seed=8201,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        real_compute_window_coupling=True,
        expert_v10_portfolio=True,
        loaded_repair_mode="haven_v12",
        loaded_repair_threshold=20,
        record_trajectory=True,
        external_proposer=proposer,
        execution_aware=True,
        p4_choke_robust=True,
    )
    ctrl = RuleV10PortfolioPolicy(epoch_interval=16)
    ctrl.reset()
    obs, info = env.reset(seed=8201)
    done = False
    while not done:
        assert env.sim is not None and env.tracker is not None
        o9 = compute_expert_v9_state(
            env.sim, env.tracker,
            last_replan_tick=env.last_replan_tick,
            recent_conflict_ticks=env.recent_conflict_ticks,
            decision_logs=env.decision_logs,
        )
        iu = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
        act = ctrl.select_action(o9, iu, sim=env.sim)
        obs, _, tm, tr, info = env.step(act)
        done = tm or tr
    assert env.sim is not None and env.tracker is not None
    aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
    assert aud.pass_audit
    m = env.compute_episode_metrics()
    assert m["task_completion_rate"] > 0.0


def test_six_2026_baselines_closed_loop_audit() -> None:
    """Verify Wu EJOR'26, Choi IJCAI'26, Ma TII'26, Sow JAIR'26, Parimi MIT'26 & Zhang ICRA'26 pass full trajectory audit."""
    for prop in (
        WuEJOR26Proposer(seed=1, sample_rollouts=2),
        IJCAI26PGMPOProposer(seed=1, c_policies=4),
        TII26E2GMoEProposer(seed=1, rollout_samples=2),
        SowJAIR26DyMABProposer(seed=1),
        ParimiMIT26TAPFPCLNSProposer(seed=1),
        ICRA26KittingHGNNProposer(seed=1, rollout_samples=2),
    ):
        _run_closed_loop_sanity(prop)


def test_2026_mechanisms_faithful_properties() -> None:
    """Verify exact mathematical invariants of JAIR'26 DyMAB, TII'26 TopK HMoE, IJCAI'26 Bit-DAN, MIT'26 closure & ICRA'26 Eq.(4)."""
    import torch
    from amr_sched.baselines.baselines_2026 import (
        ChoiIJCAI26PGMPONet,
        MaTII26E2GMoENet,
        ZhangICRA26KittingHGNNNet,
        step_dynamic_state,
    )
    from amr_sched.cvcr.snapshot import NativeSnapshot, OperationSnapshot, RobotSnapshot, TaskSnapshot

    # 1. DyMAB sliding window queue evicts stale non-stationary rewards
    dymab = SowJAIR26DyMABProposer(window_w=3, seed=7)
    for r in (5.0, 5.0, 5.0, 0.0, 0.0, 0.0):
        dymab._queues["H_rand"].append(r)
    assert len(dymab._queues["H_rand"]) == 3
    assert sum(dymab._queues["H_rand"]) == 0.0

    # 2. Ma TII'26 Top-K HMoE activates strictly <= top_k experts per state
    net_moe = MaTII26E2GMoENet(n_experts=8, top_k=2)
    net_moe.eval()
    op_x = torch.randn(2, 4, 8)
    res_x = torch.randn(2, 3, 6)
    pair_x = torch.randn(2, 4, 3, 6)
    pred_adj = torch.eye(4).unsqueeze(0).unsqueeze(-1).expand(2, 4, 4, 3)
    dist_6d = torch.zeros(2, 6)
    _, _, omega = net_moe(op_x, res_x, pair_x, pred_adj, dist_6d)
    assert omega.shape == (2, 8)
    assert int((omega[0] > 1e-6).sum().item()) <= 2

    # 3. Choi IJCAI'26 C=8 unique bit vectors produce distinct policy logits
    net_pg = ChoiIJCAI26PGMPONet()
    bits = net_pg.bit_vectors(8, 4)
    assert len({tuple(row.tolist()) for row in bits}) == 8

    # 4. Parimi MIT'26 transitive successor closure collects downstream chain
    mit = ParimiMIT26TAPFPCLNSProposer()
    snap = NativeSnapshot(
        schema="test", tick=0, version={}, world_fingerprint="w",
        robots=[RobotSnapshot("R0", "N0", "idle")],
        tasks=[
            TaskSnapshot("T1", "N0", "W1", 0, 50, "pending", kit_op_id="OP1"),
            TaskSnapshot("T2", "N0", "W1", 0, 80, "pending", kit_op_id="OP2"),
            TaskSnapshot("T3", "N0", "W1", 0, 120, "pending", kit_op_id="OP3"),
        ],
        operations=[
            OperationSnapshot("OP1", ("T1",), "W1", 0, 0, 5, 50),
            OperationSnapshot("OP2", ("T2",), "W1", 0, 0, 5, 80, predecessor_op_id="OP1"),
            OperationSnapshot("OP3", ("T3",), "W1", 0, 0, 5, 120, predecessor_op_id="OP2"),
        ],
        reservations=[], now_monotonic=0.0,
    )
    closure = mit._transitive_closure_tasks(["T1"], {"T1", "T2", "T3"}, snap)
    assert set(closure) == {"T1", "T2", "T3"}

    # 5. Zhang ICRA'26 Eq.(4) selective cost-sensitive kitting buffer edge weight + HGNN forward
    net_kit = ZhangICRA26KittingHGNNNet()
    sw_est = torch.tensor([[0.0, 2.0, 5.0]], dtype=torch.float32)
    is_kit = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32)
    w_ij = net_kit.buffer_edge_weight(sw_est, is_kit, alpha=0.3)
    assert float(w_ij[0, 0].item()) == 0.0
    assert float(w_ij[0, 2].item()) > float(w_ij[0, 1].item()) > 0.5
    logits_k, val_k = net_kit(
        torch.randn(1, 3, 8),
        torch.randn(1, 2, 6),
        torch.randn(1, 3, 2, 6),
        torch.eye(3).unsqueeze(0).unsqueeze(-1).expand(1, 3, 3, 3),
        torch.randn(1, 4),
        torch.stack([is_kit, sw_est], dim=-1),
    )
    assert logits_k.shape == (1, 3, 2) and val_k.shape == (1,)

    # 6. Sequential dynamic load update increments chosen AGV queue & decreases slack
    r_in = torch.zeros(2, 3, 6)
    p_in = torch.zeros(2, 4, 3, 6)
    r_nxt, p_nxt = step_dynamic_state(r_in, p_in, torch.tensor([1, 2]))
    assert float(r_nxt[0, 1, 4].item()) > 0.0 and float(r_nxt[0, 0, 4].item()) == 0.0
    assert float(p_nxt[1, 0, 2, 2].item()) > 0.0 and float(p_nxt[1, 0, 2, 3].item()) < 0.0

    # 7. Semicircle bay allocator never mutates op.bay_id once any sibling kit task is PICKED (ACID lock)
    from amr_sched.baselines.baselines_2026 import _assign_semicircle_priority_bays
    from amr_sched.network import Edge, Network, Node
    net_tiny = Network()
    net_tiny.add_node(Node("N0", 0.0, 0.0))
    net_tiny.add_node(Node("W1", 1.0, 0.0))
    net_tiny.add_node(Node("W2", 5.0, 0.0))
    net_tiny.add_edge(Edge("N0", "W1", length=1.0, speed_limit=1.0))
    net_tiny.add_edge(Edge("N0", "W2", length=5.0, speed_limit=1.0))
    snap_lock = NativeSnapshot(
        schema="test", tick=5, version={}, world_fingerprint="w",
        robots=[RobotSnapshot("R0", "N0", "idle")],
        tasks=[
            TaskSnapshot("KT1", "N0", "W2", 0, 50, "picked", kit_op_id="OP_LOCK", pickup_step=3),
            TaskSnapshot("KT2", "N0", "W2", 0, 50, "assigned", assigned_agv="R0", kit_op_id="OP_LOCK"),
        ],
        operations=[OperationSnapshot("OP_LOCK", ("KT1", "KT2"), "W2", 0, 0, 5, 50, started=False, eligible_bays=("W1", "W2"))],
        reservations=[], now_monotonic=0.0,
    )
    bays_out, _ = _assign_semicircle_priority_bays(snap_lock, net_tiny)
    assert bays_out["OP_LOCK"] == "W2"


def test_baselines_2026_strict_missing_checkpoint_and_allow_untrained(tmp_path) -> None:
    """All 4 neural 2026 proposers fail loudly on missing checkpoints unless allow_untrained=True."""
    import pytest
    from amr_sched.baselines.baselines_2026 import (
        ICRA26KittingHGNNProposer,
        IJCAI26PGMPOProposer,
        TII26E2GMoEProposer,
        WuEJOR26Proposer,
    )

    missing = tmp_path / "does_not_exist_ckpt.pt"
    for cls in (
        WuEJOR26Proposer,
        IJCAI26PGMPOProposer,
        TII26E2GMoEProposer,
        ICRA26KittingHGNNProposer,
    ):
        with pytest.raises(FileNotFoundError):
            cls(ckpt_path=missing)
        untrained = cls(ckpt_path=missing, allow_untrained=True)
        assert getattr(untrained, "_checkpoint_loaded", False) is False
        assert getattr(untrained, "_checkpoint_sha256", None) is None


def test_wu_and_pgmpo_torch_numpy_numerical_parity() -> None:
    """Wu EJOR26 and Choi IJCAI26 PGMPO achieve <1e-5 PyTorch-NumPy forward parity and identical greedy proposals."""
    import numpy as np
    import pytest
    from amr_sched.baselines.baselines_2026 import IJCAI26PGMPOProposer, WuEJOR26Proposer
    from amr_sched.baselines.policy_registry import create_discrete_policy_env
    from amr_sched.cvcr.snapshot import NativeSnapshotCapture
    from tools.benchmarks.compare_workshop_algorithms import build_track2_scenario

    torch = pytest.importorskip("torch")

    sc, p2_cfg, _ = build_track2_scenario(101, "kitting_windows")
    _, _, env = create_discrete_policy_env("rule_v10", seed=101, max_steps=60, use_rust=True, p2_cfg=p2_cfg)
    env.reset(seed=101, options={"scenario_tuple": sc})
    for _ in range(15):
        env.step(0)
    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations)

    wu_t = WuEJOR26Proposer(backend="torch")
    wu_n = WuEJOR26Proposer(backend="numpy")
    prop_wu_t = wu_t(snap, env.sim.network, budget_ms=5000.0)
    prop_wu_n = wu_n(snap, env.sim.network, budget_ms=5000.0)
    assert prop_wu_t.candidate_digest == prop_wu_n.candidate_digest
    assert prop_wu_t.proposed_assignment == prop_wu_n.proposed_assignment
    assert prop_wu_t.telemetry["checkpoint_loaded"] is True
    assert prop_wu_n.telemetry["checkpoint_loaded"] is True
    assert len(str(prop_wu_t.telemetry["checkpoint_sha256"])) == 64
    assert len(str(prop_wu_n.telemetry["checkpoint_sha256"])) == 64
    assert prop_wu_t.telemetry["inference_backend"] == "torch"
    assert prop_wu_n.telemetry["inference_backend"] == "numpy"

    pg_t = IJCAI26PGMPOProposer(backend="torch")
    pg_n = IJCAI26PGMPOProposer(backend="numpy")
    prop_pg_t = pg_t(snap, env.sim.network, budget_ms=5000.0)
    prop_pg_n = pg_n(snap, env.sim.network, budget_ms=5000.0)
    assert prop_pg_t.candidate_digest == prop_pg_n.candidate_digest
    assert prop_pg_t.proposed_assignment == prop_pg_n.proposed_assignment
    assert prop_pg_t.telemetry["checkpoint_loaded"] is True
    assert prop_pg_n.telemetry["checkpoint_loaded"] is True
    assert len(str(prop_pg_t.telemetry["checkpoint_sha256"])) == 64
    assert len(str(prop_pg_n.telemetry["checkpoint_sha256"])) == 64
    assert prop_pg_t.telemetry["inference_backend"] == "torch"
    assert prop_pg_n.telemetry["inference_backend"] == "numpy"

    assert wu_t._net is not None and wu_n._net is not None
    assert pg_t._net is not None and pg_n._net is not None

    rng = np.random.default_rng(7)
    st = rng.normal(size=(2, 12, 5)).astype(np.float32)
    dyn = rng.normal(size=(2, 8, 5)).astype(np.float32)
    dec = rng.normal(size=(2, 12)).astype(np.float32)
    hp = rng.normal(size=(2, 128)).astype(np.float32)
    msk = np.array([[True, True, False, True, True], [True, False, True, True, False]])
    with torch.no_grad():
        th = wu_t._net.encode_static(torch.from_numpy(st))
        tl, thn = wu_t._net.step_pointer(
            th,
            torch.from_numpy(dyn),
            torch.from_numpy(dec),
            torch.from_numpy(hp),
            torch.from_numpy(msk),
        )
        tv = wu_t._net.estimate_value(torch.from_numpy(st), torch.from_numpy(dyn))
    nh = wu_n._net.encode_static(st)
    nl, nhn = wu_n._net.step_pointer(nh, dyn, dec, hp, msk)
    nv = wu_n._net.estimate_value(st, dyn)
    assert float(np.max(np.abs(tl.numpy() - nl))) < 1e-5
    assert float(np.max(np.abs(thn.numpy() - nhn))) < 1e-5
    assert float(np.max(np.abs(tv.numpy() - nv))) < 1e-5

    op_x = rng.normal(size=(2, 4, 8)).astype(np.float32)
    res_x = rng.normal(size=(2, 3, 6)).astype(np.float32)
    pair_x = rng.normal(size=(2, 4, 3, 6)).astype(np.float32)
    z_bit = rng.normal(size=(2, 4)).astype(np.float32)
    with torch.no_grad():
        tp = pg_t._net(
            torch.from_numpy(op_x),
            torch.from_numpy(res_x),
            torch.from_numpy(pair_x),
            torch.from_numpy(z_bit),
        )
    np_out = pg_n._net.forward(op_x, res_x, pair_x, z_bit)
    assert float(np.max(np.abs(tp.numpy() - np_out))) < 1e-5
