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
