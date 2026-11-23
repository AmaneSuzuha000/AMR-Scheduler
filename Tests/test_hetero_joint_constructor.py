"""Tests for V19 Heterogeneous-Graph Critical-Chain Joint Constructor ."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.cvcr.transaction import NativeTransactionOwner
from amr_sched.entities import TaskStatus
from amr_sched.rl import AMRReschedulingEnv, RuleV10PortfolioPolicy, compute_expert_v10_action_mask, compute_expert_v9_state
from amr_sched.rl.v19_hetero_constructor import (
    ABLATION_MODES,
    ACTION_STAGES,
    NumpyV19HeteroConstructorNet,
    V19HeteroConstructorNet,
    V19HeteroConstructorProposer,
    decode_v19_candidate,
    export_v19_npz,
    graph_to_torch_tensors,
)
from amr_sched.rl.v19_hetero_graph import apply_graph_ablation, extract_v19_hetero_graph
from tools.benchmarks.compare_workshop_algorithms import build_track1_scenario, build_track2_scenario

ROOT = Path(__file__).resolve().parents[1]


def _capture_mid_episode_snapshot(track: str = "track2", regime: str = "kitting_windows", seed: int = 101):
    sc_tuple, p2_cfg, max_steps = (
        build_track1_scenario(seed, regime) if track == "track1" else build_track2_scenario(seed, regime)
    )
    env = AMRReschedulingEnv(
        seed=seed,
        max_steps=max_steps,
        use_rust=True,
        execution_perturbation=p2_cfg,
        execution_aware=True,
        p4_choke_robust=True,
    )
    env.reset(seed=seed, options={"scenario_tuple": sc_tuple})
    ctrl = RuleV10PortfolioPolicy(epoch_interval=16)
    ctrl.reset()
    # Advance until at least one task is picked or faulted, or 60 ticks elapsed
    for _ in range(90):
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
        has_picked = any(t.status == TaskStatus.PICKED for t in env.sim.tasks)
        if has_picked and env.current_step >= 25:
            break
    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
    return snap, env.sim.network


def test_torch_free_import_of_v19_modules() -> None:
    """Both `v19_hetero_graph` and `v19_hetero_constructor` import cleanly without `torch` or `gymnasium`."""
    code = (
        "import sys; "
        "sys.modules['torch'] = None; "
        "sys.modules['gymnasium'] = None; "
        "import amr_sched.rl.v19_hetero_graph as g; "
        "import amr_sched.rl.v19_hetero_constructor as c; "
        "assert not c._HAS_TORCH; "
        "print('OK')"
    )
    res = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "OK" in res.stdout


def test_hard_commitment_and_illegal_masks() -> None:
    """Picked/completed tasks & started ops are strictly frozen; faulted AGVs & ineligible slots masked out."""
    snap, net = _capture_mid_episode_snapshot("track2", "kitting_windows", seed=101)
    # Mark first robot as faulted to test AGV fault masking and start op_0 to test op lock
    first_rob = snap.robots[0]
    from dataclasses import replace

    mod_robs = [replace(first_rob, state="faulted", faulted_until=snap.tick + 50)] + list(snap.robots[1:])
    mod_ops = [replace(snap.operations[0], started=True, start_step=snap.tick - 2)] + list(snap.operations[1:])
    mod_snap = replace(snap, robots=mod_robs, operations=mod_ops)

    clf = classify_commitments(mod_snap)
    graph = extract_v19_hetero_graph(mod_snap, net, clf=clf)

    for ftid in clf.frozen_tasks:
        assert ftid not in graph.rel_tids

    faulted_idx = graph.agv_ids.index(first_rob.robot_id)
    assert not graph.task_agv_mask[:, faulted_idx].any()

    op0_idx = graph.op_ids.index(mod_ops[0].op_id)
    assert mod_ops[0].op_id in graph.frozen_ops
    assert int(graph.op_bay_mask[op0_idx].sum()) == 1
    assert graph.op_bay_mask[op0_idx, graph.op_incumbent_bay_idx[op0_idx]]
    assert int(graph.op_crew_mask[op0_idx].sum()) == 1
    assert graph.op_crew_mask[op0_idx, graph.op_incumbent_crew_idx[op0_idx]]

    # Check graph ablation masks prod/comm channels while preserving masks & coords
    abl = apply_graph_ablation(graph, "no_prod_comm")
    assert float(np.abs(abl.op_feats[:, [0, 1, 2, 3, 4, 5, 9, 10, 11]]).max()) == 0.0
    assert float(np.abs(abl.task_feats[:, [7, 8, 9, 10]]).max()) == 0.0
    assert float(np.abs(abl.task_agv_feats[:, :, [3, 4, 7]]).max()) == 0.0
    assert np.array_equal(abl.op_bay_mask, graph.op_bay_mask)
    assert np.array_equal(abl.task_agv_mask, graph.task_agv_mask)


@pytest.mark.parametrize("action_stage", ACTION_STAGES)
@pytest.mark.parametrize("ablation_mode", ABLATION_MODES)
def test_pytorch_vs_numpy_parity_and_npz_export(tmp_path: Path, action_stage: str, ablation_mode: str) -> None:
    """Exported `.npz` and PyTorch `.pt` agree to `< 1e-5` on logits, value, and proposal digest."""
    snap, net = _capture_mid_episode_snapshot("track1", "kitting_windows", seed=102)
    raw_graph = extract_v19_hetero_graph(snap, net)
    graph = apply_graph_ablation(raw_graph, ablation_mode)

    torch.manual_seed(1234)
    pt_net = V19HeteroConstructorNet(
        hidden_dim=64,
        n_heads=4,
        n_layers=2,
        action_stage=action_stage,
        ablation_mode=ablation_mode,
    )
    pt_net.eval()

    pt_path = tmp_path / f"v19_{action_stage}_{ablation_mode}.pt"
    npz_path = tmp_path / f"v19_{action_stage}_{ablation_mode}.npz"
    torch.save({"model": pt_net.state_dict(), "config": pt_net.config_dict()}, pt_path)
    export_v19_npz(pt_path, npz_path)

    npz_data = dict(np.load(str(npz_path), allow_pickle=False))
    np_net = NumpyV19HeteroConstructorNet(npz_data)

    tensors = graph_to_torch_tensors(graph, device="cpu")
    with torch.inference_mode():
        pt_enc = pt_net.encode_entities(
            tensors["op_x"],
            tensors["task_x"],
            tensors["agv_x"],
            tensors["bay_x"],
            tensors["crew_x"],
            tensors["task_to_op"],
            tensors["op_dag_adj"],
        )
        pt_h12 = pt_net.a1_a2_heads(pt_enc, tensors["op_bay_x"], tensors["op_bay_mask"], tensors["op_crew_mask"])
    np_enc = np_net.encode_entities(graph)
    np_h12 = np_net.a1_a2_heads(np_enc, graph)

    for key in ("task_focus_logits", "bay_logits", "crew_logits", "value"):
        diff = float(np.max(np.abs(pt_h12[key].numpy() - np_h12[key])))
        assert diff < 1e-5, f"{key} max abs diff {diff} >= 1e-5"

    cand_pt = decode_v19_candidate(pt_net, graph, snap, greedy=True, np_rng=np.random.default_rng(7))
    cand_np = decode_v19_candidate(np_net, graph, snap, greedy=True, np_rng=np.random.default_rng(7))
    assert cand_pt.assign == cand_np.assign
    assert cand_pt.bays == cand_np.bays
    assert cand_pt.crews == cand_np.crews
    assert cand_pt.task_sequences == cand_np.task_sequences
    if len(graph.rel_tids) > 0:
        seq_diff = float(np.max(np.abs(cand_pt.seq_scores - cand_np.seq_scores)))
        assert seq_diff < 1e-5

    prop_pt = V19HeteroConstructorProposer(pt_path, backend="torch", use_rust=True, seed=77)
    prop_np = V19HeteroConstructorProposer(npz_path, backend="numpy", use_rust=True, seed=77)
    p1 = prop_pt(snap, net, budget_ms=120.0)
    p2 = prop_np(snap, net, budget_ms=120.0)
    assert p1.candidate_digest == p2.candidate_digest
    assert p1.telemetry["checkpoint_loaded"] is True
    assert p2.telemetry["checkpoint_loaded"] is True
    assert p1.telemetry["inference_backend"] == "torch"
    assert p2.telemetry["inference_backend"] == "numpy"


def test_missing_checkpoint_raises_unless_allowed(tmp_path: Path) -> None:
    """Strict default forbids running untrained checkpoints without `allow_untrained=True`."""
    missing = tmp_path / "does_not_exist.pt"
    snap, net = _capture_mid_episode_snapshot("track2", "nominal", seed=101)
    proposer_strict = V19HeteroConstructorProposer(missing)
    with pytest.raises(FileNotFoundError):
        proposer_strict(snap, net, budget_ms=50.0)

    proposer_ok = V19HeteroConstructorProposer(missing, allow_untrained=True)
    prop = proposer_ok(snap, net, budget_ms=50.0)
    assert prop.telemetry["checkpoint_loaded"] is False


@pytest.mark.parametrize("stage", ACTION_STAGES)
def test_v19_closed_loop_episode_audits_cleanly_on_both_tracks(tmp_path: Path, stage: str) -> None:
    """Closed-loop `V19HeteroConstructorProposer` achieves 100% completion and passes IndependentAuditor."""
    torch.manual_seed(3)
    pt_net = V19HeteroConstructorNet(action_stage=stage, ablation_mode="full")
    pt_path = tmp_path / f"smoke_{stage}.pt"
    torch.save({"model": pt_net.state_dict(), "config": pt_net.config_dict()}, pt_path)

    for builder, seed in ((build_track1_scenario, 101), (build_track2_scenario, 101)):
        sc_tuple, p2_cfg, max_steps = builder(seed, "nominal")
        proposer = V19HeteroConstructorProposer(pt_path, policy_id=f"V19_TEST_{stage}", use_rust=True, seed=seed)
        env = AMRReschedulingEnv(
            seed=seed,
            max_steps=max_steps,
            use_rust=True,
            record_trajectory=True,
            loaded_repair_mode="haven_v12",
            loaded_repair_threshold=20,
            execution_perturbation=p2_cfg,
            execution_aware=True,
            p4_choke_robust=True,
            expert_v10_portfolio=True,
            external_proposer=proposer,
        )
        ctrl = RuleV10PortfolioPolicy(epoch_interval=16)
        ctrl.reset()
        _, info = env.reset(seed=seed, options={"scenario_tuple": sc_tuple})
        tm = False
        tr = False
        while not (tm or tr):
            feats = compute_expert_v9_state(
                env.sim,
                env.tracker,
                last_replan_tick=env.last_replan_tick,
                recent_conflict_ticks=env.recent_conflict_ticks,
                decision_logs=env.decision_logs,
            )
            s_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
            act = ctrl.select_action(feats, s_info, sim=env.sim)
            _, _, tm, tr, info = env.step(act)
        m = env.compute_episode_metrics()
        aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
        assert aud.pass_audit, f"Audit failed: {aud.issues}"
        assert m["task_completion_rate"] == pytest.approx(1.0)
        assert m["production_completion_rate"] == pytest.approx(1.0)
        assert any(row["receipt_accepted"] for row in env.decision_logs)
        assert all(row["receipt_accepted"] or row["window_expired"] for row in env.decision_logs)


def test_tight_budget_safe_fallback(tmp_path: Path) -> None:
    """Tight decision budget stops candidate sampling cleanly and always returns valid proposal."""
    snap, net = _capture_mid_episode_snapshot("track1", "kitting_windows", seed=102)
    pt_net = V19HeteroConstructorNet(action_stage="A123", ablation_mode="full")
    pt_path = tmp_path / "budget_test.pt"
    torch.save({"model": pt_net.state_dict(), "config": pt_net.config_dict()}, pt_path)

    prop = V19HeteroConstructorProposer(pt_path, use_rust=True, seed=5)(snap, net, budget_ms=0.001)
    assert prop.telemetry["neural_candidates_evaluated"] <= 1
    assert prop.predicted_obj > 0.0


@pytest.mark.parametrize("ablation_mode", ABLATION_MODES)
def test_v19_orthogonal_graph_ablations_preserve_masks_and_parity(tmp_path: Path, ablation_mode: str) -> None:
    """All 8 orthogonal ablation modes preserve safety masks and `< 1e-5` Torch-NumPy parity."""
    from amr_sched.baselines.numpy_backends import load_numpy_state_dict

    snap, net = _capture_mid_episode_snapshot("track1", "kitting_windows", seed=103)
    raw_g = extract_v19_hetero_graph(snap, net)
    g = apply_graph_ablation(raw_g, ablation_mode)
    assert np.array_equal(g.op_bay_mask, raw_g.op_bay_mask)
    assert np.array_equal(g.op_crew_mask, raw_g.op_crew_mask)
    assert np.array_equal(g.task_agv_mask, raw_g.task_agv_mask)

    torch.manual_seed(19)
    pt_net = V19HeteroConstructorNet(action_stage="A1234", ablation_mode=ablation_mode)
    pt_net.eval()
    pt_path = tmp_path / f"ortho_{ablation_mode}.pt"
    torch.save({"model": pt_net.state_dict(), "config": pt_net.config_dict()}, pt_path)
    npz_path = export_v19_npz(pt_path)
    weights, _, _ = load_numpy_state_dict(npz_path, prefer_npz=True)
    np_net = NumpyV19HeteroConstructorNet(weights, action_stage="A1234", ablation_mode=ablation_mode)

    ten = graph_to_torch_tensors(g)
    with torch.inference_mode():
        enc_t = pt_net.encode_entities(
            ten["op_x"], ten["task_x"], ten["agv_x"], ten["bay_x"], ten["crew_x"], ten["task_to_op"], ten["op_dag_adj"]
        )
        h12_t = pt_net.a1_a2_heads(enc_t, ten["op_bay_x"], ten["op_bay_mask"], ten["op_crew_mask"])
    enc_n = np_net.encode_entities(g)
    h12_n = np_net.a1_a2_heads(enc_n, g)
    assert np.max(np.abs(h12_t["task_focus_logits"].numpy() - h12_n["task_focus_logits"])) < 1e-5
    assert np.max(np.abs(h12_t["bay_logits"].numpy() - h12_n["bay_logits"])) < 1e-5


def test_v19_a1_plackett_luce_gradient_flow() -> None:
    """Enabling `include_a1_plackett_luce=True` flows direct policy gradients to `a1_w1` and `a1_w2`."""
    snap, net = _capture_mid_episode_snapshot("track1", "kitting_windows", seed=101)
    g = extract_v19_hetero_graph(snap, net)
    ten = graph_to_torch_tensors(g)
    N_op = len(g.op_ids)
    N_t = len(g.rel_tids)
    assert N_t >= 2

    batch = dict(
        ten,
        op_valid=torch.ones((1, N_op), dtype=torch.bool),
        task_valid=torch.ones((1, N_t), dtype=torch.bool),
        agv_valid=torch.ones((1, len(g.agv_ids)), dtype=torch.bool),
        bay_valid=torch.ones((1, len(g.bay_ids)), dtype=torch.bool),
        crew_valid=torch.ones((1, len(g.crew_ids)), dtype=torch.bool),
        n_tasks=torch.tensor([N_t], dtype=torch.long),
    )
    chosen_bays = torch.from_numpy(g.op_incumbent_bay_idx).unsqueeze(0)
    chosen_crews = torch.from_numpy(g.op_incumbent_crew_idx).unsqueeze(0)
    chosen_agvs = torch.zeros((1, N_t), dtype=torch.long)
    focus_order = torch.arange(N_t, dtype=torch.long).unsqueeze(0)
    chosen_seq = torch.zeros((1, N_t), dtype=torch.float32)

    torch.manual_seed(29)
    net_off = V19HeteroConstructorNet(action_stage="A1234", ablation_mode="full")
    out_off = net_off.evaluate_joint_actions_batch(
        batch,
        chosen_bays=chosen_bays,
        chosen_crews=chosen_crews,
        chosen_agvs=chosen_agvs,
        focus_order=focus_order,
        chosen_seq=chosen_seq,
        include_a1_plackett_luce=False,
    )
    out_off["mean_logp"].sum().backward()
    assert net_off.a1_w1.weight.grad is None or float(net_off.a1_w1.weight.grad.abs().sum().item()) == 0.0

    torch.manual_seed(29)
    net_on = V19HeteroConstructorNet(action_stage="A1234", ablation_mode="full")
    out_on = net_on.evaluate_joint_actions_batch(
        batch,
        chosen_bays=chosen_bays,
        chosen_crews=chosen_crews,
        chosen_agvs=chosen_agvs,
        focus_order=focus_order,
        chosen_seq=chosen_seq,
        include_a1_plackett_luce=True,
        return_probs=True,
    )
    out_on["mean_logp"].sum().backward()
    assert net_on.a1_w1.weight.grad is not None
    assert net_on.a1_w2.weight.grad is not None
    assert float(net_on.a1_w1.weight.grad.abs().sum().item()) > 1e-6
    assert float(net_on.a1_w2.weight.grad.abs().sum().item()) > 1e-6
    assert "focus_logits" in out_on["probs"]


def test_v19_adaptive_coupling_proposer_and_b2_footprint(tmp_path: Path) -> None:
    """`V19AdaptiveCouplingProposer` (`AHG-RL`) proposes valid candidates and reports B2 incremental telemetry."""
    from amr_sched.rl.v19_hetero_constructor import (
        V19AdaptiveCouplingProposer,
        compute_dynamic_coupling_features,
    )

    snap, net = _capture_mid_episode_snapshot("track1", "kitting_windows", seed=101)
    g = extract_v19_hetero_graph(snap, net)
    phi = compute_dynamic_coupling_features(g)
    assert phi.shape == (6,)
    assert np.all((phi >= 0.0) & (phi <= 1.0))

    torch.manual_seed(31)
    f_net = V19HeteroConstructorNet(action_stage="A1234", ablation_mode="full")
    s_net = V19HeteroConstructorNet(action_stage="A1234", ablation_mode="staged")
    f_pt = tmp_path / "full_smoke.pt"
    s_pt = tmp_path / "staged_smoke.pt"
    torch.save({"model": f_net.state_dict(), "config": f_net.config_dict()}, f_pt)
    torch.save({"model": s_net.state_dict(), "config": s_net.config_dict()}, s_pt)

    for ev_mode in ("B2", "B2_SEEDED", "B1"):
        proposer = V19AdaptiveCouplingProposer(
            f_pt,
            s_pt,
            mode="bandit",
            evaluator_mode=ev_mode,
            use_rust=True,
            seed=101,
        )
        prop = proposer(snap, net, budget_ms=50.0)
        assert prop.predicted_obj > 0.0
        assert prop.telemetry["evaluator_mode"] == ev_mode
        assert 0.0 <= float(prop.telemetry["mean_changed_amr_ratio"]) <= 1.0
        assert 0.0 <= float(prop.telemetry["mean_reused_amr_ratio"]) <= 1.0


