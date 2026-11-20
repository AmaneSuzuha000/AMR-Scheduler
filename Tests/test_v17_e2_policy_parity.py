"""Stage 3 (E2) Tests: V17 Causal Policy, 50-D Pairwise Ranker, PyTorch/NumPy Parity, & Attribution Toggles."""

from __future__ import annotations

import numpy as np
import pytest

from amr_sched.cvcr.evaluator import EvalResult
from amr_sched.cvcr.v15_ea_joint import V15_GLOBAL_FEAT_DIM, V15Candidate
from amr_sched.cvcr.v16_frontier import V16_CTX_DIM
from amr_sched.cvcr.v17_window_protection import (
    V17_CAND_FEAT_DIM,
    V17_PAIRWISE_DIM,
    V17AdaptiveController,
    V17WindowProtectionPolicy,
    build_v17_cand_matrix,
    load_v17_window_policy,
)


def _dummy_weights(seed: int = 123) -> dict[str, np.ndarray]:
    """Build deterministic dummy V17 weight dictionary for parity testing."""
    rng = np.random.default_rng(seed)
    return {
        "v17_m_mu": rng.normal(0.0, 1.0, size=(V16_CTX_DIM,)).astype(np.float32),
        "v17_m_sd": np.maximum(0.2, rng.uniform(0.5, 2.0, size=(V16_CTX_DIM,))).astype(np.float32),
        "v17_mode_w1": rng.normal(0.0, 0.2, size=(V16_CTX_DIM, 48)).astype(np.float32),
        "v17_mode_b1": rng.normal(0.0, 0.05, size=(48,)).astype(np.float32),
        "v17_mode_w2": rng.normal(0.0, 0.2, size=(48, 3)).astype(np.float32),
        "v17_mode_b2": rng.normal(0.0, 0.05, size=(3,)).astype(np.float32),
        "v17_c_mu": rng.normal(0.0, 1.0, size=(V17_CAND_FEAT_DIM,)).astype(np.float32),
        "v17_c_sd": np.maximum(0.2, rng.uniform(0.5, 2.0, size=(V17_CAND_FEAT_DIM,))).astype(np.float32),
        "v17_g_mu": rng.normal(0.0, 1.0, size=(V15_GLOBAL_FEAT_DIM,)).astype(np.float32),
        "v17_g_sd": np.maximum(0.2, rng.uniform(0.5, 2.0, size=(V15_GLOBAL_FEAT_DIM,))).astype(np.float32),
        "v17_cand_w1": rng.normal(0.0, 0.2, size=(V17_CAND_FEAT_DIM, 48)).astype(np.float32),
        "v17_cand_b1": rng.normal(0.0, 0.05, size=(48,)).astype(np.float32),
        "v17_att_q": rng.normal(0.0, 0.2, size=(48, 24)).astype(np.float32),
        "v17_att_k": rng.normal(0.0, 0.2, size=(48, 24)).astype(np.float32),
        "v17_att_v": rng.normal(0.0, 0.2, size=(48, 48)).astype(np.float32),
        "v17_cand_w2": rng.normal(0.0, 0.2, size=(96, 48)).astype(np.float32),
        "v17_cand_b2": rng.normal(0.0, 0.05, size=(48,)).astype(np.float32),
        "v17_cand_w3": rng.normal(0.0, 0.2, size=(48, 1)).astype(np.float32),
        "v17_cand_b3": rng.normal(0.0, 0.05, size=(1,)).astype(np.float32),
        "v17_g_w1": rng.normal(0.0, 0.2, size=(V15_GLOBAL_FEAT_DIM, 24)).astype(np.float32),
        "v17_g_b1": rng.normal(0.0, 0.05, size=(24,)).astype(np.float32),
        "v17_g_w2": rng.normal(0.0, 0.2, size=(24, 1)).astype(np.float32),
        "v17_g_b2": rng.normal(0.0, 0.05, size=(1,)).astype(np.float32),
    }


def test_v17_pairwise_pytorch_numpy_exact_parity_and_keep_zeroing() -> None:
    """Assert NumPy `V17WindowProtectionPolicy.score_candidates` matches PyTorch within `1e-6`."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(99)
    w = _dummy_weights(seed=77)
    pol = V17WindowProtectionPolicy(weights=w, conservative_gate=0.25, zero_keep_residual=True, ranker_mode="pairwise")

    cands: list[V15Candidate] = []
    for idx in range(5):
        sk = "keep_incumbent" if idx == 0 else "fast_detour_regret"
        feat36 = rng.normal(0.0, 1.0, size=(36,)).astype(np.float32)
        feat36[16] = 1.0 if sk == "keep_incumbent" else 0.0
        pw14 = rng.normal(0.0, 1.0, size=(V17_PAIRWISE_DIM,)).astype(np.float32)
        ds = float(0.0 if sk == "keep_incumbent" else rng.normal(1.0, 1.5))
        valid = idx != 4
        ev = EvalResult(
            robot_paths={},
            robot_order_seq={},
            order_unload_ticks={},
            op_start={},
            op_end={},
            op_res_ready={},
            objective=500.0 - ds * 20.0,
            valid=valid,
            starvation_ticks=0,
            eval_latency_ms=0.0,
            mode="B2",
        )
        cands.append(
            V15Candidate(
                cand_id=f"c_{idx}",
                source_kind=sk,
                assignment={},
                bay_overrides={},
                crew_overrides={},
                op_priority={},
                eval_res=ev,
                traffic_weights={},
                traffic_cutoff=0,
                guide_routes={},
                commit_seq_horizon=0,
                destroy_tids=(),
                cliff_saved=False,
                det_score=ds,
                features=feat36,
                pairwise_features=pw14,
            )
        )

    gf = rng.normal(0.0, 1.0, size=(V15_GLOBAL_FEAT_DIM,)).astype(np.float32)
    mat = build_v17_cand_matrix(cands)
    assert mat.shape == (5, 50)
    np_scores = pol.score_candidates(gf, cands)
    assert np_scores[0] == cands[0].det_score  # keep_incumbent residual strictly zeroed
    assert np_scores[4] <= -1e8  # invalid candidate masked

    # Reference PyTorch evaluation using the identical weights
    cn = torch.tensor(np.clip((mat - w["v17_c_mu"]) / np.maximum(0.05, w["v17_c_sd"]), -6.0, 6.0), dtype=torch.float32)
    gn = torch.tensor(np.clip((gf - w["v17_g_mu"]) / np.maximum(0.05, w["v17_g_sd"]), -6.0, 6.0), dtype=torch.float32)
    with torch.no_grad():
        h = torch.relu(cn @ torch.tensor(w["v17_cand_w1"]) + torch.tensor(w["v17_cand_b1"]))
        q = h @ torch.tensor(w["v17_att_q"])
        k = h @ torch.tensor(w["v17_att_k"])
        v = h @ torch.tensor(w["v17_att_v"])
        att = torch.softmax((q @ k.T) / (24.0 ** 0.5), dim=-1)
        h_att = torch.relu(torch.cat([h, att @ v], dim=-1) @ torch.tensor(w["v17_cand_w2"]) + torch.tensor(w["v17_cand_b2"]))
        delta = (h_att @ torch.tensor(w["v17_cand_w3"]) + torch.tensor(w["v17_cand_b3"])).squeeze(-1)
        hg = torch.relu(gn @ torch.tensor(w["v17_g_w1"]) + torch.tensor(w["v17_g_b1"]))
        trust = torch.sigmoid(hg @ torch.tensor(w["v17_g_w2"]) + torch.tensor(w["v17_g_b2"])).squeeze() * 2.0 * 0.25
        adj = torch.tanh(delta / 2.0) * trust
        km = torch.tensor((mat[:, 16] > 0.5).astype(np.float32))
        pt_scores = torch.tensor([c.det_score for c in cands], dtype=torch.float32) + adj * (1.0 - km)

    for i in range(4):
        assert abs(float(np_scores[i]) - float(pt_scores[i])) < 1e-6


def test_v17_attribution_toggles_router_ranker_restraint() -> None:
    """Verify `router_mode`, `ranker_mode`, and `enable_restraint` attribution toggles behave cleanly."""
    pol_det = load_v17_window_policy(router_mode="rule", ranker_mode="det", enable_restraint=False)
    assert pol_det.router_mode == "rule"
    assert pol_det.ranker_mode == "det"
    assert pol_det.enable_restraint is False

    ctx = np.zeros((V16_CTX_DIM,), dtype=np.float32)
    ctx[0] = 0.0  # no resource windows
    ctx[1] = 0.02  # few predecessors
    assert pol_det.select_portfolio_mode(ctx, profile="quality") == "V10_GUIDED"
    ctx[4] = 1.0  # drift present
    assert pol_det.select_portfolio_mode(ctx, profile="quality") == "V7_CALM"

    ctrl = V17AdaptiveController(policy=pol_det, enable_restraint=True)
    assert pol_det.enable_restraint is True
    assert ctrl.policy is pol_det


def test_v17_surgical_repair_triggers_even_under_v7_calm_mode() -> None:
    """Verify `V17AdaptiveController` triggers action=1 surgical repair regardless of locked mode."""
    from types import SimpleNamespace
    from amr_sched.entities import TaskStatus

    ctrl = V17AdaptiveController(policy=load_v17_window_policy(router_mode="rule", ranker_mode="det"), stall_threshold=15)
    ctrl.reset()
    ctrl.locked_mode = "V7_CALM"

    t0 = SimpleNamespace(task_id="T0", status=TaskStatus.PICKED, assigned_agv="agv_0")
    a0 = SimpleNamespace(agv_id="agv_0", cur_task="T0", cur_node="N1", start_node="N0", route=["N1", "N2"], state="blocked")
    sim = SimpleNamespace(
        network=SimpleNamespace(nodes={}, edges={}),
        tasks=[t0],
        agvs=[a0],
        _task_by_id={"T0": t0},
        _faulted_agvs=set(),
        _transit={},
        _blocked_since={"agv_0": 0},
        _load_end={"agv_0": 0},
        _drop_end={},
        _trajectory=[],
        _step_cursor=25,
        _jobs={"agv_0": SimpleNamespace(task_id="T0", leg="to_dropoff", hops=[], idx=0)},
        _p2_execution_aware=False,
        _perturbation_model=None,
    )
    mask = np.ones((9,), dtype=np.int32)
    act = ctrl.select_action(np.zeros((44,), dtype=np.float32), {"step": 25, "action_mask": mask}, sim=sim)
    assert act == 1

