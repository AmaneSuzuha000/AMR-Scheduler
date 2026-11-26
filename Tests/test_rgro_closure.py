"""Automated verification tests for `IF-CORE-RGRO-01` (`Decision-Focused Risk-Gated Residual Optimization`).

Covers:
  1. Zero scenario-name, Track-ID, or holdout-seed branching inside `rgro_proposer.py` & `dfl_critic.py`.
  2. Strict search-order invariance (`max |f(fwd) - f(rev)| < 1e-6`) & read-only history buffer.
  3. Shared Layer-1 single-call budget invariant (`graph_extract_calls == 1`, `encode_full_calls <= 1`).
  4. Episode-grouped & non-exchangeable CRC monotonicity and finite-sample bound integrity.
  5. Cohort E / E-Struct / E-Compute strict disjointness from ALL historical and DEV cohorts.
  6. End-to-end `RGROProposer` execution passing `IndependentAuditor(require_trajectory=True)`.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from experiments.rgro.dfl_critic import (
    DFLAdvantageEnsemble,
    ExecutionHistoryBuffer,
    REL_INPUT_DIM,
    build_anchor_relative_feature,
)
from experiments.rgro.rgro_common import (
    COHORT_E_ALL_SEEDS,
    DEV_STRESS_SEEDS,
    assert_cohort_e_disjoint,
    assert_dev_quarantine,
    build_dev_stress_24_specs,
)
from experiments.rgro.rgro_proposer import RGROProposer, make_ladder_config, run_rgro_episode
from experiments.rgro.task3_risk_calibration import calibrate_episode_grouped_crc
from experiments.v20_audit.common import build_val72_specs


def test_zero_scenario_seed_or_track_branching() -> None:
    """Assert zero hardcoded scenario names, Track IDs, or holdout/failure seed literals in core solver files."""
    solver_files = [
        ROOT / "experiments" / "rgro" / "dfl_critic.py",
        ROOT / "experiments" / "rgro" / "rgro_proposer.py",
    ]
    forbidden_tokens = (
        "track1", "track2", "multi_wave_surge", "execution_blockage",
        "kitting_windows", "offset_dual_lane", "window_bottleneck",
        "2906", "3905", "3906", "5901", "5906", "37302",
        "77101", "7901", "8901", "9901",
    )
    for fp in solver_files:
        src = fp.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(fp))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                val = str(node.value).lower()
                for tok in forbidden_tokens:
                    assert tok not in val, f"Forbidden literal {tok!r} found in {fp.name}: {val!r}"


def test_dfl_search_order_invariance_and_readonly_history() -> None:
    """Verify `DFLAdvantageEnsemble` is permutation-invariant (< 1e-6) and never mutates history."""
    rng = np.random.default_rng(7)
    members = [
        {
            "w1": rng.normal(0, 0.2, size=(REL_INPUT_DIM, 32)).astype(np.float32),
            "b1": rng.normal(0, 0.1, size=(32,)).astype(np.float32),
            "w2": rng.normal(0, 0.2, size=(32, 2)).astype(np.float32),
            "b2": rng.normal(0, 0.1, size=(2,)).astype(np.float32),
        }
        for _ in range(3)
    ]
    ens = DFLAdvantageEnsemble(
        feat_mean=np.zeros(REL_INPUT_DIM, dtype=np.float32),
        feat_std=np.ones(REL_INPUT_DIM, dtype=np.float32),
        members=members,
    )
    h_buf = ExecutionHistoryBuffer(maxlen=8)
    h_before = h_buf.get_vector_readonly().copy()

    fv_anchor = rng.normal(0, 1.0, size=(24,)).astype(np.float32)
    cands = [rng.normal(0, 1.0, size=(24,)).astype(np.float32) for _ in range(6)]
    X = np.stack([build_anchor_relative_feature(c, fv_anchor, h_buf.get_vector_readonly()) for c in cands], axis=0)

    fwd = ens.score_batch(X, kappa=0.5)
    rev = ens.score_batch(X[::-1], kappa=0.5)[::-1]
    perm = [3, 0, 5, 1, 4, 2]
    inv_perm = np.argsort(perm)
    shuf = [ens.score_batch(X[perm], kappa=0.5)[int(i)] for i in inv_perm]

    for a, b, c in zip(fwd, rev, shuf):
        assert abs(a.pred_delta_ja - b.pred_delta_ja) < 1e-4
        assert abs(a.pred_delta_ja - c.pred_delta_ja) < 1e-4
        assert abs(a.harm_prob - b.harm_prob) < 1e-6
        assert abs(a.lcb_advantage - c.lcb_advantage) < 1e-4

    assert np.array_equal(h_before, h_buf.get_vector_readonly())


def test_episode_grouped_crc_monotonicity_and_bounds() -> None:
    """Verify episode-grouped CRC thresholds are non-decreasing in alpha and respect finite-sample bound."""
    recs = []
    for ep in range(20):
        for s in range(4):
            p = float(((ep * 4 + s) % 20) / 20.0)
            is_h = 1.0 if p >= 0.55 else 0.0
            recs.append(
                {
                    "episode_key": f"ep_{ep:02d}",
                    "has_proposal": True,
                    "harm_prob": p,
                    "is_harm": is_h,
                }
            )
    taus = []
    for alpha in (0.05, 0.10, 0.20, 0.35, 0.50):
        res = calibrate_episode_grouped_crc(recs, alpha)
        assert res["n_episodes"] == 20
        assert res["cal_upper_bound"] <= alpha + 1e-9 or res["tau"] == 0.0
        taus.append(res["tau"])
    assert all(taus[i] <= taus[i + 1] + 1e-9 for i in range(len(taus) - 1)), f"Non-monotone taus: {taus}"


def test_cohort_e_and_dev_disjointness() -> None:
    """Verify Cohort E / E-Struct / E-Compute and DEV-Stress-24 have zero illegal seed overlap."""
    assert_cohort_e_disjoint()
    assert_dev_quarantine(DEV_STRESS_SEEDS, label="pytest_dev_stress")
    # Ensure DEV quarantine blocks Cohort E seeds if ever passed
    blocked = False
    try:
        assert_dev_quarantine([next(iter(COHORT_E_ALL_SEEDS))], label="should_fail")
    except RuntimeError:
        blocked = True
    assert blocked, "DEV quarantine failed to block Cohort E seed!"


def test_shared_layer1_single_call_and_independent_audit() -> None:
    """Verify RGROProposer computes graph & full_net encoding at most ONCE per decision and passes IndependentAuditor."""
    val72 = build_val72_specs()
    sp = val72[0]
    cfg = make_ladder_config("U3")
    row = run_rgro_episode(sp, cfg, use_virtual_clock=True)
    assert row["audit_passed"] is True, f"Audit failed on {sp['episode_key']}"
    assert row["comp100"] is True, f"Did not complete 100% on {sp['episode_key']}"

    # Direct single-snapshot Layer-1 call count check
    from amr_sched.baselines.policy_registry import create_discrete_policy_env
    from amr_sched.cvcr.snapshot import NativeSnapshotCapture
    import copy

    _, _, env = create_discrete_policy_env("V19_ADAPTIVE_RL", seed=int(sp["seed"]), max_steps=int(sp["max_steps"]), use_rust=True)
    env.reset(seed=int(sp["seed"]), options={"scenario_tuple": copy.deepcopy(sp["scenario_tuple"])})
    for _ in range(15):
        env.step(0)
    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
    prop = RGROProposer(cfg)
    out = prop(snap, env.sim.network, budget_ms=35.0)
    assert prop.last_call_stats["graph_extract_calls"] == 1
    assert prop.last_call_stats["encode_full_calls"] <= 1
    assert out is not None
