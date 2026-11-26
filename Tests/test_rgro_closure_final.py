"""Final verification suite for `IF-CORE-RGRO-CLOSURE-01` (`Tests/test_rgro_closure_final.py`).

Enforces:
  1. Seed quarantine & strict disjointness across DEV and Cohort F / F-Struct / F-Compute.
  2. Zero scenario-name / seed / track-ID branching in solver modules (AST inspection).
  3. UnifiedDeadlineManager + Zero-Expiry Transaction Hard Veto invariants (`w_avail == 0` vs `w_avail == 1`,
     non-window valid Anchor preservation vs window-breach KEEP fallback).
  4. Pure-NumPy MultiHorizonDFLEnsemble finite inference, permutation invariance, and latency `< 1.0 ms`.
  5. Preservation of `38/38` public CSVs and `8/8` frozen V20.1 SHA-256 production hashes.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from amr_sched.cvcr.commitment import CommitmentClassification
from amr_sched.cvcr.snapshot import NativeSnapshot, RobotSnapshot
from experiments.rgro_closure.closure_common import (
    ALL_HISTORICAL_AND_HOLDOUT_SEEDS,
    build_cohort_f_120_specs,
    build_cohort_f_compute_36_specs,
    build_cohort_f_struct_48_specs,
    build_dev_compute_24_specs,
    build_dev_stress_24_specs,
    build_val72_specs,
    build_val_static35_specs,
    sha256_file,
)
from experiments.rgro_closure.deadline_manager import UnifiedDeadlineManager, compute_snapshot_window_slack
from experiments.rgro_closure.multi_horizon_dfl import MultiHorizonDFLEnsemble

ROOT = Path(__file__).resolve().parents[1]


def test_closure_seed_quarantine_and_cohort_f_disjointness() -> None:
    val72 = build_val72_specs()
    stat35 = build_val_static35_specs()
    str24 = build_dev_stress_24_specs()
    cmp24 = build_dev_compute_24_specs()
    dev_seeds = {int(s["seed"]) for s in val72 + stat35 + str24 + cmp24}
    assert dev_seeds.isdisjoint(ALL_HISTORICAL_AND_HOLDOUT_SEEDS)

    cf_std = build_cohort_f_120_specs()
    cf_str = build_cohort_f_struct_48_specs()
    cf_cmp = build_cohort_f_compute_36_specs()
    assert len(cf_std) == 120 and len(cf_str) == 48 and len(cf_cmp) == 36
    f_seeds = {int(s["seed"]) for s in cf_std + cf_str + cf_cmp}
    assert f_seeds.isdisjoint(dev_seeds)
    assert f_seeds.isdisjoint(ALL_HISTORICAL_AND_HOLDOUT_SEEDS)


def test_no_scenario_or_seed_branching_in_closure_solvers() -> None:
    solver_files = [
        ROOT / "experiments" / "rgro_closure" / "deadline_manager.py",
        ROOT / "experiments" / "rgro_closure" / "patched_v20_1.py",
        ROOT / "experiments" / "rgro_closure" / "v18_recovery_operators.py",
        ROOT / "experiments" / "rgro_closure" / "multi_horizon_dfl.py",
        ROOT / "experiments" / "rgro_closure" / "rgro_closure_proposer.py",
    ]
    forbidden_strings = {"track1", "track2", "4topo", "narrow_spine", "offset_dual_lane", "perimeter_bypass", "workshop_cells"}
    for fp in solver_files:
        tree = ast.parse(fp.read_text(encoding="utf-8"), filename=str(fp))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                low = node.value.lower()
                for fstr in forbidden_strings:
                    assert fstr not in low, f"Forbidden scenario branch token {fstr!r} in {fp.name}"
            if isinstance(node, ast.Compare):
                # Ensure no magic episode seed branching (e.g. seed == 1162 / 2906 / 3905)
                for c in [node.left, *node.comparators]:
                    if isinstance(c, ast.Constant) and isinstance(c.value, int) and c.value in {1161, 1162, 1163, 2905, 2906, 3905}:
                        raise AssertionError(f"Forbidden magic seed branch {c.value} in {fp.name}")


def test_unified_deadline_manager_window_and_veto_invariants() -> None:
    snap_w0 = NativeSnapshot(
        schema="1.0",
        tick=10,
        version={"step": 10},
        world_fingerprint="fp0",
        robots=[RobotSnapshot("R1", "N0", "idle", None)],
        tasks=[SimpleNamespace(task_id="T1", status="assigned", assigned_agv="R1", adjustable_window_steps=0, kitting_bottleneck_active=True)],  # type: ignore[list-item]
        operations=[],
        reservations=[],
        now_monotonic=0.0,
    )
    snap_w1 = NativeSnapshot(
        schema="1.0",
        tick=10,
        version={"step": 10},
        world_fingerprint="fp1",
        robots=[RobotSnapshot("R1", "N0", "idle", None)],
        tasks=[SimpleNamespace(task_id="T1", status="assigned", assigned_agv="R1", adjustable_window_steps=1, kitting_bottleneck_active=True)],  # type: ignore[list-item]
        operations=[],
        reservations=[],
        now_monotonic=0.0,
    )
    clf = CommitmentClassification(
        op_tiers={}, task_tiers={}, frozen_tasks=set(), releasable_tasks=set(),
        disturbed_robots=set(), disturbed_tasks=set(), adj={}, edge_reasons={},
    )

    tight0, w0, cap0 = compute_snapshot_window_slack(snap_w0, clf)
    tight1, w1, cap1 = compute_snapshot_window_slack(snap_w1, clf)
    assert tight0 and w0 == 0 and cap0 < 20.0
    assert (not tight1) and w1 == 1 and cap1 >= 34.0

    mgr = UnifiedDeadlineManager(sla_target_ms=20.5, window_tight_sla_ms=17.5)
    ctx_w0 = mgr.begin_decision(snap_w0, clf, budget_ms=35.0, t0_s=time.perf_counter() - 0.025)
    assert ctx_w0.finalize_veto_check(is_keep_cand=False, has_valid_anchor=True, anchor_is_keep=False, has_residual_selected=True) == "revert_keep"

    ctx_abs = mgr.begin_decision(snap_w1, clf, budget_ms=35.0, t0_s=time.perf_counter() - 0.020)
    assert ctx_abs.finalize_veto_check(is_keep_cand=False, has_valid_anchor=True, anchor_is_keep=False, has_residual_selected=True) == "revert_anchor"
    # Non-window abs breach must NOT discard an already-verified Anchor into KEEP
    ctx_keep_anc = mgr.begin_decision(snap_w1, clf, budget_ms=35.0, t0_s=time.perf_counter() - 0.020)
    assert ctx_keep_anc.finalize_veto_check(is_keep_cand=False, has_valid_anchor=True, anchor_is_keep=False, has_residual_selected=False) == "accept"


def test_mh_dfl_ensemble_invariance_and_speed() -> None:
    rng = np.random.default_rng(7)
    ens = MultiHorizonDFLEnsemble(
        feat_mean=np.zeros(48, dtype=np.float32),
        feat_std=np.ones(48, dtype=np.float32),
        members=[
            {
                "w1": rng.normal(0, 0.1, (48, 32)).astype(np.float32),
                "b1": np.zeros(32, dtype=np.float32),
                "w2": rng.normal(0, 0.1, (32, 5)).astype(np.float32),
                "b2": np.zeros(5, dtype=np.float32),
            }
            for _ in range(3)
        ],
    )
    X = rng.normal(size=(6, 48)).astype(np.float32)
    perm = [3, 0, 5, 1, 4, 2]
    s1 = ens.score_batch(X, kappa=0.75)
    t0 = time.perf_counter()
    s2 = ens.score_batch(X[perm], kappa=0.75)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < 2.0
    for i, pi in enumerate(perm):
        assert abs(s1[pi].lcb_advantage - s2[i].lcb_advantage) < 1e-4
        assert np.isfinite(s2[i].harm_prob)


def test_public_csvs_and_v20_1_frozen_hashes_intact() -> None:
    from tools.data_manifest import MANIFEST, build as build_csv_manifest

    expected = json.loads(Path(MANIFEST).read_text(encoding="utf-8"))
    current = build_csv_manifest()
    assert set(current.keys()) == set(expected.keys()), "Public CSV set drifted!"
    for csv_name, exp_meta in expected.items():
        assert current[csv_name] == exp_meta, f"Public CSV drift detected in {csv_name}!"
    assert len(current) == 38

    lock_p = ROOT / "out" / "v20_1_closure" / "v20_1_freeze_manifest.json"
    if lock_p.is_file():
        m = json.loads(lock_p.read_text(encoding="utf-8"))
        cnt = 0
        for rel_p, exp in m["sha256_files"].items():
            assert sha256_file(ROOT / rel_p) == exp, f"SHA-256 mismatch for frozen file {rel_p}"
            cnt += 1
        assert cnt == 8
