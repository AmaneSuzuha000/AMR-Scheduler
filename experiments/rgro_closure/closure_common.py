"""Shared quarantine, Cohort F suite builders, deterministic compute jitter, and stats
for `IF-CORE-RGRO-CLOSURE-01`.

Formal Quarantine Hierarchy:
  - Historical Diagnostic Archives (ZERO DEV training/tuning): Cohort A, B, C, D, D-Struct, Cohort E, E-Struct, E-Compute.
  - Quarantined DEV Splits (Tasks 1-4 ONLY): TRAIN (4-Topo 16101..16304, Workshop 1001..1015), VAL-72, VAL-Static-35, DEV-Stress-24.
  - Brand-New Unseen Confirmation Cohorts (Task 5 Single-Shot ONLY after SHA-256 lock):
      * Cohort F (120 eps): 4-Topo (107101..107302) + Workshop (10901..10906)
      * Cohort F-Struct (48 eps): (11901..11912) x 4 OOD structural families
      * Cohort F-Compute (36 eps): (127101, 127201, 127301) + (12901..12903) under active CPU jitter
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from experiments.rgro.rgro_common import (
    ALL_HISTORICAL_AND_HOLDOUT_SEEDS,
    COHORT_E_ALL_SEEDS,
    DEV_STRESS_SEEDS,
    _build_structural_stress_specs,
    bootstrap_ci_mean,
    build_dev_stress_24_specs,
    build_dev_train_specs,
    cvar95,
    paired_stats,
    pct,
    rollout_multi_horizon,
    sha256_file,
    summarize_rows,
)
from experiments.train_v19_bc import (
    TRAIN_4TOPO_SEEDS,
    TRAIN_WORKSHOP_SEEDS,
    VAL_4TOPO_SEEDS,
    VAL_WORKSHOP_SEEDS,
    _build_4topo_episode_spec,
    _build_workshop_episode_spec,
)
from experiments.v20_audit.common import build_val72_specs, build_val_static35_specs
from experiments.v18_confirmation_scenarios import LAYOUTS as TOPO4_LAYOUTS, REGIMES as TOPO4_REGIMES
from tools.benchmarks.compare_workshop_algorithms import REGIMES as WORKSHOP_REGIMES

OUT_CLOSURE_DIR = ROOT / "out" / "rgro_closure"

# Brand-new unseen Cohort F / F-Struct / F-Compute seeds (Task 5 Single-Shot ONLY)
COHORT_F_4TOPO_SEEDS: tuple[int, ...] = (107101, 107102, 107201, 107202, 107301, 107302)
COHORT_F_WORKSHOP_SEEDS: tuple[int, ...] = (10901, 10902, 10903, 10904, 10905, 10906)
COHORT_F_STRUCT_SEEDS: tuple[int, ...] = tuple(range(11901, 11913))
COHORT_F_COMPUTE_4TOPO_SEEDS: tuple[int, ...] = (127101, 127201, 127301)
COHORT_F_COMPUTE_WS_SEEDS: tuple[int, ...] = (12901, 12902, 12903)

COHORT_F_ALL_SEEDS: frozenset[int] = frozenset(
    set(COHORT_F_4TOPO_SEEDS)
    | set(COHORT_F_WORKSHOP_SEEDS)
    | set(COHORT_F_STRUCT_SEEDS)
    | set(COHORT_F_COMPUTE_4TOPO_SEEDS)
    | set(COHORT_F_COMPUTE_WS_SEEDS)
)

ALL_PRIOR_AND_DOWNGRADED_HOLDOUTS: frozenset[int] = frozenset(
    set(ALL_HISTORICAL_AND_HOLDOUT_SEEDS) | set(COHORT_E_ALL_SEEDS)
)


def assert_closure_dev_quarantine(
    seeds: list[int] | tuple[int, ...] | set[int] | frozenset[int],
    *,
    label: str,
) -> None:
    """Assert seeds are strictly DEV (zero overlap with Cohort A-E, G1/Blind, OR Cohort F)."""
    s = {int(x) for x in seeds}
    forbidden = set(ALL_PRIOR_AND_DOWNGRADED_HOLDOUTS) | set(COHORT_F_ALL_SEEDS)
    bad = sorted(s & forbidden)
    if bad:
        raise RuntimeError(f"CLOSURE DEV QUARANTINE VIOLATION in {label}: {bad[:12]}")


def assert_cohort_f_disjoint() -> None:
    """Verify Cohort F / F-Struct / F-Compute have ZERO overlap with any DEV or historical Cohort A-E."""
    dev_all = (
        set(TRAIN_4TOPO_SEEDS)
        | set(TRAIN_WORKSHOP_SEEDS)
        | set(VAL_4TOPO_SEEDS)
        | set(VAL_WORKSHOP_SEEDS)
        | set(DEV_STRESS_SEEDS)
    )
    prior_all = set(ALL_PRIOR_AND_DOWNGRADED_HOLDOUTS) | dev_all
    overlap = sorted(set(COHORT_F_ALL_SEEDS) & prior_all)
    if overlap:
        raise RuntimeError(f"COHORT F DISJOINTNESS VIOLATION: {overlap[:12]}")

    f_std = set(COHORT_F_4TOPO_SEEDS) | set(COHORT_F_WORKSHOP_SEEDS)
    f_str = set(COHORT_F_STRUCT_SEEDS)
    f_cmp = set(COHORT_F_COMPUTE_4TOPO_SEEDS) | set(COHORT_F_COMPUTE_WS_SEEDS)
    if (f_std & f_str) or (f_std & f_cmp) or (f_str & f_cmp):
        raise RuntimeError("Internal overlap among Cohort F / F-Struct / F-Compute seeds!")


def build_dev_compute_24_specs() -> list[dict[str, Any]]:
    """Build 24 quarantined DEV compute-jitter stress specs (strictly using VAL + DEV-Stress seeds)."""
    specs: list[dict[str, Any]] = []
    val72 = build_val72_specs()
    # 12 VAL 4-Topo + 12 VAL Workshop (multi_wave_surge & execution_blockage & kitting_windows)
    s4 = [s for s in val72 if s.get("suite_group") == "4topo" and s.get("regime") in ("standard", "window_bottleneck")][:12]
    sw = [s for s in val72 if s.get("suite_group") in ("track1", "track2") and s.get("regime") != "nominal"][:12]
    for sp in s4 + sw:
        sp_c = dict(sp)
        sp_c["episode_key"] = f"devcomp_{sp['episode_key']}"
        sp_c["suite_group"] = f"DEV-Compute-{sp.get('suite_group', '')}"
        specs.append(sp_c)
    assert_closure_dev_quarantine([int(s["seed"]) for s in specs], label="DEV_Compute24")
    return specs


def build_cohort_f_120_specs() -> list[dict[str, Any]]:
    """Build 120 brand-new unseen standard Cohort F episodes (Task 5 single-shot ONLY)."""
    assert_cohort_f_disjoint()
    specs: list[dict[str, Any]] = []
    for s in COHORT_F_4TOPO_SEEDS:
        for lay in TOPO4_LAYOUTS:
            for reg in TOPO4_REGIMES:
                sp = _build_4topo_episode_spec(lay, int(s), reg)
                sp["suite_group"] = "4topo"
                specs.append(sp)
    for tr in ("track1", "track2"):
        for s in COHORT_F_WORKSHOP_SEEDS:
            for reg in WORKSHOP_REGIMES:
                sp = _build_workshop_episode_spec(tr, int(s), reg)
                sp["suite_group"] = tr
                specs.append(sp)
    assert len(specs) == 120
    return specs


def build_cohort_f_struct_48_specs() -> list[dict[str, Any]]:
    """Build 48 brand-new unseen OOD structural Cohort F-Struct episodes (Task 5 single-shot ONLY)."""
    assert_cohort_f_disjoint()
    specs = _build_structural_stress_specs(COHORT_F_STRUCT_SEEDS, prefix="fstruct", group_prefix="F-Struct")
    assert len(specs) == 48
    return specs


def build_cohort_f_compute_36_specs() -> list[dict[str, Any]]:
    """Build 36 brand-new unseen compute-jitter Cohort F-Compute episodes (Task 5 single-shot ONLY)."""
    assert_cohort_f_disjoint()
    specs: list[dict[str, Any]] = []
    for s in COHORT_F_COMPUTE_4TOPO_SEEDS:
        for lay in TOPO4_LAYOUTS:
            for reg in ("standard", "window_bottleneck"):
                sp = _build_4topo_episode_spec(lay, int(s), reg)
                sp["suite_group"] = "F-Compute-4Topo"
                sp["episode_key"] = f"fcomp_{sp['episode_key']}"
                specs.append(sp)
    for tr in ("track1", "track2"):
        for s in COHORT_F_COMPUTE_WS_SEEDS:
            for reg in ("multi_wave_surge", "execution_blockage"):
                sp = _build_workshop_episode_spec(tr, int(s), reg)
                sp["suite_group"] = f"F-Compute-{tr}"
                sp["episode_key"] = f"fcomp_{sp['episode_key']}"
                specs.append(sp)
    assert len(specs) == 36
    return specs


def deterministic_compute_jitter_ms(tick: int, seed: int) -> float:
    """Reproducible deterministic host compute contention profile (+12/+16 ms bursts)."""
    h = (int(tick) * 2654435761 + int(seed) * 40503) & 0xFFFFFFFF
    if (h % 3) == 0:
        return 12.0
    if (h % 7) == 0:
        return 16.0
    return 0.0
