"""Gate G5 (`EXP-13`): Untouched Pristine 120-Episode Frozen Blind Holdout (Cohort B).

Evaluates ALL 12 frozen core methods in ONE single continuous uncontended wall-clock run
(`clock_mode="wall"`, `include_decision_latencies=True`) strictly ONCE after final code freeze
on a brand-new, never-seen 120-episode holdout (`Cohort B`) disjoint from ALL prior splits:
  - Train seeds:        (101..104, 201..204, 301..304, 601..604)
  - Val seeds:          (16701..16704, 1161..1163)
  - RL seeds:           (9101..9108, 701..704)
  - G1 / Historical:    FORBIDDEN_BLIND_SEEDS, FORBIDDEN_DEV_SEEDS
  - Cohort A (Gate G3): (27101, 27102, 27201, 27202, 27301, 27302, 2901..2906)
  - Cohort B (Gate G5): 4-Topo = (37101, 37102, 37201, 37202, 37301, 37302) x 4 layouts x 3 regimes = 72 ep
                        Track 1 & Track 2 = (3901..3906) x 4 regimes x 2 tracks = 48 ep

Writes:
  - `out/v19_scientific_audit/exp13_g5_pristine_blind_holdout.{json,md}`
  - `results/g5_pristine_blind_holdout.csv` (120 x 12 = 1,440 audited rows)
"""

from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_MOD_DIR = ROOT / "_scratch"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.baselines.policy_registry import run_discrete_episode
from experiments.run_g3_fresh_frozen_holdout import (
    FRESH_4TOPO_SEEDS,
    FRESH_WORKSHOP_SEEDS,
    G3_CORE_12_POLICIES,
)
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
from tools.benchmarks.compare_workshop_algorithms import REGIMES as WORKSHOP_REGIMES
from experiments.v18_confirmation_scenarios import LAYOUTS as TOPO4_LAYOUTS, REGIMES as TOPO4_REGIMES

OUT_DIR = ROOT / "out" / "v19_scientific_audit"
RESULTS_DIR = ROOT / "results"

PRISTINE_4TOPO_SEEDS: tuple[int, ...] = (37101, 37102, 37201, 37202, 37301, 37302)
PRISTINE_WORKSHOP_SEEDS: tuple[int, ...] = (3901, 3902, 3903, 3904, 3905, 3906)


def assert_pristine_blind_quarantine() -> None:
    """Programmatically assert ZERO overlap between Cohort B and ALL prior splits."""
    prior: set[int] = (
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
    pristine = set(PRISTINE_4TOPO_SEEDS) | set(PRISTINE_WORKSHOP_SEEDS)
    overlap = sorted(pristine & prior)
    if overlap:
        raise RuntimeError(f"GATE G5 PRISTINE BLIND QUARANTINE VIOLATION! Overlap: {overlap}")


def build_pristine_120_specs() -> list[dict[str, Any]]:
    assert_pristine_blind_quarantine()
    specs: list[dict[str, Any]] = []
    for s in PRISTINE_4TOPO_SEEDS:
        for lay in TOPO4_LAYOUTS:
            for reg in TOPO4_REGIMES:
                sp = _build_4topo_episode_spec(lay, s, reg)
                sp["suite_group"] = "4topo"
                specs.append(sp)
    for track, grp in (("track1", "track1"), ("track2", "track2")):
        for s in PRISTINE_WORKSHOP_SEEDS:
            for reg in WORKSHOP_REGIMES:
                sp = _build_workshop_episode_spec(track, s, reg)
                sp["suite_group"] = grp
                specs.append(sp)
    assert len(specs) == 120, f"Expected 120 pristine blind episodes, got {len(specs)}"
    return specs


def _h1(r: dict[str, Any]) -> float:
    return float(r["cmax"] + r["task_tardiness_cost"] + r["op_tardiness_cost"])


def _comp100(r: dict[str, Any]) -> bool:
    return bool(float(r.get("task_comp", 0.0)) >= 0.9999 and float(r.get("prod_comp", 0.0)) >= 0.9999)


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = float(q) * float(len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = pos - float(lo)
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _paired_stats(core_rows: list[dict[str, Any]], base_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cb = {r["episode_key"]: r for r in core_rows}
    bb = {r["episode_key"]: r for r in base_rows}
    keys = sorted(set(cb) & set(bb))
    c_ja = np.array([float(cb[k]["net_obj"]) for k in keys], dtype=np.float64)
    b_ja = np.array([float(bb[k]["net_obj"]) for k in keys], dtype=np.float64)
    diff = b_ja - c_ja
    pct = 100.0 * float(np.mean(b_ja) - np.mean(c_ja)) / max(1e-9, float(np.mean(b_ja)))
    if np.all(np.abs(diff) < 1e-9):
        p_val = 1.0
    else:
        p_val = float(wilcoxon(b_ja, c_ja, zero_method="wilcox", alternative="two-sided").pvalue)
    rng = np.random.default_rng(20261124)
    idx = rng.integers(0, len(keys), size=(2000, len(keys)))
    boots = 100.0 * (np.mean(b_ja[idx], axis=1) - np.mean(c_ja[idx], axis=1)) / np.maximum(1e-9, np.mean(b_ja[idx], axis=1))
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    wins = int(np.sum(diff > 1e-6))
    losses = int(np.sum(diff < -1e-6))
    ties = len(keys) - wins - losses

    mut_keys = [k for k in keys if _comp100(cb[k]) and _comp100(bb[k])]
    if mut_keys:
        ch1 = float(np.mean([_h1(cb[k]) for k in mut_keys]))
        bh1 = float(np.mean([_h1(bb[k]) for k in mut_keys]))
        h1_gain = 100.0 * (bh1 - ch1) / max(1e-9, bh1)
    else:
        ch1 = bh1 = h1_gain = 0.0
    return {
        "n": len(keys),
        "delta_ja_pct": round(pct, 2),
        "ci95_lo": round(ci_lo, 2),
        "ci95_hi": round(ci_hi, 2),
        "wilcoxon_p": float(f"{p_val:.4e}"),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mutual_n": len(mut_keys),
        "core_h1_mutual": round(ch1, 1),
        "base_h1_mutual": round(bh1, 1),
        "h1_gain_mutual_pct": round(h1_gain, 2),
    }


def _summarize(rows: list[dict[str, Any]], lats: list[float]) -> dict[str, Any]:
    s_lats = sorted(float(x) for x in lats)
    comp_rows = [r for r in rows if _comp100(r)]
    return {
        "n": len(rows),
        "audit_pass": sum(1 for r in rows if r["audit_passed"]),
        "comp100_count": len(comp_rows),
        "comp100_pct": round(100.0 * len(comp_rows) / max(1, len(rows)), 1),
        "mean_ja": round(float(np.mean([r["net_obj"] for r in rows])), 1),
        "median_ja": round(float(np.median([r["net_obj"] for r in rows])), 1),
        "pure_h1_mean": round(float(np.mean([_h1(r) for r in comp_rows])), 1) if comp_rows else 0.0,
        "churn_mean": round(float(np.mean([r["churn"] for r in rows])), 1),
        "expired_windows": sum(int(r.get("expired_windows", 0)) for r in rows),
        "transaction_rejections": sum(int(r.get("transaction_rejections", 0)) for r in rows),
        "pooled_p95_ms": round(_percentile(s_lats, 0.95), 2),
        "pooled_p99_ms": round(_percentile(s_lats, 0.99), 2),
        "mean_ep_p95_ms": round(float(np.mean([r["decision_ms_p95"] for r in rows])), 2),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    specs = build_pristine_120_specs()
    t_all = time.perf_counter()

    # Warmup outside timed session
    for pol, _ in G3_CORE_12_POLICIES:
        _ = run_discrete_episode(
            pol,
            copy.deepcopy(specs[0]["scenario_tuple"]),
            seed=specs[0]["seed"],
            max_steps=specs[0]["max_steps"],
            use_rust=True,
            p2_cfg=specs[0]["p2_cfg"],
            episode_key=f"warmup_g5_{pol}",
            clock_mode="wall",
            include_decision_latencies=False,
        )

    rows_by_pol: dict[str, list[dict[str, Any]]] = {}
    lats_by_pol: dict[str, list[float]] = {}
    csv_rows: list[dict[str, Any]] = []

    for pol, lbl in G3_CORE_12_POLICIES:
        t0 = time.perf_counter()
        prows: list[dict[str, Any]] = []
        plats: list[float] = []
        for sp in specs:
            res = run_discrete_episode(
                pol,
                copy.deepcopy(sp["scenario_tuple"]),
                seed=sp["seed"],
                max_steps=sp["max_steps"],
                use_rust=True,
                p2_cfg=sp["p2_cfg"],
                episode_key=sp["episode_key"],
                clock_mode="wall",
                include_decision_latencies=True,
            )
            plats.extend(float(x) for x in res.get("decision_latencies_ms", []))
            h1_val = float(res["cmax"] + res["task_tardiness_cost"] + res["op_tardiness_cost"])
            comp100 = bool(float(res.get("task_comp", 0.0)) >= 0.9999 and float(res.get("prod_comp", 0.0)) >= 0.9999)
            row = {
                "policy": pol,
                "label": lbl,
                "episode_key": sp["episode_key"],
                "suite_group": sp["suite_group"],
                "seed": sp["seed"],
                "net_obj": res["net_obj"],
                "h1": round(h1_val, 4),
                "cmax": res["cmax"],
                "task_tardiness_cost": res["task_tardiness_cost"],
                "op_tardiness_cost": res["op_tardiness_cost"],
                "unfinished_task_penalty": res["unfinished_task_penalty"],
                "unfinished_op_penalty": res["unfinished_op_penalty"],
                "churn_cost": res["churn_cost"],
                "compute_cost": res["compute_cost"],
                "churn": res["churn"],
                "task_comp": res["task_comp"],
                "prod_comp": res["prod_comp"],
                "comp100": comp100,
                "audit_passed": res["audit_passed"],
                "decision_count": res["decision_count"],
                "mean_decision_ms": res["mean_decision_ms"],
                "decision_ms_p95": res["decision_ms_p95"],
                "decision_ms_p99": res["decision_ms_p99"],
                "expired_windows": res["expired_windows"],
                "transaction_rejections": res["transaction_rejections"],
                "checkpoint_sha256": res["checkpoint_sha256"],
            }
            prows.append(row)
            csv_rows.append(row)
        rows_by_pol[pol] = prows
        lats_by_pol[pol] = plats
        s = _summarize(prows, plats)
        dt = time.perf_counter() - t0
        print(
            f"[{lbl:17s}] {dt:5.1f}s | {s['comp100_count']:3d}/120 ({s['comp100_pct']:5.1f}%) | "
            f"aud={s['audit_pass']}/120 | JA={s['mean_ja']:7.1f} H1={s['pure_h1_mean']:7.1f} | "
            f"wE={s['expired_windows']} rej={s['transaction_rejections']} | "
            f"PoolP95={s['pooled_p95_ms']:5.2f} P99={s['pooled_p99_ms']:5.2f} EpP95={s['mean_ep_p95_ms']:5.2f}",
            flush=True,
        )

    csv_path = RESULTS_DIR / "g5_pristine_blind_holdout.csv"
    fieldnames = list(csv_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    core_rows = rows_by_pol["V19_ADAPTIVE_RL"]
    summaries: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    per_suite: dict[str, dict[str, Any]] = {}
    for pol, lbl in G3_CORE_12_POLICIES:
        prows = rows_by_pol[pol]
        summaries[pol] = _summarize(prows, lats_by_pol[pol])
        if pol != "V19_ADAPTIVE_RL":
            paired[pol] = _paired_stats(core_rows, prows)
        smap: dict[str, Any] = {}
        for sk in ("4topo", "track1", "track2"):
            sub = [r for r in prows if r["suite_group"] == sk]
            smap[sk] = {
                "n": len(sub),
                "comp100_count": sum(1 for r in sub if _comp100(r)),
                "mean_ja": round(float(np.mean([r["net_obj"] for r in sub])), 1),
                "mean_h1": round(float(np.mean([_h1(r) for r in sub])), 1),
            }
        per_suite[pol] = smap

    elapsed_all = round(time.perf_counter() - t_all, 1)
    out_json = {
        "experiment": "EXP-13-G5-PRISTINE-BLIND-HOLDOUT",
        "pristine_seeds_4topo": list(PRISTINE_4TOPO_SEEDS),
        "pristine_seeds_workshop": list(PRISTINE_WORKSHOP_SEEDS),
        "episodes_per_method": len(specs),
        "total_rows": len(csv_rows),
        "session_wall_seconds": elapsed_all,
        "summary": summaries,
        "paired_vs_core_sched": paired,
        "per_suite": per_suite,
    }
    json_path = OUT_DIR / "exp13_g5_pristine_blind_holdout.json"
    json_path.write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    md = [
        "# Gate G5 (`EXP-13-G5`): Untouched Pristine 120-Episode Frozen Blind Holdout (Cohort B)",
        "",
        f"- **Pristine Quarantined Blind Seeds (Cohort B)**: `4-Topo = {list(PRISTINE_4TOPO_SEEDS)}` (72 ep), `Workshop Track 1 & 2 = {list(PRISTINE_WORKSHOP_SEEDS)}` (48 ep). Zero overlap with train, VAL, RL, G1, or Cohort A seeds verified.",
        f"- **Session Wall-Clock Elapsed**: `{elapsed_all} s` ({len(csv_rows)} audited episodes).",
        "",
        "## Main 12-Method Comparison (Pooled Wall-Clock P95/P99 & Mean Episode P95)",
        "",
        "| Method | Mean $J_A$ | Median $J_A$ | Comp (100%) | Pure $H_1$ ($H_2=0$) | Churn | **Pooled P95 (ms)** | **Pooled P99 (ms)** | Mean Ep-P95 (ms) | `CoRe` $\\Delta J_A$ | 95% CI | Wilcoxon $p$ | W/T/L |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for pol, lbl in G3_CORE_12_POLICIES:
        s = summaries[pol]
        if pol == "V19_ADAPTIVE_RL":
            md.append(
                f"| **`{lbl}`** | **{s['mean_ja']:.1f}** | **{s['median_ja']:.1f}** | "
                f"**{s['comp100_count']}/120 ({s['comp100_pct']:.1f}%)** | **{s['pure_h1_mean']:.1f}** | {s['churn_mean']:.1f} | "
                f"**{s['pooled_p95_ms']:.2f}** | **{s['pooled_p99_ms']:.2f}** | {s['mean_ep_p95_ms']:.2f} | — | — | — | — |"
            )
        else:
            p = paired[pol]
            md.append(
                f"| `{lbl}` | {s['mean_ja']:.1f} | {s['median_ja']:.1f} | "
                f"{s['comp100_count']}/120 ({s['comp100_pct']:.1f}%) | {s['pure_h1_mean']:.1f} | {s['churn_mean']:.1f} | "
                f"{s['pooled_p95_ms']:.2f} | {s['pooled_p99_ms']:.2f} | {s['mean_ep_p95_ms']:.2f} | "
                f"{p['delta_ja_pct']:+.2f}% | [{p['ci95_lo']:+.1f}%, {p['ci95_hi']:+.1f}%] | {p['wilcoxon_p']:.4g} | {p['wins']}/{p['ties']}/{p['losses']} |"
            )
    md.extend([
        "",
        "## Per-Suite Breakdown (`4-Topo` 72 ep, `Track 1` 24 ep, `Track 2` 24 ep)",
        "",
        "| Method | `4-Topo` Mean $J_A$ (Comp) | `Track 1` Mean $J_A$ (Comp) | `Track 2` Mean $J_A$ (Comp) | Mutually-Completed Pairwise $H_1$ (`CoRe` vs Base, $N$) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for pol, lbl in G3_CORE_12_POLICIES:
        sm = per_suite[pol]
        if pol == "V19_ADAPTIVE_RL":
            mut_s = "—"
        else:
            p = paired[pol]
            mut_s = f"{p['core_h1_mutual']:.1f} vs {p['base_h1_mutual']:.1f} ({p['h1_gain_mutual_pct']:+.2f}%, $N={p['mutual_n']}$)"
        md.append(
            f"| `{lbl}` | {sm['4topo']['mean_ja']:.1f} ({sm['4topo']['comp100_count']}/72) | "
            f"{sm['track1']['mean_ja']:.1f} ({sm['track1']['comp100_count']}/24) | "
            f"{sm['track2']['mean_ja']:.1f} ({sm['track2']['comp100_count']}/24) | {mut_s} |"
        )
    (OUT_DIR / "exp13_g5_pristine_blind_holdout.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
