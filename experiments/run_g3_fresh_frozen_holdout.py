"""Gate G3: Single-Session Uncontended Fresh 120-Episode Frozen Blind Holdout (`EXP-11-G3`).

Evaluates ALL 12 frozen core methods in ONE single continuous uncontended wall-clock run
(`clock_mode="wall"`, `include_decision_latencies=True`) on a brand-new, never-touched
120-episode blind holdout strictly quarantined from ALL prior dev/train/val/old-blind seeds:
  - 72 episodes: `4-Topo` (`FRESH_4TOPO_SEEDS = (27101, 27102, 27201, 27202, 27301, 27302)` x 4 layouts x 3 regimes)
  - 24 episodes: `Track 1` (`FRESH_WORKSHOP_SEEDS = (2901..2906)` x 4 regimes, 62-node Rail Workshop, 6 AMRs)
  - 24 episodes: `Track 2` (`FRESH_WORKSHOP_SEEDS = (2901..2906)` x 4 regimes, 55-node 3-AMR High-Density Flow)

Methods Evaluated (12 Frozen Core Methods):
  1. `V19_ADAPTIVE_RL` (`CoRe-Sched` — frozen after Phase A DEV/VAL audit)
  2. `V19_HETERO_RL` (`AHG-RL Full` ablation)
  3. `V19_STAGED_RL` (`AHG-RL Staged` ablation)
  4. `V18_EA_JOINT` (`V18-EA` strong search + rule baseline)
  5. `rule_v10` (`Rule-V10` calibrated multi-expert rule portfolio)
  6. `IJCAI26_PGMPO` (Choi et al., IJCAI 2026 — pure FJSP neural baseline)
  7. `EJOR26_Wu_GRU_Attn` (Wu et al., EJOR 2026 — logistics neural baseline)
  8. `RAL26_DetourDelta` (Hirayama et al., IEEE RA-L 2026 — online MAPD baseline)
  9. `SPGA_AMR` (Jia Li et al., COR 2026 — dynamic FJSP-AGV parallel GA)
  10. `IJPR25_DFJSPT_HMARL` (Wang et al., IJPR 2025 — 3-level HMARL for DFJSP-T)
  11. `ASC26_MA_TRANS` (Shen, Zhang & Jin, ASC 193, 2026 — Transformer + Resource GAT Dual-Agent PPO)
  12. `FGCS26_REN_GNN` (Ren, Liu, Huang & Cao, FGCS 182, 2026 — 3D-DG GNN + DRL)

Reports BOTH:
  - **Pooled Wall-Clock P95 / P99** (computed over ALL concatenated raw decision latencies across 120 episodes)
  - **Mean Episode-Level P95** (explicitly distinguished for methodological transparency)
  - Full paired statistics vs `CoRe-Sched`: paired Delta J_A (%), 95% Bootstrap CI, Wilcoxon p, W/T/L,
    100%-completion rate, pure physical H1 (H2=0), mutually-completed paired H1, churn, and per-suite splits.

Writes:
  - `out/v19_scientific_audit/exp11_g3_fresh_frozen_holdout.{json,md}`
  - `results/g3_fresh_blind_holdout.csv` (clean public per-episode ledger: 120 x 12 = 1,440 rows)
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.baselines.policy_registry import run_discrete_episode
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

FRESH_4TOPO_SEEDS: tuple[int, ...] = (27101, 27102, 27201, 27202, 27301, 27302)
FRESH_WORKSHOP_SEEDS: tuple[int, ...] = (2901, 2902, 2903, 2904, 2905, 2906)

G3_CORE_12_POLICIES: tuple[tuple[str, str], ...] = (
    ("V19_ADAPTIVE_RL", "CoRe-Sched"),
    ("V19_HETERO_RL", "AHG-RL Full"),
    ("V19_STAGED_RL", "AHG-RL Staged"),
    ("V18_EA_JOINT", "V18-EA"),
    ("rule_v10", "Rule-V10"),
    ("IJCAI26_PGMPO", "IJCAI26-PGMPO"),
    ("EJOR26_Wu_GRU_Attn", "EJOR26-Wu-GRU"),
    ("RAL26_DetourDelta", "RAL26-DetourDelta"),
    ("SPGA_AMR", "COR26-SPGA"),
    ("IJPR25_DFJSPT_HMARL", "IJPR25-DFJSP-T"),
    ("ASC26_MA_TRANS", "ASC26-MA-Trans"),
    ("FGCS26_REN_GNN", "FGCS26-Ren-GNN"),
)


def assert_fresh_blind_quarantine() -> None:
    """Programmatically assert ZERO overlap between Gate G3 fresh blind seeds and ALL historical seeds."""
    historical: set[int] = (
        set(FORBIDDEN_BLIND_SEEDS)
        | set(FORBIDDEN_DEV_SEEDS)
        | set(TRAIN_4TOPO_SEEDS)
        | set(TRAIN_WORKSHOP_SEEDS)
        | set(VAL_4TOPO_SEEDS)
        | set(VAL_WORKSHOP_SEEDS)
        | set(RL_4TOPO_SEEDS)
        | set(RL_WORKSHOP_SEEDS)
    )
    fresh = set(FRESH_4TOPO_SEEDS) | set(FRESH_WORKSHOP_SEEDS)
    overlap = sorted(fresh & historical)
    if overlap:
        raise RuntimeError(f"GATE G3 FRESH SEED QUARANTINE VIOLATION! Overlap: {overlap}")


def build_fresh_120_specs() -> list[dict[str, Any]]:
    assert_fresh_blind_quarantine()
    specs: list[dict[str, Any]] = []
    for s in FRESH_4TOPO_SEEDS:
        for lay in TOPO4_LAYOUTS:
            for reg in TOPO4_REGIMES:
                sp = _build_4topo_episode_spec(lay, s, reg)
                sp["suite_group"] = "4topo"
                specs.append(sp)
    for track, grp in (("track1", "track1"), ("track2", "track2")):
        for s in FRESH_WORKSHOP_SEEDS:
            for reg in WORKSHOP_REGIMES:
                sp = _build_workshop_episode_spec(track, s, reg)
                sp["suite_group"] = grp
                specs.append(sp)
    assert len(specs) == 120, f"Expected 120 fresh blind episodes, got {len(specs)}"
    return specs


def _h1(r: dict[str, Any]) -> float:
    return float(r["cmax"] + r["task_tardiness_cost"] + r["op_tardiness_cost"])


def _comp100(r: dict[str, Any]) -> bool:
    return bool(float(r.get("task_comp", 0.0)) >= 0.9999 and float(r.get("prod_comp", 0.0)) >= 0.9999)


def _paired_stats(core_rows: list[dict[str, Any]], base_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cb = {r["episode_key"]: r for r in core_rows}
    bb = {r["episode_key"]: r for r in base_rows}
    keys = sorted(set(cb) & set(bb))
    c_ja = np.array([float(cb[k]["net_obj"]) for k in keys], dtype=np.float64)
    b_ja = np.array([float(bb[k]["net_obj"]) for k in keys], dtype=np.float64)
    diffs = b_ja - c_ja  # positive => CoRe-Sched lower (better)

    mean_c, mean_b = float(c_ja.mean()), float(b_ja.mean())
    pct_imp = 100.0 * (mean_b - mean_c) / max(1e-6, mean_b)

    rng = np.random.default_rng(202603)
    boot_pct = np.empty(5000, dtype=np.float64)
    n = len(keys)
    for bi in range(5000):
        idx = rng.integers(0, n, size=n)
        bc = c_ja[idx].mean()
        bb_m = b_ja[idx].mean()
        boot_pct[bi] = 100.0 * (bb_m - bc) / max(1e-6, bb_m)
    ci_lo, ci_hi = float(np.percentile(boot_pct, 2.5)), float(np.percentile(boot_pct, 97.5))

    nz = diffs[np.abs(diffs) > 1e-6]
    if len(nz) >= 5:
        _, p_val = wilcoxon(nz, alternative="two-sided")
        p_val = float(p_val)
    else:
        p_val = 1.0

    wins = int(np.sum(diffs > 1e-4))
    losses = int(np.sum(diffs < -1e-4))
    ties = int(n - wins - losses)

    # Mutually-completed paired pure physical H1
    mut_keys = [k for k in keys if _comp100(cb[k]) and _comp100(bb[k])]
    if mut_keys:
        mc_h1 = float(np.mean([_h1(cb[k]) for k in mut_keys]))
        mb_h1 = float(np.mean([_h1(bb[k]) for k in mut_keys]))
        mut_pct = 100.0 * (mb_h1 - mc_h1) / max(1e-6, mb_h1)
    else:
        mc_h1, mb_h1, mut_pct = None, None, None

    return {
        "n_paired": n,
        "core_mean_ja": round(mean_c, 2),
        "base_mean_ja": round(mean_b, 2),
        "delta_ja_pct": round(pct_imp, 2),
        "boot_ci95_pct": [round(ci_lo, 2), round(ci_hi, 2)],
        "wilcoxon_p": p_val,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mutually_completed_n": len(mut_keys),
        "mut_core_h1": round(mc_h1, 2) if mc_h1 is not None else None,
        "mut_base_h1": round(mb_h1, 2) if mb_h1 is not None else None,
        "mut_h1_delta_pct": round(mut_pct, 2) if mut_pct is not None else None,
    }


def _summarize_policy(rows: list[dict[str, Any]], all_lats: list[float]) -> dict[str, Any]:
    ja = np.array([float(r["net_obj"]) for r in rows], dtype=np.float64)
    comp = [r for r in rows if _comp100(r)]
    aud_ok = sum(1 for r in rows if bool(r.get("audit_passed", False)))
    ep_p95s = [float(r["decision_ms_p95"]) for r in rows]
    ep_p99s = [float(r["decision_ms_p99"]) for r in rows]
    churns = [int(r["churn"]) for r in rows]

    by_suite: dict[str, Any] = {}
    for sg in ("4topo", "track1", "track2"):
        sub = [r for r in rows if r["suite_group"] == sg]
        sub_c = [r for r in sub if _comp100(r)]
        by_suite[sg] = {
            "n": len(sub),
            "mean_ja": round(float(np.mean([r["net_obj"] for r in sub])), 2),
            "median_ja": round(float(np.median([r["net_obj"] for r in sub])), 2),
            "comp100_n": len(sub_c),
            "comp100_pct": round(100.0 * len(sub_c) / max(1, len(sub)), 2),
            "mean_h1_comp": round(float(np.mean([_h1(r) for r in sub_c])), 2) if sub_c else None,
            "mean_ep_p95_ms": round(float(np.mean([r["decision_ms_p95"] for r in sub])), 2),
        }

    return {
        "n_episodes": len(rows),
        "audit_passed_n": aud_ok,
        "mean_ja": round(float(ja.mean()), 2),
        "median_ja": round(float(np.median(ja)), 2),
        "std_ja": round(float(ja.std()), 2),
        "comp100_n": len(comp),
        "comp100_pct": round(100.0 * len(comp) / max(1, len(rows)), 2),
        "mean_h1_all": round(float(np.mean([_h1(r) for r in rows])), 2),
        "mean_h1_comp": round(float(np.mean([_h1(r) for r in comp])), 2) if comp else None,
        "mean_churn": round(float(np.mean(churns)), 2),
        "total_decisions": len(all_lats),
        "pooled_wall_p50_ms": round(float(np.percentile(all_lats, 50)), 2) if all_lats else 0.0,
        "pooled_wall_p95_ms": round(float(np.percentile(all_lats, 95)), 2) if all_lats else 0.0,
        "pooled_wall_p99_ms": round(float(np.percentile(all_lats, 99)), 2) if all_lats else 0.0,
        "mean_episode_p95_ms": round(float(np.mean(ep_p95s)), 2),
        "mean_episode_p99_ms": round(float(np.mean(ep_p99s)), 2),
        "by_suite": by_suite,
    }


def run_g3_fresh_frozen_holdout() -> dict[str, Any]:
    t0 = time.perf_counter()
    specs = build_fresh_120_specs()
    print(f"[Gate G3] Starting single-session uncontended fresh 120-episode blind holdout across {len(G3_CORE_12_POLICIES)} frozen methods!", flush=True)

    # Warm up Rust & NumPy once on a dummy call outside timing so cold-import jitter doesn't pollute policy 1
    _ = run_discrete_episode(
        "rule_v10",
        copy.deepcopy(specs[0]["scenario_tuple"]),
        seed=specs[0]["seed"],
        max_steps=80,
        clock_mode="wall",
    )

    all_rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    all_lats_by_policy: dict[str, list[float]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    ckpt_shas: dict[str, str | None] = {}

    for p_idx, (pol_key, label) in enumerate(G3_CORE_12_POLICIES, 1):
        pt0 = time.perf_counter()
        pol_rows: list[dict[str, Any]] = []
        pol_lats: list[float] = []
        for ep_i, sp in enumerate(specs, 1):
            rec = run_discrete_episode(
                pol_key,
                copy.deepcopy(sp["scenario_tuple"]),
                seed=int(sp["seed"]),
                max_steps=int(sp["max_steps"]),
                p2_cfg=copy.deepcopy(sp["p2_cfg"]),
                episode_key=sp["episode_key"],
                clock_mode="wall",
                include_decision_latencies=True,
            )
            lats = [float(v) for v in rec.pop("decision_latencies_ms", [])]
            pol_lats.extend(lats)
            rec["suite_group"] = sp["suite_group"]
            rec["label"] = label
            rec["h1"] = round(_h1(rec), 2)
            rec["comp100"] = _comp100(rec)
            # Drop bulky scheduled_obstacles from summary JSON
            rec.pop("scheduled_obstacles", None)
            pol_rows.append(rec)
            if ep_i == 1:
                ckpt_shas[pol_key] = rec.get("checkpoint_sha256")
            if ep_i % 40 == 0 or ep_i == len(specs):
                cur_ja = float(np.mean([r["net_obj"] for r in pol_rows]))
                cur_c = sum(1 for r in pol_rows if r["comp100"])
                cur_p95 = float(np.percentile(pol_lats, 95)) if pol_lats else 0.0
                print(
                    f"[Gate G3] ({p_idx:02d}/{len(G3_CORE_12_POLICIES)}) {label:18s} | {ep_i:3d}/120 | "
                    f"J_A={cur_ja:8.1f} | Comp={cur_c:3d}/{ep_i} | PoolP95={cur_p95:5.2f}ms ({time.perf_counter()-pt0:.1f}s)",
                    flush=True,
                )
        all_rows_by_policy[pol_key] = pol_rows
        all_lats_by_policy[pol_key] = pol_lats
        summaries[pol_key] = _summarize_policy(pol_rows, pol_lats)
        summaries[pol_key]["label"] = label
        summaries[pol_key]["checkpoint_sha256"] = ckpt_shas[pol_key]

    paired_vs_core: dict[str, dict[str, Any]] = {}
    core_rows = all_rows_by_policy["V19_ADAPTIVE_RL"]
    for pol_key, label in G3_CORE_12_POLICIES:
        if pol_key == "V19_ADAPTIVE_RL":
            continue
        st = _paired_stats(core_rows, all_rows_by_policy[pol_key])
        st["label"] = label
        paired_vs_core[pol_key] = st

    payload = {
        "experiment": "EXP-11-G3_Single_Session_Fresh_120ep_Frozen_Blind_Holdout",
        "protocol": {
            "clock_mode": "wall",
            "single_continuous_uncontended_session": True,
            "fresh_4topo_seeds": list(FRESH_4TOPO_SEEDS),
            "fresh_workshop_seeds": list(FRESH_WORKSHOP_SEEDS),
            "zero_overlap_verified": True,
            "n_episodes": 120,
            "n_policies": len(G3_CORE_12_POLICIES),
        },
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "summaries": summaries,
        "paired_vs_core_sched": paired_vs_core,
        "rows_by_policy": all_rows_by_policy,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    jp = OUT_DIR / "exp11_g3_fresh_frozen_holdout.json"
    jp.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Export clean public verification CSV ledger (1,440 rows)
    csv_path = RESULTS_DIR / "g3_fresh_blind_holdout.csv"
    csv_fields = [
        "policy",
        "label",
        "episode_key",
        "suite_group",
        "seed",
        "net_obj",
        "h1",
        "cmax",
        "task_tardiness_cost",
        "op_tardiness_cost",
        "unfinished_task_penalty",
        "unfinished_op_penalty",
        "churn_cost",
        "compute_cost",
        "churn",
        "task_comp",
        "prod_comp",
        "comp100",
        "audit_passed",
        "decision_count",
        "mean_decision_ms",
        "decision_ms_p95",
        "decision_ms_p99",
        "checkpoint_sha256",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_fields)
        w.writeheader()
        for pol_key, _ in G3_CORE_12_POLICIES:
            for r in all_rows_by_policy[pol_key]:
                w.writerow({k: r.get(k) for k in csv_fields})

    md = [
        "# Gate G3 (`EXP-11-G3`): Single-Session Uncontended Fresh 120-Episode Frozen Blind Holdout",
        "",
        f"- **Fresh Quarantined Blind Seeds**: `4-Topo = {list(FRESH_4TOPO_SEEDS)}` (72 ep), `Workshop Track 1 & 2 = {list(FRESH_WORKSHOP_SEEDS)}` (48 ep). Zero overlap with ALL prior dev/train/val/old-blind seeds programmatically verified.",
        f"- **Session Wall-Clock Elapsed**: `{payload['elapsed_s']} s` (1,440 total audited episodes).",
        "",
        "## Main 12-Method Comparison (Pooled Wall-Clock P95/P99 & Mean Episode P95)",
        "",
        "| Method | Mean $J_A$ | Median $J_A$ | Comp (100%) | Pure $H_1$ ($H_2=0$) | Churn | **Pooled P95 (ms)** | **Pooled P99 (ms)** | Mean Ep-P95 (ms) | `CoRe` $\\Delta J_A$ | 95% CI | Wilcoxon $p$ | W/T/L |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for pol_key, label in G3_CORE_12_POLICIES:
        s = summaries[pol_key]
        h1_s = f"{s['mean_h1_comp']:.1f}" if s["mean_h1_comp"] is not None else "N/A"
        if pol_key == "V19_ADAPTIVE_RL":
            md.append(
                f"| **`{label}`** | **{s['mean_ja']:.1f}** | **{s['median_ja']:.1f}** | "
                f"**{s['comp100_n']}/120 ({s['comp100_pct']:.1f}%)** | **{h1_s}** | {s['mean_churn']:.1f} | "
                f"**{s['pooled_wall_p95_ms']:.2f}** | **{s['pooled_wall_p99_ms']:.2f}** | {s['mean_episode_p95_ms']:.2f} | — | — | — | — |"
            )
        else:
            p = paired_vs_core[pol_key]
            p_str = f"{p['wilcoxon_p']:.2e}" if p["wilcoxon_p"] < 0.001 else f"{p['wilcoxon_p']:.4f}"
            md.append(
                f"| `{label}` | {s['mean_ja']:.1f} | {s['median_ja']:.1f} | "
                f"{s['comp100_n']}/120 ({s['comp100_pct']:.1f}%) | {h1_s} | {s['mean_churn']:.1f} | "
                f"{s['pooled_wall_p95_ms']:.2f} | {s['pooled_wall_p99_ms']:.2f} | {s['mean_episode_p95_ms']:.2f} | "
                f"{p['delta_ja_pct']:+.2f}% | [{p['boot_ci95_pct'][0]:+.1f}%, {p['boot_ci95_pct'][1]:+.1f}%] | {p_str} | {p['wins']}/{p['ties']}/{p['losses']} |"
            )

    md.extend([
        "",
        "## Per-Suite Breakdown (`4-Topo` 72 ep, `Track 1` 24 ep, `Track 2` 24 ep)",
        "",
        "| Method | `4-Topo` Mean $J_A$ (Comp) | `Track 1` Mean $J_A$ (Comp) | `Track 2` Mean $J_A$ (Comp) | Mutually-Completed Pairwise $H_1$ (`CoRe` vs Base, $N$) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for pol_key, label in G3_CORE_12_POLICIES:
        bs = summaries[pol_key]["by_suite"]
        if pol_key == "V19_ADAPTIVE_RL":
            mut_s = "—"
        else:
            p = paired_vs_core[pol_key]
            mut_s = (
                f"{p['mut_core_h1']:.1f} vs {p['mut_base_h1']:.1f} ({p['mut_h1_delta_pct']:+.2f}%, $N={p['mutually_completed_n']}$)"
                if p["mut_core_h1"] is not None
                else "N/A"
            )
        md.append(
            f"| `{label}` | {bs['4topo']['mean_ja']:.1f} ({bs['4topo']['comp100_n']}/72) | "
            f"{bs['track1']['mean_ja']:.1f} ({bs['track1']['comp100_n']}/24) | "
            f"{bs['track2']['mean_ja']:.1f} ({bs['track2']['comp100_n']}/24) | {mut_s} |"
        )

    mp = OUT_DIR / "exp11_g3_fresh_frozen_holdout.md"
    mp.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[Gate G3] COMPLETE! Saved {jp}, {mp}, and {csv_path} ({payload['elapsed_s']}s)", flush=True)
    return payload


if __name__ == "__main__":
    run_g3_fresh_frozen_holdout()
