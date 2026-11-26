"""F0: Freeze & Archive V20.0 (`V20_NAJS`) as Official Historical Baseline with Complete Paired Statistics.

Explicitly separates:
  - Historical Regression (`Cohort A` 120 + `Cohort B` 120 = 240 episodes)
  - Independent `V20.0` Confirmation (`Cohort C` 120 episodes, all 14 methods in one session)
and reports paired delta %, Cliff's delta, Cohen's d, 2000-resample bootstrap 95% CI, paired Wilcoxon p,
mutual-completion H1, Pooled P95/P99/Max, and CVaR_95 against EVERY competitor.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v20_audit.common import OUT_AUDIT_DIR
from experiments.v20_audit.run_v20_final_blind_holdout import (
    FREEZE_MANIFEST_PATH,
    verify_freeze_manifest,
)

OUT_CLOSURE_DIR = ROOT / "out" / "v20_1_closure"
RESULTS_DIR = ROOT / "results"


def _load_rows(csv_path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            r = dict(raw)
            for k in (
                "net_obj", "h1", "cmax", "task_tardiness_cost", "op_tardiness_cost",
                "unfinished_task_penalty", "unfinished_op_penalty", "churn_cost", "compute_cost",
                "task_comp", "prod_comp", "mean_decision_ms", "decision_ms_p95", "decision_ms_p99",
            ):
                r[k] = float(r[k])
            for k in ("seed", "churn", "decision_count", "expired_windows", "transaction_rejections", "pdg_alerts"):
                r[k] = int(float(r[k])) if (k in r and r[k] != "") else 0
            r["comp100"] = str(r["comp100"]).lower() == "true"
            r["audit_passed"] = str(r["audit_passed"]).lower() == "true"
            out.setdefault(str(r["policy"]), []).append(r)
    return out


def _cvar(vals: np.ndarray, alpha: float = 0.95) -> float:
    if vals.size == 0:
        return 0.0
    s = np.sort(vals)
    k = max(1, int(np.ceil((1.0 - alpha) * s.size)))
    return float(np.mean(s[-k:]))


def _cliffs_delta(v20: np.ndarray, base: np.ndarray) -> float:
    """Cliff's delta > 0 means `v20` tends to be strictly lower (better) than `base`."""
    # Exact pairwise dominance: P(base > v20) - P(base < v20)
    diff = base[:, None] - v20[None, :]
    gt = float(np.sum(diff > 1e-9))
    lt = float(np.sum(diff < -1e-9))
    return (gt - lt) / float(max(1, v20.size * base.size))


def _paired_full(v20_rows: list[dict[str, Any]], base_rows: list[dict[str, Any]]) -> dict[str, Any]:
    vb = {r["episode_key"]: r for r in v20_rows}
    bb = {r["episode_key"]: r for r in base_rows}
    keys = sorted(set(vb) & set(bb))
    v_ja = np.array([vb[k]["net_obj"] for k in keys], dtype=np.float64)
    b_ja = np.array([bb[k]["net_obj"] for k in keys], dtype=np.float64)
    diffs = b_ja - v_ja  # positive => V20 lower (better)

    mean_v, mean_b = float(np.mean(v_ja)), float(np.mean(b_ja))
    pct = 100.0 * (mean_b - mean_v) / max(1e-9, mean_b)
    p_less = (
        float(wilcoxon(diffs, zero_method="wilcox", alternative="greater").pvalue)
        if np.any(np.abs(diffs) > 1e-9)
        else 1.0
    )
    p_two = (
        float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
        if np.any(np.abs(diffs) > 1e-9)
        else 1.0
    )
    sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    cohen_d = float(np.mean(diffs) / sd_diff) if sd_diff > 1e-9 else 0.0
    cliffs = _cliffs_delta(v_ja, b_ja)

    rng = np.random.default_rng(20261125)
    boots = np.empty(2000, dtype=np.float64)
    for b in range(2000):
        idx = rng.integers(0, len(keys), size=len(keys))
        bm = float(np.mean(b_ja[idx]))
        boots[b] = 100.0 * (bm - float(np.mean(v_ja[idx]))) / max(1e-9, bm)
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    mut = [k for k in keys if vb[k]["comp100"] and bb[k]["comp100"]]
    v_h1_mut = float(np.mean([vb[k]["h1"] for k in mut])) if mut else 0.0
    b_h1_mut = float(np.mean([bb[k]["h1"] for k in mut])) if mut else 0.0
    h1_gain = 100.0 * (b_h1_mut - v_h1_mut) / max(1e-9, b_h1_mut) if mut else 0.0

    return {
        "n": len(keys),
        "v20_mean_ja": round(mean_v, 2),
        "base_mean_ja": round(mean_b, 2),
        "delta_ja_pct": round(pct, 2),
        "ci95_lo_pct": round(ci_lo, 2),
        "ci95_hi_pct": round(ci_hi, 2),
        "cohen_d_paired": round(cohen_d, 4),
        "cliffs_delta": round(cliffs, 4),
        "wilcoxon_p_one_sided": float(f"{p_less:.4e}"),
        "wilcoxon_p_two_sided": float(f"{p_two:.4e}"),
        "wins": int(np.sum(diffs > 1e-6)),
        "ties": int(np.sum(np.abs(diffs) <= 1e-6)),
        "losses": int(np.sum(diffs < -1e-6)),
        "mutual_comp_n": len(mut),
        "v20_h1_mutual": round(v_h1_mut, 2),
        "base_h1_mutual": round(b_h1_mut, 2),
        "h1_gain_mutual_pct": round(h1_gain, 2),
    }


def _summarize_group(rows: list[dict[str, Any]], lats: list[float] | None = None) -> dict[str, Any]:
    ja = np.array([r["net_obj"] for r in rows], dtype=np.float64)
    h1 = np.array([r["h1"] for r in rows], dtype=np.float64)
    comp = [r for r in rows if r["comp100"]]
    sl = sorted(float(x) for x in (lats or []))
    by_suite = {}
    for sk in ("4topo", "track1", "track2"):
        sub = [r for r in rows if r["suite_group"] == sk]
        sub_ja = np.array([r["net_obj"] for r in sub], dtype=np.float64) if sub else np.zeros(0)
        by_suite[sk] = {
            "n": len(sub),
            "comp100": sum(1 for r in sub if r["comp100"]),
            "mean_ja": round(float(np.mean(sub_ja)), 2) if sub else 0.0,
            "median_ja": round(float(np.median(sub_ja)), 2) if sub else 0.0,
            "cvar95_ja": round(_cvar(sub_ja, 0.95), 2) if sub else 0.0,
        }
    return {
        "n": len(rows),
        "audit_pass": sum(1 for r in rows if r["audit_passed"]),
        "comp100_count": len(comp),
        "comp100_pct": round(100.0 * len(comp) / max(1, len(rows)), 2),
        "mean_ja": round(float(np.mean(ja)), 2),
        "median_ja": round(float(np.median(ja)), 2),
        "cvar95_ja": round(_cvar(ja, 0.95), 2),
        "pure_h1_mean": round(float(np.mean(h1)), 2),
        "churn_mean": round(float(np.mean([r["churn"] for r in rows])), 2),
        "expired_windows": sum(int(r["expired_windows"]) for r in rows),
        "transaction_rejections": sum(int(r["transaction_rejections"]) for r in rows),
        "pooled_p95_ms": round(float(np.percentile(sl, 95)), 3) if sl else None,
        "pooled_p99_ms": round(float(np.percentile(sl, 99)), 3) if sl else None,
        "max_decision_ms": round(float(sl[-1]), 3) if sl else None,
        "overrun_35ms_rate": round(float(np.mean([1.0 if x > 35.0 else 0.0 for x in sl])), 5) if sl else None,
        "by_suite": by_suite,
    }


def run_f0_archive() -> dict[str, Any]:
    OUT_CLOSURE_DIR.mkdir(parents=True, exist_ok=True)
    freeze_man = verify_freeze_manifest()

    ca_fresh = _load_rows(OUT_AUDIT_DIR / "v20_cohort_a_rows.csv")
    cb_fresh = _load_rows(OUT_AUDIT_DIR / "v20_cohort_b_rows.csv")
    cc_fresh = _load_rows(OUT_AUDIT_DIR / "v20_cohort_c_rows.csv")
    la = json.loads((OUT_AUDIT_DIR / "v20_cohort_a_lats.json").read_text(encoding="utf-8"))
    lb = json.loads((OUT_AUDIT_DIR / "v20_cohort_b_lats.json").read_text(encoding="utf-8"))
    lc = json.loads((OUT_AUDIT_DIR / "v20_cohort_c_lats.json").read_text(encoding="utf-8"))
    g3_hist = _load_rows(RESULTS_DIR / "g3_fresh_blind_holdout.csv")
    g5_hist = _load_rows(RESULTS_DIR / "g5_pristine_blind_holdout.csv")

    # 1. Historical Regression (Cohort A + Cohort B = 240 eps)
    ab_v20 = ca_fresh["V20_NAJS"] + cb_fresh["V20_NAJS"]
    ab_v20_lats = la["V20_NAJS"] + lb["V20_NAJS"]
    hist_reg_summary: dict[str, Any] = {"V20_NAJS": _summarize_group(ab_v20, ab_v20_lats)}
    hist_reg_paired: dict[str, Any] = {}
    for pol in ("V19_ADAPTIVE_RL", "V18_EA_JOINT", "BALANS_IJCAI25"):
        rows_ab = ca_fresh[pol] + cb_fresh[pol]
        lats_ab = la[pol] + lb[pol]
        hist_reg_summary[f"{pol}__FRESH_SAME_MACHINE"] = _summarize_group(rows_ab, lats_ab)
        hist_reg_paired[f"{pol}__FRESH_SAME_MACHINE"] = _paired_full(ab_v20, rows_ab)
    for pol in sorted(set(g3_hist) & set(g5_hist)):
        rows_ab_h = g3_hist[pol] + g5_hist[pol]
        hist_reg_summary[f"{pol}__HISTORICAL_CSV"] = _summarize_group(rows_ab_h, None)
        hist_reg_paired[f"{pol}__HISTORICAL_CSV"] = _paired_full(ab_v20, rows_ab_h)

    # 2. Independent V20.0 Confirmation (Cohort C = 120 eps, all 14 methods in ONE session)
    c_v20 = cc_fresh["V20_NAJS"]
    c_summary: dict[str, Any] = {}
    c_paired: dict[str, Any] = {}
    for pol, rows in cc_fresh.items():
        c_summary[pol] = _summarize_group(rows, lc.get(pol))
        if pol != "V20_NAJS":
            c_paired[pol] = _paired_full(c_v20, rows)

    payload = {
        "schema": "v20_1_closure.f0_historical_baseline.v1",
        "v20_0_freeze_manifest": str(FREEZE_MANIFEST_PATH),
        "v20_0_locked_files_count": len(freeze_man["sha256_files"]),
        "historical_regression_cohort_ab_240": {
            "note": "Cohort A (120) + Cohort B (120) treated strictly as historical regression evidence, NOT fresh blind test.",
            "summary": hist_reg_summary,
            "paired_vs_v20_0": hist_reg_paired,
        },
        "independent_confirmation_cohort_c_120": {
            "note": "Brand-new unseen Cohort C (120) run strictly once after V20.0 SHA-256 freeze across all 14 methods.",
            "summary": c_summary,
            "paired_vs_v20_0": c_paired,
        },
    }
    (OUT_CLOSURE_DIR / "f0_v20_0_historical_baseline.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # Markdown report
    lines = [
        "# F0: Frozen `V20.0 (V20_NAJS)` Historical Baseline Archive & Complete Paired Statistics",
        "",
        f"- **V20.0 Freeze Manifest Verified**: `{len(freeze_man['sha256_files'])}` SHA-256 locks intact.",
        "",
        "## 1. Historical Regression Split (`Cohort A + B`, 240 Episodes)",
        "",
        "| Method | Comp100 | Mean JA ↓ | Median JA ↓ | CVaR95 JA ↓ | 4-Topo | Track 1 | Track 2 | ΔJA vs V20 | Bootstrap 95% CI | Cliff's δ | Cohen's d | Wilcoxon p (1s) | Mutual H1 Gain |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pol, s in sorted(hist_reg_summary.items(), key=lambda kv: kv[1]["mean_ja"]):
        pr = hist_reg_paired.get(pol)
        bs = s["by_suite"]
        if pr:
            ci = f"[{pr['ci95_lo_pct']:+.1f}%, {pr['ci95_hi_pct']:+.1f}%]"
            lines.append(
                f"| {pol} | {s['comp100_count']}/240 ({s['comp100_pct']:.1f}%) | {s['mean_ja']:.1f} | {s['median_ja']:.1f} | {s['cvar95_ja']:.1f} | "
                f"{bs['4topo']['mean_ja']:.1f} | {bs['track1']['mean_ja']:.1f} | {bs['track2']['mean_ja']:.1f} | "
                f"{pr['delta_ja_pct']:+.2f}% | {ci} | {pr['cliffs_delta']:+.3f} | {pr['cohen_d_paired']:+.3f} | {pr['wilcoxon_p_one_sided']:.2e} | {pr['h1_gain_mutual_pct']:+.2f}% |"
            )
        else:
            lines.append(
                f"| **{pol}** | **{s['comp100_count']}/240 ({s['comp100_pct']:.1f}%)** | **{s['mean_ja']:.1f}** | **{s['median_ja']:.1f}** | **{s['cvar95_ja']:.1f}** | "
                f"{bs['4topo']['mean_ja']:.1f} | {bs['track1']['mean_ja']:.1f} | {bs['track2']['mean_ja']:.1f} | ref | - | - | - | - | - |"
            )

    lines.extend([
        "",
        "## 2. Independent `V20.0` Confirmation Split (`Cohort C`, 120 Episodes — All 14 Methods Uncontended)",
        "",
        "| Method | Comp100 | Mean JA ↓ | Median JA ↓ | CVaR95 JA ↓ | Pool P95 | Pool P99 | Max ms | >35ms % | ΔJA vs V20 | Bootstrap 95% CI | Cliff's δ | Wilcoxon p (1s) | Mutual H1 Gain |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for pol, s in sorted(c_summary.items(), key=lambda kv: kv[1]["mean_ja"]):
        pr = c_paired.get(pol)
        ovr = f"{100.0 * (s['overrun_35ms_rate'] or 0.0):.2f}%"
        if pr:
            ci = f"[{pr['ci95_lo_pct']:+.1f}%, {pr['ci95_hi_pct']:+.1f}%]"
            lines.append(
                f"| {pol} | {s['comp100_count']}/120 ({s['comp100_pct']:.1f}%) | {s['mean_ja']:.1f} | {s['median_ja']:.1f} | {s['cvar95_ja']:.1f} | "
                f"{s['pooled_p95_ms']:.2f} | {s['pooled_p99_ms']:.2f} | {s['max_decision_ms']:.1f} | {ovr} | "
                f"{pr['delta_ja_pct']:+.2f}% | {ci} | {pr['cliffs_delta']:+.3f} | {pr['wilcoxon_p_one_sided']:.2e} | {pr['h1_gain_mutual_pct']:+.2f}% |"
            )
        else:
            lines.append(
                f"| **{pol}** | **{s['comp100_count']}/120 ({s['comp100_pct']:.1f}%)** | **{s['mean_ja']:.1f}** | **{s['median_ja']:.1f}** | **{s['cvar95_ja']:.1f}** | "
                f"**{s['pooled_p95_ms']:.2f}** | **{s['pooled_p99_ms']:.2f}** | **{s['max_decision_ms']:.1f}** | **{ovr}** | ref | - | - | - | - |"
            )

    (OUT_CLOSURE_DIR / "f0_v20_0_historical_baseline.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[F0 DONE] Archived V20.0 baseline -> {OUT_CLOSURE_DIR / 'f0_v20_0_historical_baseline.md'}")
    print(f"  Cohort A+B (240): V20_NAJS={hist_reg_summary['V20_NAJS']['mean_ja']} CVaR95={hist_reg_summary['V20_NAJS']['cvar95_ja']}")
    print(f"  Cohort C   (120): V20_NAJS={c_summary['V20_NAJS']['mean_ja']} CVaR95={c_summary['V20_NAJS']['cvar95_ja']} Max={c_summary['V20_NAJS']['max_decision_ms']}ms Over35ms={c_summary['V20_NAJS']['overrun_35ms_rate']}")
    return payload


if __name__ == "__main__":
    run_f0_archive()
