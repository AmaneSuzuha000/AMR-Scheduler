"""Task 2 (F2): PDG 2.0 Falsification — Independent & Combined Ablation of D1, D2, E1, E2.

Strictly calibrated on quarantined DEV splits (`VAL-72`, `VAL-Static-35`, `DEV-Stress-24`).
Historical failure cases (`2906`, `3905`, `37302`, `3906`) are evaluated ONLY in read-only diagnostic mode
after DEV thresholds are locked — zero parameters are tuned on Cohort A/B/C/D.

Outputs Deliverable 2 (`out/v20_1_closure/deliverable2_pdg2_ablation.{json,md}`).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.cvcr.p2_perturbation import P2PerturbationConfig
from experiments.run_g3_fresh_frozen_holdout import build_fresh_120_specs
from experiments.run_g5_pristine_blind_holdout import build_pristine_120_specs
from experiments.v20_1_closure.pdg2_escape import build_pdg2_v20_proposer
from experiments.v20_1_closure.pdg2_monitor import PDG2Config, PDG2LivenessMonitor
from experiments.v20_audit.common import (
    assert_stage1_quarantine,
    build_task_c_outlier_specs,
    build_val72_specs,
    build_val_static35_specs,
    run_wrapped_episode,
    summarize_episode_rows,
)
from experiments.v20_audit.v20_e_gate_eval import _build_val_extreme_stress_12

OUT_CLOSURE_DIR = ROOT / "out" / "v20_1_closure"


def build_dev_stress_24_specs() -> list[dict[str, Any]]:
    """Build 24 quarantined DEV stress episodes (12 extreme perturbation + 12 progressive starvation / corridor)."""
    base12 = _build_val_extreme_stress_12()
    s72 = build_val72_specs()
    # 12 additional DEV progressive-starvation & multi-obstacle corridor cases strictly on quarantined VAL seeds
    starv_keys = (
        "track1_nominal_1161",
        "track1_kitting_windows_1161",
        "track1_execution_blockage_1162",
        "track1_multi_wave_surge_1163",
        "track2_nominal_1161",
        "track2_kitting_windows_1162",
        "track2_kitting_windows_1163",
        "track2_execution_blockage_1163",
        "topo4_offset_dual_lane_window_bottleneck_16701",
        "topo4_narrow_spine_window_bottleneck_16702",
        "topo4_workshop_cells_window_bottleneck_16703",
        "topo4_perimeter_bypass_execution_blockage_16704",
    )
    extra12: list[dict[str, Any]] = []
    for i, ek in enumerate(starv_keys):
        sp = copy.deepcopy(next(s for s in s72 if s["episode_key"] == ek))
        sp["episode_key"] = f"dev_starv_{ek}"
        # Heavy sustained obstacles + travel drift + planner fails to induce real corridor/window stalls in DEV
        sp["p2_cfg"] = P2PerturbationConfig(
            seed=int(sp["seed"]) + 900 + i,
            enable_travel_drift=True,
            drift_prob=0.58,
            early_max_steps=1,
            late_max_steps=6,
            enable_dynamic_obstacles=True,
            obstacle_count=7,
            obstacle_duration_steps=36,
            obstacle_window_range=(15, 160),
            enable_local_planner_fail=True,
            local_fail_count=5,
            local_fail_duration_steps=24,
            local_fail_window_range=(18, 160),
        )
        extra12.append(sp)
    all24 = base12 + extra12
    assert_stage1_quarantine([sp["seed"] for sp in all24], label="Task2_DEV_Stress_24")
    return all24


def _run_pdg2_episode(
    variant_id: str,
    sp: dict[str, Any],
    *,
    d1: bool,
    d2: bool,
    e1: bool,
    e2: bool,
    cfg_overrides: dict[str, Any] | None = None,
    hard_budget_ms: float = 35.0,
) -> dict[str, Any]:
    cfg_kw = dict(enable_legacy_pdg=True, enable_d1=d1, enable_d2=d2, cooldown_ticks=12)
    if cfg_overrides:
        cfg_kw.update(cfg_overrides)
    mon = PDG2LivenessMonitor(cfg=PDG2Config(**cfg_kw))
    prop = build_pdg2_v20_proposer(hard_budget_ms=hard_budget_ms, seed=0, enable_e1=e1, enable_e2=e2)

    def _hook(env, info, act, traces):
        return mon.observe_and_maybe_escalate(env, info, act, traces, proposer=prop)

    row, _ = run_wrapped_episode(
        variant_id=variant_id,
        spec=sp,
        base_env_policy="V19_ADAPTIVE_RL",
        external_proposer=prop,
        hard_budget_ms=hard_budget_ms,
        outer_action_hook=_hook,
    )
    row["pdg_alert_ticks"] = len(mon.alert_ticks)
    row["pdg_first_alert_tick"] = mon.alert_ticks[0] if mon.alert_ticks else None
    row["pdg_escalations"] = len(mon.escalation_ticks)
    row["pdg_inner_recoveries"] = int(prop.pdg_recoveries_triggered)
    row["pdg_alert_sources"] = dict(mon.alert_sources)
    row["pdg_levels_used"] = dict(prop.escalation_level_counts)
    row["pdg_recovery_ms_mean"] = round(float(np.mean(prop.recovery_durations_ms)), 3) if prop.recovery_durations_ms else 0.0
    row["pdg_recovery_ms_p95"] = round(float(np.percentile(prop.recovery_durations_ms, 95)), 3) if prop.recovery_durations_ms else 0.0
    row["pdg_lead_ticks"] = (int(sp["max_steps"]) - int(mon.alert_ticks[0])) if mon.alert_ticks else 0
    row["pdg_irrecoverable_reports"] = int(prop.irrecoverable_reports)
    return row


def _summarize_pdg2_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = summarize_episode_rows(rows)
    alerted = [r for r in rows if r["pdg_alert_ticks"] > 0]
    incomp = [r for r in rows if not r["comp100"]]
    fn_incomp = [r for r in incomp if r["pdg_alert_ticks"] == 0]
    rec_alerted = [r for r in alerted if r["comp100"]]
    rec_ms = [r["pdg_recovery_ms_mean"] for r in rows if r["pdg_inner_recoveries"] > 0]
    lead_ts = [r["pdg_lead_ticks"] for r in alerted]
    base.update(
        {
            "alert_episodes": len(alerted),
            "alert_rate": round(len(alerted) / max(1, len(rows)), 4),
            "missed_failure_fn_count": len(fn_incomp),
            "alerted_recovery_success_count": len(rec_alerted),
            "alerted_recovery_rate": round(len(rec_alerted) / len(alerted), 4) if alerted else None,
            "mean_detection_lead_ticks": round(float(np.mean(lead_ts)), 1) if lead_ts else 0.0,
            "mean_recovery_compute_ms": round(float(np.mean(rec_ms)), 3) if rec_ms else 0.0,
            "total_inner_recoveries": sum(int(r["pdg_inner_recoveries"]) for r in rows),
            "irrecoverable_reports": sum(int(r["pdg_irrecoverable_reports"]) for r in rows),
        }
    )
    return base


ABLATION_MATRIX: list[tuple[str, bool, bool, bool, bool]] = [
    ("A0_V20_Baseline", False, False, False, False),
    ("A1_D1_Only", True, False, False, False),
    ("A2_D2_Only", False, True, False, False),
    ("A3_D1_D2", True, True, False, False),
    ("A4_E1_Only", False, False, True, False),
    ("A5_E2_Only", False, False, False, True),
    ("A6_PDG2_Full", True, True, True, True),
]


def run_task2_ablation(*, check_dev_fpr_only: bool = False) -> dict[str, Any]:
    OUT_CLOSURE_DIR.mkdir(parents=True, exist_ok=True)
    val72 = build_val72_specs()
    stat35 = build_val_static35_specs()
    assert_stage1_quarantine([s["seed"] for s in val72 + stat35], label="Task2_VAL_FPR")

    if check_dev_fpr_only:
        print("=== [Task 2 DEV FPR Check] Scanning VAL-72 & VAL-Static-35 on A6_PDG2_Full ===", flush=True)
        rows72 = [_run_pdg2_episode("fpr72", sp, d1=True, d2=True, e1=True, e2=True) for sp in val72]
        rows35 = [_run_pdg2_episode("fpr35", sp, d1=True, d2=True, e1=True, e2=True) for sp in stat35]
        s72 = _summarize_pdg2_rows(rows72)
        s35 = _summarize_pdg2_rows(rows35)
        print(f"  VAL-72: net={s72['mean_net_obj']} comp={s72['comp100_count']}/72 alerts={s72['alert_episodes']}/72 p95={s72['pooled_p95_ms']}ms", flush=True)
        for r in rows72:
            if r["pdg_alert_ticks"] > 0:
                print(f"    ALERT72: {r['episode_key']} net={r['net_obj']} comp={r['comp100']} first={r['pdg_first_alert_tick']} src={r['pdg_alert_sources']} lvls={r['pdg_levels_used']}", flush=True)
        print(f"  Stat35: H1={s35['mean_h1_all']} comp={s35['comp100_count']}/35 alerts={s35['alert_episodes']}/35 p95={s35['pooled_p95_ms']}ms", flush=True)
        for r in rows35:
            if r["pdg_alert_ticks"] > 0:
                print(f"    ALERT35: {r['episode_key']} H1={r['h1']} comp={r['comp100']} first={r['pdg_first_alert_tick']} src={r['pdg_alert_sources']}", flush=True)
        return {"val72": s72, "stat35": s35}

    dev_stress = build_dev_stress_24_specs()
    # Read-only diagnostic historical archetypes (NOT used for calibration)
    hist_diag_specs = [
        next(s for s in build_fresh_120_specs() if s["episode_key"] == "track1_nominal_2906"),
        next(s for s in build_pristine_120_specs() if s["episode_key"] == "track2_kitting_windows_3905"),
    ] + build_task_c_outlier_specs()

    results: dict[str, Any] = {
        "schema": "v20_1_closure.deliverable2_pdg2_ablation.v1",
        "config": PDG2Config().__dict__,
        "val72": {},
        "val_static35": {},
        "dev_stress24": {},
        "historical_diagnostic_4cases": {},
    }

    # 1. DEV-Stress-24 across ALL 7 ablation variants
    print("=== [Task 2.1] Running DEV-Stress-24 across all 7 PDG 2.0 ablation variants ===", flush=True)
    for name, d1, d2, e1, e2 in ABLATION_MATRIX:
        rows = [_run_pdg2_episode(name, sp, d1=d1, d2=d2, e1=e1, e2=e2) for sp in dev_stress]
        sm = _summarize_pdg2_rows(rows)
        results["dev_stress24"][name] = sm
        print(
            f"  Stress24 {name:16s}: Comp={sm['comp100_count']:2d}/24 JA={sm['mean_net_obj']:8.2f} "
            f"Alerts={sm['alert_episodes']:2d} FN={sm['missed_failure_fn_count']} RecRate={sm['alerted_recovery_rate']} "
            f"Lead={sm['mean_detection_lead_ticks']:5.1f}t P95={sm['pooled_p95_ms']:5.2f}ms RecMs={sm['mean_recovery_compute_ms']:5.2f}",
            flush=True,
        )

    # 2. Nominal VAL-72 & VAL-Static-35 across detector variants + full PDG 2.0 (FPR & zero-regression check)
    print("=== [Task 2.2] Running VAL-72 (FPR & nominal J_A regression check) ===", flush=True)
    for name, d1, d2, e1, e2 in (
        ("A0_V20_Baseline", False, False, False, False),
        ("A1_D1_Only", True, False, False, False),
        ("A2_D2_Only", False, True, False, False),
        ("A3_D1_D2", True, True, False, False),
        ("A6_PDG2_Full", True, True, True, True),
    ):
        rows72 = [_run_pdg2_episode(f"v72_{name}", sp, d1=d1, d2=d2, e1=e1, e2=e2) for sp in val72]
        sm72 = _summarize_pdg2_rows(rows72)
        results["val72"][name] = sm72
        print(
            f"  VAL-72   {name:16s}: Comp={sm72['comp100_count']:2d}/72 JA={sm72['mean_net_obj']:7.2f} "
            f"FPR={sm72['alert_episodes']}/72 P95={sm72['pooled_p95_ms']:5.2f}ms",
            flush=True,
        )

    print("=== [Task 2.3] Running VAL-Static-35 (Baseline vs PDG2_Full) ===", flush=True)
    for name, d1, d2, e1, e2 in (("A0_V20_Baseline", False, False, False, False), ("A6_PDG2_Full", True, True, True, True)):
        rows35 = [_run_pdg2_episode(f"s35_{name}", sp, d1=d1, d2=d2, e1=e1, e2=e2) for sp in stat35]
        sm35 = _summarize_pdg2_rows(rows35)
        results["val_static35"][name] = sm35
        print(f"  Stat35   {name:16s}: Comp={sm35['comp100_count']:2d}/35 H1={sm35['mean_h1_all']:6.2f} FPR={sm35['alert_episodes']}/35 P95={sm35['pooled_p95_ms']:5.2f}ms", flush=True)

    # 3. Read-only diagnostic falsification on 4 historical failure archetypes
    print("=== [Task 2.4] Read-Only Diagnostic Verification on 4 Historical Failure Archetypes ===", flush=True)
    for name, d1, d2, e1, e2 in (
        ("A0_V20_Baseline", False, False, False, False),
        ("A4_E1_Only", False, False, True, False),
        ("A5_E2_Only", False, False, False, True),
        ("A6_PDG2_Full", True, True, True, True),
    ):
        case_out = {}
        for sp in hist_diag_specs:
            r = _run_pdg2_episode(f"diag_{name}", sp, d1=d1, d2=d2, e1=e1, e2=e2)
            case_out[sp["episode_key"]] = {
                "net_obj": r["net_obj"],
                "cmax": r["cmax"],
                "comp100": r["comp100"],
                "alerts": r["pdg_alert_ticks"],
                "first_alert": r["pdg_first_alert_tick"],
                "inner_recoveries": r["pdg_inner_recoveries"],
                "sources": r["pdg_alert_sources"],
                "levels": r["pdg_levels_used"],
            }
        results["historical_diagnostic_4cases"][name] = case_out
        print(f"  Diag {name:16s}: " + " | ".join(f"{k.split('_')[-1]}={v['net_obj']}({v['comp100']},a={v['alerts']})" for k, v in case_out.items()), flush=True)

    (OUT_CLOSURE_DIR / "deliverable2_pdg2_ablation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Markdown Deliverable 2
    md = [
        "# Deliverable 2: PDG 2.0 Detection (`D1/D2`) & Escape (`E1/E2`) Ablation Report",
        "",
        "> Strictly calibrated on quarantined DEV splits (`VAL-72`, `VAL-Static-35`, `DEV-Stress-24`). Historical failure cases evaluated read-only after thresholds locked.",
        "",
        "## 1. Quarantined `DEV-Stress-24` Independent & Combined Ablation",
        "",
        "| Variant | Comp100 | Mean JA ↓ | Median JA ↓ | Alert Eps | Missed Fail (FN) | Alerted Rec Rate | Mean Lead (t) | Recovery ms | Pool P95 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, *_ in ABLATION_MATRIX:
        s = results["dev_stress24"][name]
        rr = f"{100.0*s['alerted_recovery_rate']:.1f}%" if s["alerted_recovery_rate"] is not None else "-"
        md.append(
            f"| {name} | {s['comp100_count']}/24 ({100*s['comp100_rate']:.1f}%) | {s['mean_net_obj']:.2f} | {s['median_net_obj']:.2f} | "
            f"{s['alert_episodes']} | {s['missed_failure_fn_count']} | {rr} | {s['mean_detection_lead_ticks']:.1f} | {s['mean_recovery_compute_ms']:.2f} | {s['pooled_p95_ms']:.2f} ms |"
        )

    md.extend([
        "",
        "## 2. Nominal `VAL-72` & `VAL-Static-35` (False-Positive Rate & Regression Check)",
        "",
        "| Split / Variant | Comp100 | Mean JA ↓ | Pure H1 ↓ | FPR (Alert Eps) | Pool P95 | Pool P99 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name, s in results["val72"].items():
        md.append(f"| VAL-72 · {name} | {s['comp100_count']}/72 | {s['mean_net_obj']:.2f} | {s['mean_h1_all']:.2f} | {s['alert_episodes']}/72 ({100*s['alert_rate']:.2f}%) | {s['pooled_p95_ms']:.2f} ms | {s['pooled_p99_ms']:.2f} ms |")
    for name, s in results["val_static35"].items():
        md.append(f"| Static-35 · {name} | {s['comp100_count']}/35 | {s['mean_net_obj']:.2f} | {s['mean_h1_all']:.2f} | {s['alert_episodes']}/35 ({100*s['alert_rate']:.2f}%) | {s['pooled_p95_ms']:.2f} ms | {s['pooled_p99_ms']:.2f} ms |")

    md.extend([
        "",
        "## 3. Read-Only Diagnostic Verification on 4 Historical Failure Archetypes (`2906`, `3905`, `37302`, `3906`)",
        "",
        "| Variant | `2906` (Comp, Alerts) | `3905` (Comp, Alerts) | `37302` (Comp, Alerts) | `3906` (Comp, Alerts) |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ])
    for name, cmap in results["historical_diagnostic_4cases"].items():
        cols = []
        for ek in (
            "track1_nominal_2906",
            "track2_kitting_windows_3905",
            "topo4_offset_dual_lane_window_bottleneck_37302",
            "track2_kitting_windows_3906",
        ):
            v = cmap[ek]
            cols.append(f"{v['net_obj']:.1f} ({v['comp100']}, a={v['alerts']}, r={v['inner_recoveries']})")
        md.append(f"| {name} | " + " | ".join(cols) + " |")

    (OUT_CLOSURE_DIR / "deliverable2_pdg2_ablation.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[TASK 2 DONE] Wrote Deliverable 2 -> {OUT_CLOSURE_DIR / 'deliverable2_pdg2_ablation.md'}", flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-dev-fpr", action="store_true")
    args = ap.parse_args()
    run_task2_ablation(check_dev_fpr_only=args.check_dev_fpr)
