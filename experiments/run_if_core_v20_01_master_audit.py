"""Master orchestrator for Stage-1 (`IF-CORE-V20-01`) Scientific Headroom & Mechanism Audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v20_audit.common import (
    OUT_AUDIT_DIR,
    assert_git_amr_sched_clean_relative_to_baseline,
)
from experiments.v20_audit.task_a_funnel_audit import run_task_a
from experiments.v20_audit.task_b_expert_complementarity import run_task_b
from experiments.v20_audit.task_c_liveness_forensics import run_task_c
from experiments.v20_audit.task_d_budget_frontier import run_task_d


def _load_or_run(name: str, json_path: Path, runner: Any, selected: set[str], refresh: bool) -> dict[str, Any]:
    if name in selected and (refresh or not json_path.is_file()):
        return runner()
    if json_path.is_file():
        return json.loads(json_path.read_text(encoding="utf-8"))
    if name in selected:
        return runner()
    return {}


def build_go_nogo_synthesis(
    rep_a: dict[str, Any],
    rep_b: dict[str, Any],
    rep_c: dict[str, Any],
    rep_d: dict[str, Any],
) -> dict[str, Any]:
    """Synthesize quantitative GO/NO-GO decision for Phase 2 (`CoRe-Sched-AR`) from Task A-D evidence."""
    p0_base = rep_a.get("phase0_ladder", {}).get("P0.0_CoRe_v121_Frozen", {}).get("overall", {})
    p0_best = rep_a.get("phase0_ladder", {}).get("P0.3_VOI_CleanBudget_NoSeqLock", {}).get("overall", {})
    p0_v18 = rep_a.get("phase0_ladder", {}).get("V18_EA_Ref", {}).get("overall", {})

    base_net = float(p0_base.get("mean_net_obj", 0.0) or 0.0)
    p03_net = float(p0_best.get("mean_net_obj", 0.0) or 0.0)
    p0_gain_pct = 100.0 * (base_net - p03_net) / max(1e-6, base_net) if base_net > 0 else 0.0

    rel_headroom = float(rep_b.get("headroom", {}).get("rel_headroom_vs_v121_pct", 0.0) or 0.0)
    loo_map = rep_b.get("expert_complementarity_and_loo", {})
    non_redundant_experts = [
        eid
        for eid, d in loo_map.items()
        if eid != "E0_KEEP" and (float(d.get("loo_mean_oracle_degradation", 0.0)) >= 2.0 or float(d.get("win_pct", 0.0)) >= 3.0)
    ]

    cal_ja = rep_b.get("proxy_calibration", {}).get("proxies", {}).get("JA_Unified", {})
    spearman_48 = float(cal_ja.get("mean_spearman_rho_48", 0.0) or 0.0)

    pdg_val = rep_c.get("pdg_generalization_on_val72", {})
    pdg_fpr = float(pdg_val.get("val72_healthy_fpr", 1.0) if pdg_val.get("val72_healthy_fpr") is not None else 1.0)
    c_ep_leads = [
        int(ep.get("policy_summaries", {}).get("V19_ADAPTIVE_RL", {}).get("pdg_lead_before_terminal", 0) or 0)
        for ep in rep_c.get("episodes", {}).values()
    ]
    min_pdg_lead = min(c_ep_leads) if c_ep_leads else 0

    b35 = rep_d.get("frontier", {}).get("35ms", {})
    s5_35 = b35.get("S5_Unified_E0_E9_Proto", {}).get("val72", {})
    s1_35 = b35.get("S1_CoRe_v121", {}).get("val72", {})
    s4_35 = b35.get("S4_V18_EA_SeqDiff", {}).get("val72", {})
    s3_35 = b35.get("S3_V18_EA_NoSeqDiff", {}).get("val72", {})

    gate_checks = {
        "criterion_1_phase0_repair_positive": {
            "pass": p0_gain_pct > 2.0,
            "val72_delta_pct": round(p0_gain_pct, 2),
            "p0_net_obj": p0_base.get("mean_net_obj"),
            "p03_sd_net_obj": p0_best.get("mean_net_obj"),
            "v18_ref_net_obj": p0_v18.get("mean_net_obj"),
        },
        "criterion_2_oracle_headroom_ge_5pct_and_complementarity": {
            "pass": rel_headroom >= 5.0 and len(non_redundant_experts) >= 4,
            "rel_headroom_pct": round(rel_headroom, 2),
            "non_redundant_experts_count": len(non_redundant_experts),
            "non_redundant_experts": non_redundant_experts,
        },
        "criterion_3_pdg_lead_ge_30_and_low_fpr": {
            "pass": min_pdg_lead >= 30 and pdg_fpr <= 0.15,
            "min_lead_ticks_on_outliers": min_pdg_lead,
            "healthy_val72_fpr": pdg_fpr,
        },
        "criterion_4_equal_budget_35ms_scaling_and_proxy_calibration": {
            "pass": bool(s5_35) and float(s5_35.get("pooled_p95_ms", 999)) <= 35.0,
            "s1_v121_35ms_net_obj": s1_35.get("mean_net_obj"),
            "s3_v18_nosd_35ms_net_obj": s3_35.get("mean_net_obj"),
            "s4_v18_sd_35ms_net_obj": s4_35.get("mean_net_obj"),
            "s5_unified_35ms_net_obj": s5_35.get("mean_net_obj"),
            "s5_unified_35ms_p95_ms": s5_35.get("pooled_p95_ms"),
            "ja_unified_spearman_rho_48": spearman_48,
        },
    }
    go_decision = all(v["pass"] for v in gate_checks.values())
    return {
        "schema": "if-core-v20-01.synthesis.v1",
        "decision": "GO_TO_PHASE_2_CORE_SCHED_AR" if go_decision else "REVIEW_GAPS_BEFORE_PHASE_2",
        "gate_checks": gate_checks,
        "recommended_active_expert_portfolio": ["E0_KEEP", *non_redundant_experts],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Stage-1 IF-CORE-V20-01 Scientific Audit.")
    parser.add_argument("--tasks", nargs="+", default=["A", "B", "C", "D"], choices=["A", "B", "C", "D"])
    parser.add_argument("--refresh", action="store_true", help="Re-run selected tasks even if JSON exists.")
    parser.add_argument("--baseline-patch", type=str, default="", help="Optional baseline git diff amr_sched/ snapshot.")
    parser.add_argument("--b-snaps", type=int, default=640, help="Target snapshots for Task B (>= 600).")
    parser.add_argument("--b-cal", type=int, default=180, help="Target 48-tick rollout calibration snapshots.")
    args = parser.parse_args()

    selected = set(args.tasks)
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    patch_p = Path(args.baseline_patch) if args.baseline_patch else None
    t_start = time.perf_counter()

    rep_a = _load_or_run("A", OUT_AUDIT_DIR / "task_a_funnel_and_phase0.json", run_task_a, selected, args.refresh)
    rep_b = _load_or_run(
        "B",
        OUT_AUDIT_DIR / "task_b_expert_oracle.json",
        lambda: run_task_b(target_snapshots=args.b_snaps, rollout_cal_snapshots=args.b_cal),
        selected,
        args.refresh,
    )
    rep_c = _load_or_run("C", OUT_AUDIT_DIR / "task_c_liveness_forensics.json", run_task_c, selected, args.refresh)
    rep_d = _load_or_run("D", OUT_AUDIT_DIR / "task_d_budget_frontier.json", run_task_d, selected, args.refresh)

    if patch_p is not None:
        assert_git_amr_sched_clean_relative_to_baseline(patch_p)

    syn = build_go_nogo_synthesis(rep_a, rep_b, rep_c, rep_d)
    syn["total_elapsed_s"] = round(time.perf_counter() - t_start, 2)
    (OUT_AUDIT_DIR / "IF_CORE_V20_01_SYNTHESIS.json").write_text(json.dumps(syn, indent=2), encoding="utf-8")

    gc = syn["gate_checks"]
    md = [
        "# IF-CORE-V20-01 Stage-1 Scientific Synthesis & GO/NO-GO Decision\n",
        f"- **Decision**: `{syn['decision']}`",
        f"- **Total Wall Time**: `{syn['total_elapsed_s']} s`",
        f"- **Recommended Non-Redundant Expert Portfolio**: `{syn['recommended_active_expert_portfolio']}`\n",
        "## Gate Criteria Checklist\n",
        f"1. **Phase-0 Non-Invasive Repair Lift (`Task A`)**: `pass={gc['criterion_1_phase0_repair_positive']['pass']}` | "
        f"`P0.0={gc['criterion_1_phase0_repair_positive']['p0_net_obj']}` -> `P0.3+SD={gc['criterion_1_phase0_repair_positive']['p03_sd_net_obj']}` "
        f"(`Δ={gc['criterion_1_phase0_repair_positive']['val72_delta_pct']}%`, `V18-EA={gc['criterion_1_phase0_repair_positive']['v18_ref_net_obj']}`)",
        f"2. **Oracle Headroom & Multi-Expert Complementarity (`Task B`)**: `pass={gc['criterion_2_oracle_headroom_ge_5pct_and_complementarity']['pass']}` | "
        f"`Headroom={gc['criterion_2_oracle_headroom_ge_5pct_and_complementarity']['rel_headroom_pct']}%`, "
        f"`Active Non-Redundant Experts={gc['criterion_2_oracle_headroom_ge_5pct_and_complementarity']['non_redundant_experts']}`",
        f"3. **PDG Early Warning Lead >= 30t & Low Healthy FPR (`Task C`)**: `pass={gc['criterion_3_pdg_lead_ge_30_and_low_fpr']['pass']}` | "
        f"`Min Lead={gc['criterion_3_pdg_lead_ge_30_and_low_fpr']['min_lead_ticks_on_outliers']} ticks`, "
        f"`Healthy VAL-72 FPR={gc['criterion_3_pdg_lead_ge_30_and_low_fpr']['healthy_val72_fpr']}`",
        f"4. **Equal-Budget 35 ms Scaling & Proxy Calibration (`Task D` & `Task B`)**: `pass={gc['criterion_4_equal_budget_35ms_scaling_and_proxy_calibration']['pass']}` | "
        f"`S1_v121={gc['criterion_4_equal_budget_35ms_scaling_and_proxy_calibration']['s1_v121_35ms_net_obj']}`, "
        f"`S3_V18(noSD)={gc['criterion_4_equal_budget_35ms_scaling_and_proxy_calibration']['s3_v18_nosd_35ms_net_obj']}`, "
        f"`S4_V18(SD)={gc['criterion_4_equal_budget_35ms_scaling_and_proxy_calibration']['s4_v18_sd_35ms_net_obj']}`, "
        f"`S5_Proto={gc['criterion_4_equal_budget_35ms_scaling_and_proxy_calibration']['s5_unified_35ms_net_obj']}` "
        f"(P95={gc['criterion_4_equal_budget_35ms_scaling_and_proxy_calibration']['s5_unified_35ms_p95_ms']}ms), "
        f"`Spearman rho(48t)={gc['criterion_4_equal_budget_35ms_scaling_and_proxy_calibration']['ja_unified_spearman_rho_48']}`",
    ]
    (OUT_AUDIT_DIR / "IF_CORE_V20_01_SYNTHESIS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[SYNTHESIS] {syn['decision']} -> {OUT_AUDIT_DIR / 'IF_CORE_V20_01_SYNTHESIS.md'}", flush=True)


if __name__ == "__main__":
    main()
