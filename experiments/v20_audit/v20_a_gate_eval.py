"""V20-A Task A3: Uncontended closed-loop Gate A verification on VAL-72 & VAL-Static-35."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v20_audit.common import (
    OUT_AUDIT_DIR,
    assert_stage1_quarantine,
    build_val72_specs,
    build_val_static35_specs,
    run_wrapped_episode,
    summarize_episode_rows,
)
from experiments.v20_audit.proposer_variants import UnifiedE0E9PrototypeProposer

CRITIC_NPZ = OUT_AUDIT_DIR / "v20_a_critic_weights.npz"


def _summarize_by_suite(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"overall": summarize_episode_rows(rows)}
    for grp in ("4topo", "track1", "track2"):
        sub = [r for r in rows if r.get("suite_group") == grp]
        if sub:
            out[grp] = summarize_episode_rows(sub)
    return out


def run_v20_a_gate_eval() -> dict[str, Any]:
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    specs72 = build_val72_specs()
    specs35 = build_val_static35_specs()
    assert_stage1_quarantine([sp["seed"] for sp in specs72 + specs35], label="V20A_GateA_Eval")

    base_kw: dict[str, Any] = dict(
        soft_deadline_ms=15.0,
        enable_e4=False,
        enable_e5=False,
        enable_e6=False,
        enable_e7=True,
        enable_e9=False,
        enable_r2_global_ws=False,
        r2_zc_ws_mode="never",
        include_staged_g="on_invalid",
        commit_unwin_seq=False,
        unwin_seq_horizon=2,
    )
    variants: list[tuple[str, dict[str, Any]]] = [
        (
            "V20A_Unified_NoCritic",
            {**base_kw, "critic_weights_path": None, "critic_beta": 0.0},
        ),
        (
            "V20A_Unified_Critic025",
            {**base_kw, "critic_weights_path": CRITIC_NPZ, "critic_beta": 0.25},
        ),
        (
            "V20A_Unified_Critic035",
            {**base_kw, "critic_weights_path": CRITIC_NPZ, "critic_beta": 0.35},
        ),
    ]

    val72_report: dict[str, Any] = {}
    best_vid = None
    best_net = 1e18
    for vid, kw in variants:
        prop = UnifiedE0E9PrototypeProposer(hard_budget_ms=35.0, seed=0, **kw)
        rows = []
        fails = []
        for sp in specs72:
            r, _ = run_wrapped_episode(
                variant_id=vid,
                spec=sp,
                base_env_policy="V19_ADAPTIVE_RL",
                external_proposer=prop,
                hard_budget_ms=35.0,
            )
            rows.append(r)
            if not r["comp100"] or not r["audit_passed"]:
                fails.append((r["episode_key"], r["net_obj"], r["comp100"], r["audit_passed"]))
        summ = _summarize_by_suite(rows)
        summ["incomplete_or_unaudited"] = fails
        ov = summ["overall"]
        gate_pass = bool(
            ov["comp100_count"] == 72
            and ov["audit_pass_rate"] >= 0.9999
            and ov["pooled_p95_ms"] <= 28.0
            and ov["mean_net_obj"] <= 4150.0
        )
        summ["gate_a_pass"] = gate_pass
        val72_report[vid] = summ
        print(
            f"  [{vid}] VAL-72 net={ov['mean_net_obj']} (med={ov['median_net_obj']}), "
            f"comp={ov['comp100_count']}/72, mean_ms={ov['mean_decision_ms']}ms, p95={ov['pooled_p95_ms']}ms, GateA={gate_pass} | "
            f"4T={summ['4topo']['mean_net_obj']} T1={summ['track1']['mean_net_obj']} T2={summ['track2']['mean_net_obj']}",
            flush=True,
        )
        if ov["comp100_count"] == 72 and ov["pooled_p95_ms"] <= 28.0 and ov["mean_net_obj"] < best_net:
            best_net = ov["mean_net_obj"]
            best_vid = vid

    if best_vid is None:
        best_vid = variants[1][0]

    best_kw = dict(variants)[best_vid]
    prop_st = UnifiedE0E9PrototypeProposer(hard_budget_ms=35.0, seed=0, **best_kw)
    rows_st = []
    fails_st = []
    for sp in specs35:
        r, _ = run_wrapped_episode(
            variant_id=f"{best_vid}_Static35",
            spec=sp,
            base_env_policy="V19_ADAPTIVE_RL",
            external_proposer=prop_st,
            hard_budget_ms=35.0,
        )
        rows_st.append(r)
        if not r["comp100"] or not r["audit_passed"]:
            fails_st.append((r["episode_key"], r["net_obj"], r["comp100"], r["audit_passed"]))
    st_summ = summarize_episode_rows(rows_st)
    st_summ["incomplete_or_unaudited"] = fails_st
    st_pass = bool(
        st_summ["comp100_count"] == 35
        and st_summ["audit_pass_rate"] >= 0.9999
        and st_summ["pooled_p95_ms"] <= 28.0
    )
    st_summ["gate_a_static35_pass"] = st_pass
    print(
        f"  [{best_vid}] VAL-Static-35 H1={st_summ['mean_h1_all']}, net={st_summ['mean_net_obj']}, "
        f"comp={st_summ['comp100_count']}/35, p95={st_summ['pooled_p95_ms']}ms, GateA_Static={st_pass}, fails={fails_st}",
        flush=True,
    )

    gate_a_overall_pass = bool(val72_report[best_vid]["gate_a_pass"] and st_pass)
    full_report = {
        "schema": "v20_a.gate_eval.v3_unified",
        "baselines_ref": {"v1_2_1_val72": 4190.21, "v18_ea_val72": 4298.40, "gate_a_target_net_obj": 4150.0},
        "val72": val72_report,
        "selected_winner": best_vid,
        "val_static_35": st_summ,
        "gate_a_overall_pass": gate_a_overall_pass,
    }
    (OUT_AUDIT_DIR / "v20_a_gate_eval.json").write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    return full_report


if __name__ == "__main__":
    run_v20_a_gate_eval()
