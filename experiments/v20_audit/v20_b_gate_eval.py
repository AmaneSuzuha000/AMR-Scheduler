"""Phase V20-B: Gate B Evaluation + Bidirectional V18<->V19 Ablation + RQ3 (Seq-Diff ON vs OFF)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v20_audit.common import (
    OUT_AUDIT_DIR,
    SeqDiffEvalContext,
    assert_stage1_quarantine,
    build_val72_specs,
    build_val_static35_specs,
    run_wrapped_episode,
    summarize_episode_rows,
)
from experiments.v20_audit.v20_b_bidirectional import V20BProposer

CRITIC_NPZ = OUT_AUDIT_DIR / "v20_a_critic_weights.npz"


def _summarize_suite(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"overall": summarize_episode_rows(rows)}
    for grp in ("4topo", "track1", "track2"):
        sub = [r for r in rows if r.get("suite_group") == grp]
        if sub:
            out[grp] = summarize_episode_rows(sub)
    return out


def _eval_with_eval_counts(
    vid: str,
    specs: list[dict[str, Any]],
    proposer: V20BProposer,
    *,
    budget_ms: float = 35.0,
) -> tuple[list[dict[str, Any]], float]:
    eval_cnts: list[int] = []

    def _wrapped(snap, net, *, budget_ms: float = 35.0):
        prop = proposer(snap, net, budget_ms=budget_ms)
        ec = int((prop.telemetry or {}).get("eval_cnt", 0))
        if ec > 0:
            eval_cnts.append(ec)
        return prop

    rows = []
    for sp in specs:
        r, _ = run_wrapped_episode(
            variant_id=vid,
            spec=sp,
            base_env_policy="V19_ADAPTIVE_RL",
            external_proposer=_wrapped,
            hard_budget_ms=budget_ms,
        )
        rows.append(r)
    mean_k = round(float(sum(eval_cnts) / max(1, len(eval_cnts))), 3)
    return rows, mean_k


def run_v20_b_gate_eval() -> dict[str, Any]:
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    specs72 = build_val72_specs()
    specs35 = build_val_static35_specs()
    assert_stage1_quarantine([sp["seed"] for sp in specs72 + specs35], label="V20B_GateB_Eval")

    base_kw: dict[str, Any] = dict(
        soft_deadline_ms=15.0,
        enable_e4=True,
        e4_selector="fast_surr",
        e4_include_zc=False,
        e4_mode="ws_or_invalid",
        staged_mode="ws_or_invalid",
        enable_e6_static_or_headroom=False,
        enable_e7=True,
        enable_e9=False,
        enable_early_stop=False,
        compact_calm_ms=None,
        enable_r2_global_ws=False,
        r2_zc_ws_mode="never",
        include_staged_g="on_invalid",
        unwin_seq_horizon=2,
        seq_diff_b2=True,
    )

    variants: list[tuple[str, dict[str, Any]]] = [
        ("V20B_NoBidir", {**base_kw, "enable_bidir_v18_to_v19": False, "critic_weights_path": None, "critic_beta": 0.0}),
        ("V20B_Bidir_NoCritic", {**base_kw, "enable_bidir_v18_to_v19": True, "critic_weights_path": None, "critic_beta": 0.0}),
        (
            "V20B_Bidir_CriticWS025",
            {**base_kw, "enable_bidir_v18_to_v19": True, "critic_weights_path": CRITIC_NPZ, "critic_beta": 0.25, "critic_mode": "ws_only"},
        ),
    ]

    val72_res: dict[str, Any] = {}
    val72_raw: dict[str, list[dict[str, Any]]] = {}
    best_vid = None
    best_net = 1e18

    print("=== [V20-B] Running VAL-72 Gate B Ablation ===", flush=True)
    for vid, kw in variants:
        prop = V20BProposer(hard_budget_ms=35.0, seed=0, **kw)
        rows, mean_k = _eval_with_eval_counts(vid, specs72, prop, budget_ms=35.0)
        val72_raw[vid] = rows
        summ = _summarize_suite(rows)
        ov = summ["overall"]
        summ["mean_k_eval"] = mean_k
        gate_b_val72 = bool(
            ov["comp100_count"] == 72
            and ov["audit_pass_rate"] >= 0.9999
            and ov["pooled_p95_ms"] <= 28.0
            and ov["mean_net_obj"] < 4190.21
            and summ["4topo"]["mean_net_obj"] <= 750.0
        )
        summ["gate_b_val72_pass"] = gate_b_val72
        val72_res[vid] = summ
        print(
            f"  [{vid}] net={ov['mean_net_obj']} (med={ov['median_net_obj']}) comp={ov['comp100_count']}/72 "
            f"p95={ov['pooled_p95_ms']}ms K_eval={mean_k} | 4T={summ['4topo']['mean_net_obj']} "
            f"T1={summ['track1']['mean_net_obj']} T2={summ['track2']['mean_net_obj']} | Pass72={gate_b_val72}",
            flush=True,
        )
        if ov["comp100_count"] == 72 and ov["pooled_p95_ms"] <= 28.0 and ov["mean_net_obj"] < best_net:
            best_net = ov["mean_net_obj"]
            best_vid = vid

    if best_vid is None:
        best_vid = variants[1][0]
    win_kw = dict(variants)[best_vid]

    print(f"=== [V20-B] Running VAL-Static-35 on winner {best_vid} ===", flush=True)
    prop_st = V20BProposer(hard_budget_ms=35.0, seed=0, **win_kw)
    rows_st, mean_k_st = _eval_with_eval_counts(f"{best_vid}_Static35", specs35, prop_st, budget_ms=35.0)
    st_summ = summarize_episode_rows(rows_st)
    st_summ["mean_k_eval"] = mean_k_st
    gate_b_st = bool(
        st_summ["comp100_count"] == 35
        and st_summ["audit_pass_rate"] >= 0.9999
        and st_summ["pooled_p95_ms"] <= 28.0
        and st_summ["mean_h1_all"] <= 400.0
    )
    st_summ["gate_b_static35_pass"] = gate_b_st
    gate_b_pass = bool(val72_res[best_vid]["gate_b_val72_pass"] and gate_b_st)
    print(
        f"  [{best_vid}] VAL-Static-35 H1={st_summ['mean_h1_all']} net={st_summ['mean_net_obj']} "
        f"comp={st_summ['comp100_count']}/35 p95={st_summ['pooled_p95_ms']}ms GateB={gate_b_pass}",
        flush=True,
    )

    print("=== [V20-B / RQ3] Equal-Budget Seq-Diff ON vs OFF on VAL-72 (10 / 20 / 35 ms) ===", flush=True)
    rq3_rows: dict[str, Any] = {}
    for b_ms in (10.0, 20.0, 35.0):
        sd_map: dict[bool, dict[str, Any]] = {}
        raw_pair: dict[bool, list[dict[str, Any]]] = {}
        sd_soft = min(15.0, max(5.5, b_ms * 0.65))
        for sd_flag in (True, False):
            kw_b = {**win_kw, "soft_deadline_ms": sd_soft, "seq_diff_b2": sd_flag}
            p_b = V20BProposer(hard_budget_ms=b_ms, seed=0, **kw_b)
            with SeqDiffEvalContext(enabled=sd_flag):
                rows_b, k_b = _eval_with_eval_counts(f"rq3_{int(b_ms)}_sd{int(sd_flag)}", specs72, p_b, budget_ms=b_ms)
            ov_b = summarize_episode_rows(rows_b)
            ov_b["mean_k_eval"] = k_b
            sd_map["on" if sd_flag else "off"] = ov_b
            raw_pair[sd_flag] = rows_b
            print(
                f"  RQ3 B={b_ms:4.0f}ms SeqDiff={'ON ' if sd_flag else 'OFF'}: "
                f"net={ov_b['mean_net_obj']:7.2f} K_eval={k_b:5.2f} comp={ov_b['comp100_count']}/72 "
                f"mean_ms={ov_b['mean_decision_ms']:6.3f} p95={ov_b['pooled_p95_ms']:6.3f}ms",
                flush=True,
            )
        diffs = [float(a["net_obj"]) - float(b["net_obj"]) for a, b in zip(raw_pair[True], raw_pair[False])]
        p_wil = float(wilcoxon(diffs, zero_method="wilcox", alternative="less").pvalue) if any(abs(x) > 1e-9 for x in diffs) else 1.0
        sd_map["paired_wilcoxon_p_less"] = round(p_wil, 6)
        rq3_rows[f"{int(b_ms)}ms"] = sd_map

    report = {
        "schema": "v20_b.gate_eval.v1",
        "val72": val72_res,
        "selected_winner": best_vid,
        "winner_config": {k: (str(v) if isinstance(v, Path) else v) for k, v in win_kw.items()},
        "val_static_35": st_summ,
        "gate_b_pass": gate_b_pass,
        "rq3_seq_diff": rq3_rows,
    }
    (OUT_AUDIT_DIR / "v20_b_gate_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    run_v20_b_gate_eval()
