"""Task D (`IF-CORE-V20-01`): Equal-Budget Quality-Latency Frontier across VAL-72 & VAL-Static-35."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.baselines.erdogan_ejor26 import solve_erdogan_ils_lp_ejor26
from amr_sched.baselines.lbbd_cp_cor26 import solve_lbbd_cp_cor26
from experiments.v20_audit.common import (
    OUT_AUDIT_DIR,
    assert_stage1_quarantine,
    build_val72_specs,
    build_val_static35_specs,
    make_v18_external_proposer,
    run_wrapped_episode,
    summarize_episode_rows,
)
from experiments.v20_audit.proposer_variants import (
    Phase0Config,
    Phase0RestoredProposer,
    UnifiedE0E9PrototypeProposer,
)

BUDGET_LEVELS_MS: tuple[float, ...] = (10.0, 20.0, 35.0, 50.0, 100.0)


def _compute_static_offline_references(static_specs: list[dict[str, Any]], *, time_limit_s: float = 10.0) -> dict[str, dict[str, Any]]:
    """Compute exact `COR26-Ref` (Phase-1 LB_phys & Sim UB) and `EJOR26-Erdogan` on VAL-Static-35."""
    refs: dict[str, dict[str, Any]] = {}
    for sp in static_specs:
        ek = str(sp["episode_key"])
        sc = copy.deepcopy(sp["scenario_tuple"])
        m_steps = int(sp["max_steps"])
        seed = int(sp["seed"])
        res_cor = solve_lbbd_cp_cor26(
            copy.deepcopy(sc),
            time_limit_s=float(time_limit_s),
            max_iters=25,
            num_workers=8,
            max_steps=m_steps,
            verify_in_simulator=True,
            seed=seed,
        )
        res_ejor = solve_erdogan_ils_lp_ejor26(
            copy.deepcopy(sc),
            time_limit_s=float(time_limit_s),
            max_steps=m_steps,
            seed=seed,
        )
        cor_ub = (
            float(res_cor.upper_bound)
            if (res_cor.sim_audit_passed and res_cor.sim_100pct_completed and res_cor.upper_bound is not None)
            else None
        )
        erd_h1 = (
            float(res_ejor.sim_h1_obj)
            if (res_ejor.sim_audit_passed and res_ejor.sim_100pct_completed and res_ejor.sim_h1_obj is not None)
            else None
        )
        refs[ek] = {
            "cor26_lb_phys": float(res_cor.lower_bound),
            "cor26_ub_h1": cor_ub,
            "cor26_optimal": bool(str(res_cor.status).upper() == "OPTIMAL"),
            "ejor26_h1": erd_h1,
        }
    return refs


def run_task_d() -> dict[str, Any]:
    """Run uncontended single-process equal-budget sweep across {10, 20, 35, 50, 100} ms."""
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    val72_specs = build_val72_specs()
    static35_specs = build_val_static35_specs()
    assert_stage1_quarantine(
        [sp["seed"] for sp in val72_specs] + [sp["seed"] for sp in static35_specs],
        label="TaskD_Suites",
    )

    print("[Task D] Computing offline COR26-Ref LB_phys & EJOR26-Erdogan baselines on VAL-Static-35...", flush=True)
    static_refs = _compute_static_offline_references(static35_specs)
    mean_lb_phys = float(np.mean([v["cor26_lb_phys"] for v in static_refs.values()]))
    cor_ubs = [v["cor26_ub_h1"] for v in static_refs.values() if v["cor26_ub_h1"] is not None]
    ejor_h1s = [v["ejor26_h1"] for v in static_refs.values() if v["ejor26_h1"] is not None]
    mean_cor_ub = float(np.mean(cor_ubs)) if cor_ubs else 0.0
    mean_ejor_h1 = float(np.mean(ejor_h1s)) if ejor_h1s else 0.0

    all_csv_rows: list[dict[str, Any]] = []
    frontier_summary: dict[str, dict[str, Any]] = {}

    for b_ms in BUDGET_LEVELS_MS:
        b_key = f"{int(b_ms)}ms"
        print(f"[Task D] Evaluating 5 solvers at equal hard budget B = {b_key}...", flush=True)
        solvers: list[tuple[str, str, Any]] = [
            ("S1_CoRe_v121", "V19_ADAPTIVE_RL", None),
            (
                "S2_CoRe_Phase0_Scaled",
                "V19_ADAPTIVE_RL",
                Phase0RestoredProposer(
                    Phase0Config(
                        name=f"P0.3_{b_key}",
                        fix_a4_seq=False,
                        unlock_full_pool_and_bays=True,
                        attach_cliff_locks=False,
                        wire_voi_and_scale_budget=True,
                        use_seq_diff_b2=True,
                        hard_budget_ms=b_ms,
                    ),
                    backend="numpy",
                    use_rust=True,
                ),
            ),
            (
                "S3_V18_EA_NoSeqDiff",
                "V18_EA_JOINT",
                make_v18_external_proposer(hard_budget_ms=b_ms, seq_diff_b2=False, policy_label=f"V18_NoSD_{b_key}"),
            ),
            (
                "S4_V18_EA_SeqDiff",
                "V18_EA_JOINT",
                make_v18_external_proposer(hard_budget_ms=b_ms, seq_diff_b2=True, policy_label=f"V18_SD_{b_key}"),
            ),
            (
                "S5_Unified_E0_E9_Proto",
                "V19_ADAPTIVE_RL",
                UnifiedE0E9PrototypeProposer(hard_budget_ms=b_ms, seed=0),
            ),
        ]

        frontier_summary[b_key] = {}
        for s_name, base_pol, proposer in solvers:
            val_rows: list[dict[str, Any]] = []
            for sp in val72_specs:
                rec, _ = run_wrapped_episode(
                    variant_id=f"{s_name}@{b_key}",
                    spec=sp,
                    base_env_policy=base_pol,
                    external_proposer=proposer,
                    hard_budget_ms=b_ms,
                    collect_traces=False,
                )
                rec["budget_ms"] = int(b_ms)
                rec["solver"] = s_name
                rec["split"] = "VAL-72"
                val_rows.append(rec)

            stat_rows: list[dict[str, Any]] = []
            stat_lb_gaps: list[float] = []
            for sp in static35_specs:
                rec, _ = run_wrapped_episode(
                    variant_id=f"{s_name}@{b_key}",
                    spec=sp,
                    base_env_policy=base_pol,
                    external_proposer=proposer,
                    hard_budget_ms=b_ms,
                    collect_traces=False,
                )
                rec["budget_ms"] = int(b_ms)
                rec["solver"] = s_name
                rec["split"] = "VAL-Static-35"
                ek = str(sp["episode_key"])
                lb = float(static_refs[ek]["cor26_lb_phys"])
                rec["cor26_lb_phys"] = lb
                rec["gap_to_lb_pct"] = round(100.0 * (float(rec["h1"]) - lb) / max(1.0, lb), 2)
                if rec["comp100"]:
                    stat_lb_gaps.append(rec["gap_to_lb_pct"])
                stat_rows.append(rec)

            ov_val = summarize_episode_rows(val_rows)
            ov_stat = summarize_episode_rows(stat_rows)
            by_suite_val = {
                grp: summarize_episode_rows([r for r in val_rows if r["suite_group"] == grp])
                for grp in ("4topo", "track1", "track2")
            }
            frontier_summary[b_key][s_name] = {
                "val72": ov_val,
                "val72_by_suite": by_suite_val,
                "static35": {
                    **ov_stat,
                    "mean_gap_to_cor26_lb_pct": round(float(np.mean(stat_lb_gaps)), 2) if stat_lb_gaps else None,
                },
            }

            for r in val_rows + stat_rows:
                row_flat = {k: v for k, v in r.items() if k != "decision_latencies_ms"}
                all_csv_rows.append(row_flat)

            print(
                f"  -> [{b_key}] {s_name}: VAL72={ov_val['mean_net_obj']:.2f} "
                f"(4T={by_suite_val['4topo']['mean_net_obj']:.1f}, comp={ov_val['comp100_count']}/72, "
                f"P95={ov_val['pooled_p95_ms']:.2f}ms) | Static35_H1={ov_stat['mean_h1_all']:.1f} "
                f"(GapLB={frontier_summary[b_key][s_name]['static35']['mean_gap_to_cor26_lb_pct']}%)",
                flush=True,
            )

    payload = {
        "schema": "if-core-v20-01.task_d.v1",
        "budget_levels_ms": list(BUDGET_LEVELS_MS),
        "static35_offline_references": {
            "mean_cor26_lb_phys": round(mean_lb_phys, 2),
            "mean_cor26_ub_h1": round(mean_cor_ub, 2),
            "mean_ejor26_h1": round(mean_ejor_h1, 2),
        },
        "frontier": frontier_summary,
    }
    (OUT_AUDIT_DIR / "task_d_budget_frontier.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if all_csv_rows:
        all_keys: list[str] = list(dict.fromkeys(k for r in all_csv_rows for k in r.keys()))
        with (OUT_AUDIT_DIR / "task_d_budget_frontier.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=all_keys)
            w.writeheader()
            for r in all_csv_rows:
                w.writerow({k: r.get(k, "") for k in all_keys})

    md = [
        "# Task D: Equal-Budget Quality-Latency Frontier (`10 / 20 / 35 / 50 / 100 ms`)\n",
        f"- **VAL-Static-35 Offline Reference**: `COR26-Ref LB_phys = {mean_lb_phys:.1f}`, `COR26-Ref UB = {mean_cor_ub:.1f}`, `EJOR26-Erdogan H1 = {mean_ejor_h1:.1f}`\n",
        "| Budget | Solver | VAL-72 Net Obj | 4-Topo | Track 1 | Track 2 | Comp100 | Pooled P95 (ms) | Pooled P99 (ms) | Static-35 H1 | Gap vs LB |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b_ms in BUDGET_LEVELS_MS:
        b_key = f"{int(b_ms)}ms"
        for s_name, d in frontier_summary[b_key].items():
            v = d["val72"]
            bs = d["val72_by_suite"]
            st = d["static35"]
            md.append(
                f"| {b_key} | {s_name} | {v['mean_net_obj']:.2f} | {bs['4topo']['mean_net_obj']:.1f} | "
                f"{bs['track1']['mean_net_obj']:.1f} | {bs['track2']['mean_net_obj']:.1f} | "
                f"{v['comp100_count']}/72 | {v['pooled_p95_ms']:.2f} | {v['pooled_p99_ms']:.2f} | "
                f"{st['mean_h1_all']:.1f} | {st['mean_gap_to_cor26_lb_pct']}% |"
            )
    (OUT_AUDIT_DIR / "task_d_budget_frontier.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    run_task_d()
