"""IF-CORE-RGRO-CLOSURE-01 · Task 1 (P0): Unified Deadline Closure & Zero-Expiry Hard Veto.

Strictly isolated on quarantined DEV (`VAL-72`, `DEV-Stress-24`, `DEV-Compute-24`) under real
uncontended `P2` wall clock. Measures and compares:
  - `V19_Frozen` (`V19_ADAPTIVE_RL`)
  - `Old_RGRO_U4` (`experiments/rgro` prior cutoff, without unified commit reserve & zero-expiry veto)
  - `U4_DeadlineClosed` (`RGROClosureProposer` + `UnifiedDeadlineManager` + Zero-Expiry Hard Veto)
  - `U5_DeadlineClosed` (`U4_DeadlineClosed` + CRC gate)

Explicitly separates:
  1. `internal_algo_ms` (algorithmic compute excluding GC & OS preemption gaps),
  2. `os_jitter_ms` (GC pauses + host scheduling preemption / injected contention),
  3. `validate_commit_ms` (`NativeTransactionOwner.validate_and_commit`),
  4. `actual_response_ms` (`t_propose_ms + t_validate_commit_ms`),
  5. `fallback_response_ms` (E2E latency specifically when deadline cutoff / hard veto triggers fallback).

Outputs:
  - `out/rgro_closure/deliverable1_deadline_closure.{json,md}`
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from experiments.rgro.dfl_critic import DFLAdvantageEnsemble
from experiments.rgro.rgro_proposer import make_ladder_config, run_rgro_episode
from experiments.rgro.task2_rgro_ablation import _run_frozen_baseline
from experiments.rgro_closure.closure_common import (
    OUT_CLOSURE_DIR,
    assert_closure_dev_quarantine,
    build_dev_compute_24_specs,
    build_dev_stress_24_specs,
    build_val72_specs,
    deterministic_compute_jitter_ms,
    paired_stats,
    pct,
    summarize_rows,
)
from experiments.rgro_closure.rgro_closure_proposer import (
    ClosureConfig,
    RGROClosureProposer,
    run_closure_episode,
)


def _lat_decomposition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dec = [x for r in rows for x in r.get("decision_latencies_ms", [])]
    val = [x for r in rows for x in r.get("validate_commit_ms", [])]
    ia = [x for r in rows for x in r.get("internal_algo_ms", [])]
    oj = [x for r in rows for x in r.get("os_jitter_ms", [])]
    fb = [x for r in rows for x in r.get("fallback_response_ms", [])]
    sm = summarize_rows(rows)
    sm.update(
        {
            "veto_count_total": sum(int(r.get("veto_count", 0)) for r in rows),
            "fallback_count_total": sum(int(r.get("fallback_count", 0)) for r in rows),
            "internal_overrun_total": sum(int(r.get("internal_overrun_count", 0)) for r in rows),
            "internal_algo_p95_p99_max": [round(pct(ia, 95.0), 2), round(pct(ia, 99.0), 2), round(max(ia), 2)] if ia else [0.0, 0.0, 0.0],
            "os_jitter_p95_p99_max": [round(pct(oj, 95.0), 2), round(pct(oj, 99.0), 2), round(max(oj), 2)] if oj else [0.0, 0.0, 0.0],
            "validate_commit_p95_p99_max": [round(pct(val, 95.0), 2), round(pct(val, 99.0), 2), round(max(val), 2)] if val else [0.0, 0.0, 0.0],
            "fallback_n": len(fb),
            "fallback_p95_p99_max": [round(pct(fb, 95.0), 2), round(pct(fb, 99.0), 2), round(max(fb), 2)] if fb else [0.0, 0.0, 0.0],
            "over35_count": sum(1 for x in dec if x > 35.0),
        }
    )
    return sm


def main() -> None:
    OUT_CLOSURE_DIR.mkdir(parents=True, exist_ok=True)
    ens_v19 = DFLAdvantageEnsemble.load_npz(ROOT / "out" / "rgro" / "dfl_critic_v19.npz")

    val72 = build_val72_specs()
    stress24 = build_dev_stress_24_specs()
    comp24 = build_dev_compute_24_specs()
    assert_closure_dev_quarantine([int(s["seed"]) for s in val72 + stress24 + comp24], label="Task1_DeadlineClosure")

    cfg_old_u4 = make_ladder_config("U4", dfl_tau_gain=20.0, lcb_kappa=0.75)
    cfg_closed_u4 = ClosureConfig(policy_id="U4_DeadlineClosed", critic_mode="dfl_lcb", sla_target_ms=20.5, window_tight_sla_ms=17.5)
    cfg_closed_u5 = ClosureConfig(policy_id="U5_DeadlineClosed", critic_mode="dfl_lcb_crc", sla_target_ms=20.5, window_tight_sla_ms=17.5)

    suites = [
        ("VAL72_StdDEV", val72, None),
        ("DEV_Stress24", stress24, None),
        ("DEV_Compute24", comp24, deterministic_compute_jitter_ms),
    ]

    print("=== [Task 1] Unified Deadline Closure & Zero-Expiry Hard Veto (Quarantined DEV, P2 WallClock) ===", flush=True)
    results: dict[str, Any] = {}
    all_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for s_name, specs, j_fn in suites:
        rows_by_pol: dict[str, list[dict[str, Any]]] = {}
        t0 = time.perf_counter()

        if j_fn is None:
            rows_by_pol["V19_Frozen"] = [_run_frozen_baseline(sp, "V19_ADAPTIVE_RL", use_virtual_clock=False) for sp in specs]
        else:
            from experiments.rgro.task4_cohort_e_confirmation import _run_frozen_policy

            rows_by_pol["V19_Frozen"] = [_run_frozen_policy(sp, "V19_ADAPTIVE_RL", jitter_fn=j_fn) for sp in specs]

        rows_by_pol["Old_RGRO_U4"] = [
            run_rgro_episode(sp, cfg_old_u4, dfl_ensemble=ens_v19, use_virtual_clock=False, compute_jitter_ms_fn=j_fn)
            for sp in specs
        ]
        rows_by_pol["U4_DeadlineClosed"] = [
            run_closure_episode(
                sp,
                proposer_factory=lambda: RGROClosureProposer(cfg_closed_u4, dfl_ensemble=ens_v19),
                use_pdg2=True,
                use_virtual_clock=False,
                jitter_fn=j_fn,
            )
            for sp in specs
        ]
        rows_by_pol["U5_DeadlineClosed"] = [
            run_closure_episode(
                sp,
                proposer_factory=lambda: RGROClosureProposer(cfg_closed_u5, dfl_ensemble=ens_v19),
                use_pdg2=True,
                use_virtual_clock=False,
                jitter_fn=j_fn,
            )
            for sp in specs
        ]

        all_rows[s_name] = rows_by_pol
        decomps = {p: _lat_decomposition(rows_by_pol[p]) for p in rows_by_pol}
        results[s_name] = {
            "summary": decomps,
            "paired_vs_Old_U4": {
                p: paired_stats(rows_by_pol[p], rows_by_pol["Old_RGRO_U4"])
                for p in ("U4_DeadlineClosed", "U5_DeadlineClosed")
            },
            "paired_vs_V19": {
                p: paired_stats(rows_by_pol[p], rows_by_pol["V19_Frozen"])
                for p in ("Old_RGRO_U4", "U4_DeadlineClosed", "U5_DeadlineClosed")
            },
        }
        el = time.perf_counter() - t0
        print(f"  [{s_name} finished in {el:.1f}s]", flush=True)
        for p, d in decomps.items():
            print(
                f"    {p:18s} | Comp={d['comp100_count']:2d}/{d['n']} ExpW={d['expired_windows_total']} "
                f"| MeanJA={d['mean_ja']:8.2f} CVaR95={d['cvar95_ja']:8.2f} "
                f"| E2E P95/P99/Max={d['p95_ms']:5.2f}/{d['p99_ms']:5.2f}/{d['max_ms']:5.2f}ms (>35:{d['over35_rate_pct']:.2f}%) "
                f"| IntAlgoP99={d['internal_algo_p99_p99_max'][1] if False else d['internal_algo_p95_p99_max'][1]:5.2f} "
                f"ValP99={d['validate_commit_p95_p99_max'][1]:5.2f} FB_P99={d['fallback_p95_p99_max'][1]:5.2f} Vetoes={d['veto_count_total']}",
                flush=True,
            )

    # Pooled VAL-72 + DEV-Stress-24 latency check (direct analogue to Cohort E+E-Struct gate)
    pooled_96: dict[str, Any] = {}
    for p in ("V19_Frozen", "Old_RGRO_U4", "U4_DeadlineClosed", "U5_DeadlineClosed"):
        pooled_rows = all_rows["VAL72_StdDEV"][p] + all_rows["DEV_Stress24"][p]
        pooled_96[p] = _lat_decomposition(pooled_rows)

    payload = {
        "suites": results,
        "pooled_val72_stress24_96eps": pooled_96,
        "sla_config": {"sla_target_ms": cfg_closed_u4.sla_target_ms, "window_tight_sla_ms": cfg_closed_u4.window_tight_sla_ms},
    }
    (OUT_CLOSURE_DIR / "deliverable1_deadline_closure.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Deliverable 1 (`IF-CORE-RGRO-CLOSURE-01`): Deadline Closure & Zero-Expiry Hard Veto Report",
        "",
        f"> Quarantined DEV Evaluation (`VAL-72` + `DEV-Stress-24` + `DEV-Compute-24`) under uncontended `P2` real wall clock. `UnifiedDeadlineManager` (`sla_target_ms={cfg_closed_u4.sla_target_ms}`, `window_tight_sla_ms={cfg_closed_u4.window_tight_sla_ms}`) unifies Preamble, Anchor, PDG 2.0, Residual Search, DFL/LCB, Validate/Commit EMA reserve, and snapshot window slack `w_avail`.",
        "",
        "## 1. Pooled `VAL-72 + DEV-Stress-24` (`96` DEV Episodes) 4-Way Latency Decomposition",
        "",
        "| Policy | Comp100 | ExpWin | Vetoes | Fallbacks | Mean `J_A` | `CVaR_95` | `E2E P95/P99/Max` (ms) | `>35ms`% | `Internal Algo P95/P99/Max` | `OS Jitter P99/Max` | `Validate/Commit P99` | `Fallback P99/Max` (`N`) |",
        "| :--- | :---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | :---: | :---: | ---: | :---: |",
    ]
    for p, d in pooled_96.items():
        ia = d["internal_algo_p95_p99_max"]
        oj = d["os_jitter_p95_p99_max"]
        vc = d["validate_commit_p95_p99_max"]
        fb = d["fallback_p95_p99_max"]
        md.append(
            f"| `{p}` | {d['comp100_count']}/{d['n']} | {d['expired_windows_total']} | {d['veto_count_total']} | {d['fallback_count_total']} | "
            f"{d['mean_ja']:.2f} | {d['cvar95_ja']:.2f} | {d['p95_ms']:.2f}/{d['p99_ms']:.2f}/{d['max_ms']:.2f} | {d['over35_rate_pct']:.2f}% | "
            f"{ia[0]:.2f}/{ia[1]:.2f}/{ia[2]:.2f} | {oj[1]:.2f}/{oj[2]:.2f} | {vc[1]:.2f} | {fb[1]:.2f}/{fb[2]:.2f} ({d['fallback_n']}) |"
        )

    md += ["", "## 2. Per-Suite Latency & Quality Summary", ""]
    for s_name, _, _ in suites:
        md.append(f"### {s_name}")
        md.append("")
        md.append("| Policy | Comp100 | ExpWin | Mean `J_A` | `CVaR_95` | `E2E P95/P99/Max` (ms) | `>35ms`% | Vetoes | $\\Delta$ vs `V19_Frozen` |")
        md.append("| :--- | :---: | ---: | ---: | ---: | :---: | ---: | ---: | :--- |")
        for p, d in results[s_name]["summary"].items():
            if p == "V19_Frozen":
                vs = "Reference"
            else:
                ps = results[s_name]["paired_vs_V19"][p]
                vs = f"{ps['mean_delta']:+.2f} ({ps['rel_delta_pct']:+.2f}%, {ps['wins']}/{ps['ties']}/{ps['losses']}, p={ps['wilcoxon_p_2sided']:.4f})"
            md.append(
                f"| `{p}` | {d['comp100_count']}/{d['n']} | {d['expired_windows_total']} | {d['mean_ja']:.2f} | {d['cvar95_ja']:.2f} | "
                f"{d['p95_ms']:.2f}/{d['p99_ms']:.2f}/{d['max_ms']:.2f} | {d['over35_rate_pct']:.2f}% | {d['veto_count_total']} | {vs} |"
            )
        md.append("")

    (OUT_CLOSURE_DIR / "deliverable1_deadline_closure.md").write_text("\n".join(md), encoding="utf-8")
    print("=== [Task 1] COMPLETED! Wrote out/rgro_closure/deliverable1_deadline_closure.{json,md} ===", flush=True)


if __name__ == "__main__":
    main()
