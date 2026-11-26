"""IF-CORE-RGRO-CLOSURE-01 · Task 2 (P0): Engineering vs. Algorithm Attribution (`8` Comparators).

Cleanly separates software bug-fix gains (`V20.1-Patched` vs frozen `V20.1`) from genuine algorithmic
gains (`V19`, `U1: V19+PDG2`, `RGRO_Det`, `RGRO_DFL_noLCB`, `U4: RGRO_DFL_LCB`, `U5: RGRO_DFL_LCB_CRC`)
across quarantined DEV (`VAL-72`, `VAL-Static-35`, `DEV-Stress-24`) under BOTH `P1` and `P2` clocks,
sharing identical evaluator, budget, and P2 perturbation protocols.

Outputs:
  - `out/rgro_closure/deliverable2_eng_algo_attribution.{json,md}`
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from experiments.rgro.dfl_critic import DFLAdvantageEnsemble
from experiments.rgro.task2_rgro_ablation import _run_frozen_baseline
from experiments.rgro_closure.closure_common import (
    OUT_CLOSURE_DIR,
    assert_closure_dev_quarantine,
    build_dev_stress_24_specs,
    build_val72_specs,
    build_val_static35_specs,
    paired_stats,
    summarize_rows,
)
from experiments.rgro_closure.patched_v20_1 import V201PatchedProposer
from experiments.rgro_closure.rgro_closure_proposer import (
    ClosureConfig,
    RGROClosureProposer,
    run_closure_episode,
)


def main() -> None:
    OUT_CLOSURE_DIR.mkdir(parents=True, exist_ok=True)
    ens_v19 = DFLAdvantageEnsemble.load_npz(ROOT / "out" / "rgro" / "dfl_critic_v19.npz")

    val72 = build_val72_specs()
    static35 = build_val_static35_specs()
    stress24 = build_dev_stress_24_specs()
    assert_closure_dev_quarantine([int(s["seed"]) for s in val72 + static35 + stress24], label="Task2_Attribution")

    arms: list[tuple[str, str, Any]] = [
        ("V19_Frozen", "frozen", "V19_ADAPTIVE_RL"),
        ("U1_V19_PDG2", "closure", ClosureConfig(policy_id="U1_V19_PDG2", enable_pdg2=True, enable_v20_residual=False, critic_mode="none")),
        ("V20_1_Frozen", "frozen", "V20_1_NAJS"),
        ("V20_1_Patched", "patched_v20_1", None),
        ("RGRO_Det", "closure", ClosureConfig(policy_id="RGRO_Det", enable_pdg2=True, enable_v20_residual=True, critic_mode="none")),
        ("RGRO_DFL_noLCB", "closure", ClosureConfig(policy_id="RGRO_DFL_noLCB", enable_pdg2=True, enable_v20_residual=True, critic_mode="dfl_nolcb", lcb_kappa=0.0)),
        ("RGRO_DFL_LCB_U4", "closure", ClosureConfig(policy_id="RGRO_DFL_LCB_U4", enable_pdg2=True, enable_v20_residual=True, critic_mode="dfl_lcb", lcb_kappa=0.75)),
        ("RGRO_DFL_LCB_CRC_U5", "closure", ClosureConfig(policy_id="RGRO_DFL_LCB_CRC_U5", enable_pdg2=True, enable_v20_residual=True, critic_mode="dfl_lcb_crc", lcb_kappa=0.75, crc_harm_tau=0.15)),
    ]

    suites = [("VAL72", val72), ("VAL_Static35", static35), ("DEV_Stress24", stress24)]
    out: dict[str, Any] = {}

    for clock_name, use_vc in (("P1_Virtual", True), ("P2_WallClock", False)):
        out[clock_name] = {}
        pooled_by_arm: dict[str, list[dict[str, Any]]] = {a[0]: [] for a in arms}
        for s_name, specs in suites:
            rows_map: dict[str, list[dict[str, Any]]] = {}
            for a_name, kind, obj in arms:
                t0 = time.perf_counter()
                if kind == "frozen":
                    rows = [_run_frozen_baseline(sp, str(obj), use_virtual_clock=use_vc) for sp in specs]
                elif kind == "patched_v20_1":
                    rows = [
                        run_closure_episode(
                            sp,
                            proposer_factory=lambda s=sp: V201PatchedProposer(seed=int(s["seed"])),
                            base_env_policy="V19_ADAPTIVE_RL",
                            use_pdg2=True,
                            use_virtual_clock=use_vc,
                        )
                        for sp in specs
                    ]
                else:
                    cfg_obj: ClosureConfig = obj
                    rows = [
                        run_closure_episode(
                            sp,
                            proposer_factory=lambda c=cfg_obj: RGROClosureProposer(c, dfl_ensemble=ens_v19),
                            base_env_policy="V19_ADAPTIVE_RL",
                            use_pdg2=bool(cfg_obj.enable_pdg2),
                            use_virtual_clock=use_vc,
                        )
                        for sp in specs
                    ]
                rows_map[a_name] = rows
                pooled_by_arm[a_name].extend(rows)
                sm = summarize_rows(rows)
                print(
                    f"  [{clock_name:12s} | {s_name:12s}] {a_name:19s} | Comp={sm['comp100_count']:2d}/{sm['n']} ExpW={sm['expired_windows_total']} "
                    f"| MeanJA={sm['mean_ja']:8.2f} Med={sm['median_ja']:7.2f} CVaR95={sm['cvar95_ja']:8.2f} "
                    f"| P99={sm['p99_ms']:5.2f}ms ({time.perf_counter()-t0:.1f}s)",
                    flush=True,
                )

            out[clock_name][s_name] = {
                "summary": {a: summarize_rows(rows_map[a]) for a, _, _ in arms},
                "contrasts": {
                    "V20_1_Patched_vs_Frozen": paired_stats(rows_map["V20_1_Patched"], rows_map["V20_1_Frozen"]),
                    "U1_vs_V19": paired_stats(rows_map["U1_V19_PDG2"], rows_map["V19_Frozen"]),
                    "RGRO_Det_vs_U1": paired_stats(rows_map["RGRO_Det"], rows_map["U1_V19_PDG2"]),
                    "U4_vs_U1": paired_stats(rows_map["RGRO_DFL_LCB_U4"], rows_map["U1_V19_PDG2"]),
                    "U4_vs_V20_1_Patched": paired_stats(rows_map["RGRO_DFL_LCB_U4"], rows_map["V20_1_Patched"]),
                    "U4_vs_V19": paired_stats(rows_map["RGRO_DFL_LCB_U4"], rows_map["V19_Frozen"]),
                    "U4_vs_RGRO_Det": paired_stats(rows_map["RGRO_DFL_LCB_U4"], rows_map["RGRO_Det"]),
                    "U4_vs_DFL_noLCB": paired_stats(rows_map["RGRO_DFL_LCB_U4"], rows_map["RGRO_DFL_noLCB"]),
                    "U5_vs_U4": paired_stats(rows_map["RGRO_DFL_LCB_CRC_U5"], rows_map["RGRO_DFL_LCB_U4"]),
                },
            }

        # Pooled across all 131 DEV episodes (`VAL-72 + Static-35 + DEV-Stress-24`)
        out[clock_name]["POOLED_131_DEV"] = {
            "summary": {a: summarize_rows(pooled_by_arm[a]) for a, _, _ in arms},
            "contrasts": {
                "V20_1_Patched_vs_Frozen": paired_stats(pooled_by_arm["V20_1_Patched"], pooled_by_arm["V20_1_Frozen"]),
                "U1_vs_V19": paired_stats(pooled_by_arm["U1_V19_PDG2"], pooled_by_arm["V19_Frozen"]),
                "U4_vs_U1": paired_stats(pooled_by_arm["RGRO_DFL_LCB_U4"], pooled_by_arm["U1_V19_PDG2"]),
                "U4_vs_V20_1_Patched": paired_stats(pooled_by_arm["RGRO_DFL_LCB_U4"], pooled_by_arm["V20_1_Patched"]),
                "U4_vs_V19": paired_stats(pooled_by_arm["RGRO_DFL_LCB_U4"], pooled_by_arm["V19_Frozen"]),
                "U4_vs_RGRO_Det": paired_stats(pooled_by_arm["RGRO_DFL_LCB_U4"], pooled_by_arm["RGRO_Det"]),
                "U5_vs_U4": paired_stats(pooled_by_arm["RGRO_DFL_LCB_CRC_U5"], pooled_by_arm["RGRO_DFL_LCB_U4"]),
            },
        }

    (OUT_CLOSURE_DIR / "deliverable2_eng_algo_attribution.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = [
        "# Deliverable 2 (`IF-CORE-RGRO-CLOSURE-01`): Engineering vs. Algorithm Attribution Matrix",
        "",
        "> Clean 8-arm DEV attribution isolating the 4 engineering fixes (`V20.1-Patched` vs frozen `V20.1`) from algorithmic decomposition (`V19 -> U1 -> RGRO_Det -> DFL_noLCB -> U4_DFL_LCB -> U5_CRC`) across `131` quarantined DEV episodes (`VAL-72`, `VAL-Static-35`, `DEV-Stress-24`) under both `P1` virtual and `P2` wall clock.",
        "",
    ]
    for clk in ("P1_Virtual", "P2_WallClock"):
        md.append(f"## {clk}")
        md.append("")
        for s_key in ("VAL72", "VAL_Static35", "DEV_Stress24", "POOLED_131_DEV"):
            blk = out[clk][s_key]
            md.append(f"### {s_key} ({clk})")
            md.append("")
            md.append("| Comparator | Comp100 | ExpWin | Mean `J_A` | Med `J_A` | `CVaR_95` | `P95/P99/Max` (ms) |")
            md.append("| :--- | :---: | ---: | ---: | ---: | ---: | :---: |")
            for a, _, _ in arms:
                sm = blk["summary"][a]
                md.append(
                    f"| `{a}` | {sm['comp100_count']}/{sm['n']} | {sm['expired_windows_total']} | {sm['mean_ja']:.2f} | "
                    f"{sm['median_ja']:.2f} | {sm['cvar95_ja']:.2f} | {sm['p95_ms']:.2f}/{sm['p99_ms']:.2f}/{sm['max_ms']:.2f} |"
                )
            md.append("")
            md.append("**Key Paired Contrasts (95% Bootstrap CI & Wilcoxon `p`)**:")
            for ck, cp in blk["contrasts"].items():
                if not cp:
                    continue
                md.append(
                    f"- `{ck}`: `ΔMeanJA = {cp['mean_delta']:+.2f}` (`{cp['rel_delta_pct']:+.2f}%`), "
                    f"95% CI `[{cp['ci95_delta'][0]:+.2f}, {cp['ci95_delta'][1]:+.2f}]`, "
                    f"W/T/L=`{cp['wins']}/{cp['ties']}/{cp['losses']}`, Wilcoxon `p = {cp['wilcoxon_p_2sided']:.4f}`"
                )
            md.append("")

    (OUT_CLOSURE_DIR / "deliverable2_eng_algo_attribution.md").write_text("\n".join(md), encoding="utf-8")
    print("=== [Task 2] COMPLETED! Wrote out/rgro_closure/deliverable2_eng_algo_attribution.{json,md} ===", flush=True)


if __name__ == "__main__":
    main()
