"""`IF-CORE-ROUTING-CPSA-01` Controlled Routing Benchmark, 2^3 Factorial Falsification,
and Cross-Layer Diagnostic Study on Quarantined DEV Splits ONLY.

Strictly runs AFTER unthrottled Cohort F confirmation closes (`Step 0`), on quarantined
DEV splits (`VAL-72`, `DEV-Stress-24`, `VAL-Static-35` — zero Cohort F or A-E overlap).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import amr_sched.cvcr.evaluator as _ev_mod
import amr_sched.rl.v19_hetero_constructor as _v19_mod
from experiments.cps_astar_routing.cpr_incremental_router import (
    CPRIncrementalEvaluator,
    CPRRoutingConfig,
)
from experiments.cps_astar_routing.cps_astar_graph import (
    CPSAStarConfig,
    CPSAStarGraphEvaluator,
    CPSAStarGreedyMAPDProposer,
)
from experiments.rgro.dfl_critic import DFLAdvantageEnsemble
from experiments.rgro_closure.closure_common import (
    assert_closure_dev_quarantine,
    build_dev_stress_24_specs,
    paired_stats,
    pct,
    summarize_rows,
)
import experiments.rgro_closure.rgro_closure_proposer as _rgro_mod
from experiments.rgro_closure.rgro_closure_proposer import (
    ClosureConfig,
    RGROClosureProposer,
    run_closure_episode,
)
from experiments.v20_audit.common import build_val72_specs, build_val_static35_specs

OUT_STUDY_DIR = ROOT / "out" / "cps_astar_routing"


def _check_cpu_unthrottled() -> tuple[bool, float]:
    t0 = time.perf_counter()
    acc = 0
    for i in range(5_000_000):
        acc += i
    dt = time.perf_counter() - t0
    _ = acc
    return dt <= 1.25, round(dt, 3)


@contextmanager
def _patch_cvcr_evaluator(factory: Callable[..., Any]) -> Iterator[None]:
    orig_v19 = _v19_mod.CVCREvaluator
    orig_rgro = _rgro_mod.CVCREvaluator
    _v19_mod.CVCREvaluator = factory  # type: ignore[assignment]
    _rgro_mod.CVCREvaluator = factory  # type: ignore[assignment]
    try:
        yield
    finally:
        _v19_mod.CVCREvaluator = orig_v19
        _rgro_mod.CVCREvaluator = orig_rgro


class _LTMStaticQ3Evaluator(CVCREvaluator := _ev_mod.CVCREvaluator):  # type: ignore[misc]
    """Applies static incumbent-derived IJCAI-26 LTM edge Q3 penalties symmetrically."""

    def set_incumbent(self, assignment: dict[str, str], eval_result: Any, *args: Any, **kwargs: Any) -> None:
        super().set_incumbent(assignment, eval_result, *args, **kwargs)
        if eval_result.valid:
            ltm_w = self.compute_ltm_weights_from_paths(eval_result.robot_paths)
            if ltm_w:
                merged = dict(self._traffic_weights_q3)
                for ek, w in ltm_w.items():
                    merged[ek] = max(merged.get(ek, 0), w)
                self.set_traffic_weights(merged, cutoff_step=max(self._traffic_cutoff_step, self.t_commit + 36))


def _make_eval_factory(kernel: str) -> Callable[..., Any]:
    if kernel == "R0_HardRes":
        return _ev_mod.CVCREvaluator
    if kernel == "R1_LTM_StaticQ3":
        return _LTMStaticQ3Evaluator
    if kernel == "R2_CPSAStar_OneShot":
        return lambda net, snap, **kw: CPSAStarGraphEvaluator(net, snap, CPSAStarConfig(mode="oneshot"), **{**kw, "use_rust": False})
    if kernel == "R3_CPSAStar_Lifelong":
        return lambda net, snap, **kw: CPSAStarGraphEvaluator(net, snap, CPSAStarConfig(mode="lifelong"), **{**kw, "use_rust": False})
    if kernel == "R4_CPR_Distilled":
        cfg = CPRRoutingConfig(enable_slot_wait_pressure=True, enable_congestion_priority=True, enable_bounded_rollback=True)
        return lambda net, snap, **kw: CPRIncrementalEvaluator(net, snap, cfg, **kw)
    if kernel.startswith("FACT_"):
        bits = kernel.removeprefix("FACT_")
        cfg = CPRRoutingConfig(
            enable_slot_wait_pressure=(bits[0] == "W"),
            enable_congestion_priority=(bits[1] == "P"),
            enable_bounded_rollback=(bits[2] == "R"),
        )
        return lambda net, snap, **kw: CPRIncrementalEvaluator(net, snap, cfg, **kw)
    raise ValueError(f"Unknown kernel: {kernel}")


def _run_upper_with_kernel(
    specs: list[dict[str, Any]],
    upper_name: str,
    kernel_name: str,
    *,
    use_virtual: bool,
    ens_v19: DFLAdvantageEnsemble,
) -> dict[str, Any]:
    ev_factory = _make_eval_factory(kernel_name)
    rec_flags = {"enable_o1_opening": True, "enable_o2_queue": True, "enable_o3_diverse": False}
    rows: list[dict[str, Any]] = []

    def _mk_prop() -> Any:
        if upper_name == "V19_ADAPTIVE_RL":
            return None
        cfg = ClosureConfig(policy_id=f"{upper_name}+{kernel_name}", critic_mode="dfl_lcb", sla_target_ms=20.5, window_tight_sla_ms=17.5, **rec_flags)
        return RGROClosureProposer(cfg, dfl_ensemble=ens_v19)

    with _patch_cvcr_evaluator(ev_factory):
        for sp in specs:
            if upper_name == "V19_ADAPTIVE_RL":
                row = run_closure_episode(
                    sp, proposer_factory=None, base_env_policy="V19_ADAPTIVE_RL",
                    use_pdg2=False, use_virtual_clock=use_virtual,
                )
                row["policy"] = f"V19+{kernel_name}"
            else:
                row = run_closure_episode(
                    sp, proposer_factory=_mk_prop, base_env_policy="V19_ADAPTIVE_RL",
                    use_pdg2=True, use_virtual_clock=use_virtual,
                )
            rows.append(row)
    return {"summary": summarize_rows(rows), "rows": rows}


def _evaluate_dev_gates(
    base_stress: dict[str, Any],
    cand_stress: dict[str, Any],
    base_val72: dict[str, Any],
    cand_val72: dict[str, Any],
    base_st35: dict[str, Any],
    cand_st35: dict[str, Any],
) -> dict[str, Any]:
    s_b, s_c = base_stress["summary"], cand_stress["summary"]
    v_b, v_c = base_val72["summary"], cand_val72["summary"]
    t_b, t_c = base_st35["summary"], cand_st35["summary"]

    stress_gain_pct = 100.0 * (s_b["mean_ja"] - s_c["mean_ja"]) / max(1e-9, s_b["mean_ja"])
    g1_stress = (stress_gain_pct >= 3.0) or (s_c["comp100_count"] > s_b["comp100_count"])
    g2_std = (
        v_c["comp100_count"] >= v_b["comp100_count"]
        and t_c["comp100_count"] >= t_b["comp100_count"]
        and v_c["mean_ja"] <= 1.005 * v_b["mean_ja"]
        and t_c["mean_ja"] <= 1.005 * t_b["mean_ja"]
    )
    g3_rt = (
        v_c["p95_ms"] <= 17.5 and v_c["p99_ms"] <= 20.5 and v_c["expired_windows_total"] == 0
        and s_c["p99_ms"] <= 20.5 and s_c["expired_windows_total"] == 0
    )
    g4_audit = (v_c["audit_pass_rate"] >= 0.9999) and (s_c["audit_pass_rate"] >= 0.9999) and (t_c["audit_pass_rate"] >= 0.9999)
    gates = {"G1_Stress_Gain": g1_stress, "G2_Std_No_Regress": g2_std, "G3_Realtime_SLA": g3_rt, "G4_Audit_100": g4_audit}
    return {"gates": gates, "stress_gain_pct": round(stress_gain_pct, 2), "all_pass": all(gates.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke check on 2 stress + 2 val72 episodes without CPU speed lock.")
    parser.add_argument("--allow-throttled", action="store_true", help="Override CPU clock health guard (for debugging ONLY).")
    args = parser.parse_args()

    ok_cpu, loop_s = _check_cpu_unthrottled()
    print(f"[Guard] 5M Python loop = {loop_s:.3f}s (unthrottled <= 1.25s: {ok_cpu})", flush=True)
    if (not ok_cpu) and (not args.allow_throttled) and (not args.dry_run):
        raise RuntimeError(
            f"Host CPU appears throttled (5M loop={loop_s:.3f}s > 1.25s). Clear CPU throttle before running P2_WallClock study, or pass --dry-run."
        )

    val72 = build_val72_specs()
    stress24 = build_dev_stress_24_specs()
    static35 = build_val_static35_specs()
    assert_closure_dev_quarantine([int(s["seed"]) for s in val72 + stress24 + static35], label="CPS_A*_DEV_Study")

    if args.dry_run:
        val72, stress24, static35 = val72[:2], stress24[:2], static35[:2]

    ens_v19 = DFLAdvantageEnsemble.load_npz(ROOT / "out" / "rgro" / "dfl_critic_v19.npz")
    OUT_STUDY_DIR.mkdir(parents=True, exist_ok=True)

    # Experiment A: Controlled lower-layer comparison under identical RGRO_U4_Closure & V19_ADAPTIVE_RL
    exp_a: dict[str, Any] = {}
    for upper in ("RGRO_U4_Closure", "V19_ADAPTIVE_RL"):
        exp_a[upper] = {}
        for k_name in ("R0_HardRes", "R1_LTM_StaticQ3", "R2_CPSAStar_OneShot", "R3_CPSAStar_Lifelong", "R4_CPR_Distilled"):
            res_v = _run_upper_with_kernel(val72, upper, k_name, use_virtual=False, ens_v19=ens_v19)
            res_s = _run_upper_with_kernel(stress24, upper, k_name, use_virtual=False, ens_v19=ens_v19)
            exp_a[upper][k_name] = {"VAL72": res_v["summary"], "DEV_Stress24": res_s["summary"]}
            print(f"[Exp A] {upper:16s} | {k_name:22s} | Stress MeanJA={res_s['summary']['mean_ja']:8.2f} P99={res_s['summary']['p99_ms']:5.2f}ms", flush=True)

    # Experiment B: 2^3 Factorial ablation on RGRO_U4_Closure under P1_Virtual and P2_WallClock
    exp_b: dict[str, Any] = {"P1_Virtual": {}, "P2_WallClock": {}}
    fact_codes = ("000", "W00", "0P0", "00R", "WP0", "W0R", "0PR", "WPR")
    p2_raw: dict[str, dict[str, Any]] = {}
    for clk_name, is_v in (("P1_Virtual", True), ("P2_WallClock", False)):
        for code in fact_codes:
            k_name = f"FACT_{code}"
            rv = _run_upper_with_kernel(val72, "RGRO_U4_Closure", k_name, use_virtual=is_v, ens_v19=ens_v19)
            rs = _run_upper_with_kernel(stress24, "RGRO_U4_Closure", k_name, use_virtual=is_v, ens_v19=ens_v19)
            rt = _run_upper_with_kernel(static35, "RGRO_U4_Closure", k_name, use_virtual=is_v, ens_v19=ens_v19)
            exp_b[clk_name][code] = {"VAL72": rv["summary"], "DEV_Stress24": rs["summary"], "VAL_Static35": rt["summary"]}
            if not is_v:
                p2_raw[code] = {"VAL72": rv, "DEV_Stress24": rs, "VAL_Static35": rt}

    gates_wpr = _evaluate_dev_gates(
        p2_raw["000"]["DEV_Stress24"], p2_raw["WPR"]["DEV_Stress24"],
        p2_raw["000"]["VAL72"], p2_raw["WPR"]["VAL72"],
        p2_raw["000"]["VAL_Static35"], p2_raw["WPR"]["VAL_Static35"],
    )
    paired_stress_wpr = paired_stats(p2_raw["WPR"]["DEV_Stress24"]["rows"], p2_raw["000"]["DEV_Stress24"]["rows"])

    # Experiment C: Cross-layer diagnostic on DEV-Stress-24
    greedy_rows = [
        run_closure_episode(
            sp,
            proposer_factory=lambda: CPSAStarGreedyMAPDProposer(),
            base_env_policy="V19_ADAPTIVE_RL",
            use_pdg2=False,
            use_virtual_clock=False,
        )
        for sp in stress24
    ]
    ral_rows = [
        run_closure_episode(
            sp,
            proposer_factory=None,
            base_env_policy="RAL26_DetourDelta",
            use_pdg2=False,
            use_virtual_clock=False,
        )
        for sp in stress24
    ]
    exp_c = {
        "CPSAStar_GreedyMAPD": summarize_rows(greedy_rows),
        "RAL26_DetourDelta": summarize_rows(ral_rows),
        "RGRO_U4+R4_CPR": p2_raw["WPR"]["DEV_Stress24"]["summary"],
    }

    report = {
        "protocol": "IF-CORE-ROUTING-CPSA-01 DEV Study",
        "dry_run": bool(args.dry_run),
        "cpu_5m_loop_s": loop_s,
        "exp_a_controlled_routing": exp_a,
        "exp_b_2x3_factorial": exp_b,
        "paired_wpr_vs_000_stress": paired_stress_wpr,
        "dev_falsification_gates_wpr": gates_wpr,
        "exp_c_cross_layer_diagnostic": exp_c,
    }
    (OUT_STUDY_DIR / "cpsa_routing_study_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# `IF-CORE-ROUTING-CPSA-01`: CPS-A* Controlled Routing & 2^3 Factorial DEV Report",
        "",
        f"- **DEV Falsification Gates (`WPR` vs `000`, `P2_WallClock`)**: `{gates_wpr['gates']}` -> **{'PASS' if gates_wpr['all_pass'] else 'FAIL'}** (`Stress Gain={gates_wpr['stress_gain_pct']:+.2f}%`)",
        "",
        "## Exp A: Controlled Lower-Layer Routing Comparison (`RGRO_U4_Closure`)",
        "| Kernel | VAL72 Comp | VAL72 Mean `J_A` | VAL72 P99 (ms) | Stress24 Comp | Stress24 Mean `J_A` | Stress24 P99 (ms) |",
        "| :--- | :---: | ---: | ---: | :---: | ---: | ---: |",
    ]
    for k_name, d_k in exp_a["RGRO_U4_Closure"].items():
        sv, ss = d_k["VAL72"], d_k["DEV_Stress24"]
        md.append(f"| `{k_name}` | {sv['comp100_count']}/{sv['n']} | {sv['mean_ja']:.2f} | {sv['p99_ms']:.2f} | {ss['comp100_count']}/{ss['n']} | {ss['mean_ja']:.2f} | {ss['p99_ms']:.2f} |")
    md.extend([
        "",
        "## Exp C: Cross-Layer Diagnostic (`DEV-Stress-24`)",
        "| Solver | Comp100 | Mean `J_A` | Med `J_A` | P99 (ms) |",
        "| :--- | :---: | ---: | ---: | ---: |",
    ])
    for name_c, sc in exp_c.items():
        md.append(f"| `{name_c}` | {sc['comp100_count']}/{sc['n']} | {sc['mean_ja']:.2f} | {sc['median_ja']:.2f} | {sc['p99_ms']:.2f} |")

    (OUT_STUDY_DIR / "cpsa_routing_study_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"=== Wrote {OUT_STUDY_DIR / 'cpsa_routing_study_report.md'} | Gates={gates_wpr['gates']} ===", flush=True)


if __name__ == "__main__":
    main()
