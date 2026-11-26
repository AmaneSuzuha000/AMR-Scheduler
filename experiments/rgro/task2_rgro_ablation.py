"""IF-CORE-RGRO-01 · Task 2: Unified `U0..U5` VAPI + Shared-Budget `RGRO` DEV Ablation.

Evaluates the full incremental ladder on quarantined DEV suites (`VAL-72`, `VAL-Static-35`,
and `DEV-Stress-24`) under BOTH `P1` (deterministic virtual work clock) AND `P2` (uncontended
real wall clock), alongside frozen `V19_ADAPTIVE_RL` (`R0`) and frozen `V20_1_NAJS` (`R1`):

  - `R0_V19_Frozen` : Exact frozen production `V19_ADAPTIVE_RL`
  - `R1_V20_1_Frozen`: Exact frozen production `V20_1_NAJS`
  - `U1_V19_PDG2`   : `V19` Anchor + Independent `PDG 2.0` (minimal liveness upgrade tested FIRST!)
  - `U2_U1_V18Polish`: `U1` + Low-Cost `V18` Residual Polish (Mode 1 typical-instance quality)
  - `U3_U1_V20Res`  : `U1` + Shared-Encoder `V20.1` Residual Search (deterministic `JA_Unified` gate)
  - `U4_U3_DFL_LCB` : `U3` + Relative-Advantage `DFL` Ensemble + Epistemic `LCB` (`A_LCB >= tau_gain`)
  - `U5_U4_CRC_Gate`: `U4` + Episode-Grouped `CRC` Harm-Risk Gate (`p_harm <= tau_crc*`)

Outputs:
  - `out/rgro/deliverable2_rgro_ablation.{json,md}` (Deliverable 2)
"""

from __future__ import annotations

import copy
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

from amr_sched.baselines.policy_registry import create_discrete_policy_env
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state
from experiments.rgro.dfl_critic import DFLAdvantageEnsemble
from experiments.rgro.rgro_common import (
    OUT_RGRO_DIR,
    assert_dev_quarantine,
    build_dev_stress_24_specs,
    paired_stats,
    summarize_rows,
)
from experiments.rgro.rgro_proposer import make_ladder_config, run_rgro_episode
from experiments.v20_audit.common import build_val72_specs, build_val_static35_specs


def _run_frozen_baseline(spec: dict[str, Any], policy_name: str, *, use_virtual_clock: bool) -> dict[str, Any]:
    def _exec() -> dict[str, Any]:
        _, ctrl, env = create_discrete_policy_env(
            policy_name,
            seed=int(spec["seed"]),
            max_steps=int(spec["max_steps"]),
            use_rust=True,
            p2_cfg=spec.get("p2_cfg"),
        )
        obs, info = env.reset(seed=int(spec["seed"]), options={"scenario_tuple": copy.deepcopy(spec["scenario_tuple"])})
        prop = getattr(env, "_raw_external_proposer", None)
        mon = getattr(env, "_pdg_monitor", None)
        tm = tr = False
        while not (tm or tr):
            feats = compute_expert_v9_state(
                env.sim, env.tracker, last_replan_tick=env.last_replan_tick,
                recent_conflict_ticks=env.recent_conflict_ticks, decision_logs=env.decision_logs,
            )
            s_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
            act = int(ctrl.select_action(feats, s_info, sim=env.sim))
            if mon is not None:
                act = int(mon.observe_and_maybe_escalate(env, s_info, act, proposer=prop))
            obs, _, tm, tr, info = env.step(act)

        m = env.compute_episode_metrics()
        aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
        dec_ms = [float(r["t_propose_ms"]) + float(r["t_validate_commit_ms"]) for r in env.decision_logs]
        return {
            "policy": policy_name,
            "episode_key": str(spec["episode_key"]),
            "suite_group": str(spec.get("suite_group", "")),
            "seed": int(spec["seed"]),
            "net_obj": round(float(m["net_obj"]), 4),
            "h1": round(float(m["cmax"]) + float(m.get("task_tardiness_cost", 0.0)) + float(m.get("op_tardiness_cost", 0.0)), 4),
            "churn": int(m["total_churn"]),
            "comp100": bool(float(m["task_completion_rate"]) >= 0.9999 and float(m["production_completion_rate"]) >= 0.9999),
            "audit_passed": bool(aud.pass_audit),
            "expired_windows": int(env.total_window_expirations),
            "pdg_alerts": len(mon.alert_ticks) if mon is not None else 0,
            "res_evals": 0,
            "res_accepted": 0,
            "decision_latencies_ms": [round(float(x), 4) for x in dec_ms],
        }

    if use_virtual_clock:
        with virtual_clock_scope(0.0):
            return _exec()
    return _exec()


def main() -> None:
    OUT_RGRO_DIR.mkdir(parents=True, exist_ok=True)
    ens_v19 = DFLAdvantageEnsemble.load_npz(OUT_RGRO_DIR / "dfl_critic_v19.npz")

    lock_path = OUT_RGRO_DIR / "crc_calibration_lock.json"
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        tau_crc = float(lock["locked_tau_crc"])
        tau_gain = float(lock.get("locked_dfl_tau_gain", 5.0))
        kappa = float(lock.get("locked_lcb_kappa", 0.50))
    else:
        tau_crc, tau_gain, kappa = 0.25, 5.0, 0.50

    val72 = build_val72_specs()
    static35 = build_val_static35_specs()
    stress24 = build_dev_stress_24_specs()
    assert_dev_quarantine([int(s["seed"]) for s in val72 + static35 + stress24], label="Task2_DEV_Ablation")

    variants = [
        ("R0_V19_Frozen", "frozen", "V19_ADAPTIVE_RL"),
        ("R1_V20_1_Frozen", "frozen", "V20_1_NAJS"),
        ("U1_V19_PDG2", "rgro", make_ladder_config("U1")),
        ("U2_U1_V18Polish", "rgro", make_ladder_config("U2")),
        ("U3_U1_V20Res", "rgro", make_ladder_config("U3")),
        ("U4_U3_DFL_LCB", "rgro", make_ladder_config("U4", dfl_tau_gain=tau_gain, lcb_kappa=kappa)),
        ("U5_U4_CRC_Gate", "rgro", make_ladder_config("U5", crc_harm_tau=tau_crc, dfl_tau_gain=tau_gain, lcb_kappa=kappa)),
    ]

    print(
        f"=== [Task 2] Unified U0..U5 + RGRO DEV Ablation (VAL-72 + Static-35 + DEV-Stress-24; "
        f"tau_crc={tau_crc:.3f}, tau_gain={tau_gain:.1f}, kappa={kappa:.2f}) ===",
        flush=True,
    )

    suites = [
        ("VAL72", val72),
        ("VAL_Static35", static35),
        ("DEV_Stress24", stress24),
    ]

    all_results: dict[str, Any] = {"crc_lock_used": {"tau_crc": tau_crc, "tau_gain": tau_gain, "kappa": kappa}}
    for clock_name, use_vc in (("P1_Virtual", True), ("P2_WallClock", False)):
        all_results[clock_name] = {}
        for suite_name, specs in suites:
            rows_by_v: dict[str, list[dict[str, Any]]] = {}
            for v_name, kind, obj in variants:
                t0 = time.perf_counter()
                if kind == "frozen":
                    rows = [_run_frozen_baseline(sp, str(obj), use_virtual_clock=use_vc) for sp in specs]
                else:
                    rows = [
                        run_rgro_episode(sp, obj, dfl_ensemble=ens_v19, use_virtual_clock=use_vc)
                        for sp in specs
                    ]
                rows_by_v[v_name] = rows
                sm = summarize_rows(rows)
                r_acc = sum(int(r.get("res_accepted", 0)) for r in rows)
                r_ev = sum(int(r.get("res_evals", 0)) for r in rows)
                pdg_cnt = sum(int(r.get("pdg_alerts", 0)) for r in rows)
                print(
                    f"  [{clock_name:12s} | {suite_name:12s}] {v_name:16s} | Comp={sm['comp100_count']:2d}/{sm['n']} "
                    f"| MeanJA={sm['mean_ja']:8.2f} Med={sm['median_ja']:7.2f} CVaR95={sm['cvar95_ja']:8.2f} "
                    f"| P95/P99/Max={sm['p95_ms']:5.2f}/{sm['p99_ms']:5.2f}/{sm['max_ms']:5.2f}ms (>35:{sm['over35_rate_pct']:.2f}%) "
                    f"| PDG={pdg_cnt} Res={r_acc}/{r_ev} ({time.perf_counter()-t0:.1f}s)",
                    flush=True,
                )

            all_results[clock_name][suite_name] = {
                "summary": {vn: summarize_rows(rows_by_v[vn]) for vn, _, _ in variants},
                "paired_vs_R0_V19": {
                    vn: paired_stats(rows_by_v[vn], rows_by_v["R0_V19_Frozen"])
                    for vn, _, _ in variants[1:]
                },
                "paired_vs_R1_V20_1": {
                    vn: paired_stats(rows_by_v[vn], rows_by_v["R1_V20_1_Frozen"])
                    for vn, _, _ in variants
                    if vn != "R1_V20_1_Frozen"
                },
                "marginal_step_deltas": {
                    "U1_vs_R0": paired_stats(rows_by_v["U1_V19_PDG2"], rows_by_v["R0_V19_Frozen"]),
                    "U2_vs_U1": paired_stats(rows_by_v["U2_U1_V18Polish"], rows_by_v["U1_V19_PDG2"]),
                    "U3_vs_U2": paired_stats(rows_by_v["U3_U1_V20Res"], rows_by_v["U2_U1_V18Polish"]),
                    "U4_vs_U3": paired_stats(rows_by_v["U4_U3_DFL_LCB"], rows_by_v["U3_U1_V20Res"]),
                    "U5_vs_U4": paired_stats(rows_by_v["U5_U4_CRC_Gate"], rows_by_v["U4_U3_DFL_LCB"]),
                },
            }

    # Pick the strictly best DEV candidate among U1..U5 (scored on P2_WallClock pooled VAL72 + DEV_Stress24)
    cand_pool = ("U1_V19_PDG2", "U2_U1_V18Polish", "U3_U1_V20Res", "U4_U3_DFL_LCB", "U5_U4_CRC_Gate")
    dev_scores: dict[str, dict[str, Any]] = {}
    p2_v72 = all_results["P2_WallClock"]["VAL72"]["summary"]
    p2_st = all_results["P2_WallClock"]["VAL_Static35"]["summary"]
    p2_ds = all_results["P2_WallClock"]["DEV_Stress24"]["summary"]
    for vn in cand_pool:
        pooled_mean = (72.0 * p2_v72[vn]["mean_ja"] + 24.0 * p2_ds[vn]["mean_ja"]) / 96.0
        max_p99 = max(p2_v72[vn]["p99_ms"], p2_ds[vn]["p99_ms"])
        dev_scores[vn] = {
            "pooled_val72_stress24_ja": round(pooled_mean, 2),
            "val72_ja": p2_v72[vn]["mean_ja"],
            "static35_ja": p2_st[vn]["mean_ja"],
            "stress24_ja": p2_ds[vn]["mean_ja"],
            "max_p99_ms": max_p99,
            "val72_cvar95": p2_v72[vn]["cvar95_ja"],
            "passes_dev_latency_gate": bool(max_p99 <= 23.0),
        }
    eligible_dev = [vn for vn in cand_pool if dev_scores[vn]["passes_dev_latency_gate"]]
    if not eligible_dev:
        eligible_dev = list(cand_pool)
    selected_dev_winner = min(eligible_dev, key=lambda vn: dev_scores[vn]["pooled_val72_stress24_ja"])
    all_results["dev_candidate_selection"] = {
        "scores": dev_scores,
        "selected_winner_for_cohort_e": selected_dev_winner,
        "always_also_confirm_U1_and_U5": True,
    }
    print(f"  [DEV WINNER] Selected={selected_dev_winner} | scores={dev_scores[selected_dev_winner]}", flush=True)

    (OUT_RGRO_DIR / "deliverable2_rgro_ablation.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    # Markdown Deliverable 2
    md: list[str] = [
        "# Deliverable 2 (`IF-CORE-RGRO-01`): Unified `U0..U5` VAPI + Shared-Budget `RGRO` DEV Ablation",
        "",
        f"> Quarantined DEV Suites: `VAL-72` (72 eps), `VAL-Static-35` (35 eps), `DEV-Stress-24` (24 eps) — evaluated under BOTH deterministic `P1` work clock and uncontended `P2` wall clock. Locked DEV CRC threshold: $\\tau_{{\\mathrm{{CRC}}}} = {tau_crc:.3f}$, $\\tau_{{\\mathrm{{gain}}}} = {tau_gain:.1f}$, $\\kappa = {kappa:.2f}$. Selected DEV winner: **`{selected_dev_winner}`**.",
        "",
    ]
    for clock_name in ("P1_Virtual", "P2_WallClock"):
        md.append(f"## {clock_name}")
        md.append("")
        for suite_name, _ in suites:
            blk = all_results[clock_name][suite_name]
            md.append(f"### {suite_name} ({clock_name})")
            md.append("")
            md.append("| Variant | Comp100 | Mean `J_A` | Med `J_A` | `CVaR_95` | `P95/P99/Max` (ms) | `>35ms`% | $\\Delta$ vs Frozen `V19` (W/T/L, p_2s) |")
            md.append("| :--- | :---: | ---: | ---: | ---: | :---: | ---: | :--- |")
            for vn, _, _ in variants:
                sm = blk["summary"][vn]
                if vn == "R0_V19_Frozen":
                    vs = "Reference"
                else:
                    ps = blk["paired_vs_R0_V19"][vn]
                    vs = f"{ps['mean_delta']:+.2f} ({ps['rel_delta_pct']:+.2f}%, {ps['wins']}/{ps['ties']}/{ps['losses']}, p={ps['wilcoxon_p_2sided']:.4f})"
                md.append(
                    f"| `{vn}` | {sm['comp100_count']}/{sm['n']} | {sm['mean_ja']:.2f} | {sm['median_ja']:.2f} | "
                    f"{sm['cvar95_ja']:.2f} | {sm['p95_ms']:.2f}/{sm['p99_ms']:.2f}/{sm['max_ms']:.2f} | {sm['over35_rate_pct']:.2f}% | {vs} |"
                )
            md.append("")
            md.append("**Marginal Stepwise Deltas (`U1`→`U5`)**:")
            for mk, mp in blk["marginal_step_deltas"].items():
                md.append(
                    f"- `{mk}`: `ΔMeanJA = {mp['mean_delta']:+.2f}` (`{mp['rel_delta_pct']:+.2f}%`), W/T/L=`{mp['wins']}/{mp['ties']}/{mp['losses']}`, Wilcoxon 2-sided `p = {mp['wilcoxon_p_2sided']:.4f}`"
                )
            md.append("")

    (OUT_RGRO_DIR / "deliverable2_rgro_ablation.md").write_text("\n".join(md), encoding="utf-8")
    print("=== [Task 2] COMPLETED! Wrote Deliverable 2 -> out/rgro/deliverable2_rgro_ablation.{json,md} ===", flush=True)


if __name__ == "__main__":
    main()
