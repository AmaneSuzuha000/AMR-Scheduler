"""Gate G1: Pure Zero-Fault Static Common-Subproblem Benchmark (`EXP-10-G1`).

Apples-to-apples evaluation on 54 pure zero-fault static instances (all mid-episode dynamic
faults and urgent insertions stripped via `strip_dynamic_disruptions`) from the fresh seed
cohort (`27101..27302` for 4-Topo, `2901..2906` for Track 1):
  - 24 instances: `4-Topo standard` (zero faults)
  - 24 instances: `4-Topo window_bottleneck` (zero faults)
  -  6 instances: `Track 1 static` (62-node Rail Freight Workshop, 6 AMRs, zero faults)

Evaluates 5 solvers inside the EXACT SAME zero-fault `AMRReschedulingEnv` + `Simulator` +
`IndependentAuditor` plant on pure physical efficiency `H1 = C_max + 2 * T_task + 2 * T_op`:
  1. `COR26-Ref` (van Os & Basten, COR 187, 2025/2026): Offline CP-SAT Physical Relaxation
     Master (`LB_phys` frozen strictly at Phase 1) + post-LB search-diversification nogood
     cuts + exact Simulator-verified `UB_sim` (`15 s` offline budget).
  2. `EJOR26-Erdogan` (Erdogan et al., EJOR 2026): Offline ILS + Exact Earliest-Start Timing
     Matheuristic + exact Simulator-verified `H1` (`15 s` offline budget).
  3. `CoRe-Sched` (frozen online millisecond solver, `35 ms` decision budget).
  4. `V18-EA` (online evolutionary local search, `35 ms` decision budget).
  5. `IJCAI26-PGMPO` (online FJSP neural baseline, `35 ms` decision budget).

Mathematically asserts `LB_phys <= H1_sim + 1e-3` across EVERY completed audited execution
of ALL 5 solvers across ALL 54 instances (`0` relaxation violations allowed!).

Writes:
  - `out/v19_scientific_audit/exp10_g1_static_reference.{json,md}`
  - `results/g1_static_reference_subproblem.csv` (clean public verification ledger)
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.baselines.erdogan_ejor26 import solve_erdogan_ils_lp_ejor26
from amr_sched.baselines.lbbd_cp_cor26 import solve_lbbd_cp_cor26, strip_dynamic_disruptions
from amr_sched.baselines.policy_registry import run_discrete_episode
from experiments.train_v19_bc import _build_4topo_episode_spec, _build_workshop_episode_spec
from experiments.v18_confirmation_scenarios import LAYOUTS as TOPO4_LAYOUTS

OUT_DIR = ROOT / "out" / "v19_scientific_audit"
RESULTS_DIR = ROOT / "results"

FRESH_4TOPO_SEEDS = (27101, 27102, 27201, 27202, 27301, 27302)
FRESH_WORKSHOP_SEEDS = (2901, 2902, 2903, 2904, 2905, 2906)


def _row_h1(row: dict[str, Any]) -> float:
    return float(row["cmax"] + row["task_tardiness_cost"] + row["op_tardiness_cost"])


def _row_comp_100(row: dict[str, Any]) -> bool:
    t_c = float(row.get("task_comp", row.get("task_completion_rate", 0.0)))
    p_c = float(row.get("prod_comp", row.get("production_completion_rate", 0.0)))
    return bool(t_c >= 0.9999 and p_c >= 0.9999 and bool(row.get("audit_passed", False)))


def run_g1_static_reference(time_limit_s: float = 15.0) -> dict[str, Any]:
    t0 = time.perf_counter()

    specs: list[dict[str, Any]] = []
    for s in FRESH_4TOPO_SEEDS:
        for lay in TOPO4_LAYOUTS:
            for reg in ("standard", "window_bottleneck"):
                sp = _build_4topo_episode_spec(lay, s, reg)
                sp["scenario_tuple"] = strip_dynamic_disruptions(sp["scenario_tuple"])
                sp["group"] = f"4topo_{reg}"
                specs.append(sp)
    for s in FRESH_WORKSHOP_SEEDS:
        sp = _build_workshop_episode_spec("track1", s, "nominal")
        sp["scenario_tuple"] = strip_dynamic_disruptions(sp["scenario_tuple"])
        sp["group"] = "track1_static"
        specs.append(sp)

    rows: list[dict[str, Any]] = []
    n_violations = 0
    core_all_lats_ms: list[float] = []

    for idx, sp in enumerate(specs, 1):
        ek = sp["episode_key"]
        sc_static = sp["scenario_tuple"]
        m_steps = int(sp["max_steps"])
        seed = int(sp["seed"])

        # 1. Offline COR26-Ref (Phase-1 valid LB_phys + Simulator-verified UB_sim)
        res_cor = solve_lbbd_cp_cor26(
            copy.deepcopy(sc_static),
            time_limit_s=time_limit_s,
            max_iters=25,
            num_workers=8,
            max_steps=m_steps,
            verify_in_simulator=True,
            seed=seed,
        )

        # 2. Offline EJOR26-Erdogan (ILS + exact timing + Simulator verification)
        res_erd = solve_erdogan_ils_lp_ejor26(
            copy.deepcopy(sc_static),
            time_limit_s=time_limit_s,
            max_steps=m_steps,
            seed=seed,
        )

        # 3-5. Online millisecond solvers on the EXACT SAME pure zero-fault static scenario
        online_res: dict[str, dict[str, Any]] = {}
        for pol in ("V19_ADAPTIVE_RL", "V18_EA_JOINT", "IJCAI26_PGMPO"):
            ep_out = run_discrete_episode(
                pol,
                copy.deepcopy(sc_static),
                seed=seed,
                max_steps=m_steps,
                clock_mode="wall",
                include_decision_latencies=(pol == "V19_ADAPTIVE_RL"),
            )
            online_res[pol] = ep_out
            if pol == "V19_ADAPTIVE_RL":
                core_all_lats_ms.extend(float(v) for v in ep_out.get("decision_latencies_ms", []))

        lb = float(res_cor.lower_bound)
        cor_ub = float(res_cor.upper_bound) if (res_cor.sim_audit_passed and res_cor.sim_100pct_completed) else None
        erd_h1 = float(res_erd.sim_h1_obj) if (res_erd.sim_audit_passed and res_erd.sim_100pct_completed) else None

        core_r = online_res["V19_ADAPTIVE_RL"]
        v18_r = online_res["V18_EA_JOINT"]
        pg_r = online_res["IJCAI26_PGMPO"]

        core_h1 = round(_row_h1(core_r), 2)
        core_comp = _row_comp_100(core_r)
        v18_h1 = round(_row_h1(v18_r), 2)
        v18_comp = _row_comp_100(v18_r)
        pg_h1 = round(_row_h1(pg_r), 2)
        pg_comp = _row_comp_100(pg_r)

        # Strict relaxation soundness assertion across ALL completed audited executions!
        sound = True
        for name_s, val_s, ok_s in [
            ("COR26-Ref", cor_ub, cor_ub is not None),
            ("EJOR26-Erdogan", erd_h1, erd_h1 is not None),
            ("CoRe-Sched", core_h1, core_comp),
            ("V18-EA", v18_h1, v18_comp),
            ("IJCAI26-PGMPO", pg_h1, pg_comp),
        ]:
            if ok_s and val_s is not None and lb > float(val_s) + 1e-3:
                sound = False
                n_violations += 1
                raise AssertionError(
                    f"Gate G1 RELAXATION SOUNDNESS VIOLATION on {ek} ({name_s}): LB_phys={lb} > H1_sim={val_s}"
                )

        rows.append(
            {
                "episode_key": ek,
                "group": sp["group"],
                "lb_phys": round(lb, 2),
                "cor26_status": res_cor.status,
                "cor26_sim_ub": round(cor_ub, 2) if cor_ub is not None else None,
                "cor26_audit_ok": bool(res_cor.sim_audit_passed and res_cor.sim_100pct_completed),
                "cor26_gap_pct": res_cor.gap_pct,
                "cor26_iters": res_cor.iterations,
                "cor26_cuts": res_cor.cuts_added,
                "cor26_solve_s": res_cor.solve_time_s,
                "erdogan_sim_h1": round(erd_h1, 2) if erd_h1 is not None else None,
                "erdogan_lp_obj": res_erd.lp_timing_obj,
                "erdogan_audit_ok": bool(res_erd.sim_audit_passed and res_erd.sim_100pct_completed),
                "erdogan_solve_s": res_erd.solve_time_s,
                "core_sched_h1": core_h1,
                "core_sched_comp": core_comp,
                "core_sched_p95_ms": round(float(core_r.get("decision_ms_p95", 0.0)), 2),
                "v18_ea_h1": v18_h1,
                "v18_ea_comp": v18_comp,
                "pgmpo_h1": pg_h1,
                "pgmpo_comp": pg_comp,
                "lb_sound": sound,
            }
        )
        if idx % 6 == 0 or idx == len(specs):
            print(
                f"[Gate G1] {idx:02d}/{len(specs)} {ek:42s} | LB={lb:6.1f} | "
                f"COR26={cor_ub if cor_ub is not None else float('inf'):6.1f} ({res_cor.solve_time_s:.1f}s) | "
                f"Erd={erd_h1 if erd_h1 is not None else float('inf'):6.1f} ({res_erd.solve_time_s:.1f}s) | "
                f"CoRe={core_h1:6.1f}({'OK' if core_comp else 'NC'}) | V18={v18_h1:6.1f}",
                flush=True,
            )

    by_group: dict[str, Any] = {}
    for grp in ("4topo_standard", "4topo_window_bottleneck", "track1_static"):
        sub = [r for r in rows if r["group"] == grp]
        if not sub:
            continue
        cor_ok = [r for r in sub if r["cor26_audit_ok"] and r["cor26_sim_ub"] is not None]
        erd_ok = [r for r in sub if r["erdogan_audit_ok"] and r["erdogan_sim_h1"] is not None]
        core_ok = [r for r in sub if r["core_sched_comp"]]
        v18_ok = [r for r in sub if r["v18_ea_comp"]]
        pg_ok = [r for r in sub if r["pgmpo_comp"]]
        by_group[grp] = {
            "n_instances": len(sub),
            "n_relaxation_violations": sum(1 for r in sub if not r["lb_sound"]),
            "mean_lb_phys": round(float(np.mean([r["lb_phys"] for r in sub])), 2),
            "cor26_completed": len(cor_ok),
            "cor26_proven_optimal": sum(1 for r in cor_ok if r["cor26_status"] == "OPTIMAL"),
            "mean_cor26_sim_ub": round(float(np.mean([r["cor26_sim_ub"] for r in cor_ok])), 2) if cor_ok else None,
            "mean_cor26_solve_s": round(float(np.mean([r["cor26_solve_s"] for r in sub])), 2),
            "erdogan_completed": len(erd_ok),
            "mean_erdogan_sim_h1": round(float(np.mean([r["erdogan_sim_h1"] for r in erd_ok])), 2) if erd_ok else None,
            "mean_erdogan_solve_s": round(float(np.mean([r["erdogan_solve_s"] for r in sub])), 2),
            "core_sched_completed": len(core_ok),
            "mean_core_sched_h1_all": round(float(np.mean([r["core_sched_h1"] for r in sub])), 2),
            "mean_core_sched_h1_comp": round(float(np.mean([r["core_sched_h1"] for r in core_ok])), 2) if core_ok else None,
            "mean_core_sched_ep_p95_ms": round(float(np.mean([r["core_sched_p95_ms"] for r in sub])), 2),
            "v18_ea_completed": len(v18_ok),
            "mean_v18_ea_h1_all": round(float(np.mean([r["v18_ea_h1"] for r in sub])), 2),
            "mean_v18_ea_h1_comp": round(float(np.mean([r["v18_ea_h1"] for r in v18_ok])), 2) if v18_ok else None,
            "pgmpo_completed": len(pg_ok),
            "mean_pgmpo_h1_all": round(float(np.mean([r["pgmpo_h1"] for r in sub])), 2),
            "mean_pgmpo_h1_comp": round(float(np.mean([r["pgmpo_h1"] for r in pg_ok])), 2) if pg_ok else None,
        }

    pooled_core_p95 = round(float(np.percentile(core_all_lats_ms, 95)), 2) if core_all_lats_ms else 0.0
    pooled_core_p99 = round(float(np.percentile(core_all_lats_ms, 99)), 2) if core_all_lats_ms else 0.0

    payload = {
        "experiment": "EXP-10-G1_Static_Common_Subproblem_Reference",
        "references": {
            "COR26-Ref": "R. W. M. van Os & T. Basten (COR 187, 2025/2026, DOI: 10.1016/j.cor.2025.107342)",
            "EJOR26-Erdogan": "B. Erdogan, Q.-V. Dang, M. Mohammadi, & I. J. B. F. Adan (EJOR 2026, DOI: 10.1016/j.ejor.2026.06.038)",
        },
        "methodological_note": (
            "Pure zero-fault static common subproblem (`strip_dynamic_disruptions`). All 5 solvers "
            "are executed inside the exact same zero-fault Simulator + IndependentAuditor plant. "
            "COR26-Ref freezes LB_phys strictly at Phase 1 (pure un-cut physical relaxation Master); "
            "subsequent nogood cuts are strictly post-LB search-diversification heuristics."
        ),
        "time_limit_s": time_limit_s,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "n_instances": len(rows),
        "n_relaxation_violations": n_violations,
        "core_sched_pooled_p95_ms": pooled_core_p95,
        "core_sched_pooled_p99_ms": pooled_core_p99,
        "by_group": by_group,
        "rows": rows,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    jp = OUT_DIR / "exp10_g1_static_reference.json"
    jp.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = RESULTS_DIR / "g1_static_reference_subproblem.csv"
    fieldnames = [
        "episode_key",
        "group",
        "lb_phys",
        "cor26_status",
        "cor26_sim_ub",
        "cor26_audit_ok",
        "cor26_solve_s",
        "erdogan_sim_h1",
        "erdogan_audit_ok",
        "erdogan_solve_s",
        "core_sched_h1",
        "core_sched_comp",
        "core_sched_p95_ms",
        "v18_ea_h1",
        "v18_ea_comp",
        "pgmpo_h1",
        "pgmpo_comp",
        "lb_sound",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    md = [
        "# Gate G1 (`EXP-10-G1`): Pure Zero-Fault Static Common-Subproblem Reference Benchmark",
        "",
        f"- **Instances**: `{len(rows)}` pure static zero-fault scenarios (`27101..27302`, `2901..2906`), executed end-to-end inside the exact same `Simulator` + `IndependentAuditor` plant.",
        f"- **Relaxation Soundness Violations (`LB_phys > H1_sim`)**: **`{n_violations}`** (strictly `0` across all 5 solvers and all 54 instances).",
        f"- **`CoRe-Sched` Static Pooled Latency**: P95 = `{pooled_core_p95:.2f} ms`, P99 = `{pooled_core_p99:.2f} ms`.",
        "",
        "| Static Group | $N$ | Valid `LB_phys` | `COR26-Ref` Sim-`UB` (Comp, Opt, Time) | `EJOR26-Erdogan` Sim-$H_1$ (Comp, Time) | `CoRe-Sched` Online $H_1$ (Comp, Ep-P95) | `V18-EA` $H_1$ (Comp) | `PGMPO` $H_1$ (Comp) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for grp, s in by_group.items():
        cor_str = f"**{s['mean_cor26_sim_ub']:.1f}** ({s['cor26_completed']}/{s['n_instances']}, opt={s['cor26_proven_optimal']}, {s['mean_cor26_solve_s']:.1f}s)" if s["mean_cor26_sim_ub"] is not None else "N/A"
        erd_str = f"**{s['mean_erdogan_sim_h1']:.1f}** ({s['erdogan_completed']}/{s['n_instances']}, {s['mean_erdogan_solve_s']:.1f}s)" if s["mean_erdogan_sim_h1"] is not None else "N/A"
        core_str = f"**{s['mean_core_sched_h1_comp']:.1f}** ({s['core_sched_completed']}/{s['n_instances']}, {s['mean_core_sched_ep_p95_ms']:.1f}ms)" if s["mean_core_sched_h1_comp"] is not None else "N/A"
        v18_str = f"{s['mean_v18_ea_h1_comp']:.1f} ({s['v18_ea_completed']}/{s['n_instances']})" if s["mean_v18_ea_h1_comp"] is not None else "N/A"
        pg_str = f"{s['mean_pgmpo_h1_comp']:.1f} ({s['pgmpo_completed']}/{s['n_instances']})" if s["mean_pgmpo_h1_comp"] is not None else "N/A"
        md.append(
            f"| `{grp}` | {s['n_instances']} | **{s['mean_lb_phys']:.1f}** | {cor_str} | {erd_str} | {core_str} | {v18_str} | {pg_str} |"
        )
    mp = OUT_DIR / "exp10_g1_static_reference.md"
    mp.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[Gate G1] COMPLETE! Violations={n_violations}. Saved {jp}, {mp}, {csv_path} ({payload['elapsed_s']}s)", flush=True)
    return payload


if __name__ == "__main__":
    run_g1_static_reference()
