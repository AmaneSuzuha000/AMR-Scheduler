"""Offline Exact Reference Bound Comparison via COR 2026 LBBD/CP (`van Os & Basten, COR 2026`).

Evaluates `solve_lbbd_cp_cor26` (OR-Tools CP-SAT Master + Rust conflict-free spacetime Subproblem
+ Logic-Based Benders Cuts, `15 s` offline budget per instance) across all 54 static/known
semantically compatible instances from the 120-episode blind holdout:
  - 24 episodes: `4-Topo standard`
  - 24 episodes: `4-Topo window_bottleneck`
  -  6 episodes: `Track 1 nominal` (62-node Rail Freight Workshop, 6 AMRs)

Compares provable Lower Bound (`LB`) and verified conflict-free Upper Bound (`UB`) against
online millisecond `CoRe-Sched` (`~12.5 ms`) and `V18-EA` on pure physical efficiency
`H1 = C_max + 2 * T_task + 2 * T_op`.

Writes outputs to `out/v19_scientific_audit/exp10_lbbd_exact_reference.{json,md}`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_MOD_DIR = ROOT / "_scratch"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.baselines.lbbd_cp_cor26 import solve_lbbd_cp_cor26
from experiments.train_v19_bc import _build_4topo_episode_spec, _build_workshop_episode_spec
from experiments.v18_confirmation_scenarios import LAYOUTS as TOPO4_LAYOUTS

OUT_DIR = ROOT / "out" / "v19_scientific_audit"
BLIND_4TOPO_SEEDS = (17101, 17102, 17201, 17202, 17301, 17302)
BLIND_WORKSHOP_SEEDS = (901, 902, 903, 904, 905, 906)


def _row_h1(row: dict[str, Any]) -> float:
    return float(row["cmax"] + row["task_tardiness_cost"] + row["op_tardiness_cost"])


def run_lbbd_exact_reference(time_limit_s: float = 15.0) -> dict[str, Any]:
    t0 = time.perf_counter()
    e8 = json.loads((OUT_DIR / "exp08_learning_b2_synergy.json").read_text(encoding="utf-8"))
    bh = json.loads((ROOT / "out" / "g4_blind_holdout" / "blind_holdout_report.json").read_text(encoding="utf-8"))
    core_by_k = {r["episode_key"]: r for r in e8["part_c_120ep_registered_ahg_rl"]["rows"]}
    v18_by_k = {r["episode_key"]: r for r in bh["rows"]["V18_EA_JOINT"]}
    pgmpo_by_k = {r["episode_key"]: r for r in bh["rows"]["IJCAI26_PGMPO"]}

    specs: list[dict[str, Any]] = []
    for s in BLIND_4TOPO_SEEDS:
        for lay in TOPO4_LAYOUTS:
            for reg in ("standard", "window_bottleneck"):
                sp = _build_4topo_episode_spec(lay, s, reg)
                sp["group"] = f"4topo_{reg}"
                specs.append(sp)
    for s in BLIND_WORKSHOP_SEEDS:
        sp = _build_workshop_episode_spec("track1", s, "nominal")
        sp["group"] = "track1_nominal"
        specs.append(sp)

    rows: list[dict[str, Any]] = []
    for idx, sp in enumerate(specs, 1):
        ek = sp["episode_key"]
        res = solve_lbbd_cp_cor26(
            copy.deepcopy(sp["scenario_tuple"]),
            time_limit_s=time_limit_s,
            max_iters=30,
            num_workers=8,
            max_steps=int(sp["max_steps"]),
        )
        assert ek in core_by_k, f"Missing CoRe-Sched blind holdout episode key: {ek}"
        core_h1 = _row_h1(core_by_k[ek])
        core_ms = float(core_by_k[ek].get("decision_ms_p95", 12.5))
        v18_h1 = _row_h1(v18_by_k[ek]) if ek in v18_by_k else None
        pgmpo_h1 = _row_h1(pgmpo_by_k[ek]) if ek in pgmpo_by_k else None
        rows.append(
            {
                "episode_key": ek,
                "group": sp["group"],
                "status": res.status,
                "lbbd_lb": res.lower_bound,
                "lbbd_ub": res.upper_bound if res.conflict_free_verified else None,
                "lbbd_cmax": res.best_cmax,
                "lbbd_gap_pct": res.gap_pct,
                "lbbd_iters": res.iterations,
                "lbbd_cuts": res.cuts_added,
                "lbbd_solve_s": res.solve_time_s,
                "verified": res.conflict_free_verified,
                "core_sched_h1": round(core_h1, 2) if core_h1 is not None else None,
                "core_sched_p95_ms": round(core_ms, 2) if core_ms is not None else None,
                "v18_ea_h1": round(v18_h1, 2) if v18_h1 is not None else None,
                "pgmpo_h1": round(pgmpo_h1, 2) if pgmpo_h1 is not None else None,
            }
        )
        if idx % 10 == 0 or idx == len(specs):
            print(
                f"[COR26 LBBD/CP] {idx:02d}/{len(specs)} {ek:42s} | {res.status:8s} "
                f"LB={res.lower_bound:6.1f} UB={res.upper_bound:6.1f} ({res.solve_time_s:.2f}s) | CoRe H1={core_h1:.1f}",
                flush=True,
            )

    by_group: dict[str, Any] = {}
    for grp in ("4topo_standard", "4topo_window_bottleneck", "track1_nominal"):
        sub = [r for r in rows if r["group"] == grp and r["verified"]]
        if not sub:
            continue
        lbs = [r["lbbd_lb"] for r in sub]
        ubs = [r["lbbd_ub"] for r in sub]
        cores = [r["core_sched_h1"] for r in sub if r["core_sched_h1"] is not None]
        v18s = [r["v18_ea_h1"] for r in sub if r["v18_ea_h1"] is not None]
        pgs = [r["pgmpo_h1"] for r in sub if r["pgmpo_h1"] is not None]
        times_s = [r["lbbd_solve_s"] for r in sub]
        core_ms_l = [r["core_sched_p95_ms"] for r in sub if r["core_sched_p95_ms"] is not None]
        n_opt = sum(1 for r in sub if r["status"] == "OPTIMAL")
        mean_ub = float(np.mean(ubs))
        mean_lb = float(np.mean(lbs))
        mean_core = float(np.mean(cores))
        by_group[grp] = {
            "n_instances": len(sub),
            "n_proven_optimal": n_opt,
            "mean_lbbd_lb": round(mean_lb, 2),
            "mean_lbbd_ub": round(mean_ub, 2),
            "mean_lbbd_gap_pct": round(float(np.mean([r["lbbd_gap_pct"] for r in sub])), 2),
            "mean_lbbd_solve_s": round(float(np.mean(times_s)), 3),
            "mean_lbbd_cuts": round(float(np.mean([r["lbbd_cuts"] for r in sub])), 2),
            "mean_core_sched_h1": round(mean_core, 2),
            "mean_core_sched_p95_ms": round(float(np.mean(core_ms_l)), 2),
            "speedup_core_vs_lbbd": round((float(np.mean(times_s)) * 1000.0) / max(0.1, float(np.mean(core_ms_l))), 1),
            "core_ratio_vs_ub": round(mean_core / max(1.0, mean_ub), 3),
            "mean_v18_ea_h1": round(float(np.mean(v18s)), 2),
            "mean_pgmpo_h1": round(float(np.mean(pgs)), 2),
        }

    payload = {
        "experiment": "EXP-10_COR26_LBBD_CP_Exact_Reference_Bound",
        "reference": "van Os & Basten (COR 187, 2025/2026, DOI: 10.1016/j.cor.2025.107342)",
        "reproduction_tier": "paper_faithful_reproduction (offline full-information static reference)",
        "time_limit_s": time_limit_s,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "n_instances": len(rows),
        "by_group": by_group,
        "rows": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jp = OUT_DIR / "exp10_lbbd_exact_reference.json"
    jp.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# EXP-10: COR 2026 LBBD/CP Offline Exact Reference Bound vs. Online CoRe-Sched",
        "",
        "| Instance Group | $N$ | Proven Opt | LBBD Lower Bound (`LB`) | LBBD Conflict-Free `UB` | LBBD Time (s) | `CoRe-Sched` Online $H_1$ | `CoRe-Sched` P95 (ms) | Speedup | `V18-EA` $H_1$ | `PGMPO` $H_1$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for grp, s in by_group.items():
        md.append(
            f"| `{grp}` | {s['n_instances']} | {s['n_proven_optimal']}/{s['n_instances']} | "
            f"**{s['mean_lbbd_lb']:.1f}** | **{s['mean_lbbd_ub']:.1f}** | {s['mean_lbbd_solve_s']:.2f} s | "
            f"**{s['mean_core_sched_h1']:.1f}** | **{s['mean_core_sched_p95_ms']:.2f} ms** | "
            f"**{s['speedup_core_vs_lbbd']:.0f}×** | {s['mean_v18_ea_h1']:.1f} | {s['mean_pgmpo_h1']:.1f} |"
        )
    mp = OUT_DIR / "exp10_lbbd_exact_reference.md"
    mp.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[COR26 LBBD/CP] Saved {jp} & {mp} ({payload['elapsed_s']}s)", flush=True)
    return payload


if __name__ == "__main__":
    run_lbbd_exact_reference()
