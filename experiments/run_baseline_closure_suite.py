"""EXP-09 Baseline Closure Suite: COR 2026 SPGA & IJPR 2025 DFJSP-T H-MARL on 120 Blind Holdouts.

Evaluates the two direct production-AMR baselines in exact `clock_mode="wall"` parity with
`out/g4_blind_holdout/blind_holdout_report.json` and `out/v19_scientific_audit/exp08_learning_b2_synergy.json`:
  1. `SPGA_AMR` (Jia Li et al., Computers & Operations Research 193, 2026, 107524) [Paper-faithful]
  2. `IJPR25_DFJSPT_HMARL` (Wang et al., IJPR 2025, ClouDaDaDa/DFJSPT_code) [Paper-faithful]

Computes:
  - Full 120-episode wall-clock net objective, 100%-completion rate, 7-term decomposition,
    split means (`4-Topo`, `Track 1`, `Track 2`), and wall-clock P95 decision latency.
  - Exact paired Wilcoxon p-values, 95% Bootstrap CIs, and W/T/L counts for `CoRe-Sched` vs both.
  - Complete pure-efficiency `H1 = C_max + 2*T_task + 2*T_op` attribution table (both overall
    and on strictly mutually 100%-completed episodes where `H2 == 0` for both methods, across
    all 120 episodes and on the 48 complex industrial workshop episodes `Track 1 + Track 2`).

Writes outputs to `out/v19_scientific_audit/exp09_baseline_closure.{json,md}`.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.baselines.policy_registry import run_discrete_episode
from experiments.train_v19_bc import _build_4topo_episode_spec, _build_workshop_episode_spec
from tools.benchmarks.compare_workshop_algorithms import REGIMES as WORKSHOP_REGIMES
from experiments.v18_confirmation_scenarios import LAYOUTS as TOPO4_LAYOUTS, REGIMES as TOPO4_REGIMES

OUT_DIR = ROOT / "out" / "v19_scientific_audit"
BLIND_4TOPO_SEEDS = (17101, 17102, 17201, 17202, 17301, 17302)
BLIND_WORKSHOP_SEEDS = (901, 902, 903, 904, 905, 906)


def _build_120_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for s in BLIND_4TOPO_SEEDS:
        for lay in TOPO4_LAYOUTS:
            for reg in TOPO4_REGIMES:
                sp = _build_4topo_episode_spec(lay, s, reg)
                sp["split"] = "blind_4topo"
                specs.append(sp)
    for s in BLIND_WORKSHOP_SEEDS:
        for tr in ("track1", "track2"):
            for reg in WORKSHOP_REGIMES:
                sp = _build_workshop_episode_spec(tr, s, reg)
                sp["split"] = f"blind_{tr}"
                specs.append(sp)
    return specs


def _paired_bootstrap_ci(deltas: np.ndarray, n_boot: int = 10_000, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


def _summarize_rows(policy_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    objs = np.array([float(r["net_obj"]) for r in rows], dtype=float)
    cmax = np.array([float(r["cmax"]) for r in rows], dtype=float)
    tt = np.array([float(r["task_tardiness_cost"]) for r in rows], dtype=float)
    ot = np.array([float(r["op_tardiness_cost"]) for r in rows], dtype=float)
    h1 = cmax + tt + ot
    unf = np.array([float(r["unfinished_task_penalty"]) + float(r["unfinished_op_penalty"]) for r in rows], dtype=float)
    churn = np.array([float(r["churn_cost"]) for r in rows], dtype=float)
    comp = np.array([float(r["compute_cost"]) for r in rows], dtype=float)
    p95 = np.array([float(r.get("decision_ms_p95", r.get("mean_decision_ms", 0.0))) for r in rows], dtype=float)
    n100 = sum(1 for r in rows if float(r["task_comp"]) >= 0.9999 and float(r["prod_comp"]) >= 0.9999)
    cvar10 = float(np.mean(sorted(objs, reverse=True)[:12]))
    by_sp = {}
    for sk in ("blind_4topo", "blind_track1", "blind_track2"):
        sub = [float(r["net_obj"]) for r in rows if r["split"] == sk]
        by_sp[sk] = round(float(np.mean(sub)), 2) if sub else 0.0
    return {
        "policy": policy_name,
        "num_episodes": len(rows),
        "num_100pct_completed": int(n100),
        "audit_pass_rate": round(float(np.mean([1.0 if r["audit_passed"] else 0.0 for r in rows])), 4),
        "mean_net_obj": round(float(np.mean(objs)), 2),
        "median_net_obj": round(float(np.median(objs)), 2),
        "cvar10_net_obj": round(cvar10, 2),
        "mean_h1": round(float(np.mean(h1)), 2),
        "mean_cmax": round(float(np.mean(cmax)), 2),
        "mean_tardiness_cost": round(float(np.mean(tt + ot)), 2),
        "mean_unfinished_penalty": round(float(np.mean(unf)), 2),
        "mean_churn_cost": round(float(np.mean(churn)), 2),
        "mean_compute_cost": round(float(np.mean(comp)), 2),
        "mean_p95_ms": round(float(np.mean(p95)), 2),
        "split_mean_net_obj": by_sp,
    }


def _paired_comparison(core_rows: list[dict[str, Any]], bl_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cm = {r["episode_key"]: r for r in core_rows}
    bm = {r["episode_key"]: r for r in bl_rows}
    keys = [k for k in cm if k in bm]
    assert len(keys) == len(cm) == 120, f"Episode key mismatch: paired={len(keys)} vs expected 120"
    c_ja = np.array([float(cm[k]["net_obj"]) for k in keys], dtype=float)
    b_ja = np.array([float(bm[k]["net_obj"]) for k in keys], dtype=float)
    d_ja = c_ja - b_ja
    nz = d_ja[np.abs(d_ja) > 1e-9]
    w_p = float(wilcoxon(nz, alternative="two-sided")[1]) if len(nz) >= 1 else 1.0

    # Pure H1 comparison on mutually 100%-completed episodes (where H2 == 0 for both!)
    both_ok = [
        k for k in keys
        if (float(cm[k]["task_comp"]) >= 0.9999 and float(cm[k]["prod_comp"]) >= 0.9999)
        and (float(bm[k]["task_comp"]) >= 0.9999 and float(bm[k]["prod_comp"]) >= 0.9999)
    ]
    ws_both_ok = [k for k in both_ok if cm[k]["split"] in ("blind_track1", "blind_track2")]

    def _h1(r: dict[str, Any]) -> float:
        return float(r["cmax"]) + float(r["task_tardiness_cost"]) + float(r["op_tardiness_cost"])

    c_h1_both = float(np.mean([_h1(cm[k]) for k in both_ok])) if both_ok else 0.0
    b_h1_both = float(np.mean([_h1(bm[k]) for k in both_ok])) if both_ok else 0.0
    c_h1_ws = float(np.mean([_h1(cm[k]) for k in ws_both_ok])) if ws_both_ok else 0.0
    b_h1_ws = float(np.mean([_h1(bm[k]) for k in ws_both_ok])) if ws_both_ok else 0.0

    return {
        "n_paired": len(keys),
        "paired_delta_mean": round(float(np.mean(d_ja)), 2),
        "pct_change_vs_baseline": round(100.0 * float(np.mean(d_ja)) / max(1e-9, float(np.mean(b_ja))), 2),
        "bootstrap_95ci": _paired_bootstrap_ci(d_ja),
        "wilcoxon_p": w_p,
        "wins": int(np.sum(d_ja < -1e-6)),
        "ties": int(np.sum(np.abs(d_ja) <= 1e-6)),
        "losses": int(np.sum(d_ja > 1e-6)),
        "n_both_100pct": len(both_ok),
        "core_h1_on_both_100pct": round(c_h1_both, 2),
        "baseline_h1_on_both_100pct": round(b_h1_both, 2),
        "delta_h1_on_both_100pct": round(c_h1_both - b_h1_both, 2),
        "n_ws_both_100pct": len(ws_both_ok),
        "core_h1_on_ws_both_100pct": round(c_h1_ws, 2),
        "baseline_h1_on_ws_both_100pct": round(b_h1_ws, 2),
        "delta_h1_on_ws_both_100pct": round(c_h1_ws - b_h1_ws, 2),
    }


def run_closure_suite(policies: tuple[str, ...] = ("SPGA_AMR", "IJPR25_DFJSPT_HMARL"), *, reuse_existing: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "exp09_baseline_closure.json"
    existing = json.loads(out_json.read_text(encoding="utf-8")) if out_json.is_file() else {}
    saved_rows: dict[str, list[dict[str, Any]]] = dict(existing.get("rows", {}))

    specs = _build_120_specs()
    for pol in policies:
        tp = time.perf_counter()
        prev_by_key = {r["episode_key"]: r for r in saved_rows.get(pol, [])} if reuse_existing else {}
        rows: list[dict[str, Any]] = []
        n_reused = 0
        for idx, sp in enumerate(specs, 1):
            ek = sp["episode_key"]
            if ek in prev_by_key:
                rows.append(prev_by_key[ek])
                n_reused += 1
            else:
                rec = run_discrete_episode(
                    pol,
                    copy.deepcopy(sp["scenario_tuple"]),
                    seed=int(sp["seed"]),
                    max_steps=int(sp["max_steps"]),
                    use_rust=True,
                    p2_cfg=copy.deepcopy(sp["p2_cfg"]),
                    clock_mode="wall",
                    episode_key=ek,
                    backend="numpy",
                )
                assert rec["audit_passed"], f"{pol} audit failed on {ek}"
                rows.append(
                    {
                        "episode_key": str(rec["episode_key"]),
                        "split": str(sp["split"]),
                        "net_obj": round(float(rec["net_obj"]), 4),
                        "cmax": round(float(rec["cmax"]), 4),
                        "task_tardiness_cost": round(float(rec["task_tardiness_cost"]), 4),
                        "op_tardiness_cost": round(float(rec["op_tardiness_cost"]), 4),
                        "unfinished_task_penalty": round(float(rec["unfinished_task_penalty"]), 4),
                        "unfinished_op_penalty": round(float(rec["unfinished_op_penalty"]), 4),
                        "churn_cost": round(float(rec["churn_cost"]), 4),
                        "compute_cost": round(float(rec["compute_cost"]), 4),
                        "task_comp": round(float(rec["task_comp"]), 6),
                        "prod_comp": round(float(rec["prod_comp"]), 6),
                        "mean_decision_ms": round(float(rec["mean_decision_ms"]), 3),
                        "decision_ms_p95": round(float(rec["decision_ms_p95"]), 3),
                        "audit_passed": bool(rec["audit_passed"]),
                    }
                )
            if idx % 30 == 0 or idx == len(specs):
                print(
                    f"[EXP-09 {pol}] {idx:03d}/120 (reused={n_reused}) | cur_mean={np.mean([r['net_obj'] for r in rows]):7.1f} ({time.perf_counter()-tp:.1f}s)",
                    flush=True,
                )
        saved_rows[pol] = rows

    # Combine with CoRe-Sched & blind_holdout_report.json for complete unified pure-H1 & paired attribution
    e8 = json.loads((OUT_DIR / "exp08_learning_b2_synergy.json").read_text(encoding="utf-8"))
    bh = json.loads((ROOT / "out" / "g4_blind_holdout" / "blind_holdout_report.json").read_text(encoding="utf-8"))
    core_rows = e8["part_c_120ep_registered_ahg_rl"]["rows"]

    all_policies_rows: dict[str, list[dict[str, Any]]] = {"CoRe-Sched": core_rows}
    for p_k, r_l in saved_rows.items():
        all_policies_rows[p_k] = r_l
    for p_k, r_l in bh["rows"].items():
        all_policies_rows[p_k] = r_l

    summaries = {p: _summarize_rows(p, rs) for p, rs in all_policies_rows.items()}
    paired = {p: _paired_comparison(core_rows, rs) for p, rs in all_policies_rows.items() if p != "CoRe-Sched"}

    payload = {
        "experiment": "EXP-09_Baseline_Closure_SPGA_DFJSPT_and_PureH1_Audit",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "new_policies_evaluated": list(saved_rows.keys()),
        "summaries": summaries,
        "paired_vs_core_sched": paired,
        "rows": saved_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# EXP-09: Baseline Closure (`COR26-SPGA` & `IJPR25-DFJSP-T`) & Pure-Efficiency $H_1$ Audit",
        "",
        "| Method | 100% (`/120`) | Mean $J_A$ | Pure $H_1$ (All) | $H_1$ Both-100% (CoRe vs BL) | WS $H_1$ Both-100% (CoRe vs BL) | Wall P95 (ms) | Wilcoxon $p$ | 95% CI |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for p, s in summaries.items():
        if p == "CoRe-Sched":
            md.append(f"| **{p}** | **{s['num_100pct_completed']}/120** | **{s['mean_net_obj']:.2f}** | **{s['mean_h1']:.1f}** | — | — | **{s['mean_p95_ms']:.2f}** | — | — |")
        else:
            ps = paired[p]
            md.append(
                f"| `{p}` | {s['num_100pct_completed']}/120 | {s['mean_net_obj']:.2f} | {s['mean_h1']:.1f} | "
                f"{ps['core_h1_on_both_100pct']:.1f} vs {ps['baseline_h1_on_both_100pct']:.1f} (`{ps['delta_h1_on_both_100pct']:+.1f}`) | "
                f"{ps['core_h1_on_ws_both_100pct']:.1f} vs {ps['baseline_h1_on_ws_both_100pct']:.1f} (`{ps['delta_h1_on_ws_both_100pct']:+.1f}`) | "
                f"{s['mean_p95_ms']:.2f} | {ps['wilcoxon_p']:.4g} | {ps['bootstrap_95ci']} |"
            )
    (OUT_DIR / "exp09_baseline_closure.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[EXP-09] Saved {out_json} ({payload['elapsed_s']}s)", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXP-09 baseline closure suite.")
    parser.add_argument("--policies", nargs="+", default=["SPGA_AMR", "IJPR25_DFJSPT_HMARL"])
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    run_closure_suite(tuple(args.policies), reuse_existing=bool(args.reuse_existing))


if __name__ == "__main__":
    main()
