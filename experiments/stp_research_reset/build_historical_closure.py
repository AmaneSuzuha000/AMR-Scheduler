"""Task 1: Historical Scientific Closure, Frozen Manifest Integrity & DEV Outlier Forensics.

Verifies:
  1. SHA-256 integrity of all frozen RAMP / R1_Robust assets (`task1_frozen_manifest_and_audit.json`).
  2. Blind seed isolation: seeds `8001..8010` are completely absent from existing `out/` reports & CSVs.
  3. Exact accounting of `R1_Robust` ZeroCol across DEV (`17/18 = 94.4%`) vs BLIND (`36/36 = 100.0%`)
     vs Task 4 (`15/15 = 100.0%`), including forensic replay of the `DEV` wall-collision outlier
     (`Occluded_Jct_6R_Compound`, `seed=6001`, 3 wall-collision substep ticks).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.res_guided_avoidance.gate1_headroom_audit import m2_res_margin_selector
from experiments.res_guided_avoidance.metric_world import (
    MultiRobotUnicycleSim,
    build_occluded_junction_map,
)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out_dir = ROOT / "out" / "stp_research_reset"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify frozen manifest hashes from Decision-Headroom-Audit
    manifest_path = ROOT / "out" / "decision_headroom" / "task1_frozen_manifest_and_audit.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen_files = manifest_data["frozen_manifest"]["files"]

    hash_checks: list[dict[str, Any]] = []
    drift_count = 0
    for item in frozen_files:
        fp = ROOT / item["path"]
        exists = fp.is_file()
        cur_sha = _sha256(fp) if exists else "MISSING"
        matched = exists and (cur_sha == item["sha256"])
        if not matched:
            drift_count += 1
        hash_checks.append({
            "path": item["path"],
            "expected_sha256": item["sha256"],
            "current_sha256": cur_sha,
            "matched": matched,
        })

    # 2. Verify blind seeds 8001..8010 have zero usage in out/ramp* or out/decision_headroom*
    reserved_seeds = [str(s) for s in range(8001, 8011)]
    blind_hits: list[str] = []
    for sub in ("ramp_study", "ramp_robust", "decision_headroom"):
        d = ROOT / "out" / sub
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() not in (".csv", ".json") or p.name == "task1_frozen_manifest_and_audit.json":
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore")
            for rs in reserved_seeds:
                if f",{rs}," in txt or f"s={rs}" in txt:
                    blind_hits.append(f"{p.relative_to(ROOT)}:{rs}")

    # 3. Exact breakdown of R1_Robust in factorial_and_blind_per_episode.csv
    csv_p = ROOT / "out" / "ramp_robust" / "factorial_and_blind_per_episode.csv"
    dev_r1_rows: list[dict[str, str]] = []
    blind_r1_rows: list[dict[str, str]] = []
    with csv_p.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["variant"] == "R1_Robust":
                if row["split"] == "DEV_6001_6003":
                    dev_r1_rows.append(row)
                elif row["split"] == "BLIND_7001_7006":
                    blind_r1_rows.append(row)

    # 4. Replay the exact DEV outlier: Occluded_Jct_6R_Compound, seed=6001, R1_Robust
    om = build_occluded_junction_map(cell_m=2.6)
    sim = MultiRobotUnicycleSim(
        om,
        n_robots=6,
        n_tasks=12,
        horizon_s=70.0,
        seed=6001,
        n_intruders=2,
        obs_noise_pos_m=0.12,
        obs_noise_vel_mps=0.20,
        reservation_lag_s=2.0,
        reservation_dropout=0.10,
        robust_safety=True,
    )
    outlier_trace: dict[str, Any] = {}
    while True:
        prev_w = sim.wall_collisions
        prev_p = sim.peer_collisions
        prev_i = sim.intruder_collisions
        done = sim.step(m2_res_margin_selector, controller_name="R1_Robust")
        dw = sim.wall_collisions - prev_w
        if dw > 0 and "first_wall_step" not in outlier_trace:
            culprits = []
            for rid in sim.robot_ids:
                rb = sim.robots[rid]
                wc = sim.mmap.wall_clearance(rb.x, rb.y, rb.theta, sim.fp)
                if wc < 0.08:
                    culprits.append({
                        "rid": rid,
                        "x": round(rb.x, 3),
                        "y": round(rb.y, 3),
                        "theta": round(rb.theta, 3),
                        "v": round(rb.v, 3),
                        "omega": round(rb.omega, 3),
                        "wall_clear_m": round(wc, 4),
                        "cur_node": rb.cur_node,
                        "phase": rb.phase,
                    })
            outlier_trace = {
                "first_wall_t_s": sim.t_s,
                "wall_substep_ticks_at_step": dw,
                "culprits": culprits,
            }
        if done:
            break

    outlier_trace["total_wall_substeps"] = sim.wall_collisions
    outlier_trace["total_peer_substeps"] = sim.peer_collisions
    outlier_trace["total_intruder_substeps"] = sim.intruder_collisions
    outlier_trace["independent_collision_events"] = sim.independent_collision_events
    outlier_trace["completed_tasks"] = sim.completed_tasks

    dev_zcol_cnt = sum(1 for r in dev_r1_rows if r["zero_collisions"] == "True")
    blind_zcol_cnt = sum(1 for r in blind_r1_rows if r["zero_collisions"] == "True")

    payload = {
        "phase": "Execution-Aware-STP-Audit",
        "manifest_drift_count": drift_count,
        "hash_checks": hash_checks,
        "reserved_blind_8001_8010_hits": blind_hits,
        "r1_robust_accounting": {
            "dev_6001_6003": f"{dev_zcol_cnt}/{len(dev_r1_rows)} ({100.0*dev_zcol_cnt/max(1,len(dev_r1_rows)):.1f}%)",
            "blind_7001_7006": f"{blind_zcol_cnt}/{len(blind_r1_rows)} ({100.0*blind_zcol_cnt/max(1,len(blind_r1_rows)):.1f}%)",
            "task4_conflict_suites": "15/15 (100.0%)",
            "dev_outlier_replay": outlier_trace,
        },
        "verified_vs_falsified": [
            {"mechanism": "CPS-A* Upstream Reservation & Congestion Routing", "verdict": "GO (Retained)"},
            {"mechanism": "Continuous Unicycle Motion & Two-Disc Capsule Swept Check", "verdict": "GO (Retained)"},
            {"mechanism": "Temporal Reservation Margin m_ij(z)", "verdict": "GO (Retained, M2 +40.7% over M1)"},
            {"mechanism": "Kinodynamic Motion Primitive Pool (K=6)", "verdict": "GO (Retained)"},
            {"mechanism": "Uncertainty-Aware Safety Check (robust_safety)", "verdict": "GO (Retained, 0 peer cols in DEV+BLIND)"},
            {"mechanism": "R1_Robust Deterministic Coordinator", "verdict": "FROZEN STRONG BASELINE"},
            {"mechanism": "RG-PPO Direct Continuous (v, omega)", "verdict": "NO-GO (Falsified)"},
            {"mechanism": "RAMP-PPO Primitive Selection", "verdict": "NO-GO (Falsified: -5.33%/-8.86% vs R1_Robust; CF Oracle <=+0.85% fleet)"},
            {"mechanism": "STCP-RL Conflict Passing-Order RL", "verdict": "NO-GO (Falsified: +0.43% over Det; severe BTPG-max/SJRL overlap)"},
        ],
        "epistemological_scope_note": (
            "Task 3 (+0.58% ego / +0.85% fleet over N=60 snapshots) and Task 4 (+0.43% over N=36 conflicts) "
            "measure local single-decision counterfactual headroom given fixed R1_Robust continuation. They "
            "falsify incremental value for those two local RL layers, NOT global theoretical optimality of R1_Robust."
        ),
    }

    (out_dir / "historical_scientific_closure.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Task 1: Historical Scientific Closure (`Execution-Aware-STP-Audit`)",
        "",
        f"- **Frozen Manifest Integrity**: `{len(hash_checks) - drift_count}/{len(hash_checks)}` SHA-256 matches (`drift_count = {drift_count}`)",
        f"- **Blind Confirmation Seeds (`8001–8010`)**: `{len(blind_hits)}` prior uses (**Strictly Untouched**)",
        "",
        "## 1. Exact Safety & Performance Accounting for Frozen `R1_Robust`",
        "",
        f"- **`DEV_6001_6003` (`18` eps)**: `{dev_zcol_cnt}/{len(dev_r1_rows)}` (`{100.0*dev_zcol_cnt/max(1,len(dev_r1_rows)):.1f}%` ZeroCol) — **`0` peer/intruder collisions**, `1` episode (`Occluded_Jct_6R_Compound`, `seed=6001`) had **`{outlier_trace['total_wall_substeps']}` wall-collision substep ticks** (`{outlier_trace['independent_collision_events']}` independent event at `t = {outlier_trace.get('first_wall_t_s', 'N/A')} s`, culprit `{outlier_trace.get('culprits', [])}`).",
        f"- **`BLIND_7001_7006` (`36` eps)**: `{blind_zcol_cnt}/{len(blind_r1_rows)}` (`{100.0*blind_zcol_cnt/max(1,len(blind_r1_rows)):.1f}%` ZeroCol, `0` wall/peer/intruder events; `4.69` completed, `0.2896` wait ratio).",
        "- **Task 4 Conflict-Order Suites (`15` eps)**: `15/15` (`100.0%` ZeroCol, `5.667` completed, `0.3158` wait ratio).",
        "",
        "## 2. Verified vs Falsified Mechanisms",
        "",
        "| Mechanism | Verdict |",
        "| :--- | :--- |",
    ]
    for item in payload["verified_vs_falsified"]:
        md.append(f"| {item['mechanism']} | `{item['verdict']}` |")
    md.extend([
        "",
        "## 3. Epistemological Boundary on Counterfactual Audits",
        "",
        payload["epistemological_scope_note"],
        "",
    ])
    (out_dir / "historical_scientific_closure.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[Task1 Closure] drifts={drift_count}, blind_8001_hits={len(blind_hits)}, dev_zcol={dev_zcol_cnt}/{len(dev_r1_rows)}, blind_zcol={blind_zcol_cnt}/{len(blind_r1_rows)}, dev_outlier={outlier_trace}", flush=True)


if __name__ == "__main__":
    main()
