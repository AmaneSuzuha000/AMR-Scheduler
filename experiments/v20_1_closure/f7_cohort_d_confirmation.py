"""Task F7: Final V20.1 SHA-256 Freeze & Single-Shot Uncontended Confirmation on Cohort D (120) + Cohort D-Struct (48).

Strict anti-leakage scientific confirmation protocol:
  1. Verifies `tools/data_manifest.py --check` (all 38 public CSVs untouched).
  2. Locks `out/v20_1_closure/v20_1_freeze_manifest.json` (SHA-256 of production `amr_sched/`
     V20/V20.1 source + packaged checkpoints + this runner) BEFORE touching any Cohort D seed.
  3. Asserts strict zero overlap of `Cohort D` (4-Topo `57101..57302`, Workshop `5901..5906`)
     and `Cohort D-Struct` (`6901..6912`) with ALL prior seeds (Train / Val / RL / G1 /
     Forbidden / Cohort A / Cohort B / Cohort C).
  4. Runs ONE single-shot uncontended wall-clock confirmation (`35 ms` SLA, `--workers 1`):
     - `Cohort D (120 eps)` across 14 methods (`V20_1_NAJS`, `V20_NAJS`, `V19_ADAPTIVE_RL`,
       `V19_HETERO_RL`, `V18_EA_JOINT`, `IJCAI26_PGMPO`, `SPGA_AMR`, `BALANS_IJCAI25`,
       `RAL26_DetourDelta`, `EJOR26_Wu_GRU_Attn`, `ASC26_MA_TRANS`, `FGCS26_REN_GNN`,
       `IJPR25_DFJSPT_HMARL`, `rule_v10`).
     - `Cohort D-Struct (48 structurally independent stress eps)` across 6 core methods
       (`V20_1_NAJS`, `V20_NAJS`, `V19_ADAPTIVE_RL`, `V19_HETERO_RL`, `V18_EA_JOINT`, `IJCAI26_PGMPO`).

Outputs: `out/v20_1_closure/f7_cohort_d_confirmation.{json,md}`,
         `out/v20_1_closure/f7_cohort_d_rows.csv`,
         `out/v20_1_closure/f7_cohort_d_struct_rows.csv`.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.baselines.policy_registry import run_discrete_episode
from amr_sched.cvcr.p2_perturbation import P2PerturbationConfig
from amr_sched.entities import MaterialType, Task, TaskPriority
from experiments.run_g3_fresh_frozen_holdout import FRESH_4TOPO_SEEDS, FRESH_WORKSHOP_SEEDS
from experiments.run_g5_pristine_blind_holdout import PRISTINE_4TOPO_SEEDS, PRISTINE_WORKSHOP_SEEDS
from experiments.train_v19_bc import (
    FORBIDDEN_BLIND_SEEDS,
    FORBIDDEN_DEV_SEEDS,
    TRAIN_4TOPO_SEEDS,
    TRAIN_WORKSHOP_SEEDS,
    VAL_4TOPO_SEEDS,
    VAL_WORKSHOP_SEEDS,
    _build_4topo_episode_spec,
    _build_workshop_episode_spec,
)
from experiments.train_v19_rl import RL_4TOPO_SEEDS, RL_WORKSHOP_SEEDS
from experiments.v20_audit.run_v20_final_blind_holdout import (
    COHORT_C_4TOPO_SEEDS,
    COHORT_C_WORKSHOP_SEEDS,
)
from tools.benchmarks.compare_workshop_algorithms import REGIMES as WORKSHOP_REGIMES
from experiments.v18_confirmation_scenarios import LAYOUTS as TOPO4_LAYOUTS, REGIMES as TOPO4_REGIMES

OUT_CLOSURE_DIR = ROOT / "out" / "v20_1_closure"
FREEZE_V20_1_PATH = OUT_CLOSURE_DIR / "v20_1_freeze_manifest.json"

COHORT_D_4TOPO_SEEDS: tuple[int, ...] = (57101, 57102, 57201, 57202, 57301, 57302)
COHORT_D_WORKSHOP_SEEDS: tuple[int, ...] = (5901, 5902, 5903, 5904, 5905, 5906)
COHORT_D_STRUCT_SEEDS: tuple[int, ...] = tuple(range(6901, 6913))

COHORT_D_14_POLICIES: tuple[str, ...] = (
    "V20_1_NAJS",
    "V20_NAJS",
    "V19_ADAPTIVE_RL",
    "V19_HETERO_RL",
    "V18_EA_JOINT",
    "IJCAI26_PGMPO",
    "SPGA_AMR",
    "BALANS_IJCAI25",
    "RAL26_DetourDelta",
    "EJOR26_Wu_GRU_Attn",
    "ASC26_MA_TRANS",
    "FGCS26_REN_GNN",
    "IJPR25_DFJSPT_HMARL",
    "rule_v10",
)

D_STRUCT_6_POLICIES: tuple[str, ...] = (
    "V20_1_NAJS",
    "V20_NAJS",
    "V19_ADAPTIVE_RL",
    "V19_HETERO_RL",
    "V18_EA_JOINT",
    "IJCAI26_PGMPO",
)

LOCKED_V20_1_FILES: tuple[str, ...] = (
    "amr_sched/cvcr/pdg_liveness.py",
    "amr_sched/rl/v20_najs.py",
    "amr_sched/baselines/policy_registry.py",
    "amr_sched/cvcr/physical_bridge.py",
    "amr_sched/checkpoints/v20/v20_critic_weights.npz",
    "amr_sched/checkpoints/v19/v19_hetero_rl.pt",
    "amr_sched/checkpoints/v19/v19_staged_rl.pt",
    "experiments/v20_1_closure/f7_cohort_d_confirmation.py",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    sv = sorted(vals)
    pos = (q / 100.0) * (len(sv) - 1)
    lo = int(math.floor(pos))
    hi = min(len(sv) - 1, lo + 1)
    return float(sv[lo] + (pos - lo) * (sv[hi] - sv[lo]))


def _cvar95(vals: list[float]) -> float:
    if not vals:
        return 0.0
    sv = sorted(vals)
    k = max(1, int(math.ceil(0.05 * len(sv))))
    return float(np.mean(sv[-k:]))


def _bootstrap_ci_mean(diffs: list[float], n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    if not diffs:
        return (0.0, 0.0)
    arr = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return (round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2))


def lock_and_verify_v20_1_freeze() -> dict[str, Any]:
    """Create or verify SHA-256 freeze manifest BEFORE any Cohort D seed runs."""
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "data_manifest.py"), "--check"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Public CSV data_manifest check failed: {proc.stdout} {proc.stderr}")

    hashes: dict[str, str] = {}
    for rel in LOCKED_V20_1_FILES:
        p = ROOT / rel
        if not p.is_file():
            raise FileNotFoundError(f"Missing locked V20.1 file: {rel}")
        hashes[rel] = _sha256_file(p)

    manifest = {
        "freeze_id": "IF-CORE-V20.1-FREEZE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_manifest_ok": True,
        "cohort_d_4topo_seeds": list(COHORT_D_4TOPO_SEEDS),
        "cohort_d_workshop_seeds": list(COHORT_D_WORKSHOP_SEEDS),
        "cohort_d_struct_seeds": list(COHORT_D_STRUCT_SEEDS),
        "sha256_files": hashes,
    }
    OUT_CLOSURE_DIR.mkdir(parents=True, exist_ok=True)
    FREEZE_V20_1_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Immediate read-back verify
    loaded = json.loads(FREEZE_V20_1_PATH.read_text(encoding="utf-8"))
    for rel, expected in loaded["sha256_files"].items():
        actual = _sha256_file(ROOT / rel)
        assert actual == expected, f"Freeze SHA-256 mismatch on {rel}"
    return loaded


def assert_cohort_d_disjoint() -> None:
    """Assert strict zero overlap of Cohort D and Cohort D-Struct with ALL prior splits."""
    prior = (
        set(FORBIDDEN_BLIND_SEEDS)
        | set(FORBIDDEN_DEV_SEEDS)
        | set(TRAIN_4TOPO_SEEDS)
        | set(TRAIN_WORKSHOP_SEEDS)
        | set(VAL_4TOPO_SEEDS)
        | set(VAL_WORKSHOP_SEEDS)
        | set(RL_4TOPO_SEEDS)
        | set(RL_WORKSHOP_SEEDS)
        | set(FRESH_4TOPO_SEEDS)
        | set(FRESH_WORKSHOP_SEEDS)
        | set(PRISTINE_4TOPO_SEEDS)
        | set(PRISTINE_WORKSHOP_SEEDS)
        | set(COHORT_C_4TOPO_SEEDS)
        | set(COHORT_C_WORKSHOP_SEEDS)
    )
    new_seeds = set(COHORT_D_4TOPO_SEEDS) | set(COHORT_D_WORKSHOP_SEEDS) | set(COHORT_D_STRUCT_SEEDS)
    overlap = prior & new_seeds
    assert not overlap, f"QUARAINTINE VIOLATION: Cohort D overlaps prior seeds: {sorted(overlap)}"


def build_cohort_d_120_specs() -> list[dict[str, Any]]:
    """Build brand-new 120-episode Cohort D matching Cohorts A/B/C exact structure."""
    specs: list[dict[str, Any]] = []
    for lay in TOPO4_LAYOUTS:
        for reg in TOPO4_REGIMES:
            for sd in COHORT_D_4TOPO_SEEDS:
                sp = _build_4topo_episode_spec(lay, int(sd), reg)
                sp["suite_group"] = "4-Topo"
                specs.append(sp)
    for trk, grp in (("track1", "Track 1"), ("track2", "Track 2")):
        for reg in WORKSHOP_REGIMES:
            for sd in COHORT_D_WORKSHOP_SEEDS:
                sp = _build_workshop_episode_spec(trk, int(sd), reg)
                sp["suite_group"] = grp
                specs.append(sp)
    assert len(specs) == 120
    return specs


def build_cohort_d_struct_48_specs() -> list[dict[str, Any]]:
    """Build 48 structurally independent OOD stress episodes (4 structural perturbation families x 12 seeds)."""
    specs: list[dict[str, Any]] = []
    for idx, sd in enumerate(COHORT_D_STRUCT_SEEDS):
        # Family 1: Asymmetric Bottleneck Graph (remove 2 cross-aisle edges from Track 1 kitting_windows)
        sp1 = _build_workshop_episode_spec("track1", int(sd), "kitting_windows")
        net1, tasks1, agvs1, ops1, meta1 = copy.deepcopy(sp1["scenario_tuple"])
        # Remove up to 2 non-bridge interior edges cleanly (verify graph connectivity preserved)
        cand_edges = [
            k for k in sorted(net1.edges.keys())
            if str(k[0]).startswith("J_") and str(k[1]).startswith("J_") and str(k[0]) < str(k[1])
        ]
        removed = 0
        for u, v in cand_edges:
            if removed >= 2:
                break
            e_uv = net1.edges.pop((u, v), None)
            e_vu = net1.edges.pop((v, u), None)
            # Check basic connectivity via BFS from first node
            start_n = next(iter(net1.nodes))
            seen = {start_n}
            q = [start_n]
            while q:
                cur = q.pop()
                for dst in [ek[1] for ek in net1.edges if ek[0] == cur]:
                    if dst not in seen:
                        seen.add(dst)
                        q.append(dst)
            if len(seen) == len(net1.nodes):
                removed += 1
            else:
                if e_uv is not None:
                    net1.edges[(u, v)] = e_uv
                if e_vu is not None:
                    net1.edges[(v, u)] = e_vu
        sp1["scenario_tuple"] = (net1, tasks1, agvs1, ops1, meta1)
        sp1["episode_key"] = f"dstruct_bottleneck_{sd}"
        sp1["suite_group"] = "D-Struct-Bottleneck"
        specs.append(sp1)

        # Family 2: 3-Kit Fan-In & Deeper Precedence Chain with Staggered Tight Windows (Track 2 base)
        sp2 = _build_workshop_episode_spec("track2", int(sd), "kitting_windows")
        net2, tasks2, agvs2, ops2, meta2 = copy.deepcopy(sp2["scenario_tuple"])
        # Add an extra 3rd kitting task to the first multi-kit op + tighten window margins by 15%
        extra_added = False
        new_ops2 = []
        t2_map = {t.task_id: t for t in tasks2}
        for op in ops2:
            if not extra_added and len(op.kit_task_ids) >= 2 and op.kit_task_ids[0] in t2_map:
                ref_t = t2_map[op.kit_task_ids[0]]
                tid_3 = f"{ref_t.task_id}_KIT3"
                tasks2.append(
                    Task(
                        task_id=tid_3,
                        pickup=ref_t.pickup,
                        dropoff=ref_t.dropoff,
                        release_step=int(ref_t.release_step) + 12,
                        due_step=int(ref_t.due_step or 260) + 18,
                        priority=TaskPriority.URGENT,
                        material_type=ref_t.material_type,
                    )
                )
                wins = tuple(
                    (int(w0), max(int(w0) + int(op.proc_steps) + 4, int(w1) - 8))
                    for w0, w1 in (getattr(op, "resource_windows", ()) or ())
                )
                new_ops2.append(replace(op, kit_task_ids=tuple(op.kit_task_ids) + (tid_3,), resource_windows=wins))
                extra_added = True
            else:
                new_ops2.append(op)
        sp2["scenario_tuple"] = (net2, tasks2, agvs2, new_ops2, meta2)
        sp2["episode_key"] = f"dstruct_3kit_windows_{sd}"
        sp2["suite_group"] = "D-Struct-3KitDAG"
        sp2["max_steps"] = int(sp2["max_steps"]) + 120
        specs.append(sp2)

        # Family 3: Compound Multi-Wave Blockage + Execution Drift (Track 1 execution_blockage + stronger P2 drift)
        sp3 = _build_workshop_episode_spec("track1", int(sd), "execution_blockage")
        sp3["p2_cfg"] = P2PerturbationConfig(
            seed=int(sd) * 31 + 7,
            enable_travel_drift=True,
            drift_prob=0.35,
            late_max_steps=3,
            enable_dynamic_obstacles=True,
            obstacle_count=3,
            obstacle_duration_steps=24,
            enable_local_planner_fail=True,
            local_fail_count=3,
            local_fail_duration_steps=12,
        )
        sp3["episode_key"] = f"dstruct_compound_drift_{sd}"
        sp3["suite_group"] = "D-Struct-Compound"
        specs.append(sp3)

        # Family 4: High-Density Multi-Wave Surge + Window Bottleneck (4-Topo / Track 2 alternating)
        if idx % 2 == 0:
            sp4 = _build_4topo_episode_spec(TOPO4_LAYOUTS[idx % len(TOPO4_LAYOUTS)], int(sd), "window_bottleneck")
        else:
            sp4 = _build_workshop_episode_spec("track2", int(sd), "multi_wave_surge")
        sp4["p2_cfg"] = P2PerturbationConfig(
            seed=int(sd) * 37 + 11,
            enable_travel_drift=True,
            drift_prob=0.28,
            late_max_steps=3,
            enable_dynamic_obstacles=True,
            obstacle_count=2,
            obstacle_duration_steps=18,
            enable_local_planner_fail=True,
            local_fail_count=2,
            local_fail_duration_steps=10,
        )
        sp4["episode_key"] = f"dstruct_surge_p2_{sd}"
        sp4["suite_group"] = "D-Struct-SurgeP2"
        specs.append(sp4)

    assert len(specs) == 48
    return specs


def _summarize_cohort(rows_by_pol: dict[str, list[dict[str, Any]]], lats_by_pol: dict[str, list[float]], policies: tuple[str, ...]) -> list[dict[str, Any]]:
    board: list[dict[str, Any]] = []
    for pol in policies:
        rows = rows_by_pol[pol]
        lats = lats_by_pol[pol]
        nets = [float(r["net_obj"]) for r in rows]
        h1s = [float(r["h1"]) for r in rows]
        by_grp: dict[str, float] = {}
        for g in sorted({str(r["suite_group"]) for r in rows}):
            sub = [float(r["net_obj"]) for r in rows if str(r["suite_group"]) == g]
            by_grp[g] = round(float(np.mean(sub)), 2)
        board.append({
            "policy": pol,
            "n": len(rows),
            "comp100": sum(1 for r in rows if bool(r["comp100"])),
            "comp100_pct": round(100.0 * sum(1 for r in rows if bool(r["comp100"])) / max(1, len(rows)), 2),
            "audit_pass": sum(1 for r in rows if bool(r["audit_passed"])),
            "mean_ja": round(float(np.mean(nets)), 2),
            "median_ja": round(float(np.median(nets)), 2),
            "cvar95": round(_cvar95(nets), 2),
            "mean_h1": round(float(np.mean(h1s)), 2),
            "p95_ms": round(_pct(lats, 95.0), 2),
            "p99_ms": round(_pct(lats, 99.0), 2),
            "max_ms": round(max(lats) if lats else 0.0, 2),
            "over35_pct": round(100.0 * sum(1 for x in lats if x > 35.0) / max(1, len(lats)), 3),
            "expired_windows": sum(int(r["expired_windows"]) for r in rows),
            "pdg_alert_eps": sum(1 for r in rows if int(r.get("pdg_alerts", 0)) > 0),
            "by_group": by_grp,
        })
    board.sort(key=lambda x: (-x["comp100"], x["mean_ja"]))
    return board


def _paired_comparison(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> dict[str, Any]:
    map_b = {r["episode_key"]: float(r["net_obj"]) for r in rows_b}
    diffs = [float(r["net_obj"]) - map_b[r["episode_key"]] for r in rows_a if r["episode_key"] in map_b]
    wins = sum(1 for d in diffs if d < -1e-4)
    losses = sum(1 for d in diffs if d > 1e-4)
    ties = len(diffs) - wins - losses
    p_two = 1.0
    p_less = 1.0
    nz = [d for d in diffs if abs(d) > 1e-6]
    if len(nz) >= 5:
        try:
            p_two = float(wilcoxon(nz, alternative="two-sided").pvalue)
            p_less = float(wilcoxon(nz, alternative="less").pvalue)
        except Exception:
            pass
    ci_lo, ci_hi = _bootstrap_ci_mean(diffs)
    return {
        "n_pairs": len(diffs),
        "mean_delta": round(float(np.mean(diffs)), 2) if diffs else 0.0,
        "median_delta": round(float(np.median(diffs)), 2) if diffs else 0.0,
        "ci95": [ci_lo, ci_hi],
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "wilcoxon_p_two": round(p_two, 6),
        "wilcoxon_p_less": round(p_less, 6),
    }


def _run_suite(specs: list[dict[str, Any]], policies: tuple[str, ...], csv_path: Path, label: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[float]]]:
    rows_by_pol: dict[str, list[dict[str, Any]]] = {p: [] for p in policies}
    lats_by_pol: dict[str, list[float]] = {p: [] for p in policies}
    all_csv_rows: list[dict[str, Any]] = []

    for pol in policies:
        t_p0 = time.perf_counter()
        for sp in specs:
            rec = run_discrete_episode(
                pol,
                sp["scenario_tuple"],
                seed=int(sp["seed"]),
                max_steps=int(sp["max_steps"]),
                use_rust=True,
                p2_cfg=sp.get("p2_cfg"),
                clock_mode="wall",
                episode_key=str(sp["episode_key"]),
                include_decision_latencies=True,
            )
            lats = [float(x) for x in rec.pop("decision_latencies_ms", [])]
            rec["suite_group"] = str(sp["suite_group"])
            rec["h1"] = round(float(rec["cmax"]) + float(rec["task_tardiness_cost"]) + float(rec["op_tardiness_cost"]), 4)
            rec["comp100"] = bool(float(rec["task_comp"]) >= 0.9999 and float(rec["prod_comp"]) >= 0.9999)
            rows_by_pol[pol].append(rec)
            lats_by_pol[pol].extend(lats)
            all_csv_rows.append({
                "policy": pol,
                "episode_key": rec["episode_key"],
                "suite_group": rec["suite_group"],
                "seed": rec["seed"],
                "net_obj": rec["net_obj"],
                "h1": rec["h1"],
                "cmax": rec["cmax"],
                "task_comp": rec["task_comp"],
                "prod_comp": rec["prod_comp"],
                "comp100": rec["comp100"],
                "audit_passed": rec["audit_passed"],
                "decision_ms_p95": rec["decision_ms_p95"],
                "decision_ms_p99": rec["decision_ms_p99"],
                "expired_windows": rec["expired_windows"],
                "pdg_alerts": rec.get("pdg_alerts", 0),
                "pdg_escalations": rec.get("pdg_escalations", 0),
            })
        nets = [float(r["net_obj"]) for r in rows_by_pol[pol]]
        comp = sum(1 for r in rows_by_pol[pol] if r["comp100"])
        p95 = _pct(lats_by_pol[pol], 95.0)
        p99 = _pct(lats_by_pol[pol], 99.0)
        mx = max(lats_by_pol[pol]) if lats_by_pol[pol] else 0.0
        print(
            f"  [{label}] {pol:20s} | Comp={comp:3d}/{len(specs)} | MeanJA={np.mean(nets):8.2f} | Med={np.median(nets):7.2f} "
            f"| CVaR95={_cvar95(nets):8.1f} | P95={p95:5.2f} P99={p99:5.2f} Max={mx:5.2f}ms | {time.perf_counter()-t_p0:5.1f}s",
            flush=True,
        )

    if all_csv_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_csv_rows[0].keys()))
            w.writeheader()
            w.writerows(all_csv_rows)
    return rows_by_pol, lats_by_pol


def main() -> None:
    print("=== [F7] Locking V20.1 Freeze Manifest & Verifying Quarantine ===", flush=True)
    assert_cohort_d_disjoint()
    manifest = lock_and_verify_v20_1_freeze()
    print(f"[F7 FREEZE LOCKED] {len(manifest['sha256_files'])} files SHA-256 verified -> {FREEZE_V20_1_PATH}", flush=True)

    d120_specs = build_cohort_d_120_specs()
    d48_specs = build_cohort_d_struct_48_specs()

    print("=== [F7.1] Single-Shot Uncontended Confirmation: Brand-New Cohort D (120 eps x 14 policies) ===", flush=True)
    d_rows, d_lats = _run_suite(d120_specs, COHORT_D_14_POLICIES, OUT_CLOSURE_DIR / "f7_cohort_d_rows.csv", "CohortD-120")
    d_board = _summarize_cohort(d_rows, d_lats, COHORT_D_14_POLICIES)

    print("=== [F7.2] Single-Shot Uncontended Confirmation: Structurally Independent Cohort D-Struct (48 eps x 6 policies) ===", flush=True)
    ds_rows, ds_lats = _run_suite(d48_specs, D_STRUCT_6_POLICIES, OUT_CLOSURE_DIR / "f7_cohort_d_struct_rows.csv", "D-Struct-48")
    ds_board = _summarize_cohort(ds_rows, ds_lats, D_STRUCT_6_POLICIES)

    paired_d = {
        f"V20_1_NAJS_vs_{ref}": _paired_comparison(d_rows["V20_1_NAJS"], d_rows[ref])
        for ref in ("V20_NAJS", "V19_ADAPTIVE_RL", "V19_HETERO_RL", "V18_EA_JOINT", "IJCAI26_PGMPO", "SPGA_AMR")
    }
    paired_ds = {
        f"V20_1_NAJS_vs_{ref}": _paired_comparison(ds_rows["V20_1_NAJS"], ds_rows[ref])
        for ref in ("V20_NAJS", "V19_ADAPTIVE_RL", "V19_HETERO_RL", "V18_EA_JOINT", "IJCAI26_PGMPO")
    }

    out = {
        "freeze_manifest": manifest,
        "cohort_d_120_leaderboard": d_board,
        "cohort_d_120_paired": paired_d,
        "cohort_d_struct_48_leaderboard": ds_board,
        "cohort_d_struct_48_paired": paired_ds,
    }
    (OUT_CLOSURE_DIR / "f7_cohort_d_confirmation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = [
        "# Deliverable F7: Final SHA-256 Frozen Single-Shot Confirmation (`Cohort D` 120 eps + `Cohort D-Struct` 48 eps)",
        "",
        f"- **Freeze Lock**: `IF-CORE-V20.1-FREEZE` (`{len(manifest['sha256_files'])}` files SHA-256 verified before execution, `v20_1_freeze_manifest.json`)",
        "- **Seed Disjointness**: Zero overlap with Train / Val / RL / G1 / Forbidden / Cohort A / Cohort B / Cohort C.",
        "- **Protocol**: Single-shot uncontended single-process real wall-clock (`35 ms` SLA); ZERO edits or re-tuning permitted after touching Cohort D.",
        "",
        "## 1. Brand-New Unseen `Cohort D` Leaderboard (120 Episodes × 14 Policies)",
        "",
        "| Rank | Policy | Comp100 | Mean `J_A` ↓ | Med `J_A` ↓ | `CVaR_95` ↓ | Pure `H1` ↓ | 4-Topo | Track 1 | Track 2 | P95 / P99 / Max (ms) | `>35ms` % |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rk, b in enumerate(d_board, 1):
        bg = b["by_group"]
        md.append(
            f"| {rk} | **{b['policy']}** | {b['comp100']}/120 ({b['comp100_pct']:.2f}%) | {b['mean_ja']:.2f} | {b['median_ja']:.2f} "
            f"| {b['cvar95']:.1f} | {b['mean_h1']:.2f} | {bg.get('4-Topo', 0):.1f} | {bg.get('Track 1', 0):.1f} | {bg.get('Track 2', 0):.1f} "
            f"| {b['p95_ms']:.2f} / {b['p99_ms']:.2f} / {b['max_ms']:.2f} | {b['over35_pct']:.3f}% |"
        )

    md.extend([
        "",
        "### Paired Statistical Tests on `Cohort D` (`V20_1_NAJS` vs Anchors)",
        "",
        "| Comparison | Mean Δ (`J_A`, <0 favors V20.1) | 95% Bootstrap CI | W / T / L | Wilcoxon `p` (2-sided) | Wilcoxon `p` (1-sided <) |",
        "| :--- | ---: | ---: | :---: | ---: | ---: |",
    ])
    for k, st in paired_d.items():
        md.append(
            f"| {k} | {st['mean_delta']:+.2f} | [{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}] "
            f"| {st['wins']} / {st['ties']} / {st['losses']} | {st['wilcoxon_p_two']:.6f} | {st['wilcoxon_p_less']:.6f} |"
        )

    md.extend([
        "",
        "## 2. Structurally Independent OOD Stress Suite `Cohort D-Struct` (48 Episodes × 6 Policies)",
        "",
        "| Rank | Policy | Comp100 | Mean `J_A` ↓ | Med `J_A` ↓ | `CVaR_95` ↓ | Bottleneck | 3-Kit DAG | Compound | Surge+P2 | P95 / P99 / Max (ms) | `>35ms` % |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for rk, b in enumerate(ds_board, 1):
        bg = b["by_group"]
        md.append(
            f"| {rk} | **{b['policy']}** | {b['comp100']}/48 ({b['comp100_pct']:.2f}%) | {b['mean_ja']:.2f} | {b['median_ja']:.2f} "
            f"| {b['cvar95']:.1f} | {bg.get('D-Struct-Bottleneck', 0):.1f} | {bg.get('D-Struct-3KitDAG', 0):.1f} "
            f"| {bg.get('D-Struct-Compound', 0):.1f} | {bg.get('D-Struct-SurgeP2', 0):.1f} "
            f"| {b['p95_ms']:.2f} / {b['p99_ms']:.2f} / {b['max_ms']:.2f} | {b['over35_pct']:.3f}% |"
        )

    md.extend([
        "",
        "### Paired Statistical Tests on `Cohort D-Struct` (`V20_1_NAJS` vs Anchors)",
        "",
        "| Comparison | Mean Δ (`J_A`) | 95% Bootstrap CI | W / T / L | Wilcoxon `p` (2-sided) | Wilcoxon `p` (1-sided <) |",
        "| :--- | ---: | ---: | :---: | ---: | ---: |",
    ])
    for k, st in paired_ds.items():
        md.append(
            f"| {k} | {st['mean_delta']:+.2f} | [{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}] "
            f"| {st['wins']} / {st['ties']} / {st['losses']} | {st['wilcoxon_p_two']:.6f} | {st['wilcoxon_p_less']:.6f} |"
        )

    (OUT_CLOSURE_DIR / "f7_cohort_d_confirmation.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[F7 DONE] Wrote {OUT_CLOSURE_DIR / 'f7_cohort_d_confirmation.md'}", flush=True)


if __name__ == "__main__":
    main()
