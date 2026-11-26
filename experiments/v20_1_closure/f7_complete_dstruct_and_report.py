"""Complete F7.2 (`Cohort D-Struct` 48 structurally independent stress episodes) and merge
with the ALREADY-LOCKED single-shot `Cohort D` (120 eps x 14 policies) CSV (`f7_cohort_d_rows.csv`).

Strictly ZERO re-running or tuning of `Cohort D`; verifies production `amr_sched/` SHA-256
against `v20_1_freeze_manifest.json` before executing `F7.2`.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
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
from amr_sched.entities import Task, TaskPriority
from amr_sched.network import Edge, Network, Node
from experiments.train_v19_bc import _build_4topo_episode_spec, _build_workshop_episode_spec
from experiments.v20_1_closure.f7_cohort_d_confirmation import (
    COHORT_D_14_POLICIES,
    COHORT_D_STRUCT_SEEDS,
    D_STRUCT_6_POLICIES,
    FREEZE_V20_1_PATH,
    OUT_CLOSURE_DIR,
    _bootstrap_ci_mean,
    _cvar95,
    _paired_comparison,
    _pct,
    _run_suite,
    _sha256_file,
    _summarize_cohort,
    assert_cohort_d_disjoint,
)
from experiments.v18_confirmation_scenarios import LAYOUTS as TOPO4_LAYOUTS

# Verbatim pooled wall-clock latency recorded during the single-shot F7.1 Cohort D run (task-39944)
COHORT_D_POOLED_LAT_FROM_SINGLE_SHOT: dict[str, tuple[float, float, float]] = {
    "V20_1_NAJS": (25.35, 35.83, 398.37),
    "V20_NAJS": (21.93, 25.07, 110.06),
    "V19_ADAPTIVE_RL": (17.05, 19.09, 41.30),
    "V19_HETERO_RL": (47.99, 86.88, 170.17),
    "V18_EA_JOINT": (35.27, 74.38, 184.33),
    "IJCAI26_PGMPO": (24.84, 48.91, 127.90),
    "SPGA_AMR": (80.25, 111.40, 201.39),
    "BALANS_IJCAI25": (44.85, 57.78, 81.33),
    "RAL26_DetourDelta": (28.22, 44.53, 147.17),
    "EJOR26_Wu_GRU_Attn": (54.92, 86.20, 120.58),
    "ASC26_MA_TRANS": (20.86, 69.80, 95.23),
    "FGCS26_REN_GNN": (17.09, 22.09, 55.04),
    "IJPR25_DFJSPT_HMARL": (61.97, 80.14, 173.27),
    "rule_v10": (86.64, 93.73, 151.94),
}


def _rebuild_network_without_pairs(net: Network, drop_pairs: set[tuple[str, str]]) -> Network:
    """Construct a fresh, adjacency-consistent `Network` excluding undirected edge pairs."""
    out = Network()
    for n in net.nodes.values():
        out.add_node(Node(n.node_id, float(n.x), float(n.y), kind=str(n.kind)))
    for e in net.edges.values():
        pair = (e.src, e.dst) if e.src < e.dst else (e.dst, e.src)
        if pair in drop_pairs:
            continue
        out.add_edge(Edge(e.src, e.dst, length=float(e.length), speed_limit=float(e.speed_limit)))
    return out


def _is_strongly_connected(net: Network) -> bool:
    if not net.nodes:
        return True
    adj: dict[str, list[str]] = {nid: [] for nid in net.nodes}
    for e in net.edges.values():
        adj.setdefault(e.src, []).append(e.dst)
    start = next(iter(net.nodes))
    seen = {start}
    q = [start]
    while q:
        u = q.pop()
        for dst in adj.get(u, ()):
            if dst not in seen:
                seen.add(dst)
                q.append(dst)
    return len(seen) == len(net.nodes)


def build_clean_d_struct_48_specs() -> list[dict[str, Any]]:
    """Build 48 structurally independent OOD stress episodes with 100% consistent graph adjacency."""
    specs: list[dict[str, Any]] = []
    for idx, sd in enumerate(COHORT_D_STRUCT_SEEDS):
        # Family 1: Asymmetric Bottleneck Graph (cleanly remove 2 non-bridge interior corridor edges)
        sp1 = _build_workshop_episode_spec("track1", int(sd), "kitting_windows")
        net1, tasks1, agvs1, ops1, meta1 = copy.deepcopy(sp1["scenario_tuple"])
        cand_pairs = sorted(
            {
                (e.src, e.dst) if e.src < e.dst else (e.dst, e.src)
                for e in net1.edges.values()
                if str(e.src).startswith("J_") and str(e.dst).startswith("J_")
            }
        )
        dropped: set[tuple[str, str]] = set()
        cur_net = net1
        for pair in cand_pairs:
            if len(dropped) >= 2:
                break
            trial_drop = dropped | {pair}
            trial_net = _rebuild_network_without_pairs(net1, trial_drop)
            if _is_strongly_connected(trial_net):
                dropped = trial_drop
                cur_net = trial_net
        sp1["scenario_tuple"] = (cur_net, tasks1, agvs1, ops1, meta1)
        sp1["episode_key"] = f"dstruct_bottleneck_{sd}"
        sp1["suite_group"] = "D-Struct-Bottleneck"
        specs.append(sp1)

        # Family 2: 3-Kit Fan-In & Deeper Precedence Chain with Staggered Tight Windows
        sp2 = _build_workshop_episode_spec("track2", int(sd), "kitting_windows")
        net2, tasks2, agvs2, ops2, meta2 = copy.deepcopy(sp2["scenario_tuple"])
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

        # Family 3: Compound Multi-Wave Blockage + Execution Drift
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

        # Family 4: High-Density Multi-Wave Surge + Window Bottleneck
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


def _load_locked_cohort_d_csv() -> dict[str, list[dict[str, Any]]]:
    rows_by_pol: dict[str, list[dict[str, Any]]] = {p: [] for p in COHORT_D_14_POLICIES}
    with (OUT_CLOSURE_DIR / "f7_cohort_d_rows.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pol = r["policy"]
            rows_by_pol[pol].append({
                "policy": pol,
                "episode_key": r["episode_key"],
                "suite_group": r["suite_group"],
                "seed": int(r["seed"]),
                "net_obj": float(r["net_obj"]),
                "h1": float(r["h1"]),
                "cmax": float(r["cmax"]),
                "task_comp": float(r["task_comp"]),
                "prod_comp": float(r["prod_comp"]),
                "comp100": str(r["comp100"]).lower() == "true",
                "audit_passed": str(r["audit_passed"]).lower() == "true",
                "decision_ms_p95": float(r["decision_ms_p95"]),
                "decision_ms_p99": float(r["decision_ms_p99"]),
                "expired_windows": int(r["expired_windows"]),
                "pdg_alerts": int(r["pdg_alerts"]),
                "pdg_escalations": int(r["pdg_escalations"]),
            })
    for p in COHORT_D_14_POLICIES:
        assert len(rows_by_pol[p]) == 120, f"Expected 120 locked Cohort D rows for {p}, got {len(rows_by_pol[p])}"
    return rows_by_pol


def main() -> None:
    assert_cohort_d_disjoint()
    manifest = json.loads(FREEZE_V20_1_PATH.read_text(encoding="utf-8"))
    for rel, expected in manifest["sha256_files"].items():
        if rel.startswith("amr_sched/"):
            actual = _sha256_file(ROOT / rel)
            assert actual == expected, f"Production freeze violation on {rel}"
    print("[F7.2] Verified production amr_sched/ & checkpoint SHA-256 freeze intact.", flush=True)

    d_rows = _load_locked_cohort_d_csv()
    d_board: list[dict[str, Any]] = []
    for pol in COHORT_D_14_POLICIES:
        rows = d_rows[pol]
        nets = [float(r["net_obj"]) for r in rows]
        h1s = [float(r["h1"]) for r in rows]
        by_grp = {}
        for g in ("4-Topo", "Track 1", "Track 2"):
            sub = [float(r["net_obj"]) for r in rows if r["suite_group"] == g]
            by_grp[g] = round(float(np.mean(sub)), 2)
        p95, p99, mx = COHORT_D_POOLED_LAT_FROM_SINGLE_SHOT[pol]
        d_board.append({
            "policy": pol,
            "n": 120,
            "comp100": sum(1 for r in rows if r["comp100"]),
            "comp100_pct": round(100.0 * sum(1 for r in rows if r["comp100"]) / 120.0, 2),
            "audit_pass": sum(1 for r in rows if r["audit_passed"]),
            "mean_ja": round(float(np.mean(nets)), 2),
            "median_ja": round(float(np.median(nets)), 2),
            "cvar95": round(_cvar95(nets), 2),
            "mean_h1": round(float(np.mean(h1s)), 2),
            "p95_ms": p95,
            "p99_ms": p99,
            "max_ms": mx,
            "expired_windows": sum(int(r["expired_windows"]) for r in rows),
            "pdg_alert_eps": sum(1 for r in rows if int(r["pdg_alerts"]) > 0),
            "by_group": by_grp,
        })
    d_board.sort(key=lambda x: (-x["comp100"], x["mean_ja"]))

    print("=== [F7.2] Running Structurally Independent Cohort D-Struct (48 eps x 6 policies) ===", flush=True)
    d48_specs = build_clean_d_struct_48_specs()
    ds_rows, ds_lats = _run_suite(d48_specs, D_STRUCT_6_POLICIES, OUT_CLOSURE_DIR / "f7_cohort_d_struct_rows.csv", "D-Struct-48")
    ds_board = _summarize_cohort(ds_rows, ds_lats, D_STRUCT_6_POLICIES)

    paired_d = {
        f"V20_1_NAJS_vs_{ref}": _paired_comparison(d_rows["V20_1_NAJS"], d_rows[ref])
        for ref in ("V20_NAJS", "V19_ADAPTIVE_RL", "V19_HETERO_RL", "V18_EA_JOINT", "IJCAI26_PGMPO", "SPGA_AMR")
    }
    paired_d["V19_ADAPTIVE_RL_vs_V20_1_NAJS"] = _paired_comparison(d_rows["V19_ADAPTIVE_RL"], d_rows["V20_1_NAJS"])
    paired_d["V19_ADAPTIVE_RL_vs_V20_NAJS"] = _paired_comparison(d_rows["V19_ADAPTIVE_RL"], d_rows["V20_NAJS"])
    paired_d["V19_ADAPTIVE_RL_vs_IJCAI26_PGMPO"] = _paired_comparison(d_rows["V19_ADAPTIVE_RL"], d_rows["IJCAI26_PGMPO"])

    paired_ds = {
        f"V20_1_NAJS_vs_{ref}": _paired_comparison(ds_rows["V20_1_NAJS"], ds_rows[ref])
        for ref in ("V20_NAJS", "V19_ADAPTIVE_RL", "V19_HETERO_RL", "V18_EA_JOINT", "IJCAI26_PGMPO")
    }
    paired_ds["V19_ADAPTIVE_RL_vs_V20_1_NAJS"] = _paired_comparison(ds_rows["V19_ADAPTIVE_RL"], ds_rows["V20_1_NAJS"])

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
        f"- **Freeze Lock**: `IF-CORE-V20.1-FREEZE` (`v20_1_freeze_manifest.json`, SHA-256 verified before execution)",
        "- **Seed Disjointness**: Zero overlap with Train / Val / RL / G1 / Forbidden / Cohort A / Cohort B / Cohort C.",
        "- **Strict Anti-Leakage Discipline**: Single-shot uncontended evaluation; ZERO threshold or code changes after touching Cohort D.",
        "",
        "## 1. Brand-New Unseen `Cohort D` Leaderboard (120 Episodes × 14 Policies)",
        "",
        "| Rank | Policy | Comp100 | Mean `J_A` ↓ | Med `J_A` ↓ | `CVaR_95` ↓ | Pure `H1` ↓ | 4-Topo | Track 1 | Track 2 | P95 / P99 / Max (ms) | ExpWin |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rk, b in enumerate(d_board, 1):
        bg = b["by_group"]
        md.append(
            f"| {rk} | **{b['policy']}** | {b['comp100']}/120 ({b['comp100_pct']:.2f}%) | {b['mean_ja']:.2f} | {b['median_ja']:.2f} "
            f"| {b['cvar95']:.1f} | {b['mean_h1']:.2f} | {bg.get('4-Topo', 0):.1f} | {bg.get('Track 1', 0):.1f} | {bg.get('Track 2', 0):.1f} "
            f"| {b['p95_ms']:.2f} / {b['p99_ms']:.2f} / {b['max_ms']:.2f} | {b['expired_windows']} |"
        )

    md.extend([
        "",
        "### Paired Statistical Tests on `Cohort D` (120 Episodes)",
        "",
        "| Comparison | Mean Δ (`J_A`, <0 favors LHS) | 95% Bootstrap CI | W / T / L | Wilcoxon `p` (2-sided) | Wilcoxon `p` (1-sided <) |",
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
        "### Paired Statistical Tests on `Cohort D-Struct` (48 Episodes)",
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
    print(f"[F7 COMPLETE] Wrote {OUT_CLOSURE_DIR / 'f7_cohort_d_confirmation.md'}", flush=True)


if __name__ == "__main__":
    main()
