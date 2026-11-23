"""EXP-04 (P1): Controlled Production-Transport Coupling Strength Gradient (`C0` -> `C3`).

Holds network topology (`rail_freight_workshop_v1`, 62 nodes), fleet size (`6` AMRs),
total transport task count (`8` tasks), operation count (`6` operations), and decision
budget (`35 ms`) strictly constant across 4 controlled coupling levels (`C0..C3`):
  - `C0_Weak` (Independent Operations): 1 task/op, dedicated bay/crew, no DAG, no blockage.
  - `C1_Moderate` (Shared Bay/Crew Contention): unlocks shared `eligible_bays` & `eligible_crews` pool.
  - `C2_Strong` (Multi-Task Kitting + Precedence DAG): 2-task kitting convergence + 3-stage op DAG.
  - `C3_Extreme` (Kitting + DAG + Resource Windows + Dynamic Corridor Blockage): full cross-domain coupling.

Writes outputs to `out/v19_scientific_audit/exp04_coupling_gradient.json` and `exp04_coupling_gradient.md`.
"""

from __future__ import annotations

import os
import platform

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Optional

if sys.platform == "win32" and getattr(platform, "_wmi", None) is not None:
    platform._wmi = None  # type: ignore[attr-defined]

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.baselines.policy_registry import run_discrete_episode
from amr_sched.cvcr.p2_perturbation import P2PerturbationConfig
from amr_sched.cvcr.production import OperationSnapshot
from amr_sched.entities import Agv, MaterialType, Task, TaskPriority
from amr_sched.network import NodeKind
from tools.benchmarks.compare_workshop_algorithms import _load_track1_network, _travel_estimate

OUT_DIR = ROOT / "out" / "v19_scientific_audit"
COUPLING_LEVELS: tuple[str, ...] = ("C0_Weak", "C1_Moderate", "C2_Strong", "C3_Extreme")
SEEDS: tuple[int, ...] = (901, 902, 903, 904, 905, 906)
POLICIES: tuple[str, ...] = (
    "V19_HETERO_RL",
    "V19_STAGED_RL",
    "V19_HETERO_BC",
    "V19_NO_PROD_COMM",
    "V18_EA_JOINT",
    "rule_v10",
    "IJCAI26_PGMPO",
)


def build_coupled_track1_scenario(
    seed: int,
    coupling_level: str,
) -> tuple[tuple[Any, ...], Optional[P2PerturbationConfig], int]:
    """Build strictly workload-matched 62-node Track 1 scenario at coupling level `C0..C3`."""
    if coupling_level not in COUPLING_LEVELS:
        raise ValueError(f"Unknown coupling_level {coupling_level!r}")

    # Deterministic seed invariant across levels so base arrivals & AGV starts match
    rng = random.Random(int(seed) * 1009 + 77)
    scene, net = _load_track1_network()
    starts = ["J_S_65", "J_N_45", "J_N_85", "W_DOOR", "W_COUPLER", "W_PARK"]
    if rng.random() < 0.5:
        starts[0], starts[2] = starts[2], starts[0]
    agvs = [
        Agv(f"R{i + 1}", max_speed=1.0, start_node=starts[i], load_time=2.0, unload_time=2.0)
        for i in range(6)
    ]

    core_specs = [
        ("Q_BOGIE", "D_BOGIE", MaterialType.BOGIE, ("D_BOGIE", "D_WHEELSET")),
        ("J_S_65", "D_WHEELSET", MaterialType.WHEELSET, ("D_WHEELSET", "D_BOGIE")),
        ("J_N_85", "D_BRAKE", MaterialType.BRAKE_VALVE, ("D_BRAKE", "D_STORE")),
        ("R_COUPLER", "D_COUPLER", MaterialType.BRAKE_VALVE, ("D_COUPLER", "D_DOOR")),
        ("R_DOOR", "D_DOOR", MaterialType.BRAKE_VALVE, ("D_DOOR", "D_COUPLER")),
        ("J_S_125", "D_ASSEMBLY", MaterialType.WHEELSET, ("D_ASSEMBLY", "D_WHEELSET")),
    ]
    buf_nodes = [n.node_id for n in net.nodes_of_kind(NodeKind.BUFFER, NodeKind.DISASSEMBLY)] or ["J_S_45", "Q_BOGIE"]
    aux_src_1 = rng.choice([nid for nid in buf_nodes if nid != "D_WHEELSET"] or ["J_S_45"])
    aux_src_2 = rng.choice([nid for nid in buf_nodes if nid != "D_BRAKE"] or ["Q_BOGIE"])

    tasks: list[Task] = []
    ops: list[OperationSnapshot] = []
    rels = [(idx // 2) * 16 + rng.randint(0, 6) for idx in range(6)]
    slack_mult = 1.65

    # Build 6 primary tasks (1..6) + 2 secondary tasks (7..8) so total task count is ALWAYS 8
    for idx, (pu, do, mat, alt_bays) in enumerate(core_specs):
        op_id = f"OP_C_{idx + 1}"
        tid_a = f"K_C_{idx + 1}"
        rel = rels[idx]
        est_t = _travel_estimate(net, pu, do, 0.8)
        t_due = int(round(rel + max(45.0, est_t * slack_mult)))
        is_urg = idx in (1, 5)
        t_obj = Task(
            tid_a,
            pu,
            do,
            release_step=rel,
            due_step=t_due,
            priority=TaskPriority.URGENT if is_urg else TaskPriority.NORMAL,
            material_type=mat,
        )
        t_obj.kit_op_id = op_id
        tasks.append(t_obj)

    # Secondary tasks 7 and 8: in C0/C1 they are standalone non-kitting tasks; in C2/C3 they are kitting partners for OP_C_2 and OP_C_3
    for j, (src_b, do_b, target_op_idx) in enumerate(((aux_src_1, "D_WHEELSET", 1), (aux_src_2, "D_BRAKE", 2)), start=1):
        tid_b = f"K_C_AUX_{j}"
        rel_b = rels[target_op_idx] + 5
        est_b = _travel_estimate(net, src_b, do_b, 0.8)
        t_b = Task(
            tid_b,
            src_b,
            do_b,
            release_step=rel_b,
            due_step=int(round(rel_b + max(48.0, est_b * slack_mult))),
            material_type=MaterialType.WHEELSET if j == 1 else MaterialType.BRAKE_VALVE,
        )
        if coupling_level in ("C2_Strong", "C3_Extreme"):
            t_b.kit_op_id = f"OP_C_{target_op_idx + 1}"
        tasks.append(t_b)

    # Construct the 6 operations according to coupling level
    for idx, (pu, do, mat, alt_bays) in enumerate(core_specs):
        op_id = f"OP_C_{idx + 1}"
        tid_a = f"K_C_{idx + 1}"
        rel = rels[idx]
        est_t = _travel_estimate(net, pu, do, 0.8)
        base_due = int(round(rel + max(45.0, est_t * slack_mult) + 28))

        if coupling_level == "C0_Weak":
            el_bays = (do,)
            el_crews = (idx + 1,)
            c_id = idx + 1
            kit_ids = (tid_a,)
            pred = None
            r_wins: tuple[tuple[int, int], ...] = ()
            op_due = base_due + 15
        elif coupling_level == "C1_Moderate":
            el_bays = tuple(alt_bays)
            el_crews = (1, 2)
            c_id = (idx % 2) + 1
            kit_ids = (tid_a,)
            pred = None
            r_wins = ()
            op_due = base_due + 5
        elif coupling_level == "C2_Strong":
            el_bays = tuple(alt_bays)
            el_crews = (1, 2)
            c_id = (idx % 2) + 1
            if idx == 1:
                kit_ids = (tid_a, "K_C_AUX_1")
            elif idx == 2:
                kit_ids = (tid_a, "K_C_AUX_2")
            else:
                kit_ids = (tid_a,)
            pred_map = {1: "OP_C_1", 5: "OP_C_2", 3: "OP_C_3", 4: "OP_C_4"}
            pred = pred_map.get(idx)
            r_wins = ()
            op_due = base_due + (22 if pred in ("OP_C_1", "OP_C_3") else (42 if pred in ("OP_C_2", "OP_C_4") else 0))
        else:  # C3_Extreme
            el_bays = tuple(alt_bays)
            el_crews = (1, 2)
            c_id = (idx % 2) + 1
            if idx == 1:
                kit_ids = (tid_a, "K_C_AUX_1")
            elif idx == 2:
                kit_ids = (tid_a, "K_C_AUX_2")
            else:
                kit_ids = (tid_a,)
            pred_map = {1: "OP_C_1", 5: "OP_C_2", 3: "OP_C_3", 4: "OP_C_4"}
            pred = pred_map.get(idx)
            w_open = int(rel + est_t * 0.85)
            r_wins = ((0, w_open + 36), (w_open + 82, 1200)) if idx in (1, 2, 3, 4) else ()
            op_due = base_due + (18 if pred in ("OP_C_1", "OP_C_3") else (36 if pred in ("OP_C_2", "OP_C_4") else 0))

        ops.append(
            OperationSnapshot(
                op_id=op_id,
                bay_id=do,
                crew_id=c_id,
                tool_id=(idx % 3) + 1,
                proc_steps=12 + (idx % 3) * 3,
                due_step=op_due,
                predecessor_op_id=pred,
                predecessor_op_ids=(pred,) if pred else (),
                is_urgent=(idx in (1, 5)),
                kit_task_ids=kit_ids,
                eligible_bays=el_bays,
                eligible_crews=el_crews,
                resource_windows=r_wins,
            )
        )

    disruptions: dict[str, Any] = {
        "fault_agv": agvs[int(seed) % len(agvs)].agv_id,
        "fault_step": 24,
        "fault_window": 4 if coupling_level in ("C2_Strong", "C3_Extreme") else 0,
        "event2_regime": "OPEN_WINDOW_KITTING_BOTTLENECK" if coupling_level in ("C2_Strong", "C3_Extreme") else "NONE",
        "kitting_bottleneck_window": 4,
        "kitting_bottleneck_delay": 32 if coupling_level == "C3_Extreme" else (20 if coupling_level == "C2_Strong" else 0),
        "urgent_task_id": "T_URGENT_C",
        "urgent_pickup": "Q_BOGIE",
        "urgent_dropoff": "D_ASSEMBLY",
        "urgent_release_step": 52,
        "urgent_due_step": 160,
        "post_pickup_fault_step": 95,
        "post_pickup_fault_duration": 18 if coupling_level == "C3_Extreme" else 0,
    }

    p2_cfg: Optional[P2PerturbationConfig] = None
    if coupling_level == "C3_Extreme":
        p2_cfg = P2PerturbationConfig(
            seed=int(seed) * 29 + 11,
            enable_travel_drift=True,
            drift_prob=0.18,
            early_max_steps=1,
            late_max_steps=4,
            enable_dynamic_obstacles=True,
            obstacle_count=2,
            obstacle_duration_steps=28,
            obstacle_durations=(18, 42),
            obstacle_window_range=(25, 180),
            est_clear_horizon_steps=16,
            enable_local_planner_fail=False,
        )

    return (net, tasks, agvs, ops, disruptions), p2_cfg, 1000


def main() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_level: dict[str, dict[str, Any]] = {}
    raw_rows: list[dict[str, Any]] = []

    for lvl in COUPLING_LEVELS:
        pol_rows: dict[str, list[dict[str, Any]]] = {p: [] for p in POLICIES}
        for s in SEEDS:
            sc_tuple, p2_cfg, max_steps = build_coupled_track1_scenario(s, lvl)
            for pol in POLICIES:
                row = run_discrete_episode(
                    pol,
                    sc_tuple,
                    seed=s,
                    max_steps=max_steps,
                    p2_cfg=p2_cfg,
                    use_rust=True,
                    episode_key=f"{lvl}_s{s}",
                    clock_mode="wall",
                    backend="numpy",
                )
                row["coupling_level"] = lvl
                pol_rows[pol].append(row)
                raw_rows.append(row)

        lvl_summary: dict[str, Any] = {}
        for pol in POLICIES:
            rs = pol_rows[pol]
            lvl_summary[pol] = {
                "mean_net_obj": round(float(np.mean([r["net_obj"] for r in rs])), 2),
                "mean_cmax": round(float(np.mean([r["cmax"] for r in rs])), 2),
                "mean_tardiness": round(float(np.mean([r["tardiness"] for r in rs])), 2),
                "mean_task_comp": round(float(np.mean([r["task_comp"] for r in rs])), 4),
                "mean_prod_comp": round(float(np.mean([r["prod_comp"] for r in rs])), 4),
                "audit_pass_rate": round(float(np.mean([1.0 if r["audit_passed"] else 0.0 for r in rs])), 4),
            }
        full_net = lvl_summary["V19_HETERO_RL"]["mean_net_obj"]
        staged_net = lvl_summary["V19_STAGED_RL"]["mean_net_obj"]
        bc_net = lvl_summary["V19_HETERO_BC"]["mean_net_obj"]
        rule_net = lvl_summary["rule_v10"]["mean_net_obj"]
        by_level[lvl] = {
            "policies": lvl_summary,
            "delta_full_minus_staged": round(full_net - staged_net, 2),
            "pct_full_vs_staged": round(100.0 * (full_net - staged_net) / max(1.0, staged_net), 2),
            "delta_full_minus_bc": round(full_net - bc_net, 2),
            "delta_full_minus_rule_v10": round(full_net - rule_net, 2),
        }
        print(
            f"[EXP-04 {lvl}] Full={full_net:.2f} | Staged={staged_net:.2f} "
            f"(Delta={full_net - staged_net:+.2f}, {100.0*(full_net-staged_net)/max(1.0,staged_net):+.2f}%) | "
            f"BC={bc_net:.2f} | rule_v10={rule_net:.2f}",
            flush=True,
        )

    deltas = [by_level[lvl]["delta_full_minus_staged"] for lvl in COUPLING_LEVELS]
    report = {
        "experiment": "EXP-04_Coupling_Strength_Gradient",
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "coupling_levels": list(COUPLING_LEVELS),
        "seeds": list(SEEDS),
        "policies": list(POLICIES),
        "by_level": by_level,
        "full_minus_staged_trajectory": deltas,
    }

    json_path = OUT_DIR / "exp04_coupling_gradient.json"
    md_path = OUT_DIR / "exp04_coupling_gradient.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# EXP-04: Controlled Production-Transport Coupling Strength Gradient (`C0` -> `C3`)",
        "",
        "| Coupling Level | `V19_HETERO_RL` (Full) | `V19_STAGED_RL` (Staged) | $\\Delta_{\\text{Full}-\\text{Staged}}$ (%) | `V19_HETERO_BC` | `V19_NO_PROD_COMM` | `V18_EA_JOINT` | `rule_v10` | `IJCAI26_PGMPO` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for lvl in COUPLING_LEVELS:
        b = by_level[lvl]
        ps = b["policies"]
        md_lines.append(
            f"| **`{lvl}`** | **{ps['V19_HETERO_RL']['mean_net_obj']:.2f}** | {ps['V19_STAGED_RL']['mean_net_obj']:.2f} | "
            f"**{b['delta_full_minus_staged']:+.2f} ({b['pct_full_vs_staged']:+.2f}%)** | "
            f"{ps['V19_HETERO_BC']['mean_net_obj']:.2f} | {ps['V19_NO_PROD_COMM']['mean_net_obj']:.2f} | "
            f"{ps['V18_EA_JOINT']['mean_net_obj']:.2f} | {ps['rule_v10']['mean_net_obj']:.2f} | "
            f"{ps['IJCAI26_PGMPO']['mean_net_obj']:.2f} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[EXP-04] Saved {json_path} and {md_path} in {report['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
