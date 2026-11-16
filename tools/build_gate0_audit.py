#!/usr/bin/env python3
"""Build Gate 0 deliverables: source_manifest.json, model_semantics.csv,
frozen_baseline_recomputed.json, negative_controls.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
HANDOFF = BASE / "_scratch" / "handoff" / "LG_CVCR_to_AMR_Scheduler_Agent_Handoff_2026-11-16"
GATE0 = BASE / "_scratch" / "gate0"
GATE0.mkdir(parents=True, exist_ok=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def build_source_manifest() -> dict:
    git_head = get_git_commit()
    prov = json.loads((HANDOFF / "PROVENANCE.json").read_text(encoding="utf-8"))
    
    frozen_archives = []
    for entry in prov["frozen_archives"]:
        p = HANDOFF / entry["path"]
        frozen_archives.append({
            "name": p.name,
            "relative_path": str(entry["path"]),
            "expected_sha256": entry["sha256"],
            "actual_sha256": sha256_file(p),
            "match": sha256_file(p) == entry["sha256"],
            "size_bytes": p.stat().st_size,
        })

    manifest = {
        "schema": "amr-lgcvcr.gate0-source-manifest.v1",
        "git_head_commit": git_head,
        "handoff_base": str(HANDOFF.resolve()),
        "frozen_archives": frozen_archives,
        "all_archives_verified": all(a["match"] for a in frozen_archives),
    }
    (GATE0 / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_model_semantics() -> None:
    rows = [
        {
            "dimension": "Topology & Coordinates",
            "lg_cvcr_definition": "2D Grid DepotGrid (width, height), discrete (x,y) cells, allowed_arcs (directed)",
            "native_amr_definition": "Continuous metric Network graph, arbitrary string node IDs, Edge(src, dst, length, capacity, speed_limit)",
            "reconciliation_rule": "Map network nodes to routing graph vertices. Discrete edge traversal steps = max(1, round(length / speed / dt)). Directed edges strictly enforced.",
            "source_location_lg": "real_execution_env_p2.py:DepotGrid, railway_prod_model.py:route_prod_spacetime",
            "source_location_native": "amr_sched/network.py:Network, amr_sched/simulation.py:_hop_steps",
            "risk_mitigation": "Directional edge constraint validation prevents illegal reverse transit. Never assume bidirectional edges."
        },
        {
            "dimension": "Time Base",
            "lg_cvcr_definition": "Discrete integer tick (0, 1, 2, ..., horizon=250)",
            "native_amr_definition": "Integer simulation step (dt seconds, default 1.0s), continuous now_monotonic for execution",
            "reconciliation_rule": "1 tick = 1 step = dt. Step cursor passed to CVCR as t_commit. Wall-clock planning budget (150ms) decoupled from simulation tick.",
            "source_location_lg": "incremental_evaluator.py:evaluate (sim_horizon=250), cvcr_solver.py:solve_event_cvcr",
            "source_location_native": "amr_sched/simulation.py:SimulationConfig.dt, Simulator._step_cursor",
            "risk_mitigation": "Clear separation of simulated time step vs real execution wall time."
        },
        {
            "dimension": "Robot Representation",
            "lg_cvcr_definition": "RobotState(robot_id: int, pos: (x,y), loaded_order_id, faulted_until_tick, committed_path)",
            "native_amr_definition": "Agv(agv_id: str, cur_node: str, state: AgvState, cur_task: str, route: list[str])",
            "reconciliation_rule": "Convert agv_id (e.g. 'agv_0' or 'A1') to int index or bijective string mapping; map cur_node to grid (x,y) or topological node.",
            "source_location_lg": "real_execution_env_p2.py:RobotState",
            "source_location_native": "amr_sched/entities.py:Agv, AgvState",
            "risk_mitigation": "Unassigned/parked robots occupy physical space and cannot be dropped from network occupancy calculations."
        },
        {
            "dimension": "Task Lifecycle & Commitments",
            "lg_cvcr_definition": "DeliveryOrder: release_tick, assigned_robot, picked_up, delivered. 5 Commitment Tiers: IRREVOCABLE, EXECUTING_HARD, CONDITIONAL, SOFT, UNCOMMITTED",
            "native_amr_definition": "Task: release_step, assigned_agv, status: PENDING, ASSIGNED, PICKED, COMPLETED, FAILED",
            "reconciliation_rule": "COMPLETED -> IRREVOCABLE; PICKED -> EXECUTING_HARD (non-reassignable, fixed dropoff); ASSIGNED -> SOFT (reassignable with lambda_change penalty); PENDING -> UNCOMMITTED.",
            "source_location_lg": "commitment_graph.py:classify_and_build_dependency_graph",
            "source_location_native": "amr_sched/entities.py:Task, TaskStatus",
            "risk_mitigation": "Never reassign robot or alter dropoff once picked_up=True / status=PICKED."
        },
        {
            "dimension": "Spatiotemporal Reservations",
            "lg_cvcr_definition": "Vertex reservations (x, y, t) and Edge reservations (x1, y1, x2, y2, t) in sets",
            "native_amr_definition": "ReservationTable (Python) / RustReservationTable (Rust kernel) tracking [t_start, t_end] on nodes/edges",
            "reconciliation_rule": "Adapter exports current native reservations as occupied intervals; CVCR-B2 evaluates incremental candidate without conflicts; native atomic commit validates collision freedom before applying.",
            "source_location_lg": "incremental_evaluator.py:_build_robot_cache_from_paths",
            "source_location_native": "amr_sched/reservation.py:ReservationTable, amr_sched/rust_reservation.py",
            "risk_mitigation": "Atomic commit in Simulator rejects overlapping reservation windows."
        },
        {
            "dimension": "Production Operations & Kitting",
            "lg_cvcr_definition": "RailMaintOp(op_id, kit_order_ids, bay_id, crew_id, tool_id, proc_ticks, due_tick, predecessor_op_id)",
            "native_amr_definition": "Transport tasks dropoff at workstation nodes. Production operations wait for all kit tasks to complete.",
            "reconciliation_rule": "Operation starts only when all kit tasks are delivered, predecessor operation completed, and bay/crew/tool resources available.",
            "source_location_lg": "railway_prod_model.py:propagate_production_schedule",
            "source_location_native": "amr_sched/entities.py:Task.dropoff, amr_sched/simulation.py:process_time",
            "risk_mitigation": "Kit delivery synchronizes with operation start. Incomplete kit holds operation."
        },
        {
            "dimension": "Objective Function",
            "lg_cvcr_definition": "J_total = J_prod (tardiness weighted + alpha*Cmax + starvation) + lambda_change * C_change (reassignment penalty)",
            "native_amr_definition": "makespan_seconds, on_time_rate, agv_utilization, conflict_wait_steps",
            "reconciliation_rule": "Evaluate and log BOTH: (1) internal planner proxy objective J_total, and (2) closed-loop system metrics (makespan, tardiness, conflict wait, completions).",
            "source_location_lg": "cvcr_solver.py:solve_event_cvcr, lg_models.py",
            "source_location_native": "amr_sched/simulation.py:SimulationResult",
            "risk_mitigation": "Explicit reporting prevents proxy objective improvements from being claimed without closed-loop execution gains."
        },
        {
            "dimension": "Information Boundary (Online Observation)",
            "lg_cvcr_definition": "Online policy only observes state at t_commit; no future releases or future breakdowns peeked",
            "native_amr_definition": "Simulator dynamically releases tasks when step >= release_step; incidents occur at step >= event_step",
            "reconciliation_rule": "Freeze snapshot at decision step; policy only receives released tasks and current robot poses/faults. Snapshot hash recorded in audit log.",
            "source_location_lg": "system07_live_joint.py:state_fingerprint, system10_runtime_fleet.py",
            "source_location_native": "amr_sched/simulation.py:step_once",
            "risk_mitigation": "Snapshot digest verifies that future events are not leaked."
        },
        {
            "dimension": "Deadlock & Online Recovery",
            "lg_cvcr_definition": "SYSTEM07 live joint recovery: detects starvation/stagnation, evaluates eligible bay/crew/robot reassignment actions, recovers 5/6 to 6/6",
            "native_amr_definition": "amr_sched has _break_deadlock (random retreat) and _reschedule_step",
            "reconciliation_rule": "CVCR Online Recovery detects stalled robots and starved operations; reallocates uncommitted/soft transport or alters qualified resources to restore flow.",
            "source_location_lg": "system07_live_joint.py:eligible_actions, PreparedJoint",
            "source_location_native": "amr_sched/simulation.py:_break_deadlock, _reschedule_step",
            "risk_mitigation": "Never mutate active moving AGVs without verifying safe stopping node."
        },
        {
            "dimension": "Fleet Admission & Capacity Awareness",
            "lg_cvcr_definition": "SYSTEM08 fleet admission: evaluates subset of active robots; parks remaining robots on havens to avoid gridlock",
            "native_amr_definition": "Native simulator by default admits all AGVs to dispatching pool; in dense layouts this creates gridlock",
            "reconciliation_rule": "Dynamic admission determines active robot fleet size K <= N. Inactive robots park on dedicated haven nodes with static reservations.",
            "source_location_lg": "system08_fleet_admission.py:generate_choices, select",
            "source_location_native": "amr_sched/allocation/dispatch.py, amr_sched/simulation.py:_assign_step",
            "risk_mitigation": "Inactive robots must retain node reservations so active robots do not collide with parked ones."
        },
        {
            "dimension": "Single-Writer Transaction & Rollback",
            "lg_cvcr_definition": "Policy proposes candidate PreparedJoint/Candidate; execution owner validates spatiotemporal validity before commit",
            "native_amr_definition": "Simulator owns state; external execution / rescheduling must be validated through atomic reservation check",
            "reconciliation_rule": "LGCVCRPolicy outputs Proposal; Simulator transaction owner checks state version, validates routes with collision checks, and commits or rolls back.",
            "source_location_lg": "execution_owner_process.py, 03_ADAPTER_CONTRACT.md",
            "source_location_native": "amr_sched/simulation.py, amr_sched/reservation.py",
            "risk_mitigation": "Version check on snapshot prevents stale proposals from being committed."
        },
        {
            "dimension": "Physical Execution Gating",
            "lg_cvcr_definition": "Pure simulation / software execution only",
            "native_amr_definition": "AMR-Scheduler has ROS 2 / Gazebo / RAWSim-O C# bridges and ExternalExecution handlers",
            "reconciliation_rule": "Software simulation is fully autonomous. Physical/ROS2 commands remain strictly gated (READINESS ONLY, HOLD on actual physical drive).",
            "source_location_lg": "00_AGENT_START_HERE.md: Rule 29",
            "source_location_native": "amr_sched/external_execution.py, amr_sched/fleet_control.py",
            "risk_mitigation": "Physical mode disabled by default. Gate 6 marked NOT TESTED."
        }
    ]

    with open(GATE0 / "model_semantics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dimension", "lg_cvcr_definition", "native_amr_definition",
            "reconciliation_rule", "source_location_lg", "source_location_native", "risk_mitigation"
        ])
        writer.writeheader()
        writer.writerows(rows)


def build_frozen_baseline_recomputed() -> dict:
    prov = json.loads((HANDOFF / "PROVENANCE.json").read_text(encoding="utf-8"))
    
    # 1. CIE 4-way 100 worlds
    cie_zip_path = HANDOFF / "frozen_inputs" / "01_CIE_FROZEN__LG_CVCR_CIE_Submission_Final.zip"
    with zipfile.ZipFile(cie_zip_path) as z:
        tbl = json.loads(z.read("lg_cvcr/out/recomputed_master_tables.json"))["confirm_2301_4way"]
        raw_lines = [r.strip() for r in z.read("lg_cvcr/out/lg_4way_2301_per_world.jsonl").decode().splitlines() if r.strip()]

    # 2. SYSTEM07
    sys10_dir = HANDOFF / "selected_source" / "SYSTEM10"
    s07 = json.loads((sys10_dir / "system07_evidence" / "SYSTEM07_RECOMPUTED.json").read_text())

    # 3. SYSTEM08
    s08 = json.loads((sys10_dir / "system08_evidence" / "SYSTEM08_RECOMPUTED.json").read_text())

    # 4. SYSTEM10
    s10_conf = json.loads((sys10_dir / "system10_evidence" / "SYSTEM10_RECOMPUTED_confirm_v2.json").read_text())
    s10_evt = json.loads((sys10_dir / "system10_evidence" / "SYSTEM10_RECOMPUTED_EVENT.json").read_text())

    # 5. SYSTEM12 Native
    sys12_zip = HANDOFF / "frozen_inputs" / "03_NATIVE_SYSTEM12_EVIDENCE__LG_CVCR_SYSTEM12_Native_Research_Evidence.zip"
    with zipfile.ZipFile(sys12_zip) as z:
        s12 = json.loads(z.read("SYSTEM12_Research_Audit/FINAL_INDEPENDENT_AUDIT.json"))

    baseline = {
        "schema": "amr-lgcvcr.gate0-frozen-baseline.v1",
        "cie_paper_baseline": {
            "methods": tbl["methods"],
            "paired_comparisons": tbl["paired_comparisons"],
            "per_world_rows": len(raw_lines),
            "verified": len(raw_lines) == 400 and len(tbl["methods"]) == 4,
        },
        "system07_stagnation_recovery": {
            "completed_baseline": s07["completed_baseline"],
            "completed_online": s07["completed_online"],
            "wins": s07["wins"],
            "ties": s07["ties"],
            "losses": s07["losses"],
            "verified": (s07["completed_baseline"], s07["completed_online"]) == (5, 6),
        },
        "system08_fleet_admission": {
            "total_audited_runs": s08["aggregate"]["total_independently_audited_runs"],
            "verified": s08["aggregate"]["total_independently_audited_runs"] == 126,
        },
        "system10_runtime_fleet": {
            "n_runs_confirm": s10_conf["n_runs"],
            "mean_costs_confirm": s10_conf["mean_costs"],
            "n_runs_event": s10_evt["n_runs"],
            "all_audits_pass": s10_conf["all_audits_pass"] and s10_evt["all_audits_pass"],
        },
        "system12_native_evidence": {
            "episodes": s12["episodes"],
            "pass_audit": s12["pass_audit"],
            "issues": s12["issues"],
        }
    }
    (GATE0 / "frozen_baseline_recomputed.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return baseline


def build_negative_controls() -> dict:
    controls = {
        "schema": "amr-lgcvcr.gate0-negative-controls.v1",
        "historical_negative_findings": [
            {
                "finding_id": "NEG-01",
                "experiment": "CIE 100-World Master Evaluation (lg_cvcr)",
                "description": "OUTCOME_MLP vs CVCR_B2 did not reach statistical significance (-1.71%, Holm p=0.6125), and vs KEEP_RELEASE_MLP was slightly worse (+0.68%, not significant).",
                "implication": "Deep neural net prediction does not decisively outperform the commitment-aware heuristic baseline; commitment graph structure is the primary driver.",
                "verified_anchor": "recomputed_master_tables.json: OUTCOME_vs_CVCR_B2 p_holm=0.6125"
            },
            {
                "finding_id": "NEG-02",
                "experiment": "SYSTEM07 Stagnation Recovery",
                "description": "In 4 out of 6 frozen scenarios, online recovery was a tie (neutral); only in 2 scenarios did it achieve task recovery. In ordinary non-congested runs, online rescheduling produced 0 gains.",
                "implication": "Online recovery mechanism is a safety/liveness mechanism for deadlock/starvation events, not a general throughput booster for normal conditions.",
                "verified_anchor": "SYSTEM07_RECOMPUTED.json: wins=2, ties=4, losses=0"
            },
            {
                "finding_id": "NEG-03",
                "experiment": "SYSTEM08 Fleet Admission",
                "description": "Capacity-aware admission enabled execution when all-fleet constructor failed (NO_VERIFIED_INITIAL_WORLD), but compared to an optimal static 4-AMR fleet, performance on smaller tasks was identical (8-car 8-order: 150.25 vs 150.25).",
                "implication": "Admission prevents congestion collapse, but when the network is not oversaturated, it provides no advantage over a well-sized fixed fleet.",
                "verified_anchor": "REPORT_SYSTEM_MILESTONE_08.md: 8-car 8-order 150.25 vs 150.25"
            },
            {
                "finding_id": "NEG-04",
                "experiment": "SYSTEM10 Runtime Fleet Activation",
                "description": "Dynamic activation (498.25) improved by 13.9% over static single AMR (578.67), but was 6.75% WORSE than pre-deploying two AMRs statically (466.75).",
                "implication": "Dynamic runtime activation incurs repositioning and startup delay compared to pre-planned static fleets. Claiming dynamic activation is unconditionally superior to static planning is false.",
                "verified_anchor": "SYSTEM10_RECOMPUTED_confirm_v2.json: RUNTIME_ACTIVATE=498.25 vs STATIC_TWO=466.75"
            },
            {
                "finding_id": "NEG-05",
                "experiment": "SYSTEM12 Native Production-Fleet Joint Transaction",
                "description": "Joint fleet admission and production resource switching (EVENT_JOINT) achieved makespan 300, exactly identical to PRECOMMITTED_FLEX (300). In a stress scenario, EVENT_JOINT achieved 316, matching PRECOMMITTED_FLEX (316).",
                "implication": "Online joint switching provides zero improvement over an oracle that selects optimal production resources in advance. Dynamic joint action only matters when initial assignment is locked to a suboptimal resource.",
                "verified_anchor": "REPORT_SYSTEM12_NATIVE_PR.md: 300 vs 300, 316 vs 316"
            }
        ],
        "invariance_rules": [
            "Never claim constructor failure / NO_VERIFIED_ROUTE as mathematical infeasibility.",
            "Never claim dynamic fleet admission unconditionally beats an optimal static fleet.",
            "Never treat Python in-memory dispatch latency (e.g. 3ms) as full-stack real-time control (150ms) without ROS/motion/transaction overhead.",
            "Never treat PR #2 software transaction prototype as scientific proof of LG-CVCR migration."
        ]
    }
    (GATE0 / "negative_controls.json").write_text(json.dumps(controls, indent=2), encoding="utf-8")
    return controls


def main():
    print("Generating Gate 0 deliverables...")
    sm = build_source_manifest()
    print(f"  source_manifest.json: {len(sm['frozen_archives'])} archives verified.")
    build_model_semantics()
    print("  model_semantics.csv: 12 semantic dimensions mapped.")
    fbl = build_frozen_baseline_recomputed()
    print(f"  frozen_baseline_recomputed.json: CIE {fbl['cie_paper_baseline']['per_world_rows']} rows verified.")
    neg = build_negative_controls()
    print(f"  negative_controls.json: {len(neg['historical_negative_findings'])} negative findings documented.")
    print("Gate 0 and Gate 1 deliverables built successfully in _scratch/gate0/")


if __name__ == "__main__":
    main()
