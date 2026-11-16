#!/usr/bin/env python3
"""Run Differential Benchmark: A1 vs B1 vs B2 incremental evaluator equivalence,
speedup metrics across topologies (ring, grid, workshop 62-node), Python vs Rust backends,
shadow evaluation, and native transaction validation.

Outputs:
- evidence/differential_b1_vs_b2.jsonl
- evidence/native_shadow_proposals.jsonl
- evidence/native_commits_receipts.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.policy import LGCVCRPolicy
from amr_sched.cvcr.snapshot import (
    NativeSnapshot,
    NativeSnapshotCapture,
    OperationSnapshot,
    RobotSnapshot,
    TaskSnapshot,
)
from amr_sched.cvcr.transaction import NativeTransactionOwner
from amr_sched.entities import Agv, Task
from amr_sched.layouts import build_layout
from amr_sched.network import Network
from amr_sched.planners.rust_spacetime_astar import (
    RustSpaceTimeAStarPlanner,
    is_rust_planner_available,
)
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.workshop import WorkshopScene, build_scenario

EVIDENCE = BASE / "evidence"
EVIDENCE.mkdir(parents=True, exist_ok=True)


def build_ring_world(n_agvs: int = 4, n_tasks: int = 12) -> tuple[Network, NativeSnapshot]:
    """Construct a ring world snapshot with operations and kitting dependencies."""
    net = build_layout("ring")
    robots = [
        RobotSnapshot(robot_id=f"agv_{i}", cur_node=f"W{i % 6 + 1}", state="idle")
        for i in range(n_agvs)
    ]
    tasks = []
    for i in range(n_tasks):
        p = f"W{i % 6 + 1}"
        d = f"W{(i + 2) % 6 + 1}"
        op_id = f"OP_{i // 2 + 1}"
        tasks.append(
            TaskSnapshot(
                task_id=f"T{i + 1}",
                pickup=p,
                dropoff=d,
                release_step=(i // 4) * 10,
                due_step=100 + i * 20,
                status="assigned" if i < n_agvs else "pending",
                assigned_agv=f"agv_{i}" if i < n_agvs else None,
                kit_op_id=op_id,
            )
        )
    operations = []
    for op_idx in range(n_tasks // 2):
        op_id = f"OP_{op_idx + 1}"
        pred = f"OP_{op_idx}" if op_idx > 0 else None
        operations.append(
            OperationSnapshot(
                op_id=op_id,
                kit_task_ids=(f"T{op_idx * 2 + 1}", f"T{op_idx * 2 + 2}"),
                bay_id=f"W{(op_idx + 1) % 6 + 1}",
                crew_id=(op_idx % 2) + 1,
                tool_id=(op_idx % 2) + 1,
                proc_steps=20,
                due_step=120 + op_idx * 30,
                predecessor_op_id=pred,
            )
        )
    snap = NativeSnapshot(
        schema="amr-lgcvcr.snapshot.v1",
        tick=0,
        version={"world": 1, "reservation": 1, "production": 1, "fleet": 1},
        world_fingerprint=f"ring_bench_{n_agvs}a_{n_tasks}t",
        robots=robots,
        tasks=tasks,
        operations=operations,
        reservations=[],
        now_monotonic=time.monotonic(),
    )
    return net, snap


def build_grid_world(n_agvs: int = 4, n_tasks: int = 12) -> tuple[Network, NativeSnapshot]:
    """Construct a 4x5 grid world snapshot with operations and kitting dependencies."""
    net = build_layout("grid")
    nodes = sorted(net.nodes.keys())
    robots = [
        RobotSnapshot(robot_id=f"agv_{i}", cur_node=nodes[i % len(nodes)], state="idle")
        for i in range(n_agvs)
    ]
    tasks = []
    for i in range(n_tasks):
        p = nodes[i % len(nodes)]
        d = nodes[(i + 4) % len(nodes)]
        op_id = f"OP_{i // 2 + 1}"
        tasks.append(
            TaskSnapshot(
                task_id=f"T{i + 1}",
                pickup=p,
                dropoff=d,
                release_step=(i // 4) * 10,
                due_step=120 + i * 20,
                status="assigned" if i < n_agvs else "pending",
                assigned_agv=f"agv_{i}" if i < n_agvs else None,
                kit_op_id=op_id,
            )
        )
    operations = []
    for op_idx in range(n_tasks // 2):
        op_id = f"OP_{op_idx + 1}"
        pred = f"OP_{op_idx}" if op_idx > 0 else None
        operations.append(
            OperationSnapshot(
                op_id=op_id,
                kit_task_ids=(f"T{op_idx * 2 + 1}", f"T{op_idx * 2 + 2}"),
                bay_id=nodes[(op_idx + 2) % len(nodes)],
                crew_id=(op_idx % 2) + 1,
                tool_id=(op_idx % 2) + 1,
                proc_steps=20,
                due_step=150 + op_idx * 30,
                predecessor_op_id=pred,
            )
        )
    snap = NativeSnapshot(
        schema="amr-lgcvcr.snapshot.v1",
        tick=0,
        version={"world": 1, "reservation": 1, "production": 1, "fleet": 1},
        world_fingerprint=f"grid_bench_{n_agvs}a_{n_tasks}t",
        robots=robots,
        tasks=tasks,
        operations=operations,
        reservations=[],
        now_monotonic=time.monotonic(),
    )
    return net, snap


def build_workshop_world(n_agvs: int = 4, n_tasks: int = 12) -> tuple[Network, NativeSnapshot]:
    """Construct a 62-node Workshop world snapshot."""
    net, tasks_raw, agvs_raw = build_scenario(n_tasks=n_tasks, n_agvs=n_agvs, seed=42)
    robots = [
        RobotSnapshot(robot_id=a.agv_id, cur_node=a.cur_node or a.start_node, state="idle")
        for a in agvs_raw
    ]
    tasks = []
    for i, t in enumerate(tasks_raw):
        op_id = f"OP_{i // 2 + 1}"
        tasks.append(
            TaskSnapshot(
                task_id=t.task_id,
                pickup=t.pickup,
                dropoff=t.dropoff,
                release_step=t.release_step,
                due_step=t.due_step,
                status="assigned" if i < n_agvs else "pending",
                assigned_agv=robots[i % n_agvs].robot_id if i < n_agvs else None,
                kit_op_id=op_id,
            )
        )
    operations = []
    for op_idx in range(n_tasks // 2):
        op_id = f"OP_{op_idx + 1}"
        pred = f"OP_{op_idx}" if op_idx > 0 else None
        operations.append(
            OperationSnapshot(
                op_id=op_id,
                kit_task_ids=(tasks[op_idx * 2].task_id, tasks[op_idx * 2 + 1].task_id),
                bay_id=tasks[op_idx * 2].dropoff,
                crew_id=(op_idx % 2) + 1,
                tool_id=(op_idx % 2) + 1,
                proc_steps=25,
                due_step=200 + op_idx * 40,
                predecessor_op_id=pred,
            )
        )
    snap = NativeSnapshot(
        schema="amr-lgcvcr.snapshot.v1",
        tick=0,
        version={"world": 1, "reservation": 1, "production": 1, "fleet": 1},
        world_fingerprint=f"workshop_bench_{n_agvs}a_{n_tasks}t",
        robots=robots,
        tasks=tasks,
        operations=operations,
        reservations=[],
        now_monotonic=time.monotonic(),
    )
    return net, snap


def run_benchmark() -> None:
    """Run full differential equivalence and performance benchmarks."""
    diff_records = []
    shadow_records = []
    commit_records = []

    print("Running differential equivalence and speedup benchmark (A1 vs B1 vs B2)...")
    suites = [
        ("ring", 4, 8),
        ("ring", 4, 12),
        ("ring", 6, 12),
        ("grid", 4, 8),
        ("grid", 4, 12),
        ("workshop", 4, 12),
    ]

    for topo, n_a, n_t in suites:
        print(f"  Benchmarking topology={topo}, scale={n_a}a_{n_t}t ...")
        if topo == "ring":
            net, snap = build_ring_world(n_agvs=n_a, n_tasks=n_t)
        elif topo == "grid":
            net, snap = build_grid_world(n_agvs=n_a, n_tasks=n_t)
        else:
            net, snap = build_workshop_world(n_agvs=n_a, n_tasks=n_t)

        eval_a1 = CVCREvaluator(net, snap, mode="A1", use_rust=True)
        eval_b1 = CVCREvaluator(net, snap, mode="B1", use_rust=True)
        eval_b2 = CVCREvaluator(net, snap, mode="B2", use_rust=True)

        policy = LGCVCRPolicy("CVCR_B2", net, use_rust=True)
        init_prop = policy.propose(snap, budget_ms=150.0)
        base_assign = init_prop.proposed_assignment

        r_a1_init = eval_a1.evaluate(base_assign)
        r_b1_init = eval_b1.evaluate(base_assign)
        r_b2_init = eval_b2.evaluate(base_assign)

        eval_b1.set_incumbent(base_assign, r_b1_init)
        eval_b2.set_incumbent(base_assign, r_b2_init)

        # 20 candidate evaluations (perturbing 1-2 tasks)
        rids = [r.robot_id for r in snap.robots]
        tids = [t.task_id for t in snap.tasks]
        for cand_idx in range(20):
            cand = dict(base_assign)
            tid = tids[cand_idx % len(tids)]
            new_r = rids[(cand_idx + 1) % len(rids)]
            cand[tid] = new_r

            t0 = time.perf_counter()
            res_a1 = eval_a1.evaluate(cand)
            a1_lat = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            res_b1 = eval_b1.evaluate(cand)
            b1_lat = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            res_b2 = eval_b2.evaluate(cand)
            b2_lat = (time.perf_counter() - t0) * 1000.0

            diff_b1_b2 = abs(res_b1.objective - res_b2.objective)
            speedup_vs_b1 = b1_lat / max(1e-4, b2_lat)
            speedup_vs_a1 = a1_lat / max(1e-4, b2_lat)

            rec = {
                "topology": topo,
                "scale": f"{n_a}a_{n_t}t",
                "candidate_index": cand_idx,
                "reassigned_task": tid,
                "target_robot": new_r,
                "backend": "rust",
                "a1_valid": res_a1.valid,
                "b1_valid": res_b1.valid,
                "b2_valid": res_b2.valid,
                "valid_match_b1_b2": res_b1.valid == res_b2.valid,
                "a1_obj": round(res_a1.objective, 4),
                "b1_obj": round(res_b1.objective, 4),
                "b2_obj": round(res_b2.objective, 4),
                "diff_b1_b2": round(diff_b1_b2, 6),
                "a1_latency_ms": round(a1_lat, 3),
                "b1_latency_ms": round(b1_lat, 3),
                "b2_latency_ms": round(b2_lat, 3),
                "speedup_b2_vs_b1": round(speedup_vs_b1, 2),
                "speedup_b2_vs_a1": round(speedup_vs_a1, 2),
                "b2_inc_hits": eval_b2.b2_inc_hits,
                "equivalent": diff_b1_b2 < 1e-4 and (res_b1.valid == res_b2.valid),
            }
            diff_records.append(rec)

        # Run shadow evaluation across all policies
        all_policies = [
            "CVCR_B2",
            "SACC_B2",
            "OUTCOME_MLP_FULL",
            "KEEP_RELEASE_MLP_BALANCED",
            "BUDGET_MATCHED_ALNS",
            "IDEA_CORRIDOR_CVCR_JOINT",
        ]
        for pol_id in all_policies:
            pol = LGCVCRPolicy(pol_id, net, use_rust=True)
            prop = pol.propose(snap, budget_ms=150.0)
            shadow_records.append({
                "topology": topo,
                "scale": f"{n_a}a_{n_t}t",
                "policy": prop.policy_id,
                "snapshot_digest": prop.snapshot_digest,
                "candidate_digest": prop.candidate_digest,
                "predicted_obj": prop.predicted_obj,
                "proposed_admitted_agvs": prop.proposed_admitted_agvs,
                "telemetry": prop.telemetry,
            })

            # Native transaction validation on ring and workshop
            if pol_id == "IDEA_CORRIDOR_CVCR_JOINT":
                tasks_sim = [
                    Task(t.task_id, pickup=t.pickup, dropoff=t.dropoff, release_step=t.release_step, due_step=t.due_step)
                    for t in snap.tasks
                ]
                agvs_sim = [
                    Agv(r.robot_id, start_node=r.cur_node, cur_node=r.cur_node)
                    for r in snap.robots
                ]
                sim = Simulator(
                    net,
                    tasks_sim,
                    agvs_sim,
                    planner=RustSpaceTimeAStarPlanner(max_speed=1.0, dt=1.0) if is_rust_planner_available() else None,
                    allocator=None,
                    config=SimulationConfig(dt=1.0, max_steps=100),
                )
                sim.reset()
                owner = NativeTransactionOwner(sim, snap.operations)
                receipt = owner.validate_and_commit(prop)
                commit_records.append({
                    "topology": topo,
                    "scale": f"{n_a}a_{n_t}t",
                    "policy": pol_id,
                    "receipt_digest": receipt.receipt_digest,
                    "accepted": receipt.accepted,
                    "reasons": receipt.reasons,
                    "committed_tick": receipt.committed_tick,
                    "physical_authorization": receipt.physical_authorization,
                })

    # Write evidence files
    with open(EVIDENCE / "differential_b1_vs_b2.jsonl", "w", encoding="utf-8") as f:
        for r in diff_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(diff_records)} records to evidence/differential_b1_vs_b2.jsonl")

    with open(EVIDENCE / "native_shadow_proposals.jsonl", "w", encoding="utf-8") as f:
        for r in shadow_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(shadow_records)} records to evidence/native_shadow_proposals.jsonl")

    with open(EVIDENCE / "native_commits_receipts.jsonl", "w", encoding="utf-8") as f:
        for r in commit_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(commit_records)} records to evidence/native_commits_receipts.jsonl")

    # Summary
    all_eq = all(r["equivalent"] for r in diff_records)
    mean_speedup_b1 = sum(r["speedup_b2_vs_b1"] for r in diff_records) / len(diff_records)
    mean_speedup_a1 = sum(r["speedup_b2_vs_a1"] for r in diff_records) / len(diff_records)
    all_deadlines_met = all(r["telemetry"]["deadline_met"] for r in shadow_records)
    all_receipts_accepted = all(r["accepted"] for r in commit_records)
    all_physical_auth_false = all(r["physical_authorization"] is False for r in commit_records)

    print("\n--- Gate 2 / Differential Benchmark Summary ---")
    print(f"Total candidates evaluated: {len(diff_records)}")
    print(f"Differential equivalence verified (B1 == B2): {all_eq}")
    print(f"Mean speedup B2 over B1: {mean_speedup_b1:.2f}x")
    print(f"Mean speedup B2 over A1: {mean_speedup_a1:.2f}x")
    print(f"All shadow policies met 150ms deadline: {all_deadlines_met}")
    print(f"All transactions accepted: {all_receipts_accepted}")
    print(f"Strict physical authorization disabled (G6): {all_physical_auth_false}")


if __name__ == "__main__":
    run_benchmark()
