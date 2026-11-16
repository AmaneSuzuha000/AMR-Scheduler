#!/usr/bin/env python3
"""Run Differential Benchmark: B1 vs B2 incremental evaluator equivalence,
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
from amr_sched.simulation import SimulationConfig, Simulator

BASE = Path(__file__).resolve().parents[1]
EVIDENCE = BASE / "evidence"
EVIDENCE.mkdir(parents=True, exist_ok=True)


def build_test_world(n_agvs: int = 4, n_tasks: int = 12) -> tuple[Network, NativeSnapshot]:
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


def run_benchmark():
    diff_records = []
    shadow_records = []
    commit_records = []

    print("Running differential equivalence benchmark (B1 vs B2)...")
    for scale in [(4, 8), (4, 12), (6, 12)]:
        n_a, n_t = scale
        net, snap = build_test_world(n_agvs=n_a, n_tasks=n_t)

        eval_b1 = CVCREvaluator(net, snap, mode="B1")
        eval_b2 = CVCREvaluator(net, snap, mode="B2")

        # Base candidate
        policy = LGCVCRPolicy("CVCR_B2", net)
        init_proposal = policy.propose(snap, budget_ms=150.0)
        base_assign = init_proposal.proposed_assignment

        r_b1 = eval_b1.evaluate(base_assign)
        r_b2 = eval_b2.evaluate(base_assign)

        eval_b1.set_incumbent(base_assign, r_b1)
        eval_b2.set_incumbent(base_assign, r_b2)

        # Generate 20 test variations (modifying 1 or 2 robots)
        for cand_idx in range(20):
            cand = dict(base_assign)
            # Pick a task to reassign
            tid = f"T{(cand_idx % n_t) + 1}"
            new_r = f"agv_{(cand_idx + 1) % n_a}"
            cand[tid] = new_r

            t_b1_start = time.perf_counter()
            res_b1 = eval_b1.evaluate(cand)
            b1_lat = (time.perf_counter() - t_b1_start) * 1000.0

            t_b2_start = time.perf_counter()
            res_b2 = eval_b2.evaluate(cand)
            b2_lat = (time.perf_counter() - t_b2_start) * 1000.0

            diff = abs(res_b1.objective - res_b2.objective)
            speedup = b1_lat / max(1e-4, b2_lat)

            rec = {
                "scale": f"{n_a}a_{n_t}t",
                "candidate_index": cand_idx,
                "reassigned_task": tid,
                "target_robot": new_r,
                "b1_valid": res_b1.valid,
                "b2_valid": res_b2.valid,
                "valid_match": res_b1.valid == res_b2.valid,
                "b1_obj": round(res_b1.objective, 4),
                "b2_obj": round(res_b2.objective, 4),
                "obj_diff": round(diff, 6),
                "b1_latency_ms": round(b1_lat, 3),
                "b2_latency_ms": round(b2_lat, 3),
                "speedup_x": round(speedup, 2),
                "equivalent": diff < 1e-4 and res_b1.valid == res_b2.valid,
            }
            diff_records.append(rec)

        # Run Policy shadow proposal
        policy = LGCVCRPolicy("CVCR_B2", net)
        proposal = policy.propose(snap, budget_ms=150.0)
        shadow_records.append({
            "scale": f"{n_a}a_{n_t}t",
            "policy": proposal.policy_id,
            "snapshot_digest": proposal.snapshot_digest,
            "candidate_digest": proposal.candidate_digest,
            "predicted_obj": proposal.predicted_obj,
            "telemetry": proposal.telemetry,
        })

        # Validate & commit via NativeTransactionOwner
        tasks = [
            Task(t.task_id, pickup=t.pickup, dropoff=t.dropoff, release_step=t.release_step, due_step=t.due_step)
            for t in snap.tasks
        ]
        agvs = [
            Agv(r.robot_id, start_node=r.cur_node, cur_node=r.cur_node)
            for r in snap.robots
        ]
        sim = Simulator(net, tasks, agvs, planner=None, allocator=None, config=SimulationConfig(dt=1.0, max_steps=100))
        sim.reset()
        owner = NativeTransactionOwner(sim, snap.operations)
        receipt = owner.validate_and_commit(proposal)
        commit_records.append({
            "scale": f"{n_a}a_{n_t}t",
            "receipt_digest": receipt.receipt_digest,
            "accepted": receipt.accepted,
            "reasons": receipt.reasons,
            "committed_tick": receipt.committed_tick,
            "physical_authorization": receipt.physical_authorization,
        })

    # Write differential records
    with open(EVIDENCE / "differential_b1_vs_b2.jsonl", "w", encoding="utf-8") as f:
        for r in diff_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(diff_records)} records to evidence/differential_b1_vs_b2.jsonl")

    # Write shadow proposals
    with open(EVIDENCE / "native_shadow_proposals.jsonl", "w", encoding="utf-8") as f:
        for r in shadow_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(shadow_records)} records to evidence/native_shadow_proposals.jsonl")

    # Write commit receipts
    with open(EVIDENCE / "native_commits_receipts.jsonl", "w", encoding="utf-8") as f:
        for r in commit_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(commit_records)} records to evidence/native_commits_receipts.jsonl")

    # Summary statistics
    all_eq = all(r["equivalent"] for r in diff_records)
    mean_speedup = sum(r["speedup_x"] for r in diff_records) / len(diff_records)
    print(f"Differential equivalence verified: {all_eq} across {len(diff_records)} candidates.")
    print(f"Mean B2 evaluation speedup: {mean_speedup:.2f}x over B1.")


if __name__ == "__main__":
    run_benchmark()
