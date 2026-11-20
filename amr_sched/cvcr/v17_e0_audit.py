"""Stage 1 (E0) Event-Level Dual-Order Reconciliation Audit Runner.

Tests 12 controlled 2-AMR bridge bottleneck cases:
  3 snapshot states (pre_pickup, loaded_at_staging, in_edge)
  x 2 window orderings (A_tight_B_loose, B_tight_A_loose)
  x 2 ID/robot permutations (normal_ids: agv_0->T_A, swapped_ids: agv_0->T_B)
plus 2 negative controls:
  - neg_loose_windows (wide windows for both -> minimal/zero window difference)
  - neg_shared_long_predecessor (long predecessor gates both -> zero op_start difference)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.cvcr.production import NativeProductionTracker
from amr_sched.cvcr.snapshot import NativeSnapshotCapture, OperationSnapshot
from amr_sched.cvcr.transaction import NativeTransactionOwner
from amr_sched.cvcr.v17_window_protection import plan_bridge_order_control
from amr_sched.entities import Agv, AgvState, Task, TaskStatus
from amr_sched.network import Edge, Network, Node, NodeKind
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.reservation import Interval
from amr_sched.simulation import Hop, SimulationConfig, Simulator, _Job


def _build_e0_bridge_network() -> Network:
    """Build symmetric 2-bay bridge network with safe staging buffers and 1-lane bridge ``C0<->C1``."""
    net = Network()
    nodes = {
        "S_A": (0.0, 2.0, NodeKind.BUFFER, 2),
        "S_B": (0.0, -2.0, NodeKind.BUFFER, 2),
        "LA": (2.0, 2.0, NodeKind.BUFFER, 2),
        "LB": (2.0, -2.0, NodeKind.BUFFER, 2),
        "C0": (4.0, 0.0, NodeKind.AISLE, 1),
        "C1": (8.0, 0.0, NodeKind.AISLE, 1),
        "WA": (10.0, 2.0, NodeKind.WORKSTATION, 2),
        "WB": (10.0, -2.0, NodeKind.WORKSTATION, 2),
    }
    for nid, (x, y, kind, cap) in nodes.items():
        net.add_node(Node(nid, x, y, kind=kind, capacity=cap))
    for u, v, d in (
        ("S_A", "LA", 2.0),
        ("S_B", "LB", 2.0),
        ("LA", "C0", 2.0),
        ("LB", "C0", 2.0),
        ("C0", "C1", 4.0),
        ("C1", "WA", 2.0),
        ("C1", "WB", 2.0),
    ):
        net.add_edge(Edge(u, v, length=d, speed_limit=1.0, capacity=1))
        net.add_edge(Edge(v, u, length=d, speed_limit=1.0, capacity=1))
    return net


def _make_e0_scenario(
    snapshot_state: str,
    window_order: str,
    id_perm: str,
    *,
    max_steps: int = 180,
) -> tuple[Simulator, NativeProductionTracker, dict[str, str], dict[str, int]]:
    """Construct exact controlled E0 scenario in target snapshot state."""
    net = _build_e0_bridge_network()
    if snapshot_state == "pre_pickup":
        win_tight, win_loose = ((0, 15),), ((16, 120),)
        due_tight, due_loose = 20, 40
    else:
        win_tight, win_loose = ((0, 11),), ((12, 120),)
        due_tight, due_loose = 16, 36

    extra_ops: list[OperationSnapshot] = []
    if window_order == "A_tight_B_loose":
        w_a, w_b = win_tight, win_loose
        d_a, d_b = due_tight, due_loose
        pred_a = pred_b = None
    elif window_order == "B_tight_A_loose":
        w_a, w_b = win_loose, win_tight
        d_a, d_b = due_loose, due_tight
        pred_a = pred_b = None
    elif window_order == "neg_loose_windows":
        w_a, w_b = ((0, 120),), ((0, 120),)
        d_a, d_b = 45, 45
        pred_a = pred_b = None
    elif window_order == "neg_shared_long_predecessor":
        w_a, w_b = ((0, 120),), ((0, 120),)
        d_a, d_b = 60, 60
        pred_a = pred_b = "OP_PRED"
        # Place OP_PRED on S_A so only pred_end=45 gates WA/WB without occupying them.
        extra_ops.append(
            OperationSnapshot(
                op_id="OP_PRED",
                kit_task_ids=(),
                bay_id="S_A",
                crew_id=9,
                tool_id=9,
                proc_steps=45,
                due_step=50,
                started=True,
                start_step=0,
                end_step=45,
            )
        )
    else:
        raise ValueError(window_order)

    t_a = Task("T_A", "LA", "WA", release_step=0, due_step=d_a)
    t_a.kit_op_id = "OP_A"  # type: ignore[attr-defined]
    t_b = Task("T_B", "LB", "WB", release_step=0, due_step=d_b)
    t_b.kit_op_id = "OP_B"  # type: ignore[attr-defined]
    tasks = [t_a, t_b]
    ops = extra_ops + [
        OperationSnapshot(
            op_id="OP_A",
            kit_task_ids=("T_A",),
            bay_id="WA",
            crew_id=1,
            tool_id=1,
            proc_steps=4,
            due_step=d_a + 4,
            predecessor_op_id=pred_a,
            resource_windows=w_a,
        ),
        OperationSnapshot(
            op_id="OP_B",
            kit_task_ids=("T_B",),
            bay_id="WB",
            crew_id=2,
            tool_id=2,
            proc_steps=4,
            due_step=d_b + 4,
            predecessor_op_id=pred_b,
            resource_windows=w_b,
        ),
    ]

    if id_perm == "normal_ids":
        agv_for_a, agv_for_b = "agv_0", "agv_1"
    elif id_perm == "swapped_ids":
        agv_for_a, agv_for_b = "agv_1", "agv_0"
    else:
        raise ValueError(id_perm)

    start_a = "S_A" if snapshot_state == "pre_pickup" else "LA"
    start_b = "S_B" if snapshot_state == "pre_pickup" else "LB"
    agv_map = {
        agv_for_a: Agv(agv_for_a, max_speed=1.0, start_node=start_a),
        agv_for_b: Agv(agv_for_b, max_speed=1.0, start_node=start_b),
    }
    agvs = [agv_map["agv_0"], agv_map["agv_1"]]

    sim = Simulator(
        network=net,
        tasks=tasks,
        agvs=agvs,
        planner=SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0),
        allocator=GreedyAllocator(),
        config=SimulationConfig(
            dt=1.0,
            max_steps=max_steps,
            load_time=2.0,
            unload_time=2.0,
            reschedule_policy=ReschedulePolicy.NONE,
            record_trajectory=True,
            idle_retreat=False,
        ),
    )
    sim.reset()
    base_assign = {"T_A": agv_for_a, "T_B": agv_for_b}
    base_prio = {"OP_A": 10, "OP_B": 10}
    if "OP_PRED" in {o.op_id for o in ops}:
        base_prio["OP_PRED"] = 0

    if snapshot_state in ("loaded_at_staging", "in_edge"):
        for agv, task, st_nd, dr_nd in (
            (agv_map[agv_for_a], t_a, "LA", "WA"),
            (agv_map[agv_for_b], t_b, "LB", "WB"),
        ):
            task.status = TaskStatus.PICKED
            task.assigned_agv = agv.agv_id
            task.assigned_step = 0
            task.pickup_step = 0
            agv.cur_node = st_nd
            agv.cur_task = task.task_id
            agv.state = AgvState.TO_DROPOFF
            h_init = [Hop(st_nd, "C0", 0, 2), Hop("C0", "C1", 2, 6), Hop("C1", dr_nd, 6, 8)]
            sim._jobs[agv.agv_id] = _Job(
                task_id=task.task_id, leg="to_dropoff", hops=h_init, idx=0, goal=dr_nd, reserved_upto=0
            )
            sim._load_end[agv.agv_id] = 0
            sim._drop_end[agv.agv_id] = 10

    if snapshot_state == "in_edge":
        sim._transit[agv_for_a] = ("LA", "C0", 0, 2)
        sim._jobs[agv_for_a].idx = 1
        sim._res.reserve_edge("LA", "C0", Interval(0, 2), agv_for_a)
        sim._res.reserve_node("C0", Interval(2, 3), agv_for_a)
        sim._node_claim[agv_for_a] = ("C0", Interval(2, 3))

    tracker = NativeProductionTracker(ops, sim.tasks)
    return sim, tracker, base_assign, base_prio


def _run_branch_execution(
    snapshot_state: str,
    window_order: str,
    id_perm: str,
    choice: str,
) -> dict[str, Any]:
    """Build snapshot, plan bridge control (``protect_A`` vs ``protect_B``), commit via Owner, and run Simulator."""
    sim, tracker, base_assign, base_prio = _make_e0_scenario(snapshot_state, window_order, id_perm)
    snap = NativeSnapshotCapture.capture(sim, tracker.operations, hide_future_releases=True)
    owner = NativeTransactionOwner(sim, operations=tracker.operations, production_tracker=tracker)
    ev = CVCREvaluator(sim.network, snap, use_rust=True, mode="B1", max_speed=1.0)

    leader_op, yielder_op = ("OP_A", "OP_B") if choice == "protect_A" else ("OP_B", "OP_A")
    plan = plan_bridge_order_control(
        sim.network,
        snap,
        ev,
        base_assign,
        base_prio,
        leader_op,
        yielder_op,
        sim=sim,
        allow_in_edge=False,
    )

    owner_attempt_reasons: list[str] = []
    committed = False
    fallback_reason = "none"
    if not plan.eligible:
        fallback_reason = plan.reason
        if snapshot_state == "in_edge":
            # Verify NativeTransactionOwner ALSO rejects illegal mid-edge repair if ever emitted.
            unsafe_plan = plan_bridge_order_control(
                sim.network,
                snap,
                ev,
                base_assign,
                base_prio,
                leader_op,
                yielder_op,
                sim=sim,
                allow_in_edge=True,
            )
            prop_bad = CandidateProposal(
                schema="amr-lgcvcr.proposal.v1",
                policy_id="V17_E0_PROBE",
                snapshot_digest=snap.compute_digest(),
                affected_robots=sorted({plan.leader_agv or "", plan.yielder_agv or ""} - {""}),
                affected_tasks=["T_A", "T_B"],
                affected_operations=[leader_op, yielder_op],
                proposed_assignment=dict(base_assign),
                proposed_routes=(unsafe_plan.eval_res.robot_paths if unsafe_plan.eval_res else {}),
                proposed_bays={},
                proposed_crews={},
                predicted_obj=0.0,
                predicted_makespan=0,
                candidate_digest=f"e0_bad_{choice}",
                telemetry={},
                proposed_op_priorities=dict(unsafe_plan.op_priority),
                proposed_live_repairs=unsafe_plan.live_repairs,
                proposed_bottleneck_locks=unsafe_plan.bottleneck_locks,
            )
            rcpt_bad = owner.validate_and_commit(prop_bad)
            owner_attempt_reasons = list(rcpt_bad.reasons)
    else:
        assert plan.eval_res is not None and plan.eval_res.valid
        prop = CandidateProposal(
            schema="amr-lgcvcr.proposal.v1",
            policy_id="V17_E0_PROBE",
            snapshot_digest=snap.compute_digest(),
            affected_robots=sorted({plan.leader_agv or "", plan.yielder_agv or ""} - {""}),
            affected_tasks=["T_A", "T_B"] if snapshot_state == "pre_pickup" else [],
            affected_operations=[leader_op, yielder_op],
            proposed_assignment=dict(base_assign),
            proposed_routes=plan.eval_res.robot_paths,
            proposed_bays={},
            proposed_crews={},
            predicted_obj=plan.eval_res.objective,
            predicted_makespan=max(plan.eval_res.op_end.values(), default=0),
            candidate_digest=f"e0_{choice}",
            telemetry={},
            proposed_op_priorities=dict(plan.op_priority),
            proposed_live_repairs=plan.live_repairs,
            proposed_bottleneck_locks=plan.bottleneck_locks,
        )
        rcpt = owner.validate_and_commit(prop)
        committed = bool(rcpt.accepted)
        owner_attempt_reasons = list(rcpt.reasons)

    bridge_enter: dict[str, int] = {}
    bridge_exit: dict[str, int] = {}
    first_exec: dict[str, str] = {}

    agv_for_op = {"OP_A": base_assign["T_A"], "OP_B": base_assign["T_B"]}
    while sim._step_cursor < sim.config.max_steps:
        step = int(sim._step_cursor)
        tracker.set_op_priorities(getattr(sim, "_op_priorities", None))
        tracker.step(step)
        all_t_done = all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) for t in sim.tasks)
        if all_t_done and tracker.all_done(step):
            break
        more = sim.step_once()
        cur_step = int(sim._step_cursor)
        if sim._trajectory:
            rec = sim._trajectory[-1]
            for op_k, aid in agv_for_op.items():
                pos = rec.agv_positions.get(aid, "")
                init_stage = "LA" if op_k == "OP_A" else "LB"
                init_buf = "S_A" if op_k == "OP_A" else "S_B"
                if op_k not in first_exec and pos and pos not in (init_stage, init_buf):
                    first_exec[op_k] = f"{pos}@{cur_step}"
                if op_k not in bridge_enter and "C0" in pos:
                    bridge_enter[op_k] = cur_step
                if op_k in bridge_enter and op_k not in bridge_exit and pos in ("WA", "WB"):
                    bridge_exit[op_k] = cur_step
        if not more and not tracker.all_done(cur_step):
            sim._step_cursor += 1
        elif not more and tracker.all_done(cur_step):
            break

    aud = IndependentAuditor.audit_simulation_run(sim, tracker.operations, require_trajectory=True)
    tmap = {t.task_id: t for t in sim.tasks}
    omap = {o.op_id: o for o in tracker.operations}
    cmax = int(sim._step_cursor)
    tard_task = sum(
        max(0, (int(t.complete_step) if t.complete_step is not None else cmax) - int(t.due_step or 0))
        for t in sim.tasks
    )
    tard_op = sum(
        max(0, (int(o.end_step) if o.end_step is not None else cmax) - int(o.due_step))
        for o in tracker.operations
    )
    uncomp_t = sum(1 for t in sim.tasks if t.status is not TaskStatus.COMPLETED)
    uncomp_o = sum(1 for o in tracker.operations if not (o.started and o.end_step is not None))
    c_tard_t = 2.0 * tard_task
    c_tard_o = 2.0 * tard_op
    c_unf_t = 5000.0 * uncomp_t
    c_unf_o = 3000.0 * uncomp_o
    net_obj = cmax + c_tard_t + c_tard_o + c_unf_t + c_unf_o

    return {
        "choice": choice,
        "requested": choice,
        "selected": choice if plan.eligible else "masked",
        "fallback_reason": fallback_reason,
        "eligible": bool(plan.eligible),
        "committed": committed,
        "receipt_reasons": owner_attempt_reasons,
        "first_exec": first_exec,
        "bridge_enter_A": bridge_enter.get("OP_A"),
        "bridge_exit_A": bridge_exit.get("OP_A"),
        "bridge_enter_B": bridge_enter.get("OP_B"),
        "bridge_exit_B": bridge_exit.get("OP_B"),
        "unload_A": tmap["T_A"].complete_step,
        "unload_B": tmap["T_B"].complete_step,
        "op_start_A": omap["OP_A"].start_step,
        "op_start_B": omap["OP_B"].start_step,
        "tard_A": max(0, int(omap["OP_A"].end_step or cmax) - int(omap["OP_A"].due_step)),
        "tard_B": max(0, int(omap["OP_B"].end_step or cmax) - int(omap["OP_B"].due_step)),
        "pred_obj": round(float(plan.eval_res.objective), 2) if (plan.eval_res and plan.eval_res.valid) else None,
        "cmax": cmax,
        "task_tardiness_cost": c_tard_t,
        "op_tardiness_cost": c_tard_o,
        "unfinished_task_penalty": c_unf_t,
        "unfinished_op_penalty": c_unf_o,
        "churn_cost": 0.0,
        "compute_cost": 0.0,
        "net_obj": round(net_obj, 2),
        "aud_pass": bool(aud.pass_audit),
        "aud_issues": aud.issues,
    }


def _write_e0_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Serialize E0 JSON and CSV reconciliation artifacts into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "v17_e0_reconciliation.json"
    out_csv = output_dir / "v17_e0_reconciliation.csv"
    out_json.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "case_id", "kind", "branch", "eligible", "committed", "fallback_reason",
            "bridge_enter_A", "bridge_enter_B", "unload_A", "unload_B",
            "op_start_A", "op_start_B", "net_obj", "aud_pass", "winner", "delta_net_B_minus_A",
        ])
        for r in rows:
            for b in (r["protect_A"], r["protect_B"]):
                w.writerow([
                    r["case_id"], r["kind"], b["choice"], b["eligible"], b["committed"], b["fallback_reason"],
                    b["bridge_enter_A"], b["bridge_enter_B"], b["unload_A"], b["unload_B"],
                    b["op_start_A"], b["op_start_B"], b["net_obj"], b["aud_pass"], r["winner"], r["delta_net_B_minus_A"],
                ])


def run_e0_suite(output_dir: Optional[Path] = None) -> dict[str, Any]:
    """Run all 12 controlled configs + 2 negative controls and return full E0 audit report."""
    cases: list[dict[str, Any]] = []
    for st in ("pre_pickup", "loaded_at_staging", "in_edge"):
        for wo in ("A_tight_B_loose", "B_tight_A_loose"):
            for ip in ("normal_ids", "swapped_ids"):
                cases.append({"case_id": f"{st}|{wo}|{ip}", "kind": "controlled_12", "state": st, "win": wo, "perm": ip})

    cases.append({"case_id": "neg_loose_windows", "kind": "neg_ctrl", "state": "loaded_at_staging", "win": "neg_loose_windows", "perm": "normal_ids"})
    cases.append({"case_id": "neg_shared_long_predecessor", "kind": "neg_ctrl", "state": "loaded_at_staging", "win": "neg_shared_long_predecessor", "perm": "normal_ids"})

    rows: list[dict[str, Any]] = []
    for c in cases:
        b_a = _run_branch_execution(c["state"], c["win"], c["perm"], "protect_A")
        b_b = _run_branch_execution(c["state"], c["win"], c["perm"], "protect_B")
        winner = (
            "protect_A"
            if b_a["net_obj"] < b_b["net_obj"]
            else ("protect_B" if b_b["net_obj"] < b_a["net_obj"] else "tie")
        )
        rows.append(
            {
                **c,
                "winner": winner,
                "delta_net_B_minus_A": round(float(b_b["net_obj"]) - float(b_a["net_obj"]), 2),
                "protect_A": b_a,
                "protect_B": b_b,
            }
        )

    if output_dir is not None:
        _write_e0_outputs(rows, output_dir)
    return {"rows": rows}


if __name__ == "__main__":
    report = run_e0_suite(output_dir=Path(__file__).resolve().parents[2] / "evidence")
    for r in report["rows"]:
        a, b = r["protect_A"], r["protect_B"]
        print(
            f"{r['case_id']:45s} win={r['winner']:10s} dNet={r['delta_net_B_minus_A']:7.1f} | "
            f"A(ent=({a['bridge_enter_A']},{a['bridge_enter_B']}), st=({a['op_start_A']},{a['op_start_B']}), net={a['net_obj']}, aud={a['aud_pass']}, rc={a['receipt_reasons'][:1]}) | "
            f"B(ent=({b['bridge_enter_A']},{b['bridge_enter_B']}), st=({b['op_start_A']},{b['op_start_B']}), net={b['net_obj']}, aud={b['aud_pass']}, rc={b['receipt_reasons'][:1]})"
        )
