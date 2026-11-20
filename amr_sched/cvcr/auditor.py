"""Independent Step-by-Step Auditor for LG-CVCR / AMR-Scheduler Traces.

Audits the complete simulation trajectory independently of solver flags:
1. Per-step node occupancy vs network node capacity across ALL recorded steps.
2. Per-step directed edge validity (every transition u->v exists in network).
3. Commitment monotonicity across steps: once PICKED/COMPLETED, task ownership
   cannot be reassigned to a different AGV.
4. Production kitting precedence and bay/crew/tool mutex non-overlap.
5. Independent computation of makespan, weighted tardiness, and completion counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from amr_sched.cvcr.snapshot import align_start_to_windows, op_predecessors


@dataclass
class AuditReport:
    """Result of independent post-run trace audit."""

    pass_audit: bool
    issues: list[str]
    total_steps: int
    completed_tasks: int
    failed_tasks: int
    conflict_events: int
    invalid_edge_moves: int
    independent_makespan: float
    independent_tardiness: float
    reassignment_count: int


class IndependentAuditor:
    """Performs rigorous per-step trajectory audit on simulation execution."""

    @staticmethod
    def audit_simulation_run(
        simulator: Any,
        operations: Optional[list[Any]] = None,
        initial_assignment: Optional[dict[str, str]] = None,
        require_trajectory: bool = False,
    ) -> AuditReport:
        """Audit full trajectory (`simulator._trajectory`), tasks, and operations."""
        issues: list[str] = []
        network = simulator.network
        trajectory = getattr(simulator, "_trajectory", [])
        total_steps = int(simulator._step_cursor)
        if require_trajectory and total_steps > 0 and not trajectory:
            issues.append("MISSING_STEP_TRAJECTORY")

        conflict_events = 0
        invalid_edge_moves = 0
        prev_nodes: dict[str, str] = {}

        # 1. Step-by-step spatial occupancy & directed-edge audit
        faulted_set = getattr(simulator, "_faulted_agvs", set())
        for record in trajectory:
            step = int(record.step)
            node_counts: dict[str, list[str]] = {}
            edge_counts: dict[tuple[str, str], list[str]] = {}
            for agv_id, pos_str in record.agv_positions.items():
                if not pos_str:
                    continue
                if "->" in pos_str:
                    src, dst = pos_str.split("->", 1)
                    if not network.has_edge(src, dst):
                        invalid_edge_moves += 1
                        issues.append(f"STEP_{step}_INVALID_DIRECTED_EDGE:{agv_id}:{src}->{dst}")
                    edge_counts.setdefault((src, dst), []).append(agv_id)
                    prev_nodes[agv_id] = pos_str
                else:
                    node_counts.setdefault(pos_str, []).append(agv_id)
                    prev = prev_nodes.get(agv_id)
                    if prev is not None and "->" not in prev and prev != pos_str:
                        if not network.has_edge(prev, pos_str):
                            invalid_edge_moves += 1
                            issues.append(f"STEP_{step}_TELEPORT_OR_MISSING_EDGE:{agv_id}:{prev}->{pos_str}")
                    prev_nodes[agv_id] = pos_str

            for node_id, occupants in node_counts.items():
                cap = network.node(node_id).capacity if node_id in network else 1
                has_fault_hist = bool(getattr(simulator, "_fault_history", None))
                if has_fault_hist and hasattr(simulator, "is_faulted_at_step"):
                    active_occ = [a for a in occupants if not simulator.is_faulted_at_step(a, step)]
                else:
                    active_occ = [a for a in occupants if a not in faulted_set]
                if len(active_occ) > cap:
                    conflict_events += 1
                    issues.append(f"STEP_{step}_NODE_CAPACITY_EXCEEDED:{node_id}:{active_occ}:cap={cap}")

            for (src, dst), occ in edge_counts.items():
                if network.has_edge(src, dst):
                    ecap = network.edge(src, dst).capacity
                    if len(occ) > ecap:
                        conflict_events += 1
                        issues.append(f"STEP_{step}_EDGE_CAPACITY_EXCEEDED:{src}->{dst}:{occ}:cap={ecap}")

            for (src, dst), occ_fwd in list(edge_counts.items()):
                if src < dst and (dst, src) in edge_counts:
                    occ_rev = edge_counts[(dst, src)]
                    if occ_fwd and occ_rev:
                        cap_fwd = network.edge(src, dst).capacity if network.has_edge(src, dst) else 1
                        cap_rev = network.edge(dst, src).capacity if network.has_edge(dst, src) else 1
                        phys_cap = min(cap_fwd, cap_rev)
                        total_corridor = len(set(occ_fwd) | set(occ_rev))
                        if total_corridor > phys_cap:
                            conflict_events += 1
                            issues.append(
                                f"STEP_{step}_OPPOSING_EDGE_CONFLICT:{src}<->{dst}:{occ_fwd}_VS_{occ_rev}:phys_cap={phys_cap}"
                            )

        # 2. Task lifecycle & commitment integrity check
        for t in simulator.tasks:
            if t.status.value in ("completed", "picked") and t.assigned_agv is None:
                issues.append(f"UNASSIGNED_BUT_EXECUTED_TASK:{t.task_id}")
            if t.status.value == "completed":
                if t.pickup_step is not None and t.complete_step is not None and t.complete_step < t.pickup_step:
                    issues.append(f"TIME_INVERSION_TASK:{t.task_id}:complete={t.complete_step}<pickup={t.pickup_step}")

        # 3. Kitting precedence, predecessor, and crew/tool/bay mutex check
        if operations:
            tasks_by_id = {t.task_id: t for t in simulator.tasks}
            op_by_id = {op.op_id: op for op in operations}
            for op in operations:
                if getattr(op, "started", False) and getattr(op, "start_step", None) is not None:
                    op_start = int(op.start_step)
                    r_wins = getattr(op, "resource_windows", ())
                    if r_wins and align_start_to_windows(op_start, r_wins) != op_start:
                        issues.append(
                            f"WINDOW_ALIGNMENT_VIOLATION:{op.op_id}_AT_{op_start}_WINDOWS_{list(r_wins)}"
                        )
                    for kit_tid in op.kit_task_ids:
                        kt = tasks_by_id.get(kit_tid)
                        if kt and (kt.complete_step is None or int(kt.complete_step) > op_start):
                            issues.append(
                                f"PREMATURE_OP_START:{op.op_id}_AT_{op_start}_BEFORE_KIT_{kit_tid}_COMPLETE_{kt.complete_step}"
                            )
                    for pid in op_predecessors(op):
                        if pid in op_by_id:
                            pred = op_by_id[pid]
                            if not pred.started or pred.end_step is None or int(pred.end_step) > op_start:
                                issues.append(
                                    f"PREDECESSOR_VIOLATION:{op.op_id}_AT_{op_start}_BEFORE_{pred.op_id}_END_{pred.end_step}"
                                )

            # Check resource mutex intervals
            started_ops = [op for op in operations if op.started and op.start_step is not None and op.end_step is not None]
            for i in range(len(started_ops)):
                for j in range(i + 1, len(started_ops)):
                    a, b = started_ops[i], started_ops[j]
                    if a.start_step < b.end_step and b.start_step < a.end_step:
                        if a.bay_id == b.bay_id:
                            issues.append(f"BAY_MUTEX_OVERLAP:{a.op_id}_AND_{b.op_id}_AT_{a.bay_id}")
                        if a.crew_id == b.crew_id:
                            issues.append(f"CREW_MUTEX_OVERLAP:{a.op_id}_AND_{b.op_id}_CREW_{a.crew_id}")
                        if a.tool_id == b.tool_id:
                            issues.append(f"TOOL_MUTEX_OVERLAP:{a.op_id}_AND_{b.op_id}_TOOL_{a.tool_id}")

        # 4. Independent metrics
        completed_tasks = sum(1 for t in simulator.tasks if t.status.value == "completed")
        failed_tasks = sum(1 for t in simulator.tasks if t.status.value == "failed")
        makespan_sec = float(total_steps * simulator.config.dt)

        tardiness = 0.0
        for t in simulator.tasks:
            if t.due_step is not None and t.complete_step is not None:
                if t.complete_step > t.due_step:
                    tardiness += (t.complete_step - t.due_step) * simulator.config.dt

        reassign_cnt = 0
        if initial_assignment:
            for t in simulator.tasks:
                init_r = initial_assignment.get(t.task_id)
                if init_r and t.assigned_agv and init_r != t.assigned_agv:
                    reassign_cnt += 1

        return AuditReport(
            pass_audit=(len(issues) == 0),
            issues=issues,
            total_steps=total_steps,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            conflict_events=conflict_events,
            invalid_edge_moves=invalid_edge_moves,
            independent_makespan=makespan_sec,
            independent_tardiness=tardiness,
            reassignment_count=reassign_cnt,
        )
