"""Cross-Domain Commitment Dependency Model for LG-CVCR / AMR-Scheduler.

Faithfully implements the 5 commitment tiers and 4 coupling domains from LG-CVCR:
Tiers:
1. PHYSICALLY_IRREVOCABLE (completed tasks, finished operations)
2. EXECUTING_HARD (picked-up tasks on board AMR, actively running operations)
3. CONDITIONAL (operations with unfinished predecessors)
4. SOFT (assigned tasks not yet picked up - reassignable with penalty lambda_change)
5. UNCOMMITTED (pending tasks not yet assigned)

Domains:
- KITTING: tasks required for the same production operation
- PRECEDENCE: tasks feeding predecessor <-> successor operations
- SHARED_RESOURCE: tasks whose operations share bays, crews, or tools
- ROBOT_QUEUE: tasks sequentially queued on the same AMR
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from amr_sched.cvcr.snapshot import NativeSnapshot, OperationSnapshot, TaskSnapshot, op_predecessors

TIER_IRREVOCABLE = "PHYSICALLY_IRREVOCABLE"
TIER_EXECUTING_HARD = "EXECUTING_HARD"
TIER_CONDITIONAL = "CONDITIONAL"
TIER_SOFT = "SOFT"
TIER_UNCOMMITTED = "UNCOMMITTED"


@dataclass
class CommitmentClassification:
    op_tiers: dict[str, str]
    task_tiers: dict[str, str]
    frozen_tasks: set[str]
    releasable_tasks: set[str]
    disturbed_robots: set[str]
    disturbed_tasks: set[str]
    adj: dict[str, set[str]]  # task_id <-> task_id across 4 domains
    edge_reasons: dict[tuple[str, str], set[str]]


def classify_commitments(
    snapshot: NativeSnapshot,
    disturbed_robots: Optional[set[str]] = None,
) -> CommitmentClassification:
    t_commit = snapshot.tick

    # 1. Operation commitment tiers
    op_tiers: dict[str, str] = {}
    for op in snapshot.operations:
        if op.end_step is not None and op.end_step <= t_commit:
            op_tiers[op.op_id] = TIER_IRREVOCABLE
        elif op.started:
            op_tiers[op.op_id] = TIER_EXECUTING_HARD
        elif op_predecessors(op):
            op_tiers[op.op_id] = TIER_CONDITIONAL
        else:
            op_tiers[op.op_id] = TIER_SOFT

    # 2. Task commitment tiers
    task_tiers: dict[str, str] = {}
    frozen_tasks: set[str] = set()
    releasable_tasks: set[str] = set()

    for t in snapshot.tasks:
        st = t.status.lower()
        if st == "completed" or (t.complete_step is not None and t.complete_step <= t_commit):
            task_tiers[t.task_id] = TIER_IRREVOCABLE
            frozen_tasks.add(t.task_id)
        elif st in ("picked", "loading", "unloading") or t.pickup_step is not None:
            task_tiers[t.task_id] = TIER_EXECUTING_HARD
            frozen_tasks.add(t.task_id)
        elif t.assigned_agv is None or st == "pending":
            task_tiers[t.task_id] = TIER_UNCOMMITTED
            releasable_tasks.add(t.task_id)
        else:  # assigned, not yet picked up
            task_tiers[t.task_id] = TIER_SOFT
            releasable_tasks.add(t.task_id)

    # 3. Disturbed robots & tasks (including L1/L2 execution feedback faults & significant drifts)
    d_robots = set(disturbed_robots or set())
    fb = getattr(snapshot, "execution_feedback", None)
    delayed_robots: set[str] = set()
    for r in snapshot.robots:
        if r.faulted_until > t_commit or r.state == "faulted":
            d_robots.add(r.robot_id)
        if fb is not None:
            r_fb = fb.robot_states.get(r.robot_id)
            if r_fb is not None:
                if r_fb.is_hard_faulted:
                    d_robots.add(r.robot_id)
                elif r_fb.is_degraded and fb.robot_ready_offset_steps(r.robot_id) >= 4:
                    delayed_robots.add(r.robot_id)

    d_tasks: set[str] = set()
    for t in snapshot.tasks:
        if t.task_id in releasable_tasks:
            if t.assigned_agv is not None and (t.assigned_agv in d_robots or t.assigned_agv in delayed_robots):
                d_tasks.add(t.task_id)
            elif (
                getattr(t, "is_urgent", False)
                or str(getattr(t, "priority", "")).lower() in ("urgent", "high", "1")
                or (t.release_step == t_commit and t_commit > 0)
            ):
                d_tasks.add(t.task_id)

    # 4. Build 4-domain coupling graph over tasks
    adj: dict[str, set[str]] = {t.task_id: set() for t in snapshot.tasks}
    edge_reasons: dict[tuple[str, str], set[str]] = {}

    def _add_edge(u: str, v: str, reason: str) -> None:
        if u == v:
            return
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
        edge_reasons.setdefault((u, v), set()).add(reason)
        edge_reasons.setdefault((v, u), set()).add(reason)

    tasks_by_op: dict[str, list[str]] = {op.op_id: list(op.kit_task_ids) for op in snapshot.operations}
    for t in snapshot.tasks:
        if t.kit_op_id and t.kit_op_id in tasks_by_op:
            if t.task_id not in tasks_by_op[t.kit_op_id]:
                tasks_by_op[t.kit_op_id].append(t.task_id)

    # Domain 1: KITTING (tasks required for the same operation)
    for op_id, tids in tasks_by_op.items():
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                _add_edge(tids[i], tids[j], "KITTING")

    # Domain 2: PRECEDENCE (tasks feeding predecessor <-> successor ops)
    for op in snapshot.operations:
        for pid in op_predecessors(op):
            if pid in tasks_by_op:
                for u in tasks_by_op[pid]:
                    for v in tasks_by_op.get(op.op_id, []):
                        _add_edge(u, v, "PRECEDENCE")

    # Domain 3: SHARED_RESOURCE (ops sharing crew_id or tool_id or bay_id)
    ops = snapshot.operations
    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            if (
                ops[i].crew_id == ops[j].crew_id
                or ops[i].tool_id == ops[j].tool_id
                or ops[i].bay_id == ops[j].bay_id
            ):
                for u in tasks_by_op.get(ops[i].op_id, []):
                    for v in tasks_by_op.get(ops[j].op_id, []):
                        _add_edge(u, v, "SHARED_RESOURCE")

    # Domain 4: ROBOT_QUEUE (tasks assigned to the same AMR)
    by_robot: dict[str, list[str]] = {}
    for t in snapshot.tasks:
        if t.assigned_agv is not None:
            by_robot.setdefault(t.assigned_agv, []).append(t.task_id)
    for agv_id, tids in by_robot.items():
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                _add_edge(tids[i], tids[j], "ROBOT_QUEUE")

    return CommitmentClassification(
        op_tiers=op_tiers,
        task_tiers=task_tiers,
        frozen_tasks=frozen_tasks,
        releasable_tasks=releasable_tasks,
        disturbed_robots=d_robots,
        disturbed_tasks=d_tasks,
        adj=adj,
        edge_reasons=edge_reasons,
    )


def select_release_set(
    scope_mode: str,
    clf: CommitmentClassification,
    snapshot: NativeSnapshot,
    max_release: int = 12,
) -> set[str]:
    """Select the commitment release set U_t <= releasable_tasks.

    Modes:
    - S0_DIRECT_AFFECTED: only disturbed tasks
    - S1_FIXED_NEIGHBORHOOD: disturbed tasks + immediate graph neighbors up to budget
    - S2_CROSS_DOMAIN_CLOSURE: full 4-domain closure
    - S3_VALUE_EXPANDED_CVCR: production urgency leverage-guided expansion
    - S4_GLOBAL_RESCHEDULING: all releasable tasks
    """
    releasable = clf.releasable_tasks
    if not releasable:
        return set()

    if scope_mode in ("S4_GLOBAL_RESCHEDULING", "GLOBAL"):
        return set(releasable)

    seeds_list = sorted(t for t in clf.disturbed_tasks if t in releasable)
    if not seeds_list:
        seeds_list = sorted(releasable)[:2]
    seeds = set(seeds_list)

    if scope_mode == "S0_DIRECT_AFFECTED":
        return seeds

    if scope_mode == "S1_FIXED_NEIGHBORHOOD":
        out = set(seeds)
        for u in seeds_list:
            for v in sorted(clf.adj.get(u, ())):
                if v in releasable:
                    out.add(v)
                if len(out) >= max(4, len(seeds) + 3):
                    return out
        return out

    if scope_mode == "S2_CROSS_DOMAIN_CLOSURE":
        out = set(seeds)
        q = deque(seeds_list)
        while q and len(out) < max_release:
            u = q.popleft()
            for v in sorted(clf.adj.get(u, ())):
                if v in releasable and v not in out:
                    out.add(v)
                    q.append(v)
                    if len(out) >= max_release:
                        break
        return out

    # S3_VALUE_EXPANDED_CVCR:
    # Prioritizes tasks feeding high-urgency operations or tight deadlines
    op_by_id = {op.op_id: op for op in snapshot.operations}
    task_by_id = {t.task_id: t for t in snapshot.tasks}

    def urgency_score(tid: str) -> float:
        task = task_by_id.get(tid)
        if not task:
            return 0.0
        score = 0.0
        if task.due_step is not None:
            slack = task.due_step - snapshot.tick
            score += max(0.0, 100.0 - slack)
        if task.kit_op_id and task.kit_op_id in op_by_id:
            op = op_by_id[task.kit_op_id]
            if op.is_urgent:
                score += 50.0
            op_slack = op.due_step - snapshot.tick
            score += max(0.0, 100.0 - op_slack)
        return score

    out = set(seeds)
    candidates = sorted(sorted(releasable - seeds), key=urgency_score, reverse=True)
    for c in candidates:
        if len(out) >= max_release:
            break
        # Connect through coupling graph if possible, else take highest urgency
        if any(c in clf.adj.get(u, ()) for u in out) or len(out) < len(seeds) + 2:
            out.add(c)

    return out
