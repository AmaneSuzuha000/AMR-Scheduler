"""Causal Joint Repair Cone (I01), VOI Budgeted Operator Search (I02), and J_A-Aligned Scorer (P-1).

Provides modular, zero-extra-dependency components designed to close the synergy loop between
AHG-RL global neural proposals and CVCR-IR (Seq-Diff) millisecond incremental evaluation:
  1. `score_candidate_ja_aligned`: Computes the exact J_A-aligned online evaluation proxy
     `C_max + 2*T_task + 2*T_op + 15*C_churn + unfinished_proxy` directly from `EvalResult`
     (`order_unload_ticks`, `op_end`), fixing `CVCREvaluator.objective`'s omission of
     per-task `due_step` tardiness.
  2. `build_causal_repair_cone`: Constructs the multi-layer cross-domain impact graph
     `H = (V_task U V_op U V_AMR U V_res, E)` rooted at the critical tardy/makespan bottleneck
     operation/task, propagating along kitting convergence, precedence DAG, shared bay/crew
     contention, and overlapping AMR spacetime paths, strictly preserving all `LOCKED` entities.
  3. `VOIBudgetedOperatorSearch`: Selects local cross-layer repair operators (`amr_reassign`,
     `amr_swap`, `queue_2opt`, `bay_crew_repair`, `kit_group_coalign`) inside the causal cone
     by expected gain per millisecond (`VOI = (g_hat + bonus) / (t_hat + eps)`), leveraging
     `CVCR-IR (Seq-Diff)` so 1-2 AMR surgical repairs reuse unchanged fleet reservations in-place.

Also includes clean falsifier neighborhood modes (`rand_lns`, `amr_only`, `op_only`) for
strict apples-to-apples ablation on quarantined DEV/VAL splits.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.snapshot import NativeSnapshot, op_predecessors
from amr_sched.cvcr.v12_proactive_guard import (
    compute_detour_delta_signals,
    compute_kitting_marginal_start_value,
)
from amr_sched.network import Network
from amr_sched.rl.v19_hetero_graph import V19HeteroGraph


def compute_candidate_ja_components(
    snapshot: NativeSnapshot,
    graph: V19HeteroGraph,
    ev_res: EvalResult,
    assign: dict[str, str],
    bays: dict[str, str],
    crews: dict[str, int],
    task_sequences: Optional[dict[str, list[str]]],
    rob_cur_healthy: dict[str, str],
    *,
    lambda_change: float = 15.0,
    sim_horizon: int = 1200,
    est_compute_ms: float = 0.0,
) -> dict[str, float]:
    """Compute exact J_A-aligned proxy score and decomposed efficiency/churn terms from `EvalResult`."""
    if not ev_res.valid:
        pen = float(ev_res.objective) + 20_000.0
        return {
            "score": pen,
            "cmax": float(sim_horizon),
            "t_task_cost": 0.0,
            "t_op_cost": 0.0,
            "tardiness_cost": 0.0,
            "unf_pen": 20_000.0,
            "churn_count": 0.0,
            "compute_cost": 0.05 * max(0.0, float(est_compute_ms)),
        }

    op_ends = ev_res.op_end
    unloads = ev_res.order_unload_ticks
    cmax_vals = [int(snapshot.tick)]
    if op_ends:
        cmax_vals.extend(int(v) for v in op_ends.values())
    if unloads:
        cmax_vals.extend(int(v) for v in unloads.values())
    cmax = max(cmax_vals)

    # Exact 2 * T_task + delivery slack term + unfinished task proxy
    t_task_cost = 0.0
    unf_pen = 0.0
    tot_deliv = 0.0
    for t in snapshot.tasks:
        if t.status.lower() == "completed":
            continue
        u_t = unloads.get(t.task_id)
        if u_t is None or int(u_t) >= sim_horizon:
            unf_pen += 5000.0
            u_val = sim_horizon
        else:
            u_val = int(u_t)
        tot_deliv += float(u_val)
        if t.due_step is not None:
            t_task_cost += 2.0 * float(max(0, u_val - int(t.due_step)))

    # Exact 2 * T_op + unfinished op proxy (uniform 2.0x weight matching env.py:662)
    t_op_cost = 0.0
    for op in snapshot.operations:
        e_o = op_ends.get(op.op_id)
        if e_o is None or int(e_o) >= sim_horizon:
            unf_pen += 3000.0
            e_val = sim_horizon
        else:
            e_val = int(e_o)
        tard_o = max(0, e_val - int(op.due_step))
        t_op_cost += 2.0 * float(tard_o)

    # Exact soft-commitment churn matching env.py:1241-1265
    chg_tasks = sum(
        1
        for t in snapshot.tasks
        if t.status.lower() == "assigned"
        and t.assigned_agv is not None
        and assign.get(t.task_id, t.assigned_agv) != t.assigned_agv
    )
    urg_new = sum(
        1
        for t in snapshot.tasks
        if t.assigned_agv is None
        and getattr(t, "is_urgent", False)
        and int(t.release_step) <= int(snapshot.tick)
        and assign.get(t.task_id) is not None
    )
    chg_ops = sum(
        1
        for oid in graph.op_ids
        if bays.get(oid) != graph.base_bays.get(oid) or crews.get(oid) != graph.base_crews.get(oid)
    )
    qh_preempts = 0
    if task_sequences:
        for aid_r, cur_tid in rob_cur_healthy.items():
            seq_r = task_sequences.get(aid_r)
            if seq_r and seq_r[0] != cur_tid and cur_tid in graph.clf.releasable_tasks:
                qh_preempts += 1

    churn_cnt = float(chg_tasks + urg_new + chg_ops + qh_preempts)
    deliv_cost = 0.5 * tot_deliv
    comp_cost = 0.05 * max(0.0, float(est_compute_ms))
    tie_break = 0.01 * float(ev_res.starvation_ticks)
    total = (
        float(cmax)
        + t_task_cost
        + t_op_cost
        + deliv_cost
        + unf_pen
        + float(lambda_change) * churn_cnt
        + comp_cost
        + tie_break
    )
    return {
        "score": total,
        "cmax": float(cmax),
        "t_task_cost": t_task_cost,
        "t_op_cost": t_op_cost,
        "tardiness_cost": t_task_cost + t_op_cost,
        "deliv_cost": deliv_cost,
        "unf_pen": unf_pen,
        "churn_count": churn_cnt,
        "compute_cost": comp_cost,
    }


def score_candidate_ja_aligned(
    snapshot: NativeSnapshot,
    graph: V19HeteroGraph,
    ev_res: EvalResult,
    assign: dict[str, str],
    bays: dict[str, str],
    crews: dict[str, int],
    task_sequences: Optional[dict[str, list[str]]],
    rob_cur_healthy: dict[str, str],
    *,
    lambda_change: float = 15.0,
    sim_horizon: int = 1200,
    est_compute_ms: float = 0.0,
) -> float:
    """Compute exact J_A-aligned proxy score from `EvalResult` (`C_max + 2*T_task + 2*T_op + 0.5*deliv + 15*churn + unf`)."""
    return float(
        compute_candidate_ja_components(
            snapshot,
            graph,
            ev_res,
            assign,
            bays,
            crews,
            task_sequences,
            rob_cur_healthy,
            lambda_change=lambda_change,
            sim_horizon=sim_horizon,
            est_compute_ms=est_compute_ms,
        )["score"]
    )


def observable_op_closure(snapshot: NativeSnapshot) -> set[str]:
    """Compute causal transitive closure of operations whose kits & predecessors are visible."""
    visible_tids = {t.task_id for t in snapshot.tasks}
    obs: set[str] = set()
    changed = True
    while changed:
        changed = False
        for op in snapshot.operations:
            if op.op_id in obs:
                continue
            if op.started or (
                all(kt in visible_tids for kt in op.kit_task_ids)
                and all(p in obs for p in op_predecessors(op))
            ):
                obs.add(op.op_id)
                changed = True
    return obs


def score_candidate_ja_unified(
    snapshot: NativeSnapshot,
    network: Network,
    graph: V19HeteroGraph,
    ev_res: EvalResult,
    assign: dict[str, str],
    bays: dict[str, str],
    crews: dict[str, int],
    obs_ops: set[str],
    fast_surr: Optional["FastJAWKittingSurrogate"],
    *,
    lambda_change: float = 15.0,
    sim_horizon: int = 1200,
) -> tuple[float, float]:
    """Causal horizon-clean J_A scorer unifying compact aisle cells and specialized workshops."""
    if not ev_res.valid:
        return float(ev_res.objective) + 20_000.0, 0.0

    tick = int(snapshot.tick)
    unloads = ev_res.order_unload_ticks
    op_ends = ev_res.op_end
    t_cost = 0.0
    unf_pen = 0.0
    deliv = 0.0
    for t in snapshot.tasks:
        if t.status.lower() == "completed":
            continue
        u = unloads.get(t.task_id)
        if u is None or int(u) >= sim_horizon:
            unf_pen += 5000.0
            uv = sim_horizon
        else:
            uv = int(u)
        deliv += float(uv)
        if t.due_step is not None:
            t_cost += 2.0 * float(max(0, uv - int(t.due_step)))

    chg_t = sum(
        1
        for t in snapshot.tasks
        if t.status.lower() == "assigned"
        and t.assigned_agv is not None
        and assign.get(t.task_id, t.assigned_agv) != t.assigned_agv
    )
    urg_new = sum(
        1
        for t in snapshot.tasks
        if t.assigned_agv is None
        and getattr(t, "is_urgent", False)
        and int(t.release_step) <= tick
        and assign.get(t.task_id) is not None
    )
    chg_o = sum(
        1
        for oid in graph.op_ids
        if bays.get(oid) != graph.base_bays.get(oid) or crews.get(oid) != graph.base_crews.get(oid)
    )

    is_workshop = len(network.nodes) >= 30
    op_cost = 0.0
    cmax_vals = [tick] + [int(v) for v in unloads.values()]
    if is_workshop:
        bload: dict[str, int] = {}
        for op in snapshot.operations:
            b = bays.get(op.op_id, op.bay_id)
            if not op.started:
                bload[b] = bload.get(b, 0) + 1
            if op.op_id in obs_ops:
                e = op_ends.get(op.op_id)
                if e is None or int(e) >= sim_horizon:
                    unf_pen += 3000.0
                    evv = sim_horizon
                else:
                    evv = int(e)
                    cmax_vals.append(evv)
                op_cost += 2.0 * float(max(0, evv - int(op.due_step)))
            elif fast_surr is not None:
                for kt in op.kit_task_ids:
                    to = fast_surr.tasks_by_id.get(kt)
                    if to is not None and to.status.lower() != "completed":
                        op_cost += 0.25 * float(fast_surr.tt(to.pickup, b))
        for ld in bload.values():
            if ld > 1:
                op_cost += 2.5 * float(ld * (ld - 1))
        sc = (
            1.5 * float(max(cmax_vals))
            + t_cost
            + op_cost
            + 0.5 * deliv
            + unf_pen
            + float(lambda_change) * float(chg_t + urg_new + chg_o)
            + 0.01 * float(ev_res.starvation_ticks)
        )
        return sc, t_cost + op_cost

    has_phantom = any(int(v) >= sim_horizon for v in op_ends.values())
    if not has_phantom:
        for v in op_ends.values():
            cmax_vals.append(int(v))
        for op in snapshot.operations:
            e = int(op_ends.get(op.op_id, sim_horizon))
            op_cost += 2.0 * float(max(0, e - int(op.due_step)))
        bpen = 0.0
        op_w = float(lambda_change)
    else:
        bload_c: dict[str, int] = {}
        for op in snapshot.operations:
            if not op.started:
                b = bays.get(op.op_id, op.bay_id)
                bload_c[b] = bload_c.get(b, 0) + 1
            if op.op_id in obs_ops:
                e = op_ends.get(op.op_id)
                if e is None or int(e) >= sim_horizon:
                    unf_pen += 3000.0
                    evv = sim_horizon
                else:
                    evv = int(e)
                    cmax_vals.append(evv)
                op_cost += 2.0 * float(max(0, evv - int(op.due_step)))
        bpen = sum(6.0 * max(0, c - 1) for c in bload_c.values())
        op_w = 0.0 if tick <= 4 else float(lambda_change)
    sc = (
        3.0 * float(max(cmax_vals))
        + t_cost
        + op_cost
        + 0.5 * deliv
        + bpen
        + unf_pen
        + float(lambda_change) * float(chg_t + urg_new)
        + op_w * float(chg_o)
    )
    return sc, t_cost + op_cost


@dataclass
class CausalRepairCone:
    """Multi-layer causal repair neighborhood strictly respecting LOCKED commitments."""

    root_op_id: Optional[str]
    root_task_id: Optional[str]
    cone_tasks: list[str]        # Releasable tasks in causal cone (ordered by criticality)
    cone_ops: list[str]          # Unstarted/modifiable ops in causal cone
    cone_amrs: list[str]         # Implicated + relief AMRs (capped to preserve CVCR-IR cache hits)
    cone_bays: list[str]
    cone_crews: list[int]
    bottleneck_kit_group: list[str]


def build_causal_repair_cone(
    snapshot: NativeSnapshot,
    graph: V19HeteroGraph,
    cur_eval: EvalResult,
    cur_assign: dict[str, str],
    cur_bays: dict[str, str],
    cur_crews: dict[str, int],
    *,
    max_cone_amrs: int = 3,
    max_cone_tasks: int = 6,
    max_cone_ops: int = 4,
) -> CausalRepairCone:
    """Trace cross-layer causal dependencies from the dominant bottleneck operation/task."""
    rel_set = set(graph.rel_tids)
    op_by_id = {op.op_id: op for op in snapshot.operations}
    task_by_id = {t.task_id: t for t in snapshot.tasks}
    op_ends = cur_eval.op_end
    op_starts = cur_eval.op_start
    op_res_ready = cur_eval.op_res_ready
    unloads = cur_eval.order_unload_ticks
    cmax = max([int(snapshot.tick)] + [int(v) for v in op_ends.values()] + [int(v) for v in unloads.values()])

    # 1. Rank operations by exact H1 bottleneck contribution (op tardiness + kit tardiness + makespan tail)
    op_scores: list[tuple[float, str]] = []
    for op in snapshot.operations:
        e_o = int(op_ends.get(op.op_id, cmax))
        t_o = max(0, e_o - int(op.due_step))
        kit_t = sum(
            max(0, int(unloads.get(kt, cmax)) - int(task_by_id[kt].due_step))
            for kt in op.kit_task_ids
            if kt in task_by_id and task_by_id[kt].due_step is not None
        )
        sc = 2.0 * float(t_o) + 2.0 * float(kit_t) + (25.0 if e_o >= cmax - 2 else 0.0)
        if op.is_urgent and t_o > 0:
            sc += 30.0
        # Prefer ops that have at least one releasable kit or modifiable bay/crew
        has_actionable = (
            any(kt in rel_set for kt in op.kit_task_ids)
            or (op.op_id not in graph.frozen_ops and (len(op.eligible_bays) > 1 or len(op.eligible_crews) > 1))
        )
        if has_actionable:
            op_scores.append((sc, op.op_id))
    op_scores.sort(key=lambda x: (-x[0], x[1]))

    # Also rank releasable tasks directly by task tardiness + downstream op tardiness
    task_scores: list[tuple[float, str]] = []
    for tid in graph.rel_tids:
        t = task_by_id.get(tid)
        if t is None:
            continue
        u_t = int(unloads.get(tid, cmax))
        t_tard = max(0, u_t - int(t.due_step)) if t.due_step is not None else 0
        op_tard = 0
        if t.kit_op_id and t.kit_op_id in op_by_id:
            o = op_by_id[t.kit_op_id]
            op_tard = max(0, int(op_ends.get(o.op_id, cmax)) - int(o.due_step))
        sc_t = 2.0 * float(t_tard) + 1.5 * float(op_tard) + 0.05 * float(u_t)
        task_scores.append((sc_t, tid))
    task_scores.sort(key=lambda x: (-x[0], x[1]))

    root_op_id = op_scores[0][1] if op_scores else None
    root_task_id = task_scores[0][1] if task_scores else None

    cone_ops_ordered: list[str] = []
    cone_tasks_ordered: list[str] = []
    bottleneck_kit_group: list[str] = []

    def _add_op(oid: str) -> None:
        if oid not in cone_ops_ordered and oid in op_by_id and len(cone_ops_ordered) < max_cone_ops:
            cone_ops_ordered.append(oid)

    def _add_task(tid: str) -> None:
        if tid in rel_set and tid not in cone_tasks_ordered and len(cone_tasks_ordered) < max_cone_tasks:
            cone_tasks_ordered.append(tid)

    # 2. Propagate cross-layer causal edges from root_op_id
    if root_op_id is not None:
        _add_op(root_op_id)
        root_op = op_by_id[root_op_id]
        # Layer 1: Kitting convergence edges (Op -> latest/tardy kit tasks)
        kit_ranked = sorted(
            [kt for kt in root_op.kit_task_ids if kt in rel_set],
            key=lambda kt: -int(unloads.get(kt, 0)),
        )
        bottleneck_kit_group = list(kit_ranked)
        for kt in kit_ranked:
            _add_task(kt)

        # Layer 2: Precedence DAG & shared bay/crew blocking edges
        r_bay = cur_bays.get(root_op_id, root_op.bay_id)
        r_crew = cur_crews.get(root_op_id, root_op.crew_id)
        r_ready = int(op_res_ready.get(root_op_id, snapshot.tick))
        for pid in op_predecessors(root_op):
            if pid in op_by_id and int(op_ends.get(pid, 0)) >= r_ready - 2:
                _add_op(pid)
                for pkt in op_by_id[pid].kit_task_ids:
                    _add_task(pkt)
        # Contending ops on the exact same bay or crew ending right before root_op starts
        for o2 in snapshot.operations:
            if o2.op_id == root_op_id or o2.started:
                continue
            e2 = int(op_ends.get(o2.op_id, 0))
            if abs(e2 - r_ready) <= 4 and (
                cur_bays.get(o2.op_id, o2.bay_id) == r_bay
                or cur_crews.get(o2.op_id, o2.crew_id) == r_crew
            ):
                _add_op(o2.op_id)
                for kt2 in o2.kit_task_ids:
                    _add_task(kt2)

    # Fill remaining task slots from highest-tardiness releasable tasks & ops
    for _, tid in task_scores:
        _add_task(tid)
    for _, oid in op_scores:
        _add_op(oid)

    # 3. Layer 3: AMR ownership + overlapping corridor conflict AMRs + least-loaded relief AMR
    cone_amrs: list[str] = []
    for tid in cone_tasks_ordered:
        rid = cur_assign.get(tid)
        if rid and rid not in cone_amrs and rid in graph.agv_ids:
            cone_amrs.append(rid)

    # Detect spacetime corridor/node overlap with cone AMRs in cur_eval.robot_paths
    if cone_amrs and cur_eval.robot_paths and len(cone_amrs) < max_cone_amrs:
        cone_occ: dict[str, list[int]] = {}
        for rid in cone_amrs:
            for nd, tk in cur_eval.robot_paths.get(rid, []):
                cone_occ.setdefault(nd, []).append(int(tk))
        overlap_scores: list[tuple[int, str]] = []
        for other_r in graph.agv_ids:
            if other_r in cone_amrs:
                continue
            ov = 0
            for nd, tk in cur_eval.robot_paths.get(other_r, []):
                ts_list = cone_occ.get(nd)
                if ts_list and any(abs(int(tk) - t0) <= 5 for t0 in ts_list):
                    ov += 1
            if ov > 0:
                overlap_scores.append((ov, other_r))
        overlap_scores.sort(key=lambda x: (-x[0], x[1]))
        for _, other_r in overlap_scores:
            if len(cone_amrs) >= max_cone_amrs:
                break
            cone_amrs.append(other_r)

    # Ensure at least one low-load relief AMR is available inside cone_amrs for transfer/swap
    if len(cone_amrs) < max_cone_amrs:
        amr_load = {aid: 0 for aid in graph.agv_ids}
        for tid_all, aid_all in cur_assign.items():
            if aid_all in amr_load:
                t_obj = task_by_id.get(tid_all)
                if t_obj is None or t_obj.status.lower() != "completed":
                    amr_load[aid_all] += 1
        # Prefer healthy (non-faulted) least-loaded AMRs
        faulted = set(graph.clf.disturbed_robots)
        relief_sorted = sorted(
            [aid for aid in graph.agv_ids if aid not in cone_amrs],
            key=lambda aid: (1 if aid in faulted else 0, amr_load.get(aid, 0), aid),
        )
        for aid in relief_sorted:
            if len(cone_amrs) >= max_cone_amrs:
                break
            cone_amrs.append(aid)

    cone_bays = sorted({cur_bays.get(oid, op_by_id[oid].bay_id) for oid in cone_ops_ordered if oid in op_by_id})
    cone_crews = sorted({cur_crews.get(oid, op_by_id[oid].crew_id) for oid in cone_ops_ordered if oid in op_by_id})
    return CausalRepairCone(
        root_op_id=root_op_id,
        root_task_id=root_task_id,
        cone_tasks=cone_tasks_ordered,
        cone_ops=cone_ops_ordered,
        cone_amrs=cone_amrs,
        cone_bays=cone_bays,
        cone_crews=cone_crews,
        bottleneck_kit_group=bottleneck_kit_group,
    )


class VOIBudgetedOperatorSearch:
    """Online Value-of-Information (VOI) budgeted local operator search over Causal Joint Repair Cone."""

    # Strictly surgical 1-2 AMR / 1-op moves preserving CVCR-IR (Seq-Diff) in-place reservation reuse
    OPERATORS: tuple[str, ...] = (
        "bay_crew_repair",
        "amr_reassign",
        "queue_2opt",
        "amr_swap",
    )

    def __init__(self, *, neighborhood_mode: str = "causal_cone", seed: int = 0) -> None:
        if neighborhood_mode not in ("causal_cone", "rand_lns", "amr_only", "op_only"):
            raise ValueError(f"Unsupported neighborhood_mode: {neighborhood_mode!r}")
        self.neighborhood_mode = neighborhood_mode
        self.seed = int(seed)
        # Persistent online VOI statistics per operator across epochs
        self._op_gain: dict[str, float] = {op: 25.0 for op in self.OPERATORS}
        self._op_ms: dict[str, float] = {op: 0.6 for op in self.OPERATORS}
        self._op_n: dict[str, float] = {op: 1.0 for op in self.OPERATORS}
        self._total_calls: int = 0

    def _allowed_operators(self) -> list[str]:
        if self.neighborhood_mode == "amr_only":
            return ["amr_reassign", "queue_2opt", "amr_swap"]
        if self.neighborhood_mode == "op_only":
            return ["bay_crew_repair"]
        return list(self.OPERATORS)

    def _rank_operators_voi(self) -> list[str]:
        ops = self._allowed_operators()
        ln_tot = float(np.log(max(2.0, float(self._total_calls) + 2.0)))
        scored: list[tuple[float, str]] = []
        for op in ops:
            n = max(1.0, self._op_n[op])
            g_hat = self._op_gain[op] / n
            t_hat = max(0.15, self._op_ms[op] / n)
            bonus = 18.0 * float(np.sqrt(ln_tot / n))
            voi = (g_hat + bonus) / (t_hat + 0.25)
            scored.append((voi, op))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [op for _, op in scored]

    def _record_voi(self, op_name: str, gain: float, elapsed_ms: float) -> None:
        self._total_calls += 1
        # Slight exponential forgetting (0.97) to stay adaptive across non-stationary regimes
        self._op_gain[op_name] = 0.97 * self._op_gain[op_name] + max(0.0, float(gain))
        self._op_ms[op_name] = 0.97 * self._op_ms[op_name] + max(0.1, float(elapsed_ms))
        self._op_n[op_name] = 0.97 * self._op_n[op_name] + 1.0

    def _generate_move(
        self,
        op_name: str,
        cone: CausalRepairCone,
        snapshot: NativeSnapshot,
        graph: V19HeteroGraph,
        cur_eval: EvalResult,
        cur_assign: dict[str, str],
        cur_bays: dict[str, str],
        cur_crews: dict[str, int],
        cur_prio: dict[str, int],
        cur_seq: Optional[dict[str, list[str]]],
        rng: np.random.Generator,
        attempt: int,
    ) -> Optional[tuple[dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]]]]:
        """Construct one strictly LOCKED-safe surgical move inside the active neighborhood."""
        rel_set = set(graph.rel_tids)
        if not rel_set and op_name != "bay_crew_repair":
            return None
        op_by_id = {op.op_id: op for op in snapshot.operations}
        agv_idx = {aid: i for i, aid in enumerate(graph.agv_ids)}
        task_idx = {tid: i for i, tid in enumerate(graph.rel_tids)}

        # Choose active tasks / ops / AMRs according to neighborhood_mode (for clean I01 falsification!)
        if self.neighborhood_mode == "rand_lns":
            cand_tasks = list(graph.rel_tids)
            rng.shuffle(cand_tasks)
            cand_tasks = cand_tasks[: max(1, min(4, len(cand_tasks)))]
            cand_amrs = list(graph.agv_ids)
            rng.shuffle(cand_amrs)
            cand_amrs = cand_amrs[: max(2, min(3, len(cand_amrs)))]
            cand_ops = [o.op_id for o in snapshot.operations if o.op_id not in graph.frozen_ops]
            rng.shuffle(cand_ops)
            cand_ops = cand_ops[: max(1, min(3, len(cand_ops)))]
        elif self.neighborhood_mode == "amr_only":
            # Conflict-AMR only: pick AMRs with longest finish time, ignore op DAG/bay
            amr_end = {
                aid: (cur_eval.robot_paths[aid][-1][1] if cur_eval.robot_paths.get(aid) else 0)
                for aid in graph.agv_ids
            }
            ranked_amrs = sorted(graph.agv_ids, key=lambda a: -amr_end.get(a, 0))
            cand_amrs = ranked_amrs[: min(3, len(ranked_amrs))]
            cand_tasks = [tid for tid in graph.rel_tids if cur_assign.get(tid) in cand_amrs]
            cand_ops = []
        elif self.neighborhood_mode == "op_only":
            cand_tasks = []
            cand_amrs = []
            cand_ops = [oid for oid in cone.cone_ops if oid not in graph.frozen_ops]
        else:
            cand_tasks = list(cone.cone_tasks)
            cand_amrs = list(cone.cone_amrs)
            cand_ops = [oid for oid in cone.cone_ops if oid not in graph.frozen_ops]

        if op_name == "amr_reassign":
            if not cand_tasks or len(cand_amrs) < 2:
                return None
            tid = cand_tasks[attempt % len(cand_tasks)]
            ti = task_idx.get(tid)
            if ti is None:
                return None
            cur_r = cur_assign.get(tid)
            # Pick best feasible alternative AMR in cone_amrs (lowest estimated arrival/load)
            alts = [
                aid for aid in cand_amrs
                if aid != cur_r and aid in agv_idx and bool(graph.task_agv_mask[ti, agv_idx[aid]])
            ]
            if not alts:
                alts = [
                    aid for aid in graph.agv_ids
                    if aid != cur_r and bool(graph.task_agv_mask[ti, agv_idx[aid]])
                ]
            if not alts:
                return None
            # Rank by travel distance + existing load from task_agv_feats[:, :, 0]
            alts.sort(key=lambda aid: (float(graph.task_agv_feats[ti, agv_idx[aid], 0]), aid))
            new_r = alts[(attempt // max(1, len(cand_tasks))) % len(alts)]
            n_assign = dict(cur_assign)
            n_assign[tid] = new_r
            # Update sequence cleanly if cur_seq exists so unchanged AMRs stay bit-identical
            n_seq = None
            if cur_seq is not None:
                n_seq = {k: list(v) for k, v in cur_seq.items()}
                if cur_r in n_seq and tid in n_seq[cur_r]:
                    n_seq[cur_r].remove(tid)
                n_seq.setdefault(new_r, []).append(tid)
            return n_assign, dict(cur_bays), dict(cur_crews), dict(cur_prio), n_seq

        if op_name == "queue_2opt":
            # Promote bottleneck task ahead inside its owning AMR's queue (changes ONLY 1 AMR -> ~0.3ms Seq-Diff hit!)
            base_seqs = cur_seq if cur_seq is not None else {k: list(v) for k, v in cur_eval.robot_order_seq.items()}
            for off in range(len(cand_tasks)):
                tid = cand_tasks[(attempt + off) % len(cand_tasks)]
                rid = cur_assign.get(tid)
                if not rid or rid not in base_seqs:
                    continue
                seq_r = list(base_seqs[rid])
                if tid not in seq_r:
                    continue
                pos = seq_r.index(tid)
                # Find earliest releasable index before pos
                target_pos = None
                for p_i in range(pos):
                    if seq_r[p_i] in rel_set:
                        target_pos = p_i
                        break
                if target_pos is None:
                    continue
                seq_r.pop(pos)
                seq_r.insert(target_pos, tid)
                n_seq = {k: list(v) for k, v in base_seqs.items()}
                n_seq[rid] = seq_r
                # Also boost tid's op priority slightly so production propagation matches
                n_prio = dict(cur_prio)
                t_obj = next((t for t in snapshot.tasks if t.task_id == tid), None)
                if t_obj and t_obj.kit_op_id and t_obj.kit_op_id in n_prio:
                    n_prio[t_obj.kit_op_id] = max(int(snapshot.tick), n_prio[t_obj.kit_op_id] - 15)
                return dict(cur_assign), dict(cur_bays), dict(cur_crews), n_prio, n_seq
            return None

        if op_name == "amr_swap":
            if len(cand_tasks) < 2:
                return None
            # Swap owners of two releasable tasks on different AMRs inside the cone (changes ONLY 2 AMRs!)
            pairs: list[tuple[str, str]] = []
            for i in range(len(cand_tasks)):
                t1 = cand_tasks[i]
                r1 = cur_assign.get(t1)
                ti1 = task_idx.get(t1)
                if not r1 or ti1 is None:
                    continue
                for j in range(i + 1, len(cand_tasks)):
                    t2 = cand_tasks[j]
                    r2 = cur_assign.get(t2)
                    ti2 = task_idx.get(t2)
                    if not r2 or r2 == r1 or ti2 is None:
                        continue
                    if bool(graph.task_agv_mask[ti1, agv_idx[r2]]) and bool(graph.task_agv_mask[ti2, agv_idx[r1]]):
                        pairs.append((t1, t2))
            if not pairs:
                return None
            t1, t2 = pairs[attempt % len(pairs)]
            r1, r2 = cur_assign[t1], cur_assign[t2]
            n_assign = dict(cur_assign)
            n_assign[t1], n_assign[t2] = r2, r1
            n_seq = None
            if cur_seq is not None:
                n_seq = {k: list(v) for k, v in cur_seq.items()}
                if r1 in n_seq and t1 in n_seq[r1]:
                    n_seq[r1][n_seq[r1].index(t1)] = t2
                if r2 in n_seq and t2 in n_seq[r2]:
                    n_seq[r2][n_seq[r2].index(t2)] = t1
            return n_assign, dict(cur_bays), dict(cur_crews), dict(cur_prio), n_seq

        if op_name == "bay_crew_repair":
            # Switch an unlocked cone operation to its earliest-available eligible bay or crew
            op_idx_map = {oid: i for i, oid in enumerate(graph.op_ids)}
            bay_idx_map = {bid: i for i, bid in enumerate(graph.bay_ids)}
            crew_idx_map = {int(cid): i for i, cid in enumerate(graph.crew_ids)}
            for off in range(len(cand_ops)):
                oid = cand_ops[(attempt + off) % len(cand_ops)]
                oi = op_idx_map.get(oid)
                op = op_by_id.get(oid)
                if oi is None or op is None or oid in graph.frozen_ops:
                    continue
                cur_b = cur_bays.get(oid, op.bay_id)
                cur_c = cur_crews.get(oid, op.crew_id)
                alt_bays = [
                    b for b in graph.bay_ids
                    if b != cur_b and bool(graph.op_bay_mask[oi, bay_idx_map[b]])
                ]
                alt_crews = [
                    int(c) for c in graph.crew_ids
                    if int(c) != cur_c and bool(graph.op_crew_mask[oi, crew_idx_map[int(c)]])
                ]
                if alt_bays:
                    # Pick bay with lowest current load/free-time in op_bay_feats
                    alt_bays.sort(key=lambda b: (float(graph.op_bay_feats[oi, bay_idx_map[b], 0]), b))
                    n_bays = dict(cur_bays)
                    n_bays[oid] = alt_bays[0]
                    return dict(cur_assign), n_bays, dict(cur_crews), dict(cur_prio), (
                        {k: list(v) for k, v in cur_seq.items()} if cur_seq is not None else None
                    )
                if alt_crews:
                    n_crews = dict(cur_crews)
                    n_crews[oid] = alt_crews[0]
                    return dict(cur_assign), dict(cur_bays), n_crews, dict(cur_prio), (
                        {k: list(v) for k, v in cur_seq.items()} if cur_seq is not None else None
                    )
            return None

        if op_name == "kit_group_coalign":
            # Co-assign all releasable kits of the bottleneck op onto distinct earliest-ready AMRs so kits arrive in parallel!
            kits = [kt for kt in cone.bottleneck_kit_group if kt in rel_set]
            if len(kits) < 2 or len(graph.agv_ids) < 2:
                return None
            n_assign = dict(cur_assign)
            used_r: set[str] = set()
            changed = False
            for kt in kits:
                ti = task_idx.get(kt)
                if ti is None:
                    continue
                feas = [
                    aid for aid in graph.agv_ids
                    if bool(graph.task_agv_mask[ti, agv_idx[aid]])
                ]
                if not feas:
                    continue
                # Prefer unused feasible AMR with minimal transit cost
                feas.sort(key=lambda aid: (1 if aid in used_r else 0, float(graph.task_agv_feats[ti, agv_idx[aid], 0]), aid))
                pick_r = feas[0]
                used_r.add(pick_r)
                if n_assign.get(kt) != pick_r:
                    n_assign[kt] = pick_r
                    changed = True
            if not changed:
                return None
            return n_assign, dict(cur_bays), dict(cur_crews), dict(cur_prio), None

        return None

    def refine(
        self,
        *,
        ev: CVCREvaluator,
        snapshot: NativeSnapshot,
        graph: V19HeteroGraph,
        seed_states: list[tuple[str, EvalResult, float, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]]]],
        score_fn: Any,
        seen_sigs: set[tuple[Any, ...]],
        deadline_s: float,
        max_evals: int = 6,
        hysteresis_margin: float = 10.0,
        keep_eval: EvalResult,
    ) -> tuple[EvalResult, float, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]], str, int]:
        """Run VOI-budgeted local operator search starting from best seed(s) inside remaining ms budget."""
        rng = np.random.default_rng(self.seed + int(snapshot.tick) * 31)
        # Start from the best valid seed state; also track KEEP if distinct
        valid_seeds = [s for s in seed_states if s[1].valid]
        if not valid_seeds:
            best = seed_states[0]
            return best[1], best[2], best[3], best[4], best[5], best[6], best[7], best[0], 0
        valid_seeds.sort(key=lambda s: s[2])
        cur_tag, cur_ev, cur_sc, cur_asn, cur_bays, cur_crews, cur_prio, cur_seq = valid_seeds[0]
        # Track overall best across all seeds
        best_tag, best_ev, best_sc = cur_tag, cur_ev, cur_sc
        best_asn, best_bays, best_crews, best_prio, best_seq = (
            dict(cur_asn), dict(cur_bays), dict(cur_crews), dict(cur_prio),
            ({k: list(v) for k, v in cur_seq.items()} if cur_seq is not None else None),
        )

        # Also consider polishing KEEP directly if best seed is neural (catches 1-churn surgical repairs!)
        polish_roots = [valid_seeds[0]]
        for s in valid_seeds[1:]:
            if s[0] == "keep":
                polish_roots.append(s)
                break

        evals_done = 0
        no_impr_streak = 0
        for root_idx, root in enumerate(polish_roots):
            if evals_done >= max_evals or _cvcr_now_s() >= deadline_s:
                break
            r_tag, r_ev, r_sc, r_asn, r_bays, r_crews, r_prio, r_seq = root
            cone = build_causal_repair_cone(snapshot, graph, r_ev, r_asn, r_bays, r_crews)
            op_order = self._rank_operators_voi()
            root_budget = max(2, max_evals - evals_done) if root_idx == len(polish_roots) - 1 else max(2, max_evals // 2)
            local_done = 0
            attempt = 0
            while local_done < root_budget and evals_done < max_evals and _cvcr_now_s() < deadline_s:
                op_name = op_order[attempt % len(op_order)]
                attempt += 1
                if attempt > max_evals * 3:
                    break
                mv = self._generate_move(
                    op_name, cone, snapshot, graph, r_ev, r_asn, r_bays, r_crews, r_prio, r_seq, rng, attempt - 1
                )
                if mv is None:
                    continue
                n_asn, n_bays, n_crews, n_prio, n_seq = mv
                sig = (
                    tuple(sorted(n_asn.items())),
                    tuple(sorted(n_bays.items())),
                    tuple(sorted(n_crews.items())),
                    tuple(sorted((aid, tuple(sq)) for aid, sq in n_seq.items())) if n_seq else None,
                )
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)

                t_op0 = _cvcr_now_s()
                ev_m = ev.evaluate(
                    n_asn,
                    op_priority=n_prio,
                    op_bay_overrides=n_bays,
                    op_crew_overrides=n_crews,
                    task_order_override=n_seq,
                )
                dt_ms = (_cvcr_now_s() - t_op0) * 1000.0
                evals_done += 1
                local_done += 1
                if not ev_m.valid:
                    self._record_voi(op_name, 0.0, dt_ms)
                    no_impr_streak += 1
                    continue

                sc_m = float(score_fn(ev_m, n_asn, n_bays, n_crews, n_seq))
                gain_local = max(0.0, r_sc - sc_m)
                self._record_voi(op_name, gain_local, dt_ms)

                # Accept locally if improved (hill-climb inside cone)
                if sc_m + 1e-4 < r_sc:
                    r_ev, r_sc = ev_m, sc_m
                    r_asn, r_bays, r_crews, r_prio, r_seq = n_asn, n_bays, n_crews, n_prio, n_seq
                    cone = build_causal_repair_cone(snapshot, graph, r_ev, r_asn, r_bays, r_crews)
                    no_impr_streak = 0
                else:
                    no_impr_streak += 1

                # Update global winner respecting hysteresis vs KEEP
                margin = hysteresis_margin if best_ev is keep_eval else 1e-4
                if sc_m + margin < best_sc:
                    best_ev, best_sc = ev_m, sc_m
                    best_asn, best_bays, best_crews, best_prio, best_seq = n_asn, n_bays, n_crews, n_prio, n_seq
                    best_tag = f"{r_tag}+{op_name}"

                if no_impr_streak >= 4:
                    break

        return best_ev, best_sc, best_asn, best_bays, best_crews, best_prio, best_seq, best_tag, evals_done


# ---------------------------------------------------------------------------
# Step 2 NCSR Primitives: Stability Sets, Zero-Churn Projection,
# Microsecond J_A Surrogate Screening & Causal Cone Local Repair
# ---------------------------------------------------------------------------


def compute_stability_sets(
    snapshot: NativeSnapshot,
    graph: V19HeteroGraph,
) -> tuple[set[str], set[str], dict[str, str]]:
    """Partition modifiable entities into fresh/pending vs. healthy-anchored stability fences."""
    tasks_by_id = {t.task_id: t for t in snapshot.tasks}
    healthy_set = {graph.agv_ids[i] for i in graph.healthy_agv_indices}
    is_opening = int(snapshot.tick) <= 4

    fresh_op_ids: set[str] = set()
    for op in snapshot.operations:
        if op.op_id in graph.frozen_ops or op.started:
            continue
        vis_kits = [
            tasks_by_id[kt]
            for kt in op.kit_task_ids
            if kt in tasks_by_id and tasks_by_id[kt].status.lower() != "completed"
        ]
        if not vis_kits:
            if is_opening:
                fresh_op_ids.add(op.op_id)
            continue
        if is_opening or any(
            (
                t.assigned_agv is None
                or t.status.lower() == "pending"
                or t.assigned_agv in graph.clf.disturbed_robots
                or t.assigned_agv not in healthy_set
            )
            for t in vis_kits
        ):
            fresh_op_ids.add(op.op_id)

    pending_or_dist_set: set[str] = set()
    for tid in graph.rel_tids:
        t = tasks_by_id.get(tid)
        if t is None:
            continue
        if (
            t.assigned_agv is None
            or t.status.lower() == "pending"
            or t.assigned_agv in graph.clf.disturbed_robots
            or t.assigned_agv not in healthy_set
        ):
            pending_or_dist_set.add(tid)

    healthy_anchor_assign = {
        tid: aid
        for tid, aid in graph.base_assign.items()
        if tid not in pending_or_dist_set
    }
    return fresh_op_ids, pending_or_dist_set, healthy_anchor_assign


def adaptive_stability_margin(
    snapshot: NativeSnapshot,
    graph: V19HeteroGraph,
    cand_assign: dict[str, str],
    cand_bays: dict[str, str],
    cand_crews: dict[str, int],
    fresh_op_ids: set[str],
    *,
    tardiness_gain: float = 0.0,
    for_surrogate: bool = False,
) -> float:
    """Compute non-rescue stability margin preventing mid-episode churn while freeing pending dispatches."""
    if int(snapshot.tick) <= 4:
        return 2.0
    tasks_by_id = {t.task_id: t for t in snapshot.tasks}
    n_non_rescue = 0
    for tid, new_agv in cand_assign.items():
        t_orig = tasks_by_id.get(tid)
        if (
            t_orig is not None
            and t_orig.status.lower() == "assigned"
            and t_orig.assigned_agv is not None
            and t_orig.assigned_agv != new_agv
            and t_orig.assigned_agv not in graph.clf.disturbed_robots
        ):
            n_non_rescue += 1

    n_stale_op_flips = sum(
        1
        for oid in graph.op_ids
        if oid not in fresh_op_ids
        and (
            cand_bays.get(oid, graph.base_bays.get(oid)) != graph.base_bays.get(oid)
            or cand_crews.get(oid, graph.base_crews.get(oid)) != graph.base_crews.get(oid)
        )
    )
    if n_non_rescue == 0 and n_stale_op_flips == 0:
        return 2.0 if for_surrogate else 4.0

    raw_req = 8.0 + 12.0 * float(n_non_rescue) + 16.0 * float(n_stale_op_flips)
    if for_surrogate:
        return 0.5 * raw_req
    scale = max(0.25, 1.0 - min(0.75, max(0.0, float(tardiness_gain)) / 80.0))
    return raw_req * scale


def project_zero_churn_candidate(
    snapshot: NativeSnapshot,
    graph: V19HeteroGraph,
    assign: dict[str, str],
    bays: dict[str, str],
    crews: dict[str, int],
    op_prio: dict[str, int],
    task_sequences: Optional[dict[str, list[str]]],
    fresh_op_ids: set[str],
    pending_or_dist_set: set[str],
    rob_cur_healthy: dict[str, str],
) -> tuple[dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]]]:
    """Project any joint candidate onto non-disruptive `pending_or_dist` + `fresh_op_ids` moves."""
    tasks_by_id = {t.task_id: t for t in snapshot.tasks}
    zc_assign = dict(graph.base_assign)
    for tid in pending_or_dist_set:
        if tid in assign:
            zc_assign[tid] = assign[tid]

    zc_bays = dict(graph.base_bays)
    zc_crews = dict(graph.base_crews)
    for oid in fresh_op_ids:
        if oid in bays:
            zc_bays[oid] = bays[oid]
        if oid in crews:
            zc_crews[oid] = crews[oid]

    zc_seq: Optional[dict[str, list[str]]] = None
    if task_sequences:
        # Rebuild exact per-robot queues aligned with zc_assign while preserving healthy heads
        orig_rank: dict[str, int] = {}
        rank_i = 0
        for aid in sorted(task_sequences):
            for tid in task_sequences[aid]:
                orig_rank[tid] = rank_i
                rank_i += 1
        by_agv: dict[str, list[str]] = {}
        for tid in graph.rel_tids:
            aid = zc_assign.get(tid)
            if aid:
                by_agv.setdefault(aid, []).append(tid)
        zc_seq = {}
        for aid, tids in by_agv.items():
            head = rob_cur_healthy.get(aid)
            tids_sorted = sorted(
                tids,
                key=lambda tid: (
                    0 if tid == head else 1,
                    orig_rank.get(tid, 9999),
                    int(tasks_by_id[tid].due_step or 9999) if tid in tasks_by_id else 9999,
                    tid,
                ),
            )
            if tids_sorted:
                zc_seq[aid] = tids_sorted

    return zc_assign, zc_bays, zc_crews, dict(op_prio), zc_seq


class FastJAWKittingSurrogate:
    """Microsecond zero-Rust analytical evaluator computing J_A-aligned proxy scores."""

    _STATIC_TT_CACHE: dict[tuple[int, int, int, float], dict[str, dict[str, float]]] = {}

    def __init__(
        self,
        network: Network,
        snapshot: NativeSnapshot,
        evaluator: CVCREvaluator,
        graph: V19HeteroGraph,
        *,
        lambda_change: float = 15.0,
    ) -> None:
        self.network = network
        self.snapshot = snapshot
        self.evaluator = evaluator
        self.graph = graph
        self.lambda_change = float(lambda_change)
        self.max_speed = float(getattr(evaluator, "max_speed", 3.0))
        self.tasks = list(snapshot.tasks)
        self.tasks_by_id = {t.task_id: t for t in self.tasks}
        self.robots_by_id = {r.robot_id: r for r in snapshot.robots}
        self.rob_cur_healthy = {
            r.robot_id: r.cur_task
            for r in snapshot.robots
            if r.cur_task and r.robot_id not in graph.clf.disturbed_robots
        }

        if graph is not None and len(graph.bay_ids) == len(graph.bay_detour_norm):
            self.bay_delta = {
                graph.bay_ids[i]: float(graph.bay_detour_norm[i]) * 15.0
                for i in range(len(graph.bay_ids))
            }
        else:
            dd_sig = compute_detour_delta_signals(network, snapshot, simulator=None)
            self.bay_delta = dict(dd_sig["bay_detour_delta"])
        self.bay_delays: dict[str, float] = {}
        for b, d in self.bay_delta.items():
            if d > 0.5:
                self.bay_delays[b] = min(6.0, 0.25 * float(d))

        c_key = (id(network), len(network.nodes), len(network.edges), round(self.max_speed, 4))
        tt_tbl = self.__class__._STATIC_TT_CACHE.get(c_key)
        if tt_tbl is None:
            if len(self.__class__._STATIC_TT_CACHE) >= 16:
                self.__class__._STATIC_TT_CACHE.clear()
            tt_tbl = {}
            self.__class__._STATIC_TT_CACHE[c_key] = tt_tbl
        self._sp_cache = tt_tbl
        self.fb = getattr(snapshot, "execution_feedback", None)

    def tt(self, src: str, dst: str) -> float:
        if src == dst:
            return 0.0
        row = self._sp_cache.get(src)
        if row is None:
            row = self.network.shortest_travel_times(src, self.max_speed) if src in self.network.nodes else {}
            self._sp_cache[src] = row
        val = row.get(dst)
        if val is not None:
            return float(val)
        ns, nd = self.network.nodes.get(src), self.network.nodes.get(dst)
        if ns is not None and nd is not None:
            return float(math.hypot(float(ns.x) - float(nd.x), float(ns.y) - float(nd.y)))
        return 60.0

    def bay_delay(self, node_id: str) -> float:
        return float(self.bay_delays.get(node_id, 0.0))

    def predict_unloads(
        self,
        assign: dict[str, str],
        op_prio: dict[str, int],
        op_bays: dict[str, str],
        task_sequences: Optional[dict[str, list[str]]] = None,
    ) -> dict[str, int]:
        unloads: dict[str, int] = {}
        by_r: dict[str, list[Any]] = {}
        tick = int(self.snapshot.tick)
        for t in self.tasks:
            st = t.status.lower()
            if st == "completed":
                unloads[t.task_id] = int(t.complete_step or tick)
            else:
                rid = assign.get(t.task_id)
                if rid:
                    by_r.setdefault(rid, []).append(t)

        for rid, tlist in by_r.items():
            rob = self.robots_by_id.get(rid)
            if rob is None:
                continue
            fb_off = float(self.fb.robot_ready_offset_steps(rid)) if self.fb is not None else 0.0
            cur_n = rob.cur_node
            cur_t = float(max(tick, rob.faulted_until)) + fb_off
            seq_r = task_sequences.get(rid) if task_sequences else None
            if seq_r:
                pos_map = {tid: i for i, tid in enumerate(seq_r)}
                tlist.sort(
                    key=lambda x: (
                        0 if x.status.lower() in ("picked", "loading", "to_dropoff", "unloading") else 1,
                        pos_map.get(x.task_id, 9999),
                        op_prio.get(x.kit_op_id or "", 0),
                        x.release_step,
                        x.task_id,
                    )
                )
            else:
                tlist.sort(
                    key=lambda x: (
                        0 if x.status.lower() in ("picked", "loading", "to_dropoff", "unloading") else 1,
                        op_prio.get(x.kit_op_id or "", 0),
                        x.release_step,
                        x.task_id,
                    )
                )
            cong = 1.0 + 0.045 * max(0, len(tlist) - 2)
            for idx_t, t in enumerate(tlist):
                st = t.status.lower()
                drop = (
                    op_bays.get(t.kit_op_id, t.dropoff)
                    if (t.kit_op_id and st in ("pending", "assigned"))
                    else t.dropoff
                )
                if st in ("picked", "loading", "to_dropoff", "unloading"):
                    tr2 = cong * self.tt(cur_n, drop)
                    if self.fb is not None and idx_t == 0:
                        tr2 += float(self.fb.robot_leg_step_buffer(rid, int(round(tr2))))
                    arr = cur_t + tr2 + self.bay_delay(drop) + 2.0
                else:
                    tr1 = cong * self.tt(cur_n, t.pickup)
                    if self.fb is not None and idx_t == 0:
                        tr1 += float(self.fb.robot_leg_step_buffer(rid, int(round(tr1))))
                    tp = max(cur_t + tr1, float(t.release_step)) + 2.0
                    tr2 = cong * self.tt(t.pickup, drop)
                    arr = tp + tr2 + self.bay_delay(drop) + 2.0
                unloads[t.task_id] = int(round(arr + 0.499))
                cur_t = arr
                cur_n = drop
        return unloads

    def score(
        self,
        assign: dict[str, str],
        op_prio: dict[str, int],
        op_bays: dict[str, str],
        op_crews: dict[str, int],
        task_sequences: Optional[dict[str, list[str]]] = None,
    ) -> float:
        unloads = self.predict_unloads(assign, op_prio, op_bays, task_sequences)
        op_s, op_e, op_rr, _, _, starv = self.evaluator._propagate_production(
            unloads, op_prio, op_bays, op_crews
        )
        dummy_ev = EvalResult(
            robot_paths={},
            robot_order_seq={},
            order_unload_ticks=unloads,
            op_start=op_s,
            op_end=op_e,
            op_res_ready=op_rr,
            objective=0.0,
            valid=True,
            starvation_ticks=starv,
            eval_latency_ms=0.0,
            mode="SURR",
        )
        return score_candidate_ja_aligned(
            self.snapshot,
            self.graph,
            dummy_ev,
            assign,
            op_bays,
            op_crews,
            task_sequences,
            self.rob_cur_healthy,
            lambda_change=self.lambda_change,
            sim_horizon=self.evaluator.sim_horizon,
        )

    def regret_k_assignment(
        self,
        healthy_robots: list[str],
        fixed_assign: dict[str, str],
        tasks_to_insert: list[str],
        op_bays: dict[str, str],
        op_prio: dict[str, int],
    ) -> dict[str, str]:
        """Fast Potvin-Rousseau Regret-2 insertion with op-due criticality and cached shortest-path times."""
        assign = dict(fixed_assign)
        if not healthy_robots or not tasks_to_insert:
            return assign
        op_map = {op.op_id: op for op in self.snapshot.operations}
        rob_node: dict[str, str] = {}
        rob_free: dict[str, float] = {}
        rob_loads: dict[str, int] = {}
        by_r: dict[str, list[Any]] = {r: [] for r in healthy_robots}
        for rid in healthy_robots:
            rob = self.robots_by_id[rid]
            rob_node[rid] = rob.cur_node
            fb_o = float(self.fb.robot_ready_offset_steps(rid)) if self.fb is not None else 0.0
            rob_free[rid] = float(max(self.snapshot.tick, rob.faulted_until)) + fb_o
            rob_loads[rid] = 0
        for tid, rid in assign.items():
            if rid in by_r and tid in self.tasks_by_id:
                t = self.tasks_by_id[tid]
                if t.status.lower() != "completed":
                    by_r[rid].append(t)
        for rid, tlist in by_r.items():
            tlist.sort(
                key=lambda x: (
                    0 if x.status.lower() in ("picked", "loading", "to_dropoff", "unloading") else 1,
                    op_prio.get(x.kit_op_id or "", 0),
                    x.release_step,
                    x.task_id,
                )
            )
            for t in tlist:
                drop = (
                    op_bays.get(t.kit_op_id, t.dropoff)
                    if (op_bays and t.kit_op_id and t.status.lower() in ("pending", "assigned"))
                    else t.dropoff
                )
                if t.status.lower() in ("picked", "loading", "to_dropoff", "unloading"):
                    arr = rob_free[rid] + self.tt(rob_node[rid], drop) + self.bay_delay(drop) + 2.0
                else:
                    tp = max(rob_free[rid] + self.tt(rob_node[rid], t.pickup), float(t.release_step)) + 2.0
                    arr = tp + self.tt(t.pickup, drop) + self.bay_delay(drop) + 2.0
                rob_free[rid] = arr
                rob_node[rid] = drop
                rob_loads[rid] += 1

        agv_idx_map = {aid: i for i, aid in enumerate(self.graph.agv_ids)}
        rel_idx_map = {tid: i for i, tid in enumerate(self.graph.rel_tids)}
        rem = [
            tid
            for tid in tasks_to_insert
            if tid in self.tasks_by_id and self.tasks_by_id[tid].status.lower() != "completed"
        ]
        while rem:
            best = None
            for tid in rem:
                t = self.tasks_by_id[tid]
                drop = op_bays.get(t.kit_op_id, t.dropoff) if (op_bays and t.kit_op_id) else t.dropoff
                due_eff = float(t.due_step if t.due_step is not None else 999)
                if t.kit_op_id and t.kit_op_id in op_map:
                    op = op_map[t.kit_op_id]
                    due_eff = min(due_eff, float(op.due_step - op.proc_steps)) - (15.0 if op.is_urgent else 0.0)
                ti = rel_idx_map.get(tid)
                costs: list[tuple[float, str, float]] = []
                for rid in healthy_robots:
                    if ti is not None and not bool(self.graph.task_agv_mask[ti, agv_idx_map[rid]]):
                        continue
                    tp = max(rob_free[rid] + self.tt(rob_node[rid], t.pickup), float(t.release_step)) + 2.0
                    comp = tp + self.tt(t.pickup, drop) + self.bay_delay(drop) + 2.0
                    tard = 3.0 * max(0.0, comp - due_eff)
                    chg = (
                        15.0
                        if (t.assigned_agv and t.assigned_agv != rid and t.status.lower() == "assigned")
                        else 0.0
                    )
                    costs.append((comp + tard + chg + 1.5 * rob_loads[rid], rid, comp))
                if not costs:
                    r0 = healthy_robots[0]
                    costs = [(rob_free[r0] + 60.0, r0, rob_free[r0] + 60.0)]
                costs.sort(key=lambda x: (x[0], x[1]))
                c1, r1, cmp1 = costs[0]
                c2 = costs[1][0] if len(costs) >= 2 else c1 + 50.0
                urg = 40.0 if (t.is_urgent or t.assigned_agv not in healthy_robots) else 0.0
                key = ((c2 - c1) + urg - 0.15 * due_eff, -c1, tid)
                if best is None or key > best[0]:
                    best = (key, tid, r1, drop, cmp1)
            assert best is not None
            _, ptid, prid, pdrop, pcomp = best
            rem.remove(ptid)
            assign[ptid] = prid
            rob_free[prid] = pcomp
            rob_node[prid] = pdrop
            rob_loads[prid] += 1
        return assign


def _compute_fresh_semicircle_bays(
    snapshot: NativeSnapshot,
    network: Network,
    graph: V19HeteroGraph,
    fast_surr: FastJAWKittingSurrogate,
    fresh_op_ids: set[str],
) -> tuple[dict[str, str], dict[str, int], dict[str, int]]:
    """Co-locate `fresh_op_ids` bays via semicircle proximity + Detour-Delta & polish crews on surrogate."""
    bays = dict(graph.base_bays)
    crews = dict(graph.base_crews)
    km = compute_kitting_marginal_start_value(snapshot)
    tmap = fast_surr.tasks_by_id
    bay_load: dict[str, int] = {}
    is_opening = int(snapshot.tick) <= 4
    ops_sorted = sorted(
        snapshot.operations,
        key=lambda o: (
            0 if o.started else 1,
            0 if o.op_id not in fresh_op_ids else 1,
            0 if o.is_urgent else 1,
            int(o.due_step),
            -len(o.kit_task_ids),
            o.op_id,
        ),
    )
    km_prio: dict[str, int] = {}
    for op in ops_sorted:
        k_min = min(
            (km[kt][0] * 1000 + km[kt][4] for kt in op.kit_task_ids if kt in km),
            default=int(op.due_step),
        )
        km_prio[op.op_id] = max(int(snapshot.tick), int(min(k_min, op.due_step - op.proc_steps)))
        if (
            op.started
            or op.op_id in graph.frozen_ops
            or op.op_id not in fresh_op_ids
            or len(op.eligible_bays) <= 1
        ):
            cur_b = bays.get(op.op_id, op.bay_id)
            bay_load[cur_b] = bay_load.get(cur_b, 0) + 1
            continue
        kits = [tmap[tid] for tid in op.kit_task_ids if tid in tmap and tmap[tid].status.lower() != "completed"]
        if not kits and not is_opening:
            cur_b = bays.get(op.op_id, op.bay_id)
            bay_load[cur_b] = bay_load.get(cur_b, 0) + 1
            continue
        pm = (1.5 if op.is_urgent else 1.0) + 0.25 * len(kits)
        has_win = bool(getattr(op, "resource_windows", ()))
        best_b = op.bay_id
        best_cost = float("inf")
        for cand_b in op.eligible_bays:
            if cand_b not in network.nodes:
                continue
            d_sum = sum(float(fast_surr.tt(t.pickup, cand_b)) for t in kits) if kits else 0.0
            ld = bay_load.get(cand_b, 0)
            sw_est = 0.8 * ld + 0.5 * max(0, len(kits) - 1) + (1.2 if has_win else 0.0)
            sw_w = 1.0 / (1.0 + math.exp(-0.3 * sw_est))
            cong_pen = 6.0 * ld + 2.5 * sw_w
            dd_pen = 1.0 * float(fast_surr.bay_delta.get(cand_b, 0.0))
            sc = pm * d_sum + cong_pen + dd_pen
            if sc < best_cost:
                best_cost = sc
                best_b = cand_b
        bays[op.op_id] = best_b
        bay_load[best_b] = bay_load.get(best_b, 0) + 1

    # Quick greedy crew balance on fresh ops using fast_surr
    if any(oid in fresh_op_ids and len(o.eligible_crews) > 1 for o in snapshot.operations for oid in (o.op_id,)):
        best_sc = fast_surr.score(graph.base_assign, km_prio, bays, crews)
        for op in snapshot.operations:
            if op.started or op.op_id not in fresh_op_ids or len(op.eligible_crews) <= 1:
                continue
            cur_c = int(crews.get(op.op_id, op.crew_id))
            for cand_c in op.eligible_crews:
                if int(cand_c) == cur_c:
                    continue
                crews[op.op_id] = int(cand_c)
                sc = fast_surr.score(graph.base_assign, km_prio, bays, crews)
                if sc + 1e-3 < best_sc:
                    best_sc = sc
                    cur_c = int(cand_c)
                else:
                    crews[op.op_id] = cur_c
    return bays, crews, km_prio


def _regret2_insert_tasks(
    snapshot: NativeSnapshot,
    graph: V19HeteroGraph,
    fast_surr: FastJAWKittingSurrogate,
    anchor_assign: dict[str, str],
    insert_tids: list[str],
    op_prio: dict[str, int],
    bays: dict[str, str],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Potvin-Rousseau Regret-2 insertion over `insert_tids` holding `anchor_assign` fixed."""
    assign = dict(anchor_assign)
    healthy_agvs = [graph.agv_ids[i] for i in graph.healthy_agv_indices]
    tmap = fast_surr.tasks_by_id
    rob_map = fast_surr.robots_by_id
    tick = int(snapshot.tick)

    if not insert_tids or not healthy_agvs:
        return assign, {}

    # Seed per-AMR busy state from anchor_assign
    end_t: dict[str, float] = {}
    end_node: dict[str, str] = {}
    q_cnt: dict[str, int] = {}
    for aid in healthy_agvs:
        rob = rob_map[aid]
        t_cur = float(max(tick, rob.faulted_until))
        if fast_surr.fb is not None:
            t_cur += float(fast_surr.fb.robot_ready_offset_steps(aid))
        n_cur = rob.cur_node
        fixed_for_r = [
            tmap[tid]
            for tid, r_id in assign.items()
            if r_id == aid and tid in tmap and tmap[tid].status.lower() != "completed"
        ]
        fixed_for_r.sort(
            key=lambda x: (
                0 if x.status.lower() in ("picked", "loading", "to_dropoff", "unloading") else 1,
                op_prio.get(x.kit_op_id or "", 0),
                x.release_step,
                x.task_id,
            )
        )
        for ft in fixed_for_r:
            drop = bays.get(ft.kit_op_id, ft.dropoff) if (ft.kit_op_id and ft.status.lower() in ("pending", "assigned")) else ft.dropoff
            if ft.status.lower() in ("picked", "loading", "to_dropoff", "unloading"):
                t_cur = t_cur + fast_surr.tt(n_cur, drop) + fast_surr.bay_delay(drop) + 2.0
            else:
                tp = max(t_cur + fast_surr.tt(n_cur, ft.pickup), float(ft.release_step)) + 2.0
                t_cur = tp + fast_surr.tt(ft.pickup, drop) + fast_surr.bay_delay(drop) + 2.0
            n_cur = drop
        end_t[aid] = t_cur
        end_node[aid] = n_cur
        q_cnt[aid] = len(fixed_for_r)

    agv_idx_map = {aid: i for i, aid in enumerate(graph.agv_ids)}
    rel_idx_map = {tid: i for i, tid in enumerate(graph.rel_tids)}
    unassigned = [tid for tid in insert_tids if tid in tmap]
    while unassigned:
        best_pick: Optional[tuple[float, float, str, str, float, str]] = None
        for tid in unassigned:
            t = tmap[tid]
            drop = bays.get(t.kit_op_id, t.dropoff) if t.kit_op_id else t.dropoff
            ti = rel_idx_map.get(tid)
            feas_costs: list[tuple[float, str, float]] = []
            for aid in healthy_agvs:
                if ti is not None and not bool(graph.task_agv_mask[ti, agv_idx_map[aid]]):
                    continue
                cong = 1.0 + 0.045 * max(0, q_cnt[aid] - 1)
                tp = max(end_t[aid] + cong * fast_surr.tt(end_node[aid], t.pickup), float(t.release_step)) + 2.0
                finish = tp + cong * fast_surr.tt(t.pickup, drop) + fast_surr.bay_delay(drop) + 2.0
                # Tardiness-aware insertion cost
                due = float(t.due_step if t.due_step is not None else (tick + 120))
                tard_cost = 2.0 * max(0.0, finish - due)
                cost = finish + tard_cost
                feas_costs.append((cost, aid, finish))
            if not feas_costs:
                aid0 = healthy_agvs[0]
                feas_costs = [(end_t[aid0] + 60.0, aid0, end_t[aid0] + 60.0)]
            feas_costs.sort(key=lambda x: (x[0], x[1]))
            c1, a1, f1 = feas_costs[0]
            regret = (feas_costs[1][0] - c1) if len(feas_costs) >= 2 else 0.0
            if getattr(t, "is_urgent", False):
                regret += 25.0
            key_p = (regret, -c1, tid)
            if best_pick is None or key_p > (best_pick[0], best_pick[1], best_pick[2]):
                best_pick = (regret, -c1, tid, a1, f1, drop)

        assert best_pick is not None
        _, _, pick_tid, pick_aid, pick_finish, pick_drop = best_pick
        assign[pick_tid] = pick_aid
        end_t[pick_aid] = pick_finish
        end_node[pick_aid] = pick_drop
        q_cnt[pick_aid] += 1
        unassigned.remove(pick_tid)

    # Order per-AMR sequence cleanly respecting active healthy head
    seqs: dict[str, list[str]] = {}
    for aid in healthy_agvs:
        head = fast_surr.rob_cur_healthy.get(aid)
        a_tids = [tid for tid in graph.rel_tids if assign.get(tid) == aid]
        if a_tids:
            a_tids.sort(
                key=lambda tid: (
                    0 if tid == head else 1,
                    op_prio.get(tmap[tid].kit_op_id or "", 0),
                    int(tmap[tid].release_step),
                    tid,
                )
            )
            seqs[aid] = a_tids
    return assign, seqs


def generate_causal_cone_candidates(
    snapshot: NativeSnapshot,
    network: Network,
    graph: V19HeteroGraph,
    ev: CVCREvaluator,
    fast_surr: FastJAWKittingSurrogate,
    keep_eval: EvalResult,
    base_op_prio: dict[str, int],
    neural_pool: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]], bool]],
    fresh_op_ids: set[str],
    pending_or_dist_set: set[str],
    *,
    budget_ms_limit: float = 0.7,
) -> list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]], bool]]:
    """Generate high-leverage Causal Cone + Regret-2 + Polished candidates inside <0.7ms."""
    t_start = _cvcr_now_s()
    out: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]], bool]] = []
    if not graph.rel_tids or len(graph.healthy_agv_indices) < 1:
        return out

    healthy_agvs = [graph.agv_ids[i] for i in graph.healthy_agv_indices]
    agv_idx_map = {aid: i for i, aid in enumerate(graph.agv_ids)}
    task_idx_map = {tid: i for i, tid in enumerate(graph.rel_tids)}
    rob_cur_h = fast_surr.rob_cur_healthy

    bays_semi, crews_semi, km_prio = _compute_fresh_semicircle_bays(
        snapshot, network, graph, fast_surr, fresh_op_ids
    )
    cone = build_causal_repair_cone(
        snapshot, graph, keep_eval, graph.base_assign, graph.base_bays, graph.base_crews
    )

    # 1. Regret-2 Zero-Churn (`cone_regret2_zc`): keeps healthy_anchor fixed, only pending_or_dist + fresh_op_ids
    if pending_or_dist_set:
        anchor_zc = {tid: aid for tid, aid in graph.base_assign.items() if tid not in pending_or_dist_set}
        r2_asn_zc, r2_seq_zc = _regret2_insert_tasks(
            snapshot, graph, fast_surr, anchor_zc, sorted(pending_or_dist_set), base_op_prio, bays_semi
        )
        out.append(("cone_regret2_zc", r2_asn_zc, bays_semi, crews_semi, base_op_prio, r2_seq_zc, False))

    # 2. Causal-Cone + Pending Regret-2 Destroy-and-Repair (`cone_regret2_full`)
    destroy_set = set(pending_or_dist_set) | set(cone.cone_tasks)
    if destroy_set and len(healthy_agvs) >= 2:
        anchor_cone = {tid: aid for tid, aid in graph.base_assign.items() if tid not in destroy_set}
        r2_asn_c, r2_seq_c = _regret2_insert_tasks(
            snapshot, graph, fast_surr, anchor_cone, sorted(destroy_set), km_prio, bays_semi
        )
        out.append(("cone_regret2_full", r2_asn_c, bays_semi, crews_semi, km_prio, r2_seq_c, True))

    # 3. Bottleneck Kitting Parallel Co-Alignment (`cone_kit_coalign`)
    rel_set = set(graph.rel_tids)
    kits = [kt for kt in cone.bottleneck_kit_group if kt in rel_set]
    if len(kits) >= 2 and len(healthy_agvs) >= 2:
        ka_asn = dict(graph.base_assign)
        used_r: set[str] = set()
        changed_ka = False
        for kt in kits:
            ti = task_idx_map.get(kt)
            if ti is None:
                continue
            feas = [
                aid for aid in healthy_agvs
                if bool(graph.task_agv_mask[ti, agv_idx_map[aid]])
            ]
            if not feas:
                continue
            feas.sort(key=lambda aid: (1 if aid in used_r else 0, float(graph.task_agv_feats[ti, agv_idx_map[aid], 0]), aid))
            pick_r = feas[0]
            used_r.add(pick_r)
            if ka_asn.get(kt) != pick_r:
                ka_asn[kt] = pick_r
                changed_ka = True
        if changed_ka:
            ka_seq = {aid: [tid for tid in graph.rel_tids if ka_asn.get(tid) == aid] for aid in healthy_agvs}
            ka_seq = {k: v for k, v in ka_seq.items() if v}
            out.append(("cone_kit_coalign", ka_asn, dict(graph.base_bays), dict(graph.base_crews), dict(base_op_prio), ka_seq, True))

    # 4. Critical Queue-Head Promotion (`cone_queue_promote`) on keep_eval sequences
    if keep_eval.robot_order_seq and cone.cone_tasks:
        t0_q = cone.cone_tasks[0]
        rid_q = graph.base_assign.get(t0_q)
        if rid_q and rid_q in keep_eval.robot_order_seq:
            seq_q = [x for x in keep_eval.robot_order_seq[rid_q] if x in rel_set or x == rob_cur_h.get(rid_q)]
            if t0_q in seq_q:
                pos = seq_q.index(t0_q)
                tgt = 1 if (seq_q and seq_q[0] == rob_cur_h.get(rid_q)) else 0
                if pos > tgt:
                    new_seq_r = list(seq_q)
                    new_seq_r.pop(pos)
                    new_seq_r.insert(tgt, t0_q)
                    seqs_qp = {k: list(v) for k, v in keep_eval.robot_order_seq.items() if v}
                    seqs_qp[rid_q] = new_seq_r
                    prio_qp = dict(base_op_prio)
                    t_obj = fast_surr.tasks_by_id.get(t0_q)
                    if t_obj and t_obj.kit_op_id and t_obj.kit_op_id in prio_qp:
                        prio_qp[t_obj.kit_op_id] = max(int(snapshot.tick), prio_qp[t_obj.kit_op_id] - 15)
                    out.append(("cone_queue_promote", dict(graph.base_assign), dict(graph.base_bays), dict(graph.base_crews), prio_qp, seqs_qp, False))

    # 5. Surrogate 1-Pass Causal Cone Polish on Top Seed(s) (neural + ZC)
    if (_cvcr_now_s() - t_start) * 1000.0 < budget_ms_limit and len(healthy_agvs) >= 2:
        seeds_to_polish: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]], bool]] = []
        if neural_pool:
            scored_np = [
                (fast_surr.score(asn, op_p, bs, cs, sq), tag, asn, bs, cs, op_p, sq, is_f)
                for tag, asn, bs, cs, op_p, sq, is_f in neural_pool
            ]
            scored_np.sort(key=lambda item: item[0])
            top0 = scored_np[0]
            seeds_to_polish.append((top0[1], top0[2], top0[3], top0[4], top0[5], top0[6], top0[7]))
            for item in scored_np[1:]:
                if item[1].endswith("_zc") or item[1].startswith("zc"):
                    seeds_to_polish.append((item[1], item[2], item[3], item[4], item[5], item[6], item[7]))
                    break
        elif out:
            seeds_to_polish.append(out[0])

        focus_tasks = cone.cone_tasks[:6] if cone.cone_tasks else graph.rel_tids[:6]
        for s_tag, s_asn, s_bs, s_cs, s_op, s_sq, s_full in seeds_to_polish:
            if (_cvcr_now_s() - t_start) * 1000.0 >= budget_ms_limit:
                break
            cur_a = dict(s_asn)
            cur_b = dict(s_bs)
            cur_c = dict(s_cs)
            cur_sq = {k: list(v) for k, v in s_sq.items()} if s_sq else None
            best_sc = fast_surr.score(cur_a, s_op, cur_b, cur_c, cur_sq)
            changed = False
            restrict_pending = s_tag.endswith("_zc")
            for tid in focus_tasks:
                if restrict_pending and tid not in pending_or_dist_set:
                    continue
                ti = task_idx_map.get(tid)
                if ti is None:
                    continue
                orig_r = cur_a.get(tid)
                best_r = orig_r
                for aid in healthy_agvs:
                    if aid == orig_r or not bool(graph.task_agv_mask[ti, agv_idx_map[aid]]):
                        continue
                    cur_a[tid] = aid
                    # Update sequence minimally for accurate queue position simulation
                    trial_sq = None
                    if cur_sq is not None:
                        trial_sq = {k: list(v) for k, v in cur_sq.items()}
                        if orig_r in trial_sq and tid in trial_sq[orig_r]:
                            trial_sq[orig_r].remove(tid)
                        trial_sq.setdefault(aid, []).append(tid)
                    sc = fast_surr.score(cur_a, s_op, cur_b, cur_c, trial_sq)
                    if sc + 1.0 < best_sc:
                        best_sc = sc
                        best_r = aid
                        changed = True
                if best_r is not None:
                    if cur_sq is not None and best_r != orig_r:
                        if orig_r in cur_sq and tid in cur_sq[orig_r]:
                            cur_sq[orig_r].remove(tid)
                        cur_sq.setdefault(best_r, []).append(tid)
                    cur_a[tid] = best_r
            if changed:
                out.append((f"{s_tag}+cone", cur_a, cur_b, cur_c, dict(s_op), cur_sq, s_full))

    return out
