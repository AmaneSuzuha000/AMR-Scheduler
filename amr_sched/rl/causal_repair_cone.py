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
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.snapshot import NativeSnapshot, op_predecessors
from amr_sched.rl.v19_hetero_graph import V19HeteroGraph


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
) -> float:
    """Compute exact J_A-aligned proxy score from `EvalResult` (`C_max + 2*T_task + 2*T_op + 15*churn + unf`)."""
    if not ev_res.valid:
        return float(ev_res.objective) + 20_000.0

    # 1. Exact makespan C_max across operations and transport tasks
    op_ends = ev_res.op_end
    unloads = ev_res.order_unload_ticks
    cmax_vals = [int(snapshot.tick)]
    if op_ends:
        cmax_vals.extend(int(v) for v in op_ends.values())
    if unloads:
        cmax_vals.extend(int(v) for v in unloads.values())
    cmax = max(cmax_vals)

    # 2. Exact 2 * T_task + unfinished task proxy
    t_task_cost = 0.0
    unf_pen = 0.0
    for t in snapshot.tasks:
        if t.status.lower() == "completed":
            continue
        u_t = unloads.get(t.task_id)
        if u_t is None or int(u_t) >= sim_horizon:
            unf_pen += 5000.0
            u_val = sim_horizon
        else:
            u_val = int(u_t)
        if t.due_step is not None:
            t_task_cost += 2.0 * float(max(0, u_val - int(t.due_step)))

    # 3. Exact 2 * T_op + unfinished op proxy (plus gentle urgent tie-break)
    t_op_cost = 0.0
    for op in snapshot.operations:
        e_o = op_ends.get(op.op_id)
        if e_o is None or int(e_o) >= sim_horizon:
            unf_pen += 3000.0
            e_val = sim_horizon
        else:
            e_val = int(e_o)
        tard_o = max(0, e_val - int(op.due_step))
        t_op_cost += (2.5 if op.is_urgent else 2.0) * float(tard_o)

    # 4. Exact soft-commitment churn (task reassignments + bay/crew changes + healthy head preempts)
    chg_tasks = sum(
        1
        for tid in graph.rel_tids
        if assign.get(tid) != graph.base_assign.get(tid)
        and graph.tasks_by_id_status(tid) == "assigned"
    ) if hasattr(graph, "tasks_by_id_status") else sum(
        1
        for t in snapshot.tasks
        if t.status.lower() == "assigned"
        and t.assigned_agv is not None
        and assign.get(t.task_id, t.assigned_agv) != t.assigned_agv
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

    # Tiny tie-breaking on starvation/delivery (0.01 weight so H1 + churn strictly dominates)
    tie_break = 0.01 * float(ev_res.starvation_ticks)
    return float(cmax) + t_task_cost + t_op_cost + unf_pen + float(lambda_change) * float(chg_tasks + chg_ops + qh_preempts) + tie_break


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
