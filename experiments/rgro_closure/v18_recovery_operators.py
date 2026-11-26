"""Structured `V18` Quality Recovery Operators (`IF-CORE-RGRO-CLOSURE-01` Task 3).

Explicitly FORBIDS the falsified unconstrained `U2` mid-episode task-swap operator.
Decomposes V18's typical/static combinatorial strengths into three clean, toggleable operators:
  - `O1` Opening Combinatorial Constructor: fires ONLY when zero execution commitments exist
    (`tick <= 4` and zero tasks `picked`/`completed`, zero ops `started`), combining `Wu'26`
    semi-circle bays, fresh crew polish, `PGMPO (C=8)`, `Regret-2`, and `MIT'26` closure.
  - `O2` Assignment-Preserving Queue Refinement: optimizes per-robot `task_orders` (`2-opt`/swap)
    while holding ALL `assigned_agv`, `bays`, and `crews` 100% FIXED -> ZERO assignment churn.
  - `O3` Diverse Low-Churn Residual Polish: fresh-op bay/crew improvement + MIT'26 low-slack
    precedence-closure regret insertion without touching stable healthy assignments.
"""

from __future__ import annotations

from typing import Any, Optional

from amr_sched.cvcr.commitment import CommitmentClassification
from amr_sched.cvcr.evaluator import EvalResult
from amr_sched.cvcr.policy import _LazyKittingSurrogate
from amr_sched.cvcr.snapshot import NativeSnapshot
from amr_sched.cvcr.v18_frontier_synthesis import (
    _mit26_precedence_closure_destroy,
    compute_semicircle_kitting_bays,
)
from amr_sched.network import Network
from amr_sched.rl.causal_repair_cone import FastJAWKittingSurrogate
from amr_sched.rl.v19_hetero_graph import V19HeteroGraph
from amr_sched.rl.v20_najs import _compute_fresh_semicircle_bays, _default_task_order_from_prio

CandTuple = tuple[
    str,
    dict[str, str],
    dict[str, str],
    dict[str, int],
    dict[str, int],
    Optional[dict[str, list[str]]],
    bool,
]


def is_opening_zero_commitment(snapshot: NativeSnapshot) -> bool:
    """Return True iff `tick <= 4` and ZERO physical pickups/completions or started ops exist."""
    if int(snapshot.tick) > 4:
        return False
    for t in snapshot.tasks:
        st = str(getattr(t, "status", "")).lower()
        if st in ("picked", "loading", "to_dropoff", "unloading", "completed"):
            return False
    for op in snapshot.operations:
        if bool(getattr(op, "started", False)) or bool(getattr(op, "completed", False)):
            return False
    return True


def _polish_crews_on_ops(
    snapshot: NativeSnapshot,
    fast_surr: FastJAWKittingSurrogate,
    assign: dict[str, str],
    op_prio: dict[str, int],
    bays: dict[str, str],
    base_crews: dict[str, int],
    eligible_ops: set[str],
) -> dict[str, int]:
    cur_c = dict(base_crews)
    cur_sc = float(fast_surr.score(assign, op_prio, bays, cur_c))
    for op in snapshot.operations:
        if op.started or op.op_id not in eligible_ops or len(op.eligible_crews) <= 1:
            continue
        oc = int(cur_c.get(op.op_id, op.crew_id))
        for xc in op.eligible_crews:
            if int(xc) == oc:
                continue
            cur_c[op.op_id] = int(xc)
            sc = float(fast_surr.score(assign, op_prio, bays, cur_c))
            if sc + 1e-2 < cur_sc:
                cur_sc, oc = sc, int(xc)
            else:
                cur_c[op.op_id] = oc
    return cur_c


def generate_o1_opening_candidates(
    snapshot: NativeSnapshot,
    network: Network,
    clf: CommitmentClassification,
    graph: V19HeteroGraph,
    fast_surr: FastJAWKittingSurrogate,
    base_prio: dict[str, int],
    bay_delta: dict[str, float],
    healthy_agvs: list[str],
    fresh_op_ids: set[str],
    pgmpo_gen: Optional[Any],
    *,
    max_candidates: int = 3,
) -> list[CandTuple]:
    """O1: Un-constrained V18 opening constructor when zero execution commitments exist."""
    if (not is_opening_zero_commitment(snapshot)) or (not graph.rel_tids) or len(healthy_agvs) < 2:
        return []

    lazy_surr = _LazyKittingSurrogate(
        network, snapshot, base_prio, alpha_makespan=3.0, lambda_change=15.0, bay_detour_delta=bay_delta
    )
    bays_semi, km_p = compute_semicircle_kitting_bays(
        snapshot, network, clf, lazy_surr, dict(graph.base_bays), bay_delta,
        eligible_op_ids=fresh_op_ids, spread_empty_kits=True,
    )
    _, _, km_p2 = _compute_fresh_semicircle_bays(snapshot, network, graph, fast_surr, fresh_op_ids)
    eff_prio = km_p2 or km_p or dict(base_prio)
    crews_opt = _polish_crews_on_ops(
        snapshot, fast_surr, graph.base_assign, eff_prio, bays_semi, dict(graph.base_crews), fresh_op_ids
    )

    pool: list[tuple[float, CandTuple]] = []
    rel_all = list(graph.rel_tids)
    fixed_assign = dict(graph.fixed_assign)

    # 1) Full Regret-2 on semi-circle co-located bays & base bays
    reg_semi = fast_surr.regret_k_assignment(healthy_agvs, fixed_assign, rel_all, bays_semi, eff_prio)
    ord_rs = _default_task_order_from_prio(snapshot, graph, reg_semi, eff_prio)
    sc_rs = float(fast_surr.score(reg_semi, eff_prio, bays_semi, crews_opt, ord_rs))
    pool.append((sc_rs, ("o1_regret2_semi", reg_semi, bays_semi, crews_opt, eff_prio, ord_rs, True)))

    if bays_semi != graph.base_bays:
        reg_base = fast_surr.regret_k_assignment(healthy_agvs, fixed_assign, rel_all, dict(graph.base_bays), base_prio)
        ord_rb = _default_task_order_from_prio(snapshot, graph, reg_base, base_prio)
        sc_rb = float(fast_surr.score(reg_base, base_prio, dict(graph.base_bays), dict(graph.base_crews), ord_rb))
        pool.append((sc_rb, ("o1_regret2_base", reg_base, dict(graph.base_bays), dict(graph.base_crews), dict(base_prio), ord_rb, True)))

    # 2) Pure-NumPy PGMPO C=8 policies scored on bays_semi
    if pgmpo_gen is not None and getattr(pgmpo_gen, "ready", False):
        asns = pgmpo_gen.generate_assignments(snapshot, network, clf, healthy_agvs, fixed_assign, bays_semi, lazy_surr)
        for idx_p, a_p in enumerate(asns):
            ord_p = _default_task_order_from_prio(snapshot, graph, a_p, eff_prio)
            sc_p = float(fast_surr.score(a_p, eff_prio, bays_semi, crews_opt, ord_p))
            pool.append((sc_p, (f"o1_pgmpo_semi_{idx_p}", dict(a_p), bays_semi, crews_opt, eff_prio, ord_p, True)))

    # Deduplicate by assignment+bay signature & return top surrogate hypotheses
    pool.sort(key=lambda x: x[0])
    out: list[CandTuple] = []
    seen: set[tuple[Any, ...]] = set()
    for _, c in pool:
        sig = (tuple(sorted(c[1].items())), tuple(sorted(c[2].items())), tuple(sorted(c[3].items())))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
        if len(out) >= int(max_candidates):
            break
    return out


def generate_o2_queue_refinement(
    snapshot: NativeSnapshot,
    graph: V19HeteroGraph,
    fast_surr: FastJAWKittingSurrogate,
    anchor_assign: dict[str, str],
    anchor_bays: dict[str, str],
    anchor_crews: dict[str, int],
    anchor_prio: dict[str, int],
    anchor_eval: Optional[EvalResult],
) -> list[CandTuple]:
    """O2: STRICTLY assignment-preserving per-robot queue refinement (ZERO task reassignment churn!)."""
    if not graph.rel_tids:
        return []

    t_by_id = {t.task_id: t for t in snapshot.tasks}
    healthy = [graph.agv_ids[i] for i in graph.healthy_agv_indices]
    base_ord = _default_task_order_from_prio(snapshot, graph, anchor_assign, anchor_prio)
    if anchor_eval is not None and anchor_eval.robot_order_seq:
        for aid, q in anchor_eval.robot_order_seq.items():
            if aid in healthy and q:
                base_ord[aid] = [tid for tid in q if anchor_assign.get(tid) == aid]

    multi_robots = [aid for aid, q in base_ord.items() if len(q) >= 2]
    if not multi_robots:
        return []

    cur_ord = {aid: list(q) for aid, q in base_ord.items()}
    base_sc = float(fast_surr.score(anchor_assign, anchor_prio, anchor_bays, anchor_crews, cur_ord))
    cur_sc = base_sc
    improved = False

    # Lock active in-transit/picked head task at index 0; swap ONLY trailing pending/assigned queue slots
    for aid in sorted(multi_robots):
        q = list(cur_ord[aid])
        start_idx = 0
        head_t = t_by_id.get(q[0])
        if head_t is not None and str(head_t.status).lower() in ("picked", "loading", "to_dropoff", "unloading"):
            start_idx = 1
        if len(q) - start_idx < 2:
            continue
        for i in range(start_idx, min(len(q), start_idx + 5)):
            for j in range(i + 1, min(len(q), start_idx + 6)):
                q[i], q[j] = q[j], q[i]
                cur_ord[aid] = q
                sc_t = float(fast_surr.score(anchor_assign, anchor_prio, anchor_bays, anchor_crews, cur_ord))
                if sc_t + 1.0 < cur_sc:
                    cur_sc = sc_t
                    improved = True
                else:
                    q[i], q[j] = q[j], q[i]
                    cur_ord[aid] = q

    out: list[CandTuple] = []
    if improved:
        out.append(
            (
                "o2_queue_opt",
                dict(anchor_assign),  # BIT-IDENTICAL assignment!
                dict(anchor_bays),
                dict(anchor_crews),
                dict(anchor_prio),
                cur_ord,
                False,
            )
        )
    return out


def generate_o3_diverse_low_churn(
    snapshot: NativeSnapshot,
    graph: V19HeteroGraph,
    fast_surr: FastJAWKittingSurrogate,
    anchor_assign: dict[str, str],
    anchor_bays: dict[str, str],
    anchor_crews: dict[str, int],
    anchor_prio: dict[str, int],
    anchor_eval: Optional[EvalResult],
    healthy_agvs: list[str],
    fresh_op_ids: set[str],
) -> list[CandTuple]:
    """O3: Diverse low-churn bay/crew refinement on `fresh_op_ids` + MIT'26 low-slack destroy-reinsert."""
    out: list[CandTuple] = []
    if not graph.rel_tids or len(healthy_agvs) < 2:
        return out

    # 1) Low-churn fresh-op bay + crew polish WITHOUT changing any assignment
    cur_b = dict(anchor_bays)
    cur_c = dict(anchor_crews)
    cur_sc = float(fast_surr.score(anchor_assign, anchor_prio, cur_b, cur_c))
    bc_changed = False
    for op in snapshot.operations:
        if op.started or op.op_id not in fresh_op_ids:
            continue
        if len(op.eligible_bays) > 1:
            ob = cur_b.get(op.op_id)
            for xb in op.eligible_bays:
                if xb == ob:
                    continue
                cur_b[op.op_id] = xb
                sc = float(fast_surr.score(anchor_assign, anchor_prio, cur_b, cur_c))
                if sc + 2.0 < cur_sc:
                    cur_sc, ob, bc_changed = sc, xb, True
                elif ob is not None:
                    cur_b[op.op_id] = ob
        if len(op.eligible_crews) > 1:
            oc = int(cur_c.get(op.op_id, op.crew_id))
            for xc in op.eligible_crews:
                if int(xc) == oc:
                    continue
                cur_c[op.op_id] = int(xc)
                sc = float(fast_surr.score(anchor_assign, anchor_prio, cur_b, cur_c))
                if sc + 2.0 < cur_sc:
                    cur_sc, oc, bc_changed = sc, int(xc), True
                else:
                    cur_c[op.op_id] = oc
    if bc_changed:
        out.append(("o3_fresh_bay_crew", dict(anchor_assign), cur_b, cur_c, dict(anchor_prio), None, False))

    # 2) MIT'26 low-slack precedence-closure subset destroy + Regret-2 (capped to <= 5 tasks for low churn!)
    if anchor_eval is not None and anchor_eval.valid:
        dest = _mit26_precedence_closure_destroy(snapshot, list(graph.rel_tids), anchor_eval, anchor_assign, 0)[:5]
        if dest:
            dest_set = set(dest)
            anc = {tid: aid for tid, aid in anchor_assign.items() if tid not in dest_set}
            mit_a = fast_surr.regret_k_assignment(healthy_agvs, anc, dest, anchor_bays, anchor_prio)
            if mit_a != anchor_assign:
                out.append(
                    (
                        "o3_mit_lowslack_r2",
                        mit_a,
                        dict(anchor_bays),
                        dict(anchor_crews),
                        dict(anchor_prio),
                        _default_task_order_from_prio(snapshot, graph, mit_a, anchor_prio),
                        False,
                    )
                )
    return out
