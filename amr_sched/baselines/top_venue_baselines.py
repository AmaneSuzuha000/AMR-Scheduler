"""External CS Top-Venue Baseline Proposers under Unified Causal Snapshot/Transaction Entry.

Implements peer top-venue methods operating cleanly inside `AMRReschedulingEnv.external_proposer`
so comparisons against RL-CVCR v7..v12 share the exact same physical simulation, causal
snapshot hiding future releases, atomic transaction validation, and net-cost accounting:
  1. `DetourDeltaRAL26Proposer` (IEEE RA-L 2026, Hirayama et al.):
     Reservation-aware Detour-Delta (`Delta = eta - d_bar`) top-k scoring for online
     task assignment and congestion avoidance without LNS/BALANS/CP-SAT.
  2. `BalansIJCAI25PlainProposer` (IJCAI 2025, Cai et al.):
     Zero-offline-prior pure online Multi-Armed Bandit ALNS selecting destroy subset radii
     (`k in {3, 6, 10, all}`) with deterministic B2 insertion evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from typing import Any, Optional

from amr_sched.cvcr.commitment import CommitmentClassification, classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.policy import CandidateProposal, LGCVCRPolicy, _LazyKittingSurrogate
from amr_sched.cvcr.snapshot import NativeSnapshot
from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
from amr_sched.network import Network


def _build_external_proposal(
    *,
    policy_id: str,
    snapshot: NativeSnapshot,
    clf: CommitmentClassification,
    best_assign: dict[str, str],
    best_bays: dict[str, str],
    best_crews: dict[str, int],
    best_eval: EvalResult | None,
    op_prio: dict[str, int],
    eval_cnt: int,
    t0: float,
    budget_ms: float,
    traffic_weights: Optional[dict[tuple[str, str], int]] = None,
    traffic_cutoff: int = 0,
    task_sequences: Optional[dict[str, list[str]]] = None,
) -> CandidateProposal:
    """Package solution cleanly under the CandidateProposal commitment contract."""
    if best_eval is None:
        best_assign, best_bays, best_crews, op_prio = {}, {}, {}, {}
        task_sequences = None
    snap_digest = snapshot.compute_digest()
    snap_by_id = {t.task_id: t for t in snapshot.tasks}
    commit_assign: dict[str, str] = {}
    affected_tasks: list[str] = []
    for tid, new_agv in best_assign.items():
        t_orig = snap_by_id.get(tid)
        if t_orig is None:
            continue
        orig_agv = t_orig.assigned_agv
        is_dist = orig_agv is not None and orig_agv in clf.disturbed_robots
        is_rel = orig_agv is not None and orig_agv != new_agv and tid in clf.releasable_tasks
        is_urg = orig_agv is None and getattr(t_orig, "is_urgent", False)
        if orig_agv is not None or is_urg:
            commit_assign[tid] = orig_agv if (orig_agv is not None and not is_rel and not is_dist) else new_agv
            if commit_assign[tid] != orig_agv:
                affected_tasks.append(tid)
    affected_tasks.sort()
    payload: dict[str, Any] = {
        "policy": policy_id,
        "snapshot_digest": snap_digest,
        "assignment": sorted(commit_assign.items()),
        "bays": sorted(best_bays.items()),
        "crews": sorted(best_crews.items()),
        "obj": round(best_eval.objective, 4) if best_eval is not None else None,
    }
    if traffic_weights:
        payload["traffic"] = sorted((k[0], k[1], v) for k, v in traffic_weights.items())
        payload["cutoff"] = int(traffic_cutoff)
    if task_sequences:
        payload["task_sequences"] = sorted((str(k), list(v)) for k, v in task_sequences.items())
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return CandidateProposal(
        schema="amr-lgcvcr.proposal.v1",
        policy_id=policy_id,
        snapshot_digest=snap_digest,
        affected_robots=sorted({commit_assign[t] for t in affected_tasks}),
        affected_tasks=affected_tasks,
        affected_operations=[],
        proposed_assignment=commit_assign,
        proposed_routes=best_eval.robot_paths if best_eval is not None else {},
        proposed_bays=best_bays,
        proposed_crews=best_crews,
        predicted_obj=best_eval.objective if best_eval is not None else 0.0,
        predicted_makespan=max(best_eval.op_end.values(), default=snapshot.tick) if best_eval is not None else snapshot.tick,
        candidate_digest=digest,
        proposed_traffic_weights=traffic_weights or None,
        proposed_traffic_cutoff=int(traffic_cutoff) if traffic_weights else 0,
        proposed_task_sequences=dict(task_sequences) if task_sequences else None,
        proposed_op_priorities=op_prio,
        telemetry={
            "eval_count": eval_cnt,
            "elapsed_ms": round(elapsed_ms, 3),
            "deadline_met": elapsed_ms <= budget_ms,
            "cp_stats": {},
        },
    )


class DetourDeltaRAL26Proposer:
    """IEEE RA-L 2026 Detour-Delta (`Delta = eta - d_bar`) Online MAPD Proposer."""

    def __init__(
        self,
        *,
        lambda_change: float = 15.0,
        alpha_makespan: float = 3.0,
        use_rust: bool = True,
    ) -> None:
        self.lambda_change = float(lambda_change)
        self.alpha_makespan = float(alpha_makespan)
        self.use_rust = bool(use_rust)

    def __call__(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        budget_ms: float = 50.0,
    ) -> CandidateProposal:
        t0 = time.perf_counter()
        deadline = t0 + max(0.008, (float(budget_ms) * 0.78) / 1000.0)
        clf = classify_commitments(snapshot)
        max_spd = max((e.speed_limit for e in network.edges.values()), default=3.0)
        dd = compute_detour_delta_signals(network, snapshot)
        bay_delta = dd["bay_detour_delta"]
        tw_q3 = dd["weights_q3"]

        pol = LGCVCRPolicy(
            "RAL26_DETOUR_DELTA",
            network,
            lambda_change=self.lambda_change,
            alpha_makespan=self.alpha_makespan,
            use_rust=self.use_rust,
        )
        op_prio = {op.op_id: max(snapshot.tick, op.due_step - op.proc_steps) for op in snapshot.operations}
        healthy = pol._healthy_robots(snapshot, clf.disturbed_robots)
        base_assign = pol._base_assignment(snapshot, healthy, op_prio)
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}

        evaluator = CVCREvaluator(
            network=network,
            snapshot=snapshot,
            mode="B2",
            max_speed=max_spd,
            lambda_change=self.lambda_change,
            alpha_makespan=self.alpha_makespan,
            use_rust=self.use_rust,
        )
        if tw_q3:
            evaluator.set_traffic_weights(tw_q3, cutoff_step=snapshot.tick + 45)
        best_eval = evaluator.evaluate(base_assign, op_priority=op_prio)
        best_assign = dict(base_assign)
        eval_cnt = 1

        if len(healthy) >= 2 and time.perf_counter() < deadline:
            surr = _LazyKittingSurrogate(
                network,
                snapshot,
                op_prio,
                self.alpha_makespan,
                self.lambda_change,
                bay_detour_delta=bay_delta,
            )
            rel_tids = sorted(clf.releasable_tasks)
            if rel_tids:
                cand_assign = surr.regret_k_assignment(
                    healthy,
                    {k: v for k, v in best_assign.items() if k not in set(rel_tids)},
                    rel_tids,
                    op_bays=best_bays,
                )
                if cand_assign != best_assign and time.perf_counter() < deadline:
                    ev = evaluator.evaluate(cand_assign, op_priority=op_prio)
                    eval_cnt += 1
                    if ev.valid and ev.objective < best_eval.objective:
                        best_eval = ev
                        best_assign = cand_assign
                if time.perf_counter() < deadline:
                    best_assign, best_eval, c1 = pol._run_1opt_and_swap_ranked(
                        snapshot, clf, rel_tids, evaluator, best_assign, best_eval,
                        healthy, op_prio, deadline, allow_swaps=False, surr=surr, hys_margin=0.0,
                    )
                    eval_cnt += c1

        return _build_external_proposal(
            policy_id="RAL26_DETOUR_DELTA",
            snapshot=snapshot,
            clf=clf,
            best_assign=best_assign,
            best_bays=best_bays,
            best_crews=best_crews,
            best_eval=best_eval,
            op_prio=op_prio,
            eval_cnt=eval_cnt,
            t0=t0,
            budget_ms=budget_ms,
            traffic_weights=tw_q3,
            traffic_cutoff=(snapshot.tick + 45 if tw_q3 else 0),
        )


class BalansIJCAI25PlainProposer:
    """IJCAI 2025 Plain Online MAB-ALNS Proposer (zero prior, multi-radius destroy arms)."""

    RADII = (3, 6, 10, 999)

    def __init__(self, *, seed: int = 0, use_rust: bool = True) -> None:
        self.seed = int(seed)
        self.use_rust = bool(use_rust)
        self.arm_n: dict[int, float] = {k: 1.0 for k in self.RADII}
        self.arm_val: dict[int, float] = {k: 0.1 for k in self.RADII}
        self.rng = random.Random(self.seed + 307)

    def reset(self) -> None:
        self.arm_n = {k: 1.0 for k in self.RADII}
        self.arm_val = {k: 0.1 for k in self.RADII}
        self.rng = random.Random(self.seed + 307)

    def __call__(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        budget_ms: float = 100.0,
    ) -> CandidateProposal:
        t0 = time.perf_counter()
        deadline = t0 + max(0.008, (float(budget_ms) * 0.78) / 1000.0)
        clf = classify_commitments(snapshot)
        max_spd = max((e.speed_limit for e in network.edges.values()), default=3.0)
        pol = LGCVCRPolicy("IJCAI25_BALANS_PLAIN", network, use_rust=self.use_rust, seed=self.seed)
        op_prio = {op.op_id: max(snapshot.tick, op.due_step - op.proc_steps) for op in snapshot.operations}
        healthy = pol._healthy_robots(snapshot, clf.disturbed_robots)
        base_assign = pol._base_assignment(snapshot, healthy, op_prio)
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}

        evaluator = CVCREvaluator(network=network, snapshot=snapshot, mode="B2", max_speed=max_spd, use_rust=self.use_rust)
        best_eval = evaluator.evaluate(base_assign, op_priority=op_prio)
        best_assign = dict(base_assign)
        eval_cnt = 1

        tot_n = sum(self.arm_n.values())
        arm_k = max(
            self.RADII,
            key=lambda k: self.arm_val[k] + 0.5 * math.sqrt(math.log(max(2.0, tot_n)) / self.arm_n[k]),
        )
        if len(healthy) >= 2 and time.perf_counter() < deadline:
            surr = _LazyKittingSurrogate(network, snapshot, op_prio, 3.0, 15.0)
            rel_all = sorted(clf.releasable_tasks)
            self.rng.shuffle(rel_all)
            rel_sub = rel_all[: min(len(rel_all), arm_k)]
            best_bays, best_crews, op_prio, best_eval, cb = pol._divert_prod_extended(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, op_prio,
                min(deadline, time.perf_counter() + 0.020),
            )
            eval_cnt += cb
            init_obj = best_eval.objective
            if rel_sub and time.perf_counter() < deadline:
                cand_a = surr.regret_k_assignment(
                    healthy,
                    {t: a for t, a in best_assign.items() if t not in set(rel_sub)},
                    rel_sub,
                    op_bays=best_bays,
                    rng=self.rng,
                    noise=0.05,
                )
                if cand_a != best_assign:
                    ev = evaluator.evaluate(cand_a, op_priority=op_prio, op_bay_overrides=best_bays, op_crew_overrides=best_crews)
                    eval_cnt += 1
                    if ev.valid and ev.objective < best_eval.objective:
                        best_eval = ev
                        best_assign = cand_a
                if time.perf_counter() < deadline:
                    best_assign, best_eval, c1 = pol._run_1opt_and_swap_ranked(
                        snapshot, clf, rel_sub, evaluator, best_assign, best_eval, healthy, op_prio, deadline,
                        op_bay_overrides=best_bays, op_crew_overrides=best_crews, allow_swaps=True, surr=surr,
                    )
                    eval_cnt += c1
            gain = max(-0.5, min(1.5, (init_obj - best_eval.objective) / 80.0))
            self.arm_n[arm_k] = self.arm_n[arm_k] * 0.8 + 1.0
            self.arm_val[arm_k] += (gain - self.arm_val[arm_k]) / self.arm_n[arm_k]

        return _build_external_proposal(
            policy_id="IJCAI25_BALANS_PLAIN",
            snapshot=snapshot,
            clf=clf,
            best_assign=best_assign,
            best_bays=best_bays,
            best_crews=best_crews,
            best_eval=best_eval,
            op_prio=op_prio,
            eval_cnt=eval_cnt,
            t0=t0,
            budget_ms=budget_ms,
        )
