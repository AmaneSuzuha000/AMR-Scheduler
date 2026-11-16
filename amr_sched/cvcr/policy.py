"""LG-CVCR Research Brain Policy Suite for AMR-Scheduler.

Implements both ported frozen scientific policies and the new IDEA:
- `INCUMBENT_B1`: Full-rebuild reservation baseline
- `SACC_B2`: Cross-domain closure release + B2 incremental evaluation
- `CVCR_B2`: Value-expanded S3 release + B2 incremental evaluation + 1-opt/swap
- `OUTCOME_MLP_FULL`: Frozen CIE `mlp_full` 13-feature outcome regressor + B2
- `KEEP_RELEASE_MLP_BALANCED`: Frozen CIE `mlp_keep_release_balanced` classifier + B2
- `BUDGET_MATCHED_ALNS`: Deterministic ruin-and-recreate ALNS under equal 150 ms budget
- `SYSTEM07_RECOVERY`: Online stagnation/starvation transport + resource recovery
- `SYSTEM08_ADMISSION`: Capacity-aware fleet admission gating (`K <= N`)
- `IDEA_CORRIDOR_CVCR_JOINT`: Proposed Corridor-Saturation Dynamic Fleet Admission
  + Bottleneck-Criticality Commitment Value + Joint Production-Transport Switching + Rust-B2
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

from amr_sched.cvcr.commitment import classify_commitments, select_release_set
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.features import (
    extract_task_features,
    load_frozen_cie_models,
    predict_mlp_keep_release_proba,
    predict_mlp_regression,
)
from amr_sched.cvcr.snapshot import NativeSnapshot
from amr_sched.network import Network


@dataclass
class CandidateProposal:
    """Scheduling proposal emitted by LGCVCRPolicy (read-only until owner commits)."""

    schema: str
    policy_id: str
    snapshot_digest: str
    affected_robots: list[str]
    affected_tasks: list[str]
    affected_operations: list[str]
    proposed_assignment: dict[str, str]
    proposed_routes: dict[str, list[tuple[str, int]]]
    proposed_bays: dict[str, str]
    proposed_crews: dict[str, int]
    predicted_obj: float
    predicted_makespan: int
    candidate_digest: str
    telemetry: dict[str, Any]
    proposed_admitted_agvs: Optional[list[str]] = None


class LGCVCRPolicy:
    """Scientific decision brain generating execution-feasible joint candidates."""

    def __init__(
        self,
        method_id: str,
        network: Network,
        lambda_change: float = 15.0,
        alpha_makespan: float = 3.0,
        use_rust: bool = True,
        seed: int = 0,
    ) -> None:
        self.method_id = method_id
        self.network = network
        self.lambda_change = lambda_change
        self.alpha_makespan = alpha_makespan
        self.use_rust = use_rust
        self.seed = seed

    def _healthy_robots(self, snapshot: NativeSnapshot, disturbed: set[str]) -> list[str]:
        healthy = [
            r.robot_id
            for r in snapshot.robots
            if r.robot_id not in disturbed and r.state not in ("faulted", "blocked")
        ]
        if not healthy:
            healthy = [r.robot_id for r in snapshot.robots if r.state != "faulted"]
        if not healthy:
            healthy = [r.robot_id for r in snapshot.robots]
        return sorted(healthy)

    def _base_assignment(
        self,
        snapshot: NativeSnapshot,
        healthy_robots: list[str],
        op_prio: dict[str, int],
    ) -> dict[str, str]:
        load: dict[str, int] = {rid: 0 for rid in healthy_robots}
        base_assign: dict[str, str] = {}
        for t in snapshot.tasks:
            if t.status.lower() == "completed":
                continue
            if t.assigned_agv is not None:
                base_assign[t.task_id] = t.assigned_agv
                if t.assigned_agv in load:
                    load[t.assigned_agv] += 1

        pending_unassigned = [
            t for t in snapshot.tasks
            if t.status.lower() != "completed" and t.task_id not in base_assign
        ]
        pending_unassigned.sort(
            key=lambda t: (t.release_step, op_prio.get(t.kit_op_id or "", t.due_step or 999), t.task_id)
        )
        for t in pending_unassigned:
            rb = min(healthy_robots, key=lambda r: (load.get(r, 0), r))
            base_assign[t.task_id] = rb
            load[rb] = load.get(rb, 0) + 1
        return base_assign

    def propose(
        self,
        snapshot: NativeSnapshot,
        budget_ms: float = 150.0,
    ) -> CandidateProposal:
        """Produce a legal scheduling candidate strictly within `budget_ms`."""
        t0 = time.perf_counter()
        # Keep reserve margin so full policy step stays strictly inside budget_ms
        deadline = t0 + max(0.008, (budget_ms / 1000.0) * 0.78)
        snap_digest = snapshot.compute_digest()

        clf = classify_commitments(snapshot)
        eval_mode = (
            "A1"
            if self.method_id == "ABLATION_NO_B2_INCREMENTAL"
            else ("B1" if self.method_id in ("INCUMBENT_B1", "E0") else "B2")
        )
        evaluator = CVCREvaluator(
            network=self.network,
            snapshot=snapshot,
            mode=eval_mode,
            lambda_change=self.lambda_change,
            alpha_makespan=self.alpha_makespan,
            use_rust=self.use_rust,
        )

        op_prio = {
            op.op_id: max(snapshot.tick, op.due_step - op.proc_steps)
            for op in snapshot.operations
        }
        healthy_robots = self._healthy_robots(snapshot, clf.disturbed_robots)
        base_assign = self._base_assignment(snapshot, healthy_robots, op_prio)

        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
        proposed_admitted: Optional[list[str]] = None

        if self.method_id == "SYSTEM08_ADMISSION":
            best_assign, best_eval, eval_count, proposed_admitted = self._search_fleet_admission(
                snapshot, evaluator, healthy_robots, op_prio, deadline
            )
        else:
            base_eval = evaluator.evaluate(base_assign, op_priority=op_prio)
            if base_eval.valid:
                evaluator.set_incumbent(base_assign, base_eval, op_priority=op_prio)
            best_eval = base_eval
            best_assign = dict(base_assign)
            eval_count = 1

            if self.method_id == "SYSTEM07_RECOVERY":
                best_assign, best_bays, best_crews, best_eval, eval_count = self._search_system07_recovery(
                    snapshot, clf, evaluator, base_assign, best_eval, healthy_robots, op_prio, deadline
                )
            elif self.method_id == "BUDGET_MATCHED_ALNS":
                best_assign, best_eval, eval_count = self._search_alns(
                    snapshot, clf, evaluator, base_assign, best_eval, healthy_robots, op_prio, deadline
                )
            elif self.method_id in (
                "IDEA_CORRIDOR_CVCR_JOINT",
                "ABLATION_NO_CORRIDOR_GATE",
                "ABLATION_NO_BAY_CREW_DIVERSION",
                "ABLATION_NO_B2_INCREMENTAL",
            ):
                best_assign, best_bays, best_crews, best_eval, eval_count, proposed_admitted = (
                    self._search_idea_corridor_cvcr_joint(
                        snapshot,
                        clf,
                        evaluator,
                        base_assign,
                        best_eval,
                        healthy_robots,
                        op_prio,
                        deadline,
                        skip_corridor_gate=(self.method_id == "ABLATION_NO_CORRIDOR_GATE"),
                        skip_bay_crew=(self.method_id == "ABLATION_NO_BAY_CREW_DIVERSION"),
                    )
                )
            else:
                # CVCR_B2 / SACC_B2 / INCUMBENT_B1 / OUTCOME_MLP_FULL / KEEP_RELEASE_MLP_BALANCED
                best_assign, best_eval, eval_count = self._search_release_guided(
                    snapshot, clf, evaluator, base_assign, best_eval, healthy_robots, op_prio, deadline
                )

        affected_tasks = sorted(
            tid for tid in best_assign
            if base_assign.get(tid) != best_assign[tid]
        )
        affected_robots = sorted(
            {best_assign[tid] for tid in affected_tasks}
            | {base_assign[tid] for tid in affected_tasks if tid in base_assign}
        )
        affected_ops = sorted(
            op.op_id for op in snapshot.operations
            if op.bay_id != best_bays.get(op.op_id, op.bay_id)
            or op.crew_id != best_crews.get(op.op_id, op.crew_id)
        )

        cand_payload = {
            "policy": self.method_id,
            "snapshot_digest": snap_digest,
            "assignment": sorted(best_assign.items()),
            "bays": sorted(best_bays.items()),
            "crews": sorted(best_crews.items()),
            "admitted": sorted(proposed_admitted) if proposed_admitted is not None else None,
            "obj": round(best_eval.objective, 4),
        }
        cand_digest = hashlib.sha256(json.dumps(cand_payload, sort_keys=True).encode()).hexdigest()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return CandidateProposal(
            schema="amr-lgcvcr.proposal.v1",
            policy_id=self.method_id,
            snapshot_digest=snap_digest,
            affected_robots=affected_robots,
            affected_tasks=affected_tasks,
            affected_operations=affected_ops,
            proposed_assignment=best_assign,
            proposed_routes=best_eval.robot_paths,
            proposed_bays=best_bays,
            proposed_crews=best_crews,
            predicted_obj=best_eval.objective,
            predicted_makespan=max(best_eval.op_end.values()) if best_eval.op_end else snapshot.tick,
            candidate_digest=cand_digest,
            proposed_admitted_agvs=proposed_admitted,
            telemetry={
                "eval_count": eval_count,
                "backend": "rust" if evaluator.use_rust else "python",
                "b2_inc_hits": evaluator.b2_inc_hits,
                "b2_fallbacks": evaluator.b2_fallbacks,
                "elapsed_ms": round(elapsed_ms, 3),
                "budget_ms": budget_ms,
                "deadline_met": elapsed_ms <= budget_ms,
                "eval_latencies_ms": [round(l, 3) for l in evaluator.eval_lats_ms],
            },
        )

    def _select_ordered_release(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        base_eval: EvalResult,
    ) -> list[str]:
        """Select and order releasable tasks according to policy_id."""
        if self.method_id in ("OUTCOME_MLP_FULL", "KEEP_RELEASE_MLP_BALANCED"):
            models = load_frozen_cie_models()
            tids, feats = extract_task_features(
                snapshot,
                clf,
                base_eval.op_res_ready,
                base_eval.order_unload_ticks,
                base_eval.robot_paths,
            )
            if not tids:
                return []
            if self.method_id == "OUTCOME_MLP_FULL":
                scores = predict_mlp_regression(models["mlp_full"], feats)
                scored = sorted(zip(scores.tolist(), tids), reverse=True)
                return [tid for _, tid in scored[:10]]
            else:
                probs = predict_mlp_keep_release_proba(models["mlp_keep_release_balanced"], feats)
                thr = float(models.get("mlp_keep_release_balanced_thr", 0.5))
                scored = sorted(zip(probs.tolist(), tids), reverse=True)
                chosen = [tid for p, tid in scored if p >= thr][:12]
                if len(chosen) < 4:
                    chosen = [tid for _, tid in scored[:6]]
                return chosen

        scope = (
            "S2_CROSS_DOMAIN_CLOSURE"
            if self.method_id in ("SACC_B2", "INCUMBENT_B1")
            else "S3_VALUE_EXPANDED_CVCR"
        )
        return sorted(select_release_set(scope, clf, snapshot, max_release=12))

    def _run_1opt_and_swap(
        self,
        release_tids: list[str],
        evaluator: CVCREvaluator,
        cur_assign: dict[str, str],
        best_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        op_bay_overrides: Optional[dict[str, str]] = None,
        op_crew_overrides: Optional[dict[str, int]] = None,
    ) -> tuple[dict[str, str], EvalResult, int]:
        best_assign = dict(cur_assign)
        eval_count = 0

        for tid in release_tids:
            if time.perf_counter() >= deadline:
                break
            if tid not in best_assign:
                continue
            cur_r = best_assign[tid]
            for new_r in healthy_robots:
                if new_r == cur_r or time.perf_counter() >= deadline:
                    continue
                cand = dict(best_assign)
                cand[tid] = new_r
                res = evaluator.evaluate(
                    cand,
                    op_priority=op_prio,
                    op_bay_overrides=op_bay_overrides,
                    op_crew_overrides=op_crew_overrides,
                )
                eval_count += 1
                if res.valid and (not best_eval.valid or res.objective < best_eval.objective):
                    best_eval = res
                    best_assign = cand
                    evaluator.set_incumbent(
                        best_assign, best_eval, op_priority=op_prio, op_bay_overrides=op_bay_overrides
                    )

        # Pairwise swap on top-8 released tasks within deadline
        top_k = [tid for tid in release_tids[:8] if tid in best_assign]
        for i in range(len(top_k)):
            if time.perf_counter() >= deadline:
                break
            for j in range(i + 1, len(top_k)):
                if time.perf_counter() >= deadline:
                    break
                ti, tj = top_k[i], top_k[j]
                if best_assign[ti] == best_assign[tj]:
                    continue
                cand = dict(best_assign)
                cand[ti], cand[tj] = cand[tj], cand[ti]
                res = evaluator.evaluate(
                    cand,
                    op_priority=op_prio,
                    op_bay_overrides=op_bay_overrides,
                    op_crew_overrides=op_crew_overrides,
                )
                eval_count += 1
                if res.valid and (not best_eval.valid or res.objective < best_eval.objective):
                    best_eval = res
                    best_assign = cand
                    evaluator.set_incumbent(
                        best_assign, best_eval, op_priority=op_prio, op_bay_overrides=op_bay_overrides
                    )

        return best_assign, best_eval, eval_count

    def _search_release_guided(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        base_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
    ) -> tuple[dict[str, str], EvalResult, int]:
        if len(healthy_robots) < 2:
            return base_assign, base_eval, 1
        rel_tids = self._select_ordered_release(snapshot, clf, base_eval)
        if not rel_tids:
            return base_assign, base_eval, 1
        best_assign, best_eval, cnt = self._run_1opt_and_swap(
            rel_tids, evaluator, base_assign, base_eval, healthy_robots, op_prio, deadline
        )
        return best_assign, best_eval, 1 + cnt

    def _search_alns(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        base_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
    ) -> tuple[dict[str, str], EvalResult, int]:
        """Budget-matched ALNS destroy-repair baseline over releasable tasks."""
        releasable = sorted(clf.releasable_tasks)
        if not releasable or len(healthy_robots) < 2:
            return base_assign, base_eval, 1

        rng = random.Random(self.seed + snapshot.tick * 131 + 17)
        best_assign = dict(base_assign)
        best_eval = base_eval
        cur_assign = dict(base_assign)
        eval_count = 1
        iter_idx = 0

        while time.perf_counter() < deadline:
            iter_idx += 1
            k_destroy = min(len(releasable), rng.randint(1, min(4, len(releasable))))
            if iter_idx % 2 == 0:
                # Shaw/cluster removal: pick tasks with shared op or robot
                seed_t = rng.choice(releasable)
                neighbors = [v for v in sorted(clf.adj.get(seed_t, ())) if v in clf.releasable_tasks]
                chosen = ([seed_t] + neighbors)[:k_destroy]
            else:
                chosen = rng.sample(releasable, k_destroy)

            cand = dict(cur_assign)
            for tid in chosen:
                cand[tid] = rng.choice(healthy_robots)

            res = evaluator.evaluate(cand, op_priority=op_prio)
            eval_count += 1
            if res.valid and (not best_eval.valid or res.objective < best_eval.objective):
                best_eval = res
                best_assign = cand
                cur_assign = cand
                evaluator.set_incumbent(best_assign, best_eval, op_priority=op_prio)

        return best_assign, best_eval, eval_count

    def _search_system07_recovery(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        base_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, int], EvalResult, int]:
        """SYSTEM07 online recovery: rescues disturbed tasks & switches contended resources."""
        best_assign = dict(base_assign)
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
        best_eval = base_eval
        eval_count = 1

        # Prioritise tasks currently stuck on disturbed/faulted robots first
        stalled_tasks = sorted(
            t.task_id
            for t in snapshot.tasks
            if t.task_id in clf.releasable_tasks and best_assign.get(t.task_id) in clf.disturbed_robots
        )
        other_rel = sorted(t for t in clf.releasable_tasks if t not in stalled_tasks)
        ordered_rel = stalled_tasks + other_rel

        # Force-clear any stalled task onto a healthy robot right away if currently invalid
        for tid in stalled_tasks:
            if healthy_robots:
                best_assign[tid] = healthy_robots[0]
        if stalled_tasks:
            r_rescue = evaluator.evaluate(best_assign, op_priority=op_prio)
            eval_count += 1
            if r_rescue.valid:
                best_eval = r_rescue
                evaluator.set_incumbent(best_assign, best_eval, op_priority=op_prio)

        best_assign, best_eval, c1 = self._run_1opt_and_swap(
            ordered_rel, evaluator, best_assign, best_eval, healthy_robots, op_prio, deadline
        )
        eval_count += c1

        # Production resource recovery: joint bay / crew switching for unstarted, unpicked ops
        for op in snapshot.operations:
            if op.started or time.perf_counter() >= deadline:
                continue
            kit_tasks = [t for t in snapshot.tasks if t.kit_op_id == op.op_id]
            any_picked = any(t.status.lower() in ("picked", "loading", "to_dropoff", "completed") for t in kit_tasks)

            for crew in op.eligible_crews:
                if crew == best_crews[op.op_id] or time.perf_counter() >= deadline:
                    continue
                cand_crews = dict(best_crews)
                cand_crews[op.op_id] = crew
                res = evaluator.evaluate(
                    best_assign,
                    op_priority=op_prio,
                    op_bay_overrides=best_bays,
                    op_crew_overrides=cand_crews,
                )
                eval_count += 1
                if res.valid and res.objective < best_eval.objective:
                    best_eval = res
                    best_crews = cand_crews

            if not any_picked:
                for bay in op.eligible_bays:
                    if bay == best_bays[op.op_id] or time.perf_counter() >= deadline:
                        continue
                    cand_bays = dict(best_bays)
                    cand_bays[op.op_id] = bay
                    res = evaluator.evaluate(
                        best_assign,
                        op_priority=op_prio,
                        op_bay_overrides=cand_bays,
                        op_crew_overrides=best_crews,
                    )
                    eval_count += 1
                    if res.valid and res.objective < best_eval.objective:
                        best_eval = res
                        best_bays = cand_bays

        return best_assign, best_bays, best_crews, best_eval, eval_count

    def _search_fleet_admission(
        self,
        snapshot: NativeSnapshot,
        evaluator: CVCREvaluator,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
    ) -> tuple[dict[str, str], EvalResult, int, list[str]]:
        """SYSTEM08 capacity-aware fleet admission gating over active subsets K <= N."""
        n_rob = len(healthy_robots)
        if n_rob == 0:
            return {}, evaluator.evaluate({}), 1, []

        # Preserve frozen/executing tasks' AGVs as mandatory admitted members
        mandatory = {
            t.assigned_agv
            for t in snapshot.tasks
            if t.assigned_agv is not None and t.status.lower() in ("picked", "loading", "to_dropoff")
        }

        candidate_ks = sorted({k for k in [3, 4, max(2, n_rob // 2), min(6, n_rob), n_rob] if 1 <= k <= n_rob})
        best_assign: dict[str, str] = {}
        best_eval: EvalResult | None = None
        best_score = float("inf")
        best_admitted: list[str] = list(healthy_robots)
        eval_count = 0
        last_res: EvalResult | None = None
        tried_n_rob = False

        # Quadratic bottleneck-queueing congestion tax scaled by single-lane corridor fraction
        edge_caps = [e.capacity for e in self.network.edges.values()]
        single_lane_frac = sum(1 for c in edge_caps if c <= 1) / max(1, len(edge_caps))
        congestion_coeff = 12.0 * single_lane_frac + 1.0

        active_tasks = [t for t in snapshot.tasks if t.status.lower() != "completed"]

        for k in candidate_ks:
            rem_s = deadline - time.perf_counter()
            if rem_s <= 0.0 or (best_eval is not None and rem_s < 0.025):
                break
            if k == n_rob:
                tried_n_rob = True
            # Build subset of size >= len(mandatory)
            subset = sorted(mandatory)
            for r in healthy_robots:
                if len(subset) >= k:
                    break
                if r not in subset:
                    subset.append(r)

            cand_assign: dict[str, str] = {}
            for i, t in enumerate(active_tasks):
                if t.status.lower() in ("picked", "loading", "to_dropoff") and t.assigned_agv:
                    cand_assign[t.task_id] = t.assigned_agv
                else:
                    cand_assign[t.task_id] = subset[i % len(subset)]

            res = evaluator.evaluate(cand_assign, op_priority=op_prio)
            last_res = res
            eval_count += 1

            # Combine spatiotemporal objective + active fleet congestion tax
            if res.valid:
                cap_score = res.objective + (len(subset) ** 2) * congestion_coeff
                if cap_score < best_score:
                    best_score = cap_score
                    best_eval = res
                    best_assign = cand_assign
                    best_admitted = subset

        if best_eval is None:
            cand_assign = {t.task_id: healthy_robots[i % n_rob] for i, t in enumerate(active_tasks)}
            if tried_n_rob and last_res is not None:
                best_eval = last_res
            else:
                best_eval = evaluator.evaluate(cand_assign, op_priority=op_prio)
                eval_count += 1
            best_assign = cand_assign
            best_admitted = list(healthy_robots)

        # Only retain assignments for tasks already assigned to admitted robots;
        # leave PENDING/unassigned tasks unpinned so native dispatch operates dynamically inside best_admitted.
        admitted_set = set(best_admitted)
        pinned_assign = {
            t.task_id: t.assigned_agv
            for t in active_tasks
            if t.assigned_agv is not None and t.assigned_agv in admitted_set
        }
        return pinned_assign, best_eval, eval_count, best_admitted

    def _search_idea_corridor_cvcr_joint(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        base_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        skip_corridor_gate: bool = False,
        skip_bay_crew: bool = False,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, int], EvalResult, int, list[str]]:
        """Proposed IDEA: Corridor-Saturation Adaptive Admission + Cross-Domain CVCR-B2 + Joint Bay/Crew.

        1. Estimates bottleneck corridor load and task slack pressure from current snapshot.
        2. Evaluates candidate active fleet gates K(t) in {ceil(N/2), ..., N} jointly with
           bottleneck-prioritized release assignments via Rust-B2 incremental evaluation.
        3. Explores legal crew/bay diversion on contended operations without violating picked kits.
        """
        n_rob = len(healthy_robots)
        active_tasks = [t for t in snapshot.tasks if t.status.lower() != "completed"]
        if n_rob == 0 or not active_tasks:
            return base_assign, {op.op_id: op.bay_id for op in snapshot.operations}, {op.op_id: op.crew_id for op in snapshot.operations}, base_eval, 1, healthy_robots

        mandatory = {
            t.assigned_agv
            for t in snapshot.tasks
            if t.assigned_agv in healthy_robots and t.status.lower() in ("picked", "loading", "to_dropoff")
        }

        # Measure bottleneck density & urgent backlog pressure
        edge_caps = [e.capacity for e in self.network.edges.values()]
        single_lane_frac = sum(1 for c in edge_caps if c <= 1) / max(1, len(edge_caps))
        n_urgent = sum(1 for op in snapshot.operations if not op.started and op.is_urgent)
        n_blocked_now = sum(1 for r in snapshot.robots if r.state in ("blocked", "parking"))

        # Adaptive target fleet bounds: keep tighter when single-lane and blocked, expand for urgent/backlog
        min_k = max(len(mandatory), min(n_rob, max(3, n_rob // 2)))
        candidate_ks = sorted({min_k, min(n_rob, min_k + 1), min(6, n_rob), n_rob})

        best_assign = dict(base_assign)
        best_eval = base_eval
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
        best_admitted = list(healthy_robots)
        base_tax_coeff = max(1.0, 18.0 * single_lane_frac + 5.0 * n_blocked_now - 2.0 * n_urgent)
        best_score = base_eval.objective + (len(healthy_robots) ** 2) * base_tax_coeff if base_eval.valid else float("inf")
        eval_count = 1

        # First evaluate candidate fleet gates using fast Rust-B2 (unless skipped by ablation)
        if not skip_corridor_gate:
            for k in candidate_ks:
                rem_s = deadline - time.perf_counter()
                if rem_s <= 0.0 or (best_eval.valid and rem_s < 0.025):
                    break
                subset = sorted(mandatory)
                for r in healthy_robots:
                    if len(subset) >= k:
                        break
                    if r not in subset:
                        subset.append(r)

                if k == n_rob:
                    cand_assign = dict(base_assign)
                else:
                    cand_assign = {}
                    for i, t in enumerate(active_tasks):
                        if t.task_id in clf.frozen_tasks and t.assigned_agv:
                            cand_assign[t.task_id] = t.assigned_agv
                        else:
                            cand_assign[t.task_id] = subset[i % len(subset)]

                res = evaluator.evaluate(cand_assign, op_priority=op_prio)
                eval_count += 1
                if not res.valid:
                    continue

                # Corridor-congestion penalty shrinks when urgent ops need throughput
                corridor_tax = (len(subset) ** 2) * base_tax_coeff
                score = res.objective + corridor_tax
                if score < best_score:
                    best_score = score
                    best_eval = res
                    best_assign = cand_assign
                    best_admitted = subset

        # Seed B2 incumbent with winning admitted subset, then run S3 release-set 1-opt/swap inside admitted subset
        if best_eval.valid:
            evaluator.set_incumbent(best_assign, best_eval, op_priority=op_prio)
        rel_tids = sorted(select_release_set("S3_VALUE_EXPANDED_CVCR", clf, snapshot, max_release=12))
        if len(best_admitted) >= 2 and rel_tids:
            opt_assign, opt_eval, c1 = self._run_1opt_and_swap(
                rel_tids, evaluator, best_assign, best_eval, best_admitted, op_prio, deadline
            )
            eval_count += c1
            if opt_eval.valid and opt_eval.objective < best_eval.objective:
                best_assign = opt_assign
                best_eval = opt_eval

        # Joint bay/crew switching for contended uncommitted operations (unless skipped by ablation)
        if not skip_bay_crew:
            for op in snapshot.operations:
                if op.started or time.perf_counter() >= deadline:
                    continue
                kit_tasks = [t for t in snapshot.tasks if t.kit_op_id == op.op_id]
                any_picked = any(t.status.lower() in ("picked", "loading", "to_dropoff", "completed") for t in kit_tasks)

                for crew in op.eligible_crews:
                    if crew == best_crews[op.op_id] or time.perf_counter() >= deadline:
                        continue
                    cand_crews = dict(best_crews)
                    cand_crews[op.op_id] = crew
                    res = evaluator.evaluate(
                        best_assign,
                        op_priority=op_prio,
                        op_bay_overrides=best_bays,
                        op_crew_overrides=cand_crews,
                    )
                    eval_count += 1
                    if res.valid and res.objective < best_eval.objective:
                        best_eval = res
                        best_crews = cand_crews

                if not any_picked:
                    for bay in op.eligible_bays:
                        if bay == best_bays[op.op_id] or time.perf_counter() >= deadline:
                            continue
                        cand_bays = dict(best_bays)
                        cand_bays[op.op_id] = bay
                        res = evaluator.evaluate(
                            best_assign,
                            op_priority=op_prio,
                            op_bay_overrides=cand_bays,
                            op_crew_overrides=best_crews,
                        )
                        eval_count += 1
                        if res.valid and res.objective < best_eval.objective:
                            best_eval = res
                            best_bays = cand_bays

        return best_assign, best_bays, best_crews, best_eval, eval_count, best_admitted
