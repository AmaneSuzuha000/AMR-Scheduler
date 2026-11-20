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
import math
import platform
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

_SCIPY_LSA: Any = False


def linear_sum_assignment(cost_matrix: Any) -> tuple[np.ndarray, np.ndarray]:
    """Lazily dispatch rectangular assignment via SciPy if available, else Hungarian fallback."""
    global _SCIPY_LSA
    if _SCIPY_LSA is False:
        if sys.platform == "win32" and getattr(platform, "_wmi", None) is not None:
            platform._wmi = None  # type: ignore[attr-defined]
        try:
            from scipy.optimize import linear_sum_assignment as _lsa

            _SCIPY_LSA = _lsa
        except Exception:
            _SCIPY_LSA = None
    if _SCIPY_LSA is not None:
        return _SCIPY_LSA(cost_matrix)

    from amr_sched.allocation.hungarian import hungarian as _hungarian_sq

    arr = np.asarray(cost_matrix, dtype=float)
    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    n_r, n_c = arr.shape
    n_m = max(n_r, n_c)
    pad_val = float(np.max(np.abs(arr))) + 1e6
    sq = np.full((n_m, n_m), pad_val, dtype=float)
    sq[:n_r, :n_c] = arr
    pairs = sorted(_hungarian_sq(sq.tolist()))
    kept = [(r, c) for (r, c) in pairs if r < n_r and c < n_c]
    return np.asarray([r for r, _ in kept], dtype=np.int64), np.asarray([c for _, c in kept], dtype=np.int64)

_ORT_CP_MODEL: Any = False


def _get_ortools_cp_model() -> Any:
    """Lazily import OR-Tools `cp_model` only when local CP-SAT refinement is invoked."""
    global _ORT_CP_MODEL
    if _ORT_CP_MODEL is False:
        if sys.platform == "win32" and getattr(platform, "_wmi", None) is not None:
            platform._wmi = None  # type: ignore[attr-defined]
        try:
            from ortools.sat.python import cp_model as _cp

            _ORT_CP_MODEL = _cp
        except Exception:
            _ORT_CP_MODEL = None
    return _ORT_CP_MODEL

from amr_sched.cvcr.clock import (
    apply_cpsat_time_limit,
    is_virtual_clock_active,
    now_s as _cvcr_now_s,
    record_work_ms as _cvcr_record_work_ms,
)
from amr_sched.cvcr.commitment import classify_commitments, select_release_set
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.features import (
    extract_task_features,
    load_frozen_cie_models,
    predict_mlp_keep_release_proba,
    predict_mlp_regression,
)
from amr_sched.cvcr.snapshot import NativeSnapshot, op_predecessors
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
    proposed_task_sequences: Optional[dict[str, list[str]]] = None
    proposed_traffic_weights: Optional[dict[tuple[str, str], int]] = None
    proposed_traffic_cutoff: int = 0
    proposed_guide_routes: Optional[dict[str, list[tuple[str, str]]]] = None
    proposed_op_priorities: Optional[dict[str, int]] = None
    proposed_live_repairs: Optional[dict[str, dict[str, Any]]] = None
    proposed_bottleneck_locks: Optional[dict[str, list[tuple[str, int, int]]]] = None


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
        cp_mode: Optional[str] = None,
        lrho_predictor: Optional[Any] = None,
        destroy_selector: Optional[Any] = None,
        bay_detour_delta: Optional[dict[str, float]] = None,
        detour_weights_q3: Optional[dict[tuple[str, str], int]] = None,
        kitting_ranker: Optional[Any] = None,
        v14_policy: Optional[Any] = None,
        v15_policy: Optional[Any] = None,
    ) -> None:
        self.method_id = method_id
        self.network = network
        self.lambda_change = lambda_change
        self.alpha_makespan = alpha_makespan
        self.use_rust = use_rust
        self.seed = seed
        self.cp_mode = cp_mode
        self.lrho_predictor = lrho_predictor
        self.destroy_selector = destroy_selector
        self.bay_detour_delta = dict(bay_detour_delta) if bay_detour_delta else None
        self.detour_weights_q3 = dict(detour_weights_q3) if detour_weights_q3 else None
        self.kitting_ranker = kitting_ranker
        self.v14_policy = v14_policy
        self.v15_policy = v15_policy
        self._last_v15_commit_seq_horizon: int = 0
        self._last_v15_order_seq: dict[str, list[str]] = {}
        self._last_v17_live_repairs: Optional[dict[str, dict[str, Any]]] = None
        self._last_v17_bottleneck_locks: Optional[dict[str, list[tuple[str, int, int]]]] = None
        self._last_kitting_decision: dict[str, Any] = {}
        self._prev_ltm_weights: dict[tuple[str, str], int] = {}
        self._last_cp_stats: dict[str, int] = {
            "cp_invoked": 0,
            "cp_feasible": 0,
            "cp_entered_eval": 0,
            "cp_accepted_improved": 0,
            "cp_queue_only": 0,
            "cp_skipped_no_change": 0,
            "cp_skipped_no_budget": 0,
        }

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
        healthy_set = set(healthy_robots)
        load: dict[str, int] = {rid: 0 for rid in healthy_robots}
        base_assign: dict[str, str] = {}
        for t in snapshot.tasks:
            if t.status.lower() == "completed":
                continue
            if t.assigned_agv is not None and (
                t.assigned_agv in healthy_set
                or t.status.lower() in ("picked", "loading", "to_dropoff")
            ):
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
        orig_method_id = self.method_id
        sub_scope = getattr(self.v15_policy, "active_subscope", None) if orig_method_id.startswith(("V16_", "V17_")) else None
        if sub_scope and str(sub_scope).startswith(("V7_E", "V10_E")):
            self.method_id = str(sub_scope)
        try:
            prop = self._propose_impl(snapshot, budget_ms=budget_ms)
        finally:
            self.method_id = orig_method_id
        prop.policy_id = orig_method_id
        return prop

    def _propose_impl(
        self,
        snapshot: NativeSnapshot,
        budget_ms: float = 150.0,
    ) -> CandidateProposal:
        """Internal solver dispatch across legacy, V10, V15, and V16 candidate portfolios."""
        t0 = _cvcr_now_s()
        self._last_v17_live_repairs = None
        self._last_v17_bottleneck_locks = None
        rl_unpinned_methods = {
            "S0_DIRECT_AFFECTED",
            "S2_CROSS_DOMAIN_CLOSURE",
            "S3_VALUE_EXPANDED_CVCR",
            "CVCR_S0",
            "CVCR_S2",
            "BEST_CVCR_FULL",
            "OUTCOME_MLP_FULLSTACK",
            "KEEP_RELEASE_MLP_FULLSTACK",
            "ALNS_FULLSTACK",
            "SYSTEM07_UNPINNED",
            "EXP_E1_FAST_RESCUE",
            "EXP_E2_PROD_FOCUSED",
            "EXP_E3_JOINT_CORE",
            "EXP_E4_CONGESTION_PACER",
            "EXP_E5_AGGRESSIVE_GRID",
            "V6_E1_FAST_RESCUE",
            "V6_E2_CRIT_PROD_CI",
            "V6_E3_JOINT_CI_HYS",
            "V6_E4_PACER_CI",
            "V6_E5_AGGRESSIVE_CI",
            "V7_E1_FAST_RESCUE",
            "V7_E2_CRIT_PROD_CI",
            "V7_E3_JOINT_CI_HYS",
            "V7_E4_PACER_CI",
            "V7_E5_AGGRESSIVE_CI",
            "V7_E6_BALANCE_KM",
            "V8_E1_FAST_RESCUE",
            "V8_E2_CRIT_PROD_CI",
            "V8_E3_JOINT_PC_HYS",
            "V8_E4_LTM_PACER_CI",
            "V8_E5_AGGRESSIVE_LTM",
            "V8_E6_BALANCE_KM_PC",
            "V9_E1_FAST_RESCUE",
            "V9_E2_CRIT_PROD_CI",
            "V9_E3_JOINT_PC_HYS",
            "V9_E4_LTM_GUIDE_CI",
            "V9_E5_AGGRESSIVE_GUIDE",
            "V9_E6_BALANCE_KM_GUIDE",
            "V9_E7_CPSAT_JOINT_GUIDE",
            "V10_E1_FAST_RESCUE",
            "V10_E2_CRIT_PROD_CI",
            "V10_E3_JOINT_PC_HYS",
            "V10_E4_LTM_GUIDE_CI",
            "V10_E5_AGGRESSIVE_GUIDE",
            "V10_E6_BALANCE_KM_LRHO",
            "V10_E7_COND_CPSAT",
            "V12_SURGICAL_TPG_REPAIR",
            "V12_RESCUE_DETOUR_FAST",
            "V12_CRIT_PROD_DESTROY_S",
            "V12_PACER_CONSERVATIVE",
            "V12_BALANCED_DESTROY_M",
            "V12_BALANCED_DESTROY_L",
            "V12_AGGRESSIVE_GUIDE_TPG",
            "V12_SAFE_CPSAT_JOINT",
            "V13_KITTING_DEFAULT",
            "V13_KITTING_DET",
            "V13_KITTING_SUP",
            "V13_KITTING_RL",
            "V14_JOINT_RL",
            "V15_EA_JOINT",
            "V15_EA_FAST",
            "V16_EA_JOINT",
            "V16_CALM_FAST",
            "V16_BALANCED",
            "V17_EA_JOINT",
            "V17_DUAL_OP",
            "V17_BALANCED",
            "V17_CALM_FAST",
        }
        is_rl_scope = self.method_id in rl_unpinned_methods or self.method_id.startswith("V17_")
        if budget_ms <= 25.0:
            max_evals = 6
        elif budget_ms <= 60.0:
            max_evals = 20
        elif budget_ms <= 110.0:
            max_evals = 40
        else:
            max_evals = 80

        # Real anytime wall-clock deadline with headroom for tail eval & commit serialization
        if budget_ms <= 25.0:
            deadline = t0 + max(0.004, (float(budget_ms) * 0.78) / 1000.0)
        elif budget_ms <= 65.0:
            deadline = t0 + (float(budget_ms) * 0.80) / 1000.0
        elif budget_ms <= 110.0:
            deadline = t0 + (float(budget_ms) * 0.78) / 1000.0
        else:
            deadline = t0 + (float(budget_ms) * 0.70) / 1000.0
        snap_digest = snapshot.compute_digest()
        self._last_kitting_decision = {}
        self._last_cp_stats = {
            "cp_invoked": 0,
            "cp_feasible": 0,
            "cp_entered_eval": 0,
            "cp_accepted_improved": 0,
            "cp_queue_only": 0,
            "cp_skipped_no_change": 0,
            "cp_skipped_no_budget": 0,
        }

        clf = classify_commitments(snapshot)
        if clf.disturbed_robots and budget_ms > 35.0 and not self.method_id.startswith(("V15_", "V16_")):
            deadline = min(deadline, t0 + 0.025)
        eval_mode = (
            "A1"
            if self.method_id == "ABLATION_NO_B2_INCREMENTAL"
            else ("B1" if self.method_id in ("INCUMBENT_B1", "E0") else "B2")
        )
        max_spd = max((e.speed_limit for e in self.network.edges.values()), default=1.0)
        evaluator = CVCREvaluator(
            network=self.network,
            snapshot=snapshot,
            mode=eval_mode,
            max_speed=max_spd,
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
        proposed_traffic_weights: Optional[dict[tuple[str, str], int]] = None
        proposed_traffic_cutoff: int = 0
        proposed_guide_routes: Optional[dict[str, list[tuple[str, str]]]] = None
        proposed_op_priorities: Optional[dict[str, int]] = None
        rel_tids_used: list[str] = []

        search_t0 = _cvcr_now_s()
        if self.method_id == "SYSTEM08_ADMISSION":
            best_assign, best_eval, eval_count, proposed_admitted = self._search_fleet_admission(
                snapshot, evaluator, healthy_robots, op_prio, deadline
            )
            base_eval = best_eval
        else:
            base_eval = evaluator.evaluate(base_assign, op_priority=op_prio)
            if base_eval.valid and _cvcr_now_s() < deadline:
                evaluator.set_incumbent(base_assign, base_eval, op_priority=op_prio)
            best_eval = base_eval
            best_assign = dict(base_assign)
            eval_count = 1

            if self.method_id in ("SYSTEM07_RECOVERY", "SYSTEM07_UNPINNED"):
                best_assign, best_bays, best_crews, best_eval, eval_count = self._search_system07_recovery(
                    snapshot, clf, evaluator, base_assign, best_eval, healthy_robots, op_prio, deadline
                )
            elif self.method_id == "BUDGET_MATCHED_ALNS":
                best_assign, best_eval, eval_count = self._search_alns(
                    snapshot, clf, evaluator, base_assign, best_eval, healthy_robots, op_prio, deadline
                )
            elif self.method_id == "ALNS_FULLSTACK":
                best_assign, best_bays, best_crews, best_eval, eval_count = self._search_alns_fullstack(
                    snapshot, clf, evaluator, base_assign, best_eval, healthy_robots, op_prio, deadline, budget_ms=budget_ms
                )
            elif self.method_id == "BEST_CVCR_FULL":
                best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids_used, proposed_admitted = (
                    self._search_best_cvcr_full(
                        snapshot,
                        clf,
                        evaluator,
                        base_assign,
                        best_eval,
                        healthy_robots,
                        op_prio,
                        deadline,
                        budget_ms=budget_ms,
                    )
                )
            elif self.method_id in ("OUTCOME_MLP_FULLSTACK", "KEEP_RELEASE_MLP_FULLSTACK"):
                best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids_used = (
                    self._search_mlp_fullstack(
                        snapshot,
                        clf,
                        evaluator,
                        base_assign,
                        best_eval,
                        healthy_robots,
                        op_prio,
                        deadline,
                        budget_ms=budget_ms,
                    )
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
            elif self.method_id in (
                "EXP_E1_FAST_RESCUE",
                "EXP_E2_PROD_FOCUSED",
                "EXP_E3_JOINT_CORE",
                "EXP_E4_CONGESTION_PACER",
                "EXP_E5_AGGRESSIVE_GRID",
            ):
                best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids_used, proposed_admitted = (
                    self._search_expert_portfolio(
                        snapshot,
                        clf,
                        evaluator,
                        base_assign,
                        best_eval,
                        healthy_robots,
                        op_prio,
                        deadline,
                        budget_ms=budget_ms,
                    )
                )
            elif self.method_id in (
                "V6_E1_FAST_RESCUE",
                "V6_E2_CRIT_PROD_CI",
                "V6_E3_JOINT_CI_HYS",
                "V6_E4_PACER_CI",
                "V6_E5_AGGRESSIVE_CI",
                "V7_E1_FAST_RESCUE",
                "V7_E2_CRIT_PROD_CI",
                "V7_E3_JOINT_CI_HYS",
                "V7_E4_PACER_CI",
                "V7_E5_AGGRESSIVE_CI",
                "V7_E6_BALANCE_KM",
            ):
                best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids_used, proposed_admitted = (
                    self._search_v6_expert_portfolio(
                        snapshot,
                        clf,
                        evaluator,
                        base_assign,
                        best_eval,
                        healthy_robots,
                        op_prio,
                        deadline,
                        budget_ms=budget_ms,
                    )
                )
            elif self.method_id in (
                "V8_E1_FAST_RESCUE",
                "V8_E2_CRIT_PROD_CI",
                "V8_E3_JOINT_PC_HYS",
                "V8_E4_LTM_PACER_CI",
                "V8_E5_AGGRESSIVE_LTM",
                "V8_E6_BALANCE_KM_PC",
            ):
                best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids_used, proposed_admitted, proposed_traffic_weights = (
                    self._search_v8_expert_portfolio(
                        snapshot,
                        clf,
                        evaluator,
                        base_assign,
                        best_eval,
                        healthy_robots,
                        op_prio,
                        deadline,
                        budget_ms=budget_ms,
                    )
                )
            elif self.method_id in (
                "V9_E1_FAST_RESCUE",
                "V9_E2_CRIT_PROD_CI",
                "V9_E3_JOINT_PC_HYS",
                "V9_E4_LTM_GUIDE_CI",
                "V9_E5_AGGRESSIVE_GUIDE",
                "V9_E6_BALANCE_KM_GUIDE",
                "V9_E7_CPSAT_JOINT_GUIDE",
            ):
                best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids_used, proposed_admitted, proposed_traffic_weights, proposed_traffic_cutoff, proposed_guide_routes = (
                    self._search_v9_expert_portfolio(
                        snapshot,
                        clf,
                        evaluator,
                        base_assign,
                        best_eval,
                        healthy_robots,
                        op_prio,
                        deadline,
                        budget_ms=budget_ms,
                    )
                )
            elif self.method_id in (
                "V10_E1_FAST_RESCUE",
                "V10_E2_CRIT_PROD_CI",
                "V10_E3_JOINT_PC_HYS",
                "V10_E4_LTM_GUIDE_CI",
                "V10_E5_AGGRESSIVE_GUIDE",
                "V10_E6_BALANCE_KM_LRHO",
                "V10_E7_COND_CPSAT",
            ):
                (
                    best_assign,
                    best_bays,
                    best_crews,
                    best_eval,
                    eval_count,
                    rel_tids_used,
                    proposed_admitted,
                    proposed_traffic_weights,
                    proposed_traffic_cutoff,
                    proposed_guide_routes,
                    proposed_op_priorities,
                ) = self._search_v10_expert_portfolio(
                    snapshot,
                    clf,
                    evaluator,
                    base_assign,
                    best_eval,
                    healthy_robots,
                    op_prio,
                    deadline,
                    budget_ms=budget_ms,
                )
            elif self.method_id in (
                "V12_SURGICAL_TPG_REPAIR",
                "V12_RESCUE_DETOUR_FAST",
                "V12_CRIT_PROD_DESTROY_S",
                "V12_PACER_CONSERVATIVE",
                "V12_BALANCED_DESTROY_M",
                "V12_BALANCED_DESTROY_L",
                "V12_AGGRESSIVE_GUIDE_TPG",
                "V12_SAFE_CPSAT_JOINT",
            ):
                (
                    best_assign,
                    best_bays,
                    best_crews,
                    best_eval,
                    eval_count,
                    rel_tids_used,
                    proposed_admitted,
                    proposed_traffic_weights,
                    proposed_traffic_cutoff,
                    proposed_guide_routes,
                    proposed_op_priorities,
                ) = self._search_v12_expert_portfolio(
                    snapshot,
                    clf,
                    evaluator,
                    base_assign,
                    best_eval,
                    healthy_robots,
                    op_prio,
                    deadline,
                    budget_ms=budget_ms,
                )
            elif self.method_id in (
                "V15_EA_JOINT",
                "V15_EA_FAST",
                "V16_EA_JOINT",
                "V16_CALM_FAST",
                "V16_BALANCED",
            ) or self.method_id.startswith("V17_"):
                (
                    best_assign,
                    best_bays,
                    best_crews,
                    best_eval,
                    eval_count,
                    rel_tids_used,
                    proposed_admitted,
                    proposed_traffic_weights,
                    proposed_traffic_cutoff,
                    proposed_guide_routes,
                    proposed_op_priorities,
                ) = self._search_v15_execution_aware(
                    snapshot,
                    clf,
                    evaluator,
                    base_assign,
                    best_eval,
                    healthy_robots,
                    op_prio,
                    deadline,
                    budget_ms=budget_ms,
                )
            elif self.method_id == "V14_JOINT_RL":
                (
                    best_assign,
                    best_bays,
                    best_crews,
                    best_eval,
                    eval_count,
                    rel_tids_used,
                    proposed_admitted,
                    proposed_traffic_weights,
                    proposed_traffic_cutoff,
                    proposed_guide_routes,
                    proposed_op_priorities,
                ) = self._search_v14_joint_rl(
                    snapshot,
                    clf,
                    evaluator,
                    base_assign,
                    best_eval,
                    healthy_robots,
                    op_prio,
                    deadline,
                    budget_ms=budget_ms,
                )
            elif self.method_id in (
                "V13_KITTING_DEFAULT",
                "V13_KITTING_DET",
                "V13_KITTING_SUP",
                "V13_KITTING_RL",
            ):
                (
                    best_assign,
                    best_bays,
                    best_crews,
                    best_eval,
                    eval_count,
                    rel_tids_used,
                    proposed_admitted,
                    proposed_traffic_weights,
                    proposed_traffic_cutoff,
                    proposed_guide_routes,
                    proposed_op_priorities,
                ) = self._search_v13_kitting_rl(
                    snapshot,
                    clf,
                    evaluator,
                    base_assign,
                    best_eval,
                    healthy_robots,
                    op_prio,
                    deadline,
                    budget_ms=budget_ms,
                )
            elif is_rl_scope:
                best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids_used = (
                    self._search_rl_scoped(
                        snapshot,
                        clf,
                        evaluator,
                        base_assign,
                        best_eval,
                        healthy_robots,
                        op_prio,
                        deadline,
                        budget_ms=budget_ms,
                        max_evals=max_evals,
                    )
                )
            else:
                # CVCR_B2 / SACC_B2 / INCUMBENT_B1 / OUTCOME_MLP_FULL / KEEP_RELEASE_MLP_BALANCED
                best_assign, best_eval, eval_count = self._search_release_guided(
                    snapshot, clf, evaluator, base_assign, best_eval, healthy_robots, op_prio, deadline
                )
        search_ms = (_cvcr_now_s() - search_t0) * 1000.0
        timed_out_gracefully = _cvcr_now_s() >= deadline

        initial_obj = float(base_eval.objective) if base_eval.valid else float(best_eval.objective) + 5000.0
        best_obj = float(best_eval.objective)
        improved_delta = max(0.0, initial_obj - best_obj) if best_eval.valid else 0.0
        snap_by_id = {t.task_id: t for t in snapshot.tasks}

        if is_rl_scope:
            affected_set: set[str] = set()
            for tid, new_agv in best_assign.items():
                t_orig = snap_by_id.get(tid)
                if t_orig is None:
                    continue
                orig_agv = t_orig.assigned_agv
                if orig_agv is not None and orig_agv != new_agv and tid in clf.releasable_tasks:
                    affected_set.add(tid)
                elif orig_agv is not None and orig_agv in clf.disturbed_robots:
                    affected_set.add(tid)
                elif orig_agv is None and getattr(t_orig, "is_urgent", False):
                    affected_set.add(tid)
            affected_tasks = sorted(affected_set)
            # Commit disturbed rescues, urgent dispatches, and reassigned healthy tasks;
            # keep non-urgent PENDING tasks unpinned so GreedyAllocator handles future arrivals dynamically.
            commit_assign = {
                tid: agv
                for tid, agv in best_assign.items()
                if tid in affected_set
                or (snap_by_id.get(tid) is not None and snap_by_id[tid].assigned_agv is not None)
            }
            proposed_task_sequences: Optional[dict[str, list[str]]] = None
            h_lim = int(getattr(self, "_last_v15_commit_seq_horizon", 0))
            fb_snap_init = getattr(snapshot, "execution_feedback", None)
            v16_use_seq_commit = (
                self.method_id.startswith(("V16_", "V17_"))
                and h_lim >= 2
                and not any(bool(getattr(op, "resource_windows", ())) for op in snapshot.operations)
                and not (fb_snap_init is not None and not fb_snap_init.is_empty())
            )
            if self.method_id.startswith("V15_") or (self.method_id.startswith(("V16_", "V17_")) and not v16_use_seq_commit):
                commit_assign = {}
                affected_set_v15: set[str] = set()
                for tid, new_agv in best_assign.items():
                    t_orig = snap_by_id.get(tid)
                    if t_orig is None:
                        continue
                    orig_agv = t_orig.assigned_agv
                    is_dist = orig_agv is not None and orig_agv in clf.disturbed_robots
                    is_rel = orig_agv is not None and orig_agv != new_agv and tid in clf.releasable_tasks
                    is_urg = orig_agv is None and getattr(t_orig, "is_urgent", False)
                    if orig_agv is not None or is_urg:
                        chosen_agv = (
                            orig_agv
                            if (orig_agv is not None and not is_rel and not is_dist)
                            else new_agv
                        )
                        commit_assign[tid] = chosen_agv
                        if chosen_agv != orig_agv:
                            affected_set_v15.add(tid)
                if h_lim > 0:
                    ord_seq = getattr(self, "_last_v15_order_seq", {})
                    rob_by_id = {r.robot_id: r for r in snapshot.robots}
                    for rid, seq in sorted(ord_seq.items()):
                        rob = rob_by_id.get(rid)
                        if (
                            rob is None
                            or rob.state.lower() in ("blocked", "faulted")
                            or rob.faulted_until > snapshot.tick
                        ):
                            continue
                        for tid in seq:
                            t_obj = snap_by_id.get(tid)
                            if t_obj is None or t_obj.status.lower() != "pending":
                                continue
                            # Stale-safe soft head-pinning: never pin aged pending tasks or lock multi-task queues
                            if (int(snapshot.tick) - int(t_obj.release_step)) < 16 and int(snapshot.tick) < 200:
                                if commit_assign.get(tid) != rid:
                                    commit_assign[tid] = rid
                                    affected_set_v15.add(tid)
                            break
                affected_set = affected_set_v15
                affected_tasks = sorted(affected_set)
            else:
                if (
                    (
                        self.method_id.startswith(("V6_E", "V7_E", "V8_E", "V9_E", "V10_E", "V12_", "V13_", "V14_"))
                        and self.method_id != "V12_SURGICAL_TPG_REPAIR"
                    )
                    or v16_use_seq_commit
                ):
                    commit_assign = self._order_commit_by_ci(
                        commit_assign, best_assign, best_eval, include_heads=True
                    )
                if (
                    (
                        self.method_id.startswith(("V7_E", "V8_E", "V9_E", "V10_E", "V12_", "V13_", "V14_"))
                        and self.method_id != "V12_SURGICAL_TPG_REPAIR"
                    )
                    or v16_use_seq_commit
                ):
                    proposed_task_sequences = {}
                    for rid, seq in sorted(best_eval.robot_order_seq.items()):
                        q_tids = [tid for tid in seq if tid in snap_by_id and snap_by_id[tid].status.lower() != "completed"]
                        if q_tids:
                            proposed_task_sequences[rid] = q_tids
                            for qtid in q_tids:
                                commit_assign[qtid] = rid
                                if snap_by_id[qtid].assigned_agv is None:
                                    affected_set.add(qtid)
                    affected_tasks = sorted(affected_set)
        else:
            proposed_task_sequences = None
            affected_tasks = sorted(
                tid for tid in best_assign
                if base_assign.get(tid) != best_assign[tid]
                or (
                    snap_by_id.get(tid) is not None
                    and snap_by_id[tid].assigned_agv is not None
                    and snap_by_id[tid].assigned_agv != best_assign[tid]
                )
            )
            commit_assign = best_assign

        affected_robots = sorted(
            {commit_assign[tid] for tid in affected_tasks if tid in commit_assign}
            | {
                snap_by_id[tid].assigned_agv
                for tid in affected_tasks
                if tid in snap_by_id and snap_by_id[tid].assigned_agv is not None
            }
        )
        affected_ops = sorted(
            op.op_id for op in snapshot.operations
            if op.bay_id != best_bays.get(op.op_id, op.bay_id)
            or op.crew_id != best_crews.get(op.op_id, op.crew_id)
        )

        fb_snap = getattr(snapshot, "execution_feedback", None)
        if fb_snap is not None:
            fb_blk = fb_snap.blocked_intervals(t_commit=int(snapshot.tick), default_horizon=12)
            if fb_blk:
                merged_tw = dict(proposed_traffic_weights) if proposed_traffic_weights is not None else {}
                for ek_fb, wv_fb in fb_snap.corridor_q3_penalties().items():
                    merged_tw[ek_fb] = max(merged_tw.get(ek_fb, 0), int(wv_fb))
                proposed_traffic_weights = merged_tw
                max_blk_en = max(int(en_fb) for en_fb in fb_blk.values())
                proposed_traffic_cutoff = max(int(proposed_traffic_cutoff), max_blk_en)

        cand_payload = {
            "policy": self.method_id,
            "snapshot_digest": snap_digest,
            "assignment": sorted(commit_assign.items()),
            "bays": sorted(best_bays.items()),
            "crews": sorted(best_crews.items()),
            "admitted": sorted(proposed_admitted) if proposed_admitted is not None else None,
            "queues": (
                sorted((k, tuple(v)) for k, v in proposed_task_sequences.items())
                if proposed_task_sequences is not None
                else None
            ),
            "obj": round(best_eval.objective, 4),
        }
        if proposed_traffic_weights is not None:
            cand_payload["traffic"] = sorted((k[0], k[1], v) for k, v in proposed_traffic_weights.items())
        if proposed_traffic_cutoff > 0:
            cand_payload["traffic_cutoff"] = int(proposed_traffic_cutoff)
        if proposed_guide_routes is not None:
            cand_payload["guides"] = sorted(
                (rid, tuple(edges)) for rid, edges in proposed_guide_routes.items()
            )
        if proposed_op_priorities is not None:
            cand_payload["op_priorities"] = sorted(
                (str(k), int(v)) for k, v in proposed_op_priorities.items()
            )
        proposed_live_repairs = getattr(self, "_last_v17_live_repairs", None)
        proposed_bottleneck_locks = getattr(self, "_last_v17_bottleneck_locks", None)
        if proposed_live_repairs:
            cand_payload["live_repairs"] = sorted(
                (aid, r.get("kind"), r.get("task_id"), r.get("goal"), len(r.get("hops") or ()))
                for aid, r in proposed_live_repairs.items()
            )
        if proposed_bottleneck_locks:
            cand_payload["bottleneck_locks"] = sorted(
                (aid, tuple(spans)) for aid, spans in proposed_bottleneck_locks.items()
            )
        cand_digest = hashlib.sha256(json.dumps(cand_payload, sort_keys=True).encode()).hexdigest()
        elapsed_ms = (_cvcr_now_s() - t0) * 1000.0

        return CandidateProposal(
            schema="amr-lgcvcr.proposal.v1",
            policy_id=self.method_id,
            snapshot_digest=snap_digest,
            affected_robots=affected_robots,
            affected_tasks=affected_tasks,
            affected_operations=affected_ops,
            proposed_assignment=commit_assign,
            proposed_routes=best_eval.robot_paths,
            proposed_bays=best_bays,
            proposed_crews=best_crews,
            predicted_obj=best_eval.objective,
            predicted_makespan=max(best_eval.op_end.values()) if best_eval.op_end else snapshot.tick,
            candidate_digest=cand_digest,
            proposed_admitted_agvs=proposed_admitted,
            proposed_task_sequences=proposed_task_sequences,
            proposed_traffic_weights=proposed_traffic_weights,
            proposed_traffic_cutoff=proposed_traffic_cutoff,
            proposed_guide_routes=proposed_guide_routes,
            proposed_op_priorities=proposed_op_priorities,
            proposed_live_repairs=proposed_live_repairs,
            proposed_bottleneck_locks=proposed_bottleneck_locks,
            telemetry={
                "eval_count": eval_count,
                "max_evals": max_evals,
                "initial_obj": round(initial_obj, 4),
                "best_obj": round(best_obj, 4),
                "improved_delta": round(improved_delta, 4),
                "backend": "rust" if evaluator.use_rust else "python",
                "b2_inc_hits": evaluator.b2_inc_hits,
                "b2_fallbacks": evaluator.b2_fallbacks,
                "requested_budget_ms": round(float(budget_ms), 3),
                "search_ms": round(search_ms, 3),
                "elapsed_ms": round(elapsed_ms, 3),
                "budget_ms": budget_ms,
                "deadline_met": elapsed_ms <= budget_ms,
                "timed_out_gracefully": timed_out_gracefully,
                "eval_latencies_ms": [round(l, 3) for l in evaluator.eval_lats_ms],
                "cp_stats": dict(self._last_cp_stats),
                "kitting_decision": dict(self._last_kitting_decision),
            },
        )

    def _select_ordered_release(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        base_eval: EvalResult,
    ) -> list[str]:
        """Select and order releasable tasks according to policy_id."""
        if self.method_id in (
            "OUTCOME_MLP_FULL",
            "OUTCOME_MLP_FULLSTACK",
            "KEEP_RELEASE_MLP_BALANCED",
            "KEEP_RELEASE_MLP_FULLSTACK",
        ):
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
            if self.method_id in ("OUTCOME_MLP_FULL", "OUTCOME_MLP_FULLSTACK"):
                scores = predict_mlp_regression(models["mlp_full"], feats)
                scored = sorted(zip(scores.tolist(), tids), reverse=True)
                return [tid for _, tid in scored[:12]]
            probs = predict_mlp_keep_release_proba(models["mlp_keep_release_balanced"], feats)
            thr = float(models.get("mlp_keep_release_balanced_thr", 0.5))
            scored = sorted(zip(probs.tolist(), tids), reverse=True)
            chosen = [tid for p, tid in scored if p >= thr][:12]
            if len(chosen) < 4:
                chosen = [tid for _, tid in scored[:6]]
            return chosen

        if self.method_id in (
            "BEST_CVCR_FULL",
            "EXP_E2_PROD_FOCUSED",
            "EXP_E3_JOINT_CORE",
            "EXP_E4_CONGESTION_PACER",
            "EXP_E5_AGGRESSIVE_GRID",
            "V6_E2_CRIT_PROD_CI",
            "V6_E3_JOINT_CI_HYS",
            "V6_E4_PACER_CI",
            "V6_E5_AGGRESSIVE_CI",
            "V7_E2_CRIT_PROD_CI",
            "V7_E3_JOINT_CI_HYS",
            "V7_E4_PACER_CI",
            "V7_E5_AGGRESSIVE_CI",
            "V7_E6_BALANCE_KM",
            "V8_E2_CRIT_PROD_CI",
            "V8_E3_JOINT_PC_HYS",
            "V8_E4_LTM_PACER_CI",
            "V8_E5_AGGRESSIVE_LTM",
            "V8_E6_BALANCE_KM_PC",
            "V9_E2_CRIT_PROD_CI",
            "V9_E3_JOINT_PC_HYS",
            "V9_E4_LTM_GUIDE_CI",
            "V9_E5_AGGRESSIVE_GUIDE",
            "V9_E6_BALANCE_KM_GUIDE",
            "V9_E7_CPSAT_JOINT_GUIDE",
            "V10_E2_CRIT_PROD_CI",
            "V10_E3_JOINT_PC_HYS",
            "V10_E4_LTM_GUIDE_CI",
            "V10_E5_AGGRESSIVE_GUIDE",
            "V10_E6_BALANCE_KM_LRHO",
            "V10_E7_COND_CPSAT",
        ):
            # Hybrid priority ordering: S3 urgency closure merged with Keep/Release-MLP scores
            s3_set = select_release_set("S3_VALUE_EXPANDED_CVCR", clf, snapshot, max_release=12)
            krm_scores: dict[str, float] = {}
            try:
                models = load_frozen_cie_models()
                tids, feats = extract_task_features(
                    snapshot,
                    clf,
                    base_eval.op_res_ready,
                    base_eval.order_unload_ticks,
                    base_eval.robot_paths,
                )
                if tids:
                    probs = predict_mlp_keep_release_proba(models["mlp_keep_release_balanced"], feats)
                    krm_scores = {tid: float(p) for tid, p in zip(tids, probs.tolist())}
            except Exception:
                krm_scores = {}

            snap_by_id = {t.task_id: t for t in snapshot.tasks}
            merged = set(s3_set)
            for tid, p in sorted(krm_scores.items(), key=lambda x: -x[1])[:8]:
                if p >= 0.35 or len(merged) < 8:
                    merged.add(tid)
            return sorted(
                merged,
                key=lambda tid: (
                    0 if (snap_by_id.get(tid) and snap_by_id[tid].assigned_agv in clf.disturbed_robots) else 1,
                    0 if tid in s3_set else 1,
                    -krm_scores.get(tid, 0.0),
                    tid,
                ),
            )[:14]

        if self.method_id in ("CVCR_S0", "S0_DIRECT_AFFECTED"):
            scope = "S0_DIRECT_AFFECTED"
        elif self.method_id in ("SACC_B2", "CVCR_S2", "S2_CROSS_DOMAIN_CLOSURE", "INCUMBENT_B1"):
            scope = "S2_CROSS_DOMAIN_CLOSURE"
        else:
            scope = "S3_VALUE_EXPANDED_CVCR"
        return sorted(select_release_set(scope, clf, snapshot, max_release=12))

    @staticmethod
    def _has_budget(
        deadline: float,
        evaluator: CVCREvaluator,
        min_ms: Optional[float] = None,
    ) -> bool:
        rem_ms = (deadline - _cvcr_now_s()) * 1000.0
        if rem_ms <= 0.0:
            return False
        if min_ms is not None:
            return rem_ms >= min_ms
        lats = evaluator.eval_lats_ms
        if not lats:
            return True
        avg_recent = sum(lats[-3:]) / len(lats[-3:])
        return rem_ms >= min(25.0, max(3.0, avg_recent * 0.65))

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
        max_evals: Optional[int] = None,
        allow_swaps: bool = True,
    ) -> tuple[dict[str, str], EvalResult, int]:
        best_assign = dict(cur_assign)
        eval_count = 0

        for tid in release_tids:
            if not self._has_budget(deadline, evaluator) or (max_evals is not None and eval_count >= max_evals):
                break
            if tid not in best_assign:
                continue
            cur_r = best_assign[tid]
            for new_r in healthy_robots:
                if new_r == cur_r:
                    continue
                if not self._has_budget(deadline, evaluator) or (max_evals is not None and eval_count >= max_evals):
                    break
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

        if not allow_swaps:
            return best_assign, best_eval, eval_count

        # Pairwise swap on top-8 released tasks within deadline & max_evals
        top_k = [tid for tid in release_tids[:8] if tid in best_assign]
        for i in range(len(top_k)):
            if not self._has_budget(deadline, evaluator) or (max_evals is not None and eval_count >= max_evals):
                break
            for j in range(i + 1, len(top_k)):
                if not self._has_budget(deadline, evaluator) or (max_evals is not None and eval_count >= max_evals):
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

    def _divert_bays_and_crews(
        self,
        snapshot: NativeSnapshot,
        evaluator: CVCREvaluator,
        best_assign: dict[str, str],
        best_bays: dict[str, str],
        best_crews: dict[str, int],
        best_eval: EvalResult,
        op_prio: dict[str, int],
        deadline: float,
        max_evals: Optional[int] = None,
        rel_set: Optional[set[str]] = None,
        min_gain: float = 0.25,
    ) -> tuple[dict[str, str], dict[str, int], EvalResult, int]:
        """Greedy crew-first, bay-second diversion over unstarted, unpicked operations."""
        eval_count = 0
        unstarted_ops: list[tuple[Any, bool]] = []
        for op in snapshot.operations:
            if op.started:
                continue
            kit_tasks = [t for t in snapshot.tasks if t.kit_op_id == op.op_id]
            if rel_set is not None and not any(t.task_id in rel_set for t in kit_tasks):
                continue
            any_picked = any(
                t.status.lower() in ("picked", "loading", "to_dropoff", "completed")
                for t in kit_tasks
            )
            unstarted_ops.append((op, any_picked))

        # Pass 1: Crew mutex diversion first (hits B2 route cache, <0.3ms per eval)
        for op, _ in unstarted_ops:
            if not self._has_budget(deadline, evaluator, min_ms=0.5) or (max_evals is not None and eval_count >= max_evals):
                break
            for crew in op.eligible_crews:
                if crew == best_crews[op.op_id]:
                    continue
                if not self._has_budget(deadline, evaluator, min_ms=0.5) or (max_evals is not None and eval_count >= max_evals):
                    break
                cand_crews = dict(best_crews)
                cand_crews[op.op_id] = crew
                res = evaluator.evaluate(
                    best_assign,
                    op_priority=op_prio,
                    op_bay_overrides=best_bays,
                    op_crew_overrides=cand_crews,
                )
                eval_count += 1
                if res.valid and res.objective + min_gain < best_eval.objective:
                    best_eval = res
                    best_crews = cand_crews

        # Pass 2: Unpicked bay diversion next (re-routes only affected kit AGVs, ~7-10ms)
        for op, any_picked in unstarted_ops:
            if any_picked or not self._has_budget(deadline, evaluator, min_ms=9.0) or (max_evals is not None and eval_count >= max_evals):
                continue
            for bay in op.eligible_bays:
                if bay == best_bays[op.op_id]:
                    continue
                if not self._has_budget(deadline, evaluator, min_ms=9.0) or (max_evals is not None and eval_count >= max_evals):
                    break
                cand_bays = dict(best_bays)
                cand_bays[op.op_id] = bay
                res = evaluator.evaluate(
                    best_assign,
                    op_priority=op_prio,
                    op_bay_overrides=cand_bays,
                    op_crew_overrides=best_crews,
                )
                eval_count += 1
                if res.valid and res.objective + min_gain < best_eval.objective:
                    best_eval = res
                    best_bays = cand_bays
                    evaluator.set_incumbent(
                        best_assign, best_eval, op_priority=op_prio, op_bay_overrides=best_bays
                    )

        return best_bays, best_crews, best_eval, eval_count

    def _prioritize_disturbed(self, snapshot: NativeSnapshot, clf: Any, rel_tids_raw: list[str]) -> list[str]:
        snap_by_id = {t.task_id: t for t in snapshot.tasks}
        stalled = [
            tid for tid in rel_tids_raw
            if snap_by_id.get(tid) is not None and snap_by_id[tid].assigned_agv in clf.disturbed_robots
        ]
        others = [tid for tid in rel_tids_raw if tid not in stalled]
        return stalled + others

    def _search_rl_scoped(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        base_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        budget_ms: float,
        max_evals: int,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, int], EvalResult, int, list[str]]:
        """Budgeted scoped search for RL-CVCR actions under real wall-clock deadline."""
        best_assign = dict(base_assign)
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
        best_eval = base_eval
        eval_count = 1

        if len(healthy_robots) < 2:
            return best_assign, best_bays, best_crews, best_eval, eval_count, []

        rel_tids_raw = self._select_ordered_release(snapshot, clf, base_eval)
        if not rel_tids_raw:
            return best_assign, best_bays, best_crews, best_eval, eval_count, []

        rel_tids = self._prioritize_disturbed(snapshot, clf, rel_tids_raw)
        is_s0 = self.method_id in ("CVCR_S0", "S0_DIRECT_AFFECTED")
        allow_bay_crew = (not is_s0) and budget_ms >= 35.0 and _cvcr_now_s() < deadline

        if allow_bay_crew:
            bay_deadline = min(deadline, _cvcr_now_s() + max(0.008, (float(budget_ms) * 0.38) / 1000.0))
            rel_filter = set(rel_tids) if self.method_id in ("CVCR_S2", "S2_CROSS_DOMAIN_CLOSURE") else None
            best_bays, best_crews, best_eval, cb = self._divert_bays_and_crews(
                snapshot,
                evaluator,
                best_assign,
                best_bays,
                best_crews,
                best_eval,
                op_prio,
                bay_deadline,
                max_evals=max(1, (max_evals - eval_count) // 2),
                rel_set=rel_filter,
            )
            eval_count += cb

        rem_evals = max(1, max_evals - eval_count)
        if _cvcr_now_s() < deadline:
            best_assign, best_eval, c1 = self._run_1opt_and_swap(
                rel_tids,
                evaluator,
                best_assign,
                best_eval,
                healthy_robots,
                op_prio,
                deadline,
                op_bay_overrides=best_bays,
                op_crew_overrides=best_crews,
                max_evals=rem_evals,
                allow_swaps=(not is_s0 and budget_ms >= 35.0),
            )
            eval_count += c1

        return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids

    def _search_mlp_fullstack(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        base_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        budget_ms: float,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, int], EvalResult, int, list[str]]:
        """Unlocked CIE MLP selector + Bay/Crew diversion + 1-opt/Swap + unpinned PENDING."""
        best_assign = dict(base_assign)
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
        best_eval = base_eval
        eval_count = 1
        if len(healthy_robots) < 2:
            return best_assign, best_bays, best_crews, best_eval, eval_count, []

        rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, base_eval))
        if budget_ms >= 35.0 and _cvcr_now_s() < deadline:
            bay_deadline = min(deadline, _cvcr_now_s() + max(0.008, (float(budget_ms) * 0.40) / 1000.0))
            best_bays, best_crews, best_eval, cb = self._divert_bays_and_crews(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, op_prio, bay_deadline
            )
            eval_count += cb

        if rel_tids and _cvcr_now_s() < deadline:
            best_assign, best_eval, c1 = self._run_1opt_and_swap(
                rel_tids,
                evaluator,
                best_assign,
                best_eval,
                healthy_robots,
                op_prio,
                deadline,
                op_bay_overrides=best_bays,
                op_crew_overrides=best_crews,
                allow_swaps=(budget_ms >= 35.0),
            )
            eval_count += c1
        return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids

    def _search_best_cvcr_full(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        base_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        budget_ms: float,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, int], EvalResult, int, list[str], Optional[list[str]]]:
        """Fullstack portfolio: Bay/Crew Alignment -> S3+MLP 1-opt/Swap -> ALNS -> Conditional SYSTEM08."""
        best_assign = dict(base_assign)
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
        best_eval = base_eval
        eval_count = 1
        proposed_admitted: Optional[list[str]] = None

        if len(healthy_robots) < 2:
            return best_assign, best_bays, best_crews, best_eval, eval_count, [], proposed_admitted

        # Stage 1: Highest ROI/ms — Production Bay/Crew Diversion
        if budget_ms >= 35.0 and _cvcr_now_s() < deadline:
            bay_deadline = min(deadline, _cvcr_now_s() + max(0.010, (float(budget_ms) * 0.40) / 1000.0))
            best_bays, best_crews, best_eval, cb = self._divert_bays_and_crews(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, op_prio, bay_deadline
            )
            eval_count += cb

        # Stage 2: S3+MLP hybrid ordered 1-opt & pairwise swap
        rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, base_eval))
        if rel_tids and _cvcr_now_s() < deadline:
            best_assign, best_eval, c1 = self._run_1opt_and_swap(
                rel_tids,
                evaluator,
                best_assign,
                best_eval,
                healthy_robots,
                op_prio,
                deadline,
                op_bay_overrides=best_bays,
                op_crew_overrides=best_crews,
                allow_swaps=(budget_ms >= 35.0),
            )
            eval_count += c1

        # Stage 3: Warm-started ALNS polish within remaining wall-clock budget
        if budget_ms >= 85.0 and _cvcr_now_s() < deadline:
            best_assign, best_eval, ca = self._search_alns(
                snapshot,
                clf,
                evaluator,
                best_assign,
                best_eval,
                healthy_robots,
                op_prio,
                deadline,
                op_bay_overrides=best_bays,
                op_crew_overrides=best_crews,
                init_eval_count=0,
            )
            eval_count += ca

        # Stage 4: Conditional SYSTEM08 fleet admission strictly for dense single-lane large fleets
        edge_caps = [e.capacity for e in self.network.edges.values()]
        single_lane_frac = sum(1 for c in edge_caps if c <= 1) / max(1, len(edge_caps))
        if single_lane_frac >= 0.40 and len(healthy_robots) >= 8 and _cvcr_now_s() + 0.025 < deadline:
            adm_assign, adm_eval, cadm, adm_list = self._search_fleet_admission(
                snapshot, evaluator, healthy_robots, op_prio, deadline
            )
            eval_count += cadm
            if adm_eval.valid and adm_eval.objective + 0.25 < best_eval.objective:
                best_assign = adm_assign
                best_eval = adm_eval
                proposed_admitted = adm_list

        return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, proposed_admitted

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

    def _search_alns_fullstack(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        base_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        budget_ms: float,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, int], EvalResult, int]:
        """Unlocked ALNS with Bay/Crew diversion and unpinned PENDING tasks."""
        best_assign = dict(base_assign)
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
        best_eval = base_eval
        eval_count = 1
        if budget_ms >= 35.0 and _cvcr_now_s() < deadline:
            bay_deadline = min(deadline, _cvcr_now_s() + max(0.010, (float(budget_ms) * 0.40) / 1000.0))
            best_bays, best_crews, best_eval, cb = self._divert_bays_and_crews(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, op_prio, bay_deadline
            )
            eval_count += cb
        if _cvcr_now_s() < deadline:
            best_assign, best_eval, ca = self._search_alns(
                snapshot,
                clf,
                evaluator,
                best_assign,
                best_eval,
                healthy_robots,
                op_prio,
                deadline,
                op_bay_overrides=best_bays,
                op_crew_overrides=best_crews,
                init_eval_count=0,
            )
            eval_count += ca
        return best_assign, best_bays, best_crews, best_eval, eval_count

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
        op_bay_overrides: Optional[dict[str, str]] = None,
        op_crew_overrides: Optional[dict[str, int]] = None,
        init_eval_count: int = 1,
    ) -> tuple[dict[str, str], EvalResult, int]:
        """DyMAB Sliding-Window UCB (JAIR 2026) adaptive ALNS over releasable tasks."""
        releasable = sorted(clf.releasable_tasks)
        if not releasable or len(healthy_robots) < 2:
            return base_assign, base_eval, init_eval_count

        from collections import deque
        import math

        rng = random.Random(self.seed + snapshot.tick * 131 + 17)
        best_assign = dict(base_assign)
        best_eval = base_eval
        cur_assign = dict(base_assign)
        eval_count = init_eval_count
        snap_by_id = {t.task_id: t for t in snapshot.tasks}

        # Pre-rank critical/tight-due releasable tasks once for directed destroy arms
        crit_ranked = sorted(
            releasable,
            key=lambda tid: (
                float(snap_by_id[tid].due_step - snapshot.tick)
                if (tid in snap_by_id and snap_by_id[tid].due_step is not None)
                else 999.0,
                tid,
            ),
        )

        # 4 non-stationary destroy-repair arms (Sliding-Window UCB, W=8)
        n_arms = 4
        wins: list[deque[float]] = [deque(maxlen=8) for _ in range(n_arms)]
        pulls = [0] * n_arms
        total_pulls = 0
        reward_scale = 25.0

        while self._has_budget(deadline, evaluator):
            if total_pulls < n_arms:
                arm = total_pulls
            else:
                log_t = math.log(float(total_pulls + 1))
                arm = max(
                    range(n_arms),
                    key=lambda a: (
                        (sum(wins[a]) / max(1, len(wins[a]))) / reward_scale
                        + 0.75 * math.sqrt(log_t / max(1, pulls[a]))
                    ),
                )

            loads = {r: 0 for r in healthy_robots}
            for tid, agv in cur_assign.items():
                if agv in loads:
                    loads[agv] += 1
            least_loaded = sorted(healthy_robots, key=lambda r: (loads[r], r))
            most_loaded = least_loaded[::-1]

            k_destroy = min(len(releasable), rng.randint(1, min(4, len(releasable))))
            cand = dict(cur_assign)

            if arm == 0:
                # Arm 0: Critical-slack destroy + least-loaded-robot greedy repair
                pool_top = crit_ranked[: max(2, min(len(crit_ranked), 8))]
                chosen = rng.sample(pool_top, min(k_destroy, len(pool_top)))
                for idx_c, tid in enumerate(chosen):
                    r_pool = least_loaded[: min(len(least_loaded), 3)]
                    alts = [r for r in r_pool if r != cand.get(tid)] or r_pool
                    cand[tid] = rng.choice(alts) if idx_c > 0 else alts[0]
            elif arm == 1:
                # Arm 1: Conflict-neighborhood destroy + load-biased repair
                seed_t = rng.choice(crit_ranked[: max(1, len(crit_ranked) // 2)])
                neighbors = [v for v in sorted(clf.adj.get(seed_t, ())) if v in clf.releasable_tasks]
                chosen = ([seed_t] + neighbors)[:k_destroy]
                r_pool = least_loaded[: min(len(least_loaded), 3)]
                for tid in chosen:
                    cand[tid] = rng.choice(r_pool)
            elif arm == 2:
                # Arm 2: Overloaded -> Underloaded migration or 2-task swap
                hi_r = most_loaded[0]
                lo_r = least_loaded[0]
                hi_tids = [tid for tid in releasable if cand.get(tid) == hi_r]
                lo_tids = [tid for tid in releasable if cand.get(tid) == lo_r]
                if hi_tids and hi_r != lo_r:
                    t_hi = rng.choice(hi_tids)
                    if lo_tids and rng.random() < 0.45:
                        t_lo = rng.choice(lo_tids)
                        cand[t_hi], cand[t_lo] = cand[t_lo], cand[t_hi]
                    else:
                        cand[t_hi] = lo_r
                else:
                    chosen = rng.sample(releasable, k_destroy)
                    for tid in chosen:
                        cand[tid] = rng.choice(healthy_robots)
            else:
                # Arm 3: Unbiased random perturbation escape
                chosen = rng.sample(releasable, k_destroy)
                for tid in chosen:
                    cand[tid] = rng.choice(healthy_robots)

            res = evaluator.evaluate(
                cand,
                op_priority=op_prio,
                op_bay_overrides=op_bay_overrides,
                op_crew_overrides=op_crew_overrides,
            )
            eval_count += 1
            pulls[arm] += 1
            total_pulls += 1

            gain = 0.0
            if res.valid and (not best_eval.valid or res.objective < best_eval.objective):
                gain = max(0.0, float(best_eval.objective) - float(res.objective)) if best_eval.valid else 50.0
                reward_scale = max(reward_scale, gain)
                best_eval = res
                best_assign = cand
                cur_assign = cand
                evaluator.set_incumbent(
                    best_assign, best_eval, op_priority=op_prio, op_bay_overrides=op_bay_overrides
                )
            wins[arm].append(gain)

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

        s07_deadline = min(deadline, _cvcr_now_s() + 0.060)
        best_assign, best_eval, c1 = self._run_1opt_and_swap(
            ordered_rel,
            evaluator,
            best_assign,
            best_eval,
            healthy_robots,
            op_prio,
            s07_deadline,
            max_evals=10,
            allow_swaps=False,
        )
        eval_count += c1

        # Production resource recovery: joint bay / crew switching for unstarted, unpicked ops
        for op in snapshot.operations:
            if op.started or _cvcr_now_s() >= s07_deadline:
                continue
            kit_tasks = [t for t in snapshot.tasks if t.kit_op_id == op.op_id]
            any_picked = any(t.status.lower() in ("picked", "loading", "to_dropoff", "completed") for t in kit_tasks)

            for crew in op.eligible_crews:
                if crew == best_crews[op.op_id] or _cvcr_now_s() >= s07_deadline:
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
                    if bay == best_bays[op.op_id] or _cvcr_now_s() >= s07_deadline:
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
            rem_s = deadline - _cvcr_now_s()
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
                rem_s = deadline - _cvcr_now_s()
                last_lat_s = (evaluator.eval_lats_ms[-1] / 1000.0) if evaluator.eval_lats_ms else 0.025
                if rem_s <= 0.0 or (best_eval.valid and rem_s < max(0.025, last_lat_s * 0.9)):
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
                if op.started or not self._has_budget(deadline, evaluator, min_ms=0.5):
                    continue
                kit_tasks = [t for t in snapshot.tasks if t.kit_op_id == op.op_id]
                any_picked = any(t.status.lower() in ("picked", "loading", "to_dropoff", "completed") for t in kit_tasks)

                for crew in op.eligible_crews:
                    if crew == best_crews[op.op_id] or not self._has_budget(deadline, evaluator, min_ms=0.5):
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
                        if bay == best_bays[op.op_id] or not self._has_budget(deadline, evaluator, min_ms=20.0):
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

    def _divert_prod_extended(
        self,
        snapshot: NativeSnapshot,
        evaluator: CVCREvaluator,
        best_assign: dict[str, str],
        best_bays: dict[str, str],
        best_crews: dict[str, int],
        best_eval: EvalResult,
        op_prio: dict[str, int],
        deadline: float,
        min_gain: float = 0.25,
    ) -> tuple[dict[str, str], dict[str, int], dict[str, int], EvalResult, int]:
        """Extended production search: 1-op crew/bay + 2-op crew swaps + critical op_prio shifts."""
        best_bays, best_crews, best_eval, cb = self._divert_bays_and_crews(
            snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, op_prio, deadline, min_gain=min_gain
        )
        cur_prio = dict(op_prio)
        eval_count = cb
        unstarted = [op for op in snapshot.operations if not op.started]

        # 2-op Crew swaps (<0.3ms per eval via B2 unchanged-route hit)
        for i in range(len(unstarted)):
            if not self._has_budget(deadline, evaluator, min_ms=0.5):
                break
            oi = unstarted[i]
            for j in range(i + 1, len(unstarted)):
                if not self._has_budget(deadline, evaluator, min_ms=0.5):
                    break
                oj = unstarted[j]
                ci, cj = best_crews[oi.op_id], best_crews[oj.op_id]
                if ci != cj and cj in oi.eligible_crews and ci in oj.eligible_crews:
                    cand_crews = dict(best_crews)
                    cand_crews[oi.op_id] = cj
                    cand_crews[oj.op_id] = ci
                    res = evaluator.evaluate(
                        best_assign,
                        op_priority=cur_prio,
                        op_bay_overrides=best_bays,
                        op_crew_overrides=cand_crews,
                    )
                    eval_count += 1
                    if res.valid and res.objective + min_gain < best_eval.objective:
                        best_eval = res
                        best_crews = cand_crews

        # Critical-chain op_priority boost on top-2 tardy unstarted operations
        tardy_ops = sorted(
            [op for op in unstarted if best_eval.op_end.get(op.op_id, 0) > op.due_step],
            key=lambda o: -(best_eval.op_end.get(o.op_id, 0) - o.due_step) * (12 if o.is_urgent else 4),
        )
        for op in tardy_ops[:2]:
            if not self._has_budget(deadline, evaluator, min_ms=8.0):
                break
            cand_prio = dict(cur_prio)
            cand_prio[op.op_id] = cur_prio[op.op_id] - 25
            res = evaluator.evaluate(
                best_assign,
                op_priority=cand_prio,
                op_bay_overrides=best_bays,
                op_crew_overrides=best_crews,
            )
            eval_count += 1
            if res.valid and res.objective + 1.0 < best_eval.objective:
                best_eval = res
                cur_prio = cand_prio
                evaluator.set_incumbent(best_assign, best_eval, op_priority=cur_prio, op_bay_overrides=best_bays)

        return best_bays, best_crews, cur_prio, best_eval, eval_count

    def _run_1opt_and_swap_ranked(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        release_tids: list[str],
        evaluator: CVCREvaluator,
        cur_assign: dict[str, str],
        best_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        op_bay_overrides: Optional[dict[str, str]] = None,
        op_crew_overrides: Optional[dict[str, int]] = None,
        allow_swaps: bool = True,
        surr: Optional[_LazyKittingSurrogate] = None,
        hys_margin: float = 0.0,
    ) -> tuple[dict[str, str], EvalResult, int]:
        """Run B2-local 1-opt & pairwise swaps with optional surrogate ordering & hysteresis margin."""
        if surr is None and hys_margin <= 0.0:
            return self._run_1opt_and_swap(
                release_tids,
                evaluator,
                cur_assign,
                best_eval,
                healthy_robots,
                op_prio,
                deadline,
                op_bay_overrides=op_bay_overrides,
                op_crew_overrides=op_crew_overrides,
                allow_swaps=allow_swaps,
            )

        snap_by_id = {t.task_id: t for t in snapshot.tasks}
        best_assign = dict(cur_assign)
        eval_count = 0

        def _req_margin(cand: dict[str, str]) -> float:
            if hys_margin <= 0.0:
                return 0.0
            has_unrescued = any(
                snap_by_id.get(tid)
                and snap_by_id[tid].assigned_agv in clf.disturbed_robots
                and best_assign.get(tid) in clf.disturbed_robots
                for tid in best_assign
            )
            if has_unrescued:
                return 0.0
            non_rescue = sum(
                1
                for tid, nr in cand.items()
                if snap_by_id.get(tid)
                and snap_by_id[tid].assigned_agv is not None
                and snap_by_id[tid].assigned_agv != nr
                and snap_by_id[tid].assigned_agv not in clf.disturbed_robots
            )
            return hys_margin if non_rescue > 0 else 0.0

        if surr is not None:
            moves: list[tuple[int, float, str, str]] = []
            for tid in release_tids:
                if tid not in best_assign:
                    continue
                cur_r = best_assign[tid]
                is_dist = 0 if (snap_by_id.get(tid) and snap_by_id[tid].assigned_agv in clf.disturbed_robots) else 1
                for nr in healthy_robots:
                    if nr == cur_r:
                        continue
                    cand = dict(best_assign)
                    cand[tid] = nr
                    s_val = surr.fast_score(cand, op_bay_overrides, op_crew_overrides)
                    moves.append((is_dist, s_val, tid, nr))
            moves.sort(key=lambda x: (x[0], x[1]))
            ordered_pairs = [(tid, nr) for _, _, tid, nr in moves]
        else:
            ordered_pairs = [
                (tid, nr)
                for tid in release_tids
                if tid in best_assign
                for nr in healthy_robots
                if nr != best_assign[tid]
            ]

        no_imp_1opt = 0
        for tid, nr in ordered_pairs:
            if not self._has_budget(deadline, evaluator):
                break
            if best_assign.get(tid) == nr:
                continue
            cand = dict(best_assign)
            cand[tid] = nr
            res = evaluator.evaluate(
                cand,
                op_priority=op_prio,
                op_bay_overrides=op_bay_overrides,
                op_crew_overrides=op_crew_overrides,
            )
            eval_count += 1
            if res.valid:
                req = _req_margin(cand)
                if not best_eval.valid or res.objective + req < best_eval.objective:
                    best_eval = res
                    best_assign = cand
                    no_imp_1opt = 0
                    evaluator.set_incumbent(best_assign, best_eval, op_priority=op_prio, op_bay_overrides=op_bay_overrides)
                elif surr is not None:
                    no_imp_1opt += 1
            elif surr is not None:
                no_imp_1opt += 1
            if surr is not None and no_imp_1opt >= 8:
                break

        if allow_swaps and self._has_budget(deadline, evaluator):
            top_k = [tid for tid in release_tids[:8] if tid in best_assign]
            swaps: list[tuple[float, str, str]] = []
            for i in range(len(top_k)):
                for j in range(i + 1, len(top_k)):
                    ti, tj = top_k[i], top_k[j]
                    if best_assign[ti] != best_assign[tj]:
                        if surr is not None:
                            cand = dict(best_assign)
                            cand[ti], cand[tj] = cand[tj], cand[ti]
                            swaps.append((surr.fast_score(cand, op_bay_overrides, op_crew_overrides), ti, tj))
                        else:
                            swaps.append((0.0, ti, tj))
            if surr is not None:
                swaps.sort(key=lambda x: x[0])
                swaps = swaps[:10]
            no_imp_swap = 0
            for _, ti, tj in swaps:
                if not self._has_budget(deadline, evaluator):
                    break
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
                if res.valid:
                    req = _req_margin(cand)
                    if not best_eval.valid or res.objective + req < best_eval.objective:
                        best_eval = res
                        best_assign = cand
                        no_imp_swap = 0
                        evaluator.set_incumbent(best_assign, best_eval, op_priority=op_prio, op_bay_overrides=op_bay_overrides)
                    elif surr is not None:
                        no_imp_swap += 1
                elif surr is not None:
                    no_imp_swap += 1
                if surr is not None and no_imp_swap >= 5:
                    break

        return best_assign, best_eval, eval_count

    def _search_alns_local_b2(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        base_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        op_bay_overrides: Optional[dict[str, str]] = None,
        op_crew_overrides: Optional[dict[str, int]] = None,
        surr: Optional[_LazyKittingSurrogate] = None,
        hys_margin: float = 0.0,
    ) -> tuple[dict[str, str], EvalResult, int]:
        """B2-locality-preserving DyMAB ALNS (changed_rids <= 2 across all 4 arms)."""
        releasable = sorted(clf.releasable_tasks)
        if not releasable or len(healthy_robots) < 2:
            return base_assign, base_eval, 0

        rng = random.Random(self.seed + snapshot.tick * 131 + 17)
        snap_by_id = {t.task_id: t for t in snapshot.tasks}
        best_assign = dict(base_assign)
        best_eval = base_eval
        cur_assign = dict(base_assign)
        eval_count = 0

        crit_ranked = sorted(
            releasable,
            key=lambda tid: (
                float(snap_by_id[tid].due_step - snapshot.tick)
                if (tid in snap_by_id and snap_by_id[tid].due_step is not None)
                else 999.0,
                tid,
            ),
        )

        n_arms = 4
        wins: list[deque[float]] = [deque(maxlen=8) for _ in range(n_arms)]
        pulls = [0] * n_arms
        total_pulls = 0
        reward_scale = 25.0

        while self._has_budget(deadline, evaluator):
            if total_pulls < n_arms:
                arm = total_pulls
            else:
                log_t = math.log(float(total_pulls + 1))
                arm = max(
                    range(n_arms),
                    key=lambda a: (
                        (sum(wins[a]) / max(1, len(wins[a]))) / reward_scale
                        + 0.75 * math.sqrt(log_t / max(1, pulls[a]))
                    ),
                )

            loads = {r: 0 for r in healthy_robots}
            for _, agv in cur_assign.items():
                if agv in loads:
                    loads[agv] += 1
            least_loaded = sorted(healthy_robots, key=lambda r: (loads[r], r))
            most_loaded = least_loaded[::-1]
            cand = dict(cur_assign)

            if arm == 0:
                pool_top = crit_ranked[: max(2, min(len(crit_ranked), 8))]
                t_a = rng.choice(pool_top)
                alts = [r for r in least_loaded[:3] if r != cand.get(t_a)] or healthy_robots
                r_b = rng.choice(alts)
                r_b_tids = [tid for tid in releasable if cand.get(tid) == r_b and tid != t_a]
                cand[t_a] = r_b
                if r_b_tids and rng.random() < 0.5:
                    t_b = rng.choice(r_b_tids)
                    old_a = cur_assign.get(t_a)
                    if old_a in healthy_robots:
                        cand[t_b] = old_a
            elif arm == 1 and surr is not None:
                r1 = most_loaded[0]
                r2_cands = [r for r in least_loaded[:3] if r != r1]
                r2 = rng.choice(r2_cands) if r2_cands else least_loaded[-1]
                sub_tids = [tid for tid in releasable if cand.get(tid) in (r1, r2)]
                if len(sub_tids) >= 2:
                    k_rem = min(len(sub_tids), rng.randint(2, min(4, len(sub_tids))))
                    rem = rng.sample(sub_tids, k_rem)
                    fixed_keep = {tid: agv for tid, agv in cand.items() if tid not in rem}
                    cand = surr.regret_k_assignment([r1, r2], fixed_keep, rem, op_bays=op_bay_overrides, rng=rng, noise=0.06)
                else:
                    t_a = rng.choice(releasable)
                    cand[t_a] = rng.choice(least_loaded[:2])
            elif arm == 2:
                hi_r, lo_r = most_loaded[0], least_loaded[0]
                hi_tids = [tid for tid in releasable if cand.get(tid) == hi_r]
                lo_tids = [tid for tid in releasable if cand.get(tid) == lo_r]
                if hi_tids and hi_r != lo_r:
                    t_hi = rng.choice(hi_tids)
                    if lo_tids and rng.random() < 0.45:
                        t_lo = rng.choice(lo_tids)
                        cand[t_hi], cand[t_lo] = cand[t_lo], cand[t_hi]
                    else:
                        cand[t_hi] = lo_r
                else:
                    cand[rng.choice(releasable)] = rng.choice(healthy_robots)
            else:
                seed_t = rng.choice(crit_ranked[: max(1, len(crit_ranked) // 2)])
                target_r = rng.choice([r for r in healthy_robots if r != cand.get(seed_t)] or healthy_robots)
                cand[seed_t] = target_r

            res = evaluator.evaluate(
                cand,
                op_priority=op_prio,
                op_bay_overrides=op_bay_overrides,
                op_crew_overrides=op_crew_overrides,
            )
            eval_count += 1
            pulls[arm] += 1
            total_pulls += 1

            gain = 0.0
            if res.valid:
                req = 0.0
                if hys_margin > 0.0:
                    non_rescue = sum(
                        1
                        for tid, nr in cand.items()
                        if snap_by_id.get(tid)
                        and snap_by_id[tid].assigned_agv is not None
                        and snap_by_id[tid].assigned_agv != nr
                        and snap_by_id[tid].assigned_agv not in clf.disturbed_robots
                    )
                    if non_rescue > 0:
                        req = hys_margin
                if not best_eval.valid or res.objective + req < best_eval.objective:
                    gain = max(0.0, float(best_eval.objective) - float(res.objective)) if best_eval.valid else 50.0
                    reward_scale = max(reward_scale, gain)
                    best_eval = res
                    best_assign = cand
                    cur_assign = cand
                    evaluator.set_incumbent(best_assign, best_eval, op_priority=op_prio, op_bay_overrides=op_bay_overrides)
            wins[arm].append(gain)

        return best_assign, best_eval, eval_count

    def _search_congestion_pacer(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        base_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        budget_ms: float,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, int], EvalResult, int, list[str], Optional[list[str]]]:
        """E4: Compact prod + congestion-spread 1-opt (hys=6.0) + conservative N-1 gate on dense blocks."""
        best_assign = dict(base_assign)
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
        best_eval = base_eval
        eval_count = 1
        cur_prio = dict(op_prio)
        proposed_admitted: Optional[list[str]] = None

        if len(healthy_robots) < 2:
            return best_assign, best_bays, best_crews, best_eval, eval_count, [], proposed_admitted

        # Compact production (25% max budget) so crew/bay flexibility isn't sacrificed
        if budget_ms >= 35.0 and _cvcr_now_s() < deadline:
            p_deadline = min(deadline, _cvcr_now_s() + max(0.008, (float(budget_ms) * 0.25) / 1000.0))
            best_bays, best_crews, cur_prio, best_eval, cb = self._divert_prod_extended(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, p_deadline
            )
            eval_count += cb

        snap_by_id = {t.task_id: t for t in snapshot.tasks}
        loads = {r: 0 for r in healthy_robots}
        for _, rid in best_assign.items():
            if rid in loads:
                loads[rid] += 1
        blocked_now = sum(1 for r in snapshot.robots if r.state in ("blocked", "parking"))

        surr = _LazyKittingSurrogate(self.network, snapshot, cur_prio, self.alpha_makespan, self.lambda_change) if budget_ms >= 35.0 else None
        rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))

        if surr is not None and rel_tids and _cvcr_now_s() < deadline:
            moves: list[tuple[int, float, str, str]] = []
            for tid in rel_tids:
                if tid not in best_assign:
                    continue
                t_obj = snap_by_id.get(tid)
                cur_r = best_assign[tid]
                is_dist = 0 if (t_obj and t_obj.assigned_agv in clf.disturbed_robots) else 1
                for nr in healthy_robots:
                    if nr == cur_r:
                        continue
                    cand = dict(best_assign)
                    cand[tid] = nr
                    base_s = surr.fast_score(cand, best_bays, best_crews)
                    crowd_tax = 8.0 * max(0, loads.get(nr, 0) - 2)
                    moves.append((is_dist, base_s + crowd_tax, tid, nr))
            moves.sort(key=lambda x: (x[0], x[1]))

            has_unrescued = any(
                snap_by_id.get(tid)
                and snap_by_id[tid].assigned_agv in clf.disturbed_robots
                and best_assign.get(tid) in clf.disturbed_robots
                for tid in best_assign
            )
            for _, _, tid, nr in moves[:16]:
                if not self._has_budget(deadline, evaluator):
                    break
                if best_assign.get(tid) == nr:
                    continue
                cand = dict(best_assign)
                cand[tid] = nr
                res = evaluator.evaluate(cand, op_priority=cur_prio, op_bay_overrides=best_bays, op_crew_overrides=best_crews)
                eval_count += 1
                if res.valid:
                    non_rescue = (
                        snap_by_id.get(tid) is not None
                        and snap_by_id[tid].assigned_agv is not None
                        and snap_by_id[tid].assigned_agv not in clf.disturbed_robots
                    )
                    req = 0.0 if (has_unrescued or not non_rescue) else 6.0
                    if not best_eval.valid or res.objective + req < best_eval.objective:
                        best_eval = res
                        best_assign = cand
                        evaluator.set_incumbent(best_assign, best_eval, op_priority=cur_prio, op_bay_overrides=best_bays)
        elif rel_tids and _cvcr_now_s() < deadline:
            best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                op_bay_overrides=best_bays, op_crew_overrides=best_crews, allow_swaps=False, hys_margin=6.0,
            )
            eval_count += c1

        if blocked_now >= 2 and len(healthy_robots) >= 5 and _cvcr_now_s() + 0.015 < deadline:
            mandatory = {
                t.assigned_agv
                for t in snapshot.tasks
                if t.assigned_agv in healthy_robots and t.status.lower() in ("picked", "loading", "to_dropoff")
            }
            if len(mandatory) <= len(healthy_robots) - 1:
                least_busy = min([r for r in healthy_robots if r not in mandatory], key=lambda r: (loads.get(r, 0), r), default=None)
                if least_busy is not None:
                    subset = [r for r in healthy_robots if r != least_busy]
                    cand = dict(best_assign)
                    for tid, rid in list(cand.items()):
                        if rid == least_busy and tid in clf.releasable_tasks:
                            cand[tid] = min(subset, key=lambda r: (loads.get(r, 0), r))
                    res = evaluator.evaluate(cand, op_priority=cur_prio, op_bay_overrides=best_bays, op_crew_overrides=best_crews)
                    eval_count += 1
                    if res.valid and res.objective + 8.0 < best_eval.objective:
                        best_eval = res
                        best_assign = cand
                        proposed_admitted = subset

        return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, proposed_admitted

    def _search_expert_portfolio(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        base_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        budget_ms: float,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, int], EvalResult, int, list[str], Optional[list[str]]]:
        """Dispatch one of the 5 complementary CVCR experts (EXP_E1..EXP_E5)."""
        best_assign = dict(base_assign)
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
        best_eval = base_eval
        eval_count = 1
        cur_prio = dict(op_prio)

        if len(healthy_robots) < 2:
            return best_assign, best_bays, best_crews, best_eval, eval_count, [], None

        if self.method_id == "EXP_E1_FAST_RESCUE":
            rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))[:6]
            if rel_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                    snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    allow_swaps=False, hys_margin=6.0,
                )
                eval_count += c1
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None

        if self.method_id == "EXP_E2_PROD_FOCUSED":
            if budget_ms >= 35.0 and _cvcr_now_s() < deadline:
                prod_deadline = min(deadline, _cvcr_now_s() + max(0.012, (float(budget_ms) * 0.60) / 1000.0))
                best_bays, best_crews, cur_prio, best_eval, cb = self._divert_prod_extended(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, prod_deadline
                )
                eval_count += cb
            rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
            snap_by_id = {t.task_id: t for t in snapshot.tasks}
            changed_ops = {
                op.op_id for op in snapshot.operations
                if best_bays[op.op_id] != op.bay_id or best_crews[op.op_id] != op.crew_id
            }
            focused_tids = [
                tid for tid in rel_tids
                if (snap_by_id.get(tid) and snap_by_id[tid].assigned_agv in clf.disturbed_robots)
                or (snap_by_id.get(tid) and snap_by_id[tid].kit_op_id in changed_ops)
                or (snap_by_id.get(tid) and snap_by_id[tid].is_urgent)
            ] or rel_tids[:4]
            if focused_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                    snapshot, clf, focused_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    op_bay_overrides=best_bays, op_crew_overrides=best_crews, allow_swaps=False, hys_margin=5.0,
                )
                eval_count += c1
            return best_assign, best_bays, best_crews, best_eval, eval_count, focused_tids, None

        if self.method_id == "EXP_E3_JOINT_CORE":
            surr = _LazyKittingSurrogate(self.network, snapshot, cur_prio, self.alpha_makespan, self.lambda_change) if budget_ms >= 35.0 else None
            if budget_ms >= 35.0 and _cvcr_now_s() < deadline:
                bay_deadline = min(deadline, _cvcr_now_s() + max(0.010, (float(budget_ms) * 0.40) / 1000.0))
                best_bays, best_crews, cur_prio, best_eval, cb = self._divert_prod_extended(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, bay_deadline
                )
                eval_count += cb
            rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
            if rel_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap(
                    rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    op_bay_overrides=best_bays, op_crew_overrides=best_crews, allow_swaps=(budget_ms >= 35.0),
                )
                eval_count += c1
            if budget_ms >= 85.0 and _cvcr_now_s() < deadline:
                best_assign, best_eval, ca = self._search_alns_local_b2(
                    snapshot, clf, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    op_bay_overrides=best_bays, op_crew_overrides=best_crews, surr=surr, hys_margin=0.0,
                )
                eval_count += ca
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None

        if self.method_id == "EXP_E4_CONGESTION_PACER":
            return self._search_congestion_pacer(
                snapshot, clf, evaluator, base_assign, base_eval, healthy_robots, cur_prio, deadline, budget_ms
            )

        # EXP_E5_AGGRESSIVE_GRID: Wide surrogate-ranked 1-opt/swap + 2-Robot B2 Regret LNS
        surr = _LazyKittingSurrogate(self.network, snapshot, cur_prio, self.alpha_makespan, self.lambda_change) if budget_ms >= 35.0 else None
        if budget_ms >= 35.0 and _cvcr_now_s() < deadline:
            bay_deadline = min(deadline, _cvcr_now_s() + max(0.010, (float(budget_ms) * 0.35) / 1000.0))
            best_bays, best_crews, cur_prio, best_eval, cb = self._divert_prod_extended(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, bay_deadline
            )
            eval_count += cb
        rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
        if rel_tids and _cvcr_now_s() < deadline:
            best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                op_bay_overrides=best_bays, op_crew_overrides=best_crews, allow_swaps=(budget_ms >= 35.0),
                surr=surr, hys_margin=0.0,
            )
            eval_count += c1
        if budget_ms >= 85.0 and _cvcr_now_s() < deadline:
            best_assign, best_eval, ca = self._search_alns_local_b2(
                snapshot, clf, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                op_bay_overrides=best_bays, op_crew_overrides=best_crews, surr=surr, hys_margin=0.0,
            )
            eval_count += ca
        return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None

    @staticmethod
    def _order_commit_by_ci(
        commit_assign: dict[str, str],
        best_assign: dict[str, str],
        best_eval: EvalResult,
        include_heads: bool = True,
    ) -> dict[str, str]:
        """Order commit_assign interleaved by per-AGV c_i queue so idle AGVs pick #1 first."""
        ordered_tids: list[str] = []
        seen_tids: set[str] = set()
        seqs = [best_eval.robot_order_seq.get(rid, []) for rid in sorted(best_eval.robot_order_seq.keys())]
        max_len = max((len(s) for s in seqs), default=0)
        for idx in range(max_len):
            for s in seqs:
                if idx < len(s) and s[idx] in best_assign and s[idx] not in seen_tids:
                    ordered_tids.append(s[idx])
                    seen_tids.add(s[idx])
        for tid in best_assign:
            if tid not in seen_tids:
                ordered_tids.append(tid)
        out = {tid: commit_assign[tid] for tid in ordered_tids if tid in commit_assign}
        if include_heads:
            for rid, seq in sorted(best_eval.robot_order_seq.items()):
                if seq:
                    head_tid = seq[0]
                    if head_tid in best_assign and head_tid not in out:
                        out[head_tid] = best_assign[head_tid]
        return out

    def _divert_prod_v6(
        self,
        snapshot: NativeSnapshot,
        evaluator: CVCREvaluator,
        best_assign: dict[str, str],
        best_bays: dict[str, str],
        best_crews: dict[str, int],
        best_eval: EvalResult,
        op_prio: dict[str, int],
        deadline: float,
        min_gain: float = 0.25,
    ) -> tuple[dict[str, str], dict[str, int], dict[str, int], EvalResult, int]:
        """Critical-path-ordered production search (1-op/2-op crew -> critical bays -> disjunctive prio fix)."""
        crit = compute_hybrid_criticality(
            snapshot, best_eval, best_bays, best_crews, alpha_makespan=self.alpha_makespan
        )
        op_w = crit["op_weight"]
        cur_prio = dict(op_prio)
        unstarted = sorted(
            [op for op in snapshot.operations if not op.started],
            key=lambda o: (-op_w.get(o.op_id, 0.0), o.due_step, o.op_id),
        )
        eval_count = 0

        # 1. Zero-reroute B2 1-op crew diversions sorted by critical path weight
        for op in unstarted:
            if not self._has_budget(deadline, evaluator, min_ms=0.5):
                break
            cur_c = best_crews[op.op_id]
            for alt_c in op.eligible_crews:
                if alt_c == cur_c or not self._has_budget(deadline, evaluator, min_ms=0.5):
                    continue
                cand_crews = dict(best_crews)
                cand_crews[op.op_id] = alt_c
                res = evaluator.evaluate(
                    best_assign, op_priority=cur_prio, op_bay_overrides=best_bays, op_crew_overrides=cand_crews
                )
                eval_count += 1
                if res.valid and res.objective + min_gain < best_eval.objective:
                    best_eval = res
                    best_crews = cand_crews

        # 2. Critical 2-op crew swaps (<0.3ms each) before slower routing moves
        crit_subset = unstarted[:8]
        for i in range(len(crit_subset)):
            if not self._has_budget(deadline, evaluator, min_ms=0.5):
                break
            oi = crit_subset[i]
            for j in range(i + 1, len(unstarted)):
                if not self._has_budget(deadline, evaluator, min_ms=0.5):
                    break
                oj = unstarted[j]
                ci, cj = best_crews[oi.op_id], best_crews[oj.op_id]
                if ci != cj and cj in oi.eligible_crews and ci in oj.eligible_crews:
                    cand_crews = dict(best_crews)
                    cand_crews[oi.op_id] = cj
                    cand_crews[oj.op_id] = ci
                    res = evaluator.evaluate(
                        best_assign, op_priority=cur_prio, op_bay_overrides=best_bays, op_crew_overrides=cand_crews
                    )
                    eval_count += 1
                    if res.valid and res.objective + min_gain < best_eval.objective:
                        best_eval = res
                        best_crews = cand_crews

        # 3. Critical-first 1-op bay diversion
        for op in unstarted:
            if not self._has_budget(deadline, evaluator, min_ms=5.0):
                break
            cur_b = best_bays[op.op_id]
            for alt_b in op.eligible_bays:
                if alt_b == cur_b or not self._has_budget(deadline, evaluator, min_ms=5.0):
                    continue
                cand_bays = dict(best_bays)
                cand_bays[op.op_id] = alt_b
                res = evaluator.evaluate(
                    best_assign, op_priority=cur_prio, op_bay_overrides=cand_bays, op_crew_overrides=best_crews
                )
                eval_count += 1
                if res.valid and res.objective + min_gain < best_eval.objective:
                    best_eval = res
                    best_bays = cand_bays
                    evaluator.set_incumbent(best_assign, best_eval, op_priority=cur_prio, op_bay_overrides=best_bays)

        # 4. Exact disjunctive blocker priority inversion on top-2 tardy critical ops (B1 fallback -> tail of budget)
        crit2 = compute_hybrid_criticality(
            snapshot, best_eval, best_bays, best_crews, alpha_makespan=self.alpha_makespan
        )
        tardy_crit = sorted(
            [op for op in unstarted if best_eval.op_end.get(op.op_id, 0) > op.due_step],
            key=lambda o: (-crit2["op_weight"].get(o.op_id, 0.0), o.due_step),
        )
        for op in tardy_crit[:2]:
            if not self._has_budget(deadline, evaluator, min_ms=8.0):
                break
            cand_prio = dict(cur_prio)
            shifted = False
            for blk_map in (crit2["blocked_by_bay"], crit2["blocked_by_crew"], crit2["blocked_by_tool"]):
                prev_oid = blk_map.get(op.op_id)
                if prev_oid and prev_oid in cur_prio and cur_prio.get(op.op_id, 0) >= cur_prio[prev_oid]:
                    cand_prio[op.op_id] = cur_prio[prev_oid] - 1
                    shifted = True
                    break
            if not shifted:
                cand_prio[op.op_id] = cur_prio[op.op_id] - 25
            res = evaluator.evaluate(
                best_assign, op_priority=cand_prio, op_bay_overrides=best_bays, op_crew_overrides=best_crews
            )
            eval_count += 1
            if res.valid and res.objective + 1.0 < best_eval.objective:
                best_eval = res
                cur_prio = cand_prio
                evaluator.set_incumbent(best_assign, best_eval, op_priority=cur_prio, op_bay_overrides=best_bays)

        return best_bays, best_crews, cur_prio, best_eval, eval_count

    def _search_intra_agent_ci(
        self,
        snapshot: NativeSnapshot,
        evaluator: CVCREvaluator,
        best_assign: dict[str, str],
        best_bays: dict[str, str],
        best_crews: dict[str, int],
        best_eval: EvalResult,
        cur_prio: dict[str, int],
        deadline: float,
        surr: Optional[_LazyKittingSurrogate] = None,
        max_candidates: int = 10,
    ) -> tuple[dict[str, list[str]], EvalResult, int]:
        """Single-robot (changed_rids=1) intra-agent task sequencing c_i with zero churn tax."""
        snap_by_id = {t.task_id: t for t in snapshot.tasks}
        rob_by_id = {r.robot_id: r for r in snapshot.robots}
        crit = compute_hybrid_criticality(
            snapshot, best_eval, best_bays, best_crews, alpha_makespan=self.alpha_makespan
        )
        t_weight = crit["task_weight"]
        k_slack = crit["kitting_slack"]

        best_orders: dict[str, list[str]] = {
            rid: list(seq) for rid, seq in best_eval.robot_order_seq.items() if len(seq) >= 2
        }
        cand_moves: list[tuple[float, str, list[str]]] = []

        for rid, seq in best_orders.items():
            locked = [
                tid
                for tid in seq
                if snap_by_id.get(tid)
                and snap_by_id[tid].status.lower() in ("picked", "loading", "to_dropoff", "unloading")
            ]
            free_q = [
                tid for tid in seq
                if tid in snap_by_id
                and tid not in locked
                and snap_by_id[tid].status.lower() != "completed"
            ]
            if len(free_q) < 2:
                continue

            rob = rob_by_id.get(rid)
            start_n = rob.cur_node if rob else snap_by_id[free_q[0]].pickup

            for i in range(len(free_q)):
                for j in range(i + 1, min(len(free_q), i + 4)):
                    ti, tj = free_q[i], free_q[j]
                    si = k_slack.get(ti, 0)
                    sj = k_slack.get(tj, 0)
                    wi = t_weight.get(ti, 0.0)
                    wj = t_weight.get(tj, 0.0)
                    slack_gain = max(0.0, float(si - max(0, sj))) * 0.4 + max(0.0, wj - wi)

                    dist_gain = 0.0
                    if surr is not None and j == i + 1:
                        ti_obj, tj_obj = snap_by_id[ti], snap_by_id[tj]
                        if i == 0:
                            prev_n = start_n
                        else:
                            prev_t = snap_by_id[free_q[i - 1]]
                            prev_n = best_bays.get(prev_t.kit_op_id or "", prev_t.dropoff)
                        di = best_bays.get(ti_obj.kit_op_id or "", ti_obj.dropoff)
                        dj = best_bays.get(tj_obj.kit_op_id or "", tj_obj.dropoff)
                        c_orig = surr.tt(prev_n, ti_obj.pickup) + surr.tt(di, tj_obj.pickup)
                        c_swap = surr.tt(prev_n, tj_obj.pickup) + surr.tt(dj, ti_obj.pickup)
                        dist_gain = c_orig - c_swap
                        if i + 2 < len(free_q):
                            tk_obj = snap_by_id[free_q[i + 2]]
                            dist_gain += surr.tt(dj, tk_obj.pickup) - surr.tt(di, tk_obj.pickup)

                    score = slack_gain + max(0.0, dist_gain)
                    if score >= 1.5:
                        new_free = list(free_q)
                        new_free.pop(j)
                        new_free.insert(i, tj)
                        cand_moves.append((score, rid, locked + new_free))

        cand_moves.sort(key=lambda x: -x[0])
        eval_count = 0
        for _, rid, new_seq in cand_moves[:max_candidates]:
            if not self._has_budget(deadline, evaluator, min_ms=2.0):
                break
            cand_orders = dict(best_orders)
            cand_orders[rid] = new_seq
            res = evaluator.evaluate(
                best_assign,
                op_priority=cur_prio,
                op_bay_overrides=best_bays,
                op_crew_overrides=best_crews,
                task_order_override=cand_orders,
            )
            eval_count += 1
            if res.valid and res.objective + 0.5 < best_eval.objective:
                best_eval = res
                best_orders = cand_orders
                evaluator.set_incumbent(
                    best_assign,
                    best_eval,
                    op_priority=cur_prio,
                    op_bay_overrides=best_bays,
                    task_order_override=best_orders,
                )
        return best_orders, best_eval, eval_count

    def _search_v6_expert_portfolio(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        base_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        budget_ms: float,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, int], EvalResult, int, list[str], Optional[list[str]]]:
        """Dispatch V6 Critical-Path + Intra-Agent c_i Sequencing Experts (V6_E1..V6_E5)."""
        best_assign = dict(base_assign)
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
        best_eval = base_eval
        eval_count = 1
        cur_prio = dict(op_prio)
        best_orders: dict[str, list[str]] = {}

        if len(healthy_robots) < 2:
            return best_assign, best_bays, best_crews, best_eval, eval_count, [], None

        surr = (
            _LazyKittingSurrogate(self.network, snapshot, cur_prio, self.alpha_makespan, self.lambda_change)
            if budget_ms >= 35.0
            else None
        )

        if self.method_id in ("V6_E1_FAST_RESCUE", "V7_E1_FAST_RESCUE"):
            rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
            if rel_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                    snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    allow_swaps=False, hys_margin=5.0,
                )
                eval_count += c1
            if _cvcr_now_s() + 0.002 < deadline:
                _, best_eval, c_ci = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, max_candidates=8,
                )
                eval_count += c_ci
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None

        if self.method_id in ("V6_E2_CRIT_PROD_CI", "V7_E2_CRIT_PROD_CI"):
            if budget_ms >= 35.0 and _cvcr_now_s() < deadline:
                prod_deadline = min(deadline, _cvcr_now_s() + max(0.012, (float(budget_ms) * 0.58) / 1000.0))
                best_bays, best_crews, cur_prio, best_eval, cb = self._divert_prod_v6(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, prod_deadline
                )
                eval_count += cb
            rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
            snap_by_id = {t.task_id: t for t in snapshot.tasks}
            changed_ops = {
                op.op_id for op in snapshot.operations
                if best_bays[op.op_id] != op.bay_id or best_crews[op.op_id] != op.crew_id
            }
            focused_tids = [
                tid for tid in rel_tids
                if (snap_by_id.get(tid) and snap_by_id[tid].assigned_agv in clf.disturbed_robots)
                or (snap_by_id.get(tid) and snap_by_id[tid].kit_op_id in changed_ops)
                or (snap_by_id.get(tid) and snap_by_id[tid].is_urgent)
            ] or rel_tids[:4]
            if focused_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                    snapshot, clf, focused_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    op_bay_overrides=best_bays, op_crew_overrides=best_crews, allow_swaps=False, hys_margin=5.0,
                )
                eval_count += c1
            if _cvcr_now_s() + 0.002 < deadline:
                _, best_eval, c_ci = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, max_candidates=8,
                )
                eval_count += c_ci
            return best_assign, best_bays, best_crews, best_eval, eval_count, focused_tids, None

        if self.method_id in ("V6_E4_PACER_CI", "V7_E4_PACER_CI"):
            best_assign, best_bays, best_crews, best_eval, c4, rel_tids, admitted = self._search_congestion_pacer(
                snapshot, clf, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline, budget_ms
            )
            eval_count += max(0, c4 - 1)
            if _cvcr_now_s() + 0.002 < deadline:
                _, best_eval, c_ci = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, max_candidates=6,
                )
                eval_count += c_ci
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, admitted

        if self.method_id == "V7_E6_BALANCE_KM":
            if budget_ms >= 35.0 and _cvcr_now_s() < deadline:
                bay_deadline = min(deadline, _cvcr_now_s() + max(0.010, (float(budget_ms) * 0.35) / 1000.0))
                best_bays, best_crews, cur_prio, best_eval, cb = self._divert_prod_extended(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, bay_deadline
                )
                eval_count += cb
            rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
            if rel_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                    snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    op_bay_overrides=best_bays, op_crew_overrides=best_crews,
                    allow_swaps=(budget_ms >= 35.0), surr=surr, hys_margin=0.0,
                )
                eval_count += c1
            if _cvcr_now_s() + 0.003 < deadline:
                ci_deadline = min(deadline, _cvcr_now_s() + max(0.012, (float(budget_ms) * 0.22) / 1000.0))
                _, best_eval, c_ci = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, ci_deadline,
                    surr=surr, max_candidates=10,
                )
                eval_count += c_ci
            if budget_ms >= 85.0 and surr is not None and _cvcr_now_s() < deadline:
                best_assign, best_eval, ca = self._search_balance_km_sa(
                    snapshot, clf, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    best_bays, best_crews, surr, hys_margin=0.0,
                )
                eval_count += ca
            if _cvcr_now_s() + 0.003 < deadline:
                _, best_eval, c_ci2 = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, max_candidates=6,
                )
                eval_count += c_ci2
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None

        # V6/V7 E3 (hys=5.0) vs V6/V7 E5 (hys=0.0)
        is_e3 = self.method_id in ("V6_E3_JOINT_CI_HYS", "V7_E3_JOINT_CI_HYS")
        hys = 5.0 if is_e3 else 0.0
        if budget_ms >= 35.0 and _cvcr_now_s() < deadline:
            bay_frac = 0.40 if is_e3 else 0.35
            bay_deadline = min(deadline, _cvcr_now_s() + max(0.010, (float(budget_ms) * bay_frac) / 1000.0))
            best_bays, best_crews, cur_prio, best_eval, cb = self._divert_prod_extended(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, bay_deadline
            )
            eval_count += cb
        rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
        if rel_tids and _cvcr_now_s() < deadline:
            best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                op_bay_overrides=best_bays, op_crew_overrides=best_crews,
                allow_swaps=(budget_ms >= 35.0), surr=surr, hys_margin=hys,
            )
            eval_count += c1
        if _cvcr_now_s() + 0.003 < deadline:
            ci_deadline = min(deadline, _cvcr_now_s() + max(0.012, (float(budget_ms) * 0.22) / 1000.0))
            best_orders, best_eval, c_ci = self._search_intra_agent_ci(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, ci_deadline,
                surr=surr, max_candidates=10,
            )
            eval_count += c_ci
        if budget_ms >= 85.0 and _cvcr_now_s() < deadline:
            best_assign, best_eval, ca = self._search_alns_local_b2(
                snapshot, clf, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                op_bay_overrides=best_bays, op_crew_overrides=best_crews, surr=surr, hys_margin=hys,
            )
            eval_count += ca
        return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None

    def _build_km_assignment(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        healthy_robots: list[str],
        base_assign: dict[str, str],
        cur_eval: EvalResult,
        bays: dict[str, str],
        crews: dict[str, int],
        surr: _LazyKittingSurrogate,
        target_tids: list[str],
    ) -> Optional[dict[str, str]]:
        """ALNS-KM bipartite Kuhn-Munkres matcher over last-kit bottleneck tasks."""
        if not target_tids or len(healthy_robots) < 2:
            return None
        snap_by_id = {t.task_id: t for t in snapshot.tasks}
        op_by_id = {op.op_id: op for op in snapshot.operations}
        rob_by_id = {r.robot_id: r for r in snapshot.robots}
        remove_set = {tid for tid in target_tids if tid in clf.releasable_tasks and tid in snap_by_id}
        if not remove_set:
            return None

        rob_free: dict[str, float] = {}
        rob_node: dict[str, str] = {}
        rob_fixed_cnt: dict[str, int] = {}
        for rid in healthy_robots:
            rob = rob_by_id[rid]
            cur_t = float(max(snapshot.tick, rob.faulted_until))
            cur_n = rob.cur_node
            fixed_cnt = 0
            for tid in cur_eval.robot_order_seq.get(rid, []):
                if tid in remove_set:
                    continue
                t = snap_by_id.get(tid)
                if t is None or t.status.lower() == "completed":
                    continue
                drop = bays.get(t.kit_op_id or "", t.dropoff) if t.status.lower() in ("pending", "assigned") else t.dropoff
                if t.status.lower() in ("picked", "loading", "to_dropoff", "unloading"):
                    cur_t += surr.tt(cur_n, drop) + 2.0
                else:
                    tp = max(cur_t + surr.tt(cur_n, t.pickup), float(t.release_step)) + 2.0
                    cur_t = tp + surr.tt(t.pickup, drop) + 2.0
                cur_n = drop
                fixed_cnt += 1
            rob_free[rid] = cur_t
            rob_node[rid] = cur_n
            rob_fixed_cnt[rid] = fixed_cnt

        op_fixed_mat: dict[str, float] = {}
        for op in snapshot.operations:
            mx = float(snapshot.tick)
            for ktid in op.kit_task_ids:
                if ktid not in remove_set:
                    mx = max(mx, float(cur_eval.order_unload_ticks.get(ktid, snapshot.tick)))
            op_fixed_mat[op.op_id] = mx

        cand_assign = dict(base_assign)
        pool_tids = sorted(
            remove_set,
            key=lambda tid: (
                float(op_by_id[snap_by_id[tid].kit_op_id].due_step - op_by_id[snap_by_id[tid].kit_op_id].proc_steps)
                if (snap_by_id[tid].kit_op_id and snap_by_id[tid].kit_op_id in op_by_id)
                else float(snap_by_id[tid].due_step or 999),
                tid,
            ),
        )
        v_list = list(healthy_robots)
        n_v = len(v_list)
        idx = 0
        while idx < len(pool_tids):
            batch = pool_tids[idx : idx + n_v]
            idx += n_v
            m = len(batch)
            cost = np.zeros((n_v, m), dtype=np.float64)
            for r_i, rid in enumerate(v_list):
                for c_j, tid in enumerate(batch):
                    t = snap_by_id[tid]
                    drop = bays.get(t.kit_op_id or "", t.dropoff)
                    tp = max(rob_free[rid] + surr.tt(rob_node[rid], t.pickup), float(t.release_step)) + 2.0
                    comp = tp + surr.tt(t.pickup, drop) + surr.bay_delay(drop) + 2.0
                    chg = self.lambda_change if (t.assigned_agv is not None and t.assigned_agv != rid and t.status.lower() == "assigned") else 0.0
                    tard_pen = 0.0
                    if t.kit_op_id and t.kit_op_id in op_by_id:
                        op = op_by_id[t.kit_op_id]
                        due_eff = float(op.due_step - op.proc_steps)
                        op_w = 12.0 if op.is_urgent else 4.0
                        wait_sib = op_fixed_mat.get(op.op_id, 0.0)
                        tard_pen = (op_w if comp >= wait_sib - 2.0 else 0.35 * op_w) * max(0.0, comp - due_eff)
                    elif t.due_step is not None:
                        tard_pen = 3.0 * max(0.0, comp - float(t.due_step))
                    cost[r_i, c_j] = comp + tard_pen + chg + 2.0 * rob_fixed_cnt[rid]

            row_ind, col_ind = linear_sum_assignment(cost)
            for r_i, c_j in zip(row_ind.tolist(), col_ind.tolist()):
                rid = v_list[r_i]
                tid = batch[c_j]
                cand_assign[tid] = rid
                t = snap_by_id[tid]
                drop = bays.get(t.kit_op_id or "", t.dropoff)
                tp = max(rob_free[rid] + surr.tt(rob_node[rid], t.pickup), float(t.release_step)) + 2.0
                rob_free[rid] = tp + surr.tt(t.pickup, drop) + 2.0
                rob_node[rid] = drop
                rob_fixed_cnt[rid] += 1

        return None if cand_assign == base_assign else cand_assign

    def _search_balance_km_sa(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        best_assign: dict[str, str],
        best_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        op_bay_overrides: dict[str, str],
        op_crew_overrides: dict[str, int],
        surr: _LazyKittingSurrogate,
        hys_margin: float = 0.0,
    ) -> tuple[dict[str, str], EvalResult, int]:
        """Dual-mode BALANCE Thompson LNS (N in {2,3,KM}) + Last-Kit Cluster + Metropolis SA."""
        releasable = sorted(clf.releasable_tasks)
        if len(releasable) < 2 or len(healthy_robots) < 2:
            return best_assign, best_eval, 0

        rng = random.Random(self.seed + snapshot.tick * 197 + 31)
        snap_by_id = {t.task_id: t for t in snapshot.tasks}
        best_a = dict(best_assign)
        best_e = best_eval
        cur_a = dict(best_assign)
        cur_e = best_eval
        eval_count = 0

        n_H, n_N = 4, 3
        h_cnt = [0] * n_H
        h_sum = [0.0] * n_H
        h_ss = [0.0] * n_H
        n_cnt = [[0] * n_N for _ in range(n_H)]
        n_sum = [[0.0] * n_N for _ in range(n_H)]
        n_ss = [[0.0] * n_N for _ in range(n_H)]

        def _sample_ng(cnt: int, s: float, ss: float) -> float:
            if cnt == 0:
                return 2.0 + rng.random()
            lam = 1.0 + cnt
            mu = s / lam
            alpha = 1.0 + 0.5 * cnt
            mean_x = s / cnt
            var_part = max(0.0, ss - cnt * (mean_x * mean_x))
            beta = 1.0 + 0.5 * var_part + 0.5 * (cnt / lam) * (mean_x * mean_x)
            tau = rng.gammavariate(alpha, 1.0 / max(1e-6, beta))
            sigma = 1.0 / math.sqrt(max(1e-6, lam * tau))
            return rng.gauss(mu, min(sigma, 15.0))

        temp = 14.0
        has_unrescued = any(
            snap_by_id.get(tid) and snap_by_id[tid].assigned_agv in clf.disturbed_robots and cur_a.get(tid) in clf.disturbed_robots
            for tid in cur_a
        )

        # Conditional upfront ALNS-KM seed ONLY when residual tardiness pressure > 60
        res_tard0 = sum(
            (12.0 if op.is_urgent else 4.0) * max(0, cur_e.op_end.get(op.op_id, 0) - op.due_step)
            for op in snapshot.operations
        )
        if res_tard0 >= 60.0 and self._has_budget(deadline, evaluator, min_ms=16.0):
            crit0 = compute_hybrid_criticality(snapshot, cur_e, op_bay_overrides, op_crew_overrides, self.alpha_makespan)
            tw0 = crit0["task_weight"]
            last_kit_pool = sorted(
                releasable,
                key=lambda tid: (-tw0.get(tid, 0.0), crit0["kitting_slack"].get(tid, 999), tid),
            )[: max(3, min(len(healthy_robots) + 2, len(releasable)))]
            km_cand = self._build_km_assignment(
                snapshot, clf, healthy_robots, cur_a, cur_e, op_bay_overrides, op_crew_overrides, surr, last_kit_pool
            )
            if km_cand is not None:
                res_km = evaluator.evaluate(
                    km_cand, op_priority=op_prio, op_bay_overrides=op_bay_overrides, op_crew_overrides=op_crew_overrides
                )
                eval_count += 1
                if res_km.valid and res_km.objective + hys_margin < best_e.objective:
                    best_e = res_km
                    best_a = km_cand
                    cur_e = res_km
                    cur_a = dict(km_cand)
                    evaluator.set_incumbent(best_a, best_e, op_priority=op_prio, op_bay_overrides=op_bay_overrides)

        seen_cands: set[tuple[tuple[str, str], ...]] = {
            tuple(sorted(best_a.items())),
            tuple(sorted(cur_a.items())),
        }
        no_improve_streak = 0
        dup_streak = 0
        bal_evals = 0
        max_bal_evals = 16

        while self._has_budget(deadline, evaluator, min_ms=6.0):
            if bal_evals >= max_bal_evals or no_improve_streak >= 6 or dup_streak >= 8:
                break
            h_arm = max(range(n_H), key=lambda i: _sample_ng(h_cnt[i], h_sum[i], h_ss[i]))
            n_arm = max(range(n_N), key=lambda j: _sample_ng(n_cnt[h_arm][j], n_sum[h_arm][j], n_ss[h_arm][j]))
            rem_ms = (deadline - _cvcr_now_s()) * 1000.0
            res_tard = sum(
                (12.0 if op.is_urgent else 4.0) * max(0, cur_e.op_end.get(op.op_id, 0) - op.due_step)
                for op in snapshot.operations
            )
            if rem_ms < 16.0 or res_tard < 40.0:
                n_arm = 0

            crit = compute_hybrid_criticality(snapshot, cur_e, op_bay_overrides, op_crew_overrides, self.alpha_makespan)
            tw = crit["task_weight"]
            ks = crit["kitting_slack"]

            cand: Optional[dict[str, str]] = None
            target_r_cap = 2 if n_arm == 0 else (3 if n_arm == 1 else min(len(healthy_robots) - 1, 4))

            if h_arm == 0:
                last_kits = sorted(releasable, key=lambda tid: (-tw.get(tid, 0.0), abs(ks.get(tid, 0)), tid))
                if n_arm == 2 and len(last_kits) >= 3:
                    cand = self._build_km_assignment(
                        snapshot, clf, healthy_robots, cur_a, cur_e, op_bay_overrides, op_crew_overrides,
                        surr, last_kits[: min(len(healthy_robots), len(last_kits))],
                    )
                else:
                    seed_tid = last_kits[rng.randrange(min(4, len(last_kits)))]
                    chosen_r = rng.sample(healthy_robots, min(target_r_cap, len(healthy_robots)))
                    sub = [tid for tid in last_kits if cur_a.get(tid) in chosen_r or tid == seed_tid][: (target_r_cap + 2)]
                    if sub:
                        sub_rids = sorted(set(chosen_r) | ({cur_a[seed_tid]} if cur_a.get(seed_tid) in healthy_robots else set()))[:target_r_cap]
                        fixed = {tid: rid for tid, rid in cur_a.items() if tid not in sub}
                        cand = surr.regret_k_assignment(sub_rids, fixed, sub, op_bays=op_bay_overrides, rng=rng, noise=0.05)
            elif h_arm == 1:
                ranked = sorted(releasable, key=lambda tid: (-tw.get(tid, 0.0), float(snap_by_id[tid].due_step or 999), tid))
                sub_rids = rng.sample(healthy_robots, min(target_r_cap, len(healthy_robots)))
                sub_tids = [tid for tid in ranked if cur_a.get(tid) in sub_rids]
                if len(sub_tids) >= 2:
                    if n_arm == 0:
                        t1 = sub_tids[0]
                        cand_r2 = [tid for tid in sub_tids if cur_a.get(tid) != cur_a.get(t1)]
                        if cand_r2:
                            t2 = rng.choice(cand_r2)
                            cand = dict(cur_a)
                            cand[t1], cand[t2] = cand[t2], cand[t1]
                    else:
                        cand = self._build_km_assignment(
                            snapshot, clf, sub_rids, cur_a, cur_e, op_bay_overrides, op_crew_overrides,
                            surr, sub_tids[: min(len(sub_tids), target_r_cap + 1)],
                        )
            elif h_arm == 2:
                sub_rids = rng.sample(healthy_robots, min(target_r_cap, len(healthy_robots)))
                sub_tids = [tid for tid in releasable if cur_a.get(tid) in sub_rids]
                if sub_tids:
                    anchor = snap_by_id[rng.choice(sub_tids)]
                    sub_tids.sort(key=lambda tid: surr.tt(anchor.pickup, snap_by_id[tid].pickup))
                    rem_k = sub_tids[: min(len(sub_tids), target_r_cap + 2)]
                    fixed = {tid: rid for tid, rid in cur_a.items() if tid not in rem_k}
                    cand = surr.regret_k_assignment(sub_rids, fixed, rem_k, op_bays=op_bay_overrides, rng=rng, noise=0.08)
            else:
                sub_rids = rng.sample(healthy_robots, min(target_r_cap, len(healthy_robots)))
                picks = []
                for rid in sub_rids:
                    rtids = [tid for tid in releasable if cur_a.get(tid) == rid]
                    if rtids:
                        picks.append(rng.choice(rtids))
                if len(picks) >= 2:
                    cand = dict(cur_a)
                    owners = [cur_a[tid] for tid in picks]
                    for idx_p, tid in enumerate(picks):
                        cand[tid] = owners[(idx_p + 1) % len(picks)]

            if cand is None or cand == cur_a:
                h_cnt[h_arm] += 1
                n_cnt[h_arm][n_arm] += 1
                dup_streak += 1
                temp *= 0.85
                if dup_streak >= 8:
                    break
                continue
            cand_key = tuple(sorted(cand.items()))
            if cand_key in seen_cands:
                h_cnt[h_arm] += 1
                n_cnt[h_arm][n_arm] += 1
                dup_streak += 1
                temp *= 0.85
                if dup_streak >= 8:
                    break
                continue
            seen_cands.add(cand_key)
            dup_streak = 0

            res = evaluator.evaluate(cand, op_priority=op_prio, op_bay_overrides=op_bay_overrides, op_crew_overrides=op_crew_overrides)
            eval_count += 1
            bal_evals += 1

            raw_gain = 0.0
            improved_best = False
            if res.valid:
                req = 0.0 if (has_unrescued or hys_margin <= 0.0) else hys_margin
                if res.objective + req < best_e.objective:
                    raw_gain = best_e.objective - res.objective
                    best_e = res
                    best_a = dict(cand)
                    cur_e = res
                    cur_a = dict(cand)
                    improved_best = True
                    evaluator.set_incumbent(best_a, best_e, op_priority=op_prio, op_bay_overrides=op_bay_overrides)
                elif res.objective < cur_e.objective:
                    raw_gain = 0.5 * (cur_e.objective - res.objective)
                    cur_e = res
                    cur_a = dict(cand)
                else:
                    diff = res.objective - cur_e.objective
                    if temp > 0.5 and rng.random() < math.exp(-diff / max(1.0, temp)):
                        cur_e = res
                        cur_a = dict(cand)
            if improved_best:
                no_improve_streak = 0
            else:
                no_improve_streak += 1

            rew = raw_gain if raw_gain > 0.0 else 0.0
            h_cnt[h_arm] += 1
            h_sum[h_arm] += rew
            h_ss[h_arm] += rew * rew
            n_cnt[h_arm][n_arm] += 1
            n_sum[h_arm][n_arm] += rew
            n_ss[h_arm][n_arm] += rew * rew
            temp *= 0.85

        evaluator.set_incumbent(best_a, best_e, op_priority=op_prio, op_bay_overrides=op_bay_overrides)
        return best_a, best_e, eval_count

    @staticmethod
    def _eval_wait_steps(ev: EvalResult) -> int:
        """Sum spatiotemporal waiting ticks along evaluated robot trajectories."""
        tot = 0
        for path in ev.robot_paths.values():
            for idx in range(len(path) - 1):
                if path[idx][0] == path[idx + 1][0]:
                    tot += max(0, int(path[idx + 1][1]) - int(path[idx][1]))
        return tot

    def _try_ltm_refinement(
        self,
        snapshot: NativeSnapshot,
        evaluator: CVCREvaluator,
        best_assign: dict[str, str],
        best_bays: dict[str, str],
        best_crews: dict[str, int],
        best_eval: EvalResult,
        cur_prio: dict[str, int],
        deadline: float,
        aggressive: bool = False,
        best_orders: Optional[dict[str, list[str]]] = None,
        min_gain: float = 0.01,
        cutoff_step: int = 0,
    ) -> tuple[EvalResult, int, dict[tuple[str, str], int]]:
        """Run online IJCAI-26 LTM rerouting pass, keeping weights iff schedule improves."""
        eval_count = 0
        if snapshot.tick <= 1:
            self._prev_ltm_weights.clear()

        if not self._has_budget(deadline, evaluator, min_ms=4.5):
            evaluator.set_traffic_weights(None)
            return best_eval, 0, {}

        ltm_w = evaluator.compute_ltm_weights_from_paths(
            best_eval.robot_paths,
            prev_weights=self._prev_ltm_weights if self._prev_ltm_weights else None,
        )
        if not ltm_w:
            self._prev_ltm_weights = {}
            evaluator.set_traffic_weights(None)
            return best_eval, 0, {}

        def _accepts_ltm(cand_ev: EvalResult, ref_ev: EvalResult) -> bool:
            if not cand_ev.valid:
                return False
            if cand_ev.objective + min_gain < ref_ev.objective:
                return True
            if min_gain <= 0.1 and cand_ev.objective <= ref_ev.objective + 0.25:
                return self._eval_wait_steps(cand_ev) + 2 < self._eval_wait_steps(ref_ev)
            return False

        order_arg = best_orders if best_orders else None
        best_tw: dict[tuple[str, str], int] = {}
        evaluator.set_traffic_weights(ltm_w, cutoff_step=cutoff_step)
        res_ltm = evaluator.evaluate(
            best_assign,
            op_priority=cur_prio,
            op_bay_overrides=best_bays,
            op_crew_overrides=best_crews,
            task_order_override=order_arg,
        )
        eval_count += 1
        if _accepts_ltm(res_ltm, best_eval):
            best_eval = res_ltm
            best_tw = dict(ltm_w)
            if aggressive and self._has_budget(deadline, evaluator, min_ms=5.0):
                ltm_w2 = evaluator.compute_ltm_weights_from_paths(res_ltm.robot_paths, prev_weights=ltm_w)
                if ltm_w2 and ltm_w2 != ltm_w:
                    evaluator.set_traffic_weights(ltm_w2, cutoff_step=cutoff_step)
                    res_ltm2 = evaluator.evaluate(
                        best_assign,
                        op_priority=cur_prio,
                        op_bay_overrides=best_bays,
                        op_crew_overrides=best_crews,
                        task_order_override=order_arg,
                    )
                    eval_count += 1
                    if _accepts_ltm(res_ltm2, best_eval):
                        best_eval = res_ltm2
                        best_tw = dict(ltm_w2)

        if best_tw:
            evaluator.set_traffic_weights(best_tw, cutoff_step=cutoff_step)
            evaluator.set_incumbent(
                best_assign,
                best_eval,
                op_priority=cur_prio,
                op_bay_overrides=best_bays,
                task_order_override=order_arg,
            )
            self._prev_ltm_weights = dict(best_tw)
            return best_eval, eval_count, best_tw

        evaluator.set_traffic_weights(None)
        self._prev_ltm_weights = {}
        return best_eval, eval_count, {}

    def _v8_precedence_chain_candidate(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        healthy_robots: list[str],
        cur_a: dict[str, str],
        cur_e: EvalResult,
        crit: dict[str, Any],
        op_bay_overrides: dict[str, str],
        op_crew_overrides: dict[str, int],
        surr: _LazyKittingSurrogate,
        succ_map: dict[str, list[str]],
        target_r_cap: int,
        use_km: bool,
        rng: random.Random,
    ) -> Optional[dict[str, str]]:
        """AAAI-26 TAPF-PC joint destruction across op precedence + resource-mutex chains."""
        op_by_id = {op.op_id: op for op in snapshot.operations}
        snap_by_id = {t.task_id: t for t in snapshot.tasks}
        releasable_set = set(clf.releasable_tasks)
        op_w = crit.get("op_weight", {})

        scored_ops = sorted(
            (op for op in snapshot.operations if not op.started),
            key=lambda op: (
                -(op_w.get(op.op_id, 0.0) + 6.0 * max(0, cur_e.op_end.get(op.op_id, 0) - op.due_step)),
                op.due_step,
                op.op_id,
            ),
        )
        if not scored_ops:
            return None
        root_op = scored_ops[rng.randrange(min(4, len(scored_ops)))]
        chain_oids = [root_op.op_id]
        for pid in op_predecessors(root_op):
            if pid in op_by_id:
                chain_oids.append(pid)
        for ch_oid in succ_map.get(root_op.op_id, ())[:2]:
            chain_oids.append(ch_oid)
        for blk_key in ("blocked_by_bay", "blocked_by_crew", "blocked_by_tool"):
            blk = crit.get(blk_key, {}).get(root_op.op_id)
            if blk:
                chain_oids.append(blk)

        pool: list[str] = []
        seen_tids: set[str] = set()
        for oid in chain_oids:
            op_obj = op_by_id.get(oid)
            if op_obj is None:
                continue
            for tid in op_obj.kit_task_ids:
                if tid in releasable_set and tid not in seen_tids:
                    pool.append(tid)
                    seen_tids.add(tid)
        if not pool:
            return None
        pool = pool[: min(len(pool), max(3, target_r_cap + 3))]

        owners = {cur_a[tid] for tid in pool if cur_a.get(tid) in healthy_robots}
        loads = {rid: 0 for rid in healthy_robots}
        for _, agv in cur_a.items():
            if agv in loads:
                loads[agv] += 1
        fast_rids = sorted(healthy_robots, key=lambda r: (loads[r], r))
        sub_rids = sorted(owners)
        for r in fast_rids:
            if len(sub_rids) >= target_r_cap:
                break
            if r not in owners:
                sub_rids.append(r)
        sub_rids = sub_rids[: max(2, min(len(healthy_robots), target_r_cap))]
        if len(sub_rids) < 2:
            return None

        if use_km and len(pool) >= 2:
            return self._build_km_assignment(
                snapshot, clf, sub_rids, cur_a, cur_e, op_bay_overrides, op_crew_overrides, surr, pool
            )
        fixed = {tid: rid for tid, rid in cur_a.items() if tid not in seen_tids}
        return surr.regret_k_assignment(sub_rids, fixed, pool, op_bays=op_bay_overrides, rng=rng, noise=0.05)

    def _v8_entropy_hotspot_candidate(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        healthy_robots: list[str],
        cur_a: dict[str, str],
        cur_e: EvalResult,
        op_bay_overrides: dict[str, str],
        surr: _LazyKittingSurrogate,
        target_r_cap: int,
        rng: random.Random,
    ) -> Optional[dict[str, str]]:
        """JAIR-26 DyMAB congestion-hotspot destruction using trajectory waiting & sharing."""
        releasable = [tid for tid in sorted(clf.releasable_tasks)]
        if len(releasable) < 2 or len(healthy_robots) < 2:
            return None
        edge_users: dict[tuple[str, str], set[str]] = {}
        wait_score: dict[str, float] = {rid: 0.0 for rid in healthy_robots}
        for rid in healthy_robots:
            path = cur_e.robot_paths.get(rid, [])
            for idx in range(len(path) - 1):
                u_n, u_t = path[idx]
                v_n, v_t = path[idx + 1]
                if u_n == v_n:
                    if idx < len(path) - 1:
                        wait_score[rid] += float(max(0, v_t - u_t))
                else:
                    pe = (u_n, v_n) if u_n <= v_n else (v_n, u_n)
                    edge_users.setdefault(pe, set()).add(rid)
        for users in edge_users.values():
            if len(users) >= 2:
                for rid in users:
                    if rid in wait_score:
                        wait_score[rid] += 4.0 * float(len(users) - 1)

        loads = {rid: 0 for rid in healthy_robots}
        for _, agv in cur_a.items():
            if agv in loads:
                loads[agv] += 1

        by_cong = sorted(healthy_robots, key=lambda r: (-wait_score[r], -loads[r], r))
        by_free = sorted(healthy_robots, key=lambda r: (wait_score[r], loads[r], r))
        sub_set: set[str] = {by_cong[0], by_free[0]}
        for r in by_cong[1:] + by_free[1:]:
            if len(sub_set) >= target_r_cap:
                break
            sub_set.add(r)
        sub_rids = sorted(sub_set)
        if len(sub_rids) < 2:
            return None

        pool = [tid for tid in releasable if cur_a.get(tid) in sub_set][: min(len(releasable), target_r_cap + 2)]
        if not pool:
            return None
        pool_set = set(pool)
        fixed = {tid: rid for tid, rid in cur_a.items() if tid not in pool_set}
        return surr.regret_k_assignment(sub_rids, fixed, pool, op_bays=op_bay_overrides, rng=rng, noise=0.06)

    def _search_v8_balance_pc(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        best_assign: dict[str, str],
        best_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        op_bay_overrides: dict[str, str],
        op_crew_overrides: dict[str, int],
        surr: _LazyKittingSurrogate,
        hys_margin: float = 0.0,
        preserve_orders: bool = False,
        lrho_frozen: Optional[set[str]] = None,
    ) -> tuple[dict[str, str], EvalResult, int]:
        """DyMAB Discounted Thompson BALANCE-KM + Precedence-Chain + Hotspot Entropy LNS."""
        releasable = sorted(clf.releasable_tasks)
        if lrho_frozen:
            unfrozen = [tid for tid in releasable if tid not in lrho_frozen]
            if len(unfrozen) >= 2:
                releasable = unfrozen
        if len(releasable) < 2 or len(healthy_robots) < 2:
            return best_assign, best_eval, 0
        inc_orders: Optional[dict[str, list[str]]] = (
            {rid: list(seq) for rid, seq in best_eval.robot_order_seq.items() if seq} or None
            if preserve_orders
            else None
        )

        rng = random.Random(self.seed + snapshot.tick * 211 + 43)
        snap_by_id = {t.task_id: t for t in snapshot.tasks}
        succ_map: dict[str, list[str]] = {}
        for op in snapshot.operations:
            for pid in op_predecessors(op):
                succ_map.setdefault(pid, []).append(op.op_id)

        best_a = dict(best_assign)
        best_e = best_eval
        cur_a = dict(best_assign)
        cur_e = best_eval
        eval_count = 0

        n_H, n_N = 5, 3
        h_cnt = [0] * n_H
        h_sum = [0.0] * n_H
        h_ss = [0.0] * n_H
        n_cnt = [[0] * n_N for _ in range(n_H)]
        n_sum = [[0.0] * n_N for _ in range(n_H)]
        n_ss = [[0.0] * n_N for _ in range(n_H)]

        def _sample_ng(cnt: int, s: float, ss: float) -> float:
            if cnt == 0:
                return 2.0 + rng.random()
            lam = 1.0 + cnt
            mu = s / lam
            alpha = 1.0 + 0.5 * cnt
            mean_x = s / cnt
            var_part = max(0.0, ss - cnt * (mean_x * mean_x))
            beta = 1.0 + 0.5 * var_part + 0.5 * (cnt / lam) * (mean_x * mean_x)
            tau = rng.gammavariate(alpha, 1.0 / max(1e-6, beta))
            sigma = 1.0 / math.sqrt(max(1e-6, lam * tau))
            return rng.gauss(mu, min(sigma, 15.0))

        temp = 14.0
        has_unrescued = any(
            snap_by_id.get(tid) and snap_by_id[tid].assigned_agv in clf.disturbed_robots and cur_a.get(tid) in clf.disturbed_robots
            for tid in cur_a
        )

        res_tard0 = sum(
            (12.0 if op.is_urgent else 4.0) * max(0, cur_e.op_end.get(op.op_id, 0) - op.due_step)
            for op in snapshot.operations
        )
        if res_tard0 >= 60.0 and self._has_budget(deadline, evaluator, min_ms=16.0):
            crit0 = compute_hybrid_criticality(snapshot, cur_e, op_bay_overrides, op_crew_overrides, self.alpha_makespan)
            tw0 = crit0["task_weight"]
            last_kit_pool = sorted(
                releasable,
                key=lambda tid: (-tw0.get(tid, 0.0), crit0["kitting_slack"].get(tid, 999), tid),
            )[: max(3, min(len(healthy_robots) + 2, len(releasable)))]
            km_cand = self._build_km_assignment(
                snapshot, clf, healthy_robots, cur_a, cur_e, op_bay_overrides, op_crew_overrides, surr, last_kit_pool
            )
            if km_cand is not None:
                res_km = evaluator.evaluate(
                    km_cand,
                    op_priority=op_prio,
                    op_bay_overrides=op_bay_overrides,
                    op_crew_overrides=op_crew_overrides,
                    task_order_override=inc_orders,
                )
                eval_count += 1
                if res_km.valid and res_km.objective + hys_margin < best_e.objective:
                    best_e = res_km
                    best_a = km_cand
                    cur_e = res_km
                    cur_a = dict(km_cand)
                    if preserve_orders:
                        inc_orders = {rid: list(seq) for rid, seq in res_km.robot_order_seq.items() if seq} or None
                    evaluator.set_incumbent(
                        best_a, best_e, op_priority=op_prio, op_bay_overrides=op_bay_overrides, task_order_override=inc_orders
                    )

        seen_cands: set[tuple[tuple[str, str], ...]] = {
            tuple(sorted(best_a.items())),
            tuple(sorted(cur_a.items())),
        }
        no_improve_streak = 0
        dup_streak = 0
        bal_evals = 0
        max_bal_evals = 16

        while self._has_budget(deadline, evaluator, min_ms=6.0):
            if bal_evals >= max_bal_evals or no_improve_streak >= 6 or dup_streak >= 8:
                break
            h_arm = max(range(n_H), key=lambda i: _sample_ng(h_cnt[i], h_sum[i], h_ss[i]))
            n_arm = max(range(n_N), key=lambda j: _sample_ng(n_cnt[h_arm][j], n_sum[h_arm][j], n_ss[h_arm][j]))
            rem_ms = (deadline - _cvcr_now_s()) * 1000.0
            res_tard = sum(
                (12.0 if op.is_urgent else 4.0) * max(0, cur_e.op_end.get(op.op_id, 0) - op.due_step)
                for op in snapshot.operations
            )
            if rem_ms < 16.0 or res_tard < 40.0:
                n_arm = 0

            crit = compute_hybrid_criticality(snapshot, cur_e, op_bay_overrides, op_crew_overrides, self.alpha_makespan)
            tw = crit["task_weight"]
            ks = crit["kitting_slack"]

            cand: Optional[dict[str, str]] = None
            target_r_cap = 2 if n_arm == 0 else (3 if n_arm == 1 else min(len(healthy_robots) - 1, 4))

            if h_arm == 0:
                last_kits = sorted(releasable, key=lambda tid: (-tw.get(tid, 0.0), abs(ks.get(tid, 0)), tid))
                if n_arm == 2 and len(last_kits) >= 3:
                    cand = self._build_km_assignment(
                        snapshot, clf, healthy_robots, cur_a, cur_e, op_bay_overrides, op_crew_overrides,
                        surr, last_kits[: min(len(healthy_robots), len(last_kits))],
                    )
                else:
                    seed_tid = last_kits[rng.randrange(min(4, len(last_kits)))]
                    chosen_r = rng.sample(healthy_robots, min(target_r_cap, len(healthy_robots)))
                    sub = [tid for tid in last_kits if cur_a.get(tid) in chosen_r or tid == seed_tid][: (target_r_cap + 2)]
                    if sub:
                        sub_rids = sorted(set(chosen_r) | ({cur_a[seed_tid]} if cur_a.get(seed_tid) in healthy_robots else set()))[:target_r_cap]
                        fixed = {tid: rid for tid, rid in cur_a.items() if tid not in sub}
                        cand = surr.regret_k_assignment(sub_rids, fixed, sub, op_bays=op_bay_overrides, rng=rng, noise=0.05)
            elif h_arm == 1:
                ranked = sorted(releasable, key=lambda tid: (-tw.get(tid, 0.0), float(snap_by_id[tid].due_step or 999), tid))
                sub_rids = rng.sample(healthy_robots, min(target_r_cap, len(healthy_robots)))
                sub_tids = [tid for tid in ranked if cur_a.get(tid) in sub_rids]
                if len(sub_tids) >= 2:
                    if n_arm == 0:
                        t1 = sub_tids[0]
                        cand_r2 = [tid for tid in sub_tids if cur_a.get(tid) != cur_a.get(t1)]
                        if cand_r2:
                            t2 = rng.choice(cand_r2)
                            cand = dict(cur_a)
                            cand[t1], cand[t2] = cand[t2], cand[t1]
                    else:
                        cand = self._build_km_assignment(
                            snapshot, clf, sub_rids, cur_a, cur_e, op_bay_overrides, op_crew_overrides,
                            surr, sub_tids[: min(len(sub_tids), target_r_cap + 1)],
                        )
            elif h_arm == 2:
                if res_tard >= 40.0 and rng.random() < 0.5:
                    cand = self._v8_entropy_hotspot_candidate(
                        snapshot, clf, healthy_robots, cur_a, cur_e, op_bay_overrides, surr, target_r_cap, rng
                    )
                if cand is None:
                    sub_rids = rng.sample(healthy_robots, min(target_r_cap, len(healthy_robots)))
                    sub_tids = [tid for tid in releasable if cur_a.get(tid) in sub_rids]
                    if sub_tids:
                        anchor = snap_by_id[rng.choice(sub_tids)]
                        sub_tids.sort(key=lambda tid: surr.tt(anchor.pickup, snap_by_id[tid].pickup))
                        rem_k = sub_tids[: min(len(sub_tids), target_r_cap + 2)]
                        fixed = {tid: rid for tid, rid in cur_a.items() if tid not in rem_k}
                        cand = surr.regret_k_assignment(sub_rids, fixed, rem_k, op_bays=op_bay_overrides, rng=rng, noise=0.08)
            elif h_arm == 3:
                sub_rids = rng.sample(healthy_robots, min(target_r_cap, len(healthy_robots)))
                picks = []
                for rid in sub_rids:
                    rtids = [tid for tid in releasable if cur_a.get(tid) == rid]
                    if rtids:
                        picks.append(rng.choice(rtids))
                if len(picks) >= 2:
                    cand = dict(cur_a)
                    owners = [cur_a[tid] for tid in picks]
                    for idx_p, tid in enumerate(picks):
                        cand[tid] = owners[(idx_p + 1) % len(picks)]
            else:
                cand = self._v8_precedence_chain_candidate(
                    snapshot, clf, healthy_robots, cur_a, cur_e, crit,
                    op_bay_overrides, op_crew_overrides, surr, succ_map, max(2, target_r_cap),
                    use_km=(n_arm >= 1), rng=rng,
                )

            if cand is None or cand == cur_a:
                h_cnt[h_arm] += 1
                n_cnt[h_arm][n_arm] += 1
                dup_streak += 1
                temp *= 0.85
                if dup_streak >= 8:
                    break
                continue
            cand_key = tuple(sorted(cand.items()))
            if cand_key in seen_cands:
                h_cnt[h_arm] += 1
                n_cnt[h_arm][n_arm] += 1
                dup_streak += 1
                temp *= 0.85
                if dup_streak >= 8:
                    break
                continue
            seen_cands.add(cand_key)
            dup_streak = 0

            res = evaluator.evaluate(
                cand,
                op_priority=op_prio,
                op_bay_overrides=op_bay_overrides,
                op_crew_overrides=op_crew_overrides,
                task_order_override=inc_orders,
            )
            eval_count += 1
            bal_evals += 1

            raw_gain = 0.0
            improved_best = False
            if res.valid:
                req = 0.0 if (has_unrescued or hys_margin <= 0.0) else hys_margin
                if res.objective + req < best_e.objective:
                    raw_gain = best_e.objective - res.objective
                    best_e = res
                    best_a = dict(cand)
                    cur_e = res
                    cur_a = dict(cand)
                    improved_best = True
                    if preserve_orders:
                        inc_orders = {rid: list(seq) for rid, seq in res.robot_order_seq.items() if seq} or None
                    evaluator.set_incumbent(
                        best_a, best_e, op_priority=op_prio, op_bay_overrides=op_bay_overrides, task_order_override=inc_orders
                    )
                elif res.objective < cur_e.objective:
                    raw_gain = 0.5 * (cur_e.objective - res.objective)
                    cur_e = res
                    cur_a = dict(cand)
                else:
                    diff = res.objective - cur_e.objective
                    if temp > 0.5 and rng.random() < math.exp(-diff / max(1.0, temp)):
                        cur_e = res
                        cur_a = dict(cand)
            if improved_best:
                no_improve_streak = 0
            else:
                no_improve_streak += 1

            rew = raw_gain if raw_gain > 0.0 else 0.0
            h_cnt[h_arm] += 1
            h_sum[h_arm] += rew
            h_ss[h_arm] += rew * rew
            n_cnt[h_arm][n_arm] += 1
            n_sum[h_arm][n_arm] += rew
            n_ss[h_arm][n_arm] += rew * rew
            temp *= 0.85

        evaluator.set_incumbent(
            best_a, best_e, op_priority=op_prio, op_bay_overrides=op_bay_overrides, task_order_override=inc_orders
        )
        return best_a, best_e, eval_count

    def _search_v8_expert_portfolio(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        base_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        budget_ms: float,
    ) -> tuple[
        dict[str, str],
        dict[str, str],
        dict[str, int],
        EvalResult,
        int,
        list[str],
        Optional[list[str]],
        dict[tuple[str, str], int],
    ]:
        """Dispatch V8 LTM + Precedence-Chain + DyMAB Entropy + c_i Experts (V8_E1..V8_E6)."""
        best_assign = dict(base_assign)
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
        best_eval = base_eval
        eval_count = 1
        cur_prio = dict(op_prio)
        best_orders: dict[str, list[str]] = {}
        best_traffic_w: dict[tuple[str, str], int] = {}

        if len(healthy_robots) < 2:
            evaluator.set_traffic_weights(None)
            return best_assign, best_bays, best_crews, best_eval, eval_count, [], None, {}

        surr = (
            _LazyKittingSurrogate(self.network, snapshot, cur_prio, self.alpha_makespan, self.lambda_change)
            if budget_ms >= 35.0
            else None
        )

        if self.method_id == "V8_E1_FAST_RESCUE":
            rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
            if rel_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                    snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    allow_swaps=False, hys_margin=5.0,
                )
                eval_count += c1
            if _cvcr_now_s() + 0.002 < deadline:
                best_orders, best_eval, c_ci = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, max_candidates=8,
                )
                eval_count += c_ci
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None, {}

        if self.method_id == "V8_E2_CRIT_PROD_CI":
            if budget_ms >= 35.0 and _cvcr_now_s() < deadline:
                prod_deadline = min(deadline, _cvcr_now_s() + max(0.012, (float(budget_ms) * 0.58) / 1000.0))
                best_bays, best_crews, cur_prio, best_eval, cb = self._divert_prod_v6(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, prod_deadline
                )
                eval_count += cb
            rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
            snap_by_id = {t.task_id: t for t in snapshot.tasks}
            changed_ops = {
                op.op_id for op in snapshot.operations
                if best_bays[op.op_id] != op.bay_id or best_crews[op.op_id] != op.crew_id
            }
            focused_tids = [
                tid for tid in rel_tids
                if (snap_by_id.get(tid) and snap_by_id[tid].assigned_agv in clf.disturbed_robots)
                or (snap_by_id.get(tid) and snap_by_id[tid].kit_op_id in changed_ops)
                or (snap_by_id.get(tid) and snap_by_id[tid].is_urgent)
            ] or rel_tids[:4]
            if focused_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                    snapshot, clf, focused_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    op_bay_overrides=best_bays, op_crew_overrides=best_crews, allow_swaps=False, hys_margin=5.0,
                )
                eval_count += c1
            if _cvcr_now_s() + 0.002 < deadline:
                best_orders, best_eval, c_ci = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, max_candidates=10,
                )
                eval_count += c_ci
            return best_assign, best_bays, best_crews, best_eval, eval_count, focused_tids, None, {}

        if self.method_id == "V8_E4_LTM_PACER_CI":
            pacer_deadline = max(_cvcr_now_s(), deadline - (0.012 if budget_ms >= 55.0 else 0.004))
            best_assign, best_bays, best_crews, best_eval, c4, rel_tids, admitted = self._search_congestion_pacer(
                snapshot, clf, evaluator, best_assign, best_eval, healthy_robots, cur_prio, pacer_deadline, budget_ms
            )
            eval_count += max(0, c4 - 1)
            if _cvcr_now_s() + 0.004 < deadline:
                best_eval, c_ltm, best_traffic_w = self._try_ltm_refinement(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    aggressive=True, best_orders=None, min_gain=0.01,
                )
                eval_count += c_ltm
            if _cvcr_now_s() + 0.002 < deadline:
                best_orders, best_eval, c_ci = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, max_candidates=8,
                )
                eval_count += c_ci
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, admitted, best_traffic_w

        if self.method_id == "V8_E6_BALANCE_KM_PC":
            if budget_ms >= 35.0 and _cvcr_now_s() < deadline:
                bay_deadline = min(deadline, _cvcr_now_s() + max(0.010, (float(budget_ms) * 0.35) / 1000.0))
                best_bays, best_crews, cur_prio, best_eval, cb = self._divert_prod_extended(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, bay_deadline
                )
                eval_count += cb
            rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
            if rel_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                    snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    op_bay_overrides=best_bays, op_crew_overrides=best_crews,
                    allow_swaps=(budget_ms >= 35.0), surr=surr, hys_margin=0.0,
                )
                eval_count += c1
            if _cvcr_now_s() + 0.003 < deadline:
                ci_deadline = min(deadline, _cvcr_now_s() + max(0.012, (float(budget_ms) * 0.22) / 1000.0))
                best_orders, best_eval, c_ci = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, ci_deadline,
                    surr=surr, max_candidates=10,
                )
                eval_count += c_ci
            if budget_ms >= 85.0 and surr is not None and _cvcr_now_s() < deadline:
                sa_deadline = max(_cvcr_now_s(), deadline - 0.005)
                best_assign, best_eval, ca = self._search_v8_balance_pc(
                    snapshot, clf, evaluator, best_assign, best_eval, healthy_robots, cur_prio, sa_deadline,
                    best_bays, best_crews, surr, hys_margin=0.0,
                )
                eval_count += ca
            if self._eval_wait_steps(best_eval) >= 16 and self._has_budget(deadline, evaluator, min_ms=5.0):
                best_eval, c_ltm, best_traffic_w = self._try_ltm_refinement(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    aggressive=False, best_orders=None, min_gain=2.5,
                )
                eval_count += c_ltm
            if _cvcr_now_s() + 0.002 < deadline:
                best_orders, best_eval, c_ci2 = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, max_candidates=6,
                )
                eval_count += c_ci2
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None, best_traffic_w

        # V8_E3_JOINT_PC_HYS (hys=5.0) vs V8_E5_AGGRESSIVE_LTM (hys=0.0)
        is_e3 = self.method_id == "V8_E3_JOINT_PC_HYS"
        hys = 5.0 if is_e3 else 0.0
        if budget_ms >= 35.0 and _cvcr_now_s() < deadline:
            bay_frac = 0.40 if is_e3 else 0.35
            bay_deadline = min(deadline, _cvcr_now_s() + max(0.010, (float(budget_ms) * bay_frac) / 1000.0))
            best_bays, best_crews, cur_prio, best_eval, cb = self._divert_prod_extended(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, bay_deadline
            )
            eval_count += cb
        rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
        if rel_tids and _cvcr_now_s() < deadline:
            best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                op_bay_overrides=best_bays, op_crew_overrides=best_crews,
                allow_swaps=(budget_ms >= 35.0), surr=surr, hys_margin=hys,
            )
            eval_count += c1
        if _cvcr_now_s() + 0.003 < deadline:
            ci_deadline = min(deadline, _cvcr_now_s() + max(0.012, (float(budget_ms) * 0.22) / 1000.0))
            best_orders, best_eval, c_ci = self._search_intra_agent_ci(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, ci_deadline,
                surr=surr, max_candidates=10,
            )
            eval_count += c_ci
        if budget_ms >= 85.0 and _cvcr_now_s() < deadline:
            sa_deadline = max(_cvcr_now_s(), deadline - 0.005)
            best_assign, best_eval, ca = self._search_alns_local_b2(
                snapshot, clf, evaluator, best_assign, best_eval, healthy_robots, cur_prio, sa_deadline,
                op_bay_overrides=best_bays, op_crew_overrides=best_crews, surr=surr, hys_margin=hys,
            )
            eval_count += ca
        if not is_e3 and self._eval_wait_steps(best_eval) >= 16 and self._has_budget(deadline, evaluator, min_ms=5.0):
            best_eval, c_ltm, best_traffic_w = self._try_ltm_refinement(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                aggressive=False, best_orders=None, min_gain=2.5,
            )
            eval_count += c_ltm
        if _cvcr_now_s() + 0.002 < deadline:
            best_orders, best_eval, c_ci2 = self._search_intra_agent_ci(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                surr=surr, max_candidates=6,
            )
            eval_count += c_ci2
        return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None, best_traffic_w

    def _search_local_cpsat_window(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        best_assign: dict[str, str],
        best_bays: dict[str, str],
        best_crews: dict[str, int],
        best_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        surr: Optional[_LazyKittingSurrogate],
        allow_order_only: bool = False,
        min_improve: float = 2.0,
        solve_time_cap_s: float = 0.011,
        disable_presolve: bool = False,
        preserve_nonfocal_order: bool = False,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, int], dict[str, int], dict[str, list[str]], EvalResult, int]:
        """Solve a bounded focal bottleneck window jointly over (AGV, Bay, Crew, Tool, Order) via OR-Tools CP-SAT."""
        cp_model = _get_ortools_cp_model()
        if cp_model is None:
            return best_assign, best_bays, best_crews, dict(op_prio), {}, best_eval, 0

        if len(healthy_robots) < 2 or not self._has_budget(deadline, evaluator, min_ms=14.0):
            return best_assign, best_bays, best_crews, dict(op_prio), {}, best_eval, 0

        if surr is None:
            surr = _LazyKittingSurrogate(self.network, snapshot, op_prio, self.alpha_makespan, self.lambda_change)

        t_now = int(snapshot.tick)
        op_map = {op.op_id: op for op in snapshot.operations}
        snap_by_id = {t.task_id: t for t in snapshot.tasks}
        unstarted = [op for op in snapshot.operations if not op.started]
        releasable_set = set(clf.releasable_tasks)
        if not unstarted or not releasable_set:
            return best_assign, best_bays, best_crews, dict(op_prio), {}, best_eval, 0

        crit = compute_hybrid_criticality(
            snapshot, best_eval, best_bays, best_crews, alpha_makespan=self.alpha_makespan
        )
        op_w = crit["op_weight"]
        task_w = crit["task_weight"]
        unstarted_ranked = sorted(
            unstarted,
            key=lambda o: (
                -max(0, best_eval.op_end.get(o.op_id, t_now) - o.due_step) * (12 if o.is_urgent else 4),
                -op_w.get(o.op_id, 0.0),
                o.due_step,
                o.op_id,
            ),
        )
        focal_ops = unstarted_ranked[:5]
        focal_op_ids = {op.op_id for op in focal_ops}
        for op in list(focal_ops):
            for pred_id in op_predecessors(op):
                if pred_id not in focal_op_ids and pred_id in op_map and not op_map[pred_id].started:
                    if len(focal_ops) < 7:
                        focal_ops.append(op_map[pred_id])
                        focal_op_ids.add(pred_id)

        focal_tids_pool = sorted(
            releasable_set,
            key=lambda tid: (
                0 if (snap_by_id.get(tid) and snap_by_id[tid].kit_op_id in focal_op_ids) else 1,
                0 if (snap_by_id.get(tid) and (snap_by_id[tid].assigned_agv in clf.disturbed_robots or tid not in best_assign)) else 1,
                -task_w.get(tid, 0.0),
                tid,
            ),
        )
        focal_tids = focal_tids_pool[:7]
        if not focal_tids:
            return best_assign, best_bays, best_crews, dict(op_prio), {}, best_eval, 0
        focal_tid_set = set(focal_tids)

        max_due = max([op.due_step for op in snapshot.operations] + [t_now])
        horizon = max(t_now + 1200, max_due + 600)
        bay_intervals: dict[str, list[Any]] = {}
        crew_intervals: dict[int, list[Any]] = {}
        tool_intervals: dict[int, list[Any]] = {}
        fixed_op_ends: dict[str, int] = {}
        fixed_mk = t_now

        model = cp_model.CpModel()
        for op in snapshot.operations:
            if op.op_id in focal_op_ids:
                continue
            st = int(op.start_step) if (op.started and op.start_step is not None) else int(best_eval.op_start.get(op.op_id, t_now))
            en = int(op.end_step) if (op.started and op.end_step is not None) else int(best_eval.op_end.get(op.op_id, st + op.proc_steps))
            st = max(t_now if not op.started else st, min(horizon, st))
            en = max(st + 1, min(horizon + op.proc_steps, en))
            fixed_op_ends[op.op_id] = en
            fixed_mk = max(fixed_mk, en)
            if en > t_now:
                iv = model.NewFixedSizeIntervalVar(max(0, st), en - max(0, st), f"fop_{op.op_id}")
                bay_intervals.setdefault(best_bays[op.op_id], []).append(iv)
                crew_intervals.setdefault(best_crews[op.op_id], []).append(iv)
                tool_intervals.setdefault(op.tool_id, []).append(iv)

        rob_map = {r.robot_id: r for r in snapshot.robots}
        rob_busy_until = {rid: max(t_now, int(rob_map[rid].faulted_until)) for rid in healthy_robots}
        rob_frontier_node = {rid: rob_map[rid].cur_node for rid in healthy_robots}
        fixed_task_unloads: dict[str, int] = {}
        frozen_last_u: dict[str, int] = {rid: -1 for rid in healthy_robots}
        for t in snapshot.tasks:
            if t.status.lower() == "completed":
                fixed_task_unloads[t.task_id] = int(t.complete_step or t_now)
                continue
            if t.task_id not in focal_tid_set:
                est_u = int(best_eval.order_unload_ticks.get(t.task_id, horizon))
                fixed_task_unloads[t.task_id] = min(horizon, max(t_now, est_u))
                rid = best_assign.get(t.task_id)
                is_fixed_exec = (t.task_id in clf.frozen_tasks) or (
                    t.status.lower() in ("picked", "loading", "to_dropoff", "unloading")
                )
                if rid in rob_busy_until and is_fixed_exec and est_u < horizon:
                    rob_busy_until[rid] = max(rob_busy_until[rid], est_u)
                    if est_u >= frozen_last_u[rid]:
                        frozen_last_u[rid] = est_u
                        rob_frontier_node[rid] = (
                            best_bays[t.kit_op_id]
                            if (t.kit_op_id and t.kit_op_id in best_bays and t.status.lower() in ("pending", "assigned"))
                            else t.dropoff
                        )

        all_drops = list({best_bays.get(t.kit_op_id, t.dropoff) for t in snapshot.tasks if t.status.lower() != "completed"})
        op_s_var: dict[str, Any] = {}
        op_e_var: dict[str, Any] = {}
        b_vars: dict[tuple[str, str], Any] = {}
        c_vars: dict[tuple[str, int], Any] = {}
        op_elig_bays: dict[str, list[str]] = {}
        op_elig_crews: dict[str, list[int]] = {}
        tard_terms: list[Any] = []

        for op in focal_ops:
            dur = max(1, int(op.proc_steps))
            sv = model.NewIntVar(t_now, horizon, f"os_{op.op_id}")
            ev = model.NewIntVar(t_now + dur, horizon + dur, f"oe_{op.op_id}")
            model.Add(ev == sv + dur)
            op_s_var[op.op_id] = sv
            op_e_var[op.op_id] = ev
            inc_st = min(horizon, max(t_now, int(best_eval.op_start.get(op.op_id, t_now + 20))))
            model.AddHint(sv, inc_st)
            model.AddHint(ev, inc_st + dur)

            b_cands = list(dict.fromkeys(list(op.eligible_bays) + [best_bays[op.op_id]]))[:4]
            op_elig_bays[op.op_id] = b_cands
            b_row: list[Any] = []
            for b in b_cands:
                bv = model.NewBoolVar(f"b_{op.op_id}_{b}")
                b_vars[(op.op_id, b)] = bv
                b_row.append(bv)
                model.AddHint(bv, 1 if b == best_bays[op.op_id] else 0)
                bay_intervals.setdefault(b, []).append(
                    model.NewOptionalFixedSizeIntervalVar(sv, dur, bv, f"bi_{op.op_id}_{b}")
                )
            model.AddExactlyOne(b_row)

            c_cands = list(dict.fromkeys(list(op.eligible_crews) + [best_crews[op.op_id]]))[:4]
            op_elig_crews[op.op_id] = c_cands
            c_row: list[Any] = []
            for c in c_cands:
                cv = model.NewBoolVar(f"c_{op.op_id}_{c}")
                c_vars[(op.op_id, c)] = cv
                c_row.append(cv)
                model.AddHint(cv, 1 if c == best_crews[op.op_id] else 0)
                crew_intervals.setdefault(c, []).append(
                    model.NewOptionalFixedSizeIntervalVar(sv, dur, cv, f"ci_{op.op_id}_{c}")
                )
            model.AddExactlyOne(c_row)
            tool_intervals.setdefault(op.tool_id, []).append(
                model.NewFixedSizeIntervalVar(sv, dur, f"ti_{op.op_id}")
            )

            tv = model.NewIntVar(0, horizon + dur, f"tard_{op.op_id}")
            model.Add(tv >= ev - int(op.due_step))
            w_t = 24 if op.is_urgent else 8
            tard_terms.append(w_t * tv)

        for op in focal_ops:
            for pid in op_predecessors(op):
                if pid in op_e_var:
                    model.Add(op_s_var[op.op_id] >= op_e_var[pid])
                elif pid in fixed_op_ends:
                    model.Add(op_s_var[op.op_id] >= fixed_op_ends[pid])

        agv_intervals: dict[str, list[Any]] = {rid: [] for rid in healthy_robots}
        x_vars: dict[tuple[str, str], Any] = {}
        task_s_var: dict[str, Any] = {}
        task_e_var: dict[str, Any] = {}
        task_unload_expr: dict[str, Any] = {}
        churn_terms: list[Any] = []

        for tid in focal_tids:
            t = snap_by_id[tid]
            inter_dh = min([int(math.ceil(surr.tt(d, t.pickup))) for d in all_drops] + [2])
            if t.kit_op_id and t.kit_op_id in op_elig_bays and t.status.lower() in ("pending", "assigned"):
                b_list = op_elig_bays[t.kit_op_id]
                pd_costs = {b: 4 + int(math.ceil(surr.tt(t.pickup, b) + surr.bay_delay(b))) for b in b_list}
                min_pd = min(pd_costs.values())
                bay_delta = sum(b_vars[(t.kit_op_id, b)] * (pd_costs[b] - min_pd) for b in b_list)
            else:
                fix_drop = (
                    best_bays.get(t.kit_op_id, t.dropoff)
                    if (t.kit_op_id and t.status.lower() in ("pending", "assigned"))
                    else t.dropoff
                )
                min_pd = 4 + int(math.ceil(surr.tt(t.pickup, fix_drop) + surr.bay_delay(fix_drop)))
                bay_delta = 0

            agv_dur = max(4, inter_dh + min_pd)
            ts = model.NewIntVar(t_now, horizon, f"ts_{tid}")
            te = model.NewIntVar(t_now + agv_dur, horizon + agv_dur, f"te_{tid}")
            model.Add(te == ts + agv_dur)
            model.Add(te + bay_delta >= max(t_now, int(t.release_step)) + min_pd)
            task_s_var[tid] = ts
            task_e_var[tid] = te
            task_unload_expr[tid] = te + bay_delta

            inc_u = min(horizon, max(t_now + agv_dur, int(best_eval.order_unload_ticks.get(tid, t_now + agv_dur + 8))))
            model.AddHint(ts, max(t_now, inc_u - agv_dur))
            model.AddHint(te, inc_u)

            inc_r = best_assign.get(tid)
            if inc_r not in healthy_robots:
                inc_r = healthy_robots[0]
            x_row: list[Any] = []
            for rid in healthy_robots:
                xv = model.NewBoolVar(f"x_{tid}_{rid}")
                x_vars[(tid, rid)] = xv
                x_row.append(xv)
                model.AddHint(xv, 1 if rid == inc_r else 0)
                agv_intervals[rid].append(
                    model.NewOptionalFixedSizeIntervalVar(ts, agv_dur, xv, f"ai_{tid}_{rid}")
                )
                dh_init = max(0, int(math.ceil(surr.tt(rob_frontier_node[rid], t.pickup))) - inter_dh)
                model.Add(ts >= rob_busy_until[rid] + dh_init).OnlyEnforceIf(xv)
                if tid in evaluator.snapshot_assign and evaluator.snapshot_assign[tid] != rid and t.status.lower() == "assigned":
                    churn_terms.append(int(round(2.0 * self.lambda_change)) * xv)
            model.AddExactlyOne(x_row)

        for ivs in list(bay_intervals.values()) + list(crew_intervals.values()) + list(tool_intervals.values()) + list(agv_intervals.values()):
            if len(ivs) >= 2:
                model.AddNoOverlap(ivs)

        for op in focal_ops:
            for tid in op.kit_task_ids:
                if tid in task_unload_expr:
                    model.Add(op_s_var[op.op_id] >= task_unload_expr[tid])
                elif tid in fixed_task_unloads:
                    model.Add(op_s_var[op.op_id] >= fixed_task_unloads[tid])

        mk_var = model.NewIntVar(fixed_mk, horizon + 120, "mk")
        if op_e_var:
            model.AddMaxEquality(mk_var, [fixed_mk] + list(op_e_var.values()))

        deliv_expr = sum(task_unload_expr[tid] for tid in focal_tids)
        mk_coeff = int(round(2.0 * self.alpha_makespan))
        model.Minimize(mk_coeff * mk_var + sum(tard_terms) + deliv_expr + sum(churn_terms))

        rem_s = max(0.002, deadline - _cvcr_now_s())
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = int(self.seed)
        if disable_presolve:
            solver.parameters.cp_model_presolve = False
        apply_cpsat_time_limit(
            solver.parameters,
            min(float(solve_time_cap_s), max(0.004, rem_s * 0.35)),
        )

        self._last_cp_stats["cp_invoked"] += 1
        status = solver.Solve(model)
        if is_virtual_clock_active():
            _cvcr_record_work_ms(
                min(float(solve_time_cap_s) * 1000.0, 2.2 + 0.30 * len(focal_ops) + 0.05 * len(focal_tids))
            )
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return best_assign, best_bays, best_crews, dict(op_prio), {}, best_eval, 0
        self._last_cp_stats["cp_feasible"] += 1

        cand_assign = dict(best_assign)
        for tid in focal_tids:
            for rid in healthy_robots:
                if solver.Value(x_vars[(tid, rid)]) == 1:
                    cand_assign[tid] = rid
                    break

        cand_bays = dict(best_bays)
        cand_crews = dict(best_crews)
        cand_prio = dict(op_prio)
        prio_slots = sorted(int(op_prio.get(op.op_id, op.due_step)) for op in focal_ops)
        cp_ops_ordered = sorted(
            focal_ops,
            key=lambda o: (int(solver.Value(op_s_var[o.op_id])), o.due_step, o.op_id),
        )
        for idx_o, op in enumerate(cp_ops_ordered):
            for b in op_elig_bays[op.op_id]:
                if solver.Value(b_vars[(op.op_id, b)]) == 1:
                    cand_bays[op.op_id] = b
                    break
            for c in op_elig_crews[op.op_id]:
                if solver.Value(c_vars[(op.op_id, c)]) == 1:
                    cand_crews[op.op_id] = c
                    break
            cand_prio[op.op_id] = prio_slots[idx_o]

        cand_orders: dict[str, list[str]] = {}
        focal_finish = {tid: int(solver.Value(task_e_var[tid])) for tid in focal_tids}
        for rid in healthy_robots:
            r_tasks = [t for t in snapshot.tasks if t.status.lower() != "completed" and cand_assign.get(t.task_id) == rid]
            if r_tasks:
                inc_seq = list(best_eval.robot_order_seq.get(rid, ()))
                inc_pos = {tid: idx for idx, tid in enumerate(inc_seq)}
                base_ord = len(inc_seq) + 1
                r_tasks.sort(
                    key=lambda t: (
                        0 if t.status.lower() in ("picked", "loading", "to_dropoff", "unloading") else 1,
                        (inc_pos.get(t.task_id, base_ord) if (preserve_nonfocal_order and t.task_id not in focal_finish) else (0 if t.task_id in focal_finish else 1)),
                        focal_finish.get(t.task_id, cand_prio.get(t.kit_op_id or "", 9999))
                        if not (preserve_nonfocal_order and t.task_id not in focal_finish)
                        else inc_pos.get(t.task_id, base_ord),
                        t.release_step,
                        t.task_id,
                    )
                )
                cand_orders[rid] = [t.task_id for t in r_tasks]

        orders_changed = bool(
            allow_order_only
            and cand_orders
            and any(
                cand_orders.get(rid, []) != list(best_eval.robot_order_seq.get(rid, []))
                for rid in cand_orders
            )
        )
        non_order_changed = (
            cand_assign != best_assign
            or cand_bays != best_bays
            or cand_crews != best_crews
            or any(cand_prio.get(op.op_id) != op_prio.get(op.op_id) for op in focal_ops)
        )
        changed = non_order_changed or orders_changed
        if not changed:
            self._last_cp_stats["cp_skipped_no_change"] += 1
            return best_assign, best_bays, best_crews, dict(op_prio), {}, best_eval, 0
        if not self._has_budget(deadline, evaluator, min_ms=4.0):
            self._last_cp_stats["cp_skipped_no_budget"] += 1
            return best_assign, best_bays, best_crews, dict(op_prio), {}, best_eval, 0

        self._last_cp_stats["cp_entered_eval"] += 1
        if orders_changed and not non_order_changed:
            self._last_cp_stats["cp_queue_only"] += 1
        res = evaluator.evaluate(
            cand_assign,
            op_priority=cand_prio,
            op_bay_overrides=cand_bays,
            op_crew_overrides=cand_crews,
            task_order_override=cand_orders or None,
        )
        eval_count = 1
        if res.valid and res.objective + float(min_improve) < best_eval.objective:
            self._last_cp_stats["cp_accepted_improved"] += 1
            if surr is not None:
                surr.op_prio = dict(cand_prio)
            evaluator.set_incumbent(
                cand_assign, res, op_priority=cand_prio, op_bay_overrides=cand_bays, task_order_override=cand_orders or None
            )
            return cand_assign, cand_bays, cand_crews, cand_prio, cand_orders, res, eval_count
        return best_assign, best_bays, best_crews, dict(op_prio), {}, best_eval, eval_count

    def _polish_v9_traffic_and_guides(
        self,
        snapshot: NativeSnapshot,
        evaluator: CVCREvaluator,
        best_assign: dict[str, str],
        best_bays: dict[str, str],
        best_crews: dict[str, int],
        best_eval: EvalResult,
        cur_prio: dict[str, int],
        deadline: float,
        surr: Optional[_LazyKittingSurrogate],
        aggressive_ltm: bool = False,
        min_ltm_gain: float = 2.5,
        min_wait_ltm: int = 16,
        ci_candidates: int = 6,
        cutoff_horizon: int = 55,
        emit_guides: bool = True,
        preserve_orders: bool = False,
    ) -> tuple[EvalResult, int, dict[tuple[str, str], int], int, dict[str, list[tuple[str, str]]]]:
        """Run windowed LTM refinement, intra-AGV CI ordering, and validated corridor extraction."""
        eval_count = 0
        best_traffic_w: dict[tuple[str, str], int] = {}
        cutoff_step = int(snapshot.tick) + int(cutoff_horizon) if int(cutoff_horizon) > 0 else 0
        use_cutoff = 0

        if (aggressive_ltm or self._eval_wait_steps(best_eval) >= min_wait_ltm) and self._has_budget(deadline, evaluator, min_ms=5.0):
            init_orders = (
                {rid: list(seq) for rid, seq in best_eval.robot_order_seq.items() if seq} or None
                if preserve_orders
                else None
            )
            best_eval, c_ltm, best_traffic_w = self._try_ltm_refinement(
                snapshot,
                evaluator,
                best_assign,
                best_bays,
                best_crews,
                best_eval,
                cur_prio,
                deadline,
                aggressive=aggressive_ltm,
                best_orders=init_orders,
                min_gain=min_ltm_gain,
                cutoff_step=cutoff_step,
            )
            eval_count += c_ltm
            if best_traffic_w and cutoff_step > 0:
                use_cutoff = cutoff_step

        if _cvcr_now_s() + 0.002 < deadline:
            _best_orders, best_eval, c_ci = self._search_intra_agent_ci(
                snapshot,
                evaluator,
                best_assign,
                best_bays,
                best_crews,
                best_eval,
                cur_prio,
                deadline,
                surr=surr,
                max_candidates=ci_candidates,
            )
            eval_count += c_ci

        guides: dict[str, list[tuple[str, str]]] = {}
        if emit_guides and best_eval.valid and best_eval.robot_paths:
            raw_guides = CVCREvaluator.extract_guide_corridors(
                best_eval.robot_paths, snapshot.tick, horizon_steps=max(20, int(cutoff_horizon or 55))
            )
            for rid, edges in raw_guides.items():
                valid_edges = [(u, v) for (u, v) in edges if self.network.has_edge(u, v)]
                if valid_edges:
                    guides[rid] = valid_edges
            if guides and use_cutoff == 0:
                use_cutoff = int(snapshot.tick) + max(20, int(cutoff_horizon or 55))
        return best_eval, eval_count, best_traffic_w, use_cutoff, guides

    def _search_v9_expert_portfolio(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        base_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        budget_ms: float = 80.0,
    ) -> tuple[
        dict[str, str],
        dict[str, str],
        dict[str, int],
        EvalResult,
        int,
        list[str],
        Optional[list[str]],
        dict[tuple[str, str], int],
        int,
        dict[str, list[tuple[str, str]]],
    ]:
        """V9 Portfolio: CP-SAT joint bottleneck window, windowed LTM, and corridor guide dispatch."""
        best_assign = dict(base_assign)
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
        best_eval = base_eval
        cur_prio = dict(op_prio)
        eval_count = 0

        if len(healthy_robots) < 2:
            evaluator.set_traffic_weights(None)
            return best_assign, best_bays, best_crews, best_eval, eval_count, [], None, {}, 0, {}

        surr = (
            _LazyKittingSurrogate(self.network, snapshot, cur_prio, self.alpha_makespan, self.lambda_change)
            if budget_ms >= 35.0
            else None
        )

        if self.method_id == "V9_E1_FAST_RESCUE":
            rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
            if rel_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                    snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    allow_swaps=False, hys_margin=3.5,
                )
                eval_count += c1
            if _cvcr_now_s() + 0.002 < deadline:
                _ord, best_eval, c_ci = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, max_candidates=8,
                )
                eval_count += c_ci
            evaluator.set_traffic_weights(None)
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None, {}, 0, {}

        if self.method_id == "V9_E2_CRIT_PROD_CI":
            if budget_ms >= 35.0 and _cvcr_now_s() < deadline:
                prod_deadline = min(deadline, _cvcr_now_s() + max(0.012, (float(budget_ms) * 0.56) / 1000.0))
                best_bays, best_crews, cur_prio, best_eval, cb = self._divert_prod_v6(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, prod_deadline
                )
                eval_count += cb
            rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
            snap_by_id = {t.task_id: t for t in snapshot.tasks}
            changed_ops = {
                op.op_id for op in snapshot.operations
                if best_bays[op.op_id] != op.bay_id or best_crews[op.op_id] != op.crew_id
            }
            focused_tids = [
                tid for tid in rel_tids
                if (snap_by_id.get(tid) and snap_by_id[tid].assigned_agv in clf.disturbed_robots)
                or (snap_by_id.get(tid) and snap_by_id[tid].kit_op_id in changed_ops)
                or (snap_by_id.get(tid) and snap_by_id[tid].is_urgent)
            ] or rel_tids[:4]
            if focused_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                    snapshot, clf, focused_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    op_bay_overrides=best_bays, op_crew_overrides=best_crews, allow_swaps=False, hys_margin=4.0,
                )
                eval_count += c1
            if _cvcr_now_s() + 0.002 < deadline:
                _ord, best_eval, c_ci = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, max_candidates=10,
                )
                eval_count += c_ci
            evaluator.set_traffic_weights(None)
            return best_assign, best_bays, best_crews, best_eval, eval_count, focused_tids, None, {}, 0, {}

        if self.method_id == "V9_E4_LTM_GUIDE_CI":
            pacer_deadline = max(_cvcr_now_s(), deadline - (0.014 if budget_ms >= 55.0 else 0.005))
            best_assign, best_bays, best_crews, best_eval, c4, rel_tids, admitted = self._search_congestion_pacer(
                snapshot, clf, evaluator, best_assign, best_eval, healthy_robots, cur_prio, pacer_deadline, budget_ms
            )
            eval_count += max(0, c4 - 1)
            best_eval, c_pol, best_tw, cutoff_step, guides = self._polish_v9_traffic_and_guides(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                surr=surr, aggressive_ltm=True, min_ltm_gain=0.01, min_wait_ltm=8, ci_candidates=8, cutoff_horizon=55, emit_guides=True,
            )
            eval_count += c_pol
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, admitted, best_tw, cutoff_step, guides

        if self.method_id == "V9_E7_CPSAT_JOINT_GUIDE":
            if budget_ms >= 35.0 and _cvcr_now_s() < deadline:
                bay_deadline = min(deadline, _cvcr_now_s() + max(0.008, (float(budget_ms) * 0.28) / 1000.0))
                best_bays, best_crews, cur_prio, best_eval, cb = self._divert_prod_extended(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, bay_deadline
                )
                eval_count += cb
            rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
            if rel_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                    snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    op_bay_overrides=best_bays, op_crew_overrides=best_crews,
                    allow_swaps=(budget_ms >= 35.0), surr=surr, hys_margin=0.0,
                )
                eval_count += c1
            if _cvcr_now_s() + 0.003 < deadline:
                ci_deadline = min(deadline, _cvcr_now_s() + max(0.010, (float(budget_ms) * 0.18) / 1000.0))
                _ord1, best_eval, c_ci1 = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, ci_deadline,
                    surr=surr, max_candidates=8,
                )
                eval_count += c_ci1
            if budget_ms >= 35.0 and surr is not None and self._has_budget(deadline, evaluator, min_ms=16.0):
                best_assign, best_bays, best_crews, cur_prio, _cp_ord, best_eval, c_cp = self._search_local_cpsat_window(
                    snapshot, clf, evaluator, best_assign, best_bays, best_crews, best_eval, healthy_robots, cur_prio, deadline, surr
                )
                eval_count += c_cp
            if budget_ms >= 85.0 and surr is not None and _cvcr_now_s() < deadline:
                sa_deadline = max(_cvcr_now_s(), deadline - 0.005)
                best_assign, best_eval, c_bal = self._search_v8_balance_pc(
                    snapshot, clf, evaluator, best_assign, best_eval, healthy_robots, cur_prio, sa_deadline,
                    best_bays, best_crews, surr, hys_margin=0.0,
                )
                eval_count += c_bal
            best_eval, c_pol, best_tw, cutoff_step, guides = self._polish_v9_traffic_and_guides(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                surr=surr, aggressive_ltm=False, min_ltm_gain=2.5, min_wait_ltm=16, ci_candidates=6, cutoff_horizon=55, emit_guides=True,
            )
            eval_count += c_pol
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None, best_tw, cutoff_step, guides

        # V9_E3_JOINT_PC_HYS, V9_E5_AGGRESSIVE_GUIDE, V9_E6_BALANCE_KM_GUIDE
        is_e3 = self.method_id == "V9_E3_JOINT_PC_HYS"
        is_e5 = self.method_id == "V9_E5_AGGRESSIVE_GUIDE"
        hys = 5.0 if is_e3 else 0.0
        if budget_ms >= 35.0 and _cvcr_now_s() < deadline:
            bay_frac = 0.40 if is_e3 else (0.28 if is_e5 else 0.35)
            bay_deadline = min(deadline, _cvcr_now_s() + max(0.010, (float(budget_ms) * bay_frac) / 1000.0))
            best_bays, best_crews, cur_prio, best_eval, cb = self._divert_prod_extended(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, bay_deadline
            )
            eval_count += cb
        rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
        if rel_tids and _cvcr_now_s() < deadline:
            best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                op_bay_overrides=best_bays, op_crew_overrides=best_crews,
                allow_swaps=(budget_ms >= 35.0), surr=surr, hys_margin=hys,
            )
            eval_count += c1
        if _cvcr_now_s() + 0.003 < deadline:
            ci_deadline = min(deadline, _cvcr_now_s() + max(0.012, (float(budget_ms) * 0.22) / 1000.0))
            _ord1, best_eval, c_ci1 = self._search_intra_agent_ci(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, ci_deadline,
                surr=surr, max_candidates=10,
            )
            eval_count += c_ci1
        if not is_e5 and budget_ms >= 85.0 and surr is not None and _cvcr_now_s() < deadline:
            sa_deadline = max(_cvcr_now_s(), deadline - 0.005)
            best_assign, best_eval, c_bal = self._search_v8_balance_pc(
                snapshot, clf, evaluator, best_assign, best_eval, healthy_robots, cur_prio, sa_deadline,
                best_bays, best_crews, surr, hys_margin=hys,
            )
            eval_count += c_bal
        if is_e3:
            if _cvcr_now_s() + 0.002 < deadline:
                _ord, best_eval, c_ci = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, max_candidates=6,
                )
                eval_count += c_ci
            evaluator.set_traffic_weights(None)
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None, {}, 0, {}

        best_eval, c_pol, best_tw, cutoff_step, guides = self._polish_v9_traffic_and_guides(
            snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
            surr=surr, aggressive_ltm=is_e5, min_ltm_gain=(0.01 if is_e5 else 2.5), min_wait_ltm=(8 if is_e5 else 16),
            ci_candidates=6, cutoff_horizon=55, emit_guides=is_e5,
        )
        eval_count += c_pol
        return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None, best_tw, cutoff_step, guides

    @staticmethod
    def _has_real_resource_overlap(
        snapshot: NativeSnapshot,
        eval_res: EvalResult,
        op_bays: dict[str, str],
        op_crews: dict[str, int],
        min_pairs: int = 1,
    ) -> bool:
        """Return True when unstarted ops exhibit real predicted resource interval contention/tardiness."""
        unstarted = [op for op in snapshot.operations if not op.started]
        if len(unstarted) < 2:
            return False
        windows: list[tuple[int, int, int, str, int, int, bool]] = []
        for op in unstarted:
            oid = op.op_id
            bay = op_bays.get(oid, op.bay_id)
            crew = op_crews.get(oid, op.crew_id)
            tool = int(op.tool_id)
            st = int(eval_res.op_start.get(oid, snapshot.tick))
            en = int(eval_res.op_end.get(oid, st + op.proc_steps))
            kit_rdy = max(
                [snapshot.tick] + [int(eval_res.order_unload_ticks.get(tid, snapshot.tick)) for tid in op.kit_task_ids]
            )
            slack_after_kit = st - kit_rdy
            due_slack = int(op.due_step) - en
            is_pressured = (due_slack <= 6) or (0 <= slack_after_kit <= 2 and en >= op.due_step - 12)
            windows.append((max(snapshot.tick, kit_rdy), st, en, bay, crew, tool, bool(op.is_urgent or is_pressured)))

        overlap_pairs = 0
        for i in range(len(windows)):
            ready_i, si, ei, bi, ci, ti, pi = windows[i]
            for j in range(i + 1, len(windows)):
                ready_j, sj, ej, bj, cj, tj, pj = windows[j]
                if bi != bj and ci != cj and ti != tj:
                    continue
                touch_or_queue = max(si, sj) <= min(ei, ej) + 2
                earliest_overlap = max(ready_i, ready_j) < min(ei, ej)
                if (touch_or_queue or earliest_overlap) and (pi or pj):
                    overlap_pairs += 1
                    if overlap_pairs >= min_pairs:
                        return True
        return False

    def _filter_lrho_releasable(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        eval_res: EvalResult,
        rel_tids: list[str],
        op_bays: dict[str, str],
        op_crews: dict[str, int],
    ) -> tuple[list[str], set[str]]:
        """Filter confidently invariant tasks via L-RHO predictor while never freezing critical/disturbed tasks."""
        if self.lrho_predictor is None or len(rel_tids) <= 2:
            return rel_tids, set()
        frozen = set(self.lrho_predictor.predict_frozen_tasks(snapshot, clf, eval_res, op_bays, op_crews))
        if not frozen:
            return rel_tids, set()
        # Keep non-frozen subset; preserve at least 2 tasks if total releasable >= 2.
        unfrozen = [tid for tid in rel_tids if tid not in frozen]
        if len(unfrozen) >= 2:
            return unfrozen, frozen
        all_unfrozen = [tid for tid in sorted(clf.releasable_tasks) if tid not in frozen]
        if len(all_unfrozen) >= 2:
            return rel_tids, frozen
        return rel_tids, set()

    def _search_v10_expert_portfolio(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        base_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        budget_ms: float = 80.0,
    ) -> tuple[
        dict[str, str],
        dict[str, str],
        dict[str, int],
        EvalResult,
        int,
        list[str],
        Optional[list[str]],
        dict[tuple[str, str], int],
        int,
        dict[str, list[tuple[str, str]]],
        dict[str, int],
    ]:
        """V10 Portfolio: Priority/order closure, L-RHO invariance freeze, and conditional overlap CP-SAT."""
        best_assign = dict(base_assign)
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
        best_eval = base_eval
        cur_prio = dict(op_prio)
        eval_count = 0

        if len(healthy_robots) < 2:
            evaluator.set_traffic_weights(None)
            return best_assign, best_bays, best_crews, best_eval, eval_count, [], None, {}, 0, {}, cur_prio

        surr = (
            _LazyKittingSurrogate(self.network, snapshot, cur_prio, self.alpha_makespan, self.lambda_change)
            if budget_ms >= 35.0
            else None
        )

        if self.method_id == "V10_E1_FAST_RESCUE":
            rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
            if rel_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                    snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    allow_swaps=False, surr=None, hys_margin=0.0,
                )
                eval_count += c1
            if _cvcr_now_s() + 0.002 < deadline:
                _ord, best_eval, c_ci = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, max_candidates=6,
                )
                eval_count += c_ci
            evaluator.set_traffic_weights(None)
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None, {}, 0, {}, cur_prio

        if self.method_id == "V10_E2_CRIT_PROD_CI":
            bay_deadline = min(deadline, _cvcr_now_s() + max(0.010, (float(budget_ms) * 0.42) / 1000.0))
            best_bays, best_crews, cur_prio, best_eval, cb = self._divert_prod_extended(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, bay_deadline
            )
            eval_count += cb
            rel_raw = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
            rel_tids, _ = self._filter_lrho_releasable(snapshot, clf, best_eval, rel_raw, best_bays, best_crews)
            if rel_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                    snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    op_bay_overrides=best_bays, op_crew_overrides=best_crews,
                    allow_swaps=(budget_ms >= 35.0), surr=surr, hys_margin=0.0,
                )
                eval_count += c1
            best_eval, c_pol, best_tw, cutoff_step, guides = self._polish_v9_traffic_and_guides(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                surr=surr, aggressive_ltm=False, min_ltm_gain=2.5, min_wait_ltm=10, ci_candidates=12,
                cutoff_horizon=55, emit_guides=False, preserve_orders=True,
            )
            eval_count += c_pol
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None, best_tw, cutoff_step, guides, cur_prio

        if self.method_id == "V10_E4_LTM_GUIDE_CI":
            pacer_deadline = min(deadline, _cvcr_now_s() + max(0.014, (float(budget_ms) * 0.62) / 1000.0))
            best_assign, best_bays, best_crews, best_eval, c4, rel_tids, admitted = self._search_congestion_pacer(
                snapshot, clf, evaluator, best_assign, best_eval, healthy_robots, cur_prio, pacer_deadline, budget_ms
            )
            eval_count += max(0, c4 - 1)
            best_eval, c_pol, best_tw, cutoff_step, guides = self._polish_v9_traffic_and_guides(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                surr=surr, aggressive_ltm=True, min_ltm_gain=0.01, min_wait_ltm=8, ci_candidates=8,
                cutoff_horizon=55, emit_guides=True, preserve_orders=True,
            )
            eval_count += c_pol
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, admitted, best_tw, cutoff_step, guides, cur_prio

        # Parameterized handling for V10_E3, V10_E5, V10_E6_BALANCE_KM_LRHO, V10_E7_COND_CPSAT
        is_e3 = self.method_id == "V10_E3_JOINT_PC_HYS"
        is_e5 = self.method_id == "V10_E5_AGGRESSIVE_GUIDE"
        is_e7 = self.method_id == "V10_E7_COND_CPSAT"
        hys = 5.0 if is_e3 else 0.0
        cp_mode = self.cp_mode or ("cond_overlap" if is_e7 else "skip")

        if budget_ms >= 35.0 and _cvcr_now_s() < deadline:
            bay_frac = 0.36 if is_e3 else (0.28 if is_e5 else ((0.28 if budget_ms < 85.0 else 0.24) if is_e7 else 0.34))
            bay_deadline = min(deadline, _cvcr_now_s() + min(0.012, max(0.008, (float(budget_ms) * bay_frac) / 1000.0)))
            best_bays, best_crews, cur_prio, best_eval, cb = self._divert_prod_extended(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, bay_deadline
            )
            eval_count += cb

        rel_raw = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
        rel_tids, frozen_tids = self._filter_lrho_releasable(snapshot, clf, best_eval, rel_raw, best_bays, best_crews)

        def _maybe_run_cp(cur_assign, cur_bays, cur_crews, cur_ev, cur_pr):
            if not is_e7 or cp_mode == "skip" or surr is None:
                return cur_assign, cur_bays, cur_crews, cur_ev, cur_pr, 0
            if cp_mode == "always_50":
                allow_b = budget_ms >= 35.0 and self._has_budget(deadline, evaluator, min_ms=16.0)
            elif cp_mode == "always":
                allow_b = budget_ms >= 85.0 and self._has_budget(deadline, evaluator, min_ms=16.0)
            else:
                allow_b = (
                    budget_ms >= 85.0
                    and self._has_budget(deadline, evaluator, min_ms=28.0)
                    and self._has_real_resource_overlap(snapshot, cur_ev, cur_bays, cur_crews)
                )
            if not allow_b:
                return cur_assign, cur_bays, cur_crews, cur_ev, cur_pr, 0
            rem_s = deadline - _cvcr_now_s()
            cp_deadline = (deadline - 0.014) if rem_s > 0.028 else deadline
            na, nb, nc, npr, _cp_ord, nev, c_cp = self._search_local_cpsat_window(
                snapshot, clf, evaluator, cur_assign, cur_bays, cur_crews, cur_ev, healthy_robots, cur_pr,
                cp_deadline, surr, allow_order_only=True, min_improve=0.5, solve_time_cap_s=0.014, disable_presolve=True,
            )
            return na, nb, nc, nev, npr, c_cp

        if is_e7 and cp_mode == "cond_early":
            if rel_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                    snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    op_bay_overrides=best_bays, op_crew_overrides=best_crews,
                    allow_swaps=(budget_ms >= 35.0), surr=surr, hys_margin=0.0,
                )
                eval_count += c1
            best_assign, best_bays, best_crews, best_eval, cur_prio, c_cp = _maybe_run_cp(
                best_assign, best_bays, best_crews, best_eval, cur_prio
            )
            eval_count += c_cp
            if _cvcr_now_s() + 0.003 < deadline:
                ci_deadline = min(deadline, _cvcr_now_s() + max(0.008, (float(budget_ms) * 0.14) / 1000.0))
                _ord1, best_eval, c_ci1 = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, ci_deadline,
                    surr=surr, max_candidates=8,
                )
                eval_count += c_ci1
        else:
            if rel_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                    snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    op_bay_overrides=best_bays, op_crew_overrides=best_crews,
                    allow_swaps=(budget_ms >= 35.0), surr=surr, hys_margin=hys,
                )
                eval_count += c1
            if _cvcr_now_s() + 0.003 < deadline:
                ci_frac = (0.18 if budget_ms < 85.0 else 0.14) if is_e7 else 0.20
                ci_deadline = min(deadline, _cvcr_now_s() + max(0.008, (float(budget_ms) * ci_frac) / 1000.0))
                _ord1, best_eval, c_ci1 = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, ci_deadline,
                    surr=surr, max_candidates=(8 if is_e7 else 10),
                )
                eval_count += c_ci1
            best_assign, best_bays, best_crews, best_eval, cur_prio, c_cp = _maybe_run_cp(
                best_assign, best_bays, best_crews, best_eval, cur_prio
            )
            eval_count += c_cp

        if not is_e5 and budget_ms >= 85.0 and surr is not None and _cvcr_now_s() < deadline:
            sa_deadline = max(_cvcr_now_s(), deadline - 0.006)
            best_assign, best_eval, c_bal = self._search_v8_balance_pc(
                snapshot, clf, evaluator, best_assign, best_eval, healthy_robots, cur_prio, sa_deadline,
                best_bays, best_crews, surr, hys_margin=hys, preserve_orders=True, lrho_frozen=frozen_tids or None,
            )
            eval_count += c_bal

        if is_e3:
            if _cvcr_now_s() + 0.002 < deadline:
                _ord, best_eval, c_ci = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, max_candidates=6,
                )
                eval_count += c_ci
            evaluator.set_traffic_weights(None)
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None, {}, 0, {}, cur_prio

        best_eval, c_pol, best_tw, cutoff_step, guides = self._polish_v9_traffic_and_guides(
            snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
            surr=surr, aggressive_ltm=is_e5, min_ltm_gain=(0.01 if is_e5 else 2.5), min_wait_ltm=(8 if is_e5 else 16),
            ci_candidates=6, cutoff_horizon=55, emit_guides=True, preserve_orders=True,
        )
        eval_count += c_pol
        return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None, best_tw, cutoff_step, guides, cur_prio

    def _search_v12_expert_portfolio(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        base_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        budget_ms: float = 80.0,
    ) -> tuple[
        dict[str, str],
        dict[str, str],
        dict[str, int],
        EvalResult,
        int,
        list[str],
        Optional[list[str]],
        dict[tuple[str, str], int],
        int,
        dict[str, list[tuple[str, str]]],
        dict[str, int],
    ]:
        """V12 Top-Venue Synergistic Portfolio: Detour-Delta, HeteroDestroy, TPG Guides & Safe CP-SAT."""
        from amr_sched.cvcr.v12_destroy_selector import load_hetero_destroy_selector
        from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals

        best_assign = dict(base_assign)
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
        best_eval = base_eval
        cur_prio = dict(op_prio)
        eval_count = 0

        if len(healthy_robots) < 1:
            evaluator.set_traffic_weights(None)
            return best_assign, best_bays, best_crews, best_eval, eval_count, [], None, {}, 0, {}, cur_prio

        # Compute causal Detour-Delta bay & edge signals cleanly from current snapshot + weights
        dd_sig = compute_detour_delta_signals(self.network, snapshot)
        bay_delta = dict(self.bay_detour_delta) if self.bay_detour_delta is not None else dict(dd_sig["bay_detour_delta"])
        tw_seed = dict(self.detour_weights_q3) if self.detour_weights_q3 is not None else dict(dd_sig["weights_q3"])
        for ek, wv in (self._prev_ltm_weights or {}).items():
            if wv > 2:
                tw_seed[ek] = max(tw_seed.get(ek, 0), min(32, int(round(0.45 * wv))))

        # Mode 0: Zero-task-churn surgical Switchable-TPG / corridor guide & live-repair shell
        if self.method_id == "V12_SURGICAL_TPG_REPAIR":
            cutoff_step = int(snapshot.tick) + 35
            fb_s = getattr(snapshot, "execution_feedback", None)
            if fb_s is not None:
                fb_b = fb_s.blocked_intervals(t_commit=int(snapshot.tick), default_horizon=12)
                if fb_b:
                    for ek_fb, wv_fb in fb_s.corridor_q3_penalties().items():
                        tw_seed[ek_fb] = max(tw_seed.get(ek_fb, 0), int(wv_fb))
                    cutoff_step = max(cutoff_step, max(int(en_fb) for en_fb in fb_b.values()))
            if tw_seed and self._has_budget(deadline, evaluator, min_ms=2.5):
                evaluator.set_traffic_weights(tw_seed, cutoff_step=cutoff_step)
                res_tw = evaluator.evaluate(
                    best_assign,
                    op_priority=cur_prio,
                    op_bay_overrides=best_bays,
                    op_crew_overrides=best_crews,
                    task_order_override={r: list(s) for r, s in best_eval.robot_order_seq.items() if s} or None,
                )
                eval_count += 1
                if res_tw.valid and res_tw.objective <= best_eval.objective + 1.0:
                    best_eval = res_tw
                else:
                    evaluator.set_traffic_weights(None)
                    tw_seed = {}
            guides: dict[str, list[tuple[str, str]]] = {}
            if best_eval.valid and best_eval.robot_paths:
                raw_guides = CVCREvaluator.extract_guide_corridors(best_eval.robot_paths, snapshot.tick, horizon_steps=35)
                for rid, edges in raw_guides.items():
                    ve = [(u, v) for (u, v) in edges if self.network.has_edge(u, v)]
                    if ve:
                        guides[rid] = ve
            return best_assign, best_bays, best_crews, best_eval, eval_count, [], None, tw_seed, (cutoff_step if (tw_seed or guides) else 0), guides, cur_prio

        if len(healthy_robots) < 2:
            evaluator.set_traffic_weights(tw_seed or None, cutoff_step=(snapshot.tick + 40 if tw_seed else 0))
            return best_assign, best_bays, best_crews, best_eval, eval_count, [], None, tw_seed, (snapshot.tick + 40 if tw_seed else 0), {}, cur_prio

        surr = _LazyKittingSurrogate(
            self.network,
            snapshot,
            cur_prio,
            self.alpha_makespan,
            self.lambda_change,
            bay_detour_delta=bay_delta,
        )

        # Fast disturbed rescue with Detour-Delta scoring (20ms)
        if self.method_id == "V12_RESCUE_DETOUR_FAST":
            rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
            if rel_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                    snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    allow_swaps=False, surr=surr, hys_margin=0.0,
                )
                eval_count += c1
            if _cvcr_now_s() + 0.002 < deadline:
                _ord, best_eval, c_ci = self._search_intra_agent_ci(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, max_candidates=6,
                )
                eval_count += c_ci
            guides_f = {}
            if (tw_seed or bay_delta) and best_eval.valid and best_eval.robot_paths:
                for rid, edges in CVCREvaluator.extract_guide_corridors(best_eval.robot_paths, snapshot.tick, horizon_steps=40).items():
                    ve = [(u, v) for (u, v) in edges if self.network.has_edge(u, v)]
                    if ve:
                        guides_f[rid] = ve
            if not tw_seed:
                evaluator.set_traffic_weights(None)
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None, tw_seed, (snapshot.tick + 40 if tw_seed else 0), guides_f, cur_prio

        # Conservative congestion pacer (50ms)
        if self.method_id == "V12_PACER_CONSERVATIVE":
            pacer_dl = min(deadline, _cvcr_now_s() + max(0.014, (float(budget_ms) * 0.62) / 1000.0))
            best_assign, best_bays, best_crews, best_eval, c4, rel_tids, admitted = self._search_congestion_pacer(
                snapshot, clf, evaluator, best_assign, best_eval, healthy_robots, cur_prio, pacer_dl, budget_ms
            )
            eval_count += max(0, c4 - 1)
            best_eval, c_pol, best_tw, cutoff_step, guides = self._polish_v9_traffic_and_guides(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                surr=surr, aggressive_ltm=False, min_ltm_gain=2.5, min_wait_ltm=10, ci_candidates=8,
                cutoff_horizon=50, emit_guides=True, preserve_orders=True,
            )
            eval_count += c_pol
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, admitted, best_tw, cutoff_step, guides, cur_prio

        # Production + Heterogeneous Destroy LNS variants
        is_small = self.method_id == "V12_CRIT_PROD_DESTROY_S"
        is_large = self.method_id == "V12_BALANCED_DESTROY_L"
        is_aggr = self.method_id == "V12_AGGRESSIVE_GUIDE_TPG"
        is_cp = self.method_id == "V12_SAFE_CPSAT_JOINT"
        hys = 0.0

        if is_small:
            bay_dl = min(deadline, _cvcr_now_s() + max(0.010, (float(budget_ms) * 0.42) / 1000.0))
            best_bays, best_crews, cur_prio, best_eval, cb = self._divert_prod_extended(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, bay_dl
            )
            eval_count += cb
            rel_raw = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
            rel_tids, _ = self._filter_lrho_releasable(snapshot, clf, best_eval, rel_raw, best_bays, best_crews)
            if rel_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                    snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    op_bay_overrides=best_bays, op_crew_overrides=best_crews,
                    allow_swaps=(budget_ms >= 35.0), surr=surr, hys_margin=0.0,
                )
                eval_count += c1
            best_eval, c_pol, best_tw, cutoff_step, guides = self._polish_v9_traffic_and_guides(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                surr=surr, aggressive_ltm=False, min_ltm_gain=2.5, min_wait_ltm=10, ci_candidates=12,
                cutoff_horizon=55, emit_guides=False, preserve_orders=True,
            )
            eval_count += c_pol
            for ek, wv in tw_seed.items():
                best_tw[ek] = max(best_tw.get(ek, 0), int(wv))
            if best_tw and cutoff_step == 0:
                cutoff_step = int(snapshot.tick) + 55
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None, best_tw, cutoff_step, guides, cur_prio

        if _cvcr_now_s() < deadline:
            bay_dl = min(deadline, _cvcr_now_s() + 0.012)
            best_bays, best_crews, cur_prio, best_eval, cb = self._divert_prod_extended(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, bay_dl
            )
            eval_count += cb

        rel_raw = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
        rel_lrho, frozen_lrho = self._filter_lrho_releasable(snapshot, clf, best_eval, rel_raw, best_bays, best_crews)
        d_sel = self.destroy_selector if self.destroy_selector is not None else load_hetero_destroy_selector()
        radius = "large_stochastic" if is_large else "medium"
        rng_d = random.Random(self.seed + snapshot.tick * 131 + 17)
        d_tids, frozen_het, _scores = d_sel.select_destroy_tasks(
            self.network,
            snapshot,
            clf,
            best_eval,
            best_bays,
            best_crews,
            radius=radius,
            bay_detour_delta=bay_delta,
            rng=rng_d,
        )
        d_set = set(d_tids)
        rel_tids = [t for t in rel_lrho if t in d_set]
        if len(rel_tids) < 2:
            rel_tids = d_tids if len(d_tids) >= 2 else rel_lrho
        frozen_tids = set(frozen_lrho) if frozen_lrho else set(frozen_het)

        if rel_tids and _cvcr_now_s() < deadline:
            best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                op_bay_overrides=best_bays, op_crew_overrides=best_crews,
                allow_swaps=(budget_ms >= 35.0), surr=surr, hys_margin=hys,
            )
            eval_count += c1

        if is_cp and _cvcr_now_s() + 0.003 < deadline:
            ci_dl = min(deadline, _cvcr_now_s() + max(0.008, (float(budget_ms) * 0.14) / 1000.0))
            _ord1, best_eval, c_ci = self._search_intra_agent_ci(
                snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, ci_dl,
                surr=surr, max_candidates=6,
            )
            eval_count += c_ci

        # Safe closed-loop CP-SAT (strict headroom, non-focal queue preservation, higher improve guard)
        if (
            is_cp
            and budget_ms >= 85.0
            and self._has_budget(deadline, evaluator, min_ms=26.0)
            and self._has_real_resource_overlap(snapshot, best_eval, best_bays, best_crews)
        ):
            rem_s = deadline - _cvcr_now_s()
            cp_dl = (deadline - 0.016) if rem_s > 0.030 else deadline
            best_assign, best_bays, best_crews, cur_prio, _cp_ord, best_eval, c_cp = self._search_local_cpsat_window(
                snapshot, clf, evaluator, best_assign, best_bays, best_crews, best_eval, healthy_robots, cur_prio,
                cp_dl, surr, allow_order_only=True, min_improve=8.0, solve_time_cap_s=0.014,
                disable_presolve=True, preserve_nonfocal_order=True,
            )
            eval_count += c_cp

        if not is_small and not is_aggr and budget_ms >= 80.0 and _cvcr_now_s() < deadline:
            sa_dl = max(_cvcr_now_s(), deadline - 0.006)
            best_assign, best_eval, c_bal = self._search_v8_balance_pc(
                snapshot, clf, evaluator, best_assign, best_eval, healthy_robots, cur_prio, sa_dl,
                best_bays, best_crews, surr, hys_margin=hys, preserve_orders=True, lrho_frozen=frozen_tids or None,
            )
            eval_count += c_bal

        best_eval, c_pol, best_tw, cutoff_step, guides = self._polish_v9_traffic_and_guides(
            snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
            surr=surr, aggressive_ltm=is_aggr, min_ltm_gain=(0.01 if is_aggr else 2.5),
            min_wait_ltm=(8 if is_aggr else 14), ci_candidates=8, cutoff_horizon=50,
            emit_guides=True, preserve_orders=True,
        )
        eval_count += c_pol
        for ek, wv in tw_seed.items():
            best_tw[ek] = max(best_tw.get(ek, 0), int(wv))
        if best_tw and cutoff_step == 0:
            cutoff_step = int(snapshot.tick) + 50
        return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None, best_tw, cutoff_step, guides, cur_prio

    def _search_v13_kitting_rl(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        base_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        budget_ms: float = 80.0,
    ) -> tuple[
        dict[str, str],
        dict[str, str],
        dict[str, int],
        EvalResult,
        int,
        list[str],
        Optional[list[str]],
        dict[tuple[str, str], int],
        int,
        dict[str, list[tuple[str, str]]],
        dict[str, int],
    ]:
        """V13 Hierarchical Production-Kitting Protection & Joint Neighborhood RL Search."""
        from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
        from amr_sched.cvcr.v13_kitting_rl import (
            extract_v13_global_features,
            generate_kitting_joint_candidates,
            load_kitting_window_ranker,
        )

        best_assign = dict(base_assign)
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
        best_eval = base_eval
        cur_prio = dict(op_prio)
        eval_count = 0

        if len(healthy_robots) < 1:
            return best_assign, best_bays, best_crews, best_eval, eval_count, [], None, {}, 0, {}, cur_prio

        dd_sig = compute_detour_delta_signals(self.network, snapshot)
        bay_delta = dict(self.bay_detour_delta) if self.bay_detour_delta is not None else dict(dd_sig["bay_detour_delta"])
        tw_seed = dict(self.detour_weights_q3) if self.detour_weights_q3 is not None else dict(dd_sig["weights_q3"])
        surr = _LazyKittingSurrogate(
            self.network,
            snapshot,
            cur_prio,
            self.alpha_makespan,
            self.lambda_change,
            bay_detour_delta=bay_delta,
        )

        if self.method_id == "V13_KITTING_DEFAULT":
            rel_tids = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
            if rel_tids and _cvcr_now_s() < deadline:
                best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                    snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                    op_bay_overrides=best_bays, op_crew_overrides=best_crews,
                    allow_swaps=False, surr=surr, hys_margin=0.0,
                )
                eval_count += c1
            self._last_kitting_decision = {
                "cand_id": "keep_default",
                "source_kind": "default_short",
                "target_op_id": None,
                "n_candidates": 1,
                "advanced_op_steps": 0.0,
                "yielded_op_steps": 0.0,
                "cliff_saved": False,
                "continue_search": False,
            }
            return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None, tw_seed, (int(snapshot.tick) + 40 if tw_seed else 0), {}, cur_prio

        cands = generate_kitting_joint_candidates(
            self.network,
            snapshot,
            clf,
            evaluator,
            surr,
            base_assign,
            best_bays,
            best_crews,
            base_eval,
            cur_prio,
            healthy_robots,
            bay_delta,
            max_protected_ops=3,
            lambda_change=self.lambda_change,
            include_learned_proposal=(self.method_id in ("V13_KITTING_SUP", "V13_KITTING_RL")),
        )
        eval_count += max(0, len(cands) - 1)

        chosen = cands[0]
        cont_prob = 0.0
        if self.method_id == "V13_KITTING_DET":
            valid_cands = [c for c in cands if c.eval_res.valid]
            if valid_cands:
                chosen = max(valid_cands, key=lambda c: c.det_score)
            cont_prob = 1.0 if budget_ms >= 55.0 else 0.0
        else:
            ranker = self.kitting_ranker
            if ranker is None:
                mode = "sup_ranker" if self.method_id == "V13_KITTING_SUP" else "smdp_rl"
                ranker = load_kitting_window_ranker(mode=mode)
            g_feats = extract_v13_global_features(snapshot, base_eval, clf, bay_delta)
            scores, cont_prob = ranker.score_candidates(g_feats, cands)
            chosen = cands[int(np.argmax(scores))]

        best_assign = dict(chosen.assignment)
        best_bays = dict(chosen.bay_overrides)
        best_crews = dict(chosen.crew_overrides)
        cur_prio = dict(chosen.op_priority)
        best_eval = chosen.eval_res
        # On standard topologies without discrete resource windows, only keep aggressive op_priority
        # shifts if they beat the base objective by >= 8.0 or truly save a due-cliff; otherwise let V12
        # multi-operator search refine from the cleaner base state without distorting priorities.
        has_res_win = any(bool(getattr(op, "resource_windows", ())) for op in snapshot.operations if not op.started)
        if (
            not has_res_win
            and not chosen.discrete_cliff_saved
            and base_eval.valid
            and chosen.source_kind != "keep_default"
            and chosen.eval_res.objective > base_eval.objective - 8.0
        ):
            best_assign = dict(base_assign)
            best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
            best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
            cur_prio = dict(op_prio)
            best_eval = base_eval

        rel_tids = list(chosen.destroy_tids) or self._prioritize_disturbed(
            snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval)
        )
        surr.op_prio = dict(cur_prio)
        if best_eval.valid:
            evaluator.set_incumbent(
                best_assign,
                best_eval,
                op_priority=cur_prio,
                op_bay_overrides=best_bays,
                task_order_override=chosen.task_order_override if best_eval is chosen.eval_res else None,
            )

        # DR-ALNS anytime continue-vs-commit gate + V12 multi-operator backend fusion (Step 4)
        from amr_sched.cvcr.v12_destroy_selector import load_hetero_destroy_selector

        has_disturbed = bool(clf.disturbed_robots)
        any_unassigned = any(t.status.lower() != "completed" and t.assigned_agv is None for t in snapshot.tasks)
        hys_v13 = 0.0 if (has_disturbed or not base_eval.valid) else 5.0
        skip_calm_keep = (
            (chosen.source_kind == "keep_default")
            and base_eval.valid
            and (not has_disturbed)
            and (not has_res_win)
            and (not any_unassigned)
            and int(snapshot.tick) > 16
        )
        do_continue = (
            (not skip_calm_keep)
            and budget_ms >= 35.0
            and (cont_prob >= 0.40 or self.method_id == "V13_KITTING_DET" or has_disturbed)
            and self._has_budget(deadline, evaluator, min_ms=6.0)
        )
        # Preserve V13's tight protective neighborhood on discrete resource-window problems or
        # explicit cliff-saving joint proposals; engage V12's heavy multi-operator backend when
        # no discrete cliff protection was selected (standard/unseen/flow-inversion topologies).
        preserve_v13_focus = has_res_win or (
            chosen.discrete_cliff_saved and chosen.source_kind in ("protect_op_joint", "learned_swap_joint")
        )
        if do_continue:
            if preserve_v13_focus:
                if rel_tids and _cvcr_now_s() < deadline:
                    best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                        snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                        op_bay_overrides=best_bays, op_crew_overrides=best_crews,
                        allow_swaps=(budget_ms >= 55.0), surr=surr, hys_margin=hys_v13,
                    )
                    eval_count += c1
                best_eval, c_pol, best_tw, cutoff_step, guides = self._polish_v9_traffic_and_guides(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, aggressive_ltm=False, min_ltm_gain=2.5, min_wait_ltm=12, ci_candidates=6,
                    cutoff_horizon=50, emit_guides=True, preserve_orders=True,
                )
                eval_count += c_pol
                for ek, wv in tw_seed.items():
                    best_tw[ek] = max(best_tw.get(ek, 0), int(wv))
                if best_tw and cutoff_step == 0:
                    cutoff_step = int(snapshot.tick) + 50
            else:
                if _cvcr_now_s() < deadline:
                    bay_dl = min(deadline, _cvcr_now_s() + 0.012)
                    best_bays, best_crews, cur_prio, best_eval, cb = self._divert_prod_extended(
                        snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, bay_dl
                    )
                    eval_count += cb
                    surr.op_prio = dict(cur_prio)

                rel_raw = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
                rel_lrho, frozen_lrho = self._filter_lrho_releasable(snapshot, clf, best_eval, rel_raw, best_bays, best_crews)
                d_sel = self.destroy_selector if self.destroy_selector is not None else load_hetero_destroy_selector()
                radius = "large_stochastic" if budget_ms >= 90.0 else "medium"
                rng_d = random.Random(self.seed + snapshot.tick * 131 + 17)
                d_tids, frozen_het, _scores = d_sel.select_destroy_tasks(
                    self.network,
                    snapshot,
                    clf,
                    best_eval,
                    best_bays,
                    best_crews,
                    radius=radius,
                    bay_detour_delta=bay_delta,
                    rng=rng_d,
                )
                d_set = set(d_tids)
                merged_rel = list(dict.fromkeys(list(chosen.destroy_tids) + [t for t in rel_lrho if t in d_set]))
                if len(merged_rel) < 2:
                    merged_rel = list(dict.fromkeys(list(chosen.destroy_tids) + (d_tids if len(d_tids) >= 2 else rel_lrho)))
                rel_tids = merged_rel or rel_tids
                frozen_tids = set(frozen_lrho) if frozen_lrho else set(frozen_het)

                if rel_tids and _cvcr_now_s() < deadline:
                    best_assign, best_eval, c1 = self._run_1opt_and_swap_ranked(
                        snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                        op_bay_overrides=best_bays, op_crew_overrides=best_crews,
                        allow_swaps=(budget_ms >= 35.0), surr=surr, hys_margin=hys_v13,
                    )
                    eval_count += c1

                if len(healthy_robots) >= 2 and _cvcr_now_s() + 0.003 < deadline:
                    ci_dl = min(deadline, _cvcr_now_s() + max(0.008, (float(budget_ms) * 0.14) / 1000.0))
                    _ord1, best_eval, c_ci = self._search_intra_agent_ci(
                        snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, ci_dl,
                        surr=surr, max_candidates=6,
                    )
                    eval_count += c_ci

                if (
                    len(healthy_robots) >= 2
                    and budget_ms >= 85.0
                    and self._has_budget(deadline, evaluator, min_ms=26.0)
                    and self._has_real_resource_overlap(snapshot, best_eval, best_bays, best_crews)
                ):
                    rem_s = deadline - _cvcr_now_s()
                    cp_dl = (deadline - 0.016) if rem_s > 0.030 else deadline
                    best_assign, best_bays, best_crews, cur_prio, _cp_ord, best_eval, c_cp = self._search_local_cpsat_window(
                        snapshot, clf, evaluator, best_assign, best_bays, best_crews, best_eval, healthy_robots, cur_prio,
                        cp_dl, surr, allow_order_only=True, min_improve=8.0, solve_time_cap_s=0.014,
                        disable_presolve=True, preserve_nonfocal_order=True,
                    )
                    eval_count += c_cp

                if len(healthy_robots) >= 2 and budget_ms >= 75.0 and _cvcr_now_s() < deadline:
                    sa_dl = max(_cvcr_now_s(), deadline - 0.006)
                    best_assign, best_eval, c_bal = self._search_v8_balance_pc(
                        snapshot, clf, evaluator, best_assign, best_eval, healthy_robots, cur_prio, sa_dl,
                        best_bays, best_crews, surr, hys_margin=hys_v13, preserve_orders=True, lrho_frozen=frozen_tids or None,
                    )
                    eval_count += c_bal

                best_eval, c_pol, best_tw, cutoff_step, guides = self._polish_v9_traffic_and_guides(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, aggressive_ltm=False, min_ltm_gain=2.5, min_wait_ltm=12, ci_candidates=8,
                    cutoff_horizon=50, emit_guides=True, preserve_orders=True,
                )
                eval_count += c_pol
                for ek, wv in tw_seed.items():
                    best_tw[ek] = max(best_tw.get(ek, 0), int(wv))
                if best_tw and cutoff_step == 0:
                    cutoff_step = int(snapshot.tick) + 50
        else:
            best_tw = tw_seed
            cutoff_step = int(snapshot.tick) + 45 if tw_seed else 0
            guides = {}
            if best_eval.valid and best_eval.robot_paths:
                for rid, edges in CVCREvaluator.extract_guide_corridors(best_eval.robot_paths, snapshot.tick, horizon_steps=40).items():
                    ve = [(u, v) for (u, v) in edges if self.network.has_edge(u, v)]
                    if ve:
                        guides[rid] = ve

        self._last_kitting_decision = {
            "cand_id": chosen.cand_id,
            "source_kind": chosen.source_kind,
            "target_op_id": chosen.target_op_id,
            "n_candidates": len(cands),
            "advanced_op_steps": round(chosen.advanced_op_steps, 2),
            "yielded_op_steps": round(chosen.yielded_op_steps, 2),
            "cliff_saved": bool(chosen.discrete_cliff_saved),
            "continue_search": bool(do_continue),
            "continue_prob": round(float(cont_prob), 4),
        }
        return best_assign, best_bays, best_crews, best_eval, eval_count, rel_tids, None, best_tw, cutoff_step, guides, cur_prio


    def _search_v14_joint_rl(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        best_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        budget_ms: float = 100.0,
    ) -> tuple[
        dict[str, str],
        dict[str, str],
        dict[str, int],
        EvalResult,
        int,
        list[str],
        Optional[list[str]],
        dict[tuple[str, str], int],
        int,
        dict[str, list[tuple[str, str]]],
        dict[str, int],
    ]:
        """Run 3-layer V14 Hierarchical Joint RL search (Op Portfolio + Set-Attention + DR-ALNS)."""
        from amr_sched.cvcr.v12_destroy_selector import load_hetero_destroy_selector
        from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
        from amr_sched.cvcr.v13_kitting_rl import extract_v13_global_features
        from amr_sched.cvcr.v14_joint_rl import (
            ANYTIME_OPS,
            V14JointPolicy,
            extract_v14_op_features,
            generate_v14_joint_candidates,
            load_v14_joint_policy,
        )

        best_assign = dict(base_assign)
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
        base_eval = best_eval
        cur_prio = dict(op_prio)
        eval_count = 0

        if len(healthy_robots) < 1:
            return best_assign, best_bays, best_crews, best_eval, eval_count, [], None, {}, 0, {}, cur_prio

        dd_sig = compute_detour_delta_signals(self.network, snapshot)
        b_delta = dict(self.bay_detour_delta) if self.bay_detour_delta is not None else dict(dd_sig["bay_detour_delta"])
        tw_seed = dict(self.detour_weights_q3) if self.detour_weights_q3 is not None else dict(dd_sig["weights_q3"])

        if self.v14_policy is not None:
            v14_pol = self.v14_policy
        elif isinstance(self.kitting_ranker, V14JointPolicy):
            v14_pol = self.kitting_ranker
        else:
            v14_pol = load_v14_joint_policy()

        pre_lats = len(evaluator.eval_lats_ms)
        initial_obj = float(best_eval.objective) if best_eval.valid else 1e9
        surr = _LazyKittingSurrogate(
            self.network,
            snapshot,
            cur_prio,
            self.alpha_makespan,
            self.lambda_change,
            bay_detour_delta=b_delta,
        )

        # Layer 1 + Layer 2: Op-window portfolio + active RL Swap/Drift proposals
        cands = generate_v14_joint_candidates(
            self.network,
            snapshot,
            clf,
            evaluator,
            surr,
            best_assign,
            best_bays,
            best_crews,
            best_eval,
            cur_prio,
            healthy_robots,
            bay_delta=b_delta,
            policy=v14_pol,
            lambda_change=self.lambda_change,
        )
        if not cands:
            rel_empty = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
            return best_assign, best_bays, best_crews, best_eval, 1, rel_empty, None, tw_seed, (int(snapshot.tick) + 45 if tw_seed else 0), {}, cur_prio

        succ_counts: dict[str, int] = {op.op_id: 0 for op in snapshot.operations}
        for op in snapshot.operations:
            for pid in op_predecessors(op):
                if pid in succ_counts:
                    succ_counts[pid] += 1
        unstarted, op_feats = extract_v14_op_features(snapshot, best_eval, b_delta, succ_counts)
        g_feats = extract_v13_global_features(snapshot, best_eval, clf, b_delta)
        op_prot_top: Optional[str] = None
        if len(unstarted) > 0:
            prot_l, _ = v14_pol.score_operations(op_feats, g_feats)
            op_prot_top = unstarted[int(np.argmax(prot_l))].op_id

        budget_s = max(0.01, float(budget_ms) / 1000.0)
        rem_ratio = max(0.0, min(1.0, (deadline - _cvcr_now_s()) / budget_s))
        scores, cont_prob = v14_pol.score_candidates(
            g_feats,
            cands,
            snapshot,
            best_eval,
            b_delta,
            rem_budget_ratio=rem_ratio,
        )
        best_i = int(np.argmax(scores))
        chosen = cands[best_i]

        # Causal regret safety shield against DET best on extreme out-of-distribution jumps
        valid_cands = [c for c in cands if c.eval_res.valid]
        det_best = max(valid_cands, key=lambda c: c.det_score) if valid_cands else cands[0]
        if (
            chosen is not det_best
            and det_best.eval_res.valid
            and chosen.eval_res.valid
            and not (chosen.discrete_cliff_saved and not det_best.discrete_cliff_saved)
            and chosen.eval_res.objective > det_best.eval_res.objective + 10.0
        ):
            chosen = det_best
            best_i = cands.index(det_best)

        best_assign = dict(chosen.assignment)
        best_bays = dict(chosen.bay_overrides)
        best_crews = dict(chosen.crew_overrides)
        cur_prio = dict(chosen.op_priority)
        best_eval = chosen.eval_res

        # Same standard-topology distortion guard as V13_KITTING_DET
        has_res_win = any(bool(getattr(op, "resource_windows", ())) for op in snapshot.operations if not op.started)
        if (
            not has_res_win
            and not chosen.discrete_cliff_saved
            and base_eval.valid
            and chosen.source_kind != "keep_default"
            and chosen.eval_res.objective > base_eval.objective - 8.0
        ):
            best_assign = dict(base_assign)
            best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
            best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
            cur_prio = dict(op_prio)
            best_eval = base_eval

        rel_tids = list(chosen.destroy_tids) or self._prioritize_disturbed(
            snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval)
        )
        surr.op_prio = dict(cur_prio)
        if best_eval.valid:
            evaluator.set_incumbent(
                best_assign,
                best_eval,
                op_priority=cur_prio,
                op_bay_overrides=best_bays,
                task_order_override=chosen.task_order_override if best_eval is chosen.eval_res else None,
            )

        # Layer 3: DR-ALNS Anytime Operator Fusion over Proven V13+V12 Safety Backbone
        has_disturbed = bool(clf.disturbed_robots)
        has_dist_f = 1.0 if (has_disturbed or any(r.state in ("faulted", "blocked") for r in snapshot.robots)) else 0.0
        any_unassigned = any(t.status.lower() != "completed" and t.assigned_agv is None for t in snapshot.tasks)
        hys_v13 = 0.0 if (has_disturbed or not base_eval.valid) else 5.0
        skip_calm_keep = (
            (chosen.source_kind == "keep_default")
            and best_eval.valid
            and (not has_disturbed)
            and (not has_res_win)
            and (not any_unassigned)
            and int(snapshot.tick) > 16
        )

        rem_r = max(0.0, min(1.0, (deadline - _cvcr_now_s()) / budget_s))
        gain_ratio = float(
            np.clip(
                (initial_obj - (best_eval.objective if best_eval.valid else initial_obj))
                / max(1.0, abs(initial_obj)),
                -0.5,
                1.0,
            )
        )
        search_state = np.asarray(
            [
                rem_r,
                min(1.0, len(cands) / 10.0),
                gain_ratio,
                0.0,
                1.0 if has_res_win else 0.0,
                has_dist_f,
                float(np.clip(cont_prob, 0.0, 1.0)),
                min(1.0, len(chosen.destroy_tids) / 8.0),
            ],
            dtype=np.float32,
        )
        preserve_v13_focus = has_res_win or (
            chosen.discrete_cliff_saved and chosen.source_kind in ("protect_op_joint", "rl_swap_joint", "learned_swap_joint")
        )
        allowed_any = [0, 1, 3]
        if (not preserve_v13_focus) and self._has_budget(deadline, evaluator, min_ms=16.0):
            allowed_any.append(2)
        op_idx = v14_pol.select_anytime_operator(g_feats, search_state, allowed_ops=tuple(sorted(allowed_any)))
        any_trace: list[str] = [ANYTIME_OPS[op_idx]]

        do_continue = (
            (not skip_calm_keep)
            and budget_ms >= 35.0
            and ((op_idx != 0) or has_disturbed or preserve_v13_focus)
            and self._has_budget(deadline, evaluator, min_ms=6.0)
        )
        run_heavy_v12 = (op_idx in (2, 3)) or has_disturbed or (not base_eval.valid)
        best_tw: dict[tuple[str, str], int] = dict(tw_seed)
        cutoff_step = int(snapshot.tick) + 45 if tw_seed else 0
        guides: dict[str, list[tuple[str, str]]] = {}

        if do_continue:
            if preserve_v13_focus:
                if rel_tids and (op_idx != 0 or has_disturbed) and _cvcr_now_s() < deadline:
                    best_assign, best_eval, _ = self._run_1opt_and_swap_ranked(
                        snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                        op_bay_overrides=best_bays, op_crew_overrides=best_crews,
                        allow_swaps=(budget_ms >= 55.0 and op_idx != 0), surr=surr, hys_margin=hys_v13,
                    )
                best_eval, _, pol_tw, pol_cut, guides = self._polish_v9_traffic_and_guides(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, aggressive_ltm=False, min_ltm_gain=2.5, min_wait_ltm=12, ci_candidates=6,
                    cutoff_horizon=50, emit_guides=True, preserve_orders=True,
                )
                for ek, wv in pol_tw.items():
                    best_tw[ek] = max(best_tw.get(ek, 0), int(wv))
                if pol_cut > 0:
                    cutoff_step = pol_cut
                elif best_tw:
                    cutoff_step = int(snapshot.tick) + 50
            else:
                if _cvcr_now_s() < deadline and (run_heavy_v12 or op_idx == 1):
                    bay_dl = min(deadline, _cvcr_now_s() + 0.012)
                    best_bays, best_crews, cur_prio, best_eval, _ = self._divert_prod_extended(
                        snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, bay_dl
                    )
                    surr.op_prio = dict(cur_prio)

                if run_heavy_v12:
                    rel_raw = self._prioritize_disturbed(snapshot, clf, self._select_ordered_release(snapshot, clf, best_eval))
                    rel_lrho, frozen_lrho = self._filter_lrho_releasable(snapshot, clf, best_eval, rel_raw, best_bays, best_crews)
                    d_sel = self.destroy_selector if self.destroy_selector is not None else load_hetero_destroy_selector()
                    radius = "large_stochastic" if budget_ms >= 90.0 else "medium"
                    rng_d = random.Random(self.seed + snapshot.tick * 131 + 17)
                    d_tids, frozen_het, _scores = d_sel.select_destroy_tasks(
                        self.network, snapshot, clf, best_eval, best_bays, best_crews,
                        radius=radius, bay_detour_delta=b_delta, rng=rng_d,
                    )
                    d_set = set(d_tids)
                    merged_rel = list(dict.fromkeys(list(chosen.destroy_tids) + [t for t in rel_lrho if t in d_set]))
                    if len(merged_rel) < 2:
                        merged_rel = list(dict.fromkeys(list(chosen.destroy_tids) + (d_tids if len(d_tids) >= 2 else rel_lrho)))
                    rel_tids = merged_rel or rel_tids
                    frozen_tids = set(frozen_lrho) if frozen_lrho else set(frozen_het)
                else:
                    frozen_tids = set()

                if rel_tids and _cvcr_now_s() < deadline:
                    best_assign, best_eval, _ = self._run_1opt_and_swap_ranked(
                        snapshot, clf, rel_tids, evaluator, best_assign, best_eval, healthy_robots, cur_prio, deadline,
                        op_bay_overrides=best_bays, op_crew_overrides=best_crews,
                        allow_swaps=(budget_ms >= 35.0), surr=surr, hys_margin=hys_v13,
                    )

                if len(healthy_robots) >= 2 and run_heavy_v12 and _cvcr_now_s() + 0.003 < deadline:
                    ci_dl = min(deadline, _cvcr_now_s() + max(0.008, (float(budget_ms) * 0.14) / 1000.0))
                    _ord1, best_eval, _ = self._search_intra_agent_ci(
                        snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, ci_dl,
                        surr=surr, max_candidates=6,
                    )

                if (
                    len(healthy_robots) >= 2
                    and budget_ms >= 85.0
                    and (op_idx == 2 or run_heavy_v12)
                    and self._has_budget(deadline, evaluator, min_ms=26.0)
                    and self._has_real_resource_overlap(snapshot, best_eval, best_bays, best_crews)
                ):
                    rem_s = deadline - _cvcr_now_s()
                    cp_dl = (deadline - 0.016) if rem_s > 0.030 else deadline
                    best_assign, best_bays, best_crews, cur_prio, _cp_ord, best_eval, _ = self._search_local_cpsat_window(
                        snapshot, clf, evaluator, best_assign, best_bays, best_crews, best_eval, healthy_robots, cur_prio,
                        cp_dl, surr, allow_order_only=True, min_improve=8.0, solve_time_cap_s=0.014,
                        disable_presolve=True, preserve_nonfocal_order=True,
                    )

                if len(healthy_robots) >= 2 and run_heavy_v12 and budget_ms >= 75.0 and _cvcr_now_s() < deadline:
                    sa_dl = max(_cvcr_now_s(), deadline - 0.006)
                    best_assign, best_eval, _ = self._search_v8_balance_pc(
                        snapshot, clf, evaluator, best_assign, best_eval, healthy_robots, cur_prio, sa_dl,
                        best_bays, best_crews, surr, hys_margin=hys_v13, preserve_orders=True, lrho_frozen=frozen_tids or None,
                    )

                best_eval, _, pol_tw, pol_cut, guides = self._polish_v9_traffic_and_guides(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, aggressive_ltm=False, min_ltm_gain=2.5, min_wait_ltm=12, ci_candidates=8,
                    cutoff_horizon=50, emit_guides=True, preserve_orders=True,
                )
                for ek, wv in pol_tw.items():
                    best_tw[ek] = max(best_tw.get(ek, 0), int(wv))
                if pol_cut > 0:
                    cutoff_step = pol_cut
                elif best_tw:
                    cutoff_step = int(snapshot.tick) + 50
        else:
            if best_eval.valid and best_eval.robot_paths:
                best_eval, _, pol_tw, pol_cut, guides = self._polish_v9_traffic_and_guides(
                    snapshot, evaluator, best_assign, best_bays, best_crews, best_eval, cur_prio, deadline,
                    surr=surr, aggressive_ltm=False, min_ltm_gain=2.5, min_wait_ltm=12, ci_candidates=4,
                    cutoff_horizon=45, emit_guides=True, preserve_orders=True,
                )
                for ek, wv in pol_tw.items():
                    best_tw[ek] = max(best_tw.get(ek, 0), int(wv))
                if pol_cut > 0:
                    cutoff_step = pol_cut

        if not guides and best_eval.valid and best_eval.robot_paths:
            for rid, edges in CVCREvaluator.extract_guide_corridors(best_eval.robot_paths, snapshot.tick, horizon_steps=40).items():
                ve = [(u, v) for (u, v) in edges if self.network.has_edge(u, v)]
                if ve:
                    guides[rid] = ve

        self._last_kitting_decision = {
            "policy_version": "V14_JOINT_RL",
            "cand_id": chosen.cand_id,
            "source_kind": chosen.source_kind,
            "selected_cand_id": chosen.cand_id,
            "selected_source": chosen.source_kind,
            "selected_target_op": chosen.target_op_id,
            "op_prot_top": op_prot_top,
            "num_candidates": len(cands),
            "cont_prob": round(float(cont_prob), 4),
            "det_score": round(float(chosen.det_score), 4),
            "rl_score": round(float(scores[best_i]), 4),
            "anytime_ops_trace": any_trace,
            "anytime_ops_used": len(any_trace),
            "disturbed_count": len(clf.disturbed_robots),
        }
        eval_count = max(1, len(evaluator.eval_lats_ms) - pre_lats)
        return (
            best_assign,
            best_bays,
            best_crews,
            best_eval,
            eval_count,
            rel_tids,
            None,
            best_tw,
            cutoff_step,
            guides,
            cur_prio,
        )

    def _search_v15_execution_aware(
        self,
        snapshot: NativeSnapshot,
        clf: Any,
        evaluator: CVCREvaluator,
        base_assign: dict[str, str],
        best_eval: EvalResult,
        healthy_robots: list[str],
        op_prio: dict[str, int],
        deadline: float,
        budget_ms: float = 50.0,
    ) -> tuple[
        dict[str, str],
        dict[str, str],
        dict[str, int],
        EvalResult,
        int,
        list[str],
        Optional[list[str]],
        dict[tuple[str, str], int],
        int,
        dict[str, list[tuple[str, str]]],
        dict[str, int],
    ]:
        """Run unified V15 Execution-Aware Search (Detour-Regret + BALANS + V10 CI/Bal + Selective Cliff)."""
        from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
        from amr_sched.cvcr.v15_ea_joint import (
            V15ExecutionAwarePolicy,
            generate_v15_unified_candidates,
            load_v15_ea_policy,
        )
        cand_gen = generate_v15_unified_candidates
        if self.method_id.startswith("V16_"):
            from amr_sched.cvcr.v16_frontier import generate_v16_unified_candidates
            cand_gen = generate_v16_unified_candidates

        best_assign = dict(base_assign)
        best_bays = {op.op_id: op.bay_id for op in snapshot.operations}
        best_crews = {op.op_id: op.crew_id for op in snapshot.operations}
        base_eval = best_eval
        cur_prio = dict(op_prio)
        self._last_v15_commit_seq_horizon = 0
        self._last_v15_order_seq: dict[str, list[str]] = {}

        if len(healthy_robots) < 1:
            return best_assign, best_bays, best_crews, best_eval, 0, [], None, {}, 0, {}, cur_prio

        dd_sig = compute_detour_delta_signals(self.network, snapshot)
        b_delta = dict(self.bay_detour_delta) if self.bay_detour_delta is not None else dict(dd_sig["bay_detour_delta"])
        tw_seed: dict[tuple[str, str], int] = {}
        fb_snap = getattr(snapshot, "execution_feedback", None)
        if fb_snap is not None and not fb_snap.is_empty():
            fb_blk = fb_snap.blocked_intervals(t_commit=int(snapshot.tick), default_horizon=12)
            if fb_blk:
                tw_seed = dict(fb_snap.corridor_q3_penalties())

        if isinstance(self.v15_policy, V15ExecutionAwarePolicy):
            v15_pol = self.v15_policy
        elif self.method_id.startswith(("V16_", "V17_")):
            from amr_sched.cvcr.v16_frontier import load_v16_frontier_policy
            v15_pol = load_v16_frontier_policy()
        else:
            v15_pol = load_v15_ea_policy()

        pre_lats = len(evaluator.eval_lats_ms)
        surr = _LazyKittingSurrogate(
            self.network,
            snapshot,
            cur_prio,
            self.alpha_makespan,
            self.lambda_change,
            bay_detour_delta=b_delta,
        )

        v17_meta: Optional[dict[str, Any]] = None
        if self.method_id.startswith("V17_"):
            from amr_sched.cvcr.v17_window_protection import generate_v17_dual_op_candidates

            cands, g_feats, v17_meta = generate_v17_dual_op_candidates(
                self,
                self.network,
                snapshot,
                clf,
                evaluator,
                surr,
                best_assign,
                best_bays,
                best_crews,
                base_eval,
                cur_prio,
                healthy_robots,
                b_delta,
                tw_seed,
                v15_pol,
                deadline,
                budget_ms,
                _cvcr_now_s,
            )
        else:
            cands, g_feats = cand_gen(
                self,
                self.network,
                snapshot,
                clf,
                evaluator,
                surr,
                best_assign,
                best_bays,
                best_crews,
                base_eval,
                cur_prio,
                healthy_robots,
                b_delta,
                tw_seed,
                v15_pol,
                deadline,
                budget_ms,
                _cvcr_now_s,
            )
        if not cands:
            return best_assign, best_bays, best_crews, best_eval, 1, [], None, tw_seed, (int(snapshot.tick) + 45 if tw_seed else 0), {}, cur_prio

        from amr_sched.cvcr.v15_ea_joint import compute_action_diff_digest

        scores = v15_pol.score_candidates(g_feats, cands)
        req_i = int(np.argmax(scores))
        req_cand = cands[req_i]
        best_i = req_i
        chosen = req_cand
        fallback_reason = "none"

        valid_cands = [c for c in cands if c.eval_res.valid]
        det_best = max(valid_cands, key=lambda c: c.det_score) if valid_cands else cands[0]
        allow_override = bool(getattr(v15_pol, "allow_probe_override", False))
        if (
            not allow_override
            and chosen is not det_best
            and det_best.eval_res.valid
            and chosen.eval_res.valid
            and not (chosen.cliff_saved and not det_best.cliff_saved)
            and chosen.eval_res.objective > det_best.eval_res.objective + 12.0
        ):
            chosen = det_best
            best_i = cands.index(det_best)
            fallback_reason = "det_guard_obj_gt_12"

        if base_eval.valid and chosen.eval_res.valid:
            reward_norm = (float(base_eval.objective) - float(chosen.eval_res.objective)) / 80.0
            v15_pol.observe_outcome(chosen.source_kind, chosen.radius_used, reward_norm)

        diff_digest = compute_action_diff_digest(
            chosen,
            best_assign,
            best_bays,
            best_crews,
            cur_prio,
            live_repairs=getattr(chosen, "live_repairs", None),
        )
        self._last_v15_commit_seq_horizon = int(chosen.commit_seq_horizon)
        self._last_v15_order_seq = dict(chosen.eval_res.robot_order_seq) if chosen.eval_res.valid else {}
        self._last_v17_live_repairs = dict(chosen.live_repairs) if getattr(chosen, "live_repairs", None) else None
        self._last_v17_bottleneck_locks = (
            {aid: list(sp) for aid, sp in chosen.bottleneck_locks.items()}
            if getattr(chosen, "bottleneck_locks", None)
            else None
        )
        self._last_kitting_decision = {
            "policy_version": str(self.method_id),
            "cand_id": chosen.cand_id,
            "requested_cand_id": req_cand.cand_id,
            "selected_cand_id": chosen.cand_id,
            "fallback_reason": fallback_reason,
            "allow_probe_override": allow_override,
            "action_diff_digest": diff_digest,
            "source_kind": chosen.source_kind,
            "commit_seq_horizon": int(chosen.commit_seq_horizon),
            "cliff_saved": bool(chosen.cliff_saved),
            "num_candidates": len(cands),
            "det_score": round(float(chosen.det_score), 4),
            "rl_score": round(float(scores[best_i]), 4),
            "v17_meta": v17_meta,
        }
        eval_count = max(1, len(evaluator.eval_lats_ms) - pre_lats)
        return (
            dict(chosen.assignment),
            dict(chosen.bay_overrides),
            dict(chosen.crew_overrides),
            chosen.eval_res,
            eval_count,
            list(chosen.destroy_tids),
            None,
            dict(chosen.traffic_weights),
            int(chosen.traffic_cutoff),
            dict(chosen.guide_routes),
            dict(chosen.op_priority),
        )


def compute_hybrid_criticality(
    snapshot: NativeSnapshot,
    eval_res: EvalResult,
    bays: dict[str, str],
    crews: dict[str, int],
    alpha_makespan: float = 3.0,
) -> dict[str, Any]:
    """Jia et al. (2025) Alg. 3 backward hybrid critical-path & MIT w_prec kitting-slack analyzer."""
    op_map = {op.op_id: op for op in snapshot.operations}
    op_s = eval_res.op_start
    op_e = eval_res.op_end
    unloads = eval_res.order_unload_ticks
    if not op_map or not op_e:
        return {
            "op_weight": {},
            "task_weight": {},
            "kitting_slack": {},
            "blocked_by_bay": {},
            "blocked_by_crew": {},
            "blocked_by_tool": {},
            "tardy_ops": [],
        }

    makespan = max(op_e.values())
    op_weight: dict[str, float] = {}
    tardy_ops: list[str] = []
    for op in snapshot.operations:
        en = op_e.get(op.op_id, snapshot.tick)
        w = 0.0
        if en > op.due_step:
            w += 12.0 if op.is_urgent else 4.0
            tardy_ops.append(op.op_id)
        elif op.due_step - en <= 4:
            w += 1.5 if op.is_urgent else 0.5
        if en == makespan:
            w += float(alpha_makespan)
        op_weight[op.op_id] = w

    by_bay: dict[str, list[str]] = {}
    by_crew: dict[int, list[str]] = {}
    by_tool: dict[int, list[str]] = {}
    for op in snapshot.operations:
        oid = op.op_id
        by_bay.setdefault(bays.get(oid, op.bay_id), []).append(oid)
        by_crew.setdefault(crews.get(oid, op.crew_id), []).append(oid)
        by_tool.setdefault(op.tool_id, []).append(oid)

    blocked_by_bay: dict[str, str] = {}
    blocked_by_crew: dict[str, str] = {}
    blocked_by_tool: dict[str, str] = {}
    for grp, target in ((by_bay, blocked_by_bay), (by_crew, blocked_by_crew), (by_tool, blocked_by_tool)):
        for oids in grp.values():
            oids.sort(key=lambda o: (op_s.get(o, 0), op_e.get(o, 0), o))
            for k in range(1, len(oids)):
                prev_o, cur_o = oids[k - 1], oids[k]
                if op_e.get(prev_o, 0) >= op_s.get(cur_o, 0) - 1:
                    target[cur_o] = prev_o

    for oid in sorted(op_map.keys(), key=lambda o: (-op_e.get(o, 0), -op_s.get(o, 0), o)):
        w = op_weight.get(oid, 0.0)
        if w <= 0.0:
            continue
        st = op_s.get(oid, snapshot.tick)
        op = op_map[oid]
        for pid in op_predecessors(op):
            if pid in op_map and op_e.get(pid, 0) >= st - 1:
                op_weight[pid] = op_weight.get(pid, 0.0) + w
        for blk_map in (blocked_by_bay, blocked_by_crew, blocked_by_tool):
            prev_o = blk_map.get(oid)
            if prev_o is not None and not op_map[prev_o].started:
                op_weight[prev_o] = op_weight.get(prev_o, 0.0) + 0.85 * w

    task_weight: dict[str, float] = {}
    kitting_slack: dict[str, int] = {}
    for op in snapshot.operations:
        if op.started:
            continue
        st = op_s.get(op.op_id, snapshot.tick)
        w = op_weight.get(op.op_id, 0.0)
        for tid in op.kit_task_ids:
            u = unloads.get(tid, 9999)
            slack = st - u
            kitting_slack[tid] = slack
            if slack <= 2:
                task_weight[tid] = task_weight.get(tid, 0.0) + max(0.5, w)
            else:
                task_weight.setdefault(tid, 0.0)

    return {
        "op_weight": op_weight,
        "task_weight": task_weight,
        "kitting_slack": kitting_slack,
        "blocked_by_bay": blocked_by_bay,
        "blocked_by_crew": blocked_by_crew,
        "blocked_by_tool": blocked_by_tool,
        "tardy_ops": tardy_ops,
    }


class _LazyKittingSurrogate:
    """Lazy per-source shortest-path kitting proxy with zero cold-start 20ms penalty."""

    _SRC_CACHE: dict[tuple[int, float], dict[str, dict[str, float]]] = {}

    @classmethod
    def _get_src_map(cls, network: Network, max_speed: float) -> dict[str, dict[str, float]]:
        key = (id(network), round(float(max_speed), 4))
        src_map = cls._SRC_CACHE.get(key)
        if src_map is None:
            if len(cls._SRC_CACHE) >= 16:
                cls._SRC_CACHE.clear()
            src_map = {}
            cls._SRC_CACHE[key] = src_map
        return src_map

    def __init__(
        self,
        network: Network,
        snapshot: NativeSnapshot,
        op_prio: dict[str, int],
        alpha_makespan: float = 3.0,
        lambda_change: float = 15.0,
        bay_detour_delta: Optional[dict[str, float]] = None,
    ) -> None:
        self.network = network
        self.snapshot = snapshot
        self.op_prio = op_prio
        self.alpha_makespan = alpha_makespan
        self.lambda_change = lambda_change
        self.bay_detour_delta = dict(bay_detour_delta) if bay_detour_delta else {}
        self.max_speed = max(0.5, max((e.speed_limit for e in network.edges.values()), default=3.0))
        self._src_map = self._get_src_map(network, self.max_speed)
        self.tasks = list(snapshot.tasks)
        self.tasks_by_id = {t.task_id: t for t in self.tasks}
        self.robots_by_id = {r.robot_id: r for r in snapshot.robots}
        self.ops = list(snapshot.operations)
        self.snap_assign = {t.task_id: t.assigned_agv for t in self.tasks if t.assigned_agv is not None}

    def bay_delay(self, bay_node: str) -> float:
        """Return causal Detour-Delta congestion estimate for `bay_node` (0.0 when unset)."""
        return float(self.bay_detour_delta.get(bay_node, 0.0)) if self.bay_detour_delta else 0.0

    def tt(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        d = self._src_map.get(a)
        if d is None:
            d = self.network.shortest_travel_times(a, self.max_speed)
            self._src_map[a] = d
        return float(d.get(b, 200.0))

    def regret_k_assignment(
        self,
        healthy_robots: list[str],
        fixed_assign: dict[str, str],
        tasks_to_insert: list[str],
        op_bays: Optional[dict[str, str]] = None,
        rng: Optional[random.Random] = None,
        noise: float = 0.0,
    ) -> dict[str, str]:
        assign = dict(fixed_assign)
        if not healthy_robots:
            return assign
        op_map = {op.op_id: op for op in self.ops}
        rob_node: dict[str, str] = {}
        rob_free: dict[str, float] = {}
        rob_loads: dict[str, int] = {}
        for rid in healthy_robots:
            rob = self.robots_by_id[rid]
            rob_node[rid] = rob.cur_node
            rob_free[rid] = float(max(self.snapshot.tick, rob.faulted_until))
            rob_loads[rid] = 0

        by_r: dict[str, list[Any]] = {r: [] for r in healthy_robots}
        for tid, rid in assign.items():
            if rid in by_r and tid in self.tasks_by_id:
                t = self.tasks_by_id[tid]
                if t.status.lower() != "completed":
                    by_r[rid].append(t)
        for rid, tlist in by_r.items():
            tlist.sort(
                key=lambda x: (
                    0 if x.status.lower() in ("picked", "loading", "to_dropoff", "unloading") else 1,
                    self.op_prio.get(x.kit_op_id or "", 0),
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
                    arr = rob_free[rid] + self.tt(rob_node[rid], drop) + 2.0
                else:
                    tp = max(rob_free[rid] + self.tt(rob_node[rid], t.pickup), float(t.release_step)) + 2.0
                    arr = tp + self.tt(t.pickup, drop) + 2.0
                rob_free[rid] = arr
                rob_node[rid] = drop
                rob_loads[rid] += 1

        remaining = [
            tid for tid in tasks_to_insert if tid in self.tasks_by_id and self.tasks_by_id[tid].status.lower() != "completed"
        ]
        while remaining:
            best_tid: Optional[str] = None
            best_rid: Optional[str] = None
            best_key = (-1e18, -1e18)
            for tid in remaining:
                t = self.tasks_by_id[tid]
                drop = op_bays.get(t.kit_op_id, t.dropoff) if (op_bays and t.kit_op_id) else t.dropoff
                due_eff = float(t.due_step if t.due_step is not None else 999)
                if t.kit_op_id and t.kit_op_id in op_map:
                    op = op_map[t.kit_op_id]
                    due_eff = min(due_eff, float(op.due_step - op.proc_steps)) - (15.0 if op.is_urgent else 0.0)
                costs: list[tuple[float, str]] = []
                for rid in healthy_robots:
                    tp = max(rob_free[rid] + self.tt(rob_node[rid], t.pickup), float(t.release_step)) + 2.0
                    comp = tp + self.tt(t.pickup, drop) + self.bay_delay(drop) + 2.0
                    tard = 3.0 * max(0.0, comp - due_eff)
                    chg = self.lambda_change if (tid in self.snap_assign and self.snap_assign[tid] != rid and t.status.lower() == "assigned") else 0.0
                    c = comp + tard + chg + 1.5 * rob_loads[rid]
                    if noise > 0.0 and rng is not None:
                        c *= 1.0 + rng.uniform(-noise, noise)
                    costs.append((c, rid))
                costs.sort(key=lambda x: x[0])
                c1, r1 = costs[0]
                c2 = costs[1][0] if len(costs) >= 2 else c1 + 50.0
                urg_boost = 40.0 if (t.is_urgent or t.assigned_agv not in healthy_robots) else 0.0
                key = ((c2 - c1) + urg_boost - 0.15 * due_eff, -c1)
                if best_tid is None or key > best_key:
                    best_key = key
                    best_tid = tid
                    best_rid = r1
            assert best_tid is not None and best_rid is not None
            remaining.remove(best_tid)
            assign[best_tid] = best_rid
            t = self.tasks_by_id[best_tid]
            drop = op_bays.get(t.kit_op_id, t.dropoff) if (op_bays and t.kit_op_id) else t.dropoff
            tp = max(rob_free[best_rid] + self.tt(rob_node[best_rid], t.pickup), float(t.release_step)) + 2.0
            rob_free[best_rid] = tp + self.tt(t.pickup, drop) + 2.0
            rob_node[best_rid] = drop
            rob_loads[best_rid] += 1
        return assign

    def fast_score(
        self,
        assign: dict[str, str],
        op_bays: Optional[dict[str, str]] = None,
        op_crews: Optional[dict[str, int]] = None,
    ) -> float:
        unloads: dict[str, int] = {}
        by_r: dict[str, list[Any]] = {}
        for t in self.tasks:
            if t.status.lower() == "completed":
                unloads[t.task_id] = int(t.complete_step or self.snapshot.tick)
            else:
                rid = assign.get(t.task_id)
                if rid:
                    by_r.setdefault(rid, []).append(t)

        for rid, tlist in by_r.items():
            rob = self.robots_by_id.get(rid)
            if rob is None:
                continue
            cur_n = rob.cur_node
            cur_t = float(max(self.snapshot.tick, rob.faulted_until))
            tlist.sort(
                key=lambda x: (
                    0 if x.status.lower() in ("picked", "loading", "to_dropoff", "unloading") else 1,
                    self.op_prio.get(x.kit_op_id or "", 0),
                    x.release_step,
                    x.task_id,
                )
            )
            cong = 1.0 + 0.04 * max(0, len(tlist) - 2)
            for t in tlist:
                drop = (
                    op_bays.get(t.kit_op_id, t.dropoff)
                    if (op_bays and t.kit_op_id and t.status.lower() in ("pending", "assigned"))
                    else t.dropoff
                )
                if t.status.lower() in ("picked", "loading", "to_dropoff", "unloading"):
                    arr = cur_t + cong * self.tt(cur_n, drop) + self.bay_delay(drop) + 2.0
                else:
                    tp = max(cur_t + cong * self.tt(cur_n, t.pickup), float(t.release_step)) + 2.0
                    arr = tp + cong * self.tt(t.pickup, drop) + self.bay_delay(drop) + 2.0
                unloads[t.task_id] = int(round(arr + 0.499))
                cur_t = arr
                cur_n = drop

        bay_free: dict[str, int] = {}
        crew_free: dict[int, int] = {}
        tool_free: dict[int, int] = {}
        op_end: dict[str, int] = {}
        ops_sorted = sorted(
            self.ops,
            key=lambda op: (0 if op.started else 1, self.op_prio.get(op.op_id, 0), op.due_step),
        )
        resolved: set[str] = set()
        for _ in range(len(ops_sorted) + 2):
            for op in ops_sorted:
                if op.op_id in resolved:
                    continue
                preds = op_predecessors(op)
                if any(pid not in resolved for pid in preds):
                    continue
                bay = op_bays.get(op.op_id, op.bay_id) if op_bays else op.bay_id
                crew = op_crews.get(op.op_id, op.crew_id) if op_crews else op.crew_id
                if op.started and op.start_step is not None and op.end_step is not None:
                    op_end[op.op_id] = op.end_step
                    bay_free[bay] = max(bay_free.get(bay, 0), op.end_step)
                    crew_free[crew] = max(crew_free.get(crew, 0), op.end_step)
                    tool_free[op.tool_id] = max(tool_free.get(op.tool_id, 0), op.end_step)
                else:
                    mat_t = max((unloads.get(tid, 999) for tid in op.kit_task_ids), default=self.snapshot.tick)
                    pred_t = max((op_end.get(pid, 0) for pid in preds), default=0)
                    r_ready = max(
                        self.snapshot.tick,
                        pred_t,
                        bay_free.get(bay, self.snapshot.tick),
                        crew_free.get(crew, self.snapshot.tick),
                        tool_free.get(op.tool_id, self.snapshot.tick),
                    )
                    en = max(r_ready, mat_t) + op.proc_steps
                    op_end[op.op_id] = en
                    bay_free[bay] = en
                    crew_free[crew] = en
                    tool_free[op.tool_id] = en
                resolved.add(op.op_id)

        mk = max(op_end.values()) if op_end else self.snapshot.tick
        tard = sum(
            (12.0 if op.is_urgent else 4.0) * max(0, op_end.get(op.op_id, 999) - op.due_step)
            for op in self.ops
        )
        tot_deliv = sum(unloads.get(t.task_id, 999) for t in self.tasks if t.status.lower() != "completed")
        chg = sum(
            1
            for tid, rid in assign.items()
            if tid in self.snap_assign and self.snap_assign[tid] != rid and self.tasks_by_id[tid].status.lower() == "assigned"
        )
        return float(tard + self.alpha_makespan * mk + 0.5 * tot_deliv + self.lambda_change * chg)
