"""Literature Mechanism Adapters under Unified Causal Snapshot & Transaction Entry.

Implements four complementary literature-grounded mechanisms evaluated fairly inside
`AMRReschedulingEnv.external_proposer`:
  1. `WangESWA24RepairProposer` (Wang et al., ESWA 2024):
     Low-latency localized fault-repair + neighbor exchange baseline.
  2. `ShiAEI25BilevelProposer` (Shi et al., AEI 2025):
     Bi-level production-goal urgency selector -> executable transport dispatch.
  3. `BalansBanditPhan24Proposer` (Huang AAAI 2022 + Phan AAAI 2024):
     Discounted Thompson/UCB1 multi-armed bandit over `(operator, radius)` + Keep/Release.
  4. `ReijnenDRALNS24Proposer` (Reijnen et al., ICAPS 2024 DR-ALNS):
     Sequential batch-continuation controller trading extra search batches against compute latency.
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional

from amr_sched.baselines.top_venue_baselines import _build_external_proposal
from amr_sched.cvcr.clock import is_virtual_clock_active, now_s as _cvcr_now_s, record_work_ms as _cvcr_record_work_ms
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.policy import CandidateProposal, LGCVCRPolicy, _LazyKittingSurrogate
from amr_sched.cvcr.snapshot import NativeSnapshot, op_predecessors
from amr_sched.cvcr.v12_proactive_guard import compute_kitting_marginal_start_value
from amr_sched.network import Network


def _op_succ_counts(snapshot: NativeSnapshot) -> dict[str, int]:
    """Count direct successor operations per operation ID."""
    cnt: dict[str, int] = {}
    for op in snapshot.operations:
        for pid in op_predecessors(op):
            cnt[pid] = cnt.get(pid, 0) + 1
    return cnt


class WangESWA24RepairProposer:
    """Wang et al. (ESWA 2024): Low-latency localized repair baseline prioritizing immediate rescue."""

    def __init__(self, *, lambda_change: float = 15.0, alpha_makespan: float = 3.0, use_rust: bool = True) -> None:
        """Initialize localized fast-repair proposer."""
        self.lambda_change = float(lambda_change)
        self.alpha_makespan = float(alpha_makespan)
        self.use_rust = bool(use_rust)

    def __call__(self, snapshot: NativeSnapshot, network: Network, budget_ms: float = 20.0) -> CandidateProposal:
        """Produce fast localized rescue & immediate-neighbor swap proposal within tight latency cap."""
        t0 = _cvcr_now_s()
        cap_ms = min(float(budget_ms), 22.0)
        deadline = t0 + max(0.004, cap_ms * 0.78 / 1000.0)
        clf = classify_commitments(snapshot)
        lg = LGCVCRPolicy("V12_RESCUE_DETOUR_FAST", network, lambda_change=self.lambda_change, alpha_makespan=self.alpha_makespan, use_rust=self.use_rust)
        healthy = lg._healthy_robots(snapshot, clf.disturbed_robots)
        op_prio = {op.op_id: max(snapshot.tick, op.due_step - op.proc_steps) for op in snapshot.operations}
        base_assign = lg._base_assignment(snapshot, healthy, op_prio)
        bays = {op.op_id: op.bay_id for op in snapshot.operations}
        crews = {op.op_id: op.crew_id for op in snapshot.operations}

        ev = CVCREvaluator(network, snapshot, alpha_makespan=self.alpha_makespan, lambda_change=self.lambda_change, use_rust=self.use_rust, mode="B2")
        best_eval = ev.evaluate(base_assign, op_priority=op_prio)
        best_assign = dict(base_assign)
        if healthy and len(healthy) >= 2:
            ev.set_incumbent(best_assign, best_eval, op_priority=op_prio)
            # Immediate disturbed + urgent subset only (localized repair entry)
            imp_tids = [
                t.task_id
                for t in snapshot.tasks
                if t.task_id in clf.releasable_tasks
                and (t.assigned_agv in clf.disturbed_robots or t.is_urgent)
            ]
            if not imp_tids:
                imp_tids = sorted(clf.releasable_tasks)[:4]
            best_assign, best_eval, _ = lg._run_1opt_and_swap_ranked(
                snapshot, clf, imp_tids[:6], ev, best_assign, best_eval, healthy, op_prio, deadline, allow_swaps=True, hys_margin=0.0
            )
        if is_virtual_clock_active():
            _cvcr_record_work_ms(1.5)
        return _build_external_proposal(
            policy_id="WANG_ESWA24_FAST_REPAIR",
            snapshot=snapshot,
            clf=clf,
            best_assign=best_assign,
            best_bays=bays,
            best_crews=crews,
            best_eval=best_eval,
            op_prio=op_prio,
            eval_cnt=ev.calls,
            t0=t0,
            budget_ms=cap_ms,
        )


class ShiAEI25BilevelProposer:
    """Shi et al. (AEI 2025): Bi-level production-goal selection -> executable logistics dispatch."""

    def __init__(self, *, lambda_change: float = 15.0, alpha_makespan: float = 3.0, use_rust: bool = True) -> None:
        """Initialize bi-level production-driven proposer."""
        self.lambda_change = float(lambda_change)
        self.alpha_makespan = float(alpha_makespan)
        self.use_rust = bool(use_rust)

    def __call__(self, snapshot: NativeSnapshot, network: Network, budget_ms: float = 50.0) -> CandidateProposal:
        """Select production bay/crew/priority upper-level goals then solve lower-level routing."""
        t0 = _cvcr_now_s()
        deadline = t0 + max(0.008, float(budget_ms) * 0.80 / 1000.0)
        clf = classify_commitments(snapshot)
        lg = LGCVCRPolicy("V10_E2_CRIT_PROD_CI", network, lambda_change=self.lambda_change, alpha_makespan=self.alpha_makespan, use_rust=self.use_rust)
        healthy = lg._healthy_robots(snapshot, clf.disturbed_robots)
        succ_cnt = _op_succ_counts(snapshot)
        km = compute_kitting_marginal_start_value(snapshot)

        # Upper level: production-goal priority balancing critical windows + DAG fanout + readiness
        op_prio: dict[str, int] = {}
        for op in snapshot.operations:
            k_min = min((km[kt][0] * 1000 + km[kt][4] for kt in op.kit_task_ids if kt in km), default=op.due_step)
            win_bias = -250 if getattr(op, "resource_windows", ()) else 0
            op_prio[op.op_id] = int(k_min - succ_cnt.get(op.op_id, 0) * 60 + win_bias)

        base_assign = lg._base_assignment(snapshot, healthy, op_prio)
        bays = {op.op_id: op.bay_id for op in snapshot.operations}
        crews = {op.op_id: op.crew_id for op in snapshot.operations}

        ev = CVCREvaluator(network, snapshot, alpha_makespan=self.alpha_makespan, lambda_change=self.lambda_change, use_rust=self.use_rust, mode="B2")
        best_eval = ev.evaluate(base_assign, op_priority=op_prio)
        if len(healthy) >= 2 and lg._has_budget(deadline, ev, min_ms=4.0):
            bays, crews, op_prio, best_eval, _ = lg._divert_prod_extended(
                snapshot, ev, base_assign, bays, crews, best_eval, op_prio, deadline
            )
            ev.set_incumbent(base_assign, best_eval, op_priority=op_prio, op_bay_overrides=bays)
            rel_tids = sorted(clf.releasable_tasks)[:8]
            if rel_tids and lg._has_budget(deadline, ev, min_ms=3.0):
                base_assign, best_eval, _ = lg._run_1opt_and_swap_ranked(
                    snapshot, clf, rel_tids, ev, base_assign, best_eval, healthy, op_prio, deadline,
                    op_bay_overrides=bays, op_crew_overrides=crews, allow_swaps=True, hys_margin=0.0,
                )
        return _build_external_proposal(
            policy_id="SHI_AEI25_BILEVEL",
            snapshot=snapshot,
            clf=clf,
            best_assign=base_assign,
            best_bays=bays,
            best_crews=crews,
            best_eval=best_eval,
            op_prio=op_prio,
            eval_cnt=ev.calls,
            t0=t0,
            budget_ms=budget_ms,
        )


class BalansBanditPhan24Proposer:
    """Huang AAAI 2022 + Phan AAAI 2024: Bandit ALNS over `(operator, radius)` + Keep/Release."""

    ARMS: tuple[tuple[str, int], ...] = (
        ("regret", 3),
        ("regret", 6),
        ("regret", 999),
        ("random", 4),
        ("random", 8),
        ("crit_op", 6),
    )

    def __init__(
        self,
        *,
        lambda_change: float = 15.0,
        alpha_makespan: float = 3.0,
        use_rust: bool = True,
        seed: int = 2024,
    ) -> None:
        """Initialize discounted UCB/Thompson arm statistics."""
        self.lambda_change = float(lambda_change)
        self.alpha_makespan = float(alpha_makespan)
        self.use_rust = bool(use_rust)
        self._rng = random.Random(int(seed))
        self._n: dict[tuple[str, int], float] = {a: 1.0 for a in self.ARMS}
        self._v: dict[tuple[str, int], float] = {a: 0.05 for a in self.ARMS}

    def __call__(self, snapshot: NativeSnapshot, network: Network, budget_ms: float = 50.0) -> CandidateProposal:
        """Select destroy operator and radius via discounted UCB1 and apply 1-opt/swap LNS."""
        t0 = _cvcr_now_s()
        deadline = t0 + max(0.008, float(budget_ms) * 0.80 / 1000.0)
        clf = classify_commitments(snapshot)
        lg = LGCVCRPolicy("V12_BALANCED_DESTROY_M", network, lambda_change=self.lambda_change, alpha_makespan=self.alpha_makespan, use_rust=self.use_rust)
        healthy = lg._healthy_robots(snapshot, clf.disturbed_robots)
        op_prio = {op.op_id: max(snapshot.tick, op.due_step - op.proc_steps) for op in snapshot.operations}
        best_assign = lg._base_assignment(snapshot, healthy, op_prio)
        bays = {op.op_id: op.bay_id for op in snapshot.operations}
        crews = {op.op_id: op.crew_id for op in snapshot.operations}

        ev = CVCREvaluator(network, snapshot, alpha_makespan=self.alpha_makespan, lambda_change=self.lambda_change, use_rust=self.use_rust, mode="B2")
        base_eval = ev.evaluate(best_assign, op_priority=op_prio)
        best_eval = base_eval

        rel = sorted(clf.releasable_tasks)
        if len(healthy) >= 2 and rel and lg._has_budget(deadline, ev, min_ms=3.5):
            tot_n = sum(self._n.values())
            arm = max(self.ARMS, key=lambda a: self._v[a] + 0.55 * math.sqrt(math.log(max(2.0, tot_n)) / self._n[a]))
            op_kind, rad = arm
            surr = _LazyKittingSurrogate(network, snapshot, op_prio, self.alpha_makespan, self.lambda_change)
            tmap = {t.task_id: t for t in snapshot.tasks}
            if op_kind == "crit_op":
                sub = sorted(rel, key=lambda tid: (int(tmap[tid].due_step or 999), tid))[: min(len(rel), rad)]
            elif op_kind == "random":
                shuf = list(rel)
                self._rng.shuffle(shuf)
                sub = shuf[: min(len(shuf), rad)]
            else:
                sub = rel[: min(len(rel), rad)]

            cand_a = surr.regret_k_assignment(healthy, {t: a for t, a in best_assign.items() if t not in set(sub)}, sub, op_bays=bays, rng=self._rng, noise=0.05)
            ev.set_incumbent(best_assign, best_eval, op_priority=op_prio)
            if cand_a != best_assign:
                ev_c = ev.evaluate(cand_a, op_priority=op_prio)
                if ev_c.valid and (not best_eval.valid or ev_c.objective + 0.5 < best_eval.objective):
                    best_assign, best_eval = cand_a, ev_c
                    ev.set_incumbent(best_assign, best_eval, op_priority=op_prio)
            if lg._has_budget(deadline, ev, min_ms=2.0):
                best_assign, best_eval, _ = lg._run_1opt_and_swap_ranked(
                    snapshot, clf, sub, ev, best_assign, best_eval, healthy, op_prio, deadline, allow_swaps=True, surr=surr, hys_margin=0.0
                )
            gain = max(-1.0, min(2.0, (float(base_eval.objective) - float(best_eval.objective)) / 80.0)) if (base_eval.valid and best_eval.valid) else 0.0
            self._n[arm] = 0.85 * self._n[arm] + 1.0
            self._v[arm] += (gain - self._v[arm]) / self._n[arm]

        return _build_external_proposal(
            policy_id="BALANS_PHAN24_BANDIT",
            snapshot=snapshot,
            clf=clf,
            best_assign=best_assign,
            best_bays=bays,
            best_crews=crews,
            best_eval=best_eval,
            op_prio=op_prio,
            eval_cnt=ev.calls,
            t0=t0,
            budget_ms=budget_ms,
        )


class ReijnenDRALNS24Proposer:
    """Reijnen et al. (ICAPS 2024 DR-ALNS): Adaptive sequential fixed-eval search-batch stopping."""

    def __init__(
        self,
        *,
        lambda_change: float = 15.0,
        alpha_makespan: float = 3.0,
        use_rust: bool = True,
        batch_evals: int = 5,
        max_batches: int = 3,
        stop_on_tight_slack: bool = True,
        min_gain_stop: float = 2.0,
        virtual_batch_ms: float = 10.0,
        seed: int = 2024,
    ) -> None:
        """Initialize adaptive batch-stopping ALNS controller."""
        self.lambda_change = float(lambda_change)
        self.alpha_makespan = float(alpha_makespan)
        self.use_rust = bool(use_rust)
        self.batch_evals = max(2, int(batch_evals))
        self.max_batches = max(0, int(max_batches))
        self.stop_on_tight_slack = bool(stop_on_tight_slack)
        self.min_gain_stop = float(min_gain_stop)
        self.virtual_batch_ms = float(virtual_batch_ms)
        self._rng = random.Random(int(seed))

    def __call__(self, snapshot: NativeSnapshot, network: Network, budget_ms: float = 80.0) -> CandidateProposal:
        """Run up to `max_batches` sequential destroy-repair batches, stopping early when marginal VoI drops."""
        t0 = _cvcr_now_s()
        deadline = t0 + max(0.008, float(budget_ms) * 0.80 / 1000.0)
        clf = classify_commitments(snapshot)
        lg = LGCVCRPolicy("V12_BALANCED_DESTROY_M", network, lambda_change=self.lambda_change, alpha_makespan=self.alpha_makespan, use_rust=self.use_rust)
        healthy = lg._healthy_robots(snapshot, clf.disturbed_robots)
        op_prio = {op.op_id: max(snapshot.tick, op.due_step - op.proc_steps) for op in snapshot.operations}
        best_assign = lg._base_assignment(snapshot, healthy, op_prio)
        bays = {op.op_id: op.bay_id for op in snapshot.operations}
        crews = {op.op_id: op.crew_id for op in snapshot.operations}

        ev = CVCREvaluator(network, snapshot, alpha_makespan=self.alpha_makespan, lambda_change=self.lambda_change, use_rust=self.use_rust, mode="B2")
        best_eval = ev.evaluate(best_assign, op_priority=op_prio)
        batches_run = 0
        rel = sorted(clf.releasable_tasks)
        # Check minimum urgency slack to guard compute latency against tight windows
        min_slack = min((op.due_step - op.proc_steps - snapshot.tick for op in snapshot.operations if not op.started), default=999)

        if len(healthy) >= 2 and rel and self.max_batches > 0:
            surr = _LazyKittingSurrogate(network, snapshot, op_prio, self.alpha_makespan, self.lambda_change)
            for _b in range(self.max_batches):
                if not lg._has_budget(deadline, ev, min_ms=4.0):
                    break
                if self.stop_on_tight_slack and _b >= 1 and min_slack <= 10:
                    # Compute-time vs window expiration guard: commit now before window closes
                    break
                batches_run += 1
                prev_obj = float(best_eval.objective) if best_eval.valid else 1e9
                ev.set_incumbent(best_assign, best_eval, op_priority=op_prio)
                sub = list(rel)
                self._rng.shuffle(sub)
                sub = sub[: min(len(sub), 6)]
                batch_dl = min(deadline, _cvcr_now_s() + 0.015)
                best_assign, best_eval, _ = lg._run_1opt_and_swap_ranked(
                    snapshot, clf, sub, ev, best_assign, best_eval, healthy, op_prio, batch_dl, allow_swaps=True, surr=surr, hys_margin=0.0
                )
                if is_virtual_clock_active() and self.virtual_batch_ms > 0.0:
                    _cvcr_record_work_ms(self.virtual_batch_ms)
                gain = prev_obj - (float(best_eval.objective) if best_eval.valid else prev_obj)
                if self.min_gain_stop >= 0.0 and gain < self.min_gain_stop:
                    # Diminishing marginal return -> stop searching early
                    break

        prop = _build_external_proposal(
            policy_id="REIJNEN_ICAPS24_DRALNS",
            snapshot=snapshot,
            clf=clf,
            best_assign=best_assign,
            best_bays=bays,
            best_crews=crews,
            best_eval=best_eval,
            op_prio=op_prio,
            eval_cnt=ev.calls,
            t0=t0,
            budget_ms=budget_ms,
        )
        prop.telemetry["dralns_batches_run"] = batches_run
        return prop
