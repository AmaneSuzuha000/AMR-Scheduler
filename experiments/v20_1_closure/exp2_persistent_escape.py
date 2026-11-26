"""EXP-2 (P0): Persistent Escape Search Memory M_t (Real-Time LaCAM Inspired).

Maintains state-compatible recovery memory across consecutive replanning/escalation ticks:
    M_t = { (state_sig_i, neighborhood_i, action_sig_i, outcome_i) }
Eliminates repeated dead-end recovery attempts on identical blocking configurations while
strictly validating commitment compatibility (never replaying stale pre-pickup plans and
never over-claiming MAPF completeness under irreversible production constraints).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.cvcr.snapshot import NativeSnapshot
from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals, plan_haven_safe_corridor_repair
from amr_sched.cvcr.v18_frontier_synthesis import compute_semicircle_kitting_bays
from amr_sched.network import Network
from amr_sched.rl.causal_repair_cone import (
    FastJAWKittingSurrogate,
    _compute_fresh_semicircle_bays,
    _regret2_insert_tasks,
    observable_op_closure,
)
from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
from experiments.v20_1_closure.pdg2_escape import (
    PDG2Proposer,
    _build_critical_bottleneck_locks,
    _plan_idle_blocker_evacuations,
)
from experiments.v20_audit.common import CandidateArtifact
from experiments.v20_audit.proposer_variants import (
    WindowAwareKittingSurrogate,
    _LazyKittingSurrogate,
    _default_task_order_from_prio,
    _healthy_robots_from_graph,
    derive_base_op_prio,
)


@dataclass
class RecoveryMemoryEntry:
    state_sig: str
    commitment_sig: str
    neighborhood_key: str
    action_sig: str
    tick: int
    outcome: str = "pending"  # pending | no_progress | progressed


@dataclass
class PersistentEscapeMemory:
    """State- and commitment-compatible search memory preventing duplicate failed escape cycles."""

    max_entries: int = 128
    entries: list[RecoveryMemoryEntry] = field(default_factory=list)
    duplicate_candidates_pruned: int = 0
    frontier_expansions: int = 0
    _last_entry: Optional[RecoveryMemoryEntry] = None
    _last_completed_counts: tuple[int, int] = (0, 0)

    @staticmethod
    def compute_signatures(snapshot: NativeSnapshot, pdg: dict[str, Any]) -> tuple[str, str]:
        """Compute blocking-state signature and hard-commitment compatibility signature."""
        dist = set(pdg.get("stalled_agvs") or ()) | set(pdg.get("blocked_healthy_agvs") or ()) | set(pdg.get("wait_cycle_agvs") or ())
        r_sig = sorted(
            (r.robot_id, r.cur_node, r.state, r.cur_task)
            for r in snapshot.robots
            if r.robot_id in dist or r.state.lower() in ("blocked", "faulted")
        )
        t_done = sum(1 for t in snapshot.tasks if t.status.lower() == "completed")
        o_done = sum(1 for op in snapshot.operations if op.end_step is not None)
        state_raw = json.dumps({"r": r_sig, "td": t_done, "od": o_done, "ops": pdg.get("top_risk_ops", [])[:3]}, separators=(",", ":")).encode()
        comm_raw = json.dumps(
            {
                "picked": sorted((t.task_id, t.assigned_agv) for t in snapshot.tasks if t.status.lower() == "picked"),
                "started_ops": sorted(op.op_id for op in snapshot.operations if op.started),
            },
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(state_raw).hexdigest()[:14], hashlib.sha256(comm_raw).hexdigest()[:14]

    def update_previous_outcome(self, snapshot: NativeSnapshot, pdg: dict[str, Any]) -> tuple[str, str]:
        s_sig, c_sig = self.compute_signatures(snapshot, pdg)
        c_counts = (
            sum(1 for t in snapshot.tasks if t.status.lower() == "completed"),
            sum(1 for op in snapshot.operations if op.end_step is not None),
        )
        if self._last_entry is not None and self._last_entry.outcome == "pending":
            if c_counts > self._last_completed_counts or s_sig != self._last_entry.state_sig:
                self._last_entry.outcome = "progressed"
            else:
                self._last_entry.outcome = "no_progress"
        self._last_completed_counts = c_counts
        return s_sig, c_sig

    def failed_action_sigs(self, state_sig: str, commitment_sig: str) -> set[str]:
        return {
            e.action_sig
            for e in self.entries
            if e.state_sig == state_sig and e.commitment_sig == commitment_sig and e.outcome == "no_progress"
        }

    def failed_neighborhoods(self, state_sig: str, commitment_sig: str) -> set[str]:
        return {
            e.neighborhood_key
            for e in self.entries
            if e.state_sig == state_sig and e.commitment_sig == commitment_sig and e.outcome == "no_progress"
        }

    def record_attempt(self, state_sig: str, commitment_sig: str, neighborhood_key: str, action_sig: str, tick: int) -> None:
        entry = RecoveryMemoryEntry(
            state_sig=state_sig,
            commitment_sig=commitment_sig,
            neighborhood_key=neighborhood_key,
            action_sig=action_sig,
            tick=tick,
            outcome="pending",
        )
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
        self._last_entry = entry


class PersistentPDG2Proposer(PDG2Proposer):
    """PDG 2.0 Escape augmented with Real-Time LaCAM-inspired persistent memory M_t."""

    def __init__(self, *, enable_persistent_memory: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.enable_persistent_memory = bool(enable_persistent_memory)
        self.memory = PersistentEscapeMemory()

    def __call__(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        budget_ms: float = 35.0,
    ) -> CandidateProposal:
        if not (self._active_pdg and self._active_pdg.get("pdg_alert")):
            return super().__call__(snapshot, network, budget_ms=budget_ms)
        if not self.enable_persistent_memory:
            return super().__call__(snapshot, network, budget_ms=budget_ms)

        pdg = self._active_pdg
        s_sig, c_sig = self.memory.update_previous_outcome(snapshot, pdg)
        tabu_actions = self.memory.failed_action_sigs(s_sig, c_sig)
        tried_nh = self.memory.failed_neighborhoods(s_sig, c_sig)

        # If this blocking signature has NOT failed before, run standard PDG2 escape & record its action signature
        if not tabu_actions:
            prop = super().__call__(snapshot, network, budget_ms=budget_ms)
            a_sig = hashlib.sha256(
                json.dumps(
                    {
                        "a": sorted(prop.proposed_assignment.items()),
                        "b": sorted(prop.proposed_bays.items()),
                        "lr": sorted((prop.proposed_live_repairs or {}).keys()),
                    },
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()[:16]
            self.memory.record_attempt(s_sig, c_sig, str(prop.telemetry.get("best_tag", "base")), a_sig, int(snapshot.tick))
            prop.telemetry["persistent_memory_pruned"] = self.memory.duplicate_candidates_pruned
            return prop

        # Persistent Frontier Expansion: blocking state repeated after previous attempt failed!
        # Systematically explore UNVISITED AGV rotations, eligible bay/crew shifts, and idle-blocker evacuations.
        self.pdg_recoveries_triggered += 1
        self.memory.frontier_expansions += 1
        t0 = _cvcr_now_s()
        cap_ms = float(self.hard_budget_ms) if self.hard_budget_ms > 0 else float(budget_ms)
        deadline = t0 + max(5.0, min(cap_ms * 0.72, 23.0)) / 1000.0

        clf = classify_commitments(snapshot)
        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        healthy = _healthy_robots_from_graph(graph)
        base_prio = derive_base_op_prio(snapshot)
        top_risk_ops = list(pdg.get("top_risk_ops") or ())
        boosted_prio = dict(base_prio)
        min_p = min(boosted_prio.values()) if boosted_prio else 0
        for i_r, oid in enumerate(top_risk_ops):
            if oid in boosted_prio:
                boosted_prio[oid] = int(min_p - 80 + i_r * 10)

        ev_b2 = CVCREvaluator(
            network, snapshot, mode="B2_SEEDED", alpha_makespan=3.0, lambda_change=15.0,
            use_rust=True, seq_diff_b2=bool(self.seq_diff_b2),
        )
        dd_sig = compute_detour_delta_signals(network, snapshot)
        bay_delta = dd_sig["bay_detour_delta"] if dd_sig else {}
        lazy_surr = _LazyKittingSurrogate(
            network, snapshot, boosted_prio, alpha_makespan=3.0, lambda_change=15.0, bay_detour_delta=bay_delta
        )
        w_surr = WindowAwareKittingSurrogate(lazy_surr, ev_b2, snapshot)
        fast_surr = FastJAWKittingSurrogate(network, snapshot, ev_b2, graph, lambda_change=15.0)
        fresh_ops = observable_op_closure(snapshot)
        rob_cur_h = {
            r.robot_id: r.cur_task for r in snapshot.robots if r.cur_task and r.robot_id not in clf.disturbed_robots
        }

        keep_art = CandidateArtifact(
            "E0_KEEP", "keep", dict(graph.base_assign), dict(graph.base_bays), dict(graph.base_crews),
            dict(base_prio), None, False,
        )
        self._suite._finish_artifact(
            keep_art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr,
            rob_cur_h=rob_cur_h, obs_ops=fresh_ops, fast_surr=fast_surr, compute_details=False,
        )
        keep_sc = float(keep_art.ja_unified_score or 1e9)
        keep_eval = keep_art.eval_res_b2
        assert keep_eval is not None

        e1_lr: dict[str, dict[str, Any]] = {}
        if self._sim_ref is not None:
            hv, _ = plan_haven_safe_corridor_repair(self._sim_ref, snapshot, stall_threshold=2, enable_choke_oscillation_guard=True)
            e1_lr.update(hv)
            e1_lr.update(
                _plan_idle_blocker_evacuations(
                    snapshot, network, self._sim_ref,
                    [a for a in (pdg.get("idle_blocker_agvs") or ()) if a not in e1_lr],
                    set(clf.disturbed_robots),
                )
            )
        e1_lk = _build_critical_bottleneck_locks(
            snapshot, network, sorted(set(pdg.get("stalled_agvs") or ()) | set(pdg.get("blocked_healthy_agvs") or ())),
            set(clf.disturbed_robots),
        )

        # Generate unvisited frontier candidates:
        #   1) Rotated healthy-AGV assignments for releasable tasks (rotating offset by len(tried_nh))
        #   2) Alternative eligible bays/crews on top_risk_ops (for window/bay conflicts)
        frontier: list[tuple[str, CandidateArtifact]] = []
        rot = len(tried_nh) + 1
        if healthy and graph.rel_tids:
            for shift in range(rot, rot + min(4, len(healthy))):
                asn_rot = dict(graph.base_assign)
                for ti, tid in enumerate(graph.rel_tids):
                    asn_rot[tid] = healthy[(ti + shift) % len(healthy)]
                sq_rot = _default_task_order_from_prio(snapshot, graph, asn_rot, boosted_prio)
                frontier.append(
                    (
                        f"rot_{shift}",
                        CandidateArtifact(
                            "EXP2_Persist_Rot", f"persist_rot_{shift}",
                            asn_rot, dict(graph.base_bays), dict(graph.base_crews), dict(boosted_prio), sq_rot, False,
                            live_repairs=(dict(e1_lr) if e1_lr else None),
                            bottleneck_locks=(dict(e1_lk) if e1_lk else None),
                        ),
                    )
                )
            # Alternative eligible bay/crew switch on modifiable top_risk_ops
            op_by_id = {op.op_id: op for op in snapshot.operations}
            alt_bays = dict(graph.base_bays)
            alt_crews = dict(graph.base_crews)
            switched = False
            for oid in top_risk_ops:
                if oid in graph.frozen_ops or oid not in op_by_id:
                    continue
                op_o = op_by_id[oid]
                eb = [b for b in op_o.eligible_bays if b != alt_bays.get(oid)]
                if eb:
                    alt_bays[oid] = eb[rot % len(eb)]
                    switched = True
                ec = [int(c) for c in op_o.eligible_crews if int(c) != alt_crews.get(oid)]
                if ec:
                    alt_crews[oid] = ec[rot % len(ec)]
                    switched = True
            if switched:
                a_bc, sq_bc = _regret2_insert_tasks(
                    snapshot, graph, fast_surr, dict(graph.fixed_assign), list(graph.rel_tids), boosted_prio, alt_bays
                )
                frontier.append(
                    (
                        f"baycrew_{rot}",
                        CandidateArtifact(
                            "EXP2_Persist_BayCrew", f"persist_bc_{rot}",
                            a_bc, alt_bays, alt_crews, dict(boosted_prio), sq_bc, True,
                            live_repairs=(dict(e1_lr) if e1_lr else None),
                        ),
                    )
                )

        best = keep_art
        best_sc = 1e9
        best_nh = "keep"
        eval_cnt = 1
        for nh_key, art in frontier:
            if _cvcr_now_s() >= deadline:
                break
            a_sig = hashlib.sha256(
                json.dumps(
                    {
                        "a": sorted(art.assign.items()),
                        "b": sorted(art.bays.items()),
                        "lr": sorted((art.live_repairs or {}).keys()),
                    },
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()[:16]
            if a_sig in tabu_actions:
                self.memory.duplicate_candidates_pruned += 1
                continue
            self._suite._finish_artifact(
                art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr,
                rob_cur_h=rob_cur_h, obs_ops=fresh_ops, fast_surr=fast_surr, compute_details=False,
            )
            eval_cnt += 1
            if art.eval_res_b2 and art.eval_res_b2.valid:
                sc = float(art.ja_unified_score or 1e9) - (160.0 if art.live_repairs else 0.0) - 180.0
                if sc < best_sc:
                    best_sc = sc
                    best = art
                    best_nh = nh_key

        fallback_keep = (best is keep_art)
        prop = best.to_proposal(
            policy_id=f"EXP2_PERSIST_{int(round(cap_ms))}",
            snapshot=snapshot,
            clf=clf,
            eval_res=best.eval_res_b2,
            eval_cnt=eval_cnt,
            t0=t0,
            budget_ms=cap_ms,
            fallback_keep=fallback_keep,
            include_eval_seq_when_unwindowed=True,
        )
        if not fallback_keep and best.live_repairs:
            prop.proposed_live_repairs = dict(best.live_repairs)
        if not fallback_keep and best.bottleneck_locks:
            prop.proposed_bottleneck_locks = dict(best.bottleneck_locks)

        sel_sig = hashlib.sha256(
            json.dumps(
                {
                    "a": sorted(prop.proposed_assignment.items()),
                    "b": sorted(prop.proposed_bays.items()),
                    "lr": sorted((prop.proposed_live_repairs or {}).keys()),
                },
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:16]
        self.memory.record_attempt(s_sig, c_sig, best_nh, sel_sig, int(snapshot.tick))
        dur_ms = (_cvcr_now_s() - t0) * 1000.0
        self.recovery_durations_ms.append(round(dur_ms, 3))
        prop.telemetry.update(
            {
                "best_expert": best.expert_id,
                "best_tag": best.tag,
                "persistent_frontier_expanded": True,
                "persistent_memory_pruned": self.memory.duplicate_candidates_pruned,
                "fallback_keep": fallback_keep,
            }
        )
        return prop
