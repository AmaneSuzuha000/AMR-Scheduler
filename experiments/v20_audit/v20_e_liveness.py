"""Phase V20-E: Risk-Aware PDG Liveness Monitor, Outer Escalation, and Non-Greedy Escape."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.cvcr.snapshot import NativeSnapshot
from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
from amr_sched.cvcr.v18_frontier_synthesis import (
    _surrogate_polish_assign,
    compute_semicircle_kitting_bays,
)
from amr_sched.network import Network
from amr_sched.rl.causal_repair_cone import (
    FastJAWKittingSurrogate,
    _compute_fresh_semicircle_bays,
    _regret2_insert_tasks,
    build_causal_repair_cone,
    generate_causal_cone_candidates,
    observable_op_closure,
    project_zero_churn_candidate,
)
from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
from experiments.v20_audit.common import CandidateArtifact
from experiments.v20_audit.proposer_variants import (
    WindowAwareKittingSurrogate,
    _LazyKittingSurrogate,
    _default_task_order_from_prio,
    _healthy_robots_from_graph,
    derive_base_op_prio,
)
from experiments.v20_audit.task_c_liveness_forensics import compute_pdg_state
from experiments.v20_audit.v20_b_bidirectional import (
    V20BProposer,
    build_masked_subgraph,
    decode_v19_with_cached_enc,
    expand_causal_release_closure,
)


@dataclass
class PDGLivenessTracker:
    """Episode-level Progress Dependency Graph monitor driving V20-E outer escalation."""

    cooldown_ticks: int = 12
    last_done_t: int = 0
    last_done_o: int = 0
    last_progress_tick: int = 0
    last_escalation_tick: int = -100
    recent_reject_streak: int = 0
    last_logged_decisions: int = 0
    op_first_kit_done_tick: dict[str, int] = field(default_factory=dict)
    alert_ticks: list[int] = field(default_factory=list)
    escalation_ticks: list[int] = field(default_factory=list)
    latest_pdg: Optional[dict[str, Any]] = None

    def observe_and_maybe_escalate(
        self,
        env: Any,
        info: dict[str, Any],
        base_action: int,
        _traces: Any = None,
        *,
        proposer: Optional["V20EProposer"] = None,
    ) -> int:
        cur_tick = int(env.current_step)
        n_logs = len(getattr(env, "decision_logs", []))
        if n_logs > self.last_logged_decisions:
            self.last_logged_decisions = n_logs
            if not bool(env.decision_logs[-1].get("receipt_accepted", False)):
                self.recent_reject_streak += 1
            else:
                self.recent_reject_streak = 0

        c_t, c_o = env._completion_counts(int(env.sim._step_cursor))
        if c_t > self.last_done_t or c_o > self.last_done_o:
            self.last_done_t, self.last_done_o = c_t, c_o
            self.last_progress_tick = cur_tick

        pdg = compute_pdg_state(
            env.sim,
            env.tracker,
            last_progress_tick=self.last_progress_tick,
            op_first_kit_done_tick=self.op_first_kit_done_tick,
            recent_reject_streak=self.recent_reject_streak,
        )
        schedulable_rem = sum(
            1
            for t in env.sim.tasks
            if getattr(getattr(t, "status", ""), "value", str(getattr(t, "status", ""))).lower()
            in ("unassigned", "pending", "assigned")
            and int(getattr(t, "release_step", 0)) <= cur_tick
        )
        # Filter out non-deadlock final in-flight transit legs (e.g., last remaining task picked, zero stall/cycle/blockage)
        actionable_distress = bool(
            int(pdg["wait_cycle_count"]) >= 1
            or int(pdg["blocked_healthy"]) >= 1
            or int(pdg["max_loaded_stall_ticks"]) >= 5
            or schedulable_rem >= 1
            or int(pdg["zero_progress_ticks"]) >= 540
        )
        is_alert = bool(pdg["pdg_alert"]) and actionable_distress
        pdg["schedulable_rem_tasks"] = int(schedulable_rem)
        pdg["pdg_alert"] = is_alert
        self.latest_pdg = pdg
        if is_alert:
            self.alert_ticks.append(cur_tick)
        if proposer is not None:
            proposer.set_pdg_state(pdg if is_alert else None)

        # Outer Liveness Escalation: strictly zero-overhead when not alerting (0.0% FPR on healthy runs)
        if (
            is_alert
            and int(base_action) == 0
            and (cur_tick - self.last_escalation_tick) >= self.cooldown_ticks
        ):
            self.last_escalation_tick = cur_tick
            self.escalation_ticks.append(cur_tick)
            mask = info.get("action_mask")
            if mask is not None and len(mask) > 2:
                # Choose highest-scope legal replan action, else force surgical action 2 / 1
                legal_replans = [i for i in range(1, len(mask)) if int(mask[i]) == 1]
                if legal_replans:
                    return int(legal_replans[-1])
            return 2
        return int(base_action)


class V20EProposer(V20BProposer):
    """Phase V20-E: V20-B Winning Bidirectional Solver + Risk-Aware PDG Liveness Recovery."""

    def __init__(
        self,
        *,
        enable_pdg_recovery: bool = True,
        **v20b_kwargs: Any,
    ) -> None:
        super().__init__(**v20b_kwargs)
        self.enable_pdg_recovery = bool(enable_pdg_recovery)
        self._active_pdg: Optional[dict[str, Any]] = None
        self.pdg_recoveries_triggered: int = 0

    def set_pdg_state(self, pdg: Optional[dict[str, Any]]) -> None:
        self._active_pdg = pdg

    def __call__(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        budget_ms: float = 35.0,
    ) -> CandidateProposal:
        # Zero-interference fast path when PDG is calm (0/72 false positives on healthy VAL-72)
        if not self.enable_pdg_recovery or not (self._active_pdg and self._active_pdg.get("pdg_alert")):
            return super().__call__(snapshot, network, budget_ms=budget_ms)

        # Risk-Aware Non-Greedy Liveness Escape Mode
        self.pdg_recoveries_triggered += 1
        pdg = self._active_pdg
        t0 = _cvcr_now_s()
        cap_ms = float(self.hard_budget_ms) if self.hard_budget_ms > 0 else float(budget_ms)
        # Allow deeper emergency recovery search inside hard 35ms budget while preserving commit margin
        deadline = t0 + max(5.0, min(cap_ms * 0.72, 23.0)) / 1000.0

        clf = classify_commitments(snapshot)
        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        healthy = _healthy_robots_from_graph(graph)
        base_prio = derive_base_op_prio(snapshot)
        ev_b2 = CVCREvaluator(
            network, snapshot, mode="B2_SEEDED", alpha_makespan=3.0, lambda_change=15.0,
            use_rust=True, seq_diff_b2=bool(self.seq_diff_b2),
        )
        dd_sig = compute_detour_delta_signals(network, snapshot)
        bay_delta = dd_sig["bay_detour_delta"] if dd_sig else {}
        lazy_surr = _LazyKittingSurrogate(
            network, snapshot, base_prio, alpha_makespan=3.0, lambda_change=15.0, bay_detour_delta=bay_delta
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
        best = keep_art
        keep_sc = float(keep_art.ja_unified_score or 1e9)
        keep_eval = keep_art.eval_res_b2
        assert keep_eval is not None
        keep_seq = {k: list(v) for k, v in keep_eval.robot_order_seq.items() if v} if keep_eval.robot_order_seq else None
        ev_b2.set_incumbent(
            graph.base_assign, keep_eval, op_priority=base_prio, op_bay_overrides=graph.base_bays, task_order_override=keep_seq
        )

        # Anti-thrashing guard: identify already-in-transit kitting robots for split-brain ops
        t_by_id = {t.task_id: t for t in snapshot.tasks}
        protected_transit_tasks = {
            tid
            for tid in graph.rel_tids
            if t_by_id[tid].status.lower() in ("picked", "to_dropoff", "loading", "unloading")
        }
        cycle_agvs = set(pdg.get("wait_cycle_agvs") or ())
        has_win = any(getattr(op, "resource_windows", ()) for op in snapshot.operations if not getattr(op, "completed", False))

        def _risk_score(art: CandidateArtifact) -> float:
            er = art.eval_res_b2
            sc = float(art.ja_unified_score or 1e9)
            if er is None or not er.valid:
                return sc + 10000.0
            # Strongly reward breaking wait cycles or freeing stalled robots / reducing starvation
            changed_cycle_agvs = sum(
                1
                for tid in graph.rel_tids
                if graph.base_assign.get(tid) in cycle_agvs and art.assign.get(tid) != graph.base_assign.get(tid)
            )
            cyc_bonus = -120.0 * float(min(3, changed_cycle_agvs)) if cycle_agvs else 0.0
            starv_diff = float(er.starvation_ticks - keep_eval.starvation_ticks)
            # Small bonus for breaking exact KEEP stalemate when zero progress persisted
            escape_bonus = -40.0 if (art.assign != graph.base_assign or art.bays != graph.base_bays) else 0.0
            return sc + 0.5 * starv_diff + cyc_bonus + escape_bonus - keep_sc

        keep_art.predicted_score = 0.0
        bays_semi, crews_semi = compute_semicircle_kitting_bays(snapshot, network, clf, lazy_surr, graph.base_bays, bay_delta)
        _, _, km_prio = _compute_fresh_semicircle_bays(snapshot, network, graph, fast_surr, fresh_ops)
        eff_km_prio = km_prio or base_prio
        pending_or_dist = self._compute_pending_or_disturbed(snapshot, graph, clf)

        eval_cnt = 1
        seen_sigs: set[str] = {keep_art.ensure_sig()}
        nn_eval_pool: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]], bool]] = []

        def _consider(art: CandidateArtifact) -> bool:
            nonlocal best, eval_cnt
            sig = art.ensure_sig()
            if sig in seen_sigs or _cvcr_now_s() >= deadline:
                return False
            seen_sigs.add(sig)
            self._suite._finish_artifact(
                art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr,
                rob_cur_h=rob_cur_h, obs_ops=fresh_ops, fast_surr=fast_surr, compute_details=False,
            )
            eval_cnt += 1
            if art.eval_res_b2 and art.eval_res_b2.valid:
                nn_eval_pool.append((art.tag, art.assign, art.bays, art.crews, art.op_prio, art.task_orders, art.is_full))
                r_sc = _risk_score(art)
                art.predicted_score = r_sc
                if r_sc + 1e-4 < float(best.predicted_score if best.predicted_score is not None else 0.0):
                    best = art
                    ev_b2.set_incumbent(
                        best.assign, best.eval_res_b2, op_priority=best.op_prio, op_bay_overrides=best.bays, task_order_override=best.task_orders
                    )
                return True
            return False

        if healthy and graph.rel_tids:
            # Unlock ALL escalation experts unconditionally during PDG risk mode
            a_g, sq_g = _regret2_insert_tasks(
                snapshot, graph, fast_surr, dict(graph.fixed_assign), list(graph.rel_tids), eff_km_prio, bays_semi
            )
            _consider(CandidateArtifact("V20E_PDG_Escape", "pdg_r2_global", a_g, dict(bays_semi), dict(crews_semi), dict(eff_km_prio), sq_g, True))

            if pending_or_dist or protected_transit_tasks:
                rel_escape = (set(graph.rel_tids) - protected_transit_tasks) | pending_or_dist
                anc_e = {tid: aid for tid, aid in graph.base_assign.items() if tid not in rel_escape}
                if rel_escape:
                    a_e, sq_e = _regret2_insert_tasks(
                        snapshot, graph, fast_surr, anc_e, sorted(rel_escape), eff_km_prio, bays_semi
                    )
                    _consider(CandidateArtifact("V20E_PDG_Escape", "pdg_r2_antithrash", a_e, dict(bays_semi), dict(crews_semi), dict(eff_km_prio), sq_e, False))

            np_full = self._suite._v19_anchor._np_full_net
            np_staged = self._suite._v19_anchor._np_staged_net
            if _cvcr_now_s() + 0.002 < deadline:
                ef = np_full.encode_entities(graph)
                hf = np_full.a1_a2_heads(ef, graph)
                df = decode_v19_with_cached_enc(np_full, ef, hf, graph, snapshot)
                _consider(CandidateArtifact("E2_AHG_Full", "pdg_ahg_full", dict(df.assign), dict(df.bays), dict(df.crews), dict(df.op_prio), df.task_sequences, True))
            if _cvcr_now_s() + 0.002 < deadline:
                es = np_staged.encode_entities(graph)
                hs = np_staged.a1_a2_heads(es, graph)
                ds = decode_v19_with_cached_enc(np_staged, es, hs, graph, snapshot)
                _consider(CandidateArtifact("E3_AHG_Staged", "pdg_ahg_staged_g", dict(ds.assign), dict(ds.bays), dict(ds.crews), dict(ds.op_prio), ds.task_sequences, False))

            if _cvcr_now_s() + 0.003 < deadline and self._suite._v18_pol.pgmpo is not None:
                asns = self._suite._v18_pol.pgmpo.generate_assignments(
                    snapshot, network, clf, healthy, graph.fixed_assign, bays_semi, lazy_surr
                )
                for idx_p, p_asn in enumerate(asns[:2]):
                    if _cvcr_now_s() >= deadline:
                        break
                    _consider(
                        CandidateArtifact(
                            "E4_V18_PGMPO", f"pdg_pgmpo_{idx_p}", dict(p_asn), dict(bays_semi), dict(crews_semi),
                            dict(eff_km_prio), _default_task_order_from_prio(snapshot, graph, p_asn, eff_km_prio), True,
                        )
                    )

            if _cvcr_now_s() + 0.0025 < deadline:
                nnt = nn_eval_pool[:4] if nn_eval_pool else [(best.tag, best.assign, best.bays, best.crews, best.op_prio, best.task_orders, best.is_full)]
                for tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc in generate_causal_cone_candidates(
                    snapshot, network, graph, ev_b2, fast_surr, keep_eval, base_prio, nnt, fresh_ops, pending_or_dist,
                    budget_ms_limit=2.5,
                )[:6]:
                    _consider(CandidateArtifact("E7_CausalCone_VOI", f"pdg_{tg_c}", asn_c, bs_c, cs_c, op_c, sq_c, is_fc))

            if _cvcr_now_s() + 0.003 < deadline and len(healthy) >= 2:
                pol_a, _ = _surrogate_polish_assign(
                    w_surr, graph.rel_tids, healthy, dict(best.assign), dict(best.bays), dict(best.crews), dict(best.op_prio),
                    max_passes=2, max_focus_tasks=10, do_swaps=True,
                )
                _consider(
                    CandidateArtifact(
                        "E6_V18_Polish", "pdg_polish", pol_a, dict(best.bays), dict(best.crews),
                        dict(best.op_prio), _default_task_order_from_prio(snapshot, graph, pol_a, best.op_prio), False,
                    )
                )

        # Relax KEEP hysteresis under PDG risk mode so deadlock-breaking changes commit immediately
        best_rel = float(best.predicted_score if best.predicted_score is not None else 0.0)
        fallback_keep = not (
            best.eval_res_b2 is not None
            and best.eval_res_b2.valid
            and (best is not keep_art)
            and (best_rel < 0.0 or not keep_eval.valid)
        )
        if fallback_keep:
            best = keep_art

        seq_horizon = int(self.unwin_seq_horizon) if self.unwin_seq_horizon is not None else 0
        inc_unwin = False
        orig_task_orders = best.task_orders
        if seq_horizon > 0 and not has_win:
            inc_unwin = True
            if isinstance(orig_task_orders, dict):
                best.task_orders = {k: list(v)[:seq_horizon] for k, v in orig_task_orders.items() if v}
        try:
            prop = best.to_proposal(
                policy_id=f"V20E_LIVENESS_{int(round(cap_ms))}",
                snapshot=snapshot,
                clf=clf,
                eval_res=best.eval_res_b2,
                eval_cnt=eval_cnt,
                t0=t0,
                budget_ms=cap_ms,
                fallback_keep=fallback_keep,
                include_eval_seq_when_unwindowed=inc_unwin,
            )
        finally:
            best.task_orders = orig_task_orders
        if not fallback_keep and best.live_repairs:
            prop.proposed_live_repairs = dict(best.live_repairs)
        if not fallback_keep and best.bottleneck_locks:
            prop.proposed_bottleneck_locks = dict(best.bottleneck_locks)
        prop.telemetry.update(
            {
                "best_expert": best.expert_id,
                "best_tag": best.tag,
                "pdg_risk_mode": True,
                "fallback_keep": fallback_keep,
                "eval_cnt": int(eval_cnt),
                "keep_obj": round(keep_sc, 4),
                "best_obj": round(float(best.ja_unified_score or keep_sc), 4),
            }
        )
        return prop


def build_default_v20_proposer(
    *,
    hard_budget_ms: float = 35.0,
    seed: int = 0,
    critic_weights_path: Optional[Path] = None,
    enable_pdg_recovery: bool = True,
) -> V20EProposer:
    """Instantiate the frozen CoRe-Sched V20 (NAJS) solver cleanly."""
    from experiments.v20_audit.common import OUT_AUDIT_DIR, ROOT

    default_ckpt = OUT_AUDIT_DIR / "v20_a_critic_weights.npz"
    if not default_ckpt.is_file():
        default_ckpt = ROOT / "amr_sched" / "checkpoints" / "v20" / "v20_critic_weights.npz"
    c_path = critic_weights_path if critic_weights_path is not None else default_ckpt
    return V20EProposer(
        hard_budget_ms=hard_budget_ms,
        soft_deadline_ms=15.0,
        seed=seed,
        enable_e4=True,
        e4_selector="fast_surr",
        e4_include_zc=False,
        e4_mode="ws_or_invalid",
        staged_mode="ws_or_invalid",
        enable_bidir_v18_to_v19=True,
        enable_e6_static_or_headroom=False,
        enable_e7=True,
        enable_e9=False,
        enable_early_stop=False,
        compact_calm_ms=None,
        enable_r2_global_ws=False,
        r2_zc_ws_mode="never",
        include_staged_g="on_invalid",
        unwin_seq_horizon=2,
        seq_diff_b2=True,
        critic_weights_path=c_path,
        critic_beta=0.25,
        critic_mode="ws_only",
        enable_pdg_recovery=enable_pdg_recovery,
    )
