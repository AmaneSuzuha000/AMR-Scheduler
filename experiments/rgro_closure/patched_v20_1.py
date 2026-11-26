"""Independent Experimental Comparator `V20_1_Patched` (`IF-CORE-RGRO-CLOSURE-01` Task 2).

Frozen `V20_1_NAJS` in `amr_sched/rl/v20_najs.py` remains 100% unmodified.
`V20_1_Patched_Proposer` applies ONLY the 4 identified engineering fixes from Deliverable 4,
with ZERO RGRO algorithmic mechanisms (no V19 Anchor pinning, no DFL, no LCB, no CRC):
  1. `CVCREvaluator(..., mode="B2", seq_diff_b2=True)` enabling `_pending_b1_seed` warm-start.
  2. Live `ev_b2.set_traffic_weights(tw, cutoff_step=cut_step)` from `execution_feedback`.
  3. Lazy/budget-guarded bay & critic preamble (never wastes unguarded time before candidate eval).
  4. Predictive `est_ev_s` (`1.2 * t_keep`) cutoff before every `ev_b2.evaluate()` in nominal & PDG 2.0 modes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.pdg_liveness import build_critical_bottleneck_locks, plan_idle_blocker_evacuations
from amr_sched.cvcr.policy import CandidateProposal, _LazyKittingSurrogate
from amr_sched.cvcr.snapshot import NativeSnapshot
from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals, plan_haven_safe_corridor_repair
from amr_sched.cvcr.v18_frontier_synthesis import WindowAwareKittingSurrogate, compute_semicircle_kitting_bays
from amr_sched.network import Network
from amr_sched.rl.causal_repair_cone import (
    FastJAWKittingSurrogate,
    build_causal_repair_cone,
    observable_op_closure,
)
from amr_sched.rl.v19_hetero_constructor import (
    _build_core_sched_proposal,
    compute_candidate_amr_footprint,
)
from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
from amr_sched.rl.v20_najs import (
    V201NAJSProposer,
    V20CandidateArtifact,
    _DEFAULT_V20_CRITIC_CKPT,
    _compute_fresh_semicircle_bays,
    _default_task_order_from_prio,
    _healthy_robots_from_graph,
    _regret2_insert_tasks,
    build_keep_baseline_stats,
    build_masked_subgraph,
    decode_v19_with_cached_enc,
    derive_base_op_prio,
    expand_causal_release_closure,
    extract_critic_features,
    generate_causal_cone_candidates,
)


class V201PatchedProposer(V201NAJSProposer):
    """Engineering-only patched comparator isolating Deliverable 4 fixes from RGRO mechanisms."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("policy_id", "V20_1_Patched")
        kwargs.setdefault("soft_deadline_ms", 14.5)
        kwargs.setdefault("critic_weights_path", _DEFAULT_V20_CRITIC_CKPT)
        super().__init__(**kwargs)

    def propose_nominal(self, snapshot: NativeSnapshot, network: Network, *, budget_ms: float = 35.0) -> CandidateProposal:
        t0 = _cvcr_now_s()
        cap_ms = float(budget_ms) if float(budget_ms) > 0 else float(self.hard_budget_ms)
        soft_ms = min(float(self.soft_deadline_ms or 14.5), max(3.5, cap_ms * 0.62))
        deadline = t0 + max(3.5, soft_ms) / 1000.0
        hard_budget_s = t0 + max(4.5, min(22.0, cap_ms * 0.78)) / 1000.0
        tick = int(snapshot.tick)

        clf = classify_commitments(snapshot)
        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        healthy = _healthy_robots_from_graph(graph)
        base_prio = derive_base_op_prio(snapshot)

        # Fix #1: literal mode="B2" enabling _pending_b1_seed
        ev_b2 = CVCREvaluator(
            network, snapshot, mode="B2", alpha_makespan=3.0, lambda_change=15.0,
            use_rust=self.use_rust, seq_diff_b2=bool(self.seq_diff_b2),
        )
        # Fix #2: live execution_feedback traffic weights
        fb = getattr(snapshot, "execution_feedback", None)
        tw: dict[tuple[str, str], int] = {}
        cut_step = 0
        if fb is not None and not fb.is_empty() and fb.blocked_intervals(t_commit=tick, default_horizon=12):
            tw = {
                (str(u), str(v)): int(w)
                for (u, v), w in fb.corridor_q3_penalties().items()
                if network.has_edge(u, v)
            }
            if tw:
                cut_step = tick + 45
                ev_b2.set_traffic_weights(tw, cutoff_step=cut_step)

        dd_sig = compute_detour_delta_signals(network, snapshot)
        bay_delta = dd_sig["bay_detour_delta"] if dd_sig else {}
        lazy_surr = _LazyKittingSurrogate(
            network, snapshot, base_prio, alpha_makespan=3.0, lambda_change=15.0, bay_detour_delta=bay_delta
        )
        fast_surr = FastJAWKittingSurrogate(network, snapshot, ev_b2, graph, lambda_change=15.0)
        fresh_ops = observable_op_closure(snapshot)
        rob_cur_h = {r.robot_id: r.cur_task for r in snapshot.robots if r.cur_task and r.robot_id not in clf.disturbed_robots}

        keep_art = V20CandidateArtifact(
            "E0_KEEP", "keep", dict(graph.base_assign), dict(graph.base_bays), dict(graph.base_crews), dict(base_prio), None, False
        )
        t_k0 = _cvcr_now_s()
        self._finish_artifact(keep_art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, obs_ops=fresh_ops, fast_surr=fast_surr)
        keep_eval_s = max(0.001, _cvcr_now_s() - t_k0)

        best = keep_art
        keep_sc = float(keep_art.ja_unified_score or 1e9)
        keep_eval = keep_art.eval_res_b2
        assert keep_eval is not None
        keep_seq = {k: list(v) for k, v in keep_eval.robot_order_seq.items() if v} if keep_eval.robot_order_seq else None
        ev_b2.set_incumbent(graph.base_assign, keep_eval, op_priority=base_prio, op_bay_overrides=graph.base_bays, task_order_override=keep_seq)

        is_ws_net = len(network.nodes) >= 30
        use_critic = bool(
            self._critic is not None and self.critic_beta > 0.0 and keep_eval.valid and (self.critic_mode != "ws_only" or is_ws_net)
        )
        keep_stats = None
        keep_critic_raw = 0.0
        sim_h_critic = int(keep_art.ja_proxy_details.get("sim_horizon", max(600, tick + 300)))
        if use_critic:
            keep_stats = build_keep_baseline_stats(snapshot, graph, keep_eval, keep_sc, rob_cur_h, sim_h_critic)
            fv_keep = extract_critic_features(
                snapshot, network, graph, keep_eval, keep_sc,
                keep_art.assign, keep_art.bays, keep_art.crews, keep_art.task_orders,
                keep_stats, rob_cur_h, sim_h_critic, is_keep_cand=True,
            )
            keep_critic_raw = self._critic.predict_delta_keep(fv_keep)
        keep_art.predicted_score = 0.0

        def _score_cand_delta(art: V20CandidateArtifact) -> float:
            ja_delta = float(art.ja_unified_score or 1e9) - keep_sc
            if not use_critic or keep_stats is None or art.eval_res_b2 is None or not art.eval_res_b2.valid:
                return ja_delta
            fv_c = extract_critic_features(
                snapshot, network, graph, art.eval_res_b2, float(art.ja_unified_score or keep_sc),
                art.assign, art.bays, art.crews, art.task_orders,
                keep_stats, rob_cur_h, sim_h_critic, is_keep_cand=False,
            )
            c_delta = float(self._critic.predict_delta_keep(fv_c) - keep_critic_raw)
            return float(ja_delta + self.critic_beta * c_delta)

        has_win = any(getattr(op, "resource_windows", ()) for op in snapshot.operations if not getattr(op, "completed", False))
        eval_cnt = 1
        seen_sigs: set[str] = {keep_art.ensure_sig()}
        nn_eval_pool: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]], bool]] = []
        v18_skeletons: list[V20CandidateArtifact] = []
        max_inc_eval_s = max(0.001, 0.55 * keep_eval_s)
        max_full_eval_s = keep_eval_s

        def _consider(art: V20CandidateArtifact, *, is_skeleton: bool = False) -> bool:
            nonlocal best, eval_cnt, max_inc_eval_s, max_full_eval_s
            sig = art.ensure_sig()
            if sig in seen_sigs or _cvcr_now_s() >= deadline:
                return False
            # Fix #4: predictive t_keep & footprint guard before starting evaluation
            fp = compute_candidate_amr_footprint(
                snapshot, best.assign, best.bays,
                best.eval_res_b2.robot_order_seq if best.eval_res_b2 else None,
                art.assign, art.bays, art.task_orders,
            )
            est_ev_s = max_inc_eval_s if fp["is_b2_incremental_hit"] else max(max_full_eval_s, 1.15 * keep_eval_s)
            if (_cvcr_now_s() + est_ev_s) >= hard_budget_s:
                return False

            seen_sigs.add(sig)
            t_ev0 = _cvcr_now_s()
            self._finish_artifact(art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, obs_ops=fresh_ops, fast_surr=fast_surr)
            dt_ev_s = max(0.0005, _cvcr_now_s() - t_ev0)
            if fp["is_b2_incremental_hit"]:
                max_inc_eval_s = max(max_inc_eval_s, dt_ev_s)
            else:
                max_full_eval_s = max(max_full_eval_s, dt_ev_s)
            eval_cnt += 1

            if art.eval_res_b2 and art.eval_res_b2.valid:
                nn_eval_pool.append((art.tag, art.assign, art.bays, art.crews, art.op_prio, art.task_orders, art.is_full))
                if is_skeleton:
                    v18_skeletons.append(art)
                sc_new = _score_cand_delta(art)
                art.predicted_score = sc_new
                if sc_new + 1e-4 < float(best.predicted_score if best.predicted_score is not None else 0.0):
                    best = art
                    if art.eval_res_b2.valid:
                        ev_b2.set_incumbent(best.assign, art.eval_res_b2, op_priority=best.op_prio, op_bay_overrides=best.bays, task_order_override=best.task_orders)
                return True
            return False

        # Fix #3: Run fast V19 NN constructor FIRST if tight, before semicircle preamble
        np_full_net, np_staged_net = self._v19_anchor._ensure_infer_nets()
        full_cache: Optional[tuple[dict[str, Any], dict[str, Any]]] = None

        def _ensure_full_enc() -> tuple[dict[str, Any], dict[str, Any]]:
            nonlocal full_cache
            if full_cache is None:
                ef = np_full_net.encode_entities(graph)
                hf = np_full_net.a1_a2_heads(ef, graph)
                full_cache = (ef, hf)
            return full_cache

        if graph.rel_tids and healthy and _cvcr_now_s() + 0.001 < deadline:
            if _cvcr_now_s() + 0.002 < deadline:
                ef, hf = _ensure_full_enc()
                dec_f = decode_v19_with_cached_enc(np_full_net, ef, hf, graph, snapshot)
                _consider(
                    V20CandidateArtifact(
                        "E2_AHG_Full", "ahg_full_g",
                        dict(dec_f.assign), dict(dec_f.bays), dict(dec_f.crews), dict(dec_f.op_prio),
                        dict(dec_f.task_sequences) if dec_f.task_sequences else None, True,
                    )
                )

            pending_or_dist = self._compute_pending_or_disturbed(snapshot, graph, clf)
            bays_semi, crews_semi = dict(graph.base_bays), dict(graph.base_crews)
            eff_km_prio = dict(base_prio)
            if _cvcr_now_s() + 0.003 < deadline:
                bays_semi, crews_semi = compute_semicircle_kitting_bays(snapshot, network, clf, lazy_surr, graph.base_bays, bay_delta)
                _, _, km_prio = _compute_fresh_semicircle_bays(snapshot, network, graph, fast_surr, fresh_ops)
                eff_km_prio = km_prio or base_prio

                if pending_or_dist:
                    anc = {tid: aid for tid, aid in graph.base_assign.items() if tid not in pending_or_dist}
                    a_z, sq_z = _regret2_insert_tasks(snapshot, graph, fast_surr, anc, sorted(pending_or_dist), dict(eff_km_prio), dict(bays_semi))
                    _consider(V20CandidateArtifact("E1_V18_Regret2", "r2_zc", a_z, dict(bays_semi), dict(crews_semi), dict(eff_km_prio), sq_z, False), is_skeleton=True)

            if (is_ws_net or not keep_eval.valid) and _cvcr_now_s() + 0.002 < deadline:
                enc_s = np_staged_net.encode_entities(graph)
                h12_s = np_staged_net.a1_a2_heads(enc_s, graph)
                dec_s = decode_v19_with_cached_enc(np_staged_net, enc_s, h12_s, graph, snapshot)
                if not keep_eval.valid:
                    _consider(
                        V20CandidateArtifact(
                            "E3_AHG_Staged", "ahg_staged_g",
                            dict(dec_s.assign), dict(dec_s.bays), dict(dec_s.crews), dict(dec_s.op_prio),
                            dict(dec_s.task_sequences) if dec_s.task_sequences else None, False,
                        )
                    )
                from amr_sched.rl.v20_najs import project_zero_churn_candidate

                zc_a, zc_b, zc_c, zc_p, zc_sq = project_zero_churn_candidate(
                    snapshot, graph, dec_s.assign, dec_s.bays, dec_s.crews, dec_s.op_prio, dec_s.task_sequences,
                    fresh_ops, pending_or_dist, rob_cur_h,
                )
                _consider(V20CandidateArtifact("E3_AHG_Staged", "ahg_staged_zc", zc_a, zc_b, zc_c, zc_p, zc_sq, False))

            if (is_ws_net or not keep_eval.valid) and (_cvcr_now_s() + 0.003 < deadline) and getattr(self._v18_pol, "pgmpo", None) is not None and self._v18_pol.pgmpo.ready:
                asns = self._v18_pol.pgmpo.generate_assignments(snapshot, network, clf, healthy, graph.fixed_assign, bays_semi, lazy_surr)
                if asns:
                    top_p = min(
                        asns,
                        key=lambda a: float(
                            fast_surr.score(a, eff_km_prio, bays_semi, crews_semi, _default_task_order_from_prio(snapshot, graph, a, eff_km_prio))
                        ),
                    )
                    p_ord = _default_task_order_from_prio(snapshot, graph, top_p, eff_km_prio)
                    if _cvcr_now_s() + 0.002 < deadline:
                        _consider(
                            V20CandidateArtifact("E4_V18_PGMPO", "pgmpo_top", dict(top_p), dict(bays_semi), dict(crews_semi), dict(eff_km_prio), p_ord, True),
                            is_skeleton=True,
                        )

            if (_cvcr_now_s() + 0.002 < deadline) and v18_skeletons:
                best_skel = min(v18_skeletons, key=lambda c: float(c.predicted_score if c.predicted_score is not None else 0.0))
                if best_skel.eval_res_b2 and best_skel.eval_res_b2.valid:
                    ef, hf = _ensure_full_enc()
                    cone = build_causal_repair_cone(
                        snapshot, graph, best_skel.eval_res_b2, best_skel.assign, best_skel.bays, best_skel.crews,
                        max_cone_amrs=3, max_cone_tasks=5, max_cone_ops=3,
                    )
                    rel_t_sub, rel_o_sub = expand_causal_release_closure(
                        snapshot, graph, cone.cone_tasks, cone.cone_ops, max_tasks=6, max_ops=4,
                    )
                    if rel_t_sub or rel_o_sub:
                        sub_g = build_masked_subgraph(
                            graph, release_tasks=rel_t_sub, release_ops=rel_o_sub,
                            skel_assign=best_skel.assign, skel_bays=best_skel.bays, skel_crews=best_skel.crews,
                            restrict_agvs=set(cone.cone_amrs) if cone.cone_amrs else None,
                        )
                        dec_bi = decode_v19_with_cached_enc(np_full_net, ef, hf, sub_g, snapshot)
                        _consider(
                            V20CandidateArtifact(
                                "V20B_Bidir_V18ToV19", "bidir_v18_to_v19",
                                dict(dec_bi.assign), dict(dec_bi.bays), dict(dec_bi.crews), dict(dec_bi.op_prio),
                                dict(dec_bi.task_sequences) if dec_bi.task_sequences else None, False,
                            )
                        )

            if _cvcr_now_s() + 0.002 < deadline:
                nn_t = nn_eval_pool[:4] if nn_eval_pool else [(best.tag, best.assign, best.bays, best.crews, best.op_prio, best.task_orders, best.is_full)]
                for tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc in generate_causal_cone_candidates(
                    snapshot, network, graph, ev_b2, fast_surr, keep_eval, base_prio, nn_t, fresh_ops, pending_or_dist,
                    budget_ms_limit=1.2,
                )[:4]:
                    _consider(V20CandidateArtifact("E7_CausalCone_VOI", tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc))

        changed_ops_cnt = sum(
            1
            for oid in graph.op_ids
            if best.bays.get(oid, graph.base_bays.get(oid)) != graph.base_bays.get(oid)
            or int(best.crews.get(oid, graph.base_crews.get(oid, 0))) != int(graph.base_crews.get(oid, 0))
        )
        changed_tasks_cnt = sum(
            1 for tid in graph.rel_tids if graph.base_assign.get(tid) and best.assign.get(tid) != graph.base_assign.get(tid)
        )
        eff_hys = 8.0 if (changed_ops_cnt > 0 or changed_tasks_cnt > 0) else 0.5
        best_rel_delta = float(best.predicted_score if best.predicted_score is not None else 0.0)
        fallback_keep = not (best.eval_res_b2 is not None and best.eval_res_b2.valid and best_rel_delta + eff_hys < 0.0)
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
                policy_id=self.policy_id,
                snapshot=snapshot,
                clf=clf,
                eval_res=best.eval_res_b2,
                eval_cnt=eval_cnt,
                t0=t0,
                budget_ms=cap_ms,
                fallback_keep=fallback_keep,
                include_eval_seq_when_unwindowed=inc_unwin,
            )
            if tw:
                prop.proposed_traffic_weights = tw
                prop.proposed_traffic_cutoff = cut_step
        finally:
            best.task_orders = orig_task_orders

        prop.telemetry.update({"best_expert": best.expert_id, "best_tag": best.tag, "fallback_keep": fallback_keep, "eval_cnt": int(eval_cnt), "patched": True})
        return prop

    def __call__(self, snapshot: NativeSnapshot, network: Network, *, budget_ms: float = 35.0) -> CandidateProposal:
        if not (self._active_pdg and self._active_pdg.get("pdg_alert")):
            return self.propose_nominal(snapshot, network, budget_ms=budget_ms)

        self.pdg_recoveries_triggered += 1
        pdg = self._active_pdg
        t0 = _cvcr_now_s()
        cap_ms = min(float(budget_ms) if float(budget_ms) > 0 else float(self.hard_budget_ms), 29.0)
        deadline = t0 + max(4.0, min(14.5, cap_ms * 0.58)) / 1000.0
        hard_budget_s = t0 + max(5.0, min(18.0, cap_ms * 0.72)) / 1000.0
        tick = int(snapshot.tick)

        clf = classify_commitments(snapshot)
        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        healthy = _healthy_robots_from_graph(graph)
        base_prio = derive_base_op_prio(snapshot)
        lvl = int(pdg.get("escalation_level", 1))
        lvl_key = f"L{max(1, min(3, lvl))}"
        top_risk_ops = list(pdg.get("top_risk_ops") or ())
        top_risk_set = set(top_risk_ops)
        boosted_prio = dict(base_prio)
        min_p = min(boosted_prio.values()) if boosted_prio else 0
        for rank_i, oid in enumerate(top_risk_ops):
            if oid in boosted_prio:
                boosted_prio[oid] = int(min_p - 60 + rank_i * 10)

        ev_b2 = CVCREvaluator(
            network, snapshot, mode="B2", alpha_makespan=3.0, lambda_change=15.0,
            use_rust=self.use_rust, seq_diff_b2=bool(self.seq_diff_b2),
        )
        fb = getattr(snapshot, "execution_feedback", None)
        tw: dict[tuple[str, str], int] = {}
        cut_step = 0
        if fb is not None and not fb.is_empty() and fb.blocked_intervals(t_commit=tick, default_horizon=12):
            tw = {
                (str(u), str(v)): int(w)
                for (u, v), w in fb.corridor_q3_penalties().items()
                if network.has_edge(u, v)
            }
            if tw:
                cut_step = tick + 45
                ev_b2.set_traffic_weights(tw, cutoff_step=cut_step)

        dd_sig = compute_detour_delta_signals(network, snapshot)
        bay_delta = dd_sig["bay_detour_delta"] if dd_sig else {}
        fast_surr = FastJAWKittingSurrogate(network, snapshot, ev_b2, graph, lambda_change=15.0)
        fresh_ops = observable_op_closure(snapshot)

        t_k0 = _cvcr_now_s()
        keep_art = V20CandidateArtifact("E0_KEEP", "keep", dict(graph.base_assign), dict(graph.base_bays), dict(graph.base_crews), dict(base_prio), None, False)
        self._finish_artifact(keep_art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, obs_ops=fresh_ops, fast_surr=fast_surr)
        keep_eval_s = max(0.001, _cvcr_now_s() - t_k0)

        best = keep_art
        keep_sc = float(keep_art.ja_unified_score or 1e9)
        keep_eval = keep_art.eval_res_b2
        assert keep_eval is not None
        keep_seq = {k: list(v) for k, v in keep_eval.robot_order_seq.items() if v} if keep_eval.robot_order_seq else None
        ev_b2.set_incumbent(graph.base_assign, keep_eval, op_priority=base_prio, op_bay_overrides=graph.base_bays, task_order_override=keep_seq)
        keep_art.predicted_score = 0.0

        op_by_id = {op.op_id: op for op in snapshot.operations}
        cycle_agvs = set(pdg.get("wait_cycle_agvs") or ())
        stalled_agvs = set(pdg.get("stalled_agvs") or ())
        blocked_agvs = set(pdg.get("blocked_healthy_agvs") or ())
        idle_blockers = list(pdg.get("idle_blocker_agvs") or ())
        distressed_agvs = cycle_agvs | stalled_agvs | blocked_agvs
        unblocked_healthy = [a for a in healthy if a not in distressed_agvs]
        risk_tasks = {kt for oid in top_risk_ops if oid in op_by_id for kt in op_by_id[oid].kit_task_ids if kt in graph.rel_tids}
        has_win = any(getattr(op, "resource_windows", ()) for op in snapshot.operations if op.end_step is None)

        e1_live_repairs: dict[str, dict[str, Any]] = {}
        if self._sim_ref is not None and not getattr(self._sim_ref, "executions", None) and len(network.nodes) <= 100 and (_cvcr_now_s() + 0.004 < deadline):
            haven_r, _ = plan_haven_safe_corridor_repair(self._sim_ref, snapshot, stall_threshold=3, enable_choke_oscillation_guard=True)
            e1_live_repairs.update(haven_r)
            idle_r = plan_idle_blocker_evacuations(
                snapshot, network, self._sim_ref, [a for a in idle_blockers if a not in e1_live_repairs], set(clf.disturbed_robots)
            )
            e1_live_repairs.update(idle_r)
        e1_locks = (
            build_critical_bottleneck_locks(snapshot, network, sorted(distressed_agvs), set(clf.disturbed_robots))
            if (len(network.nodes) <= 100 and _cvcr_now_s() + 0.003 < deadline)
            else {}
        )

        s_sig, c_sig = self.memory.update_previous_outcome(snapshot, pdg)
        _tabu_actions = self.memory.failed_action_sigs(s_sig, c_sig)
        eval_cnt = 1
        seen_sigs: set[str] = {keep_art.ensure_sig()}
        max_inc_s = max(0.001, 0.60 * keep_eval_s)

        def _risk_score(art: V20CandidateArtifact) -> float:
            er = art.eval_res_b2
            sc = float(art.ja_unified_score or 1e9)
            if er is None or not er.valid:
                return sc + 10000.0
            relieved_blocked = sum(
                1 for tid in graph.rel_tids
                if graph.base_assign.get(tid) in distressed_agvs
                and art.assign.get(tid) != graph.base_assign.get(tid)
                and art.assign.get(tid) in unblocked_healthy
            )
            relieved_cycle = sum(
                1 for tid in graph.rel_tids
                if graph.base_assign.get(tid) in cycle_agvs and art.assign.get(tid) != graph.base_assign.get(tid)
            )
            changed_risk_tasks = sum(1 for tid in risk_tasks if art.assign.get(tid) != graph.base_assign.get(tid))
            relief = (
                -260.0 * float(min(3, relieved_blocked))
                - 180.0 * float(min(3, relieved_cycle))
                - 150.0 * float(min(3, changed_risk_tasks))
                - (160.0 if art.live_repairs else 0.0)
                - (90.0 if art.bottleneck_locks else 0.0)
            )
            return (sc - keep_sc) + 0.4 * float(er.starvation_ticks - keep_eval.starvation_ticks) + relief

        def _consider(art: V20CandidateArtifact) -> bool:
            nonlocal best, eval_cnt, max_inc_s
            sig = art.ensure_sig()
            if sig in seen_sigs or _cvcr_now_s() >= deadline:
                return False
            fp = compute_candidate_amr_footprint(
                snapshot, best.assign, best.bays,
                best.eval_res_b2.robot_order_seq if best.eval_res_b2 else None,
                art.assign, art.bays, art.task_orders,
            )
            est_s = max_inc_s if fp["is_b2_incremental_hit"] else max(keep_eval_s, 1.15 * keep_eval_s)
            if (_cvcr_now_s() + est_s) >= hard_budget_s:
                return False
            seen_sigs.add(sig)
            t_ev0 = _cvcr_now_s()
            self._finish_artifact(art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, obs_ops=fresh_ops, fast_surr=fast_surr)
            dt_s = max(0.0005, _cvcr_now_s() - t_ev0)
            if fp["is_b2_incremental_hit"]:
                max_inc_s = max(max_inc_s, dt_s)
            eval_cnt += 1
            if art.eval_res_b2 and art.eval_res_b2.valid:
                r_sc = _risk_score(art)
                art.predicted_score = r_sc
                if r_sc + 1e-4 < float(best.predicted_score if best.predicted_score is not None else 0.0):
                    best = art
                    ev_b2.set_incumbent(best.assign, art.eval_res_b2, op_priority=best.op_prio, op_bay_overrides=best.bays, task_order_override=best.task_orders)
                return True
            return False

        if e1_live_repairs or e1_locks or risk_tasks:
            hoisted_seq = _default_task_order_from_prio(snapshot, graph, graph.base_assign, boosted_prio)
            _consider(
                V20CandidateArtifact(
                    "PDG2_E1_QueueCorridor", f"e1_direct_{lvl_key}",
                    dict(graph.base_assign), dict(graph.base_bays), dict(graph.base_crews), dict(boosted_prio), hoisted_seq, False,
                    live_repairs=(dict(e1_live_repairs) if e1_live_repairs else None),
                    bottleneck_locks=(dict(e1_locks) if e1_locks else None),
                )
            )

        if healthy and graph.rel_tids and _cvcr_now_s() < deadline:
            blocked_rel = [tid for tid in graph.rel_tids if graph.base_assign.get(tid) in distressed_agvs]
            if blocked_rel and unblocked_healthy:
                asn_ev = dict(graph.base_assign)
                for idx_t, tid in enumerate(blocked_rel):
                    asn_ev[tid] = unblocked_healthy[idx_t % len(unblocked_healthy)]
                _consider(
                    V20CandidateArtifact(
                        "PDG2_R1_EvictBlocked", f"r1_evict_{lvl_key}",
                        asn_ev, dict(graph.base_bays), dict(graph.base_crews), dict(boosted_prio),
                        _default_task_order_from_prio(snapshot, graph, asn_ev, boosted_prio), False,
                        live_repairs=(dict(e1_live_repairs) if e1_live_repairs else None),
                        bottleneck_locks=(dict(e1_locks) if e1_locks else None),
                    )
                )

        fallback_keep = (best is keep_art) and not (best.live_repairs or best.bottleneck_locks)
        prop = _build_core_sched_proposal(
            policy_id=self.policy_id,
            snapshot=snapshot,
            clf=clf,
            best_assign=best.assign,
            best_bays=best.bays,
            best_crews=best.crews,
            best_eval=best.eval_res_b2,
            op_prio=best.op_prio,
            eval_cnt=eval_cnt,
            t0=t0,
            budget_ms=cap_ms,
            fallback_keep=fallback_keep,
            task_sequences=best.task_orders,
            predicted_score=best.predicted_score,
            include_eval_seq_when_unwindowed=not has_win,
        )
        if tw:
            prop.proposed_traffic_weights = tw
            prop.proposed_traffic_cutoff = cut_step
        if best.live_repairs:
            prop.proposed_live_repairs = dict(best.live_repairs)
        if best.bottleneck_locks:
            prop.proposed_bottleneck_locks = dict(best.bottleneck_locks)
        prop.telemetry.update({"best_expert": best.expert_id, "best_tag": best.tag, "patched": True, "eval_cnt": int(eval_cnt)})
        return prop
