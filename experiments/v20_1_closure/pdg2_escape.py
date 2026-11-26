"""PDG 2.0 Structured Escape Recovery Module (E1 + E2).

Implements two independently toggleable recovery mechanisms on top of V20BProposer:
  - E1: Structured queue reorder (`proposed_task_sequences`), corridor priority locks
        (`proposed_bottleneck_locks`), and commitment-preserving corridor/idle-blocker evacuation
        (`proposed_live_repairs`: `picked_reroute`, `assigned_reroute`, `idle_park`).
  - E2: State-driven escalation ladder `R0 -> R1 -> R2 -> R3 -> R4` governed by observable
        blocking structure, consecutive non-recovery steps, and remaining adjustable slack,
        with risk-adjusted escape scoring that breaks short-term KEEP stalemate (`M3`).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Optional

from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.policy import CandidateProposal, LGCVCRPolicy
from amr_sched.cvcr.snapshot import NativeSnapshot, align_start_to_windows
from amr_sched.cvcr.v12_proactive_guard import (
    analyze_network_havens,
    compute_detour_delta_signals,
    plan_haven_safe_corridor_repair,
)
from amr_sched.cvcr.v18_frontier_synthesis import (
    _surrogate_polish_assign,
    compute_semicircle_kitting_bays,
)
from amr_sched.network import Network
from amr_sched.rl.causal_repair_cone import (
    FastJAWKittingSurrogate,
    _compute_fresh_semicircle_bays,
    _regret2_insert_tasks,
    generate_causal_cone_candidates,
    observable_op_closure,
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
from experiments.v20_audit.v20_b_bidirectional import (
    V20BProposer,
    build_masked_subgraph,
    decode_v19_with_cached_enc,
    expand_causal_release_closure,
)


def _plan_idle_blocker_evacuations(
    snapshot: NativeSnapshot,
    network: Network,
    sim_ref: Any,
    idle_blocker_agvs: list[str],
    faulted_robots: set[str],
) -> dict[str, dict[str, Any]]:
    """Build valid, reservation-checked `idle_park` live repairs moving idle AGVs off active/blocked nodes."""
    if not idle_blocker_agvs or sim_ref is None:
        return {}
    tick = int(snapshot.tick)
    max_steps = int(getattr(getattr(sim_ref, "config", None), "max_steps", 2000))
    transit = getattr(sim_ref, "_transit", {})
    agv_by_id = {str(a.agv_id): a for a in sim_ref.agvs}

    occupied_nodes = {str(r.cur_node) for r in snapshot.robots if r.cur_node}
    needed_nodes: set[str] = set()
    for t in snapshot.tasks:
        if t.status.lower() not in ("completed", "failed") and int(t.release_step) <= tick:
            needed_nodes.add(str(t.pickup))
            needed_nodes.add(str(t.dropoff))

    havens = analyze_network_havens(network)
    preferred_targets = list(havens.safe_havens) + [
        nid for nid in sorted(network.nodes) if nid not in set(havens.safe_havens)
    ]

    # Build adjacency map from directed network edges
    adj: dict[str, list[tuple[str, int]]] = {nid: [] for nid in network.nodes}
    for e in network.edges.values():
        step_t = max(1, int(round(float(getattr(e, "length", 1.0)) / max(0.2, float(getattr(e, "speed_limit", 1.0))))))
        adj.setdefault(str(e.src), []).append((str(e.dst), min(6, step_t)))

    repairs: dict[str, dict[str, Any]] = {}
    claimed_targets: set[str] = set()

    for aid in idle_blocker_agvs[:2]:
        if aid in faulted_robots or aid in transit:
            continue
        agv = agv_by_id.get(aid)
        if agv is None or agv.cur_task is not None:
            continue
        start_nd = str(agv.cur_node or agv.start_node)
        if start_nd not in adj:
            continue
        # BFS up to depth 4 to find nearest free parking node not blocking active pickups/dropoffs
        q: deque[tuple[str, list[tuple[str, str, int]]]] = deque([(start_nd, [])])
        visited = {start_nd}
        found_path: Optional[tuple[str, list[tuple[str, str, int]]]] = None
        while q:
            u, path = q.popleft()
            if path and u not in occupied_nodes and u not in needed_nodes and u not in claimed_targets:
                found_path = (u, path)
                break
            if len(path) >= 4:
                continue
            # Prefer haven nodes first when expanding neighbors
            nbrs = sorted(adj.get(u, ()), key=lambda x: (0 if x[0] in havens.safe_havens else 1, x[1], x[0]))
            for v, dt in nbrs:
                if v not in visited and v not in {str(r.cur_node) for r in snapshot.robots if r.robot_id in faulted_robots}:
                    visited.add(v)
                    q.append((v, path + [(u, v, dt)]))
        if found_path is None:
            # Fallback: step 1 hop to any unoccupied neighbor even if in needed_nodes
            for v, dt in adj.get(start_nd, ()):
                if v not in occupied_nodes and v not in claimed_targets:
                    found_path = (v, [(start_nd, v, dt)])
                    break
        if found_path is not None:
            goal_nd, edges_p = found_path
            cur_t = tick
            hops: list[tuple[str, str, int, int]] = []
            valid_h = True
            for u, v, dt in edges_p:
                if not network.has_edge(u, v) or cur_t + dt >= max_steps - 2:
                    valid_h = False
                    break
                hops.append((u, v, cur_t, cur_t + dt))
                cur_t += dt
            if valid_h and hops:
                claimed_targets.add(goal_nd)
                occupied_nodes.add(goal_nd)
                repairs[aid] = {
                    "kind": "idle_park",
                    "goal": goal_nd,
                    "hops": hops,
                    "min_reserve_hops": len(hops),
                }
    return repairs


def _build_critical_bottleneck_locks(
    snapshot: NativeSnapshot,
    network: Network,
    stalled_agvs: list[str],
    faulted_robots: set[str],
) -> dict[str, list[tuple[str, int, int]]]:
    """Emit short, transaction-legal corridor passage locks on choke junctions for stalled healthy AGVs."""
    if not stalled_agvs:
        return {}
    tick = int(snapshot.tick)
    havens = analyze_network_havens(network)
    r_by_id = {r.robot_id: r for r in snapshot.robots}
    locks: dict[str, list[tuple[str, int, int]]] = {}
    for idx, aid in enumerate(stalled_agvs[:2]):
        if aid in faulted_robots:
            continue
        r = r_by_id.get(aid)
        if r is None or not r.route:
            continue
        # Lock first choke node ahead on stalled AGV's route
        for nd in r.route[1:4]:
            if nd in havens.choke_nodes and nd in network.nodes:
                lo = tick + 1 + idx * 3
                hi = lo + 4
                locks[aid] = [(str(nd), lo, hi)]
                break
    return locks


class PDG2Proposer(V20BProposer):
    """V20-B Winning Bidirectional Solver + Modular PDG 2.0 Escape (`E1` & `E2`)."""

    def __init__(
        self,
        *,
        enable_e1: bool = True,
        enable_e2: bool = True,
        max_r4_retries: int = 6,
        **v20b_kwargs: Any,
    ) -> None:
        super().__init__(**v20b_kwargs)
        self.enable_e1 = bool(enable_e1)
        self.enable_e2 = bool(enable_e2)
        self.max_r4_retries = int(max_r4_retries)
        self._active_pdg: Optional[dict[str, Any]] = None
        self._sim_ref: Any = None
        self.pdg_recoveries_triggered: int = 0
        self.escalation_level_counts: dict[str, int] = {}
        self.recovery_durations_ms: list[float] = []
        self.irrecoverable_reports: int = 0

    def set_pdg_state(self, pdg: Optional[dict[str, Any]]) -> None:
        self._active_pdg = pdg

    def set_pdg2_context(self, pdg: Optional[dict[str, Any]], *, sim_ref: Any = None) -> None:
        self._active_pdg = pdg
        self._sim_ref = sim_ref

    def _select_escalation_level(self, pdg: dict[str, Any], n_rel: int) -> int:
        """State-driven R1..R4 ladder governed by blocking mode, consecutive non-recovery, and adjustable slack."""
        if not self.enable_e2:
            # E1-only mode stays at R2 (queue + corridor repair)
            return 2 if self.enable_e1 else 1
        mode = str(pdg.get("blocking_mode", "global_deadlock"))
        no_rec = int(pdg.get("consecutive_no_recovery", 0))
        if n_rel == 0:
            return 2 if (self.enable_e1 and mode == "corridor_stall") else 4
        if no_rec >= 36:
            return 4
        if mode == "corridor_stall":
            return 2 if no_rec < 18 else 3
        if mode == "kitting_window_starvation":
            return 3 if no_rec >= 12 else 1
        if mode == "critical_chain_stall":
            return 1 if no_rec < 14 else (2 if no_rec < 26 else 3)
        return 3 if no_rec >= 12 else 2

    def __call__(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        budget_ms: float = 35.0,
    ) -> CandidateProposal:
        # Zero-overhead nominal fast path when PDG 2.0 is calm
        if not (self._active_pdg and self._active_pdg.get("pdg_alert")):
            return super().__call__(snapshot, network, budget_ms=budget_ms)
        if not self.enable_e1 and not self.enable_e2:
            # Fallback to V20.0 legacy PDG escape behavior when neither E1 nor E2 enabled
            return self._legacy_v20_escape(snapshot, network, budget_ms=budget_ms)

        self.pdg_recoveries_triggered += 1
        pdg = self._active_pdg
        t0 = _cvcr_now_s()
        cap_ms = float(self.hard_budget_ms) if self.hard_budget_ms > 0 else float(budget_ms)
        deadline = t0 + max(5.0, min(cap_ms * 0.72, 23.0)) / 1000.0

        clf = classify_commitments(snapshot)
        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        healthy = _healthy_robots_from_graph(graph)
        level = self._select_escalation_level(pdg, len(graph.rel_tids))
        lvl_key = f"R{level}"
        self.escalation_level_counts[lvl_key] = self.escalation_level_counts.get(lvl_key, 0) + 1

        # R4 safety exit: if prolonged R4 cannot recover, halt churning and report irrecoverable state honestly
        if level == 4 and int(pdg.get("consecutive_no_recovery", 0)) >= 36 + self.max_r4_retries * 12:
            self.irrecoverable_reports += 1

        base_prio = derive_base_op_prio(snapshot)
        top_risk_ops = list(pdg.get("top_risk_ops") or ())
        top_risk_set = set(top_risk_ops)
        # Boost critical-chain risk ops priority so queue/scheduler stops starving them
        boosted_prio = dict(base_prio)
        min_p = min(boosted_prio.values()) if boosted_prio else 0
        for rank_i, oid in enumerate(top_risk_ops):
            if oid in boosted_prio:
                boosted_prio[oid] = int(min_p - 60 + rank_i * 10)

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
        best = keep_art
        keep_sc = float(keep_art.ja_unified_score or 1e9)
        keep_eval = keep_art.eval_res_b2
        assert keep_eval is not None
        keep_seq = {k: list(v) for k, v in keep_eval.robot_order_seq.items() if v} if keep_eval.robot_order_seq else None
        ev_b2.set_incumbent(
            graph.base_assign, keep_eval, op_priority=base_prio, op_bay_overrides=graph.base_bays, task_order_override=keep_seq
        )

        # Identify structural relief targets
        t_by_id = {t.task_id: t for t in snapshot.tasks}
        op_by_id = {op.op_id: op for op in snapshot.operations}
        cycle_agvs = set(pdg.get("wait_cycle_agvs") or ())
        stalled_agvs = set(pdg.get("stalled_agvs") or ())
        blocked_agvs = set(pdg.get("blocked_healthy_agvs") or ())
        idle_blockers = list(pdg.get("idle_blocker_agvs") or ())
        distressed_agvs = cycle_agvs | stalled_agvs | blocked_agvs
        unblocked_healthy = [a for a in healthy if a not in distressed_agvs]
        risk_tasks = {
            kt
            for oid in top_risk_ops
            if oid in op_by_id
            for kt in op_by_id[oid].kit_task_ids
            if kt in graph.rel_tids
        }
        has_win = any(getattr(op, "resource_windows", ()) for op in snapshot.operations if op.end_step is None)

        # E1: Precompute commitment-preserving corridor & idle-blocker live repairs + bottleneck locks
        e1_live_repairs: dict[str, dict[str, Any]] = {}
        e1_locks: dict[str, list[tuple[str, int, int]]] = {}
        if self.enable_e1:
            if self._sim_ref is not None:
                haven_r, _ = plan_haven_safe_corridor_repair(
                    self._sim_ref, snapshot, stall_threshold=3, enable_choke_oscillation_guard=True
                )
                e1_live_repairs.update(haven_r)
                idle_r = _plan_idle_blocker_evacuations(
                    snapshot, network, self._sim_ref,
                    [a for a in idle_blockers if a not in e1_live_repairs],
                    set(clf.disturbed_robots),
                )
                e1_live_repairs.update(idle_r)
            e1_locks = _build_critical_bottleneck_locks(
                snapshot, network, sorted(distressed_agvs), set(clf.disturbed_robots)
            )

        def _risk_score(art: CandidateArtifact) -> float:
            er = art.eval_res_b2
            sc = float(art.ja_unified_score or 1e9)
            if er is None or not er.valid:
                return sc + 10000.0
            relieved_blocked = sum(
                1
                for tid in graph.rel_tids
                if graph.base_assign.get(tid) in distressed_agvs
                and art.assign.get(tid) != graph.base_assign.get(tid)
                and art.assign.get(tid) in unblocked_healthy
            )
            relieved_cycle = sum(
                1
                for tid in graph.rel_tids
                if graph.base_assign.get(tid) in cycle_agvs
                and art.assign.get(tid) != graph.base_assign.get(tid)
            )
            changed_risk_tasks = sum(
                1 for tid in risk_tasks if art.assign.get(tid) != graph.base_assign.get(tid)
            )
            changed_risk_bays = sum(
                1 for oid in top_risk_set if art.bays.get(oid) != graph.base_bays.get(oid)
            )
            queue_hoisted = bool(
                art.task_orders
                and any(
                    seq and seq[0] in risk_tasks and (keep_seq is None or keep_seq.get(aid, [None])[0] != seq[0])
                    for aid, seq in art.task_orders.items()
                )
            )
            has_lr = bool(art.live_repairs)
            has_lk = bool(art.bottleneck_locks)

            relief_bonus = (
                -260.0 * float(min(3, relieved_blocked))
                - 180.0 * float(min(3, relieved_cycle))
                - 150.0 * float(min(3, changed_risk_tasks))
                - 140.0 * float(min(2, changed_risk_bays))
                - (160.0 if has_lr else 0.0)
                - (90.0 if has_lk else 0.0)
                - (110.0 if queue_hoisted else 0.0)
            )
            starv_diff = float(er.starvation_ticks - keep_eval.starvation_ticks)
            # Only reward structural delta, never blind churn when nothing distressed was touched
            return (sc - keep_sc) + 0.4 * starv_diff + relief_bonus

        keep_art.predicted_score = 0.0
        bays_semi, crews_semi = compute_semicircle_kitting_bays(snapshot, network, clf, lazy_surr, graph.base_bays, bay_delta)
        _, _, km_prio = _compute_fresh_semicircle_bays(snapshot, network, graph, fast_surr, fresh_ops)
        eff_prio = dict(boosted_prio)
        if km_prio:
            for k_o, v_o in km_prio.items():
                if k_o not in top_risk_set:
                    eff_prio[k_o] = v_o
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

        # Candidate E1-Direct: KEEP assignment + verified corridor/idle-blocker live repairs + hoisted queue + bottleneck locks
        if self.enable_e1 and (e1_live_repairs or e1_locks or risk_tasks):
            hoisted_seq = _default_task_order_from_prio(snapshot, graph, graph.base_assign, eff_prio)
            _consider(
                CandidateArtifact(
                    "PDG2_E1_QueueCorridor", f"e1_direct_{lvl_key}",
                    dict(graph.base_assign), dict(graph.base_bays), dict(graph.base_crews),
                    dict(eff_prio), hoisted_seq, False,
                    live_repairs=(dict(e1_live_repairs) if e1_live_repairs else None),
                    bottleneck_locks=(dict(e1_locks) if e1_locks else None),
                )
            )

        if healthy and graph.rel_tids:
            # R1+: Reassign tasks off distressed/blocked healthy AGVs onto unblocked healthy AGVs
            blocked_rel = [tid for tid in graph.rel_tids if graph.base_assign.get(tid) in distressed_agvs]
            if blocked_rel and unblocked_healthy:
                asn_evict = dict(graph.base_assign)
                for idx_t, tid in enumerate(blocked_rel):
                    asn_evict[tid] = unblocked_healthy[idx_t % len(unblocked_healthy)]
                sq_ev = _default_task_order_from_prio(snapshot, graph, asn_evict, eff_prio)
                _consider(
                    CandidateArtifact(
                        "PDG2_R1_EvictBlocked", f"r1_evict_{lvl_key}",
                        asn_evict, dict(graph.base_bays), dict(graph.base_crews), dict(eff_prio), sq_ev, False,
                        live_repairs=(dict(e1_live_repairs) if (self.enable_e1 and e1_live_repairs) else None),
                        bottleneck_locks=(dict(e1_locks) if (self.enable_e1 and e1_locks) else None),
                    )
                )

            # R1+: Critical-chain focused Regret-2 insertion on risk_tasks + pending_or_dist
            focus_t = sorted(set(risk_tasks) | set(blocked_rel) | set(pending_or_dist))
            if focus_t:
                anc_f = {tid: aid for tid, aid in graph.base_assign.items() if tid not in focus_t}
                a_f, sq_f = _regret2_insert_tasks(snapshot, graph, fast_surr, anc_f, focus_t, eff_prio, bays_semi)
                _consider(
                    CandidateArtifact(
                        "PDG2_R1_CritRegret2", f"r1_crit_r2_{lvl_key}",
                        a_f, dict(bays_semi), dict(crews_semi), dict(eff_prio), sq_f, False,
                        live_repairs=(dict(e1_live_repairs) if (self.enable_e1 and e1_live_repairs) else None),
                    )
                )

            # R3/R4: Cross-domain blocked subgraph neural reconstruction (V19 masked subgraph on top_risk_ops + unblocked AGVs)
            if self.enable_e2 and level >= 3 and _cvcr_now_s() + 0.0025 < deadline:
                rt, ro = expand_causal_release_closure(
                    snapshot, graph, focus_t[:6] or list(graph.rel_tids)[:6], top_risk_ops[:4], max_tasks=8, max_ops=8
                )
                if rt or ro:
                    sub_g = build_masked_subgraph(
                        graph,
                        release_tasks=rt,
                        release_ops=ro,
                        skel_assign=best.assign,
                        skel_bays=best.bays,
                        skel_crews=best.crews,
                        restrict_agvs=set(unblocked_healthy) if unblocked_healthy else None,
                    )
                    np_full = self._suite._v19_anchor._np_full_net
                    ef = np_full.encode_entities(sub_g)
                    hf = np_full.a1_a2_heads(ef, sub_g)
                    df = decode_v19_with_cached_enc(np_full, ef, hf, sub_g, snapshot)
                    _consider(
                        CandidateArtifact(
                            "PDG2_R3_SubgraphAHG", f"r3_sub_ahg_{lvl_key}",
                            dict(df.assign), dict(df.bays), dict(df.crews), dict(eff_prio), df.task_sequences, False,
                            live_repairs=(dict(e1_live_repairs) if (self.enable_e1 and e1_live_repairs) else None),
                            bottleneck_locks=(dict(e1_locks) if (self.enable_e1 and e1_locks) else None),
                        )
                    )

            # R3/R4: Causal cone + full Regret-2 re-init on soft set
            if level >= 2 and _cvcr_now_s() + 0.0025 < deadline:
                nnt = nn_eval_pool[:4] if nn_eval_pool else [(best.tag, best.assign, best.bays, best.crews, best.op_prio, best.task_orders, best.is_full)]
                for tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc in generate_causal_cone_candidates(
                    snapshot, network, graph, ev_b2, fast_surr, keep_eval, eff_prio, nnt, fresh_ops, pending_or_dist,
                    budget_ms_limit=2.5,
                )[:5]:
                    _consider(
                        CandidateArtifact(
                            "E7_CausalCone_VOI", f"pdg2_{tg_c}", asn_c, bs_c, cs_c, op_c, sq_c, is_fc,
                            live_repairs=(dict(e1_live_repairs) if (self.enable_e1 and e1_live_repairs) else None),
                        )
                    )

            if level >= 3 and _cvcr_now_s() + 0.003 < deadline:
                a_g, sq_g = _regret2_insert_tasks(
                    snapshot, graph, fast_surr, dict(graph.fixed_assign), list(graph.rel_tids), eff_prio, bays_semi
                )
                _consider(
                    CandidateArtifact(
                        "PDG2_R4_SoftReinit", f"r4_reinit_{lvl_key}",
                        a_g, dict(bays_semi), dict(crews_semi), dict(eff_prio), sq_g, True,
                        live_repairs=(dict(e1_live_repairs) if (self.enable_e1 and e1_live_repairs) else None),
                    )
                )

            if level >= 3 and _cvcr_now_s() + 0.004 < deadline and len(healthy) >= 2:
                pol_a, _ = _surrogate_polish_assign(
                    w_surr, graph.rel_tids, unblocked_healthy if len(unblocked_healthy) >= 2 else healthy,
                    dict(best.assign), dict(best.bays), dict(best.crews), dict(best.op_prio),
                    max_passes=2, max_focus_tasks=10, do_swaps=True,
                )
                _consider(
                    CandidateArtifact(
                        "E6_V18_Polish", f"pdg2_polish_{lvl_key}", pol_a, dict(best.bays), dict(best.crews),
                        dict(best.op_prio), _default_task_order_from_prio(snapshot, graph, pol_a, best.op_prio), False,
                        live_repairs=(dict(e1_live_repairs) if (self.enable_e1 and e1_live_repairs) else None),
                    )
                )

        best_rel = float(best.predicted_score if best.predicted_score is not None else 0.0)
        fallback_keep = not (
            best.eval_res_b2 is not None
            and best.eval_res_b2.valid
            and (best is not keep_art)
            and (best_rel < -1e-4 or not keep_eval.valid)
        )
        if fallback_keep:
            best = keep_art

        seq_horizon = int(self.unwin_seq_horizon) if self.unwin_seq_horizon is not None else 0
        inc_unwin = False
        orig_task_orders = best.task_orders
        if seq_horizon > 0 and not has_win and not (best.expert_id == "PDG2_E1_QueueCorridor"):
            inc_unwin = True
            if isinstance(orig_task_orders, dict):
                best.task_orders = {k: list(v)[:seq_horizon] for k, v in orig_task_orders.items() if v}
        elif best.expert_id == "PDG2_E1_QueueCorridor" and best.task_orders:
            inc_unwin = True
        try:
            prop = best.to_proposal(
                policy_id=f"V20_1_PDG2_{lvl_key}_{int(round(cap_ms))}",
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

        dur_ms = (_cvcr_now_s() - t0) * 1000.0
        self.recovery_durations_ms.append(round(dur_ms, 3))
        prop.telemetry.update(
            {
                "best_expert": best.expert_id,
                "best_tag": best.tag,
                "pdg2_mode": True,
                "escalation_level": lvl_key,
                "blocking_mode": pdg.get("blocking_mode"),
                "fallback_keep": fallback_keep,
                "eval_cnt": int(eval_cnt),
                "keep_obj": round(keep_sc, 4),
                "best_obj": round(float(best.ja_unified_score or keep_sc), 4),
                "liveness_status": (
                    "IRRECOVERABLE_ESCALATED"
                    if (fallback_keep and level == 4 and int(pdg.get("consecutive_no_recovery", 0)) >= 36 + self.max_r4_retries * 12)
                    else ("ESCAPING" if not fallback_keep else "STALLED_KEEP")
                ),
            }
        )
        return prop

    def _legacy_v20_escape(self, snapshot: NativeSnapshot, network: Network, *, budget_ms: float = 35.0) -> CandidateProposal:
        """Delegate to frozen V20EProposer legacy escape for clean D1/D2-only ablations."""
        from experiments.v20_audit.v20_e_liveness import V20EProposer

        leg = V20EProposer(
            hard_budget_ms=self.hard_budget_ms,
            soft_deadline_ms=self.soft_deadline_ms,
            seed=self.seed,
            enable_e4=self.enable_e4,
            e4_selector=self.e4_selector,
            e4_include_zc=self.e4_include_zc,
            e4_mode=self.e4_mode,
            staged_mode=self.staged_mode,
            enable_bidir_v18_to_v19=self.enable_bidir_v18_to_v19,
            enable_e6_static_or_headroom=self.enable_e6_static_or_headroom,
            enable_e7=self.enable_e7,
            enable_e9=self.enable_e9,
            enable_early_stop=self.enable_early_stop,
            compact_calm_ms=self.compact_calm_ms,
            enable_r2_global_ws=self.enable_r2_global_ws,
            r2_zc_ws_mode=self.r2_zc_ws_mode,
            include_staged_g=self.include_staged_g,
            unwin_seq_horizon=self.unwin_seq_horizon,
            seq_diff_b2=self.seq_diff_b2,
            enable_pdg_recovery=True,
        )
        leg._critic = self._critic
        leg._suite = self._suite
        leg.set_pdg_state(self._active_pdg)
        self.pdg_recoveries_triggered += 1
        t0 = _cvcr_now_s()
        p = leg(snapshot, network, budget_ms=budget_ms)
        self.recovery_durations_ms.append(round((_cvcr_now_s() - t0) * 1000.0, 3))
        return p


def build_pdg2_v20_proposer(
    *,
    hard_budget_ms: float = 35.0,
    seed: int = 0,
    critic_weights_path: Optional[Path] = None,
    enable_e1: bool = True,
    enable_e2: bool = True,
) -> PDG2Proposer:
    """Instantiate V20-B backbone wrapped with PDG 2.0 escape toggles."""
    from experiments.v20_audit.common import OUT_AUDIT_DIR

    c_path = critic_weights_path if critic_weights_path is not None else (OUT_AUDIT_DIR / "v20_a_critic_weights.npz")
    return PDG2Proposer(
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
        enable_e1=enable_e1,
        enable_e2=enable_e2,
    )
