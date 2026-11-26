"""EXP-3 (P1): Cross-Domain ADDRESS — Production-Unlock Top-K + Restricted Thompson Sampling.

Inspired by AAAI 2025 ADDRESS (Adaptive Delay-Based Heuristic):
Instead of high-dimensional 408-dim neighborhood prediction (pruned in V20-C), computes an
ultra-fast (< 0.1 ms) Cross-Domain Seed Importance score per releasable task i:
    I_i = alpha * D_i + beta * U_i + gamma * W_i + delta * C_i
where:
  - D_i: transport delay / tardiness proxy
  - U_i: downstream production unlock value (DAG descendants + partial kitting completion bonus)
  - W_i: resource window urgency (proximity / missed windows)
  - C_i: structural waiting centrality (wait-for dependencies on the task / its kit op)

Supports controlled 4-way falsification under equal evaluator & budget:
  - N0_CausalCone           (frozen V20-B baseline)
  - N1_Address_TopK         (deterministic Top-K seed tasks by I_i -> causal closure repair)
  - N1_Random_TopK          (random Top-K control)
  - N2_Address_RestrictedTS (Top-K restricted Beta-Bernoulli Thompson Sampling updated online)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.cvcr.snapshot import NativeSnapshot, align_start_to_windows, op_predecessors
from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
from amr_sched.cvcr.v18_frontier_synthesis import compute_semicircle_kitting_bays
from amr_sched.network import Network
from amr_sched.rl.causal_repair_cone import (
    FastJAWKittingSurrogate,
    _compute_fresh_semicircle_bays,
    _regret2_insert_tasks,
    observable_op_closure,
)
from amr_sched.rl.v19_hetero_graph import V19HeteroGraph, extract_v19_hetero_graph
from experiments.v20_audit.common import CandidateArtifact
from experiments.v20_audit.proposer_variants import (
    WindowAwareKittingSurrogate,
    _LazyKittingSurrogate,
    _healthy_robots_from_graph,
    derive_base_op_prio,
)
from experiments.v20_audit.v20_b_bidirectional import (
    V20BProposer,
    build_masked_subgraph,
    decode_v19_with_cached_enc,
    expand_causal_release_closure,
)


def compute_cross_domain_seed_importance(
    snapshot: NativeSnapshot,
    graph: V19HeteroGraph,
    inc_eval: Any,
    *,
    alpha: float = 1.0,
    beta: float = 1.2,
    gamma: float = 1.1,
    delta: float = 0.8,
) -> dict[str, float]:
    """Compute scalar Cross-Domain Seed Importance I_i for every releasable task in `graph.rel_tids`."""
    tick = int(snapshot.tick)
    t_by_id = {t.task_id: t for t in snapshot.tasks}
    op_by_id = {op.op_id: op for op in snapshot.operations}

    # Build op descendant counts & bay contention centrality
    children: dict[str, list[str]] = {op.op_id: [] for op in snapshot.operations}
    bay_load: dict[str, int] = {}
    for op in snapshot.operations:
        if op.end_step is None:
            bay_load[op.bay_id] = bay_load.get(op.bay_id, 0) + 1
        for p in op_predecessors(op):
            if p in children:
                children[p].append(op.op_id)

    desc_cnt: dict[str, int] = {}
    for op in snapshot.operations:
        vis: set[str] = set()
        stk = list(children.get(op.op_id, ()))
        while stk:
            cur = stk.pop()
            if cur not in vis:
                vis.add(cur)
                stk.extend(children.get(cur, ()))
        desc_cnt[op.op_id] = len(vis)

    unloads = getattr(inc_eval, "order_unload_ticks", {}) or {}
    agv_load: dict[str, int] = {}
    for tid in graph.rel_tids:
        aid = graph.base_assign.get(tid)
        if aid:
            agv_load[aid] = agv_load.get(aid, 0) + 1

    scores: dict[str, float] = {}
    for tid in graph.rel_tids:
        t = t_by_id.get(tid)
        if t is None:
            scores[tid] = 0.0
            continue
        due = int(t.due_step) if t.due_step is not None else (tick + 120)
        est_fin = int(unloads.get(tid, tick + 40))
        # D_i: transport delay / tardiness
        d_i = max(0.0, float(est_fin - due)) / 100.0 + (0.6 if t.is_urgent else 0.0)

        # U_i: production unlock value (descendants + partial kit completion bonus)
        u_i = 0.0
        w_i = 0.0
        c_i = 0.25 * float(agv_load.get(str(graph.base_assign.get(tid, "")), 0))
        op = op_by_id.get(str(t.kit_op_id or ""))
        if op is not None and op.end_step is None:
            k_tot = max(1, len(op.kit_task_ids))
            k_done = sum(1 for kt in op.kit_task_ids if kt in t_by_id and t_by_id[kt].status.lower() == "completed")
            partial_bonus = 1.2 if (0 < k_done < k_tot) else (0.5 if k_tot == 1 else 0.2)
            u_i = 0.35 * float(min(6, desc_cnt.get(op.op_id, 0))) + partial_bonus
            # W_i: resource window urgency
            wins = tuple(getattr(op, "resource_windows", ()) or ())
            if wins:
                missed = sum(1 for _, we in wins if est_fin > int(we))
                align_gap = max(0, align_start_to_windows(est_fin, wins) - est_fin)
                w_i = 0.9 * float(missed) + min(1.5, float(align_gap) / 120.0)
            # C_i: bay contention + kit fan-in centrality
            c_i += 0.3 * float(bay_load.get(op.bay_id, 0)) + 0.2 * float(k_tot)
        scores[tid] = float(alpha * d_i + beta * u_i + gamma * w_i + delta * c_i)
    return scores


class CrossDomainAddressProposer(V20BProposer):
    """EXP-3 Proposer augmenting V20-B with ADDRESS-style Cross-Domain Top-K / Restricted TS neighborhood repair."""

    def __init__(
        self,
        *,
        address_mode: str = "N1_Address_TopK",  # N0_CausalCone | N1_Address_TopK | N1_Random_TopK | N2_Address_RestrictedTS
        top_k: int = 4,
        ts_pool_k: int = 6,
        **v20b_kwargs: Any,
    ) -> None:
        super().__init__(**v20b_kwargs)
        self.address_mode = str(address_mode)
        self.top_k = int(top_k)
        self.ts_pool_k = int(ts_pool_k)
        self._rng = np.random.default_rng(int(self.seed) + 202503)
        # Restricted Thompson Sampling Beta(alpha, beta) arms over rank buckets 0..ts_pool_k-1
        self._ts_alpha = np.ones((self.ts_pool_k,), dtype=np.float64)
        self._ts_beta = np.ones((self.ts_pool_k,), dtype=np.float64)
        self.address_improvements: int = 0
        self.address_calls: int = 0

    def __call__(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        budget_ms: float = 35.0,
    ) -> CandidateProposal:
        if self.address_mode == "N0_CausalCone":
            return super().__call__(snapshot, network, budget_ms=budget_ms)

        t0 = _cvcr_now_s()
        cap_ms = float(self.hard_budget_ms) if self.hard_budget_ms > 0 else float(budget_ms)
        soft_ms = (
            min(float(self.soft_deadline_ms), max(3.5, cap_ms * 0.72))
            if self.soft_deadline_ms is not None
            else max(3.5, cap_ms * 0.68)
        )
        # Run V20-B baseline first reserving ~2.5 ms for targeted ADDRESS neighborhood refinement
        orig_soft = self.soft_deadline_ms
        self.soft_deadline_ms = max(4.0, soft_ms - 2.2)
        try:
            base_prop = super().__call__(snapshot, network, budget_ms=cap_ms)
        finally:
            self.soft_deadline_ms = orig_soft

        deadline = t0 + soft_ms / 1000.0
        if _cvcr_now_s() + 0.0012 >= deadline:
            return base_prop

        clf = classify_commitments(snapshot)
        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        healthy = _healthy_robots_from_graph(graph)
        if not healthy or len(graph.rel_tids) < 2:
            return base_prop

        self.address_calls += 1
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

        inc_assign = {**graph.base_assign, **dict(base_prop.proposed_assignment)}
        inc_bays = {**graph.base_bays, **dict(base_prop.proposed_bays)}
        inc_crews = {**graph.base_crews, **dict(base_prop.proposed_crews)}
        inc_prio = {**base_prio, **dict(base_prop.proposed_op_priorities or {})}
        inc_art = CandidateArtifact(
            "V20B_Incumbent", "incumbent", inc_assign, inc_bays, inc_crews, inc_prio,
            dict(base_prop.proposed_task_sequences) if base_prop.proposed_task_sequences else None,
            False,
        )
        self._suite._finish_artifact(
            inc_art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr,
            rob_cur_h=rob_cur_h, obs_ops=fresh_ops, fast_surr=fast_surr, compute_details=False,
        )
        if inc_art.eval_res_b2 is None:
            return base_prop
        inc_sc = float(inc_art.ja_unified_score or 1e9)

        imp_map = compute_cross_domain_seed_importance(snapshot, graph, inc_art.eval_res_b2)
        ranked_tids = sorted(graph.rel_tids, key=lambda tid: (-imp_map.get(tid, 0.0), tid))
        k_eff = min(self.top_k, len(ranked_tids))

        chosen_arm = 0
        if self.address_mode == "N1_Address_TopK":
            seed_tasks = ranked_tids[:k_eff]
        elif self.address_mode == "N1_Random_TopK":
            idx_r = self._rng.choice(len(ranked_tids), size=k_eff, replace=False)
            seed_tasks = [ranked_tids[int(i)] for i in idx_r]
        else:
            # N2_Address_RestrictedTS: sample seed rank from Beta posterior restricted to Top-ts_pool_k
            m_arms = min(self.ts_pool_k, len(ranked_tids))
            draws = [float(self._rng.beta(self._ts_alpha[i], self._ts_beta[i])) for i in range(m_arms)]
            chosen_arm = int(np.argmax(draws))
            seed_t = ranked_tids[chosen_arm]
            # Pair chosen high-importance seed with top remaining neighbors up to k_eff
            seed_tasks = [seed_t] + [t for t in ranked_tids[:k_eff] if t != seed_t][: max(0, k_eff - 1)]

        rt, ro = expand_causal_release_closure(snapshot, graph, seed_tasks, [], max_tasks=max(k_eff, 4), max_ops=6)
        if not rt:
            return base_prop

        bays_semi, crews_semi = compute_semicircle_kitting_bays(snapshot, network, clf, lazy_surr, inc_bays, bay_delta)
        _, _, km_prio = _compute_fresh_semicircle_bays(snapshot, network, graph, fast_surr, fresh_ops)
        eff_prio = km_prio or inc_prio
        anc = {tid: aid for tid, aid in inc_assign.items() if tid not in rt}

        cand_list: list[CandidateArtifact] = []
        a_r2, sq_r2 = _regret2_insert_tasks(snapshot, graph, fast_surr, anc, sorted(rt), eff_prio, inc_bays)
        cand_list.append(CandidateArtifact(f"EXP3_{self.address_mode}", "addr_r2", a_r2, dict(inc_bays), dict(inc_crews), dict(eff_prio), sq_r2, False))

        if _cvcr_now_s() + 0.0018 < deadline:
            sub_g = build_masked_subgraph(
                graph, release_tasks=rt, release_ops=ro,
                skel_assign=inc_assign, skel_bays=inc_bays, skel_crews=inc_crews,
            )
            np_f = self._suite._v19_anchor._np_full_net
            ef = np_f.encode_entities(sub_g)
            hf = np_f.a1_a2_heads(ef, sub_g)
            df = decode_v19_with_cached_enc(np_f, ef, hf, sub_g, snapshot)
            cand_list.append(
                CandidateArtifact(
                    f"EXP3_{self.address_mode}", "addr_sub_ahg",
                    dict(df.assign), dict(df.bays), dict(df.crews), dict(df.op_prio), df.task_sequences, False,
                )
            )

        best_art = None
        best_sc = inc_sc
        for art in cand_list:
            if _cvcr_now_s() >= deadline:
                break
            self._suite._finish_artifact(
                art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr,
                rob_cur_h=rob_cur_h, obs_ops=fresh_ops, fast_surr=fast_surr, compute_details=False,
            )
            if art.eval_res_b2 and art.eval_res_b2.valid and float(art.ja_unified_score or 1e9) + 12.0 < best_sc:
                best_sc = float(art.ja_unified_score or 1e9)
                best_art = art

        improved = best_art is not None
        if self.address_mode == "N2_Address_RestrictedTS":
            if improved:
                self._ts_alpha[chosen_arm] += 1.0
            else:
                self._ts_beta[chosen_arm] += 1.0

        if not improved:
            return base_prop

        self.address_improvements += 1
        assert best_art is not None
        has_win = any(getattr(op, "resource_windows", ()) for op in snapshot.operations if op.end_step is None)
        seq_h = int(self.unwin_seq_horizon) if self.unwin_seq_horizon is not None else 0
        inc_unwin = False
        orig_seq = best_art.task_orders
        if seq_h > 0 and not has_win:
            inc_unwin = True
            if isinstance(orig_seq, dict):
                best_art.task_orders = {k: list(v)[:seq_h] for k, v in orig_seq.items() if v}
        try:
            prop_out = best_art.to_proposal(
                policy_id=f"EXP3_{self.address_mode}",
                snapshot=snapshot,
                clf=clf,
                eval_res=best_art.eval_res_b2,
                eval_cnt=int(base_prop.telemetry.get("eval_cnt", 1)) + len(cand_list),
                t0=t0,
                budget_ms=cap_ms,
                fallback_keep=False,
                include_eval_seq_when_unwindowed=inc_unwin,
            )
        finally:
            best_art.task_orders = orig_seq
        prop_out.telemetry["address_mode"] = self.address_mode
        prop_out.telemetry["address_improved"] = True
        return prop_out
