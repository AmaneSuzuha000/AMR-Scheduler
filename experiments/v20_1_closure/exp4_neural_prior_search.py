"""EXP-4 (P1): Neural-Prior Guided Search (ICAPS 2024 Inspired).

Reuses the EXISTING frozen `V19 AHG-RL` NumPy network (`_np_full_net`) with ZERO retraining!
Extracts full calibrated softmax distributions P_theta(agv | task, s_t) and P_theta(bay | op, s_t)
from cached Layer-1 entity encodings to:
  1. Generate top-2/3 high-prior neural joint branches on high-uncertainty (smallest top-1/top-2 margin)
     tasks instead of argmax-only decoding;
  2. Order candidate verification priority via:
        f(x) = J_hat_A(x) + lambda * R(x) - gamma * log P_theta(x | s_t)
     strictly influencing search/branch exploration while keeping external B2/JA evaluation and
     commitment verification 100% unchanged!

Supports controlled 4-way falsification:
  - P0_V18_DetSearch      (deterministic V18-style Regret-2 + PGMPO + CausalCone without V19 neural prior)
  - P1_V20_Bidirectional  (frozen V20.0 argmax-only V19 + V18 bidirectional)
  - P2_NeuralPrior_Guided (V20 + V19 full probability distribution branch expansion & prior-guided ordering)
  - P3_RandomPrior_Ctrl   (identical branch budget driven by uniform random prior control)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.cvcr.snapshot import NativeSnapshot
from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
from amr_sched.network import Network
from amr_sched.rl.causal_repair_cone import FastJAWKittingSurrogate, observable_op_closure
from amr_sched.rl.v19_hetero_constructor import DecodedJointCandidate, NumpyV19HeteroConstructorNet
from amr_sched.rl.v19_hetero_graph import V19HeteroGraph, extract_v19_hetero_graph
from experiments.v20_audit.common import CandidateArtifact
from experiments.v20_audit.proposer_variants import (
    WindowAwareKittingSurrogate,
    _LazyKittingSurrogate,
    _healthy_robots_from_graph,
    derive_base_op_prio,
)
from experiments.v20_audit.v20_b_bidirectional import V20BProposer


def _log_softmax_1d(logits: np.ndarray) -> np.ndarray:
    m = float(np.max(logits))
    shifted = logits - m
    lse = float(np.log(np.sum(np.exp(np.clip(shifted, -50.0, 0.0))) + 1e-12))
    return shifted - lse


def decode_neural_prior_branches(
    np_net: NumpyV19HeteroConstructorNet,
    enc: dict[str, np.ndarray],
    h12_raw: dict[str, np.ndarray],
    graph: V19HeteroGraph,
    snapshot: NativeSnapshot,
    *,
    n_alt_branches: int = 2,
    random_prior: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> list[tuple[DecodedJointCandidate, float, str]]:
    """Decode greedy joint plan + top `n_alt_branches` high-prior alternative joint branches from P_theta.

    Returns list of `(DecodedJointCandidate, total_log_prob, branch_tag)`.
    """
    N_op = len(graph.op_ids)
    N_t = len(graph.rel_tids)
    N_a = len(graph.agv_ids)
    if N_t == 0 or N_a == 0:
        return []

    task_focus_np = h12_raw["task_focus_logits"][0]
    bay_logits_np = np.where(graph.op_bay_mask, h12_raw["bay_logits"][0], -1e9)
    crew_logits_np = np.where(graph.op_crew_mask, h12_raw["crew_logits"][0], -1e9)
    eff_stage = getattr(np_net, "action_stage", "A123")

    bay_indices = np.zeros((N_op,), dtype=np.int64)
    crew_indices = np.zeros((N_op,), dtype=np.int64)
    bays_dict: dict[str, str] = {}
    crews_dict: dict[str, int] = {}
    bay_logp_sum = 0.0
    for oi, oid in enumerate(graph.op_ids):
        if oid in graph.frozen_ops:
            bi = int(graph.op_incumbent_bay_idx[oi])
        else:
            lsm_b = _log_softmax_1d(bay_logits_np[oi]) if bay_logits_np.shape[1] > 0 else np.zeros(1)
            bi = int(np.argmax(bay_logits_np[oi])) if bay_logits_np.shape[1] > 0 else 0
            bay_logp_sum += float(lsm_b[bi])
        bay_indices[oi] = bi
        bays_dict[oid] = graph.bay_ids[bi]
        if graph.op_crew_mask[oi].sum() <= 1:
            ci = int(graph.op_incumbent_crew_idx[oi])
        else:
            ci = int(np.argmax(crew_logits_np[oi])) if crew_logits_np.shape[1] > 0 else 0
        crew_indices[oi] = ci
        crews_dict[oid] = int(graph.crew_ids[ci])

    order = sorted(range(N_t), key=lambda i: (-float(task_focus_np[i]), i))

    def _run_decode(branch_step_override: Optional[tuple[int, int]] = None) -> tuple[DecodedJointCandidate, float, list[tuple[float, int, int]]]:
        assign_dict = dict(graph.fixed_assign)
        agv_indices = np.zeros((N_t,), dtype=np.int64)
        added_load = np.zeros((N_a,), dtype=np.float32)
        added_dur = np.zeros((N_a,), dtype=np.float32)
        logp_tot = bay_logp_sum
        margins: list[tuple[float, int, int]] = []  # (margin_top1_top2, step_pos, second_best_ai)

        for pos, ti in enumerate(order):
            oi = int(graph.task_to_op[ti])
            sel_bi = int(bay_indices[oi]) if oi >= 0 else int(graph.task_incumbent_bay_idx[ti])
            b_tr = float(graph.bay_xy_travel_from_pickup[ti, sel_bi])
            raw_lg = np_net.a3_step_logits(enc, graph, ti, sel_bi, added_load, added_dur)[0]
            if random_prior and rng is not None:
                valid_m = graph.task_agv_mask[ti]
                lg = np.where(valid_m, rng.normal(0.0, 1.0, size=raw_lg.shape), -1e9)
            else:
                lg = raw_lg
            lsm = _log_softmax_1d(lg)
            ranked_a = np.argsort(-lg)
            best_ai = int(ranked_a[0])
            if len(ranked_a) >= 2 and lg[ranked_a[1]] > -1e8:
                sec_ai = int(ranked_a[1])
                margins.append((float(lsm[best_ai] - lsm[sec_ai]), pos, sec_ai))
            if branch_step_override is not None and pos == branch_step_override[0]:
                ai = int(branch_step_override[1])
            else:
                ai = best_ai
            logp_tot += float(lsm[ai])
            agv_indices[ti] = ai
            assign_dict[graph.rel_tids[ti]] = graph.agv_ids[ai]
            added_load[ai] += 1.0
            added_dur[ai] += b_tr + float(graph.task_agv_feats[ti, ai, 0])

        seq_scores_np = np_net.a4_seq_scores(enc, agv_indices)[0] if eff_stage == "A1234" else np.zeros((N_t,), dtype=np.float32)
        task_sequences: Optional[dict[str, list[str]]] = None
        if eff_stage == "A1234":
            by_agv_rel: dict[str, list[tuple[float, str]]] = {}
            for ti, tid in enumerate(graph.rel_tids):
                by_agv_rel.setdefault(assign_dict[tid], []).append((-float(seq_scores_np[ti]), tid))
            task_sequences = {aid: [tid for _, tid in sorted(ps)] for aid, ps in by_agv_rel.items() if ps}
            op_best_rank: dict[str, float] = {}
            for ti, _ in enumerate(graph.rel_tids):
                oi = int(graph.task_to_op[ti])
                if oi >= 0:
                    oid = graph.op_ids[oi]
                    op_best_rank[oid] = min(op_best_rank.get(oid, 1e9), -float(seq_scores_np[ti]))
            op_prio = {
                op.op_id: int(round(max(snapshot.tick, op.due_step - op.proc_steps) + 8.0 * op_best_rank.get(op.op_id, 0.0)))
                for op in snapshot.operations
            }
        else:
            op_prio = {op.op_id: max(int(snapshot.tick), int(op.due_step) - int(op.proc_steps)) for op in snapshot.operations}

        dec = DecodedJointCandidate(
            assign=assign_dict, bays=dict(bays_dict), crews=dict(crews_dict),
            op_prio=op_prio, task_sequences=task_sequences,
            bay_indices=bay_indices, crew_indices=crew_indices,
            agv_indices=agv_indices, seq_scores=seq_scores_np,
        )
        return dec, float(logp_tot), margins

    dec0, lp0, margins = _run_decode(None)
    branches: list[tuple[DecodedJointCandidate, float, str]] = [(dec0, lp0, "prior_top1")]
    # Branch on the `n_alt_branches` most ambiguous tasks (smallest log-prob gap between top-1 & top-2 AGVs)
    margins.sort(key=lambda x: x[0])
    for idx_b, (_, step_pos, sec_ai) in enumerate(margins[:n_alt_branches], 1):
        dec_b, lp_b, _ = _run_decode((step_pos, sec_ai))
        if dec_b.assign != dec0.assign:
            branches.append((dec_b, lp_b, f"prior_branch_{idx_b}"))
    return branches


class NeuralPriorGuidedProposer(V20BProposer):
    """EXP-4 Proposer evaluating Neural-Prior Guided Search vs V20.0 Bidirectional & Random-Prior Control."""

    def __init__(
        self,
        *,
        prior_mode: str = "P2_NeuralPrior_Guided",  # P1_V20_Bidirectional | P2_NeuralPrior_Guided | P3_RandomPrior_Ctrl
        n_alt_branches: int = 2,
        gamma_prior_order: float = 12.0,
        **v20b_kwargs: Any,
    ) -> None:
        super().__init__(**v20b_kwargs)
        self.prior_mode = str(prior_mode)
        self.n_alt_branches = int(n_alt_branches)
        self.gamma_prior_order = float(gamma_prior_order)
        self._rng = np.random.default_rng(int(self.seed) + 202404)
        self.prior_branch_wins: int = 0
        self.prior_calls: int = 0

    def __call__(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        budget_ms: float = 35.0,
    ) -> CandidateProposal:
        if self.prior_mode == "P1_V20_Bidirectional":
            return super().__call__(snapshot, network, budget_ms=budget_ms)

        t0 = _cvcr_now_s()
        cap_ms = float(self.hard_budget_ms) if self.hard_budget_ms > 0 else float(budget_ms)
        soft_ms = (
            min(float(self.soft_deadline_ms), max(3.5, cap_ms * 0.72))
            if self.soft_deadline_ms is not None
            else max(3.5, cap_ms * 0.68)
        )
        orig_soft = self.soft_deadline_ms
        self.soft_deadline_ms = max(4.0, soft_ms - 2.4)
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
        if not healthy or not graph.rel_tids:
            return base_prop

        self.prior_calls += 1
        np_full = self._suite._v19_anchor._np_full_net
        enc = np_full.encode_entities(graph)
        h12 = np_full.a1_a2_heads(enc, graph)
        branches = decode_neural_prior_branches(
            np_full, enc, h12, graph, snapshot,
            n_alt_branches=self.n_alt_branches,
            random_prior=(self.prior_mode == "P3_RandomPrior_Ctrl"),
            rng=self._rng,
        )
        # Order alternative branches by prior score f_order = -gamma * log P_theta (highest prior first!)
        alt_branches = sorted(branches[1:], key=lambda x: -self.gamma_prior_order * x[1])
        if not alt_branches:
            return base_prop

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
        inc_sc = float(inc_art.ja_unified_score or 1e9)
        best_art = None
        best_sc = inc_sc
        for dec_b, lp_b, tag_b in alt_branches:
            if _cvcr_now_s() >= deadline:
                break
            art = CandidateArtifact(
                f"EXP4_{self.prior_mode}", tag_b,
                dict(dec_b.assign), dict(dec_b.bays), dict(dec_b.crews), dict(dec_b.op_prio),
                dict(dec_b.task_sequences) if dec_b.task_sequences else None, True,
            )
            self._suite._finish_artifact(
                art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr,
                rob_cur_h=rob_cur_h, obs_ops=fresh_ops, fast_surr=fast_surr, compute_details=False,
            )
            # External evaluation strictly uses unbiased B2/JA objective (prior only guided search expansion!)
            if art.eval_res_b2 and art.eval_res_b2.valid and float(art.ja_unified_score or 1e9) + 12.0 < best_sc:
                best_sc = float(art.ja_unified_score or 1e9)
                best_art = art

        if best_art is None:
            return base_prop

        self.prior_branch_wins += 1
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
                policy_id=f"EXP4_{self.prior_mode}",
                snapshot=snapshot,
                clf=clf,
                eval_res=best_art.eval_res_b2,
                eval_cnt=int(base_prop.telemetry.get("eval_cnt", 1)) + len(alt_branches),
                t0=t0,
                budget_ms=cap_ms,
                fallback_keep=False,
                include_eval_seq_when_unwindowed=inc_unwin,
            )
        finally:
            best_art.task_orders = orig_seq
        prop_out.telemetry["prior_mode"] = self.prior_mode
        prop_out.telemetry["prior_branch_won"] = True
        return prop_out
