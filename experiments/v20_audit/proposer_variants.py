"""Non-invasive candidate experts (E0-E9), Phase-0 repair ladder, and E0-E9 unified prototype."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from amr_sched.baselines.policy_registry import (
    _V19_RL_CKPT,
    _V19_STAGED_RL_CKPT,
    load_internal_cvcr_policy,
)
from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.policy import CandidateProposal, LGCVCRPolicy, _LazyKittingSurrogate
from amr_sched.cvcr.snapshot import NativeSnapshot, op_predecessors
from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
from amr_sched.cvcr.v17_window_protection import (
    plan_bridge_order_control,
    select_coupled_op_pair,
)
from amr_sched.cvcr.v18_frontier_synthesis import (
    WindowAwareKittingSurrogate,
    _mit26_precedence_closure_destroy,
    _surrogate_polish_assign,
    compute_semicircle_kitting_bays,
)
from amr_sched.network import Network
from amr_sched.rl.causal_repair_cone import (
    FastJAWKittingSurrogate,
    VOIBudgetedOperatorSearch,
    _compute_fresh_semicircle_bays,
    _regret2_insert_tasks,
    compute_candidate_ja_components,
    generate_causal_cone_candidates,
    observable_op_closure,
    project_zero_churn_candidate,
    score_candidate_ja_unified,
)
from amr_sched.rl.v19_hetero_constructor import (
    DecodedJointCandidate,
    V19AdaptiveCouplingProposer,
    _build_core_sched_proposal,
    compute_candidate_amr_footprint,
    compute_dynamic_coupling_features,
    decode_v19_candidate,
)
from amr_sched.rl.v19_hetero_graph import V19HeteroGraph, extract_v19_hetero_graph
from experiments.v20_audit.common import (
    CandidateArtifact,
    candidate_action_signature,
)


def derive_base_op_prio(snapshot: NativeSnapshot) -> dict[str, int]:
    """Build baseline Slack-first operation priority map from snapshot."""
    return {op.op_id: int(op.due_step) - int(op.proc_steps) for op in snapshot.operations}


def compute_extended_amr_footprint(
    snapshot: NativeSnapshot,
    graph: V19HeteroGraph,
    base_robot_seq: dict[str, list[str]],
    cand_assign: dict[str, str],
    cand_bays: dict[str, str],
    cand_crews: dict[str, int],
    cand_op_prio: dict[str, int],
    base_op_prio: dict[str, int],
    cand_seq: Optional[dict[str, list[str]]],
) -> dict[str, Any]:
    """Compare legacy footprint (`cand_seq=None`) vs true footprint including `op_priority` and queue diffs."""
    legacy = compute_candidate_amr_footprint(
        snapshot,
        graph.base_assign,
        graph.base_bays,
        base_robot_seq,
        cand_assign,
        cand_bays,
        None,
    )
    full = compute_candidate_amr_footprint(
        snapshot,
        graph.base_assign,
        graph.base_bays,
        base_robot_seq,
        cand_assign,
        cand_bays,
        cand_seq,
    )
    changed_amrs = set(full.get("changed_amrs", set()))
    changed_prios = {
        oid
        for oid, pr in cand_op_prio.items()
        if int(pr) != int(base_op_prio.get(oid, pr))
    }
    changed_crews = {
        oid
        for oid, cr in cand_crews.items()
        if int(cr) != int(graph.base_crews.get(oid, cr))
    }
    if changed_prios or changed_crews:
        affected_ops = changed_prios | changed_crews
        for t in snapshot.tasks:
            if t.status.lower() == "completed":
                continue
            if t.kit_op_id in affected_ops:
                for aid in (graph.base_assign.get(t.task_id), cand_assign.get(t.task_id)):
                    if aid:
                        changed_amrs.add(aid)
    legacy_set = set(legacy.get("changed_amrs", set()))
    return {
        "legacy_changed_amrs": sorted(legacy_set),
        "full_changed_amrs": sorted(changed_amrs),
        "legacy_count": len(legacy_set),
        "full_count": len(changed_amrs),
        "missed_queue_or_prio_amrs": sorted(changed_amrs - legacy_set),
        "changed_op_prios_count": len(changed_prios),
        "changed_crews_count": len(changed_crews),
    }


def _healthy_robots_from_graph(graph: V19HeteroGraph) -> list[str]:
    return [graph.agv_ids[i] for i in graph.healthy_agv_indices]


def _default_task_order_from_prio(
    snapshot: NativeSnapshot,
    graph: V19HeteroGraph,
    assign: dict[str, str],
    op_prio: dict[str, int],
) -> dict[str, list[str]]:
    """Deterministic SLA/op_prio task sequence per healthy AGV when no explicit queue order exists."""
    t_by_id = {t.task_id: t for t in snapshot.tasks}
    out: dict[str, list[str]] = {}
    for aid in _healthy_robots_from_graph(graph):
        tids = [tid for tid in graph.rel_tids if assign.get(tid) == aid]
        if tids:
            tids.sort(
                key=lambda tid: (
                    0 if t_by_id[tid].status.lower() in ("picked", "loading", "to_dropoff", "unloading") else 1,
                    int(op_prio.get(t_by_id[tid].kit_op_id or "", 99999)),
                    int(t_by_id[tid].due_step),
                    int(t_by_id[tid].release_step),
                    tid,
                )
            )
            out[aid] = tids
    return out


def _score_ja_unified(
    snapshot: NativeSnapshot,
    network: Network,
    graph: V19HeteroGraph,
    ev_res: EvalResult,
    assign: dict[str, str],
    bays: dict[str, str],
    crews: dict[str, int],
    orders: Optional[dict[str, list[str]]],
    rob_cur_h: dict[str, str],
    sim_horizon: int,
    lambda_change: float = 15.0,
    obs_ops: Optional[set[str]] = None,
    fast_surr: Optional[FastJAWKittingSurrogate] = None,
) -> tuple[float, dict[str, float]]:
    eff_ops = obs_ops if obs_ops is not None else observable_op_closure(snapshot)
    sc_u, _ = score_candidate_ja_unified(
        snapshot,
        network,
        graph,
        ev_res,
        assign,
        bays,
        crews,
        eff_ops,
        fast_surr,
        lambda_change=lambda_change,
        sim_horizon=sim_horizon,
    )
    comps = compute_candidate_ja_components(
        snapshot,
        graph,
        ev_res,
        assign,
        bays,
        crews,
        orders,
        rob_cur_h,
        lambda_change=lambda_change,
        sim_horizon=sim_horizon,
    )
    return float(sc_u), dict(comps)


@dataclass
class Phase0Config:
    """Toggle configuration for non-invasive `Phase0_Restored` repair ladder (`P0.0..P0.3`)."""

    name: str = "P0.0"
    fix_a4_seq: bool = False
    commit_unwindowed_seq: bool = False
    unlock_full_pool_and_bays: bool = False
    attach_cliff_locks: bool = False
    wire_voi_and_scale_budget: bool = False
    use_seq_diff_b2: bool = False
    hard_budget_ms: Optional[float] = None


class Phase0RestoredProposer(V19AdaptiveCouplingProposer):
    """Non-invasive configurable repair proposer isolating F1-F5 fixes on top of frozen v1.2.1."""

    def __init__(self, cfg: Phase0Config, **kwargs: Any) -> None:
        super().__init__(_V19_RL_CKPT, _V19_STAGED_RL_CKPT, **kwargs)
        self.cfg = cfg

    def __call__(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        budget_ms: float = 35.0,
    ) -> CandidateProposal:
        t0 = _cvcr_now_s()
        raw_budget = float(self.cfg.hard_budget_ms) if self.cfg.hard_budget_ms is not None else float(budget_ms)
        if not self.cfg.wire_voi_and_scale_budget:
            eff_budget_ms = min(29.0, max(17.5, raw_budget * 0.85))
            deadline = t0 + eff_budget_ms * 0.78 / 1000.0
        else:
            # Reserve ~28% of budget for post-propose loaded_corridor_repair + validate_and_commit so pooled P95 <= B
            eff_budget_ms = max(6.0, raw_budget * 0.70)
            deadline = t0 + eff_budget_ms / 1000.0

        self._ensure_nets()
        self._ensure_infer_nets()
        net_full = self._np_full_net if self._effective_backend() == "numpy" else self._full_net
        net_staged = self._np_staged_net if self._effective_backend() == "numpy" else self._staged_net

        clf = classify_commitments(snapshot)
        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        ev = CVCREvaluator(
            network,
            snapshot,
            mode=self.evaluator_mode,
            alpha_makespan=self.alpha_makespan,
            lambda_change=self.lambda_change,
            use_rust=self.use_rust,
            seq_diff_b2=bool(self.cfg.use_seq_diff_b2),
        )
        base_prio = derive_base_op_prio(snapshot)
        keep_eval = ev.evaluate(
            graph.base_assign,
            op_priority=base_prio,
            op_bay_overrides=graph.base_bays,
            op_crew_overrides=graph.base_crews,
        )
        keep_seq = {k: list(v) for k, v in keep_eval.robot_order_seq.items() if v} if keep_eval.robot_order_seq else None
        ev.set_incumbent(
            graph.base_assign,
            keep_eval,
            op_priority=base_prio,
            op_bay_overrides=graph.base_bays,
            task_order_override=keep_seq,
        )

        rob_cur_h = {
            r.robot_id: r.cur_task
            for r in snapshot.robots
            if r.cur_task and r.robot_id not in clf.disturbed_robots
        }

        def _score_res(res: EvalResult, asn: dict[str, str], bs: dict[str, str], cs: dict[str, int], sq: Optional[dict[str, list[str]]]) -> float:
            sc, _ = _score_ja_unified(
                snapshot, network, graph, res, asn, bs, cs, sq, rob_cur_h, ev.sim_horizon, self.lambda_change, fresh_ops, fast_surr
            )
            return sc

        fast_surr = FastJAWKittingSurrogate(network, snapshot, ev, graph, lambda_change=self.lambda_change)
        fresh_ops = observable_op_closure(snapshot)
        bays_semi, crews_semi, km_prio = _compute_fresh_semicircle_bays(
            snapshot, network, graph, fast_surr, fresh_ops
        )
        keep_score = _score_res(keep_eval, graph.base_assign, graph.base_bays, graph.base_crews, keep_seq)
        best_eval = keep_eval
        best_score = keep_score
        best_assign = dict(graph.base_assign)
        best_bays = dict(graph.base_bays)
        best_crews = dict(graph.base_crews)
        best_prio = dict(base_prio)
        best_seq: Optional[dict[str, list[str]]] = keep_seq if self.cfg.fix_a4_seq else None
        best_tag = "KEEP"
        best_live_repairs: Optional[dict[str, dict[str, Any]]] = None
        best_bottleneck_locks: Optional[dict[str, list[tuple[str, int, int]]]] = None
        eval_cnt = 1

        phi = compute_dynamic_coupling_features(graph)
        x = np.concatenate([[1.0], phi.astype(np.float64)])
        A_inv_0 = np.linalg.inv(self._A[0])
        A_inv_1 = np.linalg.inv(self._A[1])
        theta_0 = A_inv_0 @ self._b[0]
        theta_1 = A_inv_1 @ self._b[1]
        ucb_0 = float(x @ theta_0 + 0.35 * math.sqrt(max(1e-8, float(x @ A_inv_0 @ x))))
        ucb_1 = float(x @ theta_1 + 0.35 * math.sqrt(max(1e-8, float(x @ A_inv_1 @ x))))
        arm_pick = 0 if ucb_0 >= ucb_1 else 1
        arm_best_scores: dict[int, float] = {0: keep_score, 1: keep_score}
        t_by_id = {t.task_id: t for t in snapshot.tasks}
        pending_or_dist = {
            tid
            for tid in graph.rel_tids
            if t_by_id[tid].status.lower() == "pending"
            or (
                t_by_id[tid].assigned_agv is not None
                and t_by_id[tid].assigned_agv in clf.disturbed_robots
            )
            or (
                clf.disturbed_robots
                and any(
                    op.kit_task_ids and tid in op.kit_task_ids
                    for op in snapshot.operations
                    if any(
                        t_by_id.get(kt) and t_by_id[kt].assigned_agv in clf.disturbed_robots
                        for kt in op.kit_task_ids
                    )
                )
            )
        }
        base_robot_seq = {k: list(v) for k, v in keep_eval.robot_order_seq.items()} if keep_eval.robot_order_seq else {}

        # Generate neural candidates matching V19AdaptiveCouplingProposer
        rng = np.random.default_rng(int(self.seed) * 100003 + int(snapshot.tick) + 17)
        nn_cands: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]], bool, int]] = []

        def _push_nn(tag: str, cand: DecodedJointCandidate, is_full: bool, arm_id: int) -> None:
            asn = dict(cand.assign)
            bays = dict(cand.bays)
            crews = dict(cand.crews)
            op_p = dict(cand.op_prio)
            seq = cand.task_sequences if self.cfg.fix_a4_seq else None
            # When unlock_full_pool_and_bays=False, emulate v1.2.1 bay forcing on >= 4 AMRs when workshop bays lock
            if not self.cfg.unlock_full_pool_and_bays and len(network.nodes) >= 30 and not is_full:
                bays = dict(bays_semi)
                crews = dict(crews_semi)
            nn_cands.append((tag, asn, bays, crews, op_p, seq, is_full, arm_id))

        if self.enable_neural_candidates and len(graph.rel_tids) > 0:
            c_full_g = decode_v19_candidate(net_full, graph, snapshot, greedy=True, np_rng=rng)
            _push_nn("full_greedy", c_full_g, True, 0)
            for s_i in range(min(2, self.sample_candidates)):
                c_s = decode_v19_candidate(net_full, graph, snapshot, greedy=False, np_rng=rng, tau=0.65)
                _push_nn(f"full_s{s_i}", c_s, True, 0)

            c_st_g = decode_v19_candidate(net_staged, graph, snapshot, greedy=True, np_rng=rng)
            _push_nn("staged_greedy", c_st_g, False, 1)
            if self.enable_zero_churn_projection:
                zc_a, zc_b, zc_c, zc_p, zc_sq = project_zero_churn_candidate(
                    snapshot,
                    graph,
                    c_st_g.assign,
                    c_st_g.bays,
                    c_st_g.crews,
                    c_st_g.op_prio,
                    c_st_g.task_sequences if self.cfg.fix_a4_seq else None,
                    fresh_ops,
                    pending_or_dist,
                    rob_cur_h,
                )
                nn_cands.append(("staged_zc", zc_a, zc_b, zc_c, zc_p, zc_sq if self.cfg.fix_a4_seq else None, False, 1))

        # Surrogate shortlist or blind [:3] truncation
        seen_sigs: set[str] = {
            candidate_action_signature(graph.base_assign, graph.base_bays, graph.base_crews, base_prio, keep_seq)
        }
        if self.cfg.unlock_full_pool_and_bays:
            scored_nn: list[tuple[float, Any]] = []
            for item in nn_cands:
                tag_s, asn_s, bs_s, cs_s, op_s, sq_s, is_f, arm_id = item
                surr_val = fast_surr.score(asn_s, op_s, bs_s, cs_s, sq_s)
                scored_nn.append((float(surr_val), item))
            scored_nn.sort(key=lambda x: x[0])
            shortlist_nn = [item for _, item in scored_nn[: max(int(self.max_rust_verifications), len(scored_nn))]]
        else:
            shortlist_nn = nn_cands[: min(3, len(nn_cands))]

        neural_pool_for_cone: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]], bool]] = []
        verified_states_for_voi: list[Any] = []

        def _eval_and_update(
            tag_s: str,
            asn_s: dict[str, str],
            bs_s: dict[str, str],
            cs_s: dict[str, int],
            op_s: dict[str, int],
            sq_s: Optional[dict[str, list[str]]],
            is_f: bool,
            *,
            arm_id: Optional[int] = None,
            live_rep: Optional[dict[str, dict[str, Any]]] = None,
            locks: Optional[dict[str, list[tuple[str, int, int]]]] = None,
            force: bool = False,
        ) -> None:
            nonlocal best_eval, best_score, best_assign, best_bays, best_crews, best_prio, best_seq, best_tag
            nonlocal best_live_repairs, best_bottleneck_locks, eval_cnt
            if not force and _cvcr_now_s() >= deadline:
                return
            sig = candidate_action_signature(asn_s, bs_s, cs_s, op_s, sq_s, live_rep, locks)
            if sig in seen_sigs and not force:
                return
            seen_sigs.add(sig)
            t_ord_arg = sq_s if (self.cfg.fix_a4_seq and sq_s) else None
            res = ev.evaluate(
                asn_s,
                op_priority=op_s,
                op_bay_overrides=bs_s,
                op_crew_overrides=cs_s,
                task_order_override=t_ord_arg,
            )
            eval_cnt += 1
            eff_sq = {k: list(v) for k, v in res.robot_order_seq.items() if v} if res.robot_order_seq else t_ord_arg
            sc = _score_res(res, asn_s, bs_s, cs_s, eff_sq)
            if arm_id is not None and res.valid:
                arm_best_scores[arm_id] = min(arm_best_scores[arm_id], sc)
            if res.valid:
                verified_states_for_voi.append(
                    (tag_s, res, sc, dict(asn_s), dict(bs_s), dict(cs_s), dict(op_s), eff_sq)
                )
            if res.valid and sc + 1e-4 < best_score:
                best_score = sc
                best_eval = res
                best_assign = dict(asn_s)
                best_bays = dict(bs_s)
                best_crews = dict(cs_s)
                best_prio = dict(op_s)
                best_seq = eff_sq if self.cfg.fix_a4_seq else None
                best_tag = tag_s
                best_live_repairs = copy_optional_dict(live_rep)
                best_bottleneck_locks = copy_optional_dict(locks)
                ev.set_incumbent(
                    best_assign,
                    best_eval,
                    op_priority=best_prio,
                    op_bay_overrides=best_bays,
                    task_order_override=best_seq,
                )

        for tag_s, asn_s, bs_s, cs_s, op_s, sq_s, is_f, arm_id in shortlist_nn:
            if _cvcr_now_s() >= deadline:
                break
            _eval_and_update(tag_s, asn_s, bs_s, cs_s, op_s, sq_s, is_f, arm_id=arm_id)
            neural_pool_for_cone.append((tag_s, asn_s, bs_s, cs_s, op_s, sq_s, is_f))

        # Causal Cone repair stage
        if self.enable_causal_cone_repair and (_cvcr_now_s() + 0.002 < deadline) and len(graph.rel_tids) > 0:
            cone_b_lim = 1.6 if self.cfg.wire_voi_and_scale_budget else 0.7
            ccands = generate_causal_cone_candidates(
                snapshot,
                network,
                graph,
                ev,
                fast_surr,
                keep_eval,
                base_prio,
                neural_pool_for_cone,
                fresh_ops,
                pending_or_dist,
                budget_ms_limit=cone_b_lim,
            )
            if self.cfg.unlock_full_pool_and_bays:
                ccands_scored = [
                    (float(fast_surr.score(c[1], c[4], c[2], c[3], c[5])), c)
                    for c in ccands
                ]
                ccands_scored.sort(key=lambda x: x[0])
                cc_iter = [c for _, c in ccands_scored[:8]]
            else:
                cc_iter = ccands[:5]
            for tag_c, asn_c, bs_c, cs_c, op_c, sq_c, is_f_c in cc_iter:
                if _cvcr_now_s() >= deadline:
                    break
                _eval_and_update(tag_c, asn_c, bs_c, cs_c, op_c, sq_c, is_f_c)

        # Cliff Bridge order control (with live_repairs + bottleneck_locks preservation in P0.2+)
        if (_cvcr_now_s() + 0.0025 < deadline) and best_eval.valid:
            pair = select_coupled_op_pair(network, snapshot, best_eval, best_assign, best_prio)
            if pair is not None:
                op_a, op_b, _meta = pair
                for ld, yd in ((op_a, op_b), (op_b, op_a)):
                    if _cvcr_now_s() >= deadline:
                        break
                    bp_p = plan_bridge_order_control(
                        network,
                        snapshot,
                        ev,
                        best_assign,
                        best_prio,
                        ld,
                        yd,
                        base_bays=best_bays,
                        base_crews=best_crews,
                    )
                    if bp_p.eligible and bp_p.eval_res is not None and bp_p.eval_res.valid:
                        lr_arg = bp_p.live_repairs if self.cfg.attach_cliff_locks else None
                        lk_arg = bp_p.bottleneck_locks if self.cfg.attach_cliff_locks else None
                        _eval_and_update(
                            f"cliff_bridge_{ld}_{yd}",
                            best_assign,
                            best_bays,
                            best_crews,
                            bp_p.op_priority,
                            best_seq,
                            False,
                            live_rep=lr_arg,
                            locks=lk_arg,
                        )

        # P0.3: Wire existing VOIBudgetedOperatorSearch if >= 6ms margin remains
        if self.cfg.wire_voi_and_scale_budget and (_cvcr_now_s() + 0.006 < deadline) and len(graph.rel_tids) >= 2:
            voi = VOIBudgetedOperatorSearch(neighborhood_mode="causal_cone", seed=int(self.seed))
            if not verified_states_for_voi:
                verified_states_for_voi.append(
                    ("KEEP", keep_eval, keep_score, dict(graph.base_assign), dict(graph.base_bays), dict(graph.base_crews), dict(base_prio), keep_seq)
                )
            seen_tuples: set[tuple[Any, ...]] = set()
            v_eval, v_sc, v_asn, v_bs, v_cs, v_op, v_sq, v_tag, v_cnt = voi.refine(
                ev=ev,
                snapshot=snapshot,
                graph=graph,
                seed_states=verified_states_for_voi,
                score_fn=lambda r, a, b, c, sq: _score_res(r, a, b, c, sq),
                seen_sigs=seen_tuples,
                deadline_s=max(_cvcr_now_s() + 0.001, deadline - 0.003),
                max_evals=4,
                hysteresis_margin=0.5,
                keep_eval=keep_eval,
            )
            eval_cnt += int(v_cnt)
            if v_eval.valid and float(v_sc) + 1e-4 < best_score:
                best_score = float(v_sc)
                best_eval = v_eval
                best_assign = dict(v_asn)
                best_bays = dict(v_bs)
                best_crews = dict(v_cs)
                best_prio = dict(v_op)
                best_seq = v_sq if self.cfg.fix_a4_seq else None
                best_tag = f"voi_{v_tag}"
                best_live_repairs = None
                best_bottleneck_locks = None

        # Hysteresis check vs KEEP
        changed_ops_cnt = sum(
            1
            for oid in graph.op_ids
            if best_bays.get(oid, graph.base_bays.get(oid)) != graph.base_bays.get(oid)
            or int(best_crews.get(oid, graph.base_crews.get(oid, 0))) != int(graph.base_crews.get(oid, 0))
        )
        changed_tasks_cnt = sum(
            1
            for tid in graph.rel_tids
            if graph.base_assign.get(tid) and best_assign.get(tid) != graph.base_assign.get(tid)
        )
        eff_hys = float(self.hysteresis_margin) if (changed_ops_cnt > 0 or changed_tasks_cnt > 0) else 0.5
        fallback_keep = not (best_eval.valid and best_score + eff_hys < keep_score)
        if fallback_keep:
            best_eval = keep_eval
            best_score = keep_score
            best_assign = dict(graph.base_assign)
            best_bays = dict(graph.base_bays)
            best_crews = dict(graph.base_crews)
            best_prio = dict(base_prio)
            best_seq = keep_seq if self.cfg.fix_a4_seq else None
            best_tag = "KEEP"
            best_live_repairs = None
            best_bottleneck_locks = None

        # LinUCB update (F5: misattributed in P0.0/P0.1, cleanly arm-attributed in P0.2+)
        if self.cfg.unlock_full_pool_and_bays:
            reward_0 = max(-5.0, min(5.0, (keep_score - arm_best_scores[0]) / 250.0))
            reward_1 = max(-5.0, min(5.0, (keep_score - arm_best_scores[1]) / 250.0))
            self._A[0] += np.outer(x, x)
            self._b[0] += reward_0 * x
            self._A[1] += np.outer(x, x)
            self._b[1] += reward_1 * x
        else:
            reward = max(-5.0, min(5.0, (keep_score - best_score) / 250.0))
            self._A[arm_pick] += np.outer(x, x)
            self._b[arm_pick] += reward * x

        prop = _build_core_sched_proposal(
            policy_id=f"{self.policy_id}_{self.cfg.name}",
            snapshot=snapshot,
            clf=clf,
            best_assign=best_assign,
            best_bays=best_bays,
            best_crews=best_crews,
            best_eval=best_eval,
            op_prio=best_prio,
            eval_cnt=eval_cnt,
            t0=t0,
            budget_ms=raw_budget,
            fallback_keep=fallback_keep,
            task_sequences=best_seq,
            predicted_score=best_score,
            include_eval_seq_when_unwindowed=bool(self.cfg.commit_unwindowed_seq),
        )
        if not fallback_keep and best_live_repairs:
            prop.proposed_live_repairs = best_live_repairs
        if not fallback_keep and best_bottleneck_locks:
            prop.proposed_bottleneck_locks = best_bottleneck_locks
        prop.telemetry.update(
            {
                "phase0_cfg": self.cfg.name,
                "best_tag": best_tag,
                "selected_arm": "full" if arm_pick == 0 else "staged",
                "fallback_keep": fallback_keep,
                "keep_obj": round(keep_score, 4),
                "best_obj": round(best_score, 4),
                "eff_budget_ms": round(eff_budget_ms, 2),
                "checkpoint_sha256": self._checkpoint_sha256,
                "inference_backend": self._effective_backend(),
            }
        )
        return prop


def copy_optional_dict(d: Optional[dict[Any, Any]]) -> Optional[dict[Any, Any]]:
    if not d:
        return None
    return {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in d.items()}


EXPERT_IDS: tuple[str, ...] = (
    "E0_KEEP",
    "E1_V18_Regret2",
    "E2_AHG_Full",
    "E3_AHG_Staged",
    "E4_V18_PGMPO",
    "E5_V18_MIT_DyMAB_LNS",
    "E6_V18_Polish",
    "E7_CausalCone_VOI",
    "E8_Cliff_Bridge",
    "E9_Local_CPSAT",
)


class ExpertSuiteEvaluator:
    """Evaluates all 10 candidate experts `E0-E9` uniformly on frozen `(snapshot, network)` states."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        self._v19_anchor = V19AdaptiveCouplingProposer(_V19_RL_CKPT, _V19_STAGED_RL_CKPT, backend="numpy", use_rust=True, seed=self.seed)
        self._v19_anchor._ensure_nets()
        self._v19_anchor._ensure_infer_nets()
        self._v18_pol = load_internal_cvcr_policy("V18_EA_JOINT", allow_missing_pgmpo=False)

    def _finish_artifact(
        self,
        art: CandidateArtifact,
        *,
        snapshot: NativeSnapshot,
        graph: V19HeteroGraph,
        ev_b2: CVCREvaluator,
        w_surr: WindowAwareKittingSurrogate,
        rob_cur_h: dict[str, str],
        obs_ops: Optional[set[str]] = None,
        fast_surr: Optional[FastJAWKittingSurrogate] = None,
        compute_details: bool = True,
    ) -> CandidateArtifact:
        t_eval0 = _cvcr_now_s()
        t_ord = art.task_orders if art.task_orders else None
        res = ev_b2.evaluate(
            art.assign,
            op_priority=art.op_prio,
            op_bay_overrides=art.bays,
            op_crew_overrides=art.crews,
            task_order_override=t_ord,
        )
        art.b2_ms = (_cvcr_now_s() - t_eval0) * 1000.0
        art.eval_res_b2 = res
        eff_seq = {k: list(v) for k, v in res.robot_order_seq.items() if v} if res.robot_order_seq else t_ord
        if art.task_orders is None and eff_seq:
            art.task_orders = eff_seq
        eff_ops = obs_ops if obs_ops is not None else observable_op_closure(snapshot)
        sc_u, _ = score_candidate_ja_unified(
            snapshot,
            ev_b2.network,
            graph,
            res,
            art.assign,
            art.bays,
            art.crews,
            eff_ops,
            fast_surr,
            lambda_change=15.0,
            sim_horizon=ev_b2.sim_horizon,
        )
        art.ja_unified_score = float(sc_u)
        if compute_details:
            comps = compute_candidate_ja_components(
                snapshot,
                graph,
                res,
                art.assign,
                art.bays,
                art.crews,
                eff_seq,
                rob_cur_h,
                lambda_change=15.0,
                sim_horizon=ev_b2.sim_horizon,
            )
            art.ja_proxy_details = dict(comps)
            try:
                art.w_surr_score = float(w_surr.score(art.assign, art.op_prio, art.bays, art.crews))
            except Exception:
                art.w_surr_score = None
        art.ja_proxy_details["sim_horizon"] = int(ev_b2.sim_horizon)
        art.ensure_sig()
        return art

    def _pick_best_cand(
        self,
        cands: list[CandidateArtifact],
        *,
        snapshot: NativeSnapshot,
        graph: V19HeteroGraph,
        ev_b2: CVCREvaluator,
        w_surr: WindowAwareKittingSurrogate,
        rob_cur_h: dict[str, str],
        fallback: CandidateArtifact,
    ) -> CandidateArtifact:
        if not cands:
            return fallback
        evaluated: list[CandidateArtifact] = []
        for c in cands:
            evaluated.append(
                self._finish_artifact(
                    c,
                    snapshot=snapshot,
                    graph=graph,
                    ev_b2=ev_b2,
                    w_surr=w_surr,
                    rob_cur_h=rob_cur_h,
                )
            )
        evaluated.sort(
            key=lambda x: (
                0 if (x.eval_res_b2 is not None and x.eval_res_b2.valid) else 1,
                float(x.ja_unified_score if x.ja_unified_score is not None else 1e12),
            )
        )
        return evaluated[0]

    def evaluate_all_experts(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        expert_ids: Optional[tuple[str, ...]] = None,
    ) -> dict[str, CandidateArtifact]:
        """Generate and B2-evaluate best candidate per expert in `EXPERT_IDS`."""
        requested = set(expert_ids or EXPERT_IDS)
        clf = classify_commitments(snapshot)
        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        healthy = _healthy_robots_from_graph(graph)
        base_prio = derive_base_op_prio(snapshot)
        ev_b2 = CVCREvaluator(
            network,
            snapshot,
            mode="B2_SEEDED",
            alpha_makespan=3.0,
            lambda_change=15.0,
            use_rust=True,
            seq_diff_b2=True,
        )
        dd_sig = compute_detour_delta_signals(network, snapshot)
        bay_delta = dd_sig["bay_detour_delta"] if dd_sig else {}
        lazy_surr = _LazyKittingSurrogate(network, snapshot, base_prio, alpha_makespan=3.0, lambda_change=15.0, bay_detour_delta=bay_delta)
        w_surr = WindowAwareKittingSurrogate(lazy_surr, ev_b2, snapshot)
        fast_surr = FastJAWKittingSurrogate(network, snapshot, ev_b2, graph, lambda_change=15.0)
        rob_cur_h = {
            r.robot_id: r.cur_task
            for r in snapshot.robots
            if r.cur_task and r.robot_id not in clf.disturbed_robots
        }

        # E0_KEEP always computed as anchor & incumbent
        t0 = _cvcr_now_s()
        keep_art = CandidateArtifact(
            expert_id="E0_KEEP",
            tag="keep",
            assign=dict(graph.base_assign),
            bays=dict(graph.base_bays),
            crews=dict(graph.base_crews),
            op_prio=dict(base_prio),
            task_orders=None,
            is_full=False,
        )
        self._finish_artifact(
            keep_art,
            snapshot=snapshot,
            graph=graph,
            ev_b2=ev_b2,
            w_surr=w_surr,
            rob_cur_h=rob_cur_h,
        )
        keep_art.gen_ms = (_cvcr_now_s() - t0) * 1000.0
        keep_eval = keep_art.eval_res_b2
        assert keep_eval is not None
        keep_seq = {k: list(v) for k, v in keep_eval.robot_order_seq.items() if v} if keep_eval.robot_order_seq else None
        ev_b2.set_incumbent(
            graph.base_assign,
            keep_eval,
            op_priority=base_prio,
            op_bay_overrides=graph.base_bays,
            task_order_override=keep_seq,
        )
        results: dict[str, CandidateArtifact] = {"E0_KEEP": keep_art}

        fresh_ops = observable_op_closure(snapshot)
        bays_semi, crews_semi = compute_semicircle_kitting_bays(
            snapshot, network, clf, lazy_surr, graph.base_bays, bay_delta, spread_empty_kits=False
        )
        # Reconstruct km_prio via fresh bays if compute_semicircle returned bays & crews cleanly
        _, _, km_prio_fast = _compute_fresh_semicircle_bays(snapshot, network, graph, fast_surr, fresh_ops)
        eff_km_prio = km_prio_fast or base_prio
        t_by_id = {t.task_id: t for t in snapshot.tasks}
        pending_or_dist = {
            tid
            for tid in graph.rel_tids
            if t_by_id[tid].status.lower() == "pending"
            or (t_by_id[tid].assigned_agv is not None and t_by_id[tid].assigned_agv in clf.disturbed_robots)
        }
        rng = np.random.default_rng(self.seed * 100003 + int(snapshot.tick) + 17)

        # E1_V18_Regret2
        if "E1_V18_Regret2" in requested:
            t_s = _cvcr_now_s()
            c_e1: list[CandidateArtifact] = []
            if healthy and graph.rel_tids:
                a_global, seq_global = _regret2_insert_tasks(
                    snapshot, graph, fast_surr, dict(graph.fixed_assign), list(graph.rel_tids), eff_km_prio, bays_semi
                )
                c_e1.append(
                    CandidateArtifact(
                        expert_id="E1_V18_Regret2",
                        tag="regret2_global",
                        assign=a_global,
                        bays=dict(bays_semi),
                        crews=dict(crews_semi),
                        op_prio=dict(eff_km_prio),
                        task_orders=seq_global,
                        is_full=True,
                        gen_ms=(_cvcr_now_s() - t_s) * 1000.0,
                    )
                )
                if pending_or_dist:
                    anchor_zc = {tid: aid for tid, aid in graph.base_assign.items() if tid not in pending_or_dist}
                    a_zc, seq_zc = _regret2_insert_tasks(
                        snapshot, graph, fast_surr, anchor_zc, sorted(pending_or_dist), eff_km_prio, bays_semi
                    )
                    c_e1.append(
                        CandidateArtifact(
                            expert_id="E1_V18_Regret2",
                            tag="regret2_zc",
                            assign=a_zc,
                            bays=dict(bays_semi),
                            crews=dict(crews_semi),
                            op_prio=dict(eff_km_prio),
                            task_orders=seq_zc,
                            is_full=False,
                            gen_ms=(_cvcr_now_s() - t_s) * 1000.0,
                        )
                    )
            results["E1_V18_Regret2"] = self._pick_best_cand(
                c_e1, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr, rob_cur_h=rob_cur_h, fallback=keep_art
            )

        # E2_AHG_Full (A1..A4 with task_sequences preserved)
        neural_pool_artifacts: list[CandidateArtifact] = []
        if "E2_AHG_Full" in requested or "E7_CausalCone_VOI" in requested:
            t_s = _cvcr_now_s()
            c_e2: list[CandidateArtifact] = []
            if healthy and graph.rel_tids:
                net_f = self._v19_anchor._np_full_net
                dec_g = decode_v19_candidate(net_f, graph, snapshot, greedy=True, np_rng=rng)
                cands_dec = [("ahg_full_greedy", dec_g)]
                for si in range(3):
                    cands_dec.append(
                        (f"ahg_full_s{si}", decode_v19_candidate(net_f, graph, snapshot, greedy=False, np_rng=rng, tau=0.65))
                    )
                scored_e2 = sorted(
                    ((fast_surr.score(dc.assign, dc.op_prio, dc.bays, dc.crews, dc.task_sequences), tg, dc) for tg, dc in cands_dec),
                    key=lambda x: x[0],
                )
                gen_ms_e2 = (_cvcr_now_s() - t_s) * 1000.0
                for _, tg, dc in scored_e2[:3]:
                    art = CandidateArtifact(
                        expert_id="E2_AHG_Full",
                        tag=tg,
                        assign=dict(dc.assign),
                        bays=dict(dc.bays),
                        crews=dict(dc.crews),
                        op_prio=dict(dc.op_prio),
                        task_orders=dict(dc.task_sequences) if dc.task_sequences else None,
                        is_full=True,
                        gen_ms=gen_ms_e2,
                    )
                    c_e2.append(art)
            best_e2 = self._pick_best_cand(
                c_e2, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr, rob_cur_h=rob_cur_h, fallback=keep_art
            )
            if "E2_AHG_Full" in requested:
                results["E2_AHG_Full"] = best_e2
            neural_pool_artifacts.extend(c_e2)

        # E3_AHG_Staged + ZC projection
        if "E3_AHG_Staged" in requested or "E7_CausalCone_VOI" in requested:
            t_s = _cvcr_now_s()
            c_e3: list[CandidateArtifact] = []
            if healthy and graph.rel_tids:
                net_st = self._v19_anchor._np_staged_net
                dc_g = decode_v19_candidate(net_st, graph, snapshot, greedy=True, np_rng=rng)
                zc_a, zc_b, zc_c, zc_p, zc_sq = project_zero_churn_candidate(
                    snapshot, graph, dc_g.assign, dc_g.bays, dc_g.crews, dc_g.op_prio, dc_g.task_sequences, fresh_ops, pending_or_dist, rob_cur_h
                )
                gen_ms_e3 = (_cvcr_now_s() - t_s) * 1000.0
                c_e3.append(
                    CandidateArtifact(
                        expert_id="E3_AHG_Staged",
                        tag="ahg_staged_zc",
                        assign=zc_a,
                        bays=zc_b,
                        crews=zc_c,
                        op_prio=zc_p,
                        task_orders=zc_sq,
                        is_full=False,
                        gen_ms=gen_ms_e3,
                    )
                )
                c_e3.append(
                    CandidateArtifact(
                        expert_id="E3_AHG_Staged",
                        tag="ahg_staged_raw",
                        assign=dict(dc_g.assign),
                        bays=dict(dc_g.bays),
                        crews=dict(dc_g.crews),
                        op_prio=dict(dc_g.op_prio),
                        task_orders=dict(dc_g.task_sequences) if dc_g.task_sequences else None,
                        is_full=False,
                        gen_ms=gen_ms_e3,
                    )
                )
            best_e3 = self._pick_best_cand(
                c_e3, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr, rob_cur_h=rob_cur_h, fallback=keep_art
            )
            if "E3_AHG_Staged" in requested:
                results["E3_AHG_Staged"] = best_e3
            neural_pool_artifacts.extend(c_e3)

        # E4_V18_PGMPO
        if "E4_V18_PGMPO" in requested:
            t_s = _cvcr_now_s()
            c_e4: list[CandidateArtifact] = []
            if healthy and graph.rel_tids and self._v18_pol.pgmpo is not None:
                raw_asns = self._v18_pol.pgmpo.generate_assignments(
                    snapshot, network, clf, healthy, graph.fixed_assign, bays_semi, lazy_surr
                )
                scored_p = sorted(
                    ((float(w_surr.score(asn, eff_km_prio, bays_semi, crews_semi)), idx_p, asn) for idx_p, asn in enumerate(raw_asns)),
                    key=lambda x: x[0],
                )
                gen_ms_e4 = (_cvcr_now_s() - t_s) * 1000.0
                for _, idx_p, asn in scored_p[:3]:
                    c_e4.append(
                        CandidateArtifact(
                            expert_id="E4_V18_PGMPO",
                            tag=f"pgmpo_{idx_p}",
                            assign=dict(asn),
                            bays=dict(bays_semi),
                            crews=dict(crews_semi),
                            op_prio=dict(eff_km_prio),
                            task_orders=_default_task_order_from_prio(snapshot, graph, asn, eff_km_prio),
                            is_full=True,
                            gen_ms=gen_ms_e4,
                        )
                    )
            results["E4_V18_PGMPO"] = self._pick_best_cand(
                c_e4, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr, rob_cur_h=rob_cur_h, fallback=keep_art
            )

        # E5_V18_MIT_DyMAB_LNS
        if "E5_V18_MIT_DyMAB_LNS" in requested:
            t_s = _cvcr_now_s()
            c_e5: list[CandidateArtifact] = []
            if len(healthy) >= 2 and graph.rel_tids:
                scored_lns: list[tuple[float, str, dict[str, str], dict[str, list[str]]]] = []
                for m_idx in range(3):
                    d_tids = _mit26_precedence_closure_destroy(snapshot, graph.rel_tids, keep_eval, graph.base_assign, m_idx)
                    d_set = (set(d_tids) | set(pending_or_dist)) - set(graph.fixed_assign.keys())
                    if not d_set:
                        continue
                    anchor = {tid: aid for tid, aid in graph.base_assign.items() if tid not in d_set}
                    asn_l, sq_l = _regret2_insert_tasks(
                        snapshot, graph, fast_surr, anchor, sorted(d_set), eff_km_prio, bays_semi
                    )
                    scored_lns.append((float(w_surr.score(asn_l, eff_km_prio, bays_semi, crews_semi)), f"mit_m{m_idx}", asn_l, sq_l))
                # DyMAB worst-AMR + tardiest-task destroy neighborhood
                worst_tids = sorted(
                    [tid for tid in graph.rel_tids if tid not in graph.fixed_assign],
                    key=lambda tid: (int(t_by_id[tid].due_step), tid),
                )[: max(2, len(graph.rel_tids) // 3)]
                d_dy = set(worst_tids) | set(pending_or_dist)
                if d_dy:
                    anc_d = {tid: aid for tid, aid in graph.base_assign.items() if tid not in d_dy}
                    asn_d, sq_d = _regret2_insert_tasks(
                        snapshot, graph, fast_surr, anc_d, sorted(d_dy), eff_km_prio, bays_semi
                    )
                    scored_lns.append((float(w_surr.score(asn_d, eff_km_prio, bays_semi, crews_semi)), "dymab_tard", asn_d, sq_d))
                scored_lns.sort(key=lambda x: x[0])
                gen_ms_e5 = (_cvcr_now_s() - t_s) * 1000.0
                for _, tg, asn, sq in scored_lns[:3]:
                    c_e5.append(
                        CandidateArtifact(
                            expert_id="E5_V18_MIT_DyMAB_LNS",
                            tag=tg,
                            assign=asn,
                            bays=dict(bays_semi),
                            crews=dict(crews_semi),
                            op_prio=dict(eff_km_prio),
                            task_orders=sq,
                            is_full=True,
                            gen_ms=gen_ms_e5,
                        )
                    )
            results["E5_V18_MIT_DyMAB_LNS"] = self._pick_best_cand(
                c_e5, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr, rob_cur_h=rob_cur_h, fallback=keep_art
            )

        # E6_V18_Polish
        if "E6_V18_V18_Polish" in requested or "E6_V18_Polish" in requested:
            t_s = _cvcr_now_s()
            c_e6: list[CandidateArtifact] = []
            if len(healthy) >= 2 and graph.rel_tids:
                pol_base_a, _ = _surrogate_polish_assign(
                    w_surr, graph.rel_tids, healthy, dict(graph.base_assign), bays_semi, crews_semi, eff_km_prio,
                    max_passes=2, max_focus_tasks=10, do_swaps=True,
                )
                gen_ms_e6 = (_cvcr_now_s() - t_s) * 1000.0
                c_e6.append(
                    CandidateArtifact(
                        expert_id="E6_V18_Polish",
                        tag="polish_swap_semi",
                        assign=pol_base_a,
                        bays=dict(bays_semi),
                        crews=dict(crews_semi),
                        op_prio=dict(eff_km_prio),
                        task_orders=_default_task_order_from_prio(snapshot, graph, pol_base_a, eff_km_prio),
                        is_full=False,
                        gen_ms=gen_ms_e6,
                    )
                )
                # Also polish starting from E1 Regret2 if present
                if "E1_V18_Regret2" in results and results["E1_V18_Regret2"].eval_res_b2 is not None:
                    e1 = results["E1_V18_Regret2"]
                    pol_e1_a, _ = _surrogate_polish_assign(
                        w_surr, graph.rel_tids, healthy, dict(e1.assign), e1.bays, e1.crews, e1.op_prio,
                        max_passes=2, max_focus_tasks=10, do_swaps=True,
                    )
                    c_e6.append(
                        CandidateArtifact(
                            expert_id="E6_V18_Polish",
                            tag="polish_swap_e1",
                            assign=pol_e1_a,
                            bays=dict(e1.bays),
                            crews=dict(e1.crews),
                            op_prio=dict(e1.op_prio),
                            task_orders=_default_task_order_from_prio(snapshot, graph, pol_e1_a, e1.op_prio),
                            is_full=True,
                            gen_ms=(_cvcr_now_s() - t_s) * 1000.0,
                        )
                    )
            results["E6_V18_Polish"] = self._pick_best_cand(
                c_e6, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr, rob_cur_h=rob_cur_h, fallback=keep_art
            )

        # E7_CausalCone_VOI
        if "E7_CausalCone_VOI" in requested:
            t_s = _cvcr_now_s()
            c_e7: list[CandidateArtifact] = []
            if graph.rel_tids:
                nn_tuples = [
                    (a.tag, a.assign, a.bays, a.crews, a.op_prio, a.task_orders, a.is_full)
                    for a in neural_pool_artifacts[:3]
                ]
                ccands = generate_causal_cone_candidates(
                    snapshot, network, graph, ev_b2, fast_surr, keep_eval, base_prio, nn_tuples, fresh_ops, pending_or_dist, budget_ms_limit=2.0
                )
                seed_states_v: list[Any] = [
                    ("KEEP", keep_eval, float(keep_art.ja_unified_score or 1e9), dict(graph.base_assign), dict(graph.base_bays), dict(graph.base_crews), dict(base_prio), keep_seq)
                ]
                for tg, asn_c, bs_c, cs_c, op_c, sq_c, is_fc in ccands[:4]:
                    art_c = self._finish_artifact(
                        CandidateArtifact(
                            expert_id="E7_CausalCone_VOI",
                            tag=tg,
                            assign=asn_c,
                            bays=bs_c,
                            crews=cs_c,
                            op_prio=op_c,
                            task_orders=sq_c,
                            is_full=is_fc,
                            gen_ms=(_cvcr_now_s() - t_s) * 1000.0,
                        ),
                        snapshot=snapshot,
                        graph=graph,
                        ev_b2=ev_b2,
                        w_surr=w_surr,
                        rob_cur_h=rob_cur_h,
                    )
                    c_e7.append(art_c)
                    if art_c.eval_res_b2 and art_c.eval_res_b2.valid:
                        seed_states_v.append(
                            (tg, art_c.eval_res_b2, float(art_c.ja_unified_score or 1e9), asn_c, bs_c, cs_c, op_c, sq_c)
                        )
                voi = VOIBudgetedOperatorSearch(neighborhood_mode="causal_cone", seed=self.seed)
                v_eval, v_sc, v_asn, v_bs, v_cs, v_op, v_sq, v_tag, _ = voi.refine(
                    ev=ev_b2,
                    snapshot=snapshot,
                    graph=graph,
                    seed_states=seed_states_v,
                    score_fn=lambda r, a, b, c, sq: _score_ja_unified(snapshot, network, graph, r, a, b, c, sq, rob_cur_h, ev_b2.sim_horizon, 15.0, fresh_ops, fast_surr)[0],
                    seen_sigs=set(),
                    deadline_s=_cvcr_now_s() + 0.008,
                    max_evals=6,
                    hysteresis_margin=0.5,
                    keep_eval=keep_eval,
                )
                if v_eval.valid:
                    v_art = CandidateArtifact(
                        expert_id="E7_CausalCone_VOI",
                        tag=f"voi_{v_tag}",
                        assign=v_asn,
                        bays=v_bs,
                        crews=v_cs,
                        op_prio=v_op,
                        task_orders=v_sq,
                        is_full=False,
                        gen_ms=(_cvcr_now_s() - t_s) * 1000.0,
                    )
                    self._finish_artifact(v_art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr, rob_cur_h=rob_cur_h)
                    c_e7.append(v_art)
            c_e7.sort(key=lambda x: (0 if (x.eval_res_b2 and x.eval_res_b2.valid) else 1, float(x.ja_unified_score or 1e12)))
            results["E7_CausalCone_VOI"] = c_e7[0] if c_e7 else keep_art

        # E8_Cliff_Bridge
        if "E8_Cliff_Bridge" in requested:
            t_s = _cvcr_now_s()
            c_e8: list[CandidateArtifact] = []
            pair = select_coupled_op_pair(network, snapshot, keep_eval, graph.base_assign, base_prio)
            if pair is not None:
                op_a, op_b, _ = pair
                for ld, yd in ((op_a, op_b), (op_b, op_a)):
                    bp_p = plan_bridge_order_control(
                        network,
                        snapshot,
                        ev_b2,
                        graph.base_assign,
                        base_prio,
                        ld,
                        yd,
                        base_bays=graph.base_bays,
                        base_crews=graph.base_crews,
                    )
                    if bp_p.eligible and bp_p.eval_res is not None and bp_p.eval_res.valid:
                        art8 = CandidateArtifact(
                            expert_id="E8_Cliff_Bridge",
                            tag=f"bridge_{ld}_{yd}",
                            assign=dict(graph.base_assign),
                            bays=dict(graph.base_bays),
                            crews=dict(graph.base_crews),
                            op_prio=dict(bp_p.op_priority),
                            task_orders={k: list(v) for k, v in bp_p.eval_res.robot_order_seq.items() if v} or None,
                            is_full=False,
                            live_repairs=copy_optional_dict(bp_p.live_repairs),
                            bottleneck_locks=copy_optional_dict(bp_p.bottleneck_locks),
                            gen_ms=(_cvcr_now_s() - t_s) * 1000.0,
                        )
                        c_e8.append(art8)
            results["E8_Cliff_Bridge"] = self._pick_best_cand(
                c_e8, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr, rob_cur_h=rob_cur_h, fallback=keep_art
            )

        # E9_Local_CPSAT (focal bottleneck CP-SAT over AGV, Bay, Crew, Tool, Order)
        if "E9_Local_CPSAT" in requested:
            t_s = _cvcr_now_s()
            c_e9: list[CandidateArtifact] = []
            if healthy and graph.rel_tids:
                cp_owner = LGCVCRPolicy("V18_EA_JOINT", network, use_rust=True)
                # Start from keep AND from best_e1 if available so CP-SAT covers order/bay/AGV focal windows
                seeds_cp: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], EvalResult]] = [
                    ("cpsat_keep", dict(graph.base_assign), dict(graph.base_bays), dict(graph.base_crews), dict(base_prio), keep_eval)
                ]
                e1_cand = results.get("E1_V18_Regret2")
                if e1_cand is not None and e1_cand.eval_res_b2 is not None and e1_cand.eval_res_b2.valid:
                    seeds_cp.append(
                        ("cpsat_e1", dict(e1_cand.assign), dict(e1_cand.bays), dict(e1_cand.crews), dict(e1_cand.op_prio), e1_cand.eval_res_b2)
                    )
                for c_tag, c_asn, c_bs, c_cs, c_op, c_ev in seeds_cp:
                    a_cp, b_cp, cr_cp, pr_cp, ord_cp, res_cp, _ = cp_owner._search_local_cpsat_window(
                        snapshot,
                        clf,
                        ev_b2,
                        c_asn,
                        c_bs,
                        c_cs,
                        c_ev,
                        healthy,
                        c_op,
                        _cvcr_now_s() + 0.015,
                        lazy_surr,
                        allow_order_only=True,
                        min_improve=0.1,
                        solve_time_cap_s=0.012,
                    )
                    if res_cp.valid:
                        c_e9.append(
                            CandidateArtifact(
                                expert_id="E9_Local_CPSAT",
                                tag=c_tag,
                                assign=dict(a_cp),
                                bays=dict(b_cp),
                                crews=dict(cr_cp),
                                op_prio=dict(pr_cp),
                                task_orders={k: list(v) for k, v in (ord_cp or res_cp.robot_order_seq).items() if v} or None,
                                is_full=False,
                                gen_ms=(_cvcr_now_s() - t_s) * 1000.0,
                            )
                        )
            results["E9_Local_CPSAT"] = self._pick_best_cand(
                c_e9, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr, rob_cur_h=rob_cur_h, fallback=keep_art
            )

        return results


class UnifiedE0E9PrototypeProposer:
    """Budget-adaptive E0-E9 union prototype (`S5` / `V20-A`) for equal-budget benchmarking & Gate A."""

    def __init__(
        self,
        *,
        hard_budget_ms: float = 35.0,
        seed: int = 0,
        enable_e4: bool = True,
        enable_e5: bool = True,
        enable_e6: bool = True,
        enable_e7: bool = False,
        enable_e9: bool = True,
        commit_unwin_seq: bool = True,
        critic_weights_path: Optional[Path] = None,
        critic_beta: float = 0.0,
        tier2_ws_only: bool = False,
        enable_r2_global_ws: bool = True,
        include_staged_g: bool | str = False,
        r2_zc_ws_mode: str = "always",
        soft_deadline_ms: Optional[float] = None,
        unwin_seq_horizon: Optional[int] = None,
    ) -> None:
        self.hard_budget_ms = float(hard_budget_ms)
        self.soft_deadline_ms = None if soft_deadline_ms is None else float(soft_deadline_ms)
        self.unwin_seq_horizon = None if unwin_seq_horizon is None else int(unwin_seq_horizon)
        self.seed = int(seed)
        self.enable_e4 = bool(enable_e4)
        self.enable_e5 = bool(enable_e5)
        self.enable_e6 = bool(enable_e6)
        self.enable_e7 = bool(enable_e7)
        self.enable_e9 = bool(enable_e9)
        self.commit_unwin_seq = bool(commit_unwin_seq)
        self.critic_beta = float(critic_beta)
        self.tier2_ws_only = bool(tier2_ws_only)
        self.enable_r2_global_ws = bool(enable_r2_global_ws)
        self.include_staged_g: bool | str = include_staged_g if isinstance(include_staged_g, str) else bool(include_staged_g)
        self.r2_zc_ws_mode = str(r2_zc_ws_mode)
        self._critic = None
        if critic_weights_path is not None and float(critic_beta) > 0.0:
            from experiments.v20_audit.critic_model import NumpyValueCritic

            self._critic = NumpyValueCritic(Path(critic_weights_path))
        self._suite = ExpertSuiteEvaluator(seed=self.seed)

    def __call__(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        budget_ms: float = 35.0,
    ) -> CandidateProposal:
        t0 = _cvcr_now_s()
        cap_ms = float(self.hard_budget_ms) if self.hard_budget_ms > 0 else float(budget_ms)
        soft_ms = (
            float(self.soft_deadline_ms)
            if getattr(self, "soft_deadline_ms", None) is not None
            else max(3.5, cap_ms * 0.68)
        )
        deadline = t0 + max(3.5, soft_ms) / 1000.0

        clf = classify_commitments(snapshot)
        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        healthy = _healthy_robots_from_graph(graph)
        base_prio = derive_base_op_prio(snapshot)
        ev_b2 = CVCREvaluator(
            network, snapshot, mode="B2_SEEDED", alpha_makespan=3.0, lambda_change=15.0, use_rust=True, seq_diff_b2=True
        )
        dd_sig = compute_detour_delta_signals(network, snapshot)
        bay_delta = dd_sig["bay_detour_delta"] if dd_sig else {}
        lazy_surr = _LazyKittingSurrogate(network, snapshot, base_prio, alpha_makespan=3.0, lambda_change=15.0, bay_detour_delta=bay_delta)
        w_surr = WindowAwareKittingSurrogate(lazy_surr, ev_b2, snapshot)
        fast_surr = FastJAWKittingSurrogate(network, snapshot, ev_b2, graph, lambda_change=15.0)
        fresh_ops = observable_op_closure(snapshot)
        rob_cur_h = {
            r.robot_id: r.cur_task for r in snapshot.robots if r.cur_task and r.robot_id not in clf.disturbed_robots
        }

        keep_art = CandidateArtifact(
            expert_id="E0_KEEP",
            tag="keep",
            assign=dict(graph.base_assign),
            bays=dict(graph.base_bays),
            crews=dict(graph.base_crews),
            op_prio=dict(base_prio),
            task_orders=None,
            is_full=False,
        )
        self._suite._finish_artifact(
            keep_art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr, rob_cur_h=rob_cur_h,
            obs_ops=fresh_ops, fast_surr=fast_surr, compute_details=False,
        )
        best = keep_art
        keep_sc = float(keep_art.ja_unified_score or 1e9)
        keep_eval = keep_art.eval_res_b2
        assert keep_eval is not None
        keep_seq = {k: list(v) for k, v in keep_eval.robot_order_seq.items() if v} if keep_eval.robot_order_seq else None
        ev_b2.set_incumbent(graph.base_assign, keep_eval, op_priority=base_prio, op_bay_overrides=graph.base_bays, task_order_override=keep_seq)

        keep_stats = None
        keep_critic_raw = 0.0
        sim_h_critic = int(keep_art.ja_proxy_details.get("sim_horizon", max(600, int(snapshot.tick) + 300)))
        if self._critic is not None and self.critic_beta > 0.0 and keep_eval.valid:
            from experiments.v20_audit.critic_model import build_keep_baseline_stats, extract_critic_features

            keep_stats = build_keep_baseline_stats(snapshot, graph, keep_eval, keep_sc, rob_cur_h, sim_h_critic)
            fv_keep = extract_critic_features(
                snapshot, network, graph, keep_eval, keep_sc,
                keep_art.assign, keep_art.bays, keep_art.crews, keep_art.task_orders,
                keep_stats, rob_cur_h, sim_h_critic, is_keep_cand=True,
            )
            keep_critic_raw = self._critic.predict_delta_keep(fv_keep)
        keep_art.predicted_score = 0.0

        def _score_cand_delta(art: CandidateArtifact) -> float:
            ja_delta = float(art.ja_unified_score or 1e9) - keep_sc
            if self._critic is None or self.critic_beta <= 0.0 or keep_stats is None or art.eval_res_b2 is None:
                return ja_delta
            from experiments.v20_audit.critic_model import extract_critic_features

            fv_c = extract_critic_features(
                snapshot, network, graph, art.eval_res_b2, float(art.ja_unified_score or keep_sc),
                art.assign, art.bays, art.crews, art.task_orders,
                keep_stats, rob_cur_h, sim_h_critic, is_keep_cand=False,
            )
            c_delta = self._critic.predict_delta_keep(fv_c) - keep_critic_raw
            return float(ja_delta + self.critic_beta * c_delta)

        # Tier 1 fast experts (always executed first while within deadline): E2, E3, E1
        bays_semi, crews_semi = compute_semicircle_kitting_bays(snapshot, network, clf, lazy_surr, graph.base_bays, bay_delta)
        _, _, km_prio = _compute_fresh_semicircle_bays(snapshot, network, graph, fast_surr, fresh_ops)
        eff_km_prio = km_prio or base_prio
        t_by_id = {t.task_id: t for t in snapshot.tasks}
        is_ws_net = len(network.nodes) >= 30
        pending_or_dist = {
            tid
            for tid in graph.rel_tids
            if t_by_id[tid].status.lower() == "pending"
            or (t_by_id[tid].assigned_agv is not None and t_by_id[tid].assigned_agv in clf.disturbed_robots)
            or (
                clf.disturbed_robots
                and any(
                    op.kit_task_ids and tid in op.kit_task_ids
                    for op in snapshot.operations
                    if any(
                        t_by_id.get(kt) and t_by_id[kt].assigned_agv in clf.disturbed_robots
                        for kt in op.kit_task_ids
                    )
                )
            )
        }

        eval_cnt = 1
        seen_sigs: set[str] = {keep_art.ensure_sig()}
        nn_eval_pool: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]], bool]] = []

        def _consider(art: CandidateArtifact) -> None:
            nonlocal best, eval_cnt
            sig = art.ensure_sig()
            if sig in seen_sigs or _cvcr_now_s() >= deadline:
                return
            seen_sigs.add(sig)
            self._suite._finish_artifact(
                art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr, rob_cur_h=rob_cur_h,
                obs_ops=fresh_ops, fast_surr=fast_surr, compute_details=False,
            )
            eval_cnt += 1
            if art.eval_res_b2 and art.eval_res_b2.valid:
                nn_eval_pool.append((art.tag, art.assign, art.bays, art.crews, art.op_prio, art.task_orders, art.is_full))
                sc_new = _score_cand_delta(art)
                art.predicted_score = sc_new
                sc_best = float(best.predicted_score if best.predicted_score is not None else 0.0)
                if sc_new + 1e-4 < sc_best:
                    best = art
                    ev_b2.set_incumbent(
                        best.assign, best.eval_res_b2, op_priority=best.op_prio, op_bay_overrides=best.bays, task_order_override=best.task_orders
                    )

        rng = np.random.default_rng(self.seed * 100003 + int(snapshot.tick) + 17)
        if healthy and graph.rel_tids:
            # E1 regret2: full global re-assign on compact 4-Topo; ZC-anchored on >=30-node Workshop unless enabled
            if (not is_ws_net) or self.enable_r2_global_ws:
                a_g, sq_g = _regret2_insert_tasks(snapshot, graph, fast_surr, dict(graph.fixed_assign), list(graph.rel_tids), eff_km_prio, bays_semi)
                _consider(CandidateArtifact("E1_V18_Regret2", "r2_global", a_g, dict(bays_semi), dict(crews_semi), dict(eff_km_prio), sq_g, True))
            if pending_or_dist and _cvcr_now_s() < deadline:
                has_win = any(getattr(op, "resource_windows", ()) for op in snapshot.operations if not getattr(op, "completed", False))
                run_r2_zc = (not is_ws_net) or (
                    self.r2_zc_ws_mode == "always"
                    or (self.r2_zc_ws_mode == "windows_or_invalid" and (has_win or not keep_eval.valid))
                )
                if run_r2_zc:
                    anc = {tid: aid for tid, aid in graph.base_assign.items() if tid not in pending_or_dist}
                    use_bays_zc = dict(bays_semi) if (not is_ws_net or has_win or not keep_eval.valid) else dict(graph.base_bays)
                    use_crews_zc = dict(crews_semi) if (not is_ws_net or has_win or not keep_eval.valid) else dict(graph.base_crews)
                    use_prio_zc = dict(eff_km_prio) if (not is_ws_net or has_win or not keep_eval.valid) else dict(base_prio)
                    a_z, sq_z = _regret2_insert_tasks(snapshot, graph, fast_surr, anc, sorted(pending_or_dist), use_prio_zc, use_bays_zc)
                    _consider(CandidateArtifact("E1_V18_Regret2", "r2_zc", a_z, use_bays_zc, use_crews_zc, use_prio_zc, sq_z, False))

            # E2 AHG-Full & E3 Staged
            if _cvcr_now_s() + 0.002 < deadline:
                dec_f = decode_v19_candidate(self._suite._v19_anchor._np_full_net, graph, snapshot, greedy=True, np_rng=rng)
                _consider(CandidateArtifact("E2_AHG_Full", "ahg_full_g", dict(dec_f.assign), dict(dec_f.bays), dict(dec_f.crews), dict(dec_f.op_prio), dict(dec_f.task_sequences) if dec_f.task_sequences else None, True))
            if _cvcr_now_s() + 0.002 < deadline:
                dec_s = decode_v19_candidate(self._suite._v19_anchor._np_staged_net, graph, snapshot, greedy=True, np_rng=rng)
                staged_g_flag = (
                    (not keep_eval.valid)
                    if str(self.include_staged_g) == "on_invalid"
                    else bool(self.include_staged_g)
                )
                if staged_g_flag:
                    _consider(CandidateArtifact("E3_AHG_Staged", "ahg_staged_g", dict(dec_s.assign), dict(dec_s.bays), dict(dec_s.crews), dict(dec_s.op_prio), dict(dec_s.task_sequences) if dec_s.task_sequences else None, False))
                zc_a, zc_b, zc_c, zc_p, zc_sq = project_zero_churn_candidate(
                    snapshot, graph, dec_s.assign, dec_s.bays, dec_s.crews, dec_s.op_prio, dec_s.task_sequences, fresh_ops, pending_or_dist, rob_cur_h
                )
                _consider(CandidateArtifact("E3_AHG_Staged", "ahg_staged_zc", zc_a, zc_b, zc_c, zc_p, zc_sq, False))

        # Tier 2 when headroom >= 4ms: E4 PGMPO, E5 MIT-LNS, E6 Polish, E7 Cone
        allow_heavy_tier2 = (not self.tier2_ws_only) or is_ws_net
        if allow_heavy_tier2 and self.enable_e4 and (_cvcr_now_s() + 0.004 < deadline) and healthy and graph.rel_tids and self._suite._v18_pol.pgmpo is not None:
            asns = self._suite._v18_pol.pgmpo.generate_assignments(snapshot, network, clf, healthy, graph.fixed_assign, bays_semi, lazy_surr)
            if asns:
                top_p = min(asns, key=lambda a: float(w_surr.score(a, eff_km_prio, bays_semi, crews_semi)))
                _consider(CandidateArtifact("E4_V18_PGMPO", "pgmpo_top", dict(top_p), dict(bays_semi), dict(crews_semi), dict(eff_km_prio), _default_task_order_from_prio(snapshot, graph, top_p, eff_km_prio), True))

        if allow_heavy_tier2 and self.enable_e5 and (_cvcr_now_s() + 0.004 < deadline) and len(healthy) >= 2 and graph.rel_tids:
            d_tids = _mit26_precedence_closure_destroy(snapshot, graph.rel_tids, keep_eval, graph.base_assign, 0)
            d_set = (set(d_tids) | set(pending_or_dist)) - set(graph.fixed_assign.keys())
            if d_set:
                anc_m = {tid: aid for tid, aid in graph.base_assign.items() if tid not in d_set}
                a_m, sq_m = _regret2_insert_tasks(snapshot, graph, fast_surr, anc_m, sorted(d_set), eff_km_prio, bays_semi)
                _consider(CandidateArtifact("E5_V18_MIT_DyMAB_LNS", "mit_m0", a_m, dict(bays_semi), dict(crews_semi), dict(eff_km_prio), sq_m, True))

        if allow_heavy_tier2 and self.enable_e6 and (_cvcr_now_s() + 0.004 < deadline) and len(healthy) >= 2 and graph.rel_tids:
            pol_a, _ = _surrogate_polish_assign(w_surr, graph.rel_tids, healthy, dict(best.assign), dict(best.bays), dict(best.crews), dict(best.op_prio), max_passes=1, max_focus_tasks=8, do_swaps=True)
            _consider(CandidateArtifact("E6_V18_Polish", "polish_best", pol_a, dict(best.bays), dict(best.crews), dict(best.op_prio), _default_task_order_from_prio(snapshot, graph, pol_a, best.op_prio), False))

        if self.enable_e7 and (_cvcr_now_s() + 0.003 < deadline) and graph.rel_tids:
            nn_t = nn_eval_pool[:4] if nn_eval_pool else [(best.tag, best.assign, best.bays, best.crews, best.op_prio, best.task_orders, best.is_full)]
            for tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc in generate_causal_cone_candidates(
                snapshot, network, graph, ev_b2, fast_surr, keep_eval, base_prio, nn_t, fresh_ops, pending_or_dist, budget_ms_limit=1.5
            )[:4]:
                _consider(CandidateArtifact("E7_CausalCone_VOI", tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc))

        # Tier 3 when headroom >= 8ms: Focal CP-SAT around incumbent best
        if allow_heavy_tier2 and self.enable_e9 and (_cvcr_now_s() + 0.008 < deadline) and healthy and graph.rel_tids and best.eval_res_b2 and best.eval_res_b2.valid:
            cp_owner = LGCVCRPolicy("V18_EA_JOINT", network, use_rust=True)
            rem_s = max(0.002, deadline - _cvcr_now_s() - 0.003)
            a_cp, b_cp, cr_cp, pr_cp, ord_cp, res_cp, n_cp = cp_owner._search_local_cpsat_window(
                snapshot,
                clf,
                ev_b2,
                dict(best.assign),
                dict(best.bays),
                dict(best.crews),
                best.eval_res_b2,
                healthy,
                dict(best.op_prio),
                deadline,
                lazy_surr,
                allow_order_only=True,
                min_improve=0.1,
                solve_time_cap_s=min(0.010, rem_s),
            )
            eval_cnt += int(n_cp)
            if res_cp.valid:
                cp_art = CandidateArtifact(
                    expert_id="E9_Local_CPSAT",
                    tag="cpsat_focal",
                    assign=dict(a_cp),
                    bays=dict(b_cp),
                    crews=dict(cr_cp),
                    op_prio=dict(pr_cp),
                    task_orders={k: list(v) for k, v in (ord_cp or res_cp.robot_order_seq).items() if v} or None,
                    is_full=False,
                )
                self._suite._finish_artifact(cp_art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr, rob_cur_h=rob_cur_h)
                if cp_art.eval_res_b2 and cp_art.eval_res_b2.valid:
                    cp_delta = _score_cand_delta(cp_art)
                    cp_art.predicted_score = cp_delta
                    best_delta_cur = float(best.predicted_score if best.predicted_score is not None else 0.0)
                    if cp_delta + 1e-4 < best_delta_cur:
                        best = cp_art

        changed_ops_cnt = sum(
            1
            for oid in graph.op_ids
            if best.bays.get(oid, graph.base_bays.get(oid)) != graph.base_bays.get(oid)
            or int(best.crews.get(oid, graph.base_crews.get(oid, 0))) != int(graph.base_crews.get(oid, 0))
        )
        changed_tasks_cnt = sum(
            1
            for tid in graph.rel_tids
            if graph.base_assign.get(tid) and best.assign.get(tid) != graph.base_assign.get(tid)
        )
        eff_hys = 8.0 if (changed_ops_cnt > 0 or changed_tasks_cnt > 0) else 0.5
        best_rel_delta = float(best.predicted_score if best.predicted_score is not None else 0.0)
        fallback_keep = not (
            best.eval_res_b2 is not None
            and best.eval_res_b2.valid
            and best_rel_delta + eff_hys < 0.0
        )
        if fallback_keep:
            best = keep_art
        seq_horizon = (
            int(self.unwin_seq_horizon)
            if getattr(self, "unwin_seq_horizon", None) is not None
            else 0
        )
        inc_unwin = bool(self.commit_unwin_seq)
        orig_task_orders = best.task_orders
        if seq_horizon > 0:
            has_win = any(
                getattr(op, "resource_windows", ())
                for op in snapshot.operations
                if not getattr(op, "completed", False)
            )
            if not has_win:
                inc_unwin = True
                if isinstance(orig_task_orders, dict):
                    best.task_orders = {
                        k: list(v)[:seq_horizon]
                        for k, v in orig_task_orders.items()
                        if v
                    }
        try:
            prop = best.to_proposal(
                policy_id=f"V20_UNIFIED_PROTO_{int(round(cap_ms))}",
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
                "fallback_keep": fallback_keep,
                "keep_obj": round(keep_sc, 4),
                "best_obj": round(float(best.ja_unified_score or keep_sc), 4),
            }
        )
        return prop
