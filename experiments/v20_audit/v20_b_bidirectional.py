"""Phase V20-B: Deterministic Bidirectional Joint Search (V18 <-> V19) + Shared Layer-1 Encoder."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.policy import (
    CandidateProposal,
    LGCVCRPolicy,
)
from amr_sched.cvcr.snapshot import NativeSnapshot
from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
from amr_sched.cvcr.v18_frontier_synthesis import (
    _mit26_precedence_closure_destroy,
    _surrogate_polish_assign,
    compute_semicircle_kitting_bays,
)
from amr_sched.network import Network
from amr_sched.rl.causal_repair_cone import (
    CausalRepairCone,
    FastJAWKittingSurrogate,
    _compute_fresh_semicircle_bays,
    _regret2_insert_tasks,
    build_causal_repair_cone,
    generate_causal_cone_candidates,
    observable_op_closure,
    project_zero_churn_candidate,
)
from amr_sched.rl.v19_hetero_constructor import (
    DecodedJointCandidate,
    NumpyV19HeteroConstructorNet,
    _load_v19_checkpoint,
)
from amr_sched.rl.v19_hetero_graph import V19HeteroGraph, extract_v19_hetero_graph
from experiments.v20_audit.common import CandidateArtifact
from experiments.v20_audit.proposer_variants import (
    _V19_RL_CKPT,
    _V19_STAGED_RL_CKPT,
    ExpertSuiteEvaluator,
    WindowAwareKittingSurrogate,
    _LazyKittingSurrogate,
    _default_task_order_from_prio,
    _healthy_robots_from_graph,
    derive_base_op_prio,
)


def decode_v19_with_cached_enc(
    np_net: NumpyV19HeteroConstructorNet,
    enc: dict[str, np.ndarray],
    h12_raw: dict[str, np.ndarray],
    graph: V19HeteroGraph,
    snapshot: NativeSnapshot,
) -> DecodedJointCandidate:
    """Decode greedy A1->A2->A3->A4 candidate reusing cached Layer-1 entity encoding `enc`."""
    N_op = len(graph.op_ids)
    N_t = len(graph.rel_tids)
    N_a = len(graph.agv_ids)
    eff_stage = getattr(np_net, "action_stage", "A123")

    task_focus_np = h12_raw["task_focus_logits"][0]
    bay_logits_np = np.where(graph.op_bay_mask, h12_raw["bay_logits"][0], -1e9)
    crew_logits_np = np.where(graph.op_crew_mask, h12_raw["crew_logits"][0], -1e9)

    bay_indices = np.zeros((N_op,), dtype=np.int64)
    crew_indices = np.zeros((N_op,), dtype=np.int64)
    bays_dict: dict[str, str] = {}
    crews_dict: dict[str, int] = {}
    for oi, oid in enumerate(graph.op_ids):
        if oid in graph.frozen_ops:
            bi = int(graph.op_incumbent_bay_idx[oi])
        else:
            bi = int(np.argmax(bay_logits_np[oi])) if bay_logits_np.shape[1] > 0 else 0
        bay_indices[oi] = bi
        bays_dict[oid] = graph.bay_ids[bi]

        if graph.op_crew_mask[oi].sum() <= 1:
            ci = int(graph.op_incumbent_crew_idx[oi])
        else:
            ci = int(np.argmax(crew_logits_np[oi])) if crew_logits_np.shape[1] > 0 else 0
        crew_indices[oi] = ci
        crews_dict[oid] = int(graph.crew_ids[ci])

    assign_dict = dict(graph.fixed_assign)
    agv_indices = np.zeros((N_t,), dtype=np.int64)
    if N_t > 0:
        order = sorted(range(N_t), key=lambda i: (-float(task_focus_np[i]), i))
        added_load = np.zeros((N_a,), dtype=np.float32)
        added_dur = np.zeros((N_a,), dtype=np.float32)
        for ti in order:
            oi = int(graph.task_to_op[ti])
            sel_bi = int(bay_indices[oi]) if oi >= 0 else int(graph.task_incumbent_bay_idx[ti])
            b_tr = float(graph.bay_xy_travel_from_pickup[ti, sel_bi])
            lg = np_net.a3_step_logits(enc, graph, ti, sel_bi, added_load, added_dur)[0]
            ai = int(np.argmax(lg))
            agv_indices[ti] = ai
            assign_dict[graph.rel_tids[ti]] = graph.agv_ids[ai]
            added_load[ai] += 1.0
            added_dur[ai] += b_tr + float(graph.task_agv_feats[ti, ai, 0])

    if N_t > 0 and eff_stage == "A1234":
        seq_scores_np = np_net.a4_seq_scores(enc, agv_indices)[0]
        by_agv_rel: dict[str, list[tuple[float, str]]] = {}
        for ti, tid in enumerate(graph.rel_tids):
            aid = assign_dict[tid]
            by_agv_rel.setdefault(aid, []).append((-float(seq_scores_np[ti]), tid))
        task_sequences: Optional[dict[str, list[str]]] = {}
        snap_by_id = {t.task_id: t for t in snapshot.tasks}
        rob_by_id = {r.robot_id: r for r in snapshot.robots}
        for aid, pairs in by_agv_rel.items():
            pairs.sort()
            ordered = [tid for _, tid in pairs]
            rob = rob_by_id.get(aid)
            if rob and rob.cur_task:
                cur_t = snap_by_id.get(rob.cur_task)
                if cur_t is not None and cur_t.assigned_agv == aid:
                    is_frozen_head = rob.cur_task in graph.frozen_tasks
                    is_active_head = (
                        rob.cur_task in ordered
                        and rob.cur_task not in graph.clf.disturbed_tasks
                        and not (
                            ordered
                            and snap_by_id.get(ordered[0]) is not None
                            and snap_by_id[ordered[0]].is_urgent
                            and not cur_t.is_urgent
                        )
                    )
                    if is_frozen_head and cur_t.task_id not in ordered:
                        ordered = [cur_t.task_id] + ordered
                    elif is_active_head and ordered[0] != rob.cur_task:
                        ordered = [rob.cur_task] + [tid for tid in ordered if tid != rob.cur_task]
            if ordered:
                task_sequences[aid] = ordered
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
        seq_scores_np = np.zeros((N_t,), dtype=np.float32)
        task_sequences = None
        op_prio = {
            op.op_id: max(int(snapshot.tick), int(op.due_step) - int(op.proc_steps))
            for op in snapshot.operations
        }

    return DecodedJointCandidate(
        assign=assign_dict,
        bays=bays_dict,
        crews=crews_dict,
        op_prio=op_prio,
        task_sequences=task_sequences,
        bay_indices=bay_indices,
        crew_indices=crew_indices,
        agv_indices=agv_indices,
        seq_scores=seq_scores_np,
    )


def build_masked_subgraph(
    graph: V19HeteroGraph,
    *,
    release_tasks: set[str],
    release_ops: set[str],
    skel_assign: dict[str, str],
    skel_bays: dict[str, str],
    skel_crews: dict[str, int],
    restrict_agvs: Optional[set[str]] = None,
) -> V19HeteroGraph:
    """Create lightweight masked V19HeteroGraph releasing ONLY `(release_tasks, release_ops)` on top of skeleton."""
    sub = copy.copy(graph)
    sub.frozen_ops = set(graph.frozen_ops)
    sub.op_crew_mask = graph.op_crew_mask.copy()
    sub.task_agv_mask = graph.task_agv_mask.copy()
    sub.op_incumbent_bay_idx = graph.op_incumbent_bay_idx.copy()
    sub.op_incumbent_crew_idx = graph.op_incumbent_crew_idx.copy()
    sub.task_incumbent_agv_idx = graph.task_incumbent_agv_idx.copy()

    agv_idx = {aid: i for i, aid in enumerate(graph.agv_ids)}
    bay_idx = {bid: i for i, bid in enumerate(graph.bay_ids)}
    crew_idx = {int(cid): i for i, cid in enumerate(graph.crew_ids)}

    for oi, oid in enumerate(graph.op_ids):
        if oid in release_ops or oid in graph.frozen_ops:
            continue
        sub.frozen_ops.add(oid)
        sb = skel_bays.get(oid)
        if sb in bay_idx and graph.op_bay_mask[oi, bay_idx[sb]]:
            sub.op_incumbent_bay_idx[oi] = bay_idx[sb]
        sc = skel_crews.get(oid)
        if sc is not None and int(sc) in crew_idx and graph.op_crew_mask[oi, crew_idx[int(sc)]]:
            ci = crew_idx[int(sc)]
            sub.op_incumbent_crew_idx[oi] = ci
            sub.op_crew_mask[oi, :] = False
            sub.op_crew_mask[oi, ci] = True

    restrict_indices = (
        {agv_idx[aid] for aid in restrict_agvs if aid in agv_idx}
        if restrict_agvs
        else None
    )
    for ti, tid in enumerate(graph.rel_tids):
        sa = skel_assign.get(tid)
        if sa in agv_idx and graph.task_agv_mask[ti, agv_idx[sa]]:
            ai = agv_idx[sa]
            sub.task_incumbent_agv_idx[ti] = ai
            if tid not in release_tasks:
                sub.task_agv_mask[ti, :] = False
                sub.task_agv_mask[ti, ai] = True
            elif restrict_indices:
                allowed = [idx for idx in restrict_indices if graph.task_agv_mask[ti, idx]]
                if ai not in allowed and graph.task_agv_mask[ti, ai]:
                    allowed.append(ai)
                if allowed:
                    sub.task_agv_mask[ti, :] = False
                    for idx in allowed:
                        sub.task_agv_mask[ti, idx] = True
    return sub


def expand_causal_release_closure(
    snapshot: NativeSnapshot,
    graph: V19HeteroGraph,
    seed_tasks: list[str],
    seed_ops: list[str],
    *,
    max_tasks: int,
    max_ops: int,
) -> tuple[set[str], set[str]]:
    """Expand seed tasks/ops with coupled kit partners and predecessor/same-bay ops strictly within soft limits."""
    rel_set = set(graph.rel_tids)
    modifiable_ops = {oid for oid in graph.op_ids if oid not in graph.frozen_ops}
    op_by_id = {op.op_id: op for op in snapshot.operations}
    t_by_id = {t.task_id: t for t in snapshot.tasks}

    out_ops: list[str] = []
    out_tasks: list[str] = []

    def _add_op(oid: str) -> None:
        if oid in modifiable_ops and oid not in out_ops and len(out_ops) < max_ops:
            out_ops.append(oid)

    def _add_task(tid: str) -> None:
        if tid in rel_set and tid not in out_tasks and len(out_tasks) < max_tasks:
            out_tasks.append(tid)

    for oid in seed_ops:
        _add_op(oid)
    for tid in seed_tasks:
        _add_task(tid)
        t_obj = t_by_id.get(tid)
        if t_obj and t_obj.kit_op_id:
            _add_op(t_obj.kit_op_id)
            op_obj = op_by_id.get(t_obj.kit_op_id)
            if op_obj:
                for kt in op_obj.kit_task_ids:
                    _add_task(kt)
    for oid in list(out_ops):
        op_obj = op_by_id.get(oid)
        if op_obj:
            for kt in op_obj.kit_task_ids:
                _add_task(kt)
    return set(out_tasks), set(out_ops)


class V20BProposer:
    """Phase V20-B: Deterministic Bidirectional Joint Search (V18 <-> V19) with Shared Layer-1 Encoder."""

    def __init__(
        self,
        *,
        hard_budget_ms: float = 35.0,
        soft_deadline_ms: Optional[float] = 15.0,
        seed: int = 0,
        enable_e4: bool = True,
        e4_selector: str = "fast_surr",
        e4_include_zc: bool = True,
        e4_mode: str = "always",
        staged_mode: str = "always",
        enable_bidir_v18_to_v19: bool = True,
        enable_e6_static_or_headroom: bool = True,
        enable_e7: bool = True,
        enable_e9: bool = False,
        enable_early_stop: bool = True,
        compact_calm_ms: Optional[float] = 7.2,
        enable_r2_global_ws: bool = False,
        r2_zc_ws_mode: str = "never",
        include_staged_g: bool | str = "on_invalid",
        unwin_seq_horizon: Optional[int] = 2,
        critic_weights_path: Optional[Path] = None,
        critic_beta: float = 0.0,
        critic_mode: str = "ws_only",
        seq_diff_b2: bool = True,
    ) -> None:
        self.hard_budget_ms = float(hard_budget_ms)
        self.soft_deadline_ms = None if soft_deadline_ms is None else float(soft_deadline_ms)
        self.seed = int(seed)
        self.enable_e4 = bool(enable_e4)
        self.e4_selector = str(e4_selector)
        self.e4_include_zc = bool(e4_include_zc)
        self.e4_mode = str(e4_mode)
        self.staged_mode = str(staged_mode)
        self.enable_bidir_v18_to_v19 = bool(enable_bidir_v18_to_v19)
        self.enable_e6_static_or_headroom = bool(enable_e6_static_or_headroom)
        self.enable_e7 = bool(enable_e7)
        self.enable_e9 = bool(enable_e9)
        self.enable_early_stop = bool(enable_early_stop)
        self.compact_calm_ms = None if compact_calm_ms is None else float(compact_calm_ms)
        self.enable_r2_global_ws = bool(enable_r2_global_ws)
        self.r2_zc_ws_mode = str(r2_zc_ws_mode)
        self.include_staged_g: bool | str = (
            include_staged_g if isinstance(include_staged_g, str) else bool(include_staged_g)
        )
        self.unwin_seq_horizon = None if unwin_seq_horizon is None else int(unwin_seq_horizon)
        self.critic_beta = float(critic_beta)
        self.critic_mode = str(critic_mode)
        self.seq_diff_b2 = bool(seq_diff_b2)
        self._critic = None
        if critic_weights_path is not None and float(critic_beta) > 0.0:
            from experiments.v20_audit.critic_model import NumpyValueCritic

            self._critic = NumpyValueCritic(Path(critic_weights_path))
        self._suite = ExpertSuiteEvaluator(seed=self.seed)

    def _compute_pending_or_disturbed(
        self,
        snapshot: NativeSnapshot,
        graph: V19HeteroGraph,
        clf: Any,
    ) -> set[str]:
        t_by_id = {t.task_id: t for t in snapshot.tasks}
        return {
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
            min(float(self.soft_deadline_ms), max(3.5, cap_ms * 0.72))
            if self.soft_deadline_ms is not None
            else max(3.5, cap_ms * 0.68)
        )
        deadline = t0 + max(3.5, soft_ms) / 1000.0

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
            seq_diff_b2=bool(self.seq_diff_b2),
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
            keep_art,
            snapshot=snapshot,
            graph=graph,
            ev_b2=ev_b2,
            w_surr=w_surr,
            rob_cur_h=rob_cur_h,
            obs_ops=fresh_ops,
            fast_surr=fast_surr,
            compute_details=False,
        )
        best = keep_art
        keep_sc = float(keep_art.ja_unified_score or 1e9)
        keep_eval = keep_art.eval_res_b2
        assert keep_eval is not None
        keep_seq = {k: list(v) for k, v in keep_eval.robot_order_seq.items() if v} if keep_eval.robot_order_seq else None
        ev_b2.set_incumbent(
            graph.base_assign, keep_eval, op_priority=base_prio, op_bay_overrides=graph.base_bays, task_order_override=keep_seq
        )

        is_ws_net = len(network.nodes) >= 30
        has_blocked_or_fb = any(
            str(getattr(r, "state", "")).lower() == "faulted"
            or float(getattr(r, "stall_steps", 0)) >= 6
            for r in snapshot.robots
        ) or bool(
            getattr(snapshot, "execution_feedback", None) is not None
            and not snapshot.execution_feedback.is_empty()
        )
        compact_calm_stop = bool(
            self.compact_calm_ms is not None
            and (not is_ws_net)
            and keep_eval.valid
            and (not clf.disturbed_robots)
            and (not clf.disturbed_tasks)
            and (not has_blocked_or_fb)
        )
        use_critic = bool(
            self._critic is not None
            and self.critic_beta > 0.0
            and keep_eval.valid
            and (self.critic_mode != "ws_only" or is_ws_net)
        )
        keep_stats = None
        keep_critic_raw = 0.0
        sim_h_critic = int(keep_art.ja_proxy_details.get("sim_horizon", max(600, int(snapshot.tick) + 300)))
        if use_critic:
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
            if not use_critic or keep_stats is None or art.eval_res_b2 is None or not art.eval_res_b2.valid:
                return ja_delta
            from experiments.v20_audit.critic_model import extract_critic_features

            fv_c = extract_critic_features(
                snapshot, network, graph, art.eval_res_b2, float(art.ja_unified_score or keep_sc),
                art.assign, art.bays, art.crews, art.task_orders,
                keep_stats, rob_cur_h, sim_h_critic, is_keep_cand=False,
            )
            c_delta = float(self._critic.predict_delta_keep(fv_c) - keep_critic_raw)
            if self.critic_mode == "rel_capped":
                cap_d = 0.25 * max(40.0, abs(keep_sc))
                c_delta = max(-cap_d, min(cap_d, c_delta))
            return float(ja_delta + self.critic_beta * c_delta)

        bays_semi, crews_semi = compute_semicircle_kitting_bays(snapshot, network, clf, lazy_surr, graph.base_bays, bay_delta)
        _, _, km_prio = _compute_fresh_semicircle_bays(snapshot, network, graph, fast_surr, fresh_ops)
        eff_km_prio = km_prio or base_prio
        pending_or_dist = self._compute_pending_or_disturbed(snapshot, graph, clf)
        has_win = any(getattr(op, "resource_windows", ()) for op in snapshot.operations if not getattr(op, "completed", False))

        eval_cnt = 1
        seen_sigs: set[str] = {keep_art.ensure_sig()}
        nn_eval_pool: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]], bool]] = []
        v18_skeletons: list[CandidateArtifact] = []

        def _consider(art: CandidateArtifact, *, is_skeleton: bool = False) -> bool:
            nonlocal best, eval_cnt
            sig = art.ensure_sig()
            if sig in seen_sigs or _cvcr_now_s() >= deadline:
                return False
            seen_sigs.add(sig)
            self._suite._finish_artifact(
                art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr, rob_cur_h=rob_cur_h,
                obs_ops=fresh_ops, fast_surr=fast_surr, compute_details=False,
            )
            eval_cnt += 1
            if art.eval_res_b2 and art.eval_res_b2.valid:
                nn_eval_pool.append((art.tag, art.assign, art.bays, art.crews, art.op_prio, art.task_orders, art.is_full))
                if is_skeleton:
                    v18_skeletons.append(art)
                sc_new = _score_cand_delta(art)
                art.predicted_score = sc_new
                sc_best = float(best.predicted_score if best.predicted_score is not None else 0.0)
                if sc_new + 1e-4 < sc_best:
                    best = art
                    ev_b2.set_incumbent(
                        best.assign, best.eval_res_b2, op_priority=best.op_prio, op_bay_overrides=best.bays, task_order_override=best.task_orders
                    )
                return True
            return False

        # Shared Layer-1 Encoders (encoded strictly once per snapshot)
        np_full_net = self._suite._v19_anchor._np_full_net
        np_staged_net = self._suite._v19_anchor._np_staged_net
        enc_full = None
        h12_full = None
        enc_staged = None
        h12_staged = None

        def _ensure_full_enc() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
            nonlocal enc_full, h12_full
            if enc_full is None:
                enc_full = np_full_net.encode_entities(graph)
                h12_full = np_full_net.a1_a2_heads(enc_full, graph)
            assert enc_full is not None and h12_full is not None
            return enc_full, h12_full

        early_stopped = False
        if healthy and graph.rel_tids:
            # Tier 1: E1 Regret-2 (compact 4-Topo always; Workshop per mode)
            if (not is_ws_net) or self.enable_r2_global_ws:
                a_g, sq_g = _regret2_insert_tasks(
                    snapshot, graph, fast_surr, dict(graph.fixed_assign), list(graph.rel_tids), eff_km_prio, bays_semi
                )
                _consider(
                    CandidateArtifact("E1_V18_Regret2", "r2_global", a_g, dict(bays_semi), dict(crews_semi), dict(eff_km_prio), sq_g, True),
                    is_skeleton=True,
                )
            if pending_or_dist and _cvcr_now_s() < deadline:
                run_r2_zc = (not is_ws_net) or (
                    self.r2_zc_ws_mode == "always"
                    or (self.r2_zc_ws_mode == "windows_or_invalid" and (has_win or not keep_eval.valid))
                )
                if run_r2_zc:
                    anc = {tid: aid for tid, aid in graph.base_assign.items() if tid not in pending_or_dist}
                    use_bays_zc = dict(bays_semi) if (not is_ws_net or has_win or not keep_eval.valid) else dict(graph.base_bays)
                    use_crews_zc = dict(crews_semi) if (not is_ws_net or has_win or not keep_eval.valid) else dict(graph.base_crews)
                    use_prio_zc = dict(eff_km_prio) if (not is_ws_net or has_win or not keep_eval.valid) else dict(base_prio)
                    a_z, sq_z = _regret2_insert_tasks(
                        snapshot, graph, fast_surr, anc, sorted(pending_or_dist), use_prio_zc, use_bays_zc
                    )
                    _consider(
                        CandidateArtifact("E1_V18_Regret2", "r2_zc", a_z, use_bays_zc, use_crews_zc, use_prio_zc, sq_z, False),
                        is_skeleton=True,
                    )

            # Tier 1: Shared Layer-1 V19 Joint & Staged
            if _cvcr_now_s() + 0.002 < deadline:
                ef, hf = _ensure_full_enc()
                dec_f = decode_v19_with_cached_enc(np_full_net, ef, hf, graph, snapshot)
                _consider(
                    CandidateArtifact(
                        "E2_AHG_Full", "ahg_full_g",
                        dict(dec_f.assign), dict(dec_f.bays), dict(dec_f.crews), dict(dec_f.op_prio),
                        dict(dec_f.task_sequences) if dec_f.task_sequences else None, True,
                    )
                )
            run_staged = self.staged_mode != "ws_or_invalid" or is_ws_net or (not keep_eval.valid)
            if run_staged and _cvcr_now_s() + 0.002 < deadline:
                enc_staged = np_staged_net.encode_entities(graph)
                h12_staged = np_staged_net.a1_a2_heads(enc_staged, graph)
                dec_s = decode_v19_with_cached_enc(np_staged_net, enc_staged, h12_staged, graph, snapshot)
                staged_g_flag = (
                    (not keep_eval.valid)
                    if str(self.include_staged_g) == "on_invalid"
                    else bool(self.include_staged_g)
                )
                if staged_g_flag:
                    _consider(
                        CandidateArtifact(
                            "E3_AHG_Staged", "ahg_staged_g",
                            dict(dec_s.assign), dict(dec_s.bays), dict(dec_s.crews), dict(dec_s.op_prio),
                            dict(dec_s.task_sequences) if dec_s.task_sequences else None, False,
                        )
                    )
                zc_a, zc_b, zc_c, zc_p, zc_sq = project_zero_churn_candidate(
                    snapshot, graph, dec_s.assign, dec_s.bays, dec_s.crews, dec_s.op_prio, dec_s.task_sequences,
                    fresh_ops, pending_or_dist, rob_cur_h,
                )
                _consider(CandidateArtifact("E3_AHG_Staged", "ahg_staged_zc", zc_a, zc_b, zc_c, zc_p, zc_sq, False))

            # Deterministic <10 ms scale gate (compact calm) or early tail convergence stop
            if compact_calm_stop:
                early_stopped = True
            elif (
                self.enable_early_stop
                and keep_eval.valid
                and int(keep_eval.starvation_ticks) == 0
                and (not clf.disturbed_robots)
                and (not clf.disturbed_tasks)
                and (not has_blocked_or_fb)
                and (not has_win)
                and len(graph.rel_tids) <= 4
                and int(snapshot.tick) > 2
                and float(best.ja_unified_score or keep_sc) <= 1.005 * max(1.0, keep_sc)
            ):
                early_stopped = True

            # E4 PGMPO (fast_surr selected instead of noisy w_surr) + ZC projection
            run_e4 = self.e4_mode != "ws_or_invalid" or is_ws_net or (not keep_eval.valid)
            if (not early_stopped) and self.enable_e4 and run_e4 and (_cvcr_now_s() + 0.0035 < deadline) and self._suite._v18_pol.pgmpo is not None:
                asns = self._suite._v18_pol.pgmpo.generate_assignments(
                    snapshot, network, clf, healthy, graph.fixed_assign, bays_semi, lazy_surr
                )
                if asns:
                    if self.e4_selector == "fast_surr":
                        top_p = min(
                            asns,
                            key=lambda a: float(
                                fast_surr.score(
                                    a, eff_km_prio, bays_semi, crews_semi,
                                    _default_task_order_from_prio(snapshot, graph, a, eff_km_prio),
                                )
                            ),
                        )
                    else:
                        top_p = min(asns, key=lambda a: float(w_surr.score(a, eff_km_prio, bays_semi, crews_semi)))
                    p_ord = _default_task_order_from_prio(snapshot, graph, top_p, eff_km_prio)
                    if self.e4_include_zc and (not is_ws_net) and pending_or_dist and keep_eval.valid:
                        z_pa, z_pb, z_pc, z_pp, z_psq = project_zero_churn_candidate(
                            snapshot, graph, top_p, bays_semi, crews_semi, eff_km_prio, p_ord,
                            fresh_ops, pending_or_dist, rob_cur_h,
                        )
                        _consider(
                            CandidateArtifact("E4_V18_PGMPO", "pgmpo_zc", z_pa, z_pb, z_pc, z_pp, z_psq, False),
                            is_skeleton=True,
                        )
                    if _cvcr_now_s() + 0.002 < deadline:
                        _consider(
                            CandidateArtifact(
                                "E4_V18_PGMPO", "pgmpo_top", dict(top_p), dict(bays_semi), dict(crews_semi),
                                dict(eff_km_prio), p_ord, True,
                            ),
                            is_skeleton=True,
                        )

            # Reverse Bidirectional (V18 skeleton -> CausalRepairCone subgraph -> V19 local co-decode)
            if (not early_stopped) and self.enable_bidir_v18_to_v19 and (_cvcr_now_s() + 0.0025 < deadline) and v18_skeletons:
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
                            graph,
                            release_tasks=rel_t_sub,
                            release_ops=rel_o_sub,
                            skel_assign=best_skel.assign,
                            skel_bays=best_skel.bays,
                            skel_crews=best_skel.crews,
                            restrict_agvs=set(cone.cone_amrs) if cone.cone_amrs else None,
                        )
                        dec_bi = decode_v19_with_cached_enc(np_full_net, ef, hf, sub_g, snapshot)
                        _consider(
                            CandidateArtifact(
                                "V20B_Bidir_V18ToV19", "bidir_v18_to_v19",
                                dict(dec_bi.assign), dict(dec_bi.bays), dict(dec_bi.crews), dict(dec_bi.op_prio),
                                dict(dec_bi.task_sequences) if dec_bi.task_sequences else None, False,
                            )
                        )

            # Forward Direction (V19/incumbent -> E7_CausalCone_VOI)
            if (not early_stopped) and self.enable_e7 and (_cvcr_now_s() + 0.0025 < deadline):
                nn_t = nn_eval_pool[:4] if nn_eval_pool else [
                    (best.tag, best.assign, best.bays, best.crews, best.op_prio, best.task_orders, best.is_full)
                ]
                for tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc in generate_causal_cone_candidates(
                    snapshot, network, graph, ev_b2, fast_surr, keep_eval, base_prio, nn_t, fresh_ops, pending_or_dist,
                    budget_ms_limit=1.5,
                )[:4]:
                    _consider(CandidateArtifact("E7_CausalCone_VOI", tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc))

            # Optional Polish when static / headroom exists
            if (
                (not early_stopped)
                and self.enable_e6_static_or_headroom
                and (_cvcr_now_s() + 0.004 < deadline)
                and len(healthy) >= 2
                and ((not clf.disturbed_robots and int(snapshot.tick) <= 5) or (_cvcr_now_s() + 0.006 < deadline))
            ):
                pol_a, _ = _surrogate_polish_assign(
                    w_surr, graph.rel_tids, healthy, dict(best.assign), dict(best.bays), dict(best.crews), dict(best.op_prio),
                    max_passes=1, max_focus_tasks=6, do_swaps=True,
                )
                _consider(
                    CandidateArtifact(
                        "E6_V18_Polish", "polish_best", pol_a, dict(best.bays), dict(best.crews),
                        dict(best.op_prio), _default_task_order_from_prio(snapshot, graph, pol_a, best.op_prio), False,
                    )
                )

            # Optional Focal CP-SAT when large headroom remains
            if (not early_stopped) and self.enable_e9 and (_cvcr_now_s() + 0.007 < deadline) and best.eval_res_b2 and best.eval_res_b2.valid:
                cp_owner = LGCVCRPolicy("V18_EA_JOINT", network, use_rust=True)
                rem_s = max(0.002, deadline - _cvcr_now_s() - 0.002)
                a_cp, b_cp, cr_cp, pr_cp, ord_cp, res_cp, n_cp = cp_owner._search_local_cpsat_window(
                    snapshot, clf, ev_b2, dict(best.assign), dict(best.bays), dict(best.crews),
                    best.eval_res_b2, healthy, dict(best.op_prio), deadline, lazy_surr,
                    allow_order_only=True, min_improve=0.1, solve_time_cap_s=min(0.006, rem_s),
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
                    self._suite._finish_artifact(
                        cp_art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr,
                        rob_cur_h=rob_cur_h, obs_ops=fresh_ops, fast_surr=fast_surr, compute_details=False,
                    )
                    if cp_art.eval_res_b2 and cp_art.eval_res_b2.valid:
                        cp_delta = _score_cand_delta(cp_art)
                        cp_art.predicted_score = cp_delta
                        if cp_delta + 1e-4 < float(best.predicted_score if best.predicted_score is not None else 0.0):
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

        seq_horizon = int(self.unwin_seq_horizon) if self.unwin_seq_horizon is not None else 0
        inc_unwin = False
        orig_task_orders = best.task_orders
        if seq_horizon > 0 and not has_win:
            inc_unwin = True
            if isinstance(orig_task_orders, dict):
                best.task_orders = {k: list(v)[:seq_horizon] for k, v in orig_task_orders.items() if v}
        try:
            prop = best.to_proposal(
                policy_id=f"V20B_BIDIR_{int(round(cap_ms))}",
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
                "early_stopped": early_stopped,
                "eval_cnt": int(eval_cnt),
                "keep_obj": round(keep_sc, 4),
                "best_obj": round(float(best.ja_unified_score or keep_sc), 4),
            }
        )
        return prop
