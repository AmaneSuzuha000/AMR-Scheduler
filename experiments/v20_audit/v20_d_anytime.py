"""Phase V20-D: Contextual Inner-Loop Anytime Controller a_k = (N_k, o_k, b_k, STOP)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.policy import CandidateProposal, LGCVCRPolicy
from amr_sched.cvcr.snapshot import NativeSnapshot
from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
from amr_sched.cvcr.v18_frontier_synthesis import compute_semicircle_kitting_bays
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
from experiments.v20_audit.v20_b_bidirectional import (
    V20BProposer,
    build_masked_subgraph,
    decode_v19_with_cached_enc,
    expand_causal_release_closure,
)

ANYTIME_OPS: tuple[str, ...] = (
    "op_regret2_local",
    "op_ahg_full",
    "op_ahg_staged",
    "op_e4_pgmpo",
    "op_bidir_v18_to_v19",
    "op_e7_causal_cone",
    "op_e9_cpsat_focal",
)
ANYTIME_BASE_COSTS_MS: dict[str, float] = {
    "op_regret2_local": 0.4,
    "op_ahg_full": 1.6,
    "op_ahg_staged": 1.6,
    "op_e4_pgmpo": 2.0,
    "op_bidir_v18_to_v19": 1.0,
    "op_e7_causal_cone": 2.0,
    "op_e9_cpsat_focal": 6.5,
}
ANYTIME_FEAT_DIM = 13 + len(ANYTIME_OPS)  # 20


def build_anytime_state_vector(
    *,
    rem_ms: float,
    cap_ms: float,
    is_ws_net: bool,
    n_rel_tids: int,
    n_ops: int,
    n_frozen_ops: int,
    has_win: bool,
    keep_valid: bool,
    has_disturbed: bool,
    blocked_frac: float,
    keep_sc: float,
    best_sc: float,
    eval_cnt: int,
    tried_mask: list[bool],
) -> np.ndarray:
    """Build 20-dim contextual state features for inner anytime step selection & STOP."""
    best_imp = max(-1.0, min(2.0, (keep_sc - best_sc) / max(25.0, abs(keep_sc))))
    vec = [
        max(0.0, min(1.0, rem_ms / max(5.0, cap_ms))),
        max(0.0, min(2.5, rem_ms / 20.0)),
        1.0 if is_ws_net else 0.0,
        min(2.0, float(n_rel_tids) / 12.0),
        float(n_frozen_ops) / float(max(1, n_ops)),
        1.0 if has_win else 0.0,
        1.0 if keep_valid else 0.0,
        1.0 if has_disturbed else 0.0,
        min(1.0, max(0.0, blocked_frac)),
        best_imp,
        1.0 if best_sc + 1e-4 < keep_sc else 0.0,
        min(2.0, float(eval_cnt) / 8.0),
        float(np.arcsinh(max(0.0, keep_sc) / 200.0)),
    ]
    vec.extend(1.0 if t else 0.0 for t in tried_mask)
    return np.asarray(vec, dtype=np.float32)


class NumpyAnytimeController:
    """Pure-NumPy linear Ridge VOI predictor + STOP threshold gate."""

    def __init__(self, weights_path: Path) -> None:
        data = np.load(str(weights_path), allow_pickle=False)
        self.mu = np.asarray(data["mu"], dtype=np.float32)
        self.sig = np.maximum(1e-4, np.asarray(data["sig"], dtype=np.float32))
        self.W = np.asarray(data["W"], dtype=np.float32)  # [M_ops, F]
        self.b = np.asarray(data["b"], dtype=np.float32)  # [M_ops]
        self.stop_threshold = float(data["stop_threshold"][0]) if "stop_threshold" in data.files else 0.05

    def predict_voi(self, feat_vec: np.ndarray) -> np.ndarray:
        x = np.clip((feat_vec.astype(np.float32) - self.mu) / self.sig, -5.0, 5.0)
        return (self.W @ x + self.b).astype(np.float32)


class V20DAnytimeProposer(V20BProposer):
    """Phase V20-D: Multi-step Inner Anytime Loop a_k = (N_k, o_k, b_k, STOP)."""

    def __init__(
        self,
        *,
        anytime_weights_path: Optional[Path] = None,
        stop_threshold_override: Optional[float] = None,
        collect_step_log: Optional[list[dict[str, Any]]] = None,
        **v20b_kwargs: Any,
    ) -> None:
        super().__init__(**v20b_kwargs)
        self._any_ctrl = (
            NumpyAnytimeController(Path(anytime_weights_path))
            if anytime_weights_path is not None
            else None
        )
        self.stop_threshold_override = stop_threshold_override
        self.collect_step_log = collect_step_log

    def __call__(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        budget_ms: float = 35.0,
    ) -> CandidateProposal:
        t0 = _cvcr_now_s()
        cap_ms = float(self.hard_budget_ms) if self.hard_budget_ms > 0 else float(budget_ms)
        # Enforce safe compute-step protection ceiling (~15ms soft on 35ms, scaling gently at 10/20/50ms)
        if self.soft_deadline_ms is not None:
            soft_ms = min(float(self.soft_deadline_ms), max(3.5, cap_ms * 0.72))
        else:
            soft_ms = min(19.5, max(3.5, cap_ms * 0.56))
        deadline = t0 + soft_ms / 1000.0

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
            expert_id="E0_KEEP", tag="keep",
            assign=dict(graph.base_assign), bays=dict(graph.base_bays), crews=dict(graph.base_crews),
            op_prio=dict(base_prio), task_orders=None, is_full=False,
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

        is_ws_net = len(network.nodes) >= 30
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
            return float(ja_delta + self.critic_beta * c_delta)

        bays_semi, crews_semi = compute_semicircle_kitting_bays(snapshot, network, clf, lazy_surr, graph.base_bays, bay_delta)
        _, _, km_prio = _compute_fresh_semicircle_bays(snapshot, network, graph, fast_surr, fresh_ops)
        eff_km_prio = km_prio or base_prio
        pending_or_dist = self._compute_pending_or_disturbed(snapshot, graph, clf)
        has_win = any(getattr(op, "resource_windows", ()) for op in snapshot.operations if not getattr(op, "completed", False))
        blocked_frac = float(
            sum(1 for r in snapshot.robots if str(getattr(r, "state", "")).lower() in ("blocked", "faulted"))
        ) / float(max(1, len(snapshot.robots)))

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
                art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, w_surr=w_surr,
                rob_cur_h=rob_cur_h, obs_ops=fresh_ops, fast_surr=fast_surr, compute_details=False,
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

        np_full_net = self._suite._v19_anchor._np_full_net
        np_staged_net = self._suite._v19_anchor._np_staged_net
        enc_full = None
        h12_full = None

        def _ensure_full_enc() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
            nonlocal enc_full, h12_full
            if enc_full is None:
                enc_full = np_full_net.encode_entities(graph)
                h12_full = np_full_net.a1_a2_heads(enc_full, graph)
            assert enc_full is not None and h12_full is not None
            return enc_full, h12_full

        def _exec_operator(op_name: str) -> None:
            nonlocal best, eval_cnt
            if op_name == "op_regret2_local":
                if (not is_ws_net) or self.enable_r2_global_ws:
                    a_g, sq_g = _regret2_insert_tasks(
                        snapshot, graph, fast_surr, dict(graph.fixed_assign), list(graph.rel_tids), eff_km_prio, bays_semi
                    )
                    _consider(CandidateArtifact("E1_V18_Regret2", "r2_global", a_g, dict(bays_semi), dict(crews_semi), dict(eff_km_prio), sq_g, True), is_skeleton=True)
                if pending_or_dist and (not is_ws_net) and _cvcr_now_s() < deadline:
                    anc = {tid: aid for tid, aid in graph.base_assign.items() if tid not in pending_or_dist}
                    a_z, sq_z = _regret2_insert_tasks(snapshot, graph, fast_surr, anc, sorted(pending_or_dist), eff_km_prio, bays_semi)
                    _consider(CandidateArtifact("E1_V18_Regret2", "r2_zc", a_z, dict(bays_semi), dict(crews_semi), dict(eff_km_prio), sq_z, False), is_skeleton=True)
            elif op_name == "op_ahg_full":
                ef, hf = _ensure_full_enc()
                dec_f = decode_v19_with_cached_enc(np_full_net, ef, hf, graph, snapshot)
                _consider(
                    CandidateArtifact(
                        "E2_AHG_Full", "ahg_full_g",
                        dict(dec_f.assign), dict(dec_f.bays), dict(dec_f.crews), dict(dec_f.op_prio),
                        dict(dec_f.task_sequences) if dec_f.task_sequences else None, True,
                    )
                )
            elif op_name == "op_ahg_staged":
                enc_s = np_staged_net.encode_entities(graph)
                h12_s = np_staged_net.a1_a2_heads(enc_s, graph)
                dec_s = decode_v19_with_cached_enc(np_staged_net, enc_s, h12_s, graph, snapshot)
                if (not keep_eval.valid) if str(self.include_staged_g) == "on_invalid" else bool(self.include_staged_g):
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
            elif op_name == "op_e4_pgmpo":
                if self._suite._v18_pol.pgmpo is not None:
                    asns = self._suite._v18_pol.pgmpo.generate_assignments(
                        snapshot, network, clf, healthy, graph.fixed_assign, bays_semi, lazy_surr
                    )
                    if asns:
                        top_p = min(
                            asns,
                            key=lambda a: float(
                                fast_surr.score(
                                    a, eff_km_prio, bays_semi, crews_semi,
                                    _default_task_order_from_prio(snapshot, graph, a, eff_km_prio),
                                )
                            ),
                        )
                        p_ord = _default_task_order_from_prio(snapshot, graph, top_p, eff_km_prio)
                        _consider(
                            CandidateArtifact(
                                "E4_V18_PGMPO", "pgmpo_top", dict(top_p), dict(bays_semi), dict(crews_semi),
                                dict(eff_km_prio), p_ord, True,
                            ),
                            is_skeleton=True,
                        )
            elif op_name == "op_bidir_v18_to_v19":
                skel = (
                    min(v18_skeletons, key=lambda c: float(c.predicted_score if c.predicted_score is not None else 0.0))
                    if v18_skeletons
                    else best
                )
                if skel.eval_res_b2 and skel.eval_res_b2.valid:
                    ef, hf = _ensure_full_enc()
                    cone = build_causal_repair_cone(
                        snapshot, graph, skel.eval_res_b2, skel.assign, skel.bays, skel.crews,
                        max_cone_amrs=3, max_cone_tasks=5, max_cone_ops=3,
                    )
                    rt_b, ro_b = expand_causal_release_closure(
                        snapshot, graph, cone.cone_tasks, cone.cone_ops, max_tasks=6, max_ops=4,
                    )
                    if rt_b or ro_b:
                        sub_b = build_masked_subgraph(
                            graph, release_tasks=rt_b, release_ops=ro_b,
                            skel_assign=skel.assign, skel_bays=skel.bays, skel_crews=skel.crews,
                            restrict_agvs=set(cone.cone_amrs) if cone.cone_amrs else None,
                        )
                        dec_b = decode_v19_with_cached_enc(np_full_net, ef, hf, sub_b, snapshot)
                        _consider(
                            CandidateArtifact(
                                "V20B_Bidir_V18ToV19", "bidir_v18_to_v19",
                                dict(dec_b.assign), dict(dec_b.bays), dict(dec_b.crews), dict(dec_b.op_prio),
                                dict(dec_b.task_sequences) if dec_b.task_sequences else None, False,
                            )
                        )
            elif op_name == "op_e7_causal_cone":
                nn_t = nn_eval_pool[:4] if nn_eval_pool else [
                    (best.tag, best.assign, best.bays, best.crews, best.op_prio, best.task_orders, best.is_full)
                ]
                for tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc in generate_causal_cone_candidates(
                    snapshot, network, graph, ev_b2, fast_surr, keep_eval, base_prio, nn_t, fresh_ops, pending_or_dist,
                    budget_ms_limit=1.4,
                )[:4]:
                    _consider(CandidateArtifact("E7_CausalCone_VOI", tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc))
            elif op_name == "op_e9_cpsat_focal":
                if best.eval_res_b2 and best.eval_res_b2.valid:
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
                            expert_id="E9_Local_CPSAT", tag="cpsat_focal",
                            assign=dict(a_cp), bays=dict(b_cp), crews=dict(cr_cp), op_prio=dict(pr_cp),
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

        stopped_early = False
        steps_taken = 0
        if healthy and graph.rel_tids:
            tried = [False] * len(ANYTIME_OPS)
            # Determine legal candidate operator mask consistent with V20-B structural discoveries
            def _is_legal_op(m_idx: int, rem_ms_now: float) -> bool:
                op_nm = ANYTIME_OPS[m_idx]
                if tried[m_idx]:
                    return False
                if rem_ms_now + 0.25 < ANYTIME_BASE_COSTS_MS[op_nm]:
                    return False
                if op_nm == "op_regret2_local" and is_ws_net and (not self.enable_r2_global_ws):
                    return False
                if op_nm in ("op_ahg_staged", "op_e4_pgmpo") and (not is_ws_net) and keep_eval.valid:
                    # Wait: allow trained controller to decide in data-collection or if self.e4_mode=="always"
                    if self.e4_mode == "ws_or_invalid":
                        return False
                if op_nm == "op_e9_cpsat_focal" and rem_ms_now < 7.0:
                    return False
                return True

            while _cvcr_now_s() + 0.0005 < deadline and steps_taken < len(ANYTIME_OPS):
                rem_ms = max(0.0, (deadline - _cvcr_now_s()) * 1000.0)
                legal_idx = [m for m in range(len(ANYTIME_OPS)) if _is_legal_op(m, rem_ms)]
                if not legal_idx:
                    break
                cur_best_sc = float(best.ja_unified_score or keep_sc)
                fvec = build_anytime_state_vector(
                    rem_ms=rem_ms,
                    cap_ms=cap_ms,
                    is_ws_net=is_ws_net,
                    n_rel_tids=len(graph.rel_tids),
                    n_ops=len(graph.op_ids),
                    n_frozen_ops=len(graph.frozen_ops),
                    has_win=has_win,
                    keep_valid=bool(keep_eval.valid),
                    has_disturbed=bool(clf.disturbed_robots),
                    blocked_frac=blocked_frac,
                    keep_sc=keep_sc,
                    best_sc=cur_best_sc,
                    eval_cnt=eval_cnt,
                    tried_mask=tried,
                )
                if self._any_ctrl is not None:
                    voi_preds = self._any_ctrl.predict_voi(fvec)
                    pick_m = max(legal_idx, key=lambda m: float(voi_preds[m]))
                    stop_thr = (
                        float(self.stop_threshold_override)
                        if self.stop_threshold_override is not None
                        else float(self._any_ctrl.stop_threshold)
                    )
                    if (
                        steps_taken >= 2
                        and keep_eval.valid
                        and best.eval_res_b2 is not None
                        and best.eval_res_b2.valid
                        and float(voi_preds[pick_m]) < stop_thr
                    ):
                        stopped_early = True
                        break
                else:
                    # Exploration / default order
                    pick_m = legal_idx[0]

                op_chosen = ANYTIME_OPS[pick_m]
                tried[pick_m] = True
                t_op0 = _cvcr_now_s()
                before_sc = float(best.ja_unified_score or keep_sc)
                _exec_operator(op_chosen)
                dt_op_ms = (_cvcr_now_s() - t_op0) * 1000.0
                after_sc = float(best.ja_unified_score or keep_sc)
                if self.collect_step_log is not None:
                    self.collect_step_log.append(
                        {
                            "feat": fvec,
                            "op_idx": int(pick_m),
                            "delta_ja": max(0.0, before_sc - after_sc),
                            "dt_ms": max(0.2, dt_op_ms),
                            "tick": int(snapshot.tick),
                        }
                    )
                steps_taken += 1

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
                policy_id=f"V20D_ANYTIME_{int(round(cap_ms))}",
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
                "eval_cnt": int(eval_cnt),
                "steps_taken": int(steps_taken),
                "stopped_early": stopped_early,
                "keep_obj": round(keep_sc, 4),
                "best_obj": round(float(best.ja_unified_score or keep_sc), 4),
            }
        )
        return prop
