"""Phase V20-C: Commitment-Aware Neural Neighborhood Selector (N_k = U_t) & RQ1/RQ2 Strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.cvcr.snapshot import NativeSnapshot, op_predecessors
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
from amr_sched.rl.v19_hetero_graph import V19HeteroGraph, extract_v19_hetero_graph
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

NEIGHBORHOOD_PHYS_DIM = 24
NEIGHBORHOOD_FEAT_DIM = 64 + 64 + 64 + 192 + NEIGHBORHOOD_PHYS_DIM  # 408


def _elu_np(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.0, x, np.expm1(np.clip(x, -20.0, 0.0))).astype(np.float32, copy=False)


class NumpyNeighborhoodSelector:
    """Pure-NumPy commitment-aware neighborhood scorer attached to V19 Layer-1 embeddings."""

    def __init__(self, weights_path: Path) -> None:
        data = np.load(str(weights_path), allow_pickle=False)
        self.mu = np.asarray(data["mu"], dtype=np.float32)
        self.sig = np.maximum(1e-4, np.asarray(data["sig"], dtype=np.float32))
        self.w1 = np.asarray(data["w1"], dtype=np.float32)
        self.b1 = np.asarray(data["b1"], dtype=np.float32)
        self.w2 = np.asarray(data["w2"], dtype=np.float32)
        self.b2 = np.asarray(data["b2"], dtype=np.float32)
        self.w_lin = np.asarray(data["w_lin"], dtype=np.float32) if "w_lin" in data.files else None
        self.mlp_mix = float(data["mlp_mix"][0]) if "mlp_mix" in data.files else 1.0

    def score_tasks(self, feat_mat: np.ndarray) -> np.ndarray:
        """Return scalar release priority scores `[N_t]`."""
        if feat_mat.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        x = np.clip((feat_mat.astype(np.float32) - self.mu[None, :]) / self.sig[None, :], -6.0, 6.0)
        h = _elu_np(x @ self.w1 + self.b1[None, :])
        mlp_sc = (h @ self.w2 + self.b2).reshape(-1).astype(np.float32)
        if self.w_lin is not None:
            lin_sc = (x @ self.w_lin).reshape(-1).astype(np.float32)
            return float(self.mlp_mix) * mlp_sc + (1.0 - float(self.mlp_mix)) * lin_sc
        return mlp_sc


def extract_task_release_features(
    snapshot: NativeSnapshot,
    network: Network,
    graph: V19HeteroGraph,
    enc: dict[str, np.ndarray],
    inc_eval: EvalResult,
    inc_assign: dict[str, str],
    inc_bays: dict[str, str],
    inc_crews: dict[str, int],
) -> np.ndarray:
    """Build `[N_t, 408]` joint (Layer-1 neural + cross-domain physical coupling) matrix for releasable tasks."""
    N_t = len(graph.rel_tids)
    if N_t == 0:
        return np.zeros((0, NEIGHBORHOOD_FEAT_DIM), dtype=np.float32)

    op_by_id = {op.op_id: op for op in snapshot.operations}
    task_by_id = {t.task_id: t for t in snapshot.tasks}
    rob_by_id = {r.robot_id: r for r in snapshot.robots}
    agv_idx = {aid: i for i, aid in enumerate(graph.agv_ids)}
    op_idx = {oid: i for i, oid in enumerate(graph.op_ids)}

    h_task = enc["h_task"][0]  # [N_t, 64]
    h_op = enc["h_op"][0]      # [N_op, 64]
    h_agv = enc["h_agv"][0]    # [N_a, 64]
    h_g = enc["h_g"][0]        # [192]
    mean_h_op = h_op.mean(axis=0) if h_op.shape[0] > 0 else np.zeros((64,), dtype=np.float32)
    mean_h_agv = h_agv.mean(axis=0) if h_agv.shape[0] > 0 else np.zeros((64,), dtype=np.float32)

    op_ends = inc_eval.op_end
    op_starts = inc_eval.op_start
    unloads = inc_eval.order_unload_ticks
    cmax = max([int(snapshot.tick) + 1] + [int(v) for v in op_ends.values()] + [int(v) for v in unloads.values()])

    # AMR workload & spacetime corridor overlap counts from inc_eval
    amr_loads: dict[str, int] = {aid: 0 for aid in graph.agv_ids}
    for tid, aid in inc_assign.items():
        if aid in amr_loads:
            t_o = task_by_id.get(tid)
            if t_o is None or t_o.status.lower() != "completed":
                amr_loads[aid] += 1
    mean_amr_load = max(1.0, float(sum(amr_loads.values())) / max(1, len(graph.agv_ids)))

    amr_overlap: dict[str, float] = {aid: 0.0 for aid in graph.agv_ids}
    if inc_eval.robot_paths:
        occ: dict[str, list[tuple[int, str]]] = {}
        for aid, pts in inc_eval.robot_paths.items():
            for nd, tk in pts:
                occ.setdefault(nd, []).append((int(tk), aid))
        for aid, pts in inc_eval.robot_paths.items():
            cnt_ov = 0
            for nd, tk in pts:
                for t0, other_a in occ.get(nd, ()):
                    if other_a != aid and abs(int(tk) - t0) <= 4:
                        cnt_ov += 1
            amr_overlap[aid] = min(4.0, float(cnt_ov) / 10.0)

    # Kitting operation latest arrival & split lag
    op_max_unload: dict[str, int] = {}
    op_min_unload: dict[str, int] = {}
    for op in snapshot.operations:
        us = [int(unloads.get(kt, cmax)) for kt in op.kit_task_ids if kt in task_by_id]
        if us:
            op_max_unload[op.op_id] = max(us)
            op_min_unload[op.op_id] = min(us)

    # Precedence child fanout
    succ_cnt: dict[str, int] = {op.op_id: 0 for op in snapshot.operations}
    for op in snapshot.operations:
        for pid in op_predecessors(op):
            if pid in succ_cnt:
                succ_cnt[pid] += 1

    is_ws = 1.0 if len(network.nodes) >= 30 else 0.0
    inv_valid = 0.0 if inc_eval.valid else 1.0
    norm_cmax = float(max(60, cmax - int(snapshot.tick)))

    rows = np.zeros((N_t, NEIGHBORHOOD_FEAT_DIM), dtype=np.float32)
    for ti, tid in enumerate(graph.rel_tids):
        t = task_by_id[tid]
        oi = int(graph.task_to_op[ti])
        aid = inc_assign.get(tid) or t.assigned_agv or ""
        ai = agv_idx.get(aid, -1)

        ht_v = h_task[ti]
        ho_v = h_op[oi] if 0 <= oi < h_op.shape[0] else mean_h_op
        ha_v = h_agv[ai] if 0 <= ai < h_agv.shape[0] else mean_h_agv

        u_t = int(unloads.get(tid, cmax))
        t_tard = float(max(0, u_t - int(t.due_step))) / 60.0 if t.due_step is not None else 0.0
        u_rel = float(u_t - int(snapshot.tick)) / norm_cmax

        op_tard = 0.0
        is_crit_tail = 0.0
        is_kit_bottleneck = 0.0
        kit_split_span = 0.0
        op_fanout = 0.0
        bay_flex = 0.0
        win_cliff = 0.0
        if t.kit_op_id and t.kit_op_id in op_by_id:
            op = op_by_id[t.kit_op_id]
            e_o = int(op_ends.get(op.op_id, cmax))
            op_tard = float(max(0, e_o - int(op.due_step))) / 60.0
            is_crit_tail = 1.0 if e_o >= cmax - 3 else 0.0
            omax = op_max_unload.get(op.op_id, u_t)
            omin = op_min_unload.get(op.op_id, u_t)
            is_kit_bottleneck = 1.0 if u_t >= omax - 1 else 0.0
            kit_split_span = min(4.0, float(max(0, omax - omin)) / 30.0)
            op_fanout = min(4.0, float(succ_cnt.get(op.op_id, 0)) / 3.0)
            bay_flex = 1.0 if (op.op_id not in graph.frozen_ops and len(op.eligible_bays) > 1) else 0.0
            if op.resource_windows:
                win_cliff = 1.0

        rob = rob_by_id.get(aid)
        rob_disturbed = 1.0 if (aid in graph.clf.disturbed_robots) else 0.0
        rob_blocked = 1.0 if (rob and str(getattr(rob, "state", "")).lower() in ("blocked", "faulted")) else 0.0
        rob_load_ratio = float(amr_loads.get(aid, 0)) / mean_amr_load
        rob_ov = float(amr_overlap.get(aid, 0.0))
        is_pending = 1.0 if t.status.lower() == "pending" else 0.0
        dot_to = float(np.dot(ht_v, ho_v)) / 64.0
        dot_ta = float(np.dot(ht_v, ha_v)) / 64.0

        phys = np.array(
            [
                *graph.task_feats[ti].tolist(),  # 12
                t_tard,
                op_tard,
                u_rel,
                is_crit_tail,
                is_kit_bottleneck,
                kit_split_span,
                op_fanout,
                bay_flex,
                win_cliff,
                rob_load_ratio,
                rob_ov,
                0.5 * rob_disturbed + 0.5 * rob_blocked + 0.25 * is_pending + 0.25 * inv_valid + 0.1 * is_ws + 0.01 * (dot_to + dot_ta),
            ],
            dtype=np.float32,
        )
        rows[ti] = np.clip(np.concatenate([ht_v, ho_v, ha_v, h_g, phys], axis=0), -8.0, 8.0)
    return rows


def select_release_neighborhood(
    strategy: str,
    *,
    k_max: Optional[int],
    snapshot: NativeSnapshot,
    network: Network,
    graph: V19HeteroGraph,
    enc: dict[str, np.ndarray],
    inc_eval: EvalResult,
    inc_assign: dict[str, str],
    inc_bays: dict[str, str],
    inc_crews: dict[str, int],
    selector: Optional[NumpyNeighborhoodSelector] = None,
    rng: Optional[np.random.Generator] = None,
    precomputed_feats: Optional[np.ndarray] = None,
) -> tuple[set[str], set[str], set[str]]:
    """Select strictly `(release_tasks, release_ops, release_amrs)` under strategy and optional cap `k_max`."""
    rel_list = list(graph.rel_tids)
    N_t = len(rel_list)
    if N_t == 0 or strategy == "zero_release":
        return set(), set(), set()

    if strategy == "soft_all":
        rel_ops = {oid for oid in graph.op_ids if oid not in graph.frozen_ops}
        return set(rel_list), rel_ops, set(graph.agv_ids)

    eff_k = N_t if k_max is None else max(1, min(N_t, int(k_max)))
    op_by_id = {op.op_id: op for op in snapshot.operations}
    task_by_id = {t.task_id: t for t in snapshot.tasks}
    rob_by_id = {r.robot_id: r for r in snapshot.robots}
    unloads = inc_eval.order_unload_ticks
    op_ends = inc_eval.op_end
    cmax = max([int(snapshot.tick) + 1] + [int(v) for v in op_ends.values()] + [int(v) for v in unloads.values()])

    if strategy == "learned_dep_guided":
        assert selector is not None
        feats = (
            precomputed_feats
            if precomputed_feats is not None
            else extract_task_release_features(snapshot, network, graph, enc, inc_eval, inc_assign, inc_bays, inc_crews)
        )
        scores = selector.score_tasks(feats)
        ranked = [rel_list[i] for i in sorted(range(N_t), key=lambda i: (-float(scores[i]), rel_list[i]))]
        seeds = ranked[: max(1, (eff_k + 1) // 2)]
        # Propagate 1-hop causal kitting closure capped cleanly at eff_k
        rel_t, rel_o = expand_causal_release_closure(
            snapshot, graph, seeds, [], max_tasks=eff_k, max_ops=max(2, eff_k // 2 + 1)
        )
        for tid in ranked:
            if len(rel_t) >= eff_k:
                break
            rel_t.add(tid)
        if not rel_o:
            for tid in rel_t:
                oid = task_by_id[tid].kit_op_id
                if oid and oid not in graph.frozen_ops:
                    rel_o.add(oid)
        rel_a = {inc_assign[tid] for tid in rel_t if tid in inc_assign}
        return rel_t, rel_o, rel_a

    if strategy == "det_causal_cone":
        cone = build_causal_repair_cone(
            snapshot, graph, inc_eval, inc_assign, inc_bays, inc_crews,
            max_cone_amrs=max(2, min(len(graph.agv_ids), 3)),
            max_cone_tasks=eff_k,
            max_cone_ops=max(2, eff_k // 2 + 1),
        )
        rel_t = set(cone.cone_tasks[:eff_k])
        rel_o = {oid for oid in cone.cone_ops if oid not in graph.frozen_ops}
        if len(rel_t) < eff_k:
            for tid in rel_list:
                if tid not in rel_t:
                    rel_t.add(tid)
                if len(rel_t) >= eff_k:
                    break
        return rel_t, rel_o, set(cone.cone_amrs)

    if strategy == "transport_local":
        # Score purely by AMR transport congestion / stall / fault / pending distance
        amr_t_scores: list[tuple[float, str]] = []
        for tid in rel_list:
            t = task_by_id[tid]
            aid = inc_assign.get(tid) or t.assigned_agv or ""
            rob = rob_by_id.get(aid)
            sc = 0.0
            if aid in graph.clf.disturbed_robots:
                sc += 100.0
            if t.status.lower() == "pending":
                sc += 40.0
            if rob and str(getattr(rob, "state", "")).lower() in ("blocked", "faulted"):
                sc += 25.0 + float(getattr(rob, "stall_steps", 0))
            u_t = int(unloads.get(tid, cmax))
            sc += float(max(0, u_t - int(t.due_step))) * 0.5 if t.due_step is not None else 0.0
            amr_t_scores.append((sc, tid))
        amr_t_scores.sort(key=lambda x: (-x[0], x[1]))
        rel_t = {tid for _, tid in amr_t_scores[:eff_k]}
        rel_a = {inc_assign[tid] for tid in rel_t if tid in inc_assign}
        return rel_t, set(), rel_a

    if strategy == "critical_path_only":
        # Score purely by operation tardiness / makespan tail without transport or kit propagation
        cp_scores: list[tuple[float, str]] = []
        for tid in rel_list:
            t = task_by_id[tid]
            op_sc = 0.0
            if t.kit_op_id and t.kit_op_id in op_by_id:
                op = op_by_id[t.kit_op_id]
                e_o = int(op_ends.get(op.op_id, cmax))
                op_sc = float(max(0, e_o - int(op.due_step))) + (30.0 if e_o >= cmax - 3 else 0.0)
                if op.is_urgent:
                    op_sc += 15.0
            cp_scores.append((op_sc, tid))
        cp_scores.sort(key=lambda x: (-x[0], x[1]))
        rel_t = {tid for _, tid in cp_scores[:eff_k]}
        rel_o = {
            task_by_id[tid].kit_op_id
            for tid in rel_t
            if task_by_id[tid].kit_op_id and task_by_id[tid].kit_op_id not in graph.frozen_ops
        }
        rel_a = {inc_assign[tid] for tid in rel_t if tid in inc_assign}
        return rel_t, rel_o, rel_a

    if strategy == "random_subset":
        local_rng = rng if rng is not None else np.random.default_rng(int(snapshot.tick) + 42)
        chosen = local_rng.choice(rel_list, size=eff_k, replace=False).tolist()
        rel_t = set(str(x) for x in chosen)
        rel_o = {
            task_by_id[tid].kit_op_id
            for tid in rel_t
            if task_by_id[tid].kit_op_id and task_by_id[tid].kit_op_id not in graph.frozen_ops
        }
        rel_a = {inc_assign[tid] for tid in rel_t if tid in inc_assign}
        return rel_t, rel_o, rel_a

    raise ValueError(f"Unknown neighborhood strategy: {strategy}")


class V20CNeighborhoodProposer(V20BProposer):
    """Phase V20-C: V20-B + Learned Commitment-Aware Neighborhood Selection (U_t) & Strategy Mode."""

    def __init__(
        self,
        *,
        neighborhood_weights_path: Optional[Path] = None,
        neighborhood_strategy: str = "learned_dep_guided",
        release_k_max: Optional[int] = None,
        release_frac: Optional[float] = None,
        strict_fixed_release: bool = False,
        **v20b_kwargs: Any,
    ) -> None:
        super().__init__(**v20b_kwargs)
        self.neighborhood_strategy = str(neighborhood_strategy)
        self.release_k_max = None if release_k_max is None else int(release_k_max)
        self.release_frac = None if release_frac is None else float(release_frac)
        self.strict_fixed_release = bool(strict_fixed_release)
        self._nh_selector = (
            NumpyNeighborhoodSelector(Path(neighborhood_weights_path))
            if neighborhood_weights_path is not None
            else None
        )

    def _effective_k(self, n_rel: int) -> Optional[int]:
        if self.release_k_max is not None:
            return max(1, min(n_rel, self.release_k_max))
        if self.release_frac is not None:
            if self.release_frac <= 0.0:
                return 0
            return max(1, min(n_rel, int(round(self.release_frac * float(n_rel)))))
        return min(n_rel, max(3, min(8, int(round(0.45 * float(n_rel))))))

    def __call__(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        budget_ms: float = 35.0,
    ) -> CandidateProposal:
        # When strict_fixed_release=False, augment V20-B seamlessly OR evaluate controlled U_t
        if not self.strict_fixed_release and self.neighborhood_strategy == "none":
            return super().__call__(snapshot, network, budget_ms=budget_ms)

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

        rng = np.random.default_rng(self.seed * 100003 + int(snapshot.tick) + 17)
        released_count = 0

        if healthy and graph.rel_tids:
            ef, hf = _ensure_full_enc()
            if self.strict_fixed_release:
                # Strictly controlled RQ1 / RQ2 mode: ALL non-keep search operates inside U_t of size <= eff_k
                if self.neighborhood_strategy == "global_unanchored":
                    if (not is_ws_net) or self.enable_r2_global_ws:
                        a_g, sq_g = _regret2_insert_tasks(
                            snapshot, graph, fast_surr, dict(graph.fixed_assign), list(graph.rel_tids), eff_km_prio, bays_semi
                        )
                        _consider(CandidateArtifact("E1_V18_Regret2", "r2_global", a_g, dict(bays_semi), dict(crews_semi), dict(eff_km_prio), sq_g, True))
                    dec_f = decode_v19_with_cached_enc(np_full_net, ef, hf, graph, snapshot)
                    _consider(
                        CandidateArtifact(
                            "E2_AHG_Full", "ahg_full_g",
                            dict(dec_f.assign), dict(dec_f.bays), dict(dec_f.crews), dict(dec_f.op_prio),
                            dict(dec_f.task_sequences) if dec_f.task_sequences else None, True,
                        )
                    )
                    released_count = len(graph.rel_tids)
                elif self.neighborhood_strategy == "rule_frozen":
                    rel_t = set(pending_or_dist)
                    released_count = len(rel_t)
                    if rel_t:
                        anc = {tid: aid for tid, aid in graph.base_assign.items() if tid not in rel_t}
                        a_r, sq_r = _regret2_insert_tasks(snapshot, graph, fast_surr, anc, sorted(rel_t), eff_km_prio, graph.base_bays)
                        _consider(CandidateArtifact("RQ_RuleFrozen", "rf_r2", a_r, dict(graph.base_bays), dict(graph.base_crews), dict(eff_km_prio), sq_r, False))
                        sub_g = build_masked_subgraph(
                            graph, release_tasks=rel_t, release_ops=set(), skel_assign=graph.base_assign, skel_bays=graph.base_bays, skel_crews=graph.base_crews
                        )
                        dec_l = decode_v19_with_cached_enc(np_full_net, ef, hf, sub_g, snapshot)
                        _consider(
                            CandidateArtifact(
                                "RQ_RuleFrozen", "rf_v19",
                                dict(dec_l.assign), dict(dec_l.bays), dict(dec_l.crews), dict(dec_l.op_prio),
                                dict(dec_l.task_sequences) if dec_l.task_sequences else None, False,
                            )
                        )
                else:
                    eff_k = self._effective_k(len(graph.rel_tids))
                    if eff_k == 0 or self.neighborhood_strategy == "zero_release":
                        rel_t, rel_o, rel_a = set(), set(), set()
                    else:
                        rel_t, rel_o, rel_a = select_release_neighborhood(
                            self.neighborhood_strategy,
                            k_max=eff_k,
                            snapshot=snapshot,
                            network=network,
                            graph=graph,
                            enc=ef,
                            inc_eval=keep_eval,
                            inc_assign=graph.base_assign,
                            inc_bays=graph.base_bays,
                            inc_crews=graph.base_crews,
                            selector=self._nh_selector,
                            rng=rng,
                        )
                    released_count = len(rel_t)
                    if rel_t:
                        anc = {tid: aid for tid, aid in graph.base_assign.items() if tid not in rel_t}
                        use_bays = dict(bays_semi) if rel_o else dict(graph.base_bays)
                        a_u, sq_u = _regret2_insert_tasks(snapshot, graph, fast_surr, anc, sorted(rel_t), eff_km_prio, use_bays)
                        _consider(CandidateArtifact("V20C_Release", f"{self.neighborhood_strategy}_r2", a_u, use_bays, dict(graph.base_crews), dict(eff_km_prio), sq_u, False))

                        sub_g = build_masked_subgraph(
                            graph,
                            release_tasks=rel_t,
                            release_ops=rel_o,
                            skel_assign=graph.base_assign,
                            skel_bays=graph.base_bays,
                            skel_crews=graph.base_crews,
                            restrict_agvs=(rel_a | set(healthy[:2])) if is_ws_net else None,
                        )
                        dec_u = decode_v19_with_cached_enc(np_full_net, ef, hf, sub_g, snapshot)
                        _consider(
                            CandidateArtifact(
                                "V20C_Release", f"{self.neighborhood_strategy}_v19",
                                dict(dec_u.assign), dict(dec_u.bays), dict(dec_u.crews), dict(dec_u.op_prio),
                                dict(dec_u.task_sequences) if dec_u.task_sequences else None, False,
                            )
                        )
            else:
                # Full V20-B + Learned Commitment-Aware Neighborhood Proposal U_t
                if (not is_ws_net) or self.enable_r2_global_ws:
                    a_g, sq_g = _regret2_insert_tasks(
                        snapshot, graph, fast_surr, dict(graph.fixed_assign), list(graph.rel_tids), eff_km_prio, bays_semi
                    )
                    _consider(CandidateArtifact("E1_V18_Regret2", "r2_global", a_g, dict(bays_semi), dict(crews_semi), dict(eff_km_prio), sq_g, True), is_skeleton=True)
                if pending_or_dist and _cvcr_now_s() < deadline and (not is_ws_net):
                    anc = {tid: aid for tid, aid in graph.base_assign.items() if tid not in pending_or_dist}
                    a_z, sq_z = _regret2_insert_tasks(snapshot, graph, fast_surr, anc, sorted(pending_or_dist), eff_km_prio, bays_semi)
                    _consider(CandidateArtifact("E1_V18_Regret2", "r2_zc", a_z, dict(bays_semi), dict(crews_semi), dict(eff_km_prio), sq_z, False), is_skeleton=True)

                if _cvcr_now_s() + 0.002 < deadline:
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

                # Learned Commitment-Aware Selective Neighborhood N_k = U_t attached to Layer-1 encoder
                # Apply selective pruning only when non-trivial Workshop subgraph exists (is_ws_net and N_t >= 6)
                if self._nh_selector is not None and is_ws_net and len(graph.rel_tids) >= 6 and _cvcr_now_s() + 0.002 < deadline:
                    eff_k = self._effective_k(len(graph.rel_tids))
                    base_ref = best if (best.eval_res_b2 and best.eval_res_b2.valid) else keep_art
                    rel_t_l, rel_o_l, rel_a_l = select_release_neighborhood(
                        self.neighborhood_strategy,
                        k_max=eff_k,
                        snapshot=snapshot,
                        network=network,
                        graph=graph,
                        enc=ef,
                        inc_eval=base_ref.eval_res_b2 or keep_eval,
                        inc_assign=base_ref.assign,
                        inc_bays=base_ref.bays,
                        inc_crews=base_ref.crews,
                        selector=self._nh_selector,
                        rng=rng,
                    )
                    released_count = len(rel_t_l)
                    if rel_t_l:
                        sub_g_l = build_masked_subgraph(
                            graph,
                            release_tasks=rel_t_l,
                            release_ops=rel_o_l,
                            skel_assign=base_ref.assign,
                            skel_bays=base_ref.bays,
                            skel_crews=base_ref.crews,
                            restrict_agvs=rel_a_l if rel_a_l else None,
                        )
                        dec_l = decode_v19_with_cached_enc(np_full_net, ef, hf, sub_g_l, snapshot)
                        _consider(
                            CandidateArtifact(
                                "V20C_Neural_Nk", "neural_nk_v19",
                                dict(dec_l.assign), dict(dec_l.bays), dict(dec_l.crews), dict(dec_l.op_prio),
                                dict(dec_l.task_sequences) if dec_l.task_sequences else None, False,
                            )
                        )
                        if _cvcr_now_s() + 0.0015 < deadline:
                            anc_l = {tid: aid for tid, aid in base_ref.assign.items() if tid not in rel_t_l}
                            a_nl, sq_nl = _regret2_insert_tasks(
                                snapshot, graph, fast_surr, anc_l, sorted(rel_t_l), eff_km_prio, base_ref.bays
                            )
                            _consider(
                                CandidateArtifact(
                                    "V20C_Neural_Nk", "neural_nk_r2",
                                    a_nl, dict(base_ref.bays), dict(base_ref.crews), dict(eff_km_prio), sq_nl, False,
                                )
                            )

                run_e4 = self.e4_mode != "ws_or_invalid" or is_ws_net or (not keep_eval.valid)
                if self.enable_e4 and run_e4 and (_cvcr_now_s() + 0.0035 < deadline) and self._suite._v18_pol.pgmpo is not None:
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

                if self.enable_bidir_v18_to_v19 and (_cvcr_now_s() + 0.0025 < deadline) and v18_skeletons:
                    best_skel = min(v18_skeletons, key=lambda c: float(c.predicted_score if c.predicted_score is not None else 0.0))
                    if best_skel.eval_res_b2 and best_skel.eval_res_b2.valid:
                        cone = build_causal_repair_cone(
                            snapshot, graph, best_skel.eval_res_b2, best_skel.assign, best_skel.bays, best_skel.crews,
                            max_cone_amrs=3, max_cone_tasks=5, max_cone_ops=3,
                        )
                        rt_b, ro_b = expand_causal_release_closure(
                            snapshot, graph, cone.cone_tasks, cone.cone_ops, max_tasks=6, max_ops=4,
                        )
                        if rt_b or ro_b:
                            sub_b = build_masked_subgraph(
                                graph, release_tasks=rt_b, release_ops=ro_b,
                                skel_assign=best_skel.assign, skel_bays=best_skel.bays, skel_crews=best_skel.crews,
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

                if self.enable_e7 and (_cvcr_now_s() + 0.0025 < deadline):
                    nn_t = nn_eval_pool[:4] if nn_eval_pool else [
                        (best.tag, best.assign, best.bays, best.crews, best.op_prio, best.task_orders, best.is_full)
                    ]
                    for tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc in generate_causal_cone_candidates(
                        snapshot, network, graph, ev_b2, fast_surr, keep_eval, base_prio, nn_t, fresh_ops, pending_or_dist,
                        budget_ms_limit=1.5,
                    )[:4]:
                        _consider(CandidateArtifact("E7_CausalCone_VOI", tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc))

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
                policy_id=f"V20C_NEIGHBORHOOD_{int(round(cap_ms))}",
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
                "released_count": int(released_count),
                "keep_obj": round(keep_sc, 4),
                "best_obj": round(float(best.ja_unified_score or keep_sc), 4),
            }
        )
        return prop
