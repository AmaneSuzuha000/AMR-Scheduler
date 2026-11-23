"""V14 Execution-Aware Hierarchical Production-Logistics 3-Layer RL (`V14_JOINT_RL`).

Implements all three coupled RL decision layers directly tied to work-order execution:
  1. **Target Operation-Window Portfolio (`V14OpFleetScorer`)**:
     Causal per-operation + execution-drift representation across ALL unstarted operations
     (preserving downstream DAG criticality, kitting readiness, speed-drift exposure,
     discrete resource-window cliffs, and early-wait waste).
  2. **Joint Repair Neighborhood Proposer + Set-Attention Candidate Ranker (`V14JointPolicy`)**:
     Active cross-op swap + execution-drift kit-rescue candidate generation, scored with
     multi-head self-attention across competing candidates and a conservative causal gate
     preventing myopic overrides unless multi-step trajectory advantage warrants it.
  3. **Iterative DR-ALNS Anytime Stopping Controller (`select_anytime_operator`)**:
     Dynamic 10-20ms loop over `{COMMIT_NOW, OP_LOCAL_REFINE, BAY_CREW_CPSAT, HETERO_V8_POLISH}`
     conditioned on live incumbent gain, remaining wall-clock budget, and resource contention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.checkpoints import resolve_default_checkpoint
from amr_sched.cvcr.commitment import CommitmentClassification
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.snapshot import NativeSnapshot, op_predecessors
from amr_sched.cvcr.v13_kitting_rl import (
    KITTING_CAND_FEAT_DIM,
    KITTING_GLOBAL_FEAT_DIM,
    KittingJointCandidate,
    _ancestor_chain,
    _best_bay_and_crew_for_op,
    _compute_op_window_stats,
    extract_kitting_candidate_features,
    extract_v13_global_features,
    generate_kitting_joint_candidates,
    score_det_kitting_rule,
)
from amr_sched.network import Network

V14_OP_FEAT_DIM = 16
V14_CAND_EXTRA_DIM = 8
V14_CAND_FEAT_DIM = KITTING_CAND_FEAT_DIM + V14_CAND_EXTRA_DIM  # 32-D per candidate
V14_SEARCH_FEAT_DIM = 8
ANYTIME_OPS = ("COMMIT_NOW", "LOCAL_OP_1OPT", "BAY_CREW_CPSAT", "HETERO_LNS_POLISH")

DEFAULT_V14_CKPT = resolve_default_checkpoint("v14_joint_rl.npz")


def _relu(x: np.ndarray) -> np.ndarray:
    """Elementwise ReLU nonlinearity."""
    return np.maximum(0.0, x)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically clipped logistic sigmoid."""
    arr = np.clip(x, -15.0, 15.0)
    return 1.0 / (1.0 + np.exp(-arr))


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    mx = np.max(logits, axis=axis, keepdims=True)
    ex = np.exp(np.clip(logits - mx, -30.0, 30.0))
    return ex / np.maximum(1e-9, np.sum(ex, axis=axis, keepdims=True))


def extract_v14_op_features(
    snapshot: NativeSnapshot,
    base_eval: EvalResult,
    bay_delta: dict[str, float],
    succ_counts: dict[str, int],
) -> tuple[list[Any], np.ndarray]:
    """Extract causal 16-D `(unstarted_ops, [N_ops, 16])` matrix including execution-drift risk."""
    unstarted = [op for op in snapshot.operations if not op.started]
    if not unstarted:
        return [], np.zeros((0, V14_OP_FEAT_DIM), dtype=np.float32)

    fb = getattr(snapshot, "execution_feedback", None)
    fb_states = getattr(fb, "robot_states", {}) if fb is not None else {}
    t_map = {t.task_id: t for t in snapshot.tasks}
    t_now = float(snapshot.tick)

    bay_load: dict[str, int] = {}
    crew_load: dict[int, int] = {}
    for op in unstarted:
        bay_load[op.bay_id] = bay_load.get(op.bay_id, 0) + 1
        crew_load[op.crew_id] = crew_load.get(op.crew_id, 0) + 1

    rows: list[np.ndarray] = []
    for op in unstarted:
        st_w = _compute_op_window_stats(op, snapshot, base_eval, bay_delta, succ_counts)
        drift_hits = 0.0
        n_kits = max(1, len(op.kit_task_ids))
        for tid in op.kit_task_ids:
            t = t_map.get(tid)
            if t is None or t.status.lower() == "completed":
                continue
            aid = t.assigned_agv
            if aid and aid in fb_states:
                st_r = fb_states[aid]
                if float(getattr(st_r, "speed_scale", 1.0)) < 0.88 or int(getattr(st_r, "eta_drift_steps", 0)) > 0:
                    drift_hits += 1.0
        alt_bays = max(0, len(getattr(op, "eligible_bays", ())) - 1)
        alt_crews = max(0, len(getattr(op, "eligible_crews", ())) - 1)
        has_win = 1.0 if getattr(op, "resource_windows", ()) else 0.0
        st0 = float(base_eval.op_start.get(op.op_id, t_now + 20.0))
        rows.append(
            np.asarray(
                [
                    float(st_w["missing_frac"]),
                    float(np.clip(st_w["spread"] / 20.0, 0.0, 2.5)),
                    float(np.clip(st_w["win_margin"] / 20.0, -2.0, 2.5)),
                    float(st_w["near_cliff"]),
                    float(np.clip(st_w["early_waste"] / 15.0, 0.0, 2.5)),
                    float(st_w["is_urg"]),
                    float(np.clip(st_w["succ_cnt"] / 3.0, 0.0, 2.5)),
                    float(np.clip(st_w["slack"] / 30.0, -2.5, 2.5)),
                    float(np.clip(st_w["bay_dd"] / 15.0, 0.0, 2.5)),
                    float(drift_hits) / float(n_kits),
                    has_win,
                    float(min(3, alt_bays)) / 3.0,
                    float(min(3, alt_crews)) / 3.0,
                    float(min(4, bay_load.get(op.bay_id, 1) - 1)) / 4.0,
                    float(min(4, crew_load.get(op.crew_id, 1) - 1)) / 4.0,
                    float(np.clip((st0 - t_now) / 40.0, 0.0, 3.0)),
                ],
                dtype=np.float32,
            )
        )
    return unstarted, np.stack(rows, axis=0)


def augment_candidate_features_v14(
    cand: KittingJointCandidate,
    snapshot: NativeSnapshot,
    base_eval: EvalResult,
    bay_delta: dict[str, float],
    rem_budget_ratio: float = 1.0,
) -> np.ndarray:
    """Extend 24-D candidate vector to 32-D with execution-drift and contention relief features."""
    fb = getattr(snapshot, "execution_feedback", None)
    fb_states = getattr(fb, "robot_states", {}) if fb is not None else {}
    t_map = {t.task_id: t for t in snapshot.tasks}

    drift_assigned = 0.0
    n_dest = max(1, len(cand.destroy_tids))
    for tid in cand.destroy_tids:
        aid = cand.assignment.get(tid)
        if aid and aid in fb_states:
            st_r = fb_states[aid]
            if float(getattr(st_r, "speed_scale", 1.0)) < 0.88 or int(getattr(st_r, "eta_drift_steps", 0)) > 0:
                drift_assigned += 1.0

    # Choke-hub corridor hold + starvation relief proxy from candidate paths
    choke_wait_delta = 0.0
    if base_eval.valid and cand.eval_res.valid:
        def _path_hold_steps(ev_obj: EvalResult) -> int:
            tot = 0
            for pts in ev_obj.robot_paths.values():
                for idx in range(len(pts) - 1):
                    if pts[idx][0] == pts[idx + 1][0] and pts[idx + 1][1] > pts[idx][1]:
                        tot += int(pts[idx + 1][1] - pts[idx][1])
            return tot

        hold_delta = float(_path_hold_steps(base_eval) - _path_hold_steps(cand.eval_res))
        starv_delta = float(base_eval.starvation_ticks - cand.eval_res.starvation_ticks)
        choke_wait_delta = (hold_delta + 0.5 * starv_delta) / 15.0

    has_res_win = 1.0 if any(getattr(op, "resource_windows", ()) for op in snapshot.operations if not op.started) else 0.0
    det_norm = float(np.clip(cand.det_score / 4.0, -3.0, 4.0)) if cand.eval_res.valid else -3.0
    is_keep = 1.0 if cand.source_kind == "keep_default" else 0.0
    is_swap = 1.0 if "swap" in cand.source_kind else 0.0
    is_rescue = 1.0 if "drift" in cand.source_kind else 0.0

    extra = np.asarray(
        [
            1.0 if cand.eval_res.valid else 0.0,
            det_norm,
            float(drift_assigned) / float(n_dest),
            float(np.clip(choke_wait_delta, -2.0, 2.0)),
            has_res_win,
            is_keep,
            is_swap,
            float(np.clip(rem_budget_ratio + 0.25 * is_rescue, 0.0, 1.5)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([np.asarray(cand.features, dtype=np.float32), extra], axis=0)


class V14JointPolicy:
    """CPU/NumPy ultra-fast inference engine matching RTX 5060 PyTorch Set-Attention weights."""

    def __init__(self, weights: Optional[dict[str, np.ndarray]] = None, *, conservative_gate: float = 0.45) -> None:
        self.w = weights
        self.conservative_gate = float(conservative_gate)

    @classmethod
    def from_npz(cls, path: str | Path = DEFAULT_V14_CKPT) -> "V14JointPolicy":
        """Load trained V14 Set-Attention & Anytime controller weights from `.npz` checkpoint."""
        data = np.load(str(path), allow_pickle=False)
        w = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
        return cls(weights=w)

    def score_operations(
        self,
        op_feats: np.ndarray,
        global_feats: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict per-operation `(protect_logits, yield_logits)` over all unstarted ops."""
        n = int(op_feats.shape[0])
        if n == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        if self.w is None or "op_w1" not in self.w:
            # Well-calibrated structural prior when checkpoint unavailable
            prot = (
                1.8 * op_feats[:, 3]        # near_cliff
                + 1.1 * op_feats[:, 6]      # downstream DAG succ
                + 0.9 * op_feats[:, 5]      # urgency
                + 0.7 * op_feats[:, 9]      # kit drift exposure
                - 0.8 * op_feats[:, 4]      # early wait waste
                - 0.6 * op_feats[:, 7]      # low slack -> protect
            )
            yld = (
                1.6 * op_feats[:, 4]        # high early waste -> safe to defer
                + 0.9 * op_feats[:, 7]      # high positive slack
                - 1.5 * op_feats[:, 3]      # never yield a near-cliff op
                - 1.2 * op_feats[:, 5]      # never yield urgent op
            )
            return prot.astype(np.float32), yld.astype(np.float32)

        w = self.w
        g_rep = np.broadcast_to(global_feats[:KITTING_GLOBAL_FEAT_DIM][None, :], (n, KITTING_GLOBAL_FEAT_DIM))
        raw = np.concatenate([op_feats, g_rep], axis=-1)
        x = (raw - w["op_mu"]) / np.maximum(1e-5, w["op_sd"])
        h = _relu(x @ w["op_w1"] + w["op_b1"])
        # Mean-pooled fleet-wide cross-op context injection
        ctx = np.broadcast_to(np.mean(h, axis=0, keepdims=True), h.shape)
        h2 = _relu(np.concatenate([h, ctx], axis=-1) @ w["op_w2"] + w["op_b2"])
        logits = h2 @ w["op_head"] + w["op_hb"]
        return logits[:, 0].astype(np.float32), logits[:, 1].astype(np.float32)

    def score_candidates(
        self,
        global_feats: np.ndarray,
        cands: list[KittingJointCandidate],
        snapshot: NativeSnapshot,
        base_eval: EvalResult,
        bay_delta: dict[str, float],
        *,
        rem_budget_ratio: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """Rank candidates via cross-candidate self-attention + conservative DET safety gate."""
        m = len(cands)
        if m == 0:
            return np.zeros((0,), dtype=np.float32), 0.0
        det_vec = np.asarray([score_det_kitting_rule(c, base_eval) if c.det_score == 0.0 and c.source_kind != "keep_default" else c.det_score for c in cands], dtype=np.float32)
        if self.w is None or "cand_w1" not in self.w:
            for i, c in enumerate(cands):
                if not c.eval_res.valid:
                    det_vec[i] = -1e9
            return det_vec, 0.65

        w = self.w
        g = np.asarray(global_feats, dtype=np.float32)[:KITTING_GLOBAL_FEAT_DIM]
        mat = np.stack(
            [
                np.concatenate(
                    [augment_candidate_features_v14(c, snapshot, base_eval, bay_delta, rem_budget_ratio), g],
                    axis=0,
                )
                for c in cands
            ],
            axis=0,
        )
        x = (mat - w["c_mu"]) / np.maximum(1e-5, w["c_sd"])
        h = _relu(x @ w["cand_w1"] + w["cand_b1"])  # [M, d]
        # Single-head self-attention across all M competing joint candidates
        q = h @ w["att_q"]
        k = h @ w["att_k"]
        v = h @ w["att_v"]
        scale = 1.0 / np.sqrt(float(q.shape[-1]))
        att_w = _softmax((q @ k.T) * scale, axis=-1)
        h_att = _relu(np.concatenate([h, att_w @ v], axis=-1) @ w["cand_w2"] + w["cand_b2"])
        delta = (h_att @ w["cand_w3"] + w["cand_b3"]).reshape(-1)

        # Learned global trust gate in [0, conservative_gate * 3.0] modulating DET prior vs RL residual
        g_norm = (g - w["g_mu"]) / np.maximum(1e-5, w["g_sd"])
        hg = _relu(g_norm @ w["g_w1"] + w["g_b1"])
        trust_gain = float(_sigmoid((hg @ w["g_w2"] + w["g_b2"]).reshape(-1)[0])) * 2.4 * self.conservative_gate

        adj = np.tanh(delta / 2.0) * trust_gain
        for i, c in enumerate(cands):
            if c.source_kind == "keep_default":
                adj[i] = 0.0
        scores = det_vec + adj
        for i, c in enumerate(cands):
            if not c.eval_res.valid:
                scores[i] = -1e9
        best_i = int(np.argmax(scores))
        cont_logit = float((h_att[best_i : best_i + 1] @ w["cw"] + w["cb"]).reshape(-1)[0]) if "cw" in w else 0.0
        return scores.astype(np.float32), float(_sigmoid(cont_logit))

    def select_anytime_operator(
        self,
        global_feats: np.ndarray,
        search_state: np.ndarray,
        *,
        allowed_ops: tuple[int, ...] = (0, 1, 2, 3),
    ) -> int:
        """Select anytime operator `{0: COMMIT, 1: LOCAL_1OPT, 2: BAY_CPSAT, 3: HETERO_V8}`."""
        if not allowed_ops:
            return 0
        if float(search_state[0]) <= 0.08:
            return 0
        if self.w is None or "any_w1" not in self.w:
            # Heuristic fallback matching calibrated anytime thresholds
            has_win = float(search_state[4]) > 0.5
            has_dist = float(search_state[5]) > 0.5
            stall = float(search_state[3])
            if stall >= 0.65 and float(search_state[2]) <= 0.02 and not has_dist:
                return 0
            if has_win and 2 in allowed_ops:
                return 2
            if 3 in allowed_ops and float(search_state[0]) >= 0.35:
                return 3
            return 1 if 1 in allowed_ops else 0

        w = self.w
        vec = np.concatenate([global_feats[:KITTING_GLOBAL_FEAT_DIM], search_state[:V14_SEARCH_FEAT_DIM]], axis=0)
        xn = (vec - w["any_mu"]) / np.maximum(1e-5, w["any_sd"])
        h = _relu(xn @ w["any_w1"] + w["any_b1"])
        logits = (h @ w["any_w2"] + w["any_b2"]).reshape(-1)
        best_op = max(allowed_ops, key=lambda idx: float(logits[idx]))
        return int(best_op)


def load_v14_joint_policy(path: Optional[str | Path] = None) -> V14JointPolicy:
    """Load trained `V14JointPolicy` if available, else deterministic structural attention prior."""
    p = Path(path) if path is not None else DEFAULT_V14_CKPT
    if p.exists():
        return V14JointPolicy.from_npz(p)
    return V14JointPolicy(weights=None)


def generate_v14_joint_candidates(
    network: Network,
    snapshot: NativeSnapshot,
    clf: CommitmentClassification,
    evaluator: CVCREvaluator,
    surr: Any,
    base_assign: dict[str, str],
    base_bays: dict[str, str],
    base_crews: dict[str, int],
    base_eval: EvalResult,
    base_op_prio: dict[str, int],
    healthy_robots: list[str],
    bay_delta: Optional[dict[str, float]] = None,
    policy: Optional[V14JointPolicy] = None,
    *,
    lambda_change: float = 15.0,
) -> list[KittingJointCandidate]:
    """Build combined V13 base candidates PLUS RL-proposed Cross-Op Swap & Drift-Rescue neighborhoods."""
    cands = generate_kitting_joint_candidates(
        network,
        snapshot,
        clf,
        evaluator,
        surr,
        base_assign,
        base_bays,
        base_crews,
        base_eval,
        base_op_prio,
        healthy_robots,
        bay_delta,
        max_protected_ops=3,
        lambda_change=lambda_change,
        include_learned_proposal=True,
    )
    if not healthy_robots or surr is None:
        return cands

    b_delta = dict(bay_delta or {})
    tasks_by_id = {t.task_id: t for t in snapshot.tasks}
    op_map = {op.op_id: op for op in snapshot.operations}
    succ_counts: dict[str, int] = {op.op_id: 0 for op in snapshot.operations}
    for op in snapshot.operations:
        for pid in op_predecessors(op):
            if pid in succ_counts:
                succ_counts[pid] += 1

    unstarted, op_feats = extract_v14_op_features(snapshot, base_eval, b_delta, succ_counts)
    if len(unstarted) < 1:
        return cands

    g_feats = extract_v13_global_features(snapshot, base_eval, clf, b_delta)
    pol = policy or V14JointPolicy(weights=None)
    prot_logits, yield_logits = pol.score_operations(op_feats, g_feats)

    releasable = [
        t.task_id
        for t in snapshot.tasks
        if t.status.lower() in ("pending", "assigned")
        and t.task_id in base_assign
        and (
            t.assigned_agv not in healthy_robots
            or t.assigned_agv is None
            or t.task_id in clf.releasable_tasks
            or len(clf.releasable_tasks) == 0
        )
    ]
    rel_set = set(releasable)
    if not rel_set:
        return cands

    seen_sigs = {
        (
            tuple(sorted(c.assignment.items())),
            tuple(sorted(c.bay_overrides.items())),
            tuple(sorted(c.crew_overrides.items())),
            tuple((o.op_id, c.op_priority.get(o.op_id, 0)) for o in snapshot.operations if not o.started),
        )
        for c in cands
    }

    def _try_append(
        cid: str,
        skind: str,
        tgt_oid: Optional[str],
        prio: dict[str, int],
        bays: dict[str, str],
        crews: dict[str, int],
        nbr: list[str],
        focus_set: set[str],
    ) -> None:
        if not nbr:
            return
        fixed_sub = {k: v for k, v in base_assign.items() if k not in set(nbr)}
        old_p = getattr(surr, "op_prio", None)
        surr.op_prio = dict(prio)
        assign_n = surr.regret_k_assignment(healthy_robots, fixed_sub, nbr, op_bays=bays)
        if old_p is not None:
            surr.op_prio = old_p

        orders: dict[str, list[str]] = {}
        for rid in healthy_robots:
            rt = [
                t for t in snapshot.tasks
                if t.status.lower() != "completed" and assign_n.get(t.task_id) == rid
            ]
            if rt:
                rt.sort(
                    key=lambda t: (
                        0 if t.status.lower() in ("picked", "loading", "to_dropoff", "unloading") else 1,
                        0 if t.task_id in focus_set else 1,
                        prio.get(t.kit_op_id or "", 0),
                        t.release_step,
                        t.task_id,
                    )
                )
                orders[rid] = [t.task_id for t in rt]

        ev_n = evaluator.evaluate(
            assign_n,
            op_priority=prio,
            op_bay_overrides=bays,
            op_crew_overrides=crews,
            task_order_override=orders or None,
        )
        if not ev_n.valid:
            return
        sig = (
            tuple(sorted(assign_n.items())),
            tuple(sorted(bays.items())),
            tuple(sorted(crews.items())),
            tuple((o.op_id, prio.get(o.op_id, 0)) for o in snapshot.operations if not o.started),
        )
        if sig in seen_sigs:
            return
        seen_sigs.add(sig)
        feats, adv_s, yld_s, cliff = extract_kitting_candidate_features(
            snapshot, base_assign, base_bays, base_crews, base_eval,
            assign_n, bays, crews, ev_n, tgt_oid, skind, b_delta, succ_counts,
            lambda_change=lambda_change,
        )
        c = KittingJointCandidate(
            cand_id=cid,
            source_kind=skind,
            target_op_id=tgt_oid,
            op_priority=dict(prio),
            assignment=dict(assign_n),
            bay_overrides=dict(bays),
            crew_overrides=dict(crews),
            task_order_override=orders or None,
            destroy_tids=tuple(nbr),
            eval_res=ev_n,
            features=feats,
            yielded_op_steps=yld_s,
            advanced_op_steps=adv_s,
            discrete_cliff_saved=cliff,
        )
        c.det_score = score_det_kitting_rule(c, base_eval)
        cands.append(c)

    min_prio_val = min((base_op_prio.get(o.op_id, 0) for o in unstarted), default=0)
    has_res_win = any(bool(getattr(o, "resource_windows", ())) for o in unstarted)

    # Layer 2 Proposal A: RL Cross-Op Protect-vs-Yield Joint Neighborhood (only when resource window or real cliff pressure exists)
    if len(unstarted) >= 2 and has_res_win:
        p_rank = sorted(range(len(unstarted)), key=lambda i: -float(prot_logits[i]))
        y_rank = sorted(range(len(unstarted)), key=lambda i: -float(yield_logits[i]))
        prot_idx = p_rank[0]
        yield_idx = next((idx for idx in y_rank if idx != prot_idx), None)
        if yield_idx is not None:
            op_p = unstarted[prot_idx]
            op_y = unstarted[yield_idx]
            anc = _ancestor_chain(op_p.op_id, op_map)
            prot_ids = {op_p.op_id} | set(anc)
            p_prio = dict(base_op_prio)
            for oid in prot_ids:
                if oid in op_map and not op_map[oid].started:
                    p_prio[oid] = min_prio_val - 48
            p_prio[op_y.op_id] = min_prio_val + 70
            p_bays = dict(base_bays)
            p_crews = dict(base_crews)
            nb, nc = _best_bay_and_crew_for_op(op_p, snapshot, base_bays, base_crews, b_delta, tasks_by_id)
            p_bays[op_p.op_id] = nb
            p_crews[op_p.op_id] = nc
            elig_c_y = list(getattr(op_y, "eligible_crews", ()) or ())
            if len(elig_c_y) > 1 and int(p_crews.get(op_y.op_id, op_y.crew_id)) == int(nc):
                alt_c = [int(c) for c in elig_c_y if int(c) != int(nc)]
                if alt_c:
                    p_crews[op_y.op_id] = alt_c[0]
            focus = [tid for oid in ([op_p.op_id] + anc) if oid in op_map for tid in op_map[oid].kit_task_ids if tid in rel_set]
            yield_tids = [tid for tid in op_y.kit_task_ids if tid in rel_set and tid not in focus]
            nbr = focus[:5] + yield_tids[:3]
            _try_append(f"rl_swap_{op_p.op_id}_over_{op_y.op_id}", "rl_swap_joint", op_p.op_id, p_prio, p_bays, p_crews, nbr, set(focus))

    # Layer 2 Proposal B: Execution-Drift / Choke Kit Rescue Neighborhood
    fb = getattr(snapshot, "execution_feedback", None)
    fb_states = getattr(fb, "robot_states", {}) if fb is not None else {}
    drifted_tids = [
        tid
        for tid in releasable
        if (aid := base_assign.get(tid)) is not None
        and aid in fb_states
        and (
            float(getattr(fb_states[aid], "speed_scale", 1.0)) < 0.88
            or int(getattr(fb_states[aid], "eta_drift_steps", 0)) > 0
            or aid in clf.disturbed_robots
        )
    ]
    if drifted_tids:
        top_p_op = unstarted[int(np.argmax(prot_logits))]
        focus_d = {tid for tid in top_p_op.kit_task_ids if tid in rel_set} | set(drifted_tids[:4])
        nbr_d = list(dict.fromkeys(drifted_tids[:4] + [tid for tid in top_p_op.kit_task_ids if tid in rel_set]))[:6]
        d_prio = dict(base_op_prio)
        d_prio[top_p_op.op_id] = min_prio_val - 40
        _try_append(f"rl_drift_{top_p_op.op_id}", "rl_drift_rescue", top_p_op.op_id, d_prio, dict(base_bays), dict(base_crews), nbr_d, focus_d)

    return cands
