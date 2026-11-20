"""Production-Kitting Window Protection & Joint-Repair Neighborhood RL (V13).

Implements hierarchical production-to-logistics joint repair selection inside strict
causal snapshot & atomic transaction guarantees:
  - Stage 1 (Target Operation + Joint Neighborhood Proposal): builds legally distinct
    joint neighborhoods (`op*` priority + missing kitting tasks + contending choke
    neighbor + alternate bay/crew + queue-head promotion) on top of `CVCREvaluator(B2)`.
  - Stage 2 (Causal 24-D Feature Representation): measures discrete window cliffs saved,
    operation start advancement, yielded-order cost, downstream DAG unlock value,
    Detour-Delta choke contention, and assignment churn with zero future leakage.
  - Stage 3 (Equal-Pool Policies & DR-ALNS Continue Gate): compares deterministic
    kitting criticality (`det_rule`), supervised counterfactual ranker (`sup_ranker`),
    and multi-step SMDP RL policy (`smdp_rl`) over the identical candidate pool & clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.checkpoints import resolve_default_checkpoint
from amr_sched.cvcr.commitment import CommitmentClassification
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.snapshot import (
    NativeSnapshot,
    align_start_to_windows,
    op_predecessors,
)
from amr_sched.cvcr.v12_proactive_guard import compute_kitting_marginal_start_value
from amr_sched.network import Network

KITTING_CAND_FEAT_DIM = 24
KITTING_GLOBAL_FEAT_DIM = 12
DEFAULT_V13_SUP_CKPT = resolve_default_checkpoint("v13_kitting_sup.npz")
DEFAULT_V13_SMDP_CKPT = resolve_default_checkpoint("v13_kitting_smdp.npz")


@dataclass
class KittingJointCandidate:
    """Single transaction-verifiable joint repair candidate evaluated under CVCREvaluator."""

    cand_id: str
    source_kind: str  # "keep_default" | "det_hybrid_crit" | "protect_op_joint" | "learned_swap_joint"
    target_op_id: Optional[str]
    op_priority: dict[str, int]
    assignment: dict[str, str]
    bay_overrides: dict[str, str]
    crew_overrides: dict[str, int]
    task_order_override: Optional[dict[str, list[str]]]
    destroy_tids: tuple[str, ...]
    eval_res: EvalResult
    features: np.ndarray
    det_score: float = 0.0
    yielded_op_steps: float = 0.0
    advanced_op_steps: float = 0.0
    discrete_cliff_saved: bool = False


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def extract_v13_global_features(
    snapshot: NativeSnapshot,
    base_eval: EvalResult,
    clf: CommitmentClassification,
    bay_detour_delta: Optional[dict[str, float]] = None,
) -> np.ndarray:
    """Extract 12-D strictly causal global state summary for candidate conditioning & continue gate."""
    t_now = float(snapshot.tick)
    ops = [op for op in snapshot.operations if not op.started]
    n_ops = max(1, len(snapshot.operations))
    n_un = max(1, len(ops))

    n_unassigned = sum(
        1 for t in snapshot.tasks if t.status.lower() != "completed" and t.assigned_agv is None
    )
    n_picked = sum(1 for t in snapshot.tasks if t.status.lower() == "picked")
    n_active_t = max(
        1,
        sum(1 for t in snapshot.tasks if t.status.lower() != "completed"),
    )

    slacks: list[float] = []
    window_cliffs = 0
    for op in ops:
        st = float(base_eval.op_start.get(op.op_id, t_now + 20.0))
        en = float(base_eval.op_end.get(op.op_id, st + op.proc_steps))
        slacks.append(float(op.due_step - en) / 30.0)
        r_wins = getattr(op, "resource_windows", ())
        if r_wins:
            est_kit = max(
                (float(base_eval.order_unload_ticks.get(tid, t_now + 50.0)) for tid in op.kit_task_ids),
                default=t_now,
            )
            delayed = align_start_to_windows(int(round(est_kit + 5.0)), r_wins)
            if delayed - int(round(st)) >= 15:
                window_cliffs += 1

    min_slack = float(min(slacks)) if slacks else 1.0
    mean_slack = float(np.mean(slacks)) if slacks else 1.0
    bay_d = bay_detour_delta or {}
    mean_dd = float(np.mean(list(bay_d.values()))) / 20.0 if bay_d else 0.0
    blocked_cnt = sum(1 for r in snapshot.robots if r.state.lower() == "blocked")
    return np.asarray(
        [
            float(len(clf.disturbed_robots)) / max(1.0, float(len(snapshot.robots))),
            float(n_unassigned) / float(n_active_t),
            float(n_picked) / float(n_active_t),
            float(len(ops)) / float(n_ops),
            float(np.clip(min_slack, -2.0, 2.0)),
            float(np.clip(mean_slack, -2.0, 2.0)),
            float(window_cliffs) / float(n_un),
            float(base_eval.starvation_ticks) / 40.0,
            float(np.clip(mean_dd, 0.0, 2.0)),
            float(blocked_cnt) / max(1.0, float(len(snapshot.robots))),
            t_now / 300.0,
            float(np.clip(base_eval.objective / 2000.0, 0.0, 5.0)),
        ],
        dtype=np.float32,
    )


def _compute_op_window_stats(
    op: Any,
    snapshot: NativeSnapshot,
    eval_res: EvalResult,
    bay_delta: dict[str, float],
    succ_counts: dict[str, int],
) -> dict[str, float]:
    """Compute strictly causal readiness, resource window margin & slack for `op`."""
    t_now = float(snapshot.tick)
    tasks_by_id = {t.task_id: t for t in snapshot.tasks}
    etas: list[float] = []
    n_missing = 0
    for tid in op.kit_task_ids:
        t = tasks_by_id.get(tid)
        if t is None or t.status.lower() != "completed":
            n_missing += 1
        etas.append(float(eval_res.order_unload_ticks.get(tid, t_now + 50.0)))

    max_kit = max(etas) if etas else t_now
    spread = (max(etas) - min(etas)) if len(etas) >= 2 else 0.0
    st = float(eval_res.op_start.get(op.op_id, max_kit))
    en = float(eval_res.op_end.get(op.op_id, st + op.proc_steps))
    r_ready = float(eval_res.op_res_ready.get(op.op_id, t_now))

    r_wins = tuple(getattr(op, "resource_windows", ()) or ())
    win_close = 9999.0
    win_open = 0.0
    if r_wins:
        for ws, we in sorted((int(a), int(b)) for a, b in r_wins):
            if r_ready <= float(we):
                win_open = float(ws)
                win_close = float(we)
                break
        if win_close >= 9000.0 and r_wins:
            last_we = max(int(b) for _, b in r_wins)
            win_open = float(last_we + 40)
            win_close = 9999.0

    # Wait when res / window isn't even open yet -> speeding up cargo early saves 0 steps
    eff_open = max(r_ready, win_open)
    early_waste = max(0.0, eff_open - max_kit)
    win_margin = (win_close - max_kit) if win_close < 9000.0 else 40.0
    near_cliff = 1.0 if (win_close < 9000.0 and -8.0 <= win_margin <= 12.0) else 0.0
    # Resource contention cliff even without explicit windows (shared crew/bay queue jump)
    if win_close >= 9000.0:
        slack_to_due = float(op.due_step) - en
        if -12.0 <= slack_to_due <= 10.0:
            near_cliff = max(near_cliff, 0.5)

    return {
        "max_kit": max_kit,
        "spread": spread,
        "missing_frac": float(n_missing) / float(max(1, len(op.kit_task_ids))),
        "early_waste": early_waste,
        "win_margin": win_margin,
        "near_cliff": near_cliff,
        "is_urg": 1.0 if op.is_urgent else 0.0,
        "succ_cnt": float(succ_counts.get(op.op_id, 0)),
        "slack": float(op.due_step) - en,
        "bay_dd": float(bay_delta.get(op.bay_id, 0.0)),
    }


def extract_kitting_candidate_features(
    snapshot: NativeSnapshot,
    base_assign: dict[str, str],
    base_bays: dict[str, str],
    base_crews: dict[str, int],
    base_eval: EvalResult,
    cand_assign: dict[str, str],
    cand_bays: dict[str, str],
    cand_crews: dict[str, int],
    cand_eval: EvalResult,
    target_op_id: Optional[str],
    source_kind: str,
    bay_delta: dict[str, float],
    succ_counts: dict[str, int],
    rem_budget_ratio: float = 1.0,
    lambda_change: float = 15.0,
) -> tuple[np.ndarray, float, float, bool]:
    """Compute 24-D causal feature vector `(feats, advanced_steps, yielded_steps, cliff_saved)`."""
    t_map = {t.task_id: t for t in snapshot.tasks}
    snap_assign = {t.task_id: t.assigned_agv for t in snapshot.tasks if t.assigned_agv is not None}

    churn_cnt = 0.0
    for tid, rid in cand_assign.items():
        t = t_map.get(tid)
        if t is not None and t.status.lower() == "assigned" and tid in snap_assign and snap_assign[tid] != rid:
            churn_cnt += 1.0
    bay_sw = sum(1 for oid, b in cand_bays.items() if b != base_bays.get(oid))
    crew_sw = sum(1 for oid, c in cand_crews.items() if c != base_crews.get(oid))
    churn_total = churn_cnt + 0.35 * float(bay_sw + crew_sw)

    adv_steps = 0.0
    yield_steps = 0.0
    yielded_ops = 0
    cliff_saved = False
    target_adv = 0.0
    op_map = {op.op_id: op for op in snapshot.operations}

    for op in snapshot.operations:
        if op.started:
            continue
        oid = op.op_id
        s0 = float(base_eval.op_start.get(oid, snapshot.tick + 50))
        s1 = float(cand_eval.op_start.get(oid, snapshot.tick + 50))
        delta = s0 - s1
        has_win = bool(getattr(op, "resource_windows", ()))
        crossed_due_or_win = (
            (has_win and delta >= 12.0)
            or (
                not has_win
                and delta >= 15.0
                and op.due_step > 0
                and s0 > float(op.due_step - op.proc_steps) >= s1
            )
        )
        if delta > 0.0:
            adv_steps += delta
            if crossed_due_or_win:
                cliff_saved = True
        elif delta < 0.0:
            yield_steps += -delta
            yielded_ops += 1
        if target_op_id == oid:
            target_adv = delta

    unstarted = [op for op in snapshot.operations if not op.started]
    n_un = max(1, len(unstarted))
    tgt_op = op_map.get(target_op_id or "")
    if tgt_op is None and unstarted:
        # Pick the operation whose start improved most (or most urgent)
        tgt_op = max(
            unstarted,
            key=lambda o: (
                float(base_eval.op_start.get(o.op_id, 0) - cand_eval.op_start.get(o.op_id, 0)),
                1 if o.is_urgent else 0,
                -o.due_step,
            ),
        )
    if tgt_op is not None:
        os_st = _compute_op_window_stats(tgt_op, snapshot, base_eval, bay_delta, succ_counts)
    else:
        os_st = {
            "max_kit": float(snapshot.tick),
            "spread": 0.0,
            "missing_frac": 0.0,
            "early_waste": 0.0,
            "win_margin": 20.0,
            "near_cliff": 0.0,
            "is_urg": 0.0,
            "succ_cnt": 0.0,
            "slack": 10.0,
            "bay_dd": 0.0,
        }

    obj_gain = float(base_eval.objective - cand_eval.objective)
    base_churn = float(
        sum(
            1
            for tid, rid in base_assign.items()
            if (t := t_map.get(tid)) is not None
            and t.status.lower() == "assigned"
            and tid in snap_assign
            and snap_assign[tid] != rid
        )
    )
    net_gain = obj_gain - float(lambda_change) * (churn_cnt - base_churn)

    base_tard = sum(
        (12.0 if op.is_urgent else 4.0) * max(0.0, float(base_eval.op_end.get(op.op_id, 999) - op.due_step))
        for op in snapshot.operations
    )
    cand_tard = sum(
        (12.0 if op.is_urgent else 4.0) * max(0.0, float(cand_eval.op_end.get(op.op_id, 999) - op.due_step))
        for op in snapshot.operations
    )
    base_mk = max((float(base_eval.op_end.get(o.op_id, 0)) for o in snapshot.operations), default=0.0)
    cand_mk = max((float(cand_eval.op_end.get(o.op_id, 0)) for o in snapshot.operations), default=0.0)

    feats = np.asarray(
        [
            float(np.clip(obj_gain / 100.0, -3.0, 5.0)),
            float(np.clip(adv_steps / 25.0, -2.0, 5.0)),
            float(np.clip(target_adv / 25.0, -2.0, 5.0)),
            1.0 if cliff_saved else 0.0,
            float(np.clip(yield_steps / 25.0, 0.0, 5.0)),
            float(yielded_ops) / float(n_un),
            float(np.clip(churn_total / 6.0, 0.0, 3.0)),
            float(np.clip(net_gain / 100.0, -3.0, 5.0)),
            float(os_st["missing_frac"]),
            float(np.clip(os_st["spread"] / 20.0, 0.0, 2.0)),
            float(np.clip(os_st["win_margin"] / 20.0, -1.5, 2.0)),
            float(os_st["near_cliff"]),
            1.0 if os_st["early_waste"] >= 6.0 else 0.0,
            float(os_st["is_urg"]),
            float(np.clip(os_st["succ_cnt"] / 3.0, 0.0, 2.0)),
            float(np.clip(os_st["slack"] / 30.0, -2.0, 2.0)),
            float(np.clip(os_st["bay_dd"] / 15.0, 0.0, 2.0)),
            1.0 if bay_sw > 0 else 0.0,
            1.0 if crew_sw > 0 else 0.0,
            float(np.clip((base_eval.starvation_ticks - cand_eval.starvation_ticks) / 15.0, -2.0, 3.0)),
            float(np.clip((base_tard - cand_tard) / 30.0, -3.0, 5.0)),
            float(np.clip((base_mk - cand_mk) / 15.0, -2.0, 3.0)),
            1.0 if source_kind == "det_hybrid_crit" else 0.0,
            1.0 if source_kind in ("protect_op_joint", "learned_swap_joint") else 0.0,
        ],
        dtype=np.float32,
    )
    if feats.shape[0] != KITTING_CAND_FEAT_DIM:
        raise ValueError(f"Expected {KITTING_CAND_FEAT_DIM}-D feats, got {feats.shape}")
    return feats, float(adv_steps), float(yield_steps), bool(cliff_saved)


def score_det_kitting_rule(cand: KittingJointCandidate, base_eval: EvalResult) -> float:
    """Strong deterministic rule baseline scoring the identical candidate pool."""
    if not cand.eval_res.valid:
        return -1e9
    f = cand.features
    # Baseline keep_default gets exact 0.0 anchor so weak/noisy switches never oscillate
    if cand.source_kind == "keep_default":
        return 0.0
    net_g = float(f[7])
    churn_norm = float(f[6])
    # Require positive net gain after hysteresis on calm valid states unless catching a window cliff
    hys = 0.22 if (base_eval.valid and churn_norm > 0.05 and float(f[3]) < 0.5) else 0.0
    return float(
        1.25 * net_g              # net objective gain (already includes -15*churn)
        + 0.22 * float(f[1])      # total op start advancement
        + 0.18 * float(f[13])     # urgency bonus
        + 0.15 * float(f[14])     # downstream DAG unlock
        + 0.12 * float(f[19])     # starvation reduction
        - 0.28 * float(f[4])      # yielded other ops penalty
        - 0.30 * churn_norm       # churn anti-oscillation guard
        - hys
    )


class KittingWindowRanker:
    """Learned MLP ranker + anytime continue gate over V13 joint repair neighborhoods."""

    def __init__(self, weights: Optional[dict[str, np.ndarray]] = None, mode: str = "smdp_rl") -> None:
        self.w = weights
        self.mode = str(mode)

    @classmethod
    def from_npz(cls, path: str | Path, mode: str = "smdp_rl") -> "KittingWindowRanker":
        """Load trained supervised ranker or SMDP RL weights from `.npz` file."""
        data = np.load(str(path), allow_pickle=False)
        w = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
        return cls(weights=w, mode=mode)

    def score_candidates(
        self,
        global_feats: np.ndarray,
        cands: list[KittingJointCandidate],
    ) -> tuple[np.ndarray, float]:
        """Return `(per_cand_scores, continue_prob)` using shared row-wise MLP."""
        if not cands:
            return np.zeros((0,), dtype=np.float32), 0.0
        if self.w is None:
            sc = np.asarray([c.det_score for c in cands], dtype=np.float32)
            return sc, 0.5

        w = self.w
        g = np.asarray(global_feats, dtype=np.float32)[:KITTING_GLOBAL_FEAT_DIM]
        mat = np.stack([np.concatenate([c.features, g], axis=0) for c in cands], axis=0)
        x = (mat - w["mu"]) / np.maximum(1e-6, w["sd"])
        h = _relu(x @ w["w1"] + w["b1"])
        h = _relu(h @ w["w2"] + w["b2"])
        raw = (h @ w["w3"] + w["b3"]).reshape(-1)
        det_vec = np.asarray([c.det_score for c in cands], dtype=np.float32)
        # Residual scoring: tightly bounded (+/-1.0) adjustment preventing 100-group OOD variance
        raw_adj = 1.0 * np.tanh(raw / 2.0)
        for i, c in enumerate(cands):
            if c.source_kind == "keep_default":
                raw_adj[i] = 0.0
        blended = det_vec + raw_adj
        for i, c in enumerate(cands):
            if not c.eval_res.valid:
                blended[i] = -1e9
        best_i = int(np.argmax(blended))
        cont_logit = float((h[best_i : best_i + 1] @ w["cw"] + w["cb"]).reshape(-1)[0]) if "cw" in w else 0.0
        cont_prob = float(1.0 / (1.0 + np.exp(-np.clip(cont_logit, -15.0, 15.0))))
        return blended.astype(np.float32), cont_prob


def load_kitting_window_ranker(
    mode: str = "smdp_rl",
    path: Optional[str | Path] = None,
) -> KittingWindowRanker:
    """Load requested V13 ranker (`sup_ranker` | `smdp_rl` | `det_rule`), falling back gracefully."""
    if mode == "det_rule":
        return KittingWindowRanker(weights=None, mode="det_rule")
    if path is not None:
        p = Path(path)
    elif mode == "sup_ranker":
        p = DEFAULT_V13_SUP_CKPT
    else:
        p = DEFAULT_V13_SMDP_CKPT
    if p.exists():
        return KittingWindowRanker.from_npz(p, mode=mode)
    return KittingWindowRanker(weights=None, mode=mode)


def _best_bay_and_crew_for_op(
    op: Any,
    snapshot: NativeSnapshot,
    base_bays: dict[str, str],
    base_crews: dict[str, int],
    bay_delta: dict[str, float],
    tasks_by_id: dict[str, Any],
) -> tuple[str, int]:
    """Select best legal (bay, crew) for `op` when unpicked tasks allow diversion."""
    cur_bay = base_bays.get(op.op_id, op.bay_id)
    cur_crew = base_crews.get(op.op_id, op.crew_id)
    any_picked = any(
        (t := tasks_by_id.get(tid)) is not None
        and t.status.lower() in ("picked", "loading", "to_dropoff", "unloading", "completed")
        for tid in op.kit_task_ids
    )
    elig_bays = list(getattr(op, "eligible_bays", ()) or (cur_bay,))
    best_bay = cur_bay
    if (not any_picked) and len(elig_bays) > 1:
        bay_load: dict[str, int] = {}
        for o2 in snapshot.operations:
            if not o2.started and o2.op_id != op.op_id:
                b2 = base_bays.get(o2.op_id, o2.bay_id)
                bay_load[b2] = bay_load.get(b2, 0) + 1
        best_bay = min(
            elig_bays,
            key=lambda b: (
                bay_load.get(b, 0) * 6.0 + float(bay_delta.get(b, 0.0)),
                0 if b == cur_bay else 1,
                b,
            ),
        )

    elig_crews = list(getattr(op, "eligible_crews", ()) or (cur_crew,))
    best_crew = cur_crew
    if len(elig_crews) > 1:
        crew_load: dict[int, int] = {}
        for o2 in snapshot.operations:
            if not o2.started and o2.op_id != op.op_id:
                c2 = base_crews.get(o2.op_id, o2.crew_id)
                crew_load[c2] = crew_load.get(c2, 0) + 1
        best_crew = min(
            elig_crews,
            key=lambda c: (crew_load.get(int(c), 0), 0 if int(c) == int(cur_crew) else 1, int(c)),
        )
    return str(best_bay), int(best_crew)


def _ancestor_chain(op_id: str, op_map: dict[str, Any]) -> list[str]:
    res: list[str] = []
    seen = {op_id}
    queue = list(op_predecessors(op_map[op_id])) if op_id in op_map else []
    while queue:
        pid = queue.pop(0)
        if pid in seen or pid not in op_map:
            continue
        seen.add(pid)
        res.append(pid)
        queue.extend(op_predecessors(op_map[pid]))
    return res


def generate_kitting_joint_candidates(
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
    *,
    max_protected_ops: int = 3,
    lambda_change: float = 15.0,
    include_learned_proposal: bool = True,
) -> list[KittingJointCandidate]:
    """Generate distinct transaction-verifiable joint repair candidates on `CVCREvaluator(B2)`."""
    del network
    b_delta = dict(bay_delta or {})
    tasks_by_id = {t.task_id: t for t in snapshot.tasks}
    op_map = {op.op_id: op for op in snapshot.operations}

    succ_counts: dict[str, int] = {op.op_id: 0 for op in snapshot.operations}
    for op in snapshot.operations:
        for pid in op_predecessors(op):
            if pid in succ_counts:
                succ_counts[pid] += 1

    cands: list[KittingJointCandidate] = []
    seen_sigs: set[Any] = set()

    def _add_candidate(
        cid: str,
        skind: str,
        tgt_oid: Optional[str],
        op_prio: dict[str, int],
        assign: dict[str, str],
        bays: dict[str, str],
        crews: dict[str, int],
        t_orders: Optional[dict[str, list[str]]],
        destroy_tids: tuple[str, ...],
        ev: EvalResult,
    ) -> None:
        sig = (
            tuple(sorted(assign.items())),
            tuple(sorted(bays.items())),
            tuple(sorted(crews.items())),
            tuple((o.op_id, op_prio.get(o.op_id, 0)) for o in snapshot.operations if not o.started),
            tuple((r, tuple(v)) for r, v in sorted((t_orders or {}).items())),
        )
        if sig in seen_sigs and skind != "keep_default":
            return
        seen_sigs.add(sig)
        feats, adv_s, yld_s, cliff = extract_kitting_candidate_features(
            snapshot,
            base_assign,
            base_bays,
            base_crews,
            base_eval,
            assign,
            bays,
            crews,
            ev,
            tgt_oid,
            skind,
            b_delta,
            succ_counts,
            lambda_change=lambda_change,
        )
        c = KittingJointCandidate(
            cand_id=cid,
            source_kind=skind,
            target_op_id=tgt_oid,
            op_priority=dict(op_prio),
            assignment=dict(assign),
            bay_overrides=dict(bays),
            crew_overrides=dict(crews),
            task_order_override=({r: list(v) for r, v in t_orders.items()} if t_orders else None),
            destroy_tids=tuple(destroy_tids),
            eval_res=ev,
            features=feats,
            yielded_op_steps=yld_s,
            advanced_op_steps=adv_s,
            discrete_cliff_saved=cliff,
        )
        c.det_score = score_det_kitting_rule(c, base_eval)
        cands.append(c)

    # 0. Keep / Default baseline candidate
    _add_candidate(
        "keep_default",
        "keep_default",
        None,
        base_op_prio,
        base_assign,
        base_bays,
        base_crews,
        ({r: list(s) for r, s in base_eval.robot_order_seq.items() if s} or None),
        (),
        base_eval,
    )

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
    if not releasable:
        releasable = [
            t.task_id for t in snapshot.tasks if t.status.lower() in ("pending", "assigned") and t.task_id in base_assign
        ]
    rel_set = set(releasable)

    # 1. Deterministic Hybrid Criticality & Kitting Marginal Value baseline (using existing heuristic)
    unstarted = [op for op in snapshot.operations if not op.started]
    if healthy_robots and unstarted and surr is not None:
        kv_map = compute_kitting_marginal_start_value(snapshot)
        crit_prio = dict(base_op_prio)
        for op in unstarted:
            best_t_score = min(
                (kv_map[tid] for tid in op.kit_task_ids if tid in kv_map),
                default=(2, 1, 0, 0, int(op.due_step), ""),
            )
            # Lower op_priority executes earlier in production & routing
            crit_prio[op.op_id] = (
                int(best_t_score[0]) * 40
                + int(best_t_score[1]) * 25
                + int(best_t_score[2]) * 12
                + int(best_t_score[4])
            )
        crit_tasks = sorted(
            [tid for tid in rel_set if tid in tasks_by_id],
            key=lambda tid: (
                kv_map.get(tid, (2, 1, 0, 0, 999, tid))[:5],
                tid,
            ),
        )[:6]
        if crit_tasks:
            fixed_map = {k: v for k, v in base_assign.items() if k not in set(crit_tasks)}
            old_p = getattr(surr, "op_prio", None)
            surr.op_prio = dict(crit_prio)
            crit_assign = surr.regret_k_assignment(healthy_robots, fixed_map, crit_tasks, op_bays=base_bays)
            if old_p is not None:
                surr.op_prio = old_p
            ev_crit = evaluator.evaluate(
                crit_assign,
                op_priority=crit_prio,
                op_bay_overrides=base_bays,
                op_crew_overrides=base_crews,
            )
            if ev_crit.valid:
                _add_candidate(
                    "det_hybrid_crit",
                    "det_hybrid_crit",
                    None,
                    crit_prio,
                    crit_assign,
                    base_bays,
                    base_crews,
                    ({r: list(s) for r, s in ev_crit.robot_order_seq.items() if s} or None),
                    tuple(crit_tasks),
                    ev_crit,
                )

    # 2. Target-Operation Joint Repair Neighborhoods (`protect_op_joint::<op_id>`)
    scored_ops: list[tuple[tuple[Any, ...], Any, dict[str, float]]] = []
    for op in unstarted:
        st_w = _compute_op_window_stats(op, snapshot, base_eval, b_delta, succ_counts)
        has_rel_kit = any(tid in rel_set for tid in op.kit_task_ids)
        # Candidate ops include near-window-cliff ops, high-succ DAG blockers, and urgent/tardy ops
        if not has_rel_kit and len(getattr(op, "eligible_bays", ())) <= 1 and len(getattr(op, "eligible_crews", ())) <= 1:
            continue
        rank_key = (
            0 if st_w["near_cliff"] >= 0.5 else 1,
            0 if st_w["early_waste"] < 5.0 else 1,
            -int(st_w["succ_cnt"]),
            0 if st_w["is_urg"] > 0.5 else 1,
            float(op.due_step),
            op.op_id,
        )
        scored_ops.append((rank_key, op, st_w))

    scored_ops.sort(key=lambda x: x[0])
    min_prio_val = min((base_op_prio.get(o.op_id, 0) for o in unstarted), default=0)

    for _rk, tgt_op, st_w in scored_ops[: max(1, int(max_protected_ops))]:
        if not healthy_robots or surr is None:
            break
        ancestors = _ancestor_chain(tgt_op.op_id, op_map)
        prot_set = {tgt_op.op_id} | set(ancestors)
        cand_prio = dict(base_op_prio)
        for oid in prot_set:
            if oid in op_map and not op_map[oid].started:
                cand_prio[oid] = min_prio_val - (35 if oid == tgt_op.op_id else 25)
        # Slightly de-prioritize unstarted ops that have large early-wait waste so protected op clears choke first
        for o2 in unstarted:
            if o2.op_id not in prot_set:
                w2 = _compute_op_window_stats(o2, snapshot, base_eval, b_delta, succ_counts)
                if w2["early_waste"] >= 6.0:
                    cand_prio[o2.op_id] = max(cand_prio.get(o2.op_id, 0), min_prio_val + 45)

        cand_bays = dict(base_bays)
        cand_crews = dict(base_crews)
        nb, nc = _best_bay_and_crew_for_op(tgt_op, snapshot, base_bays, base_crews, b_delta, tasks_by_id)
        cand_bays[tgt_op.op_id] = nb
        cand_crews[tgt_op.op_id] = nc

        # Joint neighborhood = tgt_op's unpicked missing kitting tasks + predecessors' missing tasks + 1 contending peer
        focus_tids: list[str] = []
        for oid in [tgt_op.op_id] + ancestors:
            op_o = op_map.get(oid)
            if op_o is None:
                continue
            for tid in op_o.kit_task_ids:
                if tid in rel_set and tid not in focus_tids:
                    focus_tids.append(tid)
        # Add up to 2 lowest-urgency tasks assigned to fast AGVs serving tgt_op's bay corridor to allow freeing an AGV
        contenders = [
            tid
            for tid in releasable
            if tid not in focus_tids
            and (t_obj := tasks_by_id.get(tid)) is not None
            and (t_obj.kit_op_id or "") not in prot_set
        ]
        contenders.sort(key=lambda tid: -(tasks_by_id[tid].due_step or 999))
        neighborhood = focus_tids[:5] + contenders[:2]
        if not neighborhood:
            neighborhood = list(releasable[:4])

        fixed_sub = {k: v for k, v in base_assign.items() if k not in set(neighborhood)}
        old_p = getattr(surr, "op_prio", None)
        surr.op_prio = dict(cand_prio)
        cand_assign = surr.regret_k_assignment(healthy_robots, fixed_sub, neighborhood, op_bays=cand_bays)
        if old_p is not None:
            surr.op_prio = old_p

        # Promote protected op's unpicked tasks ahead in shared AGV queues
        cand_orders: dict[str, list[str]] = {}
        focus_set = set(focus_tids)
        for rid in healthy_robots:
            rtasks = [
                t
                for t in snapshot.tasks
                if t.status.lower() != "completed" and cand_assign.get(t.task_id) == rid
            ]
            if rtasks:
                rtasks.sort(
                    key=lambda t: (
                        0 if t.status.lower() in ("picked", "loading", "to_dropoff", "unloading") else 1,
                        0 if t.task_id in focus_set else 1,
                        cand_prio.get(t.kit_op_id or "", 0),
                        t.release_step,
                        t.task_id,
                    )
                )
                cand_orders[rid] = [t.task_id for t in rtasks]

        ev_j = evaluator.evaluate(
            cand_assign,
            op_priority=cand_prio,
            op_bay_overrides=cand_bays,
            op_crew_overrides=cand_crews,
            task_order_override=cand_orders or None,
        )
        if ev_j.valid:
            _add_candidate(
                f"protect_{tgt_op.op_id}",
                "protect_op_joint",
                tgt_op.op_id,
                cand_prio,
                cand_assign,
                cand_bays,
                cand_crews,
                cand_orders or None,
                tuple(neighborhood),
                ev_j,
            )

    # 3. Learning-Augmented Cross-Op Swap Joint Neighborhood (`learned_swap_joint`):
    #    Pairs a near-closing discrete-window op (`near_cliff`) with holding back a relaxed/early-wait op.
    if include_learned_proposal and len(scored_ops) >= 2 and healthy_robots and surr is not None:
        cliff_ops = [op for _rk, op, sw in scored_ops if sw["near_cliff"] >= 0.5]
        waste_ops = [op for _rk, op, sw in scored_ops if sw["early_waste"] >= 5.0]
        if cliff_ops and (waste_ops or len(scored_ops) >= 2):
            tgt_a = cliff_ops[0]
            yield_b = waste_ops[0] if waste_ops and waste_ops[0].op_id != tgt_a.op_id else scored_ops[-1][1]
            if tgt_a.op_id != yield_b.op_id:
                sw_prio = dict(base_op_prio)
                sw_prio[tgt_a.op_id] = min_prio_val - 45
                sw_prio[yield_b.op_id] = min_prio_val + 65
                sw_bays = dict(base_bays)
                sw_crews = dict(base_crews)
                nb_a, nc_a = _best_bay_and_crew_for_op(tgt_a, snapshot, base_bays, base_crews, b_delta, tasks_by_id)
                sw_bays[tgt_a.op_id] = nb_a
                sw_crews[tgt_a.op_id] = nc_a
                elig_c_b = list(getattr(yield_b, "eligible_crews", ()) or (sw_crews.get(yield_b.op_id, yield_b.crew_id),))
                if len(elig_c_b) > 1 and int(sw_crews.get(yield_b.op_id, yield_b.crew_id)) == int(nc_a):
                    alt_c = [int(c) for c in elig_c_b if int(c) != int(nc_a)]
                    if alt_c:
                        sw_crews[yield_b.op_id] = alt_c[0]

                nbr_swap = [
                    tid
                    for tid in (list(tgt_a.kit_task_ids) + list(yield_b.kit_task_ids))
                    if tid in rel_set
                ]
                if nbr_swap:
                    fixed_sw = {k: v for k, v in base_assign.items() if k not in set(nbr_swap)}
                    old_p = getattr(surr, "op_prio", None)
                    surr.op_prio = dict(sw_prio)
                    sw_assign = surr.regret_k_assignment(healthy_robots, fixed_sw, nbr_swap, op_bays=sw_bays)
                    if old_p is not None:
                        surr.op_prio = old_p

                    tgt_set = {tid for tid in tgt_a.kit_task_ids if tid in rel_set}
                    sw_orders: dict[str, list[str]] = {}
                    for rid in healthy_robots:
                        rtasks = [
                            t for t in snapshot.tasks
                            if t.status.lower() != "completed" and sw_assign.get(t.task_id) == rid
                        ]
                        if rtasks:
                            rtasks.sort(
                                key=lambda t: (
                                    0 if t.status.lower() in ("picked", "loading", "to_dropoff", "unloading") else 1,
                                    0 if t.task_id in tgt_set else 1,
                                    sw_prio.get(t.kit_op_id or "", 0),
                                    t.release_step,
                                    t.task_id,
                                )
                            )
                            sw_orders[rid] = [t.task_id for t in rtasks]

                    ev_sw = evaluator.evaluate(
                        sw_assign,
                        op_priority=sw_prio,
                        op_bay_overrides=sw_bays,
                        op_crew_overrides=sw_crews,
                        task_order_override=sw_orders or None,
                    )
                    if ev_sw.valid:
                        _add_candidate(
                            f"swap_{tgt_a.op_id}_over_{yield_b.op_id}",
                            "learned_swap_joint",
                            tgt_a.op_id,
                            sw_prio,
                            sw_assign,
                            sw_bays,
                            sw_crews,
                            sw_orders or None,
                            tuple(nbr_swap),
                            ev_sw,
                        )

    return cands
