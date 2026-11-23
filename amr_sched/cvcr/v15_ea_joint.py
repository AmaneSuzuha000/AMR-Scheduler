"""V15 Execution-Aware Integrated Production and Multi-AMR Scheduling Brain (`V15_EA_JOINT`).

Unifies fast online regret/BALANS adaptive destroy-repair, V10 CI/load-balance polishing,
selective zero-starvation kitting cliff rescue, and drift-adaptive soft commitment horizons
under an RTX 5060 CUDA-trained Set-Attention ranker + online discounted UCB calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.checkpoints import resolve_default_checkpoint
from amr_sched.cvcr.commitment import CommitmentClassification
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.snapshot import NativeSnapshot, align_start_to_windows, op_predecessors
from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
from amr_sched.network import Network

ACTION_KEEP = 0

DEFAULT_V15_CKPT = resolve_default_checkpoint("v15_ea_joint.npz")

V15_GLOBAL_FEAT_DIM = 14
V15_CAND_ONLY_DIM = 22
V15_CAND_FEAT_DIM = V15_CAND_ONLY_DIM + V15_GLOBAL_FEAT_DIM  # 36
V15_FAMILIES: tuple[str, ...] = (
    "keep_incumbent",
    "fast_detour_regret",
    "balans_adaptive_destroy",
    "v10_crit_ci_balance",
    "selective_cliff_rescue",
)
BALANS_RADII: tuple[int, ...] = (3, 6, 10, 999)


def _relu(x: np.ndarray) -> np.ndarray:
    """Elementwise ReLU activation."""
    return np.maximum(0.0, x)


def _sigmoid(x: float) -> float:
    """Numerically clipped scalar logistic sigmoid."""
    z = max(-30.0, min(30.0, float(x)))
    return 1.0 / (1.0 + float(np.exp(-z)))


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable 1D or last-axis softmax."""
    mx = np.max(x, axis=-1, keepdims=True)
    e = np.exp(np.clip(x - mx, -30.0, 30.0))
    return e / np.maximum(1e-9, np.sum(e, axis=-1, keepdims=True))


@dataclass
class V15Candidate:
    """Evaluated joint candidate generated inside V15 unified search."""

    cand_id: str
    source_kind: str
    assignment: dict[str, str]
    bay_overrides: dict[str, str]
    crew_overrides: dict[str, int]
    op_priority: dict[str, int]
    eval_res: EvalResult
    traffic_weights: dict[tuple[str, str], int]
    traffic_cutoff: int
    guide_routes: dict[str, list[tuple[str, str]]]
    commit_seq_horizon: int
    destroy_tids: tuple[str, ...]
    cliff_saved: bool
    det_score: float
    features: np.ndarray
    radius_used: int = 999
    live_repairs: Optional[dict[str, dict[str, Any]]] = None
    bottleneck_locks: Optional[dict[str, list[tuple[str, int, int]]]] = None
    pairwise_features: Optional[np.ndarray] = None
    action_diff_digest: str = ""


def extract_v15_global_features(
    snapshot: NativeSnapshot,
    base_eval: EvalResult,
    clf: CommitmentClassification,
    bay_delta: dict[str, float],
) -> np.ndarray:
    """Extract causal 14-D scene-level state descriptor for `V15_EA_JOINT`."""
    tasks = snapshot.tasks
    ops = snapshot.operations
    robs = snapshot.robots
    n_t = max(1, len(tasks))
    n_op = max(1, len(ops))
    n_r = max(1, len(robs))

    uncomp_t = [t for t in tasks if t.status.lower() != "completed"]
    unstarted_ops = [op for op in ops if not op.started]
    has_res_win = 1.0 if any(getattr(op, "resource_windows", ()) for op in unstarted_ops) else 0.0
    rel_ratio = float(len(clf.releasable_tasks)) / float(n_t)
    dist_ratio = float(len(clf.disturbed_robots)) / float(n_r)
    unassg_ratio = float(sum(1 for t in uncomp_t if t.assigned_agv is None)) / float(n_t)

    urg_ratio = float(sum(1 for op in unstarted_ops if op.is_urgent)) / float(n_op)
    late_op_ratio = 0.0
    cmax_est = float(max(base_eval.op_end.values())) if (base_eval.valid and base_eval.op_end) else float(snapshot.tick + 120)
    if base_eval.valid and unstarted_ops:
        late_op_ratio = float(sum(1 for op in unstarted_ops if base_eval.op_end.get(op.op_id, 0) > op.due_step)) / float(n_op)

    tot_wait = 0
    if base_eval.valid and base_eval.robot_paths:
        for pts in base_eval.robot_paths.values():
            for idx in range(len(pts) - 1):
                if pts[idx][0] == pts[idx + 1][0] and pts[idx + 1][1] > pts[idx][1]:
                    tot_wait += int(pts[idx + 1][1] - pts[idx][1])

    mean_bd = float(np.mean(list(bay_delta.values()))) if bay_delta else 0.0
    fb = getattr(snapshot, "execution_feedback", None)
    fb_states = getattr(fb, "robot_states", None) or {}
    fb_corridors = getattr(fb, "corridor_updates", None) or {}
    max_eta_drift = max((max(0, int(getattr(st, "eta_drift_steps", 0))) for st in fb_states.values()), default=0)
    min_spd = min((float(getattr(st, "speed_scale", 1.0)) for st in fb_states.values()), default=1.0)
    blocked_corridors = sum(1 for cf in fb_corridors.values() if getattr(cf, "status", "open") == "blocked")

    return np.asarray(
        [
            float(snapshot.tick) / 400.0,
            rel_ratio,
            dist_ratio,
            unassg_ratio,
            has_res_win,
            urg_ratio,
            late_op_ratio,
            min(2.0, float(tot_wait) / 120.0),
            min(2.0, float(base_eval.starvation_ticks) / 100.0) if base_eval.valid else 1.5,
            min(2.0, mean_bd / 12.0),
            min(2.0, cmax_est / 400.0),
            min(1.5, float(blocked_corridors) / 4.0),
            min(2.0, float(max_eta_drift) / 6.0),
            max(0.0, min(1.0, 1.0 - min_spd)),
        ],
        dtype=np.float32,
    )


def _count_wait_steps(ev_res: EvalResult) -> int:
    """Count total in-transit node dwell steps across all planned robot trajectories."""
    if not ev_res.valid or not ev_res.robot_paths:
        return 0
    tot = 0
    for pts in ev_res.robot_paths.values():
        for idx in range(len(pts) - 1):
            if pts[idx][0] == pts[idx + 1][0] and pts[idx + 1][1] > pts[idx][1]:
                tot += int(pts[idx + 1][1] - pts[idx][1])
    return tot


def _build_candidate_features(
    source_kind: str,
    ev_res: EvalResult,
    base_eval: EvalResult,
    assign: dict[str, str],
    base_assign: dict[str, str],
    bays: dict[str, str],
    crews: dict[str, int],
    snapshot: NativeSnapshot,
    commit_seq_horizon: int,
    cliff_saved: bool,
    online_bonus: float,
    rem_budget_ratio: float,
    g_feats: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute exact 36-D candidate vector and deterministic structural score."""
    base_obj = float(base_eval.objective) if base_eval.valid else 1e6
    cand_obj = float(ev_res.objective) if ev_res.valid else (base_obj + 5e4)
    obj_gain = (base_obj - cand_obj) / 80.0
    churn = sum(1 for tid, r in assign.items() if base_assign.get(tid) != r)
    bay_chg = sum(1 for op in snapshot.operations if bays.get(op.op_id, op.bay_id) != op.bay_id)
    crew_chg = sum(1 for op in snapshot.operations if crews.get(op.op_id, op.crew_id) != op.crew_id)
    wait_base = _count_wait_steps(base_eval)
    wait_cand = _count_wait_steps(ev_res)
    wait_gain = float(wait_base - wait_cand) / 15.0
    starv_gain = float((base_eval.starvation_ticks if base_eval.valid else 0) - ev_res.starvation_ticks) / 15.0

    uncomp_t = [t for t in snapshot.tasks if t.status.lower() != "completed"]
    n_healthy = max(
        1,
        sum(
            1
            for r in snapshot.robots
            if r.faulted_until <= snapshot.tick and r.state.lower() != "faulted"
        ),
    )
    tasks_per_agv = float(len(uncomp_t)) / float(n_healthy)
    has_res_win = float(g_feats[4]) > 0.5
    has_exec_drift = float(g_feats[11]) > 0.01 or float(g_feats[12]) > 0.25 or float(g_feats[13]) > 0.12
    calm_low_load = (
        not has_res_win
        and not has_exec_drift
        and float(g_feats[2]) == 0.0
        and wait_base < 35
        and tasks_per_agv < 2.5
    )

    hp = (0.12 if commit_seq_horizon == 1 else 0.0) if calm_low_load else (0.12 if commit_seq_horizon == 0 else -0.20)
    cliff_bonus = 0.25 if (cliff_saved and obj_gain >= -0.25 and not has_exec_drift) else 0.0

    # Structural deterministic prior score anchored strictly to B2 improvement + safe horizon prior
    det_score = obj_gain + hp + cliff_bonus + 0.05 * online_bonus
    if not ev_res.valid:
        det_score = -1e8
    if source_kind == "keep_incumbent":
        det_score = 0.0 if base_eval.valid else -1e8

    fam_onehot = [1.0 if source_kind == f else 0.0 for f in V15_FAMILIES]
    cand_only = np.asarray(
        [
            1.0 if ev_res.valid else 0.0,
            float(np.clip(obj_gain, -10.0, 10.0)),
            float(np.clip(det_score / 4.0, -5.0, 5.0)) if ev_res.valid else -5.0,
            min(2.0, float(churn) / 6.0),
            min(2.0, float(bay_chg + crew_chg) / 4.0),
            float(np.clip(wait_gain, -3.0, 3.0)),
            float(np.clip(starv_gain, -3.0, 3.0)),
            float(commit_seq_horizon) / 2.0,
            1.0 if cliff_saved else 0.0,
            float(np.clip(online_bonus, -1.5, 1.5)),
            float(np.clip(rem_budget_ratio, 0.0, 1.5)),
            min(2.0, float(max(ev_res.op_end.values())) / 400.0) if (ev_res.valid and ev_res.op_end) else 1.5,
            min(2.0, float(max(base_eval.op_end.values())) / 400.0) if (base_eval.valid and base_eval.op_end) else 1.5,
            min(2.0, tasks_per_agv / 4.0),
            float(wait_cand) / 120.0,
            float(ev_res.starvation_ticks) / 100.0,
            *fam_onehot,  # +5 -> total 21
            0.0 if commit_seq_horizon == 0 else 1.0,  # -> total 22 = V15_CAND_ONLY_DIM
        ],
        dtype=np.float32,
    )
    return np.concatenate([cand_only, g_feats], axis=0), float(det_score)


def compute_action_diff_digest(
    cand: "V15Candidate",
    base_assign: dict[str, str],
    base_bays: dict[str, str],
    base_crews: dict[str, int],
    base_prio: dict[str, int],
    *,
    live_repairs: Optional[dict[str, dict[str, Any]]] = None,
) -> str:
    """Compute canonical SHA-256 digest of physical executable diffs relative to incumbent."""
    assign_diff = sorted(
        (tid, agv) for tid, agv in cand.assignment.items() if base_assign.get(tid) != agv
    )
    bay_diff = sorted(
        (op_id, b) for op_id, b in cand.bay_overrides.items() if base_bays.get(op_id) != b
    )
    crew_diff = sorted(
        (op_id, int(c)) for op_id, c in cand.crew_overrides.items() if base_crews.get(op_id) != int(c)
    )
    prio_diff = sorted(
        (op_id, int(p)) for op_id, p in cand.op_priority.items() if int(base_prio.get(op_id, 999)) != int(p)
    )
    tw_diff = sorted((u, v, int(w)) for (u, v), w in cand.traffic_weights.items())
    guide_diff = sorted((rid, tuple(edges)) for rid, edges in cand.guide_routes.items() if edges)
    repair_diff = sorted(
        (
            aid,
            str(r.get("kind", "")),
            str(r.get("task_id", "")),
            tuple(tuple(h) for h in (r.get("hops") or ())),
            tuple(tuple(lk) for lk in (r.get("extra_node_locks") or ())),
        )
        for aid, r in (live_repairs or {}).items()
    )
    payload = {
        "assign_diff": assign_diff,
        "bay_diff": bay_diff,
        "crew_diff": crew_diff,
        "prio_diff": prio_diff,
        "tw_diff": tw_diff,
        "tw_cutoff": int(cand.traffic_cutoff) if tw_diff else 0,
        "guide_diff": guide_diff,
        "horizon": int(cand.commit_seq_horizon),
        "repair_diff": repair_diff,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


class V15ExecutionAwarePolicy:
    """Unified CPU/NumPy inference engine for `V15_EA_JOINT` with online bandit calibration."""

    def __init__(
        self,
        weights: Optional[dict[str, np.ndarray]] = None,
        *,
        conservative_gate: float = 0.25,
        zero_keep_residual: bool = True,
        allow_probe_override: bool = False,
    ) -> None:
        """Initialize weights and reset episodic online bandit statistics."""
        self.w = weights
        self.conservative_gate = float(conservative_gate)
        self.zero_keep_residual = bool(zero_keep_residual)
        self.allow_probe_override = bool(allow_probe_override)
        self.reset()

    @classmethod
    def from_npz(cls, path: str | Path = DEFAULT_V15_CKPT) -> "V15ExecutionAwarePolicy":
        """Load trained `V15ExecutionAwarePolicy` weights from `.npz` file."""
        data = np.load(str(path), allow_pickle=False)
        w = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
        return cls(weights=w)

    def reset(self) -> None:
        """Reset per-episode discounted bandit arm tables and online objective tracker."""
        self._rad_n: dict[int, float] = {k: 1.0 for k in BALANS_RADII}
        self._rad_val: dict[int, float] = {k: 0.05 for k in BALANS_RADII}
        self._fam_n: dict[str, float] = {f: 1.0 for f in V15_FAMILIES}
        self._fam_val: dict[str, float] = {f: 0.0 for f in V15_FAMILIES}
        self._rng = random.Random(20261119)

    def select_destroy_radius(self) -> int:
        """Select BALANS destroy subset radius via online discounted UCB1."""
        tot = sum(self._rad_n.values())
        return max(
            BALANS_RADII,
            key=lambda k: self._rad_val[k] + 0.45 * float(np.sqrt(np.log(max(2.0, tot)) / max(0.5, self._rad_n[k]))),
        )

    def family_ucb_bonus(self, family: str) -> float:
        """Return bounded online empirical utility estimate for operator family `family`."""
        if family not in self._fam_val:
            return 0.0
        n_f = self._fam_n.get(family, 1.0)
        tot = sum(self._fam_n.values())
        if n_f <= 1.01:
            return 0.0
        return float(np.clip(self._fam_val[family] + 0.15 * np.sqrt(np.log(max(2.0, tot)) / n_f), -1.0, 1.0))

    def observe_outcome(self, family: str, radius: int, reward_norm: float) -> None:
        """Update discounted Normal/UCB running averages after a candidate commits."""
        r = float(np.clip(reward_norm, -1.5, 2.0))
        if radius in self._rad_n:
            self._rad_n[radius] = self._rad_n[radius] * 0.80 + 1.0
            self._rad_val[radius] += (r - self._rad_val[radius]) / self._rad_n[radius]
        if family in self._fam_n:
            self._fam_n[family] = self._fam_n[family] * 0.82 + 1.0
            self._fam_val[family] += (r - self._fam_val[family]) / self._fam_n[family]

    def score_candidates(self, g_feats: np.ndarray, cands: list[V15Candidate]) -> np.ndarray:
        """Rank candidates via cross-candidate self-attention residual around true B2 det_score."""
        if not cands:
            return np.zeros((0,), dtype=np.float32)
        det_vec = np.asarray([c.det_score for c in cands], dtype=np.float32)
        if self.w is None or "cand_w1" not in self.w:
            for i, c in enumerate(cands):
                if not c.eval_res.valid:
                    det_vec[i] = -1e9
            return det_vec

        w = self.w
        mat = np.stack([c.features for c in cands], axis=0)
        x = np.clip((mat - w["c_mu"]) / np.maximum(0.05, w["c_sd"]), -6.0, 6.0)
        h = _relu(x @ w["cand_w1"] + w["cand_b1"])
        q = h @ w["att_q"]
        k = h @ w["att_k"]
        v = h @ w["att_v"]
        att = _softmax((q @ k.T) / np.sqrt(float(q.shape[-1])))
        h_att = _relu(np.concatenate([h, att @ v], axis=-1) @ w["cand_w2"] + w["cand_b2"])
        delta = (h_att @ w["cand_w3"] + w["cand_b3"]).reshape(-1)

        g_norm = np.clip((g_feats[:V15_GLOBAL_FEAT_DIM] - w["g_mu"]) / np.maximum(0.05, w["g_sd"]), -6.0, 6.0)
        hg = _relu(g_norm @ w["g_w1"] + w["g_b1"])
        trust_gain = float(_sigmoid((hg @ w["g_w2"] + w["g_b2"]).reshape(-1)[0])) * 2.0 * self.conservative_gate

        adj = np.tanh(delta / 2.0) * trust_gain
        if getattr(self, "zero_keep_residual", True):
            for i, c in enumerate(cands):
                if c.source_kind == "keep_incumbent":
                    adj[i] = 0.0
        scores = det_vec + adj
        for i, c in enumerate(cands):
            if not c.eval_res.valid:
                scores[i] = -1e9
        return scores.astype(np.float32)

    def select_budget_action(
        self,
        obs_v12: np.ndarray,
        mask: np.ndarray,
        *,
        tick: int = 16,
        has_imminent_cliff: bool = False,
        has_exec_drift: bool = False,
        has_resource_windows: bool = False,
    ) -> int:
        """Choose V12/V15 portfolio budget action cleanly (100ms on high coupling/fault/cliff, 50ms/100ms learned)."""
        if bool(int(tick) <= 4 or has_imminent_cliff or has_exec_drift) and len(mask) > 5 and mask[5] == 1:
            return 5

        if self.w is not None and "tier_w1" in self.w and len(obs_v12) >= 44:
            vec = np.asarray(obs_v12[:44], dtype=np.float32)
            x = np.clip((vec - self.w["t_mu"]) / np.maximum(0.05, self.w["t_sd"]), -6.0, 6.0)
            h = _relu(x @ self.w["tier_w1"] + self.w["tier_b1"])
            logits = (h @ self.w["tier_w2"] + self.w["tier_b2"]).reshape(-1)
            # Upgrade to 100ms ONLY when not under discrete resource windows AND neural margin > 1.5
            if (
                not has_resource_windows
                and len(mask) > 5
                and mask[5] == 1
                and float(logits[2]) > float(logits[1]) + 1.5
            ):
                return 5
            if len(mask) > 3 and mask[3] == 1:
                return 3

        if len(mask) > 3 and mask[3] == 1:
            return 3
        if len(mask) > 5 and mask[5] == 1:
            return 5
        if len(mask) > 2 and mask[2] == 1:
            return 2
        for idx in range(len(mask) - 1, 0, -1):
            if mask[idx] == 1:
                return idx
        return ACTION_KEEP


def load_v15_ea_policy(path: Optional[str | Path] = None) -> V15ExecutionAwarePolicy:
    """Load trained `V15ExecutionAwarePolicy` if present on disk, else calibrated structural prior."""
    p = Path(path) if path is not None else DEFAULT_V15_CKPT
    if p.exists():
        return V15ExecutionAwarePolicy.from_npz(p)
    return V15ExecutionAwarePolicy(weights=None)


def _find_danger_cliff_ops(
    snapshot: NativeSnapshot,
    eval_res: EvalResult,
    bay_delta: dict[str, float],
) -> list[tuple[float, str]]:
    """Return unstarted ops with discrete `resource_windows` sitting in a closeable danger band."""
    if not eval_res.valid:
        return []
    danger: list[tuple[float, str]] = []
    for op in snapshot.operations:
        if op.started:
            continue
        wins = tuple(getattr(op, "resource_windows", ()))
        if not wins:
            continue
        cur_end = int(eval_res.op_end.get(op.op_id, 0))
        cur_start = int(eval_res.op_start.get(op.op_id, snapshot.tick))
        kit_ready = max(
            [snapshot.tick] + [int(eval_res.order_unload_ticks.get(tid, snapshot.tick)) for tid in op.kit_task_ids]
        )
        # Check if kit arrival misses an earlier window close by <= 24 steps or is squeezed <= 12 steps before close
        for ws, we in wins:
            gap_to_close = int(we) - int(kit_ready)
            if -24 <= gap_to_close <= 18 and cur_start > int(we):
                # Rescuing this window saves the next-window wait jump
                jump_urgency = float(cur_end - op.due_step) * (3.0 if op.is_urgent else 1.5) + max(0.0, float(bay_delta.get(op.op_id, 0.0)))
                danger.append((jump_urgency, op.op_id))
                break
    danger.sort(key=lambda item: (-item[0], item[1]))
    return danger


def generate_v15_unified_candidates(
    policy_owner: Any,
    network: Network,
    snapshot: NativeSnapshot,
    clf: CommitmentClassification,
    evaluator: CVCREvaluator,
    surr: Any,
    base_assign: dict[str, str],
    base_bays: dict[str, str],
    base_crews: dict[str, int],
    base_eval: EvalResult,
    base_prio: dict[str, int],
    healthy_robots: list[str],
    bay_delta: dict[str, float],
    tw_seed: dict[tuple[str, str], int],
    v15_pol: V15ExecutionAwarePolicy,
    deadline: float,
    budget_ms: float,
    now_fn: Any,
) -> tuple[list[V15Candidate], np.ndarray]:
    """Generate unified C0..C4 V15 candidates without ever truncating global production-fleet optimization."""
    g_feats = extract_v15_global_features(snapshot, base_eval, clf, bay_delta)
    rem_r0 = max(0.0, min(1.0, (deadline - now_fn()) / max(0.01, float(budget_ms) / 1000.0)))
    cands: list[V15Candidate] = []
    cutoff_base = int(snapshot.tick) + 45 if tw_seed else 0

    def _push_cand(
        cand_id: str,
        kind: str,
        assign: dict[str, str],
        bays: dict[str, str],
        crews: dict[str, int],
        prio: dict[str, int],
        ev: EvalResult,
        tw: dict[tuple[str, str], int],
        cut: int,
        guides: dict[str, list[tuple[str, str]]],
        horizon: int,
        rel_tids: tuple[str, ...],
        cliff: bool,
        radius_used: int = 999,
    ) -> None:
        rem_now = max(0.0, min(1.0, (deadline - now_fn()) / max(0.01, float(budget_ms) / 1000.0)))
        feat, ds = _build_candidate_features(
            kind,
            ev,
            base_eval,
            assign,
            base_assign,
            bays,
            crews,
            snapshot,
            horizon,
            cliff,
            v15_pol.family_ucb_bonus(kind),
            rem_now,
            g_feats,
        )
        cands.append(
            V15Candidate(
                cand_id=cand_id,
                source_kind=kind,
                assignment=dict(assign),
                bay_overrides=dict(bays),
                crew_overrides=dict(crews),
                op_priority=dict(prio),
                eval_res=ev,
                traffic_weights=dict(tw),
                traffic_cutoff=int(cut),
                guide_routes=dict(guides),
                commit_seq_horizon=int(horizon),
                destroy_tids=rel_tids,
                cliff_saved=bool(cliff),
                det_score=float(ds),
                features=feat,
                radius_used=int(radius_used),
            )
        )

    def _guides_from(ev: EvalResult) -> dict[str, list[tuple[str, str]]]:
        return {}

    tw_active = dict(tw_seed) if tw_seed else {}

    # C0: Keep incumbent (anchored at zero-churn baseline)
    _push_cand(
        "C0_keep",
        "keep_incumbent",
        base_assign,
        base_bays,
        base_crews,
        base_prio,
        base_eval,
        tw_active,
        cutoff_base,
        _guides_from(base_eval),
        0,
        (),
        False,
    )

    if len(healthy_robots) < 2 or not policy_owner._has_budget(deadline, evaluator, min_ms=4.0):
        return cands, g_feats

    rel_all = sorted(clf.releasable_tasks)

    # Stage 1 + Stage 2: Bay/Crew Divert -> BALANS Shuffled Multi-Radius Destroy + Swaps (pure B2 surrogate, zero guides)
    evaluator.set_traffic_weights(tw_active or None, cutoff_step=cutoff_base)
    pure_surr = type(surr)(
        network,
        snapshot,
        dict(base_prio),
        policy_owner.alpha_makespan,
        policy_owner.lambda_change,
        bay_detour_delta=None,
    )
    ba = dict(base_assign)
    bb = dict(base_bays)
    bc = dict(base_crews)
    bp = dict(base_prio)
    bev = base_eval
    div_dl = min(deadline, now_fn() + min(0.016, max(0.004, (deadline - now_fn()) * 0.25)))
    bb, bc, bp, bev, _ = policy_owner._divert_prod_extended(
        snapshot, evaluator, ba, bb, bc, bev, bp, div_dl
    )
    pure_surr.op_prio = dict(bp)

    rad_k = v15_pol.select_destroy_radius()
    rel_shuf = list(rel_all)
    v15_pol._rng.shuffle(rel_shuf)
    rel_sub = rel_shuf[: min(len(rel_shuf), rad_k)] if rel_shuf else []
    if rel_sub and policy_owner._has_budget(deadline, evaluator, min_ms=2.5):
        reg_b = pure_surr.regret_k_assignment(
            healthy_robots,
            {t: a for t, a in ba.items() if t not in set(rel_sub)},
            rel_sub,
            op_bays=bb,
            rng=v15_pol._rng,
            noise=0.05,
        )
        if reg_b != ba:
            ev_reg = evaluator.evaluate(
                reg_b, op_priority=bp, op_bay_overrides=bb, op_crew_overrides=bc
            )
            if ev_reg.valid and (not bev.valid or ev_reg.objective + 0.5 < bev.objective):
                ba = reg_b
                bev = ev_reg
        if policy_owner._has_budget(deadline, evaluator, min_ms=2.5):
            sw_dl = min(deadline, now_fn() + max(0.006, (deadline - now_fn()) * 0.72))
            ba, bev, _ = policy_owner._run_1opt_and_swap_ranked(
                snapshot,
                clf,
                rel_sub,
                evaluator,
                ba,
                bev,
                healthy_robots,
                bp,
                sw_dl,
                op_bay_overrides=bb,
                op_crew_overrides=bc,
                allow_swaps=True,
                surr=pure_surr,
                hys_margin=0.0,
            )
    _push_cand(
        f"C1_balans_r{rad_k}_h0",
        "balans_adaptive_destroy",
        ba,
        bb,
        bc,
        bp,
        bev,
        tw_active,
        cutoff_base,
        _guides_from(bev),
        0,
        tuple(rel_sub),
        False,
        radius_used=rad_k,
    )

    # Stage 3: Progressive V8 Load-Balance (preserve_orders=False) + Intra-Agent CI Refinement -> C2 (h=0) & C3 (softpin h=1)
    c3_assign = dict(ba)
    c3_bays = dict(bb)
    c3_crews = dict(bc)
    c3_prio = dict(bp)
    c3_eval = bev
    if policy_owner._has_budget(deadline, evaluator, min_ms=3.5):
        evaluator.set_incumbent(c3_assign, c3_eval, op_priority=c3_prio, op_bay_overrides=c3_bays)
        bal_dl = min(deadline, now_fn() + max(0.004, (deadline - now_fn()) * 0.55))
        c3_assign, c3_eval, _ = policy_owner._search_v8_balance_pc(
            snapshot, clf, evaluator, c3_assign, c3_eval, healthy_robots, c3_prio, bal_dl,
            c3_bays, c3_crews, pure_surr, hys_margin=0.0, preserve_orders=False, lrho_frozen=None,
        )
    if policy_owner._has_budget(deadline, evaluator, min_ms=2.0):
        _, c3_eval, _ = policy_owner._search_intra_agent_ci(
            snapshot, evaluator, c3_assign, c3_bays, c3_crews, c3_eval, c3_prio, deadline,
            surr=pure_surr, max_candidates=6,
        )
    _push_cand(
        "C2_joint_h0",
        "fast_detour_regret",
        c3_assign,
        c3_bays,
        c3_crews,
        c3_prio,
        c3_eval,
        tw_active,
        cutoff_base,
        _guides_from(c3_eval),
        0,
        tuple(rel_sub),
        False,
    )
    _push_cand(
        "C3_softpin_h1",
        "v10_crit_ci_balance",
        c3_assign,
        c3_bays,
        c3_crews,
        c3_prio,
        c3_eval,
        tw_active,
        cutoff_base,
        _guides_from(c3_eval),
        1,
        tuple(rel_sub),
        False,
    )

    best_so_far = max((c for c in cands if c.eval_res.valid), key=lambda c: c.det_score, default=cands[0])

    # Stage 4: Selective Zero-Starvation Cliff Rescue (strictly gated on real close-window danger, zero queue lock)
    danger_ops = _find_danger_cliff_ops(snapshot, best_so_far.eval_res, bay_delta)
    if danger_ops and policy_owner._has_budget(deadline, evaluator, min_ms=5.0):
        top_op_id = danger_ops[0][1]
        op_obj = next((o for o in snapshot.operations if o.op_id == top_op_id), None)
        if op_obj is not None:
            c4_prio = dict(best_so_far.op_priority)
            # Bounded priority boost (-18 instead of extreme priority domination) + full flow re-balance
            c4_prio[top_op_id] = int(c4_prio.get(top_op_id, 100)) - 18
            c4_assign = dict(best_so_far.assignment)
            c4_bays = dict(best_so_far.bay_overrides)
            c4_crews = dict(best_so_far.crew_overrides)
            pure_surr.op_prio = dict(c4_prio)
            kit_rel = [tid for tid in op_obj.kit_task_ids if tid in clf.releasable_tasks]
            rem_rel = [tid for tid in rel_all if tid not in set(kit_rel)]
            # Optimize ALL releasable tasks together so parallel operations are NEVER starved
            reord_rel = kit_rel + rem_rel
            if reord_rel:
                c4_assign = pure_surr.regret_k_assignment(
                    healthy_robots,
                    {t: a for t, a in c4_assign.items() if t not in set(reord_rel)},
                    reord_rel,
                    op_bays=c4_bays,
                )
            c4_eval = evaluator.evaluate(
                c4_assign, op_priority=c4_prio, op_bay_overrides=c4_bays, op_crew_overrides=c4_crews
            )
            if policy_owner._has_budget(deadline, evaluator, min_ms=3.5):
                c4_bays, c4_crews, c4_prio, c4_eval, _ = policy_owner._divert_prod_extended(
                    snapshot, evaluator, c4_assign, c4_bays, c4_crews, c4_eval, c4_prio,
                    min(deadline, now_fn() + 0.010),
                )
            if reord_rel and policy_owner._has_budget(deadline, evaluator, min_ms=3.0):
                c4_assign, c4_eval, _ = policy_owner._run_1opt_and_swap_ranked(
                    snapshot, clf, reord_rel, evaluator, c4_assign, c4_eval, healthy_robots, c4_prio,
                    min(deadline, now_fn() + 0.014), op_bay_overrides=c4_bays, op_crew_overrides=c4_crews,
                    allow_swaps=True, surr=pure_surr, hys_margin=0.0,
                )
            before_start = int(best_so_far.eval_res.op_start.get(top_op_id, 999999)) if best_so_far.eval_res.valid else 999999
            after_start = int(c4_eval.op_start.get(top_op_id, 999999)) if c4_eval.valid else 999999
            op_wins = getattr(op_obj, "resource_windows", ())
            saved = bool(c4_eval.valid and align_start_to_windows(after_start, op_wins) < align_start_to_windows(before_start, op_wins))
            _push_cand(
                f"C4_cliff_{top_op_id}",
                "selective_cliff_rescue",
                c4_assign,
                c4_bays,
                c4_crews,
                c4_prio,
                c4_eval,
                tw_active,
                cutoff_base,
                _guides_from(c4_eval),
                0,
                tuple(reord_rel),
                saved,
            )

    return cands, g_feats


class V15AdaptiveController:
    """Closed-loop meta-controller driving `V15_EA_JOINT` inside `AMRReschedulingEnv`."""

    def __init__(
        self,
        *,
        epoch_interval: int = 16,
        stall_threshold: int = 20,
        policy: Optional[V15ExecutionAwarePolicy] = None,
        trigger: Optional[Any] = None,
    ) -> None:
        """Initialize lean V10 RCDR trigger + persistent V15 policy (zero nominal leaf-hazard spam)."""
        self.epoch_interval = int(epoch_interval)
        self.stall_threshold = int(stall_threshold)
        self.policy = policy if policy is not None else load_v15_ea_policy()
        self.trigger = trigger
        self.last_dec: int = -999
        self.prev_emerg: tuple[bool, bool] = (False, False)
        self._last_cliff_check_tick: int = -999
        self._last_surgical: int = -999
        self._last_p2_global_key: Any = None
        self._last_p2_surg_key: Any = None

    def reset(self) -> None:
        """Reset trigger state and online contextual bandit tables for a new episode."""
        self.policy.reset()
        if self.trigger is not None and hasattr(self.trigger, "reset"):
            self.trigger.reset()
        self.last_dec = -999
        self.prev_emerg = (False, False)
        self._last_cliff_check_tick = -999
        self._last_surgical = -999
        self._last_p2_global_key = None
        self._last_p2_surg_key = None

    def check_surgical_repair_action(self, mask: np.ndarray, tick: int, sim: Any) -> Optional[int]:
        """Return `1` for surgical loaded/oscillation repair if any physical stall hazard triggers."""
        if sim is None or len(mask) <= 1 or mask[1] != 1:
            return None
        if getattr(sim, "_p2_execution_aware", False) and getattr(sim, "_perturbation_model", None) is not None:
            p2_sig = sim._perturbation_model.active_disturbance_sig(sim, tick)
            blk_p = tuple(p2_sig.get("blocked_picked_agvs") or ())
            blk_a = tuple(p2_sig.get("blocked_assigned_agvs") or ())
            blk_e = tuple(p2_sig.get("blocked_edges") or ())
            if blk_p or blk_a:
                s_key = (blk_p, blk_a, blk_e)
                if s_key != self._last_p2_surg_key and (self._last_surgical < 0 or (tick - self._last_surgical) >= 4):
                    self._last_p2_surg_key = s_key
                    self._last_surgical = tick
                    return 1
            else:
                self._last_p2_surg_key = None

        from amr_sched.cvcr.p4_belief_choke import detect_choke_oscillation_agvs
        if detect_choke_oscillation_agvs(sim):
            if self._last_surgical < 0 or (tick - self._last_surgical) >= 8:
                self._last_surgical = tick
                return 1

        from amr_sched.cvcr.loaded_corridor_repair import detect_stalled_picked_agvs
        if detect_stalled_picked_agvs(sim, stall_threshold=self.stall_threshold, include_edge_gate_stalls=True):
            if self._last_surgical < 0 or (tick - self._last_surgical) >= 8:
                self._last_surgical = tick
                return 1
        return None

    def select_action(self, obs_v12: np.ndarray, info: dict[str, Any], sim: Any = None) -> int:
        """Select action causally via V10 RCDR + execution-aware P2/P4 wakeups + imminent cliffs."""
        from amr_sched.rl.spaces import is_in_secondary_hold, should_query_v10_controller

        mask = info["action_mask"]
        tick = int(info.get("step", 0))
        if sim is not None and is_in_secondary_hold(sim):
            return ACTION_KEEP

        has_imminent_cliff = False
        has_exec_drift = False
        has_resource_windows = False
        p2_global_wake = False

        if sim is not None:
            fb_pkt = getattr(sim, "_execution_feedback", None)
            if fb_pkt is not None and not fb_pkt.is_empty():
                has_exec_drift = True

            # P2 Execution-Aware wakeups (strictly gated on active perturbation model)
            if getattr(sim, "_p2_execution_aware", False) and getattr(sim, "_perturbation_model", None) is not None:
                p2_sig = sim._perturbation_model.active_disturbance_sig(sim, tick)
                fail_assg = tuple(p2_sig.get("fail_assigned_agvs") or ())
                if fail_assg:
                    if fail_assg != self._last_p2_global_key and (self.last_dec < 0 or (tick - self.last_dec) >= 6):
                        self._last_p2_global_key = fail_assg
                        p2_global_wake = True
                        has_exec_drift = True
                else:
                    self._last_p2_global_key = None

            surg_act = self.check_surgical_repair_action(mask, tick, sim)
            if surg_act is not None:
                return surg_act

            # Wake up cleanly if an unstarted discrete resource window closes within 20 steps
            tracker = getattr(sim, "_prod_tracker", None)
            if tracker is not None:
                for op in getattr(tracker, "operations", ()):
                    if getattr(op, "resource_windows", ()):
                        has_resource_windows = True
                        if not op.started:
                            for _, we in op.resource_windows:
                                if 0 <= int(we) - tick <= 20:
                                    has_imminent_cliff = True
                                    break

        q_v10, new_emerg = should_query_v10_controller(
            obs_v12[:38],
            mask,
            tick,
            self.last_dec,
            self.epoch_interval,
            self.prev_emerg,
            in_secondary_hold=False,
        )
        self.prev_emerg = new_emerg

        should_replan = bool(p2_global_wake or q_v10)
        if not should_replan and has_imminent_cliff and (tick - self._last_cliff_check_tick) >= 12:
            if any(mask[i] == 1 for i in (3, 5)):
                should_replan = True

        if not should_replan:
            return ACTION_KEEP

        self.last_dec = tick
        if has_imminent_cliff:
            self._last_cliff_check_tick = tick
        return self.policy.select_budget_action(
            obs_v12,
            mask,
            tick=tick,
            has_imminent_cliff=has_imminent_cliff,
            has_exec_drift=has_exec_drift,
            has_resource_windows=has_resource_windows,
        )
