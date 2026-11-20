"""L-RHO Look-Ahead Invariance Predictor (Sirui Li et al., ICLR 2025, arXiv:2502.15791).

Learns which previously assigned overlapping tasks/operations keep their exact same
resource assignment across consecutive rolling-horizon full solves:
  O^*_fix,r = { t in O_overlap,r | m_r(t) = m_{r-1}(t) }

Uses pure NumPy runtime inference from `.npz` weights (sub-0.1ms overhead inside the
real-time G1 planning budget) following L-RHO's dual-entity architecture:
  - Task/Op MLP encoder (19-D) -> 64-D embedding
  - AGV summary MLP encoder (6-D) -> 64-D embedding
  - Concat [h_task, h_assigned_agv, h_global_pool] (192-D) -> 64 -> logit
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.checkpoints import resolve_default_checkpoint
from amr_sched.cvcr.commitment import CommitmentClassification
from amr_sched.cvcr.evaluator import EvalResult
from amr_sched.cvcr.features import extract_task_features
from amr_sched.cvcr.snapshot import NativeSnapshot
from amr_sched.network import Network

LRHO_TASK_FEAT_DIM = 19
LRHO_AGV_FEAT_DIM = 6
DEFAULT_LRHO_CKPT = resolve_default_checkpoint("lrho_keep_release_v10.npz")


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def extract_lrho_features(
    snapshot: NativeSnapshot,
    clf: CommitmentClassification,
    eval_res: EvalResult,
    op_bays: dict[str, str],
    op_crews: dict[str, int],
    network: Network | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, list[int]]:
    """Extract L-RHO task (N x 19), AGV (M x 6), and assigned-AGV index features."""
    t_now = int(snapshot.tick)
    base_tids, base_x = extract_task_features(
        snapshot,
        clf,
        eval_res.op_res_ready,
        eval_res.order_unload_ticks,
        eval_res.robot_paths,
    )
    if not base_tids:
        return [], np.zeros((0, LRHO_TASK_FEAT_DIM), dtype=np.float32), np.zeros((max(1, len(snapshot.robots)), LRHO_AGV_FEAT_DIM), dtype=np.float32), []

    snap_by_id = {t.task_id: t for t in snapshot.tasks}
    op_by_id = {op.op_id: op for op in snapshot.operations}
    agvs = sorted(snapshot.robots, key=lambda r: r.robot_id)
    agv_idx_map = {r.robot_id: idx for idx, r in enumerate(agvs)}

    # Precompute shortest travel times cleanly if network provided
    tt_src: dict[str, dict[str, float]] = {}
    max_spd = 3.0
    if network is not None and network.edges:
        max_spd = max(0.5, max(e.speed_limit for e in network.edges.values()))

    def _tt(a: str, b: str) -> float:
        if network is None or a == b:
            return 0.0
        d = tt_src.get(a)
        if d is None:
            d = network.shortest_travel_times(a, max_spd)
            tt_src[a] = d
        return float(d.get(b, 50.0))

    # AGV summaries (6-D)
    agv_feats = np.zeros((max(1, len(agvs)), LRHO_AGV_FEAT_DIM), dtype=np.float32)
    agv_tasks: dict[str, list[Any]] = {r.robot_id: [] for r in agvs}
    for t in snapshot.tasks:
        if t.status.lower() != "completed" and t.assigned_agv in agv_tasks:
            agv_tasks[t.assigned_agv].append(t)

    for idx, r in enumerate(agvs):
        tlist = agv_tasks.get(r.robot_id, [])
        slacks = [
            float((t.due_step if t.due_step is not None else t_now + 100) - eval_res.order_unload_ticks.get(t.task_id, t_now)) / 30.0
            for t in tlist
        ]
        urg_cnt = sum(1 for t in tlist if getattr(t, "is_urgent", False))
        ov_cnt = sum(1 for t in tlist if t.task_id in clf.releasable_tasks)
        agv_feats[idx, 0] = 1.0 if r.robot_id in clf.disturbed_robots else 0.0
        agv_feats[idx, 1] = float(len(tlist)) / 6.0
        agv_feats[idx, 2] = float(max(0, r.faulted_until - t_now)) / 20.0
        agv_feats[idx, 3] = float(ov_cnt) / 6.0
        agv_feats[idx, 4] = float(urg_cnt) / 3.0
        agv_feats[idx, 5] = float(np.mean(slacks)) if slacks else 1.0

    # Concatenate 13 base + 6 overlap/lookahead features -> 19-D
    task_feats = np.zeros((len(base_tids), LRHO_TASK_FEAT_DIM), dtype=np.float32)
    task_feats[:, :13] = base_x.astype(np.float32)
    assigned_agv_idx: list[int] = []
    healthy_agvs = [r for r in agvs if r.robot_id not in clf.disturbed_robots and r.state != "faulted"]

    for row_i, tid in enumerate(base_tids):
        t = snap_by_id[tid]
        rb = t.assigned_agv
        in_overlap = 1.0 if (rb is not None and rb in agv_idx_map and rb not in clf.disturbed_robots) else 0.0
        u_tick = float(eval_res.order_unload_ticks.get(tid, t_now + 15))
        t_due = float(t.due_step if t.due_step is not None else t_now + 120)
        u_slack = (t_due - u_tick) / 30.0
        cur_load = float(len(agv_tasks.get(rb or "", []))) / 6.0 if rb else 0.0

        alt_delta = 0.0
        if network is not None and rb is not None and rb in agv_idx_map and len(healthy_agvs) >= 2:
            cur_r = next((r for r in agvs if r.robot_id == rb), None)
            cur_tt = _tt(cur_r.cur_node, t.pickup) if cur_r is not None else 20.0
            best_alt_tt = min(
                (_tt(r.cur_node, t.pickup) + float(len(agv_tasks.get(r.robot_id, []))) * 6.0)
                for r in healthy_agvs
                if r.robot_id != rb
            )
            cur_cost = cur_tt + float(max(0, len(agv_tasks.get(rb, [])) - 1)) * 6.0
            alt_delta = float(np.clip((best_alt_tt - cur_cost) / 25.0, -2.0, 2.0))

        op = op_by_id.get(t.kit_op_id or "")
        op_flex = 0.0
        is_crit_kit = 0.0
        if op is not None:
            has_flex = len(op.eligible_bays) > 1 or len(op.eligible_crews) > 1
            op_flex = 1.0 if has_flex else 0.0
            en = float(eval_res.op_end.get(op.op_id, t_now))
            if base_x[row_i, 7] > 0.0 or en > float(op.due_step):
                is_crit_kit = 1.0

        task_feats[row_i, 13] = in_overlap
        task_feats[row_i, 14] = cur_load
        task_feats[row_i, 15] = u_slack
        task_feats[row_i, 16] = alt_delta
        task_feats[row_i, 17] = op_flex
        task_feats[row_i, 18] = is_crit_kit
        assigned_agv_idx.append(agv_idx_map.get(rb or "", 0))

    return base_tids, task_feats, agv_feats, assigned_agv_idx


class LRHOPredictor:
    """Pure-NumPy runtime predictor for L-RHO hindsight invariance freezing."""

    def __init__(
        self,
        weights: dict[str, np.ndarray],
        prob_threshold: float = 0.72,
        network: Network | None = None,
    ) -> None:
        self.w = weights
        self.prob_threshold = float(prob_threshold)
        self.network = network

    @classmethod
    def from_npz(
        cls,
        path: str | Path = DEFAULT_LRHO_CKPT,
        prob_threshold: float | None = None,
        network: Network | None = None,
    ) -> "LRHOPredictor":
        """Load trained L-RHO `.npz` weights into a fast NumPy predictor."""
        data = np.load(str(path), allow_pickle=False)
        w = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
        thr = float(prob_threshold) if prob_threshold is not None else float(w.get("threshold", np.asarray([0.72]))[0])
        return cls(w, prob_threshold=thr, network=network)

    def predict_proba(
        self,
        task_x: np.ndarray,
        agv_x: np.ndarray,
        assigned_agv_idx: list[int],
    ) -> np.ndarray:
        """Predict per-task assignment-invariance probability in (0, 1)."""
        if len(task_x) == 0:
            return np.zeros((0,), dtype=np.float32)
        tx = (task_x - self.w["t_mu"]) / np.maximum(1e-6, self.w["t_sd"])
        ax = (agv_x - self.w["a_mu"]) / np.maximum(1e-6, self.w["a_sd"])

        ht = _relu(tx @ self.w["t_w1"] + self.w["t_b1"])
        ht = _relu(ht @ self.w["t_w2"] + self.w["t_b2"])
        ha = _relu(ax @ self.w["a_w1"] + self.w["a_b1"])
        ha = _relu(ha @ self.w["a_w2"] + self.w["a_b2"])

        ha_local = ha[np.asarray(assigned_agv_idx, dtype=np.int64)]
        g_pool = np.broadcast_to(0.5 * (ht.mean(axis=0, keepdims=True) + ha.mean(axis=0, keepdims=True)), ht.shape)
        cat = np.concatenate([ht, ha_local, g_pool], axis=-1)
        h_out = _relu(cat @ self.w["o_w1"] + self.w["o_b1"])
        logits = np.clip((h_out @ self.w["o_w2"] + self.w["o_b2"]).reshape(-1), -20.0, 20.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def predict_frozen_tasks(
        self,
        snapshot: NativeSnapshot,
        clf: CommitmentClassification,
        eval_res: EvalResult,
        op_bays: dict[str, str],
        op_crews: dict[str, int],
    ) -> set[str]:
        """Return subset of overlapping non-critical releasable tasks predicted assignment-invariant."""
        tids, tx, ax, a_idx = extract_lrho_features(
            snapshot, clf, eval_res, op_bays, op_crews, network=self.network
        )
        if not tids:
            return set()
        probs = self.predict_proba(tx, ax, a_idx)
        frozen: set[str] = set()
        for i, tid in enumerate(tids):
            # Strict safety guard: never freeze unassigned, disturbed, urgent, or genuine last-missing-kit bottleneck tasks
            is_overlap = tx[i, 13] > 0.5
            is_urgent = tx[i, 5] > 0.5
            is_last_kit_blocker = tx[i, 7] > 0.0
            if is_overlap and not is_urgent and not is_last_kit_blocker:
                if float(probs[i]) >= self.prob_threshold:
                    frozen.add(tid)
        return frozen


def load_lrho_predictor(
    path: str | Path = DEFAULT_LRHO_CKPT,
    prob_threshold: float | None = None,
    network: Network | None = None,
) -> Optional[LRHOPredictor]:
    """Load L-RHO predictor if checkpoint file exists on disk, else return None."""
    p = Path(path)
    if not p.exists():
        return None
    return LRHOPredictor.from_npz(p, prob_threshold=prob_threshold, network=network)
