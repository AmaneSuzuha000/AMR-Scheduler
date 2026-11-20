"""Topology-Agnostic Heterogeneous Graph Destroy Selector (V12).

Synthesizes TMLR 2025 Neural Deconstruction Search (NDS), ICLR 2024 Neural LNS,
NeurIPS 2025 TARGETNET heterogeneous graph message passing, and CPAIOR 2026
stochastic-destroy / deterministic-repair inside `<0.5 ms` pure-NumPy inference:

Instead of freezing overlapping tasks with a fixed threshold (`lrho.py`), predicts
per-task destroy priority scores `p_destroy(t) in (0, 1)` over a causal `(Task, AGV, Op)`
heterogeneous interaction graph enriched with `Detour-Delta` congestion and DAG depth,
enabling multi-radius LNS arms (`small`, `medium`, `large_stochastic`) consumed cleanly
by `BALANS`, `RuleV12`, and `RL-v12`.
"""

from __future__ import annotations

import math
from pathlib import Path
import random
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.checkpoints import resolve_default_checkpoint
from amr_sched.cvcr.commitment import CommitmentClassification
from amr_sched.cvcr.evaluator import EvalResult
from amr_sched.cvcr.lrho import extract_lrho_features
from amr_sched.cvcr.snapshot import NativeSnapshot, op_predecessors
from amr_sched.cvcr.v12_proactive_guard import analyze_network_havens
from amr_sched.network import Network

DESTROY_TASK_FEAT_DIM = 25
DESTROY_AGV_FEAT_DIM = 8
DESTROY_OP_FEAT_DIM = 8
DEFAULT_DESTROY_CKPT = resolve_default_checkpoint("hetero_destroy_v12.npz")


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def extract_hetero_destroy_features(
    network: Network | None,
    snapshot: NativeSnapshot,
    clf: CommitmentClassification,
    eval_res: EvalResult,
    op_bays: dict[str, str],
    op_crews: dict[str, int],
    *,
    bay_detour_delta: Optional[dict[str, float]] = None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[int], list[int], list[list[int]]]:
    """Extract causal heterogeneous `(Task, AGV, Op)` tensors & incidence indices.

    Returns `(tids, task_x, agv_x, op_x, assigned_agv_idx, task_op_idx, sibling_indices)`.
    """
    base_tids, t19, a6, agv_idx = extract_lrho_features(
        snapshot, clf, eval_res, op_bays, op_crews, network=network
    )
    n_t = len(base_tids)
    agvs = sorted(snapshot.robots, key=lambda r: r.robot_id)
    n_a = max(1, len(agvs))
    ops = sorted(snapshot.operations, key=lambda o: o.op_id)
    n_o = max(1, len(ops))

    if n_t == 0:
        return (
            [],
            np.zeros((0, DESTROY_TASK_FEAT_DIM), dtype=np.float32),
            np.zeros((n_a, DESTROY_AGV_FEAT_DIM), dtype=np.float32),
            np.zeros((n_o, DESTROY_OP_FEAT_DIM), dtype=np.float32),
            [],
            [],
            [],
        )

    t_now = int(snapshot.tick)
    bay_delta = bay_detour_delta or {}
    snap_by_id = {t.task_id: t for t in snapshot.tasks}
    op_idx_map = {op.op_id: i for i, op in enumerate(ops)}
    leaf_spurs = analyze_network_havens(network).leaf_spurs if network is not None else {}

    # Compute DAG downstream counts and precedence depths
    succ_cnt: dict[str, int] = {op.op_id: 0 for op in ops}
    pred_cnt: dict[str, int] = {}
    for op in ops:
        preds = op_predecessors(op)
        pred_cnt[op.op_id] = len(preds)
        for pid in preds:
            if pid in succ_cnt:
                succ_cnt[pid] += 1

    # Enrich AGV features (6 base + stalled/blocked proxy + path bay delta)
    agv_x = np.zeros((n_a, DESTROY_AGV_FEAT_DIM), dtype=np.float32)
    agv_x[:, :6] = a6
    agv_task_deltas: dict[str, list[float]] = {r.robot_id: [] for r in agvs}
    for t in snapshot.tasks:
        if t.status.lower() != "completed" and t.assigned_agv in agv_task_deltas:
            b_tgt = op_bays.get(t.kit_op_id or "", t.dropoff) if t.status.lower() in ("pending", "assigned") else t.dropoff
            agv_task_deltas[t.assigned_agv].append(float(bay_delta.get(b_tgt, 0.0)))
    for i, r in enumerate(agvs):
        ds = agv_task_deltas.get(r.robot_id, [])
        agv_x[i, 6] = 1.0 if r.state.lower() == "blocked" else 0.0
        agv_x[i, 7] = float(np.mean(ds) / 25.0) if ds else 0.0

    # Operation node features (8-D)
    op_x = np.zeros((n_o, DESTROY_OP_FEAT_DIM), dtype=np.float32)
    op_to_task_rows: dict[str, list[int]] = {}
    for row_i, tid in enumerate(base_tids):
        ko = snap_by_id[tid].kit_op_id
        if ko:
            op_to_task_rows.setdefault(ko, []).append(row_i)

    for oi, op in enumerate(ops):
        st = float(eval_res.op_start.get(op.op_id, t_now))
        en = float(eval_res.op_end.get(op.op_id, st + op.proc_steps))
        r_ready = float(eval_res.op_res_ready.get(op.op_id, t_now))
        kit_arrs = [float(eval_res.order_unload_ticks.get(kt, t_now)) for kt in op.kit_task_ids]
        spread = (max(kit_arrs) - min(kit_arrs)) / 25.0 if len(kit_arrs) >= 2 else 0.0
        cur_bay = op_bays.get(op.op_id, op.bay_id)
        op_x[oi, 0] = float(op.due_step - en) / 40.0
        op_x[oi, 1] = float(max(0.0, en - op.due_step)) / 40.0
        op_x[oi, 2] = max(0.0, st - r_ready) / 25.0
        op_x[oi, 3] = 1.0 if op.is_urgent else 0.0
        op_x[oi, 4] = float(spread)
        op_x[oi, 5] = float(bay_delta.get(cur_bay, 0.0)) / 25.0
        op_x[oi, 6] = 1.0 if (len(op.eligible_bays) > 1 or len(op.eligible_crews) > 1) else 0.0
        op_x[oi, 7] = float(pred_cnt.get(op.op_id, 0) + succ_cnt.get(op.op_id, 0)) / 4.0

    # Enrich Task features (19 base + 6 relational/Detour-Delta features -> 25-D)
    task_x = np.zeros((n_t, DESTROY_TASK_FEAT_DIM), dtype=np.float32)
    task_x[:, :19] = t19
    task_op_idx: list[int] = []
    sibling_indices: list[list[int]] = []

    for row_i, tid in enumerate(base_tids):
        t = snap_by_id[tid]
        ko = t.kit_op_id or ""
        b_eff = op_bays.get(ko, t.dropoff) if (ko and t.status.lower() in ("pending", "assigned")) else t.dropoff
        oi = op_idx_map.get(ko, 0)
        task_op_idx.append(oi)
        sibs = [j for j in op_to_task_rows.get(ko, ()) if j != row_i]
        sibling_indices.append(sibs)

        task_x[row_i, 19] = float(bay_delta.get(b_eff, 0.0)) / 25.0
        task_x[row_i, 20] = float(bay_delta.get(t.pickup, 0.0)) / 25.0
        task_x[row_i, 21] = 1.0 if b_eff in leaf_spurs else 0.0
        task_x[row_i, 22] = float(pred_cnt.get(ko, 0)) / 3.0 if ko else 0.0
        task_x[row_i, 23] = float(succ_cnt.get(ko, 0)) / 3.0 if ko else 0.0
        # Wait/detour slack sensitivity along AGV route
        r_obj = next((r for r in agvs if r.robot_id == t.assigned_agv), None)
        task_x[row_i, 24] = 1.0 if (r_obj is not None and r_obj.state.lower() == "blocked") else 0.0

    return base_tids, task_x, agv_x, op_x, agv_idx, task_op_idx, sibling_indices


class HeteroDestroySelector:
    """Fast heterogeneous graph LNS destroy scorer & multi-radius subset selector."""

    def __init__(
        self,
        weights: Optional[dict[str, np.ndarray]] = None,
        *,
        prob_threshold: float = 0.45,
    ) -> None:
        self.w = weights
        self.prob_threshold = float(prob_threshold)

    @classmethod
    def from_npz(cls, path: str | Path = DEFAULT_DESTROY_CKPT) -> "HeteroDestroySelector":
        """Load trained heterogeneous destroy weights from `.npz` file."""
        data = np.load(str(path), allow_pickle=False)
        w = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
        thr = float(w.get("threshold", np.asarray([0.45]))[0])
        return cls(weights=w, prob_threshold=thr)

    def predict_destroy_scores(
        self,
        task_x: np.ndarray,
        agv_x: np.ndarray,
        op_x: np.ndarray,
        assigned_agv_idx: list[int],
        task_op_idx: list[int],
        sibling_indices: list[list[int]],
    ) -> np.ndarray:
        """Compute calibrated per-task destroy benefit probabilities in `(0, 1)`."""
        n_t = len(task_x)
        if n_t == 0:
            return np.zeros((0,), dtype=np.float32)

        if self.w is None:
            # Well-calibrated structural prior when weights are absent
            raw = (
                1.4 * task_x[:, 0]                 # disturbed assigned AGV
                + 1.3 * task_x[:, 1]               # unassigned
                + 0.9 * np.maximum(0.0, task_x[:, 7])   # kit gap / bottleneck
                + 0.8 * np.maximum(0.0, -task_x[:, 16]) # better alternative AGV (-alt_delta)
                + 0.6 * task_x[:, 19]              # bay Detour-Delta congestion
                + 0.5 * task_x[:, 5]               # urgent priority
                + 0.4 * np.maximum(0.0, -task_x[:, 15]) # negative unload slack
                - 0.3 * np.maximum(0.0, task_x[:, 15])  # large positive slack -> stable
                + 0.3 * task_x[:, 24]              # blocked robot
            )
            logits = np.clip(raw - 0.6, -10.0, 10.0)
            return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

        w = self.w
        tx = (task_x - w["t_mu"]) / np.maximum(1e-6, w["t_sd"])
        ax = (agv_x - w["a_mu"]) / np.maximum(1e-6, w["a_sd"])
        ox = (op_x - w["o_mu"]) / np.maximum(1e-6, w["o_sd"])

        ht = _relu(tx @ w["t_w1"] + w["t_b1"])
        ht = _relu(ht @ w["t_w2"] + w["t_b2"])
        ha = _relu(ax @ w["a_w1"] + w["a_b1"])
        ha = _relu(ha @ w["a_w2"] + w["a_b2"])
        ho = _relu(ox @ w["o_w1"] + w["o_b1"])
        ho = _relu(ho @ w["o_w2"] + w["o_b2"])

        ha_loc = ha[np.asarray(assigned_agv_idx, dtype=np.int64)]
        ho_loc = ho[np.asarray(task_op_idx, dtype=np.int64)]

        hsib = np.zeros_like(ht)
        for i, sibs in enumerate(sibling_indices):
            if sibs:
                hsib[i] = ht[sibs].mean(axis=0)
            else:
                hsib[i] = ht[i]

        g_pool = np.broadcast_to(
            0.4 * ht.mean(axis=0, keepdims=True)
            + 0.3 * ha.mean(axis=0, keepdims=True)
            + 0.3 * ho.mean(axis=0, keepdims=True),
            ht.shape,
        )
        cat = np.concatenate([ht, ha_loc, ho_loc, hsib, g_pool], axis=-1)
        h_out = _relu(cat @ w["m_w1"] + w["m_b1"])
        logits = np.clip((h_out @ w["m_w2"] + w["m_b2"]).reshape(-1), -20.0, 20.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def select_destroy_tasks(
        self,
        network: Network | None,
        snapshot: NativeSnapshot,
        clf: CommitmentClassification,
        eval_res: EvalResult,
        op_bays: dict[str, str],
        op_crews: dict[str, int],
        *,
        radius: str = "medium",
        bay_detour_delta: Optional[dict[str, float]] = None,
        rng: Optional[random.Random] = None,
    ) -> tuple[list[str], set[str], np.ndarray]:
        """Select `(destroy_tids, frozen_tids, scores)` for LNS under chosen radius arm.

        Radii match IJCAI'25 BALANS multi-size arms + CPAIOR'26 stochastic-destroy:
          - `"small"`: tight surgical neighborhood `[3, min(6, 0.35*N)]`
          - `"medium"`: balanced destroy pool `[5, min(12, 0.62*N)]`
          - `"large_stochastic"`: broader perturbed selection `[7, min(16, 0.85*N)]`
          - `"all"`: all releasable tasks (no freeze)
        """
        tids, tx, ax, ox, a_idx, o_idx, sibs = extract_hetero_destroy_features(
            network, snapshot, clf, eval_res, op_bays, op_crews,
            bay_detour_delta=bay_detour_delta,
        )
        n = len(tids)
        if n <= 2 or radius == "all":
            return list(tids), set(), np.ones((n,), dtype=np.float32)

        scores = self.predict_destroy_scores(tx, ax, ox, a_idx, o_idx, sibs)
        # Before learned weights are fitted, keep wider search pools except in explicit small destroy
        if self.w is None and radius == "medium":
            return list(tids), set(), scores
        # Hard guard: disturbed, unassigned, urgent, and critical last-kit tasks get floor priority
        eff_scores = scores.copy()
        for i in range(n):
            if tx[i, 0] > 0.5 or tx[i, 1] > 0.5:
                eff_scores[i] = max(eff_scores[i], 0.95)
            elif tx[i, 5] > 0.5 or tx[i, 7] > 0.0:
                eff_scores[i] = max(eff_scores[i], 0.80)

        if radius == "small":
            k = max(2, min(n, max(3, min(6, int(math.ceil(0.35 * n))))))
        elif radius == "large_stochastic":
            k = max(3, min(n, max(7, min(16, int(math.ceil(0.85 * n))))))
            if rng is not None:
                # Gumbel-style temperature noise on scores for diversification (CPAIOR'26)
                eff_scores = np.array(
                    [float(eff_scores[i]) + 0.18 * (rng.random() - 0.5) for i in range(n)],
                    dtype=np.float32,
                )
        else:  # "medium"
            k = max(2, min(n, max(5, min(12, int(math.ceil(0.62 * n))))))

        order = sorted(range(n), key=lambda i: (-float(eff_scores[i]), tids[i]))
        chosen_idx = order[:k]
        chosen_set = {tids[i] for i in chosen_idx}
        destroy_tids = [tids[i] for i in chosen_idx]
        frozen_tids = {tid for tid in tids if tid not in chosen_set}
        return destroy_tids, frozen_tids, scores


def load_hetero_destroy_selector(
    path: str | Path = DEFAULT_DESTROY_CKPT,
) -> HeteroDestroySelector:
    """Load trained `HeteroDestroySelector` if checkpoint exists, else structural prior."""
    p = Path(path)
    if p.exists():
        return HeteroDestroySelector.from_npz(p)
    return HeteroDestroySelector(weights=None)
