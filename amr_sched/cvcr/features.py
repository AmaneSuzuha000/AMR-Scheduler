"""Exact 13-Feature Cross-Domain Extractor and Frozen CIE MLP Models.

Implements the exact 13 features from CIE `lg_features_and_data.py` over NativeSnapshot:
- 6 Base features: is_disturbed_robot, is_unassigned, robot_fault_rem, robot_load,
  slack_ratio, is_urgent
- 7 Cross-domain features: kit_size, kit_bottleneck_gap, res_wait_gap,
  downstream_chain_w, shared_res_degree, path_conflict_degree, is_1hop_from_disturbed

Loads frozen MLP weights from `frozen_weights.json` (`mlp_full`,
`mlp_keep_release_balanced`, `mlp_keep_release_balanced_thr`) for authentic cross-platform
scientific evaluation without modifying frozen CIE parameters.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from amr_sched.cvcr.commitment import CommitmentClassification
from amr_sched.cvcr.snapshot import NativeSnapshot

FEATURE_NAMES = [
    "is_disturbed_robot",
    "is_unassigned",
    "robot_fault_rem",
    "robot_load",
    "slack_ratio",
    "is_urgent",
    "kit_size",
    "kit_bottleneck_gap",
    "res_wait_gap",
    "downstream_chain_w",
    "shared_res_degree",
    "path_conflict_degree",
    "is_1hop_from_disturbed",
]

_WEIGHTS_PATH = Path(__file__).resolve().parent / "frozen_weights.json"
_CACHED_MODELS: dict[str, Any] | None = None


def load_frozen_cie_models() -> dict[str, Any]:
    """Load frozen CIE MLP weights from disk (cached)."""
    global _CACHED_MODELS
    if _CACHED_MODELS is None:
        _CACHED_MODELS = json.loads(_WEIGHTS_PATH.read_text(encoding="utf-8"))
    return _CACHED_MODELS


def predict_mlp_regression(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Predict outcome delta_J score using frozen 13->24->1 ReLU MLP."""
    if len(X) == 0:
        return np.array([], dtype=np.float64)
    W1 = np.asarray(model["W1"], dtype=np.float64)
    b1 = np.asarray(model["b1"], dtype=np.float64)
    W2 = np.asarray(model["W2"], dtype=np.float64)
    b2 = np.asarray(model["b2"], dtype=np.float64)
    mu = np.asarray(model["mu"], dtype=np.float64)
    sd = np.maximum(1e-6, np.asarray(model["sd"], dtype=np.float64))
    Xn = (X - mu) / sd
    h1 = np.maximum(0.0, Xn @ W1 + b1)
    return (h1 @ W2 + b2).reshape(-1)


def predict_mlp_keep_release_proba(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Predict Keep/Release probability using frozen 13->24->1 BCE MLP."""
    if len(X) == 0:
        return np.array([], dtype=np.float64)
    W1 = np.asarray(model["W1"], dtype=np.float64)
    b1 = np.asarray(model["b1"], dtype=np.float64)
    W2 = np.asarray(model["W2"], dtype=np.float64)
    b2 = np.asarray(model["b2"], dtype=np.float64)
    mu = np.asarray(model["mu"], dtype=np.float64)
    sd = np.maximum(1e-6, np.asarray(model["sd"], dtype=np.float64))
    Xn = (X - mu) / sd
    h1 = np.maximum(0.0, Xn @ W1 + b1)
    logits = np.clip(h1 @ W2 + b2, -20.0, 20.0)
    return (1.0 / (1.0 + np.exp(-logits))).reshape(-1)


def extract_task_features(
    snapshot: NativeSnapshot,
    clf: CommitmentClassification,
    op_res_ready: dict[str, int],
    est_unload: dict[str, int],
    robot_routes: dict[str, list[tuple[str, int]]] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Extract aligned 13-dim feature vectors for all releasable tasks in snapshot."""
    t_commit = snapshot.tick
    op_by_id = {op.op_id: op for op in snapshot.operations}
    rob_by_id = {r.robot_id: r for r in snapshot.robots}

    by_op: dict[str, list[Any]] = {op.op_id: [] for op in snapshot.operations}
    for t in snapshot.tasks:
        if t.kit_op_id and t.kit_op_id in by_op:
            by_op[t.kit_op_id].append(t)

    # Downstream chain weight
    chain_w = {op.op_id: (12.0 if op.is_urgent else 4.0) for op in snapshot.operations}
    for op in snapshot.operations:
        if op.predecessor_op_id and op.predecessor_op_id in chain_w:
            chain_w[op.predecessor_op_id] += chain_w[op.op_id]

    # Current robot queue load
    r_load = {r.robot_id: 0 for r in snapshot.robots}
    for t in snapshot.tasks:
        if t.status.lower() != "completed" and t.assigned_agv is not None:
            r_load[t.assigned_agv] = r_load.get(t.assigned_agv, 0) + 1

    # Path conflict pairs from current routes if available
    future_nodes_by_agv: dict[str, set[str]] = {}
    if robot_routes:
        for rid, route in robot_routes.items():
            future_nodes_by_agv[rid] = {n for n, st in route if st >= t_commit}
    else:
        for r in snapshot.robots:
            future_nodes_by_agv[r.robot_id] = set(r.route)

    path_conflict_pairs: set[tuple[str, str]] = set()
    rids = sorted(future_nodes_by_agv.keys())
    for i in range(len(rids)):
        for j in range(i + 1, len(rids)):
            if future_nodes_by_agv[rids[i]] & future_nodes_by_agv[rids[j]]:
                path_conflict_pairs.add((rids[i], rids[j]))
                path_conflict_pairs.add((rids[j], rids[i]))

    one_hop_dist: set[str] = set()
    for u in clf.disturbed_tasks:
        one_hop_dist |= clf.adj.get(u, set())

    tids: list[str] = []
    feats: list[list[float]] = []

    for t in snapshot.tasks:
        if t.task_id not in clf.releasable_tasks:
            continue
        op = op_by_id.get(t.kit_op_id) if t.kit_op_id else None
        rb = t.assigned_agv

        is_dist = 1.0 if (rb is not None and rb in clf.disturbed_robots) else 0.0
        is_unass = 1.0 if rb is None else 0.0
        f_rem = (
            max(0.0, float(rob_by_id[rb].faulted_until - t_commit)) / 10.0
            if (rb is not None and rb in rob_by_id)
            else 0.0
        )
        load_v = float(r_load.get(rb, 0)) / 5.0 if rb is not None else 0.0

        if op is not None:
            slack = float(op.due_step - op.proc_steps - t_commit) / 50.0
            urg = 1.0 if op.is_urgent else 0.0
            kit_list = by_op.get(op.op_id, [t])
            k_size = float(len(kit_list)) / 4.0
            u_self = float(est_unload.get(t.task_id, t_commit + 10))
            other_arrs = [
                float(est_unload.get(k.task_id, t_commit))
                for k in kit_list
                if k.task_id != t.task_id
            ]
            max_other = max(other_arrs) if other_arrs else 0.0
            kit_gap = max(0.0, u_self - max_other) / 20.0
            r_ready = float(op_res_ready.get(op.op_id, t_commit))
            res_gap = max(0.0, u_self - r_ready) / 20.0
            cw = chain_w.get(op.op_id, 4.0) / 24.0
        else:
            due = t.due_step if t.due_step is not None else t_commit + 100
            slack = float(due - t_commit) / 50.0
            urg = 0.0
            k_size = 0.25
            u_self = float(est_unload.get(t.task_id, t_commit + 10))
            kit_gap = 0.0
            res_gap = 0.0
            cw = 4.0 / 24.0

        sr_deg = 0
        pc_deg = 0
        for nb in clf.adj.get(t.task_id, ()):
            rs = clf.edge_reasons.get((t.task_id, nb), set())
            if "SHARED_RESOURCE" in rs:
                sr_deg += 1
            if "ROBOT_QUEUE" in rs:
                pc_deg += 1

        if rb is not None:
            for other_t in snapshot.tasks:
                if other_t.assigned_agv and (rb, other_t.assigned_agv) in path_conflict_pairs:
                    pc_deg += 1

        tids.append(t.task_id)
        feats.append([
            is_dist,
            is_unass,
            f_rem,
            load_v,
            slack,
            urg,
            k_size,
            kit_gap,
            res_gap,
            cw,
            float(sr_deg) / 10.0,
            float(pc_deg) / 10.0,
            1.0 if t.task_id in one_hop_dist else 0.0,
        ])

    return tids, np.asarray(feats, dtype=np.float64)
