"""Architecture-Faithful, Domain-Adapted Reimplementation of Ren et al. (FGCS 2026): 3D-DG GNN + DRL.

Reference:
  Fujie Ren, Haibin Liu, He Huang, and Yang Cao (2026),
  "Dynamic scheduling of flexible job-shop considering automatic guided vehicle transportation
  via deep reinforcement learning", Future Generation Computer Systems, Vol. 182 (2026).

Core Architecture:
  1. **3-Plane 3D Disjunctive Graph (3D-DG) Message Passing** (2 hops, `hidden_dim=64`):
     - **Plane 1 (Operation Conjunctive + Machine-Disjunctive Plane)**: message passing across
       precedence edges (`op_dag_adj` pred/succ) AND shared-eligible-bay disjunctive conflicts
       (`op_bay_mask @ op_bay_mask.T > 0`).
     - **Plane 2 (Operation–Bay / Crew Compatibility Plane)**: edge-conditioned bipartite message
       passing between operations, eligible bays (`op_bay_feats`), and eligible crews.
     - **Plane 3 (Task–AMR Transport-Disjunctive Plane)**: edge-conditioned bipartite message
       passing between releasable kitting tasks and eligible AMRs (`task_agv_feats`), coupled
       with parent-operation states and AGV fleet congestion pooling.
  2. **Dynamic Disruption-Aware Residual Gating**:
     Each GNN hop applies a sigmoid disruption/congestion gate conditioned on the pooled 3D-DG
     global state + residual `LayerNorm`, updating node embeddings smoothly upon dynamic events
     (AGV faults, urgent insertions, route congestion).
  3. **Composite Single-Agent Masked PPO Policy & Value Heads**:
     Jointly outputs task priority (`job_logits`), bay choice (`bay_logits`), crew choice
     (`crew_logits`), and autoregressive load-aware AMR allocation (`agv_logits`), all strictly
     respecting hard `-1e9` commitment/eligibility masks.

Explicit Domain-Mapping Adaptations (documented transparently):
  - Multi-kit fan-out & crew assignment: maps each operation's kit tasks to Plane 3 and crew
    compatibility to Plane 2.
  - Commitment locking & spacetime realization: `LOCKED` entities are strictly masked out and
    verified inside the identical `CVCREvaluator` (`35 ms` wall budget) with `KEEP` fallback —
    WITHOUT adding CoRe-Sched's 4th-stage queue-sequencing head, Contextual LinUCB Full/Staged
    coupling switcher, dual-arm probe, or `J_A`-aligned surrogate.
  - Unified quarantined training: trained under the exact same quarantined 264-trajectory IL +
    288-episode closed-loop SMDP-PPO protocol and exports a `< 1e-5` pure-NumPy runtime mirror.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

    class _MissingNNModule:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError("PyTorch (`torch`) is required to instantiate FGCS26RenGNNNet.")

    class _MissingNNNamespace:
        Module = _MissingNNModule

    nn = _MissingNNNamespace  # type: ignore[assignment]

from amr_sched.baselines.numpy_backends import (
    file_sha256,
    load_numpy_state_dict,
    resolve_numpy_checkpoint_path,
    softmax as _np_softmax,
)
from amr_sched.baselines.top_venue_baselines import _build_external_proposal
from amr_sched.cvcr.checkpoints import resolve_default_checkpoint
from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.cvcr.snapshot import NativeSnapshot
from amr_sched.network import Network
from amr_sched.rl.v19_hetero_graph import (
    AGV_FEAT_DIM,
    BAY_FEAT_DIM,
    CREW_FEAT_DIM,
    OP_BAY_EDGE_DIM,
    OP_FEAT_DIM,
    TASK_AGV_EDGE_DIM,
    TASK_FEAT_DIM,
    V19HeteroGraph,
    extract_v19_hetero_graph,
)

FGCS26_REN_GNN_CKPT = resolve_default_checkpoint("baselines_2026/fgcs26_ren_gnn.pt")
_CKPT_CACHE: dict[str, tuple[Any, str]] = {}


def _relu_np(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.float32(0.0), x).astype(np.float32, copy=False)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))).astype(np.float32)


def _layernorm_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * weight + bias


def _linear_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (x @ weight.T + bias).astype(np.float32)


def _masked_mean_torch(x: "torch.Tensor", valid_mask: Optional["torch.Tensor"] = None) -> "torch.Tensor":
    if x.shape[1] == 0:
        return torch.zeros((x.shape[0], x.shape[2]), dtype=x.dtype, device=x.device)
    if valid_mask is None:
        return x.mean(dim=1)
    m = valid_mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / torch.clamp(m.sum(dim=1), min=1.0)


def _masked_mean_np(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros((x.shape[-1],), dtype=np.float32)
    return x.mean(axis=0).astype(np.float32)


class _Disjunctive3DLayer(nn.Module):
    """One 3-plane 3D Disjunctive Graph GNN message-passing hop with dynamic residual gating."""

    def __init__(self, d: int = 64) -> None:
        super().__init__()
        self.d = int(d)
        # Plane 1 + 2 -> Operation update: [self, pred_agg, succ_agg, disj_agg, bay_agg, crew_agg] = 6*d
        self.op_upd = nn.Linear(6 * self.d, self.d)
        self.op_gate = nn.Linear(2 * self.d, self.d)
        self.op_ln = nn.LayerNorm(self.d)

        # Plane 2 -> Bay & Crew updates: [self, op_agg] = 2*d
        self.bay_edge_proj = nn.Linear(OP_BAY_EDGE_DIM, self.d)
        self.bay_upd = nn.Linear(2 * self.d, self.d)
        self.bay_ln = nn.LayerNorm(self.d)

        self.crew_upd = nn.Linear(2 * self.d, self.d)
        self.crew_ln = nn.LayerNorm(self.d)

        # Plane 3 -> Task & AGV updates
        self.ta_edge_proj = nn.Linear(TASK_AGV_EDGE_DIM, self.d)
        self.task_upd = nn.Linear(3 * self.d, self.d)  # [self, parent_op, agv_agg]
        self.task_gate = nn.Linear(2 * self.d, self.d)
        self.task_ln = nn.LayerNorm(self.d)

        self.agv_upd = nn.Linear(3 * self.d, self.d)   # [self, task_agg, fleet_mean]
        self.agv_gate = nn.Linear(2 * self.d, self.d)
        self.agv_ln = nn.LayerNorm(self.d)

    def forward(
        self,
        h_op: "torch.Tensor",
        h_task: "torch.Tensor",
        h_bay: "torch.Tensor",
        h_crew: "torch.Tensor",
        h_agv: "torch.Tensor",
        tensors: dict[str, "torch.Tensor"],
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        B = h_op.shape[0]
        No, Nt, Nb, Nc, Na = h_op.shape[1], h_task.shape[1], h_bay.shape[1], h_crew.shape[1], h_agv.shape[1]
        D = self.d
        dev = h_op.device
        dt = h_op.dtype

        op_valid = tensors.get("op_valid")
        task_valid = tensors.get("task_valid")
        bay_valid = tensors.get("bay_valid")
        crew_valid = tensors.get("crew_valid")
        agv_valid = tensors.get("agv_valid")

        # Global dynamic disruption/state context for gating
        g_dyn = (
            _masked_mean_torch(h_op, op_valid)
            + _masked_mean_torch(h_task, task_valid)
            + _masked_mean_torch(h_agv, agv_valid)
        ) / 3.0

        # Plane 2 edge projections
        ob_m = tensors["op_bay_mask"].to(dt)  # [B, No, Nb]
        oc_m = tensors["op_crew_mask"].to(dt)  # [B, No, Nc]
        ta_m = tensors["task_agv_mask"].to(dt)  # [B, Nt, Na]
        if op_valid is not None:
            ob_m = ob_m * op_valid.to(dt).unsqueeze(-1)
            oc_m = oc_m * op_valid.to(dt).unsqueeze(-1)
        if bay_valid is not None:
            ob_m = ob_m * bay_valid.to(dt).unsqueeze(1)
        if crew_valid is not None:
            oc_m = oc_m * crew_valid.to(dt).unsqueeze(1)
        if task_valid is not None:
            ta_m = ta_m * task_valid.to(dt).unsqueeze(-1)
        if agv_valid is not None:
            ta_m = ta_m * agv_valid.to(dt).unsqueeze(1)

        # Plane 1 + 2: Operation update
        if No > 0:
            dag = tensors["op_dag_adj"].to(dt)  # [B, No, No, 3]
            a_pred = dag[:, :, :, 0]
            a_succ = dag[:, :, :, 2]
            if op_valid is not None:
                ov_pair = op_valid.to(dt).unsqueeze(2) * op_valid.to(dt).unsqueeze(1)
                a_pred = a_pred * ov_pair
                a_succ = a_succ * ov_pair
            m_pred = torch.bmm(a_pred, h_op) / torch.clamp(a_pred.sum(dim=-1, keepdim=True), min=1.0)
            m_succ = torch.bmm(a_succ, h_op) / torch.clamp(a_succ.sum(dim=-1, keepdim=True), min=1.0)
            # Machine-disjunctive edges: operations sharing at least one eligible bay (excluding self)
            if Nb > 0:
                disj = (torch.bmm(ob_m, ob_m.transpose(1, 2)) > 0.5).to(dt)
                eye = torch.eye(No, dtype=dt, device=dev).unsqueeze(0)
                disj = disj * (1.0 - eye)
                m_disj = torch.bmm(disj, h_op) / torch.clamp(disj.sum(dim=-1, keepdim=True), min=1.0)
                e_ob = F.relu(self.bay_edge_proj(tensors["op_bay_x"]))  # [B, No, Nb, D]
                bay_msg = (h_bay.unsqueeze(1) + e_ob) * ob_m.unsqueeze(-1)
                m_bay = bay_msg.sum(dim=2) / torch.clamp(ob_m.sum(dim=-1, keepdim=True), min=1.0)
            else:
                m_disj = torch.zeros_like(h_op)
                m_bay = torch.zeros_like(h_op)
            if Nc > 0:
                m_crew = torch.bmm(oc_m, h_crew) / torch.clamp(oc_m.sum(dim=-1, keepdim=True), min=1.0)
            else:
                m_crew = torch.zeros_like(h_op)
            op_cand = F.relu(self.op_upd(torch.cat([h_op, m_pred, m_succ, m_disj, m_bay, m_crew], dim=-1)))
            g_o = torch.sigmoid(self.op_gate(torch.cat([h_op, g_dyn.unsqueeze(1).expand(B, No, D)], dim=-1)))
            h_op_new = self.op_ln(h_op + g_o * op_cand)
            if op_valid is not None:
                h_op_new = h_op_new * op_valid.to(dt).unsqueeze(-1)
        else:
            h_op_new = h_op
            e_ob = None

        # Plane 2: Bay & Crew updates
        if Nb > 0:
            if No > 0 and e_ob is not None:
                op_to_b = (h_op.unsqueeze(2) + e_ob) * ob_m.unsqueeze(-1)
                mb_op = op_to_b.sum(dim=1) / torch.clamp(ob_m.sum(dim=1).unsqueeze(-1), min=1.0)
            else:
                mb_op = torch.zeros_like(h_bay)
            h_bay_new = self.bay_ln(h_bay + F.relu(self.bay_upd(torch.cat([h_bay, mb_op], dim=-1))))
            if bay_valid is not None:
                h_bay_new = h_bay_new * bay_valid.to(dt).unsqueeze(-1)
        else:
            h_bay_new = h_bay

        if Nc > 0:
            if No > 0:
                mc_op = torch.bmm(oc_m.transpose(1, 2), h_op) / torch.clamp(oc_m.sum(dim=1).unsqueeze(-1), min=1.0)
            else:
                mc_op = torch.zeros_like(h_crew)
            h_crew_new = self.crew_ln(h_crew + F.relu(self.crew_upd(torch.cat([h_crew, mc_op], dim=-1))))
            if crew_valid is not None:
                h_crew_new = h_crew_new * crew_valid.to(dt).unsqueeze(-1)
        else:
            h_crew_new = h_crew

        # Plane 3: Task & AGV updates
        e_ta = F.relu(self.ta_edge_proj(tensors["task_agv_x"])) if (Nt > 0 and Na > 0) else None
        if Nt > 0:
            if No > 0:
                t2o = tensors["task_to_op"]
                has_o = (t2o >= 0) & (t2o < No)
                par_o = h_op_new.gather(1, torch.clamp(t2o, 0, No - 1).unsqueeze(-1).expand(B, Nt, D)) * has_o.unsqueeze(-1).to(dt)
            else:
                par_o = torch.zeros_like(h_task)
            if Na > 0 and e_ta is not None:
                mt_agv = ((h_agv.unsqueeze(1) + e_ta) * ta_m.unsqueeze(-1)).sum(dim=2) / torch.clamp(ta_m.sum(dim=-1, keepdim=True), min=1.0)
            else:
                mt_agv = torch.zeros_like(h_task)
            t_cand = F.relu(self.task_upd(torch.cat([h_task, par_o, mt_agv], dim=-1)))
            g_t = torch.sigmoid(self.task_gate(torch.cat([h_task, g_dyn.unsqueeze(1).expand(B, Nt, D)], dim=-1)))
            h_task_new = self.task_ln(h_task + g_t * t_cand)
            if task_valid is not None:
                h_task_new = h_task_new * task_valid.to(dt).unsqueeze(-1)
        else:
            h_task_new = h_task

        if Na > 0:
            if Nt > 0 and e_ta is not None:
                ma_t = ((h_task.unsqueeze(2) + e_ta) * ta_m.unsqueeze(-1)).sum(dim=1) / torch.clamp(ta_m.sum(dim=1).unsqueeze(-1), min=1.0)
            else:
                ma_t = torch.zeros_like(h_agv)
            fleet_mean = _masked_mean_torch(h_agv, agv_valid).unsqueeze(1).expand(B, Na, D)
            a_cand = F.relu(self.agv_upd(torch.cat([h_agv, ma_t, fleet_mean], dim=-1)))
            g_a = torch.sigmoid(self.agv_gate(torch.cat([h_agv, g_dyn.unsqueeze(1).expand(B, Na, D)], dim=-1)))
            h_agv_new = self.agv_ln(h_agv + g_a * a_cand)
            if agv_valid is not None:
                h_agv_new = h_agv_new * agv_valid.to(dt).unsqueeze(-1)
        else:
            h_agv_new = h_agv

        return h_op_new, h_task_new, h_bay_new, h_crew_new, h_agv_new


def _np_3d_layer(
    h_op: np.ndarray,
    h_task: np.ndarray,
    h_bay: np.ndarray,
    h_crew: np.ndarray,
    h_agv: np.ndarray,
    g: V19HeteroGraph,
    w: dict[str, np.ndarray],
    p: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    No, Nt, Nb, Nc, Na = h_op.shape[0], h_task.shape[0], h_bay.shape[0], h_crew.shape[0], h_agv.shape[0]
    D = h_op.shape[-1] if No > 0 else (h_agv.shape[-1] if Na > 0 else 64)
    g_dyn = (_masked_mean_np(h_op) + _masked_mean_np(h_task) + _masked_mean_np(h_agv)) / np.float32(3.0)

    ob_m = g.op_bay_mask.astype(np.float32)
    oc_m = g.op_crew_mask.astype(np.float32)
    ta_m = g.task_agv_mask.astype(np.float32)

    e_ob = _relu_np(_linear_np(g.op_bay_feats, w[f"{p}.bay_edge_proj.weight"], w[f"{p}.bay_edge_proj.bias"])) if (No > 0 and Nb > 0) else None
    if No > 0:
        a_pred = g.op_dag_adj[:, :, 0].astype(np.float32)
        a_succ = g.op_dag_adj[:, :, 2].astype(np.float32)
        m_pred = (a_pred @ h_op) / np.maximum(1.0, a_pred.sum(axis=-1, keepdims=True))
        m_succ = (a_succ @ h_op) / np.maximum(1.0, a_succ.sum(axis=-1, keepdims=True))
        if Nb > 0 and e_ob is not None:
            disj = ((ob_m @ ob_m.T) > 0.5).astype(np.float32) * (1.0 - np.eye(No, dtype=np.float32))
            m_disj = (disj @ h_op) / np.maximum(1.0, disj.sum(axis=-1, keepdims=True))
            m_bay = ((h_bay[None, :, :] + e_ob) * ob_m[:, :, None]).sum(axis=1) / np.maximum(1.0, ob_m.sum(axis=-1, keepdims=True))
        else:
            m_disj = np.zeros_like(h_op)
            m_bay = np.zeros_like(h_op)
        if Nc > 0:
            m_crew = (oc_m @ h_crew) / np.maximum(1.0, oc_m.sum(axis=-1, keepdims=True))
        else:
            m_crew = np.zeros_like(h_op)
        op_cand = _relu_np(_linear_np(np.concatenate([h_op, m_pred, m_succ, m_disj, m_bay, m_crew], axis=-1), w[f"{p}.op_upd.weight"], w[f"{p}.op_upd.bias"]))
        g_o = _sigmoid_np(_linear_np(np.concatenate([h_op, np.broadcast_to(g_dyn, (No, D))], axis=-1), w[f"{p}.op_gate.weight"], w[f"{p}.op_gate.bias"]))
        h_op_new = _layernorm_np(h_op + g_o * op_cand, w[f"{p}.op_ln.weight"], w[f"{p}.op_ln.bias"])
    else:
        h_op_new = h_op

    if Nb > 0:
        if No > 0 and e_ob is not None:
            mb_op = ((h_op[:, None, :] + e_ob) * ob_m[:, :, None]).sum(axis=0) / np.maximum(1.0, ob_m.sum(axis=0)[:, None])
        else:
            mb_op = np.zeros_like(h_bay)
        h_bay_new = _layernorm_np(h_bay + _relu_np(_linear_np(np.concatenate([h_bay, mb_op], axis=-1), w[f"{p}.bay_upd.weight"], w[f"{p}.bay_upd.bias"])), w[f"{p}.bay_ln.weight"], w[f"{p}.bay_ln.bias"])
    else:
        h_bay_new = h_bay

    if Nc > 0:
        if No > 0:
            mc_op = (oc_m.T @ h_op) / np.maximum(1.0, oc_m.sum(axis=0)[:, None])
        else:
            mc_op = np.zeros_like(h_crew)
        h_crew_new = _layernorm_np(h_crew + _relu_np(_linear_np(np.concatenate([h_crew, mc_op], axis=-1), w[f"{p}.crew_upd.weight"], w[f"{p}.crew_upd.bias"])), w[f"{p}.crew_ln.weight"], w[f"{p}.crew_ln.bias"])
    else:
        h_crew_new = h_crew

    e_ta = _relu_np(_linear_np(g.task_agv_feats, w[f"{p}.ta_edge_proj.weight"], w[f"{p}.ta_edge_proj.bias"])) if (Nt > 0 and Na > 0) else None
    if Nt > 0:
        par_o = np.zeros((Nt, D), dtype=np.float32)
        for j in range(Nt):
            oi = int(g.task_to_op[j])
            if 0 <= oi < No:
                par_o[j] = h_op_new[oi]
        if Na > 0 and e_ta is not None:
            mt_agv = ((h_agv[None, :, :] + e_ta) * ta_m[:, :, None]).sum(axis=1) / np.maximum(1.0, ta_m.sum(axis=-1, keepdims=True))
        else:
            mt_agv = np.zeros_like(h_task)
        t_cand = _relu_np(_linear_np(np.concatenate([h_task, par_o, mt_agv], axis=-1), w[f"{p}.task_upd.weight"], w[f"{p}.task_upd.bias"]))
        g_t = _sigmoid_np(_linear_np(np.concatenate([h_task, np.broadcast_to(g_dyn, (Nt, D))], axis=-1), w[f"{p}.task_gate.weight"], w[f"{p}.task_gate.bias"]))
        h_task_new = _layernorm_np(h_task + g_t * t_cand, w[f"{p}.task_ln.weight"], w[f"{p}.task_ln.bias"])
    else:
        h_task_new = h_task

    if Na > 0:
        if Nt > 0 and e_ta is not None:
            ma_t = ((h_task[:, None, :] + e_ta) * ta_m[:, :, None]).sum(axis=0) / np.maximum(1.0, ta_m.sum(axis=0)[:, None])
        else:
            ma_t = np.zeros_like(h_agv)
        fleet_mean = np.broadcast_to(_masked_mean_np(h_agv), (Na, D))
        a_cand = _relu_np(_linear_np(np.concatenate([h_agv, ma_t, fleet_mean], axis=-1), w[f"{p}.agv_upd.weight"], w[f"{p}.agv_upd.bias"]))
        g_a = _sigmoid_np(_linear_np(np.concatenate([h_agv, np.broadcast_to(g_dyn, (Na, D))], axis=-1), w[f"{p}.agv_gate.weight"], w[f"{p}.agv_gate.bias"]))
        h_agv_new = _layernorm_np(h_agv + g_a * a_cand, w[f"{p}.agv_ln.weight"], w[f"{p}.agv_ln.bias"])
    else:
        h_agv_new = h_agv

    return h_op_new.astype(np.float32), h_task_new.astype(np.float32), h_bay_new.astype(np.float32), h_crew_new.astype(np.float32), h_agv_new.astype(np.float32)


class FGCS26RenGNNNet(nn.Module):
    """3D Disjunctive Graph GNN + Disruption-Aware Composite DRL network (Ren et al., FGCS 2026)."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        D = self.hidden_dim

        self.op_in = nn.Linear(OP_FEAT_DIM, D)
        self.task_in = nn.Linear(TASK_FEAT_DIM, D)
        self.bay_in = nn.Linear(BAY_FEAT_DIM, D)
        self.crew_in = nn.Linear(CREW_FEAT_DIM, D)
        self.agv_in = nn.Linear(AGV_FEAT_DIM, D)

        self.layer1 = _Disjunctive3DLayer(D)
        self.layer2 = _Disjunctive3DLayer(D)

        # Composite joint heads
        self.job_fc1 = nn.Linear(2 * D, D)
        self.job_fc2 = nn.Linear(D, 1)

        self.bay_fc1 = nn.Linear(3 * D + OP_BAY_EDGE_DIM, D)
        self.bay_fc2 = nn.Linear(D, 1)

        self.crew_fc1 = nn.Linear(3 * D, D)
        self.crew_fc2 = nn.Linear(D, 1)

        self.agv_fc1 = nn.Linear(4 * D + TASK_AGV_EDGE_DIM + 2, D)
        self.agv_fc2 = nn.Linear(D, 1)

        self.critic_fc1 = nn.Linear(5 * D, D)
        self.critic_fc2 = nn.Linear(D, 1)

    def config_dict(self) -> dict[str, Any]:
        return {"hidden_dim": self.hidden_dim, "arch": "FGCS26_REN_GNN"}

    def encode(self, tensors: dict[str, "torch.Tensor"]) -> dict[str, "torch.Tensor"]:
        D = self.hidden_dim
        B = tensors["op_x"].shape[0]
        dev = tensors["op_x"].device
        dt = tensors["op_x"].dtype

        h_op = F.relu(self.op_in(tensors["op_x"])) if tensors["op_x"].shape[1] > 0 else torch.zeros((B, 0, D), dtype=dt, device=dev)
        h_task = F.relu(self.task_in(tensors["task_x"])) if tensors["task_x"].shape[1] > 0 else torch.zeros((B, 0, D), dtype=dt, device=dev)
        h_bay = F.relu(self.bay_in(tensors["bay_x"])) if tensors["bay_x"].shape[1] > 0 else torch.zeros((B, 0, D), dtype=dt, device=dev)
        h_crew = F.relu(self.crew_in(tensors["crew_x"])) if tensors["crew_x"].shape[1] > 0 else torch.zeros((B, 0, D), dtype=dt, device=dev)
        h_agv = F.relu(self.agv_in(tensors["agv_x"])) if tensors["agv_x"].shape[1] > 0 else torch.zeros((B, 0, D), dtype=dt, device=dev)

        h_op, h_task, h_bay, h_crew, h_agv = self.layer1(h_op, h_task, h_bay, h_crew, h_agv, tensors)
        h_op, h_task, h_bay, h_crew, h_agv = self.layer2(h_op, h_task, h_bay, h_crew, h_agv, tensors)

        mo = _masked_mean_torch(h_op, tensors.get("op_valid"))
        mt = _masked_mean_torch(h_task, tensors.get("task_valid"))
        mb = _masked_mean_torch(h_bay, tensors.get("bay_valid"))
        mc = _masked_mean_torch(h_crew, tensors.get("crew_valid"))
        ma = _masked_mean_torch(h_agv, tensors.get("agv_valid"))
        g_pool = (mo + mt + mb + mc + ma) / 5.0
        v_in = torch.cat([mo, mt, mb, mc, ma], dim=-1)
        val = self.critic_fc2(F.relu(self.critic_fc1(v_in))).squeeze(-1)
        return {
            "h_op": h_op,
            "h_task": h_task,
            "h_bay": h_bay,
            "h_crew": h_crew,
            "h_agv": h_agv,
            "g_pool": g_pool,
            "value": val,
        }

    def forward_static_heads(
        self,
        enc: dict[str, "torch.Tensor"],
        tensors: dict[str, "torch.Tensor"],
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        h_op, h_task, h_bay, h_crew, gp = enc["h_op"], enc["h_task"], enc["h_bay"], enc["h_crew"], enc["g_pool"]
        B, No, D = h_op.shape
        Nt, Nb, Nc = h_task.shape[1], h_bay.shape[1], h_crew.shape[1]
        dev = h_op.device

        if Nt > 0:
            job_logits = self.job_fc2(F.relu(self.job_fc1(torch.cat([h_task, gp.unsqueeze(1).expand(B, Nt, D)], dim=-1)))).squeeze(-1)
        else:
            job_logits = torch.zeros((B, 0), dtype=h_op.dtype, device=dev)

        if No > 0 and Nb > 0:
            b_in = torch.cat(
                [
                    h_op.unsqueeze(2).expand(B, No, Nb, D),
                    h_bay.unsqueeze(1).expand(B, No, Nb, D),
                    gp.unsqueeze(1).unsqueeze(2).expand(B, No, Nb, D),
                    tensors["op_bay_x"],
                ],
                dim=-1,
            )
            bay_logits = self.bay_fc2(F.relu(self.bay_fc1(b_in))).squeeze(-1)
            bay_logits = bay_logits.masked_fill(~tensors["op_bay_mask"].bool(), -1e9)
        else:
            bay_logits = torch.zeros((B, No, Nb), dtype=h_op.dtype, device=dev)

        if No > 0 and Nc > 0:
            c_in = torch.cat(
                [
                    h_op.unsqueeze(2).expand(B, No, Nc, D),
                    h_crew.unsqueeze(1).expand(B, No, Nc, D),
                    gp.unsqueeze(1).unsqueeze(2).expand(B, No, Nc, D),
                ],
                dim=-1,
            )
            crew_logits = self.crew_fc2(F.relu(self.crew_fc1(c_in))).squeeze(-1)
            crew_logits = crew_logits.masked_fill(~tensors["op_crew_mask"].bool(), -1e9)
        else:
            crew_logits = torch.zeros((B, No, Nc), dtype=h_op.dtype, device=dev)

        return job_logits, bay_logits, crew_logits

    def forward_agv_step(
        self,
        h_task_j: "torch.Tensor",
        h_agv: "torch.Tensor",
        task_agv_x_j: "torch.Tensor",
        h_sel_bay_j: "torch.Tensor",
        dyn_load: "torch.Tensor",
        g_pool: "torch.Tensor",
        mask_j: "torch.Tensor",
    ) -> "torch.Tensor":
        B, Na, D = h_agv.shape
        if Na == 0:
            return torch.zeros((B, 0), dtype=h_agv.dtype, device=h_agv.device)
        a_in = torch.cat(
            [
                h_task_j.unsqueeze(1).expand(B, Na, D),
                h_agv,
                h_sel_bay_j.unsqueeze(1).expand(B, Na, D),
                g_pool.unsqueeze(1).expand(B, Na, D),
                task_agv_x_j,
                dyn_load,
            ],
            dim=-1,
        )
        lg = self.agv_fc2(F.relu(self.agv_fc1(a_in))).squeeze(-1)
        return lg.masked_fill(~mask_j.bool(), -1e9)

    def forward_teacher_forced(
        self,
        tensors: dict[str, "torch.Tensor"],
        target_bays: "torch.Tensor",
        target_agvs: "torch.Tensor",
        task_focus: Optional["torch.Tensor"] = None,
    ) -> dict[str, "torch.Tensor"]:
        enc = self.encode(tensors)
        job_logits, bay_logits, crew_logits = self.forward_static_heads(enc, tensors)
        h_task, h_bay, h_agv, gp = enc["h_task"], enc["h_bay"], enc["h_agv"], enc["g_pool"]
        B, Nt, D = h_task.shape
        Na, Nb, No = h_agv.shape[1], h_bay.shape[1], enc["h_op"].shape[1]
        dev = h_task.device

        if Nt == 0 or Na == 0:
            agv_logits = torch.zeros((B, Nt, Na), dtype=h_task.dtype, device=dev)
        else:
            t_valid = tensors.get("task_valid")
            if task_focus is not None:
                sort_key = task_focus.clone()
                if t_valid is not None:
                    sort_key = sort_key.masked_fill(~t_valid.bool(), -1e9)
                order_idx = torch.argsort(sort_key, dim=1, descending=True)
            else:
                order_idx = torch.arange(Nt, device=dev).unsqueeze(0).expand(B, Nt)

            t2o = tensors["task_to_op"]
            inc_b = tensors["task_inc_bay"]
            agv_logits = torch.zeros((B, Nt, Na), dtype=h_task.dtype, device=dev)
            dyn_load = torch.zeros((B, Na, 2), dtype=h_task.dtype, device=dev)
            b_idx = torch.arange(B, device=dev)

            for step in range(Nt):
                j = order_idx[:, step]
                ht_j = h_task[b_idx, j]
                ta_j = tensors["task_agv_x"][b_idx, j]
                mk_j = tensors["task_agv_mask"][b_idx, j]

                oi_j = t2o[b_idx, j]
                has_op = (oi_j >= 0) & (oi_j < max(1, No))
                safe_oi = torch.clamp(oi_j, 0, max(0, No - 1))
                tb_j = target_bays[b_idx, safe_oi] if No > 0 else torch.full((B,), -1, dtype=torch.long, device=dev)
                bi_j = torch.where(has_op & (tb_j >= 0), tb_j, inc_b[b_idx, j])
                has_b = (bi_j >= 0) & (bi_j < max(1, Nb))
                if Nb > 0:
                    sb_j = h_bay[b_idx, torch.clamp(bi_j, 0, Nb - 1)] * has_b.unsqueeze(-1).to(h_task.dtype)
                else:
                    sb_j = torch.zeros((B, D), dtype=h_task.dtype, device=dev)

                lg_j = self.forward_agv_step(ht_j, h_agv, ta_j, sb_j, dyn_load, gp, mk_j)
                agv_logits[b_idx, j] = lg_j

                chosen_a = target_agvs[b_idx, j]
                upd_ok = (chosen_a >= 0) & (chosen_a < Na)
                if t_valid is not None:
                    upd_ok = upd_ok & t_valid[b_idx, j].bool()
                if upd_ok.any():
                    safe_a = torch.clamp(chosen_a, 0, Na - 1)
                    one_hot = F.one_hot(safe_a, num_classes=Na).to(h_task.dtype) * upd_ok.unsqueeze(-1).to(h_task.dtype)
                    tr_dur = ta_j[b_idx, safe_a, 0].unsqueeze(-1)
                    dyn_load = dyn_load + torch.stack([one_hot * 0.25, one_hot * tr_dur], dim=-1)

        return {
            "job_logits": job_logits,
            "bay_logits": bay_logits,
            "crew_logits": crew_logits,
            "agv_logits": agv_logits,
            "value": enc["value"],
        }


class NumpyFGCS26RenGNNNet:
    """Exact pure-NumPy float32 mirror of `FGCS26RenGNNNet` (`< 1e-5` max abs diff vs PyTorch)."""

    def __init__(self, weights: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
        self.w = {str(k): np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        self.hidden_dim = int(meta.get("hidden_dim", 64))

    def encode(self, g: V19HeteroGraph) -> dict[str, np.ndarray]:
        w = self.w
        D = self.hidden_dim
        No, Nt, Nb, Nc, Na = len(g.op_ids), len(g.rel_tids), len(g.bay_ids), len(g.crew_ids), len(g.agv_ids)

        h_op = _relu_np(_linear_np(g.op_feats, w["op_in.weight"], w["op_in.bias"])) if No > 0 else np.zeros((0, D), dtype=np.float32)
        h_task = _relu_np(_linear_np(g.task_feats, w["task_in.weight"], w["task_in.bias"])) if Nt > 0 else np.zeros((0, D), dtype=np.float32)
        h_bay = _relu_np(_linear_np(g.bay_feats, w["bay_in.weight"], w["bay_in.bias"])) if Nb > 0 else np.zeros((0, D), dtype=np.float32)
        h_crew = _relu_np(_linear_np(g.crew_feats, w["crew_in.weight"], w["crew_in.bias"])) if Nc > 0 else np.zeros((0, D), dtype=np.float32)
        h_agv = _relu_np(_linear_np(g.agv_feats, w["agv_in.weight"], w["agv_in.bias"])) if Na > 0 else np.zeros((0, D), dtype=np.float32)

        h_op, h_task, h_bay, h_crew, h_agv = _np_3d_layer(h_op, h_task, h_bay, h_crew, h_agv, g, w, "layer1")
        h_op, h_task, h_bay, h_crew, h_agv = _np_3d_layer(h_op, h_task, h_bay, h_crew, h_agv, g, w, "layer2")

        gp = (_masked_mean_np(h_op) + _masked_mean_np(h_task) + _masked_mean_np(h_bay) + _masked_mean_np(h_crew) + _masked_mean_np(h_agv)) / np.float32(5.0)
        return {
            "h_op": h_op,
            "h_task": h_task,
            "h_bay": h_bay,
            "h_crew": h_crew,
            "h_agv": h_agv,
            "g_pool": gp.astype(np.float32),
        }

    def predict_static_heads(self, enc: dict[str, np.ndarray], g: V19HeteroGraph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = self.w
        D = self.hidden_dim
        h_op, h_task, h_bay, h_crew, gp = enc["h_op"], enc["h_task"], enc["h_bay"], enc["h_crew"], enc["g_pool"]
        No, Nt, Nb, Nc = len(g.op_ids), len(g.rel_tids), len(g.bay_ids), len(g.crew_ids)

        if Nt > 0:
            job_lg = _linear_np(_relu_np(_linear_np(np.concatenate([h_task, np.broadcast_to(gp, (Nt, D))], axis=-1), w["job_fc1.weight"], w["job_fc1.bias"])), w["job_fc2.weight"], w["job_fc2.bias"]).reshape(Nt)
        else:
            job_lg = np.zeros((0,), dtype=np.float32)

        if No > 0 and Nb > 0:
            b_in = np.concatenate(
                [
                    np.broadcast_to(h_op[:, None, :], (No, Nb, D)),
                    np.broadcast_to(h_bay[None, :, :], (No, Nb, D)),
                    np.broadcast_to(gp[None, None, :], (No, Nb, D)),
                    g.op_bay_feats,
                ],
                axis=-1,
            )
            bay_lg = _linear_np(_relu_np(_linear_np(b_in, w["bay_fc1.weight"], w["bay_fc1.bias"])), w["bay_fc2.weight"], w["bay_fc2.bias"]).reshape(No, Nb)
            bay_lg = np.where(g.op_bay_mask, bay_lg, np.float32(-1e9))
        else:
            bay_lg = np.zeros((No, Nb), dtype=np.float32)

        if No > 0 and Nc > 0:
            c_in = np.concatenate(
                [
                    np.broadcast_to(h_op[:, None, :], (No, Nc, D)),
                    np.broadcast_to(h_crew[None, :, :], (No, Nc, D)),
                    np.broadcast_to(gp[None, None, :], (No, Nc, D)),
                ],
                axis=-1,
            )
            crew_lg = _linear_np(_relu_np(_linear_np(c_in, w["crew_fc1.weight"], w["crew_fc1.bias"])), w["crew_fc2.weight"], w["crew_fc2.bias"]).reshape(No, Nc)
            crew_lg = np.where(g.op_crew_mask, crew_lg, np.float32(-1e9))
        else:
            crew_lg = np.zeros((No, Nc), dtype=np.float32)

        return job_lg.astype(np.float32), bay_lg.astype(np.float32), crew_lg.astype(np.float32)

    def predict_agv_step(
        self,
        enc: dict[str, np.ndarray],
        g: V19HeteroGraph,
        j: int,
        h_sel_bay: np.ndarray,
        dyn_load: np.ndarray,
    ) -> np.ndarray:
        w = self.w
        D = self.hidden_dim
        h_agv = enc["h_agv"]
        Na = h_agv.shape[0]
        if Na == 0:
            return np.zeros((0,), dtype=np.float32)
        a_in = np.concatenate(
            [
                np.broadcast_to(enc["h_task"][j : j + 1], (Na, D)),
                h_agv,
                np.broadcast_to(h_sel_bay[None, :], (Na, D)),
                np.broadcast_to(enc["g_pool"][None, :], (Na, D)),
                g.task_agv_feats[j],
                dyn_load,
            ],
            axis=-1,
        )
        lg = _linear_np(_relu_np(_linear_np(a_in, w["agv_fc1.weight"], w["agv_fc1.bias"])), w["agv_fc2.weight"], w["agv_fc2.bias"]).reshape(Na)
        return np.where(g.task_agv_mask[j], lg, np.float32(-1e9)).astype(np.float32)


def export_fgcs26_ren_npz(model: FGCS26RenGNNNet, out_path: Path | str) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    arrays["__meta_hidden_dim__"] = np.array([model.hidden_dim], dtype=np.int64)
    np.savez(out, **arrays)
    return out


def load_fgcs26_ren_model(
    ckpt_path: Path | str = FGCS26_REN_GNN_CKPT,
    *,
    backend: str = "auto",
    allow_untrained: bool = False,
) -> tuple[Any, str, Optional[str]]:
    p = Path(ckpt_path)
    npz_p = resolve_numpy_checkpoint_path(p)
    use_np = backend == "numpy" or (backend == "auto" and npz_p is not None and npz_p.exists())
    if use_np and npz_p is not None and npz_p.exists() and npz_p.suffix.lower() == ".npz":
        sha = file_sha256(npz_p)
        key = f"np::{npz_p.resolve()}::{sha}"
        if key in _CKPT_CACHE:
            m, cached_sha = _CKPT_CACHE[key]
            return m, "numpy", cached_sha
        raw_w, _, sha = load_numpy_state_dict(npz_p)
        weights = dict(raw_w)
        meta = {"hidden_dim": int(weights.pop("__meta_hidden_dim__", np.array([64]))[0])}
        m = NumpyFGCS26RenGNNNet(weights, meta)
        _CKPT_CACHE[key] = (m, sha)
        return m, "numpy", sha

    if p.exists() and _HAS_TORCH:
        sha = file_sha256(p)
        key = f"pt::{p.resolve()}::{sha}"
        if key in _CKPT_CACHE:
            m, cached_sha = _CKPT_CACHE[key]
            return m, "torch", cached_sha
        payload = torch.load(p, map_location="cpu", weights_only=False)
        cfg = payload.get("config", {})
        net = FGCS26RenGNNNet(hidden_dim=int(cfg.get("hidden_dim", 64)))
        sd = payload.get("state_dict") if "state_dict" in payload else payload.get("model", payload)
        net.load_state_dict(sd, strict=True)
        net.eval()
        _CKPT_CACHE[key] = (net, sha)
        return net, "torch", sha

    if allow_untrained and _HAS_TORCH:
        net = FGCS26RenGNNNet()
        net.eval()
        return net, "torch", None
    raise FileNotFoundError(f"FGCS26 Ren-GNN checkpoint not found at {p} (or .npz).")


@dataclass
class DecodedFGCS26Candidate:
    assign: dict[str, str]
    bays: dict[str, str]
    crews: dict[str, int]
    op_prio: dict[str, int]
    chosen_bay_idx: list[int]
    chosen_crew_idx: list[int]
    chosen_agv_idx: list[int]
    task_order: list[int]


def _sample_cat_np(logits: np.ndarray, mask: np.ndarray, greedy: bool, tau: float, rng: np.random.Generator) -> int:
    valid_idx = np.flatnonzero(mask)
    if len(valid_idx) == 0:
        return 0
    if len(valid_idx) == 1:
        return int(valid_idx[0])
    sub = logits[valid_idx].astype(np.float32)
    if greedy:
        return int(valid_idx[int(np.argmax(sub))])
    probs = _np_softmax(sub / max(0.05, float(tau)))
    return int(valid_idx[int(rng.choice(len(valid_idx), p=probs))])


def decode_fgcs26_candidate(
    model: Any,
    g: V19HeteroGraph,
    snapshot: NativeSnapshot,
    *,
    greedy: bool = True,
    tau: float = 0.45,
    rng: Optional[np.random.Generator] = None,
) -> DecodedFGCS26Candidate:
    use_rng = rng if rng is not None else np.random.default_rng(0)
    No, Nt, Nb, Nc, Na = len(g.op_ids), len(g.rel_tids), len(g.bay_ids), len(g.crew_ids), len(g.agv_ids)

    if isinstance(model, NumpyFGCS26RenGNNNet):
        enc = model.encode(g)
        job_lg, bay_lg, crew_lg = model.predict_static_heads(enc, g)
    else:
        from amr_sched.rl.v19_hetero_constructor import graph_to_torch_tensors

        dev = next(model.parameters()).device
        ten = graph_to_torch_tensors(g, device=dev)
        with torch.no_grad():
            enc_t = model.encode(ten)
            jl_t, bl_t, cl_t = model.forward_static_heads(enc_t, ten)
            enc = {k: v[0].cpu().numpy() for k, v in enc_t.items() if k != "value"}
            job_lg = jl_t[0].cpu().numpy()
            bay_lg = bl_t[0].cpu().numpy()
            crew_lg = cl_t[0].cpu().numpy()

    if Nt == 0:
        order: list[int] = []
    elif greedy:
        order = [int(i) for i in np.argsort(-job_lg, kind="mergesort")]
    else:
        rem = list(range(Nt))
        order = []
        while rem:
            sub = np.array([job_lg[i] for i in rem], dtype=np.float32)
            probs = _np_softmax(sub / max(0.05, float(tau)))
            order.append(rem.pop(int(use_rng.choice(len(rem), p=probs))))

    bays = dict(g.base_bays)
    crews = dict(g.base_crews)
    chosen_bay_idx = [-1] * No
    chosen_crew_idx = [-1] * No
    for oi, oid in enumerate(g.op_ids):
        if Nb > 0:
            bi = _sample_cat_np(bay_lg[oi], g.op_bay_mask[oi], greedy, tau, use_rng)
            chosen_bay_idx[oi] = bi
            bays[oid] = g.bay_ids[bi]
        if Nc > 0:
            ci = _sample_cat_np(crew_lg[oi], g.op_crew_mask[oi], greedy, tau, use_rng)
            chosen_crew_idx[oi] = ci
            crews[oid] = int(g.crew_ids[ci])

    assign = dict(g.base_assign)
    chosen_agv_idx = [-1] * Nt
    dyn_load = np.zeros((Na, 2), dtype=np.float32)
    D = enc["h_bay"].shape[-1] if Nb > 0 else enc["g_pool"].shape[-1]

    for j in order:
        if Na == 0:
            break
        oi = int(g.task_to_op[j])
        bi = chosen_bay_idx[oi] if (0 <= oi < No and chosen_bay_idx[oi] >= 0) else int(g.task_incumbent_bay_idx[j])
        h_sel_bay = enc["h_bay"][bi] if (0 <= bi < Nb) else np.zeros((D,), dtype=np.float32)
        if isinstance(model, NumpyFGCS26RenGNNNet):
            agv_lg = model.predict_agv_step(enc, g, j, h_sel_bay, dyn_load)
        else:
            dev = next(model.parameters()).device
            with torch.no_grad():
                agv_lg = (
                    model.forward_agv_step(
                        torch.from_numpy(enc["h_task"][j : j + 1]).to(dev),
                        torch.from_numpy(enc["h_agv"][None, :, :]).to(dev),
                        torch.from_numpy(g.task_agv_feats[j : j + 1]).to(dev),
                        torch.from_numpy(h_sel_bay[None, :]).to(dev),
                        torch.from_numpy(dyn_load[None, :, :]).to(dev),
                        torch.from_numpy(enc["g_pool"][None, :]).to(dev),
                        torch.from_numpy(g.task_agv_mask[j : j + 1]).to(dev),
                    )[0]
                    .cpu()
                    .numpy()
                )
        ai = _sample_cat_np(agv_lg, g.task_agv_mask[j], greedy, tau, use_rng)
        chosen_agv_idx[j] = ai
        assign[g.rel_tids[j]] = g.agv_ids[ai]
        dyn_load[ai, 0] += 0.25
        dyn_load[ai, 1] += float(g.task_agv_feats[j, ai, 0])

    op_prio = {op.op_id: max(int(snapshot.tick), int(op.due_step) - int(op.proc_steps)) for op in snapshot.operations}
    return DecodedFGCS26Candidate(
        assign=assign,
        bays=bays,
        crews=crews,
        op_prio=op_prio,
        chosen_bay_idx=chosen_bay_idx,
        chosen_crew_idx=chosen_crew_idx,
        chosen_agv_idx=chosen_agv_idx,
        task_order=order,
    )


class FGCS26RenGNNProposer:
    """Online proposer wrapper for Ren et al. (FGCS 2026) 3D-DG GNN + DRL."""

    def __init__(
        self,
        ckpt_path: Path | str = FGCS26_REN_GNN_CKPT,
        *,
        policy_id: str = "FGCS26_REN_GNN",
        backend: str = "auto",
        allow_untrained: bool = False,
        use_rust: bool = True,
        seed: int = 0,
        sample_candidates: int = 3,
        alpha_makespan: float = 3.0,
        lambda_change: float = 15.0,
        hysteresis_margin: float = 10.0,
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.policy_id = str(policy_id)
        self.backend = str(backend)
        self.allow_untrained = bool(allow_untrained)
        self.use_rust = bool(use_rust)
        self.seed = int(seed)
        self.sample_candidates = max(1, int(sample_candidates))
        self.alpha_makespan = float(alpha_makespan)
        self.lambda_change = float(lambda_change)
        self.hysteresis_margin = float(hysteresis_margin)
        self._model: Any = None
        self._eff_backend: Optional[str] = None
        self._checkpoint_sha256: Optional[str] = None

    def _get_net(self) -> Any:
        if self._model is None:
            m, b, sha = load_fgcs26_ren_model(
                self.ckpt_path, backend=self.backend, allow_untrained=self.allow_untrained
            )
            self._model = m
            self._eff_backend = b
            self._checkpoint_sha256 = sha
        return self._model

    def __call__(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        budget_ms: float = 35.0,
    ) -> CandidateProposal:
        model = self._get_net()
        t0 = _cvcr_now_s()
        deadline = t0 + max(0.0, float(budget_ms) * 0.78 / 1000.0)
        clf = classify_commitments(snapshot)
        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)

        ev = CVCREvaluator(
            network,
            snapshot,
            alpha_makespan=self.alpha_makespan,
            lambda_change=self.lambda_change,
            use_rust=self.use_rust,
            mode="B2",
            seq_diff_b2=True,
        )
        base_op_prio = {
            op.op_id: max(int(snapshot.tick), int(op.due_step) - int(op.proc_steps))
            for op in snapshot.operations
        }
        keep_eval = ev.evaluate(
            graph.base_assign,
            op_priority=base_op_prio,
            op_bay_overrides=graph.base_bays,
            op_crew_overrides=graph.base_crews,
        )
        if keep_eval.valid:
            ev.set_incumbent(
                graph.base_assign,
                keep_eval,
                op_priority=base_op_prio,
                op_bay_overrides=graph.base_bays,
            )

        best_eval: EvalResult = keep_eval
        best_score = float(keep_eval.objective)
        best_assign = dict(graph.base_assign)
        best_bays = dict(graph.base_bays)
        best_crews = dict(graph.base_crews)
        best_op_prio = dict(base_op_prio)
        cands_eval = 0

        if len(graph.rel_tids) > 0 and len(graph.agv_ids) >= 1:
            rng = np.random.default_rng(self.seed + int(snapshot.tick))
            seen = {
                (
                    tuple(sorted(graph.base_assign.items())),
                    tuple(sorted(graph.base_bays.items())),
                    tuple(sorted(graph.base_crews.items())),
                )
            }
            for k in range(self.sample_candidates):
                if k > 0 and _cvcr_now_s() >= deadline:
                    break
                cand = decode_fgcs26_candidate(model, graph, snapshot, greedy=(k == 0), tau=0.45, rng=rng)
                sig = (
                    tuple(sorted(cand.assign.items())),
                    tuple(sorted(cand.bays.items())),
                    tuple(sorted(cand.crews.items())),
                )
                if sig in seen:
                    continue
                seen.add(sig)
                ev_c = ev.evaluate(
                    cand.assign,
                    op_priority=cand.op_prio,
                    op_bay_overrides=cand.bays,
                    op_crew_overrides=cand.crews,
                )
                cands_eval += 1
                changed_ops = sum(
                    1
                    for oid in graph.op_ids
                    if cand.bays.get(oid) != graph.base_bays.get(oid)
                    or cand.crews.get(oid) != graph.base_crews.get(oid)
                )
                score = float(ev_c.objective) + self.lambda_change * float(changed_ops)
                margin = self.hysteresis_margin if best_eval is keep_eval else 1e-4
                if ev_c.valid and (not best_eval.valid or score + margin < best_score):
                    best_eval = ev_c
                    best_score = score
                    best_assign = cand.assign
                    best_bays = cand.bays
                    best_crews = cand.crews
                    best_op_prio = cand.op_prio

        prop = _build_external_proposal(
            policy_id=self.policy_id,
            snapshot=snapshot,
            clf=clf,
            best_assign=best_assign,
            best_bays=best_bays,
            best_crews=best_crews,
            best_eval=best_eval,
            op_prio=best_op_prio,
            eval_cnt=ev.calls,
            t0=t0,
            budget_ms=budget_ms,
        )
        prop.telemetry["neural_candidates_evaluated"] = cands_eval
        prop.telemetry["reproduction_tier"] = "architecture_faithful_domain_adapted_reimplementation"
        return prop
