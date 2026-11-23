"""V19 Heterogeneous-Graph Critical-Chain Joint Constructor (PyTorch + Pure-NumPy Parity).

Decoupled neural joint constructor operating strictly on `V19HeteroGraph` snapshots
and guarded by the Transaction Safety Layer (`clf.frozen_tasks` lock, Rust `CVCREvaluator` B2
truth verification, and atomic `NativeTransactionOwner` fallback to incumbent `KEEP`).

Supports:
  - Lazy `torch` imports so torch-free WSL2 ROS 2 + Gazebo bridges load `.npz` natively.
  - Exact PyTorch <-> Pure-NumPy eval parity (`< 1e-5` max abs diff) using deterministic
    `np.random.default_rng` rollouts.
  - Graded action stages: `"A123"` (focus + bay/crew + AGV) and `"A1234"` (+ queue order).
  - Causal mechanism ablations: `"full"`, `"staged"`, and `"no_prod_comm"`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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
        """Minimal stand-in raising ModuleNotFoundError if PyTorch models are constructed without torch."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError("PyTorch (`torch`) is required to instantiate V19HeteroConstructorNet.")

    class _MissingNNNamespace:
        Module = _MissingNNModule

    nn = _MissingNNNamespace  # type: ignore[assignment]

from amr_sched.baselines.numpy_backends import (
    elu as _np_elu,
    file_sha256,
    layer_norm as _np_layer_norm,
    load_numpy_state_dict,
    mha_batch as _np_mha_batch,
    resolve_numpy_checkpoint_path,
    softmax as _np_softmax,
)
from amr_sched.baselines.top_venue_baselines import _build_external_proposal
from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.cvcr.snapshot import NativeSnapshot, op_predecessors
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
    apply_graph_ablation,
    extract_v19_hetero_graph,
)

ACTION_STAGES = ("A123", "A1234")
ABLATION_MODES = (
    "full",
    "staged",
    "no_prod_comm",
    "no_op_dag",
    "no_kit_features",
    "no_commit_features",
    "no_cross_edge_msg",
    "no_a3_bay_cond",
)
_CKPT_CACHE: dict[str, tuple[Any, str, dict[str, Any]]] = {}


def _require_torch() -> None:
    """Raise ModuleNotFoundError when PyTorch-only code is invoked without torch."""
    if not _HAS_TORCH:
        raise ModuleNotFoundError("PyTorch (`torch`) is required for this operation.")


def _mha_masked_torch(
    mod: "nn.MultiheadAttention",
    q_in: "torch.Tensor",
    kv_in: "torch.Tensor",
    kv_valid_mask: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Run `nn.MultiheadAttention` safely for empty or variable-length sequences."""
    if q_in.shape[1] == 0 or kv_in.shape[1] == 0:
        return torch.zeros_like(q_in)
    if kv_valid_mask is None:
        out, _ = mod(q_in, kv_in, kv_in, need_weights=False)
        return out
    pad_mask = ~kv_valid_mask.bool()
    out, _ = mod(q_in, kv_in, kv_in, key_padding_mask=pad_mask, need_weights=False)
    return torch.nan_to_num(out, nan=0.0)


def _masked_mean_torch(x: "torch.Tensor", valid_mask: Optional["torch.Tensor"]) -> "torch.Tensor":
    """Return `[B, H]` mean over entity axis respecting optional validity mask."""
    if x.shape[1] == 0:
        return torch.zeros((x.shape[0], x.shape[2]), dtype=x.dtype, device=x.device)
    if valid_mask is None:
        return x.mean(dim=1)
    m = valid_mask.unsqueeze(-1).to(x.dtype)
    denom = torch.clamp(m.sum(dim=1), min=1.0)
    return (x * m).sum(dim=1) / denom


def _masked_mean_np(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros((x.shape[0], x.shape[2]), dtype=np.float32)
    return x.mean(axis=1).astype(np.float32, copy=False)


class V19HeteroConstructorNet(nn.Module):
    """Relational Heterogeneous-Graph Critical-Chain Joint Constructor + Value Network."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        action_stage: str = "A123",
        ablation_mode: str = "full",
    ) -> None:
        _require_torch()
        super().__init__()
        if action_stage not in ACTION_STAGES:
            raise ValueError(f"Unsupported action_stage {action_stage!r}")
        if ablation_mode not in ABLATION_MODES:
            raise ValueError(f"Unsupported ablation_mode {ablation_mode!r}")

        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.action_stage = str(action_stage)
        self.ablation_mode = str(ablation_mode)
        H = self.hidden_dim

        self.op_enc = nn.Linear(OP_FEAT_DIM, H)
        self.task_enc = nn.Linear(TASK_FEAT_DIM, H)
        self.agv_enc = nn.Linear(AGV_FEAT_DIM, H)
        self.bay_enc = nn.Linear(BAY_FEAT_DIM, H)
        self.crew_enc = nn.Linear(CREW_FEAT_DIM, H)

        self.op_dag_msg = nn.ModuleList([nn.Linear(H * 3, H) for _ in range(self.n_layers)])
        self.kit_op_to_task = nn.ModuleList([nn.Linear(H, H) for _ in range(self.n_layers)])
        self.kit_task_to_op = nn.ModuleList([nn.Linear(H, H) for _ in range(self.n_layers)])

        self.op_self_mha = nn.ModuleList([nn.MultiheadAttention(H, n_heads, batch_first=True) for _ in range(self.n_layers)])
        self.op_norm1 = nn.ModuleList([nn.LayerNorm(H) for _ in range(self.n_layers)])
        self.op_bay_mha = nn.ModuleList([nn.MultiheadAttention(H, n_heads, batch_first=True) for _ in range(self.n_layers)])
        self.op_crew_mha = nn.ModuleList([nn.MultiheadAttention(H, n_heads, batch_first=True) for _ in range(self.n_layers)])
        self.op_norm2 = nn.ModuleList([nn.LayerNorm(H) for _ in range(self.n_layers)])

        self.bay_op_mha = nn.ModuleList([nn.MultiheadAttention(H, n_heads, batch_first=True) for _ in range(self.n_layers)])
        self.bay_norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(self.n_layers)])
        self.crew_op_mha = nn.ModuleList([nn.MultiheadAttention(H, n_heads, batch_first=True) for _ in range(self.n_layers)])
        self.crew_norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(self.n_layers)])

        self.task_self_mha = nn.ModuleList([nn.MultiheadAttention(H, n_heads, batch_first=True) for _ in range(self.n_layers)])
        self.task_norm1 = nn.ModuleList([nn.LayerNorm(H) for _ in range(self.n_layers)])
        self.task_agv_mha = nn.ModuleList([nn.MultiheadAttention(H, n_heads, batch_first=True) for _ in range(self.n_layers)])
        self.task_norm2 = nn.ModuleList([nn.LayerNorm(H) for _ in range(self.n_layers)])

        self.agv_task_mha = nn.ModuleList([nn.MultiheadAttention(H, n_heads, batch_first=True) for _ in range(self.n_layers)])
        self.agv_norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(self.n_layers)])

        # A1: Critical-chain focus priority over releasable tasks
        self.a1_w1 = nn.Linear(H * 4, H // 2)
        self.a1_w2 = nn.Linear(H // 2, 1)

        # A2: Bay & Crew heads over (Op, Bay/Crew, graph context)
        self.bay_w1 = nn.Linear(H * 5 + OP_BAY_EDGE_DIM, H)
        self.bay_w2 = nn.Linear(H, H // 2)
        self.bay_w3 = nn.Linear(H // 2, 1)

        self.crew_w1 = nn.Linear(H * 5, H // 2)
        self.crew_w2 = nn.Linear(H // 2, 1)

        # A3: Sequential AMR assignment conditioned on selected A2 bays + dynamic load
        self.sel_bay_proj = nn.Linear(H, H)
        self.agv_dyn_proj = nn.Linear(2, H)
        self.agv_w1 = nn.Linear(H * 5 + TASK_AGV_EDGE_DIM + 2, H)
        self.agv_w2 = nn.Linear(H, H // 2)
        self.agv_w3 = nn.Linear(H // 2, 1)
        self.incumbent_keep_bias = nn.Parameter(torch.zeros(1))

        # A4: Queue insertion priority score head
        self.seq_w1 = nn.Linear(H * 5, H // 2)
        self.seq_w2 = nn.Linear(H // 2, 1)

        # Shared Value critic head for closed-loop SMDP PPO
        self.val_w1 = nn.Linear(H * 3, H)
        self.val_w2 = nn.Linear(H, 1)

    def config_dict(self) -> dict[str, Any]:
        """Return serializable architecture config."""
        return {
            "hidden_dim": self.hidden_dim,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "action_stage": self.action_stage,
            "ablation_mode": self.ablation_mode,
        }

    def encode_entities(
        self,
        op_x: "torch.Tensor",
        task_x: "torch.Tensor",
        agv_x: "torch.Tensor",
        bay_x: "torch.Tensor",
        crew_x: "torch.Tensor",
        task_to_op: "torch.Tensor",
        op_dag_adj: "torch.Tensor",
        *,
        op_valid: Optional["torch.Tensor"] = None,
        task_valid: Optional["torch.Tensor"] = None,
        agv_valid: Optional["torch.Tensor"] = None,
        bay_valid: Optional["torch.Tensor"] = None,
        crew_valid: Optional["torch.Tensor"] = None,
    ) -> dict[str, "torch.Tensor"]:
        """Run `L` heterogeneous message-passing layers and return contextual node & graph embeddings."""
        h_op = self.op_enc(op_x)
        h_task = self.task_enc(task_x)
        h_agv = self.agv_enc(agv_x)
        h_bay = self.bay_enc(bay_x)
        h_crew = self.crew_enc(crew_x)

        B, N_t = task_x.shape[:2]
        N_op = op_x.shape[1]
        is_joint = self.ablation_mode != "staged"
        has_dag_msg = self.ablation_mode not in ("staged", "no_op_dag")
        has_cross_msg = self.ablation_mode not in ("staged", "no_cross_edge_msg")

        for l_idx in range(self.n_layers):
            if has_dag_msg and N_op > 0:
                deg_pred = torch.clamp(op_dag_adj[:, :, :, 0].sum(dim=-1, keepdim=True), min=1.0)
                deg_succ = torch.clamp(op_dag_adj[:, :, :, 2].sum(dim=-1, keepdim=True), min=1.0)
                m_pred = torch.bmm(op_dag_adj[:, :, :, 0], h_op) / deg_pred
                m_succ = torch.bmm(op_dag_adj[:, :, :, 2], h_op) / deg_succ
                h_op = h_op + F.elu(self.op_dag_msg[l_idx](torch.cat([h_op, m_pred, m_succ], dim=-1)))

            o_self = _mha_masked_torch(self.op_self_mha[l_idx], h_op, h_op, op_valid)
            h_op = self.op_norm1[l_idx](h_op + o_self)

            o_bay = _mha_masked_torch(self.op_bay_mha[l_idx], h_op, h_bay, bay_valid)
            o_crew = _mha_masked_torch(self.op_crew_mha[l_idx], h_op, h_crew, crew_valid)
            if has_cross_msg and N_op > 0 and N_t > 0:
                # Aggregate kit tasks belonging to each operation
                valid_link = (task_to_op >= 0) & (task_to_op < N_op)
                safe_idx = torch.clamp(task_to_op, min=0, max=max(0, N_op - 1))
                one_hot = F.one_hot(safe_idx, num_classes=N_op).to(h_task.dtype)
                one_hot = one_hot * valid_link.unsqueeze(-1).to(h_task.dtype)
                raw_cnt = one_hot.sum(dim=1, keepdim=True).transpose(1, 2)
                has_rel = (raw_cnt > 0.5).to(h_task.dtype)
                kit_pool = torch.bmm(one_hot.transpose(1, 2), h_task) / torch.clamp(raw_cnt, min=1.0)
                kit_msg = F.elu(self.kit_task_to_op[l_idx](kit_pool)) * has_rel
                h_op = self.op_norm2[l_idx](h_op + o_bay + o_crew + kit_msg)
            else:
                h_op = self.op_norm2[l_idx](h_op + o_bay + o_crew)

            h_bay = self.bay_norm[l_idx](h_bay + _mha_masked_torch(self.bay_op_mha[l_idx], h_bay, h_op, op_valid))
            h_crew = self.crew_norm[l_idx](h_crew + _mha_masked_torch(self.crew_op_mha[l_idx], h_crew, h_op, op_valid))

            if has_cross_msg and N_op > 0 and N_t > 0:
                valid_link = (task_to_op >= 0) & (task_to_op < N_op)
                safe_idx = torch.clamp(task_to_op, min=0, max=max(0, N_op - 1))
                gathered_op = h_op.gather(1, safe_idx.unsqueeze(-1).expand(B, N_t, self.hidden_dim))
                op_msg = F.elu(self.kit_op_to_task[l_idx](gathered_op)) * valid_link.unsqueeze(-1).to(h_task.dtype)
                h_task = h_task + op_msg

            t_self = _mha_masked_torch(self.task_self_mha[l_idx], h_task, h_task, task_valid)
            h_task = self.task_norm1[l_idx](h_task + t_self)
            t_agv = _mha_masked_torch(self.task_agv_mha[l_idx], h_task, h_agv, agv_valid)
            h_task = self.task_norm2[l_idx](h_task + t_agv)
            h_agv = self.agv_norm[l_idx](h_agv + _mha_masked_torch(self.agv_task_mha[l_idx], h_agv, h_task, task_valid))

        m_op = _masked_mean_torch(h_op, op_valid)
        m_task = _masked_mean_torch(h_task, task_valid)
        m_agv = _masked_mean_torch(h_agv, agv_valid)
        m_bay = _masked_mean_torch(h_bay, bay_valid)
        m_crew = _masked_mean_torch(h_crew, crew_valid)

        h_g = torch.cat([m_op, m_task, m_agv], dim=-1)
        if is_joint:
            h_g_op = h_g
            h_g_tr = h_g
        else:
            # Strict disjoint global pools for staged ablation
            h_g_op = torch.cat([m_op, m_bay, m_crew], dim=-1)
            h_g_tr = torch.cat([m_task, m_agv, torch.zeros_like(m_task)], dim=-1)

        return {
            "h_op": h_op,
            "h_task": h_task,
            "h_agv": h_agv,
            "h_bay": h_bay,
            "h_crew": h_crew,
            "h_g": h_g,
            "h_g_op": h_g_op,
            "h_g_tr": h_g_tr,
        }

    def a1_a2_heads(
        self,
        enc: dict[str, "torch.Tensor"],
        op_bay_x: "torch.Tensor",
        op_bay_mask: "torch.Tensor",
        op_crew_mask: "torch.Tensor",
    ) -> dict[str, "torch.Tensor"]:
        """Compute `task_focus_logits`, `bay_logits`, `crew_logits`, and scalar `value`."""
        h_op = enc["h_op"]
        h_task = enc["h_task"]
        h_bay = enc["h_bay"]
        h_crew = enc["h_crew"]
        h_g = enc["h_g"]
        h_g_op = enc["h_g_op"]
        h_g_tr = enc["h_g_tr"]
        H = self.hidden_dim

        B, N_op, _ = h_op.shape
        N_b = h_bay.shape[1]
        N_c = h_crew.shape[1]
        N_t = h_task.shape[1]

        if N_t > 0:
            focus_in = torch.cat([h_task, h_g_tr.unsqueeze(1).expand(B, N_t, 3 * H)], dim=-1)
            task_focus_logits = self.a1_w2(F.elu(self.a1_w1(focus_in))).squeeze(-1)
        else:
            task_focus_logits = torch.zeros((B, 0), dtype=h_op.dtype, device=h_op.device)

        if N_op > 0 and N_b > 0:
            bay_in = torch.cat(
                [
                    h_op.unsqueeze(2).expand(B, N_op, N_b, H),
                    h_bay.unsqueeze(1).expand(B, N_op, N_b, H),
                    h_g_op.unsqueeze(1).unsqueeze(2).expand(B, N_op, N_b, 3 * H),
                    op_bay_x,
                ],
                dim=-1,
            )
            bay_logits = self.bay_w3(F.elu(self.bay_w2(F.elu(self.bay_w1(bay_in))))).squeeze(-1)
            bay_logits = bay_logits.masked_fill(~op_bay_mask.bool(), -1e9)
        else:
            bay_logits = torch.zeros((B, N_op, N_b), dtype=h_op.dtype, device=h_op.device)

        if N_op > 0 and N_c > 0:
            crew_in = torch.cat(
                [
                    h_op.unsqueeze(2).expand(B, N_op, N_c, H),
                    h_crew.unsqueeze(1).expand(B, N_op, N_c, H),
                    h_g_op.unsqueeze(1).unsqueeze(2).expand(B, N_op, N_c, 3 * H),
                ],
                dim=-1,
            )
            crew_logits = self.crew_w2(F.elu(self.crew_w1(crew_in))).squeeze(-1)
            crew_logits = crew_logits.masked_fill(~op_crew_mask.bool(), -1e9)
        else:
            crew_logits = torch.zeros((B, N_op, N_c), dtype=h_op.dtype, device=h_op.device)

        value = self.val_w2(F.elu(self.val_w1(h_g))).squeeze(-1)
        return {
            "task_focus_logits": task_focus_logits,
            "bay_logits": bay_logits,
            "crew_logits": crew_logits,
            "value": value,
        }

    def a3_step_logits(
        self,
        enc: dict[str, "torch.Tensor"],
        ti: int | "torch.Tensor",
        sel_bay_idx_ti: "torch.Tensor",
        task_agv_x_ti: "torch.Tensor",
        task_agv_mask_ti: "torch.Tensor",
        bay_travel_ti: "torch.Tensor",
        bay_dd_ti: "torch.Tensor",
        agv_added_load: "torch.Tensor",
        agv_added_dur: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute `[B, N_a]` AGV logits for task index `ti` conditioned on selected bay & current queue."""
        h_task = enc["h_task"]
        h_agv = enc["h_agv"]
        h_bay = enc["h_bay"]
        h_g_tr = enc["h_g_tr"]
        B, N_a, H = h_agv.shape

        if isinstance(ti, int):
            ht = h_task[:, ti, :]  # [B, H]
        else:
            ti_idx = ti.view(B, 1, 1).expand(B, 1, H)
            ht = h_task.gather(1, ti_idx).squeeze(1)
        if self.ablation_mode not in ("staged", "no_a3_bay_cond") and h_bay.shape[1] > 0:
            sb = h_bay.gather(1, sel_bay_idx_ti.view(B, 1, 1).expand(B, 1, H)).squeeze(1)
            ht = ht + F.elu(self.sel_bay_proj(sb))

        dyn_vec = torch.stack([agv_added_load / 4.0, agv_added_dur / 4.0], dim=-1)  # [B, N_a, 2]
        ha_dyn = h_agv + F.elu(self.agv_dyn_proj(dyn_vec))

        b_tr_e = bay_travel_ti.view(B, 1).expand(B, N_a)
        b_dd_e = bay_dd_ti.view(B, 1).expand(B, N_a)
        pair_in = torch.cat(
            [
                ht.unsqueeze(1).expand(B, N_a, H),
                ha_dyn,
                h_g_tr.unsqueeze(1).expand(B, N_a, 3 * H),
                task_agv_x_ti,
                b_tr_e.unsqueeze(-1),
                b_dd_e.unsqueeze(-1),
            ],
            dim=-1,
        )
        logits = self.agv_w3(F.elu(self.agv_w2(F.elu(self.agv_w1(pair_in))))).squeeze(-1)
        if self.ablation_mode not in ("no_prod_comm", "no_commit_features"):
            logits = logits + self.incumbent_keep_bias * task_agv_x_ti[:, :, 3]
        return logits.masked_fill(~task_agv_mask_ti.bool(), -1e9)

    def a4_seq_scores(
        self,
        enc: dict[str, "torch.Tensor"],
        chosen_agv_idx: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute `[B, N_t]` queue insertion priority scores conditioned on chosen AGV."""
        h_task = enc["h_task"]
        h_agv = enc["h_agv"]
        h_g_tr = enc["h_g_tr"]
        B, N_t, H = h_task.shape
        if N_t == 0:
            return torch.zeros((B, 0), dtype=h_task.dtype, device=h_task.device)
        ha_c = h_agv.gather(1, chosen_agv_idx.unsqueeze(-1).expand(B, N_t, H))
        seq_in = torch.cat([h_task, ha_c, h_g_tr.unsqueeze(1).expand(B, N_t, 3 * H)], dim=-1)
        return self.seq_w2(F.elu(self.seq_w1(seq_in))).squeeze(-1)

    def evaluate_teacher_actions(
        self,
        graph: V19HeteroGraph,
        target_bays: "torch.Tensor",
        target_crews: "torch.Tensor",
        target_agvs: "torch.Tensor",
        *,
        device: "torch.device",
    ) -> dict[str, "torch.Tensor"]:
        """Run differentiable teacher-forced forward pass on one `V19HeteroGraph` for BC/PPO loss."""
        tensors = graph_to_torch_tensors(graph, device=device)
        enc = self.encode_entities(
            tensors["op_x"],
            tensors["task_x"],
            tensors["agv_x"],
            tensors["bay_x"],
            tensors["crew_x"],
            tensors["task_to_op"],
            tensors["op_dag_adj"],
        )
        h12 = self.a1_a2_heads(enc, tensors["op_bay_x"], tensors["op_bay_mask"], tensors["op_crew_mask"])
        N_t = len(graph.rel_tids)
        N_a = len(graph.agv_ids)
        agv_logits_rows: list["torch.Tensor"] = []
        if N_t > 0:
            with torch.no_grad():
                focus_np = h12["task_focus_logits"][0].detach().cpu().numpy()
                order = sorted(range(N_t), key=lambda i: (-float(focus_np[i]), i))
            added_load = torch.zeros((1, N_a), dtype=torch.float32, device=device)
            added_dur = torch.zeros((1, N_a), dtype=torch.float32, device=device)
            slot_logits: list[Optional["torch.Tensor"]] = [None] * N_t
            for ti in order:
                oi = int(graph.task_to_op[ti])
                sel_bay = target_bays[oi].view(1) if oi >= 0 else tensors["task_inc_bay"][0, ti].view(1)
                bi = int(sel_bay.item())
                b_tr = tensors["bay_travel"][0, ti, bi].view(1)
                b_dd = tensors["bay_dd_norm"][0, bi].view(1)
                lg = self.a3_step_logits(
                    enc,
                    ti,
                    sel_bay,
                    tensors["task_agv_x"][:, ti, :, :],
                    tensors["task_agv_mask"][:, ti, :],
                    b_tr,
                    b_dd,
                    added_load,
                    added_dur,
                )
                slot_logits[ti] = lg[0]
                ai = int(target_agvs[ti].item())
                mask_row = torch.zeros((1, N_a), dtype=torch.float32, device=device)
                mask_row[0, ai] = 1.0
                added_load = added_load + mask_row
                added_dur = added_dur + mask_row * float(b_tr.item() + float(tensors["task_agv_x"][0, ti, ai, 0].item()))
            agv_logits = torch.stack([row for row in slot_logits if row is not None], dim=0).unsqueeze(0)
        else:
            agv_logits = torch.zeros((1, 0, N_a), dtype=torch.float32, device=device)

        seq_scores = self.a4_seq_scores(enc, target_agvs.unsqueeze(0)) if N_t > 0 else torch.zeros((1, 0), device=device)
        return {
            "task_focus_logits": h12["task_focus_logits"],
            "bay_logits": h12["bay_logits"],
            "crew_logits": h12["crew_logits"],
            "agv_logits": agv_logits,
            "seq_scores": seq_scores,
            "value": h12["value"],
        }

    def evaluate_joint_actions_batch(
        self,
        batch: dict[str, "torch.Tensor"],
        *,
        chosen_bays: "torch.Tensor",
        chosen_crews: "torch.Tensor",
        chosen_agvs: "torch.Tensor",
        focus_order: "torch.Tensor",
        chosen_seq: Optional["torch.Tensor"] = None,
        ref_probs: Optional[dict[str, "torch.Tensor"]] = None,
        seq_sigma: float = 0.35,
        return_probs: bool = False,
        include_a1_plackett_luce: bool = False,
        a1_tau: float = 0.65,
        a1_weight: float = 0.35,
    ) -> dict[str, Any]:
        """Evaluate joint `(A1 -> A2 -> A3 [-> A4])` SMDP PPO stats across padded batch."""
        enc = self.encode_entities(
            batch["op_x"],
            batch["task_x"],
            batch["agv_x"],
            batch["bay_x"],
            batch["crew_x"],
            batch["task_to_op"],
            batch["op_dag_adj"],
            op_valid=batch["op_valid"],
            task_valid=batch["task_valid"],
            agv_valid=batch["agv_valid"],
            bay_valid=batch["bay_valid"],
            crew_valid=batch["crew_valid"],
        )
        h12 = self.a1_a2_heads(enc, batch["op_bay_x"], batch["op_bay_mask"], batch["op_crew_mask"])
        B, T_m = chosen_agvs.shape
        A_m = batch["agv_x"].shape[1]
        device = chosen_agvs.device

        def _cat_stats(
            logits: "torch.Tensor",
            mask: "torch.Tensor",
            chosen: "torch.Tensor",
            ref_p: Optional["torch.Tensor"],
        ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            any_ok = mask.any(dim=-1, keepdim=True)
            safe_lg = torch.where(any_ok, logits, torch.zeros_like(logits))
            lp_all = F.log_softmax(safe_lg, dim=-1)
            p_all = lp_all.exp() * mask.float()
            lp_clean = torch.where(mask, lp_all, torch.zeros_like(lp_all))
            ent = -(p_all * lp_clean).sum(dim=-1)
            c_lp = lp_clean.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
            if ref_p is not None:
                kl_e = p_all * (lp_clean - torch.log(torch.clamp(ref_p, min=1e-8)))
                kl = torch.where(mask, kl_e, torch.zeros_like(kl_e)).sum(dim=-1)
            else:
                kl = torch.zeros_like(ent)
            return c_lp, ent, kl, p_all

        bay_mask = batch["op_bay_mask"]
        crew_mask = batch["op_crew_mask"]
        bay_active = batch["op_valid"] & (bay_mask.sum(dim=-1) > 1)
        crew_active = batch["op_valid"] & (crew_mask.sum(dim=-1) > 1)

        lp_b, ent_b, kl_b, prob_b = _cat_stats(
            h12["bay_logits"], bay_mask, chosen_bays, ref_probs["bay_probs"] if ref_probs is not None else None
        )
        lp_c, ent_c, kl_c, prob_c = _cat_stats(
            h12["crew_logits"], crew_mask, chosen_crews, ref_probs["crew_probs"] if ref_probs is not None else None
        )

        sum_lp = (lp_b * bay_active.float()).sum(dim=1) + (lp_c * crew_active.float()).sum(dim=1)
        sum_ent = (ent_b * bay_active.float()).sum(dim=1) + (ent_c * crew_active.float()).sum(dim=1)
        sum_kl = (kl_b * bay_active.float()).sum(dim=1) + (kl_c * crew_active.float()).sum(dim=1)
        n_cat = bay_active.float().sum(dim=1) + crew_active.float().sum(dim=1)

        # Optional A1 Plackett-Luce listwise focus permutation log-likelihood & KL
        if include_a1_plackett_luce and T_m > 1:
            rem_mask = batch["task_valid"].clone()
            scaled_focus = h12["task_focus_logits"] / max(0.05, float(a1_tau))
            ref_focus_scaled = (
                ref_probs["focus_logits"] / max(0.05, float(a1_tau))
                if (ref_probs is not None and "focus_logits" in ref_probs)
                else None
            )
            w_a1 = float(a1_weight)
            for step_k in range(T_m - 1):
                active_f = (step_k + 1) < batch["n_tasks"]
                if not active_f.any():
                    break
                ti_b = focus_order[:, step_k].clamp(min=0, max=max(0, T_m - 1))
                step_lg = scaled_focus.masked_fill(~rem_mask, -1e9)
                ref_p_step: Optional["torch.Tensor"] = None
                if ref_focus_scaled is not None:
                    ref_lg = ref_focus_scaled.masked_fill(~rem_mask, -1e9)
                    ref_p_step = F.softmax(ref_lg, dim=-1) * rem_mask.float()
                lp_f, ent_f, kl_f, _ = _cat_stats(step_lg, rem_mask, ti_b, ref_p_step)
                vf = (active_f & (rem_mask.sum(dim=-1) > 1)).float() * w_a1
                sum_lp = sum_lp + lp_f * vf
                sum_ent = sum_ent + ent_f * vf
                sum_kl = sum_kl + kl_f * vf
                n_cat = n_cat + vf
                rem_mask = rem_mask.scatter(1, ti_b.unsqueeze(1), False)

        step_prob_list: list["torch.Tensor"] = []
        if T_m > 0:
            b_idx = torch.arange(B, device=device)
            added_load = torch.zeros((B, A_m), dtype=torch.float32, device=device)
            added_dur = torch.zeros((B, A_m), dtype=torch.float32, device=device)
            for step_k in range(T_m):
                ti_b = focus_order[:, step_k]
                active_b = step_k < batch["n_tasks"]
                op_idx_b = batch["task_to_op"][b_idx, ti_b]
                has_op = op_idx_b >= 0
                sel_bay_b = torch.where(
                    has_op,
                    chosen_bays[b_idx, op_idx_b.clamp(min=0)],
                    batch["task_inc_bay"][b_idx, ti_b],
                )
                b_tr_b = batch["bay_travel"][b_idx, ti_b, sel_bay_b]
                b_dd_b = batch["bay_dd_norm"][b_idx, sel_bay_b]
                tx_b = batch["task_agv_x"][b_idx, ti_b]
                tmask_b = batch["task_agv_mask"][b_idx, ti_b]
                lg = self.a3_step_logits(
                    enc, ti_b, sel_bay_b, tx_b, tmask_b, b_tr_b, b_dd_b, added_load, added_dur
                )
                chosen_a = chosen_agvs[b_idx, ti_b]
                ref_ap = ref_probs["agv_probs"][:, step_k, :] if ref_probs is not None else None
                lp_a, ent_a, kl_a, prob_a = _cat_stats(lg, tmask_b, chosen_a, ref_ap)
                if return_probs:
                    step_prob_list.append(prob_a)

                valid_a = (active_b & (tmask_b.sum(dim=-1) > 1)).float()
                sum_lp = sum_lp + lp_a * valid_a
                sum_ent = sum_ent + ent_a * valid_a
                sum_kl = sum_kl + kl_a * valid_a
                n_cat = n_cat + valid_a

                upd_l = torch.zeros_like(added_load)
                upd_l.scatter_(1, chosen_a.unsqueeze(1), active_b.float().unsqueeze(1))
                dur_val = (b_tr_b + tx_b[b_idx, chosen_a, 0]) * active_b.float()
                upd_d = torch.zeros_like(added_dur)
                upd_d.scatter_(1, chosen_a.unsqueeze(1), dur_val.unsqueeze(1))
                added_load = added_load + upd_l
                added_dur = added_dur + upd_d

        agv_probs = torch.stack(step_prob_list, dim=1) if (return_probs and step_prob_list) else torch.zeros((B, 0, A_m), device=device)

        n_tot = n_cat.clone()
        seq_mu = self.a4_seq_scores(enc, chosen_agvs) if (self.action_stage == "A1234" and T_m > 0) else torch.zeros((B, T_m), device=device)
        if self.action_stage == "A1234" and T_m > 0:
            tv_f = batch["task_valid"].float()
            inv_sig2 = 1.0 / max(1e-4, float(seq_sigma) ** 2)
            if chosen_seq is not None:
                lp_s = -0.5 * ((chosen_seq - seq_mu).pow(2)) * inv_sig2
                sum_lp = sum_lp + 0.25 * (lp_s * tv_f).sum(dim=1)
            if ref_probs is not None and "seq_mu" in ref_probs:
                kl_s = 0.5 * ((seq_mu - ref_probs["seq_mu"]).pow(2)) * inv_sig2
                sum_kl = sum_kl + 0.25 * (kl_s * tv_f).sum(dim=1)
            n_tot = n_tot + 0.25 * tv_f.sum(dim=1)

        denom_tot = torch.clamp(n_tot, min=1.0)
        denom_cat = torch.clamp(n_cat, min=1.0)
        out: dict[str, Any] = {
            "mean_logp": sum_lp / denom_tot,
            "mean_entropy": sum_ent / denom_cat,
            "mean_kl": sum_kl / denom_tot,
            "value": h12["value"].view(-1),
            "has_decisions": n_tot > 0.0,
        }
        if return_probs:
            out["probs"] = {
                "bay_probs": prob_b,
                "crew_probs": prob_c,
                "agv_probs": agv_probs,
                "seq_mu": seq_mu,
                "focus_logits": h12["task_focus_logits"],
            }
        return out


def graph_to_torch_tensors(graph: V19HeteroGraph, *, device: Any = "cpu") -> dict[str, Any]:
    """Convert single `V19HeteroGraph` into `[1, ...]` PyTorch tensors."""
    _require_torch()
    return {
        "op_x": torch.from_numpy(graph.op_feats).unsqueeze(0).to(device),
        "task_x": torch.from_numpy(graph.task_feats).unsqueeze(0).to(device),
        "agv_x": torch.from_numpy(graph.agv_feats).unsqueeze(0).to(device),
        "bay_x": torch.from_numpy(graph.bay_feats).unsqueeze(0).to(device),
        "crew_x": torch.from_numpy(graph.crew_feats).unsqueeze(0).to(device),
        "task_to_op": torch.from_numpy(graph.task_to_op).unsqueeze(0).to(device),
        "op_dag_adj": torch.from_numpy(graph.op_dag_adj).unsqueeze(0).to(device),
        "op_bay_x": torch.from_numpy(graph.op_bay_feats).unsqueeze(0).to(device),
        "op_bay_mask": torch.from_numpy(graph.op_bay_mask).unsqueeze(0).to(device),
        "op_crew_mask": torch.from_numpy(graph.op_crew_mask).unsqueeze(0).to(device),
        "task_agv_x": torch.from_numpy(graph.task_agv_feats).unsqueeze(0).to(device),
        "task_agv_mask": torch.from_numpy(graph.task_agv_mask).unsqueeze(0).to(device),
        "task_inc_bay": torch.from_numpy(graph.task_incumbent_bay_idx).unsqueeze(0).to(device),
        "bay_travel": torch.from_numpy(graph.bay_xy_travel_from_pickup).unsqueeze(0).to(device),
        "bay_dd_norm": torch.from_numpy(graph.bay_detour_norm).unsqueeze(0).to(device),
    }


def _np_mha_safe(
    q_in: np.ndarray,
    kv_in: np.ndarray,
    weights: dict[str, np.ndarray],
    prefix: str,
    n_heads: int,
) -> np.ndarray:
    if q_in.shape[1] == 0 or kv_in.shape[1] == 0:
        return np.zeros_like(q_in, dtype=np.float32)
    return _np_mha_batch(
        q_in,
        kv_in,
        weights[f"{prefix}.in_proj_weight"],
        weights[f"{prefix}.in_proj_bias"],
        weights[f"{prefix}.out_proj.weight"],
        weights[f"{prefix}.out_proj.bias"],
        n_heads=n_heads,
    )


def _np_linear(x: np.ndarray, weights: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    return (x @ weights[f"{prefix}.weight"].T + weights[f"{prefix}.bias"]).astype(np.float32, copy=False)


class NumpyV19HeteroConstructorNet:
    """Pure-NumPy forward inference engine matching `V19HeteroConstructorNet.eval()` to `< 1e-5`."""

    def __init__(
        self,
        weights: dict[str, np.ndarray],
        *,
        hidden_dim: Optional[int] = None,
        n_heads: Optional[int] = None,
        n_layers: Optional[int] = None,
        action_stage: Optional[str] = None,
        ablation_mode: Optional[str] = None,
    ) -> None:
        self.w = weights
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else int(weights.get("__cfg_hidden_dim__", np.array([weights["op_enc.weight"].shape[0]]))[0]))
        self.n_heads = int(n_heads if n_heads is not None else int(weights.get("__cfg_n_heads__", np.array([4]))[0]))
        inferred_layers = sum(1 for k in weights if k.startswith("op_self_mha.") and k.endswith(".out_proj.weight"))
        self.n_layers = int(n_layers if n_layers is not None else int(weights.get("__cfg_n_layers__", np.array([inferred_layers]))[0]))

        if action_stage is not None:
            self.action_stage = str(action_stage)
        elif "__cfg_action_stage__" in weights:
            self.action_stage = bytes(int(round(float(x))) for x in weights["__cfg_action_stage__"].reshape(-1)).decode("utf-8")
        else:
            self.action_stage = "A123"

        if ablation_mode is not None:
            self.ablation_mode = str(ablation_mode)
        elif "__cfg_ablation_mode__" in weights:
            self.ablation_mode = bytes(int(round(float(x))) for x in weights["__cfg_ablation_mode__"].reshape(-1)).decode("utf-8")
        else:
            self.ablation_mode = "full"

    def encode_entities(self, graph: V19HeteroGraph) -> dict[str, np.ndarray]:
        """Execute pure-NumPy heterogeneous message passing over a single `V19HeteroGraph`."""
        w = self.w
        H = self.hidden_dim
        is_joint = self.ablation_mode != "staged"
        has_dag_msg = self.ablation_mode not in ("staged", "no_op_dag")
        has_cross_msg = self.ablation_mode not in ("staged", "no_cross_edge_msg")

        op_x = graph.op_feats[None, :, :]
        task_x = graph.task_feats[None, :, :]
        agv_x = graph.agv_feats[None, :, :]
        bay_x = graph.bay_feats[None, :, :]
        crew_x = graph.crew_feats[None, :, :]
        task_to_op = graph.task_to_op
        op_dag_adj = graph.op_dag_adj[None, :, :, :]

        h_op = _np_linear(op_x, w, "op_enc")
        h_task = _np_linear(task_x, w, "task_enc")
        h_agv = _np_linear(agv_x, w, "agv_enc")
        h_bay = _np_linear(bay_x, w, "bay_enc")
        h_crew = _np_linear(crew_x, w, "crew_enc")

        N_op = op_x.shape[1]
        N_t = task_x.shape[1]

        for l_idx in range(self.n_layers):
            if has_dag_msg and N_op > 0:
                deg_p = np.maximum(1.0, op_dag_adj[:, :, :, 0].sum(axis=-1, keepdims=True))
                deg_s = np.maximum(1.0, op_dag_adj[:, :, :, 2].sum(axis=-1, keepdims=True))
                m_p = np.matmul(op_dag_adj[:, :, :, 0], h_op) / deg_p
                m_s = np.matmul(op_dag_adj[:, :, :, 2], h_op) / deg_s
                h_op = h_op + _np_elu(_np_linear(np.concatenate([h_op, m_p, m_s], axis=-1), w, f"op_dag_msg.{l_idx}"))

            o_self = _np_mha_safe(h_op, h_op, w, f"op_self_mha.{l_idx}", self.n_heads)
            h_op = _np_layer_norm(h_op + o_self, w[f"op_norm1.{l_idx}.weight"], w[f"op_norm1.{l_idx}.bias"])

            o_bay = _np_mha_safe(h_op, h_bay, w, f"op_bay_mha.{l_idx}", self.n_heads)
            o_crew = _np_mha_safe(h_op, h_crew, w, f"op_crew_mha.{l_idx}", self.n_heads)
            if has_cross_msg and N_op > 0 and N_t > 0:
                one_hot = np.zeros((N_op, N_t), dtype=np.float32)
                for ti, oi in enumerate(task_to_op.tolist()):
                    if 0 <= oi < N_op:
                        one_hot[oi, ti] = 1.0
                raw_cnt = one_hot.sum(axis=1, keepdims=True)
                has_rel = (raw_cnt > 0.5).astype(np.float32)
                cnt = np.maximum(1.0, raw_cnt)
                kit_pool = (one_hot @ h_task[0]) / cnt
                kit_msg = _np_elu(_np_linear(kit_pool[None, :, :], w, f"kit_task_to_op.{l_idx}")) * has_rel[None, :, :]
                h_op = _np_layer_norm(h_op + o_bay + o_crew + kit_msg, w[f"op_norm2.{l_idx}.weight"], w[f"op_norm2.{l_idx}.bias"])
            else:
                h_op = _np_layer_norm(h_op + o_bay + o_crew, w[f"op_norm2.{l_idx}.weight"], w[f"op_norm2.{l_idx}.bias"])

            h_bay = _np_layer_norm(
                h_bay + _np_mha_safe(h_bay, h_op, w, f"bay_op_mha.{l_idx}", self.n_heads),
                w[f"bay_norm.{l_idx}.weight"],
                w[f"bay_norm.{l_idx}.bias"],
            )
            h_crew = _np_layer_norm(
                h_crew + _np_mha_safe(h_crew, h_op, w, f"crew_op_mha.{l_idx}", self.n_heads),
                w[f"crew_norm.{l_idx}.weight"],
                w[f"crew_norm.{l_idx}.bias"],
            )

            if has_cross_msg and N_op > 0 and N_t > 0:
                gathered = np.zeros((1, N_t, H), dtype=np.float32)
                for ti, oi in enumerate(task_to_op.tolist()):
                    if 0 <= oi < N_op:
                        gathered[0, ti] = h_op[0, oi]
                op_msg = _np_elu(_np_linear(gathered, w, f"kit_op_to_task.{l_idx}"))
                for ti, oi in enumerate(task_to_op.tolist()):
                    if not (0 <= oi < N_op):
                        op_msg[0, ti] = 0.0
                h_task = h_task + op_msg

            t_self = _np_mha_safe(h_task, h_task, w, f"task_self_mha.{l_idx}", self.n_heads)
            h_task = _np_layer_norm(h_task + t_self, w[f"task_norm1.{l_idx}.weight"], w[f"task_norm1.{l_idx}.bias"])
            t_agv = _np_mha_safe(h_task, h_agv, w, f"task_agv_mha.{l_idx}", self.n_heads)
            h_task = _np_layer_norm(h_task + t_agv, w[f"task_norm2.{l_idx}.weight"], w[f"task_norm2.{l_idx}.bias"])
            h_agv = _np_layer_norm(
                h_agv + _np_mha_safe(h_agv, h_task, w, f"agv_task_mha.{l_idx}", self.n_heads),
                w[f"agv_norm.{l_idx}.weight"],
                w[f"agv_norm.{l_idx}.bias"],
            )

        m_op = _masked_mean_np(h_op)
        m_task = _masked_mean_np(h_task)
        m_agv = _masked_mean_np(h_agv)
        m_bay = _masked_mean_np(h_bay)
        m_crew = _masked_mean_np(h_crew)
        h_g = np.concatenate([m_op, m_task, m_agv], axis=-1)
        if is_joint:
            h_g_op = h_g
            h_g_tr = h_g
        else:
            h_g_op = np.concatenate([m_op, m_bay, m_crew], axis=-1)
            h_g_tr = np.concatenate([m_task, m_agv, np.zeros_like(m_task)], axis=-1)
        return {
            "h_op": h_op,
            "h_task": h_task,
            "h_agv": h_agv,
            "h_bay": h_bay,
            "h_crew": h_crew,
            "h_g": h_g,
            "h_g_op": h_g_op,
            "h_g_tr": h_g_tr,
        }

    def a1_a2_heads(self, enc: dict[str, np.ndarray], graph: V19HeteroGraph) -> dict[str, np.ndarray]:
        """Compute `task_focus_logits`, `bay_logits`, `crew_logits`, and `value` in pure NumPy."""
        w = self.w
        H = self.hidden_dim
        h_op, h_task, h_bay, h_crew = enc["h_op"], enc["h_task"], enc["h_bay"], enc["h_crew"]
        h_g, h_g_op, h_g_tr = enc["h_g"], enc["h_g_op"], enc["h_g_tr"]
        N_op, N_t, N_b, N_c = h_op.shape[1], h_task.shape[1], h_bay.shape[1], h_crew.shape[1]

        if N_t > 0:
            hg_t = np.broadcast_to(h_g_tr[:, None, :], (1, N_t, 3 * H))
            focus_in = np.concatenate([h_task, hg_t], axis=-1)
            task_focus = _np_linear(_np_elu(_np_linear(focus_in, w, "a1_w1")), w, "a1_w2")[..., 0]
        else:
            task_focus = np.zeros((1, 0), dtype=np.float32)

        if N_op > 0 and N_b > 0:
            bay_in = np.concatenate(
                [
                    np.broadcast_to(h_op[:, :, None, :], (1, N_op, N_b, H)),
                    np.broadcast_to(h_bay[:, None, :, :], (1, N_op, N_b, H)),
                    np.broadcast_to(h_g_op[:, None, None, :], (1, N_op, N_b, 3 * H)),
                    graph.op_bay_feats[None, :, :, :],
                ],
                axis=-1,
            )
            bl = _np_linear(_np_elu(_np_linear(_np_elu(_np_linear(bay_in, w, "bay_w1")), w, "bay_w2")), w, "bay_w3")[..., 0]
            bay_logits = np.where(graph.op_bay_mask[None, :, :], bl, -1e9).astype(np.float32)
        else:
            bay_logits = np.zeros((1, N_op, N_b), dtype=np.float32)

        if N_op > 0 and N_c > 0:
            crew_in = np.concatenate(
                [
                    np.broadcast_to(h_op[:, :, None, :], (1, N_op, N_c, H)),
                    np.broadcast_to(h_crew[:, None, :, :], (1, N_op, N_c, H)),
                    np.broadcast_to(h_g_op[:, None, None, :], (1, N_op, N_c, 3 * H)),
                ],
                axis=-1,
            )
            cl = _np_linear(_np_elu(_np_linear(crew_in, w, "crew_w1")), w, "crew_w2")[..., 0]
            crew_logits = np.where(graph.op_crew_mask[None, :, :], cl, -1e9).astype(np.float32)
        else:
            crew_logits = np.zeros((1, N_op, N_c), dtype=np.float32)

        value = _np_linear(_np_elu(_np_linear(h_g, w, "val_w1")), w, "val_w2")[..., 0]
        return {
            "task_focus_logits": task_focus,
            "bay_logits": bay_logits,
            "crew_logits": crew_logits,
            "value": value,
        }

    def a3_step_logits(
        self,
        enc: dict[str, np.ndarray],
        graph: V19HeteroGraph,
        ti: int,
        sel_bay_idx: int,
        agv_added_load: np.ndarray,
        agv_added_dur: np.ndarray,
    ) -> np.ndarray:
        """Compute `[1, N_a]` AGV logits for task `ti` in pure NumPy."""
        w = self.w
        H = self.hidden_dim
        h_agv = enc["h_agv"]
        h_bay = enc["h_bay"]
        h_g_tr = enc["h_g_tr"]
        N_a = h_agv.shape[1]

        ht = enc["h_task"][:, ti, :]
        if self.ablation_mode not in ("staged", "no_a3_bay_cond") and h_bay.shape[1] > 0:
            ht = ht + _np_elu(_np_linear(h_bay[:, sel_bay_idx, :], w, "sel_bay_proj"))

        dyn_vec = np.stack([agv_added_load / 4.0, agv_added_dur / 4.0], axis=-1)[None, :, :]
        ha_dyn = h_agv + _np_elu(_np_linear(dyn_vec, w, "agv_dyn_proj"))

        b_tr = float(graph.bay_xy_travel_from_pickup[ti, sel_bay_idx])
        b_dd = float(graph.bay_detour_norm[sel_bay_idx])
        b_tr_arr = np.full((1, N_a, 1), b_tr, dtype=np.float32)
        b_dd_arr = np.full((1, N_a, 1), b_dd, dtype=np.float32)
        ta_edge = graph.task_agv_feats[None, ti, :, :]

        pair_in = np.concatenate(
            [
                np.broadcast_to(ht[:, None, :], (1, N_a, H)),
                ha_dyn,
                np.broadcast_to(h_g_tr[:, None, :], (1, N_a, 3 * H)),
                ta_edge,
                b_tr_arr,
                b_dd_arr,
            ],
            axis=-1,
        )
        logits = _np_linear(_np_elu(_np_linear(_np_elu(_np_linear(pair_in, w, "agv_w1")), w, "agv_w2")), w, "agv_w3")[..., 0]
        if self.ablation_mode not in ("no_prod_comm", "no_commit_features"):
            logits = logits + float(w["incumbent_keep_bias"][0]) * ta_edge[:, :, 3]
        return np.where(graph.task_agv_mask[None, ti, :], logits, -1e9).astype(np.float32)

    def a4_seq_scores(self, enc: dict[str, np.ndarray], chosen_agv_idx: np.ndarray) -> np.ndarray:
        """Compute `[1, N_t]` queue sequencing scores in pure NumPy."""
        w = self.w
        H = self.hidden_dim
        h_task = enc["h_task"]
        N_t = h_task.shape[1]
        if N_t == 0:
            return np.zeros((1, 0), dtype=np.float32)
        ha_c = enc["h_agv"][0, chosen_agv_idx, :][None, :, :]
        seq_in = np.concatenate(
            [h_task, ha_c, np.broadcast_to(enc["h_g_tr"][:, None, :], (1, N_t, 3 * H))],
            axis=-1,
        )
        return _np_linear(_np_elu(_np_linear(seq_in, w, "seq_w1")), w, "seq_w2")[..., 0]


def export_v19_npz(pt_path: Path | str, npz_path: Path | str | None = None) -> Path:
    """Export a `.pt` V19 checkpoint cleanly to a pickle-free float32 `.npz` for WSL2."""
    _require_torch()
    src = Path(pt_path)
    dst = Path(npz_path) if npz_path is not None else src.with_suffix(".npz")
    sd = torch.load(src, map_location="cpu", weights_only=True)
    raw = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    cfg = sd.get("config", {}) if isinstance(sd, dict) else {}
    arrays: dict[str, np.ndarray] = {
        str(k): v.detach().cpu().numpy().astype(np.float32) for k, v in raw.items()
    }
    arrays["__cfg_hidden_dim__"] = np.array([int(cfg.get("hidden_dim", 64))], dtype=np.int32)
    arrays["__cfg_n_heads__"] = np.array([int(cfg.get("n_heads", 4))], dtype=np.int32)
    arrays["__cfg_n_layers__"] = np.array([int(cfg.get("n_layers", 2))], dtype=np.int32)
    arrays["__cfg_action_stage__"] = np.frombuffer(str(cfg.get("action_stage", "A123")).encode("utf-8"), dtype=np.uint8)
    arrays["__cfg_ablation_mode__"] = np.frombuffer(str(cfg.get("ablation_mode", "full")).encode("utf-8"), dtype=np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **arrays)
    return dst


def _load_v19_checkpoint(
    ckpt_path: Path | str,
    *,
    backend: str = "auto",
    allow_untrained: bool = False,
    action_stage: Optional[str] = None,
    ablation_mode: Optional[str] = None,
) -> tuple[Any, str, Optional[str], dict[str, Any]]:
    """Load PyTorch or pure-NumPy V19 net strictly checking checkpoint existence."""
    p = Path(ckpt_path)
    eff_backend = backend
    if eff_backend == "auto":
        if not _HAS_TORCH or p.suffix.lower() == ".npz":
            eff_backend = "numpy"
        else:
            eff_backend = "torch" if p.is_file() else ("numpy" if p.with_suffix(".npz").is_file() else "torch")
    if eff_backend not in ("torch", "numpy"):
        raise ValueError(f"Unsupported backend {backend!r}; expected 'auto', 'torch', or 'numpy'.")

    if eff_backend == "numpy":
        resolved = resolve_numpy_checkpoint_path(p, prefer_npz=True)
        if not resolved.is_file():
            if not allow_untrained:
                raise FileNotFoundError(f"Required V19 checkpoint not found: {p}")
            _require_torch()
            net = V19HeteroConstructorNet(
                action_stage=action_stage or "A123",
                ablation_mode=ablation_mode or "full",
            )
            raw_w = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
            np_net = NumpyV19HeteroConstructorNet(
                raw_w,
                action_stage=action_stage or "A123",
                ablation_mode=ablation_mode or "full",
            )
            return np_net, "numpy", None, net.config_dict()

        sha = file_sha256(resolved)
        ck = f"np:{resolved}:{sha}:{action_stage}:{ablation_mode}"
        if ck not in _CKPT_CACHE:
            weights, _, _ = load_numpy_state_dict(resolved, prefer_npz=True)
            np_net = NumpyV19HeteroConstructorNet(
                weights,
                action_stage=action_stage,
                ablation_mode=ablation_mode,
            )
            cfg = {
                "hidden_dim": np_net.hidden_dim,
                "n_heads": np_net.n_heads,
                "n_layers": np_net.n_layers,
                "action_stage": np_net.action_stage,
                "ablation_mode": np_net.ablation_mode,
            }
            _CKPT_CACHE[ck] = (np_net, sha, cfg)
        np_net, sha, cfg = _CKPT_CACHE[ck]
        return np_net, "numpy", sha, cfg

    _require_torch()
    if not p.is_file():
        if not allow_untrained:
            raise FileNotFoundError(f"Required V19 checkpoint not found: {p}")
        net = V19HeteroConstructorNet(
            action_stage=action_stage or "A123",
            ablation_mode=ablation_mode or "full",
        )
        net.eval()
        return net, "torch", None, net.config_dict()

    sha = file_sha256(p)
    ck = f"pt:{p.resolve()}:{sha}:{action_stage}:{ablation_mode}"
    if ck not in _CKPT_CACHE:
        sd = torch.load(p, map_location="cpu", weights_only=True)
        cfg = dict(sd.get("config", {})) if isinstance(sd, dict) else {}
        if action_stage is not None:
            cfg["action_stage"] = action_stage
        if ablation_mode is not None:
            cfg["ablation_mode"] = ablation_mode
        net = V19HeteroConstructorNet(
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            n_heads=int(cfg.get("n_heads", 4)),
            n_layers=int(cfg.get("n_layers", 2)),
            action_stage=str(cfg.get("action_stage", "A123")),
            ablation_mode=str(cfg.get("ablation_mode", "full")),
        )
        raw = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        net.load_state_dict(raw)
        net.eval()
        _CKPT_CACHE[ck] = (net, sha, net.config_dict())
    net, sha, cfg = _CKPT_CACHE[ck]
    return net, "torch", sha, cfg


@dataclass
class DecodedJointCandidate:
    """Legal joint construction decoded from `V19HeteroConstructorNet` logits."""

    assign: dict[str, str]
    bays: dict[str, str]
    crews: dict[str, int]
    op_prio: dict[str, int]
    task_sequences: Optional[dict[str, list[str]]]
    bay_indices: np.ndarray
    crew_indices: np.ndarray
    agv_indices: np.ndarray
    seq_scores: np.ndarray


def decode_v19_candidate(
    net_or_np: Any,
    graph: V19HeteroGraph,
    snapshot: NativeSnapshot,
    *,
    greedy: bool,
    np_rng: np.random.Generator,
    tau: float = 0.65,
) -> DecodedJointCandidate:
    """Run autoregressive `(A1 -> A2 -> A3 [-> A4])` construction cleanly on Torch or NumPy."""
    N_op = len(graph.op_ids)
    N_t = len(graph.rel_tids)
    N_a = len(graph.agv_ids)
    is_np = isinstance(net_or_np, NumpyV19HeteroConstructorNet)
    eff_stage = getattr(net_or_np, "action_stage", "A123")

    if is_np:
        enc = net_or_np.encode_entities(graph)
        h12 = net_or_np.a1_a2_heads(enc, graph)
        task_focus_np = h12["task_focus_logits"][0]
        bay_logits_np = h12["bay_logits"][0]
        crew_logits_np = h12["crew_logits"][0]
        tensors = None
    else:
        _require_torch()
        tensors = graph_to_torch_tensors(graph, device="cpu")
        with torch.inference_mode():
            enc = net_or_np.encode_entities(
                tensors["op_x"],
                tensors["task_x"],
                tensors["agv_x"],
                tensors["bay_x"],
                tensors["crew_x"],
                tensors["task_to_op"],
                tensors["op_dag_adj"],
            )
            h12 = net_or_np.a1_a2_heads(enc, tensors["op_bay_x"], tensors["op_bay_mask"], tensors["op_crew_mask"])
        task_focus_np = h12["task_focus_logits"][0].cpu().numpy()
        bay_logits_np = h12["bay_logits"][0].cpu().numpy()
        crew_logits_np = h12["crew_logits"][0].cpu().numpy()

    def _pick(logits_1d: np.ndarray) -> int:
        if greedy:
            return int(np.argmax(logits_1d))
        scaled = np.asarray(logits_1d, dtype=np.float64) / max(0.05, float(tau))
        probs = _np_softmax(scaled, axis=-1).astype(np.float64)
        probs = probs / max(1e-12, float(probs.sum()))
        return int(np_rng.choice(len(logits_1d), p=probs))

    # A2: Bays & Crews (respecting op_bay_mask & op_crew_mask)
    bay_indices = np.zeros((N_op,), dtype=np.int64)
    crew_indices = np.zeros((N_op,), dtype=np.int64)
    bays_dict: dict[str, str] = {}
    crews_dict: dict[str, int] = {}
    for oi, oid in enumerate(graph.op_ids):
        if oid in graph.frozen_ops:
            bi = int(graph.op_incumbent_bay_idx[oi])
        else:
            bi = _pick(bay_logits_np[oi])
        bay_indices[oi] = bi
        bays_dict[oid] = graph.bay_ids[bi]

        if graph.op_crew_mask[oi].sum() <= 1:
            ci = int(graph.op_incumbent_crew_idx[oi])
        else:
            ci = _pick(crew_logits_np[oi])
        crew_indices[oi] = ci
        crews_dict[oid] = int(graph.crew_ids[ci])

    # A1 ordering + A3 sequential AMR pointer
    assign_dict = dict(graph.fixed_assign)
    agv_indices = np.zeros((N_t,), dtype=np.int64)
    if N_t > 0:
        order = sorted(range(N_t), key=lambda i: (-float(task_focus_np[i]), i))
        added_load = np.zeros((N_a,), dtype=np.float32)
        added_dur = np.zeros((N_a,), dtype=np.float32)
        if not is_np and tensors is not None:
            added_load_t = torch.zeros((1, N_a), dtype=torch.float32)
            added_dur_t = torch.zeros((1, N_a), dtype=torch.float32)
        for ti in order:
            oi = int(graph.task_to_op[ti])
            sel_bi = int(bay_indices[oi]) if oi >= 0 else int(graph.task_incumbent_bay_idx[ti])
            b_tr = float(graph.bay_xy_travel_from_pickup[ti, sel_bi])
            if is_np:
                lg = net_or_np.a3_step_logits(enc, graph, ti, sel_bi, added_load, added_dur)[0]
            else:
                with torch.inference_mode():
                    lg_t = net_or_np.a3_step_logits(
                        enc,
                        ti,
                        torch.tensor([sel_bi], dtype=torch.long),
                        tensors["task_agv_x"][:, ti, :, :],
                        tensors["task_agv_mask"][:, ti, :],
                        torch.tensor([b_tr], dtype=torch.float32),
                        torch.tensor([float(graph.bay_detour_norm[sel_bi])], dtype=torch.float32),
                        added_load_t,
                        added_dur_t,
                    )
                lg = lg_t[0].cpu().numpy()
            ai = _pick(lg)
            agv_indices[ti] = ai
            assign_dict[graph.rel_tids[ti]] = graph.agv_ids[ai]
            added_load[ai] += 1.0
            added_dur[ai] += b_tr + float(graph.task_agv_feats[ti, ai, 0])
            if not is_np:
                added_load_t[0, ai] = float(added_load[ai])
                added_dur_t[0, ai] = float(added_dur[ai])

    # A4 sequencing (if active) + consistent op_priority
    if N_t > 0 and eff_stage == "A1234":
        if is_np:
            seq_scores_np = net_or_np.a4_seq_scores(enc, agv_indices)[0]
        else:
            with torch.inference_mode():
                seq_scores_np = net_or_np.a4_seq_scores(
                    enc, torch.from_numpy(agv_indices).unsqueeze(0)
                )[0].cpu().numpy()
        by_agv_rel: dict[str, list[tuple[float, str]]] = {}
        for ti, tid in enumerate(graph.rel_tids):
            aid = assign_dict[tid]
            by_agv_rel.setdefault(aid, []).append((-float(seq_scores_np[ti]), tid))
        task_sequences: Optional[dict[str, list[str]]] = {}
        # Keep physically frozen / active non-disturbed cur_task at head unless preempted by urgent task
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
                        and not (ordered and snap_by_id.get(ordered[0]) is not None and snap_by_id[ordered[0]].is_urgent and not cur_t.is_urgent)
                    )
                    if is_frozen_head and cur_t.task_id not in ordered:
                        ordered = [cur_t.task_id] + ordered
                    elif is_active_head and ordered[0] != rob.cur_task:
                        ordered = [rob.cur_task] + [tid for tid in ordered if tid != rob.cur_task]
            if len(ordered) >= 1:
                task_sequences[aid] = ordered
        # Derive op_priority aligned with A4 sequence & critical-chain due
        op_best_rank: dict[str, float] = {}
        for ti, tid in enumerate(graph.rel_tids):
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


def compute_dynamic_coupling_features(graph: V19HeteroGraph) -> np.ndarray:
    """Extract topology-agnostic 6-D instantaneous production-transport coupling vector `phi(s_t)`."""
    n_op = max(1, len(graph.op_ids))
    n_t = max(1, len(graph.rel_tids))

    # phi_0: Active multi-task kitting & bottleneck unlock intensity
    if graph.task_feats.shape[0] > 0:
        kit_unlock_ratio = float(np.mean(graph.task_feats[:, 7] > 0.05))
    else:
        kit_unlock_ratio = 0.0
    if graph.op_feats.shape[0] > 0:
        kit_rem_ratio = float(np.mean(graph.op_feats[:, 2] > 0.05))
    else:
        kit_rem_ratio = 0.0
    phi_kit = 0.5 * (kit_unlock_ratio + kit_rem_ratio)

    # phi_1: Operation precedence DAG coupling density
    if graph.op_dag_adj.shape[0] > 1:
        dag_edges = float(np.sum(graph.op_dag_adj[:, :, 0]) + np.sum(graph.op_dag_adj[:, :, 2]))
        phi_dag = min(1.0, dag_edges / float(2 * n_op))
    else:
        phi_dag = 0.0

    # phi_2: Workstation resource window cliff urgency
    if graph.op_feats.shape[0] > 0:
        has_win = graph.op_feats[:, 4] > 0.5
        tight_win = has_win & (graph.op_feats[:, 5] < 0.75)
        phi_win = float(np.mean(tight_win))
    else:
        phi_win = 0.0

    # phi_3: Bay flexible routing & corridor detour dispersion
    if graph.op_bay_mask.shape[0] > 0:
        flex_ops = float(np.mean(np.sum(graph.op_bay_mask, axis=-1) > 1))
    else:
        flex_ops = 0.0
    detour_std = float(np.std(graph.bay_detour_norm)) if len(graph.bay_detour_norm) > 1 else 0.0
    phi_bay_corr = min(1.0, 0.6 * flex_ops + 0.8 * detour_std)

    # phi_4: Fleet execution disturbance & delay pressure
    if graph.agv_feats.shape[0] > 0:
        agv_delay = float(np.mean(np.abs(graph.agv_feats[:, 5]) + np.abs(graph.agv_feats[:, 6]) + graph.agv_feats[:, 7]))
        phi_disturb = min(1.0, agv_delay)
    else:
        phi_disturb = 0.0

    # phi_5: Normalized releasable task concurrency
    phi_load = min(1.0, float(n_t) / max(1.0, float(len(graph.agv_ids)) * 2.0))

    return np.array([phi_kit, phi_dag, phi_win, phi_bay_corr, phi_disturb, phi_load], dtype=np.float32)


def compute_candidate_amr_footprint(
    snapshot: NativeSnapshot,
    base_assign: dict[str, str],
    base_bays: dict[str, str],
    base_robot_seq: dict[str, list[str]],
    cand_assign: dict[str, str],
    cand_bays: dict[str, str],
    cand_seq: Optional[dict[str, list[str]]],
) -> dict[str, Any]:
    """Compute exact CVCR-B2 per-robot modification footprint for a joint candidate.

    Returns which AMRs had their task assignment (`A3`), kitting bay dropoff (`A2`),
    or effective intra-robot task sequence (`A4`) modified relative to the incumbent,
    plus the filtered `diff_task_order` containing only AMRs with genuine sequence changes.
    """
    n_robots = max(1, len(snapshot.robots))
    assign_changed_rids = {
        cand_assign[tid]
        for tid in cand_assign
        if cand_assign[tid] != base_assign.get(tid)
    } | {
        base_assign[tid]
        for tid in base_assign
        if cand_assign.get(tid) != base_assign[tid]
    }
    assign_changed_rids.discard("")

    changed_ops = {
        op.op_id
        for op in snapshot.operations
        if base_bays.get(op.op_id, op.bay_id) != cand_bays.get(op.op_id, op.bay_id)
    }
    bay_changed_rids = {
        str(cand_assign.get(t.task_id))
        for t in snapshot.tasks
        if t.kit_op_id in changed_ops
        and t.status.lower() in ("pending", "assigned")
        and cand_assign.get(t.task_id)
    }

    order_changed_rids: set[str] = set()
    diff_task_order: dict[str, list[str]] = {}
    if cand_seq:
        for rid, seq in cand_seq.items():
            inc_seq = base_robot_seq.get(rid, [])
            seq_set = set(seq)
            inc_proj = [tid for tid in inc_seq if tid in seq_set]
            if rid in assign_changed_rids or list(seq) != inc_proj:
                order_changed_rids.add(rid)
                diff_task_order[rid] = list(seq)

    changed_rids = sorted(assign_changed_rids | bay_changed_rids | order_changed_rids)
    n_changed = len(changed_rids)
    return {
        "assign_changed_rids": sorted(assign_changed_rids),
        "bay_changed_rids": sorted(bay_changed_rids),
        "order_changed_rids": sorted(order_changed_rids),
        "changed_rids": changed_rids,
        "n_changed_robots": n_changed,
        "n_robots": n_robots,
        "changed_amr_ratio": float(n_changed) / float(n_robots),
        "reused_amr_ratio": float(max(0, n_robots - n_changed)) / float(n_robots),
        "is_b2_incremental_hit": n_changed < n_robots,
        "is_b2_zero_route_hit": n_changed == 0,
        "is_b2_full_fallback": n_changed >= n_robots,
        "diff_task_order": diff_task_order if diff_task_order else None,
    }


class V19HeteroConstructorProposer:
    """Safe online joint constructor proposing verified `CandidateProposal` objects."""

    def __init__(
        self,
        ckpt_path: Path | str,
        *,
        policy_id: str = "V19_HETERO_BC",
        action_stage: Optional[str] = None,
        ablation_mode: Optional[str] = None,
        backend: str = "auto",
        allow_untrained: bool = False,
        use_rust: bool = True,
        seed: int = 0,
        sample_candidates: int = 3,
        alpha_makespan: float = 3.0,
        lambda_change: float = 15.0,
        hysteresis_margin: float = 10.0,
        evaluator_mode: str = "B2",
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.policy_id = str(policy_id)
        self.req_action_stage = action_stage
        self.req_ablation_mode = ablation_mode
        self.backend = str(backend)
        self.allow_untrained = bool(allow_untrained)
        self.use_rust = bool(use_rust)
        self.seed = int(seed)
        self.sample_candidates = max(1, int(sample_candidates))
        self.alpha_makespan = float(alpha_makespan)
        self.lambda_change = float(lambda_change)
        self.hysteresis_margin = float(hysteresis_margin)
        self.evaluator_mode = str(evaluator_mode)

        self._net: Any = None
        self._eff_backend: Optional[str] = None
        self._checkpoint_loaded: bool = False
        self._checkpoint_sha256: Optional[str] = None
        self._cfg: dict[str, Any] = {}
        self.total_cands_evaluated: int = 0
        self.total_b2_inc_hits: int = 0
        self.total_b2_fallbacks: int = 0
        self.total_b1_calls: int = 0
        self.total_a1_calls: int = 0
        self.total_changed_ratios: list[float] = []
        self.total_eval_lats_ms: list[float] = []

    def _get_net(self) -> Any:
        if self._net is None:
            net, used_b, sha, cfg = _load_v19_checkpoint(
                self.ckpt_path,
                backend=self.backend,
                allow_untrained=self.allow_untrained,
                action_stage=self.req_action_stage,
                ablation_mode=self.req_ablation_mode,
            )
            self._net = net
            self._eff_backend = used_b
            self._checkpoint_loaded = sha is not None
            self._checkpoint_sha256 = sha
            self._cfg = cfg
        return self._net

    def _effective_backend(self) -> str:
        if self._eff_backend is None:
            self._get_net()
        assert self._eff_backend is not None
        return self._eff_backend

    def __call__(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        budget_ms: float = 35.0,
    ) -> CandidateProposal:
        t0 = _cvcr_now_s()
        deadline = t0 + max(0.008, float(budget_ms) * 0.78 / 1000.0)
        net_model = self._get_net()
        eff_ablation = str(self._cfg.get("ablation_mode", "full"))
        eff_stage = str(self._cfg.get("action_stage", "A123"))

        clf = classify_commitments(snapshot)
        raw_graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        graph = apply_graph_ablation(raw_graph, eff_ablation)

        seed_b2 = self.evaluator_mode == "B2_SEEDED"
        raw_ev_mode = "B2" if seed_b2 else self.evaluator_mode
        ev = CVCREvaluator(
            network,
            snapshot,
            alpha_makespan=self.alpha_makespan,
            lambda_change=self.lambda_change,
            use_rust=self.use_rust,
            mode=raw_ev_mode,
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
        if seed_b2 and keep_eval.valid:
            ev.set_incumbent(
                graph.base_assign,
                keep_eval,
                op_priority=base_op_prio,
                op_bay_overrides=graph.base_bays,
            )

        best_eval: EvalResult = keep_eval
        best_score: float = float(keep_eval.objective)
        best_assign = dict(graph.base_assign)
        best_bays = dict(graph.base_bays)
        best_crews = dict(graph.base_crews)
        best_op_prio = dict(base_op_prio)
        best_seq: Optional[dict[str, list[str]]] = None
        fallback_keep = True
        neural_cands_eval = 0
        changed_ratios: list[float] = []
        b2_struct_hits = 0
        b2_struct_fallbacks = 0
        rob_cur_healthy = {
            r.robot_id: r.cur_task
            for r in snapshot.robots
            if r.cur_task and r.robot_id not in clf.disturbed_robots
        }

        if len(graph.rel_tids) > 0 and len(graph.agv_ids) >= 1:
            np_rng = np.random.default_rng(self.seed + int(snapshot.tick))
            seen_sigs: set[tuple[Any, ...]] = {
                (
                    tuple(sorted(graph.base_assign.items())),
                    tuple(sorted(graph.base_bays.items())),
                    tuple(sorted(graph.base_crews.items())),
                    None,
                )
            }
            for k in range(self.sample_candidates):
                if k > 0 and _cvcr_now_s() >= deadline:
                    break
                cand = decode_v19_candidate(
                    net_model,
                    graph,
                    snapshot,
                    greedy=(k == 0),
                    np_rng=np_rng,
                    tau=0.45,
                )
                sig = (
                    tuple(sorted(cand.assign.items())),
                    tuple(sorted(cand.bays.items())),
                    tuple(sorted(cand.crews.items())),
                    tuple(sorted((aid, tuple(sq)) for aid, sq in cand.task_sequences.items()))
                    if cand.task_sequences
                    else None,
                )
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)
                fp = compute_candidate_amr_footprint(
                    snapshot,
                    graph.base_assign,
                    graph.base_bays,
                    keep_eval.robot_order_seq,
                    cand.assign,
                    cand.bays,
                    cand.task_sequences,
                )
                changed_ratios.append(float(fp["changed_amr_ratio"]))
                if fp["is_b2_incremental_hit"]:
                    b2_struct_hits += 1
                else:
                    b2_struct_fallbacks += 1

                ev_c = ev.evaluate(
                    cand.assign,
                    op_priority=base_op_prio if seed_b2 else cand.op_prio,
                    op_bay_overrides=cand.bays,
                    op_crew_overrides=cand.crews,
                    task_order_override=fp["diff_task_order"] if seed_b2 else cand.task_sequences,
                )
                neural_cands_eval += 1
                changed_ops = sum(
                    1
                    for oid in graph.op_ids
                    if cand.bays.get(oid) != graph.base_bays.get(oid)
                    or cand.crews.get(oid) != graph.base_crews.get(oid)
                )
                qh_preempts = 0
                if cand.task_sequences:
                    for aid_r, cur_tid in rob_cur_healthy.items():
                        seq_r = cand.task_sequences.get(aid_r)
                        if seq_r and seq_r[0] != cur_tid and cur_tid in graph.clf.releasable_tasks:
                            qh_preempts += 1
                cand_score = (
                    float(ev_c.objective)
                    + self.lambda_change * float(changed_ops + qh_preempts)
                )
                margin = self.hysteresis_margin if best_eval is keep_eval else 1e-4
                if ev_c.valid and (not best_eval.valid or cand_score + margin < best_score):
                    best_eval = ev_c
                    best_score = cand_score
                    best_assign = cand.assign
                    best_bays = cand.bays
                    best_crews = cand.crews
                    best_op_prio = cand.op_prio
                    best_seq = cand.task_sequences
                    fallback_keep = False

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
            task_sequences=best_seq if eff_stage == "A1234" else None,
        )
        beat_keep = bool(
            not fallback_keep
            and best_eval.valid
            and keep_eval.valid
            and best_eval.objective + 1e-4 < keep_eval.objective
        )
        self.total_cands_evaluated += int(neural_cands_eval)
        self.total_b2_inc_hits += int(ev.b2_inc_hits)
        self.total_b2_fallbacks += int(ev.b2_fallbacks)
        self.total_b1_calls += int(ev.b1_calls)
        self.total_a1_calls += int(ev.a1_calls)
        self.total_changed_ratios.extend(changed_ratios)
        self.total_eval_lats_ms.extend(ev.eval_lats_ms)
        prop.telemetry.update(
            checkpoint_loaded=self._checkpoint_loaded,
            checkpoint_sha256=self._checkpoint_sha256,
            inference_backend=self._effective_backend(),
            action_stage=eff_stage,
            ablation_mode=eff_ablation,
            evaluator_mode=self.evaluator_mode,
            neural_candidates_evaluated=neural_cands_eval,
            b2_inc_hits=int(ev.b2_inc_hits),
            b2_fallbacks=int(ev.b2_fallbacks),
            b1_calls=int(ev.b1_calls),
            a1_calls=int(ev.a1_calls),
            b2_structural_hits=int(b2_struct_hits),
            b2_structural_fallbacks=int(b2_struct_fallbacks),
            mean_changed_amr_ratio=round(float(np.mean(changed_ratios)) if changed_ratios else 0.0, 4),
            mean_reused_amr_ratio=round(float(1.0 - np.mean(changed_ratios)) if changed_ratios else 1.0, 4),
            mean_eval_lat_ms=round(float(np.mean(ev.eval_lats_ms)) if ev.eval_lats_ms else 0.0, 4),
            fallback_keep=fallback_keep,
            beat_keep=beat_keep,
            keep_obj=round(float(keep_eval.objective), 4),
            best_obj=round(float(best_eval.objective), 4),
        )
        return prop


class V19AdaptiveCouplingProposer:
    """Adaptive Heterogeneous-Graph Reinforcement Learning (`AHG-RL` / `V19_ADAPTIVE_RL`) proposer.

    Combines Engine A (Hetero-GNN + BC->SMDP-PPO + Contextual LinUCB over 6-D coupling vector
    `phi(s_t)`) with Engine B (`CVCR-B2` commitment-preserving incremental evaluation) and
    Engine C (atomic transaction proposal).
    """

    def __init__(
        self,
        full_ckpt_path: Path | str,
        staged_ckpt_path: Path | str,
        *,
        policy_id: str = "V19_ADAPTIVE_RL",
        mode: str = "bandit",
        backend: str = "auto",
        allow_untrained: bool = False,
        use_rust: bool = True,
        seed: int = 0,
        sample_candidates: int = 3,
        alpha_makespan: float = 3.0,
        lambda_change: float = 15.0,
        hysteresis_margin: float = 10.0,
        evaluator_mode: str = "B2",
    ) -> None:
        if mode not in ("rule", "bandit", "b2_guard"):
            raise ValueError(f"Unsupported adaptive mode {mode!r}; expected 'rule', 'bandit', or 'b2_guard'.")
        self.full_ckpt_path = Path(full_ckpt_path)
        self.staged_ckpt_path = Path(staged_ckpt_path)
        self.policy_id = str(policy_id)
        self.mode = str(mode)
        self.backend = str(backend)
        self.allow_untrained = bool(allow_untrained)
        self.use_rust = bool(use_rust)
        self.seed = int(seed)
        self.sample_candidates = max(1, int(sample_candidates))
        self.alpha_makespan = float(alpha_makespan)
        self.lambda_change = float(lambda_change)
        self.hysteresis_margin = float(hysteresis_margin)
        self.evaluator_mode = str(evaluator_mode)

        self._full_net: Any = None
        self._staged_net: Any = None
        self._eff_backend: Optional[str] = None
        self._checkpoint_loaded: bool = False
        self._checkpoint_sha256: Optional[str] = None

        # Online Contextual LinUCB state (arm 0 = Staged, arm 1 = Full Joint)
        self._A = [np.eye(7, dtype=np.float64) * 1.5 for _ in range(2)]
        self._b = [np.zeros((7,), dtype=np.float64) for _ in range(2)]
        self._b[1][0] = 0.15
        self.full_selected_calls = 0
        self.staged_selected_calls = 0
        self.total_cands_evaluated: int = 0
        self.total_b2_inc_hits: int = 0
        self.total_b2_fallbacks: int = 0
        self.total_b1_calls: int = 0
        self.total_a1_calls: int = 0
        self.total_changed_ratios: list[float] = []
        self.total_eval_lats_ms: list[float] = []

    def _ensure_nets(self) -> tuple[Any, Any]:
        if self._full_net is None or self._staged_net is None:
            f_net, f_b, f_sha, _ = _load_v19_checkpoint(
                self.full_ckpt_path,
                backend=self.backend,
                allow_untrained=self.allow_untrained,
                action_stage="A1234",
                ablation_mode="full",
            )
            s_net, _, s_sha, _ = _load_v19_checkpoint(
                self.staged_ckpt_path,
                backend=self.backend,
                allow_untrained=self.allow_untrained,
                action_stage="A1234",
                ablation_mode="staged",
            )
            self._full_net = f_net
            self._staged_net = s_net
            self._eff_backend = f_b
            self._checkpoint_loaded = (f_sha is not None) and (s_sha is not None)
            self._checkpoint_sha256 = f_sha
        return self._full_net, self._staged_net

    def _effective_backend(self) -> str:
        if self._eff_backend is None:
            self._ensure_nets()
        assert self._eff_backend is not None
        return self._eff_backend

    def __call__(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        budget_ms: float = 35.0,
    ) -> CandidateProposal:
        full_net, staged_net = self._ensure_nets()
        t0 = _cvcr_now_s()
        deadline = t0 + max(0.008, float(budget_ms) * 0.78 / 1000.0)

        clf = classify_commitments(snapshot)
        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        phi = compute_dynamic_coupling_features(graph)
        coupling_score = float(0.30 * phi[0] + 0.25 * phi[1] + 0.20 * phi[2] + 0.25 * phi[3])

        seed_b2 = self.evaluator_mode == "B2_SEEDED"
        raw_ev_mode = "B2" if seed_b2 else self.evaluator_mode
        ev = CVCREvaluator(
            network,
            snapshot,
            alpha_makespan=self.alpha_makespan,
            lambda_change=self.lambda_change,
            use_rust=self.use_rust,
            mode=raw_ev_mode,
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
        if seed_b2 and keep_eval.valid:
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
        best_seq: Optional[dict[str, list[str]]] = None
        fallback_keep = True
        chosen_arm = "keep"
        selected_mode = "keep"
        neural_cands_eval = 0
        changed_ratios: list[float] = []
        b2_struct_hits = 0
        b2_struct_fallbacks = 0

        rob_cur_healthy = {
            r.robot_id: r.cur_task
            for r in snapshot.robots
            if r.cur_task and r.robot_id not in clf.disturbed_robots
        }

        def _score_cand(cand: Any, ev_c: EvalResult, is_full: bool) -> float:
            changed_ops = sum(
                1
                for oid in graph.op_ids
                if cand.bays.get(oid) != graph.base_bays.get(oid)
                or cand.crews.get(oid) != graph.base_crews.get(oid)
            )
            qh_preempts = 0
            if cand.task_sequences:
                for aid_r, cur_tid in rob_cur_healthy.items():
                    seq_r = cand.task_sequences.get(aid_r)
                    if seq_r and seq_r[0] != cur_tid and cur_tid in graph.clf.releasable_tasks:
                        qh_preempts += 1
            raw_s = float(ev_c.objective) + self.lambda_change * float(changed_ops + qh_preempts)
            if self.mode == "b2_guard" and is_full and changed_ops > 0:
                keep_max_op = max(keep_eval.op_end.values(), default=snapshot.tick)
                cand_max_op = max(ev_c.op_end.values(), default=snapshot.tick)
                if coupling_score < 0.22 or cand_max_op > keep_max_op + 12:
                    raw_s += 25.0 * float(changed_ops)
            return raw_s

        if len(graph.rel_tids) > 0 and len(graph.agv_ids) >= 1:
            np_rng = np.random.default_rng(self.seed + int(snapshot.tick))
            if self.mode == "bandit":
                x = np.concatenate([[1.0], phi.astype(np.float64)])
                ucb = []
                for arm_i in range(2):
                    A_inv = np.linalg.inv(self._A[arm_i])
                    theta = A_inv @ self._b[arm_i]
                    bonus = 0.25 * float(np.sqrt(max(0.0, x @ A_inv @ x)))
                    ucb.append(float(theta @ x) + bonus)
                arm_pick = 1 if ucb[1] >= ucb[0] else 0
            else:
                arm_pick = 1 if coupling_score >= 0.24 else 0

            net_m = full_net if arm_pick == 1 else staged_net
            selected_mode = "full" if arm_pick == 1 else "staged"
            for k in range(self.sample_candidates):
                if k > 0 and _cvcr_now_s() >= deadline:
                    break
                cand = decode_v19_candidate(net_m, graph, snapshot, greedy=(k == 0), np_rng=np_rng, tau=0.45)
                fp = compute_candidate_amr_footprint(
                    snapshot,
                    graph.base_assign,
                    graph.base_bays,
                    keep_eval.robot_order_seq,
                    cand.assign,
                    cand.bays,
                    cand.task_sequences,
                )
                changed_ratios.append(float(fp["changed_amr_ratio"]))
                if fp["is_b2_incremental_hit"]:
                    b2_struct_hits += 1
                else:
                    b2_struct_fallbacks += 1

                ev_c = ev.evaluate(
                    cand.assign,
                    op_priority=base_op_prio if seed_b2 else cand.op_prio,
                    op_bay_overrides=cand.bays,
                    op_crew_overrides=cand.crews,
                    task_order_override=fp["diff_task_order"] if seed_b2 else cand.task_sequences,
                )
                neural_cands_eval += 1
                cs = _score_cand(cand, ev_c, arm_pick == 1)
                margin = self.hysteresis_margin if best_eval is keep_eval else 1e-4
                if ev_c.valid and (not best_eval.valid or cs + margin < best_score):
                    best_eval, best_score = ev_c, cs
                    best_assign, best_bays, best_crews = cand.assign, cand.bays, cand.crews
                    best_op_prio, best_seq = cand.op_prio, cand.task_sequences
                    chosen_arm = selected_mode
                    fallback_keep = False

            if self.mode == "bandit":
                impr = max(-1.0, min(1.0, (float(keep_eval.objective) - best_score) / 200.0))
                self._A[arm_pick] += np.outer(x, x)
                self._b[arm_pick] += impr * x
            if arm_pick == 1:
                self.full_selected_calls += 1
            else:
                self.staged_selected_calls += 1

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
            task_sequences=best_seq,
        )
        beat_keep = bool(
            not fallback_keep
            and best_eval.valid
            and keep_eval.valid
            and best_eval.objective + 1e-4 < keep_eval.objective
        )
        self.total_cands_evaluated += int(neural_cands_eval)
        self.total_b2_inc_hits += int(ev.b2_inc_hits)
        self.total_b2_fallbacks += int(ev.b2_fallbacks)
        self.total_b1_calls += int(ev.b1_calls)
        self.total_a1_calls += int(ev.a1_calls)
        self.total_changed_ratios.extend(changed_ratios)
        self.total_eval_lats_ms.extend(ev.eval_lats_ms)
        prop.telemetry.update(
            checkpoint_loaded=self._checkpoint_loaded,
            checkpoint_sha256=self._checkpoint_sha256,
            inference_backend=self._effective_backend(),
            adaptive_mode=self.mode,
            evaluator_mode=self.evaluator_mode,
            selected_arm=selected_mode,
            chosen_arm=chosen_arm,
            coupling_score=round(coupling_score, 4),
            phi=[round(float(x), 4) for x in phi],
            neural_candidates_evaluated=neural_cands_eval,
            b2_inc_hits=int(ev.b2_inc_hits),
            b2_fallbacks=int(ev.b2_fallbacks),
            b1_calls=int(ev.b1_calls),
            a1_calls=int(ev.a1_calls),
            b2_structural_hits=int(b2_struct_hits),
            b2_structural_fallbacks=int(b2_struct_fallbacks),
            mean_changed_amr_ratio=round(float(np.mean(changed_ratios)) if changed_ratios else 0.0, 4),
            mean_reused_amr_ratio=round(float(1.0 - np.mean(changed_ratios)) if changed_ratios else 1.0, 4),
            mean_eval_lat_ms=round(float(np.mean(ev.eval_lats_ms)) if ev.eval_lats_ms else 0.0, 4),
            fallback_keep=fallback_keep,
            beat_keep=beat_keep,
            keep_obj=round(float(keep_eval.objective), 4),
            best_obj=round(float(best_eval.objective), 4),
        )
        return prop


__all__ = [
    "ABLATION_MODES",
    "ACTION_STAGES",
    "DecodedJointCandidate",
    "NumpyV19HeteroConstructorNet",
    "V19AdaptiveCouplingProposer",
    "V19HeteroConstructorNet",
    "V19HeteroConstructorProposer",
    "compute_candidate_amr_footprint",
    "compute_dynamic_coupling_features",
    "decode_v19_candidate",
    "export_v19_npz",
    "graph_to_torch_tensors",
]
