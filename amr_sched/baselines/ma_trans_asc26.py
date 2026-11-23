"""Architecture-Faithful, Domain-Adapted Reimplementation of MA-Trans (Shen, Zhang & Jin, ASC 2026).

Reference:
  Yuxiang Shen, Xutang Zhang, and Tianguo Jin (2026),
  "Transformer-based multi-agent reinforcement learning for flexible job shop scheduling with AGVs",
  Applied Soft Computing, Vol. 193, 114899.

Core Architecture:
  1. **Job/Operation Transformer Encoder** (2-layer Pre-LN Multi-Head Self-Attention, `d_model=64`,
     `n_heads=4`, `d_ff=128`, GELU-tanh): encodes operations fused with their aggregated kit-task
     summaries, plus releasable tasks fused with parent-operation Transformer tokens.
  2. **Resource Graph Attention (GAT) Encoder** (1-layer Multi-Head Self-Attention over unified
     resource tokens `[Bay; Crew; AMR]`): enables explicit cross-resource interaction between
     machine/bay congestion, crew availability, and AMR fleet utilization.
  3. **Dual-Agent Coupled PPO Heads**:
     - **Machine/Operation Agent**: ranks releasable tasks (`job_logits`), selects compatible
       bays (`bay_logits`) and crews (`crew_logits`) with strict `-1e9` commitment/eligibility masks.
     - **AGV Transport Agent**: autoregressively assigns AMRs (`agv_logits`) conditioned on
       Transformer task tokens, GAT AMR & chosen-bay tokens, pairwise transport edge features,
       and dynamic incremental AMR load (`dyn_load`).
     - **Centralized Dual-Agent Critic**: averages Machine and AGV value heads over pooled
       Transformer + Resource GAT context.

Explicit Domain-Mapping Adaptations (documented transparently):
  - Multi-kit fan-out: when an operation requires multiple material kits, the AGV Transport
    Agent allocates an AMR to each releasable kit task sequentially in Job-Agent order.
  - Crew & commitment locking: `LOCKED` entities are strictly masked out via hard safety masks,
    and decoded candidates are verified inside the exact same `CVCREvaluator` (`35 ms` wall budget)
    with `KEEP` fallback — WITHOUT adding CoRe-Sched's 4th-stage queue-sequencing pointer head,
    Contextual LinUCB Full/Staged coupling switcher, dual-arm probe, or `J_A`-aligned surrogate.
  - Unified quarantined training: trained under the exact same quarantined 264-trajectory IL +
    288-episode closed-loop SMDP-PPO protocol and exports a `< 1e-5` pure-NumPy runtime mirror.
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
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError("PyTorch (`torch`) is required to instantiate MATransNet.")

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

ASC26_MA_TRANS_CKPT = resolve_default_checkpoint("baselines_2026/asc26_ma_trans.pt")
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_CKPT_CACHE: dict[str, tuple[Any, str]] = {}


def _gelu_tanh_np(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32, copy=False)
    return (0.5 * x32 * (1.0 + np.tanh(_SQRT_2_OVER_PI * (x32 + 0.044715 * (x32 * x32 * x32))))).astype(np.float32)


def _layernorm_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * weight + bias


def _linear_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (x @ weight.T + bias).astype(np.float32)


def _softmax_last_np(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(np.clip(x - m, -60.0, 0.0))
    return (e / np.maximum(1e-12, np.sum(e, axis=-1, keepdims=True))).astype(np.float32)


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


class _PreLNTransformerBlock(nn.Module):
    """Explicit Pre-LN Multi-Head Self-Attention + FFN block with deterministic NumPy parity."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 128) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads
        self.ln1 = nn.LayerNorm(self.d_model)
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.o_proj = nn.Linear(self.d_model, self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)
        self.ff1 = nn.Linear(self.d_model, int(d_ff))
        self.ff2 = nn.Linear(int(d_ff), self.d_model)

    def forward(self, x: "torch.Tensor", valid_mask: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        B, N, D = x.shape
        if N == 0:
            return x
        h = self.ln1(x)
        q = self.q_proj(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, N, dh]
        k = self.k_proj(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.d_head ** -0.5)  # [B, H, N, N]
        if valid_mask is not None:
            key_invalid = (~valid_mask.bool()).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            scores = scores.masked_fill(key_invalid, -1e9)
        attn = F.softmax(scores, dim=-1)
        if valid_mask is not None:
            attn = attn * valid_mask.to(attn.dtype).unsqueeze(1).unsqueeze(-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.o_proj(ctx)
        h2 = self.ln2(x)
        x = x + self.ff2(F.gelu(self.ff1(h2), approximate="tanh"))
        if valid_mask is not None:
            x = x * valid_mask.to(x.dtype).unsqueeze(-1)
        return x


def _np_transformer_block(x: np.ndarray, w: dict[str, np.ndarray], prefix: str, n_heads: int = 4) -> np.ndarray:
    """Exact float32 NumPy mirror of `_PreLNTransformerBlock` for unbatched `[N, D]` tokens."""
    N, D = x.shape
    if N == 0:
        return x.astype(np.float32)
    dh = D // n_heads
    h = _layernorm_np(x, w[f"{prefix}.ln1.weight"], w[f"{prefix}.ln1.bias"])
    q = _linear_np(h, w[f"{prefix}.q_proj.weight"], w[f"{prefix}.q_proj.bias"]).reshape(N, n_heads, dh).transpose(1, 0, 2)
    k = _linear_np(h, w[f"{prefix}.k_proj.weight"], w[f"{prefix}.k_proj.bias"]).reshape(N, n_heads, dh).transpose(1, 0, 2)
    v = _linear_np(h, w[f"{prefix}.v_proj.weight"], w[f"{prefix}.v_proj.bias"]).reshape(N, n_heads, dh).transpose(1, 0, 2)
    scores = np.matmul(q, np.swapaxes(k, -2, -1)) * np.float32(dh ** -0.5)
    attn = _softmax_last_np(scores)
    ctx = np.matmul(attn, v).transpose(1, 0, 2).reshape(N, D)
    x = x + _linear_np(ctx, w[f"{prefix}.o_proj.weight"], w[f"{prefix}.o_proj.bias"])
    h2 = _layernorm_np(x, w[f"{prefix}.ln2.weight"], w[f"{prefix}.ln2.bias"])
    ff = _linear_np(_gelu_tanh_np(_linear_np(h2, w[f"{prefix}.ff1.weight"], w[f"{prefix}.ff1.bias"])), w[f"{prefix}.ff2.weight"], w[f"{prefix}.ff2.bias"])
    return (x + ff).astype(np.float32)


class MATransNet(nn.Module):
    """Transformer + Resource GAT Dual-Agent PPO network (Shen, Zhang & Jin, ASC 2026)."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 128) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_ff = int(d_ff)

        # Input entity projections
        self.op_in = nn.Linear(OP_FEAT_DIM, self.d_model)
        self.task_in = nn.Linear(TASK_FEAT_DIM + self.d_model, self.d_model)
        self.bay_in = nn.Linear(BAY_FEAT_DIM, self.d_model)
        self.crew_in = nn.Linear(CREW_FEAT_DIM, self.d_model)
        self.agv_in = nn.Linear(AGV_FEAT_DIM, self.d_model)
        # Resource-type embeddings (0=bay, 1=crew, 2=agv)
        self.res_type_emb = nn.Embedding(3, self.d_model)

        # 2-Layer Job/Operation Transformer Encoder
        self.op_tf1 = _PreLNTransformerBlock(self.d_model, self.n_heads, self.d_ff)
        self.op_tf2 = _PreLNTransformerBlock(self.d_model, self.n_heads, self.d_ff)
        self.task_tf = _PreLNTransformerBlock(self.d_model, self.n_heads, self.d_ff)

        # 1-Layer Resource GAT Encoder over [Bay; Crew; AMR]
        self.res_gat = _PreLNTransformerBlock(self.d_model, self.n_heads, self.d_ff)

        # Agent 1 (Machine / Job Agent) heads
        self.job_fc1 = nn.Linear(3 * self.d_model, self.d_model)
        self.job_fc2 = nn.Linear(self.d_model, 1)

        self.bay_fc1 = nn.Linear(3 * self.d_model + OP_BAY_EDGE_DIM, self.d_model)
        self.bay_fc2 = nn.Linear(self.d_model, 1)

        self.crew_fc1 = nn.Linear(3 * self.d_model, self.d_model)
        self.crew_fc2 = nn.Linear(self.d_model, 1)

        # Agent 2 (AGV Transport Agent) head
        self.agv_fc1 = nn.Linear(4 * self.d_model + TASK_AGV_EDGE_DIM + 2, self.d_model)
        self.agv_fc2 = nn.Linear(self.d_model, 1)

        # Centralized Dual-Agent Critic (Machine critic + AGV critic)
        self.machine_critic_1 = nn.Linear(2 * self.d_model, self.d_model)
        self.machine_critic_2 = nn.Linear(self.d_model, 1)
        self.agv_critic_1 = nn.Linear(2 * self.d_model, self.d_model)
        self.agv_critic_2 = nn.Linear(self.d_model, 1)

    def config_dict(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "arch": "ASC26_MA_TRANS",
        }

    def encode(self, tensors: dict[str, "torch.Tensor"]) -> dict[str, "torch.Tensor"]:
        op_x = tensors["op_x"]
        task_x = tensors["task_x"]
        bay_x = tensors["bay_x"]
        crew_x = tensors["crew_x"]
        agv_x = tensors["agv_x"]
        op_valid = tensors.get("op_valid")
        task_valid = tensors.get("task_valid")
        bay_valid = tensors.get("bay_valid")
        crew_valid = tensors.get("crew_valid")
        agv_valid = tensors.get("agv_valid")
        B = op_x.shape[0]
        No = op_x.shape[1]
        Nt = task_x.shape[1]
        Nb = bay_x.shape[1]
        Nc = crew_x.shape[1]
        Na = agv_x.shape[1]
        dev = op_x.device

        # 1. Operation 2-layer Transformer
        if No > 0:
            h_op = F.gelu(self.op_in(op_x), approximate="tanh")
            h_op = self.op_tf2(self.op_tf1(h_op, op_valid), op_valid)
        else:
            h_op = torch.zeros((B, 0, self.d_model), dtype=op_x.dtype, device=dev)

        # 2. Task Transformer conditioned on parent operation token
        if Nt > 0:
            if No > 0:
                t2o = tensors["task_to_op"]
                has_op = (t2o >= 0) & (t2o < No)
                safe_t2o = torch.clamp(t2o, 0, max(0, No - 1))
                par_op = h_op.gather(1, safe_t2o.unsqueeze(-1).expand(B, Nt, self.d_model))
                par_op = par_op * has_op.unsqueeze(-1).to(par_op.dtype)
            else:
                par_op = torch.zeros((B, Nt, self.d_model), dtype=task_x.dtype, device=dev)
            h_task = F.gelu(self.task_in(torch.cat([task_x, par_op], dim=-1)), approximate="tanh")
            h_task = self.task_tf(h_task, task_valid)
        else:
            h_task = torch.zeros((B, 0, self.d_model), dtype=task_x.dtype, device=dev)

        # 3. Unified Resource GAT over [Bay; Crew; AMR]
        rt = self.res_type_emb.weight  # [3, D]
        h_bay0 = F.gelu(self.bay_in(bay_x), approximate="tanh") + rt[0].view(1, 1, -1) if Nb > 0 else torch.zeros((B, 0, self.d_model), dtype=op_x.dtype, device=dev)
        h_crew0 = F.gelu(self.crew_in(crew_x), approximate="tanh") + rt[1].view(1, 1, -1) if Nc > 0 else torch.zeros((B, 0, self.d_model), dtype=op_x.dtype, device=dev)
        h_agv0 = F.gelu(self.agv_in(agv_x), approximate="tanh") + rt[2].view(1, 1, -1) if Na > 0 else torch.zeros((B, 0, self.d_model), dtype=op_x.dtype, device=dev)
        res_all = torch.cat([h_bay0, h_crew0, h_agv0], dim=1)
        if bay_valid is not None and crew_valid is not None and agv_valid is not None:
            res_valid = torch.cat([bay_valid, crew_valid, agv_valid], dim=1)
        else:
            res_valid = None
        if res_all.shape[1] > 0:
            res_out = self.res_gat(res_all, res_valid)
        else:
            res_out = res_all
        h_bay = res_out[:, :Nb, :]
        h_crew = res_out[:, Nb : Nb + Nc, :]
        h_agv = res_out[:, Nb + Nc :, :]

        # Pooled Transformer & Resource contexts
        g_job = 0.5 * (_masked_mean_torch(h_op, op_valid) + _masked_mean_torch(h_task, task_valid))
        g_res = (_masked_mean_torch(h_bay, bay_valid) + _masked_mean_torch(h_crew, crew_valid) + _masked_mean_torch(h_agv, agv_valid)) / 3.0
        g_ctx = torch.cat([g_job, g_res], dim=-1)  # [B, 2*D]

        v_m = self.machine_critic_2(F.gelu(self.machine_critic_1(g_ctx), approximate="tanh")).squeeze(-1)
        v_a = self.agv_critic_2(F.gelu(self.agv_critic_1(g_ctx), approximate="tanh")).squeeze(-1)
        return {
            "h_op": h_op,
            "h_task": h_task,
            "h_bay": h_bay,
            "h_crew": h_crew,
            "h_agv": h_agv,
            "g_job": g_job,
            "g_res": g_res,
            "value": 0.5 * (v_m + v_a),
        }

    def forward_machine_heads(
        self,
        enc: dict[str, "torch.Tensor"],
        tensors: dict[str, "torch.Tensor"],
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        h_op, h_task, h_bay, h_crew = enc["h_op"], enc["h_task"], enc["h_bay"], enc["h_crew"]
        g_job, g_res = enc["g_job"], enc["g_res"]
        B, No, D = h_op.shape
        Nt = h_task.shape[1]
        Nb = h_bay.shape[1]
        Nc = h_crew.shape[1]
        dev = h_op.device

        if Nt > 0:
            j_in = torch.cat([h_task, g_job.unsqueeze(1).expand(B, Nt, D), g_res.unsqueeze(1).expand(B, Nt, D)], dim=-1)
            job_logits = self.job_fc2(F.gelu(self.job_fc1(j_in), approximate="tanh")).squeeze(-1)
        else:
            job_logits = torch.zeros((B, 0), dtype=h_op.dtype, device=dev)

        if No > 0 and Nb > 0:
            b_in = torch.cat(
                [
                    h_op.unsqueeze(2).expand(B, No, Nb, D),
                    h_bay.unsqueeze(1).expand(B, No, Nb, D),
                    g_res.unsqueeze(1).unsqueeze(2).expand(B, No, Nb, D),
                    tensors["op_bay_x"],
                ],
                dim=-1,
            )
            bay_logits = self.bay_fc2(F.gelu(self.bay_fc1(b_in), approximate="tanh")).squeeze(-1)
            bay_logits = bay_logits.masked_fill(~tensors["op_bay_mask"].bool(), -1e9)
        else:
            bay_logits = torch.zeros((B, No, Nb), dtype=h_op.dtype, device=dev)

        if No > 0 and Nc > 0:
            c_in = torch.cat(
                [
                    h_op.unsqueeze(2).expand(B, No, Nc, D),
                    h_crew.unsqueeze(1).expand(B, No, Nc, D),
                    g_res.unsqueeze(1).unsqueeze(2).expand(B, No, Nc, D),
                ],
                dim=-1,
            )
            crew_logits = self.crew_fc2(F.gelu(self.crew_fc1(c_in), approximate="tanh")).squeeze(-1)
            crew_logits = crew_logits.masked_fill(~tensors["op_crew_mask"].bool(), -1e9)
        else:
            crew_logits = torch.zeros((B, No, Nc), dtype=h_op.dtype, device=dev)

        return job_logits, bay_logits, crew_logits

    def forward_agv_step(
        self,
        h_task_j: "torch.Tensor",       # [B, D]
        h_agv: "torch.Tensor",          # [B, Na, D]
        task_agv_x_j: "torch.Tensor",   # [B, Na, 8]
        h_sel_bay_j: "torch.Tensor",    # [B, D]
        dyn_load: "torch.Tensor",       # [B, Na, 2]
        g_res: "torch.Tensor",          # [B, D]
        mask_j: "torch.Tensor",         # [B, Na]
    ) -> "torch.Tensor":
        B, Na, D = h_agv.shape
        if Na == 0:
            return torch.zeros((B, 0), dtype=h_agv.dtype, device=h_agv.device)
        a_in = torch.cat(
            [
                h_task_j.unsqueeze(1).expand(B, Na, D),
                h_agv,
                h_sel_bay_j.unsqueeze(1).expand(B, Na, D),
                g_res.unsqueeze(1).expand(B, Na, D),
                task_agv_x_j,
                dyn_load,
            ],
            dim=-1,
        )
        logits = self.agv_fc2(F.gelu(self.agv_fc1(a_in), approximate="tanh")).squeeze(-1)
        return logits.masked_fill(~mask_j.bool(), -1e9)

    def forward_teacher_forced(
        self,
        tensors: dict[str, "torch.Tensor"],
        target_bays: "torch.Tensor",
        target_agvs: "torch.Tensor",
        task_focus: Optional["torch.Tensor"] = None,
    ) -> dict[str, "torch.Tensor"]:
        """Differentiable teacher-forced batched forward pass for IL & vectorized SMDP-PPO."""
        enc = self.encode(tensors)
        job_logits, bay_logits, crew_logits = self.forward_machine_heads(enc, tensors)
        h_task, h_bay, h_agv, g_res = enc["h_task"], enc["h_bay"], enc["h_agv"], enc["g_res"]
        B, Nt, D = h_task.shape
        Na = h_agv.shape[1]
        Nb = h_bay.shape[1]
        No = enc["h_op"].shape[1]
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

                lg_j = self.forward_agv_step(ht_j, h_agv, ta_j, sb_j, dyn_load, g_res, mk_j)
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


class NumpyMATransNet:
    """Exact pure-NumPy float32 mirror of `MATransNet` (`< 1e-5` max abs diff vs PyTorch)."""

    def __init__(self, weights: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
        self.w = {str(k): np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        self.d_model = int(meta.get("d_model", 64))
        self.n_heads = int(meta.get("n_heads", 4))

    def encode(self, g: V19HeteroGraph) -> dict[str, np.ndarray]:
        w = self.w
        D = self.d_model
        No, Nt, Nb, Nc, Na = len(g.op_ids), len(g.rel_tids), len(g.bay_ids), len(g.crew_ids), len(g.agv_ids)

        if No > 0:
            h_op = _gelu_tanh_np(_linear_np(g.op_feats, w["op_in.weight"], w["op_in.bias"]))
            h_op = _np_transformer_block(h_op, w, "op_tf1", self.n_heads)
            h_op = _np_transformer_block(h_op, w, "op_tf2", self.n_heads)
        else:
            h_op = np.zeros((0, D), dtype=np.float32)

        if Nt > 0:
            par_op = np.zeros((Nt, D), dtype=np.float32)
            for j in range(Nt):
                oi = int(g.task_to_op[j])
                if 0 <= oi < No:
                    par_op[j] = h_op[oi]
            h_task = _gelu_tanh_np(_linear_np(np.concatenate([g.task_feats, par_op], axis=-1), w["task_in.weight"], w["task_in.bias"]))
            h_task = _np_transformer_block(h_task, w, "task_tf", self.n_heads)
        else:
            h_task = np.zeros((0, D), dtype=np.float32)

        rt = w["res_type_emb.weight"]
        h_bay0 = _gelu_tanh_np(_linear_np(g.bay_feats, w["bay_in.weight"], w["bay_in.bias"])) + rt[0] if Nb > 0 else np.zeros((0, D), dtype=np.float32)
        h_crew0 = _gelu_tanh_np(_linear_np(g.crew_feats, w["crew_in.weight"], w["crew_in.bias"])) + rt[1] if Nc > 0 else np.zeros((0, D), dtype=np.float32)
        h_agv0 = _gelu_tanh_np(_linear_np(g.agv_feats, w["agv_in.weight"], w["agv_in.bias"])) + rt[2] if Na > 0 else np.zeros((0, D), dtype=np.float32)
        res_all = np.concatenate([h_bay0, h_crew0, h_agv0], axis=0)
        res_out = _np_transformer_block(res_all, w, "res_gat", self.n_heads) if res_all.shape[0] > 0 else res_all
        h_bay = res_out[:Nb]
        h_crew = res_out[Nb : Nb + Nc]
        h_agv = res_out[Nb + Nc :]

        g_job = 0.5 * (_masked_mean_np(h_op) + _masked_mean_np(h_task))
        g_res = (_masked_mean_np(h_bay) + _masked_mean_np(h_crew) + _masked_mean_np(h_agv)) / 3.0
        return {
            "h_op": h_op,
            "h_task": h_task,
            "h_bay": h_bay,
            "h_crew": h_crew,
            "h_agv": h_agv,
            "g_job": g_job.astype(np.float32),
            "g_res": g_res.astype(np.float32),
        }

    def predict_machine_heads(self, enc: dict[str, np.ndarray], g: V19HeteroGraph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = self.w
        h_op, h_task, h_bay, h_crew = enc["h_op"], enc["h_task"], enc["h_bay"], enc["h_crew"]
        g_job, g_res = enc["g_job"], enc["g_res"]
        No, Nt, Nb, Nc = len(g.op_ids), len(g.rel_tids), len(g.bay_ids), len(g.crew_ids)

        if Nt > 0:
            j_in = np.concatenate([h_task, np.broadcast_to(g_job, (Nt, self.d_model)), np.broadcast_to(g_res, (Nt, self.d_model))], axis=-1)
            job_lg = _linear_np(_gelu_tanh_np(_linear_np(j_in, w["job_fc1.weight"], w["job_fc1.bias"])), w["job_fc2.weight"], w["job_fc2.bias"]).reshape(Nt)
        else:
            job_lg = np.zeros((0,), dtype=np.float32)

        if No > 0 and Nb > 0:
            b_in = np.concatenate(
                [
                    np.broadcast_to(h_op[:, None, :], (No, Nb, self.d_model)),
                    np.broadcast_to(h_bay[None, :, :], (No, Nb, self.d_model)),
                    np.broadcast_to(g_res[None, None, :], (No, Nb, self.d_model)),
                    g.op_bay_feats,
                ],
                axis=-1,
            )
            bay_lg = _linear_np(_gelu_tanh_np(_linear_np(b_in, w["bay_fc1.weight"], w["bay_fc1.bias"])), w["bay_fc2.weight"], w["bay_fc2.bias"]).reshape(No, Nb)
            bay_lg = np.where(g.op_bay_mask, bay_lg, np.float32(-1e9))
        else:
            bay_lg = np.zeros((No, Nb), dtype=np.float32)

        if No > 0 and Nc > 0:
            c_in = np.concatenate(
                [
                    np.broadcast_to(h_op[:, None, :], (No, Nc, self.d_model)),
                    np.broadcast_to(h_crew[None, :, :], (No, Nc, self.d_model)),
                    np.broadcast_to(g_res[None, None, :], (No, Nc, self.d_model)),
                ],
                axis=-1,
            )
            crew_lg = _linear_np(_gelu_tanh_np(_linear_np(c_in, w["crew_fc1.weight"], w["crew_fc1.bias"])), w["crew_fc2.weight"], w["crew_fc2.bias"]).reshape(No, Nc)
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
        h_agv = enc["h_agv"]
        Na = h_agv.shape[0]
        if Na == 0:
            return np.zeros((0,), dtype=np.float32)
        a_in = np.concatenate(
            [
                np.broadcast_to(enc["h_task"][j : j + 1], (Na, self.d_model)),
                h_agv,
                np.broadcast_to(h_sel_bay[None, :], (Na, self.d_model)),
                np.broadcast_to(enc["g_res"][None, :], (Na, self.d_model)),
                g.task_agv_feats[j],
                dyn_load,
            ],
            axis=-1,
        )
        lg = _linear_np(_gelu_tanh_np(_linear_np(a_in, w["agv_fc1.weight"], w["agv_fc1.bias"])), w["agv_fc2.weight"], w["agv_fc2.bias"]).reshape(Na)
        return np.where(g.task_agv_mask[j], lg, np.float32(-1e9)).astype(np.float32)


def export_ma_trans_npz(model: MATransNet, out_path: Path | str) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    arrays["__meta_d_model__"] = np.array([model.d_model], dtype=np.int64)
    arrays["__meta_n_heads__"] = np.array([model.n_heads], dtype=np.int64)
    np.savez(out, **arrays)
    return out


def load_ma_trans_model(
    ckpt_path: Path | str = ASC26_MA_TRANS_CKPT,
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
        meta = {
            "d_model": int(weights.pop("__meta_d_model__", np.array([64]))[0]),
            "n_heads": int(weights.pop("__meta_n_heads__", np.array([4]))[0]),
        }
        m = NumpyMATransNet(weights, meta)
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
        net = MATransNet(
            d_model=int(cfg.get("d_model", 64)),
            n_heads=int(cfg.get("n_heads", 4)),
            d_ff=int(cfg.get("d_ff", 128)),
        )
        sd = payload.get("state_dict") if "state_dict" in payload else payload.get("model", payload)
        net.load_state_dict(sd, strict=True)
        net.eval()
        _CKPT_CACHE[key] = (net, sha)
        return net, "torch", sha

    if allow_untrained and _HAS_TORCH:
        net = MATransNet()
        net.eval()
        return net, "torch", None
    raise FileNotFoundError(f"ASC26 MA-Trans checkpoint not found at {p} (or .npz).")


@dataclass
class DecodedMATransCandidate:
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


def decode_ma_trans_candidate(
    model: Any,
    g: V19HeteroGraph,
    snapshot: NativeSnapshot,
    *,
    greedy: bool = True,
    tau: float = 0.45,
    rng: Optional[np.random.Generator] = None,
) -> DecodedMATransCandidate:
    use_rng = rng if rng is not None else np.random.default_rng(0)
    No, Nt, Nb, Nc, Na = len(g.op_ids), len(g.rel_tids), len(g.bay_ids), len(g.crew_ids), len(g.agv_ids)

    if isinstance(model, NumpyMATransNet):
        enc = model.encode(g)
        job_lg, bay_lg, crew_lg = model.predict_machine_heads(enc, g)
    else:
        from amr_sched.rl.v19_hetero_constructor import graph_to_torch_tensors

        dev = next(model.parameters()).device
        ten = graph_to_torch_tensors(g, device=dev)
        with torch.no_grad():
            enc_t = model.encode(ten)
            jl_t, bl_t, cl_t = model.forward_machine_heads(enc_t, ten)
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
    D = enc["h_bay"].shape[-1] if Nb > 0 else enc["g_res"].shape[-1]

    for j in order:
        if Na == 0:
            break
        oi = int(g.task_to_op[j])
        bi = chosen_bay_idx[oi] if (0 <= oi < No and chosen_bay_idx[oi] >= 0) else int(g.task_incumbent_bay_idx[j])
        h_sel_bay = enc["h_bay"][bi] if (0 <= bi < Nb) else np.zeros((D,), dtype=np.float32)
        if isinstance(model, NumpyMATransNet):
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
                        torch.from_numpy(enc["g_res"][None, :]).to(dev),
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
    return DecodedMATransCandidate(
        assign=assign,
        bays=bays,
        crews=crews,
        op_prio=op_prio,
        chosen_bay_idx=chosen_bay_idx,
        chosen_crew_idx=chosen_crew_idx,
        chosen_agv_idx=chosen_agv_idx,
        task_order=order,
    )


class MATransASC26Proposer:
    """Online proposer wrapper for Shen, Zhang & Jin (ASC 2026) MA-Trans."""

    def __init__(
        self,
        ckpt_path: Path | str = ASC26_MA_TRANS_CKPT,
        *,
        policy_id: str = "ASC26_MA_TRANS",
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
            m, b, sha = load_ma_trans_model(
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
                cand = decode_ma_trans_candidate(model, graph, snapshot, greedy=(k == 0), tau=0.45, rng=rng)
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
