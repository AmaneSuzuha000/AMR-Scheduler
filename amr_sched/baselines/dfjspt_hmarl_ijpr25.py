"""Faithful Reproduction of Wang et al. (IJPR 2025): Hierarchical Multi-Agent DRL for DFJSP-T.

Reference:
  W. Wang, Y. Zhang, Y. Wang, and Y. Feng (2025),
  "Hierarchical multi-agent deep reinforcement learning for dynamic flexible job-shop
  scheduling with transportation", International Journal of Production Research (IJPR).
  Upstream reference implementation: https://github.com/ClouDaDaDa/DFJSPT_code

Reproduces the three-level independent action-masked MLP MARL architecture (`TorchFC`
hidden=[256, 256], Tanh activations, `clamp(log(mask), -1e9)` masking, PPO + IL loss):
  1. High-Level Agent (`JobActionMaskMLP`): Prioritizes releasable tasks/operations
     from flat local + global mean-pooled features (no cross-domain GNN message passing).
  2. Mid-Level Agent (`MachineActionMaskMLP`): Conditioned on each operation, selects
     compatible `(bay, crew)` via independent masked MLPs.
  3. Low-Level Agent (`TransbotActionMaskMLP`): Conditioned on the selected task and
     assigned bay, allocates an eligible AMR via an independent masked MLP.

Explicit domain-mapping adaptations documented transparently:
  - Multi-kit fan-out: when an operation requires multiple material kits, Level-3
    `TransbotActionMaskMLP` allocates an AMR to each releasable kit task in Level-1 order.
  - Commitment locking & spacetime realization: `LOCKED` entities are strictly masked out
    and decoded proposals are verified via `CVCR-IR` inside the identical `35 ms` budget
    (without adding CoRe-Sched's heterogeneous graph attention, queue-sequencing head, or
    Contextual LinUCB coupling switcher).
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
            raise ModuleNotFoundError("PyTorch (`torch`) is required to instantiate DFJSPTHMARLNet.")

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

IJPR25_DFJSPT_CKPT = resolve_default_checkpoint("baselines_2026/ijpr25_dfjspt_hmarl.pt")
GLOBAL_FLAT_DIM = OP_FEAT_DIM + TASK_FEAT_DIM + AGV_FEAT_DIM + BAY_FEAT_DIM + CREW_FEAT_DIM  # 46
JOB_IN_DIM = TASK_FEAT_DIM + OP_FEAT_DIM + GLOBAL_FLAT_DIM  # 70
BAY_IN_DIM = OP_FEAT_DIM + BAY_FEAT_DIM + OP_BAY_EDGE_DIM + GLOBAL_FLAT_DIM  # 70
CREW_IN_DIM = OP_FEAT_DIM + CREW_FEAT_DIM + GLOBAL_FLAT_DIM  # 62
TRANSBOT_IN_DIM = TASK_FEAT_DIM + AGV_FEAT_DIM + TASK_AGV_EDGE_DIM + BAY_FEAT_DIM + 2 + GLOBAL_FLAT_DIM  # 86

_CKPT_CACHE: dict[str, tuple[Any, str]] = {}


def _np_tanh_mlp(x: np.ndarray, w: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    """Evaluate a 2-layer Tanh MLP (`fc1 -> tanh -> fc2 -> tanh -> out`) in float32 NumPy."""
    h = np.tanh(x @ w[f"{prefix}.0.weight"].T + w[f"{prefix}.0.bias"])
    h = np.tanh(h @ w[f"{prefix}.2.weight"].T + w[f"{prefix}.2.bias"])
    return h @ w[f"{prefix}.4.weight"].T + w[f"{prefix}.4.bias"]


def _masked_mean_np(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros((x.shape[-1],), dtype=np.float32)
    return x.mean(axis=0).astype(np.float32)


def _global_flat_summary_np(g: V19HeteroGraph) -> np.ndarray:
    return np.concatenate(
        [
            _masked_mean_np(g.op_feats),
            _masked_mean_np(g.task_feats),
            _masked_mean_np(g.agv_feats),
            _masked_mean_np(g.bay_feats),
            _masked_mean_np(g.crew_feats),
        ],
        axis=0,
    ).astype(np.float32)


def _masked_mean_torch(x: "torch.Tensor", valid_mask: Optional["torch.Tensor"] = None) -> "torch.Tensor":
    if x.shape[1] == 0:
        return torch.zeros((x.shape[0], x.shape[2]), dtype=x.dtype, device=x.device)
    if valid_mask is None:
        return x.mean(dim=1)
    m = valid_mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / torch.clamp(m.sum(dim=1), min=1.0)


def _make_fc_mlp(in_dim: int, hidden_dim: int, out_dim: int = 1) -> "nn.Sequential":
    """Build TorchFC `[hidden_dim, hidden_dim]` Tanh MLP matching DFJSPT_code."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, out_dim),
    )


class DFJSPTHMARLNet(nn.Module):
    """Three-level independent action-masked MLP MARL network (Wang et al., IJPR 2025)."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        # Level 1: High-Level Job/Task Prioritization Agent
        self.job_actor = _make_fc_mlp(JOB_IN_DIM, self.hidden_dim, 1)
        self.job_critic = _make_fc_mlp(GLOBAL_FLAT_DIM, self.hidden_dim, 1)
        # Level 2: Mid-Level Machine/Bay & Crew Assignment Agent
        self.bay_actor = _make_fc_mlp(BAY_IN_DIM, self.hidden_dim, 1)
        self.crew_actor = _make_fc_mlp(CREW_IN_DIM, self.hidden_dim, 1)
        self.machine_critic = _make_fc_mlp(GLOBAL_FLAT_DIM, self.hidden_dim, 1)
        # Level 3: Low-Level Transbot (AMR) Allocation Agent
        self.transbot_actor = _make_fc_mlp(TRANSBOT_IN_DIM, self.hidden_dim, 1)
        self.transbot_critic = _make_fc_mlp(GLOBAL_FLAT_DIM, self.hidden_dim, 1)

    def config_dict(self) -> dict[str, Any]:
        return {"hidden_dim": self.hidden_dim, "arch": "IJPR25_DFJSPT_HMARL"}

    def global_summary(self, tensors: dict[str, "torch.Tensor"]) -> "torch.Tensor":
        return torch.cat(
            [
                _masked_mean_torch(tensors["op_x"], tensors.get("op_valid")),
                _masked_mean_torch(tensors["task_x"], tensors.get("task_valid")),
                _masked_mean_torch(tensors["agv_x"], tensors.get("agv_valid")),
                _masked_mean_torch(tensors["bay_x"], tensors.get("bay_valid")),
                _masked_mean_torch(tensors["crew_x"], tensors.get("crew_valid")),
            ],
            dim=-1,
        )

    def forward_job_agent(
        self,
        tensors: dict[str, "torch.Tensor"],
        g_flat: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return `(job_priority_logits [B, Nt], job_value [B])`."""
        task_x = tensors["task_x"]
        op_x = tensors["op_x"]
        B, Nt, _ = task_x.shape
        if Nt == 0:
            z = torch.zeros((B, 0), dtype=task_x.dtype, device=task_x.device)
            return z, self.job_critic(g_flat).squeeze(-1)
        t2o = tensors["task_to_op"].clamp(min=0, max=max(0, op_x.shape[1] - 1))
        if op_x.shape[1] > 0:
            gather_idx = t2o.unsqueeze(-1).expand(-1, -1, op_x.shape[-1])
            parent_op = torch.gather(op_x, 1, gather_idx)
            valid_parent = (tensors["task_to_op"] >= 0).unsqueeze(-1).to(task_x.dtype)
            parent_op = parent_op * valid_parent
        else:
            parent_op = torch.zeros((B, Nt, OP_FEAT_DIM), dtype=task_x.dtype, device=task_x.device)
        g_exp = g_flat.unsqueeze(1).expand(-1, Nt, -1)
        x_in = torch.cat([task_x, parent_op, g_exp], dim=-1)
        logits = self.job_actor(x_in).squeeze(-1)
        if "task_valid" in tensors:
            mask = tensors["task_valid"].to(logits.dtype)
            logits = logits + torch.clamp(torch.log(torch.clamp(mask, min=0.0) + 1e-12), min=-1e9)
        return logits, self.job_critic(g_flat).squeeze(-1)

    def forward_machine_agent(
        self,
        tensors: dict[str, "torch.Tensor"],
        g_flat: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Return `(bay_logits [B, No, Nb], crew_logits [B, No, Nc], machine_value [B])`."""
        op_x = tensors["op_x"]
        bay_x = tensors["bay_x"]
        crew_x = tensors["crew_x"]
        op_bay_x = tensors["op_bay_x"]
        B, No, _ = op_x.shape
        Nb = bay_x.shape[1]
        Nc = crew_x.shape[1]
        if No == 0 or Nb == 0:
            bay_logits = torch.zeros((B, No, Nb), dtype=op_x.dtype, device=op_x.device)
        else:
            g_b = g_flat.unsqueeze(1).unsqueeze(2).expand(-1, No, Nb, -1)
            op_b = op_x.unsqueeze(2).expand(-1, -1, Nb, -1)
            bay_b = bay_x.unsqueeze(1).expand(-1, No, -1, -1)
            bay_in = torch.cat([op_b, bay_b, op_bay_x, g_b], dim=-1)
            bay_logits = self.bay_actor(bay_in).squeeze(-1)
            bay_mask = tensors["op_bay_mask"].to(bay_logits.dtype)
            bay_logits = bay_logits + torch.clamp(torch.log(torch.clamp(bay_mask, min=0.0) + 1e-12), min=-1e9)

        if No == 0 or Nc == 0:
            crew_logits = torch.zeros((B, No, Nc), dtype=op_x.dtype, device=op_x.device)
        else:
            g_c = g_flat.unsqueeze(1).unsqueeze(2).expand(-1, No, Nc, -1)
            op_c = op_x.unsqueeze(2).expand(-1, -1, Nc, -1)
            crew_c = crew_x.unsqueeze(1).expand(-1, No, -1, -1)
            crew_in = torch.cat([op_c, crew_c, g_c], dim=-1)
            crew_logits = self.crew_actor(crew_in).squeeze(-1)
            crew_mask = tensors["op_crew_mask"].to(crew_logits.dtype)
            crew_logits = crew_logits + torch.clamp(torch.log(torch.clamp(crew_mask, min=0.0) + 1e-12), min=-1e9)

        return bay_logits, crew_logits, self.machine_critic(g_flat).squeeze(-1)

    def forward_transbot_step(
        self,
        task_feat: "torch.Tensor",       # [B, 12]
        agv_x: "torch.Tensor",           # [B, Na, 10]
        task_agv_feat: "torch.Tensor",   # [B, Na, 8]
        chosen_bay_feat: "torch.Tensor", # [B, 8]
        dyn_load: "torch.Tensor",        # [B, Na, 2]
        g_flat: "torch.Tensor",          # [B, 46]
        agv_mask: "torch.Tensor",        # [B, Na]
    ) -> "torch.Tensor":
        """Compute masked AMR selection logits `[B, Na]` for one task given chosen bay & running load."""
        Na = agv_x.shape[1]
        tf = task_feat.unsqueeze(1).expand(-1, Na, -1)
        bf = chosen_bay_feat.unsqueeze(1).expand(-1, Na, -1)
        gf = g_flat.unsqueeze(1).expand(-1, Na, -1)
        x_in = torch.cat([tf, agv_x, task_agv_feat, bf, dyn_load, gf], dim=-1)
        logits = self.transbot_actor(x_in).squeeze(-1)
        mask = agv_mask.to(logits.dtype)
        return logits + torch.clamp(torch.log(torch.clamp(mask, min=0.0) + 1e-12), min=-1e9)

    def forward_teacher_forced(
        self,
        tensors: dict[str, "torch.Tensor"],
        target_bays: "torch.Tensor",
        target_agvs: "torch.Tensor",
        target_focus: "torch.Tensor",
    ) -> dict[str, "torch.Tensor"]:
        """Vectorized forward pass over all three hierarchical agents for IL & PPO training."""
        g_flat = self.global_summary(tensors)
        job_logits, v_job = self.forward_job_agent(tensors, g_flat)
        bay_logits, crew_logits, v_mach = self.forward_machine_agent(tensors, g_flat)
        v_trans = self.transbot_critic(g_flat).squeeze(-1)

        task_x = tensors["task_x"]
        agv_x = tensors["agv_x"]
        bay_x = tensors["bay_x"]
        task_agv_x = tensors["task_agv_x"]
        task_agv_mask = tensors["task_agv_mask"]
        task_to_op = tensors["task_to_op"]
        task_inc_bay = tensors["task_inc_bay"]
        B, Nt, _ = task_x.shape
        Na = agv_x.shape[1]
        Nb = bay_x.shape[1]

        agv_logits_out = torch.zeros((B, Nt, Na), dtype=task_x.dtype, device=task_x.device)
        if Nt > 0 and Na > 0:
            # Vectorized resolution of selected bay feature per task (zero GPU-CPU syncs)
            sel_bay_feat = torch.zeros((B, Nt, BAY_FEAT_DIM), dtype=task_x.dtype, device=task_x.device)
            if Nb > 0:
                No_tb = target_bays.shape[1]
                if No_tb > 0:
                    has_op = (task_to_op >= 0) & (task_to_op < No_tb)
                    safe_oi = task_to_op.clamp(min=0, max=No_tb - 1)
                    op_tb = torch.gather(target_bays, 1, safe_oi)
                    use_op_tb = has_op & (op_tb >= 0) & (op_tb < Nb)
                    bi_t = torch.where(use_op_tb, op_tb, task_inc_bay)
                else:
                    bi_t = task_inc_bay
                valid_bi = (bi_t >= 0) & (bi_t < Nb)
                safe_bi = bi_t.clamp(min=0, max=Nb - 1)
                sel_bay_feat = torch.gather(bay_x, 1, safe_bi.unsqueeze(-1).expand(-1, -1, BAY_FEAT_DIM))
                sel_bay_feat = sel_bay_feat * valid_bi.unsqueeze(-1).to(task_x.dtype)

            # Sequential Level-3 allocation along Level-1 order (matching greedy/target focus order)
            order = torch.argsort(target_focus, dim=1, descending=True)
            dyn_load = torch.zeros((B, Na, 2), dtype=task_x.dtype, device=task_x.device)
            b_idx = torch.arange(B, device=task_x.device)
            for step in range(Nt):
                tj = order[:, step]
                tf = task_x[b_idx, tj]
                tax = task_agv_x[b_idx, tj]
                tam = task_agv_mask[b_idx, tj]
                bf = sel_bay_feat[b_idx, tj]
                lg = self.forward_transbot_step(tf, agv_x, tax, bf, dyn_load, g_flat, tam)
                agv_logits_out[b_idx, tj] = lg
                chosen_a = target_agvs[b_idx, tj]
                valid_a = (chosen_a >= 0) & (chosen_a < Na)
                if "task_valid" in tensors:
                    valid_a = valid_a & tensors["task_valid"][b_idx, tj].bool()
                if valid_a.any():
                    dyn_load = dyn_load.clone()
                    va_b = b_idx[valid_a]
                    va_a = chosen_a[valid_a]
                    va_t = tj[valid_a]
                    dyn_load[va_b, va_a, 0] = dyn_load[va_b, va_a, 0] + 0.25
                    dyn_load[va_b, va_a, 1] = dyn_load[va_b, va_a, 1] + task_agv_x[va_b, va_t, va_a, 0]

        value = (v_job + v_mach + v_trans) / 3.0
        return {
            "job_logits": job_logits,
            "bay_logits": bay_logits,
            "crew_logits": crew_logits,
            "agv_logits": agv_logits_out,
            "value": value,
            "v_job": v_job,
            "v_mach": v_mach,
            "v_trans": v_trans,
        }


class NumpyDFJSPTHMARLNet:
    """Zero-PyTorch pure-NumPy inference replica of `DFJSPTHMARLNet` (`< 1e-5` max abs parity)."""

    def __init__(self, weights: dict[str, np.ndarray], config: dict[str, Any]) -> None:
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        self.config = dict(config)

    def predict_job_logits(self, g: V19HeteroGraph, g_flat: np.ndarray) -> np.ndarray:
        Nt = len(g.rel_tids)
        if Nt == 0:
            return np.zeros((0,), dtype=np.float32)
        No = len(g.op_ids)
        parent_op = np.zeros((Nt, OP_FEAT_DIM), dtype=np.float32)
        for j in range(Nt):
            oi = int(g.task_to_op[j])
            if 0 <= oi < No:
                parent_op[j] = g.op_feats[oi]
        g_exp = np.broadcast_to(g_flat[None, :], (Nt, GLOBAL_FLAT_DIM))
        x_in = np.concatenate([g.task_feats, parent_op, g_exp], axis=-1).astype(np.float32)
        return _np_tanh_mlp(x_in, self.w, "job_actor").squeeze(-1)

    def predict_machine_logits(self, g: V19HeteroGraph, g_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        No, Nb, Nc = len(g.op_ids), len(g.bay_ids), len(g.crew_ids)
        if No == 0 or Nb == 0:
            bay_lg = np.zeros((No, Nb), dtype=np.float32)
        else:
            op_b = np.broadcast_to(g.op_feats[:, None, :], (No, Nb, OP_FEAT_DIM))
            bay_b = np.broadcast_to(g.bay_feats[None, :, :], (No, Nb, BAY_FEAT_DIM))
            g_b = np.broadcast_to(g_flat[None, None, :], (No, Nb, GLOBAL_FLAT_DIM))
            x_b = np.concatenate([op_b, bay_b, g.op_bay_feats, g_b], axis=-1).reshape(No * Nb, BAY_IN_DIM)
            bay_lg = _np_tanh_mlp(x_b, self.w, "bay_actor").reshape(No, Nb)
            bay_lg = bay_lg + np.clip(np.log(np.clip(g.op_bay_mask.astype(np.float32), 0.0, 1.0) + 1e-12), -1e9, 0.0)

        if No == 0 or Nc == 0:
            crew_lg = np.zeros((No, Nc), dtype=np.float32)
        else:
            op_c = np.broadcast_to(g.op_feats[:, None, :], (No, Nc, OP_FEAT_DIM))
            crew_c = np.broadcast_to(g.crew_feats[None, :, :], (No, Nc, CREW_FEAT_DIM))
            g_c = np.broadcast_to(g_flat[None, None, :], (No, Nc, GLOBAL_FLAT_DIM))
            x_c = np.concatenate([op_c, crew_c, g_c], axis=-1).reshape(No * Nc, CREW_IN_DIM)
            crew_lg = _np_tanh_mlp(x_c, self.w, "crew_actor").reshape(No, Nc)
            crew_lg = crew_lg + np.clip(np.log(np.clip(g.op_crew_mask.astype(np.float32), 0.0, 1.0) + 1e-12), -1e9, 0.0)
        return bay_lg.astype(np.float32), crew_lg.astype(np.float32)

    def predict_transbot_step(
        self,
        g: V19HeteroGraph,
        j: int,
        chosen_bay_feat: np.ndarray,
        dyn_load: np.ndarray,
        g_flat: np.ndarray,
    ) -> np.ndarray:
        Na = len(g.agv_ids)
        tf = np.broadcast_to(g.task_feats[j : j + 1, :], (Na, TASK_FEAT_DIM))
        bf = np.broadcast_to(chosen_bay_feat[None, :], (Na, BAY_FEAT_DIM))
        gf = np.broadcast_to(g_flat[None, :], (Na, GLOBAL_FLAT_DIM))
        x_in = np.concatenate([tf, g.agv_feats, g.task_agv_feats[j], bf, dyn_load, gf], axis=-1).astype(np.float32)
        lg = _np_tanh_mlp(x_in, self.w, "transbot_actor").squeeze(-1)
        mask = g.task_agv_mask[j].astype(np.float32)
        return (lg + np.clip(np.log(np.clip(mask, 0.0, 1.0) + 1e-12), -1e9, 0.0)).astype(np.float32)


def export_dfjspt_npz(pt_path: Path | str, npz_path: Path | str) -> Path:
    """Export PyTorch `DFJSPTHMARLNet` checkpoint to exact float32 `.npz` pure-NumPy archive."""
    if not _HAS_TORCH:
        raise ModuleNotFoundError("PyTorch is required to export .pt to .npz.")
    payload = torch.load(Path(pt_path), map_location="cpu", weights_only=False)
    sd = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    cfg = payload.get("config", {"hidden_dim": 256}) if isinstance(payload, dict) else {"hidden_dim": 256}
    arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in sd.items()}
    arrays["__config_hidden_dim__"] = np.asarray([int(cfg.get("hidden_dim", 256))], dtype=np.int32)
    dst = Path(npz_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **arrays)
    return dst


def load_dfjspt_model(
    ckpt_path: Path | str = IJPR25_DFJSPT_CKPT,
    *,
    backend: str = "auto",
    allow_untrained: bool = False,
) -> tuple[Any, str, Optional[str]]:
    """Load trained `IJPR25_DFJSPT_HMARL` checkpoint in NumPy (default) or PyTorch backend."""
    p = Path(ckpt_path)
    npz_p = resolve_numpy_checkpoint_path(p, prefer_npz=True)
    use_np = (backend == "numpy") or (backend == "auto" and (npz_p.is_file() or not _HAS_TORCH))

    if use_np and npz_p.is_file():
        sha = file_sha256(npz_p)
        key = f"np:{npz_p.resolve()}:{sha}"
        if key not in _CKPT_CACHE:
            with np.load(npz_p, allow_pickle=False) as z:
                arrs = {k: z[k] for k in z.files}
            hdim = int(arrs.pop("__config_hidden_dim__", np.asarray([256]))[0])
            _CKPT_CACHE[key] = (NumpyDFJSPTHMARLNet(arrs, {"hidden_dim": hdim}), sha)
        return _CKPT_CACHE[key][0], "numpy", _CKPT_CACHE[key][1]

    if _HAS_TORCH and p.is_file():
        sha = file_sha256(p)
        key = f"pt:{p.resolve()}:{sha}"
        if key not in _CKPT_CACHE:
            payload = torch.load(p, map_location="cpu", weights_only=False)
            cfg = payload.get("config", {"hidden_dim": 256}) if isinstance(payload, dict) else {"hidden_dim": 256}
            net = DFJSPTHMARLNet(hidden_dim=int(cfg.get("hidden_dim", 256)))
            net.load_state_dict(payload["model"] if "model" in payload else payload, strict=True)
            net.eval()
            _CKPT_CACHE[key] = (net, sha)
        return _CKPT_CACHE[key][0], "torch", _CKPT_CACHE[key][1]

    if not allow_untrained:
        raise FileNotFoundError(f"Required IJPR25_DFJSPT_HMARL checkpoint not found: {p} / {npz_p}")
    if _HAS_TORCH:
        net = DFJSPTHMARLNet(hidden_dim=256)
        net.eval()
        return net, "torch", None
    raise ModuleNotFoundError("PyTorch required to instantiate untrained DFJSPTHMARLNet.")


@dataclass
class DecodedDFJSPTCandidate:
    assign: dict[str, str]
    bays: dict[str, str]
    crews: dict[str, int]
    op_prio: dict[str, int]
    chosen_bay_idx: list[int]
    chosen_crew_idx: list[int]
    chosen_agv_idx: list[int]
    task_order: list[int]


def _sample_categorical_np(logits: np.ndarray, valid_mask: np.ndarray, *, greedy: bool, tau: float, rng: np.random.Generator) -> int:
    valid_idxs = np.flatnonzero(valid_mask > 0.5)
    if len(valid_idxs) == 0:
        return 0
    if greedy or len(valid_idxs) == 1:
        return int(valid_idxs[int(np.argmax(logits[valid_idxs]))])
    probs = _np_softmax(logits[valid_idxs] / max(0.05, float(tau)))
    return int(rng.choice(valid_idxs, p=probs))


def decode_dfjspt_candidate(
    model: Any,
    g: V19HeteroGraph,
    snapshot: NativeSnapshot,
    *,
    greedy: bool = True,
    tau: float = 0.45,
    rng: Optional[np.random.Generator] = None,
) -> DecodedDFJSPTCandidate:
    """Run hierarchical 3-agent decoding: Level-1 Job -> Level-2 Machine -> Level-3 Transbot."""
    use_rng = rng if rng is not None else np.random.default_rng(0)
    No, Nt, Na, Nb, Nc = len(g.op_ids), len(g.rel_tids), len(g.agv_ids), len(g.bay_ids), len(g.crew_ids)

    if isinstance(model, NumpyDFJSPTHMARLNet):
        g_flat = _global_flat_summary_np(g)
        job_lg = model.predict_job_logits(g, g_flat)
        bay_lg, crew_lg = model.predict_machine_logits(g, g_flat)
    else:
        from amr_sched.rl.v19_hetero_constructor import graph_to_torch_tensors

        dev = next(model.parameters()).device
        with torch.no_grad():
            ten = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in graph_to_torch_tensors(g).items()}
            g_flat_t = model.global_summary(ten)
            job_lg_t, _ = model.forward_job_agent(ten, g_flat_t)
            bay_lg_t, crew_lg_t, _ = model.forward_machine_agent(ten, g_flat_t)
            g_flat = g_flat_t[0].cpu().numpy()
            job_lg = job_lg_t[0].cpu().numpy()
            bay_lg = bay_lg_t[0].cpu().numpy()
            crew_lg = crew_lg_t[0].cpu().numpy()

    # Level 1 (JobAgent): prioritize releasable tasks
    if Nt == 0:
        order = []
    elif greedy:
        order = [int(i) for i in np.argsort(-job_lg, kind="mergesort")]
    else:
        rem = list(range(Nt))
        order = []
        while rem:
            sub = np.array([job_lg[i] for i in rem], dtype=np.float32)
            probs = _np_softmax(sub / max(0.05, float(tau)))
            pick_pos = int(use_rng.choice(len(rem), p=probs))
            order.append(rem.pop(pick_pos))

    # Level 2 (MachineAgent): assign eligible (bay, crew) for each operation respecting LOCKED mask
    bays = dict(g.base_bays)
    crews = dict(g.base_crews)
    chosen_bay_idx = [-1] * No
    chosen_crew_idx = [-1] * No
    for oi, oid in enumerate(g.op_ids):
        if Nb > 0:
            bi = _sample_categorical_np(bay_lg[oi], g.op_bay_mask[oi], greedy=greedy, tau=tau, rng=use_rng)
            chosen_bay_idx[oi] = bi
            bays[oid] = g.bay_ids[bi]
        if Nc > 0:
            ci = _sample_categorical_np(crew_lg[oi], g.op_crew_mask[oi], greedy=greedy, tau=tau, rng=use_rng)
            chosen_crew_idx[oi] = ci
            crews[oid] = int(g.crew_ids[ci])

    # Level 3 (TransbotAgent): assign each releasable kitting task in Level-1 priority order
    assign = dict(g.base_assign)
    chosen_agv_idx = [-1] * Nt
    dyn_load = np.zeros((Na, 2), dtype=np.float32)
    for j in order:
        if Na == 0:
            break
        oi = int(g.task_to_op[j])
        bi = chosen_bay_idx[oi] if (0 <= oi < No and chosen_bay_idx[oi] >= 0) else int(g.task_incumbent_bay_idx[j])
        bay_feat = g.bay_feats[bi] if (0 <= bi < Nb) else np.zeros((BAY_FEAT_DIM,), dtype=np.float32)
        if isinstance(model, NumpyDFJSPTHMARLNet):
            agv_lg = model.predict_transbot_step(g, j, bay_feat, dyn_load, g_flat)
        else:
            dev = next(model.parameters()).device
            with torch.no_grad():
                agv_lg = (
                    model.forward_transbot_step(
                        torch.from_numpy(g.task_feats[j : j + 1]).to(dev),
                        torch.from_numpy(g.agv_feats[None, :, :]).to(dev),
                        torch.from_numpy(g.task_agv_feats[j : j + 1]).to(dev),
                        torch.from_numpy(bay_feat[None, :]).to(dev),
                        torch.from_numpy(dyn_load[None, :, :]).to(dev),
                        torch.from_numpy(g_flat[None, :]).to(dev),
                        torch.from_numpy(g.task_agv_mask[j : j + 1]).to(dev),
                    )[0]
                    .cpu()
                    .numpy()
                )
        ai = _sample_categorical_np(agv_lg, g.task_agv_mask[j], greedy=greedy, tau=tau, rng=use_rng)
        chosen_agv_idx[j] = ai
        assign[g.rel_tids[j]] = g.agv_ids[ai]
        dyn_load[ai, 0] += 0.25
        dyn_load[ai, 1] += float(g.task_agv_feats[j, ai, 0])

    # Standard earliest-due operation priority (no A4 queue-sequencing head, faithful to DFJSP-T)
    op_prio = {op.op_id: max(int(snapshot.tick), int(op.due_step) - int(op.proc_steps)) for op in snapshot.operations}
    return DecodedDFJSPTCandidate(
        assign=assign,
        bays=bays,
        crews=crews,
        op_prio=op_prio,
        chosen_bay_idx=chosen_bay_idx,
        chosen_crew_idx=chosen_crew_idx,
        chosen_agv_idx=chosen_agv_idx,
        task_order=order,
    )


class DFJSPTHMARLProposer:
    """Online proposer wrapper for Wang et al. (IJPR 2025) 3-Agent Hierarchical MARL."""

    def __init__(
        self,
        ckpt_path: Path | str = IJPR25_DFJSPT_CKPT,
        *,
        policy_id: str = "IJPR25_DFJSPT_HMARL",
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
            m, b, sha = load_dfjspt_model(
                self.ckpt_path, backend=self.backend, allow_untrained=self.allow_untrained
            )
            self._model = m
            self._eff_backend = b
            self._checkpoint_sha256 = sha
        return self._model

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
                cand = decode_dfjspt_candidate(model, graph, snapshot, greedy=(k == 0), tau=0.45, rng=rng)
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
        prop.telemetry["reproduction_tier"] = "paper_faithful_reproduction"
        return prop
