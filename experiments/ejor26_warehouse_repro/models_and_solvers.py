"""Solvers for Track A (Wu et al., EJOR 2026 3D Warehouse VRP Independent Reproduction).

Implements 5 solvers operating on identical `WarehouseVRPInstance` inputs:
  1. `WuGRUAttentionVRP`: Conv1d + 1-layer GRU pointer-attention actor + 3-layer Conv1d critic (REINFORCE).
  2. `AttentionModelVRP`: Transformer encoder + glimpse pointer decoder (REINFORCE with greedy rollout baseline).
  3. `PPOVRPSolver`: Clipped PPO + GAE on Conv1d-GRU pointer-attention actor-critic.
  4. `DiscreteSACVRPSolver`: Twin-Q Discrete Soft Actor-Critic with replay buffer & target entropy.
  5. `GeneticAlgorithmVRP`: Order Crossover (OX) + 2-opt mutation with capacity-feasible split decoding.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.ejor26_warehouse_repro.storage_policies import assign_storage_slots
from experiments.ejor26_warehouse_repro.warehouse_3d_env import (
    Warehouse3DEnv,
    WarehouseVRPInstance,
)


def _batch_from_instances(
    instances: list[WarehouseVRPInstance],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Pack a uniform-size list of `WarehouseVRPInstance` into batched GPU/CPU tensors."""
    B = len(instances)
    M = instances[0].coords_3d.shape[0]
    coords = np.stack([inst.coords_3d for inst in instances], axis=0)  # [B, M, 3]
    demands = np.stack([inst.demands for inst in instances], axis=0)   # [B, M]
    prios = np.stack([inst.priorities for inst in instances], axis=0)  # [B, M]
    dist = np.stack([inst.dist_matrix for inst in instances], axis=0)  # [B, M, M]
    caps = np.array([inst.capacity for inst in instances], dtype=np.float32)  # [B]

    coord_scale = np.maximum(1.0, coords.max(axis=(0, 1), keepdims=True))
    d_scale = np.maximum(1.0, dist.max(axis=(1, 2), keepdims=True))
    coords_n = coords / coord_scale
    d_depot = dist[:, 0:1, :].transpose(0, 2, 1) / d_scale  # [B, M, 1]
    dem_n = (demands / np.maximum(1.0, caps[:, None]))[..., None]  # [B, M, 1]
    prio_n = prios[..., None]  # [B, M, 1]
    static_x = np.concatenate([coords_n, dem_n, prio_n, d_depot], axis=-1).astype(np.float32)  # [B, M, 6]

    return {
        "static_x": torch.from_numpy(static_x).to(device),         # [B, M, 6]
        "coords": torch.from_numpy(coords).to(device),             # [B, M, 3]
        "demands": torch.from_numpy(demands).to(device),           # [B, M]
        "dist": torch.from_numpy(dist).to(device),                 # [B, M, M]
        "d_scale": torch.from_numpy(d_scale[:, 0, 0]).to(device),  # [B]
        "cap": torch.from_numpy(caps).to(device),                  # [B]
        "M": torch.tensor(M, device=device),
    }


def _build_dynamic_and_mask(
    batch: dict[str, torch.Tensor],
    cur_idx: torch.Tensor,
    rem_cap: torch.Tensor,
    unvisited: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `dyn_x [B, M, 3]` and boolean feasible mask `[B, M]`."""
    B, M = unvisited.shape
    dist = batch["dist"]
    d_scale = batch["d_scale"].view(B, 1)
    cap = batch["cap"].view(B, 1)
    dem = batch["demands"]

    cur_d = dist.gather(1, cur_idx.view(B, 1, 1).expand(B, 1, M)).squeeze(1) / d_scale
    rem_norm = (rem_cap.view(B, 1) / torch.clamp(cap, min=1.0)).expand(B, M)
    vis_flag = (~unvisited).float()
    vis_flag[:, 0] = (cur_idx == 0).float()
    dyn_x = torch.stack([rem_norm, cur_d, vis_flag], dim=-1)  # [B, M, 3]

    mask = torch.zeros((B, M), dtype=torch.bool, device=unvisited.device)
    cust_ok = unvisited[:, 1:] & (dem[:, 1:] <= rem_cap.view(B, 1) + 1e-5)
    mask[:, 1:] = cust_ok
    has_unvisited = unvisited[:, 1:].any(dim=-1)
    finished = ~has_unvisited
    # Allow depot return whenever away from depot OR finished (as no-op completion anchor).
    mask[:, 0] = (cur_idx != 0) | finished
    return dyn_x, mask


def _rollout_step_update(
    batch: dict[str, torch.Tensor],
    action: torch.Tensor,
    cur_idx: torch.Tensor,
    rem_cap: torch.Tensor,
    unvisited: torch.Tensor,
    finished: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance VRP batch one step and return `(next_cur, next_rem_cap, next_unvisited, step_cost, next_finished)`."""
    B = cur_idx.shape[0]
    step_d = batch["dist"][torch.arange(B, device=cur_idx.device), cur_idx, action]
    step_cost = torch.where(finished, torch.zeros_like(step_d), step_d)

    next_cur = torch.where(finished, cur_idx, action)
    is_depot = action == 0
    dem_taken = batch["demands"].gather(1, action.view(B, 1)).squeeze(1)
    next_rem = torch.where(
        finished,
        rem_cap,
        torch.where(is_depot, batch["cap"], torch.clamp(rem_cap - dem_taken, min=0.0)),
    )
    next_unvisited = unvisited.clone()
    active_cust = (~finished) & (~is_depot)
    if active_cust.any():
        idx_a = torch.arange(B, device=cur_idx.device)[active_cust]
        next_unvisited[idx_a, action[active_cust]] = False
    next_finished = ~next_unvisited[:, 1:].any(dim=-1)
    return next_cur, next_rem, next_unvisited, step_cost, next_finished


class WuGRUAttentionNet(nn.Module):
    """Wu et al. (EJOR 2026) Conv1d + 1-Layer GRU Pointer Actor + 3-Layer Conv1d Critic."""

    def __init__(self, static_dim: int = 6, dynamic_dim: int = 3, hidden_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.static_dim = int(static_dim)
        self.hidden_dim = int(hidden_dim)
        self.static_enc = nn.Conv1d(static_dim, hidden_dim, kernel_size=1)
        self.dynamic_enc = nn.Conv1d(dynamic_dim, hidden_dim, kernel_size=1)
        self.dec_enc = nn.Conv1d(static_dim, hidden_dim, kernel_size=1)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(p=dropout)
        self.w_attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v_attn = nn.Linear(hidden_dim, 1, bias=False)
        self.w_ptr = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v_ptr = nn.Linear(hidden_dim, 1, bias=False)
        self.critic_c1 = nn.Conv1d(hidden_dim * 2, 20, kernel_size=1)
        self.critic_c2 = nn.Conv1d(20, 20, kernel_size=1)
        self.critic_c3 = nn.Conv1d(20, 1, kernel_size=1)

    def encode_static(self, static_x: torch.Tensor) -> torch.Tensor:
        """static_x: [B, M, S] -> hs: [B, H, M]."""
        return self.static_enc(static_x.transpose(1, 2))

    def estimate_value(self, static_x: torch.Tensor, dynamic_x: torch.Tensor) -> torch.Tensor:
        hs = self.encode_static(static_x)
        hd = self.dynamic_enc(dynamic_x.transpose(1, 2))
        x = F.relu(self.critic_c1(torch.cat([hs, hd], dim=1)))
        x = F.relu(self.critic_c2(x))
        return self.critic_c3(x).sum(dim=(1, 2))

    def step_pointer(
        self,
        hs: torch.Tensor,
        dynamic_x: torch.Tensor,
        dec_in: torch.Tensor,
        h_prev: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, M = hs.shape
        hd = self.dynamic_enc(dynamic_x.transpose(1, 2))  # [B, H, M]
        h_dec = self.dec_enc(dec_in.unsqueeze(-1)).squeeze(-1)  # [B, H]
        h_t = self.gru(h_dec, h_prev)
        h_exp = h_t.unsqueeze(-1).expand(B, H, M)
        cat_e = torch.cat([hs, hd, h_exp], dim=1).transpose(1, 2)  # [B, M, 3H]
        e = self.v_attn(self.drop(torch.tanh(self.w_attn(cat_e)))).squeeze(-1)
        e = e.masked_fill(~mask, -1e9)
        attns = F.softmax(e, dim=-1).unsqueeze(1)  # [B, 1, M]
        ctx = torch.bmm(attns, hs.transpose(1, 2)).expand(B, M, H)
        cat_p = torch.cat([hs.transpose(1, 2), hd.transpose(1, 2), ctx], dim=-1)
        logits = self.v_ptr(self.drop(torch.tanh(self.w_ptr(cat_p)))).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        return logits, h_t


class AttentionModelNet(nn.Module):
    """Kool et al. Transformer Encoder + Context Glimpse Pointer Net adapted to 3D VRP."""

    def __init__(self, static_dim: int = 6, hidden_dim: int = 96, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.node_emb = nn.Linear(static_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ctx_proj = nn.Linear(hidden_dim * 2 + 2, hidden_dim)
        self.glimpse_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.glimpse_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.glimpse_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ptr_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ptr_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dist_scale = nn.Parameter(torch.tensor(1.0))

    def encode(self, static_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(self.node_emb(static_x))  # [B, M, H]
        return h, h.mean(dim=1)                    # [B, M, H], [B, H]

    def step_logits(
        self,
        h_nodes: torch.Tensor,
        h_graph: torch.Tensor,
        cur_idx: torch.Tensor,
        rem_cap_norm: torch.Tensor,
        step_frac: torch.Tensor,
        cur_d_norm: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, M, H = h_nodes.shape
        h_cur = h_nodes.gather(1, cur_idx.view(B, 1, 1).expand(B, 1, H)).squeeze(1)
        q_raw = torch.cat([h_graph, h_cur, rem_cap_norm.view(B, 1), step_frac.view(B, 1)], dim=-1)
        q = self.ctx_proj(q_raw).unsqueeze(1)  # [B, 1, H]

        att_logits = torch.bmm(self.glimpse_q(q), self.glimpse_k(h_nodes).transpose(1, 2)).squeeze(1) / math.sqrt(H)
        att_logits = att_logits - F.softplus(self.dist_scale) * cur_d_norm
        att_logits = att_logits.masked_fill(~mask, -1e9)
        att_w = F.softmax(att_logits, dim=-1).unsqueeze(1)
        ctx = torch.bmm(att_w, self.glimpse_v(h_nodes))  # [B, 1, H]

        logits = torch.bmm(self.ptr_q(ctx), self.ptr_k(h_nodes).transpose(1, 2)).squeeze(1) / math.sqrt(H)
        logits = logits - F.softplus(self.dist_scale) * cur_d_norm
        logits = 10.0 * torch.tanh(logits)
        return logits.masked_fill(~mask, -1e9)


class DiscreteSACQNet(nn.Module):
    """Dual Q-network over dynamic 3D VRP node embeddings `[B, M, 9]` -> `[B, M]` Q-values."""

    def __init__(self, in_dim: int = 9, hidden_dim: int = 128) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.enc(state_x)  # [B, M, H]
        g = h.mean(dim=1, keepdim=True).expand_as(h)
        q = self.q_head(torch.cat([h, g], dim=-1)).squeeze(-1)
        return q.masked_fill(~mask, -1e6)


def _decode_and_score_routes(
    instances: list[WarehouseVRPInstance],
    routes: list[list[int]],
) -> tuple[list[float], list[int]]:
    costs: list[float] = []
    reloads: list[int] = []
    for inst, rt in zip(instances, routes):
        res = Warehouse3DEnv.evaluate_tour(inst, rt)
        costs.append(float(res["total_time"]))
        reloads.append(int(res["reloads"]))
    return costs, reloads


def _sample_training_batch(
    env: Warehouse3DEnv,
    rng: random.Random,
    batch_size: int,
    n_stops: int,
) -> list[WarehouseVRPInstance]:
    policies = ("random", "within_aisle", "class_abc", "semi_circle")
    cat_seed = rng.randint(1, 100000)
    catalog = env.build_sku_catalog(cat_seed)
    smaps = {pol: assign_storage_slots(catalog, env, pol, seed=cat_seed) for pol in policies}
    out: list[WarehouseVRPInstance] = []
    for i in range(batch_size):
        pol = policies[i % len(policies)]
        out.append(
            env.generate_instance(catalog, smaps[pol], n_stops, rng.randint(1, 10_000_000), pol, f"tr_{i}")
        )
    return out


class WuGRUAttentionVRP:
    """REINFORCE + 3-layer Conv1d Critic solver wrapping `WuGRUAttentionNet`."""

    name = "Wu_GRU_Attn"

    def __init__(self, device: torch.device | None = None, hidden_dim: int = 128) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = WuGRUAttentionNet(hidden_dim=hidden_dim).to(self.device)

    def _rollout(
        self,
        instances: list[WarehouseVRPInstance],
        *,
        greedy: bool,
    ) -> tuple[list[list[int]], torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = _batch_from_instances(instances, self.device)
        B = len(instances)
        M = int(batch["M"].item())
        static_x = batch["static_x"]
        hs = self.net.encode_static(static_x)

        cur_idx = torch.zeros((B,), dtype=torch.long, device=self.device)
        rem_cap = batch["cap"].clone()
        unvisited = torch.ones((B, M), dtype=torch.bool, device=self.device)
        unvisited[:, 0] = False
        finished = torch.zeros((B,), dtype=torch.bool, device=self.device)
        h_t = torch.zeros((B, self.net.hidden_dim), device=self.device)
        dec_in = static_x[:, 0, :]

        dyn0, _ = _build_dynamic_and_mask(batch, cur_idx, rem_cap, unvisited)
        v_pred = self.net.estimate_value(static_x, dyn0)

        routes: list[list[int]] = [[0] for _ in range(B)]
        logp_sum = torch.zeros((B,), device=self.device)
        cost_sum = torch.zeros((B,), device=self.device)

        max_steps = 2 * (M - 1) + 2
        for _ in range(max_steps):
            if finished.all():
                break
            dyn_x, mask = _build_dynamic_and_mask(batch, cur_idx, rem_cap, unvisited)
            logits, h_t = self.net.step_pointer(hs, dyn_x, dec_in, h_t, mask)
            if greedy:
                action = torch.argmax(logits, dim=-1)
                step_logp = torch.zeros((B,), device=self.device)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                step_logp = dist.log_prob(action)

            was_fin = finished.clone()
            cur_idx, rem_cap, unvisited, step_c, finished = _rollout_step_update(
                batch, action, cur_idx, rem_cap, unvisited, finished
            )
            logp_sum = logp_sum + torch.where(was_fin, torch.zeros_like(step_logp), step_logp)
            cost_sum = cost_sum + step_c
            dec_in = static_x.gather(1, cur_idx.view(B, 1, 1).expand(B, 1, static_x.shape[-1])).squeeze(1)

            act_list = action.tolist()
            fin_list = was_fin.tolist()
            for b in range(B):
                if not fin_list[b]:
                    routes[b].append(int(act_list[b]))

        # Close open routes back at depot 0.
        ret_d = batch["dist"][torch.arange(B, device=self.device), cur_idx, torch.zeros_like(cur_idx)]
        need_ret = cur_idx != 0
        cost_sum = cost_sum + torch.where(need_ret, ret_d, torch.zeros_like(ret_d))
        for b, nr in enumerate(need_ret.tolist()):
            if nr or routes[b][-1] != 0:
                routes[b].append(0)
        return routes, cost_sum, logp_sum, v_pred

    def fit(self, env: Warehouse3DEnv, n_stops: int, *, updates: int = 240, batch_size: int = 64, seed: int = 2026) -> float:
        t0 = time.perf_counter()
        rng = random.Random(seed)
        torch.manual_seed(seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=1.5e-3)
        self.net.train()
        for _ in range(updates):
            insts = _sample_training_batch(env, rng, batch_size, n_stops)
            _, cost_sum, logp_sum, v_pred = self._rollout(insts, greedy=False)
            target_scaled = cost_sum.detach() / 50.0
            raw_adv = target_scaled - v_pred.detach()
            adv = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-6)
            actor_loss = (adv * logp_sum).mean()
            critic_loss = F.smooth_l1_loss(v_pred, target_scaled)
            loss = actor_loss + 0.5 * critic_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 2.0)
            opt.step()
        self.net.eval()
        return time.perf_counter() - t0

    def solve(self, inst: WarehouseVRPInstance) -> tuple[list[int], float]:
        self.net.eval()
        with torch.inference_mode():
            routes, _, _, _ = self._rollout([inst], greedy=True)
        res = Warehouse3DEnv.evaluate_tour(inst, routes[0])
        return routes[0], float(res["total_time"])


class AttentionModelVRP:
    """Transformer Attention Model (Kool et al. baseline) trained with greedy rollout baseline."""

    name = "AttentionModel"

    def __init__(self, device: torch.device | None = None, hidden_dim: int = 96) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = AttentionModelNet(hidden_dim=hidden_dim).to(self.device)
        self.baseline_net = AttentionModelNet(hidden_dim=hidden_dim).to(self.device)
        self.baseline_net.load_state_dict(self.net.state_dict())
        self.baseline_net.eval()

    def _rollout_with(
        self,
        model: AttentionModelNet,
        instances: list[WarehouseVRPInstance],
        *,
        greedy: bool,
    ) -> tuple[list[list[int]], torch.Tensor, torch.Tensor]:
        batch = _batch_from_instances(instances, self.device)
        B = len(instances)
        M = int(batch["M"].item())
        static_x = batch["static_x"]
        h_nodes, h_graph = model.encode(static_x)

        cur_idx = torch.zeros((B,), dtype=torch.long, device=self.device)
        rem_cap = batch["cap"].clone()
        unvisited = torch.ones((B, M), dtype=torch.bool, device=self.device)
        unvisited[:, 0] = False
        finished = torch.zeros((B,), dtype=torch.bool, device=self.device)

        routes: list[list[int]] = [[0] for _ in range(B)]
        logp_sum = torch.zeros((B,), device=self.device)
        cost_sum = torch.zeros((B,), device=self.device)
        max_steps = 2 * (M - 1) + 2

        for st in range(max_steps):
            if finished.all():
                break
            _, mask = _build_dynamic_and_mask(batch, cur_idx, rem_cap, unvisited)
            rem_norm = rem_cap / torch.clamp(batch["cap"], min=1.0)
            step_frac = torch.full((B,), float(st) / float(max_steps), device=self.device)
            cur_d_norm = (
                batch["dist"].gather(1, cur_idx.view(B, 1, 1).expand(B, 1, M)).squeeze(1)
                / batch["d_scale"].view(B, 1)
            )
            logits = model.step_logits(h_nodes, h_graph, cur_idx, rem_norm, step_frac, cur_d_norm, mask)
            if greedy:
                action = torch.argmax(logits, dim=-1)
                step_logp = torch.zeros((B,), device=self.device)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                step_logp = dist.log_prob(action)

            was_fin = finished.clone()
            cur_idx, rem_cap, unvisited, step_c, finished = _rollout_step_update(
                batch, action, cur_idx, rem_cap, unvisited, finished
            )
            logp_sum = logp_sum + torch.where(was_fin, torch.zeros_like(step_logp), step_logp)
            cost_sum = cost_sum + step_c
            act_list = action.tolist()
            fin_list = was_fin.tolist()
            for b in range(B):
                if not fin_list[b]:
                    routes[b].append(int(act_list[b]))

        ret_d = batch["dist"][torch.arange(B, device=self.device), cur_idx, torch.zeros_like(cur_idx)]
        need_ret = cur_idx != 0
        cost_sum = cost_sum + torch.where(need_ret, ret_d, torch.zeros_like(ret_d))
        for b, nr in enumerate(need_ret.tolist()):
            if nr or routes[b][-1] != 0:
                routes[b].append(0)
        return routes, cost_sum, logp_sum

    def fit(self, env: Warehouse3DEnv, n_stops: int, *, updates: int = 240, batch_size: int = 64, seed: int = 2027) -> float:
        t0 = time.perf_counter()
        rng = random.Random(seed)
        torch.manual_seed(seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=6e-4)
        self.net.train()
        for ep in range(updates):
            insts = _sample_training_batch(env, rng, batch_size, n_stops)
            _, cost_s, logp_s = self._rollout_with(self.net, insts, greedy=False)
            with torch.no_grad():
                _, cost_b, _ = self._rollout_with(self.baseline_net, insts, greedy=True)
            raw_adv = cost_s.detach() - cost_b.detach()
            adv = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-6)
            loss = (adv * logp_s).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.5)
            opt.step()
            if (ep + 1) % 15 == 0:
                with torch.no_grad():
                    _, cost_g, _ = self._rollout_with(self.net, insts[:16], greedy=True)
                    _, cost_bg, _ = self._rollout_with(self.baseline_net, insts[:16], greedy=True)
                    if cost_g.mean().item() < cost_bg.mean().item():
                        self.baseline_net.load_state_dict(self.net.state_dict())
        self.net.eval()
        return time.perf_counter() - t0

    def solve(self, inst: WarehouseVRPInstance) -> tuple[list[int], float]:
        self.net.eval()
        with torch.inference_mode():
            routes, _, _ = self._rollout_with(self.net, [inst], greedy=True)
        res = Warehouse3DEnv.evaluate_tour(inst, routes[0])
        return routes[0], float(res["total_time"])


class PPOVRPSolver:
    """Clipped PPO + GAE baseline sharing WuGRUAttentionNet pointer-critic architecture."""

    name = "PPO"

    def __init__(self, device: torch.device | None = None, hidden_dim: int = 128) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = WuGRUAttentionNet(hidden_dim=hidden_dim).to(self.device)

    def fit(self, env: Warehouse3DEnv, n_stops: int, *, updates: int = 160, batch_size: int = 48, seed: int = 2028) -> float:
        t0 = time.perf_counter()
        rng = random.Random(seed)
        torch.manual_seed(seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.net.train()
        for _ in range(updates):
            insts = _sample_training_batch(env, rng, batch_size, n_stops)
            batch = _batch_from_instances(insts, self.device)
            B = len(insts)
            M = int(batch["M"].item())
            static_x = batch["static_x"]

            cur_idx = torch.zeros((B,), dtype=torch.long, device=self.device)
            rem_cap = batch["cap"].clone()
            unvisited = torch.ones((B, M), dtype=torch.bool, device=self.device)
            unvisited[:, 0] = False
            finished = torch.zeros((B,), dtype=torch.bool, device=self.device)
            h_t = torch.zeros((B, self.net.hidden_dim), device=self.device)
            dec_in = static_x[:, 0, :]

            dyn_list: list[torch.Tensor] = []
            dec_list: list[torch.Tensor] = []
            hp_list: list[torch.Tensor] = []
            msk_list: list[torch.Tensor] = []
            act_list: list[torch.Tensor] = []
            lp_list: list[torch.Tensor] = []
            rew_list: list[torch.Tensor] = []
            val_list: list[torch.Tensor] = []
            actv_list: list[torch.Tensor] = []
            max_steps = 2 * (M - 1) + 2
            with torch.no_grad():
                hs = self.net.encode_static(static_x)
                for _step in range(max_steps):
                    if finished.all():
                        break
                    dyn_x, mask = _build_dynamic_and_mask(batch, cur_idx, rem_cap, unvisited)
                    hd_cur = self.net.dynamic_enc(dyn_x.transpose(1, 2))
                    cx_cur = F.relu(self.net.critic_c1(torch.cat([hs, hd_cur], dim=1)))
                    cx_cur = F.relu(self.net.critic_c2(cx_cur))
                    v_cur = self.net.critic_c3(cx_cur).sum(dim=(1, 2))
                    logits, h_next = self.net.step_pointer(hs, dyn_x, dec_in, h_t, mask)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    old_logp = dist.log_prob(action)
                    was_fin = finished.clone()
                    cur_idx, rem_cap, unvisited, step_c, finished = _rollout_step_update(
                        batch, action, cur_idx, rem_cap, unvisited, finished
                    )
                    rew = -step_c / 20.0
                    if finished.any():
                        ret_d = batch["dist"][torch.arange(B, device=self.device), cur_idx, torch.zeros_like(cur_idx)]
                        just_fin = (~was_fin) & finished & (cur_idx != 0)
                        rew = rew - torch.where(just_fin, ret_d / 20.0, torch.zeros_like(rew))
                    active_f = (~was_fin).float()
                    dyn_list.append(dyn_x)
                    dec_list.append(dec_in)
                    hp_list.append(h_t)
                    msk_list.append(mask)
                    act_list.append(action)
                    lp_list.append(old_logp)
                    rew_list.append(rew * active_f)
                    val_list.append(v_cur * active_f)
                    actv_list.append(active_f)
                    h_t = h_next
                    dec_in = static_x.gather(1, cur_idx.view(B, 1, 1).expand(B, 1, static_x.shape[-1])).squeeze(1)

            T = len(dyn_list)
            rew_t = torch.stack(rew_list, dim=0)
            val_t = torch.stack(val_list, dim=0)
            actv_t = torch.stack(actv_list, dim=0)
            returns = torch.zeros((T, B), device=self.device)
            advs = torch.zeros((T, B), device=self.device)
            gae = torch.zeros((B,), device=self.device)
            next_v = torch.zeros((B,), device=self.device)
            for t in reversed(range(T)):
                delta = rew_t[t] + next_v * actv_t[t] - val_t[t]
                gae = (delta + 0.95 * gae) * actv_t[t]
                advs[t] = gae
                returns[t] = gae + val_t[t]
                next_v = val_t[t]
            valid_mask = actv_t > 0.5
            if valid_mask.any():
                a_mean = advs[valid_mask].mean()
                a_std = advs[valid_mask].std() + 1e-6
                advs = (advs - a_mean) / a_std

            dyn_flat = torch.cat(dyn_list, dim=0)
            dec_flat = torch.cat(dec_list, dim=0)
            hp_flat = torch.cat(hp_list, dim=0)
            msk_flat = torch.cat(msk_list, dim=0)
            act_flat = torch.cat(act_list, dim=0)
            old_lp_flat = torch.cat(lp_list, dim=0)
            ret_flat = returns.reshape(-1)
            adv_flat = advs.reshape(-1)
            actv_flat = actv_t.reshape(-1)
            denom = torch.clamp(actv_flat.sum(), min=1.0)

            for _epoch in range(3):
                hs_base = self.net.encode_static(static_x)
                hs_flat = hs_base.unsqueeze(0).expand(T, B, self.net.hidden_dim, M).reshape(T * B, self.net.hidden_dim, M)
                hd_flat = self.net.dynamic_enc(dyn_flat.transpose(1, 2))
                c_cat = torch.cat([hs_flat, hd_flat], dim=1)
                cx = F.relu(self.net.critic_c1(c_cat))
                cx = F.relu(self.net.critic_c2(cx))
                v_p = self.net.critic_c3(cx).sum(dim=(1, 2))

                logits, _ = self.net.step_pointer(hs_flat, dyn_flat, dec_flat, hp_flat, msk_flat)
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(act_flat)
                ratio = torch.exp(torch.clamp(new_lp - old_lp_flat, -4.0, 4.0))
                surr1 = ratio * adv_flat
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_flat
                p_loss = -(torch.min(surr1, surr2) * actv_flat).sum() / denom
                v_loss = 0.5 * (F.smooth_l1_loss(v_p, ret_flat, reduction="none") * actv_flat).sum() / denom
                ent_loss = -0.005 * (dist.entropy() * actv_flat).sum() / denom
                tot_loss = p_loss + v_loss + ent_loss
                opt.zero_grad(set_to_none=True)
                tot_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.5)
                opt.step()

        self.net.eval()
        return time.perf_counter() - t0

    def solve(self, inst: WarehouseVRPInstance) -> tuple[list[int], float]:
        self.net.eval()
        with torch.inference_mode():
            batch = _batch_from_instances([inst], self.device)
            M = int(batch["M"].item())
            static_x = batch["static_x"]
            hs = self.net.encode_static(static_x)
            cur_idx = torch.zeros((1,), dtype=torch.long, device=self.device)
            rem_cap = batch["cap"].clone()
            unvisited = torch.ones((1, M), dtype=torch.bool, device=self.device)
            unvisited[:, 0] = False
            finished = torch.zeros((1,), dtype=torch.bool, device=self.device)
            h_t = torch.zeros((1, self.net.hidden_dim), device=self.device)
            dec_in = static_x[:, 0, :]
            route = [0]
            for _ in range(2 * (M - 1) + 2):
                if finished.all():
                    break
                dyn_x, mask = _build_dynamic_and_mask(batch, cur_idx, rem_cap, unvisited)
                logits, h_t = self.net.step_pointer(hs, dyn_x, dec_in, h_t, mask)
                action = torch.argmax(logits, dim=-1)
                cur_idx, rem_cap, unvisited, _, finished = _rollout_step_update(
                    batch, action, cur_idx, rem_cap, unvisited, finished
                )
                route.append(int(action.item()))
                dec_in = static_x[:, int(cur_idx.item()), :]
            if route[-1] != 0:
                route.append(0)
        res = Warehouse3DEnv.evaluate_tour(inst, route)
        return route, float(res["total_time"])


class DiscreteSACVRPSolver:
    """Discrete Soft Actor-Critic (Twin-Q + Target Nets + Entropy Policy) for 3D VRP."""

    name = "Discrete_SAC"

    def __init__(self, device: torch.device | None = None, hidden_dim: int = 128) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = DiscreteSACQNet(in_dim=9, hidden_dim=hidden_dim).to(self.device)
        self.q1 = DiscreteSACQNet(in_dim=9, hidden_dim=hidden_dim).to(self.device)
        self.q2 = DiscreteSACQNet(in_dim=9, hidden_dim=hidden_dim).to(self.device)
        self.q1_t = DiscreteSACQNet(in_dim=9, hidden_dim=hidden_dim).to(self.device)
        self.q2_t = DiscreteSACQNet(in_dim=9, hidden_dim=hidden_dim).to(self.device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.alpha = 0.05

    def fit(self, env: Warehouse3DEnv, n_stops: int, *, updates: int = 180, batch_size: int = 256, seed: int = 2029) -> float:
        t0 = time.perf_counter()
        rng = random.Random(seed)
        torch.manual_seed(seed)
        opt_a = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=1.5e-3)
        replay: deque[tuple[torch.Tensor, ...]] = deque(maxlen=8192)
        eff_bs = max(batch_size, 128)
        gamma = 0.98

        for ep in range(updates):
            if ep % 2 == 0 or len(replay) < eff_bs:
                insts = _sample_training_batch(env, rng, 24, n_stops)
                batch = _batch_from_instances(insts, self.device)
                B = len(insts)
                M = int(batch["M"].item())
                static_x = batch["static_x"]
                cur_idx = torch.zeros((B,), dtype=torch.long, device=self.device)
                rem_cap = batch["cap"].clone()
                unvisited = torch.ones((B, M), dtype=torch.bool, device=self.device)
                unvisited[:, 0] = False
                finished = torch.zeros((B,), dtype=torch.bool, device=self.device)
                with torch.no_grad():
                    for _ in range(2 * (M - 1) + 2):
                        if finished.all():
                            break
                        dyn_x, mask = _build_dynamic_and_mask(batch, cur_idx, rem_cap, unvisited)
                        # Discourage premature depot returns during SAC exploration unless needed
                        cust_feasible = mask[:, 1:].any(dim=-1)
                        exp_mask = mask.clone()
                        exp_mask[:, 0] = (~cust_feasible) | finished
                        sx = torch.cat([static_x, dyn_x], dim=-1)
                        logits = self.actor(sx, exp_mask)
                        probs = F.softmax(logits, dim=-1)
                        act = torch.distributions.Categorical(probs=probs).sample()
                        was_fin = finished.clone()
                        cur_idx, rem_cap, unvisited, step_c, finished = _rollout_step_update(
                            batch, act, cur_idx, rem_cap, unvisited, finished
                        )
                        rew = -step_c / 20.0
                        if finished.any():
                            ret_d = batch["dist"][torch.arange(B, device=self.device), cur_idx, torch.zeros_like(cur_idx)]
                            just_fin = (~was_fin) & finished & (cur_idx != 0)
                            rew = rew - torch.where(just_fin, ret_d / 20.0, torch.zeros_like(rew))
                        ndyn_x, nmask = _build_dynamic_and_mask(batch, cur_idx, rem_cap, unvisited)
                        ncust = nmask[:, 1:].any(dim=-1)
                        nmask[:, 0] = (~ncust) | finished
                        nsx = torch.cat([static_x, ndyn_x], dim=-1)
                        keep = (~was_fin).cpu()
                        for bi in range(B):
                            if keep[bi]:
                                replay.append((
                                    sx[bi].cpu(),
                                    exp_mask[bi].cpu(),
                                    act[bi].cpu(),
                                    rew[bi].cpu(),
                                    nsx[bi].cpu(),
                                    nmask[bi].cpu(),
                                    finished[bi].float().cpu(),
                                ))

            if len(replay) < eff_bs:
                continue
            rep_list = list(replay)
            for _g in range(2):
                sample = rng.sample(rep_list, eff_bs)
                s_b = torch.stack([x[0] for x in sample]).to(self.device)
                m_b = torch.stack([x[1] for x in sample]).to(self.device)
                a_b = torch.stack([x[2] for x in sample]).to(self.device)
                r_b = torch.stack([x[3] for x in sample]).to(self.device)
                ns_b = torch.stack([x[4] for x in sample]).to(self.device)
                nm_b = torch.stack([x[5] for x in sample]).to(self.device)
                d_b = torch.stack([x[6] for x in sample]).to(self.device)

                with torch.no_grad():
                    n_logits = self.actor(ns_b, nm_b)
                    n_probs = F.softmax(n_logits, dim=-1)
                    n_logp = torch.log(torch.clamp(n_probs, min=1e-9))
                    q1_next = self.q1_t(ns_b, nm_b)
                    q2_next = self.q2_t(ns_b, nm_b)
                    min_qn = torch.min(q1_next, q2_next)
                    v_next = (n_probs * (min_qn - self.alpha * n_logp)).sum(dim=-1)
                    y_targ = r_b + gamma * (1.0 - d_b) * v_next

                q1_val = self.q1(s_b, m_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                q2_val = self.q2(s_b, m_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                q_loss = F.smooth_l1_loss(q1_val, y_targ) + F.smooth_l1_loss(q2_val, y_targ)
                opt_q.zero_grad(set_to_none=True)
                q_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 2.0)
                opt_q.step()

                logits = self.actor(s_b, m_b)
                probs = F.softmax(logits, dim=-1)
                logp = torch.log(torch.clamp(probs, min=1e-9))
                with torch.no_grad():
                    min_q = torch.min(self.q1(s_b, m_b), self.q2(s_b, m_b))
                a_loss = (probs * (self.alpha * logp - min_q)).sum(dim=-1).mean()
                opt_a.zero_grad(set_to_none=True)
                a_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 2.0)
                opt_a.step()

                with torch.no_grad():
                    for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
                        pt.data.mul_(0.95).add_(0.05 * p.data)
                    for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
                        pt.data.mul_(0.95).add_(0.05 * p.data)

        self.actor.eval()
        return time.perf_counter() - t0

    def solve(self, inst: WarehouseVRPInstance) -> tuple[list[int], float]:
        self.actor.eval()
        with torch.inference_mode():
            batch = _batch_from_instances([inst], self.device)
            M = int(batch["M"].item())
            static_x = batch["static_x"]
            cur_idx = torch.zeros((1,), dtype=torch.long, device=self.device)
            rem_cap = batch["cap"].clone()
            unvisited = torch.ones((1, M), dtype=torch.bool, device=self.device)
            unvisited[:, 0] = False
            finished = torch.zeros((1,), dtype=torch.bool, device=self.device)
            route = [0]
            for _ in range(2 * (M - 1) + 2):
                if finished.all():
                    break
                dyn_x, mask = _build_dynamic_and_mask(batch, cur_idx, rem_cap, unvisited)
                cust_ok = mask[:, 1:].any(dim=-1)
                mask[:, 0] = (~cust_ok) | finished
                sx = torch.cat([static_x, dyn_x], dim=-1)
                logits = self.actor(sx, mask)
                action = torch.argmax(logits, dim=-1)
                cur_idx, rem_cap, unvisited, _, finished = _rollout_step_update(
                    batch, action, cur_idx, rem_cap, unvisited, finished
                )
                route.append(int(action.item()))
            if route[-1] != 0:
                route.append(0)
        res = Warehouse3DEnv.evaluate_tour(inst, route)
        return route, float(res["total_time"])


class GeneticAlgorithmVRP:
    """Order Crossover (OX) + 2-opt mutation + capacity split decoding baseline (CPU)."""

    name = "GA_OX_2opt"

    def __init__(self, *, pop_size: int = 24, generations: int = 20, seed: int = 2030) -> None:
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.seed = int(seed)

    def fit(self, env: Warehouse3DEnv, n_stops: int, **_: Any) -> float:
        return 0.0

    @staticmethod
    def _decode_split(inst: WarehouseVRPInstance, perm: list[int]) -> tuple[list[int], float]:
        dist = inst.dist_matrix
        dem = inst.demands
        cap = float(inst.capacity)
        route = [0]
        cur = 0
        rem = cap
        tot = 0.0
        for nxt in perm:
            d_need = float(dem[nxt])
            if d_need > rem + 1e-6:
                tot += float(dist[cur, 0])
                route.append(0)
                cur = 0
                rem = cap
            tot += float(dist[cur, nxt])
            route.append(nxt)
            cur = nxt
            rem -= d_need
        if cur != 0:
            tot += float(dist[cur, 0])
            route.append(0)
        return route, tot

    @staticmethod
    def _two_opt_perm(inst: WarehouseVRPInstance, perm: list[int], max_swaps: int = 8) -> list[int]:
        best = list(perm)
        _, best_c = GeneticAlgorithmVRP._decode_split(inst, best)
        n = len(best)
        swaps = 0
        for i in range(n - 1):
            for j in range(i + 1, min(n, i + 7)):
                cand = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                _, c = GeneticAlgorithmVRP._decode_split(inst, cand)
                if c + 1e-6 < best_c:
                    best = cand
                    best_c = c
                    swaps += 1
                    if swaps >= max_swaps:
                        return best
        return best

    def solve(self, inst: WarehouseVRPInstance) -> tuple[list[int], float]:
        n_stops = int(inst.coords_3d.shape[0] - 1)
        rng = random.Random(self.seed + sum(ord(c) for c in inst.instance_id))
        dist = inst.dist_matrix

        # Seed with nearest-neighbor permutation + random permutations
        base: list[int] = []
        rem = set(range(1, n_stops + 1))
        cur = 0
        while rem:
            nxt = min(rem, key=lambda j: float(dist[cur, j]))
            base.append(nxt)
            rem.remove(nxt)
            cur = nxt

        pop: list[list[int]] = [self._two_opt_perm(inst, base)]
        for _ in range(self.pop_size - 1):
            p = list(range(1, n_stops + 1))
            rng.shuffle(p)
            pop.append(p)

        scored = [(self._decode_split(inst, p)[1], p) for p in pop]
        scored.sort(key=lambda x: x[0])

        for _ in range(self.generations):
            next_pop: list[list[int]] = [scored[0][1], scored[1][1]]
            while len(next_pop) < self.pop_size:
                p1 = min(rng.sample(scored[: max(4, self.pop_size // 2)], 2), key=lambda x: x[0])[1]
                p2 = min(rng.sample(scored[: max(4, self.pop_size // 2)], 2), key=lambda x: x[0])[1]
                a, b = sorted((rng.randrange(n_stops), rng.randrange(n_stops)))
                child = [-1] * n_stops
                child[a : b + 1] = p1[a : b + 1]
                used = set(child[a : b + 1])
                fill = [x for x in p2 if x not in used]
                fi = 0
                for idx in range(n_stops):
                    if child[idx] == -1:
                        child[idx] = fill[fi]
                        fi += 1
                if rng.random() < 0.25 and n_stops >= 4:
                    i, j = sorted(rng.sample(range(n_stops), 2))
                    child[i : j + 1] = child[i : j + 1][::-1]
                next_pop.append(child)
            scored = [(self._decode_split(inst, p)[1], p) for p in next_pop]
            scored.sort(key=lambda x: x[0])

        best_perm = self._two_opt_perm(inst, scored[0][1], max_swaps=16)
        route, _ = self._decode_split(inst, best_perm)
        res = Warehouse3DEnv.evaluate_tour(inst, route)
        return route, float(res["total_time"])
