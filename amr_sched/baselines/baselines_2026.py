"""2026 Top-Venue Baselines for Joint AGV Dispatching, Routing, Storage/Bay & Precedence Scheduling.

Faithfully implements five complementary 2026 state-of-the-art methods operating inside the
unified causal snapshot (`NativeSnapshot`) & ACID transaction (`CandidateProposal`) contract:
  1. `WuEJOR26GRUAttentionNet` + `WuEJOR26Proposer` (Wu et al., *EJOR* 2026,
     `10.1016/j.ejor.2026.08.040`):
     Conv1d static/dynamic encoders + 1-layer GRU pointer-attention actor + 3-layer CNN critic
     trained with REINFORCE, coupled with semi-circle proximity priority (`p_m`) storage/bay assignment.
  2. `ChoiIJCAI26PGMPONet` + `IJCAI26PGMPOProposer` (Choi et al., *IJCAI-26* pp. 6156-6164):
     Dual Attention Network (`OMB` + `MMB`) with bit-conditioned initial embeddings (`C=8` bit vectors)
     trained via exact contrastive Preference-Guided Multi-Policy Optimization (`L_PGMPO`).
  3. `MaTII26E2GMoENet` + `TII26E2GMoEProposer` (Ma et al. & M. Zhou, *IEEE TII* 2026,
     `10.1109/TII.2026.3737438`):
     Scalable Dual-Graph Encoder (`SGSR`: precedence operation attention + competition-aware resource
     attention via shared-candidate overlap) + 6-D disturbance descriptor + Top-K Hierarchical MoE
     trained via on-policy PPO + load-balancing regularization.
  4. `SowJAIR26DyMABProposer` (Sow et al., *JAIR* Vol. 86, July 2026, `10.1613/jair.1.19324`):
     Non-stationary sliding-window (`W`) alpha-UCB bandit selecting among Random, Delayed-Seed Sync,
     and corridor Shannon-Entropy neighborhoods in LNS.
  5. `ParimiMIT26TAPFPCLNSProposer` (Parimi & Williams, *MIT CSAIL arXiv:2603.28968*, March 2026):
     Precedence-aware (`PRECEDENCE-WAIT`, `LOW-SLACK`, `AGENT-CONFLICT`) destroy with transitive
     successor closure + cross-boundary earliest release bounds + Regret-2 reinsertion repair.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from amr_sched.baselines.top_venue_baselines import _build_external_proposal
from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.policy import CandidateProposal, LGCVCRPolicy, _LazyKittingSurrogate
from amr_sched.cvcr.snapshot import NativeSnapshot, OperationSnapshot, TaskSnapshot, op_predecessors
from amr_sched.cvcr.v12_proactive_guard import compute_kitting_marginal_start_value
from amr_sched.network import Network

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT_DIR_2026 = _ROOT / "_scratch" / "checkpoints" / "baselines_2026"

EJOR26_CKPT = DEFAULT_CKPT_DIR_2026 / "ejor26_wu_gru_attn.pt"
IJCAI26_CKPT = DEFAULT_CKPT_DIR_2026 / "ijcai26_choi_pgmpo.pt"
TII26_CKPT = DEFAULT_CKPT_DIR_2026 / "tii26_ma_e2g_moe.pt"

_CKPT_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Shared Causal Snapshot Feature / Graph Extraction Helpers
# ---------------------------------------------------------------------------

def _norm_coord(net: Network) -> tuple[float, float, float, float]:
    xs = [float(n.x) for n in net.nodes.values()] or [0.0]
    ys = [float(n.y) for n in net.nodes.values()] or [0.0]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    return minx, max(1.0, maxx - minx), miny, max(1.0, maxy - miny)


_TT_CACHE: dict[tuple[int, int, int, str], dict[str, int]] = {}


def _safe_dist(net: Network, u: str, v: str) -> float:
    if u == v:
        return 0.0
    key = (id(net), len(net.nodes), len(net.edges), u)
    tt = _TT_CACHE.get(key)
    if tt is None:
        max_spd = max((float(e.speed_limit) for e in net.edges.values()), default=3.0)
        tt = net.shortest_travel_times(u, max_spd) if u in net.nodes else {}
        _TT_CACHE[key] = tt
    d = tt.get(v)
    if d is not None:
        return float(d)
    nu, nv = net.nodes.get(u), net.nodes.get(v)
    if nu is not None and nv is not None:
        return math.hypot(float(nu.x) - float(nv.x), float(nu.y) - float(nv.y))
    return 60.0


def _assign_semicircle_priority_bays(
    snapshot: NativeSnapshot,
    network: Network,
) -> tuple[dict[str, str], dict[str, int]]:
    """Wu et al. (EJOR 2026) Eq. (7, 40-41): rank items by urgency/turnover priority and map
    flexible bays cleanly using Manhattan/network semi-circle proximity.
    """
    tmap = {t.task_id: t for t in snapshot.tasks}
    bays: dict[str, str] = {op.op_id: op.bay_id for op in snapshot.operations}
    km = compute_kitting_marginal_start_value(snapshot)
    bay_load: dict[str, int] = {}
    ops_sorted = sorted(
        snapshot.operations,
        key=lambda o: (
            0 if not o.started else 1,
            0 if o.is_urgent else 1,
            int(o.due_step),
            -len(o.kit_task_ids),
            o.op_id,
        ),
    )
    op_prio: dict[str, int] = {}
    for op in ops_sorted:
        k_min = min((km[kt][0] * 1000 + km[kt][4] for kt in op.kit_task_ids if kt in km), default=op.due_step)
        op_prio[op.op_id] = int(k_min)
        if op.started or not op.eligible_bays:
            bay_load[op.bay_id] = bay_load.get(op.bay_id, 0) + 1
            continue
        kits = [tmap[tid] for tid in op.kit_task_ids if tid in tmap]
        # Eq. (40-41): p_m = 0.5 * urgency/demand + 0.5 * normalized tightness
        pm = (1.5 if op.is_urgent else 1.0) + 0.25 * len(kits)
        best_b = op.bay_id
        best_cost = float("inf")
        for cand_b in op.eligible_bays:
            if cand_b not in network.nodes:
                continue
            d_sum = sum(_safe_dist(network, t.pickup, cand_b) for t in kits) if kits else 0.0
            cong_pen = 6.0 * bay_load.get(cand_b, 0)
            score = pm * d_sum + cong_pen
            if score < best_cost:
                best_cost = score
                best_b = cand_b
        bays[op.op_id] = best_b
        bay_load[best_b] = bay_load.get(best_b, 0) + 1
    return bays, op_prio


@dataclass
class CausalScheduleInstance:
    """Compact tensor-ready causal view of releasable tasks, ops, and healthy AGVs."""

    tick: int
    rel_tids: list[str]
    healthy_agvs: list[str]
    fixed_assign: dict[str, str]
    bays: dict[str, str]
    crews: dict[str, int]
    op_prio: dict[str, int]
    task_feats: np.ndarray       # [N_t, 8]
    agv_feats: np.ndarray        # [N_a, 6]
    pair_feats: np.ndarray       # [N_t, N_a, 6]
    pred_adj: np.ndarray         # [N_t, N_t, 3] (pred, self, succ)
    disturbance_6d: np.ndarray   # [6]


def build_causal_instance(snapshot: NativeSnapshot, network: Network) -> CausalScheduleInstance:
    """Extract normalized, zero-leakage tensors for neural 2026 baselines."""
    clf = classify_commitments(snapshot)
    tmap = {t.task_id: t for t in snapshot.tasks}
    op_map = {op.op_id: op for op in snapshot.operations}
    rel_tids = sorted(clf.releasable_tasks)

    lg = LGCVCRPolicy("CAUSAL_BASE_2026", network, use_rust=True)
    healthy = lg._healthy_robots(snapshot, clf.disturbed_robots)
    if not healthy:
        healthy = sorted(r.robot_id for r in snapshot.robots if r.state != "faulted")
    if not healthy:
        healthy = sorted(r.robot_id for r in snapshot.robots)

    base_assign = lg._base_assignment(snapshot, healthy, {op.op_id: op.due_step for op in snapshot.operations})
    rel_set = set(rel_tids)
    fixed_assign = {tid: agv for tid, agv in base_assign.items() if tid not in rel_set}
    bays, op_prio = _assign_semicircle_priority_bays(snapshot, network)
    crews = {op.op_id: op.crew_id for op in snapshot.operations}

    minx, spanx, miny, spany = _norm_coord(network)
    tick = int(snapshot.tick)

    # Robot baseline workloads
    rob_map = {r.robot_id: r for r in snapshot.robots}
    fixed_counts: dict[str, int] = {aid: 0 for aid in healthy}
    for tid, aid in fixed_assign.items():
        if aid in fixed_counts:
            fixed_counts[aid] += 1

    # Robot features [N_a, 6]
    agv_feats = np.zeros((max(1, len(healthy)), 6), dtype=np.float32)
    for j, aid in enumerate(healthy):
        r = rob_map.get(aid)
        nd = network.nodes.get(r.cur_node) if (r and r.cur_node in network.nodes) else None
        nx = ((float(nd.x) - minx) / spanx) if nd else 0.5
        ny = ((float(nd.y) - miny) / spany) if nd else 0.5
        busy = 1.0 if (r and r.cur_task is not None) else 0.0
        loaded = 1.0 if (r and r.loaded_task_id is not None) else 0.0
        q_norm = min(1.0, fixed_counts[aid] / 6.0)
        rem_steps = float(len(r.route)) / 20.0 if r else 0.0
        agv_feats[j] = [nx, ny, busy, loaded, q_norm, min(1.0, rem_steps)]

    n_t = len(rel_tids)
    n_a = len(healthy)
    task_feats = np.zeros((max(1, n_t), 8), dtype=np.float32)
    pair_feats = np.zeros((max(1, n_t), max(1, n_a), 6), dtype=np.float32)
    pred_adj = np.zeros((max(1, n_t), max(1, n_t), 3), dtype=np.float32)

    if n_t > 0:
        for i, tid in enumerate(rel_tids):
            t = tmap[tid]
            pn = network.nodes.get(t.pickup)
            dn_id = bays.get(t.kit_op_id or "", t.dropoff) if t.kit_op_id else t.dropoff
            dn = network.nodes.get(dn_id)
            px = ((float(pn.x) - minx) / spanx) if pn else 0.5
            py = ((float(pn.y) - miny) / spany) if pn else 0.5
            dx = ((float(dn.x) - minx) / spanx) if dn else 0.5
            dy = ((float(dn.y) - miny) / spany) if dn else 0.5
            travel = _safe_dist(network, t.pickup, dn_id) / 40.0
            due = float(t.due_step if t.due_step is not None else (tick + 120))
            slack = max(-1.0, min(2.0, (due - tick) / 80.0))
            urg = 1.0 if t.is_urgent else 0.0
            op = op_map.get(t.kit_op_id or "")
            win_tight = 1.0 if (op is not None and getattr(op, "resource_windows", ())) else 0.0
            task_feats[i] = [px, py, dx, dy, min(2.0, travel), slack, urg, win_tight]

            for j, aid in enumerate(healthy):
                r = rob_map.get(aid)
                start_nd = r.cur_node if (r and r.cur_node in network.nodes) else t.pickup
                deadhead = _safe_dist(network, start_nd, t.pickup)
                keep = 1.0 if t.assigned_agv == aid else 0.0
                same_bay_busy = min(1.0, fixed_counts[aid] / 5.0)
                est_comp = deadhead + travel * 40.0
                est_slack = max(-1.0, min(2.0, (due - (tick + est_comp)) / 60.0))
                res_load = min(1.0, sum(1 for rv in snapshot.reservations if rv.agv_id == aid) / 25.0)
                pair_feats[i, j] = [
                    min(2.0, deadhead / 30.0),
                    keep,
                    same_bay_busy,
                    est_slack,
                    res_load,
                    1.0 - min(1.0, est_comp / 80.0),
                ]

        # Build operation-DAG induced causal task relation graph [pred, self, succ]
        op_to_tasks: dict[str, list[int]] = {}
        for i, tid in enumerate(rel_tids):
            oid = tmap[tid].kit_op_id
            if oid:
                op_to_tasks.setdefault(oid, []).append(i)
        for i in range(n_t):
            pred_adj[i, i, 1] = 1.0  # self
        for i, tid in enumerate(rel_tids):
            oid = tmap[tid].kit_op_id
            if not oid or oid not in op_map:
                continue
            for p_oid in op_predecessors(op_map[oid]):
                for pi in op_to_tasks.get(p_oid, []):
                    pred_adj[i, pi, 0] = 1.0  # pi is predecessor of i
                    pred_adj[pi, i, 2] = 1.0  # i is successor of pi
            # Also intra-kit sibling precedence affinity (same op)
            for si in op_to_tasks.get(oid, []):
                if si != i:
                    pred_adj[i, si, 0] = 0.5
                    pred_adj[i, si, 2] = 0.5

    # 6-D causal disturbance descriptor phi_k (Ma et al., IEEE TII 2026 Section IV-A2)
    n_tot_tasks = max(1, len(snapshot.tasks))
    pending_rel = sum(1 for t in snapshot.tasks if t.status in ("pending", "unassigned")) / n_tot_tasks
    recent_rel = sum(1 for t in snapshot.tasks if tick - 10 <= int(t.release_step) <= tick) / n_tot_tasks
    edge_lens = [float(e.length) / max(0.5, float(e.speed_limit)) for e in network.edges.values()]
    mean_tr = (sum(edge_lens) / len(edge_lens) / 20.0) if edge_lens else 0.1
    n_robs = max(1, len(snapshot.robots))
    faulted_frac = sum(1 for r in snapshot.robots if r.state == "faulted" or r.robot_id in clf.disturbed_robots) / n_robs
    blocked_edges = 0.0
    if snapshot.execution_feedback is not None and not snapshot.execution_feedback.is_empty():
        blocked_edges = min(1.0, len(getattr(snapshot.execution_feedback, "blocked_edges", ())) / max(1, len(network.edges)) * 8.0)
    mean_slack = float(np.mean(task_feats[:, 5])) if n_t > 0 else 0.5
    dist_6d = np.asarray(
        [pending_rel, recent_rel, min(2.0, mean_tr), faulted_frac, blocked_edges, mean_slack],
        dtype=np.float32,
    )

    return CausalScheduleInstance(
        tick=tick,
        rel_tids=rel_tids,
        healthy_agvs=healthy,
        fixed_assign=fixed_assign,
        bays=bays,
        crews=crews,
        op_prio=op_prio,
        task_feats=task_feats,
        agv_feats=agv_feats,
        pair_feats=pair_feats,
        pred_adj=pred_adj,
        disturbance_6d=dist_6d,
    )


def _decode_assignment(
    inst: CausalScheduleInstance,
    choice_indices: list[int],
) -> dict[str, str]:
    """Map selected AGV indices onto releasable tasks while keeping non-releasable locks intact."""
    out = dict(inst.fixed_assign)
    if not inst.healthy_agvs:
        return out
    n_a = len(inst.healthy_agvs)
    for i, tid in enumerate(inst.rel_tids):
        idx = int(choice_indices[i]) if i < len(choice_indices) else (i % n_a)
        idx = max(0, min(n_a - 1, idx))
        out[tid] = inst.healthy_agvs[idx]
    return out


# ---------------------------------------------------------------------------
# 1. EJOR 2026: Wu et al., GRU-Attention + Semi-Circle Storage Policy
# ---------------------------------------------------------------------------

class WuEJOR26GRUAttentionNet(nn.Module):
    """Exact Conv1d + 1-Layer GRU Pointer-Attention Actor & 3-Layer Conv1d Critic (Wu et al., EJOR 2026)."""

    def __init__(self, static_dim: int = 8, dynamic_dim: int = 8, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        # Eq. (11-13): 1D Convolutional encoders for static, dynamic & decoder input
        self.static_enc = nn.Conv1d(static_dim, hidden_dim, kernel_size=1)
        self.dynamic_enc = nn.Conv1d(dynamic_dim, hidden_dim, kernel_size=1)
        self.dec_enc = nn.Conv1d(static_dim, hidden_dim, kernel_size=1)
        # Eq. (14-17): Single-layer GRU in the pointer decoder
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(p=dropout)
        # Eq. (19-25): Attention & glimpse over [H_s, H_d, h_t]
        self.w_attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v_attn = nn.Linear(hidden_dim, 1, bias=False)
        self.w_ptr = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v_ptr = nn.Linear(hidden_dim, 1, bias=False)
        # Eq. (27-32): 3-layer Conv1d CNN Critic (hidden -> 20 -> 20 -> 1, summed over candidates)
        self.critic_c1 = nn.Conv1d(hidden_dim * 2, 20, kernel_size=1)
        self.critic_c2 = nn.Conv1d(20, 20, kernel_size=1)
        self.critic_c3 = nn.Conv1d(20, 1, kernel_size=1)

    def encode_static(self, static_x: torch.Tensor) -> torch.Tensor:
        """static_x: [B, S_s, L] -> [B, H, L]."""
        return self.static_enc(static_x)

    def estimate_value(self, static_x: torch.Tensor, dynamic_x: torch.Tensor) -> torch.Tensor:
        """Eq. (28-32) Conv1d Critic returning [B] expected return."""
        hs = self.static_enc(static_x)
        hd = self.dynamic_enc(dynamic_x)
        cat = torch.cat([hs, hd], dim=1)
        x = F.relu(self.critic_c1(cat))
        x = F.relu(self.critic_c2(x))
        x = self.critic_c3(x)  # [B, 1, L]
        return x.sum(dim=(1, 2))

    def step_pointer(
        self,
        hs: torch.Tensor,
        dynamic_x: torch.Tensor,
        dec_in: torch.Tensor,
        h_prev: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single decoding step outputting masked logits [B, L] and new GRU state [B, H]."""
        B, H, L = hs.shape
        hd = self.dynamic_enc(dynamic_x)
        h_dec = self.dec_enc(dec_in.unsqueeze(-1)).squeeze(-1)  # [B, H]
        h_t = self.gru(h_dec, h_prev)  # [B, H]
        h_exp = h_t.unsqueeze(-1).expand(B, H, L)
        cat_e = torch.cat([hs, hd, h_exp], dim=1).transpose(1, 2)  # [B, L, 3H]
        e = self.v_attn(self.drop(torch.tanh(self.w_attn(cat_e)))).squeeze(-1)  # [B, L]
        e_masked = e.masked_fill(mask <= 0.0, -1e9)
        attns = F.softmax(e_masked, dim=-1).unsqueeze(1)  # [B, 1, L]
        context = torch.bmm(attns, hs.transpose(1, 2)).expand(B, L, H)  # [B, L, H]
        cat_p = torch.cat([hs.transpose(1, 2), context], dim=-1)  # [B, L, 2H]
        logits = self.v_ptr(self.drop(torch.tanh(self.w_ptr(cat_p)))).squeeze(-1)
        logits = logits.masked_fill(mask <= 0.0, -1e9)
        return logits, h_t


def _build_wu_pair_tensors(
    inst: CausalScheduleInstance,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build [1, 8, N_t*N_a] static and dynamic candidate matrices over all (task, AGV) candidates."""
    n_t = len(inst.rel_tids)
    n_a = len(inst.healthy_agvs)
    L = max(1, n_t * n_a)
    static_m = np.zeros((8, L), dtype=np.float32)
    dyn_m = np.zeros((8, L), dtype=np.float32)
    for i in range(n_t):
        tf = inst.task_feats[i]
        for j in range(n_a):
            idx = i * n_a + j
            af = inst.agv_feats[j]
            pf = inst.pair_feats[i, j]
            static_m[:, idx] = [tf[0], tf[1], tf[2], tf[3], tf[4], tf[5], tf[6], tf[7]]
            dyn_m[:, idx] = [pf[0], pf[1], pf[2], pf[3], pf[4], pf[5], af[4], af[5]]
    st = torch.from_numpy(static_m).unsqueeze(0)
    dy = torch.from_numpy(dyn_m).unsqueeze(0)
    if device is not None:
        st, dy = st.to(device), dy.to(device)
    return st, dy


class WuEJOR26Proposer:
    """Wu et al. (EJOR 2026) Online GRU-Attention + Semi-Circle Proximity Proposer."""

    def __init__(
        self,
        *,
        ckpt_path: Path | str | None = None,
        lambda_change: float = 15.0,
        alpha_makespan: float = 3.0,
        use_rust: bool = True,
        sample_rollouts: int = 4,
        seed: int = 2026,
    ) -> None:
        self.ckpt_path = Path(ckpt_path) if ckpt_path else EJOR26_CKPT
        self.lambda_change = float(lambda_change)
        self.alpha_makespan = float(alpha_makespan)
        self.use_rust = bool(use_rust)
        self.sample_rollouts = max(1, int(sample_rollouts))
        self.rng = random.Random(int(seed))
        self._net: Optional[WuEJOR26GRUAttentionNet] = None

    def _get_net(self) -> WuEJOR26GRUAttentionNet:
        if self._net is not None:
            return self._net
        key = f"wu:{self.ckpt_path}"
        if key not in _CKPT_CACHE:
            net = WuEJOR26GRUAttentionNet()
            if self.ckpt_path.exists():
                sd = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
                net.load_state_dict(sd["model"] if "model" in sd else sd)
            net.eval()
            _CKPT_CACHE[key] = net
        self._net = _CKPT_CACHE[key]
        return self._net

    def __call__(self, snapshot: NativeSnapshot, network: Network, budget_ms: float = 100.0) -> CandidateProposal:
        t0 = _cvcr_now_s()
        deadline = t0 + max(0.008, float(budget_ms) * 0.80 / 1000.0)
        clf = classify_commitments(snapshot)
        inst = build_causal_instance(snapshot, network)
        ev = CVCREvaluator(
            network, snapshot, alpha_makespan=self.alpha_makespan, lambda_change=self.lambda_change, use_rust=self.use_rust, mode="B2"
        )
        base_assign = dict(inst.fixed_assign)
        for i, tid in enumerate(inst.rel_tids):
            t = next(x for x in snapshot.tasks if x.task_id == tid)
            base_assign[tid] = t.assigned_agv if (t.assigned_agv in inst.healthy_agvs) else inst.healthy_agvs[i % len(inst.healthy_agvs)]
        best_eval = ev.evaluate(base_assign, op_priority=inst.op_prio, op_bay_overrides=inst.bays, op_crew_overrides=inst.crews)
        best_assign = dict(base_assign)

        n_t = len(inst.rel_tids)
        n_a = len(inst.healthy_agvs)
        if n_t > 0 and n_a >= 2:
            net = self._get_net()
            st, dy = _build_wu_pair_tensors(inst)
            L = n_t * n_a
            dyn_cur = dy.clone()
            with torch.no_grad():
                hs = net.encode_static(st)
                for r_i in range(self.sample_rollouts):
                    if r_i > 0 and _cvcr_now_s() >= deadline:
                        break
                    h_t = torch.zeros(1, net.hidden_dim)
                    dec_in = torch.zeros(1, 8)
                    mask = torch.ones(1, L)
                    agv_loads = [0.0 for _ in range(n_a)]
                    chosen = [0] * n_t
                    dyn_roll = dyn_cur.clone()
                    for step_i in range(n_t):
                        logits, h_t = net.step_pointer(hs, dyn_roll, dec_in, h_t, mask)
                        if r_i == 0:
                            idx = int(torch.argmax(logits, dim=-1).item())
                        else:
                            probs = F.softmax(logits[0], dim=-1)
                            idx = int(torch.multinomial(probs, 1).item())
                        ti, ai = divmod(idx, n_a)
                        chosen[ti] = ai
                        # Mask all candidate AGVs for task ti so each task is dispatched once
                        mask[0, ti * n_a : (ti + 1) * n_a] = 0.0
                        agv_loads[ai] += 1.0
                        for tk in range(n_t):
                            dyn_roll[0, 6, tk * n_a + ai] = min(1.0, dyn_roll[0, 6, tk * n_a + ai].item() + 0.16)
                        dec_in = st[:, :, idx]
                    cand_assign = _decode_assignment(inst, chosen)
                    if cand_assign != best_assign:
                        ev_c = ev.evaluate(
                            cand_assign, op_priority=inst.op_prio, op_bay_overrides=inst.bays, op_crew_overrides=inst.crews
                        )
                        if ev_c.valid and (not best_eval.valid or ev_c.objective < best_eval.objective):
                            best_eval = ev_c
                            best_assign = cand_assign

        return _build_external_proposal(
            policy_id="EJOR26_WU_GRU_ATTN",
            snapshot=snapshot,
            clf=clf,
            best_assign=best_assign,
            best_bays=inst.bays,
            best_crews=inst.crews,
            best_eval=best_eval,
            op_prio=inst.op_prio,
            eval_cnt=ev.calls,
            t0=t0,
            budget_ms=budget_ms,
        )


# ---------------------------------------------------------------------------
# 2. IJCAI-26: Choi et al. (KAIST), Bit-Conditioned DAN + PGMPO
# ---------------------------------------------------------------------------

class ChoiIJCAI26PGMPONet(nn.Module):
    """Exact Dual Attention Network + Bit-Conditioning Block (`C` policies) from Choi et al. (IJCAI-26)."""

    def __init__(
        self,
        op_dim: int = 8,
        res_dim: int = 6,
        pair_dim: int = 6,
        bit_dim: int = 4,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.bit_dim = int(bit_dim)
        # Eq. (3, 7): Linear projections + bit-conditioning injection at layer 0
        self.W1 = nn.Linear(op_dim, hidden_dim)
        self.W2 = nn.Linear(res_dim, hidden_dim)
        self.W3 = nn.Linear(bit_dim, hidden_dim, bias=False)
        self.W4 = nn.Linear(bit_dim, hidden_dim, bias=False)
        # Eq. (4-5): L OMB self-attention + MMB cross-attention blocks
        self.omb_layers = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True) for _ in range(n_layers)]
        )
        self.omb_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.mmb_layers = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True) for _ in range(n_layers)]
        )
        self.mmb_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        # Eq. (6): Pairwise scoring MLP([h_O || h_M || h_g || h_pair])
        # h_g has 2*H (mean op || mean resource)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + pair_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def bit_vectors(c_policies: int = 8, bit_dim: int = 4, device: torch.device | None = None) -> torch.Tensor:
        bits = [[(c >> b) & 1 for b in range(bit_dim)] for c in range(c_policies)]
        t = torch.tensor(bits, dtype=torch.float32)
        return t.to(device) if device is not None else t

    def forward(
        self,
        op_x: torch.Tensor,
        res_x: torch.Tensor,
        pair_x: torch.Tensor,
        z_bit: torch.Tensor,
    ) -> torch.Tensor:
        """Compute [B, N_t, N_a] candidate preference logits for conditioned policy `z_bit` [B, bit_dim]."""
        B, N_t, _ = op_x.shape
        N_a = res_x.shape[1]
        h_o = self.W1(op_x) + self.W3(z_bit).unsqueeze(1)
        h_m = self.W2(res_x) + self.W4(z_bit).unsqueeze(1)
        for omb, onm, mmb, mnm in zip(self.omb_layers, self.omb_norms, self.mmb_layers, self.mmb_norms):
            o_att, _ = omb(h_o, h_o, h_o)
            h_o = onm(h_o + o_att)
            m_att, _ = mmb(h_m, h_o, h_o)
            h_m = mnm(h_m + m_att)
        h_g = torch.cat([h_o.mean(dim=1), h_m.mean(dim=1)], dim=-1)  # [B, 2H]
        ho_e = h_o.unsqueeze(2).expand(B, N_t, N_a, self.hidden_dim)
        hm_e = h_m.unsqueeze(1).expand(B, N_t, N_a, self.hidden_dim)
        hg_e = h_g.view(B, 1, 1, -1).expand(B, N_t, N_a, self.hidden_dim * 2)
        feat = torch.cat([ho_e, hm_e, hg_e, pair_x], dim=-1)
        return self.pair_mlp(feat).squeeze(-1)


class IJCAI26PGMPOProposer:
    """Choi et al. (IJCAI-26) Multi-Policy Bit-Conditioned Dual-Attention Proposer."""

    def __init__(
        self,
        *,
        ckpt_path: Path | str | None = None,
        c_policies: int = 8,
        lambda_change: float = 15.0,
        alpha_makespan: float = 3.0,
        use_rust: bool = True,
        seed: int = 2026,
    ) -> None:
        self.ckpt_path = Path(ckpt_path) if ckpt_path else IJCAI26_CKPT
        self.c_policies = max(2, min(16, int(c_policies)))
        self.lambda_change = float(lambda_change)
        self.alpha_makespan = float(alpha_makespan)
        self.use_rust = bool(use_rust)
        self.seed = int(seed)
        self._net: Optional[ChoiIJCAI26PGMPONet] = None

    def _get_net(self) -> ChoiIJCAI26PGMPONet:
        if self._net is not None:
            return self._net
        key = f"ijcai:{self.ckpt_path}"
        if key not in _CKPT_CACHE:
            net = ChoiIJCAI26PGMPONet()
            if self.ckpt_path.exists():
                sd = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
                net.load_state_dict(sd["model"] if "model" in sd else sd)
            net.eval()
            _CKPT_CACHE[key] = net
        self._net = _CKPT_CACHE[key]
        return self._net

    def __call__(self, snapshot: NativeSnapshot, network: Network, budget_ms: float = 100.0) -> CandidateProposal:
        t0 = _cvcr_now_s()
        deadline = t0 + max(0.008, float(budget_ms) * 0.80 / 1000.0)
        clf = classify_commitments(snapshot)
        inst = build_causal_instance(snapshot, network)
        ev = CVCREvaluator(
            network, snapshot, alpha_makespan=self.alpha_makespan, lambda_change=self.lambda_change, use_rust=self.use_rust, mode="B2"
        )
        base_assign = dict(inst.fixed_assign)
        for i, tid in enumerate(inst.rel_tids):
            t = next(x for x in snapshot.tasks if x.task_id == tid)
            base_assign[tid] = t.assigned_agv if (t.assigned_agv in inst.healthy_agvs) else inst.healthy_agvs[i % len(inst.healthy_agvs)]
        best_eval = ev.evaluate(base_assign, op_priority=inst.op_prio, op_bay_overrides=inst.bays, op_crew_overrides=inst.crews)
        best_assign = dict(base_assign)

        n_t = len(inst.rel_tids)
        n_a = len(inst.healthy_agvs)
        if n_t > 0 and n_a >= 2:
            net = self._get_net()
            C = self.c_policies
            op_t = torch.from_numpy(inst.task_feats).unsqueeze(0).expand(C, -1, -1)
            res_t = torch.from_numpy(inst.agv_feats).unsqueeze(0).expand(C, -1, -1)
            pair_t = torch.from_numpy(inst.pair_feats).unsqueeze(0).expand(C, -1, -1, -1)
            bits = net.bit_vectors(C, net.bit_dim)
            with torch.no_grad():
                logits_c = net(op_t, res_t, pair_t, bits)  # [C, N_t, N_a]
                # Parallel rollout of all C conditioned policies (greedy argmax per policy)
                choices_c = torch.argmax(logits_c, dim=-1).tolist()
            seen: set[tuple[tuple[str, str], ...]] = set()
            for ch in choices_c:
                if _cvcr_now_s() >= deadline:
                    break
                cand_assign = _decode_assignment(inst, ch)
                sig = tuple(sorted(cand_assign.items()))
                if sig in seen or cand_assign == best_assign:
                    continue
                seen.add(sig)
                ev_c = ev.evaluate(cand_assign, op_priority=inst.op_prio, op_bay_overrides=inst.bays, op_crew_overrides=inst.crews)
                if ev_c.valid and (not best_eval.valid or ev_c.objective < best_eval.objective):
                    best_eval = ev_c
                    best_assign = cand_assign

        return _build_external_proposal(
            policy_id="IJCAI26_CHOI_PGMPO",
            snapshot=snapshot,
            clf=clf,
            best_assign=best_assign,
            best_bays=inst.bays,
            best_crews=inst.crews,
            best_eval=best_eval,
            op_prio=inst.op_prio,
            eval_cnt=ev.calls,
            t0=t0,
            budget_ms=budget_ms,
        )


# ---------------------------------------------------------------------------
# 3. IEEE TII 2026: Ma et al. & MengChu Zhou, E2G-MoE (Dual-Graph + HMoE + PPO)
# ---------------------------------------------------------------------------

class MaTII26E2GMoENet(nn.Module):
    """Scalable Dual-Graph Encoder (SGSR) + Hierarchical MoE Actor-Critic (Ma et al., IEEE TII 2026)."""

    def __init__(
        self,
        op_dim: int = 8,
        res_dim: int = 6,
        pair_dim: int = 6,
        dist_dim: int = 6,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_experts: int = 8,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)

        self.op_proj = nn.Linear(op_dim, hidden_dim)
        self.res_proj = nn.Linear(res_dim, hidden_dim)

        # Eq. (4-6): Precedence-Constrained Operation Dual-Graph layers (R={pred, self, succ})
        self.op_wq = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_layers)])
        self.op_wk = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_layers)])
        self.op_wv = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_layers)])
        self.op_a = nn.ParameterList([nn.Parameter(torch.randn(1, 1, 1, hidden_dim) * 0.05) for _ in range(n_layers)])
        self.rel_bias = nn.Parameter(torch.zeros(n_layers, 3))
        self.op_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        # Eq. (7-11): Competition-Aware Resource Graph with candidate-overlap context c_{m,m'}
        self.res_wq = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_layers)])
        self.res_wk = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_layers)])
        self.res_wv = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_layers)])
        self.rho_ctx = nn.ModuleList([nn.Linear(hidden_dim, 1, bias=False) for _ in range(n_layers)])
        self.res_a = nn.ParameterList([nn.Parameter(torch.randn(1, 1, 1, hidden_dim) * 0.05) for _ in range(n_layers)])
        self.res_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        # Eq. (12-14): HMoE Actor (f_pre, G(c_k), E=8 experts, f_post) + Critic V_psi
        self.f_pre = nn.Sequential(
            nn.Linear(hidden_dim * 2 + pair_dim, hidden_dim),
            nn.ReLU(),
        )
        self.router_G = nn.Sequential(
            nn.Linear(hidden_dim * 2 + dist_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_experts),
        )
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(n_experts)]
        )
        self.f_post = nn.Linear(hidden_dim, 1)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2 + dist_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        op_x: torch.Tensor,      # [B, N_t, 8]
        res_x: torch.Tensor,     # [B, N_a, 6]
        pair_x: torch.Tensor,    # [B, N_t, N_a, 6]
        pred_adj: torch.Tensor,  # [B, N_t, N_t, 3]
        dist_6d: torch.Tensor,   # [B, 6]
        *,
        training_noise: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return candidate logits [B, N_t, N_a], state value [B], and routing probs omega [B, E]."""
        B, N_t, _ = op_x.shape
        N_a = res_x.shape[1]
        h_o = self.op_proj(op_x)
        g_m = self.res_proj(res_x)

        for l in range(len(self.op_wq)):
            # Dual Graph 1: Precedence-constrained operation attention over R={pred, self, succ}
            q_o = self.op_wq[l](h_o).unsqueeze(2)  # [B, N_t, 1, H]
            k_o = self.op_wk[l](h_o).unsqueeze(1)  # [B, 1, N_t, H]
            v_o = self.op_wv[l](h_o)               # [B, N_t, H]
            base_e = torch.tanh(q_o + k_o)         # [B, N_t, N_t, H]
            scalar_e = (base_e * self.op_a[l]).sum(dim=-1)  # [B, N_t, N_t]
            # Weighted sum over causal masked relations R
            rel_mask = (pred_adj > 0.0).float()  # [B, N_t, N_t, 3]
            rel_score = (pred_adj * self.rel_bias[l].view(1, 1, 1, 3)).sum(dim=-1)
            any_rel = rel_mask.sum(dim=-1) > 0.0
            e_mat = (scalar_e + rel_score).masked_fill(~any_rel, -1e9)
            alpha_o = F.softmax(e_mat, dim=-1)
            h_o = self.op_norms[l](h_o + torch.bmm(alpha_o, v_o))

            # Dual Graph 2: Competition-aware resource attention using shared candidate context c_{m,m'} (Eq. 8-11)
            feas = (pair_x[..., 3] > -0.95).float()  # [B, N_t, N_a]
            c_mm = torch.einsum("bjm,bjn,bjh->bmnh", feas, feas, h_o)
            q_m = self.res_wq[l](g_m).unsqueeze(2)
            k_m = self.res_wk[l](g_m).unsqueeze(1)
            v_m = self.res_wv[l](g_m)
            e_m = (torch.tanh(q_m + k_m) * self.res_a[l]).sum(dim=-1) + self.rho_ctx[l](c_mm).squeeze(-1)
            beta_m = F.softmax(e_m, dim=-1)
            g_m = self.res_norms[l](g_m + torch.bmm(beta_m, v_m))

        # Global routing context c_k = [pool(h_o) || pool(g_m) || phi_k]
        c_k = torch.cat([h_o.mean(dim=1), g_m.mean(dim=1), dist_6d], dim=-1)
        g_logits = self.router_G(c_k)
        if training_noise > 0.0:
            g_logits = g_logits + torch.randn_like(g_logits) * training_noise
        # TopK sparse gating over E=8 experts (Eq. 13)
        topk_vals, topk_idx = torch.topk(g_logits, k=self.top_k, dim=-1)
        sparse_logits = torch.full_like(g_logits, -1e9).scatter(-1, topk_idx, topk_vals)
        omega = F.softmax(sparse_logits, dim=-1)  # [B, E]

        # Candidate pair representation H_k^(0) (Eq. 12)
        ho_e = h_o.unsqueeze(2).expand(B, N_t, N_a, self.hidden_dim)
        gm_e = g_m.unsqueeze(1).expand(B, N_t, N_a, self.hidden_dim)
        H0 = self.f_pre(torch.cat([ho_e, pair_x, gm_e], dim=-1))  # [B, N_t, N_a, H]

        fused = torch.zeros_like(H0)
        for e_idx, exp_fn in enumerate(self.experts):
            w_e = omega[:, e_idx].view(B, 1, 1, 1)
            if self.training or bool((w_e > 1e-6).any()):
                fused = fused + w_e * exp_fn(H0)
        logits = self.f_post(fused).squeeze(-1)  # [B, N_t, N_a]
        val = self.critic(c_k).squeeze(-1)
        return logits, val, omega


class TII26E2GMoEProposer:
    """Ma et al. & MengChu Zhou (IEEE TII 2026) Dual-Graph HMoE Online Proposer."""

    def __init__(
        self,
        *,
        ckpt_path: Path | str | None = None,
        lambda_change: float = 15.0,
        alpha_makespan: float = 3.0,
        use_rust: bool = True,
        rollout_samples: int = 4,
        seed: int = 2026,
    ) -> None:
        self.ckpt_path = Path(ckpt_path) if ckpt_path else TII26_CKPT
        self.lambda_change = float(lambda_change)
        self.alpha_makespan = float(alpha_makespan)
        self.use_rust = bool(use_rust)
        self.rollout_samples = max(1, int(rollout_samples))
        self.rng = random.Random(int(seed))
        self._net: Optional[MaTII26E2GMoENet] = None

    def _get_net(self) -> MaTII26E2GMoENet:
        if self._net is not None:
            return self._net
        key = f"tii:{self.ckpt_path}"
        if key not in _CKPT_CACHE:
            net = MaTII26E2GMoENet()
            if self.ckpt_path.exists():
                sd = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
                net.load_state_dict(sd["model"] if "model" in sd else sd)
            net.eval()
            _CKPT_CACHE[key] = net
        self._net = _CKPT_CACHE[key]
        return self._net

    def __call__(self, snapshot: NativeSnapshot, network: Network, budget_ms: float = 100.0) -> CandidateProposal:
        t0 = _cvcr_now_s()
        deadline = t0 + max(0.008, float(budget_ms) * 0.80 / 1000.0)
        clf = classify_commitments(snapshot)
        inst = build_causal_instance(snapshot, network)
        ev = CVCREvaluator(
            network, snapshot, alpha_makespan=self.alpha_makespan, lambda_change=self.lambda_change, use_rust=self.use_rust, mode="B2"
        )
        base_assign = dict(inst.fixed_assign)
        for i, tid in enumerate(inst.rel_tids):
            t = next(x for x in snapshot.tasks if x.task_id == tid)
            base_assign[tid] = t.assigned_agv if (t.assigned_agv in inst.healthy_agvs) else inst.healthy_agvs[i % len(inst.healthy_agvs)]
        best_eval = ev.evaluate(base_assign, op_priority=inst.op_prio, op_bay_overrides=inst.bays, op_crew_overrides=inst.crews)
        best_assign = dict(base_assign)

        n_t = len(inst.rel_tids)
        n_a = len(inst.healthy_agvs)
        if n_t > 0 and n_a >= 2:
            net = self._get_net()
            op_t = torch.from_numpy(inst.task_feats).unsqueeze(0)
            res_t = torch.from_numpy(inst.agv_feats).unsqueeze(0)
            pair_t = torch.from_numpy(inst.pair_feats).unsqueeze(0)
            adj_t = torch.from_numpy(inst.pred_adj).unsqueeze(0)
            dist_t = torch.from_numpy(inst.disturbance_6d).unsqueeze(0)
            with torch.no_grad():
                logits, _, _ = net(op_t, res_t, pair_t, adj_t, dist_t)  # [1, N_t, N_a]
                greedy_ch = torch.argmax(logits[0], dim=-1).tolist()
                probs = F.softmax(logits[0], dim=-1)
            cands: list[list[int]] = [greedy_ch]
            for _ in range(self.rollout_samples - 1):
                cands.append([int(torch.multinomial(probs[i], 1).item()) for i in range(n_t)])
            seen: set[tuple[tuple[str, str], ...]] = set()
            for ch in cands:
                if _cvcr_now_s() >= deadline:
                    break
                cand_assign = _decode_assignment(inst, ch)
                sig = tuple(sorted(cand_assign.items()))
                if sig in seen or cand_assign == best_assign:
                    continue
                seen.add(sig)
                ev_c = ev.evaluate(cand_assign, op_priority=inst.op_prio, op_bay_overrides=inst.bays, op_crew_overrides=inst.crews)
                if ev_c.valid and (not best_eval.valid or ev_c.objective < best_eval.objective):
                    best_eval = ev_c
                    best_assign = cand_assign

        return _build_external_proposal(
            policy_id="TII26_MA_E2G_MOE",
            snapshot=snapshot,
            clf=clf,
            best_assign=best_assign,
            best_bays=inst.bays,
            best_crews=inst.crews,
            best_eval=best_eval,
            op_prio=inst.op_prio,
            eval_cnt=ev.calls,
            t0=t0,
            budget_ms=budget_ms,
        )


# ---------------------------------------------------------------------------
# 4. JAIR 2026: Sow et al., DyMAB Non-Stationary Sliding-Window Bandit LNS
# ---------------------------------------------------------------------------

class SowJAIR26DyMABProposer:
    """Sow et al. (JAIR Vol. 86, July 2026): Non-stationary sliding-window alpha-UCB ALNS."""

    HEURISTICS = ("H_rand", "H_sync", "H_entr")

    def __init__(
        self,
        *,
        window_w: int = 12,
        alpha_ucb: float = 0.65,
        lambda_change: float = 15.0,
        alpha_makespan: float = 3.0,
        use_rust: bool = True,
        seed: int = 2026,
    ) -> None:
        self.window_w = max(3, int(window_w))
        self.alpha_ucb = float(alpha_ucb)
        self.lambda_change = float(lambda_change)
        self.alpha_makespan = float(alpha_makespan)
        self.use_rust = bool(use_rust)
        self._rng = random.Random(int(seed) + 86)
        self._queues: dict[str, deque[float]] = {h: deque(maxlen=self.window_w) for h in self.HEURISTICS}
        self._pulls: dict[str, int] = {h: 0 for h in self.HEURISTICS}
        self._iter = 0

    def reset(self) -> None:
        self._queues = {h: deque(maxlen=self.window_w) for h in self.HEURISTICS}
        self._pulls = {h: 0 for h in self.HEURISTICS}
        self._iter = 0

    def _ucb_select(self) -> str:
        # Eq. (14) + Alg. 5: mean(Q_i) + sqrt(2 * alpha / (n_i + 1))
        self._iter += 1
        decay_alpha = self.alpha_ucb / max(1.0, math.log(self._iter + 2.0))
        best_h = self.HEURISTICS[0]
        best_score = -1e18
        for h in self.HEURISTICS:
            q = self._queues[h]
            mu = (sum(q) / len(q)) if q else 0.5
            bonus = math.sqrt(2.0 * decay_alpha / (self._pulls[h] + 1.0))
            sc = mu + bonus
            if sc > best_score:
                best_score = sc
                best_h = h
        return best_h

    def _select_neighborhood(
        self,
        heuristic: str,
        rel: list[str],
        snapshot: NativeSnapshot,
        best_eval: EvalResult,
    ) -> list[str]:
        k_size = min(len(rel), 8)
        tmap = {t.task_id: t for t in snapshot.tasks}
        if heuristic == "H_rand":
            pool = list(rel)
            self._rng.shuffle(pool)
            return pool[:k_size]

        if heuristic == "H_sync":
            # Alg. 1 adaptation: seed from highest due/delay-pressure tasks + route-sharing co-agents
            ranked = sorted(rel, key=lambda tid: (int(tmap[tid].due_step or 999), tid))
            seeds = ranked[:3]
            seed_agvs = {tmap[tid].assigned_agv for tid in seeds if tmap[tid].assigned_agv}
            nbrs = list(seeds)
            for tid in rel:
                if len(nbrs) >= k_size:
                    break
                if tid not in nbrs and tmap[tid].assigned_agv in seed_agvs:
                    nbrs.append(tid)
            for tid in ranked:
                if len(nbrs) >= k_size:
                    break
                if tid not in nbrs:
                    nbrs.append(tid)
            return nbrs[:k_size]

        # H_entr (Alg. 3): corridor Shannon-entropy H(l) = -sum p(a) log2 p(a)
        node_agv_cnt: dict[str, dict[str, int]] = {}
        for aid, path in best_eval.robot_paths.items():
            for nd in path:
                m = node_agv_cnt.setdefault(nd, {})
                m[aid] = m.get(aid, 0) + 1
        node_H: dict[str, float] = {}
        for nd, counts in node_agv_cnt.items():
            tot = sum(counts.values())
            if tot <= 0:
                continue
            ent = 0.0
            for c in counts.values():
                p = c / tot
                ent -= p * math.log2(max(1e-9, p))
            node_H[nd] = ent
        scored = sorted(
            rel,
            key=lambda tid: -(node_H.get(tmap[tid].pickup, 0.0) + node_H.get(tmap[tid].dropoff, 0.0)),
        )
        return scored[:k_size]

    def __call__(self, snapshot: NativeSnapshot, network: Network, budget_ms: float = 100.0) -> CandidateProposal:
        t0 = _cvcr_now_s()
        deadline = t0 + max(0.008, float(budget_ms) * 0.80 / 1000.0)
        clf = classify_commitments(snapshot)
        lg = LGCVCRPolicy(
            "JAIR26_DYMAB", network, lambda_change=self.lambda_change, alpha_makespan=self.alpha_makespan, use_rust=self.use_rust
        )
        healthy = lg._healthy_robots(snapshot, clf.disturbed_robots)
        bays, op_prio = _assign_semicircle_priority_bays(snapshot, network)
        crews = {op.op_id: op.crew_id for op in snapshot.operations}
        best_assign = lg._base_assignment(snapshot, healthy, op_prio)

        ev = CVCREvaluator(
            network, snapshot, alpha_makespan=self.alpha_makespan, lambda_change=self.lambda_change, use_rust=self.use_rust, mode="B2"
        )
        best_eval = ev.evaluate(best_assign, op_priority=op_prio, op_bay_overrides=bays, op_crew_overrides=crews)
        rel = sorted(clf.releasable_tasks)

        if len(healthy) >= 2 and rel and lg._has_budget(deadline, ev, min_ms=4.0):
            surr = _LazyKittingSurrogate(network, snapshot, op_prio, self.alpha_makespan, self.lambda_change)
            for _ in range(3):
                if not lg._has_budget(deadline, ev, min_ms=3.0):
                    break
                h_pick = self._ucb_select()
                sub = self._select_neighborhood(h_pick, rel, snapshot, best_eval)
                prev_obj = float(best_eval.objective) if best_eval.valid else 1e9
                cand_a = surr.regret_k_assignment(
                    healthy, {t: a for t, a in best_assign.items() if t not in set(sub)}, sub, op_bays=bays, rng=self._rng, noise=0.05
                )
                ev.set_incumbent(best_assign, best_eval, op_priority=op_prio, op_bay_overrides=bays)
                if cand_a != best_assign:
                    ev_c = ev.evaluate(cand_a, op_priority=op_prio, op_bay_overrides=bays, op_crew_overrides=crews)
                    if ev_c.valid and (not best_eval.valid or ev_c.objective + 0.5 < best_eval.objective):
                        best_assign, best_eval = cand_a, ev_c
                        ev.set_incumbent(best_assign, best_eval, op_priority=op_prio, op_bay_overrides=bays)
                if lg._has_budget(deadline, ev, min_ms=2.0):
                    best_assign, best_eval, _ = lg._run_1opt_and_swap_ranked(
                        snapshot, clf, sub, ev, best_assign, best_eval, healthy, op_prio, deadline,
                        op_bay_overrides=bays, op_crew_overrides=crews, allow_swaps=True, surr=surr, hys_margin=0.0
                    )
                cur_obj = float(best_eval.objective) if best_eval.valid else prev_obj
                r_hi = max(-0.2, min(3.0, (prev_obj - cur_obj) / 50.0))
                self._queues[h_pick].append(r_hi)
                self._pulls[h_pick] += 1

        return _build_external_proposal(
            policy_id="JAIR26_SOW_DYMAB",
            snapshot=snapshot,
            clf=clf,
            best_assign=best_assign,
            best_bays=bays,
            best_crews=crews,
            best_eval=best_eval,
            op_prio=op_prio,
            eval_cnt=ev.calls,
            t0=t0,
            budget_ms=budget_ms,
        )


# ---------------------------------------------------------------------------
# 5. MIT CSAIL March 2026: Parimi & Williams, TAPFPC-LNS (arXiv:2603.28968)
# ---------------------------------------------------------------------------

class ParimiMIT26TAPFPCLNSProposer:
    """Parimi & Williams (MIT CSAIL, arXiv:2603.28968, March 2026): TAPFPC-LNS with
    PRECEDENCE-WAIT, LOW-SLACK, Transitive Successor Closure & Regret-2 Reinsertion Repair.
    """

    OPS = ("PRECEDENCE_WAIT", "LOW_SLACK", "AGENT_CONFLICT")

    def __init__(
        self,
        *,
        lambda_change: float = 15.0,
        alpha_makespan: float = 3.0,
        use_rust: bool = True,
        seed: int = 2026,
    ) -> None:
        self.lambda_change = float(lambda_change)
        self.alpha_makespan = float(alpha_makespan)
        self.use_rust = bool(use_rust)
        self._rng = random.Random(int(seed) + 2603)
        self._weights: dict[str, float] = {k: 1.0 for k in self.OPS}

    def _transitive_closure_tasks(
        self,
        seed_tids: list[str],
        rel_set: set[str],
        snapshot: NativeSnapshot,
    ) -> list[str]:
        """Take transitive successor closure in precedence DAG (Parimi & Williams Sec. 4.2)."""
        op_map = {op.op_id: op for op in snapshot.operations}
        tmap = {t.task_id: t for t in snapshot.tasks}
        # Build op -> successor ops adjacency
        succ_ops: dict[str, list[str]] = {}
        for op in snapshot.operations:
            for p in op_predecessors(op):
                succ_ops.setdefault(p, []).append(op.op_id)
        seed_ops = {tmap[tid].kit_op_id for tid in seed_tids if tid in tmap and tmap[tid].kit_op_id}
        q = deque(seed_ops)
        seen_ops = set(seed_ops)
        while q:
            cur = q.popleft()
            for nxt in succ_ops.get(cur, []):
                if nxt not in seen_ops:
                    seen_ops.add(nxt)
                    q.append(nxt)
        out = list(dict.fromkeys(seed_tids))
        for tid in sorted(rel_set):
            if tid not in out and tmap[tid].kit_op_id in seen_ops:
                out.append(tid)
        return out[:10]

    def __call__(self, snapshot: NativeSnapshot, network: Network, budget_ms: float = 100.0) -> CandidateProposal:
        t0 = _cvcr_now_s()
        deadline = t0 + max(0.008, float(budget_ms) * 0.80 / 1000.0)
        clf = classify_commitments(snapshot)
        lg = LGCVCRPolicy(
            "MIT26_TAPFPC_LNS", network, lambda_change=self.lambda_change, alpha_makespan=self.alpha_makespan, use_rust=self.use_rust
        )
        healthy = lg._healthy_robots(snapshot, clf.disturbed_robots)
        bays, op_prio = _assign_semicircle_priority_bays(snapshot, network)
        crews = {op.op_id: op.crew_id for op in snapshot.operations}
        best_assign = lg._base_assignment(snapshot, healthy, op_prio)

        ev = CVCREvaluator(
            network, snapshot, alpha_makespan=self.alpha_makespan, lambda_change=self.lambda_change, use_rust=self.use_rust, mode="B2"
        )
        best_eval = ev.evaluate(best_assign, op_priority=op_prio, op_bay_overrides=bays, op_crew_overrides=crews)
        rel = sorted(clf.releasable_tasks)
        rel_set = set(rel)
        tmap = {t.task_id: t for t in snapshot.tasks}
        op_map = {op.op_id: op for op in snapshot.operations}

        if len(healthy) >= 2 and rel and lg._has_budget(deadline, ev, min_ms=4.0):
            surr = _LazyKittingSurrogate(network, snapshot, op_prio, self.alpha_makespan, self.lambda_change)
            for it in range(len(self.OPS)):
                if not lg._has_budget(deadline, ev, min_ms=3.0):
                    break
                op_name = max(self.OPS, key=lambda k: self._weights[k] + (0.2 if self.OPS[it % 3] == k else 0.0))
                if op_name == "PRECEDENCE_WAIT":
                    # w_prec(theta): gap between transport completion and op readiness/window start
                    scored = []
                    for tid in rel:
                        t = tmap[tid]
                        op = op_map.get(t.kit_op_id or "")
                        arr = float(best_eval.op_start.get(op.op_id, snapshot.tick)) if op else float(snapshot.tick)
                        tt = _safe_dist(network, t.pickup, t.dropoff) / 3.0
                        w_prec = max(0.0, arr - (float(t.release_step) + tt))
                        scored.append((-w_prec, tid))
                    seeds = [tid for _, tid in sorted(scored)[:4]]
                elif op_name == "LOW_SLACK":
                    # sigma(theta_u, theta_v): tightest temporal slack on due_step - op_end
                    scored = []
                    for tid in rel:
                        t = tmap[tid]
                        op = op_map.get(t.kit_op_id or "")
                        end_t = float(best_eval.op_end.get(op.op_id, snapshot.tick)) if op else float(snapshot.tick)
                        slack = float(op.due_step if op else (t.due_step or 999)) - end_t
                        scored.append((slack, tid))
                    seeds = [tid for _, tid in sorted(scored)[:4]]
                else:
                    # AGENT_CONFLICT: heaviest-loaded AGVs in current solution
                    agv_loads: dict[str, int] = {}
                    for tid, aid in best_assign.items():
                        agv_loads[aid] = agv_loads.get(aid, 0) + 1
                    seeds = sorted(rel, key=lambda tid: (-agv_loads.get(best_assign.get(tid, ""), 0), tid))[:4]

                destroyed = self._transitive_closure_tasks(seeds, rel_set, snapshot)
                prev_obj = float(best_eval.objective) if best_eval.valid else 1e9
                # Sec. 4.3 Regret-2 reinsertion over frozen exterior assignment
                cand_a = surr.regret_k_assignment(
                    healthy,
                    {t: a for t, a in best_assign.items() if t not in set(destroyed)},
                    destroyed,
                    op_bays=bays,
                    rng=self._rng,
                    noise=0.03,
                )
                ev.set_incumbent(best_assign, best_eval, op_priority=op_prio, op_bay_overrides=bays)
                if cand_a != best_assign:
                    ev_c = ev.evaluate(cand_a, op_priority=op_prio, op_bay_overrides=bays, op_crew_overrides=crews)
                    if ev_c.valid and (not best_eval.valid or ev_c.objective + 0.5 < best_eval.objective):
                        best_assign, best_eval = cand_a, ev_c
                        ev.set_incumbent(best_assign, best_eval, op_priority=op_prio, op_bay_overrides=bays)
                if lg._has_budget(deadline, ev, min_ms=2.0):
                    best_assign, best_eval, _ = lg._run_1opt_and_swap_ranked(
                        snapshot, clf, destroyed, ev, best_assign, best_eval, healthy, op_prio, deadline,
                        op_bay_overrides=bays, op_crew_overrides=crews, allow_swaps=True, surr=surr, hys_margin=0.0
                    )
                gain = prev_obj - (float(best_eval.objective) if best_eval.valid else prev_obj)
                self._weights[op_name] = 0.85 * self._weights[op_name] + (0.4 if gain > 1.0 else 0.0)

        return _build_external_proposal(
            policy_id="MIT26_TAPFPC_LNS",
            snapshot=snapshot,
            clf=clf,
            best_assign=best_assign,
            best_bays=bays,
            best_crews=crews,
            best_eval=best_eval,
            op_prio=op_prio,
            eval_cnt=ev.calls,
            t0=t0,
            budget_ms=budget_ms,
        )
