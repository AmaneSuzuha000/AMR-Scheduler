"""Track L (`Liquid-CoRe` + `DFL` + `CRC`): Matched-Input Continuous-Time Temporal Models (`L1..L6`).

Implements the exact 6 learned models on a matched parameter & hidden-state budget (`d_h = 24`)
strictly separating:
  1. Value of explicit history features (`L1_MLP_Hist`)
  2. Value of discrete recurrence (`L2_GRU`, `L3_LSTM`)
  3. Value of closed-form continuous-time liquid dynamics (`L4_CfC_FixedDt`, `L5_CfC_RealDt`, Hasani et al. NMI 2022)
  4. Value of ODE-stepped liquid time constants (`L6_LTC_RealDt`, Hasani et al. AAAI 2021)
  5. Value of Decision-Focused Learning (`DFL` regret-weighted ranking loss, Mandi et al. ICML 2022)
  6. Value of Conformal Risk Control (`CRC` split-conformal risk calibration, Angelopoulos et al. ICLR 2024)

Engineering Invariants Enforced by Design:
  - `dt_sim` is strictly elapsed SIMULATION ticks (never CPU wall-clock time).
  - Weights remain strictly frozen at inference (`model.eval()`, `torch.no_grad()`).
  - Hidden state `h_t` is updated ONLY on real committed execution steps and passed READ-ONLY
    into candidate scoring (`score_candidates(h_t_readonly, cand_feats)`), guaranteeing zero
    search-order pollution.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CfCCell(nn.Module):
    """Exact Closed-form Continuous-time (CfC) recurrent cell (Hasani et al., NMI 2022)."""

    def __init__(self, input_dim: int, hidden_dim: int = 24) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        in_cat = self.input_dim + self.hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_cat, self.hidden_dim),
            nn.SiLU(),
        )
        self.ff1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ff2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.time_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.time_b = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, z_k: torch.Tensor, h_prev: torch.Tensor, dt_sim: torch.Tensor) -> torch.Tensor:
        # dt_sim: [B, 1] scaled simulation interval >= 0
        x = self.backbone(torch.cat([z_k, h_prev], dim=-1))
        ff1 = torch.tanh(self.ff1(x))
        ff2 = torch.tanh(self.ff2(x))
        t_interp = torch.sigmoid(self.time_a(x) * dt_sim + self.time_b(x))
        return ff1 * (1.0 - t_interp) + t_interp * ff2


class LTCCell(nn.Module):
    """Semi-implicit ODE Liquid Time-Constant (LTC) cell (Hasani et al., AAAI 2021)."""

    def __init__(self, input_dim: int, hidden_dim: int = 24, ode_substeps: int = 3) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.ode_substeps = max(1, int(ode_substeps))
        self.w_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.w_rec = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.log_tau = nn.Parameter(torch.zeros(self.hidden_dim))
        self.v_leak = nn.Parameter(torch.zeros(self.hidden_dim))
        self.v_syn = nn.Parameter(torch.ones(self.hidden_dim))

    def forward(self, z_k: torch.Tensor, h_prev: torch.Tensor, dt_sim: torch.Tensor) -> torch.Tensor:
        h = h_prev
        w_tau = F.softplus(self.log_tau) + 0.1
        sub_dt = dt_sim / float(self.ode_substeps)
        in_drive = self.w_in(z_k)
        for _ in range(self.ode_substeps):
            act = torch.sigmoid(in_drive + self.w_rec(h))
            num = h + sub_dt * (w_tau * self.v_leak + self.v_syn * act)
            den = 1.0 + sub_dt * (w_tau + act)
            h = torch.tanh(num / den)
        return h


class TemporalSequenceEncoder(nn.Module):
    """Unified matched-budget sequence encoder supporting L1..L6 architectures."""

    def __init__(self, arch: str, input_dim: int, hidden_dim: int = 24, hist_extra_dim: int = 0) -> None:
        super().__init__()
        self.arch = str(arch)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.hist_extra_dim = int(hist_extra_dim)

        if self.arch == "L1_MLP_Hist":
            self.net = nn.Sequential(
                nn.Linear(self.input_dim + self.hist_extra_dim, self.hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.SiLU(),
            )
        elif self.arch == "L2_GRU":
            self.cell = nn.GRUCell(self.input_dim, self.hidden_dim)
        elif self.arch == "L3_LSTM":
            self.cell = nn.LSTMCell(self.input_dim, self.hidden_dim)
        elif self.arch in ("L4_CfC_FixedDt", "L5_CfC_RealDt"):
            self.cell = CfCCell(self.input_dim, self.hidden_dim)
        elif self.arch == "L6_LTC_RealDt":
            self.cell = LTCCell(self.input_dim, self.hidden_dim, ode_substeps=3)
        else:
            raise ValueError(f"Unsupported temporal arch: {self.arch}")

    def forward_sequence(
        self,
        z_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        hist_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode `[B, T, D]` sequence into `[B, T, H]` hidden states strictly causally."""
        B, T, _ = z_seq.shape
        device = z_seq.device
        if self.arch == "L1_MLP_Hist":
            inp = torch.cat([z_seq, hist_seq], dim=-1) if hist_seq is not None else z_seq
            return self.net(inp)

        h = torch.zeros((B, self.hidden_dim), dtype=z_seq.dtype, device=device)
        c = torch.zeros((B, self.hidden_dim), dtype=z_seq.dtype, device=device)
        outs = []
        ones_dt = torch.ones((B, 1), dtype=z_seq.dtype, device=device)
        for t in range(T):
            zt = z_seq[:, t, :]
            dtt = dt_seq[:, t : t + 1]
            if self.arch == "L2_GRU":
                h = self.cell(zt, h)
            elif self.arch == "L3_LSTM":
                h, c = self.cell(zt, (h, c))
            elif self.arch == "L4_CfC_FixedDt":
                h = self.cell(zt, h, ones_dt)
            elif self.arch in ("L5_CfC_RealDt", "L6_LTC_RealDt"):
                h = self.cell(zt, h, dtt)
            outs.append(h)
        return torch.stack(outs, dim=1)

    def step_single(
        self,
        z_k: torch.Tensor,
        dt_k: torch.Tensor,
        state: Any,
        hist_k: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Single real-event step update (used for micro-benchmarking & online event-only updates)."""
        if self.arch == "L1_MLP_Hist":
            inp = torch.cat([z_k, hist_k], dim=-1) if hist_k is not None else z_k
            h = self.net(inp)
            return h, None
        if self.arch == "L2_GRU":
            h = self.cell(z_k, state)
            return h, h
        if self.arch == "L3_LSTM":
            h, c = self.cell(z_k, state)
            return h, (h, c)
        if self.arch == "L4_CfC_FixedDt":
            h = self.cell(z_k, state, torch.ones_like(dt_k))
            return h, h
        h = self.cell(z_k, state, dt_k)
        return h, h


class LiquidPDGModel(nn.Module):
    """Gate L1 Risk Predictor: predicts progressive stall probability `p_k = P(stall | h_k)`."""

    def __init__(self, arch: str, input_dim: int, hidden_dim: int = 24, hist_extra_dim: int = 0) -> None:
        super().__init__()
        self.encoder = TemporalSequenceEncoder(arch, input_dim, hidden_dim=hidden_dim, hist_extra_dim=hist_extra_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        hist_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_seq = self.encoder.forward_sequence(z_seq, dt_seq, hist_seq)
        return self.head(h_seq).squeeze(-1)


class LiquidCriticModel(nn.Module):
    """Gate L2 Candidate Value Critic: predicts relative cost delta `Delta Q(s_t, h_t, a_t)` vs KEEP.

    STRICT INVARIANT: `h_t` is computed ONCE from real execution history and passed READ-ONLY
    across all candidate actions `a_t` at snapshot `t` (zero search-order pollution).
    """

    def __init__(
        self,
        arch: str,
        state_dim: int,
        cand_dim: int,
        hidden_dim: int = 24,
        hist_extra_dim: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = TemporalSequenceEncoder(arch, state_dim, hidden_dim=hidden_dim, hist_extra_dim=hist_extra_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + cand_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def score_snapshot_candidates(
        self,
        z_hist: torch.Tensor,
        dt_hist: torch.Tensor,
        cand_feats: torch.Tensor,
        hist_extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`z_hist: [1, T, D_s]`, `cand_feats: [K, D_c]` -> predicted relative delta cost `[K]` (lower is better)."""
        h_all = self.encoder.forward_sequence(z_hist, dt_hist, hist_extra)  # [1, T, H]
        h_t = h_all[:, -1, :]  # [1, H] read-only fixed temporal summary at snapshot t
        K = cand_feats.shape[0]
        h_rep = h_t.expand(K, -1)
        return self.scorer(torch.cat([h_rep, cand_feats], dim=-1)).squeeze(-1)


def decision_focused_regret_loss(pred_scores: torch.Tensor, true_deltas: torch.Tensor, temp: float = 0.25) -> torch.Tensor:
    """Decision-Focused Learning (DFL) loss inspired by Mandi et al. (ICML 2022).

    Minimizes expected selection regret under smoothed Boltzmann selection `softmax(-pred / temp)`
    combined with regret-weighted pairwise margin loss so the optimizer focuses capacity on
    distinguishing high-regret mistakes rather than fitting MSE across irrelevant poor candidates.
    """
    # Normalize true deltas per snapshot to scale-free regret units >= 0
    best_true = torch.min(true_deltas)
    regrets = true_deltas - best_true  # >= 0; 0 at oracle optimal candidate
    scale = torch.clamp(torch.std(true_deltas).detach(), min=0.25)
    norm_regret = regrets / scale

    # Differentiable expected selection regret via temperature-scaled softmin over predicted costs
    probs = F.softmax(-pred_scores / max(0.05, float(temp)), dim=0)
    exp_regret = torch.sum(probs * norm_regret)

    # Regret-weighted pairwise ranking hinge/logistic loss (Mandi et al. ListNet/Pairwise DFL lens)
    diff_true = true_deltas.unsqueeze(0) - true_deltas.unsqueeze(1)  # [K, K] > 0 when j worse than i
    diff_pred = pred_scores.unsqueeze(0) - pred_scores.unsqueeze(1)  # want diff_pred > 0 when diff_true > 0
    mask = (diff_true > 0.05).float()
    weights = torch.clamp(diff_true / scale, min=0.0, max=4.0) * mask
    pair_loss = F.softplus(-diff_pred) * weights
    pair_term = pair_loss.sum() / torch.clamp(weights.sum(), min=1.0)
    return exp_regret + 0.5 * pair_term


def calibrate_conformal_risk_threshold(
    calib_probs: np.ndarray,
    calib_labels: np.ndarray,
    *,
    alpha_fnr: float = 0.10,
) -> dict[str, float]:
    """Split-Conformal Risk Control (Angelopoulos et al., ICLR 2024) for bounded False-Negative Rate.

    Finds the highest probability threshold `tau_hat` such that finite-sample conformal FNR
    bound `(sum(y==1 & p<tau) + 1) / (n_pos + 1) <= alpha_fnr`, providing calibrated
    fallback/escalation guarantees without trusting raw neural probabilities blindly.
    """
    p = np.asarray(calib_probs, dtype=np.float64)
    y = np.asarray(calib_labels, dtype=np.int64)
    pos_probs = p[y == 1]
    n_pos = len(pos_probs)
    if n_pos == 0:
        return {"tau_crc": 0.5, "calib_n_pos": 0, "calib_fnr": 0.0, "alpha_target": float(alpha_fnr)}
    grid = np.linspace(0.01, 0.99, 199)
    chosen_tau = 0.01
    for tau in grid:
        emp_fn = float(np.sum(pos_probs < tau))
        rc_bound = (emp_fn + 1.0) / float(n_pos + 1)
        if rc_bound <= float(alpha_fnr):
            chosen_tau = float(tau)
        else:
            break
    calib_fnr = float(np.mean(pos_probs < chosen_tau))
    return {
        "tau_crc": round(chosen_tau, 4),
        "calib_n_pos": int(n_pos),
        "calib_fnr": round(calib_fnr, 4),
        "alpha_target": float(alpha_fnr),
    }


def benchmark_single_step_ms(model: LiquidPDGModel, input_dim: int, hist_extra_dim: int = 0, n_iters: int = 400) -> float:
    """Measure frozen single-step CPU inference latency in milliseconds."""
    model.eval()
    arch = model.encoder.arch
    h_dim = model.encoder.hidden_dim
    z = torch.zeros((1, input_dim), dtype=torch.float32)
    dt = torch.ones((1, 1), dtype=torch.float32)
    hx = torch.zeros((1, hist_extra_dim), dtype=torch.float32) if hist_extra_dim > 0 else None
    if arch == "L3_LSTM":
        st: Any = (torch.zeros((1, h_dim)), torch.zeros((1, h_dim)))
    elif arch == "L1_MLP_Hist":
        st = None
    else:
        st = torch.zeros((1, h_dim))
    with torch.no_grad():
        for _ in range(30):
            h, st = model.encoder.step_single(z, dt, st, hx)
            _ = model.head(h)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            h, st = model.encoder.step_single(z, dt, st, hx)
            _ = model.head(h)
        dt_ms = (time.perf_counter() - t0) * 1000.0 / float(n_iters)
    return round(dt_ms, 4)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
