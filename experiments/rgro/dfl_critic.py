"""Decision-Focused Relative-Advantage Critic & Event-Only History Tracker (`IF-CORE-RGRO-01`).

Key Scientific & Engineering Properties:
  1. Event-Only Read-Only Execution History (`ExecutionHistoryBuffer`):
     - Updated STRICTLY ONCE per real committed simulation step (never during candidate scoring).
     - Encodes 16-D rolling temporal dynamics + Lyapunov quadratic Progress Debt `Phi(s_t)`.
  2. Anchor-Relative Candidate Representation (`48-D`):
     - `x_in = [fv_cand - fv_anchor (24), fv_cand[:8] (8), h_vec (16)]`
     - Guarantees `fv_cand - fv_anchor == 0` identically at the reference anchor `x_b`,
       strictly zero search-order pollution (`max |f(fwd) - f(rev)| == 0.0`), and native
       generalization across anchor types (`E0_KEEP` in Task 1, `V19_Anchor` in Task 2).
  3. Pure-NumPy 3-Member Ensemble (`DFLAdvantageEnsemble`, ~0.04 ms online CPU, zero torch runtime):
     - Predicts Relative Advantage `A_hat(x_c, x_b) = J(x_b) - J(x_c)` (positive = gain over anchor),
     - Epistemic Uncertainty `sigma_hat(x_c, x_b)` across 3 independently initialized members,
     - Conservative LCB Advantage `A_LCB = A_hat - kappa * sigma_hat` (Optimizer's Curse guard),
     - Calibrated Harm Probability `p_harm(x_c, x_b) = P(J(x_c) > J(x_b) + delta)`.
  4. Decision-Focused Learning Objective (`Mandi et al., ICML 2022 / JAIR 2024`):
     - Combines Self-Normalized Expected Selection Regret + ListNet-NCE Ranking Loss +
       Asymmetric Harm-Selection Penalty + Weighted Harm BCE + weak Huber scale anchoring.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.snapshot import NativeSnapshot

HIST_DIM = 16
CAND_RAW_DIM = 24
REL_INPUT_DIM = 48  # 24 (fv_c - fv_b) + 8 (fv_c[:8]) + 16 (h_vec)
DELTA_SCALE = 100.0  # 1.0 scaled unit = 100.0 J_A points


def _clip(v: float, lo: float = -6.0, hi: float = 6.0) -> float:
    if not math.isfinite(v):
        return 0.0
    return float(max(lo, min(hi, v)))


class ExecutionHistoryBuffer:
    """Strictly event-only temporal history tracker updated ONLY on committed simulation ticks.

    Read-only during candidate generation and scoring -> guarantees zero search-order leakage.
    """

    def __init__(self, maxlen: int = 12) -> None:
        self._buf: deque[tuple[int, np.ndarray]] = deque(maxlen=max(4, int(maxlen)))
        self._ema_blocked: float = 0.0
        self._ema_debt: float = 0.0
        self._last_tick: int = -1
        self._rej_streak: int = 0
        self._cached_vec: np.ndarray = np.zeros(HIST_DIM, dtype=np.float32)

    def reset(self) -> None:
        self._buf.clear()
        self._ema_blocked = 0.0
        self._ema_debt = 0.0
        self._last_tick = -1
        self._rej_streak = 0
        self._cached_vec = np.zeros(HIST_DIM, dtype=np.float32)

    def update_from_snapshot(
        self,
        snapshot: NativeSnapshot,
        *,
        rej_streak: int = 0,
    ) -> np.ndarray:
        """Record a committed real tick snapshot (idempotent if called twice on the same tick)."""
        tick = int(snapshot.tick)
        if tick == self._last_tick and len(self._buf) > 0:
            return self._cached_vec.copy()

        n_robs = max(1, len(snapshot.robots))
        blocked_cnt = 0
        stalled_cnt = 0
        for r in snapshot.robots:
            st = str(getattr(r, "status", "")).lower()
            if st in ("blocked", "faulted", "failed", "down"):
                blocked_cnt += 1
            if int(getattr(r, "wait_steps", 0)) >= 3 or st in ("waiting", "stalled"):
                stalled_cnt += 1
        blocked_ratio = float(blocked_cnt) / float(n_robs)
        stalled_ratio = float(stalled_cnt) / float(n_robs)

        active_ops = [op for op in snapshot.operations if not getattr(op, "completed", False)]
        n_ops = max(1, len(snapshot.operations))
        comp_ops = len(snapshot.operations) - len(active_ops)
        comp_rate = float(comp_ops) / float(n_ops)

        # Lyapunov quadratic Progress Debt Phi(s_t) & window tightness pressure
        win_tight = 0
        split_lag_sum = 0.0
        debt_sum = 0.0
        t_by_id = {t.task_id: t for t in snapshot.tasks}
        for op in active_ops:
            due = int(getattr(op, "due_step", tick + 120))
            proc = int(getattr(op, "proc_steps", 10))
            lateness_norm = max(0.0, float(tick - (due - proc)) / 60.0)
            q_o = min(4.0, lateness_norm)
            kit_ids = tuple(getattr(op, "kit_task_ids", ()) or ())
            if len(kit_ids) >= 2:
                done_flags = [
                    1 if (tid in t_by_id and str(t_by_id[tid].status).lower() == "completed") else 0
                    for tid in kit_ids
                ]
                if 0 < sum(done_flags) < len(done_flags):
                    split_lag_sum += 1.0
                    q_o += 0.5
            wins = getattr(op, "resource_windows", ()) or ()
            if wins:
                next_end = min((int(w1) for _, w1 in wins if int(w1) >= tick), default=tick)
                if (next_end - tick) <= max(20, proc + 8):
                    win_tight += 1
                    q_o += 0.5
            debt_sum += q_o * q_o

        n_act = max(1, len(active_ops))
        win_pressure = float(win_tight) / float(n_act)
        split_lag_norm = float(split_lag_sum) / float(n_act)
        progress_debt_phi = min(6.0, float(debt_sum) / float(n_act))

        # Execution feedback drift signal (if P2 active)
        fb = getattr(snapshot, "execution_feedback", None)
        drift_ratio = 0.0
        if fb is not None and not fb.is_empty():
            try:
                blks = fb.blocked_intervals(t_commit=tick, default_horizon=12)
                drift_ratio = min(3.0, 0.25 * float(len(blks)))
            except Exception:
                drift_ratio = 0.5

        cur_core = np.array(
            [blocked_ratio, stalled_ratio, win_pressure, split_lag_norm, progress_debt_phi, comp_rate],
            dtype=np.float32,
        )

        if self._last_tick >= 0 and len(self._buf) > 0:
            prev_tick, prev_core = self._buf[-1]
            dt_raw = max(1, tick - prev_tick)
            dt_sim_norm = min(5.0, float(dt_raw) / 20.0)
            # Finite-difference velocities over ~4-step horizon when available
            ref_tick, ref_core = self._buf[max(0, len(self._buf) - 4)]
            span = max(1.0, float(tick - ref_tick) / 20.0)
            d_blocked = _clip((blocked_ratio - float(ref_core[0])) / span, -3.0, 3.0)
            d_stalled = _clip((stalled_ratio - float(ref_core[1])) / span, -3.0, 3.0)
            d_debt = _clip((progress_debt_phi - float(ref_core[4])) / span, -4.0, 4.0)
            d_comp_vel = _clip((comp_rate - float(ref_core[5])) / span, -2.0, 2.0)
            alpha = 1.0 - math.exp(-float(dt_raw) / 25.0)
            self._ema_blocked = (1.0 - alpha) * self._ema_blocked + alpha * blocked_ratio
            self._ema_debt = (1.0 - alpha) * self._ema_debt + alpha * progress_debt_phi
        else:
            dt_sim_norm = 0.25
            d_blocked = 0.0
            d_stalled = 0.0
            d_debt = 0.0
            d_comp_vel = 0.0
            self._ema_blocked = blocked_ratio
            self._ema_debt = progress_debt_phi

        self._buf.append((tick, cur_core))
        self._last_tick = tick
        self._rej_streak = int(rej_streak)
        obs_fill = float(len(self._buf)) / float(self._buf.maxlen or 12)

        vec = np.array(
            [
                _clip(blocked_ratio),
                _clip(stalled_ratio),
                _clip(win_pressure),
                _clip(split_lag_norm),
                _clip(progress_debt_phi),
                _clip(comp_rate),
                _clip(d_blocked),
                _clip(d_stalled),
                _clip(d_debt),
                _clip(d_comp_vel),
                _clip(dt_sim_norm),
                _clip(self._ema_blocked),
                _clip(self._ema_debt),
                _clip(drift_ratio),
                _clip(min(4.0, 0.5 * float(self._rej_streak))),
                _clip(obs_fill),
            ],
            dtype=np.float32,
        )
        self._cached_vec = vec
        return vec.copy()

    def get_vector_readonly(self) -> np.ndarray:
        """Read-only accessor guaranteed never to mutate internal state."""
        return self._cached_vec.copy()


def build_anchor_relative_feature(
    fv_cand: np.ndarray,
    fv_anchor: np.ndarray,
    h_vec: np.ndarray,
) -> np.ndarray:
    """Construct 48-D anchor-relative feature vector `[fv_c - fv_b (24), fv_c[:8] (8), h_vec (16)]`."""
    c = np.asarray(fv_cand, dtype=np.float32).reshape(CAND_RAW_DIM)
    b = np.asarray(fv_anchor, dtype=np.float32).reshape(CAND_RAW_DIM)
    h = np.asarray(h_vec, dtype=np.float32).reshape(HIST_DIM)
    diff = np.clip(c - b, -8.0, 8.0)
    base8 = np.clip(c[:8], -8.0, 8.0)
    return np.concatenate([diff, base8, np.clip(h, -6.0, 6.0)], axis=0).astype(np.float32)


@dataclass
class CandidateDFLScore:
    """Output of `DFLAdvantageEnsemble.score_candidates` for one candidate vs anchor."""

    pred_delta_ja: float       # Predicted J(x_c) - J(x_b) in J_A units (negative = better than anchor)
    pred_advantage: float      # -pred_delta_ja (positive = predicted gain over anchor)
    epistemic_std: float       # Ensemble std across members in J_A units
    lcb_advantage: float       # pred_advantage - kappa * epistemic_std
    harm_prob: float           # Calibrated probability P(J(x_c) > J(x_b) + delta)


class DFLAdvantageEnsemble:
    """Pure-NumPy 3-member MLP ensemble (`48 -> H -> 2`) for sub-0.1 ms online inference."""

    def __init__(
        self,
        *,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        members: list[dict[str, np.ndarray]],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.feat_mean = np.asarray(feat_mean, dtype=np.float32).reshape(1, -1)
        self.feat_std = np.maximum(1e-4, np.asarray(feat_std, dtype=np.float32).reshape(1, -1))
        self.members = members
        self.metadata = dict(metadata or {})

    @classmethod
    def load_npz(cls, path: Path) -> "DFLAdvantageEnsemble":
        data = np.load(str(path), allow_pickle=False)
        n_mem = int(data["n_members"])
        members: list[dict[str, np.ndarray]] = []
        for i in range(n_mem):
            members.append(
                {
                    "w1": np.asarray(data[f"m{i}_w1"], dtype=np.float32),
                    "b1": np.asarray(data[f"m{i}_b1"], dtype=np.float32),
                    "w2": np.asarray(data[f"m{i}_w2"], dtype=np.float32),
                    "b2": np.asarray(data[f"m{i}_b2"], dtype=np.float32),
                }
            )
        meta: dict[str, Any] = {}
        if "meta_json" in data:
            import json

            meta = json.loads(str(data["meta_json"]))
        return cls(
            feat_mean=data["feat_mean"],
            feat_std=data["feat_std"],
            members=members,
            metadata=meta,
        )

    def save_npz(self, path: Path) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "n_members": np.asarray(len(self.members), dtype=np.int32),
            "feat_mean": self.feat_mean.reshape(-1).astype(np.float32),
            "feat_std": self.feat_std.reshape(-1).astype(np.float32),
            "meta_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        for i, m in enumerate(self.members):
            arrays[f"m{i}_w1"] = m["w1"].astype(np.float32)
            arrays[f"m{i}_b1"] = m["b1"].astype(np.float32)
            arrays[f"m{i}_w2"] = m["w2"].astype(np.float32)
            arrays[f"m{i}_b2"] = m["b2"].astype(np.float32)
        np.savez(str(path), **arrays)

    def score_batch(
        self,
        X_rel: np.ndarray,
        *,
        kappa: float = 0.5,
    ) -> list[CandidateDFLScore]:
        """Vectorized pure-NumPy inference over K candidates (strictly permutation-invariant)."""
        X = np.asarray(X_rel, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        Xn = np.clip((X - self.feat_mean) / self.feat_std, -6.0, 6.0)

        deltas_list: list[np.ndarray] = []
        harms_list: list[np.ndarray] = []
        for m in self.members:
            # SiLU activation: x * sigmoid(x)
            z1 = Xn @ m["w1"] + m["b1"]
            sig1 = 1.0 / (1.0 + np.exp(-np.clip(z1, -20.0, 20.0)))
            h1 = z1 * sig1
            out = h1 @ m["w2"] + m["b2"]  # (K, 2): [delta_scaled, harm_logit]
            deltas_list.append(out[:, 0] * DELTA_SCALE)
            harms_list.append(1.0 / (1.0 + np.exp(-np.clip(out[:, 1], -20.0, 20.0))))

        deltas_mat = np.stack(deltas_list, axis=0)  # (M, K)
        harms_mat = np.stack(harms_list, axis=0)    # (M, K)
        mean_delta = np.mean(deltas_mat, axis=0)
        std_delta = np.std(deltas_mat, axis=0)
        mean_harm = np.mean(harms_mat, axis=0)

        results: list[CandidateDFLScore] = []
        for k in range(X.shape[0]):
            d = float(mean_delta[k])
            s = float(std_delta[k])
            adv = -d
            results.append(
                CandidateDFLScore(
                    pred_delta_ja=d,
                    pred_advantage=adv,
                    epistemic_std=s,
                    lcb_advantage=adv - float(kappa) * s,
                    harm_prob=float(mean_harm[k]),
                )
            )
        return results


def train_dfl_advantage_ensemble(
    train_snaps: list[dict[str, Any]],
    *,
    loss_mode: str = "DFL",
    n_members: int = 3,
    hidden_dim: int = 48,
    epochs: int = 55,
    lr: float = 3e-3,
    weight_decay: float = 1e-3,
    harm_delta_ja: float = 50.0,
    base_seed: int = 42,
    anchor_field: str = "anchor_idx",
    target_field: str = "true_delta_anchor",
) -> DFLAdvantageEnsemble:
    """Train a 3-member `DFLAdvantageEnsemble` in PyTorch and export to pure-NumPy weights.

    `loss_mode`:
      - `"DFL"`: Mandi et al. (ICML 2022 / JAIR 2024) ListNet-NCE + Expected Selection Regret +
                 Asymmetric Harm Selection Penalty + Weighted Harm BCE + weak scale anchor.
      - `"MSE"`: Pure Huber/MSE regression on scaled relative delta + Weighted Harm BCE (control).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Gather all training feature rows to compute standardization statistics
    all_rows: list[np.ndarray] = []
    for sp in train_snaps:
        all_rows.append(np.asarray(sp["X_rel"], dtype=np.float32))
    X_all = np.concatenate(all_rows, axis=0)
    feat_mean = np.mean(X_all, axis=0).astype(np.float32)
    feat_std = np.maximum(1e-3, np.std(X_all, axis=0)).astype(np.float32)

    # Pre-build normalized snapshot tensors
    snap_tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    pos_cnt = 0.0
    tot_cnt = 0.0
    for sp in train_snaps:
        X = np.clip((np.asarray(sp["X_rel"], dtype=np.float32) - feat_mean) / feat_std, -6.0, 6.0)
        y = np.asarray(sp[target_field], dtype=np.float32)  # J(cand) - J(anchor)
        harm = (y > float(harm_delta_ja)).astype(np.float32)
        pos_cnt += float(np.sum(harm))
        tot_cnt += float(len(harm))
        snap_tensors.append(
            (
                torch.from_numpy(X),
                torch.from_numpy(y),
                torch.from_numpy(harm),
            )
        )
    pos_weight = torch.tensor([min(6.0, max(1.0, (tot_cnt - pos_cnt) / max(1.0, pos_cnt)))], dtype=torch.float32)

    members_np: list[dict[str, np.ndarray]] = []
    for m_idx in range(int(n_members)):
        sd = int(base_seed) + 101 * m_idx
        torch.manual_seed(sd)
        np.random.seed(sd)

        # Bootstrap-subsample snapshots slightly (90%) when n_members > 1 for genuine epistemic diversity
        rng = np.random.default_rng(sd)
        n_snaps = len(snap_tensors)
        if n_members > 1 and n_snaps >= 20:
            sel_ids = rng.choice(n_snaps, size=max(16, int(round(0.90 * n_snaps))), replace=True).tolist()
        else:
            sel_ids = list(range(n_snaps))

        net = nn.Sequential(
            nn.Linear(REL_INPUT_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

        for _ in range(int(epochs)):
            order = rng.permutation(len(sel_ids))
            opt.zero_grad()
            loss_acc = torch.tensor(0.0)
            batch_cnt = 0
            for idx_pos in order:
                Xt, yt, ht = snap_tensors[sel_ids[int(idx_pos)]]
                out = net(Xt)
                pred_scaled = out[:, 0]
                harm_logit = out[:, 1]
                y_scaled = yt / DELTA_SCALE

                bce = F.binary_cross_entropy_with_logits(harm_logit, ht, pos_weight=pos_weight)
                huber = F.smooth_l1_loss(pred_scaled, y_scaled, beta=1.0)

                if loss_mode.upper() == "MSE":
                    snap_loss = huber + 0.35 * bce
                else:
                    # Mandi et al. (ICML 2022 / JAIR 2024): Decision-Focused ListNet-NCE + Expected Regret + Asymmetric Harm
                    tau_pred = 0.65
                    tau_true = max(0.35, float(torch.std(y_scaled).item()) * 0.6)
                    log_p = F.log_softmax(-pred_scaled / tau_pred, dim=0)
                    p_sel = torch.exp(log_p)
                    q_true = F.softmax(-y_scaled / tau_true, dim=0)

                    # 1. Expected Selection Regret (in scaled units)
                    y_best = torch.min(y_scaled)
                    regret_loss = torch.sum(p_sel * (y_scaled - y_best))
                    # 2. ListNet cross-entropy over candidate distribution
                    listnet_loss = -torch.sum(q_true * log_p)
                    # 3. Asymmetric Harm-Selection Penalty (heavy penalty for picking candidate worse than anchor!)
                    harm_excess = F.relu(y_scaled - (float(harm_delta_ja) / DELTA_SCALE))
                    asym_harm_loss = torch.sum(p_sel * harm_excess)

                    snap_loss = (
                        1.00 * regret_loss
                        + 0.45 * listnet_loss
                        + 0.80 * asym_harm_loss
                        + 0.15 * huber
                        + 0.45 * bce
                    )

                loss_acc = loss_acc + snap_loss
                batch_cnt += 1
                if batch_cnt >= 16:
                    (loss_acc / float(batch_cnt)).backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 3.0)
                    opt.step()
                    opt.zero_grad()
                    loss_acc = torch.tensor(0.0)
                    batch_cnt = 0
            if batch_cnt > 0:
                (loss_acc / float(batch_cnt)).backward()
                nn.utils.clip_grad_norm_(net.parameters(), 3.0)
                opt.step()

        # Export pure-NumPy linear layers (transpose PyTorch [out, in] -> NumPy [in, out])
        l1: nn.Linear = net[0]  # type: ignore[assignment]
        l2: nn.Linear = net[2]  # type: ignore[assignment]
        members_np.append(
            {
                "w1": l1.weight.detach().cpu().numpy().T.astype(np.float32),
                "b1": l1.bias.detach().cpu().numpy().astype(np.float32),
                "w2": l2.weight.detach().cpu().numpy().T.astype(np.float32),
                "b2": l2.bias.detach().cpu().numpy().astype(np.float32),
            }
        )

    return DFLAdvantageEnsemble(
        feat_mean=feat_mean,
        feat_std=feat_std,
        members=members_np,
        metadata={
            "loss_mode": loss_mode.upper(),
            "n_members": int(n_members),
            "hidden_dim": int(hidden_dim),
            "harm_delta_ja": float(harm_delta_ja),
            "base_seed": int(base_seed),
            "n_train_snapshots": len(train_snaps),
        },
    )
