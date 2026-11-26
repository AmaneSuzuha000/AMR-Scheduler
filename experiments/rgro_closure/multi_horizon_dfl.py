"""Multi-Horizon Decision-Focused Advantage Ensemble (`IF-CORE-RGRO-CLOSURE-01` Task 4).

Models vector advantage `A(s, a) = [A_16, A_48, A_128, A_full]` simultaneously alongside
calibrated full-episode harm probability `P(J_full(x_c) > J_full(x_b) + delta_full)` in a pure-NumPy
3-member MLP ensemble (`48 -> H -> 5`, `~0.14 ms/snap` online CPU, zero search-order pollution).

Trained strictly on quarantined DEV counterfactual snapshots with horizon-weighted
ListNet-NCE + Expected Selection Regret + Asymmetric Full-Horizon Harm Penalty (`Mandi et al. JAIR 2024`),
explicitly testing whether multi-horizon supervision cuts short-vs-long rank reversals & `FAR_full`.
Zero LNN / LTC / CfC retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from experiments.rgro.dfl_critic import REL_INPUT_DIM

SHORT_DELTA_SCALE = 100.0
FULL_DELTA_SCALE = 500.0
HARM_DELTA_48 = 50.0
HARM_DELTA_FULL = 150.0


@dataclass
class MultiHorizonScore:
    """Per-candidate predictions across `[+16t, +48t, +128t, J_full]` + uncertainty & harm risk."""

    pred_delta_16: float
    pred_delta_48: float
    pred_delta_128: float
    pred_delta_full: float
    pred_advantage_48: float
    pred_advantage_full: float
    pred_composite_advantage: float
    std_48: float
    std_full: float
    lcb_advantage: float
    horizon_reversal_flag: bool
    harm_prob: float


class MultiHorizonDFLEnsemble:
    """Pure-NumPy 3-member multi-horizon MLP ensemble (`48 -> H -> 5`)."""

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
    def load_npz(cls, path: Path) -> "MultiHorizonDFLEnsemble":
        data = np.load(str(path), allow_pickle=False)
        n_mem = int(data["n_members"])
        members = [
            {
                "w1": np.asarray(data[f"m{i}_w1"], dtype=np.float32),
                "b1": np.asarray(data[f"m{i}_b1"], dtype=np.float32),
                "w2": np.asarray(data[f"m{i}_w2"], dtype=np.float32),
                "b2": np.asarray(data[f"m{i}_b2"], dtype=np.float32),
            }
            for i in range(n_mem)
        ]
        meta = json.loads(str(data["meta_json"])) if "meta_json" in data else {}
        return cls(feat_mean=data["feat_mean"], feat_std=data["feat_std"], members=members, metadata=meta)

    def save_npz(self, path: Path) -> None:
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
        kappa: float = 0.75,
        use_full_risk_term: bool = True,
    ) -> list[MultiHorizonScore]:
        """Vectorized pure-NumPy inference over K candidates (permutation-invariant)."""
        X = np.asarray(X_rel, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        Xn = np.clip((X - self.feat_mean) / self.feat_std, -6.0, 6.0)

        d16_m, d48_m, d128_m, df_m, h_m = [], [], [], [], []
        for m in self.members:
            z1 = Xn @ m["w1"] + m["b1"]
            sig1 = 1.0 / (1.0 + np.exp(-np.clip(z1, -20.0, 20.0)))
            out = (z1 * sig1) @ m["w2"] + m["b2"]  # (K, 5)
            d16_m.append(out[:, 0] * SHORT_DELTA_SCALE)
            d48_m.append(out[:, 1] * SHORT_DELTA_SCALE)
            d128_m.append(out[:, 2] * SHORT_DELTA_SCALE)
            df_m.append(out[:, 3] * FULL_DELTA_SCALE)
            h_m.append(1.0 / (1.0 + np.exp(-np.clip(out[:, 4], -20.0, 20.0))))

        d16 = np.mean(np.stack(d16_m, axis=0), axis=0)
        d48_mat = np.stack(d48_m, axis=0)
        d48 = np.mean(d48_mat, axis=0)
        std48 = np.std(d48_mat, axis=0)
        d128 = np.mean(np.stack(d128_m, axis=0), axis=0)
        df_mat = np.stack(df_m, axis=0)
        dfull = np.mean(df_mat, axis=0)
        stdfull = np.std(df_mat, axis=0)
        harms = np.mean(np.stack(h_m, axis=0), axis=0)

        results: list[MultiHorizonScore] = []
        for k in range(X.shape[0]):
            a16 = -float(d16[k])
            a48 = -float(d48[k])
            a128 = -float(d128[k])
            afull = -float(dfull[k])
            s48 = float(std48[k])
            sfull = float(stdfull[k])

            # Normalize A_full onto ~100-pt comparable scale (FULL / 5.0)
            a_full_scaled = afull / 5.0
            s_full_scaled = sfull / 5.0
            comp_adv = 0.15 * a16 + 0.30 * a48 + 0.25 * a128 + 0.30 * a_full_scaled
            rev_flag = bool(a48 > 10.0 and (a128 < -15.0 or afull < -120.0))
            if use_full_risk_term:
                rev_pen = 35.0 if rev_flag else 0.0
                lcb = comp_adv - float(kappa) * (0.55 * s48 + 0.45 * s_full_scaled) - rev_pen
            else:
                lcb = comp_adv

            results.append(
                MultiHorizonScore(
                    pred_delta_16=float(d16[k]),
                    pred_delta_48=float(d48[k]),
                    pred_delta_128=float(d128[k]),
                    pred_delta_full=float(dfull[k]),
                    pred_advantage_48=a48,
                    pred_advantage_full=afull,
                    pred_composite_advantage=comp_adv,
                    std_48=s48,
                    std_full=sfull,
                    lcb_advantage=lcb,
                    horizon_reversal_flag=rev_flag,
                    harm_prob=float(harms[k]),
                )
            )
        return results


def train_multi_horizon_ensemble(
    train_snaps: list[dict[str, Any]],
    *,
    mode: str = "MH_DFL",
    n_members: int = 3,
    hidden_dim: int = 64,
    epochs: int = 65,
    lr: float = 2.5e-3,
    weight_decay: float = 1e-3,
    base_seed: int = 42,
) -> MultiHorizonDFLEnsemble:
    """Train 5-output Multi-Horizon ensemble under controlled ablation modes:
      - `MSE_SINGLE`: pure SmoothL1 + Harm BCE on +48t only.
      - `MSE_MH`: multi-horizon SmoothL1 over [16, 48, 128, Full] + Harm BCE.
      - `DFL_48`: Decision-Focused ListNet-NCE + Regret + Harm ONLY on +48t horizon.
      - `MH_DFL`: Full Multi-Horizon Decision-Focused ListNet-NCE + Expected Regret + Asymmetric Full Harm.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    X_all = np.concatenate([np.asarray(sp["X_rel"], dtype=np.float32) for sp in train_snaps], axis=0)
    feat_mean = np.mean(X_all, axis=0).astype(np.float32)
    feat_std = np.maximum(1e-3, np.std(X_all, axis=0)).astype(np.float32)

    snap_tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, torch.Tensor]] = []
    pos_cnt = 0.0
    tot_cnt = 0.0
    for sp in train_snaps:
        X = np.clip((np.asarray(sp["X_rel"], dtype=np.float32) - feat_mean) / feat_std, -6.0, 6.0)
        d16 = np.asarray(sp["d16"], dtype=np.float32)
        d48 = np.asarray(sp["d48"], dtype=np.float32)
        d128 = np.asarray(sp["d128"], dtype=np.float32)
        has_full = sp.get("dfull") is not None
        dfull = np.asarray(sp["dfull"], dtype=np.float32) if has_full else np.zeros_like(d48)
        harm = (
            (dfull > HARM_DELTA_FULL) | (d48 > HARM_DELTA_48)
            if has_full
            else (d48 > HARM_DELTA_48)
        ).astype(np.float32)
        pos_cnt += float(np.sum(harm))
        tot_cnt += float(len(harm))
        snap_tensors.append(
            (
                torch.from_numpy(X),
                torch.from_numpy(d16),
                torch.from_numpy(d48),
                torch.from_numpy(d128),
                torch.from_numpy(dfull),
                has_full,
                torch.from_numpy(harm),
            )
        )
    pos_w = torch.tensor([min(6.0, max(1.0, (tot_cnt - pos_cnt) / max(1.0, pos_cnt)))], dtype=torch.float32)

    def _dfl_head_loss(pred_s: torch.Tensor, true_s: torch.Tensor, harm_thr_s: float) -> torch.Tensor:
        tau_p = 0.65
        tau_t = max(0.35, float(torch.std(true_s).item()) * 0.6)
        log_p = F.log_softmax(-pred_s / tau_p, dim=0)
        p_sel = torch.exp(log_p)
        q_true = F.softmax(-true_s / tau_t, dim=0)
        reg = torch.sum(p_sel * (true_s - torch.min(true_s)))
        lnet = -torch.sum(q_true * log_p)
        asym = torch.sum(p_sel * F.relu(true_s - harm_thr_s))
        return 1.0 * reg + 0.45 * lnet + 0.85 * asym + 0.15 * F.smooth_l1_loss(pred_s, true_s, beta=1.0)

    members_np: list[dict[str, np.ndarray]] = []
    mode_u = mode.upper()
    for m_i in range(int(n_members)):
        sd = int(base_seed) + 101 * m_i
        torch.manual_seed(sd)
        np.random.seed(sd)
        rng = np.random.default_rng(sd)
        n_snaps = len(snap_tensors)
        sel_ids = (
            rng.choice(n_snaps, size=max(16, int(round(0.90 * n_snaps))), replace=True).tolist()
            if (n_members > 1 and n_snaps >= 20)
            else list(range(n_snaps))
        )
        net = nn.Sequential(nn.Linear(REL_INPUT_DIM, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 5))
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

        for _ in range(int(epochs)):
            order = rng.permutation(len(sel_ids))
            opt.zero_grad()
            loss_acc = torch.tensor(0.0)
            b_cnt = 0
            for idx_p in order:
                Xt, y16, y48, y128, yfull, has_f, h_t = snap_tensors[sel_ids[int(idx_p)]]
                out = net(Xt)
                p16, p48, p128, pf, hl = out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4]
                s16, s48, s128 = y16 / SHORT_DELTA_SCALE, y48 / SHORT_DELTA_SCALE, y128 / SHORT_DELTA_SCALE
                sfull = yfull / FULL_DELTA_SCALE
                bce = F.binary_cross_entropy_with_logits(hl, h_t, pos_weight=pos_w)

                if mode_u == "MSE_SINGLE":
                    sl = F.smooth_l1_loss(p48, s48, beta=1.0) + 0.35 * bce
                elif mode_u == "MSE_MH":
                    sl = (
                        0.20 * F.smooth_l1_loss(p16, s16, beta=1.0)
                        + 0.35 * F.smooth_l1_loss(p48, s48, beta=1.0)
                        + 0.25 * F.smooth_l1_loss(p128, s128, beta=1.0)
                        + (0.50 * F.smooth_l1_loss(pf, sfull, beta=1.0) if has_f else 0.0)
                        + 0.35 * bce
                    )
                elif mode_u == "DFL_48":
                    sl = _dfl_head_loss(p48, s48, HARM_DELTA_48 / SHORT_DELTA_SCALE) + 0.45 * bce
                    # Keep unused heads gently regularized toward +48t so outputs remain finite
                    sl = sl + 0.05 * (
                        F.smooth_l1_loss(p16, s16, beta=1.0)
                        + F.smooth_l1_loss(p128, s128, beta=1.0)
                        + (F.smooth_l1_loss(pf, sfull, beta=1.0) if has_f else 0.0)
                    )
                else:
                    # MH_DFL: Joint Multi-Horizon Decision-Focused Supervision
                    l16 = _dfl_head_loss(p16, s16, HARM_DELTA_48 / SHORT_DELTA_SCALE)
                    l48 = _dfl_head_loss(p48, s48, HARM_DELTA_48 / SHORT_DELTA_SCALE)
                    l128 = _dfl_head_loss(p128, s128, HARM_DELTA_48 / SHORT_DELTA_SCALE)
                    lfull = _dfl_head_loss(pf, sfull, HARM_DELTA_FULL / FULL_DELTA_SCALE) if has_f else 0.0
                    sl = 0.15 * l16 + 0.35 * l48 + 0.35 * l128 + (0.65 * lfull if has_f else 0.0) + 0.45 * bce

                loss_acc = loss_acc + sl
                b_cnt += 1
                if b_cnt >= 16:
                    (loss_acc / float(b_cnt)).backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 3.0)
                    opt.step()
                    opt.zero_grad()
                    loss_acc = torch.tensor(0.0)
                    b_cnt = 0
            if b_cnt > 0:
                (loss_acc / float(b_cnt)).backward()
                nn.utils.clip_grad_norm_(net.parameters(), 3.0)
                opt.step()

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

    return MultiHorizonDFLEnsemble(
        feat_mean=feat_mean,
        feat_std=feat_std,
        members=members_np,
        metadata={
            "mode": mode_u,
            "n_members": int(n_members),
            "hidden_dim": int(hidden_dim),
            "base_seed": int(base_seed),
            "n_train_snapshots": len(train_snaps),
        },
    )
