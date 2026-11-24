"""V10 Cross-Version Opening Selectors & Churn-Calibrated Rule Controller.

Implements strictly causal t=0 opening portfolio selectors trained ONLY on `train+val`
(`1001..1045`, `2001..2012`) and evaluated zero-shot on blind test suites & unseen layouts:
  - `HistoricalTopoSelector`: lookup best historical champion on seen layouts, fallback on unseen
  - `CausalOpeningTreeSelector`: shallow CART cost-sensitive decision tree (depth <= 2/3)
    operating purely on the 16-D `extract_opening_structural_features` vector (NO layout IDs)
  - `CausalOpeningContextualSelector`: ridge linear cost estimator predicting net objective
    of each candidate controller from the 16-D opening structural features
  - `RuleV10PortfolioPolicy`: regime-and-coupling rule selector over V10 actions under RCDR
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from amr_sched.rl.spaces import (
    ACTION_KEEP,
    NUM_EXPERT_V10_ACTIONS,
    should_query_v10_controller,
)
from amr_sched.scenarios_v10 import OPENING_FEATURE_NAMES


@dataclass
class _TreeNode:
    """Single node inside CausalOpeningTreeSelector."""

    pred_idx: int
    mean_costs: list[float]
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None


class HistoricalTopoSelector:
    """Baseline t=0 selector mapping seen topology names to Val36 best, with choke-aware unseen fallback."""

    def __init__(
        self,
        candidates: list[str],
        topo_winners: dict[str, str],
        fallback: str,
        *,
        unseen_choke_fallback: Optional[str] = None,
    ) -> None:
        self.candidates = list(candidates)
        self.topo_winners = dict(topo_winners)
        self.fallback = str(fallback)
        self.unseen_choke_fallback = str(unseen_choke_fallback) if unseen_choke_fallback else None

    def predict(self, topo: str, feat16: np.ndarray | None = None) -> str:
        """Select champion name using seen topology table, or choke-aware fallback on unseen graphs."""
        if str(topo) in self.topo_winners:
            return self.topo_winners[str(topo)]
        if self.unseen_choke_fallback is not None and feat16 is not None:
            f = np.asarray(feat16, dtype=np.float64).reshape(-1)
            if len(f) > 3 and (float(f[0]) > 3.18 or float(f[3]) >= 0.25):
                return self.unseen_choke_fallback
        return self.fallback

    def to_dict(self) -> dict[str, Any]:
        """Serialize selector parameters cleanly."""
        return {
            "kind": "historical_topo",
            "candidates": self.candidates,
            "topo_winners": self.topo_winners,
            "fallback": self.fallback,
            "unseen_choke_fallback": self.unseen_choke_fallback,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HistoricalTopoSelector":
        """Restore `HistoricalTopoSelector` cleanly from serialized dictionary."""
        return cls(
            list(d["candidates"]),
            dict(d["topo_winners"]),
            str(d["fallback"]),
            unseen_choke_fallback=d.get("unseen_choke_fallback"),
        )


class CausalOpeningTreeSelector:
    """Shallow cost-sensitive CART over strictly causal 16-D t=0 structural features."""

    def __init__(
        self,
        candidates: list[str],
        max_depth: int = 2,
        min_samples_leaf: int = 12,
        allowed_features: Optional[tuple[int, ...]] = None,
    ) -> None:
        self.candidates = list(candidates)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.allowed_features = (
            tuple(int(f) for f in allowed_features) if allowed_features is not None else tuple(range(15))
        )
        self.root: Optional[_TreeNode] = None

    def _fit_node(self, X: np.ndarray, Y: np.ndarray, depth: int) -> _TreeNode:
        mean_costs = Y.mean(axis=0).tolist()
        best_c = int(np.argmin(mean_costs))
        node = _TreeNode(pred_idx=best_c, mean_costs=mean_costs)
        if depth >= self.max_depth or len(X) < 2 * self.min_samples_leaf:
            return node

        best_total_cost = float(np.sum(Y[:, best_c]))
        best_split: Optional[tuple[int, float, np.ndarray]] = None

        n_samples, n_feats = X.shape
        feat_candidates = [f_i for f_i in self.allowed_features if 0 <= f_i < n_feats] or list(range(n_feats))
        for f_i in feat_candidates:
            vals = np.unique(np.round(X[:, f_i], 5))
            if len(vals) < 2:
                continue
            mids = 0.5 * (vals[:-1] + vals[1:])
            for thr in mids:
                left_mask = X[:, f_i] <= thr
                n_left = int(left_mask.sum())
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                cost_l = float(Y[left_mask].sum(axis=0).min())
                cost_r = float(Y[~left_mask].sum(axis=0).min())
                tot = cost_l + cost_r
                if tot + 1e-6 < best_total_cost:
                    best_total_cost = tot
                    best_split = (f_i, float(thr), left_mask)

        if best_split is None:
            return node
        f_i, thr, left_mask = best_split
        node.feature_idx = f_i
        node.threshold = thr
        node.left = self._fit_node(X[left_mask], Y[left_mask], depth + 1)
        node.right = self._fit_node(X[~left_mask], Y[~left_mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "CausalOpeningTreeSelector":
        """Fit shallow minimum-cost decision tree on `(N, 16)` features and `(N, C)` cost matrix."""
        self.root = self._fit_node(np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64), 0)
        return self

    def predict(self, feat16: np.ndarray) -> str:
        """Predict winning controller candidate from causal 16-D opening structural vector."""
        assert self.root is not None
        node = self.root
        x = np.asarray(feat16, dtype=np.float64).reshape(-1)
        while node.feature_idx is not None and node.threshold is not None:
            node = node.left if x[node.feature_idx] <= node.threshold else node.right  # type: ignore[assignment]
            assert node is not None
        return self.candidates[node.pred_idx]

    def describe_rules(self, node: Optional[_TreeNode] = None, indent: str = "") -> list[str]:
        """Return human-readable interpretable split rules for scientific inspection."""
        if node is None:
            node = self.root
        if node is None:
            return []
        if node.feature_idx is None or node.threshold is None:
            return [f"{indent}-> SELECT {self.candidates[node.pred_idx]} (mean_net={node.mean_costs[node.pred_idx]:.2f})"]
        fname = OPENING_FEATURE_NAMES[node.feature_idx]
        lines = [f"{indent}if {fname} <= {node.threshold:.4f}:"]
        lines.extend(self.describe_rules(node.left, indent + "  "))
        lines.append(f"{indent}else ({fname} > {node.threshold:.4f}):")
        lines.extend(self.describe_rules(node.right, indent + "  "))
        return lines

    def to_dict(self) -> dict[str, Any]:
        """Serialize fitted CART tree cleanly to JSON-compatible dict."""
        def _ser(n: Optional[_TreeNode]) -> Optional[dict[str, Any]]:
            if n is None:
                return None
            return {
                "pred_idx": n.pred_idx,
                "mean_costs": list(n.mean_costs),
                "feature_idx": n.feature_idx,
                "threshold": n.threshold,
                "left": _ser(n.left),
                "right": _ser(n.right),
            }
        return {
            "kind": "causal_tree",
            "candidates": list(self.candidates),
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "allowed_features": list(self.allowed_features),
            "root": _ser(self.root),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CausalOpeningTreeSelector":
        """Restore `CausalOpeningTreeSelector` from serialized dict."""
        def _de(nd: Optional[dict[str, Any]]) -> Optional[_TreeNode]:
            if nd is None:
                return None
            return _TreeNode(
                pred_idx=int(nd["pred_idx"]),
                mean_costs=[float(v) for v in nd["mean_costs"]],
                feature_idx=(int(nd["feature_idx"]) if nd.get("feature_idx") is not None else None),
                threshold=(float(nd["threshold"]) if nd.get("threshold") is not None else None),
                left=_de(nd.get("left")),
                right=_de(nd.get("right")),
            )
        af = d.get("allowed_features")
        obj = cls(
            list(d["candidates"]),
            max_depth=int(d.get("max_depth", 2)),
            min_samples_leaf=int(d.get("min_samples_leaf", 12)),
            allowed_features=(tuple(int(x) for x in af) if af is not None else None),
        )
        obj.root = _de(d.get("root"))
        return obj


class CausalOpeningContextualSelector:
    """Linear Ridge contextual selector over normalized 16-D causal opening features."""

    def __init__(self, candidates: list[str], reg_lam: float = 20.0) -> None:
        self.candidates = list(candidates)
        self.reg_lam = float(reg_lam)
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None  # (C, 17)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "CausalOpeningContextualSelector":
        """Fit ridge regression estimating per-candidate cost offset relative to sample mean."""
        X_f = np.asarray(X, dtype=np.float64)
        Y_f = np.asarray(Y, dtype=np.float64)
        # Center Y per episode so scale differences don't wash out relative candidate advantage
        Y_rel = Y_f - Y_f.mean(axis=1, keepdims=True)
        self.mu = X_f.mean(axis=0, keepdims=True)
        self.sd = np.maximum(1e-4, X_f.std(axis=0, keepdims=True))
        Xn = (X_f - self.mu) / self.sd
        Xa = np.concatenate([Xn, np.ones((len(Xn), 1), dtype=np.float64)], axis=1)
        A = Xa.T @ Xa + self.reg_lam * np.eye(Xa.shape[1], dtype=np.float64)
        # W: (C, d+1) predicting relative cost (lower is better)
        W_t = np.linalg.solve(A, Xa.T @ Y_rel)
        self.W = W_t.T
        return self

    def predict(self, feat16: np.ndarray) -> str:
        """Select candidate with lowest predicted relative net objective cost."""
        assert self.W is not None and self.mu is not None and self.sd is not None
        x = (np.asarray(feat16, dtype=np.float64).reshape(1, -1) - self.mu) / self.sd
        xa = np.concatenate([x, np.ones((1, 1), dtype=np.float64)], axis=1)[0]
        pred_rel = self.W @ xa
        return self.candidates[int(np.argmin(pred_rel))]


class MultiHeadRegretSelector:
    """Pricing-adaptive multi-head Ridge regret selector (Chen et al. 2024; Müller et al. 2022).

    Predicts relative operational, churn, and compute deltas versus `anchor` (`rl_v7`)
    using spread-weighted ridge regression on strictly causal opening features, so a
    single fitted model adapts zero-shot across arbitrary `(w_op, w_churn, w_ms)` pricing.
    """

    def __init__(
        self,
        candidates: list[str],
        *,
        anchor: str = "rl_v7",
        reg_lam: float = 2.0,
    ) -> None:
        self.candidates = list(candidates)
        self.anchor = anchor if anchor in self.candidates else self.candidates[0]
        self.reg_lam = float(reg_lam)
        self.mu = np.zeros(16, dtype=np.float64)
        self.sd = np.ones(16, dtype=np.float64)
        self.W_op: Optional[np.ndarray] = None
        self.W_churn: Optional[np.ndarray] = None
        self.W_ms: Optional[np.ndarray] = None

    @staticmethod
    def _solve_weighted_ridge(Z: np.ndarray, Y_delta: np.ndarray, weights: np.ndarray, lam: float) -> np.ndarray:
        """Fit weighted Ridge regression preserving zero intercept regularization."""
        w_sqrt = np.sqrt(np.maximum(1e-6, weights)).reshape(-1, 1)
        Zw = Z * w_sqrt
        Yw = Y_delta * w_sqrt
        pen = np.eye(Z.shape[1], dtype=np.float64) * lam
        pen[0, 0] = 0.0
        return np.linalg.solve(Zw.T @ Zw + pen, Zw.T @ Yw)

    @staticmethod
    def _best_ridge_cv(
        Z: np.ndarray,
        Y_delta: np.ndarray,
        weights: np.ndarray,
        lam_grid: tuple[float, ...],
    ) -> tuple[np.ndarray, float]:
        """Select Ridge L2 via exact weighted GCV / 6-fold CV strictly inside training data."""
        n = len(Z)
        if n < 12 or len(lam_grid) <= 1:
            lam0 = float(lam_grid[0]) if lam_grid else 2.0
            return MultiHeadRegretSelector._solve_weighted_ridge(Z, Y_delta, weights, lam0), lam0
        n_folds = min(6, n // 4)
        idx = np.arange(n)
        best_lam = float(lam_grid[0])
        best_mse = float("inf")
        for lam in lam_grid:
            sq_err = 0.0
            cnt = 0
            for f in range(n_folds):
                va = idx[f::n_folds]
                tr_m = np.ones(n, dtype=bool)
                tr_m[va] = False
                W_f = MultiHeadRegretSelector._solve_weighted_ridge(Z[tr_m], Y_delta[tr_m], weights[tr_m], float(lam))
                pred = Z[va] @ W_f
                sq_err += float(np.sum((pred - Y_delta[va]) ** 2))
                cnt += int(pred.size)
            mse = sq_err / max(1, cnt)
            if mse < best_mse:
                best_mse = mse
                best_lam = float(lam)
        return MultiHeadRegretSelector._solve_weighted_ridge(Z, Y_delta, weights, best_lam), best_lam

    def fit(
        self,
        X: np.ndarray,
        Y_op: np.ndarray,
        Y_churn: np.ndarray,
        Y_ms: np.ndarray,
        *,
        sample_weights: Optional[np.ndarray] = None,
        lam_grid: Optional[tuple[float, ...]] = None,
    ) -> "MultiHeadRegretSelector":
        """Fit 3 relative-delta heads `(op_obj, churn, compute_ms)` vs `anchor`."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.mu = X_arr.mean(axis=0)
        self.sd = np.maximum(1e-4, X_arr.std(axis=0))
        Z = np.concatenate([np.ones((len(X_arr), 1), dtype=np.float64), (X_arr - self.mu) / self.sd], axis=1)

        a_idx = self.candidates.index(self.anchor)
        d_op = np.asarray(Y_op, dtype=np.float64) - np.asarray(Y_op, dtype=np.float64)[:, [a_idx]]
        d_ch = np.asarray(Y_churn, dtype=np.float64) - np.asarray(Y_churn, dtype=np.float64)[:, [a_idx]]
        d_ms = np.asarray(Y_ms, dtype=np.float64) - np.asarray(Y_ms, dtype=np.float64)[:, [a_idx]]

        if sample_weights is None:
            # Chen et al. spread-based weighting focuses capacity on high-complementarity instances
            spread = np.ptp(d_op + 15.0 * d_ch + 0.05 * d_ms, axis=1)
            med = max(1.0, float(np.median(spread)))
            w = np.clip(spread / med, 0.25, 4.0)
        else:
            w = np.asarray(sample_weights, dtype=np.float64)

        grid = lam_grid if lam_grid is not None else (self.reg_lam,)
        self.W_op, _ = self._best_ridge_cv(Z, d_op, w, grid)
        self.W_churn, _ = self._best_ridge_cv(Z, d_ch, w, grid)
        self.W_ms, _ = self._best_ridge_cv(Z, d_ms, w, grid)
        return self

    def predict_deltas(self, feat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return predicted `(delta_op, delta_churn, delta_ms)` vectors relative to anchor."""
        assert self.W_op is not None and self.W_churn is not None and self.W_ms is not None
        x_norm = (np.asarray(feat, dtype=np.float64) - self.mu) / self.sd
        z = np.concatenate([[1.0], x_norm])
        return z @ self.W_op, z @ self.W_churn, z @ self.W_ms

    def predict(
        self,
        feat: np.ndarray,
        *,
        w_op: float = 1.0,
        w_churn: float = 15.0,
        w_ms: float = 0.05,
    ) -> str:
        """Select candidate minimizing priced composite delta `w_op*d_op + w_churn*d_ch + w_ms*d_ms`."""
        d_op, d_ch, d_ms = self.predict_deltas(feat)
        score = float(w_op) * d_op + float(w_churn) * d_ch + float(w_ms) * d_ms
        return self.candidates[int(np.argmin(score))]

    def to_dict(self) -> dict[str, Any]:
        """Serialize fitted selector weights to JSON-compatible dict."""
        assert self.W_op is not None and self.W_churn is not None and self.W_ms is not None
        return {
            "candidates": list(self.candidates),
            "anchor": self.anchor,
            "reg_lam": self.reg_lam,
            "mu": self.mu.tolist(),
            "sd": self.sd.tolist(),
            "W_op": self.W_op.tolist(),
            "W_churn": self.W_churn.tolist(),
            "W_ms": self.W_ms.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MultiHeadRegretSelector":
        """Restore `MultiHeadRegretSelector` from dictionary."""
        obj = cls(list(d["candidates"]), anchor=str(d.get("anchor", "rl_v7")), reg_lam=float(d.get("reg_lam", 2.0)))
        obj.mu = np.asarray(d["mu"], dtype=np.float64)
        obj.sd = np.asarray(d["sd"], dtype=np.float64)
        obj.W_op = np.asarray(d["W_op"], dtype=np.float64)
        obj.W_churn = np.asarray(d["W_churn"], dtype=np.float64)
        obj.W_ms = np.asarray(d["W_ms"], dtype=np.float64)
        return obj


class CEWinnerClassifier:
    """Linear Cross-Entropy winner classifier baseline (`CE`) for VBS-SBS regret comparison."""

    def __init__(self, candidates: list[str], *, l2: float = 1e-2, lr: float = 0.15, steps: int = 350) -> None:
        self.candidates = list(candidates)
        self.l2 = float(l2)
        self.lr = float(lr)
        self.steps = int(steps)
        self.mu = np.zeros(16, dtype=np.float64)
        self.sd = np.ones(16, dtype=np.float64)
        self.W: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y_cost: np.ndarray) -> "CEWinnerClassifier":
        """Train multinomial logistic regression to predict `argmin_j Y_cost[i, j]`."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.mu = X_arr.mean(axis=0)
        self.sd = np.maximum(1e-4, X_arr.std(axis=0))
        Z = np.concatenate([np.ones((len(X_arr), 1), dtype=np.float64), (X_arr - self.mu) / self.sd], axis=1)
        y_cls = np.argmin(np.asarray(Y_cost, dtype=np.float64), axis=1)
        k = len(self.candidates)
        W = np.zeros((Z.shape[1], k), dtype=np.float64)
        n = max(1, len(Z))
        one_hot = np.zeros((n, k), dtype=np.float64)
        one_hot[np.arange(n), y_cls] = 1.0
        for _ in range(self.steps):
            logits = Z @ W
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= np.maximum(1e-12, probs.sum(axis=1, keepdims=True))
            grad = (Z.T @ (probs - one_hot)) / float(n)
            grad[1:, :] += self.l2 * W[1:, :]
            W -= self.lr * grad
        self.W = W
        return self

    def predict(self, feat: np.ndarray) -> str:
        """Return candidate with highest predicted win probability."""
        assert self.W is not None
        z = np.concatenate([[1.0], (np.asarray(feat, dtype=np.float64) - self.mu) / self.sd])
        return self.candidates[int(np.argmax(z @ self.W))]

    def to_dict(self) -> dict[str, Any]:
        """Serialize CE classifier."""
        assert self.W is not None
        return {"candidates": list(self.candidates), "mu": self.mu.tolist(), "sd": self.sd.tolist(), "W": self.W.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CEWinnerClassifier":
        """Restore `CEWinnerClassifier` from dictionary."""
        obj = cls(list(d["candidates"]))
        obj.mu = np.asarray(d["mu"], dtype=np.float64)
        obj.sd = np.asarray(d["sd"], dtype=np.float64)
        obj.W = np.asarray(d["W"], dtype=np.float64)
        return obj


HANDOFF_FEATURE_DIM = 24


def extract_handoff_features(
    opening_feat16: np.ndarray,
    obs_v9: np.ndarray,
    tick: int,
    *,
    is_fault: bool = False,
    is_urgent: bool = False,
    is_stalled: bool = False,
    opening_policy: str = "rl_v7",
) -> np.ndarray:
    """Build 24-D causal state vector at first disturbance handoff (`opening16 + runtime8`)."""
    congestion = float(obs_v9[9]) if len(obs_v9) > 9 else 0.0
    late_ops = float(obs_v9[32]) if len(obs_v9) > 32 else 0.0
    pred_drift = float(obs_v9[30]) if len(obs_v9) > 30 else 0.0
    runtime8 = np.asarray(
        [
            min(1.0, float(tick) / 600.0),
            congestion,
            late_ops,
            pred_drift,
            1.0 if is_fault else 0.0,
            1.0 if is_urgent else 0.0,
            1.0 if is_stalled else 0.0,
            1.0 if opening_policy == "rl_v7" else 0.0,
        ],
        dtype=np.float64,
    )
    return np.concatenate([np.asarray(opening_feat16, dtype=np.float64), runtime8])


class TwoPointCausalSelector:
    """Two-point causal controller selector: t=0 opening pick + at most one disturbance handoff."""

    def __init__(
        self,
        opening_selector: MultiHeadRegretSelector,
        candidates: list[str],
        *,
        reg_lam: float = 3.0,
        min_switch_margin: float = 25.0,
    ) -> None:
        self.opening_selector = opening_selector
        self.candidates = list(candidates)
        self.reg_lam = float(reg_lam)
        self.min_switch_margin = float(min_switch_margin)
        self.mu = np.zeros(HANDOFF_FEATURE_DIM, dtype=np.float64)
        self.sd = np.ones(HANDOFF_FEATURE_DIM, dtype=np.float64)
        self.W_op: Optional[np.ndarray] = None
        self.W_churn: Optional[np.ndarray] = None
        self.W_ms: Optional[np.ndarray] = None

    def fit_handoff(
        self,
        H: np.ndarray,
        Y_op: np.ndarray,
        Y_churn: np.ndarray,
        Y_ms: np.ndarray,
        opening_cands: list[str],
    ) -> "TwoPointCausalSelector":
        """Fit handoff regression predicting `(delta_op, delta_churn, delta_ms)` vs current `opening_cand`."""
        H_arr = np.asarray(H, dtype=np.float64)
        self.mu = H_arr.mean(axis=0)
        self.sd = np.maximum(1e-4, H_arr.std(axis=0))
        Z = np.concatenate([np.ones((len(H_arr), 1), dtype=np.float64), (H_arr - self.mu) / self.sd], axis=1)

        base_idx = np.asarray([self.candidates.index(c) for c in opening_cands], dtype=np.int64)
        rows = np.arange(len(H_arr))
        d_op = np.asarray(Y_op, dtype=np.float64) - np.asarray(Y_op, dtype=np.float64)[rows, base_idx][:, None]
        d_ch = np.asarray(Y_churn, dtype=np.float64) - np.asarray(Y_churn, dtype=np.float64)[rows, base_idx][:, None]
        d_ms = np.asarray(Y_ms, dtype=np.float64) - np.asarray(Y_ms, dtype=np.float64)[rows, base_idx][:, None]

        spread = np.ptp(d_op + 15.0 * d_ch + 0.05 * d_ms, axis=1)
        med = max(1.0, float(np.median(spread)))
        w = np.clip(spread / med, 0.25, 4.0)

        self.W_op = MultiHeadRegretSelector._solve_weighted_ridge(Z, d_op, w, self.reg_lam)
        self.W_churn = MultiHeadRegretSelector._solve_weighted_ridge(Z, d_ch, w, self.reg_lam)
        self.W_ms = MultiHeadRegretSelector._solve_weighted_ridge(Z, d_ms, w, self.reg_lam)
        return self

    def select_opening(
        self, feat16: np.ndarray, *, w_op: float = 1.0, w_churn: float = 15.0, w_ms: float = 0.05
    ) -> str:
        """Choose controller at t=0."""
        return self.opening_selector.predict(feat16, w_op=w_op, w_churn=w_churn, w_ms=w_ms)

    def select_handoff(
        self,
        handoff_feat: np.ndarray,
        current_cand: str,
        *,
        w_op: float = 1.0,
        w_churn: float = 15.0,
        w_ms: float = 0.05,
    ) -> str:
        """Optionally switch once at first disturbance if expected gain exceeds `min_switch_margin`."""
        if self.W_op is None or self.W_churn is None or self.W_ms is None:
            return current_cand
        z = np.concatenate([[1.0], (np.asarray(handoff_feat, dtype=np.float64) - self.mu) / self.sd])
        score = float(w_op) * (z @ self.W_op) + float(w_churn) * (z @ self.W_churn) + float(w_ms) * (z @ self.W_ms)
        cur_i = self.candidates.index(current_cand) if current_cand in self.candidates else 0
        best_i = int(np.argmin(score))
        if best_i != cur_i and (float(score[cur_i]) - float(score[best_i])) >= self.min_switch_margin:
            cand_best = self.candidates[best_i]
            if cand_best == "rl_v7" and len(handoff_feat) >= 4 and (float(handoff_feat[0]) > 3.18 or float(handoff_feat[3]) >= 0.25):
                return current_cand
            return cand_best
        return current_cand

    def to_dict(self) -> dict[str, Any]:
        """Serialize `TwoPointCausalSelector`."""
        return {
            "opening_selector": self.opening_selector.to_dict(),
            "candidates": list(self.candidates),
            "reg_lam": self.reg_lam,
            "min_switch_margin": self.min_switch_margin,
            "mu": self.mu.tolist(),
            "sd": self.sd.tolist(),
            "W_op": self.W_op.tolist() if self.W_op is not None else None,
            "W_churn": self.W_churn.tolist() if self.W_churn is not None else None,
            "W_ms": self.W_ms.tolist() if self.W_ms is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TwoPointCausalSelector":
        """Restore `TwoPointCausalSelector` from dictionary."""
        obj = cls(
            MultiHeadRegretSelector.from_dict(d["opening_selector"]),
            list(d["candidates"]),
            reg_lam=float(d.get("reg_lam", 3.0)),
            min_switch_margin=float(d.get("min_switch_margin", 25.0)),
        )
        obj.mu = np.asarray(d["mu"], dtype=np.float64)
        obj.sd = np.asarray(d["sd"], dtype=np.float64)
        obj.W_op = np.asarray(d["W_op"], dtype=np.float64) if d.get("W_op") is not None else None
        obj.W_churn = np.asarray(d["W_churn"], dtype=np.float64) if d.get("W_churn") is not None else None
        obj.W_ms = np.asarray(d["W_ms"], dtype=np.float64) if d.get("W_ms") is not None else None
        return obj


class RuleV10PortfolioPolicy:
    """Deterministic causal rule policy over V10 Portfolio actions under V10 RCDR."""

    def __init__(
        self,
        epoch_interval: int = 16,
        *,
        enable_v11_stall_override: bool = False,
        stall_threshold: int = 8,
    ) -> None:
        self.epoch_interval = int(epoch_interval)
        self.enable_v11_stall_override = bool(enable_v11_stall_override)
        self.stall_threshold = int(stall_threshold)
        self.last_dec = -999
        self.last_surgical = -999
        self.prev_emerg: Any = (False, False)

    def reset(self) -> None:
        """Reset stateful query counters between episodes."""
        self.last_dec = -999
        self.last_surgical = -999
        self.prev_emerg = (False, False)

    def select_action(self, obs_v9: np.ndarray, info: dict[str, Any], sim: Any = None) -> int:
        """Pick V10 expert action causally given current 38-D state & legal mask."""
        from amr_sched.rl.spaces import is_in_secondary_hold, should_query_v11_controller
        mask = info["action_mask"]
        tick = int(info.get("step", 0))
        stalled = False
        if self.enable_v11_stall_override and sim is not None:
            from amr_sched.cvcr.loaded_corridor_repair import detect_stalled_picked_agvs
            stalled = bool(detect_stalled_picked_agvs(sim, stall_threshold=self.stall_threshold))
        if self.enable_v11_stall_override:
            q, ce = should_query_v11_controller(
                obs_v9,
                mask,
                tick,
                self.last_dec,
                self.epoch_interval,
                self.prev_emerg,
                in_secondary_hold=(is_in_secondary_hold(sim) if sim is not None else False),
                stalled_picked=stalled,
            )
        else:
            q, ce = should_query_v10_controller(
                obs_v9,
                mask,
                tick,
                self.last_dec,
                self.epoch_interval,
                self.prev_emerg,
                in_secondary_hold=(is_in_secondary_hold(sim) if sim is not None else False),
            )
        self.prev_emerg = ce
        if not q:
            if (
                sim is not None
                and getattr(sim, "_p4_choke_robust", False)
                and len(mask) > 1
                and mask[1] == 1
                and (tick - self.last_surgical) >= 8
            ):
                from amr_sched.cvcr.p4_belief_choke import detect_choke_oscillation_agvs

                if detect_choke_oscillation_agvs(sim):
                    self.last_surgical = tick
                    return 1
            return ACTION_KEEP
        self.last_dec = tick
        if stalled and len(mask) > 8 and mask[8] == 1:
            return 8  # Trigger full-scope + loaded corridor repair when vehicles are stalled loaded

        is_fault = float(obs_v9[18]) > 0.5
        is_urg = float(obs_v9[19]) > 0.5
        win_tight = float(obs_v9[36]) if len(obs_v9) > 36 else 1.0
        coupling = float(obs_v9[37]) if len(obs_v9) > 37 else 0.0
        congestion = float(obs_v9[9])

        if is_fault or is_urg:
            if win_tight <= 0.40 and mask[1] == 1:
                return 1  # V10_E1_FAST_RESCUE @ 20ms
            if is_urg and mask[2] == 1:
                return 2  # V10_E2_CRIT_PROD_CI @ 50ms
            if mask[3] == 1:
                return 3  # V10_E3_JOINT_PC_HYS @ 50ms
            return 1 if mask[1] == 1 else ACTION_KEEP

        # Calm strategic checkpoint: conditional CP-SAT when coupled + sufficient headroom, else L-RHO BALANCE
        if coupling >= 0.65 and win_tight >= 0.70 and mask[8] == 1:
            return 8  # V10_E7_COND_CPSAT @ 100ms
        if congestion >= 0.25 and mask[6] == 1:
            return 6  # V10_E5_AGGRESSIVE_GUIDE @ 100ms
        if mask[7] == 1:
            return 7  # V10_E6_BALANCE_KM_LRHO @ 100ms
        for cand in range(min(len(mask), NUM_EXPERT_V10_ACTIONS) - 1, 0, -1):
            if mask[cand] == 1:
                return cand
        return ACTION_KEEP
