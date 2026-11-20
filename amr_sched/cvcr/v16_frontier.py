"""V16 Unified Execution-Aware Frontier (`V16_EA_JOINT`, `V16_CALM_FAST`, `V16_BALANCED`).

Unifies three verified algorithmic regimes strictly from topology-agnostic causal signals:
  1. `V15_SOFT`: V15 BALANS/Detour-Regret/Selective-Cliff search (`C0..C4`, `h<=1`) with
     learned `v7` restraint gating and `P4.5_VoI` probe (dominates discrete `resource_windows`
     and contiguous kitting precedence bottlenecks).
  2. `V7_CALM`: Pure-NumPy `TriRegimeAnytimeNet` calm portfolio (`V7_E1..V7_E6`, `haven_v12`,
     `P4.5_VoI`), delivering minimal compute (`110..205ms`), low churn, and top performance
     under multi-axis stress and stochastic execution disturbances.
  3. `V10_GUIDED`: Coherent `V10_E7_COND_CPSAT / V10_E2 / V10_E1` + `haven_v12` portfolio
     for flat/acyclic regular transport graphs (`op_predecessors == 0`).

A Stage-R3 counterfactual contextual selector + Set-Attention candidate ranker loaded from
pure NumPy `.npz` checkpoints (`v16_frontier.npz`) picks coherent modes without toxic intra-run
queue ping-ponging or topology-name hardcoding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.checkpoints import resolve_default_checkpoint
from amr_sched.cvcr.commitment import CommitmentClassification
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.snapshot import NativeSnapshot, op_predecessors
from amr_sched.cvcr.v15_ea_joint import (
    ACTION_KEEP,
    DEFAULT_V15_CKPT,
    V15AdaptiveController,
    V15Candidate,
    V15ExecutionAwarePolicy,
    generate_v15_unified_candidates,
)
from amr_sched.network import Network

DEFAULT_V16_CKPT = resolve_default_checkpoint("v16_frontier.npz")

FEATURE_SCHEMA_VERSION = "v17_causal_v1"
V16_CTX_DIM = 20
V16_MODES: tuple[str, ...] = ("V15_SOFT", "V7_CALM", "V10_GUIDED")

_V7_BUDGET_ACT: dict[str, int] = {
    "V7_E1_FAST_RESCUE": 2,
    "V7_E2_CRIT_PROD_CI": 3,
    "V7_E3_JOINT_CI_HYS": 3,
    "V7_E4_PACER_CI": 3,
    "V7_E5_AGGRESSIVE_CI": 3,
    "V7_E6_BALANCE_KM": 5,
}
_V10_BUDGET_ACT: dict[str, int] = {
    "V10_E1_FAST_RESCUE": 2,
    "V10_E2_CRIT_PROD_CI": 3,
    "V10_E7_COND_CPSAT": 5,
}


def _relu(x: np.ndarray) -> np.ndarray:
    """Elementwise ReLU activation."""
    return np.maximum(0.0, x)


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Apply trailing-axis LayerNorm matching `torch.nn.LayerNorm`."""
    mu = float(np.mean(x))
    var = float(np.var(x))
    return ((x - mu) / float(np.sqrt(var + eps))) * weight + bias


def _dueling_q_forward(
    w: dict[str, np.ndarray],
    prefix: str,
    state: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Evaluate one `DuelingQHead` (`Linear -> LayerNorm -> ReLU -> Linear -> ReLU -> V+A`)."""
    h = state @ w[f"{prefix}.backbone.0.weight"].T + w[f"{prefix}.backbone.0.bias"]
    h = _layer_norm(h, w[f"{prefix}.backbone.1.weight"], w[f"{prefix}.backbone.1.bias"])
    h = _relu(h)
    h = _relu(h @ w[f"{prefix}.backbone.3.weight"].T + w[f"{prefix}.backbone.3.bias"])
    v = float((h @ w[f"{prefix}.val_head.weight"].T + w[f"{prefix}.val_head.bias"])[0])
    a = h @ w[f"{prefix}.adv_head.weight"].T + w[f"{prefix}.adv_head.bias"]
    m_f = np.asarray(mask, dtype=np.float32)
    legal_cnt = max(1.0, float(np.sum(m_f)))
    mean_legal_a = float(np.sum(a * m_f)) / legal_cnt
    q = v + (a - mean_legal_a)
    return q - (1.0 - m_f) * 1e8


def extract_v16_context(
    sim: Any,
    tracker: Any,
    obs_v12: np.ndarray,
    obs_v7: np.ndarray,
    *,
    tick: int = 0,
    has_exec_drift: bool = False,
    causal: bool = True,
) -> np.ndarray:
    """Extract strictly topology-free 20-D structural context vector for mode routing."""
    from amr_sched.rl.spaces import EXPERT_V12_FEATURE_NAMES

    ops = getattr(tracker, "operations", ()) if tracker is not None else ()
    n_ops = max(1, len(ops))
    has_res_win = 1.0 if any(bool(getattr(op, "resource_windows", ())) for op in ops) else 0.0
    n_preds = sum(len(op_predecessors(op)) for op in ops)
    kit_sizes = [len(getattr(op, "kit_task_ids", ())) for op in ops]
    max_kit = max(kit_sizes, default=0)
    avg_kit = float(sum(kit_sizes)) / float(n_ops)
    tool_cnt: dict[int, int] = {}
    slacks: list[float] = []
    for op in ops:
        tid_t = int(getattr(op, "tool_id", 0))
        tool_cnt[tid_t] = tool_cnt.get(tid_t, 0) + 1
        slacks.append(float(getattr(op, "due_step", 100) - getattr(op, "proc_steps", 0) - tick))
    max_tool_share = float(max(tool_cnt.values(), default=1)) / float(n_ops)
    mean_slack = float(np.mean(slacks)) / 200.0 if slacks else 0.5

    net = getattr(sim, "network", None) if sim is not None else None
    n_nodes = max(1, len(getattr(net, "nodes", {}))) if net is not None else 1
    edges = list(getattr(net, "edges", {}).values()) if net is not None else []
    n_edges = max(1, len(edges))
    edge_caps = [float(e.capacity) for e in edges] or [1.0]
    frac_single_lane = float(sum(1 for c in edge_caps if c <= 1.0)) / float(len(edge_caps))
    mean_edge_cap = float(np.mean(edge_caps)) / 2.0
    if net is not None and hasattr(net, "out_edges") and getattr(net, "nodes", {}):
        degs = [len(net.out_edges(nid)) for nid in net.nodes]
        low_deg_frac = float(sum(1 for d in degs if d <= 2)) / float(len(degs))
        mean_out_deg = (float(sum(degs)) / float(len(degs))) / 4.0
    else:
        low_deg_frac = 0.0
        mean_out_deg = 0.5

    agvs = getattr(sim, "agvs", ()) if sim is not None else ()
    n_agv = max(1, len(agvs))
    faulted = sum(1 for a in agvs if str(getattr(getattr(a, "state", ""), "value", getattr(a, "state", ""))).lower() == "faulted")
    if causal:
        fb_pkt = getattr(sim, "_execution_feedback", None) if sim is not None else None
        has_causal_fb = fb_pkt is not None and not fb_pkt.is_empty()
        causal_hist = bool(getattr(sim, "_causal_drift_observed", False)) if sim is not None else False
        p2_active = 1.0 if (has_exec_drift or has_causal_fb or causal_hist) else 0.0
        uncomp_tasks = sum(
            1
            for t in (getattr(sim, "tasks", ()) if sim is not None else ())
            if str(getattr(getattr(t, "status", ""), "value", getattr(t, "status", ""))).lower()
            not in ("completed", "failed")
            and int(getattr(t, "release_step", 0)) <= int(tick)
            and not getattr(t, "hidden_until_release", False)
        )
        idx_cwr = EXPERT_V12_FEATURE_NAMES.index("conflict_wait_rate")
        idx_dd90 = EXPERT_V12_FEATURE_NAMES.index("detour_delta_p90")
        wait_sig = float(obs_v12[idx_cwr]) if len(obs_v12) > idx_cwr else 0.0
        bay_dd_p90 = float(obs_v12[idx_dd90]) if len(obs_v12) > idx_dd90 else 0.0
    else:
        p2_active = 1.0 if (sim is not None and getattr(sim, "_perturbation_model", None) is not None) else 0.0
        uncomp_tasks = sum(
            1
            for t in (getattr(sim, "tasks", ()) if sim is not None else ())
            if str(getattr(getattr(t, "status", ""), "value", getattr(t, "status", ""))).lower() not in ("completed", "failed")
        )
        wait_sig = float(obs_v12[4]) if len(obs_v12) > 4 else 0.0
        bay_dd_p90 = float(obs_v12[40]) if len(obs_v12) > 40 else 0.0
    urgent_open = float(obs_v7[19]) if len(obs_v7) > 19 else 0.0

    return np.asarray(
        [
            has_res_win,
            min(2.0, float(n_preds) / 8.0),
            min(2.0, float(max_kit) / 4.0),
            min(2.0, avg_kit / 4.0),
            1.0 if (has_exec_drift or p2_active > 0.5) else 0.0,
            min(1.0, float(faulted) / float(n_agv)),
            p2_active,
            min(2.0, (float(uncomp_tasks) / float(n_agv)) / 4.0),
            min(2.0, wait_sig),
            min(2.0, bay_dd_p90),
            min(1.0, urgent_open),
            min(1.0, float(tick) / 200.0),
            float(np.log1p(float(n_nodes))) / 5.0,
            float(n_edges) / float(n_nodes) / 4.0,
            frac_single_lane,
            mean_edge_cap,
            low_deg_frac,
            mean_out_deg,
            max_tool_share,
            float(np.clip(mean_slack, -1.0, 3.0)),
        ],
        dtype=np.float32,
    )


class V16FrontierPolicy(V15ExecutionAwarePolicy):
    """Unified CPU/NumPy V16 policy combining V7 restraint, V16 mode router, and V15 ranker."""

    def __init__(
        self,
        weights: Optional[dict[str, np.ndarray]] = None,
        *,
        conservative_gate: float = 0.25,
        zero_keep_residual: bool = True,
        allow_probe_override: bool = False,
        feature_schema_version: str = FEATURE_SCHEMA_VERSION,
    ) -> None:
        """Initialize NumPy weights, bandit priors, and active sub-scope hint."""
        super().__init__(
            weights=weights,
            conservative_gate=conservative_gate,
            zero_keep_residual=zero_keep_residual,
            allow_probe_override=allow_probe_override,
        )
        self.active_subscope: Optional[str] = None
        self.feature_schema_version: str = str(feature_schema_version)

    @classmethod
    def from_npz(cls, path: str | Path, *, conservative_gate: float = 0.25) -> "V16FrontierPolicy":
        """Load pure-NumPy `.npz` weights into `V16FrontierPolicy` (cached by mtime)."""
        weights = _load_cached_npz_weights(Path(path))
        return cls(weights=weights, conservative_gate=conservative_gate)

    def has_v7_restraint_weights(self) -> bool:
        """Return True iff bundled `TriRegimeAnytimeNet` restraint tensors exist in `self.w`."""
        return self.w is not None and "v7.opt_net.backbone.0.weight" in self.w

    def eval_v7_restraint(self, obs30: np.ndarray, mask9: np.ndarray) -> tuple[int, float, float]:
        """Run exact pure-NumPy `TriRegimeAnytimeNet` inference returning `(act, q_act, q_keep)`."""
        if not self.has_v7_restraint_weights() or self.w is None:
            legal = np.where(mask9 == 1)[0]
            fallback = int(legal[1]) if len(legal) > 1 else ACTION_KEEP
            return fallback, 0.0, 0.0
        o = np.asarray(obs30[:30], dtype=np.float32)
        if float(o[18]) > 0.5:
            prefix = "v7.fault_net"
        elif float(o[19]) > 0.5:
            prefix = "v7.urgent_net"
        else:
            prefix = "v7.opt_net"
        qvals = _dueling_q_forward(self.w, prefix, o, mask9[:9])
        act = int(np.argmax(qvals))
        return act, float(qvals[act]), float(qvals[0])

    def select_portfolio_mode(self, ctx14: np.ndarray, *, profile: str = "quality") -> str:
        """Select coherent portfolio mode from causal structural context without topology names."""
        has_res_win = float(ctx14[0]) > 0.5
        if has_res_win:
            return "V15_SOFT"
        if profile == "balanced":
            return "V15_SOFT"
        if profile == "calm":
            return "V7_CALM"

        # Learned Stage-R3 counterfactual contextual selector if present
        if self.w is not None and "mode_w1" in self.w:
            x = np.clip((ctx14[:V16_CTX_DIM] - self.w["m_mu"]) / np.maximum(0.05, self.w["m_sd"]), -6.0, 6.0)
            h = _relu(x @ self.w["mode_w1"] + self.w["mode_b1"])
            logits = h @ self.w["mode_w2"] + self.w["mode_b2"]
            return V16_MODES[int(np.argmax(logits))]

        # Calibrated causal fallback prior (strictly from structural signals)
        n_preds_norm = float(ctx14[1])
        max_kit_norm = float(ctx14[2])
        has_drift = float(ctx14[4]) > 0.5
        fault_ratio = float(ctx14[5])
        tasks_per_agv = float(ctx14[7])
        bay_dd_p90 = float(ctx14[9])
        if n_preds_norm <= 0.05 and not has_drift:
            return "V10_GUIDED"
        if has_drift or max_kit_norm >= 0.75 or bay_dd_p90 > 0.15 or fault_ratio > 0.01 or tasks_per_agv >= 0.85:
            return "V7_CALM"
        return "V15_SOFT"


_V16_NPZ_WEIGHTS_CACHE: dict[tuple[str, int], dict[str, np.ndarray]] = {}


def _load_cached_npz_weights(path: Path) -> dict[str, np.ndarray]:
    """Load and cache read-only `.npz` tensor dict keyed by resolved path and mtime_ns."""
    resolved = path.resolve()
    key = (str(resolved), int(resolved.stat().st_mtime_ns))
    cached = _V16_NPZ_WEIGHTS_CACHE.get(key)
    if cached is not None:
        return cached
    with np.load(str(resolved), allow_pickle=False) as data:
        weights = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
    for arr in weights.values():
        arr.setflags(write=False)
    _V16_NPZ_WEIGHTS_CACHE[key] = weights
    return weights


def load_v16_frontier_policy(path: Optional[str | Path] = None) -> V16FrontierPolicy:
    """Load `V16FrontierPolicy` from `v16_frontier.npz`, falling back cleanly to `v15_ea_joint.npz`."""
    p = Path(path) if path is not None else DEFAULT_V16_CKPT
    if p.exists():
        return V16FrontierPolicy.from_npz(p)
    if DEFAULT_V15_CKPT.exists():
        return V16FrontierPolicy.from_npz(DEFAULT_V15_CKPT)
    return V16FrontierPolicy(weights=None)


def generate_v16_unified_candidates(
    policy_owner: Any,
    network: Network,
    snapshot: NativeSnapshot,
    clf: CommitmentClassification,
    evaluator: CVCREvaluator,
    surr: Any,
    base_assign: dict[str, str],
    base_bays: dict[str, str],
    base_crews: dict[str, int],
    base_eval: EvalResult,
    base_prio: dict[str, int],
    healthy_robots: list[str],
    bay_delta: dict[str, float],
    tw_seed: dict[tuple[str, str], int],
    v15_pol: V15ExecutionAwarePolicy,
    deadline: float,
    budget_ms: float,
    now_fn: Any,
) -> tuple[list[V15Candidate], np.ndarray]:
    """Generate clean shared-budget V16 candidates (`C0..C4`) without budget halving."""
    return generate_v15_unified_candidates(
        policy_owner,
        network,
        snapshot,
        clf,
        evaluator,
        surr,
        base_assign,
        base_bays,
        base_crews,
        base_eval,
        base_prio,
        healthy_robots,
        bay_delta,
        tw_seed,
        v15_pol,
        deadline,
        budget_ms,
        now_fn,
    )


class V16AdaptiveController(V15AdaptiveController):
    """Unified V16 controller orchestrating `v7` restraint, causal mode routing, and safety guards."""

    def __init__(
        self,
        *,
        epoch_interval: int = 16,
        stall_threshold: int = 20,
        policy: Optional[V16FrontierPolicy] = None,
        trigger: Optional[Any] = None,
        profile: str = "quality",
    ) -> None:
        """Initialize V16 controller with profile (`quality`, `calm`, `balanced`) and sub-state."""
        if policy is None:
            pol: V16FrontierPolicy = load_v16_frontier_policy()
        elif isinstance(policy, V16FrontierPolicy):
            pol = policy
        else:
            w = dict(policy.w) if getattr(policy, "w", None) is not None else {}
            if "v7.opt_net.backbone.0.weight" not in w and DEFAULT_V16_CKPT.exists():
                base_v16 = V16FrontierPolicy.from_npz(DEFAULT_V16_CKPT)
                if base_v16.w is not None:
                    for k, val in base_v16.w.items():
                        w.setdefault(k, val)
            pol = V16FrontierPolicy(weights=w or None, conservative_gate=getattr(policy, "conservative_gate", 0.25))
        super().__init__(
            epoch_interval=epoch_interval,
            stall_threshold=stall_threshold,
            policy=pol,
            trigger=trigger,
        )
        self.profile = str(profile)
        self.v7_last_dec: int = -999
        self.v7_prev_emerg: tuple[bool, bool] = (False, False)
        self.v10_last_dec: int = -999
        self.v10_prev_emerg: tuple[bool, bool] = (False, False)
        self.locked_mode: Optional[str] = None
        self.decision_count: int = 0

    def reset(self) -> None:
        """Reset episodic trigger cooldowns, emergency trackers, and coherent mode latch."""
        super().reset()
        self.v7_last_dec = -999
        self.v7_prev_emerg = (False, False)
        self.v10_last_dec = -999
        self.v10_prev_emerg = (False, False)
        self.locked_mode = None
        self.decision_count = 0
        if isinstance(self.policy, V16FrontierPolicy):
            self.policy.active_subscope = None

    def _map_budget_to_action(self, target_budget_ms: float, mask: np.ndarray) -> int:
        """Map requested solver budget `{20, 50, 100}ms` to legal V12 portfolio action index."""
        pref = 2 if target_budget_ms <= 25.0 else (3 if target_budget_ms <= 65.0 else 5)
        if len(mask) > pref and mask[pref] == 1:
            return pref
        for idx in (3, 5, 2, 1):
            if len(mask) > idx and mask[idx] == 1:
                return idx
        return ACTION_KEEP

    def select_action(self, obs_v12: np.ndarray, info: dict[str, Any], sim: Any = None) -> int:
        """Choose V16 action causally with coherent mode latching, `v7` restraint, and safety guards."""
        from amr_sched.rl.spaces import (
            EXPERT_V10_SPECS,
            EXPERT_V7_SPECS,
            compute_expert_v10_action_mask,
            compute_expert_v7_action_mask,
            compute_expert_v7_state,
            compute_expert_v9_state,
            is_in_secondary_hold,
            should_query_expert_controller,
            should_query_v10_controller,
        )

        mask = info["action_mask"]
        tick = int(info.get("step", 0))
        if sim is not None and is_in_secondary_hold(sim):
            return ACTION_KEEP

        pol = self.policy
        if not isinstance(pol, V16FrontierPolicy) or sim is None:
            if isinstance(pol, V16FrontierPolicy):
                pol.active_subscope = None
            return super().select_action(obs_v12, info, sim=sim)

        tracker = getattr(sim, "_prod_tracker", None)
        env_last_replan = int(getattr(sim, "_env_last_replan_tick", max(0, self.last_dec)))
        recent_c = int(getattr(sim, "_env_recent_conflicts", 0))
        dec_logs = getattr(sim, "_env_decision_logs", None)
        obs_v7 = compute_expert_v7_state(
            sim,
            tracker,
            obs_v12[:22],
            last_replan_tick=env_last_replan,
            recent_conflict_ticks=recent_c,
            decision_logs=dec_logs,
        )
        m7 = compute_expert_v7_action_mask(sim, last_replan_tick=env_last_replan)
        fb_pkt = getattr(sim, "_execution_feedback", None)
        has_drift = bool(fb_pkt is not None and not fb_pkt.is_empty())
        ctx14 = extract_v16_context(sim, tracker, obs_v12, obs_v7, tick=tick, has_exec_drift=has_drift)

        has_res_win = float(ctx14[0]) > 0.5
        if has_res_win:
            mode = "V15_SOFT"
        elif self.locked_mode is not None:
            # Allow coherent upgrade to V7_CALM only once if runtime disturbance emerges later
            if has_drift and self.profile != "balanced" and self.locked_mode != "V7_CALM":
                self.locked_mode = "V7_CALM"
            mode = self.locked_mode
        else:
            mode = pol.select_portfolio_mode(ctx14, profile=self.profile)
            self.locked_mode = mode

        if mode == "V15_SOFT":
            pol.active_subscope = None
            c_act = super().select_action(obs_v12, info, sim=sim)
            if c_act in (0, 1) or tick <= 4 or has_drift:
                return c_act
            # Restraint filter via learned v7 Q-head when no imminent resource window closes
            ops = getattr(tracker, "operations", ()) if tracker is not None else ()
            imminent = any(
                (not op.started) and any(0 <= int(we) - tick <= 20 for _, we in getattr(op, "resource_windows", ()))
                for op in ops
            )
            if imminent or (not getattr(pol, "enable_restraint", True)) or (not pol.has_v7_restraint_weights()):
                return c_act
            q7, self.v7_prev_emerg = should_query_expert_controller(
                obs_v7, m7, tick, self.v7_last_dec, 12, self.v7_prev_emerg, is_in_secondary_hold(sim)
            )
            if not q7:
                return ACTION_KEEP
            self.v7_last_dec = tick
            v7_a, _, _ = pol.eval_v7_restraint(obs_v7, m7)
            if v7_a == 0:
                return ACTION_KEEP
            return self._map_budget_to_action(50.0 if v7_a <= 5 else 100.0, mask)

        if mode == "V7_CALM":
            q7, self.v7_prev_emerg = should_query_expert_controller(
                obs_v7, m7, tick, self.v7_last_dec, 12, self.v7_prev_emerg, is_in_secondary_hold(sim)
            )
            if not q7:
                return ACTION_KEEP
            self.v7_last_dec = tick
            v7_a, _, _ = pol.eval_v7_restraint(obs_v7, m7)
            if v7_a == 0:
                return ACTION_KEEP
            sc_v7, bms_v7, _ = EXPERT_V7_SPECS[v7_a]
            pol.active_subscope = sc_v7 or "V7_E3_JOINT_CI_HYS"
            return self._map_budget_to_action(bms_v7, mask)

        # V10_GUIDED mode (for acyclic unwindowed regular transport topologies)
        o9 = compute_expert_v9_state(
            sim,
            tracker,
            last_replan_tick=env_last_replan,
            recent_conflict_ticks=recent_c,
            decision_logs=dec_logs,
        )
        m10 = compute_expert_v10_action_mask(sim, last_replan_tick=env_last_replan)
        q10, self.v10_prev_emerg = should_query_v10_controller(
            o9, m10, tick, self.v10_last_dec, 16, self.v10_prev_emerg, is_in_secondary_hold(sim)
        )
        if not q10:
            return ACTION_KEEP
        self.v10_last_dec = tick
        # Replicate RuleV10PortfolioPolicy choice cleanly across {V10_E7, V10_E2, V10_E1}
        has_f = float(o9[18]) > 0.5
        has_u = float(o9[19]) > 0.5
        act10 = 1 if (has_f and not has_u and m10[1] == 1) else (7 if m10[7] == 1 else (3 if m10[3] == 1 else 0))
        if act10 == 0:
            return ACTION_KEEP
        sc_v10, bms_v10, _ = EXPERT_V10_SPECS[act10]
        pol.active_subscope = sc_v10 or "V10_E7_COND_CPSAT"
        return self._map_budget_to_action(bms_v10, mask)
