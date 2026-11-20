"""V12 Adaptive Meta-Controllers: When-to-Replan Trigger, Online BALANS MAB & MP-ASIL RL.

Implements three clean, causal controllers over the V12 9-action portfolio:
  1. `WhenToReplanTrigger` (ICRA 2024 POMDP Trigger + NeurIPS 2023 RDM):
     Separates zero-churn surgical Haven/TPG corridor repair (`action=1`) from global
     LNS/CP replanning triggers, optionally refined by trained net-benefit `.npz` weights.
  2. `RuleV12ProactivePolicy`:
     Deterministic rule controller adding surgical corridor/leaf-spur handling, Detour-Delta
     scoring, HeteroDestroy selection, and safe CP-SAT headroom gating.
  3. `BalansOnlineController` (IJCAI 2025 BALANS):
     Discounted Normal-Gamma Thompson Sampling / UCB online multi-armed bandit adapting
     destroy radius and search operators live within every episode from observed net return
     (`-delta_wait - 15*churn - 0.05*compute_ms + stall_recovery`), eliminating opening
     selector drift on unseen topologies and multi-axis stress.
  4. `RLV12MPASILController` (NeurIPS 2025 MP-ASIL + Joint Trigger):
     Lightweight PyTorch multi-head regime-specialized policy (`3` soft-gated heads over
     `44-D` state + E2G dual graph) trained via adaptive self-imitation + advantage weighting
     with `KEEP` vs `SURGICAL` vs `DESTROY` vs `SAFE_CP` full decision control.
"""

from __future__ import annotations

import math
from pathlib import Path
import random
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.loaded_corridor_repair import detect_stalled_picked_agvs
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
from amr_sched.rl.spaces import (
    ACTION_KEEP,
    EXPERT_V12_ACTION_NAMES,
    EXPERT_V12_STATE_DIM,
    NUM_EXPERT_V12_ACTIONS,
    V9_AGV_FEAT_DIM,
    V9_OP_FEAT_DIM,
    is_in_secondary_hold,
    should_query_v10_controller,
)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _stagnation_break_action(mask: np.ndarray, tick: int) -> Optional[int]:
    """Diversify non-CP LNS/TPG operators only after abnormal long-horizon stalls (`tick >= 360`)."""
    if tick < 360:
        return None
    cycle = (3, 6, 7, 5)
    start_idx = (tick // 16) % len(cycle)
    for off in range(len(cycle)):
        cand = cycle[(start_idx + off) % len(cycle)]
        if cand < len(mask) and mask[cand] == 1:
            return cand
    return None


class WhenToReplanTrigger:
    """ICRA 2024 + RDM causal replanning & surgical-repair trigger gate."""

    def __init__(
        self,
        epoch_interval: int = 16,
        *,
        stall_threshold: int = 8,
        surgical_cooldown: int = 12,
        weights: Optional[dict[str, np.ndarray]] = None,
        min_net_benefit: float = 0.0,
        suppress_calm_heuristic: bool = True,
    ) -> None:
        self.epoch_interval = int(epoch_interval)
        self.stall_threshold = int(stall_threshold)
        self.surgical_cooldown = int(surgical_cooldown)
        self.w = weights
        self.min_net_benefit = float(min_net_benefit)
        self.suppress_calm_heuristic = bool(suppress_calm_heuristic)
        self.last_dec = -999
        self.last_surgical = -999
        self.prev_emerg: tuple[bool, bool] = (False, False)
        self._consec_npz_skips = 0
        self._last_p2_global_key: Any = None
        self._last_p2_surg_key: Any = None

    @classmethod
    def from_npz(cls, path: str | Path, **kwargs: Any) -> "WhenToReplanTrigger":
        """Load trained net-benefit trigger MLP from `.npz` weights."""
        data = np.load(str(path), allow_pickle=False)
        w = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
        return cls(weights=w, **kwargs)

    def reset(self) -> None:
        self.last_dec = -999
        self.last_surgical = -999
        self.prev_emerg = (False, False)
        self._consec_npz_skips = 0
        self._last_p2_global_key = None
        self._last_p2_surg_key = None

    def predict_global_net_benefit(self, obs_v12: np.ndarray) -> float:
        """Predict expected NetObj improvement of firing a global LNS replan vs KEEP."""
        if self.w is None:
            # Calibrated causal surrogate net-benefit score
            cong = float(obs_v12[9])
            starv = float(obs_v12[8])
            late = float(obs_v12[32]) if len(obs_v12) > 32 else 0.0
            drift = float(obs_v12[30]) if len(obs_v12) > 30 else 0.0
            dd_p90 = float(obs_v12[41]) if len(obs_v12) > 41 else 0.0
            comm = float(obs_v12[2])
            return 25.0 * (0.8 * starv + 0.7 * cong + 0.9 * late + 0.6 * drift + 0.5 * dd_p90) - 22.0 * comm
        w = self.w
        x = (obs_v12[: len(w["mu"])] - w["mu"]) / np.maximum(1e-6, w["sd"])
        h = _relu(x @ w["w1"] + w["b1"])
        h = _relu(h @ w["w2"] + w["b2"])
        return float((h @ w["w3"] + w["b3"]).reshape(-1)[0])

    def should_query(
        self,
        obs_v12: np.ndarray,
        mask: np.ndarray,
        tick: int,
        sim: Any = None,
    ) -> tuple[str, bool]:
        """Decide trigger mode: `("none" | "surgical" | "global", has_stall_or_leaf)`.

        Separates zero-churn surgical Haven/TPG action (`action=1`) on loaded stalls
        or occupied dead-end leaf conflicts from global LNS triggers so narrow corridors
        never force unnecessary global `E7 CP-SAT` churn.
        """
        in_hold = is_in_secondary_hold(sim) if sim is not None else False
        if in_hold:
            return "none", False
        if not any(mask[idx] == 1 for idx in range(1, min(len(mask), NUM_EXPERT_V12_ACTIONS))):
            return "none", False

        cur_f = bool(float(obs_v12[18]) > 0.5)
        cur_u = bool(float(obs_v12[19]) > 0.5)
        prev_f, prev_u = self.prev_emerg
        self.prev_emerg = (cur_f, cur_u)

        stalled_any = False
        secondary_ready = False
        if sim is not None:
            stalled_any = bool(
                detect_stalled_picked_agvs(
                    sim,
                    stall_threshold=self.stall_threshold,
                    include_edge_gate_stalls=True,
                )
            )
            fset = getattr(sim, "_faulted_agvs", set())
            if fset:
                secondary_ready = any(
                    getattr(t, "window_missed", False)
                    and tick >= int(getattr(t, "secondary_window_open_step", 999999))
                    and getattr(t, "assigned_agv", None) in fset
                    and getattr(getattr(t, "status", ""), "value", str(getattr(t, "status", ""))).lower() not in ("completed", "failed")
                    for t in getattr(sim, "tasks", [])
                )
        leaf_haz = len(obs_v12) > 39 and float(obs_v12[39]) > 0.12
        gate_stall = len(obs_v12) > 38 and float(obs_v12[38]) > 0.01
        eff_unassigned = 0.0 if (float(obs_v12[0]) >= 0.95 and not cur_f) else float(obs_v12[10])
        starv = float(obs_v12[8])

        # 1. Episode kickoff, open secondary fault window, or immediate rising disturbance -> global closure
        if self.last_dec < 0 and tick <= 8:
            self.last_dec = tick
            self._consec_npz_skips = 0
            return "global", stalled_any
        if secondary_ready:
            self.last_dec = tick
            self._consec_npz_skips = 0
            return "global", stalled_any
        if ((cur_f and not prev_f) or (cur_u and not prev_u)) and float(obs_v12[16]) < 0.5:
            self.last_dec = tick
            self._consec_npz_skips = 0
            return "global", stalled_any

        # 1b. P2 Execution-Aware wakeup (strictly disabled in Execution-Blind mode)
        if sim is not None and getattr(sim, "_p2_execution_aware", False) and getattr(sim, "_perturbation_model", None) is not None:
            p2_sig = sim._perturbation_model.active_disturbance_sig(sim, tick)
            fail_assg = tuple(p2_sig.get("fail_assigned_agvs") or ())
            if fail_assg:
                if fail_assg != self._last_p2_global_key and (self.last_dec < 0 or (tick - self.last_dec) >= 6):
                    self._last_p2_global_key = fail_assg
                    self.last_dec = tick
                    self._consec_npz_skips = 0
                    return "global", stalled_any
            else:
                self._last_p2_global_key = None
            blk_p = tuple(p2_sig.get("blocked_picked_agvs") or ())
            blk_a = tuple(p2_sig.get("blocked_assigned_agvs") or ())
            blk_e = tuple(p2_sig.get("blocked_edges") or ())
            if (blk_p or blk_a) and len(mask) > 1 and mask[1] == 1:
                s_key = (blk_p, blk_a, blk_e)
                if s_key != self._last_p2_surg_key and (self.last_surgical < 0 or (tick - self.last_surgical) >= 4):
                    self._last_p2_surg_key = s_key
                    self.last_surgical = tick
                    return "surgical", True
            else:
                self._last_p2_surg_key = None

        # 1c. P4 Choke-Robust causal ping-pong / loaded_yield wakeup (never muted by remote cur_f or calm comm>=0.85)
        if sim is not None and getattr(sim, "_p4_choke_robust", False) and len(mask) > 1 and mask[1] == 1:
            from amr_sched.cvcr.p4_belief_choke import detect_choke_oscillation_agvs
            if detect_choke_oscillation_agvs(sim):
                if self.last_surgical < 0 or (tick - self.last_surgical) >= min(self.surgical_cooldown, 8):
                    self.last_surgical = tick
                    return "surgical", True

        # 2. Surgical zero-task-churn repair for corridor stalls or occupied leaf spurs
        #    Do not let repeated surgical calls starve overdue global replanning when unassigned/starved backlog exists.
        overdue_backlog = (tick - self.last_dec) >= (self.epoch_interval + 4) and (eff_unassigned > 0.01 or starv >= 0.25 or tick >= 260)
        if not cur_f and not overdue_backlog and (stalled_any or gate_stall or leaf_haz) and len(mask) > 1 and mask[1] == 1:
            if self.last_surgical < 0 or (tick - self.last_surgical) >= self.surgical_cooldown:
                self.last_surgical = tick
                return "surgical", True

        # 3. Anti-spam guard ONLY for already-addressed non-fault urgencies (cur_u) without backlog or tight windows;
        #    never dampen open faults (`cur_f`) which can silently miss primary/secondary rescue deadlines!
        win_tight = float(obs_v12[36]) if len(obs_v12) > 36 else 1.0
        cong = float(obs_v12[9])
        if (
            (not cur_f)
            and cur_u
            and eff_unassigned <= 0.14
            and win_tight > 0.58
            and starv < 0.16
            and cong < 0.18
            and float(obs_v12[16]) < 0.25
            and not stalled_any
            and not gate_stall
            and not (cur_u and not prev_u)
            and tick < 250
        ):
            if (tick - self.last_dec) < max(self.epoch_interval + 4, 20):
                return "none", stalled_any
            if self.suppress_calm_heuristic and float(obs_v12[11]) >= 0.70 and float(obs_v12[2]) >= 0.72:
                return "none", stalled_any

        # 4. Global RCDR / trained When-to-Replan net-benefit gate
        # Rescue ONLY genuine post-plateau fleet stranding (~>=45% AGVs idle for >=32 ticks with >=15% unassigned tasks)
        stranded_backlog = (
            eff_unassigned >= 0.15
            and float(obs_v12[0]) <= 0.55
            and tick >= 220
            and (tick - self.last_dec) >= max(32, self.epoch_interval * 2)
        )
        q_v10, _ = should_query_v10_controller(
            obs_v12[:38],
            mask,
            tick,
            self.last_dec,
            epoch_interval=self.epoch_interval,
            prev_emerg=(prev_f, prev_u),
            in_secondary_hold=False,
        )
        if not q_v10 and not stranded_backlog:
            return "none", stalled_any
        # High-commitment calm-plateau suppression only when zero effective unassigned tasks & genuinely smooth execution
        if self.suppress_calm_heuristic and not (cur_f or cur_u) and tick >= 24 and tick < 240:
            comm = float(obs_v12[2])
            cong = float(obs_v12[9])
            drift = float(obs_v12[30]) if len(obs_v12) > 30 else 0.0
            dd_p90 = float(obs_v12[41]) if len(obs_v12) > 41 else 0.0
            if eff_unassigned <= 0.01 and comm >= 0.58 and starv < 0.20 and cong < 0.16 and drift < 0.42 and dd_p90 < 0.40:
                return "none", stalled_any
        if self.w is not None and not (cur_f or cur_u):
            # Bounded NPZ suppression: max 2 consecutive epoch skips and never when backlog/late horizon builds
            can_skip = (
                self._consec_npz_skips < 2
                and tick < 240
                and eff_unassigned <= 0.01
                and starv < 0.22
                and float(obs_v12[9]) < 0.28
                and float(obs_v12[12]) <= 0.90
            )
            if can_skip and self.predict_global_net_benefit(obs_v12) < self.min_net_benefit:
                self._consec_npz_skips += 1
                return "none", stalled_any
        self._consec_npz_skips = 0
        self.last_dec = tick
        return "global", stalled_any


class RuleV12ProactivePolicy:
    """Deterministic V12 proactive controller (no bandit, isolates Detour-Delta + HeteroDestroy + Haven/Safe-CP)."""

    def __init__(
        self,
        epoch_interval: int = 16,
        *,
        stall_threshold: int = 8,
        trigger: Optional[WhenToReplanTrigger] = None,
    ) -> None:
        self.trigger = trigger or WhenToReplanTrigger(
            epoch_interval=epoch_interval,
            stall_threshold=stall_threshold,
        )

    def reset(self) -> None:
        self.trigger.reset()

    def select_action(self, obs_v12: np.ndarray, info: dict[str, Any], sim: Any = None) -> int:
        mask = info["action_mask"]
        tick = int(info.get("step", 0))
        q_mode, _ = self.trigger.should_query(obs_v12, mask, tick, sim=sim)
        if q_mode == "none":
            return ACTION_KEEP
        if q_mode == "surgical" and len(mask) > 1 and mask[1] == 1:
            return 1  # V12_SURGICAL_TPG_REPAIR_15MS (0 task-assignment churn)

        is_fault = float(obs_v12[18]) > 0.5
        is_urg = float(obs_v12[19]) > 0.5
        win_tight = float(obs_v12[36]) if len(obs_v12) > 36 else 1.0
        coupling = float(obs_v12[37]) if len(obs_v12) > 37 else 0.0
        prog = float(obs_v12[11])
        cong = float(obs_v12[9])
        gate_stall = float(obs_v12[38]) if len(obs_v12) > 38 else 0.0
        cp_headroom = float(obs_v12[43]) if len(obs_v12) > 43 else 0.0

        if is_fault:
            if win_tight <= 0.52 and mask[2] == 1:
                return 2  # V12_RESCUE_DETOUR_FAST_20MS
            if win_tight >= 0.65 and cp_headroom >= 0.35 and mask[8] == 1:
                return 8  # V12_SAFE_CPSAT_JOINT_100MS
            if mask[3] == 1:
                return 3  # V12_CRIT_PROD_DESTROY_S_50MS
            return 2 if mask[2] == 1 else ACTION_KEEP

        if is_urg:
            if win_tight <= 0.35 and mask[2] == 1:
                return 2  # V12_RESCUE_DETOUR_FAST_20MS
            if win_tight >= 0.65 and cp_headroom >= 0.40 and mask[8] == 1:
                return 8  # V12_SAFE_CPSAT_JOINT_100MS
            if mask[3] == 1:
                return 3  # V12_CRIT_PROD_DESTROY_S_50MS
            if mask[4] == 1:
                return 4  # V12_PACER_CONSERVATIVE_50MS
            return 2 if mask[2] == 1 else ACTION_KEEP

        stag = _stagnation_break_action(mask, tick)
        if stag is not None:
            return stag

        if win_tight >= 0.56 and cp_headroom >= 0.34 and (coupling >= 0.45 or prog < 0.72) and mask[8] == 1:
            return 8  # V12_SAFE_CPSAT_JOINT_100MS
        if (cong >= 0.36 or gate_stall > 0.01) and mask[7] == 1:
            return 7  # V12_AGGRESSIVE_GUIDE_TPG_100MS (only under real congestion or choke stall)
        if float(obs_v12[8]) >= 0.28 and mask[6] == 1:
            return 6  # V12_BALANCED_DESTROY_L_100MS
        if mask[5] == 1:
            return 5  # V12_BALANCED_DESTROY_M_100MS
        for cand in range(len(mask) - 1, 0, -1):
            if mask[cand] == 1:
                return cand
        return ACTION_KEEP


class BalansOnlineController:
    """IJCAI 2025 BALANS Online Discounted Thompson Sampling over V12 Destroy & Search Arms.

    Maintains separate discounted statistics per arm within every episode and adapts
    on the fly from realized net-cost signals (`improved_delta`, `step_churn`,
    `compute_ms`, blocked reduction, and `window_expired`), avoiding single-algorithm
    overfitting on unseen topologies or abrupt multi-axis stress.
    """

    ARMS = (2, 3, 4, 5, 6, 7, 8)

    def __init__(
        self,
        epoch_interval: int = 16,
        *,
        stall_threshold: int = 8,
        discount: float = 0.78,
        seed: int = 0,
        trigger: Optional[WhenToReplanTrigger] = None,
        contextual_priors: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        self.trigger = trigger or WhenToReplanTrigger(
            epoch_interval=epoch_interval,
            stall_threshold=stall_threshold,
        )
        self.discount = float(discount)
        self.base_seed = int(seed)
        self.priors = contextual_priors or {}
        self.reset()

    def reset(self, episode_seed: Optional[int] = None) -> None:
        self.trigger.reset()
        sd = self.base_seed if episode_seed is None else int(episode_seed)
        self.rng = random.Random(sd + 1013)
        # Discounted Bayesian Gaussian reward estimates (mean/count) per arm
        self.arm_n: dict[int, float] = {a: 1.2 for a in self.ARMS}
        self.arm_mean: dict[int, float] = {
            2: 0.12,
            3: 0.30,   # CritProd priority LNS
            4: 0.28,   # Pacer conservative LNS
            5: 0.26,   # HeteroDestroy M
            6: 0.22,   # HeteroDestroy L
            7: 0.18,   # Aggressive TPG guide (scoped to high congestion)
            8: 0.34,   # Safe CP-SAT joint closure
        }
        self.arm_m2: dict[int, float] = {a: 0.25 for a in self.ARMS}
        self._pending_arm: Optional[int] = None
        self._pre_obs: Optional[np.ndarray] = None

    def _context_bonus(self, arm: int, obs_v12: np.ndarray) -> float:
        """Lightweight causal contextual prior boost reflecting structural fit."""
        is_emerg = (float(obs_v12[18]) > 0.5) or (float(obs_v12[19]) > 0.5)
        win_tight = float(obs_v12[36]) if len(obs_v12) > 36 else 1.0
        coupling = float(obs_v12[37]) if len(obs_v12) > 37 else 0.0
        prog = float(obs_v12[11])
        cong = float(obs_v12[9])
        gate_stall = float(obs_v12[38]) if len(obs_v12) > 38 else 0.0
        dd_p90 = float(obs_v12[41]) if len(obs_v12) > 41 else 0.0
        cp_headroom = float(obs_v12[43]) if len(obs_v12) > 43 else 0.0
        starv = float(obs_v12[8])

        if self.priors and "W" in self.priors:
            # Blend Train-fit linear contextual prior with lightweight structural headroom guard
            feat = obs_v12[: min(len(obs_v12), self.priors["W"].shape[0])]
            logits = feat @ self.priors["W"][: len(feat)] + self.priors["b"]
            idx = self.ARMS.index(arm)
            score = float(np.clip(0.45 * logits[idx], -0.6, 0.6))
            if arm == 8 and win_tight >= 0.55 and cp_headroom >= 0.38 and not is_emerg:
                score += 0.25
            if arm == 7 and cong < 0.30 and gate_stall <= 0.01:
                score -= 0.22
            return score

        if is_emerg:
            if win_tight <= 0.45:
                return 0.45 if arm == 2 else (-0.45 if arm >= 6 else 0.15)
            return 0.36 if arm in (3, 4) else (0.18 if arm == 5 else -0.20)

        if arm == 8:
            if win_tight >= 0.55 and cp_headroom >= 0.38 and (coupling >= 0.50 or prog < 0.72):
                return 0.32
            return -0.28
        if arm == 7:
            return 0.24 if (cong >= 0.32 or gate_stall > 0.01 or dd_p90 >= 0.55) else -0.22
        if arm == 6:
            return 0.18 if starv >= 0.28 else -0.05
        if arm == 5:
            return 0.18
        if arm == 4:
            return 0.22 if float(obs_v12[2]) >= 0.50 else 0.08
        if arm == 3:
            return 0.26 if float(obs_v12[2]) >= 0.45 else 0.10
        return 0.0

    def select_action(self, obs_v12: np.ndarray, info: dict[str, Any], sim: Any = None) -> int:
        mask = info["action_mask"]
        tick = int(info.get("step", 0))
        q_mode, _ = self.trigger.should_query(obs_v12, mask, tick, sim=sim)
        if q_mode == "none":
            self._pending_arm = None
            return ACTION_KEEP
        if q_mode == "surgical" and len(mask) > 1 and mask[1] == 1:
            self._pending_arm = None
            return 1

        legal_arms = [a for a in self.ARMS if a < len(mask) and mask[a] == 1]
        if not legal_arms:
            self._pending_arm = None
            return ACTION_KEEP

        is_emerg = (float(obs_v12[18]) > 0.5) or (float(obs_v12[19]) > 0.5)
        win_tight = float(obs_v12[36]) if len(obs_v12) > 36 else 1.0
        if is_emerg and win_tight <= 0.52 and 2 in legal_arms:
            chosen = 2
            best_score = 1.0
        elif (stag := _stagnation_break_action(mask, tick)) is not None and stag in legal_arms:
            chosen = stag
        else:
            best_score = -1e9
            chosen = legal_arms[0]
            for a in legal_arms:
                n_a = max(0.5, self.arm_n[a])
                std = math.sqrt(max(0.04, self.arm_m2[a]) / n_a)
                sample = self.arm_mean[a] + std * self.rng.gauss(0.0, 0.22)
                score = sample + self._context_bonus(a, obs_v12)
                if score > best_score:
                    best_score = score
                    chosen = a

        self._pending_arm = chosen
        self._pre_obs = obs_v12.copy()
        return chosen

    def observe_outcome(self, step_info: dict[str, Any], next_obs_v12: Optional[np.ndarray] = None) -> None:
        """Update discounted online MAB posterior after an armed step completes."""
        arm = self._pending_arm
        self._pending_arm = None
        if arm is None or arm not in self.arm_n:
            return
        churn = float(step_info.get("churn", 0.0))
        cms = float(step_info.get("compute_ms", 0.0))
        imp = float(step_info.get("improved_delta", 0.0))
        expired = bool(step_info.get("window_expired", False))
        accepted = bool(step_info.get("receipt_accepted", True))

        if expired or not accepted:
            r = -1.2
        else:
            net_gain = (imp - 15.0 * churn - 0.05 * cms) / 80.0
            blocked_bonus = 0.0
            if self._pre_obs is not None and next_obs_v12 is not None and len(next_obs_v12) > 42:
                blocked_bonus = 0.6 * (float(self._pre_obs[42]) - float(next_obs_v12[42]))
                blocked_bonus += 0.4 * (float(self._pre_obs[39]) - float(next_obs_v12[39]))
            r = float(np.clip(net_gain + blocked_bonus, -1.5, 1.5))

        for k in self.ARMS:
            self.arm_n[k] = max(0.25, self.arm_n[k] * self.discount)
        n_new = self.arm_n[arm] + 1.0
        delta = r - self.arm_mean[arm]
        self.arm_mean[arm] += delta / n_new
        self.arm_m2[arm] = max(0.04, 0.85 * self.arm_m2[arm] + delta * (r - self.arm_mean[arm]))
        self.arm_n[arm] = n_new


def build_mpasil_v12_model():
    """Construct lightweight NeurIPS'25 MP-ASIL + ICRA'24 Joint Trigger Network for V12 (torch)."""
    import torch
    import torch.nn as nn

    class MPASILNet(nn.Module):
        def __init__(
            self,
            global_dim: int = EXPERT_V12_STATE_DIM,
            op_dim: int = V9_OP_FEAT_DIM,
            agv_dim: int = V9_AGV_FEAT_DIM,
            action_dim: int = NUM_EXPERT_V12_ACTIONS,
            num_policies: int = 3,
        ) -> None:
            super().__init__()
            self.g_enc = nn.Sequential(nn.Linear(global_dim, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU())
            self.op_enc = nn.Sequential(nn.Linear(op_dim, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU())
            self.agv_enc = nn.Sequential(nn.Linear(agv_dim, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU())
            ctx_dim = 96 + 48 + 48
            self.gate = nn.Sequential(nn.Linear(ctx_dim, 48), nn.SiLU(), nn.Linear(48, num_policies))
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(ctx_dim, 64), nn.SiLU(), nn.Linear(64, action_dim))
                    for _ in range(num_policies)
                ]
            )
            self.value_head = nn.Sequential(nn.Linear(ctx_dim, 64), nn.SiLU(), nn.Linear(64, 1))

        def forward(
            self,
            g: torch.Tensor,
            op_nodes: torch.Tensor,
            op_mask: torch.Tensor,
            agv_nodes: torch.Tensor,
            agv_mask: torch.Tensor,
            action_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            hg = self.g_enc(g)
            op_h = self.op_enc(op_nodes) * op_mask.unsqueeze(-1)
            op_pool = op_h.sum(dim=1) / op_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            ag_h = self.agv_enc(agv_nodes) * agv_mask.unsqueeze(-1)
            ag_pool = ag_h.sum(dim=1) / agv_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            ctx = torch.cat([hg, op_pool, ag_pool], dim=-1)
            gates = torch.softmax(self.gate(ctx), dim=-1)
            head_logits = torch.stack([hd(ctx) for hd in self.heads], dim=1)  # [B, P, A]
            logits = (gates.unsqueeze(-1) * head_logits).sum(dim=1)
            logits = logits.masked_fill(action_mask <= 0.5, -1e9)
            val = self.value_head(ctx).squeeze(-1)
            return logits, val, gates

    return MPASILNet()


class RLV12MPASILPolicy:
    """Closed-loop inference wrapper for trained MP-ASIL + ICRA'24 Joint Trigger V12 policy."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        epoch_interval: int = 16,
        stall_threshold: int = 8,
        trigger: Optional[WhenToReplanTrigger] = None,
        hybrid_trigger_gate: bool = True,
    ) -> None:
        import torch

        torch.set_num_threads(1)
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        self.net = build_mpasil_v12_model()
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.hybrid_trigger_gate = bool(hybrid_trigger_gate)
        self.trigger = trigger or WhenToReplanTrigger(
            epoch_interval=int(ckpt.get("epoch_interval", epoch_interval)),
            stall_threshold=stall_threshold,
            suppress_calm_heuristic=False,
        )

    def reset(self) -> None:
        self.trigger.reset()

    def select_action(self, dg12: dict[str, np.ndarray], info: dict[str, Any], sim: Any = None) -> int:
        import torch

        obs_v12 = dg12["global_vec"]
        mask = info["action_mask"]
        tick = int(info.get("step", 0))
        if self.hybrid_trigger_gate:
            q_mode, _ = self.trigger.should_query(obs_v12, mask, tick, sim=sim)
            if q_mode == "none":
                return ACTION_KEEP
            if q_mode == "surgical" and len(mask) > 1 and mask[1] == 1:
                return 1
            op_mask = mask.astype(np.float32).copy()  # Keep ACTION_KEEP (0) legal so MP-ASIL can abstain on low-leverage triggers!
        else:
            if is_in_secondary_hold(sim) if sim is not None else False:
                return ACTION_KEEP
            op_mask = mask.astype(np.float32).copy()

        # Causal fault-window & starvation safety invariant:
        # Never execute slow (>20ms) or no-op KEEP on tight active fault/urgent windows or open faults!
        is_fault = float(obs_v12[18]) > 0.5
        is_urg = float(obs_v12[19]) > 0.5
        win_tight = float(obs_v12[36]) if len(obs_v12) > 36 else 1.0
        if (is_fault or is_urg) and win_tight <= 0.52 and len(mask) > 2 and mask[2] == 1:
            return 2
        stag = _stagnation_break_action(mask, tick)
        if stag is not None:
            return stag
        if is_fault or float(obs_v12[10]) > 0.05 or tick >= 240:
            op_mask[ACTION_KEEP] = 0.0
            if float(op_mask.sum()) <= 0.5:
                return ACTION_KEEP

        with torch.no_grad():
            logits, _val, _gates = self.net(
                torch.from_numpy(obs_v12).unsqueeze(0),
                torch.from_numpy(dg12["op_nodes"]).unsqueeze(0),
                torch.from_numpy(dg12["op_mask"]).unsqueeze(0),
                torch.from_numpy(dg12["agv_nodes"]).unsqueeze(0),
                torch.from_numpy(dg12["agv_mask"]).unsqueeze(0),
                torch.from_numpy(op_mask).unsqueeze(0),
            )
            return int(torch.argmax(logits, dim=-1).item())


def classify_v12_regime(obs_v12: np.ndarray) -> int:
    """Classify current V12 state into Fault (0), Urgent (1), or Opportunity (2) regime."""
    if float(obs_v12[18]) > 0.5:
        return 0
    if float(obs_v12[19]) > 0.5:
        return 1
    return 2


def build_smdp_v12_qnet(
    state_dim: int = EXPERT_V12_STATE_DIM,
    action_dim: int = NUM_EXPERT_V12_ACTIONS,
):
    """Construct Tri-Regime Double-Dueling SMDP Q-Network over V12 44-D state and 9-action portfolio."""
    import torch
    import torch.nn as nn

    class DuelingQHeadV12(nn.Module):
        def __init__(self, s_dim: int, a_dim: int) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(s_dim, 96),
                nn.LayerNorm(96),
                nn.SiLU(),
                nn.Linear(96, 64),
                nn.SiLU(),
            )
            self.val_head = nn.Linear(64, 1)
            self.adv_head = nn.Linear(64, a_dim)
            with torch.no_grad():
                self.val_head.bias.zero_()
                self.adv_head.bias.zero_()

        def forward(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            h = self.backbone(state)
            v = self.val_head(h)
            a = self.adv_head(h)
            mf = mask.float()
            legal_cnt = mf.sum(dim=-1, keepdim=True).clamp(min=1.0)
            q = v + (a - (a * mf).sum(dim=-1, keepdim=True) / legal_cnt)
            return q - (1.0 - mf) * 1e8

    class TriRegimeSMDPV12Net(nn.Module):
        def __init__(self, s_dim: int = state_dim, a_dim: int = action_dim) -> None:
            super().__init__()
            self.state_dim = s_dim
            self.action_dim = a_dim
            self.fault_net = DuelingQHeadV12(s_dim, a_dim)
            self.urgent_net = DuelingQHeadV12(s_dim, a_dim)
            self.opt_net = DuelingQHeadV12(s_dim, a_dim)

        def forward_one(self, state: torch.Tensor, mask: torch.Tensor, regime_id: int) -> torch.Tensor:
            if regime_id == 0:
                return self.fault_net(state, mask)
            if regime_id == 1:
                return self.urgent_net(state, mask)
            return self.opt_net(state, mask)

    return TriRegimeSMDPV12Net()


def should_query_v12_smdp(
    obs_v12: np.ndarray,
    mask: np.ndarray,
    tick: int,
    last_decision_tick: int,
    epoch_interval: int = 12,
    prev_emerg: tuple[bool, bool] = (False, False),
    sim: Any = None,
    last_surgical: int = -999,
    surgical_cooldown: int = 12,
    stall_threshold: int = 8,
    p2_wake_cache: Optional[dict[str, Any]] = None,
) -> tuple[str, tuple[bool, bool], np.ndarray]:
    """V12 SMDP cadence combining RL-v7 low-churn opportunity gating with Haven/secondary-window safety."""
    cur_f = bool(float(obs_v12[18]) > 0.5)
    cur_u = bool(float(obs_v12[19]) > 0.5)
    cur_emerg = (cur_f, cur_u)
    exec_mask = np.asarray(mask, dtype=np.int8).copy()

    if sim is not None and is_in_secondary_hold(sim):
        return "none", cur_emerg, exec_mask
    if not any(exec_mask[i] == 1 for i in range(1, min(len(exec_mask), NUM_EXPERT_V12_ACTIONS))):
        return "none", cur_emerg, exec_mask

    stalled_any = False
    secondary_ready = False
    if sim is not None:
        stalled_any = bool(detect_stalled_picked_agvs(sim, stall_threshold=stall_threshold, include_edge_gate_stalls=True))
        fset = getattr(sim, "_faulted_agvs", set())
        if fset:
            secondary_ready = any(
                getattr(t, "window_missed", False)
                and tick >= int(getattr(t, "secondary_window_open_step", 999999))
                and getattr(t, "assigned_agv", None) in fset
                and getattr(getattr(t, "status", ""), "value", str(getattr(t, "status", ""))).lower() not in ("completed", "failed")
                for t in getattr(sim, "tasks", [])
            )

    win_tight = float(obs_v12[36]) if len(obs_v12) > 36 else 1.0
    if cur_f and win_tight <= 0.52 and len(exec_mask) > 2 and exec_mask[2] == 1:
        exec_mask[:] = 0
        exec_mask[2] = 1
    elif cur_f or (float(obs_v12[0]) < 0.75 and float(obs_v12[10]) > 0.12) or (tick >= 360 and float(obs_v12[10]) > 0.01):
        exec_mask[ACTION_KEEP] = 0
        if not np.any(exec_mask > 0):
            exec_mask[ACTION_KEEP] = 1

    prev_f, prev_u = prev_emerg
    rising = (cur_f and not prev_f) or (cur_u and not prev_u)
    if secondary_ready or (last_decision_tick < 0 and tick <= 8) or (rising and float(obs_v12[16]) < 0.5):
        return "global", cur_emerg, exec_mask

    if sim is not None and getattr(sim, "_p2_execution_aware", False) and getattr(sim, "_perturbation_model", None) is not None:
        p2_sig = sim._perturbation_model.active_disturbance_sig(sim, tick)
        cache = p2_wake_cache if p2_wake_cache is not None else {}
        fail_assg = tuple(p2_sig.get("fail_assigned_agvs") or ())
        if fail_assg:
            if fail_assg != cache.get("g") and (last_decision_tick < 0 or (tick - last_decision_tick) >= 6):
                cache["g"] = fail_assg
                return "global", cur_emerg, exec_mask
        else:
            cache.pop("g", None)
        blk_p = tuple(p2_sig.get("blocked_picked_agvs") or ())
        blk_a = tuple(p2_sig.get("blocked_assigned_agvs") or ())
        blk_e = tuple(p2_sig.get("blocked_edges") or ())
        if (blk_p or blk_a) and len(exec_mask) > 1 and exec_mask[1] == 1:
            s_key = (blk_p, blk_a, blk_e)
            if s_key != cache.get("s") and (last_surgical < 0 or (tick - last_surgical) >= 4):
                cache["s"] = s_key
                return "surgical", cur_emerg, exec_mask
        else:
            cache.pop("s", None)

    gate_stall = len(obs_v12) > 38 and float(obs_v12[38]) > 0.01
    if not cur_f and (stalled_any or gate_stall) and len(exec_mask) > 1 and exec_mask[1] == 1:
        if last_surgical < 0 or (tick - last_surgical) >= surgical_cooldown:
            return "surgical", cur_emerg, exec_mask

    if cur_f or cur_u:
        if (tick - last_decision_tick) >= min(epoch_interval, 12):
            return "global", cur_emerg, exec_mask
        return "none", cur_emerg, exec_mask

    # Smooth-execution suppression matching RL-v7 SMDP rhythm (allow rescue only if horizon > 320 or backlog)
    eff_unassigned = 0.0 if float(obs_v12[0]) >= 0.95 else float(obs_v12[10])
    if tick < 320 and eff_unassigned <= 0.01 and (float(obs_v12[11]) >= 0.92 or float(obs_v12[2]) >= 0.92):
        return "none", cur_emerg, exec_mask
    if (
        tick < 320
        and float(obs_v12[0]) >= 0.95
        and float(obs_v12[8]) < 0.16
        and float(obs_v12[9]) < 0.10
        and float(obs_v12[32]) < 0.20
        and (tick - last_decision_tick) < 24
    ):
        return "none", cur_emerg, exec_mask

    cooldown_ready = float(obs_v12[7]) >= 0.08 and (tick - last_decision_tick) >= epoch_interval
    has_leverage = (
        float(obs_v12[8]) >= 0.08
        or float(obs_v12[13]) >= 0.12
        or float(obs_v12[9]) >= 0.04
        or float(obs_v12[4]) < 0.30
        or float(obs_v12[17]) >= 0.25
        or float(obs_v12[10]) >= 0.05
        or (tick >= 320)
    )
    if cooldown_ready and has_leverage:
        return "global", cur_emerg, exec_mask
    return "none", cur_emerg, exec_mask


class RLV12SMDPPolicy:
    """Closed-loop inference wrapper for trained Tri-Regime SMDP Double-Dueling Q-Network V12."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        epoch_interval: int = 12,
        stall_threshold: int = 8,
        surgical_cooldown: int = 12,
    ) -> None:
        import torch

        torch.set_num_threads(1)
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        self.epoch_interval = int(ckpt.get("epoch_interval", epoch_interval))
        self.stall_threshold = int(stall_threshold)
        self.surgical_cooldown = int(surgical_cooldown)
        self.net = build_smdp_v12_qnet(
            state_dim=int(ckpt.get("state_dim", EXPERT_V12_STATE_DIM)),
            action_dim=int(ckpt.get("action_dim", NUM_EXPERT_V12_ACTIONS)),
        )
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.reset()

    def reset(self) -> None:
        self.last_dec = -999
        self.last_surgical = -999
        self.prev_emerg: tuple[bool, bool] = (False, False)
        self._p2_wake_cache: dict[str, Any] = {}

    def select_action(self, obs_v12: np.ndarray, info: dict[str, Any], sim: Any = None) -> int:
        import torch

        mask = info["action_mask"]
        tick = int(info.get("step", 0))
        q_mode, cur_emerg, exec_mask = should_query_v12_smdp(
            obs_v12,
            mask,
            tick,
            self.last_dec,
            epoch_interval=self.epoch_interval,
            prev_emerg=self.prev_emerg,
            sim=sim,
            last_surgical=self.last_surgical,
            surgical_cooldown=self.surgical_cooldown,
            stall_threshold=self.stall_threshold,
            p2_wake_cache=self._p2_wake_cache,
        )
        self.prev_emerg = cur_emerg
        if q_mode == "none":
            return ACTION_KEEP
        if q_mode == "surgical":
            self.last_surgical = tick
            return 1
        self.last_dec = tick
        is_fault = float(obs_v12[18]) > 0.5
        win_tight = float(obs_v12[36]) if len(obs_v12) > 36 else 1.0
        if not (is_fault and win_tight <= 0.52):
            stag = _stagnation_break_action(mask, tick)
            if stag is not None:
                return stag

        reg = classify_v12_regime(obs_v12)
        with torch.no_grad():
            qvals = self.net.forward_one(
                torch.from_numpy(obs_v12).float().unsqueeze(0),
                torch.from_numpy(exec_mask).float().unsqueeze(0),
                reg,
            )
            return int(torch.argmax(qvals, dim=-1).item())
