"""Deterministic and Heuristic Rescheduling Baseline Policies.

Provides standard operational & control baselines to benchmark RL-CVCR against:
1. NeverReplanPolicy: Pure static execution without upper-layer replanning.
2. PeriodicPolicy: Fixed-interval periodic replanning (every k steps).
3. ReactiveFullPolicy: Full-scope reactive replanning while disruption signal is present.
4. SlackThresholdPolicy: Urgency-driven threshold heuristic on minimum task slack.
5. MaskedRandomPolicy: Uniform random selection among legal actions under action mask.
6. FixedScopeBudgetPolicy: Fixed scope/budget trigger upon fault/urgent observation.
7. EventWithBackoffPolicy: Edge-triggered event rule with failure/cooldown backoff.
8. FrequencyMatchedRandomPolicy: Status-agnostic random triggering matched to RL frequency.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from amr_sched.rl.spaces import (
    ACTION_KEEP,
    ACTION_TABLE,
    NUM_ACTIONS,
)


def _find_action_idx(scope: str, budget_ms: float) -> int:
    """Find action index matching given scope and budget."""
    for spec in ACTION_TABLE:
        if spec.is_replan and spec.scope == scope and abs(spec.budget_ms - budget_ms) < 1.0:
            return spec.action_idx
    return NUM_ACTIONS - 1


class BaseReschedulingPolicy:
    """Abstract interface for rescheduling policies."""

    def __init__(self, name: str) -> None:
        self.name = name

    def select_action(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class NeverReplanPolicy(BaseReschedulingPolicy):
    """Never replan; maintain initial assignments unless KEEP is physically illegal."""

    def __init__(self) -> None:
        super().__init__("NEVER_REPLAN")

    def select_action(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        mask = info.get("action_mask", np.ones(NUM_ACTIONS, dtype=np.int8))
        if mask[ACTION_KEEP] == 1:
            return ACTION_KEEP
        legal = np.where(mask == 1)[0]
        return int(legal[0]) if len(legal) > 0 else ACTION_KEEP



class PeriodicPolicy(BaseReschedulingPolicy):
    """Periodically invoke CVCR replanning every `period` steps."""

    def __init__(
        self,
        period: int = 10,
        scope: str = "S3_VALUE_EXPANDED_CVCR",
        budget_ms: float = 150.0,
    ) -> None:
        super().__init__(f"PERIODIC_{period}")
        self.period = period
        self.target_action = _find_action_idx(scope, budget_ms)

    def select_action(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        mask = info.get("action_mask", np.ones(NUM_ACTIONS, dtype=np.int8))
        step = info.get("step", 0)

        if (step > 0 and step % self.period == 0) or mask[ACTION_KEEP] == 0:
            if mask[self.target_action] == 1:
                return self.target_action
            legal = [a for a in range(1, NUM_ACTIONS) if mask[a] == 1]
            if legal:
                return legal[-1]

        if mask[ACTION_KEEP] == 1:
            return ACTION_KEEP
        legal = np.where(mask == 1)[0]
        return int(legal[0]) if len(legal) > 0 else ACTION_KEEP



class ReactiveFullPolicy(BaseReschedulingPolicy):
    """Reactive replanning: triggers full S3 @ 150ms whenever a disturbance is present."""

    def __init__(
        self,
        scope: str = "S3_VALUE_EXPANDED_CVCR",
        budget_ms: float = 150.0,
    ) -> None:
        super().__init__("REACTIVE_FULL")
        self.target_action = _find_action_idx(scope, budget_ms)

    def select_action(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        mask = info.get("action_mask", np.ones(NUM_ACTIONS, dtype=np.int8))
        faulted_ratio = obs[5]
        urgent_ratio = obs[6]
        is_disrupted = (faulted_ratio > 0.0 or urgent_ratio > 0.0 or mask[ACTION_KEEP] == 0)

        if is_disrupted:
            if mask[self.target_action] == 1:
                return self.target_action
            legal_replans = [a for a in range(1, NUM_ACTIONS) if mask[a] == 1]
            if legal_replans:
                return legal_replans[-1]

        if mask[ACTION_KEEP] == 1:
            return ACTION_KEEP
        legal = np.where(mask == 1)[0]
        return int(legal[0]) if len(legal) > 0 else ACTION_KEEP



class SlackThresholdPolicy(BaseReschedulingPolicy):
    """Trigger replanning when minimum task slack drops below threshold."""

    def __init__(
        self,
        threshold: float = 0.2,
        scope: str = "S3_VALUE_EXPANDED_CVCR",
        budget_ms: float = 50.0,
    ) -> None:
        super().__init__(f"SLACK_THRESHOLD_{threshold}")
        self.threshold = threshold
        self.target_action = _find_action_idx(scope, budget_ms)

    def select_action(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        mask = info.get("action_mask", np.ones(NUM_ACTIONS, dtype=np.int8))
        min_slack = obs[4]
        faulted_ratio = obs[5]
        urgent_ratio = obs[6]

        should_replan = (
            min_slack < self.threshold
            or faulted_ratio > 0.0
            or urgent_ratio > 0.0
            or mask[ACTION_KEEP] == 0
        )

        if should_replan:
            if mask[self.target_action] == 1:
                return self.target_action
            legal_replans = [a for a in range(1, NUM_ACTIONS) if mask[a] == 1]
            if legal_replans:
                return legal_replans[len(legal_replans) // 2]

        if mask[ACTION_KEEP] == 1:
            return ACTION_KEEP
        legal = np.where(mask == 1)[0]
        return int(legal[0]) if len(legal) > 0 else ACTION_KEEP



class MaskedRandomPolicy(BaseReschedulingPolicy):
    """Selects uniformly at random among legal actions allowed by the physical mask."""

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__("MASKED_RANDOM")
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        mask = info.get("action_mask", np.ones(NUM_ACTIONS, dtype=np.int8))
        legal = np.where(mask == 1)[0]
        if len(legal) == 0:
            return ACTION_KEEP
        return int(self.rng.choice(legal))


class FixedScopeBudgetPolicy(BaseReschedulingPolicy):
    """Fixed-scope and budget emergency replanning baseline."""

    def __init__(self, scope: str = "S0_DIRECT_AFFECTED", budget_ms: float = 20.0) -> None:
        super().__init__(f"FIXED_{scope[:2]}_{int(budget_ms)}MS")
        self.target_action = _find_action_idx(scope, budget_ms)

    def select_action(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        mask = info.get("action_mask", np.ones(NUM_ACTIONS, dtype=np.int8))
        faulted_ratio = obs[5]
        urgent_ratio = obs[6]
        is_disrupted = (faulted_ratio > 0.0 or urgent_ratio > 0.0 or mask[ACTION_KEEP] == 0)

        if is_disrupted:
            if mask[self.target_action] == 1:
                return self.target_action
            legal_replans = [a for a in range(1, NUM_ACTIONS) if mask[a] == 1]
            if legal_replans:
                return legal_replans[-1]

        if mask[ACTION_KEEP] == 1:
            return ACTION_KEEP
        legal = np.where(mask == 1)[0]
        return int(legal[0]) if len(legal) > 0 else ACTION_KEEP



class EventWithBackoffPolicy(BaseReschedulingPolicy):
    """Edge-triggered event rule with cooldown backoff (`EVENT_BACKOFF`).

    Triggers only on *new* disturbance onset (rising fault_ratio or urgent_ratio),
    subject to a cooldown backoff period, avoiding repeated futile replans while
    still executing S3 @ 150ms blindly when triggered.
    """

    def __init__(
        self,
        backoff_steps: int = 20,
        scope: str = "S3_VALUE_EXPANDED_CVCR",
        budget_ms: float = 150.0,
    ) -> None:
        super().__init__("EVENT_BACKOFF")
        self.backoff_steps = backoff_steps
        self.target_action = _find_action_idx(scope, budget_ms)
        self.last_fault_ratio: float = 0.0
        self.last_urgent_ratio: float = 0.0
        self.cooldown_until_step: int = -1

    def reset(self) -> None:
        self.last_fault_ratio = 0.0
        self.last_urgent_ratio = 0.0
        self.cooldown_until_step = -1

    def select_action(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        mask = info.get("action_mask", np.ones(NUM_ACTIONS, dtype=np.int8))
        step = int(info.get("step", 0))
        faulted_ratio = float(obs[5])
        urgent_ratio = float(obs[6])

        new_event = (
            faulted_ratio > self.last_fault_ratio + 1e-4
            or urgent_ratio > self.last_urgent_ratio + 1e-4
        )
        self.last_fault_ratio = faulted_ratio
        self.last_urgent_ratio = urgent_ratio

        if (new_event and step >= self.cooldown_until_step) or mask[ACTION_KEEP] == 0:
            self.cooldown_until_step = step + self.backoff_steps
            if mask[self.target_action] == 1:
                return self.target_action
            legal_replans = [a for a in range(1, NUM_ACTIONS) if mask[a] == 1]
            if legal_replans:
                return legal_replans[-1]

        if mask[ACTION_KEEP] == 1:
            return ACTION_KEEP
        legal = np.where(mask == 1)[0]
        return int(legal[0]) if len(legal) > 0 else ACTION_KEEP



class FrequencyMatchedRandomPolicy(BaseReschedulingPolicy):
    """State-agnostic random triggering matched to RL's ~1.2-1.5% replan frequency."""

    def __init__(self, replan_prob: float = 0.015, seed: Optional[int] = 42) -> None:
        super().__init__("FREQ_MATCHED_RANDOM")
        self.replan_prob = replan_prob
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        mask = info.get("action_mask", np.ones(NUM_ACTIONS, dtype=np.int8))
        legal_replans = [a for a in range(1, NUM_ACTIONS) if mask[a] == 1]
        if legal_replans and self.rng.random() < self.replan_prob:
            return int(self.rng.choice(legal_replans))
        if mask[ACTION_KEEP] == 1:
            return ACTION_KEEP
        return int(legal_replans[0]) if legal_replans else ACTION_KEEP


class MinimalFaultRecoveryPolicy(BaseReschedulingPolicy):
    """P0 minimal rule challenger: IF PRE_PICKUP_FAULT AND LEGAL THEN S0@20ms ELSE KEEP."""

    def __init__(self, cooldown_steps: int = 15) -> None:
        super().__init__("MINIMAL_FAULT_RECOVERY")
        self.cooldown_steps = cooldown_steps
        self.target_action = _find_action_idx("S0_DIRECT_AFFECTED", 20.0)
        self.cooldown_until_step: int = -1

    def reset(self) -> None:
        self.cooldown_until_step = -1

    def select_action(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        mask = info.get("action_mask", np.ones(NUM_ACTIONS, dtype=np.int8))
        step = int(info.get("step", 0))
        fault_open = float(obs[5]) > 0.0 and float(obs[8]) >= 0.20
        if fault_open and step >= self.cooldown_until_step and mask[self.target_action] == 1:
            self.cooldown_until_step = step + self.cooldown_steps
            return self.target_action
        if mask[ACTION_KEEP] == 1:
            return ACTION_KEEP
        legal = np.where(mask == 1)[0]
        return int(legal[0]) if len(legal) > 0 else ACTION_KEEP



class TunedThresholdRulePolicy(BaseReschedulingPolicy):
    """Validation-tuned threshold rule with fixed (scope, budget) and cooldown."""

    def __init__(
        self,
        slack_threshold: float = 0.25,
        scope: str = "S2_CROSS_DOMAIN_CLOSURE",
        budget_ms: float = 50.0,
        cooldown_steps: int = 15,
    ) -> None:
        super().__init__("TUNED_RULE_VAL")
        self.slack_threshold = slack_threshold
        self.target_action = _find_action_idx(scope, budget_ms)
        self.cooldown_steps = cooldown_steps
        self.cooldown_until_step: int = -1
        self.last_fault_ratio: float = 0.0
        self.last_urgent_ratio: float = 0.0

    def reset(self) -> None:
        self.cooldown_until_step = -1
        self.last_fault_ratio = 0.0
        self.last_urgent_ratio = 0.0

    def select_action(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        mask = info.get("action_mask", np.ones(NUM_ACTIONS, dtype=np.int8))
        step = int(info.get("step", 0))
        f_ratio = float(obs[5])
        u_ratio = float(obs[6])
        min_slack = float(obs[4])
        new_fault = f_ratio > self.last_fault_ratio + 1e-4 and float(obs[8]) >= 0.15
        new_urgent = u_ratio > self.last_urgent_ratio + 1e-4
        self.last_fault_ratio = f_ratio
        self.last_urgent_ratio = u_ratio

        triggered = new_fault or new_urgent or (min_slack < self.slack_threshold and float(obs[8]) >= 0.15)
        if triggered and step >= self.cooldown_until_step:
            self.cooldown_until_step = step + self.cooldown_steps
            if mask[self.target_action] == 1:
                return self.target_action
            legal_replans = [a for a in range(1, NUM_ACTIONS) if mask[a] == 1]
            if legal_replans:
                return legal_replans[0]
        if mask[ACTION_KEEP] == 1:
            return ACTION_KEEP
        legal = np.where(mask == 1)[0]
        return int(legal[0]) if len(legal) > 0 else ACTION_KEEP



class LiteratureBudgetedPolicy(BaseReschedulingPolicy):
    """Literature adaptation (ICML 2026 Budgeted Replanning): severity-scaled budget without window awareness."""

    def __init__(self, cooldown_steps: int = 15) -> None:
        super().__init__("LIT_BUDGETED_ADAPTIVE")
        self.cooldown_steps = cooldown_steps
        self.cooldown_until_step: int = -1
        self.last_fault_ratio: float = 0.0
        self.last_urgent_ratio: float = 0.0
        self.act_s0_20 = _find_action_idx("S0_DIRECT_AFFECTED", 20.0)
        self.act_s2_100 = _find_action_idx("S2_CROSS_DOMAIN_CLOSURE", 100.0)
        self.act_s3_150 = _find_action_idx("S3_VALUE_EXPANDED_CVCR", 150.0)

    def reset(self) -> None:
        self.cooldown_until_step = -1
        self.last_fault_ratio = 0.0
        self.last_urgent_ratio = 0.0

    def select_action(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        mask = info.get("action_mask", np.ones(NUM_ACTIONS, dtype=np.int8))
        step = int(info.get("step", 0))
        f_ratio = float(obs[5])
        u_ratio = float(obs[6])
        b_load = float(obs[1])

        new_event = f_ratio > self.last_fault_ratio + 1e-4 or u_ratio > self.last_urgent_ratio + 1e-4
        self.last_fault_ratio = f_ratio
        self.last_urgent_ratio = u_ratio

        if new_event and step >= self.cooldown_until_step:
            self.cooldown_until_step = step + self.cooldown_steps
            severity = f_ratio + u_ratio + 0.35 * b_load
            if severity >= 0.45:
                pref = self.act_s3_150
            elif severity >= 0.20:
                pref = self.act_s2_100
            else:
                pref = self.act_s0_20
            if mask[pref] == 1:
                return pref
            legal_replans = [a for a in range(1, NUM_ACTIONS) if mask[a] == 1]
            if legal_replans:
                return legal_replans[-1]
        if mask[ACTION_KEEP] == 1:
            return ACTION_KEEP
        legal = np.where(mask == 1)[0]
        return int(legal[0]) if len(legal) > 0 else ACTION_KEEP


class TwoEventRulePolicy(BaseReschedulingPolicy):
    """Edge-triggered two-event heuristic firing only on rising fault & urgent events."""

    def __init__(
        self,
        fault_scope: str = "S0_DIRECT_AFFECTED",
        fault_budget_ms: float = 20.0,
        urgent_scope: str = "S2_CROSS_DOMAIN_CLOSURE",
        urgent_budget_ms: float = 50.0,
        name: str = "TWO_EVENT_RULE",
    ) -> None:
        super().__init__(name)
        self.fault_action = _find_action_idx(fault_scope, fault_budget_ms)
        self.urgent_action = _find_action_idx(urgent_scope, urgent_budget_ms)
        self.prev_fault: float = 0.0
        self.prev_urgent: float = 0.0

    def reset(self) -> None:
        self.prev_fault = 0.0
        self.prev_urgent = 0.0

    def select_action(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        mask = info.get("action_mask", np.ones(NUM_ACTIONS, dtype=np.int8))
        fault_r = float(obs[5])
        urgent_r = float(obs[6])
        rising_fault = fault_r > 0.0 and self.prev_fault <= 0.0
        rising_urgent = urgent_r > 0.0 and self.prev_urgent <= 0.0
        self.prev_fault = fault_r
        self.prev_urgent = urgent_r

        if rising_fault and mask[self.fault_action] == 1:
            return self.fault_action
        if rising_urgent and mask[self.urgent_action] == 1:
            return self.urgent_action
        if mask[ACTION_KEEP] == 1:
            return ACTION_KEEP
        legal = np.where(mask == 1)[0]
        return int(legal[0]) if len(legal) > 0 else ACTION_KEEP


class CooldownEventRulePolicy(BaseReschedulingPolicy):
    """Dynamic two-event + cooldown congestion-triggered optimization rule."""

    def __init__(
        self,
        fault_budget_ms: float = 20.0,
        urgent_budget_ms: float = 50.0,
        opt_budget_ms: float = 50.0,
        opt_interval_norm: float = 0.06,
        min_starvation: float = 0.04,
        max_committed: float = 0.90,
        name: str = "TUNED_RULE_VAL",
    ) -> None:
        super().__init__(name)
        self.fault_action = _find_action_idx("S3_VALUE_EXPANDED_CVCR", fault_budget_ms)
        self.urgent_action = _find_action_idx("S3_VALUE_EXPANDED_CVCR", urgent_budget_ms)
        self.opt_action = _find_action_idx("S3_VALUE_EXPANDED_CVCR", opt_budget_ms)
        self.opt_interval_norm = float(opt_interval_norm)
        self.min_starvation = float(min_starvation)
        self.max_committed = float(max_committed)
        self.prev_fault: float = 0.0
        self.prev_urgent: float = 0.0

    def reset(self) -> None:
        self.prev_fault = 0.0
        self.prev_urgent = 0.0

    def select_action(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        mask = info.get("action_mask", np.ones(NUM_ACTIONS, dtype=np.int8))
        fault_r = float(obs[5])
        urgent_r = float(obs[6])
        rising_fault = fault_r > 0.0 and self.prev_fault <= 0.0
        rising_urgent = urgent_r > 0.0 and self.prev_urgent <= 0.0
        self.prev_fault = fault_r
        self.prev_urgent = urgent_r

        if rising_fault and mask[self.fault_action] == 1:
            return self.fault_action
        if rising_urgent and mask[self.urgent_action] == 1:
            return self.urgent_action

        # Non-emergency opportunistic optimization under backlog starvation & cooldown
        if (
            fault_r <= 0.0
            and urgent_r <= 0.0
            and float(obs[11]) < 0.92
            and float(obs[2]) < self.max_committed
            and float(obs[7]) >= self.opt_interval_norm
            and (float(obs[8]) >= self.min_starvation or float(obs[9]) > 0.05)
            and mask[self.opt_action] == 1
        ):
            return self.opt_action

        if mask[ACTION_KEEP] == 1:
            return ACTION_KEEP
        legal = np.where(mask == 1)[0]
        return int(legal[0]) if len(legal) > 0 else ACTION_KEEP

