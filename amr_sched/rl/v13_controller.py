"""V13 Hierarchical Production-Kitting Window & Joint Neighborhood Policy Wrapper.

Uses the identical `WhenToReplanTrigger`, Haven surgical corridor repair (`action=1`),
and fast open-fault detour rescue (`action=2`) as V12 so differences between:
  - `V13_KITTING_DEFAULT` (`P0`): short default without kitting neighborhood expansion
  - `V13_KITTING_DET`     (`P1`): deterministic kitting criticality + hybrid heuristic
  - `V13_KITTING_SUP`     (`P2`): supervised counterfactual pairwise ranker
  - `V13_KITTING_RL`      (`P3`): multi-step SMDP RL + anytime continue-vs-commit
isolate the causal contribution of operation-window protection & joint repair selection.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from amr_sched.rl.spaces import ACTION_KEEP
from amr_sched.rl.v12_controllers import WhenToReplanTrigger


class V13KittingPolicy:
    """Closed-loop meta-controller driving `V13_KITTING_*` inside `AMRReschedulingEnv`."""

    def __init__(
        self,
        *,
        epoch_interval: int = 16,
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
            return 1

        is_fault = float(obs_v12[18]) > 0.5
        is_urg = float(obs_v12[19]) > 0.5
        win_tight = float(obs_v12[36]) if len(obs_v12) > 36 else 1.0
        if is_fault and win_tight <= 0.52 and len(mask) > 2 and mask[2] == 1:
            return 2
        if is_urg and win_tight <= 0.35 and len(mask) > 2 and mask[2] == 1:
            return 2
        # Allocate 100ms portfolio slot during active kitting/coupling plateaus, else 50ms
        prog = float(obs_v12[11])
        coupling = float(obs_v12[37]) if len(obs_v12) > 37 else 0.0
        if (coupling >= 0.40 or prog < 0.72) and len(mask) > 5 and mask[5] == 1:
            return 5
        if len(mask) > 3 and mask[3] == 1:
            return 3
        for idx in range(len(mask) - 1, 0, -1):
            if mask[idx] == 1:
                return idx
        return ACTION_KEEP
