"""Guard the V18 reduced observation path against controller input drift."""

from pathlib import Path

import numpy as np
import pytest

from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.cvcr.v18_frontier_synthesis import V18AdaptiveController, V18FrontierSynthesisPolicy
from amr_sched.rl import AMRReschedulingEnv


@pytest.mark.parametrize("windowed", [False, True])
def test_v18_reduced_observation_matches_full_controller(tmp_path: Path, windowed: bool) -> None:
    """Exercise reduced and full observations against the same live simulator state."""
    policy_args = {"pgmpo_ckpt": tmp_path / "unused.npz", "allow_missing_pgmpo": True}
    lean_policy = V18FrontierSynthesisPolicy(**policy_args)
    lean = V18AdaptiveController(policy=lean_policy, profile="calm")
    full = V18AdaptiveController(policy=V18FrontierSynthesisPolicy(**policy_args), profile="calm")
    env = AMRReschedulingEnv(
        topology="grid", seed=6001, expert_v12_portfolio=True,
        v13_kitting_mode="V18_EA_JOINT", v15_policy=lean_policy,
    )
    with virtual_clock_scope():
        base_obs, info = env.reset(seed=6001)
        if windowed:
            env.tracker.operations[0].resource_windows = ((0, 24), (50, 80))
        observed_sizes = set()
        for _ in range(48):
            lean_obs = lean.observe(env, base_obs)
            full_obs = env.get_expert_v12_obs(base_obs)
            observed_sizes.add(len(lean_obs))
            np.testing.assert_array_equal(lean_obs, full_obs[:len(lean_obs)])
            act = lean.select_action(lean_obs, info, sim=env.sim)
            assert act == full.select_action(full_obs, info, sim=env.sim)
            base_obs, _, terminated, truncated, info = env.step(act)
            if terminated or truncated:
                break
    assert (38 if windowed else 22) in observed_sizes
