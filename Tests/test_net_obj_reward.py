"""Verify dense reward settlement against the episode objective."""

from math import fsum

import pytest

from amr_sched.cvcr.snapshot import OperationSnapshot
from amr_sched.entities import Agv, Task
from amr_sched.network import Network, Node
from amr_sched.rl.env import AMRReschedulingEnv
from amr_sched.rl.spaces import ACTION_KEEP


def make_case():
    """Give one operation a transport predecessor and an overdue completion."""
    network = Network()
    network.add_node(Node("pickup", 0.0, 0.0))
    network.add_node(Node("bay", 4.0, 0.0))
    network.add_bidirectional("pickup", "bay", 4.0)
    task = Task("material", "pickup", "bay", due_step=2)
    operation = OperationSnapshot("repair", (task.task_id,), "bay", 0, 0, 4, 5)
    return network, [task], [Agv("agv", max_speed=1.0, start_node="pickup")], [operation], {"fault_step": 999, "urgent_release_step": 999}


@pytest.mark.parametrize(
    "horizon,expect_truncated,expected_cost", [(3, True, 8005.0), (50, False, 71.0)],
)
def test_net_obj_reward_reconciles_and_resets(horizon, expect_truncated, expected_cost):
    """Check both settlement paths and that the first delta resets between episodes."""
    env = AMRReschedulingEnv(max_steps=horizon, reward_mode="net_obj", use_rust=False)
    for _ in range(2):
        env.reset(seed=1, options={"scenario_tuple": make_case()})
        rewards = []
        terminated = truncated = False
        while not (terminated or truncated):
            _, reward, terminated, truncated, _ = env.step(ACTION_KEEP)
            rewards.append(reward)
            if not (terminated or truncated):
                assert reward > -100.0
        score = env.compute_episode_metrics()
        assert truncated is expect_truncated
        assert -fsum(rewards) == pytest.approx(expected_cost, abs=0.005001)
        assert score["net_obj"] == expected_cost
        assert score["unfinished_task_penalty"] == (5000.0 if expect_truncated else 0.0)
        assert score["unfinished_op_penalty"] == (3000.0 if expect_truncated else 0.0)
        if not expect_truncated:
            assert score["task_tardiness_cost"] > 0
            assert score["op_tardiness_cost"] > 0


def test_unknown_reward_mode_is_rejected():
    """Reject misspelled objective selection at environment construction."""
    with pytest.raises(ValueError, match="reward_mode"):
        AMRReschedulingEnv(reward_mode="net-object")


def test_smdp_variable_interval_reward_reconciles_with_net_obj_both_truncated_and_completed() -> None:
    """Aggregating per-step net_obj rewards across variable SMDP intervals telescopes exactly to -net_obj."""
    from amr_sched.baselines.policy_registry import create_discrete_policy_env
    from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state
    from tools.benchmarks.compare_workshop_algorithms import build_track2_scenario

    sc, p2_cfg, _ = build_track2_scenario(101, "nominal")
    for max_steps, expect_tm, expect_tr in ((80, False, True), (1600, True, False)):
        _, ctrl, env = create_discrete_policy_env("rule_v10", seed=101, max_steps=max_steps, use_rust=True, p2_cfg=p2_cfg)
        env.reward_mode = "net_obj"
        obs, info = env.reset(seed=101, options={"scenario_tuple": sc})
        smdp_returns: list[float] = []
        acc = 0.0
        steps = 0
        tm = False
        tr = False
        while not (tm or tr):
            feats = compute_expert_v9_state(
                env.sim,
                env.tracker,
                last_replan_tick=env.last_replan_tick,
                recent_conflict_ticks=env.recent_conflict_ticks,
                decision_logs=env.decision_logs,
            )
            step_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
            act = ctrl.select_action(feats, step_info, sim=env.sim)
            obs, rew, tm, tr, info = env.step(act)
            acc += float(rew)
            steps += 1
            if (steps % 9 == 0) or tm or tr:
                smdp_returns.append(acc)
                acc = 0.0
        assert tm is expect_tm and tr is expect_tr
        metrics = env.compute_episode_metrics()
        err = abs(sum(smdp_returns) - (-float(metrics["net_obj"])))
        assert err <= 0.005001, f"SMDP reconciliation residual {err} exceeded 0.005001"
