"""Stage 3 (E3) & Literature Adapters verification tests."""

from __future__ import annotations

from amr_sched.baselines.literature_adapters import (
    BalansBanditPhan24Proposer,
    ReijnenDRALNS24Proposer,
    ShiAEI25BilevelProposer,
    WangESWA24RepairProposer,
)
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.v17_event_smdp import (
    ACT_COMMIT_DETOUR,
    ACT_COMMIT_DIRECT,
    ACT_WAIT_OBSERVE,
    evaluate_e3_trajectory,
    generate_e3_scenarios,
    run_e3_smdp_experiment,
)
from amr_sched.rl import (
    AMRReschedulingEnv,
    RuleV10PortfolioPolicy,
    compute_expert_v10_action_mask,
    compute_expert_v9_state,
)


def test_e3_smdp_exact_telescoping_and_voi_counterfactual() -> None:
    """Verify exact 2-step telescoping ledger, reversibility/irreversibility, and Obs-Mask VoI drop."""
    specs = generate_e3_scenarios(6, base_seed=31000)
    for sp in specs:
        tr1 = evaluate_e3_trajectory(sp, ACT_COMMIT_DETOUR)
        tr2 = evaluate_e3_trajectory(sp, ACT_WAIT_OBSERVE, ACT_COMMIT_DETOUR)
        tr3 = evaluate_e3_trajectory(sp, ACT_WAIT_OBSERVE, ACT_COMMIT_DIRECT)
        assert tr1["telescoping_err"] < 1e-6
        assert tr2["telescoping_err"] < 1e-6
        assert tr3["telescoping_err"] < 1e-6
        # Observation mask hides step-2 clearance flag (`obs_cleared_at_dec2` stays real physically, state feature masks)
        tr_mask = evaluate_e3_trajectory(sp, ACT_WAIT_OBSERVE, ACT_COMMIT_DIRECT, mask_intermediate_obs=True)
        assert tr_mask["s2"][9] == 0.0

    rep = run_e3_smdp_experiment(n_train=90, n_eval=60, seed=31500)
    unm = rep["unmasked_eval"]
    mskd = rep["masked_counterfactual_eval"]
    # Two-step FQI learns strictly lower reversible cost than 1-step myopic while avoiding irreversible trap
    assert unm["reversible_mean_j_net"]["TwoStep_FQI_RL"] <= unm["reversible_mean_j_net"]["OneStep_Myopic"]
    assert unm["irreversible_mean_j_net"]["TwoStep_FQI_RL"] <= unm["irreversible_mean_j_net"]["OneStep_Myopic"] + 1e-6
    assert rep["causal_voi_gain_fqi"] > 0.0
    assert mskd["mean_j_net"]["TwoStep_FQI_RL"] > unm["mean_j_net"]["TwoStep_FQI_RL"]


def test_all_four_literature_adapters_closed_loop_audit() -> None:
    """Verify Wang24, Shi25, Balans/Phan24, and Reijnen24 adapters pass IndependentAuditor cleanly."""
    for cls in (WangESWA24RepairProposer, ShiAEI25BilevelProposer, BalansBanditPhan24Proposer, ReijnenDRALNS24Proposer):
        prop = cls()
        env = AMRReschedulingEnv(
            topology="unseen_choke_yard",
            seed=7001,
            allow_kitting_shortcut=False,
            pure_physical_reward=True,
            real_compute_window_coupling=True,
            expert_v10_portfolio=True,
            loaded_repair_mode="haven_v12",
            record_trajectory=True,
            external_proposer=prop,
        )
        obs, info = env.reset(seed=7001)
        pol = RuleV10PortfolioPolicy(epoch_interval=16)
        pol.reset()
        done = False
        while not done:
            assert env.sim is not None and env.tracker is not None
            o9 = compute_expert_v9_state(
                env.sim,
                env.tracker,
                last_replan_tick=env.last_replan_tick,
                recent_conflict_ticks=env.recent_conflict_ticks,
                decision_logs=env.decision_logs,
            )
            iu = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
            act = pol.select_action(o9, iu, sim=env.sim)
            obs, _, tm, tr, info = env.step(act)
            done = tm or tr
        assert env.sim is not None and env.tracker is not None
        aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
        assert aud.pass_audit, f"{cls.__name__} failed audit: {aud}"
        m = env.compute_episode_metrics()
        assert m["task_completion_rate"] >= 0.9999
        assert abs(
            (
                m["cmax"]
                + m["task_tardiness_cost"]
                + m["op_tardiness_cost"]
                + m["unfinished_task_penalty"]
                + m["unfinished_op_penalty"]
                + m["churn_cost"]
                + m["compute_cost"]
            )
            - m["net_obj"]
        ) < 0.05
