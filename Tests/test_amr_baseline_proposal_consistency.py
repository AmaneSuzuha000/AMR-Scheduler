"""Regression tests for the executable candidates submitted by the AMR baselines."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from amr_sched.baselines.spga_cor2026 import (
    SPGAConfig,
    SPGADecodedPlan,
    SPGAProblem,
    SPGASolver,
    run_spga_policy_proposal,
)
from amr_sched.cvcr.snapshot import NativeSnapshotCapture, TaskSnapshot
from amr_sched.rl import AMRReschedulingEnv


def _opening():
    env = AMRReschedulingEnv(topology="grid", seed=2001, max_steps=60)
    env.reset(seed=2001, options={"topology": "grid"})
    snapshot = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
    return env, snapshot


def test_spga_returns_pairs_with_matching_b2_evaluation(monkeypatch):
    env, snapshot = _opening()
    problem = SPGAProblem(snapshot, env.sim.network)
    solver = SPGASolver(problem, SPGAConfig(pop_size=12, max_iters=1, mode="g1_unified"))
    monkeypatch.setattr("amr_sched.baselines.spga_cor2026.now_s", lambda: 0.0)
    chromosome = solver._seed_chromosome()
    incumbent = replace(solver.decode_relaxed(chromosome), objective=10.0)
    op_id = problem.ops[0].op_id
    improved_surrogate = replace(incumbent, objective=1.0, crews={**incumbent.crews, op_id: 4})
    monkeypatch.setattr(solver, "_seed_chromosome", lambda: chromosome)
    monkeypatch.setattr(solver, "_random_chromosome", lambda: chromosome)
    monkeypatch.setattr(solver, "decode_relaxed", Mock(side_effect=[incumbent] * 12 + [improved_surrogate] * 17))
    checked_incumbent = SimpleNamespace(valid=True, objective=20.0)
    checked_challenger = SimpleNamespace(valid=True, objective=30.0)
    evaluator = Mock()
    evaluator.evaluate.side_effect = [checked_incumbent, checked_challenger]

    plan, evaluation, checked = solver.solve(evaluator)

    assert checked == 2
    assert plan.signature() == solver._submission_plan(incumbent).signature()
    assert evaluation is checked_incumbent
    assert evaluator.evaluate.call_args_list[1].kwargs["op_crew_overrides"] == improved_surrogate.crews
    evaluator.set_incumbent.assert_called_once()


def test_spga_excludes_unreleased_and_keeps_frozen_tasks_in_scored_submission():
    env, snapshot = _opening()
    first, second = [robot.robot_id for robot in snapshot.robots[:2]]
    snapshot.tasks = [
        TaskSnapshot("frozen", "a", "b", 0, 70, "picked", assigned_agv=first),
        TaskSnapshot("ready", "a", "b", 0, 70, "pending"),
        TaskSnapshot("future", "a", "b", snapshot.tick + 1, 70, "pending"),
    ]
    solver = SPGASolver(SPGAProblem(snapshot, env.sim.network), SPGAConfig(mode="g1_unified"))
    raw = SPGADecodedPlan(
        {"frozen": second, "ready": second, "future": second}, {}, {}, {},
        {second: ["ready", "frozen", "future"]}, 10, 0, 30.0,
    )
    submitted = solver._submission_plan(raw)
    assert submitted.assign == {"frozen": first, "ready": second}
    assert submitted.task_orders == {second: ["ready"]}


def test_spga_budget_exhaustion_leaves_a_commit_safe_keep_proposal():
    env, snapshot = _opening()
    proposal = run_spga_policy_proposal(
        snapshot, env.sim.network, SPGAConfig(mode="g1_unified", budget_ms=0),
    )
    assert proposal.telemetry["fallback_keep"]
    assert proposal.telemetry["eval_count"] == 0
    assert proposal.proposed_assignment == proposal.proposed_routes == proposal.proposed_bays == {}
    assert proposal.proposed_task_sequences is None
    assert proposal.proposed_op_priorities is None
    receipt = env.owner.validate_and_commit(proposal)
    assert receipt.accepted, receipt.reasons


def test_pgmpo_scores_the_submitted_released_assignments(monkeypatch):
    import pytest

    pytest.importorskip("torch")
    from amr_sched.baselines.baselines_2026 import IJCAI26PGMPOProposer

    env, snapshot = _opening()
    first = snapshot.robots[0].robot_id
    node = snapshot.robots[0].cur_node
    ready = TaskSnapshot("ready", node, node, snapshot.tick, snapshot.tick + 70, "pending")
    future = replace(ready, task_id="future", release_step=snapshot.tick + 1)
    snapshot.tasks = [ready, future]
    instance = SimpleNamespace(
        fixed_assign={}, rel_tids=[ready.task_id, future.task_id], healthy_agvs=[first],
        op_prio={}, bays={}, crews={},
    )
    evaluated = SimpleNamespace(valid=True, objective=123.0, robot_paths={}, op_end={})
    evaluator = Mock(calls=1)
    evaluator.evaluate.return_value = evaluated
    constructor = Mock(return_value=evaluator)
    monkeypatch.setattr("amr_sched.baselines.baselines_2026.build_causal_instance", lambda *_: instance)
    monkeypatch.setattr("amr_sched.baselines.baselines_2026.CVCREvaluator", constructor)

    proposal = IJCAI26PGMPOProposer()(snapshot, env.sim.network)

    assert evaluator.evaluate.call_args.args[0] == {"ready": first}
    assert proposal.proposed_assignment == {}
    assert proposal.affected_tasks == []
    assert proposal.predicted_obj == evaluated.objective
    assert constructor.call_args.kwargs["max_speed"] == max(e.speed_limit for e in env.sim.network.edges.values())
