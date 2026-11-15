"""Tests for allocation strategies and dispatch rules (task specification R2.4a)."""

from __future__ import annotations

import itertools
import random

import pytest

from amr_sched.allocation.dispatch import DispatchRule, order_tasks
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.allocation.hungarian import hungarian
from amr_sched.allocation.rule_based import RuleBasedAllocator
from amr_sched.network import build_demo_network
from amr_sched.scenarios import ScenarioConfig, generate_scenario


# ---------- Hungarian algorithm ----------

def _brute_force(cost: list[list[float]]) -> float:
    n = len(cost)
    best = float("inf")
    for perm in itertools.permutations(range(n)):
        best = min(best, sum(cost[i][perm[i]] for i in range(n)))
    return best


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_hungarian_matches_brute_force(n):
    """The Hungarian algorithm must produce the same minimum total cost as brute force."""
    rnd = random.Random(n * 7717)
    for _ in range(15):
        cost = [[rnd.randint(0, 99) for _ in range(n)] for _ in range(n)]
        pairs = hungarian(cost)
        assert len(pairs) == n
        got = sum(cost[i][j] for i, j in pairs)
        assert got == _brute_force(cost)


def test_hungarian_rejects_non_square():
    with pytest.raises(ValueError, match="方阵"):
        hungarian([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_hungarian_empty():
    assert hungarian([]) == []


def test_hungarian_pairs_unique():
    """The solution must be a permutation: no repeated rows or columns."""
    rnd = random.Random(1234)
    cost = [[rnd.random() * 10 for _ in range(6)] for _ in range(6)]
    pairs = hungarian(cost)
    rows = [i for i, _ in pairs]
    cols = [j for _, j in pairs]
    assert sorted(rows) == list(range(6))
    assert sorted(cols) == list(range(6))


# ---------- Dispatch rules ----------

@pytest.fixture()
def scenario():
    sc = ScenarioConfig(n_tasks=30, n_agvs=4, seed=7, urgent_ratio=0.3)
    return generate_scenario(sc)


def test_fifo_orders_by_release(scenario):
    net, tasks, _ = scenario
    ordered = order_tasks(list(tasks), DispatchRule.FIFO, 100, net, 1.0, {})
    releases = [t.release_step for t in ordered]
    assert releases == sorted(releases)


def test_edd_orders_by_due(scenario):
    net, tasks, _ = scenario
    ordered = order_tasks(list(tasks), DispatchRule.EDD, 100, net, 1.0, {})
    dues = [t.due_step for t in ordered]
    assert dues == sorted(dues)


def test_priority_puts_urgent_first(scenario):
    net, tasks, _ = scenario
    ordered = order_tasks(list(tasks), DispatchRule.PRIORITY, 100, net, 1.0, {})
    urgent = [t.is_urgent for t in ordered]
    # every urgent task must come before the normal ones
    assert urgent == sorted(urgent, reverse=True)


def test_spt_orders_by_leg_time(scenario):
    net, tasks, _ = scenario
    ordered = order_tasks(list(tasks), DispatchRule.SPT, 0, net, 1.0, {})
    legs = [net.shortest_travel_times(t.pickup, 1.0)[t.dropoff] for t in ordered]
    assert legs == sorted(legs)


def test_order_does_not_mutate_input(scenario):
    net, tasks, _ = scenario
    before = [t.task_id for t in tasks]
    order_tasks(list(tasks), DispatchRule.EDD, 0, net, 1.0, {})
    assert [t.task_id for t in tasks] == before


def test_unknown_rule_raises(scenario):
    net, tasks, _ = scenario
    with pytest.raises(ValueError, match="未知调度规则"):
        order_tasks(list(tasks), "nope", 0, net, 1.0, {})


# ---------- Zero-distance regression ----------

def test_zero_travel_time_when_already_at_pickup():
    """When an AGV is already parked at the pickup point, the shortest travel
    time must be 0, not inf.

    This is a pitfall we actually hit: early on, shortest_travel_times dropped
    the source node itself, so the most desirable pairing of all -- "the AGV is
    already at the pickup point" -- looked up to inf and was discarded as a
    whole pair, which showed up as two AGVs parked at the pickup point while not
    a single task could be assigned.
    """
    net = build_demo_network()
    for node in ("B1", "R0", "W1", "D"):
        d = net.shortest_travel_times(node, 1.0)
        assert node in d, "the source node must be in the shortest-path table"
        assert d[node] == 0.0


def test_allocator_assigns_when_agv_at_pickup():
    """With every AGV parked at the pickup point, the allocator must still
    produce pairs."""
    sc = ScenarioConfig(n_tasks=20, n_agvs=2, seed=1)
    net, tasks, agvs = generate_scenario(sc)
    for a in agvs:
        a.cur_node = a.start_node
    alloc = RuleBasedAllocator(DispatchRule.PRIORITY, "greedy")
    pairs = alloc.allocate(net, list(agvs), list(tasks), 100, 1.0, {})
    assert pairs, "AGVs are already at the pickup point yet no task can be " \
                  "assigned, so zero-distance handling is still broken"


# ---------- Every combination works ----------

@pytest.mark.parametrize("assign", ["greedy", "nearest", "hungarian"])
@pytest.mark.parametrize("rule", list(DispatchRule))
def test_all_combinations_return_valid_pairs(rule, assign, scenario):
    """Any combination must return valid (AGV, task) pairs, without duplicates."""
    net, tasks, agvs = scenario
    for a in agvs:
        a.cur_node = a.start_node
    alloc = RuleBasedAllocator(rule, assign)
    pairs = alloc.allocate(net, list(agvs), list(tasks), 100, 1.0, {})
    agv_ids = [a.agv_id for a, _ in pairs]
    task_ids = [t.task_id for _, t in pairs]
    assert len(agv_ids) == len(set(agv_ids)), "the same AGV was assigned several tasks"
    assert len(task_ids) == len(set(task_ids)), "the same task was assigned to several AGVs"
    assert len(pairs) <= len(agvs)


def test_invalid_assign_raises():
    with pytest.raises(ValueError, match="未知分配算法"):
        RuleBasedAllocator(DispatchRule.FIFO, "magic")


def test_strategy_name_is_descriptive():
    assert RuleBasedAllocator(DispatchRule.EDD, "hungarian").name == "edd+hungarian"


# ---------- Consistency with the existing greedy allocator ----------

def test_rule_based_greedy_close_to_legacy_greedy():
    """priority+greedy and the old GreedyAllocator should behave similarly
    (same physical model)."""
    from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
    from amr_sched.simulation import SimulationConfig, Simulator

    sc = ScenarioConfig(n_tasks=20, n_agvs=2, seed=0)
    net, tasks, agvs = generate_scenario(sc)
    a1 = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(1.0, 1.0),
                   GreedyAllocator(), SimulationConfig(max_steps=100_000)).run()

    net2, tasks2, agvs2 = generate_scenario(sc)
    a2 = Simulator(net2, tasks2, agvs2, SpaceTimeAStarPlanner(1.0, 1.0),
                   RuleBasedAllocator(DispatchRule.PRIORITY, "greedy"),
                   SimulationConfig(max_steps=100_000)).run()

    s1, s2 = a1.summary(), a2.summary()
    assert s1["tasks_completed"] == s2["tasks_completed"] == 20
    # allow a 5% difference: the two break ties in a different order
    rel = abs(s1["makespan_seconds"] - s2["makespan_seconds"]) / s1["makespan_seconds"]
    assert rel < 0.05, "the two greedy implementations differ too much: %.2f%%" % (100 * rel)


def test_all_strategies_finish_all_tasks():
    """Every strategy must finish all tasks at 20/2 (no silent stalling)."""
    from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
    from amr_sched.simulation import SimulationConfig, Simulator

    sc = ScenarioConfig(n_tasks=20, n_agvs=2, seed=1)
    for rule in DispatchRule:
        for assign in ("greedy", "nearest", "hungarian"):
            net, tasks, agvs = generate_scenario(sc)
            r = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(1.0, 1.0),
                          RuleBasedAllocator(rule, assign),
                          SimulationConfig(max_steps=100_000)).run()
            assert r.summary()["tasks_completed"] == 20, \
                "%s did not finish all tasks" % RuleBasedAllocator(rule, assign).name
