"""加急任务动态重调度测试（任务书 R2.5）。"""

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.entities import Agv, Task, TaskPriority, TaskStatus
from amr_sched.network import build_demo_network
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator


def _preemption_fixture(policy=ReschedulePolicy.PREEMPT_UNPICKED, *, loaded=False,
                        unreachable=False, unload_time=2):
    from amr_sched.network import Network, Node, Edge
    net = Network()
    for i, name in enumerate("ABCDE"):
        net.add_node(Node(name, i * 4, 0, capacity=2))
    for a, b in zip("ABC", "BCD"):
        net.add_edge(Edge(a, b, 4))
        net.add_edge(Edge(b, a, 4))
    normal = Task("normal", "A" if loaded else "C", "D")
    urgent = Task("urgent", "E" if unreachable else "B", "A", release_step=6 if loaded else 1,
                  priority=TaskPriority.URGENT)
    sim = Simulator(net, [normal, urgent], [Agv("R", start_node="A")],
                    SpaceTimeAStarPlanner(1, 1), GreedyAllocator(),
                    SimulationConfig(max_steps=200, load_time=2, unload_time=unload_time,
                                     deadlock_timeout=0, idle_retreat=False,
                                     reschedule_policy=policy))
    sim.reset()
    return sim, normal, urgent


def _assignment_snapshot(sim):
    from copy import deepcopy
    return deepcopy((sim.tasks, sim.agvs, sim._jobs, sim._transit, sim._res.__dict__,
                     sim._load_end, sim._drop_end, sim._node_hold, sim._node_claim))


def test_cancellation_preserves_active_channel_and_arrival_claim():
    sim, _, _ = _preemption_fixture()
    sim.step_once()
    assert "R" in sim._transit
    before = _assignment_snapshot(sim)
    assert sim._cancel_job(sim.agvs[0]) is False
    assert _assignment_snapshot(sim) == before


def test_urgent_waits_for_actual_node_then_preempts_exactly_once():
    sim, normal, urgent = _preemption_fixture()
    sim.step_once()
    transit = sim._transit["R"]
    while "R" in sim._transit:
        assert sim._transit["R"] == transit
        assert sim._preemptions == 0
        sim.step_once()
    assert sim.agvs[0].cur_node == "B"
    assert sim._step_cursor == transit[3]
    sim.step_once()
    assert urgent.assigned_agv == "R"
    assert urgent.assigned_step == transit[3]
    assert normal.status is TaskStatus.PENDING
    while sim.step_once():
        pass
    assert sim._preemptions == 1
    assert all(t.status is TaskStatus.COMPLETED for t in sim.tasks)
    assert sim._hard_failures == 0


def test_infeasible_urgent_does_not_discard_original_assignment():
    from amr_sched.reservation import Interval
    sim, normal, _ = _preemption_fixture()
    assert sim._try_assign(sim.agvs[0], normal, 0)
    # B is topologically reachable, but two other owners hold both slots
    # through the whole planning horizon. The replacement probe must fail.
    sim.config.planning_horizon = 5
    for owner in ("other1", "other2"):
        sim._res.reserve_node("B", Interval(0, 200), owner)
    before = _assignment_snapshot(sim)
    sim._reschedule_step(1)
    assert _assignment_snapshot(sim) == before
    assert sim._preemptions == 0


def test_failed_first_urgent_does_not_starve_feasible_second_urgent():
    sim, normal, _ = _preemption_fixture(unreachable=True)
    second = Task("urgent2", "B", "A", release_step=1, priority=TaskPriority.URGENT)
    sim.tasks.append(second)
    sim._task_by_id[second.task_id] = second
    assert sim._try_assign(sim.agvs[0], normal, 0)
    sim._reschedule_step(1)
    assert second.assigned_agv == "R"
    assert sim._preemptions == 1


def test_idle_vehicle_serves_urgent_without_preempting_busy_vehicle():
    sim, normal, urgent = _preemption_fixture()
    assert sim._try_assign(sim.agvs[0], normal, 0)
    sim.agvs.append(Agv("idle", start_node="B"))
    sim._reschedule_step(1)
    assert urgent.assigned_agv == "idle"
    assert normal.assigned_agv == "R"
    assert sim._preemptions == 0


def test_none_rescheduling_leaves_assignments_unchanged():
    sim, normal, _ = _preemption_fixture(ReschedulePolicy.NONE)
    assert sim._try_assign(sim.agvs[0], normal, 0)
    before = _assignment_snapshot(sim)
    sim._reschedule_step(1)
    assert _assignment_snapshot(sim) == before


def test_direct_cancellation_never_removes_loaded_custody():
    sim, normal, _ = _preemption_fixture(ReschedulePolicy.PREEMPT_ANY, loaded=True)
    while sim._step_cursor < 6:
        sim.step_once()
    assert "R" not in sim._transit
    assert normal.status is TaskStatus.PICKED
    before = _assignment_snapshot(sim)
    assert sim._cancel_job(sim.agvs[0]) is False
    assert _assignment_snapshot(sim) == before


def test_loaded_preemption_returns_and_unloads_before_releasing_custody():
    sim, normal, urgent = _preemption_fixture(ReschedulePolicy.PREEMPT_ANY, loaded=True)
    while sim._step_cursor < 6:
        sim.step_once()
    old_pickup = normal.pickup_step
    sim.step_once()
    assert sim._preemptions == 1
    assert sim._jobs["R"].goal == normal.pickup
    assert sim._transit["R"][:2] == ("B", "A")
    assert normal.assigned_agv == "R"
    assert normal.status is TaskStatus.PICKED
    assert urgent.status is TaskStatus.PENDING
    # Return arrives at 10, then the two-step simulated unloading must finish.
    while sim._step_cursor < 12:
        sim.step_once()
        assert normal.assigned_agv == "R"
        assert normal.status is TaskStatus.PICKED
        assert normal.pickup_step == old_pickup
        assert (normal.pickup, normal.dropoff) == ("A", "D")
        assert sim._preemptions == 1
    assert sim.agvs[0].cur_node == "A"
    sim.step_once()
    assert normal.status is TaskStatus.PENDING
    assert normal.assigned_agv is None
    assert urgent.assigned_agv == "R"
    assert sim._material_waste_steps == 8
    while sim.step_once():
        pass
    assert all(t.status is TaskStatus.COMPLETED for t in sim.tasks)
    assert sim._preemptions == 1
    assert sim._hard_failures == 0


def _run(tasks, agvs, policy, net=None):
    net = net or build_demo_network()
    return Simulator(
        net, tasks, agvs,
        SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(max_steps=100_000, reschedule_policy=policy),
    ).run()


def test_default_policy_is_none():
    assert SimulationConfig().reschedule_policy is ReschedulePolicy.NONE


def test_policy_enum_values():
    assert ReschedulePolicy.NONE.value == "none"
    assert ReschedulePolicy.PREEMPT_UNPICKED.value == "preempt_unpicked"


def test_no_preemption_without_urgent_tasks():
    """全普通任务时，抢占策略不应产生任何抢占。"""
    net = build_demo_network()
    tasks = [
        Task("T1", pickup="D", dropoff="W1", release_step=0, due_step=500),
        Task("T2", pickup="B1", dropoff="W4", release_step=0, due_step=500),
        Task("T3", pickup="D", dropoff="S", release_step=0, due_step=500),
    ]
    agvs = [Agv("A0", start_node="B1"), Agv("A1", start_node="D")]
    r = _run(tasks, agvs, ReschedulePolicy.PREEMPT_UNPICKED, net)
    assert r.preemption_count == 0
    assert r.summary()["tasks_completed"] == 3


def test_preemption_never_cancels_loaded_agv():
    """抢占只作用于未取货任务：被抢占任务的 pickup 步骤必须仍为 None 之后重排。"""
    sc = ScenarioConfig(n_tasks=40, n_agvs=4, seed=3, urgent_ratio=0.4)
    net, tasks, agvs = generate_scenario(sc)
    r = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(1.0, 1.0),
                  GreedyAllocator(),
                  SimulationConfig(max_steps=500_000,
                                  reschedule_policy=ReschedulePolicy.PREEMPT_UNPICKED)).run()
    # 所有任务最终都完成，没有因为抢占而丢失
    assert r.summary()["tasks_completed"] == len(tasks)
    assert r.hard_failures == 0


def test_all_tasks_complete_under_both_policies():
    """两种策略下任务都不应丢失。"""
    for policy in (ReschedulePolicy.NONE, ReschedulePolicy.PREEMPT_UNPICKED):
        sc = ScenarioConfig(n_tasks=30, n_agvs=3, seed=7, urgent_ratio=0.3)
        net, tasks, agvs = generate_scenario(sc)
        r = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(1.0, 1.0),
                      GreedyAllocator(),
                      SimulationConfig(max_steps=500_000,
                                       reschedule_policy=policy)).run()
        assert r.summary()["tasks_completed"] == len(tasks), policy
        assert r.hard_failures == 0, policy


def test_urgent_metrics_present():
    sc = ScenarioConfig(n_tasks=20, n_agvs=2, seed=0, urgent_ratio=0.5)
    net, tasks, agvs = generate_scenario(sc)
    r = _run(tasks, agvs, ReschedulePolicy.NONE, net)
    s = r.summary()
    for k in ("urgent_on_time_rate", "mean_urgent_response_steps",
              "mean_response_steps", "preemption_count",
              "planning_retries", "hard_failures"):
        assert k in s, k
    assert 0.0 <= s["urgent_on_time_rate"] <= 1.0


def test_deterministic_with_preemption():
    """同一 seed 下抢占策略必须完全可复现。"""
    cfg = dict(n_tasks=30, n_agvs=3, seed=11, urgent_ratio=0.35)
    outs = []
    for _ in range(2):
        sc = ScenarioConfig(**cfg)
        net, tasks, agvs = generate_scenario(sc)
        r = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(1.0, 1.0),
                      GreedyAllocator(),
                      SimulationConfig(max_steps=500_000,
                                       reschedule_policy=ReschedulePolicy.PREEMPT_UNPICKED)).run()
        outs.append((r.summary()["makespan_seconds"], r.preemption_count))
    assert outs[0] == outs[1]


def test_hard_failures_zero_and_retries_counted():
    """hard_failures 与 planning_retries 必须是独立字段。"""
    sc = ScenarioConfig(n_tasks=20, n_agvs=2, seed=5)
    net, tasks, agvs = generate_scenario(sc)
    r = _run(tasks, agvs, ReschedulePolicy.NONE, net)
    assert r.hard_failures == 0
    assert r.planning_retries >= 0
    assert "failed_planning" not in r.summary()


def test_preempt_any_registered():
    """三档抢占策略必须都在枚举里，否则实验跑不出对照。"""
    from amr_sched.reschedule import ReschedulePolicy
    vals = {p.value for p in ReschedulePolicy}
    assert vals == {"none", "preempt_unpicked", "preempt_any"}


def test_preempt_any_only_policy_with_material_waste():
    """只有 preempt_any 会抢已装货车，只有它产生物料浪费。

    这条守住核心结论：preempt_unpicked 是**零物料代价**的抢占。
    若将来某个改动让 unpicked 也开始浪费物料，说明它抢了不该抢的车。
    """
    from amr_sched.reschedule import ReschedulePolicy
    from amr_sched.allocation.greedy import GreedyAllocator
    from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
    from amr_sched.scenarios import ScenarioConfig, generate_scenario
    from amr_sched.simulation import SimulationConfig, Simulator

    waste = {}
    for pol in ReschedulePolicy:
        total = 0.0
        for sd in range(3):
            sc = ScenarioConfig(n_tasks=50, n_agvs=4, seed=sd)
            net, tasks, agvs = generate_scenario(sc)
            sim = Simulator(net, tasks, agvs,
                            SpaceTimeAStarPlanner(1.0, 1.0),
                            GreedyAllocator(),
                            SimulationConfig(max_steps=500_000,
                                             reschedule_policy=pol))
            total += float(sim.run().summary()["material_waste_steps"])
        waste[pol.value] = total

    assert waste["none"] == 0.0, "不抢占却有物料浪费"
    assert waste["preempt_unpicked"] == 0.0, \
        "preempt_unpicked 产生了物料浪费（%.1fs），它本应只抢空驶车" % \
        waste["preempt_unpicked"]
    assert waste["preempt_any"] > 0.0, \
        "preempt_any 没有记录到物料浪费，已装货抢占可能未生效"


def test_preempt_any_can_victimize_loaded_agv():
    """preempt_any 必须真的能抢到已装货车（否则退化成 unpicked）。"""
    from amr_sched.reschedule import ReschedulePolicy
    from amr_sched.allocation.greedy import GreedyAllocator
    from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
    from amr_sched.scenarios import ScenarioConfig, generate_scenario
    from amr_sched.simulation import SimulationConfig, Simulator

    def count(pol):
        n = 0
        for sd in range(3):
            sc = ScenarioConfig(n_tasks=50, n_agvs=4, seed=sd)
            net, tasks, agvs = generate_scenario(sc)
            sim = Simulator(net, tasks, agvs,
                            SpaceTimeAStarPlanner(1.0, 1.0),
                            GreedyAllocator(),
                            SimulationConfig(max_steps=500_000,
                                             reschedule_policy=pol))
            n += int(sim.run().summary()["preemption_count"])
        return n

    a = count(ReschedulePolicy.PREEMPT_UNPICKED)
    b = count(ReschedulePolicy.PREEMPT_ANY)
    assert b > a, \
        "preempt_any 的抢占次数(%d)没有多于 unpicked(%d)，说明已装货抢占没生效" % (b, a)
