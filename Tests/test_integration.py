"""系统集成测试：跨模块端到端流程（任务书第五部分）。

单元测试验证「每个函数算得对」，本文件验证「六个模块串起来能跑通」：

    scenarios → network → allocation → planners → simulation
        → reschedule → storage → viz

每个用例都走真实的公开接口，不 mock，确保集成缝是真的通的。
"""

from __future__ import annotations

import json
import os
import sqlite3

import pytest

from amr_sched import __main__ as cli
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.allocation.dispatch import DispatchRule
from amr_sched.allocation.rule_based import RuleBasedAllocator
from amr_sched.entities import AgvState, TaskStatus
from amr_sched.network import build_demo_network
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import Simulator, SimulationConfig
from amr_sched.storage import ResultStore


def _run(n_tasks=8, n_agvs=2, seed=0, planner=None, policy=None,
         record_trajectory=False, max_steps=60_000):
    """跑一次最小但完整的仿真，返回 (sim, result)。"""
    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed,
                        dt=1.0, urgent_ratio=0.15, max_speed=1.0)
    net, tasks, agvs = generate_scenario(sc)
    pl = planner or SpaceTimeAStarPlanner(max_speed=sc.max_speed, dt=sc.dt)
    sim = Simulator(
        net, tasks, agvs, pl, GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=max_steps, seed=seed,
                         reschedule_policy=policy or ReschedulePolicy.NONE,
                         record_trajectory=record_trajectory))
    result = sim.run()
    result.total_distance = sim._odometer
    return sim, result


# ----------------------------------------------------------------
# 一、全链路：从场景生成到结果汇总
# ----------------------------------------------------------------

def test_full_pipeline_all_tasks_completed():
    """最核心的集成断言：生成的任务必须全部送完，不能丢。"""
    sim, result = _run(n_tasks=10, n_agvs=3)
    assert len(result.completed_tasks()) == 10
    assert len(result.completed_tasks()) == 10
    # 每个任务都要有完整的生命周期时间戳
    for t in sim.tasks:
        assert t.status is TaskStatus.COMPLETED
        assert t.release_step is not None
        assert t.pickup_step is not None
        assert t.complete_step is not None
        assert t.complete_step >= t.pickup_step >= t.release_step


def test_full_pipeline_summary_has_all_four_metrics():
    """任务书 R5.5 要求四项指标，缺一不可。"""
    _, result = _run(n_tasks=8, n_agvs=2)
    s = result.summary()
    for key in ("makespan_seconds", "on_time_rate",
                "agv_utilization", "conflict_wait_steps"):
        assert key in s, "缺少指标 %s" % key
        assert isinstance(s[key], (int, float))
    # 取值范围检查
    assert 0.0 <= s["on_time_rate"] <= 1.0
    assert 0.0 <= s["agv_utilization"] <= 1.0
    assert s["makespan_seconds"] > 0
    assert s["conflict_wait_steps"] >= 0


def test_full_pipeline_summary_is_json_serialisable():
    """CLI --json 必须能序列化，否则前端/脚本无法消费。"""
    _, result = _run(n_tasks=5, n_agvs=2)
    s = result.summary()
    text = json.dumps(s, ensure_ascii=False)
    back = json.loads(text)
    assert back["tasks_completed"] == s["tasks_completed"]


def test_full_pipeline_reproducible_with_same_seed():
    """同种子两次运行必须完全一致——这是数据可信的前提。"""
    _, r1 = _run(n_tasks=8, n_agvs=2, seed=42)
    _, r2 = _run(n_tasks=8, n_agvs=2, seed=42)
    s1, s2 = r1.summary(), r2.summary()
    for k in ("makespan_seconds", "on_time_rate", "agv_utilization",
              "conflict_wait_steps", "tasks_completed"):
        assert s1[k] == s2[k], "指标 %s 不可复现" % k


def test_full_pipeline_different_seeds_differ():
    """不同种子应当产生不同结果，否则种子参数是摆设。"""
    _, r1 = _run(n_tasks=10, n_agvs=2, seed=1)
    _, r2 = _run(n_tasks=10, n_agvs=2, seed=2)
    assert (r1.finished_step != r2.finished_step
            or r1.summary()["conflict_wait_steps"]
            != r2.summary()["conflict_wait_steps"])


# ----------------------------------------------------------------
# 二、跨模块：规划器与分配器的可插拔性
# ----------------------------------------------------------------

@pytest.mark.parametrize("planner_cls", [
    StaticDijkstraPlanner, SpaceTimeAStarPlanner])
def test_both_planners_complete_all_tasks(planner_cls):
    """两种规划器都必须能独立完成全部任务，不能有「只跑通一个」的情况。"""
    _, result = _run(n_tasks=8, n_agvs=2,
                     planner=planner_cls(max_speed=1.0, dt=1.0))
    assert len(result.completed_tasks()) == 8


@pytest.mark.parametrize("policy", [
    ReschedulePolicy.NONE, ReschedulePolicy.PREEMPT_UNPICKED,
    ReschedulePolicy.PREEMPT_ANY])
def test_all_reschedule_policies_complete_all_tasks(policy):
    """三档抢占策略都必须能跑完——抢占逻辑不能破坏任务完成。"""
    _, result = _run(n_tasks=10, n_agvs=3, policy=policy)
    assert len(result.completed_tasks()) == 10


def test_rule_allocator_integrates_with_simulator():
    """规则型分配器（另一条代码路径）也要能接入仿真器。"""
    sc = ScenarioConfig(n_tasks=8, n_agvs=2, seed=7, dt=1.0,
                        urgent_ratio=0.15, max_speed=1.0)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(
        net, tasks, agvs,
        StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
        RuleBasedAllocator(dispatch=DispatchRule.FIFO),
        SimulationConfig(dt=1.0, max_steps=60_000, seed=7))
    result = sim.run()
    assert len(result.completed_tasks()) == 8


# ----------------------------------------------------------------
# 三、跨模块：仿真结果写入数据库再读回
# ----------------------------------------------------------------

def test_simulation_results_persist_to_sqlite(tmp_path):
    """仿真 → 存储 → 查询，三个模块的真实数据流。"""
    db = str(tmp_path / "e2e.db")
    sim, result = _run(n_tasks=6, n_agvs=2, record_trajectory=True)
    with ResultStore(db) as store:
        rid = store.save_run(result, label="e2e/basic", config={"n": 6},
                            record_steps=True)
        assert rid > 0
        assert store.run_count() == 1
        # 任务表行数必须与完成数一致
        rows = store.query("SELECT COUNT(*) AS c FROM tasks WHERE run_id=?",
                           (rid,))
        assert rows[0]["c"] == 6
        # 车辆表必须与车队规模一致
        agv_rows = store.query(
            "SELECT COUNT(*) AS c FROM agvs WHERE run_id=?", (rid,))
        assert agv_rows[0]["c"] == 2
        # 轨迹表非空
        step_rows = store.query(
            "SELECT COUNT(*) AS c FROM steps WHERE run_id=?", (rid,))
        assert step_rows[0]["c"] > 0


def test_agv_track_from_db_is_contiguous(tmp_path):
    """从数据库读回的轨迹必须连续（步号不跳），否则可视化会出错。"""
    db = str(tmp_path / "track.db")
    _, result = _run(n_tasks=6, n_agvs=2, record_trajectory=True)
    with ResultStore(db) as store:
        rid = store.save_run(result, label="e2e/track", config={},
                            record_steps=True)
        track = store.agv_track(rid, "AGV00")
    assert len(track) > 1
    steps = [r["step"] for r in track]
    assert steps == sorted(steps), "轨迹未按步号排序"
    assert steps == list(range(steps[0], steps[0] + len(steps))), (
        "轨迹步号不连续：首=%d 尾=%d 条数=%d" % (steps[0], steps[-1], len(steps)))


def test_db_records_survive_reopen(tmp_path):
    """关库再开，数据还在——证明真的落盘了。"""
    db = str(tmp_path / "persist.db")
    _, result = _run(n_tasks=5, n_agvs=2)
    with ResultStore(db) as store:
        store.save_run(result, label="e2e/persist", config={})
    # 新连接
    assert os.path.exists(db)
    with ResultStore(db) as store2:
        assert store2.run_count() == 1
        assert store2.label_count("e2e/persist") == 1


# ----------------------------------------------------------------
# 四、CLI 端到端（子进程级）
# ----------------------------------------------------------------

def test_cli_runs_and_reports(capsys):
    """直接调用 CLI main()，验证参数解析→仿真→打印全链路。"""
    rc = cli.main(["--tasks", "6", "--agvs", "2", "--seed", "3",
                   "--max-steps", "60000"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Completed        6/6" in out
    assert "Completion time" in out
    assert "On-time rate" in out


def test_cli_json_output_is_valid(capsys):
    """CLI --json 的输出必须是合法 JSON，供外部脚本消费。"""
    rc = cli.main(["--tasks", "5", "--agvs", "2", "--seed", "1",
                   "--max-steps", "60000", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["tasks_completed"] == 5
    assert "makespan_seconds" in data


def test_cli_save_db_writes_database(tmp_path, capsys):
    """CLI --save-db 必须真的生成可查询的数据库文件。"""
    db = str(tmp_path / "cli.db")
    rc = cli.main(["--tasks", "5", "--agvs", "2", "--seed", "2",
                   "--max-steps", "60000", "--save-db", db])
    assert rc == 0
    assert os.path.exists(db)
    with ResultStore(db) as store:
        assert store.run_count() == 1


def test_cli_all_reschedule_policies_run(capsys):
    """CLI 的 --reschedule 三个取值都要能跑通。"""
    for pol in ("none", "preempt_unpicked", "preempt_any"):
        rc = cli.main(["--tasks", "6", "--agvs", "2", "--seed", "0",
                       "--max-steps", "60000", "--reschedule", pol, "--json"])
        assert rc == 0, "策略 %s 运行失败" % pol
        capsys.readouterr()


# ----------------------------------------------------------------
# 五、数据一致性：仿真内部状态 vs 汇总输出
# ----------------------------------------------------------------

def test_summary_matches_internal_state():
    """summary() 的数字必须与仿真器内部状态一致，不能各说各话。"""
    sim, result = _run(n_tasks=10, n_agvs=3)
    s = result.summary()
    assert s["tasks_completed"] == sum(
        1 for t in sim.tasks if t.status is TaskStatus.COMPLETED)
    assert s["finished_step"] == result.finished_step
    # 每台车的忙碌步数之和不应超过 总步数 × 车数
    busy_total = sum(a.busy_steps for a in sim.agvs)
    assert busy_total <= result.finished_step * len(sim.agvs)


def test_no_agv_reports_hard_failure():
    """集成层面守门：正常算例不应出现硬故障（死锁/无路径）。"""
    _, result = _run(n_tasks=10, n_agvs=2, max_steps=80_000)
    s = result.summary()
    assert s["hard_failures"] == 0
    assert s["tasks_completed"] == 10


def test_distance_accounting_consistent():
    """里程计账目：实际里程不得小于理想里程。"""
    sim, result = _run(n_tasks=8, n_agvs=2)
    assert result.total_distance > 0
    # detour_ratio() 是纯查询：返回比值但不回写 ideal_distance
    ratio = result.detour_ratio(sim.network)
    assert ratio >= 1.0 - 1e-6, "绕行系数小于 1 说明里程计账目有误"
    # 独立复算：实际里程 / 比值 应还原出理想里程且为正
    ideal = result.total_distance / ratio
    assert ideal > 0
    assert ideal <= result.total_distance + 1e-6


def test_task_counts_monotonic():
    """任务计数只能单调不变或上升，不能倒退。"""
    sim, result = _run(n_tasks=10, n_agvs=2)
    prev = -1
    for rec in result.trajectory:
        done = rec.task_counts.get("done", 0)
        assert done >= prev, "第 %d 步完成任务数倒退" % rec.step
        prev = done


# ----------------------------------------------------------------
# 六、网络与规划的一致性
# ----------------------------------------------------------------

def test_network_from_scenario_matches_standalone():
    """scenario 生成的网络必须与直接构建的一致，否则两处会漂移。"""
    net_a = build_demo_network()
    sc = ScenarioConfig(n_tasks=4, n_agvs=2, seed=0, dt=1.0,
                        max_speed=1.0)
    net_b, _, _ = generate_scenario(sc)
    assert set(net_a.nodes.keys()) == set(net_b.nodes.keys())
    assert len(net_a.edges) == len(net_b.edges)


def test_shortest_times_usable_by_planner():
    """最短路表必须能被规划器直接使用。"""
    net = build_demo_network()
    for src in list(net.nodes.keys())[:5]:
        times = net.shortest_travel_times(src, max_speed=1.0)
        assert src in times
        assert times[src] == pytest.approx(0.0)


def test_every_task_endpoint_reachable():
    """生成的每个任务的起终点都必须在最短路表中可达。"""
    sc = ScenarioConfig(n_tasks=20, n_agvs=2, seed=5, dt=1.0,
                        max_speed=1.0)
    net, tasks, _ = generate_scenario(sc)
    for t in tasks:
        times = net.shortest_travel_times(t.pickup, max_speed=1.0)
        assert t.pickup in times, "取货点不可达: %s" % t.pickup
        assert t.dropoff in times, (
            "任务 %s 的 %s→%s 不可达"
            % (t.task_id, t.pickup, t.dropoff))
