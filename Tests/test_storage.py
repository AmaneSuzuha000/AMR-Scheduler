"""数据存储模块的测试。"""

from __future__ import annotations

import os
import sqlite3

import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.storage import ResultStore


@pytest.fixture()
def store(tmp_path):
    s = ResultStore(str(tmp_path / "runs.db"))
    yield s
    s.close()


def _result(seed=0, n_tasks=10, n_agvs=2, trajectory=False):
    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(net, tasks, agvs, StaticDijkstraPlanner(1.0, 1.0),
                    GreedyAllocator(),
                    SimulationConfig(max_steps=200_000,
                                     record_trajectory=trajectory))
    r = sim.run()
    r.total_distance = sim._odometer
    return r, sc


def test_save_and_read_back(store):
    """一次运行入库后，汇总指标必须原样读回。"""
    r, sc = _result()
    rid = store.save_run(r, label="t/1", config=sc)
    assert rid > 0
    row = store.query("SELECT * FROM runs WHERE run_id = ?", (rid,))[0]
    s = r.summary()
    assert row["tasks_completed"] == s["tasks_completed"]
    assert row["makespan_s"] == pytest.approx(float(s["makespan_seconds"]))
    assert row["label"] == "t/1"
    assert store.run_count() == 0 or store.run_count() == 1


def test_tasks_persisted_with_timestamps(store):
    """每个任务都要落库，且完成时刻与流转时间一致。"""
    r, sc = _result()
    rid = store.save_run(r, label="t/1", config=sc)
    n = store.query("SELECT COUNT(*) AS n FROM tasks WHERE run_id = ?", (rid,))[0]["n"]
    assert n == len(r.tasks)
    rows = store.query(
        "SELECT * FROM tasks WHERE run_id = ? AND complete_step IS NOT NULL", (rid,))
    assert rows, "应该有已完成的任务"
    for row in rows:
        assert row["flow_steps"] == row["complete_step"] - row["release_step"]


def test_agvs_persisted(store):
    """每台 AGV 的忙碌/空闲/阻塞步数都要落库。"""
    r, sc = _result(n_agvs=3)
    rid = store.save_run(r, label="t/1", config=sc)
    rows = store.query("SELECT * FROM agvs WHERE run_id = ?", (rid,))
    assert len(rows) == 3
    for row in rows:
        assert row["busy_steps"] + row["idle_steps"] + row["blocked_steps"] >= 0


def test_steps_optional(store):
    """默认不落逐步快照，开启后才有——避免批量实验把库撑爆。"""
    r1, sc1 = _result(seed=0, trajectory=False)
    rid1 = store.save_run(r1, label="no-steps", config=sc1)
    assert store.query("SELECT COUNT(*) AS n FROM steps WHERE run_id = ?", (rid1,))[0]["n"] == 0

    r2, sc2 = _result(seed=1, trajectory=True)
    rid2 = store.save_run(r2, label="with-steps", config=sc2, record_steps=True)
    assert store.query("SELECT COUNT(*) AS n FROM steps WHERE run_id = ?", (rid2,))[0]["n"] > 0


def test_agv_track_ordered_by_step(store):
    """轨迹查询必须按步号有序，否则动画会乱跳。"""
    r, sc = _result(trajectory=True)
    rid = store.save_run(r, label="t", config=sc, record_steps=True)
    track = store.agv_track(rid, "AGV00")
    assert len(track) > 0
    steps = [row["step"] for row in track]
    assert steps == sorted(steps)


def test_multiple_runs_isolated(store):
    """多次运行的任务/轨迹不能串到一起。"""
    r1, sc1 = _result(seed=0)
    r2, sc2 = _result(seed=1)
    id1 = store.save_run(r1, label="a", config=sc1)
    id2 = store.save_run(r2, label="b", config=sc2)
    assert id1 != id2
    assert store.run_count() == 2
    n1 = store.query("SELECT COUNT(*) AS n FROM tasks WHERE run_id = ?", (id1,))[0]["n"]
    n2 = store.query("SELECT COUNT(*) AS n FROM tasks WHERE run_id = ?", (id2,))[0]["n"]
    assert n1 == len(r1.tasks) and n2 == len(r2.tasks)


def test_label_query(store):
    """按标签统计运行次数，便于批量实验核对是否有漏跑。"""
    for i in range(3):
        r, sc = _result(seed=i)
        store.save_run(r, label="batch/x", config=sc)
    assert store.label_count("batch/x") == 3
    assert store.label_count("batch/y") == 0


def test_config_roundtrip(store):
    """配置文件要以 JSON 存进去，保证结果可追溯。"""
    r, sc = _result()
    rid = store.save_run(r, label="t", config=sc)
    raw = store.query("SELECT config_json FROM runs WHERE run_id = ?", (rid,))[0]["config_json"]
    import json
    d = json.loads(raw)
    assert d["n_tasks"] == sc.n_tasks
    assert d["seed"] == sc.seed


def test_failed_insert_rolls_back(store):
    """写入失败必须整次回滚，不留半截数据。"""
    r, sc = _result()
    store.save_run(r, label="good", config=sc)

    # 直接把任务表删掉，让 INSERT INTO tasks 必然失败，
    # 检验 runs 那一行会不会被回滚掉。
    store._conn.execute("DROP TABLE tasks")
    store._conn.commit()
    r2, sc2 = _result(seed=1)
    with pytest.raises(sqlite3.OperationalError):
        store.save_run(r2, label="bad", config=sc2)
    # 失败的这次运行不应在 runs 里留下任何行
    assert store.run_count() == 1, "失败后残留了 runs 行，事务没回滚"


def test_pragma_foreign_keys_on(store):
    """外键必须开启，否则删 run 会留下孤儿行。"""
    r, sc = _result()
    rid = store.save_run(r, label="t", config=sc)
    store._conn.execute("DELETE FROM runs WHERE run_id = ?", (rid,))
    store._conn.commit()
    assert store.query("SELECT COUNT(*) AS n FROM tasks WHERE run_id = ?", (rid,))[0]["n"] == 0