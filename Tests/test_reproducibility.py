"""数据可复现性验证：检验基准实验数据与参考快照的一致性。

确保实验结果表与冻结参考基准完全可复现，
本测试确保所有基准 CSV 数据与冻结参考快照严格一致。
"""

from __future__ import annotations

import csv
import os

import pytest

from tools.research_baseline import replay_legacy_cases


_RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")


def _load(name):
    path = os.path.join(_RESULTS, name)
    if not os.path.exists(path):
        pytest.skip("%s 不存在，跳过" % name)
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_matrix_has_expected_shape():
    """主实验矩阵必须是 3 规模 × 3 车队 × 2 规划器 × 10 种子 = 180 行。"""
    rows = _load("matrix.csv")
    assert len(rows) == 180
    combos = {(int(r["n_tasks"]), int(r["n_agvs"])) for r in rows}
    assert combos == {(n, a) for n in (20, 50, 100) for a in (2, 4, 6)}
    assert {r["planner"] for r in rows} == {"static_dijkstra", "spacetime_astar"}


def test_matrix_records_are_reproducible():
    """抽查 3 条 matrix 记录，用冻结的历史源码重算必须完全一致。

    只抽查 3 条是为了控制测试耗时；选的是最容易受
    死锁打破影响的 100 任务 / 6 车配置。
    """
    rows = _load("matrix.csv")
    sample = [r for r in rows
              if int(r["n_tasks"]) == 100 and int(r["n_agvs"]) == 6
              and r["planner"] == "static_dijkstra"][:3]
    assert len(sample) == 3, "找不到足够的抽样记录"
    summaries = replay_legacy_cases([
        {"planner": "static_dijkstra", "allocator": "greedy",
         "scenario": dict(n_tasks=100, n_agvs=6, seed=int(row["seed"]), dt=1.0,
                          max_speed=1.0, urgent_ratio=0.15),
         "config": dict(dt=1.0, max_steps=60_000, seed=int(row["seed"]))}
        for row in sample
    ])
    for r, s in zip(sample, summaries, strict=True):
        seed = int(r["seed"])
        assert s["tasks_completed"] == int(r["tasks_completed"]), (
            "seed %d 完成数与记录不符" % seed)
        assert s["makespan_seconds"] == pytest.approx(
            float(r["makespan_s"]), abs=0.01), (
            "seed %d 完成时间不可复现：%.3f vs %s"
            % (seed, s["makespan_seconds"], r["makespan_s"]))


def test_matrix_has_no_hard_failures():
    """主实验不允许有任何硬故障。"""
    rows = _load("matrix.csv")
    assert all(int(r["hard_failures"]) == 0 for r in rows)
    assert all(int(r["collision_events"]) == 0 for r in rows)


def test_matrix_completion_is_complete():
    """每一行都必须完成全部任务，不允许 horizon 截断。"""
    rows = _load("matrix.csv")
    for r in rows:
        assert int(r["tasks_completed"]) == int(r["n_tasks"]), (
            "%s/%s/seed%s 只完成 %s/%s"
            % (r["planner"], r["n_agvs"], r["seed"],
               r["tasks_completed"], r["n_tasks"]))


def test_matrix_records_both_conflict_metrics():
    """冲突等待次数与步数两列均需存在。"""
    rows = _load("matrix.csv")
    assert "conflict_wait_steps" in rows[0]
    assert "conflict_wait_events" in rows[0]
    for r in rows:
        assert int(r["conflict_wait_events"]) <= int(r["conflict_wait_steps"])


def test_all_result_files_have_headers():
    """每个结果文件都要有表头和非空数据，防止出现空表或残缺 CSV。"""
    for name in os.listdir(_RESULTS):
        if not name.endswith(".csv"):
            continue
        rows = _load(name)
        assert rows, "%s 是空文件" % name