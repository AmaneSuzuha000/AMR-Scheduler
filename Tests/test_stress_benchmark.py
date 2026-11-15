# -*- coding: utf-8 -*-
"""调度内核极限压力与延迟分布测试 (CI Regression Guard)."""

import pytest
from experiments.stress_benchmark import run_single_stress_case


def test_kernel_stress_benchmark_ring_performance():
    """验证 Ring 布局下 100 任务 / 6 车并发性能与内存约束."""
    res = run_single_stress_case(
        layout="ring",
        n_tasks=100,
        n_agvs=6,
        seed=42,
        planner_name="static_dijkstra",
        allocator_name="greedy",
        aisle_capacity=2,
    )
    assert res["tasks_completed"] == 100, f"Expected 100 completed tasks, got {res['tasks_completed']}"
    assert res["p99_latency_ms"] < 50.0, f"P99 latency exceeded 50ms: {res['p99_latency_ms']:.3f}ms"
    assert res["peak_memory_mb"] < 10.0, f"Peak memory exceeded 10MB: {res['peak_memory_mb']:.2f}MB"
    assert res["total_steps"] > 1000


def test_kernel_stress_benchmark_workshop_realism():
    """验证 Workshop 车间拓扑下 100 任务 / 6 车并发执行与低延迟."""
    res = run_single_stress_case(
        layout="workshop",
        n_tasks=100,
        n_agvs=6,
        seed=42,
        planner_name="static_dijkstra",
        allocator_name="greedy",
        aisle_capacity=2,
    )
    assert res["tasks_completed"] == 100, f"Expected 100 completed tasks, got {res['tasks_completed']}"
    assert res["p99_latency_ms"] < 50.0, f"P99 latency exceeded 50ms: {res['p99_latency_ms']:.3f}ms"
    assert res["peak_memory_mb"] < 10.0, f"Peak memory exceeded 10MB: {res['peak_memory_mb']:.2f}MB"
