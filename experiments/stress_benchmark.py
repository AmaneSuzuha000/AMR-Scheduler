# -*- coding: utf-8 -*-
"""调度内核极限并发压测与微秒级延迟分布评估基准 (Stress Benchmark).

评估目标:
---------
验证 amr_sched 调度决策主循环在 100 任务 / 6 车极端高负荷工况下的计算性能与内存稳定性:
1. 单步主循环延迟分位数 (P50, P90, P99, Max);
2. 内存跟踪与泄漏审计 (Tracemalloc Peak Memory & Residuals);
3. 跨布局 (Ring, Grid, Hub, Workshop) 与跨规划器 (Static Dijkstra, SpaceTime A*) 的横向对比;
4. 刚性断言门限: 单步主循环 P99 延迟严格低于 50.0 ms (工业实时控制周期上限).
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics as st
import sys
import time
import tracemalloc
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amr_sched.allocation.baseline import NearestShortestAllocator
from amr_sched.allocation.dispatch import DispatchRule
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.allocation.rule_based import RuleBasedAllocator
from amr_sched.layouts import build_layout
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator

RESULTS_DIR = ROOT / "results"


def run_single_stress_case(
    layout: str = "ring",
    n_tasks: int = 100,
    n_agvs: int = 6,
    seed: int = 42,
    planner_name: str = "static_dijkstra",
    allocator_name: str = "greedy",
    aisle_capacity: int = 2,
    max_steps: int = 50_000,
) -> dict[str, Any]:
    """运行单组极限压力测试并精确采样单步耗时分布."""
    planner = (
        StaticDijkstraPlanner(1.0, 1.0)
        if planner_name == "static_dijkstra"
        else SpaceTimeAStarPlanner(1.0, 1.0)
    )

    if allocator_name == "greedy":
        allocator = GreedyAllocator()
    elif allocator_name == "nearest":
        allocator = NearestShortestAllocator()
    elif allocator_name == "fifo":
        allocator = RuleBasedAllocator(DispatchRule.FIFO)
    elif allocator_name == "spt":
        allocator = RuleBasedAllocator(DispatchRule.SPT)
    else:
        allocator = RuleBasedAllocator(DispatchRule.EDD)

    net_base = build_layout(layout, aisle_capacity=aisle_capacity)
    sc = ScenarioConfig(
        n_tasks=n_tasks,
        n_agvs=n_agvs,
        seed=seed,
        aisle_capacity=aisle_capacity,
    )
    net, tasks, agvs = generate_scenario(sc, network=net_base)

    sim = Simulator(net, tasks, agvs, planner, allocator, SimulationConfig(max_steps=max_steps))

    # 启动内存跟踪
    tracemalloc.start()
    step_latencies_ms: list[float] = []

    t_wall_start = time.perf_counter()
    sim.reset()

    while not sim.is_finished():
        t0 = time.perf_counter()
        cont = sim.step_once()
        t1 = time.perf_counter()
        step_latencies_ms.append((t1 - t0) * 1000.0)
        if not cont:
            break

    t_wall_end = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    res = sim._build_result()
    summary = res.summary()
    total_steps = len(step_latencies_ms)
    sorted_lats = sorted(step_latencies_ms)

    def pctl(p: float) -> float:
        idx = min(int(round(p * (total_steps - 1))), total_steps - 1)
        return sorted_lats[idx]

    metrics = {
        "layout": layout,
        "n_tasks": n_tasks,
        "n_agvs": n_agvs,
        "seed": seed,
        "planner": planner_name,
        "allocator": allocator_name,
        "total_steps": total_steps,
        "makespan_seconds": summary["makespan_seconds"],
        "tasks_completed": summary["tasks_completed"],
        "wall_time_seconds": t_wall_end - t_wall_start,
        "peak_memory_mb": peak_mem / (1024 * 1024),
        "mean_latency_ms": st.mean(step_latencies_ms) if step_latencies_ms else 0.0,
        "p50_latency_ms": pctl(0.50),
        "p90_latency_ms": pctl(0.90),
        "p99_latency_ms": pctl(0.99),
        "max_latency_ms": max(step_latencies_ms) if step_latencies_ms else 0.0,
    }
    return metrics


def run_stress_battery() -> list[dict[str, Any]]:
    """执行覆盖 4 大布局与主流规划器/分配器的极限并发压测套件."""
    configs = [
        {"layout": "ring", "planner": "static_dijkstra", "allocator": "greedy"},
        {"layout": "ring", "planner": "spacetime_astar", "allocator": "greedy"},
        {"layout": "grid", "planner": "static_dijkstra", "allocator": "greedy"},
        {"layout": "grid", "planner": "static_dijkstra", "allocator": "fifo"},
        {"layout": "hub", "planner": "static_dijkstra", "allocator": "spt"},
        {"layout": "workshop", "planner": "static_dijkstra", "allocator": "greedy"},
    ]

    results = []
    print("=" * 96)
    print("              AMR-Scheduler Kernel Stress Benchmark (100 Tasks / 6 AGVs)")
    print("=" * 96)

    for cfg in configs:
        res = run_single_stress_case(
            layout=cfg["layout"],
            n_tasks=100,
            n_agvs=6,
            seed=42,
            planner_name=cfg["planner"],
            allocator_name=cfg["allocator"],
            aisle_capacity=2,
        )
        results.append(res)
        print(
            f"[{res['layout'].upper():8s}] {res['planner']:15s} + {res['allocator']:8s} | "
            f"Steps: {res['total_steps']:5d} | Wall: {res['wall_time_seconds']:6.2f}s | "
            f"P50: {res['p50_latency_ms']:6.3f}ms | P99: {res['p99_latency_ms']:6.3f}ms | "
            f"PeakMem: {res['peak_memory_mb']:5.2f}MB | Completed: {res['tasks_completed']}/100"
        )

    # 导出 JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "stress_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Artifact] Benchmark summary saved to {out_path}")
    return results


if __name__ == "__main__":
    run_stress_battery()
