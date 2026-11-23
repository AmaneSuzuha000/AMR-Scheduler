# -*- coding: utf-8 -*-
"""RUST-00: 调度内核 CPU 与内存微观剖析基准脚本 (Micro-Profiling Baseline).

目标:
-----
精准定量分析 Python 调度内核在 100 任务 / 6 车极限工况下的算力开销分布，
为 Rust 核心重构（reservation_core, spacetime_planner）提供第一手量化基准依据。
"""

from __future__ import annotations

import cProfile
import json
import os
from pathlib import Path
import pstats
import time
import tracemalloc
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.layouts import build_layout
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator


def profile_scenario(
    layout_name: str = "workshop",
    n_tasks: int = 100,
    n_agvs: int = 6,
    seed: int = 42,
    planner_type: str = "spacetime_astar",
    aisle_capacity: int = 2,
    max_steps: int = 20_000,
) -> dict[str, Any]:
    """对单组工况进行全量 cProfile 与内存追踪."""
    planner = (
        SpaceTimeAStarPlanner(1.0, 1.0)
        if planner_type == "spacetime_astar"
        else StaticDijkstraPlanner(1.0, 1.0)
    )
    allocator = GreedyAllocator()
    net_base = build_layout(layout_name, aisle_capacity=aisle_capacity)
    sc = ScenarioConfig(
        n_tasks=n_tasks,
        n_agvs=n_agvs,
        seed=seed,
        aisle_capacity=aisle_capacity,
    )
    net, tasks, agvs = generate_scenario(sc, network=net_base)

    sim = Simulator(
        net,
        tasks,
        agvs,
        planner,
        allocator,
        SimulationConfig(max_steps=max_steps),
    )

    tracemalloc.start()
    pr = cProfile.Profile()
    t0 = time.perf_counter()

    pr.enable()
    sim_res = sim.run()
    pr.disable()

    wall_time_s = time.perf_counter() - t0
    cur_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    summary = sim_res.summary()

    # 统计函数调用特征
    stream = pstats.Stats(pr)
    stream.strip_dirs()

    # 提取关键热点函数的耗时
    hotspots: list[dict[str, Any]] = []
    for func, (cc, nc, tt, ct, callers) in sorted(
        stream.stats.items(), key=lambda kv: kv[1][3], reverse=True
    )[:25]:
        filename, line, func_name = func
        hotspots.append({
            "func": f"{filename}:{line}({func_name})",
            "cumtime_s": round(ct, 4),
            "tottime_s": round(tt, 4),
            "calls": nc,
            "pct_total": round(ct / max(wall_time_s, 1e-6) * 100, 2),
        })

    return {
        "layout": layout_name,
        "planner": planner_type,
        "tasks": n_tasks,
        "agvs": n_agvs,
        "seed": seed,
        "wall_time_s": round(wall_time_s, 4),
        "steps_simulated": sim_res.finished_step,
        "tasks_completed": summary["tasks_completed"],
        "peak_mem_mb": round(peak_mem / (1024 * 1024), 3),
        "hotspots": hotspots,
    }


def main() -> None:
    print("=== 开始执行 RUST-00 调度内核微观性能剖析 ===")
    cases = [
        ("ring", "static_dijkstra"),
        ("ring", "spacetime_astar"),
        ("workshop", "static_dijkstra"),
        ("workshop", "spacetime_astar"),
    ]
    results = []
    for layout, planner in cases:
        print(f"--> Profiling layout={layout}, planner={planner}, tasks=100, agvs=6...")
        res = profile_scenario(
            layout_name=layout,
            planner_type=planner,
            n_tasks=100,
            n_agvs=6,
            seed=42,
            max_steps=15_000,
        )
        print(f"    Wall time: {res['wall_time_s']:.3f} s, Peak mem: {res['peak_mem_mb']:.2f} MB, Steps: {res['steps_simulated']}")
        # 打印 Top 5 热点
        for h in res["hotspots"][:5]:
            print(f"      - {h['func']}: {h['cumtime_s']:.3f}s ({h['pct_total']}%), calls={h['calls']}")
        results.append(res)

    out_dir = ROOT / "results" / "rust_transition"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "baseline_profile.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 剖析结果已保存至: {out_file}")


if __name__ == "__main__":
    main()
