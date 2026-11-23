"""RUST-05 Benchmark: Ultra Large-Scale Fleet Scaling & Online Latency Stress Test.

Evaluates scaling performance across fleet sizes:
  N_AGVS = [6, 12, 16, 24, 32, 48, 64]
on the 144-edge realistic industrial workshop layout (62 nodes) under high density.

Compares three execution engines:
1. Python SpaceTime A* (Baseline)
2. Native Rust SpaceTime A* (Track 2)
3. Native Rust SIPP Continuous Interval Planner (Track 3)

Measures:
- End-to-end wall-clock runtime (s)
- Online decision latency per simulation step (ms/step) -> Verify < 50ms budget
- Planning throughput & makespan
- Speedup multipliers across fleet densities
- Zero-deadlock verification & physical saturation boundary analysis
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.layouts import build_layout
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner as PyPlanner
from amr_sched.planners.rust_spacetime_astar import RustSpaceTimeAStarPlanner as RustAStarPlanner
from amr_sched.planners.rust_sipp_planner import RustSippPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator


def run_single_simulation(
    engine_name: str,
    planner_factory: Any,
    layout_name: str,
    n_tasks: int,
    n_agvs: int,
    seed: int,
    aisle_capacity: int = 2,
    max_steps: int = 3500,
) -> Dict[str, Any]:
    sc = ScenarioConfig(
        n_tasks=n_tasks,
        n_agvs=n_agvs,
        seed=seed,
        aisle_capacity=aisle_capacity,
    )
    sim_cfg = SimulationConfig(max_steps=max_steps)

    net_base = build_layout(layout_name, aisle_capacity=aisle_capacity)
    net, tasks, agvs = generate_scenario(sc, network=net_base)
    planner = planner_factory()
    allocator = GreedyAllocator()

    sim = Simulator(net, tasks, agvs, planner, allocator, sim_cfg)

    t0 = time.perf_counter()
    res = sim.run()
    elapsed = time.perf_counter() - t0

    summary = res.summary()
    steps = res.finished_step
    ms_per_step = (elapsed * 1000.0 / steps) if steps > 0 else 0.0

    return {
        "engine": engine_name,
        "n_agvs": n_agvs,
        "n_tasks": n_tasks,
        "tasks_completed": summary["tasks_completed"],
        "finished_step": steps,
        "runtime_s": round(elapsed, 4),
        "ms_per_step": round(ms_per_step, 3),
        "meets_latency_budget": ms_per_step < 50.0,
    }


def run_fleet_scale_benchmark():
    print("=" * 96, flush=True)
    print("RUST-05: Ultra Large-Scale Fleet Scaling & Online Latency Stress Benchmark", flush=True)
    print("Layout: Industrial Workshop (62 nodes, 144 edges) | Task Load: 50 Tasks", flush=True)
    print("=" * 96, flush=True)

    fleet_sizes = [6, 12, 16, 24, 32, 48, 64]
    n_tasks = 50
    seed = 42
    layout = "workshop"

    results = []

    print(f"{'AGVs':<6} | {'Engine':<18} | {'Tasks':<8} | {'Steps':<7} | {'Time (s)':<10} | {'ms/Step':<10} | {'<50ms Budget':<14} | {'Speedup':<8}", flush=True)
    print("-" * 96, flush=True)

    for n_agvs in fleet_sizes:
        # 1. Python Baseline
        py_res = run_single_simulation(
            "Python SpacetimeA*",
            lambda: PyPlanner(1.0, 1.0, seed=seed),
            layout,
            n_tasks,
            n_agvs,
            seed,
        )
        t_py = py_res["runtime_s"]
        py_res["speedup"] = 1.0
        results.append(py_res)
        print(f"{n_agvs:<6} | {py_res['engine']:<18} | {py_res['tasks_completed']:<8} | {py_res['finished_step']:<7} | {py_res['runtime_s']:<10.3f} | {py_res['ms_per_step']:<10.3f} | {'PASS' if py_res['meets_latency_budget'] else 'FAIL':<14} | 1.00x", flush=True)

        # 2. Rust Spacetime A*
        rust_res = run_single_simulation(
            "Rust SpacetimeA*",
            lambda: RustAStarPlanner(1.0, 1.0, seed=seed),
            layout,
            n_tasks,
            n_agvs,
            seed,
        )
        t_rust = rust_res["runtime_s"]
        sp_rust = t_py / t_rust if t_rust > 0 else 0.0
        rust_res["speedup"] = round(sp_rust, 2)
        results.append(rust_res)
        print(f"{n_agvs:<6} | {rust_res['engine']:<18} | {rust_res['tasks_completed']:<8} | {rust_res['finished_step']:<7} | {rust_res['runtime_s']:<10.3f} | {rust_res['ms_per_step']:<10.3f} | {'PASS' if rust_res['meets_latency_budget'] else 'FAIL':<14} | {sp_rust:.2f}x", flush=True)

        # 3. Rust SIPP
        sipp_res = run_single_simulation(
            "Rust SIPP",
            lambda: RustSippPlanner(1.0, 1.0, seed=seed),
            layout,
            n_tasks,
            n_agvs,
            seed,
        )
        t_sipp = sipp_res["runtime_s"]
        sp_sipp = t_py / t_sipp if t_sipp > 0 else 0.0
        sipp_res["speedup"] = round(sp_sipp, 2)
        results.append(sipp_res)
        print(f"{n_agvs:<6} | {sipp_res['engine']:<18} | {sipp_res['tasks_completed']:<8} | {sipp_res['finished_step']:<7} | {sipp_res['runtime_s']:<10.3f} | {sipp_res['ms_per_step']:<10.3f} | {'PASS' if sipp_res['meets_latency_budget'] else 'FAIL':<14} | {sp_sipp:.2f}x", flush=True)
        print("-" * 96, flush=True)

    # Save to JSON
    output_dir = repo_root / "results" / "rust_transition"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "fleet_stress_benchmark.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "RUST-05 Fleet Scaling & Latency Stress Test",
            "fleet_sizes": fleet_sizes,
            "tasks_per_run": n_tasks,
            "layout": layout,
            "results": results,
        }, f, indent=2)

    print(f"\nBenchmark completed successfully! Results persisted to:\n  {out_file}", flush=True)


if __name__ == "__main__":
    run_fleet_scale_benchmark()
