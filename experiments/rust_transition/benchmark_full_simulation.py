"""RUST-02 Benchmark: End-to-end Simulation Acceleration Comparison.

Runs the identical 100-task x 6-AGV high-load workshop scenario under:
1. Python baseline (SpaceTimeAStarPlanner)
2. Rust accelerated engine (RustSpaceTimeAStarPlanner)

Validates 100% equivalence in completed tasks and measures wall-clock speedup.
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.layouts import build_layout
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner as PyPlanner
from amr_sched.planners.rust_spacetime_astar import RustSpaceTimeAStarPlanner as RustPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator


def run_comparison(n_tasks: int = 100, n_agvs: int = 6, seed: int = 42, aisle_capacity: int = 2):
    print("=" * 75)
    print(f"RUST-02: End-to-End Simulation Comparison ({n_tasks} Tasks x {n_agvs} AGVs, Workshop)")
    print("=" * 75)

    layout_name = "workshop"
    sc = ScenarioConfig(
        n_tasks=n_tasks,
        n_agvs=n_agvs,
        seed=seed,
        aisle_capacity=aisle_capacity,
    )
    sim_cfg = SimulationConfig(max_steps=20_000)

    # 1. Run Python Planner
    print("\n[1/2] Running with Python SpaceTimeAStarPlanner...")
    net_base_py = build_layout(layout_name, aisle_capacity=aisle_capacity)
    net_py, tasks_py, agvs_py = generate_scenario(sc, network=net_base_py)
    py_planner = PyPlanner(1.0, 1.0, seed=seed)
    allocator_py = GreedyAllocator()
    sim_py = Simulator(
        net_py,
        tasks_py,
        agvs_py,
        py_planner,
        allocator_py,
        sim_cfg,
    )

    t0 = time.perf_counter()
    res_py = sim_py.run()
    t_py = time.perf_counter() - t0
    summary_py = res_py.summary()

    print(f"  > Python Finished in: {t_py:.3f} s")
    print(f"  > Simulation Steps:  {res_py.finished_step}")
    print(f"  > Tasks Completed:   {summary_py['tasks_completed']}/{n_tasks}")

    # 2. Run Rust Planner
    print("\n[2/2] Running with RustSpaceTimeAStarPlanner...")
    net_base_rust = build_layout(layout_name, aisle_capacity=aisle_capacity)
    net_rust, tasks_rust, agvs_rust = generate_scenario(sc, network=net_base_rust)
    rust_planner = RustPlanner(1.0, 1.0, seed=seed)
    allocator_rust = GreedyAllocator()
    sim_rust = Simulator(
        net_rust,
        tasks_rust,
        agvs_rust,
        rust_planner,
        allocator_rust,
        sim_cfg,
    )

    t0 = time.perf_counter()
    res_rust = sim_rust.run()
    t_rust = time.perf_counter() - t0
    summary_rust = res_rust.summary()

    print(f"  > Rust Finished in:   {t_rust:.3f} s")
    print(f"  > Simulation Steps:  {res_rust.finished_step}")
    print(f"  > Tasks Completed:   {summary_rust['tasks_completed']}/{n_tasks}")

    # Assert correctness
    assert summary_py['tasks_completed'] == summary_rust['tasks_completed'] == n_tasks, "Task completion mismatch!"

    speedup = t_py / t_rust if t_rust > 0 else 0
    time_saved_pct = (t_py - t_rust) / t_py * 100

    print("\n" + "=" * 75)
    print("ACCELERATION SUMMARY:")
    print(f"  Python Wall-Clock Time: {t_py:.3f} s")
    print(f"  Rust Wall-Clock Time:   {t_rust:.3f} s")
    print(f"  End-to-End Speedup:     {speedup:.2f}x faster ({time_saved_pct:.1f}% time saved)")
    print("=" * 75)

    # Save benchmark result
    result_data = {
        "scenario": "workshop",
        "n_tasks": n_tasks,
        "n_agvs": n_agvs,
        "seed": seed,
        "python": {
            "wall_clock_s": t_py,
            "steps": res_py.finished_step,
            "tasks_completed": summary_py["tasks_completed"],
        },
        "rust": {
            "wall_clock_s": t_rust,
            "steps": res_rust.finished_step,
            "tasks_completed": summary_rust["tasks_completed"],
        },
        "speedup": speedup,
        "time_saved_pct": time_saved_pct,
    }

    out_file = Path("results/rust_transition/end_to_end_speedup.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved benchmark results to {out_file}")


if __name__ == "__main__":
    run_comparison(100, 6, 42)
