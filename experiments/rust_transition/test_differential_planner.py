"""RUST-02 & RUST-04: Differential Testing between Python and Rust Spacetime A* Planners.
"""

from __future__ import annotations

import random
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from amr_sched.layouts import build_layout
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner as PyPlanner
from amr_sched.planners.rust_spacetime_astar import RustSpaceTimeAStarPlanner as RustPlanner, is_rust_planner_available
from amr_sched.reservation import Interval, ReservationTable as PyReservationTable
from amr_sched.rust_reservation import RustReservationTable


def test_differential_planning(num_trials: int = 100):
    print("=" * 70)
    print(f"RUST-02: Differential Spacetime A* Planning Verification ({num_trials} Trials)")
    print("=" * 70)

    if not is_rust_planner_available():
        raise RuntimeError("Rust planner library is not available.")

    py_planner = PyPlanner(max_speed=1.0, dt=1.0)
    rust_planner = RustPlanner(max_speed=1.0, dt=1.0)

    networks = [("Ring", build_layout("ring")), ("Workshop", build_layout("workshop"))]

    total_py_time = 0.0
    total_rust_time = 0.0
    trials_matched = 0

    random.seed(999)

    for net_name, net in networks:
        print(f"\n--- Testing on Layout: {net_name} ({len(net.nodes)} nodes, {len(net.edges)} edges) ---")
        node_names = list(net.nodes.keys())

        # Test with various obstacle patterns
        for trial in range(num_trials // len(networks)):
            start = random.choice(node_names)
            goal = random.choice(node_names)
            while goal == start:
                goal = random.choice(node_names)

            # Create synchronized reservation tables with dynamic traffic
            py_res = PyReservationTable()
            rust_res = RustReservationTable()

            # Insert 5 to 20 random reservations
            for _ in range(random.randint(5, 20)):
                s = random.randint(0, 100)
                e = s + random.randint(2, 10)
                span = Interval(s, e)
                n = random.choice(node_names)
                py_res.reserve_node(n, span, "AGV_OBSTACLE")
                rust_res.reserve_node(n, span, "AGV_OBSTACLE")

            horizon = 200
            start_step = random.randint(0, 50)

            # 1. Run Python Planner
            t0 = time.perf_counter()
            py_path = py_planner.plan(net, start, goal, start_step, py_res, "AGV_TEST", horizon)
            t_py = time.perf_counter() - t0
            total_py_time += t_py

            # 2. Run Rust Planner
            t0 = time.perf_counter()
            rust_path = rust_planner.plan(net, start, goal, start_step, rust_res, "AGV_TEST", horizon)
            t_rust = time.perf_counter() - t0
            total_rust_time += t_rust

            # Verify equivalence
            if py_path is None:
                assert rust_path is None, f"Mismatch at trial {trial}: Py was None, Rust was {rust_path}"
            else:
                assert rust_path is not None, f"Mismatch at trial {trial}: Py succeeded, Rust was None"
                assert py_path.nodes == rust_path.nodes, (
                    f"Nodes mismatch at trial {trial}:\nPy:   {py_path.nodes}\nRust: {rust_path.nodes}"
                )
                assert py_path.arrival_steps == rust_path.arrival_steps, (
                    f"Arrivals mismatch at trial {trial}:\nPy:   {py_path.arrival_steps}\nRust: {rust_path.arrival_steps}"
                )
                assert py_path.wait_steps == rust_path.wait_steps, (
                    f"Wait steps mismatch at trial {trial}:\nPy:   {py_path.wait_steps}\nRust: {rust_path.wait_steps}"
                )

            trials_matched += 1

    speedup = total_py_time / total_rust_time if total_rust_time > 0 else 0
    print("\n" + "=" * 70)
    print(f"Total Trials Tested: {trials_matched}/{num_trials}")
    print(f"Total Python Search Time: {total_py_time * 1000:.2f} ms")
    print(f"Total Rust Search Time:   {total_rust_time * 1000:.2f} ms")
    print(f"Average Python plan():    {total_py_time / trials_matched * 1000:.3f} ms")
    print(f"Average Rust plan():      {total_rust_time / trials_matched * 1000:.3f} ms")
    print(f"Search Engine Speedup:    {speedup:.2f}x FASTER!")
    print("Verdict: 100% IDENTICAL PATH QUALITY & TRAJECTORY EQUIVALENCE CONFIRMED")
    print("=" * 70)


if __name__ == "__main__":
    test_differential_planning(100)
