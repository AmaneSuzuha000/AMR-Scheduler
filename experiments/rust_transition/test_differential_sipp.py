"""RUST-03: SIPP vs Spacetime A* Equivalence & Performance Benchmark.
"""

from __future__ import annotations

import random
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from amr_sched.layouts import build_layout
from amr_sched.planners.rust_spacetime_astar import RustSpaceTimeAStarPlanner
from amr_sched.planners.rust_sipp_planner import RustSippPlanner, is_rust_sipp_available
from amr_sched.rust_reservation import RustReservationTable
from amr_sched.reservation import Interval


def test_sipp_comparison(num_trials: int = 100):
    print("=" * 70)
    print(f"RUST-03: SIPP vs. Spacetime A* Equivalence Benchmark ({num_trials} Trials)")
    print("=" * 70)

    if not is_rust_sipp_available():
        raise RuntimeError("Rust SIPP library is not available.")

    astar_planner = RustSpaceTimeAStarPlanner(max_speed=1.0, dt=1.0)
    sipp_planner = RustSippPlanner(max_speed=1.0, dt=1.0)

    net = build_layout("workshop")
    node_names = list(net.nodes.keys())

    random.seed(42)
    astar_total_time = 0.0
    sipp_total_time = 0.0
    matches = 0

    for trial in range(num_trials):
        start = random.choice(node_names)
        goal = random.choice(node_names)
        while goal == start:
            goal = random.choice(node_names)

        # Synchronized reservations
        res_astar = RustReservationTable()
        res_sipp = RustReservationTable()

        for _ in range(random.randint(5, 15)):
            s = random.randint(0, 100)
            e = s + random.randint(3, 15)
            span = Interval(s, e)
            n = random.choice(node_names)
            res_astar.reserve_node(n, span, "OBSTACLE")
            res_sipp.reserve_node(n, span, "OBSTACLE")

        horizon = 300
        start_step = random.randint(0, 50)

        # 1. A*
        t0 = time.perf_counter()
        p_astar = astar_planner.plan(net, start, goal, start_step, res_astar, "AGV_1", horizon)
        astar_total_time += time.perf_counter() - t0

        # 2. SIPP
        t0 = time.perf_counter()
        p_sipp = sipp_planner.plan(net, start, goal, start_step, res_sipp, "AGV_1", horizon)
        sipp_total_time += time.perf_counter() - t0

        if p_astar is None:
            assert p_sipp is None, f"Mismatch at trial {trial}: A* was None, SIPP was {p_sipp}"
        else:
            assert p_sipp is not None, f"Mismatch at trial {trial}: A* succeeded, SIPP was None"
            # Both must reach the goal with the same optimal makespan / final arrival
            assert p_astar.arrival_steps[-1] == p_sipp.arrival_steps[-1], (
                f"Makespan mismatch at trial {trial}:\nA* arrival:   {p_astar.arrival_steps[-1]}\nSIPP arrival: {p_sipp.arrival_steps[-1]}"
            )

        matches += 1

    print(f"\nTotal Trials Passed: {matches}/{num_trials}")
    print(f"Total Spacetime A* Time: {astar_total_time * 1000:.2f} ms (avg {astar_total_time / num_trials * 1000:.3f} ms)")
    print(f"Total SIPP Time:         {sipp_total_time * 1000:.2f} ms (avg {sipp_total_time / num_trials * 1000:.3f} ms)")
    print("Verdict: 100% IDENTICAL OPTIMAL ARRIVAL TIME CONFIRMED")
    print("=" * 70)


if __name__ == "__main__":
    test_sipp_comparison(100)
