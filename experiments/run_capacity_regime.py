"""Applicability boundary of the reservation mechanism: fixed-duration throughput comparison.

Why throughput instead of completion time
--------------------------
With a single aisle (capacity 1) about 4% of the tasks are never completed (confirmed to be a genuine deadlock),
so the "completion time" is undefined there -- if the run does not finish, there is no makespan.

Instead we use **the number of tasks completed within a fixed duration** (throughput) as the metric:
  - with two aisles there is a well-defined makespan, and throughput is still comparable
  - with a single aisle the run does not finish, but "how much work gets done in the same amount of time" is still measurable
This is consistent with what RAWSim-O does (it also compares throughput over a fixed duration).

Hypothesis (to be verified)
--------------
Mechanism diagnosis suggests that the reservation mechanism is a net burden in low-conflict scenarios,
and only pays off in **high-conflict** scenarios (single aisle).
This experiment tests that hypothesis directly.

Usage::

    python experiments/run_capacity_regime.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.baseline import NearestShortestAllocator
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator

FIELDS = [
    "strategy", "aisle_capacity", "n_agvs", "seed", "horizon_steps",
    "tasks_total", "tasks_completed", "throughput_per_ks", "completion_rate",
    "conflict_wait_steps", "deadlock_breaks", "collision_events",
    "hard_failures", "distance_m",
]

STRATEGIES = [
    ("nearest_shortest", StaticDijkstraPlanner, NearestShortestAllocator),
    ("dijkstra+greedy", StaticDijkstraPlanner, GreedyAllocator),
    ("spacetime+greedy", SpaceTimeAStarPlanner, GreedyAllocator),
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, default=100)
    p.add_argument("--agvs", type=int, nargs="+", default=[2, 4, 6])
    p.add_argument("--capacities", type=int, nargs="+", default=[2, 1])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    # The horizon must be large enough; otherwise the two-aisle runs finish everything and
    # throughput saturates, leaving no discrimination. Measured: 100 tasks with 2 AGVs and two
    # aisles needs 10536 steps, and a single aisle needs even longer, hence 60000.
    p.add_argument("--horizon", type=int, default=60_000)
    p.add_argument("--out", default="results/capacity_regime.csv")
    args = p.parse_args(argv)

    rows = []
    total = (len(STRATEGIES) * len(args.capacities) * len(args.agvs)
             * len(args.seeds))
    i = 0
    for name, pcls, acls in STRATEGIES:
        for cap in args.capacities:
            for n_agvs in args.agvs:
                for seed in args.seeds:
                    i += 1
                    sc = ScenarioConfig(n_tasks=args.tasks, n_agvs=n_agvs,
                                        seed=seed, aisle_capacity=cap)
                    net, tasks, agvs = generate_scenario(sc)
                    sim = Simulator(net, tasks, agvs,
                                    pcls(max_speed=sc.max_speed, dt=sc.dt),
                                    acls(),
                                    SimulationConfig(dt=sc.dt,
                                                     max_steps=args.horizon))
                    r = sim.run()
                    r.total_distance = sim._odometer
                    s = r.summary()
                    done = int(s["tasks_completed"])
                    rows.append({
                        "strategy": name,
                        "aisle_capacity": cap,
                        "n_agvs": n_agvs,
                        "seed": seed,
                        "horizon_steps": args.horizon,
                        "tasks_total": args.tasks,
                        "tasks_completed": done,
                        "throughput_per_ks": round(
                            done / (args.horizon / 1000.0), 4),
                        "completion_rate": round(done / args.tasks, 4),
                        "conflict_wait_steps": int(s["conflict_wait_steps"]),
                        "deadlock_breaks": int(s["deadlock_breaks"]),
                        "collision_events": int(s["collision_events"]),
                        "hard_failures": int(s["hard_failures"]),
                        "distance_m": round(float(sim._odometer), 1),
                    })
                    print("[%3d/%d] %-16s cap=%d %d AGVs seed=%d -> done %3d/%d wait %5d deadlock breaks %3d"
                          % (i, total, name, cap, n_agvs, seed, done,
                             args.tasks, s["conflict_wait_steps"],
                             s["deadlock_breaks"]), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print()
    print("wrote %s (%d rows)" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
