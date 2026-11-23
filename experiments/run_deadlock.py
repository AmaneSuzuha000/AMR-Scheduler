"""Single-aisle deadlock and prevention-policy instance (deepening of the stress case in the domain specification)

Background
----------
With two aisles (aisle_capacity=2) the system completes 100% of the tasks;
with a single aisle (=1) about 4% of the tasks are never completed. This is a **genuine deadlock**,
not a capacity bug (verify_capacity.py audited 54 runs step by step and found 0 node over-capacity events).

Root cause of the deadlock (localized to a code line)
----------------------------------------------------
The AGV has arrived next to its target node, but that node is occupied by another vehicle, so it stops in the
branch of `_advance_step`. That branch originally did **not** mark the blockage,
so `_break_deadlock` never reached its timeout (breaks stayed at 0) and the task hung forever.
After the fix breaks rose from 0 to 200+, yet the task still never finished, because the task returns to the queue
after a break, is reallocated, and falls into the same deadlock again, forming a **livelock**.

This experiment compares three handling policies
------------------------------------------------
1. ``none``         : no special handling (control group)
2. ``one_way``      : force one-way circulation on the main ring (yielding/detours remove circular waiting)
3. ``break_only``   : enable deadlock breaking only (the fixed blockage marking)

Usage::

    python experiments/run_deadlock.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.network import build_demo_network
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator

FIELDS = [
    "policy", "one_way_ring", "deadlock_timeout", "n_tasks", "n_agvs",
    "release_window", "seed", "tasks_completed", "completion_rate",
    "makespan_s", "conflict_wait_steps", "deadlock_breaks",
    "collision_events", "hard_failures",
]

POLICIES = [
    # (name, one-way ring?, deadlock timeout)
    ("none", False, 0),
    ("one_way", True, 0),
    ("break_only", False, 200),
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, default=50)
    p.add_argument("--agvs", type=int, nargs="+", default=[2, 4, 6])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--out", default="results/deadlock.csv")
    args = p.parse_args(argv)

    rows = []
    total = len(POLICIES) * len(args.agvs) * len(args.seeds)
    i = 0
    for name, one_way, timeout in POLICIES:
        for agvs in args.agvs:
            for seed in args.seeds:
                i += 1
                net = build_demo_network(aisle_capacity=1,
                                         one_way_ring=one_way)
                sc = ScenarioConfig(n_tasks=args.tasks, n_agvs=agvs,
                                    seed=seed, release_window=600,
                                    aisle_capacity=1)
                net, tasks, ag = generate_scenario(sc, network=net)
                r = Simulator(net, tasks, ag,
                              SpaceTimeAStarPlanner(1.0, 1.0),
                              GreedyAllocator(),
                              SimulationConfig(max_steps=60_000,
                                               deadlock_timeout=timeout)).run()
                s = r.summary()
                rec = {
                    "policy": name,
                    "one_way_ring": int(one_way),
                    "deadlock_timeout": timeout,
                    "n_tasks": args.tasks,
                    "n_agvs": agvs,
                    "release_window": 600,
                    "seed": seed,
                    "tasks_completed": int(s["tasks_completed"]),
                    "completion_rate": round(
                        s["tasks_completed"] / args.tasks, 4),
                    "makespan_s": round(float(s["makespan_seconds"]), 2)
                    if s["tasks_completed"] == args.tasks else "",
                    "conflict_wait_steps": int(s["conflict_wait_steps"]),
                    "deadlock_breaks": int(s["deadlock_breaks"]),
                    "collision_events": int(s["collision_events"]),
                    "hard_failures": int(s["hard_failures"]),
                }
                rows.append(rec)
                print("[%2d/%d] %-10s agv=%d seed=%d -> done %2d/%d completion_rate=%.2f breaks=%3d"
                      % (i, total, name, agvs, seed, rec["tasks_completed"],
                         args.tasks, rec["completion_rate"],
                         rec["deadlock_breaks"]), flush=True)

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
