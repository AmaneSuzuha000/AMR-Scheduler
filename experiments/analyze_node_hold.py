"""Dwell-window sweep: the table in thesis 6.6.2.

Hypothesis 2 in thesis 6.6 was "the dwell reservation window is too long, so the
space-time planner holds stations far longer than it needs to, and that is why it
loses to the greedy baseline". The way to test it is to sweep node_hold_steps and
watch makespan. The answer turned out to be no: makespan barely moves (2945 ->
2906), so the window is not the reason.

That table had no generator for a long time - the numbers were transcribed from a
one-off run and could not be reproduced. This script is the generator.

Usage::

    python experiments/analyze_node_hold.py
    python experiments/analyze_node_hold.py --out results/node_hold_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator

#: Thesis 6.6.2: 50 tasks, 4 vehicles, space-time A*, ten seeds per setting.
N_TASKS = 50
N_AGVS = 4
SEEDS = tuple(range(10))
HOLDS = (60, 30, 10, 1)

FIELDS = ["node_hold_steps", "makespan_s", "conflict_wait_steps", "tasks_completed", "seeds"]


def run_one(hold: int, seed: int) -> dict:
    sc = ScenarioConfig(n_tasks=N_TASKS, n_agvs=N_AGVS, seed=seed)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(
        net, tasks, agvs,
        SpaceTimeAStarPlanner(max_speed=sc.max_speed, dt=sc.dt),
        GreedyAllocator(),
        SimulationConfig(dt=sc.dt, max_steps=100_000, seed=seed,
                         node_hold_steps=hold),
    )
    r = sim.run()
    s = r.summary()
    return {"makespan_s": float(s["makespan_seconds"]),
            "conflict_wait_steps": float(s["conflict_wait_steps"]),
            "tasks_completed": float(s["tasks_completed"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/node_hold_sweep.csv")
    args = parser.parse_args(argv)

    rows = []
    for hold in HOLDS:
        got = [run_one(hold, sd) for sd in SEEDS]
        rows.append({
            "node_hold_steps": hold,
            "makespan_s": round(st.mean(g["makespan_s"] for g in got), 1),
            "conflict_wait_steps": round(st.mean(g["conflict_wait_steps"] for g in got), 1),
            "tasks_completed": round(st.mean(g["tasks_completed"] for g in got), 2),
            "seeds": len(SEEDS),
        })
        print("  hold=%-3d makespan=%6.0f wait=%6.0f done=%.2f" % (
            hold, rows[-1]["makespan_s"], rows[-1]["conflict_wait_steps"],
            rows[-1]["tasks_completed"]))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print("written:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
