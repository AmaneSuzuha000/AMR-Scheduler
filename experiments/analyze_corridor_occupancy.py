"""Edge-level corridor occupancy: the table in thesis 6.8.4.

The mean corridor load is a network-wide average, so it cannot say whether the
network is uniformly loaded or whether a few trunk edges are saturated while the
rest sit idle. Thesis 6.8.4 therefore reports, per fleet size, the busiest edge,
the average edge and the busiest node, plus how many edges are occupied more
than half the time.

All four numbers come from one simulation per fleet size through the ordinary
scenario generator, so the table can be reproduced rather than transcribed.

Usage::

    python experiments/analyze_corridor_occupancy.py
    python experiments/analyze_corridor_occupancy.py --out results/corridor_occupancy.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.viz.corridor import corridor_load

#: The configuration thesis 6.8.4 pins down: 50 tasks, ring, capacity 2, seed 0.
N_TASKS = 50
SEED = 0
AISLE_CAPACITY = 2
FLEET = (2, 4, 6)

FIELDS = ["n_agvs", "busiest_edge", "busiest_edge_occupancy", "mean_edge_occupancy",
          "busiest_node", "busiest_node_occupancy", "edges_over_half", "steps"]


def run_one(n_agvs: int) -> dict:
    sc = ScenarioConfig(n_tasks=N_TASKS, n_agvs=n_agvs, seed=SEED,
                        aisle_capacity=AISLE_CAPACITY)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(
        net, tasks, agvs,
        StaticDijkstraPlanner(max_speed=sc.max_speed, dt=sc.dt),
        GreedyAllocator(),
        SimulationConfig(dt=sc.dt, max_steps=100_000, seed=SEED,
                         record_trajectory=True),
    )
    result = sim.run()
    steps = max(1, result.finished_step)

    # Edge occupancy: a position token "a->b" means the vehicle spent that step
    # on that undirected edge. corridor_load already aggregates exactly that.
    loads = corridor_load(net, result)
    per_edge = {k: v["occupied"] / steps for k, v in loads.items()}
    busiest_edge, busiest_edge_occ = max(per_edge.items(), key=lambda kv: kv[1])
    mean_edge_occ = sum(per_edge.values()) / len(per_edge) if per_edge else 0.0

    # Node occupancy: tokens without an arrow are plain node positions.
    node_steps: dict[str, int] = {}
    for rec in result.trajectory:
        for token in rec.agv_positions.values():
            if "->" in token:
                continue
            node_steps[token] = node_steps.get(token, 0) + 1
    busiest_node, busiest_node_steps = max(node_steps.items(), key=lambda kv: kv[1])

    return {
        "n_agvs": n_agvs,
        "busiest_edge": "%s-%s" % busiest_edge,
        "busiest_edge_occupancy": round(busiest_edge_occ, 4),
        "mean_edge_occupancy": round(mean_edge_occ, 4),
        "busiest_node": busiest_node,
        "busiest_node_occupancy": round(busiest_node_steps / steps, 4),
        "edges_over_half": sum(1 for v in per_edge.values() if v > 0.5),
        "steps": steps,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/corridor_occupancy.csv")
    args = parser.parse_args(argv)

    rows = [run_one(n) for n in FLEET]
    print("%-5s %-12s %10s %10s %-8s %10s %8s" % (
        "AGV", "busiest edge", "edge %", "mean edge %", "node", "node %", ">50%"))
    for r in rows:
        print("%-5d %-12s %9.1f%% %9.1f%% %-8s %9.1f%% %8d" % (
            r["n_agvs"], r["busiest_edge"],
            r["busiest_edge_occupancy"] * 100, r["mean_edge_occupancy"] * 100,
            r["busiest_node"], r["busiest_node_occupancy"] * 100,
            r["edges_over_half"]))

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
