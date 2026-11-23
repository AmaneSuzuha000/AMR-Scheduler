"""主方法在车间场景上的复现：静态 Dijkstra vs 时空 A*、三种分配器。

在具有完整工艺拓扑的铁路检修车间场景下，
如果结论只在环网上成立，那它就只是"这张图上的现象"。这一组用来检查外部效度。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amr_sched.allocation.baseline import FifoShortestAllocator, NearestShortestAllocator
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.workshop import WorkshopScene, legal_task_pairs, parking_slots

PLANNERS = {"static_dijkstra": StaticDijkstraPlanner, "spacetime_astar": SpaceTimeAStarPlanner}
ALLOCATORS = {"greedy": GreedyAllocator, "nearest": NearestShortestAllocator, "fifo": FifoShortestAllocator}


def run(planner_name: str, alloc_name: str, n_tasks: int, n_agvs: int, seed: int):
    scene = WorkshopScene.load()
    net = scene.network()
    cfg = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed)
    _n, tasks, agvs = generate_scenario(
        cfg, network=net, pairs=legal_task_pairs(scene), start_pool=parking_slots(scene))
    sim = Simulator(net, tasks, agvs, PLANNERS[planner_name](max_speed=1.0, dt=1.0),
                    ALLOCATORS[alloc_name](), SimulationConfig(dt=1.0, max_steps=500_000))
    return sim.run().summary()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=30)
    ap.add_argument("--agvs", type=int, default=4)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default="results/workshop_methods.csv")
    args = ap.parse_args(argv)

    fields = ["planner", "allocator", "n_tasks", "n_agvs", "seed", "completed",
              "makespan_s", "on_time_rate", "agv_utilization", "conflict_wait_steps",
              "deadlock_breaks", "collision_events", "planning_retries", "hard_failures"]
    seeds = [int(x) for x in args.seeds.split(",")]
    with (ROOT / args.out).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for pname in PLANNERS:
            for aname in ALLOCATORS:
                for seed in seeds:
                    s = run(pname, aname, args.tasks, args.agvs, seed)
                    w.writerow({
                        "planner": pname, "allocator": aname,
                        "n_tasks": args.tasks, "n_agvs": args.agvs, "seed": seed,
                        "completed": int(s["tasks_completed"]),
                        "makespan_s": round(float(s["makespan_seconds"]), 1),
                        "on_time_rate": round(float(s["on_time_rate"]), 4),
                        "agv_utilization": round(float(s.get("agv_utilization", 0.0)), 4),
                        "conflict_wait_steps": int(s["conflict_wait_steps"]),
                        "deadlock_breaks": int(s["deadlock_breaks"]),
                        "collision_events": int(s["collision_events"]),
                        "planning_retries": int(s.get("planning_retries", 0)),
                        "hard_failures": int(s.get("hard_failures", 0)),
                    })
                    fh.flush()
                    print(pname, aname, seed, int(s["tasks_completed"]),
                          round(float(s["makespan_seconds"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())