"""交叉口容量是不是 16 台车死锁的主因：把路口容量从 2 提到 3 / 4。

上一组诊断显示，16 台车时的死锁打断全部落在主通道交叉口（J_S_85 / J_N_45 等）。
如果瓶颈真是"路口一次只容得下 2 辆车"，那么把路口容量提到 3 应该直接见效；
如果没效，说明问题在通道而不是路口，之前的判断就要改。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.workshop import WorkshopScene, legal_task_pairs, parking_slots


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="2,3,4")
    ap.add_argument("--agvs", type=int, default=16)
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results/workshop_junction_capacity.csv")
    args = ap.parse_args(argv)

    scene = WorkshopScene.load()
    fields = ["junction_capacity", "n_tasks", "n_agvs", "seed", "completed",
              "makespan_s", "conflict_wait_steps", "deadlock_breaks", "collision_events"]
    with (ROOT / args.out).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for cap in [int(x) for x in args.caps.split(",")]:
            for seed in [int(x) for x in args.seeds.split(",")]:
                net = scene.network(aisle_capacity=cap)
                cfg = ScenarioConfig(n_tasks=args.tasks, n_agvs=args.agvs, seed=seed)
                _n, tasks, agvs = generate_scenario(
                    cfg, network=net, pairs=legal_task_pairs(scene),
                    start_pool=parking_slots(scene))
                s = Simulator(net, tasks, agvs, StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                              GreedyAllocator(),
                              SimulationConfig(dt=1.0, max_steps=1_500_000)).run().summary()
                w.writerow({
                    "junction_capacity": cap, "n_tasks": args.tasks, "n_agvs": args.agvs,
                    "seed": seed, "completed": int(s["tasks_completed"]),
                    "makespan_s": round(float(s["makespan_seconds"]), 1),
                    "conflict_wait_steps": int(s["conflict_wait_steps"]),
                    "deadlock_breaks": int(s["deadlock_breaks"]),
                    "collision_events": int(s["collision_events"]),
                })
                fh.flush()
                print(cap, seed, int(s["tasks_completed"]), int(s["deadlock_breaks"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())