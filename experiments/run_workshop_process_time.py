"""工位作业时间对车间的影响：process_time 从 0 到 300 s。

车间有 6 个工位、12 个装卸交接位，工位上的作业时间直接决定车辆被占住多久。
环网上做过 process_time 的五档扫描（6.5 节），但那时的"工位"只是抽象节点；
车间场景里工位与库存、交接位是分开的资源，值得单独看一遍。
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
    ap.add_argument("--tasks", type=int, default=30)
    ap.add_argument("--agvs", default="4,6")
    ap.add_argument("--times", default="0,60,120,300")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results/workshop_process_time.csv")
    args = ap.parse_args(argv)

    scene = WorkshopScene.load()
    fields = ["process_time_s", "n_tasks", "n_agvs", "seed", "completed", "makespan_s",
              "on_time_rate", "agv_utilization", "conflict_wait_steps", "deadlock_breaks"]
    with (ROOT / args.out).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for pt in [float(x) for x in args.times.split(",")]:
            for n_agvs in [int(x) for x in args.agvs.split(",")]:
                for seed in [int(x) for x in args.seeds.split(",")]:
                    net = scene.network()
                    cfg = ScenarioConfig(n_tasks=args.tasks, n_agvs=n_agvs, seed=seed)
                    _n, tasks, agvs = generate_scenario(
                        cfg, network=net, pairs=legal_task_pairs(scene),
                        start_pool=parking_slots(scene))
                    s = Simulator(net, tasks, agvs, StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                                  GreedyAllocator(),
                                  SimulationConfig(dt=1.0, process_time=pt,
                                                   max_steps=1_500_000)).run().summary()
                    w.writerow({
                        "process_time_s": pt, "n_tasks": args.tasks, "n_agvs": n_agvs,
                        "seed": seed, "completed": int(s["tasks_completed"]),
                        "makespan_s": round(float(s["makespan_seconds"]), 1),
                        "on_time_rate": round(float(s["on_time_rate"]), 4),
                        "agv_utilization": round(float(s.get("agv_utilization", 0.0)), 4),
                        "conflict_wait_steps": int(s["conflict_wait_steps"]),
                        "deadlock_breaks": int(s["deadlock_breaks"]),
                    })
                    fh.flush()
                    print(pt, n_agvs, seed, int(s["tasks_completed"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())