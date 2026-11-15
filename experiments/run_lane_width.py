"""车道数是不是那个瓶颈：同一车间场景，单车道 vs 双车道。

MAP-01F 发现车间在 4 台车时崩溃，而三个抽象布局不会。差别指向车道数，但那是推断。
这一组把车道数单独拎出来：同一个场景文件、同一批任务、同一算法，只改通道容量。

注意：双车道**不是** v1 场景声明的几何（场景里全部 lane_count=1）。这是一次敏感性
分析，用来量化"车道数"这一项的边际影响，不能当成场景的真实配置引用。
"""

from __future__ import annotations

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


def run_case(lane_capacity: int, n_tasks: int, n_agvs: int, seed: int):
    scene = WorkshopScene.load()
    net = scene.network(lane_capacity=lane_capacity)
    cfg = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed)
    _n, tasks, agvs = generate_scenario(
        cfg, network=net,
        pairs=legal_task_pairs(scene),
        start_pool=parking_slots(scene),
    )
    sim = Simulator(
        net, tasks, agvs,
        StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=500_000),
    )
    return sim.run().summary()


def main() -> int:
    n_tasks = 30
    fleets = [4, 6]
    seeds = [0, 1, 2]
    out = ROOT / "results" / "lane_width.csv"
    fields = ["lane_capacity", "n_tasks", "n_agvs", "seed", "completed",
              "makespan_s", "on_time_rate", "deadlock_breaks", "collision_events"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for cap in (1, 2):
            for n_agvs in fleets:
                for seed in seeds:
                    s = run_case(cap, n_tasks, n_agvs, seed)
                    w.writerow({
                        "lane_capacity": cap,
                        "n_tasks": n_tasks,
                        "n_agvs": n_agvs,
                        "seed": seed,
                        "completed": int(s["tasks_completed"]),
                        "makespan_s": round(float(s["makespan_seconds"]), 1),
                        "on_time_rate": round(float(s["on_time_rate"]), 4),
                        "deadlock_breaks": int(s["deadlock_breaks"]),
                        "collision_events": int(s["collision_events"]),
                    })
                    fh.flush()
                    print(cap, n_agvs, seed, int(s["tasks_completed"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())