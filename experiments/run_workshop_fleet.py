"""车队边际收益：车间场景按 6.3 节的同一口径重跑。

6.3 节的结论是"2->4 台车省的时间明显多于 4->6"（step_ratio 0.30-0.35），
但那是在环网上。这一组把它搬到车间场景，看边际收益的形状是否相同。
口径：每格 5 个种子，取均值；s_per_vehicle_saved = 步内省下的秒数 / 新增车数。
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


def main() -> int:
    scene = WorkshopScene.load()
    fields = ["n_tasks", "n_agvs", "seed", "completed", "makespan_s", "on_time_rate",
              "agv_utilization", "conflict_wait_steps", "deadlock_breaks"]
    out = ROOT / "results" / "workshop_fleet.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for n_tasks in (20, 50, 100):
            for n_agvs in (2, 4, 6):
                for seed in range(5):
                    net = scene.network()
                    cfg = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed)
                    _n, tasks, agvs = generate_scenario(
                        cfg, network=net, pairs=legal_task_pairs(scene),
                        start_pool=parking_slots(scene))
                    s = Simulator(net, tasks, agvs, StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                                  GreedyAllocator(),
                                  SimulationConfig(dt=1.0, max_steps=1_500_000)).run().summary()
                    w.writerow({
                        "n_tasks": n_tasks, "n_agvs": n_agvs, "seed": seed,
                        "completed": int(s["tasks_completed"]),
                        "makespan_s": round(float(s["makespan_seconds"]), 1),
                        "on_time_rate": round(float(s["on_time_rate"]), 4),
                        "agv_utilization": round(float(s.get("agv_utilization", 0.0)), 4),
                        "conflict_wait_steps": int(s["conflict_wait_steps"]),
                        "deadlock_breaks": int(s["deadlock_breaks"]),
                    })
                    fh.flush()
                    print(n_tasks, n_agvs, seed, int(s["tasks_completed"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())