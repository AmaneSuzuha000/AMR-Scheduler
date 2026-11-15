"""车间场景能扛多少任务：任务数 × 车队规模的承载力扫描。

MAP-01F 只比了"能不能跑完 30 个任务"。这一组回答另一个问题：
把任务数往上加，车间什么时候开始送不完、送不完是排队还是死锁。
同时跑环网做对照——环网通道是双向 2 车道、区域是开环，若两条曲线形状不同，
说明差异来自场景结构而不是任务数本身。
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
from amr_sched.layouts import build_layout


def run_workshop(n_tasks: int, n_agvs: int, seed: int):
    scene = WorkshopScene.load()
    net = scene.network()
    cfg = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed)
    _n, tasks, agvs = generate_scenario(
        cfg, network=net, pairs=legal_task_pairs(scene), start_pool=parking_slots(scene))
    sim = Simulator(net, tasks, agvs, StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(), SimulationConfig(dt=1.0, max_steps=500_000))
    return sim.run().summary()


def run_ring(n_tasks: int, n_agvs: int, seed: int):
    net = build_layout("ring", aisle_capacity=2)
    cfg = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed)
    _n, tasks, agvs = generate_scenario(cfg, network=net)
    sim = Simulator(net, tasks, agvs, StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(), SimulationConfig(dt=1.0, max_steps=500_000))
    return sim.run().summary()


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="场景规模曲线")
    ap.add_argument("--tasks", default="10,20,30,40,60")
    ap.add_argument("--agvs", default="4,6")
    ap.add_argument("--layouts", default="workshop,ring")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results/workshop_scale.csv")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args(argv)
    tasks_lv = [int(x) for x in args.tasks.split(",")]
    agvs_lv = [int(x) for x in args.agvs.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    fns = {"workshop": run_workshop, "ring": run_ring}
    layouts = [x.strip() for x in args.layouts.split(",") if x.strip()]
    out = ROOT / args.out
    fields = ["layout", "n_tasks", "n_agvs", "seed", "completed", "makespan_s",
              "on_time_rate", "conflict_wait_steps", "deadlock_breaks", "collision_events"]
    with out.open("a" if args.append else "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not args.append:
            w.writeheader()
        for layout in layouts:
            fn = fns[layout]
            for n_tasks in tasks_lv:
                for n_agvs in agvs_lv:
                    for seed in seeds:
                        s = fn(n_tasks, n_agvs, seed)
                        w.writerow({
                            "layout": layout, "n_tasks": n_tasks, "n_agvs": n_agvs, "seed": seed,
                            "completed": int(s["tasks_completed"]),
                            "makespan_s": round(float(s["makespan_seconds"]), 1),
                            "on_time_rate": round(float(s["on_time_rate"]), 4),
                            "conflict_wait_steps": int(s["conflict_wait_steps"]),
                            "deadlock_breaks": int(s["deadlock_breaks"]),
                            "collision_events": int(s["collision_events"]),
                        })
                        fh.flush()
                        print(layout, n_tasks, n_agvs, seed, int(s["tasks_completed"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())