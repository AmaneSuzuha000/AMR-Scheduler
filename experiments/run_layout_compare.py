"""布局承载能力对照：抽象环网 vs 检修车间场景。

为什么要做这一组，以及它不能证明什么
------------------------------------
四个布局的规模不同（11/17/20/62 节点），所以直接比 makespan 没有意义——
"谁的完成时间短"只是"谁的地图小"。这一组要回答的是**承载能力**：
同一批任务、同样算法，车队加到几台开始崩（送不完 / 频繁打断死锁）。

任务来源也不同，必须写清楚：
  环网/网格/枢纽：按节点种类随机取送（抽象布局本来就没有工艺含义）；
  车间：按 14 种物料的合法路线取送（有工艺含义，不许随机）。
这不是不公平，是两种布局本身的性质差异——但它确实意味着两组的"任务难度"
不完全可比，所以报告里只比趋势，不比绝对值。
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

LAYOUTS = ["ring", "grid", "hub", "workshop"]


def build(layout: str, n_tasks: int, n_agvs: int, seed: int):
    if layout == "workshop":
        from amr_sched.workshop import build_scenario
        return build_scenario(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed)
    from amr_sched.layouts import build_layout
    net = build_layout(layout)
    cfg = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed)
    _n, tasks, agvs = generate_scenario(cfg)
    nodes = [n.node_id for n in net.nodes.values()]
    for i, t in enumerate(tasks):
        t.pickup = nodes[i % len(nodes)]
        t.dropoff = nodes[(i * 7 + 3) % len(nodes)]
        if t.pickup == t.dropoff:
            t.dropoff = nodes[(i * 7 + 4) % len(nodes)]
    for i, a in enumerate(agvs):
        a.start_node = nodes[i % len(nodes)]
    return net, tasks, agvs


def main() -> int:
    n_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    fleets = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else "1,2,3,4".split(","))]
    seeds = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else "0,1,2".split(","))]
    out = ROOT / "results" / "layout_compare.csv"
    fields = ["layout", "n_nodes", "n_edges", "n_tasks", "n_agvs", "seed",
              "completed", "makespan_s", "on_time_rate", "conflict_wait_steps",
              "deadlock_breaks", "collision_events"]
    new = not out.exists()
    done = set()
    if not new:
        with out.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                done.add((row["layout"], int(row["n_agvs"]), int(row["seed"])))
    with out.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        for layout in LAYOUTS:
            for n_agvs in fleets:
                for seed in seeds:
                    if (layout, n_agvs, seed) in done:
                        continue
                    net, tasks, agvs = build(layout, n_tasks, n_agvs, seed)
                    sim = Simulator(
                        net, tasks, agvs,
                        StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                        GreedyAllocator(),
                        SimulationConfig(dt=1.0, max_steps=500_000),
                    )
                    s = sim.run().summary()
                    w.writerow({
                        "layout": layout,
                        "n_nodes": len(net.nodes),
                        "n_edges": len(net.edges),
                        "n_tasks": n_tasks,
                        "n_agvs": n_agvs,
                        "seed": seed,
                        "completed": int(s["tasks_completed"]),
                        "makespan_s": round(float(s["makespan_seconds"]), 1),
                        "on_time_rate": round(float(s["on_time_rate"]), 4),
                        "conflict_wait_steps": int(s["conflict_wait_steps"]),
                        "deadlock_breaks": int(s["deadlock_breaks"]),
                        "collision_events": int(s["collision_events"]),
                    })
                    fh.flush()
                    print(layout, n_agvs, seed, int(s["tasks_completed"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())