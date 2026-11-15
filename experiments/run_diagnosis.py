"""主方法 vs 朴素基线的机制诊断（R5.6 补充）。

测三个量，回答"为什么主方法更慢"：
  1. 完成时间与冲突等待（已知主方法更差）
  2. 绕行系数 = 实际里程 / 静态最短路（是否在绕路）
  3. 长阻塞段占比（是否被自己的预约卡住）
"""
import csv, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.baseline import NearestShortestAllocator
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator

FIELDS = ["strategy", "n_tasks", "n_agvs", "seed", "tasks_completed",
          "makespan_s", "conflict_wait_steps", "distance_m", "detour_ratio",
          "max_block_run", "mean_block_run", "n_block_runs"]


def block_runs(r, aid):
    runs = []
    cur = 0
    for rec in r.trajectory:
        if rec.agv_states.get(aid) == "blocked":
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def main():
    rows = []
    strategies = [
        ("nearest_shortest", StaticDijkstraPlanner, NearestShortestAllocator),
        ("dijkstra+greedy", StaticDijkstraPlanner, GreedyAllocator),
        ("spacetime+greedy", SpaceTimeAStarPlanner, GreedyAllocator),
    ]
    total = len(strategies) * 9 * 5
    i = 0
    for name, pcls, acls in strategies:
        for n in (20, 50, 100):
            for a in (2, 4, 6):
                for sd in range(5):
                    i += 1
                    sc = ScenarioConfig(n_tasks=n, n_agvs=a, seed=sd)
                    net, tasks, agvs = generate_scenario(sc)
                    sim = Simulator(net, tasks, agvs,
                                    pcls(max_speed=sc.max_speed, dt=sc.dt),
                                    acls(),
                                    SimulationConfig(dt=sc.dt,
                                                     max_steps=500_000,
                                                     record_trajectory=True))
                    r = sim.run()
                    r.total_distance = sim._odometer
                    s = r.summary()
                    allruns = []
                    for agv in r.agvs:
                        allruns += block_runs(r, agv.agv_id)
                    rows.append({
                        "strategy": name,
                        "n_tasks": n, "n_agvs": a, "seed": sd,
                        "tasks_completed": int(s["tasks_completed"]),
                        "makespan_s": round(float(s["makespan_seconds"]), 2),
                        "conflict_wait_steps": int(s["conflict_wait_steps"]),
                        "distance_m": round(float(sim._odometer), 1),
                        "detour_ratio": round(float(r.detour_ratio(net)), 4),
                        "max_block_run": max(allruns) if allruns else 0,
                        "mean_block_run": round(
                            sum(allruns)/len(allruns), 2) if allruns else 0,
                        "n_block_runs": len(allruns),
                    })
                    if i % 15 == 0:
                        print("[%3d/%d] %s %d任务%d车 -> %.0fs 绕行%.3f"
                              % (i, total, name, n, a,
                                 s["makespan_seconds"],
                                 r.detour_ratio(net)), flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/diagnosis.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print()
    print("写入 results/diagnosis.csv（%d 行）" % len(rows))


if __name__ == "__main__":
    main()