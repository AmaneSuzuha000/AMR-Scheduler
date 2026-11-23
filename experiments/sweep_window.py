"""预约窗口扫描：为"冲突消解"找真正划算的配置（预约窗口参数敏感性扫描）。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator

CELLS = [(20, 6), (50, 6), (100, 6), (50, 4), (100, 4)]
WINDOWS = [0, 1, 2, 3, 5]
SEEDS = [0, 1, 2, 3, 4]


def run(nt, na, win, seed):
    sc = ScenarioConfig(n_tasks=nt, n_agvs=na, seed=seed)
    net, tasks, agvs = generate_scenario(sc)
    if win is None:
        pl = StaticDijkstraPlanner(max_speed=1.0, dt=1.0)
        cfg = SimulationConfig(max_steps=500_000)
    else:
        pl = SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0)
        cfg = SimulationConfig(max_steps=500_000, reservation_window=win)
    r = Simulator(net, tasks, agvs, pl, GreedyAllocator(), cfg).run()
    return r.makespan_seconds(), r.conflict_wait_steps()


print("cell      baseline    " + "  ".join("w=%d   " % w for w in WINDOWS))
for nt, na in CELLS:
    b = [run(nt, na, None, s)[0] for s in SEEDS]
    bm = sum(b) / len(b)
    parts = []
    for w in WINDOWS:
        v = [run(nt, na, w, s)[0] for s in SEEDS]
        vm = sum(v) / len(v)
        parts.append("%+7.2f%%" % ((vm - bm) / bm * 100))
    print("%4d/%-3d %9.1f   " % (nt, na, bm) + " ".join(parts), flush=True)
