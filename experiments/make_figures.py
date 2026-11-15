import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.viz import (corridor_load, plot_corridor_heatmap, plot_gantt,
                           plot_layout, plot_trajectories)

out = "results/figures"
os.makedirs(out, exist_ok=True)

sc = ScenarioConfig(n_tasks=20, n_agvs=3, seed=0)
net, tasks, agvs = generate_scenario(sc)
sim = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(1.0, 1.0), GreedyAllocator(),
                SimulationConfig(max_steps=200_000, record_trajectory=True))
res = sim.run()
print("makespan=%.0f finished=%d" % (res.makespan_seconds(), res.finished_step))

plot_layout(net, agvs, out_path=out + "/layout.png")
print("layout.png OK")
plot_gantt(res, out_path=out + "/gantt.png")
print("gantt.png OK")
plot_trajectories(net, res, out_path=out + "/trajectory.png")
print("trajectory.png OK")
plot_corridor_heatmap(net, res, out_path=out + "/corridor.png")
print("corridor.png OK")

load = corridor_load(net, res)
top = sorted(load.items(), key=lambda kv: -kv[1]["load"])[:3]
for (u, v), info in top:
    print("  繁忙通道 %s-%s 负荷=%.1f%% 容量=%d" % (u, v, info["load"] * 100, info["capacity"]))

for f in sorted(os.listdir(out)):
    print("  %-18s %7d B" % (f, os.path.getsize(os.path.join(out, f))))
