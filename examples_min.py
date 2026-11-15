"""最小可读示例：手工构造算例并跑一次仿真。"""

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.entities import Agv, Task, TaskPriority
from amr_sched.network import build_demo_network
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.simulation import SimulationConfig, Simulator

net = build_demo_network()

tasks = [
    Task("T1", pickup="D", dropoff="W1", release_step=0, due_step=200),
    Task("T2", pickup="B1", dropoff="W4", release_step=10, due_step=300),
    Task("T3", pickup="D", dropoff="S", release_step=20, due_step=250,
         priority=TaskPriority.URGENT),
]
agvs = [Agv("AGV0", start_node="B1"), Agv("AGV1", start_node="D")]

sim = Simulator(
    net, tasks, agvs,
    SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0),
    GreedyAllocator(),
    SimulationConfig(dt=1.0, max_steps=50_000),
)
result = sim.run()
for k, v in result.summary().items():
    print(f"{k:24s} {v}")
