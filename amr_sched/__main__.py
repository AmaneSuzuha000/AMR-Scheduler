"""Command-line entry point: ``python -m amr_sched --tasks 20 --agvs 2``"""

from __future__ import annotations

import argparse
import json
import sys

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import Simulator, SimulationConfig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Multi-AGV material delivery scheduling for railway maintenance workshops")
    p.add_argument("--tasks", type=int, default=20)
    p.add_argument("--agvs", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument(
        "--planner",
        choices=["spacetime_astar", "static_dijkstra"],
        default="spacetime_astar",
    )
    p.add_argument("--urgent-ratio", type=float, default=0.15)
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--reservation-window", type=int, default=3,
                   help="Number of path segments reserved ahead by space-time A* (0 = reserve the whole path)")
    p.add_argument("--reschedule", choices=[p.value for p in ReschedulePolicy],
                   default=ReschedulePolicy.NONE.value,
                   help="Rescheduling policy for urgent tasks")
    p.add_argument("--process-time", type=float, default=0.0, metavar="SEC",
                   help="Service time at the workstation, in seconds. 0 = completed on drop-off (legacy behavior)")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of a text summary")
    p.add_argument("--save-db", default=None, metavar="PATH",
                   help="Write this run to a SQLite database, including step-by-step trajectories")
    args = p.parse_args(argv)

    sc = ScenarioConfig(
        n_tasks=args.tasks,
        n_agvs=args.agvs,
        seed=args.seed,
        dt=args.dt,
        urgent_ratio=args.urgent_ratio,
        max_speed=1.0,
    )
    net, tasks, agvs = generate_scenario(sc)

    planner = (
        SpaceTimeAStarPlanner(max_speed=sc.max_speed, dt=sc.dt)
        if args.planner == "spacetime_astar"
        else StaticDijkstraPlanner(max_speed=sc.max_speed, dt=sc.dt)
    )
    sim = Simulator(
        net, tasks, agvs, planner, GreedyAllocator(),
        SimulationConfig(dt=args.dt, max_steps=args.max_steps, seed=args.seed,
                         reservation_window=args.reservation_window,
                         reschedule_policy=ReschedulePolicy(args.reschedule),
                         record_trajectory=bool(args.save_db),
                         process_time=args.process_time),
    )
    result = sim.run()
    result.total_distance = sim._odometer
    summary = result.summary()

    if args.save_db:
        from amr_sched.storage import ResultStore
        with ResultStore(args.save_db) as store:
            rid = store.save_run(
                result,
                label="cli/%s/%d/%d/seed%d"
                      % (planner.name, sc.n_tasks, sc.n_agvs, sc.seed),
                config=sc,
                record_steps=True)
        print(f"Saved to database {args.save_db} (run_id={rid})", file=sys.stderr)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Instance: tasks={sc.n_tasks} agvs={sc.n_agvs} seed={sc.seed} planner={planner.name}")
        print(f"  Completed        {summary['tasks_completed']:.0f}/{summary['tasks_total']:.0f}")
        print(f"  Completion time  {summary['makespan_seconds']:.1f} s")
        print(f"  On-time rate     {summary['on_time_rate']:.3f}")
        print(f"  AGV utilization  {summary['agv_utilization']:.3f}")
        print(f"  Conflict wait    {summary['conflict_wait_steps']:.0f} steps / {summary['conflict_wait_events']} events")
        print(f"  Urgent on-time   {summary['urgent_on_time_rate']:.3f}")
        print(f"  Urgent response  {summary['mean_urgent_response_steps']:.1f} steps")
        print(f"  Preemptions      {summary['preemption_count']:.0f}")
        print(f"  Planning retries {summary['planning_retries']:.0f}")
        print(f"  Hard failures    {summary['hard_failures']:.0f}")
        print(f"  Simulated steps  {summary['finished_step']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
