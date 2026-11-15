# amr_sched — multi-AGV material delivery scheduling

A Python scheduler prototype covering the four stages of the project:
**R2 (modeling & method) / R3 (system design) / R4 (implementation) /
R5 (case validation)**.

## Quick start

```bash
# smallest instance (20 tasks / 2 vehicles)
python -m amr_sched --tasks 20 --agvs 2 --seed 0

# switch to the baseline planner for comparison
python -m amr_sched --tasks 20 --agvs 2 --planner static_dijkstra

# full instance matrix (3 scales × 3 fleets × 2 planners × 10 seeds)
python experiments/run_matrix.py --out results/matrix.csv

# paired analysis
python experiments/analyze_matrix.py --csv results/matrix.csv

# urgent-task dynamic rescheduling
python experiments/run_reschedule.py
python experiments/analyze_reschedule.py

# allocation strategy comparison
python experiments/run_allocation.py
python experiments/analyze_allocation.py

# visualization report
python -m amr_sched.viz.report --tasks 20 --agvs 2 --seed 0

# tests
python -m pytest Tests -q
```

## Structure

```
amr_sched/
├── network.py            node-edge network model
├── entities.py           task and AGV objects
├── reservation.py        space-time reservation table
├── simulation.py         discrete-time simulation loop
├── scenarios.py          instance generation
├── planners/
│   ├── base.py                   planner interface
│   ├── static_dijkstra.py        static shortest path (baseline)
│   └── spacetime_astar.py        space-time A* with conflict resolution
├── allocation/
│   ├── base.py                   allocator interface
│   ├── dispatch.py               5 dispatching rules (FIFO/EDD/SPT/PRIORITY/CR)
│   ├── greedy.py                 greedy allocation (urgent-first, legacy)
│   ├── hungarian.py              Hungarian algorithm (one-step optimal matching)
│   └── rule_based.py             composable rule × allocator
├── reschedule.py         urgent-task dynamic rescheduling
└── viz/                  process visualization
    ├── layout.py                 workshop layout
    ├── gantt.py                  AGV task Gantt chart
    ├── trajectory.py             trajectory replay
    ├── corridor.py               aisle load heatmap
    ├── metrics.py                metric and baseline comparison plots
    └── report.py                 one-shot figure set + Markdown report
```

## Four acceptance metrics

| Metric | Method | Unit |
|---|---|---|
| Completion time | `SimulationResult.makespan_seconds()` | s |
| On-time rate | `SimulationResult.on_time_rate()` | ratio |
| AGV utilization | `SimulationResult.agv_utilization()` | ratio |
| Conflict wait | `SimulationResult.conflict_wait_steps()` | steps |

## Design notes

* **Discrete time**: fixed step `dt` (default 1 s) keeps runs reproducible and
  consistent with the reservation table.
* **Directed graph, physically unified aisles**: `edge(u,v)` and `edge(v,u)` share
  one physical resource, so two vehicles cannot "each own an edge" on paper while
  colliding head-on in reality.
* **Planning is separate from execution**: the planner only produces a route; the
  executor re-checks occupancy every step.
* **Planning is separate from reservation**: planners only *read* the reservation
  table; all writes go through the simulator.

## Key results (measured — do not invert)

* **Urgent preemption works**: urgent on-time rate 0.451 → 0.542 (p<0.0001),
  response 180.7 → 142.7 steps (p<0.0001), with no significant degradation in
  total completion time (p=0.30).
* **Space-time A* does not beat the static baseline**: +0.2% to +8.3%. Do not
  write it up as "better than baseline".
* **The AGV inflection point is 4 vehicles**: 2→4 gives −46.1%, 4→6 only −26.4%.
* **The allocator matters more than the dispatching rule**: changing the allocator
  moves results by 8.8–13.2%, changing the rule by 1.4–4.1%. Completion time and
  on-time rate trade off against each other, so the choice must be explicit.
* **Collision events are always zero**, task completion rate 100% (verified over
  1260 runs).
