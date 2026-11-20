# `amr_sched` — Multi-AMR Material Delivery Scheduling Core

The core Python package for **AMR-Scheduler** (`v0.3.0`), unifying discrete-event workshop simulation, Rust zero-copy spatiotemporal planning, the V10–V17 hierarchical LG-CVCR decision brain, Gymnasium SMDP RL environments, and causal literature baselines.

## Quick Start

```bash
# Minimal simulation (20 tasks / 2 vehicles)
python -m amr_sched --tasks 20 --agvs 2 --seed 0

# Workshop layout with Rust Spacetime A* or continuous SIPP
python -m amr_sched --tasks 50 --agvs 6 --layout workshop --planner rust_spacetime_astar
python -m amr_sched --tasks 50 --agvs 6 --layout workshop --planner rust_sipp

# Visualization report
python -m amr_sched.viz.report --tasks 20 --agvs 2 --seed 0

# Run tests
python -m pytest -q
```

## Subpackage Layout

```text
amr_sched/
├── allocation/         # Task dispatch & matching rules (FIFO/EDD/SPT/PRIORITY/CR, Hungarian, composable)
├── baselines/          # Causal adapters for 2024–2026 top-venue AGV/FJSP baselines
├── checkpoints/        # Packaged pure-NumPy .npz inference weights (L-RHO, V12–V17)
├── cvcr/               # Snapshot capture, B2 evaluator, V12–V17 policies, P4 belief, Owner & Auditor
├── planners/           # Static Dijkstra, Spacetime A*, Rust Spacetime A*, Rust batch SIPP
├── rl/                 # Gymnasium AMRReschedulingEnv, action masking, V10/V12/V13 controllers, CUDA helpers
├── viz/                # Workshop layout, Gantt, trajectories, corridor heatmaps, animations
├── network.py          # Node-edge logistics topology & bidirectional aisle modeling
├── reservation.py      # Reference Python spatiotemporal interval table
├── rust_reservation.py # Zero-copy C-ABI ctypes binding to crates/reservation_core
├── scenarios*.py       # Workshop, blind unseen, stress, and assembly kitting scenario generators
└── simulation.py       # Single-writer discrete-event simulator & state machine
```

## Design Invariants

- **Discrete-time physical consistency**: Fixed step `dt` (default `1.0 s`) aligns Python simulation and Rust integer half-open `[start, end)` intervals.
- **Physically unified bidirectional aisles**: `edge(u, v)` and `edge(v, u)` contend on a single shared directional resource.
- **Single-writer transactional commit**: Planners and policies emit `CandidateProposal` objects validated atomically by `NativeTransactionOwner`; active cargo commitments and mid-edge transits (`_transit`) are strictly non-preemptible.
- **Independent trajectory auditing**: `IndependentAuditor` verifies node capacity, valid directed edges, monotonic task timelines, kitting precedence, and exclusive station bay/crew/tool intervals step by step.
