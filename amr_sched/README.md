# `amr_sched` — Multi-AMR Execution-Aware Scheduling & Learning Core

The core Python package for **AMR-Scheduler** (`v1.2.0`), unifying discrete-event workshop simulation, Rust zero-copy spatiotemporal planning, heterogeneous-graph critical-chain joint construction, execution-aware CVCR rescheduling, Gymnasium SMDP RL environments, and causal literature baselines.

## Quick Start

```bash
# Minimal simulation (20 tasks / 2 vehicles)
python -m amr_sched --tasks 20 --agvs 2 --seed 0

# Workshop layout with Rust Spacetime A* or continuous SIPP
python -m amr_sched --tasks 50 --agvs 6 --layout workshop --planner rust_spacetime_astar
python -m amr_sched --tasks 50 --agvs 6 --layout workshop --planner rust_sipp

# Run unit tests
python -m pytest -q
```

## Subpackage Layout

```text
amr_sched/
├── allocation/         # Task dispatch & matching rules (FIFO/EDD/SPT/PRIORITY/CR, Hungarian, composable)
├── baselines/          # Unified policy registry, pure-NumPy inference backends & 2024–2026 literature baselines
├── checkpoints/        # Packaged pure-NumPy (.npz) and PyTorch (.pt) inference weights
├── cvcr/               # Causal snapshot, B2 evaluator, frontier synthesis, window protection, P4 belief, Owner & Auditor
├── planners/           # Static Dijkstra, Spacetime A*, Rust Spacetime A*, Rust batch SIPP
├── rl/                 # Heterogeneous-graph joint constructor, Gymnasium AMRReschedulingEnv & controllers
├── viz/                # Workshop layout, Gantt, trajectories, corridor heatmaps, animations
├── network.py          # Node-edge logistics topology & bidirectional aisle modeling
├── reservation.py      # Reference Python spatiotemporal interval table
├── rust_reservation.py # Zero-copy C-ABI ctypes binding to crates/reservation_core
└── simulation.py       # Single-writer discrete-event simulator & state machine
```
