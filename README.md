# AMR-Scheduler (`CoRe-Sched`): Learning-Guided Joint Rescheduling with Commitment-Preserving Incremental Optimization

<div align="center">

[![CI](https://github.com/AmaneSuzuha000/AMR-Scheduler/actions/workflows/ci.yml/badge.svg)](https://github.com/AmaneSuzuha000/AMR-Scheduler/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Rust 2021](https://img.shields.io/badge/Rust-Core-DEA584?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![ROS 2](https://img.shields.io/badge/ROS_2-Humble-22314E?logo=ros&logoColor=white)](https://docs.ros.org/en/humble/)
[![Gazebo](https://img.shields.io/badge/Gazebo-Classic%20%2F%20Fortress-FF6F00?logo=gazebo&logoColor=white)](https://gazebosim.org/)
[![.NET 6.0](https://img.shields.io/badge/.NET-6.0%20WPF-512BD4?logo=dotnet&logoColor=white)](https://dotnet.microsoft.com/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

**English** | [简体中文](README_zh.md)

</div>

---

## Overview

**AMR-Scheduler** (`v1.0.0`) is the open-source code and digital-twin companion repository for **`CoRe-Sched`** (*Commitment-aware Rescheduling for Production–Multi-AMR Coordination*), a dual-engine learning-and-optimization platform for dynamic kitting-constrained production and multi-AMR rescheduling in discrete-workstation, narrow-aisle railway maintenance workshops.

<p align="center">
  <img src="results/figures/fig1_core_sched_overview.png" alt="CoRe-Sched Dual-Engine Closed-Loop Overview" width="96%" />
</p>

### Core Framework (`CoRe-Sched`)
- **Learning Engine (`AHG-RL`)** (`amr_sched/rl/v19_hetero_graph.py`, `v19_hetero_constructor.py`):
  - **5-Entity Heterogeneous Graph Attention Encoder**: Encodes `Operation`, `Task`, `AMR`, `Bay`, and `Crew` nodes, directed precedence DAGs, bipartite kitting links, and corridor congestion/detour edge features via 2-layer cross-domain message passing.
  - **Four-Stage Autoregressive Semantic Constructor (`BC -> KL-Anchored SMDP-PPO`)**: Sequentially decodes *Task Prioritization*, *Resource Assignment* (bay & crew), *Fleet Dispatch* (dynamic load + soft-commitment retention bias), and *Queue Sequencing*, with exact (`< 1e-5`) pure-NumPy inference parity for torch-free ROS 2 deployment.
  - **Contextual Policy Selection (`Contextual LinUCB`)**: Observes a 6-D topology-agnostic coupling descriptor $\phi(s_t) \in [0, 1]^6$ to adaptively switch between `Full Joint` and `Staged` PPO policies online.
- **Incremental Optimization Engine (`CVCR-IR`)** (`amr_sched/cvcr/evaluator.py`, `crates/`):
  - **Sequence-Differenced Incremental Reservation Cache (`CVCR-IR (Seq-Diff)`)**: Computes exact per-AMR effective task sequences, reuses unchanged AMRs' routes and Rust half-open interval reservations in-place (`median |ΔJ| = 0.00` relative to full rebuild), and re-routes only the modified fleet subset $\Delta\mathcal{R}$ via native Rust `SpaceTime A* / SIPP` (`41 us` single-agent search).
- **Execution Layer (*Atomic Transaction Commit*) & Dual-Layer Digital Twin** (`amr_sched/cvcr/`, `tools/`, `RAWSimO.Live/`):
  - Enforces epoch freshness, irrevocable `LOCKED` commitment protection, and atomic fallback-to-`KEEP` across discrete simulation (`AMRReschedulingEnv`), ROS 2 / Nav2 / Gazebo physical closed-loop execution (`PhysicalCVCRCoordinator`), and the C# WPF digital twin.
  - **Unified Policy Registry** (`amr_sched/baselines/policy_registry.py`): Supports `core_sched` / `CoRe-Sched` (`V19_ADAPTIVE_RL`), `AHG-RL` ablations, internal CVCR optimizers (`V18_EA_JOINT`, `rule_v10`), and 2025–2026 top-venue baselines (`IJCAI26_PGMPO`, `EJOR26_Wu_GRU_Attn`, `TII26_E2G_MoE`, `ICRA26_Kitting_HGNN`, `RAL26_DetourDelta`, `JAIR26_DyMAB`, `MIT26_TAPFPC_LNS`, `SPGA_AMR`, `BALANS_IJCAI25`).

---

## Verified Benchmark Highlights

### 1. 120-Episode Unseen Blind Holdout (Identical Real Wall-Clock Execution, 100% Audit Pass)

| Method | 100%-Comp. (`/120`) | Mean $J_A$ ↓ | Mean $C_{\max}$ ↓ | Tardiness ($2T$) ↓ | Unfinished ($H_2$) | `Track 1` (62-Node) ↓ | `Track 2` (50-Node) ↓ | Wall P95 (ms) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CoRe-Sched` (`core_sched`)** | **120/120** | **5,425.53 (#1)** | **476.8 (#1)** | **4,568.8 (#1)** | **0.0 (#1)** | **4,622.9 (#1)** | **17,533.8 (#1)** | **12.51 (#1)** |
| `AHG-RL (Staged)` | 118/120 | 5,974.11 | 499.4 | 4,962.0 | 108.3 | 5,507.2 | 18,454.0 | 30.29 |
| `AHG-RL (Full)` | 118/120 | 6,001.36 | 479.4 | 4,842.3 | 275.0 | 5,085.6 | 18,834.0 | 32.17 |
| `Rule-V10 + CVCR-IR` | 117/120 | 6,034.34 | 497.9 | 5,033.1 | 250.0 | 6,805.2 | 21,009.8 | 57.36 |
| `V18-EA + CVCR-IR` | 114/120 | 6,221.95 | 484.9 | 5,001.6 | 550.0 | 6,292.4 | 23,039.6 | 26.15 |
| `IJCAI26-PGMPO` | 115/120 | 6,484.58 | 493.3 | 5,043.3 | 708.3 | 6,910.2 | 22,584.9 | 27.22 |
| `HG-BC (Pretrain)` | 115/120 | 7,407.27 | 506.2 | 5,220.3 | 1,266.7 | 6,039.0 | 24,934.2 | 32.20 |
| `EJOR26-Wu-GRU` | 116/120 | 7,988.12 | 506.2 | 5,977.0 | 1,158.3 | 7,420.1 | 29,387.8 | 34.71 |
| `RAL26-DetourDelta` | 115/120 | 8,048.61 | 561.5 | 6,515.6 | 750.0 | 7,462.8 | 24,755.9 | 32.66 |

- **Paired Wilcoxon Attribution (120 Episodes)**: `CoRe-Sched` beats fixed `AHG-RL (Full)` by **`-9.59%`** ($p = 0.0085$, `84W/36L`, **`2.57×`** faster wall-clock P95), fixed `AHG-RL (Staged)` by **`-9.18%`** ($p = 0.0200$, `81W/39L`), `HG-BC` by **`-26.75%`** ($p = 1.25 \times 10^{-4}$), and `IJCAI26-PGMPO` by **`-16.33%`** (95% Bootstrap CI `[-2,363.57, -0.02]`).
- **`CVCR-IR (Seq-Diff)` Cache Synergy**: Restores isolated neural candidate cache hit rate from **`0.00%`** (`CVCR-IR Base`) to **`73.68%`** (`1.48×` candidate speedup, `median |ΔJ| = 0.00`) and sustains **`86.8%–94.7%`** runtime-measured hit rates in closed-loop execution.
- **Physical Reconciliation & 62-Node Scaling**: 5-seed unified discrete–physical reconciliation achieves Track 1 Spearman $\rho = 0.7857$ (Makespan $\rho = 1.0000$, `CoRe-Sched` physical P95 = **`7.18 ms`**) and Track 2 Rank #2 (`4,220.0`, P95 = **`11.44 ms < 35 ms`**). At 10 AMRs on the 62-node workshop, `CoRe-Sched` sustains **`23.57 ms < 35 ms`** P95 latency (**`2.21×` faster** than full-rebuild `52.06 ms`).

<p align="center">
  <img src="results/figures/fig3_main_benchmark.png" alt="CoRe-Sched 120-Episode Main Benchmark and Latency-Quality Pareto Frontier" width="96%" />
</p>

<p align="center">
  <img src="results/figures/gazebo_overview.png" alt="1:25 Gazebo Wheelset Workshop Digital Twin" width="48%" />
  <img src="results/figures/layout.png" alt="Workshop Topology & Station Layout" width="48%" />
</p>

---

## Quick Start

```bash
git clone https://github.com/AmaneSuzuha000/AMR-Scheduler.git
cd AMR-Scheduler

# 1. Build native Rust spatiotemporal kernel (reservation_core & spacetime_planner)
cargo build --release --manifest-path crates/Cargo.toml

# 2. Install Python package and dependencies
pip install -r requirements.txt
pip install -e ".[dev,viz,web,rl]"

# 3. Run workshop simulation, CoRe-Sched dual-engine benchmarks & figures
python -m amr_sched --tasks 50 --agvs 6 --layout workshop --planner rust_spacetime_astar
python -B experiments/run_v19_exp08_learning_b2_synergy.py
python -B experiments/run_dual_layer_reconciliation.py --multi-seed
python -B experiments/run_workshop_scaling_4axis.py
python -B experiments/plot_v19_paper_figures.py

# 4. Run verification test suites (Python, Rust, C#, Manifest)
python -m pytest -q
cargo test --manifest-path crates/Cargo.toml
dotnet test Tests/RAWSimO.Live.Tests/RAWSimO.Live.Tests.csproj -c Release
python tools/data_manifest.py --check
```

---

## Project Structure

```text
AMR-Scheduler/
├── amr_sched/                  # CoRe-Sched Python core, CVCR-IR engine, RL env, and simulator
│   ├── allocation/             # Dispatching rules (FIFO/EDD/SPT/CR), Hungarian & composable allocators
│   ├── baselines/              # Unified policy registry (`core_sched`), pure-NumPy backends & 2025-2026 baselines
│   ├── checkpoints/            # Packaged pure-NumPy (.npz) and PyTorch (.pt) inference weights
│   ├── cvcr/                   # Causal snapshot, CVCR-IR (Seq-Diff) evaluator, frontier synthesis, Owner & Auditor
│   ├── planners/               # Dijkstra, Spacetime A*, Rust A*, Rust batch SIPP
│   ├── rl/                     # AHG-RL 5-entity Hetero-GNN constructor, Contextual LinUCB & Gymnasium SMDP env
│   └── viz/                    # Workshop layout, Gantt, corridor heatmaps, trajectories, animations
├── crates/                     # Native Rust Cargo workspace (reservation_core & spacetime_planner)
├── RAWSimO.Live/               # C# (.NET 6) live TCP JSONL protocol consumer & topology tracker
├── RAWSimO.Visualization/      # C# WPF 2D visualizer & live monitoring UI
├── tools/                      # Benchmarks, ROS 2 / Nav2 / Gazebo bridges, 1:25 wheelset scene
├── experiments/                # Training, mechanism audits, dual-layer reconciliation, 4-axis scaling & plotting
├── results/                    # Verified CSV datasets, MANIFEST.json, 5 publication figures (`fig1`–`fig5`)
├── docs/architecture/          # System architecture specification and ADRs
├── web/                        # FastAPI WebSocket monitoring & replay console
└── Tests/                      # Python pytest suites + C# xUnit test suites
```

---

## Documentation & License

- **Architecture & Protocols**: [Architecture Spec](docs/architecture/ARCHITECTURE.md), [ADR 001 (Rust FFI)](docs/architecture/decisions/001_cdylib_ctypes_architecture.md), [ADR 002 (Transaction & Auditor)](docs/architecture/decisions/002_transaction_auditor_data_flow.md), [PROTOCOLS.md](PROTOCOLS.md), [CHANGELOG.md](CHANGELOG.md)
- **License**: Original code is licensed under **GPL-3.0**. Upstream RAWSim-O components retain their original licenses. See [LICENSE](LICENSE) and [ThirdPartyLicenses.txt](ThirdPartyLicenses.txt).
