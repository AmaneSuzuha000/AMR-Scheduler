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

**AMR-Scheduler** (`v1.2.0`) is the open-source code and digital-twin companion repository for **`CoRe-Sched`** (*Commitment-aware Rescheduling for Production–Multi-AMR Coordination*), a dual-engine learning-and-optimization platform for dynamic kitting-constrained production and multi-AMR rescheduling in discrete-workstation, narrow-aisle railway maintenance workshops.

<p align="center">
  <img src="results/figures/fig1_core_sched_overview.png" alt="CoRe-Sched Dual-Engine Closed-Loop Overview" width="96%" />
</p>

### Core Framework (`CoRe-Sched`)
- **Learning Engine (`AHG-RL`)** (`amr_sched/rl/v19_hetero_graph.py`, `v19_hetero_constructor.py`, `causal_repair_cone.py`):
  - **5-Entity Heterogeneous Graph Attention Encoder**: Encodes `Operation`, `Task`, `AMR`, `Bay`, and `Crew` nodes, directed precedence DAGs, bipartite kitting links, and corridor congestion/detour edge features via 2-layer cross-domain message passing.
  - **Four-Stage Autoregressive Semantic Constructor (`BC -> KL-Anchored SMDP-PPO`)**: Sequentially decodes *Task Prioritization*, *Resource Assignment* (bay & crew), *Fleet Dispatch* (dynamic load + soft-commitment retention bias), and *Queue Sequencing*, with exact (`< 1e-5`) pure-NumPy inference parity for torch-free ROS 2 deployment.
  - **Contextual Policy Selection (`Contextual LinUCB`) + Guarded Dual-Arm Probe + $J_A$-Aligned Scorer**: Observes a 6-D topology-agnostic coupling descriptor $\phi(s_t) \in [0, 1]^6$ to adaptively switch between `Full Joint` and `Staged` PPO policies, reinforced by a fleet-guarded counterfactual dual-arm probe ($|\mathcal{R}| \ge 4$) and exact $J_A$-aligned candidate ranking.
- **Incremental Optimization Engine (`CVCR-IR`)** (`amr_sched/cvcr/evaluator.py`, `crates/`):
  - **Sequence-Differenced Incremental Reservation Cache (`CVCR-IR (Seq-Diff)`) + Footprint-Aware Anytime SLA Guard**: Computes exact per-AMR effective task sequences, reuses unchanged AMRs' routes and Rust half-open interval reservations in-place (`median |ΔJ| = 0.00`), re-routes only the modified fleet subset $\Delta\mathcal{R}$ via native Rust `SpaceTime A* / SIPP` (`41 us` single-agent search), and strictly bounds worst-tail wall-clock latency under `35 ms`.
- **Execution Layer (*Atomic Transaction Commit*) & Dual-Layer Digital Twin** (`amr_sched/cvcr/`, `tools/`, `RAWSimO.Live/`):
  - Enforces epoch freshness, irrevocable `LOCKED` commitment protection, and atomic fallback-to-`KEEP` across discrete simulation (`AMRReschedulingEnv`), ROS 2 / Nav2 / Gazebo physical closed-loop execution (`PhysicalCVCRCoordinator`), and the C# WPF digital twin.
  - **Unified Policy Registry (22 Registered Policies)** (`amr_sched/baselines/policy_registry.py`): Includes `core_sched` / `CoRe-Sched` (`V19_ADAPTIVE_RL`), `AHG-RL` ablations, internal CVCR optimizers (`V18_EA_JOINT`, `rule_v10`), architecture-faithful domain-adapted 2025–2026 direct joint production–AGV competitors (`IJPR25_DFJSPT_HMARL`, `ASC26_MA_TRANS`, `FGCS26_REN_GNN`, `SPGA_AMR`), static offline references (`COR26-Ref`, `EJOR26-Erdogan`), and broader 2025–2026 baselines (`IJCAI26_PGMPO`, `EJOR26_Wu_GRU_Attn`, `TII26_E2G_MoE`, `ICRA26_Kitting_HGNN`, `RAL26_DetourDelta`, `JAIR26_DyMAB`, `MIT26_TAPFPC_LNS`, `BALANS_IJCAI25`).

---

## Verified Benchmark Highlights (`v1.2.0` Three-Gate Closure)

### 1. Single-Session Fresh 120-Episode Frozen Blind Holdout (12 Methods, 100% Audit Pass)

Executed in **one single continuous uncontended wall-clock session** (`1,440` audited episodes, brand-new zero-overlap fresh blind seeds `[27101..27302]` & `[2901..2906]`; full per-episode CSV ledger in [`results/g3_fresh_blind_holdout.csv`](results/g3_fresh_blind_holdout.csv)). Methods marked `*` breach the `35 ms` real-time pooled P95 SLA.

| Method | Mean $J_A$ ↓ | Median $J_A$ ↓ | Comp (100%) | Pure $H_1$ ($H_2=0$) ↓ | Churn ↓ | **Pooled P95 (ms)** ↓ | **Pooled P99 (ms)** ↓ | Mean Ep-P95 (ms) | `Track 2` (50-Node) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CoRe-Sched` (`core_sched`)** | **5,763.0 (#1 SLA)** | 2,061.8 | **117/120 (97.5%)** | 4,873.6 | 23.9 | **32.43** | **34.91** | 33.42 | **17,390.8 (24/24)** |
| `V18-EA`* | 5,640.3 | **1,150.6** | 118/120 (98.3%) | 4,749.6 | 12.3 | *42.80* | *75.80* | 27.07 | 20,285.5 (22/24) |
| `COR26-SPGA`* | 6,155.3 | 2,282.1 | 116/120 (96.7%) | **4,446.3** | 57.5 | *64.09* | *76.53* | 62.67 | 17,089.6 (23/24) |
| `FGCS26-Ren-GNN` | 6,361.9 | 2,163.2 | 114/120 (95.0%) | 4,797.0 | 31.6 | 13.29 | 17.31 | 12.34 | 17,718.3 (24/24) |
| `IJCAI26-PGMPO` | 6,408.9 | 1,235.6 | 114/120 (95.0%) | 4,941.2 | 14.5 | 23.89 | 29.96 | 20.16 | 21,494.5 (22/24) |
| `AHG-RL (Full)` | 6,690.1 | 2,193.4 | 114/120 (95.0%) | 4,757.5 | 22.6 | 34.76 | 42.81 | 32.18 | 21,326.4 (22/24) |
| `Rule-V10`* | 6,847.5 | 1,935.3 | 117/120 (97.5%) | 5,610.3 | 16.6 | *39.13* | *54.26* | 42.78 | 19,928.3 (24/24) |
| `AHG-RL (Staged)` | 6,857.9 | 1,872.2 | 116/120 (96.7%) | 4,948.0 | 25.4 | 30.54 | 38.00 | 28.86 | 19,298.5 (23/24) |
| `ASC26-MA-Trans` | 6,987.1 | 2,570.3 | 112/120 (93.3%) | 4,885.9 | 27.6 | **12.66** | **17.41** | 11.55 | **15,440.4 (24/24)** |
| `RAL26-DetourDelta` | 7,197.7 | 2,424.6 | 113/120 (94.2%) | 5,413.1 | **10.8** | 28.70 | *72.35* | 27.74 | 24,505.1 (20/24) |
| `EJOR26-Wu-GRU` | 8,025.7 | 1,398.0 | 114/120 (95.0%) | 5,392.2 | 21.8 | 34.12 | 40.48 | 28.41 | 30,993.1 (19/24) |
| `IJPR25-DFJSP-T` | 8,736.4 | 2,146.7 | 113/120 (94.2%) | 5,344.3 | 29.5 | 16.45 | 22.86 | 14.10 | 26,423.0 (21/24) |

- **Strict Real-Time SLA Leadership**: `CoRe-Sched` ranks **#1 in Mean $J_A$ (`5,763.0`) and #1 in completion (`117/120`) among all SLA-compliant methods**, keeping even its **pooled wall-clock P99 (`34.91 ms`) strictly below `35 ms`**, whereas `V18-EA` breaches the SLA (`42.80 ms` pooled P95, `75.80 ms` P99; `-2.18%` non-significant overall mean difference $p = 0.0771$ while `CoRe-Sched` beats `V18-EA` by `-14.3%` on Track 2 and `+0.60%` on mutually-completed pure $H_1$).
- **Paired Gains vs. 2025–2026 Competitors**: Cuts mean $J_A$ by **`+34.04%`** vs. `IJPR25-DFJSP-T` ($p = 5.41 \times 10^{-4}$), **`+28.19%`** vs. `EJOR26-Wu-GRU` (95% CI `[+12.2%, +39.1%]`), **`+19.93%`** vs. `RAL26-DetourDelta` ($p = 0.0050$), **`+17.52%`** vs. `ASC26-MA-Trans` (`69W/51L`), **`+15.97%`** vs. `AHG-RL (Staged)`, **`+13.86%`** vs. `AHG-RL (Full)`, **`+10.08%`** vs. `IJCAI26-PGMPO`, **`+9.41%`** vs. `FGCS26-Ren-GNN`, and **`+6.37%`** vs. `COR26-SPGA` (`73W/47L`, `2×` faster P95, `58.4%` less churn).
- **Pure Zero-Fault Static Common-Subproblem Reference (54 Instances, `0 / 270` Violations)**: Executed end-to-end inside the identical zero-fault `Simulator` + `IndependentAuditor` ([`results/g1_static_reference_subproblem.csv`](results/g1_static_reference_subproblem.csv)). `COR26-Ref` (Phase-1 un-cut CP-SAT physical relaxation `LB_phys = 68.2 / 90.2 / 438.7`, `Sim-UB = 405.1 / 326.3 / 785.5` in `3.7–5.8 s`) and `EJOR26-Erdogan` (`ILS-LP`, `284.1 / 573.5 / 664.5` in `1.2–8.7 s`) verify `LB_phys <= H1_sim` across every solver with zero violations while quantifying the exact price of causal millisecond rescheduling.
- **`CVCR-IR (Seq-Diff)` Cache Synergy, Physical Reconciliation & Scaling**: Restores isolated neural candidate cache hit rate from **`0.00%`** (`CVCR-IR Base`) to **`73.68%`** (`1.48×` candidate speedup, `median |ΔJ| = 0.00`; `86.8%–94.7%` closed-loop hits). 5-seed ROS 2 / Nav2 / Gazebo physical reconciliation achieves Track 1 Spearman $\rho = 0.7857$ (`7.18 ms` physical P95) and 10-AMR scaling sustains **`23.57 ms < 35 ms`** P95 (`2.21×` faster than full rebuild).

<p align="center">
  <img src="results/figures/fig3_main_benchmark.png" alt="CoRe-Sched 120-Episode Main Benchmark and Latency-Quality Pareto Frontier" width="96%" />
</p>

<p align="center">
  <img src="results/figures/fig6_pure_h1_and_lbbd.png" alt="Pure Physical Efficiency H1 Attribution and Offline COR 2026 LBBD/CP Exact Reference Bounds" width="96%" />
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
