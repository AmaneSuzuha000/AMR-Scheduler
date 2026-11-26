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

**AMR-Scheduler** (`v1.4.0`) is a long-term **multi-robot research infrastructure and digital-twin platform** uniting two complementary research tracks over shared maps, spatiotemporal planning kernels, physical execution models, and evaluation suites:

- **Track A — Spatiotemporal Planning & Physical Motion Execution (`Layers 2–3`, active focus)**:
  - **Layer 3 · Multi-Robot Traffic Coordination**: Congestion-aware `CPS-A*`, `EALR`, multi-tick edge-span corrected planning (`U1a_EdgeSpan_Fix`), and right-of-way arbitration.
  - **Layer 2 · Local Physical Motion Planning**: Deterministic reservation-margin primitive coordinator **`R1_Robust`**, `K=6` kinodynamic motion primitives, two-disc swept capsule geometry, and Gaussian uncertainty `BrakingInvariantShield`.
  - **Layer 1 · Physical Execution & Sensing**: ROS 2 / Nav2 / Gazebo digital twin and SCOUT Mini CAN chassis control.
- **Track B — Production–Multi-AMR Joint Rescheduling (`Layers 4–5`, frozen validated `CoRe-Sched` stack)**:
  - **Layer 5 · Joint Production–Resource Optimization**: Operation–bay–crew–transport coupling (`V18`, `V19`, `V20_1`, `RGRO`).
  - **Layer 4 · Dynamic Transport Task Scheduling**: Commitment-aware `CVCR-IR (Seq-Diff)`, `AHG-RL` heterogeneous constructor, and dispatching allocators.

> **Architectural Repositioning & Multi-Fidelity Roadmap**: Prior `CoRe-Sched` (`v1.0–v1.3`) work focused on Layers 4–5 atop simplified transport approximations. Current active work freezes the Layer 4–5 stack as a validated benchmark while hardening **Layers 2–3** (continuous kinematics, swept geometry, physical right-of-way, execution-aware ETA calibration). Future integration will connect both tracks via a **Multi-Fidelity Execution Interface** (fast discrete simulator for large-scale search/training paired with high-fidelity `CPS-A* + R1_Robust` validation; cross-fidelity integration and paired re-evaluation remain planned future work).

<p align="center">
  <img src="results/figures/fig1_core_sched_overview.png" alt="CoRe-Sched Dual-Engine Closed-Loop Overview" width="96%" />
</p>

### Core Framework (`CoRe-Sched`)
- **Learning & Causal Repair Engine (`AHG-RL` + `CausalRepairCone`)** (`amr_sched/rl/v19_hetero_graph.py`, `v19_hetero_constructor.py`, `causal_repair_cone.py`, `v10_selector.py`):
  - **5-Entity Heterogeneous Graph Attention Encoder & Pure-NumPy Promotion**: Encodes `Operation`, `Task`, `AMR`, `Bay`, and `Crew` nodes, directed precedence DAGs, bipartite kitting links, and corridor congestion/detour edge features, eagerly promoted to `< 1e-5` pure-NumPy inference matrices inside initialization (zero tick-0 cold-start spikes).
  - **Four-Stage Autoregressive Semantic Constructor (`BC -> KL-Anchored SMDP-PPO`)**: Sequentially decodes *Task Prioritization*, *Resource Assignment* (bay & crew), *Fleet Dispatch* (dynamic load + soft-commitment retention bias), and *Queue Sequencing*.
  - **Online Contextual Selection, Guarded Dual-Arm Probe, Stage-2 `CausalRepairCone` & Oscillation Arbiter**: Combines `Contextual LinUCB` ($\phi(s_t) \in [0, 1]^6$), fleet-guarded dual-arm probing ($|\mathcal{R}| \ge 4$), $J_A$-aligned candidate ranking, bounded Stage-2 Regret-2 `CausalRepairCone` over `FastJAWKittingSurrogate`, active-loading (`_load_end`) + edge safety guards, and symmetric `p4_choke_robust` surgical oscillation wakeups.
- **Incremental Optimization Engine (`CVCR-IR`)** (`amr_sched/cvcr/evaluator.py`, `crates/`):
  - **Sequence-Differenced Incremental Reservation Cache (`CVCR-IR (Seq-Diff)`) + Tightened Footprint-Aware SLA Guard**: Computes exact per-AMR effective task sequences, reuses unchanged AMRs' routes and Rust half-open interval reservations in-place (`median |ΔJ| = 0.00`), re-routes only $\Delta\mathcal{R}$ via native Rust `SpaceTime A* / SIPP` (`41 us` single-agent search), and bounds pooled wall-clock tail latency strictly below `23 ms` (`18.55 ms / 22.62 ms` Pooled P99 `< 35 ms`).
- **Native Rust E2E Kernel, Robust Capsule Shield & Execution-Aware STP (`v1.3.0–v1.4.0`)** (`crates/`, `amr_sched/planners/rust_spacetime_astar.py`, `experiments/{track_r_cpsa, cps_astar_routing, res_guided_avoidance, stp_research_reset}/`):
  - Zero-copy native Rust `SlotCongestionField`, `plan_cps_indexed`, lazy `RustBatchSippPlanner`, and zero-shadow `RustReservationTable.fast()` delivering **`11.71×–14.40×`** end-to-end speedup across `80–200`-AGV MovingAI `Warehouse_40x40` benchmarks (`26/26` Rust kernel tests passing).
  - Converged hierarchical coordination stack unifying EJOR 2026 dispatch (`Wu et al., EJOR 2026`), cleanroom `CPS-A*` congestion-aware routing (`Guo & Huang`, [`pengguo-hub/CPSAStar-TRC`](https://github.com/pengguo-hub/CPSAStar-TRC)), DFL blocker selection + overload-aware tuning, execution-delay local repair (`EALR v2/v3`), two-disc uncertainty-aware capsule safety (`R1_Robust`, `100.0%` ZeroCol across `36` disjoint blind episodes), and multi-tick edge-span corrected `CPS-A*` (`U1a_EdgeSpan_Fix`).
- **Execution Layer (*Atomic Transaction Commit*) & Dual-Layer Digital Twin** (`amr_sched/cvcr/`, `tools/`, `RAWSimO.Live/`):
  - Enforces epoch freshness, irrevocable `LOCKED` commitment protection, `_load_end` / valid-edge checks, and atomic fallback-to-`KEEP` across discrete simulation (`AMRReschedulingEnv`), ROS 2 / Nav2 / Gazebo physical closed-loop execution (`PhysicalCVCRCoordinator` + `PDG 2.0` liveness), and the C# WPF digital twin.
  - **Unified Policy Registry (24 Registered Policies)** (`amr_sched/baselines/policy_registry.py`): Includes `core_sched` / `CoRe-Sched` (`V19_ADAPTIVE_RL`), `V20_NAJS`, `V20_1_NAJS` (`PDG 2.0`), `AHG-RL` ablations, internal CVCR optimizers (`V18_EA_JOINT`, `rule_v10`), architecture-faithful domain-adapted 2025–2026 direct joint production–AGV competitors (`IJPR25_DFJSPT_HMARL`, `ASC26_MA_TRANS`, `FGCS26_REN_GNN`, `SPGA_AMR`), static offline references (`COR26-Ref` CP-SAT master relaxation + path verification solver, `EJOR26-Erdogan`), and broader 2025–2026 baselines (`IJCAI26_PGMPO`, `EJOR26_Wu_GRU_Attn`, `TII26_E2G_MoE`, `ICRA26_Kitting_HGNN`, `RAL26_DetourDelta`, `JAIR26_DyMAB`, `MIT26_TAPFPC_LNS`, `BALANS_IJCAI25`).

---

## Verified Benchmark Highlights (`v1.2.1` NCSR Optimization & Dual-Cohort Blind Closure)

### 1. Dual 120-Episode Holdouts (`240` Blind Episodes / `2,880` Audited Runs, `100%` Audit Pass)

Executed sequentially in uncontended single-process wall-clock runs under strict seed quarantine across **Holdout Cohort A** ([`results/g3_fresh_blind_holdout.csv`](results/g3_fresh_blind_holdout.csv), seeds `[27101..27302]` & `[2901..2906]`) and **Single-Shot Blind Holdout Cohort B** ([`results/g5_pristine_blind_holdout.csv`](results/g5_pristine_blind_holdout.csv), untouched seeds `[37101..37302]` & `[3901..3906]`, executed once post-freeze). Methods marked `*` breach the `35 ms` Pooled P95/P99 real-time SLA.

| Method | **240-Ep Comp (`%`)** ↑ | Cohort A Mean $J_A$ ↓ | Cohort A P95 / P99 (ms) | Cohort B Mean / **Med $J_A$** ↓ | Cohort B P95 / P99 (ms) | **240-Ep Paired $\Delta J_A$ (`p`, `W/T/L`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CoRe-Sched` (`core_sched`)** | **238/240 (99.2% #1)** | **5,177.9 (#1)** | **15.52 / 18.55** | 6,631.5 / **827.5 (#1)** | **19.45 / 22.62** | Reference (`5,904.7` Mean, `1,575.7` Med) |
| `V18-EA`* | 234/240 (97.5%) | 5,654.1 | 27.53 / *79.38* | **5,387.0** / 991.7 | 25.49 / *77.74* | -6.96% (`p = 0.0666`, `99/0/141`) |
| `IJCAI26-PGMPO` | 237/240 (98.8%) | 6,016.5 | 23.13 / *38.81* | 5,532.7 / 843.5 | 22.97 / 34.49 | -2.25% (`p = 0.901`, `116/0/124`) |
| `Rule-V10`* | 234/240 (97.5%) | 6,674.2 | *37.16* / *52.28* | 6,175.6 / 1,455.2 | *35.90* / *51.30* | +8.10% (`p = 0.092`, `127/0/113`) |
| `AHG-RL (Full)` | 235/240 (97.9%) | 6,019.2 | 32.84 / *52.34* | 6,942.6 / 3,610.4 | 33.14 / *42.61* | **+8.89%** (**`p = 0.00135`**, `144/0/96`) |
| `AHG-RL (Staged)`* | 234/240 (97.5%) | 6,469.2 | 30.31 / *54.26* | 6,821.4 / 3,613.7 | *35.59* / *78.51* | **+11.15%** (**`p = 7.25e-5`**, `151/0/89`) |
| `FGCS26-Ren-GNN` | 235/240 (97.9%) | 7,420.4 | 12.68 / 16.05 | 6,240.2 / 3,695.2 | 13.56 / 16.81 | **+13.55%** (**`p = 1.62e-4`**, `148/0/92`) |
| `ASC26-MA-Trans` | 231/240 (96.2%) | 7,776.6 | **12.22 / 16.82** | 6,580.0 / 3,811.9 | **12.01 / 15.30** | **+17.74%** (**`p = 7.35e-4`**, `142/0/98`) |
| `RAL26-DetourDelta` | 234/240 (97.5%) | 6,998.2 | 22.10 / *57.42* | 7,486.5 / 3,683.4 | 27.37 / *61.78* | **+18.47%** (**`p = 1.32e-10`**, `160/0/80`) |
| `COR26-SPGA`* | 234/240 (97.5%) | 7,533.0 | *62.55* / *75.39* | 7,240.1 / 2,920.4 | *62.02* / *73.66* | **+20.06%** (**`p = 4.09e-7`**, `164/0/76`) |
| `EJOR26-Wu-GRU` | 228/240 (95.0%) | 8,857.7 | 33.04 / *52.49* | 7,718.7 / 1,057.2 | 33.88 / *50.04* | **+28.76%** (**`p = 4.37e-4`**, `140/0/100`) |
| `IJPR25-DFJSP-T` | 228/240 (95.0%) | 7,817.4 | 14.85 / 18.05 | 9,253.9 / 3,760.6 | 14.70 / 18.35 | **+30.82%** (**`p = 8.47e-14`**, `166/0/74`) |

- **Highest Completion & Strict Sub-`23 ms` Pooled P99 Across Both Cohorts**: `CoRe-Sched` leads all 12 methods in combined completion (**`238/240 = 99.2%`**), sweeps **Cohort A Rank #1 (`5,177.9`, `120/120`, `15.52 / 18.55 ms` P95/P99)**, leads **Cohort B Median $J_A$ (`827.5` #1)** at **`19.45 / 22.62 ms` P95/P99**, and defeats **8 of 11 comparators cleanly ($p \le 0.00135$)** across the 240 combined episodes (with full disclosure of the two Cohort B unfinished tail outliers `topo4_offset_dual_lane_window_bottleneck_37302 = 78,259.95` and `track2_kitting_windows_3906 = 117,637.65` lifting Cohort B Mean $J_A$ to `6,631.5`).
- **72-Episode Single-Factor Ablations & `2 × 2` Trigger × Solver Decoupling** ([`results/g4_val72_repair_ablation.csv`](results/g4_val72_repair_ablation.csv)): Disabling `CausalRepairCone` (`+14.32%`, $p = 4.30 \times 10^{-5}$), `J_A`-aligned scoring (`+16.21%`, $p = 3.02 \times 10^{-4}$), `AHG-RL` priors (`+28.60%`), or ZeroChurn hysteresis (`+4.60%`) consistently degrades cost, while `RuleV10 + CoRe-Sched` (`4,190.3`, `72/72`) outperforms `V18-EA` (`4,252.9`), `RuleV10 + V18-EA` (`4,359.9`), and `V18Ctrl + CoRe-Sched` (`4,864.7`, `71/72`).
- **Pure Zero-Fault Static Common-Subproblem Reference (54 Instances, `0 / 270` Violations)** ([`results/g1_static_reference_subproblem.csv`](results/g1_static_reference_subproblem.csv)): `COR26-Ref` (offline CP-SAT master relaxation + conflict-free path verification reference solver, `LB_phys = 68.2 / 90.2 / 438.7`, `Sim-UB = 412.4 / 346.8 / 793.0`, `0` claimed exact optima) and `EJOR26-Erdogan` (`ILS-LP`, `352.1 / 664.8 / 664.5`) satisfy `LB_phys <= H1_sim` across all 270 runs (`CoRe-Sched` static Pooled P95/P99 = `13.61 / 20.47 ms`).
- **`CVCR-IR (Seq-Diff)` Cache Synergy, Physical Reconciliation & Scaling**: Restores isolated neural candidate hit rate from **`0.00%`** (`CVCR-IR Base`) to **`73.68%`** (`1.48×` speedup, `median |ΔJ| = 0.00`; `86.8%–94.7%` closed-loop hits); 5-seed ROS 2 / Nav2 / Gazebo physical reconciliation achieves Track 1 $\rho = 0.7857$ (`7.18 ms` physical P95); 10-AMR scaling sustains **`23.57 ms < 35 ms`** P95 (`2.21×` faster than full rebuild).

<p align="center">
  <img src="results/figures/fig7_dual_cohort_and_ablation.png" alt="Dual-Cohort 240-Episode Consistency, VAL-72 Ablations, and 2x2 Trigger x Solver Decoupling" width="96%" />
</p>

<p align="center">
  <img src="results/figures/fig3_main_benchmark.png" alt="CoRe-Sched Cohort A Main Benchmark and Latency-Quality Pareto Frontier" width="96%" />
</p>

<p align="center">
  <img src="results/figures/fig6_pure_h1_and_lbbd.png" alt="Dual-Cohort Pure Physical Efficiency H1 Attribution and Static Simulator Reference Subproblem" width="96%" />
</p>

<p align="center">
  <img src="results/figures/gazebo_overview.png" alt="1:25 Gazebo Wheelset Workshop Digital Twin" width="48%" />
  <img src="results/figures/layout.png" alt="Workshop Topology & Station Layout" width="48%" />
</p>

### 2. Execution-Aware Physical Coordination & Counterfactual Falsification (`v1.4.0`)

- **Uncertainty-Aware Two-Disc Capsule Shield (`R1_Robust`)** ([`results/ramp_robust/robust_ramp_report.md`](results/ramp_robust/robust_ramp_report.md)): Fixing the single-disc proximity underestimate (`0.10–0.20 m`) and horizon Gaussian bounds ($\epsilon(t) = 1.96\sqrt{\sigma_p^2 + (\min(t, 1.0)\,\sigma_v)^2}$) eliminates all `12,247` blind-set peer collision ticks (`58.3% → 100.0%` ZeroCol across `36` episodes `7001–7006`), where deterministic **`R1_Robust`** (`4.69` completed) outperforms frozen (`-5.33%`) and retrained (`-8.86%`) `RAMP-PPO`.
- **Counterfactual Decision Headroom Closure** ([`results/decision_headroom/decision_headroom_report.md`](results/decision_headroom/decision_headroom_report.md)): Across `N = 60` `clone_snapshot()` rollouts (`<= +0.85%` fleet `clear_15s`) and `N = 36` exhaustive passing-order permutations (`+0.43%` over `R1_Default_Order` = `5.667` completed, `100.0%` ZeroCol), neither primitive-level nor local passing-order RL exceeds deterministic safety-bounded rules.
- **Execution-Aware Upper-Planning Audit (`U1a_EdgeSpan_Fix`)** ([`results/stp_research_reset/upper_level_headroom_report.md`](results/stp_research_reset/upper_level_headroom_report.md)): Quantifies systematic `CPS-A*` early-arrival drift (`-17.25 s`, `94.4%` early across `N = 252` DEV legs); fixing multi-tick edge-span reservations (`U1a_EdgeSpan_Fix`) lifts DEV ZeroCol from `17/18` to **`18/18 (100.0%)`** and completion **`+6.00%`** (`5.889` vs `5.556`, `0.376 ms/plan`), while finite-pool counterfactual selection (`U4`) reaches `-10.53%` (`17` vs `19` tasks).

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

# 3. Run workshop simulation, NCSR ablations, Dual-Cohort blind suites & 7 figures
python -m amr_sched --tasks 50 --agvs 6 --layout workshop --planner rust_spacetime_astar
python -B experiments/run_post_v120_audit_and_ablation.py
python -B experiments/run_g5_pristine_blind_holdout.py
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
│   ├── rl/                     # AHG-RL 5-entity Hetero-GNN constructor, CausalRepairCone, Contextual LinUCB & SMDP env
│   └── viz/                    # Workshop layout, Gantt, corridor heatmaps, trajectories, animations
├── crates/                     # Native Rust Cargo workspace (reservation_core & spacetime_planner)
├── RAWSimO.Live/               # C# (.NET 6) live TCP JSONL protocol consumer & topology tracker
├── RAWSimO.Visualization/      # C# WPF 2D visualizer & live monitoring UI
├── tools/                      # Benchmarks, ROS 2 / Nav2 / Gazebo bridges, 1:25 wheelset scene
├── experiments/                # Training, NCSR ablations, Dual-Cohort blind evaluation, reconciliation & plotting
├── results/                    # Verified CSV datasets, MANIFEST.json, 7 publication figures (`fig1`–`fig7`)
├── docs/architecture/          # System architecture specification and ADRs
├── web/                        # FastAPI WebSocket monitoring & replay console
└── Tests/                      # Python pytest suites + C# xUnit test suites
```

---

## Documentation & License

- **Architecture & Protocols**: [Architecture Spec](docs/architecture/ARCHITECTURE.md), [ADR 001 (Rust FFI)](docs/architecture/decisions/001_cdylib_ctypes_architecture.md), [ADR 002 (Transaction & Auditor)](docs/architecture/decisions/002_transaction_auditor_data_flow.md), [PROTOCOLS.md](PROTOCOLS.md), [CHANGELOG.md](CHANGELOG.md)
- **License**: Original code is licensed under **GPL-3.0**. Upstream RAWSim-O components retain their original licenses. See [LICENSE](LICENSE) and [ThirdPartyLicenses.txt](ThirdPartyLicenses.txt).
