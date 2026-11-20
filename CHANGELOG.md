# Changelog

All notable changes to the **AMR-Scheduler** platform are documented in this file.

## [0.3.0] - 2026-11-20

### Added
- **V10–V17 Hierarchical Execution-Aware Scheduling Engine (`amr_sched/cvcr/`)**:
  - `V17WindowProtectionPolicy` (`v17_window_protection.py`): causal portfolio mode routing (`V15_SOFT`, `V7_CALM`, `V10_GUIDED`), 50-D pairwise window payoff Set-Attention ranker, Keep-Zero Conservative Regret Restraint, and legal bridge-order control (`pre_pickup`, `loaded_at_staging`, `in_edge` guard).
  - `V17_E0` dual-order reconciliation runner (`v17_e0_audit.py`) decoupled from scratch artifacts.
  - P4 chronic belief updating & multi-hop choke token arbitration (`p4_belief_choke.py`), Event-SMDP execution feedback (`v17_event_smdp.py`), and physical execution bridge (`physical_bridge.py`).
  - Packaged pure-NumPy `.npz` inference checkpoints under `amr_sched/checkpoints/` with unified `resolve_default_checkpoint()` resolution (`checkpoints.py`).
- **Rust Zero-Copy FFI & Batch Parallel SIPP (`crates/`)**:
  - Indexed reservation table FFI and batch parallel SIPP queries (`crates/reservation_core/src/ffi.rs`, `crates/spacetime_planner/src/sipp.rs`).
- **Gymnasium RL Environment & 2026 Top-Venue Baselines (`amr_sched/rl/`, `amr_sched/baselines/`)**:
  - Multi-version masked RL environments (`AMRReschedulingEnv`), CUDA backend helpers, and V10/V12/V13 adaptive controllers.
  - Causal adapters for `Wu EJOR'26`, `Choi IJCAI'26`, `Ma TII'26`, `Sow JAIR'26`, `Parimi MIT'26`, and `Zhang ICRA'26` (with ACID picked-kit bay lock) alongside 2024–2025 literature baselines.
- **ROS 2 / Nav2 / AgileX SCOUT Mini Digital Twin Bridge (`tools/live_bridge/`, `tools/ros2/`)**:
  - Costmap corridor projector, RViz CVCR overlay, Nav2 SCOUT configuration (`tools/ros2/nav2_scout_config/`), and CAN 2.0B / `vcan0` Gazebo bridge.
- **V17 Empirical Benchmarks & Regression Suites**:
  - V17 E2 attribution, E4 6-track grand benchmark, and HIL detour figures (`results/figures/v17_*.png`).
  - 22 new Python test suites covering V10–V17, P1–P4.5, zero-copy Rust FFI, Nav2/vCAN, and 2026 baselines.

### Changed
- Updated `.github/workflows/ci.yml` and `pyproject.toml` (`[rl]` extra and `package-data` for `.npz` checkpoints) so clean GitHub checkouts execute full Python, RL, and Rust tests cleanly.
- Decoupled all tests in `Tests/` from `_scratch/` and replaced personal host/IP literals with `AMR_BRIDGE_HOST` / `AMR_GAZEBO_SSH` environment defaults.
- Updated `README.md`, `README_zh.md`, `docs/architecture/ARCHITECTURE.md`, `amr_sched/README.md`, and `CONTRIBUTING.md` to reflect the unified v0.3.0 architecture.

## [0.2.0] - 2026-11-16

### Added
- **Full LG-CVCR Scientific Closed-Loop Integration (G0–G6)**:
  - Factorial incremental evaluator (`CVCREvaluator`) supporting `A1`, `B1`, and `B2` modes.
  - Exact 13-feature cross-domain extractor and frozen CIE MLP weights (`frozen_weights.json`).
  - Native software fleet admission gate (`Simulator.set_admitted_agvs`), `NativeProductionTracker`, and `IndependentAuditor`.
- **Rust FFI Panic Guard (`ffi_guard!`)** and PEP 517/621 `pyproject.toml`.

## [0.1.0] - 2026-11-10
- Initial multi-AMR scheduling core, Spacetime A* planner, RAWSim-O 2D visualizer, and 1:25 Gazebo simulation baseline.
