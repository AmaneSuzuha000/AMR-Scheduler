# Changelog

All notable changes to the **AMR-Scheduler** platform are documented in this file.

## [0.2.0] - 2026-11-16

### Added
- **Full LG-CVCR Scientific Closed-Loop Integration (G0–G6)**:
  - Factorial incremental evaluator (`CVCREvaluator`) supporting `A1` (full replan), `B1` (path preserve), and `B2` (persistent incremental reservation cache with in-place release and restore).
  - Exact 13-feature cross-domain feature extractor and frozen CIE MLP models (`Outcome-MLP`, `Keep/Release-MLP`).
  - Budget-matched ALNS baseline and proposed `IDEA_CORRIDOR_CVCR_JOINT` policy.
  - Native software fleet admission gate (`Simulator.set_admitted_agvs`) preventing inactive robots from receiving dispatches while retaining physical node occupancy.
  - Step-by-step live production tracker (`NativeProductionTracker`) and trajectory invariant auditor (`IndependentAuditor`).
  - Held-out 62-node railway maintenance workshop test suite (`Tests/test_cvcr_p3_workshop.py`).
- **Rust FFI Panic Guard**:
  - Implemented `ffi_guard!` (`std::panic::catch_unwind`) across all exported C-ABI functions in `crates/reservation_core/src/ffi.rs` and `crates/spacetime_planner/src/ffi.rs`.
- **Python Packaging & Hygiene**:
  - Added PEP 517/621 `pyproject.toml` supporting `pip install -e .` with optional extras (`viz`, `web`, `dev`, `all`).
  - Added dynamic library path overrides (`AMR_RESERVATION_CORE_PATH`, `AMR_SPACETIME_PLANNER_PATH`).
  - Added dynamic GitHub Actions status badges and updated documentation.
  - Added comprehensive architecture specification (`docs/architecture/ARCHITECTURE.md`) and ADRs.

### Fixed
- **CI Disconnection**: Fixed `.github/workflows/ci.yml` so the Python job builds the Rust cdylib before running pytest, bridging the dual-language test gap.
- **Graceful Timing Guards**: Added fallback guards in `Tests/test_cvcr_*.py` when running in pure Python environments without compiled Rust planner.
- **Simulator Reset Wipeout**: Resolved issue in `run_system08_episode` where calling `sim.run()` cleared committed fleet admission decisions.
- **Production Precedence**: Corrected production operation tracking during simulated AMR breakdowns in `run_system07_episode`.

## [0.1.0] - 2026-11-10
- Initial multi-AMR scheduling core, Spacetime A* planner, RAWSim-O 2D visualizer, and 1:25 Gazebo simulation baseline.
