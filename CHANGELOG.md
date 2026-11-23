# Changelog

All notable changes to the **AMR-Scheduler** platform are documented in this file.

## [1.0.0] - 2026-11-23

### Added
- **CoRe-Sched Dual-Engine Learning & Commitment-Preserving Incremental Optimization (`amr_sched/rl/`, `amr_sched/cvcr/`, `amr_sched/baselines/`)**:
  - `AHG-RL` Learning Engine: 5-entity heterogeneous graph attention encoder (`v19_hetero_graph.py`), 4-stage autoregressive semantic constructor (`v19_hetero_constructor.py`, `BC -> KL-anchored SMDP-PPO`), online `Contextual LinUCB` coupling selector, and exact (`< 1e-5`) PyTorch / pure-NumPy parity.
  - `CVCR-IR (Seq-Diff)` Incremental Optimization Engine (`evaluator.py`): Exact per-AMR effective task-sequence differencing (`seq_diff_b2=True`) restoring incremental reservation cache hits from `0.00%` to `73.68%` isolated (`86.8%–94.7%` closed-loop) at `median |ΔJ| = 0.00`, paired with $O(1)$ runtime telemetry.
  - Unified policy registry (`policy_registry.py`) registering `core_sched` / `CoRe-Sched` (`V19_ADAPTIVE_RL`), `AHG-RL` ablations, internal CVCR optimizers, and 2025–2026 literature baselines.
  - Five 300-DPI publication figures (`results/figures/fig1_core_sched_overview.png` through `fig5_dual_layer_and_scaling.png`).
- **Dual-Layer Reconciliation, 62-Node 4-Axis Scaling & 3D Warehouse Benchmark Reproduction (`experiments/`, `tools/`)**:
  - Discrete (Layer A) vs continuous physical ROS 2 / Nav2 / Gazebo (Layer B) reconciliation runner (`experiments/run_dual_layer_reconciliation.py`).
  - 62-node 4-axis scaling and real-time budget benchmark runner (`experiments/run_workshop_scaling_4axis.py`).
  - Independent reproduction of Wu et al. (*EJOR* 2026) 3D warehouse order-picking and AGV routing (`experiments/ejor26_warehouse_repro/`).
  - Full-scale 50-node metric wheelset maintenance workshop layout and ROS 2 / Nav2 closed-loop runner (`tools/wheelset_scene/workshop_v2_layout.py`, `run_wheelset_v2_full_chain.py`).

## [0.3.0] - 2026-11-20

### Added
- **Execution-Aware CVCR Scheduling Core & Literature Baselines (`amr_sched/cvcr/`, `amr_sched/baselines/`)**:
  - Causal snapshot capture (`snapshot.py`), incremental B2 reservation evaluator (`evaluator.py`), window-protected routing (`v17_window_protection.py`), frontier synthesis (`v18_frontier_synthesis.py`), chronic corridor belief & choke arbitration (`p4_belief_choke.py`), and physical ROS 2 bridge (`physical_bridge.py`).
  - Causal adapters for 2024–2026 literature baselines (`baselines_2026.py`, `spga_cor2026.py`, `top_venue_baselines.py`).
- **Native Rust Zero-Copy FFI & Batch Parallel SIPP (`crates/`)**:
  - Half-open integer interval reservation table (`crates/reservation_core`) and packed-`u64` Spacetime A* / batch parallel SIPP (`crates/spacetime_planner`).
- **ROS 2 / Nav2 / AgileX SCOUT Mini Digital Twin Bridge (`tools/live_bridge/`, `tools/ros2/`, `RAWSimO.Live/`)**:
  - Costmap corridor projector, RViz CVCR overlay, CAN 2.0B / `vcan0` bridge, and C# `.NET 6` live TCP JSONL digital twin.
