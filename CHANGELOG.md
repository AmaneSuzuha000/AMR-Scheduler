# Changelog

All notable changes to the **AMR-Scheduler** platform are documented in this file.

## [1.2.1] - 2026-11-24

### Added & Verified (NCSR Optimization, VAL-72 Ablations & Dual 240-Ep Holdouts)
- **In-Place `CoRe-Sched` NCSR Fixes (`v19_hetero_constructor.py`, `v12_proactive_guard.py`, `v10_selector.py`)**:
  - Eagerly pre-populated pure-NumPy actors (`_np_full_net`, `_np_staged_net`) inside `_ensure_nets()` to eliminate tick-0 cold-start spikes (`< 1e-5` parity).
  - Enabled Stage-2 Regret-2 `CausalRepairCone` (`FastJAWKittingSurrogate`) with execution-consistent commits (`enable_exec_consistent_commit=True`), valid navigation-edge filtering (`network.has_edge`), active-loading shielding (`_load_end > step`), and symmetric `p4_choke_robust` corridor-oscillation wakeup arbitration (8-tick cooldown).
  - Tightened anytime SLA caps (`17.5 / 29.0 ms` under `35 ms`), locking pooled wall-clock P99 to `18.55 ms` (Cohort A) and `22.62 ms` (Blind Cohort B).
- **Quarantined `VAL-72` Ablations & `2 × 2` Trigger × Solver Decoupling (`experiments/run_post_v120_audit_and_ablation.py`, `results/g4_val72_repair_ablation.csv`, `Tests/test_post_v120_ncsr_audit.py`)**:
  - Quantified single-factor contributions (`w/o Cone` `+14.32%` $p=4.30\times 10^{-5}$; `Legacy-Scorer` `+16.21%` $p=3.02\times 10^{-4}$; `w/o AHG-RL` `+28.60%`; `w/o ZeroChurn` `+4.60%`) and isolated trigger–solver co-adaptation (`RuleV10 + CoRe-Sched` `4,190.3` `72/72`).
- **Dual 120-Episode Holdouts & Gate G1 Re-Verification (`experiments/run_g5_pristine_blind_holdout.py`, `results/g5_pristine_blind_holdout.csv`, `fig7_dual_cohort_and_ablation.png`)**:
  - **Cohort A (`120` eps)**: `CoRe-Sched` ranks **#1 overall** (`5,177.9` Mean $J_A$, `120/120` 100%, `15.52 / 18.55 ms` P95/P99).
  - **Single-Shot Blind Cohort B (`120` untouched eps)**: **#1 Median $J_A$ (`827.5`)**, `118/120` (98.3%), `19.45 / 22.62 ms` P95/P99, `7/11` paired wins ($p \le 0.0132$), transparently reporting `topo4_37302` & `track2_3906` heavy-tail outliers (`6,631.5` Mean).
  - **Combined `240` Episodes (`2,880` runs)**: **#1 completion (`238/240 = 99.2%`)** and `p \le 0.00135` paired wins over `8 / 11` comparators, plus `0 / 270` Gate G1 static relaxation violations (`COR26-Ref` offline CP-SAT master relaxation + conflict-free path verification reference solver, `0` claimed exact optima) and **Fig. 7** (`fig1`–`fig7`).

## [1.2.0] - 2026-11-24

### Added & Verified (Three-Gate Scientific Closure)
- **CoRe-Sched In-Place Upgrades (`amr_sched/rl/v19_hetero_constructor.py`, `causal_repair_cone.py`)**:
  - `J_A`-aligned candidate scoring fixing legacy task-tardiness omission (`13.92%` DEV/VAL misselection eliminated).
  - Fleet-guarded counterfactual dual-arm probe (`|R| >= 4`) addressing `31.37%` single-arm LinUCB regret.
  - `CVCR-IR` footprint-aware anytime SLA guard enforcing strict `<= 35 ms` worst-tail real-time bounds (`32.43 ms` pooled P95, `34.91 ms` pooled P99 across 120 fresh blind episodes).
- **Gate G1 — Pure Zero-Fault Static Common-Subproblem Reference (`lbbd_cp_cor26.py`, `erdogan_ejor26.py`, `results/g1_static_reference_subproblem.csv`)**:
  - Aligned `COR26-Ref` Phase-1 CP-SAT physical relaxation master to exact `Simulator` dwell/travel physics (`0 / 270` lower-bound violations across 54 pure-static instances × 5 solvers) and downgraded post-Phase-1 nogood cuts explicitly to heuristic search diversification.
  - Added `EJOR26-Erdogan` (`erdogan_ejor26.py`, Erdogan et al., *EJOR* 2026) ILS + exact timing matheuristic verified end-to-end inside the identical `Simulator`.
- **Gate G2 — 2026 Dynamic FJSP–AGV Neural Competitors (`ma_trans_asc26.py`, `fgcs26_ren_gnn.py`, `train_2026_rl_competitors.py`)**:
  - Added architecture-faithful, domain-adapted reimplementations of `ASC26_MA_TRANS` (Shen, Zhang & Jin, *ASC* 193, 2026; `pt-npz` diff `7.63e-6`) and `FGCS26_REN_GNN` (Ren et al., *FGCS* 182, 2026; `pt-npz` diff `2.86e-6`), trained under exact quarantined 264-IL + 288-PPO parity (`22` registered policies total).
- **Gate G3 — Single-Session Fresh 120-Episode Frozen Blind Holdout (`run_g3_fresh_frozen_holdout.py`, `results/g3_fresh_blind_holdout.csv`, `Tests/test_three_gates_audit.py`)**:
  - Uncontended single-session 12-method × 120-episode (`1,440` audited runs) fresh blind evaluation reporting both **Pooled Wall-Clock P95/P99** and **Mean Episode P95**, updated `fig1`/`fig3`/`fig6`, and SHA-256 verified public CSV ledgers.

## [1.1.0] - 2026-11-24

### Added
- **Direct Production–AMR Baselines & Offline Exact Reference Solver (`amr_sched/baselines/`, `experiments/`, `Tests/`)**:
  - `IJPR25_DFJSPT_HMARL` (`dfjspt_hmarl_ijpr25.py`, `train_dfjspt_hmarl_ijpr25.py`, `ijpr25_dfjspt_hmarl.{pt,npz}`): Paper-faithful 3-level hierarchical MARL reproduction (Wang et al., *IJPR* 2025) trained under exact quarantined split and 288-episode PPO+IL parity with pure-NumPy inference (`7.63e-6` parity).
  - `COR26_LBBD_CP` (`lbbd_cp_cor26.py`, `run_lbbd_exact_reference.py`): Offline exact Logic-Based Benders Decomposition reference solver (van Os & Basten, *COR* 2026) coupling OR-Tools `CP-SAT` with Rust conflict-free spacetime verification across 54 static/known instances (`6/24` proven global optima on window bottlenecks).
  - Unified 120-episode baseline closure & pure-efficiency $H_1$ audit runner (`run_baseline_closure_suite.py`), unit test suite (`Tests/test_baseline_closure.py`), updated `fig3_main_benchmark.png`, and new `fig6_pure_h1_and_lbbd.png`.

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
