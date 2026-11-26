# F0: Reservation Semantics & Multi-Tick Corridor Blast-Radius Audit

## 1. Key Findings & Blast Radius

- **`arr_i - 1` occurrences in `amr_sched/`, `crates/`, `tools/`**: `0`
- **`expand_planned_to_step_positions` callers in `amr_sched/`, `crates/`, `tools/`**: `0`
- **Exact bug root location**: `experiments/track_r_cpsa/cpsa_pure_runner.py:77` (`dep_t = arr_i - 1`, documented explicitly for unit-step grid graphs).
- **`experiments/stp_research_reset/upper_planners.py:102` classification**: only used inside soft heuristic `_build_occupancy_counts()` (`gamma` slot-pressure bias), NOT inside hard `RustReservationTable` writes (`ExecutionAuditSim` uses `_reserve_exact_span`).
- **Why `amr_sched` Track B was immune**: `Simulator._reserve_hop` (`amr_sched/simulation.py:1191`), `Simulator._commit_move` (`1395`), `SpaceTimeAStarPlanner` (`amr_sched/planners/spacetime_astar.py:165`), `RustSpaceTimeAStarPlanner` + `crates/reservation_core` (`Interval(depart, arrive)` + `collect_undirected_edge_conflicts_into`), and `CVCREvaluator._space_time_route` (`dep_t = arr_t - self._edge_steps(cur_n, nxt_n)`) always reserve full undirected multi-tick edge intervals `[depart, arrive)`.

### All `expand_planned_to_step_positions` Callers in Repo

- `experiments/foundation_audit/f0_reservation_semantics_audit.py`
- `experiments/res_guided_avoidance/metric_world.py`
- `experiments/stp_research_reset/audit_env.py`
- `experiments/track_r_cpsa/cpsa_pure_runner.py`
- `experiments/track_r_cpsa/execution_delay_sim.py`
- `experiments/track_r_cpsa/local_reservation_repair.py`
- `experiments/track_r_cpsa/route_a_lpr_ablation.py`
- `experiments/track_r_cpsa/rust_sipp_accel.py`

## 2. Workshop Scene (`rail_freight_workshop_v1`) Geometry & Capacity Audit

- Nodes: `62`, Directed edges: `144`, Bidirectional corridors: `66`
- Corridors with length in `[20, 30.5]` m: `16`
- Nodes with `raw_capacity != point_capacity(1)`: `38`
- Edges with `raw_capacity != point_capacity(1)`: `0`

| Corridor | Length (m) | Speed Limit (m/s) | Discrete Steps (dt=1s, v=1) | Macro Steps (dt=2s, v=1) | Raw Cap |
|---|---:|---:|---:|---:|---:|
| `J_N_105 <-> J_S_105` | 30.00 | 1.00 | 30 | 15 | 1/1 |
| `J_N_125 <-> J_S_125` | 30.00 | 1.00 | 30 | 15 | 1/1 |
| `J_N_25 <-> J_S_25` | 30.00 | 1.00 | 30 | 15 | 1/1 |
| `J_N_45 <-> J_S_45` | 30.00 | 1.00 | 30 | 15 | 1/1 |
| `J_N_65 <-> J_S_65` | 30.00 | 1.00 | 30 | 15 | 1/1 |

## 3. Executable Multi-Tick Corridor Verification

Tested real corridor: `J_N_105 <-> J_S_105` (30.0 m, `30` ticks @ `dt=1.0s`, `15` ticks @ `macro_step_s=2.0s`).

- **Python `SpaceTimeAStarPlanner`**: `R1 arr(v)=32`, `R2 arr(u)=62`, stagger_ok=`True`
- **`RustSpaceTimeAStarPlanner`**: `available=True`, `R2 arr(u)=62`, stagger_ok=`True`
- **`Simulator` isolated corridor (`res_window=0`)**: completed=`2`, collisions=`0`, simultaneous_corridor_ticks=`0`
- **`Simulator` isolated corridor (`res_window=1`)**: completed=`2`, collisions=`0`, simultaneous_corridor_ticks=`0`
- **`Simulator` full Workshop (`point_execution_network`)**: completed=`2`, collisions=`0`, finished_step=`101`
- **`CVCREvaluator` opposing check**: mid_corridor_conflicts=`['R1']`, stagger_ok=`True`

### Reproduced Track A `expand_planned_to_step_positions` Blind Spot on Workshop Corridor

- Actual macro traversal: `15` ticks (`[0, 15)`)
- Buggy `pts` from `expand_planned_to_step_positions`: `[('J_N_105', 0), ('J_N_105', 1), ('J_N_105', 2), ('J_N_105', 3), ('J_N_105', 4), ('J_N_105', 5), ('J_N_105', 6), ('J_N_105', 7), ('J_N_105', 8), ('J_N_105', 9), ('J_N_105', 10), ('J_N_105', 11), ('J_N_105', 12), ('J_N_105', 13), ('J_N_105', 14), ('J_S_105', 15)]`
- **Blind-spot window dropped**: `14` / `15` ticks (`93.33%`)
- Opposing query mid-corridor conflict detected (buggy `expand`): `[]` -> `R2 arr(u)=15` (HEAD-ON ALLOWED AT t=0!)
- Opposing query mid-corridor conflict detected (fixed `_reserve_exact_span`): `['R1']` -> `R2 arr(u)=32` (STAGGERED SAFELY AFTER R1)
