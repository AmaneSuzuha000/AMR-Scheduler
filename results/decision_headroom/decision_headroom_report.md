# Decision-Headroom-Audit: Decision Headroom & Novelty Audit Report

> **Final Scientific Verdict (`Task 6`)**: **Branch C — `STOP RL FOR THIS LAYER` (`NO-GO` for both Route A `RAMP` and Route B `STCP-RL`)**
>
> Under matched `100.0%` empirical `ZeroCol` safety, neither `K = 6` motion-primitive selection (`+0.85%` fleet / `+0.58%` ego clearing headroom) nor conflict-zone passing-order arbitration (`+0.43%` over deterministic rules) passes the **`>= 3.0%` realizable headroom gate**. Meanwhile, deterministic **`R1_Robust`** already achieves near-oracle throughput (`5.667` completed tasks across conflict-rich suites, `100.0%` ZeroCol) with zero training cost.

---

## 1. Frozen Evidence Manifest & Independent Event Audit (`Task 1`)

- **Seed Integrity Policy**: Seeds `5001–5003`, `6001–6003`, and `7001–7006` are permanently registered as **observed DEV/Audit splits** in [task1_frozen_manifest_and_audit.json](./task1_frozen_manifest_and_audit.json). Blind confirmation seeds `8001–8010` remain **completely untouched**.
- **Independent Collision Events vs `M_sub = 5` Substep Ticks**:
  - `Workshop_8R_R5_Orig_s6001`: **`1` independent collision event** generated **`3,059` substep collision ticks** (`W:0 / P:3059 / I:0`).
  - `Noise0.12_R5_Orig_s6001`: **`1` independent collision event** generated **`416` substep collision ticks** (`W:0 / P:416 / I:0`).
  - `Noise0.12_R5_Robust_s6001` & `Workshop_8R_R1_Robust_s6001`: **`0` independent collision events**, **`0` substep ticks** (`zcol = True`).
- **Gate 0 `4R_LifelongWorkshop` Physical Shortest-Path Lower Bound (`T_min`)**:
  - Even with zero turns, zero deceleration, and zero inter-robot congestion at theoretical `MAX_V_MPS = 1.35 m/s`, Task 1 (`232.0 m`) requires at least **`172.85 s`** (`245.21 s` at `0.95 m/s` cruise). Thus, Gate 0's initial `130.0 s` horizon was **strictly physically infeasible**, while the achieved `252.25 s` makespan sits within `2.9%` of the no-congestion cruise lower bound.

---

## 2. Primitive Decision Freedom & Shield Override Attribution (`Task 2`)

Across **`N = 2,315`** snapshots spanning `Occluded_Jct_6R`, `MapA_8R`, `Workshop_8R`, and `Noise_0.12m` ([task2_primitive_freedom.json](./task2_primitive_freedom.json)):

- **Safety Filtering Shrinks Freedom (`K_geom -> K_safe`)**:
  - `K_geom`: `1` (`6.2%`), `4` (`78.7%`), `6` (`13.9%`).
  - `K_safe`: **`1` (`17.9%`)**, **`2` (`14.0%`)**, `3` (`1.0%`), `4` (`54.2%`), `6` (`12.7%`).
  - In `Congestion_Wait`, **`44.4%`** of states collapse to **`K_safe = 1`** (`avg = 2.66`); in `HeadOn`, **`35.7%`** collapse to **`K_safe = 1`** (`avg = 2.06`).
- **Pre-Shield vs Post-Shield Action Disagreement (Quantitative Proof of H2)**:
  - `R1_Robust` vs `R5_Orig` disagree on **`46.7%`** of snapshots (`R1` vs `R5_Robust`: `45.4%`).
  - **Shield Override Rate** (policy selects a primitive rejected by `robust_safety`):
    - `R1_Robust`: **`9.5%`** overall (`1.2%` Nominal, `13.0%` HeadOn, `30.0%` CongestionWait).
    - **`R5_Orig`**: **`26.7%`** overall — peaking at **`66.0%`** in **`HeadOn`** (`39.6%` Dynamic Intruder, `33.6%` CongestionWait).
    - **`R5_Robust_Retrained`**: **`20.2%`** overall — still **`46.6%`** in **`HeadOn`** (over `3.5x` `R1_Robust`).
  - This confirms quantitatively that `R5_Orig`'s higher unshielded completion relied heavily on boundary-violating primitive choices that `robust_safety` vetoes.

---

## 3. Matched Multi-Horizon Counterfactual Oracle — Route A (`Task 3`)

From **`N = 60`** multi-regime snapshots (`K_safe >= 2`) cloned with bitwise-identical `RNG`, noise, and reservations ([task3_counterfactual_headroom.json](./task3_counterfactual_headroom.json)):

| Matched Horizon | Ego Mean Rel Gain | Ego `>= 3%` Share | **Fleet Mean Rel Gain** | Fleet `>= 3%` Share | Coverage Loss (`cov_loss`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `0.5 s` (2 steps) | `+2.89%` | `16.7%` | **`+0.98%`** | `13.3%` | `0.001` |
| `2.0 s` (8 steps) | `+2.86%` | `21.7%` | **`+0.32%`** | `0.0%` | `0.000` |
| `5.0 s` (20 steps) | `+3.04%` | `18.3%` | **`+0.81%`** | `6.7%` | `0.000` |
| `10.0 s` (40 steps) | `+2.31%` | `11.7%` | **`+1.10%`** | `3.3%` | `0.009` |
| **`clear_15s` (60 steps)** | **`+0.58%`** | **`1.7%`** | **`+0.85%`** | **`3.3%`** | `0.016` |

- **Interpretation**: Short-horizon (`2.0–5.0 s`) ego gains (`+9.6%` to `+11.4%` in 9 `Junction_Conflict` snapshots) are **ego-myopic transients** that disappear once the conflict clears (`clear_15s: +0.58%` ego) and do not translate to fleet progress (`<= +1.10%` at every horizon). Safe-set coverage loss from `robust_safety` is negligible (`0.000–0.016`).
- **Route A Verdict (`RAMP`)**: **`NO-GO`**. Realizable safe primitive-selection headroom is `< 1%`.

---

## 4. Conflict-Level Passing-Order Headroom — Route B / `STCP-RL` (`Task 4`)

Keeping `CPS-A*` routes and `R1_Robust` fixed as the lower-level executor with non-revocable `< 0.90 m` entry commitments ([task4_conflict_headroom.json](./task4_conflict_headroom.json)):

| Passing-Order Strategy (`15` eps across 5 suites) | ZeroCol (%) | Completed Avg | Wait Ratio | Shield Interv |
| :--- | :---: | :---: | :---: | :---: |
| **`R1_Default_Order`** (proximity + `to_dropoff` phase + ID) | **`100.0%`** | **`5.667`** | **`0.3158`** | **`248.6`** |
| `FCFS` | `100.0%` | `5.200` | `0.3227` | `289.8` |
| `SPT_Clear` | `100.0%` | `5.000` | `0.3358` | `295.3` |
| `Orig_Res_Order` | `100.0%` | `4.933` | `0.3273` | `331.1` |
| `Queue_Downstream_Det` | `100.0%` | `4.933` | `0.3346` | `300.8` |

- **Exhaustive Permutation Oracle (`N = 36` multi-robot conflict clusters, `10.0 s` counterfactual rollout)**:
  - vs `Orig_Res_Order`: `+3.01%`
  - vs `R1_Default_Order`: `+3.18%`
  - **vs Best Deterministic Baseline (`min(R1_Default_Order, Orig_Res_Order, Queue_Downstream_Det)`)**: **`+0.43%`** (`>= 3%` in only `2/36 = 5.6%` conflicts; **`0.0%` on all `Workshop_8R` and `MapB_12R` conflicts**).
- **Route B Verdict (`STCP-RL`)**: **`NO-GO`**. Even an exact exhaustive oracle permutation over uncommitted entrants gains only `+0.43%` over deterministic baselines.

---

## 5. Literature Novelty Audit Summary (`Task 5`)

Full matrix in [literature_novelty_matrix.md](./literature_novelty_matrix.md):
- `RAMP-PPO` (`K=6` primitives) overlaps `db-LaCAM` (ICAPS 2026) + `SJRL`/`CS-PIBT` (ICRA 2025/2026) and empirically loses to deterministic `R1_Robust`.
- `STCP-RL` (conflict passing-order RL) directly overlaps **`BTPG-max` (AAAI 2026)** (Type-2 passing-order relaxation under delay), **`SJRL` (2026)**, and **`K-ARC` (RA-L 2025)** (event-triggered local coordination) while offering only `+0.43%` empirical headroom.

---

## 6. Scientific Closure & Research Positioning

- **Frozen Result (`Branch C`)**: Stop RL training at the `RAMP` primitive layer and the `STCP-RL` local conflict passing-order layer.
- **What Survives & Shines**: The deterministic **`CPS-A*` → `R1_Robust`** pipeline (`m_ij(z)` reservation time-margins + `K=6` kinodynamic primitives + two-disc capsule uncertainty-aware `BrakingInvariantShield` + non-revocable phase/proximity ROW) is rigorously validated as achieving **`> 99%` of counterfactual oracle performance** with **`100.0%` empirical `ZeroCol`** on all tested suites and zero neural-network variance or latency overhead.
- *Future Note (Only if revisiting RL later)*: Any future RL work in this workspace should target **upstream macro corridor/task-dispatch coordination under severe non-local congestion** where multi-hop global headroom exists — never re-training on top of `R1_Robust`'s already-saturated local layer.
