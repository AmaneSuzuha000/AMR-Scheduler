# Task 3 — Upper-Level Candidate Headroom Report (`U0..U4` + Fixed `R1_Robust`)

- Full DEV Evaluation: `6 scenarios x 3 seeds (6001-6003) = 18 episodes` per variant (`8001-8010` untouched; zero RL).
- Best Deterministic Upper Variant on DEV: **`U1a_EdgeSpan_Fix`**

## 1. Full 18-Episode DEV Comparison

| Variant | ZeroCol | Comp Mean | vs U0 (%) | p (Comp) | Wait Mean | ETA Drift Replans | ROW Stops | Shield Intv | |dETA| Mean (s) | Plan ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U0_Orig_CPSA | 17/18 (94.4%) | 5.556 | +0.00% | 1.0000 | 0.3227 | 174 | 3515 | 5414 | 15.67 | 0.56 |
| U0_Buf1_Macro | 18/18 (100.0%) | 5.333 | -4.00% | 0.5552 | 0.2961 | 153 | 4934 | 6215 | 16.47 | 0.32 |
| U1a_EdgeSpan_Fix | 18/18 (100.0%) | 5.889 | +6.00% | 0.3750 | 0.2814 | 146 | 4456 | 5600 | 16.80 | 0.38 |
| U1b_KinoTime_CPSA | 17/18 (94.4%) | 5.333 | -4.00% | 0.8479 | 0.2699 | 200 | 5821 | 4959 | 2.46 | 3.70 |
| U2_FixedBuf_1.0s | 17/18 (94.4%) | 4.278 | -23.00% | 0.0448 | 0.3304 | 230 | 5607 | 7443 | 2.41 | 3.89 |
| U2_FixedBuf_2.0s | 18/18 (100.0%) | 4.167 | -25.00% | 0.0237 | 0.3450 | 239 | 5377 | 6806 | 3.24 | 4.30 |
| U3_StateDep_Buf | 18/18 (100.0%) | 4.667 | -16.00% | 0.1343 | 0.3118 | 240 | 5404 | 5363 | 2.76 | 8.30 |

## 2. Counterfactual Finite-Pool Oracle Slice (`seed=6001`, 4 Scenarios)

- **`U4_FinitePool_CF_Oracle` vs Best Deterministic (`19` tasks)**: **`-10.53%`** (`17` tasks)
- **`U4_FinitePool_CF_Oracle` vs `U1b_KinoTime_CPSA`**: **`+0.00%`**
- **`U4_FinitePool_CF_Oracle` vs `U0_Orig_CPSA`**: **`-10.53%`**
- Oracle Triggers: `112` plans (`differed_from_u1b=54`, `differed_from_u3=91`, `tags={'U1b_nom': 58, 'U0_lift': 31, 'U1b_B1s': 7, 'U1b_detour': 1, 'U1b_lowturn': 4, 'U1b_short': 3, 'U1b_B2s': 7, 'U3_sdep': 1}`)

| Variant (seed=6001 slice) | ZeroCol | Comp Sum | Comp Mean | Wait Mean | ETA Drift |
|---|---|---:|---:|---:|---:|
| U0_Orig_CPSA | 4/4 | 19 | 4.750 | 0.3027 | 39 |
| U1a_EdgeSpan_Fix | 4/4 | 17 | 4.250 | 0.3503 | 35 |
| U1b_KinoTime_CPSA | 3/4 | 17 | 4.250 | 0.2801 | 57 |
| U2_FixedBuf_1.0s | 4/4 | 19 | 4.750 | 0.3079 | 50 |
| U2_FixedBuf_2.0s | 4/4 | 15 | 3.750 | 0.3660 | 67 |
| U3_StateDep_Buf | 4/4 | 19 | 4.750 | 0.3054 | 57 |
| U4_FinitePool_CF_Oracle | 4/4 | 17 | 4.250 | 0.2913 | 36 |
