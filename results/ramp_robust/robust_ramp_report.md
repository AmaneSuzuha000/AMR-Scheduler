# Robust-RAMP-Study: 2x2 Factorial & Blind Confirmation Report

> Elapsed: `790.09 s` · Includes both Frozen and Conditionally-Retrained `R5_RAMP_PPO` (`s42, s43, s44`) · Zero ground-truth in Robust Safety.

1. Task 3 — DEV 2x2 Factorial (`6001–6003`)

| Variant | ZeroCol (%) | Col Events (W/P/I) | Completed (95% CI) | Wait Ratio (95% CI) | Shield Interv | Min Clear (m) | Ctrl ms (avg / P95 / P99) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `R1_Orig` | 100.0% | 0 (0/0/0) | 5.56 [3.8889, 7.2778] | 0.3198 [0.2368, 0.4035] | 269.3 | 13.873 | 0.64/1.14/1.22 |
| `R1_Robust` | 94.4% | 3 (3/0/0) | 5.56 [3.8889, 7.2222] | 0.3226 [0.2405, 0.4036] | 300.9 | 13.846 | 0.72/1.40/1.44 |
| `R5_Orig` | 61.1% | 10020 (0/10020/0) | 6.33 [4.6667, 7.9444] | 0.2664 [0.1931, 0.3432] | 532.6 | 14.296 | 1.01/1.81/1.82 |
| `R5_Robust_Frozen` | 100.0% | 0 (0/0/0) | 4.94 [3.4431, 6.5] | 0.3536 [0.2708, 0.4371] | 452.2 | 13.849 | 1.02/1.92/2.00 |
| `R5_Robust_Retrained` | 100.0% | 0 (0/0/0) | 5.33 [3.7222, 7.0556] | 0.3128 [0.2318, 0.3994] | 432.0 | 13.961 | 0.94/1.65/2.02 |

- **`R5_Robust_Frozen` vs `R1_Robust`**: `ΔCompleted = -0.612` (`-11.02%`, `p = 0.099609`), `ΔWaitRatio = +0.0310` (`-9.61%` wait reduction, `p = 0.495079`), **`GO = False`**
- **`R5_Robust_Retrained` vs `R1_Robust`**: `ΔCompleted = -0.223` (`-4.01%`, `p = 0.70874`), `ΔWaitRatio = -0.0098` (`+3.04%` wait reduction, `p = 0.644135`), **`GO = False`**

2. Task 4 — Fresh Disjoint Blind Confirmation (`7001–7006`)

| Variant | ZeroCol (%) | Col Events (W/P/I) | Completed (95% CI) | Wait Ratio (95% CI) | Shield Interv | Min Clear (m) | Ctrl ms (avg / P95 / P99) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `R1_Orig` | 100.0% | 0 (0/0/0) | 4.83 [3.916, 5.75] | 0.2601 [0.2023, 0.3187] | 318.4 | 14.473 | 0.68/1.16/1.45 |
| `R1_Robust` | 100.0% | 0 (0/0/0) | 4.69 [3.8056, 5.5562] | 0.2896 [0.2264, 0.3523] | 370.1 | 14.158 | 0.80/1.59/2.08 |
| `R5_Orig` | 58.3% | 12247 (0/12247/0) | 5.06 [3.9438, 6.1944] | 0.2829 [0.2186, 0.3514] | 604.8 | 14.547 | 1.13/1.91/2.51 |
| `R5_Robust_Frozen` | 100.0% | 0 (0/0/0) | 4.44 [3.5271, 5.4444] | 0.3325 [0.2602, 0.4024] | 553.5 | 14.090 | 1.11/2.21/2.56 |
| `R5_Robust_Retrained` | 100.0% | 0 (0/0/0) | 4.28 [3.2778, 5.3056] | 0.3426 [0.2762, 0.4071] | 589.1 | 14.208 | 1.20/2.17/2.68 |

- **`R5_Robust_Frozen` vs `R1_Robust`**: `ΔCompleted = -0.250` (`-5.33%`, `p = 0.344944`), `ΔWaitRatio = +0.0429` (`-14.81%` wait reduction, `p = 0.02096`), **`GO = False`**
- **`R5_Robust_Retrained` vs `R1_Robust`**: `ΔCompleted = -0.416` (`-8.86%`, `p = 0.160421`), `ΔWaitRatio = +0.0530` (`-18.30%` wait reduction, `p = 0.009959`), **`GO = False`**

