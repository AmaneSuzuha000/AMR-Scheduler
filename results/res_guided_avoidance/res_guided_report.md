# Spatiotemporal Reservation-Guided RL Multi-Robot Dynamic Avoidance Report (`P0–P4`)

> Elapsed: `3152.1 s` · Unified capsule footprint (`1.20 m × 0.90 m`, `r = 0.45 m`) + `M_sub = 5` swept collision check + Braking-Invariant Shield + Squashed-Gaussian PPO.

## 1. Bench A — Occluded Intersections & Narrow Corridors (Aggregated Across Maps & Held-Out Seeds)

| Controller | ZeroCol (%) | Completed | Makespan (s) | Wait Ratio | Mean Jerk | Shield Interv | Infeasible | Replans | Ctrl (ms/bot) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ORCA_Uni` | 100.0% | 1.11 | 100.00 | 0.6832 | 0.380 | 1778.9 | 9726 | 32.7 | 2.027 |
| `DWA` | 100.0% | 0.56 | 100.00 | 0.6896 | 0.647 | 1799.9 | 8114 | 33.0 | 3.288 |
| `Res_DWA` | 100.0% | 1.00 | 100.00 | 0.6816 | 0.672 | 1785.4 | 7985 | 32.9 | 3.438 |
| `Uni_MPC` | 100.0% | 0.44 | 100.00 | 0.6845 | 0.764 | 315.4 | 2095 | 31.9 | 1.860 |
| `Res_Yield` | 100.0% | 0.89 | 100.00 | 0.6693 | 0.328 | 1611.3 | 6257 | 34.2 | 1.984 |
| `BC_Only` | 100.0% | 1.00 | 100.00 | 0.6650 | 0.448 | 1910.2 | 10271 | 33.6 | 2.389 |
| `Pure_PPO_NoBC` | 88.9% | 0.00 | 100.00 | 0.8993 | 0.115 | 2659.7 | 7979 | 38.4 | 2.843 |
| `Plain_Kin_PPO` | 100.0% | 1.22 | 100.00 | 0.6967 | 0.314 | 2008.4 | 10566 | 34.4 | 2.622 |
| `Intent_NoTime_PPO` | 100.0% | 1.22 | 100.00 | 0.6867 | 0.347 | 1936.2 | 9242 | 33.3 | 2.538 |
| `RG_PPO` | 100.0% | 1.00 | 100.00 | 0.6559 | 0.451 | 1855.6 | 11836 | 34.1 | 2.372 |

## 2. Paired Wilcoxon Comparisons (`RG_PPO` vs Comparators on Bench A)

| Comparator | `ΔWaitRatio` (`RG - Base`) | `p_wait` | `ΔMakespan` (s) | `p_makespan` |
| :--- | :---: | :---: | :---: | :---: |
| `ORCA_Uni` | -0.0273 | 0.2031 | +0.00 | 1.0000 |
| `DWA` | -0.0337 | 0.4961 | +0.00 | 1.0000 |
| `Res_DWA` | -0.0256 | 0.6523 | +0.00 | 1.0000 |
| `Uni_MPC` | -0.0286 | 0.8203 | +0.00 | 1.0000 |
| `Res_Yield` | -0.0134 | 0.7344 | +0.00 | 1.0000 |
| `BC_Only` | -0.0091 | 0.7344 | +0.00 | 1.0000 |
| `Pure_PPO_NoBC` | -0.2434 | 0.0039 | +0.00 | 1.0000 |
| `Plain_Kin_PPO` | -0.0408 | 0.2500 | +0.00 | 1.0000 |
| `Intent_NoTime_PPO` | -0.0308 | 0.1641 | +0.00 | 1.0000 |

## 3. Bench B — Fleet Density Scaling (`N = 4..24` Simultaneous Robots)

| Map & Fleet | Controller | ZeroCol (%) | Completed | Makespan (s) | Wait Ratio | Ctrl (ms/bot) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Workshop_N4` | `ORCA_Uni` | 66.7% | 0.00/8 | 140.00 | 0.7212 | 2.673 |
| `Workshop_N4` | `Res_DWA` | 100.0% | 0.00/8 | 140.00 | 0.5456 | 2.952 |
| `Workshop_N4` | `Uni_MPC` | 100.0% | 0.00/8 | 140.00 | 0.3967 | 1.413 |
| `Workshop_N4` | `Plain_Kin_PPO` | 100.0% | 0.00/8 | 140.00 | 0.7732 | 3.407 |
| `Workshop_N4` | `Intent_NoTime_PPO` | 66.7% | 0.00/8 | 140.00 | 0.7393 | 3.417 |
| `Workshop_N4` | `RG_PPO` | 66.7% | 0.00/8 | 140.00 | 0.6355 | 2.559 |
| `Workshop_N8` | `ORCA_Uni` | 100.0% | 0.67/14 | 150.00 | 0.5943 | 1.826 |
| `Workshop_N8` | `Res_DWA` | 100.0% | 0.33/14 | 150.00 | 0.5740 | 3.184 |
| `Workshop_N8` | `Uni_MPC` | 100.0% | 0.33/14 | 150.00 | 0.5305 | 1.479 |
| `Workshop_N8` | `Plain_Kin_PPO` | 100.0% | 0.00/14 | 150.00 | 0.7708 | 3.098 |
| `Workshop_N8` | `Intent_NoTime_PPO` | 100.0% | 0.00/14 | 150.00 | 0.6427 | 2.464 |
| `Workshop_N8` | `RG_PPO` | 100.0% | 0.67/14 | 150.00 | 0.6508 | 2.516 |
| `Workshop_N12` | `ORCA_Uni` | 100.0% | 0.67/18 | 150.00 | 0.7387 | 2.114 |
| `Workshop_N12` | `Res_DWA` | 100.0% | 0.00/18 | 150.00 | 0.6496 | 3.122 |
| `Workshop_N12` | `Uni_MPC` | 100.0% | 0.00/18 | 150.00 | 0.6742 | 2.272 |
| `Workshop_N12` | `Plain_Kin_PPO` | 66.7% | 0.00/18 | 150.00 | 0.6830 | 2.734 |
| `Workshop_N12` | `Intent_NoTime_PPO` | 100.0% | 0.00/18 | 150.00 | 0.6853 | 2.945 |
| `Workshop_N12` | `RG_PPO` | 100.0% | 0.00/18 | 150.00 | 0.6740 | 2.657 |
| `Workshop_N16` | `ORCA_Uni` | 100.0% | 0.33/22 | 160.00 | 0.8227 | 2.638 |
| `Workshop_N16` | `Res_DWA` | 100.0% | 0.00/22 | 160.00 | 0.7723 | 4.109 |
| `Workshop_N16` | `Uni_MPC` | 100.0% | 0.00/22 | 160.00 | 0.7450 | 2.046 |
| `Workshop_N16` | `Plain_Kin_PPO` | 100.0% | 0.00/22 | 160.00 | 0.8074 | 3.453 |
| `Workshop_N16` | `Intent_NoTime_PPO` | 100.0% | 0.00/22 | 160.00 | 0.8154 | 3.485 |
| `Workshop_N16` | `RG_PPO` | 33.3% | 0.00/22 | 160.00 | 0.7605 | 2.998 |
| `Workshop_N24` | `ORCA_Uni` | 100.0% | 0.00/28 | 160.00 | 0.8017 | 2.121 |
| `Workshop_N24` | `Res_DWA` | 66.7% | 0.00/28 | 160.00 | 0.8629 | 4.318 |
| `Workshop_N24` | `Uni_MPC` | 100.0% | 0.00/28 | 160.00 | 0.7690 | 2.215 |
| `Workshop_N24` | `Plain_Kin_PPO` | 66.7% | 0.00/28 | 160.00 | 0.7570 | 3.078 |
| `Workshop_N24` | `Intent_NoTime_PPO` | 33.3% | 0.00/28 | 160.00 | 0.7574 | 3.056 |
| `Workshop_N24` | `RG_PPO` | 100.0% | 0.00/28 | 160.00 | 0.7763 | 2.861 |
| `MapB_N16` | `ORCA_Uni` | 100.0% | 4.67/24 | 90.00 | 0.5190 | 1.309 |
| `MapB_N16` | `Res_DWA` | 100.0% | 3.00/24 | 90.00 | 0.5570 | 2.626 |
| `MapB_N16` | `Uni_MPC` | 100.0% | 4.33/24 | 90.00 | 0.6273 | 1.221 |
| `MapB_N16` | `Plain_Kin_PPO` | 100.0% | 5.67/24 | 90.00 | 0.3735 | 1.241 |
| `MapB_N16` | `Intent_NoTime_PPO` | 100.0% | 3.67/24 | 90.00 | 0.5645 | 1.715 |
| `MapB_N16` | `RG_PPO` | 100.0% | 4.33/24 | 90.00 | 0.4436 | 1.413 |

## 4. Bench C — Stale Reservation Lag & Sensor Noise Stress (`Occluded_Junction`)

| Lag (s) / Dropout / Noise | Controller | ZeroCol (%) | Completed | Makespan (s) | Wait Ratio | Shield Interv |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `0.0s / drop=0.0 / σ=0.0m` | `Res_DWA` | 100.0% | 1.33/12 | 70.00 | 0.7892 | 1252.7 |
| `0.0s / drop=0.0 / σ=0.0m` | `Plain_Kin_PPO` | 100.0% | 1.00/12 | 70.00 | 0.8152 | 1306.7 |
| `0.0s / drop=0.0 / σ=0.0m` | `RG_PPO` | 100.0% | 1.33/12 | 70.00 | 0.7574 | 1207.7 |
| `1.0s / drop=0.0 / σ=0.08m` | `Res_DWA` | 100.0% | 1.67/12 | 70.00 | 0.7167 | 1148.3 |
| `1.0s / drop=0.0 / σ=0.08m` | `Plain_Kin_PPO` | 66.7% | 2.00/12 | 70.00 | 0.7469 | 1172.7 |
| `1.0s / drop=0.0 / σ=0.08m` | `RG_PPO` | 100.0% | 2.00/12 | 70.00 | 0.7310 | 1144.7 |
| `2.0s / drop=0.1 / σ=0.12m` | `Res_DWA` | 0.0% | 1.00/12 | 70.00 | 0.7596 | 1190.3 |
| `2.0s / drop=0.1 / σ=0.12m` | `Plain_Kin_PPO` | 0.0% | 2.67/12 | 70.00 | 0.6550 | 1044.0 |
| `2.0s / drop=0.1 / σ=0.12m` | `RG_PPO` | 0.0% | 1.67/12 | 70.00 | 0.7177 | 1116.7 |
| `4.0s / drop=0.2 / σ=0.15m` | `Res_DWA` | 33.3% | 1.00/12 | 70.00 | 0.7390 | 1166.0 |
| `4.0s / drop=0.2 / σ=0.15m` | `Plain_Kin_PPO` | 0.0% | 2.67/12 | 70.00 | 0.6676 | 1016.7 |
| `4.0s / drop=0.2 / σ=0.15m` | `RG_PPO` | 0.0% | 2.00/12 | 70.00 | 0.7010 | 1104.0 |
