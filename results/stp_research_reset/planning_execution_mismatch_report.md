# Task 2 — Planning–Execution Mismatch Audit (`U0_Orig_CPSA` + Fixed `R1_Robust`)

- DEV Seeds: `[6001, 6002, 6003]` (blind `8001–8010` untouched; zero RL training)

## 1. Per-Scenario Structural & Disturbance Breakdown

| Scenario | Comp | Wait | Replans (Drift%) | ROW Stops (Spill%) | Shield (W/P/I%) | Leg dETA Mean / P90 (s) | Late>1s / Early<-1s (%) | Straight vs Turn dETA (s) | Dur Ratio (Act/Plan) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Occluded_Jct_6R_Clean | 5.67/12 | 0.362 | 85 (25.9%) | 357 (7.0%) | 124 (8/35/0) | -10.75 / -1.34 | 0.0% / 90.0% | -8.03 vs -11.23 | 0.55x |
| MapA_8R_Clean | 5.00/16 | 0.254 | 88 (15.9%) | 390 (0.0%) | 170 (5/50/0) | -18.58 / -6.54 | 0.0% / 100.0% | +0.00 vs -18.58 | 0.51x |
| MapB_12R_Clean | 11.67/20 | 0.016 | 124 (0.0%) | 528 (0.0%) | 105 (10/0/0) | -25.84 / -8.73 | 0.0% / 98.8% | -16.05 vs -26.08 | 0.48x |
| Workshop_8R_Clean | 0.33/12 | 0.567 | 131 (71.0%) | 435 (1.4%) | 1278 (68/15/13) | -14.40 / -3.00 | 10.0% / 90.0% | -14.00 vs -14.44 | 0.77x |
| Occluded_Jct_6R_Noise0.12 | 6.00/12 | 0.405 | 89 (30.3%) | 698 (18.9%) | 1872 (64/20/0) | -10.13 / -0.65 | 2.6% / 86.8% | -8.86 vs -10.46 | 0.57x |
| Occluded_Jct_6R_Compound | 5.33/12 | 0.334 | 82 (25.6%) | 749 (12.2%) | 1624 (41/40/0) | -11.31 / -1.86 | 0.0% / 91.9% | -5.81 vs -12.83 | 0.52x |
| Occluded_Jct_6R_NoSlow | 6.00/12 | 0.362 | 88 (26.1%) | 416 (6.2%) | 137 (9/44/0) | -10.47 / -1.40 | 2.4% / 90.2% | -8.54 vs -10.94 | 0.56x |
| Occluded_Jct_6R_PureStruct | 6.00/12 | 0.362 | 88 (26.1%) | 416 (6.2%) | 137 (9/44/0) | -10.47 / -1.40 | 2.4% / 90.2% | -8.54 vs -10.94 | 0.56x |
| MapA_8R_PureStruct | 3.67/16 | 0.365 | 88 (23.9%) | 397 (0.0%) | 430 (63/27/0) | -17.69 / -7.20 | 2.8% / 97.2% | +0.00 vs -17.69 | 0.52x |
| MapB_12R_PureStruct | 11.67/20 | 0.028 | 124 (0.0%) | 499 (0.0%) | 126 (6/6/0) | -26.15 / -8.74 | 0.0% / 98.8% | -22.20 vs -26.20 | 0.47x |
| Workshop_8R_PureStruct | 0.33/12 | 0.527 | 130 (70.0%) | 367 (18.0%) | 1027 (83/11/0) | -18.55 / -4.00 | 0.0% / 90.9% | -20.00 vs -18.40 | 0.73x |

## 2. Structural vs Stochastic Mismatch Decomposition

- **Main DEV Suite** (`N=252` completed legs): `dETA mean=-17.25s` (`p50=-14.60s`, `p90=-3.31s`, `max=+6.00s`), `late>1s=0.8%`, `early<-1s=94.4%`, `r(turns)=-0.460`, `r(init_hdg)=+0.005`, `r(len)=-0.782`, `r(row)=+0.128`.
- **Pure-Structural Contrast (disturb=0, intr=0)** (`N=214` completed legs): `dETA mean=-18.33s` (`p50=-15.95s`, `p90=-3.65s`, `max=+3.60s`), `late>1s=1.4%`, `early<-1s=94.9%`, `r(turns)=-0.421`, `r(init_hdg)=+0.075`, `r(len)=-0.762`, `r(row)=+0.145`.
