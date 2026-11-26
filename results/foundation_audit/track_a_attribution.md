# Track A Attribution Closure (`A1-A2`)

## A1: 2x2 Factorial Disentanglement (`Horizon x Reservation Span`, 18 DEV Episodes)

| Variant | Horizon | Reservation | ZeroCol | Completed | vs U0 | p_comp | WaitRatio | Replans | ETA_Drift | RowGates | Plan_ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `U0_H90_ArrMinus1` | +90 | `arr_i - 1` | 17/18 (94.4%) | 5.556 | +0.00% | 1.0000 | 0.3227 | 592 | 174 | 3515 | 0.54 |
| `U0_Pure_H90_EdgeSpanFix` | +90 | `_reserve_exact_span` | 17/18 (94.4%) | 5.944 | +7.00% | 0.2930 | 0.2888 | 572 | 147 | 3965 | 0.46 |
| `U0_H120_ArrMinus1` | +120 | `arr_i - 1` | 18/18 (100.0%) | 5.667 | +2.00% | 1.0000 | 0.3297 | 591 | 178 | 3655 | 0.33 |
| `U1a_H120_EdgeSpanFix` | +120 | `_reserve_exact_span` | 18/18 (100.0%) | 5.889 | +6.00% | 0.3750 | 0.2814 | 572 | 146 | 4456 | 0.36 |

- **Pure `_reserve_exact_span` effect at H90**: `+0.3880` tasks/ep
- **Pure `_reserve_exact_span` effect at H120**: `+0.2220` tasks/ep
- **Pure `+120` horizon effect under `arr_i - 1`**: `+0.1110` tasks/ep
- **Pure `+120` horizon effect under `_reserve_exact_span`**: `-0.0550` tasks/ep

### A1 Per-Scenario Completed Tasks Mean

| Scenario | U0_H90_ArrMinus1 | U0_Pure_H90_EdgeSpanFix | U0_H120_ArrMinus1 | U1a_H120_EdgeSpanFix |
|---|---:|---:|---:|---:|
| `Occluded_Jct_6R_Clean` | 6.000 | 6.333 | 6.000 | 6.333 |
| `MapA_8R_Clean` | 3.667 | 3.333 | 3.667 | 3.333 |
| `MapB_12R_Clean` | 11.667 | 11.333 | 11.667 | 11.333 |
| `Workshop_8R_Clean` | 0.333 | 0.667 | 0.333 | 0.333 |
| `Occluded_Jct_6R_Noise0.12` | 6.667 | 7.000 | 6.333 | 7.000 |
| `Occluded_Jct_6R_Compound` | 5.000 | 7.000 | 6.000 | 7.000 |

## A2: `eta_drift` Trigger Threshold Sweep (12 Clean DEV Episodes)

| Variant | Drift Thresh | ZeroCol | Completed | vs U0(th=3) | vs Own(th=3) | WaitRatio | Replans | ETA_Drift | |dETA| (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `U0_Orig_CPSA` | `3` | 12/12 (100.0%) | 5.417 | +0.00% | +0.00% | 0.3237 | 423 | 134 | 18.133 |
| `U0_Orig_CPSA` | `6` | 12/12 (100.0%) | 5.250 | -3.08% | -3.08% | 0.3266 | 377 | 91 | 17.892 |
| `U0_Orig_CPSA` | `12` | 12/12 (100.0%) | 5.250 | -3.08% | -3.08% | 0.3069 | 332 | 45 | 17.123 |
| `U0_Orig_CPSA` | `None` | 12/12 (100.0%) | 5.250 | -3.08% | -3.08% | 0.3171 | 287 | 0 | 17.223 |
| `U0_Pure_H90_EdgeSpanFix` | `3` | 11/12 (91.7%) | 5.417 | +0.00% | +0.00% | 0.3066 | 404 | 115 | 18.162 |
| `U0_Pure_H90_EdgeSpanFix` | `6` | 12/12 (100.0%) | 5.333 | -1.54% | -1.54% | 0.3002 | 366 | 75 | 18.089 |
| `U0_Pure_H90_EdgeSpanFix` | `12` | 12/12 (100.0%) | 5.333 | -1.54% | -1.54% | 0.2977 | 333 | 42 | 17.405 |
| `U0_Pure_H90_EdgeSpanFix` | `None` | 12/12 (100.0%) | 5.333 | -1.54% | -1.54% | 0.2958 | 289 | 0 | 17.594 |
| `U1b_KinoTime_CPSA` | `3` | 11/12 (91.7%) | 5.417 | +0.00% | +0.00% | 0.2330 | 431 | 145 | 2.694 |
| `U1b_KinoTime_CPSA` | `6` | 12/12 (100.0%) | 5.417 | +0.00% | +0.00% | 0.2424 | 366 | 81 | 2.894 |
| `U1b_KinoTime_CPSA` | `12` | 12/12 (100.0%) | 5.250 | -3.08% | -3.08% | 0.2706 | 333 | 52 | 2.595 |
| `U1b_KinoTime_CPSA` | `None` | 12/12 (100.0%) | 5.250 | -3.08% | -3.08% | 0.2805 | 281 | 0 | 2.734 |

### A2 `U1b_KinoTime_CPSA` Completed Tasks Mean by Clean Scenario & Threshold

| Clean Scenario | th=3 | th=6 | th=12 | th=None |
|---|---:|---:|---:|---:|
| `Occluded_Jct_6R_Clean` | 5.667 | 5.667 | 5.667 | 5.667 |
| `MapA_8R_Clean` | 5.000 | 4.667 | 4.667 | 4.667 |
| `MapB_12R_Clean` | 10.667 | 10.667 | 10.667 | 10.667 |
| `Workshop_8R_Clean` | 0.333 | 0.667 | 0.000 | 0.000 |
