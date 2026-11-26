# Multi-Fidelity Foundation Compatibility & Calibration Audit (`F1-F5`)

## F2: Same-Task Cross-Fidelity Telemetry Alignment (`Discrete` vs `Continuous U1a + R1_Robust`)

| Scenario | Seed | Comp (D / C) | Makespan (D / C, s) | C/D Ratio | WaitRatio (D / C) | Leg Pearson r | Leg MAE (s) | Bias (C-D, s) | Cont ZeroCol |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `MapA_8R` | 6001 | 6/16 vs 8/16 | 161.0 / 180.0 | 1.118 | 0.494 / 0.590 | 0.370 | 6.45 | -3.13 | True |
| `MapB_10R` | 6001 | 17/18 vs 17/18 | 194.0 / 200.0 | 1.031 | 0.038 / 0.176 | 0.635 | 11.64 | -6.90 | True |
| `OccludedJct_6R` | 6001 | 12/12 vs 6/12 | 140.0 / 180.0 | 1.286 | 0.043 / 0.622 | 0.322 | 7.62 | +4.34 | True |
| `Workshop_DistinctBay_4R` | 6001 | 6/6 vs 5/6 | 396.0 / 450.0 | 1.136 | 0.037 / 0.360 | 0.557 | 29.22 | -13.67 | True |
| `Workshop_SpurContention_6R` | 6001 | 8/10 vs 8/10 | 449.0 / 450.0 | 1.002 | 0.139 / 0.156 | 0.070 | 41.12 | -27.00 | True |
| `MapA_8R` | 6002 | 15/16 vs 8/16 | 133.0 / 180.0 | 1.353 | 0.109 / 0.588 | 0.539 | 6.40 | -4.40 | True |
| `MapB_10R` | 6002 | 18/18 vs 12/18 | 177.0 / 200.0 | 1.130 | 0.020 / 0.375 | 0.844 | 8.12 | -6.13 | True |
| `OccludedJct_6R` | 6002 | 6/12 vs 6/12 | 98.0 / 180.0 | 1.837 | 0.681 / 0.736 | 0.906 | 6.06 | -5.56 | True |
| `Workshop_DistinctBay_4R` | 6002 | 3/6 vs 6/6 | 309.0 / 380.2 | 1.231 | 0.212 / 0.060 | 0.740 | 21.12 | -10.12 | True |
| `Workshop_SpurContention_6R` | 6002 | 6/10 vs 2/10 | 334.0 / 450.0 | 1.347 | 0.308 / 0.762 | -0.079 | 34.00 | +19.60 | True |

## F3 & F4: Fixed Upper Dispatch Replay & Cross-Fidelity Ranking Consistency

- **5 Classical Allocators Makespan Rank Correlation**: Spearman rho = `0.25`, Kendall tau = `0.1429`
- **5 Classical Allocators Tardiness Rank Correlation**: Spearman rho = `1.0`, Kendall tau = `1.0`
- **Allocator Pairwise Makespan Rank Inversions**: `3`

### F3 Fixed-Dispatch Allocator Replay (`seed=6001`)

| Scenario | Allocator | Comp (D / C) | Makespan (D / C, s) | Tardiness (D / C, s) | Cont ZeroCol |
|---|---|---:|---:|---:|---:|
| `MapA_8R` | `fifo+greedy` | 6/16 vs 4/16 | 161.0 / 180.0 | 1056.0 / 1143.4 | True |
| `MapA_8R` | `edd+hungarian` | 16/16 vs 7/16 | 163.0 / 180.0 | 245.4 / 922.6 | True |
| `MapA_8R` | `priority+nearest` | 15/16 vs 8/16 | 172.0 / 180.0 | 363.5 / 799.0 | True |
| `MapA_8R` | `spt+greedy` | 6/16 vs 4/16 | 161.0 / 180.0 | 1056.0 / 1143.4 | True |
| `MapA_8R` | `cr+greedy` | 6/16 vs 4/16 | 161.0 / 180.0 | 1056.0 / 1143.4 | True |
| `MapB_10R` | `fifo+greedy` | 17/18 vs 13/18 | 194.0 / 200.0 | 372.9 / 580.8 | True |
| `MapB_10R` | `edd+hungarian` | 11/18 vs 12/18 | 192.0 / 200.0 | 905.5 / 660.4 | True |
| `MapB_10R` | `priority+nearest` | 18/18 vs 13/18 | 175.0 / 200.0 | 330.9 / 654.9 | True |
| `MapB_10R` | `spt+greedy` | 17/18 vs 13/18 | 194.0 / 200.0 | 372.9 / 580.8 | True |
| `MapB_10R` | `cr+greedy` | 17/18 vs 13/18 | 194.0 / 200.0 | 372.9 / 580.8 | True |
| `OccludedJct_6R` | `fifo+greedy` | 12/12 vs 4/12 | 140.0 / 180.0 | 75.5 / 940.2 | True |
| `OccludedJct_6R` | `edd+hungarian` | 12/12 vs 6/12 | 145.0 / 180.0 | 172.8 / 722.8 | True |
| `OccludedJct_6R` | `priority+nearest` | 10/12 vs 4/12 | 177.0 / 180.0 | 462.5 / 921.3 | True |
| `OccludedJct_6R` | `spt+greedy` | 12/12 vs 4/12 | 140.0 / 180.0 | 75.5 / 940.2 | True |
| `OccludedJct_6R` | `cr+greedy` | 12/12 vs 4/12 | 140.0 / 180.0 | 75.5 / 940.2 | True |
| `Workshop_DistinctBay_4R` | `fifo+greedy` | 6/6 vs 5/6 | 396.0 / 450.0 | 686.1 / 678.9 | True |
| `Workshop_DistinctBay_4R` | `edd+hungarian` | 5/6 vs 4/6 | 352.0 / 450.0 | 722.9 / 710.9 | True |
| `Workshop_DistinctBay_4R` | `priority+nearest` | 4/6 vs 0/6 | 354.0 / 450.0 | 1010.1 / 1612.1 | True |
| `Workshop_DistinctBay_4R` | `spt+greedy` | 6/6 vs 5/6 | 396.0 / 450.0 | 686.1 / 678.9 | True |
| `Workshop_DistinctBay_4R` | `cr+greedy` | 6/6 vs 5/6 | 396.0 / 450.0 | 686.1 / 678.9 | True |

### Cohort A Frozen Track B Policies Replayed Across Discrete & Continuous

| Episode | Policy | Orig Track B Cmax | Comp (D / C) | Replay Makespan (D / C, s) | Tardiness (D / C, s) | EdgeLen/Euc | Cont ZeroCol |
|---|---|---:|---:|---:|---:|---:|---:|
| `track1_nominal_2901` | `V20_1_NAJS` | 548.0 | 8/8 vs 6/8 | 330.0 / 600.0 | 278.0 / 782.0 | 1.02 | True |
| `track1_nominal_2901` | `V19_ADAPTIVE_RL` | 536.0 | 8/8 vs 7/8 | 314.0 / 600.0 | 443.0 / 453.0 | 1.02 | True |
| `track1_nominal_2901` | `V18_EA_JOINT` | 455.0 | 8/8 vs 7/8 | 399.0 / 600.0 | 297.0 / 417.0 | 1.02 | True |
| `track1_nominal_2901` | `rule_v10` | 547.0 | 8/8 vs 5/8 | 565.0 / 600.0 | 786.0 / 1302.0 | 1.02 | True |
| `track1_nominal_2901` | `IJCAI26_PGMPO` | 357.0 | 8/8 vs 4/8 | 394.0 / 600.0 | 431.0 / 1813.0 | 1.02 | True |
| `track1_nominal_2901` | `EJOR26_Wu_GRU_Attn` | 423.0 | 8/8 vs 6/8 | 454.0 / 600.0 | 391.0 / 782.0 | 1.02 | True |
| `track1_nominal_2901` | `RAL26_DetourDelta` | 393.0 | 8/8 vs 6/8 | 324.0 / 600.0 | 271.0 / 782.0 | 1.02 | True |
| `track1_nominal_2902` | `V20_1_NAJS` | 376.0 | 8/8 vs 5/8 | 306.0 / 600.0 | 227.0 / 1281.0 | 1.02 | True |
| `track1_nominal_2902` | `V19_ADAPTIVE_RL` | 356.0 | 8/8 vs 5/8 | 306.0 / 600.0 | 343.0 / 1281.0 | 1.02 | True |
| `track1_nominal_2902` | `V18_EA_JOINT` | 362.0 | 8/8 vs 6/8 | 358.0 / 600.0 | 291.0 / 886.0 | 1.02 | True |
| `track1_nominal_2902` | `rule_v10` | 362.0 | 8/8 vs 6/8 | 358.0 / 600.0 | 291.0 / 886.0 | 1.02 | True |
| `track1_nominal_2902` | `IJCAI26_PGMPO` | 332.0 | 8/8 vs 5/8 | 323.0 / 600.0 | 312.0 / 1443.0 | 1.02 | True |
| `track1_nominal_2902` | `EJOR26_Wu_GRU_Attn` | 389.0 | 8/8 vs 5/8 | 446.0 / 600.0 | 377.0 / 1281.0 | 1.02 | True |
| `track1_nominal_2902` | `RAL26_DetourDelta` | 354.0 | 8/8 vs 5/8 | 408.0 / 600.0 | 367.0 / 1281.0 | 1.02 | True |
| `topo4_perimeter_bypass_standard_27101` | `V20_1_NAJS` | 336.0 | 21/21 vs 7/21 | 629.0 / 320.0 | 1926.0 / 2761.0 | 1.08 | True |
| `topo4_perimeter_bypass_standard_27101` | `V19_ADAPTIVE_RL` | 170.0 | 21/21 vs 5/21 | 327.0 / 320.0 | 827.0 / 3126.0 | 1.08 | True |
| `topo4_perimeter_bypass_standard_27101` | `V18_EA_JOINT` | 212.0 | 21/21 vs 3/21 | 327.0 / 320.0 | 688.0 / 3603.0 | 1.08 | True |
| `topo4_perimeter_bypass_standard_27101` | `rule_v10` | 202.0 | 21/21 vs 7/21 | 224.0 / 320.0 | 304.0 / 2716.0 | 1.08 | True |
| `topo4_perimeter_bypass_standard_27101` | `IJCAI26_PGMPO` | 172.0 | 21/21 vs 4/21 | 348.0 / 320.0 | 931.0 / 3461.0 | 1.08 | True |
| `topo4_perimeter_bypass_standard_27101` | `EJOR26_Wu_GRU_Attn` | 147.0 | 21/21 vs 10/21 | 265.0 / 320.0 | 413.0 / 2753.0 | 1.08 | True |
| `topo4_perimeter_bypass_standard_27101` | `RAL26_DetourDelta` | 150.0 | 21/21 vs 7/21 | 251.0 / 320.0 | 423.0 / 3084.0 | 1.08 | True |
| `topo4_narrow_spine_standard_27102` | `V20_1_NAJS` | 208.0 | 21/21 vs 6/21 | 293.0 / 320.0 | 896.0 / 2886.0 | 0.98 | True |
| `topo4_narrow_spine_standard_27102` | `V19_ADAPTIVE_RL` | 259.0 | 21/21 vs 1/21 | 442.0 / 320.0 | 2272.0 / 4059.0 | 0.98 | True |
| `topo4_narrow_spine_standard_27102` | `V18_EA_JOINT` | 160.0 | 21/21 vs 2/21 | 192.0 / 320.0 | 649.0 / 3809.0 | 0.98 | True |
| `topo4_narrow_spine_standard_27102` | `rule_v10` | 142.0 | 21/21 vs 4/21 | 178.0 / 320.0 | 234.0 / 3361.0 | 0.98 | True |
| `topo4_narrow_spine_standard_27102` | `IJCAI26_PGMPO` | 144.0 | 21/21 vs 7/21 | 171.0 / 320.0 | 163.0 / 3007.0 | 0.98 | True |
| `topo4_narrow_spine_standard_27102` | `EJOR26_Wu_GRU_Attn` | 157.0 | 21/21 vs 1/21 | 159.0 / 320.0 | 227.0 / 4109.0 | 0.98 | True |
| `topo4_narrow_spine_standard_27102` | `RAL26_DetourDelta` | 174.0 | 21/21 vs 4/21 | 325.0 / 320.0 | 890.0 / 3653.0 | 0.98 | True |

- **Cohort A `track1` Ranking Consistency (7 Track B Policies)**: Tardiness Spearman rho=`0.2965` (tau=`0.2057`), DiscMakespan vs `-ContComp` rho=`0.0591` (tau=`0.0563`), Raw Makespan/Comp rho=`0.0`/`0.0` (horizon/ceiling ties)

- **Cohort A `topo4` Ranking Consistency (7 Track B Policies)**: Tardiness Spearman rho=`0.2857` (tau=`0.2381`), DiscMakespan vs `-ContComp` rho=`-0.169` (tau=`-0.1155`), Raw Makespan/Comp rho=`0.0`/`0.0` (horizon/ceiling ties)

## F5: Non-RL Continuous -> Discrete ETA Calibration Probe (Held-out `seed=6002`)

- **Fitted speed scale multipliers (`seed=6001` train)**: `{'MapA_8R': 1.2, 'MapB_10R': 1.2567, 'OccludedJct_6R': 0.9091, 'Workshop_DistinctBay_4R': 1.1951}` (global median=`1.1976`)
- **Held-out Makespan MAE**: Uncalibrated `60.69 s` -> Calibrated Discrete `53.31 s` (`-12.15%`)
- **Held-out Per-Leg Duration MAE**: Uncalibrated `9.04 s` -> Calibrated Discrete `8.01 s` (`-11.42%`), Ridge Surrogate `4.87 s` (`-46.19%`)
- **Held-out Allocator Makespan Rank Consistency**: Spearman rho `0.9682` -> `0.0` (Kendall tau `0.9258` -> `0.0`; scalar calibration fixes Workshop partial dropouts e.g. `edd+hungarian` 307s -> 445s vs Cont 450s, while structural spur/turn bottlenecks still invert `priority+nearest` ranking)
