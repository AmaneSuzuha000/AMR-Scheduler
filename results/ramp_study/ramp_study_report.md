# RAMP-Primitive-Study Confirmation Report (`Gate 2–3`)

> Elapsed: `752.99 s` · Held-out seeds: `(6001, 6002, 6003)` · Training seeds: `(42, 43, 44)`

## 1. Independent Confirmation Across 4 Regimes (`Occluded_Jct_6R`, `MapA_8R`, `MapB_12R`, `Workshop_8R`)

| Controller | ZeroCol (%) | Completed | Makespan (s) | Wait Ratio | Shield Interv | Mean Clear (m) | Mean Jerk | Ctrl (ms/bot) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `R0_Det_Primitive` | 91.7% | 4.58 | 98.75 | 0.3132 | 350.8 | 17.007 | 0.583 | 0.851 |
| `R1_Det_ResMargin` | 100.0% | 5.42 | 98.75 | 0.3144 | 253.2 | 17.212 | 0.836 | 0.543 |
| `CPSA_Follower` | 83.3% | 5.42 | 98.75 | 0.2717 | 824.3 | 17.438 | 0.636 | 0.638 |
| `ORCA_Uni` | 91.7% | 6.25 | 98.75 | 0.1806 | 515.5 | 17.938 | 1.007 | 0.487 |
| `DWA` | 91.7% | 4.00 | 98.75 | 0.3118 | 988.2 | 18.312 | 1.622 | 1.890 |
| `Res_DWA` | 91.7% | 3.92 | 98.75 | 0.3185 | 1031.3 | 18.259 | 1.631 | 2.018 |
| `Uni_MPC` | 100.0% | 2.33 | 98.75 | 0.5819 | 60.2 | 18.538 | 1.124 | 1.340 |
| `TPG_ADG_Follower` | 83.3% | 4.33 | 98.75 | 0.2769 | 890.6 | 17.032 | 0.625 | 0.617 |
| `MAWPF_Kin_Ref` | 100.0% | 5.42 | 98.75 | 0.3376 | 1013.8 | 17.166 | 0.655 | 0.751 |
| `R2_BC_Only` | 83.3% | 5.92 | 98.75 | 0.2997 | 711.0 | 17.364 | 0.823 | 0.988 |
| `R3_Primitive_PPO` | 83.3% | 5.25 | 98.75 | 0.2069 | 740.7 | 17.737 | 0.716 | 0.898 |
| `R4_Spatial_Intent_PPO` | 83.3% | 5.25 | 98.75 | 0.2069 | 740.7 | 17.737 | 0.716 | 0.922 |
| `R5_RAMP_PPO` | 83.3% | 6.17 | 98.75 | 0.2512 | 584.7 | 17.810 | 0.884 | 0.961 |
| `R6_Orig_Cont_RGPPO` | 100.0% | 5.58 | 98.75 | 0.2372 | 720.5 | 18.342 | 0.852 | 0.890 |

## 2. Paired Wilcoxon (`R5_RAMP_PPO` vs Comparators)

| Comparator | `ΔCompleted` | `p_comp` | `ΔWaitRatio` | `p_wait` |
| :--- | :---: | :---: | :---: | :---: |
| `R0_Det_Primitive` | +1.58 | 0.0234 | -0.0620 | 0.0640 |
| `R1_Det_ResMargin` | +0.75 | 0.1406 | -0.0632 | 0.0210 |
| `CPSA_Follower` | +0.75 | 0.5273 | -0.0205 | 0.6772 |
| `ORCA_Uni` | -0.08 | 0.7305 | +0.0706 | 0.0923 |
| `DWA` | +2.17 | 0.0078 | -0.0606 | 0.0771 |
| `Res_DWA` | +2.25 | 0.0156 | -0.0673 | 0.1514 |
| `Uni_MPC` | +3.83 | 0.0020 | -0.3307 | 0.0005 |
| `TPG_ADG_Follower` | +1.83 | 0.0391 | -0.0258 | 0.3804 |
| `MAWPF_Kin_Ref` | +0.75 | 0.3906 | -0.0864 | 0.0024 |
| `R2_BC_Only` | +0.25 | 0.7031 | -0.0485 | 0.0269 |
| `R3_Primitive_PPO` | +0.92 | 0.2656 | +0.0442 | 0.3013 |
| `R4_Spatial_Intent_PPO` | +0.92 | 0.2656 | +0.0442 | 0.3013 |
| `R6_Orig_Cont_RGPPO` | +0.58 | 0.3320 | +0.0140 | 0.6772 |

## 3. Single-Variable vs Compound Stress (`Occluded_Junction`)

| Stress Condition | Controller | ZeroCol (%) | Completed | Wait Ratio | Shield Interv |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `Clean_0s` | `R1_Det_ResMargin` | 100.0% | 6.00/12 | 0.3821 | 45.0 |
| `Clean_0s` | `R4_Spatial_Intent_PPO` | 100.0% | 6.00/12 | 0.2024 | 286.7 |
| `Clean_0s` | `R5_RAMP_PPO` | 100.0% | 6.00/12 | 0.3074 | 308.7 |
| `Clean_0s` | `Res_DWA` | 100.0% | 4.33/12 | 0.4781 | 716.3 |
| `Lag_Only_1s` | `R1_Det_ResMargin` | 100.0% | 5.67/12 | 0.4190 | 44.7 |
| `Lag_Only_1s` | `R4_Spatial_Intent_PPO` | 100.0% | 6.00/12 | 0.2024 | 286.7 |
| `Lag_Only_1s` | `R5_RAMP_PPO` | 100.0% | 6.00/12 | 0.3070 | 285.7 |
| `Lag_Only_1s` | `Res_DWA` | 100.0% | 5.00/12 | 0.3780 | 544.7 |
| `Lag_Only_2s` | `R1_Det_ResMargin` | 100.0% | 5.67/12 | 0.4190 | 44.7 |
| `Lag_Only_2s` | `R4_Spatial_Intent_PPO` | 100.0% | 6.00/12 | 0.2024 | 286.7 |
| `Lag_Only_2s` | `R5_RAMP_PPO` | 100.0% | 6.00/12 | 0.3070 | 285.7 |
| `Lag_Only_2s` | `Res_DWA` | 100.0% | 5.00/12 | 0.3780 | 544.7 |
| `Drop_Only_0.2` | `R1_Det_ResMargin` | 100.0% | 5.00/12 | 0.4655 | 129.7 |
| `Drop_Only_0.2` | `R4_Spatial_Intent_PPO` | 100.0% | 6.33/12 | 0.2195 | 279.3 |
| `Drop_Only_0.2` | `R5_RAMP_PPO` | 100.0% | 6.33/12 | 0.2774 | 309.7 |
| `Drop_Only_0.2` | `Res_DWA` | 100.0% | 5.00/12 | 0.4335 | 641.3 |
| `Noise_Only_0.12m` | `R1_Det_ResMargin` | 100.0% | 6.00/12 | 0.3070 | 303.3 |
| `Noise_Only_0.12m` | `R4_Spatial_Intent_PPO` | 0.0% | 4.67/12 | 0.3563 | 550.0 |
| `Noise_Only_0.12m` | `R5_RAMP_PPO` | 0.0% | 6.67/12 | 0.2840 | 395.0 |
| `Noise_Only_0.12m` | `Res_DWA` | 66.7% | 3.67/12 | 0.5022 | 762.7 |
| `Compound_2s_0.1_0.12` | `R1_Det_ResMargin` | 100.0% | 5.67/12 | 0.3545 | 299.7 |
| `Compound_2s_0.1_0.12` | `R4_Spatial_Intent_PPO` | 0.0% | 5.00/12 | 0.3708 | 586.7 |
| `Compound_2s_0.1_0.12` | `R5_RAMP_PPO` | 33.3% | 6.67/12 | 0.3097 | 461.7 |
| `Compound_2s_0.1_0.12` | `Res_DWA` | 66.7% | 4.00/12 | 0.4210 | 657.0 |

## 4. GO / NO-GO Verdict

- R5_RAMP_PPO vs R4_Spatial_Intent_PPO: comp=+17.5%, wait=-21.4%
- R5_RAMP_PPO vs R1_Det_ResMargin: comp=+13.8%, wait=+20.1%
- Safety non-degradation: False (R5=83.3%, R4=83.3%, R1=100.0%)
- Verdict: NO-GO / PARTIAL (RAMP-PPO did not clear >=3% over both R4 and R1; report honestly without inflating complexity)
