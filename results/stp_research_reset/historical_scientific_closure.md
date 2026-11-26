# Task 1: Historical Scientific Closure (`Execution-Aware-STP-Audit`)

- **Frozen Manifest Integrity**: `13/13` SHA-256 matches (`drift_count = 0`)
- **Blind Confirmation Seeds (`8001–8010`)**: `0` prior uses (**Strictly Untouched**)

## 1. Exact Safety & Performance Accounting for Frozen `R1_Robust`

- **`DEV_6001_6003` (`18` eps)**: `17/18` (`94.4%` ZeroCol) — **`0` peer/intruder collisions**, `1` episode (`Occluded_Jct_6R_Compound`, `seed=6001`) had **`3` wall-collision substep ticks** (`1` independent event at `t = 39.25 s`, culprit `[{'rid': 'R_00', 'x': 17.648, 'y': 8.778, 'theta': -2.422, 'v': 0.0, 'omega': -0.703, 'wall_clear_m': -0.0105, 'cur_node': 'C_6_3', 'phase': 'to_dropoff'}]`).
- **`BLIND_7001_7006` (`36` eps)**: `36/36` (`100.0%` ZeroCol, `0` wall/peer/intruder events; `4.69` completed, `0.2896` wait ratio).
- **Task 4 Conflict-Order Suites (`15` eps)**: `15/15` (`100.0%` ZeroCol, `5.667` completed, `0.3158` wait ratio).

## 2. Verified vs Falsified Mechanisms

| Mechanism | Verdict |
| :--- | :--- |
| CPS-A* Upstream Reservation & Congestion Routing | `GO (Retained)` |
| Continuous Unicycle Motion & Two-Disc Capsule Swept Check | `GO (Retained)` |
| Temporal Reservation Margin m_ij(z) | `GO (Retained, M2 +40.7% over M1)` |
| Kinodynamic Motion Primitive Pool (K=6) | `GO (Retained)` |
| Uncertainty-Aware Safety Check (robust_safety) | `GO (Retained, 0 peer cols in DEV+BLIND)` |
| R1_Robust Deterministic Coordinator | `FROZEN STRONG BASELINE` |
| RG-PPO Direct Continuous (v, omega) | `NO-GO (Falsified)` |
| RAMP-PPO Primitive Selection | `NO-GO (Falsified: -5.33%/-8.86% vs R1_Robust; CF Oracle <=+0.85% fleet)` |
| STCP-RL Conflict Passing-Order RL | `NO-GO (Falsified: +0.43% over Det; severe BTPG-max/SJRL overlap)` |

## 3. Epistemological Boundary on Counterfactual Audits

Task 3 (+0.58% ego / +0.85% fleet over N=60 snapshots) and Task 4 (+0.43% over N=36 conflicts) measure local single-decision counterfactual headroom given fixed R1_Robust continuation. They falsify incremental value for those two local RL layers, NOT global theoretical optimality of R1_Robust.
