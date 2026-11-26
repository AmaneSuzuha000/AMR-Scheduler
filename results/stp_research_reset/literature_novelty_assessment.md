# Task 4: Literature & Novelty Assessment (`Execution-Aware Spatiotemporal Planning / EA-STP`)

> **Retrieval Provenance & License Notice**: External bibliographic metadata and preprints in this assessment were verified via the `literature-search-arxiv` (arXiv API Terms of Use: https://info.arxiv.org/help/api/index.html) and `literature-search-openalex` (OpenAlex documentation & terms: https://help.openalex.org/) skills, together with local repository sources under `verified archive references` and `reference CPS-A* preprint`. Users should verify the individual license terms of each linked paper prior to redistribution or downstream derivative use.

---

## 1. Purpose of This Literature Audit

Following **Branch C (`STOP RL FOR THIS LAYER`)** on lower-level motion primitives (`RAMP-PPO`) and local conflict passing orders (`STCP-RL`), Tasks 2–3 evaluated whether shifting research upward to **Execution-Aware Spatiotemporal Planning (EA-STP)** — i.e., planner calibration (`U1a`, `U1b`), fixed or state-dependent temporal reservation buffers (`U2`, `U3`), or learned/counterfactual selection over `(route_weight, buffer)` candidates (`U4`) above frozen `R1_Robust` — holds both **(a) meaningful empirical decision headroom (`>= +3%`)** and **(b) defensible scientific novelty**.

Empirically (Task 3), **no learned candidate-pool selector clears the bar**:
- Simply fixing multi-tick edge-span intervals (`U1a_EdgeSpan_Fix`) eliminates `U0`'s collision leak (`17/18 -> 18/18` ZeroCol on DEV `6001-6003`) and improves completion **`+6.00%`** (`5.556 -> 5.889`, `0.376 ms`) with **zero learning**.
- Aligning planner speeds/turns to `R1_Robust` kinematics (`U1b_KinoTime_CPSA`) cuts `|dETA|` by **`-84.3%`** (`15.672s -> 2.461s`) and shines on dense multi-route grids (`MapA_8R_Clean`: `3.667 -> 5.000`, `U2_1.0s = 6.000`, `wait 0.365 -> 0.144`), yet triggers `eta_drift` replan cascades (`146 -> 240`) and capacity lockouts on single-lane bottlenecks (`Occluded_Jct_6R`: `-40%` to `-55%`) and open grids (`MapB_12R_Clean`: `-14%` to `-20%`).
- `U4_FinitePool_CF_Oracle` (`80`-tick counterfactual rollouts across 8 candidate parameterizations on the `seed=6001` 4-scenario slice) achieves `17` tasks vs Best Deterministic's `19` tasks (**`-10.53%`** vs Best Det; **`+0.00%`** vs `U1b`).

The literature across **four directly verified research fronts** confirms why that empirical outcome occurred — and clearly maps what is **already occupied** versus what genuine open problems remain.

---

## 2. Comparison Across Research Clusters

| Method / Paper | Core Idea | Novelty Overlap with Proposed `EA-STP` | Direct Empirical / Conceptual Link to Our Tasks 2–3 |
| :--- | :--- | :--- | :--- |
| **CPS-A\*** (Guo, Huang, Jiang, Xin & D'Ariano, preprint) | Converts exact spatiotemporal reservations into Gaussian-smoothed congestion field + bounded rollback | Direct upstream substrate (`U0`); Sec. 6 Future Work flags unmodeled vehicle dynamics, comm delays & service uncertainty | Task 2 quantified `CPS-A*` -> `R1_Robust` mismatch (`-17.25s` DEV mean, `94.4%` early, plus `arr_i-1` multi-tick span leak) |
| **GGO** (`IJCAI 2024`, Zhang et al.) & **Online GGO** (`AAAI 2025`, Zang et al.) | Offline CMA-ES / online RL edge-weight maps for congestion-aware LMAPF guidance | **High overlap** on learning routing/edge-cost guidance over planners | Shows learning graph weights/guidance alone is heavily saturated in LMAPF |
| **LaCAM\*+LTM** (`IJCAI 2026`, Shen, Zhang, Chen & Harabor) | Dynamic lightweight traffic map built inside anytime LaCAM\* without pre-search overhead | **High overlap** on dynamic traffic weighting inside combinatorial search | Non-RL dynamic congestion map already removes slow static weight computation |
| **SJRL** (`arXiv:2608.05588v1`, Jiang et al., 2026) | Joint RL agent + environment edge-cost learning with Causal PIBT under `LMAPF-R2` (turns + robust safety) | **Severe overlap** on combining RL congestion/edge weights with deterministic conflict resolution | Directly warns against wrapping edge-weight RL around a conflict-resolving lower layer |
| **GRAND** (`IEEE RA-L 2026`, Gaber, Alharbi, Gammelli & Zardini) | GNN-RL macro agent-distribution guidance + min-cost flow + local assignment (`<1s`, 500 agents, `+10%`) | Medium (dispatch/rebalancing level rather than path-level buffer selection) | Indicates RL guidance pays off at **fleet rebalancing/dispatch**, not single-leg route tweaks |
| **MAWPF** (`IJCAI 2026`, Nagai & Okumura, `arXiv:2605.15799v1`) | Straight/in-place rotation, accel/decel & rear-end constraints on discrete MAPF solvers | **High overlap** on kinodynamic planner cost alignment (`U1b`) | Validates adapting MAPF to AGV turn/accel physics at solver level |
| **db-LaCAM** (`ICAPS 2026`, Moldagalieva, Okumura, Prorok & Hönig, `arXiv:2512.06796v3`) | Precomputed unicycle/3D primitives + discontinuity-bounded LaCAM (50 robots, 10x speedup) | **Direct overlap** on motion-primitive + MAPF decomposition | Non-RL joint kinodynamic search handles unicycle constraints directly |
| **PIBT\*\_T** (`arXiv:2611.23911v1`, Tao, Yu & Hu, Nov 2026) + **SMART** (`IEEE RA-L 2026`, Yan et al.) | Turn-aware iterative PIBT validated in ADG & SMART physics simulator (`44x48m` FMS: `-23.3%` / `-20.2%` makespan) | **Severe overlap** with `U1b`: explicitly measures planning-vs-execution makespan gap caused by unmodeled turns | Mirrors our `MapA` `U1b` finding (`|dETA| 17.56s->0.88s`, `comp +36.4%`), and Sec. 6 flags dynamic tasks + uncertain execution replanning |
| **TAKT** (`arXiv:2611.29548v1`, Li, Zhang & Zardini, Nov 2026) | Pre-search analytical smoothing clearance bounds + closed-form convolution margins (1,000 robots) | **High overlap** on pre-reserving kinodynamic execution margins | Analytical closed-form kinematic margin leaves little room for heuristic fixed buffers |
| **Lazy-CPBS** (`arXiv:2611.30874v1`, Signorile, Carli, Bertogli, Olmi, Koenig & Dotoli, Nov 2026) | Continuous-time `AGV-MAPF_R-CTC` + SIPP boundary conditions + lazy priority search on industrial roadmaps | Orthogonal/stronger substrate than discrete-tick A\* | Demonstrates SIPP continuous safe intervals handle heterogeneous edge speeds cleanly |
| **The Cost of Executability** (`arXiv:2610.20118v1`, Roselli, Combrink & Zhang, Oct 2026) | AOC-CBS + MPC quantifying `T`-robustness time margins vs spatial inflation (`maze-32-32-2` & `empty-16-16`) | **Fatal overlap on uniform temporal buffers (`U2`)**; *supports selective slack + online reordering as open* | Directly explains our `MapA` (`U2_1.0s` `+63.6%`), `MapB` (`-17.1%` redundant), & `Occluded_Jct` (`-50%` timeout/lockout knee) |
| **WinkTPG** (`IEEE T-ASE 2026`, Yan, Smith & Li, `arXiv:2508.01495v2`) | Rolling-window kinodynamic speed-profile optimization over TPG with bounded/Gaussian uncertainty margins | **Severe overlap** on rolling-horizon timing + uncertainty margins (non-RL LP/STP) | Leaves heavy-tailed/correlated resource delays & decentralized execution open |
| **PPSS** (`IEEE RA-L 2026`, Zhang, Piao & Piao, `10.1109/lra.2026.3732901`) | Searches spatial routes + valid passing-order insertions over AAMUs **without strict time-step reservations** | **Direct critique of `U0..U3` timestamp reservation architecture** | Explains why strict timestamps break under real deviations (`U1b..U3` `eta_drift = 200–240`) |
| **BTPG / IGSES / BTPG-max** (`AAAI 2024/2025`, Su et al., Jiang et al.) | Switchable/bidirectional passing-order relaxation on TPG under delays (`8–20%` less wait) | Closed prior `STCP-RL` passing-order niche via exact graph optimization | Confirms deterministic/graph switching absorbs local order adaptation |
| **Should I Replan?** (`IEEE ICRA 2026`, Zahrádka et al., `arXiv:2604.25567v1`) | FFN on ADG delay/slack features predicts single-replan SoC savings (`94.6%` savings recovery across 12k trials) | Distinct supervised replan-trigger layer (not PPO trajectory selector) | Proves blind/frequent replanning (`U1b..U3` `200–240` drifts) yields zero net gain & hurts outliers |
| **HARR** (`IROS 2026`, Zang et al., `arXiv:2607.17724v1`) | Shared dock calendar + rolling-horizon buffer occupancy projection for `MS-MAPD-BHS` (`+77%` thru, `-92%` backlog) | Explains our `Workshop_8R_Clean` dead-end bay deadlock (`0.33/12` comp across `U0..U4`) | Shows bay/buffer blocking cannot be fixed by route timing buffers alone |

---

## 3. Insights from Local Workspace Literature (`verified archive references`)

Reviewing the local project literature archive strengthens two methodological boundaries already indicated in internal prior audits (`researchability07–10`):
1. **Online MAPD & Temporal Coupling**: `TP/TPTS` (Ma et al., 2017), `D-TP/D-TPTS` (Makino & Ito, 2024, `arXiv:2403.12377v3`), `SCoBA` (Choudhury et al., 2022, `DOI:10.1007/s10514-021-10022-9`), `BCP-MAPD` (`DOI:10.1287/trsc.2023.0268`), `Foresee` (Cao et al., 2024), and `MAPDP` (Di Pietro et al., AAMAS 2023) demonstrate that when shared docks, dead-end bays, or task releases drive congestion, **order-to-bay dispatch and bay occupancy projection** (as formalized in `HARR`) dominate downstream path-level timing tweaks. This explains why `Workshop_8R_Clean` stays pinned at `0.333–0.667 / 12` under all upper planners `U0..U4`.
2. **Adaptive Uncertainty Handling**: `ARO` (Cohen, Postek & Shtern, `arXiv:2102.08677v3`) and dynamic pickup-and-delivery surveys (`DOI:10.1016/j.neucom.2023.126631`, `DOI:10.1109/lra.2025.3642658`) emphasize that reactive re-optimization only pays off when recourse decisions actually alter structural bottlenecks rather than shifting short-horizon timestamps against frozen peer schedules.

---

## 4. Novelty Verdict on Candidate Directions

1. **Candidate Rejected — RL Selection Over `(route_weight, temporal_buffer)` Pool (`EA-STP-RL`)**:
   - **No Novelty Gap**: Weight adaptation is covered by `GGO`, `Online GGO`, `LaCAM*+LTM`, and `SJRL`; kinodynamic timing alignment by `PIBT*_T`, `MAWPF`, and `TAKT`; temporal buffers by `Cost of Executability` and `WinkTPG`.
   - **No Empirical Headroom**: `U4_FinitePool_CF_Oracle` scores **`-10.53%`** vs Best Deterministic (`17` vs `19` tasks on matched `seed=6001` slice).
2. **What *IS* Genuinely Supported by Both Literature and Data (Non-RL / Targeted Alternatives)**:
   - **Baseline Engineering Retention (`U1a_EdgeSpan_Fix + R1_Robust`)**: Fixing the multi-tick edge-span reservation defect yields `18/18` (`100%`) DEV ZeroCol and `+6.00%` completion over `U0` cleanly with `0.376 ms` latency.
   - **Selective / Topology-Aware Replanning & Sequence-Based Coordination (Open Literature Niche, NOT PPO Control)**: Combining `Should I Replan?` (`arXiv:2604.25567v1`, trigger replanning only when predicted slack recovery exceeds cost) with order-based passing sequences (`PPSS`/`BTPG`) or selective contested-node slack (`Cost of Executability`, Sec. 5) rather than brittle strict-timestamp re-reservation — plus `HARR`-style bay staging specifically for dead-end layouts like `Workshop_8R`.

---

## 5. Complete Reference URLs Retrieved & Inspected

- CPS-A\*: Guo et al. — `CPS-A* reference manuscript` (`https://github.com/pengguo-hub/CPSAStar-TRC`)
- The Cost of Executability (Roselli, Combrink & Zhang, 2026) — https://arxiv.org/pdf/2610.20118
- WinkTPG (Yan, Smith & Li, IEEE T-ASE 2026) — https://doi.org/10.1109/tase.2026.3688563 | https://arxiv.org/pdf/2508.01495
- PIBT\*\_T (Tao, Yu & Hu, 2026) — https://arxiv.org/pdf/2611.23911
- SMART (Yan et al., IEEE RA-L 2026) — https://doi.org/10.1109/LRA.2026.3688062 | https://arxiv.org/pdf/2503.04798v3
- Should I Replan? (Zahrádka et al., IEEE ICRA 2026) — https://doi.org/10.1109/icra57385.2026.11695725 | https://arxiv.org/pdf/2604.25567v1
- PPSS (Zhang, Piao & Piao, IEEE RA-L 2026) — https://doi.org/10.1109/lra.2026.3732901 (OpenAlex `W7212160644`)
- GRAND (Gaber, Alharbi, Gammelli & Zardini, IEEE RA-L 2026) — https://doi.org/10.1109/lra.2026.3673902 (OpenAlex `W4417026743`)
- LaCAM\*+LTM (Shen, Zhang, Chen & Harabor, IJCAI 2026) — https://doi.org/10.24963/ijcai.2026/38 (OpenAlex `W7134824227`)
- TAKT (Li, Zhang & Zardini, 2026) — https://arxiv.org/pdf/2611.29548v1
- Lazy-CPBS (Signorile et al., 2026) — https://arxiv.org/pdf/2611.30874v1
- MAWPF (Nagai & Okumura, IJCAI 2026) — https://doi.org/10.24963/ijcai.2026/35 | https://arxiv.org/pdf/2605.15799v1
- db-LaCAM (Moldagalieva, Okumura, Prorok & Hönig, ICAPS 2026) — https://arxiv.org/pdf/2512.06796v3
- SJRL (Jiang et al., 2026) — https://arxiv.org/pdf/2608.05588v1
- BTPG / BTPG-max / IGSES (Su, Veerapaneni & Li; Jiang, Lin & Li) — https://arxiv.org/pdf/2401.00315v2 | https://arxiv.org/pdf/2508.04849v1 | https://arxiv.org/pdf/2412.15908v2
- HARR (Zang et al., IROS 2026) — https://arxiv.org/pdf/2607.17724v1
- GGO / Online GGO (Zhang et al., IJCAI 2024; Zang et al., AAAI 2025) — https://doi.org/10.24963/ijcai.2024/35 | https://arxiv.org/pdf/2402.01446v2 | https://arxiv.org/pdf/2411.16506v2
- Local Corpus (`verified archive references`): SCoBA (`10.1007/s10514-021-10022-9`), BCP-MAPD (`10.1287/trsc.2023.0268`), TP/TPTS (`tpts2017_author.pdf`), D-TP/D-TPTS (`arXiv:2403.12377v3`), Foresee (`foresee_arxiv2024.pdf`), MAPDP (`mapdp_aamas2023.pdf`), ARO (`arXiv:2102.08677v3`), Transferable Task Assignment (`10.1109/lra.2025.3642658`), Dynamic Pickup Survey (`10.1016/j.neucom.2023.126631`).
