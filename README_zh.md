# AMR-Scheduler（`CoRe-Sched`）：面向动态生产—多 AMR 系统的学习引导与承诺保持增量联合重调度平台

<div align="center">

[![CI](https://github.com/AmaneSuzuha000/AMR-Scheduler/actions/workflows/ci.yml/badge.svg)](https://github.com/AmaneSuzuha000/AMR-Scheduler/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Rust 2021](https://img.shields.io/badge/Rust-Core-DEA584?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![ROS 2](https://img.shields.io/badge/ROS_2-Humble-22314E?logo=ros&logoColor=white)](https://docs.ros.org/en/humble/)
[![Gazebo](https://img.shields.io/badge/Gazebo-Classic%20%2F%20Fortress-FF6F00?logo=gazebo&logoColor=white)](https://gazebosim.org/)
[![.NET 6.0](https://img.shields.io/badge/.NET-6.0%20WPF-512BD4?logo=dotnet&logoColor=white)](https://dotnet.microsoft.com/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

[English](README.md) | **简体中文**

</div>

---

## 项目概述

**AMR-Scheduler**（`v1.4.0`）是面向多移动机器人系统的**长期共享科研基础设施与数字孪生实验平台**，在统一的场景拓扑、时空规划核、连续物理执行器与独立审计体系之上支撑两条互补研究主线：

- **Track A · 底层时空规划与连续物理避障（`Layer 2–3`，当前研究重心）**：
  - **Layer 3 · 多机器人交通协调**：拥堵感知 `CPS-A*`、执行延误局部修复 `EALR`、多子步边跨度修正规划（`U1a_EdgeSpan_Fix`）与通行权协调；
  - **Layer 2 · 局部物理运动规划**：确定性预约余量基元执行器 **`R1_Robust`**、`K=6` 运动基元、双圆扫掠胶囊车体与高斯不确定性制动护盾；
  - **Layer 1 · 物理执行与感知**：ROS 2 / Nav2 / Gazebo 仿真孪生与 SCOUT Mini 线控底盘。
- **Track B · 上层工业生产—多 AMR 联合重调度（`Layer 4–5`，已验证主线基线，当前暂时冻结）**：
  - **Layer 5 · 生产资源联合优化**：工序、工位、班组与物料运输联合优化（`V18`、`V19`、`V20_1`、`RGRO`）；
  - **Layer 4 · 动态运输任务调度**：承诺保持增量评估（`CVCR-IR`）、异构图构造器（`AHG-RL`）及派车分配器。

> **架构重定位与双保真度演进路线**：早期工作（`v1.0–v1.3`）重点在简化运动假设上探索 Layer 4–5 上层智能调度；当前阶段暂时冻结上层主线并下沉深化 **Layer 2–3** 物理执行底座，用连续运动学、扫掠几何、物理让行与更精准的运输时间（ETA）建模支撑上层可信决策。各消融与证伪研究沉淀了连续仿真器、安全护盾、反事实快照评估器及确定性 `U1a + R1_Robust` 执行栈等持久工程资产；未来规划通过**双保真度执行接口（Multi-Fidelity Interface）**衔接快速离散训练与高保真 `CPS-A* + R1_Robust` 校验（跨保真度集成与上层调度器配对重测尚属后续规划工作）。

<p align="center">
  <img src="results/figures/fig1_core_sched_overview.png" alt="CoRe-Sched 双引擎闭环在线重调度架构" width="96%" />
</p>

### 核心双引擎与执行架构
- **自适应异构图学习与因果修复引擎（`AHG-RL` + `CausalRepairCone`）**（`amr_sched/rl/v19_hetero_graph.py`, `v19_hetero_constructor.py`, `causal_repair_cone.py`, `v10_selector.py`）：
  - **5 类实体异构图注意力编码器与纯 NumPy 提前提升**：编码 `Operation`、`Task`、`AMR`、`Bay`、`Crew` 节点及工序前驱 DAG、齐套二分连接与通道拥堵绕行边特征，初始化阶段即提升至与 PyTorch 误差 `< 1e-5` 的纯 NumPy 推理矩阵，根除首步冷启动延迟抖动；
  - **四阶段自回归语义构造器（`BC -> KL 锚定 SMDP-PPO`）**：依次解码*任务优先级排序*、*工位/班组资源分配*、*车队动态负载分派*与*单车队列定序*；
  - **在线上下文切换 + 双臂探测 + 二阶段 `CausalRepairCone` + 对称振荡仲裁**：耦合 `Contextual LinUCB`（$\phi(s_t) \in [0, 1]^6$）、车队守卫反事实双臂探测（$|\mathcal{R}| \ge 4$）、精确 $J_A$ 评分器、基于 `FastJAWKittingSurrogate` 的 Regret-2 局部修复锥、活跃装货（`_load_end`）/有效边安全守卫与对称 `p4_choke_robust` 外科走廊振荡唤醒。
- **承诺保持增量预约优化引擎（`CVCR-IR`）**（`amr_sched/cvcr/evaluator.py`, `crates/`）：
  - **序列差分增量预约缓存（`CVCR-IR (Seq-Diff)`）+ 紧缩 SLA 守卫**：按单车精确有效任务序列差分定位 $\Delta\mathcal{R}$，原位复用未变车辆的 Rust 半开区间预约与无冲突路径（`median |ΔJ| = 0.00`），将双队列盲测的汇总尾延迟 `Pooled P99` 严格控制在 `23 ms` 以内（`18.55 ms / 22.62 ms < 35 ms`）。
- **原生 Rust 端到端时空规划内核、鲁棒胶囊护盾与执行感知时空规划（`v1.3.0–v1.4.0`）**（`crates/`, `amr_sched/planners/rust_spacetime_astar.py`, `experiments/{track_r_cpsa, cps_astar_routing, res_guided_avoidance, stp_research_reset}/`）：
  - 原位 `SlotCongestionField`、`plan_cps_indexed`、惰性 `RustBatchSippPlanner` 与零影子 `RustReservationTable.fast()` 在 `80–200` 车 MovingAI `Warehouse_40x40` 仓库测试中取得 **`11.71×–14.40×`** 端到端规划加速（`crates/` `26/26` 单元测试全过）；
  - 分层交通协调栈融合 *EJOR 2026* 上层派车（Wu et al.）、纯净复现的 `CPS-A*` 拥堵感知路由（Guo & Huang，[`pengguo-hub/CPSAStar-TRC`](https://github.com/pengguo-hub/CPSAStar-TRC)）、积压感知 DFL 阻塞者选择、执行延误局部修复（`EALR v2/v3`）、双圆胶囊不确定性感知安全校验（`R1_Robust`，36 局不相交盲测 `100.0%` 零碰撞）及多子步边跨度修正规划（`U1a_EdgeSpan_Fix`）。
- **原子事务提交执行层（*Atomic Transaction Commit*）与双层数字孪生**：
  - 统一注册入口（共 **24** 个已注册策略）：涵盖 `core_sched` / `CoRe-Sched`（`V19_ADAPTIVE_RL`）、`V20_NAJS`、`V20_1_NAJS`（`PDG 2.0`）、2025–2026 直接联合动态 FJSP–AGV 神经与元启发式基线（`IJPR25_DFJSPT_HMARL`、`ASC26_MA_TRANS`、`FGCS26_REN_GNN`、`SPGA_AMR`）、静态离线参照（`COR26-Ref` CP-SAT 主松弛 + 路径验证离线参考求解器、`EJOR26-Erdogan`）及更广泛的 2025–2026 顶会/顶刊基线。

---

## 核心实证亮点（`v1.2.1` NCSR 优化与双队列 240 局盲测闭环）

### 1. 双 120 局隔离盲测队列（共 `240` 局 / `2,880` 局独立审计运行，100% 审计通过）

在严格零重叠种子隔离下连续执行 **Holdout Cohort A**（[`results/g3_fresh_blind_holdout.csv`](results/g3_fresh_blind_holdout.csv)，`[27101..27302]` & `[2901..2906]`）与 **单次零返工 Blind Cohort B**（[`results/g5_pristine_blind_holdout.csv`](results/g5_pristine_blind_holdout.csv)，`[37101..37302]` & `[3901..3906]`，冻结后仅运行一次）。标 `*` 方法突破 `35 ms` Pooled P95/P99 实时红线。

| 方法 | **240 局完工率** ↑ | Cohort A 均值 $J_A$ ↓ | Cohort A P95 / P99 (ms) | Cohort B 均值 / **中位数 $J_A$** ↓ | Cohort B P95 / P99 (ms) | **240 局配对 $\Delta J_A$ (`p`, `胜/平/负`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CoRe-Sched` (`core_sched`)** | **238/240 (99.2% 第 1)** | **5,177.9 (第 1)** | **15.52 / 18.55** | 6,631.5 / **827.5 (第 1)** | **19.45 / 22.62** | 基准（均值 `5,904.7`，中位 `1,575.7`） |
| `V18-EA`* | 234/240 (97.5%) | 5,654.1 | 27.53 / *79.38* | **5,387.0** / 991.7 | 25.49 / *77.74* | -6.96% (`p = 0.0666`, `99/0/141`) |
| `IJCAI26-PGMPO` | 237/240 (98.8%) | 6,016.5 | 23.13 / *38.81* | 5,532.7 / 843.5 | 22.97 / 34.49 | -2.25% (`p = 0.901`, `116/0/124`) |
| `Rule-V10`* | 234/240 (97.5%) | 6,674.2 | *37.16* / *52.28* | 6,175.6 / 1,455.2 | *35.90* / *51.30* | +8.10% (`p = 0.092`, `127/0/113`) |
| `AHG-RL (Full)` | 235/240 (97.9%) | 6,019.2 | 32.84 / *52.34* | 6,942.6 / 3,610.4 | 33.14 / *42.61* | **+8.89%** (**`p = 0.00135`**, `144/0/96`) |
| `AHG-RL (Staged)`* | 234/240 (97.5%) | 6,469.2 | 30.31 / *54.26* | 6,821.4 / 3,613.7 | *35.59* / *78.51* | **+11.15%** (**`p = 7.25e-5`**, `151/0/89`) |
| `FGCS26-Ren-GNN` | 235/240 (97.9%) | 7,420.4 | 12.68 / 16.05 | 6,240.2 / 3,695.2 | 13.56 / 16.81 | **+13.55%** (**`p = 1.62e-4`**, `148/0/92`) |
| `ASC26-MA-Trans` | 231/240 (96.2%) | 7,776.6 | **12.22 / 16.82** | 6,580.0 / 3,811.9 | **12.01 / 15.30** | **+17.74%** (**`p = 7.35e-4`**, `142/0/98`) |
| `RAL26-DetourDelta` | 234/240 (97.5%) | 6,998.2 | 22.10 / *57.42* | 7,486.5 / 3,683.4 | 27.37 / *61.78* | **+18.47%** (**`p = 1.32e-10`**, `160/0/80`) |
| `COR26-SPGA`* | 234/240 (97.5%) | 7,533.0 | *62.55* / *75.39* | 7,240.1 / 2,920.4 | *62.02* / *73.66* | **+20.06%** (**`p = 4.09e-7`**, `164/0/76`) |
| `EJOR26-Wu-GRU` | 228/240 (95.0%) | 8,857.7 | 33.04 / *52.49* | 7,718.7 / 1,057.2 | 33.88 / *50.04* | **+28.76%** (**`p = 4.37e-4`**, `140/0/100`) |
| `IJPR25-DFJSP-T` | 228/240 (95.0%) | 7,817.4 | 14.85 / 18.05 | 9,253.9 / 3,760.6 | 14.70 / 18.35 | **+30.82%** (**`p = 8.47e-14`**, `166/0/74`) |

- **全组第 1 完工率与双队列 Pooled P99 `< 23 ms`**：`CoRe-Sched` 取得 **`238/240（99.2%）`** 总完工率第 1，包揽 **Cohort A 平均第 1（`5,177.9`, `120/120`, `15.52 / 18.55 ms`）** 与 **Blind Cohort B 中位数第 1（`827.5`, `118/120`, `19.45 / 22.62 ms`）**，跨 240 局显著战胜 **8 / 11 个对比方法（$p \le 0.00135$）**，并透明披露 Cohort B 两局未完工长尾算例（`topo4_offset_dual_lane_window_bottleneck_37302 = 78,259.95`、`track2_kitting_windows_3906 = 117,637.65`）对均值的影响。
- **72 局单因子消融与 `2 × 2` 触发器 × 求解器因子解耦**（[`results/g4_val72_repair_ablation.csv`](results/g4_val72_repair_ablation.csv)）：剥离 `CausalRepairCone`（`+14.32%` $p=4.30\times 10^{-5}$）、旧评分器（`+16.21%` $p=3.02\times 10^{-4}$）、`AHG-RL` 先验（`+28.60%`）或 ZeroChurn（`+4.60%`）均稳定退化；`RuleV10 + CoRe-Sched`（`4,190.3`, `72/72`）优于 `V18-EA`（`4,252.9`）、`RuleV10 + V18-EA`（`4,359.9`）与 `V18Ctrl + CoRe-Sched`（`4,864.7`, `71/72`）。
- **54 局纯零故障静态公共子问题同仿真器参照（`0 / 270` 下界违例）**（[`results/g1_static_reference_subproblem.csv`](results/g1_static_reference_subproblem.csv)）：`COR26-Ref`（CP-SAT 主松弛 + 无冲突路径验证离线参考求解器，`LB_phys = 68.2 / 90.2 / 438.7`，`Sim-UB = 412.4 / 346.8 / 793.0`，宣称全局最优 `0`）与 `EJOR26-Erdogan`（`352.1 / 664.8 / 664.5`）严格满足 `LB_phys <= H1_sim`，`CoRe-Sched` 静态 Pooled P95/P99 为 `13.61 / 20.47 ms`。
- **`CVCR-IR (Seq-Diff)` 缓存协同、物理对账与 10 车扩展**：隔离缓存命中率从 `0.00%` 恢复至 `73.68%`（闭环 `86.8%–94.7%`，`median |ΔJ| = 0.00`）；5 种子 ROS 2 / Nav2 / Gazebo 物理对账 Track 1 $\rho = 0.7857$（物理 P95 `7.18 ms`）；62 节点扩至 10 车时 P95 仅 **`23.57 ms < 35 ms`**（提速 `2.21×`）。

<p align="center">
  <img src="results/figures/fig7_dual_cohort_and_ablation.png" alt="双 120 局盲测一致性、72 局消融与 2x2 触发器解耦" width="96%" />
</p>

<p align="center">
  <img src="results/figures/fig3_main_benchmark.png" alt="CoRe-Sched Cohort A 盲测成本分解与时延—质量帕累托前沿" width="96%" />
</p>

<p align="center">
  <img src="results/figures/fig6_pure_h1_and_lbbd.png" alt="双队列纯物理效率 H1 归因与静态同仿真器参照下界" width="96%" />
</p>

<p align="center">
  <img src="results/figures/gazebo_overview.png" alt="1:25 Gazebo 轮对检修车间数字孪生" width="48%" />
  <img src="results/figures/layout.png" alt="车间拓扑与工位布局" width="48%" />
</p>

### 2. 连续物理执行安全与上层时空规划实证闭环（`v1.4.0`）

- **双圆胶囊不确定性安全护盾（`R1_Robust`）**（[`results/ramp_robust/robust_ramp_report.md`](results/ramp_robust/robust_ramp_report.md)）：修正单圆几何低估（`0.10–0.20 m`）并引入高斯视界安全裕度（$\epsilon(t) = 1.96\sqrt{\sigma_p^2 + (\min(t, 1.0)\,\sigma_v)^2}$）后，清零全部 `12,247` 次盲测车间碰撞事件（`36` 局盲测 `7001–7006` 零碰撞率由 `58.3% → 100.0%`），确定性 **`R1_Robust`**（`4.69` 完工）稳定优于冻结（`-5.33%`）与重训（`-8.86%`）`RAMP-PPO`。
- **反事实决策空间上限审计**（[`results/decision_headroom/decision_headroom_report.md`](results/decision_headroom/decision_headroom_report.md)）：基于 `N = 60` `clone_snapshot()` 多时域回滚（全队伍 `clear_15s` 上限 `<= +0.85%`）与 `N = 36` 冲突簇穷举通行排列（相对 `R1_Default_Order` `5.667` 完工仅 `+0.43%`，`100.0%` ZeroCol），证实局部运动基元层与冲突定序层在安全约束下已接近确定性上限。
- **执行感知上层规划审计（`U1a_EdgeSpan_Fix`）**（[`results/stp_research_reset/upper_level_headroom_report.md`](results/stp_research_reset/upper_level_headroom_report.md)）：量化 `CPS-A*` → `R1_Robust` 结构性早到偏差（`N = 252` 条路段均值 `-17.25 s`，`94.4%` 早到）；修复多子步边跨度预约（`U1a_EdgeSpan_Fix`）即以 `0.376 ms` 代价使 DEV 零碰撞升至 **`18/18 (100.0%)`**、完工提升 **`+6.00%`**（`5.889` vs `5.556`），而有限候选池反事实选择器（`U4`）相对最优确定性为 `-10.53%`（`17` vs `19`）。

---

## 快速开始

```bash
git clone https://github.com/AmaneSuzuha000/AMR-Scheduler.git
cd AMR-Scheduler

# 1. 编译 Rust 原生时空预约与规划内核
cargo build --release --manifest-path crates/Cargo.toml

# 2. 安装 Python 依赖
pip install -r requirements.txt
pip install -e ".[dev,viz,web,rl]"

# 3. 运行车间仿真、NCSR 消融、双队列盲测与 7 张插图重绘
python -m amr_sched --tasks 50 --agvs 6 --layout workshop --planner rust_spacetime_astar
python -B experiments/run_post_v120_audit_and_ablation.py
python -B experiments/run_g5_pristine_blind_holdout.py
python -B experiments/run_dual_layer_reconciliation.py --multi-seed
python -B experiments/run_workshop_scaling_4axis.py
python -B experiments/plot_v19_paper_figures.py

# 4. 运行全套验证测试
python -m pytest -q
cargo test --manifest-path crates/Cargo.toml
dotnet test Tests/RAWSimO.Live.Tests/RAWSimO.Live.Tests.csproj -c Release
python tools/data_manifest.py --check
```

---

## 许可证

本项目原创代码（`amr_sched/`、`crates/`、`RAWSimO.Live/`、`experiments/`、`tools/`、`Tests/`、`web/`）基于 **GPL-3.0** 许可证开源；上游 RAWSim-O 组件保留其原始许可证。详见 [LICENSE](LICENSE) 与 [ThirdPartyLicenses.txt](ThirdPartyLicenses.txt)。
