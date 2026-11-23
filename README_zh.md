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

**AMR-Scheduler**（`v1.0.0`）是 **`CoRe-Sched`**（*Commitment-aware Rescheduling for Production–Multi-AMR Coordination*）框架的官方开源代码与数字孪生平台，面向**离散工位、多物料齐套汇合、狭窄单车道走廊**的铁路货车与轮对检修车间，实现毫秒级在线生产—多 AMR 联合重调度。

<p align="center">
  <img src="results/figures/fig1_core_sched_overview.png" alt="CoRe-Sched 双引擎闭环在线重调度架构" width="96%" />
</p>

### 核心双引擎与执行架构
- **自适应异构图学习引擎（`AHG-RL`）**（`amr_sched/rl/v19_hetero_graph.py`, `v19_hetero_constructor.py`）：
  - **5 类实体异构图注意力编码器**：编码 `Operation`、`Task`、`AMR`、`Bay`、`Crew` 节点及工序前驱 DAG、齐套二分连接与通道拥堵绕行边特征；
  - **四阶段自回归语义构造器（`BC -> KL 锚定 SMDP-PPO`）**：依次解码*任务优先级排序*、*工位/班组资源分配*、*车队动态负载分派*与*单车队列定序*，并导出与 PyTorch 误差 `< 1e-5` 的纯 NumPy `.npz` 权重，支持无 PyTorch 的 ROS 2 物理协调器直接运行；
  - **在线上下文策略选择（`Contextual LinUCB`）**：基于 6 维拓扑无关瞬时耦合状态 $\phi(s_t) \in [0, 1]^6$，在线自适应切换 `Full Joint` 与 `Staged` 策略。
- **承诺保持增量预约优化引擎（`CVCR-IR`）**（`amr_sched/cvcr/evaluator.py`, `crates/`）：
  - **序列差分增量预约缓存（`CVCR-IR (Seq-Diff)`）**：按单车精确有效任务序列差分定位变更车队子集 $\Delta\mathcal{R}$，原位复用未变车辆的 Rust 半开区间预约与无冲突路径（与全量重建 `median |ΔJ| = 0.00` 严格等价），并调用 Rust `SpaceTime A* / SIPP`（单车平均 `41 μs`）毫秒级评价全局齐套成本。
- **原子事务提交执行层（*Atomic Transaction Commit*）与双层数字孪生**：
  - 统一注册入口 `core_sched` / `CoRe-Sched`（`V19_ADAPTIVE_RL`），与 2025–2026 前沿顶会/顶刊基线（`IJCAI26_PGMPO`、`EJOR26_Wu_GRU_Attn`、`TII26_E2G_MoE`、`ICRA26_Kitting_HGNN`、`RAL26_DetourDelta`、`JAIR26_DyMAB`、`MIT26_TAPFPC_LNS`、`SPGA_AMR`、`BALANS_IJCAI25`）在同一因果快照、原子事务提交与 `IndependentAuditor` 独立轨迹审计契约下即插即用。

---

## 核心实证亮点（100% 真实墙钟同口径测试）

### 1. 120 场未见盲测主结果（100% 独立轨迹审计通过）

| 方法 | 100% 完总局 (`/120`) | 平均 $J_A$ ↓ | 平均 $C_{\max}$ ↓ | 拖期成本 ($2T$) ↓ | 未完工惩罚 ($H_2$) | `Track 1` (62 节点) ↓ | `Track 2` (50 节点) ↓ | 墙钟 P95 (ms) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CoRe-Sched` (`core_sched`)** | **120/120** | **5,425.53 (#1)** | **476.8 (#1)** | **4,568.8 (#1)** | **0.0 (#1)** | **4,622.9 (#1)** | **17,533.8 (#1)** | **12.51 (#1)** |
| `AHG-RL (Staged)` | 118/120 | 5,974.11 | 499.4 | 4,962.0 | 108.3 | 5,507.2 | 18,454.0 | 30.29 |
| `AHG-RL (Full)` | 118/120 | 6,001.36 | 479.4 | 4,842.3 | 275.0 | 5,085.6 | 18,834.0 | 32.17 |
| `Rule-V10 + CVCR-IR` | 117/120 | 6,034.34 | 497.9 | 5,033.1 | 250.0 | 6,805.2 | 21,009.8 | 57.36 |
| `V18-EA + CVCR-IR` | 114/120 | 6,221.95 | 484.9 | 5,001.6 | 550.0 | 6,292.4 | 23,039.6 | 26.15 |
| `IJCAI26-PGMPO` | 115/120 | 6,484.58 | 493.3 | 5,043.3 | 708.3 | 6,910.2 | 22,584.9 | 27.22 |
| `HG-BC (Pretrain)` | 115/120 | 7,407.27 | 506.2 | 5,220.3 | 1,266.7 | 6,039.0 | 24,934.2 | 32.20 |
| `EJOR26-Wu-GRU` | 116/120 | 7,988.12 | 506.2 | 5,977.0 | 1,158.3 | 7,420.1 | 29,387.8 | 34.71 |
| `RAL26-DetourDelta` | 115/120 | 8,048.61 | 561.5 | 6,515.6 | 750.0 | 7,462.8 | 24,755.9 | 32.66 |

- **配对 Wilcoxon 显著性**：`CoRe-Sched` 显著优于固定 `AHG-RL (Full)`（**`-9.59%`**, $p=0.0085$, `84W/36L`, 墙钟 P95 提速 **`2.57×`**）、固定 `AHG-RL (Staged)`（**`-9.18%`**, $p=0.0200$）、`HG-BC`（**`-26.75%`**, $p=1.25\times 10^{-4}$）及 `IJCAI26-PGMPO`（**`-16.33%`**, 95% CI 严格为负）。
- **`CVCR-IR (Seq-Diff)` 缓存协同**：将孤立全联合神经候选命中率从 `CVCR-IR (Base)` 的 **`0.00%` 恢复至 `73.68%`**（单候选提速 **`1.48×`**，`median |ΔJ| = 0.00`），闭环实测命中率达 **`86.8%–94.7%`**。
- **双层物理对账与 62 节点扩展**：5 种子统一目标物理对账 Track 1 Spearman $\rho = 0.7857$（`CoRe-Sched` 物理 P95 仅 **`7.18 ms`**），Track 2 物理闭环第 2 名（`4,220.0`，P95 **`11.44 ms < 35 ms`**）；62 节点扩至 10 辆 AMR 时墙钟 P95 仅 **`23.57 ms < 35 ms`**（较全量重建 `52.06 ms` 提速 **`2.21×`**）。

<p align="center">
  <img src="results/figures/fig3_main_benchmark.png" alt="CoRe-Sched 120 场盲测成本分解与时延—质量帕累托前沿" width="96%" />
</p>

<p align="center">
  <img src="results/figures/gazebo_overview.png" alt="1:25 Gazebo 轮对检修车间数字孪生" width="48%" />
  <img src="results/figures/layout.png" alt="车间拓扑与工位布局" width="48%" />
</p>

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

# 3. 运行车间仿真、CoRe-Sched 双引擎基准与 5 张插图重绘
python -m amr_sched --tasks 50 --agvs 6 --layout workshop --planner rust_spacetime_astar
python -B experiments/run_v19_exp08_learning_b2_synergy.py
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
