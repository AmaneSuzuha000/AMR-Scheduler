"""生成系统总体架构示意图。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from amr_sched.viz.style import setup

C_BG = "#ffffff"
C_MOD = "#dce9f7"
C_EDGE = "#5b7fa6"
C_TEXT = "#1f2d3d"
C_HL = "#fde8d7"
C_HL_EDGE = "#c0692b"
C_DEC = "#eaf3ea"
C_DEC_EDGE = "#4a8a5a"
C_DB = "#f0eef7"
C_DB_EDGE = "#6a5a9a"


def box(ax, x, y, w, h, text, fc=C_MOD, ec=C_EDGE, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        linewidth=1.2, facecolor=fc, edgecolor=ec, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=C_TEXT, zorder=3, linespacing=1.6,
            fontweight=("bold" if bold else "normal"))


def arrow(ax, x1, y1, x2, y2, color=C_EDGE, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
        linewidth=1.0, color=color, zorder=1,
        connectionstyle="arc3,rad=%.2f" % rad))


def band(ax, y0, y1, label):
    """左侧竖排层标签，放在图外，避免遮挡模块。"""
    ax.text(-0.035, (y0 + y1) / 2, label, ha="center", va="center",
            fontsize=9, color="#5a6b7d", fontweight="bold",
            rotation=90, zorder=5)
    ax.plot([-0.012, -0.012], [y0, y1], color="#c8d4e0",
            linewidth=0.8, zorder=0)


def main() -> int:
    setup()
    fig, ax = plt.subplots(figsize=(12.2, 7.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.06)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)

    ax.text(0.5, 1.025, "多 AGV 物料配送调度系统 — 总体架构",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color=C_TEXT)

    # ===== 表示层 y=0.885 =====
    band(ax, 0.845, 0.925, "表示层")
    viz = [
        (0.085, "网络拓扑图"), (0.245, "任务甘特图"),
        (0.405, "通道热力图"), (0.565, "轨迹图"),
        (0.725, "状态时间线"), (0.895, "过程动画"),
    ]
    for x, t in viz:
        box(ax, x, 0.885, 0.145, 0.062, t, fs=8.4)

    # ===== 应用层 y=0.700 =====
    band(ax, 0.600, 0.790, "应用层")
    mods = [
        (0.095, "① 任务管理\nentities.py\nscenarios.py", False),
        (0.275, "② 物流网络管理\nnetwork.py", False),
        (0.455, "③ 调度决策\nallocation/ · planners/\nreschedule.py", True),
        (0.635, "④ AGV 状态监控\nsimulation.py", False),
        (0.815, "⑤ 数据存储\nstorage.py (SQLite)", False),
        (0.955, "⑥ 可视化\nviz/", False),
    ]
    for x, t, hl in mods:
        box(ax, x, 0.700, 0.155, 0.115, t,
            fc=(C_HL if hl else C_MOD), ec=(C_HL_EDGE if hl else C_EDGE),
            fs=7.8, bold=hl)

    # ===== 决策层 y=0.445 =====
    band(ax, 0.300, 0.560, "决策层")
    dec = [
        (0.165, "任务分配\n\n规则：FIFO / EDD / SPT /\nPRIORITY / CR\n算法：greedy / nearest /\nhungarian"),
        (0.500, "路径规划\n\n静态 Dijkstra\n（忽略其他车辆）\n\n时空 A*\n（避开已预约时空点）"),
        (0.835, "冲突消解\n\n预约表（前馈）\n左闭右开整数区间\n\n+ 到达硬校验（反馈）\n容量违规即阻塞"),
    ]
    for x, t in dec:
        box(ax, x, 0.485, 0.285, 0.150, t,
            fc=C_DEC, ec=C_DEC_EDGE, fs=7.5)

    arrow(ax, 0.310, 0.485, 0.355, 0.485)
    arrow(ax, 0.645, 0.485, 0.690, 0.485)

    # 重调度条 y=0.335
    box(ax, 0.500, 0.330, 0.780, 0.066,
        "加急动态重调度   三档抢占粒度："
        " NONE  /  PREEMPT_UNPICKED  /  PREEMPT_ANY",
        fc="#fbeaea", ec="#a04a4a", fs=8.6, bold=True)
    arrow(ax, 0.500, 0.410, 0.500, 0.365, color="#a04a4a")

    # ===== 数据层 y=0.135 =====
    band(ax, 0.060, 0.215, "数据层")
    dbs = [
        (0.155, "配置与算例\nSimulationConfig\nScenarioConfig"),
        (0.405, "长表 CSV\n9 文件 / 2220 行\n供统计分析"),
        (0.655, "SQLite\nruns/tasks/agvs/steps\n供查询回溯"),
        (0.895, "图表与动画\n10 张 PNG + 1 GIF"),
    ]
    for x, t in dbs:
        box(ax, x, 0.135, 0.200, 0.105, t, fc=C_DB, ec=C_DB_EDGE, fs=7.6)

    # ===== 数据流 =====
    # 应用层 -> 决策层
    arrow(ax, 0.095, 0.642, 0.090, 0.562)
    arrow(ax, 0.275, 0.642, 0.235, 0.562)
    arrow(ax, 0.455, 0.642, 0.470, 0.556)
    arrow(ax, 0.635, 0.642, 0.780, 0.556)
    # 决策层 -> 表示层（走右侧空白）
    arrow(ax, 0.890, 0.556, 0.895, 0.854, color="#8a9aaa")
    # 决策层 -> 数据层
    arrow(ax, 0.300, 0.365, 0.330, 0.190, color=C_DB_EDGE)
    # 数据存储模块 <-> 数据层
    arrow(ax, 0.815, 0.642, 0.700, 0.190, color=C_DB_EDGE, rad=-0.12)

    # ===== 箭头说明 =====
    ax.text(0.012, 0.020,
            "注：橙色为调度决策核心模块。箭头表示主要数据流："
            "任务与网络→决策层；决策结果→表示层与数据层。"
            "展示各算法模块的输入输出关系。"
            , ha="left", va="center", fontsize=7.6, color="#5a6b7d")

    out = "results/figures/architecture.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print("已生成", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())