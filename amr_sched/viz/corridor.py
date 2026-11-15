"""通道负荷热力图（任务书 R5.3「通道负荷」的度量与可视化）。

定义
----
对每条**物理通道**（无向，``(u,v)`` 与 ``(v,u)`` 归一）统计：

    occupancy = Σ (某 AGV 占用该通道的步数) / (仿真总步数 × 通道容量)

即"这条通道有多满"。

**注意**：单条通道的取值**可以超过 1**，但这不代表容量违规——
物理通道是无向的，而网络里 A→B / B→A 是两条有向边、各算一次容量，
双向同时有车时分子累加 2 而分母只除 1。详见 :func:`corridor_load`。

本模块**不**做「冲突次数」的通道维度分解。
原因：`conflict_wait_events` 只在仿真器里按 AGV 计数，事件发生时
没有记录被拒的是哪条边，事后无法从轨迹反推归因。
若要该指标，需在仿真器里给被拒事件补一个 `edge` 字段——属未完成项，
不在此虚报。冲突等待的全网口径见 `SimulationResult.conflict_wait_count()`。
"""

from __future__ import annotations

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from amr_sched.network import Network
from amr_sched.simulation import SimulationResult
from amr_sched.viz import style


def corridor_load(network: Network, result: SimulationResult) -> dict[tuple[str, str], dict]:
    """统计每条物理通道的占用步数与容量。

    返回 ``{(u, v): {"occupied": int, "capacity": int, "load": float}}``，
    其中 ``(u, v)`` 已归一为 ``u < v`` 的顺序。

    这里**没有** ``conflicts`` 字段。早期版本 docstring 声称有，但代码从未实现——
    第 38 轮核查确认后删掉了这个不实说明。

    **两条口径警告（第 38 轮核查补充）**

    一、单条通道的 ``load`` **可能超过 1.0**。物理通道是无向的，
    但网络里 A→B 与 B→A 是两条有向边、各算一次容量。若某步两条
    方向各有一台车（一进一出），``occupied`` 累加 2，而分母只除一次
    容量，于是 ``load`` 到 2.0。**这不是容量违规**，是双向合并的
    计数结果。要看是否违规请查 ``collision_events`` 与节点容量校验。

    二、``load`` 的均值**不是「通道利用率」**。任意时刻全网最多只有
    ``n_agvs`` 条边被占（一台车只能占一条边），所以::

        mean_load <= n_agvs / sum(capacity)

    本布局 4 车、18 条通道（总容量 18 或 25），上限只有 22% 或 16%。
    把 13% 读成「通道很空」会得出与事实相反的结论——它其实是上限的
    52%~84%。换算请用 :func:`normalized_load`。
    """
    occupied: dict[tuple[str, str], int] = defaultdict(int)
    capacities: dict[tuple[str, str], int] = {}

    for e in network.edges.values():
        key = (e.src, e.dst) if e.src < e.dst else (e.dst, e.src)
        capacities[key] = e.capacity

    for rec in result.trajectory:
        for token in rec.agv_positions.values():
            if "->" not in token:
                continue
            a, b = token.split("->", 1)
            key = (a, b) if a < b else (b, a)
            occupied[key] += 1

    steps = max(1, result.finished_step)
    out: dict[tuple[str, str], dict] = {}
    for key, cap in capacities.items():
        occ = occupied.get(key, 0)
        out[key] = {
            "occupied": occ,
            "capacity": cap,
            "load": occ / (steps * max(1, cap)),
        }
    return out


def capacity_ceiling(network: Network, n_agvs: int) -> float:
    """``mean_corridor_load`` 的**物理可达上限**。

    这个函数存在的理由，是一个必须讲清楚的口径陷阱：

    ``load = 占用步数 / (总步数 × 通道容量)`` 这个式子里，分子受一条
    硬约束——**任意时刻全网最多只有 ``n_agvs`` 条边被占**（一台车只能
    在一条边上）。所以分母里那 18 条通道永远不可能被填满::

        sum(occupied) <= n_agvs × steps
        => 系统级负荷 <= n_agvs / sum(capacity)

    本布局 4 车、18 条通道总容量 18（多车道版 25），因此上限是
    ``4/18 ≈ 22%`` 或 ``4/25 = 16%``。**实测 12%~13.4% 已经是上限的
    52%~84%（均值 70%），属于高负荷，不是「远未饱和」。**

    把原始 ``load`` 直接读成「通道利用率」会得出完全相反的结论，
    这是第 38 轮核查发现并修正的一处论文级错误。
    """
    # 每条无向通道只算一次容量
    seen: set[tuple[str, str]] = set()
    total_cap = 0
    for e in network.edges.values():
        key = (e.src, e.dst) if e.src < e.dst else (e.dst, e.src)
        if key in seen:
            continue
        seen.add(key)
        total_cap += e.capacity
    if total_cap <= 0:
        return 1.0
    return n_agvs / total_cap


def normalized_load(network: Network, result: SimulationResult,
                    n_agvs: int) -> dict[str, float]:
    """把通道负荷换算成「占物理可达上限的百分比」。

    返回 ``{"raw": 原始均值, "ceiling": 上限, "ratio": 归一化值}``。
    论文中报告通道负荷时**必须**用 ``ratio`` 或同时给出 ``ceiling``，
    否则读者会把 13% 误读成「通道很空」。
    """
    loads = corridor_load(network, result)
    if not loads:
        return {"raw": 0.0, "ceiling": 1.0, "ratio": 0.0}
    raw = sum(v["load"] for v in loads.values()) / len(loads)
    ceil = capacity_ceiling(network, n_agvs)
    return {
        "raw": raw,
        "ceiling": ceil,
        "ratio": (raw / ceil) if ceil > 0 else 0.0,
    }

def plot_corridor_heatmap(network: Network,
                          result: SimulationResult,
                          out_path: str | None = None,
                          title: str = "通道负荷热力图"):
    """按负荷率给通道上色；同时用线宽表示容量。"""
    style.setup()
    load = corridor_load(network, result)
    if not load:
        raise ValueError("网络没有边")

    fig, ax = plt.subplots(figsize=(9.4, 5.4))

    vmax = max(v["load"] for v in load.values()) or 1.0
    cmap = plt.get_cmap("YlOrRd")
    norm = plt.Normalize(vmin=0.0, vmax=vmax)

    for (u, v), info in sorted(load.items()):
        try:
            a, b = network.node(u), network.node(v)
        except KeyError:
            continue
        ax.plot([a.x, b.x], [a.y, b.y],
                color=cmap(norm(info["load"])),
                linewidth=1.2 + 3.2 * info["load"],
                solid_capstyle="round", zorder=2, alpha=0.9)

    for n in network.nodes.values():
        ax.scatter([n.x], [n.y], s=140,
                   color=style.NODE_COLORS.get(n.kind.value, "#888"),
                   zorder=3, edgecolors="#333", linewidths=0.7)
        ax.annotate(n.node_id, (n.x, n.y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7, zorder=4)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("通道负荷率（占用步数 / (总步数 × 容量)）")

    # 标出最繁忙的 3 条通道
    top = sorted(load.items(), key=lambda kv: -kv[1]["load"])[:3]
    for rank, ((u, v), info) in enumerate(top, 1):
        try:
            a, b = network.node(u), network.node(v)
        except KeyError:
            continue
        ax.annotate("#%d %s-%s %.0f%%" % (rank, u, v, info["load"] * 100),
                    ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0),
                    fontsize=7, color="#8b0000", ha="center",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="#8b0000", lw=0.6, alpha=0.85),
                    zorder=6)

    mean_load = float(np.mean([v["load"] for v in load.values()]))
    ax.set_title("%s\n平均负荷 %.1f%% | 最繁忙通道 %.1f%% | 总步数 %d"
                 % (title, mean_load * 100, top[0][1]["load"] * 100,
                    result.finished_step), fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return fig
