"""轨迹回放：把 AGV 的时空轨迹画在车间拓扑上。

只画轨迹在拓扑上的投影；时间用颜色深浅/箭头方向表达，
因为 2D 平面没有时间维，堆积过多线会看不清。
"""

from __future__ import annotations

import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from amr_sched.network import Network
from amr_sched.simulation import SimulationResult
from amr_sched.viz import style


def _position(network: Network, token: str) -> tuple[float, float]:
    """把轨迹里的位置标识换算成平面坐标。

    轨迹记录两种形式：
      * 节点标识 ``"R3"`` —— 直接取节点坐标；
      * 在途标识 ``"R3->R4"`` —— 取该边中点（2D 投影，不含沿途进度）。
    """
    if not token:
        return (0.0, 0.0)
    if "->" in token:
        src, dst = token.split("->", 1)
        try:
            a = network.node(src)
            b = network.node(dst)
        except KeyError:
            return (0.0, 0.0)
        return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)
    try:
        n = network.node(token)
    except KeyError:
        return (0.0, 0.0)
    return (n.x, n.y)


def _extract_paths(network: Network,
                   result: SimulationResult) -> dict[str, list[tuple[int, float, float]]]:
    """抽出每台 AGV 的 (step, x, y) 序列，去掉原地停留的重复点。"""
    paths: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for rec in result.trajectory:
        for agv_id, token in rec.agv_positions.items():
            x, y = _position(network, token)
            seq = paths[agv_id]
            if not seq or (seq[-1][1], seq[-1][2]) != (x, y):
                seq.append((rec.step, x, y))
    return paths


def plot_trajectories(network: Network,
                      result: SimulationResult,
                      out_path: str | None = None,
                      title: str = "AGV 配送轨迹（颜色 = 时间先后）"):
    """画所有 AGV 的轨迹，颜色由浅到深表示时间推进。"""
    style.setup()
    if not result.trajectory:
        raise ValueError(
            "轨迹为空：请用 SimulationConfig(record_trajectory=True) 运行仿真"
        )

    paths = _extract_paths(network, result)
    fig, ax = plt.subplots(figsize=(9.6, 5.6))

    # 底图：网络
    seen = set()
    for e in network.edges.values():
        key = (min(e.src, e.dst), max(e.src, e.dst))
        if key in seen:
            continue
        seen.add(key)
        a, b = network.node(e.src), network.node(e.dst)
        ax.plot([a.x, b.x], [a.y, b.y], color="#e6eaee",
                linewidth=1.4, zorder=1)

    for n in network.nodes.values():
        ax.scatter([n.x], [n.y], s=52,
                   color=style.NODE_COLORS.get(n.kind.value, "#888"),
                   zorder=2, edgecolors="white", linewidths=0.8)
        ax.annotate(n.node_id, (n.x, n.y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=6.5,
                    color="#555", zorder=2)

    vmax = max(1, result.finished_step)
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    for i, (agv_id, seq) in enumerate(sorted(paths.items())):
        if len(seq) < 2:
            continue
        xs = [p[1] for p in seq]
        ys = [p[2] for p in seq]
        ts = [p[0] for p in seq]
        offset = (i - (len(paths) - 1) / 2.0) * 0.9   # 避免多车轨迹完全重合
        ax.plot([x + offset for x in xs], [y + offset for y in ys],
                color=style.AGV_COLORS[i % len(style.AGV_COLORS)],
                linewidth=1.5, alpha=0.75, zorder=3, label=agv_id)
        ax.scatter([xs[0] + offset], [ys[0] + offset], s=60, marker="o",
                   facecolors="none",
                   edgecolors=style.AGV_COLORS[i % len(style.AGV_COLORS)],
                   linewidths=2.0, zorder=4)
        ax.scatter([xs[-1] + offset], [ys[-1] + offset], s=70, marker="s",
                   color=style.AGV_COLORS[i % len(style.AGV_COLORS)],
                   zorder=4)
        # 时间采样点：颜色 = 时间
        step = max(1, len(seq) // 28)
        for k in range(0, len(seq), step):
            ax.scatter([xs[k] + offset], [ys[k] + offset], s=16,
                       color=cmap(norm(ts[k])), zorder=5,
                       edgecolors="none")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.02)
    cb.set_label("仿真时间 (步)")

    ax.legend(loc="upper left", bbox_to_anchor=(1.06, 1.0),
              frameon=False, fontsize=8)
    ax.set_title("%s\nstart ○ / goal ■ (dashed); %d steps total"
                 % (title, result.finished_step), fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return fig
