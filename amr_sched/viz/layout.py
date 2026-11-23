"""Workshop layout: nodes, edges, and AGV start positions."""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from amr_sched.entities import Agv
from amr_sched.network import Network
from amr_sched.viz import style


def plot_layout(network: Network,
                agvs: Iterable[Agv] = (),
                out_path: str | None = None,
                title: str = "Maintenance workshop logistics network"):
    """Draw the workshop layout; returns the figure when ``out_path`` is None."""
    style.setup()
    fig, ax = plt.subplots(figsize=(9, 5.2))

    seen = set()
    for e in network.edges.values():
        key = (min(e.src, e.dst), max(e.src, e.dst))
        if key in seen:
            continue
        seen.add(key)
        a = network.node(e.src)
        b = network.node(e.dst)
        lw = 1.0 + 0.9 * (e.capacity - 1)
        ax.plot([a.x, b.x], [a.y, b.y], color="#cfd6dd",
                linewidth=lw, zorder=1, solid_capstyle="round")

    for n in network.nodes.values():
        ax.scatter([n.x], [n.y], s=210,
                   color=style.NODE_COLORS.get(n.kind.value, "#888"),
                   zorder=3, edgecolors="white", linewidths=1.2)
        label = n.node_id
        if n.capacity > 1:
            label += "(x%d)" % n.capacity
        ax.annotate(label, (n.x, n.y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=8, zorder=4)

    agvs = list(agvs)
    for i, agv in enumerate(agvs):
        p = network.node(agv.start_node)
        ax.scatter([p.x], [p.y], s=110, marker="^",
                   color=style.AGV_COLORS[i % len(style.AGV_COLORS)],
                   zorder=5, edgecolors="black", linewidths=0.6)
        ax.annotate(agv.agv_id, (p.x, p.y), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=7, zorder=5)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8,
               markerfacecolor=c, markeredgecolor="white",
               label=style.NODE_LABELS.get(k, k))
        for k, c in style.NODE_COLORS.items()
        if any(n.kind.value == k for n in network.nodes.values())
    ]
    if agvs:
        handles.append(Line2D([0], [0], marker="^", linestyle="",
                              markersize=8, markerfacecolor="#555",
                              markeredgecolor="black", label="AGV start"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=8)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return fig
