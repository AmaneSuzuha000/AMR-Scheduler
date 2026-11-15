"""调度过程监控可视化（任务书 R4.4 AGV状态监控 + R4.5 调度过程可视化）。

为什么需要它
------------
静态图（甘特图、轨迹图）只能看"结果"，看不到"过程"。
任务书 R4.4 要求 AGV 状态监控、R4.5 要求调度过程可视化，
这两个要求指向的是**随时间演化的状态**，不是最终快照。

本模块提供三样东西：

1. ``agv_state_timeline``  ：每台车在每一时刻的状态时间线（idle/装载/阻塞/停车…）
2. ``task_progress_curve`` ：任务状态随时间的堆积曲线（待分配/已分配/已取货/已完成）
3. ``render_animation``    ：逐帧动画（GIF），直观展示车在网络上跑

用法::

    python -m amr_sched.viz.animation --tasks 20 --agvs 2 --seed 0
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from amr_sched.viz import style

#: 状态 → 颜色（与 layout 的语义配色保持同一套语言）
STATE_COLORS = {
    "idle": "#e8edf2",
    "to_pickup": "#5aa9e6",
    "loading": "#f2b134",
    "to_dropoff": "#2e8b57",
    "unloading": "#8fbf6f",
    "blocked": "#d64545",
    "parking": "#9aa5b1",
}
STATE_LABELS = {
    "idle": "空闲",
    "to_pickup": "去取货",
    "loading": "装货中",
    "to_dropoff": "去送达",
    "unloading": "卸货中",
    "blocked": "阻塞",
    "parking": "让位停车",
}


def _state_matrix(result, agv_ids):
    """把轨迹整理成 (车 × 步) 的状态矩阵与状态名列表。"""
    names = sorted(STATE_COLORS)
    idx = {n: i for i, n in enumerate(names)}
    steps = [rec.step for rec in result.trajectory]
    mat = np.zeros((len(agv_ids), len(steps)), dtype=int)
    for j, rec in enumerate(result.trajectory):
        for i, aid in enumerate(agv_ids):
            st = rec.agv_states.get(aid, "idle")
            mat[i, j] = idx.get(st, idx["idle"])
    return mat, names, steps


def agv_state_timeline(result, out_path=None,
                       title="AGV 状态时间线（R4.4 状态监控）"):
    """每台车一条横向色带，颜色表示该时刻的状态。"""
    style.setup()
    agv_ids = [a.agv_id for a in result.agvs]
    mat, names, steps = _state_matrix(result, agv_ids)

    fig, ax = plt.subplots(figsize=(11, 0.55 * len(agv_ids) + 2.2))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([STATE_COLORS[n] for n in names])
    ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest",
              vmin=-0.5, vmax=len(names) - 0.5,
              extent=[steps[0], steps[-1], len(agv_ids) - 0.5, -0.5])
    ax.set_yticks(range(len(agv_ids)))
    ax.set_yticklabels(agv_ids)
    ax.set_xlabel("Simulation time (steps)")
    ax.set_title(title)

    from matplotlib.patches import Patch
    present = {names[v] for v in np.unique(mat)}
    handles = [Patch(facecolor=STATE_COLORS[n], label=STATE_LABELS.get(n, n))
               for n in names if n in present]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=8)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return fig


def task_progress_curve(result, out_path=None,
                        title="任务状态演化（R4.5 调度过程）"):
    """四类任务状态随时间的堆积面积图。"""
    style.setup()
    steps = [rec.step for rec in result.trajectory]
    keys = ["pending", "assigned", "picked", "completed"]
    series = {k: [rec.task_counts.get(k, 0) for rec in result.trajectory]
              for k in keys}

    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = ["#cfd6dd", "#5aa9e6", "#f2b134", "#2e8b57"]
    labels = ["待分配", "已分配", "已取货(在途)", "已完成"]
    ax.stackplot(steps, [series[k] for k in keys],
                 colors=colors, labels=labels, alpha=0.92)
    ax.set_xlabel("仿真时间（步）")
    ax.set_ylabel("Tasks")
    ax.set_title(title)
    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(0, max(1, max(series["pending"]) + max(series["completed"])))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              fontsize=8)
    ax.grid(alpha=0.25, linestyle=":")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return fig


def _agv_xy(network, pos: str):
    """把"R1"或"R1->R2"解析为坐标（在途车取边中点）。"""
    if "->" in pos:
        a, b = pos.split("->")
        na, nb = network.node(a), network.node(b)
        return (na.x + nb.x) / 2.0, (na.y + nb.y) / 2.0
    n = network.node(pos)
    return n.x, n.y


def render_animation(network, result, out_path, frame_every: int = 20,
                     max_frames: int = 240):
    """把轨迹渲染成 GIF 动画。

    ``frame_every`` 控制抽帧间隔：轨迹可能上万步，全画会得到巨大的 GIF。
    """
    style.setup()
    import matplotlib.animation as anim

    frames = result.trajectory[::max(1, frame_every)][:max_frames]
    if not frames:
        raise ValueError("轨迹为空，请用 record_trajectory=True 运行仿真")

    xs = [n.x for n in network.nodes.values()]
    ys = [n.y for n in network.nodes.values()]
    pad = 8.0
    xlim = (min(xs) - pad, max(xs) + pad)
    ylim = (min(ys) - pad, max(ys) + pad)

    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    def draw(idx):
        ax.clear()
        rec = frames[idx]
        # 边
        seen = set()
        for e in network.edges.values():
            key = (min(e.src, e.dst), max(e.src, e.dst))
            if key in seen:
                continue
            seen.add(key)
            a, b = network.node(e.src), network.node(e.dst)
            ax.plot([a.x, b.x], [a.y, b.y], color="#dfe4ea",
                    linewidth=1.0 + 0.8 * (e.capacity - 1), zorder=1)
        # 节点
        for n in network.nodes.values():
            ax.scatter([n.x], [n.y], s=150,
                       color=style.NODE_COLORS.get(n.kind.value, "#888"),
                       zorder=3, edgecolors="white", linewidths=1.0)
            ax.annotate(n.node_id, (n.x, n.y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7, zorder=4)
        # AGV
        for i, a in enumerate(result.agvs):
            pos = rec.agv_positions.get(a.agv_id, "")
            if not pos:
                continue
            x, y = _agv_xy(network, pos)
            st = rec.agv_states.get(a.agv_id, "idle")
            ax.scatter([x], [y], s=190, marker="^",
                       color=STATE_COLORS.get(st, "#555"),
                       zorder=6, edgecolors="black", linewidths=0.8)
            ax.annotate(a.agv_id, (x, y), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=7,
                        zorder=6)
        c = rec.task_counts
        ax.set_title(
            "步 %d ｜ 完成 %d ｜ 在途 %d ｜ 待分配 %d ｜ 阻塞车 %d"
            % (rec.step, c.get("completed", 0), c.get("picked", 0),
               c.get("pending", 0), rec.n_blocked))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    ani = anim.FuncAnimation(fig, draw, frames=len(frames), interval=120)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        ani.save(out_path, writer="pillow", dpi=100)
    finally:
        plt.close(fig)
    return out_path


def _main(argv=None) -> int:
    p = argparse.ArgumentParser(description="生成调度过程监控可视化")
    p.add_argument("--tasks", type=int, default=20)
    p.add_argument("--agvs", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results/figures")
    p.add_argument("--frame-every", type=int, default=20)
    args = p.parse_args(argv)

    from amr_sched.allocation.greedy import GreedyAllocator
    from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
    from amr_sched.scenarios import ScenarioConfig, generate_scenario
    from amr_sched.simulation import SimulationConfig, Simulator

    sc = ScenarioConfig(n_tasks=args.tasks, n_agvs=args.agvs, seed=args.seed)
    net, tasks, agvs = generate_scenario(sc)
    r = Simulator(net, tasks, agvs,
                  SpaceTimeAStarPlanner(max_speed=sc.max_speed, dt=sc.dt),
                  GreedyAllocator(),
                  SimulationConfig(dt=sc.dt, max_steps=200_000,
                                   record_trajectory=True)).run()
    s = r.summary()
    print("完成 %d/%d，用时 %.1fs"
          % (s["tasks_completed"], args.tasks, s["makespan_seconds"]))

    os.makedirs(args.out_dir, exist_ok=True)
    agv_state_timeline(r, os.path.join(args.out_dir, "agv_states.png"))
    task_progress_curve(r, os.path.join(args.out_dir, "task_progress.png"))
    render_animation(net, r, os.path.join(args.out_dir, "animation.gif"),
                     frame_every=args.frame_every)
    for f in ("agv_states.png", "task_progress.png", "animation.gif"):
        path = os.path.join(args.out_dir, f)
        print("  %-20s %8d B" % (f, os.path.getsize(path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
