"""生成论文图 6-2：抢占粒度的代价-收益曲线。"""

from __future__ import annotations

import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from amr_sched.viz.style import setup

POLS = ["none", "preempt_unpicked", "preempt_any"]
LABELS = ["不抢占", "抢未取货", "抢已装货"]
COLORS = ["#9aa5b1", "#2f7ed8", "#c0392b"]


def main() -> int:
    setup()
    rows = list(csv.DictReader(
        open("results/preempt_tiers.csv", encoding="utf-8")))

    def agg(pol: str, metric: str) -> float:
        return statistics.mean(
            [float(r[metric]) for r in rows if r["policy"] == pol])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # (a) 加急准时率 + 总准时率
    ax = axes[0]
    x = range(len(POLS))
    urgent = [agg(p, "urgent_on_time_rate") for p in POLS]
    overall = [agg(p, "on_time_rate") for p in POLS]
    w = 0.36
    ax.bar([i - w / 2 for i in x], urgent, w, label="Urgent on-time rate",
           color="#2f7ed8")
    ax.bar([i + w / 2 for i in x], overall, w, label="Overall on-time rate",
           color="#8fbcd4")
    ax.set_xticks(list(x))
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("On-time rate")
    ax.set_title("(a) On-time rate: preempt-unpicked is best")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (b) 加急响应时间
    ax = axes[1]
    resp = [agg(p, "mean_urgent_response_steps") for p in POLS]
    bars = ax.bar(list(x), resp, color=COLORS, width=0.55)
    ax.set_xticks(list(x))
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Mean urgent response (steps)")
    ax.set_title("(b) Response time: improves tier by tier")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, resp):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, "%.1f" % v,
                ha="center", fontsize=9)

    # (c) 物料浪费
    ax = axes[2]
    waste = [agg(p, "material_waste_steps") for p in POLS]
    bars = ax.bar(list(x), waste, color=COLORS, width=0.55)
    ax.set_xticks(list(x))
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Material waste (s)")
    ax.set_title("(c) Cost: only preempting loaded tasks incurs waste")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, waste):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, "%.0f" % v,
                ha="center", fontsize=9)

    fig.suptitle("Preemption granularity: cost-benefit curve (90 runs per tier)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = "results/figures/preempt_tiers.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已生成", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())