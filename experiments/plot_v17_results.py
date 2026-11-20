"""Generate V17 empirical attribution, grand benchmark, and HIL visual figures.

Reads frozen V17 JSON/PNG artifacts in `evidence/` and writes clean thesis-ready
figures into `results/figures/`:
- `v17_e2_attribution.png`
- `v17_grand_benchmark.png`
- `v17_hil_detour_keyframe.png`
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from amr_sched.viz.style import setup as _setup_style

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "results" / "figures"


def plot_e2_attribution(out_path: Path) -> None:
    """Plot E2 2x2+Restraint comparison and same-controller P4.5-VoI delta."""
    data = json.loads((REPO / "evidence" / "v17_e2_attribution.json").read_text(encoding="utf-8"))
    overall = data["attribution_2x2_plus_restraint"]["OVERALL"]
    voi = data["p45_voi_same_controller_ablation"]

    arms = [
        ("Rule_Det_RestraintON", "Rule + Det\n(Restraint ON)"),
        ("Rule_Pairwise_RestraintON", "Rule + 50D\n(Restraint ON)"),
        ("Learned_Det_RestraintON", "Learned + Det\n(Restraint ON)"),
        ("Learned_Pairwise_RestraintON", "Learned + 50D\n(Restraint ON)"),
        ("Learned_Pairwise_RestraintOFF", "Learned + 50D\n(Restraint OFF)"),
    ]
    means = [float(overall[k]["mean_net"]) for k, _ in arms]
    meds = [float(overall[k]["median_net"]) for k, _ in arms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8))
    x = np.arange(len(arms))
    w = 0.36
    b1 = ax1.bar(x - w / 2, means, width=w, color="#2f6fb3", label="均值净成本 (Mean J_net)")
    b2 = ax1.bar(x + w / 2, meds, width=w, color="#4ea874", label="中位数净成本 (Median J_net)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in arms], fontsize=8)
    ax1.set_ylabel("净成本 J_net（越低越好）")
    ax1.set_title("E2 2x2 + Keep 零残差机制归因 (n=44, 100% 完单)")
    ax1.set_ylim(0, max(means) * 1.18)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right", fontsize=8)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 35, f"{h:.0f}", ha="center", va="bottom", fontsize=7)

    ordered = sorted(voi.items(), key=lambda kv: float(kv[1]["delta_on_minus_off"]))
    labels = [k for k, _ in ordered]
    deltas = [float(v["delta_on_minus_off"]) for _, v in ordered]
    colors = ["#2e8b57" if d < 0 else ("#7f8c8d" if abs(d) < 1e-6 else "#d64545") for d in deltas]
    y = np.arange(len(labels))
    bars = ax2.barh(y, deltas, color=colors, height=0.62)
    ax2.axvline(0.0, color="#222222", linewidth=1.0)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("delta_on_minus_off (VoI ON - VoI OFF，负值为净改善)")
    ax2.set_title("同控制器 V17_EA_JOINT P4.5-VoI 纯开关消融 (每组 n=8)")
    ax2.set_xlim(min(deltas) - 35, max(deltas) + 40)
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    for bar, d in zip(bars, deltas):
        offset = -4 if d < 0 else 4
        ha = "right" if d < 0 else "left"
        ax2.text(d + offset, bar.get_y() + bar.get_height() / 2, f"{d:+.2f}", va="center", ha=ha, fontsize=7.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_grand_benchmark(out_path: Path) -> None:
    """Plot E4 6-track V17_EA_JOINT vs V16/V15/rules/RL/best-literature comparison."""
    data = json.loads((REPO / "evidence" / "v17_grand_benchmark.json").read_text(encoding="utf-8"))
    tracks = data["tracks"]
    suites = [
        ("UNSEEN_STD", "UNSEEN_STD (n=24)"),
        ("HOLDOUT_V17_FINAL", "HOLDOUT_V17 (n=24)"),
        ("NEW_A", "NEW_A (n=72)"),
        ("STRESS_5AXIS", "STRESS_5AXIS (n=24)"),
        ("EXP_D_160", "EXP_D_160 (n=160)"),
        ("EXEC_P45_VOI", "EXEC_P45_VOI (n=144)"),
    ]
    methods = [
        ("V17_EA_JOINT", "V17_EA_JOINT", "#1f4e79"),
        ("V16_EA_JOINT", "V16_EA_JOINT", "#3b82c4"),
        ("V15_EA_JOINT", "V15_EA_JOINT", "#7ca9d8"),
        ("rule_v10", "rule_v10", "#d97724"),
        ("rl_v7", "rl_v7", "#2e8b57"),
        ("BEST_LIT", "最优文献基线", "#8c564b"),
    ]
    lit_names = ("WangESWA24", "ShiAEI25", "BalansBanditPhan24", "ReijnenDRALNS24")

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    axes_flat = axes.ravel()
    for ax, (sk, title) in zip(axes_flat, suites):
        m_dict = tracks[sk]["methods"]
        avail_lit = [ln for ln in lit_names if ln in m_dict]
        best_lit_key = min(avail_lit, key=lambda ln: float(m_dict[ln]["mean_net"]))
        vals = []
        unfs = []
        lbls = []
        cols = []
        for mk, m_lbl, col in methods:
            actual_k = best_lit_key if mk == "BEST_LIT" else mk
            row = m_dict[actual_k]
            vals.append(float(row["mean_net"]))
            unfs.append(float(row["mean_unfinished_penalty"]))
            lbls.append(f"{m_lbl}\n({best_lit_key[:8]})" if mk == "BEST_LIT" else m_lbl)
            cols.append(col)
        x = np.arange(len(vals))
        bars = ax.bar(x, vals, color=cols, width=0.62)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(lbls, fontsize=7, rotation=18, ha="right")
        ax.set_ylabel("Mean J_net")
        ymax = max(vals) * 1.24
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, v, u in zip(bars, vals, unfs):
            tag = f"{v:.0f}*" if u > 0.0 else f"{v:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2, v + ymax * 0.02, tag, ha="center", va="bottom", fontsize=7)

    fig.suptitle(
        "E4 跨赛道全审计大基准 (448 独立场景 / 3,872 局; * 表示含未完工罚分 > 0)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def copy_hil_keyframe(out_path: Path) -> None:
    """Copy verified V17 WSL2 HIL autonomous bypass detour frame into results/figures."""
    src = REPO / "evidence" / "v17_hil" / "frames" / "frame_03_detour_reroute.png"
    if not src.exists():
        raise FileNotFoundError(f"Missing V17 HIL keyframe: {src}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_path)


def main() -> None:
    """Generate all V17 thesis figures."""
    _setup_style(9.0)
    f_e2 = FIG_DIR / "v17_e2_attribution.png"
    f_e4 = FIG_DIR / "v17_grand_benchmark.png"
    f_hil = FIG_DIR / "v17_hil_detour_keyframe.png"
    plot_e2_attribution(f_e2)
    plot_grand_benchmark(f_e4)
    copy_hil_keyframe(f_hil)
    for fp in (f_e2, f_e4, f_hil):
        print(f"generated {fp.relative_to(REPO)} ({os.path.getsize(fp)} B)")


if __name__ == "__main__":
    main()
