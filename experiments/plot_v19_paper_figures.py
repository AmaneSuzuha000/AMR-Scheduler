"""Generate 3 publication-grade 300-DPI multi-panel figures for the AHG-RL + CVCR-B2 paper.

Reads directly from frozen JSON experiment ledgers:
  - `out/v19_scientific_audit/exp01_02_03_report.json`
  - `out/v19_scientific_audit/exp04_coupling_gradient.json`
  - `out/v19_scientific_audit/exp05_orthogonal_ablations.json`
  - `out/v19_scientific_audit/exp06_adaptive_coupling.json`
  - `out/v19_scientific_audit/exp07_unified_dual_layer.json`
  - `out/v19_scientific_audit/exp08_learning_b2_synergy.json`
  - `out/g4_scale_4axis/scale_4axis_report.json`

Outputs:
  1. `results/figures/paper_v19_main_benchmark.png`
  2. `results/figures/paper_v19_coupling_and_ablation.png`
  3. `results/figures/paper_v19_dual_layer_and_scaling.png`
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "out" / "v19_scientific_audit"
SCALE_DIR = ROOT / "out" / "g4_scale_4axis"
FIG_DIR = ROOT / "results" / "figures"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.labelweight": "bold",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    }
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_fig1_main_benchmark() -> Path:
    """Figure 1: 120-Episode Unseen Holdout Cost Decomposition & H1/H2 Attribution."""
    exp02 = _load_json(AUDIT_DIR / "exp01_02_03_report.json")["exp02_cost_attribution"]
    exp06 = _load_json(AUDIT_DIR / "exp06_adaptive_coupling.json")["summary"]
    ledgers = exp02["policy_ledgers"]

    # Build unified 8-policy comparison table
    ahg_rec = exp06["V19_ADAPTIVE_BANDIT"]
    policies = [
        ("AHG-RL\n(Adaptive)", ahg_rec["mean_net_obj"], ahg_rec["mean_cmax"], 2.0 * ahg_rec["mean_tardiness"], ahg_rec["unfinished_penalty_total"], max(0.0, ahg_rec["mean_net_obj"] - ahg_rec["mean_cmax"] - 2.0 * ahg_rec["mean_tardiness"] - ahg_rec["unfinished_penalty_total"])),
        ("HG-DRL\n(Staged)", ledgers["V19_STAGED_RL"]["mean_net_obj"], ledgers["V19_STAGED_RL"]["cost_terms"]["cmax"], ledgers["V19_STAGED_RL"]["cost_terms"]["task_tardiness_cost"] + ledgers["V19_STAGED_RL"]["cost_terms"]["op_tardiness_cost"], ledgers["V19_STAGED_RL"]["failure_penalty_subtotal"], ledgers["V19_STAGED_RL"]["control_overhead_subtotal"]),
        ("HG-DRL\n(Full Joint)", ledgers["V19_HETERO_RL"]["mean_net_obj"], ledgers["V19_HETERO_RL"]["cost_terms"]["cmax"], ledgers["V19_HETERO_RL"]["cost_terms"]["task_tardiness_cost"] + ledgers["V19_HETERO_RL"]["cost_terms"]["op_tardiness_cost"], ledgers["V19_HETERO_RL"]["failure_penalty_subtotal"], ledgers["V19_HETERO_RL"]["control_overhead_subtotal"]),
        ("V18-EA\n(CVCR-B2)", ledgers["V18_EA_JOINT"]["mean_net_obj"], ledgers["V18_EA_JOINT"]["cost_terms"]["cmax"], ledgers["V18_EA_JOINT"]["cost_terms"]["task_tardiness_cost"] + ledgers["V18_EA_JOINT"]["cost_terms"]["op_tardiness_cost"], ledgers["V18_EA_JOINT"]["failure_penalty_subtotal"], ledgers["V18_EA_JOINT"]["control_overhead_subtotal"]),
        ("IJCAI26\nPGMPO", ledgers["IJCAI26_PGMPO"]["mean_net_obj"], ledgers["IJCAI26_PGMPO"]["cost_terms"]["cmax"], ledgers["IJCAI26_PGMPO"]["cost_terms"]["task_tardiness_cost"] + ledgers["IJCAI26_PGMPO"]["cost_terms"]["op_tardiness_cost"], ledgers["IJCAI26_PGMPO"]["failure_penalty_subtotal"], ledgers["IJCAI26_PGMPO"]["control_overhead_subtotal"]),
        ("HG-BC\n(Pretrain)", ledgers["V19_HETERO_BC"]["mean_net_obj"], ledgers["V19_HETERO_BC"]["cost_terms"]["cmax"], ledgers["V19_HETERO_BC"]["cost_terms"]["task_tardiness_cost"] + ledgers["V19_HETERO_BC"]["cost_terms"]["op_tardiness_cost"], ledgers["V19_HETERO_BC"]["failure_penalty_subtotal"], ledgers["V19_HETERO_BC"]["control_overhead_subtotal"]),
        ("EJOR26\nWu-GRU", ledgers["EJOR26_Wu_GRU_Attn"]["mean_net_obj"], ledgers["EJOR26_Wu_GRU_Attn"]["cost_terms"]["cmax"], ledgers["EJOR26_Wu_GRU_Attn"]["cost_terms"]["task_tardiness_cost"] + ledgers["EJOR26_Wu_GRU_Attn"]["cost_terms"]["op_tardiness_cost"], ledgers["EJOR26_Wu_GRU_Attn"]["failure_penalty_subtotal"], ledgers["EJOR26_Wu_GRU_Attn"]["control_overhead_subtotal"]),
        ("RAL26\nDetourDelta", ledgers["RAL26_DetourDelta"]["mean_net_obj"], ledgers["RAL26_DetourDelta"]["cost_terms"]["cmax"], ledgers["RAL26_DetourDelta"]["cost_terms"]["task_tardiness_cost"] + ledgers["RAL26_DetourDelta"]["cost_terms"]["op_tardiness_cost"], ledgers["RAL26_DetourDelta"]["failure_penalty_subtotal"], ledgers["RAL26_DetourDelta"]["control_overhead_subtotal"]),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8), gridspec_kw={"width_ratios": [1.35, 1.0]})

    names = [p[0] for p in policies]
    totals = np.array([p[1] for p in policies])
    cmax = np.array([p[2] for p in policies])
    tard = np.array([p[3] for p in policies])
    fail = np.array([p[4] for p in policies])
    ctrl = np.array([p[5] for p in policies])

    x = np.arange(len(names))
    w = 0.56
    ax1.bar(x, cmax, w, label=r"Makespan ($C_{\max}$)", color="#2b5c8f")
    ax1.bar(x, tard, w, bottom=cmax, label=r"Tardiness Cost ($2T_{\mathrm{task}} + 2T_{\mathrm{op}}$)", color="#488f31")
    ax1.bar(x, ctrl, w, bottom=cmax + tard, label=r"Churn & Compute ($15C + 0.05M$)", color="#e5a93b")
    ax1.bar(x, fail, w, bottom=cmax + tard + ctrl, label=r"Unfinished Penalty ($H_2$)", color="#d9534f")

    for i, tot in enumerate(totals):
        ax1.text(i, tot + 180, f"{tot:.0f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Mean Net Objective ($J_A$, lower is better)")
    ax1.set_title("(a) 120-Episode Unseen Benchmark Cost Decomposition")
    ax1.set_ylim(0, max(totals) * 1.16)
    ax1.legend(loc="upper left", frameon=True)

    # Panel (b): Paired 100%-completed intersection vs Overall Net Obj (BC -> Full -> Staged -> AHG-RL)
    comp_labels = ["PPO vs BC\n(All 120 Eps)", "PPO vs BC\n(113 Both-100%)", "Full vs Staged\n(117 Both-100%)", "AHG-RL vs Full\n(All 120 Eps)", "AHG-RL vs Staged\n(All 120 Eps)"]
    deltas = [-1405.92, -98.03, -272.58, 5637.77 - 6001.36, 5637.77 - 5974.11]
    pcts = ["-18.98%", "-1.92% (66W/47L)", "-4.78% (H1 Gain)", "-6.06% (#1)", "-5.63% (#1)"]
    colors_b = ["#1b7837", "#5aae61", "#2166ac", "#08519c", "#2b8cbe"]

    y_pos = np.arange(len(comp_labels))
    bars = ax2.barh(y_pos, deltas, height=0.52, color=colors_b, edgecolor="#222222", linewidth=0.7)
    ax2.axvline(0, color="#333333", linewidth=1.0)
    for bar, d, pct in zip(bars, deltas, pcts):
        yc = bar.get_y() + bar.get_height() / 2.0
        if d < -800:
            ax2.text(d * 0.5, yc, f"{d:.1f} ({pct})", va="center", ha="center", color="white", fontweight="bold", fontsize=8.3)
        else:
            ax2.text(d - 22, yc, f"{d:.1f} ({pct})", va="center", ha="right", color="#111111", fontweight="bold", fontsize=8.1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(comp_labels)
    ax2.invert_yaxis()
    ax2.set_xlabel(r"Paired Net Objective Reduction ($\Delta J_A < 0$ is improvement)")
    ax2.set_title(r"(b) Paired $H_1$ Efficiency & $H_2$ Reliability Attribution")
    ax2.set_xlim(-1750, 120)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_p = FIG_DIR / "paper_v19_main_benchmark.png"
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    return out_p


def plot_fig2_coupling_ablation_and_b2() -> Path:
    """Figure 2: C0-C3 Coupling Law (EXP-04), Orthogonal Ablations (EXP-05), and CVCR-B2 Cache Audit (EXP-08)."""
    exp04 = _load_json(AUDIT_DIR / "exp04_coupling_gradient.json")
    exp05 = _load_json(AUDIT_DIR / "exp05_orthogonal_ablations.json")["summary_table"]
    exp08_path = AUDIT_DIR / "exp08_learning_b2_synergy.json"
    exp08 = _load_json(exp08_path) if exp08_path.is_file() else None

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # Panel (a): C0 -> C3 Coupling Gradient
    levels = ["C0_Weak", "C1_Moderate", "C2_Strong", "C3_Extreme"]
    x_lbls = ["C0: Weak\n(Indep. Ops)", "C1: Moderate\n(Shared Bay/Crew)", "C2: Strong\n(Kitting + DAG)", "C3: Extreme\n(+Win & Blockage)"]
    x = np.arange(len(levels))

    series = [
        ("V19_HETERO_RL", "HG-DRL (Full Joint)", "#d95f02", "-o"),
        ("V19_STAGED_RL", "HG-DRL (Staged)", "#1b9e77", "--s"),
        ("V19_HETERO_BC", "HG-BC (Pretrain)", "#7570b3", ":^"),
        ("rule_v10", "Rule-V10 + CVCR-B2", "#666666", "-.d"),
    ]
    for p_key, lbl, col, fmt in series:
        vals = [exp04["by_level"][lv]["policies"][p_key]["mean_net_obj"] for lv in levels]
        ax1.plot(x, vals, fmt, color=col, linewidth=2.0, markersize=6, label=lbl)

    ax1.annotate("Staged wins\n(-58.4%)", xy=(0, 4035), xytext=(0.18, 4600), fontsize=8.0, fontweight="bold", color="#1b9e77")
    ax1.annotate("Full Joint wins\n(-26.2% / -20.2%)", xy=(2, 7027), xytext=(1.22, 5300), fontsize=8.0, fontweight="bold", color="#d95f02")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_lbls)
    ax1.set_ylabel("Mean Net Objective ($J_A$)")
    ax1.set_title("(a) C0-C3 Cross-Domain Coupling Crossover")
    ax1.legend(loc="upper left", frameon=True)

    # Panel (b): Orthogonal Single-Factor Ablations (EXP-05)
    abl_order = [
        ("no_cross_edge_msg", "No Op<->Task Cross Msg"),
        ("no_kit_features", "No Kitting Bottleneck Feats"),
        ("no_prod_comm", "No Prod+Commit Channels"),
        ("no_a4_seq", "No A4 Queue Sequencing"),
        ("no_op_dag", "No Op Precedence DAG"),
        ("no_commit_features", "No Commitment Tier/Bias"),
        ("staged", "Staged Encoder/Decoder"),
        ("no_a3_bay_cond", "No A2->A3 Hard Projection"),
    ]
    a_labels = [it[1] for it in abl_order]
    a_pcts = [float(exp05[it[0]]["pct_degradation_vs_full"]) for it in abl_order]
    a_cols = ["#b2182b" if v > 25 else ("#ef8a62" if v > 0 else "#2166ac") for v in a_pcts]

    y = np.arange(len(a_labels))
    bars = ax2.barh(y, a_pcts, height=0.56, color=a_cols, edgecolor="#222222", linewidth=0.6)
    ax2.axvline(0, color="#333333", linewidth=1.0)
    for bar, v in zip(bars, a_pcts):
        ax2.text(
            v + 0.8 if v >= 0 else v - 0.8,
            bar.get_y() + bar.get_height() / 2.0,
            f"{v:+.1f}%",
            va="center",
            ha="left" if v >= 0 else "right",
            fontweight="bold",
            fontsize=8.0,
        )
    ax2.set_yticks(y)
    ax2.set_yticklabels(a_labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Cost Degradation vs. Full Joint (%)")
    ax2.set_title("(b) Orthogonal Mechanism Ablations (EXP-05)")
    ax2.set_xlim(-17, 58)

    # Panel (c): CVCR-B2 Cache Hit Rate & Fleet Reuse Footprint (EXP-08 Part A)
    if exp08 is not None and "part_a_candidate_footprint_and_cache_audit" in exp08:
        pa = exp08["part_a_candidate_footprint_and_cache_audit"]
        m_keys = [
            ("Local_1Opt_Swap_CVCR_B2", "1-Opt/Swap\n(CVCR-B2)"),
            ("V19_Full_A123_NoA4", "HG-DRL (A123)\n+ Seeded B2"),
            ("V19_Full_Joint_A1234", "HG-DRL (Full)\n+ Seeded B2"),
            ("V19_Staged_A1234", "HG-DRL (Staged)\n+ Seeded B2"),
        ]
        c_names = [it[1] for it in m_keys]
        hit_rates = [float(pa[it[0]]["seeded_b2_cache_hit_rate_pct"]) for it in m_keys]
        reused_fleet = [float(pa[it[0]]["mean_reused_amr_ratio_pct"]) for it in m_keys]
        x3 = np.arange(len(c_names))
        w3 = 0.34
        b_hit = ax3.bar(x3 - w3 / 2, hit_rates, w3, label="B2 Incremental Hit Rate (%)", color="#2b5c8f")
        b_reu = ax3.bar(x3 + w3 / 2, reused_fleet, w3, label="Reused Unchanged Fleet (%)", color="#488f31")
        for b in b_hit:
            ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.1f}%", ha="center", fontsize=7.8, fontweight="bold")
        for b in b_reu:
            ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.1f}%", ha="center", fontsize=7.8, fontweight="bold")
        ax3.set_xticks(x3)
        ax3.set_xticklabels(c_names)
        ax3.set_ylabel("Percentage (%)")
        ax3.set_ylim(0, 100)
        ax3.set_title("(c) Learning Action Footprint & CVCR-B2 Reuse")
        ax3.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    out_p = FIG_DIR / "paper_v19_coupling_and_ablation.png"
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    return out_p


def plot_fig3_dual_layer_and_scaling() -> Path:
    """Figure 3: Unified Dual-Layer Reconciliation (EXP-07) & 62-Node Fleet/Budget Scaling (G4)."""
    exp07 = _load_json(AUDIT_DIR / "exp07_unified_dual_layer.json")
    scale_rep = _load_json(SCALE_DIR / "scale_4axis_report.json")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.5))

    # Panel (a): Dual-Layer Reconciliation on Track 1 & Track 2
    t1_rows = exp07["tracks"]["track1"]["paired_rows"]
    t2_rows = exp07["tracks"]["track2"]["paired_rows"]
    offsets = {
        "V19_HETERO_RL": (-46, -12),
        "V18_EA_JOINT": (6, 4),
        "IJCAI26_PGMPO": (6, 9),
        "EJOR26_Wu_GRU_Attn": (6, -3),
        "RAL26_DetourDelta": (6, -14),
        "rule_v10": (6, -6),
    }
    short_names = {
        "V19_HETERO_RL": "HG-DRL (V19)",
        "V18_EA_JOINT": "V18-EA",
        "IJCAI26_PGMPO": "IJCAI26-PGMPO",
        "EJOR26_Wu_GRU_Attn": "EJOR26-Wu",
        "RAL26_DetourDelta": "RAL26-Detour",
        "rule_v10": "Rule-V10",
    }
    for row in t1_rows:
        pk = row["policy"]
        ax1.scatter(
            row["layer_a_net_obj"],
            row["layer_b_net_obj"],
            color="#d95f02" if pk == "V19_HETERO_RL" else "#1f78b4",
            s=62,
            zorder=4,
        )
        ax1.annotate(
            short_names.get(pk, pk),
            (row["layer_a_net_obj"], row["layer_b_net_obj"]),
            fontsize=7.8,
            fontweight="bold" if pk == "V19_HETERO_RL" else "normal",
            xytext=offsets.get(pk, (5, 3)),
            textcoords="offset points",
        )
    ax1.set_xlim(1208, 1304)
    ax1.set_ylim(1366, 1460)
    ax1.set_xlabel(r"Layer A Discrete Sim Objective ($J_A$)")
    ax1.set_ylabel(r"Layer B Physical Closed-Loop Objective ($J_{\mathrm{unified}}$)")
    ax1.set_title(r"(a) Track 1 Dual-Layer Alignment ($\rho = 0.8857$)")

    # Panel (b): Track 2 Physical Execution Objective & P95 Latency
    pols_t2 = [r["policy"].replace("V19_HETERO_RL", "HG-DRL").replace("V18_EA_JOINT", "V18-EA").replace("IJCAI26_PGMPO", "PGMPO").replace("EJOR26_Wu_GRU_Attn", "Wu-GRU").replace("RAL26_DetourDelta", "RAL26") for r in t2_rows]
    lb_objs = [r["layer_b_net_obj"] for r in t2_rows]
    p95_ms = [r["layer_b_propose_ms_p95"] for r in t2_rows]
    x2 = np.arange(len(pols_t2))
    bars2 = ax2.bar(x2, lb_objs, width=0.52, color=["#d95f02" if "HG-DRL" in p else "#377eb8" for p in pols_t2])
    for b, val, lat in zip(bars2, lb_objs, p95_ms):
        ax2.text(b.get_x() + b.get_width() / 2, val + 120, f"{val:.0f}\n({lat:.1f}ms)", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(pols_t2)
    ax2.set_ylabel(r"Layer B Physical Objective ($J_{\mathrm{unified}}$)")
    ax2.set_ylim(0, max(lb_objs) * 1.24)
    ax2.set_title(r"(b) Track 2 Physical Execution ($\rho = 0.7714$, P95 < 18ms)")

    # Panel (c): 62-Node Disruption Regime Stress & Real-Time Performance (Axis 3)
    ax3_data = scale_rep["axes"]["axis3_disruption_regime"]
    regimes = ["nominal", "kitting_windows", "execution_blockage", "multi_wave_surge"]
    r_x = np.arange(len(regimes))
    for p_key, lbl, col, fmt in (
        ("V19_HETERO_RL", "HG-DRL (V19)", "#d95f02", "-o"),
        ("V18_EA_JOINT", "V18-EA (CVCR-B2)", "#1b9e77", "--s"),
        ("IJCAI26_PGMPO", "IJCAI26-PGMPO", "#377eb8", "-.^"),
        ("rule_v10", "Rule-V10", "#666666", ":d"),
    ):
        y_vals = [ax3_data[rk][p_key]["mean_net_obj"] for rk in regimes]
        ax3.plot(r_x, y_vals, fmt, color=col, linewidth=2.0, markersize=6, label=lbl)
    ax3.set_xticks(r_x)
    ax3.set_xticklabels(["Nominal", "Kitting+\nWindows", "Corridor\nBlockage", "Multi-Wave\nSurge"])
    ax3.set_xlabel("62-Node Workshop Disruption & Coupling Regime")
    ax3.set_ylabel("Mean Net Objective ($J_A$)")
    ax3.set_title("(c) 62-Node Disruption Stress Resilience (Wall-Clock)")
    ax3.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    out_p = FIG_DIR / "paper_v19_dual_layer_and_scaling.png"
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    return out_p


def main() -> None:
    p1 = plot_fig1_main_benchmark()
    p2 = plot_fig2_coupling_ablation_and_b2()
    p3 = plot_fig3_dual_layer_and_scaling()
    print(f"Generated publication figures:\n  - {p1}\n  - {p2}\n  - {p3}")


if __name__ == "__main__":
    main()
