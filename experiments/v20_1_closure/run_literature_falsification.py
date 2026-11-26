"""Runner for `IF-CORE-V20-LITERATURE-01`: EXP-1 (Structural SCC), EXP-2 (Persistent Escape),
EXP-3 (Cross-Domain ADDRESS), and EXP-4 (Neural-Prior Guided Search).

All experiments enforce strict anti-leakage guardrails:
  - Fixed deterministic work budget (`virtual_clock_scope`) evaluated FIRST for pure mechanism
    attribution, followed by uncontended real wall-clock checks on quarantined DEV splits
    (`VAL-36` / `VAL-72` / `DEV-Stress-24`).
  - Zero edits to frozen `V20.0` or `amr_sched/`.
  - Emits `out/v20_1_closure/literature_falsification_report.{json,md}` + P2/P3 interface analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.cvcr.clock import virtual_clock_scope
from experiments.v20_1_closure.exp1_structural_pdg_scc import StructuralPDGMonitor
from experiments.v20_1_closure.exp2_persistent_escape import PersistentPDG2Proposer
from experiments.v20_1_closure.exp3_cross_domain_address import CrossDomainAddressProposer
from experiments.v20_1_closure.exp4_neural_prior_search import NeuralPriorGuidedProposer
from experiments.v20_1_closure.pdg2_escape import build_pdg2_v20_proposer
from experiments.v20_1_closure.task2_pdg2_ablation import build_dev_stress_24_specs
from experiments.v20_1_closure.task4_compute_robustness import _select_stratified_val36
from experiments.v20_audit.common import (
    OUT_AUDIT_DIR,
    assert_stage1_quarantine,
    make_v18_external_proposer,
    run_wrapped_episode,
    summarize_episode_rows,
)

OUT_CLOSURE_DIR = ROOT / "out" / "v20_1_closure"


def _base_v20b_kw() -> dict[str, Any]:
    return dict(
        hard_budget_ms=35.0,
        soft_deadline_ms=15.0,
        seed=0,
        enable_e4=True,
        e4_selector="fast_surr",
        e4_include_zc=False,
        e4_mode="ws_or_invalid",
        staged_mode="ws_or_invalid",
        enable_bidir_v18_to_v19=True,
        enable_e6_static_or_headroom=False,
        enable_e7=True,
        enable_e9=False,
        enable_early_stop=False,
        compact_calm_ms=None,
        enable_r2_global_ws=False,
        r2_zc_ws_mode="never",
        include_staged_g="on_invalid",
        unwin_seq_horizon=2,
        seq_diff_b2=True,
        critic_weights_path=OUT_AUDIT_DIR / "v20_a_critic_weights.npz",
        critic_beta=0.25,
        critic_mode="ws_only",
    )


class _null_ctx:
    def __enter__(self):
        return None
    def __exit__(self, *args):
        return False


def _paired_p(cand_rows: list[dict[str, Any]], ref_rows: list[dict[str, Any]]) -> float:
    d = np.array([float(a["net_obj"]) - float(b["net_obj"]) for a, b in zip(cand_rows, ref_rows)], dtype=np.float64)
    if not np.any(np.abs(d) > 1e-9):
        return 1.0
    return float(wilcoxon(d, zero_method="wilcox", alternative="less").pvalue)


def run_literature_falsification() -> dict[str, Any]:
    OUT_CLOSURE_DIR.mkdir(parents=True, exist_ok=True)
    val36 = _select_stratified_val36()
    stress24 = build_dev_stress_24_specs()
    assert_stage1_quarantine([s["seed"] for s in val36 + stress24], label="Literature_Falsification_DEV")

    report: dict[str, Any] = {
        "schema": "v20_1_closure.literature_falsification.v1",
        "exp1_structural_pdg_scc": {},
        "exp2_persistent_escape": {},
        "exp3_cross_domain_address": {},
        "exp4_neural_prior_search": {},
    }

    # --- EXP-1: Structural Progress Dependency + Tarjan SCC (DEV-Stress-24 & VAL-36 FPR) ---
    print("=== [EXP-1 / P0] Structural Wait-For Graph + Tarjan SCC (4-way ablation) ===", flush=True)
    for mode in ("S0_Legacy_PDG", "S1_Local_D1_Only", "S2_WaitGraph_SCC_Only", "S3_Joint_SCC_Plus_Local"):
        st_rows = []
        cls_tot: dict[str, int] = {}
        alerts_st = 0
        for sp in stress24:
            mon = StructuralPDGMonitor(mode=mode)
            prop = build_pdg2_v20_proposer(hard_budget_ms=35.0, seed=0, enable_e1=True, enable_e2=True)
            with virtual_clock_scope(poll_ms=0.002):
                r, _ = run_wrapped_episode(
                    variant_id=f"exp1_{mode}", spec=sp, base_env_policy="V19_ADAPTIVE_RL",
                    external_proposer=prop, hard_budget_ms=35.0,
                    outer_action_hook=lambda env, info, act, tr, _m=mon, _p=prop: _m.observe_and_maybe_escalate(env, info, act, tr, proposer=_p),
                )
            r["alerts"] = len(mon.alert_ticks)
            if mon.alert_ticks:
                alerts_st += 1
            for k_c, v_c in mon.state_class_counts.items():
                cls_tot[k_c] = cls_tot.get(k_c, 0) + v_c
            st_rows.append(r)

        # FPR check on VAL-36 (deterministic virtual clock)
        v_fpr = 0
        v_rows = []
        for sp in val36:
            mon = StructuralPDGMonitor(mode=mode)
            prop = build_pdg2_v20_proposer(hard_budget_ms=35.0, seed=0, enable_e1=True, enable_e2=True)
            with virtual_clock_scope(poll_ms=0.002):
                r, _ = run_wrapped_episode(
                    variant_id=f"exp1_v_{mode}", spec=sp, base_env_policy="V19_ADAPTIVE_RL",
                    external_proposer=prop, hard_budget_ms=35.0,
                    outer_action_hook=lambda env, info, act, tr, _m=mon, _p=prop: _m.observe_and_maybe_escalate(env, info, act, tr, proposer=_p),
                )
            if mon.alert_ticks:
                v_fpr += 1
            v_rows.append(r)

        sm_st = summarize_episode_rows(st_rows)
        sm_v = summarize_episode_rows(v_rows)
        report["exp1_structural_pdg_scc"][mode] = {
            "stress24": {**sm_st, "alert_episodes": alerts_st, "state_class_ticks": cls_tot},
            "val36_fpr_episodes": v_fpr,
            "val36_mean_net_obj": sm_v["mean_net_obj"],
            "val36_comp100": sm_v["comp100_count"],
        }
        print(
            f"  EXP-1 {mode:24s}: Stress24 Comp={sm_st['comp100_count']}/24 JA={sm_st['mean_net_obj']:7.1f} Alerts={alerts_st:2d} "
            f"| VAL36 Comp={sm_v['comp100_count']}/36 JA={sm_v['mean_net_obj']:7.1f} FPR={v_fpr}/36 | Classes={cls_tot}",
            flush=True,
        )

    # --- EXP-2: Persistent Escape Search Memory M_t (Stateless vs Persistent on DEV-Stress-24, Virtual + Wall) ---
    print("=== [EXP-2 / P0] Persistent Escape Search Memory M_t (Real-Time LaCAM Inspired) ===", flush=True)
    for en_mem, lbl in ((False, "Stateless_PDG2"), (True, "Persistent_Memory_PDG2")):
        rows_v, rows_w = [], []
        pruned_tot = exp_tot = 0
        for sp in stress24:
            mon = StructuralPDGMonitor(mode="S3_Joint_SCC_Plus_Local")
            prop = PersistentPDG2Proposer(enable_persistent_memory=en_mem, enable_e1=True, enable_e2=True, **_base_v20b_kw())
            with virtual_clock_scope(poll_ms=0.002):
                rv, _ = run_wrapped_episode(
                    variant_id=f"exp2_v_{lbl}", spec=sp, base_env_policy="V19_ADAPTIVE_RL",
                    external_proposer=prop, hard_budget_ms=35.0,
                    outer_action_hook=lambda env, info, act, tr, _m=mon, _p=prop: _m.observe_and_maybe_escalate(env, info, act, tr, proposer=_p),
                )
            pruned_tot += prop.memory.duplicate_candidates_pruned
            exp_tot += prop.memory.frontier_expansions
            rows_v.append(rv)

            mon_w = StructuralPDGMonitor(mode="S3_Joint_SCC_Plus_Local")
            prop_w = PersistentPDG2Proposer(enable_persistent_memory=en_mem, enable_e1=True, enable_e2=True, **_base_v20b_kw())
            rw, _ = run_wrapped_episode(
                variant_id=f"exp2_w_{lbl}", spec=sp, base_env_policy="V19_ADAPTIVE_RL",
                external_proposer=prop_w, hard_budget_ms=35.0,
                outer_action_hook=lambda env, info, act, tr, _m=mon_w, _p=prop_w: _m.observe_and_maybe_escalate(env, info, act, tr, proposer=_p),
            )
            rows_w.append(rw)

        smv, smw = summarize_episode_rows(rows_v), summarize_episode_rows(rows_w)
        report["exp2_persistent_escape"][lbl] = {
            "virtual_stress24": {**smv, "duplicates_pruned": pruned_tot, "frontier_expansions": exp_tot},
            "wall_stress24": smw,
        }
        print(
            f"  EXP-2 {lbl:24s}: Virt Comp={smv['comp100_count']}/24 JA={smv['mean_net_obj']:7.1f} (Pruned={pruned_tot}, Exp={exp_tot}) "
            f"| Wall Comp={smw['comp100_count']}/24 JA={smw['mean_net_obj']:7.1f} P95={smw['pooled_p95_ms']:5.2f}ms",
            flush=True,
        )

    # --- EXP-3: Cross-Domain ADDRESS (N0 vs N1_TopK vs N1_Rand vs N2_RestrictedTS on VAL-36, Virtual + Wall) ---
    print("=== [EXP-3 / P1] Cross-Domain ADDRESS (Top-K & Restricted Thompson Sampling) ===", flush=True)
    exp3_virt_rows: dict[str, list[dict[str, Any]]] = {}
    for amode in ("N0_CausalCone", "N1_Address_TopK", "N1_Random_TopK", "N2_Address_RestrictedTS"):
        rv_list, rw_list = [], []
        imps = calls = 0
        for sp in val36:
            pv = CrossDomainAddressProposer(address_mode=amode, **_base_v20b_kw())
            with virtual_clock_scope(poll_ms=0.002):
                rv, _ = run_wrapped_episode(variant_id=f"exp3_v_{amode}", spec=sp, base_env_policy="V19_ADAPTIVE_RL", external_proposer=pv, hard_budget_ms=35.0)
            imps += pv.address_improvements
            calls += pv.address_calls
            rv_list.append(rv)

            pw = CrossDomainAddressProposer(address_mode=amode, **_base_v20b_kw())
            rw, _ = run_wrapped_episode(variant_id=f"exp3_w_{amode}", spec=sp, base_env_policy="V19_ADAPTIVE_RL", external_proposer=pw, hard_budget_ms=35.0)
            rw_list.append(rw)

        exp3_virt_rows[amode] = rv_list
        smv, smw = summarize_episode_rows(rv_list), summarize_episode_rows(rw_list)
        pv_n0 = _paired_p(rv_list, exp3_virt_rows["N0_CausalCone"]) if amode != "N0_CausalCone" else 1.0
        by_s_v = {g: summarize_episode_rows([r for r in rv_list if r["suite_group"] == g])["mean_net_obj"] for g in ("4topo", "track1", "track2")}
        report["exp3_cross_domain_address"][amode] = {
            "virtual_val36": {**smv, "by_suite_net": by_s_v, "wilcoxon_p_lt_n0": round(pv_n0, 6), "improvements": imps, "calls": calls},
            "wall_val36": smw,
        }
        print(
            f"  EXP-3 {amode:24s}: Virt JA={smv['mean_net_obj']:7.1f} (4T={by_s_v['4topo']:.1f} T1={by_s_v['track1']:.1f} T2={by_s_v['track2']:.1f}, p={pv_n0:.4f}) "
            f"| Wall JA={smw['mean_net_obj']:7.1f} Comp={smw['comp100_count']}/36 P95={smw['pooled_p95_ms']:5.2f}ms",
            flush=True,
        )

    # --- EXP-4: Neural-Prior Guided Search (V18-Det vs V20-Bidir vs NeuralPrior vs RandomPrior on VAL-36) ---
    print("=== [EXP-4 / P1] Neural-Prior Guided Search (Frozen V19 P_theta Distribution) ===", flush=True)
    exp4_virt_rows: dict[str, list[dict[str, Any]]] = {}
    for pmode in ("P0_V18_DetSearch", "P1_V20_Bidirectional", "P2_NeuralPrior_Guided", "P3_RandomPrior_Ctrl"):
        rv_list, rw_list = [], []
        wins_p = calls_p = 0
        for sp in val36:
            if pmode == "P0_V18_DetSearch":
                pv = make_v18_external_proposer(hard_budget_ms=35.0, seq_diff_b2=True)
                pw = make_v18_external_proposer(hard_budget_ms=35.0, seq_diff_b2=True)
            else:
                pv = NeuralPriorGuidedProposer(prior_mode=pmode, n_alt_branches=2, **_base_v20b_kw())
                pw = NeuralPriorGuidedProposer(prior_mode=pmode, n_alt_branches=2, **_base_v20b_kw())
            with virtual_clock_scope(poll_ms=0.002):
                rv, _ = run_wrapped_episode(variant_id=f"exp4_v_{pmode}", spec=sp, base_env_policy="V19_ADAPTIVE_RL", external_proposer=pv, hard_budget_ms=35.0)
            if isinstance(pv, NeuralPriorGuidedProposer):
                wins_p += pv.prior_branch_wins
                calls_p += pv.prior_calls
            rv_list.append(rv)

            rw, _ = run_wrapped_episode(variant_id=f"exp4_w_{pmode}", spec=sp, base_env_policy="V19_ADAPTIVE_RL", external_proposer=pw, hard_budget_ms=35.0)
            rw_list.append(rw)

        exp4_virt_rows[pmode] = rv_list
        smv, smw = summarize_episode_rows(rv_list), summarize_episode_rows(rw_list)
        pv_p1 = _paired_p(rv_list, exp4_virt_rows["P1_V20_Bidirectional"]) if "P1_V20_Bidirectional" in exp4_virt_rows and pmode != "P1_V20_Bidirectional" else 1.0
        by_s_v = {g: summarize_episode_rows([r for r in rv_list if r["suite_group"] == g])["mean_net_obj"] for g in ("4topo", "track1", "track2")}
        report["exp4_neural_prior_search"][pmode] = {
            "virtual_val36": {**smv, "by_suite_net": by_s_v, "wilcoxon_p_lt_v20": round(pv_p1, 6), "branch_wins": wins_p, "calls": calls_p},
            "wall_val36": smw,
        }
        print(
            f"  EXP-4 {pmode:24s}: Virt JA={smv['mean_net_obj']:7.1f} (4T={by_s_v['4topo']:.1f} T1={by_s_v['track1']:.1f} T2={by_s_v['track2']:.1f}, p={pv_p1:.4f}) "
            f"| Wall JA={smw['mean_net_obj']:7.1f} Comp={smw['comp100_count']}/36 P95={smw['pooled_p95_ms']:5.2f}ms",
            flush=True,
        )

    (OUT_CLOSURE_DIR / "literature_falsification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Clean Markdown Report
    md = [
        "# `IF-CORE-V20-LITERATURE-01`: Literature-Driven Mechanism Falsification Report (`EXP-1` to `EXP-4`)",
        "",
        "## 1. EXP-1 (P0): Structural Wait-For Graph + Tarjan SCC (Petri Net Inspired)",
        "",
        "| Mode | Stress-24 Comp | Stress-24 JA ↓ | Alert Eps | VAL-36 Comp | VAL-36 JA ↓ | VAL-36 FPR |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, d in report["exp1_structural_pdg_scc"].items():
        st = d["stress24"]
        md.append(f"| {mode} | {st['comp100_count']}/24 | {st['mean_net_obj']:.1f} | {st['alert_episodes']} | {d['val36_comp100']}/36 | {d['val36_mean_net_obj']:.1f} | {d['val36_fpr_episodes']}/36 |")

    md.extend([
        "",
        "## 2. EXP-2 (P0): Persistent Escape Search Memory $\\mathcal{M}_t$ (Real-Time LaCAM Inspired)",
        "",
        "| Variant | Virtual Stress-24 Comp | Virtual JA ↓ | Duplicates Pruned | Frontier Expansions | Wall Stress-24 Comp | Wall JA ↓ | Wall P95 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for lbl, d in report["exp2_persistent_escape"].items():
        v, w = d["virtual_stress24"], d["wall_stress24"]
        md.append(f"| {lbl} | {v['comp100_count']}/24 | {v['mean_net_obj']:.1f} | {v['duplicates_pruned']} | {v['frontier_expansions']} | {w['comp100_count']}/24 | {w['mean_net_obj']:.1f} | {w['pooled_p95_ms']:.2f} ms |")

    md.extend([
        "",
        "## 3. EXP-3 (P1): Cross-Domain ADDRESS (AAAI 2025 Inspired, `VAL-36`)",
        "",
        "| Mode | Virt Comp | Virt JA ↓ | 4-Topo | Track 1 | Track 2 | p (< N0) | Wall Comp | Wall JA ↓ | Wall P95 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for am, d in report["exp3_cross_domain_address"].items():
        v, w = d["virtual_val36"], d["wall_val36"]
        bs = v["by_suite_net"]
        md.append(f"| {am} | {v['comp100_count']}/36 | {v['mean_net_obj']:.1f} | {bs['4topo']:.1f} | {bs['track1']:.1f} | {bs['track2']:.1f} | {v['wilcoxon_p_lt_n0']:.4f} | {w['comp100_count']}/36 | {w['mean_net_obj']:.1f} | {w['pooled_p95_ms']:.2f} ms |")

    md.extend([
        "",
        "## 4. EXP-4 (P1): Neural-Prior Guided Search (ICAPS 2024 Inspired, `VAL-36`)",
        "",
        "| Mode | Virt Comp | Virt JA ↓ | 4-Topo | Track 1 | Track 2 | p (< V20) | Branch Wins | Wall Comp | Wall JA ↓ | Wall P95 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for pm, d in report["exp4_neural_prior_search"].items():
        v, w = d["virtual_val36"], d["wall_val36"]
        bs = v["by_suite_net"]
        md.append(f"| {pm} | {v['comp100_count']}/36 | {v['mean_net_obj']:.1f} | {bs['4topo']:.1f} | {bs['track1']:.1f} | {bs['track2']:.1f} | {v['wilcoxon_p_lt_v20']:.4f} | {v['branch_wins']} | {w['comp100_count']}/36 | {w['mean_net_obj']:.1f} | {w['pooled_p95_ms']:.2f} ms |")

    md.extend([
        "",
        "## 5. P2 / P3 Interface & Theoretical Feasibility Note (`BTPG` & `PIE`)",
        "- **P2 (`BTPG-max` Dual-Layer Commitment $\\mathcal{K}_t = \\mathcal{K}_t^{\\mathrm{prod}} \\cup \\mathcal{K}_t^{\\mathrm{traffic}}$ & Conditional Advance Plans)**: Maps cleanly onto `ReservationSnapshot` + `proposed_bottleneck_locks` in `amr_sched/cvcr/transaction.py` by tagging future unexecuted edge/node intervals as `swappable_pair` only after an acyclic Wait-For SCC check; deferred to Layer B/C Gazebo SIL verification so V20.1 core remains minimal.",
        "- **P3 (`PIE` Concurrent Planning & Execution)**: Maps onto `PhysicalCVCRCoordinator` by decoupling 35 ms foreground commit from background speculative refinement of uncommitted tasks, strictly accounting total CPU work and requiring matched budget parity for all baselines.",
    ])

    (OUT_CLOSURE_DIR / "literature_falsification_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[LITERATURE EXP-1..4 DONE] Wrote {OUT_CLOSURE_DIR / 'literature_falsification_report.md'}", flush=True)
    return report


if __name__ == "__main__":
    run_literature_falsification()
