"""IF-CORE-RGRO-01 · Task 3: Episode-Grouped Conformal Risk Control (`CRC`),
Non-Exchangeable Weighted CRC (`Farinhas et al., ICLR 2024`), & Risk–Coverage Frontier.

Strictly isolated on quarantined DEV splits (`cal` episodes for calibration, `test_dev`
standard vs `DEV-Stress-24` OOD episodes for honest calibration-gap auditing).

Scientific Framing (No Over-Claiming):
  - Physical safety (collisions, capacity, cargo custody) is 100% guaranteed by deterministic
    `NativeTransactionOwner` + `CVCREvaluator` + `IndependentAuditor` (NEVER attributed to CRC).
  - CRC strictly calibrates *residual harm risk* `L_delta(x_c, x_b) = 1[J(x_c) > J(x_b) + delta]`
    at the episode-group level, and explicitly audits OOD calibration drift under structural shift.

Outputs:
  - `out/rgro/crc_calibration_lock.json` (locked DEV operating point `alpha*`, `tau_crc*`, `tau_nex*`)
  - `out/rgro/deliverable3_risk_coverage_frontier.{json,md}` (Deliverable 3)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from experiments.rgro.dfl_critic import DFLAdvantageEnsemble
from experiments.rgro.rgro_common import (
    OUT_RGRO_DIR,
    assert_dev_quarantine,
    build_dev_stress_24_specs,
    paired_stats,
    summarize_rows,
)
from experiments.rgro.rgro_proposer import make_ladder_config, run_rgro_episode
from experiments.rgro.task1_dfl_closed_loop import HARM_DELTA_JA, load_corpus
from experiments.v20_audit.common import build_val72_specs

ALPHA_GRID: tuple[float, ...] = (0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30)


def extract_snapshot_selection_records(
    snaps: list[dict[str, Any]],
    ensemble: DFLAdvantageEnsemble,
    *,
    lcb_kappa: float = 0.50,
    dfl_tau_gain: float = 0.0,
) -> list[dict[str, Any]]:
    """For each snapshot, pick the top non-anchor residual candidate (if any passes LCB filter)
    and record its harm probability, true +48t delta vs V19 Anchor, harm indicator, and state vector."""
    recs: list[dict[str, Any]] = []
    for s in snaps:
        v_idx = int(s["v19_anchor_idx"])
        X = np.asarray(s["X_rel_v19"], dtype=np.float32)
        j48 = np.asarray(s["j48"], dtype=np.float64)
        jau = np.asarray(s["l0_ja_u"], dtype=np.float64)
        scs = ensemble.score_batch(X, kappa=lcb_kappa)

        # Pick best non-anchor candidate by DFL+JA combined utility (mirroring RGRO Stage 4)
        best_k = None
        best_u = -1e18
        for k, sc in enumerate(scs):
            if k == v_idx:
                continue
            ja_gain = float(jau[v_idx] - jau[k])
            if sc.lcb_advantage < dfl_tau_gain:
                continue
            u = float(sc.lcb_advantage + 0.35 * ja_gain)
            if u > best_u:
                best_u = u
                best_k = k

        if best_k is None:
            # No residual candidate proposed -> trivially abstained (0 harm, 0 gain)
            recs.append(
                {
                    "episode_key": str(s["episode_key"]),
                    "is_stress": bool(s["is_stress"]),
                    "has_proposal": False,
                    "harm_prob": 1.0,
                    "lcb_advantage": -1e9,
                    "true_delta_v19": 0.0,
                    "is_harm": 0.0,
                    "is_gain": 0.0,
                    "h_vec": np.asarray(s["h_vec"], dtype=np.float32),
                }
            )
        else:
            td = float(j48[best_k] - j48[v_idx])
            recs.append(
                {
                    "episode_key": str(s["episode_key"]),
                    "is_stress": bool(s["is_stress"]),
                    "has_proposal": True,
                    "harm_prob": float(scs[best_k].harm_prob),
                    "lcb_advantage": float(scs[best_k].lcb_advantage),
                    "true_delta_v19": td,
                    "is_harm": 1.0 if td > HARM_DELTA_JA else 0.0,
                    "is_gain": 1.0 if td < -HARM_DELTA_JA else 0.0,
                    "h_vec": np.asarray(s["h_vec"], dtype=np.float32),
                }
            )
    return recs


def _group_by_episode(recs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_ep: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        by_ep.setdefault(str(r["episode_key"]), []).append(r)
    return by_ep


def calibrate_episode_grouped_crc(
    cal_recs: list[dict[str, Any]],
    alpha: float,
    *,
    grid: Optional[np.ndarray] = None,
    episode_weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Exact finite-sample Episode-Grouped CRC (Angelopoulos et al. ICLR 2024) and optional
    Relevance-Weighted Non-Exchangeable CRC (Barber et al. 2023; Farinhas et al. ICLR 2024).

    Episode loss `L_e(tau) = mean_{t in S_e} 1[has_prop & p_harm(t) <= tau] * is_harm(t)` in [0, 1]
    is strictly non-decreasing in `tau`.
    """
    by_ep = _group_by_episode(cal_recs)
    ep_keys = sorted(by_ep.keys())
    n_ep = len(ep_keys)
    if n_ep == 0:
        return {"tau": 0.0, "cal_upper_bound": 1.0, "n_episodes": 0}

    weights = [float(episode_weights.get(k, 1.0)) if episode_weights is not None else 1.0 for k in ep_keys]
    w_sum = float(sum(weights))
    taus = grid if grid is not None else np.linspace(0.0, 1.0, 201)

    best_tau = 0.0
    best_ub = 1.0
    for tau in taus:
        t_f = float(tau)
        weighted_loss = 0.0
        for k, w in zip(ep_keys, weights):
            ep_r = by_ep[k]
            l_e = float(
                np.mean(
                    [
                        r["is_harm"] if (r["has_proposal"] and r["harm_prob"] <= t_f) else 0.0
                        for r in ep_r
                    ]
                )
            )
            weighted_loss += w * l_e
        ub = (weighted_loss + 1.0) / (w_sum + 1.0)
        if ub <= float(alpha) + 1e-12:
            best_tau = t_f
            best_ub = float(ub)
    return {
        "tau": round(best_tau, 4),
        "cal_upper_bound": round(best_ub, 4),
        "n_episodes": n_ep,
        "effective_weight_sum": round(w_sum, 3),
    }


def compute_non_exchangeable_weights(
    cal_recs: list[dict[str, Any]],
    target_recs: list[dict[str, Any]],
    *,
    bandwidth: float = 3.0,
) -> dict[str, float]:
    """Compute bounded RBF relevance weights `w_e in (0, 1]` between each calibration episode and target distribution."""
    cal_by_ep = _group_by_episode(cal_recs)
    if not target_recs:
        return {k: 1.0 for k in cal_by_ep}
    target_mean_h = np.mean([r["h_vec"] for r in target_recs], axis=0)
    w_out: dict[str, float] = {}
    denom = 2.0 * float(bandwidth) * float(bandwidth)
    for k, ep_r in cal_by_ep.items():
        ep_h = np.mean([r["h_vec"] for r in ep_r], axis=0)
        dist2 = float(np.sum((ep_h - target_mean_h) ** 2))
        w_out[k] = float(max(0.05, math.exp(-dist2 / max(1e-6, denom))))
    return w_out


def evaluate_gate_on_records(recs: list[dict[str, Any]], tau: float) -> dict[str, float]:
    """Evaluate Coverage, Episode-Mean Empirical Harm Risk, FAR, FRR, Fallback Rate, and Mean Delta vs V19."""
    if not recs:
        return {}
    by_ep = _group_by_episode(recs)
    ep_risks: list[float] = []
    n_tot = len(recs)
    n_acc = 0
    n_harm_acc = 0
    n_gain_tot = 0
    n_gain_rej = 0
    realized_deltas: list[float] = []

    for ep_r in by_ep.values():
        ep_h = []
        for r in ep_r:
            accepted = bool(r["has_proposal"] and r["harm_prob"] <= float(tau))
            if r["is_gain"] > 0.5:
                n_gain_tot += 1
                if not accepted:
                    n_gain_rej += 1
            if accepted:
                n_acc += 1
                if r["is_harm"] > 0.5:
                    n_harm_acc += 1
                ep_h.append(float(r["is_harm"]))
                realized_deltas.append(float(r["true_delta_v19"]))
            else:
                ep_h.append(0.0)
                realized_deltas.append(0.0)
        ep_risks.append(float(np.mean(ep_h)))

    cov = float(n_acc) / float(max(1, n_tot))
    emp_ep_risk = float(np.mean(ep_risks))
    far = float(n_harm_acc) / float(max(1, n_acc)) if n_acc > 0 else 0.0
    frr = float(n_gain_rej) / float(max(1, n_gain_tot)) if n_gain_tot > 0 else 0.0
    return {
        "n_episodes": len(by_ep),
        "n_snapshots": n_tot,
        "coverage_pct": round(100.0 * cov, 2),
        "fallback_to_v19_pct": round(100.0 * (1.0 - cov), 2),
        "empirical_ep_harm_risk": round(emp_ep_risk, 4),
        "conditional_far_pct": round(100.0 * far, 2),
        "missed_gain_frr_pct": round(100.0 * frr, 2),
        "mean_realized_delta_vs_v19": round(float(np.mean(realized_deltas)), 2),
    }


def main() -> None:
    OUT_RGRO_DIR.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(OUT_RGRO_DIR / "dev_counterfactual_corpus.npz", OUT_RGRO_DIR / "dev_counterfactual_meta.json")
    ens_v19 = DFLAdvantageEnsemble.load_npz(OUT_RGRO_DIR / "dfl_critic_v19.npz")

    cal_snaps = [s for s in corpus if s["split"] == "cal"]
    te_std_snaps = [s for s in corpus if s["split"] == "test_dev" and not s["is_stress"]]
    te_str_snaps = [s for s in corpus if s["split"] == "test_dev" and s["is_stress"]]
    cal_std_snaps = [s for s in cal_snaps if not s["is_stress"]]

    print(
        f"=== [Task 3] Episode-Grouped CRC & Non-Exchangeable OOD Calibration Audit "
        f"(cal={len(cal_snaps)}, te_std={len(te_std_snaps)}, te_stress={len(te_str_snaps)}) ===",
        flush=True,
    )

    cal_recs_all = extract_snapshot_selection_records(cal_snaps, ens_v19, lcb_kappa=0.75, dfl_tau_gain=20.0)
    cal_recs_std = extract_snapshot_selection_records(cal_std_snaps, ens_v19, lcb_kappa=0.75, dfl_tau_gain=20.0)
    te_std_recs = extract_snapshot_selection_records(te_std_snaps, ens_v19, lcb_kappa=0.75, dfl_tau_gain=20.0)
    te_str_recs = extract_snapshot_selection_records(te_str_snaps, ens_v19, lcb_kappa=0.75, dfl_tau_gain=20.0)

    # Non-exchangeable weights targeting shifted OOD structural stress distribution (computed using feature-only h_vec!)
    nex_weights_stress = compute_non_exchangeable_weights(cal_recs_all, te_str_recs, bandwidth=3.0)

    frontier_offline: list[dict[str, Any]] = []
    for alpha in ALPHA_GRID:
        # 1) Pooled DEV Episode-Grouped CRC (calibrated on cal_all)
        c_pool = calibrate_episode_grouped_crc(cal_recs_all, alpha)
        # 2) Standard-Only Episode-Grouped CRC (calibrated strictly on standard cal_std -> exposes pure OOD shift!)
        c_std = calibrate_episode_grouped_crc(cal_recs_std, alpha)
        # 3) Relevance-Weighted Non-Exchangeable CRC (Farinhas et al. ICLR 2024)
        c_nex = calibrate_episode_grouped_crc(cal_recs_all, alpha, episode_weights=nex_weights_stress)

        ev_std_pool = evaluate_gate_on_records(te_std_recs, c_pool["tau"])
        ev_str_pool = evaluate_gate_on_records(te_str_recs, c_pool["tau"])
        ev_str_stdcal = evaluate_gate_on_records(te_str_recs, c_std["tau"])
        ev_str_nex = evaluate_gate_on_records(te_str_recs, c_nex["tau"])

        row = {
            "alpha": float(alpha),
            "tau_crc_pooled": c_pool["tau"],
            "tau_crc_std_only": c_std["tau"],
            "tau_nex_weighted": c_nex["tau"],
            "eval_std_pooled_crc": ev_std_pool,
            "eval_stress_pooled_crc": ev_str_pool,
            "eval_stress_std_only_crc": ev_str_stdcal,
            "eval_stress_nex_crc": ev_str_nex,
            "calib_gap_std": round(ev_std_pool["empirical_ep_harm_risk"] - float(alpha), 4),
            "calib_gap_ood_from_stdcal": round(ev_str_stdcal["empirical_ep_harm_risk"] - float(alpha), 4),
            "calib_gap_ood_pooled": round(ev_str_pool["empirical_ep_harm_risk"] - float(alpha), 4),
            "calib_gap_ood_nex": round(ev_str_nex["empirical_ep_harm_risk"] - float(alpha), 4),
        }
        frontier_offline.append(row)
        print(
            f"  alpha={alpha:4.2f} | tau_pool={c_pool['tau']:.3f} tau_nex={c_nex['tau']:.3f} "
            f"| STD cov={ev_std_pool['coverage_pct']:5.1f}% risk={ev_std_pool['empirical_ep_harm_risk']:.3f} (gap={row['calib_gap_std']:+.3f}) "
            f"| OOD(std-cal) risk={ev_str_stdcal['empirical_ep_harm_risk']:.3f} (gap={row['calib_gap_ood_from_stdcal']:+.3f}) "
            f"| OOD(NEX) cov={ev_str_nex['coverage_pct']:5.1f}% risk={ev_str_nex['empirical_ep_harm_risk']:.3f} (gap={row['calib_gap_ood_nex']:+.3f})",
            flush=True,
        )

    # Closed-loop DEV Risk-Coverage Frontier sweep on stratified VAL-24 + DEV-Stress-24 (P1 deterministic clock for clean alpha comparison)
    print("=== [Task 3B] Closed-Loop DEV Risk–Coverage Sweep (VAL-24 + DEV-Stress-24, P1 Clock) ===", flush=True)
    val72 = build_val72_specs()
    val24 = (
        [s for s in val72 if s.get("suite_group") == "4topo"][:12]
        + [s for s in val72 if s.get("suite_group") == "track1"][:6]
        + [s for s in val72 if s.get("suite_group") == "track2"][:6]
    )
    stress24 = build_dev_stress_24_specs()
    assert_dev_quarantine([int(s["seed"]) for s in val24 + stress24], label="Task3_CL_Sweep")

    # Anchor baseline U1 (V19 + PDG2) for paired closed-loop delta across alphas
    u1_cfg = make_ladder_config("U1")
    u1_std_rows = [run_rgro_episode(sp, u1_cfg, use_virtual_clock=True) for sp in val24]
    u1_str_rows = [run_rgro_episode(sp, u1_cfg, use_virtual_clock=True) for sp in stress24]
    u1_std_sm = summarize_rows(u1_std_rows)
    u1_str_sm = summarize_rows(u1_str_rows)
    print(
        f"  [Ref U1_V19_PDG2] VAL-24 MeanJA={u1_std_sm['mean_ja']:.2f} CVaR95={u1_std_sm['cvar95_ja']:.2f} "
        f"| DEV-Stress-24 MeanJA={u1_str_sm['mean_ja']:.2f} CVaR95={u1_str_sm['cvar95_ja']:.2f}",
        flush=True,
    )

    cl_sweep: list[dict[str, Any]] = []
    # Sweep representative operating points covering strict -> permissive (CRC alpha* + fine tau probe)
    tau_08 = next(r["tau_crc_pooled"] for r in frontier_offline if abs(r["alpha"] - 0.08) < 1e-6)
    tau_10 = next(r["tau_crc_pooled"] for r in frontier_offline if abs(r["alpha"] - 0.10) < 1e-6)
    tau_12 = next(r["tau_crc_pooled"] for r in frontier_offline if abs(r["alpha"] - 0.12) < 1e-6)
    sweep_points = [
        ("alpha_0.08", 0.08, tau_08),
        ("alpha_0.10", 0.10, tau_10),
        ("tau_0.06_str", 0.09, 0.06),
        ("tau_0.15_mid", 0.11, 0.15),
        ("alpha_0.12", 0.12, tau_12),
        ("unfiltered_U4", 1.00, 1.00),
    ]
    seen_taus: dict[float, dict[str, Any]] = {}
    for lbl, alp, tau_val in sweep_points:
        t_round = round(float(tau_val), 4)
        if t_round in seen_taus:
            rec = dict(seen_taus[t_round], label=lbl, alpha=float(alp))
            cl_sweep.append(rec)
            continue
        cfg_a = make_ladder_config("U5" if alp < 0.99 else "U4", crc_harm_tau=t_round, dfl_tau_gain=20.0, lcb_kappa=0.75)
        std_rows = [run_rgro_episode(sp, cfg_a, dfl_ensemble=ens_v19, use_virtual_clock=True) for sp in val24]
        str_rows = [run_rgro_episode(sp, cfg_a, dfl_ensemble=ens_v19, use_virtual_clock=True) for sp in stress24]
        sm_std = summarize_rows(std_rows)
        sm_str = summarize_rows(str_rows)
        ps_std = paired_stats(std_rows, u1_std_rows)
        ps_str = paired_stats(str_rows, u1_str_rows)
        tot_evals = sum(int(r["res_evals"]) for r in std_rows + str_rows)
        tot_acc = sum(int(r["res_accepted"]) for r in std_rows + str_rows)
        rec = {
            "label": lbl,
            "alpha": float(alp),
            "tau_crc": t_round,
            "cl_residual_accept_rate_pct": round(100.0 * tot_acc / max(1, tot_evals), 2),
            "val24_summary": sm_std,
            "stress24_summary": sm_str,
            "val24_vs_u1": ps_std,
            "stress24_vs_u1": ps_str,
            "pooled_mean_ja": round(float(np.mean([r["net_obj"] for r in std_rows + str_rows])), 2),
        }
        seen_taus[t_round] = rec
        cl_sweep.append(rec)
        print(
            f"  [{lbl:13s} tau={t_round:.3f}] Acc%={rec['cl_residual_accept_rate_pct']:5.1f}% "
            f"| VAL24={sm_std['mean_ja']:7.2f} (CVaR={sm_std['cvar95_ja']:7.1f}, ΔU1={ps_std['mean_delta']:+6.2f}) "
            f"| STR24={sm_str['mean_ja']:8.2f} (CVaR={sm_str['cvar95_ja']:8.1f}, ΔU1={ps_str['mean_delta']:+7.2f})",
            flush=True,
        )

    # Select locked DEV operating point alpha* strictly on DEV (non-trivial CRC tau <= 0.50 protecting VAL24 CVaR95!)
    non_triv = [r for r in cl_sweep if 0.0 < r["tau_crc"] <= 0.50]
    feasible_pts = [
        r for r in non_triv
        if r["val24_summary"]["cvar95_ja"] <= 1.03 * u1_std_sm["cvar95_ja"] + 1e-6
    ]
    if not feasible_pts:
        feasible_pts = non_triv if non_triv else [r for r in cl_sweep if r["alpha"] <= 0.30]
    best_pt = min(feasible_pts, key=lambda r: (r["pooled_mean_ja"], r["val24_summary"]["cvar95_ja"]))
    lock_payload = {
        "locked_alpha_star": float(best_pt["alpha"]),
        "locked_tau_crc": float(best_pt["tau_crc"]),
        "locked_dfl_tau_gain": 20.0,
        "locked_lcb_kappa": 0.75,
        "selection_rule": "Minimum pooled DEV (VAL24 + DEV-Stress24) Mean J_A among active CRC gates (0 < tau <= 0.50) with VAL24 CVaR95 <= 1.03 * U1",
    }
    (OUT_RGRO_DIR / "crc_calibration_lock.json").write_text(json.dumps(lock_payload, indent=2), encoding="utf-8")
    print(f"  [LOCKED DEV CRC] alpha*={lock_payload['locked_alpha_star']:.2f}, tau_crc={lock_payload['locked_tau_crc']:.4f}", flush=True)

    full_out = {
        "lock": lock_payload,
        "offline_frontier": frontier_offline,
        "closed_loop_u1_ref": {"val24": u1_std_sm, "stress24": u1_str_sm},
        "closed_loop_sweep": cl_sweep,
    }
    (OUT_RGRO_DIR / "deliverable3_risk_coverage_frontier.json").write_text(json.dumps(full_out, indent=2), encoding="utf-8")

    # Markdown Deliverable 3
    md: list[str] = [
        "# Deliverable 3 (`IF-CORE-RGRO-01`): Episode-Grouped & Non-Exchangeable CRC Risk–Coverage Frontier",
        "",
        "> **Strict Scope Separation**: Physical safety (collisions, capacity, custody) is enforced 100% deterministically by `NativeTransactionOwner` + `CVCREvaluator` + `IndependentAuditor`. Conformal Risk Control (`CRC`) strictly calibrates **episode-grouped residual harm probability** $L_\\delta(x_c, x_b) = \\mathbb{I}[J(x_c) > J(x_b) + 50]$ relative to the `V19` Anchor $x_b$.",
        "",
        f"- **Locked DEV Operating Point**: $\\alpha^* = {lock_payload['locked_alpha_star']:.2f}$, $\\hat{{\\tau}}_{{\\mathrm{{CRC}}}} = {lock_payload['locked_tau_crc']:.4f}$ (`dfl_tau_gain={lock_payload['locked_dfl_tau_gain']:.1f}`, `lcb_kappa={lock_payload['locked_lcb_kappa']:.2f}`).",
        "",
        "## 1. Offline Episode-Grouped CRC vs. Non-Exchangeable Weighted CRC (`Farinhas et al., ICLR 2024`)",
        "",
        "| $\\alpha$ | $\\hat{\\tau}_{\\mathrm{pool}}$ | $\\hat{\\tau}_{\\mathrm{std}}$ | $\\hat{\\tau}_{\\mathrm{NEX}}$ | STD Cov% | STD Emp Risk (Gap) | OOD Risk from STD-Cal (Gap) | OOD Pooled-Cal Risk (Gap) | OOD NEX-Weighted Cov% / Risk (Gap) |",
        "| ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: | :---: |",
    ]
    for r in frontier_offline:
        es = r["eval_std_pooled_crc"]
        eo_s = r["eval_stress_std_only_crc"]
        eo_p = r["eval_stress_pooled_crc"]
        eo_n = r["eval_stress_nex_crc"]
        md.append(
            f"| {r['alpha']:.2f} | {r['tau_crc_pooled']:.3f} | {r['tau_crc_std_only']:.3f} | {r['tau_nex_weighted']:.3f} | "
            f"{es['coverage_pct']:.1f}% | {es['empirical_ep_harm_risk']:.3f} ({r['calib_gap_std']:+.3f}) | "
            f"{eo_s['empirical_ep_harm_risk']:.3f} ({r['calib_gap_ood_from_stdcal']:+.3f}) | "
            f"{eo_p['empirical_ep_harm_risk']:.3f} ({r['calib_gap_ood_pooled']:+.3f}) | "
            f"{eo_n['coverage_pct']:.1f}% / {eo_n['empirical_ep_harm_risk']:.3f} ({r['calib_gap_ood_nex']:+.3f}) |"
        )

    md += [
        "",
        "### Detailed FAR / FRR / Fallback-to-`V19` Tradeoff (Pooled Episode-Grouped CRC on `test_dev` Standard)",
        "",
        "| $\\alpha$ | $\\hat{\\tau}$ | Coverage% | Fallback-to-`V19`% | Cond. Harm FAR% | Missed-Gain FRR% | Mean Realized $\\Delta J_{48}$ vs `V19` |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in frontier_offline:
        es = r["eval_std_pooled_crc"]
        md.append(
            f"| {r['alpha']:.2f} | {r['tau_crc_pooled']:.3f} | {es['coverage_pct']:.1f}% | {es['fallback_to_v19_pct']:.1f}% | "
            f"{es['conditional_far_pct']:.1f}% | {es['missed_gain_frr_pct']:.1f}% | {es['mean_realized_delta_vs_v19']:+.2f} |"
        )

    md += [
        "",
        "## 2. Closed-Loop DEV Risk–Coverage Sweep (`VAL-24` + `DEV-Stress-24`, Deterministic `P1` Clock)",
        "",
        "| Operating Point | $\\tau_{\\mathrm{CRC}}$ | Res Accept% | `VAL-24` Mean `J_A` | `VAL-24` `CVaR_95` | `VAL-24` $\\Delta$ vs `U1` | `DEV-Stress-24` Mean `J_A` | `DEV-Stress-24` `CVaR_95` | `DEV-Stress-24` $\\Delta$ vs `U1` |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| `Ref: U1 (V19+PDG2)` | — | 0.0% | {u1_std_sm['mean_ja']:.2f} | {u1_std_sm['cvar95_ja']:.2f} | 0.00 | {u1_str_sm['mean_ja']:.2f} | {u1_str_sm['cvar95_ja']:.2f} | 0.00 |",
    ]
    for r in cl_sweep:
        s1 = r["val24_summary"]
        s2 = r["stress24_summary"]
        p1 = r["val24_vs_u1"]
        p2 = r["stress24_vs_u1"]
        md.append(
            f"| `{r['label']}` | {r['tau_crc']:.3f} | {r['cl_residual_accept_rate_pct']:.1f}% | "
            f"{s1['mean_ja']:.2f} | {s1['cvar95_ja']:.2f} | {p1['mean_delta']:+.2f} ({p1['wins']}/{p1['ties']}/{p1['losses']}) | "
            f"{s2['mean_ja']:.2f} | {s2['cvar95_ja']:.2f} | {p2['mean_delta']:+.2f} ({p2['wins']}/{p2['ties']}/{p2['losses']}) |"
        )

    (OUT_RGRO_DIR / "deliverable3_risk_coverage_frontier.md").write_text("\n".join(md), encoding="utf-8")
    print("=== [Task 3] COMPLETED! Wrote Deliverable 3 -> out/rgro/deliverable3_risk_coverage_frontier.{json,md} ===", flush=True)


if __name__ == "__main__":
    main()
