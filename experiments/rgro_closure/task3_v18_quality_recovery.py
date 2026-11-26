"""IF-CORE-RGRO-CLOSURE-01 · Task 3 (P1): V18 Typical/Static Quality Recovery.

Two-part scientific investigation STRICTLY on quarantined DEV suites (NO Cohort A-F leakage,
NO G1-54 tuning, ZERO falsified unconstrained `U2` mid-episode task-swap operator):
  - Part A (`Oracle Headroom Audit`): evaluates counterfactual `{Anchor, O1, O2, O3}` choices at
    opening & mid-episode snapshots across `VAL-Static-35`, `TRAIN-Static-16`, and low-fault
    `VAL-72` to quantify exact `+48t` & `J_full` oracle headroom over `V19 Anchor` and `V18_EA_JOINT`.
  - Part B (`Incremental Closed-Loop DEV Ablation`): evaluates
      `V18_EA_JOINT`, `V20_1_Patched`, `U4_Base`, `U4+O1`, `U4+O1+O2`, `U4+O1+O2+O3`
    on `VAL-Static-35`, `TRAIN-Static-16`, `VAL-72`, and `DEV-Stress-24` under `P1` and `P2` clocks.

Outputs:
  - `out/rgro_closure/deliverable3_v18_quality_recovery.{json,md}`
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.baselines.lbbd_cp_cor26 import strip_dynamic_disruptions
from amr_sched.baselines.policy_registry import create_discrete_policy_env, load_internal_cvcr_policy
from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state
from amr_sched.rl.causal_repair_cone import FastJAWKittingSurrogate, observable_op_closure
from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
from amr_sched.rl.v20_najs import _default_task_order_from_prio, _healthy_robots_from_graph, derive_base_op_prio
from experiments.rgro.dfl_critic import DFLAdvantageEnsemble
from experiments.rgro.task2_rgro_ablation import _run_frozen_baseline
from experiments.v20_audit.common import RecordedStep
from experiments.rgro_closure.closure_common import (
    OUT_CLOSURE_DIR,
    assert_closure_dev_quarantine,
    build_dev_stress_24_specs,
    build_dev_train_specs,
    build_val72_specs,
    build_val_static35_specs,
    paired_stats,
    rollout_multi_horizon,
    summarize_rows,
)
from experiments.rgro_closure.patched_v20_1 import V201PatchedProposer
from experiments.rgro_closure.rgro_closure_proposer import (
    ClosureConfig,
    RGROClosureProposer,
    run_closure_episode,
)
from experiments.rgro_closure.v18_recovery_operators import (
    generate_o1_opening_candidates,
    generate_o2_queue_refinement,
    generate_o3_diverse_low_churn,
    is_opening_zero_commitment,
)


def _build_train_static_16() -> list[dict[str, Any]]:
    raw = build_dev_train_specs(max_4topo_seeds=2, max_ws_seeds=2)
    out: list[dict[str, Any]] = []
    # Pick 12 4-topo + 4 workshop nominal/standard static instances
    t4 = [s for s in raw if s.get("suite_group") == "4topo"][:12]
    ws = [s for s in raw if s.get("suite_group") in ("track1", "track2") and s.get("regime") in ("nominal", "kitting_windows")][:4]
    for sp in t4 + ws:
        c = copy.deepcopy(sp)
        c["scenario_tuple"] = strip_dynamic_disruptions(c["scenario_tuple"])
        c["episode_key"] = f"trstatic_{c['episode_key']}"
        c["suite_group"] = f"TRAIN-Static-{c.get('suite_group', '')}"
        out.append(c)
    assert_closure_dev_quarantine([int(s["seed"]) for s in out], label="TRAIN_Static16")
    return out


def _build_oracle_proposal(
    snap: Any,
    prop_holder: RGROClosureProposer,
    op_prio: dict[str, int],
    assign: dict[str, str],
    bays: dict[str, str],
    crews: dict[str, int],
    seq: dict[str, list[str]] | None,
    ev: CVCREvaluator,
) -> CandidateProposal | None:
    er = ev.evaluate(assign, op_priority=op_prio, op_bay_overrides=bays, op_crew_overrides=crews, task_order_override=seq)
    if not er.valid:
        return None
    from amr_sched.rl.v19_hetero_constructor import _build_core_sched_proposal

    clf = classify_commitments(snap)
    has_win = any(bool(getattr(o, "resource_windows", ())) for o in snap.operations)
    return _build_core_sched_proposal(
        policy_id="ORACLE_CAND",
        snapshot=snap,
        clf=clf,
        best_assign=assign,
        best_bays=bays,
        best_crews=crews,
        best_eval=er,
        op_prio=op_prio,
        eval_cnt=1,
        t0=0.0,
        budget_ms=35.0,
        fallback_keep=False,
        task_sequences=seq,
        include_eval_seq_when_unwindowed=(not has_win),
    )


def _run_part_a_oracle_audit(
    oracle_specs: list[dict[str, Any]],
    ens_v19: DFLAdvantageEnsemble,
) -> dict[str, Any]:
    """Audit exact counterfactual Oracle headroom of `{O1, O2, O3}` over `V19 Anchor`."""
    print("  [Task 3A] Running Oracle Headroom Audit on DEV static/typical snapshots...", flush=True)
    prop = RGROClosureProposer(
        ClosureConfig(policy_id="U4_OracleRef", critic_mode="none", enable_o1_opening=False),
        dfl_ensemble=ens_v19,
    )
    pgmpo = prop._ensure_pgmpo()

    records: list[dict[str, Any]] = []
    with virtual_clock_scope(0.0):
        for sp in oracle_specs:
            _, ctrl, env = create_discrete_policy_env(
                "V19_ADAPTIVE_RL", seed=int(sp["seed"]), max_steps=int(sp["max_steps"]), use_rust=True, p2_cfg=None
            )
            obs, info = env.reset(seed=int(sp["seed"]), options={"scenario_tuple": copy.deepcopy(sp["scenario_tuple"])})
            prop.reset_episode_state()
            traces: list[RecordedStep] = []
            captured: list[tuple[int, int, Any]] = []

            def _rec_prop(snap_in: Any, net_in: Any, *, budget_ms: float = 35.0) -> CandidateProposal:
                p_out = prop(snap_in, net_in, budget_ms=budget_ms)
                traces.append(
                    RecordedStep(
                        step_before=int(snap_in.tick),
                        action=2,
                        triggered_replan=True,
                        proposal=copy.deepcopy(p_out),
                        propose_ms=0.0,
                    )
                )
                return p_out

            env.external_proposer = _rec_prop
            tm = tr = False
            sample_ticks = {0, 30, 75, 130}
            while not (tm or tr):
                tick = int(env.current_step)
                feats = compute_expert_v9_state(
                    env.sim, env.tracker, last_replan_tick=env.last_replan_tick,
                    recent_conflict_ticks=env.recent_conflict_ticks, decision_logs=env.decision_logs,
                )
                s_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
                act = int(ctrl.select_action(feats, s_info, sim=env.sim))
                if tick in sample_ticks and (tick == 0 or act != 0):
                    snap_cap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
                    captured.append((tick, max(1, act), snap_cap))
                nb = len(traces)
                obs, _, tm, tr, info = env.step(act)
                if len(traces) > nb:
                    traces[-1].action = int(act)
                else:
                    traces.append(RecordedStep(step_before=tick, action=int(act), triggered_replan=False, proposal=None))

            net_obj = env.sim.network
            for tick, act_t, snap in captured:
                clf = classify_commitments(snap)
                graph = extract_v19_hetero_graph(snap, net_obj, clf=clf)
                if not graph.rel_tids:
                    continue
                ev = CVCREvaluator(net_obj, snap, alpha_makespan=3.0, lambda_change=15.0, use_rust=True, mode="B2", seq_diff_b2=True)
                base_prio = derive_base_op_prio(snap)
                anc_prop = prop(snap, net_obj, budget_ms=35.0)
                anc_res = rollout_multi_horizon(
                    sp, traces, tick, anc_prop, action_at_target=act_t,
                    horizons=(48,), full_continuation=(tick == 0),
                )
                anc_48 = float(anc_res["j_48"])
                anc_full = float(anc_res.get("j_full", anc_48))

                anc_asn = dict(anc_prop.proposed_assignment or graph.base_assign)
                anc_bays = dict(anc_prop.proposed_bays or graph.base_bays)
                anc_crews = dict(anc_prop.proposed_crews or graph.base_crews)
                anc_prio = dict(anc_prop.proposed_op_priorities or base_prio)
                anc_ev = ev.evaluate(anc_asn, op_priority=anc_prio, op_bay_overrides=anc_bays, op_crew_overrides=anc_crews)
                fast_surr = FastJAWKittingSurrogate(net_obj, snap, ev, graph, lambda_change=15.0)
                healthy = _healthy_robots_from_graph(graph)
                fresh_ops = {o.op_id for o in snap.operations if not o.started}
                bay_d = {graph.bay_ids[i]: float(graph.bay_detour_norm[i]) * 15.0 for i in range(len(graph.bay_ids))}

                cands_by_op: dict[str, list[Any]] = {"O1": [], "O2": [], "O3": []}
                if is_opening_zero_commitment(snap):
                    cands_by_op["O1"] = generate_o1_opening_candidates(
                        snap, net_obj, clf, graph, fast_surr, base_prio, bay_d, healthy, fresh_ops, pgmpo, max_candidates=2
                    )
                cands_by_op["O2"] = generate_o2_queue_refinement(
                    snap, graph, fast_surr, anc_asn, anc_bays, anc_crews, anc_prio, anc_ev if anc_ev.valid else None
                )
                cands_by_op["O3"] = generate_o3_diverse_low_churn(
                    snap, graph, fast_surr, anc_asn, anc_bays, anc_crews, anc_prio,
                    anc_ev if anc_ev.valid else None, healthy, fresh_ops
                )

                best_48 = {"Anchor": anc_48, "O1": anc_48, "O2": anc_48, "O3": anc_48}
                best_full = {"Anchor": anc_full, "O1": anc_full, "O2": anc_full, "O3": anc_full}
                for op_tag, clist in cands_by_op.items():
                    for _, ca, cb, cc, cp, csq, _ in clist[:2]:
                        pr_c = _build_oracle_proposal(snap, prop, cp, ca, cb, cc, csq, ev)
                        if pr_c is None:
                            continue
                        rr = rollout_multi_horizon(
                            sp, traces, tick, pr_c, action_at_target=act_t,
                            horizons=(48,), full_continuation=(tick == 0),
                        )
                        best_48[op_tag] = min(best_48[op_tag], float(rr["j_48"]))
                        best_full[op_tag] = min(best_full[op_tag], float(rr.get("j_full", rr["j_48"])))

                oracle_48 = min(best_48.values())
                oracle_full = min(best_full.values())
                records.append({
                    "episode_key": sp["episode_key"],
                    "tick": tick,
                    "anchor_48": anc_48,
                    "oracle_48": oracle_48,
                    "gain_o1_48": anc_48 - best_48["O1"],
                    "gain_o2_48": anc_48 - best_48["O2"],
                    "gain_o3_48": anc_48 - best_48["O3"],
                    "gain_oracle_48": anc_48 - oracle_48,
                    "anchor_full": anc_full,
                    "oracle_full": oracle_full,
                    "gain_o1_full": anc_full - best_full["O1"],
                    "gain_o2_full": anc_full - best_full["O2"],
                    "gain_o3_full": anc_full - best_full["O3"],
                    "gain_oracle_full": anc_full - oracle_full,
                })

    open_recs = [r for r in records if r["tick"] == 0]
    mid_recs = [r for r in records if r["tick"] > 0]
    summary = {
        "n_snapshots": len(records),
        "n_opening_snaps": len(open_recs),
        "n_mid_snaps": len(mid_recs),
        "mean_anchor_48": float(np.mean([r["anchor_48"] for r in records])) if records else 0.0,
        "mean_oracle_48": float(np.mean([r["oracle_48"] for r in records])) if records else 0.0,
        "mean_gain_oracle_48": float(np.mean([r["gain_oracle_48"] for r in records])) if records else 0.0,
        "mean_gain_o1_open_full": float(np.mean([r["gain_o1_full"] for r in open_recs])) if open_recs else 0.0,
        "mean_gain_oracle_open_full": float(np.mean([r["gain_oracle_full"] for r in open_recs])) if open_recs else 0.0,
        "o1_win_rate_open_full": float(np.mean([r["gain_o1_full"] > 1.0 for r in open_recs])) if open_recs else 0.0,
        "o2_win_rate_48": float(np.mean([r["gain_o2_48"] > 1.0 for r in records])) if records else 0.0,
        "o3_win_rate_48": float(np.mean([r["gain_o3_48"] > 1.0 for r in records])) if records else 0.0,
    }
    print(
        f"  [Task 3A Done] Snaps={summary['n_snapshots']} (Open={summary['n_opening_snaps']}) "
        f"| Open O1 Full Gain={summary['mean_gain_o1_open_full']:+.2f} (Win={summary['o1_win_rate_open_full']*100:.1f}%) "
        f"| +48t Oracle Gain={summary['mean_gain_oracle_48']:+.2f} (O2Win={summary['o2_win_rate_48']*100:.1f}%, O3Win={summary['o3_win_rate_48']*100:.1f}%)",
        flush=True,
    )
    return {"summary": summary, "rows": records}


def main() -> None:
    OUT_CLOSURE_DIR.mkdir(parents=True, exist_ok=True)
    ens_v19 = DFLAdvantageEnsemble.load_npz(ROOT / "out" / "rgro" / "dfl_critic_v19.npz")

    val_static35 = build_val_static35_specs()
    train_static16 = _build_train_static_16()
    val72 = build_val72_specs()
    stress24 = build_dev_stress_24_specs()
    assert_closure_dev_quarantine(
        [int(s["seed"]) for s in val_static35 + train_static16 + val72 + stress24], label="Task3_QualityRecovery"
    )

    val72_lowfault = [s for s in val72 if s.get("regime") in ("standard", "nominal")][:16]
    part_a = _run_part_a_oracle_audit(val_static35[:14] + train_static16[:10] + val72_lowfault[:8], ens_v19)

    arms: list[tuple[str, str, Any]] = [
        ("V18_EA_JOINT", "frozen", "V18_EA_JOINT"),
        ("V20_1_Patched", "patched", None),
        ("U4_Base", "closure", ClosureConfig(policy_id="U4_Base", critic_mode="dfl_lcb")),
        ("U4_O1", "closure", ClosureConfig(policy_id="U4_O1", critic_mode="dfl_lcb", enable_o1_opening=True)),
        ("U4_O1_O2", "closure", ClosureConfig(policy_id="U4_O1_O2", critic_mode="dfl_lcb", enable_o1_opening=True, enable_o2_queue=True)),
        ("U4_O1_O2_O3", "closure", ClosureConfig(policy_id="U4_O1_O2_O3", critic_mode="dfl_lcb", enable_o1_opening=True, enable_o2_queue=True, enable_o3_diverse=True)),
    ]
    suites = [
        ("VAL_Static35", val_static35),
        ("TRAIN_Static16", train_static16),
        ("VAL72", val72),
        ("DEV_Stress24", stress24),
    ]

    ablation: dict[str, Any] = {}
    for clk, use_vc in (("P1_Virtual", True), ("P2_WallClock", False)):
        ablation[clk] = {}
        for s_name, specs in suites:
            rows_map: dict[str, list[dict[str, Any]]] = {}
            for a_name, kind, obj in arms:
                t0 = time.perf_counter()
                if kind == "frozen":
                    rows = [_run_frozen_baseline(sp, str(obj), use_virtual_clock=use_vc) for sp in specs]
                elif kind == "patched":
                    rows = [
                        run_closure_episode(
                            sp, proposer_factory=lambda s=sp: V201PatchedProposer(seed=int(s["seed"])),
                            base_env_policy="V19_ADAPTIVE_RL", use_pdg2=True, use_virtual_clock=use_vc,
                        )
                        for sp in specs
                    ]
                else:
                    cfg_o: ClosureConfig = obj
                    rows = [
                        run_closure_episode(
                            sp, proposer_factory=lambda c=cfg_o: RGROClosureProposer(c, dfl_ensemble=ens_v19),
                            base_env_policy="V19_ADAPTIVE_RL", use_pdg2=True, use_virtual_clock=use_vc,
                        )
                        for sp in specs
                    ]
                rows_map[a_name] = rows
                sm = summarize_rows(rows)
                print(
                    f"  [Task 3B {clk:12s} | {s_name:14s}] {a_name:13s} | Comp={sm['comp100_count']:2d}/{sm['n']} "
                    f"ExpW={sm['expired_windows_total']} MeanJA={sm['mean_ja']:8.2f} Med={sm['median_ja']:7.2f} "
                    f"CVaR95={sm['cvar95_ja']:8.2f} P99={sm['p99_ms']:5.2f}ms ({time.perf_counter()-t0:.1f}s)",
                    flush=True,
                )
            ablation[clk][s_name] = {
                "summary": {a: summarize_rows(rows_map[a]) for a, _, _ in arms},
                "contrasts": {
                    f"{a}_vs_U4_Base": paired_stats(rows_map[a], rows_map["U4_Base"])
                    for a in ("U4_O1", "U4_O1_O2", "U4_O1_O2_O3")
                }
                | {
                    f"{a}_vs_V18": paired_stats(rows_map[a], rows_map["V18_EA_JOINT"])
                    for a in ("U4_Base", "U4_O1", "U4_O1_O2", "U4_O1_O2_O3")
                },
            }

    payload = {"part_a_oracle_audit": part_a["summary"], "part_b_ablation": ablation}
    (OUT_CLOSURE_DIR / "deliverable3_v18_quality_recovery.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    s_a = part_a["summary"]
    md = [
        "# Deliverable 3 (`IF-CORE-RGRO-CLOSURE-01`): V18 Typical/Static Quality Recovery Report",
        "",
        "> Strictly quarantined on DEV (`VAL-Static-35`, `TRAIN-Static-16`, `VAL-72`, `DEV-Stress-24`). Zero `U2` unconstrained mid-episode task swaps; zero `Cohort A-F` or `G1-54` tuning.",
        "",
        "## 1. Part A — DEV Counterfactual Oracle Headroom (`{Anchor, O1, O2, O3}`)",
        "",
        f"- Snapshots audited: `{s_a['n_snapshots']}` (`{s_a['n_opening_snaps']}` opening + `{s_a['n_mid_snaps']}` mid-episode)",
        f"- Opening `O1` full-continuation `J_full` mean gain over Anchor: **`{s_a['mean_gain_o1_open_full']:+.2f}`** (win rate `{s_a['o1_win_rate_open_full']*100:.1f}%`; combined opening Oracle gain `{s_a['mean_gain_oracle_open_full']:+.2f}`)",
        f"- Overall `+48t` Oracle mean improvement over Anchor: **`{s_a['mean_gain_oracle_48']:+.2f}`** (`O2` win rate `{s_a['o2_win_rate_48']*100:.1f}%`, `O3` win rate `{s_a['o3_win_rate_48']*100:.1f}%`)",
        "",
        "## 2. Part B — Incremental Closed-Loop DEV Ablation (`U4_Base -> U4+O1 -> U4+O1+O2 -> U4+O1+O2+O3`)",
        "",
    ]
    for clk in ("P1_Virtual", "P2_WallClock"):
        md.append(f"### {clk}")
        md.append("")
        for s_name, _ in suites:
            blk = ablation[clk][s_name]
            md.append(f"#### {s_name} ({clk})")
            md.append("")
            md.append("| Variant | Comp100 | ExpWin | Mean `J_A` | Med `J_A` | `CVaR_95` | `P95/P99/Max` (ms) | `Δ vs U4_Base` |")
            md.append("| :--- | :---: | ---: | ---: | ---: | ---: | :---: | :--- |")
            for a, _, _ in arms:
                sm = blk["summary"][a]
                if f"{a}_vs_U4_Base" in blk["contrasts"]:
                    ct = blk["contrasts"][f"{a}_vs_U4_Base"]
                    ct_s = f"{ct['mean_delta']:+.2f} ({ct['rel_delta_pct']:+.2f}%, p={ct['wilcoxon_p_2sided']:.4f})"
                else:
                    ct_s = "—"
                md.append(
                    f"| `{a}` | {sm['comp100_count']}/{sm['n']} | {sm['expired_windows_total']} | {sm['mean_ja']:.2f} | "
                    f"{sm['median_ja']:.2f} | {sm['cvar95_ja']:.2f} | {sm['p95_ms']:.2f}/{sm['p99_ms']:.2f}/{sm['max_ms']:.2f} | {ct_s} |"
                )
            md.append("")

    (OUT_CLOSURE_DIR / "deliverable3_v18_quality_recovery.md").write_text("\n".join(md), encoding="utf-8")
    print("=== [Task 3] COMPLETED! Wrote out/rgro_closure/deliverable3_v18_quality_recovery.{json,md} ===", flush=True)


if __name__ == "__main__":
    main()
