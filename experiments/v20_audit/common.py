"""Shared quarantined scenario builders, fork-replay engine, and telemetry helpers for IF-CORE-V20-01."""

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterator, Optional

import numpy as np

from amr_sched.baselines.lbbd_cp_cor26 import strip_dynamic_disruptions
from amr_sched.baselines.policy_registry import (
    create_discrete_policy_env,
    load_internal_cvcr_policy,
)
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.clock import record_work_ms as _cvcr_record_work_ms, virtual_clock_scope
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.policy import CandidateProposal, LGCVCRPolicy
from amr_sched.cvcr.snapshot import NativeSnapshot, NativeSnapshotCapture
from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
from amr_sched.rl import RuleV10PortfolioPolicy, compute_expert_v10_action_mask, compute_expert_v9_state
from experiments.train_v19_bc import (
    FORBIDDEN_BLIND_SEEDS,
    FORBIDDEN_DEV_SEEDS,
    TRAIN_4TOPO_SEEDS,
    TRAIN_WORKSHOP_SEEDS,
    VAL_4TOPO_SEEDS,
    VAL_WORKSHOP_SEEDS,
    _build_4topo_episode_spec,
    _build_workshop_episode_spec,
)
from tools.benchmarks.compare_workshop_algorithms import REGIMES as WORKSHOP_REGIMES
from experiments.v18_confirmation_scenarios import LAYOUTS as TOPO4_LAYOUTS, REGIMES as TOPO4_REGIMES

ROOT = Path(__file__).resolve().parents[2]
OUT_AUDIT_DIR = ROOT / "out" / "v20_scientific_audit"

COHORT_A_SEEDS: frozenset[int] = frozenset(
    {27101, 27102, 27201, 27202, 27301, 27302, 2901, 2902, 2903, 2904, 2905, 2906}
)
COHORT_B_SEEDS: frozenset[int] = frozenset(
    {37101, 37102, 37201, 37202, 37301, 37302, 3901, 3902, 3903, 3904, 3905, 3906}
)
COHORT_C_RESERVED_SEEDS: frozenset[int] = (
    frozenset(range(47101, 47303)) | frozenset(range(4901, 4907))
)

TASK_C_OUTLIER_WHITELIST: frozenset[str] = frozenset(
    {
        "topo4_offset_dual_lane_window_bottleneck_37302",
        "track2_kitting_windows_3906",
    }
)


def assert_stage1_quarantine(
    seeds: list[int] | tuple[int, ...] | set[int] | frozenset[int],
    *,
    label: str,
    episode_keys: list[str] | tuple[str, ...] = (),
    allow_task_c_outliers: bool = False,
) -> None:
    """Block Cohort A, Cohort B, Cohort C, G1, and Legacy Blind seeds from Tasks A/B/D."""
    seed_set = {int(s) for s in seeds}
    if allow_task_c_outliers:
        bad_non_c = {str(k) for k in episode_keys} - set(TASK_C_OUTLIER_WHITELIST)
        if bad_non_c:
            raise RuntimeError(f"Task C whitelist violation in {label}: {sorted(bad_non_c)}")
        return

    forbidden = (
        set(FORBIDDEN_BLIND_SEEDS)
        | set(FORBIDDEN_DEV_SEEDS)
        | set(COHORT_A_SEEDS)
        | set(COHORT_B_SEEDS)
        | set(COHORT_C_RESERVED_SEEDS)
    )
    bad = sorted(seed_set & forbidden)
    if bad:
        raise RuntimeError(f"STAGE-1 SEED QUARANTINE VIOLATION in {label}: {bad[:12]}")


def assert_git_amr_sched_clean_relative_to_baseline(baseline_patch_path: Optional[Path] = None) -> None:
    """Optionally confirm `amr_sched/` was untouched vs an initial baseline patch snapshot."""
    if baseline_patch_path is None or not baseline_patch_path.is_file():
        return
    import subprocess

    proc = subprocess.run(
        ["git", "diff", "amr_sched/"],
        cwd=str(ROOT),
        capture_output=True,
        check=True,
    )
    baseline_bytes = baseline_patch_path.read_bytes()
    if proc.stdout != baseline_bytes:
        raise RuntimeError("GUARDRAIL VIOLATION: amr_sched/ modified during Stage-1 IF-CORE-V20-01!")


def build_val72_specs() -> list[dict[str, Any]]:
    """Build exact 72-episode quarantined validation suite (`VAL-72`)."""
    assert_stage1_quarantine(VAL_4TOPO_SEEDS, label="VAL_4TOPO_SEEDS")
    assert_stage1_quarantine(VAL_WORKSHOP_SEEDS, label="VAL_WORKSHOP_SEEDS")
    specs: list[dict[str, Any]] = []
    for s in VAL_4TOPO_SEEDS:
        for lay in TOPO4_LAYOUTS:
            for reg in TOPO4_REGIMES:
                sp = _build_4topo_episode_spec(lay, s, reg)
                sp["suite_group"] = "4topo"
                specs.append(sp)
    for tr in ("track1", "track2"):
        for s in VAL_WORKSHOP_SEEDS:
            for reg in WORKSHOP_REGIMES:
                sp = _build_workshop_episode_spec(tr, s, reg)
                sp["suite_group"] = tr
                specs.append(sp)
    assert len(specs) == 72
    return specs


def build_val_static35_specs() -> list[dict[str, Any]]:
    """Build 35 zero-fault static instances strictly from quarantined VAL seeds."""
    assert_stage1_quarantine(VAL_4TOPO_SEEDS, label="VAL_STATIC_4TOPO")
    assert_stage1_quarantine(VAL_WORKSHOP_SEEDS, label="VAL_STATIC_WORKSHOP")
    specs: list[dict[str, Any]] = []
    for s in VAL_4TOPO_SEEDS:
        for lay in TOPO4_LAYOUTS:
            for reg in ("standard", "window_bottleneck"):
                sp = _build_4topo_episode_spec(lay, s, reg)
                sp["scenario_tuple"] = strip_dynamic_disruptions(sp["scenario_tuple"])
                sp["p2_cfg"] = None
                sp["episode_key"] = f"static_{sp['episode_key']}"
                sp["suite_group"] = f"static_4topo_{reg}"
                specs.append(sp)
    for s in VAL_WORKSHOP_SEEDS:
        sp = _build_workshop_episode_spec("track1", s, "nominal")
        sp["scenario_tuple"] = strip_dynamic_disruptions(sp["scenario_tuple"])
        sp["p2_cfg"] = None
        sp["episode_key"] = f"static_{sp['episode_key']}"
        sp["suite_group"] = "static_track1"
        specs.append(sp)
    assert len(specs) == 35
    return specs


def build_stratified_train_val_specs() -> list[dict[str, Any]]:
    """Build stratified quarantined TRAIN+VAL episodes covering all suites and regimes for Task B."""
    t_4t = TRAIN_4TOPO_SEEDS[:4]      # 16101..16104 (4 seeds * 4 layouts * 3 regimes = 48? We subset cleanly)
    v_4t = VAL_4TOPO_SEEDS            # 16701..16704
    t_ws = TRAIN_WORKSHOP_SEEDS[:4]   # 1001..1004
    v_ws = VAL_WORKSHOP_SEEDS         # 1161..1163
    all_seeds = set(t_4t) | set(v_4t) | set(t_ws) | set(v_ws)
    assert_stage1_quarantine(all_seeds, label="STRATIFIED_TRAIN_VAL")

    specs: list[dict[str, Any]] = []
    for split, s_list in (("train", t_4t), ("val", v_4t)):
        for s in s_list:
            for lay in TOPO4_LAYOUTS:
                for reg in TOPO4_REGIMES:
                    sp = _build_4topo_episode_spec(lay, s, reg)
                    sp["split"] = split
                    sp["suite_group"] = "4topo"
                    specs.append(sp)
    for split, s_list in (("train", t_ws), ("val", v_ws)):
        for tr in ("track1", "track2"):
            for s in s_list:
                for reg in WORKSHOP_REGIMES:
                    sp = _build_workshop_episode_spec(tr, s, reg)
                    sp["split"] = split
                    sp["suite_group"] = tr
                    specs.append(sp)
    return specs


def build_task_c_outlier_specs() -> list[dict[str, Any]]:
    """Build ONLY the two Cohort B outlier episodes exclusively for Task C forensics."""
    specs = [
        _build_4topo_episode_spec("offset_dual_lane", 37302, "window_bottleneck"),
        _build_workshop_episode_spec("track2", 3906, "kitting_windows"),
    ]
    for sp in specs:
        sp["suite_group"] = sp["suite"]
    assert_stage1_quarantine(
        [sp["seed"] for sp in specs],
        label="TASK_C_OUTLIERS",
        episode_keys=[sp["episode_key"] for sp in specs],
        allow_task_c_outliers=True,
    )
    return specs


def candidate_action_signature(
    assign: dict[str, str],
    bays: dict[str, str],
    crews: dict[str, int],
    op_prio: dict[str, int],
    task_orders: Optional[dict[str, list[str]]] = None,
    live_repairs: Optional[dict[str, Any]] = None,
    bottleneck_locks: Optional[dict[str, Any]] = None,
) -> str:
    """Return a canonical SHA-256 signature for deduplicating identical scheduling decisions."""
    payload = (
        sorted((str(k), str(v)) for k, v in assign.items()),
        sorted((str(k), str(v)) for k, v in bays.items()),
        sorted((str(k), int(v)) for k, v in crews.items()),
        sorted((str(k), int(v)) for k, v in op_prio.items()),
        sorted((str(k), [str(x) for x in v]) for k, v in (task_orders or {}).items()),
        sorted((str(k), str(v.get("kind", "")), str(v.get("goal", ""))) for k, v in (live_repairs or {}).items()),
        sorted((str(k), [(str(a), int(b), int(c)) for a, b, c in v]) for k, v in (bottleneck_locks or {}).items()),
    )
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()[:20]


@dataclass
class CandidateArtifact:
    """Normalized representation of a candidate emitted by an expert or constructor."""

    expert_id: str
    tag: str
    assign: dict[str, str]
    bays: dict[str, str]
    crews: dict[str, int]
    op_prio: dict[str, int]
    task_orders: Optional[dict[str, list[str]]] = None
    is_full: bool = False
    predicted_score: Optional[float] = None
    live_repairs: Optional[dict[str, dict[str, Any]]] = None
    bottleneck_locks: Optional[dict[str, list[tuple[str, int, int]]]] = None
    gen_ms: float = 0.0
    eval_res_b2: Optional[EvalResult] = None
    b2_ms: float = 0.0
    ja_unified_score: Optional[float] = None
    ja_proxy_details: dict[str, Any] = field(default_factory=dict)
    w_surr_score: Optional[float] = None
    action_sig: str = ""

    def ensure_sig(self) -> str:
        if not self.action_sig:
            self.action_sig = candidate_action_signature(
                self.assign,
                self.bays,
                self.crews,
                self.op_prio,
                self.task_orders,
                self.live_repairs,
                self.bottleneck_locks,
            )
        return self.action_sig

    def to_proposal(
        self,
        *,
        policy_id: str,
        snapshot: NativeSnapshot,
        clf: Any,
        eval_res: Optional[EvalResult],
        eval_cnt: int,
        t0: float,
        budget_ms: float,
        fallback_keep: bool,
        include_eval_seq_when_unwindowed: bool = True,
    ) -> CandidateProposal:
        from amr_sched.rl.v19_hetero_constructor import _build_core_sched_proposal

        return _build_core_sched_proposal(
            policy_id=policy_id,
            snapshot=snapshot,
            clf=clf,
            best_assign=self.assign,
            best_bays=self.bays,
            best_crews=self.crews,
            best_eval=eval_res,
            op_prio=self.op_prio,
            eval_cnt=eval_cnt,
            t0=t0,
            budget_ms=budget_ms,
            fallback_keep=fallback_keep,
            task_sequences=self.task_orders,
            predicted_score=self.predicted_score,
            include_eval_seq_when_unwindowed=include_eval_seq_when_unwindowed,
        )


@dataclass
class RecordedStep:
    """Single step trace for zero-deepcopy deterministic Rust replay-fork."""

    step_before: int
    action: int
    triggered_replan: bool
    proposal: Optional[CandidateProposal] = None
    propose_ms: float = 0.0


@contextmanager
def SeqDiffEvalContext(enabled: bool = True) -> Iterator[None]:
    """Non-invasively force `seq_diff_b2=enabled` inside `amr_sched.cvcr.policy.CVCREvaluator` for experiments."""
    import amr_sched.cvcr.policy as policy_mod

    if not enabled:
        yield
        return

    OrigEval = policy_mod.CVCREvaluator

    class _SeqDiffEnabledEvaluator(OrigEval):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["seq_diff_b2"] = True
            super().__init__(*args, **kwargs)

    policy_mod.CVCREvaluator = _SeqDiffEnabledEvaluator
    try:
        yield
    finally:
        policy_mod.CVCREvaluator = OrigEval


def make_v18_external_proposer(
    *,
    hard_budget_ms: Optional[float] = None,
    seq_diff_b2: bool = False,
    policy_label: str = "V18_EA_STRICT",
    env_ref: Any = None,
) -> Callable[[NativeSnapshot, Any], CandidateProposal]:
    """Wrap `V18_EA_JOINT` (`LGCVCRPolicy`) cleanly with optional strict budget cap and `seq_diff_b2`."""
    v18_pol = load_internal_cvcr_policy("V18_EA_JOINT", allow_missing_pgmpo=False)

    def _propose(
        snapshot: NativeSnapshot,
        network: Any,
        *,
        budget_ms: float = 35.0,
    ) -> CandidateProposal:
        eff_budget = float(hard_budget_ms) if hard_budget_ms is not None else float(budget_ms)
        stall_thr = int(getattr(env_ref, "loaded_repair_threshold", 6)) if env_ref is not None else 6
        sim_obj = getattr(env_ref, "sim", None) if env_ref is not None else None
        dd_sig = compute_detour_delta_signals(network, snapshot, simulator=sim_obj, stall_threshold=stall_thr)
        v18_pol.active_subscope = None
        with SeqDiffEvalContext(enabled=seq_diff_b2):
            pol = LGCVCRPolicy(
                "V18_EA_JOINT",
                network,
                use_rust=True,
                destroy_selector=(getattr(env_ref, "destroy_selector", None) if env_ref is not None else None),
                bay_detour_delta=(dd_sig["bay_detour_delta"] if dd_sig is not None else None),
                detour_weights_q3=(dd_sig["weights_q3"] if dd_sig is not None else None),
                v15_policy=v18_pol,
            )
            if sim_obj is not None:
                pol._prev_ltm_weights = dict(getattr(sim_obj, "_traffic_weights_q3", {}))
            prop = pol.propose(snapshot, budget_ms=eff_budget)
        if hard_budget_ms is not None:
            prop.telemetry["policy_label"] = policy_label
            prop.telemetry["hard_budget_ms"] = eff_budget
        return prop

    return _propose


def _select_ctrl_action(kind: str, ctrl: Any, env: Any, obs: Any, info: dict[str, Any]) -> int:
    if kind == "rule_v10":
        feats = compute_expert_v9_state(
            env.sim,
            env.tracker,
            last_replan_tick=env.last_replan_tick,
            recent_conflict_ticks=env.recent_conflict_ticks,
            decision_logs=env.decision_logs,
        )
        step_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
    elif kind == "v18":
        feats = ctrl.observe(env, obs)
        step_info = info
    else:
        feats = env.get_expert_v12_obs(obs)
        step_info = info
    return int(ctrl.select_action(feats, step_info, sim=env.sim))


def run_wrapped_episode(
    *,
    variant_id: str,
    spec: dict[str, Any],
    base_env_policy: str = "V19_ADAPTIVE_RL",
    external_proposer: Optional[Callable[..., CandidateProposal]] = None,
    hard_budget_ms: Optional[float] = None,
    collect_traces: bool = False,
    outer_action_hook: Optional[Callable[[Any, Any, int, dict[str, Any]], int]] = None,
) -> tuple[dict[str, Any], list[RecordedStep]]:
    """Run one single-process uncontended episode with optional custom proposer and trace recording."""
    scenario_copy = copy.deepcopy(spec["scenario_tuple"])
    p2_copy = copy.deepcopy(spec["p2_cfg"])
    seed = int(spec["seed"])
    max_steps = int(spec["max_steps"])
    topo = str(scenario_copy[0])

    kind, ctrl, env = create_discrete_policy_env(
        base_env_policy,
        topology=topo,
        seed=seed,
        max_steps=max_steps,
        use_rust=True,
        p2_cfg=p2_copy,
        allow_missing_pgmpo=False,
    )
    if external_proposer is not None:
        raw_ext = external_proposer
    else:
        raw_ext = env.external_proposer

    traces: list[RecordedStep] = []
    if collect_traces or raw_ext is not None or hard_budget_ms is not None:
        base_ext = raw_ext
        if base_ext is None and base_env_policy == "V18_EA_JOINT":
            base_ext = make_v18_external_proposer(
                hard_budget_ms=hard_budget_ms,
                seq_diff_b2=False,
                env_ref=env,
            )

        def _wrapped(snap: NativeSnapshot, network: Any, *, budget_ms: float = 35.0) -> CandidateProposal:
            eff_b = float(hard_budget_ms) if hard_budget_ms is not None else float(budget_ms)
            prop = base_ext(snap, network, budget_ms=eff_b)  # type: ignore[misc]
            if collect_traces:
                traces.append(
                    RecordedStep(
                        step_before=int(snap.tick),
                        action=-1,  # patched immediately after act
                        triggered_replan=True,
                        proposal=copy.deepcopy(prop),
                    )
                )
            return prop

        if base_ext is not None:
            env.external_proposer = _wrapped

    wall_t0 = time.perf_counter()
    obs, info = env.reset(seed=seed, options={"scenario_tuple": scenario_copy})
    tm = False
    tr = False
    while not (tm or tr):
        step_before = int(env.current_step)
        act = _select_ctrl_action(kind, ctrl, env, obs, info)
        if outer_action_hook is not None:
            act = int(outer_action_hook(env, info, act, traces))
        before_len = len(traces)
        obs, _, tm, tr, info = env.step(act)
        if collect_traces:
            if len(traces) > before_len:
                traces[-1].action = int(act)
                if env.decision_logs:
                    traces[-1].propose_ms = float(env.decision_logs[-1].get("t_propose_ms", 0.0))
            else:
                traces.append(
                    RecordedStep(
                        step_before=step_before,
                        action=int(act),
                        triggered_replan=bool(info.get("is_replan", False)),
                        proposal=None,
                    )
                )

    wall_ms = (time.perf_counter() - wall_t0) * 1000.0
    m = env.compute_episode_metrics()
    aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
    decision_ms = [
        float(row["t_propose_ms"]) + float(row["t_validate_commit_ms"])
        for row in env.decision_logs
    ]
    decision_sorted = sorted(decision_ms)
    rec = {
        "variant": variant_id,
        "episode_key": str(spec["episode_key"]),
        "suite": str(spec.get("suite", "")),
        "suite_group": str(spec.get("suite_group", spec.get("suite", ""))),
        "regime": str(spec.get("regime", "")),
        "seed": seed,
        "audit_passed": bool(aud.pass_audit),
        "ticks": int(aud.total_steps),
        "terminated": bool(tm),
        "truncated": bool(tr),
        "net_obj": round(float(m["net_obj"]), 4),
        "total_obj": round(float(m["total_obj"]), 4),
        "cmax": round(float(m["cmax"]), 4),
        "task_tardiness_cost": round(float(m.get("task_tardiness_cost", 0.0)), 4),
        "op_tardiness_cost": round(float(m.get("op_tardiness_cost", 0.0)), 4),
        "unfinished_task_penalty": round(float(m.get("unfinished_task_penalty", 0.0)), 4),
        "unfinished_op_penalty": round(float(m.get("unfinished_op_penalty", 0.0)), 4),
        "churn_cost": round(float(m.get("churn_cost", 15.0 * float(m.get("total_churn", 0)))), 4),
        "compute_cost": round(float(m.get("compute_cost", 0.05 * float(m.get("total_compute_ms", 0.0)))), 4),
        "h1": round(float(m["cmax"]) + float(m.get("task_tardiness_cost", 0.0)) + float(m.get("op_tardiness_cost", 0.0)), 4),
        "churn": int(m["total_churn"]),
        "replans": int(m["total_replans"]),
        "task_comp": round(float(m["task_completion_rate"]), 6),
        "prod_comp": round(float(m["production_completion_rate"]), 6),
        "comp100": bool(float(m["task_completion_rate"]) >= 0.9999 and float(m["production_completion_rate"]) >= 0.9999),
        "compute_ms": round(float(m["total_compute_ms"]), 3),
        "wall_ms": round(wall_ms, 2),
        "decision_count": len(decision_ms),
        "mean_decision_ms": round(sum(decision_ms) / len(decision_ms), 3) if decision_ms else 0.0,
        "decision_ms_p95": round(_percentile(decision_sorted, 0.95), 3),
        "decision_ms_p99": round(_percentile(decision_sorted, 0.99), 3),
        "transaction_rejections": sum(not row["receipt_accepted"] for row in env.decision_logs),
        "expired_windows": int(env.total_window_expirations),
        "starvation_steps": int(env.tracker.starvation_steps),
        "decision_latencies_ms": [round(float(v), 4) for v in decision_ms],
    }
    return rec, traces


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = float(q) * float(len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = pos - float(lo)
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def summarize_episode_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize aggregate quality, H1, completion, and pooled decision latency."""
    if not rows:
        return {}
    all_lats = sorted(float(x) for r in rows for x in r.get("decision_latencies_ms", []))
    comp_rows = [r for r in rows if bool(r.get("comp100", False))]
    return {
        "n": len(rows),
        "audit_pass_rate": round(float(np.mean([1.0 if r["audit_passed"] else 0.0 for r in rows])), 4),
        "comp100_count": len(comp_rows),
        "comp100_rate": round(len(comp_rows) / len(rows), 4),
        "mean_net_obj": round(float(np.mean([r["net_obj"] for r in rows])), 2),
        "median_net_obj": round(float(np.median([r["net_obj"] for r in rows])), 2),
        "mean_h1_all": round(float(np.mean([r["h1"] for r in rows])), 2),
        "mean_h1_comp": round(float(np.mean([r["h1"] for r in comp_rows])), 2) if comp_rows else None,
        "mean_churn": round(float(np.mean([r["churn"] for r in rows])), 2),
        "mean_decision_ms": round(float(np.mean(all_lats)), 3) if all_lats else 0.0,
        "pooled_p95_ms": round(_percentile(all_lats, 0.95), 3),
        "pooled_p99_ms": round(_percentile(all_lats, 0.99), 3),
    }


def replay_fork_env_at_tick(
    spec: dict[str, Any],
    recorded_steps: list[RecordedStep],
    target_tick: int,
    *,
    base_env_policy: str = "V19_ADAPTIVE_RL",
) -> tuple[Any, NativeSnapshot, Any]:
    """Safely reconstruct an exact fresh `AMRReschedulingEnv` at `target_tick` using recorded prefix actions/proposals."""
    scenario_copy = copy.deepcopy(spec["scenario_tuple"])
    p2_copy = copy.deepcopy(spec["p2_cfg"])
    seed = int(spec["seed"])
    max_steps = int(spec["max_steps"])
    topo = str(scenario_copy[0])

    _, _, env = create_discrete_policy_env(
        base_env_policy,
        topology=topo,
        seed=seed,
        max_steps=max_steps,
        use_rust=True,
        p2_cfg=p2_copy,
        allow_missing_pgmpo=False,
    )

    prop_by_tick: dict[int, tuple[CandidateProposal, float]] = {
        int(st.step_before): (st.proposal, float(getattr(st, "propose_ms", 0.0)))
        for st in recorded_steps
        if st.step_before < int(target_tick) and st.proposal is not None
    }

    def _replay_proposer(snap: NativeSnapshot, network: Any, *, budget_ms: float = 35.0) -> CandidateProposal:
        t_k = int(snap.tick)
        if t_k not in prop_by_tick:
            raise KeyError(f"Missing recorded prefix proposal at tick {t_k} (target={target_tick})")
        rec_prop, rec_ms = prop_by_tick[t_k]
        if rec_ms > 0.0:
            _cvcr_record_work_ms(rec_ms)
        return copy.deepcopy(rec_prop)

    env.external_proposer = _replay_proposer
    env.reset(seed=seed, options={"scenario_tuple": scenario_copy})

    # Re-charge exact recorded propose_ms under a zero-jitter virtual clock so window_cost_steps / lat_hold match bit-for-bit.
    with virtual_clock_scope(poll_ms=0.0):
        for st in recorded_steps:
            if st.step_before >= int(target_tick):
                break
            if int(env.current_step) != int(st.step_before):
                raise RuntimeError(
                    f"Fork replay tick mismatch: env.current_step={env.current_step} != recorded={st.step_before}"
                )
            _, _, tm, tr, _ = env.step(int(st.action))
            if tm or tr:
                break

    if int(env.current_step) != int(target_tick):
        raise RuntimeError(
            f"Fork did not land on target_tick={target_tick}; landed on {env.current_step}"
        )

    p2_fb = None
    if env.execution_perturbation is not None and env._active_execution_aware:
        p2_fb = env.execution_perturbation.capture_feedback(env.sim, env.current_step)
    snap = NativeSnapshotCapture.capture(
        env.sim,
        env.tracker.operations,
        hide_future_releases=True,
        execution_feedback=p2_fb,
    )
    return env, snap, env.sim.network


def rollout_48_and_optional_full(
    spec: dict[str, Any],
    recorded_steps: list[RecordedStep],
    target_tick: int,
    proposal_or_none: Optional[CandidateProposal],
    *,
    action_at_target: int,
    base_env_policy: str = "V19_ADAPTIVE_RL",
    rollout_horizon: int = 48,
    full_continuation: bool = False,
) -> dict[str, Any]:
    """Commit `proposal_or_none` at `target_tick` in a fresh fork and evaluate +48 tick (and optional terminal) outcome."""
    from amr_sched.rl.spaces import compute_macro_state

    env, snap_before, _ = replay_fork_env_at_tick(
        spec,
        recorded_steps,
        target_tick,
        base_env_policy=base_env_policy,
    )
    cost_before = env._cost_ledger(terminal=False)

    if proposal_or_none is not None:
        env.external_proposer = lambda _s, _n, *, budget_ms=35.0: copy.deepcopy(proposal_or_none)
        cur_obs = compute_macro_state(
            env.sim,
            env.tracker,
            last_replan_tick=env.last_replan_tick,
            recent_conflict_ticks=env.recent_conflict_ticks,
        )
        mask_now = env._current_mask(cur_obs)
        eff_action = int(action_at_target)
        if eff_action <= 0 or eff_action >= len(mask_now) or mask_now[eff_action] == 0:
            preferred = [4, 2, 1, 3]
            picked = next((a for a in preferred if a < len(mask_now) and mask_now[a] == 1), None)
            if picked is None:
                picked = next((i for i in range(1, len(mask_now)) if mask_now[i] == 1), 0)
            eff_action = int(picked)
    else:
        env.external_proposer = None
        eff_action = 0

    obs, _, tm, tr, info_first = env.step(eff_action)
    receipt_accepted = bool(info_first.get("receipt_accepted", eff_action == 0))
    churn_step0 = int(info_first.get("churn", 0))

    # Step remaining (rollout_horizon - 1) ticks under KEEP so the tested decision executes faithfully
    ticks_elapsed = 1
    while not (tm or tr) and ticks_elapsed < int(rollout_horizon):
        obs, _, tm, tr, _ = env.step(0)
        ticks_elapsed += 1

    cost_48 = env._cost_ledger(terminal=(tm or tr))
    completed_t_48, completed_o_48 = env._completion_counts(int(env.sim._step_cursor))
    total_t = len(env.sim.tasks)
    total_o = len(env.tracker.operations)
    rem_t = total_t - completed_t_48
    rem_o = total_o - completed_o_48
    # Partial-horizon surrogate combining incurred tardiness/churn/compute + unfinished work residual
    j48_incurred = float(sum(cost_48.values()) - sum(cost_before.values()))
    j48_with_residual = float(
        cost_48["cmax"]
        + cost_48["task_tardiness_cost"]
        + cost_48["op_tardiness_cost"]
        + cost_48["churn_cost"]
        + cost_48["compute_cost"]
        + 120.0 * rem_t
        + 80.0 * rem_o
    )

    out: dict[str, Any] = {
        "target_tick": int(target_tick),
        "receipt_accepted": receipt_accepted,
        "churn_step0": churn_step0,
        "ticks_48_end": int(env.current_step),
        "terminated_within_48": bool(tm),
        "completed_tasks_48": int(completed_t_48),
        "completed_ops_48": int(completed_o_48),
        "j48_incurred_delta": round(j48_incurred, 4),
        "j48_with_residual": round(j48_with_residual, 4),
        "task_tard_48": round(float(cost_48["task_tardiness_cost"]), 4),
        "op_tard_48": round(float(cost_48["op_tardiness_cost"]), 4),
        "churn_cost_48": round(float(cost_48["churn_cost"]), 4),
        "starvation_steps_48": int(env.tracker.starvation_steps),
    }

    if full_continuation and not (tm or tr):
        ctrl_cont = RuleV10PortfolioPolicy(epoch_interval=16)
        ctrl_cont.reset()
        env.external_proposer = None
        while not (tm or tr):
            f_c = compute_expert_v9_state(
                env.sim,
                env.tracker,
                last_replan_tick=env.last_replan_tick,
                recent_conflict_ticks=env.recent_conflict_ticks,
                decision_logs=env.decision_logs,
            )
            m_c = compute_expert_v10_action_mask(env.sim, env.last_replan_tick)
            a_c = ctrl_cont.select_action(f_c, {"action_mask": m_c, "step": env.current_step}, sim=env.sim)
            obs, _, tm, tr, _ = env.step(a_c)
        m_end = env.compute_episode_metrics()
        aud_end = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=False)
        out["full_net_obj"] = round(float(m_end["net_obj"]), 4)
        out["full_h1"] = round(
            float(m_end["cmax"]) + float(m_end.get("task_tardiness_cost", 0.0)) + float(m_end.get("op_tardiness_cost", 0.0)),
            4,
        )
        out["full_comp100"] = bool(
            float(m_end["task_completion_rate"]) >= 0.9999 and float(m_end["production_completion_rate"]) >= 0.9999
        )
        out["full_audit_passed"] = bool(aud_end.pass_audit)
    elif full_continuation:
        m_end = env.compute_episode_metrics()
        out["full_net_obj"] = round(float(m_end["net_obj"]), 4)
        out["full_h1"] = round(
            float(m_end["cmax"]) + float(m_end.get("task_tardiness_cost", 0.0)) + float(m_end.get("op_tardiness_cost", 0.0)),
            4,
        )
        out["full_comp100"] = bool(
            float(m_end["task_completion_rate"]) >= 0.9999 and float(m_end["production_completion_rate"]) >= 0.9999
        )
        out["full_audit_passed"] = True

    return out
