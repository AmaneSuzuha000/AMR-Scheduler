"""Tests for Work-Calibrated Clock, Resource Windows, Paired V13 Dev Scenarios, and V13 Kitting RL."""

from __future__ import annotations

import time
import numpy as np

from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.clock import is_virtual_clock_active, now_s, record_work_ms, virtual_clock_scope
from amr_sched.cvcr.production import NativeProductionTracker
from amr_sched.cvcr.snapshot import OperationSnapshot, align_start_to_windows
from amr_sched.cvcr.v13_kitting_rl import KITTING_CAND_FEAT_DIM, KittingWindowRanker
from amr_sched.rl import AMRReschedulingEnv, V13KittingPolicy, WhenToReplanTrigger
from amr_sched.rl.spaces import EXPERT_V12_STATE_DIM, NUM_EXPERT_V12_ACTIONS
from amr_sched.scenarios_v13_kitting import V13_FACTOR_COMBOS, build_v13_paired_dev_scenario


def test_virtual_clock_isolation_and_work_determinism() -> None:
    assert not is_virtual_clock_active()
    orig_perf = time.perf_counter
    samples: list[float] = []
    for _ in range(2):
        with virtual_clock_scope(poll_ms=0.002, start_s=500.0):
            assert time.perf_counter is orig_perf
            t0 = now_s()
            record_work_ms(1.25)
            record_work_ms(0.75)
            t1 = now_s()
            samples.append(round((t1 - t0) * 1000.0, 6))
    assert not is_virtual_clock_active()
    assert samples[0] == samples[1]
    assert abs(samples[0] - 2.002) < 1e-4


def test_align_start_to_windows_and_tracker_audit() -> None:
    wins = ((10, 46), (86, 200))
    assert align_start_to_windows(42, wins) == 42
    assert align_start_to_windows(46, wins) == 46
    assert align_start_to_windows(49, wins) == 86
    assert align_start_to_windows(205, wins) == 240

    op_ok = OperationSnapshot(
        op_id="OP_W",
        kit_task_ids=(),
        bay_id="B1",
        crew_id=1,
        tool_id=1,
        proc_steps=10,
        due_step=80,
        resource_windows=wins,
    )
    tracker = NativeProductionTracker([op_ok], [])
    assert tracker.step(47) == []
    assert not op_ok.started
    assert tracker.step(86) == ["OP_W"]
    assert op_ok.started and op_ok.start_step == 86


def test_v13_paired_dev_scenarios_and_kitting_controllers() -> None:
    assert len(V13_FACTOR_COMBOS) == 16
    for idx, (sh, ms, cm, wn) in enumerate(V13_FACTOR_COMBOS[::5]):
        spec = {
            "shift_timing": sh,
            "missing_count": ms,
            "corridor_mode": cm,
            "window_count": wn,
        }
        mode = ("V13_KITTING_DET", "V13_KITTING_SUP", "V13_KITTING_RL", "V13_KITTING_DEFAULT")[idx % 4]
        env = AMRReschedulingEnv(
            topology="unseen_choke_yard",
            seed=8201 + idx,
            allow_kitting_shortcut=False,
            pure_physical_reward=True,
            real_compute_window_coupling=True,
            expert_v12_portfolio=True,
            loaded_repair_mode="haven_v12",
            loaded_repair_threshold=8,
            record_trajectory=True,
            kitting_ranker=KittingWindowRanker(weights=None, mode="smdp_rl"),
            v13_kitting_mode=mode,
        )
        obs, info = env.reset(seed=8201 + idx, options={"v13_kitting_spec": spec})
        pol = V13KittingPolicy(epoch_interval=16)
        pol.reset()
        done = False
        while not done:
            o12 = env.get_expert_v12_obs(obs)
            act = pol.select_action(o12, info, sim=env.sim)
            obs, _, tm, tr, info = env.step(act)
            done = tm or tr
        m = env.compute_episode_metrics()
        aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
        assert aud.pass_audit, f"Audit issues on {spec}: {aud.issues}"
        assert m["task_completion_rate"] == 1.0
        k_decs = [d["kitting_decision"] for d in env.decision_logs if d.get("kitting_decision")]
        assert len(k_decs) >= 1
        assert "cand_id" in k_decs[0] and "source_kind" in k_decs[0]


def test_when_to_replan_rescues_idle_fleet_stranded_backlog() -> None:
    """When >=45% of the fleet sits idle with >=15% unassigned backlog, trigger must still wake up."""
    trig = WhenToReplanTrigger(epoch_interval=16, stall_threshold=8)
    trig.reset()
    trig.last_dec = 197
    mask = np.ones((NUM_EXPERT_V12_ACTIONS,), dtype=np.int64)

    obs = np.zeros((EXPERT_V12_STATE_DIM,), dtype=np.float32)
    obs[0] = 0.333   # 4/6 AGVs idle
    obs[2] = 1.0     # committed ratio among assigned == 1.0 (only remaining assigned are picked)
    obs[10] = 0.294  # 5/17 released tasks stranded unassigned in queues
    obs[11] = 0.24   # mid-late progress
    obs[32] = 1.0    # late ops high

    qm, _ = trig.should_query(obs, mask, tick=240, sim=None)
    assert qm == "global"

    # Conversely, when nearly full fleet is busy (obs[0] >= 0.80), keep calm V10 suppression intact
    trig.reset()
    trig.last_dec = 197
    obs_busy = obs.copy()
    obs_busy[0] = 0.833
    obs_busy[32] = 0.0
    obs_busy[11] = 0.85
    qm_busy, _ = trig.should_query(obs_busy, mask, tick=240, sim=None)
    assert qm_busy == "none"


def test_v13_v12_fusion_completes_flow_inversion_7104() -> None:
    """Verify V13 frontend + V12 multi-operator backend completes flow_inversion seed 7104 cleanly."""
    with virtual_clock_scope(poll_ms=0.002):
        for mode in ("V13_KITTING_DET", "V13_KITTING_RL"):
            env = AMRReschedulingEnv(
                topology="unseen_choke_yard",
                seed=7104,
                allow_kitting_shortcut=False,
                pure_physical_reward=True,
                real_compute_window_coupling=True,
                expert_v12_portfolio=True,
                loaded_repair_mode="haven_v12",
                loaded_repair_threshold=8,
                record_trajectory=True,
                kitting_ranker=KittingWindowRanker(weights=None, mode=("det_rule" if mode == "V13_KITTING_DET" else "smdp_rl")),
                v13_kitting_mode=mode,
            )
            obs, info = env.reset(seed=7104, options={"topology": "unseen_choke_yard", "stress_axis": "flow_inversion"})
            pol = V13KittingPolicy(epoch_interval=16)
            pol.reset()
            done = False
            while not done:
                act = pol.select_action(env.get_expert_v12_obs(obs), info, sim=env.sim)
                obs, _, tm, tr, info = env.step(act)
                done = tm or tr
            m = env.compute_episode_metrics()
            aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
            assert aud.pass_audit, f"{mode} audit failed: {aud.issues}"
            assert m["task_completion_rate"] == 1.0, f"{mode} failed 7104 completion: {m}"

