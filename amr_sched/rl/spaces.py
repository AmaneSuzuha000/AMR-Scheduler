"""Action and Observation Space Definitions for RL-CVCR Meta-Controller.

Defines:
1. 12-dimensional continuous macro state vector s_t with exact physical normalization.
2. 13-action discrete space combining KEEP and (Scope, Budget) tuples.
3. Physics-grounded action mask M(s_t) pruning redundant or infeasible transitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

STATE_FEATURE_NAMES = [
    "fleet_utilization",
    "bottleneck_load",
    "committed_ratio",
    "mean_slack_ratio",
    "min_slack_ratio",
    "faulted_agv_ratio",
    "urgent_pending_ratio",
    "time_since_replan",
    "starvation_gap",
    "conflict_wait_rate",
    "unassigned_ratio",
    "progress_ratio",
]

STATE_DIM = len(STATE_FEATURE_NAMES)

SCOPES = [
    "S0_DIRECT_AFFECTED",
    "S2_CROSS_DOMAIN_CLOSURE",
    "S3_VALUE_EXPANDED_CVCR",
]

BUDGETS_MS = [20.0, 50.0, 100.0, 150.0]

ACTION_KEEP = 0


@dataclass(frozen=True)
class ActionSpec:
    action_idx: int
    is_replan: bool
    scope: Optional[str]
    budget_ms: float
    name: str


def _build_action_table() -> list[ActionSpec]:
    table = [
        ActionSpec(
            action_idx=ACTION_KEEP,
            is_replan=False,
            scope=None,
            budget_ms=0.0,
            name="KEEP",
        )
    ]
    idx = 1
    for scope in SCOPES:
        for b_ms in BUDGETS_MS:
            name = f"REPLAN_{scope}_{int(b_ms)}MS"
            table.append(
                ActionSpec(
                    action_idx=idx,
                    is_replan=True,
                    scope=scope,
                    budget_ms=b_ms,
                    name=name,
                )
            )
            idx += 1
    return table


ACTION_TABLE: list[ActionSpec] = _build_action_table()
ACTION_NAMES: list[str] = [spec.name for spec in ACTION_TABLE]
NUM_ACTIONS = len(ACTION_TABLE)


def decode_action(action_idx: int) -> ActionSpec:
    """Decode an integer action index into its ActionSpec."""
    if 0 <= action_idx < NUM_ACTIONS:
        return ACTION_TABLE[action_idx]
    raise ValueError(f"Invalid action index {action_idx}, must be in [0, {NUM_ACTIONS - 1}]")


def compute_macro_state(
    sim: Any,
    tracker: Any,
    last_replan_tick: int,
    recent_conflict_ticks: int = 0,
    window_steps: int = 10,
) -> np.ndarray:
    """Compute normalized 12-dimensional continuous macro state vector.

    Strict causal observability: ignores future/hidden unreleased tasks across every feature.
    """
    tick = int(sim._step_cursor)
    max_steps = max(1, getattr(sim.config, "max_steps", 1000))
    n_agvs = max(1, len(sim.agvs))

    def _task_stat(t: Any) -> str:
        s = getattr(t, "status", "")
        return getattr(s, "value", str(s)).lower()

    def _agv_stat(a: Any) -> str:
        s = getattr(a, "state", "")
        return getattr(s, "value", str(s)).lower()

    visible_tasks = [
        t for t in sim.tasks
        if t.release_step <= tick and not getattr(t, "hidden_until_release", False)
    ]
    active_visible_tasks = [
        t for t in visible_tasks if _task_stat(t) not in ("completed", "failed")
    ]
    n_vis = max(1, len(visible_tasks))

    # 1. Fleet utilization
    busy_agvs = sum(1 for a in sim.agvs if _agv_stat(a) in ("to_pickup", "loading", "to_dropoff", "unloading"))
    fleet_util = float(busy_agvs) / n_agvs

    # 2. Bottleneck edge load (vehicle-ID-invariant over _transit.values())
    edge_counts: dict[tuple[str, str], int] = {}
    transit_map = getattr(sim, "_transit", {})
    transit_vals = transit_map.values() if isinstance(transit_map, dict) else transit_map
    transit_agvs = set(transit_map.keys()) if isinstance(transit_map, dict) else set()
    for tr in transit_vals:
        if len(tr) >= 2:
            edge = (str(tr[0]), str(tr[1]))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    for a in sim.agvs:
        if a.agv_id not in transit_agvs and a.cur_node and a.route:
            edge = (str(a.cur_node), str(a.route[0]))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    max_edge_load = 0.0
    for edge, count in edge_counts.items():
        cap = float(sim.network.edge_capacity(edge[0], edge[1])) if hasattr(sim.network, "edge_capacity") else 1.0
        cap = max(1.0, cap)
        max_edge_load = max(max_edge_load, count / cap)
    bottleneck_load = min(2.0, max_edge_load)

    # 3. Committed ratio (visible tasks in progress or non-reassignable)
    assigned_tasks = [t for t in active_visible_tasks if t.assigned_agv is not None]
    committed_tasks = [t for t in assigned_tasks if _task_stat(t) in ("picked", "delivering")]
    committed_ratio = float(len(committed_tasks)) / max(1, len(assigned_tasks))

    # 4 & 5. Slack ratios (visible active tasks only)
    slacks: list[float] = []
    for t in active_visible_tasks:
        if t.due_step is not None:
            rem_time = max(1.0, float(t.due_step - tick))
            est_rem = 20.0
            if t.assigned_agv is not None:
                for a in sim.agvs:
                    if a.agv_id == t.assigned_agv:
                        est_rem = max(1.0, float(a.remaining_job_steps))
                        break
            slack_ratio = rem_time / est_rem
            slacks.append(float(np.clip(slack_ratio, -2.0, 5.0) / 5.0))
    mean_slack = float(np.mean(slacks)) if slacks else 1.0
    min_slack = float(np.min(slacks)) if slacks else 1.0

    # 6. Physical active faulted AGV ratio & Event 1 pre-pickup indicator
    faulted_set = getattr(sim, "_faulted_agvs", set())
    active_faulted_agvs = sum(
        1 for a in sim.agvs
        if (a.agv_id in faulted_set or getattr(a, "faulted", False) or _agv_stat(a) == "faulted")
        and any(
            t.assigned_agv == a.agv_id and _task_stat(t) not in ("completed", "failed")
            for t in visible_tasks
        )
    )
    has_open_fault = any(
        _task_stat(t) == "assigned"
        and t.assigned_agv in faulted_set
        and tick >= getattr(t, "secondary_window_open_step", 0)
        for t in active_visible_tasks
    )
    faulted_ratio = 1.0 if has_open_fault else float(active_faulted_agvs) / n_agvs

    # 7. Urgent pending / open kitting bottleneck indicator (visible tasks only)
    urgent_pending = sum(
        1 for t in active_visible_tasks
        if (
            _task_stat(t) in ("pending", "unassigned")
            and (
                getattr(t, "is_urgent", False)
                or str(getattr(t, "priority", "")).lower() in ("urgent", "1", "taskpriority.urgent")
            )
        )
        or (
            _task_stat(t) in ("pending", "unassigned", "assigned")
            and getattr(t, "kitting_bottleneck_active", False)
        )
    )
    has_open_kitting = any(
        _task_stat(t) in ("pending", "unassigned", "assigned")
        and getattr(t, "kitting_bottleneck_active", False)
        and int(getattr(t, "adjustable_window_steps", 0)) > 0
        for t in active_visible_tasks
    )
    urgent_ratio = 1.0 if has_open_kitting else float(urgent_pending) / n_vis

    # 8. Time since last replan
    time_since_replan = min(100.0, float(tick - last_replan_tick)) / 100.0

    # 9. Starvation gap, Last-Adjustable-Window & Kitting-Bottleneck signals
    starvation_sum = 0.0
    ops = getattr(tracker, "operations", [])
    if ops:
        for op in ops:
            if not op.started:
                kit_tasks = [sim._task_by_id.get(tid) for tid in op.kit_task_ids if tid in sim._task_by_id]
                all_delivered = bool(kit_tasks and all(_task_stat(t) == "completed" for t in kit_tasks))
                if all_delivered:
                    starvation_sum += 0.2
    # Add unpicked faulted-robot tasks (+6.0) and open-window kitting bottleneck tasks (+3.0)
    open_windows: list[int] = []
    for t in active_visible_tasks:
        st_t = _task_stat(t)
        if (
            st_t == "assigned"
            and t.assigned_agv in faulted_set
            and tick >= getattr(t, "secondary_window_open_step", 0)
        ):
            starvation_sum += 6.0
            open_windows.append(int(getattr(t, "adjustable_window_steps", 6)))
        elif st_t in ("pending", "unassigned", "assigned") and getattr(t, "kitting_bottleneck_active", False):
            w_t = int(getattr(t, "adjustable_window_steps", 3))
            if w_t > 0:
                starvation_sum += 3.0
                open_windows.append(w_t)
    starvation_gap = min(2.0, starvation_sum / max(1, len(ops)))

    # 10. Conflict wait rate (captures both BLOCKED/PARKING and LOADING leg-2 planner stalls)
    blocked_since = getattr(sim, "_blocked_since", {})
    currently_waiting = sum(
        1 for a in sim.agvs
        if a.agv_id not in faulted_set
        and not getattr(a, "faulted", False)
        and (a.agv_id in blocked_since or _agv_stat(a) in ("blocked", "parking"))
    )
    conflict_wait_rate = min(
        1.0,
        max(
            float(currently_waiting) / n_agvs,
            float(recent_conflict_ticks) / max(1, window_steps * n_agvs),
        ),
    )

    # 11. Unassigned task ratio & normalized adjustable window margin during active disturbance
    unassigned = sum(
        1 for t in active_visible_tasks
        if _task_stat(t) in ("pending", "unassigned") or t.assigned_agv is None
    )
    unassigned_ratio = float(unassigned) / n_vis
    if open_windows:
        window_norm = min(1.0, float(max(open_windows)) / 10.0)
        unassigned_ratio = min(1.0, unassigned_ratio + 0.5 * window_norm)

    # 12. Progress ratio
    progress_ratio = min(1.0, float(tick) / max_steps)

    state = np.array(
        [
            fleet_util,
            bottleneck_load,
            committed_ratio,
            mean_slack,
            min_slack,
            faulted_ratio,
            urgent_ratio,
            time_since_replan,
            starvation_gap,
            conflict_wait_rate,
            unassigned_ratio,
            progress_ratio,
        ],
        dtype=np.float32,
    )
    return state


def compute_action_mask(
    sim: Any,
    tracker: Any,
    state: np.ndarray,
    last_replan_tick: int,
) -> np.ndarray:
    """Compute physical and logical action mask M(s_t) in {0, 1}^13.

    1 indicates the action is legal; 0 indicates the action is masked out.
    """
    mask = np.ones(NUM_ACTIONS, dtype=np.int8)
    tick = int(sim._step_cursor)

    def _task_stat(t: Any) -> str:
        s = getattr(t, "status", "")
        return getattr(s, "value", str(s)).lower()

    def _agv_stat(a: Any) -> str:
        s = getattr(a, "state", "")
        return getattr(s, "value", str(s)).lower()

    visible_tasks = [
        t for t in sim.tasks
        if t.release_step <= tick and not getattr(t, "hidden_until_release", False)
    ]
    active_visible = [t for t in visible_tasks if _task_stat(t) not in ("completed", "failed")]

    # 1. Faulted AGVs with assigned unfinished visible tasks
    faulted_set = getattr(sim, "_faulted_agvs", set())
    faulted_agv_ids = {
        a.agv_id for a in sim.agvs
        if a.agv_id in faulted_set or getattr(a, "faulted", False) or _agv_stat(a) == "faulted"
    }
    stranded_tasks = [
        t for t in active_visible
        if t.assigned_agv in faulted_agv_ids
        and _task_stat(t) in ("pending", "unassigned", "assigned")
    ]

    # 2. Urgent unassigned visible tasks
    urgent_unassigned = [
        t for t in active_visible
        if _task_stat(t) in ("pending", "unassigned") and (
            getattr(t, "is_urgent", False)
            or str(getattr(t, "priority", "")).lower() in ("urgent", "1", "taskpriority.urgent")
        )
    ]

    has_active_emergency = bool(stranded_tasks or urgent_unassigned)

    # KEEP is ALWAYS physically legal: robots can always continue executing current commitments
    mask[ACTION_KEEP] = 1

    # If no active visible tasks remain, replanning is physically meaningless
    if not active_visible:
        mask[1:] = 0
        return mask

    # At the exact same tick where replanning was already executed, do not replan again
    if tick == last_replan_tick and tick > 0:
        mask[1:] = 0
        return mask

    # Scope S0 pruning:
    # If there are no faulted robots and no urgent tasks, S0 has no disturbance seeds to release
    if not has_active_emergency:
        mask[1:5] = 0

    return mask


ANYTIME_FEATURE_NAMES = STATE_FEATURE_NAMES + [
    "unstarted_ops_ratio",
    "load_imbalance_norm",
    "last_improved_norm",
    "last_churn_norm",
    "last_expired_flag",
    "active_backlog_density",
    "has_unrescued_fault",
    "has_urgent_unassigned",
    "aisle_degree_norm",
    "due_pressure_norm",
]

ANYTIME_STATE_DIM = len(ANYTIME_FEATURE_NAMES)
ANYTIME_BUDGET_ACTIONS = (ACTION_KEEP, 9, 10, 11, 12)


def compute_anytime_state(
    sim: Any,
    tracker: Any,
    base_obs: Optional[np.ndarray] = None,
    *,
    last_replan_tick: int = -999,
    recent_conflict_ticks: int = 0,
    decision_logs: Optional[list[dict[str, Any]]] = None,
) -> np.ndarray:
    """Extend the 12-D macro state with 8 search-feedback & urgency signals (20-D)."""
    if base_obs is None:
        base_obs = compute_macro_state(
            sim,
            tracker,
            last_replan_tick=last_replan_tick,
            recent_conflict_ticks=recent_conflict_ticks,
        )

    def _tstat(t: Any) -> str:
        s = getattr(t, "status", "")
        return getattr(s, "value", str(s)).lower()

    tick = int(getattr(sim, "_step_cursor", 0))
    faulted_set = set(getattr(sim, "_faulted_agvs", set()))
    active_vis = [
        t for t in sim.tasks
        if t.release_step <= tick
        and not getattr(t, "hidden_until_release", False)
        and _tstat(t) not in ("completed", "failed")
    ]

    has_unrescued_fault = 1.0 if any(
        t.assigned_agv in faulted_set
        and _tstat(t) in ("pending", "unassigned", "assigned")
        and tick >= getattr(t, "secondary_window_open_step", 0)
        for t in active_vis
    ) else 0.0
    has_urgent_unassigned = 1.0 if any(
        _tstat(t) in ("pending", "unassigned")
        and (getattr(t, "is_urgent", False) or str(getattr(t, "priority", "")).lower() in ("urgent", "1", "taskpriority.urgent"))
        for t in active_vis
    ) else 0.0

    if tracker is not None and getattr(tracker, "operations", None):
        n_ops = max(1, len(tracker.operations))
        unstarted_ops_ratio = float(sum(1 for op in tracker.operations if not op.started)) / n_ops
    else:
        unstarted_ops_ratio = 0.0

    healthy_ids = [
        a.agv_id for a in sim.agvs
        if a.agv_id not in faulted_set and not getattr(a, "faulted", False)
    ]
    if healthy_ids and active_vis:
        counts = {aid: 0 for aid in healthy_ids}
        for t in active_vis:
            if t.assigned_agv in counts:
                counts[t.assigned_agv] += 1
        vals = list(counts.values())
        load_imbalance_norm = float(max(vals) - min(vals)) / max(1.0, float(len(active_vis)))
    else:
        load_imbalance_norm = 0.0

    last_log = decision_logs[-1] if decision_logs else None
    if last_log is not None:
        last_improved_norm = float(np.clip(float(last_log.get("improved_delta", 0.0)) / 200.0, 0.0, 1.0))
        last_churn_norm = float(np.clip(float(last_log.get("step_churn", 0)) / 6.0, 0.0, 1.0))
        last_expired_flag = 1.0 if bool(last_log.get("window_expired", False)) else 0.0
    else:
        last_improved_norm = 0.0
        last_churn_norm = 0.0
        last_expired_flag = 0.0

    active_backlog_density = float(np.clip(float(len(active_vis)) / max(1.0, 3.0 * len(sim.agvs)), 0.0, 1.0))

    aisle_nodes = [
        n for n in getattr(sim.network, "nodes", {}).values()
        if str(getattr(getattr(n, "kind", None), "value", getattr(n, "kind", ""))).lower() == "aisle"
    ]
    if aisle_nodes and hasattr(sim.network, "out_edges"):
        mean_deg = float(np.mean([len(sim.network.out_edges(n.node_id)) for n in aisle_nodes]))
        aisle_degree_norm = float(np.clip(mean_deg / 5.0, 0.0, 1.0))
    else:
        aisle_degree_norm = 0.0

    due_terms = [
        float(np.clip((float(tick) + 12.0 - float(t.due_step)) / 40.0, 0.0, 1.0))
        for t in active_vis
        if t.due_step is not None
    ]
    due_pressure_norm = float(np.mean(due_terms)) if due_terms else 0.0

    extra = np.array(
        [
            unstarted_ops_ratio,
            float(np.clip(load_imbalance_norm, 0.0, 1.0)),
            last_improved_norm,
            last_churn_norm,
            last_expired_flag,
            active_backlog_density,
            has_unrescued_fault,
            has_urgent_unassigned,
            aisle_degree_norm,
            due_pressure_norm,
        ],
        dtype=np.float32,
    )
    return np.concatenate([np.asarray(base_obs, dtype=np.float32), extra]).astype(np.float32)


def is_in_secondary_hold(sim: Any) -> bool:
    """Return True when faulted assigned tasks exist but all are frozen until secondary_window_open_step."""
    if sim is None:
        return False
    tick = int(getattr(sim, "_step_cursor", 0))
    faulted_set = set(getattr(sim, "_faulted_agvs", set()))
    if not faulted_set:
        return False
    has_held = False
    for t in sim.tasks:
        st = getattr(t, "status", "")
        st_v = getattr(st, "value", str(st)).lower()
        if st_v == "assigned" and t.assigned_agv in faulted_set:
            if tick >= getattr(t, "secondary_window_open_step", 0):
                return False
            has_held = True
    return has_held


def compute_schedule_potential(sim: Any, tracker: Any) -> float:
    """Pure physical remaining-lateness & stall lower-bound potential Phi(s) (G3 label-free).

    Inspired by E2G-MoE (IEEE TII 2026) and MJDP (JMSY 2025): evaluates only the physical
    backlog state (estimated remaining work, tardiness excess, unrescued fault stall bound,
    and urgent unassigned exposure) with zero dependence on rescheduling action labels.
    """
    if sim is None:
        return 0.0
    tick = int(getattr(sim, "_step_cursor", 0))
    faulted_set = set(getattr(sim, "_faulted_agvs", set()))
    agv_rem = {a.agv_id: max(0.0, float(getattr(a, "remaining_job_steps", 0))) for a in sim.agvs}

    pot = 0.0
    for t in sim.tasks:
        if t.release_step > tick or getattr(t, "hidden_until_release", False):
            continue
        st = getattr(t, "status", "")
        st_v = getattr(st, "value", str(st)).lower()
        if st_v in ("completed", "failed"):
            continue

        on_fault = bool(t.assigned_agv in faulted_set and st_v == "assigned")
        is_urgent = bool(
            getattr(t, "is_urgent", False)
            or str(getattr(t, "priority", "")).lower() in ("urgent", "1", "taskpriority.urgent")
        )
        if on_fault:
            held = tick < getattr(t, "secondary_window_open_step", 0)
            est_rem = 48.0 if held else 30.0
            pot += 40.0 if held else 28.0
        elif st_v in ("pending", "unassigned"):
            est_rem = 12.0 if is_urgent else 8.0
            if is_urgent:
                pot += 18.0
        else:
            est_rem = max(2.0, agv_rem.get(t.assigned_agv or "", 6.0))

        est_finish = float(tick) + est_rem
        due = getattr(t, "due_step", None)
        if due is not None:
            pot += max(0.0, est_finish - float(due))
        pot += 0.25 * est_rem
    if tracker is not None and getattr(tracker, "operations", None):
        for op in tracker.operations:
            if not getattr(op, "completed", False):
                pot += 0.35 * float(getattr(op, "proc_steps", 0))
    return float(pot / 50.0)


def should_query_anytime_controller(
    obs20: np.ndarray,
    base_mask: np.ndarray,
    tick: int,
    last_decision_tick: int,
    epoch_interval: int = 12,
    prev_emerg: Any = False,
    in_secondary_hold: bool = False,
) -> tuple[bool, tuple[bool, bool]]:
    """Opportunity-gated SMDP trigger: rising fault/urgent edge, or cooled-down mid-episode bottleneck."""
    cur_f = bool(float(obs20[18]) > 0.5)
    cur_u = bool(float(obs20[19]) > 0.5)
    if isinstance(prev_emerg, tuple) and len(prev_emerg) == 2:
        prev_f, prev_u = bool(prev_emerg[0]), bool(prev_emerg[1])
    else:
        prev_f = prev_u = bool(prev_emerg)
    cur_state = (cur_f, cur_u)

    if in_secondary_hold:
        return False, cur_state
    if not any(idx < len(base_mask) and base_mask[idx] == 1 for idx in ANYTIME_BUDGET_ACTIONS if idx != ACTION_KEEP):
        return False, cur_state
    rising_f = cur_f and not prev_f
    rising_u = cur_u and not prev_u
    if (rising_f or rising_u) and float(obs20[16]) < 0.5:
        return True, cur_state
    if cur_f or cur_u:
        return (tick - last_decision_tick) >= min(epoch_interval, 12), cur_state
    if float(obs20[11]) >= 0.92 or float(obs20[2]) >= 0.92:
        return False, cur_state
    # Calm optimization opportunity gate: open periodic backlog checkpoints so RL learns where to intervene
    cooldown_ready = float(obs20[7]) >= 0.08 and (tick - last_decision_tick) >= 12
    has_leverage = (
        float(obs20[8]) >= 0.08
        or float(obs20[13]) >= 0.12
        or float(obs20[9]) >= 0.04
        or float(obs20[4]) < 0.30
        or float(obs20[17]) >= 0.25
        or float(obs20[10]) >= 0.08
        or (float(obs20[12]) >= 0.85 and float(obs20[7]) >= 0.10)
    )
    return bool(cooldown_ready and has_leverage), cur_state


def compute_anytime_action_mask(base_mask: np.ndarray) -> np.ndarray:
    """Restrict mask to the 5 orthogonal anytime compute budgets (KEEP, 20, 50, 100, 150 ms)."""
    out = np.zeros_like(base_mask, dtype=np.int8)
    for idx in ANYTIME_BUDGET_ACTIONS:
        if idx < len(base_mask) and base_mask[idx] == 1:
            out[idx] = 1
    if out.sum() == 0:
        out[ACTION_KEEP] = 1
    return out


# ---------------------------------------------------------------------------
# Multi-Expert Portfolio Action & State Extensions (FULLSTACK-PERFORMANCE-02)
# ---------------------------------------------------------------------------

EXPERT_SPECS: tuple[tuple[Optional[str], float, str], ...] = (
    (None, 0.0, "KEEP"),
    ("EXP_E1_FAST_RESCUE", 20.0, "E1_FAST_RESCUE_20MS"),
    ("EXP_E2_PROD_FOCUSED", 50.0, "E2_PROD_FOCUSED_50MS"),
    ("EXP_E2_PROD_FOCUSED", 100.0, "E2_PROD_FOCUSED_100MS"),
    ("EXP_E3_JOINT_CORE", 50.0, "E3_JOINT_CORE_50MS"),
    ("EXP_E3_JOINT_CORE", 100.0, "E3_JOINT_CORE_100MS"),
    ("EXP_E4_CONGESTION_PACER", 50.0, "E4_CONGESTION_PACER_50MS"),
    ("EXP_E5_AGGRESSIVE_GRID", 50.0, "E5_AGGRESSIVE_GRID_50MS"),
    ("EXP_E5_AGGRESSIVE_GRID", 100.0, "E5_AGGRESSIVE_GRID_100MS"),
)

EXPERT_ACTION_TABLE: list[ActionSpec] = [
    ActionSpec(
        action_idx=i,
        is_replan=(sc is not None),
        scope=sc,
        budget_ms=bms,
        name=nm,
    )
    for i, (sc, bms, nm) in enumerate(EXPERT_SPECS)
]
EXPERT_ACTION_NAMES: list[str] = [s.name for s in EXPERT_ACTION_TABLE]
NUM_EXPERT_ACTIONS = len(EXPERT_ACTION_TABLE)

EXPERT_FEATURE_NAMES = ANYTIME_FEATURE_NAMES + [
    "n_nodes_norm",
    "bay_crew_flex_ratio",
]
EXPERT_STATE_DIM = len(EXPERT_FEATURE_NAMES)


def decode_expert_action(action_idx: int) -> ActionSpec:
    """Decode integer action index inside the 9-action Multi-Expert Portfolio."""
    if 0 <= action_idx < NUM_EXPERT_ACTIONS:
        return EXPERT_ACTION_TABLE[action_idx]
    raise ValueError(f"Invalid expert action index {action_idx}, must be in [0, {NUM_EXPERT_ACTIONS - 1}]")


def compute_expert_action_mask(
    sim: Any,
    last_replan_tick: int,
) -> np.ndarray:
    """Compute strictly causal 9-action legal mask for Multi-Expert Portfolio."""
    mask = np.ones(NUM_EXPERT_ACTIONS, dtype=np.int8)
    tick = int(getattr(sim, "_step_cursor", 0))

    def _task_stat(t: Any) -> str:
        s = getattr(t, "status", "")
        return getattr(s, "value", str(s)).lower()

    active_visible = [
        t for t in sim.tasks
        if t.release_step <= tick
        and not getattr(t, "hidden_until_release", False)
        and _task_stat(t) not in ("completed", "failed")
    ]
    mask[ACTION_KEEP] = 1
    if not active_visible or (tick == last_replan_tick and tick > 0):
        mask[1:] = 0
    return mask


def compute_expert_state(
    sim: Any,
    tracker: Any,
    base_obs: Optional[np.ndarray] = None,
    *,
    last_replan_tick: int = -999,
    recent_conflict_ticks: int = 0,
    decision_logs: Optional[list[dict[str, Any]]] = None,
) -> np.ndarray:
    """Compute 24-D Multi-Expert SMDP state adding strictly causal layout & flexibility signals."""
    obs22 = compute_anytime_state(
        sim,
        tracker,
        base_obs,
        last_replan_tick=last_replan_tick,
        recent_conflict_ticks=recent_conflict_ticks,
        decision_logs=decision_logs,
    )
    n_nodes = len(getattr(getattr(sim, "network", None), "nodes", {}))
    n_nodes_norm = float(np.clip(float(n_nodes) / 80.0, 0.0, 1.0))

    ops = getattr(tracker, "operations", []) if tracker is not None else []
    unstarted = [op for op in ops if not getattr(op, "started", False)]
    if unstarted:
        flex_cnt = sum(
            1
            for op in unstarted
            if len(getattr(op, "eligible_bays", ())) > 1 or len(getattr(op, "eligible_crews", ())) > 1
        )
        bay_crew_flex_ratio = float(flex_cnt) / float(len(unstarted))
    else:
        bay_crew_flex_ratio = 0.0

    extra2 = np.array([n_nodes_norm, bay_crew_flex_ratio], dtype=np.float32)
    return np.concatenate([obs22, extra2]).astype(np.float32)


# ---------------------------------------------------------------------------
# V6 Multi-Expert Portfolio + Hybrid Criticality & Intra-Agent c_i Extensions
# ---------------------------------------------------------------------------

EXPERT_V6_SPECS: tuple[tuple[Optional[str], float, str], ...] = (
    (None, 0.0, "KEEP"),
    ("V6_E1_FAST_RESCUE", 20.0, "V6_E1_FAST_RESCUE_20MS"),
    ("V6_E2_CRIT_PROD_CI", 50.0, "V6_E2_CRIT_PROD_CI_50MS"),
    ("V6_E2_CRIT_PROD_CI", 100.0, "V6_E2_CRIT_PROD_CI_100MS"),
    ("V6_E3_JOINT_CI_HYS", 50.0, "V6_E3_JOINT_CI_HYS_50MS"),
    ("V6_E3_JOINT_CI_HYS", 100.0, "V6_E3_JOINT_CI_HYS_100MS"),
    ("V6_E4_PACER_CI", 50.0, "V6_E4_PACER_CI_50MS"),
    ("V6_E5_AGGRESSIVE_CI", 50.0, "V6_E5_AGGRESSIVE_CI_50MS"),
    ("V6_E5_AGGRESSIVE_CI", 100.0, "V6_E5_AGGRESSIVE_CI_100MS"),
)

EXPERT_V6_ACTION_TABLE: list[ActionSpec] = [
    ActionSpec(
        action_idx=i,
        is_replan=(sc is not None),
        scope=sc,
        budget_ms=bms,
        name=nm,
    )
    for i, (sc, bms, nm) in enumerate(EXPERT_V6_SPECS)
]
EXPERT_V6_ACTION_NAMES: list[str] = [s.name for s in EXPERT_V6_ACTION_TABLE]
NUM_EXPERT_V6_ACTIONS = len(EXPERT_V6_ACTION_TABLE)

EXPERT_V6_FEATURE_NAMES = EXPERT_FEATURE_NAMES + [
    "split_kit_ratio",
    "bay_contention_norm",
]
EXPERT_V6_STATE_DIM = len(EXPERT_V6_FEATURE_NAMES)


def decode_expert_v6_action(action_idx: int) -> ActionSpec:
    """Decode integer action index inside the 9-action V6 Multi-Expert Portfolio."""
    if 0 <= action_idx < NUM_EXPERT_V6_ACTIONS:
        return EXPERT_V6_ACTION_TABLE[action_idx]
    raise ValueError(f"Invalid expert_v6 action index {action_idx}, must be in [0, {NUM_EXPERT_V6_ACTIONS - 1}]")


def compute_expert_v6_action_mask(
    sim: Any,
    last_replan_tick: int,
) -> np.ndarray:
    """Compute strictly causal 9-action legal mask for V6 Multi-Expert Portfolio."""
    return compute_expert_action_mask(sim, last_replan_tick)


def compute_expert_v6_state(
    sim: Any,
    tracker: Any,
    base_obs: Optional[np.ndarray] = None,
    *,
    last_replan_tick: int = -999,
    recent_conflict_ticks: int = 0,
    decision_logs: Optional[list[dict[str, Any]]] = None,
) -> np.ndarray:
    """Compute strictly causal 26-D V6 Multi-Expert state adding split-kitting & bay contention."""
    obs24 = compute_expert_state(
        sim,
        tracker,
        base_obs,
        last_replan_tick=last_replan_tick,
        recent_conflict_ticks=recent_conflict_ticks,
        decision_logs=decision_logs,
    )
    ops = getattr(tracker, "operations", []) if tracker is not None else []
    unstarted = [op for op in ops if not getattr(op, "started", False)]
    task_map = {t.task_id: t for t in getattr(sim, "tasks", [])}

    split_cnt = 0
    bay_loads: dict[str, int] = {}
    for op in unstarted:
        bid = getattr(op, "bay_id", "")
        bay_loads[bid] = bay_loads.get(bid, 0) + 1
        agvs: set[str] = set()
        pending_kits = 0
        for tid in getattr(op, "kit_task_ids", ()):
            t = task_map.get(tid)
            if t is None:
                continue
            st = getattr(getattr(t, "status", ""), "value", str(getattr(t, "status", ""))).lower()
            if st not in ("completed", "failed"):
                pending_kits += 1
                if getattr(t, "assigned_agv", None):
                    agvs.add(t.assigned_agv)
        if pending_kits >= 2 and len(agvs) >= 2:
            split_cnt += 1

    split_kit_ratio = float(split_cnt) / float(len(unstarted)) if unstarted else 0.0
    max_bay_q = max(bay_loads.values(), default=0)
    bay_contention_norm = float(np.clip(float(max_bay_q) / 6.0, 0.0, 1.0))

    extra2 = np.array([split_kit_ratio, bay_contention_norm], dtype=np.float32)
    return np.concatenate([obs24, extra2]).astype(np.float32)


# ---------------------------------------------------------------------------
# V7 Multi-Expert Portfolio + Explicit c_i Execution Queue + BALANCE-KM
# ---------------------------------------------------------------------------

EXPERT_V7_SPECS: tuple[tuple[Optional[str], float, str], ...] = (
    (None, 0.0, "KEEP"),
    ("V7_E1_FAST_RESCUE", 20.0, "V7_E1_FAST_RESCUE_20MS"),
    ("V7_E2_CRIT_PROD_CI", 50.0, "V7_E2_CRIT_PROD_CI_50MS"),
    ("V7_E3_JOINT_CI_HYS", 50.0, "V7_E3_JOINT_CI_HYS_50MS"),
    ("V7_E3_JOINT_CI_HYS", 100.0, "V7_E3_JOINT_CI_HYS_100MS"),
    ("V7_E4_PACER_CI", 50.0, "V7_E4_PACER_CI_50MS"),
    ("V7_E5_AGGRESSIVE_CI", 50.0, "V7_E5_AGGRESSIVE_CI_50MS"),
    ("V7_E5_AGGRESSIVE_CI", 100.0, "V7_E5_AGGRESSIVE_CI_100MS"),
    ("V7_E6_BALANCE_KM", 100.0, "V7_E6_BALANCE_KM_100MS"),
)

EXPERT_V7_ACTION_TABLE: list[ActionSpec] = [
    ActionSpec(
        action_idx=i,
        is_replan=(sc is not None),
        scope=sc,
        budget_ms=bms,
        name=nm,
    )
    for i, (sc, bms, nm) in enumerate(EXPERT_V7_SPECS)
]
EXPERT_V7_ACTION_NAMES: list[str] = [s.name for s in EXPERT_V7_ACTION_TABLE]
NUM_EXPERT_V7_ACTIONS = len(EXPERT_V7_ACTION_TABLE)

EXPERT_V7_FEATURE_NAMES = EXPERT_V6_FEATURE_NAMES + [
    "corridor_bottleneck_frac",
    "mean_node_degree_norm",
    "last_kit_tight_ratio",
    "active_queue_depth_norm",
]
EXPERT_V7_STATE_DIM = len(EXPERT_V7_FEATURE_NAMES)


def decode_expert_v7_action(action_idx: int) -> ActionSpec:
    """Decode integer action index inside the 9-action V7 Multi-Expert Portfolio."""
    if 0 <= action_idx < NUM_EXPERT_V7_ACTIONS:
        return EXPERT_V7_ACTION_TABLE[action_idx]
    raise ValueError(
        f"Invalid expert_v7 action index {action_idx}, must be in [0, {NUM_EXPERT_V7_ACTIONS - 1}]"
    )


def compute_expert_v7_action_mask(
    sim: Any,
    last_replan_tick: int,
) -> np.ndarray:
    """Compute strictly causal 9-action legal mask for V7 Multi-Expert Portfolio."""
    return compute_expert_action_mask(sim, last_replan_tick)


def compute_expert_v7_state(
    sim: Any,
    tracker: Any,
    base_obs: Optional[np.ndarray] = None,
    *,
    last_replan_tick: int = -999,
    recent_conflict_ticks: int = 0,
    decision_logs: Optional[list[dict[str, Any]]] = None,
) -> np.ndarray:
    """Compute strictly causal 30-D V7 state with topology-agnostic graph & last-kit bottleneck signals."""
    obs26 = compute_expert_v6_state(
        sim,
        tracker,
        base_obs,
        last_replan_tick=last_replan_tick,
        recent_conflict_ticks=recent_conflict_ticks,
        decision_logs=decision_logs,
    )
    net = getattr(sim, "network", None)
    nodes = getattr(net, "nodes", {}) if net is not None else {}
    if nodes and hasattr(net, "out_edges"):
        degs = [len(net.out_edges(nid)) for nid in nodes]
        corridor_bottleneck_frac = float(sum(1 for d in degs if d <= 2)) / float(len(degs))
        mean_node_degree_norm = float(np.clip((sum(degs) / float(len(degs))) / 6.0, 0.0, 1.0))
    else:
        corridor_bottleneck_frac = 0.0
        mean_node_degree_norm = 0.0

    ops = getattr(tracker, "operations", []) if tracker is not None else []
    unstarted = [op for op in ops if not getattr(op, "started", False)]
    task_map = {t.task_id: t for t in getattr(sim, "tasks", [])}
    cur_step = float(getattr(sim, "_step_cursor", 0))

    last_kit_tight = 0
    for op in unstarted:
        pending_kits = 0
        for tid in getattr(op, "kit_task_ids", ()):
            t = task_map.get(tid)
            if t is None:
                continue
            st = getattr(getattr(t, "status", ""), "value", str(getattr(t, "status", ""))).lower()
            if st not in ("completed", "failed"):
                pending_kits += 1
        if pending_kits == 1:
            due = float(getattr(op, "due_step", 10**6))
            proc = float(getattr(op, "process_steps", 0))
            if (due - cur_step - proc) < 25.0:
                last_kit_tight += 1
    last_kit_tight_ratio = float(last_kit_tight) / float(len(unstarted)) if unstarted else 0.0

    agv_loads: dict[str, int] = {}
    for t in getattr(sim, "tasks", []):
        st = getattr(getattr(t, "status", ""), "value", str(getattr(t, "status", ""))).lower()
        aid = getattr(t, "assigned_agv", None)
        if aid and st in ("assigned", "picked"):
            agv_loads[aid] = agv_loads.get(aid, 0) + 1
    max_agv_q = max(agv_loads.values(), default=0)
    active_queue_depth_norm = float(np.clip(float(max_agv_q) / 6.0, 0.0, 1.0))

    extra4 = np.array(
        [
            corridor_bottleneck_frac,
            mean_node_degree_norm,
            last_kit_tight_ratio,
            active_queue_depth_norm,
        ],
        dtype=np.float32,
    )
    return np.concatenate([obs26, extra4]).astype(np.float32)


def should_query_expert_controller(
    obs_exp: np.ndarray,
    exp_mask: np.ndarray,
    tick: int,
    last_decision_tick: int,
    epoch_interval: int = 12,
    prev_emerg: Any = False,
    in_secondary_hold: bool = False,
) -> tuple[bool, tuple[bool, bool]]:
    """Opportunity-gated SMDP trigger matching G3 causality across the 9-action Expert Portfolio."""
    cur_f = bool(float(obs_exp[18]) > 0.5)
    cur_u = bool(float(obs_exp[19]) > 0.5)
    if isinstance(prev_emerg, tuple) and len(prev_emerg) == 2:
        prev_f, prev_u = bool(prev_emerg[0]), bool(prev_emerg[1])
    else:
        prev_f = prev_u = bool(prev_emerg)
    cur_state = (cur_f, cur_u)

    if in_secondary_hold:
        return False, cur_state
    if not any(exp_mask[idx] == 1 for idx in range(1, min(len(exp_mask), NUM_EXPERT_ACTIONS))):
        return False, cur_state
    rising_f = cur_f and not prev_f
    rising_u = cur_u and not prev_u
    if (rising_f or rising_u) and float(obs_exp[16]) < 0.5:
        return True, cur_state
    if cur_f or cur_u:
        return (tick - last_decision_tick) >= min(epoch_interval, 12), cur_state
    if float(obs_exp[11]) >= 0.92 or float(obs_exp[2]) >= 0.92:
        return False, cur_state
    cooldown_ready = float(obs_exp[7]) >= 0.08 and (tick - last_decision_tick) >= 12
    has_leverage = (
        float(obs_exp[8]) >= 0.08
        or float(obs_exp[13]) >= 0.12
        or float(obs_exp[9]) >= 0.04
        or float(obs_exp[4]) < 0.30
        or float(obs_exp[17]) >= 0.25
        or float(obs_exp[10]) >= 0.08
        or (float(obs_exp[12]) >= 0.85 and float(obs_exp[7]) >= 0.10)
    )
    return bool(cooldown_ready and has_leverage), cur_state


# ---------------------------------------------------------------------------
# V8 LTM + TAPF-PC + RCDR Conformal Drift + E2G Dual-Graph State Extensions
# ---------------------------------------------------------------------------

EXPERT_V8_SPECS: tuple[tuple[Optional[str], float, str], ...] = (
    (None, 0.0, "KEEP"),
    ("V8_E1_FAST_RESCUE", 20.0, "V8_E1_FAST_RESCUE_20MS"),
    ("V8_E2_CRIT_PROD_CI", 50.0, "V8_E2_CRIT_PROD_CI_50MS"),
    ("V8_E3_JOINT_PC_HYS", 50.0, "V8_E3_JOINT_PC_HYS_50MS"),
    ("V8_E3_JOINT_PC_HYS", 100.0, "V8_E3_JOINT_PC_HYS_100MS"),
    ("V8_E4_LTM_PACER_CI", 50.0, "V8_E4_LTM_PACER_CI_50MS"),
    ("V8_E5_AGGRESSIVE_LTM", 50.0, "V8_E5_AGGRESSIVE_LTM_50MS"),
    ("V8_E5_AGGRESSIVE_LTM", 100.0, "V8_E5_AGGRESSIVE_LTM_100MS"),
    ("V8_E6_BALANCE_KM_PC", 100.0, "V8_E6_BALANCE_KM_PC_100MS"),
)

EXPERT_V8_ACTION_TABLE: list[ActionSpec] = [
    ActionSpec(
        action_idx=i,
        is_replan=(sc is not None),
        scope=sc,
        budget_ms=bms,
        name=nm,
    )
    for i, (sc, bms, nm) in enumerate(EXPERT_V8_SPECS)
]
EXPERT_V8_ACTION_NAMES: list[str] = [s.name for s in EXPERT_V8_ACTION_TABLE]
NUM_EXPERT_V8_ACTIONS = len(EXPERT_V8_ACTION_TABLE)

EXPERT_V8_FEATURE_NAMES = EXPERT_V7_FEATURE_NAMES + [
    "rcdr_pred_drift_norm",
    "rcdr_ewma_miscoverage",
    "rcdr_late_op_ratio",
    "ltm_active_pressure_norm",
    "sim_wait_rate_norm",
    "replan_budget_spent_norm",
]
EXPERT_V8_STATE_DIM = len(EXPERT_V8_FEATURE_NAMES)

V8_MAX_OPS = 12
V8_OP_FEAT_DIM = 10
V8_MAX_AGVS = 8
V8_AGV_FEAT_DIM = 8
V8_GRAPH_SUMMARY_DIM = EXPERT_V8_STATE_DIM + V8_OP_FEAT_DIM + V8_AGV_FEAT_DIM


def decode_expert_v8_action(action_idx: int) -> ActionSpec:
    """Decode integer action index inside the 9-action V8 Multi-Expert Portfolio."""
    if 0 <= action_idx < NUM_EXPERT_V8_ACTIONS:
        return EXPERT_V8_ACTION_TABLE[action_idx]
    raise ValueError(
        f"Invalid expert_v8 action index {action_idx}, must be in [0, {NUM_EXPERT_V8_ACTIONS - 1}]"
    )


def compute_expert_v8_action_mask(
    sim: Any,
    last_replan_tick: int,
) -> np.ndarray:
    """Compute strictly causal 9-action legal mask for V8 Multi-Expert Portfolio."""
    return compute_expert_action_mask(sim, last_replan_tick)


def compute_rcdr_drift_signals(
    sim: Any,
    tracker: Any,
    decision_logs: Optional[list[dict[str, Any]]] = None,
) -> tuple[float, float, float, float, float, float]:
    """Compute strictly causal IJAETR-26 RCDR rolling conformal drift & LTM status features."""
    tick = int(getattr(sim, "_step_cursor", 0))
    faulted_set = set(getattr(sim, "_faulted_agvs", set()))

    def _tstat(t: Any) -> str:
        s = getattr(t, "status", "")
        return getattr(s, "value", str(s)).lower()

    def _astat(a: Any) -> str:
        s = getattr(a, "state", "")
        return getattr(s, "value", str(s)).lower()

    agv_rem = {a.agv_id: max(0.0, float(getattr(a, "remaining_job_steps", 0))) for a in sim.agvs}
    est_active_cmax = float(tick)
    for t in getattr(sim, "tasks", []):
        if t.release_step > tick or getattr(t, "hidden_until_release", False):
            continue
        st = _tstat(t)
        if st in ("completed", "failed"):
            continue
        rem = agv_rem.get(t.assigned_agv or "", 14.0) if t.assigned_agv else 18.0
        if t.assigned_agv in faulted_set and st == "assigned":
            rem = max(rem, 32.0)
        est_active_cmax = max(est_active_cmax, float(tick) + max(2.0, rem))

    ops = getattr(tracker, "operations", []) if tracker is not None else []
    uncompleted_ops = [
        op for op in ops
        if not (getattr(op, "started", False) and getattr(op, "end_step", None) is not None and tick >= int(op.end_step))
    ]
    late_ops = 0
    for op in uncompleted_ops:
        if getattr(op, "started", False) and getattr(op, "end_step", None) is not None:
            est_en = float(op.end_step)
        else:
            est_en = max(float(tick), est_active_cmax - 4.0) + float(getattr(op, "proc_steps", 15))
        if est_en > float(getattr(op, "due_step", 10**6)):
            late_ops += 1
        est_active_cmax = max(est_active_cmax, est_en)
    rcdr_late_op_ratio = float(late_ops) / float(len(uncompleted_ops)) if uncompleted_ops else 0.0

    logs = decision_logs or []
    ewma_miscov = 0.0
    pred_drift = 0.0
    if logs:
        last_pred_mk = 0
        for row in reversed(logs):
            pmk = int(row.get("predicted_makespan", 0))
            if pmk > 0:
                last_pred_mk = pmk
                break
        if last_pred_mk > 0:
            pred_drift = float(np.clip(max(0.0, est_active_cmax - float(last_pred_mk)) / 80.0, 0.0, 1.0))

        # Rolling EWMA conformal interval miscoverage from prior accepted prediction checkpoints
        valid_hist = [row for row in logs if row.get("predicted_makespan", 0) > 0]
        if valid_hist:
            alpha_ewma = 0.35
            for idx_r, row in enumerate(valid_hist[-8:]):
                p_mk = float(row.get("predicted_makespan", 0))
                if idx_r + 1 < len(valid_hist[-8:]):
                    nxt = valid_hist[-8:][idx_r + 1]
                    nxt_mk = float(nxt.get("predicted_makespan", p_mk))
                    slip = max(0.0, nxt_mk - p_mk) > 10.0
                else:
                    slip = (est_active_cmax - p_mk) > 12.0
                ewma_miscov = (1.0 - alpha_ewma) * ewma_miscov + alpha_ewma * (1.0 if slip else 0.0)

    tw_map = getattr(sim, "_traffic_weights_q3", {}) or {}
    ltm_pressure = float(np.clip(float(sum(tw_map.values())) / 120.0, 0.0, 1.0))
    blocked_since = getattr(sim, "_blocked_since", {})
    n_wait = sum(
        1
        for a in sim.agvs
        if a.agv_id not in faulted_set and (a.agv_id in blocked_since or _astat(a) in ("blocked", "parking"))
    )
    sim_wait_rate = float(np.clip(float(n_wait) / max(1.0, float(len(sim.agvs))), 0.0, 1.0))
    replan_spent = float(np.clip(float(len(logs)) / 24.0, 0.0, 1.0))
    return (
        pred_drift,
        float(np.clip(ewma_miscov, 0.0, 1.0)),
        rcdr_late_op_ratio,
        ltm_pressure,
        sim_wait_rate,
        replan_spent,
    )


def compute_expert_v8_state(
    sim: Any,
    tracker: Any,
    base_obs: Optional[np.ndarray] = None,
    *,
    last_replan_tick: int = -999,
    recent_conflict_ticks: int = 0,
    decision_logs: Optional[list[dict[str, Any]]] = None,
) -> np.ndarray:
    """Compute strictly causal 36-D V8 state combining 30-D V7 signals + 6 RCDR/LTM drift signals."""
    obs30 = compute_expert_v7_state(
        sim,
        tracker,
        base_obs,
        last_replan_tick=last_replan_tick,
        recent_conflict_ticks=recent_conflict_ticks,
        decision_logs=decision_logs,
    )
    extra6 = np.array(
        compute_rcdr_drift_signals(sim, tracker, decision_logs=decision_logs),
        dtype=np.float32,
    )
    return np.concatenate([obs30, extra6]).astype(np.float32)


def compute_expert_v8_dual_graph(
    sim: Any,
    tracker: Any,
    base_obs: Optional[np.ndarray] = None,
    *,
    last_replan_tick: int = -999,
    recent_conflict_ticks: int = 0,
    decision_logs: Optional[list[dict[str, Any]]] = None,
) -> dict[str, np.ndarray]:
    """Compute strictly G3-causal E2G-MoE Precedence-Operation & AGV competition dual graph."""
    global_vec = compute_expert_v8_state(
        sim,
        tracker,
        base_obs,
        last_replan_tick=last_replan_tick,
        recent_conflict_ticks=recent_conflict_ticks,
        decision_logs=decision_logs,
    )
    tick = int(getattr(sim, "_step_cursor", 0))
    faulted_set = set(getattr(sim, "_faulted_agvs", set()))
    admitted_set = getattr(sim, "_admitted_agvs", None)
    blocked_since = getattr(sim, "_blocked_since", {})
    net = getattr(sim, "network", None)

    def _tstat(t: Any) -> str:
        s = getattr(t, "status", "")
        return getattr(s, "value", str(s)).lower()

    def _astat(a: Any) -> str:
        s = getattr(a, "state", "")
        return getattr(s, "value", str(s)).lower()

    task_map = {t.task_id: t for t in getattr(sim, "tasks", [])}
    ops_raw = sorted(
        list(getattr(tracker, "operations", []) if tracker is not None else []),
        key=lambda o: (getattr(o, "due_step", 9999), getattr(o, "op_id", "")),
    )[:V8_MAX_OPS]
    agvs_raw = sorted(list(getattr(sim, "agvs", [])), key=lambda a: a.agv_id)[:V8_MAX_AGVS]

    op_idx_by_id = {op.op_id: i for i, op in enumerate(ops_raw)}
    agv_idx_by_id = {a.agv_id: i for i, a in enumerate(agvs_raw)}

    op_nodes = np.zeros((V8_MAX_OPS, V8_OP_FEAT_DIM), dtype=np.float32)
    op_mask = np.zeros((V8_MAX_OPS,), dtype=np.float32)
    op_adj = np.zeros((3, V8_MAX_OPS, V8_MAX_OPS), dtype=np.float32)
    agv_nodes = np.zeros((V8_MAX_AGVS, V8_AGV_FEAT_DIM), dtype=np.float32)
    agv_mask = np.zeros((V8_MAX_AGVS,), dtype=np.float32)
    agv_op_adj = np.zeros((V8_MAX_AGVS, V8_MAX_OPS), dtype=np.float32)

    bay_active_cnt: dict[str, int] = {}
    op_completed: dict[str, bool] = {}
    for op in ops_raw:
        is_comp = bool(
            getattr(op, "started", False)
            and getattr(op, "end_step", None) is not None
            and tick >= int(op.end_step)
        )
        op_completed[op.op_id] = is_comp
        if not is_comp:
            bid = str(getattr(op, "bay_id", ""))
            bay_active_cnt[bid] = bay_active_cnt.get(bid, 0) + 1

    # Bipartite counts over strictly visible released tasks
    agv_vis_q: dict[str, int] = {a.agv_id: 0 for a in agvs_raw}
    agv_picked: dict[str, float] = {a.agv_id: 0.0 for a in agvs_raw}
    for t in getattr(sim, "tasks", []):
        if t.release_step > tick or getattr(t, "hidden_until_release", False):
            continue
        st = _tstat(t)
        if st in ("completed", "failed"):
            continue
        aid = getattr(t, "assigned_agv", None)
        if aid in agv_vis_q:
            agv_vis_q[aid] += 1
            if st == "picked":
                agv_picked[aid] = 1.0
            o_i = op_idx_by_id.get(getattr(t, "kit_op_id", "") or "")
            a_i = agv_idx_by_id.get(aid)
            if o_i is not None and a_i is not None:
                n_kits = max(1, len(getattr(ops_raw[o_i], "kit_task_ids", ()) or ()))
                agv_op_adj[a_i, o_i] += 1.0 / float(n_kits)

    for i, op in enumerate(ops_raw):
        op_mask[i] = 1.0
        is_st = 1.0 if getattr(op, "started", False) else 0.0
        is_cp = 1.0 if op_completed.get(op.op_id, False) else 0.0
        ktids = getattr(op, "kit_task_ids", ()) or ()
        ready_score = 0.0
        has_kb = 0.0
        for tid in ktids:
            t = task_map.get(tid)
            if t is None or t.release_step > tick or getattr(t, "hidden_until_release", False):
                continue
            st = _tstat(t)
            if st == "completed":
                ready_score += 1.0
            elif st in ("picked", "loading", "to_dropoff", "unloading"):
                ready_score += 0.65
            elif st == "assigned":
                ready_score += 0.30
            if getattr(t, "kitting_bottleneck_active", False) or (
                st == "assigned" and getattr(t, "assigned_agv", None) in faulted_set
            ):
                has_kb = 1.0
        kit_ready_ratio = float(ready_score) / float(max(1, len(ktids)))
        proc_s = float(getattr(op, "proc_steps", 0))
        slack_norm = float(np.clip((float(getattr(op, "due_step", tick)) - float(tick) - proc_s) / 100.0, -1.0, 2.0))
        is_urg = 1.0 if getattr(op, "is_urgent", False) else 0.0
        from amr_sched.cvcr.snapshot import op_predecessors
        preds = op_predecessors(op)
        pred_done = 1.0 if (not preds or all(op_completed.get(pid, False) for pid in preds)) else 0.0
        bay_flex = float(np.clip(float(len(getattr(op, "eligible_bays", ()))) / 4.0, 0.0, 1.0))
        crew_flex = float(np.clip(float(len(getattr(op, "eligible_crews", ()))) / 3.0, 0.0, 1.0))
        bay_cnt = float(np.clip(float(bay_active_cnt.get(str(getattr(op, "bay_id", "")), 0)) / 4.0, 0.0, 1.0))

        op_nodes[i] = np.array(
            [
                is_st,
                is_cp,
                kit_ready_ratio,
                slack_norm,
                is_urg,
                pred_done,
                bay_flex,
                crew_flex,
                bay_cnt,
                has_kb,
            ],
            dtype=np.float32,
        )

        # Self loops across typed channels + Precedence & Mutex edges
        op_adj[:, i, i] = 1.0
        for pid in preds:
            if pid in op_idx_by_id:
                p_i = op_idx_by_id[pid]
                op_adj[0, p_i, i] = 1.0  # pred -> succ
                op_adj[1, i, p_i] = 1.0  # succ -> pred
        if not is_cp:
            bid_i = getattr(op, "bay_id", None)
            cid_i = getattr(op, "crew_id", None)
            for j, op_j in enumerate(ops_raw):
                if i != j and not op_completed.get(op_j.op_id, False):
                    if getattr(op_j, "bay_id", None) == bid_i or getattr(op_j, "crew_id", None) == cid_i:
                        op_adj[2, i, j] = 1.0

    # Row-normalize adjacency channels cleanly for stable GraphConv aggregation
    for ch in range(3):
        row_sums = np.maximum(1.0, op_adj[ch].sum(axis=1, keepdims=True))
        op_adj[ch] = op_adj[ch] / row_sums

    for a_i, agv in enumerate(agvs_raw):
        agv_mask[a_i] = 1.0
        aid = agv.agv_id
        st = _astat(agv)
        is_f = 1.0 if (aid in faulted_set or getattr(agv, "faulted", False) or st == "faulted") else 0.0
        is_b = 1.0 if (getattr(agv, "cur_task", None) is not None or st in ("to_pickup", "loading", "to_dropoff", "unloading")) else 0.0
        is_w = 1.0 if (aid in blocked_since or st in ("blocked", "parking")) else 0.0
        q_dep = float(np.clip(float(agv_vis_q.get(aid, 0)) / 5.0, 0.0, 1.0))
        rem_n = float(np.clip(float(getattr(agv, "remaining_job_steps", 0)) / 60.0, 0.0, 1.0))
        has_pk = float(agv_picked.get(aid, 0.0))
        unadm = 1.0 if (admitted_set is not None and aid not in admitted_set) else 0.0
        cur_n = getattr(agv, "cur_node", None) or getattr(agv, "start_node", "")
        deg_n = float(np.clip(float(len(net.out_edges(cur_n))) / 5.0, 0.0, 1.0)) if (net is not None and cur_n and hasattr(net, "out_edges")) else 0.5
        agv_nodes[a_i] = np.array([is_f, is_b, is_w, q_dep, rem_n, has_pk, unadm, deg_n], dtype=np.float32)

    valid_ops = int(max(1.0, float(op_mask.sum())))
    valid_agvs = int(max(1.0, float(agv_mask.sum())))
    op_pool = op_nodes[: len(ops_raw)].mean(axis=0) if ops_raw else np.zeros(V8_OP_FEAT_DIM, dtype=np.float32)
    agv_pool = agv_nodes[: len(agvs_raw)].mean(axis=0) if agvs_raw else np.zeros(V8_AGV_FEAT_DIM, dtype=np.float32)
    bandit_vec = np.concatenate([global_vec, op_pool, agv_pool]).astype(np.float32)

    return {
        "global_vec": global_vec,
        "op_nodes": op_nodes,
        "op_adj": op_adj,
        "op_mask": op_mask,
        "agv_nodes": agv_nodes,
        "agv_op_adj": agv_op_adj,
        "agv_mask": agv_mask,
        "bandit_vec": bandit_vec,
    }


def should_query_v8_controller(
    obs_v8: np.ndarray,
    exp_mask: np.ndarray,
    tick: int,
    last_decision_tick: int,
    epoch_interval: int = 12,
    prev_emerg: Any = False,
    in_secondary_hold: bool = False,
) -> tuple[bool, tuple[bool, bool]]:
    """IJAETR-26 RCDR Rolling Conformal Drift & kickoff-aware SMDP trigger (strict G3 causal)."""
    cur_f = bool(float(obs_v8[18]) > 0.5)
    cur_u = bool(float(obs_v8[19]) > 0.5)
    if isinstance(prev_emerg, tuple) and len(prev_emerg) == 2:
        prev_f, prev_u = bool(prev_emerg[0]), bool(prev_emerg[1])
    else:
        prev_f = prev_u = bool(prev_emerg)
    cur_state = (cur_f, cur_u)

    if in_secondary_hold:
        return False, cur_state
    if not any(exp_mask[idx] == 1 for idx in range(1, min(len(exp_mask), NUM_EXPERT_V8_ACTIONS))):
        return False, cur_state

    # 1. Initial episode kickoff optimization as soon as first tasks release (tick <= 8)
    if last_decision_tick < 0 and tick <= 8:
        return True, cur_state

    # 2. Rising endogenous disturbance edge (fault / urgent-kitting bottleneck)
    rising_f = cur_f and not prev_f
    rising_u = cur_u and not prev_u
    if (rising_f or rising_u) and float(obs_v8[16]) < 0.5:
        return True, cur_state
    if cur_f or cur_u:
        return (tick - last_decision_tick) >= min(epoch_interval, 10), cur_state

    if float(obs_v8[11]) >= 0.92 or float(obs_v8[2]) >= 0.92:
        return False, cur_state

    # 3. IJAETR-26 RCDR rolling conformal prediction drift early check (min 6-tick spacing)
    pred_drift = float(obs_v8[30]) if len(obs_v8) > 30 else 0.0
    ewma_miscov = float(obs_v8[31]) if len(obs_v8) > 31 else 0.0
    late_ops = float(obs_v8[32]) if len(obs_v8) > 32 else 0.0
    rcdr_alert = (pred_drift >= 0.18 or ewma_miscov >= 0.45) and (tick - last_decision_tick) >= 6 and float(obs_v8[11]) < 0.88
    if rcdr_alert:
        return True, cur_state

    # 4. Periodic calm leverage checkpoint
    cooldown_ready = (tick - last_decision_tick) >= epoch_interval
    has_leverage = (
        float(obs_v8[8]) >= 0.08
        or float(obs_v8[13]) >= 0.10
        or float(obs_v8[9]) >= 0.04
        or float(obs_v8[4]) < 0.35
        or float(obs_v8[17]) >= 0.22
        or float(obs_v8[10]) >= 0.08
        or late_ops >= 0.15
        or (float(obs_v8[12]) >= 0.85 and (tick - last_decision_tick) >= epoch_interval)
    )
    return bool(cooldown_ready and has_leverage), cur_state


# ---------------------------------------------------------------------------
# V9 Joint CP-SAT + Windowed LTM + Soft Corridor Guides + Calibrated RCDR
# ---------------------------------------------------------------------------

EXPERT_V9_SPECS: tuple[tuple[Optional[str], float, str], ...] = (
    (None, 0.0, "KEEP"),
    ("V9_E1_FAST_RESCUE", 20.0, "V9_E1_FAST_RESCUE_20MS"),
    ("V9_E2_CRIT_PROD_CI", 50.0, "V9_E2_CRIT_PROD_CI_50MS"),
    ("V9_E3_JOINT_PC_HYS", 50.0, "V9_E3_JOINT_PC_HYS_50MS"),
    ("V9_E7_CPSAT_JOINT_GUIDE", 50.0, "V9_E7_CPSAT_JOINT_50MS"),
    ("V9_E4_LTM_GUIDE_CI", 50.0, "V9_E4_LTM_GUIDE_CI_50MS"),
    ("V9_E5_AGGRESSIVE_GUIDE", 100.0, "V9_E5_AGGRESSIVE_GUIDE_100MS"),
    ("V9_E6_BALANCE_KM_GUIDE", 100.0, "V9_E6_BALANCE_KM_NOG_100MS"),
    ("V9_E7_CPSAT_JOINT_GUIDE", 100.0, "V9_E7_CPSAT_JOINT_GUIDE_100MS"),
)

EXPERT_V9_ACTION_TABLE: list[ActionSpec] = [
    ActionSpec(
        action_idx=i,
        is_replan=(sc is not None),
        scope=sc,
        budget_ms=bms,
        name=nm,
    )
    for i, (sc, bms, nm) in enumerate(EXPERT_V9_SPECS)
]
EXPERT_V9_ACTION_NAMES: list[str] = [s.name for s in EXPERT_V9_ACTION_TABLE]
NUM_EXPERT_V9_ACTIONS = len(EXPERT_V9_ACTION_TABLE)

EXPERT_V9_FEATURE_NAMES = EXPERT_V8_FEATURE_NAMES + [
    "min_open_window_steps_norm",
    "cpsat_coupling_pressure_norm",
]
EXPERT_V9_STATE_DIM = len(EXPERT_V9_FEATURE_NAMES)

V9_MAX_OPS = V8_MAX_OPS
V9_OP_FEAT_DIM = V8_OP_FEAT_DIM
V9_MAX_AGVS = V8_MAX_AGVS
V9_AGV_FEAT_DIM = V8_AGV_FEAT_DIM
V9_GRAPH_SUMMARY_DIM = EXPERT_V9_STATE_DIM + V9_OP_FEAT_DIM + V9_AGV_FEAT_DIM


def decode_expert_v9_action(action_idx: int) -> ActionSpec:
    """Decode integer action index inside the 9-action V9 Portfolio."""
    if 0 <= action_idx < NUM_EXPERT_V9_ACTIONS:
        return EXPERT_V9_ACTION_TABLE[action_idx]
    raise ValueError(
        f"Invalid expert_v9 action index {action_idx}, must be in [0, {NUM_EXPERT_V9_ACTIONS - 1}]"
    )


def compute_expert_v9_action_mask(
    sim: Any,
    last_replan_tick: int,
) -> np.ndarray:
    """Compute strictly causal 9-action legal mask for V9 Multi-Expert Portfolio."""
    return compute_expert_action_mask(sim, last_replan_tick)


def compute_v9_window_and_coupling_signals(
    sim: Any,
    tracker: Any,
) -> tuple[float, float]:
    """Compute strictly G3-causal open compute-window tightness and joint resource coupling pressure."""
    tick = int(getattr(sim, "_step_cursor", 0))
    faulted_set = set(getattr(sim, "_faulted_agvs", set()))

    def _tstat(t: Any) -> str:
        s = getattr(t, "status", "")
        return getattr(s, "value", str(s)).lower()

    open_ws: list[int] = []
    for t in getattr(sim, "tasks", []):
        if int(getattr(t, "release_step", 0)) > tick:
            continue
        st_t = _tstat(t)
        if (
            st_t == "assigned"
            and t.assigned_agv in faulted_set
            and tick >= int(getattr(t, "secondary_window_open_step", 0))
        ):
            w_val = int(getattr(t, "adjustable_window_steps", 6))
            if w_val > 0:
                open_ws.append(w_val)
        elif st_t in ("pending", "unassigned", "assigned") and getattr(t, "kitting_bottleneck_active", False):
            w_val = int(getattr(t, "adjustable_window_steps", 3))
            if w_val > 0:
                open_ws.append(w_val)
    min_window_norm = min(1.0, float(min(open_ws)) / 6.0) if open_ws else 1.0

    ops_unstarted = [
        op
        for op in (getattr(tracker, "operations", []) if tracker is not None else [])
        if not getattr(op, "started", False)
    ]
    if not ops_unstarted:
        return float(min_window_norm), 0.0

    bay_cnt: dict[str, int] = {}
    crew_cnt: dict[int, int] = {}
    tool_cnt: dict[int, int] = {}
    for op in ops_unstarted:
        for b in (getattr(op, "eligible_bays", ()) or (op.bay_id,)):
            bay_cnt[str(b)] = bay_cnt.get(str(b), 0) + 1
        for c in (getattr(op, "eligible_crews", ()) or (op.crew_id,)):
            crew_cnt[int(c)] = crew_cnt.get(int(c), 0) + 1
        tool_id = int(getattr(op, "tool_id", -1))
        tool_cnt[tool_id] = tool_cnt.get(tool_id, 0) + 1

    coupled = 0
    for op in ops_unstarted:
        bays = getattr(op, "eligible_bays", ()) or (op.bay_id,)
        crews = getattr(op, "eligible_crews", ()) or (op.crew_id,)
        tool_id = int(getattr(op, "tool_id", -1))
        if (
            max((bay_cnt.get(str(b), 0) for b in bays), default=0) >= 2
            or max((crew_cnt.get(int(c), 0) for c in crews), default=0) >= 2
            or tool_cnt.get(tool_id, 0) >= 2
        ):
            coupled += 1
    coupling_pressure = min(1.0, float(coupled) / float(len(ops_unstarted)))
    return float(min_window_norm), float(coupling_pressure)


def compute_expert_v9_state(
    sim: Any,
    tracker: Any,
    base_obs: Optional[np.ndarray] = None,
    *,
    last_replan_tick: int = -999,
    recent_conflict_ticks: int = 0,
    decision_logs: Optional[list[dict[str, Any]]] = None,
) -> np.ndarray:
    """Compute strictly G3-causal 38-D V9 state combining 36-D V8 state + window tightness & CP-SAT coupling."""
    obs36 = compute_expert_v8_state(
        sim,
        tracker,
        base_obs,
        last_replan_tick=last_replan_tick,
        recent_conflict_ticks=recent_conflict_ticks,
        decision_logs=decision_logs,
    )
    extra2 = np.array(
        compute_v9_window_and_coupling_signals(sim, tracker),
        dtype=np.float32,
    )
    return np.concatenate([obs36, extra2]).astype(np.float32)


def compute_expert_v9_dual_graph(
    sim: Any,
    tracker: Any,
    base_obs: Optional[np.ndarray] = None,
    *,
    last_replan_tick: int = -999,
    recent_conflict_ticks: int = 0,
    decision_logs: Optional[list[dict[str, Any]]] = None,
) -> dict[str, np.ndarray]:
    """Compute strictly G3-causal V9 dual-graph observation tensors (38-D global + E2G + 56-D bandit_vec)."""
    dg = compute_expert_v8_dual_graph(
        sim,
        tracker,
        base_obs,
        last_replan_tick=last_replan_tick,
        recent_conflict_ticks=recent_conflict_ticks,
        decision_logs=decision_logs,
    )
    extra2 = np.array(
        compute_v9_window_and_coupling_signals(sim, tracker),
        dtype=np.float32,
    )
    global38 = np.concatenate([dg["global_vec"], extra2]).astype(np.float32)
    graph_tail = dg["bandit_vec"][EXPERT_V8_STATE_DIM:]
    dg["global_vec"] = global38
    dg["bandit_vec"] = np.concatenate([global38, graph_tail]).astype(np.float32)
    return dg


def should_query_v9_controller(
    obs_v9: np.ndarray,
    exp_mask: np.ndarray,
    tick: int,
    last_decision_tick: int,
    epoch_interval: int = 14,
    prev_emerg: Any = False,
    in_secondary_hold: bool = False,
) -> tuple[bool, tuple[bool, bool]]:
    """Calibrated RCDR conformal drift & kickoff-aware SMDP trigger (strict G3 causal)."""
    cur_f = bool(float(obs_v9[18]) > 0.5)
    cur_u = bool(float(obs_v9[19]) > 0.5)
    if isinstance(prev_emerg, tuple) and len(prev_emerg) == 2:
        prev_f, prev_u = bool(prev_emerg[0]), bool(prev_emerg[1])
    else:
        prev_f = prev_u = bool(prev_emerg)
    cur_state = (cur_f, cur_u)

    if in_secondary_hold:
        return False, cur_state
    if not any(exp_mask[idx] == 1 for idx in range(1, min(len(exp_mask), NUM_EXPERT_V9_ACTIONS))):
        return False, cur_state

    # 1. Initial episode kickoff optimization as soon as first tasks release (tick <= 8)
    if last_decision_tick < 0 and tick <= 8:
        return True, cur_state

    # 2. Rising endogenous disturbance edge (fault / urgent-kitting bottleneck)
    rising_f = cur_f and not prev_f
    rising_u = cur_u and not prev_u
    if (rising_f or rising_u) and float(obs_v9[16]) < 0.5:
        return True, cur_state
    if cur_f or cur_u:
        return (tick - last_decision_tick) >= min(epoch_interval, 10), cur_state

    if float(obs_v9[11]) >= 0.90 or float(obs_v9[2]) >= 0.90:
        return False, cur_state

    # 3. Calibrated RCDR rolling conformal prediction drift early check (min 10-tick spacing)
    pred_drift = float(obs_v9[30]) if len(obs_v9) > 30 else 0.0
    ewma_miscov = float(obs_v9[31]) if len(obs_v9) > 31 else 0.0
    late_ops = float(obs_v9[32]) if len(obs_v9) > 32 else 0.0
    rcdr_alert = (
        (pred_drift >= 0.30 or ewma_miscov >= 0.60)
        and (tick - last_decision_tick) >= 10
        and float(obs_v9[11]) < 0.85
    )
    if rcdr_alert:
        return True, cur_state

    # 4. Periodic calm leverage checkpoint (high-leverage only, suppressing wasteful late tail queries)
    cooldown_ready = (tick - last_decision_tick) >= epoch_interval
    has_leverage = (
        float(obs_v9[8]) >= 0.12
        or float(obs_v9[13]) >= 0.15
        or float(obs_v9[9]) >= 0.06
        or float(obs_v9[4]) < 0.32
        or float(obs_v9[17]) >= 0.25
        or float(obs_v9[10]) >= 0.12
        or late_ops >= 0.22
    )
    return bool(cooldown_ready and has_leverage), cur_state


EXPERT_V10_SPECS: list[tuple[Optional[str], float, str]] = [
    (None, 0.0, "KEEP"),
    ("V10_E1_FAST_RESCUE", 20.0, "V10_E1_FAST_RESCUE_20MS"),
    ("V10_E2_CRIT_PROD_CI", 50.0, "V10_E2_CRIT_PROD_CI_50MS"),
    ("V10_E3_JOINT_PC_HYS", 50.0, "V10_E3_JOINT_PC_HYS_50MS"),
    ("V10_E7_COND_CPSAT", 50.0, "V10_E7_COND_CPSAT_50MS"),
    ("V10_E4_LTM_GUIDE_CI", 50.0, "V10_E4_LTM_GUIDE_CI_50MS"),
    ("V10_E5_AGGRESSIVE_GUIDE", 100.0, "V10_E5_AGGRESSIVE_GUIDE_100MS"),
    ("V10_E6_BALANCE_KM_LRHO", 100.0, "V10_E6_BALANCE_KM_LRHO_100MS"),
    ("V10_E7_COND_CPSAT", 100.0, "V10_E7_COND_CPSAT_100MS"),
]
EXPERT_V10_ACTION_TABLE: list[ActionSpec] = [
    ActionSpec(
        action_idx=i,
        is_replan=(sc is not None),
        scope=sc,
        budget_ms=bms,
        name=nm,
    )
    for i, (sc, bms, nm) in enumerate(EXPERT_V10_SPECS)
]
EXPERT_V10_ACTION_NAMES: list[str] = [s.name for s in EXPERT_V10_ACTION_TABLE]
NUM_EXPERT_V10_ACTIONS = len(EXPERT_V10_ACTION_TABLE)


def decode_expert_v10_action(action_idx: int) -> ActionSpec:
    """Decode integer action index inside the 9-action V10 Portfolio."""
    if 0 <= action_idx < NUM_EXPERT_V10_ACTIONS:
        return EXPERT_V10_ACTION_TABLE[action_idx]
    raise ValueError(
        f"Invalid expert_v10 action index {action_idx}, must be in [0, {NUM_EXPERT_V10_ACTIONS - 1}]"
    )


def compute_expert_v10_action_mask(
    sim: Any,
    last_replan_tick: int,
) -> np.ndarray:
    """Compute strictly causal 9-action legal mask for V10 Multi-Expert Portfolio."""
    return compute_expert_action_mask(sim, last_replan_tick)


def should_query_v10_controller(
    obs_v9: np.ndarray,
    exp_mask: np.ndarray,
    tick: int,
    last_decision_tick: int,
    epoch_interval: int = 16,
    prev_emerg: Any = False,
    in_secondary_hold: bool = False,
) -> tuple[bool, tuple[bool, bool]]:
    """Churn-calibrated V10 RCDR trigger suppressing low-leverage late/congested churn."""
    cur_f = bool(float(obs_v9[18]) > 0.5)
    cur_u = bool(float(obs_v9[19]) > 0.5)
    if isinstance(prev_emerg, tuple) and len(prev_emerg) == 2:
        prev_f, prev_u = bool(prev_emerg[0]), bool(prev_emerg[1])
    else:
        prev_f = prev_u = bool(prev_emerg)
    cur_state = (cur_f, cur_u)

    if in_secondary_hold:
        return False, cur_state
    if not any(exp_mask[idx] == 1 for idx in range(1, min(len(exp_mask), NUM_EXPERT_V10_ACTIONS))):
        return False, cur_state

    # 1. Episode kickoff optimization once first tasks release
    if last_decision_tick < 0 and tick <= 8:
        return True, cur_state

    # 2. Immediate emergency trigger on rising fault or urgent kitting disturbance
    rising_f = cur_f and not prev_f
    rising_u = cur_u and not prev_u
    if (rising_f or rising_u) and float(obs_v9[16]) < 0.5:
        return True, cur_state
    if cur_f or cur_u:
        return (tick - last_decision_tick) >= min(epoch_interval, 10), cur_state

    late_ops = float(obs_v9[32]) if len(obs_v9) > 32 else 0.0
    progress_t = float(obs_v9[11]) if len(obs_v9) > 11 else 0.0
    if progress_t >= 0.80 and late_ops < 0.35:
        return False, cur_state
    if float(obs_v9[2]) >= 0.85:
        return False, cur_state

    # 3. Tighter RCDR drift threshold (avoid over-triggering on benign grid waiting noise)
    pred_drift = float(obs_v9[30]) if len(obs_v9) > 30 else 0.0
    ewma_miscov = float(obs_v9[31]) if len(obs_v9) > 31 else 0.0
    rcdr_alert = (
        ((pred_drift >= 0.42 and ewma_miscov >= 0.65) or pred_drift >= 0.65)
        and (tick - last_decision_tick) >= 14
        and progress_t < 0.80
    )
    if rcdr_alert:
        return True, cur_state

    # 4. Calibrated calm leverage check with extended cooldown
    cooldown_ready = (tick - last_decision_tick) >= epoch_interval
    has_leverage = (
        float(obs_v9[8]) >= 0.16
        or float(obs_v9[13]) >= 0.18
        or float(obs_v9[9]) >= 0.10
        or float(obs_v9[4]) < 0.25
        or float(obs_v9[17]) >= 0.30
        or late_ops >= 0.28
    )
    return bool(cooldown_ready and has_leverage), cur_state


def should_query_v11_controller(
    obs_v9: np.ndarray,
    exp_mask: np.ndarray,
    tick: int,
    last_decision_tick: int,
    epoch_interval: int = 16,
    prev_emerg: Any = False,
    in_secondary_hold: bool = False,
    stalled_picked: bool = False,
    stall_cooldown: int = 12,
) -> tuple[bool, tuple[bool, bool]]:
    """V11 RCDR trigger: preserves V10 calm gating while unmuting stalled loaded vehicles."""
    cur_f = bool(float(obs_v9[18]) > 0.5)
    cur_u = bool(float(obs_v9[19]) > 0.5)
    if isinstance(prev_emerg, tuple) and len(prev_emerg) == 2:
        prev_f, prev_u = bool(prev_emerg[0]), bool(prev_emerg[1])
    else:
        prev_f = prev_u = bool(prev_emerg)
    cur_state = (cur_f, cur_u)

    if (
        stalled_picked
        and not in_secondary_hold
        and any(exp_mask[idx] == 1 for idx in range(1, min(len(exp_mask), NUM_EXPERT_V10_ACTIONS)))
        and (last_decision_tick < 0 or (tick - last_decision_tick) >= int(stall_cooldown))
    ):
        return True, cur_state

    return should_query_v10_controller(
        obs_v9,
        exp_mask,
        tick,
        last_decision_tick,
        epoch_interval=epoch_interval,
        prev_emerg=prev_emerg,
        in_secondary_hold=in_secondary_hold,
    )


EXPERT_V12_SPECS: list[tuple[Optional[str], float, str]] = [
    (None, 0.0, "KEEP"),
    ("V12_SURGICAL_TPG_REPAIR", 15.0, "V12_SURGICAL_TPG_REPAIR_15MS"),
    ("V12_RESCUE_DETOUR_FAST", 20.0, "V12_RESCUE_DETOUR_FAST_20MS"),
    ("V12_CRIT_PROD_DESTROY_S", 50.0, "V12_CRIT_PROD_DESTROY_S_50MS"),
    ("V12_PACER_CONSERVATIVE", 50.0, "V12_PACER_CONSERVATIVE_50MS"),
    ("V12_BALANCED_DESTROY_M", 100.0, "V12_BALANCED_DESTROY_M_100MS"),
    ("V12_BALANCED_DESTROY_L", 100.0, "V12_BALANCED_DESTROY_L_100MS"),
    ("V12_AGGRESSIVE_GUIDE_TPG", 100.0, "V12_AGGRESSIVE_GUIDE_TPG_100MS"),
    ("V12_SAFE_CPSAT_JOINT", 100.0, "V12_SAFE_CPSAT_JOINT_100MS"),
]
EXPERT_V12_ACTION_TABLE: list[ActionSpec] = [
    ActionSpec(
        action_idx=i,
        is_replan=(sc is not None),
        scope=sc,
        budget_ms=bms,
        name=nm,
    )
    for i, (sc, bms, nm) in enumerate(EXPERT_V12_SPECS)
]
EXPERT_V12_ACTION_NAMES: list[str] = [s.name for s in EXPERT_V12_ACTION_TABLE]
NUM_EXPERT_V12_ACTIONS = len(EXPERT_V12_ACTION_TABLE)

EXPERT_V12_FEATURE_NAMES = list(EXPERT_V9_FEATURE_NAMES) + [
    "edge_gate_stall_ratio",
    "leaf_spur_contention",
    "detour_delta_mean",
    "detour_delta_p90",
    "blocked_ratio",
    "cpsat_safety_headroom",
]
EXPERT_V12_STATE_DIM = len(EXPERT_V12_FEATURE_NAMES)


def decode_expert_v12_action(action_idx: int) -> ActionSpec:
    """Decode integer action index inside the 9-action V12 Top-Venue Synergistic Portfolio."""
    if 0 <= action_idx < NUM_EXPERT_V12_ACTIONS:
        return EXPERT_V12_ACTION_TABLE[action_idx]
    raise ValueError(
        f"Invalid expert_v12 action index {action_idx}, must be in [0, {NUM_EXPERT_V12_ACTIONS - 1}]"
    )


def compute_expert_v12_action_mask(
    sim: Any,
    last_replan_tick: int,
) -> np.ndarray:
    """Compute strictly causal 9-action legal mask for V12 Portfolio."""
    mask = compute_expert_action_mask(sim, last_replan_tick).copy()
    # Action 1 (V12_SURGICAL_TPG_REPAIR) is allowed whenever any non-KEEP action is legal or a leaf/stall conflict exists
    from amr_sched.cvcr.loaded_corridor_repair import detect_stalled_picked_agvs
    if detect_stalled_picked_agvs(sim, stall_threshold=6, include_edge_gate_stalls=True):
        mask[1] = 1
    elif getattr(sim, "_p4_choke_robust", False):
        from amr_sched.cvcr.p4_belief_choke import detect_choke_oscillation_agvs
        if detect_choke_oscillation_agvs(sim):
            mask[1] = 1
    if mask[1] == 0 and getattr(sim, "_p2_execution_aware", False) and getattr(sim, "_perturbation_model", None) is not None:
        cur_tick = int(getattr(sim, "_step_cursor", 0))
        p2_sig = sim._perturbation_model.active_disturbance_sig(sim, cur_tick)
        if p2_sig.get("blocked_picked_agvs") or p2_sig.get("blocked_assigned_agvs"):
            mask[1] = 1
    return mask


def compute_expert_v12_state(
    sim: Any,
    tracker: Any,
    base_obs: Optional[np.ndarray] = None,
    *,
    last_replan_tick: int = 0,
    recent_conflict_ticks: int = 0,
    decision_logs: Optional[list[dict[str, Any]]] = None,
) -> np.ndarray:
    """Compute 44-D strictly causal V12 state (38-D V9 + 6-D Detour-Delta/Haven/CP signals)."""
    from amr_sched.cvcr.snapshot import NativeSnapshotCapture
    from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals

    o9 = compute_expert_v9_state(
        sim,
        tracker,
        base_obs,
        last_replan_tick=last_replan_tick,
        recent_conflict_ticks=recent_conflict_ticks,
        decision_logs=decision_logs,
    )
    snap = NativeSnapshotCapture.capture(sim, tracker.operations, hide_future_releases=True)
    sig = compute_detour_delta_signals(sim.network, snap, simulator=sim, stall_threshold=6)
    sm = sig["summary"]
    win_t = float(o9[36])
    coup = float(o9[37])
    prog = float(o9[11])
    cp_headroom = float(np.clip(coup * win_t * (1.0 if prog < 0.78 else 0.2), 0.0, 1.0))
    extra = np.asarray(
        [
            sm["edge_gate_stall_ratio"],
            sm["leaf_spur_contention"],
            sm["detour_delta_mean"],
            sm["detour_delta_p90"],
            sm["blocked_ratio"],
            cp_headroom,
        ],
        dtype=np.float32,
    )
    return np.concatenate([o9, extra]).astype(np.float32)


def compute_expert_v12_dual_graph(
    sim: Any,
    tracker: Any,
    base_obs: Optional[np.ndarray] = None,
    *,
    last_replan_tick: int = 0,
    recent_conflict_ticks: int = 0,
    decision_logs: Optional[list[dict[str, Any]]] = None,
) -> dict[str, np.ndarray]:
    """Compute 44-D global + E2G dual-graph observation dictionary for RL-CVCR v12."""
    dg = compute_expert_v9_dual_graph(
        sim,
        tracker,
        base_obs,
        last_replan_tick=last_replan_tick,
        recent_conflict_ticks=recent_conflict_ticks,
        decision_logs=decision_logs,
    )
    o12 = compute_expert_v12_state(
        sim,
        tracker,
        base_obs,
        last_replan_tick=last_replan_tick,
        recent_conflict_ticks=recent_conflict_ticks,
        decision_logs=decision_logs,
    )
    dg["global_vec"] = o12
    return dg
