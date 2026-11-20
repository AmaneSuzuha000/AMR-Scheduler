"""Execution-Aware AMR State & Deviation Feedback Contract (P1 · L1/L2 -> L3 CVCR).

Defines the typed, serializable interface through which autonomous navigation (L1) and
traffic execution (L2) report live execution deviations back to the CVCR decision core (L3):
  - Continuous progress, localization source/staleness, and arrival ETA drift (`AmrExecutionStatus`)
  - Dynamic corridor degradation, congestion delay, and temporary blockage (`CorridorFeedback`)
  - Deterministic snapshot-level bundle and adapters from `ExternalExecution`,
    `ExternalHandling`, and `DeviationEvent` (`ExecutionFeedbackPacket`)

All continuous telemetry is validated at ingress and converted deterministically into integer
ticks or Q3 weights before entering CVCR reservation tables, preserving integer-step semantics
and zero-diff backward compatibility when no execution feedback is attached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from amr_sched.monitor import DeviationEvent

EXEC_FEEDBACK_SCHEMA = "amr.exec_feedback.v1"

LOCALIZATION_SOURCES = frozenset({
    "discrete",
    "gazebo",
    "odom",
    "amcl",
    "lidar_slam",
    "scout_hardware",
})

PLANNER_STATES = frozenset({
    "nominal",
    "slowed",
    "detouring",
    "waiting_obstacle",
    "local_failed",
    "stopped",
    "faulted",
    "handling_wait",
})

CORRIDOR_STATES = frozenset({"open", "degraded", "congested", "blocked"})


def _valid_finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


@dataclass(frozen=True)
class AmrExecutionStatus:
    """Validated L1/L2 execution telemetry for a single autonomous mobile robot."""

    robot_id: str
    pose_xy_yaw: Optional[tuple[float, float, float]] = None
    localization_source: str = "discrete"
    active_leg: Optional[tuple[str, str]] = None
    leg_progress: float = 0.0
    eta_drift_steps: int = 0
    speed_scale: float = 1.0
    local_planner_state: str = "nominal"
    receipt_age_s: float = 0.0
    handling_delay_steps: int = 0
    route_id: Optional[str] = None
    fault_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.robot_id, str) or not self.robot_id.strip():
            raise ValueError("robot_id must be a non-empty string")
        if self.localization_source not in LOCALIZATION_SOURCES:
            raise ValueError(f"unsupported localization_source: {self.localization_source!r}")
        if self.local_planner_state not in PLANNER_STATES:
            raise ValueError(f"unsupported local_planner_state: {self.local_planner_state!r}")
        if self.pose_xy_yaw is not None:
            if len(self.pose_xy_yaw) != 3 or not all(_valid_finite(x) for x in self.pose_xy_yaw):
                raise ValueError(f"pose_xy_yaw must be 3 finite floats, got {self.pose_xy_yaw!r}")
        if self.active_leg is not None:
            if len(self.active_leg) != 2 or not all(isinstance(n, str) and n for n in self.active_leg):
                raise ValueError(f"active_leg must be (src, dst) node strings, got {self.active_leg!r}")
        if not _valid_finite(self.leg_progress) or not (0.0 <= float(self.leg_progress) <= 1.0):
            raise ValueError(f"leg_progress must lie in [0, 1], got {self.leg_progress!r}")
        if not _valid_finite(self.speed_scale) or not (0.0 < float(self.speed_scale) <= 2.5):
            raise ValueError(f"speed_scale must lie in (0, 2.5], got {self.speed_scale!r}")
        if not _valid_finite(self.receipt_age_s) or float(self.receipt_age_s) < 0.0:
            raise ValueError(f"receipt_age_s must be non-negative finite, got {self.receipt_age_s!r}")
        if isinstance(self.eta_drift_steps, bool) or not isinstance(self.eta_drift_steps, int):
            raise ValueError(f"eta_drift_steps must be an int, got {self.eta_drift_steps!r}")
        if (
            isinstance(self.handling_delay_steps, bool)
            or not isinstance(self.handling_delay_steps, int)
            or self.handling_delay_steps < 0
        ):
            raise ValueError(f"handling_delay_steps must be non-negative int, got {self.handling_delay_steps!r}")

    @property
    def is_hard_faulted(self) -> bool:
        """Return True when L1 cannot continue without explicit external/supervisor recovery."""
        return self.local_planner_state in ("faulted", "local_failed") or bool(self.fault_reason)

    @property
    def is_degraded(self) -> bool:
        """Return True when actual execution measurably deviates from nominal static graph plan."""
        return bool(
            self.is_hard_faulted
            or self.local_planner_state != "nominal"
            or self.eta_drift_steps > 0
            or self.handling_delay_steps > 0
            or float(self.speed_scale) < 0.85
            or float(self.receipt_age_s) >= 1.0
        )

    def has_non_default_state(self) -> bool:
        """Return True if any non-trivial feedback or external pose observation is populated."""
        return bool(
            self.pose_xy_yaw is not None
            or self.localization_source != "discrete"
            or self.active_leg is not None
            or float(self.leg_progress) > 0.0
            or self.eta_drift_steps != 0
            or abs(float(self.speed_scale) - 1.0) > 1e-6
            or self.local_planner_state != "nominal"
            or float(self.receipt_age_s) > 0.0
            or self.handling_delay_steps > 0
            or self.route_id is not None
            or self.fault_reason is not None
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize status into JSON-compatible wire dictionary."""
        return {
            "robot_id": self.robot_id,
            "pose_xy_yaw": [round(float(x), 4) for x in self.pose_xy_yaw] if self.pose_xy_yaw is not None else None,
            "localization_source": self.localization_source,
            "active_leg": list(self.active_leg) if self.active_leg is not None else None,
            "leg_progress": round(float(self.leg_progress), 4),
            "eta_drift_steps": int(self.eta_drift_steps),
            "speed_scale": round(float(self.speed_scale), 4),
            "local_planner_state": self.local_planner_state,
            "receipt_age_s": round(float(self.receipt_age_s), 4),
            "handling_delay_steps": int(self.handling_delay_steps),
            "route_id": self.route_id,
            "fault_reason": self.fault_reason,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AmrExecutionStatus":
        """Deserialize and validate status from wire dictionary."""
        pose_raw = raw.get("pose_xy_yaw")
        leg_raw = raw.get("active_leg")
        return cls(
            robot_id=str(raw["robot_id"]),
            pose_xy_yaw=(float(pose_raw[0]), float(pose_raw[1]), float(pose_raw[2])) if pose_raw is not None else None,
            localization_source=str(raw.get("localization_source", "discrete")),
            active_leg=(str(leg_raw[0]), str(leg_raw[1])) if leg_raw is not None else None,
            leg_progress=float(raw.get("leg_progress", 0.0)),
            eta_drift_steps=int(raw.get("eta_drift_steps", 0)),
            speed_scale=float(raw.get("speed_scale", 1.0)),
            local_planner_state=str(raw.get("local_planner_state", "nominal")),
            receipt_age_s=float(raw.get("receipt_age_s", 0.0)),
            handling_delay_steps=int(raw.get("handling_delay_steps", 0)),
            route_id=(str(raw["route_id"]) if raw.get("route_id") is not None else None),
            fault_reason=(str(raw["fault_reason"]) if raw.get("fault_reason") is not None else None),
        )


@dataclass(frozen=True)
class CorridorFeedback:
    """Validated L1/L2 corridor passability & traversal delay update."""

    src: str
    dst: str
    status: str = "open"
    extra_delay_steps: int = 0
    blocked_until_step: Optional[int] = None
    source: str = "l1_monitor"
    reason: Optional[str] = None
    q3_penalty_override: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.src or not self.dst or self.src == self.dst:
            raise ValueError(f"corridor requires distinct non-empty endpoints, got {(self.src, self.dst)!r}")
        if self.status not in CORRIDOR_STATES:
            raise ValueError(f"unsupported corridor status: {self.status!r}")
        if (
            isinstance(self.extra_delay_steps, bool)
            or not isinstance(self.extra_delay_steps, int)
            or self.extra_delay_steps < 0
        ):
            raise ValueError(f"extra_delay_steps must be non-negative int, got {self.extra_delay_steps!r}")
        if self.blocked_until_step is not None and (
            isinstance(self.blocked_until_step, bool) or not isinstance(self.blocked_until_step, int)
        ):
            raise ValueError(f"blocked_until_step must be int or None, got {self.blocked_until_step!r}")
        if self.q3_penalty_override is not None and (
            isinstance(self.q3_penalty_override, bool)
            or not isinstance(self.q3_penalty_override, int)
            or self.q3_penalty_override < 0
        ):
            raise ValueError(f"q3_penalty_override must be non-negative int or None, got {self.q3_penalty_override!r}")

    def has_non_default_state(self) -> bool:
        """Return True when this corridor feedback carries an active deviation."""
        return bool(self.status != "open" or self.extra_delay_steps > 0 or self.blocked_until_step is not None)

    def to_dict(self) -> dict[str, Any]:
        """Serialize corridor update into JSON-compatible wire dictionary."""
        out: dict[str, Any] = {
            "src": self.src,
            "dst": self.dst,
            "status": self.status,
            "extra_delay_steps": int(self.extra_delay_steps),
            "blocked_until_step": self.blocked_until_step,
            "source": self.source,
            "reason": self.reason,
        }
        if self.q3_penalty_override is not None:
            out["q3_penalty_override"] = int(self.q3_penalty_override)
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CorridorFeedback":
        """Deserialize and validate corridor feedback from wire dictionary."""
        return cls(
            src=str(raw["src"]),
            dst=str(raw["dst"]),
            status=str(raw.get("status", "open")),
            extra_delay_steps=int(raw.get("extra_delay_steps", 0)),
            blocked_until_step=(int(raw["blocked_until_step"]) if raw.get("blocked_until_step") is not None else None),
            source=str(raw.get("source", "l1_monitor")),
            reason=(str(raw["reason"]) if raw.get("reason") is not None else None),
            q3_penalty_override=(int(raw["q3_penalty_override"]) if raw.get("q3_penalty_override") is not None else None),
        )


@dataclass(frozen=True)
class ExecutionFeedbackPacket:
    """Snapshot-aligned container of execution feedback shared between L1/L2 and CVCR L3."""

    schema: str = EXEC_FEEDBACK_SCHEMA
    tick: int = 0
    robot_states: dict[str, AmrExecutionStatus] = field(default_factory=dict)
    corridor_updates: dict[tuple[str, str], CorridorFeedback] = field(default_factory=dict)
    source_events: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def is_empty(self) -> bool:
        """Return True when no active robot or corridor execution deviation is recorded."""
        has_rob = any(st.has_non_default_state() for st in self.robot_states.values())
        has_cor = any(cf.has_non_default_state() for cf in self.corridor_updates.values())
        return not (has_rob or has_cor or bool(self.source_events))

    def robot_ready_offset_steps(self, robot_id: str, dt: float = 1.0) -> int:
        """Return non-negative L3 start-time drift (steps) induced by L1/L2 execution feedback."""
        st = self.robot_states.get(robot_id)
        if st is None:
            return 0
        offset = max(0, int(st.eta_drift_steps)) + max(0, int(st.handling_delay_steps))
        if st.local_planner_state in ("faulted", "local_failed", "stopped"):
            age_steps = max(2, int(math.ceil(float(st.receipt_age_s) / max(0.05, float(dt)))))
            offset = max(offset, age_steps)
        elif st.local_planner_state in ("waiting_obstacle", "detouring"):
            offset = max(offset, 2)
        return int(offset)

    def robot_leg_step_buffer(self, robot_id: str, nominal_steps: int) -> int:
        """Return strictly causal extra transit steps predicted from observed `speed_scale` and `eta_drift_steps`."""
        st = self.robot_states.get(robot_id)
        if st is None or nominal_steps <= 0:
            return 0
        if st.local_planner_state in ("faulted", "local_failed", "stopped"):
            return 0
        spd = float(st.speed_scale)
        eta_d = int(st.eta_drift_steps)
        if spd >= 0.90 and eta_d <= 0 and st.local_planner_state == "nominal":
            return 0
        buf = 0
        if spd < 0.90:
            buf = max(buf, int(math.ceil(float(nominal_steps) * (1.0 / max(0.35, spd) - 1.0))))
        if eta_d >= 2 and st.local_planner_state != "nominal":
            buf = max(buf, 1)
        return min(6, max(0, buf))

    def corridor_q3_penalties(self) -> dict[tuple[str, str], int]:
        """Convert corridor delays and degraded L1 speed scales into bounded Q3 edge penalties."""
        q3: dict[tuple[str, str], int] = {}
        for (src, dst), cf in self.corridor_updates.items():
            if cf.q3_penalty_override is not None:
                pen = max(0, min(64, int(cf.q3_penalty_override)))
                if pen > 0:
                    q3[(src, dst)] = max(q3.get((src, dst), 0), pen)
            elif cf.status in ("degraded", "congested") or cf.extra_delay_steps > 0:
                pen = min(32, max(3, int(cf.extra_delay_steps) * 4 if cf.extra_delay_steps > 0 else 6))
                q3[(src, dst)] = max(q3.get((src, dst), 0), pen)
        for st in self.robot_states.values():
            if st.active_leg is not None and (float(st.speed_scale) < 0.75 or st.local_planner_state in ("waiting_obstacle", "detouring")):
                u, v = st.active_leg
                pen = 16 if st.local_planner_state == "waiting_obstacle" else 10
                q3[(u, v)] = max(q3.get((u, v), 0), pen)
        return q3

    def blocked_intervals(self, t_commit: int, default_horizon: int = 30) -> dict[tuple[str, str], int]:
        """Return `{directed_edge: end_step}` for corridors currently blocked by L1/L2 obstacles."""
        out: dict[tuple[str, str], int] = {}
        for (src, dst), cf in self.corridor_updates.items():
            if cf.status != "blocked":
                continue
            b_end = (
                int(cf.blocked_until_step)
                if cf.blocked_until_step is not None
                else int(t_commit) + max(1, int(default_horizon))
            )
            if b_end > int(t_commit):
                out[(src, dst)] = b_end
        return out

    def canonical_payload(self) -> dict[str, Any]:
        """Return deterministic dictionary used for snapshot digest & evaluator cache keys."""
        return {
            "schema": self.schema,
            "tick": int(self.tick),
            "robot_states": {
                k: self.robot_states[k].to_dict()
                for k in sorted(self.robot_states)
                if self.robot_states[k].has_non_default_state()
            },
            "corridor_updates": {
                f"{u}->{v}": self.corridor_updates[(u, v)].to_dict()
                for (u, v) in sorted(self.corridor_updates)
                if self.corridor_updates[(u, v)].has_non_default_state()
            },
            "source_events": list(self.source_events),
        }

    def signature(self) -> tuple[Any, ...]:
        """Return compact hashable signature for CVCREvaluator incremental cache invalidation."""
        rob_sig = tuple(
            (
                rid,
                st.local_planner_state,
                int(st.eta_drift_steps),
                int(st.handling_delay_steps),
                round(float(st.speed_scale), 2),
                st.fault_reason or "",
            )
            for rid, st in sorted(self.robot_states.items())
            if st.has_non_default_state()
        )
        cor_sig = tuple(
            (u, v, cf.status, int(cf.extra_delay_steps), cf.blocked_until_step, cf.q3_penalty_override)
            for (u, v), cf in sorted(self.corridor_updates.items())
            if cf.has_non_default_state()
        )
        return (rob_sig, cor_sig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize full packet for logging, IPC, and replay persistence."""
        return self.canonical_payload()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExecutionFeedbackPacket":
        """Reconstruct validated `ExecutionFeedbackPacket` from JSON dictionary."""
        r_states = {
            str(rid): AmrExecutionStatus.from_dict(st)
            for rid, st in (raw.get("robot_states") or {}).items()
        }
        c_updates: dict[tuple[str, str], CorridorFeedback] = {}
        for k, cf_raw in (raw.get("corridor_updates") or {}).items():
            cf = CorridorFeedback.from_dict(cf_raw)
            c_updates[(cf.src, cf.dst)] = cf
        events = tuple(dict(ev) for ev in (raw.get("source_events") or ()))
        return cls(
            schema=str(raw.get("schema", EXEC_FEEDBACK_SCHEMA)),
            tick=int(raw.get("tick", 0)),
            robot_states=r_states,
            corridor_updates=c_updates,
            source_events=events,
        )

    @classmethod
    def from_simulator_and_events(
        cls,
        simulator: Any,
        *,
        deviations: Sequence[DeviationEvent | Mapping[str, Any]] = (),
        robot_telemetry: Optional[Mapping[str, Mapping[str, Any] | AmrExecutionStatus]] = None,
        corridor_updates: Optional[Mapping[tuple[str, str], CorridorFeedback]] = None,
        now_wall: Optional[float] = None,
    ) -> Optional["ExecutionFeedbackPacket"]:
        """Build an `ExecutionFeedbackPacket` from simulator execution gates and L1/L2 events.

        Returns `None` when the simulator has no attached physical execution gates, no closed/degraded
        corridors, no deviation events, and no overrides — preserving nominal zero-overhead execution.
        """
        executions = getattr(simulator, "executions", None) or {}
        handlings = getattr(simulator, "handlings", None) or {}
        closed_corr = getattr(simulator, "_closed_corridors", None) or {}
        override_pkt = getattr(simulator, "_execution_feedback", None)
        if not executions and not handlings and not closed_corr and not deviations and not robot_telemetry and not corridor_updates and override_pkt is None:
            return None

        tick = int(getattr(simulator, "_step_cursor", 0))
        transit = getattr(simulator, "_transit", None) or {}
        detours = getattr(simulator, "_physical_detours", None) or {}
        robot_states: dict[str, AmrExecutionStatus] = {}
        if isinstance(override_pkt, ExecutionFeedbackPacket):
            robot_states.update(override_pkt.robot_states)

        for agv in getattr(simulator, "agvs", ()):
            rid = str(agv.agv_id)
            gate = executions.get(rid)
            hdl = handlings.get(rid)
            tr = transit.get(rid)
            active_leg = (str(tr[0]), str(tr[1])) if tr is not None and len(tr) >= 2 else None

            if gate is None and hdl is None and rid not in (robot_telemetry or {}):
                continue

            pose_tuple: Optional[tuple[float, float, float]] = None
            src_name = "discrete"
            route_id: Optional[str] = None
            fault_reason: Optional[str] = None
            receipt_age = 0.0
            lp_state = "nominal"
            if gate is not None:
                snap = gate.snapshot()
                src_raw = str(getattr(gate, "identity", {}).get("source", "gazebo"))
                src_name = src_raw if src_raw in LOCALIZATION_SOURCES else "gazebo"
                p = snap.get("pose")
                if isinstance(p, Mapping):
                    pose_tuple = (float(p["x"]), float(p["y"]), float(p["yaw"]))
                route_id = snap.get("route_id")
                fault_reason = snap.get("fault")
                last_r = snap.get("last_receipt")
                if last_r is not None and now_wall is not None:
                    receipt_age = max(0.0, float(now_wall) - float(last_r))
                if fault_reason:
                    lp_state = "faulted"
                elif snap.get("stopped"):
                    lp_state = "stopped"
                elif rid in detours:
                    lp_state = "detouring"
            if hdl is not None and getattr(hdl, "command", None) is not None and not getattr(hdl, "completed", True):
                if lp_state == "nominal":
                    lp_state = "handling_wait"

            ovr = (robot_telemetry or {}).get(rid)
            if isinstance(ovr, AmrExecutionStatus):
                robot_states[rid] = ovr
            else:
                ovd = dict(ovr) if isinstance(ovr, Mapping) else {}
                robot_states[rid] = AmrExecutionStatus(
                    robot_id=rid,
                    pose_xy_yaw=ovd.get("pose_xy_yaw", pose_tuple),
                    localization_source=str(ovd.get("localization_source", src_name)),
                    active_leg=ovd.get("active_leg", active_leg),
                    leg_progress=float(ovd.get("leg_progress", 0.0)),
                    eta_drift_steps=int(ovd.get("eta_drift_steps", 0)),
                    speed_scale=float(ovd.get("speed_scale", 1.0)),
                    local_planner_state=str(ovd.get("local_planner_state", lp_state)),
                    receipt_age_s=float(ovd.get("receipt_age_s", receipt_age)),
                    handling_delay_steps=int(ovd.get("handling_delay_steps", 0)),
                    route_id=ovd.get("route_id", route_id),
                    fault_reason=ovd.get("fault_reason", fault_reason),
                )

        if robot_telemetry:
            for rid, ovr in robot_telemetry.items():
                if rid not in robot_states:
                    if isinstance(ovr, AmrExecutionStatus):
                        robot_states[str(rid)] = ovr
                    else:
                        robot_states[str(rid)] = AmrExecutionStatus.from_dict({"robot_id": str(rid), **dict(ovr)})

        corridors: dict[tuple[str, str], CorridorFeedback] = {}
        if isinstance(override_pkt, ExecutionFeedbackPacket):
            corridors.update(override_pkt.corridor_updates)
        network = getattr(simulator, "network", None)

        if isinstance(closed_corr, Mapping):
            items_c = closed_corr.items()
        else:
            items_c = ((edge, None) for edge in closed_corr)
        for edge, until_s in items_c:
            if len(edge) >= 2:
                u, v = str(edge[0]), str(edge[1])
                for a, b in ((u, v), (v, u)):
                    if network is None or network.has_edge(a, b):
                        corridors[(a, b)] = CorridorFeedback(
                            src=a,
                            dst=b,
                            status="blocked",
                            blocked_until_step=(int(until_s) if isinstance(until_s, int) else None),
                            source="simulator_corridor_lock",
                        )

        source_events: list[dict[str, Any]] = []
        for raw_ev in (deviations or ()):
            dev = raw_ev if isinstance(raw_ev, DeviationEvent) else DeviationEvent.from_dict(dict(raw_ev))
            u, v = dev.blocked_edge
            for a, b in ((u, v), (v, u)):
                if network is None or network.has_edge(a, b):
                    corridors[(a, b)] = CorridorFeedback(
                        src=a,
                        dst=b,
                        status="blocked",
                        source="deviation_event",
                        reason=dev.event,
                    )
            source_events.append({
                "event": dev.event,
                "robot_id": dev.robot_id,
                "blocked_edge": [u, v],
                "timestamp": float(dev.timestamp),
            })

        if corridor_updates:
            for edge_k, cf in corridor_updates.items():
                corridors[(str(edge_k[0]), str(edge_k[1]))] = cf

        pkt = cls(
            schema=EXEC_FEEDBACK_SCHEMA,
            tick=tick,
            robot_states=robot_states,
            corridor_updates=corridors,
            source_events=tuple(source_events),
        )
        return None if pkt.is_empty() else pkt
