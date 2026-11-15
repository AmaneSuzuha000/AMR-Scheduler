"""Read-only preflight for a physical fleet membership transaction.

This function NEVER returns a movement permit and deliberately exposes no
physical commit endpoint. All physical route leases and checkpoint reconciliation
remain owned by the existing FleetRuntime; this checks candidate readiness only.
"""
from __future__ import annotations
import math
from amr_sched.entities import AgvState
from amr_sched.network import NodeKind

def inspect_physical_membership(owner, robot_id, *, expected_routes, now):
    sim=getattr(owner, "sim", None)
    reasons=[]
    def require(ok,why):
        if not ok:reasons.append(why)
    require(sim is not None and bool(getattr(sim,'executions',None)), "PHYSICAL_OWNER_REQUIRED")
    if sim is None or not getattr(sim,"executions",None):
        return dict(ready=False,reasons=reasons,physical_authorization=False)
    require(bool(getattr(owner,"ready",False)) and not getattr(owner,"failed",True)
            and not getattr(owner,"closing",True),"DURABLE_OWNER_NOT_READY")
    require(bool(getattr(sim,"_operator_paused",False)),"FLEET_NOT_PAUSED")
    require(robot_id in sim.executions and robot_id in getattr(owner,"nodes",{}),"ROBOT_NOT_IN_OWNER")
    current={r:g.recovery_token() for r,g in sim.executions.items()}
    require(type(expected_routes) is dict and expected_routes==current,"STALE_GLOBAL_ROUTE_TOKENS")
    require(type(now) in (int,float) and math.isfinite(now),"MONOTONIC_NOW_INVALID")
    robot=next((a for a in sim.agvs if a.agv_id==robot_id),None)
    require(robot is not None,"ROBOT_MISSING")
    if robot:
        require(robot.cur_node in sim.network.nodes and
                sim.network.node(robot.cur_node).kind is NodeKind.BUFFER,
                "NO_VERIFIED_BUFFER_PARKING")
        require(robot.cur_task is None and robot.state is AgvState.IDLE and not robot.route
                and robot_id not in sim._jobs and robot_id not in sim._transit,
                "COMMITTED_ROBOT_MOTION_OR_CARGO")
        require(robot_id not in sim._node_claim,"UNSETTLED_NODE_CLAIM")
        require(not any(t.assigned_agv==robot_id and t.status.value != 'completed' for t in sim.tasks),
                "CARGO_OR_ASSIGNED_TASK_COMMITMENT")
    for rid,gate in sim.executions.items():
        node=getattr(owner,"nodes",{}).get(rid)
        if node is None:
            require(False,"MISSING_ADAPTER_"+rid);continue
        require(gate.bound and gate.pose is not None and gate.last_receipt is not None,
                "UNBOUND_OR_MISSING_POSE_"+rid)
        require(gate.fault in {'operator fleet pause','runtime restarted; reconciliation required'},
                "PAUSE_FAULT_NOT_LATCHED_"+rid)
        # STOPPED feedback deliberately retains the latched fault state: the
        # scheduler MUST NOT mistake a stop acknowledgement for un-latching.
        require(gate.command is None and gate.stopped and gate.state in ('idle','fault'),
                "MOTION_OR_STOP_UNCONFIRMED_"+rid)
        require(not gate._outbox, "PENDING_CANCEL_OR_MOVE_COMMAND_"+rid)
        require(not getattr(node,"persistence_failed",True) and
                not getattr(node,"native_pending",True) and
                getattr(node,"goal_handle",object()) is None and
                getattr(node,"guard_pending",object()) is None,
                "UNSETTLED_NATIVE_GOAL_"+rid)
        require(bool(node.guard_fresh()) and bool(node.feedback_available())
                and node.guard_status.state=='latched'
                and node.guard_status.stopped,"STALE_OR_MOVING_GUARD_"+rid)
        if gate.last_receipt is not None and type(now) in (int,float) and math.isfinite(now):
            require(gate.last_receipt<=now<=gate.last_receipt+gate.timeout,
                    "EXECUTION_RECEIPT_EXPIRED_"+rid)
    # An owner cannot infer that a load has been physically released from pose alone.
    handling=getattr(sim,"handlings",{})
    for rid,h in handling.items():
        require(h.command is None or bool(h.completed),
                "UNSETTLED_HANDLING_"+rid)
    return dict(ready=not reasons,reasons=reasons,robot_id=robot_id,
                route_versions=current,
                action="PREPARE_ONLY",physical_authorization=False,
                permit_issued=False,commit_enabled=False)
