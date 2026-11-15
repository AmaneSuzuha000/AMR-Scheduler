"""SYSTEM12 physical-readiness is observational; no motor commands or grant."""
from copy import deepcopy
import pytest
from amr_sched.network import Node,NodeKind
from amr_sched.native_physical_admission_audit import inspect_physical_membership
from test_fleet_control import owner_for
from test_external_execution import fleet, fleet_report

def paused_fixture():
    sim=fleet()
    for robot in sim.executions:
        assert fleet_report(sim,robot,'POSE',now=0.)['accepted']
        gate=sim.executions[robot]
        gate.fail('operator fleet pause')
        assert fleet_report(sim,robot,'STOPPED',now=.1)['accepted']
    sim._operator_paused=True
    # Synthetic *declared* parking nodes for the isolated positive readiness test.
    # This changes only test construction, not any real factory topology.
    for robot in ('R1','R2'):
        agv=next(a for a in sim.agvs if a.agv_id==robot)
        n=sim.network.node(agv.cur_node)
        sim.network.nodes[agv.cur_node]=Node(n.node_id,n.x,n.y,NodeKind.BUFFER,n.capacity)
    owner=owner_for(sim)
    return owner

def inspect(owner,robot='R1',routes=None,now=.2):
    return inspect_physical_membership(owner,robot,
      expected_routes=routes if routes is not None else
         {r:g.recovery_token() for r,g in owner.sim.executions.items()},now=now)

def test_full_preflight_never_grants_physical_motion():
    owner=paused_fixture()
    before=deepcopy([(a.cur_node,a.cur_task) for a in owner.sim.agvs])
    old=deepcopy(owner.sim._res.pins())
    v=inspect(owner)
    assert v['ready'] and v['action']=='PREPARE_ONLY'
    assert not v['permit_issued'] and not v['commit_enabled'] and not v['physical_authorization']
    assert owner.sim._res.pins()==old
    assert before==[(a.cur_node,a.cur_task) for a in owner.sim.agvs]

@pytest.mark.parametrize('case,expected',[
    ('unpaused','FLEET_NOT_PAUSED'),
    ('bad_token','STALE_GLOBAL_ROUTE_TOKENS'),
    ('stale_pose','EXECUTION_RECEIPT_EXPIRED_R1'),
    ('native_pending','UNSETTLED_NATIVE_GOAL_R1'),
    ('moving_guard','STALE_OR_MOVING_GUARD_R1'),
    ('unverified_parking','NO_VERIFIED_BUFFER_PARKING'),
    ('committed_task','COMMITTED_ROBOT_MOTION_OR_CARGO'),
    ('unsettled_handling','UNSETTLED_HANDLING_R1'),
])
def test_physical_candidate_fails_closed(case,expected):
    owner=paused_fixture()
    routes=None;now=.2
    if case=='unpaused':owner.sim._operator_paused=False
    if case=='bad_token':routes={'R1':'stale'}
    if case=='stale_pose':now=10.
    if case=='native_pending':owner.nodes['R1'].native_pending=True
    if case=='moving_guard':owner.nodes['R1'].guard_status.stopped=False
    if case=='unverified_parking':
        n=owner.sim.network.node('W')
        owner.sim.network.nodes['W']=Node(n.node_id,n.x,n.y,NodeKind.AISLE,n.capacity)
    if case=='committed_task':owner.sim.agvs[0].cur_task='T1'
    if case=='unsettled_handling':
        from types import SimpleNamespace
        owner.sim.handlings['R1']=SimpleNamespace(command={'operation':'LOAD'},completed=False)
    v=inspect(owner,routes=routes,now=now)
    assert not v['ready'] and expected in v['reasons']
    assert v['physical_authorization'] is False
