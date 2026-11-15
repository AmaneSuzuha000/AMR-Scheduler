"""Operator actions retain the physical owner, versions, load and permanent pins."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import select
import socket
import time
from types import SimpleNamespace

import pytest

from amr_sched.checkpoint import capture, definition, restore
from amr_sched.fleet_control import FleetControl, FleetControlServer, MAX_LINE
from amr_sched.stream import FrameBuilder
from test_external_execution import bound_fleet, fleet, fleet_report


def owner_for(sim, *, write_failure=False):
    path=Path(__file__).resolve().parents[1]/'tools/live_bridge/fleet_scheduler.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FleetRuntime')
    scope=dict(deepcopy=deepcopy,time=SimpleNamespace(monotonic=lambda:.4),FleetControlServer=FleetControlServer)
    exec(compile(ast.Module(body=[klass],type_ignores=[]),str(path),'exec'),scope)
    owner=scope['FleetRuntime'].__new__(scope['FleetRuntime'])
    owner.sim=sim
    owner.ready=True
    owner.failed=owner.closing=owner.done=False
    owner.control_server=None
    owner.builder=FrameBuilder(sim)
    owner.nodes={r:SimpleNamespace(persistence_failed=False,native_pending=False,goal_handle=None,
        guard_pending=None,guard_fresh=lambda:True,feedback_available=lambda:True,
        guard_status=SimpleNamespace(state='latched',stopped=True),stop_guard=lambda:None)
        for r in sim.executions}
    owner.writes=[]
    def persist():
        owner.writes.append(capture(sim,definition(sim),{}))
        if write_failure:
            owner.failed=True
            for gate in sim.executions.values():gate.fail('injected write failure')
            for node in owner.nodes.values():node.persistence_failed=True
            return False
        return True
    owner.persist=persist
    owner.records=[]
    owner.record=lambda kind,**v:owner.records.append((kind,v))
    return owner


def request(control,op='pause',payload=None,rid='req1'):
    s=control.state()
    return dict(type='control_request',session_id=s['session_id'],scene_id=s['scene_id'],
                run_id=s['run_id'],request_id=rid,operation=op,
                payload={'routes':s['routes']} if payload is None else payload)


@pytest.mark.parametrize('clean',[False,True])
def test_begin_close_latches_before_input_drain_without_business_progress(clean):
    owner=owner_for(bound_fleet());events=[]
    before=deepcopy((owner.sim.tasks,owner.sim._res.pins(),owner.sim._step_cursor))
    owner.control_server=SimpleNamespace(close=lambda:events.append('control_closed'))
    for robot,node in owner.nodes.items():
        node.query_feedback=lambda r=robot:events.append('feedback_'+r)
        node.clean_idle_shutdown_ready=lambda:clean
        node.suppress_feedback=True
        node.stop_guard=lambda r=robot:events.append('stop_'+r)
    owner.begin_close()
    # The harness can now drain equipment with the ROS/services still alive.
    events.append('equipment_drain')
    owner.begin_close()
    assert owner.closing and owner.control_server is None
    assert events.count('control_closed')==1 and owner.writes==[]
    assert (owner.sim.tasks,owner.sim._res.pins(),owner.sim._step_cursor)==before
    if clean:
        assert not owner._close_needs_stop
        assert not any(e.startswith('stop_') for e in events)
    else:
        assert owner._close_needs_stop
        assert all(g.fault=='fleet runtime shutdown' for g in owner.sim.executions.values())
        assert all(not n.suppress_feedback for n in owner.nodes.values())
        assert all(events.index('stop_'+r)<events.index('equipment_drain') for r in owner.nodes)


def test_begin_close_input_failure_cannot_prevent_stop_requests_to_other_robots():
    owner=owner_for(bound_fleet());stopped=[]
    for robot,node in owner.nodes.items():
        node.clean_idle_shutdown_ready=lambda:False
        node.stop_guard=lambda r=robot:stopped.append(r)
        node.query_feedback=lambda:(_ for _ in ()).throw(RuntimeError('input fault'))
    with pytest.raises(RuntimeError,match='input fault'):
        owner.begin_close()
    assert set(stopped)==set(owner.nodes) and owner._close_requested and owner.closing
    assert all(g.fault=='fleet runtime shutdown' for g in owner.sim.executions.values())


def test_shutdown_robot_pump_observes_without_recovery_dispatch_or_renewal():
    path=Path(__file__).resolve().parents[1]/'tools/live_bridge/fleet_scheduler.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FleetRobotNode')
    scope=dict(SensorSchedulerNode=object)
    exec(compile(ast.Module(body=[klass],type_ignores=[]),str(path),'exec'),scope)
    node=scope['FleetRobotNode'].__new__(scope['FleetRobotNode'])
    events=[]
    node.closed=False;node.fleet_owner=SimpleNamespace(closing=True)
    node.sensor_ingress=SimpleNamespace(pull=lambda:events.append('sensor'))
    node.stop_guard=lambda:events.append('stop')
    node.query_feedback=lambda:events.append('feedback')
    # No scheduling/execution state is supplied: any access would fail.
    node.tick()
    assert events==['sensor','stop','feedback']


def stopped_owner():
    owner=owner_for(bound_fleet())
    c=FleetControl(owner)
    assert c.handle(request(c))['accepted']
    for r,g in owner.sim.executions.items():
        g.drain_commands()
        xy=(g.pose['x'],g.pose['y'])
        assert fleet_report(owner.sim,r,'STOPPED',xy,now=.3)['accepted']
    return owner,c


def test_pause_freezes_handling_time_and_preserves_task_load_and_pins():
    owner=owner_for(bound_fleet())
    sim=owner.sim
    before=deepcopy((sim._res.pins(),sim._transit,sim.tasks,sim._step_cursor))
    c=FleetControl(owner)
    answer=c.handle(request(c))
    assert answer['accepted'] and sim._operator_paused and len(owner.writes)==1
    for _ in range(30):assert sim.step_once()
    assert (sim._res.pins(),sim._transit,sim.tasks,sim._step_cursor)==before
    assert all(g.fault=='operator fleet pause' for g in sim.executions.values())
    assert all(x['kind']=='CANCEL' for g in sim.executions.values() for x in g.drain_commands())


def test_resume_preflights_whole_fleet_then_persists_new_versions():
    owner,c=stopped_owner()
    sim=owner.sim
    before=deepcopy((sim._res.pins(),sim.tasks))
    old=c.state()['routes']
    answer=c.handle(request(c,'resume',rid='resume'))
    assert answer['accepted'] and not sim._operator_paused
    assert (sim._res.pins(),sim.tasks)==before
    assert all(c.state()['routes'][r]!=v for r,v in old.items())
    assert len(owner.writes)==2


@pytest.mark.parametrize('case',['native_pending','goal_handle','guard_pending','stale_guard',
    'moving_guard','not_stopped','stale_sensor','prior_fault','bad_pose','pending_cancel'])
def test_one_unready_robot_refuses_entire_fleet_without_new_move(case):
    owner,c=stopped_owner()
    robot=next(reversed(owner.nodes))
    node=owner.nodes[robot]
    gate=owner.sim.executions[robot]
    if case in {'native_pending','goal_handle','guard_pending'}:setattr(node,case,True)
    elif case=='stale_guard':node.guard_fresh=lambda:False
    elif case=='moving_guard':node.guard_status.state='active'
    elif case=='not_stopped':node.guard_status.stopped=False
    elif case=='stale_sensor':node.feedback_available=lambda:False
    elif case=='prior_fault':gate.fault='unrelated collision'
    elif case=='bad_pose':gate.pose['x']=100
    elif case=='pending_cancel':gate._outbox.append({'kind':'CANCEL'})
    before=deepcopy({r:vars(g) for r,g in owner.sim.executions.items()})
    answer=c.handle(request(c,'resume',rid='resume'))
    assert not answer['accepted'] and owner.sim._operator_paused
    assert before=={r:vars(g) for r,g in owner.sim.executions.items()}
    assert len(owner.writes)==1


@pytest.mark.parametrize('field',['scene_id','run_id','session_id','route'])
def test_cross_run_and_stale_actions_never_mutate_physical_owner(field):
    owner=owner_for(bound_fleet())
    c=FleetControl(owner)
    req=request(c)
    if field=='route':req['payload']['routes']['R1']='old'
    else:req[field]='old'
    assert not c.handle(req)['accepted']
    assert not owner.sim._operator_paused and not owner.writes


def test_receipt_retry_is_immutable_even_after_resume_and_original_reply_is_mutated():
    owner,c=stopped_owner()
    first=request(c,'submit_task',dict(task_id='extra',pickup='W',dropoff='E'),rid='task')
    result=c.handle(first)
    assert result['accepted']
    result['accepted']=False
    assert c.handle(first)['accepted'] and len(owner.writes)==2
    changed=deepcopy(first);changed['payload']['dropoff']='S'
    assert not c.handle(changed)['accepted'] and len(owner.writes)==2
    assert len([t for t in owner.sim.tasks if t.task_id=='extra'])==1


def test_rejected_resume_does_not_turn_into_accepted_retry_when_stop_becomes_fresh():
    owner,c=stopped_owner()
    node=next(iter(owner.nodes.values()))
    node.native_pending=True
    req=request(c,'resume',rid='retry')
    answer=c.handle(req)
    assert not answer['accepted']
    node.native_pending=False
    assert c.handle(req)==answer and owner.sim._operator_paused
    assert c.handle(request(c,'resume',rid='new'))['accepted']


@pytest.mark.parametrize('operation',['pause','resume','submit_task'])
def test_disk_failure_never_returns_success_or_allows_movement(operation):
    owner,c=stopped_owner() if operation=='resume' else (owner_for(bound_fleet()),None)
    c=c or FleetControl(owner)
    def failed():
        owner.failed=True
        for g in owner.sim.executions.values():g.fail('injected disk failure')
        return False
    owner.persist=failed
    payload=dict(task_id='extra',pickup='W',dropoff='E') if operation=='submit_task' else None
    assert not c.handle(request(c,operation,payload,rid='fail'))['accepted']
    assert owner.failed and all(g.fault for g in owner.sim.executions.values())
    for _ in range(5):owner.sim.step_once()
    assert all(g.fault for g in owner.sim.executions.values())


def test_operator_pause_survives_checkpoint_but_receipts_do_not_cross_restart():
    owner,c=stopped_owner()
    sim=owner.sim
    checkpoint=capture(sim,definition(sim),{})
    restored=fleet()
    restore(restored,checkpoint,definition(restored))
    assert restored._operator_paused and restored._res.pins()==sim._res.pins()
    assert all(g.pose is None and not g.stopped and g.fault for g in restored.executions.values())
    new=FleetControl(owner_for(restored))
    assert not new.handle(request(c))['accepted']


@pytest.mark.parametrize('payload',[
    dict(task_id='bad',pickup='W',dropoff='E',priority=True),
    dict(task_id='bad',pickup='W',dropoff='E',due_step=1.5),
    dict(task_id='bad',pickup='W',dropoff='E',route=['W','E']),
    dict(task_id='bad',pickup='missing',dropoff='E'),
])
def test_task_input_does_not_accept_foreign_planning_or_invalid_fields(payload):
    owner,c=stopped_owner()
    count=len(owner.sim.tasks)
    assert not c.handle(request(c,'submit_task',payload,rid='bad'))['accepted']
    assert len(owner.sim.tasks)==count and len(owner.writes)==1


def test_receipt_capacity_refuses_new_ids_without_eviction_or_reexecution():
    owner=owner_for(bound_fleet())
    c=FleetControl(owner,max_requests=1)
    req=request(c)
    first=c.handle(req)
    assert first['accepted']
    assert not c.handle(request(c,rid='two'))['accepted']
    assert c.handle(req)==first and len(owner.writes)==1


def test_socket_fragments_reconnect_and_request_retry_share_one_owner():
    owner=owner_for(bound_fleet())
    server=FleetControlServer(owner)
    frame=owner.builder.frame()
    try:
        req=request(server.control)
        wire=(json.dumps(req)+'\n').encode()
        for attempt in range(2):
            with socket.create_connection(('127.0.0.1',server.port),timeout=1) as client:
                client.settimeout(1)
                server.poll(frame)
                assert b'"hello"' in client.recv(MAX_LINE)
                client.sendall(wire[:30]);server.poll(frame)
                assert len(owner.writes)==attempt
                client.sendall(wire[30:]);server.poll(frame)
                receipt=client.recv(MAX_LINE)
                assert b'"control_result"' in receipt and b'"accepted":true' in receipt
            server.poll(frame)
        assert len(owner.writes)==1 and owner.sim._operator_paused
    finally:server.close()


def test_oversize_unterminated_socket_input_is_dropped_without_mutation():
    owner=owner_for(bound_fleet())
    server=FleetControlServer(owner)
    try:
        with socket.create_connection(('127.0.0.1',server.port),timeout=1) as client:
            server.poll(owner.builder.frame())
            client.sendall(b'x'*(MAX_LINE+1))
            # TCP may deliver the final segment after a tight polling loop has
            # finished. Wait for readable bytes within a bounded wall deadline;
            # the server must still enforce the same size and mutation limits.
            deadline=time.monotonic()+1.
            while server.clients and time.monotonic()<deadline:
                select.select(list(server.clients),[],[],max(0.,min(.05,deadline-time.monotonic())))
                server.poll(owner.builder.frame())
                assert all(len(s['buffer'])<=MAX_LINE for s in server.clients.values())
                assert not owner.writes
            assert not server.clients and not owner.writes
    finally:server.close()
