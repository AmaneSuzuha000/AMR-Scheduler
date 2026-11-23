"""Restart retains physical task/load/pins and requires new observations and authority."""
import copy
from concurrent.futures import Future
import ast
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from amr_sched.checkpoint import CheckpointStore, capture, definition, restore
from amr_sched.entities import AgvState, TaskStatus
from test_external_execution import build, moving, report, stopped_midway, fleet, bound_fleet, fleet_report, detour_stopped_at_node


@pytest.mark.parametrize('config,valid',[
    ({},True),({'front_tracking':True,'tracking_offset_m':.463951/2,'latest_tf':False},True),
    ({'front_tracking':True,'tracking_offset_m':0.},False),
    ({'front_tracking':False,'tracking_offset_m':.463951/2},False),
    ({'front_tracking':'true','tracking_offset_m':.463951/2},False),
    ({'tracking_offset_m':float('nan')},False),({'latest_tf':1},False),
])
def test_physical_worker_restarts_with_the_recorded_body_reference(config,valid):
    path=Path(__file__).resolve().parents[1]/'out/archive/fleet_execution/fleet_worker.py'
    if not path.exists():
        pytest.skip("Optional archived ROS 2 script not present in checkout")
    tree=ast.parse(path.read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='runtime_options')
    scope={'math':math}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
    if not valid:
        with pytest.raises(ValueError):scope['runtime_options'](config)
    else:
        result=scope['runtime_options'](config)
        assert result==dict(tracking_offset_m=config.get('tracking_offset_m',0.),latest_tf=config.get('latest_tf',False))


@pytest.mark.parametrize('xy,locked',[((.5,.34),False),((.5,.36),True),((-.36,0.),True)])
def test_restarted_worker_keeps_the_owned_closed_segment_boundary(xy,locked):
    path=Path(__file__).resolve().parents[1]/'out/archive/fleet_execution/fleet_worker.py'
    if not path.exists():
        pytest.skip("Optional archived ROS 2 script not present in checkout")
    tree=ast.parse(path.read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='enforce_hop_bounds')
    scope={'math':math,'deepcopy':copy.deepcopy}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
    actions=[]
    gate=SimpleNamespace(command={'kind':'MOVE','nodes':['A','B']},pose=dict(x=xy[0],y=xy[1]),fault=None)
    def fail(reason):gate.fault=reason
    gate.fail=fail
    runtime=SimpleNamespace(nodes={'R':SimpleNamespace(stop_guard=lambda:actions.append('stop'))},
        sim=SimpleNamespace(execution_for=lambda robot:gate),record=lambda *a,**k:actions.append('record'))
    scope['enforce_hop_bounds'](runtime,{'A':(0.,0.),'B':(1.,0.)},.35)
    assert bool(gate.fault)==locked
    assert actions==(['stop','record'] if locked else [])


@pytest.mark.parametrize('case',['accepted','native_pending','goal_handle','guard_pending','stale_guard',
                                 'moving_guard','not_stopped','write_failed'])
def test_native_node_detour_requires_reconciliation_and_durable_commit(tmp_path,case):
    path=Path(__file__).resolve().parents[1]/'tools/live_bridge/nav2_scheduler.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SchedulerNode')
    klass.body=[n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='replan_execution']
    scope=dict(time=SimpleNamespace(monotonic=lambda:.7),RouteFollower=object)
    exec(compile(ast.Module(body=[klass],type_ignores=[]),str(path),'exec'),scope)
    node=scope['SchedulerNode'].__new__(scope['SchedulerNode'])
    sim=detour_stopped_at_node()
    node.sim=sim
    node.persistence_failed=node.native_pending=False
    node.goal_handle=node.guard_pending=None
    node.guard_status=SimpleNamespace(state='latched',stopped=True)
    node.guard_fresh=lambda:case!='stale_guard'
    node.active=node.waiting_move={'old':'terminated action'}
    node.result_status=4
    node.cancel_requested=True
    node.stop_since=.6
    if case=='native_pending': node.native_pending=True
    elif case=='goal_handle': node.goal_handle=object()
    elif case=='guard_pending': node.guard_pending=object()
    elif case=='moving_guard': node.guard_status.state='active'
    elif case=='not_stopped': node.guard_status.stopped=False
    writes=[]
    def persist():
        writes.append(capture(sim,definition(sim),{}))
        if case=='write_failed':
            node.persistence_failed=True
            sim.execution.fail('injected durable write failure')
            return False
        return True
    node.persist=persist
    records=[]
    node.record=lambda kind,**v:records.append((kind,v))
    pins=copy.deepcopy(sim._res.pins())
    answer=node.replan_execution(sim.execution.recovery_token())
    assert answer['accepted'] is (case=='accepted')
    assert sim._res.pins()==pins and sim.tasks[0].status is TaskStatus.PICKED
    assert records[-1][0]=='physical_replan'
    if case in {'accepted','write_failed'}:
        assert len(writes)==1 and node.active is None and node.waiting_move is None
    else:
        assert writes==[] and sim.execution.fault
    if case=='write_failed':
        for _ in range(20): sim.step_once()
        assert sim.execution.fault and all(c['kind']=='CANCEL' for c in sim.execution.drain_commands())
        assert sim._res.pins()==pins


def test_restart_analysis_reports_incomplete_recovery_without_a_success_result(tmp_path):
    import importlib.util
    root=Path(__file__).resolve().parents[1]
    path=root/'out/archive/fleet_execution/analyze_fleet_navigation.py'
    if not path.exists():
        pytest.skip("Optional archived ROS 2 script not present in checkout")
    spec=importlib.util.spec_from_file_location('fleet_acceptance',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path/'run_config.json').write_text(json.dumps({'fault_case':'scheduler_kill'}))
    result=module.evaluate(tmp_path,restart=True)
    assert not result['passed']
    assert {'scheduler_result.json','restart_result.json','at_kill.checkpoint.json'}<=set(result['missing_artifacts'])
    assert not result['checks']['actual_process_replaced']
    assert not result['checks']['saved_checkpoint_integrity']
    assert not result['checks']['fresh_stop_then_explicit_restore']
    assert not result['checks']['time_limit']
    # Explicit mode flags do not turn an unrelated run into a declared trial.
    (tmp_path/'run_config.json').write_text(json.dumps({'fault_case':None}))
    with pytest.raises(ValueError,match='declared scheduler_kill'):
        module.evaluate(tmp_path,restart=True)


def test_partial_fleet_construction_cannot_overwrite_saved_native_identities():
    path=Path(__file__).resolve().parents[1]/'tools/live_bridge/fleet_scheduler.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FleetRuntime')
    scope={}
    exec(compile(ast.Module(body=[klass],type_ignores=[]),str(path),'exec'),scope)
    owner=scope['FleetRuntime'].__new__(scope['FleetRuntime'])
    writes=[]
    owner.failed=False
    owner.constructing=True
    owner.store=SimpleNamespace(save=writes.append)
    assert owner.persist() is False
    assert writes==[]


def test_explicit_fleet_pump_runs_without_timer_delivery_and_never_catches_up(monkeypatch):
    import time
    path=Path(__file__).resolve().parents[1]/'tools/live_bridge/fleet_scheduler.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FleetRuntime')
    scope={'time':time}
    exec(compile(ast.Module(body=[klass],type_ignores=[]),str(path),'exec'),scope)
    owner=scope['FleetRuntime'].__new__(scope['FleetRuntime'])
    events=[];wall=[100.]
    monkeypatch.setattr(time,'monotonic',lambda:wall[0])
    owner.timer=SimpleNamespace(cancel=lambda:events.append('cancel fleet timer'))
    owner.record=lambda *a,**k:None
    owner.tick=lambda:events.append('owner')
    owner.nodes={name:SimpleNamespace(
        timer=SimpleNamespace(cancel=lambda name=name:events.append('cancel '+name)),
        tick=lambda name=name:events.append(name)) for name in ('a','b')}
    owner.start_owner_pump()
    assert events==['cancel fleet timer','cancel a','cancel b']
    events.clear()
    assert owner.pump_owner() and events==['owner','a','b']
    wall[0]=100.01
    assert not owner.pump_owner() and events==['owner','a','b']
    # A stalled callback may delay a turn, but cannot cause a burst of stale
    # scheduling/lease turns. Original per-node safety checks remain in tick.
    wall[0]=102.
    assert owner.pump_owner() and events==['owner','a','b']*2
    assert not owner.pump_owner()
    with pytest.raises(RuntimeError,match='already started'):owner.start_owner_pump()


def test_explicit_fleet_pump_rejects_receiver_thread_before_any_mutation():
    import threading,time
    path=Path(__file__).resolve().parents[1]/'tools/live_bridge/fleet_scheduler.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FleetRuntime')
    scope={'time':time}
    exec(compile(ast.Module(body=[klass],type_ignores=[]),str(path),'exec'),scope)
    owner=scope['FleetRuntime'].__new__(scope['FleetRuntime'])
    events=[];errors=[]
    owner.timer=SimpleNamespace(cancel=lambda:None)
    owner.record=lambda *a,**k:None
    owner.tick=lambda:events.append('illegal scheduling')
    owner.nodes={}
    owner.start_owner_pump()
    def receive():
        try:owner.pump_owner()
        except RuntimeError as exc:errors.append(str(exc))
    worker=threading.Thread(target=receive);worker.start();worker.join(timeout=1.)
    assert not worker.is_alive() and not events
    assert errors==['only the decision thread may pump the fleet']


@pytest.mark.parametrize('write_failure',[False,True])
def test_fleet_command_batch_is_durable_and_disk_failure_keeps_stop_and_pins(tmp_path,write_failure):
    """Idle adapter ticks skip duplicate writes; a MOVE still requires fsync."""
    from amr_sched.checkpoint import _decode
    path=Path(__file__).resolve().parents[1]/'tools/live_bridge/fleet_scheduler.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in
             ('RobotExecutionView','FleetRobotNode','FleetRuntime')]

    class _AdapterBase:
        """Stand-in for the real sensor adapter; its terminal take-over is inert."""

        def dock_step(self,now):
            return None

    scope=dict(SensorSchedulerNode=_AdapterBase,capture=capture,time=time)
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(path),'exec'),scope)
    sim=fleet()
    for robot in sim.executions:
        assert fleet_report(sim,robot)['accepted']
    owner=scope['FleetRuntime'].__new__(scope['FleetRuntime'])
    owner.sim=sim
    owner.failed=owner.constructing=owner.closing=False
    owner.fingerprint=definition(sim)
    store=CheckpointStore(tmp_path/'fleet.json')
    writes=[]
    def save(payload):
        writes.append(payload)
        if write_failure:
            raise OSError('injected disk failure')
        store.save(payload)
    owner.store=SimpleNamespace(save=save)
    owner.nodes={}
    for robot in sim.executions:
        node=scope['FleetRobotNode'].__new__(scope['FleetRobotNode'])
        node.fleet_owner=owner
        node.sim=scope['RobotExecutionView'](sim,robot)
        node.closed=node.guard_authorized=node.persistence_failed=False
        node.sensor_ingress=None  # This disk-durability fixture has no ROS receiver.
        node.dock_spec=None  # No declared terminal take-over in this fixture.
        node.active=node.requested_recovery=None
        node.guard_id=node.permit=node.native_goal_id=None
        node.native_state='none'
        node.clock_s=0.
        node.seq=-1
        node.stop_guard=node.reconcile_native=node.query_feedback=lambda:None
        node.authorize_waiting_move=node.publish_lease=lambda:None
        owner.nodes[robot]=node
    sent=[]
    def dispatch(command):
        if command['kind']=='MOVE':
            saved=_decode(store.load())
            assert saved['gate'][command['robot_id']]['command']==command
            assert saved['reservations']==vars(sim._res)
        sent.append(command)
    for node in owner.nodes.values():
        node.dispatch=dispatch
        node.tick()
    assert not writes and not sent
    for _ in range(10):
        sim.step_once()
        if sim._transit:
            break
    pins=copy.deepcopy(sim._res.pins())
    pending=[g.command for g in sim.executions.values() if g.command]
    assert pending
    for node in owner.nodes.values():
        node.tick()
    assert writes
    if write_failure:
        assert owner.failed and all(n.persistence_failed for n in owner.nodes.values())
        for node in owner.nodes.values():
            node.tick()
        assert sent and all(c['kind']=='CANCEL' for c in sent)
        assert sim._res.pins()==pins
        assert all(g.fault for g in sim.executions.values())
    else:
        assert sent==pending
        count=len(writes)
        for node in owner.nodes.values():
            node.tick()
        assert len(writes)==count
        assert owner.persist() and len(writes)==count+1
        saved=_decode(store.load())
        assert saved['reservations']==vars(sim._res)
    store.close()


def test_lost_native_cancel_response_retries_same_uuid_without_claiming_termination():
    path=Path(__file__).resolve().parents[1]/'tools/live_bridge/nav2_scheduler.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SchedulerNode')
    methods=[n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name in
             ('reconcile_native','on_reconcile_cancel')]
    now=[10.]
    scope=dict(time=SimpleNamespace(monotonic=lambda:now[0]),
        CancelGoal=SimpleNamespace(Request=lambda:SimpleNamespace(goal_info=SimpleNamespace())),
        UUID=lambda uuid:SimpleNamespace(uuid=uuid))
    exec(compile(ast.Module(body=methods,type_ignores=[]),str(path),'exec'),scope)
    old=Future()
    replacement=Future()
    removed,requests,events=[],[],[]
    node=SimpleNamespace(native_pending=True,native_goal_id='0123456789abcdef'*2,
        native_state='active',closed=False,sim=SimpleNamespace(execution=SimpleNamespace(fault='restart')),
        reconcile_pending=old,last_reconcile=9.5,
        native_cancel=SimpleNamespace(service_is_ready=lambda:True,remove_pending_request=removed.append,
            call_async=lambda request:requests.append(request) or replacement),
        record=lambda kind,**data:events.append((kind,data)))
    node.on_reconcile_cancel=lambda f:scope['on_reconcile_cancel'](node,f)
    old.add_done_callback(node.on_reconcile_cancel)
    scope['reconcile_native'](node)
    assert not requests and not old.cancelled()
    now[0]=10.5
    scope['reconcile_native'](node)
    assert old.cancelled() and removed==[old] and len(requests)==1
    assert bytes(requests[0].goal_info.goal_id.uuid).hex()==node.native_goal_id
    assert node.reconcile_pending is replacement
    assert node.native_pending and node.native_state=='active'
    assert events==[('native_cancel_timeout',{'goal_id':node.native_goal_id})]
    replacement.set_result(SimpleNamespace(return_code=0,goals_canceling=[object()]))
    assert events[-1][0]=='native_cancel_ack'
    assert node.native_pending and node.native_state=='active'
    # An ACK is insufficient; only the exact terminal UUID status may clear it.
    now[0]=10.6
    scope['reconcile_native'](node)
    assert len(requests)==1


def test_fleet_restart_restores_one_shared_occupancy_table_and_latches_every_robot(tmp_path):
    sim=bound_fleet()
    robot=next(iter(sim._transit))
    command=copy.deepcopy(sim.execution_for(robot).command)
    original=copy.deepcopy(sim._res.pins())
    native={r:{'native_goal_id':r+'-native','native_state':'active' if r==robot else 'none'}
            for r in sim.executions}
    store=CheckpointStore(tmp_path/'fleet.json')
    try:
        store.save(capture(sim,definition(sim),native))
        recovered=fleet()
        assert restore(recovered,store.load(),definition(recovered))==native
    finally:
        store.close()
    assert recovered._res.pins()==original
    assert recovered.tasks[0] is recovered._task_by_id['T1']
    assert recovered._transit==sim._transit and recovered._jobs==sim._jobs
    for r,gate in recovered.executions.items():
        assert gate.fault and not gate.stopped and gate.pose is None and gate.last_receipt is None
        assert gate.identity==sim.execution_for(r).identity
        gate.drain_commands()
    assert not recovered.resume_execution(command['route_id'],1.,robot)['accepted']
    xy=(-1.,0.) if robot=='R1' else (0.,1.)
    assert fleet_report(recovered,robot,'POSE',xy,now=1.)['accepted']
    assert not recovered.resume_execution(command['route_id'],1.,robot)['accepted']
    assert fleet_report(recovered,robot,'STOPPED',xy,now=1.1)['accepted']
    assert recovered.resume_execution(command['route_id'],1.2,robot)['accepted']
    assert recovered._res.pins()==original
    other=next(r for r in recovered.executions if r!=robot)
    assert recovered.execution_for(other).fault and recovered.execution_for(other).pose is None
    assert not fleet_report(recovered,robot,'ARRIVED',(0,0),now=1.3,route_id=command['route_id'])['accepted']


@pytest.mark.parametrize("loaded", [False, True])
def test_durable_roundtrip_retains_task_load_pins_statistics_and_rejects_old_proof(tmp_path, loaded):
    sim, command, xy = stopped_midway(loaded)
    fingerprint = definition(sim)
    original = copy.deepcopy(sim._res.pins())
    with_store = CheckpointStore(tmp_path / "state.json")
    try:
        with_store.save(capture(sim, fingerprint, {"native_goal_id": "0123"}))
        recovered = build()
        context = restore(recovered, with_store.load(), definition(recovered))
    finally:
        with_store.close()
    assert context["native_goal_id"] == "0123"
    assert recovered._res.pins() == original
    assert recovered.tasks[0] is recovered._task_by_id["T"]
    assert recovered.tasks[0].status is (TaskStatus.PICKED if loaded else TaskStatus.ASSIGNED)
    assert recovered._transit == sim._transit and recovered._jobs == sim._jobs
    assert recovered._odometer == sim._odometer
    assert recovered.execution.identity == sim.execution.identity
    assert recovered.execution.pose is None and not recovered.execution.stopped
    assert recovered.execution.fault and recovered.agvs[0].state is AgvState.BLOCKED
    assert not recovered.resume_execution(command["route_id"], 1.4)["accepted"]
    recovered.execution.drain_commands()
    assert report(recovered, xy=xy, now=1.5)["accepted"]
    assert not recovered.resume_execution(command["route_id"], 1.5)["accepted"]
    assert report(recovered, "STOPPED", xy, now=1.6)["accepted"]
    assert recovered.resume_execution(command["route_id"], 1.7)["accepted"]
    assert report(recovered, "ARRIVED", command["target_xy"], now=1.8)["accepted"]
    recovered.step_once()
    assert recovered._odometer == (20 if loaded else 10)


@pytest.mark.parametrize("change", ["network", "config", "task", "planner"])
def test_checkpoint_definition_mismatch_never_mutates_receiver(change):
    sim = moving()
    payload = capture(sim, definition(sim), {})
    other = build()
    if change == "network":
        other.network._edges.pop(("A", "B"))
    elif change == "config":
        other.config.dt = .5
    elif change == "task":
        other.tasks[0].dropoff = "B"
    else:
        other.planner.max_speed = 2
    before = copy.deepcopy(other.execution.snapshot())
    with pytest.raises(ValueError, match="definition mismatch"):
        restore(other, payload, definition(other))
    assert other.execution.snapshot() == before
    assert other.tasks[0].status is TaskStatus.PENDING


def test_torn_or_changed_checkpoint_is_rejected_and_second_writer_is_locked(tmp_path):
    store = CheckpointStore(tmp_path / "state.json")
    try:
        store.save({"value": 1})
        with pytest.raises(RuntimeError, match="live owner"):
            CheckpointStore(store.path)
        data = json.loads(store.path.read_bytes())
        data["payload"]["value"] = 2
        store.path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="checksum"):
            store.load()
        store.path.write_bytes(b'{"schema":1')
        with pytest.raises(ValueError):
            store.load()
    finally:
        store.close()
    replacement = CheckpointStore(store.path)
    replacement.close()


def test_failed_atomic_replace_keeps_previous_snapshot_and_cleans_temporary_file(tmp_path, monkeypatch):
    store = CheckpointStore(tmp_path / "state.json")
    try:
        store.save({"value": 1})
        def fail_replace(*args):
            raise OSError("injected disk failure")
        monkeypatch.setattr(os, "replace", fail_replace)
        with pytest.raises(OSError, match="disk failure"):
            store.save({"value": 2})
        assert store.load() == {"value": 1}
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        store.close()


def test_unknown_checkpoint_type_cannot_import_or_execute_code():
    sim = build()
    with pytest.raises(ValueError, match="unknown checkpoint type"):
        restore(sim, {"type": "os.system", "fields": {"command": "anything"}}, definition(sim))


def test_planner_random_stream_and_execution_file_identity_survive_checkpoint(tmp_path):
    sim = moving()
    sim.planner._rng = random.Random(17)
    sim.planner._rng.random()
    adapter = tmp_path / "adapter.py"
    adapter.write_text("version = 1")
    fingerprint = definition(sim, [adapter])
    payload = capture(sim, fingerprint, {})
    other = build()
    restore(other, payload, definition(other, [adapter]))
    assert [other.planner._rng.random() for _ in range(5)] == [sim.planner._rng.random() for _ in range(5)]
    adapter.write_text("version = 2")
    with pytest.raises(ValueError, match="definition mismatch"):
        restore(build(), payload, definition(build(), [adapter]))


@pytest.mark.parametrize("phase", ["idle", "loading", "unloading"])
def test_stationary_fault_freezes_handling_and_resumes_same_node_without_move(phase):
    sim = build() if phase == "idle" else moving()
    if phase == "idle":
        assert report(sim)["accepted"]
    else:
        assert report(sim, "ARRIVED", (10, 0), now=1)["accepted"]
        sim.step_once()
        sim.step_once()
        if phase == "unloading":
            for _ in range(8):
                sim.step_once()
                if sim.execution.command:
                    break
            assert report(sim, "ARRIVED", (20, 0), now=1.1)["accepted"]
            sim.step_once()
            sim.step_once()
    assert sim.execution.command is None
    token = sim.execution.recovery_token()
    sim.execution.drain_commands()
    sim.execution.fail("injected stationary fault")
    sim.execution.drain_commands()
    agv = sim.agvs[0]
    task_status, pins = sim.tasks[0].status, copy.deepcopy(sim._res.pins())
    timers = sim._load_end if phase == "loading" else sim._drop_end
    remaining = timers.get("R", sim._step_cursor) - sim._step_cursor
    for _ in range(100):
        sim.step_once()
    assert sim.tasks[0].status is task_status
    assert sim._res.pins() == pins
    if phase != "idle":
        assert timers["R"] - sim._step_cursor == remaining
    node = sim.network.node(agv.cur_node)
    assert report(sim, "STOPPED", (node.x, node.y), now=1.5)["accepted"]
    assert sim.resume_execution(token, 1.6)["accepted"]
    assert sim.execution.command is None
    assert not sim.resume_execution(token, 1.7)["accepted"]
    if phase != "idle":
        for _ in range(max(0, remaining)):
            sim.step_once()
            assert sim.tasks[0].status is task_status
        sim.step_once()
        if phase == "loading":
            assert sim.execution.command is not None
        else:
            assert sim.tasks[0].status is TaskStatus.COMPLETED



@pytest.mark.parametrize('case',['accepted','native_pending','goal_handle','guard_pending','stale_guard',
                                 'moving_guard','not_stopped','write_failed'])
def test_native_urgent_switch_requires_termination_fresh_guard_and_durable_commit(case):
    from test_external_execution import urgent_stopped_at_node
    path=Path(__file__).resolve().parents[1]/'tools/live_bridge/nav2_scheduler.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SchedulerNode')
    klass.body=[n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='preempt_execution']
    scope=dict(time=SimpleNamespace(monotonic=lambda:.7),RouteFollower=object)
    exec(compile(ast.Module(body=[klass],type_ignores=[]),str(path),'exec'),scope)
    node=scope['SchedulerNode'].__new__(scope['SchedulerNode'])
    sim=urgent_stopped_at_node()
    node.sim=sim
    node.persistence_failed=False
    node.native_pending=case=='native_pending'
    node.goal_handle=object() if case=='goal_handle' else None
    node.guard_pending=object() if case=='guard_pending' else None
    node.guard_fresh=lambda:case!='stale_guard'
    node.guard_status=SimpleNamespace(state='active' if case=='moving_guard' else 'latched',stopped=case!='not_stopped')
    writes=[]
    def persist():
        writes.append(copy.deepcopy(sim._res.pins()))
        if case=='write_failed':
            node.persistence_failed=True
            sim.execution.fail('injected durable write failure')
            return False
        return True
    node.persist=persist
    records=[]
    node.record=lambda kind,**v:records.append((kind,v))
    pins=copy.deepcopy(sim._res.pins())
    answer=node.preempt_execution(sim.execution.recovery_token())
    assert answer['accepted'] is (case=='accepted')
    assert sim._res.pins()==pins and records[-1][0]=='physical_preemption'
    if case in {'accepted','write_failed'}:
        assert len(writes)==1 and node.active is None and node.waiting_move is None
        assert sim.tasks[0].status is TaskStatus.PENDING and sim.tasks[1].status is TaskStatus.ASSIGNED
    else:
        assert writes==[] and sim.execution.fault and sim.tasks[0].status is TaskStatus.ASSIGNED
    if case=='write_failed':
        for _ in range(20):sim.step_once()
        assert sim.execution.fault and all(c['kind']=='CANCEL' for c in sim.execution.drain_commands())
        assert sim._res.pins()==pins



@pytest.mark.parametrize('operation',['submit','request','reopen'])
@pytest.mark.parametrize('write_failure',[False,True])
def test_fleet_dynamic_commands_persist_before_actuation_and_disk_failure_latches_all(operation,write_failure):
    from amr_sched.entities import Task,TaskPriority
    from test_external_execution import detour_stopped_at_node
    path=Path(__file__).resolve().parents[1]/'tools/live_bridge/fleet_scheduler.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FleetRuntime')
    scope=dict(capture=capture,time=SimpleNamespace(monotonic=lambda:.7))
    exec(compile(ast.Module(body=[klass],type_ignores=[]),str(path),'exec'),scope)
    owner=scope['FleetRuntime'].__new__(scope['FleetRuntime'])
    sim=detour_stopped_at_node() if operation=='reopen' else moving()
    sim.execution.drain_commands()
    if operation=='request':
        assert sim.submit_physical_task(Task('U','B','A',release_step=sim._step_cursor,priority=TaskPriority.URGENT))['accepted']
    owner.sim=sim
    owner.failed=False
    owner.constructing=False
    owner.closing=False
    owner.done=True
    owner.fingerprint=definition(sim)
    stops=[]
    node=SimpleNamespace(guard_id='guard',permit=1,native_goal_id=None,native_state='none',clock_s=.7,
                         seq=1,persistence_failed=False,stop_guard=lambda:stops.append('R'))
    owner.nodes={'R':node}
    writes=[]
    def save(value):
        if write_failure:raise OSError('injected disk failure')
        writes.append(value)
    owner.store=SimpleNamespace(save=save)
    owner.log_file=object()
    owner.record=lambda *args,**kw:None
    pins=copy.deepcopy(sim._res.pins())
    if operation=='submit':
        result=owner.submit_physical_task(Task('U','B','A',release_step=sim._step_cursor,priority=TaskPriority.URGENT))
    elif operation=='request':result=owner.request_physical_preemption('U')
    else:result=owner.reopen_physical_corridor(['B','C'],sim._corridor_revisions[('B','C')])
    assert result['accepted'] is (not write_failure)
    assert sim._res.pins()==pins
    if write_failure:
        assert owner.failed and node.persistence_failed and sim.execution.fault and stops==['R']
        assert not writes and all(v['kind']=='CANCEL' for v in sim.execution.drain_commands())
    else:
        assert len(writes)==1 and not owner.failed
        if operation=='submit':assert owner.done is False
