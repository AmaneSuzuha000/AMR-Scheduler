"""Physical feedback, not elapsed ETA, owns task progression and occupancy."""
import math
import copy
import importlib.util
import ast
from pathlib import Path

import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.entities import Agv, AgvState, Task, TaskPriority, TaskStatus
from amr_sched.external_execution import ExternalExecution
from amr_sched.network import Edge, Network, Node
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.reservation import Interval, ReservationTable
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.stream import FrameBuilder


def build():
    net = Network()
    for name, x in (("A", 0), ("B", 10), ("C", 20)):
        net.add_node(Node(name, x, 0))
    for src, dst in (("A", "B"), ("B", "C")):
        net.add_edge(Edge(src, dst, 10))
        net.add_edge(Edge(dst, src, 10))
    gate = ExternalExecution("test_scene", "R", (0, 0), timeout=2)
    sim = Simulator(net, [Task("T", "B", "C")], [Agv("R", start_node="A")],
                    SpaceTimeAStarPlanner(), GreedyAllocator(),
                    SimulationConfig(load_time=2, unload_time=2, max_steps=5000,
                                     idle_retreat=False, deadlock_timeout=0,
                                     reschedule_policy=ReschedulePolicy.NONE), gate)
    sim.reset()
    return sim


@pytest.mark.parametrize('case',['legacy','first','second','unlisted','invalid_option'])
def test_physical_route_acceptance_requires_frozen_equal_length_connected_choices(case):
    root=Path(__file__).resolve().parents[1]
    path=root/'docs/证据/G07-多车物理执行/analyze_fleet_navigation.py'
    if not path.exists():
        pytest.skip("内部证据脚本不在本 checkout 中")
    spec=importlib.util.spec_from_file_location('declared_route_acceptance',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first=[['S','Q'],['Q','E'],['E','X'],['X','S']]
    second=[['S','Q'],['Q','E'],['E','Q'],['Q','S']]
    fixture=dict(points={'S':(0,-1.4),'Q':(1.8,-1.4),'E':(1.8,0),'X':(0,0)},
                 edges=[('S','Q'),('Q','E'),('E','X'),('X','S')],expected_routes={'R2':first})
    route=first if case=='first' else second
    if case!='legacy':
        fixture['expected_route_options']={'R2':[first,second]}
    if case=='unlisted':
        route=[['S','X'],['X','E'],['E','Q'],['Q','S']]
    if case=='invalid_option':
        fixture['expected_route_options']['R2'].append([['S','E']])
    moves=[{'value':{'nodes':edge}} for edge in route]
    assert module.approved_routes(moves,'R2',fixture)==(case in {'first','second'})


def report(sim, kind="POSE", xy=(0, 0), now=0, **changes):
    gate = sim.execution
    msg = {**gate.envelope(), "event": kind, "pose": [*xy, 0],
           "seq": gate.last_seq + 1, "timestamp": now, **changes}
    return sim.execution_feedback(msg, now)


def moving():
    sim = build()
    assert report(sim)["accepted"]
    sim.step_once()
    assert sim.execution.state == "moving"
    return sim


def fleet():
    net = Network()
    for name, x, y in [('W',-2,0),('X',0,0),('E',2,0),('N',0,2),('S',0,-2)]:
        net.add_node(Node(name,x,y))
    for a,b in [('W','X'),('X','E'),('N','X'),('X','S')]:
        net.add_bidirectional(a,b,2.)
    gates={robot:ExternalExecution('cross',robot,xy,run_id='fleet-run',namespace_routes=True)
           for robot,xy in [('R1',(-2,0)),('R2',(0,2))]}
    sim=Simulator(net,[Task('T1','W','E'),Task('T2','N','S')],
                  [Agv('R1',start_node='W'),Agv('R2',start_node='N')],SpaceTimeAStarPlanner(dt=.1),GreedyAllocator(),
                  SimulationConfig(dt=.1,load_time=.2,unload_time=.2,max_steps=5000,
                                   idle_retreat=False,deadlock_timeout=0,
                                   reschedule_policy=ReschedulePolicy.NONE),executions=gates)
    sim.reset()
    return sim


def fleet_report(sim,robot,event='POSE',xy=None,now=0.,**changes):
    gate=sim.execution_for(robot)
    xy=xy or (tuple(gate.pose[k] for k in ('x','y')) if gate.pose else gate.start_xy)
    return sim.execution_feedback({**gate.envelope(),'event':event,'pose':[*xy,0.],
        'seq':gate.last_seq+1,'timestamp':now,**changes},now)


def bound_fleet():
    sim=fleet()
    for robot in sim.executions:
        assert fleet_report(sim,robot)['accepted']
    for _ in range(10):
        sim.step_once()
        if sim._transit:
            break
    return sim


@pytest.mark.parametrize('count',[2,4,6])
def test_declared_physical_fixture_serializes_every_loaded_robot_after_expired_eta(count):
    path=Path(__file__).resolve().parents[1]/'docs/证据/G07-多车物理执行/fleet_scene.py'
    if not path.exists():
        pytest.skip("内部证据脚本不在本 checkout 中")
    spec=importlib.util.spec_from_file_location('declared_fleet_scene',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture=module.scenario(count)
    sim=module.build_scheduler(count)
    endpoints=[xy for name,xy in fixture['points'].items() if name!='X']
    # Every parked endpoint, including a completed robot, stays outside every
    # other endpoint's circumscribed footprint for all headings.
    assert min(math.dist(a,b) for i,a in enumerate(endpoints) for b in endpoints[i+1:])>2*math.hypot(.315,.252)
    routes={r:[] for r in sim.executions}
    seen={}
    retained=0
    for step in range(3000):
        now=step*.1
        for robot,gate in sim.executions.items():
            command=gate.command
            event='POSE'
            xy=gate.start_xy if gate.pose is None else (gate.pose['x'],gate.pose['y'])
            if command:
                route=command['route_id']
                if route not in seen:
                    seen[route]=step
                    routes[robot].append(command['nodes'])
                if step-seen[route]>=150:
                    event,xy='ARRIVED',command['target_xy']
                elif step-seen[route]>125:
                    assert sim._res.pins()['nodes']['X']==[robot]
                    retained+=1
            assert fleet_report(sim,robot,event,xy,now=now)['accepted']
        sim.step_once()
        pins=sim._res.pins()
        assert all(len(owners)==1 for category in pins.values() for owners in category.values())
        assert sum(bool(g.command) for g in sim.executions.values())<=1
        assert not any(g.fault for g in sim.executions.values())
        if sim._all_done():
            break
    assert sim._all_done() and retained
    assert routes==fixture['expected_routes']
    assert sim._res.pins()==dict(nodes={v['dropoff']:[r] for r,v in fixture['robots'].items()},edges={})


@pytest.mark.parametrize('kind',['junction','reverse_edge'])
def test_permanent_physical_cut_skips_futile_time_search_and_retries_after_release(kind):
    sim=bound_fleet()
    robot=next(iter(sim._transit))
    other=next(a for a in sim.agvs if a.agv_id!=robot)
    start,goal=other.start_node,('S' if other.start_node=='N' else 'E')
    if kind=='reverse_edge':
        sim._res.transfer_pins(robot,[],[('X',start)])
    calls=[]
    original=sim.planner.plan
    def measured(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    sim.planner.plan=measured
    before=copy.deepcopy(sim._res.pins())
    assert sim._plan_and_reserve(other,start,goal,100,False) is None
    assert not calls and sim._res.pins()==before
    # A real confirmed transfer changes topology; failed searches are not cached.
    sim._res.transfer_pins(robot,['E' if goal=='S' else 'S'])
    assert sim._plan_and_reserve(other,start,goal,100,False) is not None
    assert len(calls)==1


def test_physical_reachability_keeps_spare_capacity_and_alternate_routes():
    sim=fleet()
    net=Network()
    for name,x,y in [('W',-2,0),('X',0,0),('E',2,0),('N',0,2),('S',0,-2),('Y',-3,0)]:
        net.add_node(Node(name,x,y,capacity=2 if name=='X' else 1))
    for a,b in [('W','X'),('X','E'),('N','X'),('X','S')]:
        net.add_bidirectional(a,b,2.,capacity=2)
    sim.network=net
    other=sim.agvs[1]
    sim._res.transfer_pins('R1',['X'],[('N','X'),('X','N')])
    assert sim._physical_route_possible('R2','N','S')
    assert sim._plan_and_reserve(other,'N','S',0,False) is not None
    # A second independent owner fills capacity; union counts a robot once even
    # when both directed entries describe the same physical road.
    sim._res.transfer_pins('R3',['X'],[('N','X'),('X','N')])
    assert not sim._physical_route_possible('R2','N','S')
    net.add_bidirectional('N','Y',4.)
    net.add_bidirectional('Y','S',4.)
    hops,_,_=sim._plan_and_reserve(other,'N','S',0,False)
    assert [(h.src,h.dst) for h in hops]==[('N','Y'),('Y','S')]


def test_same_scheduler_serializes_shared_junction_despite_expired_etas():
    sim=bound_fleet()
    assert len(sim._transit)==1
    robot=next(iter(sim._transit))
    other=next(r for r in sim.executions if r!=robot)
    for _ in range(100):
        sim.step_once()
    assert len(sim._transit)==1
    assert sim._res.pins()['nodes']['X']==[robot]
    assert not sim.execution_for(other).command
    assert fleet_report(sim,robot,'ARRIVED',(0,0),now=1.)['accepted']
    sim.step_once()
    for _ in range(10):
        sim.step_once()
        if sim.execution_for(robot).command:
            break
    assert sim._res.pins()['nodes']['X']==[robot]
    target=sim.execution_for(robot).command['target_xy']
    assert target != [0,0]
    assert fleet_report(sim,robot,'ARRIVED',target,now=1.1)['accepted']
    sim.step_once()
    for _ in range(10):
        sim.step_once()
        if sim.execution_for(other).command:
            break
    assert sim.execution_for(other).command['nodes']==[sim.agvs[1].start_node,'X']
    assert sim._res.pins()['nodes']['X']==[other]


def test_fleet_fault_stop_and_resume_retain_task_and_all_shared_resources():
    sim=bound_fleet()
    robot=next(iter(sim._transit))
    gate=sim.execution_for(robot)
    old=copy.deepcopy(gate.command)
    pins=sim._res.pins()
    task=next(t for t in sim.tasks if t.task_id==old['task_id'])
    gate.fail('injected physical loss')
    sim.poll_execution(.1)
    for _ in range(100):
        sim.step_once()
    assert sim._res.pins()==pins and task.status is TaskStatus.PICKED
    assert not sim.resume_execution(old['route_id'],.2,robot)['accepted']
    gate.drain_commands()
    assert fleet_report(sim,robot,'STOPPED',(-1.,0.) if robot=='R1' else (0.,1.),now=.3)['accepted']
    assert not sim.resume_execution(old['route_id'],.3)['accepted']
    answer=sim.resume_execution(old['route_id'],.3,robot)
    assert answer['accepted'] and sim._res.pins()==pins
    assert answer['command']['route_id']!=old['route_id']
    assert answer['command']['task_id']==old['task_id'] and task.status is TaskStatus.PICKED
    assert not fleet_report(sim,robot,'ARRIVED',(0,0),now=.4,route_id=old['route_id'])['accepted']


def test_fleet_frames_never_project_another_robots_pose_or_refresh_crossed_packets():
    sim=bound_fleet()
    r1,r2=sim.execution_for('R1'),sim.execution_for('R2')
    crossed={**r1.envelope(),'robot_id':'R2','event':'POSE','pose':[-1.,0.,0.],
             'seq':10000,'timestamp':.5}
    before=copy.deepcopy(vars(r2))
    assert not sim.execution_feedback(crossed,.5)['accepted']
    assert vars(r2)==before
    assert fleet_report(sim,'R1',xy=(-1.,0.),now=.5)['accepted']
    builder=FrameBuilder(sim)
    frame=builder.frame()
    poses={a['agv_id']:a['pose'] for a in frame['agvs']}
    assert [poses['R1']['x'],poses['R1']['y']]==[-1.,0.]
    assert [poses['R2']['x'],poses['R2']['y']]==[0.,2.]
    assert poses['R1']['source']==poses['R2']['source']=='gazebo'
    assert set(frame['executions'])=={'R1','R2'}
    sim.poll_execution(2.1)
    frame=builder.frame()
    poses={a['agv_id']:a['pose'] for a in frame['agvs']}
    assert poses['R1']['available'] and not poses['R2']['available']
    assert frame['physical_holds']==sim._res.pins()


def test_fleet_route_ids_are_unique_within_one_shared_run():
    sim=fleet()
    for robot,gate in sim.executions.items():
        assert fleet_report(sim,robot)['accepted']
        gate.begin('test','W','X',(0.,0.),.2)
    identifiers=[gate.command['route_id'] for gate in sim.executions.values()]
    assert len(set(identifiers))==2
    assert identifiers==['fleet-run:R1:1','fleet-run:R2:1']


@pytest.mark.parametrize('field,value', [('scene_id','other'),('run_id','other'),
    ('frame_id','other'),('time_base','wall'),('source','localized')])
def test_fleet_rejects_mixed_authority_identity(field,value):
    sim=fleet()
    sim.executions['R2'].identity[field]=value
    with pytest.raises(ValueError,match='must share'):
        Simulator(sim.network,sim.tasks,sim.agvs,sim.planner,sim.allocator,sim.config,
                  executions=sim.executions)


def test_physical_mapping_must_cover_every_robot_and_have_unique_routes():
    sim=fleet()
    with pytest.raises(ValueError,match='must not be empty'):
        Simulator(sim.network,sim.tasks,sim.agvs,sim.planner,sim.allocator,sim.config,executions={})
    with pytest.raises(ValueError,match='exactly one'):
        Simulator(sim.network,sim.tasks,sim.agvs,sim.planner,sim.allocator,sim.config,
                  executions={'R1':sim.executions['R1']})
    sim.executions['R2'].namespace_routes=False
    with pytest.raises(ValueError,match='namespaced'):
        Simulator(sim.network,sim.tasks,sim.agvs,sim.planner,sim.allocator,sim.config,
                  executions=sim.executions)


def test_fleet_adapters_share_the_actual_scheduler_and_cannot_advance_it():
    path=Path(__file__).resolve().parents[1]/'tools/live_bridge/fleet_scheduler.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RobotExecutionView')
    scope={}
    exec(compile(ast.Module(body=[klass],type_ignores=[]),str(path),'exec'),scope)
    sim=bound_fleet()
    views=[scope['RobotExecutionView'](sim,robot) for robot in sim.executions]
    for view in views:
        assert view.executions is sim.executions and view._res is sim._res and view.tasks is sim.tasks
        assert view.execution is sim.execution_for(view.robot_id)
        with pytest.raises(RuntimeError,match='only the fleet owner'):
            view.step_once()
        with pytest.raises(RuntimeError,match='cannot reset'):
            view.reset()


def test_two_feedback_driven_jobs_finish_in_one_scheduler_without_eta_arrivals():
    sim=bound_fleet()
    admitted={}
    arrivals={robot:0 for robot in sim.executions}
    for step in range(1000):
        now=.01*(step+1)
        for robot,gate in sim.executions.items():
            assert fleet_report(sim,robot,now=now)['accepted']
            for command in gate.drain_commands():
                assert command['kind']=='MOVE'
                admitted[command['route_id']]=step
            if gate.command:
                route=gate.command['route_id']
                if step-admitted[route]>=25:
                    assert fleet_report(sim,robot,'ARRIVED',gate.command['target_xy'],now=now)['accepted']
                    arrivals[robot]+=1
        if not sim.step_once():
            break
    assert sim._all_done() and arrivals=={'R1':2,'R2':2}
    assert all(t.status is TaskStatus.COMPLETED for t in sim.tasks)
    assert sim._res.pins()=={'nodes':{'E':['R1'],'S':['R2']},'edges':{}}
    assert sim._odometer==8. and sim._material_waste_steps==0.


def test_eta_never_completes_physical_movement_and_pins_survive_cleanup():
    sim = moving()
    for _ in range(1000):
        sim.step_once()
    assert sim.agvs[0].cur_node == "A"
    assert sim.tasks[0].status is TaskStatus.ASSIGNED
    sim._res.release_all("R")
    sim._res.clear()
    sim._res.purge_before(90000)
    assert not sim._res.is_edge_free("B", "A", Interval(99999, 100000), 1)
    assert sim._res.node_conflicts("B", Interval(99999, 100000)) == ["R"]


def test_actual_arrivals_advance_the_existing_job_then_complete_after_handling():
    sim = moving()
    for _ in range(30):
        sim.step_once()
    assert report(sim, "ARRIVED", (10, 0), now=1)["accepted"]
    sim.step_once()
    assert sim.agvs[0].cur_node == "B"
    assert sim._res.pins()["edges"] == {}
    assert sim._res.pins()["nodes"] == {"B": ["R"]}
    sim.step_once()
    assert sim.tasks[0].status is TaskStatus.PICKED
    for _ in range(5):
        sim.step_once()
        if sim.execution.command:
            break
    assert sim.execution.command["nodes"] == ["B", "C"]
    assert sim.tasks[0].complete_step is None
    assert report(sim, "ARRIVED", (20, 0), now=1.5)["accepted"]
    for _ in range(5):
        sim.step_once()
    assert sim.tasks[0].status is TaskStatus.COMPLETED
    assert sim._odometer == 20


@pytest.mark.parametrize("field,value", [
    ("run_id", "old"), ("task_id", "other"), ("route_id", "old"),
    ("robot_id", "X"), ("scene_id", "other"), ("frame_id", "world"),
    ("time_base", "wall"), ("source", "odom"), ("seq", 0),
    ("seq", True), ("timestamp", -1), ("timestamp", math.nan),
    ("pose", [math.inf, 0, 0]), ("pose", [10, 0]), ("event", "SUCCESS"),
])
def test_malformed_or_misbound_feedback_cannot_advance_or_refresh(field, value):
    sim = moving()
    assert not report(sim, "ARRIVED", (10, 0), now=1, **{field: value})["accepted"]
    assert sim.execution.state == "moving"
    assert sim.execution.last_receipt == 0


def test_arrival_requires_target_pose_and_duplicate_cannot_finish_another_leg():
    sim = moving()
    assert not report(sim, "ARRIVED", (8, 0), now=1)["accepted"]
    old = {**sim.execution.envelope(), "event": "ARRIVED", "pose": [10, 0, 0],
           "seq": 1, "timestamp": 1}
    assert sim.execution_feedback(old, 1)["accepted"]
    assert not sim.execution_feedback(old, 1.1)["accepted"]
    for _ in range(5):
        sim.step_once()
    assert sim.execution.command["nodes"] == ["B", "C"]
    assert not sim.execution_feedback({**old, "seq": 999}, 1.2)["accepted"]
    assert sim.tasks[0].status is TaskStatus.PICKED


@pytest.mark.parametrize("fault", ["FAILED", "timeout"])
def test_fault_retains_task_resources_and_late_success_cannot_recover(fault):
    sim = moving()
    sim.execution.drain_commands()
    if fault == "FAILED":
        assert report(sim, "FAILED", (3, 0), now=1)["accepted"]
    else:
        sim.poll_execution(2.1)
    for _ in range(30):
        sim.step_once()
    commands = sim.execution.drain_commands()
    assert [c["kind"] for c in commands] == ["CANCEL"]
    assert not report(sim, "ARRIVED", (10, 0), now=2.2)["accepted"]
    assert sim.tasks[0].status is TaskStatus.ASSIGNED
    assert sim.agvs[0].state is AgvState.BLOCKED
    assert sim._res.pins()["edges"]["A->B"] == ["R"]
    assert report(sim, "STOPPED", (3, 0), now=2.3)["accepted"]
    assert sim.execution.stopped and sim.execution.fault
    assert sim._cancel_job(sim.agvs[0]) is False
    assert sim.report_deviation("R", ("A", "B"), [3, 0, 0], 2.3)["action"] == "rejected"
    with pytest.raises(RuntimeError, match="cannot reset"):
        sim.reset()


def test_late_arrival_alone_cannot_resurrect_a_timed_out_source():
    sim = moving()
    assert not report(sim, "ARRIVED", (10, 0), now=3)["accepted"]
    assert sim.execution.fault == "feedback timeout"


def test_recovered_source_updates_truth_but_cannot_resume_a_faulted_task():
    sim = moving()
    sim.poll_execution(2.1)
    assert sim.execution.pose is None
    assert report(sim, xy=(3, 0), now=2.2)["accepted"]
    assert sim.execution.pose["x"] == 3
    assert sim.execution.fault == "feedback timeout"
    assert not report(sim, "ARRIVED", (10, 0), now=2.3)["accepted"]
    sim.step_once()
    assert sim.agvs[0].state is AgvState.BLOCKED
    assert sim._res.pins()["edges"]["A->B"] == ["R"]


def test_new_sequences_on_a_frozen_source_stamp_do_not_refresh_the_watchdog():
    sim = moving()
    assert report(sim, xy=(3, 0), now=1)["accepted"]
    answer = report(sim, xy=(4, 0), now=2.9, timestamp=1)
    assert not answer["accepted"]
    assert sim.execution.pose["x"] == 3
    assert sim.execution.last_receipt == 1
    sim.poll_execution(3.01)
    assert sim.execution.fault == "feedback timeout"
    assert sim.execution.pose is None


def test_a_second_event_on_one_observation_cannot_extend_freshness():
    sim = moving()
    assert report(sim, "FAILED", (3, 0), now=1)["accepted"]
    assert report(sim, "STOPPED", (3, 0), now=2.9, timestamp=1)["accepted"]
    assert sim.execution.last_receipt == 1
    sim.poll_execution(3.01)
    assert sim.execution.pose is None
    assert sim.tasks[0].status is TaskStatus.ASSIGNED
    assert sim._res.pins()["edges"]["A->B"] == ["R"]


def stopped_midway(loaded=False):
    sim = moving()
    if loaded:
        assert report(sim, "ARRIVED", (10, 0), now=1)["accepted"]
        for _ in range(8):
            sim.step_once()
            if sim.execution.command and sim.execution.command["nodes"] == ["B", "C"]:
                break
        assert sim.tasks[0].status is TaskStatus.PICKED
    old = dict(sim.execution.command)
    sim.execution.drain_commands()
    xy = (13, 0) if loaded else (3, 0)
    assert report(sim, "FAILED", xy, now=1.2)["accepted"]
    assert [c["kind"] for c in sim.execution.drain_commands()] == ["CANCEL"]
    assert report(sim, "STOPPED", xy, now=1.3)["accepted"]
    return sim, old, xy


@pytest.mark.parametrize("loaded", [False, True])
def test_explicit_mid_edge_recovery_preserves_task_load_and_pins_until_new_arrival(loaded):
    sim, old, xy = stopped_midway(loaded)
    pins = copy.deepcopy(sim._res.pins())
    task_status = sim.tasks[0].status
    transit = sim._transit.copy()
    node_before = sim.agvs[0].cur_node
    answer = sim.resume_execution(old["route_id"], 1.4)
    assert answer["accepted"]
    new = answer["command"]
    assert new["route_id"] != old["route_id"] and new["revision"] == old["revision"] + 1
    assert new["resumes_route_id"] == old["route_id"]
    assert new["resume_from"] == list(xy)
    assert new["nodes"] == old["nodes"] and new["task_id"] == old["task_id"]
    assert sim._transit == transit and sim.agvs[0].cur_node == node_before
    assert sim._res.pins() == pins and sim.tasks[0].status is task_status
    obsolete = {**old, "event": "ARRIVED", "pose": [*old["target_xy"], 0],
                "seq": 999, "timestamp": 1.5}
    assert not sim.execution_feedback(obsolete, 1.5)["accepted"]
    for _ in range(30):
        sim.step_once()
    assert sim.agvs[0].cur_node == node_before and sim._res.pins() == pins
    assert report(sim, "ARRIVED", old["target_xy"], now=1.6)["accepted"]
    sim.step_once()
    assert sim.agvs[0].cur_node == old["nodes"][1]
    assert sim._res.pins()["edges"] == {}
    assert sim._odometer == (20 if loaded else 10)
    assert sim._material_waste_steps == 0


@pytest.mark.parametrize("case", ["no_stop", "stale", "wrong_action", "outside", "lost_pins", "pending_cancel"])
def test_recovery_requires_fresh_stop_correct_action_and_retained_corridor(case):
    sim, old, xy = stopped_midway()
    now = 1.4
    expected = old["route_id"]
    if case == "no_stop":
        sim.execution.stopped = False
    elif case == "stale":
        now = 3.4
    elif case == "wrong_action":
        expected = "obsolete_action"
    elif case == "outside":
        assert report(sim, xy=(3, 1), now=1.35)["accepted"]
    elif case == "lost_pins":
        sim._res.transfer_pins("R", ["A"])
    elif case == "pending_cancel":
        sim.execution._outbox.append({"kind": "CANCEL"})
    pins = copy.deepcopy(sim._res.pins())
    revision = sim.execution.revision
    assert not sim.resume_execution(expected, now)["accepted"]
    assert sim.execution.fault and sim.execution.revision == revision
    assert sim._res.pins() == pins and sim.tasks[0].status is TaskStatus.ASSIGNED


def test_recovery_is_explicit_and_cannot_be_replayed_after_it_succeeds():
    sim, old, _ = stopped_midway()
    assert sim.resume_execution(old["route_id"], 1.4)["accepted"]
    new = sim.execution.command.copy()
    assert not sim.resume_execution(old["route_id"], 1.5)["accepted"]
    assert sim.execution.command == new


def test_startup_waits_for_pose_at_the_declared_start():
    sim = build()
    for _ in range(5):
        sim.step_once()
    assert sim.tasks[0].status is TaskStatus.PENDING
    assert not report(sim, xy=(10, 0))["accepted"]
    assert sim.execution.drain_commands() == []
    assert report(sim)["accepted"]
    sim.step_once()
    assert sim.execution.command["nodes"] == ["A", "B"]


def test_display_uses_actual_pose_and_hides_stale_truth_without_extrapolation():
    sim = moving()
    builder = FrameBuilder(sim)
    assert report(sim, xy=(2.34, .02), now=1)["accepted"]
    for _ in range(50):
        sim.step_once()
    frame = builder.frame()
    assert frame["agvs"][0]["pose"]["x"] == 2.34
    assert frame["trail_sources"] == {"R": "gazebo"}
    assert frame["agvs"][0]["path"] == ["A", "B"]
    assert "physical" not in frame
    trail = frame["trails"]["R"].copy()
    sim.poll_execution(3.1)
    frame = builder.frame()
    assert frame["agvs"][0]["pose"]["available"] is False
    assert frame["agvs"][0]["pose"]["x"] is None
    assert frame["trails"]["R"] == trail


def test_pins_count_owner_once_when_a_time_window_also_exists():
    res = ReservationTable()
    res.transfer_pins("R", ["B"], [("A", "B")])
    res.reserve_node("B", Interval(1, 3), "R")
    assert res.is_node_free("B", Interval(1, 2), 2)
    assert res.is_node_free("B", Interval(1, 2), 1, "R")
    res.transfer_pins("R", ["C"])
    res.purge_before(4)
    assert res.node_conflicts("B", Interval(5, 6)) == []
    assert res.node_conflicts("C", Interval(5, 6)) == ["R"]


def detour_sim():
    """A loaded route with a genuine graph bypass, all coordinates in metres."""
    net = Network()
    for name, x, y in [('A',0,0),('B',1,0),('C',2,0),('D',1,1),('E',2,1)]:
        net.add_node(Node(name,x,y))
    for src,dst in [('A','B'),('B','C'),('B','D'),('D','E'),('E','C')]:
        a,b=net.node(src),net.node(dst)
        net.add_bidirectional(src,dst,math.hypot(b.x-a.x,b.y-a.y))
    gate=ExternalExecution('detour','R',(0,0))
    sim=Simulator(net,[Task('T','A','C')],[Agv('R',start_node='A')],
        SpaceTimeAStarPlanner(dt=.1),GreedyAllocator(),
        SimulationConfig(dt=.1,load_time=.1,unload_time=.1,max_steps=2000,
            idle_retreat=False,deadlock_timeout=0,reschedule_policy=ReschedulePolicy.NONE),gate)
    sim.reset()
    assert report(sim)['accepted']
    for _ in range(20):
        sim.step_once()
        if gate.command:
            break
    assert gate.command['nodes']==['A','B']
    assert sim.tasks[0].status is TaskStatus.PICKED
    gate.drain_commands()
    return sim


def detour_stopped_at_node():
    sim=detour_sim()
    gate=sim.execution
    assert report(sim,xy=(.4,0),now=.1)['accepted']
    token=gate.recovery_token()
    assert sim.request_physical_detour('R',token,['B','C'],.2)['accepted']
    gate.drain_commands()
    assert report(sim,'STOPPED',(.4,0),now=.3)['accepted']
    assert not sim.replan_execution(token,.4)['accepted']
    assert sim.resume_execution(token,.4)['accepted']
    gate.drain_commands()
    assert report(sim,'ARRIVED',(1,0),now=.5)['accepted']
    sim.step_once()
    assert gate.command is None and gate.fault and sim.agvs[0].cur_node=='B'
    gate.drain_commands()
    assert report(sim,'STOPPED',(1,0),now=.6)['accepted']
    return sim


def test_physical_detour_retains_loaded_hop_until_arrival_then_reserves_a_real_bypass():
    sim=detour_sim()
    gate=sim.execution
    old=copy.deepcopy(gate.command)
    pins=copy.deepcopy(sim._res.pins())
    transit=copy.deepcopy(sim._transit)
    assert report(sim,xy=(.4,0),now=.1)['accepted']
    assert sim.request_physical_detour('R',old['route_id'],['B','C'],.2)['accepted']
    gate.drain_commands()
    assert report(sim,'STOPPED',(.4,0),now=.3)['accepted']
    assert not sim.replan_execution(old['route_id'],.4)['accepted']
    assert sim._res.pins()==pins and sim._transit==transit
    assert sim.agvs[0].cur_node=='A' and gate.pose['x']==.4
    assert sim.resume_execution(old['route_id'],.4)['accepted']
    retained=copy.deepcopy(gate.command)
    assert retained['nodes']==old['nodes'] and retained['resume_from']==[.4,0.]
    gate.drain_commands()
    assert report(sim,'ARRIVED',(1,0),now=.5)['accepted']
    sim.step_once()
    assert gate.fault and gate.command is None and sim.agvs[0].cur_node=='B'
    assert sim._res.pins()=={'nodes':{'B':['R']},'edges':{}}
    assert not sim.resume_execution(gate.recovery_token(),.55)['accepted']
    gate.drain_commands()
    assert report(sim,'STOPPED',(1,0),now=.6)['accepted']
    pins=copy.deepcopy(sim._res.pins())
    token=gate.recovery_token()
    answer=sim.replan_execution(token,.7)
    assert answer['accepted'] and answer['nodes']==['B','D','E','C']
    assert sim._res.pins()==pins and gate.drain_commands()==[]
    assert sim.tasks[0].status is TaskStatus.PICKED
    assert not sim.replan_execution(token,.71)['accepted']
    assert not report(sim,'ARRIVED',(1,0),now=.8,route_id=retained['route_id'])['accepted']
    routes=[]
    now=.9
    for _ in range(200):
        sim.step_once()
        for command in gate.drain_commands():
            assert command['kind']=='MOVE'
            routes.append(command['nodes'])
            assert command['revision']>retained['revision']
            assert report(sim,'ARRIVED',command['target_xy'],now=now)['accepted']
            now+=.1
        if sim._all_done():
            break
    assert routes==[['B','D'],['D','E'],['E','C']]
    assert sim._all_done() and sim.tasks[0].assigned_agv=='R'
    assert sim._material_waste_steps==0 and sim._res.pins()=={'nodes':{'C':['R']},'edges':{}}
    assert sim.network.has_edge('B','C') and not sim._routing_network().has_edge('C','B')


@pytest.mark.parametrize('case',['unknown_robot','obsolete_route','unknown_edge','unrelated_edge','nan_time','future_receipt'])
def test_physical_detour_rejects_misbound_reports_without_closing_or_releasing(case):
    sim=detour_sim()
    gate=sim.execution
    robot,token,edge,now='R',gate.recovery_token(),['B','C'],.1
    if case=='unknown_robot': robot='OTHER'
    elif case=='obsolete_route': token='old'
    elif case=='unknown_edge': edge=['MISSING','C']
    elif case=='unrelated_edge': edge=['D','E']
    elif case=='nan_time': now=math.nan
    elif case=='future_receipt': now=-.1
    pins=copy.deepcopy(sim._res.pins())
    assert not sim.request_physical_detour(robot,token,edge,now)['accepted']
    assert not gate.fault and not sim._physical_detours and not sim._closed_corridors
    assert sim._res.pins()==pins and sim.tasks[0].status is TaskStatus.PICKED


def test_closed_current_physical_corridor_cannot_be_resumed_or_released_from_mid_edge():
    sim=detour_sim()
    gate=sim.execution
    assert report(sim,xy=(.4,0),now=.1)['accepted']
    pins=copy.deepcopy(sim._res.pins())
    assert sim.request_physical_detour('R',gate.recovery_token(),['B','A'],.2)['accepted']
    gate.drain_commands()
    assert report(sim,'STOPPED',(.4,0),now=.3)['accepted']
    assert not sim.resume_execution(gate.recovery_token(),.4)['accepted']
    assert not sim.replan_execution(gate.recovery_token(),.4)['accepted']
    assert sim._res.pins()==pins and sim.agvs[0].cur_node=='A'
    assert not sim._can_enter(sim.agvs[0],'A','B',sim._step_cursor)
    assert not sim._can_enter(sim.agvs[0],'B','A',sim._step_cursor)


@pytest.mark.parametrize('case',['no_stop','stale','obsolete_route','outside_node','lost_node','extra_edge','pending_cancel','blocked_bypass'])
def test_physical_node_detour_failure_retains_task_and_original_reservations(case):
    sim=detour_stopped_at_node()
    gate=sim.execution
    now,token=.7,gate.recovery_token()
    if case=='no_stop': gate.stopped=False
    elif case=='stale': now=2.7
    elif case=='obsolete_route': token='old'
    elif case=='outside_node': assert report(sim,xy=(1,.2),now=.65)['accepted']
    elif case=='lost_node': sim._res.transfer_pins('R',[])
    elif case=='extra_edge': sim._res.transfer_pins('R',['B'],[('A','B')])
    elif case=='pending_cancel': gate._outbox.append({'kind':'CANCEL'})
    elif case=='blocked_bypass': sim._res.transfer_pins('other',['D'])
    pins=copy.deepcopy(sim._res.pins())
    reservations=copy.deepcopy(vars(sim._res))
    job=copy.deepcopy(sim._jobs['R'])
    revision=gate.revision
    assert not sim.replan_execution(token,now)['accepted']
    assert gate.fault and gate.revision==revision
    assert sim._res.pins()==pins and vars(sim._res)==reservations
    assert sim._jobs['R']==job and sim.tasks[0].status is TaskStatus.PICKED
    assert sim._physical_detours and sim._closed_corridors=={('B','C')}


def test_physical_detour_checkpoint_keeps_closure_and_discards_stale_stop():
    from amr_sched.checkpoint import capture,definition,restore
    sim=detour_stopped_at_node()
    fingerprint=definition(sim)
    payload=capture(sim,fingerprint,{})
    recovered=detour_sim()
    restore(recovered,payload,fingerprint)
    assert recovered._closed_corridors=={('B','C')}
    assert recovered._physical_detours==sim._physical_detours
    assert recovered._res.pins()==sim._res.pins()
    token=recovered.execution.recovery_token()
    assert not recovered.replan_execution(token,.7)['accepted']
    recovered.execution.drain_commands()
    assert report(recovered,'STOPPED',(1,0),now=.8)['accepted']
    assert recovered.replan_execution(token,.9)['accepted']



def urgent_stopped_at_node(pickup='B'):
    sim=build()
    sim.tasks[0].pickup=pickup
    assert report(sim)['accepted']
    sim.step_once()
    gate=sim.execution
    gate.drain_commands()
    assert report(sim,xy=(4,0),now=.1)['accepted']
    urgent=Task('U','B','A',release_step=sim._step_cursor,priority=TaskPriority.URGENT)
    assert sim.submit_physical_task(urgent)['accepted']
    assert sim.request_physical_preemption('U',.2)['accepted']
    gate.drain_commands()
    assert report(sim,'STOPPED',(4,0),now=.3)['accepted']
    assert sim.resume_execution(gate.recovery_token(),.4)['accepted']
    gate.drain_commands()
    assert report(sim,'ARRIVED',(10,0),now=.5)['accepted']
    sim.step_once()
    assert gate.command is None and gate.fault and sim.tasks[0].status is TaskStatus.ASSIGNED
    gate.drain_commands()
    assert report(sim,'STOPPED',(10,0),now=.6)['accepted']
    return sim


def test_physical_corridor_clear_is_versioned_and_never_releases_or_resumes():
    sim=detour_sim()
    gate=sim.execution
    answer=sim.request_physical_detour('R',gate.recovery_token(),['B','C'],.1)
    revision=answer['closure_revision']
    pins=copy.deepcopy(sim._res.pins())
    assert sim.reopen_physical_corridor(['C','B'],revision)['accepted']
    assert gate.fault and sim._res.pins()==pins and sim._physical_detours
    assert sim._routing_network().has_edge('B','C')
    assert not sim.reopen_physical_corridor(['B','C'],revision)['accepted']
    closed=sim.request_physical_detour('R',gate.recovery_token(),['B','C'],.2)
    assert closed['accepted'] and closed['closure_revision']>revision
    assert not sim.reopen_physical_corridor(['B','C'],revision)['accepted']
    assert sim._closed_corridors=={('B','C')} and sim._res.pins()==pins
    assert gate.drain_commands()[0]['kind']=='CANCEL'


@pytest.mark.parametrize('edge,revision',[(None,1),(['B'],1),(['C','B'],True),(['unknown','C'],1),(['B','C'],0)])
def test_physical_corridor_clear_rejects_invalid_or_old_events(edge,revision):
    sim=detour_stopped_at_node()
    pins=copy.deepcopy(vars(sim._res))
    assert not sim.reopen_physical_corridor(edge,revision)['accepted']
    assert vars(sim._res)==pins and sim._closed_corridors=={('B','C')} and sim.execution.fault


def test_dynamic_physical_tasks_are_owned_idempotent_and_survive_restart():
    from amr_sched.checkpoint import capture,definition,restore
    sim=moving()
    fingerprint=definition(sim)
    request=Task('U','B','A',release_step=sim._step_cursor,priority=TaskPriority.URGENT)
    assert sim.submit_physical_task(request)==dict(accepted=True,duplicate=False,task_id='U',release_step=sim._step_cursor)
    assert sim._task_by_id['U'] is not request
    assert sim.submit_physical_task(copy.deepcopy(request))['duplicate']
    request.dropoff='C'
    assert not sim.submit_physical_task(request)['accepted']
    assert sim._task_by_id['U'].dropoff=='A'
    payload=capture(sim,fingerprint,{})
    other=build()
    restore(other,payload,fingerprint)
    assert len(other.tasks)==2 and other._task_by_id['U'] is other.tasks[1]
    assert other.submit_physical_task(copy.deepcopy(sim._submitted_tasks['U']))['duplicate']


@pytest.mark.parametrize('case',['duplicate_initial','unknown_node','raw_priority','past_release','bool_release',
                                 'assigned','picked','runtime_stamp','invalid_due','invalid_material'])
def test_dynamic_physical_task_validation_keeps_existing_lifecycle_and_pins(case):
    sim=moving()
    request=Task('U','B','A',release_step=sim._step_cursor,priority=TaskPriority.URGENT)
    if case=='duplicate_initial':request.task_id='T'
    elif case=='unknown_node':request.pickup='missing'
    elif case=='raw_priority':request.priority=1
    elif case=='past_release':request.release_step=0
    elif case=='bool_release':request.release_step=True
    elif case=='assigned':request.assigned_agv='R'
    elif case=='picked':request.status=TaskStatus.PICKED
    elif case=='runtime_stamp':request.pickup_step=2
    elif case=='invalid_due':request.due_step=float('nan')
    elif case=='invalid_material':request.material_type='wheelset'
    pins=copy.deepcopy(sim._res.pins())
    assert not sim.submit_physical_task(request)['accepted']
    assert len(sim.tasks)==1 and not sim._submitted_tasks and sim._res.pins()==pins
    assert sim.tasks[0].status is TaskStatus.ASSIGNED


def test_physical_urgent_switch_preserves_transit_until_node_and_completes_both_tasks():
    sim=moving()
    gate=sim.execution
    gate.drain_commands()
    assert report(sim,xy=(4,0),now=.1)['accepted']
    assert sim.submit_physical_task(Task('U','B','A',release_step=sim._step_cursor,priority=TaskPriority.URGENT))['accepted']
    pins=copy.deepcopy(sim._res.pins())
    answer=sim.request_physical_preemption('U',.2)
    assert answer['accepted'] and answer['robot_id']=='R' and gate.fault
    gate.drain_commands()
    assert not sim.preempt_execution(gate.recovery_token(),.2)['accepted']
    assert report(sim,'STOPPED',(4,0),now=.3)['accepted']
    assert not sim.preempt_execution(gate.recovery_token(),.4)['accepted']
    assert sim._res.pins()==pins and sim._transit and sim.agvs[0].cur_node=='A'
    assert sim.tasks[0].status is TaskStatus.ASSIGNED and sim.tasks[1].status is TaskStatus.PENDING
    assert sim.resume_execution(gate.recovery_token(),.4)['accepted']
    old=copy.deepcopy(gate.command)
    assert old['nodes']==['A','B'] and old['resume_from']==[4.,0.]
    gate.drain_commands()
    assert report(sim,'ARRIVED',(10,0),now=.5)['accepted']
    sim.step_once()
    assert gate.fault and sim.tasks[0].status is TaskStatus.ASSIGNED
    assert not sim.resume_execution(gate.recovery_token(),.55)['accepted']
    gate.drain_commands()
    assert report(sim,'STOPPED',(10,0),now=.6)['accepted']
    pins=copy.deepcopy(sim._res.pins())
    token=gate.recovery_token()
    answer=sim.preempt_execution(token,.7)
    assert answer['accepted'] and answer['nodes']==['B']
    assert sim._res.pins()==pins=={'nodes':{'B':['R']},'edges':{}}
    assert sim.tasks[0].status is TaskStatus.PENDING and sim.tasks[0].assigned_agv is None
    assert sim.tasks[0].pickup_step is None and sim.tasks[1].assigned_agv=='R'
    assert not sim.preempt_execution(token,.71)['accepted']
    assert not report(sim,'ARRIVED',(10,0),now=.8,route_id=old['route_id'])['accepted']
    now=.9
    routes=[]
    for _ in range(500):
        assert report(sim,xy=tuple(gate.pose[k] for k in ('x','y')),now=now)['accepted']
        now+=.01
        sim.step_once()
        for command in gate.drain_commands():
            assert command['kind']=='MOVE'
            routes.append((command['task_id'],command['nodes']))
            assert report(sim,'ARRIVED',command['target_xy'],now=now)['accepted']
            now+=.01
        if sim._all_done():break
    assert sim._all_done() and routes==[('U',['B','A']),('T',['A','B']),('T',['B','C'])]
    assert sim._preemptions==1 and sim._material_waste_steps==0
    assert sim.tasks[1].complete_step<sim.tasks[0].complete_step
    assert sim._res.pins()=={'nodes':{'C':['R']},'edges':{}}


@pytest.mark.parametrize('case',['no_stop','stale','wrong_route','outside','lost_node','extra_edge',
                                 'pending_cancel','loaded','urgent_assigned','blocked_pickup'])
def test_failed_physical_urgent_commit_keeps_task_load_route_and_reservations(case):
    sim=urgent_stopped_at_node()
    gate=sim.execution
    now,token=.7,gate.recovery_token()
    if case=='no_stop':gate.stopped=False
    elif case=='stale':now=2.7
    elif case=='wrong_route':token='old'
    elif case=='outside':assert report(sim,xy=(10,.2),now=.65)['accepted']
    elif case=='lost_node':sim._res.transfer_pins('R',[])
    elif case=='extra_edge':sim._res.transfer_pins('R',['B'],[('A','B')])
    elif case=='pending_cancel':gate._outbox.append({'kind':'CANCEL'})
    elif case=='loaded':sim.tasks[0].status=TaskStatus.PICKED
    elif case=='urgent_assigned':sim.tasks[1].status=TaskStatus.ASSIGNED
    elif case=='blocked_pickup':
        sim.tasks[1].pickup='A'
        sim._res.transfer_pins('other',['A'])
    reservations=copy.deepcopy(vars(sim._res))
    job=copy.deepcopy(sim._jobs['R'])
    tasks=copy.deepcopy(sim.tasks)
    assert not sim.preempt_execution(token,now)['accepted']
    assert gate.fault and sim.tasks==tasks and sim._jobs['R']==job
    assert vars(sim._res)==reservations and sim._physical_preemptions


@pytest.mark.parametrize('case',['loaded','old_request','normal','unreleased','invalid_time'])
def test_physical_urgent_request_rejects_ineligible_targets(case):
    sim=detour_sim() if case=='loaded' else moving()
    release=sim._step_cursor+int(case=='unreleased')
    priority=TaskPriority.NORMAL if case=='normal' else TaskPriority.URGENT
    assert sim.submit_physical_task(Task('U','B','A',release_step=release,priority=priority))['accepted']
    if case=='old_request':assert sim.request_physical_preemption('U',.1)['accepted']
    pins=copy.deepcopy(sim._res.pins())
    answer=sim.request_physical_preemption('U',math.nan if case=='invalid_time' else .2)
    assert not answer['accepted'] and sim._res.pins()==pins
    assert sim._preemptions==0 and sim._material_waste_steps==0


def test_pending_physical_urgent_restart_requires_new_stop_and_keeps_both_tasks():
    from amr_sched.checkpoint import capture,definition,restore
    initial=build()
    fingerprint=definition(initial)
    sim=urgent_stopped_at_node()
    recovered=build()
    restore(recovered,capture(sim,fingerprint,{}),fingerprint)
    assert recovered._physical_preemptions==sim._physical_preemptions
    assert recovered._res.pins()==sim._res.pins() and len(recovered.tasks)==2
    token=recovered.execution.recovery_token()
    assert not recovered.preempt_execution(token,.7)['accepted']
    recovered.execution.drain_commands()
    assert report(recovered,'STOPPED',(10,0),now=.8)['accepted']
    assert recovered.preempt_execution(token,.9)['accepted']



def test_road_closure_during_pending_urgent_request_stays_closed_and_is_replanned_by_same_owner():
    sim=urgent_stopped_at_node(pickup='C')
    gate=sim.execution
    closed=sim.request_physical_detour('R',gate.recovery_token(),['B','C'],.65)
    assert closed['accepted'] and sim._physical_preemptions and sim._physical_detours
    assert not sim.replan_execution(gate.recovery_token(),.7)['accepted']
    # The urgent destination is A, so its reserved route can coexist with B-C closed.
    assert sim.preempt_execution(gate.recovery_token(),.7)['accepted']
    assert sim._closed_corridors=={('B','C')} and not sim._physical_detours
    assert not sim._physical_preemptions and sim._res.pins()=={'nodes':{'B':['R']},'edges':{}}
    assert sim.tasks[0].status is TaskStatus.PENDING and sim.tasks[1].status is TaskStatus.ASSIGNED
