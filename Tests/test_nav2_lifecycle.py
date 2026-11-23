"""Completed idle shutdown must retain stop proof and resource ownership."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import time

import pytest


from tools.live_bridge.terminal_docking import DockRegulator, DockSpec


class _Twist:
    """Body-frame twist stand-in; the adapter only ever writes x and z."""

    def __init__(self):
        self.linear=SimpleNamespace(x=0.,y=0.,z=0.)
        self.angular=SimpleNamespace(x=0.,y=0.,z=0.)


def adapter_type():
    """Load the actual adapter class without requiring ROS on the Windows host."""
    path=Path(__file__).resolve().parents[1]/'tools/live_bridge/nav2_scheduler.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SchedulerNode')
    scope={'RouteFollower':object,'time':time,'math':math,'DEMO_FRAME':None,
           'Twist':_Twist,'DockSpec':DockSpec,'DockRegulator':DockRegulator}
    exec(compile(ast.Module(body=[klass],type_ignores=[]),str(path),'exec'),scope)
    return scope['SchedulerNode']


def completed_adapter():
    klass=adapter_type()
    node=klass.__new__(klass)
    gate=SimpleNamespace(fault=None,bound=True,state='idle',command=None,
                         pose={'timestamp':10.},last_receipt=time.monotonic())
    node.sim=SimpleNamespace(execution=gate,_all_done=lambda:True)
    node.persistence_failed=False
    node.native_pending=node.guard_authorized=False
    node.goal_handle=node.active=node.waiting_move=node.guard_pending=None
    # __init__ always declares these; the helper mirrors the real object.
    node.dock_spec=node.dock=node.dock_candidate=node.dock_pub=node.dock_pending=None
    node.last_motion=None
    node.clock_s=10.
    node.suppress_feedback=False
    node.guard_fresh=lambda:True
    node.permit={'guard_id':'g','run_id':'run','route_id':'route','revision':2}
    node.guard_status=SimpleNamespace(**node.permit,state='latched',stopped=True,
        actual_speed=.001,actual_wz=.001,output_vx=0.,output_wz=0.)
    return node


def test_completed_idle_uses_fresh_matching_physical_stop_without_fault_acknowledgement():
    node=completed_adapter()
    assert node.clean_idle_shutdown_ready()
    assert node.sim.execution.state=='idle' and node.sim.execution.fault is None


def test_active_native_terminal_status_closes_permit_before_guard_command_timeout():
    node=completed_adapter()
    node.closed=False
    goal_id='ab'*16
    node.native_goal_id=goal_id
    node.reconciling=node.native_pending=False
    node.active={'route_id':'route'}
    node.goal_handle=object()
    node.guard_authorized=True
    stopped=[]; records=[]
    node.stop_guard=lambda:stopped.append('stop')
    node.record=lambda kind,**values:records.append((kind,values))
    status=SimpleNamespace(status=4,goal_info=SimpleNamespace(
        goal_id=SimpleNamespace(uuid=bytes.fromhex(goal_id))))
    node.on_native_status(SimpleNamespace(status_list=[status]))
    assert stopped==['stop']
    assert records==[('native_terminal_seen',{'goal_id':goal_id,'status':4})]
    assert node.active is not None and node.goal_handle is not None


def test_nonterminal_native_status_does_not_close_an_active_permit():
    node=completed_adapter()
    node.closed=False
    goal_id='cd'*16
    node.native_goal_id=goal_id
    node.reconciling=node.native_pending=False
    node.active={'route_id':'route'}
    node.guard_authorized=True
    stopped=[]
    node.stop_guard=lambda:stopped.append('stop')
    node.record=lambda *args,**kwargs:None
    status=SimpleNamespace(status=2,goal_info=SimpleNamespace(
        goal_id=SimpleNamespace(uuid=bytes.fromhex(goal_id))))
    node.on_native_status(SimpleNamespace(status_list=[status]))
    assert stopped==[]


@pytest.mark.parametrize('subscribers',[0,1])
def test_motion_preparation_checks_the_actual_remapped_speed_publisher(subscribers):
    node=completed_adapter()
    node.waiting_move={'route_id':'r2:1'}
    node.waiting_since=time.monotonic()
    node.guard_client=SimpleNamespace(service_is_ready=lambda:True)
    node._client=SimpleNamespace(server_is_ready=lambda:True)
    node.speed_pub=SimpleNamespace(get_subscription_count=lambda:subscribers)
    node.lease_pub=SimpleNamespace(get_subscription_count=lambda:1)
    calls=[]
    node.start_native=calls.append
    node.authorize_waiting_move()
    assert calls==([node.waiting_move] if subscribers else [])


def _guard_status(**changes):
    values=dict(guard_id='g',run_id='run',route_id='route',revision=2,
        state='armed',reason='',stopped=False,actual_speed=0.,actual_wz=0.,
        command_age=0.,lease_age=-1.,output_vx=0.,output_wz=0.,
        header=SimpleNamespace(stamp=SimpleNamespace(sec=10,nanosec=0)))
    values.update(changes)
    return SimpleNamespace(**values)


def test_fresh_exact_armed_status_starts_lease_before_service_reply():
    """The guard's fresh action identity acknowledges its asynchronous arm."""
    node=completed_adapter()
    node.closed=False
    node.guard_id='g'
    node.guard_pending=object()
    node.guard_authorized=False
    node.persistence_failed=False
    node.clock_s=10.
    node.permit=dict(guard_id='g',run_id='run',route_id='route',revision=2)
    node.sim.execution.command=dict(run_id='run',route_id='route',revision=2)
    node.guard_fresh=node.__class__.guard_fresh.__get__(node,node.__class__)
    class _Lease:
        def __init__(self):
            self.header=SimpleNamespace(stamp=None)
    class _Time:
        def __init__(self,seconds):
            self.seconds=seconds
        def to_msg(self):
            return self.seconds
    node.on_guard.__func__.__globals__.update(MotionLease=_Lease,
        rclpy=SimpleNamespace(time=SimpleNamespace(Time=_Time)))
    node.lease_seq=0
    node.published=[]
    node.lease_pub=SimpleNamespace(publish=node.published.append,
        get_subscription_count=lambda:1)
    node.record=lambda *args,**kwargs:None
    node.on_guard(_guard_status(),received_at=time.monotonic())
    assert node.guard_authorized
    assert len(node.published)==1
    lease=node.published[0]
    assert (lease.guard_id,lease.run_id,lease.route_id,lease.revision,lease.seq)==('g','run','route',2,1)


@pytest.mark.parametrize('changes',[
    {'guard_id':'other'}, {'run_id':'other'}, {'route_id':'other'},
    {'revision':3}, {'state':'latched'},
])
def test_pending_arm_status_must_match_the_exact_guard_action(changes):
    node=completed_adapter()
    node.closed=False
    node.guard_id='g'
    node.guard_pending=object()
    node.guard_authorized=False
    node.persistence_failed=False
    node.permit=dict(guard_id='g',run_id='run',route_id='route',revision=2)
    node.sim.execution.command=dict(run_id='run',route_id='route',revision=2)
    node.sim.execution.fail=lambda reason:setattr(node.sim.execution,'fault',reason)
    node.guard_fresh=lambda:True
    node.published=[]
    node.publish_lease=lambda:node.published.append('lease')
    node.record=lambda *args,**kwargs:None
    node.on_guard(_guard_status(**changes),received_at=time.monotonic())
    assert not node.guard_authorized
    assert node.published==[]


@pytest.mark.parametrize('case',['stale','fault','persistence_failed','changed_command'])
def test_pending_arm_status_cannot_authorize_stale_or_invalid_owner_state(case):
    node=completed_adapter()
    node.closed=False
    node.guard_id='g'
    node.guard_pending=object()
    node.guard_authorized=False
    node.persistence_failed=False
    node.permit=dict(guard_id='g',run_id='run',route_id='route',revision=2)
    node.sim.execution.command=dict(run_id='run',route_id='route',revision=2)
    node.guard_fresh=lambda:case!='stale'
    if case=='fault':
        node.sim.execution.fault='latched execution fault'
    elif case=='persistence_failed':
        node.persistence_failed=True
    elif case=='changed_command':
        node.sim.execution.command['route_id']='new-route'
    node.published=[]
    node.publish_lease=lambda:node.published.append('lease')
    node.record=lambda *args,**kwargs:None
    node.on_guard(_guard_status(),received_at=time.monotonic())
    assert not node.guard_authorized
    assert node.published==[]


def test_delayed_success_reply_does_not_rewind_lease_sequence():
    from concurrent.futures import Future
    node=completed_adapter()
    node.closed=False
    command=dict(run_id='run',route_id='route',revision=2)
    node.guard_pending=object()
    node.guard_authorized=False
    node.persistence_failed=False
    node.sim.execution.command=command
    node.active=command
    node.goal_handle=object()
    node.waiting_move=command
    node.lease_seq=7
    node.record=lambda *args,**kwargs:None
    published=[]
    node.publish_lease=lambda:published.append(node.lease_seq)
    future=Future()
    future.set_result(SimpleNamespace(accepted=True,reason='accepted'))
    node.on_authorized(future,command)
    assert node.guard_authorized and node.lease_seq==7
    assert published==[7]


def test_display_marks_last_known_coordinates_unavailable_as_source_feedback_expires():
    node=completed_adapter()
    assert node.feedback_available()
    node.sim.execution.last_receipt=time.monotonic()-1.
    assert not node.feedback_available()
    node.sim.execution.last_receipt=time.monotonic()
    node.sim.execution.pose={'timestamp':9.}
    assert not node.feedback_available()
    node.sim.execution.pose={'timestamp':10.}
    node.suppress_feedback=True
    assert not node.feedback_available()


@pytest.mark.parametrize('change',[
    lambda n:setattr(n,'native_pending',True),
    lambda n:setattr(n,'guard_authorized',True),
    lambda n:setattr(n.sim.execution,'command',{'route_id':'unfinished'}),
    lambda n:setattr(n.sim,'_all_done',lambda:False),
    lambda n:setattr(n,'guard_fresh',lambda:False),
    lambda n:setattr(n.guard_status,'route_id','obsolete'),
    lambda n:setattr(n.guard_status,'actual_speed',.01),
    lambda n:setattr(n.guard_status,'actual_wz',.02),
    lambda n:setattr(n.guard_status,'output_vx',.01),
    lambda n:setattr(n.sim.execution,'last_receipt',time.monotonic()-1),
    lambda n:setattr(n.sim.execution,'pose',{'timestamp':9.}),
    lambda n:setattr(n.sim.execution,'fault','retained fault'),
])
def test_shutdown_keeps_stop_and_fault_handling_for_unfinished_stale_or_moving_state(change):
    node=completed_adapter()
    change(node)
    assert not node.clean_idle_shutdown_ready()


@pytest.mark.parametrize('reason,retained',[
    ('duplicate source timestamp',True),('identity mismatch',False),('invalid timestamp',False)])
def test_repeated_sensor_stamp_is_no_new_evidence_but_does_not_reset_stop_window(reason,retained):
    node=completed_adapter()
    node.sim.execution.fault='cancelled'
    node.suppress_feedback=False
    node.coordinate_frame=SimpleNamespace(to_network=lambda x,y:(x,y),to_network_yaw=lambda yaw:yaw)
    original=(time.monotonic(),10.)
    node.stop_since=original
    node.emit=lambda *a,**kw:dict(accepted=False,reason=reason)
    node.process_observation(1.,2.,0.,0.,0.,0.,10.,{})
    assert node.stop_since==(original if retained else None)


@pytest.mark.parametrize('source,transform_time',[('localized',0.),('gazebo',43.5)])
@pytest.mark.parametrize('resumes_route_id',[None,'old:2'])
def test_static_map_route_uses_live_tf_after_localization_correction_without_restamping_feedback(source,transform_time,resumes_route_id):
    """Recorded recovery: an old map->odom correction must not move the goal."""
    from concurrent.futures import Future
    import uuid

    node=completed_adapter()
    node.sim.execution.identity={'source':source}
    node.clock_s=43.5
    node.sim.execution.pose.update(x=1.48,y=.38)
    observation=dict(node.sim.execution.pose)
    node.coordinate_frame=SimpleNamespace(to_world=lambda x,y:(x,y))
    node.sim.network=object()
    poses=[SimpleNamespace(header=SimpleNamespace(stamp=None),
            pose=SimpleNamespace(position=SimpleNamespace(x=x,y=y))) for x,y in [(1.48,.38),(1.4,1.4)]]
    path=SimpleNamespace(header=SimpleNamespace(frame_id='map',stamp=None),poses=poses)
    built=[]
    node.build_path=lambda *args,**kwargs:built.append((args,kwargs)) or path
    node.set_speed=lambda command:None
    node.persist=lambda:True
    node.record=lambda *args,**kwargs:None
    sent=[]
    node._client=SimpleNamespace(send_goal_async=lambda goal,**kwargs:sent.append(goal) or Future())
    node.start_native.__globals__.update(FollowPath=SimpleNamespace(Goal=SimpleNamespace),
        rclpy=SimpleNamespace(time=SimpleNamespace(Time=lambda seconds:SimpleNamespace(to_msg=lambda:seconds))),
        uuid=uuid,UUID=lambda **kwargs:kwargs)
    node.start_native(dict(route_id='current:3',nodes=['B','C'],resumes_route_id=resumes_route_id))
    assert built[0][0][0]==(['C'] if resumes_route_id else ['B','C'])
    assert built[0][1]['prepend_pose']==((1.48,.38) if resumes_route_id else None)
    assert sent[0].path.header.stamp==transform_time
    assert all(p.header.stamp==transform_time for p in sent[0].path.poses)
    # The original observation keeps its source time and coordinates. A static
    # route's TF policy must never disguise an old sensor sample as current.
    assert node.sim.execution.pose==observation
    assert sent[0].path.poses[-1].pose.position.x==1.4
    assert sent[0].path.poses[-1].pose.position.y==1.4
    assert node.native_pending and not node.guard_authorized


def docking_adapter():
    """A retained station hop already inside its declared capture window."""
    node=completed_adapter()
    node.closed=False
    node.record=lambda kind,**data: node.events.append(dict(kind=kind,**data))
    node.events=[]; node.cancels=[]; node.stops=[]; node.faults=[]
    node.published=[]
    node.dock_spec=DockSpec(capture_radius_m=.35,position_tolerance_m=.015,yaw_tolerance_rad=.04)
    node.dock_candidate=dict(route_id='route',target_xy=[1.,0.],target_yaw=0.)
    node.dock=None
    node.dock_pub=SimpleNamespace(publish=lambda message: node.published.append((message.linear.x,message.angular.z)))
    node.active={'route_id':'route'}
    node.last_motion=None
    node.guard_authorized=True
    node.persistence_failed=False
    node.persist=lambda: True
    node.stop_guard=lambda: node.stops.append(1)
    node.goal_handle=SimpleNamespace(cancel_goal_async=lambda: node.cancels.append(1))
    node.coordinate_frame=SimpleNamespace(to_network=lambda x,y:(x,y),to_network_yaw=lambda yaw:yaw)
    node.sim.execution.pose=dict(x=.7,y=0.,yaw=0.,timestamp=10.)
    node.sim.execution.fail=lambda reason: node.faults.append(reason)
    return node


def test_terminal_takeover_is_absent_unless_the_owner_declares_it():
    """An undeclared take-over must not publish, cancel or touch the guard."""
    node=docking_adapter(); node.dock_spec=None
    node.dock_step(1.)
    assert node.dock is None and node.cancels==[] and node.stops==[]


def test_terminal_takeover_waits_until_the_declared_capture_window():
    """Outside the window the transit controller keeps the terminal."""
    node=docking_adapter()
    node.sim.execution.pose=dict(x=.5,y=0.,yaw=0.,timestamp=10.)
    node.dock_step(1.)
    assert node.dock is None and node.cancels==[] and node.published==[]
    node.sim.execution.pose=dict(x=.7,y=0.,yaw=0.,timestamp=10.)
    node.dock_step(1.)
    assert node.dock is not None and node.cancels==[1]
    assert node.stops==[] and node.sim.execution.fault is None
    assert [row for row in node.events if row['kind']=='dock_takeover'][0]['distance']==pytest.approx(.3)


def test_takeover_holds_the_wheels_until_nav2_stops_publishing():
    """The guard's bounded park allowance covers the cancel handshake."""
    node=docking_adapter()
    node.dock_step(1.)
    node.dock_step(1.05)
    assert node.published==[(0.,0.)]
    assert [row['phase'] for row in node.events if row['kind']=='dock_cmd']==['handover']


def test_cancelled_action_during_takeover_is_not_an_execution_failure():
    """The owner cancelled this action on purpose; the permit must survive."""
    from concurrent.futures import Future
    node=docking_adapter()
    node.dock_step(1.)
    future=Future(); future.set_result(SimpleNamespace(status=5))
    node.on_result(future,{'route_id':'route'})
    assert node.goal_handle is None and node.dock is not None
    assert node.active=={'route_id':'route'} and node.guard_authorized and node.stops==[]
    assert node.sim.execution.fault is None
    assert [row['kind'] for row in node.events if row['kind']=='dock_native_cancel']==['dock_native_cancel']


def test_native_terminal_status_during_takeover_keeps_the_permit():
    """Nav2 reports status 5 before the result callback; the take-over still owns the wheels."""
    node=docking_adapter()
    node.native_goal_id='ab'*16
    node.reconciling=False
    node.dock_step(1.)
    node.on_native_status(SimpleNamespace(status_list=[SimpleNamespace(
        goal_info=SimpleNamespace(goal_id=SimpleNamespace(uuid=bytes.fromhex('ab'*16))),status=5)]))
    assert node.guard_authorized and node.stops==[]
    assert not [row for row in node.events if row['kind']=='native_terminal_seen']


def test_takeover_refuses_a_terminal_that_disagrees_with_the_retained_target():
    """Two different destinations must not be silently averaged."""
    node=docking_adapter()
    poses=[SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=1.,y=1.))),
           SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=2.,y=2.)))]
    assert node.dock_target({'route_id':'route','target_xy':[1.,1.]},
                            SimpleNamespace(poses=poses)) is None
    assert [row['kind'] for row in node.events if row['kind']=='dock_target_mismatch']==['dock_target_mismatch']
    poses[1].pose.position.x, poses[1].pose.position.y = 1., 1.
    assert node.dock_target({'route_id':'route','target_xy':[1.,1.]},
                            SimpleNamespace(poses=poses))==dict(route_id='route',target_xy=[1.,1.],target_yaw=0.)


def test_staged_target_uses_declared_load_heading_and_side_without_guessing():
    node=docking_adapter()
    node.dock_spec=DockSpec(.35,.005,.04,steering_mode='staged',
        approach_stations=({'target_xy':[1.,1.],'yaw_rad':-math.pi/2,'approach_side':1},))
    poses=[SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=1.,y=0.))),
           SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=1.,y=1.)))]
    row=node.dock_target({'route_id':'route','target_xy':[1.,1.]},SimpleNamespace(poses=poses))
    # Path arrival bearing is +pi/2; this actual LOAD rack needs -pi/2 and
    # reverse insertion from the front side of the target.
    assert row['target_yaw']==-math.pi/2 and row['approach_side']==1
    node.dock_spec=DockSpec(.35,.005,.04,steering_mode='staged')
    assert node.dock_target({'route_id':'route','target_xy':[1.,1.]},
                            SimpleNamespace(poses=poses)) is None
    assert node.events[-1]['kind']=='dock_approach_undefined'


def test_completed_dock_returns_the_unchanged_arrival_status():
    """The take-over ends in the same stop proof the transit completion used."""
    node=docking_adapter()
    node.dock_step(1.)
    node.goal_handle=None
    node.last_motion=dict(speed=0.,wz=0.,timestamp=10.)
    node.sim.execution.pose=dict(x=1.,y=0.,yaw=0.,timestamp=10.)
    node.dock_step(1.1)
    node.sim.execution.pose=dict(x=1.,y=0.,yaw=0.,timestamp=10.)
    node.dock_step(1.45)
    assert node.dock is None and node.dock_candidate is None and node.active is None
    assert node.result_status==4 and node.stops==[1]
    assert [row['kind'] for row in node.events if row['kind']=='dock_aligned']==['dock_aligned']
    node.result_status=None  # the unchanged arrival gate accepted the event
    node.dock_step(1.5)
    assert node.dock_pending is None and node.sim.execution.fault is None


def test_missing_stop_proof_after_handover_is_a_bounded_fault_not_a_hang():
    node=docking_adapter()
    node.dock_step(1.)
    node.goal_handle=None
    node.last_motion=dict(speed=0.,wz=0.,timestamp=10.)
    node.sim.execution.pose=dict(x=1.,y=0.,yaw=0.,timestamp=10.)
    node.dock_step(1.1)
    node.sim.execution.pose=dict(x=1.,y=0.,yaw=0.,timestamp=10.)
    node.dock_step(1.45)
    assert node.result_status==4 and node.dock_pending is not None
    node.dock_step(12.)
    assert node.dock_pending is None
    assert node.faults==['terminal docking stop proof did not follow']


def test_takeover_deadline_latches_a_fault_instead_of_commanding_forever():
    node=docking_adapter()
    node.dock_step(1.)
    node.goal_handle=None
    node.dock['started']=1.-46.
    node.dock_step(1.)
    assert node.dock is None and node.dock_candidate is None
    assert node.faults==['terminal docking did not converge'] and node.stops==[1]


def test_takeover_waits_instead_of_regulating_against_an_invented_speed():
    """A missing motion sample is not evidence that the vehicle is standing still."""
    node=docking_adapter()
    node.dock_step(1.)
    node.goal_handle=None
    node.dock_step(1.1)
    assert node.last_motion is None
    assert node.published[-1]==(0.,0.)
    assert node.dock is not None and node.sim.execution.fault is None


def test_station_terminal_selects_the_docking_goal_checker():
    """A terminal at a station needs the handoff tolerance; a junction does not."""
    node=completed_adapter()
    kinds={'receiving':'dock','bay_west':'buffer','west':'aisle'}
    node.sim.network=SimpleNamespace(node=lambda node_id:SimpleNamespace(kind=kinds[node_id]))
    assert node.terminal_goal_checker({'nodes':['east','receiving']})=='dock_goal_checker'
    assert node.terminal_goal_checker({'nodes':['west','bay_west']})=='dock_goal_checker'
    assert node.terminal_goal_checker({'nodes':['west','east']})=='general_goal_checker'


def test_undeclared_terminal_keeps_the_transit_goal_checker():
    """Missing geometry must not invent a tighter requirement."""
    node=completed_adapter()
    def missing(node_id):
        raise KeyError(node_id)
    node.sim.network=SimpleNamespace(node=missing)
    assert node.terminal_goal_checker({'nodes':['unknown']})=='general_goal_checker'
    node.sim.network=object()
    assert node.terminal_goal_checker({'nodes':['anything']})=='general_goal_checker'
    assert node.terminal_goal_checker({})=='general_goal_checker'


def test_station_terminal_sends_the_docking_checker_with_the_action():
    """The selected checker must reach the FollowPath goal, not just a log line."""
    from concurrent.futures import Future
    import uuid

    node=completed_adapter()
    node.sim.execution.identity={'source':'localized'}
    node.clock_s=10.
    node.sim.execution.pose.update(x=1.,y=1.)
    node.coordinate_frame=SimpleNamespace(to_world=lambda x,y:(x,y))
    node.sim.network=SimpleNamespace(node=lambda node_id:SimpleNamespace(kind='dock'))
    poses=[SimpleNamespace(header=SimpleNamespace(stamp=None),
            pose=SimpleNamespace(position=SimpleNamespace(x=x,y=y))) for x,y in [(1.,1.),(2.,2.)]]
    path=SimpleNamespace(header=SimpleNamespace(frame_id='map',stamp=None),poses=poses)
    node.build_path=lambda *args,**kwargs:path
    node.set_speed=lambda command:None
    node.persist=lambda:True
    recorded=[]
    node.record=lambda kind,**values:recorded.append((kind,values))
    sent=[]
    node._client=SimpleNamespace(send_goal_async=lambda goal,**kwargs:sent.append(goal) or Future())
    node.start_native.__globals__.update(FollowPath=SimpleNamespace(Goal=SimpleNamespace),
        rclpy=SimpleNamespace(time=SimpleNamespace(Time=lambda seconds:SimpleNamespace(to_msg=lambda:seconds))),
        uuid=uuid,UUID=lambda **kwargs:kwargs)
    node.start_native(dict(route_id='current:3',nodes=['west','receiving']))
    assert sent[0].controller_id=='FollowPath'
    assert sent[0].goal_checker_id=='dock_goal_checker'
    assert dict(recorded)['path']['goal_checker']=='dock_goal_checker'
def guard_probe(**values):
    klass = adapter_type()
    node = klass.__new__(klass)
    node.events = []
    node.record = lambda kind, **payload: node.events.append(dict(kind=kind, **payload))
    node.guard_status = SimpleNamespace(header=SimpleNamespace(
        stamp=SimpleNamespace(sec=int(values.get('stamp', 10.)),
                              nanosec=round((values.get('stamp', 10.) % 1)*1e9))))
    node.guard_received_at = time.monotonic()-values.get('wall_age', .01)
    node.clock_s = values.get('clock', 10.)
    return node


def test_guard_fresh_records_which_condition_rejected_feedback():
    fresh = guard_probe(stamp=10., clock=10., wall_age=.01)
    assert fresh.guard_fresh() is True and fresh.events == []

    late = guard_probe(stamp=10., clock=10., wall_age=.4)
    assert late.guard_fresh() is False
    assert late.events[-1]['clause'] == 'wall_age'
    assert .39 < late.events[-1]['wall_age_s'] < .41

    skewed = guard_probe(stamp=10., clock=11., wall_age=.01)
    assert skewed.guard_fresh() is False
    assert skewed.events[-1]['clause'] == 'clock_delta'
    assert skewed.events[-1]['clock_minus_stamp_s'] == 1.

    missing = guard_probe()
    missing.guard_status = None
    assert missing.guard_fresh() is False
    assert missing.events[-1]['clause'] == 'missing_sample'

    # The recorder is rate limited to 5 Hz per node.
    count = len(late.events)
    late._last_guard_freshness_rejected = time.monotonic()
    late.guard_fresh()
    assert len(late.events) == count
