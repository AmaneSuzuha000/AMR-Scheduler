"""Scene bends must survive dispatch, native sampling, stop recovery and display."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import sys
import time

import pytest

from amr_sched.checkpoint import definition, capture, restore, _encode, _decode
from amr_sched.geometry import project_polyline, remaining_polyline, point_along_polyline
from amr_sched.network import Edge, Network, Node
from amr_sched.workshop import WorkshopScene
from amr_sched.stream import NetworkSpec
from test_external_execution import build, report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/live_bridge'))
from gz_frame import GzFrame
from nav2_path import sample_polyline, shift_tracking_reference


def test_display_network_publishes_every_directed_scene_centerline():
    net = WorkshopScene.load().network()
    wire = NetworkSpec.from_network(net)
    assert len(wire.edges) == 144
    for road in wire.edges:
        assert road['centerline'] == [list(p) for p in net.edge_centerline(road['src'],road['dst'])]
    legacy = Network()
    legacy.add_node(Node('A',0.,0.)); legacy.add_node(Node('B',1.,0.))
    legacy.add_bidirectional('A','B',1.)
    assert NetworkSpec.from_network(legacy).edges[1]['centerline'] == [[1.,0.],[0.,0.]]


def bent_sim():
    sim = build()
    net = Network()
    for node in sim.network.nodes.values():
        net.add_node(node)
    net.add_bidirectional('A','B',14.,centerline=((0.,0.),(0.,2.),(10.,2.),(10.,0.)))
    net.add_bidirectional('B','C',10.)
    sim.network = net
    sim.reset()
    assert report(sim)['accepted']
    sim.step_once()
    return sim


def test_canonical_scene_compiles_all_directed_vertices_and_keeps_declared_costs():
    scene = WorkshopScene.load()
    net = scene.network()
    bent = 0
    for lane in scene.lanes:
        assert net.edge_centerline(lane.src,lane.dst) == lane.centerline
        assert net.edge(lane.src,lane.dst).length == lane.polyline_length
        if lane.two_way:
            assert net.edge_centerline(lane.dst,lane.src) == tuple(reversed(lane.centerline))
        if len(lane.centerline)>2:
            bent += 1
    assert len(net.edges)==144 and bent==12
    assert net.edge_centerline('W_WHEELSET','R_WHEELSET') == (
        (65.,80.),(66.6,80.),(66.6,63.),(65.,63.))
    assert net.edge('W_WHEELSET','R_WHEELSET').length == pytest.approx(20.2)


@pytest.mark.parametrize('points,length',[
    (((0.,0.),(math.nan,1.),(1.,0.)),2.),
    (((0.,0.),(0.,0.),(1.,0.)),1.),
    (((0.,0.),(0.,1.),(1.,0.)),1.),
    (((0.,0.),(0.,1.,2.),(1.,0.)),2.),
])
def test_invalid_centerlines_fail_before_a_road_is_added(points,length):
    with pytest.raises(ValueError,match='centerline'):
        Edge('A','B',length,centerline=points)


def test_endpoint_mismatch_and_undeclared_hops_are_rejected():
    net = Network()
    net.add_node(Node('A',0.,0.)); net.add_node(Node('B',1.,0.))
    with pytest.raises(ValueError,match='endpoints'):
        net.add_edge(Edge('A','B',1.,centerline=((0.,1.),(1.,1.))))
    assert not net.edges
    with pytest.raises(KeyError):
        net.route_centerline(['A','B'])


@pytest.mark.parametrize('xy,expected',[
    ((.05,1.),((.05,1.),(0.,1.),(0.,2.),(10.,2.),(10.,0.))),
    ((5.,2.04),((5.,2.04),(5.,2.),(10.,2.),(10.,0.))),
    ((10.,1.),((10.,1.),(10.,0.))),
])
def test_retained_hop_recovery_keeps_future_corners_and_resources(xy,expected):
    sim = bent_sim()
    gate = sim.execution
    command = deepcopy(gate.command)
    assert command['path_xy'] == [[0.,0.],[0.,2.],[10.,2.],[10.,0.]]
    gate.drain_commands()
    assert report(sim,xy=xy,now=.1)['accepted']
    pins = deepcopy(sim._res.pins())
    gate.fail('injected stop')
    gate.drain_commands()
    assert report(sim,'STOPPED',xy=xy,now=.2)['accepted']
    answer = sim.resume_execution(command['route_id'],.2)
    assert answer['accepted']
    assert answer['command']['resume_path_xy'] == [list(p) for p in expected]
    assert answer['command']['path_xy'] == command['path_xy']
    assert answer['command']['route_id'] != command['route_id']
    assert gate.drain_commands()[0] == answer['command']
    assert sim._res.pins()==pins and sim.agvs[0].cur_task=='T'


def test_endpoint_chord_is_outside_bent_road_and_recovery_stays_locked():
    sim = bent_sim(); gate=sim.execution
    gate.drain_commands(); gate.fail('injected stop'); gate.drain_commands()
    assert report(sim,'STOPPED',xy=(5.,0.),now=.2)['accepted']
    pins=deepcopy(sim._res.pins())
    answer=sim.resume_execution(gate.recovery_token(),.2)
    assert not answer['accepted'] and 'outside retained corridor' in answer['reason']
    assert sim._res.pins()==pins and gate.fault and not gate.drain_commands()


def extract_method(path,klass,method):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==klass)
    return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==method)


def test_native_path_preserves_actual_scene_corners_in_rotated_world_metres():
    def message():
        return SimpleNamespace(header=SimpleNamespace(),poses=[])
    def pose():
        return SimpleNamespace(header=SimpleNamespace(),pose=SimpleNamespace(
            position=SimpleNamespace(),orientation=SimpleNamespace()))
    method=extract_method(ROOT/'tools/live_bridge/nav2_route_follower.py','RouteFollower','build_path')
    scope=dict(Path=message,PoseStamped=pose,sample_polyline=sample_polyline,
               shift_tracking_reference=shift_tracking_reference,math=math)
    exec(compile(ast.Module(body=[method],type_ignores=[]),'native_path_contract','exec'),scope)
    frame=GzFrame(scale=.5,rotation=.7,origin=(2.,3.))
    node=SimpleNamespace(frame_id='map',coordinate_frame=frame,path_spacing_m=.05,tracking_offset_m=0.,
        get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(to_msg=lambda:0.)))
    net=WorkshopScene.load().network(); ids=['W_WHEELSET','R_WHEELSET']
    path=scope['build_path'](node,ids,net)
    xy=[(p.pose.position.x,p.pose.position.y) for p in path.poses]
    for point in net.edge_centerline(*ids):
        assert frame.to_world(*point) in xy
    assert sum(math.dist(a,b) for a,b in zip(xy,xy[1:]))==pytest.approx(10.1)
    assert max(math.dist(a,b) for a,b in zip(xy,xy[1:]))<=.05+1e-12


def test_native_sensor_guard_accepts_bend_and_stops_on_shortcut_without_releasing_pins():
    method=extract_method(ROOT/'tools/live_bridge/nav2_scheduler.py','SchedulerNode','emit')
    scope=dict(time=time)
    exec(compile(ast.Module(body=[method],type_ignores=[]),'native_guard_contract','exec'),scope)
    sim=bent_sim(); sim.execution.last_receipt=time.monotonic()
    node=SimpleNamespace(sim=sim,seq=sim.execution.last_seq,record=lambda *a,**k:None)
    stops=[]; node.stop_guard=lambda:stops.append(True)
    pins=deepcopy(sim._res.pins())
    assert scope['emit'](node,'POSE',[5.,2.,0.],sim.execution.envelope(),.1)['accepted']
    assert not sim.execution.fault and not stops
    assert scope['emit'](node,'POSE',[5.,0.,0.],sim.execution.envelope(),.2)['accepted']
    assert sim.execution.fault and stops==[True] and sim._res.pins()==pins


def test_checkpoint_binds_geometry_even_when_topology_and_travel_cost_are_equal():
    sim=bent_sim(); fingerprint=definition(sim)
    edge=sim.network.edge('A','B')
    assert _decode(_encode(edge))==edge
    snapshot=capture(sim,fingerprint,{})
    other=bent_sim()
    net=Network()
    for node in other.network.nodes.values():net.add_node(node)
    net.add_bidirectional('A','B',14.,centerline=((0.,0.),(0.,-2.),(10.,-2.),(10.,0.)))
    net.add_bidirectional('B','C',10.)
    other.network=net
    assert definition(other)!=fingerprint
    with pytest.raises(ValueError,match='definition mismatch'):
        restore(other,snapshot,definition(other))


def test_algorithm_display_interpolates_arc_length_without_changing_physical_pose():
    from amr_sched.stream import FrameBuilder
    sim=bent_sim(); agv=sim.agvs[0]
    builder=FrameBuilder(sim)
    sim.execution.pose.update(x=1.,y=2.)
    assert builder._xy(agv,1)==[1.,2.]
    sim.executions={}; sim.execution=None
    src,dst,depart,arrive=sim._transit[agv.agv_id]
    xy=builder._xy(agv,(depart+arrive)/2)
    assert xy==pytest.approx([5.,2.])
    assert point_along_polyline(((0.,0.),(0.,2.),(10.,2.),(10.,0.)),.5)==(5.,2.)


def test_operator_closure_before_task_release_is_owned_durable_and_versioned():
    sim=build()
    before=deepcopy(sim._res.pins())
    answer=sim.close_physical_corridor(['A','B'],0.)
    assert answer['accepted'] and answer['stopped_robots']==[]
    assert not sim._routing_network().has_edge('A','B')
    assert not sim._routing_network().has_edge('B','A')
    assert sim.network.has_edge('A','B') and sim._res.pins()==before
    assert not sim.close_physical_corridor(['B','A'],0.)['accepted']
    assert not sim.reopen_physical_corridor(['A','B'],0)['accepted']
    assert sim.reopen_physical_corridor(['A','B'],answer['closure_revision'])['accepted']
    assert sim._routing_network().has_edge('A','B') and not sim.execution.bound


def test_operator_closure_stops_affected_action_and_preserves_task_and_holds():
    sim=bent_sim(); gate=sim.execution
    pins=deepcopy(sim._res.pins()); command=deepcopy(gate.command)
    answer=sim.close_physical_corridor(['A','B'],.1)
    assert answer['accepted'] and answer['stopped_robots']==['R']
    assert gate.fault and gate.command==command and sim._res.pins()==pins
    assert [c['kind'] for c in gate.drain_commands()]==['CANCEL']
    assert not sim.resume_execution(gate.recovery_token(),.1)['accepted']


def test_robot_report_keeps_its_identity_in_the_shared_closure_event():
    sim=bent_sim(); gate=sim.execution
    answer=sim.request_physical_detour('R',gate.recovery_token(),['A','B'],.1)
    assert answer['accepted'] and answer['robot_id']=='R'
    assert sim.deviation_events[-1]['robot_id']=='R'


@pytest.mark.parametrize('xy,locked',[((5.,2.34),False),((5.,2.36),True),((5.,0.),True)])
def test_restarted_sensor_guard_uses_the_same_bent_road(xy,locked):
    path=ROOT/'out/archive/fleet_execution/fleet_worker.py'
    if not path.exists():
        pytest.skip("Optional archived ROS 2 script not present in checkout")
    tree=ast.parse(path.read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='enforce_hop_bounds')
    scope=dict(math=math,deepcopy=deepcopy)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
    sim=bent_sim();sim.execution.pose.update(x=xy[0],y=xy[1]);actions=[]
    runtime=SimpleNamespace(sim=sim,nodes={'R':SimpleNamespace(stop_guard=lambda:actions.append('stop'))},
        record=lambda *a,**k:actions.append('record'))
    scope['enforce_hop_bounds'](runtime,{'A':(0.,0.),'B':(10.,0.)},.35)
    assert bool(sim.execution.fault)==locked
    assert actions==(['stop','record'] if locked else [])
