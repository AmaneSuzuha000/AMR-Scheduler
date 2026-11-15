"""Assumed high-level XYZ supply shuttle with measured joint completion."""
from dataclasses import dataclass
import math
from .actuated_world import inertia,joint,box_visual
from .build import e


SOURCES={'inspection_consumables':(60.,69.2,.7),'small_tools':(6.,69.2,.7)}
DESTINATIONS={'magnetic':(35.45,50.,1.0),'derust':(21.,50.,1.0)}


def add_bridge(world):
    model=e(world,'model',name='assumed_supply_bridge');e(model,'pose','0 0 0 0 0 0')
    base=e(model,'link',name='base');inertia(base,5,[1,1,1]);joint(model,'supply_anchor','fixed','world','base',None)
    parent='base'
    for axis,vector,mass in [('x','1 0 0',10),('y','0 1 0',5),('z','0 0 1',2)]:
        link=e(model,'link',name=axis);inertia(link,mass,[.3,.3,.3])
        joint(model,'supply_'+axis,'prismatic',parent,axis,vector,[0,80,400,2.0],3)
        parent=axis
    box_visual(link,'assumed_material_tote','.38 .28 .20','1 .65 .05 1')
    plugin=e(model,'plugin',name='supply_feedback',filename='libgazebo_ros_joint_state_publisher.so')
    e(e(plugin,'ros'),'namespace','/supply_bridge');e(plugin,'update_rate',40)
    for axis in 'xyz':e(plugin,'joint_name','supply_'+axis)
    return dict(design='Assumed XYZ overhead shuttle, explicitly added beyond video; constrained effort-driven axes',
        gravity=False,contact=False,stock_commit='Measured final position and stopped joints after dock pickup/lift/traverse/lower',
        travel_height_m=3.,masses_kg=[10,5,2],not_claimed='Recovered factory mechanism, contact gripping, actual material transfer or industrial safety')


@dataclass(frozen=True)
class BridgeSample:
    time_s:float
    xyz:tuple
    velocity:tuple


class BridgeAction:
    def __init__(self,command,sample):
        self.command=command;self.events=[];self.index=0;self.started=sample.time_s;self.phase_started=sample.time_s
        self.complete=False;self.fault=None;self.last=-math.inf
        source=SOURCES[command['task_id']];destination=DESTINATIONS[command['station']]
        self.waypoints=[(sample.xyz[0],sample.xyz[1],3.),(source[0],source[1],3.),source,
            (source[0],source[1],3.),(destination[0],destination[1],3.),destination]

    @property
    def target(self):return self.waypoints[min(self.index,len(self.waypoints)-1)]

    def update(self,sample):
        if sample.time_s<=self.last or self.complete or self.fault:return
        self.last=sample.time_s
        if sample.time_s-self.phase_started>75:
            self.fault='bridge_waypoint_timeout';return
        if math.dist(sample.xyz,self.target)<.025 and max(abs(v) for v in sample.velocity)<.035:
            self.events.append(dict(token=self.command['token'],waypoint=self.index,time_s=sample.time_s,
                xyz=list(sample.xyz),velocity=list(sample.velocity),target=list(self.target)))
            self.index+=1;self.phase_started=sample.time_s
            self.complete=self.index==len(self.waypoints)


def bridge_efforts(sample,target):
    # Force-limited velocity servo, with 17/7/2kg total descendant axis mass.
    result={}
    for axis,p,v,t,mass in zip('xyz',sample.xyz,sample.velocity,target,[17,7,2]):
        desired=max(-1.5,min(1.5,2*(t-p)))
        result['supply_'+axis]=max(-400.,min(400.,mass*6*(desired-v)+3*v))
    return result
