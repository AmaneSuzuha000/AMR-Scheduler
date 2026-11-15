"""Derived front-to-press world, preserving frozen visual exports."""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .actuated_world import inertia,joint,box_visual
from .assembly_flow import ASSEMBLY_STAGES,PRESS,components
from .build import SCENE,e,save_xml
from .front_flow import TRAVEL_Z,transfer_points
from .front_world import add_body,build_front_world,planned_clearance,static_boxes,swept_hits


EXTRA_SNAPSHOTS={'wheelset_088','greasing_r23_wheelset_1','wheelset_093'}


def add_ram(world,name,side):
    sign=-1 if side=='left' else 1
    origin=(PRESS[0],PRESS[1]+sign*1.55,PRESS[2])
    model=e(world,'model',name=name);e(model,'pose',' '.join(map(str,origin))+' 0 0 0')
    body=e(model,'link',name='body');inertia(body,30,[3,3,3])
    joint(model,name+'_anchor','fixed','world','body',None)
    box_visual(body,'ram_mount','.26 .12 .26','.12 .3 .55 1')
    head=e(model,'link',name='probe');inertia(head,5,[.1,.1,.1])
    box_visual(head,'ram_head','.20 .20 .20','.95 .30 .05 1',f'0 {-sign*.12} 0 0 0 0')
    joint(model,name+'_probe','prismatic','body','probe',f'0 {-sign} 0',[0,.25,200,.5],4)
    plugin=e(model,'plugin',name=name+'_servo',filename='libwheelset_axis_servo.so')
    e(e(plugin,'ros'),'namespace','/'+name);e(plugin,'lease_s',1.)
    # The scanner PD can reach the 0.5m/s joint limit (with solver overshoot).
    # Use the existing position/velocity outer loop for the dedicated ram;
    # its <=0.25m remaining error requests <=0.2m/s, retaining the .5m/s audit.
    axis=e(plugin,'axis');e(axis,'joint',name+'_probe');e(axis,'mode','position')
    e(axis,'mass',5);e(axis,'limit',200);e(axis,'damping',4)
    return dict(origin=origin,kind='ram',axes=['probe'],side=side,direction_y=-sign)


def build_assembly_world(output):
    output=Path(output);meta=build_front_world(output)
    scene=json.loads(SCENE.read_text(encoding='utf-8'));derived=copy.deepcopy(scene)
    removed=set(meta['removed_static_snapshot_ids'])|EXTRA_SNAPSHOTS
    derived['instances']=[i for i in derived['instances'] if i['id'] not in removed]
    tree=ET.parse(output/'front.world');world=tree.getroot().find('world')
    for include in list(world.findall('include')):
        if include.findtext('name') in EXTRA_SNAPSHOTS:world.remove(include)
    # Wide fixed overview records the continuous same-workpiece run.
    world.find("model[@name='review_camera']/pose").text='75 -77 120 0 .75 1.57079632679'
    for name,item in components().items():
        meta['definitions'][name]=add_body(world,name,item['origin'],item['kind'])
    for side in ['left','right']:
        name='press_ram_'+side;meta['definitions'][name]=add_ram(world,name,side)
    boxes,sources=static_boxes(derived)
    meta.update(removed_static_snapshot_ids=sorted(removed),static_obstacles=boxes,
        collision_source_sha256=sources,stages=ASSEMBLY_STAGES,components=components())
    meta['assumptions'].update(output='Measured assembly outfeed; closing/run-in/storage still pending',
        preparation='Wiping/greasing reuse an assumed measured axial tool stroke, no surface chemistry/quality model',
        component_preparation='Finite identified bearings/back rings. Simulated identities and measured handling/gauge strokes; no real dimensional measurement',
        press='Two dedicated 0-.25m prismatic rams; .04m approach, .22m insertion endpoint, >=.25s stopped hold, retraction. 200N requested force limit; no contact/interference fit or product-quality claim',
        ram_control='Native position/velocity outer loop; target range bounds requested velocity <=.2m/s. Command-lease fault holds ram position; explicit fault recovery retracts before reissue',
        press_offset='Service center +1.2m X from visual cell; clears frozen feed/bridge collision boxes',
        replacement_parts='Independent XYZ axes track measured ram insertion, then remain fitted; no contact attachment',
        material_supply='Right back ring initially unreceived; measured overhead delivery releases missing-stock wait. Not an AGV delivery in this slice')
    save_xml(tree.getroot(),output/'front.world');return meta


def assembly_clearance(meta):
    result=planned_clearance(meta);checks=result['checks'];boxes=meta['static_obstacles']
    def check(stage,name,a,b,kind='part'):
        checks.append(dict(stage=stage,entity=name,start=a,end=b,hits=swept_hits(a,b,boxes,kind)))
    current=ASSEMBLY_STAGES[8]['target']
    for stage in ASSEMBLY_STAGES[9:]:
        points=[current,*transfer_points(current,stage)]
        for a,b in zip(points,points[1:]):
            check(stage['id'],'ws001',a,b,'wheelset')
            if stage['id']=='assembly_outfeed':
                for name,item in components().items():
                    check(stage['id'],name,[p+o for p,o in zip(a,item['offset'])],[p+o for p,o in zip(b,item['offset'])])
        current=stage['target']
    for name,item in components().items():
        r=item['receiving'];s=item['staged'];f=item['fitted']
        points=[item['origin'],r,(*r[:2],r[2]+.12),r,(*r[:2],TRAVEL_Z),(*s[:2],TRAVEL_Z),s,f]
        for a,b in zip(points,points[1:]):check('component_preparation',name,a,b)
    result['passed']=all(not c['hits'] for c in checks);return result
