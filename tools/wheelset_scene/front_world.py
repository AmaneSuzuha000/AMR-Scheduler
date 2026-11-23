"""Derived XYZ wheelset/part test rig; original visual assets stay frozen."""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .actuated_world import inertia,joint,box_visual
from .assets import wheelset
from .build import ROOT,SCENE,e,make_world,save_xml
from .front_flow import ORIGIN,STAGES,TRAVEL_Z,parts,transfer_points
from .light_material_network import transform,pose


def envelopes(kind):
    if kind=='wheelset':
        return [((0.,side*.77,0.),(.49,.105,.49)) for side in [-1,1]]+[
            ((0.,0.,0.),(.11,1.14,.11))]
    return [((0.,0.,0.),(.145,.10,.145))]


def segment_hit(a,b,bounds):
    lo,hi=0.,1.
    for p,q,mn,mx in zip(a,b,bounds[:3],bounds[3:]):
        d=q-p
        if abs(d)<1e-12:
            if p<mn or p>mx:return False
        else:
            u,v=sorted(((mn-p)/d,(mx-p)/d));lo=max(lo,u);hi=min(hi,v)
            if lo>hi:return False
    return True


def swept_hits(a,b,boxes,kind):
    hits=set()
    for center,half in envelopes(kind):
        start=[p+c for p,c in zip(a,center)];end=[p+c for p,c in zip(b,center)]
        for box in boxes:
            bounds=box['bounds'];expanded=[bounds[i]-half[i]-.01 for i in range(3)]+[
                bounds[i+3]+half[i]+.01 for i in range(3)]
            if segment_hit(start,end,expanded):hits.add(box['id'])
    return sorted(hits)


def static_boxes(scene):
    result=[];sources={}
    def append(name,size,transforms):
        corners=[(x*size[0]/2,y*size[1]/2,z*size[2]/2) for x in [-1,1] for y in [-1,1] for z in [-1,1]]
        for t in transforms:corners=[transform(p,t) for p in corners]
        bounds=[min(p[i] for p in corners) for i in range(3)]+[max(p[i] for p in corners) for i in range(3)]
        if bounds[5]>.02:result.append(dict(id=name,bounds=bounds))
    for item in scene['boxes']:
        append(item['id'],item['size'],[item['center']+[0,0,item.get('yaw',0)]])
    for item in scene['instances']:
        path=ROOT/'assets/wheelset_workshop/models'/('wsv1_'+item['asset'])/'model.sdf'
        sources[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        model=ET.parse(path).getroot().find('model')
        for link in model.findall('link'):
            for collision in link.findall('collision'):
                size=list(map(float,collision.findtext('geometry/box/size').split()))
                append(item['id']+'/'+collision.get('name'),size,[pose(collision),pose(link),pose(model),item['pose']])
    return result,sources


def planned_clearance(metadata):
    boxes=metadata['static_obstacles'];current=ORIGIN;items=parts();mounted=set(items);checks=[]
    def check(stage,name,a,b,kind):
        checks.append(dict(stage=stage,entity=name,start=a,end=b,hits=swept_hits(a,b,boxes,kind)))
    for stage in STAGES:
        end=stage['target'];points=[current,*transfer_points(current,stage)]
        for a,b in zip(points,points[1:]):
            check(stage['id'],'ws001',a,b,'wheelset')
            for name in sorted(mounted):
                offset=items[name]['offset']
                check(stage['id'],name,[p+o for p,o in zip(a,offset)],
                    [p+o for p,o in zip(b,offset)],'part')
        current=end;kind={'caps':'cap','bearings':'bearing'}.get(stage['action'])
        if kind:
            for name,item in items.items():
                if item['kind']!=kind:continue
                start=tuple(p+o for p,o in zip(end,item['offset']))
                released=(start[0],start[1]+item['sign']*item['separation_m'],start[2]);dest=item['destination']
                points=[start,released,(released[0],released[1],TRAVEL_Z),(dest[0],dest[1],TRAVEL_Z),dest]
                for a,b in zip(points,points[1:]):check(stage['id'],name,a,b,'part')
                mounted.remove(name)
    return dict(passed=all(not c['hits'] for c in checks),static_boxes_checked=len(boxes),checks=checks,
        boundary='Planned cargo boxes against frozen static collision boxes; no tool/carrier/contact or full visual-mesh clearance claim')


def add_body(world,name,origin,kind,mesh=None):
    model=e(world,'model',name=name);e(model,'pose',' '.join(map(str,origin))+' 0 0 0')
    base=e(model,'link',name='base');inertia(base,100 if kind=='wheelset' else 5,[100,100,100] if kind=='wheelset' else [1,1,1])
    joint(model,name+'_anchor','fixed','world','base',None)
    mass=1000 if kind=='wheelset' else 2.
    parent='base';axes=[]
    for axis,vector in zip('xyz',['1 0 0','0 1 0','0 0 1']):
        child=e(model,'link',name=axis)
        # A 1000kg payload on three 5kg/1kg.m2 links produced transverse ODE
        # constraint drift. Represent substantial assumed carriage frames;
        # do not mask that failure by loosening world/joint or stop checks.
        inertia(child,300 if kind=='wheelset' else 5,[300,300,300] if kind=='wheelset' else [1,1,1])
        joint(model,name+'_'+axis,'prismatic',parent,axis,vector,[-80,260,15000 if kind=='wheelset' else 400,2.5],3)
        parent=axis;axes.append(axis)
    body=e(model,'link',name='body');inertia(body,mass,[400,160,400] if kind=='wheelset' else [.05,.05,.05])
    if kind=='wheelset':
        joint(model,name+'_spin','revolute','z','body','0 1 0',[-1e16,1e16,1000,2.5],3);axes.append('spin')
        visual=e(body,'visual',name='bare_wheelset');e(visual,'pose','0 0 -.56 0 0 0')
        e(e(e(visual,'geometry'),'mesh'),'uri',Path(mesh).as_uri())
        box_visual(body,'rotation_mark','.065 .025 .26','1 .35 .03 1','0 -.90 .24 0 0 0')
        probe=e(model,'link',name='probe');e(probe,'pose','0 -1.57 0 0 0 0');inertia(probe,5,[.1,.1,.1])
        box_visual(probe,'assumed_scanner','.16 .30 .18','1 .3 .02 1')
        joint(model,name+'_probe','prismatic','z','probe','0 1 0',[0,.25,200,.5],4);axes.append('probe')
    else:
        joint(model,name+'_body','fixed','z','body',None)
        visual=e(body,'visual',name=kind);e(visual,'pose','0 0 0 1.57079632679 0 0')
        shape=e(e(visual,'geometry'),'cylinder');e(shape,'radius',.12 if kind=='cap' else .14)
        e(shape,'length',.04 if kind=='cap' else .06 if kind=='ring' else .16)
        material=e(visual,'material');e(material,'ambient','1 .5 .05 1' if kind=='cap' else '.25 .3 .35 1')
        e(material,'diffuse','1 .5 .05 1' if kind=='cap' else '.25 .3 .35 1')
    masses=[1905,1605,1305] if kind=='wheelset' else [17,12,7]
    plugin=e(model,'plugin',name=name+'_servo',filename='libwheelset_axis_servo.so')
    e(e(plugin,'ros'),'namespace','/'+name);e(plugin,'lease_s',1.)
    for axis in axes:
        definition=e(plugin,'axis');e(definition,'joint',name+'_'+axis)
        e(definition,'mode','velocity' if axis=='spin' else 'probe' if axis=='probe' else 'position')
        e(definition,'mass',160 if axis=='spin' else 5 if axis=='probe' else masses['xyz'.index(axis)])
        e(definition,'limit',1000 if axis=='spin' else 200 if axis=='probe' else 15000 if kind=='wheelset' else 400)
        e(definition,'damping',4 if axis=='probe' else 3)
    return dict(origin=origin,kind=kind,axes=axes,axis_masses_kg=masses)


def build_front_world(output):
    output=Path(output);scene=json.loads(SCENE.read_text(encoding='utf-8'));derived=copy.deepcopy(scene)
    removed=[i['id'] for i in derived['instances'] if i['asset']=='wheelset'
        and (abs(i['pose'][1]-9)<1e-6 or abs(i['pose'][1]-50)<1e-6)]
    derived['instances']=[i for i in derived['instances'] if i['id'] not in removed]
    camera=dict(id='front_to_inspection',position=[31,-36,59],target=[31,28,1.],
        horizontal_fov=1.1,width=1280,height=720,reference_second=None)
    tree=make_world(derived,camera);world=tree.find('world');physics=world.find('physics')
    physics.find('max_step_size').text='.002';physics.find('real_time_update_rate').text='500'
    ode=e(physics,'ode');solver=e(ode,'solver');e(solver,'type','quick');e(solver,'iters',400);e(solver,'sor',1)
    constraint=e(ode,'constraints');e(constraint,'cfm',1e-9);e(constraint,'erp',.8)
    mesh=wheelset();keep=[i for i,p in enumerate(mesh.parts) if not p.startswith(('bearing_','endcap_'))]
    mesh.faces=[mesh.faces[i] for i in keep];mesh.materials=[mesh.materials[i] for i in keep];mesh.parts=[mesh.parts[i] for i in keep]
    mesh_path=output/'bare_wheelset.dae';mesh.write_dae(mesh_path)
    definitions={'ws001':add_body(world,'ws001',ORIGIN,'wheelset',mesh_path)}
    for name,item in parts().items():
        origin=tuple(a+b for a,b in zip(ORIGIN,item['offset']))
        definitions[name]=add_body(world,name,origin,item['kind'])
    boxes,sources=static_boxes(derived)
    assumptions=dict(design='Assumed independent world-anchored XYZ axes for wheelset and four identified detachable items',
        gravity=False,contact=False,travel_height_m=TRAVEL_Z,
        wheelset_mass_kg=1000,xyz_carriage_link_mass_kg=300,xyz_carriage_inertia_kg_m2=[300,300,300],part_mass_kg=2,
        station_offsets='Accessible approach positions beside visual stations, explicit offsets from frozen asset origins',
        mounted_parts='Independent constrained axes follow the same position target; not Gazebo attachment/contact',
        identification='Geometric simulated tag read at measured scanner extension; not image OCR or a real tag reader',
        cargo_clearance='Conservative wheel/axle and part boxes swept against static SDF collision boxes; excludes carrier/tool and unmodelled visual geometry',
        output='Measured stop at inspection outfeed; external unload and downstream line not included',
        effort_control='Native Gazebo per-step bounded force servo; monotonic target sequences; 1s simulation-time command lease latches local braking/hold. Raw joint effort field records requested native forces, not contact force measurement.')
    save_xml(tree,output/'front.world')
    return dict(definitions=definitions,removed_static_snapshot_ids=removed,static_obstacles=boxes,
        collision_source_sha256=sources,assumptions=assumptions,stages=STAGES)
