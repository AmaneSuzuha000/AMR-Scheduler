"""End-line world with exact box structure for the one operated storage bay."""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .assembly_flow import components
from .assembly_world import build_assembly_world,assembly_clearance
from .assets import rack_bay
from .build import ROOT,SCENE,e,save_xml
from .finishing_flow import FULL_STAGES,PAINT,STORAGE,STORAGE_APPROACH,STORAGE_UNLOAD,STORED_JOB
from .front_flow import TRAVEL_Z,transfer_points,parts
from .front_world import add_body,static_boxes,swept_hits


FINISH_SNAPSHOTS={'wheelset_098','wheelset_099','finishing_r15_spray_wheel_8'}
RACK='storage_r07_wheelset_storage_bay_001'


def rack_collision_boxes():
    """rack_bay contains axis-aligned boxes, 12 triangles per actual box."""
    mesh=rack_bay();result=[]
    assert len(mesh.faces)%12==0
    for index in range(0,len(mesh.faces),12):
        vertices=[mesh.vertices[v] for face in mesh.faces[index:index+12] for v in face]
        result.append(dict(id=f'{RACK}/structure_{index//12}',
            bounds=[min(v[i] for v in vertices) for i in range(3)]+[max(v[i] for v in vertices) for i in range(3)]))
    return result


def finishing_static(scene):
    derived=copy.deepcopy(scene);rack=next(i for i in derived['instances'] if i['id']==RACK)
    assert rack['pose']==[209.1,30.700000000000003,0,0,0,0]
    derived['instances']=[i for i in derived['instances'] if i['id']!=RACK]
    boxes,sources=static_boxes(derived)
    for box in rack_collision_boxes():
        box['bounds']=[v+rack['pose'][i%3] for i,v in enumerate(box['bounds'])];boxes.append(box)
    relative='assets/wheelset_workshop/models/wsv1_wheelset_storage_bay/model.sdf'
    sources[relative]=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    return boxes,sources


def build_finishing_world(output):
    output=Path(output);meta=build_assembly_world(output)
    scene=json.loads(SCENE.read_text(encoding='utf-8'));removed=set(meta['removed_static_snapshot_ids'])|FINISH_SNAPSHOTS
    scene['instances']=[i for i in scene['instances'] if i['id'] not in removed]
    boxes,sources=finishing_static(scene)
    tree=ET.parse(output/'front.world');world=tree.getroot().find('world')
    for include in list(world.findall('include')):
        if include.findtext('name') in FINISH_SNAPSHOTS|{RACK}:world.remove(include)
    path=ROOT/'assets/wheelset_workshop/models/wsv1_wheelset_storage_bay/model.sdf'
    model=ET.parse(path).getroot().find('model');model.set('name',RACK);e(model,'pose','209.1 30.7 0 0 0 0')
    link=model.find('link')
    for collision in list(link.findall('collision')):link.remove(collision)
    for box in rack_collision_boxes():
        bounds=box['bounds'];collision=e(link,'collision',name=box['id'].split('/')[1])
        e(collision,'pose',' '.join(str((bounds[i]+bounds[i+3])/2) for i in range(3))+' 0 0 0')
        e(e(e(collision,'geometry'),'box'),'size',' '.join(str(bounds[i+3]-bounds[i]) for i in range(3)))
    world.append(model)
    # Dynamic existing content is a separate measured object, not a ledger-only occupant.
    meta['definitions'][STORED_JOB]=add_body(world,STORED_JOB,STORAGE,'wheelset',output/'bare_wheelset.dae')
    spray=(PAINT[0]-.70,PAINT[1],PAINT[2]+.85)
    meta['definitions']['spray_tool']=add_body(world,'spray_tool',spray,'spray_tool')
    world.find("model[@name='review_camera']/pose").text='118 -95 145 0 .78 1.57079632679'
    meta.update(stages=FULL_STAGES,removed_static_snapshot_ids=sorted(removed),static_obstacles=boxes,
        collision_source_sha256=sources,storage_target=STORAGE,storage_approach=STORAGE_APPROACH,
        storage_unload=STORAGE_UNLOAD,rack_structural_collision_boxes=rack_collision_boxes())
    meta['assumptions'].update(output='Measured finite storage slot commit for same workpiece; full multi-job/fleet integration still pending',
        closing='Reuse two tracked removed caps in demonstration; no refurbishment/torque quality claim',
        run_in='Measured sustained spindle speed and at least two turns, then stopped; no bearing temperature/quality model',
        painting='Measured separate spray-tool traverse; simulated coating event, no paint fluid/contact model',
        storage='One finite slot initially occupied by a dynamic prior wheelset; actual removal enables reservation. Selected rack box collision refined from its exact existing mesh boxes, original visual assets unchanged. No contact-supported loading claim')
    save_xml(tree.getroot(),output/'front.world');return meta


def finishing_clearance(meta):
    result=assembly_clearance(meta);checks=result['checks'];boxes=meta['static_obstacles']
    def check(stage,name,a,b,kind='part'):
        checks.append(dict(stage=stage,entity=name,start=a,end=b,hits=swept_hits(a,b,boxes,kind)))
    def path(stage,name,points,kind='part'):
        for a,b in zip(points,points[1:]):check(stage,name,a,b,kind)
    current=FULL_STAGES[12]['target']
    for stage in FULL_STAGES[13:]:
        points=[current,*transfer_points(current,stage)];path(stage['id'],'ws001',points,'wheelset')
        for name,item in {**components(),**{n:i for n,i in parts().items() if i['kind']=='cap' and stage['id']!='closing'}}.items():
            path(stage['id'],name,[[p+o for p,o in zip(point,item['offset'])] for point in points])
        current=stage['target']
    for name,item in parts().items():
        if item['kind']!='cap':continue
        r=item['destination'];f=[a+b for a,b in zip(FULL_STAGES[13]['target'],item['offset'])]
        s=[*f];s[1]+=item['sign']*.35
        path('cap_retrieval',name,[r,(*r[:2],6.),(*s[:2],6.),s,f])
    path('storage_release',STORED_JOB,[STORAGE,STORAGE_APPROACH,(*STORAGE_APPROACH[:2],6.),(*STORAGE_UNLOAD[:2],6.),STORAGE_UNLOAD],'wheelset')
    a=meta['definitions']['spray_tool']['origin'];b=(PAINT[0]+.70,PAINT[1],PAINT[2]+.85)
    path('spray_tool','spray_tool',[a,b,a])
    result['passed']=all(not c['hits'] for c in checks);return result
