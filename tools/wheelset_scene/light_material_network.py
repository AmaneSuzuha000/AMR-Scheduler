"""Independent light-material road and conservative static collision audit.

The north service corridor is an explicit layout addition. Handoffs are outside
the production boundary; downstream loading mechanisms remain unmodelled.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .build import ROOT,SCENE
from .evidence import build_fingerprint


def transform(p,pose):
    x,y,z=p;tx,ty,tz,roll,pitch,yaw=pose
    y,z=y*math.cos(roll)-z*math.sin(roll),y*math.sin(roll)+z*math.cos(roll)
    x,z=x*math.cos(pitch)+z*math.sin(pitch),-x*math.sin(pitch)+z*math.cos(pitch)
    return (tx+x*math.cos(yaw)-y*math.sin(yaw),ty+x*math.sin(yaw)+y*math.cos(yaw),tz+z)


def pose(element):return list(map(float,element.findtext('pose','0 0 0 0 0 0').split()))


def occupied_boxes(scene):
    result=[];sources={}
    def append(name,size,transforms):
        corners=[(sx*size[0]/2,sy*size[1]/2,sz*size[2]/2) for sx in [-1,1] for sy in [-1,1] for sz in [-1,1]]
        for t in transforms:corners=[transform(p,t) for p in corners]
        bounds=[min(p[d] for p in corners) for d in range(3)]+[max(p[d] for p in corners) for d in range(3)]
        if bounds[5]>.03 and bounds[2]<.8:result.append(dict(id=name,bounds=bounds))
    scale=scene['global_scene_scale']
    for box in scene['boxes']:
        append(box['id'],[v*scale for v in box['size']],[[v*scale for v in box['center']]+[0,0,box.get('yaw',0)]])
    for instance in scene['instances']:
        path=ROOT/'assets/wheelset_workshop/models'/('wsv1_'+instance['asset'])/'model.sdf'
        sources[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        model=ET.parse(path).getroot().find('model');world_pose=instance['pose'][:]
        world_pose[:3]=[v*scale for v in world_pose[:3]]
        for link in model.findall('link'):
            for collision in link.findall('collision'):
                size=collision.findtext('geometry/box/size')
                if size is None:raise ValueError('Uncovered collision shape: '+instance['id'])
                append(instance['id']+'/'+collision.get('name'),list(map(float,size.split())),
                    [pose(collision),pose(link),pose(model),world_pose])
    return result,sources


def point_segment(p,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1];den=dx*dx+dy*dy
    t=max(0.,min(1.,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/den)) if den else 0.
    return math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy)


def segment_box(a,b,bounds):
    left,bottom,_,right,top,_=bounds;low,high=0.,1.
    for origin,direction,mn,mx in [(a[0],b[0]-a[0],left,right),(a[1],b[1]-a[1],bottom,top)]:
        if abs(direction)<1e-12:
            if not mn<=origin<=mx:low,high=1.,0.;break
        else:
            t0,t1=sorted(((mn-origin)/direction,(mx-origin)/direction));low=max(low,t0);high=min(high,t1)
    if low<=high:return 0.
    def point_box(p):return math.hypot(max(left-p[0],0,p[0]-right),max(bottom-p[1],0,p[1]-top))
    return min(point_box(a),point_box(b),*(point_segment(p,a,b) for p in [(left,bottom),(left,top),(right,bottom),(right,top)]))


def proposed_network():
    nodes=[dict(id=name,x=x,y=y,kind=kind) for name,x,y,kind in [
        ('supply',6.,69.2,'handoff'),('west',20.,69.2,'aisle'),('east',40.,69.2,'aisle'),
        ('receiving',60.,69.2,'handoff'),('bay_west',20.,71.,'buffer'),('bay_east',40.,71.,'buffer')]]
    lanes=[dict(id=name,src=a,dst=b,width_m=1.8,speed_limit_mps=.5,capacity=1,two_way=True)
        for name,a,b in [('west_approach','supply','west'),('shared_middle','west','east'),
            ('east_approach','east','receiving'),('west_bay','west','bay_west'),('east_bay','east','bay_east')]]
    return dict(id='wheelset_light_material_v1',source='explicit_extension_not_video_measured',
        coordinates='frozen scene world metres',robot_enclosing_radius_m=.55,safety_margin_m=.2,
        nodes=nodes,lanes=lanes,robots=[dict(id='scout_a',start='bay_west'),dict(id='scout_b',start='bay_east')],
        allowed_materials=['inspection_consumables','small_tools','bearing_accessories'],
        excluded_materials=['wheelset','axle','bogie'],
        handoff_boundary='North service corridor outside low production boundary; downstream handoff mechanisms unmodelled')


def audit(network,scene):
    boxes,sources=occupied_boxes(scene);nodes={n['id']:n for n in network['nodes']};checks=[]
    radius=network['robot_enclosing_radius_m']+network['safety_margin_m'];width,height=scene['bounds_m']
    for lane in network['lanes']:
        a=nodes[lane['src']];b=nodes[lane['dst']];a=(a['x'],a['y']);b=(b['x'],b['y'])
        clearance,obstacle=min((segment_box(a,b,box['bounds']),box['id']) for box in boxes)
        floor=min(a[0],b[0],width-a[0],width-b[0],a[1],b[1],height-a[1],height-b[1])
        checks.append(dict(lane=lane['id'],nearest_obstacle=obstacle,static_clearance_m=clearance,
            floor_edge_clearance_m=floor,required_radius_m=radius,
            passed=clearance>=radius and floor>=radius and lane['width_m']/2>=radius))
    return dict(passed=all(c['passed'] for c in checks),network=network,checks=checks,
        static_collision_boxes_checked=len(boxes),collision_source_sha256=sources,build_fingerprint=build_fingerprint(),
        claim_boundary='Conservative static box clearance for a proposed independent light-material corridor; assumed robot envelope. No robot navigation, contact, localization, traffic execution, device delivery or factory safety acceptance.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    result=audit(proposed_network(),json.loads(SCENE.read_text(encoding='utf-8')))
    (args.output/'scene_layout.json').write_text(json.dumps(result['network'],indent=2)+'\n')
    (args.output/'clearance_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['passed','static_collision_boxes_checked','checks']}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
