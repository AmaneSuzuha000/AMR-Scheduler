"""Reopen the saved editable scene with Blender and compare it with the authoritative layout."""
import hashlib
import json
import math
from pathlib import Path
import sys
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.wheelset_scene.evidence import build_fingerprint


def digest(path):
    with path.open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def main():
    source=ROOT/'scenes/wheelset_workshop_video_v1.json'
    saved=ROOT/'assets/wheelset_workshop/source/wheelset_workshop.blend'
    scene=json.loads(source.read_text(encoding='utf-8'));scale=scene['global_scene_scale']
    bpy.ops.wm.open_mainfile(filepath=str(saved))
    for item in scene['instances']:
        obj=bpy.data.objects[item['id']]
        assert obj.instance_collection.name=='ASSET_'+item['asset'],item['id']
        assert math.dist(obj.location,[v*scale for v in item['pose'][:3]])<1e-4,item['id']
        assert math.dist(obj.rotation_euler,item['pose'][3:])<1e-4,item['id']
        assert math.dist(obj.scale,[scale]*3)<1e-4,item['id']
    for camera in scene['cameras']:
        obj=bpy.data.objects['CAM_'+camera['id']]
        assert math.dist(obj.location,[v*scale for v in camera['position']])<1e-4,camera['id']
        direction=(Vector(camera['target'])-Vector(camera['position'])).normalized()
        forward=obj.rotation_euler.to_matrix()@Vector((0,0,-1))
        assert forward.dot(direction)>1-1e-6,camera['id']
        assert abs(obj.data.angle-camera['horizontal_fov'])<1e-6,camera['id']
    camera=bpy.data.objects['CAM_top'];projected=[]
    for x in [0,scene['bounds_m'][0]]:
        for y in [0,scene['bounds_m'][1]]:
            point=world_to_camera_view(bpy.context.scene,camera,Vector((x*scale,y*scale,0)))
            assert 0<point.x<1 and 0<point.y<1,'Floor corner outside top view'
            projected.append(list(point))
    report={'passed':True,'build_fingerprint':build_fingerprint(),'scene_sha256':digest(source),
        'blend_sha256':digest(saved),'saved_instances_checked':len(scene['instances']),
        'saved_cameras_checked':len(scene['cameras']),'camera_orientation_and_fov_checked':True,
        'top_floor_corners_in_frame':True,'top_corner_projections':projected,
        'claim_boundary':'Saved editable scene identity and poses only; no visual-fidelity or dynamics claim.'}
    (ROOT/'simulation/wheelset_workshop/blender_check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
