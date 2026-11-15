"""Build portable COLLADA assets and a Gazebo Classic world from one scene JSON.

python -m tools.wheelset_scene.build --init
python -m tools.wheelset_scene.build
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .assets import make_assets, collision_boxes
from .geometry import PALETTE
from .layout import initial_scene
from .evidence import build_fingerprint

ROOT=Path(__file__).resolve().parents[2]
SCENE=ROOT/"scenes/wheelset_workshop_video_v1.json"
ASSET_ROOT=ROOT/"assets/wheelset_workshop"
SIM_ROOT=ROOT/"simulation/wheelset_workshop"


def e(parent,tag,text=None,**attrs):
    node=ET.SubElement(parent,tag,{k:str(v) for k,v in attrs.items()})
    if text is not None:node.text=str(text)
    return node


def numbers(values):
    return " ".join(f"{v:.7g}" for v in values)


def save_xml(root,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(path,encoding="utf-8",xml_declaration=True)


def camera_pose(camera,scale=1):
    eye=camera["position"];target=camera["target"]
    dx,dy,dz=[target[i]-eye[i] for i in range(3)]
    yaw=camera.get("yaw_rad",math.atan2(dy,dx))
    return [x*scale for x in eye]+[0,math.atan2(-dz,math.hypot(dx,dy)),yaw]


def geometry_box(parent,size):
    e(e(e(parent,"geometry"),"box"),"size",numbers(size))


def box_model(world,item,scale=1,collision=True):
    m=e(world,"model",name=item["id"]);e(m,"static","true")
    center=[x*scale for x in item["center"]]
    size=[x*scale for x in item["size"]]
    e(m,"pose",numbers(center+[0,0,item.get("yaw",0)]))
    link=e(m,"link",name="body")
    if collision:geometry_box(e(link,"collision",name="collision"),size)
    visual=e(link,"visual",name="visual");geometry_box(visual,size)
    mat=e(visual,"material")
    rgba=numbers(PALETTE[item["material"]]);e(mat,"ambient",rgba);e(mat,"diffuse",rgba)
    e(mat,"specular","0.05 0.05 0.05 1")


def paint_strips(scene):
    strips=[]
    for road in scene["agv_routes"]:
        a,b=road["start"],road["end"]
        strips.append({"id":"paint_"+road["id"],"center":[(a[0]+b[0])/2,(a[1]+b[1])/2,.006],
                       "size":[math.dist(a,b),.14,.008],"material":"yellow",
                       "yaw":math.atan2(b[1]-a[1],b[0]-a[0])})
    return strips


def make_world(scene,camera):
    scale=scene["global_scene_scale"]
    root=ET.Element("sdf",version="1.6");world=e(root,"world",name=scene["scene_id"])
    e(world,"gravity","0 0 -9.81")
    physics=e(world,"physics",name="static_visual_review",type="ode")
    e(physics,"max_step_size","0.01");e(physics,"real_time_update_rate","100")
    settings=e(world,"scene");e(settings,"ambient","0.60 0.60 0.60 1");e(settings,"background","0.60 0.62 0.65 1")
    e(settings,"shadows","true")
    light=e(world,"light",name="sun",type="directional")
    e(light,"pose","0 0 50 0 0 0");e(light,"diffuse","0.75 0.75 0.73 1")
    e(light,"specular","0.22 0.22 0.22 1");e(light,"direction","-0.5 0.3 -0.8");e(light,"cast_shadows","true")
    gui=e(world,"gui",fullscreen="false");cam=e(gui,"camera",name="user_camera")
    e(cam,"pose",numbers(camera_pose(camera,scale)));e(cam,"view_controller","orbit")
    for box in scene["boxes"]:box_model(world,box,scale)
    for strip in paint_strips(scene):box_model(world,strip,scale,collision=False)
    for item in scene["instances"]:
        inc=e(world,"include");e(inc,"uri","model://wsv1_"+item["asset"]);e(inc,"name",item["id"])
        pose=item["pose"][:];pose[:3]=[v*scale for v in pose[:3]];e(inc,"pose",numbers(pose))
    state=e(world,"plugin",name="wheelset_state",filename="libgazebo_ros_state.so")
    e(e(state,"ros"),"namespace","/wheelset_review");e(state,"update_rate","20")
    model=e(world,"model",name="review_camera");e(model,"static","true")
    e(model,"pose",numbers(camera_pose(camera,scale)))
    link=e(model,"link",name="camera_link");sensor=e(link,"sensor",name="review_sensor",type="camera")
    e(sensor,"always_on","true");e(sensor,"update_rate","2");e(sensor,"visualize","false")
    optics=e(sensor,"camera");e(optics,"horizontal_fov",camera["horizontal_fov"])
    im=e(optics,"image");e(im,"width",camera["width"]);e(im,"height",camera["height"]);e(im,"format","R8G8B8")
    clip=e(optics,"clip");e(clip,"near","0.1");e(clip,"far",max(300,4*max(scene['bounds_m']))*scale)
    plugin=e(sensor,"plugin",name="review_camera_ros",filename="libgazebo_ros_camera.so")
    ros=e(plugin,"ros");e(ros,"namespace","/wheelset_review")
    e(ros,"remapping","image_raw:=image_raw");e(ros,"remapping","camera_info:=camera_info")
    e(plugin,"camera_name","reference");e(plugin,"frame_name","camera_link")
    return root


def build(scene_path=SCENE,init=False):
    if init and not scene_path.exists():
        scene_path.parent.mkdir(parents=True,exist_ok=True)
        scene_path.write_text(json.dumps(initial_scene(),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    scene=json.loads(scene_path.read_text(encoding="utf-8"))
    scale=scene["global_scene_scale"]
    if not math.isfinite(scale) or scale<=0:raise ValueError("global_scene_scale must be positive")
    assets=make_assets();bundle={"palette":PALETTE,"assets":{}}
    for name,mesh in assets.items():
        directory=ASSET_ROOT/"models"/("wsv1_"+name)
        mesh.write_dae(directory/"meshes"/(name+".dae"))
        config=ET.Element("model");e(config,"name","wsv1_"+name);e(config,"version","1.0")
        e(config,"sdf","model.sdf",version="1.6")
        e(config,"description","Video-reference visual reconstruction. Dimensions estimated; static visual release.")
        save_xml(config,directory/"model.config")
        root=ET.Element("sdf",version="1.6");model=e(root,"model",name="wsv1_"+name);e(model,"static","true")
        link=e(model,"link",name="asset");visual=e(link,"visual",name="visual")
        geom=e(e(visual,"geometry"),"mesh");e(geom,"uri",f"model://wsv1_{name}/meshes/{name}.dae")
        e(geom,"scale",numbers([scale]*3))
        boxes=collision_boxes(name)
        if boxes is None:
            low,high=mesh.bounds();boxes=[([(a+b)/2 for a,b in zip(low,high)],[b-a for a,b in zip(low,high)])]
        for i,(center,size) in enumerate(boxes):
            collision=e(link,"collision",name=f"collision_{i}")
            e(collision,"pose",numbers([v*scale for v in center]+[0,0,0]))
            geometry_box(collision,[v*scale for v in size])
        save_xml(root,directory/"model.sdf")
        bundle["assets"][name]=mesh.as_dict()
    source=ASSET_ROOT/"source";source.mkdir(parents=True,exist_ok=True)
    with gzip.open(source/"asset_library.json.gz","wt",encoding="utf-8") as f:json.dump(bundle,f,separators=(",",":"))
    SIM_ROOT.mkdir(parents=True,exist_ok=True)
    (SIM_ROOT/"cameras.json").write_text(json.dumps(scene["cameras"],ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for camera in scene["cameras"]:
        save_xml(make_world(scene,camera),SIM_ROOT/"worlds"/("camera_"+camera["id"]+".world"))
    save_xml(make_world(scene,scene["cameras"][0]),SIM_ROOT/"worlds/wheelset_workshop.world")
    referenced=Counter(seg for item in scene["instances"] for seg in item["source_segments"])
    report={"scene_id":scene["scene_id"],"visual_revision":scene.get("visual_revision","0.1"),"build_fingerprint":build_fingerprint(),"scene_sha256":hashlib.sha256(scene_path.read_bytes()).hexdigest(),
            "asset_types":len(assets),"instances":len(scene["instances"]),"instances_by_asset":dict(Counter(i["asset"] for i in scene["instances"])),
            "instances_by_zone":dict(Counter(i["zone"] for i in scene["instances"])),
            "environment_boxes":len(scene["boxes"]),"camera_bookmarks":len(scene["cameras"]),
            "reference_segment_instance_counts":dict(referenced),"library_triangles":sum(len(a.faces) for a in assets.values()),
            "instanced_triangles":sum(len(assets[i["asset"]].faces) for i in scene["instances"]),
            "status":"BUILT_NOT_RUNTIME_VERIFIED","camera_registration":"INITIAL_ESTIMATE",
            "physics_validation":False,"dynamic_process_validation":False,"scheduler_integration":False}
    (SIM_ROOT/"build_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ["asset_types","instances","camera_bookmarks","library_triangles","instanced_triangles"]}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--init",action="store_true")
    parser.add_argument("--scene",type=Path,default=SCENE)
    args=parser.parse_args();build(args.scene,args.init)


if __name__=="__main__":main()
