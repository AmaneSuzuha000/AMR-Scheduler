"""Blender entrypoint: creates an editable asset library and collection instances.

blender --background --python tools/wheelset_scene/blender_scene.py -- --render overview
"""
import argparse
import gzip
import json
import os
import hashlib
import math
from pathlib import Path
import sys
from collections import defaultdict
import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.wheelset_scene.evidence import build_fingerprint
from tools.wheelset_scene import render_profile
parser=argparse.ArgumentParser()
parser.add_argument("--render",default="overview")
# Blender 5.2 renamed the EEVEE engine enum: BLENDER_EEVEE_NEXT no longer exists.
# Both names are accepted so old command lines keep working.
parser.add_argument("--engine",default="CYCLES",choices=["CYCLES","BLENDER_EEVEE","BLENDER_EEVEE_NEXT"])
parser.add_argument("--profile",default="contrast",choices=render_profile.available())
parser.add_argument("--samples",type=int,default=24)
# A/B variants need to be built without overwriting the shipped scene file, so the scene
# path is a parameter. Defaulting to the committed scene keeps every existing command line
# behaving exactly as before.
parser.add_argument("--scene",default=str(ROOT/"scenes/wheelset_workshop_video_v1.json"))
parser.add_argument("--blend-out",default="")
args=parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
scene_path=Path(args.scene)
if not scene_path.is_absolute():scene_path=ROOT/scene_path
scene_data=json.loads(scene_path.read_text(encoding="utf-8"))
with gzip.open(ROOT/"assets/wheelset_workshop/source/asset_library.json.gz","rt",encoding="utf-8") as f:library=json.load(f)
bpy.ops.object.select_all(action="SELECT");bpy.ops.object.delete(use_global=False)
scene=bpy.context.scene
scene.unit_settings.system="METRIC";scene.unit_settings.scale_length=1
scene["evidence_status"]="Video-referenced geometry with estimated dimensions. Camera registration pending."
scale=scene_data["global_scene_scale"]
materials={}
for name,rgba in library["palette"].items():
    mat=bpy.data.materials.new(name);mat.diffuse_color=rgba;mat.use_nodes=True
    bsdf=mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value=rgba
    bsdf.inputs["Metallic"].default_value=.8 if name in {"steel","polished","dark_steel"} else .18
    bsdf.inputs["Roughness"].default_value=.25 if name=="polished" else .38 if name=="steel" else .55
    bsdf.inputs["Alpha"].default_value=rgba[3]
    materials[name]=mat
# Dark-surface albedo override, for calibrating against the reference's tonal range.
# Measured over all 29 views, surfaces the reference renders near 0.21 grey come back from
# our render at 0.47, while the floor matches (0.611 against 0.635) - the error is confined
# to the dark surfaces, so it is albedo, not exposure. A swatch ladder rendered under this
# same profile puts albedo 0.31 (the current "wall") at 0.341 and albedo near 0.14 at the
# reference's 0.21, so the override exists to test that specific claim across the suite.
# Format: "wall=0.14,black=0.02". Empty means every material keeps its palette value.
_override=os.environ.get("WS_ALBEDO_OVERRIDE","")
if _override:
    for _pair in _override.split(","):
        if not _pair.strip():continue
        _key,_value=_pair.split("=")
        _key=_key.strip();_value=float(_value)
        if _key not in materials:continue
        _mat=materials[_key]
        _bsdf=_mat.node_tree.nodes.get("Principled BSDF")
        _current=list(_bsdf.inputs["Base Color"].default_value)
        _bsdf.inputs["Base Color"].default_value=(_value,_value,_value,_current[3])
        _mat.diffuse_color=(_value,_value,_value,_current[3])
        print("ALBEDO_OVERRIDE",_key,_value)
# Metallic/roughness override, same A/B mechanism. Measured reason: in storage_reverse the
# rack region spans p5 0.223 to p95 0.878 in the reference but only 0.312 to 0.625 in ours,
# while member pitch (21/23, 36/29, 37.5/39 px) and width differ by only ~30%. The members
# are in the right place at the right size - they just do not catch specular highlights or
# go dark on their shadowed faces, which is a material response, not geometry or albedo.
# Format: "gray:metallic=0.85,roughness=0.25".
_shiny=os.environ.get("WS_MATERIAL_SHINY","")
if _shiny:
    for _spec in _shiny.split(";"):
        if ":" not in _spec:continue
        _key,_props=_spec.split(":",1)
        _key=_key.strip()
        if _key not in materials:continue
        _bsdf=materials[_key].node_tree.nodes.get("Principled BSDF")
        for _pair in _props.split(","):
            _name,_value=_pair.split("=")
            _name=_name.strip()
            if _name not in ("metallic","roughness"):continue
            _bsdf.inputs["Metallic" if _name=="metallic" else "Roughness"].default_value=float(_value)
        print("MATERIAL_SHINY",_key,_props)

prototypes={}
for name,asset in library["assets"].items():
    collection=bpy.data.collections.new("ASSET_"+name)
    collection.use_fake_user=True
    groups=defaultdict(list)
    for i,part in enumerate(asset["parts"]):groups[part].append(i)
    for part,face_indexes in groups.items():
        vertex_ids=sorted({v for i in face_indexes for v in asset["faces"][i]})
        mapping={v:i for i,v in enumerate(vertex_ids)}
        mesh=bpy.data.meshes.new(name+"__"+part)
        mesh.from_pydata([asset["vertices"][i] for i in vertex_ids],[],
                         [[mapping[v] for v in asset["faces"][i]] for i in face_indexes])
        mesh.update()
        used=sorted({asset["materials"][i] for i in face_indexes})
        for material in used:mesh.materials.append(materials[material])
        for poly,i in zip(mesh.polygons,face_indexes):
            poly.material_index=used.index(asset["materials"][i])
            poly.use_smooth=any(t in part for t in ["wheel","axle","shaft","bearing","joint","roller","ram","tread","tool","cable"])
        obj=bpy.data.objects.new(part,mesh);collection.objects.link(obj)
        obj["asset_part"]=part
    prototypes[name]=collection

zones={}
for zone in [z["id"] for z in scene_data["zones"]]+["environment"]:
    collection=bpy.data.collections.new(zone);scene.collection.children.link(collection);zones[zone]=collection
for item in scene_data["instances"]:
    obj=bpy.data.objects.new(item["id"],None)
    obj.instance_type="COLLECTION";obj.instance_collection=prototypes[item["asset"]]
    obj.location=[v*scale for v in item["pose"][:3]];obj.rotation_euler=item["pose"][3:]
    obj.scale=(scale,)*3
    obj["source_segments"]=",".join(item["source_segments"]);obj["pose_source"]="estimated"
    zones[item["zone"]].objects.link(obj)

sys.path.insert(0,str(ROOT))
from tools.wheelset_scene.build import paint_strips
for item in scene_data["boxes"]+paint_strips(scene_data):
    bpy.ops.mesh.primitive_cube_add(size=1,location=[v*scale for v in item["center"]])
    obj=bpy.context.object;obj.name=item["id"];obj.dimensions=[v*scale for v in item["size"]]
    obj.rotation_euler[2]=item.get("yaw",0);obj.data.materials.append(materials[item["material"]])
    for coll in list(obj.users_collection):coll.objects.unlink(obj)
    zones["environment"].objects.link(obj)

cameras={}
for camera in scene_data["cameras"]:
    data=bpy.data.cameras.new(camera["id"]);obj=bpy.data.objects.new("CAM_"+camera["id"],data)
    scene.collection.objects.link(obj);obj.location=[v*scale for v in camera["position"]]
    direction=Vector([v*scale for v in camera["target"]])-obj.location
    obj.rotation_euler=direction.to_track_quat("-Z","Y").to_euler()
    data.type="PERSP";data.sensor_fit="HORIZONTAL";data.angle=camera["horizontal_fov"]
    data.clip_end=1000*scale;obj["reference_second"]=camera["reference_second"] if camera["reference_second"] is not None else -1
    cameras[camera["id"]]=obj
# Lighting and colour management come from render_profile so that a comparison can
# name the rig that produced it. Measured 2026-11-11: the old inline rig rendered
# tonal standard deviation 0.123 against 0.193 in the reference frames (a 1.57x
# contrast deficit) while brightness was already correct. The "contrast" profile
# recovers about 62% of that gap and scores +0.027 over 8 views, so it is the
# default. Pass --profile legacy to reproduce the pre-2026-11-11 previews.
profile_report=render_profile.apply_profile(scene,args.profile)
print("RENDER_PROFILE",json.dumps(profile_report,ensure_ascii=False))
scene.render.engine=("BLENDER_EEVEE" if args.engine=="BLENDER_EEVEE_NEXT" else args.engine)
if args.engine=="CYCLES":
    scene.cycles.samples=args.samples;scene.cycles.use_denoising=True
    try:
        prefs=bpy.context.preferences.addons["cycles"].preferences;prefs.compute_device_type="OPTIX";prefs.get_devices()
        available=[d for d in prefs.devices if d.type=="OPTIX"]
        if available:
            for d in prefs.devices:d.use=d.type=="OPTIX"
            scene.cycles.device="GPU"
            print("CYCLES_DEVICE",[d.name for d in available])
    except Exception as error:print("GPU selection fallback",error)
scene.render.resolution_x=1280;scene.render.resolution_y=720;scene.render.resolution_percentage=100
scene.render.image_settings.file_format="PNG"
scene.camera=cameras.get(args.render,cameras["overview"])
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=="VIEW_3D":
            area.spaces.active.region_3d.view_perspective="CAMERA"
out=(Path(args.blend_out) if args.blend_out else ROOT/"assets/wheelset_workshop/source/wheelset_workshop.blend")
if not out.is_absolute():out=ROOT/out
out.parent.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(out))
renders=ROOT/"visual_comparison/wheelset_workshop_v1/blender_previews";renders.mkdir(parents=True,exist_ok=True)
selected=list(cameras) if args.render=="all" else args.render.split(",")
report_path=renders/"render_report.json"
fingerprint=build_fingerprint()
scene_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest()
report=json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
if report.get("build_fingerprint")!=fingerprint:report={"build_fingerprint":fingerprint,"cameras":{}}
for name in selected:
    scene.camera=cameras[name];scene.render.filepath=str(renders/(name+".png"))
    bpy.ops.render.render(write_still=True)
    report["cameras"][name]={"sha256":hashlib.sha256((renders/(name+".png")).read_bytes()).hexdigest(),"engine":args.engine,"samples":args.samples}
assert build_fingerprint()==fingerprint, 'Scene or exported assets changed during rendering; rerender the current build.'
report['scene_sha256']=scene_sha256
report_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print("BLENDER_SCENE_COMPLETE",out)
