"""Check generated assets and cross-file invariants; does not certify visual fidelity or dynamics."""
import gzip
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from .assets import collision_boxes
from .build import ROOT,ASSET_ROOT,SIM_ROOT,SCENE
from .geometry import cross


def validate():
    scene=json.loads(SCENE.read_text(encoding="utf-8"))
    with gzip.open(ASSET_ROOT/"source/asset_library.json.gz","rt",encoding="utf-8") as f:library=json.load(f)
    issues=[];assets=library["assets"]
    ids=[i["id"] for i in scene["instances"]]+[i["id"] for i in scene["boxes"]]
    if len(ids)!=len(set(ids)):issues.append("Duplicate scene instance ID")
    zone_ids=[z['id'] for z in scene['zones']]
    if len(zone_ids)!=len(set(zone_ids)):issues.append('Duplicate scene zone ID')
    if scene["global_scene_scale"]<=0:issues.append("Non-positive scene scale")
    if any(c["horizontal_fov"]!=scene["cameras"][0]["horizontal_fov"] for c in scene["cameras"]):
        issues.append("Live camera poses require shared optics, but FOVs differ")
    for item in scene["instances"]:
        if item["asset"] not in assets:issues.append("Unknown asset "+item["asset"])
        if item['zone'] not in zone_ids:issues.append('Unknown zone '+item['zone']+' for '+item['id'])
        if len(item["pose"])!=6 or not all(math.isfinite(v) for v in item["pose"]):issues.append("Invalid pose "+item["id"])
        if item["pose_source"] not in {"estimated","observed","assumed"}:issues.append("Missing provenance "+item["id"])
    covered={s for i in scene["instances"] for s in i["source_segments"]}
    required={s["id"] for s in scene["video_segments"]}
    if required-covered:issues.append("Unrepresented reference segments "+str(sorted(required-covered)))
    for name,asset in assets.items():
        count=len(asset["vertices"])
        if not all(math.isfinite(v) for row in asset["vertices"] for v in row):issues.append("Non-finite mesh "+name)
        if not all(len(f)==3 and all(0<=i<count for i in f) for f in asset["faces"]):issues.append("Invalid triangle "+name)
        if not len(asset["faces"])==len(asset["parts"])==len(asset["materials"]):issues.append("Mesh metadata mismatch "+name)
        directory=ASSET_ROOT/"models"/("wsv1_"+name)
        for f in [directory/"model.config",directory/"model.sdf",directory/"meshes"/(name+".dae")]:
            if not f.exists():issues.append("Missing resource "+str(f));continue
            ET.parse(f)
        sdf=ET.parse(directory/"model.sdf")
        for size in sdf.findall(".//collision/geometry/box/size"):
            if any(float(v)<=0 for v in size.text.split()):issues.append("Non-positive collision size "+name)
    # Gazebo culls inward-facing surfaces that Blender can show from both sides.
    # Closed extruded part groups must have positive signed volume. This catches
    # a reversed complete group, not arbitrary mixed winding or visual fidelity.
    winding_checks=[]
    for name,parts in [('bearing_press_cell',{'faceted_white_hood','hood_faces'}),
            ('back_ring_measuring_booth',{'rounded_black_cover','cover_edges'}),
            ('wiping_station',{'perforated_wiping_bracket'})]:
        asset=assets[name];volume=0.
        for face,part in zip(asset['faces'],asset['parts']):
            if part not in parts:continue
            a,b,c=[asset['vertices'][i] for i in face]
            volume+=sum(x*y for x,y in zip(a,cross(b,c)))/6
        if volume<=0:issues.append('Inward or empty closed-part winding in '+name)
        winding_checks.append({'asset':name,'parts':sorted(parts),'signed_volume_m3':volume,
            'positive_oriented_volume':volume>0})
    # A wheelset axis must fit through the centre of a portal; one solid collision box would fail.
    passages=["portal_press","cap_portal","identification_hood","derust_hood","magnetic_test_hood","ultrasonic_test_hood","bearing_press_cell",
              "opening_tool_station","opening_portal","opening_frontage_portal","bearing_demount_station","bearing_demount_frontage","wiping_station","greasing_station",
              "bearing_run_in_stand","spray_portal","preparation_infeed_bench"]
    for name in passages:
        for center,size in collision_boxes(name):
            if abs(center[1])<=size[1]/2 and abs(center[2]-.75)<=size[2]/2:
                issues.append("Collision closes wheelset passage in "+name)
    world=ET.parse(SIM_ROOT/"worlds/wheelset_workshop.world")
    includes=world.findall(".//world/include")
    if len(includes)!=len(scene["instances"]):issues.append("World instance count differs from source JSON")
    for include in includes:
        uri=include.findtext("uri")
        if not (ASSET_ROOT/"models"/uri.removeprefix("model://")/"model.sdf").exists():issues.append("Unresolved include "+uri)
    for camera in scene["cameras"]:ET.parse(SIM_ROOT/"worlds"/("camera_"+camera["id"]+".world"))
    import hashlib
    from .evidence import build_fingerprint
    report={"passed":not issues,"build_fingerprint":build_fingerprint(),"scene_sha256":hashlib.sha256(SCENE.read_bytes()).hexdigest(),"issues":issues,"asset_types":len(assets),"instances":len(ids),
            "reference_segments_represented":len(covered),"reference_segments_required":len(required),
            "open_portals_checked":len(passages),"camera_bookmarks":len(scene["cameras"]),
            "closed_part_winding_checks":winding_checks,
            "claim_boundary":"File/schema/static geometry checks only. Camera registration and dynamics remain unverified."}
    (SIM_ROOT/"validation.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report))
    if issues:raise SystemExit(1)


if __name__=="__main__":validate()
