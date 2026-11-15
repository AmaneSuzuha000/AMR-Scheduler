"""Render true silhouette masks: one alpha pass per registered camera.

Why this exists: the scoring pipeline used to derive the render's foreground by
"difference from the median border colour, then Otsu". That is a proxy, and it
breaks exactly when the render stops being a flat backdrop. Measured 2026-11-11:
switching only the colour management moved inspection from 0.222 to 0.819
foreground fraction with the geometry untouched - the Otsu threshold had slid onto
the background gradient. Any score built on that proxy is unusable for comparing
render settings.

The alpha channel with film_transparent is not a proxy: it is the exact set of
pixels where geometry was hit. It cannot be gamed by brightness, and it depends
only on cameras and geometry, which is what the geometry-fidelity score should
measure.

Output: <out>/<view>.png RGBA, alpha 255 where geometry is present.
"""
import json
import os

import bpy
from mathutils import Vector

ROOT = r"E:\Remote\项目\AMR"
OUT = os.path.join(ROOT, "out", "visual_baseline_01", "masks")
BLEND = os.path.join(ROOT, "assets", "wheelset_workshop", "source", "wheelset_workshop.blend")
CAMS = {c["id"]: c for c in json.load(open(os.path.join(ROOT, "simulation", "wheelset_workshop", "cameras.json"), encoding="utf-8"))}
INDEX = json.load(open(os.path.join(ROOT, "visual_comparison", "wheelset_workshop_v1", "reference_index.json"), encoding="utf-8"))
VIEWS = [v["id"] for v in INDEX["views"]]

os.makedirs(OUT, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=BLEND)
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"   # alpha does not depend on shading
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"

camera_object = next(o for o in bpy.data.objects if o.type == "CAMERA")
scene.camera = camera_object

# Hide the ground slab. The reference frames' "foreground" is equipment seen against
# the floor, so a mask that includes the floor covers the whole frame (measured 0.974
# against a reference mean of 0.328) and says nothing. Excluding the slab makes the
# two masks mean the same thing: built structure.
hidden = []
for name in ("floor", "floor_base"):
    obj = bpy.data.objects.get(name)
    if obj is not None:
        obj.hide_render = True
        hidden.append(name)

written = []
for view_id in VIEWS:
    cam = CAMS[view_id]
    camera_object.location = Vector(cam["position"])
    camera_object.rotation_euler = (Vector(cam["target"]) - Vector(cam["position"])).to_track_quat("-Z", "Y").to_euler()
    camera_object.data.angle = cam["horizontal_fov"]
    path = os.path.join(OUT, f"{view_id}.png")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    written.append(view_id)

with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as handle:
    json.dump(
        {"views": written, "kind": "alpha_silhouette", "engine": "BLENDER_WORKBENCH",
         "hidden_objects": hidden,
         "definition": "alpha>0.5 where built structure is present, ground slab excluded"},
        handle, ensure_ascii=False, indent=1)
print("MASKS " + json.dumps({"views": len(written)}))
