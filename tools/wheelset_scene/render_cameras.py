"""Blender-side batch camera renderer.

Run as:  blender --background --python tools/wheelset_scene/render_cameras.py -- job.json

job.json:
{
  "blend": "path/to/scene.blend",
  "out_dir": "path/to/out",
  "engine": "BLENDER_EEVEE",
  "resolution": [640, 360],
  "samples": 8,
  "cameras": [{"id": "...", "position": [x,y,z], "target": [x,y,z], "fov": 0.92}]
}

The job file is the only input; nothing about the scene is assumed beyond a single
camera object being reused.  A manifest with per-image sha256 is written next to
the images so downstream scoring can prove which bytes it read.
"""
import hashlib
import json
import math
import os
import sys
import time

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("expected: blender -b -P render_cameras.py -- job.json")
    return argv[argv.index("--") + 1 :]


def look_at(camera_object, position, target):
    camera_object.location = Vector(position)
    direction = Vector(target) - Vector(position)
    rotation = direction.to_track_quat("-Z", "Y")
    camera_object.rotation_euler = rotation.to_euler()


def main():
    args = parse_args()
    if not args:
        raise SystemExit("missing job json path")
    job = json.loads(open(args[0], encoding="utf-8").read())

    bpy.ops.wm.open_mainfile(filepath=job["blend"])
    scene = bpy.context.scene
    scene.render.engine = job.get("engine", "BLENDER_EEVEE")
    width, height = job.get("resolution", [1280, 720])
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    samples = job.get("samples")
    if samples and scene.render.engine == "BLENDER_EEVEE":
        try:
            scene.eevee.taa_render_samples = int(samples)
        except Exception:
            pass

    camera_object = next((o for o in bpy.data.objects if o.type == "CAMERA"), None)
    if camera_object is None:
        raise SystemExit("no camera object in scene")
    scene.camera = camera_object

    out_dir = job["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    started = time.time()
    for spec in job["cameras"]:
        path = os.path.join(out_dir, spec["id"] + ".png")
        look_at(camera_object, spec["position"], spec["target"])
        camera_object.data.angle = float(spec["fov"])
        scene.render.filepath = path
        entry = {"id": spec["id"]}
        try:
            bpy.ops.render.render(write_still=True)
            entry["ok"] = os.path.exists(path)
            if entry["ok"]:
                payload = open(path, "rb").read()
                entry["sha256"] = hashlib.sha256(payload).hexdigest()
                entry["bytes"] = len(payload)
        except Exception as error:  # keep going: one bad pose must not stop the batch
            entry["ok"] = False
            entry["error"] = str(error)[:200]
        rows.append(entry)

    manifest = {
        "blend": job["blend"],
        "engine": scene.render.engine,
        "resolution": [int(width), int(height)],
        "samples": samples,
        "requested": len(job["cameras"]),
        "rendered": sum(1 for r in rows if r.get("ok")),
        "seconds": round(time.time() - started, 2),
        "rows": rows,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=1)
    print("RENDERCAM " + json.dumps({"rendered": manifest["rendered"], "seconds": manifest["seconds"]}))


main()
