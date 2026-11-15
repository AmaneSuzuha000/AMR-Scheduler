"""Create separate metre-scale collision worlds from the canonical scene graph.

Equipment boxes are explicit synthetic obstacles with central service passages;
they are not CAD, surveyed assets, or the concurrent visual reconstruction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def tag(parent, element_name, text=None, **attrs):
    element = ET.SubElement(parent, element_name, **attrs)
    element.text = text
    return element


def world_root():
    root = ET.Element("sdf", version="1.6")
    world = tag(root, "world", name="amr_metric")
    tag(world, "gravity", "0 0 -9.81")
    physics = tag(world, "physics", name="ode", type="ode")
    tag(physics, "max_step_size", "0.001")
    tag(physics, "real_time_update_rate", "1000")
    ode = tag(physics, "ode")
    solver = tag(ode, "solver")
    tag(solver, "type", "quick")
    tag(solver, "iters", "50")
    for model in ("ground_plane", "sun"):
        tag(tag(world, "include"), "uri", "model://"+model)
    tag(world, "plugin", name="state", filename="libgazebo_ros_state.so")
    return root, world


def box(world, name, center, size, *, collision=True, yaw=0., color="0.35 0.45 0.55 1"):
    model = tag(world, "model", name=name)
    tag(model, "static", "true")
    tag(model, "pose", " ".join(map(str, [*center, 0, 0, yaw])))
    link = tag(model, "link", name="body")
    if collision:
        tag(tag(tag(link, "collision", name="solid"), "geometry"), "box")
        tag(link.find("collision/geometry/box"), "size", " ".join(map(str, size)))
    visual = tag(link, "visual", name="visual")
    tag(tag(tag(visual, "geometry"), "box"), "size", " ".join(map(str, size)))
    tag(tag(visual, "material"), "diffuse", color)


def build_scene(scene):
    if scene["units"]["length"] != "m":
        raise ValueError("metre scene required")
    root, world = world_root()
    obstacles = []
    for zone in scene["zones"]:
        if zone["kind"] == "traffic":
            continue
        xs, ys = zip(*zone["polygon"])
        left, right, bottom, top = min(xs), max(xs), min(ys), max(ys)
        free_left, free_right = (left+right)/2-1.1, (left+right)/2+1.1
        # Account for the canonical one-way return lane as well as the central
        # service lane. Its side branch must not be buried in equipment boxes.
        for lane in scene["lanes"]:
            half = lane["width_m"]/2
            for a,b in zip(lane["centerline"], lane["centerline"][1:]):
                if a[0] != b[0] and a[1] != b[1]:
                    raise ValueError("axis-aligned scene lanes required for envelope extrusion")
                xmin,xmax,ymin,ymax = min(a[0],b[0])-half,max(a[0],b[0])+half,min(a[1],b[1])-half,max(a[1],b[1])+half
                if xmax >= left and xmin <= right and ymax >= bottom and ymin <= top:
                    free_left, free_right = min(free_left,xmin), max(free_right,xmax)
        for suffix, lo, hi in (("west", left, free_left), ("east", free_right, right)):
            if hi <= lo:
                raise ValueError("zone cannot accommodate service passage")
            name = zone["id"]+"_"+suffix
            center, size = [(lo+hi)/2, (bottom+top)/2, .6], [hi-lo, top-bottom, 1.2]
            box(world, name, center, size)
            obstacles.append(dict(id=name, center=center, size=size, assumption="synthetic equipment envelope"))
    width, height = scene["bounds_m"]
    for name, center, size in (
        ("boundary_w", [-.1,height/2,1], [.2,height,2]),
        ("boundary_e", [width+.1,height/2,1], [.2,height,2]),
        ("boundary_s", [width/2,-.1,1], [width,.2,2]),
        ("boundary_n", [width/2,height+.1,1], [width,.2,2])):
        box(world, name, center, size)
        obstacles.append(dict(id=name, center=center, size=size, assumption="synthetic boundary"))
    for lane in scene["lanes"]:
        points = lane["centerline"]
        for index, (a,b) in enumerate(zip(points, points[1:])):
            length = math.dist(a,b)
            box(world, f"paint_{lane['id']}_{index}", [(a[0]+b[0])/2,(a[1]+b[1])/2,.001],
                [length,lane["width_m"],.002], collision=False,
                yaw=math.atan2(b[1]-a[1],b[0]-a[0]), color=".18 .22 .24 1")
    return root, dict(scene_id=scene["scene_id"], frame_id="map", scale=1., obstacles=obstacles,
                      scope="process-referenced synthetic workshop; no site survey or detailed equipment dynamics")


def calibration_world():
    root, world = world_root()
    for name, center, size in (
        ("front_wall", [3.1,0,.6], [.2,6,1.2]),
        ("left_wall", [0,3.1,.6], [6,.2,1.2]),
        ("right_wall", [0,-3.1,.6], [6,.2,1.2])):
        box(world, name, center, size)
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=Path(__file__).resolve().parents[2]/"scenes/rail_freight_workshop_v1.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root, manifest = build_scene(json.loads(args.scene.read_text(encoding="utf-8")))
    ET.indent(root)
    ET.ElementTree(root).write(args.out/"rail_workshop_metric.world", encoding="utf-8", xml_declaration=True)
    root = calibration_world()
    ET.indent(root)
    ET.ElementTree(root).write(args.out/"sensor_contact_metric.world", encoding="utf-8", xml_declaration=True)
    manifest["scene_sha256"] = hashlib.sha256(args.scene.read_bytes()).hexdigest()
    manifest["world_sha256"] = hashlib.sha256((args.out/"rail_workshop_metric.world").read_bytes()).hexdigest()
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("metric world built:", len(manifest["obstacles"]), "collision boxes")


if __name__ == "__main__":
    main()
