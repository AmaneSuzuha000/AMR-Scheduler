"""Rasterize the metric world's actual static collision boxes for localization.

The resulting occupancy map shares world metres and box geometry. It is a known
synthetic map, not a mapping result, site survey, or representation of low paint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def collision_boxes(root):
    boxes=[]
    for model in root.findall("world/model"):
        if model.findtext("static") != "true":
            continue
        solids=model.findall("link/collision")
        if not solids:
            continue
        pose=list(map(float,model.findtext("pose", "0 0 0 0 0 0").split()))
        if any(abs(pose[k]) > 1e-12 for k in (3,4,5)):
            raise ValueError("only axis aligned collision boxes are supported")
        for collision in solids:
            size=collision.findtext("geometry/box/size")
            if size is None:
                raise ValueError("non-box static collision")
            if collision.find("pose") is not None or model.find("link/pose") is not None:
                raise ValueError("local collision/link transforms need explicit support")
            sx,sy,sz=map(float,size.split())
            if pose[2]+sz/2 < .35 or pose[2]-sz/2 > .45:
                continue  # occupancy at the configured ~0.4m laser plane
            boxes.append((pose[0]-sx/2,pose[1]-sy/2,pose[0]+sx/2,pose[1]+sy/2))
    return boxes


def rasterize(boxes, bounds, resolution=.05):
    xmin,ymin,xmax,ymax=map(float,bounds)
    if not all(math.isfinite(v) for v in (*bounds,resolution)) or resolution <= 0 or xmin>=xmax or ymin>=ymax:
        raise ValueError("finite positive grid geometry required")
    width,height=math.ceil((xmax-xmin)/resolution),math.ceil((ymax-ymin)/resolution)
    if width*height > 20_000_000:
        raise ValueError("occupancy map exceeds cell limit")
    pixels=bytearray([254])*(width*height)
    # Mark every cell intersecting a solid box; never round a narrow wall away.
    for left,bottom,right,top in boxes:
        i0=max(0,math.floor((left-xmin)/resolution))
        i1=min(width,math.ceil((right-xmin)/resolution))
        j0=max(0,math.floor((bottom-ymin)/resolution))
        j1=min(height,math.ceil((top-ymin)/resolution))
        if i1<=i0 or j1<=j0:
            continue
        for j in range(j0,j1):
            start=(height-1-j)*width+i0
            pixels[start:start+i1-i0]=bytes(i1-i0)
    return width,height,pixels


def write_map(world, out, bounds, resolution=.05):
    out.mkdir(parents=True,exist_ok=True)
    boxes=collision_boxes(ET.parse(world).getroot())
    width,height,pixels=rasterize(boxes,bounds,resolution)
    image=out/"map.pgm"
    image.write_bytes(f"P5\n{width} {height}\n255\n".encode()+pixels)
    (out/"map.yaml").write_text(f"image: map.pgm\nmode: trinary\nresolution: {resolution}\n"
        f"origin: [{bounds[0]}, {bounds[1]}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    (out/"manifest.json").write_text(json.dumps(dict(world_sha256=hashlib.sha256(world.read_bytes()).hexdigest(),
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),bounds=bounds,resolution=resolution,
        width=width,height=height,collision_boxes=len(boxes),occupied_cells=pixels.count(0),
        scope="known synthetic collision map at the laser plane; not SLAM or a surveyed site"),indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--world",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--bounds",type=float,nargs=4,required=True)
    args=parser.parse_args()
    write_map(args.world,args.out,args.bounds)


if __name__ == "__main__":
    main()
