"""Known-map raster geometry must preserve physical walls and free corridors."""
import json
from pathlib import Path

import pytest

from tools.live_bridge.metric_world import build_scene,calibration_world
from tools.live_bridge.sensor_map import collision_boxes,rasterize

ROOT=Path(__file__).resolve().parents[1]


def occupancy(grid,x,y,bounds,res=.05):
    width,height,pixels=grid
    col=int((x-bounds[0])/res)
    row=int((y-bounds[1])/res)
    return pixels[(height-1-row)*width+col]


def test_three_physical_sensor_walls_survive_rasterization_and_pgm_orientation():
    bounds=(-4.,-4.,4.,4.)
    boxes=collision_boxes(calibration_world())
    grid=rasterize(boxes,bounds)
    assert len(boxes)==3
    for x,y in ((3.1,0.),(0.,3.1),(0.,-3.1)):
        assert occupancy(grid,x,y,bounds)==0
    for x,y in ((0.,0.),(2.9,0.),(0.,2.9),(0.,-2.9)):
        assert occupancy(grid,x,y,bounds)==254


def test_metric_workshop_occupied_boxes_and_all_route_vertices_agree():
    scene=json.loads((ROOT/'scenes/rail_freight_workshop_v1.json').read_text(encoding='utf-8'))
    root,manifest=build_scene(scene)
    bounds=(-1.,-1.,141.,91.)
    grid=rasterize(collision_boxes(root),bounds)
    for obstacle in manifest['obstacles']:
        assert occupancy(grid,*obstacle['center'][:2],bounds)==0
    for lane in scene['lanes']:
        for point in lane['centerline']:
            assert occupancy(grid,*point,bounds)==254,(lane['id'],point)


@pytest.mark.parametrize('bounds,res', [((0,0,1,1),0),((0,0,0,1),.05),((0,0,1,1),float('nan'))])
def test_invalid_raster_geometry_is_rejected(bounds,res):
    with pytest.raises(ValueError):
        rasterize([],bounds,res)
