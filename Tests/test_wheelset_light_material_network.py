import math
from tools.wheelset_scene.light_material_network import transform,segment_box,audit,proposed_network


def test_clearance_checks_segment_interior_and_not_only_endpoints():
    box=[2,-.1,0,3,.1,1]
    assert segment_box((0,0),(5,0),box)==0
    assert abs(segment_box((0,1),(5,1),box)-.9)<1e-9
    assert abs(segment_box((0,0),(1,0),box)-1)<1e-9


def test_collision_transform_applies_rotation_before_translation():
    p=transform((2,0,1),(10,20,3,0,0,math.pi/2))
    assert all(abs(a-b)<1e-9 for a,b in zip(p,(10,22,4)))


def test_corridor_rejects_wall_crossing_and_floor_edge_overhang():
    scene=dict(bounds_m=[258,72],global_scene_scale=1,instances=[],boxes=[
        dict(id='north_boundary',size=[126,.22,1.6],center=[69,67.1,.8])])
    network=proposed_network();assert audit(network,scene)['passed']
    network['nodes'][0]['y']=65
    assert not audit(network,scene)['passed']
    network=proposed_network();network['nodes'][-1]['y']=71.8
    assert not audit(network,scene)['passed']
