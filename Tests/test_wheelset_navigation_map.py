import math
import xml.etree.ElementTree as ET

import pytest

from tools.wheelset_scene.workshop_navigation import (
    LANDMARKS, box_slice, collision_polygons, landmark_clearance, rasterize,
    configure_review_camera, build_bundle, build_argument_parser,
    AMCL_DEFAULT, validate_amcl_cost,
    crop_localization_map,
)
from tools.wheelset_scene.light_material_network import proposed_network


def test_omitting_unused_camera_preserves_all_collision_models():
    world = ET.fromstring('''<world><model name="review_camera"><static>true</static>
      <link><sensor type="camera"/></link></model><model name="wall"><link>
      <collision name="wall_collision"/></link></model></world>''')
    expected = ET.tostring(world.find("model[@name='wall']"))
    configure_review_camera(world, False)
    assert world.find("model[@name='review_camera']") is None
    assert ET.tostring(world.find("model[@name='wall']")) == expected


def test_localization_subset_keeps_pixel_rows_and_world_origin_exact():
    pixels = bytes(range(100))
    nodes = [dict(id='a', x=4., y=5.), dict(id='b', x=6., y=7.)]
    crop = crop_localization_map(10, 10, pixels, 1., nodes, 1.)
    assert crop['source_cells_xyxy'] == [3, 4, 7, 8]
    assert crop['origin'] == [3., 4., 0.]
    assert crop['width'] == crop['height'] == 4
    assert crop['pixels'] == bytes([23,24,25,26,33,34,35,36,43,44,45,46,53,54,55,56])
    # Global and local cells refer to the same world coordinates and values.
    for row in range(crop['height']):
        for column in range(crop['width']):
            assert crop['pixels'][row*4+column] == pixels[(row+2)*10+column+3]


def test_localization_subset_clips_only_at_the_existing_full_map_boundary():
    pixels = bytes(range(100))
    nodes = [dict(id='a', x=0., y=0.), dict(id='b', x=10., y=10.)]
    crop = crop_localization_map(10, 10, pixels, 1., nodes, 12.)
    assert crop['source_cells_xyxy'] == [0, 0, 10, 10]
    assert crop['pixels'] == pixels and crop['origin'] == [0., 0., 0.]


def test_localization_subset_rounds_outward_and_covers_all_route_endpoints():
    nodes = [dict(id='a', x=4.13, y=4.27), dict(id='b', x=6.28, y=6.33)]
    crop = crop_localization_map(100, 100, bytes(10000), .1, nodes, 1.)
    x0, y0, x1, y1 = crop['source_cells_xyxy']
    for node in nodes:
        assert x0*.1 <= node['x']-1. and node['x']+1. <= x1*.1
        assert y0*.1 <= node['y']-1. and node['y']+1. <= y1*.1
    assert crop['origin'] == [x0*.1, y0*.1, 0.]


@pytest.mark.parametrize('margin', [0., 11.99, float('nan'), float('inf')])
def test_insufficient_localization_support_refuses_before_any_output(tmp_path, margin):
    with pytest.raises(ValueError, match='at least 12'):
        build_bundle(tmp_path/'refused', localization_map_margin_m=margin)
    assert not (tmp_path/'refused').exists()


def test_station_refinement_is_explicit_and_requires_its_declared_map_and_docks(tmp_path):
    defaults=build_argument_parser().parse_args(['--output','unused'])
    assert not defaults.station_scan_refinement
    with pytest.raises(ValueError,match='station refinement requires'):
        build_bundle(tmp_path/'refused_station',station_scan_refinement=True)
    assert not (tmp_path/'refused_station').exists()


def test_platform_lift_requires_contact_geometry_and_refuses_before_output(tmp_path):
    with pytest.raises(ValueError,match='platform lift requires'):
        build_bundle(tmp_path/'refused_lift',contact_bin_platform_lift_m=.02)
    assert not (tmp_path/'refused_lift').exists()


def test_localization_subset_rejects_nodes_outside_the_full_map():
    with pytest.raises(ValueError, match='outside'):
        crop_localization_map(10, 10, bytes(100), 1., [dict(id='bad', x=11., y=5.)], 2.)


def test_camera_removal_rejects_a_model_with_collision():
    world = ET.fromstring('<world><model name="review_camera"><link><collision/></link></model></world>')
    with pytest.raises(ValueError): configure_review_camera(world, False)


def test_slab_filters_floor_and_retains_rotated_thin_wall():
    assert box_slice([10, 10, .1], [[0, 0, -.1, 0, 0, 0]]) == []
    polygon = box_slice([2, .02, 1], [[3, 4, .5, 0, 0, math.pi/4]])
    assert len(polygon) == 4
    w, h, pixels = rasterize([polygon], [0, 0, 6, 8], .05)
    assert pixels[(h-1-int(4/.05))*w+int(3/.05)] == 0
    # A point inside the enclosing AABB, far from the diagonal wall, stays free.
    assert pixels[(h-1-int(4.5/.05))*w+int(2.5/.05)] == 254


def test_local_link_collision_and_include_poses_are_composed(tmp_path):
    model = tmp_path/'models/unit'; model.mkdir(parents=True)
    (model/'model.sdf').write_text('''<sdf><model name="unit"><static>true</static>
      <pose>1 0 0 0 0 0</pose><link name="l"><pose>0 2 0 0 0 0</pose>
      <collision name="c"><pose>1 0 .5 0 0 0</pose><geometry><box><size>1 1 1</size>
      </box></geometry></collision></link></model></sdf>''')
    world = tmp_path/'test.world'
    world.write_text('''<sdf><world><include><uri>model://unit</uri><name>placed</name>
      <pose>10 20 0 0 0 1.5707963267948966</pose></include></world></sdf>''')
    polygons, sources, skipped = collision_polygons(world, tmp_path/'models')
    assert not skipped and len(sources) == 2
    xs, ys = zip(*polygons[0]['polygon'])
    assert (min(xs), max(xs), min(ys), max(ys)) == pytest.approx((7.5, 8.5, 21.5, 22.5))
    root = ET.parse(model/'model.sdf')
    root.find('.//pose').set('relative_to', 'unknown_frame')
    root.write(model/'model.sdf')
    with pytest.raises(ValueError, match='relative_to'):
        collision_polygons(world, tmp_path/'models')


def test_pitch_slices_at_laser_height_instead_of_projecting_whole_box():
    polygon = box_slice([4, 1, .1], [[0, 0, .32, 0, math.pi/4, 0]], slab=(.32, .32))
    assert max(p[0] for p in polygon) < .08
    assert min(p[0] for p in polygon) > -.08


def test_added_landmarks_preserve_all_reserved_road_margins():
    assert landmark_clearance(proposed_network(), LANDMARKS)['passed']
    obstacle = dict(id='blocks_bay', center=[20, 71, .35], size=[.1, .1, .7])
    assert not landmark_clearance(proposed_network(), [obstacle])['passed']


def test_grid_rejects_unbounded_or_invalid_geometry():
    with pytest.raises(ValueError):
        rasterize([], [0, 0, math.inf, 1])
    with pytest.raises(ValueError):
        rasterize([], [0, 0, 10000, 10000], .01)


def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    for start, end in zip(polygon, polygon[1:]+polygon[:1]):
        if (start[1] > y) != (end[1] > y):
            crossing = (end[0]-start[0])*(y-start[1])/(end[1]-start[1])+start[0]
            if x < crossing:
                inside = not inside
    return inside


def marked_cells(pixels, width, height):
    return {(i, j) for j in range(height) for i in range(width)
            if pixels[(height-1-j)*width+i] == 0}


@pytest.mark.parametrize('resolution', [.05, .02])
def test_area_rule_covers_every_cell_the_polygon_touches(resolution):
    """The historical rule marks a cell when the polygon covers any of it."""
    polygon = [(1.9493, 1.0121), (2.0507, 1.0121), (2.0507, 1.4879), (1.9493, 1.4879)]
    bounds = [0., 0., 4., 2.]
    width, height, pixels = rasterize([polygon], bounds, resolution)
    marked = marked_cells(pixels, width, height)
    samples = [(resolution*(i+.5)/12, resolution*(j+.5)/12) for i in range(2) for j in range(2)]
    for ix in range(width):
        for jy in range(height):
            centre = (0.+(ix+.5)*resolution, 0.+(jy+.5)*resolution)
            inside = any(point_in_polygon((centre[0]+dx, centre[1]+dy), polygon) for dx, dy in samples)
            if inside:
                assert (ix, jy) in marked


@pytest.mark.parametrize('resolution', [.05, .02])
def test_centre_grid_is_a_subset_of_the_area_grid(resolution):
    polygon = [(1.9493, 1.0121), (2.0507, 1.0121), (2.0507, 1.4879), (1.9493, 1.4879)]
    bounds = [0., 0., 4., 2.]
    width, height, area = rasterize([polygon], bounds, resolution)
    _, _, centre = rasterize([polygon], bounds, resolution, rule='centre')
    assert marked_cells(centre, width, height) <= marked_cells(area, width, height)
    assert marked_cells(centre, width, height) < marked_cells(area, width, height)


def test_centre_rule_marks_only_cells_whose_centre_is_covered():
    polygon = [(1.93, 1.0), (2.07, 1.0), (2.07, 2.0), (1.93, 2.0)]
    bounds = [0., 0., 4., 4.]
    width, height, pixels = rasterize([polygon], bounds, .05, rule='centre')
    marked = marked_cells(pixels, width, height)
    assert marked
    for i, j in marked:
        assert point_in_polygon((0.+ (i+.5)*.05, 0.+(j+.5)*.05), polygon)


def test_station_face_error_is_bounded_by_the_declared_rule_and_resolution():
    """Wall face error is what the estimator inherits, so it is measured.

    The face sits just inside a 0.05m cell, which is the worst case for the
    historical area rule: the grid claims the whole cell, so the recorded face
    moves almost a full cell outwards.  The centre rule bounds that by half a
    cell, and the face error is what an estimator matching scans against this
    map inherits as a pose offset.
    """
    face = 20.049
    wall = [(face, 70.0), (face+.2, 70.0), (face+.2, 72.0), (face, 72.0)]
    bounds = [18., 68., 24., 74.]
    measured = {}
    # The area rule always starts the wall at or inside its true face, so it can
    # push the recorded face out by almost a whole cell. The centre rule may
    # start up to half a cell inside, and is bounded by half a cell outside.
    limits = {'area': (0., 1.), 'centre': (-.5, .5)}
    for rule, (low, high) in limits.items():
        for resolution in (.05, .02):
            width, height, pixels = rasterize([wall], bounds, resolution, rule=rule)
            starts = [18.+i*resolution for i, _ in marked_cells(pixels, width, height)]
            error = face-min(starts)
            assert low*resolution-1e-9 <= error <= high*resolution+1e-9, (rule, resolution, error)
            measured[(rule, resolution)] = error
    assert measured[('area', .05)] > measured[('centre', .05)]
    assert measured[('area', .05)] > measured[('area', .02)]


@pytest.mark.parametrize('rule', ['overlap', '', None])
def test_unknown_occupancy_rule_is_refused(rule):
    with pytest.raises(ValueError, match='occupancy rule'):
        rasterize([], [0, 0, 4, 4], .05, rule=rule)


@pytest.mark.parametrize('resolution', [0., -.01, math.nan, math.inf, True, '0.05'])
def test_unvalidated_map_resolution_is_refused(resolution):
    with pytest.raises(ValueError):
        rasterize([], [0, 0, 4, 4], resolution)


@pytest.mark.parametrize('cells', [0, -1, math.inf, True])
def test_map_cell_limit_must_be_positive_and_numeric(cells):
    with pytest.raises(ValueError):
        rasterize([], [0, 0, 4, 4], .01, max_cells=cells)


def test_map_cell_limit_refuses_an_oversized_grid():
    with pytest.raises(ValueError, match='cell limit'):
        rasterize([], [0, 0, 4, 4], .001, max_cells=1000)


@pytest.mark.parametrize('kwargs, message', [
    (dict(map_rule='overlap'), 'occupancy rule'),
    (dict(resolution=0.), 'resolution'),
    (dict(resolution=math.nan), 'resolution'),
    (dict(map_cell_limit=0), 'cell limit'),
])
def test_bundle_refuses_a_bad_map_declaration_before_creating_output(tmp_path, kwargs, message):
    output = tmp_path/'not-created'
    with pytest.raises(ValueError, match=message):
        build_bundle(output, **kwargs)
    assert not output.exists()


def test_bundle_records_the_map_rule_and_resolution(tmp_path):
    """A tighter grid is only admissible if the manifest states what it is."""
    output = tmp_path/'contact'
    manifest = build_bundle(output, review_camera=False)
    assert (manifest['map_rule'], manifest['resolution_m']) == ('area', .05)
    assert 'resolution: 0.05' in (output/'map.yaml').read_text()
    assert manifest['occupied_cells'] == (output/'map.pgm').read_bytes().count(0)
    width, height = manifest['width'], manifest['height']
    assert len((output/'map.pgm').read_bytes()) == width*height+len(f'P5\n{width} {height}\n255\n'.encode())
    tighter = tmp_path/'tight'
    manifest = build_bundle(tighter, review_camera=False, map_rule='centre', resolution=.02)
    assert (manifest['map_rule'], manifest['resolution_m']) == ('centre', .02)
    assert (manifest['width'], manifest['height']) == (
        math.ceil(manifest['bounds_m'][2]/.02), math.ceil(manifest['bounds_m'][3]/.02))
    assert 'resolution: 0.02' in (tighter/'map.yaml').read_text()
    assert manifest['occupied_cells'] > 0


def test_map_declaration_defaults_and_cli_are_the_same_surface():
    parser = build_argument_parser()
    default = parser.parse_args(['--output', 'bundle'])
    assert (default.map_rule, default.resolution, default.max_cells) == ('area', .05, 60_000_000)
    declared = parser.parse_args(['--output', 'bundle', '--map-rule', 'centre',
                                  '--resolution', '.02', '--max-cells', '50000000'])
    assert (declared.map_rule, declared.resolution, declared.max_cells) == ('centre', .02, 50_000_000)
    with pytest.raises(SystemExit):
        parser.parse_args(['--output', 'bundle', '--map-rule', 'overlap'])


def test_main_passes_the_map_declaration_into_the_build(tmp_path, monkeypatch):
    import sys
    from tools.wheelset_scene import workshop_navigation as subject
    seen = {}

    def fake_build(output, **kwargs):
        seen.update(kwargs)
        seen['output'] = output
        return {key: None for key in ('status', 'polygons', 'width', 'height',
                                      'occupied_cells', 'resolution_m', 'map_rule')}

    monkeypatch.setattr(subject, 'build_bundle', fake_build)
    monkeypatch.setattr(sys, 'argv', ['workshop_navigation', '--output', str(tmp_path/'b'),
                                      '--map-rule', 'centre', '--resolution', '.02',
                                      '--max-cells', '7000000'])
    subject.main()
    assert (seen['map_rule'], seen['resolution'], seen['map_cell_limit']) == ('centre', .02, 7_000_000)


def test_estimator_cost_defaults_pass_the_cli_surface_through_unchanged():
    """A build with no estimator override is the measured configuration."""
    import yaml
    parser = build_argument_parser()
    default = parser.parse_args(['--output', 'bundle'])
    assert dict(max_beams=default.amcl_max_beams, min_particles=default.amcl_min_particles,
                max_particles=default.amcl_max_particles,
                update_min_d=default.amcl_update_min_d,
                update_min_a=default.amcl_update_min_a) == AMCL_DEFAULT
    assert AMCL_DEFAULT == dict(max_beams=1080, min_particles=5000, max_particles=20000,
                                update_min_d=.005, update_min_a=.005)
    declared = parser.parse_args(['--output', 'bundle', '--amcl-min-particles', '500',
                                  '--amcl-max-particles', '2000', '--amcl-max-beams', '120',
                                  '--amcl-update-min-d', '.05', '--amcl-update-min-a', '.1'])
    assert (declared.amcl_min_particles, declared.amcl_max_particles, declared.amcl_max_beams) == (500, 2000, 120)


def test_estimator_cost_override_moves_only_the_estimator_and_is_recorded(tmp_path):
    """An A/B changes one declared number; the map and the rest are untouched."""
    import yaml
    cheap = dict(AMCL_DEFAULT, max_beams=120, min_particles=500, max_particles=2000)
    original = build_bundle(tmp_path/'original', review_camera=False)
    changed = build_bundle(tmp_path/'changed', review_camera=False, amcl=cheap)
    assert original['amcl_cost'] == AMCL_DEFAULT
    assert changed['amcl_cost'] == cheap
    assert changed['amcl_min_particles'] == 500 and changed['amcl_max_particles'] == 2000
    assert changed['amcl_max_beams'] == 120
    assert (tmp_path/'original/map.pgm').read_bytes() == (tmp_path/'changed/map.pgm').read_bytes()
    assert (tmp_path/'original/navigation.world').read_bytes() == (tmp_path/'changed/navigation.world').read_bytes()
    for robot in changed['robot_configs']:
        original_parameters = yaml.safe_load(
            (tmp_path/'original'/(robot+'_localization.yaml')).read_text())[robot]['amcl']['ros__parameters']
        parameters = yaml.safe_load(
            (tmp_path/'changed'/(robot+'_localization.yaml')).read_text())[robot]['amcl']['ros__parameters']
        assert parameters['min_particles'] == 500 and parameters['max_particles'] == 2000
        assert parameters['max_beams'] == 120
        assert parameters['min_particles'] != original_parameters['min_particles']
        # The declared estimator changes exactly the five declared keys.
        for key in ('laser_model_type', 'do_beamskip', 'robot_model_type'):
            assert parameters[key] == original_parameters[key]


def test_estimator_cost_rejects_undeclared_or_inverted_values(tmp_path):
    """An undeclared field set or an inverted band must not build a bundle."""
    with pytest.raises(ValueError):
        validate_amcl_cost(dict(AMCL_DEFAULT, extra=1))
    with pytest.raises(ValueError):
        validate_amcl_cost({k: v for k, v in AMCL_DEFAULT.items() if k != 'max_beams'})
    with pytest.raises(ValueError):
        validate_amcl_cost(dict(AMCL_DEFAULT, min_particles=20000, max_particles=5000))
    for name, value in (('max_beams', 0), ('min_particles', -1), ('max_particles', 1.5)):
        with pytest.raises(ValueError):
            validate_amcl_cost(dict(AMCL_DEFAULT, **{name: value}))
    for name in ('update_min_d', 'update_min_a'):
        for value in (0, -1e-3, float('inf'), float('nan')):
            with pytest.raises(ValueError):
                validate_amcl_cost(dict(AMCL_DEFAULT, **{name: value}))
    with pytest.raises(ValueError):
        build_bundle(tmp_path/'refused', review_camera=False, amcl=dict(AMCL_DEFAULT, max_beams=0))
    assert not (tmp_path/'refused').exists()


def test_main_passes_the_estimator_cost_into_the_build(tmp_path, monkeypatch):
    import sys
    from tools.wheelset_scene import workshop_navigation as subject
    seen = {}

    def fake_build(output, **kwargs):
        seen.clear()
        seen.update(kwargs)
        seen['output'] = output
        return {key: None for key in ('status', 'polygons', 'width', 'height',
                                      'occupied_cells', 'resolution_m', 'map_rule')}

    monkeypatch.setattr(subject, 'build_bundle', fake_build)
    monkeypatch.setattr(sys, 'argv', ['workshop_navigation', '--output', str(tmp_path/'b'),
                                      '--amcl-min-particles', '500', '--amcl-max-particles', '2000',
                                      '--amcl-max-beams', '120', '--amcl-update-min-d', '.05'])
    subject.main()
    assert seen['amcl'] == dict(max_beams=120, min_particles=500, max_particles=2000,
                                update_min_d=.05, update_min_a=.005)
    monkeypatch.setattr(sys, 'argv', ['workshop_navigation', '--output', str(tmp_path/'c')])
    subject.main()
    assert seen['amcl'] == AMCL_DEFAULT
