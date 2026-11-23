"""Build a separate sensor-navigation world and its matching known static map.

The frozen video reconstruction is read only. Optional, explicitly synthetic
landmarks make the long straight service corridor observable along its length.
No Gazebo truth is fed to the estimator or Nav2.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

from .build import ROOT, SCENE, box_model, camera_pose, numbers, save_xml
from .evidence import build_fingerprint
from .light_material_network import audit, pose, proposed_network, segment_box, transform
from tools.live_bridge.fleet_config import robot_configs


LANDMARKS = [dict(id=f'navigation_landmark_{i:02d}', center=[x, 71.92, .35],
                  size=[width, .08, .7], material='cyan')
             for i, (x, width) in enumerate([(4., .24), (10., .36), (16., .48),
                 (24., .30), (30., .44), (37., .26), (44., .40), (51., .32),
                 (58., .48), (65., .28)], 1)]
LANDMARKS += [dict(id='navigation_bay_landmark_'+name, center=[x, 71.92, .35],
                   size=[.60, .08, .7], material='cyan')
              for name, x in [('west', 20.), ('east', 40.)]]
SLICE = (.38, .44)  # Base axle .08m + LiDAR joint .32m, with settling tolerance.
# At the two bays, the narrow northern landmarks cover only about fifteen of
# 1080 rays.  Uniform 120-beam decimation can discard all of them, leaving a
# straight wall with little longitudinal information.  Preserve the existing
# sensor resolution in this scene's estimator, without changing shared tuning.
AMCL_MAX_BEAMS = 1080
# The estimator starts from a deliberately perturbed spawn prior and has to
# correct it on a long flat corridor.  With the shared 500/2000 particle band
# the corrective pass stops up to ~3cm short of the true carrier pose - larger
# than the 2.5cm contact capture gate - simply because the posterior is
# undersampled.  A wider band and a finer update cadence make the same prior
# converge to 1.5cm without touching the prior, the map or any acceptance gate.
AMCL_MIN_PARTICLES = 5000
AMCL_MAX_PARTICLES = 20000
AMCL_UPDATE_MIN_DISTANCE_M = .005
AMCL_UPDATE_MIN_ANGLE_RAD = .005
# Candidate 34: the estimator's cost is a declared build parameter rather than
# a set of frozen constants.  A heavier posterior competes for the same cores
# as the physics server, so the cost has to be selectable to run a single
# variable A/B.  The defaults below are exactly the numbers measured for the
# 1.5cm corrective residual, so a build that passes no override behaves like
# the frozen candidate 30/31/33 bundles.  Every entry is validated against the
# declared field set and recorded in the manifest.
AMCL_DEFAULT = dict(max_beams=AMCL_MAX_BEAMS, min_particles=AMCL_MIN_PARTICLES,
                    max_particles=AMCL_MAX_PARTICLES,
                    update_min_d=AMCL_UPDATE_MIN_DISTANCE_M,
                    update_min_a=AMCL_UPDATE_MIN_ANGLE_RAD)
AMCL_INTEGER_FIELDS = ('max_beams', 'min_particles', 'max_particles')
AMCL_FLOAT_FIELDS = ('update_min_d', 'update_min_a')


def validate_amcl_cost(amcl):
    """Reject an undeclared, non-finite or inverted estimator cost."""
    amcl = dict(amcl)
    if set(amcl) != set(AMCL_DEFAULT):
        raise ValueError('AMCL cost must declare exactly ' + ', '.join(sorted(AMCL_DEFAULT)))
    for name in AMCL_INTEGER_FIELDS:
        if type(amcl[name]) is not int or amcl[name] <= 0:
            raise ValueError('AMCL '+name+' must be a positive integer')
    for name in AMCL_FLOAT_FIELDS:
        value = amcl[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('AMCL '+name+' must be a positive finite number')
    if amcl['min_particles'] > amcl['max_particles']:
        raise ValueError('AMCL particle band must not be inverted')
    return amcl
# ODE's hard Coulomb lateral contact pinned the slightly more heavily loaded
# front axle while both wheel pairs were driven. A force-dependent lateral
# compliance represents compliant tyres rather than an axle locked sideways.
# This is an explicit synthetic contact model, not a measured SCOUT parameter.
LATERAL_SLIP_M_PER_NS = .1


def convex_hull(points):
    points = sorted(set(points))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    if len(points) < 3:
        return points
    low, high = [], []
    for sequence, result in ((points, low), (reversed(points), high)):
        for p in sequence:
            while len(result) >= 2 and cross(result[-2], result[-1], p) <= 0:
                result.pop()
            result.append(p)
    return low[:-1]+high[:-1]


def box_slice(size, transforms, slab=SLICE):
    """Project the intersection of a transformed box and a horizontal slab."""
    if len(size) != 3 or not all(math.isfinite(v) and v > 0 for v in size):
        raise ValueError('positive finite box dimensions required')
    if len(slab) != 2 or not all(math.isfinite(v) for v in slab) or slab[0] > slab[1]:
        raise ValueError('finite ordered laser slab required')
    keys = list(itertools.product((-1, 1), repeat=3))
    corners = {}
    for key in keys:
        p = tuple(s*v/2 for s, v in zip(key, size))
        for t in transforms:
            p = transform(p, t)
        if not all(math.isfinite(v) for v in p):
            raise ValueError('finite box transform required')
        corners[key] = p
    points = [p[:2] for p in corners.values() if slab[0] <= p[2] <= slab[1]]
    for a, b in itertools.combinations(keys, 2):
        if sum(x != y for x, y in zip(a, b)) != 1:
            continue
        p, q = corners[a], corners[b]
        for z in slab:
            if abs(q[2]-p[2]) > 1e-14:
                fraction = (z-p[2])/(q[2]-p[2])
                if 0 <= fraction <= 1:
                    points.append(tuple(p[d]+fraction*(q[d]-p[d]) for d in (0, 1)))
    return convex_hull(points)


def collision_polygons(world_path, model_root, slab=SLICE):
    """Resolve model includes and local poses; unsupported geometry fails closed."""
    sources = {}
    polygons, skipped = [], []
    def read(path):
        sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        return ET.parse(path).getroot()
    def checked_pose(element):
        local = element.find('pose')
        if local is not None and local.get('relative_to'):
            raise ValueError('relative_to frame needs explicit resolution')
        values = pose(element)
        if len(values) != 6 or not all(math.isfinite(v) for v in values):
            raise ValueError('finite six-element pose required')
        return values
    def model(element, outer, identity):
        if element.findtext('static') != 'true':
            skipped.append(identity)
            return
        model_transforms = [checked_pose(element), *outer]
        if element.findall('model') or element.findall('include'):
            raise ValueError('nested models require explicit static/frame handling')
        for link in element.findall('link'):
            for collision in link.findall('collision'):
                size = collision.findtext('geometry/box/size')
                if size is None:
                    raise ValueError('uncovered static collision: '+identity)
                polygon = box_slice(list(map(float, size.split())),
                    [checked_pose(collision), checked_pose(link), *model_transforms], slab)
                if len(polygon) >= 3:
                    polygons.append(dict(id=identity+'/'+link.get('name', '')+'/'+collision.get('name', ''),
                                         polygon=polygon))
    root = read(world_path)
    world = root.find('world')
    if world is None:
        raise ValueError('SDF world required')
    for child in world.findall('model'):
        model(child, [], child.get('name', ''))
    for include in world.findall('include'):
        uri = include.findtext('uri', '')
        name = uri.removeprefix('model://')
        if not uri.startswith('model://') or Path(name).name != name or name in ('', '.', '..'):
            raise ValueError('one local model directory required: '+uri)
        source = model_root/name/'model.sdf'
        element = read(source).find('model')
        if element is None:
            raise ValueError('included model missing')
        model(element, [checked_pose(include)], include.findtext('name', name))
    return polygons, sources, skipped


def clip_y(polygon, y, keep_above):
    result = []
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        inside_a = a[1] >= y if keep_above else a[1] <= y
        inside_b = b[1] >= y if keep_above else b[1] <= y
        if inside_a:
            result.append(a)
        if inside_a != inside_b:
            fraction = (y-a[1])/(b[1]-a[1])
            result.append((a[0]+fraction*(b[0]-a[0]), y))
    return result


MAP_RULES = ('area', 'centre')


def rasterize(polygons, bounds, resolution=.05, rule='area', max_cells=60_000_000):
    """Rasterise the collision polygons into the ROS map convention.

    Two occupancy rules are offered because they differ by up to a full cell
    on the *feature faces that a lidar actually measures*, and the estimator
    inherits that error:

    ``area`` (the original behaviour) marks a cell when the polygon covers any
    part of it, so a face can sit up to one whole cell outside the true face.
    ``centre`` marks a cell only when its centre is inside the polygon, so the
    error is bounded by half a cell.

    The same reasoning applies to the resolution: at a station whose
    coordinates are an exact multiple of the cell size the two rules coincide,
    while an arbitrary face inherits up to a cell of error at 0.05m against a
    2.5cm delivery gate.  A caller therefore chooses both explicitly and the
    choice is recorded in the manifest.

    The default keeps the historical map byte-identical; a caller that wants
    the tighter bound asks for it explicitly and records the choice.
    """
    xmin, ymin, xmax, ymax = bounds
    if (type(resolution) not in (int, float) or type(max_cells) not in (int, float)
            or not all(math.isfinite(v) for v in (*bounds, resolution)) or resolution <= 0
            or xmin >= xmax or ymin >= ymax):
        raise ValueError('finite positive map geometry required')
    if rule not in MAP_RULES:
        raise ValueError('unknown map occupancy rule: '+str(rule))
    if not math.isfinite(max_cells) or max_cells <= 0:
        raise ValueError('positive finite cell limit required')
    width, height = math.ceil((xmax-xmin)/resolution), math.ceil((ymax-ymin)/resolution)
    if width*height > max_cells:
        raise ValueError('map exceeds cell limit: %d > %d' % (width*height, max_cells))
    pixels = bytearray([254])*(width*height)
    for polygon in polygons:
        j0 = max(0, math.floor((min(p[1] for p in polygon)-ymin)/resolution))
        j1 = min(height, math.ceil((max(p[1] for p in polygon)-ymin)/resolution))
        for j in range(j0, j1):
            if rule == 'area':
                low, high = ymin+j*resolution, ymin+(j+1)*resolution
                band = clip_y(clip_y(polygon, low, True), high, False)
                if not band:
                    continue
                i0 = max(0, math.floor((min(p[0] for p in band)-xmin)/resolution))
                i1 = min(width, math.ceil((max(p[0] for p in band)-xmin)/resolution))
            else:
                # A hair-thin band at the row centre gives the polygon's own
                # cross-section there, which is what the centre rule tests.
                middle = ymin+(j+.5)*resolution
                band = clip_y(clip_y(polygon, middle-.5e-9, True), middle+.5e-9, False)
                if not band:
                    continue
                i0 = max(0, math.ceil((min(p[0] for p in band)-xmin)/resolution-.5))
                i1 = min(width, math.floor((max(p[0] for p in band)-xmin)/resolution-.5)+1)
            if i1 > i0:
                start = (height-1-j)*width+i0
                pixels[start:start+i1-i0] = bytes(i1-i0)
    return width, height, pixels


def crop_localization_map(width, height, pixels, resolution, nodes, margin_m):
    """Extract exact full-map cells around every declared route endpoint.

    PGM rows run north to south; the ROS map origin is at the southwest cell.
    Straight graph edges and their sensor/distance-field margin lie within
    this rectangle. The complete global map and world remain in the bundle.
    """
    if (type(width) is not int or type(height) is not int or width < 1 or height < 1
            or len(pixels) != width*height or not nodes
            or not math.isfinite(resolution) or resolution <= 0
            or not math.isfinite(margin_m) or margin_m <= 0):
        raise ValueError('valid raster, declared route nodes and finite positive margin required')
    points = [(item['x'], item['y']) for item in nodes]
    if any(not math.isfinite(x) or not math.isfinite(y)
           or not 0 <= x <= width*resolution or not 0 <= y <= height*resolution
           for x, y in points):
        raise ValueError('route node outside the full localization map')
    x0 = max(0, math.floor((min(x for x, _ in points)-margin_m)/resolution))
    x1 = min(width, math.ceil((max(x for x, _ in points)+margin_m)/resolution))
    y0 = max(0, math.floor((min(y for _, y in points)-margin_m)/resolution))
    y1 = min(height, math.ceil((max(y for _, y in points)+margin_m)/resolution))
    cropped = b''.join(bytes(pixels[row*width+x0:row*width+x1])
                       for row in range(height-y1, height-y0))
    return dict(width=x1-x0, height=y1-y0, pixels=cropped,
                origin=[x0*resolution, y0*resolution, 0.],
                source_cells_xyxy=[x0, y0, x1, y1],
                margin_m=margin_m, route_node_ids=[item['id'] for item in nodes])


def landmark_clearance(layout, landmarks):
    nodes = {n['id']: (n['x'], n['y']) for n in layout['nodes']}
    radius = layout['robot_enclosing_radius_m']+layout['safety_margin_m']
    checks = []
    for item in landmarks:
        x, y, z = item['center']; sx, sy, sz = item['size']
        bounds = [x-sx/2, y-sy/2, z-sz/2, x+sx/2, y+sy/2, z+sz/2]
        for lane in layout['lanes']:
            clearance = segment_box(nodes[lane['src']], nodes[lane['dst']], bounds)
            checks.append(dict(landmark=item['id'], lane=lane['id'], clearance_m=clearance,
                               required_m=radius, passed=clearance >= radius))
    return dict(passed=all(c['passed'] for c in checks), checks=checks)


def configure_review_camera(world, enabled):
    camera = world.find("model[@name='review_camera']")
    if camera is None or camera.findall('.//collision'):
        raise ValueError('one auxiliary camera without collision geometry required')
    if not enabled:
        world.remove(camera)
        return
    camera.find('pose').text = numbers(camera_pose(dict(position=[32, 56, 17], target=[32, 69.5, 0])))
    camera.find('link/sensor/update_rate').text = '.5'
    camera.find('link/sensor/camera/image/width').text = '1280'
    camera.find('link/sensor/camera/image/height').text = '720'


def configure_island_solver(world, threads):
    """Parallelize independent ODE islands without changing solver accuracy."""
    if type(threads) is not int or not 0 <= threads <= 16:
        raise ValueError('ODE island threads must be an integer from 0 to 16')
    if not threads:
        return
    solver = world.find('physics/ode/solver')
    if solver is None or solver.findtext('type') != 'quick' or solver.findtext('iters') != '400':
        raise ValueError('the explicit 400-iteration workshop ODE solver is required')
    if solver.find('island_threads') is not None:
        raise ValueError('island thread setting already exists')
    ET.SubElement(solver, 'island_threads').text = str(threads)


def build_bundle(output, *, landmarks=True, resolution=.05, map_rule='area', map_cell_limit=60_000_000,
                 fault_relay=False, whole_line_axis_build=None, review_camera=True, physics_island_threads=0,
                 contact_bin_axis_build=None, contact_bin_integral_gain=0., contact_bin_direction='lateral',
                 physics_step_s=.002, position_encoder=False, amcl=None,
                 localization_map_margin_m=None, station_scan_refinement=False,
                 contact_bin_platform_lift_m=None):
    if (type(contact_bin_integral_gain) not in (int, float) or not math.isfinite(contact_bin_integral_gain)
            or contact_bin_integral_gain < 0 or contact_bin_integral_gain and contact_bin_axis_build is None):
        raise ValueError('finite nonnegative gain requires explicit contact-bin build')
    # The estimator cost is validated with the other declared build parameters
    # so an under-sampled A/B cannot be produced with an undeclared field set.
    amcl = validate_amcl_cost(AMCL_DEFAULT if amcl is None else amcl)
    # 8 m LiDAR + 2 m likelihood field + 2 m for vehicle, lane and prior
    # envelopes. No map inside that support is removed or resampled.
    if localization_map_margin_m is not None and (
            type(localization_map_margin_m) not in (int, float)
            or not math.isfinite(localization_map_margin_m)
            or localization_map_margin_m < 12.):
        raise ValueError('localization map margin must be finite and at least 12 metres')
    if contact_bin_direction not in ('lateral', 'longitudinal') or (contact_bin_direction != 'lateral' and contact_bin_axis_build is None):
        raise ValueError('explicit contact-bin build required for longitudinal transfer direction')
    if contact_bin_platform_lift_m is None:
        contact_bin_platform_lift_m=.02 if contact_bin_axis_build is not None else 0.
    if (type(contact_bin_platform_lift_m) not in (int,float)
            or contact_bin_platform_lift_m not in (0.,.02)
            or contact_bin_platform_lift_m and contact_bin_axis_build is None):
        raise ValueError('20mm platform lift requires the explicit contact-bin build')
    if station_scan_refinement and (contact_bin_axis_build is None or resolution != .02
                                   or map_rule != 'centre' or localization_map_margin_m is None):
        raise ValueError('station refinement requires contact docks and an exact 2cm centre-rule localization map')
    if type(physics_step_s) not in (int, float) or physics_step_s not in (.002, .001, .0005):
        raise ValueError('physics step must be 0.002, 0.001 or 0.0005 seconds')
    if physics_island_threads and whole_line_axis_build is None:
        raise ValueError('parallel workshop solver requires the active whole-line world')
    # The occupancy rule and resolution decide how far a measured wall face can
    # sit outside the true face, so they are validated with the other declared
    # build parameters: a refused build must not leave a half-written bundle.
    if type(resolution) not in (int, float) or not math.isfinite(resolution) or resolution <= 0:
        raise ValueError('finite positive map resolution required')
    if map_rule not in MAP_RULES:
        raise ValueError('unknown map occupancy rule: '+str(map_rule))
    if type(map_cell_limit) not in (int, float) or not math.isfinite(map_cell_limit) or map_cell_limit <= 0:
        raise ValueError('positive finite map cell limit required')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    fingerprint = build_fingerprint()
    source = ROOT/'simulation/wheelset_workshop/worlds/wheelset_workshop.world'
    scene = json.loads(SCENE.read_text(encoding='utf-8'))
    layout = proposed_network()
    original_clearance = audit(layout, scene)
    added = LANDMARKS if landmarks else []
    added_clearance = landmark_clearance(layout, added)
    if not original_clearance['passed'] or not added_clearance['passed']:
        raise RuntimeError('navigation road static clearance failed')
    process_sources = {}
    if whole_line_axis_build is None:
        tree = ET.parse(source)
    else:
        from .navigation_workshop import build_process_world
        tree, process_sources = build_process_world(output, whole_line_axis_build)
    world = tree.getroot().find('world')
    world.find('physics/max_step_size').text = str(physics_step_s)
    world.find('physics/real_time_update_rate').text = str(1./physics_step_s)
    configure_island_solver(world, physics_island_threads)
    configure_review_camera(world, review_camera)
    contact_profile = None
    if contact_bin_axis_build is not None:
        from .scout_bin_hardware import add_docks, freeze_axis_build
        library, contact_sources = freeze_axis_build(output/'bin_axis', contact_bin_axis_build)
        process_sources.update(contact_sources)
        contact_profile = add_docks(world, layout, direction=contact_bin_direction,
                                    platform_lift_m=contact_bin_platform_lift_m)
        contact_profile['axis_library'] = str(library)
        contact_profile['pusher_integral_gain_n_per_m_s'] = contact_bin_integral_gain
        # One declared terminal take-over profile for the whole scene. The
        # owner-side regulator closes exactly the tolerance the station goal
        # checker was given, and its capture window only spans the straight
        # approach into the handoff pose, so the take-over cannot be used to
        # cut a corner somewhere a corridor is being shared.
        contact_profile['terminal_docking'] = dict(
            capture_radius_m=.35,
            position_tolerance_m=(.005 if station_scan_refinement else
                                  contact_profile['dock_position_tolerance_m'] * .6),
            yaw_tolerance_rad=contact_profile['dock_yaw_tolerance_rad'] * .8)
        if station_scan_refinement:
            contact_profile['terminal_docking']['steering_mode']='polar'
            if contact_bin_direction=='longitudinal':
                contact_profile['terminal_docking'].update(steering_mode='staged',
                    approach_stations=[dict(target_xy=s['target_xy'],yaw_rad=s['yaw_rad'],
                        approach_side=1 if operation=='LOAD' else -1)
                        for task in contact_profile['tasks'].values()
                        for operation,s in task['stations'].items()])
        (output/'contact_bins.json').write_text(json.dumps(contact_profile, indent=2)+'\n')
    for item in added:
        box_model(world, item)
    world_path = output/'navigation.world'
    save_xml(tree.getroot(), world_path)
    laser_slab=tuple(z+contact_bin_platform_lift_m for z in SLICE)
    polygons, sources, skipped = collision_polygons(world_path, ROOT/'assets/wheelset_workshop/models',
                                                   slab=laser_slab)
    bounds = [0., 0., *scene['bounds_m']]
    width, height, pixels = rasterize([p['polygon'] for p in polygons], bounds, resolution,
                                      rule=map_rule, max_cells=map_cell_limit)
    (output/'map.pgm').write_bytes(f'P5\n{width} {height}\n255\n'.encode()+pixels)
    (output/'map.yaml').write_text(yaml.safe_dump(dict(image='map.pgm', mode='trinary',
        resolution=resolution, origin=[0., 0., 0.], negate=0, occupied_thresh=.65, free_thresh=.196)), encoding='utf-8')
    localization_map = None
    if localization_map_margin_m is not None:
        crop = crop_localization_map(width, height, pixels, resolution,
                                     layout['nodes'], localization_map_margin_m)
        (output/'localization_map.pgm').write_bytes(
            f"P5\n{crop['width']} {crop['height']}\n255\n".encode()+crop['pixels'])
        (output/'localization_map.yaml').write_text(yaml.safe_dump(dict(
            image='localization_map.pgm', mode='trinary', resolution=resolution,
            origin=crop['origin'], negate=0, occupied_thresh=.65, free_thresh=.196)), encoding='utf-8')
        localization_map = {key: value for key, value in crop.items() if key != 'pixels'}
        localization_map.update(source_image='map.pgm',
            source_image_sha256=hashlib.sha256((output/'map.pgm').read_bytes()).hexdigest(),
            image='localization_map.pgm', yaml='localization_map.yaml',
            scope='Exact cell subset covering every declared route and its 12m-or-greater support; '
                  'full global map, graph and physical world are retained.')
    if station_scan_refinement:
        from dataclasses import asdict
        from tools.live_bridge.station_refinement import FitSpec
        contact_profile['terminal_docking']['measurement'] = dict(
            version='station_scan_v1', map_image='localization_map.pgm',
            map_sha256=hashlib.sha256((output/'localization_map.pgm').read_bytes()).hexdigest(),
            resolution_m=resolution, origin=localization_map['origin'],
            stations={station['node_id']: station['target_xy']
                      for task in contact_profile['tasks'].values()
                      for station in task['stations'].values()},
            fit=asdict(FitSpec()),
            laser=dict(frame_id='laser_link', beams=1080, angle_min=-3.14159,
                       angle_max=3.14159, angle_increment=6.28318/1079,
                       range_min=.05, range_max=8., xyzyaw=[.15, 0., .4+contact_bin_platform_lift_m, 0.]))
        (output/'contact_bins.json').write_text(json.dumps(contact_profile, indent=2)+'\n')
    localization_source = ROOT/'tools/live_bridge/sensor_localization.yaml'
    controller_source = ROOT/'tools/live_bridge/nav2_metric.yaml'
    localization = yaml.safe_load(localization_source.read_text())
    controller = yaml.safe_load(controller_source.read_text())
    configs = {}
    for robot in layout['robots']:
        namespace = robot['id']
        dock_xy = dock_yaw = None
        if contact_profile is not None:
            # The dock capture gate reads the true carrier pose while the goal
            # checker reads the localized pose, so the checker has to converge
            # strictly inside the gate instead of exactly onto it.
            dock_xy = contact_profile['terminal_docking']['position_tolerance_m']
            dock_yaw = contact_profile['terminal_docking']['yaw_tolerance_rad']
        loc, nav = robot_configs(namespace, localization, controller,
                                 dock_tolerance_m=dock_xy, dock_yaw_tolerance_rad=dock_yaw)
        if position_encoder:
            loc[namespace]['ekf_filter_node']['ros__parameters']['odom0'] = '/'+namespace+'/wheel/position_odometry'
        loc[namespace]['amcl']['ros__parameters']['max_beams'] = amcl['max_beams']
        # Four quantities move together: a new run with only some of them
        # applied would be a different estimator than the one measured.
        loc[namespace]['amcl']['ros__parameters']['min_particles'] = amcl['min_particles']
        loc[namespace]['amcl']['ros__parameters']['max_particles'] = amcl['max_particles']
        loc[namespace]['amcl']['ros__parameters']['update_min_d'] = amcl['update_min_d']
        loc[namespace]['amcl']['ros__parameters']['update_min_a'] = amcl['update_min_a']
        # The additive likelihood-field score is dominated by the long wall;
        # use joint beam likelihoods so the few distinctive landmarks can
        # constrain encoder drift during skid turns.  Keep every beam and all
        # error/freshness gates; this is specific to the workshop parameters.
        loc[namespace]['amcl']['ros__parameters']['laser_model_type'] = 'likelihood_field_prob'
        loc[namespace]['amcl']['ros__parameters']['do_beamskip'] = False
        if (localization_map is not None and
                loc[namespace]['amcl']['ros__parameters']['laser_likelihood_max_dist'] > 2.):
            raise ValueError('localization subset requires the declared <=2m likelihood-field support')
        # The four-wheel skid surrogate can translate sideways during a turn;
        # AMCL's differential proposal constrains the lateral displacement to
        # zero and can lose the true pose even when LiDAR sees the landmarks.
        loc[namespace]['amcl']['ros__parameters']['robot_model_type'] = 'nav2_amcl::OmniMotionModel'
        # On a 1.8m lane with the existing radius/margin, translation with a
        # half-radian heading error cuts outside the reserved centre corridor.
        # Complete the stationary turn more closely before forward tracking.
        # A contact bundle instead takes its band from the docking chain, so
        # the band stays inside the tolerance the station hop must satisfy.
        if contact_profile is None:
            nav[namespace]['controller_server']['ros__parameters']['FollowPath']['rotate_to_heading_min_angle'] = .10
        if contact_profile is not None:
            nav[namespace]['local_costmap']['local_costmap']['ros__parameters']['footprint'] = json.dumps(
                contact_profile['geometry']['footprint_m'], separators=(',', ':'))
        for suffix, config in [('localization', loc), ('controller', nav)]:
            name = namespace+'_'+suffix+'.yaml'
            (output/name).write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        configs[namespace] = dict(localization=namespace+'_localization.yaml',
            controller=namespace+'_controller.yaml', start_node=robot['start'])
        if localization_map is not None:
            configs[namespace]['map'] = 'localization_map.yaml'
        if contact_profile is not None:
            configs[namespace]['contact_bin_footprint'] = True
    (output/'collision_polygons.json').write_text(json.dumps(polygons, indent=2)+'\n')
    (output/'road_clearance.json').write_text(json.dumps(dict(original=original_clearance, added=added_clearance), indent=2)+'\n')
    (output/'network.json').write_text(json.dumps(layout, indent=2)+'\n')
    source_files = [Path(__file__), source, SCENE, localization_source, controller_source,
        ROOT/'tools/live_bridge/fleet_config.py', ROOT/'tools/ros2/amr_gazebo_guard/launch/metric_model.py']
    if position_encoder:
        source_files.append(ROOT/'tools/live_bridge/position_encoder.py')
    if station_scan_refinement:
        source_files.append(ROOT/'tools/live_bridge/station_refinement.py')
    sources.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    sources.update(process_sources)
    if contact_profile is not None:
        for name in ('scout_bin_hardware.py', 'navigation_model.py', 'navigation_handling.py', 'navigation_bin_runtime.py',
                     'navigation_samples.py', 'navigation_records.py', 'navigation_equipment_ingress.py'):
            path = ROOT/'tools/wheelset_scene'/name
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if whole_line_axis_build is not None:
        sources.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((ROOT/'tools/wheelset_scene').glob('*.py'))})
    manifest = dict(build_fingerprint=fingerprint, resolution_m=resolution, bounds_m=bounds,
        map_rule=map_rule, map_cell_limit=map_cell_limit,
        encoder_source='joint_positions' if position_encoder else 'stock_velocity',
        width=width, height=height, occupied_cells=pixels.count(0), laser_slab_z_m=laser_slab,
        polygons=len(polygons), skipped_dynamic_models=skipped, added_landmarks=added,
        robot_configs=configs, laser_range_max_m=8., laser_fov_radians=2*math.pi,
        laser_update_rate_hz=30. if whole_line_axis_build is not None else None,
        amcl_max_beams=amcl['max_beams'],
        amcl_min_particles=amcl['min_particles'],
        amcl_max_particles=amcl['max_particles'],
        amcl_update_min_distance_m=amcl['update_min_d'],
        amcl_update_min_angle_rad=amcl['update_min_a'],
        amcl_cost=amcl,
        review_camera_enabled=bool(review_camera),
        physics_island_threads=physics_island_threads,
        physics_step_s=physics_step_s,
        model_parameters=dict(lateral_slip_m_per_ns=LATERAL_SLIP_M_PER_NS,
            guard_command_relay=bool(fault_relay),
            provenance='synthetic compliant lateral tyre contact, not hardware calibrated'),
        whole_line=whole_line_axis_build is not None,
        contact_bins=contact_profile,
        source_sha256=sources, files_sha256={p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.rglob('*') if p.is_file()},
        status='BUILT_NOT_SENSOR_RUNTIME_VERIFIED',
        initial_pose='Known simulator spawn prior, not a measured localization result',
        claim_boundary='Known synthetic map from actual derived-world static collision boxes, including local rotations. '
            'The occupancy grid was rasterised with the declared map_rule at the declared resolution, so every wall face '
            'carries at most one cell (area) or half a cell (centre) of outward error against the same coordinates. '
            'Landmarks are an explicit navigation design addition outside the reserved roads. '
            'No SLAM, sensor/Nav2 execution, localization accuracy or industrial acceptance established by building this bundle.')
    if localization_map is not None:
        manifest['localization_map'] = localization_map
    if fingerprint != build_fingerprint():
        raise RuntimeError('frozen scene changed while building navigation bundle')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def build_argument_parser():
    """The declared build surface, kept separate so it can be checked alone."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--without-landmarks', action='store_true')
    p.add_argument('--fault-relay', action='store_true')
    p.add_argument('--resolution', type=float, default=.05,
                   help='Static map cell size in metres; the estimator inherits up to one cell of face error')
    p.add_argument('--map-rule', choices=MAP_RULES, default='area',
                   help='area keeps the historical byte-identical grid; centre bounds the face error by half a cell')
    p.add_argument('--max-cells', type=int, default=60_000_000,
                   help='Refuse a map larger than this many cells instead of exhausting memory')
    p.add_argument('--without-review-camera', action='store_true',
                   help='Omit the unused render sensor in a runtime-only derived world')
    p.add_argument('--whole-line-axis-build', type=Path,
                   help='Build the active two-workpiece world using this audited native axis build')
    p.add_argument('--contact-bin-axis-build', type=Path,
                   help='Add assumed SCOUT light-bin decks/docks using this audited axis build')
    p.add_argument('--contact-bin-integral-gain', type=float, default=0.,
                   help='Explicit probe integral force gain in N/(m*s); default preserves historical PD control')
    p.add_argument('--contact-bin-direction', choices=('lateral', 'longitudinal'), default='lateral')
    p.add_argument('--contact-bin-platform-lift-m',type=float,choices=(0.,.02),
                   help='Contact mode defaults to 20mm chassis clearance correction; zero is legacy diagnosis only')
    p.add_argument('--physics-step-s', type=float, choices=(.002, .001, .0005), default=.002)
    p.add_argument('--localization-map-margin-m', type=float,
                   help='Use exact full-map cells around every route plus >=12m support; retain the global map')
    p.add_argument('--position-encoder', action='store_true',
                   help='Fuse signed wheel-position increments; retain stock encoder stream for diagnostics')
    p.add_argument('--station-scan-refinement', action='store_true',
                   help='Use measured full scans near declared contact stations; preserve physical capture gates')
    p.add_argument('--physics-island-threads', type=int, default=0,
                   help='Parallel ODE islands in the active workshop; preserve 400 solver iterations')
    p.add_argument('--amcl-max-beams', type=int, default=AMCL_DEFAULT['max_beams'],
                   help='LiDAR beams per AMCL update; 1080 is the residual-preserving default, not the Nav2 default')
    p.add_argument('--amcl-min-particles', type=int, default=AMCL_DEFAULT['min_particles'],
                   help='Lower edge of the AMCL particle band; 5000 is the measured residual-preserving default')
    p.add_argument('--amcl-max-particles', type=int, default=AMCL_DEFAULT['max_particles'],
                   help='Upper edge of the AMCL particle band; 20000 is the measured residual-preserving default')
    p.add_argument('--amcl-update-min-d', type=float, default=AMCL_DEFAULT['update_min_d'],
                   help='Translation that forces a new AMCL update in metres; .005 is the measured default')
    p.add_argument('--amcl-update-min-a', type=float, default=AMCL_DEFAULT['update_min_a'],
                   help='Rotation that forces a new AMCL update in radians; .005 is the measured default')
    return p


def main():
    args = build_argument_parser().parse_args()
    result = build_bundle(args.output, landmarks=not args.without_landmarks, fault_relay=args.fault_relay,
                          resolution=args.resolution, map_rule=args.map_rule, map_cell_limit=args.max_cells,
                          whole_line_axis_build=args.whole_line_axis_build,
                          review_camera=not args.without_review_camera,
                          physics_island_threads=args.physics_island_threads,
                          contact_bin_axis_build=args.contact_bin_axis_build,
                          contact_bin_integral_gain=args.contact_bin_integral_gain,
                          contact_bin_direction=args.contact_bin_direction,
                          physics_step_s=args.physics_step_s,
                          localization_map_margin_m=args.localization_map_margin_m,
                          position_encoder=args.position_encoder,
                          station_scan_refinement=args.station_scan_refinement,
                          contact_bin_platform_lift_m=args.contact_bin_platform_lift_m,
                          amcl=dict(max_beams=args.amcl_max_beams,
                                    min_particles=args.amcl_min_particles,
                                    max_particles=args.amcl_max_particles,
                                    update_min_d=args.amcl_update_min_d,
                                    update_min_a=args.amcl_update_min_a))
    print(json.dumps({k: result[k] for k in ['status', 'polygons', 'width', 'height', 'occupied_cells',
                                             'resolution_m', 'map_rule']}))


if __name__ == '__main__':
    main()
