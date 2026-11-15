"""Explicit light-bin docks and a contact deck for the SCOUT research model.

All dimensions describe an assumed accessory, not an AgileX product. The free
bin has gravity and contacts; only the chassis-mounted transfer finger is
actuated. This module supplies geometry, never a loading acknowledgement.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .actuated_world import inertia, joint
from .build import e, numbers
from .contact_conveyor_world import box, contacts


SOURCE = 'gazebo_scout_bin_contact'
GEOMETRY = dict(bin_size_m=[.16, .16, .08], bin_mass_kg=2.,
    carrier_frame='base_footprint', axle_world_z_m=.08, deck_top_local_m=.30, deck_size_m=[.50, .40, .04],
    pusher_origin_local_m=[0., -.49, .34], pusher_size_m=[.20, .04, .06],
    load_stroke_m=.39, unload_stroke_m=.79, retract_stroke_m=0.,
    maximum_stroke_m=.82, pusher_force_n=50.,
    loaded_bin_local_m=[0., 0., .34],
    footprint_m=[[-.315, -.252], [-.10, -.51], [.10, -.51],
                 [.315, -.252], [.315, .252], [-.315, .252]])


def freeze_axis_build(output, build):
    """Copy the verified build and matching source; never use an unchecked .so."""
    from .build import ROOT
    output, build = Path(output), Path(build)
    manifest = json.loads((build/'build_manifest.json').read_text())
    if manifest.get('passed') is not True:
        raise ValueError('validated axis build required')
    sources = {}
    for relative, digest in manifest['source_sha256'].items():
        source = ROOT/'tools/ros2/wheelset_axis_servo'/relative
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('axis source changed: '+relative)
        target = output/'sources'/relative; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data); sources[str(source)] = digest
    library = build/'libwheelset_axis_servo.so'; data = library.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest['library_sha256']:
        raise ValueError('axis library changed')
    output.mkdir(parents=True, exist_ok=True)
    target = output/library.name; target.write_bytes(data)
    (output/'build_manifest.json').write_bytes((build/'build_manifest.json').read_bytes())
    sources[str(library)] = manifest['library_sha256']
    return target, sources


def carrier_geometry(direction, platform_lift_m=0.):
    if direction not in ('lateral', 'longitudinal'):
        raise ValueError('explicit lateral or longitudinal transfer direction required')
    if type(platform_lift_m) not in (int,float) or platform_lift_m not in (0.,.02):
        raise ValueError('declared historical zero or 20mm raised carrier required')
    geometry = deepcopy(GEOMETRY)
    if direction == 'longitudinal':
        geometry['pusher_origin_local_m'] = [-.49, 0., .34]
        geometry['pusher_size_m'] = [.04, .20, .06]
        geometry['deck_size_m'] = [.40, .50, .04]
        geometry['footprint_m'] = [[-.51, -.10], [-.315, -.252], [.315, -.252],
                                   [.315, .252], [-.315, .252], [-.51, .10]]
    if platform_lift_m:
        geometry['platform_lift_m']=platform_lift_m
        geometry['deck_top_local_m']+=platform_lift_m
        geometry['pusher_origin_local_m'][2]+=platform_lift_m
        geometry['loaded_bin_local_m'][2]+=platform_lift_m
    return geometry


def audit_contact_clearance(sdf, world, profile):
    """Check actual chassis/rack boxes and actual laser height before spawning.

    These four flat-floor target checks do not replace road, swept-path or
    physical validation. The chassis envelope also covers 0.01rad roll/pitch
    and 2mm heave; the uncalibrated prototype must stay inside that envelope.
    """
    body=sdf.find("model/link[@name='base_footprint']")
    collision=next((c for c in body.findall('collision')
                    if c.get('name','').endswith('metric_chassis_collision')),None)
    if collision is None:
        raise ValueError('actual metric chassis collision required')
    chassis_pose=list(map(float,collision.findtext('pose','0 0 0 0 0 0').split()))
    chassis_size=list(map(float,collision.findtext('geometry/box/size').split()))
    if any(abs(v)>1e-7 for v in chassis_pose[3:]):
        raise ValueError('declared unrotated chassis box required')
    inflate_z=.002+.01*(abs(chassis_pose[0])+abs(chassis_pose[1])+
                                    chassis_size[0]/2+chassis_size[1]/2)
    inflate_xy=(profile['dock_position_tolerance_m']+
        profile['dock_yaw_tolerance_rad']*math.hypot(
            abs(chassis_pose[0])+chassis_size[0]/2,abs(chassis_pose[1])+chassis_size[1]/2)+
        .01*(abs(chassis_pose[2])+chassis_size[2]/2))
    def bounds(parent,local,size,inflation=0.):
        x,y,z,roll,pitch,yaw=parent
        if abs(roll)>1e-7 or abs(pitch)>1e-7 or any(abs(v)>1e-7 for v in local[3:5]):
            raise ValueError('explicit flat-floor box poses required')
        angle=yaw+local[5]
        if abs(math.sin(2*angle))>2e-6:
            raise ValueError('right-angle handoff targets required')
        c,s=math.cos(yaw),math.sin(yaw)
        cx,cy,cz=x+c*local[0]-s*local[1],y+s*local[0]+c*local[1],z+local[2]
        dx=(abs(math.cos(angle))*size[0]+abs(math.sin(angle))*size[1])/2
        dy=(abs(math.sin(angle))*size[0]+abs(math.cos(angle))*size[1])/2
        return [cx-dx,cy-dy,cz-size[2]/2-inflation,
                cx+dx,cy+dy,cz+size[2]/2+inflation]
    checks=[]
    for task,spec in profile['tasks'].items():
        for operation,station in spec['stations'].items():
            target=[*station['target_xy'],0.,0.,0.,station['yaw_rad']]
            robot=bounds(target,chassis_pose,chassis_size,inflate_z)
            robot=[robot[0]-inflate_xy,robot[1]-inflate_xy,robot[2],
                   robot[3]+inflate_xy,robot[4]+inflate_xy,robot[5]]
            name=station['support_collision'].split('::')[0]
            rack=world.find("model[@name='"+name+"']")
            placement=list(map(float,rack.findtext('pose').split()))
            if any(link.find('pose') is not None for link in rack.findall('link')):
                raise ValueError('explicit unshifted dock links required')
            for element in rack.findall('link/collision'):
                local=list(map(float,element.findtext('pose','0 0 0 0 0 0').split()))
                size=list(map(float,element.findtext('geometry/box/size').split()))
                obstacle=bounds(placement,local,size)
                penetration=[min(robot[i+3],obstacle[i+3])-max(robot[i],obstacle[i])
                             for i in range(3)]
                checks.append(dict(task=task,operation=operation,collision=element.get('name'),
                    penetration_xyz_m=penetration,passed=not all(v>0 for v in penetration)))
    laser=sdf.find(".//sensor[@type='ray']")
    actual=list(map(float,laser.findtext('pose','0 0 0 0 0 0').split()))
    expected=.4+profile['geometry'].get('platform_lift_m',0.)
    laser_valid=len(actual)==6 and all(abs(a-b)<1e-5 for a,b in zip(actual,[.15,0.,expected,0.,0.,0.]))
    result=dict(passed=all(c['passed'] for c in checks) and laser_valid,checks=checks,
        chassis_envelope_z_inflation_m=inflate_z,laser_pose_local=actual,
        chassis_capture_xy_inflation_m=inflate_xy,
        expected_laser_height_m=expected,laser_valid=laser_valid,
        scope='Actual flat-floor handoff chassis/rack boxes plus 0.01rad roll/pitch and2mm heave; '
              'not complete swept paths, industrial calibration or physical handoff acceptance.')
    if not result['passed']:
        raise ValueError('handoff chassis/rack or laser geometry invalid: '+json.dumps(result))
    return result


def add_carrier(sdf, robot_id, library, *, integral_gain=0., direction='lateral',platform_lift_m=0.):
    """Extend the converted SDF; retain its original wheel/guard/sensor model."""
    if type(integral_gain) not in (int, float) or not math.isfinite(integral_gain) or integral_gain < 0:
        raise ValueError('finite nonnegative probe integral gain required')
    geometry = carrier_geometry(direction,platform_lift_m)
    rotation = -math.pi/2 if direction == 'longitudinal' else 0.
    model = sdf.find('model')
    root = None if model is None else model.find("link[@name='base_footprint']")
    if root is None or not any(c.get('name', '').endswith('metric_chassis_collision') for c in root.findall('collision')):
        raise ValueError('the converted SCOUT base_footprint with audited chassis collision is required')
    if model.find("link[@name='handling_deck']") is not None:
        raise ValueError('contact deck already installed')
    library = Path(library).resolve()
    if not library.is_file():
        raise FileNotFoundError(library)
    deck = e(model, 'link', name='handling_deck')
    e(deck, 'pose', numbers([0, 0, geometry['deck_top_local_m']-.02, 0, 0, rotation]), relative_to='base_footprint')
    inertia(deck, 1., [.02, .025, .035]); deck.find('gravity').text = 'true'
    box(deck, 'support', GEOMETRY['deck_size_m'], [0, 0, 0], '.1 .5 .3 1')
    joint(model, 'handling_deck_mount', 'fixed', 'base_footprint', 'handling_deck', None)
    finger = e(model, 'link', name='bin_transfer_finger')
    e(finger, 'pose', numbers([*geometry['pusher_origin_local_m'], 0, 0, rotation]), relative_to='base_footprint')
    inertia(finger, .5, [.002, .003, .003]); finger.find('gravity').text = 'true'
    box(finger, 'face', GEOMETRY['pusher_size_m'], [0, 0, 0], '.9 .5 .05 1')
    joint(model, 'bin_transfer_stroke', 'prismatic', 'base_footprint', 'bin_transfer_finger',
          '0 1 0', [0, GEOMETRY['maximum_stroke_m'], GEOMETRY['pusher_force_n'], .08], 5)
    axis = model.find("joint[@name='bin_transfer_stroke']/axis/xyz")
    axis.set('expressed_in', 'base_footprint')
    axis.text = '1 0 0' if direction == 'longitudinal' else '0 1 0'
    drive = e(model, 'plugin', name='bin_transfer_drive', filename=str(library))
    e(e(drive, 'ros'), 'namespace', '/scout_bin/'+robot_id); e(drive, 'lease_s', 1.)
    axis = e(drive, 'axis'); e(axis, 'joint', 'bin_transfer_stroke')
    e(axis, 'mode', 'probe'); e(axis, 'mass', 20)
    e(axis, 'limit', GEOMETRY['pusher_force_n']); e(axis, 'damping', 10)
    e(axis, 'integral_gain_n_per_m_s', integral_gain)
    return dict(entity=robot_id+'::base_footprint', support_collision=robot_id+'::handling_deck::support',
                drive_namespace='/scout_bin/'+robot_id)


def add_docks(world, layout, *, direction='lateral',platform_lift_m=0.):
    """Add the two existing delivery tasks' actual input and output supports.

    Pickup yaw aligns the accessory across the bay instead of obstructing its
    exit. Loaded travel includes the retracted finger in the stated footprint.
    Extension is a stationary dock operation, outside that travel footprint.
    """
    geometry = carrier_geometry(direction,platform_lift_m)
    rotation = -math.pi/2 if direction == 'longitudinal' else 0.
    nodes = {node['id']: node for node in layout['nodes']}
    specs = {}
    for task, pickup, dropoff, yaw in [('inspection_consumables', 'bay_west', 'receiving', 0.),
                                     ('small_tools', 'bay_east', 'supply', math.pi)]:
        name = 'bin_'+task
        if world.find("model[@name='"+name+"']") is not None:
            raise ValueError('contact-bin geometry already installed')
        stations = {}
        for operation, node_id, angle, lateral, top in [
                ('LOAD', pickup, -math.pi/2, -.36, .303),
                ('UNLOAD', dropoff, yaw, .40, .297)]:
            top+=platform_lift_m
            node = nodes[node_id]
            offset_x, offset_y = ((lateral, 0.) if direction == 'longitudinal' else (0., lateral))
            x = node['x']+math.cos(angle)*offset_x-math.sin(angle)*offset_y
            y = node['y']+math.sin(angle)*offset_x+math.cos(angle)*offset_y
            rack_name = 'dock_'+task+'_'+operation.lower()
            rack = e(world, 'model', name=rack_name); e(rack, 'static', 'true')
            e(rack, 'pose', numbers([x, y, 0, 0, 0, angle+rotation]))
            body = e(rack, 'link', name='body')
            box(body, 'support', [.5, .28, .04], [0, 0, top-.02], '.1 .4 .6 1')
            for dx in (-.20, .20):
                box(body, 'leg_'+str(dx), [.04, .04, top-.04], [dx, 0, (top-.04)/2], '.2 .3 .4 1')
            stations[operation] = dict(node_id=node_id, target_xy=[node['x'], node['y']],
                yaw_rad=angle, support_collision=rack_name+'::body::support',
                bin_world_xyz=[x, y, top+GEOMETRY['bin_size_m'][2]/2])
        source = stations['LOAD']['bin_world_xyz']
        cargo = e(world, 'model', name=name)
        e(cargo, 'pose', numbers([*source[:2], source[2]+.002, 0, 0, -math.pi/2+rotation]))
        body = e(cargo, 'link', name='body')
        inertia(body, GEOMETRY['bin_mass_kg'], [.005, .005, .008])
        body.find('gravity').text = 'true'
        box(body, 'bin', GEOMETRY['bin_size_m'], [0, 0, 0], '.9 .65 .08 1')
        contacts(body, 'bin', 'bin_contacts')
        body.find('sensor/plugin/ros/namespace').text = '/scout_bin/'+task
        specs[task] = dict(cargo_entity=name, cargo_collision=name+'::body::bin', stations=stations)
    for robot in layout['robots']:
        robot['start_yaw_rad'] = -math.pi/2
    return dict(version='scout_contact_bin_v1', scene_id=layout['id'], source=SOURCE,
        robots=[robot['id'] for robot in layout['robots']], geometry=geometry, tasks=specs,
        transfer_direction=direction,
        world_to_map=dict(translation_m=[0., 0., 0.], yaw_rad=0.),
        dock_position_tolerance_m=.025, dock_yaw_tolerance_rad=.05,
        cargo_position_tolerance_m=.025, stable_support_s=.3, freshness_s=.15,
        claim_boundary='Assumed 2kg light-bin accessory and contact docks, not hardware calibration. '
            'Geometry construction does not prove docking, navigation, loading, inventory or process execution. '
            'Finger extension requires a retained stationary dock; travel requires full retraction.')
