"""Gravity-supported wheelset on two rows of physically driven conveyor rollers.

The paired conveyor layout follows visible workshop transport; roller drive,
dimensions, mass and contact parameters are explicit prototype assumptions.
"""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .actuated_world import inertia, joint, box_visual
from .assets import wheelset
from .build import e, numbers, camera_pose, save_xml


MASS = 1000.
RADIUS = .49
ROLLER_RADIUS = .12
AXLE_Z = .8
DISTANCE = 2.
ROLLER_X = [index*.24 for index in range(-4,14)]


def surface(collision, friction=.8):
    value = e(collision, 'surface')
    ode = e(e(value, 'friction'), 'ode'); e(ode, 'mu', friction); e(ode, 'mu2', friction)
    ode = e(e(value, 'contact'), 'ode')
    for key, number in [('kp', 1e7), ('kd', 2e4), ('max_vel', .03), ('min_depth', .0002)]:
        e(ode, key, number)


def box(link, name, size, center, color, *, collision=True):
    pose = numbers([*center, 0, 0, 0])
    box_visual(link, name+'_visual', numbers(size), color, pose)
    if collision:
        value = e(link, 'collision', name=name); e(value, 'pose', pose)
        e(e(e(value, 'geometry'), 'box'), 'size', numbers(size)); surface(value)


def contacts(link, collision, topic):
    sensor = e(link, 'sensor', name=topic+'_sensor', type='contact')
    e(sensor, 'always_on', 'true'); e(sensor, 'update_rate', 100)
    e(e(sensor, 'contact'), 'collision', collision)
    plugin = e(sensor, 'plugin', name=topic+'_ros', filename='libgazebo_ros_bumper.so')
    ros = e(plugin, 'ros'); e(ros, 'namespace', '/contact_conveyor')
    e(ros, 'remapping', 'bumper_states:='+topic); e(plugin, 'frame_name', 'world')


def cylinder(link, name, radius, length, *, collision=True, color='.55 .61 .66 1'):
    pose = numbers([0,0,0,math.pi/2,0,0])
    visual = e(link, 'visual', name=name+'_visual'); e(visual, 'pose', pose)
    shape = e(e(visual, 'geometry'), 'cylinder'); e(shape, 'radius', radius); e(shape, 'length', length)
    material = e(visual, 'material'); e(material, 'ambient', color); e(material, 'diffuse', color)
    if collision:
        value = e(link, 'collision', name=name); e(value, 'pose', pose)
        shape = e(e(value, 'geometry'), 'cylinder'); e(shape, 'radius', radius); e(shape, 'length', length)
        surface(value)


def build(output, library):
    output, library = Path(output).resolve(), Path(library).resolve()
    output.mkdir(parents=True, exist_ok=False); assert library.is_file()
    root = ET.Element('sdf', version='1.7'); world = e(root, 'world', name='wheelset_contact_conveyor')
    e(world, 'gravity', '0 0 -9.81')
    physics = e(world, 'physics', name='contact_validation', type='ode')
    e(physics, 'max_step_size', .001); e(physics, 'real_time_update_rate', 1000)
    ode = e(physics, 'ode'); solver = e(ode, 'solver')
    e(solver, 'type', 'quick'); e(solver, 'iters', 400); e(solver, 'sor', 1)
    constraints = e(ode, 'constraints'); e(constraints, 'cfm', 1e-9); e(constraints, 'erp', .8)
    settings = e(world, 'scene'); e(settings, 'ambient', '.65 .65 .65 1'); e(settings, 'background', '.17 .20 .24 1')
    light = e(world, 'light', name='sun', type='directional'); e(light, 'pose', '0 0 8 0 0 0')
    e(light, 'diffuse', '.8 .8 .8 1'); e(light, 'direction', '-.4 .3 -1')
    camera = dict(position=[2.8,-5.4,3.2], target=[1.,0,.55])
    gui = e(world, 'gui'); view = e(gui, 'camera', name='user_camera')
    e(view, 'pose', numbers(camera_pose(camera))); e(view, 'view_controller', 'orbit')
    fixed = e(world, 'model', name='floor'); e(fixed, 'static', 'true')
    box(e(fixed, 'link', name='body'), 'floor', [9,6,.1], [1,0,-.05], '.23 .26 .29 1')

    conveyor = e(world, 'model', name='powered_conveyor')
    base = e(conveyor, 'link', name='base'); inertia(base, 100, [10,10,10])
    joint(conveyor, 'conveyor_anchor', 'fixed', 'world', 'base', None)
    for side in (-1,1):
        for offset in (-.18,.18):
            box(base, 'beam_'+str(side)+'_'+str(offset), [4.6,.08,.10], [1.,side*.77+offset,.05], '.08 .35 .52 1')
        box(base, 'lateral_guide_'+str(side), [4.6,.04,.20], [1.,side*.86,.55], '.08 .35 .52 1')
    axes = []
    for index,x in enumerate(ROLLER_X):
        for side in (-1,1):
            name = 'roller_'+str(index)+('_left' if side < 0 else '_right')
            link = e(conveyor, 'link', name=name)
            e(link, 'pose', numbers([x,side*.77,AXLE_Z-RADIUS-ROLLER_RADIUS,0,0,0]))
            inertia(link, 5, [.05,.036,.05]); link.find('gravity').text = 'true'
            cylinder(link, name+'_contact', ROLLER_RADIUS, .24)
            box_visual(link, 'rotation_mark', '.025 .26 .025', '1 .4 .03 1', '0 0 .12 0 0 0')
            axis = name+'_joint'; joint(conveyor, axis, 'revolute', 'base', name, '0 1 0', [-1e16,1e16,200,2.5], .05)
            axes.append(axis)
            for offset in (-.18,.18):
                box(base, name+'_bearing_'+str(offset), [.14,.08,.14], [x,side*.77+offset,.15], '.08 .35 .52 1', collision=False)
    # A visible safety stop limits overrun; nominal position is measured before it.
    box(base, 'safety_stop', [.10,.60,.24], [DISTANCE+.22,0,AXLE_Z], '.85 .18 .12 1')
    box(base, 'safety_stop_support', [.18,.25,.65], [DISTANCE+.22,0,.325], '.25 .30 .35 1')
    contacts(base, 'safety_stop', 'stop_contacts')
    plugin = e(conveyor, 'plugin', name='powered_roller_servo', filename=str(library))
    e(e(plugin, 'ros'), 'namespace', '/contact_conveyor/drive'); e(plugin, 'lease_s', 1.)
    for name in axes:
        axis = e(plugin, 'axis'); e(axis, 'joint', name); e(axis, 'mode', 'velocity')
        e(axis, 'mass', 20); e(axis, 'limit', 200); e(axis, 'damping', .05)

    model = e(world, 'model', name='contact_wheelset'); e(model, 'pose', f'0 0 {AXLE_Z+.005} 0 0 0')
    payload = e(model, 'link', name='body')
    inertia(payload, MASS, [633.4,106.4,633.4]); payload.find('gravity').text = 'true'
    mesh = wheelset(); keep = [i for i,p in enumerate(mesh.parts) if not p.startswith(('bearing_','endcap_'))]
    mesh.faces = [mesh.faces[i] for i in keep]; mesh.materials = [mesh.materials[i] for i in keep]
    mesh.parts = [mesh.parts[i] for i in keep]; mesh.write_dae(output/'wheelset.dae')
    visual = e(payload, 'visual', name='video_wheelset'); e(visual, 'pose', '0 0 -.56 0 0 0')
    e(e(e(visual, 'geometry'), 'mesh'), 'uri', (output/'wheelset.dae').as_uri())
    box_visual(payload, 'rotation_reference', '.055 .02 .20', '1 .38 .04 1', '0 -.86 .30 0 0 0')
    for side in (-1,1):
        name = 'tread_left' if side < 0 else 'tread_right'
        collision = e(payload, 'collision', name=name)
        e(collision, 'pose', numbers([0,side*.77,0,math.pi/2,0,0]))
        shape = e(e(collision, 'geometry'), 'cylinder'); e(shape, 'radius', RADIUS); e(shape, 'length', .10)
        surface(collision); contacts(payload, name, name+'_contacts')
    collision = e(payload, 'collision', name='axle')
    e(collision, 'pose', numbers([0,0,0,math.pi/2,0,0]))
    shape = e(e(collision, 'geometry'), 'cylinder'); e(shape, 'radius', .11); e(shape, 'length', 2.28)
    surface(collision, .15)

    state = e(world, 'plugin', name='independent_state', filename='libgazebo_ros_state.so')
    e(e(state, 'ros'), 'namespace', '/contact_conveyor'); e(state, 'update_rate', 50)
    camera_model = e(world, 'model', name='recording_camera'); e(camera_model, 'static', 'true')
    e(camera_model, 'pose', numbers(camera_pose(camera)))
    sensor = e(e(camera_model, 'link', name='optics'), 'sensor', name='recording', type='camera')
    e(sensor, 'always_on', 'true'); e(sensor, 'update_rate', 10)
    optics = e(sensor, 'camera'); e(optics, 'horizontal_fov', 1.2)
    image = e(optics, 'image'); e(image, 'width', 960); e(image, 'height', 540); e(image, 'format', 'R8G8B8')
    clip = e(optics, 'clip'); e(clip, 'near', .05); e(clip, 'far', 30)
    plugin = e(sensor, 'plugin', name='recording_ros', filename='libgazebo_ros_camera.so')
    e(e(plugin, 'ros'), 'namespace', '/contact_conveyor'); e(plugin, 'camera_name', 'camera'); e(plugin, 'frame_name', 'world')
    save_xml(root, output/'conveyor.world')
    manifest = dict(scope='gravity_contact_conveyor_prototype', mass_kg=MASS, wheel_radius_m=RADIUS,
        roller_radius_m=ROLLER_RADIUS, axle_height_m=AXLE_Z, travel_m=DISTANCE, maximum_roller_torque_nm=200,
        roller_joints=axes, library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},
        assumptions=['Paired conveyor follows visible transport; 36 powered rollers and their internal drive are prototype assumptions.',
            'Primitive collisions approximate the video visual profile; assumed mass, inertia, dimensions and contact coefficients.',
            'Wheelset has gravity, collisions, no joints and no motion plugin; only conveyor roller joints receive torque.',
            'Nominal arrival uses measured pose and speed; physical safety buffer limits overrun.',
            'This isolated two-metre prototype does not establish full workshop conveyor connections or acceptance.'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
