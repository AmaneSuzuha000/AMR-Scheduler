"""Assumed chain dogs push a free wheelset along two fixed conveyor rails.

Only the upper straight chain stroke is represented. Hidden chain drive and
return, dimensions and force limits are prototype assumptions, not video facts.
"""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .actuated_world import inertia, joint, box_visual
from .assets import wheelset
from .build import e, numbers, camera_pose, save_xml
from .contact_conveyor_world import box, surface, contacts, MASS, RADIUS, AXLE_Z, DISTANCE


def build(output, library):
    output, library = Path(output).resolve(), Path(library).resolve()
    output.mkdir(parents=True, exist_ok=False)
    assert library.is_file()
    root = ET.Element('sdf', version='1.7')
    world = e(root, 'world', name='wheelset_contact_chain_conveyor')
    e(world, 'gravity', '0 0 -9.81')
    physics = e(world, 'physics', name='contact_validation', type='ode')
    e(physics, 'max_step_size', .001); e(physics, 'real_time_update_rate', 1000)
    ode = e(physics, 'ode'); solver = e(ode, 'solver')
    e(solver, 'type', 'quick'); e(solver, 'iters', 400); e(solver, 'sor', 1)
    constraints = e(ode, 'constraints'); e(constraints, 'cfm', 1e-9); e(constraints, 'erp', .8)
    scene = e(world, 'scene'); e(scene, 'ambient', '.65 .65 .65 1'); e(scene, 'background', '.17 .20 .24 1')
    light = e(world, 'light', name='sun', type='directional'); e(light, 'pose', '0 0 8 0 0 0')
    e(light, 'diffuse', '.8 .8 .8 1'); e(light, 'direction', '-.4 .3 -1')
    camera = dict(position=[2.8,-5.4,3.2], target=[1.,0,.55])
    gui = e(world, 'gui'); view = e(gui, 'camera', name='user_camera')
    e(view, 'pose', numbers(camera_pose(camera))); e(view, 'view_controller', 'orbit')
    floor = e(world, 'model', name='floor'); e(floor, 'static', 'true')
    box(e(floor, 'link', name='body'), 'floor', [9,6,.1], [1,0,-.05], '.23 .26 .29 1')

    conveyor = e(world, 'model', name='powered_conveyor')
    base = e(conveyor, 'link', name='base'); inertia(base, 100, [10,10,10])
    joint(conveyor, 'conveyor_anchor', 'fixed', 'world', 'base', None)
    for side in (-1,1):
        box(base, 'support_rail_'+str(side), [4.6,.12,.10], [1,side*.77,AXLE_Z-RADIUS-.05], '.42 .48 .53 1')
        box(base, 'chain_housing_'+str(side), [4.6,.12,.17], [1,side*.96,.17], '.08 .35 .52 1')
        box(base, 'lateral_guide_'+str(side), [4.6,.04,.20], [1,side*.86,.55], '.08 .35 .52 1')
        for x in (-.7,.5,1.7,2.9):
            box(base, 'foot_'+str(side)+'_'+str(x), [.16,.35,.20], [x,side*.86,.10], '.08 .35 .52 1')
        # Gate is fixed for this isolated receiving station, not an invented
        # whole-line routing mechanism. Rail contact carries the vertical load.
        box(base, 'station_stop_'+str(side), [.10,.18,.26], [DISTANCE+RADIUS+.05,side*.77,AXLE_Z], '.85 .18 .12 1')
        box(base, 'station_stop_post_'+str(side), [.12,.12,.65], [DISTANCE+RADIUS+.05,side*.96,.325], '.25 .30 .35 1')

    drive = e(conveyor, 'link', name='chain_dogs'); inertia(drive, 50, [10,5,10])
    drive.find('gravity').text = 'true'
    for side in (-1,1):
        box(drive, 'chain_dog_'+str(side), [.10,.18,.26], [-RADIUS-.055,side*.77,AXLE_Z], '1 .55 .04 1')
        box(drive, 'dog_support_'+str(side), [.10,.12,.60], [-RADIUS-.055,side*.96,.50], '.85 .49 .04 1')
        box(drive, 'dog_bridge_'+str(side), [.10,.35,.08], [-RADIUS-.055,side*.87,.77], '.85 .49 .04 1', collision=False)
    axis = 'chain_straight_stroke'
    joint(conveyor, axis, 'prismatic', 'base', 'chain_dogs', '1 0 0', [-.02,DISTANCE+.06,5000,.20], 5.)
    plugin = e(conveyor, 'plugin', name='chain_drive_servo', filename=str(library))
    e(e(plugin, 'ros'), 'namespace', '/contact_conveyor/drive'); e(plugin, 'lease_s', 1.)
    setting = e(plugin, 'axis'); e(setting, 'joint', axis); e(setting, 'mode', 'position')
    e(setting, 'mass', 1000); e(setting, 'limit', 5000); e(setting, 'damping', 5.)

    model = e(world, 'model', name='contact_wheelset'); e(model, 'pose', f'0 0 {AXLE_Z+.005} 0 0 0')
    payload = e(model, 'link', name='body'); inertia(payload, MASS, [633.4,106.4,633.4])
    payload.find('gravity').text = 'true'
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
    collision = e(payload, 'collision', name='axle'); e(collision, 'pose', numbers([0,0,0,math.pi/2,0,0]))
    shape = e(e(collision, 'geometry'), 'cylinder'); e(shape, 'radius', .11); e(shape, 'length', 2.28)
    surface(collision, .15)
    state = e(world, 'plugin', name='independent_state', filename='libgazebo_ros_state.so')
    e(e(state, 'ros'), 'namespace', '/contact_conveyor'); e(state, 'update_rate', 50)
    cam = e(world, 'model', name='recording_camera'); e(cam, 'static', 'true')
    e(cam, 'pose', numbers(camera_pose(camera)))
    sensor = e(e(cam, 'link', name='optics'), 'sensor', name='recording', type='camera')
    e(sensor, 'always_on', 'true'); e(sensor, 'update_rate', 10)
    optics = e(sensor, 'camera'); e(optics, 'horizontal_fov', 1.2)
    image = e(optics, 'image'); e(image, 'width', 960); e(image, 'height', 540); e(image, 'format', 'R8G8B8')
    clip = e(optics, 'clip'); e(clip, 'near', .05); e(clip, 'far', 30)
    plugin = e(sensor, 'plugin', name='recording_ros', filename='libgazebo_ros_camera.so')
    e(e(plugin, 'ros'), 'namespace', '/contact_conveyor'); e(plugin, 'camera_name', 'camera'); e(plugin, 'frame_name', 'world')
    # Near-critical damping for the assumed 1000 kg load. Keep the independent
    # stop-speed criteria; reduce numerical contact chatter in the model.
    for contact in world.findall('.//collision/surface/contact/ode'):
        contact.find('kd').text = '200000'
        contact.find('max_vel').text = '0.003'
        contact.find('min_depth').text = '0.00002'
    save_xml(root, output/'conveyor.world')
    manifest = dict(scope='gravity_contact_chain_conveyor_prototype', mechanism='chain_dogs', mass_kg=MASS,
        wheel_radius_m=RADIUS, axle_height_m=AXLE_Z, travel_m=DISTANCE, maximum_chain_force_n=5000,
        actuator_joints=[axis], actuator_mode='position', maximum_target=DISTANCE+.015,
        contact_damping_ns_m=200000, contact_correction_limit_m_s=.003, contact_depth_m=.00002,
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},
        assumptions=['Paired fixed conveyor rails follow visible transport; chain dogs, drive and station stop are assumed.',
            'Only a two-metre upper straight chain stroke is represented; hidden return chain is omitted.',
            'Wheelset is free under gravity, with primitive contacts; only chain-dog joint receives bounded force.',
            'Mass, inertia, dimensions and contact parameters are estimates; not industrial acceptance.',
            'This isolated conveyor is not evidence of complete workshop transport or SCOUT loading.'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
