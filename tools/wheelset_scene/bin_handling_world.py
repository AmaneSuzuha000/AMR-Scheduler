"""A small contact fixture for the AMR loading contract, not a production line.

An assumed linear carrier substitutes for a mobile base. Gravity and contacts
carry the free bin; a carrier-mounted pusher performs both dock transfers.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .actuated_world import inertia, joint
from .build import e, save_xml
from .contact_conveyor_world import box, contacts


def build(output, library):
    output, library = Path(output).resolve(), Path(library).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not library.is_file():
        raise FileNotFoundError(library)
    root = ET.Element('sdf', version='1.7')
    world = e(root, 'world', name='bin_handoff_contract_fixture')
    e(world, 'gravity', '0 0 -9.81')
    physics = e(world, 'physics', name='contact', type='ode')
    e(physics, 'max_step_size', .001); e(physics, 'real_time_update_rate', 1000)
    solver = e(e(physics, 'ode'), 'solver')
    e(solver, 'type', 'quick'); e(solver, 'iters', 100)
    floor = e(world, 'model', name='floor'); e(floor, 'static', 'true')
    box(e(floor, 'link', name='body'), 'floor', [5, 4, .1], [1, 0, -.05], '.2 .25 .3 1')
    # Small descending interfaces avoid presenting a square vertical lip to
    # a sliding bin. These are explicit fixture dimensions, not plant data.
    for name, x, y, depth, top in [('source_rack', 0, -.74, .72, .503),
                                  ('cell', 2, .78, .8, .497)]:
        rack = e(world, 'model', name=name); e(rack, 'static', 'true')
        link = e(rack, 'link', name='body')
        box(link, 'support', [.8, depth, .08], [x, y, top-.04], '.15 .4 .6 1')
        for dx in (-.32, .32):
            box(link, 'leg_'+str(dx), [.06, .06, .42], [x+dx, y, .21], '.2 .3 .4 1')
    model = e(world, 'model', name='handling_shuttle')
    base = e(model, 'link', name='base'); inertia(base, 10, [1, 1, 1])
    joint(model, 'anchor', 'fixed', 'world', 'base', None)
    carrier = e(model, 'link', name='carrier'); inertia(carrier, 20, [2, 2, 2])
    box(carrier, 'deck', [.8, .7, .08], [0, 0, .46], '.15 .55 .25 1')
    joint(model, 'travel', 'prismatic', 'base', 'carrier', '1 0 0', [-.01, 2.01, 200, .15], 10)
    # A separate unloading finger parked behind the final cargo position also
    # blocked the incoming bin. One carriage-mounted pusher loads, travels with
    # the bin, then extends again at the receiving rack; it never teleports.
    link = e(model, 'link', name='transfer_pusher'); e(link, 'pose', '0 -.95 .65 0 0 0')
    inertia(link, 1, [.02, .02, .02])
    box(link, 'face', [.38, .05, .24], [0, 0, 0], '.9 .5 .05 1')
    joint(model, 'transfer_stroke', 'prismatic', 'carrier', 'transfer_pusher', '0 1 0', [0, 1.62, 50, .10], 5)
    plugin = e(model, 'plugin', name='drive', filename=str(library))
    e(e(plugin, 'ros'), 'namespace', '/bin_handling/drive'); e(plugin, 'lease_s', 1.)
    axes = ['travel', 'transfer_stroke']
    for name in axes:
        axis = e(plugin, 'axis'); e(axis, 'joint', name)
        # The low-gain travel servo stalled against bin/rack static friction.
        # Pushers use the existing bounded contact-position controller. Its
        # 'mass' is a gain parameter; the pusher's physical mass remains 1 kg.
        e(axis, 'mode', 'position' if name == 'travel' else 'probe')
        e(axis, 'mass', 25 if name == 'travel' else 20)
        e(axis, 'limit', 200 if name == 'travel' else 50); e(axis, 'damping', 10)
    cargo = e(world, 'model', name='material_bin'); e(cargo, 'pose', '0 -.70 .631 0 0 0')
    link = e(cargo, 'link', name='body'); inertia(link, 2, [.02, .02, .02])
    link.find('gravity').text = 'true'
    box(link, 'bin', [.24, .24, .24], [0, 0, 0], '.9 .65 .08 1')
    contacts(link, 'bin', 'bin_contacts')
    link.find('sensor/plugin/ros/namespace').text = '/bin_handling'
    state = e(world, 'plugin', name='independent_state', filename='libgazebo_ros_state.so')
    e(e(state, 'ros'), 'namespace', '/bin_handling'); e(state, 'update_rate', 50)
    for contact in world.findall('.//collision/surface/contact/ode'):
        contact.find('kd').text = '20000'
        contact.find('max_vel').text = '.003'
        contact.find('min_depth').text = '.00002'
    save_xml(root, output/'handling.world')
    manifest = dict(scope='isolated_bin_handling_contact_fixture', axes=axes, bin_mass_kg=2.,
        support_top_m=dict(source_rack=.503, carrier=.500, cell=.497), interface_gap_m=.02,
        pusher_controller=dict(mode='probe', effective_mass_gain=20, physical_mass_kg=1,
                               position_gain_n_m=2000, maximum_force_n=50),
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        world_sha256=hashlib.sha256((output/'handling.world').read_bytes()).hexdigest(),
        assumptions=['Linear carrier is a contact interface fixture, not a SCOUT or Nav2 test.',
            'Two-metre travel, racks, pushers, dimensions, masses and forces are design assumptions.',
            'The bin has gravity and no joint, pose-setting service or driven motion plugin.',
            'No industrial loading accuracy or complete production-line fidelity is claimed.'])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest
