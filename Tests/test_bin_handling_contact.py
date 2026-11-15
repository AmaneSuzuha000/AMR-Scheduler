"""Support, source freshness and stop requirements for the isolated bin fixture."""
from copy import deepcopy
import hashlib
import json
import xml.etree.ElementTree as ET

import pytest

from tools.wheelset_scene.bin_handling_observer import BinHandlingObserver
from tools.wheelset_scene.bin_handling_world import build


def test_equipment_drain_budget_defaults_are_frozen():
    """The owner turn budget stays at the previously frozen 20 ms / 32 rows."""
    from tools.wheelset_scene.navigation_bin_runtime import ContactBinRuntime
    assert ContactBinRuntime.drain_budget_s == .020
    assert ContactBinRuntime.drain_rows == 32


@pytest.mark.parametrize('lift,minimum_gap',[(0.,-.008),(.02,.012)])
def test_handoff_shelf_height_exposes_legacy_chassis_overlap_and_raised_clearance(lift,minimum_gap):
    from tools.wheelset_scene.scout_bin_hardware import add_docks
    from tools.wheelset_scene.light_material_network import proposed_network
    world=ET.Element('world')
    profile=add_docks(world,proposed_network(),direction='longitudinal',platform_lift_m=lift)
    # Actual converted metric chassis: centre z=.173, full height=.184.
    chassis_top=.173+.184/2
    gaps=[]
    for task in profile['tasks'].values():
        for station in task['stations'].values():
            rack=world.find("model[@name='"+station['support_collision'].split('::')[0]+"']")
            support=rack.find("link/collision[@name='support']")
            pose=list(map(float,support.findtext('pose').split()))
            size=list(map(float,support.findtext('geometry/box/size').split()))
            gaps.append(pose[2]-size[2]/2-chassis_top)
    assert min(gaps)==pytest.approx(minimum_gap,abs=1e-12)
    if lift:
        # Preserve the original source/sink drops and sensor/cargo clearance.
        assert profile['geometry']['loaded_bin_local_m'][2]==pytest.approx(.36)
        assert profile['geometry']['deck_top_local_m']==pytest.approx(.32)
        assert .4+lift-(profile['geometry']['deck_top_local_m']+.08)==pytest.approx(.02)
        assert all(gap>.01 for gap in gaps)
    else:
        assert all(gap<0 for gap in gaps)  # Negative example remains explicit.


@pytest.mark.parametrize('direction',['lateral','longitudinal'])
def test_raised_carrier_moves_deck_finger_and_bin_reference_together(tmp_path,direction):
    from tools.wheelset_scene.scout_bin_hardware import add_carrier,carrier_geometry
    sdf=ET.fromstring('<sdf><model name="scout_a"><link name="base_footprint"><collision name="metric_chassis_collision"/></link></model></sdf>')
    library=tmp_path/'axis.so';library.write_bytes(b'geometry-only fixture')
    add_carrier(sdf,'scout_a',library,direction=direction,platform_lift_m=.02)
    deck=list(map(float,sdf.findtext("model/link[@name='handling_deck']/pose").split()))
    finger=list(map(float,sdf.findtext("model/link[@name='bin_transfer_finger']/pose").split()))
    geometry=carrier_geometry(direction,.02)
    assert deck[2]==pytest.approx(.30)
    assert finger[2]==pytest.approx(geometry['loaded_bin_local_m'][2])
    assert finger[2]==pytest.approx(.36)


@pytest.mark.parametrize('lift',[-.01,.01,.03,float('nan')])
def test_undeclared_platform_height_is_rejected_before_modifying_world(lift):
    from tools.wheelset_scene.scout_bin_hardware import add_docks
    world=ET.Element('world')
    with pytest.raises(ValueError,match='raised carrier'):
        add_docks(world,{},platform_lift_m=lift)
    assert not list(world)


def test_actual_sdf_preflight_refuses_old_overlap_and_certifies_the_raised_envelope():
    from tools.wheelset_scene.scout_bin_hardware import add_docks,audit_contact_clearance
    from tools.wheelset_scene.light_material_network import proposed_network
    for lift in (0.,.02):
        world=ET.Element('world')
        profile=add_docks(world,proposed_network(),direction='longitudinal',platform_lift_m=lift)
        sdf=ET.fromstring(f'''<sdf><model><link name="base_footprint">
          <collision name="metric_chassis_collision"><pose>.0177 0 .173 0 0 0</pose>
          <geometry><box><size>.562 .392 .184</size></box></geometry></collision>
          <sensor type="ray"><pose>.15 0 {.4+lift} 0 0 0</pose></sensor></link></model></sdf>''')
        if not lift:
            with pytest.raises(ValueError,match='geometry invalid'):
                audit_contact_clearance(sdf,world,profile)
        else:
            result=audit_contact_clearance(sdf,world,profile)
            assert result['passed'] and len(result['checks'])==12 and result['laser_valid']
            assert result['chassis_envelope_z_inflation_m']>.006
            assert result['chassis_capture_xy_inflation_m']>.04
            # The declared map/fit height must correspond to the real ray.
            sdf.find('.//sensor/pose').text='.15 0 .4 0 0 0'
            with pytest.raises(ValueError,match='geometry invalid'):
                audit_contact_clearance(sdf,world,profile)


def sample(now, operation='LOAD'):
    header = dict(stamp=dict(sec=int(now), nanosec=round((now-int(now))*1e9)), frame_id='world')
    zero = dict(x=0., y=0., z=0.)
    x = 0. if operation == 'LOAD' else 2.
    def state(x, y, z):
        return dict(header=deepcopy(header), success=True, state=dict(
            pose=dict(position=dict(x=x, y=y, z=z), orientation=dict(x=0., y=0., z=0., w=1.)),
            twist=dict(linear=dict(zero), angular=dict(zero))))
    carrier = state(x, 0., 0.)
    cargo = state(x, 0. if operation == 'LOAD' else .80, .62)
    support = 'handling_shuttle::carrier::deck' if operation == 'LOAD' else 'cell::body::support'
    contact = dict(header=deepcopy(header), states=[dict(collision1_name='material_bin::body::bin',
        collision2_name=support, info=f'time: [{now}]', contact_normals=[dict(x=0., y=0., z=1.)],
        total_wrench=dict(force=dict(x=0., y=0., z=19.62)))])
    source_now = header['stamp']['sec']+header['stamp']['nanosec']*1e-9
    return dict(source_now=source_now, carrier=carrier, cargo=cargo, contact=contact)


def command(operation='LOAD'):
    return dict(handling_id='run:carrier:1', issued_at=0., operation=operation,
                robot_id='carrier', node_id='store' if operation == 'LOAD' else 'cell',
                target_xy=[0. if operation == 'LOAD' else 2., 0.])


@pytest.mark.parametrize('operation', ['LOAD', 'UNLOAD'])
def test_only_continuous_stationary_support_emits_a_durable_receipt(operation):
    written = []
    observer = BinHandlingObserver(lambda identity, raw: written.append((identity, raw)))
    receipt = None
    for index in range(10):
        receipt = observer.observe(command(operation), **sample(index*.04, operation))
        if index < 8: assert receipt is None
        if receipt: break
    assert receipt['supported_by'] == ('carrier' if operation == 'LOAD' else 'cell')
    identity, raw = written[0]
    assert hashlib.sha256(raw).hexdigest() == identity == receipt['evidence_id']
    assert json.loads(raw)['support_force_n'] == pytest.approx(19.62)


@pytest.mark.parametrize('corruption', ['other_bin', 'other_support', 'stale', 'moving',
                                      'falling', 'no_force', 'side_contact'])
def test_arrival_or_corrupted_contact_never_confirms_a_transfer(corruption):
    written = []
    observer = BinHandlingObserver(lambda *values: written.append(values))
    for index in range(20):
        observation = sample(index*.04)
        contact = observation['contact']['states'][0]
        if corruption == 'other_bin': contact['collision1_name'] = 'other_bin::body::bin'
        if corruption == 'other_support': contact['collision2_name'] = 'source_rack::body::support'
        if corruption == 'stale': observation['source_now'] += .2
        if corruption == 'moving': observation['carrier']['state']['twist']['linear']['x'] = .02
        if corruption == 'falling': observation['cargo']['state']['pose']['position']['z'] = .50
        if corruption == 'no_force': contact['total_wrench']['force']['z'] = 0.
        if corruption == 'side_contact': contact['contact_normals'][0] = dict(x=1., y=0., z=0.)
        assert observer.observe(command(), **observation) is None
    assert written == []


def test_duplicate_samples_and_sensor_gap_do_not_accumulate_support_time():
    written = []
    observer = BinHandlingObserver(lambda *values: written.append(values))
    for now in (0., .1, .2, .2, .2, .5, .6, .7):
        assert observer.observe(command(), **sample(now)) is None
    assert observer.observe(command(), **sample(.81)) is not None
    assert len(written) == 1


def test_evidence_write_failure_prevents_the_receipt():
    def failed_write(*_): raise OSError('disk full')
    observer = BinHandlingObserver(failed_write)
    for now in (0., .1, .2):
        assert observer.observe(command(), **sample(now)) is None
    with pytest.raises(OSError, match='disk full'):
        observer.observe(command(), **sample(.31))


def test_world_drives_carrier_and_pushers_but_never_the_free_bin(tmp_path):
    library = tmp_path/'fixture.so'; library.write_bytes(b'test-library-placeholder')
    manifest = build(tmp_path/'world', library)
    world = ET.parse(tmp_path/'world/handling.world').getroot().find('world')
    cargo = world.find("model[@name='material_bin']")
    assert cargo.findtext('link/gravity') == 'true'
    assert not cargo.findall('joint') and not cargo.findall('plugin')
    assert len(cargo.findall('link/collision')) == 1
    assert cargo.findtext('link/sensor/contact/collision') == 'bin'
    carrier = world.find("model[@name='handling_shuttle']")
    assert manifest['axes'] == [a.findtext('joint') for a in carrier.findall('plugin/axis')]
    assert world.find("model[@name='source_rack']/link/collision") is not None
    assert world.find("model[@name='cell']/link/collision") is not None


def scout_observation(now, operation='LOAD'):
    import math
    from tools.wheelset_scene.light_material_network import proposed_network
    from tools.wheelset_scene.scout_bin_hardware import add_docks
    world = ET.Element('world')
    profile = add_docks(world, proposed_network())
    task = 'inspection_consumables'; robot = 'scout_a'
    station = profile['tasks'][task]['stations'][operation]
    cmd = dict(handling_id='run:scout_a:1', scene_id=profile['scene_id'], run_id='run',
        revision=1, task_id=task, issued_at=0., operation=operation, robot_id=robot,
        node_id=station['node_id'], target_xy=station['target_xy'], source=profile['source'],
        frame_id='map', time_base='ros_sim')
    inputs = sample(now, operation)
    carrier = inputs['carrier']; cargo = inputs['cargo']; contact = inputs['contact']
    p = carrier['state']['pose']['position']; p.update(x=station['target_xy'][0], y=station['target_xy'][1], z=0.)
    q = carrier['state']['pose']['orientation']; q.update(z=math.sin(station['yaw_rad']/2), w=math.cos(station['yaw_rad']/2))
    bp = cargo['state']['pose']['position']
    if operation == 'LOAD':
        bp.update(x=p['x'], y=p['y'], z=.34)
        support = robot+'::handling_deck::support'
    else:
        bp.update(zip(('x', 'y', 'z'), station['bin_world_xyz']))
        support = station['support_collision']
    contact['_topic'] = '/scout_bin/'+task+'/bin_contacts'
    contact['states'][0].update(collision1_name=profile['tasks'][task]['cargo_collision'], collision2_name=support)
    actuator = dict(header=deepcopy(contact['header']),
        name=['bin_transfer_stroke', robot+'_servo_sequence', robot+'_servo_fault'],
        position=[0., 5., 0.], velocity=[0., 0., 0.], _retraction_sequence=4,
        _topic='/scout_bin/'+robot+'/joint_states')
    def observed(entity, reply):
        return dict(request=dict(name=entity, reference_frame='world'), reply=reply)
    inputs.update(carrier=observed(robot+'::base_footprint', carrier),
                  cargo=observed(profile['tasks'][task]['cargo_entity'], cargo), actuator=actuator)
    return profile, cmd, inputs


@pytest.mark.parametrize('operation', ['LOAD', 'UNLOAD'])
def test_frozen_contact_representation_preserves_original_receipt_bytes(operation):
    from tools.wheelset_scene.navigation_handling import DockedBinObserver
    from tools.wheelset_scene.navigation_records import freeze_contact_message
    from tools.wheelset_scene.navigation_samples import EquipmentSamples
    profile,cmd,_=scout_observation(0.,operation)
    writes=[[],[]]
    observers=[DockedBinObserver(profile,lambda key,raw: writes[0].append((key,raw))),
               DockedBinObserver(profile,lambda key,raw: writes[1].append((key,raw)))]
    samples=EquipmentSamples(serialize_contacts=True)
    receipts=[None,None]
    for index in range(9):
        now=index*.04
        _,_,raw=scout_observation(now,operation)
        raw['contact']['_received_wall']=100.+now
        topic=raw['contact']['_topic']
        packet=freeze_contact_message(raw['contact'],topic,100.+now)
        samples.add(topic,packet,source_s=now,received_wall=100.+now)
        restored=samples.select(topic,source_now=now,wall_now=100.+now)
        receipts[0]=observers[0].observe(cmd,**raw)
        receipts[1]=observers[1].observe(cmd,**dict(raw,contact=restored))
    assert receipts[0] is not None and receipts[0]==receipts[1]
    assert writes[0]==writes[1] and len(writes[0])==1


@pytest.mark.parametrize('operation', ['LOAD', 'UNLOAD'])
def test_scout_handoff_uses_task_dock_geometry_and_raw_query_identities(operation):
    from tools.wheelset_scene.navigation_handling import DockedBinObserver
    profile, cmd, _ = scout_observation(0., operation); written = []
    observer = DockedBinObserver(profile, lambda key, raw: written.append((key, raw)))
    for index in range(9):
        _, _, observation = scout_observation(index*.04, operation)
        receipt = observer.observe(cmd, **observation)
    assert receipt['supported_by'] == ('scout_a' if operation == 'LOAD' else cmd['node_id'])
    key, raw = written[0]
    assert hashlib.sha256(raw).hexdigest() == receipt['evidence_id'] == key
    assert json.loads(raw)['carrier']['request']['name'] == 'scout_a::base_footprint'
    assert observer.observe(cmd, **observation) is None  # One durable receipt per request.


@pytest.mark.parametrize('corruption', ['wrong_carrier', 'wrong_cargo', 'wrong_dock', 'wrong_scene',
    'other_support', 'other_contact_topic', 'other_actuator_topic', 'contact_frame', 'stale',
    'stale_physics_force', 'moving', 'misaligned', 'heading', 'finger_extended',
    'finger_moving', 'bad_quaternion', 'nonfinite', 'wrong_mass'])
def test_scout_observer_rejects_unrelated_or_incomplete_transfer_evidence(corruption):
    from tools.wheelset_scene.navigation_handling import DockedBinObserver
    profile, cmd, _ = scout_observation(0.); written = []
    observer = DockedBinObserver(profile, lambda *args: written.append(args))
    for index in range(20):
        _, _, obs = scout_observation(.5+index*.04)
        if corruption == 'wrong_carrier': obs['carrier']['request']['name'] = 'scout_b::base_footprint'
        if corruption == 'wrong_cargo': obs['cargo']['request']['name'] = 'bin_small_tools'
        if corruption == 'wrong_dock': cmd['node_id'] = 'supply'
        if corruption == 'wrong_scene': cmd['scene_id'] = 'another_scene'
        if corruption == 'other_support': obs['contact']['states'][0]['collision2_name'] = 'dock::support'
        if corruption == 'other_contact_topic': obs['contact']['_topic'] = '/scout_bin/small_tools/bin_contacts'
        if corruption == 'other_actuator_topic': obs['actuator']['_topic'] = '/scout_bin/scout_b/joint_states'
        if corruption == 'contact_frame': obs['contact']['header']['frame_id'] = 'map'
        if corruption == 'stale': obs['source_now'] += .2
        if corruption == 'stale_physics_force': obs['contact']['states'][0]['info'] = 'time: [0]'
        if corruption == 'moving': obs['cargo']['reply']['state']['twist']['linear']['x'] = .1
        if corruption == 'misaligned': obs['carrier']['reply']['state']['pose']['position']['x'] += .10
        if corruption == 'heading': obs['carrier']['reply']['state']['pose']['orientation'].update(z=0., w=1.)
        if corruption == 'finger_extended': obs['actuator']['position'][0] = .1
        if corruption == 'finger_moving': obs['actuator']['velocity'][0] = .1
        if corruption == 'bad_quaternion': obs['cargo']['reply']['state']['pose']['orientation']['w'] = .5
        if corruption == 'nonfinite': obs['cargo']['reply']['state']['pose']['position']['z'] = float('nan')
        if corruption == 'wrong_mass': obs['contact']['states'][0]['total_wrench']['force']['z'] = 1.
        assert observer.observe(cmd, **obs) is None
    assert not written


def test_scout_observer_resets_support_interval_after_a_query_gap_and_failed_write():
    from tools.wheelset_scene.navigation_handling import DockedBinObserver
    profile, cmd, _ = scout_observation(0.)
    def fail(*args): raise OSError('disk full')
    observer = DockedBinObserver(profile, fail)
    for now in (0., .1, .2, .5, .6, .7):
        assert observer.observe(cmd, **scout_observation(now)[2]) is None
    with pytest.raises(OSError, match='disk full'):
        observer.observe(cmd, **scout_observation(.81)[2])
    assert not observer.emitted


@pytest.mark.parametrize('misalignment', ['stale', 'future', 'far', 'missing'])
def test_scout_observer_waits_for_an_aligned_localized_pose_without_resetting_physical_support(misalignment):
    from tools.wheelset_scene.navigation_handling import DockedBinObserver
    profile, cmd, _ = scout_observation(0.)
    written = []; observer = DockedBinObserver(profile, lambda key, raw: written.append((key, raw)))
    for index in range(9):
        now = 1.+index*.04
        _, _, obs = scout_observation(now)
        pose = dict(x=cmd['target_xy'][0], y=cmd['target_xy'][1], yaw=0., timestamp=now)
        if misalignment == 'stale': pose['timestamp'] -= .2
        if misalignment == 'future': pose['timestamp'] += .01
        if misalignment == 'far': pose['x'] += 1.
        if misalignment == 'missing': pose = None
        receipt = observer.observe(cmd, **obs, execution_pose=pose, execution_tolerance_m=.15)
        assert receipt is None
    assert not written and not observer.emitted
    assert observer.last_wait_reason in ('localized_pose_not_aligned', 'localized_pose_invalid')
    stable_since = observer.since
    assert stable_since == pytest.approx(1.)

    now = 1.36
    _, _, obs = scout_observation(now)
    aligned = dict(x=cmd['target_xy'][0], y=cmd['target_xy'][1], yaw=0., timestamp=now)
    receipt = observer.observe(cmd, **obs, execution_pose=aligned, execution_tolerance_m=.15)
    assert receipt is not None and receipt['timestamp'] == pytest.approx(now)
    assert observer.emitted and len(written) == 1
    key, raw = written[0]
    evidence = json.loads(raw)
    assert evidence['execution_pose'] == aligned
    assert evidence['execution_tolerance_m'] == .15
    assert evidence['stable_since'] == stable_since
    assert hashlib.sha256(raw).hexdigest() == key == receipt['evidence_id']


def test_contact_runtime_passes_the_current_owner_pose_to_receipt_observer():
    from types import SimpleNamespace
    from tools.wheelset_scene.navigation_bin_runtime import ContactBinRuntime, TransferStroke
    from tools.wheelset_scene.navigation_samples import EquipmentSamples
    profile, cmd, obs = scout_observation(1.)
    adapter = ContactBinRuntime.__new__(ContactBinRuntime)
    adapter.profile = profile; adapter.samples = EquipmentSamples()
    adapter.actuators = {}; adapter.started = {cmd['handling_id']: (10., 1.)}
    adapter.retractions = {'scout_a': 4}; adapter.inhibitions = {}; adapter.receipt_waits = {}
    stroke = TransferStroke(profile['geometry']); stroke.command = cmd; stroke.phase = 'observing'
    adapter.strokes = {'scout_a': stroke}
    for key, row in [('scout_a::base_footprint', obs['carrier']),
                     (profile['tasks'][cmd['task_id']]['cargo_entity'], obs['cargo']),
                     ('/scout_bin/'+cmd['task_id']+'/bin_contacts', obs['contact']),
                     ('/scout_bin/scout_a/joint_states', obs['actuator'])]:
        if 'reply' in row: row['received_wall'] = 10.
        else: row['_received_wall'] = 10.
        adapter.samples.add(key, row, source_s=1., received_wall=10.)
    owner_pose = dict(x=cmd['target_xy'][0], y=cmd['target_xy'][1], yaw=0., timestamp=1.)
    guard = SimpleNamespace(state='latched', stopped=True, actual_speed=0., actual_wz=0.,
                            output_vx=0., output_wz=0.)
    node = SimpleNamespace(guard_status=guard, guard_fresh=lambda: True, clock_s=1.,
                           health=SimpleNamespace(reason=lambda *_: None))
    handling = SimpleNamespace(command=cmd, completed=False,
        execution=SimpleNamespace(fault=None, pose=owner_pose, tolerance=.15))
    adapter.runtime = SimpleNamespace(nodes={'scout_a': node},
        sim=SimpleNamespace(handlings={'scout_a': handling}))
    records = []; adapter.record = lambda kind, **values: records.append(dict(kind=kind, **values))
    adapter.publish = lambda *args: None
    class Observer:
        last_wait_reason = None
        def observe(self, command, **values):
            self.command = command; self.values = values
            return None
    observer = adapter.observers = {'scout_a': Observer()}
    adapter.poll_robot('scout_a', handling, 10.02, 1.)
    assert observer['scout_a'].values['execution_pose'] == owner_pose
    assert observer['scout_a'].values['execution_tolerance_m'] == .15
    selected = next(row for row in records if row['kind'] == 'bin_selected_samples')
    assert selected['execution_pose'] == owner_pose


@pytest.mark.parametrize('integral_gain', [0., 800.])
@pytest.mark.parametrize('direction', ['lateral', 'longitudinal'])
def test_scout_accessory_preserves_free_bins_and_includes_retracted_finger_in_footprint(tmp_path, integral_gain, direction):
    import math
    from tools.wheelset_scene.light_material_network import proposed_network
    from tools.wheelset_scene.scout_bin_hardware import add_carrier, add_docks
    sdf = ET.fromstring('<sdf version="1.7"><model name="scout_a"><link name="base_footprint"><collision name="metric_chassis_collision"/></link></model></sdf>')
    library = tmp_path/'axis.so'; library.write_bytes(b'placeholder')
    carrier = add_carrier(sdf, 'scout_a', library, integral_gain=integral_gain, direction=direction)
    world = ET.Element('world'); layout = proposed_network(); profile = add_docks(world, layout, direction=direction)
    for spec in profile['tasks'].values():
        cargo = world.find("model[@name='"+spec['cargo_entity']+"']")
        assert cargo.findtext('link/gravity') == 'true'
        assert not cargo.findall('joint') and not cargo.findall('plugin')
        assert cargo.find('link/sensor/contact/collision').text == 'bin'
    assert carrier['support_collision'] == 'scout_a::handling_deck::support'
    assert sdf.find('model/joint/parent').text == 'base_footprint'
    assert float(sdf.findtext("model/plugin[@name='bin_transfer_drive']/axis/integral_gain_n_per_m_s")) == integral_gain
    assert all(math.hypot(x, y) <= layout['robot_enclosing_radius_m'] for x, y in profile['geometry']['footprint_m'])
    assert all(robot['start_yaw_rad'] < -1.5 for robot in layout['robots'])
    with pytest.raises(ValueError, match='already installed'):
        add_carrier(sdf, 'scout_a', library)


@pytest.mark.parametrize('direction', ['lateral', 'longitudinal'])
def test_transfer_axis_and_dock_geometry_match_load_and_unload_endpoints(tmp_path, direction):
    import math
    from tools.wheelset_scene.light_material_network import proposed_network
    from tools.wheelset_scene.scout_bin_hardware import add_carrier, add_docks
    sdf = ET.fromstring('<sdf version="1.7"><model name="scout_a"><link name="base_footprint"><collision name="metric_chassis_collision"/></link></model></sdf>')
    library = tmp_path/'axis.so'; library.write_bytes(b'placeholder')
    add_carrier(sdf, 'scout_a', library, direction=direction)
    world = ET.Element('world'); profile = add_docks(world, proposed_network(), direction=direction)
    geometry = profile['geometry']; dimension = 0 if direction == 'longitudinal' else 1
    axis = sdf.find("model/joint[@name='bin_transfer_stroke']/axis/xyz")
    assert axis.get('expressed_in') == 'base_footprint'
    assert list(map(float, axis.text.split())) == ([1., 0., 0.] if dimension == 0 else [0., 1., 0.])
    origin = geometry['pusher_origin_local_m'][dimension]
    offset = (geometry['pusher_size_m'][dimension]+geometry['bin_size_m'][dimension])/2
    assert origin+geometry['load_stroke_m']+offset == pytest.approx(0.)
    if direction == 'longitudinal':
        hull = geometry['footprint_m']
        for half_x, half_y, center_x in [(.315, .252, 0.), (.20, .25, 0.), (.02, .10, -.49)]:
            for sign_x in (-1, 1):
                for sign_y in (-1, 1):
                    point = (center_x+sign_x*half_x, sign_y*half_y)
                    for start, end in zip(hull, hull[1:]+hull[:1]):
                        assert (end[0]-start[0])*(point[1]-start[1])-(end[1]-start[1])*(point[0]-start[0]) >= -1e-12
    unload_offset = origin+geometry['unload_stroke_m']+offset
    for task in profile['tasks'].values():
        station = task['stations']['UNLOAD']; yaw = station['yaw_rad']
        local = [0., 0.]; local[dimension] = unload_offset
        expected = [station['target_xy'][0]+math.cos(yaw)*local[0]-math.sin(yaw)*local[1],
                    station['target_xy'][1]+math.sin(yaw)*local[0]+math.cos(yaw)*local[1]]
        assert station['bin_world_xyz'][:2] == pytest.approx(expected)
        rack = world.find("model[@name='"+station['support_collision'].split('::')[0]+"']")
        assert list(map(float, rack.findtext('pose').split()))[:2] == pytest.approx(expected)
    assert profile['transfer_direction'] == direction


@pytest.mark.parametrize('direction', [None, 'forward'])
def test_undefined_transfer_direction_cannot_create_a_dock_or_claim_geometry(direction):
    from tools.wheelset_scene.scout_bin_hardware import add_docks
    world = ET.Element('world')
    with pytest.raises(ValueError, match='direction'):
        add_docks(world, {}, direction=direction)
    assert not list(world)


@pytest.mark.parametrize('step', [0., .003, float('nan'), True])
def test_unvalidated_physics_step_is_rejected_before_creating_a_bundle(tmp_path, step):
    from tools.wheelset_scene.workshop_navigation import build_bundle
    output = tmp_path/'not-created'
    with pytest.raises(ValueError, match='physics step'):
        build_bundle(output, physics_step_s=step)
    assert not output.exists()


@pytest.mark.parametrize('operation', ['LOAD', 'UNLOAD'])
def test_contact_transfer_retracts_before_it_can_emit_a_receipt(operation):
    from tools.wheelset_scene.navigation_bin_runtime import TransferStroke
    from tools.wheelset_scene.scout_bin_hardware import GEOMETRY
    _, cmd, obs = scout_observation(1., operation)
    stroke = TransferStroke(GEOMETRY)
    extended = GEOMETRY['load_stroke_m' if operation == 'LOAD' else 'unload_stroke_m']
    assert stroke.update(cmd, obs['actuator'], 1.) == extended
    assert stroke.phase == 'extending'
    obs['actuator']['position'] = [extended]
    assert stroke.update(cmd, obs['actuator'], 1.) == 0.
    assert stroke.phase == 'retracting'
    assert stroke.update(cmd, obs['actuator'], 1.2) is None  # No stale confirmation.
    obs['actuator']['position'] = [0.]; obs['actuator']['velocity'] = [.02]
    assert stroke.update(cmd, obs['actuator'], 1.) == 0.
    assert stroke.phase == 'retracting'
    obs['actuator']['velocity'] = [0.]
    assert stroke.update(cmd, obs['actuator'], 1.) == 0.
    assert stroke.phase == 'observing'


def test_restored_contact_transfer_never_repeats_an_unconfirmed_physical_push():
    from tools.wheelset_scene.navigation_bin_runtime import TransferStroke
    from tools.wheelset_scene.scout_bin_hardware import GEOMETRY
    _, cmd, obs = scout_observation(1.)
    cmd['reconcile_only'] = True
    stroke = TransferStroke(GEOMETRY)
    assert stroke.update(cmd, obs['actuator'], 1.) == 0.
    assert stroke.phase == 'observing'


def test_ros_joint_arrays_survive_raw_logging_and_durable_contact_evidence():
    from array import array
    from tools.wheelset_scene.navigation_handling import DockedBinObserver
    from tools.wheelset_scene.navigation_records import plain_ros_value
    profile, cmd, _ = scout_observation(0.)
    written = []
    observer = DockedBinObserver(profile, lambda key, raw: written.append((key, raw)))
    receipt = None
    for index in range(9):
        _, _, obs = scout_observation(index*.04)
        for field in ('position', 'velocity'):
            obs['actuator'][field] = array('d', obs['actuator'][field])
        normalized = plain_ros_value(obs)
        logged = json.loads(json.dumps(normalized, allow_nan=False))
        assert logged['actuator']['position'] == [0., 5., 0.]
        assert isinstance(obs['actuator']['position'], array)  # Raw message remains unchanged.
        receipt = observer.observe(cmd, **normalized)
    key, raw = written[0]
    assert receipt['evidence_id'] == key == hashlib.sha256(raw).hexdigest()
    assert json.loads(raw)['actuator']['position'] == [0., 5., 0.]


def test_ros_array_normalization_keeps_nonfinite_evidence_rejected():
    from array import array
    from tools.wheelset_scene.navigation_records import plain_ros_value
    raw = dict(targets=array('d', [1., float('nan')]))
    with pytest.raises(ValueError):
        json.dumps(plain_ros_value(raw), allow_nan=False)


@pytest.mark.parametrize('handling_state', ['waiting_guard', 'no_command', 'completed'])
def test_native_axis_fault_is_visible_before_other_handoff_conditions(handling_state):
    from types import SimpleNamespace
    from tools.wheelset_scene.navigation_bin_runtime import ContactBinRuntime
    adapter = ContactBinRuntime.__new__(ContactBinRuntime)
    _, cmd, observation = scout_observation(1.)
    observation['actuator']['position'][2] = 1.
    adapter.actuators = {'scout_a': observation['actuator']}
    published = []
    adapter.publish = lambda *args: published.append(args)
    handling = SimpleNamespace(command=None if handling_state == 'no_command' else cmd,
                               completed=handling_state == 'completed')
    with pytest.raises(RuntimeError, match='lease fault: scout_a'):
        adapter.poll_robot('scout_a', handling, 1., 1.)
    assert not published


def test_idle_contact_lease_is_disabled_zero_and_cannot_create_a_retraction():
    from types import SimpleNamespace
    from tools.wheelset_scene.navigation_bin_runtime import ContactBinRuntime, TransferStroke
    from tools.wheelset_scene.scout_bin_hardware import GEOMETRY
    adapter = ContactBinRuntime.__new__(ContactBinRuntime)
    adapter.actuators = {}; adapter.strokes = {'scout_a': TransferStroke(GEOMETRY)}
    adapter.sequences = {'scout_a': -1}; adapter.targets = {'scout_a': 0.}
    adapter.target_clocks = {}; adapter.retractions = {}; adapter.source_now = 1.
    published = []
    adapter.publishers = {'scout_a': SimpleNamespace(publish=lambda message: published.append(list(message.data)))}
    adapter.message = lambda: SimpleNamespace(data=None)
    records = []
    adapter.record = lambda kind, **values: records.append(dict(kind=kind, **values))
    handling = SimpleNamespace(command=None, completed=False)
    adapter.runtime = SimpleNamespace(sim=SimpleNamespace(handlings={'scout_a': handling}))
    adapter.poll_robot('scout_a', handling, 1., 1.)
    assert published == [[0., 0., 0.]]
    assert adapter.retractions == {}
    assert adapter.strokes['scout_a'].phase == 'idle'
    assert records[-1]['active'] is False


def test_missing_active_equipment_sample_keeps_lease_alive_without_advancing_transfer():
    from types import SimpleNamespace
    from tools.wheelset_scene.navigation_bin_runtime import ContactBinRuntime, TransferStroke
    from tools.wheelset_scene.navigation_samples import EquipmentSamples
    profile, cmd, _ = scout_observation(1.)
    adapter = ContactBinRuntime.__new__(ContactBinRuntime)
    adapter.profile = profile; adapter.samples = EquipmentSamples()
    adapter.actuators = {}; adapter.contacts = {}; adapter.states = {}
    adapter.strokes = {'scout_a': TransferStroke(profile['geometry'])}
    adapter.started = {}; adapter.retractions = {}; adapter.inhibitions = {}
    adapter.sequences = {'scout_a': -1}; adapter.targets = {'scout_a': 0.}
    adapter.target_clocks = {}; adapter.source_now = 1.
    guard = SimpleNamespace(state='latched', stopped=True, actual_speed=0.,
                            actual_wz=0., output_vx=0., output_wz=0.)
    node = SimpleNamespace(guard_status=guard, guard_fresh=lambda: True, clock_s=1.,
                           health=SimpleNamespace(reason=lambda *_: None))
    handling = SimpleNamespace(command=cmd, completed=False,
                               execution=SimpleNamespace(fault=None))
    adapter.runtime = SimpleNamespace(
        nodes={'scout_a': node}, sim=SimpleNamespace(handlings={'scout_a': handling}))
    published = []; records = []
    adapter.publishers = {'scout_a': SimpleNamespace(
        publish=lambda message: published.append(list(message.data)))}
    adapter.message = lambda: SimpleNamespace(data=None)
    adapter.record = lambda kind, **values: records.append(dict(kind=kind, **values))
    adapter.poll_robot('scout_a', handling, 1., 1.)
    assert published == [[0., 0., 1.]]
    assert adapter.retractions == {}
    assert adapter.strokes['scout_a'].phase == 'idle'
    assert records[0]['kind'] == 'bin_transfer_target'
    assert records[0]['active'] is True
    assert records[0]['retraction_registration'] is False
    assert records[1]['kind'] == 'bin_transfer_inhibited'
    assert records[1]['lease_renewal'] == 'enabled_zero_target'


def test_unsafe_guard_withholds_transfer_lease_and_records_source_times():
    from types import SimpleNamespace
    from tools.wheelset_scene.navigation_bin_runtime import ContactBinRuntime
    adapter = ContactBinRuntime.__new__(ContactBinRuntime)
    _, cmd, observation = scout_observation(1.)
    observation['actuator']['_received_wall'] = 1.
    adapter.actuators = {'scout_a': observation['actuator']}
    adapter.contacts = {}; adapter.states = {}; adapter.source_now = 1.; adapter.inhibitions = {}
    guard = SimpleNamespace(state='latched', stopped=False, actual_speed=.02,
                            actual_wz=0., output_vx=0., output_wz=0.)
    node = SimpleNamespace(guard_status=guard, guard_fresh=lambda: True, clock_s=1.,
                           health=SimpleNamespace(reason=lambda *_: None))
    adapter.runtime = SimpleNamespace(nodes={'scout_a': node})
    published = []; records = []
    adapter.publish = lambda *args: published.append(args)
    adapter.record = lambda kind, **values: records.append(dict(kind=kind, **values))
    handling = SimpleNamespace(command=cmd, completed=False, execution=SimpleNamespace(fault=None))
    adapter.poll_robot('scout_a', handling, 1., 1.)
    assert not published
    assert records[0]['guard']['stopped'] is False
    assert records[0]['equipment_times']['scout_a']['source_s'] == 1.
    assert records[0]['kind'] == 'bin_transfer_inhibited'


def test_contact_runtime_uses_causal_history_without_bypassing_latest_native_fault():
    from types import SimpleNamespace
    from tools.wheelset_scene.navigation_bin_runtime import ContactBinRuntime, TransferStroke
    from tools.wheelset_scene.navigation_samples import EquipmentSamples
    adapter = ContactBinRuntime.__new__(ContactBinRuntime)
    profile, cmd, obs = scout_observation(1.)
    future = scout_observation(1.1)[2]['actuator']
    adapter.profile = profile; adapter.samples = EquipmentSamples()
    adapter.actuators = {'scout_a': future}; adapter.started = {}; adapter.retractions = {}; adapter.inhibitions = {}
    adapter.strokes = {'scout_a': TransferStroke(profile['geometry'])}
    for key, row in [('scout_a::base_footprint', obs['carrier']),
                     (profile['tasks'][cmd['task_id']]['cargo_entity'], obs['cargo']),
                     ('/scout_bin/'+cmd['task_id']+'/bin_contacts', obs['contact']),
                     ('/scout_bin/scout_a/joint_states', obs['actuator'])]:
        if 'reply' in row: row['received_wall'] = 10.
        else: row['_received_wall'] = 10.
        adapter.samples.add(key, row, source_s=1., received_wall=10.)
    adapter.samples.add('/scout_bin/scout_a/joint_states', future, source_s=1.1, received_wall=10.01)
    guard = SimpleNamespace(state='latched', stopped=True, actual_speed=0., actual_wz=0., output_vx=0., output_wz=0.)
    node = SimpleNamespace(guard_status=guard, guard_fresh=lambda: True, clock_s=1.,
                          health=SimpleNamespace(reason=lambda *_: None))
    adapter.runtime = SimpleNamespace(nodes={'scout_a': node})
    published = []; records = []
    adapter.publish = lambda *args: published.append(args)
    adapter.record = lambda kind, **values: records.append(dict(kind=kind, **values))
    handling = SimpleNamespace(command=cmd, completed=False, execution=SimpleNamespace(fault=None,
        pose=dict(x=cmd['target_xy'][0], y=cmd['target_xy'][1], yaw=0., timestamp=1.), tolerance=.15))
    adapter.poll_robot('scout_a', handling, 10.02, 1.)
    assert published == [('scout_a', profile['geometry']['load_stroke_m'])]
    assert records[0]['times']['actuator']['source_s'] == 1.
    future['position'][2] = 1.
    with pytest.raises(RuntimeError, match='lease fault'):
        adapter.poll_robot('scout_a', handling, 10.03, 1.)
    assert len(published) == 1


def test_contact_trace_replay_preserves_integer_receipt_identity_and_checks_raw_bytes(tmp_path):
    from tools.wheelset_scene.navigation_handling import DockedBinObserver
    from tools.wheelset_scene.probe_scout_transfer import digest, verify
    import tools.wheelset_scene.probe_scout_transfer as probe
    from pathlib import Path
    profile, cmd, _ = scout_observation(1.)
    rows = []
    def write(key, raw): (tmp_path/(key+'.json')).write_bytes(raw)
    live = DockedBinObserver(profile, write); receipt = None
    rows.append(dict(kind='handling_request', command=cmd))
    for index in range(9):
        now = 1.+index*.04; wall = 10.+index*.04
        _, _, obs = scout_observation(now)
        obs['actuator'].update(effort=[0., 0., 0.], _received_wall=wall)
        # The native wire protocol is float64, but its accepted sequence and
        # the durable observer marker are integers in the execution adapter.
        obs['actuator']['_retraction_sequence'] = 4
        obs['contact']['_received_wall'] = wall
        for name in ('carrier', 'cargo'): obs[name]['received_wall'] = wall
        selected = {name: obs[name] for name in ('carrier', 'cargo', 'actuator', 'contact')}
        selected['guard'] = dict(header=deepcopy(obs['contact']['header']), _received_wall=wall, actual_speed=0.)
        for key, value in selected.items():
            if key in ('carrier', 'cargo'): rows.append(dict(kind='state', key=key, observation=value))
            else: rows.append(dict(kind='sample', key=key, message=value))
        rows.append(dict(kind='selected', source_s=now, wall_now=wall,
            times={key: dict(source_s=now, received_wall=wall) for key in selected}))
        rows.append(dict(kind='target', values=[4., 0., 1.], requested_target_m=0., source_s=now, phase='observing'))
        receipt = live.observe(cmd, **{key: value for key, value in obs.items() if key != 'source_now'}, source_now=now)
    assert receipt is not None
    (tmp_path/'observations.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (tmp_path/'executed_probe.py').write_bytes(Path(probe.__file__).read_bytes())
    inputs = dict(profile=profile, fault='none', guided=False, integral_gain_n_per_m_s=800., seed=17,
                  source_sha256={Path(probe.__file__).name: digest(probe.__file__)}, boundary='synthetic audit regression only')
    (tmp_path/'inputs.json').write_text(json.dumps(inputs))
    result = dict(receipt=receipt, cleanup=dict(passed=True), error=None,
        files_sha256={path.name: digest(path) for path in tmp_path.iterdir() if path.is_file()})
    (tmp_path/'result.json').write_text(json.dumps(result))
    audited = verify(tmp_path, audit_name='replayed.json')
    assert audited['passed'] and audited['receipt'] == receipt
    assert audited['receipt']['evidence_id'] == receipt['evidence_id']
    raw_path = tmp_path/'observations.jsonl'
    raw_path.write_bytes(raw_path.read_bytes().replace(b'"kind":', b'"kind" :', 1))
    assert not verify(tmp_path, audit_name='tampered.json')['checks']['integrity']
