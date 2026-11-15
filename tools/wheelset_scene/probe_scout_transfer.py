"""Isolated, source-bound SCOUT LOAD and native lease-fault contact assay.

This tool grants no navigation, transport, inventory or hardware credit.
"""
from copy import deepcopy
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import xml.etree.ElementTree as ET

from .build import e, save_xml
from .contact_conveyor_world import box
from .navigation_bin_runtime import TransferStroke
from .navigation_handling import DockedBinObserver
from .navigation_records import plain_ros_value, Records
from .navigation_samples import EquipmentSamples
from .verify_contact_conveyor import stamp


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    bundle = args.bundle.resolve()
    profile = json.loads((bundle/'contact_bins.json').read_text())
    profile['scene_id'] = 'isolated_scout_transfer_batch_v1'
    profile['robots'] = ['scout_a']
    task = 'inspection_consumables'; spec = profile['tasks'][task]
    profile['tasks'] = {task: spec}
    offset = list(spec['stations']['LOAD']['target_xy'])
    for station in spec['stations'].values():
        station['target_xy'] = [v-d for v, d in zip(station['target_xy'], offset)]
        station['bin_world_xyz'][:2] = [v-d for v, d in zip(station['bin_world_xyz'][:2], offset)]
    tree = ET.Element('sdf', version='1.7'); world = e(tree, 'world', name='scout_transfer_assay')
    original = ET.parse(bundle/'navigation.world').getroot().find('world')
    for key in ('gravity', 'physics'):
        value = original.find(key)
        if value is not None: world.append(deepcopy(value))
    floor = e(world, 'model', name='fixture_floor'); e(floor, 'static', 'true')
    box(e(floor, 'link', name='body'), 'ground', [8, 8, .1], [0, 0, -.05], '.2 .2 .2 1')
    for name in ('dock_inspection_consumables_load', 'bin_inspection_consumables'):
        model = deepcopy(original.find(f"model[@name='{name}']")); assert model is not None
        pose = list(map(float, model.findtext('pose').split()))
        pose[:2] = [v-d for v, d in zip(pose[:2], offset)]
        model.find('pose').text = ' '.join(map(str, pose)); world.append(model)
    model = ET.parse(args.model).getroot().find('model'); model.set('name', 'scout_a')
    assert model.find('pose') is None
    drive = model.find("plugin[@name='bin_transfer_drive']"); assert drive is not None
    build = json.loads((args.axis_build/'build_manifest.json').read_text())
    library = args.axis_build/'libwheelset_axis_servo.so'
    assert build['passed'] and digest(library) == build['library_sha256']
    drive.set('filename', str(library.resolve()))
    axis = drive.find('axis'); assert axis.findtext('mode') == 'probe'
    setting = axis.find('integral_gain_n_per_m_s')
    if setting is None: e(axis, 'integral_gain_n_per_m_s', args.integral_gain)
    else: setting.text = str(args.integral_gain)
    e(model, 'pose', f'0 0 .02 0 0 {-math.pi/2}'); world.append(model)
    if args.guided:
        rails = e(world, 'model', name='assumed_dock_guides'); e(rails, 'static', 'true')
        e(rails, 'pose', f'0 0 0 0 0 {-math.pi/2}')
        body = e(rails, 'link', name='body')
        for sign in (-1, 1):
            box(body, 'wheel_side_guide_'+str(sign), [.70, .04, .10], [0, sign*.275, .05], '.8 .5 .1 1')
    state = e(world, 'plugin', name='independent_state', filename='libgazebo_ros_state.so')
    e(e(state, 'ros'), 'namespace', '/wheelset_review'); e(state, 'update_rate', 50)
    save_xml(tree, out/'fixture.world')
    for name, path in [('executed_probe.py', Path(__file__)), ('original_scout.sdf', args.model),
                       ('libwheelset_axis_servo.so', library), ('axis_build.json', args.axis_build/'build_manifest.json')]:
        (out/name).write_bytes(path.read_bytes())
    inputs = dict(guided=args.guided, integral_gain_n_per_m_s=args.integral_gain, fault=args.fault,
        seed=args.seed, profile=profile, model_sha256=digest(args.model), world_sha256=digest(out/'fixture.world'),
        native_library_sha256=digest(library), budgets=dict(source_s=45., wall_s=180.),
        source_sha256={p.name: digest(p) for p in [Path(__file__), Path(__file__).with_name('navigation_samples.py'),
            Path(__file__).with_name('navigation_bin_runtime.py'), Path(__file__).with_name('navigation_handling.py'),
            Path(__file__).with_name('navigation_records.py')]},
        boundary='Exploratory single LOAD of assumed free 2kg bin; no navigation, transport, unloading, '
            'inventory, process, system, industrial payload or hardware acceptance.')
    (out/'inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
    return out, profile, inputs


def run(args):
    from .run_navigation_readiness import stop_launch_group
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', args.port)) == 0: raise RuntimeError('occupied port; preserved')
    os.environ.update(ROS_DOMAIN_ID=str(args.domain), GAZEBO_MASTER_URI=f'http://127.0.0.1:{args.port}',
        GAZEBO_MODEL_DATABASE_URI='', LIBGL_ALWAYS_SOFTWARE='1')
    out, profile, inputs = prepare(args)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from gazebo_msgs.msg import ContactsState
    from gazebo_msgs.srv import GetEntityState
    from sensor_msgs.msg import JointState
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import Float64MultiArray
    from amr_gazebo_guard.msg import GuardStatus
    from rosidl_runtime_py.convert import message_to_ordereddict
    started = time.monotonic(); log = (out/'observations.jsonl').open('x')
    def record(kind, **values):
        log.write(json.dumps(plain_ros_value(dict(kind=kind, wall_s=time.monotonic()-started, **values)), allow_nan=False)+'\n')
        log.flush()
    rclpy.init(); node = Node('scout_transfer_batch_probe')
    latest = {}; samples = EquipmentSamples(); pending = {}; injected = None
    def receive(key, msg):
        row = plain_ros_value(message_to_ordereddict(msg)); row['_received_wall'] = time.monotonic()
        if key in ('actuator', 'contact'):
            row['_topic'] = '/scout_bin/'+('scout_a/joint_states' if key == 'actuator' else 'inspection_consumables/bin_contacts')
        latest[key] = row; record('sample', key=key, message=row)
        if key != 'clock' and not (args.fault == 'source_freeze' and injected is not None and key != 'guard'):
            samples.add(key, row, source_s=stamp(row), received_wall=row['_received_wall'])
    subscriptions = []
    for message, topic, key in [(Clock, '/clock', 'clock'),
            (JointState, '/scout_bin/scout_a/joint_states', 'actuator'),
            (ContactsState, '/scout_bin/inspection_consumables/bin_contacts', 'contact'),
            (GuardStatus, '/scout_a/motion_guard/status', 'guard')]:
        subscriptions.append(node.create_subscription(message, topic, lambda m, k=key: receive(k, m), qos_profile_sensor_data))
    publisher = node.create_publisher(Float64MultiArray, '/scout_bin/scout_a/servo_targets', 1)
    getter = node.create_client(GetEntityState, '/wheelset_review/get_entity_state')
    command = None; sequence = -1; target = 0.; target_clock = None; next_query = -math.inf
    last_publish = -math.inf; retraction_sequence = None; stroke = TransferStroke(profile['geometry'])
    receipt = None; error = None; cleanup = None; source_now = None; native_fault = None; rejected = False
    def write_evidence(identity, raw):
        assert hashlib.sha256(raw).hexdigest() == identity
        with (out/(identity+'.json')).open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    observer = DockedBinObserver(profile, write_evidence)
    commandline = ['gzserver', str(out/'fixture.world'), '--seed', str(args.seed),
        '-slibgazebo_ros_init.so', '-slibgazebo_ros_factory.so', '-slibgazebo_ros_force_system.so']
    process = None
    try:
        with (out/'gazebo.log').open('xb') as output:
            process = subprocess.Popen(commandline, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        while time.monotonic()-started < inputs['budgets']['wall_s']:
            if process.poll() is not None: raise RuntimeError('owned fixture server exited')
            rclpy.spin_once(node, timeout_sec=.005); now = time.monotonic()
            if 'clock' not in latest: continue
            clock = latest['clock']['clock']; source_now = clock['sec']+clock['nanosec']*1e-9
            if source_now > inputs['budgets']['source_s']: raise TimeoutError('fixture source budget exhausted')
            for key, (future, request) in list(pending.items()):
                if not future.done(): continue
                reply = future.result()
                if reply is None or not reply.success: raise RuntimeError('entity query failed: '+key)
                row = dict(request=request, reply=plain_ros_value(message_to_ordereddict(reply)), received_wall=now)
                latest[key] = row; record('state', key=key, observation=row); del pending[key]
                if not (args.fault == 'source_freeze' and injected is not None):
                    samples.add(key, row, source_s=stamp(row['reply']), received_wall=now)
            if source_now >= next_query and getter.service_is_ready():
                for key, entity in [('carrier', 'scout_a::base_footprint'), ('cargo', profile['tasks']['inspection_consumables']['cargo_entity'])]:
                    if key not in pending:
                        request = GetEntityState.Request(); request.name = entity; request.reference_frame = 'world'
                        pending[key] = (getter.call_async(request), message_to_ordereddict(request))
                next_query = source_now+.02
            raw = latest.get('actuator')
            if raw and raw['position'][raw['name'].index('scout_a_servo_fault')] != 0.:
                if args.fault == 'none': raise RuntimeError('unexpected native actuator lease fault')
                if native_fault is None:
                    native_fault = stamp(raw); record('native_fault', source_s=source_now, message=raw)
                if not rejected:
                    message = Float64MultiArray(); message.data = [float(sequence+10000), 0., 1.]
                    publisher.publish(message); rejected = True
                    record('after_fault_target', source_s=source_now, values=message.data)
                if stamp(raw)-native_fault >= 2.2: break
                continue
            if (args.fault != 'none' and injected is None and raw is not None
                    and raw['position'][raw['name'].index('bin_transfer_stroke')] >= .12):
                injected = source_now; record('fault_injected', mode=args.fault, source_s=source_now)
            if args.fault == 'lease' and injected is not None: continue
            selected = {key: samples.select(key, source_now=source_now, wall_now=now,
                source_age=.25 if key == 'guard' else .15, wall_age=.25 if key == 'guard' else .3)
                for key in ('carrier', 'cargo', 'actuator', 'contact', 'guard')}
            if not all(selected.values()): continue
            actuator = selected['actuator']; guard = selected['guard']
            stopped = (guard['state'] == 'latched' and guard['stopped'] and guard['actual_speed'] < .01
                and abs(guard['actual_wz']) < .02 and guard['output_vx'] == guard['output_wz'] == 0.)
            if command is None and source_now >= 2. and stopped:
                station = profile['tasks']['inspection_consumables']['stations']['LOAD']
                command = dict(handling_id='transfer-fixture:scout_a:1', scene_id=profile['scene_id'], run_id='transfer-fixture',
                    revision=1, task_id='inspection_consumables', issued_at=source_now, operation='LOAD', robot_id='scout_a',
                    node_id=station['node_id'], target_xy=station['target_xy'], source=profile['source'], frame_id='map', time_base='ros_sim')
                record('handling_request', command=command)
            if command is None or not stopped or now-last_publish < .04: continue
            desired = stroke.update(command, actuator, source_now)
            if desired is None: continue
            record('selected', source_s=source_now, wall_now=now, phase=stroke.phase,
                times={key: dict(source_s=stamp(row['reply'] if key in ('carrier', 'cargo') else row),
                    received_wall=row['received_wall'] if key in ('carrier', 'cargo') else row['_received_wall'])
                    for key, row in selected.items()})
            dt = 0. if target_clock is None else max(0., min(.1, source_now-target_clock))
            target += max(-.08*dt, min(.08*dt, desired-target)); target_clock = source_now
            sequence = int(max(sequence, raw['position'][raw['name'].index('scout_a_servo_sequence')]))+1
            message = Float64MultiArray(); message.data = [float(sequence), target, 1.]
            publisher.publish(message); last_publish = now
            record('target', values=message.data, requested_target_m=desired, source_s=source_now, phase=stroke.phase)
            if desired == 0. and abs(target) < 1e-12 and retraction_sequence is None: retraction_sequence = sequence
            if stroke.phase != 'observing' or retraction_sequence is None: continue
            evidence = dict(actuator, _retraction_sequence=retraction_sequence)
            receipt = observer.observe(command, source_now=source_now, carrier=selected['carrier'], cargo=selected['cargo'],
                contact=selected['contact'], actuator=evidence)
            if receipt is not None: record('load_receipt', receipt=receipt); break
        else: raise TimeoutError('fixture wall budget exhausted')
    except Exception as exc:
        error = repr(exc); record('failure', error=error)
    finally:
        if process is not None: cleanup = stop_launch_group(process)
        node.destroy_node(); rclpy.shutdown(); log.close()
    result = dict(error=error, receipt=receipt, native_fault_s=native_fault, injected_s=injected,
        cleanup=cleanup, source_s=source_now, elapsed_wall_s=time.monotonic()-started, command=commandline,
        files_sha256={p.name: digest(p) for p in out.iterdir() if p.is_file()}, boundary=inputs['boundary'])
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    audit = verify(out); print(json.dumps(audit), flush=True)
    return audit


def verify(out, *, audit_name='audit.json'):
    out = Path(out); result = json.loads((out/'result.json').read_text()); inputs = json.loads((out/'inputs.json').read_text())
    integrity = all(digest(out/name) == value for name, value in result['files_sha256'].items())
    samples = EquipmentSamples(); replay = DockedBinObserver(inputs['profile'], lambda key, raw: None)
    command = None; retraction = None; receipt = None; selected = None; frozen = False
    targets = []; native = []; carrier = []; guards = []; faults = []; rejected = []
    causal = True; selection_count = 0
    for row in Records(out/'observations.jsonl'):
        kind = row['kind']
        if kind in ('sample', 'state'):
            key = row['key']; value = row['message'] if kind == 'sample' else row['observation']
            if key == 'clock': continue
            if key == 'actuator': native.append(value)
            if key == 'carrier': carrier.append(value)
            if key == 'guard': guards.append(value)
            if not (inputs['fault'] == 'source_freeze' and frozen and key != 'guard'):
                samples.add(key, value, source_s=stamp(value['reply'] if kind == 'state' else value),
                    received_wall=value['received_wall'] if kind == 'state' else value['_received_wall'])
        elif kind == 'fault_injected': frozen = True
        elif kind == 'handling_request': command = row['command']
        elif kind == 'native_fault': faults.append(row)
        elif kind == 'after_fault_target': rejected.append(row)
        elif kind == 'selected':
            selected = {key: samples.select(key, source_now=row['source_s'], wall_now=row['wall_now'],
                source_age=.25 if key == 'guard' else .15, wall_age=.25 if key == 'guard' else .3) for key in row['times']}
            selection_count += 1
            for key, value in selected.items():
                causal &= value is not None
                if value is None: continue
                source = stamp(value['reply'] if key in ('carrier', 'cargo') else value)
                received = value['received_wall'] if key in ('carrier', 'cargo') else value['_received_wall']
                causal &= row['times'][key] == dict(source_s=source, received_wall=received)
        elif kind == 'target':
            targets.append(row)
            if row['requested_target_m'] == 0. and abs(row['values'][1]) < 1e-12 and retraction is None:
                retraction = int(row['values'][0])
            if row['phase'] == 'observing' and retraction is not None and selected and all(selected.values()):
                answer = replay.observe(command, source_now=row['source_s'], carrier=selected['carrier'], cargo=selected['cargo'],
                    contact=selected['contact'], actuator=dict(selected['actuator'], _retraction_sequence=retraction))
                if answer is not None: receipt = answer
    def position(value): return value['position'][value['name'].index('bin_transfer_stroke')]
    def velocity(value): return value['velocity'][value['name'].index('bin_transfer_stroke')]
    checks = dict(integrity=integrity, clean_shutdown=bool(result['cleanup'] and result['cleanup']['passed']),
        runtime_no_error=result['error'] is None, causal_selection=causal and selection_count > 0,
        replay_sources_match=all(digest(out/'executed_probe.py' if name == Path(__file__).name
            else Path(__file__).with_name(name)) == value for name, value in inputs['source_sha256'].items()),
        force_within_50n=bool(native) and all(abs(value['effort'][0]) <= 50.0000001 for value in native))
    if inputs['fault'] == 'none':
        checks.update(receipt_independently_replayed=receipt is not None and receipt == result['receipt'],
                      no_fault=bool(native) and all(value['position'][-1] == 0. for value in native))
        if receipt is not None:
            path = out/(receipt['evidence_id']+'.json')
            checks['durable_receipt_hash'] = path.exists() and digest(path) == receipt['evidence_id']
    else:
        tail = [value for value in native if faults and stamp(value) >= stamp(faults[0]['message'])]
        final = stamp(tail[-1]) if tail else 0.
        stopped = [value for value in tail if stamp(value) >= final-.5]
        checks.update(fault_was_injected=frozen, native_fault_latched=bool(tail) and all(value['position'][-1] == 1. for value in tail),
            held_away_from_home=bool(tail) and position(tail[0]) > .05,
            hold_drift_under_15mm=bool(tail) and max(abs(position(value)-position(tail[0])) for value in tail) < .015,
            stopped=bool(stopped) and all(abs(velocity(value)) < .02 for value in stopped),
            observed_two_seconds=bool(tail) and final-stamp(tail[0]) >= 2.,
            new_home_target_rejected=bool(tail and rejected) and all(value['position'][-2] == tail[0]['position'][-2] for value in tail)
                and rejected[0]['values'][0] > tail[0]['position'][-2],
            no_false_receipt=receipt is None and result['receipt'] is None)
    origin = carrier[0]['reply']['state']['pose']['position'] if carrier else None
    displacement = [math.dist(list(value['reply']['state']['pose']['position'].values()), list(origin.values())) for value in carrier] if origin else []
    audit = dict(passed=all(checks.values()), checks=checks, load_passed=inputs['fault'] == 'none' and all(checks.values()),
        guided=inputs['guided'], integral_gain_n_per_m_s=inputs['integral_gain_n_per_m_s'], seed=inputs['seed'], fault=inputs['fault'],
        runtime_error=result['error'], receipt=receipt, selected_samples=selection_count, native_samples=len(native),
        max_displacement_m=max(displacement, default=None), max_guard_speed_m_s=max((g['actual_speed'] for g in guards), default=None),
        final_stroke_m=position(native[-1]) if native else None,
        auditor_source_sha256=digest(__file__), execution_probe_sha256=digest(out/'executed_probe.py'),
        boundary=inputs['boundary'])
    (out/audit_name).write_text(json.dumps(audit, indent=2)+'\n')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--axis-build', type=Path)
    parser.add_argument('--guided', action='store_true')
    parser.add_argument('--integral-gain', type=float, default=0.)
    parser.add_argument('--fault', choices=('none', 'lease', 'source_freeze'), default='none')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--domain', type=int, default=119)
    parser.add_argument('--port', type=int, default=11419)
    args = parser.parse_args()
    answer = verify(args.output, audit_name='audit_replay02.json') if args.verify else run(args)
    print(json.dumps(answer)); raise SystemExit(0 if answer['passed'] else 1)


if __name__ == '__main__': main()
