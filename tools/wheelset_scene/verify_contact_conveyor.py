"""Recompute isolated chain-conveyor support, driven motion and stop evidence."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def stamp(message):
    value = message['header']['stamp']
    return value['sec']+value['nanosec']*1e-9


def norm(vector):
    return math.sqrt(sum(vector[key]**2 for key in ('x','y','z')))


def world_vertical_force(force, orientation):
    x,y,z,w = (orientation[key] for key in ('x','y','z','w'))
    assert abs(x*x+y*y+z*z+w*w-1.) < .001
    return ((2*x*z-2*y*w)*force['x']+(2*y*z+2*x*w)*force['y']
            +(1-2*x*x-2*y*y)*force['z'])


def rail_force(message, side, orientation):
    """Average physics steps, rotating the body's wrench into world axes.

    gazebo_ros_bumper sets frame_id but copies the Gazebo contact wrench.
    Its force retains body axes; contact normals and positions use world axes.
    Independent measured body orientation supplies the required rotation.
    """
    grouped = defaultdict(float)
    suffix = '-1' if side == 'left' else '1'
    for contact in message['states']:
        pair = {contact['collision1_name'], contact['collision2_name']}
        assert 'contact_wheelset::body::tread_'+side in pair, 'wrong observed tread'
        if 'powered_conveyor::base::support_rail_'+suffix not in pair:
            continue
        match = re.search(r'time:\s*\[([0-9.]+)\]', contact['info'])
        assert match, 'missing physics contact time'
        assert contact['contact_normals'] and all(n['z'] > .98 for n in contact['contact_normals'])
        grouped[float(match.group(1))] += world_vertical_force(contact['total_wrench']['force'],orientation)
    return sum(grouped.values())/len(grouped) if grouped else 0.


def audit(directory):
    directory = Path(directory).resolve()
    result = json.loads((directory/'result.json').read_text())
    assert result['passed'] and result['runtime_error'] is None, 'runner did not finish'
    assert result['owned_process_group_stopped'] and result['cleanup']['passed'], 'owned server still live'
    assert result['mechanism'] == 'chain', 'only chain prototype is supported by this audit'
    for name, expected in result['files_sha256'].items():
        path = (directory/name).resolve()
        assert path.is_relative_to(directory) and path.is_file() and digest(path) == expected, 'file hash: '+name
    for name, expected in result['source_sha256'].items():
        assert digest(directory/'executed_sources'/name) == expected, 'executed source changed'
    native = json.loads((directory/'native_build_manifest.json').read_text())
    assert native['passed'] and digest(directory/'native/libwheelset_axis_servo.so') == native['library_sha256']
    for name, expected in native['source_sha256'].items():
        assert digest(directory/'native/source'/name) == expected, 'native source changed'
    source = (directory/'native/source/src/axis_servo.cpp').read_text()
    assert 'a.joint->SetForce(0,a.force)' in source
    assert not re.search(r'->Set(?:Velocity|Position|WorldPose|LinearVel|AngularVel)\s*\(', source), 'kinematic actuator'

    manifest = json.loads((directory/'world/manifest.json').read_text())
    assert manifest == result['world_manifest'] and manifest['library_sha256'] == native['library_sha256']
    for name, expected in manifest['files_sha256'].items():
        assert digest(directory/'world'/name) == expected, 'world hash mismatch'
    world = ET.parse(directory/'world/conveyor.world').getroot().find('world')
    assert world.findtext('gravity') == '0 0 -9.81'
    assert world.findtext('physics/max_step_size') == '0.001'
    assert world.findtext('physics/ode/solver/iters') == '400'
    payload = world.find("model[@name='contact_wheelset']")
    assert payload.findtext('static','false') == 'false'
    assert payload.findtext('link/gravity') == 'true' and not payload.findall('plugin')
    assert not payload.findall('joint') and len(payload.findall('link/collision')) == 3
    assert len(payload.findall("link/sensor[@type='contact']")) == 2
    conveyor = world.find("model[@name='powered_conveyor']")
    assert {j.get('name') for j in world.findall('model/joint')} == {'chain_straight_stroke','conveyor_anchor'}
    assert conveyor.findtext("joint[@name='chain_straight_stroke']/child") == 'chain_dogs'
    assert conveyor.findtext('plugin/axis/joint') == 'chain_straight_stroke'
    assert conveyor.findtext('plugin/axis/mode') == 'position'
    assert conveyor.findtext('plugin/axis/limit') == '5000'

    counts = Counter(); requests = {}; replies = set(); commands = {}
    contact_latest = {}; poses = []; joints = []; phases = []; frames = []
    support_ok = support_total = dog_contact_frames = 0
    max_abs_y = max_force = 0.; min_z = float('inf'); max_z = float('-inf')
    final_stop = None
    for line in (directory/'records.jsonl').open():
        row = json.loads(line); kind = row['kind']; counts[kind] += 1
        if kind == 'runtime_error':
            raise AssertionError('raw runtime error')
        if kind == 'phase':
            phases.append(row['value'])
            if row['value'] == 'stopped': final_stop = row['clock_s']
        if kind == 'wheelset_state_request':
            query = row['query']; number = query['id']
            assert number not in requests and number == len(requests), 'query reused or skipped'
            assert query['service'] == '/contact_conveyor/get_entity_state'
            assert query['request'] == dict(name='contact_wheelset',reference_frame='world')
            requests[number] = query
        if kind == 'wheelset_state_reply':
            query = row['query']; number = query['id']
            assert number in requests and number not in replies and query == requests[number], 'unpaired state reply'
            replies.add(number); reply = row['response']
            assert reply['success'] and reply['header']['frame_id'] == 'world'
            t = stamp(reply); state = reply['state']; p = state['pose']['position']
            assert all(math.isfinite(v) for v in p.values())
            assert abs(p['y']) < .05 and .765 < p['z'] < .825, 'payload left supported envelope'
            assert -.02 < p['x'] < 2.10, 'payload left conveyor'
            if poses:
                dt = t-poses[-1][0]
                assert 0. < dt < .25, 'stale independent position'
                assert abs(p['x']-poses[-1][1]['pose']['position']['x']) < 1.2*dt+.01, 'pose jumped'
            poses.append((t,state)); max_abs_y = max(max_abs_y,abs(p['y']))
            min_z = min(min_z,p['z']); max_z = max(max_z,p['z'])
            if t >= 2.:
                support_total += 1; forces=[]
                for side in ('left','right'):
                    observed = contact_latest.get(side)
                    forces.append(rail_force(observed,side,state['pose']['orientation'])
                                  if observed and -.03 < t-stamp(observed) < .15 else 0.)
                weight = manifest['mass_kg']*9.81
                if min(forces) > .1*weight and .5*weight < sum(forces) < 1.5*weight:
                    support_ok += 1
        if kind in ('tread_left_contacts','tread_right_contacts'):
            side = 'left' if kind == 'tread_left_contacts' else 'right'
            msg = row['message']; contact_latest[side] = msg
            for contact in msg['states']:
                names = (contact['collision1_name'],contact['collision2_name'])
                if any(name.startswith('powered_conveyor::chain_dogs::chain_dog_') for name in names):
                    if abs(contact['total_wrench']['force']['x']) > 1.:
                        dog_contact_frames += 1
                        break
        if kind == 'roller_command':
            seq = row['sequence']; values = row['values']
            assert seq not in commands and values[0] == seq and len(values) == 3
            assert -.02 <= values[1] <= 2.015001 and values[2] == 1.
            commands[seq] = values[1]
        if kind == 'shutdown_zero_command':
            assert row['sequence'] not in commands
            commands[row['sequence']] = None
        if kind == 'roller_joint_state':
            msg = row['message']; names = msg['name']
            assert set(names) == {'chain_straight_stroke','powered_conveyor_servo_sequence','powered_conveyor_servo_fault'}
            i = names.index('chain_straight_stroke'); fault = names.index('powered_conveyor_servo_fault')
            assert msg['position'][fault] == 0., 'servo lease fault'
            seq = msg['position'][names.index('powered_conveyor_servo_sequence')]
            assert seq == -1 or seq in commands, 'unrecorded actuator command'
            assert abs(msg['effort'][i]) <= 5000.000001
            max_force = max(max_force,abs(msg['effort'][i])); joints.append(msg['position'][i])
        if kind == 'camera_frame': frames.append(row)

    assert phases == ['conveying','stopped'], 'missing arrival and stop'
    assert len(poses) >= 50 and len(joints) >= 100 and len(frames) >= 10, 'insufficient observation'
    assert support_total > 40 and support_ok/support_total > .95, 'missing two-sided rail load support'
    assert dog_contact_frames >= 10, 'no physical chain-dog pushing'
    assert max(joints)-min(joints) > 1.97, 'drive did not travel'
    displacement = poses[-1][1]['pose']['position']['x']-poses[0][1]['pose']['position']['x']
    assert 1.975 < displacement < 2.025, 'wheelset did not travel two metres'
    # Recompute continuous settling from sensor timestamps. Phase messages use
    # the coarser /clock callback and are not a duration measurement.
    stopped = []
    for t,s in reversed(poses):
        if (abs(s['pose']['position']['x']-2.) >= .025 or norm(s['twist']['linear']) >= .01
                or norm(s['twist']['angular']) >= .03):
            break
        stopped.append((t,s))
    stopped.reverse()
    assert stopped and stopped[-1][0]-stopped[0][0] >= 2., 'missing measured two-second stopped dwell'
    assert any(2. < frame['source_s'] < final_stop for frame in frames), 'no moving camera observations'
    return dict(passed=True, scope='isolated_assumed_chain_conveyor_contact_and_motion',
        independently_measured_displacement_m=displacement, state_replies=len(poses),
        rail_support_fraction=support_ok/support_total, chain_dog_contact_frames=dog_contact_frames,
        maximum_drive_force_n=max_force, maximum_lateral_deviation_m=max_abs_y,
        contact_wrench_axes='body wrench rotated using independently queried wheelset orientation; world rail normals',
        minimum_axle_height_m=min_z, maximum_axle_height_m=max_z,
        stopped_dwell_s=stopped[-1][0]-stopped[0][0], camera_frames=len(frames),
        verified_files=len(result['files_sha256']), counts=dict(counts),
        boundary='Assumed straight chain conveyor only; video drive identity, full line and SCOUT loading remain unverified.')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('directory', type=Path)
    args = parser.parse_args(); report = audit(args.directory)
    target = args.directory/'contact_motion_audit.json'
    with target.open('x') as stream: stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__': main()
