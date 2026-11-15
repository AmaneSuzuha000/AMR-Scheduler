"""Independently replay raw stationary sensor, localization and guard evidence.

Moving-mode data may use the same prerequisite audit, but movement acceptance
requires additional task/route/guard/trajectory audits and is never implied here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

from tools.live_bridge.sensor_health import pose_quality
from .navigation_parameters import expected_parameters, audit_parameter_records


LIMITS = dict(scan=.3, imu=.2, odom=.2, amcl=2.)


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): hasher.update(chunk)
    return hasher.hexdigest()


def stamp(message):
    value = message['header']['stamp']
    return value['sec']+value['nanosec']*1e-9


def xyz(value):
    return [value[d] for d in ('x', 'y', 'z')]


def yaw(q):
    values = [q[d] for d in ('x', 'y', 'z', 'w')]
    assert all(math.isfinite(v) for v in values)
    assert abs(sum(v*v for v in values)-1) < 1e-3
    return math.atan2(2*(q['w']*q['z']+q['x']*q['y']), 1-2*(q['y']**2+q['z']**2))


def angle_difference(a, b):
    return abs(math.atan2(math.sin(a-b), math.cos(a-b)))


def guard_stopped(message, received, now, clock):
    return (0 <= now-received < .2 and -.1 <= clock-stamp(message) < .25
        and message['state'] == 'latched' and message['stopped']
        and message['output_vx'] == message['output_wz'] == 0
        and message['actual_speed'] < .01 and abs(message['actual_wz']) < .02)


def audit_initial_priors(rows, layout):
    """Bind perturbed map priors to the frozen spawn, including contact-dock yaw."""
    nodes = {node['id']: node for node in layout['nodes']}
    robots = {robot['id']: robot for robot in layout['robots']}
    seen = set()
    for row in rows:
        if row['kind'] != 'initial_pose':
            continue
        robot = row['robot']; assert robot in robots and robot not in seen
        config = robots[robot]; point = nodes[config['start']]
        message = row['message']; pose = message['pose']['pose']
        assert message['header']['frame_id'] == 'map'
        assert pose['position'] == dict(x=point['x']+.15, y=point['y']-.08, z=0.)
        assert angle_difference(yaw(pose['orientation']), config.get('start_yaw_rad', 0.)+.05) < 1e-12
        covariance = [0.]*36
        for index in (0, 7, 35): covariance[index] = .1**2
        assert message['pose']['covariance'] == covariance
        assert row['source'] == 'known spawn prior plus fixed perturbation; independent of state queries'
        seen.add(robot)
    assert seen == set(robots)
    return len(seen)


def replay(rows, interval, *, parameter_nodes=None):
    """Recompute validity from raw events, independent of recorded ready flags."""
    robots = {'scout_a', 'scout_b'}
    state = {robot: dict(samples={}, lifecycle={}, guard=None, check=None, ready=[],
                        queries={}, consumed_queries=set(), parameter_nodes=set()) for robot in robots}
    counts = {robot: dict(scan=0, imu=0, odom=0, amcl=0, checks=0, guard=0,
                          rejected_stale=0, expired_tf=0, service_failed=0) for robot in robots}
    prior_wall = -math.inf
    def health(current, now, clock):
        for source, limit in LIMITS.items():
            sample = current['samples'].get(source)
            if sample is None or not sample['valid']:
                return False
            if not (0 <= now-sample['received'] <= limit and -.1 <= clock-sample['stamp'] <= limit):
                return False
        return True
    def source(current, name, time_s, received, valid):
        assert math.isfinite(time_s) and math.isfinite(received)
        old = current['samples'].get(name)
        if old is None or time_s > old['stamp']:
            current['samples'][name] = dict(stamp=time_s, received=received, valid=valid)
    for row in rows:
        assert math.isfinite(row['wall_s']) and row['wall_s'] >= prior_wall
        prior_wall = row['wall_s']
        robot = row.get('robot')
        if robot is None:
            continue
        assert robot in robots
        current = state[robot]; kind = row['kind']
        if kind == 'runtime_parameters':
            current['parameter_nodes'].add(row['node'])
        if 'received_wall_s' in row:
            assert 0 <= row['received_wall_s'] <= row['wall_s']
        if kind == 'scan':
            ranges = row['ranges']; assert len(ranges) == row['beams'] > 0
            assert row['frame_id'] == 'laser_link'
            assert 0 < row['range_min'] < row['range_max'] == 8.
            assert all(v is None or math.isfinite(v) for v in ranges)
            useful = sum(v is not None and row['range_min'] <= v < row['range_max'] for v in ranges)
            assert useful == row['finite_returns']
            source(current, 'scan', row['source_s'], row['received_wall_s'], useful >= max(30, .1*len(ranges)))
            counts[robot]['scan'] += 1
        elif kind == 'imu':
            source(current, 'imu', row['source_s'], row['received_wall_s'], math.isfinite(row['yaw_rate']))
            counts[robot]['imu'] += 1
        elif kind == 'filtered_odom':
            source(current, 'odom', row['source_s'], row['received_wall_s'],
                   all(math.isfinite(row[k]) for k in ('vx', 'vy', 'wz')))
            counts[robot]['odom'] += 1
        elif kind == 'amcl':
            message = row['message']; assert message['header']['frame_id'] == 'map'
            pose = message['pose']['pose']; p, q = pose['position'], pose['orientation']
            yaw(q)
            valid = all(math.isfinite(v) for v in (*xyz(p), *xyz(q), q['w']))
            valid = valid and pose_quality(message['pose']['covariance'], (p['x'], p['y']))
            source(current, 'amcl', stamp(message), row['received_wall_s'], valid)
            counts[robot]['amcl'] += 1
        elif kind == 'guard':
            message = row['message']
            assert message['guard_id']
            assert all(math.isfinite(message[k]) for k in ('actual_speed', 'actual_wz', 'output_vx', 'output_wz'))
            current['guard'] = (message, row['received_wall_s'])
            counts[robot]['guard'] += 1
        elif kind == 'lifecycle':
            assert row['node'] in ('controller_server', 'map_server', 'amcl')
            current['lifecycle'][row['node']] = (row['state'], row['received_wall_s'])
        elif kind == 'initial_pose':
            assert 'prior' not in current
            current['prior'] = row
        elif kind == 'state_query_issued':
            query = {k: row[k] for k in ('query_id', 'service', 'request', 'issued_wall_s')}
            assert query['query_id'] == len(current['queries'])
            assert query['service'] == '/wheelset_review/get_entity_state'
            assert query['request'] == dict(name=robot, reference_frame='world')
            assert 0 <= query['issued_wall_s'] <= row['wall_s']
            current['queries'][query['query_id']] = query
        elif kind == 'independent_pose_check':
            raw, tf = row['state_reply'], row['transform']
            query = row['state_query']; token = query['query_id']
            assert token in current['queries'] and query == current['queries'][token]
            assert token not in current['consumed_queries']
            current['consumed_queries'].add(token)
            assert query['issued_wall_s'] <= row['state_received_wall_s'] <= row['evaluated_wall_s']
            # gazebo_ros_state leaves EntityState.name/reference_frame empty.
            # Bind identity to the recorded request/future, never invent names
            # in the original response or accept an unbound anonymous reply.
            assert raw['success'] and raw['state']['name'] in ('', robot)
            assert raw['header']['frame_id'] == 'world'
            assert tf['header']['frame_id'] == 'map' and tf['child_frame_id'] == 'base_footprint'
            assert abs(stamp(raw)-stamp(tf)) <= 1e-8
            a = raw['state']['pose']; b = tf['transform']
            truth = [*xyz(a['position'])[:2], yaw(a['orientation'])]
            localized = [*xyz(b['translation'])[:2], yaw(b['rotation'])]
            assert row['truth'][:2] == truth[:2] and row['localized'][:2] == localized[:2]
            # x*x versus x**2 in independent quaternion reductions can differ
            # by one ulp (observed 2.22e-16rad). Keep raw positions exact and
            # bound only this floating-point yaw reconstruction roundoff.
            assert angle_difference(row['truth'][2], truth[2]) <= 1e-12
            assert angle_difference(row['localized'][2], localized[2]) <= 1e-12
            assert row['source_s'] == stamp(raw)
            assert -.1 <= row['clock_s']-stamp(raw) <= .25
            assert 0 <= row['evaluated_wall_s']-row['state_received_wall_s'] <= 1.
            error = math.dist(truth[:2], localized[:2]); angular = angle_difference(truth[2], localized[2])
            assert abs(row['error_m']-error) < 1e-12 and abs(row['error_rad']-angular) < 1e-12
            v = raw['state']['twist']['linear']
            assert row['speed_mps'] == math.hypot(v['x'], v['y'])
            current['check'] = dict(wall=row['evaluated_wall_s'], error=error, angular=angular,
                                    health=row['health'])
            counts[robot]['checks'] += 1
        elif kind == 'pose_check_expired_tf':
            # A physics reply that never paired with a same-time EKF TF inside the
            # bounded retention window. Bind it to its issued query so the audit
            # account stays closed instead of tolerating an unrecorded drop.
            query = row['state_query']; token = query['query_id']
            assert token in current['queries'] and query == current['queries'][token]
            assert token not in current['consumed_queries']
            current['consumed_queries'].add(token)
            assert isinstance(row['reason'], str) and row['reason']
            counts[robot]['expired_tf'] += 1
        elif kind == 'pose_check_rejected_stale':
            # A reply that DID pair with a TF but whose stamp was outside the
            # freshness window. Before candidate 31 this case advanced the unpaired
            # queue with no record at all, so the readiness diagnostic showed only a
            # growing last_pose_check_age and looked like a TF outage.
            query = row['state_query']; token = query['query_id']
            assert token in current['queries'] and query == current['queries'][token]
            assert token not in current['consumed_queries']
            current['consumed_queries'].add(token)
            age = row['clock_s']-row['source_s']
            assert not (-.1 <= age <= .25), (robot, 'stale rejection inside freshness window', age)
            assert abs(row['age_s']-age) < 1e-9
            assert isinstance(row['reason'], str) and row['reason']
            counts[robot]['rejected_stale'] += 1
        elif kind == 'pose_check_service_failed':
            # GetEntityState answered with success=false, or no reply object at all
            # (typically before the model exists). Candidate31 records this outcome
            # so the query account closes instead of losing the id outright.
            query = row['state_query']
            token = query['query_id']
            assert token in current['queries'] and query == current['queries'][token]
            assert token not in current['consumed_queries']
            current['consumed_queries'].add(token)
            assert row['reply_success'] in (None, False), (robot, row['reply_success'])
            assert isinstance(row['reason'], str) and row['reason']
            counts[robot]['service_failed'] += 1
        elif kind == 'readiness_state':
            now, clock = row['evaluated_wall_s'], row['source_s']
            assert 0 <= now <= row['wall_s'] and math.isfinite(clock)
            g = current['guard']; check = current['check']
            required = row.get('parameter_nodes_required', [])
            if required:
                expected = ({'amcl', 'ekf_filter_node', 'controller_server'}
                            if parameter_nodes is None else set(parameter_nodes[robot]))
                assert len(required) == len(set(required)) and set(required) == expected
            ready = bool(health(current, now, clock) and row['action_server_ready']
                and set(required) <= current['parameter_nodes']
                and all(node in current['lifecycle'] and current['lifecycle'][node][0] == 3
                    and 0 <= now-current['lifecycle'][node][1] < 2.
                    for node in ('controller_server', 'map_server', 'amcl'))
                and g is not None and guard_stopped(g[0], g[1], now, clock)
                and check is not None and 0 <= now-check['wall'] < 1.
                and check['health'] is None and check['error'] <= .15 and check['angular'] <= .20)
            assert row['ready'] == ready, (robot, 'readiness mismatches raw evidence', now)
            current['ready'].append((now, ready))
    start, end = interval['start_wall_s'], interval['end_wall_s']
    assert end-start >= 2.
    for robot, current in state.items():
        assert 'prior' in current
        # Every issued pose-check query is answered and consumed in FIFO order, so
        # the consumed query ids must be a contiguous prefix of the issued ids and
        # the only ids left over must be the still-retained tail at shutdown. A hole
        # in the middle is exactly the silent-drop defect candidate 31 removes: the
        # reply was popped from the pairing queue without any recorded outcome.
        issued = sorted(current['queries'])
        assert issued == list(range(len(issued))), (robot, 'issued query ids are not the exact sequence')
        consumed = sorted(current['consumed_queries'])
        assert consumed == list(range(len(consumed))), (
            robot, 'pose-check query consumed out of order',
            [q for q in issued if q not in current['consumed_queries']][:8])
        assert len(consumed) == (counts[robot]['checks']+counts[robot]['rejected_stale']
                                 +counts[robot]['expired_tf']
                                 +counts[robot]['service_failed']), (robot, 'pose-check accounting mismatch')
        candidate = [(t, ready) for t, ready in current['ready'] if start-1e-9 <= t <= end+1e-9]
        assert len(candidate) >= 10 and all(ready for _, ready in candidate)
        assert candidate[0][0]-start < .05 and end-candidate[-1][0] < .05
        assert all(b[0]-a[0] <= .2 for a, b in zip(candidate, candidate[1:]))
        assert all(counts[robot][name] >= 3 for name in ('scan', 'imu', 'odom', 'amcl', 'checks', 'guard'))
    return counts


def verify(directory, *, write=True, startup_only=False):
    directory = Path(directory)
    result = json.loads((directory/'result.json').read_text())
    if not startup_only:
        assert result['passed'] and result['error'] is None
    assert result['executed_sources_preserved']
    assert result['readiness_interval'] and result['owned_launch_ended'] and result['owned_process_group_stopped']
    for name, digest in result['files_sha256'].items():
        assert file_sha256(directory/name) == digest, name
    for name, digest in result['source_sha256'].items():
        assert hashlib.sha256((directory/'executed_sources'/name).read_bytes()).hexdigest() == digest
    manifest = json.loads((directory/'scene_snapshot/manifest.json').read_text())
    assert hashlib.sha256((directory/'scene_snapshot/manifest.json').read_bytes()).hexdigest() == result['bundle_manifest_sha256']
    assert manifest['build_fingerprint'] == result['build_fingerprint']
    for name, digest in manifest['files_sha256'].items():
        assert hashlib.sha256((directory/'scene_snapshot'/name).read_bytes()).hexdigest() == digest
    native = json.loads((directory/'native_guard/build_manifest.json').read_text())
    assert native == result['guard_build'] and native['passed']
    for name, digest in native['library_sha256'].items():
        assert hashlib.sha256((directory/'native_guard'/name).read_bytes()).hexdigest() == digest
    for name, digest in native['source_sha256'].items():
        assert hashlib.sha256((directory/'native_guard/sources'/name).read_bytes()).hexdigest() == digest
    for robot in ('scout_a', 'scout_b'):
        root = ET.parse(directory/'models'/(robot+'.urdf')).getroot()
        drive = root.find(".//plugin[@filename='libgazebo_ros_diff_drive.so']")
        assert drive is not None and drive.findtext('odometry_source') == '0'
        assert drive.findtext('publish_odom_tf') == 'false'
        assert 'cmd_vel:=guarded_cmd_vel' in [e.text for e in drive.findall('ros/remapping')]
        assert len(root.findall(".//plugin[@filename='libamr_motion_guard.so']")) == 1
        assert {s.get('type') for s in root.findall('.//sensor')} == {'ray', 'imu', 'contact'}
        if manifest.get('laser_update_rate_hz') is not None:
            assert float(root.findtext(".//sensor[@type='ray']/update_rate")) == manifest['laser_update_rate_hz']
        assert root.find(".//plugin[@filename='libgazebo_ros_planar_move.so']") is None
        assert all(plugin.findtext('ros/namespace') == '/'+robot for plugin in root.findall('.//plugin'))
        loc = yaml.safe_load((directory/'scene_snapshot'/(robot+'_localization.yaml')).read_text())[robot]
        ekf = loc['ekf_filter_node']['ros__parameters']
        encoder_topic = ('wheel/position_odometry' if manifest.get('encoder_source') == 'joint_positions'
                         else 'wheel/odometry')
        assert ekf['odom0'] == '/'+robot+'/'+encoder_topic
        assert ekf['imu0'] == '/'+robot+'/imu/data'
        assert [i for i, value in enumerate(ekf['odom0_config']) if value] == [6]
        assert [i for i, value in enumerate(ekf['imu0_config']) if value] == [11]
        assert loc['amcl']['ros__parameters']['scan_topic'] == '/'+robot+'/scan'
        if 'model_parameters' in manifest:
            sdf = ET.parse(directory/'models'/(robot+'.sdf')).getroot()
            if manifest.get('laser_update_rate_hz') is not None:
                assert float(sdf.findtext(".//sensor[@type='ray']/update_rate")) == manifest['laser_update_rate_hz']
            from .navigation_model import WHEELS
            for name in WHEELS:
                ode = sdf.find(f"model/link[@name='{name}_wheel_link']/collision/surface/friction/ode")
                assert ode is not None and ode.findtext('fdir1') == '0 0 1'
                assert float(ode.findtext('slip1')) == manifest['model_parameters']['lateral_slip_m_per_ns']
    parameter_nodes = None
    expected = None
    if result.get('runtime_parameters_required'):
        expected = {robot: expected_parameters(directory/'scene_snapshot', robot, config)
                    for robot, config in manifest['robot_configs'].items()}
        with (directory/'observations.jsonl').open() as f:
            parameter_nodes = audit_parameter_records((json.loads(line) for line in f), expected,
                                                      result['readiness_interval']['start_wall_s'])
    with (directory/'observations.jsonl').open() as f:
        counts = replay((json.loads(line) for line in f), result['readiness_interval'],
                        parameter_nodes=expected)
    layout = json.loads((directory/'scene_snapshot/network.json').read_text())
    with (directory/'observations.jsonl').open() as f:
        priors = audit_initial_priors((json.loads(line) for line in f), layout)
    audit = dict(passed=True, scope='stationary_sensor_readiness', runtime_mode=result['mode'],
        counts=counts, interval=result['readiness_interval'],
        runtime_passed=result['passed'], runtime_error=result['error'], startup_only=startup_only,
        configured_initial_priors_verified=priors,
        runtime_parameter_nodes_verified=parameter_nodes,
        result_sha256=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest(),
        claim_boundary='Raw sensor freshness, known-map initialization, localization-vs-state errors, '
            'active lifecycle nodes, native stopped guard and 2s stationary startup replay only. '
            'Does not prove moving localization, tasks, resource ownership, fault recovery or industrial acceptance.')
    if write:
        name = 'navigation_startup_only_audit.json' if startup_only else 'navigation_readiness_audit.json'
        (directory/name).write_text(json.dumps(audit, indent=2)+'\n')
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--startup-only', action='store_true',
                   help='Audit startup prerequisites of a later failed run; preserve runtime failure explicitly')
    args = p.parse_args()
    print(json.dumps(verify(args.directory, startup_only=args.startup_only)))


if __name__ == '__main__':
    main()
