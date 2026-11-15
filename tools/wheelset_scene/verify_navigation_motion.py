"""Audit normal sensor/Nav2 fleet motion from saved raw and owner evidence.

This supplements the stationary prerequisite audit. Fault recovery, continuous
contact safety and downstream equipment inventory require separate audits.
"""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path

from tools.live_bridge.sensor_health import SensorHealth, pose_quality
from .light_material_network import point_segment
from .verify_navigation_readiness import stamp, verify as verify_readiness
from .navigation_records import Records


IDENTITY = ('scene_id', 'run_id', 'robot_id', 'frame_id', 'time_base', 'source')


def read_rows(path):
    return Records(path)


def stopped(guard):
    return (guard['state'] == 'latched' and guard['stopped']
        and guard['actual_speed'] < .01 and abs(guard['actual_wz']) < .02
        and guard['output_vx'] == guard['output_wz'] == 0)


def audit_robot(rows, robot, layout, *, recovery=False):
    """Reconstruct dispatch -> native result -> physical stop -> arrival."""
    nodes = {n['id']: (n['x'], n['y']) for n in layout['nodes']}
    edges = {frozenset((e['src'], e['dst'])) for e in layout['lanes']}
    node = next(r['start'] for r in layout['robots'] if r['id'] == robot)
    routes = {}; arrivals = {}; arrival_guards = {}; feedback = {}; active = None; guard = None
    localization = None; guard_id = None; seq = -1; source = -math.inf
    receipt = None; prior_wall = -math.inf; identity = None; shutdown = None
    cancelled = {}; stops = {}; recoveries = {}; rejected = {}; stop_guards = {}; stopped_guards = {}
    dispatched = {r['value']['route_id']: r for r in rows
                  if r['kind'] == 'command' and r['value']['kind'] == 'MOVE'}
    for row in rows:
        now = row['wall']; assert math.isfinite(now) and now >= prior_wall
        prior_wall = now; kind = row['kind']
        assert kind not in ('truth', 'truth_rejected'), 'truth is not execution authority'
        if kind == 'command':
            c = row['value']; token = c['route_id']
            if recovery and c['kind'] == 'CANCEL':
                assert active and token == active['command']['route_id'] and token not in cancelled
                assert all(c[k] == active['command'][k] for k in (*IDENTITY, 'task_id', 'route_id'))
                assert c['reason'] in ('execution guard: velocity command timeout',
                    'execution guard: mission lease timeout', 'localization health: scan stale')
                cancelled[token] = row
                continue
            assert c['kind'] == 'MOVE' and token not in routes
            previous = c.get('resumes_route_id')
            if previous:
                assert recovery and active and active['command']['route_id'] == previous
                assert previous in recoveries and recoveries[previous]['result']['command'] == c
            else:
                assert active is None or active['command']['route_id'] in arrivals
            assert c['source'] == 'localized' and c['frame_id'] == 'map'
            assert c['time_base'] == 'ros_sim' and c['scene_id'] == layout['id'] and c['robot_id'] == robot
            if identity is None: identity = {key: c[key] for key in IDENTITY}
            assert all(c[key] == identity[key] for key in IDENTITY)
            assert c['nodes'][0] == node and frozenset(c['nodes']) in edges
            assert c['target_xy'] == list(nodes[c['nodes'][1]]) and 0 < c['speed'] <= .2
            active = dict(command=c, issued=now, accepted=False, authorize_requested=False,
                          authorized=False, path=False, native_result=None, lease_seq=0)
            routes[token] = active
        elif kind == 'localization':
            assert all(math.isfinite(row[key]) for key in ('x', 'y', 'yaw', 'stamp'))
            assert -.1 <= row['sim']-row['stamp'] <= .2
            localization = row
        elif kind == 'path':
            assert active is not None and row['route_id'] == active['command']['route_id']
            assert row['frame'] == 'map' and row['transform_time'] == 0.
            c = active['command']; xy = row['xy']; assert len(xy) >= 2
            start = c['resume_from'] if c.get('resumes_route_id') else nodes[c['nodes'][0]]
            assert math.dist(xy[0], start) <= .15
            assert math.dist(xy[-1], c['target_xy']) < 1e-8
            assert all(point_segment(p, *(nodes[n] for n in c['nodes'])) <= .15 for p in xy)
            active['path'] = True
        elif kind == 'native_accept':
            assert active and row['route_id'] == active['command']['route_id'] and row['accepted']
            assert active['path'] and not active['accepted']; active['accepted'] = True
        elif kind == 'guard_authorize_requested':
            assert active and row['route_id'] == active['command']['route_id']
            assert row['revision'] == active['command']['revision']
            assert active['path'] and active['accepted'] and not active['authorize_requested']
            active['authorize_requested'] = True
        elif kind == 'guard_authorization_observed':
            c = active['command'] if active else None
            assert active and row['route_id'] == c['route_id'] and row['revision'] == c['revision']
            assert active['path'] and active['accepted'] and active['authorize_requested']
            assert not active['authorized'] and row['source'] == 'fresh_exact_guard_status'
            assert guard is not None and guard['state'] == 'armed' and guard['guard_id']
            assert (guard['run_id'], guard['route_id'], guard['revision']) == (
                c['run_id'], c['route_id'], c['revision'])
            assert 0 <= now-guard['wall'] < .25
            assert -.1 <= row['sim']-guard['sim'] <= .25
            active['authorized'] = True
        elif kind == 'guard_authorize':
            assert active and row['route_id'] == active['command']['route_id'] and row['accepted']
            assert active['path'] and active['accepted'] and active['authorize_requested']
            active['authorized'] = True
        elif kind == 'lease':
            assert active and active['authorized']
            c = active['command']; assert (row['route_id'], row['revision']) == (c['route_id'], c['revision'])
            assert row['seq'] > active['lease_seq'] and row['subscribers'] >= 1
            assert -.1 <= row['sim']-row['stamp'] <= .25
            active['lease_seq'] = row['seq']
        elif kind == 'guard':
            if guard_id is None: guard_id = row['guard_id']
            assert row['guard_id'] == guard_id and guard_id
            assert all(math.isfinite(row[k]) for k in ('actual_speed', 'actual_wz', 'output_vx', 'output_wz'))
            if row['state'] == 'latched':
                assert row['output_vx'] == row['output_wz'] == 0
            if row['output_vx'] != 0 or row['output_wz'] != 0:
                assert active and active['authorized'] and active['lease_seq'] > 0
                c = active['command']
                assert (row['run_id'], row['route_id'], row['revision']) == (c['run_id'], c['route_id'], c['revision'])
                assert row['state'] == 'armed' and row['command_age'] <= .25 and row['lease_age'] <= .5
            guard = row
        elif recovery and kind == 'guard_stop':
            assert active and row['accepted']
            stop_guards[active['command']['route_id']] = row
        elif kind == 'native_result':
            assert active and row['route_id'] == active['command']['route_id']
            token = row['route_id']
            assert row['status'] == (5 if token in cancelled else 4)
            assert active['accepted'] and active['native_result'] is None
            active['native_result'] = now
        elif recovery and kind == 'recovery':
            token = row['previous_route_id']; answer = row['result']
            if not answer['accepted']:
                assert active and token == active['command']['route_id']+':obsolete'
                assert answer['reason'] == 'recovery action mismatch'
                rejected[active['command']['route_id']] = row
                continue
            assert active and token == active['command']['route_id'] and token not in recoveries
            assert token in cancelled and token in stops and token in stop_guards and token in rejected
            assert cancelled[token]['wall'] <= active['native_result'] <= stops[token]['wall'] <= now
            assert rejected[token]['wall'] < now and stop_guards[token]['wall'] <= stops[token]['wall']
            assert guard is not None and 0 <= now-guard['wall'] < .25 and stopped(guard)
            c = active['command']; new = answer['command']
            assert all(new[k] == c[k] for k in (*IDENTITY, 'task_id', 'nodes', 'target_xy', 'speed'))
            assert new['route_id'] not in dispatched or dispatched[new['route_id']]['value'] == new
            assert new['route_id'] != token and new['revision'] > c['revision'] and new['resumes_route_id'] == token
            assert new['resume_from'] == stops[token]['message']['pose'][:2]
            assert new['resume_path_xy'][0] == new['resume_from'] and new['resume_path_xy'][-1] == c['target_xy']
            assert all(point_segment(p, *(nodes[n] for n in c['nodes'])) <= .15 for p in new['resume_path_xy'])
            recoveries[token] = row
        elif kind == 'feedback':
            message, answer = row['message'], row['result']
            if not answer['accepted']:
                assert answer['reason'] == 'duplicate source timestamp'; continue
            assert localization is not None
            assert message['pose'] == [localization[k] for k in ('x', 'y', 'yaw')]
            assert message['timestamp'] == localization['stamp']
            assert message['source'] == 'localized' and message['frame_id'] == 'map' and message['robot_id'] == robot
            if identity is None: identity = {key: message[key] for key in IDENTITY}
            assert all(message[k] == identity[k] for k in IDENTITY)
            assert message['seq'] > seq and message['timestamp'] >= source
            assert receipt is None or now-receipt <= 2.
            if message['timestamp'] > source: receipt = now
            seq, source = message['seq'], message['timestamp']
            feedback[seq] = row
            token = message['route_id']
            if token is not None:
                if token in routes:
                    c = routes[token]['command']
                else:
                    # The single owner binds its MOVE before the adapter's next
                    # tick drains and logs it.  A stationary POSE can therefore
                    # carry that identity briefly before dispatch.  It cannot
                    # authorize movement, complete an action or invent a route.
                    assert token in dispatched
                    pending = dispatched[token]; c = pending['value']
                    assert message['event'] == 'POSE' and 0 < pending['wall']-now <= .25
                    assert guard is not None and 0 <= now-guard['wall'] < .25 and stopped(guard)
                    previous = c.get('resumes_route_id')
                    if previous:
                        assert recovery and previous in recoveries and recoveries[previous]['result']['command'] == c
                        assert math.dist(message['pose'][:2], c['resume_from']) <= .15
                    else:
                        assert math.dist(message['pose'][:2], nodes[node]) <= .15
                    assert c['nodes'][0] == node
                assert all(message[k] == c[k] for k in (*IDENTITY, 'task_id', 'route_id'))
            else:
                assert active is None or active['command']['route_id'] in arrivals
            assert message['event'] in (('POSE', 'ARRIVED', 'STOPPED') if recovery else ('POSE', 'ARRIVED'))
            if message['event'] == 'STOPPED':
                assert token in cancelled and token not in stops
                assert active['native_result'] is not None and active['native_result'] <= now
                assert guard is not None and 0 <= now-guard['wall'] < .25 and stopped(guard)
                c = active['command']
                assert (guard['run_id'], guard['route_id'], guard['revision']) == (c['run_id'], token, c['revision'])
                stops[token] = row
                stopped_guards[token] = guard
            if message['event'] == 'ARRIVED':
                assert active and token == active['command']['route_id'] and token not in arrivals
                assert token not in cancelled
                assert active['native_result'] is not None and active['native_result'] <= now
                assert active['authorized'] and active['lease_seq'] > 0
                assert guard is not None and 0 <= now-guard['wall'] < .25 and stopped(guard)
                c = active['command']
                assert (guard['run_id'], guard['route_id'], guard['revision']) == (c['run_id'], token, c['revision'])
                assert math.dist(message['pose'][:2], c['target_xy']) <= .15
                arrivals[token] = row; arrival_guards[token] = guard; node = c['nodes'][1]
        elif kind == 'shutdown':
            assert shutdown is None and row['stopped'] and row['clean_idle']
            assert row['execution']['state'] == 'idle' and row['execution']['fault'] is None
            assert guard is not None and stopped(guard) and now-guard['wall'] < .25
            shutdown = row
    assert routes and set(routes) == set(arrivals) | set(cancelled) and shutdown is not None
    assert set(cancelled) == set(recoveries) == set(stops) == set(rejected)
    return dict(routes=routes, arrivals=arrivals, arrival_guards=arrival_guards,
                feedback=feedback, final_node=node, guard_id=guard_id,
                cancelled=cancelled, recoveries=recoveries, stops=stops, stopped_guards=stopped_guards)


def audit_arrival_guards(observations, robot_audits, start_monotonic_s, *, include_stops=False):
    """Match the adapter stop to independently recorded stamped guard messages."""
    keys = ('guard_id', 'run_id', 'route_id', 'revision', 'state', 'stopped',
            'actual_speed', 'actual_wz', 'output_vx', 'output_wz', 'command_age', 'lease_age')
    endpoints = []
    for robot, audit in robot_audits.items():
        points = [(token, row, audit['arrival_guards'][token]) for token, row in audit['arrivals'].items()]
        if include_stops:
            points += [(token, row, audit['stopped_guards'][token]) for token, row in audit['stops'].items()]
        endpoints.extend((robot, token, row, guard) for token, row, guard in points)
    matched = set()
    for raw in observations:
        if raw['kind'] != 'guard': continue
        for index, (robot, token, arrival, guard) in enumerate(endpoints):
            if (raw.get('robot') == robot and all(raw['message'][k] == guard[k] for k in keys)
                    and abs(start_monotonic_s+raw['received_wall_s']-guard['wall']) < .25
                    and -.1 <= arrival['sim']-stamp(raw['message']) < .25):
                matched.add(index)
    assert matched == set(range(len(endpoints))), 'missing independent stamped stop'


def audit_frames(rows, robot_audits, layout):
    nodes = {n['id']: (n['x'], n['y']) for n in layout['nodes']}
    current = {r['id']: r['start'] for r in layout['robots']}; prior_tokens = dict.fromkeys(current)
    frames = []; prior_wall = -math.inf; opposing_wait = 0
    seen = {robot: set() for robot in current}
    for row in rows:
        assert row['wall'] >= prior_wall; prior_wall = row['wall']
        if row['kind'] != 'frame': continue
        frame = row['value']; gates = frame['executions']; pins = frame['physical_holds']
        expected_pins = dict(nodes={}, edges={})
        assert set(gates) == set(current)
        assert all(len(owners) == 1 and owners[0] in current for group in pins.values() for owners in group.values())
        for robot, gate in gates.items():
            assert gate['source'] == 'localized' and gate['frame_id'] == 'map'
            audit = robot_audits[robot]; token = gate['route_id']; prior = prior_tokens[robot]
            if gate['fault'] is not None:
                assert token in audit.get('cancelled', {})
                cancel = audit['cancelled'][token]; resumed = audit['recoveries'][token]
                assert gate['fault'] == cancel['value']['reason']
                assert cancel['wall']-.1 <= row['wall'] <= resumed['wall']
            if prior is not None and prior != token:
                if prior in audit.get('recoveries', {}):
                    resumed = audit['recoveries'][prior]
                    assert token == resumed['result']['command']['route_id'] and resumed['wall'] <= row['wall']
                else:
                    assert prior in audit['arrivals'] and audit['arrivals'][prior]['wall'] <= row['wall']
                    current[robot] = audit['routes'][prior]['command']['nodes'][1]
            if token is not None:
                seen[robot].add(token)
                assert token in audit['routes']; c = audit['routes'][token]['command']
                assert all(gate[k] == c[k] for k in (*IDENTITY, 'task_id', 'route_id'))
                a, b = c['nodes']; assert all(pins['nodes'].get(n) == [robot] for n in (a, b))
                assert all(pins['edges'].get(s+'->'+d) == [robot] for s, d in ((a, b), (b, a)))
                for n in (a, b): expected_pins['nodes'][n] = [robot]
                for s, d in ((a, b), (b, a)): expected_pins['edges'][s+'->'+d] = [robot]
            elif gate['state'] != 'unbound':
                assert pins['nodes'].get(current[robot]) == [robot]
                expected_pins['nodes'][current[robot]] = [robot]
            else:
                expected_pins['nodes'][current[robot]] = [robot]
            if gate['pose'] is not None:
                feedback = audit['feedback'][gate['last_seq']]['message']
                assert gate['pose'] == dict(zip(('x', 'y', 'yaw', 'timestamp'), [*feedback['pose'], feedback['timestamp']]))
                a = next(a for a in frame['agvs'] if a['agv_id'] == robot)['pose']
                assert all(a[k] == gate['pose'][k] for k in ('x', 'y', 'yaw', 'timestamp'))
            prior_tokens[robot] = token
        assert pins == expected_pins, 'unexplained or missing physical occupancy'
        a, b = gates['scout_a']['route_id'], gates['scout_b']['route_id']
        if a and robot_audits['scout_a']['routes'][a]['command']['nodes'] == ['west', 'east'] and b is None:
            assert pins['nodes'].get('bay_east') == ['scout_b']; opposing_wait += 1
        if row.get('sim') is not None:
            frames.append(dict(**row, audit_nodes=dict(current)))
    assert len(frames) >= 3
    assert all(seen[r] == set(a['routes']) for r, a in robot_audits.items())
    assert all(prior_tokens[r] is None and current[r] == a['final_node'] for r, a in robot_audits.items())
    return frames, opposing_wait


def audit_measurements(rows, frames, robot_audits, layout, interval, start_monotonic_s):
    """Check raw independent samples, bounded coverage and reserved corridors."""
    nodes = {n['id']: (n['x'], n['y']) for n in layout['nodes']}
    health = {r: SensorHealth() for r in robot_audits}; checks = {r: [] for r in robot_audits}
    frame_times = [f['sim'] for f in frames]; assert frame_times == sorted(frame_times)
    maximum_error = 0.; maximum_corridor = 0.; travelled = dict.fromkeys(checks, 0.)
    for row in rows:
        robot = row.get('robot')
        if robot not in health: continue
        kind = row['kind']; h = health[robot]
        if kind == 'scan':
            valid = sum(v is not None and row['range_min'] <= v < row['range_max'] for v in row['ranges'])
            h.observe('scan', row['source_s'], row['received_wall_s'], valid >= max(30, .1*len(row['ranges'])))
        elif kind in ('imu', 'filtered_odom'):
            keys = ('yaw_rate',) if kind == 'imu' else ('vx', 'vy', 'wz')
            h.observe('imu' if kind == 'imu' else 'odom', row['source_s'], row['received_wall_s'], all(math.isfinite(row[k]) for k in keys))
        elif kind == 'amcl':
            m = row['message']; p = m['pose']['pose']['position']
            h.observe('amcl', stamp(m), row['received_wall_s'], pose_quality(m['pose']['covariance'], (p['x'], p['y'])))
        elif kind == 'independent_pose_check' and interval[0] <= row['evaluated_wall_s'] <= interval[1]:
            now, source = row['evaluated_wall_s'], row['source_s']
            assert h.reason(row['clock_s'], now) is None and row['health'] is None
            assert row['error_m'] <= .15 and row['error_rad'] <= .20
            assert 0 <= row['speed_mps'] <= .3
            maximum_error = max(maximum_error, row['error_m'])
            index = bisect.bisect_right(frame_times, source)-1; assert index >= 0
            frame = frames[index]; assert 0 <= source-frame['sim'] <= .25
            assert abs(frame['wall']-(start_monotonic_s+now)) <= 1.
            token = frame['value']['executions'][robot]['route_id']
            xy = row['truth'][:2]
            if token:
                c = robot_audits[robot]['routes'][token]['command']
                distance = point_segment(xy, *(nodes[n] for n in c['nodes']))
            else:
                distance = math.dist(xy, nodes[frame['audit_nodes'][robot]])
            assert distance <= .15, (robot, 'outside reserved corridor', source, distance)
            maximum_corridor = max(maximum_corridor, distance)
            if checks[robot]:
                old = checks[robot][-1]; dt = source-old['source_s']
                assert dt > 0 and 0 <= now-old['evaluated_wall_s'] <= 1.
                displacement = math.dist(xy, old['truth'][:2]); assert displacement <= .3*dt+.015
                travelled[robot] += displacement
            checks[robot].append(row)
    for robot, samples in checks.items():
        assert len(samples) >= 3 and samples[0]['evaluated_wall_s']-interval[0] <= 1.
        assert interval[1]-samples[-1]['evaluated_wall_s'] <= 1.
        assert travelled[robot] >= 1.5
    minimum_separation = math.inf
    for a in checks['scout_a']:
        b = min(checks['scout_b'], key=lambda r: abs(r['source_s']-a['source_s']))
        skew = abs(a['source_s']-b['source_s']); assert skew <= .25
        distance = math.dist(a['truth'][:2], b['truth'][:2])-.3*skew
        assert distance >= 2*(layout['robot_enclosing_radius_m']+layout['safety_margin_m'])
        minimum_separation = min(minimum_separation, distance)
    return dict(checks={r: len(c) for r, c in checks.items()}, travelled_m=travelled,
        maximum_localization_error_m=maximum_error, maximum_reserved_corridor_error_m=maximum_corridor,
        minimum_sampled_pair_separation_m=minimum_separation)


def verify(directory, *, write=True):
    directory = Path(directory); prerequisite = verify_readiness(directory, write=write)
    result = json.loads((directory/'result.json').read_text()); mode = result['mode']
    assert mode in ('short', 'delivery') and result['fleet']['case'] == mode
    observations = read_rows(directory/'observations.jsonl')
    markers = list(observations.only('fleet_enabled', 'fleet_completed'))
    enabled = [r for r in markers if r['kind'] == 'fleet_enabled']
    completed = [r for r in markers if r['kind'] == 'fleet_completed']
    assert len(enabled) == len(completed) == 1 and completed[0]['wall_s'] > enabled[0]['wall_s']
    layout = result['network']; audits = {}
    for robot in ('scout_a', 'scout_b'):
        logs = list((directory/'fleet').glob(robot+'-*.jsonl')); assert len(logs) == 1
        audits[robot] = audit_robot(read_rows(logs[0]), robot, layout)
    assert audits['scout_a']['guard_id'] != audits['scout_b']['guard_id']
    assert len({v['command']['run_id'] for a in audits.values() for v in a['routes'].values()}) == 1
    audit_arrival_guards(observations, audits, result['start_monotonic_s'])
    logs = list((directory/'fleet').glob('fleet-events-*.jsonl')); assert len(logs) == 1
    frames, waits = audit_frames(read_rows(logs[0]), audits, layout)
    measurement = audit_measurements(observations, frames, audits, layout,
        (enabled[0]['wall_s'], completed[0]['wall_s']), result['start_monotonic_s'])
    tasks = ({'navigation_short_a', 'navigation_short_b'} if mode == 'short' else {'inspection_consumables', 'small_tools'})
    assert set(result['fleet']['tasks']) == tasks and all(v == 'completed' for v in result['fleet']['tasks'].values())
    if mode == 'delivery': assert waits > 0
    assert frames[-1]['value']['task_counts'] == {'completed': 2}
    for robot, audit in audits.items():
        expected = ('west' if robot == 'scout_a' else 'east') if mode == 'short' else ('receiving' if robot == 'scout_a' else 'supply')
        assert audit['final_node'] == expected
        assert result['fleet']['gates'][robot]['state'] == 'idle' and result['fleet']['gates'][robot]['fault'] is None
        assert result['fleet']['pins']['nodes'].get(expected) == [robot]
    audit = dict(passed=True, scope='normal_sensor_fleet_sampled_motion', mode=mode,
        readiness=prerequisite['scope'], routes_completed={r: len(a['arrivals']) for r, a in audits.items()},
        owner_frames=len(frames), opposing_wait_frames=waits, **measurement,
        result_sha256=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest(),
        claim_boundary='Normal sensor-authoritative dispatch/native-result/stopped-guard/arrival ordering, '
            'sampled localization and road ownership only. Inter-sample contact, process inventory, '
            'fault recovery, complete process termination and industrial acceptance are outside this audit.')
    if write: (directory/'navigation_motion_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('directory', type=Path)
    print(json.dumps(verify(p.parse_args().directory)))


if __name__ == '__main__': main()
