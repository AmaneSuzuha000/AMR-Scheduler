"""Synthetic causal tests only; no Gazebo motion is claimed by these fixtures."""
from copy import deepcopy

import pytest

from tools.wheelset_scene.light_material_network import proposed_network
from tools.wheelset_scene.verify_navigation_motion import audit_robot, audit_frames, audit_measurements, audit_arrival_guards


def evidence():
    layout = proposed_network(); logs = {}; observations = []; frames = []
    for robot, x, start, target in [('scout_a', 20., 'bay_west', 'west'), ('scout_b', 40., 'bay_east', 'east')]:
        identity = dict(scene_id=layout['id'], run_id='synthetic', robot_id=robot,
                        frame_id='map', time_base='ros_sim', source='localized')
        c = dict(**identity, task_id='navigation_short_'+robot[-1], route_id=robot+':1', revision=1,
                 kind='MOVE', nodes=[start, target], target_xy=[x, 69.2], speed=.2)
        events = []
        def add(t, kind, **values):
            events.append(dict(wall=100+t, sim=1+t, kind=kind, **deepcopy(values)))
        add(.1, 'command', value=c)
        add(.11, 'path', route_id=c['route_id'], frame='map', transform_time=0., xy=[[x, 71.], [x, 69.2]])
        add(.12, 'native_accept', route_id=c['route_id'], accepted=True)
        add(.125, 'guard_authorize_requested', route_id=c['route_id'], revision=1)
        add(.13, 'guard_authorize', route_id=c['route_id'], accepted=True)
        for tick in range(94):
            t = tick/10; source = 1+t; y = 71.-.2*max(0., min(9., t-.2))
            p = dict(x=x, y=y, z=0.); q = dict(x=0., y=0., z=0., w=1.)
            header = dict(frame_id='map', stamp=dict(sec=int(source), nanosec=round((source-int(source))*1e9)))
            source = header['stamp']['sec']+header['stamp']['nanosec']*1e-9
            active = .1 <= t < 9.2
            token = c['route_id'] if active or tick == 92 else None
            msg = dict(**identity, task_id=c['task_id'] if token else None, route_id=token,
                       seq=tick, timestamp=source, event='ARRIVED' if tick == 92 else 'POSE', pose=[x, y, 0.])
            if active and t >= .2:
                add(t-.01, 'lease', route_id=c['route_id'], revision=1, seq=tick, stamp=source-.01, subscribers=1)
            g = dict(guard_id=robot+'-guard', run_id=c['run_id'], route_id=token, revision=1,
                     state='armed' if active and t >= .2 else 'latched', stopped=not(active and t >= .2),
                     actual_speed=.2 if active and t >= .2 else 0., actual_wz=0., output_vx=.2 if active and t >= .2 else 0.,
                     output_wz=0., command_age=.01, lease_age=.01)
            add(t+.0001, 'guard', **g)
            if tick == 92: add(t+.0002, 'native_result', route_id=c['route_id'], status=4)
            add(t+.0003, 'localization', x=x, y=y, yaw=0., stamp=source)
            add(t+.0004, 'feedback', message=msg, result=dict(accepted=True))
            obs = lambda kind, **values: observations.append(dict(kind=kind, robot=robot, wall_s=t+.006,
                **deepcopy(values)))
            obs('scan', source_s=source, received_wall_s=t, ranges=[2.]*40, range_min=.05, range_max=8.)
            obs('imu', source_s=source, received_wall_s=t, yaw_rate=0.)
            obs('filtered_odom', source_s=source, received_wall_s=t, vx=0., vy=-.2 if active else 0., wz=0.)
            obs('amcl', received_wall_s=t, message=dict(header=header, pose=dict(pose=dict(position=p, orientation=q), covariance=[0.]*36)))
            obs('guard', received_wall_s=t, message=dict(header=header, **g))
            obs('independent_pose_check', source_s=source, evaluated_wall_s=t+.005, clock_s=source,
                health=None, error_m=0., error_rad=0., speed_mps=.2 if active else 0., truth=[x, y, 0.])
            if robot == 'scout_a': frames.append(dict(kind='frame', wall=100+t+.001, sim=source,
                value=dict(executions={}, agvs=[], physical_holds=dict(nodes={}, edges={}), task_counts={})))
            f = frames[tick]['value']; gate_token = c['route_id'] if active else None
            f['executions'][robot] = dict(**identity, task_id=c['task_id'] if gate_token else None,
                route_id=gate_token, fault=None, state='moving' if active else 'idle', last_seq=tick,
                pose=dict(x=x, y=y, yaw=0., timestamp=source))
            f['agvs'].append(dict(agv_id=robot, pose=dict(x=x, y=y, yaw=0., timestamp=source)))
            if active:
                for n in (start, target): f['physical_holds']['nodes'][n] = [robot]
                for a, b in ((start, target), (target, start)): f['physical_holds']['edges'][a+'->'+b] = [robot]
            else: f['physical_holds']['nodes'][start if t < .1 else target] = [robot]
        add(9.31, 'shutdown', stopped=True, clean_idle=True, execution=dict(state='idle', fault=None))
        logs[robot] = sorted(events, key=lambda r: r['wall'])
    return layout, logs, frames, sorted(observations, key=lambda r: r['wall_s'])


def audit(layout, logs, frames, observations):
    robots = {r: audit_robot(rows, r, layout) for r, rows in logs.items()}
    audit_arrival_guards(observations, robots, 100.)
    checked, _ = audit_frames(frames, robots, layout)
    return audit_measurements(observations, checked, robots, layout, (.2, 9.21), 100.)


def test_normal_short_sampled_motion_has_complete_causal_chain():
    result = audit(*evidence())
    assert all(1.79 < d < 1.81 for d in result['travelled_m'].values())
    assert result['minimum_sampled_pair_separation_m'] == 20.


def delayed_service_ack_evidence(corruption=None):
    layout, logs, frames, observations = evidence()
    for robot, rows in logs.items():
        rows[:] = [r for r in rows if r['kind'] not in ('guard_authorize_requested', 'guard_authorize')]
        command = next(r['value'] for r in rows if r['kind'] == 'command')
        if corruption != 'missing_request':
            rows.append(dict(wall=100.125, sim=1.125, kind='guard_authorize_requested',
                route_id=command['route_id'], revision=command['revision']))
        status = dict(wall=100.126, sim=1.126, kind='guard', guard_id=robot+'-guard',
            run_id=command['run_id'], route_id=command['route_id'], revision=command['revision'],
            state='armed', stopped=False, actual_speed=0., actual_wz=0., output_vx=0., output_wz=0.,
            command_age=-1., lease_age=-1.)
        if corruption == 'wrong_guard_route' and robot == 'scout_a':
            status['route_id'] = 'other-route'
        rows.append(status)
        rows.append(dict(wall=100.127, sim=1.127, kind='guard_authorization_observed',
            route_id=command['route_id'], revision=command['revision'],
            source='fresh_exact_guard_status'))
        rows.append(dict(wall=100.5, sim=1.5, kind='guard_authorize',
            route_id=command['route_id'], accepted=True))
        rows.sort(key=lambda r: r['wall'])
    return layout, logs, frames, observations


def test_fresh_exact_guard_status_authorizes_before_delayed_service_reply():
    result = audit(*delayed_service_ack_evidence())
    assert all(1.79 < d < 1.81 for d in result['travelled_m'].values())


@pytest.mark.parametrize('corruption', ['missing_request', 'wrong_guard_route'])
def test_guard_status_authorization_requires_pending_exact_request(corruption):
    with pytest.raises(AssertionError):
        audit(*delayed_service_ack_evidence(corruption))


def test_stationary_owner_pose_can_precede_adapter_dispatch_by_one_tick():
    layout, logs, frames, observations = evidence()
    for rows in logs.values():
        next(r for r in rows if r['kind'] == 'command')['wall'] = 100.11
        rows.sort(key=lambda r: r['wall'])
    audit(layout, logs, frames, observations)


@pytest.mark.parametrize('corruption', ['unknown_route', 'moving_guard', 'early_arrival', 'wrong_task', 'long_gap'])
def test_owner_pose_before_dispatch_remains_stationary_and_bound(corruption):
    layout, logs, frames, observations = evidence()
    rows = logs['scout_a']
    command = next(r for r in rows if r['kind'] == 'command')
    command['wall'] = 100.11
    pose = next(r for r in rows if r['kind'] == 'feedback' and r['message']['route_id'])
    if corruption == 'unknown_route': pose['message']['route_id'] = 'unknown'
    elif corruption == 'moving_guard':
        next(r for r in rows if r['kind'] == 'guard' and r['wall'] > 100.1)['actual_speed'] = .1
    elif corruption == 'early_arrival': pose['message']['event'] = 'ARRIVED'
    elif corruption == 'wrong_task': pose['message']['task_id'] = 'wrong'
    else: command['wall'] = 100.4
    rows.sort(key=lambda r: r['wall'])
    with pytest.raises(AssertionError): audit(layout, logs, frames, observations)


@pytest.mark.parametrize('corruption', ['truth_authority', 'missing_native_result', 'moving_arrival',
    'wrong_lease', 'early_release', 'lost_reverse_pin', 'false_localization', 'stale_scan',
    'outside_corridor', 'missing_measurements', 'forged_frame_pose', 'stale_arrival_guard', 'ghost_pin'])
def test_motion_audit_rejects_causal_corruption(corruption):
    layout, logs, frames, observations = evidence()
    rows = logs['scout_a']
    if corruption == 'truth_authority':
        next(r for r in rows if r['kind'] == 'command')['value']['source'] = 'gazebo_truth'
    elif corruption == 'missing_native_result':
        logs['scout_a'] = [r for r in rows if r['kind'] != 'native_result']
    elif corruption == 'moving_arrival':
        next(r for r in rows if r['kind'] == 'guard' and abs(r['wall']-109.2001) < 1e-7)['actual_speed'] = .2
    elif corruption == 'wrong_lease':
        next(r for r in rows if r['kind'] == 'lease')['revision'] = 2
    elif corruption == 'early_release':
        frames[50]['value']['executions']['scout_a']['route_id'] = None
    elif corruption == 'lost_reverse_pin':
        del frames[50]['value']['physical_holds']['edges']['west->bay_west']
    elif corruption == 'ghost_pin':
        frames[50]['value']['physical_holds']['nodes']['receiving'] = ['scout_a']
    elif corruption == 'forged_frame_pose':
        frames[50]['value']['executions']['scout_a']['pose']['x'] += .01
    elif corruption == 'false_localization':
        next(r for r in observations if r['robot'] == 'scout_a' and r['kind'] == 'independent_pose_check' and r['wall_s'] > 3.)['error_m'] = .16
    elif corruption == 'stale_scan':
        for r in observations:
            if r['robot'] == 'scout_a' and r['kind'] == 'scan' and r['wall_s'] > 3.: r['source_s'] = 3.9
    elif corruption == 'outside_corridor':
        next(r for r in observations if r['robot'] == 'scout_a' and r['kind'] == 'independent_pose_check' and r['wall_s'] > 3.)['truth'][0] += .16
    elif corruption == 'stale_arrival_guard':
        for r in observations:
            if r['robot'] == 'scout_a' and r['kind'] == 'guard' and r['wall_s'] > 9.:
                r['message']['header']['stamp'] = dict(sec=1, nanosec=0)
    else:
        observations = [r for r in observations if not (r['robot'] == 'scout_a' and r['kind'] == 'independent_pose_check' and 3. < r['wall_s'] < 5.)]
    with pytest.raises(AssertionError): audit(layout, logs, frames, observations)
