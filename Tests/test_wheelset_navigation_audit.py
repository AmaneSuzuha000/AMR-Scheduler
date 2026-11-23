from copy import deepcopy

import pytest

from tools.wheelset_scene.verify_navigation_readiness import replay
from tools.wheelset_scene.run_navigation_readiness import ReadinessWindow, SpawnPriorHandshake


def evidence():
    rows = []
    q = dict(x=0., y=0., z=0., w=1.)
    for index in range(36):
        wall = index/10; source = 1+wall
        header = dict(stamp=dict(sec=int(source), nanosec=round((source-int(source))*1e9)), frame_id='map')
        source = header['stamp']['sec']+header['stamp']['nanosec']*1e-9
        for robot, x in [('scout_a', 20.), ('scout_b', 40.)]:
            def add(kind, **values):
                rows.append(dict(wall_s=wall, robot=robot, kind=kind, **deepcopy(values)))
            p = dict(x=x, y=71., z=0.)
            if index == 0: add('initial_pose', message={})
            add('scan', source_s=source, received_wall_s=wall, beams=40, frame_id='laser_link',
                range_min=.05, range_max=8., ranges=[2.]*40, finite_returns=40)
            add('imu', source_s=source, received_wall_s=wall, yaw_rate=0.)
            add('filtered_odom', source_s=source, received_wall_s=wall, vx=0., vy=0., wz=0.)
            add('amcl', received_wall_s=wall, message=dict(header=header,
                pose=dict(pose=dict(position=p, orientation=q), covariance=[0.]*36)))
            add('guard', received_wall_s=wall, message=dict(header=header, guard_id=robot+'-guard',
                state='latched', stopped=True, actual_speed=0., actual_wz=0., output_vx=0., output_wz=0.))
            for node in ('controller_server', 'map_server', 'amcl'):
                add('lifecycle', node=node, state=3, received_wall_s=wall)
            query = dict(query_id=index, service='/wheelset_review/get_entity_state',
                         request=dict(name=robot, reference_frame='world'), issued_wall_s=wall)
            add('state_query_issued', **query)
            add('independent_pose_check', source_s=source, evaluated_wall_s=wall,
                clock_s=source, state_received_wall_s=wall, state_query=query,
                truth=[x, 71., 0.], localized=[x, 71., 0.], error_m=0., error_rad=0., speed_mps=0., health=None,
                state_reply=dict(success=True, header={**header, 'frame_id': 'world'}, state=dict(name='',
                    pose=dict(position=p, orientation=q), twist=dict(linear=dict(x=0., y=0., z=0.)))),
                transform=dict(header=header, child_frame_id='base_footprint',
                    transform=dict(translation=p, rotation=q)))
            add('readiness_state', source_s=source, evaluated_wall_s=wall, ready=True, action_server_ready=True)
    return rows


def test_independent_readiness_replay_accepts_raw_complete_startup():
    rows = evidence()
    window = ReadinessWindow()
    interval = None
    for index in range(21):
        interval = window.update([True, True], [index*.1, index*.1], True)
    assert interval is not None
    counts = replay(rows, dict(start_wall_s=interval[0], end_wall_s=interval[1]))
    assert counts['scout_a']['checks'] == counts['scout_b']['scan'] == 36


def reject_late_replies(rows, robot, indices):
    """Replace accepted pose checks with the late-reply outcome candidate 31 records."""
    replaced = []
    for row in rows:
        if row['kind'] != 'independent_pose_check' or row['robot'] != robot:
            continue
        if row['state_query']['query_id'] not in indices:
            continue
        source = row['source_s']
        late = dict(kind='pose_check_rejected_stale', wall_s=row['wall_s'], robot=robot,
                    source_s=source, clock_s=source+5., age_s=5., state_query=row['state_query'],
                    reason='physics reply outside [-0.1, 0.25] s of clock')
        row.clear(); row.update(late)
        replaced.append(row)
    return replaced


def test_late_pose_reply_is_audited_instead_of_dropped_silently():
    rows = evidence()
    replaced = reject_late_replies(rows, 'scout_a', {34, 35})
    assert len(replaced) == 2
    counts = replay(rows, dict(start_wall_s=1., end_wall_s=3.))
    assert counts['scout_a']['checks'] == 34
    assert counts['scout_a']['rejected_stale'] == 2
    assert counts['scout_b']['rejected_stale'] == 0


def test_pose_check_query_consumed_mid_stream_cannot_go_missing():
    """The old code popped a paired-but-stale reply with no event; that hole must fail."""
    rows = [row for row in evidence()
            if not (row['kind'] == 'independent_pose_check' and row['robot'] == 'scout_a'
                    and row['state_query']['query_id'] == 10)]
    with pytest.raises(AssertionError):
        replay(rows, dict(start_wall_s=1., end_wall_s=3.))


def test_spawn_prior_send_requires_active_inputs_and_locked_motion():
    handshake = SpawnPriorHandshake(40., 71., -1.57)
    assert handshake.update(10., 1., ready=False, motion_locked=True) == 'wait'
    assert handshake.update(10., 1., ready=True, motion_locked=False) == 'wait'
    assert handshake.update(10., 0., ready=True, motion_locked=True) == 'wait'
    assert handshake.attempts == 0
    assert handshake.update(10., 1., ready=True, motion_locked=True) == 'send'
    assert handshake.attempts == 1 and not handshake.confirmed


def test_spawn_prior_origin_estimate_never_acknowledges_a_remote_workshop_bay():
    handshake = SpawnPriorHandshake(40., 71., -1.57)
    assert handshake.update(10., 1., ready=True, motion_locked=True) == 'send'
    assert handshake.update(14., 4., ready=True, motion_locked=True,
        estimate=(0., 0., 0.), estimate_source_s=4., estimate_received=14.,
        estimate_valid=True) == 'wait'
    assert handshake.update(15., 5., ready=True, motion_locked=True,
        estimate=(0., 0., 0.), estimate_source_s=5., estimate_received=15.,
        estimate_valid=True) == 'send'
    assert handshake.attempts == 2 and not handshake.confirmed


@pytest.mark.parametrize('change', [
    dict(estimate_source_s=.5), dict(estimate_source_s=5.2),
    dict(estimate_received=9.), dict(estimate_received=11.),
    dict(estimate_valid=False), dict(estimate=(40., 71., -.9)),
])
def test_spawn_prior_rejects_replayed_future_stale_invalid_or_wrong_heading(change):
    handshake = SpawnPriorHandshake(40., 71., -1.57)
    handshake.update(10., 1., ready=True, motion_locked=True)
    estimate = dict(estimate=(40., 71., -1.57), estimate_source_s=5.,
                    estimate_received=14., estimate_valid=True)
    estimate.update(change)
    assert handshake.update(14., 5., ready=True, motion_locked=True, **estimate) == 'wait'
    assert not handshake.confirmed


def test_spawn_prior_confirms_fresh_estimate_once_and_never_resets_during_motion():
    handshake = SpawnPriorHandshake(40., 71., 3.13)
    handshake.update(10., 1., ready=True, motion_locked=True)
    assert handshake.update(11., 2., ready=True, motion_locked=True,
        estimate=(40.1, 70.9, -3.13), estimate_source_s=1.9,
        estimate_received=10.9, estimate_valid=True) == 'acknowledged'
    assert handshake.confirmed
    assert handshake.update(100., 50., ready=True, motion_locked=False) == 'confirmed'
    assert handshake.attempts == 1 and handshake.sent_source == 1.


def test_spawn_prior_retries_are_bounded_and_do_not_extend_the_startup_budget():
    handshake = SpawnPriorHandshake(40., 71., 0.)
    for index in range(3):
        assert handshake.update(10.+5*index, 1.+index,
                                ready=True, motion_locked=True) == 'send'
    with pytest.raises(RuntimeError, match='not acknowledged'):
        handshake.update(25., 4., ready=True, motion_locked=True)
    assert handshake.attempts == 3 and not handshake.confirmed


def test_late_pose_reply_must_bind_its_query_and_lie_outside_the_window():
    rows = evidence()
    replaced = reject_late_replies(rows, 'scout_a', {34})
    assert replay(rows, dict(start_wall_s=1., end_wall_s=3.))['scout_a']['rejected_stale'] == 1
    late = replaced[0]

    late['clock_s'] = late['source_s']+.2; late['age_s'] = .2
    with pytest.raises(AssertionError):
        replay(rows, dict(start_wall_s=1., end_wall_s=3.))

    late['clock_s'] = late['source_s']+5.; late['age_s'] = 5.
    late['state_query'] = deepcopy(late['state_query']); late['state_query']['query_id'] = 999
    with pytest.raises(AssertionError):
        replay(rows, dict(start_wall_s=1., end_wall_s=3.))

    late['state_query']['query_id'] = 34; late['age_s'] = 4.
    with pytest.raises(AssertionError):
        replay(rows, dict(start_wall_s=1., end_wall_s=3.))


def test_failed_physics_state_reply_is_recorded_instead_of_losing_its_query_id():
    """A success=false GetEntityState reply used to drop the query with no event."""
    rows = evidence()
    replaced = 0
    for row in rows:
        if row['kind'] != 'independent_pose_check' or row['robot'] != 'scout_a':
            continue
        if row['state_query']['query_id'] not in {5, 6}:
            continue
        failed = dict(kind='pose_check_service_failed', wall_s=row['wall_s'], robot='scout_a',
                      state_query=row['state_query'], reply_success=False,
                      reason='physics state service returned no usable reply')
        row.clear(); row.update(failed)
        replaced += 1
    assert replaced == 2
    counts = replay(rows, dict(start_wall_s=1., end_wall_s=3.))
    assert counts['scout_a']['service_failed'] == 2
    assert counts['scout_a']['checks'] == 34
    assert counts['scout_b']['service_failed'] == 0
    # The audit must still refuse a silently missing id once the record is absent.
    rows[:] = [row for row in rows
               if not (row.get('robot') == 'scout_a' and row['kind'] == 'pose_check_service_failed'
                       and row['state_query']['query_id'] == 5)]
    with pytest.raises(AssertionError):
        replay(rows, dict(start_wall_s=1., end_wall_s=3.))


def test_service_failed_record_cannot_claim_a_successful_reply():
    rows = evidence()
    for row in rows:
        if (row['kind'] == 'independent_pose_check' and row['robot'] == 'scout_a'
                and row['state_query']['query_id'] == 5):
            failed = dict(kind='pose_check_service_failed', wall_s=row['wall_s'], robot='scout_a',
                          state_query=row['state_query'], reply_success=True,
                          reason='physics state service returned no usable reply')
            row.clear(); row.update(failed)
            break
    with pytest.raises(AssertionError):
        replay(rows, dict(start_wall_s=1., end_wall_s=3.))


def test_contact_readiness_requires_the_configured_live_costmap_parameters():
    rows = evidence()
    nodes = ['amcl', 'ekf_filter_node', 'controller_server', 'local_costmap/local_costmap']
    expected = {robot: {name: {} for name in nodes} for robot in ('scout_a', 'scout_b')}
    extra = [dict(kind='runtime_parameters', robot=robot, node=node, wall_s=0.)
             for robot in expected for node in nodes]
    for row in rows:
        if row['kind'] == 'readiness_state': row['parameter_nodes_required'] = list(nodes)
    assert replay(extra+rows, dict(start_wall_s=1., end_wall_s=3.),
                  parameter_nodes=expected)['scout_a']['checks'] == 36
    for row in rows:
        if row['kind'] == 'readiness_state': row['parameter_nodes_required'] = nodes[:-1]
    with pytest.raises(AssertionError):
        replay(extra+rows, dict(start_wall_s=1., end_wall_s=3.), parameter_nodes=expected)


@pytest.mark.parametrize('spawn_yaw', [0., -1.5707963267948966])
def test_initial_prior_is_bound_to_configured_spawn_heading(spawn_yaw):
    import math
    from tools.wheelset_scene.verify_navigation_readiness import audit_initial_priors
    layout = dict(nodes=[dict(id='bay', x=20., y=71.)],
                  robots=[dict(id='scout_a', start='bay', start_yaw_rad=spawn_yaw)])
    q = dict(x=0., y=0., z=math.sin((spawn_yaw+.05)/2), w=math.cos((spawn_yaw+.05)/2))
    covariance = [0.]*36
    for index in (0, 7, 35): covariance[index] = .1**2
    row = dict(kind='initial_pose', robot='scout_a',
        source='known spawn prior plus fixed perturbation; independent of state queries',
        message=dict(header=dict(frame_id='map'), pose=dict(covariance=covariance,
            pose=dict(position=dict(x=20.15, y=70.92, z=0.), orientation=q))))
    assert audit_initial_priors([row], layout) == 1
    row['message']['pose']['pose']['orientation'] = dict(x=0., y=0., z=0., w=1.)
    with pytest.raises(AssertionError): audit_initial_priors([row], layout)


def test_yaw_reduction_roundoff_does_not_change_error_gate():
    rows = evidence()
    for row in rows:
        if row['kind'] == 'independent_pose_check':
            row['truth'][2] += 3e-16
    assert replay(rows, dict(start_wall_s=1., end_wall_s=3.))['scout_a']['checks'] == 36
    for row in rows:
        if row['kind'] == 'independent_pose_check':
            row['truth'][2] += 1e-8
    with pytest.raises(AssertionError):
        replay(rows, dict(start_wall_s=1., end_wall_s=3.))


def test_readiness_window_uses_actual_common_poll_times_and_resets_on_fault():
    window = ReadinessWindow(2.)
    interval = None
    for i in range(21):
        interval = window.update([True, True], [10.+i*.1, 10.01+i*.1], True)
        if i < 20:
            assert interval is None
    assert interval == (10., 12.01)
    assert window.started_at is None
    window.update([True, True], [20., 20.01], True)
    assert window.update([True, False], [20.1, 20.11], True) is None
    assert window.started_at is None


@pytest.mark.parametrize('cause', ['observer_skew', 'adapter_delay', 'gap', 'regression'])
def test_readiness_window_does_not_bridge_unobserved_or_stale_health(cause):
    window = ReadinessWindow(2.)
    for i in range(20):
        assert window.update([True, True], [10.+i*.1, 10.01+i*.1], True) is None
    times = [12., 12.01]
    checked_at = 12.02
    if cause == 'observer_skew':
        times = [12., 12.06]; checked_at = 12.07
    if cause == 'adapter_delay': checked_at = 12.08
    if cause == 'gap': times = [12.2, 12.21]; checked_at = 12.22
    if cause == 'regression': times = [11.8, 11.81]; checked_at = 11.82
    assert window.update([True, True], times, True, checked_at=checked_at) is None
    assert window.started_at is None or window.started_at == times[0]


def test_readiness_window_rejects_mismatched_or_nonfinite_timestamps():
    window = ReadinessWindow()
    with pytest.raises(ValueError):
        window.update([True], [1., 2.], True)
    with pytest.raises(ValueError):
        window.update([True], [float('nan')], True)


@pytest.mark.parametrize('corruption', ['scan_replay', 'moving_guard', 'inactive_controller',
    'false_pose', 'wrong_tf_frame', 'odom_replay', 'wrong_state_request',
    'wrong_query_binding', 'duplicate_query_response', 'wrong_state_response', 'wrong_state_frame'])
def test_independent_audit_rejects_causal_corruption_despite_true_ready_flag(corruption):
    rows = evidence()
    for row in rows:
        if row['robot'] != 'scout_a' or row['wall_s'] < 1.:
            continue
        if corruption == 'scan_replay' and row['kind'] == 'scan':
            row['source_s'] = 1.9  # New receipts and true flags cannot refresh old source data.
        if corruption == 'odom_replay' and row['kind'] == 'filtered_odom':
            row['source_s'] = 1.9
        if corruption == 'moving_guard' and row['kind'] == 'guard':
            row['message']['actual_speed'] = .3
        if corruption == 'inactive_controller' and row['kind'] == 'lifecycle' and row['node'] == 'controller_server':
            row['state'] = 2
        if corruption == 'false_pose' and row['kind'] == 'independent_pose_check':
            row['state_reply']['state']['pose']['position']['x'] += .4
            row['truth'][0] += .4
            row['error_m'] = .4  # All derived claims updated, but true readiness is now unjustified.
        if corruption == 'wrong_tf_frame' and row['kind'] == 'independent_pose_check':
            row['transform']['header']['frame_id'] = 'odom'
        if corruption == 'wrong_state_request' and row['kind'] == 'state_query_issued':
            row['request']['name'] = 'scout_b'
        if corruption == 'wrong_query_binding' and row['kind'] == 'independent_pose_check':
            row['state_query']['service'] = '/other/get_entity_state'
        if corruption == 'duplicate_query_response' and row['kind'] == 'independent_pose_check':
            row['state_query']['query_id'] = 0
            row['state_query']['issued_wall_s'] = 0.
        if corruption == 'wrong_state_response' and row['kind'] == 'independent_pose_check':
            row['state_reply']['state']['name'] = 'scout_b'
        if corruption == 'wrong_state_frame' and row['kind'] == 'independent_pose_check':
            row['state_reply']['header']['frame_id'] = 'map'
    with pytest.raises(AssertionError):
        replay(rows, dict(start_wall_s=1., end_wall_s=3.))
