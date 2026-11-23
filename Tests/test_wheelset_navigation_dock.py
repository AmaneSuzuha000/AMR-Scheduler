from copy import deepcopy

import pytest

from tools.wheelset_scene.material_supply import MaterialSupply


def receipt():
    header = dict(frame_id='world', stamp=dict(sec=10, nanosec=0))
    command = dict(scene_id='wheelset_light_material_v1', run_id='test', robot_id='scout_a',
        frame_id='map', time_base='ros_sim', source='localized', task_id='inspection_consumables',
        route_id='test:scout_a:3', revision=3, kind='MOVE', nodes=['east', 'receiving'], target_xy=[60., 69.2])
    observation = dict(pose=[60., 69.2, 0.], linear_speed=0., angular_speed=0., sim_time_s=10.)
    proof = dict(feedback={**command, 'event': 'ARRIVED', 'seq': 20, 'timestamp': 10., 'pose': [60., 69.2, 0.]},
        feedback_result=dict(accepted=True, state='arrived'),
        guard=dict(header=deepcopy(header), guard_id='guard-A', run_id='test', route_id='test:scout_a:3',
            revision=3, state='latched', stopped=True, output_vx=0., output_wz=0., actual_speed=0., actual_wz=0.),
        state_query=dict(query_id=20, service='/wheelset_review/get_entity_state',
                         request=dict(name='scout_a', reference_frame='world')),
        state_reply=dict(success=True, header=deepcopy(header), state=dict(name='',
            pose=dict(position=dict(x=60., y=69.2, z=0.), orientation=dict(x=0., y=0., z=0., w=1.)),
            twist=dict(linear=dict(x=0., y=0., z=0.), angular=dict(x=0., y=0., z=0.)))))
    return command, observation, proof


def test_sensor_dock_is_explicit_and_stock_waits_for_actual_handoff():
    c, o, p = receipt()
    assert not MaterialSupply('test').arrived(c['task_id'], 'magnetic', 4, c, o, now=10., dock_proof=p)
    s = MaterialSupply('test', dock_source='localized')
    assert not s.arrived(c['task_id'], 'magnetic', 4, c, o, now=10.)
    assert s.arrived(c['task_id'], 'magnetic', 4, c, o, now=10., dock_proof=p)
    assert s.stock['magnetic'] == 0 and not s.can_start('ws001', 'magnetic', 10.)
    assert s.events[0]['dock_source'] == 'localized'
    handoff = s.begin_handoff(c['task_id'], now=10.)
    assert s.commit(handoff['token'], dict(time_s=20., xyz=[35.45, 50., 1.], velocity=[0., 0., 0.]),
                    (35.45, 50., 1.), now=20.)
    assert s.stock['magnetic'] == 4
    assert not s.arrived(c['task_id'], 'magnetic', 4, c, o, now=20., dock_proof=p)


@pytest.mark.parametrize('corruption', ['wrong_run', 'wrong_robot', 'wrong_route', 'wrong_frame',
    'unaccepted_arrival', 'moving_guard', 'armed_guard', 'old_revision', 'stale_guard',
    'wrong_query', 'wrong_reply_frame', 'wrong_measured_pose', 'far_sensor_pose',
    'fresh_wrapper_old_state', 'nan_pose'])
def test_localized_dock_rejects_incomplete_or_misbound_physical_receipt(corruption):
    c, o, p = receipt()
    if corruption == 'wrong_run': p['feedback']['run_id'] = 'other'
    if corruption == 'wrong_robot': p['feedback']['robot_id'] = 'scout_b'
    if corruption == 'wrong_route': p['guard']['route_id'] = 'old'
    if corruption == 'wrong_frame': p['feedback']['frame_id'] = 'odom'
    if corruption == 'unaccepted_arrival': p['feedback_result']['accepted'] = False
    if corruption == 'moving_guard': p['guard']['actual_speed'] = .1
    if corruption == 'armed_guard': p['guard']['state'] = 'armed'
    if corruption == 'old_revision': p['guard']['revision'] = 2
    if corruption == 'stale_guard': p['guard']['header']['stamp']['sec'] = 9
    if corruption == 'wrong_query': p['state_query']['request']['name'] = 'scout_b'
    if corruption == 'wrong_reply_frame': p['state_reply']['header']['frame_id'] = 'odom'
    if corruption == 'wrong_measured_pose': p['state_reply']['state']['pose']['position']['x'] += .2
    if corruption == 'far_sensor_pose': p['feedback']['pose'][0] += .11
    if corruption == 'fresh_wrapper_old_state': p['state_reply']['header']['stamp']['sec'] = 9
    if corruption == 'nan_pose': o['pose'][2] = float('nan')
    s = MaterialSupply('test', dock_source='localized')
    assert not s.arrived(c['task_id'], 'magnetic', 4, c, o, now=10., dock_proof=p)
    assert s.stock['magnetic'] == 0 and not s.deliveries and not s.events
