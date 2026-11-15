import math

import pytest

from tools.wheelset_scene.navigation_fleet import build_sensor_fleet


@pytest.mark.parametrize('case', ['short', 'delivery'])
def test_sensor_fleet_uses_actual_feedback_owned_resources_for_two_jobs(case):
    sim = build_sensor_fleet(run_id='test-sensor-fleet', case=case)
    positions = {robot: gate.start_xy for robot, gate in sim.executions.items()}
    motion = {}; waited = False
    for step in range(6000):
        now = step*.2
        for robot, gate in sim.executions.items():
            command = gate.command; event = 'POSE'; xy = positions[robot]
            if command:
                token = command['route_id']
                if token not in motion:
                    motion[token] = dict(start=now, source=xy,
                        duration=math.dist(xy, command['target_xy'])/command['speed']*1.2+.2)
                action = motion[token]
                fraction = min(1., (now-action['start'])/action['duration'])
                xy = tuple(a+(b-a)*fraction for a, b in zip(action['source'], command['target_xy']))
                positions[robot] = xy
                if fraction == 1: event = 'ARRIVED'
            message = {**gate.envelope(), 'event': event, 'pose': [*xy, 0.],
                       'seq': gate.last_seq+1, 'timestamp': now}
            if step == 0:
                bad = sim.execution_feedback({**message, 'source': 'gazebo_truth'}, now)
                assert not bad['accepted'] and bad['reason'] == 'identity mismatch'
                assert not gate.bound
            answer = sim.execution_feedback(message, now)
            assert answer['accepted']
            assert gate.identity['frame_id'] == 'map' and gate.identity['source'] == 'localized'
        sim.step_once()
        assert all(len(owners) == 1 for group in sim._res.pins().values() for owners in group.values())
        if case == 'delivery':
            a, b = sim.executions['scout_a'], sim.executions['scout_b']
            if a.command and a.command['nodes'] == ['west', 'east']:
                assert b.command is None and positions['scout_b'] == (40., 71.)
                waited = True
        assert not any(g.fault for g in sim.executions.values())
        if sim._all_done(): break
    assert sim._all_done() and len(motion) >= (2 if case == 'short' else 6)
    if case == 'delivery': assert waited


def test_sensor_fleet_rejects_unverified_controller_speed():
    with pytest.raises(ValueError, match='bound'):
        build_sensor_fleet(run_id='test-too-fast', speed_mps=.5)


def test_workshop_measured_handling_is_opt_in_and_arrival_never_substitutes_for_loading():
    sim = build_sensor_fleet(run_id='test-handling-fleet', handling_source='synthetic_contract_test')
    for step in range(30):
        now = step*.2
        for robot, gate in sim.executions.items():
            assert sim.execution_feedback({**gate.envelope(), 'event':'POSE', 'pose':[*gate.start_xy, 0.],
                'seq':gate.last_seq+1, 'timestamp':now}, now)['accepted']
        sim.step_once()
    assert len(sim.handlings) == 2
    assert all(h.command['operation'] == 'LOAD' and not h.completed for h in sim.handlings.values())
    assert all(g.command is None for g in sim.executions.values())
    assert all(t.status.value == 'assigned' and t.pickup_step is None for t in sim.tasks)
    assert sim._res.pins() == dict(nodes={'bay_west':['scout_a'], 'bay_east':['scout_b']}, edges={})
    assert not build_sensor_fleet(run_id='historical-timed-mode').handlings
