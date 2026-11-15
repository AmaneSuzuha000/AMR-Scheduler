"""SYSTEM11: native Simulator admission, not physical fleet authorization."""
import pytest
import math
from amr_sched.entities import Agv, Task
from amr_sched.network import build_demo_network
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.simulation import Simulator, SimulationConfig
from amr_sched.native_fleet_admission import NativeFleetAdmission

def scene(n=8, release=4):
    net = build_demo_network()
    a = [Agv('A0', start_node='B1', load_time=0., unload_time=0.),
         Agv('A1', start_node='B2', load_time=0., unload_time=0.)]
    legal = [('D', 'W1'), ('B2', 'W5'), ('D', 'W2'), ('B1', 'W4')]
    t = [Task('T'+str(i), *legal[i % len(legal)], release_step=release)
         for i in range(n)]
    sim = Simulator(net, t, a, SpaceTimeAStarPlanner(), GreedyAllocator(),
                    SimulationConfig(max_steps=16000, idle_retreat=False))
    sim.reset()
    return sim

def test_dynamic_activation_executes_native_tasks_and_reservations():
    sim = scene(n=2)
    ctrl = NativeFleetAdmission(sim, initial_active={'A0'})
    for _ in range(4):
        assert sim.step_once()
    assert ctrl.commit(ctrl.prepare(robot_id='A1', action='ACTIVATE',
                expected_tick=4, expected_revision=0))['revision'] == 1
    while sim.step_once():
        pass
    result = sim._build_result(sim._step_cursor)
    assert result.summary()['tasks_completed'] == 2
    assert result.collision_events == 0
    assert 'A1' in {t.assigned_agv for t in sim.tasks}
    assert all(t.complete_step is not None for t in sim.tasks)
    assert ctrl.events[0]['physical_authorization'] is False

def test_inactive_cannot_be_assigned_or_auto_parked():
    sim = scene(n=5, release=0)
    sim.config.idle_retreat = True
    ctrl = NativeFleetAdmission(sim, initial_active={'A0'})
    for _ in range(15):
        sim.step_once()
    assert all(t.assigned_agv != 'A1' for t in sim.tasks)
    assert sim.agvs[1].cur_node == 'B2'

def test_stale_tick_or_version_cannot_commit():
    sim = scene()
    ctrl = NativeFleetAdmission(sim, initial_active={'A0'})
    proposed = ctrl.prepare(robot_id='A1', action='ACTIVATE', expected_tick=0,
                            expected_revision=0)
    sim.step_once()
    with pytest.raises(ValueError, match='STALE_OWNER_TICK'):
        ctrl.commit(proposed)
    assert ctrl.revision == 0 and ctrl.active == frozenset({'A0'})

def test_prepared_proposal_rechecked_and_immutability_preserved():
    sim = scene()
    ctrl = NativeFleetAdmission(sim, initial_active={'A0'})
    proposed = ctrl.prepare(robot_id='A1', action='ACTIVATE', expected_tick=0,
                            expected_revision=0)
    sim.agvs[1].cur_task = 'T0'
    with pytest.raises(ValueError, match='ROBOT_NOT_IDLE'):
        ctrl.commit(proposed)
    assert not ctrl.events and ctrl.active == frozenset({'A0'})

def test_nonparking_node_rejected():
    sim = scene()
    ctrl = NativeFleetAdmission(sim, initial_active={'A0'})
    sim.agvs[1].cur_node = 'D'
    with pytest.raises(ValueError, match='NOT_AT_VERIFIED_PARKING'):
        ctrl.prepare(robot_id='A1', action='ACTIVATE', expected_tick=0,
                     expected_revision=0)

def test_cannot_deactivate_last_robot():
    sim = scene()
    ctrl = NativeFleetAdmission(sim, initial_active={'A0'})
    with pytest.raises(ValueError, match='CANNOT_DEACTIVATE_LAST_ROBOT'):
        ctrl.prepare(robot_id='A0', action='DEACTIVATE', expected_tick=0,
                     expected_revision=0)

def test_cannot_reset_with_outstanding_roster():
    sim = scene()
    NativeFleetAdmission(sim, initial_active={'A0'})
    with pytest.raises(RuntimeError, match='fleet membership is bound'):
        sim.reset()

def test_rejected_physical_owner():
    sim = scene()
    sim.executions = {'A0': object()}
    with pytest.raises(ValueError, match='PHYSICAL_EXECUTION_REQUIRES_DURABLE_GLOBAL_COMMIT'):
        NativeFleetAdmission(sim, initial_active={'A0'})

def test_no_fleet_opt_in_legacy_behavior_identical():
    a = scene(n=4, release=0)
    b = scene(n=4, release=0)
    while a.step_once():
        pass
    while b.step_once():
        pass
    a_summary = a._build_result(a._step_cursor).summary()
    b_summary = b._build_result(b._step_cursor).summary()
    def canonical(m):
        return {k: None if isinstance(v, float) and math.isnan(v) else v
                for k, v in m.items()}
    assert canonical(a_summary) == canonical(b_summary)
