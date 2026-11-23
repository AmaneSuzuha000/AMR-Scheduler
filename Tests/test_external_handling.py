"""Measured transfer receipts, rather than elapsed service times, own cargo."""
from copy import deepcopy

import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.entities import Agv, Task, TaskPriority, TaskStatus
from amr_sched.external_execution import ExternalExecution, ExternalHandling
from amr_sched.network import Network, Node
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.simulation import Simulator, SimulationConfig


def build():
    net = Network()
    net.add_node(Node('store', 0., 0.)); net.add_node(Node('cell', 2., 0.))
    net.add_bidirectional('store', 'cell', 2.)
    gate = ExternalExecution('repair', 'R', (0., 0.), run_id='handling-test')
    handling = ExternalHandling(gate, source='synthetic_contract_test')
    sim = Simulator(net, [Task('bin', 'store', 'cell')], [Agv('R', start_node='store')],
        SpaceTimeAStarPlanner(), GreedyAllocator(),
        SimulationConfig(load_time=1, unload_time=1, idle_retreat=False,
                         deadlock_timeout=0, reschedule_policy=ReschedulePolicy.NONE),
        execution=gate, handlings={'R': handling})
    sim.source_offset = 0.
    sim.reset()
    return sim


def heartbeat(sim, now, event='POSE', xy=None):
    gate = sim.execution
    xy = xy or (gate.pose['x'], gate.pose['y']) if gate.pose else xy or gate.start_xy
    msg = dict(**gate.envelope(), event=event, pose=[*xy, 0.],
               seq=gate.last_seq+1, timestamp=now+getattr(sim, 'source_offset', 0.))
    assert sim.execution_feedback(msg, now)['accepted']


def waiting(source_offset=0.):
    sim = build()
    sim.source_offset = source_offset
    heartbeat(sim, 0.)
    sim.step_once()
    assert sim.handlings['R'].command['operation'] == 'LOAD'
    assert sim.handlings['R'].command['issued_at'] == source_offset
    return sim


def receipt(sim, now):
    command = sim.handlings['R'].command
    return dict(**deepcopy(command), event='HANDLING_COMPLETED', timestamp=now+sim.source_offset,
        robot_xy=list(command['target_xy']), stopped=True,
        supported_by='R' if command['operation'] == 'LOAD' else command['node_id'],
        evidence_id='synthetic-support-observation')


def test_waiting_for_transfer_preserves_task_dock_and_preemption_eligibility():
    sim = waiting()
    for now in range(1, 20):
        heartbeat(sim, float(now)); sim.step_once()
    assert sim.tasks[0].status is TaskStatus.ASSIGNED
    assert sim.tasks[0].pickup_step is None and sim.execution.command is None
    assert sim._res.pins() == dict(nodes={'store': ['R']}, edges={})
    assert len(sim.handlings['R'].events) == 1
    urgent = Task('urgent', 'store', 'cell', release_step=sim._step_cursor, priority=TaskPriority.URGENT)
    assert sim.submit_physical_task(urgent)['accepted']
    assert not sim.request_physical_preemption('urgent', 19.)['accepted']
    assert sim.tasks[0].assigned_agv == 'R'


@pytest.mark.parametrize('source_offset', [0., 125.])
def test_load_and_unload_require_distinct_receipts_before_task_completion(source_offset):
    sim = waiting(source_offset)
    heartbeat(sim, 1.)
    load = receipt(sim, 1.)
    assert sim.handling_feedback(load, 1., 1.+source_offset)['accepted']
    sim.step_once()
    assert sim.tasks[0].status is TaskStatus.PICKED
    assert sim.execution.command is not None
    heartbeat(sim, 2., 'ARRIVED', (2., 0.)); sim.step_once()
    heartbeat(sim, 3.); sim.step_once()
    assert sim.handlings['R'].command['operation'] == 'UNLOAD'
    for now in range(4, 10):
        heartbeat(sim, float(now)); sim.step_once()
    assert sim.tasks[0].status is TaskStatus.PICKED
    assert sim.tasks[0].complete_step is None
    assert sim._res.pins() == dict(nodes={'cell': ['R']}, edges={})
    assert not sim.handling_feedback(load, 9., 9.+source_offset)['accepted']
    unload = receipt(sim, 9.)
    assert sim.handling_feedback(unload, 9., 9.+source_offset)['accepted']
    assert not sim.handling_feedback(unload, 9., 9.+source_offset)['accepted']
    sim.step_once()
    assert sim.tasks[0].status is TaskStatus.COMPLETED
    assert sim.tasks[0].complete_step == 10
    assert sim._res.pins() == dict(nodes={'cell': ['R']}, edges={})


@pytest.mark.parametrize('corruption', ['run', 'robot', 'operation', 'handling_id', 'support',
    'evidence', 'moving', 'far_dock', 'stale_source', 'future_source', 'bad_receipt', 'fault'])
def test_incomplete_or_stale_handling_receipt_never_moves_the_load(corruption):
    sim = waiting()
    heartbeat(sim, 1.)
    message = receipt(sim, 1.)
    received = 1.
    if corruption in {'run', 'robot', 'handling_id'}:
        message[{'run':'run_id', 'robot':'robot_id', 'handling_id':'handling_id'}[corruption]] = 'old'
    if corruption == 'operation': message['operation'] = 'UNLOAD'
    if corruption == 'support': message['supported_by'] = 'other_robot'
    if corruption == 'evidence': message['evidence_id'] = ''
    if corruption == 'moving': message['stopped'] = False
    if corruption == 'far_dock': message['robot_xy'] = [.3, 0.]
    if corruption == 'stale_source': message['timestamp'] = .8
    if corruption == 'future_source': message['timestamp'] = 1.01
    if corruption == 'bad_receipt': received = float('nan')
    if corruption == 'fault': sim.execution.fail('lost sensor')
    assert not sim.handling_feedback(message, received, 1.)['accepted']
    assert not sim.handlings['R'].completed
    assert sim.tasks[0].status is TaskStatus.ASSIGNED
    assert sim._res.pins() == dict(nodes={'store': ['R']}, edges={})


def test_handling_does_not_bypass_vehicle_watchdog_or_reset_load_ownership():
    sim = waiting()
    assert not sim.handling_feedback(receipt(sim, 3.), 3., 3.)['accepted']
    assert sim.execution.fault == 'feedback timeout'
    assert not sim.handlings['R'].completed
    with pytest.raises(RuntimeError):
        sim.reset()


def test_handling_mapping_requires_the_exact_owned_physical_gate():
    sim = build()
    other = ExternalExecution('repair', 'R', (0., 0.), run_id='handling-test')
    with pytest.raises(ValueError):
        Simulator(sim.network, sim.tasks, sim.agvs, sim.planner, sim.allocator, sim.config,
                  execution=sim.execution, handlings={'R': ExternalHandling(other, source='equipment')})


@pytest.mark.parametrize('phase', ['load_pending', 'load_confirmed', 'loaded_moving',
                                  'unload_pending', 'unload_confirmed'])
def test_restart_keeps_material_ownership_and_requires_fresh_stationary_reconciliation(phase):
    from amr_sched.checkpoint import capture, definition, restore
    sim = waiting()
    fingerprint = definition(sim)
    if phase != 'load_pending':
        heartbeat(sim, 1.)
        assert sim.handling_feedback(receipt(sim, 1.), 1., 1.)['accepted']
    if phase in {'loaded_moving', 'unload_pending', 'unload_confirmed'}:
        sim.step_once()
    if phase in {'unload_pending', 'unload_confirmed'}:
        heartbeat(sim, 2., 'ARRIVED', (2., 0.)); sim.step_once()
        heartbeat(sim, 3.); sim.step_once()
        if phase == 'unload_confirmed':
            assert sim.handling_feedback(receipt(sim, 3.), 3., 3.)['accepted']
    before = deepcopy(sim._res.pins())
    old_receipt = receipt(sim, 3.)
    recovered = build()
    assert definition(recovered) == fingerprint
    restore(recovered, capture(sim, fingerprint, {'case':phase}), fingerprint)
    handling = recovered.handlings['R']
    assert handling.execution is recovered.execution
    assert recovered._res.pins() == before
    assert recovered.tasks[0].status is sim.tasks[0].status
    assert recovered.execution.fault == 'runtime restarted; reconciliation required'
    assert not recovered.handling_feedback(old_receipt, 3., 3.)['accepted']
    if phase == 'loaded_moving':
        assert handling.completed
        assert handling.command['handling_id'] == sim.handlings['R'].command['handling_id']
    else:
        assert not handling.completed
        assert handling.command['reconcile_only'] is True
        assert handling.command['handling_id'] != sim.handlings['R'].command['handling_id']
    recovered.step_once()
    assert recovered.tasks[0].status is sim.tasks[0].status
    assert recovered.tasks[0].complete_step is None
    if phase != 'loaded_moving':
        xy = (0., 0.) if phase.startswith('load_') else (2., 0.)
        heartbeat(recovered, 4., xy=xy)
        recovered.execution.drain_commands()
        heartbeat(recovered, 4., 'STOPPED', xy)
        assert recovered.resume_execution(recovered.execution.recovery_token(), 4.)['accepted']
        assert not recovered.handling_feedback(old_receipt, 4., 4.)['accepted']
        assert recovered.handling_feedback(receipt(recovered, 4.), 4., 4.)['accepted']
        for now in range(5, 8):
            heartbeat(recovered, float(now)); recovered.step_once()
        if phase.startswith('load_'):
            assert recovered.tasks[0].status is TaskStatus.PICKED
            assert recovered.execution.command is not None
        else:
            assert recovered.tasks[0].status is TaskStatus.COMPLETED


@pytest.mark.parametrize('next_operation', ['UNLOAD', 'LOAD'])
def test_restart_between_arrival_and_new_handling_request_keeps_prior_receipt_historical(next_operation):
    from amr_sched.checkpoint import capture, definition, restore

    sim = waiting()
    following = Task('next_bin', 'store', 'cell', release_step=1)
    if next_operation == 'LOAD':
        assert sim.submit_physical_task(following)['accepted']
    heartbeat(sim, 1.)
    assert sim.handling_feedback(receipt(sim, 1.), 1., 1.)['accepted']
    sim.step_once()
    heartbeat(sim, 2., 'ARRIVED', (2., 0.)); sim.step_once()
    if next_operation == 'LOAD':
        heartbeat(sim, 3.); sim.step_once()
        heartbeat(sim, 4.)
        assert sim.handling_feedback(receipt(sim, 4.), 4., 4.)['accepted']
        sim.step_once()
        heartbeat(sim, 5.); sim.step_once()
        heartbeat(sim, 6., 'ARRIVED', (0., 0.)); sim.step_once()
    assert not sim._transit
    prior = deepcopy(sim.handlings['R'].command)
    assert sim.handlings['R'].completed
    fingerprint = definition(sim)
    recovered = build()
    if next_operation == 'LOAD':
        assert recovered.submit_physical_task(following)['accepted']
    assert definition(recovered) == fingerprint
    restore(recovered, capture(sim, fingerprint, {}), fingerprint)
    before = deepcopy(sim._res.pins())
    xy = (0., 0.) if next_operation == 'LOAD' else (2., 0.)
    heartbeat(recovered, 7., xy=xy)
    recovered.execution.drain_commands()
    heartbeat(recovered, 7., 'STOPPED', xy)
    assert recovered.resume_execution(recovered.execution.recovery_token(), 7.)['accepted']
    # Arrival and request issuance occur on separate owner ticks. A completed
    # receipt from the previous dock must not become an unsettled new transfer.
    recovered.step_once()
    command = recovered.handlings['R'].command
    assert command['operation'] == next_operation
    assert command['task_id'] == ('next_bin' if next_operation == 'LOAD' else 'bin')
    assert command['target_xy'] == list(xy)
    assert command['handling_id'] != prior['handling_id']
    assert not command.get('reconcile_only', False)
    assert not recovered.handlings['R'].completed
    assert recovered._res.pins() == before
    old = dict(prior, event='HANDLING_COMPLETED', timestamp=7., robot_xy=list(xy),
               stopped=True, supported_by='R', evidence_id='prior-transfer')
    assert not recovered.handling_feedback(old, 7., 7.)['accepted']
    assert recovered.handling_feedback(receipt(recovered, 7.), 7., 7.)['accepted']
    heartbeat(recovered, 8.); recovered.step_once()
    current = recovered._task_by_id[command['task_id']]
    expected = TaskStatus.PICKED if next_operation == 'LOAD' else TaskStatus.COMPLETED
    assert current.status is expected
