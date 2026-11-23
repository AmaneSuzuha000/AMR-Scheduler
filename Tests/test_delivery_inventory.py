"""Receiving capacity and task completion share the physical owner and checkpoint."""
from copy import deepcopy

import pytest

from amr_sched.checkpoint import capture, definition, restore, _decode, _encode
from amr_sched.entities import Task, TaskStatus
from amr_sched.inventory import DeliveryInventory
from amr_sched.simulation import Simulator
from Tests.test_external_handling import build, heartbeat, receipt


def configured(*, stock=0, capacity=1):
    original = build()
    inventory = DeliveryInventory({'cell': capacity},
        {'bin': dict(node_id='cell', quantity=1)}, initial={'cell': stock})
    sim = Simulator(original.network, original.tasks, original.agvs,
        original.planner, original.allocator, original.config, execution=original.execution,
        handlings=original.handlings, inventory=inventory)
    sim.source_offset = 0.
    sim.reset()
    return sim


def at_destination(*, stock=0):
    sim = configured(stock=stock)
    heartbeat(sim, 0.); sim.step_once()
    heartbeat(sim, 1.)
    assert sim.handling_feedback(receipt(sim, 1.), 1., 1.)['accepted']
    sim.step_once()
    heartbeat(sim, 2., 'ARRIVED', (2., 0.)); sim.step_once()
    heartbeat(sim, 3.); sim.step_once()
    assert sim.tasks[0].status is TaskStatus.PICKED
    return sim


def test_full_buffer_never_issues_unload_and_keeps_cargo_and_road_pin():
    sim = at_destination(stock=1)
    inventory = sim.inventory
    assert sim.handlings['R'].command['operation'] == 'LOAD'
    for now in range(4, 8):
        heartbeat(sim, float(now)); sim.step_once()
    assert sim.tasks[0].status is TaskStatus.PICKED
    assert sim._res.pins() == dict(nodes={'cell': ['R']}, edges={})
    assert inventory.stock == {'cell': 1} and not inventory.reservations
    assert inventory.consume('service-previous-bin', 'cell', 7.)
    assert not inventory.consume('service-previous-bin', 'cell', 7.)
    heartbeat(sim, 8.); sim.step_once()
    assert sim.handlings['R'].command['operation'] == 'UNLOAD'
    assert inventory.reservations == {'bin': dict(robot_id='R', node_id='cell', quantity=1)}
    assert inventory.stock == {'cell': 0}
    assert sim.handling_feedback(receipt(sim, 8.), 8., 8.)['accepted']
    assert inventory.stock == {'cell': 1}
    assert sim.tasks[0].status is TaskStatus.PICKED
    sim.step_once()
    assert sim.tasks[0].status is TaskStatus.COMPLETED
    assert sim._res.pins() == dict(nodes={'cell': ['R']}, edges={})
    assert inventory.consume('new-service', 'cell', 8.)
    assert not inventory.consume('missing-material', 'cell', 8.)
    inventory.validate_snapshot(inventory.snapshot())


def test_two_vehicles_cannot_reserve_the_same_remaining_receiving_unit():
    inventory = DeliveryInventory({'cell': 2}, {
        'a': dict(node_id='cell', quantity=1), 'b': dict(node_id='cell', quantity=1)},
        initial={'cell': 1})
    assert inventory.reserve('a', 'R1', 'cell', 0.)
    assert inventory.reserve('a', 'R1', 'cell', 0.)
    assert not inventory.reserve('a', 'R2', 'cell', 0.)
    assert not inventory.reserve('b', 'R2', 'cell', 0.)
    assert inventory.stock['cell'] == 1
    assert inventory.consume('old-service', 'cell', 1.)
    assert inventory.reserve('b', 'R2', 'cell', 1.)
    inventory.validate_snapshot(inventory.snapshot())


@pytest.mark.parametrize('wrong', ['stale', 'other_run', 'other_dock', 'wrong_support', 'no_proof'])
def test_rejected_equipment_receipt_cannot_create_stock_or_finish_a_task(wrong):
    sim = at_destination()
    msg = receipt(sim, 3.)
    if wrong == 'stale': msg['timestamp'] = 2.
    if wrong == 'other_run': msg['run_id'] = 'old'
    if wrong == 'other_dock': msg['node_id'] = 'store'
    if wrong == 'wrong_support': msg['supported_by'] = 'R'
    if wrong == 'no_proof': msg['evidence_id'] = ''
    before = sim.inventory.snapshot()
    assert not sim.handling_feedback(msg, 3., 3.)['accepted']
    assert sim.inventory.snapshot() == before
    assert not sim.handlings['R'].completed
    assert sim.tasks[0].status is TaskStatus.PICKED


@pytest.mark.parametrize('when', ['pending', 'receipt_before_task_tick', 'completed'])
def test_restart_keeps_receiving_capacity_and_committed_stock_without_double_credit(when):
    sim = at_destination()
    if when != 'pending':
        assert sim.handling_feedback(receipt(sim, 3.), 3., 3.)['accepted']
    if when == 'completed': sim.step_once()
    old = deepcopy(receipt(sim, 3.))
    fingerprint = definition(sim)
    saved = capture(sim, fingerprint, {})
    recovered = configured()
    restore(recovered, saved, definition(recovered))
    assert recovered.inventory.snapshot() == sim.inventory.snapshot()
    assert not recovered.handling_feedback(old, 4., 4.)['accepted']
    assert recovered.inventory.snapshot() == sim.inventory.snapshot()
    heartbeat(recovered, 4., xy=(2., 0.))
    recovered.execution.drain_commands()
    heartbeat(recovered, 4., 'STOPPED', (2., 0.))
    assert recovered.resume_execution(recovered.execution.recovery_token(), 4.)['accepted']
    if when != 'completed':
        message = receipt(recovered, 4.)
        message['evidence_id'] = 'fresh-after-restart-support-proof'
        assert message['reconcile_only']
        assert recovered.handling_feedback(message, 4., 4.)['accepted']
        assert not recovered.handling_feedback(message, 4., 4.)['accepted']
        recovered.step_once()
    assert recovered.tasks[0].status is TaskStatus.COMPLETED
    assert recovered.inventory.stock == {'cell': 1}
    assert not recovered.inventory.reservations
    assert len(recovered.inventory.committed) == 1
    recovered.inventory.validate_snapshot(recovered.inventory.snapshot())


def test_corrupt_stock_checkpoint_is_rejected_before_mutating_the_live_fleet():
    original = at_destination()
    saved = _decode(capture(original, definition(original), {}))
    saved['inventory']['stock']['cell'] = 1
    target = configured()
    before = capture(target, definition(target), {})
    with pytest.raises(ValueError, match='conservation'):
        restore(target, _encode(saved), definition(target))
    assert capture(target, definition(target), {}) == before


@pytest.mark.parametrize('wrong', ['vehicle', 'accepted_support_provenance'])
def test_restored_inventory_must_match_task_owner_and_accepted_equipment_history(wrong):
    original = at_destination()
    if wrong == 'accepted_support_provenance':
        assert original.handling_feedback(receipt(original, 3.), 3., 3.)['accepted']
    saved = _decode(capture(original, definition(original), {}))
    if wrong == 'vehicle':
        saved['inventory']['reservations']['bin']['robot_id'] = 'someone-else'
    else:
        saved['inventory']['committed']['bin']['evidence_id'] = 'unaccepted-proof'
    target = configured()
    before = capture(target, definition(target), {})
    with pytest.raises(ValueError, match='ownership|provenance'):
        restore(target, _encode(saved), definition(target))
    assert capture(target, definition(target), {}) == before


def test_restart_while_buffer_is_full_preserves_historical_load_and_waits_for_capacity():
    original = at_destination(stock=1)
    recovered = configured(stock=1)
    restore(recovered, capture(original, definition(original), {}), definition(recovered))
    assert recovered.handlings['R'].completed
    assert recovered.handlings['R'].command['operation'] == 'LOAD'
    heartbeat(recovered, 4., xy=(2., 0.))
    recovered.execution.drain_commands()
    heartbeat(recovered, 4., 'STOPPED', (2., 0.))
    assert recovered.resume_execution(recovered.execution.recovery_token(), 4.)['accepted']
    recovered.step_once()
    assert recovered.tasks[0].status is TaskStatus.PICKED
    assert recovered.inventory.consume('existing-service', 'cell', 4.)
    heartbeat(recovered, 5.); recovered.step_once()
    assert recovered.handlings['R'].command['operation'] == 'UNLOAD'
    assert recovered.handling_feedback(receipt(recovered, 5.), 5., 5.)['accepted']
    recovered.step_once()
    assert recovered.inventory.stock == {'cell': 1}
    assert recovered.tasks[0].status is TaskStatus.COMPLETED


def test_dynamic_delivery_to_a_tracked_buffer_requires_a_declared_recipe():
    sim = configured()
    assert not sim.submit_physical_task(Task('unregistered', 'store', 'cell'))['accepted']
    assert 'unregistered' not in sim._task_by_id
    assert len(sim.tasks) == 1


def test_inventory_requires_physical_handling_and_preserves_definition_across_recovery():
    sim = configured()
    with pytest.raises(ValueError, match='measured handling'):
        Simulator(sim.network, sim.tasks, sim.agvs, sim.planner, sim.allocator, sim.config,
                  inventory=sim.inventory)
    assert definition(sim) != definition(configured(capacity=2))
