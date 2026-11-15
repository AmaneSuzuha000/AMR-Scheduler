"""Spawned process demand against one inventory owner; no physical claim."""
import multiprocessing
from queue import Empty
import threading
import time

import pytest

from amr_sched.checkpoint import capture, definition, restore
from tools.wheelset_scene.inventory_supply import InventorySupply
from tools.wheelset_scene.inventory_supply_ipc import inventory_process_channel
from tools.wheelset_scene.material_supply import SuppliedRuntime
from tools.wheelset_scene.resource_runtime import Place
from Tests.test_delivery_inventory import at_destination, configured
from Tests.test_external_handling import receipt


def service_child(client, ready, messages):
    """Use the existing process runtime through a real spawned pipe endpoint."""
    try:
        process = SuppliedRuntime([Place('inspection', 1, (0., 0., 0.), True),
            Place('out', 1, (1., 0., 0.))], supply=client, run_id='process')
        assert process.admit('wheelset-1', 3.)
        assert not process.dispatch(3.)
        messages.put(dict(phase='waiting', stock=client.stock))
        assert ready.wait(5.)
        action, = process.dispatch(3.1)
        assert not process.observe(action['id'], now=4., observed_at=4., success=False)
        assert process.confirm_stopped(action['id'], now=5., observed_at=5., stopped=True)
        replacement = process.recover(action['id'], 5.)
        assert process.observe(replacement, now=6., observed_at=6., device_complete=True)
        view = client.stock; view['inspection'] = 99
        messages.put(dict(phase='finished', stock=client.stock, consumed=client.consumed,
                          action=action['id'], replacement=replacement))
    except BaseException as error:
        messages.put(dict(error=repr(error)))
        raise
    finally:
        client.close()


def restored_child(client, messages):
    try:
        assert client.can_start('wheelset-1', 'inspection', 4.)
        assert not client.can_start('wheelset-2', 'inspection', 4.)
        try:
            client.consume('wheelset-1', 'inspection', 'replacement', 4.)
        except ValueError:
            pass
        else:
            raise AssertionError('restored job charged twice')
        messages.put(dict(stock=client.stock, consumed=client.consumed))
    finally:
        client.close()


def pump(owner, process, messages, source_now):
    deadline = time.monotonic()+12.
    while time.monotonic() < deadline:
        owner.poll(source_now)
        try:
            result = messages.get_nowait()
        except Empty:
            time.sleep(.005)
            continue
        assert 'error' not in result, result
        return result
    pytest.fail('spawned process did not return its identified service result')


def finish(owner, process):
    process.join(timeout=5.)
    if process.is_alive():
        process.terminate(); process.join(timeout=5.)
    owner.close()
    assert process.exitcode == 0


def test_real_child_waits_for_accepted_unload_and_charges_on_fleet_owner_once():
    context = multiprocessing.get_context('spawn')
    sim = at_destination()
    supply = InventorySupply(sim, {'inspection': 'cell'})
    durable = []
    def persist():
        durable.append(capture(sim, definition(sim), {}))
        return True
    owner, client = inventory_process_channel(supply, context, persist=persist)
    ready = context.Event(); messages = context.Queue()
    process = context.Process(target=service_child, args=(client, ready, messages))
    process.start(); client.close()
    try:
        waiting = pump(owner, process, messages, 3.1)
        assert waiting == dict(phase='waiting', stock={'inspection': 0})
        assert not sim.inventory.committed and not sim.inventory.consumed
        assert sim.handling_feedback(receipt(sim, 3.), 3., 3.)['accepted']
        ready.set()
        result = pump(owner, process, messages, 3.1)
        assert result['phase'] == 'finished' and result['stock'] == {'inspection': 0}
        assert result['consumed'] == {('wheelset-1', 'inspection')}
        assert result['replacement'] != result['action']
        assert len(sim.inventory.committed) == len(sim.inventory.consumed) == 1
        assert [e['event'] for e in sim.inventory.events][-2:] == ['STOCK_COMMITTED', 'MATERIAL_CONSUMED']
        assert sim.inventory.events[-1]['command_id'] == result['action']
        assert len(durable) == 1
        restored = configured()
        restore(restored, durable[0], definition(restored))
        assert restored.inventory.consumed == {('wheelset-1', 'cell'): 1}
        sim.inventory.validate_snapshot(sim.inventory.snapshot())
    finally:
        finish(owner, process)


def test_new_process_reads_restored_consumption_without_recharging_material():
    context = multiprocessing.get_context('spawn')
    original = configured(stock=1)
    InventorySupply(original, {'inspection': 'cell'}).consume('wheelset-1', 'inspection', 'original-action', 3.)
    saved = capture(original, definition(original), {})
    target = configured(stock=1)
    restore(target, saved, definition(target))
    owner, client = inventory_process_channel(InventorySupply(target, {'inspection': 'cell'}), context)
    messages = context.Queue()
    process = context.Process(target=restored_child, args=(client, messages))
    process.start(); client.close()
    before = target.inventory.snapshot()
    try:
        result = pump(owner, process, messages, 4.)
        assert result == dict(stock={'inspection': 0}, consumed={('wheelset-1', 'inspection')})
        assert target.inventory.snapshot() == before
        assert any(not e['accepted'] for e in owner.events)
    finally:
        finish(owner, process)


@pytest.mark.parametrize('change', [
    {'run_id': 'old-run'}, {'channel_id': 'old-process'}, {'seq': 0},
    {'seq': True}, {'operation': 'commit'},
    {'arguments': dict(job='job', station='inspection', token='action', now=3.2)},
    {'arguments': dict(job='job', station='inspection', token='action', now=float('nan'))},
    {'arguments': dict(job='job', station='inspection', token='action', now=3., quantity=99)},
])
def test_stale_or_unauthorized_process_request_never_changes_inventory(change):
    sim = configured(stock=1)
    owner, client = inventory_process_channel(InventorySupply(sim, {'inspection': 'cell'}),
                                             multiprocessing.get_context('spawn'))
    request = dict(run_id=client.run_id, channel_id=client.channel_id, seq=1, operation='consume',
                   arguments=dict(job='job', station='inspection', token='action', now=3.))
    request.update(change)
    before = sim.inventory.snapshot()
    try:
        client.connection.send(request)
        assert owner.poll(3.) == 1
        assert client.connection.recv()['accepted'] is False
        assert sim.inventory.snapshot() == before
    finally:
        client.close(); owner.close()


def test_lost_reply_keeps_uncertain_consumption_and_disables_automatic_retry():
    sim = configured(stock=1)
    owner, client = inventory_process_channel(InventorySupply(sim, {'inspection': 'cell'}),
        multiprocessing.get_context('spawn'), timeout_s=.02)
    try:
        with pytest.raises(TimeoutError, match='uncertain'):
            client.consume('job', 'inspection', 'action', 3.)
        assert client.closed and sim.inventory.stock == {'cell': 1}
        # The queued request can still have been executed after the caller's timeout.
        owner.poll(3.)
        assert sim.inventory.stock == {'cell': 0}
        assert sim.inventory.consumed == {('job', 'cell'): 1}
        with pytest.raises(RuntimeError, match='reconcile'):
            client.consume('job', 'inspection', 'retry', 3.)
        assert len(sim.inventory.consumed) == 1
    finally:
        client.close(); owner.close()


def test_background_thread_cannot_mutate_the_fleet_owned_inventory():
    sim = configured(stock=1)
    owner, client = inventory_process_channel(InventorySupply(sim, {'inspection': 'cell'}),
                                             multiprocessing.get_context('spawn'))
    errors = []
    def wrong_thread():
        try:
            owner.poll(3.)
        except RuntimeError as error:
            errors.append(str(error))
    try:
        worker = threading.Thread(target=wrong_thread); worker.start(); worker.join(timeout=2.)
        assert not worker.is_alive() and errors and 'owner thread' in errors[0]
        assert sim.inventory.stock == {'cell': 1}
    finally:
        client.close(); owner.close()


def test_process_worker_rejects_unknown_supply_mode_before_ros_initialization():
    from tools.wheelset_scene.whole_line_process import run_worker
    with pytest.raises(ValueError, match='supply mode'):
        run_worker(None, None, None, None, None, None, supply_mode='implicit-stock')


def test_persistence_failure_never_acknowledges_consumption_or_allows_more_actions():
    sim = configured(stock=1)
    owner, client = inventory_process_channel(InventorySupply(sim, {'inspection': 'cell'}),
        multiprocessing.get_context('spawn'), persist=lambda: False)
    try:
        request = dict(run_id=client.run_id, channel_id=client.channel_id, seq=1,
            operation='consume', arguments=dict(job='job', station='inspection', token='action', now=3.))
        client.connection.send(request); owner.poll(3.)
        response = client.connection.recv()
        assert not response['accepted'] and 'uncertain' in response['error']
        assert owner.persistence_failed and sim.inventory.consumed == {('job', 'cell'): 1}
        before = sim.inventory.snapshot()
        request.update(seq=2, operation='can_start', arguments=dict(job='job', station='inspection', now=3.))
        client.connection.send(request); owner.poll(3.)
        assert not client.connection.recv()['accepted']
        assert sim.inventory.snapshot() == before
    finally:
        client.close(); owner.close()
