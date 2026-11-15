"""Two workshop deliveries and process demands share one finite stock ledger.

Motion and handling observations here are synthetic software-contract inputs.
These tests do not certify physical contact, navigation or machine service.
"""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from amr_sched.checkpoint import capture, definition, restore
from amr_sched.entities import TaskStatus
from tools.wheelset_scene.inventory_supply import InventorySupply
from tools.wheelset_scene.material_supply import SuppliedRuntime
from tools.wheelset_scene.navigation_fleet import build_line_side_fleet, build_sensor_fleet
from tools.wheelset_scene.navigation_workshop import LocalizedWorkshop
from tools.wheelset_scene.resource_runtime import Place
from Tests.test_delivery_inventory import at_destination, configured
from Tests.test_external_handling import receipt


@pytest.mark.parametrize('full', [False, True])
def test_workshop_two_delivery_process_chain_waits_for_capacity_and_accepted_unload(full):
    initial = {'receiving': 4, 'supply': 2} if full else None
    sim, supply = build_line_side_fleet(run_id='line-side-contract',
        handling_source='synthetic_contract_test', initial=initial)
    processes = {station: SuppliedRuntime([
        Place(station, 1, (0., 0., 0.), True), Place('out', 1, (1., 0., 0.))],
        supply=supply, run_id='service-'+station) for station in ('magnetic', 'derust')}
    for station, process in processes.items():
        assert process.admit('service-'+station, 0.)
    positions = {r: g.start_xy for r, g in sim.executions.items()}
    motions = {}; cleared = set(); started = set(); waiting = set(); loads = set()
    receipts = set(); finished = set()
    for step in range(6000):
        now = step*.2
        for robot, gate in sim.executions.items():
            xy = positions[robot]; event = 'POSE'; move = gate.command
            if move:
                action = motions.setdefault(move['route_id'], dict(start=now, source=xy,
                    duration=math.dist(xy, move['target_xy'])/move['speed']*1.2+.2))
                fraction = min(1., (now-action['start'])/action['duration'])
                xy = tuple(a+(b-a)*fraction for a, b in zip(action['source'], move['target_xy']))
                positions[robot] = xy
                if fraction == 1.: event = 'ARRIVED'
            assert sim.execution_feedback({**gate.envelope(), 'event': event,
                'pose': [*xy, 0.], 'seq': gate.last_seq+1, 'timestamp': now}, now)['accepted']
        for robot, handling in sim.handlings.items():
            command = handling.command
            if command is None or handling.completed:
                continue
            proof = {**deepcopy(command), 'event': 'HANDLING_COMPLETED', 'timestamp': now,
                'robot_xy': list(command['target_xy']), 'stopped': True,
                'supported_by': robot if command['operation'] == 'LOAD' else command['node_id'],
                'evidence_id': 'synthetic-'+command['handling_id']}
            if command['operation'] == 'UNLOAD':
                station = {'receiving': 'magnetic', 'supply': 'derust'}[command['node_id']]
                assert supply.stock[station] == 0
                assert not processes[station].dispatch(now)
                old = {**proof, 'timestamp': now-1.}
                before = sim.inventory.snapshot()
                assert not sim.handling_feedback(old, now, now)['accepted']
                assert sim.inventory.snapshot() == before
                assert sim._task_by_id[command['task_id']].status is TaskStatus.PICKED
                receipts.add(station)
            assert sim.handling_feedback(proof, now, now)['accepted']
            if command['operation'] == 'LOAD': loads.add(command['task_id'])
        sim.step_once()
        for station, process in processes.items():
            node = supply.station_buffers[station]
            if full and station not in cleared:
                delivery = next(t for t in sim.tasks if t.dropoff == node)
                agv = next(a for a in sim.agvs if a.agv_id == delivery.assigned_agv)
                handling = sim.handlings[agv.agv_id]
                if agv.cur_node != node or delivery.status is not TaskStatus.PICKED:
                    continue
                if handling.command['operation'] == 'UNLOAD':
                    pytest.fail('full receiving buffer authorized unload before consumption')
                waiting.add(station)
                remaining = supply.stock[station]
                assert remaining > 0 and delivery.task_id not in sim.inventory.reservations
                assert sim._res.pins()['nodes'][node] == [agv.agv_id]
                supply.consume(f'previous-service-{station}-{remaining}', station,
                               f'previous-action-{station}-{remaining}', now)
                if supply.stock[station] > 0:
                    continue
                cleared.add(station)
                assert supply.stock[station] == 0
            if station not in started:
                actions = process.dispatch(now)
                if actions:
                    action, = actions
                    assert station in receipts and action['kind'] == 'device'
                    assert (action['job'], station) in supply.consumed
                    assert sim.inventory.events[-1]['command_id'] == action['id']
                    assert process.observe(action['id'], now=now, observed_at=now, device_complete=True)
                    started.add(station)
            if station in started and station not in finished:
                actions = process.dispatch(now)
                if actions:
                    action, = actions
                    assert action['kind'] == 'transfer'
                    assert process.observe(action['id'], now=now, observed_at=now, pose=(1., 0., 0.))
                    assert process.release_outfeed('service-'+station, now)
                    finished.add(station)
        sim.inventory.validate_snapshot(sim.inventory.snapshot())
        assert all(len(owners) == 1 for pins in sim._res.pins().values() for owners in pins.values())
        assert not any(g.fault for g in sim.executions.values())
        if sim._all_done() and len(finished) == 2:
            break
    assert sim._all_done() and len(loads) == 2 and receipts == started == finished == set(processes)
    assert waiting == (set(processes) if full else set())
    assert supply.stock == {'magnetic': 3, 'derust': 1}
    assert len(sim.inventory.committed) == 2


def test_process_adapter_observes_fleet_restore_and_cannot_charge_restored_job_twice():
    original = at_destination()
    assert original.handling_feedback(receipt(original, 3.), 3., 3.)['accepted']
    supply = InventorySupply(original, {'inspection': 'cell'})
    supply.consume('wheelset-1', 'inspection', 'process-action-1', 3.)
    saved = capture(original, definition(original), {})
    target = configured()
    rebound = InventorySupply(target, {'inspection': 'cell'})
    restore(target, saved, definition(target))
    assert rebound.stock == {'inspection': 0}
    assert rebound.consumed == {('wheelset-1', 'inspection')}
    assert rebound.can_start('wheelset-1', 'inspection', 4.)
    assert not rebound.can_start('wheelset-2', 'inspection', 4.)
    before = target.inventory.snapshot()
    with pytest.raises(ValueError, match='already charged'):
        rebound.consume('wheelset-1', 'inspection', 'replacement-action', 4.)
    assert target.inventory.snapshot() == before
    assert target.inventory.events[-1]['command_id'] == 'process-action-1'


def test_process_material_charge_and_fault_recovery_share_the_original_inventory():
    sim = configured(stock=1)
    supply = InventorySupply(sim, {'inspection': 'cell'})
    process = SuppliedRuntime([Place('inspection', 1, (0., 0., 0.), True),
        Place('out', 1, (1., 0., 0.))], supply=supply, run_id='process')
    assert process.admit('wheelset-1', 0.)
    action, = process.dispatch(0.)
    assert sim.inventory.stock['cell'] == 0
    assert not process.observe(action['id'], now=1., observed_at=1., success=False)
    assert process.confirm_stopped(action['id'], now=2., observed_at=2., stopped=True)
    replacement = process.recover(action['id'], 2.)
    assert process.observe(replacement, now=3., observed_at=3., device_complete=True)
    assert len(sim.inventory.consumed) == 1 and sim.inventory.stock['cell'] == 0
    view = supply.stock; view['inspection'] = 99
    assert supply.stock == {'inspection': 0}


@pytest.mark.parametrize('mapping', [{}, {'inspection': 'unknown'}, {'a': 'cell', 'b': 'cell'}])
def test_process_adapter_rejects_ambiguous_or_untracked_buffer_mapping(mapping):
    with pytest.raises(ValueError, match='distinct tracked'):
        InventorySupply(configured(), mapping)


def test_finite_workshop_rejects_missing_handling_and_legacy_second_inventory_before_ros_start():
    with pytest.raises(ValueError, match='measured handling'):
        build_line_side_fleet(run_id='invalid', handling_source=None)
    sim, _ = build_line_side_fleet(run_id='explicit', handling_source='synthetic_contract_test')
    with pytest.raises(ValueError, match='cannot duplicate'):
        LocalizedWorkshop(SimpleNamespace(sim=sim), None, None, None, None, None)
    assert build_sensor_fleet(run_id='historical').inventory is None


@pytest.mark.parametrize('job,token,now', [('', 'action', 1.), ('job', '', 1.), ('job', 'action', float('nan'))])
def test_invalid_service_request_never_changes_stock(job, token, now):
    sim = configured(stock=1); supply = InventorySupply(sim, {'inspection': 'cell'})
    before = sim.inventory.snapshot()
    with pytest.raises(ValueError):
        supply.consume(job, 'inspection', token, now)
    assert sim.inventory.snapshot() == before
