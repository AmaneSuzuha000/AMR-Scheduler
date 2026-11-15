"""Replay raw contact evidence and the shared scheduler for a bin handoff run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.entities import Agv, Task, TaskStatus
from amr_sched.external_execution import ExternalExecution, ExternalHandling
from amr_sched.inventory import DeliveryInventory
from amr_sched.network import Network, Node
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.simulation import Simulator, SimulationConfig
from .bin_handling_observer import BinHandlingObserver
from .inventory_supply import InventorySupply
from .verify_contact_conveyor import digest, norm, stamp


def audit(directory):
    directory = Path(directory).resolve()
    result = json.loads((directory/'result.json').read_bytes())
    assert result['passed'] and result['runtime_error'] is None, 'physical runner did not pass'
    assert result['source_unchanged'] and result['cleanup']['passed'], 'source/cleanup failed'
    for name, expected in result['files_sha256'].items():
        path = (directory/name).resolve()
        assert path.is_relative_to(directory) and path.is_file() and digest(path) == expected, 'file hash: '+name
    world = ET.parse(directory/'world/handling.world').getroot().find('world')
    assert world.findtext('gravity') == '0 0 -9.81'
    cargo = world.find("model[@name='material_bin']")
    assert cargo.findtext('link/gravity') == 'true' and not cargo.findall('joint') and not cargo.findall('plugin')
    assert float(cargo.findtext('link/inertial/mass')) == 2.
    assert {j.findtext('child') for j in world.findall('model/joint')} == {
        'base', 'carrier', 'transfer_pusher'}
    assert all('material_bin' not in p.get('filename', '') for p in world.findall('.//plugin'))
    network = Network()
    network.add_node(Node('store', 0., 0.)); network.add_node(Node('cell', 2., 0.))
    network.add_bidirectional('store', 'cell', 2.)
    gate = ExternalExecution('bin_handling_contact_fixture', 'carrier', (0., 0.), run_id=result['run_id'])
    handling = ExternalHandling(gate, source='gazebo_bin_contact')
    inventory_case = result.get('inventory_case')
    inventory = DeliveryInventory(**inventory_case['definition']) if inventory_case is not None else None
    if inventory is not None:
        assert inventory.definition() == dict(capacities={'cell': 1},
            deliveries={'bin_delivery': dict(node_id='cell', quantity=1)}, initial={'cell': 1})
    sim = Simulator(network, [Task('bin_delivery', 'store', 'cell')], [Agv('carrier', start_node='store')],
        SpaceTimeAStarPlanner(), GreedyAllocator(),
        SimulationConfig(dt=.05, load_time=.1, unload_time=.1, idle_retreat=False,
            deadlock_timeout=0, reschedule_policy=ReschedulePolicy.NONE),
        execution=gate, handlings={'carrier': handling}, inventory=inventory)
    sim.reset()
    supply = InventorySupply(sim, {'inspection': 'cell'}) if inventory is not None else None
    latest = {}; emitted = []; expected_receipt = None; receipts = []
    feedback_count = carried_count = snapshots = 0; maximum_cargo_offset = 0.
    last_sequence = -1; loaded = False; old_load_rejected = False
    full_wait_start = None; full_wait_last = None; service_jobs = []; inventory_final = False

    def check_evidence(identity, raw):
        assert hashlib.sha256(raw).hexdigest() == identity
        assert (directory/'support_evidence'/(identity+'.json')).read_bytes() == raw, 'support proof differs from raw replay'
        emitted.append(identity)

    observer = BinHandlingObserver(check_evidence)
    with (directory/'records.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line); kind = row['kind']
            if kind == 'runtime_error': raise AssertionError(row['error'])
            if kind == 'contact': latest['contact'] = row['message']
            elif kind == 'state_reply' and row['response'] is not None and row['response']['success']:
                assert row['entity'] in ('carrier', 'cargo')
                assert row['request'] == dict(name='handling_shuttle::carrier' if row['entity'] == 'carrier' else 'material_bin',
                                              reference_frame='world'), 'wrong independent state query'
                latest[row['entity']] = row['response']
            elif kind == 'servo_targets':
                values = row['values']
                assert len(values) == 5 and all(math.isfinite(v) for v in values)
                assert values[0] > last_sequence and values[2::2] == [1., 1.]
                assert all(0 <= v <= bound for v, bound in zip(values[1::2], (2., 1.6)))
                last_sequence = values[0]
            elif kind == 'execution_feedback':
                assert set(latest) == {'contact', 'carrier', 'cargo'}
                carrier = latest['carrier']; material = latest['cargo']
                p = carrier['state']['pose']['position']; bp = material['state']['pose']['position']
                message = row['message']
                assert message['pose'] == [p['x'], p['y'], 0.] and message['timestamp'] == stamp(carrier), 'feedback differs from actual carrier'
                assert abs(bp['z']-.62) < .08, 'bin left supported height'
                if message['event'] == 'ARRIVED':
                    assert gate.command is not None and math.dist([p['x'], p['y']], gate.command['target_xy']) < .015
                    assert all(norm(carrier['state']['twist'][axis]) < .01 for axis in ('linear', 'angular'))
                accepted = sim.execution_feedback(message, row['wall_s'])
                assert accepted == row['result'] and accepted['accepted'] and not gate.fault, 'execution replay differs'
                sim.step_once(); feedback_count += 1
                if loaded and (handling.command is None or handling.command['operation'] == 'LOAD'):
                    offset = math.dist([p['x'], p['y']], [bp['x'], bp['y']])
                    maximum_cargo_offset = max(maximum_cargo_offset, offset)
                    assert offset < .04 and abs(bp['z']-.62) < .015, 'loaded bin left carrier'
                    assert row['clock_s']-stamp(latest['contact']) <= .15, 'carried contact stale'
                    assert any({s['collision1_name'],s['collision2_name']} == {
                        'material_bin::body::bin','handling_shuttle::carrier::deck'} for s in latest['contact']['states']), 'no carried-bin deck contact'
                    carried_count += 1
                if handling.command and not handling.completed:
                    expected_receipt = observer.observe(handling.command, source_now=row['clock_s'],
                        carrier=carrier, cargo=material, contact=latest['contact'])
            elif kind == 'support_evidence':
                assert emitted and row['evidence_id'] == emitted[-1], 'unreplayed support evidence'
            elif kind == 'handling_feedback':
                assert expected_receipt is not None and row['message'] == expected_receipt, 'receipt lacks independent support replay'
                if supply is not None and row['message']['operation'] == 'UNLOAD':
                    assert service_jobs == ['initial_service'] and inventory.stock == {'cell': 0}
                accepted = sim.handling_feedback(row['message'], row['wall_s'], row['clock_s'])
                assert accepted == row['result'] and accepted['accepted']
                receipts.append(row['message']); loaded = row['message']['operation'] == 'LOAD'
                expected_receipt = None
            elif kind == 'inventory_wait':
                assert supply is not None and service_jobs == [] and loaded
                assert sim.tasks[0].status is TaskStatus.PICKED and sim.agvs[0].cur_node == 'cell'
                assert gate.command is None and handling.command['operation'] == 'LOAD'
                assert inventory.stock == row['stock'] == {'cell': 1} and not inventory.reservations
                assert sim._res.pins() == row['pins'] and row['pins']['nodes']['cell'] == ['carrier']
                if full_wait_start is None: full_wait_start = row['clock_s']
                assert abs(row['duration_s']-(row['clock_s']-full_wait_start)) < 1e-9
                full_wait_last = row['duration_s']
            elif kind == 'service_consumption':
                assert supply is not None
                expected = ['initial_service', 'delivered_service'][len(service_jobs)]
                assert row['job'] == expected and row['action'] == expected+'_action'
                if not service_jobs:
                    assert full_wait_last is not None and full_wait_last >= 3. and not inventory.committed
                else:
                    assert [r['operation'] for r in receipts] == ['LOAD', 'UNLOAD']
                    assert inventory.stock == {'cell': 1} and not inventory.reservations
                supply.consume(row['job'], 'inspection', row['action'], row['clock_s'])
                assert inventory.stock == row['stock'] and inventory.events == row['events']
                inventory.validate_snapshot(inventory.snapshot()); service_jobs.append(row['job'])
            elif kind == 'inventory_final':
                assert supply is not None and service_jobs == ['initial_service', 'delivered_service']
                assert inventory.stock == row['stock'] == {'cell': 0}
                assert inventory.events == row['events'] and inventory.committed == row['committed']
                assert inventory.reservations == row['reservations'] == {}
                inventory.validate_snapshot(inventory.snapshot()); inventory_final = True
            elif kind == 'scheduler':
                assert sim.tasks[0].status.value == row['task_status'] and sim._res.pins() == row['pins'], 'scheduler ownership differs'
                assert row['handling_id'] == (handling.command['handling_id'] if handling.command else None)
                snapshots += 1
            elif kind == 'old_load_replay':
                assert receipts
                accepted = sim.handling_feedback(receipts[0], row['wall_s'], row['clock_s'])
                assert accepted == row['result'] and not accepted['accepted']
                old_load_rejected = True
    assert receipts == result['receipts'] and [r['operation'] for r in receipts] == ['LOAD', 'UNLOAD']
    assert len(emitted) == 2 and len({r['handling_id'] for r in receipts}) == 2
    assert sim.tasks[0].status is TaskStatus.COMPLETED and sim._res.pins() == result['pins']
    assert old_load_rejected and carried_count > 20 and snapshots == feedback_count
    if inventory is not None:
        assert inventory_final and len(inventory.committed) == 1 and len(inventory.consumed) == 2
        assert inventory.stock == inventory_case['stock'] and inventory.events == inventory_case['events']
        assert full_wait_last == inventory_case['full_wait_s']
    return dict(passed=True, scope='isolated_linear_carrier_bin_handoff',
        receipt_operations=[r['operation'] for r in receipts], scheduler_snapshots=snapshots,
        carried_observations=carried_count, maximum_carried_bin_offset_m=maximum_cargo_offset,
        physical_passed=True, sensor_navigation_validated=False, workshop_delivery_validated=False,
        finite_inventory_replayed=inventory is not None,
        software_full_buffer_wait_s=full_wait_last,
        physical_machine_consumption_validated=False,
        result_sha256=digest(directory/'result.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.directory)
    with args.output.open('x', encoding='utf-8') as stream: json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
