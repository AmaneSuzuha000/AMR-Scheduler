"""Execute one Python-owned bin delivery with gravity/contact handoff evidence."""
import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.entities import Agv, Task, TaskStatus
from amr_sched.external_execution import ExternalExecution, ExternalHandling
from amr_sched.inventory import DeliveryInventory
from amr_sched.network import Network, Node as RoadNode
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.simulation import Simulator, SimulationConfig
from .bin_handling_observer import BinHandlingObserver
from .bin_handling_world import build
from .inventory_supply import InventorySupply
from .verify_contact_conveyor import norm, stamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--port', type=int, default=11371)
    parser.add_argument('--domain', type=int, default=87)
    parser.add_argument('--line-side-inventory', action='store_true',
        help='Declare one initially full software buffer and two identified service demands')
    args = parser.parse_args()
    if os.name == 'nt':
        raise RuntimeError('ROS/Gazebo Linux environment required')
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', args.port)) == 0:
            raise RuntimeError('selected Gazebo port occupied')
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    world = build(output/'world', args.library)
    sources = {}
    root = Path(__file__).resolve().parents[2]
    for directory in ('amr_sched', 'tools/wheelset_scene'):
        for path in sorted((root/directory).rglob('*.py')):
            sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output/'source_manifest.json').write_text(json.dumps(sources, indent=2)+'\n')
    (output/'libwheelset_axis_servo.so').write_bytes(args.library.read_bytes())
    os.environ.update(ROS_DOMAIN_ID=str(args.domain), GAZEBO_MASTER_URI=f'http://127.0.0.1:{args.port}')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from gazebo_msgs.msg import ContactsState
    from gazebo_msgs.srv import GetEntityState
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
    from rosgraph_msgs.msg import Clock
    from rosidl_runtime_py.convert import message_to_ordereddict
    from .run_navigation_readiness import stop_launch_group

    rclpy.init(); node = Node('bin_handoff_contract_driver')
    network = Network()
    network.add_node(RoadNode('store', 0., 0.)); network.add_node(RoadNode('cell', 2., 0.))
    network.add_bidirectional('store', 'cell', 2.)
    gate = ExternalExecution('bin_handling_contact_fixture', 'carrier', (0., 0.))
    handling = ExternalHandling(gate, source='gazebo_bin_contact')
    inventory = (DeliveryInventory({'cell': 1},
        {'bin_delivery': dict(node_id='cell', quantity=1)}, initial={'cell': 1})
        if args.line_side_inventory else None)
    sim = Simulator(network, [Task('bin_delivery', 'store', 'cell')], [Agv('carrier', start_node='store')],
        SpaceTimeAStarPlanner(), GreedyAllocator(),
        SimulationConfig(dt=.05, load_time=.1, unload_time=.1, idle_retreat=False,
            deadlock_timeout=0, reschedule_policy=ReschedulePolicy.NONE),
        execution=gate, handlings={'carrier': handling}, inventory=inventory)
    sim.reset()
    supply = InventorySupply(sim, {'inspection': 'cell'}) if inventory is not None else None
    started = time.monotonic(); clock = 0.; latest = {}; pending = {}; next_query = 0.
    next_command = 0.; next_tick = 0.; sequence = 0; feedback_sequence = 0
    targets = [0., 0.]; phase = 'settling'; arrived_since = None; finished_since = None
    process = None; cleanup = None; error = None; passed = False; receipts = []
    full_wait_since = None; full_wait_duration = None; initial_consumed = False; delivered_consumed = False
    log = (output/'records.jsonl').open('x', encoding='utf-8')
    proof = output/'support_evidence'; proof.mkdir()

    def fallback(value):
        if isinstance(value, array): return value.tolist()
        raise TypeError(type(value).__name__)

    def record(kind, **values):
        row = dict(kind=kind, clock_s=clock, wall_s=time.monotonic()-started, **values)
        log.write(json.dumps(row, default=fallback, allow_nan=False)+'\n'); log.flush()

    def write_evidence(identity, raw):
        with (proof/(identity+'.json')).open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        record('support_evidence', evidence_id=identity)

    observer = BinHandlingObserver(write_evidence)

    def on_clock(msg):
        nonlocal clock
        clock = msg.clock.sec+msg.clock.nanosec*1e-9

    def on_contact(msg):
        latest['contact'] = message_to_ordereddict(msg)
        record('contact', message=latest['contact'])

    def on_joint(msg):
        latest['joint'] = message_to_ordereddict(msg)
        record('joint', message=latest['joint'])

    subscriptions = [node.create_subscription(Clock, '/clock', on_clock, qos_profile_sensor_data),
        node.create_subscription(ContactsState, '/bin_handling/bin_contacts', on_contact, qos_profile_sensor_data),
        node.create_subscription(JointState, '/bin_handling/drive/joint_states', on_joint, qos_profile_sensor_data)]
    publisher = node.create_publisher(Float64MultiArray, '/bin_handling/drive/servo_targets', 1)
    getter = node.create_client(GetEntityState, '/bin_handling/get_entity_state')
    names = dict(carrier='handling_shuttle::carrier', cargo='material_bin')
    try:
        with (output/'gazebo.log').open('x') as stream:
            command = ['gzserver', '--verbose', str(output/'world/handling.world'),
                       '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so']
            process = subprocess.Popen(command, env=dict(os.environ, GAZEBO_MODEL_DATABASE_URI=''),
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            record('launch', pid=process.pid, command=command)
            while time.monotonic()-started < 150.:
                if process.poll() is not None:
                    raise RuntimeError('owned Gazebo exited')
                rclpy.spin_once(node, timeout_sec=.002)
                for name in list(pending):
                    future, request = pending[name]
                    if future.done():
                        reply = future.result()
                        record('state_reply', entity=name, request=request,
                               response=message_to_ordereddict(reply) if reply else None)
                        if reply is not None and reply.success:
                            latest[name] = message_to_ordereddict(reply)
                        del pending[name]
                if clock >= next_query and getter.service_is_ready():
                    for name, entity in names.items():
                        if name not in pending:
                            request = GetEntityState.Request(); request.name = entity; request.reference_frame = 'world'
                            pending[name] = (getter.call_async(request), message_to_ordereddict(request))
                    next_query = clock+.02
                if clock >= next_command and 'joint' in latest:
                    desired = [0., .80] if phase == 'loading' else [0., 0.]
                    if phase in ('travelling', 'unloading', 'finished'):
                        desired = [2., .80]
                    if phase in ('unloading', 'finished'): desired[1] = 1.60
                    for index in range(2):
                        step = .005 if index == 0 else .004
                        targets[index] += max(-step, min(step, desired[index]-targets[index]))
                    sequence += 1
                    msg = Float64MultiArray(); msg.data = [float(sequence)]+[v for t in targets for v in (t, 1.)]
                    publisher.publish(msg); record('servo_targets', values=msg.data)
                    next_command = clock+.04
                if clock < next_tick or not all(k in latest for k in ('carrier', 'cargo', 'contact', 'joint')):
                    continue
                next_tick = clock+.04
                carrier, cargo = latest['carrier'], latest['cargo']
                robot = carrier['state']; p = robot['pose']['position']; bp = cargo['state']['pose']['position']
                if clock-stamp(carrier) > .15 or clock-stamp(cargo) > .15:
                    raise RuntimeError('independent carrier/cargo state stale')
                if abs(bp['z']-.62) > .08:
                    raise RuntimeError('bin lost supported height')
                if phase == 'settling' and clock < 2.:
                    continue
                if phase == 'settling': phase = 'loading'; record('phase', value=phase)
                event = 'POSE'
                if gate.command is not None:
                    at_goal = math.dist([p['x'], p['y']], gate.command['target_xy']) < .015
                    at_goal &= norm(robot['twist']['linear']) < .01 and norm(robot['twist']['angular']) < .01
                    arrived_since = (clock if arrived_since is None else arrived_since) if at_goal else None
                    if arrived_since is not None and clock-arrived_since >= .3: event = 'ARRIVED'
                feedback_sequence += 1
                feedback = dict(**gate.envelope(), seq=feedback_sequence, timestamp=stamp(carrier),
                                pose=[p['x'], p['y'], 0.], event=event)
                accepted = sim.execution_feedback(feedback, time.monotonic()-started)
                record('execution_feedback', message=feedback, result=accepted)
                if gate.fault: raise RuntimeError(gate.fault)
                sim.step_once()
                if supply is not None and not initial_consumed and sim.agvs[0].cur_node == 'cell':
                    if (sim.tasks[0].status is not TaskStatus.PICKED or inventory.stock['cell'] != 1
                            or inventory.reservations or handling.command['operation'] != 'LOAD'
                            or gate.command is not None or sim._res.pins()['nodes']['cell'] != ['carrier']):
                        raise RuntimeError('full buffer failed to retain loaded task and dock ownership')
                    if full_wait_since is None: full_wait_since = clock
                    record('inventory_wait', stock=dict(inventory.stock), pins=sim._res.pins(),
                           duration_s=clock-full_wait_since)
                    if clock-full_wait_since >= 3.:
                        supply.consume('initial_service', 'inspection', 'initial_service_action', clock)
                        initial_consumed = True; full_wait_duration = clock-full_wait_since
                        record('service_consumption', job='initial_service', action='initial_service_action',
                               stock=dict(inventory.stock), events=inventory.events)
                command = handling.command
                if command and not handling.completed:
                    if command['operation'] == 'UNLOAD' and phase != 'unloading':
                        phase = 'unloading'; record('phase', value=phase)
                    receipt = observer.observe(command, source_now=clock, carrier=carrier,
                                               cargo=cargo, contact=latest['contact'])
                    if receipt is not None:
                        result = sim.handling_feedback(receipt, time.monotonic()-started, clock)
                        record('handling_feedback', message=receipt, result=result)
                        if not result['accepted']: raise RuntimeError(str(result))
                        receipts.append(receipt)
                        if command['operation'] == 'LOAD': phase = 'travelling'; record('phase', value=phase)
                        elif supply is not None:
                            if not initial_consumed or inventory.stock['cell'] != 1:
                                raise RuntimeError('accepted physical unload did not commit exactly one buffer unit')
                            supply.consume('delivered_service', 'inspection', 'delivered_service_action', clock)
                            delivered_consumed = True
                            record('service_consumption', job='delivered_service', action='delivered_service_action',
                                   stock=dict(inventory.stock), events=inventory.events)
                record('scheduler', task_status=sim.tasks[0].status.value, pins=sim._res.pins(),
                       handling_id=command['handling_id'] if command else None, phase=phase)
                if sim.tasks[0].status is TaskStatus.COMPLETED:
                    if finished_since is None:
                        finished_since = clock; phase = 'finished'; record('phase', value=phase)
                    if clock-finished_since >= 1.:
                        if len(receipts) != 2: raise RuntimeError('two distinct transfer receipts required')
                        duplicate = sim.handling_feedback(receipts[0], time.monotonic()-started, clock)
                        record('old_load_replay', result=duplicate)
                        if duplicate['accepted']: raise RuntimeError('old load accepted after delivery')
                        if supply is not None:
                            if not delivered_consumed or inventory.stock != {'cell': 0}:
                                raise RuntimeError('delivered service did not consume from shared receiving stock')
                            inventory.validate_snapshot(inventory.snapshot())
                            record('inventory_final', stock=dict(inventory.stock), events=inventory.events,
                                   committed=inventory.committed, reservations=inventory.reservations)
                        passed = True; break
            else:
                raise TimeoutError('bin handoff fixture wall timeout')
    except Exception as exc:
        error = repr(exc); record('runtime_error', error=error)
    finally:
        if process is not None:
            cleanup = stop_launch_group(process)
        node.destroy_node(); rclpy.shutdown(); log.close()
    unchanged = all(hashlib.sha256((root/name).read_bytes()).hexdigest() == digest for name, digest in sources.items())
    result = dict(passed=passed and error is None and unchanged and cleanup is not None and cleanup['passed'],
        runtime_error=error, source_unchanged=unchanged, cleanup=cleanup, world_manifest=world,
        run_id=gate.identity['run_id'], receipts=receipts, elapsed_wall_s=time.monotonic()-started,
        task_status=sim.tasks[0].status.value, pins=sim._res.pins(),
        inventory_case=(dict(definition=inventory.definition(), stock=dict(inventory.stock),
            events=inventory.events, full_wait_s=full_wait_duration,
            service_demands='software services; initial stock is a declared buffer unit, not physical shelf occupancy')
            if inventory is not None else None),
        claim_boundary='One linear-carrier bin handoff. Not SCOUT/Nav2, full workshop or fault/restart physical acceptance.',
        files_sha256={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in output.rglob('*') if p.is_file()})
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
