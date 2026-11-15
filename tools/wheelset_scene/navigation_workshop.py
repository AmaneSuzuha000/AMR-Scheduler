"""Same-world process and receiving-side supply boundary for sensor navigation."""
from copy import deepcopy
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from .build import ROOT


def build_process_world(output, axis_build):
    from .whole_line_world import build_whole_line_world, whole_line_clearance
    output = Path(output); axis_build = Path(axis_build)
    metadata = build_whole_line_world(output/'whole_line')
    clearance = whole_line_clearance(metadata)
    if not clearance['passed']:
        raise RuntimeError('whole-line cargo sweep preflight failed')
    (output/'whole_line_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (output/'whole_line_clearance.json').write_text(json.dumps(clearance, indent=2)+'\n')
    native = json.loads((axis_build/'build_manifest.json').read_text())
    if not native['passed']:
        raise RuntimeError('axis servo build not validated')
    snapshots = output/'native_axis'; snapshots.mkdir()
    sources = {}
    for relative, digest in native['source_sha256'].items():
        path = ROOT/'tools/ros2/wheelset_axis_servo'/relative
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError('axis source differs from build evidence: '+relative)
        sources[str(path)] = digest
        target = snapshots/'sources'/relative; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    library = axis_build/'libwheelset_axis_servo.so'; data = library.read_bytes()
    if hashlib.sha256(data).hexdigest() != native['library_sha256']:
        raise RuntimeError('axis library differs from build evidence')
    (snapshots/library.name).write_bytes(data)
    (snapshots/'build_manifest.json').write_bytes((axis_build/'build_manifest.json').read_bytes())
    sources[str(library)] = native['library_sha256']
    return ET.parse(output/'whole_line/front.world'), sources


class LocalizedWorkshop:
    """Run isolated process services with explicit legacy or fleet-owned supply."""

    def __init__(self, runtime, executor, bundle, output, owned_launch_pid, record, *, case='normal',
                 inventory_supply=None):
        if runtime.sim.inventory is not None:
            from .inventory_supply import InventorySupply
            if not isinstance(inventory_supply, InventorySupply) or inventory_supply.sim is not runtime.sim:
                raise ValueError('legacy IPC supply cannot duplicate the fleet-owned receiving inventory; explicit same-owner process supply required')
        elif inventory_supply is not None:
            raise ValueError('process inventory requires the same fleet-owned receiving inventory')
        from .whole_line_ipc import SupplyManager, SharedSupply, run_child
        from .whole_line_world import SUPPLY_HANDOFFS
        from gazebo_msgs.srv import GetEntityState
        from rclpy.node import Node
        self.runtime = runtime; self.executor = executor; self.record = record
        self.output = Path(output); self.metadata = json.loads((Path(bundle)/'whole_line_metadata.json').read_text())
        self.context = multiprocessing.get_context('spawn')
        self.inventory_owner = None
        self.manager = None; self.manager_closed = True
        run_id = next(iter(runtime.sim.executions.values())).identity['run_id']
        if inventory_supply is None:
            self.manager = SupplyManager(ctx=self.context); self.manager.start(); self.manager_closed = False
            self.supply = SharedSupply(self.manager.MaterialSupply(run_id, initial={'derust': 0, 'magnetic': 0},
                destinations=SUPPLY_HANDOFFS, dock_source='localized'))
            process_supply = self.supply
        else:
            from .inventory_supply_ipc import inventory_process_channel
            self.supply = inventory_supply
            self.inventory_owner, process_supply = inventory_process_channel(self.supply, self.context,
                                                                              persist=runtime.persist)
        self.stop = self.context.Event(); self.queue = self.context.Queue(); self.result = {}
        self.receiver = None
        if self.inventory_owner is None:
            self.receiver = Node('wheelset_localized_receiving')
            self.getter = self.receiver.create_client(GetEntityState, '/wheelset_review/get_entity_state')
            executor.add_node(self.receiver)
        self.pending = {}; self.received = set(); self.commands = {}; self.originals = {}
        self.sequence = 0; self.last_progress = -math.inf; self.finished = False
        metadata = dict(jobs=self.metadata['jobs'], whole_line=self.metadata)
        if self.inventory_owner is not None:
            metadata['supply_mode'] = 'fleet_inventory'
        self.process = self.context.Process(target=run_child, args=(self.output/'process', process_supply,
            'stalled_recovery' if case == 'faults' else 'normal', owned_launch_pid, self.stop, metadata, self.queue))
        self.process.start()
        if self.inventory_owner is not None:
            process_supply.close()  # The spawned child owns its separate endpoint.
            return
        for robot, node in runtime.nodes.items():
            self.originals[robot] = node.emit
            original = node.emit
            def emit(kind, pose, envelope, stamp, reason=None, *, robot=robot, node=node, original=original):
                command = deepcopy(node.sim.execution.command)
                guard = node.guard_status
                answer = original(kind, pose, envelope, stamp, reason)
                if kind == 'ARRIVED' and answer['accepted'] and command['nodes'][-1] in ('supply', 'receiving'):
                    self.receive(robot, command, deepcopy(node.last_arrival), deepcopy(answer), guard)
                return answer
            node.emit = emit

    def receive(self, robot, command, feedback, answer, guard):
        from gazebo_msgs.srv import GetEntityState
        from rosidl_runtime_py.convert import message_to_ordereddict
        task = command['task_id']
        if task in self.pending or task in self.received:
            raise RuntimeError('duplicate localized receiving attempt')
        if not self.getter.service_is_ready():
            raise RuntimeError('receiving-side state service unavailable')
        request = GetEntityState.Request(); request.name = robot; request.reference_frame = 'world'
        query = dict(query_id=self.sequence, service=self.getter.srv_name, request=message_to_ordereddict(request))
        self.sequence += 1
        proof = dict(feedback=feedback, feedback_result=answer, guard=message_to_ordereddict(guard), state_query=query)
        self.record('dock_state_query', robot=robot, task_id=task, command=command, proof=proof)
        self.pending[task] = dict(future=self.getter.call_async(request), command=command, proof=proof,
                                 issued=time.monotonic())

    def poll(self, now):
        from rosidl_runtime_py.convert import message_to_ordereddict
        from .whole_line_ipc import receive_results
        from .navigation_dock import stamp
        self.poll_inventory()
        for task, pending in list(self.pending.items()):
            future = pending['future']
            if not future.done():
                if now-pending['issued'] > 2.:
                    raise RuntimeError('receiving-side state query timeout')
                continue
            reply = future.result()
            if reply is None or not reply.success:
                raise RuntimeError('receiving-side state query failed')
            proof = pending['proof']; proof['state_reply'] = message_to_ordereddict(reply)
            p, q, t = reply.state.pose.position, reply.state.pose.orientation, reply.state.twist
            observation = dict(pose=[p.x, p.y, math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))],
                sim_time_s=stamp(proof['state_reply']), linear_speed=math.hypot(t.linear.x, t.linear.y), angular_speed=abs(t.angular.z))
            station, quantity = ('magnetic', 4) if task == 'inspection_consumables' else ('derust', 2)
            source_now = max(observation['sim_time_s'], *(n.clock_s for n in self.runtime.nodes.values()))
            accepted = self.supply.arrived(task, station, quantity, pending['command'], observation,
                                         now=source_now, dock_proof=proof)
            self.record('localized_dock_receipt', task_id=task, robot=pending['command']['robot_id'],
                command=pending['command'], proof=proof, observation=observation, source_now=source_now, accepted=accepted)
            if not accepted:
                raise RuntimeError('strict localized dock receipt rejected: '+task)
            self.received.add(task); self.commands[task] = pending['command']; del self.pending[task]
        receive_results(self.queue, self.result)
        if 'error' in self.result:
            raise RuntimeError('whole-line worker: '+self.result['error'])
        if 'report' in self.result:
            if not self.result['report']['passed']:
                raise RuntimeError('whole-line worker failed: '+str(self.result['report']['runtime_error']))
            self.finished = True
        elif not self.process.is_alive():
            # Queue feeder completion can trail process termination slightly.
            self.process.join(timeout=.1); receive_results(self.queue, self.result)
            if 'report' not in self.result:
                raise RuntimeError('whole-line worker exited without a report')
        if now-self.last_progress >= 30.:
            self.record('workshop_progress', received=sorted(self.received), stock=self.supply.stock,
                        process_live=self.process.is_alive(), process_finished=self.finished)
            self.last_progress = now
        return self.finished

    def poll_inventory(self):
        """Pump process requests even while the fleet awaits localization readiness."""
        if self.inventory_owner is not None:
            clocks = [node.clock_s for node in self.runtime.nodes.values() if node.clock_s is not None]
            if clocks:
                self.inventory_owner.poll(max(clocks))

    def close(self):
        self.stop.set(); deadline = time.monotonic()+35.
        # Keep pumping the owned fleet while the process proves its final stop.
        while self.process.is_alive() and time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=.01)
            self.poll_inventory()
        if self.process.is_alive():
            raise RuntimeError('whole-line worker failed to finish its stop; evidence preserved')
        self.process.join(timeout=1.)
        from .whole_line_ipc import receive_results
        receive_results(self.queue, self.result)
        events = (self.supply.events if self.inventory_owner is None
                  else self.runtime.sim.inventory.events)
        (self.output/'supply_events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        if self.inventory_owner is None:
            report = dict(process_report=self.result.get('report'), error=self.result.get('error'),
                supply_initial=self.supply.initial, supply_final=self.supply.stock,
                supply_deliveries=self.supply.deliveries, supply_capacity=self.supply.capacity,
                supply_destinations=self.supply.destinations, received=sorted(self.received), dock_source='localized')
        else:
            inventory = self.runtime.sim.inventory
            report = dict(process_report=self.result.get('report'), error=self.result.get('error'),
                supply_mode='fleet_inventory', supply_final=self.supply.stock,
                inventory_definition=inventory.definition(), inventory_stock=dict(inventory.stock),
                inventory_reservations=deepcopy(inventory.reservations),
                inventory_committed=deepcopy(inventory.committed),
                inventory_consumed=[dict(job_id=job, node_id=node, quantity=quantity)
                    for (job, node), quantity in sorted(inventory.consumed.items())],
                process_supply_requests=self.inventory_owner.events,
                received=sorted(inventory.committed), dock_source='accepted_handling_inventory',
                claim_boundary='Same-owner receiving ledger and process-service demand; buffer-to-machine consumption is software service, not physical transfer.')
            self.inventory_owner.close()
        for robot, emit in self.originals.items():
            self.runtime.nodes[robot].emit = emit
        if self.receiver is not None:
            self.executor.remove_node(self.receiver); self.receiver.destroy_node()
        if self.manager is not None:
            self.manager.shutdown()
        self.manager_closed = True
        return report

    def finish_evidence(self):
        """Called only after physical shutdown, before the parent seals files."""
        from .whole_line_ipc import settle_evidence_writer, receive_results
        result = settle_evidence_writer(self.process, self.output/'process')
        if result['writer_exited']:
            # Forced exit can interrupt a queue feeder; do not read that queue.
            if result['evidence_complete']:
                receive_results(self.queue, self.result)
            if not self.manager_closed:
                self.manager.shutdown(); self.manager_closed = True
            if self.inventory_owner is not None:
                self.inventory_owner.close()
        return result
