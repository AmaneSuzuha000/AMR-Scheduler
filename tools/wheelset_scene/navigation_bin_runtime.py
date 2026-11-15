"""Contact-bin equipment adapter driven by the existing fleet owner's requests."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

from .navigation_handling import DockedBinObserver, checked_state
from .navigation_records import ros_message_value
from .navigation_samples import EquipmentSamples
from .verify_contact_conveyor import stamp


class TransferStroke:
    """Extend once, then retract; recovery observes without repeating transfer."""

    def __init__(self, geometry):
        self.geometry = geometry
        self.command = None; self.phase = 'idle'

    def update(self, command, actuator, source_now):
        if self.command != command:
            self.command = deepcopy(command)
            self.phase = 'retracting' if command.get('reconcile_only') else 'extending'
        if not actuator or not 0 <= source_now-stamp(actuator) <= .15:
            return None
        index = actuator['name'].index('bin_transfer_stroke')
        position, velocity = actuator['position'][index], actuator['velocity'][index]
        if not all(math.isfinite(value) for value in (position, velocity)):
            return None
        extended = self.geometry['load_stroke_m' if command['operation'] == 'LOAD' else 'unload_stroke_m']
        if self.phase == 'extending' and abs(position-extended) < .005 and abs(velocity) < .01:
            self.phase = 'retracting'
        if self.phase == 'retracting' and abs(position) < .005 and abs(velocity) < .01:
            self.phase = 'observing'
        return extended if self.phase == 'extending' else self.geometry['retract_stroke_m']


class ContactBinRuntime:
    """Observe equipment separately; only Simulator commits receipt and inventory."""

    # Class-level defaults keep the frozen 20 ms / 32-row turn for any adapter
    # built without __init__ (the fault-path tests), while productions runs pass
    # an explicit, validated budget.
    drain_budget_s = .020
    drain_rows = 32

    def __init__(self, runtime, executor, profile, output, record, *, input_thread=False,
                 drain_budget_s=.020, drain_rows=32):
        from gazebo_msgs.msg import ContactsState
        from gazebo_msgs.srv import GetEntityState
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
        if (runtime.sim.inventory is None or set(runtime.sim.handlings) != set(profile['robots'])
                or any(gate.source != profile['source'] for gate in runtime.sim.handlings.values())):
            raise ValueError('explicit contact handling and same-owner receiving inventory required')
        self.runtime = runtime; self.executor = executor; self.profile = deepcopy(profile); self.record = record
        if not (0 < drain_budget_s <= .020):
            raise ValueError('equipment drain budget must be in (0, 0.020] s')
        if type(drain_rows) is not int or not 1 <= drain_rows <= 32:
            raise ValueError('equipment drain rows must be an int in [1, 32]')
        self.drain_budget_s = float(drain_budget_s); self.drain_rows = drain_rows
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=False)
        self.node = Node('wheelset_contact_bins'); executor.add_node(self.node)
        self.ingress = None
        self.getter = self.node.create_client(GetEntityState, '/wheelset_review/get_entity_state')
        self.contacts = {}; self.actuators = {}; self.states = {}; self.pending = {}
        self.samples = EquipmentSamples(serialize_contacts=input_thread)
        self.last_query = -math.inf; self.last_publish = -math.inf
        self.strokes = {}; self.observers = {}; self.started = {}; self.receipts = []
        self.publishers = {}; self.subscriptions = []; self.closed = False
        self.sequences = {robot: -1 for robot in profile['robots']}
        self.targets = {robot: 0. for robot in profile['robots']}
        self.target_clocks = {}; self.source_now = None
        self.retractions = {}
        self.inhibitions = {}; self.receipt_waits = {}
        self.message = Float64MultiArray; self.request = GetEntityState.Request
        from rosidl_runtime_py.convert import message_to_ordereddict
        def receive(store, key, topic, message):
            row = ros_message_value(message)
            row['_topic'] = topic; row['_received_wall'] = time.monotonic()
            store[key] = row
            self.samples.add(topic, row, source_s=stamp(row), received_wall=row['_received_wall'])
            self.record('bin_equipment_sample', topic=topic, message=row)
        for task in profile['tasks']:
            topic = '/scout_bin/'+task+'/bin_contacts'
            if not input_thread:
                self.subscriptions.append(self.node.create_subscription(ContactsState, topic,
                    lambda m, task=task, topic=topic: receive(self.contacts, task, topic, m), qos_profile_sensor_data))
        for robot in profile['robots']:
            topic = '/scout_bin/'+robot+'/joint_states'
            if not input_thread:
                self.subscriptions.append(self.node.create_subscription(JointState, topic,
                    lambda m, robot=robot, topic=topic: receive(self.actuators, robot, topic, m), qos_profile_sensor_data))
            self.publishers[robot] = self.node.create_publisher(Float64MultiArray,
                '/scout_bin/'+robot+'/servo_targets', 1)
            self.strokes[robot] = TransferStroke(profile['geometry'])
            self.observers[robot] = DockedBinObserver(profile, self.write_evidence)
        if input_thread:
            from .navigation_equipment_ingress import EquipmentProcessIngress
            self.node.destroy_client(self.getter)
            self.ingress = EquipmentProcessIngress(self)
            self.getter = self.ingress.getter
            self.record('equipment_ingress_started', process_id=self.ingress.thread.pid,
                        queue_limit=self.ingress.queue.limit, mode='input_representation_process')

    def consume_inputs(self):
        """Only the owner applies observations, faults and query results."""
        if not getattr(self.ingress, 'detached', False):
            self.consume_rows(self.ingress.pull())
            return
        # The detached transport supplies complete local rows to the owner.
        # A 5 ms owner budget drained only about three rows per turn in the
        # frozen 15 diagnostics, allowing fresh equipment to age out while a
        # lease was active.  Keep the same one-row fault semantics and bounded
        # loop, but use a 20 ms budget (below the 50 ms readiness observation
        # gate) so normal input bursts can be consumed before polling.
        # The owner turn budget is a *latency* control: a longer drain keeps the
        # equipment queue smaller but slows every other safety turn, and the guard
        # feedback gate needs a fresh owner turn inside 200-250 ms. The budget and
        # row cap are therefore explicit, validated inputs instead of constants.
        deadline = time.monotonic()+self.drain_budget_s
        with self.ingress.owner_turn():
            for _ in range(self.drain_rows):
                rows = self.ingress.pull(limit=1)
                self.consume_rows(rows)
                if not rows or time.monotonic() >= deadline: break

    def consume_rows(self, rows):
        detached = getattr(self.ingress, 'detached', False)
        for kind, key, topic, message, received in rows:
            if kind == 'state':
                if message is None:
                    raise RuntimeError('contact equipment query returned no reply: '+key)
                observation = dict(request=topic if detached else ros_message_value(topic),
                    reply=message if detached else ros_message_value(message), received_wall=received)
                checked_state(observation, key)
                self.states[key] = observation; self.pending.pop(key, None)
                self.samples.add(key, observation, source_s=stamp(observation['reply']), received_wall=received)
                self.record('bin_independent_state', **observation)
            else:
                row = message if detached else ros_message_value(message)
                row['_topic'] = topic; row['_received_wall'] = received
                (self.contacts if kind == 'contact' else self.actuators)[key] = row
                self.samples.add(topic, row, source_s=stamp(row), received_wall=received)
                self.record('bin_equipment_sample', topic=topic, message=row)
                if kind == 'actuator' and row['position'][row['name'].index(key+'_servo_fault')] != 0.:
                    raise RuntimeError('native contact actuator lease fault: '+key)

    def write_evidence(self, identity, raw):
        import os
        if hashlib.sha256(raw).hexdigest() != identity:
            raise ValueError('contact evidence hash mismatch')
        target = self.output/(identity+'.json')
        with target.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        # The directory entry must also survive a host restart before acknowledgement.
        directory = os.open(self.output, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)

    def publish(self, robot, target, *, register_retraction=True):
        # The native protocol is sequence, target, enabled. Limit set-point
        # speed as on the contact fixture; a step would saturate the force.
        dt = max(0., min(.1, self.source_now-self.target_clocks.get(robot, self.source_now)))
        self.targets[robot] += max(-.08*dt, min(.08*dt, target-self.targets[robot]))
        self.target_clocks[robot] = self.source_now
        actuator = self.actuators.get(robot)
        acknowledged = -1
        if actuator:
            index = actuator['name'].index(robot+'_servo_sequence')
            acknowledged = actuator['position'][index]
        self.sequences[robot] = int(max(self.sequences[robot], acknowledged))+1
        handling = self.runtime.sim.handlings[robot]
        active = handling.command is not None and not handling.completed
        message = self.message()
        message.data = [float(self.sequences[robot]), self.targets[robot], 1. if active else 0.]
        self.publishers[robot].publish(message)
        if (register_retraction and active and target == 0.
                and abs(self.targets[robot]) < 1e-12 and robot not in self.retractions):
            self.retractions[robot] = self.sequences[robot]
        self.record('bin_transfer_target', robot=robot, requested_target_m=target, values=message.data,
                    phase=self.strokes[robot].phase, active=active,
                    retraction_registration=register_retraction, source_s=self.source_now)

    def inhibit(self, robot, reason, now, *, renew_zero_lease=False):
        """Explain withheld commands; optionally renew an enabled zero lease."""
        if renew_zero_lease:
            # A missing prerequisite sample must not advance the transfer stroke
            # or manufacture retraction evidence. It also must not let the
            # native axis lease expire while the owner waits for fresh inputs.
            # Keep the already-authorized handling enabled at a zero target.
            self.publish(robot, 0., register_retraction=False)
        previous = self.inhibitions.get(robot)
        if previous and previous[0] == reason and now-previous[1] < .25:
            return
        node = self.runtime.nodes[robot]
        self.inhibitions[robot] = (reason, now)
        guard = node.guard_status
        self.record('bin_transfer_inhibited', robot=robot, reason=reason,
            decision_wall=now, source_s=self.source_now, clock_s=node.clock_s,
            health=node.health.reason(node.clock_s, now),
            guard=None if guard is None else dict(state=guard.state, stopped=guard.stopped,
                actual_speed=guard.actual_speed, actual_wz=guard.actual_wz,
                output_vx=guard.output_vx, output_wz=guard.output_wz),
            state_times={name: dict(source_s=stamp(row['reply']), wall_age_s=now-row['received_wall'])
                         for name, row in self.states.items()},
            equipment_times={name: dict(source_s=stamp(row), wall_age_s=now-row['_received_wall'])
                             for name, row in {**self.contacts, **self.actuators}.items()},
            lease_renewal='enabled_zero_target' if renew_zero_lease else 'withheld')

    def poll(self, now):
        from rosidl_runtime_py.convert import message_to_ordereddict
        if self.closed:
            raise RuntimeError('contact adapter closed')
        if self.ingress is not None:
            self.consume_inputs()
            # Apply already received localization on this same owner turn.
            # Merely pulling the input snapshot updates the clock but leaves
            # the execution pose at the previous timer turn. A valid physical
            # receipt can then fail the unchanged vehicle/handling time gate.
            for node in self.runtime.nodes.values():
                node.query_feedback()
            # The caller's timestamp precedes input consumption and observer
            # polling. Comparing it with newly received data falsely marks
            # those data as future, and can accept data that have since aged.
            now = time.monotonic()
        clocks = [node.clock_s for node in self.runtime.nodes.values()]
        if any(clock is None for clock in clocks):
            return
        source_now = min(clocks)
        self.source_now = source_now
        for entity, (future, request, issued) in list(self.pending.items()):
            if self.ingress is None and future.done():
                reply = future.result()
                if reply is None:
                    raise RuntimeError('contact equipment query returned no reply: '+entity)
                observation = dict(request=request, reply=message_to_ordereddict(reply), received_wall=now)
                checked_state(observation, entity)
                self.states[entity] = observation; del self.pending[entity]
                self.samples.add(entity, observation, source_s=stamp(observation['reply']), received_wall=now)
                self.record('bin_independent_state', **observation)
            elif now-issued > 2.:
                raise RuntimeError('contact equipment query timed out: '+entity)
        if now-self.last_query >= .04 and self.getter.service_is_ready():
            # poll() begins only after all vehicles/sensors/guards are ready.
            entities = [robot+'::'+self.profile['geometry']['carrier_frame'] for robot in self.profile['robots']]
            entities += [spec['cargo_entity'] for spec in self.profile['tasks'].values()]
            for entity in entities:
                if entity not in self.pending:
                    request = self.request(); request.name = entity; request.reference_frame = 'world'
                    future = (self.getter.call_async(request) if self.ingress is None
                              else self.ingress.query(entity, request))
                    self.pending[entity] = (future, message_to_ordereddict(request), now)
            self.last_query = now
        if now-self.last_publish < .05:
            return
        self.last_publish = now
        for robot, handling in self.runtime.sim.handlings.items():
            self.poll_robot(robot, handling, now, source_now)

    def poll_robot(self, robot, handling, now, source_now):
        """Inspect native faults even while other transfer prerequisites are absent."""
        actuator = self.actuators.get(robot)
        if actuator and actuator['position'][actuator['name'].index(robot+'_servo_fault')] != 0.:
            raise RuntimeError('native contact actuator lease fault: '+robot)
        command = handling.command
        if not command or handling.completed:
            # Renew the lease only with an explicitly disabled zero target.
            # publish() derives the enabled bit from the same owner command,
            # so idle renewal cannot extend a contact stroke.
            self.publish(robot, 0.)
            return
        node = self.runtime.nodes[robot]; guard = node.guard_status; gate = handling.execution
        if (gate.fault or not node.guard_fresh() or guard.state != 'latched' or not guard.stopped
                or guard.actual_speed >= .01 or abs(guard.actual_wz) >= .02
                or guard.output_vx != 0 or guard.output_wz != 0
                or node.health.reason(node.clock_s, now) is not None):
            self.inhibit(robot, 'execution_guard_or_sensor_health', now)
            return
        if command['handling_id'] not in self.started:
            self.started[command['handling_id']] = (now, source_now)
            self.retractions.pop(robot, None)
        wall_start, source_start = self.started[command['handling_id']]
        if now-wall_start > 600. or source_now-source_start > 90.:
            raise RuntimeError('contact handoff timeout: '+command['handling_id'])
        spec = self.profile['tasks'][command['task_id']]
        keys = dict(carrier=robot+'::'+self.profile['geometry']['carrier_frame'],
                    cargo=spec['cargo_entity'],
                    contact='/scout_bin/'+command['task_id']+'/bin_contacts',
                    actuator='/scout_bin/'+robot+'/joint_states')
        selected = {name: self.samples.select(key, source_now=source_now, wall_now=now)
                    for name, key in keys.items()}
        carrier, cargo, contact, actuator = (selected[name] for name in ('carrier', 'cargo', 'contact', 'actuator'))
        if not all((carrier, cargo, contact, actuator)):
            self.inhibit(robot, 'equipment_causal_fresh_sample_missing', now,
                         renew_zero_lease=True)
            return
        self.record('bin_selected_samples', robot=robot, decision_wall=now, source_s=source_now,
            execution_pose=deepcopy(gate.pose), execution_tolerance_m=gate.tolerance,
            times={name: dict(source_s=stamp(row['reply'] if name in ('carrier', 'cargo') else row),
                received_wall=row['received_wall'] if name in ('carrier', 'cargo') else row['_received_wall'])
                for name, row in selected.items()})
        station = spec['stations'][command['operation']]
        state = carrier['reply']['state']; p, q = state['pose']['position'], state['pose']['orientation']
        yaw = math.atan2(2*(q['w']*q['z']+q['x']*q['y']), 1-2*(q['y']**2+q['z']**2))
        distance = math.dist([p['x'], p['y']], station['target_xy'])
        yaw_error = abs(math.atan2(math.sin(yaw-station['yaw_rad']), math.cos(yaw-station['yaw_rad'])))
        if (distance > self.profile['dock_position_tolerance_m']
                or yaw_error > self.profile['dock_yaw_tolerance_rad']):
            raise RuntimeError('contact dock outside capture range: '+json.dumps(dict(
                robot=robot, node=command['node_id'], position_error_m=distance, yaw_error_rad=yaw_error)))
        target = self.strokes[robot].update(command, actuator, source_now)
        if target is None:
            self.inhibit(robot, 'actuator_sample_not_usable', now)
            return
        self.inhibitions.pop(robot, None)
        self.publish(robot, target)
        if self.strokes[robot].phase != 'observing' or robot not in self.retractions:
            return
        actuator = dict(actuator, _retraction_sequence=self.retractions[robot])
        receipt = self.observers[robot].observe(command, source_now=source_now,
            carrier=carrier, cargo=cargo, contact=contact, actuator=actuator,
            execution_pose=deepcopy(gate.pose), execution_tolerance_m=gate.tolerance)
        wait_reason = self.observers[robot].last_wait_reason
        waits = getattr(self, 'receipt_waits', None)
        if waits is None:
            waits = self.receipt_waits = {}
        if wait_reason:
            previous = waits.get(robot)
            if previous is None or previous[0] != wait_reason or now-previous[1] >= 1.:
                pose_timestamp = gate.pose.get('timestamp') if isinstance(gate.pose, dict) else None
                self.record('bin_receipt_waiting_for_localized_pose', robot=robot,
                    reason=wait_reason, contact_source_s=stamp(contact),
                    pose_source_s=pose_timestamp,
                    source_delta_s=(pose_timestamp-stamp(contact)
                        if type(pose_timestamp) in (int, float) else None))
                waits[robot] = (wait_reason, now)
        else:
            waits.pop(robot, None)
        if receipt is not None:
            answer = self.runtime.sim.handling_feedback(receipt, now, source_now)
            self.record('bin_handling_feedback', message=receipt, result=answer)
            if not answer['accepted'] or not self.runtime.persist():
                raise RuntimeError('contact receipt rejected or fleet checkpoint not durable: '+str(answer))
            self.receipts.append(receipt)

    def close(self):
        if self.closed:
            return
        # Successful receipts already require retraction. On failure retain the
        # load, task and resource state; shutdown is not a completion receipt.
        for robot, publisher in self.publishers.items():
            self.sequences[robot] += 1
            message = self.message(); message.data = [float(self.sequences[robot]), self.targets[robot], 0.]
            publisher.publish(message)
        try:
            if self.ingress is not None:
                if getattr(self.ingress, 'detached', False):
                    try: summary = self.ingress.close(consume=self.consume_inputs)
                    except Exception:
                        self.record('equipment_ingress_closed', **self.ingress.close_result)
                        raise
                else:
                    summary = self.ingress.close()
                self.record('equipment_ingress_closed', **summary)
        finally:
            self.executor.remove_node(self.node); self.node.destroy_node(); self.closed = True
        return dict(receipts=self.receipts, scope='contact receipts only; no navigation truth feedback')
