"""Receive contact equipment off the decision executor; preserve every sample."""
from contextlib import contextmanager
from tools.live_bridge.feedback_ingress import InputWorker, ReceiptQueue


class _ReservedEquipmentQueue:
    """One capacity budget includes pipe, in-flight transfer and local inbox."""

    def __init__(self, context, limit):
        self.wire = context.Queue(maxsize=limit)
        self.slots = context.BoundedSemaphore(limit)

    def put_nowait(self, value):
        import queue
        if not self.slots.acquire(block=False):
            raise queue.Full
        try: self.wire.put_nowait(value)
        except BaseException:
            self.slots.release()
            raise

    def get(self, **options): return self.wire.get(**options)
    def release(self): self.slots.release()
    def close(self): self.wire.close()
    def join_thread(self): self.wire.join_thread()
    def cancel_join_thread(self): self.wire.cancel_join_thread()


def _receive_equipment(profile, pending, commands, stopped, failed, ready,
                       accepted, counts, topics, answer):
    """Receive/convert only; no task, motion, handling or inventory authority."""
    import queue
    import time
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from gazebo_msgs.msg import ContactsState
    from gazebo_msgs.srv import GetEntityState
    from sensor_msgs.msg import JointState
    from .navigation_records import ros_message_value
    node = executor = None
    subscriptions = []; futures = {}; error = None; closing = None; end_commands = False
    initialized = False
    try:
        rclpy.init(); initialized = True
        node = Node('amr_equipment_process', use_global_arguments=False)
        executor = SingleThreadedExecutor(); executor.add_node(node)
        getter = node.create_client(GetEntityState, '/wheelset_review/get_entity_state')

        def emit(kind, key, topic, message, received, request_id=None):
            sequence = accepted.value+1
            try:
                pending.put_nowait((sequence, kind, key, topic, message, received, request_id))
            except queue.Full:
                failed.value = failed.value or 1
                return
            accepted.value = sequence

        def receive(kind, key, topic, message):
            received = time.monotonic()
            counts[topics.index(topic)] += 1
            value=ros_message_value(message)
            if kind=='contact':
                from .navigation_records import freeze_contact_message
                value=freeze_contact_message(value,topic,received)
            emit(kind, key, topic, value, received)

        for task in profile['tasks']:
            topic = '/scout_bin/'+task+'/bin_contacts'
            subscriptions.append(node.create_subscription(ContactsState, topic,
                lambda m, key=task, topic=topic: receive('contact', key, topic, m), qos_profile_sensor_data))
        for robot in profile['robots']:
            topic = '/scout_bin/'+robot+'/joint_states'
            subscriptions.append(node.create_subscription(JointState, topic,
                lambda m, key=robot, topic=topic: receive('actuator', key, topic, m), qos_profile_sensor_data))

        def issue(request_id, entity, fields):
            request = GetEntityState.Request(**fields)
            future = getter.call_async(request); futures[request_id] = future
            def completed(reply):
                received = time.monotonic()
                try:
                    value = reply.result()
                    if value is None: raise RuntimeError('entity query returned no reply: '+entity)
                    emit('state', entity, fields, ros_message_value(value), received, request_id)
                except Exception:
                    failed.value = failed.value or 2
                    raise
                finally:
                    futures.pop(request_id, None)
            future.add_done_callback(completed)

        while True:
            if stopped.is_set() and closing is None:
                closing = time.monotonic(); ready.value = 0
                for subscription in subscriptions: node.destroy_subscription(subscription)
                subscriptions.clear()
            for _ in range(32):
                try: command = commands.get_nowait()
                except queue.Empty: break
                if command is None:
                    end_commands = True
                    break
                request_id, entity, fields = command
                issue(request_id, entity, fields)
            if closing is not None:
                if not futures and end_commands: break
                if time.monotonic()-closing > 2.:
                    raise RuntimeError('entity queries unresolved during input shutdown')
            else:
                ready.value = int(getter.service_is_ready())
            if failed.value: raise RuntimeError('equipment input transport failed: '+str(failed.value))
            executor.spin_once(timeout_sec=.02)
    except BaseException as exc:
        failed.value = failed.value or 2
        error = type(exc).__name__+': '+str(exc)
        raise
    finally:
        ready.value = 0
        try:
            if executor is not None:
                if node is not None: executor.remove_node(node)
                if not executor.shutdown(timeout_sec=1.): raise RuntimeError('input executor did not stop')
            if node is not None: node.destroy_node()
            if initialized: rclpy.shutdown()
        except Exception as exc:
            failed.value = failed.value or 2
            error = error or type(exc).__name__+': '+str(exc)
        # The owner drains accepted rows while joining. Error paths cancel the
        # feeder and remain explicit failures, never successful empty queues.
        pending.close()
        if failed.value: pending.cancel_join_thread()
        else: pending.join_thread()
        commands.close()
        try:
            answer.send(dict(passed=not failed.value, error=error, accepted=accepted.value,
                callbacks=dict(zip(topics, counts[:])), unresolved_queries=len(futures)))
        finally:
            answer.close()


class _ProcessQueueView:
    limit = 2048

    def __init__(self, owner): self.owner = owner

    @property
    def pending(self):
        # Advisory only. Sequence validation and the final accepted count are
        # authoritative; neither qsize() nor empty() establishes completeness.
        return range(max(0, self.owner.accepted.value-self.owner.sequence))

    @property
    def error(self):
        code = self.owner.failed.value
        return {0: None, 1: 'input queue overflow', 2: 'equipment receiver failed'}.get(code, 'equipment receiver terminated')


class EquipmentProcessIngress:
    """FIFO input representation in a process; only the creating owner consumes."""
    detached = True

    def __init__(self, owner, *, _target=_receive_equipment):
        import multiprocessing
        import queue
        import threading
        import uuid
        context = multiprocessing.get_context('spawn')
        self.owner_thread = threading.get_ident(); self.lock = threading.RLock()
        self.worker = self; self.error = None; self.closed = False; self.closing = False
        self.sequence = 0; self.query_sequence = 0; self.queries = {}; self.query_entities = set()
        self.epoch = uuid.uuid4().hex; self.close_result = None
        self.profile = owner.profile
        self.topics = ['/scout_bin/'+task+'/bin_contacts' for task in owner.profile['tasks']]
        self.topics += ['/scout_bin/'+robot+'/joint_states' for robot in owner.profile['robots']]
        self.pending = _ReservedEquipmentQueue(context, 2048)
        self.inbox = queue.Queue(maxsize=32)
        self.transfer_stop = threading.Event(); self.transfer_error = None
        self.commands = context.Queue(maxsize=32)
        self.stopped = context.Event(); self.failed = context.Value('i', 0)
        self.ready = context.Value('i', 0); self.accepted = context.Value('q', 0, lock=False)
        self.received_counts = context.Array('q', len(self.topics), lock=False)
        self.answer, child_answer = context.Pipe(duplex=False)
        self.queue = _ProcessQueueView(self); self.getter = self
        self.thread = context.Process(target=_target, args=(owner.profile, self.pending,
            self.commands, self.stopped, self.failed, self.ready, self.accepted,
            self.received_counts, self.topics, child_answer), name='amr_equipment_input', daemon=True)
        try: self.thread.start()
        except BaseException:
            child_answer.close(); self.answer.close()
            self.pending.close(); self.commands.close()
            raise
        child_answer.close()
        self.transfer = threading.Thread(target=self._transfer_inputs,
            name='amr_equipment_transport', daemon=True)
        self.transfer.start()

    def _transfer_inputs(self):
        """Move complete FIFO messages only; never validate, decide or renew."""
        import queue
        try:
            while not self.transfer_stop.is_set():
                try: wire = self.pending.get(timeout=.05)
                except queue.Empty: continue
                while not self.transfer_stop.is_set():
                    try:
                        self.inbox.put(wire, timeout=.05)
                        break
                    except queue.Full: continue
        except BaseException as exc:
            self.transfer_error = type(exc).__name__+': '+str(exc)

    @property
    def counts(self): return dict(zip(self.topics, self.received_counts[:]))

    def check_owner(self):
        import threading
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError('only the decision thread may consume inputs')

    def check(self):
        import time
        self.check_owner()
        if self.closed: raise RuntimeError('equipment input already closed')
        self.error = self.error or self.queue.error or self.transfer_error
        if not getattr(self, '_owner_turn_active', False) and not self.transfer.is_alive():
            self.error = self.error or 'equipment input transport stopped'
        if not getattr(self, '_owner_turn_active', False) and not self.closing and not self.thread.is_alive():
            self.error = self.error or 'equipment input process stopped'
        if not self.closing and any(time.monotonic()-value[2] > 2. for value in self.queries.values()):
            self.error = self.error or 'contact equipment query timed out'
        if self.error and not getattr(self,'_shutdown_drain_active',False):
            raise RuntimeError('equipment feedback input: '+self.error)

    @contextmanager
    def owner_turn(self):
        """Check OS liveness around one bounded drain, before control may run.

        Every row still checks sticky transport errors and exact FIFO/query
        identities. Only repeated OS process polling is batched. This context
        has no publishing or scheduling authority.
        """
        self.check()
        if getattr(self, '_owner_turn_active', False):
            raise RuntimeError('equipment owner drain already active')
        self._owner_turn_active = True
        try: yield
        finally:
            self._owner_turn_active = False
            self.check()

    def service_is_ready(self):
        self.check()
        return not self.closing and bool(self.ready.value)

    def query(self, entity, request):
        from concurrent.futures import Future
        import queue
        import time
        from .navigation_records import ros_message_value
        self.check()
        if self.closing: raise RuntimeError('equipment input is closing')
        if entity in self.query_entities: raise RuntimeError('duplicate pending entity query: '+entity)
        fields = ros_message_value(request)
        if not isinstance(fields, dict) or fields.get('name') != entity:
            raise ValueError('entity query identity mismatch')
        self.query_sequence += 1
        identity = self.epoch+':'+str(self.query_sequence); future = Future()
        try: self.commands.put_nowait((identity, entity, fields))
        except queue.Full:
            self.error = 'equipment query command queue overflow'
            raise RuntimeError(self.error)
        self.queries[identity] = (entity, fields, time.monotonic(), future)
        self.query_entities.add(entity)
        return future

    def pull(self, limit=32):
        import math
        import queue
        if type(limit) is not int or not 1 <= limit <= 32:
            raise ValueError('input pull limit must be between 1 and32')
        self.check(); rows = []
        try:
            for _ in range(limit):
                # multiprocessing.Queue.get_nowait() can wait for the rest of
                # a large pipe frame after its header is readable. Only the
                # transport thread may read that pipe; the owner sees complete
                # local rows and retains every identity/fault/query mutation.
                try: wire = self.inbox.get_nowait()
                except queue.Empty: break
                self.pending.release()
                sequence, kind, key, topic, message, received, request_id = wire
                if sequence != self.sequence+1 or not math.isfinite(received):
                    raise ValueError('equipment input sequence or receipt mismatch')
                if not isinstance(message, dict): raise ValueError('detached equipment input required')
                if kind == 'state':
                    expected = self.queries.get(request_id)
                    if expected is None or key != expected[0] or topic != expected[1]:
                        raise ValueError('equipment query reply identity mismatch')
                    del self.queries[request_id]; self.query_entities.remove(key)
                    expected[3].set_result(message)
                elif (kind not in ('contact', 'actuator') or request_id is not None
                      or topic != '/scout_bin/'+key+('/bin_contacts' if kind == 'contact' else '/joint_states')
                      or key not in self.profile['tasks' if kind == 'contact' else 'robots']):
                    raise ValueError('equipment input topic identity mismatch')
                self.sequence = sequence
                rows.append((kind, key, topic, message, received))
        except Exception as exc:
            self.error = self.error or type(exc).__name__+': '+str(exc)
            raise RuntimeError('equipment feedback input: '+self.error) from exc
        self.check()
        return rows

    def close(self, consume=None):
        import time
        self.check_owner()
        if self.closed:
            if self.close_result and self.close_result['passed']: return self.close_result
            raise RuntimeError('equipment input close already failed: '+str(self.close_result))
        self.closing = True; self.stopped.set(); error = None
        # Teardown may preserve the complete FIFO after a known runtime fault.
        # Only the owner consumes; row sequence/query identity checks remain in
        # pull(). No polling, motion, task or inventory update is called here.
        self._shutdown_drain_active=True
        original_error=self.error or self.queue.error or self.transfer_error
        deadline = time.monotonic()+3.
        def consume_owned():
            nonlocal error
            if consume is None:
                if self.accepted.value != self.sequence:
                    raise RuntimeError('equipment input requires owner consumption before close')
                return
            before = self.sequence
            try:
                consume()
            except Exception as exc:
                # A verified actuator fault invalidates the run, but must not
                # hide later accepted samples/replies or force-kill a healthy
                # receiver. Continue only when the owner actually advanced;
                # transport/identity errors with no progress remain fatal.
                error = error or type(exc).__name__+': '+str(exc)
                if self.sequence == before: raise
        try:
            # Same-producer FIFO sentinel follows all accepted query requests.
            # Never infer completeness from multiprocessing.Queue.empty().
            self.commands.put_nowait(None)
            while self.thread.is_alive() and time.monotonic() < deadline:
                consume_owned()
                self.thread.join(timeout=.001)
            if self.thread.is_alive(): raise RuntimeError('equipment input process did not close within bound')
            if consume is not None:
                while self.accepted.value != self.sequence and time.monotonic() < deadline: consume_owned()
            if self.accepted.value != self.sequence: raise RuntimeError('equipment input accepted rows not consumed')
            self.check()
        except Exception as exc:
            error = error or type(exc).__name__+': '+str(exc)
        finally:
            self._shutdown_drain_active=False
            if self.thread.is_alive():
                self.thread.terminate(); self.thread.join(timeout=1.)
            if self.thread.is_alive():
                self.thread.kill(); self.thread.join(timeout=1.)
            self.transfer_stop.set(); self.transfer.join(timeout=1.)
            if self.transfer.is_alive():
                error = error or 'equipment input transport did not stop'
            error = error or self.transfer_error
            try: result = self.answer.recv() if self.answer.poll() else None
            except (EOFError, OSError): result = None
            self.answer.close()
            self.commands.cancel_join_thread(); self.commands.close()
            self.pending.cancel_join_thread(); self.pending.close()
            for value in self.queries.values(): value[3].cancel()
            self.closed = True
        passed = (error is None and self.error is None and not self.failed.value and not self.queries
            and result is not None and result['passed'] and result['accepted'] == self.sequence
            and self.thread.exitcode == 0 and not self.thread.is_alive() and not self.transfer.is_alive())
        self.close_result = dict(passed=passed, joined=not self.thread.is_alive(),
            executor_stopped=bool(result and result['passed']), error=error or self.error,
            receiver_result=result, accepted_samples=self.accepted.value, consumed_samples=self.sequence,
            pending_samples=max(0, self.accepted.value-self.sequence), queue_error=self.queue.error,
            callbacks=self.counts, process_id=self.thread.pid, unresolved_queries=len(self.queries),
            transport_stopped=not self.transfer.is_alive(), transport_error=self.transfer_error,
            original_input_error=original_error,
            input_complete=(result is not None and result['accepted']==self.sequence
                and self.accepted.value==self.sequence and not self.queries
                and not self.thread.is_alive() and not self.transfer.is_alive()))
        if not passed: raise RuntimeError('equipment input incomplete: '+str(self.close_result))
        return self.close_result


class EquipmentIngress:
    def __init__(self, owner):
        from gazebo_msgs.msg import ContactsState
        from gazebo_msgs.srv import GetEntityState
        from sensor_msgs.msg import JointState
        from rclpy.qos import qos_profile_sensor_data
        self.worker = InputWorker(owner.node, 'amr_equipment_ingress')
        self.queue = ReceiptQueue(2048)
        self.getter = self.worker.node.create_client(GetEntityState, '/wheelset_review/get_entity_state')
        for task in owner.profile['tasks']:
            self.subscribe(ContactsState, '/scout_bin/'+task+'/bin_contacts', 'contact', task, qos_profile_sensor_data)
        for robot in owner.profile['robots']:
            self.subscribe(JointState, '/scout_bin/'+robot+'/joint_states', 'actuator', robot, qos_profile_sensor_data)
        self.worker.start()

    def subscribe(self, cls, topic, kind, key, qos):
        self.worker.subscription(cls, topic,
            lambda message, received: self.queue.append((kind, key, topic, message, received)), qos)

    def query(self, entity, request):
        self.worker.check_owner()
        future = self.getter.call_async(request)
        def completed(reply):
            import time
            received = time.monotonic()
            with self.worker.lock:
                try: value = reply.result()
                except Exception as exc:
                    self.worker.error = type(exc).__name__+': '+str(exc); return
                self.queue.append(('state', entity, request, value, received))
        future.add_done_callback(completed)
        return future

    def pull(self):
        self.worker.check_owner()
        with self.worker.lock:
            error = self.worker.error or self.queue.error
            if not self.worker.thread.is_alive(): error = error or 'equipment input thread stopped'
            # Bound conversion work per owner turn without discarding history.
            # Backlog stays in the original 2048-slot queue, including faults;
            # overflow remains sticky and invalidates the complete snapshot.
            values = self.queue.drain(limit=32)
        if error: raise RuntimeError('equipment feedback input: '+error)
        return values

    def close(self):
        result = self.worker.close()
        return dict(pending_samples=len(self.queue.pending), queue_error=self.queue.error, **result)
