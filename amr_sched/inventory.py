"""Finite receiving buffers owned by the same executor as physical deliveries.

Quantities are declared buffer units, not inferred masses or vehicle payloads.
Only an accepted equipment UNLOAD receipt commits a delivery. Reservations
remain occupied during uncertain transfers and across owner restarts.
"""
from copy import deepcopy
import math


class DeliveryInventory:
    """Explicit task recipes, finite slots and identified workstation consumption."""

    def __init__(self, capacities, deliveries, *, initial=None):
        if (not isinstance(capacities, dict) or not capacities
                or any(not isinstance(k, str) or not k or type(v) is not int or v <= 0
                       for k, v in capacities.items())):
            raise ValueError('positive integer capacities for named receiving nodes required')
        self.capacities = dict(capacities)
        self.initial = dict.fromkeys(capacities, 0)
        if initial is not None:
            if not isinstance(initial, dict) or set(initial) - set(capacities):
                raise ValueError('initial stock must name configured buffers')
            self.initial.update(initial)
        if any(type(v) is not int or not 0 <= v <= capacities[k] for k, v in self.initial.items()):
            raise ValueError('initial stock exceeds declared buffer capacity')
        if not isinstance(deliveries, dict):
            raise ValueError('explicit task delivery recipes required')
        for task, recipe in deliveries.items():
            if (not isinstance(task, str) or not task or not isinstance(recipe, dict)
                    or set(recipe) != {'node_id', 'quantity'} or not isinstance(recipe['node_id'], str)
                    or recipe['node_id'] not in capacities
                    or type(recipe['quantity']) is not int
                    or not 0 < recipe['quantity'] <= capacities[recipe['node_id']]):
                raise ValueError('each delivery requires a receiving node and feasible quantity')
        self.deliveries = deepcopy(deliveries)
        self.stock = dict(self.initial)
        self.reservations = {}
        self.committed = {}
        self.consumed = {}
        self.events = []

    def definition(self):
        """Freeze units, capacities, initial stock and task-to-buffer recipes."""
        return deepcopy(dict(capacities=self.capacities, deliveries=self.deliveries, initial=self.initial))

    def validate_task(self, task):
        """Tasks to a tracked buffer must have an explicit matching recipe."""
        recipe = self.deliveries.get(task.task_id)
        return ((recipe is None and task.dropoff not in self.capacities)
                or recipe is not None and recipe['node_id'] == task.dropoff)

    def _event(self, event, source_now, **values):
        self.events.append(dict(seq=len(self.events), event=event, source_time_s=source_now,
                                **values, stock=dict(self.stock)))

    def reserve(self, task_id, robot_id, node_id, source_now):
        """Authorize unloading only after reserving all required receiving units."""
        recipe = self.deliveries.get(task_id)
        if recipe is None:
            return node_id not in self.capacities
        if (recipe['node_id'] != node_id or task_id in self.committed
                or type(source_now) not in (int, float) or not math.isfinite(source_now)):
            return False
        wanted = dict(robot_id=robot_id, **recipe)
        if task_id in self.reservations:
            return self.reservations[task_id] == wanted
        reserved = sum(r['quantity'] for r in self.reservations.values() if r['node_id'] == node_id)
        if self.stock[node_id] + reserved + recipe['quantity'] > self.capacities[node_id]:
            return False
        self.reservations[task_id] = wanted
        self._event('STOCK_RESERVED', source_now, task_id=task_id, **wanted)
        return True

    def receipt_allowed(self, message):
        """Check capacity ownership before the handling gate changes its state."""
        task_id = message.get('task_id')
        if any(not isinstance(message.get(k), str) for k in
               ('task_id', 'operation', 'robot_id', 'node_id', 'evidence_id')):
            return False
        if message.get('operation') != 'UNLOAD':
            return True
        recipe = self.deliveries.get(task_id)
        if recipe is None:
            return message.get('node_id') not in self.capacities
        expected = dict(robot_id=message.get('robot_id'), node_id=message.get('node_id'),
                        quantity=recipe['quantity'])
        if task_id in self.committed:
            previous = self.committed[task_id]
            return (message.get('reconcile_only') is True
                    and all(previous[k] == v for k, v in expected.items())
                    and message['evidence_id'] != previous['evidence_id'])
        return (task_id not in self.committed and self.reservations.get(task_id) == expected
                and message.get('evidence_id') not in {v['evidence_id'] for v in self.committed.values()})

    def _commit_accepted_receipt(self, message):
        """Called by Simulator only after its physical handling gate accepts."""
        if message['operation'] != 'UNLOAD' or message['task_id'] not in self.deliveries:
            return
        task_id = message['task_id']
        if task_id in self.committed:
            self._event('STOCK_RECONCILED', message['timestamp'], task_id=task_id,
                        handling_id=message['handling_id'], evidence_id=message['evidence_id'])
            return
        reserved = self.reservations.pop(task_id)
        self.stock[reserved['node_id']] += reserved['quantity']
        committed = dict(**reserved, handling_id=message['handling_id'], evidence_id=message['evidence_id'])
        self.committed[task_id] = committed
        self._event('STOCK_COMMITTED', message['timestamp'], task_id=task_id, **committed)

    def consume(self, job_id, node_id, source_now, *, quantity=1, command_id=None):
        """One identified service consumes once; lack of material leaves it waiting."""
        if (not isinstance(job_id, str) or not job_id or node_id not in self.stock
                or type(quantity) is not int or quantity <= 0
                or type(source_now) not in (int, float) or not math.isfinite(source_now)
                or command_id is not None and (not isinstance(command_id, str) or not command_id)):
            return False
        key = (job_id, node_id)
        if key in self.consumed or self.stock[node_id] < quantity:
            return False
        self.stock[node_id] -= quantity
        self.consumed[key] = quantity
        provenance = {} if command_id is None else dict(command_id=command_id)
        self._event('MATERIAL_CONSUMED', source_now, job_id=job_id, node_id=node_id,
                    quantity=quantity, **provenance)
        return True

    def snapshot(self):
        """Capture together with task state and traffic reservations on one owner."""
        return deepcopy(dict(stock=self.stock, reservations=self.reservations,
                             committed=self.committed, consumed=self.consumed, events=self.events))

    def validate_snapshot(self, saved):
        """Reject inconsistent inventory before mutating any restored fleet state."""
        if (not isinstance(saved, dict)
                or set(saved) != {'stock', 'reservations', 'committed', 'consumed', 'events'}
                or any(not isinstance(saved[k], dict) for k in ('stock', 'reservations', 'committed', 'consumed'))
                or set(saved['stock']) != set(self.capacities) or not isinstance(saved['events'], list)):
            raise ValueError('inventory checkpoint fields differ')
        reservations, commits, consumed = (saved[k] for k in ('reservations', 'committed', 'consumed'))
        if set(reservations) & set(commits):
            raise ValueError('inventory transfer is both reserved and committed')
        for table, fields in ((reservations, {'robot_id', 'node_id', 'quantity'}),
                (commits, {'robot_id', 'node_id', 'quantity', 'handling_id', 'evidence_id'})):
            for task, value in table.items():
                recipe = self.deliveries.get(task)
                if (recipe is None or not isinstance(value, dict) or set(value) != fields
                        or type(value.get('quantity')) is not int
                        or any(value.get(k) != v for k, v in recipe.items())
                        or any(not isinstance(value[k], str) or not value[k] for k in fields - {'quantity'})):
                    raise ValueError('inventory checkpoint transfer identity differs')
        if len({v['evidence_id'] for v in commits.values()}) != len(commits):
            raise ValueError('one support proof cannot commit multiple deliveries')
        for key, quantity in consumed.items():
            if (not isinstance(key, tuple) or len(key) != 2 or not isinstance(key[0], str)
                    or not key[0] or key[1] not in self.capacities or type(quantity) is not int or quantity <= 0):
                raise ValueError('invalid inventory consumption identity')
        for node, capacity in self.capacities.items():
            value = saved['stock'][node]
            committed = sum(v['quantity'] for v in commits.values() if v['node_id'] == node)
            used = sum(q for (_, n), q in consumed.items() if n == node)
            reserved = sum(v['quantity'] for v in reservations.values() if v['node_id'] == node)
            if (type(value) is not int or value != self.initial[node] + committed - used
                    or not 0 <= value <= capacity or value + reserved > capacity):
                raise ValueError('inventory stock conservation or capacity violated')

    def restore(self, saved):
        """Restore already validated stock and uncertain receiving-slot ownership."""
        self.validate_snapshot(saved)
        self.__dict__.update(deepcopy(saved))
