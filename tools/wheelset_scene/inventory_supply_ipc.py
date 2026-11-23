"""Process demand RPC drained exclusively by the fleet's inventory owner.

The child has no receiving, commit or restore capability and holds no stock
ledger. A timed-out request is uncertain and is never automatically retried.
"""
import math
import os
import threading
import uuid


class InventorySupplyOwner:
    """Serve bounded requests on the same thread that advances the fleet."""

    def __init__(self, supply, connection, channel_id, *, persist=None):
        self.supply = supply
        self.connection = connection
        identities = {gate.identity['run_id'] for gate in supply.sim.executions.values()}
        if len(identities) != 1:
            raise ValueError('one fleet run identity required')
        self.run_id = identities.pop()
        self.channel_id = channel_id
        self.owner = (os.getpid(), threading.get_ident())
        self.sequence = 0
        self.events = []
        self.closed = False
        self.persist = persist
        self.persistence_failed = False

    def poll(self, source_now, *, limit=64):
        """Drain on the owner thread; source time comes from its ROS inputs."""
        if self.owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError('inventory requests must run on the fleet owner thread')
        if (type(source_now) not in (int, float) or not math.isfinite(source_now)
                or type(limit) is not int or not 0 < limit <= 64):
            raise ValueError('finite owner source time and bounded request limit required')
        served = 0
        while not self.closed and served < limit:
            try:
                if not self.connection.poll():
                    break
                request = self.connection.recv()
            except (EOFError, OSError):
                self.close()
                break
            response = dict(run_id=self.run_id, channel_id=self.channel_id,
                            seq=request.get('seq') if isinstance(request, dict) else None)
            try:
                if (not isinstance(request, dict)
                        or set(request) != {'run_id', 'channel_id', 'seq', 'operation', 'arguments'}
                        or request['run_id'] != self.run_id or request['channel_id'] != self.channel_id
                        or type(request['seq']) is not int or request['seq'] != self.sequence+1):
                    raise ValueError('stale or invalid inventory request identity')
                self.sequence = request['seq']
                if any(g.identity['run_id'] != self.run_id for g in self.supply.sim.executions.values()):
                    raise ValueError('fleet run changed; establish a new process channel')
                operation, arguments = request['operation'], request['arguments']
                fields = {'snapshot': set(), 'can_start': {'job', 'station', 'now'},
                          'consume': {'job', 'station', 'token', 'now'}}
                if (not isinstance(operation, str) or operation not in fields
                        or not isinstance(arguments, dict) or set(arguments) != fields[operation]):
                    raise ValueError('process may only inspect stock or request identified consumption')
                if operation == 'snapshot':
                    value = dict(stock=self.supply.stock, consumed=sorted(self.supply.consumed))
                else:
                    if self.persistence_failed:
                        raise ValueError('inventory persistence failed; reconcile owner before process actions')
                    now = arguments['now']
                    if (type(now) not in (int, float) or not math.isfinite(now)
                            or abs(source_now-now) > .15):
                        raise ValueError('process source time differs from owner by more than 0.15s')
                    value = getattr(self.supply, operation)(**arguments)
                    if operation == 'consume' and self.persist is not None:
                        # Never acknowledge material consumption before its fleet checkpoint.
                        try:
                            if self.persist() is not True:
                                raise RuntimeError('fleet checkpoint did not confirm consumption')
                        except Exception as error:
                            self.persistence_failed = True
                            raise ValueError('consumption uncertain after persistence failure: '+str(error)) from error
                response.update(accepted=True, value=value)
            except (ValueError, TypeError) as error:
                response.update(accepted=False, error=str(error))
            self.events.append(dict(request=request, source_now=source_now,
                accepted=response['accepted'], error=response.get('error'), stock=self.supply.stock))
            try:
                self.connection.send(response)
            except (BrokenPipeError, EOFError, OSError):
                # Consumption may already be committed. Preserve it for reconciliation.
                self.close()
            served += 1
        return served

    def close(self):
        """Close only this owned request endpoint; inventory remains with the fleet."""
        if not self.closed:
            self.connection.close()
            self.closed = True


class InventorySupplyClient:
    """Spawnable process interface with request identity and no writable ledger."""

    def __init__(self, connection, run_id, channel_id, *, timeout_s=2.):
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('positive finite process request timeout required')
        self.connection = connection
        self.run_id = run_id
        self.channel_id = channel_id
        self.timeout_s = timeout_s
        self.sequence = 0
        self.closed = False

    def _request(self, operation, **arguments):
        if self.closed:
            raise RuntimeError('inventory process channel closed; reconcile before reconnecting')
        self.sequence += 1
        identity = dict(run_id=self.run_id, channel_id=self.channel_id, seq=self.sequence)
        try:
            self.connection.send(dict(**identity, operation=operation, arguments=arguments))
            if not self.connection.poll(self.timeout_s):
                raise TimeoutError('inventory owner reply missing; consumption outcome is uncertain')
            response = self.connection.recv()
            if not isinstance(response, dict) or any(response.get(k) != v for k, v in identity.items()):
                raise RuntimeError('inventory owner reply identity differs')
        except (OSError, EOFError, TimeoutError, RuntimeError):
            self.close()
            raise
        if response.get('accepted') is not True:
            raise ValueError(response.get('error', 'inventory request rejected'))
        return response['value']

    @property
    def stock(self):
        """Read an owner snapshot; modifying the returned copy cannot change stock."""
        return self._request('snapshot')['stock']

    @property
    def consumed(self):
        """Read the fleet's restored consumption identities for process recovery."""
        return {tuple(key) for key in self._request('snapshot')['consumed']}

    def can_start(self, job, station, now):
        return self._request('can_start', job=job, station=station, now=now)

    def consume(self, job, station, token, now):
        return self._request('consume', job=job, station=station, token=token, now=now)

    def close(self):
        if not self.closed:
            self.connection.close()
            self.closed = True


def inventory_process_channel(supply, context, *, timeout_s=2., persist=None):
    """Keep the Simulator in the parent; pass only the client pipe to a child."""
    parent, child = context.Pipe(duplex=True)
    channel_id = uuid.uuid4().hex
    owner = InventorySupplyOwner(supply, parent, channel_id, persist=persist)
    client = InventorySupplyClient(child, owner.run_id, channel_id, timeout_s=timeout_s)
    return owner, client
