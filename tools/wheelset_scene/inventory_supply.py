"""Same-owner process supply backed by the fleet's receiving inventory.

This adapter has no receiving/commit API and stores no second stock ledger.
It is for process services on the scheduler owner, not a separately mutable
IPC copy. Station names identify declared service demand; they do not imply
that a physical transfer from a line-side buffer to a machine was observed.
"""
import math


class InventorySupply:
    """Adapt existing process demand calls without duplicating material ownership."""

    def __init__(self, sim, station_buffers):
        if sim.inventory is None or not sim.handlings:
            raise ValueError('process supply requires the fleet-owned measured-handling inventory')
        if (not isinstance(station_buffers, dict) or not station_buffers
                or any(not isinstance(k, str) or not k or not isinstance(v, str)
                       or v not in sim.inventory.capacities for k, v in station_buffers.items())
                or len(set(station_buffers.values())) != len(station_buffers)):
            raise ValueError('distinct tracked buffers for named process services required')
        self.sim = sim
        self.station_buffers = dict(station_buffers)

    @property
    def stock(self):
        """Expose copies of current stock; restoring the fleet changes this view."""
        return {station: self.sim.inventory.stock[node]
                for station, node in self.station_buffers.items()}

    @property
    def consumed(self):
        """Read existing charges, including those restored with the fleet checkpoint."""
        stations = {node: station for station, node in self.station_buffers.items()}
        return {(job, stations[node]) for job, node in self.sim.inventory.consumed
                if node in stations}

    @staticmethod
    def _validate(job, now):
        if (not isinstance(job, str) or not job or type(now) not in (int, float)
                or not math.isfinite(now)):
            raise ValueError('identified service and finite source time required')

    def can_start(self, job, station, now):
        """Missing material keeps the process ready; a retry reuses its charge."""
        self._validate(job, now)
        if station not in self.station_buffers:
            return True  # Other process stages have no recipe in this adapter.
        node = self.station_buffers[station]
        return (job, node) in self.sim.inventory.consumed or self.sim.inventory.stock[node] > 0

    def consume(self, job, station, token, now):
        """Charge one declared service unit once, leaving receiving reservations intact."""
        self._validate(job, now)
        if not isinstance(token, str) or not token:
            raise ValueError('identified process action required')
        if station not in self.station_buffers:
            return
        node = self.station_buffers[station]
        if not self.sim.inventory.consume(job, node, now, command_id=token):
            raise ValueError('service lacks material or was already charged')
