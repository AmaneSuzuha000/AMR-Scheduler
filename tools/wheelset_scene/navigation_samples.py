"""Bounded equipment history selected against the authoritative ROS clock.

Receiving a duplicate never renews its wall-time freshness. A newer sample may
arrive before /clock; retain the preceding causal sample instead of starving
the controller or accepting a future observation. Clock resets fail closed.
"""
from collections import deque
import math


class _ContactSnapshot:
    """Only internal, locally serialized values; never load external pickle."""
    __slots__ = ('payload','metadata')

    def __init__(self, row):
        import pickle
        self.metadata=None
        if '_local_contact_payload' in row:
            from copy import deepcopy
            self.payload=row['_local_contact_payload']
            self.metadata=deepcopy({key:value for key,value in row.items()
                                   if key!='_local_contact_payload'})
        else:
            self.payload = pickle.dumps(row, protocol=5)

    def restore(self):
        import pickle
        if self.metadata is not None:
            from .navigation_records import restore_contact_message
            return restore_contact_message(dict(self.metadata,_local_contact_payload=self.payload))
        return pickle.loads(self.payload)


class EquipmentSamples:
    def __init__(self, capacity=128, *, serialize_contacts=False):
        if type(capacity) is not int or capacity < 2:
            raise ValueError('at least two bounded history slots required')
        self.capacity = capacity
        self.serialize_contacts = serialize_contacts
        self.rows = {}

    def add(self, key, row, *, source_s, received_wall):
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (source_s, received_wall)):
            raise ValueError('finite sample source and receipt times required')
        history = self.rows.setdefault(key, deque(maxlen=self.capacity))
        if history and (source_s <= history[-1][0] or received_wall < history[-1][1]):
            return False
        if key.endswith('/bin_contacts') and (self.serialize_contacts or '_local_contact_payload' in row):
            row = _ContactSnapshot(row)
        history.append((source_s, received_wall, row))
        return True

    def select(self, key, *, source_now, wall_now, source_age=.15, wall_age=.3):
        if not all(type(v) in (int, float) and math.isfinite(v)
                   for v in (source_now, wall_now, source_age, wall_age)):
            raise ValueError('finite selection clock and bounds required')
        if source_age <= 0 or wall_age <= 0:
            raise ValueError('positive freshness bounds required')
        for source_s, received, row in reversed(self.rows.get(key, ())):
            if (0 <= source_now-source_s <= source_age and 0 <= wall_now-received <= wall_age):
                return row.restore() if isinstance(row, _ContactSnapshot) else row
        return None
