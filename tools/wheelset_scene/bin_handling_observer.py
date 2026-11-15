"""Derive handling receipts from independently recorded bin support samples."""
from collections import defaultdict
import hashlib
import json
import math
import re

from .verify_contact_conveyor import norm, stamp, world_vertical_force


class BinHandlingObserver:
    """One stopped, continuously supported bin; actual records precede receipts.

    This fixture uses Gazebo truth only to observe the carrier and cargo. It is
    not a navigation/localization source and does not implement fleet routing.
    """
    def __init__(self, write_evidence, *, stable_s=.3, freshness_s=.15):
        self.write_evidence = write_evidence
        self.stable_s = stable_s
        self.freshness_s = freshness_s
        self.command_id = None
        self.since = None
        self.last_stamp = -math.inf

    def observe(self, command, *, source_now, carrier, cargo, contact):
        if command['handling_id'] != self.command_id:
            self.command_id = command['handling_id']; self.since = None; self.last_stamp = -math.inf
        ct, rt, bt = stamp(contact), stamp(carrier), stamp(cargo)
        if ct <= self.last_stamp:
            return None
        gap = ct-self.last_stamp
        self.last_stamp = ct
        robot = carrier['state']; material = cargo['state']
        rp, bp = robot['pose']['position'], material['pose']['position']
        support = ('handling_shuttle::carrier::deck' if command['operation'] == 'LOAD'
                   else 'cell::body::support')
        forces = defaultdict(float)
        for state in contact['states']:
            pair = {state['collision1_name'], state['collision2_name']}
            if pair != {'material_bin::body::bin', support}:
                continue
            match = re.search(r'time:\s*\[([0-9.]+)\]', state['info'])
            if not match or not state['contact_normals'] or any(n['z'] < .98 for n in state['contact_normals']):
                continue
            forces[float(match.group(1))] += world_vertical_force(
                state['total_wrench']['force'], material['pose']['orientation'])
        force = sum(forces.values())/len(forces) if forces else 0.
        fresh = all(0 <= source_now-t <= self.freshness_s and t >= command['issued_at'] for t in (ct, rt, bt))
        stopped = all(norm(state['twist'][axis]) < .01
                      for state in (robot, material) for axis in ('linear', 'angular'))
        target_y = 0. if command['operation'] == 'LOAD' else .80
        positioned = (math.dist([rp['x'], rp['y']], command['target_xy']) < .015
            and abs(bp['x']-rp['x']) < .025 and abs(bp['y']-target_y) < .035
            and abs(bp['z']-.62) < .015)
        valid = fresh and stopped and positioned and .8*2*9.81 < force < 1.2*2*9.81
        if not valid or gap > self.freshness_s:
            self.since = ct if valid else None
        elif self.since is None:
            self.since = ct
        if self.since is None or ct-self.since < self.stable_s:
            return None
        evidence = dict(command=command, source_now=source_now, stable_since=self.since,
                        contact=contact, carrier=carrier, cargo=cargo, support_force_n=force)
        raw = json.dumps(evidence, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
        evidence_id = hashlib.sha256(raw).hexdigest()
        self.write_evidence(evidence_id, raw)
        return dict(**command, event='HANDLING_COMPLETED', timestamp=ct,
            robot_xy=[rp['x'], rp['y']], stopped=True,
            supported_by=command['robot_id'] if command['operation'] == 'LOAD' else command['node_id'],
            evidence_id=evidence_id)
