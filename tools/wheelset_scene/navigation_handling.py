"""Independent contact evidence for task-specific SCOUT bin handoffs.

The state service is used only by the equipment observer. It supplies no
localization, navigation progress or motion command to the fleet scheduler.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import re

from .verify_contact_conveyor import norm, stamp, world_vertical_force


_UNSPECIFIED_EXECUTION_POSE = object()


def checked_state(observation, entity):
    """Keep the request identity: Gazebo may return an empty state.name."""
    request, reply = observation['request'], observation['reply']
    if (request != dict(name=entity, reference_frame='world') or reply['success'] is not True
            or reply['header']['frame_id'] != 'world'
            or reply['state'].get('name', '') not in ('', entity)):
        raise ValueError('independent state entity or frame mismatch')
    state = reply['state']; q = state['pose']['orientation']
    values = [v for part in ('position', 'orientation') for v in state['pose'][part].values()]
    values += [v for axis in ('linear', 'angular') for v in state['twist'][axis].values()]
    if (not all(type(v) in (int, float) and math.isfinite(v) for v in values)
            or abs(sum(q[key]**2 for key in ('x', 'y', 'z', 'w'))-1.) > .001):
        raise ValueError('nonfinite state or invalid quaternion')
    return reply


class DockedBinObserver:
    """Confirm a configured cargo on the correct support with a retracted finger."""

    def __init__(self, profile, write_evidence):
        if (profile['world_to_map'] != dict(translation_m=[0., 0., 0.], yaw_rad=0.)
                or profile['source'] != 'gazebo_scout_bin_contact'):
            raise ValueError('explicit identity map/world transform and equipment source required')
        for name in ('freshness_s', 'stable_support_s', 'dock_position_tolerance_m',
                     'dock_yaw_tolerance_rad', 'cargo_position_tolerance_m'):
            if type(profile[name]) not in (int, float) or not math.isfinite(profile[name]) or profile[name] <= 0:
                raise ValueError('positive finite contact tolerances required')
        if not 0 < profile['geometry']['bin_mass_kg'] < math.inf:
            raise ValueError('positive finite bin mass required')
        self.profile = deepcopy(profile); self.write_evidence = write_evidence
        self.command = None; self.since = None; self.last_stamp = -math.inf; self.emitted = False
        self.last_wait_reason = None

    def observe(self, command, *, source_now, carrier, cargo, contact, actuator,
                execution_pose=_UNSPECIFIED_EXECUTION_POSE, execution_tolerance_m=None):
        """Bad/gapped observations reset continuity; old commands cannot reuse it."""
        if command != self.command:
            self.command = deepcopy(command)
            self.since = None; self.last_stamp = -math.inf; self.emitted = False
        self.last_wait_reason = None
        if self.emitted:
            return None
        try:
            spec = self.profile['tasks'][command['task_id']]
            station = spec['stations'][command['operation']]
            robot = command['robot_id']; geometry = self.profile['geometry']
            if (command['scene_id'] != self.profile['scene_id'] or robot not in self.profile['robots']
                    or command['source'] != self.profile['source'] or command['frame_id'] != 'map'
                    or command['time_base'] != 'ros_sim' or station['node_id'] != command['node_id']
                    or station['target_xy'] != command['target_xy']
                    or contact['header']['frame_id'] != 'world'
                    or contact['_topic'] != '/scout_bin/'+command['task_id']+'/bin_contacts'
                    or actuator['_topic'] != '/scout_bin/'+robot+'/joint_states'):
                raise ValueError('handling command does not name this dock')
            carrier_observation, cargo_observation = carrier, cargo
            carrier = checked_state(carrier, robot+'::'+geometry['carrier_frame'])
            cargo = checked_state(cargo, spec['cargo_entity'])
            ct, rt, bt, at = stamp(contact), stamp(carrier), stamp(cargo), stamp(actuator)
            if not all(type(v) in (int, float) and math.isfinite(v) for v in
                       (source_now, ct, rt, bt, at, command['issued_at'])):
                raise ValueError('finite source times required')
            if ct <= self.last_stamp:
                return None
            gap = ct-self.last_stamp; self.last_stamp = ct
            fresh = all(0 <= source_now-t <= self.profile['freshness_s'] and t >= command['issued_at']
                        for t in (ct, rt, bt, at))
            rstate, bstate = carrier['state'], cargo['state']
            rp, bp = rstate['pose']['position'], bstate['pose']['position']
            q = rstate['pose']['orientation']
            yaw = math.atan2(2*(q['w']*q['z']+q['x']*q['y']), 1-2*(q['y']**2+q['z']**2))
            tilt = 1-2*(q['x']**2+q['y']**2)
            stopped = all(norm(state['twist'][axis]) < .01
                          for state in (rstate, bstate) for axis in ('linear', 'angular'))
            index = actuator['name'].index('bin_transfer_stroke')
            retracted = (abs(actuator['position'][index]) < .005 and abs(actuator['velocity'][index]) < .01)
            fault = actuator['position'][actuator['name'].index(robot+'_servo_fault')]
            sequence = actuator['position'][actuator['name'].index(robot+'_servo_sequence')]
            retracted = (retracted and fault == 0. and sequence >= actuator['_retraction_sequence'] >= 0)
            expected = station['bin_world_xyz']
            support = station['support_collision']
            if command['operation'] == 'LOAD':
                x, y, z = geometry['loaded_bin_local_m']
                expected = [rp['x']+x*math.cos(yaw)-y*math.sin(yaw),
                            rp['y']+x*math.sin(yaw)+y*math.cos(yaw), rp['z']+z]
                support = robot+'::handling_deck::support'
            positioned = (math.dist([rp['x'], rp['y']], station['target_xy']) <= self.profile['dock_position_tolerance_m']
                and abs(math.atan2(math.sin(yaw-station['yaw_rad']), math.cos(yaw-station['yaw_rad']))) <= self.profile['dock_yaw_tolerance_rad']
                and tilt > .999 and math.dist([bp['x'], bp['y'], bp['z']], expected) <= self.profile['cargo_position_tolerance_m'])
            forces = defaultdict(float)
            for state in contact['states']:
                pair = {state['collision1_name'], state['collision2_name']}
                if pair != {spec['cargo_collision'], support}:
                    continue
                match = re.search(r'time:\s*\[([0-9.]+)\]', state['info'])
                if (not match or not state['contact_normals']
                        or not all(math.isfinite(n['z']) and n['z'] > .98 for n in state['contact_normals'])):
                    continue
                physics_time = float(match.group(1))
                if not command['issued_at'] <= physics_time <= ct or ct-physics_time > self.profile['freshness_s']:
                    continue
                forces[physics_time] += world_vertical_force(state['total_wrench']['force'], bstate['pose']['orientation'])
            force = sum(forces.values())/len(forces) if forces else 0.
            weight = geometry['bin_mass_kg']*9.81
            valid = fresh and stopped and retracted and positioned and .8*weight < force < 1.2*weight
            if not valid or gap > self.profile['freshness_s']:
                self.since = ct if valid else None
            elif self.since is None:
                self.since = ct
            if self.since is None or ct-self.since < self.profile['stable_support_s']:
                return None
        except (KeyError, ValueError, IndexError, TypeError, OverflowError, AssertionError):
            self.since = None
            return None
        pose_evidence = None
        if execution_pose is not _UNSPECIFIED_EXECUTION_POSE:
            try:
                pose_timestamp = execution_pose['timestamp']
                pose_xy = (execution_pose['x'], execution_pose['y'])
                tolerance = execution_tolerance_m
                if (not all(type(value) in (int, float) and math.isfinite(value)
                            for value in (*pose_xy, pose_timestamp, tolerance))
                        or tolerance <= 0
                        or not command['issued_at'] <= pose_timestamp <= source_now
                        or abs(pose_timestamp-ct) > self.profile['freshness_s']
                        or math.dist(pose_xy, [rp['x'], rp['y']]) > tolerance
                        or math.dist(pose_xy, command['target_xy']) > tolerance):
                    self.last_wait_reason = 'localized_pose_not_aligned'
                    return None
                pose_evidence = dict(execution_pose)
            except (KeyError, TypeError, ValueError, OverflowError):
                self.last_wait_reason = 'localized_pose_invalid'
                return None
        evidence = dict(command=command, source_now=source_now, stable_since=self.since,
            carrier=carrier_observation, cargo=cargo_observation, contact=contact, actuator=actuator,
            support_force_n=force, expected_support=support, profile=self.profile)
        if pose_evidence is not None:
            evidence['execution_pose'] = pose_evidence
            evidence['execution_tolerance_m'] = float(execution_tolerance_m)
        raw = json.dumps(evidence, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
        identity = hashlib.sha256(raw).hexdigest()
        self.write_evidence(identity, raw)  # Failed persistence must prevent acknowledgement.
        self.emitted = True
        return dict(**command, event='HANDLING_COMPLETED', timestamp=ct, robot_xy=[rp['x'], rp['y']],
            stopped=True, supported_by=robot if command['operation'] == 'LOAD' else command['node_id'],
            evidence_id=identity)
