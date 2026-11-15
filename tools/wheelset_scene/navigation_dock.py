"""Validate a localized arrival plus an independent measured dock receipt.

The independent state query is a receiving-side check. It never supplies a
pose, task completion or velocity to the navigation estimator or scheduler.
"""
import math


def stamp(message):
    s = message['header']['stamp']
    return s['sec']+s['nanosec']*1e-9


def localized_dock_valid(command, observation, proof, run_id, now):
    try:
        if command['source'] != 'localized' or command['frame_id'] != 'map' or command['time_base'] != 'ros_sim':
            return False
        feedback = proof['feedback']; result = proof['feedback_result']; guard = proof['guard']
        query, state = proof['state_query'], proof['state_reply']
        identity = ('scene_id', 'run_id', 'robot_id', 'frame_id', 'time_base', 'source', 'task_id', 'route_id')
        if (command['run_id'] != run_id or any(feedback[k] != command[k] for k in identity)
                or feedback['event'] != 'ARRIVED' or result != dict(accepted=True, state='arrived')
                or type(feedback['seq']) is not int or feedback['seq'] < 0):
            return False
        if (not guard['guard_id'] or guard['state'] != 'latched' or not guard['stopped']
                or any(guard[k] != command[k] for k in ('run_id', 'route_id', 'revision'))
                or guard['output_vx'] != 0 or guard['output_wz'] != 0
                or not 0 <= guard['actual_speed'] < .01 or not abs(guard['actual_wz']) < .02):
            return False
        if (query['service'] != '/wheelset_review/get_entity_state'
                or query['request'] != dict(name=command['robot_id'], reference_frame='world')
                or type(query['query_id']) is not int or query['query_id'] < 0
                or not state['success'] or state['header']['frame_id'] != 'world'
                or state['state']['name'] not in ('', command['robot_id'])):
            return False
        raw = state['state']; p = raw['pose']['position']; q = raw['pose']['orientation']
        v, w = raw['twist']['linear'], raw['twist']['angular']
        finite = [p[k] for k in ('x', 'y', 'z')]+[q[k] for k in ('x', 'y', 'z', 'w')]
        finite += [v[k] for k in ('x', 'y', 'z')]+[w[k] for k in ('x', 'y', 'z')]
        if len(feedback['pose']) != 3 or len(observation['pose']) != 3:
            return False
        finite += [*feedback['pose'], *observation['pose'], observation['linear_speed'],
                   observation['angular_speed'], observation['sim_time_s'], feedback['timestamp'],
                   now, guard['actual_speed'], guard['actual_wz']]
        if not all(math.isfinite(value) for value in finite):
            return False
        if abs(sum(q[k]*q[k] for k in ('x', 'y', 'z', 'w'))-1) > 1e-3:
            return False
        yaw = math.atan2(2*(q['w']*q['z']+q['x']*q['y']), 1-2*(q['y']*q['y']+q['z']*q['z']))
        if (observation['pose'][:2] != [p['x'], p['y']]
                or abs(math.atan2(math.sin(observation['pose'][2]-yaw), math.cos(observation['pose'][2]-yaw))) > 1e-12
                or observation['sim_time_s'] != stamp(state)
                or observation['linear_speed'] != math.hypot(v['x'], v['y'])
                or observation['angular_speed'] != abs(w['z'])):
            return False
        if not all(0 <= now-t <= .15 for t in (stamp(state), stamp(guard), feedback['timestamp'])):
            return False
        return (math.dist(feedback['pose'][:2], command['target_xy']) <= .10
                and math.dist(feedback['pose'][:2], observation['pose'][:2]) <= .15)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return False
