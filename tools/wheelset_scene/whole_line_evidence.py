"""Streaming corroboration of whole-line events against native raw feedback.

Event assertions are registered before reading the large sample stream. Only
requested motion windows and XYZ trajectories are retained, keeping a two-job
audit independent of the live controller and bounded in memory.
"""
from collections import defaultdict, deque
import math

from .front_world import envelopes


def rows(directory, name):
    import json
    with (directory / (name + '.jsonl')).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def stopped(obs, now=None, target=None, home=False):
    values = [obs['time_s'], *obs['xyz'], *obs['velocity'], obs['spin'],
              obs['omega'], obs['probe'], obs['probe_velocity']]
    assert all(math.isfinite(v) for v in values)
    assert max(abs(v) for v in obs['velocity']) < .035
    assert abs(obs['omega']) < .025
    if now is not None:
        assert 0 <= now - obs['time_s'] <= .15
    if target is not None:
        assert math.dist(obs['xyz'], target) < .025
    if home:
        assert abs(obs['probe']) < .005 and abs(obs['probe_velocity']) < .015


class Evidence:
    def __init__(self, directory, definitions):
        self.directory = directory
        self.definitions = definitions
        self.requests = defaultdict(list)
        self.windows = defaultdict(list)
        self.retained_checks = []
        self.xyz = {n: [] for n in definitions}
        self.stamps = {n: [] for n in definitions}

    def require(self, name, obs, *, now=None, target=None, stop=True, home=False):
        assert name in self.definitions and obs['entity'] == name
        if stop:
            stopped(obs, now, target, home)
        elif now is not None:
            assert 0 <= now - obs['time_s'] <= .15
        self.requests[name].append((obs, stop))

    def window(self, name, start, end):
        assert start <= end and name in self.definitions
        result = []
        self.windows[name].append((start, end, result))
        return result

    @staticmethod
    def corroborate(actual, requested, stop):
        assert abs(actual['time_s'] - requested['time_s']) <= .05
        dt=abs(actual['time_s']-requested['time_s'])
        tolerance=.002 if stop else .002+3.7*dt
        assert math.dist(actual['xyz'], requested['xyz']) <= tolerance
        probe_tolerance=.005 if stop else .002+max(abs(actual['probe_velocity']),abs(requested['probe_velocity']))*dt+10*dt*dt
        for key, tolerance in [('probe', probe_tolerance), ('spin', .13)]:
            assert abs(actual[key] - requested[key]) <= tolerance
        if stop:
            stopped(actual)
        elif actual['time_s'] == requested['time_s']:
            assert actual == requested

    def read(self, retired, resource_commands):
        requests = {n: deque(sorted(values, key=lambda v: v[0]['time_s']))
                    for n, values in self.requests.items()}
        previous = {}
        acknowledgements = {n: {} for n in self.definitions}
        commands = iter(rows(self.directory, 'native_commands'))
        retire_time = {r['entity']: r['time_s'] for r in retired}
        disabled = []
        count = native_count = effort_count = simultaneous = 0
        previous_time = -math.inf
        maximum_skew = 0.
        for count, sample in enumerate(rows(self.directory, 'samples'), 1):
            stamp = sample['time_s']; observed = sample['observations']
            assert stamp >= previous_time
            previous_time = stamp
            assert set(observed) == set(sample['targets'])
            assert set(observed) <= set(self.definitions)
            assert set(self.definitions) - set(observed) <= set(retire_time)
            assert all(retire_time[n] <= stamp for n in set(self.definitions)-set(observed))
            skew = max(o['time_s'] for o in observed.values())-min(o['time_s'] for o in observed.values())
            maximum_skew = max(maximum_skew, skew); assert skew <= .1
            if len(sample['commands']) > 1:
                simultaneous += 1
            for name, obs in observed.items():
                d = self.definitions[name]; target = sample['targets'][name]
                c = next(commands); native_count += 1
                assert (c['entity'], c['sequence'], c['time_s']) == (name, count, stamp)
                expected_execution=resource_commands[sample['commands'][name]]['execution_id'] if name in sample['commands'] else None
                assert c['execution_id']==expected_execution
                assert len(c['axis_values']) == 2 * len(d['axes'])
                assert set(c['disabled_axes']) <= set(d['axes'])
                for i, axis in enumerate(d['axes']):
                    value = target['xyz']['xyz'.index(axis)]-d['origin']['xyz'.index(axis)] if axis in 'xyz' else target[axis]
                    assert c['axis_values'][2*i] == value
                    assert c['axis_values'][2*i+1] == (0. if axis in c['disabled_axes'] else 1.)
                ack = acknowledgements[name]
                ack[count] = (stamp, c['disabled_axes'])
                if count > 256:
                    ack.pop(count-256, None)
                assert obs['entity'] == name and not obs['servo_fault']
                assert all(math.isfinite(v) for v in [obs['time_s'], *obs['xyz'], *obs['velocity'],
                    obs['probe'], obs['probe_velocity'], obs['spin'], obs['omega']])
                assert max(abs(v) for v in obs['velocity']) <= 2.1
                assert 0 <= stamp-obs['time_s'] <= .1
                sequence = obs['servo_sequence']
                if sequence >= 0:
                    assert sequence in ack and ack[sequence][0] <= obs['time_s']+.05
                    for axis in ack[sequence][1]:
                        assert obs['applied_effort'][axis] == 0.
                assert set(obs['applied_effort']) == set(d['axes'])
                for axis, force in obs['applied_effort'].items():
                    limit = 15000 if d['kind']=='wheelset' and axis in 'xyz' else 1000 if axis=='spin' else 200 if axis=='probe' else 400
                    assert math.isfinite(force) and abs(force) <= limit
                    effort_count += 1
                if c['disabled_axes']:
                    disabled.append(c)
                old = previous.get(name)
                if old:
                    dt = obs['time_s']-old['time_s']; assert dt >= 0
                    assert math.dist(obs['xyz'], old['xyz']) <= 3.7*dt+.015
                queue = requests.get(name, ())
                while queue and queue[0][0]['time_s'] <= obs['time_s']:
                    requested, stop = queue.popleft()
                    actual = min([old, obs] if old else [obs], key=lambda s: abs(s['time_s']-requested['time_s']))
                    self.corroborate(actual, requested, stop)
                for start, end, points in self.windows[name]:
                    if start <= obs['time_s'] <= end and (not points or points[-1]['time_s'] != obs['time_s']):
                        points.append(obs)
                self.xyz[name].append(obs['xyz']); self.stamps[name].append(obs['time_s'])
                previous[name] = obs
            if 'ws001' in observed and 'ws002' in observed:
                assert math.dist(observed['ws001']['xyz'], observed['ws002']['xyz']) >= 2.6
        assert count > 100 and next(commands, None) is None
        assert set(previous) == set(self.definitions)
        for name, queue in requests.items():
            for requested, stop in queue:
                self.corroborate(previous[name], requested, stop)
        return dict(joint_sample_groups=count, native_command_records=native_count,
            requested_force_records=effort_count, simultaneous_action_samples=simultaneous,
            maximum_native_source_skew_s=maximum_skew), disabled

    def sweeps(self, boxes):
        import numpy as np
        bounds = np.array([b['bounds'] for b in boxes]); checked = 0
        for name, trajectory in self.xyz.items():
            if self.definitions[name]['kind'] == 'ram':
                continue
            xyz = np.array(trajectory)
            kind = 'wheelset' if self.definitions[name]['kind']=='wheelset' else 'part'
            for center, half in envelopes(kind):
                low = bounds[:,:3]-np.array(half)-.01; high = bounds[:,3:]+np.array(half)+.01
                for first in range(0, len(xyz)-1, 128):
                    a = xyz[first:first+128,None,:]+center
                    b = xyz[first+1:first+129,None,:]+center
                    a = a[:len(b)]; d = b-a; zero = np.abs(d)<1e-12
                    safe = np.where(zero, 1., d)
                    u = (low[None,:,:]-a)/safe; v = (high[None,:,:]-a)/safe
                    lower = np.where(zero,-np.inf,np.minimum(u,v)); upper = np.where(zero,np.inf,np.maximum(u,v))
                    outside = np.any(zero & ((a<low[None,:,:]) | (a>high[None,:,:])),axis=2)
                    hit = (np.maximum(0.,np.max(lower,axis=2)) <= np.minimum(1.,np.min(upper,axis=2))) & ~outside
                    if np.any(hit):
                        row, col = np.argwhere(hit)[0]
                        raise AssertionError(f"Measured cargo intersects {boxes[col]['id']}: {name} sample {first+row}")
                    checked += len(a)*len(boxes)
        return checked
