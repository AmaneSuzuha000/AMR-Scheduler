"""Replay explicit command/lease/adapter-input interruptions and retained recovery.

The independent observer continues receiving the real sensor stream. The third
fault concerns the dispatch adapter's scan health input, not the physical LiDAR.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from tools.live_bridge.sensor_health import SensorHealth
from .verify_navigation_motion import (read_rows, audit_robot, audit_arrival_guards,
    audit_frames, audit_measurements)
from .verify_navigation_readiness import verify as verify_readiness


CHANNELS = ('velocity_link', 'mission_lease', 'adapter_scan_input')
REASONS = ('execution guard: velocity command timeout',
           'execution guard: mission lease timeout', 'localization health: scan stale')
EVENTS = ('INPUT_INTERRUPTED', 'FAULT_LATCHED', 'INPUT_RESTORED',
          'OLD_ROUTE_REJECTED', 'RECOVERY_ATTEMPT', 'RECOVERY_AUTHORIZED')


def audit_campaign(observations, robot_rows, audits, frames, start):
    events = [{k: v for k, v in r.items() if k not in ('kind', 'wall_s')}
              for r in observations if r['kind'] == 'fault_campaign']
    assert len(events) == 3*len(EVENTS)
    a = audits['scout_a']; rows = robot_rows['scout_a']; previous_end = start
    results = []
    for index, channel in enumerate(CHANNELS):
        block = events[index*len(EVENTS):(index+1)*len(EVENTS)]
        assert [r['event'] for r in block] == list(EVENTS)
        assert all(r['index'] == index and r['robot'] == 'scout_a' for r in block)
        interrupt, latch, restored, rejected, attempt, authorized = block
        t0, t1, t2 = interrupt['wall'], restored['wall'], authorized['wall']
        assert previous_end < t0 < latch['wall'] < t1 <= t2 and .9 <= t1-t0 < 2.
        assert t2-t0 < 15. and interrupt['channel'] == restored['channel'] == channel
        c = interrupt['command']; token = c['route_id']
        assert c == a['routes'][token]['command'] == latch['command']
        assert latch['reason'] == REASONS[index] == a['cancelled'][token]['value']['reason']
        assert rejected['result'] == dict(accepted=False, reason='recovery action mismatch')
        assert attempt['result']['accepted'] and attempt['result']['command'] == authorized['command']
        assert authorized['command'] == a['recoveries'][token]['result']['command']
        assert 0 <= a['recoveries'][token]['wall']-t2 < .25
        assert interrupt['pins'] == latch['pins'] == authorized['pins']
        retained = interrupt['pins']
        held = [r for r in frames if t0 <= r['wall'] <= t2]
        assert len(held) >= 5
        for row in held:
            f = row['value']; gate = f['executions']['scout_a']
            assert gate['route_id'] == token and gate['task_id'] == c['task_id']
            pins = {g: {k: v for k, v in members.items() if 'scout_a' in v}
                    for g, members in f['physical_holds'].items()}
            assert pins == retained, 'retained physical occupancy was released during the fault'
        # Observe motion at injection, not a timeout injected into an idle robot.
        before = [r for r in rows if r['kind'] == 'guard' and t0-.2 <= r['wall'] <= t0]
        assert before and before[-1]['state'] == 'armed' and before[-1]['actual_speed'] > .05
        guards = [r for r in rows if r['kind'] == 'guard' and t0 <= r['wall'] <= t1
                  and r['route_id'] == token and r['state'] == 'latched']
        assert guards and all(r['output_vx'] == r['output_wz'] == 0 for r in guards)
        relays = [r for r in observations if r['kind'] == 'velocity_relay'
                  and r['robot'] == 'scout_a' and t0 <= start+r['wall_s'] <= t1]
        assert relays
        if index == 0:
            assert all(r['muted'] for r in relays) and any(abs(r['vx']) > .05 for r in relays)
            causal = [r for r in guards if r['reason'] == 'velocity command timeout']
            assert causal and causal[0]['command_age'] > .25 and causal[0]['lease_age'] <= .5
            assert .15 <= causal[0]['wall']-t0 <= .5
        elif index == 1:
            assert all(not r['muted'] for r in relays)
            assert not any(r['kind'] == 'lease' and t0 <= r['wall'] <= t1 for r in rows)
            causal = [r for r in guards if r['reason'] == 'mission lease timeout']
            assert causal and causal[0]['lease_age'] > .5 and causal[0]['command_age'] <= .25
            assert .35 <= causal[0]['wall']-t0 <= .75
        else:
            assert all(not r['muted'] for r in relays)
            stale = [r for r in rows if r['kind'] == 'localization_health' and t0 <= r['wall'] <= t1]
            assert len(stale) == 1 and stale[0]['reason'] == 'scan stale'
            health = SensorHealth(); row = stale[0]
            for name, sample in row['sources'].items():
                health.observe(name, sample['stamp'], sample['received'], sample['valid'], sample['reason'])
            assert health.reason(row['sim'], row['wall']) == 'scan stale'
            assert row['sources']['scan']['received'] < t0
            scans = [r for r in observations if r['kind'] == 'scan' and r['robot'] == 'scout_a'
                     and t0 <= start+r['received_wall_s'] <= t1]
            assert len(scans) >= 3 and all(r['source_s'] > row['sources']['scan']['stamp'] for r in scans)
            fresh = [r for r in rows if r['kind'] == 'localization_health' and t1 <= r['wall'] <= t2]
            assert fresh and fresh[-1]['reason'] is None
            assert .15 <= row['wall']-t0 <= .5
        # Native causes must also appear in an independently subscribed guard.
        if index < 2:
            raw = [r for r in observations if r['kind'] == 'guard' and r['robot'] == 'scout_a'
                   and r['message']['route_id'] == token and r['message']['reason'] == causal[0]['reason']]
            assert any(r['message']['guard_id'] == a['guard_id'] and
                       all(r['message'][k] == causal[0][k] for k in
                           ('run_id', 'route_id', 'revision', 'state', 'reason', 'actual_speed',
                            'actual_wz', 'output_vx', 'output_wz', 'command_age', 'lease_age')) and
                       abs(start+r['received_wall_s']-causal[0]['wall']) < .25 for r in raw)
        results.append(dict(channel=channel, route_id=token,
            resumed_route_id=authorized['command']['route_id'], latch_wall_s=latch['wall']-t0,
            stopped_wall_s=a['stops'][token]['wall']-t0, recovery_wall_s=t2-t0,
            retained_frames=len(held)))
        previous_end = t2
    assert set(a['cancelled']) == {r['route_id'] for r in results}
    return events, results


def verify(directory, *, write=True, full_delivery=False):
    directory = Path(directory); verify_readiness(directory, write=write)
    result = json.loads((directory/'result.json').read_text())
    assert result['mode'] == result['fleet']['case'] == 'faults'
    if full_delivery:
        assert result['whole_line'], 'full recovery delivery requires the same-world workshop schema'
    assert result['fault_campaign']['passed'] and result['fault_campaign']['recovered'] == 3
    assert result['fault_campaign']['channels'] == list(CHANNELS)
    observations = read_rows(directory/'observations.jsonl'); layout = result['network']
    robot_rows = {}; audits = {}
    for robot in ('scout_a', 'scout_b'):
        root = ET.parse(directory/'models'/(robot+'.sdf')).getroot()
        guard = root.find(".//plugin[@filename='libamr_motion_guard.so']")
        assert 'cmd_vel:=guard_input_cmd_vel' in [r.text for r in guard.findall('ros/remapping')]
        paths = list((directory/'fleet').glob(robot+'-*.jsonl')); assert len(paths) == 1
        robot_rows[robot] = read_rows(paths[0])
        audits[robot] = audit_robot(robot_rows[robot], robot, layout, recovery=robot == 'scout_a')
    assert audits['scout_a']['guard_id'] != audits['scout_b']['guard_id']
    assert len({v['command']['run_id'] for a in audits.values() for v in a['routes'].values()}) == 1
    audit_arrival_guards(observations, audits, result['start_monotonic_s'], include_stops=True)
    paths = list((directory/'fleet').glob('fleet-events-*.jsonl')); assert len(paths) == 1
    frames, waits = audit_frames(read_rows(paths[0]), audits, layout)
    markers = list(observations.only('fleet_enabled', 'fleet_completed'))
    enabled = [r for r in markers if r['kind'] == 'fleet_enabled']
    completed = [r for r in markers if r['kind'] == 'fleet_completed']
    assert len(enabled) == len(completed) == 1 and completed[0]['wall_s'] > enabled[0]['wall_s']
    measurement = audit_measurements(observations, frames, audits, layout,
        (enabled[0]['wall_s'], completed[0]['wall_s']), result['start_monotonic_s'])
    # The campaign is short even in a long joint run. Retain only its bounded
    # corroborating interval; full motion and sensor checks above remain streamed.
    campaign = list(observations.only('fault_campaign'))
    assert len(campaign) == 3*len(EVENTS)
    campaign_end = max(r['wall_s'] for r in campaign)+.25
    campaign_start = min(r['wall_s'] for r in campaign)-.25
    campaign_rows = []
    for row in observations:
        if row['wall_s'] > campaign_end: break
        if row['wall_s'] >= campaign_start: campaign_rows.append(row)
    bounded_robots = {}
    for robot, records in robot_rows.items():
        bounded_robots[robot] = []
        for row in records:
            if row['wall'] > result['start_monotonic_s']+campaign_end: break
            if row['wall'] >= result['start_monotonic_s']+campaign_start:
                bounded_robots[robot].append(row)
    events, recovered = audit_campaign(campaign_rows, bounded_robots, audits, frames, result['start_monotonic_s'])
    assert events == result['fault_campaign']['events']
    expected_tasks = (dict(inspection_consumables='completed', small_tools='completed') if full_delivery
                      else dict(navigation_short_a='completed', navigation_short_b='completed'))
    assert result['fleet']['tasks'] == expected_tasks
    assert frames[-1]['value']['task_counts'] == {'completed': 2}
    destinations = ('receiving', 'supply') if full_delivery else ('west', 'east')
    assert audits['scout_a']['final_node'] == destinations[0] and audits['scout_b']['final_node'] == destinations[1]
    assert len(audits['scout_a']['routes']) == (6 if full_delivery else 4)
    assert len(audits['scout_b']['routes']) == (3 if full_delivery else 1)
    if full_delivery: assert waits > 0
    for robot, expected in zip(('scout_a', 'scout_b'), destinations):
        assert result['fleet']['gates'][robot]['state'] == 'idle' and result['fleet']['gates'][robot]['fault'] is None
        assert result['fleet']['pins']['nodes'].get(expected) == [robot]
    scope = 'sensor_navigation_retained_full_delivery_recovery' if full_delivery else 'sensor_navigation_retained_short_route_recovery'
    audit = dict(passed=True, scope=scope,
        recovered=recovered, owner_frames=len(frames), opposing_wait_frames=waits, **measurement,
        result_sha256=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest(),
        claim_boundary='Three explicit software input interruptions, native cancel/stop, old-ID rejection, '
        'retained task and corridor, new route/revision, sampled physical motion and final completion. '
        'The scan fault affects adapter health input only. Physical LiDAR failure, process inventory, '
        'continuous collision safety and industrial acceptance are outside this audit.')
    if write:
        (directory/'navigation_recovery_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('directory', type=Path)
    print(json.dumps(verify(parser.parse_args().directory)))


if __name__ == '__main__':
    main()
