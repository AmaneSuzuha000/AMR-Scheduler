"""Rehash causal corruptions in isolated copies of a completed recovery run."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .build import ROOT
from .verify_navigation_recovery import verify


MUTATIONS = ('missing_native_terminal', 'moving_stop', 'old_id_accepted',
    'reused_route_id', 'wrong_revision', 'early_arrival', 'released_road',
    'changed_task', 'residual_native_output', 'velocity_still_delivered',
    'lease_still_delivered', 'scan_not_stale', 'missing_live_lidar',
    'lost_stopped_feedback', 'wrong_fault_cause')


def check(source, output):
    source = Path(source); baseline = verify(source, write=False); cases = []
    for mutation in MUTATIONS:
        with tempfile.TemporaryDirectory(prefix='navigation-recovery-causal-', dir=ROOT/'out') as temporary:
            directory = Path(temporary)/'evidence'
            shutil.copytree(source, directory, copy_function=os.link)
            report = json.loads((directory/'result.json').read_text()); edited = {}
            def records(name):
                if name not in edited:
                    edited[name] = [json.loads(line) for line in (directory/name).read_text().splitlines()]
                return edited[name]
            rows = records('fleet/scout_a-000.jsonl')
            if mutation == 'missing_native_terminal':
                rows.remove(next(r for r in rows if r['kind'] == 'native_result' and r['status'] == 5))
            elif mutation == 'moving_stop':
                stop = next(r for r in rows if r['kind'] == 'feedback' and r['message']['event'] == 'STOPPED')
                next(r for r in reversed(rows) if r['kind'] == 'guard' and r['wall'] < stop['wall'])['actual_speed'] = .1
            elif mutation == 'old_id_accepted':
                next(r for r in rows if r['kind'] == 'recovery' and not r['result']['accepted'])['result']['accepted'] = True
            elif mutation in ('reused_route_id', 'wrong_revision', 'changed_task'):
                r = next(r for r in rows if r['kind'] == 'recovery' and r['result']['accepted'])
                c = r['result']['command']
                if mutation == 'reused_route_id': c['route_id'] = r['previous_route_id']
                elif mutation == 'wrong_revision': c['revision'] = 1
                else: c['task_id'] = 'another_task'
            elif mutation in ('early_arrival', 'lost_stopped_feedback'):
                stop = next(r for r in rows if r['kind'] == 'feedback' and r['message']['event'] == 'STOPPED')
                if mutation == 'early_arrival': stop['message']['event'] = 'ARRIVED'
                else: rows.remove(stop)
            elif mutation == 'released_road':
                frames = records('fleet/fleet-events-000.jsonl')
                frame = next(r for r in frames if r['kind'] == 'frame' and r['value']['executions']['scout_a']['fault'])
                del frame['value']['physical_holds']['edges']['west->bay_west']
            elif mutation == 'wrong_fault_cause':
                next(r for r in rows if r['kind'] == 'command' and r['value']['kind'] == 'CANCEL')['value']['reason'] = 'unrelated_fault'
            else:
                obs = records('observations.jsonl'); start = report['start_monotonic_s']
                events = report['fault_campaign']['events']
                index = 1 if mutation == 'lease_still_delivered' else 2 if mutation in ('scan_not_stale', 'missing_live_lidar') else 0
                t0 = events[6*index]['wall']; t1 = events[6*index+2]['wall']
                if mutation == 'residual_native_output':
                    next(r for r in obs if r['kind'] == 'guard' and r['robot'] == 'scout_a'
                         and r['message']['reason'] == 'velocity command timeout')['message']['output_vx'] = .1
                elif mutation == 'velocity_still_delivered':
                    next(r for r in obs if r['kind'] == 'velocity_relay' and r['robot'] == 'scout_a'
                         and t0 <= start+r['wall_s'] <= t1)['muted'] = False
                elif mutation == 'lease_still_delivered':
                    lease = deepcopy(next(r for r in reversed(rows) if r['kind'] == 'lease' and r['wall'] < t0))
                    lease['wall'] = t0+.1; lease['seq'] += 1; rows.append(lease); rows.sort(key=lambda r: r['wall'])
                elif mutation == 'scan_not_stale':
                    stale = next(r for r in rows if r['kind'] == 'localization_health' and r['reason'] == 'scan stale')
                    stale['sources']['scan']['received'] = stale['wall']
                else:
                    obs[:] = [r for r in obs if not(r['kind'] == 'scan' and r['robot'] == 'scout_a'
                              and t0 <= start+r['received_wall_s'] <= t1)]
            for name, values in edited.items():
                path = directory/name
                path.unlink()  # Never edit the original inode shared by the copy.
                path.write_text(''.join(json.dumps(r)+'\n' for r in values))
                report['files_sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            path = directory/'result.json'; path.unlink(); path.write_text(json.dumps(report)+'\n')
            try:
                verify(directory, write=False)
            except AssertionError:
                cases.append(dict(mutation=mutation, rejected=True, hashes_recomputed=True))
            else:
                raise AssertionError('Causal corruption accepted: '+mutation)
    result = dict(passed=True, source=source.name, baseline_audit=baseline, cases=cases)
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    print(json.dumps(check(args.directory, args.output)))


if __name__ == '__main__':
    main()
