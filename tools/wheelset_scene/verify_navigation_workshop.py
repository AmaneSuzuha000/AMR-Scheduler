"""Independently audit sensor fleet, physical receiving, inventory and 18-stage work.

This entry point accepts only the explicit same-world run schema. It does not
substitute truth fleet success or a separate navigation run for joint evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .build import SCENE
from .evidence import build_fingerprint
from .finishing_flow import FULL_STAGES
from .finishing_world import finishing_static
from .whole_line_evidence import Evidence, rows
from .whole_line_worker_audit import verify_workers
from .verify_whole_line import verify_resources, verify_storage, verify_supply
from .verify_navigation_motion import verify as verify_motion, audit_robot, read_rows


def audit_dock_guard(query, proof, observations, robot_audit):
    """Bind the receiving proof to both independently recorded guard streams."""
    robot = query['robot']; guard = proof['guard']; token = query['command']['route_id']
    adapter_guard = robot_audit['arrival_guards'][token]
    keys = ('guard_id', 'run_id', 'route_id', 'revision', 'state', 'stopped',
            'actual_speed', 'actual_wz', 'output_vx', 'output_wz', 'command_age', 'lease_age')
    assert all(guard[k] == adapter_guard[k] for k in keys), 'receipt guard differs from adapter arrival'
    matches = [o for o in observations if o['kind'] == 'guard' and o.get('robot') == robot
               and o['message'] == guard and abs(o['received_wall_s']-query['wall_s']) < .25]
    assert matches, 'receipt guard lacks an independent raw observation'


def verify(directory, *, write=True, sweep=True, recovery=False):
    directory = Path(directory); top = json.loads((directory/'result.json').read_text())
    assert top['whole_line'] and top['mode'] == ('faults' if recovery else 'delivery'), 'explicit joint run schema required'
    if recovery:
        from .verify_navigation_recovery import verify as verify_recovery
        motion = verify_recovery(directory, write=write, full_delivery=True)
    else:
        motion = verify_motion(directory, write=write)
    process = directory/'process'; r = json.loads((process/'result.json').read_text())
    workshop = top['workshop']
    assert workshop['error'] is None and workshop['process_report'] == r and r['passed']
    assert r['runtime_error'] is None and r['mode'] == 'whole_line'
    assert r['case'] == ('stalled_recovery' if recovery else 'normal')
    assert r['build_fingerprint'] == top['build_fingerprint'] == build_fingerprint()
    for name, digest in r['data_sha256'].items():
        hasher = hashlib.sha256()
        with (process/name).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''): hasher.update(chunk)
        assert hasher.hexdigest() == digest
    scene = directory/'scene_snapshot'
    metadata = json.loads((scene/'whole_line_metadata.json').read_text())
    assert metadata == r['metadata'] and metadata['jobs'] == ['ws001', 'ws002']
    assert metadata['stages'] == json.loads(json.dumps(FULL_STAGES))
    assert r['job_count'] == 2 and set(r['completed']) == set(metadata['jobs'])
    preflight = json.loads((scene/'whole_line_clearance.json').read_text())
    assert preflight['passed'] and all(not p['hits'] for p in preflight['checks'])
    source_scene = json.loads(SCENE.read_text(encoding='utf-8'))
    source_scene['instances'] = [i for i in source_scene['instances'] if i['id'] not in metadata['removed_static_snapshot_ids']]
    boxes, sources = finishing_static(source_scene)
    assert boxes == metadata['static_obstacles'] and sources == metadata['collision_source_sha256']
    native = json.loads((scene/'native_axis/build_manifest.json').read_text())
    assert native['passed'] and hashlib.sha256((scene/'native_axis/libwheelset_axis_servo.so').read_bytes()).hexdigest() == native['library_sha256']
    for name, digest in native['source_sha256'].items():
        assert hashlib.sha256((scene/'native_axis/sources'/name).read_bytes()).hexdigest() == digest
    world = ET.parse(scene/'navigation.world').getroot().find('world')
    manifest = json.loads((scene/'manifest.json').read_text())
    if 'review_camera_enabled' in manifest:
        assert (world.find("model[@name='review_camera']") is not None) == manifest['review_camera_enabled']
    if 'physics_island_threads' in manifest:
        threads = manifest['physics_island_threads']
        assert type(threads) is int and 0 <= threads <= 16
        physics = world.find('physics'); solver = physics.find('ode/solver')
        assert physics.findtext('max_step_size') == '.002' and physics.findtext('real_time_update_rate') == '500'
        assert solver.findtext('type') == 'quick' and solver.findtext('iters') == '400'
        assert solver.findtext('sor') == '1'
        assert solver.findtext('island_threads') == (str(threads) if threads else None)
        assert physics.findtext('ode/constraints/cfm') == '1e-09'
        assert physics.findtext('ode/constraints/erp') == '0.8'
    definitions = metadata['definitions']; assert len(definitions) == 23
    assert world.find("plugin[@filename='libwheelset_fleet_truth.so']") is None
    for name, d in definitions.items():
        model = world.find(f"model[@name='{name}']"); assert model is not None
        assert list(map(float, model.findtext('pose').split())) == [*d['origin'], 0., 0., 0.]
        assert not model.findall('.//collision') and all(link.findtext('gravity') == 'false' for link in model.findall('link'))
        assert len(model.findall('joint')) == (6 if d['kind'] == 'wheelset' else 2 if d['kind'] == 'ram' else 5)
        assert len(model.findall("plugin[@filename='libwheelset_axis_servo.so']")) == 1
        assert model.findtext("plugin[@filename='libwheelset_axis_servo.so']/lease_s") == '1.0'
    evidence = Evidence(process, definitions)
    workers, deferred, summaries = verify_workers(process, r, metadata, evidence)
    resource = list(rows(process, 'events'))
    commands, ownership = verify_resources(resource, metadata['jobs'], workers, evidence)
    assert resource[-1]['snapshot'] == r['final_snapshot']
    retired, storage = verify_storage(process, r, evidence, workers)
    audits = {robot: audit_robot(read_rows(directory/'fleet'/(robot+'-000.jsonl')), robot, top['network'],
                               recovery=recovery and robot == 'scout_a')
              for robot in ('scout_a', 'scout_b')}
    supply_report = dict(workshop, routes=[v['command'] for a in audits.values() for v in a['routes'].values()])
    assert workshop['dock_source'] == 'localized'
    assert workshop['received'] == ['inspection_consumables', 'small_tools']
    supply = verify_supply(directory, supply_report, evidence, workers, localized_audits=audits)
    observations = read_rows(directory/'observations.jsonl')
    dock_records = list(observations.only('dock_state_query', 'localized_dock_receipt'))
    queries = [o for o in dock_records if o['kind'] == 'dock_state_query']
    receipts = [o for o in dock_records if o['kind'] == 'localized_dock_receipt']
    saved = [e for e in rows(directory, 'supply_events') if e['event'] == 'DOCK_RECEIPT']
    assert len(queries) == len(receipts) == len(saved) == 2
    assert {q['proof']['state_query']['query_id'] for q in queries} == {0, 1}
    for receipt in receipts:
        assert receipt['accepted']
        query = next(q for q in queries if q['task_id'] == receipt['task_id'])
        assert query['command'] == receipt['command'] and query['wall_s'] <= receipt['wall_s']
        assert query['proof'] == {k: v for k, v in receipt['proof'].items() if k != 'state_reply'}
        audit_dock_guard(query, receipt['proof'], observations, audits[query['robot']])
        event = next(e for e in saved if e['delivery']['task_id'] == receipt['task_id'])
        assert event['dock_proof'] == receipt['proof'] and event['observation'] == receipt['observation']
        assert event['source_time_s'] == receipt['source_now']
    poses = list(rows(process, 'pose_checks')); assert len(poses) > 50 and {p['entity'] for p in poses} == set(definitions)
    for p in poses:
        obs = p['joint_sample']; name = p['entity']; d = definitions[name]; expected = list(obs['xyz'])
        if d['kind'] == 'ram': expected[1] += d['direction_y']*obs['probe']
        assert expected == p['expected_xyz'] and p['error_m'] == math.dist(expected, p['world_xyz']) <= .03
        assert p['source_time_error_s'] == abs(p['time_s']-obs['time_s']) <= .05
        evidence.require(name, obs, stop=False)
    assert set(r['final_stop']) == set(definitions)-{e['entity'] for e in retired}
    for name, obs in r['final_stop'].items(): evidence.require(name, obs, home=True)
    raw, disabled = evidence.read(retired, commands)
    assert raw['simultaneous_action_samples'] > 20
    for check in [*deferred, *evidence.retained_checks]: check()
    if recovery:
        assert ownership['faults'] == ownership['stop_confirmations'] == ownership['retries'] == len(r['faults']) == 1
        assert disabled and all(c['entity'] == 'press_ram_right' and c['disabled_axes'] == ['probe'] for c in disabled)
        assert disabled[-1]['time_s']-disabled[0]['time_s'] >= 11.9
    else:
        assert not disabled and ownership['faults'] == ownership['retries'] == 0 and not r['faults']
    swept = evidence.sweeps(boxes) if sweep else None
    scope = ('same_world_recovery_sensor_fleet_inventory_18_stage' if recovery
             else 'same_world_normal_sensor_fleet_inventory_18_stage')
    result = dict(passed=True, scope=scope,
        workpieces_checked=2, ordered_operations_checked=36, **ownership, **storage, **supply,
        **raw, workers=summaries, navigation=motion, world_position_checks=len(poses),
        measured_cargo_segment_box_tests=swept,
        result_sha256=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest(),
        claim_boundary=('Recovery' if recovery else 'Normal')+' same-world sensor-authoritative two-vehicle delivery, independent receiving, '
        'measured constrained tote and finite consumables gating two 18-stage workers. '
        'Wheelset/part/tote gravity and contact carrying remain excluded; no calibrated factory dimensions, '
            'process-quality acceptance or final portable reproduction is established here. '
            'Navigation recovery, when explicitly selected, concerns three software inputs, not physical LiDAR failure.')
    if write: (directory/'navigation_workshop_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('directory', type=Path)
    parser.add_argument('--recovery', action='store_true', help='Explicit full delivery and press recovery audit')
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, recovery=args.recovery)))


if __name__ == '__main__':
    main()
