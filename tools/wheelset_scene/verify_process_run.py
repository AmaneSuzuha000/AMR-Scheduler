"""Audit saved execution observations independently of controller transitions."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def verify(directory):
    directory=Path(directory);report=json.loads((directory/'result.json').read_text())
    assert report['passed'] and report['baseline_session_preserved'] and report['cleanup']['owned_server_stopped']
    assert report['runtime_error'] is None
    for name,expected in report['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==expected
    assert hashlib.sha256((directory/'process_demo.world').read_bytes()).hexdigest()==report['world_sha256']
    for image in report['images'].values():
        assert hashlib.sha256((directory/image['file']).read_bytes()).hexdigest()==image['sha256']
    busy={};arrivals={};requested={};completed=[];failed=False;previous=-math.inf
    for seq,event in enumerate(report['events']):
        assert event['seq']==seq and event['run_id']==report['run_id']
        assert event['time_s']>=previous;previous=event['time_s']
        kind=event['event'];op=event.get('operation')
        assert not failed,'events after failure'
        if kind=='TRANSFER_REQUESTED':
            assert not busy and op not in completed
            requested[event['command_id']]=event
        elif kind=='ARRIVAL_CONFIRMED':
            command=requested[event['command_id']]
            assert command['operation']==op and event['time_s']>=command['issued_at']
            assert math.dist(event['observed_pose'],command['target'])<=.03
            arrivals[op]=event['time_s']
        elif kind=='DEVICE_BUSY':
            assert op in arrivals and not busy
            busy[op]=event
        elif kind=='DEVICE_COMPLETE':
            assert op in busy
            assert event['time_s']>=busy[op]['time_s']+busy[op]['duration_s']
            completed.append(op)
        elif kind=='DEVICE_IDLE':
            assert op in completed;del busy[op]
        elif kind=='CYCLE_FAILED':failed=True
        elif kind=='CYCLE_COMPLETE':assert len(completed)==4 and not busy
    assert completed==report['completed_operations']
    samples=[json.loads(line) for line in (directory/'trajectory.jsonl').read_text().splitlines()]
    assert len(samples)==report['motion_sample_count'] and samples
    for sample in samples:
        assert sample['command_id'] in requested
        assert math.dist(sample['requested'],sample['observed'])<=.03
    if report['injected_failure']:
        assert report['state']=='failed' and report['fault']=='execution_failed'
        assert report['retained_station']==report['injected_failure']
        assert completed==['derust_hood']
        assert all(e['event']!='CYCLE_COMPLETE' for e in report['events'])
    else:
        assert report['state']=='complete' and report['retained_station'] is None
        assert completed==['derust_hood','magnetic_test_hood','ultrasonic_test_hood','outfeed']
    return dict(passed=True,run=directory.name,state=report['state'],completed_operations=completed,
        trajectory_samples_checked=len(samples),artifact_hashes_checked=True,
        claim_boundary='Saved software event causality and Gazebo kinematic pose observations; not independent physical validation.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    result=verify(args.directory)
    (args.directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
