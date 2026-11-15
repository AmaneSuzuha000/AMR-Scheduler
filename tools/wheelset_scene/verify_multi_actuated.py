"""Audit capacity and independently integrate measured device rotations."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .verify_resource_run import verify as verify_resources


def verify(directory,*,source_directory=None,minimum_simultaneous_devices=2):
    directory=Path(directory);report=json.loads((directory/'result.json').read_text())
    resource_audit=verify_resources(directory,source_directory=source_directory,
        minimum_simultaneous_devices=minimum_simultaneous_devices)
    (directory/'audit.json').replace(directory/'resource_audit.json')
    assert report['runtime_error'] is None and report['owned_server_stopped'] and report['baseline_session_preserved']
    assert hashlib.sha256((directory/report['world']).read_bytes()).hexdigest()==report['world_sha256']
    for name,digest in report['data_sha256'].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest
    assert len(report['images'])>2
    for frame in report['images']:
        assert hashlib.sha256((directory/frame['file']).read_bytes()).hexdigest()==frame['sha256']
    def rows(name):return [json.loads(line) for line in (directory/(name+'.jsonl')).read_text().splitlines()]
    samples=rows('joint_samples');efforts=rows('effort_commands');poses=rows('pose_checks')
    actor_samples=rows('action_samples');device_events=rows('device_events');events=rows('events')
    assert len(samples)==report['joint_sample_count'] and len(efforts)==report['effort_request_count']==len(samples)*3
    assert len(device_events)==report['device_event_count']
    assert all(r['accepted'] and r['duration_s']==.25 and r['start_sim_time_s']==r['sample_time_s'] for r in efforts)
    limits={'slide':7000,'probe':200,'spin':1000}
    assert all(math.isfinite(r['effort']) and abs(r['effort'])<=limits[r['joint'].split('_')[-1]] for r in efforts)
    assert poses and all(p['sample_time_error_s']<=.05 and p['x_consistency_error_m']<=.15
        and abs(p['world_pose'][1]-50)<.02 and abs(p['world_pose'][2]-.56)<.02 for p in poses)
    jobs=sorted(report['assumptions']['fixture_world_x_origins']);origins=report['assumptions']['fixture_world_x_origins']
    for job in jobs:
        data=[s for s in samples if s['job']==job]
        assert len(data)>30 and all(b['time_s']>=a['time_s'] for a,b in zip(data,data[1:]))
        assert max(abs(s['slide_velocity']) for s in data)>1.
        assert all(abs(s['world_x']-origins[job]-s['slide'])<1e-8 for s in data)
    commands={e['command']['id']:e['command'] for e in events if e['event']=='COMMAND_ISSUED'}
    measurements=[]
    for e in events:
        if e['event']=='ARRIVAL_CONFIRMED':
            c=commands[e['command_id']]
            obs=[p for p in poses if p['command_id']==c['id'] and p['sim_time_s']<=e['time_s']]
            assert obs and e['time_s']-obs[-1]['sim_time_s']<=.1
            target=next(p['target'] for p in report['flow'] if p['id']==c['target'])
            point=obs[-1]['world_pose'];assert math.dist([point[0],point[1],point[2]-.56],target)<=.03
        if e['event']!='DEVICE_COMPLETE':continue
        token=e['command_id'];c=commands[token];trace=[d for d in device_events if d['command_id']==token]
        assert [d['event'] for d in trace]==['ACTION_STARTED','PROBE_ENGAGED','ROTATION_MEASURED','SPINDLE_STOPPED','ACTION_COMPLETE']
        assert trace[-1]['sim_time_s']<=e['time_s'] and trace[0]['sim_time_s']>=c['issued_at']
        assert all(b['sim_time_s']>a['sim_time_s'] for a,b in zip(trace,trace[1:]))
        points=[s for s in actor_samples if s['command_id']==token
            and trace[1]['sim_time_s']<=s['time_s']<=trace[2]['sim_time_s']]
        assert len(points)>20 and max(abs(s['angular_velocity']) for s in points)>.5
        # Include the engaged sample as the angle origin; use source angles,
        # never the controller's completion flag, to recalculate the full turn.
        angle=sum(math.atan2(math.sin(b['angle']-a['angle']),math.cos(b['angle']-a['angle'])) for a,b in zip(points,points[1:]))
        assert angle>=2*math.pi and abs(angle-trace[2]['angle_rad'])<1e-7
        target=next(p['target'][0]-origins[c['job']] for p in report['flow'] if p['id']==c['target'])
        assert all(abs(s['probe']-.22)<=.015 and abs(s['slide']-target)<=.025
            and abs(s['slide_velocity'])<=.035 for s in points)
        last=next(s for s in actor_samples if s['command_id']==token and s['time_s']==trace[-1]['sim_time_s'])
        assert abs(last['probe'])<=.005 and abs(last['probe_velocity'])<=.015 and abs(last['angular_velocity'])<=.025
        measurements.append(dict(job=c['job'],station=c['target'],command_id=token,rotation_rad=angle))
    assert len(measurements)==12
    assert sorted(report['deleted_after_measured_output'])==jobs and report['final_stop']=={}
    if report['case']=='normal':assert report['fault'] is None
    else:
        fault=report['fault'];assert fault['job']=='ws002' and fault['reason']=='rotate_timeout'
        trace=[d for d in device_events if d['command_id']==fault['token']]
        assert trace[-1]['event']=='ACTION_FAILED' and trace[-1]['reason']=='rotate_timeout'
        assert not any(d['event']=='ACTION_COMPLETE' for d in trace)
        points=[s for s in actor_samples if s['command_id']==fault['token'] and s['phase_before']=='rotate']
        assert points[-1]['time_s']-points[0]['time_s']>=11
        assert max(abs(s['angular_velocity']) for s in points)<.05
        commands_before_fault=[r for r in efforts if r['command_id']==fault['token'] and r['stall_injected'] and r['joint'].endswith('_spin')]
        assert len(commands_before_fault)>20 and all(r['effort']==0 for r in commands_before_fault)
        stop=report['stop_evidence'];assert stop['time_s']>fault['sim_time_s']+.5
        assert abs(stop['probe'])<.005 and abs(stop['slide_velocity'])<.035 and abs(stop['angular_velocity'])<.025
        assert report['output_saturation_observed'] and report['superseded_completion_rejected']
        assert not any(e['event']=='OUTFEED_RELEASED' and e['time_s']<120 for e in events)
    result=dict(passed=True,case=report['case'],jobs_checked=4,device_actions_completed=12,
        joint_samples_checked=len(samples),effort_requests_checked=len(efforts),rotation_measurements=measurements,
        max_simultaneous_devices=resource_audit['max_simultaneous_devices'],
        resource_ownership_and_joint_completion_checked=True,claim_boundary=report['claim_boundary'])
    (directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--replay',action='store_true')
    args=p.parse_args();print(json.dumps(verify(args.directory)))
    if args.replay:
        from .verify_resource_run import replay
        print(json.dumps(replay(args.directory)))


if __name__=='__main__':main()
