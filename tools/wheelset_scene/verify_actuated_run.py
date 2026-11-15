"""Audit recorded joint motion and interlocks; render a timed execution replay."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def verify(directory):
    directory=Path(directory);report=json.loads((directory/'result.json').read_text())
    assert report['passed'] and report['runtime_error'] is None
    assert report['baseline_session_preserved'] and report['owned_server_stopped']
    assert hashlib.sha256((directory/'actuated.world').read_bytes()).hexdigest()==report['world_sha256']
    for name,digest in report['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==digest,name
    for image in report['images']:
        assert hashlib.sha256((directory/image['file']).read_bytes()).hexdigest()==image['sha256']
    samples=[json.loads(line) for line in (directory/'joint_samples.jsonl').read_text().splitlines()]
    commands=[json.loads(line) for line in (directory/'effort_commands.jsonl').read_text().splitlines()]
    assert len(samples)==report['joint_sample_count']>30 and len(commands)==report['effort_command_count']
    assert all(b['time_s']>a['time_s'] for a,b in zip(samples,samples[1:]))
    limits={'wsd_slide':7000,'wsd_probe':200,'wsd_spin':1000}
    for row in commands:
        assert row['efforts'].keys()==limits.keys()
        assert all(math.isfinite(v) and abs(v)<=limits[k] for k,v in row['efforts'].items())
        assert len(row['request_windows'])==3
        assert all(w['accepted'] and w['start_sim_time_s']>=row['sim_time_s'] and w['duration_s']==.25
            for w in row['request_windows'])
    assert max(abs(r['efforts']['wsd_slide']) for r in commands)>100
    assert report['pose_checks'] and all(p['sample_time_error_s']<=.05 and p['x_consistency_error_m']<=.15 and abs(p['world_pose'][1]-50)<.02
        and abs(p['world_pose'][2]-.56)<.02 for p in report['pose_checks'])
    stop=report['final_stop']
    assert abs(stop['slide_velocity'])<.04 and abs(stop['angular_velocity'])<.025 and abs(stop['probe'])<.005 and abs(stop['probe_velocity'])<.015
    events=report['events'];measurements=[]
    assert all(e['seq']==n for n,e in enumerate(events))
    assert all(b['sim_time_s']>=a['sim_time_s'] for a,b in zip(events,events[1:]))
    completed=[e['station'] for e in events if e['event']=='DEVICE_COMPLETE']
    for station in completed:
        record=[e for e in events if e['station']==station]
        required=['ARRIVAL_CONFIRMED','DEVICE_BUSY','PROBE_ENGAGED','ROTATION_MEASURED','SPINDLE_STOPPED','DEVICE_COMPLETE','DEVICE_IDLE']
        relevant=[e for e in record if e['event'] in required]
        assert [e['event'] for e in relevant]==required
        entering=next(e for e in record if e['event']=='PHASE_ENTERED' and e['phase']=='rotate')
        measured=next(e for e in record if e['event']=='ROTATION_MEASURED')
        points=[s for s in samples if entering['sim_time_s']<=s['time_s']<=measured['sim_time_s']]
        angle=sum(math.atan2(math.sin(b['angle']-a['angle']),math.cos(b['angle']-a['angle'])) for a,b in zip(points,points[1:]))
        assert angle>=2*math.pi
        assert abs(angle-measured['angle_rad'])<1e-7
        assert max(abs(s['angular_velocity']) for s in points)>.5
        assert all(abs(s['probe']-.22)<=.015 for s in points)
        last=next(s for s in samples if s['time_s']==relevant[-2]['sim_time_s'])
        assert abs(last['probe'])<=.005 and abs(last['angular_velocity'])<=.025
        measurements.append(dict(station=station,measured_rotation_rad=angle,rotation_samples=len(points)))
    if report['case']=='normal':
        assert report['state']=='complete' and completed==['derust','magnetic','ultrasonic']
        assert report['completed_operations']==completed+['outfeed'] and abs(stop['slide']-42.5)<=.025
        assert events[-1]['event']=='CYCLE_COMPLETE'
    else:
        assert report['state']=='failed' and completed==['derust']
        assert report['retained_station']=='magnetic' and events[-1]['event']=='CYCLE_FAILED'
        assert not any(e['station'] in {'ultrasonic','outfeed'} for e in events)
        assert report['fault']==('rotate_timeout' if report['case']=='stalled_spindle' else 'joint_feedback_stale')
        if report['case']=='stalled_spindle':
            rot=[s for s in samples if s['station']=='magnetic' and s['phase']=='rotate']
            assert rot[-1]['time_s']-rot[0]['time_s']>=11
            assert max(abs(s['angular_velocity']) for s in rot)<.05
            assert all(c['efforts']['wsd_spin']==0 for c in commands if c['station']=='magnetic' and c['phase']=='rotate')
    result=dict(passed=True,case=report['case'],state=report['state'],completed_operations=report['completed_operations'],
        joint_samples_checked=len(samples),effort_commands_checked=len(commands),rotation_measurements=measurements,
        final_measured_axes_stopped_and_probe_home=True,artifact_hashes_checked=True,
        claim_boundary='Saved actuator commands, Gazebo joint motion and state-service consistency. No force sensor, contact, factory timing or whole-line acceptance.')
    (directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def replay(directory):
    import cv2
    import numpy as np
    directory=Path(directory);report=json.loads((directory/'result.json').read_text())
    frames=report['images'];assert len(frames)>1
    events=report['events'];samples=[json.loads(line) for line in (directory/'joint_samples.jsonl').read_text().splitlines()]
    fps=10;target=directory/'execution.mp4';width,height=1280,720
    writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    assert writer.isOpened()
    index=0;sample_index=0;event_index=0;image=None
    start,end=frames[0]['sim_time_s'],frames[-1]['sim_time_s']
    times=np.arange(start,end+1/fps,1/fps)
    for stamp in times:
        while index+1<len(frames) and frames[index+1]['sim_time_s']<=stamp:index+=1;image=None
        if image is None:image=cv2.imread(str(directory/frames[index]['file']));assert image is not None
        canvas=image.copy()
        while sample_index+1<len(samples) and samples[sample_index+1]['time_s']<=stamp:sample_index+=1
        while event_index+1<len(events) and events[event_index+1]['sim_time_s']<=stamp:event_index+=1
        s=samples[sample_index];event=events[event_index]
        cv2.rectangle(canvas,(0,0),(width,98),(15,24,30),-1)
        text=[f"Effort-driven inspection | {report['case']} | sim +{stamp-start:.1f}s",
            f"{s['station']} / {s['phase']} | slide {s['slide']:.2f} m | probe {s['probe']:.3f} m | spindle {s['angular_velocity']:.2f} rad/s",
            'Assumed constrained fixture; gravity/contact excluded. Joint feedback controls completion.']
        if event['event'] in {'CYCLE_COMPLETE','CYCLE_FAILED'}:text[1]=f"{event['event']} | {report['fault'] or 'all operations complete'}"
        for y,line in zip([27,56,84],text):cv2.putText(canvas,line,(16,y),cv2.FONT_HERSHEY_SIMPLEX,.59,(225,235,240),1,cv2.LINE_AA)
        writer.write(canvas)
    writer.release()
    video=cv2.VideoCapture(str(target));count=int(video.get(cv2.CAP_PROP_FRAME_COUNT));ok,_=video.read();video.release()
    assert ok and count==len(times)
    result=dict(file=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),frames=count,fps=fps,
        source_frames=len(frames),sim_start_s=start,sim_end_s=end,
        method='Timestamp-resampled Gazebo camera frames; text from measured joints/events; no generated device motion.')
    (directory/'replay.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();print(json.dumps(verify(args.directory)))
    if args.replay:print(json.dumps(replay(args.directory)))


if __name__=='__main__':main()
