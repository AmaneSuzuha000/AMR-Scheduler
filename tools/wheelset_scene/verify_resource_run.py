"""Independently replay ownership deltas in a saved logical resource trace."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def verify(directory,*,source_directory=None,minimum_simultaneous_devices=2):
    directory=Path(directory);report=json.loads((directory/'result.json').read_text(encoding='utf-8'))
    content=(directory/'events.jsonl').read_bytes()
    assert hashlib.sha256(content).hexdigest()==report['events_sha256']
    for name,digest in report['source_sha256'].items():
        path=Path(source_directory)/name if source_directory is not None else Path(__file__).with_name(name)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    events=[json.loads(line) for line in content.splitlines()]
    assert len(events)==report['event_count'] and report['passed']
    capacities={p['id']:p['capacity'] for p in report['flow']};order=list(capacities)
    stock={p:set() for p in order};reserved={p:set() for p in order};carrier=None;commands={};completed=[]
    busy={};done={};previous_time=-float('inf');max_busy=0
    for seq,e in enumerate(events):
        assert e['seq']==seq and e['time_s']>=previous_time;previous_time=e['time_s']
        kind=e['event']
        if kind=='ADMITTED':
            assert all(e['job'] not in items for items in stock.values());stock[order[0]].add(e['job']);done[e['job']]=[]
        elif kind=='COMMAND_ISSUED':
            c=e['command'];assert c['id'] not in commands
            assert c['job'] in stock[c['source']];commands[c['id']]=dict(c)
            if c['kind']=='transfer':
                assert carrier is None and order.index(c['target'])==order.index(c['source'])+1
                if c['source'] in report['device_durations_s']:assert c['source'] in done[c['job']]
                reserved[c['target']].add(c['job']);carrier=c['id']
            else:
                assert c['target'] not in busy and c['source']==c['target'];busy[c['target']]=c['id']
                max_busy=max(max_busy,len(busy))
        elif kind in {'ARRIVAL_CONFIRMED','DEVICE_COMPLETE'}:
            c=commands[e['command_id']];assert c['status']=='active'
            assert c['issued_at']<=e['observed_at']<=e['time_s'];c['status']='done'
            if kind=='ARRIVAL_CONFIRMED':
                assert carrier==c['id'];stock[c['source']].remove(c['job'])
                reserved[c['target']].remove(c['job']);stock[c['target']].add(c['job']);carrier=None
            else:
                assert busy.pop(c['target'])==c['id'] and c['target'] not in done[c['job']]
                done[c['job']].append(c['target'])
        elif kind=='FAULT_LATCHED':
            c=commands[e['command_id']];assert c['status']=='active';c.update(status='failed',fault_time=e['time_s'])
        elif kind=='STOP_CONFIRMED':
            c=commands[e['command_id']];assert c['status']=='failed' and e['observed_at']>=c['fault_time']
            c['stop_confirmed']=True
        elif kind=='RECOVERY_REISSUED':
            old=commands[e['old_command_id']];new=commands[e['command_id']]
            assert old['status']=='superseded' and old['stop_confirmed'] and old['job']==new['job']
        elif kind=='OUTFEED_RELEASED':
            assert done[e['job']]==list(report['device_durations_s'])
            stock[order[-1]].remove(e['job']);assert e['job'] not in completed;completed.append(e['job'])
        else:raise AssertionError(kind)
        snapshot=e['snapshot']
        assert stock=={k:set(v) for k,v in snapshot['stock'].items()}
        assert reserved=={k:set(v) for k,v in snapshot['reserved'].items()}
        assert carrier==snapshot['carrier'] and completed==snapshot['completed']
        assert all(len(stock[p])+len(reserved[p])<=capacities[p] for p in order)
        # Recovery's new COMMAND_ISSUED atomically transfers the retained locks.
        if seq+1<len(events) and events[seq+1]['event']=='COMMAND_ISSUED':
            incoming=events[seq+1]['command']
            failed=[c for c in commands.values() if c['status']=='failed' and c['job']==incoming['job']]
            if failed:
                old,=failed;assert old.get('stop_confirmed') and all(old[k]==incoming[k] for k in ['kind','source','target'])
                old['status']='superseded'
                if old['kind']=='transfer':reserved[old['target']].remove(old['job']);carrier=None
                else:assert busy.pop(old['target'])==old['id']
    assert len(completed)==report['job_count'] and completed==report['completed']
    assert all(not v for v in stock.values()) and all(not v for v in reserved.values()) and not busy
    assert max_busy>=minimum_simultaneous_devices
    if report['case']=='blocked_output':
        assert any(len(e['snapshot']['stock']['outfeed'])==capacities['outfeed'] and e['time_s']<150 for e in events)
        assert not any(e['event']=='OUTFEED_RELEASED' and e['time_s']<150 for e in events)
    if report['case'].endswith('recovery'):
        assert sum(e['event']=='FAULT_LATCHED' for e in events)==1
        assert sum(e['event']=='STOP_CONFIRMED' for e in events)==1
        assert sum(e['event']=='RECOVERY_REISSUED' for e in events)==1
        assert report['superseded_completion_rejected']
    if 'world_file' in report:
        assert report['runtime_error'] is None and report['baseline_session_preserved'] and report['owned_server_stopped']
        assert hashlib.sha256((directory/report['world_file']).read_bytes()).hexdigest()==report['world_sha256']
        assert len(report['images'])>2
        for image in report['images']:
            assert hashlib.sha256((directory/image['file']).read_bytes()).hexdigest()==image['sha256']
        rows=[json.loads(line) for line in (directory/'trajectory.jsonl').read_text().splitlines()]
        assert len(rows)==report['motion_sample_count']>30
        assert all(b['time_s']>=a['time_s'] for a,b in zip(rows,rows[1:]))
        assert all(r['position_error_m']<=.03 for r in rows)
        places={p['id']:p['target'] for p in report['flow']}
        for e in events:
            if e['event'] not in {'ARRIVAL_CONFIRMED','DEVICE_COMPLETE'}:continue
            c=commands[e['command_id']]
            records=[r for r in rows if r['command_id']==c['id'] and r['time_s']<=e['time_s']]
            assert records and e['time_s']-records[-1]['time_s']<.5
            assert math.dist(records[-1]['observed'],places[c['target']])<=.03
            if c['kind']=='device':assert e['time_s']-c['issued_at']>=report['device_durations_s'][c['target']]
        if report['case']=='blocked_recovery':
            holds=[r for r in rows if r['kind']=='fault_hold']
            assert len(holds)>2 and holds[-1]['time_s']-holds[0]['time_s']>=.75
            assert all(r['position_error_m']<.001 and r['reported_twist_norm_max']<.001 for r in holds)
            assert report['stop_feedback'] and report['output_saturation_observed']
            assert not any(e['event']=='OUTFEED_RELEASED' and e['time_s']<75 for e in events)
    result=dict(passed=True,case=report['case'],jobs_checked=len(completed),events_checked=len(events),
        max_simultaneous_devices=max_busy,capacity_ownership_and_order_checked=True,
        claim_boundary=report['claim_boundary'])
    (directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def replay(directory):
    import cv2
    import numpy as np
    directory=Path(directory);report=json.loads((directory/'result.json').read_text())
    frames=report['images'];assert len(frames)>1
    events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    start=frames[0]['time_s'];end=max(frames[-1]['time_s'],events[-1]['time_s']);fps=10
    driven='fixture_count' in report.get('assumptions',{})
    clock_label='sim' if driven else 'wall'
    target=directory/'execution.mp4'
    writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,720));assert writer.isOpened()
    index=0;event_index=0;image=None
    times=np.minimum(start+np.arange(math.ceil((end-start)*fps)+1)/fps,end)
    for stamp in times:
        while index+1<len(frames) and frames[index+1]['time_s']<=stamp:index+=1;image=None
        if image is None:image=cv2.imread(str(directory/frames[index]['file']));assert image is not None
        while event_index+1<len(events) and events[event_index+1]['time_s']<=stamp:event_index+=1
        event=events[event_index];snapshot=event['snapshot'];canvas=image.copy()
        camera_stamp=frames[index].get('sim_time_s',frames[index]['time_s']) if driven else frames[index]['time_s']
        cv2.rectangle(canvas,(0,0),(1280,126),(15,24,30),-1)
        status=' | '.join(f"{job}:{state['state']}" for job,state in sorted(snapshot['jobs'].items()))
        lines=[f"Multi-wheelset resources | {report['case']} | {clock_label} +{stamp:.1f}s | outfeed {len(snapshot['completed'])}/4",
            status,('Gazebo joint feedback; effort-driven transfer and device actions. Gravity/contact excluded.' if driven
                else 'Gazebo pose feedback; kinematic transfer and software-timed devices.'),
            f"Image recorded at {camera_stamp:.3f}s, held {max(0,stamp-camera_stamp):.2f}s; resource event at {event['time_s']:.3f}s"]
        for y,line in zip([27,56,84,112],lines):cv2.putText(canvas,line,(16,y),cv2.FONT_HERSHEY_SIMPLEX,.58,(225,235,240),1,cv2.LINE_AA)
        writer.write(canvas)
    writer.release();video=cv2.VideoCapture(str(target));count=int(video.get(cv2.CAP_PROP_FRAME_COUNT));ok,_=video.read();video.release()
    assert ok and count==len(times)
    result=dict(file=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),frames=count,fps=fps,
        source_frames=len(frames),wall_start_s=start,wall_end_s=end,clock=clock_label,
        camera_end_s=camera_stamp,final_image_hold_s=end-camera_stamp,
        final_completed_wheelsets=len(events[event_index]['snapshot']['completed']),
        method=('Gazebo frames resampled by recorded simulation time; resource events supply labels through the final event. Recorded effort-driven joint motion; held image time/age labelled.' if driven
            else 'Gazebo frames resampled by wall reception time; resource events supply labels through the final event. Device work is software timed; held image time/age labelled; no generated device motion.'))
    (directory/'replay.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--replay',action='store_true')
    args=p.parse_args();print(json.dumps(verify(args.directory)))
    if args.replay:print(json.dumps(replay(args.directory)))


if __name__=='__main__':main()
