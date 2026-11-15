"""Reconstruct joint process/logistics causality from saved observations."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .verify_light_material_run import verify as verify_fleet
from .verify_multi_actuated import verify as verify_process
from .supply_bridge import SOURCES,DESTINATIONS
from .evidence import build_fingerprint


def verify(directory):
    directory=Path(directory);report=json.loads((directory/'result.json').read_text())
    assert report['passed'] and report['process_report']['passed']
    fleet=verify_fleet(directory);(directory/'audit.json').replace(directory/'fleet_audit.json')
    snapshots=directory/'executed_sources'
    process=verify_process(directory/'process',source_directory=snapshots if snapshots.is_dir() else None,
        minimum_simultaneous_devices=2)
    assert report['owned_server_pid']==report['process_report']['owned_server_pid']
    assert report['clearance']['build_fingerprint']==report['process_report']['build_fingerprint']==build_fingerprint()
    world=directory/'light_material.world'
    assert hashlib.sha256(world.read_bytes()).hexdigest()==report['process_report']['world_sha256']
    tree=ET.parse(world);assert tree.find("world/model[@name='assumed_supply_bridge']") is not None
    assert len([m for m in tree.findall('world/model') if m.get('name').startswith('fixture_ws')])==4
    for name,digest in report['source_sha256'].items():
        path=snapshots/name if snapshots.is_dir() else Path(__file__).with_name(name)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    def rows(path):return [json.loads(line) for line in (directory/path).read_text().splitlines()]
    events=rows('supply_events.jsonl');resource=rows('process/events.jsonl');trace=rows('trace.jsonl')
    bridge=rows('process/bridge_samples.jsonl');waypoints=rows('process/bridge_events.jsonl')
    efforts=rows('process/bridge_efforts.jsonl');feedback=rows('feedback.jsonl')
    assert bridge and len(efforts)==len(bridge)*3
    final=report['process_report']['bridge_final_sample']
    assert final['time_s']==bridge[-1]['time_s']
    assert math.dist(final['xyz'],bridge[-1]['xyz'])<1e-9 and max(abs(v) for v in final['velocity'])<.035
    assert all(math.isfinite(e['effort']) and abs(e['effort'])<=400 for e in efforts)
    for a,b in zip(bridge,bridge[1:]):
        dt=b['time_s']-a['time_s'];assert dt>=0
        assert all(abs(v)<=2.1 for v in b['velocity'])
        assert math.dist(a['xyz'],b['xyz'])<=3.7*dt+.01
        # All horizontal travel crosses the workshop boundary above the low
        # wall. Lowering/raising occurs with the horizontal axes stopped.
        if min(a['xyz'][2],b['xyz'][2])<2.94:
            assert math.dist(a['xyz'][:2],b['xyz'][:2])<.035
    stock=dict(report['supply_initial']);dock={};active=None;committed={};charged={};waits={};wait_durations=[]
    device_commands={e['command']['id']:e['command'] for e in resource
        if e['event']=='COMMAND_ISSUED' and e['command']['kind']=='device'}
    for seq,event in enumerate(events):
        assert event['seq']==seq and abs(event['time_s']-event['source_time_s'])<=.15
        if seq:assert event['time_s']>=events[seq-1]['time_s']
        kind=event['event']
        if kind=='DOCK_RECEIPT':
            d=event['delivery'];task=d['task_id'];assert task not in dock
            command=event['command'];obs=event['observation']
            assert command['run_id']==d['route_id'].split(':')[0]
            assert obs['linear_speed']<.025 and obs['angular_speed']<.05
            assert math.dist(obs['pose'][:2],SOURCES[task][:2])<=.1
            arrivals=[f for f in feedback if f['message']['route_id']==d['route_id']
                and f['message']['event']=='ARRIVED' and f['answer']['accepted']]
            assert len(arrivals)==1
            assert 0<=event['source_time_s']-arrivals[0]['message']['timestamp']<1.
            nearby=min(trace,key=lambda row:abs(min(o['sim_time_s'] for o in row['observations'].values())-event['source_time_s']))
            assert nearby['tasks'][task]=='completed'
            dock[task]=event
        elif kind=='HANDOFF_ISSUED':
            assert active is None
            c=event['command'];task=c['task_id'];assert task in dock and task not in committed
            assert event['time_s']>=dock[task]['time_s']-.15
            active=c
        elif kind=='SUPPLY_COMMITTED':
            assert active and active['token']==event['token'] and active['task_id']==event['task_id']
            station=event['station'];task=event['task_id'];assert task not in committed
            points=[p for p in waypoints if p['token']==active['token']]
            assert [p['waypoint'] for p in points]==list(range(6))
            assert all(b['time_s']>a['time_s'] for a,b in zip(points,points[1:]))
            source=SOURCES[task];destination=DESTINATIONS[station]
            expected=[None,(source[0],source[1],3.),source,(source[0],source[1],3.),
                (destination[0],destination[1],3.),destination]
            assert abs(points[0]['xyz'][2]-3)<.025
            for i,p in enumerate(points):
                sample=min(bridge,key=lambda b:abs(b['time_s']-p['time_s']))
                assert abs(sample['time_s']-p['time_s'])<=.1
                assert math.dist(sample['xyz'],p['xyz'])<=.01
                assert max(abs(v) for v in p['velocity'])<.035
                if i:assert math.dist(p['xyz'],expected[i])<.025
            assert 0<=event['source_time_s']-points[-1]['time_s']<=.15
            assert math.dist(event['observation']['xyz'],destination)<.025
            assert event['observation']['time_s']==points[-1]['time_s']
            assert max(abs(v) for v in event['observation']['velocity'])<.035
            assert math.dist(event['observation']['xyz'],points[-1]['xyz'])<1e-9
            assert event['quantity']==dock[task]['delivery']['quantity']
            stock[station]+=event['quantity'];committed[task]=event;active=None
        elif kind=='MATERIAL_WAIT':
            key=(event['job'],event['station']);assert key not in waits and stock[key[1]]==0
            waits[key]=event
        elif kind=='MATERIAL_CONSUMED':
            token=event['command_id'];assert token in device_commands
            c=device_commands[token];key=(event['job'],event['station'])
            assert key not in charged and c['job']==key[0] and c['target']==key[1]
            assert abs(c['issued_at']-event['source_time_s'])<=.15 and stock[key[1]]>0
            stock[key[1]]-=1;charged[key]=event
            if key in waits:
                relevant=[e for e in committed.values() if e['station']==key[1]]
                assert relevant and relevant[-1]['time_s']<=event['time_s']
                assert waits[key]['time_s']<relevant[-1]['time_s']
                wait_durations.append(dict(job=key[0],station=key[1],wait_s=event['time_s']-waits[key]['time_s'],
                    supply_commit_time_s=relevant[-1]['time_s'],device_start_time_s=c['issued_at']))
        else:raise AssertionError(kind)
        assert all(0<=v<=report['supply_capacity'] for v in stock.values())
        assert stock==event['stock']
    assert active is None and len(committed)==2 and stock==report['supply_final']=={'derust':0,'magnetic':0}
    assert len(charged)==8 and {e['station'] for e in waits.values()}=={'derust','magnetic'}
    assert len(wait_durations)>=2 and all(w['wait_s']>20 for w in wait_durations)
    if report['case']=='feedback_recovery':
        # Independently establish the output blockage rather than relying on
        # the runner's saturation flag, including its later 300s joint cutoff.
        assert any(e['snapshot']['stock']['outfeed'] and e['time_s']<300 for e in resource)
        assert not any(e['event']=='OUTFEED_RELEASED' and e['time_s']<300 for e in resource)
        assert any(e['time_s']<300 and e['snapshot']['stock']['outfeed']
            and e['snapshot']['stock']['ultrasonic'] for e in resource)
        recipe=report['process_report'].get('magnetic_spin_recipe')
        if recipe:
            assert recipe['nominal_rad_s']==1.6 and recipe['after_recovery_rad_s']==.8
            recovery_time=next(e['time_s'] for e in resource if e['event']=='RECOVERY_REISSUED')
            joint_samples=rows('process/joint_samples.jsonl')
            affected={token for token,c in device_commands.items() if c['target']=='magnetic'
                and c['issued_at']>=recovery_time}
            assert len(affected)==3
            for token in affected:
                points=[s for s in joint_samples if s['command_id']==token and s['phase']=='rotate']
                assert len(points)>20 and all(s['target'][2]==.8 for s in points)
                assert .5<max(abs(s['angular_velocity']) for s in points)<.85
    recovered_ids={e['command_id'] for e in resource if e['event']=='RECOVERY_REISSUED'}
    assert {e['command_id'] for e in charged.values()}=={
        token for token,c in device_commands.items() if c['target'] in stock and token not in recovered_ids}
    # Both material-starved workpieces retain their physical slots, while
    # downstream work is withheld. Resource auditing verifies all other locks.
    for key,event in waits.items():
        start=event['time_s'];end=charged[key]['time_s'];job,station=key
        snapshots=[e['snapshot'] for e in resource if start<=e['time_s']<end]
        assert all(job in s['stock'][station] for s in snapshots)
    total=resource[-1]['time_s']-resource[0]['time_s'];busy=defaultdict(float);began={}
    for e in resource:
        if e['event']=='COMMAND_ISSUED':began[e['command']['id']]=e['command']
        elif e['event'] in {'DEVICE_COMPLETE','FAULT_LATCHED'}:
            c=began[e['command_id']]
            if c['kind']=='device':busy[c['target']]+=e['time_s']-c['issued_at']
    result=dict(passed=True,case=report['case'],same_gazebo_server=True,wheelsets_completed=4,deliveries_committed=2,
        device_actions_completed=12,material_units_consumed=8,starved_device_waits=wait_durations,
        fleet_audit=fleet,process_audit=process,bridge_waypoints_checked=len(waypoints),
        bridge_joint_samples_checked=len(bridge),process_elapsed_sim_s=total,
        device_busy_fraction={k:v/total for k,v in busy.items()},claim_boundary=report['claim_boundary'])
    (directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


def replay(directory):
    import cv2
    import numpy as np
    directory=Path(directory);report=json.loads((directory/'result.json').read_text())
    def rows(name):return [json.loads(line) for line in (directory/name).read_text().splitlines()]
    trace=rows('trace.jsonl');supply=rows('supply_events.jsonl');resources=rows('process/events.jsonl')
    bridge=rows('process/bridge_samples.jsonl');frames=report['images']
    start=max(frames[0]['sim_time_s'],trace[0]['observations']['scout_a']['sim_time_s'])
    end=max(frames[-1]['sim_time_s'],resources[-1]['time_s'],
        max(o['sim_time_s'] for o in trace[-1]['observations'].values()))
    fps=10;count=math.ceil((end-start)*fps)+1;fi=ti=si=ri=bi=0;photo=None
    layout=report['network'];nodes={n['id']:(n['x'],n['y']) for n in layout['nodes']}
    colours={'scout_a':(70,210,250),'scout_b':(240,165,90)}
    def pixel(point):return (round(80+(point[0]-6)*20),round(265-(point[1]-50)*5))
    writer=cv2.VideoWriter(str(directory/'joint_execution.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,1040));assert writer.isOpened()
    for tick in range(count):
        stamp=min(start+tick/fps,end)
        while fi+1<len(frames) and frames[fi+1]['sim_time_s']<=stamp:fi+=1;photo=None
        while ti+1<len(trace) and min(o['sim_time_s'] for o in trace[ti+1]['observations'].values())<=stamp:ti+=1
        while si+1<len(supply) and supply[si+1]['time_s']<=stamp:si+=1
        while ri+1<len(resources) and resources[ri+1]['time_s']<=stamp:ri+=1
        while bi+1<len(bridge) and bridge[bi+1]['time_s']<=stamp:bi+=1
        if photo is None:photo=cv2.imread(str(directory/frames[fi]['file']));assert photo is not None
        image=np.zeros((1040,1280,3),dtype=np.uint8);image[:320]=(25,25,20);image[320:]=photo
        row=trace[ti];stock=report['supply_initial'] if supply[si]['time_s']>stamp else supply[si]['stock']
        snap=resources[ri]['snapshot'];finished=len(snap['completed'])
        text=[f"Joint workshop | {report['case']} | Gazebo sim {stamp:.1f}s | wheelsets {finished}/4",
            f"Station stock: derust={stock['derust']} magnetic={stock['magnetic']} | verified road deliveries {sum(v=='completed' for v in row['tasks'].values())}/2",
            'Measured XY schematic (axes have different scales); constrained assumed shuttle; recorded Gazebo image below.',
            f"Image recorded at {frames[fi]['sim_time_s']:.3f}s, held {max(0,stamp-frames[fi]['sim_time_s']):.2f}s | road sample {min(o['sim_time_s'] for o in row['observations'].values()):.3f}s"]
        for y,line in zip([25,52,78,104],text):cv2.putText(image,line,(14,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(225,235,235),1,cv2.LINE_AA)
        for lane in layout['lanes']:
            a,b=lane['src'],lane['dst'];owners=row['pins']['edges'].get(a+'->'+b,[])
            cv2.line(image,pixel(nodes[a]),pixel(nodes[b]),colours[owners[0]] if owners else (90,90,80),4,cv2.LINE_AA)
        for robot,obs in row['observations'].items():
            p=pixel(obs['pose']);colour=colours[robot]
            if row['gates'][robot]['fault']:colour=(70,70,245)
            cv2.circle(image,p,8,colour,2,cv2.LINE_AA)
            cv2.putText(image,robot,(p[0]-30,p[1]-13),cv2.FONT_HERSHEY_SIMPLEX,.43,colour,1,cv2.LINE_AA)
        for station in ['derust','magnetic']:
            p=pixel(DESTINATIONS[station]);cv2.rectangle(image,(p[0]-18,p[1]-10),(p[0]+18,p[1]+10),(180,165,90),2)
            jobs=snap['stock'][station];state=snap['jobs'][jobs[0]]['state'] if jobs else 'empty'
            cv2.putText(image,f"{station}: {state}",(p[0]-65,p[1]+30),cv2.FONT_HERSHEY_SIMPLEX,.43,(225,215,190),1,cv2.LINE_AA)
        bs=bridge[bi];p=pixel(bs['xyz']);cv2.rectangle(image,(p[0]-6,p[1]-6),(p[0]+6,p[1]+6),(70,175,255),-1)
        cv2.putText(image,f"supply shuttle z={bs['xyz'][2]:.2f}m",(max(12,p[0]-60),p[1]-16),cv2.FONT_HERSHEY_SIMPLEX,.44,(70,175,255),1,cv2.LINE_AA)
        writer.write(image)
    writer.release();capture=cv2.VideoCapture(str(directory/'joint_execution.mp4'))
    ok,_=capture.read();actual=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));capture.release();assert ok and actual==count
    assert len(resources[ri]['snapshot']['completed'])==4
    result=dict(file='joint_execution.mp4',frames=count,fps=fps,simulation_start_s=start,simulation_end_s=end,
        source_frames=len(frames),camera_end_s=frames[-1]['sim_time_s'],
        final_image_hold_s=end-frames[-1]['sim_time_s'],final_completed_wheelsets=4,
        sha256=hashlib.sha256((directory/'joint_execution.mp4').read_bytes()).hexdigest(),
        method='Recorded Gazebo frames with measured truth/joints, station-stock and resource snapshots through final completion. Image recording time and held-image age are labelled; schematic axes differ in scale; no generated physical motion.')
    (directory/'joint_replay.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();r=verify(args.directory)
    print(json.dumps({k:r[k] for k in ['passed','case','wheelsets_completed','deliveries_committed','starved_device_waits']}))
    if args.replay:
        print(json.dumps(replay(args.directory)))
        from .verify_resource_run import replay as process_replay
        print(json.dumps(process_replay(args.directory/'process')))


if __name__=='__main__':main()
