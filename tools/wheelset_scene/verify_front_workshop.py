"""Independently reconstruct prefix ownership, part bins and measured motion."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .build import SCENE,ROOT
from .evidence import build_fingerprint
from .front_flow import STAGES,ORIGIN,parts
from .front_world import static_boxes,envelopes
from .assembly_flow import ASSEMBLY_STAGES,components
from .assembly_world import EXTRA_SNAPSHOTS
from .finishing_flow import FULL_STAGES,STORED_JOB,STORAGE_UNLOAD
from .finishing_world import FINISH_SNAPSHOTS,finishing_static
from .feedback_evidence import corroborate_stopped


def measured_sweeps(samples,boxes,name,kind):
    """Vectorized exact segment/AABB test for each conservative cargo box."""
    import numpy as np
    xyz=np.array([s['observations'][name]['xyz'] for s in samples]);bounds=np.array([b['bounds'] for b in boxes])
    checked=0
    for center,half in envelopes(kind):
        low=bounds[:,:3]-np.array(half)-.01;high=bounds[:,3:]+np.array(half)+.01
        for first in range(0,len(xyz)-1,128):
            end=min(first+128,len(xyz)-1)
            a=xyz[first:end,None,:]+center;b=xyz[first+1:end+1,None,:]+center
            d=b-a;zero=np.abs(d)<1e-12;safe=np.where(zero,1.,d)
            u=(low[None,:,:]-a)/safe;v=(high[None,:,:]-a)/safe
            lower=np.where(zero,-np.inf,np.minimum(u,v));upper=np.where(zero,np.inf,np.maximum(u,v))
            outside=np.any(zero&((a<low[None,:,:])|(a>high[None,:,:])),axis=2)
            hit=(np.maximum(0.,np.max(lower,axis=2))<=np.minimum(1.,np.min(upper,axis=2)))&~outside
            if np.any(hit):
                row,col=np.argwhere(hit)[0]
                raise AssertionError(f"Measured cargo sweep intersects {boxes[col]['id']}: {name} sample {first+row}")
            checked+=len(a)*len(boxes)
    return checked


def verify(directory):
    directory=Path(directory);r=json.loads((directory/'result.json').read_text())
    finishing=r.get('through')=='storage';assembly=r.get('through') in {'assembly','storage'}
    recipe=FULL_STAGES if finishing else ASSEMBLY_STAGES if assembly else STAGES
    assert r['passed'] and r['runtime_error'] is None and r['owned_server_stopped'] and r['baseline_session_preserved']
    assert r['build_fingerprint']==build_fingerprint()
    for name,digest in r['data_sha256'].items():assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest,name
    for name,digest in r['source_sha256'].items():assert hashlib.sha256((directory/'executed_sources'/name).read_bytes()).hexdigest()==digest,name
    for image in r['images']:assert hashlib.sha256((directory/image['file']).read_bytes()).hexdigest()==image['sha256']
    assert len(r['images'])>5
    def rows(name):return [json.loads(line) for line in (directory/(name+'.jsonl')).read_text().splitlines()]
    samples=rows('samples');efforts=rows('efforts');events=rows('events');waypoints=rows('waypoints')
    poses=rows('pose_checks');actions=rows('device_events');items=parts()
    meta=json.loads((directory/'metadata.json').read_text());planned=json.loads((directory/'planned_clearance.json').read_text())
    names=list(meta['definitions']);expected_names={'ws001',*items}
    if assembly:expected_names.update([*components(),'press_ram_left','press_ram_right'])
    if finishing:expected_names.update([STORED_JOB,'spray_tool'])
    assert set(names)==expected_names
    assert meta['stages']==json.loads(json.dumps(recipe)) and planned['passed'] and all(not c['hits'] for c in planned['checks'])
    scene=json.loads(SCENE.read_text(encoding='utf-8'));derived=copy.deepcopy(scene)
    removed=meta['removed_static_snapshot_ids'];assert set(removed)=={i['id'] for i in scene['instances']
        if i['asset']=='wheelset' and (abs(i['pose'][1]-9)<1e-6 or abs(i['pose'][1]-50)<1e-6)} | (EXTRA_SNAPSHOTS if assembly else set()) | (FINISH_SNAPSHOTS if finishing else set())
    derived['instances']=[i for i in scene['instances'] if i['id'] not in removed]
    boxes,sources=finishing_static(derived) if finishing else static_boxes(derived)
    assert boxes==meta['static_obstacles'] and sources==meta['collision_source_sha256']
    world=ET.parse(directory/'front.world').getroot().find('world')
    for name in names:
        model=world.find(f"model[@name='{name}']");assert model is not None
        assert len(model.findall('joint'))==(6 if meta['definitions'][name]['kind']=='wheelset' else 2 if name.startswith('press_ram_') else 5)
        assert not model.findall('.//collision') and all(link.findtext('gravity')=='false' for link in model.findall('link'))
    axes_count=sum(len(d['axes']) for d in meta['definitions'].values())
    assert len(samples)==r['sample_count']>100 and len(efforts)==r['effort_count']==axes_count*len(samples)
    assert all(math.isfinite(e['effort']) and abs(e['effort'])<=(15000 if meta['definitions'][e['entity']]['kind']=='wheelset' and e['axis'] in 'xyz'
        else 1000 if e['axis']=='spin' else 200 if e['axis']=='probe' else 400) for e in efforts)
    assert r['control_method']=='native_step_servo'
    native=r['native_build'];assert native['passed']
    assert native['library_sha256']==r['source_sha256']['libwheelset_axis_servo.so']
    for name,digest in native['source_sha256'].items():assert r['source_sha256']['native_plugin/'+name]==digest
    commands_raw=rows('native_commands');assert len(commands_raw)==len(names)*len(samples)
    lookup={(c['entity'],c['sequence']):c for c in commands_raw};assert len(lookup)==len(commands_raw)
    for index,sample in enumerate(samples):
        assert max(s['time_s'] for s in sample['observations'].values())-min(s['time_s'] for s in sample['observations'].values())<=.1
        for name in names:
            c=lookup[name,index+1];definition=meta['definitions'][name]
            assert c['time_s']==sample['time_s'] and c['command_id']==sample['command_id']
            assert c['target']==sample['targets'][name]
            for i,axis in enumerate(definition['axes']):
                value=c['target']['xyz']['xyz'.index(axis)]-definition['origin']['xyz'.index(axis)] if axis in 'xyz' else c['target'][axis]
                assert c['axis_values'][2*i]==value
                assert c['axis_values'][2*i+1]==(0. if axis in c['disabled_axes'] else 1.)
            obs=sample['observations'][name];assert not obs['servo_fault']
            if obs['servo_sequence']>=0:
                ack=lookup[name,obs['servo_sequence']];assert ack['time_s']<=obs['time_s']+.05
    for e in efforts:
        assert e['source']=='native_step_servo'
        if e['acknowledged_sequence']>=0:
            ack=lookup[e['entity'],e['acknowledged_sequence']]
            assert e['injected']==(e['axis'] in ack['disabled_axes'])
            if e['injected']:assert e['effort']==0.
    offset=0
    for sample in samples:
        for name in names:
            for axis,value in sample['observations'][name]['applied_effort'].items():
                e=efforts[offset];offset+=1
                assert (e['entity'],e['axis'],e['time_s'],e['effort'])==(name,axis,sample['observations'][name]['time_s'],value)
    for a,b in zip(samples,samples[1:]):
        assert b['time_s']>=a['time_s']
        for name in names:
            x=a['observations'][name];y=b['observations'][name];dt=y['time_s']-x['time_s'];assert dt>=0
            assert all(math.isfinite(v) for v in [*y['xyz'],*y['velocity'],y['omega'],y['probe']])
            assert max(abs(v) for v in y['velocity'])<=2.1
            assert math.dist(x['xyz'],y['xyz'])<=3.7*dt+.015
    for row in poses:
        assert row['source_time_error_s']<=.05 and row['error_m']<=.03
        if row.get('reason')=='ram_head':
            d=meta['definitions'][row['entity']];expected=list(d['origin'])
            expected[1]+=d['direction_y']*row['joint_sample']['probe']
            assert math.dist(row['world_xyz'],expected)<=.03 and row['expected_xyz']==expected
        else:assert math.dist(row['world_xyz'],row['joint_sample']['xyz'])<=.03
    assert len(poses)>30 and {p['entity'] for p in poses}==set(names)
    periodic=[p for p in poses if p.get('reason')=='periodic_in_motion']
    assert len(periodic)>50 and {p['entity'] for p in periodic}==set(names)
    assert sum(max(abs(v) for v in p['joint_sample']['velocity'])>.1 for p in periodic)>20
    for name in names:
        stamps=[p['time_s'] for p in periodic if p['entity']==name]
        assert max(b-a for a,b in zip(stamps,stamps[1:]))<12
    state={name:'mounted' for name in items};bins={'cap_bin':[],'bearing_bin':[]};completed=[];station='infeed'
    reserved=None;current=None;commands={};tag=None;faults=[];stop_events=[];retried=[];part_times={}
    for seq,e in enumerate(events):
        assert e['seq']==seq and (seq==0 or e['time_s']>=events[seq-1]['time_s'])
        kind=e['event']
        if kind=='COMMAND_ISSUED':
            c=dict(e['command']);assert c['id'] not in commands and c['job']=='ws001'
            assert c['stage']==len(completed) and c['station']==recipe[len(completed)]['id']
            if current is not None:
                old=commands[current];assert old['status']=='failed' and old.get('stopped')
                assert (old['kind'],old['station'])==(c['kind'],c['station']);old['status']='superseded'
            elif c['kind']=='transfer':reserved=c['station']
            else:assert station==c['station']
            commands[c['id']]=c;current=c['id']
        elif kind in {'ARRIVAL_CONFIRMED','DEVICE_COMPLETE'}:
            c=commands[e['command_id']];assert c['status']=='active' and current==c['id']
            obs=e['observation'];target=recipe[c['stage']]['target']
            assert 0<=e['time_s']-obs['time_s']<=.15 and math.dist(obs['xyz'],target)<=.03
            assert max(abs(v) for v in obs['velocity'])<.035 and abs(obs['omega'])<.025
            assert abs(obs['probe'])<.005 and abs(obs['probe_velocity'])<.015
            if kind=='ARRIVAL_CONFIRMED':
                assert c['kind']=='transfer' and reserved==c['station'];station=c['station'];reserved=None
                points=[w for w in waypoints if w['command_id']==c['id']]
                assert points and math.dist(points[-1]['target'],target)<1e-9
            else:
                assert c['kind']=='device' and station==c['station']
                if station=='identification':assert tag=='DEMO-ws001'
                if station in {'opening','demount'}:
                    part_kind='cap' if station=='opening' else 'bearing'
                    assert all(state[p]=='binned' for p,i in items.items() if i['kind']==part_kind)
                completed.append(station)
            c['status']='done';current=None
        elif kind=='SIMULATED_TAG_READ':
            c=commands[e['command_id']];assert c['station']=='identification' and c['id']==current
            assert e['tag']=='DEMO-ws001' and tag is None;tag=e['tag']
        elif kind in {'PART_SEPARATED','PART_BIN_COMMITTED'}:
            c=commands[e['command_id']];name=e['part'];item=items[name];obs=e['observation']
            assert c['id']==current and c['status']=='active'
            assert c['station']==('opening' if item['kind']=='cap' else 'demount')
            assert max(abs(v) for v in obs['velocity'])<.035 and 0<=e['time_s']-obs['time_s']<=.15
            if kind=='PART_SEPARATED':
                assert state[name]=='mounted';target=list(recipe[c['stage']]['target'])
                target=[p+o for p,o in zip(target,item['offset'])];target[1]+=item['sign']*item['separation_m']
                assert math.dist(obs['xyz'],target)<.025;state[name]='separated'
            else:
                assert state[name]=='separated' and len(bins[item['bin']])<2
                assert e['bin']==item['bin'] and math.dist(obs['xyz'],item['destination'])<.025
                state[name]='binned';bins[item['bin']].append(name);part_times[name]=e['time_s']
            raw=min(samples,key=lambda s:abs(s['observations'][name]['time_s']-obs['time_s']))['observations'][name]
            assert abs(raw['time_s']-obs['time_s'])<=.15 and math.dist(raw['xyz'],obs['xyz'])<=.03
        elif kind in {'CAP_RETRIEVED','CAP_REFITTED'}:
            assert finishing;c=commands[e['command_id']];name=e['part'];obs=e['observation'];item=items[name]
            assert c['id']==current and c['status']=='active' and c['station']=='closing' and item['kind']=='cap'
            assert 0<=e['time_s']-obs['time_s']<=.15 and max(abs(v) for v in obs['velocity'])<.035
            if kind=='CAP_RETRIEVED':
                assert state[name]=='binned' and name in bins['cap_bin'];target=item['destination']
                bins['cap_bin'].remove(name);state[name]='retrieved'
            else:
                assert state[name]=='retrieved';target=[a+b for a,b in zip(FULL_STAGES[13]['target'],item['offset'])]
                state[name]='refitted'
            assert math.dist(obs['xyz'],target)<.025
            raw=min(samples,key=lambda s:abs(s['observations'][name]['time_s']-obs['time_s']))['observations'][name]
            corroborate_stopped(raw,obs)
        elif kind=='FINISH_MEASURED':
            assert finishing;c=commands[e['command_id']]
            assert c['id']==current and c['status']=='active' and c['station'] in {'run_in','painting'}
        elif kind=='FAULT_LATCHED':
            c=commands[e['command_id']];assert c['status']=='active' and c['id']==current
            c['status']='failed';faults.append(e)
        elif kind=='STOP_CONFIRMED':
            c=commands[e['command_id']];assert c['status']=='failed';c['stopped']=True;stop_events.append(e)
            assert e['time_s']-faults[-1]['time_s']>=.5
            assert all(max(abs(v) for v in s['velocity'])<.035 for s in e['observation'].values())
            assert abs(e['observation']['ws001']['omega'])<.025 and abs(e['observation']['ws001']['probe'])<.005
        elif kind=='RECOVERY_REISSUED':
            old=commands[e['old_command_id']];new=commands[e['command_id']]
            assert old['status']=='superseded' and old.get('stopped') and new['id']==current
            assert old['station']==new['station']==('press' if r['case']=='press_recovery' else 'demount');retried.append(e)
        else:raise AssertionError(kind)
        snap=e['snapshot']
        assert snap==dict(job='ws001',station=station,reserved=reserved,part_state=state,bins=bins,completed=completed,tag=tag)
    assert completed==r['completed']==[s['id'] for s in recipe] and current is None
    assert state==r['final_snapshot']['part_state']
    assert all(s==('refitted' if finishing and items[n]['kind']=='cap' else 'binned') for n,s in state.items())
    assert len(bins['bearing_bin'])==2 and len(bins['cap_bin'])==(0 if finishing else 2)
    measured_rotations=[]
    for token,c in commands.items():
        if c['status']!='done' or c['kind']!='device':continue
        stage=recipe[c['stage']];trace=[e for e in actions if e['command_id']==token]
        if stage['action']=='rotate':
            assert [e['event'] for e in trace]==['ACTION_STARTED','PROBE_ENGAGED','ROTATION_MEASURED','SPINDLE_STOPPED','ACTION_COMPLETE']
            points=[s['observations']['ws001'] for s in samples if s['command_id']==token
                and trace[1]['sim_time_s']<=s['observations']['ws001']['time_s']<=trace[2]['sim_time_s']]
            assert len(points)>20
            angle=sum(math.atan2(math.sin(b['spin']-a['spin']),math.cos(b['spin']-a['spin'])) for a,b in zip(points,points[1:]))
            assert angle>=2*math.pi and abs(angle-trace[2]['angle_rad'])<1e-7
            assert all(abs(p['probe']-.22)<.015 and math.dist(p['xyz'],stage['target'])<.03 for p in points)
            measured_rotations.append(dict(station=stage['id'],rotation_rad=angle))
        elif stage['action']=='scan':
            assert [e['event'] for e in trace]==['scan_extend','scan_retract']
            assert abs(trace[0]['sample']['probe']-.22)<.005 and abs(trace[1]['sample']['probe'])<.005
            for e in trace:
                obs=e['sample'];raw=min(samples,key=lambda s:abs(s['observations']['ws001']['time_s']-obs['time_s']))['observations']['ws001']
                assert obs==raw and math.dist(obs['xyz'],stage['target'])<.03
                assert abs(obs['probe_velocity'])<.015
    assert len(measured_rotations)==4
    # Every reached point must be stopped, supported by raw joint data and
    # independently queried world position, not only its waypoint status flag.
    for w in waypoints:
        obs=w['observation'];name=w['entity']
        assert math.dist(obs['xyz'],w['target'])<.025 and max(abs(v) for v in obs['velocity'])<.035
        raw=min(samples,key=lambda s:abs(s['observations'][name]['time_s']-obs['time_s']))['observations'][name]
        assert abs(raw['time_s']-obs['time_s'])<=.15 and math.dist(raw['xyz'],obs['xyz'])<=.03
        assert any(p['entity']==name and p['command_id']==w['command_id'] and abs(p['time_s']-obs['time_s'])<.15 for p in poses)
    if r['case']=='normal':assert not faults and not retried and not any(e['injected'] for e in efforts)
    elif r['case']=='demount_recovery':
        assert len(faults)==len(stop_events)==len(retried)==1 and r['superseded_completion_rejected']
        f=faults[0];old=commands[f['command_id']];assert old['station']=='demount'
        assert f['reason']=='measured_waypoint_timeout' and part_times['ws001_bearing_left']<f['time_s']
        assert f['snapshot']['bins']['bearing_bin']==['ws001_bearing_left']
        assert stop_events[0]['snapshot']['bins']==retried[0]['snapshot']['bins']
        injected=[e for e in efforts if e['injected']]
        assert len(injected)>20 and all(e['entity']=='ws001_bearing_right' and e['axis']=='y' and e['effort']==0 for e in injected)
        assert injected[-1]['time_s']-injected[0]['time_s']>=11.9
        assert not any(e['event']=='DEVICE_COMPLETE' and e['command_id']==old['id'] for e in events)
    for name,s in r['final_stop'].items():
        assert max(abs(v) for v in s['velocity'])<.035
        if name=='ws001':expected=recipe[-1]['target']
        elif name in items:
            expected=[a+b for a,b in zip(recipe[-1]['target'],items[name]['offset'])] if finishing and items[name]['kind']=='cap' else items[name]['destination']
        elif name in components():expected=[a+b for a,b in zip(recipe[-1]['target'],components()[name]['offset'])]
        else:expected=meta['definitions'][name]['origin']
        if finishing and name==STORED_JOB:expected=STORAGE_UNLOAD
        assert math.dist(s['xyz'],expected)<.03
    extra={}
    if assembly:
        from .verify_assembly import verify_assembly
        extra=verify_assembly(r,meta,samples,rows('stock_events'),rows('press_samples'),actions,events,poses,commands)
        if r['case']=='press_recovery':
            injected=[e for e in efforts if e['injected']]
            assert len(injected)>20 and injected[-1]['time_s']-injected[0]['time_s']>=11.9
            assert all(e['entity']=='press_ram_right' and e['axis']=='probe' and e['effort']==0. for e in injected)
    if finishing:
        from .verify_finishing import verify_finishing
        extra.update(verify_finishing(r,meta,samples,events,rows('storage_events'),actions,world,commands))
    swept=sum(measured_sweeps(samples,boxes,name,'wheelset' if meta['definitions'][name]['kind']=='wheelset' else 'part')
        for name in names if not name.startswith('press_ram_'))
    result=dict(passed=True,case=r['case'],workpieces_checked=1,ordered_operations_checked=len(recipe),
        mainline_front_operations=5,inspection_stations=3,independent_part_bin_commits=4,
        measured_rotations=measured_rotations,joint_sample_groups=len(samples),native_effort_records=len(efforts),
        measured_cargo_segment_box_tests=swept,static_collision_boxes=len(boxes),
        periodic_world_position_checks=len(periodic),
        partial_disassembly_recovery_checked=r['case']=='demount_recovery',claim_boundary=r['claim_boundary'],**extra)
    (directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    print(json.dumps(verify(args.directory)))


if __name__=='__main__':main()
