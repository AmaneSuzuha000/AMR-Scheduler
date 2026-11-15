"""Independent full-line audit: shared ownership, motion, parts and fleet supply."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .assembly_world import EXTRA_SNAPSHOTS
from .build import SCENE
from .evidence import build_fingerprint
from .finishing_flow import FULL_STAGES, STORAGE, STORAGE_UNLOAD, STORED_JOB
from .finishing_world import FINISH_SNAPSHOTS, finishing_static
from .front_flow import ORIGIN
from .front_flow import parts
from .assembly_flow import components
from .whole_line_evidence import Evidence, rows
from .whole_line_worker_audit import verify_workers


def verify_resources(events, jobs, workers, evidence):
    places=['infeed']+[s['id'] for s in FULL_STAGES]
    stock={p:set() for p in places}; reserved={p:set() for p in places}
    states={}; completed=[]; carrier=None; commands={}; faults=[]; stops=[]; retries=[]
    for seq,e in enumerate(events):
        assert e['seq']==seq and (seq==0 or e['time_s']>=events[seq-1]['time_s'])
        kind=e['event']; t=e['time_s']
        if kind=='ADMITTED':
            job=e['job']; assert job in jobs and job not in states and not stock['infeed']
            points=evidence.window(job,t-.05,t+.05)
            def admission(p=points,stamp=t):
                from .whole_line_evidence import stopped
                assert p
                stopped(min(p,key=lambda o:abs(o['time_s']-stamp)),target=ORIGIN,home=True)
            evidence.retained_checks.append(admission)
            stock['infeed'].add(job); states[job]=dict(index=0,state='ready',device_done=False,action=None)
        elif kind=='COMMAND_ISSUED':
            c=dict(e['command']); job=c['job']; state=states[job]; token=c['id']
            assert token not in commands and c['issued_at']==t and c['status']=='active'
            assert c['source']==places[state['index']] and job in stock[c['source']]
            old=commands.get(state['action'])
            if old:
                assert old['status']=='failed' and old['stop_confirmed']
                assert (old['kind'],old['source'],old['target'])==(c['kind'],c['source'],c['target'])
                old['status']='superseded'
                if old['kind']=='transfer': reserved[old['target']].remove(job); carrier=None
            else:
                assert state['state']=='ready'
            execution=workers[c['execution_id']]
            assert (execution['job'],execution['kind'],execution['station'],execution['issued_at'])==(job,c['kind'],c['target'],t)
            if c['kind']=='transfer':
                assert state['index']==0 or state['device_done']
                assert carrier is None and c['target']==places[state['index']+1]
                assert not stock[c['target']] and not reserved[c['target']]
                reserved[c['target']].add(job); carrier=token
            else:
                assert c['kind']=='device' and c['source']==c['target'] and not state['device_done']
            state.update(state=c['kind'],action=token); commands[token]=c
        elif kind in {'ARRIVAL_CONFIRMED','DEVICE_COMPLETE'}:
            c=commands[e['command_id']]; job=c['job']; state=states[job]
            assert c['status']=='active' and state['action']==c['id'] and e['job']==job
            assert workers[c['execution_id']]['status']=='done'
            execution=workers[c['execution_id']]
            assert execution['completed_at']==t and execution['observation']['time_s']==e['observed_at']
            assert c['issued_at']<=e['observed_at']<=t
            if kind=='ARRIVAL_CONFIRMED':
                assert c['kind']=='transfer' and carrier==c['id']
                stock[c['source']].remove(job); reserved[c['target']].remove(job); stock[c['target']].add(job)
                carrier=None; state.update(index=state['index']+1,device_done=False)
            else:
                assert c['kind']=='device'; state['device_done']=True
            c['status']='done'; state.update(state='ready',action=None)
        elif kind=='FAULT_LATCHED':
            c=commands[e['command_id']]; assert c['status']=='active' and e['job']==c['job']
            c['status']='failed'; states[c['job']]['state']='failed'; faults.append(e)
        elif kind=='STOP_CONFIRMED':
            c=commands[e['command_id']]; assert c['status']=='failed' and t>=faults[-1]['time_s']+.5
            assert faults[-1]['time_s']<=e['observed_at']<=t
            c['stop_confirmed']=True; stops.append(e)
        elif kind=='RECOVERY_REISSUED':
            old=commands[e['old_command_id']]; new=commands[e['command_id']]
            assert old['status']=='superseded' and old['stop_confirmed'] and new['status']=='active'
            assert old['job']==new['job']; retries.append(e)
        elif kind=='OUTFEED_RELEASED':
            job=e['job']; state=states[job]
            assert state==dict(index=18,state='ready',device_done=True,action=None)
            stock['storage'].remove(job); state['state']='complete'; completed.append(job)
        elif kind=='PART_BIN_WAIT':
            assert states[e['job']]['state']=='ready' and e['station'] in {'opening','demount'}
        else:
            raise AssertionError(kind)
        assert all(len(stock[p])+len(reserved[p])<=1 for p in places)
        assert all(sum(j in group for group in stock.values())==(0 if s['state']=='complete' else 1) for j,s in states.items())
        snap=dict(stock={p:sorted(v) for p,v in stock.items()},reserved={p:sorted(v) for p,v in reserved.items()},
            carrier=carrier,jobs=copy.deepcopy(states),completed=list(completed))
        assert e['snapshot']==snap
    assert set(completed)==set(jobs) and all(s['state']=='complete' for s in states.values())
    assert len(commands)==len(workers) and {c['execution_id'] for c in commands.values()}==set(workers)
    return commands,dict(resource_commands=len(commands),faults=len(faults),stop_confirmations=len(stops),retries=len(retries))


def verify_storage(directory, report, evidence, workers):
    occupant=STORED_JOB; reserved=None; token=None; released=[]; committed=[]; issued={}; waits=[]
    events=list(rows(directory,'storage_events'))
    for seq,e in enumerate(events):
        assert e['seq']==seq and (seq==0 or e['time_s']>=events[seq-1]['time_s'])
        kind=e['event']; t=e['time_s']; job=e['job']
        if kind=='STORAGE_BLOCKED':
            c=workers[e['command_id']]
            assert occupant is not None or reserved is not None
            assert c['job']==job and c['station']=='storage_wait'; waits.append(e)
        elif kind=='STORAGE_RELEASE_ISSUED':
            assert job==occupant and token is None and reserved is None
            token=e['command_id']; assert token not in issued; issued[token]=e
        elif kind=='STORAGE_RELEASED':
            assert job==occupant and token==e['command_id'] and e['observation']['time_s']>=issued[token]['time_s']
            evidence.require(job,e['observation'],now=t,target=STORAGE_UNLOAD,home=True)
            points=[w for w in rows(directory,'unload_waypoints') if w['command_id']==token]
            assert [w['index'] for w in points]==[0,1,2,3]
            for w in points:
                assert w['job']==job and issued[token]['time_s']<=w['time_s']<=t
                evidence.require(job,w['observation'],now=w['time_s'],target=w['target'])
            released.append(e); occupant=None; token=None
        elif kind=='STORAGE_RESERVED':
            assert occupant is None and reserved is None
            c=workers[e['command_id']]; assert c['job']==job and c['station']=='storage_wait'
            assert c['issued_at']<=t<=c['completed_at']
            reserved=dict(job=job,command_id=e['command_id'])
        elif kind=='STORAGE_COMMITTED':
            assert occupant is None and reserved==dict(job=job,command_id=e['command_id'])
            evidence.require(job,e['observation'],now=t,target=STORAGE,home=True)
            transfer=next(c for c in workers.values() if c['job']==job and c['station']=='storage' and c['kind']=='transfer')
            reservation=next(x for x in events if x['event']=='STORAGE_RESERVED' and x['job']==job)
            assert transfer['issued_at']>=reservation['time_s'] and transfer['completed_at']<=t
            finish=next(c for c in workers.values() if c['job']==job and c['station']=='storage' and c['kind']=='device')
            assert finish['issued_at']<=t<=finish['completed_at']
            occupant=job; reserved=None; committed.append(e)
        else:
            raise AssertionError(kind)
        assert e['snapshot']==dict(capacity=1,occupant=occupant,reserved=reserved,release_token=token)
    assert [e['job'] for e in released]==[STORED_JOB,*report['completed']]
    assert {e['job'] for e in committed}==set(report['completed']) and len(committed)==2
    assert occupant is None and reserved is None and token is None
    assert report['storage_snapshot']==events[-1]['snapshot'] and waits
    retired=list(rows(directory,'retired')); expected={STORED_JOB}
    for job in report['completed']:
        expected.add(job)
        expected.update(n for n in report['metadata']['worker_geometry'][job]['parts'] if '_cap_' in n)
        expected.update(report['metadata']['worker_geometry'][job]['components'])
    assert len(retired)==len(expected) and {e['entity'] for e in retired}==expected
    for e in retired:
        n=e['entity']; job=n if n in [STORED_JOB,*report['completed']] else n.split('_')[0]
        release=next(r for r in released if r['job']==job)
        assert e['time_s']>=release['time_s']
        target=STORAGE_UNLOAD
        if n!=job:
            geometry=report['metadata']['worker_geometry'][job]
            item=geometry['parts'].get(n) or geometry['components'][n]
            target=[a+b for a,b in zip(target,item['offset'])]
        evidence.require(n,e['observation'],target=target)
    return retired,dict(storage_full_wait_events=len(waits),measured_storage_releases=3,stored_workpieces=2)


def verify_supply(directory, report, evidence, workers, *, localized_audits=None):
    from .supply_bridge import SOURCES
    stock=dict(report['supply_initial']); assert stock=={'derust':0,'magnetic':0}
    events=list(rows(directory,'supply_events')); deliveries={}; active=None; charged={}; waits=set(); commits={}
    routes={c['route_id']:c for c in report['routes']}; tote_position=evidence.definitions['line_supply_tote']['origin']
    feedback=list(rows(directory,'feedback')) if localized_audits is None else None
    for seq,e in enumerate(events):
        assert e['seq']==seq and (seq==0 or e['time_s']>=events[seq-1]['time_s'])
        assert abs(e['source_time_s']-e['time_s'])<=.15
        kind=e['event']; t=e['source_time_s']
        if kind=='DOCK_RECEIPT':
            d=e['delivery']; c=e['command']; obs=e['observation']; task=d['task_id']
            assert task not in deliveries and routes[c['route_id']]==c
            assert (d['station'],d['quantity'],c['nodes'][-1])==({'inspection_consumables':('magnetic',4,'receiving'),'small_tools':('derust',2,'supply')}[task])
            assert c['task_id']==task and c['robot_id']==d['robot'] and d['route_id']==c['route_id']
            assert math.dist(obs['pose'][:2],c['target_xy'])<=.10 and obs['linear_speed']<.025 and obs['angular_speed']<.05
            assert 0<=t-obs['sim_time_s']<=.15
            if localized_audits is None:
                matches=[f for f in feedback if f['message']['event']=='ARRIVED' and f['message']['route_id']==c['route_id'] and f['answer']['accepted']]
                assert len(matches)==1 and 0<=obs['sim_time_s']-matches[0]['message']['timestamp']<1.
                assert any(row['observations'][d['robot']]==obs and row['tasks'][task]=='completed'
                    for row in rows(directory,'trace'))
            else:
                from .navigation_dock import localized_dock_valid
                assert e['dock_source']=='localized' and c['source']=='localized'
                assert localized_dock_valid(c,obs,e['dock_proof'],c['run_id'],t)
                arrival=localized_audits[d['robot']]['arrivals'][c['route_id']]
                assert arrival['message']==e['dock_proof']['feedback'] and arrival['result']==e['dock_proof']['feedback_result']
            deliveries[task]=dict(d)
        elif kind=='HANDOFF_ISSUED':
            c=e['command']; assert active is None and deliveries[c['task_id']]['status']=='at_dock'
            active=dict(c); deliveries[c['task_id']]['status']='handoff'
        elif kind=='SUPPLY_COMMITTED':
            task=e['task_id']; d=deliveries[task]
            assert active and active['token']==e['token'] and active['task_id']==task and task not in commits
            assert e['station']==d['station'] and e['quantity']==d['quantity'] and d['status']=='handoff'
            assert e['target']==report['supply_destinations'][d['station']]
            evidence.require('line_supply_tote',e['observation'],now=t,target=e['target'])
            assert e['observation']['time_s']>=active['issued_at']
            points=[p for p in rows(directory/'process','tote_events') if p['token']==e['token']]
            assert [p['waypoint'] for p in points]==list(range(6))
            source=SOURCES[task]; destination=e['target']
            expected=[(*tote_position[:2],6.),(*source[:2],6.),source,(*source[:2],6.),(*destination[:2],6.),destination]
            for index,(p,target) in enumerate(zip(points,expected)):
                assert p['task_id']==task and active['issued_at']<=p['time_s']<=t
                # The next lift begins at the measured stopped destination,
                # whose sub-centimetre residual need not equal its set point.
                assert math.dist(p['target'],target)<(.025 if index==0 else 1e-9)
                assert p['target'][2]==target[2]
                evidence.require('line_supply_tote',p['observation'],now=p['time_s'],target=p['target'])
            assert all(a['time_s']<=b['time_s'] for a,b in zip(points,points[1:]))
            tote_position=destination
            stock[d['station']]+=d['quantity']; d['status']='committed'; commits[task]=e; active=None
        elif kind=='MATERIAL_WAIT':
            key=e['job'],e['station']; assert stock[e['station']]==0 and key not in waits; waits.add(key)
        elif kind=='MATERIAL_CONSUMED':
            key=e['job'],e['station']; c=workers[e['command_id']]
            assert key not in charged and (c['job'],c['station'],c['kind'])==(*key,'device')
            assert c['issued_at']==t and stock[e['station']]>0 and e['resumed_wait']==(key in waits)
            charged[key]=e; stock[e['station']]-=1
        else:
            raise AssertionError(kind)
        assert all(0<=v<=report['supply_capacity'] for v in stock.values()) and e['stock']==stock
    assert active is None and len(commits)==2 and stock==report['supply_final']=={'derust':0,'magnetic':2}
    assert deliveries==report['supply_deliveries'] and set(charged)=={(j,s) for j in report['process_report']['completed'] for s in stock}
    for c in workers.values():
        if c['kind']=='device' and c['station'] in stock:
            assert charged[c['job'],c['station']]['time_s']<=c['issued_at']
    return dict(measured_fleet_dock_receipts=2,native_tote_waypoints=12,material_charges=len(charged),material_waits=len(waits))


def verify(directory, *, fleet=True, sweep=True, write=True):
    directory=Path(directory); report=json.loads((directory/'result.json').read_text())
    process=directory/'process'; r=json.loads((process/'result.json').read_text())
    assert report['passed'] and report['runtime_error'] is None and report['mode']=='whole_line'
    assert report['process_report']==r and r['passed'] and r['runtime_error'] is None
    assert report['owned_server_stopped'] and report['baseline_session_preserved'] and r['owned_server_stopped'] and r['baseline_session_preserved']
    assert r['build_fingerprint']==build_fingerprint()
    for base,result in [(directory,report),(process,r)]:
        for name,digest in result['data_sha256'].items():
            hasher=hashlib.sha256()
            with (base/name).open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''): hasher.update(chunk)
            assert hasher.hexdigest()==digest,name
    for name,digest in report['source_sha256'].items():
        assert hashlib.sha256((directory/'executed_sources'/name).read_bytes()).hexdigest()==digest,name
    assert len(report['images'])>5 and len(report['images'])==len(r['images'])
    for image,shared in zip(report['images'],r['images']):
        assert hashlib.sha256((directory/image['file']).read_bytes()).hexdigest()==image['sha256']==shared['sha256']
        assert shared['file']=='../'+image['file'] and shared['time_s']==image['sim_time_s']
    meta=json.loads((directory/'whole_line_metadata.json').read_text()); assert meta==r['metadata']
    assert meta['jobs']==['ws001','ws002'] and set(r['completed'])==set(meta['jobs']) and r['job_count']==2
    assert meta['stages']==json.loads(json.dumps(FULL_STAGES))
    planned=json.loads((directory/'whole_line_clearance.json').read_text())
    assert planned['passed'] and all(not p['hits'] for p in planned['checks'])
    scene=json.loads(SCENE.read_text(encoding='utf-8')); derived=copy.deepcopy(scene)
    removed=meta['removed_static_snapshot_ids']
    expected={i['id'] for i in scene['instances'] if i['asset']=='wheelset' and (abs(i['pose'][1]-9)<1e-6 or abs(i['pose'][1]-50)<1e-6)}|EXTRA_SNAPSHOTS|FINISH_SNAPSHOTS
    assert set(removed)==expected
    derived['instances']=[i for i in scene['instances'] if i['id'] not in removed]
    boxes,sources=finishing_static(derived)
    assert boxes==meta['static_obstacles'] and sources==meta['collision_source_sha256']
    world=ET.parse(directory/'light_material.world').getroot().find('world')
    definitions=meta['definitions']; assert len(definitions)==23
    names={STORED_JOB,'spray_tool','line_supply_tote','press_ram_left','press_ram_right',*meta['jobs']}
    for index,job in enumerate(meta['jobs']):
        item_geometry=parts(job); replacement_geometry=components(job)
        for item in item_geometry.values():
            p=item['destination']; item['destination']=(p[0]+index*.70*(-1 if item['kind']=='cap' else 1),p[1],p[2])
        if index:
            for item in replacement_geometry.values():
                p=item['receiving']; item['origin']=(p[0]+index*2.,p[1],6.)
        expected=json.loads(json.dumps(dict(parts=item_geometry,components=replacement_geometry)))
        assert meta['worker_geometry'][job]==expected
        names.update(item_geometry);names.update(replacement_geometry)
    assert set(definitions)==names
    native_build=json.loads((directory/'executed_sources/build_manifest.json').read_text())
    assert native_build['passed'] and native_build['library_sha256']==report['source_sha256']['libwheelset_axis_servo.so']
    assert all(report['source_sha256']['native_plugin/'+name]==digest for name,digest in native_build['source_sha256'].items())
    for name,d in definitions.items():
        model=world.find(f"model[@name='{name}']"); assert model is not None
        assert list(map(float,model.findtext('pose').split()))==[*d['origin'],0.,0.,0.]
        assert not model.findall('.//collision') and all(link.findtext('gravity')=='false' for link in model.findall('link'))
        assert len(model.findall('joint'))==(6 if d['kind']=='wheelset' else 2 if d['kind']=='ram' else 5)
        assert len(model.findall("plugin[@filename='libwheelset_axis_servo.so']"))==1
        assert model.findtext("plugin[@filename='libwheelset_axis_servo.so']/lease_s")=='1.0'
    evidence=Evidence(process,definitions)
    workers,deferred,summaries=verify_workers(process,r,meta,evidence)
    resource=list(rows(process,'events'))
    commands,ownership=verify_resources(resource,meta['jobs'],workers,evidence)
    assert resource[-1]['snapshot']==r['final_snapshot']
    retired,storage=verify_storage(process,r,evidence,workers)
    supply=verify_supply(directory,report,evidence,workers)
    poses=list(rows(process,'pose_checks')); assert len(poses)>50 and {p['entity'] for p in poses}==set(definitions)
    for p in poses:
        obs=p['joint_sample']; name=p['entity']; d=definitions[name]; expected=list(obs['xyz'])
        if d['kind']=='ram': expected[1]+=d['direction_y']*obs['probe']
        assert expected==p['expected_xyz'] and p['error_m']==math.dist(expected,p['world_xyz'])<=.03
        assert p['source_time_error_s']==abs(p['time_s']-obs['time_s'])<=.05
        evidence.require(name,obs,stop=False)
    assert set(r['final_stop'])==set(definitions)-{e['entity'] for e in retired}
    for n,obs in r['final_stop'].items(): evidence.require(n,obs,home=True)
    raw,disabled=evidence.read(retired,commands)
    assert raw['simultaneous_action_samples']>20
    for check in [*deferred,*evidence.retained_checks]: check()
    if report['case']=='normal':
        assert not disabled and ownership['faults']==ownership['retries']==0 and not r['faults']
    else:
        assert report['case']=='feedback_recovery' and r['case']=='stalled_recovery'
        assert ownership['faults']==ownership['stop_confirmations']==ownership['retries']==len(r['faults'])==1
        assert disabled and all(c['entity']=='press_ram_right' and c['disabled_axes']==['probe'] for c in disabled)
        assert disabled[-1]['time_s']-disabled[0]['time_s']>=11.9
    fleet_result=None
    if fleet:
        from .verify_light_material_run import verify as verify_fleet
        fleet_result=verify_fleet(directory,write=False)
    swept=evidence.sweeps(boxes) if sweep else None
    result=dict(passed=True,case=report['case'],workpieces_checked=2,ordered_operations_checked=36,
        **ownership,**storage,**supply,**raw,workers=summaries,fleet=fleet_result,
        world_position_checks=len(poses),measured_cargo_segment_box_tests=swept,
        static_collision_boxes=len(boxes),claim_boundary=report['claim_boundary'],
        result_sha256=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest(),
        process_result_sha256=hashlib.sha256((process/'result.json').read_bytes()).hexdigest())
    if write: (directory/'whole_line_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('directory',type=Path)
    args=p.parse_args(); print(json.dumps(verify(args.directory)))


if __name__=='__main__': main()
