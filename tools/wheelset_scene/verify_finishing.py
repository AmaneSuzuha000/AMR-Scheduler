"""Reconstruct cap handling, spindle dwell, spray traverse and occupied-slot release."""
from bisect import bisect_left
import math

from .finishing_flow import FULL_STAGES,PAINT,STORAGE,STORAGE_APPROACH,STORAGE_UNLOAD,STORED_JOB
from .finishing_world import RACK,rack_collision_boxes
from .feedback_evidence import corroborate_stopped


def verify_finishing(result,meta,samples,events,storage,actions,world,commands):
    by_name={n:[s['observations'][n] for s in samples] for n in meta['definitions']}
    times={n:[s['time_s'] for s in rows] for n,rows in by_name.items()}
    def raw(name,obs):
        index=bisect_left(times[name],obs['time_s']);actual=min(by_name[name][max(0,index-1):index+2],key=lambda s:abs(s['time_s']-obs['time_s']))
        return actual
    def at(name,obs,target,now):
        corroborate_stopped(raw(name,obs),obs);assert 0<=now-obs['time_s']<=.15 and math.dist(obs['xyz'],target)<.025
        assert max(abs(v) for v in obs['velocity'])<.035 and abs(obs['omega'])<.025
    model=world.find(f"model[@name='{RACK}']");assert model is not None
    collisions=model.findall('link/collision');expected=rack_collision_boxes()
    assert meta['rack_structural_collision_boxes']==expected and len(collisions)==len(expected)
    for c,b in zip(collisions,expected):
        pose=list(map(float,c.findtext('pose').split()));size=list(map(float,c.findtext('geometry/box/size').split()));bounds=b['bounds']
        assert c.get('name')==b['id'].split('/')[1] and pose[3:]==[0.,0.,0.]
        assert all(abs(pose[i]-(bounds[i]+bounds[i+3])/2)<1e-9 and abs(size[i]-(bounds[i+3]-bounds[i]))<1e-9 for i in range(3))
    cap_pick=[e for e in events if e['event']=='CAP_RETRIEVED'];cap_fit=[e for e in events if e['event']=='CAP_REFITTED']
    assert len(cap_pick)==len(cap_fit)==2
    measured={commands[e['command_id']]['station']:e for e in events if e['event']=='FINISH_MEASURED'}
    assert set(measured)=={'run_in','painting'}
    run=measured['run_in'];token=run['command_id'];trace=[e for e in actions if e.get('command_id')==token]
    assert trace==run['evidence']['events'] and run['evidence']['kind']=='run_in'
    assert [e['event'] for e in trace]==['RUN_IN_STARTED','RUN_IN_SPEED_DWELL_MEASURED','RUN_IN_STOPPED']
    points=[s for s in by_name['ws001'] if trace[0]['time_s']<=s['time_s']<=trace[2]['time_s']]
    assert len(points)>100
    rotation=0.;cursor=0
    for index,s in enumerate(points):
        assert raw('ws001',s)==s
        assert math.dist(s['xyz'],FULL_STAGES[14]['target'])<.025 and max(abs(v) for v in s['velocity'])<.035
        assert abs(s['probe'])<.005 and abs(s['probe_velocity'])<.015
        if index:
            previous=points[index-1];delta=math.atan2(math.sin(s['spin']-previous['spin']),math.cos(s['spin']-previous['spin']))
            assert abs(delta)<=2.5*(s['time_s']-previous['time_s'])+.01;rotation+=delta
        if cursor<len(trace) and s['time_s']==trace[cursor]['time_s']:
            e=trace[cursor];assert e['angle']==s['spin'] and e['omega']==s['omega']
            assert abs(e['rotation_rad']-rotation)<1e-8;cursor+=1
    assert cursor==3 and trace[1]['rotation_rad']>=4*math.pi
    stable=[s for s in points if trace[1]['stable_since_s']<=s['time_s']<=trace[1]['time_s']]
    assert trace[1]['stable_s']==trace[1]['time_s']-trace[1]['stable_since_s']>=8
    assert stable and stable[0]['time_s']==trace[1]['stable_since_s'] and all(abs(s['omega']-1.6)<=.08 for s in stable)
    stopped=[s for s in points if trace[2]['time_s']-trace[2]['stopped_s']<=s['time_s']<=trace[2]['time_s']]
    assert trace[2]['stopped_s']>=.25 and len(stopped)>2 and all(abs(s['omega'])<=.025 for s in stopped)
    assert run['time_s']>=trace[2]['time_s']
    paint=measured['painting'];traverse=paint['evidence']['traverse'];assert len(traverse)==3 and paint['evidence']['kind']=='paint'
    targets=[(PAINT[0]-.7,PAINT[1],PAINT[2]+.85),(PAINT[0]+.7,PAINT[1],PAINT[2]+.85),(PAINT[0]-.7,PAINT[1],PAINT[2]+.85)]
    for row,target in zip(traverse,targets):
        assert math.dist(row['target'],target)<1e-9
        at('spray_tool',row['tool'],target,row['tool']['time_s']);at('ws001',row['fixture'],PAINT,row['fixture']['time_s'])
        assert abs(row['tool']['time_s']-row['fixture']['time_s'])<=.1 and row['tool']['time_s']<=paint['time_s']
    assert traverse[0]['tool']['time_s']<traverse[1]['tool']['time_s']<traverse[2]['tool']['time_s']
    assert math.dist(traverse[0]['tool']['xyz'],traverse[1]['tool']['xyz'])>1.37
    assert math.dist(traverse[1]['tool']['xyz'],traverse[2]['tool']['xyz'])>1.37
    assert [e['event'] for e in storage]==['STORAGE_BLOCKED','STORAGE_RELEASE_ISSUED','STORAGE_RELEASED','STORAGE_RESERVED','STORAGE_COMMITTED']
    occupant=STORED_JOB;reserved=None;release_token=None
    for seq,e in enumerate(storage):
        assert e['seq']==seq and (seq==0 or e['time_s']>=storage[seq-1]['time_s'])
        kind=e['event']
        if kind=='STORAGE_BLOCKED':
            assert occupant==STORED_JOB and e['job']=='ws001' and commands[e['command_id']]['station']=='storage_wait'
        elif kind=='STORAGE_RELEASE_ISSUED':
            assert occupant==STORED_JOB and reserved is None
            release_token=e['command_id'];assert e['time_s']-storage[0]['time_s']>=3
        elif kind=='STORAGE_RELEASED':
            assert e['command_id']==release_token and occupant==STORED_JOB
            at(STORED_JOB,e['observation'],STORAGE_UNLOAD,e['time_s']);occupant=None;release_token=None
        elif kind=='STORAGE_RESERVED':
            assert occupant is None and reserved is None and e['job']=='ws001'
            reserved=dict(job='ws001',command_id=e['command_id'])
        else:
            assert occupant is None and reserved==dict(job=e['job'],command_id=e['command_id'])
            at('ws001',e['observation'],STORAGE,e['time_s']);occupant='ws001';reserved=None
        assert e['snapshot']==dict(capacity=1,occupant=occupant,reserved=reserved,release_token=release_token)
    assert result['storage_snapshot']==storage[-1]['snapshot']
    arrival=next(e for e in events if e['event']=='ARRIVAL_CONFIRMED' and commands[e['command_id']]['station']=='storage')
    departure=next(e for e in events if e['event']=='COMMAND_ISSUED' and e['command']['station']=='storage' and e['command']['kind']=='transfer')
    assert departure['time_s']>=storage[3]['time_s'] and storage[4]['time_s']>=arrival['time_s']
    waiting=[s for s in by_name['ws001'] if storage[0]['time_s']<=s['time_s']<=storage[3]['time_s']]
    assert waiting and all(math.dist(s['xyz'],FULL_STAGES[16]['target'])<.025 for s in waiting)
    old=[s for s in by_name[STORED_JOB] if s['time_s']<=storage[1]['time_s']]
    assert old and all(math.dist(s['xyz'],STORAGE)<.025 for s in old)
    assert all(commands[e['command_id']]['status']=='done' for e in measured.values())
    return dict(caps_retrieved_and_refitted=2,run_in_rotation_rad=trace[1]['rotation_rad'],run_in_speed_dwell_s=trace[1]['stable_s'],
        spray_traverses_checked=2,finite_storage_capacity=1,storage_full_wait_s=storage[3]['time_s']-storage[0]['time_s'],
        measured_prior_storage_release=True,final_stored_job='ws001',refined_rack_collision_boxes=len(expected))
