"""Reconstruct each identified worker without invoking its runtime/action code."""
import math

from .finishing_flow import FULL_STAGES, PAINT
from .whole_line_evidence import rows


def check_rotation(points, trace, stage):
    assert len(points) > 20
    assert points[0]['time_s'] == trace[1]['sim_time_s']
    assert points[-1]['time_s'] == trace[2]['sim_time_s']
    angle = sum(math.atan2(math.sin(b['spin']-a['spin']), math.cos(b['spin']-a['spin']))
                for a, b in zip(points, points[1:]))
    assert angle >= 2*math.pi and abs(angle-trace[2]['angle_rad']) < 1e-7
    assert all(abs(s['probe']-.22)<.015 and math.dist(s['xyz'],stage['target'])<.03 for s in points)


def check_run_in(points, trace, target):
    assert len(points)>100 and points[0]['time_s']==trace[0]['time_s'] and points[-1]['time_s']==trace[-1]['time_s']
    rotation=0.; cursor=0
    for index, s in enumerate(points):
        assert math.dist(s['xyz'],target)<.025 and max(abs(v) for v in s['velocity'])<.035
        assert abs(s['probe'])<.005 and abs(s['probe_velocity'])<.015
        if index:
            old=points[index-1]; dt=s['time_s']-old['time_s']
            delta=math.atan2(math.sin(s['spin']-old['spin']),math.cos(s['spin']-old['spin']))
            assert abs(delta)<=2.5*dt+.01; rotation+=delta
        if cursor<len(trace) and s['time_s']==trace[cursor]['time_s']:
            e=trace[cursor]
            assert (e['angle'],e['omega'])==(s['spin'],s['omega']) and abs(e['rotation_rad']-rotation)<1e-8
            cursor+=1
    assert cursor==3 and trace[1]['rotation_rad']>=4*math.pi
    stable=[s for s in points if trace[1]['stable_since_s']<=s['time_s']<=trace[1]['time_s']]
    assert trace[1]['stable_s']==trace[1]['time_s']-trace[1]['stable_since_s']>=8
    assert stable and stable[0]['time_s']==trace[1]['stable_since_s']
    assert all(abs(s['omega']-1.6)<=.08 for s in stable)
    stopped=[s for s in points if trace[2]['time_s']-trace[2]['stopped_s']<=s['time_s']<=trace[2]['time_s']]
    assert trace[2]['stopped_s']>=.25 and len(stopped)>2 and all(abs(s['omega'])<=.025 for s in stopped)


def bind_press_samples(events, samples, jobs):
    """Bind the global press stream before replaying workers one at a time."""
    issued = {}
    grouped = {job: [] for job in jobs}
    for event in events:
        if event['event'] == 'COMMAND_ISSUED':
            command = event['command']
            assert command['id'] not in issued and command['job'] in grouped
            issued[command['id']] = command
    for sample in samples:
        token = sample['command_id']; assert token in issued
        command = issued[token]; job = command['job']
        assert command['kind'] == 'device' and command['station'] == 'press'
        assert sample['fixture']['entity'] == job and sample['side'] in ('left', 'right')
        grouped[job].append(sample)
    return grouped


def verify_workers(directory, report, meta, evidence):
    events=list(rows(directory,'worker_events')); stocks=list(rows(directory,'stock_events'))
    actions=list(rows(directory,'device_events')); waypoints=list(rows(directory,'waypoints'))
    press=list(rows(directory,'press_samples')); commands={}; terminal={}; deferred=[]; all_bin_events=[]
    press_by_job=bind_press_samples(events,press,meta['jobs'])
    summaries=[]
    for job in meta['jobs']:
        items=meta['worker_geometry'][job]['parts']; components=meta['worker_geometry'][job]['components']
        state={n:'mounted' for n in items}; bins={'cap_bin':[],'bearing_bin':[]}
        completed=[]; station='infeed'; reserved=None; current=None; tag=None
        local=[e for e in events if e['job']==job]; faults=[]; stops=[]; retries=[]; measured={}
        for seq, e in enumerate(local):
            assert e['seq']==seq and (seq==0 or e['time_s']>=local[seq-1]['time_s'])
            kind=e['event']; t=e['time_s']
            if kind=='COMMAND_ISSUED':
                c=dict(e['command']); assert c['job']==job and c['id'] not in commands
                assert c['stage']==len(completed) and c['station']==FULL_STAGES[len(completed)]['id']
                assert c['issued_at']==t and c['status']=='active'
                if current is not None:
                    old=commands[current]; assert old['status']=='failed' and old.get('stopped')
                    assert (old['kind'],old['station'])==(c['kind'],c['station']); old['status']='superseded'
                elif c['kind']=='transfer':
                    reserved=c['station']
                else:
                    assert c['kind']=='device' and station==c['station']
                commands[c['id']]=c; current=c['id']
            elif kind in {'ARRIVAL_CONFIRMED','DEVICE_COMPLETE'}:
                c=commands[e['command_id']]; assert c['id']==current and c['status']=='active'
                target=FULL_STAGES[c['stage']]['target']
                evidence.require(job,e['observation'],now=t,target=target,home=True)
                assert e['observation']['time_s']>=c['issued_at']
                if kind=='ARRIVAL_CONFIRMED':
                    assert c['kind']=='transfer' and reserved==c['station']; station=c['station']; reserved=None
                    points=[w for w in waypoints if w['command_id']==c['id'] and w['entity']==job]
                    assert points and math.dist(points[-1]['target'],target)<1e-9
                else:
                    assert c['kind']=='device' and station==c['station']
                    if station=='identification': assert tag=='DEMO-'+job
                    if station in {'opening','demount'}:
                        assert all(state[n]=='binned' for n,i in items.items() if i['kind']==('cap' if station=='opening' else 'bearing'))
                    completed.append(station)
                c['status']='done'; c['completed_at']=t; c['observation']=e['observation']; current=None; terminal[c['id']]=t
            elif kind=='SIMULATED_TAG_READ':
                c=commands[e['command_id']]
                assert c['id']==current and c['station']=='identification' and tag is None and e['tag']=='DEMO-'+job
                tag=e['tag']
            elif kind in {'PART_SEPARATED','PART_BIN_COMMITTED','CAP_RETRIEVED','CAP_REFITTED'}:
                c=commands[e['command_id']]; n=e['part']; item=items[n]
                assert c['id']==current and c['status']=='active'
                if kind=='PART_SEPARATED':
                    assert c['station']==('opening' if item['kind']=='cap' else 'demount') and state[n]=='mounted'
                    target=[a+b for a,b in zip(FULL_STAGES[c['stage']]['target'],item['offset'])]
                    target[1]+=item['sign']*item['separation_m']; state[n]='separated'
                elif kind=='PART_BIN_COMMITTED':
                    assert state[n]=='separated' and e['bin']==item['bin']
                    target=item['destination']; state[n]='binned'; bins[item['bin']].append(n)
                    all_bin_events.append(e)
                elif kind=='CAP_RETRIEVED':
                    assert c['station']=='closing' and item['kind']=='cap' and state[n]=='binned'
                    bins['cap_bin'].remove(n); state[n]='retrieved'; target=item['destination']; all_bin_events.append(e)
                else:
                    assert c['station']=='closing' and item['kind']=='cap' and state[n]=='retrieved'
                    target=[a+b for a,b in zip(FULL_STAGES[13]['target'],item['offset'])]; state[n]='refitted'
                evidence.require(n,e['observation'],now=t,target=target)
            elif kind=='FINISH_MEASURED':
                c=commands[e['command_id']]
                assert c['id']==current and c['status']=='active' and c['station'] in {'run_in','painting'}
                assert c['station'] not in measured; measured[c['station']]=e
            elif kind=='FAULT_LATCHED':
                c=commands[e['command_id']]; assert c['id']==current and c['status']=='active'
                c['status']='failed'; faults.append(e); terminal[c['id']]=t
            elif kind=='STOP_CONFIRMED':
                c=commands[e['command_id']]; assert c['status']=='failed'
                assert faults and t-faults[-1]['time_s']>=.5
                required={job,*items,*components}
                if c['station']=='press': required.update({'press_ram_left','press_ram_right'})
                assert required<=set(e['observation'])
                for n in required:
                    obs=e['observation'][n]; assert obs['time_s']>=faults[-1]['time_s']
                    evidence.require(n,obs,now=t,home=True)
                c['stopped']=True; stops.append(e)
            elif kind=='RECOVERY_REISSUED':
                old=commands[e['old_command_id']]; new=commands[e['command_id']]
                assert old['status']=='superseded' and old.get('stopped') and current==new['id']
                assert old['station']==new['station']=='press'; retries.append(e)
            else:
                raise AssertionError(kind)
            snap=e['snapshot']
            assert {k:v for k,v in snap.items() if k!='bins'}==dict(job=job,station=station,reserved=reserved,
                part_state=state,completed=completed,tag=tag)
            for b in bins:
                assert [n for n in snap['bins'][b] if n in items]==bins[b]
                assert len(snap['bins'][b])<=4 and len(set(snap['bins'][b]))==len(snap['bins'][b])
        assert current is None and completed==[s['id'] for s in FULL_STAGES]
        assert all(v==('refitted' if items[n]['kind']=='cap' else 'binned') for n,v in state.items())
        assert {k:v for k,v in report['worker_snapshots'][job].items() if k!='bins'}=={k:v for k,v in local[-1]['snapshot'].items() if k!='bins'}
        rotation_count=0
        for token,c in commands.items():
            if c['job']!=job or c['kind']!='device' or c['status']!='done': continue
            stage=FULL_STAGES[c['stage']]; trace=[a for a in actions if a.get('command_id')==token]
            if stage['action']=='scan':
                assert [a['event'] for a in trace]==['scan_extend','scan_retract']
                for a,extension in zip(trace,[.22,0.]):
                    obs=a['sample']; assert abs(obs['probe']-extension)<.005 and abs(obs['probe_velocity'])<.015
                    evidence.require(job,obs,now=a['time_s'],target=stage['target'])
            elif stage['action']=='rotate':
                assert [a['event'] for a in trace]==['ACTION_STARTED','PROBE_ENGAGED','ROTATION_MEASURED','SPINDLE_STOPPED','ACTION_COMPLETE']
                points=evidence.window(job,trace[1]['sim_time_s'],trace[2]['sim_time_s'])
                deferred.append(lambda p=points,a=trace,s=stage: check_rotation(p,a,s)); rotation_count+=1
        assert rotation_count==4 and set(measured)=={'run_in','painting'}
        run=measured['run_in']; trace=[a for a in actions if a.get('command_id')==run['command_id']]
        assert run['evidence']['kind']=='run_in' and trace==run['evidence']['events']
        assert [a['event'] for a in trace]==['RUN_IN_STARTED','RUN_IN_SPEED_DWELL_MEASURED','RUN_IN_STOPPED']
        points=evidence.window(job,trace[0]['time_s'],trace[-1]['time_s'])
        deferred.append(lambda p=points,a=trace: check_run_in(p,a,FULL_STAGES[14]['target']))
        painting=measured['painting']; traverse=painting['evidence']['traverse']
        assert painting['evidence']['kind']=='paint' and len(traverse)==3
        for row,dx in zip(traverse,[-.7,.7,-.7]):
            target=(PAINT[0]+dx,PAINT[1],PAINT[2]+.85)
            assert math.dist(row['target'],target)<1e-9
            evidence.require('spray_tool',row['tool'],target=target)
            evidence.require(job,row['fixture'],target=PAINT)
            assert abs(row['tool']['time_s']-row['fixture']['time_s'])<=.1
        assert traverse[0]['tool']['time_s']<traverse[1]['tool']['time_s']<traverse[2]['tool']['time_s']<=painting['time_s']
        summary=verify_stock(job,components,[e for e in stocks if e['job']==job],press_by_job[job],actions,commands,terminal,evidence)
        assert report['worker_stock'][job]==summary.pop('snapshot')
        if report['case']=='stalled_recovery' and job==meta['jobs'][-1]:
            assert len(faults)==len(stops)==len(retries)==1 and faults[0]['reason']=='insert_timeout'
            assert report['recovered_commands'][job] in {c['id'] for c in commands.values() if c['job']==job}
        else:
            assert not faults and not stops and not retries
        summaries.append(dict(job=job,ordered_operations=18,measured_rotations=rotation_count,
            caps_refitted=2,run_in_dwell_s=trace[1]['stable_s'],**summary))
    shared={'cap_bin':[],'bearing_bin':[]}
    for e in sorted(all_bin_events,key=lambda e:(e['time_s'],e['job'],e['seq'])):
        n=e['part']; bin_id='cap_bin' if '_cap_' in n else 'bearing_bin'
        if e['event']=='PART_BIN_COMMITTED':
            assert n not in shared[bin_id] and len(shared[bin_id])<4; shared[bin_id].append(n)
        else:
            shared[bin_id].remove(n)
    assert not shared['cap_bin'] and len(shared['bearing_bin'])==4
    for snap in report['worker_snapshots'].values():
        assert all(sorted(snap['bins'][b])==sorted(shared[b]) for b in shared)
    for w in waypoints:
        c=commands[w['command_id']]
        assert c['issued_at']<=w['time_s']<=terminal[c['id']]
        evidence.require(w['entity'],w['observation'],now=w['time_s'],target=w['target'])
    return commands,deferred,summaries


def verify_stock(job,items,events,press,actions,commands,terminal,evidence):
    states={n:'unreceived' for n in items}; owner=None; fitted=[]; prepared={}; commits=[]; reservations=[]
    for seq,e in enumerate(events):
        assert e['seq']==seq and (seq==0 or e['time_s']>=events[seq-1]['time_s'])
        kind=e['event']; t=e['time_s']
        if 'command_id' in e:
            c=commands[e['command_id']]; assert c['job']==job and c['station']=='press' and c['kind']=='device'
            assert c['issued_at']<=t<=terminal[c['id']]
        if kind=='STOCK_RECEIVED':
            n=e['component']; assert states[n]=='unreceived'
            evidence.require(n,e['observation'],now=t,target=items[n]['receiving']); states[n]='available'
        elif kind=='STOCK_RESERVED':
            assert owner is None and set(states.values())=={'available'}
            owner=job; states={n:'reserved' for n in states}; reservations.append(e)
        elif kind=='COMPONENT_PREPARED':
            n=e['component']; assert owner==job and states[n]=='reserved'
            evidence.require(n,e['observation'],now=t,target=items[n]['staged'])
            a,b=e['gauge_observations']; r=items[n]['receiving']
            evidence.require(n,a,target=(*r[:2],r[2]+.12)); evidence.require(n,b,target=r)
            assert a['time_s']<b['time_s']<=t and e['identity']=='DEMO-'+n
            assert .095<math.dist(a['xyz'],b['xyz'])<.145
            states[n]='prepared'; prepared[n]=e
        elif kind=='PAIR_FITTED':
            side=e['side']; assert side not in fitted
            names=[n for n,i in items.items() if i['side']==side]; assert set(e['observations'])==set(names)
            for n in names:
                assert states[n]=='prepared'
                evidence.require(n,e['observations'][n],now=t,target=items[n]['fitted']); states[n]='fitted'
            trace=[a for a in actions if a.get('command_id')==e['command_id'] and a.get('bearing_id') in names]
            assert trace==e['press_events'] and trace[-1]['time_s']<=t
            fitted.append(side); commits.append(e)
        else:
            raise AssertionError(kind)
        assert all(sum(states[n] in {'available','reserved','prepared'} for n,i in items.items() if i['kind']==kind)<=2 for kind in ['bearing','ring'])
        assert e['snapshot']==dict(component_state=states,owner=owner,fitted_sides=fitted,capacity={'bearing':2,'ring':2})
    assert len(reservations)==1 and len(prepared)==4 and fitted==['left','right']
    groups={}
    for row in press:
        token=row['command_id']; c=commands[token]
        if c['job']!=job: continue
        side=row['side']; ram='press_ram_'+side
        assert c['station']=='press' and c['issued_at']<=row['time_s']<=terminal[token]
        evidence.require(ram,row['ram'],stop=False); evidence.require(job,row['fixture'],stop=False)
        actual=row['ram']; fixture=row['fixture']
        assert actual['time_s']==row['time_s'] and abs(actual['time_s']-fixture['time_s'])<=.1
        assert (row['ram_m'],row['ram_velocity_m_s'],row['requested_effort_n'])==(actual['probe'],actual['probe_velocity'],actual['applied_effort']['probe'])
        assert row['fixture_error_m']==math.dist(fixture['xyz'],FULL_STAGES[11]['target'])<=.025
        assert row['fixture_speed_m_s']==max(abs(v) for v in fixture['velocity'])<=.035
        assert -.003<=row['ram_m']<=.253 and abs(row['ram_velocity_m_s'])<=.5 and abs(row['requested_effort_n'])<=200
        groups.setdefault((token,side),[]).append(row)
    expected=['PRESS_STARTED','RAM_HOME','RAM_APPROACH','INSERTION_STROKE_MEASURED','STOPPED_HOLD_MEASURED','PRESS_MOTION_COMPLETE']
    completed=[]; failures=[]
    for (token,side),samples in groups.items():
        bearing=f'{job}_new_bearing_{side}'; ring=f'{job}_new_ring_{side}'
        ev=[a for a in actions if a.get('command_id')==token and a.get('bearing_id')==bearing]
        assert ev and all(a['ring_id']==ring and a['job']==job for a in ev)
        assert samples[0]['time_s']>=max(prepared[bearing]['time_s'],prepared[ring]['time_s'])
        for a,b in zip(samples,samples[1:]):
            assert b['time_s']>=a['time_s'] and abs(b['ram_m']-a['ram_m'])<=.5*(b['time_s']-a['time_s'])+.002
        for e in ev:
            raw=next(s for s in samples if s['time_s']==e['time_s'])
            assert (raw['ram_m'],raw['ram_velocity_m_s'])==(e['ram_m'],e['ram_velocity_m_s'])
        if ev[-1]['event']=='PRESS_FAILED':
            assert side=='right' and ev[-1]['reason']=='insert_timeout'
            assert [a['event'] for a in ev]==expected[:3]+['PRESS_FAILED']
            assert ev[-1]['time_s']-ev[2]['time_s']>12
            failures.append((token,ev)); continue
        assert [a['event'] for a in ev]==expected
        assert abs(ev[1]['ram_m'])<.002 and abs(ev[1]['ram_velocity_m_s'])<.008
        assert abs(ev[2]['ram_m']-.04)<.002 and abs(ev[2]['ram_velocity_m_s'])<.008
        assert abs(ev[3]['ram_m']-.22)<.002 and abs(ev[3]['ram_velocity_m_s'])<.008
        assert ev[3]['ram_m']-ev[2]['ram_m']==ev[3]['stroke_m']>=.175
        hold=[s for s in samples if ev[3]['time_s']<=s['time_s']<=ev[4]['time_s']]
        assert len(hold)>2 and ev[4]['time_s']-ev[3]['time_s']==ev[4]['hold_s']>=.25
        assert all(abs(s['ram_m']-.22)<.002 and abs(s['ram_velocity_m_s'])<.008 for s in hold)
        assert abs(ev[-1]['ram_m'])<.002 and abs(ev[-1]['ram_velocity_m_s'])<.008
        assert ev[-1]['stroke_m']==ev[3]['stroke_m']
        assert ev[-1]['peak_requested_effort_n']==max(abs(s['requested_effort_n']) for s in samples)
        assert any(c['side']==side and c['command_id']==token and c['press_events']==ev for c in commits)
        completed.append(dict(side=side,stroke_m=ev[3]['stroke_m'],hold_s=ev[4]['hold_s']))
    assert len(completed)==2
    if failures:
        assert len(failures)==1 and commits[0]['time_s']<failures[0][1][-1]['time_s']<commits[1]['time_s']
        for n,i in items.items():
            if i['side']=='left':
                points=evidence.window(n,commits[0]['time_s'],commands[commits[1]['command_id']]['issued_at'])
                def retained(p=points,target=i['fitted']):
                    assert p and all(math.dist(s['xyz'],target)<.025 for s in p)
                # The caller runs these after streaming raw evidence.
                evidence.retained_checks.append(retained)
    return dict(replacement_components=4,finite_stock_reservations=1,press=completed,
        press_failures=len(failures),snapshot=events[-1]['snapshot'])
