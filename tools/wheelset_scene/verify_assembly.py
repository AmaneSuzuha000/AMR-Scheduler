"""Offline reconstruction of stock, dedicated ram motion and retained fit."""
from bisect import bisect_left
import math

from .assembly_flow import ASSEMBLY_STAGES,PRESS,components
from .feedback_evidence import corroborate_stopped


def verify_assembly(result,meta,samples,stock,press,actions,events,poses,commands):
    items=components();states={n:'unreceived' for n in items};owner=None;fitted=[]
    assert meta['components']=={n:{k:list(v) if isinstance(v,tuple) else v for k,v in i.items()} for n,i in items.items()}
    by_name={n:[s['observations'][n] for s in samples] for n in meta['definitions']}
    times={n:[s['time_s'] for s in rows] for n,rows in by_name.items()}
    def raw(name,t):
        index=bisect_left(times[name],t);candidates=by_name[name][max(0,index-1):index+2]
        value=min(candidates,key=lambda s:abs(s['time_s']-t));assert abs(value['time_s']-t)<.15
        return value
    def stopped_at(name,observation,target,now):
        assert 0<=now-observation['time_s']<=.15
        assert math.dist(observation['xyz'],target)<.025 and max(abs(v) for v in observation['velocity'])<.035
        actual=raw(name,observation['time_s']);corroborate_stopped(actual,observation)
    wait=[];reservations=[];prepared={};commits=[]
    terminal={}
    for e in events:
        if e['event'] in {'DEVICE_COMPLETE','FAULT_LATCHED'}:terminal[e['command_id']]=e['time_s']
    for seq,e in enumerate(stock):
        assert e['seq']==seq and (seq==0 or e['time_s']>=stock[seq-1]['time_s'])
        kind=e['event'];t=e['time_s']
        if 'command_id' in e:
            token=e['command_id'];c=commands[token]
            assert c['station']=='press' and c['kind']=='device'
            assert c['issued_at']<=t<=terminal[token]
        if kind=='STOCK_RECEIVED':
            n=e['component'];assert states[n]=='unreceived'
            stopped_at(n,e['observation'],items[n]['receiving'],t);states[n]='available'
            assert sum(states[k] in {'available','reserved','prepared'} for k,i in items.items()
                       if i['kind']==items[n]['kind'])<=2
        elif kind=='MATERIAL_WAIT':
            assert owner is None and e['missing']==[n for n,s in states.items() if s!='available']
            assert e['missing']==['ws001_new_ring_right'];wait.append(e)
        elif kind=='STOCK_RESERVED':
            assert owner is None and set(states.values())=={'available'}
            assert commands[e['command_id']]['station']=='press'
            owner='ws001';states={n:'reserved' for n in states};reservations.append(e)
        elif kind=='COMPONENT_PREPARED':
            n=e['component'];assert owner=='ws001' and states[n]=='reserved'
            stopped_at(n,e['observation'],items[n]['staged'],t)
            a,b=e['gauge_observations'];r=items[n]['receiving']
            stopped_at(n,a,(*r[:2],r[2]+.12),a['time_s']);stopped_at(n,b,r,b['time_s'])
            assert a['time_s']<b['time_s']<=t and e['identity']=='DEMO-'+n
            assert .095<math.dist(a['xyz'],b['xyz'])<.145
            states[n]='prepared';prepared[n]=e
        elif kind=='PAIR_FITTED':
            side=e['side'];assert side not in fitted;names=[n for n,i in items.items() if i['side']==side]
            assert set(e['observations'])==set(names)
            for n in names:
                assert states[n]=='prepared';stopped_at(n,e['observations'][n],items[n]['fitted'],t)
                assert math.dist(prepared[n]['observation']['xyz'],e['observations'][n]['xyz'])>=.155
                states[n]='fitted'
            trace=[a for a in actions if a.get('command_id')==e['command_id'] and a.get('bearing_id') in names]
            assert trace==e['press_events'];assert trace[-1]['time_s']<=t
            fitted.append(side);commits.append(e)
        else:raise AssertionError(kind)
        assert e['snapshot']==dict(component_state=states,owner=owner,fitted_sides=fitted,capacity={'bearing':2,'ring':2})
    assert len(wait)==len(reservations)==1 and len(prepared)==4 and fitted==['left','right']
    assert result['stock_snapshot']==stock[-1]['snapshot'] and set(states.values())=={'fitted'}
    delivery=next(e for e in stock if e['event']=='STOCK_RECEIVED' and e['component']=='ws001_new_ring_right')
    assert delivery['time_s']-wait[0]['time_s']>=3 and reservations[0]['time_s']>=delivery['time_s']
    assert all(s['time_s']>=reservations[0]['time_s'] for s in press)
    traces={}
    for row in press:
        token=row['command_id'];side=row['side'];ram='press_ram_'+side
        assert commands[token]['station']=='press' and commands[token]['kind']=='device'
        assert commands[token]['issued_at']<=row['time_s']<=terminal[token]
        assert row['ram']==raw(ram,row['time_s'])
        actual=row['ram'];fixture=row['fixture'];assert fixture==raw('ws001',fixture['time_s'])
        assert abs(actual['time_s']-fixture['time_s'])<=.1
        assert (row['ram_m'],row['ram_velocity_m_s'],row['requested_effort_n'])==(
            actual['probe'],actual['probe_velocity'],actual['applied_effort']['probe'])
        assert row['fixture_error_m']==math.dist(fixture['xyz'],PRESS)<=.025
        assert row['fixture_speed_m_s']==max(abs(v) for v in fixture['velocity'])<=.035
        assert -.003<=row['ram_m']<=.253 and abs(row['ram_velocity_m_s'])<=.5 and abs(row['requested_effort_n'])<=200
        traces.setdefault((token,side),[]).append(row)
    completed=[];failed=[]
    expected=['PRESS_STARTED','RAM_HOME','RAM_APPROACH','INSERTION_STROKE_MEASURED','STOPPED_HOLD_MEASURED','PRESS_MOTION_COMPLETE']
    for (token,side),trace in traces.items():
        for a,b in zip(trace,trace[1:]):
            assert b['time_s']>=a['time_s']
            assert abs(b['ram_m']-a['ram_m'])<=.5*(b['time_s']-a['time_s'])+.002
        n=f'ws001_new_bearing_{side}';ring=f'ws001_new_ring_{side}'
        ev=[a for a in actions if a.get('command_id')==token and a.get('bearing_id')==n]
        assert ev and all(a['ring_id']==ring and a['job']=='ws001' for a in ev)
        assert trace[0]['time_s']>=max(prepared[n]['time_s'],prepared[ring]['time_s'])
        for e in ev:
            s=min(trace,key=lambda r:abs(r['time_s']-e['time_s']))
            assert s['time_s']==e['time_s'] and s['ram_m']==e['ram_m'] and s['ram_velocity_m_s']==e['ram_velocity_m_s']
            assert any(p.get('reason')=='ram_head' and p['entity']=='press_ram_'+side and
                       p['command_id']==token and abs(p['time_s']-e['time_s'])<=.15 for p in poses)
        if ev[-1]['event']=='PRESS_FAILED':
            assert result['case']=='press_recovery' and side=='right' and ev[-1]['reason']=='insert_timeout'
            assert [e['event'] for e in ev]==expected[:3]+['PRESS_FAILED']
            assert ev[-1]['time_s']-ev[2]['time_s']>12;failed.append((token,ev));continue
        assert [e['event'] for e in ev]==expected
        assert abs(ev[1]['ram_m'])<.002 and abs(ev[1]['ram_velocity_m_s'])<.008
        assert abs(ev[2]['ram_m']-.04)<.002 and abs(ev[2]['ram_velocity_m_s'])<.008
        assert abs(ev[3]['ram_m']-.22)<.002 and abs(ev[3]['ram_velocity_m_s'])<.008
        assert ev[3]['ram_m']-ev[2]['ram_m']==ev[3]['stroke_m']>=.175
        hold=[s for s in trace if ev[3]['time_s']<=s['time_s']<=ev[4]['time_s']]
        assert len(hold)>2 and ev[4]['time_s']-ev[3]['time_s']==ev[4]['hold_s']>=.25
        assert all(abs(s['ram_m']-.22)<.002 and abs(s['ram_velocity_m_s'])<.008 for s in hold)
        assert abs(ev[-1]['ram_m'])<.002 and abs(ev[-1]['ram_velocity_m_s'])<.008
        assert ev[-1]['stroke_m']==ev[3]['stroke_m']
        assert ev[-1]['peak_requested_effort_n']==max(abs(s['requested_effort_n']) for s in trace)
        assert any(c['side']==side and c['command_id']==token and c['press_events']==ev for c in commits)
        completed.append(dict(side=side,stroke_m=ev[3]['stroke_m'],hold_s=ev[4]['hold_s']))
    assert len(completed)==2
    arrival=next(e for e in events if e['event']=='ARRIVAL_CONFIRMED' and commands[e['command_id']]['station']=='assembly_outfeed')
    for n,i in items.items():
        actual=raw(n,arrival['observation']['time_s']);target=[a+b for a,b in zip(ASSEMBLY_STAGES[-1]['target'],i['offset'])]
        assert math.dist(actual['xyz'],target)<.035 and max(abs(v) for v in actual['velocity'])<.035
    if result['case']=='normal':assert not failed and not result['recovered']
    elif result['case']=='press_recovery':
        assert len(failed)==1 and result['recovered'] and result['superseded_completion_rejected']
        old,trace=failed[0];fault=[e for e in events if e['event']=='FAULT_LATCHED'];stop=[e for e in events if e['event']=='STOP_CONFIRMED']
        retry=[e for e in events if e['event']=='RECOVERY_REISSUED']
        assert len(fault)==len(stop)==len(retry)==1
        assert fault[0]['command_id']==old and fault[0]['reason']=='insert_timeout'
        assert commits[0]['side']=='left' and commits[0]['time_s']<fault[0]['time_s']<commits[1]['time_s']
        assert stop[0]['time_s']>=fault[0]['time_s']+.5
        assert retry[0]['old_command_id']==old and retry[0]['time_s']>=stop[0]['time_s']
        assert commits[1]['command_id']==retry[0]['command_id']
        assert not any(e['event']=='DEVICE_COMPLETE' and e['command_id']==old for e in events)
        for n,s in stop[0]['observation'].items():
            assert max(abs(v) for v in s['velocity'])<.035
            if n.startswith('press_ram_'):assert abs(s['probe'])<.005 and abs(s['probe_velocity'])<.015
        for n in ['ws001_new_bearing_left','ws001_new_ring_left']:
            retained=[s for s in by_name[n] if commits[0]['time_s']<=s['time_s']<=retry[0]['time_s']]
            assert retained and all(math.dist(s['xyz'],items[n]['fitted'])<.025 for s in retained)
        injected=[s for s in press if s['command_id']==old and s['side']=='right' and s['time_s']>trace[2]['time_s']+.15]
        assert len(injected)>20 and all(s['requested_effort_n']==0. for s in injected)
    return dict(replacement_components_prepared=4,finite_stock_reservations=1,missing_stock_wait_checked=True,
        dedicated_linear_press=completed,measured_press_samples=len(press),partial_press_recovery_checked=bool(failed))
