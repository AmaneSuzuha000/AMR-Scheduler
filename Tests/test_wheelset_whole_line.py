"""Synthetic protocol tests, not recorded Gazebo execution evidence."""
import pytest

from tools.wheelset_scene.finishing_flow import FULL_STAGES,STORAGE,STORAGE_UNLOAD,STORED_JOB
from tools.wheelset_scene.front_flow import ORIGIN
from tools.wheelset_scene.material_supply import MaterialSupply
from tools.wheelset_scene.whole_line_runtime import WholeLineRuntime
from tools.wheelset_scene.whole_line_actions import WholeLineAction


def obs(entity,t,xyz=ORIGIN,v=0.):
    return dict(entity=entity,time_s=t,xyz=list(xyz),velocity=[v,0.,0.],omega=0.,probe=0.,probe_velocity=0.)


def at_station(r,job,station,t=0.):
    """Seed a resource/worker state for isolated downstream gate tests."""
    assert r.admit(job,t,observation=obs(job,t))
    index=next(i for i,s in enumerate(FULL_STAGES) if s['id']==station)
    r.stock['infeed'].remove(job);r.stock[station].add(job)
    r.jobs[job].update(index=index+1,device_done=False)
    w=r.workers[job];w.station=station;w.completed=[s['id'] for s in FULL_STAGES[:index]]
    return w


def test_one_carrier_retains_source_until_fresh_identified_stopped_arrival():
    r=WholeLineRuntime('r')
    assert not r.admit('a',0.,observation=obs('b',0.))
    assert r.admit('a',0.,observation=obs('a',0.))
    assert not r.admit('b',0.,observation=obs('b',0.))
    c,=r.dispatch(0.);target=FULL_STAGES[0]['target']
    assert not r.complete(c['id'],1.,obs('b',1.,target))
    assert not r.complete(c['id'],1.,obs('a',.8,target))
    assert not r.complete(c['id'],1.,obs('a',1.,target,.1))
    assert r.stock['infeed']=={'a'} and r.reserved['identification']=={'a'} and r.carrier==c['id']
    with pytest.raises(ValueError):r.observe(c['id'],now=1.,observed_at=1.,device_complete=True)
    assert r.complete(c['id'],1.,obs('a',1.,target))
    assert not r.complete(c['id'],1.,obs('a',1.,target))
    assert r.admit('b',1.,observation=obs('b',1.))
    issued=r.dispatch(1.)
    # a owns identification, so b cannot move there even with a free carrier.
    assert len(issued)==1 and issued[0]['kind']=='device' and issued[0]['job']=='a'
    r.check()


def test_part_bins_are_shared_and_capacity_wait_does_not_start_or_release_station():
    r=WholeLineRuntime('r',bin_capacity=2)
    a=at_station(r,'a','closing');b=at_station(r,'b','opening')
    a.bins['cap_bin'].extend(['a_cap_left','a_cap_right'])
    for name,item in a.items.items():
        if item['kind']=='cap':a.part_state[name]='binned'
    issued=r.dispatch(0.)
    assert len(issued)==1 and issued[0]['job']=='a'
    assert b.current is None and r.stock['opening']=={'b'}
    assert any(e['event']=='PART_BIN_WAIT' and e['job']=='b' for e in r.events)
    token=issued[0]['execution_id']
    for name,item in a.items.items():
        if item['kind']=='cap':assert a.retrieve_cap(token,name,1.,obs(name,1.,item['destination']))
    issued=r.dispatch(1.)
    assert len(issued)==1 and issued[0]['job']=='b' and b.bins is a.bins
    assert r.bin_holds['cap_bin']=={'b_cap_left','b_cap_right'}


def test_fault_retains_carrier_reservation_and_partial_work_until_all_axes_stop():
    r=WholeLineRuntime('r');assert r.admit('a',0.,observation=obs('a',0.))
    c,=r.dispatch(0.);old=c['id'];local=c['execution_id'];target=FULL_STAGES[0]['target']
    assert r.fail(old,1.,'injected')
    assert r.recover(old,1.) is None
    assert r.carrier==old and r.stock['infeed']=={'a'} and r.reserved['identification']=={'a'}
    w=r.workers['a'];names={'a',*w.items,*w.components}
    samples={n:obs(n,2.) for n in names}
    samples['a_new_ring_right']['velocity'][0]=.1
    assert not r.confirm_stopped(old,now=2.,observations=samples)
    samples['a_new_ring_right']=obs('a_new_ring_right',.9)
    assert not r.confirm_stopped(old,now=2.,observations=samples)
    samples['a_new_ring_right']=obs('a_new_ring_right',2.)
    assert r.confirm_stopped(old,now=2.,observations=samples)
    new=r.recover(old,2.);assert new and new!=old
    assert w.commands[local]['status']=='superseded' and r.carrier==new
    assert r.reserved['identification']=={'a'}
    assert not r.complete(old,3.,obs('a',3.,target))
    assert r.complete(new,3.,obs('a',3.,target))
    r.check()


def test_consumables_wait_and_retry_use_existing_shared_supply_exactly_once():
    supply=MaterialSupply('r',initial={'derust':1,'magnetic':0})
    r=WholeLineRuntime('r',supply=supply)
    a=at_station(r,'a','derust');at_station(r,'b','magnetic')
    c,=r.dispatch(0.);assert c['job']=='a'
    assert supply.stock=={'derust':0,'magnetic':0} and ('b','magnetic') in supply.waiting
    assert r.fail(c['id'],1.,'injected')
    samples={n:obs(n,2.) for n in {'a',*a.items,*a.components}}
    assert r.confirm_stopped(c['id'],now=2.,observations=samples)
    assert r.recover(c['id'],2.)
    assert len([e for e in supply.events if e['event']=='MATERIAL_CONSUMED'])==1
    assert supply.stock['derust']==0 and r.stock['magnetic']=={'b'}


def test_two_workpieces_share_one_storage_and_old_release_cannot_empty_next_occupant():
    r=WholeLineRuntime('r');w=at_station(r,'a','storage_wait')
    c,=r.dispatch(0.);token=c['execution_id']
    assert w.storage is r.storage and not r.storage.reserve('a',token,0.)
    assert not r.complete(c['id'],0.,obs('a',0.,FULL_STAGES[16]['target']))
    release=r.begin_unload(1.);assert release
    assert not r.complete_unload(release,2.,obs('a',2.,STORAGE_UNLOAD))
    assert not r.complete_unload(release,2.,obs(STORED_JOB,2.,STORAGE_UNLOAD,.1))
    assert r.complete_unload(release,2.,obs(STORED_JOB,2.,STORAGE_UNLOAD))
    assert r.storage.reserve('a',token,2.)
    assert r.complete(c['id'],2.,obs('a',2.,FULL_STAGES[16]['target']))
    move,=r.dispatch(2.);assert move['target']=='storage'
    assert r.complete(move['id'],3.,obs('a',3.,STORAGE))
    store,=r.dispatch(3.)
    assert r.storage.committed('a',token,4.,obs('a',4.,STORAGE))
    assert r.complete(store['id'],4.,obs('a',4.,STORAGE))
    assert r.completed==[] and r.stock['storage']=={'a'}
    next_release=r.begin_unload(5.);assert next_release and next_release!=release
    assert not r.complete_unload(release,6.,obs('a',6.,STORAGE_UNLOAD))
    assert r.storage.occupant=='a' and r.stock['storage']=={'a'}
    assert r.complete_unload(next_release,6.,obs('a',6.,STORAGE_UNLOAD))
    assert r.completed==['a'] and r.stock['storage']==set()
    b=at_station(r,'b','storage_wait',6.)
    gate,=r.dispatch(6.)
    # Another worker's reservation cannot complete b's storage gate.
    assert r.storage.reserve('a','old-gate',7.)
    assert not r.complete(gate['id'],7.,obs('b',7.,FULL_STAGES[16]['target']))
    assert b.current==gate['execution_id']


def test_expiry_fails_both_protocols_and_keeps_resource_locks():
    r=WholeLineRuntime('r',timeout_s=5.)
    assert r.admit('a',0.,observation=obs('a',0.))
    c,=r.dispatch(0.);r.expire(5.)
    assert r.commands[c['id']]['status']=='failed'
    assert r.workers['a'].commands[c['execution_id']]['status']=='failed'
    assert r.carrier==c['id'] and r.stock['infeed']=={'a'}
    assert not r.complete(c['id'],6.,obs('a',6.,FULL_STAGES[0]['target']))
    r.check()


def test_cooperative_scan_waits_for_extension_retraction_and_binds_each_identity():
    r=WholeLineRuntime('r');a=at_station(r,'a','identification');b=at_station(r,'b','axle_end')
    commands=r.dispatch(0.);targets={n:dict(xyz=list(w.recipe[w.stage]['target']),probe=0.,spin=0.)
        for n,w in [('a',a),('b',b)]}
    actions=[WholeLineAction(c,r.workers[c['job']],targets) for c in commands]
    samples={n:dict(obs(n,1.,targets[n]['xyz']),spin=0.) for n in targets}
    for action in actions:action.advance(samples,1.)
    assert all(not action.complete for action in actions)
    assert targets['a']['probe']==targets['b']['probe']==.22 and a.tag is None
    samples['a'].update(time_s=2.,probe=.22)
    for action in actions:action.advance(samples,2.)
    assert a.tag=='DEMO-a' and targets['a']['probe']==0. and targets['b']['probe']==.22
    assert b.tag is None and all(not action.complete for action in actions)
    samples['a'].update(time_s=3.,probe=0.)
    for action in actions:action.advance(samples,3.)
    done=next(x for x in actions if x.job=='a');waiting=next(x for x in actions if x.job=='b')
    assert done.complete and not waiting.complete
    assert r.complete(done.command['id'],3.,samples['a'])
    waiting.advance(samples,10.)
    assert waiting.fault=='scanner_timeout' and not waiting.complete
