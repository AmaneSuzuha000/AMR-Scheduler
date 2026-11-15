"""Resource ownership survives failed, delayed and superseded actions."""
from tools.wheelset_scene.resource_runtime import Place,ResourceRuntime


def tiny(capacity=1):
    return ResourceRuntime([Place('in',2,(0,0,0)),Place('machine',1,(1,0,0),True),
        Place('out',capacity,(2,0,0))],run_id='test',timeout_s=10)


def arrive(r,c,t):
    p=next(p for p in r.places if p.id==c['target'])
    return r.observe(c['id'],now=t,observed_at=t,pose=p.target,device_complete=True)


def test_source_and_destination_stay_owned_until_matching_arrival():
    r=tiny();r.admit('a',0);r.admit('b',0);c,=r.dispatch(0)
    assert r.stock['in']=={'a','b'} and r.reserved['machine']=={'a'}
    assert not r.observe('unknown',now=1,observed_at=1,pose=(1,0,0))
    assert not r.observe(c['id'],now=1,observed_at=1,pose=(.9,0,0))
    assert not r.dispatch(1)
    assert arrive(r,c,2) and r.stock['in']=={'b'}


def test_full_output_blocks_upstream_without_duplicate_work():
    r=tiny();r.admit('a',0);r.admit('b',0)
    for t in range(7):
        for c in r.dispatch(t):assert arrive(r,c,t)
    assert r.stock['out']=={'a'} and r.stock['machine']=={'b'}
    assert not r.dispatch(8) and r.completed==[]
    assert r.release_outfeed('a',8)
    c,=r.dispatch(8);assert c['job']=='b' and arrive(r,c,9)
    assert r.release_outfeed('b',9) and r.completed==['a','b']


def test_timeout_retains_carrier_and_recovery_needs_fresh_stopping_evidence():
    r=tiny();r.admit('a',0);old,=r.dispatch(0);r.expire(10)
    assert r.carrier==old['id'] and r.stock['in']=={'a'} and r.reserved['machine']=={'a'}
    assert r.recover(old['id'],10) is None
    assert not r.confirm_stopped(old['id'],now=11,observed_at=9,stopped=True)
    assert r.confirm_stopped(old['id'],now=11,observed_at=11,stopped=True)
    new=r.recover(old['id'],12);assert new!=old['id']
    assert not r.observe(old['id'],now=13,observed_at=13,pose=(1,0,0))
    assert r.carrier==new and r.reserved['machine']=={'a'}
    assert r.observe(new,now=13,observed_at=13,pose=(1,0,0))


def test_device_requires_completion_not_just_elapsed_time_or_pose():
    r=tiny();r.admit('a',0);arrive(r,r.dispatch(0)[0],1);c,=r.dispatch(1)
    assert not r.observe(c['id'],now=2,observed_at=2,pose=(1,0,0))
    r.expire(11);assert r.jobs['a']['state']=='failed' and r.stock['machine']=={'a'}
    assert not r.dispatch(12)


def test_parallel_devices_share_one_exclusive_carrier():
    r=ResourceRuntime();r.admit('a',0);r.admit('b',0)
    saw_parallel=False
    for t in range(40):
        batch=r.dispatch(t)
        assert sum(c['kind']=='transfer' for c in batch)<=1
        saw_parallel |= sum(c['kind']=='device' and c['status']=='active' for c in r.commands.values())>=2
        for c in batch:
            if c['kind']=='transfer':arrive(r,c,t)
        for c in list(r.commands.values()):
            if c['kind']=='device' and c['status']=='active' and t-c['issued_at']>=3:arrive(r,c,t)
        for job,state in list(r.jobs.items()):
            if state['index']==len(r.places)-1:r.release_outfeed(job,t)
    assert saw_parallel and sorted(r.completed)==['a','b']
