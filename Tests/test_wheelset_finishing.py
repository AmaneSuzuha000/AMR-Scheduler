import math
import pytest

from tools.wheelset_scene.finishing_action import RunInAction,RunInSample
from tools.wheelset_scene.finishing_flow import FiniteStorage,FinishingRuntime,FULL_STAGES,STORAGE,STORAGE_UNLOAD


def obs(t,xyz,v=0.):return dict(time_s=t,xyz=xyz,velocity=[v,0.,0.],omega=0.)


def test_full_storage_keeps_occupancy_until_correct_measured_unload_and_commit():
    ledger=FiniteStorage('r')
    assert not ledger.reserve('ws001','gate',0.)
    release=ledger.begin_release(1.);assert release
    assert not ledger.released('old-release',2.,obs(2.,STORAGE_UNLOAD))
    assert not ledger.released(release,2.,obs(2.,STORAGE))
    assert not ledger.released(release,2.,obs(2.,STORAGE_UNLOAD,.1))
    assert not ledger.reserve('ws001','gate',2.)
    assert ledger.released(release,3.,obs(3.,STORAGE_UNLOAD))
    assert not ledger.released(release,3.,obs(3.,STORAGE_UNLOAD))
    assert ledger.reserve('ws001','gate',4.)
    assert not ledger.committed('ws002','gate',5.,obs(5.,STORAGE))
    assert not ledger.committed('ws001','old-token',5.,obs(5.,STORAGE))
    assert not ledger.committed('ws001','gate',5.,obs(4.,STORAGE))
    assert ledger.snapshot()['reserved']==dict(job='ws001',command_id='gate')
    assert ledger.committed('ws001','gate',6.,obs(6.,STORAGE))
    assert ledger.snapshot()['occupant']=='ws001' and ledger.snapshot()['reserved'] is None
    assert not ledger.reserve('ws002','other',7.)


def test_cap_identity_leaves_bin_only_after_pickup_and_rejoins_same_wheelset():
    r=FinishingRuntime('r');r.completed=[s['id'] for s in FULL_STAGES[:13]];r.station='closing'
    n='ws001_cap_left';item=r.items[n];r.part_state[n]='binned';r.bins['cap_bin']=[n]
    token=r.issue(13,'device',0.)
    assert not r.retrieve_cap(token,n,1.,obs(1.,FULL_STAGES[13]['target']))
    assert r.retrieve_cap(token,n,2.,obs(2.,item['destination']))
    assert not r.retrieve_cap(token,n,2.,obs(2.,item['destination']))
    target=[a+b for a,b in zip(FULL_STAGES[13]['target'],item['offset'])]
    assert not r.refit_cap(token,n,3.,obs(3.,target,.1))
    assert r.refit_cap(token,n,4.,obs(4.,target))
    assert r.bins['cap_bin']==[] and r.part_state[n]=='refitted'
    assert not r.complete(token,4.,obs(4.,FULL_STAGES[13]['target']))


def sample(t,angle,omega,token='r:run_in',error=0.):
    return RunInSample(t,token,angle,omega,error,0.,0.,0.)


def test_run_in_needs_continuous_speed_dwell_two_turns_and_sustained_stop():
    a=RunInAction('r:run_in');a.update(sample(0.,0.,0.))
    assert not a.update(sample(30.,99.,0.,token='old'))
    for i in range(1,81):a.update(sample(i/10,i*.16,1.6))
    assert a.state=='running'
    a.update(sample(8.1,12.96,1.6));assert a.state=='braking' and a.rotation>=4*math.pi
    a.update(sample(8.2,13.04,0.));assert a.state=='braking'
    a.update(sample(8.3,13.04,0.));assert a.state=='braking'
    a.update(sample(8.5,13.04,0.));assert a.state=='complete'
    assert [e['event'] for e in a.events]==['RUN_IN_STARTED','RUN_IN_SPEED_DWELL_MEASURED','RUN_IN_STOPPED']


def test_run_in_stalled_rotation_or_fixture_departure_does_not_pass_from_elapsed_time():
    a=RunInAction('r:run_in')
    for i in range(251):a.update(sample(i/10,0.,0.))
    assert a.state=='running'
    a.update(sample(25.1,0.,0.));assert a.state=='failed' and a.fault=='run_in_timeout'
    b=RunInAction('r:run_in');b.update(sample(0.,0.,0.,error=.03))
    assert b.state=='failed' and b.fault=='run_in_fixture_or_tool_departure'


def test_finishing_device_cannot_be_completed_with_a_bare_success_flag():
    r=FinishingRuntime('r');r.completed=[s['id'] for s in FULL_STAGES[:14]];r.station='run_in'
    token=r.issue(14,'device',0.)
    assert not r.finish_measured(token,1.,dict(kind='run_in',events=[]))
    assert not r.complete(token,1.,obs(1.,FULL_STAGES[14]['target']))
    events=[dict(event=k,command_id=token,time_s=t) for k,t in [
        ('RUN_IN_STARTED',0.),('RUN_IN_SPEED_DWELL_MEASURED',9.),('RUN_IN_STOPPED',10.)]]
    events[1].update(stable_s=1.,rotation_rad=4*math.pi);events[2]['stopped_s']=.3
    assert not r.finish_measured(token,10.,dict(kind='run_in',events=events))
    assert 'run_in' not in r.finish_evidence


def test_storage_wait_station_keeps_workpiece_until_slot_reserved():
    r=FinishingRuntime('r');r.completed=[s['id'] for s in FULL_STAGES[:16]];r.station='storage_wait'
    token=r.issue(16,'device',0.)
    assert not r.complete(token,1.,obs(1.,FULL_STAGES[16]['target']))
    assert not r.storage.reserve('ws001',token,1.)
    assert r.current==token and len(r.completed)==16
