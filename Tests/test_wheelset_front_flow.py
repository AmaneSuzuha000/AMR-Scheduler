import pytest
from tools.wheelset_scene.front_flow import FrontRuntime,STAGES,Waypoints


def reach(r,index):
    token=r.issue(index,'transfer',r.last+1 if r.events else 0)
    assert r.complete(token,r.last+1,{})
    return r.issue(index,'device',r.last+1)


def test_identification_requires_tag_and_does_not_allow_duplicate_or_reordered_operation():
    r=FrontRuntime('r');token=reach(r,0)
    assert not r.complete(token,3,{})
    assert not r.tag_read(token,'wrong-workpiece',3)
    assert r.tag_read(token,'DEMO-ws001',3)
    assert r.complete(token,4,{})
    with pytest.raises(ValueError):r.issue(0,'device',5)
    with pytest.raises(ValueError):r.issue(2,'transfer',5)


def test_partial_dismantling_retry_retains_bin_identity_and_station_ownership():
    r=FrontRuntime('r');r.completed=['identification','exterior'];r.station='opening'
    old=r.issue(2,'device',0)
    assert r.separate(old,'ws001_cap_left',1,{})
    assert r.deposit(old,'ws001_cap_left',2,{})
    assert not r.complete(old,3,{})
    assert r.fail(old,3,'stalled')
    with pytest.raises(ValueError):r.issue(2,'device',4,replacement=old)
    assert r.stopped(old,4,{})
    new=r.issue(2,'device',5,replacement=old)
    assert not r.complete(old,6,{}) and not r.deposit(new,'ws001_cap_left',6,{})
    assert r.bins['cap_bin']==['ws001_cap_left'] and r.station=='opening'
    assert r.separate(new,'ws001_cap_right',6,{}) and r.deposit(new,'ws001_cap_right',7,{})
    assert r.complete(new,8,{})
    assert r.bins['bearing_bin']==[] and len(r.bins['cap_bin'])==2


def test_waypoint_requires_measured_stop_and_times_out_without_motion():
    w=Waypoints('r:1',[(1,2,3)],0,timeout=2)
    assert not w.update((1,2,3),(0,.1,0),1)
    assert not w.update((1.1,2,3),(0,0,0),1)
    with pytest.raises(TimeoutError):w.update((0,0,0),(0,0,0),3)
    assert not w.complete
