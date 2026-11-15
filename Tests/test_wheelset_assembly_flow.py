import pytest

from tools.wheelset_scene.assembly_flow import AssemblyRuntime,ASSEMBLY_STAGES
from tools.wheelset_scene.front_flow import transfer_points
from tools.wheelset_scene.press_action import PressAction,PressSample


def obs(t,xyz):return dict(time_s=t,xyz=xyz,velocity=[0.,0.,0.],omega=0.,probe=0.,probe_velocity=0.)


def at_press():
    r=AssemblyRuntime('test');r.completed=[s['id'] for s in ASSEMBLY_STAGES[:11]]
    r.station='press';token=r.issue(11,'device',0.)
    return r,token


def test_missing_stock_blocks_reservation_and_unmeasured_arrival_cannot_release_it():
    r,t=at_press()
    for n,i in r.components.items():
        if n=='ws001_new_ring_right':continue
        assert r.receive(n,1.,obs(1.,i['receiving']))
    assert not r.reserve_stock(t,1.)
    n='ws001_new_ring_right';i=r.components[n]
    assert not r.receive(n,2.,obs(2.,i['origin']))
    assert not r.receive(n,3.,obs(2.,i['receiving']))
    assert not r.complete(t,3.,obs(3.,ASSEMBLY_STAGES[11]['target']))
    assert r.receive(n,4.,obs(4.,i['receiving']))
    assert not r.receive(n,4.,obs(4.,i['receiving']))
    assert r.reserve_stock(t,4.)
    assert set(r.component_state.values())=={'reserved'}


def press_events(token,bearing,ring,start=8.):
    a=PressAction(token,'ws001',bearing,ring)
    for dt,p in [(0,0),(1,.04),(2,.22),(2.3,.22),(3.3,0)]:
        a.update(PressSample(start+dt,token,p,0.,0.,0.,0.))
    assert a.state=='complete'
    return a.events


def test_partial_fit_survives_stopped_reissue_without_double_stock_or_old_token_commit():
    r,t=at_press()
    for n,i in r.components.items():assert r.receive(n,1.,obs(1.,i['receiving']))
    assert r.reserve_stock(t,1.)
    for n,i in r.components.items():
        a=obs(2.,(*i['receiving'][:2],i['receiving'][2]+.12));b=obs(3.,i['receiving'])
        assert r.prepared(t,n,4.,obs(4.,i['staged']),[a,b])
    names=[n for n,i in r.components.items() if i['side']=='left']
    o={n:obs(12.,r.components[n]['fitted']) for n in names}
    events=press_events(t,*names)
    assert r.fitted(t,'left',12.,o,events)
    assert not r.fitted(t,'left',12.,o,events)
    assert r.fail(t,13.,'insert_timeout')
    with pytest.raises(ValueError):r.issue(11,'device',13.,replacement=t)
    assert r.stopped(t,14.,{})
    new=r.issue(11,'device',14.,replacement=t)
    assert r.reserve_stock(new,14.) and r.fitted_sides==['left']
    assert not r.fitted(t,'right',15.,{},events)
    assert not r.complete(t,15.,obs(15.,ASSEMBLY_STAGES[11]['target']))
    assert len([e for e in r.stock_events if e['event']=='STOCK_RESERVED'])==1


def test_cross_zone_waypoints_split_long_travel_without_changing_endpoints():
    current=ASSEMBLY_STAGES[10]['target'];stage=ASSEMBLY_STAGES[11]
    points=transfer_points(current,stage)
    assert points[-1]==stage['target']
    assert all(max(abs(x-y) for x,y in zip(a,b))<=30 for a,b in zip([current,*points],points))
