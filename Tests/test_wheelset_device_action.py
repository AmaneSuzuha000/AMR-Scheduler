import math
from tools.wheelset_scene.actuated_cell import JointSample
from tools.wheelset_scene.device_action import DeviceAction


def s(t,*,x=11,probe=.22,angle=0,omega=0):
    return JointSample(t,x,0,probe,0,angle,omega)


def test_device_action_requires_full_motion_braking_and_home():
    a=DeviceAction('a:1','ws1','derust',11);a.update(s(0));assert a.state=='rotate'
    for n in range(1,66):a.update(s(n*.1,angle=n*.1,omega=1))
    assert a.state=='brake' and a.rotation>=2*math.pi
    a.update(s(7,angle=6.5,omega=.1));assert a.state=='brake'
    a.update(s(8,angle=6.5));assert a.state=='retract'
    a.update(s(9,angle=6.5,probe=.1));assert a.state=='retract'
    a.update(s(10,angle=6.5,probe=0));assert a.state=='complete'
    assert [e['event'] for e in a.events]==['ACTION_STARTED','PROBE_ENGAGED','ROTATION_MEASURED','SPINDLE_STOPPED','ACTION_COMPLETE']


def test_stalled_action_does_not_complete_and_retry_has_new_rotation():
    old=DeviceAction('r:1','ws2','magnetic',25.45)
    old.update(s(0,x=25.45));old.update(s(13,x=25.45))
    assert old.state=='failed' and old.fault=='rotate_timeout'
    new=DeviceAction('r:2','ws2','magnetic',25.45);new.update(s(14,x=25.45,angle=7))
    assert new.state=='rotate' and new.rotation==0 and new.token!=old.token


def test_work_position_and_source_time_gate_completion():
    a=DeviceAction('r:1','ws1','derust',11);a.update(s(0))
    a.update(s(0,angle=2));assert a.rotation==0
    a.update(s(1,x=12,angle=3));assert a.state=='failed'
    a.update(s(2,probe=0,angle=9));assert a.state=='failed'
