import pytest
from tools.wheelset_scene.press_action import PressAction,PressSample


def sample(t,p,v=0.,*,token='run:press',error=0.):
    return PressSample(t,token,p,v,15.,error,0.)


def action():return PressAction('run:press','ws001','bearing_new_left','ring_new_left')


def test_press_needs_linear_stroke_stopped_hold_and_retracted_ram_before_completion():
    a=action();a.update(sample(0,0));assert a.state=='approach'
    a.update(sample(1,.04));assert a.state=='insert'
    a.update(sample(2,.22,.02));assert a.state=='insert'
    a.update(sample(2.1,.22));assert a.state=='hold'
    a.update(sample(2.2,.22));assert a.state=='hold'
    a.update(sample(2.4,.22));assert a.state=='retract'
    a.update(sample(3,0,.01));assert a.state=='retract'
    a.update(sample(3.2,0));assert a.state=='complete'
    assert a.stroke_m==pytest.approx(.18)
    assert a.events[-1]['event']=='PRESS_MOTION_COMPLETE'
    assert a.events[-1]['bearing_id']=='bearing_new_left'


def test_stalled_press_does_not_complete_from_time_and_old_token_cannot_advance_it():
    a=action();a.update(sample(0,0));a.update(sample(1,.04))
    assert not a.update(sample(20,.22,token='obsolete:press'))
    assert not a.update(sample(.5,.22))
    assert a.state=='insert'
    a.update(sample(14,.04));assert a.state=='failed' and a.fault=='insert_timeout'
    assert not any(e['event']=='PRESS_MOTION_COMPLETE' for e in a.events)


def test_fixture_motion_or_broken_ram_telemetry_fails_before_inventory_can_be_committed():
    a=action();a.update(sample(0,0));a.update(sample(1,.04))
    a.update(sample(2,.12,error=.03));assert a.state=='failed' and a.fault=='fixture_not_stopped_at_press'
    assert a.target()==.12
    b=action();b.update(sample(0,0));b.update(sample(.01,.22))
    assert b.state=='failed' and b.fault=='ram_feedback_jump'
    with pytest.raises(ValueError):PressAction('r','ws','same-part','same-part')
