"""Command issuance is insufficient: measured motion and interlocks must pass."""
import math
from tools.wheelset_scene.actuated_cell import JointSample,InspectionCycle,efforts


def sample(t,x=11,probe=0,angle=0,omega=0,vx=0):
    return JointSample(t,x,vx,probe,0,angle,omega)


def rotating():
    cycle=InspectionCycle();cycle.update(sample(0));cycle.update(sample(.1,probe=.22))
    assert cycle.state=='rotate'
    return cycle


def test_station_waits_for_stopped_carriage_and_engaged_probe():
    c=InspectionCycle();c.update(sample(0,x=10));assert c.state=='transfer'
    c.update(sample(1,vx=.2));assert c.state=='transfer'
    c.update(sample(2));assert c.state=='engage'
    c.update(sample(3,probe=.1));assert c.state=='engage'
    c.update(sample(4,probe=.22));assert c.state=='rotate'


def test_stalled_spindle_times_out_instead_of_completing_by_time():
    c=rotating()
    for t in range(1,15):c.update(sample(t,probe=.22))
    assert c.state=='failed' and c.fault=='rotate_timeout' and c.completed==[]
    assert not any(e['event']=='DEVICE_COMPLETE' for e in c.events)


def test_motion_must_brake_and_retract_before_releasing_station():
    c=rotating()
    for k in range(1,70):
        c.update(sample(.1+k*.1,probe=.22,angle=k*.1,omega=1))
        if c.state=='brake':break
    assert c.state=='brake' and c.completed==[]
    c.update(sample(8,probe=.22,angle=7,omega=.2));assert c.state=='brake'
    c.update(sample(9,probe=.22,angle=7));assert c.state=='retract'
    c.update(sample(10,probe=.1,angle=7));assert c.state=='retract'
    c.update(sample(11,angle=7));assert c.completed==['derust'] and c.state=='transfer'


def test_duplicate_source_time_cannot_accumulate_rotation():
    c=rotating();assert not c.update(sample(.1,probe=.22,angle=math.pi))
    assert c.rotation==0


def test_retracted_probe_during_rotation_latches_fault():
    c=rotating();c.update(sample(1,probe=0,angle=.5,omega=1))
    assert c.state=='failed' and c.fault=='fixture_left_work_position'
    c.update(sample(2,probe=.22,angle=1));assert c.state=='failed'


def test_effort_commands_obey_the_actuator_limits():
    f=efforts(sample(0,x=-100,probe=-10,omega=-100),[100,1,100])
    assert abs(f['wsd_slide'])<=7000 and abs(f['wsd_probe'])<=200 and abs(f['wsd_spin'])<=1000


def test_slide_servo_settles_with_one_sample_of_feedback_delay():
    x=v=0.;pending=0.;dt=.025
    for n in range(600):
        if n%2==0:
            force=efforts(sample(n*dt,x=x,vx=v),[11,0,0])['wsd_slide']
        v+=(pending-50*v)/1020*dt;x+=v*dt
        if n%2==1:pending=force
    assert abs(x-11)<.025 and abs(v)<.035
