"""One identified inspection action, completed by measured joints only."""
import math


class DeviceAction:
    def __init__(self,token,job,station,target_slide):
        self.token=token;self.job=job;self.station=station;self.target=target_slide
        self.state='engage';self.entered=None;self.last=None;self.events=[];self.fault=None
        self.rotation=0.;self.previous_angle=None

    def event(self,kind,s,**data):
        self.events.append(dict(command_id=self.token,job=self.job,station=self.station,
            event=kind,sim_time_s=s.time_s,phase=self.state,**data))

    def fail(self,reason,s):
        self.fault=reason;self.event('ACTION_FAILED',s,reason=reason);self.state='failed'

    def update(self,s):
        if self.last and s.time_s<=self.last.time_s:return
        if not all(math.isfinite(v) for v in s.__dict__.values()):raise ValueError('nonfinite joint state')
        self.last=s
        if self.state in {'complete','failed'}:return
        if self.entered is None:self.entered=s.time_s;self.event('ACTION_STARTED',s)
        at=abs(s.slide-self.target)<=.025 and abs(s.slide_velocity)<=.035
        if not at:self.fail('fixture_left_work_position',s);return
        if s.time_s-self.entered>{'engage':8,'rotate':12,'brake':8,'retract':8}[self.state]:
            self.fail(self.state+'_timeout',s);return
        home=abs(s.probe)<=.005 and abs(s.probe_velocity)<=.015
        stopped=abs(s.angular_velocity)<=.025
        next_phase=None
        if self.state=='engage' and abs(s.probe-.22)<=.005 and abs(s.probe_velocity)<=.015 and stopped:
            self.event('PROBE_ENGAGED',s,probe=s.probe);self.previous_angle=s.angle;next_phase='rotate'
        elif self.state=='rotate':
            if abs(s.probe-.22)>.015:self.fail('probe_left_work_position',s);return
            self.rotation+=math.atan2(math.sin(s.angle-self.previous_angle),math.cos(s.angle-self.previous_angle))
            self.previous_angle=s.angle
            if self.rotation>=2*math.pi:
                self.event('ROTATION_MEASURED',s,angle_rad=self.rotation);next_phase='brake'
        elif self.state=='brake' and stopped:
            self.event('SPINDLE_STOPPED',s,angular_velocity=s.angular_velocity);next_phase='retract'
        elif self.state=='retract' and home and stopped:
            self.event('ACTION_COMPLETE',s,probe=s.probe,angular_velocity=s.angular_velocity);next_phase='complete'
        if next_phase:self.state=next_phase;self.entered=s.time_s

    def targets(self):
        return (self.target,.22 if self.state in {'engage','rotate','brake'} else 0.,
            1.6 if self.state=='rotate' else 0.)


def stopped_and_home(s):
    return (abs(s.slide_velocity)<.035 and abs(s.angular_velocity)<.025
        and abs(s.probe)<.005 and abs(s.probe_velocity)<.015)
