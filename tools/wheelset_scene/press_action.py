"""Dedicated measured linear press protocol; parameters are demo assumptions.

This module does not infer interference fit or contact force from actuator
requests. A future adapter must bind the bearing/ring inventory and actual ram.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PressSample:
    time_s: float
    command_id: str
    ram_m: float
    ram_velocity_m_s: float
    requested_effort_n: float
    fixture_error_m: float
    fixture_speed_m_s: float


class PressAction:
    """Home, approach, measured insertion stroke, stopped hold, then retract.

    Inventory ownership is supplied by the resource controller. Completion is
    only a motion protocol result, never a manufacturer press-quality result.
    """
    def __init__(self,token,job,bearing_id,ring_id):
        if not all(isinstance(v,str) and v for v in [token,job,bearing_id,ring_id]):
            raise ValueError('Press requires command/workpiece/component identity')
        if bearing_id==ring_id:raise ValueError('Bearing and back ring must be distinct items')
        self.token=token;self.job=job;self.bearing=bearing_id;self.ring=ring_id
        self.state='home';self.entered=None;self.last=None;self.events=[];self.fault=None
        self.approach_m=None;self.stroke_m=0.;self.peak_requested_effort_n=0.;self.hold_since=None

    def event(self,kind,s,**data):
        self.events.append(dict(command_id=self.token,job=self.job,bearing_id=self.bearing,ring_id=self.ring,
            event=kind,time_s=s.time_s,phase=self.state,ram_m=s.ram_m,ram_velocity_m_s=s.ram_velocity_m_s,**data))

    def fail(self,reason,s):
        self.fault=reason;self.event('PRESS_FAILED',s,reason=reason);self.state='failed'

    def target(self):
        if self.state=='failed':return self.last.ram_m if self.last else 0.
        return {'home':0.,'approach':.04,'insert':.22,'hold':.22,'retract':0.,'complete':0.}[self.state]

    def update(self,s):
        if s.command_id!=self.token:return False
        if not all(math.isfinite(v) for k,v in s.__dict__.items() if k!='command_id'):
            raise ValueError('Nonfinite press observation')
        if self.last and s.time_s<=self.last.time_s:return False
        previous=self.last;self.last=s
        if self.state in {'complete','failed'}:return False
        if self.entered is None:self.entered=s.time_s;self.event('PRESS_STARTED',s)
        if s.fixture_error_m<0 or s.fixture_speed_m_s<0:raise ValueError('Negative fixture error/speed magnitude')
        if s.fixture_error_m>.025 or s.fixture_speed_m_s>.035:
            self.fail('fixture_not_stopped_at_press',s);return True
        if not -.003<=s.ram_m<=.253 or abs(s.ram_velocity_m_s)>.5 or abs(s.requested_effort_n)>200:
            self.fail('ram_motion_or_effort_bound',s);return True
        if previous and abs(s.ram_m-previous.ram_m)>.5*(s.time_s-previous.time_s)+.002:
            self.fail('ram_feedback_jump',s);return True
        if s.time_s-self.entered>{'home':8,'approach':8,'insert':12,'hold':3,'retract':10}[self.state]:
            self.fail(self.state+'_timeout',s);return True
        self.peak_requested_effort_n=max(self.peak_requested_effort_n,abs(s.requested_effort_n))
        stopped=abs(s.ram_velocity_m_s)<.008;at=abs(s.ram_m-self.target())<.002
        next_state=None
        if self.state=='home' and at and stopped:
            self.event('RAM_HOME',s);next_state='approach'
        elif self.state=='approach' and at and stopped:
            self.approach_m=s.ram_m;self.event('RAM_APPROACH',s);next_state='insert'
        elif self.state=='insert' and at and stopped:
            self.stroke_m=s.ram_m-self.approach_m
            if self.stroke_m<.175:self.fail('insertion_stroke_short',s);return True
            self.event('INSERTION_STROKE_MEASURED',s,stroke_m=self.stroke_m);next_state='hold';self.hold_since=s.time_s
        elif self.state=='hold':
            if not at or not stopped:
                self.fail('ram_left_stopped_hold',s);return True
            if s.time_s-self.hold_since>=.25:
                self.event('STOPPED_HOLD_MEASURED',s,hold_s=s.time_s-self.hold_since);next_state='retract'
        elif self.state=='retract' and at and stopped:
            self.event('PRESS_MOTION_COMPLETE',s,stroke_m=self.stroke_m,
                peak_requested_effort_n=self.peak_requested_effort_n);next_state='complete'
        if next_state:self.state=next_state;self.entered=s.time_s
        return True
