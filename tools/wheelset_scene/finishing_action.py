"""Measured spindle run-in protocol; no bearing-quality inference."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RunInSample:
    time_s:float
    command_id:str
    angle:float
    omega:float
    fixture_error_m:float
    fixture_speed_m_s:float
    probe_m:float
    probe_velocity_m_s:float


class RunInAction:
    def __init__(self,token):
        self.token=token;self.state='running';self.last=None;self.started=None
        self.stable_since=None;self.stopped_since=None;self.rotation=0.;self.events=[];self.fault=None

    def target(self):return 1.6 if self.state=='running' else 0.

    def event(self,kind,s,**data):
        self.events.append(dict(event=kind,command_id=self.token,time_s=s.time_s,
            angle=s.angle,omega=s.omega,rotation_rad=self.rotation,**data))

    def update(self,s):
        if s.command_id!=self.token or (self.last and s.time_s<=self.last.time_s):return False
        if not all(math.isfinite(v) for k,v in s.__dict__.items() if k!='command_id'):raise ValueError('Run-in nonfinite sample')
        if self.state in {'failed','complete'}:return False
        if self.started is None:self.started=s.time_s;self.event('RUN_IN_STARTED',s)
        previous=self.last;self.last=s
        if s.fixture_error_m>.025 or s.fixture_speed_m_s>.035 or abs(s.probe_m)>.005 or abs(s.probe_velocity_m_s)>.015:
            self.fault='run_in_fixture_or_tool_departure'
        if s.time_s-self.started>25:self.fault='run_in_timeout'
        if previous:
            delta=math.atan2(math.sin(s.angle-previous.angle),math.cos(s.angle-previous.angle))
            if abs(delta)>2.5*(s.time_s-previous.time_s)+.01:self.fault='run_in_feedback_jump'
            self.rotation+=delta
        if self.fault:self.event('RUN_IN_FAILED',s,reason=self.fault);self.state='failed';return True
        if self.state=='running':
            if abs(s.omega-1.6)>.08:self.stable_since=None
            elif self.stable_since is None:self.stable_since=s.time_s
            if self.stable_since is not None and s.time_s-self.stable_since>=8 and self.rotation>=4*math.pi:
                self.event('RUN_IN_SPEED_DWELL_MEASURED',s,stable_s=s.time_s-self.stable_since,
                    stable_since_s=self.stable_since);self.state='braking'
        else:
            if abs(s.omega)>.025:self.stopped_since=None
            elif self.stopped_since is None:self.stopped_since=s.time_s
            if self.stopped_since is not None and s.time_s-self.stopped_since>=.25:
                self.event('RUN_IN_STOPPED',s,stopped_s=s.time_s-self.stopped_since);self.state='complete'
        return True
