"""Feedback-driven inspection cycle and bounded effort servo.

The fixture is an explicit simulation design assumption. Its measured joint
motion advances the cycle; issuing a command or waiting alone cannot complete it.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class JointSample:
    time_s: float
    slide: float
    slide_velocity: float
    probe: float
    probe_velocity: float
    angle: float
    angular_velocity: float


STATIONS=(('derust',11.0),('magnetic',25.45),('ultrasonic',34.5),('outfeed',42.5))
PHASE_TIMEOUTS={'transfer':60,'engage':8,'rotate':12,'brake':8,'retract':8}


class InspectionCycle:
    """One workpiece; keep the current station reserved after a latched fault."""
    def __init__(self, *, turns=1.0):
        if not math.isfinite(turns) or turns<=0:raise ValueError('turns must be positive')
        self.turns=turns;self.index=0;self.state='idle';self.events=[]
        self.last=None;self.entered=None;self.fault=None;self.completed=[]
        self.rotation=0.;self.previous_angle=None;self.peak_omega=0.

    @property
    def station(self):return STATIONS[min(self.index,len(STATIONS)-1)][0]

    def _event(self,kind,now,**data):
        self.events.append(dict(seq=len(self.events),event=kind,sim_time_s=now,
            station=self.station,**data))

    def _phase(self,state,now):
        self.state=state;self.entered=now;self._event('PHASE_ENTERED',now,phase=state)

    def fail(self,reason,now):
        if self.state in {'complete','failed'}:return
        self.fault=reason;self._event('CYCLE_FAILED',now,reason=reason,
            failed_phase=self.state,retained_station=self.station)
        self.state='failed'

    def update(self,sample):
        values=list(sample.__dict__.values())
        if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite joint feedback')
        if self.last and sample.time_s<=self.last.time_s:return False
        self.last=sample;now=sample.time_s
        if self.state in {'complete','failed'}:return True
        if self.state=='idle':self._phase('transfer',now)
        if now-self.entered>PHASE_TIMEOUTS[self.state]:
            self.fail(self.state+'_timeout',now);return True
        target=STATIONS[self.index][1]
        at_station=abs(sample.slide-target)<=.025 and abs(sample.slide_velocity)<=.035
        probe_home=abs(sample.probe)<=.005 and abs(sample.probe_velocity)<=.015
        stopped=abs(sample.angular_velocity)<=.025
        if self.state=='transfer':
            if at_station and probe_home and stopped:
                self._event('ARRIVAL_CONFIRMED',now,slide=sample.slide,slide_velocity=sample.slide_velocity)
                if self.station=='outfeed':
                    self.completed.append(self.station);self._event('CYCLE_COMPLETE',now)
                    self.state='complete'
                else:
                    self._event('DEVICE_BUSY',now);self._phase('engage',now)
        elif self.state=='engage':
            if at_station and abs(sample.probe-.22)<=.005 and abs(sample.probe_velocity)<=.015:
                self._event('PROBE_ENGAGED',now,probe=sample.probe)
                self.rotation=0.;self.previous_angle=sample.angle;self.peak_omega=0.
                self._phase('rotate',now)
        elif self.state=='rotate':
            if not at_station or abs(sample.probe-.22)>.015:
                self.fail('fixture_left_work_position',now);return True
            delta=math.atan2(math.sin(sample.angle-self.previous_angle),math.cos(sample.angle-self.previous_angle))
            self.rotation+=delta;self.previous_angle=sample.angle
            self.peak_omega=max(self.peak_omega,abs(sample.angular_velocity))
            if self.rotation>=self.turns*2*math.pi:
                self._event('ROTATION_MEASURED',now,angle_rad=self.rotation,peak_omega=self.peak_omega)
                self._phase('brake',now)
        elif self.state=='brake':
            if stopped:
                self._event('SPINDLE_STOPPED',now,angular_velocity=sample.angular_velocity)
                self._phase('retract',now)
        elif self.state=='retract':
            if probe_home and stopped and at_station:
                self.completed.append(self.station)
                self._event('DEVICE_COMPLETE',now,probe=sample.probe,angular_velocity=sample.angular_velocity)
                self._event('DEVICE_IDLE',now)
                self.index+=1;self._phase('transfer',now)
        return True

    def targets(self):
        if self.last is None:return (0.,0.,0.)
        if self.state in {'failed','complete'}:return (self.last.slide,0.,0.)
        return (STATIONS[self.index][1],.22 if self.state in {'engage','rotate','brake'} else 0.,
            1.6 if self.state=='rotate' else 0.)


def clamp(value,limit):return max(-limit,min(limit,value))


def efforts(sample,targets):
    """PD/velocity servo; force N, torque Nm, assumed masses and inertia."""
    x,probe,omega=targets
    error=x-sample.slide
    # A continuous position-to-velocity target avoids chattering near the clamp
    # position under sampled ROS feedback.
    velocity=clamp(2*error,2.)
    # Static friction is not modelled. Damping terms compensate configured joints.
    force=clamp(1020*7*(velocity-sample.slide_velocity)+50*sample.slide_velocity,7000)
    probe_force=clamp(5*(100*(probe-sample.probe)-18*sample.probe_velocity)+4*sample.probe_velocity,200)
    torque=clamp(160*8*(omega-sample.angular_velocity)+3*sample.angular_velocity,1000)
    return dict(wsd_slide=force,wsd_probe=probe_force,wsd_spin=torque)
