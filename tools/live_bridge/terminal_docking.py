"""Owner-side terminal approach for a station hop.

Why this exists
---------------
The transit controller is a corridor follower. Regulated pure pursuit with a
minimum lookahead of .25 m cannot resolve a handoff pose 1.5 cm from the
station: once the remaining path is shorter than the lookahead the carrot is
clamped to the endpoint and the vehicle traces a limit cycle around it. The
recorded delivery campaign shows exactly that - the closest approach was
3.95 cm with the action still running 114 s later, when the progress checker
aborted it.

What this is not
----------------
Nothing here is a new source of truth. It produces one twist from one measured
pose. The caller keeps every existing gate: the guard lease, the retained
action, the arrival event and the handling receipt are untouched, and the
arrival still has to be proved by a fresh stopped guard on the same route.

The module is deliberately pure: no ROS, no clock, no I/O, so the whole
regulator is testable without a simulator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def wrap_angle(angle: float) -> float:
    """Normalise an angle into (-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def clamp_between(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class DockSpec:
    """Declared capture window and the twist ceiling used inside it.

    ``capture_radius_m`` is where the owner may take the terminal over, so it
    has to be strictly wider than the docking tolerance it is closing. The
    twist ceilings stay well inside the independent guard's rejection
    thresholds; they are not a tuning opportunity.
    """

    capture_radius_m: float
    position_tolerance_m: float
    yaw_tolerance_rad: float
    max_linear: float = .10
    max_angular: float = .40
    reverse_linear: float = .04
    min_linear: float = .02
    forward_gain: float = .50
    lateral_gain: float = 1.50
    max_correction_rad: float = .60
    hold_s: float = .30
    stop_speed: float = .01
    stop_wz: float = .02
    deadline_s: float = 45.
    arrival_deadline_s: float = 10.
    linear_gain: float = 1.5
    heading_gain: float = 1.5
    steering_mode: str = "axis"
    polar_turn_gain: float = 3.
    polar_yaw_gain: float = 1.
    staging_distance_m: float = .22
    staging_yaw_limit_rad: float = .08
    approach_stations: tuple = ()

    def __post_init__(self) -> None:
        for name in ("capture_radius_m", "position_tolerance_m", "yaw_tolerance_rad",
                     "max_linear", "max_angular", "reverse_linear", "min_linear",
                     "forward_gain", "lateral_gain", "max_correction_rad",
                     "stop_speed", "stop_wz", "deadline_s", "arrival_deadline_s",
                     "linear_gain", "heading_gain", "polar_turn_gain", "polar_yaw_gain",
                     "staging_distance_m", "staging_yaw_limit_rad"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not math.isfinite(value) or value <= 0:
                raise ValueError("positive finite docking parameter required: " + name)
        if not isinstance(self.hold_s, (int, float)) or isinstance(self.hold_s, bool) \
                or not math.isfinite(self.hold_s) or self.hold_s < 0:
            raise ValueError("non-negative finite hold time required")
        if self.position_tolerance_m >= self.capture_radius_m:
            raise ValueError("capture radius must exceed the docking tolerance it closes")
        if self.yaw_tolerance_rad >= math.pi:
            raise ValueError("one meaningful yaw tolerance required")
        if self.min_linear > self.max_linear or self.max_correction_rad >= math.pi:
            raise ValueError("an unbounded docking command is not a docking command")
        if self.steering_mode not in ("axis","polar","staged"):
            raise ValueError("declared axis, polar or staged docking steering required")
        if self.steering_mode in ("polar","staged") and self.polar_turn_gain <= self.linear_gain:
            raise ValueError("polar turn gain must exceed distance gain")
        if self.steering_mode == "staged" and not (
                4*self.position_tolerance_m < self.staging_distance_m < self.capture_radius_m
                and self.yaw_tolerance_rad < self.staging_yaw_limit_rad < .15):
            raise ValueError("staging must fit inside capture with a bounded insertion heading")
        for station in self.approach_stations:
            xy=station.get("target_xy",())
            heading=station.get("yaw_rad")
            side=station.get("approach_side")
            if (len(xy)!=2 or not all(type(v) in (int,float) and math.isfinite(v) for v in xy)
                    or type(heading) not in (int,float) or not math.isfinite(heading)
                    or type(side) is not int or side not in (-1,1)):
                raise ValueError("explicit finite station pose and approach side required")


class DockRegulator:
    """One measured pose in, one twist and one phase out.

    Errors are expressed on the dock axis, never as a bearing to the endpoint.
    A bearing flips by pi the moment the vehicle overshoots, and the earlier
    take-over duly spun the base around on the spot; the dock axis keeps a
    small lateral error small, and an overshoot is corrected by a bounded
    reverse instead of a turn.

    ``hold`` is a damped settle, not a blind zero: the twist is zero once the
    pose is inside both windows, and the settle window is a leaky integrator,
    so one noisy sample cannot erase a real quiet period while a body that is
    still coasting can never be certified as settled.
    """

    phases = ("approach", "hold", "done", "withdraw", "align", "insert", "blocked")

    def __init__(self, spec: DockSpec, target_xy, target_yaw: float, *, approach_side=-1) -> None:
        spec_position = tuple(float(v) for v in target_xy)
        if (len(spec_position) != 2 or not all(math.isfinite(v) for v in spec_position)
                or not isinstance(target_yaw, (int, float)) or isinstance(target_yaw, bool)
                or not math.isfinite(target_yaw)):
            raise ValueError("finite docking target required")
        self.spec = spec
        self.target_xy = spec_position
        self.target_yaw = wrap_angle(float(target_yaw))
        self.settled_s = 0.
        self.last_now = None
        if type(approach_side) is not int or approach_side not in (-1,1):
            raise ValueError("approach side must be -1 or 1 on the dock axis")
        self.approach_side=approach_side
        self.stage=None
        self.lateral_stall_s=0.
        self.course=None
        self.withdraw_distance=None

    def update(self, pose, speed: float, wz: float, now: float) -> dict:
        """One twist for the current measured pose; never infers motion."""
        x, y, yaw = (float(v) for v in pose)
        if not all(math.isfinite(v) for v in (x, y, yaw, speed, wz, now)):
            raise ValueError("finite docking measurement required")
        elapsed = 0. if self.last_now is None else max(0., now - self.last_now)
        self.last_now = now
        dx, dy = self.target_xy[0] - x, self.target_xy[1] - y
        distance = math.hypot(dx, dy)
        yaw_error = wrap_angle(self.target_yaw - yaw)
        cos_t, sin_t = math.cos(self.target_yaw), math.sin(self.target_yaw)
        forward = cos_t * dx + sin_t * dy
        lateral = -sin_t * dx + cos_t * dy
        position_inside = distance <= self.spec.position_tolerance_m
        heading_inside = abs(yaw_error) <= self.spec.yaw_tolerance_rad
        inside = position_inside and heading_inside
        answer = dict(distance=distance, forward_error=forward, lateral_error=lateral,
                      heading_error=wrap_angle(math.atan2(dy, dx) - yaw),
                      yaw_error=yaw_error, aligned=inside)
        if inside:
            self.lateral_stall_s=0.
            quiet = abs(speed) <= self.spec.stop_speed and abs(wz) <= self.spec.stop_wz
            if quiet:
                self.settled_s = min(self.spec.hold_s, self.settled_s + elapsed)
            else:
                self.settled_s = max(0., self.settled_s - 2. * elapsed)
            done = self.settled_s >= self.spec.hold_s
            return dict(answer, phase="done" if done else "hold", vx=0., wz=0.,
                        settled_s=self.settled_s, done=done)
        self.settled_s = max(0., self.settled_s - 2. * elapsed)
        if self.spec.steering_mode == "staged":
            return self._staged((x,y,yaw),speed,wz,now,answer,elapsed)
        if self.spec.steering_mode == "polar":
            if position_inside:
                # Once position is captured, turn on the spot to the declared
                # dock heading. A noisy bearing inside a 5mm ball has no value.
                return dict(answer,phase="approach",vx=0.,
                    wz=clamp(self.spec.heading_gain*yaw_error,self.spec.max_angular),
                    settled_s=self.settled_s,done=False)
            alpha=answer["heading_error"]
            reverse=math.cos(alpha)<0.
            if reverse:
                # Express a reverse approach in the virtual forward frame.
                # Overshooting the dock then requests bounded reverse motion
                # rather than a half turn around a nearly coincident target.
                alpha=wrap_angle(alpha-math.pi)
            beta=wrap_angle(yaw_error-alpha)
            ceiling=self.spec.reverse_linear if reverse else self.spec.max_linear
            linear=min(ceiling,self.spec.linear_gain*distance*max(0.,math.cos(alpha)))
            turn=self.spec.polar_turn_gain*alpha-self.spec.polar_yaw_gain*beta
            return dict(answer,phase="approach",vx=-linear if reverse else linear,
                wz=clamp(turn,self.spec.max_angular),reverse=reverse,
                settled_s=self.settled_s,done=False)
        correction = clamp(math.atan2(self.spec.lateral_gain * lateral, self.spec.forward_gain),
                           self.spec.max_correction_rad)
        heading_error = wrap_angle(self.target_yaw + correction - yaw)
        linear = self.spec.linear_gain * forward
        if forward >= self.spec.min_linear:
            linear = max(linear, self.spec.min_linear)
        linear = clamp_between(linear, -self.spec.reverse_linear, self.spec.max_linear)
        return dict(answer, phase="approach", vx=linear,
                    wz=clamp(self.spec.heading_gain * heading_error, self.spec.max_angular),
                    steer_error=heading_error, settled_s=self.settled_s, done=False)

    def _staged(self,pose,speed,wz,now,answer,elapsed):
        """Turn at an explicit retreat pose, then follow the station axis.

        LOAD racks sit behind this carrier and UNLOAD tables in front. The
        signed staging side comes from the physical station declaration, not
        a guessed common reverse direction. Transit/resource authority and
        final acceptance remain with the caller.
        """
        side=self.approach_side
        axial=-answer["forward_error"]
        clearance=side*axial
        yaw_error=answer["yaw_error"]
        if self.withdraw_distance is None:
            projected_lateral=(abs(answer['lateral_error'])
                +max(0.,self.spec.staging_distance_m-clearance)*abs(math.tan(yaw_error)))
            self.withdraw_distance=(min(.85*self.spec.capture_radius_m,.30)
                if projected_lateral>.05 else self.spec.staging_distance_m)
        distance=self.withdraw_distance
        base=dict(answer,settled_s=self.settled_s,done=False,
                  approach_side=side,staging_clearance_m=clearance)
        if answer["distance"]>self.spec.capture_radius_m or abs(yaw_error)>math.pi/3:
            return dict(base,phase="blocked",vx=0.,wz=0.)
        if self.stage is None:
            self.stage="align" if clearance>=distance*.8 else "withdraw"
        heading_limit=min(self.spec.max_correction_rad,
                          max(self.spec.staging_yaw_limit_rad,
                              self.spec.yaw_tolerance_rad+max(0.,clearance-.01)*3.))
        # At the axial endpoint, straight corrections cannot remove a remaining
        # lateral error. Trigger withdrawal once forward error is inside the dock
        # arrival neighbourhood while lateral error is outside the position gate
        # (>tol = 5mm) and heading is aligned, sustained over a leaky time window.
        lateral_outside = abs(answer['lateral_error']) > self.spec.position_tolerance_m
        axial_arrived = abs(answer['forward_error']) <= 1.5 * self.spec.position_tolerance_m
        heading_aligned = abs(yaw_error) <= self.spec.yaw_tolerance_rad
        lateral_stall = axial_arrived and lateral_outside and heading_aligned
        self.lateral_stall_s=(self.lateral_stall_s+elapsed if lateral_stall
                              else max(0.,self.lateral_stall_s-elapsed))
        if self.stage=="insert" and (
                abs(yaw_error)>.60
                or abs(answer["lateral_error"])>.12
                or self.lateral_stall_s>=max(.6,2.0*self.spec.hold_s)):
            self.stage="withdraw"
            self.lateral_stall_s=0.
            self.course=None
            self.withdraw_distance=min(min(.85*self.spec.capture_radius_m,.30),
                                       max(.08,abs(answer['lateral_error'])*18.))
        if self.stage=="withdraw":
            # Near the table, changing heading would sweep a platform corner
            # into its edge. Withdraw along the measured body heading first.
            remaining=distance-clearance
            if remaining>self.spec.position_tolerance_m:
                ceiling=self.spec.max_linear if side>0 else self.spec.reverse_linear
                velocity=side*min(ceiling,self.spec.linear_gain*remaining)
                return dict(base,phase="withdraw",vx=velocity,wz=0.)
            self.stage="align"
        if self.stage=="align":
            if clearance<distance*.65:
                self.stage="withdraw"
                return dict(base,phase="withdraw",vx=side*min(
                    self.spec.reverse_linear,self.spec.linear_gain*(distance-clearance)),wz=0.)
            entry_heading=clamp(math.atan2(side*-2*answer['lateral_error'],clearance),
                                .9*heading_limit)
            entry_error=wrap_angle(yaw_error+entry_heading)
            if abs(entry_error)<=self.spec.yaw_tolerance_rad:
                self.stage="insert"
                self.course=(clearance,-answer['lateral_error'])
            else:
                return dict(base,phase="align",vx=0.,
                    wz=clamp(self.spec.heading_gain*entry_error,self.spec.max_angular))
        if (abs(answer['forward_error'])<=self.spec.position_tolerance_m
                and abs(answer['lateral_error'])<=3*self.spec.position_tolerance_m):
            # Small axis errors can still lie outside the circular position
            # gate. Finish heading first, then keep closing the axial error;
            # only update() may accept the full position and stopped pose.
            if abs(yaw_error)<=self.spec.yaw_tolerance_rad:
                linear=clamp_between(self.spec.linear_gain*answer['forward_error'],
                                     -self.spec.reverse_linear,self.spec.max_linear)
                if answer['forward_error']>0:
                    linear=max(self.spec.stop_speed*1.2,linear)
                elif answer['forward_error']<0:
                    linear=min(-self.spec.stop_speed*1.2,linear)
            else:
                linear=0.
            return dict(base,phase="insert",vx=linear,
                        wz=clamp(self.spec.heading_gain*yaw_error,self.spec.max_angular))
        if self.course is None:
            self.stage='withdraw'
            return dict(base,phase='withdraw',vx=side*self.spec.reverse_linear,wz=0.)
        if abs(answer['lateral_error'])<=self.spec.position_tolerance_m:
            self.course=(max(clearance,.05),0.)
        length,offset=self.course
        remaining=clamp_between(clearance/length,0.,1.)
        reference_y=offset*remaining**2
        slope=side*offset*2*remaining/length
        second=offset*2/length**2
        taper=min(1.,max(0.,remaining*2.))
        correction=taper*clamp(-side*math.atan2(reference_y+answer['lateral_error'],.06),.12)
        desired_heading=clamp(math.atan(slope)+correction,.9*heading_limit)
        steer=wrap_angle(yaw_error+desired_heading)
        linear=clamp_between(self.spec.linear_gain*answer["forward_error"]*max(0.,math.cos(steer)),
                             -self.spec.reverse_linear,self.spec.max_linear)
        if clearance<.12:
            linear=min(linear,max(self.spec.min_linear,0.75*clearance/.12*self.spec.max_linear))
            if abs(yaw_error)>self.spec.yaw_tolerance_rad:
                linear=min(linear,self.spec.min_linear)
        if abs(yaw_error)>heading_limit:
            linear=0.
        feedforward=(second/(1+slope*slope)*linear*math.cos(yaw_error)
                     if clearance>=.10 else 0.)
        return dict(base,phase="insert",vx=linear,
                    wz=clamp(feedforward+max(self.spec.heading_gain,2.5)*steer,self.spec.max_angular))
