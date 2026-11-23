"""Source-time and receipt-time validity for the sensor localization authority."""
from __future__ import annotations

import math


class SensorHealth:
    """Reject stale, invalid and uncertain inputs before they refresh progress."""

    def __init__(self):
        self.samples={}
        self.limits={'scan':.3,'imu':.2,'odom':.2,'amcl':2.}

    def observe(self,source,stamp,received,valid=True,reason='invalid sample'):
        if source not in self.limits:
            raise ValueError('unknown localization input')
        previous=self.samples.get(source)
        if (not math.isfinite(stamp) or not math.isfinite(received)
                or previous and stamp<=previous['stamp']):
            # Duplicate/replayed source data must not refresh receipt freshness.
            return
        self.samples[source]=dict(stamp=stamp,received=received,valid=valid,reason=reason)

    def reason(self,clock,now):
        if not math.isfinite(clock) or not math.isfinite(now):
            return 'invalid localization clock'
        for source,limit in self.limits.items():
            sample=self.samples.get(source)
            if sample is None:
                return source+' unavailable'
            if not sample['valid']:
                return source+': '+sample['reason']
            if not 0<=now-sample['received']<=limit or not -.1<=clock-sample['stamp']<=limit:
                return source+' stale'
        return None


def pose_quality(covariance, position_xy=None):
    """AMCL map covariance is a quality check, not a physical error guarantee."""
    if len(covariance)!=36:
        return False
    if position_xy is not None and (len(position_xy)!=2 or not all(math.isfinite(v) for v in position_xy)):
        return False
    # AMCL computes E[x*x]-E[x]*E[x] with <=2000 weighted particles. At metre
    # coordinates around y=68, cancellation yielded -2.8e-10 m² after collapse.
    # Accept only bounded floating-point roundoff, never a negative uncertainty
    # of physical size. The 1e-9 m² cap is (0.032mm)²; raw data stays untouched.
    lower=[0.,0.,0.]
    if position_xy is not None:
        # Beyond 64m the capped allowance already applies. Cap before squaring
        # so even a finite but malformed huge coordinate cannot overflow here.
        lower=[-min(1e-9,2048*math.ulp(min(64.,max(1.,abs(v)))**2)) for v in position_xy]
        lower.append(-2048*math.ulp(1.))
    # Position sigma <= .25m, yaw sigma <= .20rad; physical arrival still .15m.
    return all(math.isfinite(covariance[i]) and minimum<=covariance[i]<=limit
               for (i,limit),minimum in zip(((0,.25**2),(7,.25**2),(35,.20**2)),lower))
