"""Measured station localization; no truth, target attraction or motion authority.

The bounded map fit is the numerically replayed prototype. Its profile-cost
checks reject local ambiguity; they are not calibrated uncertainty estimates.
The owner alone selects a fresh fit as the common execution/display pose.
"""
from dataclasses import asdict, dataclass
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares


@dataclass(frozen=True)
class FitSpec:
    """Declared development bounds; no pose is certified from cost alone."""

    minimum_range_m: float = .65
    maximum_seed_distance_m: float = .20
    translation_bound_m: float = .12
    yaw_bound_rad: float = .10
    robust_scale_m: float = .02
    minimum_points: int = 60
    inlier_distance_m: float = .04
    minimum_inlier_fraction: float = .70
    maximum_inlier_rms_m: float = .02
    confidence_translation_m: float = .02
    confidence_yaw_rad: float = .01
    noise_floor_m: float = .003

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('positive finite fit parameter required: '+name)
        if type(self.minimum_points) is not int or self.minimum_points < 3:
            raise ValueError('at least three physical returns required')
        if self.minimum_inlier_fraction > 1:
            raise ValueError('inlier fraction must be at most one')


class ScanMatcher:
    """Match measured ray endpoints to a fixed signed-distance map."""

    def __init__(self, pixels, resolution, origin, spec=FitSpec()):
        self.spec = spec
        image = np.asarray(pixels,dtype=np.uint8)
        if image.ndim != 2 or min(image.shape) < 3:
            raise ValueError('two-dimensional occupancy raster required')
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError('positive map resolution required')
        self.height,self.width = image.shape
        self.resolution = float(resolution)
        self.origin = np.asarray(origin[:2],dtype=float)
        if self.origin.shape != (2,) or not np.isfinite(self.origin).all():
            raise ValueError('finite map origin required')
        occupied = image < 127
        if not occupied.any() or occupied.all():
            raise ValueError('both observed free and occupied cells required')
        self.field = (distance_transform_edt(~occupied)-distance_transform_edt(occupied))*resolution

    def sample(self, points):
        """Bilinear distances and their exact planar interpolation derivatives."""
        points = np.asarray(points,dtype=float)
        gx = (points[:,0]-self.origin[0])/self.resolution-.5
        gy = self.height-.5-(points[:,1]-self.origin[1])/self.resolution
        columns,rows = np.floor(gx).astype(int),np.floor(gy).astype(int)
        inside = (columns>=0)&(columns<self.width-1)&(rows>=0)&(rows<self.height-1)
        values = np.full(len(points),.4)
        gradients = np.zeros((len(points),2))
        c,r = columns[inside],rows[inside]
        u,v = gx[inside]-c,gy[inside]-r
        a,b = self.field[r,c],self.field[r,c+1]
        d,e = self.field[r+1,c],self.field[r+1,c+1]
        values[inside] = (1-u)*(1-v)*a+u*(1-v)*b+(1-u)*v*d+u*v*e
        gradients[inside,0] = ((1-v)*(b-a)+v*(e-d))/self.resolution
        gradients[inside,1] = -((1-u)*(d-a)+u*(e-b))/self.resolution
        return values,gradients,inside

    def fit(self, ranges, angle_min, angle_increment, laser_xyyaw, prior, range_max):
        """The fitting interface deliberately has no truth or target argument."""
        started = time.perf_counter()
        seed = np.asarray(prior,dtype=float)
        values = np.asarray(ranges,dtype=float)
        extrinsic = np.asarray(laser_xyyaw,dtype=float)
        if (seed.shape != (3,) or extrinsic.shape != (3,)
                or not np.isfinite(seed).all() or not np.isfinite(extrinsic).all()
                or values.ndim != 1 or len(values)<self.spec.minimum_points
                or not all(math.isfinite(v) for v in (angle_min,angle_increment,range_max))
                or angle_increment <= 0 or range_max <= self.spec.minimum_range_m):
            return dict(valid=False,reason='invalid scan metadata')
        angles = angle_min+np.arange(len(values))*angle_increment+extrinsic[2]
        valid = np.isfinite(values)&(values>=self.spec.minimum_range_m)&(values<range_max-.02)
        base = np.column_stack((values[valid]*np.cos(angles[valid])+extrinsic[0],
                                values[valid]*np.sin(angles[valid])+extrinsic[1]))
        if len(base)<self.spec.minimum_points:
            return dict(valid=False,reason='insufficient physical returns',points=len(base))

        def world(pose):
            c,s = math.cos(pose[2]),math.sin(pose[2])
            return np.column_stack((pose[0]+c*base[:,0]-s*base[:,1],
                                    pose[1]+s*base[:,0]+c*base[:,1]))
        initial,_,inside = self.sample(world(seed))
        usable = inside&(np.abs(initial)<=self.spec.maximum_seed_distance_m)
        base = base[usable]
        if len(base)<self.spec.minimum_points:
            return dict(valid=False,reason='insufficient mapped support',points=len(base))

        def residual(pose):
            return self.sample(world(pose))[0]

        def jacobian(pose):
            _,gradient,_ = self.sample(world(pose))
            c,s = math.cos(pose[2]),math.sin(pose[2])
            turn_x = -s*base[:,0]-c*base[:,1]
            turn_y = c*base[:,0]-s*base[:,1]
            return np.column_stack((gradient,gradient[:,0]*turn_x+gradient[:,1]*turn_y))

        bounds = np.array([self.spec.translation_bound_m,self.spec.translation_bound_m,self.spec.yaw_bound_rad])
        solved = least_squares(residual,seed,jac=jacobian,bounds=(seed-bounds,seed+bounds),
            method='trf',loss='huber',f_scale=self.spec.robust_scale_m,
            max_nfev=30,ftol=1e-7,xtol=1e-7,gtol=1e-7,x_scale=[.05,.05,.05])
        r = residual(solved.x)
        inlier = np.abs(r)<self.spec.inlier_distance_m
        rms = float(np.sqrt(np.mean(r[inlier]**2))) if inlier.any() else math.inf
        jac = jacobian(solved.x)[inlier]
        eigenvalues,vectors = np.linalg.eigh(jac.T@jac) if len(jac) else (np.zeros(3),np.eye(3))
        sigma = max(self.spec.noise_floor_m,float(np.median(np.abs(r[inlier])))*1.4826) if inlier.any() else math.inf
        threshold = 3.84*sigma*sigma

        def cost(pose):
            errors = np.abs(residual(pose)[inlier])
            delta = self.spec.robust_scale_m
            return float(np.sum(np.where(errors<=delta,errors**2,2*delta*errors-delta**2)))
        center_cost = cost(solved.x)
        profiles = []
        for direction in vectors.T:
            scale = max(np.max(np.abs(direction[:2]))/self.spec.confidence_translation_m,
                        abs(direction[2])/self.spec.confidence_yaw_rad)
            step = direction/scale
            profiles.append([cost(solved.x-step)-center_cost,cost(solved.x+step)-center_cost])
        checks = dict(solver_converged=bool(solved.success),
            interior_solution=bool(np.all(np.abs(solved.x-seed)<.995*bounds)),
            enough_inliers=int(inlier.sum())>=self.spec.minimum_points,
            inlier_fraction=float(inlier.mean())>=self.spec.minimum_inlier_fraction,
            residual_rms=rms<=self.spec.maximum_inlier_rms_m,
            bounded_profile=bool(np.all(np.asarray(profiles)>threshold)))
        pose = solved.x.tolist()
        pose[2] = math.atan2(math.sin(pose[2]),math.cos(pose[2]))
        return dict(valid=all(checks.values()),checks=checks,pose=pose,prior=seed.tolist(),
            points=len(base),inliers=int(inlier.sum()),rms_m=rms,sigma_m=sigma,
            profile_cost_increases=profiles,profile_threshold=threshold,
            information_eigenvalues=eigenvalues.tolist(),evaluations=solved.nfev,
            elapsed_s=time.perf_counter()-started)


def scan_snapshot(message, received):
    """Detach a sensor message without granting it a new receipt or source time."""
    return dict(source_s=message.header.stamp.sec+message.header.stamp.nanosec*1e-9,
        received_wall_s=received, frame_id=message.header.frame_id,
        angle_min=float(message.angle_min), angle_max=float(message.angle_max),
        angle_increment=float(message.angle_increment), range_min=float(message.range_min),
        range_max=float(message.range_max), ranges=tuple(message.ranges))


def laser_pose(transform):
    """Planar projection of an observed TF, rejecting a tilted or invalid laser."""
    t, q = transform.transform.translation, transform.transform.rotation
    values = [t.x, t.y, t.z, q.x, q.y, q.z, q.w]
    if (not all(math.isfinite(v) for v in values)
            or abs(sum(v*v for v in values[3:])-1.) > 1e-5
            or abs(q.x) > 1e-5 or abs(q.y) > 1e-5):
        raise ValueError('finite horizontal normalized laser transform required')
    return [t.x, t.y, t.z, math.atan2(2*q.w*q.z, 1-2*q.z*q.z)]


class StationRefiner:
    """Owner-side scan admission, causal seeding and immutable fit provenance."""

    freshness_s = .15
    seed_age_s = .2
    capture_radius_m = .35

    def __init__(self, descriptor):
        self.descriptor = json.loads(json.dumps(descriptor, allow_nan=False))
        d = self.descriptor
        if d['version'] != 'station_scan_v1' or d['resolution_m'] != .02:
            raise ValueError('declared 2cm station map required')
        self.stations = d['stations']
        if (not self.stations or any(len(p) != 2 or not all(math.isfinite(v) for v in p)
                                    for p in self.stations.values())):
            raise ValueError('declared finite handoff stations required')
        # Bind the actual original raster, not a visual or resampled substitute.
        raw = Path(d['map_image']).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != d['map_sha256']:
            raise ValueError('station map hash mismatch')
        header, dimensions, maximum, payload = raw.split(b'\n', 3)
        if header != b'P5' or maximum != b'255':
            raise ValueError('explicit binary occupancy raster required')
        width, height = map(int, dimensions.split())
        if len(payload) != width*height:
            raise ValueError('station raster length mismatch')
        if len(d['origin']) != 3 or d['origin'][2] != 0:
            raise ValueError('unrotated station raster required')
        self.matcher = ScanMatcher(np.frombuffer(payload, dtype=np.uint8).reshape(height, width),
                                   d['resolution_m'], d['origin'], FitSpec(**d['fit']))
        self.history = deque(maxlen=64)
        self.consumed_source = -math.inf
        self.latest = None
        self.pending_fit = None

    def note_prior(self, pose, source_s):
        """Retain only new global measurements, before selecting fine feedback."""
        if (len(pose) == 3 and all(math.isfinite(v) for v in (*pose, source_s))
                and (not self.history or source_s > self.history[-1][0])):
            self.history.append((source_s, tuple(pose)))

    def station(self, prior):
        candidates = [(math.dist(prior[:2], xy), name) for name, xy in self.stations.items()]
        distance, name = min(candidates)
        return name if distance <= self.capture_radius_m else None

    def fresh(self, source_now, wall_now):
        row = self.latest
        return bool(row and row['valid'] and
            0 <= source_now-row['source_s'] <= self.freshness_s and
            0 <= wall_now-row['received_wall_s'] <= self.freshness_s)

    def measure(self, scan, extrinsic, source_now, wall_now, minimum_source_s, fit_worker=None):
        """One fit per original scan. A reject never borrows freshness from cache."""
        completed = None
        if fit_worker is not None:
            answer = fit_worker.take()
            if answer is not None:
                if self.pending_fit is None or answer['token'] != self.pending_fit['worker_token']:
                    raise RuntimeError('numerical fit has no matching admitted scan')
                completed = self.pending_fit
                self.pending_fit = None
                completed['fit'] = answer['fit']
                completed['valid'] = bool(answer['fit']['valid'])
                if (not 0 <= source_now-completed['source_s'] <= self.freshness_s
                        or not 0 <= wall_now-completed['received_wall_s'] <= self.freshness_s
                        or completed['source_s'] <= minimum_source_s):
                    completed.update(valid=False, reason='completed fit stale or superseded')
                if completed['valid']:
                    completed['pose'] = answer['fit']['pose']
                self.latest = completed
        if scan is None:
            return dict(valid=False, reason='station scan unavailable')
        stamp, received = scan['source_s'], scan['received_wall_s']
        if (all(math.isfinite(v) for v in (stamp,received,source_now,wall_now))
                and (stamp>source_now or received>wall_now)):
            # Quarantine a not-yet-causal input. It grants no progress, but
            # must not erase an earlier independently fresh accepted sample.
            # A just-completed older fit can still be admitted on its own times.
            if completed is not None:
                completed['quarantined_input_source_s']=stamp
                return completed
            return dict(valid=False,duplicate=True,reason='station input not yet causal',source_s=stamp)
        if (not all(math.isfinite(v) for v in (stamp, received, source_now, wall_now))
                or not 0 <= source_now-stamp <= self.freshness_s
                or not 0 <= wall_now-received <= self.freshness_s):
            self.latest = None
            return dict(valid=False, reason='station scan stale or future', source_s=stamp)
        if stamp <= self.consumed_source:
            return completed or dict(valid=False, reason='station scan already consumed', duplicate=True, source_s=stamp)
        if fit_worker is not None:
            # A corrupt new scan must invalidate a cached pose even while an
            # older numerical request is in flight.
            expected = self.descriptor['laser']
            numeric = ('angle_min','angle_max','angle_increment','range_min','range_max')
            if (scan['frame_id'] != expected['frame_id'] or len(scan['ranges']) != expected['beams']
                    or any(not math.isfinite(scan[k]) or abs(scan[k]-expected[k]) > 1e-5 for k in numeric)
                    or extrinsic is None or len(extrinsic) != 4
                    or any(not math.isfinite(v) or abs(v-e) > 1e-5
                           for v,e in zip(extrinsic,expected['xyzyaw']))):
                self.latest = None
                return dict(valid=False,source_s=stamp,reason='live station metadata or static TF invalid')
        if fit_worker is not None and (not fit_worker.ready or fit_worker.busy):
            return completed or dict(valid=False, reason='station numerical peer pending', duplicate=True)
        self.consumed_source = stamp
        if fit_worker is None:
            self.latest = None
        row = dict(valid=False, source_s=stamp, received_wall_s=received)
        expected = self.descriptor['laser']
        try:
            meta = {k: scan[k] for k in ('frame_id', 'angle_min', 'angle_max',
                                        'angle_increment', 'range_min', 'range_max')}
            if (meta['frame_id'] != expected['frame_id']
                    or len(scan['ranges']) != expected['beams']
                    or any(not math.isfinite(meta[k]) or abs(meta[k]-expected[k]) > 1e-5
                           for k in meta if k != 'frame_id')):
                raise ValueError('live scan metadata differs from declared physical laser')
            if extrinsic is None or len(extrinsic) != 4 or any(
                    not math.isfinite(v) or abs(v-e) > 1e-5
                    for v, e in zip(extrinsic, expected['xyzyaw'])):
                raise ValueError('live static laser TF differs from declared physical laser')
            row.update(scan={**meta, 'ranges': [v if math.isfinite(v) else None for v in scan['ranges']]},
                       nonfinite_ranges='null represents no finite return', laser_xyzyaw=list(extrinsic),
                       map_sha256=self.descriptor['map_sha256'])
            if stamp <= minimum_source_s:
                raise ValueError('station scan predates accepted execution pose')
            prior = next(((t, p) for t, p in reversed(self.history)
                          if 0 <= stamp-t <= self.seed_age_s), None)
            if prior is None:
                raise ValueError('causal global prior unavailable at scan source time')
            row.update(prior_source_s=prior[0], prior=list(prior[1]))
            inputs = (scan['ranges'], meta['angle_min'], meta['angle_increment'],
                      [extrinsic[0], extrinsic[1], extrinsic[3]], prior[1], meta['range_max'])
            if fit_worker is not None:
                if not fit_worker.submit(inputs):
                    raise RuntimeError('numerical fit admission lost its reserved slot')
                row['worker_token'] = fit_worker.sequence
                self.pending_fit = row
                return completed or dict(valid=False, duplicate=True, reason='station fit submitted')
            fit = self.matcher.fit(*inputs)
            row.update(valid=fit['valid'], fit=fit)
            if fit['valid']:
                row['pose'] = fit['pose']
        except (ValueError, KeyError, FloatingPointError) as exc:
            row['reason'] = str(exc)
        self.latest = row
        return row
