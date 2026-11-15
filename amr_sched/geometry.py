"""Metric operations on scheduler-owned road centerlines, without rerouting."""
from __future__ import annotations

import math


def project_polyline(point, points):
    """Return distance, closest point, segment index and arc length on a road."""
    if len(points) < 2 or any(not math.isfinite(v) for p in (point, *points) for v in p):
        raise ValueError('finite nondegenerate centerline required')
    best = None
    travelled = 0.
    for index, (a, b) in enumerate(zip(points, points[1:])):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        fraction = max(0., min(1., ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/(length*length)))
        closest = (a[0]+fraction*dx, a[1]+fraction*dy)
        candidate = (math.dist(point, closest), closest, index, travelled+fraction*length)
        if best is None or candidate[0] < best[0]:
            best = candidate
        travelled += length
    if best is None:
        raise ValueError('finite nondegenerate centerline required')
    return best


def remaining_polyline(point, points, tolerance):
    """Join the measured stop to its projection, retaining every future corner."""
    distance, closest, index, _ = project_polyline(point, points)
    if distance > tolerance:
        raise ValueError('actual pose outside retained corridor')
    result = []
    for p in (point, closest, *points[index+1:]):
        p = tuple(p)
        if not result or p != result[-1]:
            result.append(p)
    return tuple(result)


def point_along_polyline(points, fraction):
    """Interpolate display position by road arc length, including its corners."""
    remaining = max(0., min(1., fraction))*sum(math.dist(a,b) for a,b in zip(points,points[1:]))
    for a,b in zip(points,points[1:]):
        length = math.dist(a,b)
        if length and remaining <= length:
            return (a[0]+(b[0]-a[0])*remaining/length, a[1]+(b[1]-a[1])*remaining/length)
        remaining -= length
    return tuple(points[-1])
