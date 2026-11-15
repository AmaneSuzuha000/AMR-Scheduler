"""Sample an approved polyline for Nav2 without changing its route or corners.

Graph nodes can be metres apart. RPP prunes to the nearest *sample*, then chooses
the first sample outside the lookahead circle. With sparse input that sample can
be behind a forward-moving robot. Supply a bounded spatial spacing in world
metres; this is path representation, not another global planner.
"""
from __future__ import annotations

import math
from collections.abc import Iterable


def shift_tracking_reference(samples, offset_m=0.):
    """Express the same pose sequence at a fixed forward body reference point.

    This changes neither the center's approved route nor its heading. A native
    controller using the new frame must use the equally shifted footprint.
    """
    if not math.isfinite(offset_m) or offset_m<0 or offset_m>.315:
        raise ValueError('tracking offset must lie within the configured body footprint')
    return [(x+offset_m*math.cos(yaw),y+offset_m*math.sin(yaw),yaw) for x,y,yaw in samples]


def sample_polyline(points: Iterable[tuple[float, float]],
                    max_spacing_m: float = 0.05) -> list[tuple[float, float, float]]:
    """Return (x, y, yaw) samples, preserving every distinct input vertex.

Coordinates and spacing are in the same metric world frame. Consecutive equal
vertices are removed; nonconsecutive revisits and sharp corners remain. The last
heading follows the final segment. No smoothing, shortcutting or rerouting is
performed. Inputs must define a finite, nonzero-length path.
"""
    if not math.isfinite(max_spacing_m) or max_spacing_m <= 0:
        raise ValueError("max_spacing_m must be finite and positive")
    vertices = []
    for point in points:
        x, y = map(float, point)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("path coordinates must be finite")
        if not vertices or (x, y) != vertices[-1]:
            vertices.append((x, y))
    if len(vertices) < 2:
        raise ValueError("path needs at least two distinct consecutive points")

    samples = [vertices[0]]
    for (ax, ay), (bx, by) in zip(vertices, vertices[1:]):
        distance = math.hypot(bx - ax, by - ay)
        count_float = distance / max_spacing_m
        if not math.isfinite(count_float) or count_float > 100_000:
            raise ValueError("path sampling exceeds 100000 points")
        count = max(1, math.ceil(count_float))
        if len(samples) + count > 100_000:
            raise ValueError("path sampling exceeds 100000 points")
        for i in range(1, count):
            samples.append((ax + (bx - ax) * i / count,
                            ay + (by - ay) * i / count))
        # Keep the exact supplied endpoint, including at sharp corners.
        samples.append((bx, by))

    result = []
    for i, (x, y) in enumerate(samples):
        ax, ay = samples[i - 1] if i == len(samples) - 1 else (x, y)
        bx, by = (x, y) if i == len(samples) - 1 else samples[i + 1]
        result.append((x, y, math.atan2(by - ay, bx - ax)))
    return result
