"""Tests for the screenshot checker that proves the schedule window drew its fleet.

The checker is the only thing standing between "the panel printed some numbers" and "a picture
appeared", so the ways it could pass for the wrong reason get their own tests: a stale snapshot,
a colour paired with the wrong vehicle, and a picture that is a scaled or rotated version of
the numbers rather than the same layout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.verify_schedule_screen import (  # noqa: E402
    PALETTE,
    body_centroids,
    fit_similarity,
    parse_summary,
)


SUMMARY = """=== schedule stream status ===
status=状态: 已握手
--- summary t+4s ---
step=81 t=81.0 frames=41 scale=7.32px/m
AGV00 | to_pickup | T0000 | px=(500,588)
AGV01 | to_pickup | T0019 | px=(500,368)
AGV02 | to_dropoff | T0018 | px=(793,646)
--- summary t+7s ---
step=112 t=112.0 frames=72 scale=7.32px/m
AGV00 | to_pickup | T0000 | px=(368,683)
AGV01 | to_pickup | T0019 | px=(500,595)
AGV02 | to_dropoff | T0018 | px=(937,568)
shot=x.png ok=True
"""


def test_reads_the_newest_snapshot_not_the_first():
    # The screenshot is taken after the last sample, so an earlier block describes a different
    # moment. Reading the first one would compare the picture against stale positions and could
    # pass on a picture that never updated.
    got = parse_summary(SUMMARY)
    assert [agv for agv, _, _ in got] == ["AGV00", "AGV01", "AGV02"]
    assert got[0][1:] == (368.0, 683.0)
    assert got[2][1:] == (937.0, 568.0)


def test_a_dump_with_no_vehicles_yields_nothing():
    assert parse_summary("status=未连接\n") == []


def test_body_pixels_survive_erosion_but_label_text_does_not():
    # A body is a solid disc; the label beside it is one- and two-pixel strokes. Taking the raw
    # centroid would drag each vehicle towards its own label.
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (120, 80), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    green = PALETTE[0]
    draw.ellipse([20, 20, 32, 32], fill=green)          # a body
    draw.line([60, 20, 110, 20], fill=green, width=1)   # a one-pixel "label" stroke
    draw.line([60, 40, 110, 40], fill=green, width=1)

    found = body_centroids(img, [green])
    assert green in found
    cx, cy, n = found[green]
    assert abs(cx - 26.0) < 2.0, "centroid should be the body, not the body plus the strokes"
    assert abs(cy - 26.0) < 2.0
    assert n >= 4


def test_a_colour_used_only_for_text_is_not_a_vehicle():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (120, 80), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    purple = PALETTE[2]
    for y in (10, 20, 30, 40, 50, 60):
        draw.line([10, y, 100, y], fill=purple, width=1)
    assert body_centroids(img, [purple]) == {}


def test_recovers_a_translation_and_a_scale():
    src = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]
    dst = [(100.0, 50.0), (120.0, 50.0), (100.0, 70.0), (120.0, 70.0)]
    scale, offset, worst = fit_similarity(src, dst)
    assert scale == pytest.approx(2.0)
    assert worst < 1e-6


def test_a_rotated_or_wrong_layout_does_not_fit():
    src = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]
    # A rotation is not a rendering artefact: it means the picture is not the map.
    rotated = [(0.0, 0.0), (0.0, 10.0), (-10.0, 0.0), (-10.0, 10.0)]
    _, _, worst_rot = fit_similarity(src, rotated)
    assert worst_rot > 4.0

    # So is a vehicle drawn in the wrong place.
    swapped = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    _, _, worst_swap = fit_similarity(src, swapped)
    assert worst_swap > 4.0


def test_degenerate_input_does_not_divide_by_zero():
    scale, _, worst = fit_similarity([(1.0, 1.0)], [(5.0, 5.0)])
    assert scale == 1.0
    assert worst == float("inf")
