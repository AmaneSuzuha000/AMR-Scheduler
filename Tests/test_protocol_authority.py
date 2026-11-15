"""The edge-interpolation rule must exist exactly once, and both consumers must use it.

This module used to pin *two* implementations to each other: `LiveClient.PoseToXY` in
interop/csharp and `FleetTopology.Place` in RAWSimO.Live each clamped progress and
interpolated along an edge. They agreed on the day they were written; nothing kept them
agreeing, and a change to one would have left the other drawing vehicles somewhere the
tests never looked.

The rule now lives in `interop/csharp/PoseMath.cs`. That one file is compiled twice --
by the .NET Framework `csc.exe` for the standalone probe, and as a linked file inside
RAWSimO.Live -- so there is one body of source and no second copy to drift.

These tests do not compile C#. They check the property that makes the merge hold:
the rule is written in exactly one file, both consumers call it, both build paths list
it, and the documents say so. If someone re-derives the clamp locally, this goes red.
"""
from __future__ import annotations

import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SHARED = "interop/csharp/PoseMath.cs"
CONSUMERS = ("interop/csharp/LiveClient.cs",
             "RAWSimO.Live/Tracking/FleetTopology.cs")


def _read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def _csharp_files():
    """Every C# file this project owns (the vendored RAWSimO tree is not ours)."""
    out = []
    for folder in ("interop/csharp", "RAWSimO.Live"):
        for dirpath, _dirs, names in os.walk(os.path.join(ROOT, folder)):
            for name in names:
                if name.endswith(".cs"):
                    out.append(os.path.relpath(
                        os.path.join(dirpath, name), ROOT).replace(os.sep, "/"))
    return sorted(out)


def test_the_rule_is_defined_in_exactly_one_file():
    shared = _read(SHARED)
    assert "public static double ClampProgress" in shared
    assert "public static void Interpolate" in shared
    assert "public static bool TrySplitEdge" in shared

    for rel in CONSUMERS:
        text = _read(rel)
        assert "PoseMath." in text, (
            "%s does not call the shared rule; it is re-deriving it" % rel)
        assert "public static double ClampProgress" not in text, (
            "%s defines a second clamp" % rel)
        assert "public static void Interpolate" not in text, (
            "%s defines a second interpolation" % rel)


def test_both_build_paths_compile_the_same_file():
    project = _read("RAWSimO.Live/RAWSimO.Live.csproj")
    assert "interop\\csharp\\PoseMath.cs" in project, (
        "RAWSimO.Live no longer links the shared file, so it must have its own copy")

    live = _read("interop/LIVE.md")
    csc_line = [ln for ln in live.splitlines() if "LiveProbe.cs" in ln]
    assert csc_line, "interop/LIVE.md lost the csc command that builds the probe"
    assert "PoseMath.cs" in csc_line[0], (
        "the documented csc command does not compile PoseMath.cs, so the probe would "
        "not link")


def test_the_producer_writes_the_separator_both_consumers_expect():
    producer = _read("amr_sched/stream.py")
    assert '"->"' in producer, "the producer stopped writing the arrow separator"

    shared = _read(SHARED)
    assert 'EdgeSeparator = "->"' in shared, (
        "the separator is defined somewhere else again")
    for rel in CONSUMERS:
        assert 'src + "->"' not in _read(rel), (
            "%s still builds an edge id from a hard-coded separator" % rel)


def test_the_interpolation_formula_is_written_once():
    owners = [rel for rel in _csharp_files()
              if "(bx - ax) * p" in _read(rel) or "(b.X - a.X) * p" in _read(rel)]
    assert owners == [SHARED], (
        "the interpolation formula appears in %s; it must appear only in %s"
        % (owners or "no file", SHARED))


def test_the_clamp_is_written_once():
    shared = _read(SHARED)
    assert "if (progress < 0.0) return 0.0;" in shared
    assert "if (progress > 1.0) return 1.0;" in shared

    for rel in _csharp_files():
        if rel == SHARED:
            continue
        text = _read(rel)
        assert "pose.Progress < 0.0 ? 0.0" not in text, (
            "%s clamps progress on its own" % rel)
        assert "if (p < 0) p = 0;" not in text, (
            "%s clamps progress on its own" % rel)


def test_prose_documents_name_their_source():
    protocols = _read("PROTOCOLS.md")
    assert "PoseMath.cs" in protocols, "PROTOCOLS.md does not name the shared file"
    assert "One body of source" in protocols
    assert "They agree today --" not in protocols, (
        "PROTOCOLS.md still describes the retired two-implementation state")

    live = _read("interop/LIVE.md")
    assert "PoseMath.cs" in live, "interop/LIVE.md does not mention the shared file"

def test_the_shared_file_stays_csharp5_compatible():
    """PoseMath.cs is compiled by the .NET Framework 4.0 compiler as well.

    That compiler predates C# 6. Nothing in the SDK build complains when the shared file
    starts using newer syntax -- only the interop probe's build breaks, and that build is
    run by hand. So the constraint is checked here, where it is cheap.
    """
    shared = _read(SHARED)
    code = "\n".join(line.split("//")[0] for line in shared.splitlines())

    forbidden = [
        ("=>", "expression-bodied members need C# 6"),
        ('$"', "string interpolation needs C# 6"),
        ("nameof(", "nameof needs C# 6"),
        ("out var", "out-var needs C# 7"),
        ("string?", "nullable reference types need C# 8"),
    ]
    hits = ["%s (%s)" % (token, why) for token, why in forbidden if token in code]
    assert not hits, "PoseMath.cs no longer compiles with the .NET Framework 4.0 csc:\n" + \
        "\n".join("  " + hit for hit in hits)
