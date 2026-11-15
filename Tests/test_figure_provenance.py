"""Every figure the thesis embeds must have a script that can produce it.

Two figures -- metrics.png (thesis 5-6) and planner_cmp.png (thesis 6-1) -- were committed
with no generator anywhere in the repository. They were displayed in the thesis and in the
defence slides as evidence, and nobody could re-derive them from the data. Their numbers did
turn out to agree with results/baseline.csv, but that was luck plus a manual check, not a
property of the repository.

That is the failure these tests exist to prevent: a figure is evidence only if the path from
CSV to pixels exists in code.
"""
from __future__ import annotations

import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(ROOT, "results", "figures")
MD = os.path.join(ROOT, "docs", "论文", "论文-主文档.md")


def _referenced_figures():
    """Image filenames the thesis embeds, as bare names."""
    if not os.path.exists(MD):
        pytest.skip("thesis source is not in this checkout")
    with open(MD, encoding="utf-8") as f:
        text = f.read()
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    return [os.path.basename(r) for r in refs]


def _generators():
    """All text that could draw a figure.

    amr_sched/viz is included because some figures are produced there and called from a
    driver script, so the filename appears as an argument rather than as a literal next to
    savefig. Searching only experiments/ and tools/ reported those figures as orphans when
    they had a perfectly good generator.
    """
    out = []
    for sub in ("experiments", "tools", os.path.join("amr_sched", "viz")):
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(d, name), encoding="utf-8", errors="replace") as f:
                out.append((name, f.read()))
    return out


def test_every_thesis_figure_has_a_generator():
    """The whole point: no figure may be evidence that no code can reproduce."""
    gens = _generators()
    orphans = []
    for fig in _referenced_figures():
        if not fig.endswith((".png", ".gif")):
            continue
        if not any(fig in text for _, text in gens):
            orphans.append(fig)
    assert not orphans, (
        "these figures are shown in the thesis but no script writes them: %s" % orphans)


def test_the_generator_search_covers_where_figures_are_actually_drawn():
    """A search that misses the drawing module reports live figures as orphans.

    agv_states.png and task_progress.png are written by amr_sched/viz/animation.py and merely
    called from a driver, so a search limited to experiments/ and tools/ called them orphans.
    Guarding the guard, because a provenance test that cries wolf gets switched off.
    """
    subs = {name for name, _ in _generators()}
    assert "animation.py" in subs, (
        "the viz module draws figures and must be searched")


def test_every_thesis_figure_exists_on_disk():
    missing = [f for f in _referenced_figures()
               if not os.path.exists(os.path.join(FIGURES, f))]
    assert not missing, "thesis refers to figures that do not exist: %s" % missing


def test_the_metrics_figure_states_its_actual_seed_count():
    """The original figure claimed ten seeds while baseline.csv has five per cell.

    A figure that misstates its own sample size is worse than one with no caption, because
    the number looks authoritative. The generator must read the count from the data.
    """
    path = os.path.join(ROOT, "experiments", "plot_metrics.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert "mean over seeds" in src
    # No hardcoded seed count may appear in the title text.
    assert not re.search(r"\b(5|10|20) seeds", src), (
        "the seed count must come from the data, not be written into the title")


def test_the_planner_figure_reads_its_seed_count_from_the_csv():
    """A hardcoded seed count would be wrong the moment the seed set changes."""
    path = os.path.join(ROOT, "experiments", "plot_planner_cmp.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert "len(sub)" in src, (
        "the figure must derive its sample size from the rows it plotted")


def test_error_bars_are_a_sample_standard_deviation():
    """stdev, not pstdev: ten seeds are a sample of the seed population, not all of it.

    The difference is a factor of sqrt(n/(n-1)), which for small n is exactly the sort of
    quiet understatement that makes a result look tighter than it is.
    """
    path = os.path.join(ROOT, "experiments", "plot_planner_cmp.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert "st.stdev(" in src
    assert "st.pstdev(" not in src


def test_the_metrics_figure_reports_the_arm_it_drew():
    """It must not silently switch to whichever planner looks better."""
    path = os.path.join(ROOT, "experiments", "plot_metrics.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert re.search(r'^STRATEGY = "spacetime\+greedy"$', src, re.M), (
        "the arm is part of what the figure claims; it must be a named constant")
