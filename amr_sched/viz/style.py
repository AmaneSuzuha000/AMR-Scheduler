"""Unified plotting style (fonts and color palette)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")           # headless backend, safe on servers and in batch jobs

import matplotlib.pyplot as plt
from matplotlib import font_manager

_CJK_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
    "Source Han Sans SC", "WenQuanYi Zen Hei", "DejaVu Sans",
]


def setup(font_size: float = 9.0) -> None:
    """Configure fonts and a compact layout. Falls back to DejaVu when no CJK font is
    available (glyphs render as boxes but nothing fails)."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [n for n in _CJK_CANDIDATES if n in available] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.sans-serif": chosen,
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


# Semantic colors: each node type is distinguishable at a glance
NODE_COLORS = {
    "aisle": "#9aa5b1",
    "disassembly": "#d64545",
    "workstation": "#2f6fb3",
    "assembly": "#2e8b57",
    "buffer": "#d99b1c",
    "dock": "#6a4c93",
}
NODE_LABELS = {
    "aisle": "Aisle node",
    "disassembly": "Disassembly area",
    "workstation": "Maintenance bay",
    "assembly": "Assembly area",
    "buffer": "Buffer area",
    "dock": "Dock",
}
AGV_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c",
              "#d62728", "#9467bd", "#8c564b",
              "#e377c2", "#7f7f7f"]
