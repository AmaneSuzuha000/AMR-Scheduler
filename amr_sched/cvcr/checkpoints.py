"""Default inference checkpoint resolution helpers for pure-NumPy CVCR policies.

Resolves packaged pure-NumPy `.npz` weights under ``amr_sched/checkpoints/``
first so clean source checkouts and CI environments load trained weights without
local training artifacts, falling back to ``_scratch/checkpoints/`` during
active training sweeps.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_CKPT_DIR = _REPO_ROOT / "amr_sched" / "checkpoints"
_SCRATCH_CKPT_DIR = _REPO_ROOT / "_scratch" / "checkpoints"


def resolve_default_checkpoint(filename: str) -> Path:
    """Return tracked checkpoint path if present, else local scratch path."""
    pkg_path = _PACKAGE_CKPT_DIR / filename
    if pkg_path.exists():
        return pkg_path
    scratch_path = _SCRATCH_CKPT_DIR / filename
    if scratch_path.exists():
        return scratch_path
    return pkg_path
