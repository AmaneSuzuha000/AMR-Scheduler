"""Refuse to score a render set that did not actually render the scene.

This exists because of a real failure on 2026-11-11: an A/B sweep rendered five variants
whose render script never opened the .blend, so all five produced the default empty
viewport - uniform grey, no geometry. The scoring step then happily reported five identical
scores. Identical scores across variants that differ is the kind of result that looks like
a decisive finding ("the change does nothing!") and is really a broken pipeline. It cost a
full round and nearly produced a wrong conclusion about the reference.

The check is deliberately blunt: every view must have a plausible amount of foreground and
a plausible number of distinct tones. A blank render has one tone and near-zero foreground.

Exits non-zero when a set is unusable, so it can be put in front of any scoring step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

# Deliberately loose bounds. Real views in this suite run 0.10-0.55 silhouette coverage and
# hundreds of grey levels; a broken render sits at 0.000 coverage and 1-3 levels.
MIN_FOREGROUND = 0.02
MIN_GREY_LEVELS = 40
MIN_MEAN_SPREAD = 0.004


def inspect_set(directory: Path, kind: str) -> dict:
    files = sorted(directory.glob("*.png"))
    problems = []
    stats = {}
    for path in files:
        if kind == "mask":
            rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64) / 255.0
            foreground = float((rgba[:, :, 3] > 0.5).mean())
            grey = rgba[:, :, :3] @ np.array([0.299, 0.587, 0.114])
        else:
            rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
            border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
            background = np.median(border, axis=0)
            foreground = float((np.abs(rgb - background).max(axis=2) > 0.08).mean())
            grey = rgb @ np.array([0.299, 0.587, 0.114])
        levels = int(len(np.unique(np.round(grey, 3))))
        spread = float(grey.std())
        stats[path.name] = {"foreground": round(foreground, 4),
                            "grey_levels": levels,
                            "grey_std": round(spread, 4)}
        if foreground < MIN_FOREGROUND:
            problems.append(f"{path.name}: foreground {foreground:.4f} < {MIN_FOREGROUND}")
        if levels < MIN_GREY_LEVELS:
            problems.append(f"{path.name}: only {levels} grey levels")
        if spread < MIN_MEAN_SPREAD:
            problems.append(f"{path.name}: grey std {spread:.4f} is flat")
    return {"directory": str(directory), "kind": kind, "files": len(files),
            "problems": problems, "per_file": stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", action="append", default=[],
                        help="mask:<dir> or shaded:<dir>; repeatable")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if not args.set:
        parser.error("give at least one --set mask:<dir> or shaded:<dir>")

    results = []
    failed = False
    for spec in args.set:
        kind, _, directory = spec.partition(":")
        path = ROOT / directory
        if not path.is_dir():
            print(f"MISSING {spec}")
            failed = True
            continue
        result = inspect_set(path, kind)
        results.append(result)
        status = "OK" if not result["problems"] else "UNUSABLE"
        if result["problems"]:
            failed = True
        print(f"{status:9s} {spec}  files={result['files']}")
        for problem in result["problems"][:5]:
            print(f"           {problem}")
        if len(result["problems"]) > 5:
            print(f"           ... and {len(result['problems']) - 5} more")

    if args.output:
        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"sets": results, "usable": not failed},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
    if failed:
        print("\nREFUSING: this render set cannot support a scoring claim.")
        sys.exit(1)


if __name__ == "__main__":
    main()
