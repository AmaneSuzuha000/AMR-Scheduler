"""Check that the scheduling window really drew what it says it drew.

A probe that only reads the window's own text proves the text updated, not that a picture
appeared: the overlay could be empty, off-screen, or painted in one colour, and the summary
would read exactly the same. This reads the screenshot back and requires every reported vehicle
to be present as a painted body, laid out the way the window says it is.

Two things make a naive pixel comparison wrong, so both are handled here.

The window is captured off-screen by PrintWindow, so image coordinates differ from canvas
coordinates by a translation that depends on window chrome, and by the display scale factor.
The check therefore fits a similarity transform (one uniform scale plus a translation, no
rotation -- a rotation would mean the picture is not the map) from the reported canvas pixels
to the observed bodies, and requires the residual to be small. A wrong projection, a frame
drawn at a stale scale, or vehicles drawn in the wrong places all break that fit, while window
chrome and DPI do not.

Bodies are told apart from label text by erosion. A body is a solid disc about ten pixels
across; the label beside it uses the same colour but is made of one- and two-pixel strokes,
which erosion removes. Taking the raw centroid of the colour instead would drag each vehicle
towards its own label.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

RESIDUAL_TOLERANCE_PX = 4.0
MIN_BODY_PIXELS = 4

# The palette the C# allocator hands out, in slot order.
PALETTE = [
    (0x2E, 0x7D, 0x32),
    (0x15, 0x65, 0xC0),
    (0x6A, 0x1B, 0x9A),
    (0xEF, 0x6C, 0x00),
    (0x00, 0x83, 0x8F),
    (0xAD, 0x14, 0x57),
    (0x55, 0x8B, 0x2F),
    (0x4E, 0x34, 0x2E),
]


def parse_summary(text: str) -> list[tuple[str, float, float]]:
    """Return the *newest* run of ``AGVxx | state | task | px=(x,y)`` lines in the dump.

    The probe samples the panel more than once and writes every sample, so the file holds
    several snapshots. Only the last one was on screen when the screenshot was taken; an
    earlier one describes a different moment. Picking the wrong block makes this check compare
    the picture against stale positions, which is exactly the failure it exists to catch.
    """
    blocks: list[list[tuple[str, float, float]]] = []
    current: list[tuple[str, float, float]] = []
    for line in text.splitlines():
        m = re.match(r"^(\S+)\s*\|.*px=\(([-\d.]+),([-\d.]+)\)\s*$", line.strip())
        if m:
            current.append((m.group(1), float(m.group(2)), float(m.group(3))))
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks[-1] if blocks else []


def _erode(mask, rounds: int = 2):
    """Drop pixels not surrounded on all four sides; repeated to erase thin label strokes."""
    core = mask
    for _ in range(rounds):
        if core.sum() == 0:
            return core
        shrunk = core.copy()
        shrunk[1:, :] &= core[:-1, :]
        shrunk[:-1, :] &= core[1:, :]
        shrunk[:, 1:] &= core[:, :-1]
        shrunk[:, :-1] &= core[:, 1:]
        core = shrunk
    return core


def body_centroids(image, palette, near: int = 40):
    """Centroid of each palette colour's solid body, ignoring same-coloured label text."""
    import numpy as np
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    found = {}
    for colour in palette:
        dist = np.abs(arr - np.array(colour, dtype=np.int16)).sum(axis=2)
        mask = dist <= near
        if mask.sum() < MIN_BODY_PIXELS:
            continue
        core = _erode(mask)
        if core.sum() < MIN_BODY_PIXELS:
            continue
        ys, xs = np.nonzero(core)
        found[colour] = (float(xs.mean()), float(ys.mean()), int(core.sum()))
    return found


def fit_similarity(src, dst):
    """Least-squares uniform scale + translation taking ``src`` onto ``dst``.

    Rotation is deliberately excluded: allowing it would let a picture that is rotated with
    respect to the map still pass, and a rotated map is a wrong drawing, not a rendering
    artefact.
    """
    import numpy as np
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    a = src - src_c
    b = dst - dst_c
    denom = float((a * a).sum())
    if denom <= 0:
        return 1.0, dst_c, float("inf")
    s = float((a * b).sum()) / denom
    t = dst_c - s * src_c
    residuals = np.abs(s * src + t - dst).max(axis=1)
    return s, t, float(residuals.max())


def check(summary_path: Path, image_path: Path, verbose: bool = True) -> int:
    from PIL import Image

    text = summary_path.read_text(encoding="utf-8")
    # A frame that landed between reading the numbers and capturing the window makes the two
    # describe different instants. At 7.32 px/m a vehicle covers about 13 px in a second, which
    # is far outside the residual tolerance, so an unstable capture fails here rather than
    # being reported as a drawing error it is not.
    if "capture_stable=" in text:
        stable = re.search(r"^capture_stable=(\w+)$", text, re.MULTILINE)
        if not stable or stable.group(1).lower() != "true":
            print("FAIL: the screenshot and the reported positions were not taken from the "
                  "same frame (capture_stable is false); rerun the probe")
            return 1
    reported = parse_summary(text)
    if len(reported) < 3:
        print("FAIL: the window reported fewer than three placed vehicles; "
              "a layout cannot be checked against one point")
        print(summary_path.read_text(encoding="utf-8"))
        return 1

    image = Image.open(image_path)
    centroids = body_centroids(image, PALETTE)
    if verbose:
        print(f"image {image.size}, reported {len(reported)} vehicles, "
              f"painted bodies found: {len(centroids)}")
        for colour, (cx, cy, n) in sorted(centroids.items()):
            print(f"  #{colour[0]:02X}{colour[1]:02X}{colour[2]:02X} body=({cx:.1f},{cy:.1f}) px={n}")
        for agv, x, y in reported:
            print(f"  reported {agv} px=({x:.1f},{y:.1f})")

    # Slot order is assignment order: the first vehicle in the frame takes the first colour.
    # Distinct colours in the image must number at least as many as reported vehicles, or the
    # overlay did not paint every one of them.
    if len(centroids) < len(reported):
        print(f"FAIL: {len(reported)} vehicles were reported but only {len(centroids)} are "
              "painted as solid bodies; the overlay is not drawing every vehicle")
        return 1
    if len(centroids) != len(reported):
        print(f"FAIL: {len(centroids)} painted bodies for {len(reported)} reported vehicles; "
              "the window and the picture disagree on the size of the fleet")
        return 1

    # Every body is the same 10x10 ellipse in the same colour, so every body must cover about
    # the same number of pixels. A body that is half covered by something -- a label's opaque
    # background, another panel -- still has a centroid in the right place and still forms a
    # solid blob, so position alone cannot see it. Measured once: a vehicle near the right edge
    # came out at 94 px while its neighbours were at 183, and the position check passed.
    counts = [n for _, _, n in centroids.values()]
    if counts and min(counts) < 0.75 * max(counts):
        print(f"FAIL: painted bodies differ in size ({sorted(counts)} px); every vehicle is the "
              "same shape, so one of them is partly covered by something drawn over it")
        return 1

    # Pair the i-th reported vehicle with the i-th palette slot. Sorting the colours instead
    # would order them by their byte values, not by slot, and silently compare a vehicle's
    # position against a different vehicle's body.
    ordered = []
    for slot, colour in enumerate(PALETTE):
        if colour in centroids:
            ordered.append(centroids[colour])
    if len(ordered) != len(reported):
        print(f"FAIL: {len(ordered)} palette slots in use for {len(reported)} reported vehicles; ",
              "the colour-to-slot pairing cannot be made")
        return 1
    src = [(x, y) for _, x, y in reported]
    dst = [(c[0], c[1]) for c in ordered]
    scale, offset, worst = fit_similarity(src, dst)
    print(f"fitted scale={scale:.4f}, offset=({offset[0]:.1f},{offset[1]:.1f}), "
          f"worst residual={worst:.2f} px")
    if worst > RESIDUAL_TOLERANCE_PX:
        print(f"FAIL: the painted bodies are up to {worst:.1f} px away from a similarity "
              f"transform of the reported positions (tolerance {RESIDUAL_TOLERANCE_PX} px); "
              "the picture does not match the numbers the window reports")
        return 1

    print(f"PASS: {len(centroids)} vehicles painted, layout matches the reported pixels "
          f"(worst residual {worst:.2f} px, display scale {scale:.3f})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path,
                    default=Path("docs/证据/T5.3-schedule-stream/ui_status.txt"))
    ap.add_argument("--image", type=Path,
                    default=Path("docs/证据/T5.3-schedule-stream/gui_schedule_stream.png"))
    a = ap.parse_args(argv)
    if not a.summary.exists():
        print(f"FAIL: no status dump at {a.summary}; run scripts/schedule_stream_probe.ps1 first")
        return 1
    if not a.image.exists():
        print(f"FAIL: no screenshot at {a.image}; run scripts/schedule_stream_probe.ps1 first")
        return 1
    return check(a.summary, a.image)


if __name__ == "__main__":
    raise SystemExit(main())
