"""Render-quality profiles: the lighting and colour-management rig for all renders.

The scene file stores geometry and cameras, but the *look* of a render is decided by
lighting and colour management.  Leaving those to whatever defaults the current
Blender version ships means the visual baseline silently changes when Blender is
upgraded, and it means "the model looks wrong" can actually mean "the render is
flat".  Measured on 2026-11-11: our renders had mean tonal standard deviation 0.123
against 0.193 in the reference frames - a 1.57x contrast deficit - while the mean
brightness was already correct.

Each profile below is an explicit, named set of values so a comparison can say
*which* rig produced which score.  Profiles never touch geometry.
"""
from __future__ import annotations

import math
import os

# Sun direction is the same in every profile so that only intensity/contrast change.
SUN_ROTATION_DEG = (25.0, -30.0, -25.0)


PROFILES = {
    # What the repository used before this module existed: one sun, bright ambient,
    # Standard view transform. Kept so old previews stay reproducible.
    "legacy": {
        "description": "Original rig: sun 2.2, world 0.7, Standard transform.",
        "sun_energy": 2.2,
        "sun_angle_deg": 0.12 * 180.0 / math.pi,
        "world_strength": 0.7,
        "world_color": (0.55, 0.59, 0.65),
        "view_transform": "Standard",
        "look": "None",
        "exposure": 0.0,
    },
    # Deeper ambient, sharper sun, AgX tone mapping. Aimed squarely at the measured
    # contrast deficit without changing mean brightness much.
    # exposure was +0.15 until the sweep on 2026-11-11. Ranked by the Wasserstein tonal
    # distance over all 29 views, the worst-scoring views shared one signature: our
    # render brighter than the reference by 0.07-0.11 in mean grey, with less dark area
    # (press_front 0.57 against 0.47, closing_wide 0.59 against 0.49). Sweeping exposure
    # alone, geometry untouched:
    #     +0.15 -> w1 0.0806   (what the repo used)
    #      0.00 -> w1 0.0686
    #     -0.15 -> w1 0.0591   25 of 29 views improve, 2 regress
    #     -0.30 -> w1 0.0586   mean marginally lower but median and dark error worse
    #     -0.45 -> w1 0.0688   over-corrected
    # -0.15 is the minimum on the median and keeps the dark-share error smallest.
    "contrast": {
        "description": "Sun 3.4 tight, world 0.30, AgX Medium High Contrast, exposure -0.15.",
        "sun_energy": 3.4,
        "sun_angle_deg": 3.0,
        "world_strength": 0.30,
        "world_color": (0.52, 0.56, 0.63),
        "view_transform": "AgX",
        "look": "AgX - Medium High Contrast",
        "exposure": -0.15,
    },
    # Stronger variant: tighter sun, darker ambient, AgX "Very High Contrast".
    # Added after "contrast" recovered only ~62% of the measured contrast gap.
    "contrast_strong": {
        "description": "Sun 4.2 tight, world 0.18, AgX Very High Contrast, exposure 0.25.",
        "sun_energy": 4.2,
        "sun_angle_deg": 1.5,
        "world_strength": 0.18,
        "world_color": (0.50, 0.54, 0.62),
        "view_transform": "AgX",
        "look": "AgX - Very High Contrast",
        "exposure": -0.05,   # was +0.25; shifted with "contrast" to keep the same bias
    },
    # "contrast" fixes the average tonal spread but leaves a per-view split: some views
    # end up far darker than the reference (run_in dark share 0.316 against 0.158) and
    # some stay flat (storage 0.140 against 0.166). Both point at the same cause - a sun
    # disc tight enough (3 degrees) to cast hard, localised shadow. These two soften the
    # disc so shadow spreads instead of concentrating, and lift ambient slightly to keep
    # total exposure where "contrast" had it.
    "contrast_soft": {
        "description": "Sun 3.2 with a 12 degree disc, world 0.34, AgX Medium High Contrast.",
        "sun_energy": 3.2,
        "sun_angle_deg": 12.0,
        "world_strength": 0.34,
        "world_color": (0.52, 0.56, 0.63),
        "view_transform": "AgX",
        "look": "AgX - Medium High Contrast",
        "exposure": -0.12,   # was +0.18; shifted with "contrast"
    },
    "contrast_vsoft": {
        "description": "Sun 2.9 with a 25 degree disc (overcast-like), world 0.40, AgX Medium High Contrast.",
        "sun_energy": 2.9,
        "sun_angle_deg": 25.0,
        "world_strength": 0.40,
        "world_color": (0.54, 0.58, 0.64),
        "view_transform": "AgX",
        "look": "AgX - Medium High Contrast",
        "exposure": -0.10,   # was +0.20; shifted with "contrast"
    },
    # Same as contrast but with a fill light from the opposite side, standing in for
    # bounce off the hall floor. Tests whether shadows alone were the problem.
    "contrast_fill": {
        "description": "contrast + 0.9 sun fill from the opposite azimuth.",
        "sun_energy": 3.4,
        "sun_angle_deg": 3.0,
        "world_strength": 0.22,
        "world_color": (0.52, 0.56, 0.63),
        "view_transform": "AgX",
        "look": "AgX - Medium High Contrast",
        "exposure": 0.2,
        "fill_energy": 0.9,
    },
}


def available() -> list[str]:
    return sorted(PROFILES)


def _set_view_transform(scene, transform: str, look: str, exposure: float) -> dict:
    """Apply colour management, reporting what the build actually accepted."""
    applied = {"requested_transform": transform, "requested_look": look, "exposure": exposure}
    view = scene.view_settings
    try:
        view.view_transform = transform
    except Exception as error:
        applied["transform_error"] = str(error)[:120]
        return applied
    try:
        view.look = look
    except Exception as error:
        # 'look' names differ between Blender versions; record the miss rather than
        # pretending the look was applied.
        applied["look_error"] = str(error)[:160]
    try:
        view.exposure = exposure
    except Exception as error:
        applied["exposure_error"] = str(error)[:120]
    applied["view_transform"] = view.view_transform
    applied["look"] = view.look
    return applied


def apply_profile(scene, name: str) -> dict:
    """Apply a named profile to an open scene and return what was actually set."""
    import bpy

    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; known: {', '.join(available())}")
    profile = PROFILES[name]

    suns = [o for o in bpy.data.objects if o.type == "LIGHT" and o.data.type == "SUN"]
    if suns:
        sun = suns[0]
    else:
        data = bpy.data.lights.new("Sun", "SUN")
        sun = bpy.data.objects.new("Sun", data)
        scene.collection.objects.link(sun)
    sun.data.energy = profile["sun_energy"]
    sun.data.angle = math.radians(profile["sun_angle_deg"])
    # Sun geometry override. Measured reason: the shipped rotation (25, -30, -25) puts the
    # sun 51.7 degrees above the horizon, so the vertical faces receive cos 0.589 of the
    # direct light while the floor receives cos 0.785 - the two differ by only 25%. The
    # reference separates them by a factor of three (dark surfaces 0.21 against floor
    # 0.61), which no material, albedo or tone-curve change can reproduce while both
    # surfaces are lit almost equally. A steeper sun is the only lever that widens that
    # ratio, because cos falls off for vertical faces while the floor approaches cos 1.
    sun_rotation = SUN_ROTATION_DEG
    raw_rotation = os.environ.get("WS_SUN_ROTATION", "").strip()
    if raw_rotation:
        try:
            parsed = tuple(float(v) for v in raw_rotation.split(","))
            if len(parsed) == 3:
                sun_rotation = parsed
        except ValueError:
            pass
    sun.rotation_euler = tuple(math.radians(v) for v in sun_rotation)

    # Environment overrides, for A/B sweeps that must not edit the named profile. A
    # measured claim drives this: over all 29 views the reference's dark surfaces come
    # back from our render 0.272 grey too bright, but darkening the wall albedo by more
    # than half moved that only 5%, so albedo is not the cause. Ambient fill on every
    # shaded face is the remaining candidate - and unlike the palette it has to be
    # measured, because raising the sun instead would also brighten the floor, which
    # already matches (0.611 against 0.635).
    overrides = {}
    for key, env, cast in (
        ("sun_energy", "WS_SUN_ENERGY", float),
        ("world_strength", "WS_WORLD_STRENGTH", float),
    ):
        raw = os.environ.get(env, "")
        if raw.strip():
            try:
                setattr_value = cast(raw)
            except ValueError:
                overrides[f"{key}_error"] = f"unparsable {raw!r}"
                continue
            if key == "sun_energy":
                sun.data.energy = setattr_value
            overrides[key] = setattr_value

    fill = None
    if profile.get("fill_energy"):
        existing = bpy.data.objects.get("FillSun")
        if existing is None:
            data = bpy.data.lights.new("FillSun", "SUN")
            fill = bpy.data.objects.new("FillSun", data)
            scene.collection.objects.link(fill)
        else:
            fill = existing
        fill.data.energy = profile["fill_energy"]
        fill.data.angle = math.radians(20.0)
        fill.rotation_euler = (math.radians(-20.0), math.radians(25.0), math.radians(150.0))

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    world_strength = overrides.get("world_strength", profile["world_strength"])
    if os.environ.get("WS_WORLD_STRENGTH", "").strip():
        try:
            world_strength = float(os.environ["WS_WORLD_STRENGTH"])
        except ValueError:
            pass
    if background is not None:
        background.inputs["Color"].default_value = (*profile["world_color"], 1.0)
        background.inputs["Strength"].default_value = world_strength

    # Colour-management overrides. Measured reason: pooled over 29 views the reference
    # spans p5 0.192 to p99 0.944 while our render spans 0.226 to 0.745 - a tonal range
    # of 0.807x the reference, compressed at BOTH ends (darks float 0.064 too high,
    # highlights sit 0.198 too low). That is the signature of a tone curve, not of the
    # lighting: darkening the world instead moved the darks down but dragged the already
    # too-dark highlights down further (floor delta -0.083 -> -0.171).
    view_transform = os.environ.get("WS_VIEW_TRANSFORM", "").strip() or profile["view_transform"]
    look = os.environ.get("WS_LOOK", "").strip() or profile["look"]
    if "WS_EXPOSURE" in os.environ and os.environ["WS_EXPOSURE"].strip():
        try:
            exposure = float(os.environ["WS_EXPOSURE"])
        except ValueError:
            exposure = profile["exposure"]
    else:
        exposure = profile["exposure"]
    colour = _set_view_transform(scene, view_transform, look, exposure)
    return {
        "profile": name,
        "description": profile["description"],
        "sun_energy": profile["sun_energy"],
        "sun_angle_deg": profile["sun_angle_deg"],
        "world_strength": profile["world_strength"],
        "fill_energy": profile.get("fill_energy"),
        "colour_management": colour,
    }
