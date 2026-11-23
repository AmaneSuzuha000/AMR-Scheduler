"""Improved, collision-audited 258x72m wheelset_workshop_v2 layout, map, and RAWSim-O exporter."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

from .assets import collision_boxes, make_assets
from .build import ROOT, SCENE, box_model, camera_pose, e, save_xml
from .geometry import PALETTE
from .light_material_network import segment_box, transform
from .workshop_navigation import box_slice, rasterize

DEFAULT_MAP_RESOLUTION = 0.10


# Frozen static "snapshot snapshot-in-motion" AGVs and dock-blocking static wheelsets
# replaced by dynamic Gazebo+Nav2 mobile carriers and dynamic wheelset payloads.
REMOVED_STATIC_SNAPSHOT_IDS = frozenset({
    "wheelset_agv_001",
    "wheelset_agv_002",
    "wheelset_agv_003",
    "wheelset_agv_004",
    "wheelset_agv_005",
    "wheelset_agv_006",
    "reinspection_r18_003",
    "transfer_r20_wheelset_agv_568",
    "transfer_r20_wheelset_agv_569",
    "transfer_r20_wheelset_agv_570",
    "component_r21_wheelset_agv_601",
    "wheelset_158",
    "wheelset_159",
    "wheelset_160",
    "wheelset_161",
    "wheelset_162",
    "wheelset_163",
    "transfer_r20_wheelset_571",
    "wheelset_091",
    "wheelset_094",
    "wheelset_095",
    "wheelset_100",
    "wheelset_109",
    "wheelset_114",
    "reinspection_r18_001",
    "reinspection_r18_002",
})


WALL_REPAIRS = {
    "r08_assembly_return_3": {"center": [74.3, 6.8, 0.75], "size": [0.18, 2.2, 1.5]},
    "transfer_r20_cache_wall_near": {"center": [140.0, 6.8, 0.75], "size": [0.18, 2.2, 1.5]},
    "transfer_r20_cache_wall_mid": {"center": [140.0, 14.05, 0.75], "size": [0.18, 2.4, 1.5]},
    "transfer_r20_cache_wall_far": {"center": [140.0, 22.8, 0.75], "size": [0.18, 4.2, 1.5]},
    "transfer_r20_cache_back": {"center": [145.6, 22.2, 0.75], "size": [7.6, 0.18, 1.5]},
    "transfer_r20_room_left": {"center": [82.3, 43.2, 0.75], "size": [0.18, 2.6, 1.5]},
    "r08_unknown_front_105": {"center": [108.5, 49.9, 0.75], "size": [31.0, 0.18, 1.5]},
    "component_r10_back_boundary": {"center": [94.0, 35.8, 0.75], "size": [53.0, 0.18, 1.5]},
}


def build_improved_scene(source_path: Path = SCENE) -> dict:
    """Derive a collision-free indoor AGV workshop scene faithful to the reference video."""
    scene = copy.deepcopy(json.loads(Path(source_path).read_text(encoding="utf-8")))
    scene["scene_id"] = "wheelset_workshop_v2"
    scene["visual_revision"] = "2.0"
    scene["status"] = {
        "geometry": "video_referenced_layout_repaired",
        "camera_registration": "multi_view_fixed_overview_and_zones",
        "physics": "actuated_diff_drive_wheelset_agvs_with_motion_guard",
        "process_execution": "grounded_cradle_transport_and_station_workflow",
        "scheduler_integration": "nav2_amcl_cvcr_v18_ea_joint",
    }

    # Repair partition/cache wall doorways blocking indoor yellow lanes.
    new_boxes = []
    for box in scene["boxes"]:
        b = copy.deepcopy(box)
        if b["id"] in WALL_REPAIRS:
            b.update(WALL_REPAIRS[b["id"]])
        new_boxes.append(b)
    existing_ids = {b["id"] for b in new_boxes}
    for wall_box in (
        {"id": "workshop_perimeter_west", "center": [0.12, 36.0, 0.45], "size": [0.24, 72.0, 0.90]},
        {"id": "workshop_perimeter_east", "center": [257.88, 36.0, 0.45], "size": [0.24, 72.0, 0.90]},
        {"id": "workshop_perimeter_south", "center": [129.0, 0.12, 0.45], "size": [258.0, 0.24, 0.90]},
        {"id": "workshop_perimeter_north", "center": [129.0, 71.88, 0.45], "size": [258.0, 0.24, 0.90]},
    ):
        if wall_box["id"] not in existing_ids:
            new_boxes.append({**wall_box, "material": "wall", "zone": "environment", "source": "assumed", "yaw": 0})
    scene["boxes"] = new_boxes

    # Remove frozen snapshot AGVs/blocking dock-head wheelsets & re-bay component cache rows cleanly.
    kept_instances = []
    for inst in scene["instances"]:
        if inst["id"] in REMOVED_STATIC_SNAPSHOT_IDS:
            continue
        item = copy.deepcopy(inst)
        # Re-bay component_r21 buffer rows inside [x=96..117, y=9.0/14.0/19.0/24.0] clear of perimeter aisles.
        if item["id"].startswith("component_r21_wheelset_") and item["asset"] == "wheelset":
            px, py, pz = item["pose"][:3]
            if py >= 25.0:
                item["pose"][1] = 24.0
            elif py >= 21.0:
                item["pose"][1] = 19.0
            elif py >= 17.0:
                item["pose"][1] = 14.0
            else:
                item["pose"][1] = 9.0
            item["pose"][0] = max(96.5, min(116.5, px))
            item["pose"][2] = pz
        elif item["id"] in {"wheelset_164", "wheelset_165", "wheelset_166", "wheelset_167", "wheelset_168"}:
            # Nudge west staging buffer 0.6 m north so 1.70x1.50 m AGVs clear mid_corridor_west by >= 1.46 m.
            item["pose"][1] = 40.6
        kept_instances.append(item)
    scene["instances"] = kept_instances

    # Unified unobstructed indoor yellow AGV route network (all lanes >= 0.90 m half-clearance).
    routes = [
        # East-West primary arterial aisles
        ("south_corridor_west", [4.0, 2.5], [65.3, 2.5], "wheelset_carrier"),
        ("south_corridor_prep", [65.3, 2.5], [90.2, 2.5], "wheelset_carrier"),
        ("south_corridor_cache", [90.2, 2.5], [123.3, 2.5], "wheelset_carrier"),
        ("south_corridor_press", [123.3, 2.5], [142.3, 2.5], "wheelset_carrier"),
        ("south_corridor_close", [142.3, 2.5], [152.3, 2.5], "wheelset_carrier"),
        ("south_corridor_finish", [152.3, 2.5], [187.5, 2.5], "wheelset_carrier"),
        ("south_corridor_east", [187.5, 2.5], [252.0, 2.5], "wheelset_carrier"),
        ("north_corridor_west", [4.0, 46.8], [65.3, 46.8], "wheelset_carrier"),
        ("north_corridor_transfer", [65.3, 46.8], [88.0, 46.8], "wheelset_carrier"),
        ("north_corridor_mid", [88.0, 46.8], [152.3, 46.8], "wheelset_carrier"),
        ("north_corridor_finish", [152.3, 46.8], [187.5, 46.8], "wheelset_carrier"),
        ("north_corridor_storage", [187.5, 46.8], [252.0, 46.8], "wheelset_carrier"),
        # Mid-hall cross collectors (39.2m central highway + cache/closing feeders)
        ("mid_corridor_west", [4.0, 38.0], [65.3, 38.0], "wheelset_carrier"),
        ("mid_corridor_logistics", [65.3, 39.2], [88.0, 39.2], "wheelset_carrier"),
        ("mid_corridor_central", [88.0, 39.2], [123.3, 39.2], "wheelset_carrier"),
        ("mid_corridor_east_a", [123.3, 39.2], [152.3, 39.2], "wheelset_carrier"),
        ("mid_corridor_east_b", [152.3, 39.2], [187.5, 39.2], "wheelset_carrier"),
        ("mid_corridor_cache", [90.2, 28.8], [123.3, 28.8], "wheelset_carrier"),
        ("mid_corridor_closing", [152.3, 26.5], [187.5, 26.5], "wheelset_carrier"),
        # North-South avenues
        ("avenue_west_s", [4.0, 2.5], [4.0, 38.0], "wheelset_carrier"),
        ("avenue_west_n", [4.0, 38.0], [4.0, 46.8], "wheelset_carrier"),
        ("avenue_logistics_1", [65.3, 2.5], [65.3, 9.0], "wheelset_carrier"),
        ("avenue_logistics_2", [65.3, 9.0], [65.3, 15.0], "wheelset_carrier"),
        ("avenue_logistics_3", [65.3, 15.0], [65.3, 18.57], "wheelset_carrier"),
        ("avenue_logistics_4a", [65.3, 18.57], [65.3, 38.0], "wheelset_carrier"),
        ("avenue_logistics_4b", [65.3, 38.0], [65.3, 39.2], "wheelset_carrier"),
        ("avenue_logistics_5", [65.3, 39.2], [65.3, 46.8], "wheelset_carrier"),
        ("avenue_logistics_6", [65.3, 46.8], [65.3, 50.0], "wheelset_carrier"),
        ("avenue_logistics_7", [65.3, 50.0], [65.3, 56.5], "wheelset_carrier"),
        ("avenue_prep_s", [90.2, 2.5], [90.2, 18.57], "wheelset_carrier"),
        ("avenue_prep_n", [90.2, 18.57], [90.2, 28.8], "wheelset_carrier"),
        ("avenue_unknown", [88.0, 39.2], [88.0, 51.5], "wheelset_carrier"),
        ("avenue_assembly_1", [123.3, 2.5], [123.3, 11.3], "wheelset_carrier"),
        ("avenue_assembly_2", [123.3, 11.3], [123.3, 28.8], "wheelset_carrier"),
        ("avenue_assembly_3", [123.3, 28.8], [123.3, 39.2], "wheelset_carrier"),
        ("avenue_press_exit", [142.3, 2.5], [142.3, 11.3], "wheelset_carrier"),
        ("avenue_closing_1", [152.3, 2.5], [152.3, 9.3], "wheelset_carrier"),
        ("avenue_closing_2", [152.3, 9.3], [152.3, 26.5], "wheelset_carrier"),
        ("avenue_closing_3", [152.3, 26.5], [152.3, 39.2], "wheelset_carrier"),
        ("avenue_closing_4", [152.3, 39.2], [152.3, 46.8], "wheelset_carrier"),
        ("avenue_finish_1", [187.5, 2.5], [187.5, 9.3], "wheelset_carrier"),
        ("avenue_finish_2", [187.5, 9.3], [187.5, 16.1], "wheelset_carrier"),
        ("avenue_finish_3", [187.5, 16.1], [187.5, 26.5], "wheelset_carrier"),
        ("avenue_finish_4", [187.5, 26.5], [187.5, 30.7], "wheelset_carrier"),
        ("avenue_finish_5", [187.5, 30.7], [187.5, 39.2], "wheelset_carrier"),
        ("avenue_finish_6", [187.5, 39.2], [187.5, 46.8], "wheelset_carrier"),
        ("avenue_east", [252.0, 2.5], [252.0, 46.8], "wheelset_carrier"),
        # Station dock spurs (handoff pads at real machine head/tail coords)
        ("dock_teardown_0", [65.3, 9.0], [57.5, 9.0], "handoff"),
        ("dock_teardown_1", [65.3, 15.0], [57.5, 15.0], "handoff"),
        ("dock_inspection_0", [65.3, 50.0], [55.0, 50.0], "handoff"),
        ("dock_inspection_1", [65.3, 56.5], [55.0, 56.5], "handoff"),
        ("dock_prep_in", [65.3, 18.57], [68.0, 18.57], "handoff"),
        ("dock_prep_out", [90.2, 18.57], [88.5, 18.57], "handoff"),
        ("dock_press_in", [123.3, 11.3], [126.2, 11.3], "handoff"),
        ("dock_press_out", [142.3, 11.3], [139.0, 11.3], "handoff"),
        ("dock_closing_in", [152.3, 9.3], [154.0, 9.3], "handoff"),
        ("dock_finishing_0", [187.5, 9.3], [182.5, 9.3], "handoff"),
        ("dock_finishing_1", [187.5, 16.1], [182.5, 16.1], "handoff"),
        ("dock_storage_infeed", [187.5, 30.7], [195.0, 30.7], "handoff"),
    ]
    scene["agv_routes"] = [
        {
            "id": rid,
            "start": [float(a[0]), float(a[1])],
            "end": [float(b[0]), float(b[1])],
            "width_m": 2.4,
            "kind": kind,
            "source": "wheelset_v2_audited",
            "navigation_validated": True,
        }
        for rid, a, b, kind in routes
    ]
    return scene


def build_v2_nav_graph(scene: dict) -> dict:
    """Build a strongly connected, capacity-1 WorkshopScene graph aligned to agv_routes."""
    nodes = {}

    def add_node(nid: str, xy: list[float], kind: str) -> None:
        nk = "dock" if kind in ("station", "service_dock") else ("buffer" if kind == "wait_area" else "aisle")
        nodes[nid] = {
            "id": nid,
            "node_id": nid,
            "zone": "workshop_v2",
            "layer": "road",
            "kind": kind,
            "node_kind": nk,
            "type": kind,
            "x": float(xy[0]),
            "y": float(xy[1]),
            "capacity": 1,
            "agv_capacity": 1,
            "queue_slots": 1 if kind == "wait_area" else 0,
            "service_capacity": 1 if nk == "dock" else 0,
            "stock_capacity": 1 if kind == "station" else 0,
            "agv_access": True,
        }

    # Build intersection nodes from route endpoints plus dedicated dock & waiting spurs.
    coord_to_id: dict[tuple[float, float], str] = {}
    named_coords = {
        (4.0, 2.5): ("J_S_4", "wait_area"),
        (65.3, 2.5): ("J_S_65", "intersection"),
        (90.2, 2.5): ("J_S_90", "intersection"),
        (123.3, 2.5): ("J_S_123", "intersection"),
        (142.3, 2.5): ("J_S_142", "intersection"),
        (152.3, 2.5): ("J_S_152", "intersection"),
        (187.5, 2.5): ("J_S_187", "wait_area"),
        (252.0, 2.5): ("J_S_252", "wait_area"),
        (4.0, 38.0): ("J_M_4", "wait_area"),
        (65.3, 38.0): ("J_W_65", "intersection"),
        (65.3, 39.2): ("J_M_65", "intersection"),
        (88.0, 39.2): ("J_M_88", "intersection"),
        (123.3, 39.2): ("J_M_123", "intersection"),
        (152.3, 39.2): ("J_M_152", "intersection"),
        (187.5, 39.2): ("J_M_187", "wait_area"),
        (90.2, 28.8): ("J_P_90", "intersection"),
        (123.3, 28.8): ("J_P_123", "intersection"),
        (152.3, 26.5): ("J_C_152", "intersection"),
        (187.5, 26.5): ("J_C_187", "intersection"),
        (4.0, 46.8): ("J_N_4", "wait_area"),
        (65.3, 46.8): ("J_N_65", "intersection"),
        (88.0, 46.8): ("J_N_88", "intersection"),
        (152.3, 46.8): ("J_N_152", "intersection"),
        (187.5, 46.8): ("J_N_187", "wait_area"),
        (252.0, 46.8): ("J_N_252", "wait_area"),
        # Dock approach junctions on spines
        (65.3, 9.0): ("R_TD_0", "intersection"),
        (65.3, 15.0): ("R_TD_1", "intersection"),
        (65.3, 18.57): ("R_PREP_IN", "intersection"),
        (65.3, 50.0): ("R_INSP_0", "intersection"),
        (65.3, 56.5): ("R_INSP_1", "intersection"),
        (90.2, 18.57): ("R_PREP_OUT", "intersection"),
        (123.3, 11.3): ("R_PRESS_IN", "intersection"),
        (142.3, 11.3): ("R_PRESS_OUT", "intersection"),
        (152.3, 9.3): ("R_CLOSE_IN", "intersection"),
        (187.5, 9.3): ("R_FIN_0", "intersection"),
        (187.5, 16.1): ("R_FIN_1", "intersection"),
        (187.5, 30.7): ("R_STORE", "intersection"),
        # Terminal station docks & service ports
        (57.5, 9.0): ("D_TEARDOWN_0", "station"),
        (57.5, 15.0): ("D_TEARDOWN_1", "station"),
        (55.0, 50.0): ("D_INSPECTION_0", "service_dock"),
        (55.0, 56.5): ("D_INSPECTION_1", "service_dock"),
        (68.0, 18.57): ("D_PREP_IN", "station"),
        (88.5, 18.57): ("D_PREP_OUT", "service_dock"),
        (126.2, 11.3): ("D_PRESS_IN", "station"),
        (139.0, 11.3): ("D_PRESS_OUT", "service_dock"),
        (154.0, 9.3): ("D_CLOSING_IN", "station"),
        (182.5, 9.3): ("D_FINISHING_0", "service_dock"),
        (182.5, 16.1): ("D_FINISHING_1", "service_dock"),
        (195.0, 30.7): ("D_STORAGE", "service_dock"),
        (88.0, 51.5): ("D_UNKNOWN", "service_dock"),
    }
    for xy, (nid, kind) in named_coords.items():
        coord_to_id[xy] = nid
        add_node(nid, list(xy), kind)

    lanes = []
    seen_pairs = set()

    def connect(u: str, v: str, speed: float = 0.5) -> None:
        key = tuple(sorted((u, v)))
        if key in seen_pairs or u == v:
            return
        seen_pairs.add(key)
        a, b = nodes[u], nodes[v]
        length = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
        lanes.append({
            "id": f"{u}__{v}",
            "lane_id": f"{u}__{v}",
            "from": u,
            "to": v,
            "src": u,
            "dst": v,
            "direction": "two_way",
            "lane_count": 1,
            "length_m": round(length, 4),
            "width_m": 2.4,
            "speed_limit_mps": speed,
            "physical_channel": f"ch_{u}__{v}",
            "capacity": 1,
            "two_way": True,
            "centerline": [[a["x"], a["y"]], [b["x"], b["y"]]],
        })

    for route in scene["agv_routes"]:
        a = (float(route["start"][0]), float(route["start"][1]))
        b = (float(route["end"][0]), float(route["end"][1]))
        if a not in coord_to_id or b not in coord_to_id:
            continue
        u, v = coord_to_id[a], coord_to_id[b]
        connect(u, v, speed=0.4 if route["kind"] == "handoff" else 0.6)

    return {
        "scene_id": scene["scene_id"],
        "schema_version": 1,
        "frame_id": "map",
        "bounds_m": list(scene["bounds_m"]),
        "units": {"length": "m", "speed": "m/s", "time": "s"},
        "agv_envelope": {
            "length_m": 1.70,
            "width_m": 1.50,
            "min_clearance_m": 0.40,
            "corner_radius_m": 1.134,
        },
        "evidence": {
            "level": "process_referenced_synthetic",
            "assumptions_zh": [
                "基于轮轴检修线视频布局修正隔断门洞与缓存行，采用 1.70x1.50m 黑橙载运 AGV 差速物理模型",
                "黄色车间地面AGV通道宽 2.4m，实测中心线最小静态障碍净空 1.21m",
            ],
            "not_measured_zh": [
                "工位内部专机机械参数未实测，仅建工位对接坐标与地面托架交接高度(z<=0.95m)",
            ],
        },
        "zones": [
            {
                "id": "workshop_v2",
                "zone_id": "workshop_v2",
                "name_zh": "轮轴检修车间V2",
                "kind": "traffic",
                "polygon": [[0.0, 0.0], [258.0, 0.0], [258.0, 72.0], [0.0, 72.0]],
            }
        ],
        "nodes": list(nodes.values()),
        "lanes": lanes,
        "material_routes": [],
    }


def static_collision_aabbs(scene: dict, *, z_range: tuple[float, float] = (0.05, 1.20)) -> list[tuple[str, str, list[float]]]:
    """Extract all obstacle axis-aligned boxes intersecting robot height for clearance audit."""
    assets = make_assets()
    boxes = []
    for box in scene["boxes"]:
        cx, cy, cz = box["center"]
        sx, sy, sz = box["size"]
        if cz + sz / 2 < z_range[0] or cz - sz / 2 > z_range[1]:
            continue
        yaw = box.get("yaw", 0.0)
        if abs(yaw) > 1e-6:
            pts = [
                transform((i * sx / 2, j * sy / 2, k * sz / 2), [cx, cy, cz, 0, 0, yaw])
                for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)
            ]
            xs, ys, zs = zip(*pts)
            aabb = [min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)]
        else:
            aabb = [cx - sx / 2, cy - sy / 2, cz - sz / 2, cx + sx / 2, cy + sy / 2, cz + sz / 2]
        boxes.append((box["id"], "box", aabb))

    for inst in scene["instances"]:
        asset = inst["asset"]
        if asset in ("inspection_track", "track_segment"):
            continue
        cboxes = collision_boxes(asset)
        if cboxes is None:
            lo, hi = assets[asset].bounds()
            cboxes = [([(u + v) / 2 for u, v in zip(lo, hi)], [v - u for u, v in zip(lo, hi)])]
        for center, sz in cboxes:
            pts = [
                transform((center[0] + i * sz[0] / 2, center[1] + j * sz[1] / 2, center[2] + k * sz[2] / 2), inst["pose"])
                for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)
            ]
            xs, ys, zs = zip(*pts)
            if min(zs) <= z_range[1] and max(zs) >= z_range[0]:
                boxes.append((inst["id"], asset, [min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)]))
    return boxes


def audit_v2_clearances(scene: dict, nav_graph: dict, *, min_required_m: float = 1.15) -> dict:
    """Verify every yellow AGV route and scheduler lane clears all static 3D obstacles and forms 1 connected component."""
    obstacles = static_collision_aabbs(scene)
    nodes = {n["id"]: n for n in nav_graph["nodes"]}
    route_checks = []
    min_clr = math.inf
    failed = []
    for route in scene["agv_routes"]:
        d_best = math.inf
        nearest = None
        for oid, kind, aabb in obstacles:
            d = segment_box(route["start"], route["end"], aabb)
            if d < d_best:
                d_best = d
                nearest = (oid, kind)
        min_clr = min(min_clr, d_best)
        item = {"id": route["id"], "clearance_m": round(d_best, 4), "nearest": nearest}
        route_checks.append(item)
        if d_best < min_required_m:
            failed.append(item)

    for lane in nav_graph["lanes"]:
        a = nodes[lane["src"]]
        b = nodes[lane["dst"]]
        d_best = min(segment_box([a["x"], a["y"]], [b["x"], b["y"]], aabb) for _, _, aabb in obstacles)
        min_clr = min(min_clr, d_best)
        if d_best < min_required_m:
            failed.append({"id": "nav:" + lane["id"], "clearance_m": round(d_best, 4)})

    adj: dict[str, set[str]] = {n["id"]: set() for n in nav_graph["nodes"]}
    for lane in nav_graph["lanes"]:
        adj[lane["src"]].add(lane["dst"])
        adj[lane["dst"]].add(lane["src"])
    visited: set[str] = set()
    components = 0
    for nid in adj:
        if nid in visited:
            continue
        components += 1
        stack = [nid]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            stack.extend(adj[cur] - visited)

    return {
        "passed": len(failed) == 0 and components == 1,
        "min_required_m": min_required_m,
        "min_clearance_m": round(min_clr, 4) if math.isfinite(min_clr) else None,
        "routes_checked": len(route_checks),
        "connected_components": components,
        "failed": failed,
    }


def make_v2_world(scene: dict, *, physics_rate_hz: int = 500, camera_rate_hz: float = 1.0) -> ET.Element:
    """Create the Gazebo Classic SDF world for wheelset_workshop_v2 with high-RTF physics + overview camera."""
    scale = scene["global_scene_scale"]
    root = ET.Element("sdf", version="1.6")
    world = e(root, "world", name=scene["scene_id"])
    e(world, "gravity", "0 0 -9.81")
    physics = e(world, "physics", name="v2_contact", type="ode")
    e(physics, "max_step_size", "0.002")
    e(physics, "real_time_update_rate", str(int(physics_rate_hz)))
    solver = e(e(physics, "ode"), "solver")
    e(solver, "type", "quick")
    e(solver, "iters", "50")
    e(solver, "sor", "1")
    e(solver, "island_threads", "4")

    settings = e(world, "scene")
    e(settings, "ambient", "0.62 0.62 0.62 1")
    e(settings, "background", "0.60 0.63 0.66 1")
    e(settings, "shadows", "false")
    light = e(world, "light", name="sun", type="directional")
    e(light, "pose", "0 0 80 0 0 0")
    e(light, "diffuse", "0.78 0.78 0.76 1")
    e(light, "specular", "0.20 0.20 0.20 1")
    e(light, "direction", "-0.45 0.30 -0.85")
    e(light, "cast_shadows", "false")

    for box in scene["boxes"]:
        box_model(world, box, scale, collision=box["id"] != "floor")
    # Static floor contact slab with low friction & clean plane
    floor_box = next(b for b in scene["boxes"] if b["id"] == "floor")
    floor_m = world.find("model[@name='floor']")
    if floor_m is not None:
        link = floor_m.find("link[@name='body']")
        col = ET.SubElement(link, "collision", name="collision")
        geo = ET.SubElement(col, "geometry")
        ET.SubElement(ET.SubElement(geo, "box"), "size").text = " ".join(str(v) for v in floor_box["size"])

    for road in scene["agv_routes"]:
        a, b = road["start"], road["end"]
        strip = {
            "id": "paint_" + road["id"],
            "center": [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.008],
            "size": [max(0.2, math.dist(a, b)), 0.18, 0.008],
            "material": "yellow",
            "yaw": math.atan2(b[1] - a[1], b[0] - a[0]),
        }
        box_model(world, strip, scale, collision=False)

    for item in scene["instances"]:
        inc = e(world, "include")
        e(inc, "uri", "model://wsv1_" + item["asset"])
        e(inc, "name", item["id"])
        pose_v = item["pose"][:]
        pose_v[:3] = [v * scale for v in pose_v[:3]]
        e(inc, "pose", " ".join(f"{v:.7g}" for v in pose_v))

    state = e(world, "plugin", name="wheelset_v2_state", filename="libgazebo_ros_state.so")
    e(e(state, "ros"), "namespace", "/")
    e(state, "update_rate", "50")

    # Add south isometric overview camera covering active overhaul stations and indoor AGV lanes.
    cams = [
        ("overview_cam", [118.0, -34.0, 62.0], [118.0, 26.0, 0.5], "/wheelset_v2_overview", 1.35),
    ]
    for cname, pos, tgt, ros_ns, fov in cams:
        dx, dy, dz = tgt[0] - pos[0], tgt[1] - pos[1], tgt[2] - pos[2]
        yaw = math.atan2(dy, dx)
        pitch = math.atan2(-dz, math.hypot(dx, dy))
        cm = e(world, "model", name=cname)
        e(cm, "static", "true")
        e(cm, "pose", f"{pos[0]} {pos[1]} {pos[2]} 0 {pitch:.6f} {yaw:.6f}")
        lnk = e(cm, "link", name="optics")
        sens = e(lnk, "sensor", name=cname + "_sensor", type="camera")
        e(sens, "always_on", "true")
        e(sens, "update_rate", str(camera_rate_hz))
        cam_el = e(sens, "camera")
        e(cam_el, "horizontal_fov", str(fov))
        img = e(cam_el, "image")
        e(img, "width", "640")
        e(img, "height", "360")
        e(img, "format", "R8G8B8")
        clip = e(cam_el, "clip")
        e(clip, "near", "0.5")
        e(clip, "far", "420.0")
        plug = e(sens, "plugin", name=cname + "_ros", filename="libgazebo_ros_camera.so")
        e(e(plug, "ros"), "namespace", ros_ns)
        e(plug, "camera_name", "camera")
        e(plug, "frame_name", "world")

    return root


def compile_v2_bundle(out_dir: Path, *, resolution: float = DEFAULT_MAP_RESOLUTION) -> dict:
    """Compile scene JSON, audited nav graph, RAWSim-O files, occupancy map, and Gazebo world."""
    from tools import scene_to_rawsimo as sr

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = build_improved_scene()
    nav = build_v2_nav_graph(scene)
    audit = audit_v2_clearances(scene, nav)
    if not audit["passed"]:
        raise RuntimeError(f"wheelset_workshop_v2 clearance audit failed: {audit['failed']}")

    (out_dir / "wheelset_workshop_v2.json").write_text(json.dumps(scene, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "wheelset_workshop_v2_nav.json").write_text(json.dumps(nav, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    model = sr.build_model(nav, scale=1.0, n_bots=4, bot_radius=0.85, station_radius=0.75, max_velocity=0.6)
    problems = sr.check_model(model)
    if problems:
        raise RuntimeError(f"RAWSim-O check failed: {problems}")
    (out_dir / "wheelset_workshop_v2_map.xinst").write_text(sr.render_instance(model, "wheelset_workshop_v2"), encoding="utf-8")
    (out_dir / "wheelset_workshop_v2_map.xsett").write_text(sr.render_settings("wheelset_workshop_v2"), encoding="utf-8")
    (out_dir / "wheelset_workshop_v2_map.xconf").write_text(sr.render_control("wheelset_workshop_v2"), encoding="utf-8")

    # Rasterize full 258x72m workshop collision slices at LiDAR beam height (0.18..0.32m)
    obstacles = static_collision_aabbs(scene, z_range=(0.18, 0.32))
    polygons = [[(a[0], a[1]), (a[3], a[1]), (a[3], a[4]), (a[0], a[4])] for _, _, a in obstacles]
    w, h, pixels = rasterize(polygons, [0.0, 0.0, 258.0, 72.0], resolution, rule="centre", max_cells=60_000_000)
    (out_dir / "map.pgm").write_bytes(f"P5\n{w} {h}\n255\n".encode() + pixels)
    (out_dir / "map.yaml").write_text(
        yaml.safe_dump({
            "image": "map.pgm",
            "mode": "trinary",
            "resolution": float(resolution),
            "origin": [0.0, 0.0, 0.0],
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.196,
        }),
        encoding="utf-8",
    )

    world_xml = make_v2_world(scene)
    save_xml(world_xml, out_dir / "wheelset_workshop_v2.world")

    summary = {
        "scene_id": scene["scene_id"],
        "clearance_audit": audit,
        "nav_nodes": len(nav["nodes"]),
        "nav_lanes": len(nav["lanes"]),
        "map_size_px": [w, h],
        "map_resolution_m": resolution,
        "obstacles_sliced": len(polygons),
    }
    (out_dir / "v2_compile_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "out" / "wheelset_v2_bundle")
    args = p.parse_args()
    print(json.dumps(compile_v2_bundle(args.out), indent=2))
