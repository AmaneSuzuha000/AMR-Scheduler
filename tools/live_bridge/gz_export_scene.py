"""把车间场景导出成 Gazebo 世界（MAP-01E 第一步）。

原则
----
场景文件是唯一真源；这个脚本只做坐标换算与格式转换，不引入任何几何常量。
1 世界米 = 25 路网米（与 gz_frame.GzFrame 同一个比例，不另立一套）。

导出的世界是**缩比试验场**，不是全尺寸车间：它只用来验证"路径能不能走通、
坐标对不对"，绝不能拿来声称车间的通行能力或行驶时间。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from amr_sched.workshop import WorkshopScene
from tools.live_bridge.gz_frame import GzFrame

MODEL_TEMPLATE = """    <model name="{name}">
      <static>true</static>
      <pose>{x:.4f} {y:.4f} 0.005 0 0 {yaw:.6f}</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>{length:.4f} {width:.4f} 0.01</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>{length:.4f} {width:.4f} 0.01</size></box></geometry>
        <material><ambient>0.45 0.47 0.5 1</ambient><diffuse>0.45 0.47 0.5 1</diffuse></material></visual>
      </link>
    </model>"""

HEADER = """<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="rail_workshop_scout_demo">
    <include><uri>model://ground_plane</uri></include>
    <include><uri>model://sun</uri></include>
    <gravity>0 0 -9.8</gravity>
    <physics type="ode"><max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate></physics>
    <scene><ambient>0.4 0.4 0.4 1</ambient><background>0.7 0.7 0.7 1</background>
      <shadows>false</shadows></scene>
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros><namespace>/</namespace></ros>
      <update_rate>30.0</update_rate>
    </plugin>"""

FOOTER = """  </world>
</sdf>"""


def export(scene: WorkshopScene, scale: float, out_world: Path, out_route: Path,
           route_from: str, route_to: str) -> dict:
    frame = GzFrame(network_origin=(0.0, 0.0), origin=(0.0, 0.0), scale=scale)
    models = []
    xs: list[float] = []
    ys: list[float] = []
    for lane in scene.lanes:
        pts = lane.centerline
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            wx0, wy0 = frame.to_world(x0, y0)
            wx1, wy1 = frame.to_world(x1, y1)
            length = math.hypot(wx1 - wx0, wy1 - wy0)
            if length <= 0:
                continue
            yaw = math.atan2(wy1 - wy0, wx1 - wx0)
            models.append(MODEL_TEMPLATE.format(
                name="%s_%d" % (lane.lane_id.lower(), i),
                x=(wx0 + wx1) / 2.0, y=(wy0 + wy1) / 2.0, yaw=yaw,
                length=length, width=lane.width_m * scale))
            xs.extend([wx0, wx1])
            ys.extend([wy0, wy1])
    out_world.parent.mkdir(parents=True, exist_ok=True)
    out_world.write_text("\n".join([HEADER] + models + [FOOTER]) + "\n", encoding="utf-8")

    # A→B route along the south main aisle: open floor, no workshop-specific hazards.
    a = scene.nodes[route_from]
    b = scene.nodes[route_to]
    route = {
        "scene": scene.scene_id,
        "scale": scale,
        "frame": "world",
        "note": "缩比试验场路线；1 世界米 = %g 路网米，不可用于车间通行结论" % (1.0 / scale),
        "waypoints": [list(frame.to_world(a.x, a.y)), list(frame.to_world(b.x, b.y))],
        "network_waypoints": [[a.x, a.y], [b.x, b.y]],
        "network_distance_m": round(math.hypot(b.x - a.x, b.y - a.y), 3),
        "world_distance_m": round(math.hypot(b.x - a.x, b.y - a.y) * scale, 4),
    }
    out_route.write_text(json.dumps(route, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "segments": len(models),
        "world_bbox": [round(min(xs), 3), round(min(ys), 3), round(max(xs), 3), round(max(ys), 3)],
        "route_world_distance_m": route["world_distance_m"],
        "route_network_distance_m": route["network_distance_m"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--scale", type=float, default=1.0 / 25.0)
    ap.add_argument("--out", default=str(ROOT / "tools" / "live_bridge" / "worlds" / "rail_workshop_scout_demo.world"))
    ap.add_argument("--route-out", default=str(ROOT / "tools" / "live_bridge" / "scene_route.json"))
    ap.add_argument("--from", dest="route_from", default="J_S_8")
    ap.add_argument("--to", dest="route_to", default="J_S_125")
    args = ap.parse_args(argv)

    scene = WorkshopScene.load(args.scene) if args.scene else WorkshopScene.load()
    stats = export(scene, args.scale, Path(args.out), Path(args.route_out),
                   args.route_from, args.route_to)
    print("场景 %s -> %s" % (scene.scene_id, args.out))
    print("路段模型 %d 个；世界包围盒 %s" % (stats["segments"], stats["world_bbox"]))
    print("A→B：路网 %.1f m -> 世界 %.3f m" % (
        stats["route_network_distance_m"], stats["route_world_distance_m"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())