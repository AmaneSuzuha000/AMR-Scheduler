"""World, RViz2 config, Live V18 coordinator overlay, and 3-view video assembler for MAP-01."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def derive_map01_world(
    source_world: Path,
    target_world: Path,
    *,
    physics_rate_hz: int = 500,
    camera_topic_ns: str = "/map01_review",
    camera_rate_hz: float = 2.0,
) -> Path:
    """Create a run-scoped copy of rail_workshop_metric.world with 500 Hz 2ms ODE and review camera."""
    tree = ET.parse(source_world)
    world = tree.getroot().find("world")
    if world is None:
        raise ValueError("world element required")
    step_dt = 1.0 / float(max(100, int(physics_rate_hz)))
    step_el = world.find("physics/max_step_size")
    if step_el is not None:
        step_el.text = f"{step_dt:.6g}"
    rate_el = world.find("physics/real_time_update_rate")
    if rate_el is None:
        raise ValueError("physics real_time_update_rate required")
    rate_el.text = str(int(physics_rate_hz))
    solver_el = world.find("physics/ode/solver")
    if solver_el is not None:
        iters_el = solver_el.find("iters")
        if iters_el is not None:
            iters_el.text = "30" if int(physics_rate_hz) <= 300 else "40"
        if solver_el.find("island_threads") is None:
            ET.SubElement(solver_el, "island_threads").text = "4"
    scene_el = world.find("scene")
    if scene_el is None:
        scene_el = ET.SubElement(world, "scene")
    if scene_el.find("shadows") is None:
        ET.SubElement(scene_el, "shadows").text = "false"

    # Add passive top-south isometric review camera covering the 140x90 m workshop.
    cam_pos = (72.0, -16.0, 48.0)
    cam_target = (74.0, 48.0, 0.5)
    dx, dy, dz = cam_target[0] - cam_pos[0], cam_target[1] - cam_pos[1], cam_target[2] - cam_pos[2]
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(-dz, math.hypot(dx, dy))

    cam_model = ET.SubElement(world, "model", name="map01_review_camera")
    ET.SubElement(cam_model, "static").text = "true"
    ET.SubElement(cam_model, "pose").text = f"{cam_pos[0]} {cam_pos[1]} {cam_pos[2]} 0 {pitch:.6f} {yaw:.6f}"
    link = ET.SubElement(cam_model, "link", name="optics")
    sensor = ET.SubElement(link, "sensor", name="overview_camera", type="camera")
    ET.SubElement(sensor, "always_on").text = "true"
    ET.SubElement(sensor, "update_rate").text = str(camera_rate_hz)
    cam = ET.SubElement(sensor, "camera")
    ET.SubElement(cam, "horizontal_fov").text = "1.65"
    img = ET.SubElement(cam, "image")
    ET.SubElement(img, "width").text = "640"
    ET.SubElement(img, "height").text = "360"
    ET.SubElement(img, "format").text = "R8G8B8"
    clip = ET.SubElement(cam, "clip")
    ET.SubElement(clip, "near").text = "0.5"
    ET.SubElement(clip, "far").text = "300.0"
    plugin = ET.SubElement(sensor, "plugin", name="map01_camera_ros", filename="libgazebo_ros_camera.so")
    ros = ET.SubElement(plugin, "ros")
    ET.SubElement(ros, "namespace").text = camera_topic_ns
    ET.SubElement(plugin, "camera_name").text = "camera"
    ET.SubElement(plugin, "frame_name").text = "world"

    target_world.parent.mkdir(parents=True, exist_ok=True)
    tree.write(target_world, encoding="utf-8", xml_declaration=True)
    return target_world


def write_map01_rviz_config(
    target_rviz: Path,
    *,
    robot_namespaces: tuple[str, ...],
    robot_ids: tuple[str, ...],
    scan_colors: tuple[str, ...] = (
        "255; 70; 70",
        "70; 220; 110",
        "80; 170; 255",
        "255; 195; 50",
        "225; 95; 255",
        "80; 245; 235",
    ),
) -> Path:
    """Write a complete 62-node MAP-01 RViz2 configuration subscribing to map, scans, V18 markers & paths."""
    import yaml

    displays = [
        {
            "Alpha": 0.45,
            "Cell Size": 5,
            "Class": "rviz_default_plugins/Grid",
            "Color": "120; 130; 145",
            "Enabled": True,
            "Name": "Grid",
            "Plane": "XY",
            "Plane Cell Count": 32,
            "Reference Frame": "<Fixed Frame>",
        },
        {
            "Alpha": 0.85,
            "Class": "rviz_default_plugins/Map",
            "Color Scheme": "map",
            "Draw Behind": True,
            "Enabled": True,
            "Name": "WorkshopMap",
            "Topic": {
                "Depth": 1,
                "Durability Policy": "Transient Local",
                "History Policy": "Keep Last",
                "Reliability Policy": "Reliable",
                "Value": f"/{robot_namespaces[0]}/map",
            },
            "Use Timestamp": False,
        },
        {
            "Class": "rviz_default_plugins/MarkerArray",
            "Enabled": True,
            "Name": "CVCRTopology",
            "Namespaces": {},
            "Topic": {
                "Depth": 5,
                "Durability Policy": "Volatile",
                "History Policy": "Keep Last",
                "Reliability Policy": "Reliable",
                "Value": "/cvcr/topology_markers",
            },
        },
        {
            "Class": "rviz_default_plugins/MarkerArray",
            "Enabled": True,
            "Name": "CVCRP45Belief",
            "Namespaces": {},
            "Topic": {
                "Depth": 5,
                "Durability Policy": "Volatile",
                "History Policy": "Keep Last",
                "Reliability Policy": "Reliable",
                "Value": "/cvcr/corridor_belief_markers",
            },
        },
        {
            "Class": "rviz_default_plugins/MarkerArray",
            "Enabled": True,
            "Name": "CVCRFleetPlans",
            "Namespaces": {},
            "Topic": {
                "Depth": 5,
                "Durability Policy": "Volatile",
                "History Policy": "Keep Last",
                "Reliability Policy": "Reliable",
                "Value": "/cvcr/fleet_plan_markers",
            },
        },
        {
            "Class": "rviz_default_plugins/MarkerArray",
            "Enabled": True,
            "Name": "CVCRKittingHUD",
            "Namespaces": {},
            "Topic": {
                "Depth": 5,
                "Durability Policy": "Volatile",
                "History Policy": "Keep Last",
                "Reliability Policy": "Reliable",
                "Value": "/cvcr/kitting_hud_markers",
            },
        },
    ]

    for idx, (ns, rid) in enumerate(zip(robot_namespaces, robot_ids)):
        col = scan_colors[idx % len(scan_colors)]
        displays.append({
            "Alpha": 0.9,
            "Autocompute Intensity Bounds": True,
            "Class": "rviz_default_plugins/LaserScan",
            "Color": col,
            "Color Transformer": "FlatColor",
            "Enabled": True,
            "Name": f"{rid}_Scan",
            "Size (Pixels)": 3,
            "Size (m)": 0.14,
            "Style": "Flat Squares",
            "Topic": {
                "Depth": 5,
                "Durability Policy": "Volatile",
                "History Policy": "Keep Last",
                "Reliability Policy": "Best Effort",
                "Value": f"/{ns}/scan",
            },
        })
        displays.append({
            "Class": "rviz_default_plugins/Path",
            "Color": col,
            "Enabled": True,
            "Line Style": "Billboards",
            "Line Width": 0.25,
            "Name": f"{rid}_Path",
            "Topic": {
                "Depth": 5,
                "Durability Policy": "Volatile",
                "History Policy": "Keep Last",
                "Reliability Policy": "Reliable",
                "Value": f"/cvcr/planned_paths/{rid}",
            },
        })

    cfg = {
        "Panels": [
            {"Class": "rviz_common/Displays", "Name": "Displays", "Tree Height": 420},
        ],
        "Visualization Manager": {
            "Class": "",
            "Displays": displays,
            "Global Options": {
                "Background Color": "24; 28; 34",
                "Fixed Frame": "map",
                "Frame Rate": 2,
            },
            "Tools": [
                {"Class": "rviz_default_plugins/MoveCamera"},
                {"Class": "rviz_default_plugins/FocusCamera"},
            ],
            "Views": {
                "Current": {
                    "Class": "rviz_default_plugins/TopDownOrtho",
                    "Name": "MAP01Overview",
                    "Scale": 5.2,
                    "X": 70.0,
                    "Y": 45.0,
                    "Angle": 0.0,
                }
            },
        },
        "Window Geometry": {"Width": 1280, "Height": 760, "X": 40, "Y": 40},
    }
    target_rviz.parent.mkdir(parents=True, exist_ok=True)
    target_rviz.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return target_rviz


class Map01V18Observer:
    """Tracks V18 PhysicalCVCRCoordinator transactions and publishes live RViz2 CVCR overlays."""

    def __init__(self, coordinator, operations, record_fn):
        self.coordinator = coordinator
        self.sim = coordinator.simulator
        self.operations = operations
        self.record = record_fn
        self.receipts: list[dict] = []
        self.last_publish = -math.inf
        self.publisher = None
        self.saw_rviz_subscription = False

        orig_commit = coordinator.owner.validate_and_commit

        def audited_commit(proposal, **kwargs):
            receipt = orig_commit(proposal, **kwargs)
            kd = proposal.telemetry.get("kitting_decision") or {}
            entry = {
                "tick": int(self.sim._step_cursor),
                "policy": str(proposal.policy_id),
                "candidate": kd.get("cand_id"),
                "v17_meta": kd.get("v17_meta"),
                "assignment": dict(proposal.proposed_assignment),
                "receipt": asdict(receipt),
            }
            self.receipts.append(entry)
            self.record("v18_transaction", **entry)
            return receipt

        coordinator.owner.validate_and_commit = audited_commit

    def start_display(self, executor) -> None:
        from rclpy.parameter import Parameter
        from tools.live_bridge.rviz_cvcr_overlay import RVizCVCROverlayPublisher

        self.publisher = RVizCVCROverlayPublisher(self.sim.network, node_name="map01_cvcr_rviz_overlay")
        node = self.publisher.ros_node
        node.set_parameters([Parameter("use_sim_time", value=True)])
        executor.add_node(node)

    def publish(self, now_wall: float) -> None:
        period = 1.0 if self.saw_rviz_subscription else 0.5
        if self.publisher is None or now_wall - self.last_publish < period:
            return
        self.last_publish = now_wall
        poses = {
            rid: (float(g.pose["x"]), float(g.pose["y"]))
            for rid, g in self.sim.executions.items()
            if g.pose is not None
        }
        planned = {}
        for rid, job in self.sim._jobs.items():
            if job.idx < len(job.hops):
                seq = [(job.hops[job.idx].src, 0)] + [(h.dst, 0) for h in job.hops[job.idx:]]
                planned[rid] = seq
        self.publisher.publish_state(
            tick=int(self.sim._step_cursor),
            belief_model=getattr(self.coordinator, "_belief_model", None),
            planned_routes=planned,
            robot_poses=poses,
            operations=self.operations,
        )
        if not self.saw_rviz_subscription:
            endpoints = self.publisher.ros_node.get_subscriptions_info_by_topic("/cvcr/topology_markers")
            if any("rviz" in (info.node_name or "").lower() for info in endpoints):
                self.saw_rviz_subscription = True
                self.record("rviz_connected", endpoints=[info.node_name for info in endpoints])

    def close(self, executor=None) -> dict:
        if self.publisher is not None:
            if executor is not None:
                executor.remove_node(self.publisher.ros_node)
            self.publisher.close()
            self.publisher = None
        return {
            "policy": self.coordinator.policy_method,
            "transaction_count": len(self.receipts),
            "accepted": sum(1 for r in self.receipts if r["receipt"]["accepted"]),
            "non_keep_count": sum(1 for r in self.receipts if r["receipt"]["accepted"] and r.get("candidate") != "C0_keep"),
            "rviz_subscribed": self.saw_rviz_subscription,
        }


def assemble_three_panel_video(
    capture_dir: Path,
    output_mp4: Path,
    *,
    fps: int = 8,
    width: int = 1920,
    height: int = 1080,
    gz_title: str = "1. Gazebo 3D Physics & Sensors (6 SCOUTs @ 650Hz)",
    rviz_title: str = "2. ROS 2 RViz2 (Nav2 Scans + V18 Routes + Kitting HUD)",
    wpf_title: str = "3. RAWSim-O WPF Digital Twin (--schedule 127.0.0.1:9500)",
    footer_prefix: str = "MAP-01 62-Node Formal Chain",
) -> dict:
    """Assemble synchronized Gazebo + RViz2 + Windows RAWSim-O WPF frames into one MP4 using common UTC timestamps."""
    from PIL import Image, ImageDraw

    def load_series(jsonl_path: Path, subdir: Path):
        if not jsonl_path.is_file():
            return []
        rows = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            fp = subdir / item["file"]
            if fp.is_file():
                rows.append((float(item["utc_s"]), fp, item))
        rows.sort(key=lambda r: r[0])
        return rows

    gz_rows = load_series(capture_dir / "gazebo_frames.jsonl", capture_dir / "gazebo")
    rv_rows = load_series(capture_dir / "rviz_frames.jsonl", capture_dir / "rviz")
    wpf_rows = load_series(capture_dir / "wpf" / "wpf_frames.jsonl", capture_dir / "wpf")

    all_ts = [r[0] for r in gz_rows] + [r[0] for r in rv_rows] + [r[0] for r in wpf_rows]
    if not all_ts:
        raise RuntimeError("No captured frames found in any view")
    t_start = min(all_ts)
    t_end = max(all_ts)
    duration_s = max(1.0, t_end - t_start)

    out_frames = capture_dir / "composite"
    out_frames.mkdir(parents=True, exist_ok=True)

    half_w = width // 2
    half_h = height // 2

    def latest_before(series, ts):
        chosen = None
        for item in series:
            if item[0] <= ts:
                chosen = item
            else:
                break
        return chosen

    def fit_panel(img_path: Path, box_w: int, box_h: int, label: str) -> Image.Image:
        panel = Image.new("RGB", (box_w, box_h), (20, 24, 30))
        im = Image.open(img_path).convert("RGB")
        scale = min(box_w / im.width, (box_h - 28) / im.height)
        nw = max(1, int(im.width * scale))
        nh = max(1, int(im.height * scale))
        im = im.resize((nw, nh), Image.Resampling.BILINEAR)
        panel.paste(im, ((box_w - nw) // 2, 28 + (box_h - 28 - nh) // 2))
        draw = ImageDraw.Draw(panel)
        draw.rectangle([0, 0, box_w, 28], fill=(32, 38, 48))
        draw.text((10, 7), label, fill=(235, 240, 248))
        return panel

    n_out = max(fps * 2, int(round(duration_s * fps)))
    dt = duration_s / n_out
    for i in range(n_out):
        ts = t_start + i * dt
        canvas = Image.new("RGB", (width, height), (14, 17, 22))
        # Left full height: Gazebo 3D Overview (960x1080) or Left Top Gazebo + Left Bottom RViz2, Right WPF
        gz = latest_before(gz_rows, ts)
        rv = latest_before(rv_rows, ts)
        wpf = latest_before(wpf_rows, ts)

        if gz is not None:
            sim_lbl = f" [sim={gz[2].get('sim_s', 0):.1f}s]" if "sim_s" in gz[2] else ""
            p_gz = fit_panel(gz[1], half_w, half_h, f"{gz_title}{sim_lbl}")
            canvas.paste(p_gz, (0, 0))
        if rv is not None:
            p_rv = fit_panel(rv[1], half_w, half_h, rviz_title)
            canvas.paste(p_rv, (0, half_h))
        if wpf is not None:
            p_wpf = fit_panel(wpf[1], half_w, height, wpf_title)
            canvas.paste(p_wpf, (half_w, 0))
        else:
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([half_w, 0, width, height], fill=(22, 26, 33))
            draw.text((half_w + 40, height // 2), "Waiting for WPF Window...", fill=(180, 185, 195))

        draw = ImageDraw.Draw(canvas)
        draw.text((half_w + 12, height - 22), f"{footer_prefix} | t_wall={ts - t_start:.2f}s / {duration_s:.2f}s", fill=(180, 220, 255))
        canvas.save(out_frames / f"{i:06d}.jpg", quality=88)

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(out_frames / "%06d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output_mp4),
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(output_mp4), "-f", "null", "-"], check=True)
    return {
        "mp4": str(output_mp4),
        "frames_out": n_out,
        "fps": fps,
        "duration_s": round(duration_s, 3),
        "gazebo_frames": len(gz_rows),
        "rviz_frames": len(rv_rows),
        "wpf_frames": len(wpf_rows),
    }
