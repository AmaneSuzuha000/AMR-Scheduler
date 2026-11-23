"""Independent ROS camera + RViz2 X11 window recorder running in its own process."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from PIL import Image
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage


def _find_rviz_pids(pattern: str) -> list[int]:
    pids: list[int] = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
        except OSError:
            continue
        if "rviz2" in cmd and (not pattern or pattern in cmd):
            pids.append(int(p.name))
    return pids


def _window_size(xid: str) -> tuple[int, int]:
    r = subprocess.run(
        ["xwininfo", "-id", str(xid)],
        capture_output=True, text=True, check=False,
    )
    w = h = 0
    for line in r.stdout.splitlines():
        s = line.strip()
        if s.startswith("Width:"):
            w = int(s.split(":", 1)[1].strip())
        elif s.startswith("Height:"):
            h = int(s.split(":", 1)[1].strip())
    return w, h


def _find_xid_for_pids(pids: list[int]) -> str | None:
    candidates: list[tuple[int, str]] = []
    queries = [["xdotool", "search", "--onlyvisible", "--pid", str(pid)] for pid in pids]
    queries.append(["xdotool", "search", "--onlyvisible", "--name", " - RViz"])
    for cmd in queries:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        for line in r.stdout.splitlines():
            xid = line.strip()
            if not xid:
                continue
            w, h = _window_size(xid)
            if w >= 700 and h >= 500:
                candidates.append((w * h, xid))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


class DualViewCapture(Node):
    def __init__(self, out_dir: Path, camera_topic: str, rviz_match: str, min_cam_period: float = 0.22):
        super().__init__("map01_dual_view_capture")
        self.out_dir = out_dir
        self.gz_dir = out_dir / "gazebo"
        self.rv_dir = out_dir / "rviz"
        self.tmp_dir = out_dir / ".tmp_capture"
        self.gz_dir.mkdir(parents=True, exist_ok=True)
        self.rv_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.gz_log = (out_dir / "gazebo_frames.jsonl").open("w", encoding="utf-8")
        self.rv_log = (out_dir / "rviz_frames.jsonl").open("w", encoding="utf-8")
        self.rviz_match = rviz_match
        self.min_cam_period = min_cam_period
        self.last_cam_wall = 0.0
        self.gz_idx = 0
        self.rv_idx = 0
        self.rviz_xid: str | None = None
        self.create_subscription(RosImage, camera_topic, self.on_camera, qos_profile_sensor_data)

    def on_camera(self, msg: RosImage) -> None:
        now = time.monotonic()
        if now - self.last_cam_wall < self.min_cam_period:
            return
        self.last_cam_wall = now
        w, h = int(msg.width), int(msg.height)
        if w <= 0 or h <= 0 or msg.encoding.lower() not in ("rgb8", "bgr8"):
            return
        raw = bytes(msg.data)
        mode = "RGB" if msg.encoding.lower() == "rgb8" else "BGR"
        im = Image.frombytes("RGB", (w, h), raw, "raw", mode)
        fname = f"{self.gz_idx:06d}.jpg"
        tmp_path = self.tmp_dir / f"gz_{fname}"
        im.save(tmp_path, format="JPEG", quality=88)
        os.replace(tmp_path, self.gz_dir / fname)
        sim_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        row = {"idx": self.gz_idx, "file": fname, "sim_s": sim_s, "wall_s": now, "utc_s": time.time()}
        self.gz_log.write(json.dumps(row) + "\n")
        self.gz_log.flush()
        self.gz_idx += 1

    def poll_rviz(self) -> None:
        if self.rviz_xid is None:
            pids = _find_rviz_pids(self.rviz_match)
            if pids:
                self.rviz_xid = _find_xid_for_pids(pids)
        if not self.rviz_xid:
            return
        fname = f"{self.rv_idx:06d}.jpg"
        out_path = self.rv_dir / fname
        tmp_path = self.tmp_dir / f"rv_{fname}"
        r = subprocess.run(
            ["import", "-window", str(self.rviz_xid), str(tmp_path)],
            capture_output=True, text=True, check=False,
        )
        if r.returncode != 0 or not tmp_path.is_file() or tmp_path.stat().st_size < 1000:
            tmp_path.unlink(missing_ok=True)
            self.rviz_xid = None
            return
        os.replace(tmp_path, out_path)
        row = {"idx": self.rv_idx, "file": fname, "wall_s": time.monotonic(), "utc_s": time.time()}
        self.rv_log.write(json.dumps(row) + "\n")
        self.rv_log.flush()
        self.rv_idx += 1

    def close(self) -> None:
        self.gz_log.close()
        self.rv_log.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--camera-topic", default="/map01_review/camera/image_raw")
    p.add_argument("--rviz-match", default="map01_live.rviz")
    p.add_argument("--stop-file", type=Path, required=True)
    p.add_argument("--rviz-hz", type=float, default=3.0)
    args = p.parse_args()

    rclpy.init()
    node = DualViewCapture(args.out, args.camera_topic, args.rviz_match)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    last_rv = 0.0
    rv_period = 1.0 / max(0.5, args.rviz_hz)
    try:
        while not args.stop_file.exists():
            executor.spin_once(timeout_sec=0.05)
            now = time.monotonic()
            if now - last_rv >= rv_period:
                last_rv = now
                node.poll_rviz()
    finally:
        node.close()
        executor.remove_node(node)
        node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
