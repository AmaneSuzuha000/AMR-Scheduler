"""把 Python 规划的全局路线交给 Nav2 跟踪。

分工
----
Python 决定"经过哪些通道/路口/工位"（离散拓扑上的全局路线），本节点只做翻译：
把节点序列按 GzFrame 换算成 Gazebo 世界坐标，串成一条 Path 发给 Nav2，
由 Nav2 的控制器算出连续速度。

本节点**不做**任何路由决策：不绕障、不改节点顺序、不重新规划。如果 Nav2 跟不下去
（被堵、偏离），正确的做法是上报 ROUTE_DEVIATION 让 Python 重新预约，而不是在这里
自己找一条新路——那样就把全局资源决策又散回 L3 了。

用法
----
    python3 nav2_route_follower.py --route J_S_8,J_S_25,J_S_45

    # 或从 stdio 逐行读节点序列（给 live_bridge 用）
    python3 nav2_route_follower.py --stdin
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path as FilePath
import sys
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.node import Node

sys.path.insert(0, str(FilePath(__file__).resolve().parent))
from gz_frame import DEMO_FRAME  # noqa: E402
from nav2_path import sample_polyline,shift_tracking_reference  # noqa: E402

# 场景文件在仓库根下的 amr_sched/，和 Windows 侧调度器读的是同一份。
_REPO = str(FilePath(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


class RouteFollower(Node):
    """Sends one global route to Nav2 and tracks its progress."""

    def __init__(self, frame_id: str = "odom", path_spacing_m: float = 0.05,
                 coordinate_frame=DEMO_FRAME,namespace: str = '',tracking_offset_m: float = 0.,
                 max_tracking_offset_m: float = 0.315) -> None:
        shift_tracking_reference([],tracking_offset_m,max_offset_m=max_tracking_offset_m)  # validate before creating ROS state
        topics=('tf','tf_static','scan','imu/data','odometry/filtered','amcl_pose',
                'request_nomotion_update','motion_guard/authorize','follow_path/_action/cancel_goal',
                'motion_guard/lease','speed_limit','cmd_vel','motion_guard/status','follow_path/_action/status')
        args=['--ros-args'] if namespace else []
        for topic in topics if namespace else ():
            args += ['-r','/'+topic+':=/'+namespace+'/'+topic]
        super().__init__("nav2_route_follower",namespace=namespace,cli_args=args)
        self.frame_id = frame_id
        self.path_spacing_m = path_spacing_m
        self.coordinate_frame = coordinate_frame
        self.tracking_offset_m = tracking_offset_m
        self.max_tracking_offset_m = max_tracking_offset_m
        self._client = ActionClient(self, FollowPath, "follow_path")
        self._result: dict | None = None
        self._done = threading.Event()

    def current_pose(self) -> tuple[float, float] | None:
        """机器人在 odom 下的当前位置。路径必须从它开始。"""
        from tf2_ros import Buffer, TransformListener
        if not hasattr(self, "_tf"):
            self._tf = Buffer()
            self._listener = TransformListener(self._tf, self)
        for _ in range(50):
            try:
                # 底盘帧用 base_footprint：odom->base_footprint 是真值 TF 发布的，
                # odom->base_link 不存在（URDF 里 base_link 挂在 base_footprint 下）。
                tr = self._tf.lookup_transform(self.frame_id, "base_footprint",
                                               rclpy.time.Time())
                t = tr.transform.translation
                return (t.x, t.y)
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.1)
        return None

    def build_path(self, nodes: list[str], network, prepend_pose=None, *, path_xy=None) -> Path:
        """Convert the supplied route to a spatially sampled Nav2 Path.

        GzFrame converts first; sampling then uses WORLD metres. Retain all route
        vertices and subdivide only their original segments. RPP needs nearby
        samples so nearest-point pruning cannot retain a distant point behind
        the vehicle. prepend_pose keeps the existing start-connector behaviour;
        callers remain responsible for approving that connector's corridor.
        """
        msg = Path()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        if not nodes:
            raise ValueError("route must contain at least one node")
        points = []
        if prepend_pose is not None:
            points.append((float(prepend_pose[0]), float(prepend_pose[1])))
        geometry = network.route_centerline(nodes) if path_xy is None else path_xy
        points.extend(self.coordinate_frame.to_world(x,y) for x,y in geometry)
        samples=sample_polyline(points,self.path_spacing_m)
        for x, y, yaw in shift_tracking_reference(samples,self.tracking_offset_m,max_offset_m=getattr(self,'max_tracking_offset_m',0.315)):
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.header.stamp = msg.header.stamp
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            msg.poses.append(pose)
        return msg

    def follow(self, path: Path, timeout: float = 120.0) -> dict:
        if not self._client.wait_for_server(timeout_sec=15.0):
            return {"ok": False, "error": "Nav2 follow_path 动作服务器不在"}
        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = "FollowPath"
        goal.goal_checker_id = "general_goal_checker"
        send = self._client.send_goal_async(goal, feedback_callback=self._feedback)
        # 控制器要先激活代价地图、等 TF，第一次收目标常在 10 s 之后才回应。
        # 之前给 10 s，客户端先超时，服务端再回话就没人接（Failed to send goal response）。
        rclpy.spin_until_future_complete(self, send, timeout_sec=45.0)
        handle = send.result()
        if handle is None or not handle.accepted:
            return {"ok": False, "error": "Nav2 拒绝了这条路径"}
        res = handle.get_result_async()
        deadline = time.time() + timeout
        while not res.done() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
        if not res.done():
            handle.cancel_goal_async()
            return {"ok": False, "error": "跟踪超时 %.0fs" % timeout}
        status = res.result().status
        return {"ok": status == 4, "status": int(status),
                "poses": len(path.poses)}

    def _feedback(self, msg) -> None:
        self._result = {"distance_remaining": float(getattr(msg.feedback, "distance_to_goal", -1.0))}


def load_network():
    """场景网络。和 Python 调度器读的是同一份 scene 文件。"""
    from amr_sched.workshop import WorkshopScene
    return WorkshopScene.load().network()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="", help="逗号分隔的节点 id 序列")
    ap.add_argument("--stdin", action="store_true", help="从标准输入逐行读节点序列")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--path-spacing", type=float, default=0.05,
                    help="maximum Path sample spacing in Gazebo world metres (default: 0.05)")
    ap.add_argument("--dry-run", action="store_true", help="只打印换算后的坐标，不下发")
    args = ap.parse_args(argv)
    if not math.isfinite(args.path_spacing) or args.path_spacing <= 0:
        ap.error("--path-spacing must be finite and positive")

    network = load_network()
    rclpy.init()
    node = RouteFollower(path_spacing_m=args.path_spacing)
    try:
        if args.dry_run:
            nodes = [n for n in args.route.split(",") if n]
            path = node.build_path(nodes, network)
            # Keep the original node-coordinate list; expose the full sampled Path
            # separately instead of zipping node IDs with the now denser samples.
            out = [{"node": n, "x": DEMO_FRAME.to_world(network.nodes[n].x, network.nodes[n].y)[0],
                    "y": DEMO_FRAME.to_world(network.nodes[n].x, network.nodes[n].y)[1]} for n in nodes]
            samples = [{"x": p.pose.position.x, "y": p.pose.position.y,
                        "qz": p.pose.orientation.z, "qw": p.pose.orientation.w} for p in path.poses]
            print(json.dumps({"frame": node.frame_id, "poses": out, "path_samples": samples,
                              "path_spacing_m": args.path_spacing}, ensure_ascii=False))
            return 0
        routes = []
        if args.stdin:
            for line in sys.stdin:
                line = line.strip()
                if line:
                    routes.append([n for n in line.split(",") if n])
        else:
            routes.append([n for n in args.route.split(",") if n])
        for nodes in routes:
            start = node.current_pose()
            if start is None:
                print(json.dumps({"ok": False, "error": "读不到 odom->base_link"}), flush=True)
                continue
            path = node.build_path(nodes, network, prepend_pose=start)
            result = node.follow(path, timeout=args.timeout)
            result["route"] = nodes
            result["first"] = nodes[0] if nodes else None
            result["last"] = nodes[-1] if nodes else None
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
