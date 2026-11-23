"""Full closed-loop WSL2 runner for wheelset_workshop_v2: Gazebo + 3x wheelset_agv + ROS2 + EKF/AMCL + Nav2 + V18_EA_JOINT + RViz2."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
import os
for _env_k, _env_v in (
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("TORCH_NUM_THREADS", "1"),
    ("LP_NUM_THREADS", "2"),
    ("GALLIUM_THREAD", "1"),
):
    os.environ.setdefault(_env_k, _env_v)

from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
import yaml

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "tools/live_bridge")):
    if p not in sys.path:
        sys.path.insert(0, p)

import rclpy  # noqa: E402
from geometry_msgs.msg import PoseWithCovarianceStamped  # noqa: E402
from lifecycle_msgs.srv import GetState  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from sensor_msgs.msg import Imu, LaserScan  # noqa: E402
from std_srvs.srv import Empty  # noqa: E402
from gazebo_msgs.srv import GetEntityState, SetEntityState, SpawnEntity  # noqa: E402
from amr_gazebo_guard.msg import GuardStatus  # noqa: E402

from amr_sched.allocation.greedy import GreedyAllocator  # noqa: E402
from amr_sched.baselines.policy_registry import resolve_policy_name, summarize_physical_run  # noqa: E402
from amr_sched.cvcr.auditor import IndependentAuditor  # noqa: E402
from amr_sched.cvcr.physical_bridge import (  # noqa: E402
    NativeProductionTracker,
    OperationSnapshot,
    PhysicalCVCRCoordinator,
)
from amr_sched.entities import Agv, Task  # noqa: E402
from amr_sched.external_execution import ExternalExecution  # noqa: E402
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner  # noqa: E402
from amr_sched.simulation import SimulationConfig, Simulator  # noqa: E402
from amr_sched.workshop import load_and_validate  # noqa: E402

from fleet_config import point_execution_network, robot_configs  # noqa: E402
from fleet_scheduler import FleetRuntime  # noqa: E402
from map01_live_helpers import Map01V18Observer  # noqa: E402
from tools.wheelset_scene.wheelset_v2_agv_model import TRACKING_REFERENCE_X  # noqa: E402
from tools.wheelset_scene.wheelset_v2_live_helpers import (  # noqa: E402
    GroundedWheelsetWorkpieceSync,
    build_wheelset_v2_nav_and_loc_templates,
    write_wheelset_v2_rviz_config,
)
from tools.wheelset_scene.workshop_v2_layout import compile_v2_bundle  # noqa: E402


class _CarrierStartupProbe(Node):
    """Lightweight startup probe verifying /clock, entity spawn, sensors, and motion guard."""

    def __init__(self, robot_id: str, namespace: str) -> None:
        super().__init__(f"v2_probe_{namespace}", use_global_arguments=False)
        self.robot_id = robot_id
        self.namespace = namespace
        self.clock_s: float | None = None
        self.odom_ok = False
        self.imu_ok = False
        self.scan_ok = False
        self.guard_ok = False
        self.entity_ok = False
        self._truth_client = self.create_client(GetEntityState, "/get_entity_state")
        self._truth_pending = None
        self._last_truth_poll = -math.inf

        self.create_subscription(Clock, "/clock", self._on_clock, qos_profile_sensor_data)
        self.create_subscription(Odometry, f"/{namespace}/wheel/odometry", self._on_odom, 10)
        self.create_subscription(Imu, f"/{namespace}/imu/data", self._on_imu, 10)
        self.create_subscription(LaserScan, f"/{namespace}/scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(GuardStatus, f"/{namespace}/motion_guard/status", self._on_guard, 10)

        self.initial_pub = self.create_publisher(PoseWithCovarianceStamped, f"/{namespace}/initialpose", 1)
        self.nomotion = self.create_client(Empty, f"/{namespace}/request_nomotion_update")

    def _on_clock(self, msg: Clock) -> None:
        self.clock_s = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _on_odom(self, _msg: Odometry) -> None:
        self.odom_ok = True

    def _on_imu(self, _msg: Imu) -> None:
        self.imu_ok = True

    def _on_scan(self, _msg: LaserScan) -> None:
        self.scan_ok = True

    def _on_guard(self, _msg: GuardStatus) -> None:
        self.guard_ok = True

    def publish_initialpose(self, xy: tuple[float, float], yaw: float) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        if self.clock_s is not None:
            sec = int(self.clock_s)
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = int((self.clock_s - sec) * 1e9)
        msg.pose.pose.position.x = float(xy[0])
        msg.pose.pose.position.y = float(xy[1])
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        msg.pose.covariance[0] = 0.0025
        msg.pose.covariance[7] = 0.0025
        msg.pose.covariance[35] = 0.001
        self.initial_pub.publish(msg)

    def poll_truth(self, now_wall: float) -> None:
        if self.entity_ok or self.clock_s is None:
            return
        if self._truth_pending is not None:
            if not self._truth_pending.done():
                return
            try:
                res = self._truth_pending.result()
                if res is not None and res.success:
                    self.entity_ok = True
            except Exception:
                pass
            self._truth_pending = None
            return
        if now_wall - self._last_truth_poll < 1.0 or not self._truth_client.service_is_ready():
            return
        self._last_truth_poll = now_wall
        req = GetEntityState.Request()
        req.name = self.namespace
        req.reference_frame = "world"
        self._truth_pending = self._truth_client.call_async(req)

    def ready(self, now_wall: float) -> bool:
        self.poll_truth(now_wall)
        return (
            self.clock_s is not None
            and self.entity_ok
            and self.odom_ok
            and self.imu_ok
            and self.scan_ok
            and self.guard_ok
        )


def build_wheelset_v2_v18_scenario(bundle_dir: Path, *, policy: str = "V18_EA_JOINT", seed: int = 0):
    """Build 3-AGV Wheelset Workshop V2 simulation, grounded tasks, operations, and pluggable CVCR coordinator."""
    policy_canon = resolve_policy_name(policy)
    summary = compile_v2_bundle(bundle_dir, resolution=0.10)
    sc, issues = load_and_validate(bundle_dir / "wheelset_workshop_v2_nav.json")
    if issues:
        raise RuntimeError(f"wheelset_workshop_v2 scene issues: {issues}")
    net = point_execution_network(sc.network())

    robots_spec = {
        "AGV_1": {"namespace": "agv_1", "start": "J_S_65", "yaw": math.pi / 2.0},
        "AGV_2": {"namespace": "agv_2", "start": "J_S_90", "yaw": math.pi / 2.0},
        "AGV_3": {"namespace": "agv_3", "start": "J_S_142", "yaw": math.pi / 2.0},
    }
    agvs = [Agv(rid, max_speed=0.65, start_node=spec["start"], load_time=2.0, unload_time=2.0) for rid, spec in robots_spec.items()]

    tasks = [
        Task("T_01_TD_INSP", pickup="D_TEARDOWN_0", dropoff="D_INSPECTION_0", release_step=0, due_step=260),
        Task("T_02_INSP_PREP", pickup="D_INSPECTION_1", dropoff="D_PREP_IN", release_step=0, due_step=280),
        Task("T_03_PREP_PRESS", pickup="D_PREP_OUT", dropoff="D_PRESS_IN", release_step=0, due_step=260),
        Task("T_04_PRESS_CLOSE", pickup="D_PRESS_OUT", dropoff="D_CLOSING_IN", release_step=0, due_step=220),
        Task("T_05_FIN_STORE_A", pickup="D_FINISHING_0", dropoff="D_STORAGE", release_step=0, due_step=280),
        Task("T_06_FIN_STORE_B", pickup="D_FINISHING_1", dropoff="D_UNKNOWN", release_step=0, due_step=340),
    ]
    ops = [
        OperationSnapshot("OP_V2_INSP", ("T_01_TD_INSP",), "D_INSPECTION_0", crew_id=1, tool_id=1, proc_steps=5, due_step=280),
        OperationSnapshot("OP_V2_PREP", ("T_02_INSP_PREP",), "D_PREP_IN", crew_id=2, tool_id=2, proc_steps=5, due_step=320, predecessor_op_id="OP_V2_INSP"),
        OperationSnapshot("OP_V2_PRESS", ("T_03_PREP_PRESS",), "D_PRESS_IN", crew_id=3, tool_id=3, proc_steps=5, due_step=360, predecessor_op_id="OP_V2_PREP"),
        OperationSnapshot("OP_V2_CLOSE", ("T_04_PRESS_CLOSE",), "D_CLOSING_IN", crew_id=2, tool_id=2, proc_steps=5, due_step=400, predecessor_op_id="OP_V2_PRESS"),
        OperationSnapshot("OP_V2_STORE", ("T_05_FIN_STORE_A",), "D_STORAGE", crew_id=1, tool_id=4, proc_steps=5, due_step=480, predecessor_op_id="OP_V2_CLOSE"),
        OperationSnapshot("OP_V2_VERIFY", ("T_06_FIN_STORE_B",), "D_UNKNOWN", crew_id=3, tool_id=5, proc_steps=5, due_step=520, predecessor_op_id="OP_V2_STORE"),
    ]
    cfg = SimulationConfig(
        dt=1.0,
        max_steps=1800,
        planning_horizon=2200,
        record_trajectory=True,
        load_time=2.0,
        unload_time=2.0,
        idle_retreat=False,
        deadlock_timeout=0,
    )
    run_id = uuid.uuid4().hex
    gates = {
        rid: ExternalExecution(
            "wheelset_workshop_v2",
            rid,
            (net.nodes[spec["start"]].x, net.nodes[spec["start"]].y),
            source="localized",
            run_id=run_id,
            namespace_routes=True,
            arrival_tolerance=0.25,
            timeout=2.0,
        )
        for rid, spec in robots_spec.items()
    }
    sim = Simulator(
        net,
        tasks,
        agvs,
        SpaceTimeAStarPlanner(max_speed=0.65, dt=1.0),
        GreedyAllocator(),
        cfg,
        executions=gates,
    )
    sim.reset()
    tracker = NativeProductionTracker(ops, sim.tasks)
    coord = PhysicalCVCRCoordinator(
        sim,
        ops,
        production_tracker=tracker,
        policy_method=policy_canon,
        budget_ms=20.0,
        epoch_interval=20,
        use_rust=True,
        seed=int(seed),
    )
    workpieces = {
        "ws_wp_01_teardown": {"chain": ["T_01_TD_INSP"], "hold_xy": (net.nodes["D_TEARDOWN_0"].x, net.nodes["D_TEARDOWN_0"].y)},
        "ws_wp_02_inspect": {"chain": ["T_02_INSP_PREP"], "hold_xy": (net.nodes["D_INSPECTION_1"].x, net.nodes["D_INSPECTION_1"].y)},
        "ws_wp_03_prep": {"chain": ["T_03_PREP_PRESS"], "hold_xy": (net.nodes["D_PREP_OUT"].x, net.nodes["D_PREP_OUT"].y)},
        "ws_wp_04_press": {"chain": ["T_04_PRESS_CLOSE"], "hold_xy": (net.nodes["D_PRESS_OUT"].x, net.nodes["D_PRESS_OUT"].y)},
        "ws_wp_05_fin_a": {"chain": ["T_05_FIN_STORE_A"], "hold_xy": (net.nodes["D_FINISHING_0"].x, net.nodes["D_FINISHING_0"].y)},
        "ws_wp_06_fin_b": {"chain": ["T_06_FIN_STORE_B"], "hold_xy": (net.nodes["D_FINISHING_1"].x, net.nodes["D_FINISHING_1"].y)},
    }
    return summary, sc, net, sim, coord, ops, robots_spec, workpieces


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--schedule-port", type=int, default=9502)
    p.add_argument("--max-wall-s", type=float, default=900.0)
    p.add_argument("--wait-wpf-s", type=float, default=55.0)
    p.add_argument("--policy", type=str, default="V18_EA_JOINT")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-rviz", action="store_true")
    p.add_argument("--no-camera-capture", action="store_true")
    args = p.parse_args()
    requested_policy = str(args.policy)
    policy_canon = resolve_policy_name(requested_policy)

    out = args.out.resolve()
    bundle_dir = out / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    master_port = int(os.environ.get("GAZEBO_MASTER_URI", "http://127.0.0.1:11363").rsplit(":", 1)[-1])
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", master_port)) == 0:
            raise RuntimeError(f"Gazebo master port {master_port} is already occupied")

    summary, sc, net, sim, coord, ops, robots_spec, workpieces = build_wheelset_v2_v18_scenario(
        bundle_dir, policy=policy_canon, seed=args.seed
    )
    world_path = bundle_dir / "wheelset_workshop_v2.world"
    map_yaml = bundle_dir / "map.yaml"
    loc_base, nav_base = build_wheelset_v2_nav_and_loc_templates()
    agv_launch = ROOT / "tools/wheelset_scene/wheelset_agv_metric.launch.py"

    processes: list[subprocess.Popen] = []
    logs = []
    _rt_root = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path("/tmp")
    fast_rt = _rt_root / f"wheelset_v2_rt_{os.getpid()}"
    shutil.rmtree(fast_rt, ignore_errors=True)
    fast_rt.mkdir(parents=True, exist_ok=True)
    capture_dir = out / "capture"
    fast_capture_dir = fast_rt / "capture"
    capture_stop = fast_rt / "capture_stop.flag"

    def start_proc(cmd: list[str], name: str) -> subprocess.Popen:
        log = (out / f"{name}.log").open("w", encoding="utf-8")
        logs.append(log)
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(proc)
        return proc

    rclpy.init()
    executor = SingleThreadedExecutor()
    probes: dict[str, _CarrierStartupProbe] = {}
    inspector = Node("wheelset_v2_chain_inspector", use_global_arguments=False)
    executor.add_node(inspector)
    runtime = None
    v18_obs = None
    wp_sync = None
    error_text = None

    def spin_burst(first_timeout: float = 0.01, max_callbacks: int = 4096, max_wall_s: float = 0.15, max_passes: int = 10) -> None:
        from rclpy.executors import ShutdownException, TimeoutException
        dl = time.monotonic() + max_wall_s
        total = 0
        if first_timeout > 0.0:
            if hasattr(executor, "_cb_iter"):
                executor._cb_iter = None
            try:
                handler, _, _ = executor.wait_for_ready_callbacks(timeout_sec=first_timeout)
            except (TimeoutException, ShutdownException):
                return
            total += 1
            handler()
            if handler.exception() is not None:
                raise handler.exception()
        for _ in range(max_passes):
            if total >= max_callbacks or time.monotonic() >= dl:
                break
            if hasattr(executor, "_cb_iter"):
                executor._cb_iter = None
            yielded = 0
            while total < max_callbacks and time.monotonic() < dl:
                try:
                    handler, _, _ = executor.wait_for_ready_callbacks(timeout_sec=0.0)
                except (TimeoutException, ShutdownException):
                    break
                yielded += 1
                total += 1
                handler()
                if handler.exception() is not None:
                    raise handler.exception()
            if yielded == 0:
                break

    def pump(seconds: float) -> None:
        dl = time.monotonic() + seconds
        while time.monotonic() < dl:
            spin_burst(first_timeout=0.01)

    def until(pred, timeout: float, reason: str) -> None:
        dl = time.monotonic() + timeout
        while not pred():
            if time.monotonic() >= dl:
                raise RuntimeError(reason)
            spin_burst(first_timeout=0.02)

    def start_active(cmd: list[str], name: str, namespace: str, expected: list[str]) -> None:
        probe = probes[next(r for r, d in robots_spec.items() if d["namespace"] == namespace)]
        for attempt in range(3):
            proc = start_proc(cmd, name + (f"_retry{attempt}" if attempt else ""))
            clients = {m: probe.create_client(GetState, f"/{namespace}/{m}/get_state") for m in expected}
            states = {m: None for m in expected}
            dl = time.monotonic() + 30.0
            while proc.poll() is None and time.monotonic() < dl:
                for m, cl in clients.items():
                    if not cl.service_is_ready():
                        continue
                    fut = cl.call_async(GetState.Request())
                    rdl = time.monotonic() + 0.5
                    while not fut.done() and time.monotonic() < rdl:
                        spin_burst(first_timeout=0.01)
                    if fut.done() and fut.result() is not None:
                        states[m] = fut.result().current_state.id
                if all(st == 3 for st in states.values()):
                    for cl in clients.values():
                        probe.destroy_client(cl)
                    return
                pump(0.2)
            for cl in clients.values():
                probe.destroy_client(cl)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=3.0)
        raise RuntimeError(f"Lifecycle nodes failed to reach ACTIVE state for {namespace}: {expected}")

    try:
        for idx, (rid, spec) in enumerate(robots_spec.items()):
            ns = spec["namespace"]
            nd = net.nodes[spec["start"]]
            loc_cfg, nav_cfg = robot_configs(
                ns, loc_base, nav_base, dock_tolerance_m=0.10, dock_yaw_tolerance_rad=0.20
            )
            (out / f"{ns}_localization.yaml").write_text(yaml.safe_dump(loc_cfg), encoding="utf-8")
            (out / f"{ns}_controller.yaml").write_text(yaml.safe_dump(nav_cfg), encoding="utf-8")
            pr = _CarrierStartupProbe(rid, ns)
            probes[rid] = pr
            executor.add_node(pr)
            start_proc(
                [
                    "ros2",
                    "launch",
                    str(agv_launch),
                    f"world:={world_path}",
                    "lockstep:=false",
                    f"robot_namespace:={ns}",
                    f"start_world:={'true' if idx == 0 else 'false'}",
                    "front_tracking:=true",
                    "laser_range_max:=60.0",
                    f"laser_fov_radians:={4.18879008}",
                    f"x:={nd.x}",
                    f"y:={nd.y}",
                    "z:=0.02",
                    f"yaw:={spec['yaw']}",
                    "odom_topic:=wheel/odometry",
                    "odom_tf:=false",
                    "imu_gyro_noise:=0.002",
                    "spawn_timeout:=360",
                    f"model_out:={out / (ns + '.urdf')}",
                ],
                f"{ns}_model",
            )

        until(
            lambda: all(pr.ready(time.monotonic()) for pr in probes.values()),
            380.0,
            "Wheelset V2 carrier entities, clock, sensors, or guard failed startup readiness",
        )
        pump(1.5)

        spawn_cli = inspector.create_client(SpawnEntity, "/spawn_entity")
        get_cli = inspector.create_client(GetEntityState, "/get_entity_state")
        set_cli = inspector.create_client(SetEntityState, "/set_entity_state")
        wp_sync = GroundedWheelsetWorkpieceSync(
            workpieces,
            {t.task_id: t for t in sim.tasks},
            net,
            out / "wheelset_transfer_log.jsonl",
            spawn_client=spawn_cli,
            get_state_client=get_cli,
            set_state_client=set_cli,
            ns_by_agv={r: s["namespace"] for r, s in robots_spec.items()},
        )
        wp_sync.spawn_all(executor)

        for rid, spec in robots_spec.items():
            ns = spec["namespace"]
            start_active(
                [
                    "ros2", "launch", "amr_gazebo_guard", "sensor_localization.launch.py",
                    f"config:={out / (ns + '_localization.yaml')}",
                    f"robot_namespace:={ns}",
                    f"map:={map_yaml}",
                    "laser_max_range:=60.0",
                    "laser_model_type:=likelihood_field_prob",
                ],
                f"{ns}_localization",
                ns,
                ["map_server", "amcl"],
            )

        until(
            lambda: all(pr.initial_pub.get_subscription_count() > 0 for pr in probes.values()),
            30.0,
            "AMCL initialpose subscribers unavailable",
        )
        for rid, spec in robots_spec.items():
            nd = net.nodes[spec["start"]]
            probes[rid].publish_initialpose((nd.x, nd.y), spec["yaw"])
        for _ in range(8):
            for pr in probes.values():
                if pr.nomotion.service_is_ready():
                    fut = pr.nomotion.call_async(Empty.Request())
                    dl = time.monotonic() + 0.5
                    while not fut.done() and time.monotonic() < dl:
                        executor.spin_once(timeout_sec=0.01)
            pump(0.5)

        for rid, spec in robots_spec.items():
            ns = spec["namespace"]
            start_active(
                [
                    "ros2", "launch", "amr_gazebo_guard", "metric_controller.launch.py",
                    f"config:={out / (ns + '_controller.yaml')}",
                    f"robot_namespace:={ns}",
                ],
                f"{ns}_controller",
                ns,
                ["controller_server"],
            )
        pump(1.5)

        if not args.no_rviz:
            rviz_cfg = write_wheelset_v2_rviz_config(
                out / "wheelset_v2_live.rviz",
                robot_namespaces=tuple(s["namespace"] for s in robots_spec.values()),
                robot_ids=tuple(robots_spec.keys()),
            )
            start_proc(
                ["/opt/ros/humble/lib/rviz2/rviz2", "-d", str(rviz_cfg), "--ros-args", "-p", "use_sim_time:=true"],
                "rviz2",
            )

        if not args.no_camera_capture:
            start_proc(
                [
                    sys.executable,
                    str(ROOT / "tools/live_bridge/capture_map01_wsl_views.py"),
                    "--out", str(fast_capture_dir),
                    "--camera-topic", "/wheelset_v2_overview/camera/image_raw",
                    "--rviz-match", "wheelset_v2_live.rviz",
                    "--stop-file", str(capture_stop),
                    "--rviz-hz", "2.0",
                ],
                "wsl_capture",
            )

        for pr in probes.values():
            executor.remove_node(pr)
            pr.destroy_node()
        probes.clear()

        runtime = FleetRuntime(
            sim,
            fast_rt,
            {r: s["namespace"] for r, s in robots_spec.items()},
            executor,
            runtime_files=[Path(__file__), ROOT / "tools/wheelset_scene/wheelset_v2_live_helpers.py"],
            tracking_offset_m=TRACKING_REFERENCE_X,
            max_tracking_offset_m=0.85,
            latest_tf=True,
            input_thread=False,
            async_logs=False,
            cvcr_coordinator=coord,
            defer_ingress=True,
        )
        bound_port = runtime.start_control(port=args.schedule_port, host="0.0.0.0")
        v18_obs = Map01V18Observer(coord, ops, runtime.record)
        v18_obs.start_display(executor)

        try:
            _wsl_ips = subprocess.check_output(["hostname", "-I"], text=True).split()
            _wsl_ip = _wsl_ips[0] if _wsl_ips else "127.0.0.1"
        except Exception:
            _wsl_ip = "127.0.0.1"
        ready_tmp = out / "control_ready.json.tmp"
        ready_tmp.write_text(
            json.dumps({
                "host": _wsl_ip,
                "port": bound_port,
                "scene_id": "wheelset_workshop_v2",
                "robots": list(robots_spec),
            }),
            encoding="utf-8",
        )
        ready_tmp.replace(out / "control_ready.json")
        runtime.start_ingress()
        runtime.start_owner_pump()

        saw_wpf_client = False
        wait_dl = time.monotonic() + max(1.0, args.wait_wpf_s)
        while time.monotonic() < wait_dl:
            spin_burst(first_timeout=0.01)
            runtime.pump_owner()
            v18_obs.publish(time.monotonic())
            wp_sync.step(sim, sim.executions)
            if runtime.control_server is not None and runtime.control_server.clients:
                saw_wpf_client = True
            if saw_wpf_client and (args.no_rviz or v18_obs.saw_rviz_subscription):
                break

        runtime.enable()
        started = time.monotonic()
        completed = False
        while time.monotonic() - started < args.max_wall_s:
            spin_burst(first_timeout=0.01)
            runtime.pump_owner()
            now_w = time.monotonic()
            v18_obs.publish(now_w)
            wp_sync.step(sim, sim.executions)
            if not saw_wpf_client and runtime.control_server is not None and runtime.control_server.clients:
                saw_wpf_client = True
            if any(g.fault for g in sim.executions.values()):
                raise RuntimeError("Fleet execution latched fault: " + str({r: g.fault for r, g in sim.executions.items() if g.fault}))
            if sim._all_done() and all(n.clean_idle_shutdown_ready() for n in runtime.nodes.values()):
                while not coord.tracker.all_done(sim._step_cursor) and sim._step_cursor < sim.config.max_steps:
                    sim._step_cursor += 1
                    coord.tracker.step(sim._step_cursor)
                    v18_obs.publish(time.monotonic())
                    wp_sync.step(sim, sim.executions)
                    runtime.pump_owner()
                    pump(0.05)
                completed = coord.tracker.all_done(sim._step_cursor)
                break
        if not completed:
            raise RuntimeError("wheelset_workshop_v2 full-chain timed out before all tasks and operations finished")

        aud = IndependentAuditor.audit_simulation_run(sim, ops, require_trajectory=True)
        v18_sum = v18_obs.close(executor) if v18_obs else {}
        wp_sum = wp_sync.finalize()
        pol_sum = summarize_physical_run(coord, ops)
        report = {
            "passed": bool(
                completed
                and aud.pass_audit
                and v18_sum.get("accepted", 0) > 0
                and wp_sum["grounded_passed"]
                and summary["clearance_audit"]["passed"]
            ),
            "scene_id": "wheelset_workshop_v2",
            "policy": coord.policy_method,
            "requested_policy": requested_policy,
            "seed": int(args.seed),
            "sim_steps": int(sim._step_cursor),
            "sim_time_s": round(sim._step_cursor * sim.config.dt, 2),
            "tasks_completed": sum(1 for t in sim.tasks if getattr(t.status, "value", str(t.status)) == "completed"),
            "tasks_total": len(sim.tasks),
            "ops_completed": sum(1 for o in coord.tracker.operations if o.end_step is not None),
            "ops_total": len(ops),
            "clearance_audit": summary["clearance_audit"],
            "grounded_workpieces": wp_sum,
            "v18": v18_sum,
            "policy_metrics": pol_sum,
            "audit": {
                "passed": aud.pass_audit,
                "issues": aud.issues,
                "conflict_events": aud.conflict_events,
                "invalid_edge_moves": aud.invalid_edge_moves,
                "makespan": aud.independent_makespan,
            },
            "wpf_connected_ever": saw_wpf_client,
        }
        (out / "full_chain_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if not report["passed"]:
            raise RuntimeError(f"wheelset_workshop_v2 audit failed: {report}")
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        error_text = repr(exc)
        (out / "full_chain_error.txt").write_text(error_text + "\n", encoding="utf-8")
        raise
    finally:
        capture_stop.write_text("stop\n", encoding="utf-8")
        if wp_sync is not None and not wp_sync.log_path.is_file():
            try:
                wp_sync.finalize()
            except Exception:
                pass
        if runtime is not None:
            try:
                runtime.begin_close()
                for _ in range(15):
                    executor.spin_once(timeout_sec=0.05)
                runtime.close()
            except Exception:
                pass
        for pr in list(probes.values()):
            try:
                executor.remove_node(pr)
                pr.destroy_node()
            except Exception:
                pass
        for proc in processes:
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                except Exception:
                    pass
        t_end = time.monotonic() + 5.0
        for proc in processes:
            rem = max(0.1, t_end - time.monotonic())
            if proc.poll() is None:
                try:
                    proc.wait(timeout=rem)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        proc.kill()
        for lg in logs:
            try:
                lg.close()
            except Exception:
                pass
        try:
            executor.remove_node(inspector)
            inspector.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
        try:
            for item in fast_rt.iterdir():
                if item.is_file() and item.name.endswith((".jsonl", ".json", ".lock")):
                    shutil.copy2(item, out / item.name)
            if fast_capture_dir.is_dir():
                capture_dir.mkdir(parents=True, exist_ok=True)
                for sub in fast_capture_dir.iterdir():
                    dst = capture_dir / sub.name
                    if sub.is_dir():
                        shutil.copytree(sub, dst, dirs_exist_ok=True)
                    elif sub.is_file():
                        shutil.copy2(sub, dst)
        finally:
            shutil.rmtree(fast_rt, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
