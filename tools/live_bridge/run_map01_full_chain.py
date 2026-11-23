"""Run full 6-robot MAP-01 62-node Gazebo + Nav2 + Guard + V18_EA_JOINT + RViz2 + WPF chain."""
from __future__ import annotations

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

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "live_bridge"))

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, LaserScan
from std_srvs.srv import Empty
import yaml

from amr_gazebo_guard.msg import GuardStatus
from gazebo_msgs.srv import GetEntityState
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.baselines.policy_registry import resolve_policy_name, summarize_physical_run
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.physical_bridge import PhysicalCVCRCoordinator
from amr_sched.cvcr.production import OperationSnapshot
from amr_sched.entities import Agv, MaterialType, Task, TaskPriority
from amr_sched.external_execution import ExternalExecution
from amr_sched.geometry import project_polyline  # noqa: F401 - warm lazy import before guard lease begins
from amr_sched.planners.base import PathPlanner
from amr_sched.planners.rust_spacetime_astar import RustSpaceTimeAStarPlanner, is_rust_planner_available
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.workshop import WorkshopScene

from fleet_config import point_execution_network, robot_configs
from fleet_scheduler import FleetRuntime
from map01_live_helpers import Map01V18Observer, derive_map01_world, write_map01_rviz_config

import weakref as _weakref

_RUST_PLANNER_BY_INSTANCE: _weakref.WeakKeyDictionary[object, object] = _weakref.WeakKeyDictionary()


class _CheckpointSafeRustPlanner(PathPlanner):
    """Rust SIPP/A* planner wrapper keeping `vars(self)` checkpoint-serializable and deepcopy-safe."""

    name = "spacetime_astar"
    reservations_aware = True

    def __init__(
        self,
        max_speed: float = 0.2,
        dt: float = 0.1,
        max_expansions: int = 200_000,
        use_sipp: bool = True,
    ) -> None:
        super().__init__(max_speed=max_speed, dt=dt)
        self.max_expansions = int(max_expansions)
        self.use_sipp = bool(use_sipp)

    def _backend(self):
        b = _RUST_PLANNER_BY_INSTANCE.get(self)
        if b is None:
            if is_rust_planner_available():
                b = RustSpaceTimeAStarPlanner(
                    max_speed=self.max_speed,
                    dt=self.dt,
                    max_expansions=self.max_expansions,
                    use_sipp=self.use_sipp,
                )
            else:
                b = SpaceTimeAStarPlanner(
                    max_speed=self.max_speed,
                    dt=self.dt,
                    max_expansions=self.max_expansions,
                )
            _RUST_PLANNER_BY_INSTANCE[self] = b
        return b

    def sync_traffic_weights(self, *args, **kwargs):
        b = self._backend()
        if hasattr(b, "sync_traffic_weights"):
            return b.sync_traffic_weights(*args, **kwargs)
        return None

    def sync_agv_guides(self, *args, **kwargs):
        b = self._backend()
        if hasattr(b, "sync_agv_guides"):
            return b.sync_agv_guides(*args, **kwargs)
        return None

    def plan(self, *args, **kwargs):
        return self._backend().plan(*args, **kwargs)


def _stamp(msg_stamp) -> float:
    return float(msg_stamp.sec) + float(msg_stamp.nanosec) * 1e-9


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RobotStartupProbe(Node):
    """Startup-only ROS 2 health, entity pose check, initialpose, and nomotion trigger."""

    def __init__(self, robot: str, namespace: str):
        super().__init__(f"{namespace}_startup_probe", use_global_arguments=False)
        self.robot = robot
        self.namespace = namespace
        self.clock_s = None
        self.inputs: dict[str, tuple[float, float]] = {}
        self.guard_stopped = False
        self.truth_xy = None
        self.amcl_xy = None
        self.pending = None
        self.last_truth_poll = 0.0
        self.initial_pub = self.create_publisher(PoseWithCovarianceStamped, f"/{namespace}/initialpose", 10)
        self.nomotion = self.create_client(Empty, f"/{namespace}/request_nomotion_update")
        self.truth = self.create_client(GetEntityState, "/get_entity_state")
        self.create_subscription(Clock, "/clock", lambda m: setattr(self, "clock_s", _stamp(m.clock)), qos_profile_sensor_data)
        self.create_subscription(LaserScan, f"/{namespace}/scan", lambda m: self._note("scan", m), qos_profile_sensor_data)
        self.create_subscription(Imu, f"/{namespace}/imu/data", lambda m: self._note("imu", m), qos_profile_sensor_data)
        self.create_subscription(Odometry, f"/{namespace}/wheel/odometry", lambda m: self._note("encoder", m), qos_profile_sensor_data)
        self.create_subscription(
            PoseWithCovarianceStamped,
            f"/{namespace}/amcl_pose",
            self._on_amcl,
            10,
        )
        self.create_subscription(GuardStatus, f"/{namespace}/motion_guard/status", self._on_guard, 10)

    def _note(self, kind: str, msg) -> None:
        self.inputs[kind] = (_stamp(msg.header.stamp), time.monotonic())

    def _on_amcl(self, msg: PoseWithCovarianceStamped) -> None:
        p = msg.pose.pose.position
        self.amcl_xy = (float(p.x), float(p.y))

    def amcl_converged(self, target_xy: tuple[float, float], tol_m: float = 0.12) -> bool:
        if self.amcl_xy is None:
            return False
        return math.hypot(self.amcl_xy[0] - target_xy[0], self.amcl_xy[1] - target_xy[1]) <= tol_m

    def _on_guard(self, msg: GuardStatus) -> None:
        self.guard_stopped = bool(msg.stopped and msg.output_vx == 0.0 and msg.output_wz == 0.0)

    def poll_truth(self) -> None:
        now = time.monotonic()
        if self.clock_s is None or now - self.last_truth_poll < 1.0:
            return
        if self.truth.service_is_ready() and (self.pending is None or self.pending.done()):
            self.last_truth_poll = now
            req = GetEntityState.Request()
            req.name = self.namespace
            req.reference_frame = "world"
            self.pending = self.truth.call_async(req)
            self.pending.add_done_callback(self._on_truth)

    def _on_truth(self, future) -> None:
        try:
            resp = future.result()
            if resp and resp.success:
                p = resp.state.pose.position
                self.truth_xy = (float(p.x), float(p.y))
        except Exception:
            pass

    def ready(self, now: float) -> bool:
        self.poll_truth()
        return (
            self.clock_s is not None
            and self.truth_xy is not None
            and self.guard_stopped
            and {"scan", "imu", "encoder"} <= set(self.inputs)
            and all(now - wall <= 2.0 for _, wall in self.inputs.values())
        )

    def publish_initialpose(self, xy: tuple[float, float], yaw_rad: float) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = rclpy.time.Time(seconds=self.clock_s or 0.0).to_msg()
        # Non-truth initial prior offset matching existing localization protocol
        msg.pose.pose.position.x = float(xy[0]) + 0.2
        msg.pose.pose.position.y = float(xy[1]) - 0.15
        iy = float(yaw_rad) + 0.1
        msg.pose.pose.orientation.z = math.sin(iy / 2.0)
        msg.pose.pose.orientation.w = math.cos(iy / 2.0)
        msg.pose.covariance[0] = msg.pose.covariance[7] = 0.09
        msg.pose.covariance[35] = 0.01
        self.initial_pub.publish(msg)


def build_map01_v18_scenario(*, policy: str = "V18_EA_JOINT", seed: int = 0):
    policy_canon = resolve_policy_name(policy)
    scene = WorkshopScene.load(ROOT / "scenes" / "rail_freight_workshop_v1.json")
    net = point_execution_network(scene.network())

    robots_spec = {
        "R1": {"namespace": "r1", "start": "J_S_65", "yaw": math.pi / 2},
        "R2": {"namespace": "r2", "start": "J_N_45", "yaw": 0.0},
        "R3": {"namespace": "r3", "start": "J_N_85", "yaw": 0.0},
        "R4": {"namespace": "r4", "start": "W_DOOR", "yaw": -math.pi / 2},
        "R5": {"namespace": "r5", "start": "W_COUPLER", "yaw": -math.pi / 2},
        "R6": {"namespace": "r6", "start": "W_PARK", "yaw": math.pi / 2},
    }
    tasks = [
        Task("K_BG_1", "Q_BOGIE", "D_BOGIE", release_step=0, due_step=1200, material_type=MaterialType.BOGIE),
        Task("K_WS_1", "J_S_65", "D_WHEELSET", release_step=0, due_step=2400, material_type=MaterialType.WHEELSET),
        Task("K_BR_1", "J_N_85", "D_BRAKE", release_step=0, due_step=1500, material_type=MaterialType.BRAKE_VALVE),
        Task("K_CP_1", "R_COUPLER", "D_COUPLER", release_step=0, due_step=1800, material_type=MaterialType.BRAKE_VALVE),
        Task("K_DR_1", "R_DOOR", "D_DOOR", release_step=10, due_step=2000, material_type=MaterialType.BRAKE_VALVE),
        Task("K_AS_1", "J_S_125", "D_ASSEMBLY", release_step=25, due_step=2600,
             priority=TaskPriority.URGENT, material_type=MaterialType.WHEELSET),
    ]
    ops = [
        OperationSnapshot("OP_BOGIE", bay_id="D_BOGIE", crew_id=1, tool_id=1, proc_steps=15, due_step=1500,
                          kit_task_ids=("K_BG_1",)),
        OperationSnapshot("OP_WHEELSET", bay_id="D_WHEELSET", crew_id=2, tool_id=2, proc_steps=15, due_step=2600,
                          predecessor_op_id="OP_BOGIE", kit_task_ids=("K_WS_1",)),
        OperationSnapshot("OP_BRAKE", bay_id="D_BRAKE", crew_id=3, tool_id=3, proc_steps=15, due_step=1800,
                          kit_task_ids=("K_BR_1",)),
        OperationSnapshot("OP_COUPLER", bay_id="D_COUPLER", crew_id=4, tool_id=4, proc_steps=15, due_step=2200,
                          eligible_bays=("D_COUPLER", "D_DOOR"), kit_task_ids=("K_CP_1",)),
        OperationSnapshot("OP_DOOR", bay_id="D_DOOR", crew_id=5, tool_id=5, proc_steps=15, due_step=2400,
                          kit_task_ids=("K_DR_1",)),
        OperationSnapshot("OP_ASSEMBLY", bay_id="D_ASSEMBLY", crew_id=6, tool_id=6, proc_steps=20, due_step=2800,
                          predecessor_op_id="OP_WHEELSET", is_urgent=True, kit_task_ids=("K_AS_1",)),
    ]

    run_id = uuid.uuid4().hex
    gates = {
        rid: ExternalExecution(
            "rail_freight_workshop_v1",
            rid,
            (net.nodes[spec["start"]].x, net.nodes[spec["start"]].y),
            source="localized",
            run_id=run_id,
            namespace_routes=True,
            arrival_tolerance=0.15,
            timeout=2.0,
        )
        for rid, spec in robots_spec.items()
    }
    agvs = [
        Agv(rid, max_speed=0.2, start_node=spec["start"], load_time=0.2, unload_time=0.2)
        for rid, spec in robots_spec.items()
    ]
    sim = Simulator(
        net,
        tasks,
        agvs,
        _CheckpointSafeRustPlanner(max_speed=0.2, dt=0.1, use_sipp=True),
        GreedyAllocator(),
        SimulationConfig(
            dt=0.1,
            max_steps=20000,
            planning_horizon=12000,
            load_time=0.2,
            unload_time=0.2,
            idle_retreat=False,
            deadlock_timeout=0,
            reschedule_policy=ReschedulePolicy.NONE,
            record_trajectory=True,
        ),
        executions=gates,
    )
    sim.reset()
    closed_edges = [("W_BUFFER_OUT", "R_BUFFER_OUT"), ("J_S_85", "J_S_105")]
    for edge in closed_edges:
        assert sim.close_physical_corridor(edge, 0.0)["accepted"]
    coord = PhysicalCVCRCoordinator(
        sim,
        operations=ops,
        policy_method=policy_canon,
        budget_ms=18.0,
        epoch_interval=100,
        use_rust=True,
        seed=int(seed),
    )
    # Pre-warm native Rust graph and V18 lazy imports outside the real-time Gazebo guard loop.
    backend = sim.planner._backend()
    if hasattr(backend, "_get_or_build_graph"):
        backend._get_or_build_graph(sim._routing_network())
    if policy_canon.startswith("V18_"):
        from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals  # noqa: F401
        from amr_sched.cvcr.v18_frontier_synthesis import load_v18_synthesis_policy
        load_v18_synthesis_policy()
    return scene, net, sim, coord, ops, robots_spec, closed_edges


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--schedule-port", type=int, default=9500)
    p.add_argument("--wait-wpf-s", type=float, default=55.0)
    p.add_argument("--max-wall-s", type=float, default=900.0)
    p.add_argument("--policy", type=str, default="V18_EA_JOINT")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-rviz", action="store_true")
    p.add_argument("--no-camera-capture", action="store_true")
    p.add_argument("--startup-check-only", action="store_true")
    args = p.parse_args()
    requested_policy = str(args.policy)
    policy_canon = resolve_policy_name(requested_policy)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    master_port = int(os.environ.get("GAZEBO_MASTER_URI", "http://127.0.0.1:11362").rsplit(":", 1)[-1])
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", master_port)) == 0:
            raise RuntimeError(f"Gazebo master port {master_port} is already occupied")

    scene, net, sim, coord, ops, robots_spec, closed_edges = build_map01_v18_scenario(
        policy=policy_canon, seed=args.seed
    )
    derived_world = derive_map01_world(
        ROOT / "tools/live_bridge/worlds/metric/rail_workshop_metric.world",
        out / "rail_workshop_metric_live.world",
        physics_rate_hz=250,
        camera_rate_hz=0.5,
    )
    map_yaml = ROOT / "tools/live_bridge/maps/metric_workshop/map.yaml"
    src_urdf = ROOT / "tools/ros2/wheelset_dependencies/scout_gazebo_description/urdf/scout_mini_expanded.urdf"
    # Derive 20 Hz / 95 m LiDAR & 80 Hz odom URDF so all 6 robots see perimeter walls without overloading RTF.
    import xml.etree.ElementTree as _ET
    _utree = _ET.parse(src_urdf)
    _lidar = _utree.getroot().find(".//sensor[@type='ray']")
    if _lidar is not None:
        _ur = _lidar.find("update_rate")
        if _ur is not None:
            _ur.text = "20"
        _smp = _lidar.find("ray/scan/horizontal/samples")
        if _smp is not None:
            _smp.text = "61"
        _rmax = _lidar.find("ray/range/max")
        if _rmax is not None:
            _rmax.text = "95.0"
    for _plug in _utree.getroot().findall(".//plugin"):
        if _plug.get("name") in ("scout_mini_diff_drive", "scout_mini_joint_states"):
            _pur = _plug.find("update_rate")
            if _pur is not None:
                _pur.text = "50"
    base_urdf = out / "scout_base_opt.urdf"
    _utree.write(base_urdf, encoding="utf-8", xml_declaration=True)

    loc_base = yaml.safe_load((ROOT / "tools/live_bridge/sensor_localization.yaml").read_text(encoding="utf-8"))
    loc_base["amcl"]["ros__parameters"].update(
        sigma_hit=0.08,
        z_hit=0.85,
        z_rand=0.15,
        alpha1=0.02,
        alpha2=0.02,
        alpha3=0.03,
        alpha4=0.02,
        alpha5=0.02,
        robot_model_type="nav2_amcl::DifferentialMotionModel",
        laser_model_type="likelihood_field",
        laser_max_range=95.0,
        laser_likelihood_max_dist=2.0,
        min_particles=400,
        max_particles=1000,
        max_beams=90,
        do_beamskip=False,
        update_min_d=0.04,
        update_min_a=0.04,
        resample_interval=1,
        transform_tolerance=0.5,
    )
    nav_base = yaml.safe_load((ROOT / "tools/live_bridge/nav2_metric.yaml").read_text(encoding="utf-8"))
    dwb_fp = yaml.safe_load((ROOT / "tools/live_bridge/nav2_metric_dwb.yaml").read_text(encoding="utf-8"))["controller_server"]["ros__parameters"]["FollowPath"]
    dwb_fp["vx_samples"] = 8
    dwb_fp["vtheta_samples"] = 13
    critics = list(dwb_fp.get("critics") or [])
    if "Twirling" not in critics:
        critics.append("Twirling")
    dwb_fp["critics"] = critics
    dwb_fp["Twirling.scale"] = 2.0
    dwb_fp["Oscillation.oscillation_reset_time"] = 1.5
    dwb_fp["xy_goal_tolerance"] = 0.02
    dwb_fp["rotate_to_heading_angular_vel"] = 0.45
    dwb_fp["RotateToGoal.scale"] = 12.0
    dwb_fp["RotateToGoal.slowing_factor"] = 10.0
    dwb_fp["amr_execution::ContinuousGoalDist.scale"] = 16.0
    nav_ctrl_params = nav_base["controller_server"]["ros__parameters"]
    nav_ctrl_params["controller_frequency"] = 15.0
    nav_ctrl_params["progress_checker"]["movement_time_allowance"] = 25.0
    nav_ctrl_params["general_goal_checker"].update(
        xy_goal_tolerance=0.10,
        yaw_goal_tolerance=0.30,
    )
    if "dock_goal_checker" in nav_ctrl_params:
        nav_ctrl_params["dock_goal_checker"].update(
            xy_goal_tolerance=0.05,
            yaw_goal_tolerance=0.25,
        )
    nav_base["controller_server"]["ros__parameters"]["FollowPath"] = dwb_fp
    costmap = nav_base["local_costmap"]["local_costmap"]["ros__parameters"]
    costmap["robot_base_frame"] = "front_tracking"
    offset = 0.463951 / 2.0
    costmap["footprint"] = json.dumps([[x - offset, y] for x, y in [(-0.315, -0.252), (0.315, -0.252), (0.315, 0.252), (-0.315, 0.252)]])

    processes: list[subprocess.Popen] = []
    logs = []
    import shutil as _shutil
    _rt_root = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path("/tmp")
    fast_rt = _rt_root / f"map01_rt_{os.getpid()}"
    _shutil.rmtree(fast_rt, ignore_errors=True)
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
    probes: dict[str, RobotStartupProbe] = {}
    inspector = Node("map01_chain_inspector", use_global_arguments=False)
    executor.add_node(inspector)
    runtime = None
    v18_obs = None
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
        while time.monotonic() < dl:
            spin_burst(first_timeout=0.01)
            if pred():
                return
        raise RuntimeError(reason)

    def start_active(cmd: list[str], name: str, ns: str, managed: list[str]) -> subprocess.Popen:
        probe = probes[next(r for r, d in robots_spec.items() if d["namespace"] == ns)]
        for attempt in range(3):
            proc = start_proc(cmd, f"{name}" + (f"_retry{attempt}" if attempt else ""))
            clients = {m: probe.create_client(GetState, f"/{ns}/{m}/get_state") for m in managed}
            states = {m: None for m in managed}
            dl = time.monotonic() + 25.0
            while proc.poll() is None and time.monotonic() < dl:
                for m, cl in clients.items():
                    if not cl.service_is_ready():
                        continue
                    fut = cl.call_async(GetState.Request())
                    rdl = time.monotonic() + 0.5
                    while not fut.done() and time.monotonic() < rdl:
                        executor.spin_once(timeout_sec=0.01)
                    if fut.done() and fut.result() is not None:
                        states[m] = fut.result().current_state.id
                if all(st == 3 for st in states.values()):
                    for cl in clients.values():
                        probe.destroy_client(cl)
                    return proc
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
        raise RuntimeError(f"{name} failed to reach ACTIVE state")

    try:
        for idx, (rid, spec) in enumerate(robots_spec.items()):
            ns = spec["namespace"]
            nd = net.nodes[spec["start"]]
            loc_cfg, nav_cfg = robot_configs(
                ns, loc_base, nav_base, dock_tolerance_m=0.03, dock_yaw_tolerance_rad=0.15
            )
            (out / f"{ns}_localization.yaml").write_text(yaml.safe_dump(loc_cfg), encoding="utf-8")
            (out / f"{ns}_controller.yaml").write_text(yaml.safe_dump(nav_cfg), encoding="utf-8")
            pr = RobotStartupProbe(rid, ns)
            probes[rid] = pr
            executor.add_node(pr)
            start_proc(
                [
                    "ros2", "launch", "amr_gazebo_guard", "scout_metric.launch.py",
                    f"world:={derived_world}",
                    f"lockstep:=false",
                    f"base_urdf:={base_urdf}",
                    f"robot_namespace:={ns}",
                    f"start_world:={'true' if idx == 0 else 'false'}",
                    "front_tracking:=true",
                    "laser_range_max:=95.0",
                    f"laser_fov_radians:={2.0 * math.pi}",
                    f"x:={nd.x}",
                    f"y:={nd.y}",
                    f"yaw:={spec['yaw']}",
                    "odom_topic:=wheel/odometry",
                    "odom_tf:=false",
                    "imu_gyro_noise:=0.002",
                    "spawn_timeout:=360",
                    f"model_out:={out / (ns + '.urdf')}",
                ],
                f"{ns}_model",
            )

        until(lambda: all(p.ready(time.monotonic()) for p in probes.values()), 380.0,
              "Robot entities, clock, sensors or guard failed startup readiness")
        pump(2.0)

        for rid, spec in robots_spec.items():
            ns = spec["namespace"]
            start_active(
                [
                    "ros2", "launch", "amr_gazebo_guard", "sensor_localization.launch.py",
                    f"config:={out / (ns + '_localization.yaml')}",
                    f"robot_namespace:={ns}",
                    f"map:={map_yaml}",
                    "laser_max_range:=95.0",
                    "laser_model_type:=likelihood_field",
                ],
                f"{ns}_localization",
                ns,
                ["map_server", "amcl"],
            )

        until(lambda: all(pr.initial_pub.get_subscription_count() > 0 for pr in probes.values()), 25.0,
              "AMCL initialpose subscribers unavailable")
        rtf_t0_wall = time.monotonic()
        rtf_t0_sim = next((pr.clock_s for pr in probes.values() if pr.clock_s is not None), 0.0)
        for rid, spec in robots_spec.items():
            nd = net.nodes[spec["start"]]
            probes[rid].publish_initialpose((nd.x, nd.y), spec["yaw"])
        step_hist = []
        for step_idx in range(24):
            for pr in probes.values():
                if pr.nomotion.service_is_ready():
                    fut = pr.nomotion.call_async(Empty.Request())
                    dl = time.monotonic() + 0.5
                    while not fut.done() and time.monotonic() < dl:
                        executor.spin_once(timeout_sec=0.01)
            pump(0.45)
            cur_errs = {
                rid: (
                    round(
                        math.hypot(
                            probes[rid].amcl_xy[0] - net.nodes[spec["start"]].x,
                            probes[rid].amcl_xy[1] - net.nodes[spec["start"]].y,
                        ),
                        4,
                    )
                    if probes[rid].amcl_xy is not None
                    else None
                )
                for rid, spec in robots_spec.items()
            }
            if step_idx in (0, 1, 2, 5, 11, 23):
                step_hist.append({"step": step_idx, "errs": cur_errs, "xy": {r: probes[r].amcl_xy for r in robots_spec}})
            if all(
                probes[rid].amcl_converged(
                    (net.nodes[spec["start"]].x, net.nodes[spec["start"]].y), tol_m=0.12
                )
                for rid, spec in robots_spec.items()
            ):
                break
        for rid, pr in probes.items():
            pr.poll_truth()
        pump(0.3)
        rtf_wall = max(0.01, time.monotonic() - rtf_t0_wall)
        rtf_sim = max(0.0, (next((pr.clock_s for pr in probes.values() if pr.clock_s is not None), rtf_t0_sim) - rtf_t0_sim))
        amcl_errs = {
            rid: (
                round(
                    math.hypot(
                        probes[rid].amcl_xy[0] - net.nodes[spec["start"]].x,
                        probes[rid].amcl_xy[1] - net.nodes[spec["start"]].y,
                    ),
                    4,
                )
                if probes[rid].amcl_xy is not None
                else None
            )
            for rid, spec in robots_spec.items()
        }
        diag = {
            "amcl_errs": amcl_errs,
            "amcl_xy": {rid: probes[rid].amcl_xy for rid in robots_spec},
            "truth_xy": {rid: probes[rid].truth_xy for rid in robots_spec},
            "startup_rtf": round(rtf_sim / rtf_wall, 3),
            "step_hist": step_hist,
        }
        (out / "startup_diag.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
        print("AMCL startup diag:", json.dumps(diag), flush=True)
        if args.startup_check_only:
            return 0

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
        pump(2.0)

        if not args.no_rviz:
            rviz_cfg = write_map01_rviz_config(
                out / "map01_live.rviz",
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
                    "--camera-topic", "/map01_review/camera/image_raw",
                    "--rviz-match", "map01_live.rviz",
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
            runtime_files=[Path(__file__), ROOT / "tools/live_bridge/map01_live_helpers.py"],
            tracking_offset_m=0.463951 / 2.0,
            latest_tf=True,
            input_thread=False,
            dock_takeover=None,
            async_logs=False,
            cvcr_coordinator=coord,
            defer_ingress=False,
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
                "scene_id": "rail_freight_workshop_v1",
                "robots": list(robots_spec),
            }),
            encoding="utf-8",
        )
        ready_tmp.replace(out / "control_ready.json")
        runtime.start_ingress()
        runtime.start_owner_pump()

        # Wait briefly for RViz2 and external WPF digital twin to attach before motion begins
        saw_wpf_client = False
        wait_dl = time.monotonic() + max(1.0, args.wait_wpf_s)
        while time.monotonic() < wait_dl:
            spin_burst(first_timeout=0.01)
            runtime.pump_owner()
            v18_obs.publish(time.monotonic())
            if runtime.control_server is not None and runtime.control_server.clients:
                saw_wpf_client = True
            if saw_wpf_client and (args.no_rviz or v18_obs.saw_rviz_subscription):
                break

        runtime.enable()
        started = time.monotonic()
        completed = False
        while time.monotonic() - started < args.max_wall_s:
            tl0 = time.perf_counter()
            spin_burst(first_timeout=0.01)
            tl1 = time.perf_counter()
            runtime.pump_owner()
            tl2 = time.perf_counter()
            now_w = time.monotonic()
            v18_obs.publish(now_w)
            tl3 = time.perf_counter()
            if tl3 - tl0 >= 0.06:
                runtime.record(
                    "slow_loop",
                    spin_ms=round((tl1 - tl0) * 1000.0, 2),
                    pump_ms=round((tl2 - tl1) * 1000.0, 2),
                    obs_ms=round((tl3 - tl2) * 1000.0, 2),
                    total_ms=round((tl3 - tl0) * 1000.0, 2),
                )
            if not saw_wpf_client and runtime.control_server is not None and runtime.control_server.clients:
                saw_wpf_client = True
            if any(g.fault for g in sim.executions.values()):
                raise RuntimeError("Fleet execution latched fault: " + str({r: g.fault for r, g in sim.executions.items() if g.fault}))
            if sim._all_done() and all(n.clean_idle_shutdown_ready() for n in runtime.nodes.values()):
                while not coord.tracker.all_done(sim._step_cursor) and sim._step_cursor < sim.config.max_steps:
                    sim._step_cursor += 1
                    coord.tracker.step(sim._step_cursor)
                    v18_obs.publish(time.monotonic())
                    runtime.pump_owner()
                    pump(0.05)
                completed = coord.tracker.all_done(sim._step_cursor)
                break
        if not completed:
            raise RuntimeError("MAP-01 full-chain timed out before all tasks and operations finished")

        aud = IndependentAuditor.audit_simulation_run(sim, ops, require_trajectory=True)
        v18_sum = v18_obs.close(executor) if v18_obs else {}
        pol_sum = summarize_physical_run(coord, ops)
        report = {
            "passed": bool(completed and aud.pass_audit and v18_sum.get("accepted", 0) > 0),
            "scene_id": "rail_freight_workshop_v1",
            "policy": coord.policy_method,
            "requested_policy": requested_policy,
            "seed": int(args.seed),
            "sim_steps": int(sim._step_cursor),
            "sim_time_s": round(sim._step_cursor * sim.config.dt, 2),
            "elapsed_wall_s": round(time.monotonic() - started, 2),
            "closed_corridors": [list(e) for e in closed_edges],
            "tasks": [
                {"task_id": t.task_id, "status": t.status.value, "agv": t.assigned_agv, "pickup": t.pickup, "dropoff": t.dropoff}
                for t in sim.tasks
            ],
            "operations": [
                {"op_id": o.op_id, "bay_id": o.bay_id, "started": o.started, "start_step": o.start_step, "end_step": o.end_step}
                for o in ops
            ],
            "v18": v18_sum,
            "policy_metrics": pol_sum,
            "audit": {"passed": aud.pass_audit, "issues": list(aud.issues)},
            "wpf_connected_ever": saw_wpf_client,
        }
        (out / "full_chain_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not aud.pass_audit:
            raise RuntimeError(f"IndependentAuditor rejected run: {aud.issues}")
        print(json.dumps(report, indent=2), flush=True)
    except Exception as exc:
        error_text = repr(exc)
        (out / "full_chain_error.txt").write_text(error_text + "\n", encoding="utf-8")
        raise
    finally:
        capture_stop.write_text("stop\n", encoding="utf-8")
        if v18_obs is not None:
            try:
                v18_obs.close(executor)
            except Exception:
                pass
        if runtime is not None:
            try:
                runtime.begin_close()
                pump(0.5)
                runtime.close()
            except Exception:
                pass
        for pr in list(probes.values()):
            try:
                executor.remove_node(pr)
                pr.destroy_node()
            except Exception:
                pass
        try:
            executor.remove_node(inspector)
            inspector.destroy_node()
            executor.shutdown()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        cleanup = []
        for proc in reversed(processes):
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=5.0)
            cleanup.append({"pid": proc.pid, "returncode": proc.returncode})
        for log in logs:
            log.close()
        try:
            for item in fast_rt.iterdir():
                if item.is_file() and (item.name.endswith((".jsonl", ".json", ".lock"))):
                    _shutil.copy2(item, out / item.name)
            if fast_capture_dir.is_dir():
                capture_dir.mkdir(parents=True, exist_ok=True)
                for sub in fast_capture_dir.iterdir():
                    dst = capture_dir / sub.name
                    if sub.is_dir():
                        _shutil.copytree(sub, dst, dirs_exist_ok=True)
                    elif sub.is_file():
                        _shutil.copy2(sub, dst)
        finally:
            _shutil.rmtree(fast_rt, ignore_errors=True)
        (out / "cleanup.json").write_text(json.dumps(cleanup, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
