"""RViz2 config, metric DWB/AMCL tuning, and grounded workpiece transport sync for wheelset_workshop_v2."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import yaml

from .wheelset_v2_agv_model import (
    CARRIER_DECK_WHEELSET_Z,
    TRACKING_REFERENCE_X,
    build_dynamic_wheelset_sdf,
)

GROUNDED_WORKPIECE_MAX_Z_M = 0.95
STATION_RAIL_Z_M = 0.05


def build_wheelset_v2_nav_and_loc_templates() -> tuple[dict, dict]:
    """Return un-namespaced (localization_cfg, controller_cfg) tuned for 1.70x1.50m wheelset AGVs."""
    loc = {
        "ekf_filter_node": {
            "ros__parameters": {
                "use_sim_time": True,
                "frequency": 50.0,
                "sensor_timeout": 0.25,
                "two_d_mode": True,
                "publish_tf": True,
                "map_frame": "map",
                "odom_frame": "odom",
                "base_link_frame": "base_footprint",
                "world_frame": "odom",
                "odom0": "/wheel/odometry",
                "odom0_config": [
                    False, False, False, False, False, False,
                    True, False, False, False, False, False,
                    False, False, False,
                ],
                "odom0_queue_size": 10,
                "imu0": "/imu/data",
                "imu0_config": [
                    False, False, False, False, False, False,
                    False, False, False, False, False, True,
                    False, False, False,
                ],
                "imu0_queue_size": 20,
                "imu0_remove_gravitational_acceleration": False,
            }
        },
        "map_server": {
            "ros__parameters": {
                "use_sim_time": True,
                "frame_id": "map",
                "topic_name": "map",
            }
        },
        "amcl": {
            "ros__parameters": {
                "use_sim_time": True,
                "global_frame_id": "map",
                "odom_frame_id": "odom",
                "base_frame_id": "base_footprint",
                "scan_topic": "/scan",
                "robot_model_type": "nav2_amcl::DifferentialMotionModel",
                "tf_broadcast": True,
                "transform_tolerance": 0.5,
                "min_particles": 300,
                "max_particles": 800,
                "alpha1": 0.01,
                "alpha2": 0.01,
                "alpha3": 0.02,
                "alpha4": 0.01,
                "alpha5": 0.01,
                "laser_model_type": "likelihood_field_prob",
                "laser_max_range": 60.0,
                "laser_min_range": 0.10,
                "max_beams": 120,
                "do_beamskip": True,
                "beam_skip_distance": 0.5,
                "beam_skip_threshold": 0.3,
                "beam_skip_error_threshold": 0.9,
                "laser_likelihood_max_dist": 2.0,
                "z_hit": 0.96,
                "z_rand": 0.04,
                "sigma_hit": 0.08,
                "update_min_d": 0.05,
                "update_min_a": 0.08,
                "resample_interval": 1,
                "set_initial_pose": False,
            }
        },
    }

    off = TRACKING_REFERENCE_X
    footprint = [
        [-0.85 - off, -0.75],
        [0.85 - off, -0.75],
        [0.85 - off, 0.75],
        [-0.85 - off, 0.75],
    ]
    nav = {
        "controller_server": {
            "ros__parameters": {
                "use_sim_time": True,
                "controller_frequency": 15.0,
                "odom_topic": "/odometry/filtered",
                "min_x_velocity_threshold": 0.001,
                "min_y_velocity_threshold": 0.001,
                "min_theta_velocity_threshold": 0.001,
                "failure_tolerance": 0.5,
                "progress_checker_plugin": "progress_checker",
                "goal_checker_plugins": ["general_goal_checker"],
                "controller_plugins": ["FollowPath"],
                "progress_checker": {
                    "plugin": "nav2_controller::SimpleProgressChecker",
                    "required_movement_radius": 0.10,
                    "movement_time_allowance": 25.0,
                },
                "general_goal_checker": {
                    "plugin": "nav2_controller::SimpleGoalChecker",
                    "stateful": True,
                    "xy_goal_tolerance": 0.12,
                    "yaw_goal_tolerance": 0.30,
                },
                "FollowPath": {
                    "plugin": "nav2_rotation_shim_controller::RotationShimController",
                    "primary_controller": "dwb_core::DWBLocalPlanner",
                    "debug_trajectory_details": False,
                    "transform_tolerance": 0.25,
                    "angular_dist_threshold": 0.45,
                    "angular_disengage_threshold": 0.15,
                    "forward_sampling_distance": 0.40,
                    "rotate_to_heading_angular_vel": 0.45,
                    "max_angular_accel": 1.2,
                    "simulate_ahead_time": 1.0,
                    "rotate_to_goal_heading": False,
                    "closed_loop": True,
                    "min_vel_x": 0.0,
                    "min_vel_y": 0.0,
                    "max_vel_x": 0.65,
                    "max_vel_y": 0.0,
                    "max_vel_theta": 0.85,
                    "min_speed_xy": 0.0,
                    "max_speed_xy": 0.65,
                    "min_speed_theta": 0.0,
                    "acc_lim_x": 0.50,
                    "acc_lim_y": 0.0,
                    "acc_lim_theta": 1.0,
                    "decel_lim_x": -0.50,
                    "decel_lim_y": 0.0,
                    "decel_lim_theta": -1.0,
                    "vx_samples": 20,
                    "vy_samples": 1,
                    "vtheta_samples": 17,
                    "sim_time": 1.0,
                    "sim_period": 0.067,
                    "discretize_by_time": True,
                    "time_granularity": 0.05,
                    "linear_granularity": 0.025,
                    "angular_granularity": 0.025,
                    "xy_goal_tolerance": 0.05,
                    "trans_stopped_velocity": 0.01,
                    "short_circuit_trajectory_evaluation": True,
                    "stateful": True,
                    "prune_plan": True,
                    "prune_distance": 2.5,
                    "forward_prune_distance": 2.5,
                    "shorten_transformed_plan": True,
                    "publish_evaluation": False,
                    "publish_trajectories": False,
                    "critics": [
                        "RotateToGoal",
                        "Oscillation",
                        "ObstacleFootprint",
                        "PathAlign",
                        "GoalAlign",
                        "PathDist",
                        "GoalDist",
                        "amr_execution::ContinuousGoalDist",
                        "Twirling",
                    ],
                    "Oscillation.oscillation_reset_time": 1.5,
                    "ObstacleFootprint.scale": 0.02,
                    "PathAlign.scale": 32.0,
                    "PathAlign.forward_point_distance": 0.1,
                    "GoalAlign.scale": 12.0,
                    "GoalAlign.forward_point_distance": 0.1,
                    "PathDist.scale": 32.0,
                    "GoalDist.scale": 32.0,
                    "amr_execution::ContinuousGoalDist.scale": 16.0,
                    "Twirling.scale": 2.0,
                    "RotateToGoal.scale": 12.0,
                    "RotateToGoal.slowing_factor": 10.0,
                    "RotateToGoal.lookahead_time": -1.0,
                },
            }
        },
        "local_costmap": {
            "local_costmap": {
                "ros__parameters": {
                    "use_sim_time": True,
                    "update_frequency": 10.0,
                    "publish_frequency": 5.0,
                    "always_send_full_costmap": True,
                    "global_frame": "odom",
                    "robot_base_frame": "front_tracking",
                    "rolling_window": True,
                    "width": 12,
                    "height": 12,
                    "resolution": 0.1,
                    "footprint": json.dumps(footprint),
                    "footprint_padding": 0.02,
                    "transform_tolerance": 0.25,
                    "plugins": ["obstacle_layer", "inflation_layer"],
                    "obstacle_layer": {
                        "plugin": "nav2_costmap_2d::ObstacleLayer",
                        "enabled": True,
                        "observation_sources": "scan",
                        "scan": {
                            "topic": "/scan",
                            "data_type": "LaserScan",
                            "expected_update_rate": 0.3,
                            "marking": True,
                            "clearing": True,
                            "max_obstacle_height": 2.0,
                            "obstacle_max_range": 15.0,
                            "raytrace_max_range": 16.0,
                        },
                    },
                    "inflation_layer": {
                        "plugin": "nav2_costmap_2d::InflationLayer",
                        "cost_scaling_factor": 3.0,
                        "inflation_radius": 0.50,
                    },
                }
            }
        },
    }
    return loc, nav


def write_wheelset_v2_rviz_config(
    target_rviz: Path,
    *,
    robot_namespaces: tuple[str, ...],
    robot_ids: tuple[str, ...],
    scan_colors: tuple[str, ...] = (
        "255; 90; 40",
        "70; 220; 110",
        "80; 170; 255",
    ),
) -> Path:
    """Create RViz2 TopDownOrtho config framing the full 258x72m wheelset_workshop_v2 hall."""
    displays = [
        {
            "Class": "rviz_default_plugins/Grid",
            "Name": "Grid",
            "Enabled": True,
            "Reference Frame": "<Fixed Frame>",
            "Cell Size": 5,
            "Plane Cell Count": 55,
            "Color": "110; 110; 120",
            "Alpha": 0.35,
        },
        {
            "Class": "rviz_default_plugins/Map",
            "Name": "WorkshopMap",
            "Enabled": True,
            "Topic": {
                "Value": f"/{robot_namespaces[0]}/map",
                "Depth": 1,
                "Durability Policy": "Transient Local",
                "Reliability Policy": "Reliable",
            },
            "Alpha": 0.82,
            "Color Scheme": "map",
            "Draw Behind": True,
        },
        {
            "Class": "rviz_default_plugins/MarkerArray",
            "Name": "CVCRTopology",
            "Enabled": True,
            "Topic": {
                "Value": "/cvcr/topology_markers",
                "Depth": 10,
                "Durability Policy": "Volatile",
                "Reliability Policy": "Reliable",
            },
        },
        {
            "Class": "rviz_default_plugins/MarkerArray",
            "Name": "CVCRBeliefs",
            "Enabled": True,
            "Topic": {
                "Value": "/cvcr/corridor_belief_markers",
                "Depth": 10,
                "Durability Policy": "Volatile",
                "Reliability Policy": "Reliable",
            },
        },
        {
            "Class": "rviz_default_plugins/MarkerArray",
            "Name": "CVCRFleetPlan",
            "Enabled": True,
            "Topic": {
                "Value": "/cvcr/fleet_plan_markers",
                "Depth": 10,
                "Durability Policy": "Volatile",
                "Reliability Policy": "Reliable",
            },
        },
        {
            "Class": "rviz_default_plugins/MarkerArray",
            "Name": "CVCRKittingHUD",
            "Enabled": True,
            "Topic": {
                "Value": "/cvcr/kitting_hud_markers",
                "Depth": 10,
                "Durability Policy": "Volatile",
                "Reliability Policy": "Reliable",
            },
        },
    ]
    for idx, (ns, rid) in enumerate(zip(robot_namespaces, robot_ids)):
        col = scan_colors[idx % len(scan_colors)]
        displays.append(
            {
                "Class": "rviz_default_plugins/LaserScan",
                "Name": f"{rid}_Scan",
                "Enabled": True,
                "Topic": {
                    "Value": f"/{ns}/scan",
                    "Depth": 5,
                    "Durability Policy": "Volatile",
                    "Reliability Policy": "Best Effort",
                },
                "Size (Pixels)": 3,
                "Style": "Points",
                "Color Transformer": "FlatColor",
                "Color": col,
            }
        )
        displays.append(
            {
                "Class": "rviz_default_plugins/Path",
                "Name": f"{rid}_Path",
                "Enabled": True,
                "Topic": {
                    "Value": f"/cvcr/planned_paths/{rid}",
                    "Depth": 5,
                    "Durability Policy": "Volatile",
                    "Reliability Policy": "Reliable",
                },
                "Line Style": "Lines",
                "Color": col,
            }
        )
    cfg = {
        "Panels": [
            {"Class": "rviz_common/Displays", "Name": "Displays", "Property Tree Widget": {"Expanded": ["/CVCRTopology"]}},
        ],
        "Visualization Manager": {
            "Class": "",
            "Displays": displays,
            "Global Options": {"Background Color": "28; 32; 38", "Fixed Frame": "map", "Frame Rate": 4},
            "Tools": [{"Class": "rviz_default_plugins/MoveCamera"}],
            "Views": {
                "Current": {
                    "Class": "rviz_default_plugins/TopDownOrtho",
                    "Name": "WorkshopOverview",
                    "Angle": 0.0,
                    "Scale": 3.4,
                    "X": 125.0,
                    "Y": 30.0,
                }
            },
        },
        "Window Geometry": {"Width": 1120, "Height": 640, "X": 40, "Y": 40, "Hide Left Dock": True, "Hide Right Dock": True},
    }
    target_rviz.parent.mkdir(parents=True, exist_ok=True)
    target_rviz.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return target_rviz


class GroundedWheelsetWorkpieceSync:
    """Keeps dynamic Gazebo wheelset workpieces grounded (<=0.95m) on cradles or station rails."""

    def __init__(
        self,
        workpieces: dict[str, dict],
        tasks_by_id: dict[str, object],
        net: object,
        log_path: Path,
        *,
        spawn_client=None,
        get_state_client=None,
        set_state_client=None,
        ns_by_agv: dict[str, str] | None = None,
    ) -> None:
        self.workpieces = workpieces  # {ws_id: {"chain": [task_id,...], "hold_xy": (x,y), "z": 0.05}}
        self.tasks_by_id = tasks_by_id
        self.net = net
        self.log_path = log_path
        self.spawn_client = spawn_client
        self.get_state_client = get_state_client
        self.set_state_client = set_state_client
        self.ns_by_agv = ns_by_agv or {}
        self.max_z = 0.0
        self.min_z = math.inf
        self.carried_updates = 0
        self.station_updates = 0
        self.total_updates = 0
        self._last_sync_wall = -math.inf
        self._last_agv_pose: dict[str, tuple[float, float, float]] = {}
        self._pending_gets: dict[str, object] = {}
        self._pending_sets: list[object] = []
        self._last_pushed_pose: dict[str, tuple[float, float, float, float]] = {}
        self._records: list[dict] = []

    def spawn_all(self, executor) -> None:
        """Spawn visual wheelset workpieces on grounded initial station rails."""
        if self.spawn_client is None:
            return
        from gazebo_msgs.srv import SpawnEntity  # noqa: PLC0415

        dl = time.monotonic() + 20.0
        while not self.spawn_client.service_is_ready():
            if time.monotonic() >= dl:
                raise RuntimeError("/spawn_entity service unavailable for wheelset workpieces")
            executor.spin_once(timeout_sec=0.05)

        for ws_id, info in self.workpieces.items():
            req = SpawnEntity.Request()
            req.name = ws_id
            req.xml = build_dynamic_wheelset_sdf(ws_id)
            x, y = info["hold_xy"]
            z = STATION_RAIL_Z_M
            req.initial_pose.position.x = float(x)
            req.initial_pose.position.y = float(y)
            req.initial_pose.position.z = float(z)
            req.initial_pose.orientation.w = 1.0
            fut = self.spawn_client.call_async(req)
            while not fut.done() and time.monotonic() < dl:
                executor.spin_once(timeout_sec=0.02)
            self._record(0.0, ws_id, "station_parked", None, x, y, z, 0.0)

    def _record(self, sim_s: float, ws_id: str, phase: str, carrier: str | None, x: float, y: float, z: float, yaw: float) -> None:
        if z > GROUNDED_WORKPIECE_MAX_Z_M or z < 0.0:
            raise ValueError(f"Grounded wheelset height invariant violated: {ws_id} z={z:.3f}m")
        self.max_z = max(self.max_z, z)
        self.min_z = min(self.min_z, z)
        self.total_updates += 1
        if carrier:
            self.carried_updates += 1
        else:
            self.station_updates += 1
        self._records.append({
            "sim_s": round(sim_s, 3),
            "workpiece": ws_id,
            "phase": phase,
            "carrier": carrier,
            "x": round(x, 3),
            "y": round(y, 3),
            "z": round(z, 3),
            "yaw": round(yaw, 4),
        })

    def step(self, sim, sim_executions: dict) -> None:
        """Update workpiece poses in Gazebo strictly at grounded cradle/rail elevation (<=0.95m)."""
        # Drain completed Gazebo entity queries/sets without blocking simulation loop
        for agv_id, fut in list(self._pending_gets.items()):
            if fut.done():
                self._pending_gets.pop(agv_id)
                try:
                    res = fut.result()
                    if res.success:
                        p = res.state.pose.position
                        q = res.state.pose.orientation
                        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                        self._last_agv_pose[agv_id] = (float(p.x), float(p.y), float(yaw))
                except Exception:
                    pass
        self._pending_sets = [f for f in self._pending_sets if not f.done()]

        now = time.monotonic()
        if now - self._last_sync_wall < 0.15:
            return
        self._last_sync_wall = now
        sim_s = float(getattr(sim, "_step_cursor", 0) * getattr(sim.config, "dt", 1.0))

        # Trigger non-blocking Gazebo truth read for active carriers
        if self.get_state_client is not None and self.get_state_client.service_is_ready():
            from gazebo_msgs.srv import GetEntityState  # noqa: PLC0415
            for agv_id, ns in self.ns_by_agv.items():
                if agv_id not in self._pending_gets:
                    req = GetEntityState.Request()
                    req.name = ns
                    req.reference_frame = "world"
                    self._pending_gets[agv_id] = self.get_state_client.call_async(req)

        for ws_id, info in self.workpieces.items():
            chain = info["chain"]
            active_task = None
            last_completed = None
            for tid in chain:
                t = self.tasks_by_id[tid]
                st = getattr(t.status, "value", str(t.status))
                if st in ("picked", "in_progress"):
                    active_task = t
                    break
                if st == "completed":
                    last_completed = t
            if active_task is not None and active_task.assigned_agv:
                agv_id = active_task.assigned_agv
                pose = self._last_agv_pose.get(agv_id)
                if pose is None:
                    ex = sim_executions.get(agv_id)
                    if ex is not None and ex.pose is not None:
                        pose = (float(ex.pose["x"]), float(ex.pose["y"]), float(ex.pose["yaw"]))
                if pose is not None:
                    x, y, yaw = pose
                    z = CARRIER_DECK_WHEELSET_Z
                    phase = f"carried:{active_task.task_id}"
                    carrier = agv_id
                else:
                    continue
            elif last_completed is not None:
                dn = self.net.nodes[last_completed.dropoff]
                x, y, yaw, z = float(dn.x), float(dn.y), 0.0, STATION_RAIL_Z_M
                phase = f"docked:{last_completed.dropoff}"
                carrier = None
            else:
                first_t = self.tasks_by_id[chain[0]]
                pn = self.net.nodes[first_t.pickup]
                x, y, yaw, z = float(pn.x), float(pn.y), 0.0, STATION_RAIL_Z_M
                phase = f"staged:{first_t.pickup}"
                carrier = None

            self._push_gazebo_pose(ws_id, x, y, z, yaw)
            self._record(sim_s, ws_id, phase, carrier, x, y, z, yaw)

    def _push_gazebo_pose(self, ws_id: str, x: float, y: float, z: float, yaw: float) -> None:
        if self.set_state_client is None or not self.set_state_client.service_is_ready():
            return
        pose_key = (round(float(x), 3), round(float(y), 3), round(float(z), 3), round(float(yaw), 3))
        if self._last_pushed_pose.get(ws_id) == pose_key:
            return
        if len(self._pending_sets) > 6:
            return
        self._last_pushed_pose[ws_id] = pose_key
        from gazebo_msgs.srv import SetEntityState  # noqa: PLC0415

        req = SetEntityState.Request()
        req.state.name = ws_id
        req.state.reference_frame = "world"
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = float(z)
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)
        self._pending_sets.append(self.set_state_client.call_async(req))

    def finalize(self) -> dict:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self._records),
            encoding="utf-8",
        )
        return {
            "total_updates": self.total_updates,
            "carried_updates": self.carried_updates,
            "station_updates": self.station_updates,
            "min_z_m": round(self.min_z, 4) if math.isfinite(self.min_z) else None,
            "max_z_m": round(self.max_z, 4),
            "grounded_limit_m": GROUNDED_WORKPIECE_MAX_Z_M,
            "grounded_passed": bool(self.total_updates > 0 and self.max_z <= GROUNDED_WORKPIECE_MAX_Z_M),
        }
