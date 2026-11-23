"""RViz2 Digital-Twin L1-L2-L3 Overlay Builder and ROS 2 Publisher.

Converts live workshop topology graphs, L2 P4.5 CVaR75/forgetting corridor beliefs,
L3 `V15_EA_JOINT` multi-AGV route assignments, and assembly kitting countdowns into
structured RViz2 `visualization_msgs/MarkerArray` and `nav_msgs/Path` payloads.
Works purely offline for unit testing and binds seamlessly to `rclpy` in WSL2/Linux.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence

from amr_sched.cvcr.execution_feedback import ExecutionFeedbackPacket
from amr_sched.cvcr.p4_belief_choke import CorridorDurationBeliefModel
from amr_sched.cvcr.production import OperationSnapshot
from amr_sched.network import Network


@dataclass(frozen=True)
class MarkerSpec:
    """Platform-independent RViz2 Marker specification convertible to `visualization_msgs/Marker`."""

    ns: str
    marker_id: int
    marker_type: int  # 2=SPHERE, 3=CYLINDER, 4=LINE_STRIP, 5=LINE_LIST, 9=TEXT_VIEW_FACING
    frame_id: str = "map"
    points: tuple[tuple[float, float, float], ...] = ()
    pose_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale_xyz: tuple[float, float, float] = (0.08, 0.08, 0.08)
    rgba: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 0.9)
    text: str = ""


_ROBOT_COLORS: tuple[tuple[float, float, float, float], ...] = (
    (0.15, 0.80, 1.00, 0.95),
    (0.30, 0.95, 0.45, 0.95),
    (1.00, 0.72, 0.18, 0.95),
    (0.85, 0.40, 1.00, 0.95),
    (1.00, 0.35, 0.55, 0.95),
    (0.45, 0.95, 0.90, 0.95),
)


def build_rviz_overlay_payloads(
    network: Network,
    *,
    tick: int = 0,
    belief_model: Optional[CorridorDurationBeliefModel] = None,
    feedback_packet: Optional[ExecutionFeedbackPacket] = None,
    planned_routes: Optional[Mapping[str, Sequence[tuple[str, int]]]] = None,
    robot_poses: Optional[Mapping[str, tuple[float, float]]] = None,
    operations: Sequence[OperationSnapshot] = (),
) -> dict[str, Any]:
    """Build L1-L2-L3 digital-twin MarkerSpec lists and Path point sequences."""
    topo_markers: list[MarkerSpec] = []
    belief_markers: list[MarkerSpec] = []
    fleet_markers: list[MarkerSpec] = []
    kitting_markers: list[MarkerSpec] = []
    paths_xy: dict[str, list[tuple[float, float]]] = {}

    # 1. Static workshop graph edges & labeled nodes
    seen_undir: set[tuple[str, str]] = set()
    edge_pts: list[tuple[float, float, float]] = []
    for e in network.edges.values():
        key = (e.src, e.dst) if e.src <= e.dst else (e.dst, e.src)
        if key in seen_undir:
            continue
        seen_undir.add(key)
        nu, nv = network.nodes[e.src], network.nodes[e.dst]
        edge_pts.append((float(nu.x), float(nu.y), 0.02))
        edge_pts.append((float(nv.x), float(nv.y), 0.02))
    if edge_pts:
        topo_markers.append(
            MarkerSpec(
                ns="topology_edges",
                marker_id=0,
                marker_type=5,
                points=tuple(edge_pts),
                scale_xyz=(0.04, 0.0, 0.0),
                rgba=(0.55, 0.62, 0.72, 0.75),
            )
        )
    mid_idx = 1
    for nid, nobj in sorted(network.nodes.items()):
        topo_markers.append(
            MarkerSpec(
                ns="topology_nodes",
                marker_id=mid_idx,
                marker_type=2,
                pose_xyz=(float(nobj.x), float(nobj.y), 0.04),
                scale_xyz=(0.14, 0.14, 0.06),
                rgba=(0.25, 0.65, 0.95, 0.85),
            )
        )
        mid_idx += 1
        topo_markers.append(
            MarkerSpec(
                ns="topology_labels",
                marker_id=mid_idx,
                marker_type=9,
                pose_xyz=(float(nobj.x), float(nobj.y) + 0.16, 0.18),
                scale_xyz=(0.0, 0.0, 0.16),
                rgba=(0.92, 0.95, 1.0, 0.95),
                text=str(nid),
            )
        )
        mid_idx += 1

    # 2. L2 P4.5 Corridor Belief & Blockage Markers (CVaR75 & active closures)
    b_idx = 0
    corr_updates = getattr(feedback_packet, "corridor_updates", {}) if feedback_packet is not None else {}
    rendered_edges: set[tuple[str, str]] = set()
    for (u, v), cf in sorted(corr_updates.items()):
        uk = (u, v) if u <= v else (v, u)
        if uk in rendered_edges or u not in network.nodes or v not in network.nodes:
            continue
        rendered_edges.add(uk)
        nu, nv = network.nodes[u], network.nodes[v]
        mx, my = 0.5 * (nu.x + nv.x), 0.5 * (nu.y + nv.y)
        is_blk = getattr(cf, "status", "open") == "blocked"
        cvar_s = (
            float(belief_model.estimate_remaining(uk, tick).upper_cvar75)
            if belief_model is not None
            else float(getattr(cf, "extra_delay_steps", 0))
        )
        q3 = int(getattr(cf, "q3_penalty_override", 0) or 0)
        rgba = (1.0, 0.22, 0.20, 0.92) if is_blk else (1.0, 0.78, 0.18, 0.85)
        belief_markers.append(
            MarkerSpec(
                ns="p45_corridor_bar",
                marker_id=b_idx,
                marker_type=3,
                pose_xyz=(float(mx), float(my), 0.15 + 0.02 * min(25.0, cvar_s)),
                scale_xyz=(0.24, 0.24, max(0.18, 0.04 * min(25.0, cvar_s))),
                rgba=rgba,
            )
        )
        b_idx += 1
        belief_markers.append(
            MarkerSpec(
                ns="p45_corridor_text",
                marker_id=b_idx,
                marker_type=9,
                pose_xyz=(float(mx), float(my), 0.65),
                scale_xyz=(0.0, 0.0, 0.16),
                rgba=(1.0, 0.92, 0.85, 0.98),
                text=f"{uk[0]}-{uk[1]} {'BLK' if is_blk else 'CLR'} cvar={cvar_s:.1f}s q3={q3}",
            )
        )
        b_idx += 1

    # 3. L3 V15_EA_JOINT Multi-AGV Planned Paths & Live Positions
    f_idx = 0
    rids = sorted(set(planned_routes.keys() if planned_routes else ()) | set(robot_poses.keys() if robot_poses else ()))
    for r_ord, rid in enumerate(rids):
        col = _ROBOT_COLORS[r_ord % len(_ROBOT_COLORS)]
        if planned_routes and rid in planned_routes:
            seq = planned_routes[rid]
            pts3d: list[tuple[float, float, float]] = []
            pts2d: list[tuple[float, float]] = []
            for nid, _st in seq:
                if nid in network.nodes:
                    nd = network.nodes[nid]
                    pts3d.append((float(nd.x), float(nd.y), 0.08 + 0.02 * r_ord))
                    pts2d.append((float(nd.x), float(nd.y)))
            if pts2d:
                paths_xy[rid] = pts2d
            if len(pts3d) >= 2:
                fleet_markers.append(
                    MarkerSpec(
                        ns="v15_routes",
                        marker_id=f_idx,
                        marker_type=4,
                        points=tuple(pts3d),
                        scale_xyz=(0.06, 0.0, 0.0),
                        rgba=col,
                    )
                )
                f_idx += 1
        if robot_poses and rid in robot_poses:
            rx, ry = robot_poses[rid]
            fleet_markers.append(
                MarkerSpec(
                    ns="agv_labels",
                    marker_id=f_idx,
                    marker_type=9,
                    pose_xyz=(float(rx), float(ry), 0.48),
                    scale_xyz=(0.0, 0.0, 0.18),
                    rgba=col,
                    text=str(rid),
                )
            )
            f_idx += 1

    # 4. Kitting Window & Production Operation HUD Markers
    k_idx = 0
    for op in operations:
        if op.bay_id not in network.nodes:
            continue
        nd = network.nodes[op.bay_id]
        state_lbl = "DONE" if (op.started and op.end_step is not None and tick >= op.end_step) else ("RUN" if op.started else "WAIT")
        win_txt = ""
        wins = getattr(op, "resource_windows", ())
        if wins and not op.started:
            w_rem = min((int(we) - int(tick) for _, we in wins if int(we) >= int(tick)), default=-1)
            win_txt = f" w={w_rem}" if w_rem >= 0 else " w=MISS"
        kitting_markers.append(
            MarkerSpec(
                ns="kitting_hud",
                marker_id=k_idx,
                marker_type=9,
                pose_xyz=(float(nd.x), float(nd.y) - 0.24, 0.35 + 0.14 * (k_idx % 2)),
                scale_xyz=(0.0, 0.0, 0.14),
                rgba=(0.95, 0.98, 0.60, 0.95),
                text=f"{op.op_id}[{state_lbl}]{win_txt}",
            )
        )
        k_idx += 1

    return {
        "topology_markers": topo_markers,
        "corridor_belief_markers": belief_markers,
        "fleet_plan_markers": fleet_markers,
        "kitting_hud_markers": kitting_markers,
        "planned_paths": paths_xy,
        "total_markers": len(topo_markers) + len(belief_markers) + len(fleet_markers) + len(kitting_markers),
    }


class RVizCVCROverlayPublisher:
    """Publishes L1-L2-L3 CVCR digital-twin visualization topics directly to RViz2."""

    def __init__(self, network: Network, *, node_name: str = "cvcr_rviz_overlay") -> None:
        """Initialize ROS 2 MarkerArray & Path publishers for RViz2 visualization."""
        from geometry_msgs.msg import Point, PoseStamped
        from nav_msgs.msg import Path as NavPath
        from rclpy.node import Node as _RosNode
        from visualization_msgs.msg import Marker, MarkerArray

        self._Point = Point
        self._PoseStamped = PoseStamped
        self._NavPath = NavPath
        self._Marker = Marker
        self._MarkerArray = MarkerArray
        self._node = _RosNode(node_name)
        self.network = network

        self._pub_topo = self._node.create_publisher(MarkerArray, "/cvcr/topology_markers", 10)
        self._pub_belief = self._node.create_publisher(MarkerArray, "/cvcr/corridor_belief_markers", 10)
        self._pub_fleet = self._node.create_publisher(MarkerArray, "/cvcr/fleet_plan_markers", 10)
        self._pub_kitting = self._node.create_publisher(MarkerArray, "/cvcr/kitting_hud_markers", 10)
        self._path_pubs: dict[str, Any] = {}
        self._cached_topo_msg: Any = None
        self._last_topo_subs: int = 0
        self.published_marker_batches: int = 0
        self.published_marker_count: int = 0

    @property
    def ros_node(self) -> Any:
        """Return underlying `rclpy.node.Node` handle."""
        return self._node

    def _to_marker_array(self, specs: Sequence[MarkerSpec], stamp: Any) -> Any:
        """Convert platform-independent `MarkerSpec` list into ROS 2 `MarkerArray`."""
        arr = self._MarkerArray()
        clear_m = self._Marker()
        clear_m.header.frame_id = "map"
        clear_m.header.stamp = stamp
        clear_m.action = self._Marker.DELETEALL
        arr.markers.append(clear_m)
        for sp in specs:
            m = self._Marker()
            m.header.frame_id = sp.frame_id
            m.header.stamp = stamp
            m.ns = sp.ns
            m.id = int(sp.marker_id)
            m.type = int(sp.marker_type)
            m.action = self._Marker.ADD
            m.pose.position.x = float(sp.pose_xyz[0])
            m.pose.position.y = float(sp.pose_xyz[1])
            m.pose.position.z = float(sp.pose_xyz[2])
            m.pose.orientation.w = 1.0
            m.scale.x = float(sp.scale_xyz[0])
            m.scale.y = float(sp.scale_xyz[1])
            m.scale.z = float(sp.scale_xyz[2])
            m.color.r = float(sp.rgba[0])
            m.color.g = float(sp.rgba[1])
            m.color.b = float(sp.rgba[2])
            m.color.a = float(sp.rgba[3])
            m.text = sp.text
            for px, py, pz in sp.points:
                pt = self._Point()
                pt.x, pt.y, pt.z = float(px), float(py), float(pz)
                m.points.append(pt)
            arr.markers.append(m)
        return arr

    def publish_state(
        self,
        *,
        tick: int,
        belief_model: Optional[CorridorDurationBeliefModel] = None,
        feedback_packet: Optional[ExecutionFeedbackPacket] = None,
        planned_routes: Optional[Mapping[str, Sequence[tuple[str, int]]]] = None,
        robot_poses: Optional[Mapping[str, tuple[float, float]]] = None,
        operations: Sequence[OperationSnapshot] = (),
    ) -> int:
        """Build and broadcast one L1-L2-L3 RViz2 digital-twin overlay snapshot."""
        payloads = build_rviz_overlay_payloads(
            self.network,
            tick=tick,
            belief_model=belief_model,
            feedback_packet=feedback_packet,
            planned_routes=planned_routes,
            robot_poses=robot_poses,
            operations=operations,
        )
        stamp = self._node.get_clock().now().to_msg()
        topo_subs = self._pub_topo.get_subscription_count()
        if self._cached_topo_msg is None or topo_subs > self._last_topo_subs or self.published_marker_batches % 10 == 0:
            self._cached_topo_msg = self._to_marker_array(payloads["topology_markers"], stamp)
            self._pub_topo.publish(self._cached_topo_msg)
        self._last_topo_subs = topo_subs
        self._pub_belief.publish(self._to_marker_array(payloads["corridor_belief_markers"], stamp))
        self._pub_fleet.publish(self._to_marker_array(payloads["fleet_plan_markers"], stamp))
        self._pub_kitting.publish(self._to_marker_array(payloads["kitting_hud_markers"], stamp))

        for rid, xy_list in payloads["planned_paths"].items():
            pub = self._path_pubs.get(rid)
            if pub is None:
                pub = self._node.create_publisher(self._NavPath, f"/cvcr/planned_paths/{rid}", 10)
                self._path_pubs[rid] = pub
            p_msg = self._NavPath()
            p_msg.header.frame_id = "map"
            p_msg.header.stamp = stamp
            for px, py in xy_list:
                ps = self._PoseStamped()
                ps.header = p_msg.header
                ps.pose.position.x = float(px)
                ps.pose.position.y = float(py)
                ps.pose.orientation.w = 1.0
                p_msg.poses.append(ps)
            pub.publish(p_msg)

        n_m = int(payloads["total_markers"])
        self.published_marker_batches += 1
        self.published_marker_count += n_m
        return n_m

    def close(self) -> None:
        """Destroy the underlying ROS 2 overlay publisher node."""
        self._node.destroy_node()
