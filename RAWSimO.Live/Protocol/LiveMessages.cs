using System.Text.Json;
using System.Text.Json.Serialization;
using RAWSimO.Live.Geometry;

namespace RAWSimO.Live.Protocol;

/// <summary>Base type for every line-delimited JSON message in the live protocol.</summary>
public abstract record LiveMessage(string Type);

/// <summary>Bridge greeting. The first message after a TCP connection is established.</summary>
public sealed record HelloMessage(string Vehicle, int Proto, string Frame, IReadOnlyList<string>? Caps = null,
    /// <summary>Scene the bridge loaded, so both sides can check they share one map.</summary>
    string? Scene = null)
    : LiveMessage("hello");

/// <summary>Vehicle pose in the chassis frame: metres, radians, bridge clock.</summary>
public sealed record PoseMessage(string Vehicle, long Seq, double T, double X, double Y, double Yaw)
    : LiveMessage("pose");

/// <summary>Periodic status message from the bridge (battery, mode, error).</summary>
public sealed record StatusMessage(string? Vehicle = null, double? Battery = null, string? Mode = null, string? Error = null)
    : LiveMessage("status");

/// <summary>Result of a previously issued nav goal.</summary>
public sealed record NavResultMessage(string? Goal, string Status, string? Reason = null)
    : LiveMessage("nav_result");

/// <summary>Command sent by the GUI to the bridge: follow a path to a goal.</summary>
public sealed record NavCommandMessage(
    string Goal,
    IReadOnlyList<Point2D> Path,
    [property: JsonPropertyName("max_speed")] double? MaxSpeed = null)
    : LiveMessage("nav");

/// <summary>Command sent by the GUI to cancel a goal.</summary>
public sealed record CancelCommandMessage(string Goal) : LiveMessage("cancel");

/// <summary>
/// Asks the bridge for a route to <paramref name="Target"/>. The console never plans: the
/// planner lives in Python (tools/live_bridge/route_service.py), which is the single decision
/// core. <paramref name="From"/> is the vehicle's current map-frame position, which the bridge
/// snaps to the nearest node; leaving it null lets the bridge start at the target itself.
/// </summary>
/// <param name="Direct">
/// When true the caller wants a straight line to the target instead of a network route. It is
/// still answered by Python, so there is exactly one place that decides what gets driven.
/// </param>
public sealed record RouteRequestMessage(string Target, IReadOnlyList<double>? From = null, bool Direct = false)
    : LiveMessage("route_request");

/// <summary>
/// The bridge's answer to <see cref="RouteRequestMessage"/>. <paramref name="Nodes"/> are the
/// scene node ids in travel order and <paramref name="MapPath"/> the matching scene-frame
/// coordinates, so the console can draw the route without recomputing anything.
/// <paramref name="Reason"/> is non-null exactly when the route could not be produced.
/// </summary>
public sealed record RouteResultMessage(
    string? Target,
    IReadOnlyList<string> Nodes,
    [property: JsonPropertyName("map_path")] IReadOnlyList<Point2D> MapPath,
    string? Start = null,
    double? Meters = null,
    double? Seconds = null,
    string? Reason = null,
    /// <summary>RAWSim-O waypoint number the answer refers to, echoed so the console can match it.</summary>
    int? Waypoint = null)
    : LiveMessage("route_result");

/// <summary>Clock-synchronisation request (proto 1).</summary>
public sealed record PingMessage(double T1) : LiveMessage("ping");

/// <summary>Clock-synchronisation reply (proto 1).</summary>
public sealed record PongMessage(double T1, double T2, double T3) : LiveMessage("pong");

/// <summary>Software emergency stop. Latched until <see cref="EstopReleaseMessage"/>.</summary>
public sealed record EstopMessage() : LiveMessage("estop");

/// <summary>Releases a latched software emergency stop.</summary>
public sealed record EstopReleaseMessage() : LiveMessage("estop_release");

/// <summary>A syntactically valid message whose type is not understood. Kept so callers can log it.</summary>
public sealed record UnknownMessage(string Type, [property: JsonIgnore] JsonElement Raw) : LiveMessage(Type);

// ---------------------------------------------------------------------------
// Fleet scheduling stream (amr_sched.live)
//
// This is the **second** producer that speaks the same newline-delimited JSON
// transport. It is deliberately a separate message family rather than a
// reinterpretation of the single-vehicle messages above:
//
//   * the single-vehicle bridge ("pose" / "nav" / "nav_result") describes what
//     one robot is doing and accepts commands;
//   * the scheduling stream ("fleet_hello" / "fleet_frame") describes what the
//     whole fleet is doing and is **read-only** -- the scheduler is the only
//     writer.
//
// Keeping them apart means a GUI can subscribe to either without having to
// guess which producer is on the wire, and neither side has to grow fields it
// cannot fill.
// ---------------------------------------------------------------------------

/// <summary>One node of the static network announced by the scheduling stream.</summary>
public sealed record FleetNode(string Id, string Kind, double X, double Y, int Capacity = 1);

/// <summary>One directed edge of the static network announced by the scheduling stream.</summary>
public sealed record FleetEdge(string Src, string Dst, double Length, int Capacity = 1,
    IReadOnlyList<Point2D>? Centerline = null);

/// <summary>
/// First line of the scheduling stream: the static network plus run metadata.
/// The network never changes during a run, so it is sent exactly once.
/// </summary>
public sealed record FleetHelloMessage(
    string Schema,
    string Source,
    IReadOnlyList<FleetNode> Nodes,
    IReadOnlyList<FleetEdge> Edges,
    FleetMeta Meta)
    : LiveMessage("hello")
{
    /// <summary>Absent on legacy read-only streams.</summary>
    public FleetControlState? Control { get; init; }
}

/// <summary>Run metadata carried by <see cref="FleetHelloMessage"/>.</summary>
public sealed record FleetMeta(
    int NTasks,
    int NAgvs,
    double Dt,
    IReadOnlyList<string> AgvIds)
{
    /// <summary>Task count, exposed under the JSON name used on the wire.</summary>
    [JsonPropertyName("n_tasks")]
    public int NTasks { get; init; } = NTasks;

    /// <summary>Fleet size, exposed under the JSON name used on the wire.</summary>
    [JsonPropertyName("n_agvs")]
    public int NAgvs { get; init; } = NAgvs;

    /// <summary>Simulation timestep in seconds.</summary>
    [JsonPropertyName("dt")]
    public double Dt { get; init; } = Dt;

    /// <summary>Vehicle identifiers in a stable order.</summary>
    [JsonPropertyName("agv_ids")]
    public IReadOnlyList<string> AgvIds { get; init; } = AgvIds;
}

/// <summary>Pose of one vehicle inside a fleet frame.</summary>
/// <param name="AgvId">Vehicle identifier.</param>
/// <param name="Node">Node id when standing on a node, otherwise null.</param>
/// <param name="Edge">"A-&gt;B" when travelling along an edge, otherwise null.</param>
/// <param name="Progress">Fraction travelled along <paramref name="Edge"/>, 0..1.</param>
/// <param name="Source">"sim" / "odom" / "localized" / "gazebo"; explicit pose provenance.</param>
/// <param name="Timestamp">Producer clock in seconds; simulation time on the sim side.</param>
/// <param name="X">Actual x coordinate in the declared frame, when available.</param>
/// <param name="Y">Actual y coordinate in the declared frame, when available.</param>
/// <param name="Yaw">Actual heading in radians, when available.</param>
/// <param name="Available">Explicit truth availability; false forbids extrapolation.</param>
/// <param name="FrameId">Actual coordinate frame; the topology accepts network metres.</param>
public sealed record FleetPose(
    string AgvId,
    string? Node = null,
    string? Edge = null,
    double Progress = 0.0,
    string Source = "sim",
    double? Timestamp = null,
    double? X = null,
    double? Y = null,
    double? Yaw = null,
    bool? Available = null,
    string? FrameId = null);

/// <summary>State of one vehicle inside a fleet frame.</summary>
/// <param name="AgvId">Vehicle identifier.</param>
/// <param name="Pose">Where it is.</param>
/// <param name="State">idle / to_pickup / loading / to_dropoff / unloading / blocked / parking.</param>
/// <param name="TaskId">Task being executed, or null when idle.</param>
/// <param name="Path">Remaining node ids of the current leg; empty means unknown, not "not moving".</param>
public sealed record FleetAgv(
    string AgvId,
    FleetPose Pose,
    string State,
    string? TaskId = null,
    IReadOnlyList<string>? Path = null);

/// <summary>One preemption event carried by a fleet frame.</summary>
public sealed record FleetPreemption(string Urgent, string VictimAgv, string VictimTask, bool WasPicked);

/// <summary>
/// One simulation step of the scheduling stream. Read-only: the GUI must never
/// write back into this channel.
/// </summary>
/// <summary>
/// One node's current reservation claims, its capacity, and how many claimants are stuck.
/// <para>
/// Full is not the same as queued. A parking bay with every slot taken has nobody waiting,
/// while a junction with two blocked vehicles does; reporting both as "congestion" would
/// overstate what the map shows.
/// </para>
/// </summary>
public sealed record FleetHold(int Claims, int Capacity, IReadOnlyList<string> Agvs,
    int Blocked = 0,
    /// <summary>How many consecutive steps this node has been at or over capacity.</summary>
    /// <para>
    /// Without it a bay that is full for the whole shift and a junction stalled for three
    /// steps look identical. The bay has nobody waiting; the junction does.
    /// </para>
    int Steps = 0);

/// <summary>One point of a vehicle's actual trail, in scene metres.</summary>
/// <para>
/// The producer computes it, not the console. A trail is a claim about where the vehicle
/// really was, and only the side that owns the poses may make that claim; the console
/// projects the points it is given.
/// </para>
public sealed record FleetPoint(double X, double Y);

/// <summary>The physical robot's ground truth, as an entity of its own.</summary>
/// <para>
/// Not a replacement pose for a simulated vehicle. Overriding one would give the same
/// vehicle two positions at once — what the engine believes and what Gazebo measures —
/// which is exactly the duplication the architecture forbids. The physical robot is drawn
/// next to the planned fleet; the comparison is plan versus actual, not two opinions.
/// </para>
public sealed record FleetPhysical(
    string Id,
    string Source,
    double[]? Pose,
    IReadOnlyList<FleetPoint> Trail);

/// <summary>Durable physical occupancy reported by the Python resource owner.</summary>
public sealed record FleetResourcePins(
    IReadOnlyDictionary<string, IReadOnlyList<string>> Nodes,
    IReadOnlyDictionary<string, IReadOnlyList<string>> Edges);

/// <summary>Scene and run shared by all reported vehicle executions, independent of command availability.</summary>
public sealed record FleetExecutionIdentity(string SceneId, string RunId);

public sealed record FleetFrameMessage(
    string Schema,
    long Step,
    double T,
    IReadOnlyList<FleetAgv> Agvs,
    IReadOnlyDictionary<string, int>? TaskCounts = null,
    int NBlocked = 0,
    IReadOnlyList<FleetPreemption>? Preemptions = null,
    /// <summary>Node -> vehicles holding it now, from Python's reservation table.</summary>
    IReadOnlyDictionary<string, IReadOnlyList<string>>? Holds = null,
    /// <summary>Nodes at or over capacity — the queues worth showing.</summary>
    IReadOnlyDictionary<string, FleetHold>? Congestion = null,
    /// <summary>Vehicle -> actual positions so far, in scene metres, oldest first.</summary>
    IReadOnlyDictionary<string, IReadOnlyList<FleetPoint>>? Trails = null,
    /// <summary>Which authority produced the trail: sim / odom / localized / gazebo.</summary>
    string? TrailSource = null,
    /// <summary>Per-vehicle trail provenance, when the producer knows it vehicle by vehicle.</summary>
    IReadOnlyDictionary<string, string>? TrailSources = null,
    /// <summary>Ground-truth entity for the physical robot, when one is connected.</summary>
    FleetPhysical? Physical = null)
    : LiveMessage("frame")
{
    /// <summary>Control identity accompanies this observed fleet state.</summary>
    public FleetControlState? Control { get; init; }
    /// <summary>Physical occupancy is retained until confirmed arrival or recovery.</summary>
    public FleetResourcePins? PhysicalHolds { get; init; }
    /// <summary>Observed execution faults remain available after the command endpoint closes.</summary>
    public IReadOnlyDictionary<string, string?>? ExecutionFaults { get; init; }
    /// <summary>Execution provenance remains verifiable after the command endpoint closes.</summary>
    public FleetExecutionIdentity? ExecutionIdentity { get; init; }
}
