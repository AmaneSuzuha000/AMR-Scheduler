using System;
using AmrInterop;
using System.Collections.Generic;
using RAWSimO.Live.Protocol;
using RAWSimO.Live.Geometry;

namespace RAWSimO.Live.Tracking;

/// <summary>A drawable vehicle position derived from a <see cref="FleetFrameMessage"/>.</summary>
public sealed record FleetVehiclePlacement(
    string AgvId,
    double X,
    double Y,
    string State,
    string? TaskId,
    bool OnEdge);

/// <summary>Why a frame could not be turned into placements.</summary>
public enum FleetPlacementFailure
{
    /// <summary>No failure.</summary>
    None,

    /// <summary>The network announced at hello does not contain a node or edge the frame refers to.</summary>
    UnknownTopology,

    /// <summary>A pose carried neither a node nor an edge, so it has no position at all.</summary>
    NoPosition,
}

/// <summary>Result of placing one frame.</summary>
public sealed record FleetPlacementResult(
    IReadOnlyList<FleetVehiclePlacement> Placements,
    FleetPlacementFailure Failure,
    string? Detail)
{
    /// <summary>Whether every vehicle in the frame could be placed.</summary>
    public bool Complete => Failure == FleetPlacementFailure.None;
}

/// <summary>
/// Turns scheduling frames into drawable positions using the network from the hello.
/// <para>
/// Algorithm frames describe a node or an edge with progress. External execution
/// frames declare actual coordinates in network metres and explicit availability.
/// Placement validates that distinction before projecting into pixels.
/// </para>
/// <para>
/// The alternative -- silently dropping a vehicle that cannot be placed -- produces a map
/// that looks healthy while a robot has vanished from it. That is the failure mode this
/// type exists to prevent, so callers get the failure and the id that caused it.
/// </para>
/// </summary>
public sealed class FleetTopology
{
    private readonly Dictionary<string, FleetNode> _nodes = new(StringComparer.Ordinal);
    private readonly Dictionary<string, FleetEdge> _edges = new(StringComparer.Ordinal);
    private readonly Dictionary<string, IReadOnlyList<Point2D>> _roads = new(StringComparer.Ordinal);

    /// <summary>Builds a topology from a fleet hello.</summary>
    /// <param name="hello">The hello that announced the network.</param>
    public FleetTopology(FleetHelloMessage hello)
    {
        ArgumentNullException.ThrowIfNull(hello);
        foreach (var node in hello.Nodes) _nodes[node.Id] = node;
        foreach (var edge in hello.Edges)
        {
            var key = EdgeKey(edge.Src, edge.Dst);
            _edges[key] = edge;
            if (Node(edge.Src) is { } from && Node(edge.Dst) is { } to)
                _roads[key] = FleetRoadGeometry.Resolve(edge, from, to).ToArray();
        }
    }

    /// <summary>Number of announced nodes.</summary>
    public int NodeCount => _nodes.Count;

    /// <summary>Number of announced edges.</summary>
    public int EdgeCount => _edges.Count;

    /// <summary>Looks up a node by id.</summary>
    /// <param name="id">Node id.</param>
    /// <returns>The node, or null when it was never announced.</returns>
    public FleetNode? Node(string? id) =>
        id is not null && _nodes.TryGetValue(id, out var n) ? n : null;

    /// <summary>Key used to index a directed edge.</summary>
    public static string EdgeKey(string src, string dst) =>
        src + PoseMath.EdgeSeparator + dst;

    /// <summary>Original vertices of a declared directed road, or null for an unknown road.</summary>
    public IReadOnlyList<Point2D>? EdgePoints(string src, string dst) =>
        _roads.TryGetValue(EdgeKey(src, dst), out var points) ? points : null;

    /// <summary>Joins only the roads Python named, preserving every authoritative corner.</summary>
    public IReadOnlyList<Point2D>? RoutePoints(IReadOnlyList<string> nodes)
    {
        if (nodes.Count == 0) return Array.Empty<Point2D>();
        var first = Node(nodes[0]);
        if (first is null) return null;
        var points = new List<Point2D> { new(first.X, first.Y) };
        for (var i = 1; i < nodes.Count; i++)
        {
            var road = EdgePoints(nodes[i - 1], nodes[i]);
            if (road is null) return null;
            points.AddRange(road.Skip(1));
        }
        return points;
    }

    /// <summary>
    /// Places every vehicle of a frame. Vehicles are never skipped silently: the first
    /// one that cannot be placed stops the result and names itself.
    /// </summary>
    /// <param name="frame">The frame to place.</param>
    /// <returns>The placements, or the first failure and its detail.</returns>
    public FleetPlacementResult Place(FleetFrameMessage frame)
    {
        ArgumentNullException.ThrowIfNull(frame);
        var placed = new List<FleetVehiclePlacement>(frame.Agvs.Count);

        foreach (var agv in frame.Agvs)
        {
            var pose = agv.Pose;

            // Missing truth must disappear; do not fall back to predicted topology.
            if (pose.Available == false)
                continue;
            if (pose.Available == true)
            {
                if (pose.X is null || pose.Y is null || pose.FrameId != "network")
                    return new FleetPlacementResult(placed, FleetPlacementFailure.NoPosition,
                        agv.AgvId + " carries an invalid actual pose");
                placed.Add(new FleetVehiclePlacement(
                    agv.AgvId, pose.X.Value, pose.Y.Value, agv.State, agv.TaskId, OnEdge: false));
                continue;
            }

            if (pose.Node is not null)
            {
                var node = Node(pose.Node);
                if (node is null)
                    return Unknown(placed, agv.AgvId, "node " + pose.Node);
                placed.Add(new FleetVehiclePlacement(
                    agv.AgvId, node.X, node.Y, agv.State, agv.TaskId, OnEdge: false));
                continue;
            }

            if (pose.Edge is not null)
            {
                if (!PoseMath.TrySplitEdge(pose.Edge, out var src, out var dst))
                    return Unknown(placed, agv.AgvId, "edge " + pose.Edge);

                var road = EdgePoints(src, dst);
                if (road is null)
                    return Unknown(placed, agv.AgvId, "edge " + pose.Edge);

                // The clamp and the interpolation live in PoseMath.cs, compiled into this
                // assembly as a linked file and into the interop probe by csc. Re-deriving
                // them here is exactly the duplication this class used to take part in.
                PoseMath.InterpolatePolyline(road.Select(p => new[] { p.X, p.Y }).ToArray(),
                                            pose.Progress, out var x, out var y);
                placed.Add(new FleetVehiclePlacement(
                    agv.AgvId, x, y, agv.State, agv.TaskId, OnEdge: true));
                continue;
            }

            return new FleetPlacementResult(placed, FleetPlacementFailure.NoPosition,
                agv.AgvId + " carries neither a node nor an edge");
        }

        return new FleetPlacementResult(placed, FleetPlacementFailure.None, null);
    }

    /// <summary>
    /// Reports a frame the topology cannot fully explain.
    /// <para>
    /// The vehicles placed before the bad one are kept. They are still true — this frame really
    /// does put them there — and discarding them would turn one unknown id into an empty map,
    /// which reads as "the whole fleet died" rather than "one id is wrong".
    /// </para>
    /// </summary>
    private static FleetPlacementResult Unknown(
        IReadOnlyList<FleetVehiclePlacement> placed, string agvId, string what) =>
        new(placed, FleetPlacementFailure.UnknownTopology,
            agvId + " refers to " + what + ", which the hello never announced");
}
