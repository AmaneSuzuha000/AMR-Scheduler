using System;
using System.Collections.Generic;
using RAWSimO.Live.Geometry;
using RAWSimO.Live.Protocol;

namespace RAWSimO.Live.Tracking;

/// <summary>One vehicle as it should be drawn: pixels, colour, and the state to put beside it.</summary>
public sealed record FleetMarker(
    string AgvId,
    double CanvasX,
    double CanvasY,
    byte R,
    byte G,
    byte B,
    string State,
    string? TaskId,
    bool OnEdge);

/// <summary>
/// Turns placed vehicles into drawable pixels, and reports what it could not draw.
/// <para>
/// This is the step between "the frame arrived" and "something is on screen", and it is the
/// step that used to not exist: the scheduling stream was consumed by a console harness that
/// printed coordinates, so nothing proved a <see cref="FleetFrameMessage"/> could ever become a
/// picture. Keeping the projection here rather than inside the window means the mapping can be
/// tested without a display, which is the only way it gets tested at all.
/// </para>
/// <para>
/// The scene is sized by the network it draws, not by a fixed constant. The scheduling stream
/// reports positions in workshop metres -- tens of metres across -- while the single-vehicle
/// bridge works in a few metres of a demo bench. A scene built for one of those scales silently
/// projects every vehicle of the other one off the edge of the canvas, which looks exactly like
/// an empty fleet.
/// </para>
/// </summary>
public sealed class FleetScene
{
    private readonly Dictionary<string, int> _slots = new(StringComparer.Ordinal);
    private readonly FleetColorAllocator _colors = new();

    private FleetScene(double minX, double minY, double widthMeters, double heightMeters,
                       double pixelsPerMeter)
    {
        MinX = minX;
        MinY = minY;
        Transform = new CanvasTransform(widthMeters, heightMeters, pixelsPerMeter);
    }

    /// <summary>Creates a scene over an arbitrary map rectangle, for callers that know their own frame.</summary>
    /// <param name="minX">Map x that maps to the left edge.</param>
    /// <param name="minY">Map y that maps to the bottom edge.</param>
    /// <param name="widthMeters">Width of the drawn area in metres.</param>
    /// <param name="heightMeters">Height of the drawn area in metres.</param>
    /// <param name="pixelsPerMeter">Canvas scale.</param>
    public static FleetScene Over(double minX, double minY, double widthMeters, double heightMeters,
                                  double pixelsPerMeter = CanvasTransform.DefaultPixelsPerMeter) =>
        new(minX, minY, widthMeters, heightMeters, pixelsPerMeter);

    /// <summary>
    /// Creates a scene that fits the network announced by a fleet hello, with a margin so a
    /// vehicle sitting on the outermost node is not drawn half off the edge.
    /// </summary>
    /// <param name="hello">The hello that announced the network.</param>
    /// <param name="pixelsPerMeter">Canvas scale.</param>
    /// <param name="marginMeters">Blank space added on every side.</param>
    public static FleetScene ForNetwork(FleetHelloMessage hello,
                                        double pixelsPerMeter = CanvasTransform.DefaultPixelsPerMeter,
                                        double marginMeters = 5.0)
    {
        ArgumentNullException.ThrowIfNull(hello);
        if (hello.Nodes.Count == 0)
            throw new ArgumentException("the hello announced no nodes", nameof(hello));

        double minX = double.PositiveInfinity, minY = double.PositiveInfinity;
        double maxX = double.NegativeInfinity, maxY = double.NegativeInfinity;
        foreach (var node in NetworkPoints(hello))
        {
            if (node.X < minX) minX = node.X;
            if (node.Y < minY) minY = node.Y;
            if (node.X > maxX) maxX = node.X;
            if (node.Y > maxY) maxY = node.Y;
        }
        if (!double.IsFinite(minX) || !double.IsFinite(minY) ||
            !double.IsFinite(maxX) || !double.IsFinite(maxY))
            throw new ArgumentException("the network has non-finite node coordinates", nameof(hello));

        var margin = marginMeters > 0 ? marginMeters : 0.0;
        // A single-node network would have zero extent; a zero-sized canvas divides by zero.
        var width = Math.Max(maxX - minX, 1.0) + 2 * margin;
        var height = Math.Max(maxY - minY, 1.0) + 2 * margin;
        return new FleetScene(minX - margin, minY - margin, width, height, pixelsPerMeter);
    }

    /// <summary>
    /// Fits a network into a canvas of a known pixel size.
    /// <para>
    /// The scale cannot be a constant. The scheduling network is about 120 m across; a fixed
    /// 40 px/m would make a 4800 px canvas wider than any window, so the whole fleet would be
    /// drawn outside the visible area while the code reported success. Choosing the scale from
    /// the canvas it has to fit is the only way the picture and the data agree.
    /// </para>
    /// </summary>
    /// <param name="hello">The hello that announced the network.</param>
    /// <param name="canvasWidthPx">Width available on the canvas, in pixels.</param>
    /// <param name="canvasHeightPx">Height available on the canvas, in pixels.</param>
    /// <param name="paddingPx">Blank space kept on every side, in pixels.</param>
    /// <returns>A scene whose canvas is the requested size, or null when it cannot be fitted.</returns>
    public static FleetScene? FitNetwork(FleetHelloMessage hello,
                                         double canvasWidthPx, double canvasHeightPx,
                                         double paddingPx = 24.0)
    {
        ArgumentNullException.ThrowIfNull(hello);
        if (hello.Nodes.Count == 0) return null;
        if (!(canvasWidthPx > 0) || !(canvasHeightPx > 0)) return null;
        if (!double.IsFinite(canvasWidthPx) || !double.IsFinite(canvasHeightPx)) return null;

        double minX = double.PositiveInfinity, minY = double.PositiveInfinity;
        double maxX = double.NegativeInfinity, maxY = double.NegativeInfinity;
        foreach (var node in NetworkPoints(hello))
        {
            if (node.X < minX) minX = node.X;
            if (node.Y < minY) minY = node.Y;
            if (node.X > maxX) maxX = node.X;
            if (node.Y > maxY) maxY = node.Y;
        }
        if (!double.IsFinite(minX) || !double.IsFinite(maxX)) return null;

        var spanX = Math.Max(maxX - minX, 1e-6);
        var spanY = Math.Max(maxY - minY, 1e-6);
        var usableW = Math.Max(canvasWidthPx - 2 * paddingPx, 1.0);
        var usableH = Math.Max(canvasHeightPx - 2 * paddingPx, 1.0);
        // One scale for both axes: independent scales would stretch the workshop and make
        // distances on screen lie about distances on the floor.
        var scale = Math.Min(usableW / spanX, usableH / spanY);
        if (!(scale > 0) || !double.IsFinite(scale)) return null;

        var marginX = (canvasWidthPx / scale - spanX) / 2.0;
        var marginY = (canvasHeightPx / scale - spanY) / 2.0;
        return new FleetScene(minX - marginX, minY - marginY,
                              canvasWidthPx / scale, canvasHeightPx / scale, scale);
    }

    /// <summary>Map x that projects to the left edge of the canvas.</summary>
    public double MinX { get; }

    private static IEnumerable<Point2D> NetworkPoints(FleetHelloMessage hello) =>
        hello.Nodes.Select(n => new Point2D(n.X, n.Y)).Concat(
            hello.Edges.SelectMany(e => e.Centerline ?? Array.Empty<Point2D>()));

    /// <summary>Map y that projects to the bottom edge of the canvas.</summary>
    public double MinY { get; }

    /// <summary>The map-to-canvas transform this scene draws with.</summary>
    public CanvasTransform Transform { get; }

    /// <summary>Number of vehicles this scene has coloured.</summary>
    public int VehicleCount => _slots.Count;

    /// <summary>Canvas width in pixels.</summary>
    public double CanvasWidth => Transform.CanvasWidth;

    /// <summary>Canvas height in pixels.</summary>
    public double CanvasHeight => Transform.CanvasHeight;

    /// <summary>Returns the palette slot a vehicle was assigned, assigning one on first sight.</summary>
    public int SlotOf(string agvId)
    {
        ArgumentNullException.ThrowIfNull(agvId);
        if (_slots.TryGetValue(agvId, out var slot)) return slot;
        slot = _colors.Acquire();
        _slots[agvId] = slot;
        return slot;
    }

    /// <summary>Forgets a vehicle, returning its colour to the pool.</summary>
    public void Forget(string agvId)
    {
        if (_slots.Remove(agvId, out var slot)) _colors.Release(slot);
    }

    /// <summary>Projects one map position onto the canvas.</summary>
    public Point2D Project(double x, double y) =>
        Transform.MapToCanvas(new Point2D(x - MinX, y - MinY));

    /// <summary>
    /// Projects a placement result onto the canvas.
    /// <para>
    /// A placement that failed still returns the vehicles that <em>were</em> placed: dropping
    /// them because one vehicle is unknown would blank the whole picture for a single bad id,
    /// which is a worse failure than the one being reported.
    /// </para>
    /// </summary>
    /// <param name="result">Placements from <see cref="FleetTopology.Place"/>.</param>
    /// <returns>One marker per placed vehicle.</returns>
    public IReadOnlyList<FleetMarker> Project(FleetPlacementResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        var markers = new List<FleetMarker>(result.Placements.Count);
        foreach (var p in result.Placements)
        {
            var rgb = FleetColorAllocator.ColorOf(SlotOf(p.AgvId));
            var canvas = Project(p.X, p.Y);
            markers.Add(new FleetMarker(p.AgvId, canvas.X, canvas.Y, rgb.R, rgb.G, rgb.B,
                                        p.State, p.TaskId, p.OnEdge));
        }
        return markers;
    }

    /// <summary>True when a marker would be drawn outside the canvas, i.e. nobody could see it.</summary>
    public bool IsOffCanvas(FleetMarker marker)
    {
        ArgumentNullException.ThrowIfNull(marker);
        return marker.CanvasX < 0 || marker.CanvasY < 0 ||
               marker.CanvasX > CanvasWidth || marker.CanvasY > CanvasHeight;
    }
}
