using RAWSimO.Live.Protocol;

namespace RAWSimO.Live.Geometry;

/// <summary>Validates authoritative road vertices; never constructs a new route.</summary>
public static class FleetRoadGeometry
{
    /// <summary>Resolves legacy straight roads or checks the supplied directed centerline.</summary>
    public static IReadOnlyList<Point2D> Resolve(FleetEdge edge, FleetNode from, FleetNode to)
    {
        var points = edge.Centerline;
        if (points is null) return new[] { new Point2D(from.X, from.Y), new Point2D(to.X, to.Y) };
        if (points.Count < 2 || !double.IsFinite(edge.Length) || edge.Length <= 0)
            throw new FormatException("Road centerline needs at least two points and a positive length.");
        var length = 0.0;
        for (var i = 0; i < points.Count; i++)
        {
            if (!double.IsFinite(points[i].X) || !double.IsFinite(points[i].Y))
                throw new FormatException("Road centerline coordinates must be finite.");
            if (i == 0) continue;
            var segment = Point2D.Distance(points[i - 1], points[i]);
            if (!double.IsFinite(segment) || segment <= 0)
                throw new FormatException("Road centerline segments must be nonzero and finite.");
            length += segment;
        }
        if (!double.IsFinite(length) || Math.Abs(length - edge.Length) > 1e-6 * Math.Max(1.0, length) ||
            Point2D.Distance(points[0], new Point2D(from.X, from.Y)) > 1e-6 ||
            Point2D.Distance(points[^1], new Point2D(to.X, to.Y)) > 1e-6)
            throw new FormatException("Road centerline must match its directed endpoints and declared length.");
        return points;
    }
}
