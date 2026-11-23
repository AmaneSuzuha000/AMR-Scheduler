using RAWSimO.Live.Geometry;

namespace RAWSimO.Live.Planning;

/// <summary>Converts canvas clicks to map coordinates and picks the nearest waypoint.</summary>
public static class MapPicking
{
    /// <summary>Converts a click in the 1000x1000 canvas to map coordinates (metres).</summary>
    public static Point2D CanvasToMap(double canvasX, double canvasY, double canvasWidth, double canvasHeight, double pixelsPerMeter = 40.0)
    {
        if (!double.IsFinite(canvasWidth) || canvasWidth <= 0) canvasWidth = 1.0;
        if (!double.IsFinite(canvasHeight) || canvasHeight <= 0) canvasHeight = 1.0;
        if (!double.IsFinite(pixelsPerMeter) || pixelsPerMeter <= 0) pixelsPerMeter = 40.0;
        return new Point2D(canvasX / pixelsPerMeter, canvasHeight / pixelsPerMeter - canvasY / pixelsPerMeter);
    }

    /// <summary>Returns the index of the nearest position, or -1 when the list is empty.</summary>
    public static int NearestIndex(IReadOnlyList<Point2D> positions, Point2D mapPoint)
    {
        if (positions == null || positions.Count == 0) return -1;
        var best = -1;
        var bestDistance = double.PositiveInfinity;
        for (var i = 0; i < positions.Count; i++)
        {
            var dx = positions[i].X - mapPoint.X;
            var dy = positions[i].Y - mapPoint.Y;
            var distance = dx * dx + dy * dy;
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = i;
            }
        }
        return best;
    }
}
