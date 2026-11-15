namespace RAWSimO.Live.Geometry;

/// <summary>
/// Converts between the map frame (metres, +y up) and the WPF canvas frame
/// (pixels, +y down) used by <c>SimulationVisual2D</c>.
/// </summary>
public sealed class CanvasTransform
{
    /// <summary>Scale used by RAWSim-O's 2D visualisation: 40 pixels per metre.</summary>
    public const double DefaultPixelsPerMeter = 40.0;

    /// <summary>Creates a transform for a tier of the given size.</summary>
    public CanvasTransform(double mapWidthMeters, double mapHeightMeters, double pixelsPerMeter = DefaultPixelsPerMeter)
    {
        if (!double.IsFinite(mapWidthMeters) || mapWidthMeters <= 0) throw new ArgumentOutOfRangeException(nameof(mapWidthMeters));
        if (!double.IsFinite(mapHeightMeters) || mapHeightMeters <= 0) throw new ArgumentOutOfRangeException(nameof(mapHeightMeters));
        if (!double.IsFinite(pixelsPerMeter) || pixelsPerMeter <= 0) throw new ArgumentOutOfRangeException(nameof(pixelsPerMeter));
        MapWidthMeters = mapWidthMeters;
        MapHeightMeters = mapHeightMeters;
        PixelsPerMeter = pixelsPerMeter;
    }

    /// <summary>Tier width in metres (map x range).</summary>
    public double MapWidthMeters { get; }

    /// <summary>Tier height in metres (map y range).</summary>
    public double MapHeightMeters { get; }

    /// <summary>Pixels per metre.</summary>
    public double PixelsPerMeter { get; }

    /// <summary>Canvas width in pixels.</summary>
    public double CanvasWidth => MapWidthMeters * PixelsPerMeter;

    /// <summary>Canvas height in pixels.</summary>
    public double CanvasHeight => MapHeightMeters * PixelsPerMeter;

    /// <summary>Converts a map point to canvas pixels.</summary>
    public Point2D MapToCanvas(Point2D map) => new(map.X * PixelsPerMeter, (MapHeightMeters - map.Y) * PixelsPerMeter);

    /// <summary>Converts canvas pixels to a map point.</summary>
    public Point2D CanvasToMap(Point2D canvas) => new(canvas.X / PixelsPerMeter, MapHeightMeters - canvas.Y / PixelsPerMeter);

    /// <summary>
    /// Converts a map yaw (radians, counter-clockwise from +x) to the WPF
    /// rotation angle in degrees (clockwise from +x).
    /// </summary>
    public double MapYawToCanvasDegrees(double yawRadians)
    {
        if (!double.IsFinite(yawRadians)) return double.NaN;
        return AngleMath.NormalizeDegrees(360.0 - yawRadians * 180.0 / Math.PI);
    }

    /// <summary>Converts a WPF rotation angle in degrees back to a map yaw in radians.</summary>
    public double CanvasDegreesToMapYaw(double degrees)
    {
        if (!double.IsFinite(degrees)) return double.NaN;
        return AngleMath.Normalize((360.0 - degrees) * Math.PI / 180.0);
    }

    /// <summary>Converts a map pose to a canvas position plus rotation angle.</summary>
    public (Point2D Canvas, double Degrees) MapToCanvas(Pose2D pose) => (MapToCanvas(pose.Position), MapYawToCanvasDegrees(pose.Yaw));

    /// <summary>Converts a canvas position plus rotation angle to a map pose.</summary>
    public Pose2D CanvasToMap(Point2D canvas, double degrees) => new(CanvasToMap(canvas), CanvasDegreesToMapYaw(degrees));
}
