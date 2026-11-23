using System.Numerics;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using RAWSimO.Live.Geometry;

namespace RAWSimO.Live.Calibration;

/// <summary>Calibration model. SE(2) is a rigid transform; Sim(2) also estimates scale.</summary>
public enum CalibrationModel
{
    /// <summary>Rigid transform with three parameters: theta, tx, ty.</summary>
    SE2,
    /// <summary>Similarity transform with four parameters: theta, tx, ty, scale.</summary>
    Sim2,
}

/// <summary>One calibration correspondence in map and chassis coordinates.</summary>
public sealed record CalibrationPoint(
    Point2D Map,
    Point2D Chassis,
    double YawChassis = 0.0,
    double LooError = 0.0,
    double Residual = 0.0);

/// <summary>
/// Calibration result and the chassis-to-map transform. The transform is
/// <c>z = q * w + t</c>, where <c>q = scale * exp(i*theta)</c>. When
/// <see cref="Mirror"/> is true, the chassis point is conjugated first:
/// <c>w = conj(z_chassis)</c>.
/// </summary>
public sealed class CalibrationResult
{
    /// <summary>Format version.</summary>
    public int Version { get; init; } = 1;

    /// <summary>Vehicle identifier, e.g. "agv1".</summary>
    public string Vehicle { get; init; } = "agv1";

    /// <summary>Frame of the map coordinates: "map", "odom" or "sim".</summary>
    public string Frame { get; init; } = "map";

    /// <summary>Model name, "SE2" or "Sim2".</summary>
    public string Model { get; init; } = "SE2";

    /// <summary>Whether the chassis frame is left-handed.</summary>
    public bool Mirror { get; init; }

    /// <summary>Rotation angle in radians.</summary>
    public double Theta { get; init; }

    /// <summary>Translation along map x in metres.</summary>
    public double Tx { get; init; }

    /// <summary>Translation along map y in metres.</summary>
    public double Ty { get; init; }

    /// <summary>Diagnostic scale from the Sim(2) fit; 1 for SE(2).</summary>
    public double ScaleDiag { get; init; } = 1.0;

    /// <summary>Yaw offset between the chassis yaw zero and its forward direction.</summary>
    public double YawOffset { get; init; }

    /// <summary>Per-axis reading standard deviation used for the pass/fail threshold.</summary>
    public double Sigma { get; init; }

    /// <summary>Root-mean-square position residual of the fit, in metres.</summary>
    public double ResidualRms { get; init; }

    /// <summary>The calibration points with residual and leave-one-out error.</summary>
    public List<CalibrationPoint> Points { get; set; } = new();

    /// <summary>Creation time.</summary>
    public DateTimeOffset Created { get; init; } = DateTimeOffset.Now;

    /// <summary>SHA-256 of the map file, when known.</summary>
    public string? MapFileSha256 { get; init; }

    [JsonIgnore]
    private Complex Q => Complex.FromPolarCoordinates(ScaleDiag, Theta);

    [JsonIgnore]
    private Complex T => new(Tx, Ty);

    /// <summary>Converts a chassis point to a map point.</summary>
    public Point2D ChassisToMap(Point2D chassis)
    {
        var w = new Complex(chassis.X, chassis.Y);
        if (Mirror) w = Complex.Conjugate(w);
        var z = Q * w + T;
        return new Point2D(z.Real, z.Imaginary);
    }

    /// <summary>Converts a chassis pose to a map pose, including the yaw offset.</summary>
    public Pose2D ChassisToMap(Pose2D chassis)
    {
        var position = ChassisToMap(chassis.Position);
        var yaw = Mirror ? Theta - chassis.Yaw : Theta + chassis.Yaw;
        return new Pose2D(position, AngleMath.Normalize(yaw + YawOffset));
    }

    /// <summary>Converts a map point back to a chassis point.</summary>
    public Point2D MapToChassis(Point2D map)
    {
        var w = (new Complex(map.X, map.Y) - T) / Q;
        if (Mirror) w = Complex.Conjugate(w);
        return new Point2D(w.Real, w.Imaginary);
    }

    /// <summary>Converts a map pose back to a chassis pose, including the yaw offset.</summary>
    public Pose2D MapToChassis(Pose2D map)
    {
        var position = MapToChassis(map.Position);
        var yaw = Mirror ? Theta + YawOffset - map.Yaw : map.Yaw - Theta - YawOffset;
        return new Pose2D(position, AngleMath.Normalize(yaw));
    }

    /// <summary>Serialises the calibration to JSON.</summary>
    public string ToJson(bool indented = true) => JsonSerializer.Serialize(this, JsonOptions(indented));

    /// <summary>Loads a calibration from JSON.</summary>
    public static CalibrationResult FromJson(string json)
    {
        var result = JsonSerializer.Deserialize<CalibrationResult>(json, JsonOptions(false))
            ?? throw new FormatException("Calibration JSON is empty.");
        if (!double.IsFinite(result.Theta) || !double.IsFinite(result.Tx) ||
            !double.IsFinite(result.Ty) || !double.IsFinite(result.ScaleDiag) ||
            result.ScaleDiag <= 0.0)
            throw new FormatException("Calibration JSON contains invalid transform values.");
        return result;
    }

    /// <summary>Writes a calibration file.</summary>
    public void Save(string path) => File.WriteAllText(path, ToJson());

    /// <summary>Reads a calibration file.</summary>
    public static CalibrationResult Load(string path) => FromJson(File.ReadAllText(path));

    /// <summary>Computes the SHA-256 hex digest of a file.</summary>
    public static string ComputeFileSha256(string path)
    {
        using var sha = SHA256.Create();
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(sha.ComputeHash(stream)).ToLowerInvariant();
    }

    private static JsonSerializerOptions JsonOptions(bool indented) => new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = indented,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };
}

/// <summary>Closed-form SE(2)/Sim(2) calibration solver.</summary>
public static class CalibrationSolver
{
    /// <summary>
    /// Fits a calibration from at least three non-collinear correspondences.
    /// </summary>
    /// <param name="points">Measured map/chassis correspondences.</param>
    /// <param name="model">SE(2) or Sim(2).</param>
    /// <param name="mirror">Whether to conjugate chassis coordinates before fitting.</param>
    /// <param name="sigma">Per-axis reading standard deviation, used for diagnostics.</param>
    /// <param name="yawOffset">Yaw offset to apply when displaying chassis poses.</param>
    /// <param name="vehicle">Vehicle identifier.</param>
    /// <param name="frame">Map frame name.</param>
    /// <param name="mapFileSha256">Optional map file hash.</param>
    public static CalibrationResult Fit(
        IReadOnlyList<CalibrationPoint> points,
        CalibrationModel model = CalibrationModel.SE2,
        bool mirror = false,
        double sigma = 0.0,
        double yawOffset = 0.0,
        string vehicle = "agv1",
        string frame = "map",
        string? mapFileSha256 = null)
    {
        var core = FitCore(points, model, mirror, sigma, yawOffset, vehicle, frame, mapFileSha256);
        var withLoo = new List<CalibrationPoint>(points.Count);
        for (var i = 0; i < points.Count; i++)
        {
            var loo = 0.0;
            if (points.Count > 3)
            {
                var subset = new List<CalibrationPoint>(points.Count - 1);
                for (var j = 0; j < points.Count; j++)
                    if (j != i) subset.Add(points[j]);
                var reduced = FitCore(subset, model, mirror, sigma, yawOffset, vehicle, frame, mapFileSha256);
                loo = Point2D.Distance(reduced.ChassisToMap(points[i].Chassis), points[i].Map);
            }
            withLoo.Add(points[i] with { LooError = loo, Residual = core.Points[i].Residual });
        }
        core.Points = withLoo;
        return core;
    }

    /// <summary>
    /// Fits both handedness variants and returns the one with the smaller
    /// residual. The mirror flag in the result tells which one won.
    /// </summary>
    public static CalibrationResult FitBest(
        IReadOnlyList<CalibrationPoint> points,
        CalibrationModel model = CalibrationModel.SE2,
        double sigma = 0.0,
        double yawOffset = 0.0,
        string vehicle = "agv1",
        string frame = "map",
        string? mapFileSha256 = null)
    {
        CalibrationResult normal;
        CalibrationResult mirrored;
        try { normal = Fit(points, model, false, sigma, yawOffset, vehicle, frame, mapFileSha256); }
        catch (ArgumentException) { return Fit(points, model, true, sigma, yawOffset, vehicle, frame, mapFileSha256); }
        try { mirrored = Fit(points, model, true, sigma, yawOffset, vehicle, frame, mapFileSha256); }
        catch (ArgumentException) { return normal; }
        return mirrored.ResidualRms < normal.ResidualRms ? mirrored : normal;
    }

    /// <summary>
    /// Derives the transform from a fixed starting fixture, for odometry-only
    /// chassis. The vehicle is placed at <paramref name="mapOrigin"/> with yaw
    /// <paramref name="mapYaw"/>; the chassis reports the corresponding pose.
    /// </summary>
    public static CalibrationResult FromOrigin(
        Point2D mapOrigin,
        double mapYaw,
        Point2D chassisOrigin,
        double chassisYaw,
        bool mirror = false,
        double yawOffset = 0.0,
        string vehicle = "agv1",
        string frame = "odom")
    {
        var theta = mirror
            ? mapYaw + chassisYaw - yawOffset
            : mapYaw - chassisYaw - yawOffset;
        theta = AngleMath.Normalize(theta);
        var q = Complex.FromPolarCoordinates(1.0, theta);
        var w = new Complex(chassisOrigin.X, chassisOrigin.Y);
        if (mirror) w = Complex.Conjugate(w);
        var t = new Complex(mapOrigin.X, mapOrigin.Y) - q * w;
        return new CalibrationResult
        {
            Vehicle = vehicle,
            Frame = frame,
            Model = "SE2",
            Mirror = mirror,
            Theta = theta,
            Tx = t.Real,
            Ty = t.Imaginary,
            ScaleDiag = 1.0,
            YawOffset = yawOffset,
        };
    }

    /// <summary>Returns the pass/fail factor k for the residual threshold (n=4..6).</summary>
    public static double ResidualFactor(int pointCount) => pointCount switch
    {
        4 => 1.94,
        5 => 1.92,
        6 => 1.90,
        _ => 1.90,
    };

    private static CalibrationResult FitCore(
        IReadOnlyList<CalibrationPoint> points,
        CalibrationModel model,
        bool mirror,
        double sigma,
        double yawOffset,
        string vehicle,
        string frame,
        string? mapFileSha256)
    {
        if (points is null || points.Count < 3)
            throw new ArgumentException("At least three calibration points are required.", nameof(points));

        var meanW = Complex.Zero;
        var meanZ = Complex.Zero;
        foreach (var point in points)
        {
            meanW += ToComplex(point.Chassis, mirror);
            meanZ += new Complex(point.Map.X, point.Map.Y);
        }
        meanW /= points.Count;
        meanZ /= points.Count;

        var m = Complex.Zero;
        var sumAA = 0.0;
        foreach (var point in points)
        {
            var a = ToComplex(point.Chassis, mirror) - meanW;
            var b = new Complex(point.Map.X, point.Map.Y) - meanZ;
            m += b * Complex.Conjugate(a);
            sumAA += a.Magnitude * a.Magnitude;
        }

        if (sumAA < 1e-12 || m.Magnitude < 1e-12)
            throw new ArgumentException("Calibration points are degenerate (coincident or collinear).", nameof(points));

        var q = model == CalibrationModel.SE2 ? m / m.Magnitude : m / sumAA;
        var t = meanZ - q * meanW;
        var scale = q.Magnitude;
        var result = new CalibrationResult
        {
            Vehicle = vehicle,
            Frame = frame,
            Model = model == CalibrationModel.SE2 ? "SE2" : "Sim2",
            Mirror = mirror,
            Theta = AngleMath.Normalize(q.Phase),
            Tx = t.Real,
            Ty = t.Imaginary,
            ScaleDiag = scale,
            YawOffset = yawOffset,
            Sigma = sigma,
            MapFileSha256 = mapFileSha256,
        };

        var squared = 0.0;
        var stored = new List<CalibrationPoint>(points.Count);
        foreach (var point in points)
        {
            var predicted = result.ChassisToMap(point.Chassis);
            var residual = Point2D.Distance(predicted, point.Map);
            squared += residual * residual;
            stored.Add(point with { Residual = residual, LooError = 0.0 });
        }
        result.Points = stored;
        result = new CalibrationResult
        {
            Vehicle = result.Vehicle,
            Frame = result.Frame,
            Model = result.Model,
            Mirror = result.Mirror,
            Theta = result.Theta,
            Tx = result.Tx,
            Ty = result.Ty,
            ScaleDiag = result.ScaleDiag,
            YawOffset = result.YawOffset,
            Sigma = result.Sigma,
            ResidualRms = Math.Sqrt(squared / points.Count),
            Points = result.Points,
            Created = result.Created,
            MapFileSha256 = result.MapFileSha256,
        };
        return result;
    }

    private static Complex ToComplex(Point2D point, bool mirror)
    {
        var value = new Complex(point.X, point.Y);
        return mirror ? Complex.Conjugate(value) : value;
    }
}
