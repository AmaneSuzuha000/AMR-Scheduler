using System;
using System.Collections.Generic;

namespace RAWSimO.Live.Drivers;

/// <summary>One row of a drift measurement: how far the robot thinks it went versus how far it did.</summary>
public sealed record DriftSample(
    double TravelledDistance,
    double EstimatedX,
    double EstimatedY,
    double EstimatedYaw,
    double TrueX,
    double TrueY,
    double TrueYaw)
{
    /// <summary>Euclidean position error in metres.</summary>
    public double PositionError => Math.Sqrt(
        (EstimatedX - TrueX) * (EstimatedX - TrueX) +
        (EstimatedY - TrueY) * (EstimatedY - TrueY));

    /// <summary>Heading error in radians, wrapped to (-pi, pi].</summary>
    public double YawError => WheelOdometry.WrapAngle(EstimatedYaw - TrueYaw);
}

/// <summary>
/// Records odometry-versus-ground-truth pairs and reports the drift figures P3-A has to produce.
///
/// The point of this class is that the answer to "is bare wheel odometry good enough?" must come
/// from measurement, not opinion. Every figure it reports is relative drift since the anchor.
/// </summary>
public sealed class DriftReport
{
    private readonly List<DriftSample> _samples = new();

    /// <summary>Samples collected so far.</summary>
    public IReadOnlyList<DriftSample> Samples => _samples;

    /// <summary>Adds a measurement pair.</summary>
    public void Add(DriftSample sample) => _samples.Add(sample ?? throw new ArgumentNullException(nameof(sample)));

    /// <summary>Removes every sample.</summary>
    public void Clear() => _samples.Clear();

    /// <summary>Largest position error over all samples, in metres.</summary>
    public double MaxPositionError
    {
        get
        {
            var max = 0.0;
            foreach (var s in _samples) max = Math.Max(max, s.PositionError);
            return max;
        }
    }

    /// <summary>Position error at the last sample, in metres.</summary>
    public double FinalPositionError => _samples.Count == 0 ? 0.0 : _samples[^1].PositionError;

    /// <summary>
    /// Position error growth per metre travelled. This is the number that decides whether a
    /// longer run needs absolute localisation; a constant ratio means error is dominated by
    /// systematic scale/heading bias rather than by random noise.
    /// </summary>
    public double ErrorPerMetre
    {
        get
        {
            if (_samples.Count == 0) return 0.0;
            var last = _samples[^1];
            return last.TravelledDistance <= 1e-9 ? 0.0 : last.PositionError / last.TravelledDistance;
        }
    }

    /// <summary>
    /// True when the final error exceeds the given tolerance. A run that fails this must be
    /// reported as failing; it is not rounded away.
    /// </summary>
    public bool ExceedsTolerance(double metres) => FinalPositionError > metres;

    /// <summary>Renders a plain-text summary suitable for a log file or an evidence document.</summary>
    public string Summarise()
    {
        if (_samples.Count == 0) return "drift: no samples";

        var last = _samples[^1];
        return string.Format(
            System.Globalization.CultureInfo.InvariantCulture,
            "drift: samples={0} distance={1:F3} m final_error={2:F4} m max_error={3:F4} m error_per_metre={4:F5} final_yaw_error={5:F2} deg",
            _samples.Count,
            last.TravelledDistance,
            FinalPositionError,
            MaxPositionError,
            ErrorPerMetre,
            last.YawError * 180.0 / Math.PI);
    }
}
