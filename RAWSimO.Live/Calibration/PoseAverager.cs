using RAWSimO.Live.Geometry;

namespace RAWSimO.Live.Calibration;

/// <summary>Mean pose and dispersion of a short stationary sample.</summary>
public sealed record PoseAverage(Pose2D Mean, double PositionStdDev, double YawStdDev, int Count);

/// <summary>Averages a stationary burst of poses, using a circular mean for yaw.</summary>
public static class PoseAverager
{
    /// <summary>
    /// Averages at least one pose. Position is the arithmetic mean; yaw is
    /// <c>atan2(sum(sin), sum(cos))</c>, so samples crossing +/-pi are handled.
    /// </summary>
    /// <param name="poses">Poses captured while the vehicle is stationary.</param>
    /// <returns>Mean pose and RMS dispersion; <see cref="PoseAverage.PositionStdDev"/>
    /// is the RMS of the per-sample radial distance from the mean.</returns>
    public static PoseAverage Average(IReadOnlyList<Pose2D> poses)
    {
        if (poses is null || poses.Count == 0)
            throw new ArgumentException("At least one pose is required.", nameof(poses));

        double sx = 0.0, sy = 0.0, sin = 0.0, cos = 0.0;
        foreach (var pose in poses)
        {
            if (!double.IsFinite(pose.X) || !double.IsFinite(pose.Y) || !double.IsFinite(pose.Yaw))
                throw new ArgumentException("Poses must contain finite values.", nameof(poses));
            sx += pose.X;
            sy += pose.Y;
            sin += Math.Sin(pose.Yaw);
            cos += Math.Cos(pose.Yaw);
        }

        var mean = new Pose2D(sx / poses.Count, sy / poses.Count, AngleMath.Normalize(Math.Atan2(sin, cos)));
        double positionSquared = 0.0, yawSquared = 0.0;
        foreach (var pose in poses)
        {
            var dx = pose.X - mean.X;
            var dy = pose.Y - mean.Y;
            positionSquared += dx * dx + dy * dy;
            var dyaw = AngleMath.ShortestDifference(mean.Yaw, pose.Yaw);
            yawSquared += dyaw * dyaw;
        }

        return new PoseAverage(
            mean,
            Math.Sqrt(positionSquared / poses.Count),
            Math.Sqrt(yawSquared / poses.Count),
            poses.Count);
    }
}
