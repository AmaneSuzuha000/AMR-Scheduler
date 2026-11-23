using System;

namespace RAWSimO.Live.Drivers;

/// <summary>
/// Placed once at the start of a run: the operator puts the robot on a known point and declares
/// which map pose that point is. Everything afterwards is odometry relative to this anchor.
///
/// Kept separate from <see cref="WheelOdometry"/> so that "where does the run start" and
/// "how do wheel counts become a pose" can be tested and replaced independently.
/// </summary>
public sealed class StartPoseAnchor
{
    /// <summary>Map pose of the anchor point, in metres and radians.</summary>
    public (double X, double Y, double Yaw) MapPose { get; private set; }

    /// <summary>True once <see cref="Set"/> has been called.</summary>
    public bool IsSet { get; private set; }

    /// <summary>Declares the map pose that the robot currently occupies.</summary>
    public void Set(double x, double y, double yaw)
    {
        MapPose = (x, y, WheelOdometry.WrapAngle(yaw));
        IsSet = true;
    }

    /// <summary>Clears the anchor; the next run must set it again.</summary>
    public void Clear() => IsSet = false;

    /// <summary>Builds an integrator that starts at this anchor's map pose.</summary>
    public WheelOdometry CreateOdometry(ScoutMiniGeometry geometry)
    {
        if (!IsSet)
            throw new InvalidOperationException("start pose must be set before odometry can begin");
        return new WheelOdometry(geometry, MapPose.X, MapPose.Y, MapPose.Yaw);
    }
}

/// <summary>
/// Converts a rig-relative pose into the map frame using a start anchor.
/// This is the only place that decides "map coordinates" for the real vehicle, so that the
/// six-point calibration can later replace it without touching the integrator.
/// </summary>
public static class OdometryToMap
{
    /// <summary>
    /// Maps an odometry pose to the map frame. With an anchor at the origin this is the identity,
    /// which keeps the first version honest: no rotation or scale is invented before calibration.
    /// </summary>
    public static (double X, double Y, double Yaw) Apply(
        StartPoseAnchor anchor,
        double odomX,
        double odomY,
        double odomYaw)
    {
        if (anchor is null) throw new ArgumentNullException(nameof(anchor));
        if (!anchor.IsSet) throw new InvalidOperationException("start pose must be set before mapping");

        var (ax, ay, ayaw) = anchor.MapPose;
        var cos = Math.Cos(ayaw);
        var sin = Math.Sin(ayaw);

        return (
            ax + odomX * cos - odomY * sin,
            ay + odomX * sin + odomY * cos,
            WheelOdometry.WrapAngle(ayaw + odomYaw));
    }
}
