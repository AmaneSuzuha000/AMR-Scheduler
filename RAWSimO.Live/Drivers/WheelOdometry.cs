using System;

namespace RAWSimO.Live.Drivers;

/// <summary>Physical geometry of the differential-drive chassis, in SI units.</summary>
public sealed record ScoutMiniGeometry(
    double WheelRadius,
    double TrackWidth,
    int CountsPerRevolution)
{
    /// <summary>
    /// SCOUT Mini defaults. The wheel radius (0.0955 m) and track width (0.3645 m) come from the
    /// AgileX SCOUT Mini mechanical drawing; the encoder resolution (1024 counts/rev at the motor,
    /// before the gearbox) must be confirmed on the real chassis in P3-A before any accuracy claim.
    /// </summary>
    public static ScoutMiniGeometry Default { get; } = new(0.0955, 0.3645, 1024);
}

/// <summary>
/// Wheel-odometry integrator for a differential / skid-steer chassis.
///
/// This is the piece the AgileX SDK does not provide: 0x311 gives cumulative left/right wheel
/// counts, and the pose has to be integrated from the increments. Only the standard
/// second-order (midpoint) integration is used, which is exact for constant-curvature arcs.
///
/// The result is RELATIVE. It is never absolute localisation: skid-steer wheels slip while
/// turning, so error accumulates with distance. Callers must present it as "estimated from
/// odometry since the declared start pose", never as a measured position.
/// </summary>
public sealed class WheelOdometry
{
    private readonly ScoutMiniGeometry _geometry;
    private WheelCounters? _previous;
    private long _droppedFrames;

    /// <summary>Creates an integrator seated at the given pose.</summary>
    public WheelOdometry(ScoutMiniGeometry geometry, double startX = 0.0, double startY = 0.0, double startYaw = 0.0)
    {
        _geometry = geometry ?? throw new ArgumentNullException(nameof(geometry));
        if (geometry.WheelRadius <= 0) throw new ArgumentOutOfRangeException(nameof(geometry), "wheel radius must be positive");
        if (geometry.TrackWidth <= 0) throw new ArgumentOutOfRangeException(nameof(geometry), "track width must be positive");
        if (geometry.CountsPerRevolution <= 0) throw new ArgumentOutOfRangeException(nameof(geometry), "counts per revolution must be positive");

        X = startX;
        Y = startY;
        Yaw = startYaw;
    }

    /// <summary>Integrated x position in metres, relative to the declared start pose.</summary>
    public double X { get; private set; }

    /// <summary>Integrated y position in metres, relative to the declared start pose.</summary>
    public double Y { get; private set; }

    /// <summary>Integrated heading in radians, wrapped to [-pi, pi).</summary>
    public double Yaw { get; private set; }

    /// <summary>Total path length in metres, useful for the drift-versus-distance measurement.</summary>
    public double TravelledDistance { get; private set; }

    /// <summary>Number of frames ignored because the counter moved backwards by more than a wrap.</summary>
    public long DroppedFrames => _droppedFrames;

    /// <summary>True once at least one frame has established the counter baseline.</summary>
    public bool HasBaseline => _previous.HasValue;

    /// <summary>
    /// Largest wheel increment accepted in one frame. At the documented geometry one count is
    /// about 0.6 mm of wheel travel, so this is ~6 m in a single frame: far beyond any legitimate
    /// reporting period, yet small enough to catch a corrupted payload.
    /// </summary>
    public const int MaxCountsPerFrame = 10_000;

    /// <summary>
    /// Feeds one odometer frame and advances the pose.
    /// The first call only records the baseline; it produces no motion.
    /// </summary>
    /// <returns>Distance travelled by the wheel midpoint during this step, in metres.</returns>
    public double Update(WheelCounters counters)
    {
        if (!_previous.HasValue)
        {
            _previous = counters;
            return 0.0;
        }

        var previous = _previous.Value;
        // The counters are 32-bit and wrap, so the increment must be taken modulo 2^32;
        // a plain subtraction would report a genuine wrap as a half-turn-sized jump.
        var deltaLeft = (int)((uint)counters.Left - (uint)previous.Left);
        var deltaRight = (int)((uint)counters.Right - (uint)previous.Right);
        _previous = counters;

        // A frame is only rejected when the increment is physically impossible for one reporting
        // period. Anything plausible, including a wrap, is integrated.
        if (Math.Abs(deltaLeft) > MaxCountsPerFrame || Math.Abs(deltaRight) > MaxCountsPerFrame)
        {
            _droppedFrames++;
            return 0.0;
        }

        var metresPerCount = 2.0 * Math.PI * _geometry.WheelRadius / _geometry.CountsPerRevolution;
        var leftDistance = deltaLeft * metresPerCount;
        var rightDistance = deltaRight * metresPerCount;

        var distance = (leftDistance + rightDistance) / 2.0;
        var deltaYaw = (rightDistance - leftDistance) / _geometry.TrackWidth;

        // Midpoint integration: use the heading halfway through the step.
        var midYaw = Yaw + deltaYaw / 2.0;
        X += distance * Math.Cos(midYaw);
        Y += distance * Math.Sin(midYaw);
        Yaw = WrapAngle(Yaw + deltaYaw);
        TravelledDistance += Math.Abs(distance);

        return distance;
    }

    /// <summary>Re-seats the integrator at a known pose, e.g. after the operator places the robot on A.</summary>
    public void Reset(double x, double y, double yaw)
    {
        X = x;
        Y = y;
        Yaw = WrapAngle(yaw);
        TravelledDistance = 0.0;
        _previous = null;
        _droppedFrames = 0;
    }

    /// <summary>Wraps an angle to [-pi, pi).</summary>
    public static double WrapAngle(double angle)
    {
        var wrapped = angle % (2.0 * Math.PI);
        if (wrapped > Math.PI) wrapped -= 2.0 * Math.PI;
        else if (wrapped < -Math.PI) wrapped += 2.0 * Math.PI;
        return wrapped;
    }
}
