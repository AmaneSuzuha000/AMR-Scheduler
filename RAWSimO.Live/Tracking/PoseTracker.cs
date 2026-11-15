using System.Collections.Concurrent;
using System.Collections.Generic;
using RAWSimO.Live.Geometry;
using RAWSimO.Live.Protocol;

namespace RAWSimO.Live.Tracking;

/// <summary>Freshness of the most recent pose, derived only from the local monotonic clock.</summary>
public enum PoseFreshness
{
    /// <summary>Newer than the stale threshold.</summary>
    Fresh,
    /// <summary>Older than the stale threshold but newer than the lost threshold.</summary>
    Stale,
    /// <summary>No pose has arrived within the lost threshold, or no pose exists.</summary>
    Lost,
}

/// <summary>An immutable pose snapshot with receive-time metadata.</summary>
public sealed record LivePose(
    long Seq,
    double T,
    double X,
    double Y,
    double Yaw,
    double ReceivedMonotonic,
    DateTimeOffset ReceivedWall,
    bool OutsideMap)
{
    /// <summary>Map-frame position.</summary>
    public Point2D Position => new(X, Y);

    /// <summary>Map-frame pose.</summary>
    public Pose2D Pose => new(X, Y, Yaw);
}

/// <summary>Result of one <see cref="PoseTracker.Update"/> call.</summary>
public sealed record PoseUpdateResult(
    bool Accepted,
    bool Discarded,
    bool NewSession,
    bool SequenceGap,
    bool OutsideMap,
    string? Reason = null);

/// <summary>
/// Validates, orders and timestamps incoming pose messages. The class is
/// thread-safe for one writer and any number of readers of <see cref="Latest"/>.
/// </summary>
public sealed class PoseTracker
{
    private readonly ConcurrentQueue<LivePose> _trail = new();

    /// <summary>Creates a tracker for a map of the given size.</summary>
    public PoseTracker(double mapWidthMeters = 3.0, double mapHeightMeters = 3.0)
    {
        ConfigureMap(mapWidthMeters, mapHeightMeters);
    }

    /// <summary>Map width in metres; used for bounds checks.</summary>
    public double MapWidthMeters { get; private set; } = 3.0;

    /// <summary>Map height in metres; used for bounds checks.</summary>
    public double MapHeightMeters { get; private set; } = 3.0;

    /// <summary>
    /// Errors after which the pose on the wire must not be trusted as a live position.
    /// <para>
    /// <c>odom_stale</c> is the odd one out. The watchdog and the e-stop mean the bridge was
    /// told to hold the vehicle, so a still pose is the truth; <c>odom_stale</c> means the
    /// bridge stopped receiving poses from its own source and keeps repeating the last one.
    /// The repeated value is a well-formed message that arrives on time, so age alone can
    /// never detect it -- only the status line can.
    /// </para>
    /// </summary>
    public static readonly IReadOnlyList<string> UntrustworthyErrors = new[] { "odom_stale" };
    /// <summary>Age after which a pose is considered stale (default 0.15 s).</summary>
    public double StaleAfterSeconds { get; set; } = 0.15;

    /// <summary>Age after which the connection is considered lost (default 2 s).</summary>
    public double LostAfterSeconds { get; set; } = 2.0;

    /// <summary>Latest accepted pose, or null before the first valid pose.</summary>
    public LivePose? Latest { get; private set; }

    /// <summary>
    /// Error the source last reported about itself, or null when it reported none.
    /// Set from the bridge's periodic status message, not from the pose stream.
    /// </summary>
    public string? SourceError { get; private set; }

    /// <summary>True when the source says its own pose stream is no longer trustworthy.</summary>
    public bool SourceStale => SourceError != null && UntrustworthyErrors.Contains(SourceError);

    /// <summary>
    /// Records the error the source reported about itself.
    /// <para>
    /// Kept separate from <see cref="Update"/> because the two arrive on different messages:
    /// the pose says "the vehicle is here", the status says "and I still know that". A frozen
    /// pose repeated at 20 Hz satisfies the first and violates the second.
    /// </para>
    /// </summary>
    /// <param name="error">Error code from the status message, or null when there is none.</param>
    public void SetSourceError(string? error) => SourceError = string.IsNullOrWhiteSpace(error) ? null : error;

    /// <summary>Last accepted sequence number.</summary>
    public long? LastSequence { get; private set; }

    /// <summary>Number of skipped sequence numbers since the last session restart.</summary>
    public long GapCount { get; private set; }

    /// <summary>Number of discarded malformed or out-of-range poses.</summary>
    public long InvalidCount { get; private set; }

    /// <summary>Session counter; incremented when seq moves backwards.</summary>
    public int SessionCount { get; private set; } = 1;

    /// <summary>Maximum number of trail points retained.</summary>
    public int MaxTrailPoints { get; set; } = 3000;

    /// <summary>Raised when one or more sequence numbers are missing.</summary>
    public Action<long>? SequenceGap { get; set; }

    /// <summary>Raised when a new session starts (sequence number went backwards).</summary>
    public Action? SessionRestarted { get; set; }

    /// <summary>Raised after a valid pose is stored.</summary>
    public Action<LivePose>? PoseUpdated { get; set; }

    /// <summary>Updates the map dimensions, for example after loading another instance.</summary>
    public void ConfigureMap(double widthMeters, double heightMeters)
    {
        if (!double.IsFinite(widthMeters) || widthMeters <= 0) throw new ArgumentOutOfRangeException(nameof(widthMeters));
        if (!double.IsFinite(heightMeters) || heightMeters <= 0) throw new ArgumentOutOfRangeException(nameof(heightMeters));
        MapWidthMeters = widthMeters;
        MapHeightMeters = heightMeters;
    }

    /// <summary>Validates and stores one pose message.</summary>
    public PoseUpdateResult Update(PoseMessage message, double receivedMonotonic, DateTimeOffset receivedWall)
    {
        ArgumentNullException.ThrowIfNull(message);

        if (!double.IsFinite(message.X) || !double.IsFinite(message.Y) ||
            !double.IsFinite(message.Yaw) || !double.IsFinite(message.T) ||
            !double.IsFinite(receivedMonotonic))
        {
            InvalidCount++;
            return new PoseUpdateResult(false, true, false, false, false, "non-finite pose");
        }

        // Positions more than ten map widths away are almost certainly a
        // unit/frame error rather than a vehicle that drove off the map.
        var limitX = Math.Max(MapWidthMeters, 1.0) * 10.0;
        var limitY = Math.Max(MapHeightMeters, 1.0) * 10.0;
        if (Math.Abs(message.X) > limitX || Math.Abs(message.Y) > limitY)
        {
            InvalidCount++;
            return new PoseUpdateResult(false, true, false, false, false, "pose far outside the map");
        }

        var newSession = false;
        var sequenceGap = false;
        if (LastSequence is { } previous)
        {
            if (message.Seq <= previous)
            {
                newSession = true;
                GapCount = 0;
                SessionCount++;
                ClearTrail();
                SessionRestarted?.Invoke();
            }
            else if (message.Seq > previous + 1)
            {
                var missing = message.Seq - previous - 1;
                GapCount += missing;
                sequenceGap = true;
                SequenceGap?.Invoke(missing);
            }
        }

        var yaw = AngleMath.Normalize(message.Yaw);
        var outside = message.X < 0.0 || message.X > MapWidthMeters ||
                      message.Y < 0.0 || message.Y > MapHeightMeters;
        var pose = new LivePose(
            message.Seq,
            message.T,
            message.X,
            message.Y,
            yaw,
            receivedMonotonic,
            receivedWall,
            outside);

        Latest = pose;
        LastSequence = message.Seq;
        AppendTrail(pose);
        PoseUpdated?.Invoke(pose);
        return new PoseUpdateResult(true, false, newSession, sequenceGap, outside);
    }

    /// <summary>Returns freshness at the given monotonic timestamp.</summary>
    public PoseFreshness GetFreshness(double nowMonotonic)
    {
        if (Latest is null || !double.IsFinite(nowMonotonic)) return PoseFreshness.Lost;
        // A pose the source disowned is not fresh no matter how recently it arrived. This is
        // the case age cannot see: the bridge keeps resending one frozen value at full rate.
        if (SourceStale) return PoseFreshness.Lost;
        var age = Math.Max(0.0, nowMonotonic - Latest.ReceivedMonotonic);
        if (age < StaleAfterSeconds) return PoseFreshness.Fresh;
        if (age < LostAfterSeconds) return PoseFreshness.Stale;
        return PoseFreshness.Lost;
    }

    /// <summary>Age of the latest pose in seconds, or positive infinity when none exists.</summary>
    public double AgeSeconds(double nowMonotonic) =>
        Latest is null ? double.PositiveInfinity : Math.Max(0.0, nowMonotonic - Latest.ReceivedMonotonic);

    /// <summary>Returns a snapshot of the recent trail in arrival order.</summary>
    public IReadOnlyList<LivePose> SnapshotTrail() => _trail.ToArray();

    /// <summary>Clears the latest pose, sequence state and trail.</summary>
    public void Reset()
    {
        Latest = null;
        LastSequence = null;
        GapCount = 0;
        SourceError = null;
        ClearTrail();
    }

    private void AppendTrail(LivePose pose)
    {
        if (_trail.TryPeek(out var last) && Point2D.Distance(last.Position, pose.Position) < 0.01)
            return;
        _trail.Enqueue(pose);
        while (_trail.Count > MaxTrailPoints)
            _trail.TryDequeue(out _);
    }

    private void ClearTrail()
    {
        while (_trail.TryDequeue(out _)) { }
    }
}
