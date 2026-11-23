using System;

namespace RAWSimO.Live.Drivers;

/// <summary>
/// One receive-path abstraction for real chassis data.
///
/// P3-A has to compare USB-CAN against a CAN-to-Ethernet gateway without rewriting the decoding
/// and integration above. Both transports are therefore reduced to "hand me 0x311 payloads",
/// so the comparison is a hardware question, not a code question.
/// </summary>
public interface IScoutMiniCanSource
{
    /// <summary>Human-readable description of the transport, for the log and the evidence table.</summary>
    string Describe();

    /// <summary>Opens the transport. Returns false when the hardware is not present.</summary>
    bool TryOpen(out string error);

    /// <summary>
    /// Reads the next odometer payload. Returns false when the transport has no frame available.
    /// <paramref name="timestamp"/> is the local receive time, used to detect stalls.
    /// </summary>
    bool TryReadOdometry(out WheelCounters counters, out double timestamp);

    /// <summary>Closes the transport.</summary>
    void Close();
}

/// <summary>
/// Serves recorded odometry frames so the whole pipeline can be exercised without hardware.
///
/// This exists because the acceptance criterion for P3-A must be reproducible on a machine with
/// no CAN adapter. It is explicitly a replay source: the log line it produces says so, and no
/// document may present its output as a real-vehicle measurement.
/// </summary>
public sealed class ReplayCanSource : IScoutMiniCanSource
{
    private readonly (WheelCounters Counters, double Timestamp)[] _frames;
    private int _index;

    /// <summary>Wraps a pre-recorded frame list.</summary>
    public ReplayCanSource((WheelCounters Counters, double Timestamp)[] frames)
    {
        _frames = frames ?? throw new ArgumentNullException(nameof(frames));
    }

    /// <inheritdoc />
    public string Describe() => $"replay ({_frames.Length} frames)";

    /// <inheritdoc />
    public bool TryOpen(out string error)
    {
        error = string.Empty;
        _index = 0;
        return _frames.Length > 0;
    }

    /// <inheritdoc />
    public bool TryReadOdometry(out WheelCounters counters, out double timestamp)
    {
        if (_index >= _frames.Length)
        {
            counters = default;
            timestamp = 0.0;
            return false;
        }

        var frame = _frames[_index++];
        counters = frame.Counters;
        timestamp = frame.Timestamp;
        return true;
    }

    /// <inheritdoc />
    public void Close() => _index = 0;
}

/// <summary>
/// Drives a <see cref="WheelOdometry"/> from any <see cref="IScoutMiniCanSource"/>.
/// Keeps the integration loop in one place so USB-CAN and gateway runs behave identically.
/// </summary>
public sealed class ScoutMiniOdometryReader
{
    private readonly IScoutMiniCanSource _source;
    private readonly WheelOdometry _odometry;

    /// <summary>Wires a transport to an integrator.</summary>
    public ScoutMiniOdometryReader(IScoutMiniCanSource source, WheelOdometry odometry)
    {
        _source = source ?? throw new ArgumentNullException(nameof(source));
        _odometry = odometry ?? throw new ArgumentNullException(nameof(odometry));
    }

    /// <summary>The integrator being fed.</summary>
    public WheelOdometry Odometry => _odometry;

    /// <summary>Frames consumed since the reader was created.</summary>
    public long FramesRead { get; private set; }

    /// <summary>Local time of the most recent frame, in seconds.</summary>
    public double LastFrameTimestamp { get; private set; }

    /// <summary>Seconds since the most recent frame, given "now"; used to detect a stalled chassis.</summary>
    public double AgeSeconds(double now) => FramesRead == 0 ? double.PositiveInfinity : now - LastFrameTimestamp;

    /// <summary>Drains every frame currently available and returns how many advanced the pose.</summary>
    public int Pump()
    {
        var advanced = 0;
        while (_source.TryReadOdometry(out var counters, out var timestamp))
        {
            _odometry.Update(counters);
            LastFrameTimestamp = timestamp;
            FramesRead++;
            advanced++;
        }
        return advanced;
    }
}
