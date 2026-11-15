namespace RAWSimO.Live.Protocol;

/// <summary>
/// Estimates the bridge-minus-interface clock offset from ping/pong samples.
/// Each sample is (t1, t2, t3, t4): send, receive at bridge, send at bridge,
/// receive at interface, all in seconds. The lowest-RTT samples are retained,
/// because queueing only makes the observed RTT longer.
/// </summary>
public sealed class ClockSync
{
    private readonly int _capacity;
    private readonly List<Sample> _samples = new();

    /// <summary>Creates a clock synchroniser retaining the best samples.</summary>
    /// <param name="capacity">Maximum number of samples to retain.</param>
    public ClockSync(int capacity = 32)
    {
        if (capacity < 1) throw new ArgumentOutOfRangeException(nameof(capacity));
        _capacity = capacity;
    }

    /// <summary>Number of retained samples.</summary>
    public int Count => _samples.Count;

    /// <summary>Best observed offset (bridge clock minus interface clock), or null before a sample.</summary>
    public double? OffsetSeconds => _samples.Count == 0 ? null : _samples.MinBy(s => s.Rtt).Offset;

    /// <summary>Smallest observed round-trip time, or null before a sample.</summary>
    public double? RttSeconds => _samples.Count == 0 ? null : _samples.Min(s => s.Rtt);

    /// <summary>Adds one sample. Non-finite values or a negative RTT are ignored.</summary>
    public void Add(double t1, double t2, double t3, double t4)
    {
        if (!double.IsFinite(t1) || !double.IsFinite(t2) || !double.IsFinite(t3) || !double.IsFinite(t4)) return;
        var rtt = (t4 - t1) - (t3 - t2);
        if (rtt < 0 || !double.IsFinite(rtt)) return;
        var offset = ((t2 - t1) + (t3 - t4)) / 2.0;
        _samples.Add(new Sample(offset, rtt));
        if (_samples.Count > _capacity)
            _samples.RemoveAt(_samples.FindIndex(s => s.Rtt == _samples.Max(x => x.Rtt)));
    }

    /// <summary>Removes all samples, for example after a reconnect.</summary>
    public void Clear() => _samples.Clear();

    private readonly record struct Sample(double Offset, double Rtt);
}
