using RAWSimO.Live.Protocol;
using Xunit;

namespace RAWSimO.Live.Tests;

public class ClockSyncTests
{
    [Fact]
    public void UsesLowestRoundTripSample()
    {
        var sync = new ClockSync(32);
        sync.Add(100.0, 100.05, 100.06, 100.13);
        sync.Add(100.0, 101.0, 101.0, 103.0);
        Assert.Equal(2, sync.Count);
        Assert.Equal(-0.01, sync.OffsetSeconds!.Value, 9);
        Assert.Equal(0.12, sync.RttSeconds!.Value, 9);
    }

    [Fact]
    public void IgnoresInvalidSamples()
    {
        var sync = new ClockSync(4);
        sync.Add(1.0, 2.0, 3.0, 1.5); // negative RTT
        sync.Add(double.NaN, 1, 1, 1);
        Assert.Equal(0, sync.Count);
        Assert.Null(sync.OffsetSeconds);
    }
}
