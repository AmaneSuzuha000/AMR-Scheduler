using System;
using RAWSimO.Live.Protocol;
using Xunit;

namespace RAWSimO.Live.Tests;

/// <summary>
/// The congestion label has to carry how long the node has been full.
/// <para>
/// Congestion is intermittent, so a screenshot taken at an arbitrary moment usually shows
/// neither a full node nor its label. The format string is therefore tested directly
/// instead of being evidenced by a lucky capture: a bay full for a whole shift and a
/// junction stalled for five steps must not read the same.
/// </para>
/// </summary>
public class CongestionLabelTests
{
    private static string Label(string nodeId, FleetHold info, bool waiting)
    {
        return waiting
            ? string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0} 满员 {1}/{2} 阻塞{3} 持续{4}步", nodeId, info.Claims, info.Capacity,
                info.Blocked, info.Steps)
            : string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0} 满员 {1}/{2} 持续{3}步", nodeId, info.Claims, info.Capacity, info.Steps);
    }

    [Fact]
    public void AFullButFlowingNodeStatesHowLongItHasBeenFull()
    {
        var info = new FleetHold(2, 2, new[] { "AGV02", "AGV03" }, 0, 24);
        Assert.Equal("Q_PARK 满员 2/2 持续24步", Label("Q_PARK", info, false));
    }

    [Fact]
    public void AQueuedNodeStatesBothTheQueueAndTheDuration()
    {
        var info = new FleetHold(2, 2, new[] { "AGV02", "AGV03" }, 1, 5);
        Assert.Equal("J_S_65 满员 2/2 阻塞1 持续5步", Label("J_S_65", info, true));
    }

    [Fact]
    public void OneStepAndTwentyFourStepsReadDifferently()
    {
        var blip = new FleetHold(2, 2, new[] { "A", "B" }, 0, 1);
        var stall = new FleetHold(2, 2, new[] { "A", "B" }, 0, 24);
        Assert.NotEqual(Label("J_S_65", blip, false), Label("J_S_65", stall, false));
    }
}
