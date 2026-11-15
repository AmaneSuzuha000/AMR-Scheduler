using System;
using System.Linq;
using RAWSimO.Live.Protocol;
using Xunit;

namespace RAWSimO.Live.Tests;

/// <summary>
/// The frame now carries who holds which node (Python's reservation table) and which
/// nodes are at capacity. The display layer needs both to show junction reservations and
/// congestion; a vehicle's position alone cannot tell a vehicle parked on a junction from
/// one that has merely booked it.
/// </summary>
public class FleetHoldParsingTests
{
    private const string FrameWithHolds = """
    {"type":"frame","schema":"1.0","step":40,"t":40.0,
     "agvs":[{"agv_id":"AGV00","pose":{"agv_id":"AGV00","node":"W_PARK","edge":null,
      "progress":0.0,"source":"sim","timestamp":40.0},"state":"idle","task_id":null,"path":[]}],
     "task_counts":{"pending":3},
     "n_blocked":0,
     "holds":{"Q_PARK":["AGV02","AGV03"],"W_PARK":["AGV00"]},
     "congestion":{"Q_PARK":{"claims":2,"capacity":2,"agvs":["AGV02","AGV03"],
       "blocked":0,"steps":147}}}
    """;

    [Fact]
    public void HoldsAreParsedSoReservationsCanBeShown()
    {
        Assert.True(LiveMessageParser.TryParse(FrameWithHolds, out var message, out var error), error);
        var frame = Assert.IsType<FleetFrameMessage>(message);
        Assert.NotNull(frame.Holds);
        Assert.Equal(new[] { "AGV02", "AGV03" }, frame.Holds!["Q_PARK"]);
        Assert.Equal(new[] { "AGV00" }, frame.Holds!["W_PARK"]);
    }

    [Fact]
    public void CongestionIsParsedWithTheNodesOwnCapacity()
    {
        Assert.True(LiveMessageParser.TryParse(FrameWithHolds, out var message, out var error), error);
        var frame = Assert.IsType<FleetFrameMessage>(message);
        Assert.NotNull(frame.Congestion);
        var q = frame.Congestion!["Q_PARK"];
        Assert.Equal(2, q.Claims);
        // Capacity comes from the network, not from the claim count: a capacity-2 node with
        // two claims is full, and reporting it needs both numbers.
        Assert.Equal(2, q.Capacity);
        Assert.Equal(new[] { "AGV02", "AGV03" }, q.Agvs);
        // How long it has been full. Without it a bay full for a whole shift and a junction
        // stalled for three steps look the same on the map.
        Assert.Equal(147, q.Steps);
        // Parked vehicles fill a bay without anyone waiting; that must not read as a queue.
        Assert.Equal(0, q.Blocked);
    }

    [Fact]
    public void AFrameWithoutHoldsStillParses()
    {
        // Older producers do not send these fields. Absent must mean "not reported", not an
        // error, otherwise a version skew blanks the whole stream.
        const string bare = """
        {"type":"frame","schema":"1.0","step":1,"t":1.0,
         "agvs":[{"agv_id":"A","pose":{"agv_id":"A","node":"N","edge":null,
          "progress":0.0,"source":"sim","timestamp":1.0},"state":"idle","task_id":null,"path":[]}]}
        """;
        Assert.True(LiveMessageParser.TryParse(bare, out var message, out var error), error);
        var frame = Assert.IsType<FleetFrameMessage>(message);
        Assert.Null(frame.Holds);
        Assert.Null(frame.Congestion);
    }

    [Fact]
    public void AMalformedHoldsFieldIsRejectedRatherThanSilentlyDropped()
    {
        const string bad = """
        {"type":"frame","schema":"1.0","step":1,"t":1.0,"agvs":[],
         "holds":{"N":"AGV00"}}
        """;
        Assert.False(LiveMessageParser.TryParse(bad, out _, out var error));
        Assert.Contains("holds", error ?? string.Empty, StringComparison.OrdinalIgnoreCase);
    }
}
