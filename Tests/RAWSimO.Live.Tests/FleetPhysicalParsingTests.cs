using System;
using System.Linq;
using RAWSimO.Live.Protocol;
using Xunit;

namespace RAWSimO.Live.Tests;

/// <summary>
/// The physical robot arrives as its own entity, never as a replacement pose for a
/// simulated vehicle. Two positions for one vehicle is the duplication the architecture
/// forbids, and a parser that quietly merged them would hide it.
/// </summary>
public class FleetPhysicalParsingTests
{
    private const string FrameWithPhysical = @"
    {""type"":""frame"",""schema"":""1.0"",""step"":50,""t"":50.0,
     ""agvs"":[{""agv_id"":""AGV00"",""pose"":{""agv_id"":""AGV00"",""node"":""W_PARK"",""edge"":null,
      ""progress"":0.0,""source"":""sim"",""timestamp"":50.0},""state"":""idle"",""task_id"":null,""path"":[]}],
     ""task_counts"":{},""n_blocked"":0,
     ""trails"":{""AGV00"":[[125.0,10.0]]},""trail_sources"":{""AGV00"":""sim""},
     ""physical"":{""id"":""scout_mini"",""source"":""gazebo"",""pose"":[12.17,55.01,0.0012],
      ""trail"":[[12.17,55.01],[12.20,55.02]]}}
    ";

    [Fact]
    public void PhysicalEntityIsParsedWithItsOwnTrail()
    {
        Assert.True(LiveMessageParser.TryParse(FrameWithPhysical, out var message, out var error), error);
        var frame = Assert.IsType<FleetFrameMessage>(message);
        var physical = Assert.IsType<FleetPhysical>(frame.Physical);
        Assert.Equal("scout_mini", physical.Id);
        Assert.Equal("gazebo", physical.Source);
        Assert.Equal(12.17, physical.Pose![0], 3);
        Assert.Equal(2, physical.Trail.Count);
    }

    [Fact]
    public void TheSimulatedVehicleKeepsItsOwnPosition()
    {
        // The physical robot must not move AGV00: its pose is still node W_PARK.
        Assert.True(LiveMessageParser.TryParse(FrameWithPhysical, out var message, out _));
        var frame = Assert.IsType<FleetFrameMessage>(message);
        Assert.Single(frame.Agvs);
        Assert.Equal("W_PARK", frame.Agvs[0].Pose.Node);
    }

    [Fact]
    public void TrailSourcesAreParsedPerVehicle()
    {
        Assert.True(LiveMessageParser.TryParse(FrameWithPhysical, out var message, out var error), error);
        var frame = Assert.IsType<FleetFrameMessage>(message);
        Assert.NotNull(frame.TrailSources);
        Assert.Equal("sim", frame.TrailSources!["AGV00"]);
    }

    [Fact]
    public void AFrameWithoutAPhysicalRobotYieldsNull()
    {
        const string bare = @"
        {""type"":""frame"",""schema"":""1.0"",""step"":1,""t"":1.0,
         ""agvs"":[],""task_counts"":{},""n_blocked"":0}
        ";
        Assert.True(LiveMessageParser.TryParse(bare, out var message, out var error), error);
        Assert.Null(Assert.IsType<FleetFrameMessage>(message).Physical);
    }

    [Fact]
    public void AMalformedPhysicalBlockIsRejected()
    {
        const string broken = @"
        {""type"":""frame"",""schema"":""1.0"",""step"":1,""t"":1.0,
         ""agvs"":[],""task_counts"":{},""n_blocked"":0,
         ""physical"":""scout_mini""}
        ";
        Assert.False(LiveMessageParser.TryParse(broken, out _, out var error));
        Assert.Contains("physical", error, StringComparison.OrdinalIgnoreCase);
    }
}
