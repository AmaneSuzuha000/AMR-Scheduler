using System;
using System.Collections.Generic;
using System.Linq;
using RAWSimO.Live.Protocol;
using Xunit;

namespace RAWSimO.Live.Tests;

/// <summary>
/// The console must be able to draw where a vehicle actually was, not only where it is.
/// <para>
/// Absent is not the same as broken: a producer that publishes no trails yields null and the
/// console draws nothing, while a malformed trails field is rejected so a typo cannot
/// silently turn into "this vehicle never moved".
/// </para>
/// </summary>
public class FleetTrailParsingTests
{
    private const string FrameWithoutTrails = @"
    {""type"":""frame"",""schema"":""1.0"",""step"":10,""t"":10.0,
     ""agvs"":[{""agv_id"":""AGV00"",""pose"":{""agv_id"":""AGV00"",""node"":""W1"",""edge"":null,
      ""progress"":0.0,""source"":""sim"",""timestamp"":10.0},""state"":""idle"",""task_id"":null,""path"":[]}],
     ""task_counts"":{},""n_blocked"":0}
    ";

    [Fact]
    public void TrailsAreParsedInOrder()
    {
        const string json = @"
        {""type"":""frame"",""schema"":""1.0"",""step"":10,""t"":10.0,
         ""agvs"":[{""agv_id"":""AGV00"",""pose"":{""agv_id"":""AGV00"",""node"":""W1"",""edge"":null,
          ""progress"":0.0,""source"":""sim"",""timestamp"":10.0},""state"":""idle"",""task_id"":null,""path"":[]}],
         ""task_counts"":{},""n_blocked"":0,
         ""trails"":{""AGV00"":[[1.0,2.0],[3.5,4.25]]}}
        ";
        Assert.True(LiveMessageParser.TryParse(json, out var message, out var error), error);
        var frame = Assert.IsType<FleetFrameMessage>(message);
        var trail = frame.Trails!["AGV00"];
        Assert.Equal(2, trail.Count);
        Assert.Equal(1.0, trail[0].X);
        Assert.Equal(4.25, trail[1].Y);
    }

    [Fact]
    public void AFrameWithoutTrailsYieldsNull()
    {
        Assert.True(LiveMessageParser.TryParse(FrameWithoutTrails, out var message, out var error), error);
        Assert.Null(Assert.IsType<FleetFrameMessage>(message).Trails);
    }

    [Fact]
    public void AMalformedTrailIsRejectedRatherThanDroppedInSilence()
    {
        const string json = @"
        {""type"":""frame"",""schema"":""1.0"",""step"":10,""t"":10.0,
         ""agvs"":[{""agv_id"":""AGV00"",""pose"":{""agv_id"":""AGV00"",""node"":""W1"",""edge"":null,
          ""progress"":0.0,""source"":""sim"",""timestamp"":10.0},""state"":""idle"",""task_id"":null,""path"":[]}],
         ""task_counts"":{},""n_blocked"":0,
         ""trails"":{""AGV00"":[[""x"",""y""]]}}
        ";
        Assert.False(LiveMessageParser.TryParse(json, out _, out var error));
        Assert.Contains("trails", error, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ATrailPointMustBeAPair()
    {
        const string json = @"
        {""type"":""frame"",""schema"":""1.0"",""step"":10,""t"":10.0,
         ""agvs"":[{""agv_id"":""AGV00"",""pose"":{""agv_id"":""AGV00"",""node"":""W1"",""edge"":null,
          ""progress"":0.0,""source"":""sim"",""timestamp"":10.0},""state"":""idle"",""task_id"":null,""path"":[]}],
         ""task_counts"":{},""n_blocked"":0,
         ""trails"":{""AGV00"":[[1.0]]}}
        ";
        Assert.False(LiveMessageParser.TryParse(json, out _, out _));
    }
}
