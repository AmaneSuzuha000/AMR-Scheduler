using System;
using System.IO;
using System.Linq;
using RAWSimO.Live.Protocol;
using Xunit;

namespace RAWSimO.Live.Tests;

/// <summary>
/// Round-trip tests for the fleet scheduling stream produced by
/// <c>amr_sched.live</c>.
///
/// These matter because the two producers on the wire both speak newline-delimited
/// JSON and both open with a message called "hello". The parser tells them apart by
/// payload shape, so these tests pin that discrimination down: a regression here
/// silently turns every scheduling frame into an UnknownMessage and the GUI shows an
/// empty map with no error.
/// </summary>
public class FleetStreamParsingTests
{
    private static string SamplePath(string name) =>
        Path.Combine(AppContext.BaseDirectory, "samples", name);

    [Fact]
    public void FleetHelloIsRecognisedByItsNetworkPayload()
    {
        var path = SamplePath("fleet_hello.jsonl");
        Assert.True(File.Exists(path), $"missing sample: {path}");

        var line = File.ReadLines(path).First(l => !string.IsNullOrWhiteSpace(l));
        Assert.True(LiveMessageParser.TryParse(line, out var message, out var error), error);

        var hello = Assert.IsType<FleetHelloMessage>(message);
        Assert.Equal("hello", hello.Type);
        Assert.NotEmpty(hello.Nodes);
        Assert.NotEmpty(hello.Edges);

        // The edge length used to arrive as null because the producer iterated the
        // dict keys instead of its values. A non-positive length here would mean
        // that regression came back.
        Assert.All(hello.Edges, e => Assert.True(e.Length > 0, $"edge {e.Src}->{e.Dst} has no length"));
        Assert.All(hello.Nodes, n => Assert.False(string.IsNullOrWhiteSpace(n.Id)));

        // Every edge must reference two announced nodes, otherwise the consumer
        // cannot draw the network.
        var ids = hello.Nodes.Select(n => n.Id).ToHashSet(StringComparer.Ordinal);
        Assert.All(hello.Edges, e =>
        {
            Assert.Contains(e.Src, ids);
            Assert.Contains(e.Dst, ids);
        });
    }

    [Fact]
    public void SingleVehicleHelloIsStillParsedAsHelloMessage()
    {
        // The discrimination must not hijack the single-vehicle bridge.
        const string line = "{\"type\":\"hello\",\"vehicle\":\"agv1\",\"proto\":1,\"frame\":\"odom\"}";
        Assert.True(LiveMessageParser.TryParse(line, out var message, out var error), error);
        var hello = Assert.IsType<HelloMessage>(message);
        Assert.Equal("agv1", hello.Vehicle);
        Assert.Equal("odom", hello.Frame);
    }

    [Fact]
    public void FleetFrameCarriesEveryVehicleWithAUsablePose()
    {
        var path = SamplePath("fleet_frame.jsonl");
        Assert.True(File.Exists(path), $"missing sample: {path}");

        var line = File.ReadLines(path).First(l => !string.IsNullOrWhiteSpace(l));
        Assert.True(LiveMessageParser.TryParse(line, out var message, out var error), error);

        var frame = Assert.IsType<FleetFrameMessage>(message);
        Assert.NotEmpty(frame.Agvs);
        Assert.True(frame.Step > 0);

        foreach (var agv in frame.Agvs)
        {
            Assert.False(string.IsNullOrWhiteSpace(agv.AgvId));
            Assert.False(string.IsNullOrWhiteSpace(agv.State));

            // A pose must say either which node it is on or which edge it is on;
            // never neither, or the GUI has nowhere to draw it.
            Assert.True(agv.Pose.Node is not null || agv.Pose.Edge is not null,
                $"{agv.AgvId} has neither a node nor an edge");

            Assert.InRange(agv.Pose.Progress, 0.0, 1.0);
            Assert.Equal("sim", agv.Pose.Source);
        }
    }

    [Fact]
    public void FrameWithMissingAgvListFailsLoudly()
    {
        // A frame without vehicles is a producer bug. Silently accepting it would
        // let the GUI render an empty map and look healthy.
        const string line = "{\"type\":\"frame\",\"step\":1,\"t\":1.0}";
        Assert.False(LiveMessageParser.TryParse(line, out _, out var error));
        Assert.NotNull(error);
        Assert.Contains("agvs", error);
    }
}
