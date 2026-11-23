using RAWSimO.Live.Protocol;
using RAWSimO.Live.Tracking;
using Xunit;

namespace RAWSimO.Live.Tests;

/// <summary>External poses keep one identity and disappear when truth is missing.</summary>
public class FleetActualPoseTests
{
    [Fact]
    public void CapturedSensorOutageHidesLastKnownPoseAndReturnsWithOneAuthoritativeIdentity()
    {
        var lines = System.IO.File.ReadAllLines(System.IO.Path.Combine(
            System.AppContext.BaseDirectory, "samples", "sensor_outage.jsonl"));
        var hello = Assert.IsType<FleetHelloMessage>(LiveMessageParser.Parse(lines[0]));
        var topology = new FleetTopology(hello);
        var hidden = 0;
        var visible = 0;
        var retainedHistory = 0;
        foreach (var line in lines[1..])
        {
            var frame = Assert.IsType<FleetFrameMessage>(LiveMessageParser.Parse(line));
            var agv = Assert.Single(frame.Agvs);
            Assert.Equal("AGV_01", agv.AgvId);
            Assert.Equal("localized", agv.Pose.Source);
            Assert.Null(frame.Physical);
            var result = topology.Place(frame);
            Assert.True(result.Complete);
            if (agv.Pose.Available == false)
            {
                // Coordinates remain in the wire history; the parser removes
                // them from current placement when availability is false.
                using var raw = System.Text.Json.JsonDocument.Parse(line);
                if (raw.RootElement.GetProperty("agvs")[0].GetProperty("pose")
                    .GetProperty("x").ValueKind == System.Text.Json.JsonValueKind.Number)
                    retainedHistory++;
                Assert.Empty(result.Placements);
                hidden++;
            }
            else
            {
                var placed = Assert.Single(result.Placements);
                Assert.Equal(agv.Pose.X, placed.X);
                Assert.Equal(agv.Pose.Y, placed.Y);
                visible++;
            }
        }
        Assert.Equal(2, hidden);
        Assert.Equal(2, visible);
        Assert.Equal(1, retainedHistory);
    }

    [Fact]
    public void CapturedNativeRosFramesPlaceOneIdentityAndHideLostTruth()
    {
        var lines = System.IO.File.ReadAllLines(System.IO.Path.Combine(
            System.AppContext.BaseDirectory, "samples", "external_execution.jsonl"));
        var hello = Assert.IsType<FleetHelloMessage>(LiveMessageParser.Parse(lines[0]));
        var topology = new FleetTopology(hello);
        var hidden = 0;
        foreach (var line in lines[1..])
        {
            var frame = Assert.IsType<FleetFrameMessage>(LiveMessageParser.Parse(line));
            var agv = Assert.Single(frame.Agvs);
            Assert.Equal("AGV_01", agv.AgvId);
            Assert.Equal("gazebo", agv.Pose.Source);
            Assert.Null(frame.Physical);
            var result = topology.Place(frame);
            Assert.True(result.Complete);
            if (agv.Pose.Available == false)
            {
                Assert.Empty(result.Placements);
                hidden++;
            }
            else
            {
                var placed = Assert.Single(result.Placements);
                Assert.Equal(agv.Pose.X, placed.X);
                Assert.Equal(agv.Pose.Y, placed.Y);
            }
        }
        Assert.Equal(1, hidden);
    }

    private static FleetTopology Topology() => new((FleetHelloMessage)LiveMessageParser.Parse(
        "{\"type\":\"hello\",\"schema\":\"1.0\",\"source\":\"gazebo\",\"network\":{\"nodes\":[],\"edges\":[]},\"meta\":{}}"));

    private static FleetFrameMessage Parse(string pose)
    {
        var line = "{\"type\":\"frame\",\"schema\":\"1.0\",\"step\":10,\"t\":1," +
            "\"agvs\":[{\"agv_id\":\"R\",\"state\":\"blocked\",\"pose\":" + pose + "}]}";
        Assert.True(LiveMessageParser.TryParse(line, out var msg, out var error), error);
        return Assert.IsType<FleetFrameMessage>(msg);
    }

    [Fact]
    public void ActualCoordinatesDoNotNeedAnInventedNodeOrEdge()
    {
        var frame = Parse("{\"agv_id\":\"R\",\"source\":\"gazebo\",\"available\":true," +
            "\"x\":2.34,\"y\":0.02,\"yaw\":0.1,\"frame_id\":\"network\"}");
        var placed = Topology().Place(frame);
        var vehicle = Assert.Single(placed.Placements);
        Assert.Equal(2.34, vehicle.X);
        Assert.Equal(0.02, vehicle.Y);
    }

    [Fact]
    public void StaleTruthIsHiddenEvenIfAPredictedNodeIsPresent()
    {
        var frame = Parse("{\"agv_id\":\"R\",\"available\":false,\"node\":\"A\"}");
        var placed = Topology().Place(frame);
        Assert.Empty(placed.Placements);
    }

    [Theory]
    [InlineData("world")]
    [InlineData("odom")]
    public void ActualCoordinatesInAnotherFrameAreRejected(string frameId)
    {
        var line = "{\"type\":\"frame\",\"schema\":\"1.0\",\"step\":1,\"t\":1," +
            "\"agvs\":[{\"agv_id\":\"R\",\"state\":\"idle\",\"pose\":{\"agv_id\":\"R\"," +
            "\"available\":true,\"x\":1,\"y\":2,\"frame_id\":\"" + frameId + "\"}}]}";
        Assert.False(LiveMessageParser.TryParse(line, out _, out _));
    }
}
