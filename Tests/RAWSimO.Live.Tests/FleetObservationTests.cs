using RAWSimO.Live.Protocol;
using Xunit;

namespace RAWSimO.Live.Tests;

public class FleetObservationTests
{
    private const string Frame = "{\"type\":\"frame\",\"schema\":\"1.0\",\"step\":1,\"t\":1,\"agvs\":[]";
    private const string Control = ",\"control\":{\"session_id\":\"s\",\"scene_id\":\"scene\",\"run_id\":\"r\",\"paused\":false,\"routes\":{\"R1\":\"r:R1:1\"},\"faults\":";

    [Fact]
    public void FaultReasonsAndPhysicalRoadOccupancyPreserveOwnerEvidence()
    {
        var frame = Assert.IsType<FleetFrameMessage>(LiveMessageParser.Parse(Frame + Control +
            "{\"R1\":\"feedback timeout\"}},\"physical_holds\":{\"nodes\":{\"W\":[\"R1\"]},\"edges\":{\"W->X\":[\"R1\"]}}}"));
        Assert.Equal("feedback timeout", frame.Control!.Faults!["R1"]);
        Assert.Equal(new[] { "R1" }, frame.PhysicalHolds!.Nodes["W"]);
        Assert.Equal(new[] { "R1" }, frame.PhysicalHolds.Edges["W->X"]);
    }

    [Fact]
    public void LegacyAbsenceIsDistinctFromReportedNoFaultAndEmptyOccupancy()
    {
        var legacy = Assert.IsType<FleetFrameMessage>(LiveMessageParser.Parse(Frame + "}"));
        Assert.Null(legacy.Control);
        Assert.Null(legacy.PhysicalHolds);
        var current = Assert.IsType<FleetFrameMessage>(LiveMessageParser.Parse(Frame + Control +
            "{\"R1\":null}},\"physical_holds\":{\"nodes\":{},\"edges\":{}}}"));
        Assert.True(current.Control!.Faults!.ContainsKey("R1"));
        Assert.Null(current.Control.Faults["R1"]);
        Assert.Empty(current.PhysicalHolds!.Edges);
    }

    private const string Observed = "{\"type\":\"frame\",\"schema\":\"1.0\",\"step\":1,\"t\":1,\"agvs\":[{\"agv_id\":\"R1\",\"state\":\"blocked\",\"pose\":{\"agv_id\":\"R1\",\"source\":\"localized\",\"available\":false}}]";

    [Theory]
    [InlineData("\"execution\":{\"robot_id\":\"R1\",\"scene_id\":\"scene\",\"run_id\":\"r\",\"fault\":\"execution guard: invalid command\"}")]
    [InlineData("\"executions\":{\"R1\":{\"robot_id\":\"R1\",\"scene_id\":\"scene\",\"run_id\":\"r\",\"fault\":\"execution guard: invalid command\"}}")]
    public void EndpointClosureDoesNotDiscardObservedExecutionFaults(string executions)
    {
        var frame = Assert.IsType<FleetFrameMessage>(LiveMessageParser.Parse(Observed + "," + executions + "}"));
        Assert.Null(frame.Control);
        Assert.Equal("execution guard: invalid command", frame.ExecutionFaults!["R1"]);
        using var replay = RAWSimO.Live.Tracking.FleetReplay.Load(new StringReader(
            "{\"type\":\"hello\",\"network\":{\"nodes\":[],\"edges\":[]},\"meta\":{}}\n" + Observed + "," + executions + "}"));
        Assert.Equal(frame.ExecutionFaults["R1"], replay.At(0).Frame.ExecutionFaults!["R1"]);
    }

    [Theory]
    [InlineData("{\"R1\":null}")]
    [InlineData("[]")]
    [InlineData("{\"R1\":{\"robot_id\":\"R2\",\"scene_id\":\"scene\",\"run_id\":\"r\",\"fault\":null}}")]
    [InlineData("{\"R2\":{\"robot_id\":\"R2\",\"scene_id\":\"scene\",\"run_id\":\"r\",\"fault\":null}}")]
    [InlineData("{\"R1\":{\"robot_id\":\"R1\",\"scene_id\":\"scene\",\"run_id\":\"r\",\"fault\":false}}")]
    [InlineData("{\"R1\":{\"robot_id\":\"R1\",\"scene_id\":\"scene\",\"run_id\":\"r\"}}")]
    [InlineData("{\"R1\":{\"robot_id\":\"R1\",\"scene_id\":\"scene\",\"run_id\":\"r\",\"fault\":null},\"R1\":{\"robot_id\":\"R1\",\"scene_id\":\"scene\",\"run_id\":\"r\",\"fault\":\"failure\"}}")]
    public void AmbiguousExecutionFaultCannotBeDisplayedAsHealthy(string states)
    {
        Assert.False(LiveMessageParser.TryParse(Observed + ",\"executions\":" + states + "}", out _, out _));
    }

    [Theory]
    [InlineData("[]")]
    [InlineData("{\"R1\":false}")]
    [InlineData("{\"unknown_robot\":\"failure\"}")]
    [InlineData("{\"R1\":null,\"R1\":\"failure\"}")]
    public void MalformedOrAmbiguousFaultStateCannotSilentlyBecomeHealthy(string faults)
    {
        Assert.False(LiveMessageParser.TryParse(Frame + Control + faults + "}}", out _, out _));
    }

    [Theory]
    [InlineData("{\"nodes\":{},\"edges\":[]}")]
    [InlineData("{\"nodes\":{},\"edges\":{\"W->X\":[false]}}")]
    [InlineData("{\"nodes\":{}}")]
    public void MalformedPhysicalOccupancyCannotSilentlyLookReleased(string pins)
    {
        Assert.False(LiveMessageParser.TryParse(Frame + ",\"physical_holds\":" + pins + "}", out _, out _));
    }
}
