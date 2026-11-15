using System.Text.Json;
using RAWSimO.Live.Protocol;
using RAWSimO.Live.Tracking;
using Xunit;

namespace RAWSimO.Live.Tests;

public class FleetControlReplayTests
{
    private const string Control = "{\"session_id\":\"s\",\"scene_id\":\"scene\",\"run_id\":\"run\",\"paused\":true," +
        "\"operations\":[\"pause\",\"resume\",\"submit_task\"],\"routes\":{\"R1\":\"run:R1:1\"}}";
    private static string Hello => "{\"type\":\"hello\",\"schema\":\"1.0\",\"network\":{\"nodes\":[],\"edges\":[]},\"meta\":{},\"control\":" + Control + "}";
    private static string Frame(double t, bool available = true) => "{\"type\":\"frame\",\"schema\":\"1.0\",\"step\":1,\"t\":" +
        t.ToString(System.Globalization.CultureInfo.InvariantCulture) + ",\"agvs\":[{\"agv_id\":\"R1\",\"state\":\"blocked\",\"pose\":{" +
        "\"agv_id\":\"R1\",\"source\":\"localized\",\"available\":" + (available ? "true" : "false") +
        ",\"x\":0,\"y\":0,\"yaw\":0,\"frame_id\":\"network\"}}],\"control\":" + Control + "}";

    [Fact]
    public void RequestPreservesWireFieldNamesAndUppercaseRobotIds()
    {
        var request = new FleetControlRequestMessage("s", "scene", "run", "id", "resume",
            JsonSerializer.SerializeToElement(new { routes = new Dictionary<string, string> { ["R1"] = "run:R1:1" } }));
        using var json = JsonDocument.Parse(LiveMessageParser.Serialize(request));
        Assert.Equal("control_request", json.RootElement.GetProperty("type").GetString());
        Assert.Equal("s", json.RootElement.GetProperty("session_id").GetString());
        Assert.Equal("id", json.RootElement.GetProperty("request_id").GetString());
        Assert.Equal("run:R1:1", json.RootElement.GetProperty("payload").GetProperty("routes").GetProperty("R1").GetString());
    }

    [Fact]
    public void CapabilitiesAreOptionalOnLegacyStreamsAndTypedOnNewStreams()
    {
        var hello = Assert.IsType<FleetHelloMessage>(LiveMessageParser.Parse(Hello));
        Assert.True(hello.Control!.Paused);
        Assert.Equal("run:R1:1", hello.Control.Routes["R1"]);
        Assert.Null(Assert.IsType<FleetHelloMessage>(LiveMessageParser.Parse(Hello.Replace(",\"control\":" + Control, ""))).Control);
        var frame = Assert.IsType<FleetFrameMessage>(LiveMessageParser.Parse(Frame(0)));
        Assert.Equal(hello.Control.RunId, frame.Control!.RunId);
        var receipt = Assert.IsType<FleetControlResultMessage>(LiveMessageParser.Parse(
            "{\"type\":\"control_result\",\"request_id\":\"id\",\"operation\":\"pause\",\"accepted\":false,\"reason\":\"stale\"}"));
        Assert.False(receipt.Accepted);
        Assert.Equal("stale", receipt.Reason);
    }

    [Theory]
    [InlineData("\"paused\":true", "\"paused\":\"true\"")]
    [InlineData("\"run_id\":\"run\"", "\"run_id\":null")]
    [InlineData("\"R1\":\"run:R1:1\"", "\"R1\":123")]
    public void MalformedControlCannotEnableAnOperatorAction(string before, string after)
    {
        Assert.False(LiveMessageParser.TryParse(Hello.Replace(before, after), out _, out _));
    }

    [Fact]
    public void ReplayUsesReceiptTimeAndKeepsUnavailablePoseHiddenBetweenObservations()
    {
        var lines = new[]
        {
            "{\"kind\":\"operator_endpoint\",\"hello\":" + Hello + "}",
            "{\"kind\":\"frame\",\"wall\":100,\"value\":" + Frame(0) + "}",
            "{\"kind\":\"operator_request\",\"result\":{\"type\":\"control_request\",\"operation\":\"resume\"}}",
            "{\"kind\":\"frame\",\"wall\":105,\"value\":" + Frame(0, false) + "}",
            "{\"kind\":\"frame\",\"wall\":107,\"value\":" + Frame(1) + "}",
        };
        var replay = FleetReplay.Load(new StringReader(string.Join("\n", lines)));
        Assert.Equal(7, replay.Duration);
        Assert.Equal(3, replay.Frames.Count);
        Assert.False(replay.At(6).Frame.Agvs[0].Pose.Available);
        Assert.True(replay.At(7).Frame.Agvs[0].Pose.Available);
        Assert.Empty(new FleetTopology(replay.Hello).Place(replay.At(6).Frame).Placements);
    }

    [Theory]
    [InlineData("missing_hello")]
    [InlineData("second_hello")]
    [InlineData("time_reversed")]
    [InlineData("mixed_run")]
    public void ReplayRejectsRecordsThatCouldMisrepresentARealRun(string problem)
    {
        var content = problem switch
        {
            "missing_hello" => Frame(0),
            "second_hello" => Hello + "\n" + Frame(0) + "\n" + Hello,
            "time_reversed" => Hello + "\n" + Frame(1) + "\n" + Frame(0),
            _ => Hello + "\n" + Frame(0).Replace("\"run_id\":\"run\"", "\"run_id\":\"other\""),
        };
        Assert.Throws<FormatException>(() => FleetReplay.Load(new StringReader(content)));
    }
}
