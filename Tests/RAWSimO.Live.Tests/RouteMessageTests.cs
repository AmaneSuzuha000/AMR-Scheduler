using System.Text.Json;
using RAWSimO.Live.Geometry;
using RAWSimO.Live.Protocol;
using Xunit;

namespace RAWSimO.Live.Tests;

/// <summary>
/// The console must not plan. It sends a route_request and draws the route_result it gets
/// back, so these tests pin down the shape of both messages: the answer carries scene-frame
/// coordinates that need no calibration, and a refusal carries a reason rather than throwing.
/// </summary>
public class RouteMessageTests
{
    [Fact]
    public void SerialisesRouteRequestWithFromAndDirect()
    {
        var request = new RouteRequestMessage("125", new[] { 8.0, 30.0 }, Direct: true);
        var line = LiveMessageParser.Serialize(request);
        using var document = JsonDocument.Parse(line);
        Assert.Equal("route_request", document.RootElement.GetProperty("type").GetString());
        Assert.Equal("125", document.RootElement.GetProperty("target").GetString());
        Assert.Equal(8.0, document.RootElement.GetProperty("from")[0].GetDouble());
        Assert.True(document.RootElement.GetProperty("direct").GetBoolean());
    }

    [Fact]
    public void ParsesRouteResultWithNestedMapPath()
    {
        const string line = "{\"type\":\"route_result\",\"target\":\"125\",\"start\":\"J_S_8\"," +
                            "\"nodes\":[\"J_S_8\",\"J_S_25\"],\"map_path\":[[8.0,30.0],[25.0,30.0]]," +
                            "\"meters\":17.0,\"seconds\":17.0,\"reason\":null}";
        var result = Assert.IsType<RouteResultMessage>(LiveMessageParser.Parse(line));
        Assert.Equal("125", result.Target);
        Assert.Equal("J_S_8", result.Start);
        Assert.Equal(new[] { "J_S_8", "J_S_25" }, result.Nodes);
        Assert.Equal(2, result.MapPath.Count);
        Assert.Equal(8.0, result.MapPath[0].X, 12);
        Assert.Equal(30.0, result.MapPath[0].Y, 12);
        Assert.Equal(25.0, result.MapPath[1].X, 12);
        Assert.Equal(17.0, result.Meters);
        Assert.Null(result.Reason);
    }

    [Fact]
    public void RefusalParsesAsAnAnswerNotAnError()
    {
        // A refused target is a normal answer: the operator needs to read the reason on screen.
        const string line = "{\"type\":\"route_result\",\"target\":\"NOPE\",\"nodes\":[]," +
                            "\"map_path\":[],\"reason\":\"unknown node 'NOPE'\"}";
        var result = Assert.IsType<RouteResultMessage>(LiveMessageParser.Parse(line));
        Assert.Empty(result.Nodes);
        Assert.Empty(result.MapPath);
        Assert.Equal("unknown node 'NOPE'", result.Reason);
    }

    [Fact]
    public void MalformedMapPathFailsLoudly()
    {
        // Dropping a bad pair silently would draw a route the vehicle never planned.
        const string line = "{\"type\":\"route_result\",\"target\":\"1\",\"nodes\":[]," +
                            "\"map_path\":[[1.0],[2.0,3.0]]}";
        Assert.Throws<FormatException>(() => LiveMessageParser.Parse(line));
    }

    [Fact]
    public void RouteResultRoundTrips()
    {
        var original = new RouteResultMessage("125", new[] { "a", "b" },
            new[] { new Point2D(1.5, 2.5), new Point2D(3.5, 4.5) }, "a", 12.0, 12.0);
        var parsed = Assert.IsType<RouteResultMessage>(LiveMessageParser.Parse(LiveMessageParser.Serialize(original)));
        Assert.Equal(original.Nodes, parsed.Nodes);
        Assert.Equal(original.MapPath, parsed.MapPath);
        Assert.Equal(original.Meters, parsed.Meters);
    }
}
