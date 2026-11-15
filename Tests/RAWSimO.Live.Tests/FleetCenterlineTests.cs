using System.Text.Json.Nodes;
using RAWSimO.Live.Geometry;
using RAWSimO.Live.Protocol;
using RAWSimO.Live.Tracking;
using Xunit;

namespace RAWSimO.Live.Tests;

public class FleetCenterlineTests
{
    private static readonly Point2D[] Bend = {
        new(65,80), new(66.6,80), new(66.6,63), new(65,63) };

    private static FleetHelloMessage Hello() => new("1.0", "sim", new[] {
        new FleetNode("W", "station", 65, 80), new FleetNode("R", "junction", 65, 63) },
        new[] { new FleetEdge("W", "R", 20.2, Centerline: Bend) },
        new FleetMeta(1,1,.1,new[] { "AGV" }));

    private static string Wire(string geometry = "[[65,80],[66.6,80],[66.6,63],[65,63]]") =>
        "{\"type\":\"hello\",\"network\":{\"nodes\":[" +
        "{\"id\":\"W\",\"kind\":\"station\",\"x\":65,\"y\":80}," +
        "{\"id\":\"R\",\"kind\":\"junction\",\"x\":65,\"y\":63}]," +
        "\"edges\":[{\"src\":\"W\",\"dst\":\"R\",\"length\":20.2,\"centerline\":" + geometry + "}]},\"meta\":{}}";

    [Theory]
    [InlineData(0,65,80)]
    [InlineData(-.5,65,80)]
    [InlineData(.5,66.6,71.5)]
    [InlineData(1,65,63)]
    [InlineData(1.5,65,63)]
    public void PlacementUsesRoadArcLength(double progress, double x, double y)
    {
        var topology = new FleetTopology(Hello());
        var frame = new FleetFrameMessage("1.0",1,.1,new[] {
            new FleetAgv("AGV",new FleetPose("AGV",Edge:"W->R",Progress:progress),"to_dropoff") });
        var result = topology.Place(frame);
        Assert.True(result.Complete, result.Detail);
        var placed = Assert.Single(result.Placements);
        Assert.Equal(x,placed.X,8); Assert.Equal(y,placed.Y,8);
        Assert.Equal(Bend,topology.RoutePoints(new[] { "W","R" }));
    }

    [Theory]
    [InlineData("null")]
    [InlineData("[]")]
    [InlineData("[[65,80]]")]
    [InlineData("[[65,80],[65,63]]")]
    [InlineData("[[65,80],[65,80],[66.6,80],[66.6,63],[65,63]]")]
    [InlineData("[[65,63],[66.6,63],[66.6,80],[65,80]]")]
    [InlineData("[[65,80],[66.6,80,1],[66.6,63],[65,63]]")]
    [InlineData("[[65,80],[1e400,80],[65,63]]")]
    public void MalformedOrContradictoryGeometryIsRejected(string geometry) =>
        Assert.False(LiveMessageParser.TryParse(Wire(geometry),out _,out _));

    [Fact]
    public void ParserRetainsCornersAndLegacyHelloRemainsReadable()
    {
        var parsed = Assert.IsType<FleetHelloMessage>(LiveMessageParser.Parse(Wire()));
        Assert.Equal(Bend, Assert.Single(parsed.Edges).Centerline);
        var legacy = JsonNode.Parse(Wire())!;
        legacy["network"]!["edges"]![0]!.AsObject().Remove("centerline");
        parsed = Assert.IsType<FleetHelloMessage>(LiveMessageParser.Parse(legacy.ToJsonString()));
        Assert.Equal(new[] { Bend[0],Bend[^1] },new FleetTopology(parsed).EdgePoints("W","R"));
        var duplicate = JsonNode.Parse(Wire())!;
        duplicate["network"]!["nodes"]!.AsArray().Add(JsonNode.Parse(duplicate["network"]!["nodes"]![0]!.ToJsonString()));
        Assert.False(LiveMessageParser.TryParse(duplicate.ToJsonString(),out _,out _));
    }

    [Fact]
    public void UndeclaredReverseAndMissingIntermediateRoadAreNotDrawn()
    {
        var topology = new FleetTopology(Hello());
        Assert.Null(topology.RoutePoints(new[] { "R","W" }));
        Assert.Null(topology.RoutePoints(new[] { "W","missing","R" }));
        var result = topology.Place(new FleetFrameMessage("1.0",1,.1,new[] {
            new FleetAgv("AGV",new FleetPose("AGV",Edge:"R->W",Progress:.5),"to_dropoff") }));
        Assert.Equal(FleetPlacementFailure.UnknownTopology,result.Failure);
    }

    [Fact]
    public void ActualFeedbackAndItsAbsenceOverrideRoadInterpolation()
    {
        var topology = new FleetTopology(Hello());
        var actual = new FleetPose("AGV",Edge:"W->R",Progress:.5,Source:"localized",
            X:66.57,Y:71.6,Available:true,FrameId:"network");
        FleetFrameMessage Frame(FleetPose p) => new("1.0",1,.1,new[] { new FleetAgv("AGV",p,"moving") });
        var p = Assert.Single(topology.Place(Frame(actual)).Placements);
        Assert.Equal(66.57,p.X); Assert.Equal(71.6,p.Y);
        Assert.Empty(topology.Place(Frame(actual with { Available = false })).Placements);
    }

    [Fact]
    public void SceneFitContainsCornersBeyondNodeBounds()
    {
        var hello = Hello();
        var scene = FleetScene.FitNetwork(hello,800,600)!;
        foreach (var point in Bend)
        {
            var pixel = scene.Project(point.X,point.Y);
            Assert.InRange(pixel.X,24,776); Assert.InRange(pixel.Y,24,576);
        }
    }

    [Fact]
    public void PythonCanonicalHelloCarriesAll144RoadsAndTwelveBends()
    {
        var line = File.ReadAllText(Path.Combine(AppContext.BaseDirectory,"samples","workshop_centerlines.jsonl"));
        var hello = Assert.IsType<FleetHelloMessage>(LiveMessageParser.Parse(line));
        Assert.Equal(144,hello.Edges.Count);
        Assert.Equal(12,hello.Edges.Count(e => e.Centerline!.Count > 2));
        var topology = new FleetTopology(hello);
        Assert.Equal(Bend,topology.RoutePoints(new[] { "W_WHEELSET","R_WHEELSET" }));
    }
}
