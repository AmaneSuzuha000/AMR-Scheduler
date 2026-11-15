using System.Text.Json;
using RAWSimO.Live.Geometry;
using RAWSimO.Live.Protocol;
using Xunit;

namespace RAWSimO.Live.Tests;

public class ProtocolTests
{
    [Fact]
    public void ParsesHello()
    {
        var message = LiveMessageParser.Parse("{\"type\":\"hello\",\"vehicle\":\"agv1\",\"proto\":0,\"frame\":\"chassis\"}");
        var hello = Assert.IsType<HelloMessage>(message);
        Assert.Equal("agv1", hello.Vehicle);
        Assert.Equal(0, hello.Proto);
        Assert.Equal("chassis", hello.Frame);
    }

    [Fact]
    public void ParsesAndSerialisesPose()
    {
        var original = new PoseMessage("agv1", 42, 123.5, 1.25, 2.5, -0.3);
        var line = LiveMessageParser.Serialize(original);
        var parsed = Assert.IsType<PoseMessage>(LiveMessageParser.Parse(line));
        Assert.Equal(original, parsed);
        Assert.EndsWith("\n", LiveMessageParser.SerializeLine(original));
    }

    [Fact]
    public void ParsesNavAndRoundTripsPath()
    {
        var original = new NavCommandMessage("s1-g1", new[] { new Point2D(0.375, 0.375), new Point2D(2.625, 2.625) }, 0.2);
        var line = LiveMessageParser.Serialize(original);
        var parsed = Assert.IsType<NavCommandMessage>(LiveMessageParser.Parse(line));
        Assert.Equal("s1-g1", parsed.Goal);
        Assert.Equal(2, parsed.Path.Count);
        Assert.Equal(2.625, parsed.Path[1].X, 12);
        Assert.Equal(0.2, parsed.MaxSpeed);
    }

    [Fact]
    public void UnknownTypeIsPreserved()
    {
        var parsed = Assert.IsType<UnknownMessage>(LiveMessageParser.Parse("{\"type\":\"future\",\"x\":1}"));
        Assert.Equal("future", parsed.Type);
    }

    [Theory]
    [InlineData("")]
    [InlineData("not json")]
    [InlineData("[]")]
    [InlineData("{\"x\":1}")]
    [InlineData("{\"type\":\"pose\",\"vehicle\":\"agv1\",\"seq\":1,\"t\":1,\"x\":0,\"y\":0}")]
    [InlineData("{\"type\":\"pose\",\"vehicle\":\"agv1\",\"seq\":1,\"t\":1,\"x\":\"NaN\",\"y\":0,\"yaw\":0}")]
    [InlineData("{\"type\":\"nav\",\"goal\":\"g\",\"path\":[]}")]
    [InlineData("{\"type\":\"nav\",\"goal\":\"g\",\"path\":[[0,0],[1]]}")]
    public void RejectsInvalidMessages(string line)
    {
        Assert.False(LiveMessageParser.TryParse(line, out _, out var error));
        Assert.False(string.IsNullOrWhiteSpace(error));
    }

    [Fact]
    public void ParsesStatusAndPingPong()
    {
        var status = Assert.IsType<StatusMessage>(LiveMessageParser.Parse("{\"type\":\"status\",\"mode\":\"sim\",\"battery\":null}"));
        Assert.Equal("sim", status.Mode);
        var ping = Assert.IsType<PingMessage>(LiveMessageParser.Parse("{\"type\":\"ping\",\"t1\":12.5}"));
        Assert.Equal(12.5, ping.T1);
        var pong = Assert.IsType<PongMessage>(LiveMessageParser.Parse("{\"type\":\"pong\",\"t1\":1,\"t2\":2,\"t3\":3}"));
        Assert.Equal(3, pong.T3);
    }
}
