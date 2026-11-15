using System.Text;
using System.Text.Json;
using RAWSimO.Live.Protocol;
using RAWSimO.Live.Tracking;
using Xunit;

namespace RAWSimO.Live.Tests;

public class RecordedFrameSizeTests
{
    private const string Hello = "{\"type\":\"hello\",\"schema\":\"1.0\",\"network\":{\"nodes\":[],\"edges\":[]},\"meta\":{}}";

    private static string LongFrame(bool valid = true) => JsonSerializer.Serialize(new
    {
        type = "frame", schema = "1.0", step = 7, t = 2.5,
        agvs = new[] { new { agv_id = "R1", state = "blocked", task_id = "T1",
            pose = new { agv_id = "R1", source = "localized", available = true,
                x = 1.25, y = 2.5, yaw = -.3, timestamp = 42.0, frame_id = valid ? "network" : "wrong-frame" } } },
        trails = new { R1 = Enumerable.Range(0, 12000).Select(i => new[] { i * .01, 2.5 }).ToArray() },
        executions = new { R1 = new { robot_id = "R1", scene_id = "workshop", run_id = "r",
            fault = "execution guard: velocity command timeout" } },
        physical_holds = new { nodes = new Dictionary<string, string[]> { ["J"] = new[] { "R1" } },
            edges = new Dictionary<string, string[]> { ["J->K"] = new[] { "R1" } } }
    });

    private static void Read(bool indexed, string content, Action<FleetReplay> inspect)
    {
        if (!indexed)
        {
            using var replay = FleetReplay.Load(new StringReader(content));
            inspect(replay);
            return;
        }
        var path = Path.Combine(AppContext.BaseDirectory, "amr-record-size-" + Guid.NewGuid().ToString("N") + ".jsonl");
        File.WriteAllText(path, content, new UTF8Encoding(false));
        try
        {
            using var replay = FleetReplay.Open(path);
            inspect(replay);
        }
        finally
        {
            using (var exclusive = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
                Assert.True(exclusive.CanWrite);
            File.Delete(path);
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void LongObservedTrailsReplayWithoutRelaxingLiveLimit(bool indexed, bool wrapped)
    {
        var frame = LongFrame();
        Assert.True(Encoding.UTF8.GetByteCount(frame) > LiveMessageParser.MaxLineBytes);
        Assert.False(LiveMessageParser.TryParse(frame, out _, out var error));
        Assert.Contains("64 KiB", error);
        var line = wrapped ? "{\"kind\":\"frame\",\"wall\":100,\"value\":" + frame + "}" : frame;
        Read(indexed, Hello + "\n" + line, replay =>
        {
            var observed = replay.At(0).Frame;
            Assert.Equal(7, observed.Step);
            Assert.Equal(1.25, observed.Agvs[0].Pose.X);
            Assert.Equal(-.3, observed.Agvs[0].Pose.Yaw);
            Assert.Equal(12000, observed.Trails!["R1"].Count);
            Assert.Equal(119.99, observed.Trails["R1"][^1].X, 8);
            Assert.Equal("execution guard: velocity command timeout", observed.ExecutionFaults!["R1"]);
            Assert.Equal(new[] { "R1" }, observed.PhysicalHolds!.Edges["J->K"]);
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RecordedLineLimitCountsUtf8Bytes(bool indexed)
    {
        // Fewer than 4 MiB characters, more than 4 MiB UTF-8 bytes.
        var frame = LongFrame()[..^1] + ",\"note\":\"" + new string('甲', 1400000) + "\"}";
        Assert.True(frame.Length < 4 * 1024 * 1024);
        Assert.True(Encoding.UTF8.GetByteCount(frame) > 4 * 1024 * 1024);
        Assert.Throws<FormatException>(() => Read(indexed, Hello + "\n" + frame, _ => { }));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LongRecordStillRequiresValidObservedPose(bool indexed)
    {
        Assert.Throws<FormatException>(() => Read(indexed, Hello + "\n" + LongFrame(false), _ => { }));
    }
}
