using System.Text;
using System.Text.Json;
using RAWSimO.Live.Tracking;
using Xunit;

namespace RAWSimO.Live.Tests;

public class IndexedFleetReplayTests
{
    private const string Control = "\"control\":{\"session_id\":\"s\",\"scene_id\":\"车间\",\"run_id\":\"r\",\"paused\":false,\"operations\":[],\"routes\":{\"R1\":\"r:R1:1\"}}";
    private const string Hello = "{\"type\":\"hello\",\"schema\":\"1.0\",\"network\":{\"nodes\":[],\"edges\":[]},\"meta\":{}," + Control + "}";
    private static string Frame(int t, bool available = true) =>
        "{\"type\":\"frame\",\"schema\":\"1.0\",\"step\":" + t + ",\"t\":" + t +
        ",\"agvs\":[{\"agv_id\":\"R1\",\"state\":\"blocked\",\"pose\":{\"agv_id\":\"R1\",\"source\":\"localized\",\"available\":" +
        (available ? "true" : "false") + ",\"x\":0,\"y\":0,\"yaw\":0,\"frame_id\":\"network\"}}]," + Control + "}";
    private static string Wrap(int wall, string frame) => "{\"kind\":\"frame\",\"wall\":" + wall + ",\"value\":" + frame + "}";
    private static string TemporaryPath() => Path.Combine(AppContext.BaseDirectory, "amr-replay-" + Guid.NewGuid().ToString("N") + ".jsonl");

    [Theory]
    [InlineData("\n", false)]
    [InlineData("\r\n", true)]
    public void IndexedAndMemoryModesAgreeAcrossUtf8BoundariesAndLoss(string separator, bool bom)
    {
        var path = TemporaryPath();
        try
        {
            // Cross the scanner's 64 KiB block inside multibyte UTF-8, and omit the final newline.
            var padding = "{\"kind\":\"operator_request\",\"ignored\":\"" + new string('甲', 23000) + "\"}";
            var content = string.Join(separator, new[] { "{\"kind\":\"operator_endpoint\",\"hello\":" + Hello + "}",
                padding, Wrap(100, Frame(0)), Wrap(105, Frame(1, false)), Wrap(107, Frame(2)), Wrap(107, Frame(3)) });
            File.WriteAllText(path, content, new UTF8Encoding(bom));
            using var indexed = FleetReplay.Open(path);
            using var memory = FleetReplay.Load(new StringReader(content));
            Assert.Equal(memory.Duration, indexed.Duration);
            Assert.Equal(memory.Frames.Count, indexed.Frames.Count);
            foreach (var time in new[] { -1d, 0, 4.9, 5, 6.9, 7, 100 })
            {
                var expected = memory.At(time);
                var actual = indexed.At(time);
                Assert.Equal(expected.Seconds, actual.Seconds);
                Assert.Equal(expected.Frame.Step, actual.Frame.Step);
                Assert.Equal(expected.Frame.Agvs[0].Pose.Available, actual.Frame.Agvs[0].Pose.Available);
            }
            Assert.False(indexed.At(6).Frame.Agvs[0].Pose.Available);
            Assert.Equal(3, indexed.At(7).Frame.Step); // Last observation wins at duplicate timestamps.
        }
        finally { File.Delete(path); }
    }

    [Theory]
    [InlineData("missing_hello")]
    [InlineData("duplicate_hello")]
    [InlineData("mixed_run")]
    [InlineData("mixed_scene")]
    [InlineData("reverse_wall")]
    [InlineData("malformed")]
    public void RejectedFileReleasesItsHandle(string problem)
    {
        var path = TemporaryPath();
        try
        {
            var content = problem switch
            {
                "missing_hello" => Frame(0),
                "duplicate_hello" => Hello + "\n" + Frame(0) + "\n" + Hello,
                "mixed_run" => Hello + "\n" + Frame(0).Replace("\"run_id\":\"r\"", "\"run_id\":\"different\""),
                "mixed_scene" => Hello + "\n" + Frame(0).Replace("车间", "another"),
                "reverse_wall" => Hello + "\n" + Wrap(10, Frame(0)) + "\n" + Wrap(9, Frame(1)),
                _ => Hello + "\n{" ,
            };
            File.WriteAllText(path, content, new UTF8Encoding(false));
            if (problem == "malformed") Assert.ThrowsAny<JsonException>(() => FleetReplay.Open(path));
            else Assert.Throws<FormatException>(() => FleetReplay.Open(path));
            using var exclusive = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void LargeRecordingIsIndexedAndDisposalReleasesTheSource()
    {
        var path = TemporaryPath();
        try
        {
            var padding = new string('x', 60 * 1024);
            using (var writer = new StreamWriter(path, false, new UTF8Encoding(false)))
            {
                writer.WriteLine(Hello);
                for (int i = 0; i < 1100; i++)
                    writer.WriteLine(Frame(i)[..^1] + ",\"historical_note\":\"" + padding + "\"}");
            }
            Assert.True(new FileInfo(path).Length > 64 * 1024 * 1024);
            using (var reader = File.OpenText(path)) Assert.Throws<FormatException>(() => FleetReplay.Load(reader));
            using var recording = FleetReplay.Open(path);
            Assert.Equal(1100, recording.Frames.Count);
            Assert.Equal(1099, recording.Duration);
            Assert.Equal(1099, recording.At(2000).Frame.Step);
            Assert.Equal(0, recording.At(0).Frame.Step);
            recording.Dispose();
            Assert.Throws<ObjectDisposedException>(() => recording.At(10));
            using var exclusive = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        }
        finally { File.Delete(path); }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void BoundedScannerRejectsOversizedLinesAndInvalidUtf8(bool oversized)
    {
        var path = TemporaryPath();
        try
        {
            File.WriteAllBytes(path, oversized ? Encoding.UTF8.GetBytes(new string('x', 4 * 1024 * 1024 + 1)) : new byte[] { 0xff });
            if (oversized) Assert.Throws<FormatException>(() => FleetReplay.Open(path));
            else Assert.Throws<DecoderFallbackException>(() => FleetReplay.Open(path));
            using var exclusive = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void CancelledIndexingReleasesTheSource()
    {
        var path = TemporaryPath();
        try
        {
            File.WriteAllText(path, Hello + "\n" + Frame(0), new UTF8Encoding(false));
            using var cancellation = new CancellationTokenSource();
            cancellation.Cancel();
            Assert.Throws<OperationCanceledException>(() => FleetReplay.Open(path, cancellation.Token));
            using var exclusive = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        }
        finally { File.Delete(path); }
    }
}
