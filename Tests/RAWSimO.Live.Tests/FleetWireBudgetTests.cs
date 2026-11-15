using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using RAWSimO.Live.Networking;
using RAWSimO.Live.Protocol;
using Xunit;

namespace RAWSimO.Live.Tests;

public class FleetWireBudgetTests
{
    private const string FleetHello = "{\"type\":\"hello\",\"network\":{\"nodes\":[],\"edges\":[]},\"meta\":{}}";
    private const string SingleHello = "{\"type\":\"hello\",\"vehicle\":\"R1\",\"proto\":0,\"frame\":\"network\"}";
    private static string Frame(string run = "r", bool validPose = true, int step = 1) =>
        JsonSerializer.Serialize(new
        {
            type = "frame", schema = "1.0", step, t = (double)step,
            agvs = Enumerable.Range(1, 6).Select(i => new
            {
                agv_id = "R" + i, state = "blocked", task_id = "T" + i,
                pose = new { agv_id = "R" + i, source = "localized", available = true,
                    x = 1.234567890123456 * i, y = 9.876543210987654, yaw = -.3,
                    frame_id = validPose ? "network" : "wrong-frame", timestamp = 42.0 }
            }),
            trails = Enumerable.Range(1, 6).ToDictionary(i => "R" + i,
                i => Enumerable.Range(0, 500).Select(j => new[] { j * .01234567890123456, i * 2.5 }).ToArray()),
            executions = Enumerable.Range(1, 6).ToDictionary(i => "R" + i,
                i => new { robot_id = "R" + i, scene_id = "workshop", run_id = run,
                    fault = i == 2 ? "execution guard: velocity command timeout" : null }),
            physical_holds = new { nodes = new Dictionary<string, string[]> { ["J"] = new[] { "R2" } },
                edges = new Dictionary<string, string[]> { ["J->K"] = new[] { "R2" } } }
        });

    private static async Task Observe(string hello, byte[] tail,
        Action<IReadOnlyCollection<FleetFrameMessage>, IReadOnlyCollection<Exception>> inspect,
        bool expectFrame = false, int ordinaryLimit = 64 * 1024, int fragmentSize = int.MaxValue,
        int fleetLimit = 4 * 1024 * 1024)
    {
        using var cancel = new CancellationTokenSource(TimeSpan.FromSeconds(15));
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var server = Task.Run(async () =>
        {
            try
            {
                using var socket = await listener.AcceptTcpClientAsync(cancel.Token);
                var stream = socket.GetStream();
                await stream.WriteAsync(Encoding.UTF8.GetBytes(hello + "\n"), cancel.Token);
                for (int offset = 0; offset < tail.Length; offset += Math.Min(fragmentSize, tail.Length - offset))
                {
                    int count = Math.Min(fragmentSize, tail.Length - offset);
                    await stream.WriteAsync(tail.AsMemory(offset, count), cancel.Token);
                }
                await Task.Delay(Timeout.Infinite, cancel.Token);
            }
            catch (OperationCanceledException) { }
            catch (IOException) { }
            catch (SocketException) { }
            finally { listener.Stop(); }
        });
        await using var client = new LiveTcpClient(new LiveClientOptions
        {
            Port = ((IPEndPoint)listener.LocalEndpoint).Port, MaxLineBytes = ordinaryLimit,
            MaxFleetLineBytes = fleetLimit
        });
        var frames = new ConcurrentQueue<FleetFrameMessage>();
        var errors = new ConcurrentQueue<Exception>();
        var completed = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        client.FleetFrameReceived += frame =>
        {
            frames.Enqueue(frame);
            if (expectFrame) completed.TrySetResult();
        };
        client.Error += error => { errors.Enqueue(error); completed.TrySetResult(); };
        try
        {
            await client.StartAsync(cancel.Token);
            await completed.Task.WaitAsync(TimeSpan.FromSeconds(8));
            await client.StopAsync();
            inspect(frames, errors);
        }
        finally { cancel.Cancel(); await server; }
    }

    [Fact]
    public async Task CompleteSixRobotObservationArrivesAfterFleetHandshake()
    {
        var frame = Frame();
        Assert.True(Encoding.UTF8.GetByteCount(frame) > 64 * 1024);
        Assert.False(LiveMessageParser.TryParse(frame, out _, out _));
        await Observe(FleetHello, Encoding.UTF8.GetBytes(frame + "\r\n"), (frames, errors) =>
        {
            Assert.Empty(errors);
            var observed = Assert.Single(frames);
            Assert.Equal(6, observed.Agvs.Count);
            Assert.Equal(500, observed.Trails!["R6"].Count);
            Assert.Equal(1.234567890123456, observed.Agvs[0].Pose.X);
            Assert.Equal(-.3, observed.Agvs[0].Pose.Yaw);
            Assert.Equal("execution guard: velocity command timeout", observed.ExecutionFaults!["R2"]);
            Assert.Equal(new[] { "R2" }, observed.PhysicalHolds!.Edges["J->K"]);
        }, expectFrame: true, fragmentSize: 137);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task LargeObservationsStillRequireValidPoseAndRunIdentity(bool wrongPose)
    {
        var first = Frame();
        var second = wrongPose ? Frame(validPose: false, step: 2) : Frame(run: "foreign", step: 2);
        await Observe(FleetHello, Encoding.UTF8.GetBytes(first + "\n" + second + "\n"), (frames, errors) =>
        {
            Assert.Single(frames);
            Assert.Contains(errors, e => e is FormatException &&
                e.Message.Contains(wrongPose ? "frame" : "scene/run"));
        });
    }

    [Fact]
    public async Task SingleVehicleConnectionKeepsItsSmallMessageLimit()
    {
        await Observe(SingleHello, Encoding.UTF8.GetBytes(Frame() + "\n"), (frames, errors) =>
        {
            Assert.Empty(frames);
            Assert.Contains(errors, e => e.Message.Contains("line limit"));
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task OversizedUnterminatedInputIsRejectedWithoutWaitingForNewline(bool fleet)
    {
        var bytes = Enumerable.Repeat((byte)'x', (fleet ? 4 * 1024 * 1024 : 64 * 1024) + 1).ToArray();
        await Observe(fleet ? FleetHello : SingleHello, bytes, (frames, errors) =>
        {
            Assert.Empty(frames);
            Assert.Contains(errors, e => e is IOException && e.Message.Contains("line limit"));
        });
    }

    [Fact]
    public async Task SmallerConfiguredLimitCountsUtf8BytesBeforeTheNewline()
    {
        await Observe(SingleHello, Encoding.UTF8.GetBytes(new string('甲', 400)), (frames, errors) =>
        {
            Assert.Empty(frames);
            Assert.Contains(errors, e => e is IOException && e.Message.Contains("line limit"));
        }, ordinaryLimit: 1024, fragmentSize: 101);
    }

    [Fact]
    public async Task InvalidUtf8DoesNotBecomeReplacementCharactersInAcceptedMessages()
    {
        var bytes = Encoding.UTF8.GetBytes("{\"type\":\"unknown\",\"value\":\"").Concat(new byte[] { 0xff })
            .Concat(Encoding.UTF8.GetBytes("\"}\n")).ToArray();
        await Observe(FleetHello, bytes, (frames, errors) =>
        {
            Assert.Empty(frames);
            Assert.Contains(errors, e => e is DecoderFallbackException || e.InnerException is DecoderFallbackException);
        });
    }

    [Fact]
    public async Task LargeNonObservationDoesNotUseTheFleetObservationBudget()
    {
        var message = "{\"type\":\"ping\",\"t1\":1,\"note\":\"" + new string('a', 70000) + "\"}\n";
        await Observe(FleetHello, Encoding.UTF8.GetBytes(message), (frames, errors) =>
        {
            Assert.Empty(frames);
            Assert.Contains(errors, e => e is FormatException && e.Message.Contains("64 KiB"));
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ConfiguredFleetBudgetAcceptsExactBytesAndRejectsOneExtra(bool extra)
    {
        var frame = "{\"type\":\"frame\",\"schema\":\"1.0\",\"step\":1,\"t\":1,\"agvs\":[],\"note\":\"\"}";
        frame = frame[..^2] + new string('x', 1024 - Encoding.UTF8.GetByteCount(frame) + (extra ? 1 : 0)) + "\"}";
        Assert.Equal(extra ? 1025 : 1024, Encoding.UTF8.GetByteCount(frame));
        await Observe(FleetHello, Encoding.UTF8.GetBytes(frame + "\n"), (frames, errors) =>
        {
            if (extra)
            {
                Assert.Empty(frames);
                Assert.Contains(errors, e => e is IOException && e.Message.Contains("line limit"));
            }
            else
            {
                Assert.Single(frames);
                Assert.Empty(errors);
            }
        }, expectFrame: !extra, fleetLimit: 1024);
    }

    [Fact]
    public async Task ConfiguredOrdinaryBudgetStillAppliesOnFleetConnection()
    {
        var message = "{\"type\":\"ping\",\"t1\":1,\"note\":\"" + new string('a', 1100) + "\"}\n";
        await Observe(FleetHello, Encoding.UTF8.GetBytes(message), (frames, errors) =>
        {
            Assert.Empty(frames);
            Assert.Contains(errors, e => e is FormatException && e.Message.Contains("1024"));
        }, ordinaryLimit: 1024);
    }

    [Fact]
    public async Task Utf8ObservationSurvivesFragmentBoundariesWithoutValueChanges()
    {
        var frame = Frame()[..^1] + ",\"note\":\"" + new string('甲', 5001) + "\"}";
        await Observe(FleetHello, Encoding.UTF8.GetBytes(frame + "\n"), (frames, errors) =>
        {
            Assert.Empty(errors);
            Assert.Equal(6, Assert.Single(frames).Agvs.Count);
        }, expectFrame: true, fragmentSize: 257);
    }

    [Theory]
    [InlineData(0, 4194304)]
    [InlineData(65537, 4194304)]
    [InlineData(65536, 0)]
    [InlineData(65536, 4194305)]
    public void ConfigurationCannotDisableOrEnlargeHardBounds(int ordinary, int fleet)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new LiveTcpClient(new LiveClientOptions
        {
            MaxLineBytes = ordinary, MaxFleetLineBytes = fleet
        }));
    }
}
