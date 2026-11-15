using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using RAWSimO.Live.Networking;
using RAWSimO.Live.Protocol;
using RAWSimO.Live.Tracking;
using Xunit;

namespace RAWSimO.Live.Tests;

/// <summary>
/// A bridge that lost its own pose source, as seen by a consumer over a real socket.
/// <para>
/// The scenario is the one measured on 2026-11-05: the bridge's rclpy thread died and the
/// TCP side kept publishing one unchanged pose at 20 Hz with the sequence still advancing.
/// From the consumer side that stream is indistinguishable from a parked vehicle -- the
/// messages arrive on time and parse cleanly -- so the only evidence is the status line the
/// bridge sends about itself. This test feeds both kinds of message through the production
/// client into the production tracker, over a loopback socket.
/// </para>
/// </summary>
public class LiveSourceStalenessTests
{
    private static async Task<(int port, Task done)> ServeAsync(
        IReadOnlyList<string> lines, CancellationToken cancel)
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        var done = Task.Run(async () =>
        {
            try
            {
                using var client = await listener.AcceptTcpClientAsync(cancel);
                using var stream = client.GetStream();
                var bytes = Encoding.UTF8.GetBytes(string.Concat(lines.Select(l => l + "\n")));
                await stream.WriteAsync(bytes, cancel);
                await stream.FlushAsync(cancel);
                await Task.Delay(TimeSpan.FromSeconds(20), cancel);
            }
            catch (OperationCanceledException) { }
            catch (IOException) { }
            finally { listener.Stop(); }
        }, cancel);
        return (port, done);
    }

    [Fact]
    public async Task AFrozenPoseIsOnlyDetectableThroughTheStatusLine()
    {
        const string hello = "{\"type\":\"hello\",\"vehicle\":\"agv1\",\"proto\":1,\"frame\":\"odom\"}";
        const string frozenPose =
            "{\"type\":\"pose\",\"vehicle\":\"agv1\",\"seq\":10,\"t\":1.0,\"x\":1.2,\"y\":0.4,\"yaw\":0.0}";
        const string stale =
            "{\"type\":\"status\",\"vehicle\":\"agv1\",\"mode\":\"gazebo\",\"error\":\"odom_stale\"}";

        using var cancel = new CancellationTokenSource(TimeSpan.FromSeconds(20));
        var (port, _) = await ServeAsync(new[] { hello, frozenPose, stale, frozenPose, frozenPose },
                                         cancel.Token);

        await using var client = new LiveTcpClient(new LiveClientOptions
        {
            Host = "127.0.0.1",
            Port = port,
            HelloTimeout = TimeSpan.FromSeconds(5),
        });

        var tracker = new PoseTracker(3.0, 3.0);
        var seen = 0;
        var gotStatus = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        client.MessageReceived += message =>
        {
            // The same routing the live layer performs, minus the drawing.
            if (message is PoseMessage pose)
            {
                tracker.Update(pose, 100.0, DateTimeOffset.UtcNow);
                Interlocked.Increment(ref seen);
            }
            else if (message is StatusMessage status)
            {
                tracker.SetSourceError(status.Error);
                gotStatus.TrySetResult();
            }
        };

        await client.StartAsync(cancel.Token);
        await gotStatus.Task.WaitAsync(TimeSpan.FromSeconds(15));
        await client.StopAsync();

        Assert.True(seen >= 2, "poses did not arrive over the socket");
        Assert.Equal("odom_stale", tracker.SourceError);
        Assert.True(tracker.SourceStale);
        // The newest pose is 0 ms old by construction, so nothing about its age explains
        // the verdict: only the status line does.
        Assert.Equal(PoseFreshness.Lost, tracker.GetFreshness(100.0));
        Assert.Equal(1.2, tracker.Latest!.X, 9);
    }

    [Fact]
    public async Task ABridgeThatReportsNoErrorKeepsItsPosesFresh()
    {
        const string hello = "{\"type\":\"hello\",\"vehicle\":\"agv1\",\"proto\":1,\"frame\":\"odom\"}";
        const string pose =
            "{\"type\":\"pose\",\"vehicle\":\"agv1\",\"seq\":10,\"t\":1.0,\"x\":1.2,\"y\":0.4,\"yaw\":0.0}";
        const string ok =
            "{\"type\":\"status\",\"vehicle\":\"agv1\",\"mode\":\"gazebo\",\"error\":null}";

        using var cancel = new CancellationTokenSource(TimeSpan.FromSeconds(20));
        var (port, _) = await ServeAsync(new[] { hello, pose, ok, pose }, cancel.Token);

        await using var client = new LiveTcpClient(new LiveClientOptions
        {
            Host = "127.0.0.1",
            Port = port,
            HelloTimeout = TimeSpan.FromSeconds(5),
        });

        var tracker = new PoseTracker(3.0, 3.0);
        var gotStatus = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        client.MessageReceived += message =>
        {
            if (message is PoseMessage p) tracker.Update(p, 100.0, DateTimeOffset.UtcNow);
            else if (message is StatusMessage s)
            {
                tracker.SetSourceError(s.Error);
                gotStatus.TrySetResult();
            }
        };

        await client.StartAsync(cancel.Token);
        await gotStatus.Task.WaitAsync(TimeSpan.FromSeconds(15));
        await client.StopAsync();

        Assert.Null(tracker.SourceError);
        Assert.False(tracker.SourceStale);
        Assert.Equal(PoseFreshness.Fresh, tracker.GetFreshness(100.0));
    }
}
