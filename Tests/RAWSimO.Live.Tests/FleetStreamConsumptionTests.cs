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
/// Drives the scheduling stream through a real socket, not a single parsed line.
/// <para>
/// Parsing a captured sample proves the parser; it does not prove that a consumer
/// subscribed to <see cref="LiveTcpClient"/> ever receives anything. These tests run
/// a loopback server that writes the same newline-JSON the Python producer writes,
/// and assert on what arrives through the client and on where it puts the vehicles.
/// </para>
/// </summary>
public class FleetStreamConsumptionTests
{
    private static string SampleLine(string name) =>
        File.ReadLines(Path.Combine(AppContext.BaseDirectory, "samples", name))
            .First(l => !string.IsNullOrWhiteSpace(l));

    /// <summary>Serves a fixed set of lines to one client, then holds the socket open.</summary>
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
                var payload = string.Concat(lines.Select(l => l + "\n"));
                var bytes = Encoding.UTF8.GetBytes(payload);
                await stream.WriteAsync(bytes, cancel);
                await stream.FlushAsync(cancel);
                // Keep the connection open so the client does not treat the end of the
                // script as a dropped link and start reconnecting mid-assertion.
                await Task.Delay(TimeSpan.FromSeconds(20), cancel);
            }
            catch (OperationCanceledException) { }
            catch (IOException) { }
            finally { listener.Stop(); }
        }, cancel);
        return (port, done);
    }

    [Fact]
    public async Task FleetHelloAndFramesArriveOverARealSocket()
    {
        var hello = SampleLine("fleet_hello.jsonl");
        var frame = SampleLine("fleet_frame.jsonl");

        using var cancel = new CancellationTokenSource(TimeSpan.FromSeconds(20));
        var (port, served) = await ServeAsync(
            new[] { hello, frame, frame, frame }, cancel.Token);

        await using var client = new LiveTcpClient(new LiveClientOptions
        {
            Host = "127.0.0.1",
            Port = port,
            HelloTimeout = TimeSpan.FromSeconds(5),
        });

        var hellos = new List<FleetHelloMessage>();
        var frames = new List<FleetFrameMessage>();
        var arrived = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        client.FleetHelloReceived += h => { lock (hellos) hellos.Add(h); };
        client.FleetFrameReceived += f =>
        {
            lock (frames) frames.Add(f);
            if (frames.Count >= 3) arrived.TrySetResult();
        };

        await client.StartAsync(cancel.Token);
        await arrived.Task.WaitAsync(TimeSpan.FromSeconds(15));
        Assert.Equal(LiveConnectionState.Connected, client.State);
        await client.StopAsync();

        Assert.Single(hellos);
        Assert.Equal(3, frames.Count);

        // Every frame must be placeable against the network the hello announced; a
        // consumer that receives frames it cannot place is no better off than one that
        // receives nothing, except that it fails later and less clearly.
        var topology = new FleetTopology(hellos[0]);
        Assert.True(topology.NodeCount > 0);
        Assert.True(topology.EdgeCount > 0);

        foreach (var f in frames)
        {
            var result = topology.Place(f);
            Assert.True(result.Complete, result.Detail);
            Assert.Equal(f.Agvs.Count, result.Placements.Count);
        }

        cancel.Cancel();
        try { await served; } catch (OperationCanceledException) { }
    }

    [Fact]
    public void InTransitVehicleIsPlacedBetweenItsTwoNodes()
    {
        var hello = LiveMessageParser.Parse(SampleLine("fleet_hello.jsonl"));
        var frame = (FleetFrameMessage)LiveMessageParser.Parse(SampleLine("fleet_frame.jsonl"));
        var topology = new FleetTopology((FleetHelloMessage)hello);

        var moving = frame.Agvs.FirstOrDefault(a => a.Pose.Edge is not null);
        Assert.NotNull(moving);

        var result = topology.Place(frame);
        Assert.True(result.Complete, result.Detail);

        var placed = result.Placements.Single(p => p.AgvId == moving!.AgvId);
        Assert.True(placed.OnEdge);

        // It must lie on the segment between the edge endpoints, not at either end
        // unless progress really is 0 or 1.
        var parts = moving!.Pose.Edge!.Split("->");
        var from = topology.Node(parts[0])!;
        var to = topology.Node(parts[1])!;
        var loX = Math.Min(from.X, to.X);
        var hiX = Math.Max(from.X, to.X);
        Assert.InRange(placed.X, loX - 1e-9, hiX + 1e-9);
        // Recompute the expected interpolation independently of the code under test.
        var expectedX = from.X + (to.X - from.X) * moving.Pose.Progress;
        var expectedY = from.Y + (to.Y - from.Y) * moving.Pose.Progress;
        Assert.Equal(expectedX, placed.X, 9);
        Assert.Equal(expectedY, placed.Y, 9);

        if (moving.Pose.Progress > 1e-6 && moving.Pose.Progress < 1.0 - 1e-6)
        {
            var offFromEnd = Math.Abs(placed.X - from.X) + Math.Abs(placed.Y - from.Y);
            var offFromTo = Math.Abs(placed.X - to.X) + Math.Abs(placed.Y - to.Y);
            Assert.True(offFromEnd > 1e-9 && offFromTo > 1e-9,
                "progress strictly inside the edge must not land on an endpoint");
        }
    }

    [Fact]
    public void AVehicleWithNoPositionFailsLoudlyInsteadOfVanishing()
    {
        // A pose with neither node nor edge has nowhere to be drawn. Dropping it would
        // leave a map that looks complete while a robot is missing from it.
        var hello = (FleetHelloMessage)LiveMessageParser.Parse(SampleLine("fleet_hello.jsonl"));
        var topology = new FleetTopology(hello);
        var frame = new FleetFrameMessage("1.0", 7, 7.0, new[]
        {
            new FleetAgv("AGV00", new FleetPose("AGV00", Node: null, Edge: null), "idle"),
        });

        var result = topology.Place(frame);
        Assert.False(result.Complete);
        Assert.Equal(FleetPlacementFailure.NoPosition, result.Failure);
        Assert.Contains("AGV00", result.Detail);
    }

    [Fact]
    public void AnUnknownNodeFailsWithTheIdThatCausedIt()
    {
        var hello = (FleetHelloMessage)LiveMessageParser.Parse(SampleLine("fleet_hello.jsonl"));
        var topology = new FleetTopology(hello);
        var frame = new FleetFrameMessage("1.0", 8, 8.0, new[]
        {
            new FleetAgv("AGV99", new FleetPose("AGV99", Node: "NOWHERE"), "idle"),
        });

        var result = topology.Place(frame);
        Assert.False(result.Complete);
        Assert.Equal(FleetPlacementFailure.UnknownTopology, result.Failure);
        Assert.Contains("AGV99", result.Detail);
        Assert.Contains("NOWHERE", result.Detail);
    }

    [Fact]
    public void ProgressOutsideZeroOneIsClampedOntoTheEdge()
    {
        // A producer rounding past the ends must not draw the vehicle beyond its edge.
        var hello = (FleetHelloMessage)LiveMessageParser.Parse(SampleLine("fleet_hello.jsonl"));
        var topology = new FleetTopology(hello);

        var frame = new FleetFrameMessage("1.0", 9, 9.0, new[]
        {
            new FleetAgv("AGV00",
                new FleetPose("AGV00", Edge: "R4->R3", Progress: 1.7), "to_dropoff"),
        });

        var result = topology.Place(frame);
        Assert.True(result.Complete, result.Detail);
        var placed = result.Placements.Single();
        var from = topology.Node("R4")!;
        var to = topology.Node("R3")!;
        Assert.Equal(to.X, placed.X, 9);
        Assert.Equal(to.Y, placed.Y, 9);
    }
}
