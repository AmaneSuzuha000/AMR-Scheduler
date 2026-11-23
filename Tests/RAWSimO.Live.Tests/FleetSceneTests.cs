using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using RAWSimO.Live.Geometry;
using RAWSimO.Live.Networking;
using RAWSimO.Live.Protocol;
using RAWSimO.Live.Tracking;
using Xunit;

namespace RAWSimO.Live.Tests;

/// <summary>
/// From wire bytes to pixels: the last step of the scheduling-stream path.
/// <para>
/// A frame arriving and a vehicle appearing on the canvas are different events, and the code
/// between them used to be untested because it lived inside a WPF window. These tests keep the
/// projection outside the window so the mapping, the colour assignment and the failure paths
/// can all be checked without a display.
/// </para>
/// </summary>
public class FleetSceneTests
{
    private static string SampleLine(string name) =>
        File.ReadLines(Path.Combine(AppContext.BaseDirectory, "samples", name))
            .First(l => !string.IsNullOrWhiteSpace(l));

    private static FleetHelloMessage Hello() =>
        (FleetHelloMessage)LiveMessageParser.Parse(SampleLine("fleet_hello.jsonl"));

    private static FleetTopology Topology() => new(Hello());

    [Fact]
    public void ASceneFitsTheNetworkInsteadOfAHandPickedSize()
    {
        // The scheduling stream reports workshop metres: the sample network spans about 120 m,
        // while the single-vehicle demo works in about 3 m. A scene built for one scale projects
        // every vehicle of the other off the canvas, which looks exactly like an empty fleet.
        var hello = Hello();
        var scene = FleetScene.ForNetwork(hello);

        var placed = hello.Nodes.Select(n => scene.Project(n.X, n.Y)).ToList();
        Assert.All(placed, p => Assert.InRange(p.X, 0.0, scene.CanvasWidth));
        Assert.All(placed, p => Assert.InRange(p.Y, 0.0, scene.CanvasHeight));

        // The margin means the outermost nodes are not on the very edge.
        Assert.True(scene.MinX < hello.Nodes.Min(n => n.X));
        Assert.True(scene.MinY < hello.Nodes.Min(n => n.Y));
    }

    [Fact]
    public void ASceneFitsIntoTheCanvasItWillBeDrawnOn()
    {
        // The scheduling network is about 120 m across. A fixed scale either shrinks it into a
        // corner or pushes it off the canvas, and neither failure announces itself, so the scale
        // is derived from the canvas instead of being a constant.
        var hello = Hello();
        var scene = FleetScene.FitNetwork(hello, 800, 600, paddingPx: 20);

        Assert.NotNull(scene);
        Assert.InRange(scene!.CanvasWidth, 799.0, 801.0);
        Assert.InRange(scene.CanvasHeight, 599.0, 601.0);

        // Every node must land inside the padded area, and the extent must actually use it:
        // a scene that fits but draws everything in 10 px has "fitted" nothing.
        var pts = hello.Nodes.Select(n => scene.Project(n.X, n.Y)).ToList();
        Assert.All(pts, p => Assert.InRange(p.X, 0.0, 800.0));
        Assert.All(pts, p => Assert.InRange(p.Y, 0.0, 600.0));
        Assert.True(pts.Max(p => p.X) - pts.Min(p => p.X) > 600.0,
            "the network should span most of the canvas width");
    }

    [Fact]
    public void AFittedSceneKeepsOneMetreTheSameOnBothAxes()
    {
        // Squashing one axis to make the picture fit would make distances on screen lie about
        // distances on the floor.
        var scene = FleetScene.FitNetwork(Hello(), 800, 600, paddingPx: 20)!;
        var a = scene.Project(0.0, 0.0);
        var bx = scene.Project(10.0, 0.0);
        var by = scene.Project(0.0, 10.0);
        Assert.Equal(10.0 * scene.Transform.PixelsPerMeter, Math.Abs(bx.X - a.X), 6);
        Assert.Equal(10.0 * scene.Transform.PixelsPerMeter, Math.Abs(by.Y - a.Y), 6);
    }

    [Fact]
    public void AnUnfittableNetworkIsReportedRatherThanDrawnKiloPixelsAway()
    {
        Assert.Null(FleetScene.FitNetwork(Hello(), 0, 600));
        Assert.Null(FleetScene.FitNetwork(Hello(), 800, 0));
        Assert.Null(FleetScene.FitNetwork(Hello(), double.NaN, 600));
    }

    [Fact]
    public void MapUpBecomesCanvasDown()
    {
        // A scene that forgot the flip still produces plausible numbers, so it gets its own test.
        var scene = FleetScene.Over(0.0, 0.0, 10.0, 10.0, pixelsPerMeter: 40.0);
        Assert.Equal(0.0, scene.Project(0.0, 10.0).Y, 6);          // top of the map is y=0 px
        Assert.Equal(400.0, scene.Project(0.0, 0.0).Y, 6);          // bottom of the map is y=400 px
        Assert.Equal(0.0, scene.Project(0.0, 0.0).X, 6);
        Assert.Equal(400.0, scene.Project(10.0, 0.0).X, 6);
    }

    [Fact]
    public void APositionsVehicleOnTheCanvasWhereTheMapSaysItIs()
    {
        var hello = Hello();
        var topology = new FleetTopology(hello);
        var frame = (FleetFrameMessage)LiveMessageParser.Parse(SampleLine("fleet_frame.jsonl"));
        var scene = FleetScene.ForNetwork(hello);

        var markers = scene.Project(topology.Place(frame));

        Assert.Equal(frame.Agvs.Count, markers.Count);
        Assert.All(markers, m => Assert.False(scene.IsOffCanvas(m), m.AgvId + " projects off-canvas"));

        // Every marker must sit exactly where the placement it came from says. This is the
        // assertion that catches a scene which draws something plausible but not the truth.
        var byId = markers.ToDictionary(m => m.AgvId);
        foreach (var placement in topology.Place(frame).Placements)
        {
            var expected = scene.Project(placement.X, placement.Y);
            Assert.Equal(expected.X, byId[placement.AgvId].CanvasX, 6);
            Assert.Equal(expected.Y, byId[placement.AgvId].CanvasY, 6);
        }
    }

    [Fact]
    public void EachVehicleKeepsItsColourAcrossFrames()
    {
        var hello = Hello();
        var topology = new FleetTopology(hello);
        var frame = (FleetFrameMessage)LiveMessageParser.Parse(SampleLine("fleet_frame.jsonl"));
        var scene = FleetScene.ForNetwork(hello);

        var first = scene.Project(topology.Place(frame)).ToDictionary(m => m.AgvId);
        var second = scene.Project(topology.Place(frame)).ToDictionary(m => m.AgvId);

        foreach (var agv in first.Keys)
        {
            Assert.Equal(first[agv].R, second[agv].R);
            Assert.Equal(first[agv].G, second[agv].G);
            Assert.Equal(first[agv].B, second[agv].B);
        }
        // Distinct vehicles get distinct colours, otherwise the colours carry no information.
        Assert.Equal(first.Count, first.Values.Select(m => (m.R, m.G, m.B)).Distinct().Count());
    }

    [Fact]
    public void ARecycledVehicleIdGetsItsColourBack()
    {
        var scene = FleetScene.Over(0.0, 0.0, 10.0, 10.0);
        var a = scene.SlotOf("agv1");
        scene.Forget("agv1");
        Assert.Equal(0, scene.VehicleCount);
        Assert.Equal(a, scene.SlotOf("agv2"));
    }

    [Fact]
    public void AFailedPlacementStillDrawsTheVehiclesThatWerePlaced()
    {
        // One unknown id must not blank the whole picture: the vehicles we do know where to put
        // are still true, and hiding them would make a single bad id look like a dead fleet.
        var hello = Hello();
        var topology = new FleetTopology(hello);
        var frame = (FleetFrameMessage)LiveMessageParser.Parse(SampleLine("fleet_frame.jsonl"));
        var scene = FleetScene.ForNetwork(hello);

        var bad = new FleetFrameMessage(frame.Schema, frame.Step, frame.T, new List<FleetAgv>
        {
            frame.Agvs[0],
            new FleetAgv("agvX", new FleetPose("agvX", Node: "NOPE"), "idle"),
        });
        var placed = topology.Place(bad);
        Assert.False(placed.Complete);
        Assert.Single(placed.Placements);

        var markers = scene.Project(placed);
        Assert.Single(markers);
        Assert.Equal(frame.Agvs[0].AgvId, markers[0].AgvId);
    }

    [Fact]
    public async Task ARealSocketStreamEndsUpAsCanvasPixels()
    {
        // Everything above could pass with the projection wired to nothing. This one starts at
        // the socket: the same newline-JSON the Python producer writes, through the production
        // client, through placement, out as canvas pixels.
        var helloLine = SampleLine("fleet_hello.jsonl");
        var frameLine = SampleLine("fleet_frame.jsonl");

        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        using var cancel = new CancellationTokenSource(TimeSpan.FromSeconds(20));
        var server = Task.Run(async () =>
        {
            try
            {
                using var client = await listener.AcceptTcpClientAsync(cancel.Token);
                using var stream = client.GetStream();
                var bytes = Encoding.UTF8.GetBytes(
                    string.Join("\n", helloLine, frameLine, frameLine, frameLine) + "\n");
                await stream.WriteAsync(bytes, cancel.Token);
                await stream.FlushAsync(cancel.Token);
                await Task.Delay(TimeSpan.FromSeconds(15), cancel.Token);
            }
            catch (OperationCanceledException) { }
            catch (IOException) { }
            finally { listener.Stop(); }
        }, cancel.Token);

        await using var tcp = new LiveTcpClient(new LiveClientOptions
        {
            Host = "127.0.0.1",
            Port = port,
            HelloTimeout = TimeSpan.FromSeconds(5),
        });

        FleetScene? scene = null;
        FleetTopology? topology = null;
        var rendered = new List<IReadOnlyList<FleetMarker>>();
        var done = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        tcp.FleetHelloReceived += h =>
        {
            topology = new FleetTopology(h);
            scene = FleetScene.ForNetwork(h);
        };
        tcp.FleetFrameReceived += f =>
        {
            Assert.NotNull(topology);
            Assert.NotNull(scene);
            var markers = scene!.Project(topology!.Place(f));
            lock (rendered) rendered.Add(markers);
            if (rendered.Count >= 3) done.TrySetResult();
        };

        await tcp.StartAsync(cancel.Token);
        await done.Task.WaitAsync(TimeSpan.FromSeconds(15));
        await tcp.StopAsync();

        Assert.Equal(3, rendered.Count);
        Assert.All(rendered, frameMarkers => Assert.NotEmpty(frameMarkers));
        Assert.All(rendered, frameMarkers =>
            Assert.All(frameMarkers, m => Assert.False(scene!.IsOffCanvas(m), m.AgvId + " off-canvas")));
    }
}
