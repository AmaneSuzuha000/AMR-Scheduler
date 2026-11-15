using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using RAWSimO.Live.Networking;
using RAWSimO.Live.Protocol;
using RAWSimO.Live.Tracking;

namespace RAWSimO.Live.Harness;

/// <summary>
/// Connects to a live scheduling stream and reports what a GUI would draw.
/// <para>
/// This exists because a passing unit test only proves the parser; it does not prove
/// that a consumer subscribed to the client ever receives a frame, nor what it would
/// put on screen. Running it against the real Python producer is the only check that
/// covers the socket, the parser, the placement step, and the drawing coordinates
/// together.
/// </para>
/// </summary>
public static class Program
{
    /// <summary>Runs the harness.</summary>
    /// <param name="args">host, port, expected frames.</param>
    public static async Task<int> Main(string[] args)
    {
        var host = args.Length > 0 ? args[0] : "127.0.0.1";
        var port = args.Length > 1 ? int.Parse(args[1]) : 9411;
        var expected = args.Length > 2 ? int.Parse(args[2]) : 0;

        var topology = (FleetTopology?)null;
        var scene = (FleetScene?)null;
        var offCanvas = 0;
        double minX = double.PositiveInfinity, maxX = double.NegativeInfinity;
        double minY = double.PositiveInfinity, maxY = double.NegativeInfinity;
        var frames = 0;
        var placedVehicles = 0;
        var failures = 0;
        string? firstDetail = null;
        var lastX = double.NaN;
        var lastY = double.NaN;
        var done = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        await using var client = new LiveTcpClient(new LiveClientOptions
        {
            Host = host,
            Port = port,
            HelloTimeout = TimeSpan.FromSeconds(10),
            ConnectTimeout = TimeSpan.FromSeconds(10),
        });

        client.FleetHelloReceived += h =>
        {
            topology = new FleetTopology(h);
            // The scene is sized by the announced network, so the projected pixels mean the same
            // thing a window would draw. Without this the harness would only prove placement.
            scene = FleetScene.ForNetwork(h);
            Console.WriteLine($"hello schema={h.Schema} source={h.Source} " +
                $"nodes={topology.NodeCount} edges={topology.EdgeCount} " +
                $"tasks={h.Meta.NTasks} agvs={h.Meta.NAgvs}");
        };

        client.FleetFrameReceived += f =>
        {
            frames++;
            if (topology is null)
            {
                failures++;
                firstDetail ??= "frame arrived before hello";
                return;
            }
            var result = topology.Place(f);
            if (!result.Complete)
            {
                failures++;
                firstDetail ??= result.Detail;
                return;
            }
            placedVehicles += result.Placements.Count;
            if (scene is null) { failures++; firstDetail ??= "no scene"; return; }
            var markers = scene.Project(result);
            foreach (var m in markers)
            {
                if (scene.IsOffCanvas(m)) offCanvas++;
                if (m.CanvasX < minX) minX = m.CanvasX;
                if (m.CanvasX > maxX) maxX = m.CanvasX;
                if (m.CanvasY < minY) minY = m.CanvasY;
                if (m.CanvasY > maxY) maxY = m.CanvasY;
            }
            var lead = result.Placements[0];
            lastX = lead.X;
            lastY = lead.Y;
            if (frames <= 3 || frames % 50 == 0)
            {
                Console.WriteLine($"step {f.Step,5} t={f.T,8:F1} " +
                    string.Join(" | ", result.Placements.Select(p =>
                        $"{p.AgvId} ({p.X,6:F1},{p.Y,6:F1}) {p.State}")));
            }
            if (expected > 0 && frames >= expected) done.TrySetResult();
        };

        client.Error += e =>
        {
            // A clean close after the last frame surfaces here as an I/O error. Report
            // it but do not fail on it: the frames already arrived.
            Console.WriteLine($"note: {e.GetType().Name}: {e.Message}");
        };

        await client.StartAsync();

        if (expected > 0)
        {
            var completed = await Task.WhenAny(done.Task, Task.Delay(TimeSpan.FromSeconds(60)));
            if (completed != done.Task)
            {
                Console.WriteLine($"FAIL: only {frames}/{expected} frames arrived");
                await client.StopAsync();
                return 1;
            }
        }
        else
        {
            await Task.Delay(TimeSpan.FromSeconds(5));
        }

        await client.StopAsync();

        Console.WriteLine();
        Console.WriteLine($"frames={frames} placed_vehicles={placedVehicles} failures={failures}");
        if (firstDetail is not null) Console.WriteLine($"first failure: {firstDetail}");
        Console.WriteLine($"last lead vehicle at ({lastX:F2}, {lastY:F2})");
        if (scene is not null && frames > 0)
        {
            Console.WriteLine(string.Format(
                "canvas {0:F0}x{1:F0} px, drawn extent x[{2:F0}..{3:F0}] y[{4:F0}..{5:F0}], off-canvas {6}",
                scene.CanvasWidth, scene.CanvasHeight, minX, maxX, minY, maxY, offCanvas));
        }

        var ok = frames > 0 && failures == 0 && placedVehicles >= frames && offCanvas == 0;
        Console.WriteLine(ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
}
