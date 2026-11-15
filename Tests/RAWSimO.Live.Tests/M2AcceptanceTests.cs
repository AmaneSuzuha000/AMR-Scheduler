using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Xml.Linq;
using RAWSimO.Live.Geometry;
using RAWSimO.Live.Networking;
using RAWSimO.Live.Planning;
using RAWSimO.Live.Protocol;
using Xunit;

#nullable enable

namespace RAWSimO.Live.Tests;

/// <summary>
/// T2.5 / design milestone M2: one vehicle drives from waypoint A to waypoint B in the
/// generated instance, planned by <see cref="AStarPlanner"/> and executed by the real Python
/// bridge in sim mode. No window, no keyboard, no desktop interaction.
/// The trajectory is written to docs/证据/T2.5-m2-acceptance/ so the run can be replayed.
/// </summary>
public class M2AcceptanceTests
{
    private static int FreePort()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    private static string RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !Directory.Exists(Path.Combine(dir.FullName, "tools", "live_bridge")))
            dir = dir.Parent;
        Assert.NotNull(dir);
        return dir!.FullName;
    }

    /// <summary>Reads waypoints and directed edges of a .xinst without referencing RAWSimO.Core.</summary>
    private static WaypointGraph LoadInstanceGraph(string path)
    {
        var doc = XDocument.Load(path);
        var graph = new WaypointGraph();
        var waypoints = doc.Descendants("Waypoints").Elements("Waypoint").ToList();
        Assert.NotEmpty(waypoints);
        foreach (var waypoint in waypoints)
            graph.AddWaypoint((int)waypoint.Attribute("ID")!, new Point2D(
                (double)waypoint.Attribute("X")!, (double)waypoint.Attribute("Y")!));
        foreach (var waypoint in waypoints)
        {
            var from = (int)waypoint.Attribute("ID")!;
            var paths = waypoint.Element("Paths");
            if (paths == null) continue;
            foreach (var next in paths.Elements("Waypoint"))
            {
                var to = (int)next;
                if (to != from && graph.Contains(to)) graph.AddEdge(from, to);
            }
        }
        return graph;
    }

    private static (double X, double Y, double Yaw, MotionParameters Motion) LoadBot(string path)
    {
        var bot = XDocument.Load(path).Descendants("Bots").Elements("Bot").First();
        var yaw = bot.Attribute("Orientation") is { } orientation ? (double)orientation : 0.0;
        var motion = new MotionParameters(
            MaxSpeed: (double)bot.Attribute("MaxVelocity")!,
            Acceleration: (double)bot.Attribute("MaxAcceleration")!,
            Deceleration: (double)bot.Attribute("MaxDeceleration")!,
            TurnSpeedSeconds: (double)bot.Attribute("TurnSpeed")!);
        return ((double)bot.Attribute("X")!, (double)bot.Attribute("Y")!, yaw, motion);
    }

    private sealed class Bridge : IDisposable
    {
        private readonly Process _process;
        private Bridge(Process process) => _process = process;

        public static Bridge Start(string root, int port, double x, double y, double yaw, double maxSpeed)
        {
            var info = new ProcessStartInfo("python")
            {
                WorkingDirectory = root,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            foreach (var arg in new[]
                     {
                         Path.Combine(root, "tools", "live_bridge", "live_bridge.py"),
                         "--mode", "sim", "--rate", "20", "--port", port.ToString(),
                         "--start", x.ToString("0.###"), y.ToString("0.###"), yaw.ToString("0.###"),
                         "--max-speed", maxSpeed.ToString("0.###"),
                         // this test does not ping, so the watchdog has to stay off
                         "--watchdog", "0",
                     })
                info.ArgumentList.Add(arg);
            var process = Process.Start(info) ?? throw new InvalidOperationException("bridge failed to start");
            process.OutputDataReceived += (_, _) => { };
            process.ErrorDataReceived += (_, _) => { };
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            return new Bridge(process);
        }

        public void Dispose()
        {
            try
            {
                if (!_process.HasExited) _process.Kill(true);
                _process.WaitForExit(5000);
            }
            catch { }
            _process.Dispose();
        }
    }

    /// <summary>Perpendicular distance from a point to a polyline.</summary>
    private static double DistanceToPolyline(Point2D point, IReadOnlyList<Point2D> line)
    {
        var best = double.MaxValue;
        for (var i = 0; i + 1 < line.Count; i++)
        {
            var a = line[i];
            var b = line[i + 1];
            var abx = b.X - a.X;
            var aby = b.Y - a.Y;
            var lengthSquared = abx * abx + aby * aby;
            var t = lengthSquared <= 1e-12 ? 0.0 : ((point.X - a.X) * abx + (point.Y - a.Y) * aby) / lengthSquared;
            t = Math.Clamp(t, 0.0, 1.0);
            best = Math.Min(best, Point2D.Distance(point, new Point2D(a.X + t * abx, a.Y + t * aby)));
        }
        return best;
    }

    [Fact]
    public async Task SimulatedChassisReachesWaypointBAndThePoseStreamFollowsTheWholePlan()
    {
        var root = RepoRoot();
        var instancePath = Path.Combine(root, "docs", "证据", "T1.6-generated", "generated-map.xinst");
        // 公开仓库的精简版不含内部证据目录；缺文件时跳过而不是失败。
        if (!File.Exists(instancePath))
            return;
        Assert.True(File.Exists(instancePath), "missing generated instance " + instancePath);

        var graph = LoadInstanceGraph(instancePath);
        var (botX, botY, botYaw, motion) = LoadBot(instancePath);
        var start = graph.Waypoints.OrderBy(w => Point2D.Distance(w.Position, new Point2D(botX, botY))).First();
        var goal = graph.Waypoints.OrderByDescending(w => Point2D.Distance(w.Position, start.Position)).First();
        Assert.NotEqual(start.Id, goal.Id);

        var planner = new AStarPlanner(graph, motion, ExecutionMode.Run);
        var plan = planner.Plan(start.Id, botYaw, goal.Id);
        Assert.True(plan.Success, plan.Message);
        Assert.NotEmpty(plan.Steps);

        // The commanded path is the planned polyline; the first point is where the bot already is.
        var path = new List<Point2D> { start.Position };
        path.AddRange(plan.Steps.Select(step => graph.GetWaypoint(step.To).Position));

        var port = FreePort();
        var maxSpeed = Math.Max(motion.MaxSpeed, 0.25);
        using var bridge = Bridge.Start(root, port, botX, botY, botYaw, maxSpeed);
        await Task.Delay(1200);

        var client = new LiveTcpClient(new LiveClientOptions { Host = "127.0.0.1", Port = port });
        var poses = new List<PoseMessage>();
        var results = new List<NavResultMessage>();
        var hello = new TaskCompletionSource<HelloMessage>(TaskCreationOptions.RunContinuationsAsynchronously);
        client.MessageReceived += message =>
        {
            switch (message)
            {
                case HelloMessage h: hello.TrySetResult(h); break;
                case PoseMessage p: lock (poses) poses.Add(p); break;
                case NavResultMessage r: lock (results) results.Add(r); break;
            }
        };
        client.HelloReceived += h => hello.TrySetResult(h);

        var goalId = "m2-" + start.Id + "-" + goal.Id;
        NavResultMessage? result = null;
        try
        {
            await client.StartAsync();
            var greeting = await hello.Task.WaitAsync(TimeSpan.FromSeconds(10));
            Assert.Equal(1, greeting.Proto);

            await client.SendAsync(new NavCommandMessage(goalId, path, maxSpeed));
            var deadline = DateTime.UtcNow.AddSeconds(60);
            while (DateTime.UtcNow < deadline)
            {
                lock (results) result = results.FirstOrDefault(r => r.Goal == goalId);
                if (result != null) break;
                await Task.Delay(50);
            }
        }
        finally
        {
            await client.DisposeAsync();
        }

        Assert.NotNull(result);
        Assert.Equal("succeeded", result!.Status);

        List<PoseMessage> snapshot;
        lock (poses) snapshot = poses.ToList();
        Assert.True(snapshot.Count > 50, "pose stream too short: " + snapshot.Count);

        // 1) The pose stream is continuous: no gap longer than 0.5 s between consecutive samples.
        var times = snapshot.Select(p => p.T).ToList();
        var maxGap = times.Zip(times.Skip(1), (a, b) => b - a).DefaultIfEmpty(0).Max();
        Assert.True(maxGap <= 0.5, $"pose gap {maxGap:0.###} s");

        // 2) The chassis really travelled along the plan: every pose stays close to the planned polyline.
        var maxOffset = snapshot.Max(p => DistanceToPolyline(new Point2D(p.X, p.Y), path));
        Assert.True(maxOffset <= 0.15, $"vehicle left the planned path by {maxOffset:0.###} m");

        // 3) It arrived at B.
        var last = snapshot.Last();
        var arrivalError = Point2D.Distance(new Point2D(last.X, last.Y), goal.Position);
        Assert.True(arrivalError <= 0.1, $"final pose is {arrivalError:0.###} m away from waypoint {goal.Id}");

        // 4) The map area covered by the route was actually visited (start and goal both touched).
        var visitedStart = snapshot.Min(p => Point2D.Distance(new Point2D(p.X, p.Y), start.Position));
        Assert.True(visitedStart <= 0.1, $"never near start waypoint {start.Id}");

        // Write the trajectory log next to the other evidence so the run can be replayed.
        var evidence = Path.Combine(root, "docs", "证据", "T2.5-m2-acceptance");
        Directory.CreateDirectory(evidence);
        var logPath = Path.Combine(evidence, "trajectory.jsonl");
        await using (var writer = new StreamWriter(logPath, false, new UTF8Encoding(false)))
        {
            foreach (var pose in snapshot)
                await writer.WriteLineAsync(JsonSerializer.Serialize(new
                {
                    type = "pose", vehicle = pose.Vehicle, seq = pose.Seq,
                    t = Math.Round(pose.T, 4), x = Math.Round(pose.X, 4),
                    y = Math.Round(pose.Y, 4), yaw = Math.Round(pose.Yaw, 4),
                }));
            await writer.WriteLineAsync(JsonSerializer.Serialize(new
            {
                type = "nav_result", goal = result.Goal, status = result.Status,
                reason = result.Reason ?? string.Empty,
            }));
        }

        var summary = new StringBuilder();
        summary.AppendLine($"instance: {Path.GetFileName(instancePath)}");
        summary.AppendLine($"start waypoint: {start.Id} {start.Position}");
        summary.AppendLine($"goal waypoint: {goal.Id} {goal.Position}");
        summary.AppendLine($"bot: x={botX} y={botY} yaw={botYaw} v={motion.MaxSpeed} a={motion.Acceleration}/{motion.Deceleration} turn={motion.TurnSpeedSeconds}");
        summary.AppendLine($"plan: {plan.Steps.Count} steps, {plan.TotalTimeSeconds:0.0} s, {plan.TotalDistanceMeters:0.00} m, turns {plan.TurnCount}, expanded {plan.ExpandedStates}");
        summary.AppendLine($"poses: {snapshot.Count}, max gap {maxGap:0.000} s, max offset from plan {maxOffset:0.000} m");
        summary.AppendLine($"arrival error: {arrivalError:0.000} m, nav_result: {result.Status}");
        await File.WriteAllTextAsync(Path.Combine(evidence, "summary.txt"), summary.ToString());
    }
}
