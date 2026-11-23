using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using RAWSimO.Live.Geometry;
using RAWSimO.Live.Networking;
using RAWSimO.Live.Planning;
using RAWSimO.Live.Protocol;
using Xunit;

#nullable enable

namespace RAWSimO.Live.Tests;

/// <summary>
/// Drives the real Python bridge over TCP without any window or keyboard input.
/// The bridge is started as a child process and spoken to with the production client.
/// </summary>
public class BridgeEndToEndTests
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

    private sealed class Bridge : IDisposable
    {
        private readonly Process _process;

        private Bridge(Process process) => _process = process;

        public static Bridge Start(string root, int port, params string[] extra)
        {
            var script = Path.Combine(root, "tools", "live_bridge", "live_bridge.py");
            // Linux CI 里解释器通常叫 python3，Windows 上叫 python。
            // 允许用 AMR_PYTHON 显式指定，否则依次尝试。
            var python = Environment.GetEnvironmentVariable("AMR_PYTHON")
                         ?? (OperatingSystem.IsWindows() ? "python" : "python3");
            var info = new ProcessStartInfo(python)
            {
                WorkingDirectory = root,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            info.ArgumentList.Add(script);
            info.ArgumentList.Add("--mode");
            info.ArgumentList.Add("sim");
            info.ArgumentList.Add("--rate");
            info.ArgumentList.Add("20");
            info.ArgumentList.Add("--port");
            info.ArgumentList.Add(port.ToString());
            info.ArgumentList.Add("--start");
            info.ArgumentList.Add("0.9");
            info.ArgumentList.Add("1.35");
            info.ArgumentList.Add("0");
            // The GUI pings every 0.2 s; these tests do not, so the bridge watchdog must stay off.
            info.ArgumentList.Add("--watchdog");
            info.ArgumentList.Add("0");
            foreach (var arg in extra) info.ArgumentList.Add(arg);
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

    private static async Task<(LiveTcpClient Client, List<LiveMessage> Log, TaskCompletionSource<HelloMessage> Hello)> ConnectAsync(int port)
    {
        var client = new LiveTcpClient(new LiveClientOptions { Host = "127.0.0.1", Port = port });
        var log = new List<LiveMessage>();
        var hello = new TaskCompletionSource<HelloMessage>(TaskCreationOptions.RunContinuationsAsynchronously);
        client.MessageReceived += message =>
        {
            lock (log) log.Add(message);
            if (message is HelloMessage h) hello.TrySetResult(h);
        };
        client.HelloReceived += h => hello.TrySetResult(h);
        await client.StartAsync();
        return (client, log, hello);
    }

    private static async Task<T> WaitForAsync<T>(Func<T?> probe, int timeoutMs = 15000) where T : class
    {
        var deadline = DateTime.UtcNow.AddMilliseconds(timeoutMs);
        while (DateTime.UtcNow < deadline)
        {
            var value = probe();
            if (value != null) return value;
            await Task.Delay(50);
        }
        throw new TimeoutException($"no {typeof(T).Name} within {timeoutMs} ms");
    }

    [Fact]
    public async Task SendsNavGoalAndReceivesSuccessFromRealBridge()
    {
        var root = RepoRoot();
        var port = FreePort();
        using var bridge = Bridge.Start(root, port);
        await Task.Delay(1200);
        var (client, log, hello) = await ConnectAsync(port);
        await using (client)
        {
            var greeting = await hello.Task.WaitAsync(TimeSpan.FromSeconds(10));
            Assert.Equal(1, greeting.Proto);
            Assert.Contains("nav", greeting.Caps ?? Array.Empty<string>());

            var path = new[] { new Point2D(1.0, 1.35), new Point2D(1.2, 1.35) };
            await client.SendAsync(new NavCommandMessage("e2e-1", path, 0.3));

            var result = await WaitForAsync(() =>
            {
                lock (log) return log.OfType<NavResultMessage>().FirstOrDefault(r => r.Goal == "e2e-1");
            });
            Assert.Equal("succeeded", result.Status);
        }
    }

    [Fact]
    public async Task CancelStopsARunningGoalAtTheBridge()
    {
        var root = RepoRoot();
        var port = FreePort();
        using var bridge = Bridge.Start(root, port);
        await Task.Delay(1200);
        var (client, log, hello) = await ConnectAsync(port);
        await using (client)
        {
            await hello.Task.WaitAsync(TimeSpan.FromSeconds(10));
            var path = new[] { new Point2D(2.4, 2.4), new Point2D(2.6, 2.6) };
            await client.SendAsync(new NavCommandMessage("e2e-cancel", path, 0.2));
            await Task.Delay(300);
            await client.SendAsync(new CancelCommandMessage("e2e-cancel"));

            var result = await WaitForAsync(() =>
            {
                lock (log) return log.OfType<NavResultMessage>().FirstOrDefault(r => r.Goal == "e2e-cancel");
            });
            Assert.Equal("canceled", result.Status);
        }
    }

    [Fact]
    public async Task NewGoalReplacesTheRunningOne()
    {
        var root = RepoRoot();
        var port = FreePort();
        using var bridge = Bridge.Start(root, port);
        await Task.Delay(1200);
        var (client, log, hello) = await ConnectAsync(port);
        await using (client)
        {
            await hello.Task.WaitAsync(TimeSpan.FromSeconds(10));
            await client.SendAsync(new NavCommandMessage("first", new[] { new Point2D(2.5, 2.5) }, 0.2));
            await Task.Delay(300);
            await client.SendAsync(new NavCommandMessage("second", new[] { new Point2D(1.0, 1.2) }, 0.3));

            var replaced = await WaitForAsync(() =>
            {
                lock (log) return log.OfType<NavResultMessage>().FirstOrDefault(r => r.Goal == "first");
            });
            Assert.Equal("canceled", replaced.Status);
            Assert.Contains("second", replaced.Reason);

            var done = await WaitForAsync(() =>
            {
                lock (log) return log.OfType<NavResultMessage>().FirstOrDefault(r => r.Goal == "second");
            });
            Assert.Equal("succeeded", done.Status);
        }
    }

    [Fact]
    public async Task NavExecutionTracksTheBridgeResultEndToEnd()
    {
        var root = RepoRoot();
        var port = FreePort();
        using var bridge = Bridge.Start(root, port);
        await Task.Delay(1200);
        var (client, log, hello) = await ConnectAsync(port);
        await using (client)
        {
            await hello.Task.WaitAsync(TimeSpan.FromSeconds(10));
            var nav = new NavExecution();
            nav.Start("e2e-state", 20.0, 0.0);
            Assert.True(nav.IsActive);
            await client.SendAsync(new NavCommandMessage("e2e-state", new[] { new Point2D(1.1, 1.35) }, 0.3));
            var result = await WaitForAsync(() =>
            {
                lock (log) return log.OfType<NavResultMessage>().FirstOrDefault(r => r.Goal == "e2e-state");
            });
            Assert.True(nav.OnResult(result.Goal, result.Status, result.Reason));
            Assert.Equal(NavExecutionState.Succeeded, nav.State);
            Assert.Equal("succeeded", nav.ResultStatus);
        }
    }
}
