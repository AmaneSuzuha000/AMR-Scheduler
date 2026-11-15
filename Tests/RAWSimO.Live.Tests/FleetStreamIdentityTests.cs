using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json.Nodes;
using RAWSimO.Live.Networking;
using RAWSimO.Live.Protocol;
using Xunit;

namespace RAWSimO.Live.Tests;

public class FleetStreamIdentityTests
{
    private static JsonObject Control(string run = "r") => new()
    {
        ["session_id"] = "s", ["scene_id"] = "车间", ["run_id"] = run, ["paused"] = false,
        ["routes"] = new JsonObject { ["R1"] = run + ":R1:1" }
    };
    private static JsonObject Hello(bool control = true, string run = "r") => new()
    {
        ["type"] = "hello", ["network"] = new JsonObject { ["nodes"] = new JsonArray(), ["edges"] = new JsonArray() },
        ["meta"] = new JsonObject(), ["control"] = control ? Control(run) : null
    };
    private static JsonObject Frame(string run = "r") => new()
    {
        ["type"] = "frame", ["schema"] = "1.0", ["step"] = 1, ["t"] = 1,
        ["agvs"] = new JsonArray(new JsonObject
        {
            ["agv_id"] = "R1", ["state"] = "blocked", ["pose"] = new JsonObject
            {
                ["agv_id"] = "R1", ["source"] = "localized", ["available"] = false
            }
        }),
        ["execution"] = new JsonObject
        {
            ["robot_id"] = "R1", ["scene_id"] = "车间", ["run_id"] = run, ["fault"] = null
        }
    };

    [Theory]
    [InlineData("scene_id", true, false)]
    [InlineData("run_id", true, false)]
    [InlineData("scene_id", false, false)]
    [InlineData("run_id", false, false)]
    [InlineData("scene_id", true, true)]
    [InlineData("run_id", true, true)]
    [InlineData("scene_id", false, true)]
    [InlineData("run_id", false, true)]
    public async Task ForeignStateIsRejectedBeforeAnyConsumerEvent(string field, bool helloControl, bool receipt)
    {
        using var cancel = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var changed = receipt ? new JsonObject
        {
            ["type"] = "control_result", ["request_id"] = "id", ["accepted"] = false, ["control"] = Control()
        } : Frame();
        changed[receipt ? "control" : "execution"]![field] = "foreign";
        var serve = Task.Run(async () =>
        {
            try
            {
                using var socket = await listener.AcceptTcpClientAsync(cancel.Token);
                using var stream = socket.GetStream();
                var content = string.Join("\n", new[] { Hello(helloControl).ToJsonString(), Frame().ToJsonString(),
                    changed.ToJsonString(), "{\"type\":\"ping\",\"t1\":1}" }) + "\n";
                await stream.WriteAsync(Encoding.UTF8.GetBytes(content), cancel.Token);
                await Task.Delay(Timeout.Infinite, cancel.Token);
            }
            catch (OperationCanceledException) { }
            finally { listener.Stop(); }
        });
        await using var client = new LiveTcpClient(new LiveClientOptions { Port = ((IPEndPoint)listener.LocalEndpoint).Port });
        var messages = new ConcurrentQueue<LiveMessage>();
        var errors = new ConcurrentQueue<Exception>();
        var completed = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        client.MessageReceived += message => { messages.Enqueue(message); if (message is PingMessage) completed.TrySetResult(); };
        client.Error += error => { errors.Enqueue(error); completed.TrySetResult(); };
        try
        {
            await client.StartAsync(cancel.Token);
            await completed.Task.WaitAsync(TimeSpan.FromSeconds(5));
            await client.StopAsync();
            Assert.Single(messages.OfType<FleetFrameMessage>());
            Assert.Empty(messages.OfType<FleetControlResultMessage>());
            Assert.Contains(errors, error => error is FormatException && error.Message.Contains("scene/run"));
        }
        finally { cancel.Cancel(); await serve; }
    }

    [Fact]
    public async Task ReconnectRequiresNewHelloAndThenAcceptsTheNewRun()
    {
        using var cancel = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        var serve = Task.Run(async () =>
        {
            try
            {
                using (var first = await listener.AcceptTcpClientAsync(cancel.Token))
                {
                    var content = Hello().ToJsonString() + "\n" + Frame().ToJsonString() + "\n" + Frame("foreign").ToJsonString() + "\n";
                    await first.GetStream().WriteAsync(Encoding.UTF8.GetBytes(content), cancel.Token);
                }
                using var second = await listener.AcceptTcpClientAsync(cancel.Token);
                var next = Hello(run: "next").ToJsonString() + "\n" + Frame("next").ToJsonString() + "\n";
                await second.GetStream().WriteAsync(Encoding.UTF8.GetBytes(next), cancel.Token);
                await Task.Delay(Timeout.Infinite, cancel.Token);
            }
            catch (OperationCanceledException) { }
            finally { listener.Stop(); }
        });
        await using var client = new LiveTcpClient(new LiveClientOptions { Port = port });
        var frames = new ConcurrentQueue<FleetFrameMessage>();
        var hellos = new ConcurrentQueue<FleetHelloMessage>();
        var nextRun = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        client.FleetHelloReceived += hello => hellos.Enqueue(hello);
        client.FleetFrameReceived += frame =>
        {
            frames.Enqueue(frame);
            if (frame.ExecutionIdentity?.RunId == "next") nextRun.TrySetResult();
        };
        try
        {
            await client.StartAsync(cancel.Token);
            await nextRun.Task.WaitAsync(TimeSpan.FromSeconds(5));
            await client.StopAsync();
            Assert.Equal(new[] { "r", "next" }, frames.Select(frame => frame.ExecutionIdentity!.RunId));
            Assert.Equal(2, hellos.Count);
        }
        finally { cancel.Cancel(); await serve; }
    }
}
