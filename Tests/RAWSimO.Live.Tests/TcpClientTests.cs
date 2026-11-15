using System.Net;
using System.Net.Sockets;
using System.Text;
using RAWSimO.Live.Geometry;
using RAWSimO.Live.Networking;
using RAWSimO.Live.Protocol;
using Xunit;

namespace RAWSimO.Live.Tests;

public class TcpClientTests
{
    [Fact]
    public async Task ReceivesHelloPoseAndSendsNav()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        try
        {
            var port = ((IPEndPoint)listener.LocalEndpoint).Port;
            await using var client = new LiveTcpClient(new LiveClientOptions
            {
                Host = "127.0.0.1",
                Port = port,
                ConnectTimeout = TimeSpan.FromSeconds(2),
                HelloTimeout = TimeSpan.FromSeconds(2),
            });
            var helloTask = new TaskCompletionSource<HelloMessage>(TaskCreationOptions.RunContinuationsAsynchronously);
            var poseTask = new TaskCompletionSource<PoseMessage>(TaskCreationOptions.RunContinuationsAsynchronously);
            client.HelloReceived += hello => helloTask.TrySetResult(hello);
            client.MessageReceived += message =>
            {
                if (message is PoseMessage pose) poseTask.TrySetResult(pose);
            };
            await client.StartAsync();

            using var server = await listener.AcceptTcpClientAsync();
            using var stream = server.GetStream();
            using var writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true };
            await writer.WriteLineAsync("{\"type\":\"hello\",\"vehicle\":\"agv1\",\"proto\":0,\"frame\":\"chassis\"}");
            var hello = await helloTask.Task.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.Equal("agv1", hello.Vehicle);

            await writer.WriteLineAsync("{\"type\":\"pose\",\"vehicle\":\"agv1\",\"seq\":7,\"t\":12.5,\"x\":1.2,\"y\":2.3,\"yaw\":0.4}");
            var pose = await poseTask.Task.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.Equal(7, pose.Seq);
            Assert.Equal(1.2, pose.X, 12);

            await client.SendAsync(new NavCommandMessage("s1-g1", new[] { new Point2D(0, 0), new Point2D(1, 1) }, 0.2));
            using var reader = new StreamReader(stream, new UTF8Encoding(false), false, 4096, leaveOpen: true);
            var line = await reader.ReadLineAsync().WaitAsync(TimeSpan.FromSeconds(5));
            var nav = Assert.IsType<NavCommandMessage>(LiveMessageParser.Parse(line!));
            Assert.Equal("s1-g1", nav.Goal);
            Assert.Equal(2, nav.Path.Count);
            await client.StopAsync();
        }
        finally
        {
            listener.Stop();
        }
    }

    [Fact]
    public async Task ReconnectsAfterServerCloses()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        try
        {
            var port = ((IPEndPoint)listener.LocalEndpoint).Port;
            await using var client = new LiveTcpClient(new LiveClientOptions
            {
                Host = "127.0.0.1",
                Port = port,
                ConnectTimeout = TimeSpan.FromSeconds(2),
                HelloTimeout = TimeSpan.FromSeconds(2),
            });
            var secondHello = new TaskCompletionSource<HelloMessage>(TaskCreationOptions.RunContinuationsAsynchronously);
            var count = 0;
            client.HelloReceived += hello =>
            {
                if (Interlocked.Increment(ref count) == 2) secondHello.TrySetResult(hello);
            };
            await client.StartAsync();

            using (var first = await listener.AcceptTcpClientAsync())
            {
                using var firstStream = first.GetStream();
                using var firstWriter = new StreamWriter(firstStream, new UTF8Encoding(false)) { AutoFlush = true };
                await firstWriter.WriteLineAsync("{\"type\":\"hello\",\"vehicle\":\"agv1\",\"proto\":0,\"frame\":\"chassis\"}");
                await Task.Delay(100);
            }

            using var second = await listener.AcceptTcpClientAsync();
            using var secondStream = second.GetStream();
            using var secondWriter = new StreamWriter(secondStream, new UTF8Encoding(false)) { AutoFlush = true };
            await secondWriter.WriteLineAsync("{\"type\":\"hello\",\"vehicle\":\"agv1\",\"proto\":0,\"frame\":\"chassis\"}");
            var hello = await secondHello.Task.WaitAsync(TimeSpan.FromSeconds(8));
            Assert.Equal("agv1", hello.Vehicle);
            Assert.Equal(2, count);
            await client.StopAsync();
        }
        finally
        {
            listener.Stop();
        }
    }
}
