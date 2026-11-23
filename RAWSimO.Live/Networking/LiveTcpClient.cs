using System.Net.Sockets;
using System.Text;
using RAWSimO.Live.Protocol;
using RAWSimO.Live.Tracking;

namespace RAWSimO.Live.Networking;

/// <summary>Connection state of <see cref="LiveTcpClient"/>.</summary>
public enum LiveConnectionState
{
    /// <summary>Not connected and not trying.</summary>
    Disconnected,
    /// <summary>TCP connection is being established.</summary>
    Connecting,
    /// <summary>TCP is connected; waiting for the bridge's hello message.</summary>
    WaitingHello,
    /// <summary>Hello received; messages are flowing.</summary>
    Connected,
    /// <summary>Connection was lost and a retry is scheduled.</summary>
    Reconnecting,
}

/// <summary>Options for the live TCP client.</summary>
public sealed class LiveClientOptions
{
    /// <summary>Bridge host.</summary>
    public string Host { get; set; } = "127.0.0.1";

    /// <summary>Bridge port.</summary>
    public int Port { get; set; } = 9300;

    /// <summary>TCP connect timeout.</summary>
    public TimeSpan ConnectTimeout { get; set; } = TimeSpan.FromSeconds(2);

    /// <summary>Time allowed for the bridge's hello after TCP connects.</summary>
    public TimeSpan HelloTimeout { get; set; } = TimeSpan.FromSeconds(2);

    /// <summary>Initial hello and ordinary message limit; at most 64 KiB of UTF-8 bytes.</summary>
    public int MaxLineBytes { get; set; } = LiveMessageParser.MaxLineBytes;

    /// <summary>Observation limit after a valid fleet hello; at most 4 MiB of UTF-8 bytes.</summary>
    public int MaxFleetLineBytes { get; set; } = LiveMessageParser.MaxFleetLineBytes;
}

/// <summary>
/// Asynchronous newline-framed JSON client with automatic reconnect. The
/// receive loop never touches WPF; consumers subscribe to
/// <see cref="MessageReceived"/> and marshal to their own UI thread.
/// </summary>
public sealed class LiveTcpClient : IAsyncDisposable
{
    private readonly LiveClientOptions _options;
    private readonly int _ordinaryLineBytes;
    private readonly int _fleetLineBytes;
    private readonly SemaphoreSlim _sendLock = new(1, 1);
    private readonly object _lifecycleLock = new();
    private CancellationTokenSource? _cts;
    private Task? _loopTask;
    private TcpClient? _client;
    private NetworkStream? _stream;
    private int _state = (int)LiveConnectionState.Disconnected;
    private int _backoffAttempt;

    /// <summary>Creates a client with the supplied options.</summary>
    public LiveTcpClient(LiveClientOptions? options = null)
    {
        _options = options ?? new LiveClientOptions();
        if (_options.MaxLineBytes <= 0 || _options.MaxLineBytes > LiveMessageParser.MaxLineBytes)
            throw new ArgumentOutOfRangeException(nameof(options), "Ordinary limit must be between 1 and 64 KiB.");
        if (_options.MaxFleetLineBytes <= 0 || _options.MaxFleetLineBytes > LiveMessageParser.MaxFleetLineBytes)
            throw new ArgumentOutOfRangeException(nameof(options), "Fleet limit must be between 1 byte and 4 MiB.");
        _ordinaryLineBytes = _options.MaxLineBytes;
        _fleetLineBytes = _options.MaxFleetLineBytes;
    }

    /// <summary>Current connection state.</summary>
    public LiveConnectionState State => (LiveConnectionState)Volatile.Read(ref _state);

    /// <summary>Raised for every parsed message, including the hello.</summary>
    public event Action<LiveMessage>? MessageReceived;

    /// <summary>Raised when a single-vehicle hello is received.</summary>
    public event Action<HelloMessage>? HelloReceived;

    /// <summary>
    /// Raised when the scheduling stream announces its network. A consumer needs this
    /// before it can place any frame: a frame names nodes and edges, not coordinates.
    /// </summary>
    public event Action<FleetHelloMessage>? FleetHelloReceived;

    /// <summary>
    /// Raised for every scheduling frame. Read-only: this channel never accepts
    /// commands, so there is no matching send path.
    /// </summary>
    public event Action<FleetFrameMessage>? FleetFrameReceived;

    /// <summary>Raised whenever the connection state changes.</summary>
    public event Action<LiveConnectionState>? StateChanged;

    /// <summary>Raised for recoverable I/O or parse errors.</summary>
    public event Action<Exception>? Error;

    /// <summary>Starts the connect/receive loop. Calling it twice is a no-op.</summary>
    public Task StartAsync(CancellationToken cancellationToken = default)
    {
        lock (_lifecycleLock)
        {
            if (_loopTask is not null) return Task.CompletedTask;
            _cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            _loopTask = Task.Run(() => RunAsync(_cts.Token));
            return Task.CompletedTask;
        }
    }

    /// <summary>Stops the loop and closes the socket. No reconnect is attempted afterwards.</summary>
    public async Task StopAsync()
    {
        Task? loop;
        lock (_lifecycleLock)
        {
            _cts?.Cancel();
            _client?.Close();
            loop = _loopTask;
            _loopTask = null;
        }
        if (loop is not null)
        {
            try { await loop.ConfigureAwait(false); }
            catch (OperationCanceledException) { }
        }
        _cts?.Dispose();
        _cts = null;
    }

    /// <summary>Sends one message. Throws when the client is not connected.</summary>
    public async Task SendAsync(LiveMessage message, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(message);
        var stream = _stream ?? throw new InvalidOperationException("The live client is not connected.");
        var bytes = Encoding.UTF8.GetBytes(LiveMessageParser.SerializeLine(message));
        await _sendLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await stream.WriteAsync(bytes, cancellationToken).ConfigureAwait(false);
            await stream.FlushAsync(cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _sendLock.Release();
        }
    }

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        await StopAsync().ConfigureAwait(false);
        _sendLock.Dispose();
    }

    private async Task RunAsync(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            SetState(_backoffAttempt == 0 ? LiveConnectionState.Connecting : LiveConnectionState.Reconnecting);
            try
            {
                using var client = new TcpClient { NoDelay = true };
                using var connectCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                connectCts.CancelAfter(_options.ConnectTimeout);
                await client.ConnectAsync(_options.Host, _options.Port, connectCts.Token).ConfigureAwait(false);
                _client = client;
                var stream = client.GetStream();
                _stream = stream;
                var reader = new Utf8LineReader(stream);

                SetState(LiveConnectionState.WaitingHello);
                var helloLine = await ReadLineWithTimeoutAsync(reader, _ordinaryLineBytes,
                    _options.HelloTimeout, cancellationToken)
                    .ConfigureAwait(false);
                if (helloLine is null) throw new IOException("Connection closed before hello.");
                // Both producers open with a message called "hello"; the parser has
                // already decided which one this is by payload shape. Accept either
                // here, otherwise the scheduling stream would fail the handshake and
                // reconnect forever with no visible reason.
                var first = LiveMessageParser.Parse(helloLine);
                var singleHello = first as HelloMessage;
                var fleetHello = first as FleetHelloMessage;
                if (singleHello is null && fleetHello is null)
                    throw new IOException("The first message was not a hello.");
                var fleetIdentity = fleetHello is null ? null : new FleetRunIdentity(fleetHello.Control);
                SetState(LiveConnectionState.Connected);
                _backoffAttempt = 0;
                if (fleetHello is not null) FleetHelloReceived?.Invoke(fleetHello);
                else HelloReceived?.Invoke(singleHello!);
                MessageReceived?.Invoke(first);

                while (!cancellationToken.IsCancellationRequested)
                {
                    var line = await reader.ReadLineAsync(
                        fleetHello is null ? _ordinaryLineBytes : _fleetLineBytes, cancellationToken).ConfigureAwait(false);
                    if (line is null) break;
                    LiveMessage? message;
                    string? error;
                    var parsed = fleetHello is null
                        ? LiveMessageParser.TryParse(line, out message, out error)
                        : LiveMessageParser.TryParseFleetStream(line, _ordinaryLineBytes, out message, out error);
                    if (!parsed)
                    {
                        Error?.Invoke(new FormatException(error));
                        continue;
                    }
                    // A restarted owner must announce its network on a fresh connection.
                    // Reject foreign observations and receipts before any consumer sees them.
                    if (message is not null) fleetIdentity?.Observe(message);
                    if (message is HelloMessage repeatedHello) HelloReceived?.Invoke(repeatedHello);
                    else if (message is FleetHelloMessage repeatedFleet) FleetHelloReceived?.Invoke(repeatedFleet);
                    else if (message is FleetFrameMessage frame) FleetFrameReceived?.Invoke(frame);
                    if (message is not null) MessageReceived?.Invoke(message);
                }
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                Error?.Invoke(ex);
            }
            finally
            {
                _stream = null;
                _client?.Dispose();
                _client = null;
            }

            if (cancellationToken.IsCancellationRequested) break;
            SetState(LiveConnectionState.Reconnecting);
            try { await Task.Delay(NextBackoff(), cancellationToken).ConfigureAwait(false); }
            catch (OperationCanceledException) { break; }
        }
        SetState(LiveConnectionState.Disconnected);
    }

    private static async Task<string?> ReadLineWithTimeoutAsync(Utf8LineReader reader, int maximumBytes,
        TimeSpan timeout, CancellationToken cancellationToken)
    {
        using var helloTimeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        helloTimeout.CancelAfter(timeout);
        try
        {
            return await reader.ReadLineAsync(maximumBytes, helloTimeout.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            throw new IOException("Timed out waiting for the bridge hello.");
        }
    }

    private TimeSpan NextBackoff()
    {
        var exponent = Math.Min(_backoffAttempt, 4);
        _backoffAttempt++;
        var milliseconds = Math.Min(8000.0, 500.0 * Math.Pow(2.0, exponent));
        var jitter = 0.8 + Random.Shared.NextDouble() * 0.4;
        return TimeSpan.FromMilliseconds(milliseconds * jitter);
    }

    private void SetState(LiveConnectionState state)
    {
        var previous = Interlocked.Exchange(ref _state, (int)state);
        if (previous != (int)state)
            StateChanged?.Invoke(state);
    }
}
