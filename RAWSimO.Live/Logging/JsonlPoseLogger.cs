using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using RAWSimO.Live.Protocol;

namespace RAWSimO.Live.Logging;

/// <summary>
/// Writes the GUI-side JSONL record stream. Each line has the shape
/// <c>{"wall":...,"mono":...,"dir":"rx","msg":{...}}</c>, matching the
/// replay reader in <c>tools/live_bridge</c>. Every record is flushed
/// immediately so a crash does not lose the last seconds of a run.
/// </summary>
public sealed class JsonlPoseLogger : IDisposable
{
    private readonly object _lock = new();
    private readonly StreamWriter _writer;
    private bool _disposed;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DictionaryKeyPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = false,
    };

    /// <summary>Creates a new log file in <paramref name="directory"/>.</summary>
    public JsonlPoseLogger(string directory, string? prefix = null)
    {
        if (string.IsNullOrWhiteSpace(directory)) throw new ArgumentException("Directory is required.", nameof(directory));
        Directory.CreateDirectory(directory);
        var name = (string.IsNullOrWhiteSpace(prefix) ? "live" : prefix) +
                   "-" + DateTime.Now.ToString("yyyyMMdd-HHmmss-fff") + ".jsonl";
        FilePath = Path.Combine(directory, name);
        _writer = new StreamWriter(new FileStream(FilePath, FileMode.CreateNew, FileAccess.Write, FileShare.Read),
            new UTF8Encoding(false)) { AutoFlush = true };
    }

    /// <summary>Full path of the log file.</summary>
    public string FilePath { get; }

    /// <summary>Writes the first line, a session header containing version, calibration and clock diagnostics.</summary>
    public void WriteSessionHeader(object header) => Write("meta", header);

    /// <summary>Writes one protocol message in the given direction ("rx" or "tx").</summary>
    public void Write(string direction, LiveMessage message) => Write(direction, (object)message);

    /// <summary>Writes one arbitrary payload in the given direction.</summary>
    public void Write(string direction, object message)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(JsonlPoseLogger));
        var record = new
        {
            wall = DateTimeOffset.UtcNow.ToString("O"),
            mono = (double)System.Diagnostics.Stopwatch.GetTimestamp() / System.Diagnostics.Stopwatch.Frequency,
            dir = direction,
            msg = message,
        };
        var line = JsonSerializer.Serialize(record, JsonOptions);
        lock (_lock)
        {
            _writer.WriteLine(line);
            _writer.Flush();
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        lock (_lock) _writer.Dispose();
    }
}
