using System.Collections;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using RAWSimO.Live.Protocol;

namespace RAWSimO.Live.Tracking;

/// <summary>A recorded observation at its original relative receipt time.</summary>
public sealed record FleetReplayFrame(double Seconds, FleetFrameMessage Frame);

/// <summary>Read-only recording. It has no transport or command sender.</summary>
public sealed class FleetReplay : IDisposable
{
    private const int MaximumLineBytes = LiveMessageParser.MaxRecordedLineBytes;
    private const int MaximumFrames = 1_000_000;
    private const long MaximumFileBytes = 8L * 1024 * 1024 * 1024;
    private static readonly Encoding Utf8 = new UTF8Encoding(false, true);
    private readonly IDisposable? _owner;
    public FleetHelloMessage Hello { get; }
    public IReadOnlyList<FleetReplayFrame> Frames { get; }
    public double Duration { get; }

    private FleetReplay(FleetHelloMessage hello, IReadOnlyList<FleetReplayFrame> frames,
        double duration, IDisposable? owner = null)
    {
        Hello = hello;
        Frames = frames;
        Duration = duration;
        _owner = owner;
    }

    /// <summary>
    /// Validate a UTF-8 recording once, retaining only a bounded file index.
    /// Frames are read on demand from the original file; writers are excluded while it is open.
    /// </summary>
    public static FleetReplay Open(string path, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,
            64 * 1024, FileOptions.RandomAccess);
        try
        {
            if (file.Length > MaximumFileBytes)
                throw new FormatException("Recording exceeds the 8 GiB file limit.");
            var state = new RecordingState();
            var entries = new List<FileEntry>();
            foreach (var line in ReadLines(file))
            {
                cancellationToken.ThrowIfCancellationRequested();
                var observed = state.Accept(line.Text);
                if (observed is null) continue;
                if (entries.Count >= MaximumFrames)
                    throw new FormatException("Recording exceeds the indexed frame limit.");
                // Hash normalized UTF-8 text so any later change to an indexed observation
                // is rejected even on filesystems that do not enforce sharing flags.
                entries.Add(new FileEntry(line.Offset, line.Length, observed.Seconds,
                    SHA256.HashData(Utf8.GetBytes(line.Text))));
            }
            state.Complete();
            var frames = new FileFrames(file, entries);
            return new FleetReplay(state.Hello!, frames, entries[^1].Seconds, frames);
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    /// <summary>Accept raw hello/frame JSONL or owner event logs with an endpoint hello.</summary>
    public static FleetReplay Load(TextReader reader)
    {
        var state = new RecordingState();
        var frames = new List<FleetReplayFrame>();
        long characters = 0;
        while (reader.ReadLine() is { } line)
        {
            characters += line.Length;
            if (characters > 64 * 1024 * 1024 || Utf8.GetByteCount(line) > MaximumLineBytes)
                throw new FormatException("Recording exceeds the replay size limit.");
            var observed = state.Accept(line.TrimStart('\uFEFF'));
            if (observed is null) continue;
            if (frames.Count >= 100000)
                throw new FormatException("Recording exceeds the in-memory frame limit.");
            frames.Add(observed);
        }
        state.Complete();
        return new FleetReplay(state.Hello!, frames, frames[^1].Seconds);
    }

    private sealed class RecordingState
    {
        public FleetHelloMessage? Hello { get; private set; }
        private double? _first;
        private double _previous = double.NegativeInfinity;
        private FleetRunIdentity? _identity;

        public FleetReplayFrame? Accept(string line)
        {
            if (string.IsNullOrWhiteSpace(line)) return null;
            var (parsed, wall) = ParseLine(line);
            if (parsed is FleetHelloMessage nextHello)
            {
                if (Hello is not null) throw new FormatException("A replay must contain exactly one run hello.");
                Hello = nextHello;
                _identity = new FleetRunIdentity(Hello.Control);
            }
            else if (parsed is FleetFrameMessage frame)
            {
                if (Hello is null) throw new FormatException("Recording needs its original network hello before frames.");
                _identity!.Observe(frame);
                var timestamp = wall ?? frame.T;
                if (!double.IsFinite(timestamp) || timestamp < _previous)
                    throw new FormatException("Recording timestamps must be finite and monotonic.");
                _previous = timestamp;
                _first ??= timestamp;
                var seconds = timestamp - _first.Value;
                if (!double.IsFinite(seconds)) throw new FormatException("Recording duration must be finite.");
                return new FleetReplayFrame(seconds, frame);
            }
            return null;
        }

        public void Complete()
        {
            if (Hello is null || _first is null)
                throw new FormatException("Recording has no network and observed frames.");
        }
    }

    private static (LiveMessage? Message, double? Wall) ParseLine(string line)
    {
        using var document = JsonDocument.Parse(line);
        var root = document.RootElement;
        var message = root;
        double? wall = null;
        if (root.TryGetProperty("kind", out var kind))
        {
            if (kind.GetString() == "operator_endpoint") message = root.GetProperty("hello");
            else if (kind.GetString() == "frame")
            {
                message = root.GetProperty("value");
                wall = root.GetProperty("wall").GetDouble();
            }
            else return (null, null); // Historical requests and receipts never become commands.
        }
        return (LiveMessageParser.ParseRecording(message.GetRawText()), wall);
    }

    private sealed record FileLine(long Offset, int Length, string Text);
    private sealed record FileEntry(long Offset, int Length, double Seconds, byte[] Digest);

    private static IEnumerable<FileLine> ReadLines(FileStream file)
    {
        var buffer = new byte[64 * 1024];
        using var line = new MemoryStream();
        long offset = 0;
        int count;
        while ((count = file.Read(buffer, 0, buffer.Length)) != 0)
        {
            int start = 0;
            while (start < count)
            {
                var end = Array.IndexOf(buffer, (byte)'\n', start, count - start);
                var length = (end < 0 ? count : end) - start;
                if (line.Length + length > MaximumLineBytes)
                    throw new FormatException("Recording line exceeds the 4 MiB limit.");
                line.Write(buffer, start, length);
                if (end < 0) break;
                yield return new FileLine(offset, (int)line.Length,
                    Decode(line.GetBuffer(), (int)line.Length, offset));
                offset += line.Length + 1;
                line.SetLength(0);
                start = end + 1;
            }
        }
        if (line.Length != 0)
            yield return new FileLine(offset, (int)line.Length,
                Decode(line.GetBuffer(), (int)line.Length, offset));
    }

    private static string Decode(byte[] bytes, int length, long offset)
    {
        var text = Utf8.GetString(bytes, 0, length);
        return offset == 0 ? text.TrimStart('\uFEFF') : text;
    }

    private sealed class FileFrames : IReadOnlyList<FleetReplayFrame>, IDisposable
    {
        private readonly FileStream _file;
        private readonly List<FileEntry> _entries;
        private int _cachedIndex = -1;
        private FleetReplayFrame? _cachedFrame;
        private bool _disposed;
        public int Count => _entries.Count;

        public FileFrames(FileStream file, List<FileEntry> entries)
        {
            _file = file;
            _entries = entries;
        }

        public double SecondsAt(int index) => _entries[index].Seconds;

        public FleetReplayFrame this[int index]
        {
            get
            {
                lock (_file)
                {
                    if (_disposed) throw new ObjectDisposedException(nameof(FleetReplay));
                    if (index == _cachedIndex) return _cachedFrame!;
                    var entry = _entries[index];
                    var bytes = new byte[entry.Length];
                    int read = 0;
                    while (read < bytes.Length)
                    {
                        var count = RandomAccess.Read(_file.SafeFileHandle, bytes.AsSpan(read), entry.Offset + read);
                        if (count == 0) throw new IOException("Recording was truncated after indexing.");
                        read += count;
                    }
                    string text;
                    try { text = Decode(bytes, bytes.Length, entry.Offset); }
                    catch (DecoderFallbackException error)
                    { throw new IOException("Recording encoding changed after indexing.", error); }
                    if (!SHA256.HashData(Utf8.GetBytes(text)).AsSpan().SequenceEqual(entry.Digest))
                        throw new IOException("Recording changed after indexing.");
                    var (parsed, _) = ParseLine(text);
                    if (parsed is not FleetFrameMessage frame)
                        throw new IOException("Indexed observation is no longer a frame.");
                    _cachedIndex = index;
                    return _cachedFrame = new FleetReplayFrame(entry.Seconds, frame);
                }
            }
        }

        public IEnumerator<FleetReplayFrame> GetEnumerator()
        {
            for (int i = 0; i < Count; i++) yield return this[i];
        }
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
        public void Dispose()
        {
            lock (_file)
            {
                _disposed = true;
                _cachedFrame = null;
                _file.Dispose();
            }
        }
    }

    /// <summary>Last observed frame at this time; no interpolation over feedback loss.</summary>
    public FleetReplayFrame At(double seconds)
    {
        if (!double.IsFinite(seconds)) throw new ArgumentOutOfRangeException(nameof(seconds));
        int lo = 0, hi = Frames.Count - 1;
        while (lo < hi)
        {
            var mid = (lo + hi + 1) / 2;
            var time = Frames is FileFrames indexed ? indexed.SecondsAt(mid) : Frames[mid].Seconds;
            if (time <= seconds) lo = mid;
            else hi = mid - 1;
        }
        return Frames[lo];
    }

    /// <summary>Release an indexed recording's file handle. In-memory recordings own no reader.</summary>
    public void Dispose() => _owner?.Dispose();
}
