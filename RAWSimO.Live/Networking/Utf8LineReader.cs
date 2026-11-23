using System.Text;

namespace RAWSimO.Live.Networking;

/// <summary>Incremental JSONL transport reader; limits bytes before decoding or parsing.</summary>
internal sealed class Utf8LineReader
{
    private static readonly Encoding Utf8 = new UTF8Encoding(false, true);
    private readonly Stream _stream;
    private readonly byte[] _buffer = new byte[4096];
    private int _position;
    private int _count;

    public Utf8LineReader(Stream stream) => _stream = stream;

    public async Task<string?> ReadLineAsync(int maximumBytes, CancellationToken cancellationToken)
    {
        using var line = new MemoryStream();
        while (true)
        {
            if (_position == _count)
            {
                _count = await _stream.ReadAsync(_buffer.AsMemory(), cancellationToken).ConfigureAwait(false);
                _position = 0;
                if (_count == 0)
                    return line.Length == 0 ? null : Decode(line);
            }
            var end = Array.IndexOf(_buffer, (byte)'\n', _position, _count - _position);
            var length = (end < 0 ? _count : end) - _position;
            if (line.Length + length > maximumBytes)
                throw new IOException("Message exceeds the line limit in UTF-8 bytes.");
            line.Write(_buffer, _position, length);
            _position += length;
            if (end >= 0)
            {
                ++_position;
                return Decode(line);
            }
        }
    }

    private static string Decode(MemoryStream line)
    {
        var length = (int)line.Length;
        var bytes = line.GetBuffer();
        if (length > 0 && bytes[length - 1] == '\r') --length;
        return Utf8.GetString(bytes, 0, length);
    }
}
