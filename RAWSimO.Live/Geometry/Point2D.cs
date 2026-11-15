using System.Text.Json;
using System.Text.Json.Serialization;

namespace RAWSimO.Live.Geometry;

/// <summary>Two-dimensional point or vector in metres.</summary>
[JsonConverter(typeof(Point2DJsonConverter))]
public readonly record struct Point2D(double X, double Y)
{
    /// <summary>Euclidean length of the vector.</summary>
    public double Length => Math.Sqrt(X * X + Y * Y);

    /// <summary>Returns the sum of two points.</summary>
    public static Point2D operator +(Point2D a, Point2D b) => new(a.X + b.X, a.Y + b.Y);

    /// <summary>Returns the difference of two points.</summary>
    public static Point2D operator -(Point2D a, Point2D b) => new(a.X - b.X, a.Y - b.Y);

    /// <summary>Returns the Euclidean distance between two points.</summary>
    public static double Distance(Point2D a, Point2D b) => (a - b).Length;

    /// <inheritdoc />
    public override string ToString() => $"({X:0.######}, {Y:0.######})";
}

/// <summary>Pose in a two-dimensional plane: position in metres and yaw in radians.</summary>
public readonly record struct Pose2D(double X, double Y, double Yaw)
{
    /// <summary>Position part of the pose.</summary>
    public Point2D Position => new(X, Y);

    /// <summary>Creates a pose from a position and a yaw angle.</summary>
    public Pose2D(Point2D position, double yaw) : this(position.X, position.Y, yaw) { }
}

/// <summary>Angle normalisation and difference helpers shared by the live pipeline.</summary>
public static class AngleMath
{
    /// <summary>Normalises an angle to the interval (-pi, pi].</summary>
    public static double Normalize(double radians)
    {
        if (!double.IsFinite(radians)) return double.NaN;
        while (radians <= -Math.PI) radians += 2.0 * Math.PI;
        while (radians > Math.PI) radians -= 2.0 * Math.PI;
        return radians;
    }

    /// <summary>Normalises an angle in degrees to [0, 360).</summary>
    public static double NormalizeDegrees(double degrees)
    {
        if (!double.IsFinite(degrees)) return double.NaN;
        var value = degrees % 360.0;
        if (value < 0.0) value += 360.0;
        return value;
    }

    /// <summary>Returns the signed shortest rotation from <paramref name="from"/> to <paramref name="to"/>.</summary>
    public static double ShortestDifference(double from, double to) => Normalize(to - from);
}

/// <summary>Serialises <see cref="Point2D"/> as the compact [x, y] form used by the bridge protocol.</summary>
public sealed class Point2DJsonConverter : JsonConverter<Point2D>
{
    /// <inheritdoc />
    public override Point2D Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.StartArray)
        {
            reader.Read();
            var x = reader.GetDouble();
            reader.Read();
            var y = reader.GetDouble();
            reader.Read();
            if (reader.TokenType != JsonTokenType.EndArray)
                throw new JsonException("A point must contain exactly two numbers.");
            return new Point2D(x, y);
        }

        if (reader.TokenType == JsonTokenType.StartObject)
        {
            double? x = null, y = null;
            while (reader.Read() && reader.TokenType != JsonTokenType.EndObject)
            {
                if (reader.TokenType != JsonTokenType.PropertyName) continue;
                var name = reader.GetString();
                reader.Read();
                if (string.Equals(name, "x", StringComparison.OrdinalIgnoreCase)) x = reader.GetDouble();
                else if (string.Equals(name, "y", StringComparison.OrdinalIgnoreCase)) y = reader.GetDouble();
                else reader.Skip();
            }
            if (x is null || y is null) throw new JsonException("A point object needs x and y.");
            return new Point2D(x.Value, y.Value);
        }

        throw new JsonException("A point must be [x, y] or {x, y}.");
    }

    /// <inheritdoc />
    public override void Write(Utf8JsonWriter writer, Point2D value, JsonSerializerOptions options)
    {
        writer.WriteStartArray();
        writer.WriteNumberValue(value.X);
        writer.WriteNumberValue(value.Y);
        writer.WriteEndArray();
    }
}
