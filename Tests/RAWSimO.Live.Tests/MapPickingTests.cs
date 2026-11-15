using RAWSimO.Live.Geometry;
using RAWSimO.Live.Planning;
using Xunit;

namespace RAWSimO.Live.Tests;

public class MapPickingTests
{
    [Fact]
    public void CanvasOriginMapsToTopLeftOfMap()
    {
        var point = MapPicking.CanvasToMap(0, 0, 1000, 1000);
        Assert.Equal(0.0, point.X, 12);
        Assert.Equal(25.0, point.Y, 12);
    }

    [Fact]
    public void CanvasPixelsConvertAtFortyPerMetre()
    {
        var point = MapPicking.CanvasToMap(40, 40, 1000, 1000);
        Assert.Equal(1.0, point.X, 12);
        Assert.Equal(24.0, point.Y, 12);
    }

    [Fact]
    public void NearestIndexPicksClosestWaypoint()
    {
        var positions = new[]
        {
            new Point2D(0.5, 0.5),
            new Point2D(1.5, 0.5),
            new Point2D(2.5, 0.5),
        };
        Assert.Equal(1, MapPicking.NearestIndex(positions, new Point2D(1.6, 0.6)));
        Assert.Equal(2, MapPicking.NearestIndex(positions, new Point2D(3.0, 0.5)));
    }

    [Fact]
    public void NearestIndexReturnsMinusOneForEmptyInput()
    {
        Assert.Equal(-1, MapPicking.NearestIndex(Array.Empty<Point2D>(), new Point2D(0, 0)));
    }
}
