using RAWSimO.Live.Tracking;
using Xunit;

#nullable enable

namespace RAWSimO.Live.Tests;

/// <summary>
/// The multi-vehicle display must give every vehicle its own colour. These tests pin the
/// assignment rule: distinct while connected, reusable after a disconnect, deterministic.
/// </summary>
public class FleetColorAllocatorTests
{
    [Fact]
    public void TwoVehiclesGetDifferentColors()
    {
        var allocator = new FleetColorAllocator();
        var first = allocator.Acquire();
        var second = allocator.Acquire();

        Assert.NotEqual(first, second);
        Assert.NotEqual(FleetColorAllocator.ColorOf(first), FleetColorAllocator.ColorOf(second));
        Assert.Equal(2, allocator.Count);
    }

    [Fact]
    public void TenVehiclesAllGetDistinctColors()
    {
        var allocator = new FleetColorAllocator();
        var colors = new HashSet<(byte, byte, byte)>();
        for (var i = 0; i < 10; i++)
            colors.Add(FleetColorAllocator.ColorOf(allocator.Acquire()));

        Assert.Equal(10, colors.Count);
    }

    [Fact]
    public void ReleasedSlotIsReused()
    {
        var allocator = new FleetColorAllocator();
        var first = allocator.Acquire();
        var second = allocator.Acquire();
        allocator.Release(first);

        Assert.Equal(1, allocator.Count);
        Assert.Equal(first, allocator.Acquire());
        Assert.NotEqual(second, first);
    }

    [Fact]
    public void ClearFreesEverySlot()
    {
        var allocator = new FleetColorAllocator();
        allocator.Acquire();
        allocator.Acquire();
        allocator.Clear();

        Assert.Equal(0, allocator.Count);
        Assert.Equal(0, allocator.Acquire());
    }

    [Fact]
    public void ColorOfIsDeterministic()
    {
        for (var slot = 0; slot < 40; slot++)
            Assert.Equal(FleetColorAllocator.ColorOf(slot), FleetColorAllocator.ColorOf(slot));
    }

    [Fact]
    public void ColorsPastThePaletteStayDistinct()
    {
        // The palette is eight entries; the derived colours beyond it must not collide either.
        var colors = new HashSet<(byte, byte, byte)>();
        for (var slot = 0; slot < 40; slot++)
            colors.Add(FleetColorAllocator.ColorOf(slot));

        Assert.Equal(40, colors.Count);
    }

    [Fact]
    public void NegativeSlotIsRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => FleetColorAllocator.ColorOf(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new FleetColorAllocator().Release(-1));
    }

    [Fact]
    public void DefaultPaletteHasEightEntries()
    {
        Assert.Equal(8, FleetColorAllocator.DefaultPalette.Count);
    }
}
