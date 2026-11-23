using System;
using System.Collections.Generic;

namespace RAWSimO.Live.Tracking;

/// <summary>
/// Hands out a distinct colour per vehicle in a multi-vehicle display.
///
/// The allocator is deliberately free of WPF types so the assignment rule can be unit tested:
/// the lowest free palette slot wins, slots are released when a vehicle disconnects, and beyond
/// the palette the colour is derived deterministically from the slot index so two vehicles never
/// silently share one colour.
/// </summary>
public sealed class FleetColorAllocator
{
    /// <summary>Default palette: eight saturated colours that stay distinguishable on white.</summary>
    public static readonly IReadOnlyList<(byte R, byte G, byte B)> DefaultPalette = new[]
    {
        ((byte)0x2E, (byte)0x7D, (byte)0x32), // green
        ((byte)0x15, (byte)0x65, (byte)0xC0), // blue
        ((byte)0x6A, (byte)0x1B, (byte)0x9A), // purple
        ((byte)0xEF, (byte)0x6C, (byte)0x00), // orange
        ((byte)0x00, (byte)0x83, (byte)0x8F), // teal
        ((byte)0xAD, (byte)0x14, (byte)0x57), // magenta
        ((byte)0x55, (byte)0x8B, (byte)0x2F), // olive
        ((byte)0x4E, (byte)0x34, (byte)0x2E), // brown
    };

    private readonly HashSet<int> _used = new();

    /// <summary>Number of colours currently handed out.</summary>
    public int Count => _used.Count;

    /// <summary>Reserves the lowest free slot and returns it.</summary>
    public int Acquire()
    {
        var slot = 0;
        while (_used.Contains(slot)) slot++;
        _used.Add(slot);
        return slot;
    }

    /// <summary>Releases a slot so a later vehicle can reuse the colour.</summary>
    public void Release(int slot)
    {
        if (slot < 0) throw new ArgumentOutOfRangeException(nameof(slot));
        _used.Remove(slot);
    }

    /// <summary>Releases every slot.</summary>
    public void Clear() => _used.Clear();

    /// <summary>Returns the colour for a slot; deterministic for every non-negative index.</summary>
    public static (byte R, byte G, byte B) ColorOf(int slot)
    {
        if (slot < 0) throw new ArgumentOutOfRangeException(nameof(slot));
        if (slot < DefaultPalette.Count) return DefaultPalette[slot];
        // Past the palette: spread the hue deterministically. The three strides are coprime with
        // 256, so consecutive overflow slots differ in every channel.
        var n = slot - DefaultPalette.Count;
        return ((byte)(40 + n * 37 % 200), (byte)(90 + n * 71 % 150), (byte)(60 + n * 53 % 180));
    }
}
