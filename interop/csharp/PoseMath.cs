// PoseMath.cs -- the ONE implementation of "where on the canvas is this vehicle".
//
// Why this file exists
// --------------------
// There used to be two. "interop/csharp/LiveClient.cs" had PoseToXY and
// "RAWSimO.Live/Tracking/FleetTopology.cs" had Place, and both interpolated a pose
// along an edge. They agreed on the day they were written, and nothing kept them
// agreeing: a change to the clamp in one would silently leave the other drawing
// vehicles at a different place than the tests checked.
//
// PROTOCOLS.md said the fix was to delete PoseToXY and reference RAWSimO.Live. That
// was not doable: interop/csharp is deliberately built with the .NET Framework 4.0
// csc.exe and no SDK, so it cannot reference a net6.0 library. Raising its target
// framework would defeat the reason it exists.
//
// So the rule moved here instead. This file is compiled into both consumers:
//   * interop/csharp   -- listed on the csc command line in interop/LIVE.md
//   * RAWSimO.Live     -- linked by RAWSimO.Live.csproj as Tracking\PoseMath.cs
// One body of source, two binaries. Drift is now impossible rather than untested.
//
// Written in C# 5 on purpose: the .NET Framework 4.0 compiler must accept it, so no
// expression-bodied members, no string interpolation, no out-var, no nullable
// reference annotations. RAWSimO.Live compiles it unchanged.

using System;

namespace AmrInterop
{
    /// <summary>Edge interpolation shared by every consumer of the fleet stream.</summary>
    public static class PoseMath
    {
        /// <summary>The separator the producer writes between the two node ids of an edge.</summary>
        public const string EdgeSeparator = "->";

        /// <summary>
        /// Splits an edge id such as "R0-&gt;R1" into "R0" and "R1".
        /// Returns false -- never throws -- for anything that is not exactly two
        /// non-empty ids joined by <see cref="EdgeSeparator"/>.
        /// </summary>
        public static bool TrySplitEdge(string edge, out string src, out string dst)
        {
            src = string.Empty;
            dst = string.Empty;
            if (edge == null) return false;
            string[] parts = edge.Split(new string[] { EdgeSeparator }, StringSplitOptions.None);
            if (parts.Length != 2) return false;
            if (parts[0].Length == 0 || parts[1].Length == 0) return false;
            src = parts[0];
            dst = parts[1];
            return true;
        }

        /// <summary>
        /// Clamps progress onto 0..1.
        /// A value slightly outside the interval is a producer rounding artefact;
        /// clamping keeps the vehicle on its own edge, which is the truthful drawing.
        /// </summary>
        public static double ClampProgress(double progress)
        {
            if (progress < 0.0) return 0.0;
            if (progress > 1.0) return 1.0;
            return progress;
        }

        /// <summary>Linear interpolation of a point sitting at <paramref name="progress"/> along an edge.</summary>
        public static void Interpolate(double ax, double ay, double bx, double by,
                                       double progress, out double x, out double y)
        {
            double p = ClampProgress(progress);
            x = ax + (bx - ax) * p;
            y = ay + (by - ay) * p;
        }

        /// <summary>Interpolates declared road geometry by arc length, preserving every bend.</summary>
        public static void InterpolatePolyline(double[][] points, double progress,
                                               out double x, out double y)
        {
            if (points == null || points.Length < 2)
                throw new ArgumentException("A road needs at least two points.");
            double total = 0.0;
            double[] lengths = new double[points.Length - 1];
            for (int i = 0; i < points.Length; i++)
            {
                if (points[i] == null || points[i].Length != 2 ||
                    double.IsNaN(points[i][0]) || double.IsInfinity(points[i][0]) ||
                    double.IsNaN(points[i][1]) || double.IsInfinity(points[i][1]))
                    throw new ArgumentException("A road needs finite coordinate pairs.");
                if (i == 0) continue;
                double dx = points[i][0] - points[i - 1][0];
                double dy = points[i][1] - points[i - 1][1];
                double length = Math.Sqrt(dx * dx + dy * dy);
                if (!(length > 0.0) || double.IsInfinity(length))
                    throw new ArgumentException("A road needs nonzero finite segments.");
                lengths[i - 1] = length;
                total += length;
            }
            if (double.IsInfinity(total) || double.IsNaN(progress) || double.IsInfinity(progress))
                throw new ArgumentException("Road length and progress must be finite.");
            double remaining = ClampProgress(progress) * total;
            for (int i = 0; i < lengths.Length; i++)
            {
                if (remaining <= lengths[i] || i == lengths.Length - 1)
                {
                    Interpolate(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1],
                                remaining / lengths[i], out x, out y);
                    return;
                }
                remaining -= lengths[i];
            }
            x = points[points.Length - 1][0];
            y = points[points.Length - 1][1];
        }
    }
}
