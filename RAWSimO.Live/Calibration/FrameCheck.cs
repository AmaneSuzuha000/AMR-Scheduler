using System;

namespace RAWSimO.Live.Calibration
{
    /// <summary>
    /// The frame a bridge announces in its hello message, expressed as the question that
    /// actually matters to the GUI: can the reported pose be drawn on the map as-is?
    /// </summary>
    public enum AnnouncedFrame
    {
        /// <summary>Already in map coordinates; no calibration needed.</summary>
        Map,
        /// <summary>Odometry frame: drifts, needs a calibration to reach the map.</summary>
        Odom,
        /// <summary>Robot body frame: meaningless as a position unless calibrated.</summary>
        Chassis,
        /// <summary>Synthetic frame used by the simulated source.</summary>
        Sim,
        /// <summary>Unknown or missing frame string.</summary>
        Unknown,
    }

    /// <summary>
    /// Decides whether a hello's frame may be drawn directly or requires a calibration file.
    /// <para>
    /// Drawing a chassis- or odom-frame pose on the map without calibrating it produces a
    /// marker that looks plausible and is wrong, which is worse than refusing to draw: the
    /// user cannot tell the difference by looking. This type makes that decision explicit and
    /// testable instead of leaving it implicit in the caller.
    /// </para>
    /// </summary>
    public static class FrameCheck
    {
        /// <summary>Parses the frame string of a hello message.</summary>
        /// <param name="frame">The announced frame, may be null or empty.</param>
        /// <returns>The parsed frame.</returns>
        public static AnnouncedFrame Parse(string? frame)
        {
            if (string.IsNullOrWhiteSpace(frame)) return AnnouncedFrame.Unknown;
            switch (frame.Trim().ToLowerInvariant())
            {
                case "map": return AnnouncedFrame.Map;
                case "odom": return AnnouncedFrame.Odom;
                case "chassis": return AnnouncedFrame.Chassis;
                case "sim": return AnnouncedFrame.Sim;
                default: return AnnouncedFrame.Unknown;
            }
        }

        /// <summary>Whether poses announced in this frame can be projected without a calibration.</summary>
        /// <param name="frame">The announced frame.</param>
        /// <returns><c>true</c> if the pose is already map-relative.</returns>
        public static bool IsDirectlyDrawable(AnnouncedFrame frame) =>
            frame == AnnouncedFrame.Map || frame == AnnouncedFrame.Sim;

        /// <summary>
        /// Returns an error text when poses may not be drawn, or <c>null</c> when they may.
        /// </summary>
        /// <param name="frame">The announced frame.</param>
        /// <param name="hasCalibration">Whether a calibration file is loaded.</param>
        /// <returns>A message for the status bar, or <c>null</c>.</returns>
        public static string? Describe(AnnouncedFrame frame, bool hasCalibration)
        {
            if (IsDirectlyDrawable(frame)) return null;
            if (hasCalibration) return null;
            return frame switch
            {
                AnnouncedFrame.Chassis =>
                    "桥接声明 chassis 坐标系，但未加载标定文件：位置不能直接画到地图上。请先多点标定或加载 calibration.json。",
                AnnouncedFrame.Odom =>
                    "桥接声明 odom 坐标系，但未加载标定文件：里程计原点与地图原点不同，位置不能直接画。请先标定。",
                _ => "桥接声明的坐标系未知，且未加载标定文件：位置不能直接画到地图上。",
            };
        }
    }
}
