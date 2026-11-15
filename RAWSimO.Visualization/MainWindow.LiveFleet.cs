#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using RAWSimO.Live.Networking;
using RAWSimO.Live.Protocol;
using RAWSimO.Live.Tracking;
using RAWSimO.Visualization.Rendering;
using ShapePath = System.Windows.Shapes.Path;

namespace RAWSimO.Visualization;

/// <summary>
/// Per-vehicle rendering and connection state for the live layer.
///
/// The design document fixes one bridge per vehicle ("一个桥接脚本对应一辆车；M1 只做一辆，
/// 多车留到 M4"), so a vehicle is identified by the slot it was connected through and gets its
/// own client, tracker, canvas group, colour and trail. Nothing here touches the simulation:
/// the live layer stays an overlay on top of the map, so collisions, reservations and statistics
/// are unaffected.
/// </summary>
public partial class MainWindow
{
    /// <summary>Whether the first placed pose should centre the 2D view on itself.</summary>
    private static bool AutoCenterOnFirstPose;

    /// <summary>Set once the view has been centred, so later frames do not fight user panning.</summary>
    private static bool _autoCentered;

    /// <summary>Everything the layer needs for one connected vehicle.</summary>
    private sealed class LiveVehicleVisual
    {
        /// <summary>Connection slot ("host:port"); the dictionary key, stable across hello.</summary>
        public string SlotKey = string.Empty;

        /// <summary>Vehicle id announced by the bridge's hello, or empty before it arrives.</summary>
        public string VehicleId = string.Empty;

        /// <summary>Last pose seq whose canvas placement was written to the log.</summary>
        public long LastLoggedSeq = -1;

        /// <summary>Coordinate frame the bridge announced in its hello.</summary>
        public RAWSimO.Live.Calibration.AnnouncedFrame AnnouncedFrame = RAWSimO.Live.Calibration.AnnouncedFrame.Unknown;

        /// <summary>Capabilities the bridge advertised in its hello ("ping", "nav", "route", ...).</summary>
        public IReadOnlyList<string> Caps = Array.Empty<string>();

        /// <summary>True when the announced frame cannot be projected onto the map yet.</summary>
        public bool FrameBlocked;

        /// <summary>Why the layer is hidden, shown in the fleet panel.</summary>
        public string FrameBlockMessage = string.Empty;

        /// <summary>Palette slot reserved for this vehicle; released when it disconnects.</summary>
        public int PaletteSlot;

        /// <summary>Colour assigned to this slot.</summary>
        public Color Color;

        /// <summary>Canvas group holding this vehicle's shapes.</summary>
        public Canvas Group = null!;

        /// <summary>Vehicle body with orientation line.</summary>
        public System.Windows.Shapes.Ellipse Body = null!;

        /// <summary>Rotation of the body, in map orientation.</summary>
        public RotateTransform BodyRotate = null!;

        /// <summary>Translation of the body, in canvas pixels.</summary>
        public TranslateTransform BodyTranslate = null!;

        /// <summary>Travelled trail.</summary>
        public Polyline Trail = null!;

        /// <summary>Label showing the vehicle id, sequence and age.</summary>
        public TextBlock Label = null!;

        /// <summary>Red frame drawn when the pose falls outside the map.</summary>
        public Rectangle OutOfMapFrame = null!;

        /// <summary>Client for this slot; replaced on reconnect.</summary>
        public LiveTcpClient? Client;

        /// <summary>Pose validation and freshness tracking for this slot.</summary>
        public readonly PoseTracker Tracker = new(3.0, 3.0);

        /// <summary>Clock offset estimate (proto 1).</summary>
        public readonly ClockSync ClockSync = new();

        /// <summary>Protocol version announced by hello, or -1 before hello.</summary>
        public int Proto = -1;

        /// <summary>Monotonic time of the last ping sent.</summary>
        public double LastPingAt;

        /// <summary>Monotonic time of the last message received.</summary>
        public double LastMessageAt;

        /// <summary>Last recoverable error, shown in the status line.</summary>
        public string LastError = string.Empty;

        /// <summary>Last observed connection state, for edge detection.</summary>
        public LiveConnectionState LastState = LiveConnectionState.Disconnected;

        /// <summary>True while a watchdog restart is in progress.</summary>
        public bool WatchdogBusy;

        /// <summary>Display name: the vehicle id once known, otherwise the slot.</summary>
        public string DisplayName => string.IsNullOrEmpty(VehicleId) ? SlotKey + "（等待握手）" : VehicleId;
    }

    /// <summary>Vehicles currently connected, keyed by slot.</summary>
    private readonly Dictionary<string, LiveVehicleVisual> _liveVehicles = new(StringComparer.Ordinal);

    /// <summary>Slot of the vehicle that receives navigation commands and shows in the status line.</summary>
    private string? _liveActiveSlot;

    /// <summary>Next palette index to hand out.</summary>
    private int _liveNextPaletteIndex;

    /// <summary>Returns the active vehicle, or the first connected one when none is selected.</summary>
    private LiveVehicleVisual? ActiveLiveVehicle()
    {
        if (_liveActiveSlot != null && _liveVehicles.TryGetValue(_liveActiveSlot, out var active))
            return active;
        return _liveVehicles.Values.FirstOrDefault();
    }

    /// <summary>Colour allocator shared by every vehicle in this window.</summary>
    private readonly FleetColorAllocator _liveColors = new();

    /// <summary>Creates the canvas group and shapes for one vehicle and adds it to the overlay.</summary>
    private LiveVehicleVisual CreateLiveVehicle(string slotKey)
    {
        var paletteSlot = _liveColors.Acquire();
        var rgb = FleetColorAllocator.ColorOf(paletteSlot);
        var color = Color.FromRgb(rgb.R, rgb.G, rgb.B);
        var vehicle = new LiveVehicleVisual { SlotKey = slotKey, PaletteSlot = paletteSlot, Color = color };

        // The group holds the body and label at their real canvas coordinates, so it has to
        // span the whole canvas. A Canvas with no size still reports its children as Visible
        // with a measured size, but paints nothing: the child sits at (4999, 2512) inside a
        // zero-sized parent and is clipped away. That is why the window could show
        // "body vis=Visible actual=20.0x20.0 fill=#FF2E7D32" and still have no green pixel
        // anywhere in the screenshot.
        vehicle.Group = new Canvas { IsHitTestVisible = false, Background = null };
        Canvas.SetLeft(vehicle.Group, 0.0);
        Canvas.SetTop(vehicle.Group, 0.0);
        _liveOverlay.Children.Add(vehicle.Group);

        vehicle.OutOfMapFrame = new Rectangle
        {
            Stroke = Brushes.Red,
            StrokeThickness = 2,
            Visibility = Visibility.Collapsed,
            IsHitTestVisible = false,
        };
        Canvas.SetLeft(vehicle.OutOfMapFrame, 0);
        Canvas.SetTop(vehicle.OutOfMapFrame, 0);
        vehicle.Group.Children.Add(vehicle.OutOfMapFrame);

        vehicle.Trail = new Polyline
        {
            Stroke = MakeFrozenBrush(color.R, color.G, color.B, 0x99),
            StrokeThickness = 1.5,
            StrokeLineJoin = PenLineJoin.Round,
            Visibility = Visibility.Collapsed,
            IsHitTestVisible = false,
        };
        vehicle.Group.Children.Add(vehicle.Trail);

        var vehicleBox = 20.0;
        vehicle.BodyRotate = new RotateTransform();
        vehicle.BodyTranslate = new TranslateTransform();
        var transforms = new TransformGroup();
        transforms.Children.Add(vehicle.BodyRotate);
        transforms.Children.Add(vehicle.BodyTranslate);
        // A plain Ellipse, not a Path with a GeometryGroup. The Path carried the circle plus a
        // heading tick in one GeometryGroup, and reported Visible with a 20x20 layout slot while
        // painting nothing at all -- measured: 0 green pixels in the window although
        // TransformToAncestor placed the shape exactly where it belonged. A Shape whose geometry
        // is a group is easy to measure as empty; an Ellipse has a size of its own.
        vehicle.Body = new System.Windows.Shapes.Ellipse
        {
            Width = vehicleBox,
            Height = vehicleBox,
            Fill = MakeFrozenBrush(color.R, color.G, color.B),
            Stroke = MakeFrozenBrush(color.R, color.G, color.B),
            StrokeThickness = 2,
            RenderTransformOrigin = new System.Windows.Point(0.5, 0.5),
            RenderTransform = transforms,
            Visibility = Visibility.Collapsed,
            IsHitTestVisible = false,
        };
        // The body is parented to the overlay itself, not to the per-vehicle group. Measured
        // behaviour: with the body inside the group the group painted (23925 lime pixels when its
        // background was forced on) while the body inside it never produced a single pixel, at the
        // same coordinates the transform reported. Keeping the body a direct child of the overlay
        // removes that nesting from the render path; the group is left holding only the
        // out-of-map frame, which is positioned in overlay coordinates anyway.
        _liveOverlay.Children.Add(vehicle.Body);

        vehicle.Label = new TextBlock
        {
            Background = Brushes.White,
            Foreground = MakeFrozenBrush(color.R, color.G, color.B),
            FontSize = 11,
            Padding = new Thickness(2, 0, 2, 0),
            Visibility = Visibility.Collapsed,
            IsHitTestVisible = false,
        };
        vehicle.Group.Children.Add(vehicle.Label);

        _liveVehicles[slotKey] = vehicle;
        RefreshLiveFleetPanel();
        return vehicle;
    }

    /// <summary>Removes one vehicle and its shapes.</summary>
    private async System.Threading.Tasks.Task RemoveLiveVehicleAsync(string slotKey)
    {
        if (!_liveVehicles.TryGetValue(slotKey, out var vehicle)) return;
        _liveVehicles.Remove(slotKey);
        _liveColors.Release(vehicle.PaletteSlot);
        try { if (vehicle.Client != null) await vehicle.Client.StopAsync().ConfigureAwait(true); } catch { }
        vehicle.Client = null;
        try { _liveOverlay.Children.Remove(vehicle.Group); } catch { }
        if (_liveActiveSlot == slotKey) _liveActiveSlot = _liveVehicles.Keys.FirstOrDefault();
        RefreshLiveFleetPanel();
    }

    /// <summary>Rebuilds the list of connected vehicles in the control panel.</summary>
    private void RefreshLiveFleetPanel()
    {
        if (_liveFleetPanel == null) return;
        _liveFleetPanel.Children.Clear();
        if (_liveVehicles.Count == 0)
        {
            _liveFleetPanel.Children.Add(new TextBlock
            {
                Text = "（尚未连接任何车辆）",
                Foreground = Brushes.Gray,
                FontSize = 11,
            });
            return;
        }

        foreach (var vehicle in _liveVehicles.Values.OrderBy(v => v.SlotKey, StringComparer.Ordinal))
        {
            var row = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 1, 0, 1) };

            row.Children.Add(new Rectangle
            {
                Width = 10,
                Height = 10,
                Fill = MakeFrozenBrush(vehicle.Color.R, vehicle.Color.G, vehicle.Color.B),
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 5, 0),
            });

            var isActive = _liveActiveSlot == vehicle.SlotKey;
            var button = new Button
            {
                Content = (isActive ? "● " : "○ ") + vehicle.DisplayName + "  " + vehicle.SlotKey,
                Padding = new Thickness(4, 0, 4, 0),
                FontSize = 11,
                ToolTip = "点击把导航命令切到这一辆车",
                Background = isActive ? Brushes.LightYellow : null,
            };
            var slot = vehicle.SlotKey;
            button.Click += (_, _) =>
            {
                _liveActiveSlot = slot;
                RefreshLiveFleetPanel();
            };
            row.Children.Add(button);

            var drop = new Button
            {
                Content = "断开",
                Padding = new Thickness(4, 0, 4, 0),
                FontSize = 11,
                Margin = new Thickness(4, 0, 0, 0),
            };
            drop.Click += async (_, _) => await RemoveLiveVehicleAsync(slot).ConfigureAwait(true);
            row.Children.Add(drop);

            _liveFleetPanel.Children.Add(row);
        }
    }

    /// <summary>
    /// Renders one vehicle: body, trail, label and the out-of-map frame. The freshness colour is
    /// decided per vehicle from its own tracker, so one vehicle going stale does not affect another.
    /// </summary>
    private void RenderLiveVehicle(LiveVehicleVisual vehicle, double now, double canvasWidth, double canvasHeight,
        double mapWidth, double mapHeight)
    {
        vehicle.OutOfMapFrame.Width = canvasWidth;
        vehicle.OutOfMapFrame.Height = canvasHeight;
        vehicle.Tracker.ConfigureMap(mapWidth, mapHeight);

        var freshness = vehicle.Tracker.GetFreshness(now);
        var latest = vehicle.Tracker.Latest;
        var brush = latest is null
            ? LiveDisconnectedBrush
            : freshness == PoseFreshness.Fresh ? MakeFrozenBrush(vehicle.Color.R, vehicle.Color.G, vehicle.Color.B)
            : freshness == PoseFreshness.Stale ? LiveStaleBrush
            : LiveDisconnectedBrush;

        vehicle.Body.Fill = brush;
        vehicle.Body.Stroke = brush;
        vehicle.Label.Foreground = brush;

        if (latest is null || vehicle.FrameBlocked)
        {
            vehicle.Body.Visibility = Visibility.Collapsed;
            vehicle.Trail.Visibility = Visibility.Collapsed;
            vehicle.Label.Visibility = Visibility.Collapsed;
            vehicle.OutOfMapFrame.Visibility = Visibility.Collapsed;
            return;
        }

        var mapPose = _liveCalibration != null ? _liveCalibration.ChassisToMap(latest.Pose) : latest.Pose;
        var transformer = new Transformation2D(mapWidth, mapHeight, canvasWidth, canvasHeight);
        var x = transformer.ProjectX(mapPose.X);
        var y = transformer.ProjectY(mapPose.Y);

        vehicle.BodyRotate.Angle = Transformation2D.ProjectOrientation(mapPose.Yaw);
        // Keep the marker a constant size on screen. The body lives in canvas coordinates, so a
        // fixed 20 px box is 20 px only at scale 1: at the fitted scale for a 140x90 m workshop
        // (about 0.056) it is a single pixel, and the vehicle is invisible although it is drawn
        // in exactly the right place. Scaling the box by 1/ContentScale gives it the same size
        // whatever the zoom, which is what makes it visible in both the fitted overview and the
        // zoomed-in view.
        var uiScale = zoomAndPanControl != null && zoomAndPanControl.ContentScale > 1e-6
            ? zoomAndPanControl.ContentScale
            : 1.0;
        var ui = 1.0 / uiScale;
        var box = Math.Min(600.0, Math.Max(6.0, 14.0 * ui));
        if (Math.Abs(vehicle.Body.Width - box) > 0.5)
        {
            vehicle.Body.Width = box;
            vehicle.Body.Height = box;
            vehicle.Body.StrokeThickness = Math.Min(6.0, Math.Max(1.0, 1.6 * ui));
        }
        vehicle.BodyTranslate.X = x - box / 2.0;
        vehicle.BodyTranslate.Y = y - box / 2.0;
        vehicle.Body.Visibility = Visibility.Visible;

        // One placement line every 200 poses. Kept permanently: "drawn off screen" and "not
        // drawn at all" look identical in a screenshot, so the canvas coordinate and the current
        // zoom have to come out of the window for a capture to be checkable. It also records the
        // exact inputs of the mapping (map extent, canvas size, scale), which is what a
        // reproduction needs to recompute the expected pixel and compare it with the capture.
        if (vehicle.LastLoggedSeq != latest.Seq && (latest.Seq % 200 == 0 || vehicle.LastLoggedSeq == -1))
        {
            vehicle.LastLoggedSeq = latest.Seq;
            LogLine(string.Format(CultureInfo.InvariantCulture,
                "[live] place {0} seq={1} map=({2:F3},{3:F3}) canvas=({4:F1},{5:F1}) extent={6:F0}x{7:F0}m canvas={8:F0}x{9:F0}px scale={10:F4}",
                vehicle.DisplayName, latest.Seq, mapPose.X, mapPose.Y, x, y,
                mapWidth, mapHeight, canvasWidth, canvasHeight,
                zoomAndPanControl != null ? zoomAndPanControl.ContentScale : 0.0));

            // Zoom to the vehicle the first time a pose places it. Off by default now that the
            // marker keeps a constant screen size and the fitted overview is readable; set
            // AMR_AUTOCENTER=1 for a close-up instead.
            if (AutoCenterOnFirstPose && !_autoCentered)
            {
                _autoCentered = true;
                Dispatcher.BeginInvoke(new Action(() => CenterMapOn(x, y)),
                    System.Windows.Threading.DispatcherPriority.Background);
            }
        }

        var points = new PointCollection();
        foreach (var point in vehicle.Tracker.SnapshotTrail())
        {
            var pose = _liveCalibration != null ? _liveCalibration.ChassisToMap(point.Pose) : point.Pose;
            points.Add(new System.Windows.Point(transformer.ProjectX(pose.X), transformer.ProjectY(pose.Y)));
        }
        vehicle.Trail.Points = points;
        vehicle.Trail.Visibility = points.Count > 1 ? Visibility.Visible : Visibility.Collapsed;

        // The label lives in canvas coordinates too, so it shrinks with the zoom like the body
        // would have. Scale the font and the label box by the same factor as the marker.
        var labelScale = 1.0 / uiScale;
        vehicle.Label.FontSize = Math.Min(60.0, 10.5 * labelScale);
        Canvas.SetLeft(vehicle.Label, Math.Min(canvasWidth - 80.0 * labelScale, x + box * 0.75));
        Canvas.SetTop(vehicle.Label, Math.Max(0.0, y - box * 0.9));
        vehicle.Label.Text = string.Format(CultureInfo.InvariantCulture, "{0}  {1}  {2:F0} ms",
            vehicle.DisplayName, latest.Seq, vehicle.Tracker.AgeSeconds(now) * 1000.0);
        vehicle.Label.Visibility = Visibility.Visible;

        var outside = latest.OutsideMap || mapPose.X < 0 || mapPose.Y < 0 || mapPose.X > mapWidth || mapPose.Y > mapHeight;
        vehicle.OutOfMapFrame.Visibility = outside ? Visibility.Visible : Visibility.Collapsed;
    }
}
