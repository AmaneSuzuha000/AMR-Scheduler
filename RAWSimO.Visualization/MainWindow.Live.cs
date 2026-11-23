using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;
using System.Windows.Threading;
using Microsoft.Win32;
using RAWSimO.Live.Calibration;
using RAWSimO.Live.Geometry;
using RAWSimO.Live.Logging;
using RAWSimO.Live.Networking;
using RAWSimO.Live.Planning;
using RAWSimO.Live.Protocol;
using RAWSimO.Live.Tracking;
using RAWSimO.Visualization.Rendering;
using ShapePath = System.Windows.Shapes.Path;

namespace RAWSimO.Visualization
{
    /// <summary>
    /// Live-vehicle layer and remote-control controls. Kept in a separate
    /// partial file so the original simulation/remote-control handlers stay
    /// untouched.
    /// </summary>
    public partial class MainWindow
    {
        private const double LivePixelsPerMeter = 40.0;

        private static readonly Brush LiveFreshBrush = MakeFrozenBrush(0x2E, 0x7D, 0x32);
        private static readonly Brush LiveStaleBrush = MakeFrozenBrush(0xF9, 0xA8, 0x25);
        private static readonly Brush LiveDisconnectedBrush = MakeFrozenBrush(0x75, 0x75, 0x75);
        private static readonly Brush LiveTrailBrush = MakeFrozenBrush(0x66, 0xBB, 0x6A);

        private Canvas _liveOverlay;
        private TextBlock _liveStatusText;
        private TextBlock _livePoseText;
        private TextBox _liveHostBox;
        private TextBox _livePortBox;
        private CheckBox _liveLogCheck;
        private Button _liveConnectButton;
        private Button _liveDisconnectButton;
        private Button _liveEstopButton;
        private ComboBox _liveTargetCombo;
        private Button _liveExecuteButton;
        private Button _liveCancelButton;
        private CheckBox _liveDirectCheck;
        private TextBlock _livePlanText;
        private Polyline _livePlanLine;
        private int? _liveSelectedTarget;
        /// <summary>Target of the most recent route request, so a late answer for an older one is ignored.</summary>
        private int? _liveRequestedTarget;

        /// <summary>
        /// Scene id of the loaded map, taken from the instance name (the exporter writes
        /// "&lt;scene_id&gt;_map"). Compared with the scene the bridge announces, so a route is
        /// never drawn on one scene while the vehicle sits in another.
        /// </summary>
        private string SceneId
        {
            get
            {
                var name = _instance?.Name;
                if (string.IsNullOrWhiteSpace(name)) return string.Empty;
                return name.EndsWith("_map", StringComparison.OrdinalIgnoreCase)
                    ? name.Substring(0, name.Length - 4)
                    : name;
            }
        }
        private bool _liveRefreshingWaypoints;
        private int _liveWaypointCount = -1;
        private object _liveWaypointInstance;
        private IReadOnlyList<Point2D> _livePlannedMapPath = Array.Empty<Point2D>();
        private readonly NavExecution _liveNav = new();
        private double _liveLastPlanSeconds;

        private JsonlPoseLogger _liveLogger;
        private CalibrationResult _liveCalibration;
        private DispatcherTimer _liveTimer;
        private bool _liveInitialized;
        private string _liveLastError;
        private long _liveTimerTicks;

        /// <summary>Total pose messages received, across all slots; used to prove arrival.</summary>
        private long _livePoseMessages;
        private double _liveTimerStart;
        private StackPanel _liveFleetPanel;
        private TextBlock _liveFleetSummary;

        /// <summary>Vehicle id of the active vehicle; kept as a member so the calibration partial compiles unchanged.</summary>
        private string _liveVehicleId => ActiveLiveVehicle()?.VehicleId ?? string.Empty;

        /// <summary>Tracker of the active vehicle; the calibration partial updates this one.</summary>
        private PoseTracker _liveTracker => ActiveLiveVehicle()?.Tracker;

        private void InitializeLiveLayer()
        {
            if (_liveInitialized) return;
            _liveInitialized = true;

            _liveOverlay = new Canvas
            {
                Name = "LiveVehicleOverlay",
                IsHitTestVisible = false,
                Background = null,
            };
            Panel.SetZIndex(_liveOverlay, 100);
            content.Children.Add(_liveOverlay);
            content.MouseRightButtonUp += LiveMap_MouseRightButtonUp;

            _livePlanLine = new Polyline
            {
                Name = "LivePlanLine",
                Stroke = Brushes.DarkOrange,
                StrokeThickness = 2.0,
                StrokeDashArray = new DoubleCollection(new[] { 4.0, 3.0 }),
                StrokeLineJoin = PenLineJoin.Round,
                Visibility = Visibility.Collapsed,
                IsHitTestVisible = false,
            };
            _liveOverlay.Children.Add(_livePlanLine);

            _liveTimer = new DispatcherTimer(DispatcherPriority.Render)
            {
                Interval = TimeSpan.FromMilliseconds(33),
            };
            _liveTimer.Tick += LiveTimer_Tick;
            _liveTimerStart = Stopwatch.GetTimestamp() / (double)Stopwatch.Frequency;
            _liveTimer.Start();

            BuildLiveControls();
            InitializeScheduleStreamLayer();
            Closed += (_, _) => DisposeLiveLayer();
        }

        private void BuildLiveControls()
        {
            if (TabEvents.Content is not StackPanel panel) return;

            var group = new GroupBox
            {
                Name = "GroupBoxLive",
                Header = "实车实时位置（Live）",
                Margin = new Thickness(0, 4, 0, 0),
            };
            var stack = new StackPanel { Orientation = Orientation.Vertical };

            var connectionRow = new StackPanel { Orientation = Orientation.Horizontal };
            connectionRow.Children.Add(new TextBlock
            {
                Text = "地址: ",
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 2, 0),
            });
            _liveHostBox = new TextBox { Name = "TextBoxLiveHost", Text = "127.0.0.1", Width = 95, Height = 22 };
            connectionRow.Children.Add(_liveHostBox);
            connectionRow.Children.Add(new TextBlock
            {
                Text = " 端口: ",
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 2, 0),
            });
            _livePortBox = new TextBox { Name = "TextBoxLivePort", Text = "9300", Width = 45, Height = 22 };
            connectionRow.Children.Add(_livePortBox);
            _liveConnectButton = new Button { Name = "ButtonLiveConnect", Content = "连接", Margin = new Thickness(4, 0, 0, 0), Padding = new Thickness(8, 1, 8, 1) };
            _liveConnectButton.Click += LiveConnect_Click;
            connectionRow.Children.Add(_liveConnectButton);
            _liveDisconnectButton = new Button { Name = "ButtonLiveDisconnect", Content = "断开", Margin = new Thickness(4, 0, 0, 0), Padding = new Thickness(8, 1, 8, 1), IsEnabled = false };
            _liveDisconnectButton.Click += LiveDisconnect_Click;
            connectionRow.Children.Add(_liveDisconnectButton);
            stack.Children.Add(connectionRow);

            _liveStatusText = new TextBlock
            {
                Name = "TextBlockLiveStatus",
                Text = "状态: 未连接",
                Margin = new Thickness(0, 4, 0, 0),
                Foreground = LiveDisconnectedBrush,
                // These two lines grow without bound: the status carries a clock offset and the
                // pose carries a sequence number. A TextBlock's natural width is its whole text,
                // and this panel sits in a Width="Auto" column, so an unwrapped status line made
                // the column as wide as the longest message ever shown -- measured at 1100 px --
                // and squeezed the map column down to a sliver. Wrapping caps the width and lets
                // the map keep its share.
                TextWrapping = TextWrapping.Wrap,
                MaxWidth = 520,
            };
            stack.Children.Add(_liveStatusText);

            _livePoseText = new TextBlock
            {
                Name = "TextBlockLivePose",
                Text = "最新位姿: -",
                Margin = new Thickness(0, 2, 0, 0),
                FontFamily = new FontFamily("Consolas"),
                FontSize = 11,
                TextWrapping = TextWrapping.Wrap,
                MaxWidth = 520,
            };
            stack.Children.Add(_livePoseText);

            var toolsRow = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 4, 0, 0) };
            toolsRow.Children.Add(new TextBlock
            {
                Text = "刷新频率: 33 ms",
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 12, 0),
            });
            var calibrationLoad = new Button { Content = "加载标定", Padding = new Thickness(6, 1, 6, 1) };
            calibrationLoad.Click += LiveLoadCalibration_Click;
            toolsRow.Children.Add(calibrationLoad);
            var calibrationSave = new Button { Content = "保存标定", Padding = new Thickness(6, 1, 6, 1), Margin = new Thickness(4, 0, 0, 0) };
            calibrationSave.Click += LiveSaveCalibration_Click;
            toolsRow.Children.Add(calibrationSave);
            var fit = new Button { Content = "适应窗口", Padding = new Thickness(6, 1, 6, 1), Margin = new Thickness(4, 0, 0, 0) };
            fit.Click += LiveFitWindow_Click;
            toolsRow.Children.Add(fit);
            _liveEstopButton = new Button
            {
                Content = "软件急停",
                Padding = new Thickness(8, 1, 8, 1),
                Margin = new Thickness(4, 0, 0, 0),
                Background = Brushes.IndianRed,
                Foreground = Brushes.White,
            };
            _liveEstopButton.Click += LiveEstop_Click;
            toolsRow.Children.Add(_liveEstopButton);
            var release = new Button
            {
                Name = "ButtonLiveEstopRelease",
                Content = "解除急停",
                Margin = new Thickness(4, 0, 0, 0),
                Padding = new Thickness(8, 1, 8, 1),
            };
            release.Click += LiveEstopRelease_Click;
            toolsRow.Children.Add(release);
            stack.Children.Add(toolsRow);

            var targetRow = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 4, 0, 0) };
            targetRow.Children.Add(new TextBlock
            {
                Text = "目标路点: ",
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 2, 0),
            });
            _liveTargetCombo = new ComboBox { Name = "ComboBoxLiveTarget", Width = 170, Height = 22 };
            _liveTargetCombo.SelectionChanged += LiveTargetCombo_SelectionChanged;
            targetRow.Children.Add(_liveTargetCombo);
            _liveExecuteButton = new Button { Name = "ButtonLiveExecute", Content = "执行", Margin = new Thickness(4, 0, 0, 0), Padding = new Thickness(8, 1, 8, 1), IsEnabled = false };
            _liveExecuteButton.Click += LiveExecuteTarget_Click;
            targetRow.Children.Add(_liveExecuteButton);
            _liveCancelButton = new Button { Name = "ButtonLiveCancel", Content = "取消", Margin = new Thickness(4, 0, 0, 0), Padding = new Thickness(8, 1, 8, 1), IsEnabled = false };
            _liveCancelButton.Click += LiveCancelTarget_Click;
            targetRow.Children.Add(_liveCancelButton);
            _liveDirectCheck = new CheckBox { Name = "CheckBoxLiveDirect", Content = "直达", VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(6, 0, 0, 0) };
            targetRow.Children.Add(_liveDirectCheck);
            stack.Children.Add(targetRow);

            _livePlanText = new TextBlock
            {
                Name = "TextBlockLivePlan",
                Text = "计划: 右键选路点，或从下拉框选择",
                Margin = new Thickness(0, 2, 0, 0),
                FontFamily = new FontFamily("Consolas"),
                FontSize = 11,
                TextWrapping = TextWrapping.Wrap,
            };
            stack.Children.Add(_livePlanText);

            var fleetGroup = new GroupBox
            {
                Name = "GroupBoxLiveFleet",
                Header = "已连接车辆",
                Margin = new Thickness(0, 6, 0, 0),
            };
            var fleetStack = new StackPanel { Orientation = Orientation.Vertical };
            _liveFleetPanel = new StackPanel { Orientation = Orientation.Vertical };
            fleetStack.Children.Add(_liveFleetPanel);
            _liveFleetSummary = new TextBlock
            {
                Name = "TextBlockLiveFleetSummary",
                Text = "（无）",
                Margin = new Thickness(0, 4, 0, 0),
                FontFamily = new FontFamily("Consolas"),
                FontSize = 11,
                TextWrapping = TextWrapping.Wrap,
                MaxWidth = 520,
            };
            fleetStack.Children.Add(_liveFleetSummary);
            var fleetButtons = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 4, 0, 0) };
            var addDemo = new Button { Name = "ButtonLiveDemoFleet", Content = "连两辆（演示）", Padding = new Thickness(8, 1, 8, 1), ToolTip = "连接当前端口和下一个端口，用于双车实时连接演示" };
            addDemo.Click += LiveConnectDemoFleet_Click;
            fleetButtons.Children.Add(addDemo);
            var dropAll = new Button { Name = "ButtonLiveDisconnectAll", Content = "全部断开", Margin = new Thickness(4, 0, 0, 0), Padding = new Thickness(8, 1, 8, 1) };
            dropAll.Click += LiveDisconnectAll_Click;
            fleetButtons.Children.Add(dropAll);
            fleetStack.Children.Add(fleetButtons);
            fleetGroup.Content = fleetStack;
            stack.Children.Add(fleetGroup);
            RefreshLiveFleetPanel();

            _liveLogCheck = new CheckBox { Name = "CheckBoxLiveLog", Content = "写 JSONL 日志", Margin = new Thickness(0, 4, 0, 0) };
            _liveLogCheck.Checked += LiveLogCheck_Changed;
            _liveLogCheck.Unchecked += LiveLogCheck_Changed;
            stack.Children.Add(_liveLogCheck);

            group.Content = stack;
            panel.Children.Add(group);
            BuildCalibrationControls(panel);
        }

        private void LiveMap_MouseRightButtonUp(object sender, MouseButtonEventArgs e)
        {
            if ((Keyboard.Modifiers & ModifierKeys.Shift) != 0) return;
            if (_instance == null)
            {
                _livePlanText.Text = "计划: 尚未加载实例";
                return;
            }
            var canvasPoint = e.GetPosition(content);
            var mapPoint = LiveCanvasToMap(canvasPoint);
            var positions = _instance.Waypoints.Select(w => new Point2D(w.X, w.Y)).ToList();
            var nearestIndex = MapPicking.NearestIndex(positions, mapPoint);
            if (nearestIndex < 0)
            {
                _livePlanText.Text = "计划: 实例中没有路点";
                return;
            }
            SelectLiveTarget(_instance.Waypoints[nearestIndex].ID, false);
            e.Handled = true;
        }

        private Point2D LiveCanvasToMap(System.Windows.Point canvasPoint)
        {
            var canvasWidth = Math.Max(1.0, double.IsFinite(theGrid.Width) ? theGrid.Width : content.ActualWidth);
            var canvasHeight = Math.Max(1.0, double.IsFinite(theGrid.Height) ? theGrid.Height : content.ActualHeight);
            return MapPicking.CanvasToMap(canvasPoint.X, canvasPoint.Y, canvasWidth, canvasHeight, LivePixelsPerMeter);
        }

        private void RefreshLiveWaypointChoices()
        {
            if (_liveTargetCombo == null) return;
            if (_instance == null)
            {
                if (_liveTargetCombo.Items.Count > 0)
                {
                    _liveRefreshingWaypoints = true;
                    _liveTargetCombo.ItemsSource = null;
                    _liveRefreshingWaypoints = false;
                }
                _liveWaypointCount = -1;
                return;
            }
            var waypoints = _instance.Waypoints;
            if (ReferenceEquals(_liveWaypointInstance, _instance) &&
                waypoints.Count == _liveWaypointCount &&
                _liveTargetCombo.Items.Count == waypoints.Count) return;
            _liveWaypointInstance = _instance;
            _liveWaypointCount = waypoints.Count;
            var selected = _liveSelectedTarget;
            var choices = waypoints.OrderBy(w => w.ID)
                .Select(w => new LiveWaypointChoice(w.ID, $"路点 {w.ID} ({w.X:F2}, {w.Y:F2})"))
                .ToList();
            _liveRefreshingWaypoints = true;
            _liveTargetCombo.ItemsSource = choices;
            _liveTargetCombo.SelectedItem = selected.HasValue ? choices.FirstOrDefault(c => c.Id == selected.Value) : null;
            _liveRefreshingWaypoints = false;
        }

        private void LiveTargetCombo_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_liveRefreshingWaypoints) return;
            if (_liveTargetCombo.SelectedItem is LiveWaypointChoice choice)
                SelectLiveTarget(choice.Id, true);
        }

        private void SelectLiveTarget(int waypointId, bool fromCombo)
        {
            _liveSelectedTarget = waypointId;
            _livePlanText.Text = $"已选: 路点 {waypointId}";
            RefreshLiveWaypointChoices();
            if (!fromCombo && _liveTargetCombo.ItemsSource is IEnumerable<LiveWaypointChoice> choices)
            {
                _liveRefreshingWaypoints = true;
                _liveTargetCombo.SelectedItem = choices.FirstOrDefault(c => c.Id == waypointId);
                _liveRefreshingWaypoints = false;
            }
            _liveExecuteButton.IsEnabled = true;
            // Ask Python for the route. The console draws what it is told; it does not plan.
            RequestLiveRoute(waypointId);
        }

        /// <summary>
        /// Asks the bridge for a route to <paramref name="target"/> and draws the answer.
        /// The planner lives in Python (tools/live_bridge/route_service.py); the console must
        /// never compute a route itself, otherwise there would be two decision cores and the
        /// displayed route could disagree with the one actually executed.
        /// </summary>
        private async void RequestLiveRoute(int target)
        {
            // Ask only if the bridge said it can answer. The caps list is the point of the
            // handshake; ignoring it meant a missing capability was discovered by sending a
            // request and waiting for a refusal (or silence).
            var caps = ActiveLiveVehicle()?.Caps;
            if (caps != null && caps.Count > 0 &&
                !caps.Any(c => string.Equals(c, "route", StringComparison.OrdinalIgnoreCase)))
            {
                _livePlannedMapPath = Array.Empty<Point2D>();
                _livePlanLine.Visibility = Visibility.Collapsed;
                _livePlanText.Text = "路线: 对端未声明 route 能力，无法规划";
                LogLine("[route] skipped: bridge caps = [" + string.Join(",", caps) + "]");
                _liveExecuteButton.IsEnabled = false;
                return;
            }
            _liveRequestedTarget = target;
            _livePlanText.Text = $"路线: 已向 Python 请求 路点 {target} …";
            try
            {
                var latest = _liveTracker?.Latest;
                IReadOnlyList<double> from = null;
                if (latest != null)
                {
                    var mapPose = _liveCalibration != null ? _liveCalibration.ChassisToMap(latest.Pose) : latest.Pose;
                    from = new[] { mapPose.X, mapPose.Y };
                }
                var direct = _liveDirectCheck?.IsChecked == true;
                LogLine($"[route] request waypoint {target} from="
                    + (from == null ? "none" : string.Format(System.Globalization.CultureInfo.InvariantCulture, "({0:F2},{1:F2})", from[0], from[1]))
                    + $" direct={direct}");
                await SendLiveAsync(new RouteRequestMessage(
                    target.ToString(System.Globalization.CultureInfo.InvariantCulture), from, direct));
            }
            catch (Exception ex)
            {
                _livePlanText.Text = "路线请求失败: " + ex.Message;
                _liveExecuteButton.IsEnabled = false;
            }
        }

        /// <summary>
        /// Draws a route the bridge returned. <paramref name="mapPath"/> is already in scene
        /// coordinates, so it needs no calibration: applying one would move the line off the map.
        /// </summary>
        private void ApplyLiveRoute(RouteResultMessage result)
        {
            LogLine($"[route] arrived target={result.Target ?? "?"} waypoint={result.Waypoint?.ToString() ?? "?"} "
                + $"points={(result.MapPath?.Count ?? 0)} nodes={(result.Nodes?.Count ?? 0)} reason={result.Reason ?? "-"}");
            // Only accept the answer to the request we are still waiting for. Python echoes the
            // RAWSim-O waypoint number; older answers carry the scene node id instead, so fall
            // back to that before deciding the answer is stale.
            int? answered = result.Waypoint;
            if (!answered.HasValue && result.Target != null &&
                int.TryParse(result.Target, System.Globalization.NumberStyles.Integer,
                    System.Globalization.CultureInfo.InvariantCulture, out var parsedTarget))
            {
                answered = parsedTarget;
            }
            if (answered.HasValue && _liveRequestedTarget.HasValue && answered.Value != _liveRequestedTarget.Value)
            {
                LogLine($"[route] stale answer waypoint={answered.Value} waiting={_liveRequestedTarget.Value}");
                return;
            }
            if (!string.IsNullOrEmpty(result.Reason))
            {
                _livePlannedMapPath = Array.Empty<Point2D>();
                _livePlanLine.Visibility = Visibility.Collapsed;
                _livePlanText.Text = "路线: " + result.Reason;
                LogLine("[route] refused by Python: " + result.Reason);
                _liveExecuteButton.IsEnabled = false;
                return;
            }
            var mapPath = result.MapPath ?? Array.Empty<Point2D>();
            if (mapPath.Count < 2)
            {
                _livePlannedMapPath = Array.Empty<Point2D>();
                _livePlanLine.Visibility = Visibility.Collapsed;
                _livePlanText.Text = "路线: 空路线";
                LogLine($"[route] empty map_path (points={mapPath.Count} nodes={(result.Nodes?.Count ?? 0)} target={result.Target ?? "?"} waypoint={result.Waypoint?.ToString() ?? "?"})");
                _liveExecuteButton.IsEnabled = false;
                return;
            }
            _livePlannedMapPath = mapPath;
            _liveLastPlanSeconds = result.Seconds ?? 0.0;
            DrawLivePlanLine(mapPath);
            var nodes = result.Nodes ?? Array.Empty<string>();
            var from = nodes.Count > 0 ? nodes[0] : "?";
            var to = nodes.Count > 0 ? nodes[nodes.Count - 1] : "?";
            _livePlanText.Text = string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "路线（Python）: {0}→{1}，{2} 点，{3:F2} m，{4:F1} s",
                from, to, nodes.Count, result.Meters ?? 0.0, result.Seconds ?? 0.0);
            // Also to the log: the panel text is not evidence on its own, and this line is what
            // proves the drawn route came from Python rather than from a local planner.
            LogLine("[route] from Python: " + string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0}->{1} nodes={2} points={3} m={4:F2} s={5:F1} waypoint={6}",
                from, to, nodes.Count, mapPath.Count, result.Meters ?? 0.0,
                result.Seconds ?? 0.0, result.Waypoint?.ToString() ?? "?"));
            _liveExecuteButton.IsEnabled = true;
        }

        /// <summary>Projects a scene-frame polyline onto the 2D canvas.</summary>
        private void DrawLivePlanLine(IReadOnlyList<Point2D> mapPath)
        {
            var canvasWidth = Math.Max(1.0, double.IsFinite(theGrid.Width) ? theGrid.Width : content.ActualWidth);
            var canvasHeight = Math.Max(1.0, double.IsFinite(theGrid.Height) ? theGrid.Height : content.ActualHeight);
            var mapWidth = Math.Max(0.001, canvasWidth / LivePixelsPerMeter);
            var mapHeight = Math.Max(0.001, canvasHeight / LivePixelsPerMeter);
            var transformer = new Transformation2D(mapWidth, mapHeight, canvasWidth, canvasHeight);
            var points = new PointCollection();
            foreach (var point in mapPath)
                points.Add(new System.Windows.Point(transformer.ProjectX(point.X), transformer.ProjectY(point.Y)));
            _livePlanLine.Points = points;
            _livePlanLine.Visibility = points.Count > 1 ? Visibility.Visible : Visibility.Collapsed;
            // The panel text can claim a route while the line failed to draw (a collapsed overlay,
            // a zero-sized canvas). Log the line's own state so the claim is checkable.
            var first = points.Count > 0 ? string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "({0:F1},{1:F1})", points[0].X, points[0].Y) : "-";
            var last = points.Count > 0 ? string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "({0:F1},{1:F1})", points[points.Count - 1].X, points[points.Count - 1].Y) : "-";
            LogLine(string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "[route] line drawn points={0} vis={1} first={2} last={3} canvas={4:F0}x{5:F0} scale={6:F4}",
                points.Count, _livePlanLine.Visibility, first, last,
                canvasWidth, canvasHeight, canvasWidth / mapWidth));
        }

        private WaypointGraph BuildLiveWaypointGraph()
        {
            var graph = new WaypointGraph();
            foreach (var waypoint in _instance.Waypoints)
                graph.AddWaypoint(waypoint.ID, new Point2D(waypoint.X, waypoint.Y));
            foreach (var waypoint in _instance.Waypoints)
            {
                foreach (var next in waypoint.Paths)
                {
                    if (next.ID == waypoint.ID || !graph.Contains(next.ID)) continue;
                    graph.AddEdge(waypoint.ID, next.ID);
                }
            }
            return graph;
        }

        private void HandleLiveNavResult(NavResultMessage navResult)
        {
            var goal = string.IsNullOrEmpty(navResult.Goal) ? "(unknown)" : navResult.Goal;
            var reason = string.IsNullOrEmpty(navResult.Reason) ? string.Empty : " / " + navResult.Reason;
            var applied = _liveNav.OnResult(navResult.Goal, navResult.Status, navResult.Reason);
            Dispatcher.BeginInvoke(new Action(() =>
            {
                if (_livePlanText == null) return;
                var suffix = applied ? "（" + NavExecution.Describe(_liveNav.State) + "）" : "（忽略：非当前目标）";
                _livePlanText.Text = $"nav_result: {goal} {navResult.Status}{reason}{suffix}";
                _liveCancelButton.IsEnabled = _liveNav.IsActive;
            }));
        }


        private async System.Threading.Tasks.Task HandleLiveNavTimeoutAsync(string goal)
        {
            try
            {
                _liveCancelButton.IsEnabled = _liveNav.IsActive;
                _livePlanText.Text = "执行超时（2×计划用时+5 s）: " + goal;
                await SendLiveAsync(new CancelCommandMessage(goal));
            }
            catch (Exception ex)
            {
                _livePlanText.Text = "超时取消失败: " + ex.Message;
            }
        }

        private async void LiveExecuteTarget_Click(object sender, RoutedEventArgs e)
        {
            if (!_liveSelectedTarget.HasValue) return;
            // The route came from Python; the console only executes what was drawn. If nothing was
            // drawn yet, ask first and let the answer arrive before the operator presses again.
            var mapPath = _livePlannedMapPath;
            if (mapPath.Count < 2)
            {
                _livePlanText.Text = "路线: 还没有从 Python 拿到路线，正在请求 …";
                RequestLiveRoute(_liveSelectedTarget.Value);
                return;
            }
            try
            {
                var chassisPath = mapPath
                    .Select(p => _liveCalibration != null ? _liveCalibration.MapToChassis(p) : p)
                    .ToList();
                if (_liveNav.IsActive && _liveNav.Goal != null)
                {
                    await SendLiveAsync(new CancelCommandMessage(_liveNav.Goal));
                }
                var goal = $"wp{_liveSelectedTarget.Value}-{DateTime.Now:HHmmss}";
                await SendLiveAsync(new NavCommandMessage(goal, chassisPath, 0.2));
                _liveNav.Start(goal, _liveLastPlanSeconds, LiveNowSeconds());
                _liveCancelButton.IsEnabled = true;
                _livePlanText.Text = $"已下发: {goal}，{mapPath.Count} 个点，{NavExecution.Describe(_liveNav.State)}";
            }
            catch (Exception ex)
            {
                _livePlanText.Text = "下发失败: " + ex.Message;
            }
        }

        private async void LiveCancelTarget_Click(object sender, RoutedEventArgs e)
        {
            var goal = _liveNav.Cancel();
            if (goal == null)
            {
                _livePlanText.Text = "计划: 没有活动目标";
                return;
            }
            try
            {
                await SendLiveAsync(new CancelCommandMessage(goal));
                _livePlanText.Text = "已取消: " + goal;
                _liveCancelButton.IsEnabled = false;
            }
            catch (Exception ex)
            {
                _livePlanText.Text = "取消失败: " + ex.Message;
            }
        }

        private void LiveTimer_Tick(object sender, EventArgs e)
        {
            if (_liveOverlay == null || theGrid == null) return;

            var canvasWidth = Math.Max(1.0, double.IsFinite(theGrid.Width) ? theGrid.Width : content.ActualWidth);
            var canvasHeight = Math.Max(1.0, double.IsFinite(theGrid.Height) ? theGrid.Height : content.ActualHeight);
            _liveOverlay.Width = canvasWidth;
            _liveOverlay.Height = canvasHeight;
            Canvas.SetLeft(_liveOverlay, 0.0);
            Canvas.SetTop(_liveOverlay, 0.0);
            // The schedule overlay is deliberately not resized here. Its contents are projected
            // in their own fitted space, whose scale is derived from the size it is drawn at --
            // forcing it to the simulation grid's size makes the fit describe a canvas bigger
            // than the window, and the vehicles near the far edge get drawn past the bottom of
            // the window. It sizes itself from its container instead.

            var mapWidth = Math.Max(0.001, canvasWidth / LivePixelsPerMeter);
            var mapHeight = Math.Max(0.001, canvasHeight / LivePixelsPerMeter);

            var now = Stopwatch.GetTimestamp() / (double)Stopwatch.Frequency;

            // Each vehicle's group is a canvas that spans the whole map, so its children can be
            // positioned in absolute canvas coordinates. Keep the size in step with the overlay;
            // a zero-sized group clips its children away without reporting anything wrong.
            foreach (var vehicle in _liveVehicles.Values)
            {
                if (vehicle.Group == null) continue;
                vehicle.Group.Width = canvasWidth;
                vehicle.Group.Height = canvasHeight;
            }

            // The overlay is created in the constructor, before an instance is loaded, so it sits
            // at index 0 of the shared canvas and every map element added later paints over it.
            // A Canvas honours ZIndex only among siblings it lays out itself, which is why
            // ZIndex=100 was not enough. Re-appending puts the overlay last, on top.
            if (_liveVehicles.Count > 0 && content.Children.IndexOf(_liveOverlay) != content.Children.Count - 1)
            {
                content.Children.Remove(_liveOverlay);
                content.Children.Add(_liveOverlay);
                LogLine("[live] overlay re-appended on top (was index 0, covered by map elements)");
            }


            if (_liveTimerTicks == 0)
            {
                foreach (var v in _liveVehicles.Values)
                {
                    var p = v.Body.TransformToAncestor(_liveOverlay).Transform(new System.Windows.Point(0, 0));
                    LogLine(string.Format(System.Globalization.CultureInfo.InvariantCulture,
                        "[live] body@{0:F0},{1:F0} group={2:F0}x{3:F0} overlay={4:F0}x{5:F0}",
                        p.X, p.Y, v.Group.ActualWidth, v.Group.ActualHeight,
                        _liveOverlay.ActualWidth, _liveOverlay.ActualHeight));
                }
                LogLine(string.Format(System.Globalization.CultureInfo.InvariantCulture,
                    "[live] layout overlay={0:F0}x{1:F0} vis={2} parent={3} | content={4:F0}x{5:F0} children={6} | scroller={7:F0}x{8:F0}",
                    _liveOverlay.ActualWidth, _liveOverlay.ActualHeight, _liveOverlay.Visibility,
                    _liveOverlay.Parent?.GetType().Name ?? "null",
                    content.ActualWidth, content.ActualHeight, content.Children.Count,
                    (scroller)?.ActualWidth ?? -1, (scroller)?.ActualHeight ?? -1));
            }

            // Render every connected vehicle independently: each has its own tracker, colour and
            // freshness, so one vehicle going stale never repaints another one.
            foreach (var vehicle in _liveVehicles.Values.ToList())
                RenderLiveVehicle(vehicle, now, canvasWidth, canvasHeight, mapWidth, mapHeight);

            var active = ActiveLiveVehicle();

            // Navigation, keep-alive and watchdog belong to the active vehicle only; the others are
            // display-only, which is what the design asks for ("实车只做显示和 A→B").
            if (active != null)
            {
                var activeState = active.Client?.State ?? LiveConnectionState.Disconnected;
                if (active.LastState == LiveConnectionState.Connected && activeState != LiveConnectionState.Connected)
                {
                    var interrupted = _liveNav.OnDisconnected();
                    if (interrupted != null)
                    {
                        _livePlanText.Text = "执行中断（未知），不自动恢复: " + interrupted;
                        _liveCancelButton.IsEnabled = false;
                    }
                }
                active.LastState = activeState;

                if (active.Client != null && activeState == LiveConnectionState.Connected)
                {
                    var timedOut = _liveNav.Tick(now);
                    if (timedOut != null)
                        _ = HandleLiveNavTimeoutAsync(timedOut);
                }
            }

            // Keep-alive and message watchdog run for EVERY vehicle, not just the active one.
            // The bridge stops its vehicle when it has not heard from the GUI for 0.5 s
            // (live_bridge.py --watchdog, default 0.5), and CircleVehicle.stop() is not reversible,
            // so a display-only vehicle would freeze after half a second if it were not pinged.
            foreach (var vehicle in _liveVehicles.Values.ToList())
            {
                if (vehicle.Client == null || vehicle.Client.State != LiveConnectionState.Connected) continue;
                if (vehicle.Proto >= 1 && now - vehicle.LastPingAt >= 0.2)
                {
                    vehicle.LastPingAt = now;
                    _ = SendLivePingAsync(vehicle);
                }
                if (!vehicle.WatchdogBusy && vehicle.LastMessageAt > 0 && now - vehicle.LastMessageAt > 2.0)
                    _ = RestartLiveAfterWatchdogAsync(vehicle);
            }

            UpdateLiveStatusText(active, now);
            UpdateLiveFleetSummary(now);

            if (active?.Tracker.Latest is { } latest)
                _livePoseText.Text = string.Format(
                    "{0}: seq={1} x={2:F3} y={3:F3} yaw={4:F1}°",
                    active.DisplayName, latest.Seq, latest.X, latest.Y, latest.Yaw * 180.0 / Math.PI);
            else
                _livePoseText.Text = "最新位姿: -";

            _liveTimerTicks++;
            if (_liveTimerTicks % 100 == 0)
            {
                RefreshLiveWaypointChoices();
                var elapsed = now - _liveTimerStart;
                LogLine($"[Live] 33 ms timer: {_liveTimerTicks} ticks, average {elapsed * 1000.0 / _liveTimerTicks:F2} ms");
                try
                {
                    var liveLogDir = System.IO.Path.Combine("logs", "live");
                    System.IO.Directory.CreateDirectory(liveLogDir);
                    System.IO.File.AppendAllText(System.IO.Path.Combine(liveLogDir, "ui_timer.log"),
                        $"{DateTime.Now:O}\t{elapsed * 1000.0 / _liveTimerTicks:F2} ms average after {_liveTimerTicks} ticks\n");
                }
                catch { }
            }
        }

        /// <summary>
        /// One line per connected vehicle, in a fixed machine-readable form. The acceptance probe
        /// reads this text to prove that two vehicles are displayed at the same time with distinct
        /// poses, rather than eyeballing a screenshot.
        /// </summary>
        private void UpdateLiveFleetSummary(double now)
        {
            if (_liveFleetSummary == null) return;
            if (_liveVehicles.Count == 0)
            {
                _liveFleetSummary.Text = "（无）";
                return;
            }

            var sb = new System.Text.StringBuilder();
            foreach (var vehicle in _liveVehicles.Values.OrderBy(v => v.SlotKey, StringComparer.Ordinal))
            {
                if (sb.Length > 0) sb.Append('\n');
                var freshness = vehicle.Tracker.GetFreshness(now);
                var mark = freshness == PoseFreshness.Fresh ? "OK"
                    : freshness == PoseFreshness.Stale ? "STALE" : "LOST";
                sb.Append(string.Format(System.Globalization.CultureInfo.InvariantCulture,
                    "{0} | {1} | {2} | {3} | ",
                    vehicle.DisplayName, vehicle.SlotKey,
                    $"#{vehicle.Color.R:X2}{vehicle.Color.G:X2}{vehicle.Color.B:X2}", mark));
                if (vehicle.Tracker.Latest is { } pose)
                    sb.Append(string.Format(System.Globalization.CultureInfo.InvariantCulture,
                        "seq={0} x={1:F3} y={2:F3} yaw={3:F1} age={4:F0}ms",
                        pose.Seq, pose.X, pose.Y, pose.Yaw * 180.0 / Math.PI, vehicle.Tracker.AgeSeconds(now) * 1000.0));
                else
                    sb.Append("seq=- x=- y=- yaw=-");
            }
            _liveFleetSummary.Text = sb.ToString();
        }

        /// <summary>
        /// Status line: a one-line summary of the active vehicle plus a count of the others, so the
        /// panel stays readable when several vehicles are connected.
        /// </summary>
        private void UpdateLiveStatusText(LiveVehicleVisual active, double now)
        {
            if (active is null)
            {
                _liveStatusText.Foreground = LiveDisconnectedBrush;
                _liveStatusText.Text = "状态: 未连接";
                return;
            }

            var state = active.Client?.State ?? LiveConnectionState.Disconnected;
            var freshness = active.Tracker.GetFreshness(now);
            var stateText = state switch
            {
                LiveConnectionState.Connected => "已连接",
                LiveConnectionState.Connecting => "连接中",
                LiveConnectionState.WaitingHello => "等待握手",
                LiveConnectionState.Reconnecting => "重连中",
                _ => "未连接",
            };
            var freshnessText = freshness switch
            {
                PoseFreshness.Fresh => "新鲜",
                PoseFreshness.Stale => "过期",
                _ => "断开",
            };
            var protoText = active.Proto switch
            {
                < 0 => string.Empty,
                0 => " | proto 0（兼容模式）",
                _ => $" | proto {active.Proto}",
            };
            var clockText = active.ClockSync.OffsetSeconds is double clockOffset
                ? $" | 时钟偏差 {clockOffset * 1000.0:F1} ms"
                : string.Empty;
            var others = _liveVehicles.Count - 1;
            var othersText = others > 0 ? $" | 另有 {others} 辆车" : string.Empty;

            _liveStatusText.Foreground = active.FrameBlocked
                ? LiveStaleBrush
                : active.Tracker.Latest is null
                ? LiveDisconnectedBrush
                : freshness == PoseFreshness.Fresh ? MakeFrozenBrush(active.Color.R, active.Color.G, active.Color.B)
                : freshness == PoseFreshness.Stale ? LiveStaleBrush
                : LiveDisconnectedBrush;
            var frameText = string.IsNullOrEmpty(active.FrameBlockMessage)
                ? string.Empty
                : " | 帧: " + active.FrameBlockMessage;
            _liveStatusText.Text = $"[{active.DisplayName}] 状态: {stateText} | 数据: {freshnessText}{protoText}{clockText}{othersText}" +
                                   frameText +
                                   (string.IsNullOrEmpty(active.LastError) ? string.Empty : " | " + active.LastError);
        }

        /// <summary>
        /// Adds one more vehicle connection. Each bridge process serves exactly one vehicle
        /// (design §3: "一个桥接脚本对应一辆车"), so the slot is the host:port pair and a second
        /// click on the same pair reconnects that slot instead of stacking duplicates.
        /// </summary>
        /// <summary>
        /// Connects the single-vehicle truth overlay without a click, for the command-line
        /// demo path. Same split-and-drive approach as the fleet stream.
        /// </summary>
        private void AutoConnectLive(string hostPort)
        {
            try
            {
                if (!TrySplitHostPort(hostPort, out var host, out var port))
                {
                    LogLine("[live] 地址无效: " + hostPort);
                    return;
                }
                if (_liveHostBox != null) _liveHostBox.Text = host;
                if (_livePortBox != null) _livePortBox.Text = port.ToString(System.Globalization.CultureInfo.InvariantCulture);
                LogLine("[live] 自动连接 " + host + ":" + port);
                // The live panel is on the 远程控制 tab; leaving the window on 仿真控制 would
                // connect successfully and still show no vehicle, which reads as a dead link.
                ShowRemoteControlTab();
                LiveConnect_Click(null, null);
            }
            catch (Exception ex)
            {
                LogLine("[live] 自动连接失败: " + ex.Message);
            }
        }

        /// <summary>
        /// Asks for a route to <paramref name="waypointId"/> once the live link is up.
        /// This exists so the "console asks Python, Python answers" path can be exercised
        /// unattended (a --live-route argument), instead of only by clicking the map. It
        /// requests; it never plans.
        /// </summary>
        private void RequestLiveRouteWhenConnected(int waypointId, int attemptsLeft = 40)
        {
            if (attemptsLeft <= 0) return;
            var vehicle = ActiveLiveVehicle();
            var connected = vehicle?.Client != null && vehicle.Client.State == LiveConnectionState.Connected;
            // A route needs a start, so wait for the first pose too. Asking before any pose has
            // arrived gets a refusal ("no current position"), which is correct but useless.
            var positioned = _liveTracker?.Latest != null;
            if (!connected || !positioned)
            {
                // Waiting on a timer rather than the connect callback keeps this independent of
                // the order in which slots are created.
                var timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(250) };
                timer.Tick += (s, e) =>
                {
                    timer.Stop();
                    RequestLiveRouteWhenConnected(waypointId, attemptsLeft - 1);
                };
                timer.Start();
                return;
            }
            SelectLiveTarget(waypointId, false);
        }

        /// <summary>Brings the 远程控制 tab (which hosts the live panel) to the front.</summary>
        private void ShowRemoteControlTab()
        {
            try
            {
                if (TabEvents?.Parent is System.Windows.Controls.TabControl tabs)
                {
                    tabs.SelectedItem = TabEvents;
                    // Do NOT refit here. The live panel takes the whole left column, leaving the
                    // map a sliver; fitting to that sliver magnifies the map so far that the one
                    // vehicle is no longer inside the visible window. The first pose centres the
                    // view on the vehicle instead (see AutoCenterOnFirstPose).
                }
            }
            catch (Exception ex) { LogLine("[live] 切换标签页失败: " + ex.Message); }
        }

        private async void LiveConnect_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                var host = string.IsNullOrWhiteSpace(_liveHostBox.Text) ? "127.0.0.1" : _liveHostBox.Text.Trim();
                var port = int.TryParse(_livePortBox.Text, out var parsedPort) ? parsedPort : 9300;
                var slotKey = host + ":" + port.ToString(System.Globalization.CultureInfo.InvariantCulture);

                if (_liveVehicles.TryGetValue(slotKey, out var existing))
                {
                    // Same slot: restart it so a stale socket does not linger.
                    if (existing.Client != null)
                    {
                        try { await existing.Client.StopAsync(); } catch { }
                        existing.Client = null;
                    }
                    existing.Tracker.Reset();
                    existing.Proto = -1;
                    existing.ClockSync.Clear();
                    existing.LastPingAt = 0;
                    existing.LastMessageAt = 0;
                    existing.LastError = string.Empty;
                }
                else
                {
                    existing = CreateLiveVehicle(slotKey);
                }

                if (_liveActiveSlot == null) _liveActiveSlot = slotKey;

                var client = new LiveTcpClient(new LiveClientOptions
                {
                    Host = host,
                    Port = port,
                    ConnectTimeout = TimeSpan.FromSeconds(2),
                    HelloTimeout = TimeSpan.FromSeconds(2),
                });
                var captured = existing;
                client.MessageReceived += message => OnLiveMessage(captured, message);
                client.HelloReceived += hello => OnLiveHello(captured, hello);
                client.Error += ex =>
                {
                    captured.LastError = IsBenignDisconnect(ex) ? string.Empty : ex.Message;
                    LogLine("[live] rx error: " + ex.GetType().Name + ": " + ex.Message);
                };
                existing.Client = client;
                await client.StartAsync();

                _liveDisconnectButton.IsEnabled = true;
                RefreshLiveFleetPanel();
                LogLine($"[Live] connecting slot {slotKey}");
            }
            catch (Exception ex)
            {
                _liveLastError = ex.Message;
                _liveStatusText.Text = "状态: 连接失败 - " + ex.Message;
            }
        }

        private async void LiveDisconnect_Click(object sender, RoutedEventArgs e)
        {
            var active = ActiveLiveVehicle();
            if (active != null)
            {
                await RemoveLiveVehicleAsync(active.SlotKey);
                var interruptedOnDisconnect = _liveNav.OnDisconnected();
                if (interruptedOnDisconnect != null)
                {
                    _livePlanText.Text = "执行中断（未知），不自动恢复: " + interruptedOnDisconnect;
                    _liveCancelButton.IsEnabled = false;
                }
            }

            if (_liveVehicles.Count == 0)
            {
                _liveDisconnectButton.IsEnabled = false;
                _liveStatusText.Text = "状态: 未连接";
                _livePoseText.Text = "最新位姿: -";
                CloseLiveLogger();
            }
        }

        /// <summary>Disconnects every vehicle at once.</summary>
        private async void LiveDisconnectAll_Click(object sender, RoutedEventArgs e)
        {
            foreach (var slot in _liveVehicles.Keys.ToList())
                await RemoveLiveVehicleAsync(slot);
            _liveDisconnectButton.IsEnabled = false;
            _liveStatusText.Text = "状态: 未连接";
            _livePoseText.Text = "最新位姿: -";
            CloseLiveLogger();
        }

        /// <summary>Connects two synthetic vehicles at once, for the dual-vehicle live demo.</summary>
        private async void LiveConnectDemoFleet_Click(object sender, RoutedEventArgs e)
        {
            if (!int.TryParse(_livePortBox.Text, out var basePort)) basePort = 9300;
            var host = string.IsNullOrWhiteSpace(_liveHostBox.Text) ? "127.0.0.1" : _liveHostBox.Text.Trim();
            foreach (var offset in new[] { 0, 1 })
            {
                var portBox = _livePortBox.Text;
                _livePortBox.Text = (basePort + offset).ToString(System.Globalization.CultureInfo.InvariantCulture);
                LiveConnect_Click(null, new RoutedEventArgs());
                _livePortBox.Text = portBox;
                await System.Threading.Tasks.Task.Delay(50);
            }
        }

        private async void LiveEstop_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                await SendLiveAsync(new EstopMessage());
                _liveStatusText.Text = "状态: 已发送软件急停";
            }
            catch (Exception ex)
            {
                _liveStatusText.Text = "急停发送失败: " + ex.Message;
            }
        }

        private async void LiveEstopRelease_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                await SendLiveAsync(new EstopReleaseMessage());
                _liveStatusText.Text = "状态: 已发送解除急停";
            }
            catch (Exception ex)
            {
                _liveStatusText.Text = "解除急停失败: " + ex.Message;
            }
        }

        private double LiveNowSeconds() => Stopwatch.GetTimestamp() / (double)Stopwatch.Frequency;

        private static double LiveUnixSeconds() => DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0;

        private async System.Threading.Tasks.Task SendLivePingAsync(LiveVehicleVisual vehicle)
        {
            if (vehicle.Client == null || vehicle.Client.State != LiveConnectionState.Connected) return;
            try
            {
                await vehicle.Client.SendAsync(new PingMessage(LiveUnixSeconds()));
            }
            catch (Exception ex)
            {
                vehicle.LastError = ex.Message;
            }
        }

        private async System.Threading.Tasks.Task RestartLiveAfterWatchdogAsync(LiveVehicleVisual vehicle)
        {
            if (vehicle.WatchdogBusy) return;
            vehicle.WatchdogBusy = true;
            try
            {
                LogLine($"[Live] watchdog on {vehicle.SlotKey}: no message for > 2 s, reconnecting");
                var old = vehicle.Client;
                if (old != null) await old.StopAsync();
                vehicle.Client = null;
                vehicle.Tracker.Reset();
                vehicle.Proto = -1;
                vehicle.ClockSync.Clear();
                vehicle.LastMessageAt = 0;

                var parts = vehicle.SlotKey.Split(':');
                var client = new LiveTcpClient(new LiveClientOptions
                {
                    Host = parts[0],
                    Port = parts.Length > 1 && int.TryParse(parts[1], out var port) ? port : 9300,
                    ConnectTimeout = TimeSpan.FromSeconds(2),
                    HelloTimeout = TimeSpan.FromSeconds(2),
                });
                client.MessageReceived += message => OnLiveMessage(vehicle, message);
                client.HelloReceived += hello => OnLiveHello(vehicle, hello);
                // A cancelled read is the normal outcome of an intentional disconnect; showing the
                // framework's English text in the status bar would read like a fault.
                client.Error += ex =>
                    vehicle.LastError = IsBenignDisconnect(ex) ? string.Empty : ex.Message;
                vehicle.Client = client;
                await client.StartAsync();
            }
            finally
            {
                vehicle.WatchdogBusy = false;
            }
        }

        private void LiveFitWindow_Click(object sender, RoutedEventArgs e)
        {
            try { zoomAndPanControl.ScaleToFit(); }
            catch (Exception ex) { LogLine("[Live] fit window failed: " + ex.Message); }
        }

        private void LiveLoadCalibration_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog { Filter = "Calibration JSON (*.json)|*.json|All files (*.*)|*.*" };
            if (dialog.ShowDialog() != true) return;
            try
            {
                _liveCalibration = CalibrationResult.Load(dialog.FileName);
                ApplyFrameGate();
                _liveStatusText.Text = $"状态: 已加载标定 {System.IO.Path.GetFileName(dialog.FileName)}";
                LogLine($"[Live] calibration loaded: {dialog.FileName}");
            }
            catch (Exception ex)
            {
                _liveStatusText.Text = "标定加载失败: " + ex.Message;
            }
        }

        private void LiveSaveCalibration_Click(object sender, RoutedEventArgs e)
        {
            if (_liveCalibration == null)
            {
                _liveStatusText.Text = "没有可保存的标定";
                return;
            }
            var dialog = new SaveFileDialog { Filter = "Calibration JSON (*.json)|*.json|All files (*.*)|*.*", FileName = "calibration.json" };
            if (dialog.ShowDialog() != true) return;
            try
            {
                _liveCalibration.Save(dialog.FileName);
                LogLine($"[Live] calibration saved: {dialog.FileName}");
            }
            catch (Exception ex)
            {
                _liveStatusText.Text = "标定保存失败: " + ex.Message;
            }
        }

        private void LiveLogCheck_Changed(object sender, RoutedEventArgs e)
        {
            if (_liveLogCheck.IsChecked == true) OpenLiveLogger();
            else CloseLiveLogger();
        }

        private void OnLiveMessage(LiveVehicleVisual vehicle, LiveMessage message)
        {
            try
            {
                vehicle.LastMessageAt = LiveNowSeconds();
                // Count what actually arrived. Without this, "connected but nothing on screen"
                // is indistinguishable from "nothing arrived", and the two need different fixes.
                if (message is PoseMessage)
                {
                    _livePoseMessages++;
                    if (_livePoseMessages == 1 || _livePoseMessages % 100 == 0)
                        LogLine($"[live] rx pose #{_livePoseMessages} from {vehicle.SlotKey}");
                }
                else if (message is not StatusMessage)
                {
                    LogLine($"[live] rx {message.GetType().Name} from {vehicle.SlotKey}");
                }
                if (message is PoseMessage pose)
                {
                    vehicle.Tracker.Update(pose, vehicle.LastMessageAt, DateTimeOffset.Now);
                }
                else if (message is PongMessage pong)
                {
                    vehicle.ClockSync.Add(pong.T1, pong.T2, pong.T3, LiveUnixSeconds());
                }
                else if (message is StatusMessage status)
                {
                    // The status line is where the bridge admits it lost its own pose source.
                    // Without this the layer would paint a frozen pose as a live vehicle,
                    // because the repeated message still arrives on time.
                    vehicle.Tracker.SetSourceError(status.Error);
                }
                else if (message is NavResultMessage navResult)
                {
                    HandleLiveNavResult(navResult);
                }
                else if (message is RouteResultMessage routeResult)
                {
                    // The route came from Python. Drawing it here is the console's whole job.
                    Dispatcher.BeginInvoke(new Action(() => ApplyLiveRoute(routeResult)));
                }
                _liveLogger?.Write("rx", message);
            }
            catch (Exception ex)
            {
                vehicle.LastError = ex.Message;
            }
        }

        /// <summary>True for the exceptions a deliberate disconnect produces.</summary>
        private static bool IsBenignDisconnect(Exception ex) =>
            ex is OperationCanceledException ||
            ex is ObjectDisposedException ||
            ex is System.IO.IOException && ex.InnerException is System.Net.Sockets.SocketException;

        private void OnLiveHello(LiveVehicleVisual vehicle, HelloMessage hello)
        {
            // Runs on the receive thread. Only plain fields are touched here; every WPF call is
            // marshalled to the dispatcher, otherwise the exception would escape into the client's
            // read loop and tear the connection down.
            var duplicate = false;
            try
            {
                duplicate = !string.IsNullOrWhiteSpace(hello.Vehicle) &&
                            _liveVehicles.Values.Any(v => !ReferenceEquals(v, vehicle) && v.VehicleId == hello.Vehicle);
                vehicle.VehicleId = hello.Vehicle ?? string.Empty;
                vehicle.Proto = hello.Proto;
                vehicle.LastMessageAt = LiveNowSeconds();
                vehicle.LastError = string.Empty;
            }
            catch (Exception ex)
            {
                vehicle.LastError = ex.Message;
                return;
            }

            if (duplicate)
                LogLine($"[Live] warning: duplicate vehicle id '{hello.Vehicle}' on {vehicle.SlotKey}");
            _liveHelloFrame = string.IsNullOrWhiteSpace(hello.Frame) ? "map" : hello.Frame;
            var announced = FrameCheck.Parse(hello.Frame);
            vehicle.AnnouncedFrame = announced;
            // Record what the bridge says it can do. Asking for a route the far end does not
            // implement used to fail silently; now it is decided before the request goes out.
            vehicle.Caps = hello.Caps ?? Array.Empty<string>();
            // Both layers must be looking at the same map. If the ids disagree, a route would be
            // drawn on one scene while the vehicle sits in another.
            if (!string.IsNullOrWhiteSpace(hello.Scene) && !string.IsNullOrWhiteSpace(SceneId) &&
                !string.Equals(hello.Scene, SceneId, StringComparison.Ordinal))
            {
                LogLineOnce("[live] 场景不一致: 桥=" + hello.Scene + " 控制台=" + SceneId, "scene-mismatch");
            }
            else if (!string.IsNullOrWhiteSpace(hello.Scene))
            {
                LogLine("[live] 场景一致: " + hello.Scene);
            }

            Dispatcher.BeginInvoke(new Action(() =>
            {
                try
                {
                    ApplyFrameGate();
                    Panel.SetZIndex(vehicle.Group, 100 + _liveVehicles.Count);
                    RefreshLiveFleetPanel();
                }
                catch (Exception ex)
                {
                    vehicle.LastError = ex.Message;
                }
            }));
        }

        /// <summary>
        /// Refuses to project poses that cannot be placed on the map yet.
        /// <para>
        /// A chassis- or odom-frame pose drawn without a calibration produces a marker that
        /// looks right and is wrong; the user cannot tell by looking, so the layer is hidden
        /// and the reason is put in the status bar instead.
        /// </para>
        /// </summary>
        private readonly System.Collections.Generic.HashSet<string> _liveLoggedOnce = new();

        /// <summary>Logs a line at most once per key, so a per-tick condition does not flood the log.</summary>
        /// <param name="line">The line to log.</param>
        /// <param name="key">The deduplication key.</param>
        private void LogLineOnce(string line, string key)
        {
            if (!_liveLoggedOnce.Add(key)) return;
            LogLine(line);
        }

        private void ApplyFrameGate()
        {
            foreach (var vehicle in _liveVehicles.Values)
            {
                var blocked = vehicle.Client != null
                    ? FrameCheck.Describe(vehicle.AnnouncedFrame, _liveCalibration != null)
                    : null;
                vehicle.FrameBlocked = blocked != null;
                vehicle.FrameBlockMessage = blocked ?? string.Empty;
            }

            // The status line is rebuilt on every tick from the vehicle state, so the message
            // lives on the vehicle record (FrameBlockMessage) instead of being written here;
            // writing it here would be invisible after the next tick.
            foreach (var v in _liveVehicles.Values.Where(v => v.FrameBlocked))
                LogLineOnce("[Live] " + v.FrameBlockMessage, v.SlotKey + "|" + v.FrameBlockMessage);
        }

        private async System.Threading.Tasks.Task SendLiveAsync(LiveMessage message)
        {
            var active = ActiveLiveVehicle();
            if (active?.Client == null) throw new InvalidOperationException("未连接");
            await active.Client.SendAsync(message);
            _liveLogger?.Write("tx", message);
        }

        private void OpenLiveLogger()
        {
            if (_liveLogger != null) return;
            _liveLogger = new JsonlPoseLogger(System.IO.Path.Combine("logs", "live"), "live");
            _liveLogger.WriteSessionHeader(new
            {
                version = 2,
                vehicles = _liveVehicles.Values
                    .Select(v => new { slot = v.SlotKey, vehicle = v.VehicleId, color = $"#{v.Color.R:X2}{v.Color.G:X2}{v.Color.B:X2}" })
                    .ToList(),
                calibration = _liveCalibration?.ToJson(false),
                timerMs = 33,
            });
        }

        private void CloseLiveLogger()
        {
            _liveLogger?.Dispose();
            _liveLogger = null;
        }

        private void DisposeLiveLayer()
        {
            _liveTimer?.Stop();
            _liveTimer = null;
            foreach (var vehicle in _liveVehicles.Values)
            {
                try { vehicle.Client?.StopAsync().GetAwaiter().GetResult(); } catch { }
                vehicle.Client = null;
            }
            _liveVehicles.Clear();
            _liveColors.Clear();
            CloseLiveLogger();

            // The scheduling stream is a second, independent socket; it would otherwise keep the
            // process alive after the window is gone and hold the port for the next run.
            try { _scheduleClient?.StopAsync().GetAwaiter().GetResult(); } catch { }
            _scheduleClient = null;
        }

        private sealed class LiveWaypointChoice
        {
            public LiveWaypointChoice(int id, string label)
            {
                Id = id;
                Label = label;
            }

            public int Id { get; }
            public string Label { get; }
            public override string ToString() => Label;
        }

        private static Brush MakeFrozenBrush(byte r, byte g, byte b)
        {
            var brush = new SolidColorBrush(Color.FromRgb(r, g, b));
            brush.Freeze();
            return brush;
        }

        private static Brush MakeFrozenBrush(byte r, byte g, byte b, byte a)
        {
            var brush = new SolidColorBrush(Color.FromArgb(a, r, g, b));
            brush.Freeze();
            return brush;
        }
    }
}
