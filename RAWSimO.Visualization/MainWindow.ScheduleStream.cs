using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using System.Windows.Threading;
using RAWSimO.Live.Networking;
using RAWSimO.Live.Protocol;
using RAWSimO.Live.Tracking;

namespace RAWSimO.Visualization
{
    /// <summary>
    /// Live view of the scheduling stream: every AGV of the run, as the scheduler sees it.
    /// <para>
    /// This is a second, independent layer rather than part of the vehicle overlay, because it
    /// is a different producer in a different frame. The scheduling stream reports workshop
    /// metres across a network some 120 m wide; the single-vehicle bridge reports a few metres of
    /// bench. Sharing one canvas transform between them would put one of the two off screen, and
    /// sharing one colour table would make "AGV00" and the bridge vehicle look like the same
    /// robot. Neither mistake announces itself, so they are kept apart here instead.
    /// </para>
    /// <para>
    /// State is written only by Python. An explicit owner capability enables operator requests.
    /// </para>
    /// </summary>
    public partial class MainWindow
    {
        private Canvas _scheduleOverlay;
        private TextBox _scheduleHostBox;
        private TextBox _schedulePortBox;
        private Button _scheduleConnectButton;
        private Button _scheduleDisconnectButton;
        private TextBlock _scheduleStatusText;
        private TextBlock _scheduleSummaryText;
        private TabItem _scheduleTab;

        private LiveTcpClient _scheduleClient;
        private FleetTopology _scheduleTopology;
        private FleetScene _scheduleScene;
        /// <summary>Canvas units per screen pixel at the current zoom.</summary>
        /// <para>
        /// The overlay lives inside the same zoom/pane content as the map, so everything in it
        /// is multiplied by ContentScale. At the fitted scale for a 140x90 m workshop that is
        /// about 0.087: a 10 px body becomes 0.87 px and a 1.6 px route becomes 0.14 px, i.e.
        /// drawn correctly and invisible. Compensating with 1/ContentScale keeps the overlay
        /// readable at any zoom, which is the same rule LiveFleet already applies.
        /// </para>
        private double UiScale => zoomAndPanControl != null && zoomAndPanControl.ContentScale > 1e-6
            ? zoomAndPanControl.ContentScale
            : 1.0;

        /// <summary>Vehicle body size in canvas units, constant on screen.</summary>
        private double ScheduleBodySize => Math.Min(600.0, Math.Max(8.0, 10.0 / UiScale));

        private readonly Dictionary<string, Shape> _scheduleBodies = new(StringComparer.Ordinal);
        private readonly Dictionary<string, TextBlock> _scheduleLabels = new(StringComparer.Ordinal);
        private readonly Dictionary<string, Line> _scheduleHeadings = new(StringComparer.Ordinal);
        private FleetHelloMessage _scheduleHello;
        private FleetFrameMessage _scheduleLastFrame;
        private long _scheduleFrames;
        private double _scheduleFitWidth;
        private double _scheduleFitHeight;
        private bool _scheduleRedrawPending;

        /// <summary>
        /// The region of the canvas that is actually on screen. Vehicles are fitted into this,
        /// not into the whole canvas, because the control panel is drawn over the map and a
        /// vehicle underneath it is invisible even though it is drawn correctly.
        /// </summary>
        private Rect _scheduleVisibleRect;

        /// <summary>Where the control panel landed in canvas coordinates, for evidence.</summary>
        private Rect _schedulePanelRect = Rect.Empty;

        /// <summary>Builds the scheduling-stream panel and its overlay.</summary>
        private void InitializeScheduleStreamLayer()
        {
            _scheduleOverlay = new Canvas
            {
                Name = "ScheduleFleetOverlay",
                IsHitTestVisible = false,
                Background = null,
                Visibility = Visibility.Collapsed,
            };
            Panel.SetZIndex(_scheduleOverlay, 90);
            content.Children.Add(_scheduleOverlay);
            // A paused recording has no incoming frames to repair its projection.
            scroller.SizeChanged += (_, _) => QueueScheduleRedraw();
            _scheduleOverlay.SizeChanged += (_, _) => QueueScheduleRedraw();
            zoomAndPanControl.ContentScaleChanged += (_, _) => QueueScheduleRedraw();
            zoomAndPanControl.ContentOffsetXChanged += (_, _) => QueueScheduleRedraw();
            zoomAndPanControl.ContentOffsetYChanged += (_, _) => QueueScheduleRedraw();

            var panel = new StackPanel();
            var scroll = new ScrollViewer
            {
                Content = panel,
                VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
                HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled,
            };
            _scheduleTab = new TabItem { Name = "TabPythonSchedule", Header = "Python 调度", Content = scroll };
            ControlPanelTabs.Items.Add(_scheduleTab);
            _scheduleTab.SizeChanged += (_, _) => QueueScheduleRedraw();
            SizeChanged += (_, _) => scroll.MaxHeight = Math.Max(160, ActualHeight - 110);
            Loaded += (_, _) => scroll.MaxHeight = Math.Max(160, ActualHeight - 110);
            var group = new GroupBox
            {
                Name = "GroupBoxScheduleStream",
                Header = "Python 全队调度与回放",
                Margin = new Thickness(0, 4, 0, 0),
            };
            var stack = new StackPanel { Orientation = Orientation.Vertical };

            var row = new StackPanel { Orientation = Orientation.Horizontal };
            _scheduleHostBox = new TextBox { Name = "TextBoxScheduleHost", Text = "127.0.0.1", Width = 95, Height = 22 };
            row.Children.Add(_scheduleHostBox);
            row.Children.Add(new TextBlock { Text = " 端口: ", VerticalAlignment = VerticalAlignment.Center });
            _schedulePortBox = new TextBox { Name = "TextBoxSchedulePort", Text = "9500", Width = 45, Height = 22 };
            row.Children.Add(_schedulePortBox);
            _scheduleConnectButton = new Button { Name = "ButtonScheduleConnect", Content = "连接调度流", Margin = new Thickness(4, 0, 0, 0), Padding = new Thickness(8, 1, 8, 1) };
            _scheduleConnectButton.Click += ScheduleConnect_Click;
            row.Children.Add(_scheduleConnectButton);
            _scheduleDisconnectButton = new Button { Name = "ButtonScheduleDisconnect", Content = "断开", Margin = new Thickness(4, 0, 0, 0), Padding = new Thickness(8, 1, 8, 1), IsEnabled = false };
            _scheduleDisconnectButton.Click += ScheduleDisconnect_Click;
            row.Children.Add(_scheduleDisconnectButton);
            stack.Children.Add(row);

            _scheduleStatusText = new TextBlock
            {
                Name = "TextBlockScheduleStatus",
                Text = "状态: 未连接",
                Margin = new Thickness(0, 4, 0, 0),
                Foreground = Brushes.Gray,
                TextWrapping = TextWrapping.Wrap,
            };
            stack.Children.Add(_scheduleStatusText);

            _scheduleSummaryText = new TextBlock
            {
                Name = "TextBlockScheduleSummary",
                Text = "（无）",
                Margin = new Thickness(0, 2, 0, 0),
                FontFamily = new FontFamily("Consolas"),
                FontSize = 11,
                TextWrapping = TextWrapping.Wrap,
                MaxWidth = 520,
            };
            stack.Children.Add(new Expander { Header = "连接诊断", Content = _scheduleSummaryText, IsExpanded = false });

            InitializeScheduleOperations(stack);

            group.Content = stack;
            panel.Children.Add(group);
        }

        /// <summary>
        /// Connects the fleet stream without a click, for the command-line demo path.
        /// Splits "host:port" and drives the same code the button does, so the two entry
        /// points cannot drift apart.
        /// </summary>
        private void AutoConnectSchedule(string hostPort)
        {
            try
            {
                if (!TrySplitHostPort(hostPort, out var host, out var port))
                {
                    LogLine("[schedule] 地址无效: " + hostPort);
                    return;
                }
                if (_scheduleHostBox != null) _scheduleHostBox.Text = host;
                if (_schedulePortBox != null) _schedulePortBox.Text = port.ToString(CultureInfo.InvariantCulture);
                LogLine("[schedule] 自动连接 " + host + ":" + port);
                ScheduleConnect_Click(null, null);
            }
            catch (Exception ex)
            {
                LogLine("[schedule] 自动连接失败: " + ex.Message);
            }
        }

        /// <summary>Matches the live overlay's control names to the parsed arguments.</summary>
        private static bool TrySplitHostPort(string hostPort, out string host, out int port)
        {
            host = "127.0.0.1";
            port = 0;
            if (string.IsNullOrWhiteSpace(hostPort)) return false;
            var text = hostPort.Trim();
            var colon = text.LastIndexOf(':');
            if (colon <= 0 || colon == text.Length - 1) return false;
            host = text.Substring(0, colon);
            return int.TryParse(text.Substring(colon + 1), NumberStyles.Integer, CultureInfo.InvariantCulture, out port)
                && port > 0 && port < 65536;
        }

        private async void ScheduleConnect_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (_scheduleTab != null) _scheduleTab.IsSelected = true;
                StopScheduleReplay();
                var host = string.IsNullOrWhiteSpace(_scheduleHostBox.Text) ? "127.0.0.1" : _scheduleHostBox.Text.Trim();
                var port = int.TryParse(_schedulePortBox.Text, out var p) ? p : 9500;
                await ScheduleDisconnectAsync().ConfigureAwait(true);

                var client = new LiveTcpClient(new LiveClientOptions
                {
                    Host = host,
                    Port = port,
                    ConnectTimeout = TimeSpan.FromSeconds(2),
                    HelloTimeout = TimeSpan.FromSeconds(2),
                });
                client.FleetHelloReceived += hello => Dispatcher.BeginInvoke(new Action(() =>
                {
                    if (_scheduleClient == client) OnScheduleHello(hello);
                }));
                client.FleetFrameReceived += frame => Dispatcher.BeginInvoke(new Action(() =>
                {
                    if (_scheduleClient == client) OnScheduleFrame(frame);
                }));
                client.MessageReceived += message => Dispatcher.BeginInvoke(new Action(() =>
                {
                    if (_scheduleClient == client && message is FleetControlResultMessage result)
                        OnScheduleControlResult(result);
                }));
                client.StateChanged += state => Dispatcher.BeginInvoke(new Action(() =>
                {
                    if (_scheduleClient == client && state != LiveConnectionState.Connected)
                        SetScheduleControl(null);
                }));
                client.Error += ex => Dispatcher.BeginInvoke(new Action(() =>
                {
                    if (_scheduleClient != client) return;
                    _scheduleStatusText.Text = "状态: " + ex.GetType().Name + ": " + ex.Message;
                    _scheduleStatusText.Foreground = Brushes.Firebrick;
                }));
                _scheduleClient = client;
                await client.StartAsync().ConfigureAwait(true);
                _scheduleDisconnectButton.IsEnabled = true;
                _scheduleStatusText.Foreground = Brushes.Gray;
                _scheduleStatusText.Text = string.Format(CultureInfo.InvariantCulture,
                    "状态: 已连接 {0}:{1}，等待 hello", host, port);
            }
            catch (Exception ex)
            {
                _scheduleStatusText.Text = "状态: 连接失败 - " + ex.Message;
                _scheduleStatusText.Foreground = Brushes.Firebrick;
            }
        }

        private async void ScheduleDisconnect_Click(object sender, RoutedEventArgs e) =>
            await ScheduleDisconnectAsync().ConfigureAwait(true);

        private async System.Threading.Tasks.Task ScheduleDisconnectAsync()
        {
            SetScheduleControl(null);
            var client = _scheduleClient;
            _scheduleClient = null;
            if (client != null)
            {
                try { await client.StopAsync().ConfigureAwait(true); } catch { }
            }
            _scheduleTopology = null;
            _scheduleScene = null;
            _scheduleHello = null;
            _scheduleLastFrame = null;
            _scheduleFrames = 0;
            _scheduleFitWidth = 0;
            _scheduleFitHeight = 0;
            ClearScheduleVisuals();
            _scheduleOverlay.Visibility = Visibility.Collapsed;
            _scheduleDisconnectButton.IsEnabled = false;
            _scheduleStatusText.Text = "状态: 未连接";
            _scheduleStatusText.Foreground = Brushes.Gray;
            _scheduleSummaryText.Text = "（无）";
            _scheduleFrameAge.Reset();
            if (_scheduleAlerts != null) _scheduleAlerts.Text = "执行状态：未连接";
            if (_scheduleObservations != null) _scheduleObservations.Text = "";
        }

        /// <summary>
        /// Sizes the scene from the announced network and reports the fit, rather than assuming
        /// a canvas size. The hello is the only place the extent of the run is stated.
        /// </summary>
        private void OnScheduleHello(FleetHelloMessage hello)
        {
            SetScheduleControl(hello.Control);
            _scheduleHello = hello;
            _scheduleTopology = new FleetTopology(hello);
            _scheduleFrames = 0;
            _scheduleLastFrame = null;
            _scheduleFitWidth = _scheduleFitHeight = 0;
            _scheduleLoggedCongestion = false;
            _scheduleMaxCongestionSteps = _schedulePeakFull = _schedulePeakQueued = 0;
            ClearScheduleVisuals();
            _scheduleOverlay.Visibility = Visibility.Visible;
            _scheduleStatusText.Foreground = Brushes.Gray;
            RefitScheduleScene();
            if (_scheduleTopology.NodeCount == 0)
            {
                _scheduleStatusText.Foreground = Brushes.Firebrick;
                _scheduleStatusText.Text = "状态: 握手未宣布任何节点，无法作图";
                return;
            }
            LogLine(string.Format(CultureInfo.InvariantCulture,
                "[schedule] hello 节点={0} 边={1} 车={2} 任务={3} overlay={4:F0}x{5:F0} scene={6}",
                _scheduleTopology.NodeCount, _scheduleTopology.EdgeCount,
                hello.Meta.NAgvs, hello.Meta.NTasks,
                _scheduleOverlay.ActualWidth, _scheduleOverlay.ActualHeight,
                _scheduleScene is null ? "null" : "ok"));
            _scheduleStatusText.Text = string.Format(CultureInfo.InvariantCulture,
                "状态: 已握手 · {0} 节点 / {1} 边 · {2} 任务 / {3} 车 · {4:F2} px/m",
                _scheduleTopology.NodeCount, _scheduleTopology.EdgeCount,
                hello.Meta.NTasks, hello.Meta.NAgvs,
                _scheduleScene?.Transform.PixelsPerMeter ?? 0.0);
        }

        /// <summary>
        /// Rebuilds the projection so the announced network fills the canvas it is drawn on.
        /// <para>
        /// Called on hello and again whenever the overlay changes size, because the scale that
        /// makes a 120 m network visible depends on the window: a fixed pixels-per-metre either
        /// shrinks the fleet into a corner or puts it off the canvas entirely, and neither
        /// failure announces itself.
        /// </para>
        /// </summary>
        private void RefitScheduleScene()
        {
            if (_scheduleHello is null || _scheduleOverlay is null) return;
            // The overlay is only sized when the vehicle layer's timer ticks, and that timer does
            // not run when no single-vehicle bridge is connected -- which is the normal case for
            // a scheduling-stream-only session. Fall back to the container the overlay lives in,
            // otherwise the scene is never built and every frame is dropped in silence.
            //
            // Fit to the part of the map that is actually on screen.
            //
            // Three separate mistakes met here, and each one alone produced the same symptom --
            // a vehicle that is drawn, positioned correctly, and not in the picture:
            //
            // 1. Setting Width/Height on a Canvas freezes its ActualWidth/ActualHeight, so a
            //    size measured once stayed forever. A Canvas left at its default width is NaN,
            //    and `Math.Abs(NaN - w) > 1.0` is false, so a "has it changed?" guard written
            //    that way never assigns anything at all and the canvas silently stays 0x0.
            // 2. The drawing surface is `theGrid`, fixed at 1000x1000, and the map pane clips it.
            //    Anything the fit places past the pane is cut away no matter how much window
            //    there is.
            // 3. The control panel is drawn *over* the map, not beside it, and it is opaque
            //    across the full height of the pane. A vehicle under it is drawn correctly and
            //    still invisible, and nothing anywhere reports an error.
            //
            // So the fit uses the intersection of the surface, the pane and the region the panel
            // does not cover, and the overlay itself is moved onto that rectangle -- which makes
            // every projected coordinate relative to what can actually be seen.
            var visible = ScheduleVisibleRect();
            if (visible.IsEmpty) return;
            // (The panel rectangle was recorded inside ScheduleVisibleRect; clearing it here
            // would erase the very value the report is meant to show.)
            _scheduleVisibleRect = visible;
            var w = visible.Width;
            var h = visible.Height;
            if (!(w > 1) || !(h > 1)) return;
            if (double.IsNaN(_scheduleOverlay.Width) || Math.Abs(_scheduleOverlay.Width - w) > 1.0)
                _scheduleOverlay.Width = w;
            if (double.IsNaN(_scheduleOverlay.Height) || Math.Abs(_scheduleOverlay.Height - h) > 1.0)
                _scheduleOverlay.Height = h;
            if (double.IsNaN(Canvas.GetLeft(_scheduleOverlay)) || Math.Abs(Canvas.GetLeft(_scheduleOverlay) - visible.Left) > 1.0)
                Canvas.SetLeft(_scheduleOverlay, visible.Left);
            if (double.IsNaN(Canvas.GetTop(_scheduleOverlay)) || Math.Abs(Canvas.GetTop(_scheduleOverlay) - visible.Top) > 1.0)
                Canvas.SetTop(_scheduleOverlay, visible.Top);
            if (Math.Abs(w - _scheduleFitWidth) < 1.0 && Math.Abs(h - _scheduleFitHeight) < 1.0 &&
                _scheduleScene != null) return;

            var fitted = FleetScene.FitNetwork(_scheduleHello, w, h);
            if (fitted is null) return;
            _scheduleScene = fitted;
            _scheduleFitWidth = w;
            _scheduleFitHeight = h;

            // Every drawn shape was positioned with the old scale, so all of them have to go.
            ClearScheduleVisuals();
            DrawScheduleNetwork(_scheduleHello);
        }

        /// <summary>Release every cached visual removed with the old canvas contents.</summary>
        private void ClearScheduleVisuals()
        {
            _scheduleOverlay.Children.Clear();
            _scheduleBodies.Clear();
            _scheduleLabels.Clear();
            _scheduleHeadings.Clear();
            _scheduleTrailLines.Clear();
            _scheduleRouteLines.Clear();
            _scheduleCongestionRings.Clear();
            _scheduleCongestionLabels.Clear();
            _schedulePhysicalMarker = null;
            _schedulePhysicalTrail = null;
            _schedulePhysicalLabel = null;
            _scheduleTrailPoints = _schedulePhysicalPoints = 0;
        }

        /// <summary>Repaint an observation after layout without making old feedback fresh.</summary>
        private void QueueScheduleRedraw()
        {
            if (_scheduleRedrawPending || _scheduleLastFrame == null) return;
            _scheduleRedrawPending = true;
            Dispatcher.BeginInvoke(DispatcherPriority.ContextIdle, new Action(() =>
            {
                _scheduleRedrawPending = false;
                if (_scheduleLastFrame == null) return;
                RenderScheduleFrame(_scheduleLastFrame);
                RefreshScheduleOperations();
            }));
        }

        /// <summary>
        /// The part of the map canvas a person can actually see: what the scroll pane shows,
        /// minus the opaque control panel drawn over its left side.
        /// <para>
        /// This exists because "the vehicle is on the canvas" and "the vehicle is on the screen"
        /// are different statements, and only the second one is a demo. A shape under the control
        /// panel is drawn at a correct position, in a correct colour, and is invisible; nothing
        /// in the drawing code can tell the difference, so the fit has to be told where the
        /// visible region is instead of assuming the canvas is all of it.
        /// </para>
        /// </summary>
        /// <returns>The visible rectangle in the overlay's own coordinates, or an empty rect.</returns>
        private Rect ScheduleVisibleRect()
        {
            if (content is null || scroller is null) return Rect.Empty;
            if (!(content.ActualWidth > 1) || !(content.ActualHeight > 1)) return Rect.Empty;
            if (!(scroller.ActualWidth > 1) || !(scroller.ActualHeight > 1)) return Rect.Empty;
            try
            {
                _schedulePanelRect = Rect.Empty;
                var back = content.TransformToAncestor(this).Inverse;
                if (back is null) return Rect.Empty;

                // The pane, expressed in the canvas's own coordinates. The map may be scaled or
                // panned, so this goes through the transform rather than assuming 1:1.
                var paneInWindow = scroller.TransformToAncestor(this)
                    .TransformBounds(new Rect(0, 0, scroller.ActualWidth, scroller.ActualHeight));
                var visible = back.TransformBounds(paneInWindow);
                visible.Intersect(new Rect(0, 0, content.ActualWidth, content.ActualHeight));
                if (visible.IsEmpty || !(visible.Width > 1) || !(visible.Height > 1)) return Rect.Empty;

                // The control panel covers the left side of the pane for its full height, so
                // everything left of its right edge is unreachable however correctly it is drawn.
                var panel = ScheduleControlPanel();
                if (panel is not null && panel.ActualWidth > 1 && panel.ActualHeight > 1)
                {
                    var panelInWindow = panel.TransformToAncestor(this)
                        .TransformBounds(new Rect(0, 0, panel.ActualWidth, panel.ActualHeight));
                    var panelInCanvas = back.TransformBounds(panelInWindow);
                    _schedulePanelRect = panelInCanvas;
                    // The panel carries a small margin, so "against the left edge" is a
                    // tolerance, not an equality.
                    var startsAtLeftEdge = panelInCanvas.Left <= visible.Left + 16.0;
                    // No "does it cover the full height?" test. The strip to the right of the
                    // panel's edge is what is readable, and the panel is tall enough to hide a
                    // vehicle anywhere in the left half; requiring full coverage only produced a
                    // branch that never ran. Reserving the whole strip costs a little width --
                    // the map is drawn a little smaller than it could be -- and that is the
                    // cheaper mistake. Losing a vehicle costs the demo.
                    if (startsAtLeftEdge && panelInCanvas.Right > visible.Left)
                    {
                        var left = Math.Min(panelInCanvas.Right, visible.Right);
                        visible = new Rect(left, visible.Top, visible.Right - left, visible.Height);
                    }
                }
                // The log console overlaps the bottom of this same map viewport.
                // Fit above it so a vehicle on a boundary node stays visible.
                if (TextBoxOutput.IsVisible && TextBoxOutput.ActualHeight > 1)
                {
                    var consoleInWindow = TextBoxOutput.TransformToAncestor(this)
                        .TransformBounds(new Rect(0, 0, TextBoxOutput.ActualWidth, TextBoxOutput.ActualHeight));
                    var consoleInCanvas = back.TransformBounds(consoleInWindow);
                    if (consoleInCanvas.Left <= visible.Left && consoleInCanvas.Right >= visible.Right
                        && consoleInCanvas.Bottom >= visible.Bottom && consoleInCanvas.Top < visible.Bottom)
                    {
                        var bottom = Math.Max(visible.Top, consoleInCanvas.Top);
                        visible = new Rect(visible.Left, visible.Top, visible.Width, bottom - visible.Top);
                    }
                }
                return visible;
            }
            catch (InvalidOperationException)
            {
                // Not connected to a presentation source yet; nothing has been laid out.
                return Rect.Empty;
            }
        }

        /// <summary>The control panel that is drawn over the map, found from one of its tabs.</summary>
        private FrameworkElement? ScheduleControlPanel()
        {
            DependencyObject? node = TabEvents;
            while (node is not null && node is not TabControl)
                node = LogicalTreeHelper.GetParent(node) ?? VisualTreeHelper.GetParent(node);
            return node as FrameworkElement;
        }

        private void DrawScheduleNetwork(FleetHelloMessage hello)
        {
            if (_scheduleScene == null) return;
            var edgeBrush = new SolidColorBrush(Color.FromRgb(0xD0, 0xD0, 0xD0));
            edgeBrush.Freeze();
            var nodeBrush = new SolidColorBrush(Color.FromRgb(0x90, 0x90, 0x90));
            nodeBrush.Freeze();
            foreach (var edge in hello.Edges)
            {
                var road = _scheduleTopology.EdgePoints(edge.Src, edge.Dst);
                if (road is null) continue;
                var vertices = new PointCollection();
                foreach (var vertex in road)
                {
                    var p = _scheduleScene.Project(vertex.X, vertex.Y);
                    vertices.Add(new Point(p.X, p.Y));
                }
                _scheduleOverlay.Children.Add(new Polyline
                {
                    Points = vertices,
                    Stroke = edgeBrush,
                    StrokeThickness = 1.0,
                });
            }
            foreach (var node in hello.Nodes)
            {
                var p = _scheduleScene.Project(node.X, node.Y);
                var dot = new Ellipse { Width = 3, Height = 3, Fill = nodeBrush };
                Canvas.SetLeft(dot, p.X - 1.5);
                Canvas.SetTop(dot, p.Y - 1.5);
                _scheduleOverlay.Children.Add(dot);
            }
        }

        /// <summary>
        /// Renders the per-vehicle route the scheduler sent with the frame.
        /// <para>
        /// The node ids in <see cref="FleetAgv.Path"/> are resolved against the topology from
        /// the hello message, so a route that names a node this client does not know is
        /// dropped for that vehicle rather than guessed at. Nothing here plans: if the
        /// scheduler did not send a path, no line is drawn.
        /// </para>
        /// </summary>
        /// <summary>Draws where each vehicle actually was, underneath the planned route.</summary>
        /// <para>
        /// The two lines answer different questions: the planned route is what Python
        /// decided, the trail is what really happened. Drawn on one screen, the gap between
        /// them — a detour, a wait, a missed waypoint — stops being invisible.
        /// </para>
        /// <summary>Draws the physical robot's ground truth as its own entity.</summary>
        /// <para>
        /// Hollow square, not a filled circle: every simulated vehicle is a circle, so a
        /// different shape is the one cue that survives a black-and-white screenshot. Its
        /// trail is a separate line, and when the truth feed stops the pose disappears while
        /// the trail stays — the road already driven is a fact and should not be erased.
        /// </para>
        private void DrawSchedulePhysical(FleetFrameMessage frame)
        {
            if (_scheduleTopology is null || _scheduleScene is null || _scheduleOverlay is null) return;

            var physical = frame.Physical;
            if (physical is null)
            {
                if (_schedulePhysicalMarker != null) _schedulePhysicalMarker.Visibility = Visibility.Collapsed;
                if (_schedulePhysicalLabel != null) _schedulePhysicalLabel.Visibility = Visibility.Collapsed;
                if (_schedulePhysicalTrail != null) _schedulePhysicalTrail.Visibility = Visibility.Collapsed;
                _schedulePhysicalPoints = 0;
                return;
            }

            if (_schedulePhysicalTrail == null)
            {
                _schedulePhysicalTrail = new Polyline
                {
                    // Solid and thicker: the fleet trails are thin dashes in per-vehicle colours,
                    // one of which is orange, so a dashed orange line is indistinguishable from
                    // an AGV route. Weight and dash pattern carry the distinction even in
                    // greyscale, where colour does not.
                    Stroke = Brushes.OrangeRed,
                    IsHitTestVisible = false,
                };
                Panel.SetZIndex(_schedulePhysicalTrail, 91);
                _scheduleOverlay.Children.Add(_schedulePhysicalTrail);

                _schedulePhysicalMarker = new System.Windows.Shapes.Rectangle
                {
                    Stroke = Brushes.OrangeRed,
                    Fill = Brushes.Transparent,
                    IsHitTestVisible = false,
                };
                Panel.SetZIndex(_schedulePhysicalMarker, 97);
                _scheduleOverlay.Children.Add(_schedulePhysicalMarker);

                _schedulePhysicalLabel = new TextBlock
                {
                    Background = Brushes.White,
                    Foreground = Brushes.OrangeRed,
                    IsHitTestVisible = false,
                };
                Panel.SetZIndex(_schedulePhysicalLabel, 98);
                _scheduleOverlay.Children.Add(_schedulePhysicalLabel);
            }

            var pts = new PointCollection();
            foreach (var p in physical.Trail)
            {
                var q = _scheduleScene.Project(p.X, p.Y);
                pts.Add(new Point(q.X, q.Y));
            }
            if (pts.Count >= 2)
            {
                _schedulePhysicalTrail.Points = pts;
                _schedulePhysicalTrail.StrokeThickness = Math.Min(20.0, Math.Max(1.2, 2.6 / UiScale));
                _schedulePhysicalTrail.Visibility = Visibility.Visible;
            }
            else
            {
                _schedulePhysicalTrail.Visibility = Visibility.Collapsed;
            }
            _schedulePhysicalPoints = pts.Count;

            if (physical.Pose is { Length: >= 2 })
            {
                var marker = _scheduleScene.Project(physical.Pose[0], physical.Pose[1]);
                var box = Math.Min(600.0, Math.Max(8.0, 14.0 / UiScale));
                _schedulePhysicalMarker.Width = box;
                _schedulePhysicalMarker.Height = box;
                _schedulePhysicalMarker.StrokeThickness = Math.Min(8.0, Math.Max(0.8, 2.0 / UiScale));
                Canvas.SetLeft(_schedulePhysicalMarker, marker.X - box / 2);
                Canvas.SetTop(_schedulePhysicalMarker, marker.Y - box / 2);
                _schedulePhysicalMarker.Visibility = Visibility.Visible;

                // One line per 100 frames: "drawn off screen" and "not drawn" look identical in
                // a screenshot, so the canvas coordinate has to come out of the window.
                if (_scheduleFrames % 100 == 0)
                {
                    LogLine(string.Format(CultureInfo.InvariantCulture,
                        "[schedule] phys {0} pose=({1:F2},{2:F2}) canvas=({3:F1},{4:F1}) trail={5}",
                        physical.Id, physical.Pose[0], physical.Pose[1],
                        marker.X, marker.Y, _schedulePhysicalPoints));
                }
                _schedulePhysicalLabel.Text = physical.Id + " 实物(" + physical.Source + ")";
                _schedulePhysicalLabel.FontSize = Math.Min(160.0, Math.Max(10.0, 10.0 / UiScale));
                Canvas.SetLeft(_schedulePhysicalLabel, marker.X + box / 2 + 2.0 / UiScale);
                Canvas.SetTop(_schedulePhysicalLabel, marker.Y - box / 2);
                _schedulePhysicalLabel.Visibility = Visibility.Visible;
            }
            else
            {
                _schedulePhysicalMarker.Visibility = Visibility.Collapsed;
                _schedulePhysicalLabel.Visibility = Visibility.Collapsed;
            }
        }

        /// <summary>One label for the whole frame, listing the distinct provenances it carries.</summary>
        private static string DescribeTrailSources(FleetFrameMessage frame)
        {
            var sources = frame.TrailSources?.Values ?? Enumerable.Empty<string>();
            var distinct = sources.Where(s => !string.IsNullOrWhiteSpace(s))
                                  .Distinct(StringComparer.Ordinal).OrderBy(s => s, StringComparer.Ordinal)
                                  .ToList();
            if (distinct.Count == 0) return frame.TrailSource ?? "-";
            return string.Join("+", distinct);
        }

        private System.Windows.Shapes.Rectangle _schedulePhysicalMarker;
        private Polyline _schedulePhysicalTrail;
        private TextBlock _schedulePhysicalLabel;
        private int _schedulePhysicalPoints;

        /// <summary>Whether the first sustained-congestion frame has been logged, printed once.</summary>
        private bool _scheduleLoggedCongestion;

        /// <summary>Longest run of consecutive full steps any single node showed this session.</summary>
        private int _scheduleMaxCongestionSteps;

        private void DrawScheduleTrails(FleetFrameMessage frame)
        {
            if (_scheduleTopology is null || _scheduleScene is null || _scheduleOverlay is null) return;

            var trails = frame.Trails;
            if (trails is null)
            {
                foreach (var stale in _scheduleTrailLines.Values) stale.Visibility = Visibility.Collapsed;
                _scheduleTrailPoints = 0;
                return;
            }

            var points = 0;
            foreach (var vehicle in trails)
            {
                if (!_scheduleTrailLines.TryGetValue(vehicle.Key, out var line))
                {
                    line = new Polyline
                    {
                        StrokeThickness = 1.0,
                        Stroke = Brushes.SteelBlue,
                        StrokeDashArray = new DoubleCollection { 1, 2 },
                        IsHitTestVisible = false,
                    };
                    Panel.SetZIndex(line, 90);
                    _scheduleTrailLines[vehicle.Key] = line;
                    _scheduleOverlay.Children.Add(line);
                }

                line.StrokeThickness = Math.Min(10.0, Math.Max(0.5, 1.0 / UiScale));
                var pts = new PointCollection();
                foreach (var p in vehicle.Value)
                {
                    var q = _scheduleScene.Project(p.X, p.Y);
                    pts.Add(new Point(q.X, q.Y));
                    points++;
                }
                if (pts.Count < 2)
                {
                    line.Visibility = Visibility.Collapsed;
                    continue;
                }
                line.Points = pts;
                line.Visibility = Visibility.Visible;
            }

            foreach (var gone in _scheduleTrailLines.Keys.Where(k => !trails.ContainsKey(k)).ToList())
            {
                _scheduleOverlay.Children.Remove(_scheduleTrailLines[gone]);
                _scheduleTrailLines.Remove(gone);
            }
            _scheduleTrailPoints = points;
        }

        private readonly Dictionary<string, Polyline> _scheduleTrailLines = new(StringComparer.Ordinal);

        /// <summary>Trail points drawn in the last frame, for the checkable summary line.</summary>
        private int _scheduleTrailPoints;

        private void DrawScheduleRoutes(FleetFrameMessage frame)
        {
            if (_scheduleTopology is null || _scheduleScene is null || _scheduleOverlay is null) return;

            foreach (var agv in frame.Agvs)
            {
                if (!_scheduleRouteLines.TryGetValue(agv.AgvId, out var line))
                {
                    line = new Polyline
                    {
                        StrokeDashArray = new DoubleCollection { 3, 2 },
                        IsHitTestVisible = false,
                    };
                    Panel.SetZIndex(line, 95);
                    _scheduleRouteLines[agv.AgvId] = line;
                    _scheduleOverlay.Children.Add(line);
                }

                line.StrokeThickness = Math.Min(12.0, Math.Max(0.6, 1.6 / UiScale));
                var path = agv.Path;
                if (path == null || path.Count == 0)
                {
                    line.Visibility = Visibility.Collapsed;
                    continue;
                }

                var points = new PointCollection();
                var road = _scheduleTopology.RoutePoints(path);
                if (road is not null) foreach (var vertex in road)
                {
                    var p = _scheduleScene.Project(vertex.X, vertex.Y);
                    points.Add(new Point(p.X, p.Y));
                }
                if (points.Count < 2)
                {
                    line.Visibility = Visibility.Collapsed;
                    continue;
                }

                if (_scheduleBodies.TryGetValue(agv.AgvId, out var body) && body.Fill is SolidColorBrush fill)
                    line.Stroke = fill;
                else
                    line.Stroke = Brushes.DimGray;
                line.Points = points;
                line.Visibility = Visibility.Visible;
            }

            // A route whose vehicle disappeared must disappear with it.
            foreach (var gone in _scheduleRouteLines.Keys
                .Where(k => frame.Agvs.All(a => !string.Equals(a.AgvId, k, StringComparison.Ordinal)))
                .ToList())
            {
                _scheduleOverlay.Children.Remove(_scheduleRouteLines[gone]);
                _scheduleRouteLines.Remove(gone);
            }
        }

        /// <summary>The dashed route line drawn for each vehicle, keyed by AGV id.</summary>
        private readonly Dictionary<string, Polyline> _scheduleRouteLines = new(StringComparer.Ordinal);

        /// <summary>
        /// Draws the nodes Python reports as at or over capacity, with the claim count.
        /// <para>
        /// The numbers come from the frame's <c>congestion</c> map, which Python builds from its
        /// reservation table. Nothing is inferred here: a node without an entry is not drawn,
        /// and a node whose claims fall below capacity has its ring removed on the next frame.
        /// </para>
        /// </summary>
        private void DrawScheduleCongestion(FleetFrameMessage frame)
        {
            if (_scheduleTopology is null || _scheduleScene is null || _scheduleOverlay is null) return;

            var congestion = frame.Congestion;
            if (congestion is null || congestion.Count == 0)
            {
                foreach (var ring in _scheduleCongestionRings.Values)
                {
                    _scheduleOverlay.Children.Remove(ring);
                }
                _scheduleCongestionRings.Clear();
                _scheduleCongestionLabels.Clear();
                return;
            }

            var live = new HashSet<string>(StringComparer.Ordinal);
            foreach (var pair in congestion)
            {
                var nodeId = pair.Key;
                var info = pair.Value;
                var node = _scheduleTopology.Node(nodeId);
                if (node is null) continue;   // a node this client does not know is not guessed at
                live.Add(nodeId);
                var p = _scheduleScene.Project(node.X, node.Y);

                // Two different facts share this payload, and drawing them the same way would
                // claim more than is known: a bay with every slot taken is full, but nobody is
                // waiting; only a node with a blocked claimant is a queue. Measured on the
                // workshop scene over 1000 steps (6 AGVs, 20 tasks, seed 0): 254 full
                // node-steps, 213 of them parking bays, and only 29 with a blocked claimant
                // -- every one of those at a junction or buffer, none at a bay.
                var waiting = info.Blocked > 0;
                if (!_scheduleCongestionRings.TryGetValue(nodeId, out var ring))
                {
                    ring = new Ellipse
                    {
                        Stroke = Brushes.OrangeRed,
                        StrokeThickness = 2.0,
                        Fill = null,
                        IsHitTestVisible = false,
                    };
                    Panel.SetZIndex(ring, 90);
                    _scheduleCongestionRings[nodeId] = ring;
                    _scheduleOverlay.Children.Add(ring);

                    var label = new TextBlock
                    {
                        Foreground = waiting ? Brushes.Red : Brushes.DarkGoldenrod,
                        Background = Brushes.White,
                        FontSize = 10,
                        FontWeight = FontWeights.Bold,
                        Padding = new Thickness(1, 0, 1, 0),
                        IsHitTestVisible = false,
                    };
                    Panel.SetZIndex(label, 91);
                    _scheduleCongestionLabels[nodeId] = label;
                    _scheduleOverlay.Children.Add(label);
                }

                ring.Stroke = waiting ? Brushes.Red : Brushes.Goldenrod;
                ring.StrokeThickness = Math.Min(14.0, Math.Max(0.6, (waiting ? 3.0 : 1.2) / UiScale));
                var ringSize = Math.Min(200.0, Math.Max(6.0, 22.0 / UiScale));
                if (Math.Abs(ring.Width - ringSize) > 0.5)
                {
                    ring.Width = ringSize;
                    ring.Height = ringSize;
                }
                if (_scheduleCongestionRings.Count > _schedulePeakFull) _schedulePeakFull = _scheduleCongestionRings.Count;
                if (waiting && _scheduleCongestionRings.Count > _schedulePeakQueued) _schedulePeakQueued = _scheduleCongestionRings.Count;
                Canvas.SetLeft(ring, p.X - ringSize / 2);
                Canvas.SetTop(ring, p.Y - ringSize / 2);
                ring.Visibility = _scheduleScene.IsOffCanvas(new FleetMarker(
                    string.Empty, p.X, p.Y, 255, 0, 0, string.Empty, null, false))
                    ? Visibility.Collapsed
                    : Visibility.Visible;

                if (_scheduleCongestionLabels.TryGetValue(nodeId, out var text))
                {
                    text.Foreground = waiting ? Brushes.Red : Brushes.DarkGoldenrod;
                    text.Text = waiting
                        ? string.Format(CultureInfo.InvariantCulture, "{0} 满员 {1}/{2} 阻塞{3} 持续{4}步",
                            nodeId, info.Claims, info.Capacity, info.Blocked, info.Steps)
                        : string.Format(CultureInfo.InvariantCulture, "{0} 满员 {1}/{2} 持续{3}步",
                            nodeId, info.Claims, info.Capacity, info.Steps);
                    text.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
                    var left = p.X + ringSize / 2 + 2.0 / UiScale;
                    if (left + text.DesiredSize.Width > _scheduleOverlay.ActualWidth)
                        left = Math.Max(0, p.X - 12 - text.DesiredSize.Width);
                    Canvas.SetLeft(text, left);
                    Canvas.SetTop(text, Math.Max(0, p.Y - 16));
                    text.Visibility = ring.Visibility;
                }
            }

            foreach (var gone in _scheduleCongestionRings.Keys.Where(k => !live.Contains(k)).ToList())
            {
                _scheduleOverlay.Children.Remove(_scheduleCongestionRings[gone]);
                _scheduleCongestionRings.Remove(gone);
                if (_scheduleCongestionLabels.TryGetValue(gone, out var lbl))
                {
                    _scheduleOverlay.Children.Remove(lbl);
                    _scheduleCongestionLabels.Remove(gone);
                }
            }
        }

        /// <summary>Largest number of full nodes drawn at once over the session.</summary>
        private int _schedulePeakFull;

        /// <summary>Largest number of nodes with a blocked claimant drawn at once.</summary>
        private int _schedulePeakQueued;

        /// <summary>Congestion rings by node id, from the scheduler's reservation table.</summary>
        private readonly Dictionary<string, Ellipse> _scheduleCongestionRings = new(StringComparer.Ordinal);

        /// <summary>The "node claims/capacity" caption drawn beside each ring.</summary>
        private readonly Dictionary<string, TextBlock> _scheduleCongestionLabels = new(StringComparer.Ordinal);

        private void OnScheduleFrame(FleetFrameMessage frame)
        {
            _scheduleFrameAge.Restart();
            SetScheduleControl(frame.Control);
            _scheduleFrames++;
            _scheduleLastFrame = frame;
            RenderScheduleFrame(frame);
        }

        private void RenderScheduleFrame(FleetFrameMessage frame)
        {
            if (_scheduleTopology is null) return;
            // The overlay is sized by the layout, which happens after the hello arrives, and the
            // window can be resized at any time. Refit before the "no scene" test, otherwise the
            // first frames are dropped and a working stream looks like a dead one.
            RefitScheduleScene();
            if (_scheduleScene is null)
            {
                // Dropping every frame in silence is how a working stream looks dead. Say why.
                LogLineOnce(string.Format(CultureInfo.InvariantCulture,
                    "[schedule] 帧被丢弃: 场景未建立 overlay={0:F0}x{1:F0}",
                    _scheduleOverlay.ActualWidth, _scheduleOverlay.ActualHeight), "schedule-no-scene");
                return;
            }
            var placed = _scheduleTopology.Place(frame);
            var markers = _scheduleScene.Project(placed);

            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var marker in markers)
            {
                seen.Add(marker.AgvId);
                if (!_scheduleBodies.TryGetValue(marker.AgvId, out var body))
                {
                    var brush = new SolidColorBrush(Color.FromRgb(marker.R, marker.G, marker.B));
                    brush.Freeze();
                    body = new Ellipse { Width = 10, Height = 10, Fill = brush, Stroke = brush };
                    _scheduleBodies[marker.AgvId] = body;
                    _scheduleOverlay.Children.Add(body);

                    var label = new TextBlock
                    {
                        Background = Brushes.White,
                        Foreground = brush,
                        FontSize = 10,
                        Padding = new Thickness(1, 0, 1, 0),
                        IsHitTestVisible = false,
                    };
                    _scheduleLabels[marker.AgvId] = label;
                    _scheduleOverlay.Children.Add(label);
                }
                var visible = _scheduleScene.IsOffCanvas(marker) ? Visibility.Collapsed : Visibility.Visible;
                var box = ScheduleBodySize;
                if (Math.Abs(body.Width - box) > 0.5)
                {
                    body.Width = box;
                    body.Height = box;
                    body.StrokeThickness = Math.Min(6.0, Math.Max(0.5, 1.0 / UiScale));
                }
                body.Visibility = visible;
                Canvas.SetLeft(body, marker.CanvasX - box / 2);
                Canvas.SetTop(body, marker.CanvasY - box / 2);
                if (!_scheduleHeadings.TryGetValue(marker.AgvId, out var heading))
                {
                    heading = new Line { Stroke = Brushes.Black, IsHitTestVisible = false };
                    Panel.SetZIndex(heading, 99);
                    _scheduleHeadings[marker.AgvId] = heading;
                    _scheduleOverlay.Children.Add(heading);
                }
                var observedYaw = frame.Agvs.First(a => a.AgvId == marker.AgvId).Pose.Yaw;
                heading.Visibility = observedYaw.HasValue ? visible : Visibility.Collapsed;
                if (observedYaw is double yaw)
                {
                    heading.X1 = marker.CanvasX;
                    heading.Y1 = marker.CanvasY;
                    heading.X2 = marker.CanvasX + box * 1.2 * Math.Cos(yaw);
                    heading.Y2 = marker.CanvasY - box * 1.2 * Math.Sin(yaw);
                    heading.StrokeThickness = Math.Max(.8, 1.5 / UiScale);
                }
                if (_scheduleLabels.TryGetValue(marker.AgvId, out var lbl))
                {
                    lbl.Text = marker.AgvId + " " + marker.State;
                    var font = Math.Min(160.0, Math.Max(10.0, 10.0 / UiScale));
                    if (Math.Abs(lbl.FontSize - font) > 0.5) lbl.FontSize = font;
                    // The label is placed beside the body, and the old code slid it back to the
                    // left when it ran off the right edge -- which put its opaque white
                    // background on top of the very vehicle it names. A vehicle near the right
                    // edge came out half-painted (measured: 94 px of body instead of 183) and the
                    // pixel check could only report "fewer pixels than expected", not why.
                    // Measure the text and put it on whichever side has room; if neither has
                    // room, drop the background rather than the vehicle.
                    lbl.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
                    var labelWidth = lbl.DesiredSize.Width;
                    var canvasWidth = _scheduleOverlay.ActualWidth;
                    var gap = box / 2 + 2.0 / UiScale;
                    var left = marker.CanvasX + gap;
                    if (left + labelWidth > canvasWidth) left = marker.CanvasX - gap - labelWidth;
                    if (left < 0) left = 0;
                    var labelRight = left + labelWidth;
                    var overlapsBody = labelRight >= marker.CanvasX - box / 2
                                       && left <= marker.CanvasX + box / 2;
                    lbl.Background = overlapsBody ? null : Brushes.White;
                    Canvas.SetLeft(lbl, left);
                    Canvas.SetTop(lbl, Math.Max(0, marker.CanvasY - 14));
                    lbl.Visibility = visible;
                }
            }

            // Draw the route the scheduler planned. The path is produced by L1 -- this layer
            // never computes a route, it only renders the node sequence that came with the
            // frame. Drawing it is what makes "where the scheduler intends to go" visible next
            // to "where the vehicle actually is".
            DrawScheduleTrails(frame);
            DrawScheduleRoutes(frame);
            DrawSchedulePhysical(frame);

            // Draw where the scheduler has *booked* resources. Congestion is a property of the
            // reservation table, not of vehicle positions: a junction two edges away is already
            // claimed, and the vehicles that will queue there have not arrived yet. Without this
            // layer the picture shows traffic only after it has happened.
            DrawScheduleCongestion(frame);
            ShowScheduleObservations(frame);

            // A vehicle that vanished from the frame must vanish from the picture; leaving it
            // behind would draw a robot that the scheduler no longer has.
            foreach (var gone in _scheduleBodies.Keys.Where(k => !seen.Contains(k)).ToList())
            {
                _scheduleOverlay.Children.Remove(_scheduleBodies[gone]);
                _scheduleBodies.Remove(gone);
                if (_scheduleHeadings.Remove(gone, out var heading)) _scheduleOverlay.Children.Remove(heading);
                if (_scheduleLabels.TryGetValue(gone, out var lbl))
                {
                    _scheduleOverlay.Children.Remove(lbl);
                    _scheduleLabels.Remove(gone);
                }
            }

            var sb = new StringBuilder();
            // The canvas size goes into the report because a position means nothing without
            // it: "px=(1084,396)" is on screen or off it depending on the canvas, and the two
            // cases look identical in the numbers alone.
            sb.Append(string.Format(CultureInfo.InvariantCulture,
                "step={0} t={1:F1} frames={2} scale={3:F2}px/m canvas={4:F0}x{5:F0} fit={6:F0}x{7:F0} scroller={8:F0}x{9:F0} vis={10:F0},{11:F0},{12:F0}x{13:F0}",
                frame.Step, frame.T, _scheduleFrames, _scheduleScene.Transform.PixelsPerMeter,
                _scheduleOverlay.ActualWidth, _scheduleOverlay.ActualHeight,
                _scheduleFitWidth, _scheduleFitHeight,
                scroller?.ActualWidth ?? -1, scroller?.ActualHeight ?? -1,
                _scheduleVisibleRect.X, _scheduleVisibleRect.Y,
                _scheduleVisibleRect.Width, _scheduleVisibleRect.Height));
            // Report the route layer explicitly. A drawn vehicle and a drawn route are different
            // claims, and the summary is the only machine-checkable record of the second one.
            sb.Append(string.Format(CultureInfo.InvariantCulture, " maxFull={0}步", _scheduleMaxCongestionSteps));
            sb.Append(string.Format(CultureInfo.InvariantCulture, " trail={0}/{1} src={2}",
                _scheduleTrailLines.Values.Count(l => l.Visibility == Visibility.Visible), _scheduleTrailPoints,
                DescribeTrailSources(frame)));
            // First congested frame, verbatim: congestion is intermittent, so "I saw a ring"
            // cannot be evidenced by a screenshot taken at an arbitrary moment. The label the
            // console formats is what the reader would see, printed once.
            // Congestion is intermittent, so "I saw a ring" cannot be evidenced by a screenshot
            // taken at an arbitrary moment. Log the first frame where a node has *stayed* full
            // for at least 5 steps -- that is the case the duration field exists to distinguish
            // from a one-step blip -- and keep the longest one seen for the closing summary.
            if (frame.Congestion is { Count: > 0 })
            {
                foreach (var pair in frame.Congestion)
                {
                    if (pair.Value.Steps > _scheduleMaxCongestionSteps)
                        _scheduleMaxCongestionSteps = pair.Value.Steps;
                }
                if (!_scheduleLoggedCongestion && _scheduleMaxCongestionSteps >= 5)
                {
                    _scheduleLoggedCongestion = true;
                    foreach (var pair in frame.Congestion.OrderByDescending(p => p.Value.Steps).Take(3))
                    {
                        LogLine(string.Format(CultureInfo.InvariantCulture,
                            "[schedule] 拥堵 {0} claims={1}/{2} blocked={3} 持续={4}步",
                            pair.Key, pair.Value.Claims, pair.Value.Capacity,
                            pair.Value.Blocked, pair.Value.Steps));
                    }
                }
            }

            sb.Append(frame.Physical is null
                ? " phys=0"
                : string.Format(CultureInfo.InvariantCulture, " phys={0} physPts={1}",
                    frame.Physical.Pose != null ? 1 : 0, _schedulePhysicalPoints));
            var routesDrawn = _scheduleRouteLines.Values.Count(l => l.Visibility == Visibility.Visible);
            var routePoints = _scheduleRouteLines.Values.Where(l => l.Visibility == Visibility.Visible)
                .Select(l => l.Points.Count).DefaultIfEmpty(0).Sum();
            sb.Append(" routes=").Append(routesDrawn).Append("/").Append(frame.Agvs.Count)
              .Append(" routePts=").Append(routePoints);
            // Same reason as routes: "the congestion layer is drawn" is a different claim from
            // "the scheduler reported a full node", and only this makes the first checkable.
            // Peak as well as current: congestion in this scenario comes in bursts between
            // long quiet stretches, so a current-only count reads 0 at every hundredth frame
            // even though the layer drew rings a moment earlier.
            sb.Append(string.Format(CultureInfo.InvariantCulture, " peakFull={0} peakQueue={1}",
                _schedulePeakFull, _schedulePeakQueued));
            sb.Append(string.Format(CultureInfo.InvariantCulture, " full={0} queued={1}",
                _scheduleCongestionRings.Values.Count(r => r.Visibility == Visibility.Visible),
                _scheduleCongestionRings.Keys.Count(k => _scheduleCongestionLabels.TryGetValue(k, out var t)
                                                     && t.Visibility == Visibility.Visible
                                                     && t.Foreground == Brushes.Red)));
            sb.Append(string.Format(CultureInfo.InvariantCulture,
                " panel={0:F0},{1:F0},{2:F0}x{3:F0}",
                _schedulePanelRect.X, _schedulePanelRect.Y,
                _schedulePanelRect.Width, _schedulePanelRect.Height));
            if (!placed.Complete) sb.Append(" | 无法放置: " + placed.Detail);
            foreach (var m in markers)
            {
                sb.Append(string.Format(CultureInfo.InvariantCulture,
                    "\n{0} | {1} | {2} | px=({3:F0},{4:F0})",
                    m.AgvId, m.State, m.TaskId ?? "-", m.CanvasX, m.CanvasY));
            }
            _scheduleSummaryText.Text = sb.ToString();
            // The panel text is the human view; the log is the checkable one. Emit the summary
            // every hundred frames (and on the first) so "the congestion layer drew something"
            // can be verified without reading pixels.
            if (_scheduleFrames == 1 || _scheduleFrames % 100 == 0)
            {
                var head = sb.ToString().Split('\n')[0];
                LogLine("[schedule] " + head);
            }
        }
    }
}
