using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Threading;
using Microsoft.Win32;
using RAWSimO.Live.Protocol;
using RAWSimO.Live.Tracking;

namespace RAWSimO.Visualization
{
    public partial class MainWindow
    {
        private FleetControlState _scheduleControl;
        private FleetControlRequestMessage _scheduleLastRequest;
        private bool _scheduleRequestPending;
        private bool _scheduleRequestUncertain;
        private readonly Stopwatch _scheduleControlAge = new();
        private readonly Stopwatch _scheduleFrameAge = new();
        private readonly Stopwatch _scheduleRequestAge = new();
        private ComboBox _schedulePickup, _scheduleDropoff;
        private TextBox _scheduleTaskIdBox;
        private CheckBox _scheduleUrgent;
        private Button _scheduleSubmit, _schedulePauseFleet, _scheduleResumeFleet, _scheduleRetry;
        private TextBlock _scheduleOperationResult, _scheduleTaskList;
        private TextBlock _scheduleAlerts, _scheduleObservations;
        private DispatcherTimer _scheduleOperationTimer;
        private FleetReplay _scheduleReplay;
        private readonly Stopwatch _scheduleReplayClock = new();
        private double _scheduleReplayOffset;
        private Slider _scheduleReplayPosition;
        private Button _scheduleReplayPlay;
        private bool _scheduleReplaySeeking;
        private double _scheduleReplayObservationAge;
        private CancellationTokenSource _scheduleReplayLoadCancellation;

        private void InitializeScheduleOperations(StackPanel stack)
        {
            var tasks = new WrapPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 4, 0, 0) };
            tasks.Children.Add(new TextBlock { Text = "任务号 ", VerticalAlignment = VerticalAlignment.Center });
            _scheduleTaskIdBox = new TextBox { Name = "ScheduleTaskId", Width = 85, ToolTip = "留空则自动生成" };
            tasks.Children.Add(_scheduleTaskIdBox);
            tasks.Children.Add(new TextBlock { Text = "取货 ", VerticalAlignment = VerticalAlignment.Center });
            _schedulePickup = new ComboBox { Name = "ScheduleTaskPickup", Width = 75, IsEditable = false };
            tasks.Children.Add(_schedulePickup);
            tasks.Children.Add(new TextBlock { Text = " 送到 ", VerticalAlignment = VerticalAlignment.Center });
            _scheduleDropoff = new ComboBox { Name = "ScheduleTaskDropoff", Width = 75, IsEditable = false };
            tasks.Children.Add(_scheduleDropoff);
            _scheduleUrgent = new CheckBox { Content = "加急优先级", Margin = new Thickness(4, 0, 4, 0), VerticalAlignment = VerticalAlignment.Center };
            tasks.Children.Add(_scheduleUrgent);
            _scheduleSubmit = ScheduleButton("下单", "ScheduleSubmitTask");
            _scheduleSubmit.Click += async (_, _) =>
            {
                if (_schedulePickup.SelectedItem is not string pickup || _scheduleDropoff.SelectedItem is not string dropoff) return;
                await SendScheduleOperation("submit_task", new Dictionary<string, object>
                {
                    ["task_id"] = string.IsNullOrWhiteSpace(_scheduleTaskIdBox.Text) ? "UI_" + Guid.NewGuid().ToString("N") : _scheduleTaskIdBox.Text.Trim(),
                    ["pickup"] = pickup, ["dropoff"] = dropoff, ["priority"] = _scheduleUrgent.IsChecked == true ? 1 : 0,
                });
            };
            tasks.Children.Add(_scheduleSubmit);
            stack.Children.Add(tasks);
            var actions = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 4, 0, 0) };
            _schedulePauseFleet = ScheduleButton("全队暂停", "SchedulePauseFleet");
            _scheduleResumeFleet = ScheduleButton("全队恢复", "ScheduleResumeFleet");
            _scheduleRetry = ScheduleButton("重试上一请求", "ScheduleRetryRequest");
            _schedulePauseFleet.Click += async (_, _) => await SendScheduleVersions("pause");
            _scheduleResumeFleet.Click += async (_, _) => await SendScheduleVersions("resume");
            _scheduleRetry.Click += async (_, _) => await SendScheduleRequest(_scheduleLastRequest);
            actions.Children.Add(_schedulePauseFleet);
            actions.Children.Add(_scheduleResumeFleet);
            actions.Children.Add(_scheduleRetry);
            stack.Children.Add(actions);
            _scheduleOperationResult = new TextBlock { Name = "ScheduleOperationResult", Text = "当前连接只读", TextWrapping = TextWrapping.Wrap, MaxWidth = 490 };
            stack.Children.Add(_scheduleOperationResult);
            _scheduleTaskList = new TextBlock { Name = "ScheduleTaskList", TextWrapping = TextWrapping.Wrap, MaxWidth = 490, FontSize = 10 };
            var taskScroll = new ScrollViewer { MaxHeight = 90, Content = _scheduleTaskList, VerticalScrollBarVisibility = ScrollBarVisibility.Auto };
            stack.Children.Add(taskScroll);
            var replay = new WrapPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 4, 0, 0) };
            var open = ScheduleButton("打开历史记录", "ScheduleOpenReplay");
            open.Click += async (_, _) =>
            {
                var dialog = new OpenFileDialog { Filter = "调度记录 (*.jsonl)|*.jsonl", CheckFileExists = true };
                if (dialog.ShowDialog(this) == true) await LoadScheduleReplay(dialog.FileName);
            };
            _scheduleReplayPlay = ScheduleButton("播放", "ScheduleReplayPlay");
            _scheduleReplayPlay.Click += (_, _) =>
            {
                if (_scheduleReplay == null) return;
                if (_scheduleReplayClock.IsRunning)
                {
                    _scheduleReplayOffset = Math.Min(_scheduleReplay.Duration, _scheduleReplayOffset + _scheduleReplayClock.Elapsed.TotalSeconds);
                    _scheduleReplayClock.Reset();
                }
                else
                {
                    if (_scheduleReplayOffset >= _scheduleReplay.Duration) _scheduleReplayOffset = 0;
                    _scheduleReplayClock.Restart();
                }
                RefreshScheduleOperations();
            };
            replay.Children.Add(open);
            replay.Children.Add(_scheduleReplayPlay);
            _scheduleReplayPosition = new Slider { Name = "ScheduleReplayPosition", Width = 150, Minimum = 0, Maximum = 1, IsEnabled = false };
            _scheduleReplayPosition.ValueChanged += (_, _) =>
            {
                if (_scheduleReplay == null || _scheduleReplaySeeking) return;
                _scheduleReplayOffset = _scheduleReplayPosition.Value;
                var running = _scheduleReplayClock.IsRunning;
                _scheduleReplayClock.Reset();
                if (running) _scheduleReplayClock.Start();
                ShowScheduleReplay(_scheduleReplayOffset);
            };
            replay.Children.Add(_scheduleReplayPosition);
            stack.Children.Add(replay);
            _scheduleAlerts = new TextBlock { Name = "ScheduleAlerts", TextWrapping = TextWrapping.Wrap };
            stack.Children.Add(_scheduleAlerts);
            _scheduleObservations = new TextBlock { Name = "ScheduleObservations", TextWrapping = TextWrapping.Wrap, FontSize = 10 };
            stack.Children.Add(new ScrollViewer { MaxHeight = 150, Content = _scheduleObservations,
                VerticalScrollBarVisibility = ScrollBarVisibility.Auto, HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled });
            _scheduleOperationTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(100) };
            _scheduleOperationTimer.Tick += (_, _) =>
            {
                if (_scheduleReplay != null && _scheduleReplayClock.IsRunning)
                {
                    var position = Math.Min(_scheduleReplay.Duration, _scheduleReplayOffset + _scheduleReplayClock.Elapsed.TotalSeconds);
                    ShowScheduleReplay(position);
                    if (_scheduleReplay != null && position >= _scheduleReplay.Duration)
                    {
                        _scheduleReplayClock.Reset();
                        _scheduleReplayOffset = position;
                    }
                }
                RefreshScheduleOperations();
            };
            _scheduleOperationTimer.Start();
            Closed += (_, _) => { _scheduleOperationTimer.Stop(); StopScheduleReplay(); };
            RefreshScheduleOperations();
        }

        private static Button ScheduleButton(string text, string name) => new Button
        {
            Name = name, Content = text, Margin = new Thickness(2, 0, 2, 0), Padding = new Thickness(5, 1, 5, 1),
        };

        private void ShowScheduleObservations(FleetFrameMessage frame)
        {
            if (_scheduleAlerts == null) return;
            // A command endpoint can disappear while observations continue. Keep
            // both explicit sources; per-vehicle execution state is the newer observation.
            var faults = frame.Control?.Faults == null ? frame.ExecutionFaults
                : frame.ExecutionFaults == null ? frame.Control.Faults
                : frame.Control.Faults.Concat(frame.ExecutionFaults)
                    .GroupBy(f => f.Key, StringComparer.Ordinal).ToDictionary(g => g.Key, g => g.Last().Value, StringComparer.Ordinal);
            var activeFaults = faults?.Where(f => !string.IsNullOrWhiteSpace(f.Value)).ToArray();
            _scheduleAlerts.Text = faults == null || faults.Count == 0 && frame.Agvs.Count != 0 ? "执行告警：当前来源未提供" : activeFaults.Length == 0
                ? "执行告警：未报告故障" : "执行锁停 · " + string.Join("；", activeFaults.Select(f => f.Key + ": " + f.Value));
            _scheduleAlerts.Foreground = activeFaults?.Length > 0 ? Brushes.Firebrick : Brushes.Gray;
            var lines = frame.Agvs.Select(a => a.Pose.Available == false
                ? $"{a.AgvId} · 定位暂不可用 · {a.State}"
                : a.Pose.X is double x && a.Pose.Y is double y
                    ? $"{a.AgvId} · ({x:F2}, {y:F2}) m · " + (a.Pose.Yaw is double yaw ? $"{yaw * 180 / Math.PI:F1}°" : "朝向未提供") + $" · {a.Pose.Source} · {a.State}"
                    : $"{a.AgvId} · 算法仿真 · {a.State}").ToList();
            if (frame.PhysicalHolds is { } pins)
            {
                lines.Add("物理占用（Python 确认后释放）");
                lines.AddRange(pins.Nodes.Select(p => $"节点 {p.Key}: {string.Join(", ", p.Value)}"));
                lines.AddRange(pins.Edges.Select(p => $"道路 {p.Key}: {string.Join(", ", p.Value)}"));
            }
            else if (frame.Holds != null)
            {
                lines.Add("节点预约（Python）");
                lines.AddRange(frame.Holds.Select(p => $"{p.Key}: {string.Join(", ", p.Value)}"));
            }
            else lines.Add("占用信息：当前来源未提供");
            _scheduleObservations.Text = string.Join("\n", lines);
        }

        private void SetScheduleControl(FleetControlState control)
        {
            if (_scheduleReplay == null && _scheduleOperationResult != null)
            {
                if (control == null) _scheduleOperationResult.Text = "当前连接只读或已断开";
                else if (_scheduleControl == null) _scheduleOperationResult.Text = "Python 调度器已连接";
            }
            if (control != null && _scheduleLastRequest != null && control.SessionId != _scheduleLastRequest.SessionId)
            {
                _scheduleRequestPending = _scheduleRequestUncertain = false;
                _scheduleLastRequest = null;
            }
            _scheduleControl = _scheduleReplay == null ? control : null;
            if (control != null) _scheduleControlAge.Restart();
            if (_scheduleTaskList != null && control?.Tasks != null)
                _scheduleTaskList.Text = $"任务 {control.TaskTotal} 项（最近 {control.Tasks.Count} 项）\n" +
                    string.Join("\n", control.Tasks.Select(t => $"{t.TaskId}: {t.Pickup}→{t.Dropoff} · {t.Status} · {t.AssignedAgv}"));
            else if (_scheduleTaskList != null) _scheduleTaskList.Text = "";
            if (_schedulePickup != null && _scheduleHello != null)
            {
                var ids = _scheduleHello.Nodes.Select(n => n.Id).ToArray();
                if (_schedulePickup.Items.Count == 0 || !_schedulePickup.Items.Cast<string>().SequenceEqual(ids))
                {
                    _schedulePickup.ItemsSource = ids;
                    _scheduleDropoff.ItemsSource = ids;
                    if (ids.Length > 0) _schedulePickup.SelectedIndex = 0;
                    if (ids.Length > 1) _scheduleDropoff.SelectedIndex = 1;
                }
            }
            RefreshScheduleOperations();
        }

        private void RefreshScheduleOperations()
        {
            if (_scheduleSubmit == null) return;
            var replayGap = _scheduleReplay != null && _scheduleReplayObservationAge >= .5;
            if (replayGap || (_scheduleReplay == null && _scheduleFrameAge.IsRunning && _scheduleFrameAge.Elapsed.TotalSeconds >= .5))
            {
                foreach (var body in _scheduleBodies.Values) body.Visibility = Visibility.Collapsed;
                foreach (var label in _scheduleLabels.Values) label.Visibility = Visibility.Collapsed;
                foreach (var heading in _scheduleHeadings.Values) heading.Visibility = Visibility.Collapsed;
                if (_schedulePhysicalMarker != null) _schedulePhysicalMarker.Visibility = Visibility.Collapsed;
                if (_schedulePhysicalLabel != null) _schedulePhysicalLabel.Visibility = Visibility.Collapsed;
                if (_scheduleAlerts != null && !_scheduleAlerts.Text.StartsWith("历史状态"))
                    _scheduleAlerts.Text = (replayGap ? "历史状态 · 记录中断\n" : "历史状态 · 调度反馈已过期\n") + _scheduleAlerts.Text;
                if (_scheduleObservations != null && !_scheduleObservations.Text.StartsWith("历史观测"))
                    _scheduleObservations.Text = "历史观测 · 当前车辆位置未知\n" + _scheduleObservations.Text;
                _scheduleStatusText.Text = replayGap
                    ? $"历史回放 · 记录中断 {_scheduleReplayObservationAge:F1} 秒 · 仅保留历史轨迹"
                    : "状态: 调度反馈已过期 · 仅保留历史轨迹";
                _scheduleStatusText.Foreground = Brushes.Firebrick;
            }
            var live = _scheduleReplay == null && _scheduleClient != null && _scheduleControl != null && _scheduleControlAge.Elapsed.TotalSeconds < .5;
            if (_scheduleRequestPending && _scheduleRequestAge.Elapsed.TotalSeconds > 4)
            {
                _scheduleRequestPending = false;
                _scheduleRequestUncertain = true;
                _scheduleOperationResult.Text = "请求结果未知；重试同一请求可核对，勿重复下单。";
            }
            var enabled = live && !_scheduleRequestPending && !_scheduleRequestUncertain;
            _scheduleSubmit.IsEnabled = enabled && _scheduleControl.Operations.Contains("submit_task");
            _schedulePauseFleet.IsEnabled = enabled && !_scheduleControl.Paused && _scheduleControl.Operations.Contains("pause");
            _scheduleResumeFleet.IsEnabled = enabled && _scheduleControl.Paused && _scheduleControl.Operations.Contains("resume");
            _scheduleRetry.IsEnabled = live && !_scheduleRequestPending && _scheduleLastRequest != null && _scheduleLastRequest.SessionId == _scheduleControl.SessionId;
            _scheduleReplayPlay.IsEnabled = _scheduleReplay != null;
            _scheduleReplayPlay.Content = _scheduleReplayClock.IsRunning ? "暂停回放" : "播放";
        }

        private Task SendScheduleVersions(string operation) => _scheduleControl == null ? Task.CompletedTask :
            SendScheduleOperation(operation, new { routes = _scheduleControl.Routes });

        private Task SendScheduleOperation(string operation, object payload)
        {
            var control = _scheduleControl;
            if (control == null || _scheduleReplay != null) return Task.CompletedTask;
            return SendScheduleRequest(new FleetControlRequestMessage(control.SessionId, control.SceneId,
                control.RunId, Guid.NewGuid().ToString("N"), operation, JsonSerializer.SerializeToElement(payload)));
        }

        private async Task SendScheduleRequest(FleetControlRequestMessage request)
        {
            if (request == null || _scheduleReplay != null || _scheduleClient == null || _scheduleControl == null
                || _scheduleControlAge.Elapsed.TotalSeconds >= .5 || _scheduleRequestPending) return;
            _scheduleLastRequest = request;
            _scheduleRequestPending = true;
            _scheduleRequestAge.Restart();
            _scheduleOperationResult.Text = "已提交，等待 Python 调度器确认…";
            RefreshScheduleOperations();
            try
            {
                using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(2));
                await _scheduleClient.SendAsync(request, timeout.Token);
            }
            catch (Exception ex)
            {
                _scheduleRequestPending = false;
                _scheduleRequestUncertain = true;
                _scheduleOperationResult.Text = "发送结果未知，可重试同一请求：" + ex.Message;
            }
        }

        private void OnScheduleControlResult(FleetControlResultMessage result)
        {
            if (_scheduleReplay != null || result.RequestId != _scheduleLastRequest?.RequestId) return;
            _scheduleRequestPending = false;
            _scheduleRequestUncertain = false;
            _scheduleOperationResult.Foreground = result.Accepted ? Brushes.DarkGreen : Brushes.Firebrick;
            _scheduleOperationResult.Text = result.Accepted
                ? "Python 已确认：" + result.Operation + (result.Operation == "pause" ? "；等待车辆完成停车。" : "")
                : "Python 拒绝：" + result.Reason;
            // The receipt can be an old deduplicated reply. Only current frames
            // refresh action versions and freshness, never a historical receipt.
            RefreshScheduleOperations();
        }

        private async Task LoadScheduleReplay(string path)
        {
            _scheduleReplayLoadCancellation?.Cancel();
            using var cancellation = new CancellationTokenSource();
            _scheduleReplayLoadCancellation = cancellation;
            FleetReplay recording = null;
            try
            {
                recording = await Task.Run(() => FleetReplay.Open(path, cancellation.Token));
                cancellation.Token.ThrowIfCancellationRequested();
                await ScheduleDisconnectAsync();
                cancellation.Token.ThrowIfCancellationRequested();
                StopScheduleReplay();
                _scheduleReplay = recording;
                recording = null; // The window now owns the file handle.
                _scheduleReplayPosition.Maximum = Math.Max(.001, _scheduleReplay.Duration);
                _scheduleReplayPosition.IsEnabled = true;
                OnScheduleHello(_scheduleReplay.Hello);
                ShowScheduleReplay(0);
                _scheduleOperationResult.Text = "历史回放 · 实时控制已关闭";
            }
            catch (OperationCanceledException) { }
            catch (Exception ex) { _scheduleOperationResult.Text = "打开回放失败：" + ex.Message; }
            finally
            {
                recording?.Dispose();
                if (_scheduleReplayLoadCancellation == cancellation) _scheduleReplayLoadCancellation = null;
            }
        }

        private void ShowScheduleReplay(double seconds)
        {
            if (_scheduleReplay == null) return;
            FleetReplayFrame observation;
            try { observation = _scheduleReplay.At(seconds); }
            catch (Exception ex) when (ex is IOException or FormatException or JsonException or ObjectDisposedException)
            {
                StopScheduleReplay();
                ClearScheduleVisuals();
                _scheduleOverlay.Visibility = Visibility.Collapsed;
                _scheduleOperationResult.Text = "回放读取失败：" + ex.Message;
                RefreshScheduleOperations();
                return;
            }
            _scheduleReplayObservationAge = Math.Max(0, seconds - observation.Seconds);
            OnScheduleFrame(observation.Frame);
            RefreshScheduleOperations();
            _scheduleReplaySeeking = true;
            _scheduleReplayPosition.Value = seconds;
            _scheduleReplaySeeking = false;
            _scheduleStatusText.Text = $"历史回放 · {seconds:F1}/{_scheduleReplay.Duration:F1} 秒 · " +
                (_scheduleReplayObservationAge >= .5 ? $"记录中断 {_scheduleReplayObservationAge:F1} 秒，仅保留历史轨迹" : "采用原始观测");
            _scheduleStatusText.Foreground = _scheduleReplayObservationAge >= .5 ? Brushes.Firebrick : Brushes.Gray;
        }

        private void StopScheduleReplay()
        {
            _scheduleReplayLoadCancellation?.Cancel();
            _scheduleReplayClock.Reset();
            _scheduleReplay?.Dispose();
            _scheduleReplay = null;
            _scheduleReplayOffset = 0;
            _scheduleReplayObservationAge = 0;
            if (_scheduleReplayPosition != null) _scheduleReplayPosition.IsEnabled = false;
            _scheduleRequestPending = false;
        }
    }
}
