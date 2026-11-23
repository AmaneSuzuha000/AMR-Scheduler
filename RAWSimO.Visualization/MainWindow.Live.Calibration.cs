#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using RAWSimO.Live.Calibration;
using RAWSimO.Live.Geometry;

namespace RAWSimO.Visualization
{
    public partial class MainWindow
    {
        private readonly List<CalibrationPoint> _liveCalibrationPoints = new();
        private readonly List<double> _liveValidationErrors = new();
        private TextBox _liveCalMapXBox;
        private TextBox _liveCalMapYBox;
        private TextBox _liveCalMapYawBox;
        private TextBox _liveCalSigmaBox;
        private TextBlock _liveCalProgress;
        private TextBlock _liveCalResult;
        private Button _liveCalCaptureButton;
        private Button _liveCalFitButton;
        private Button _liveCalValidateButton;
        private Button _liveCalOdomButton;
        private Button _liveCalResetButton;
        private string _liveHelloFrame = "map";

        private void BuildCalibrationControls(StackPanel panel)
        {
            var group = new GroupBox
            {
                Name = "GroupBoxCalibration",
                Header = "多点标定（6 点 × 20 帧）",
                Margin = new Thickness(0, 4, 0, 0),
            };
            var stack = new StackPanel { Orientation = Orientation.Vertical };

            var mapRow = new WrapPanel { Orientation = Orientation.Horizontal };
            mapRow.Children.Add(new TextBlock
            {
                Text = "地图 X/Y/Yaw:",
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 4, 0),
            });
            _liveCalMapXBox = new TextBox { Name = "TextBoxCalMapX", Text = "0.375", Width = 58, Height = 22 };
            _liveCalMapYBox = new TextBox { Name = "TextBoxCalMapY", Text = "0.375", Width = 58, Height = 22, Margin = new Thickness(3, 0, 0, 0) };
            _liveCalMapYawBox = new TextBox { Name = "TextBoxCalMapYaw", Text = "0.000", Width = 58, Height = 22, Margin = new Thickness(3, 0, 0, 0) };
            mapRow.Children.Add(_liveCalMapXBox);
            mapRow.Children.Add(_liveCalMapYBox);
            mapRow.Children.Add(_liveCalMapYawBox);
            stack.Children.Add(mapRow);

            var sigmaRow = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 4, 0, 0) };
            sigmaRow.Children.Add(new TextBlock
            {
                Text = "σ (m):",
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 4, 0),
            });
            _liveCalSigmaBox = new TextBox { Name = "TextBoxCalSigma", Text = "0.010", Width = 58, Height = 22 };
            sigmaRow.Children.Add(_liveCalSigmaBox);
            stack.Children.Add(sigmaRow);

            var buttons = new WrapPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 4, 0, 0) };
            _liveCalCaptureButton = new Button { Name = "ButtonCalCapture", Content = "采集当前点（20 帧）", Padding = new Thickness(5, 1, 5, 1) };
            _liveCalCaptureButton.Click += LiveCalibrationCapture_Click;
            buttons.Children.Add(_liveCalCaptureButton);
            _liveCalFitButton = new Button { Name = "ButtonCalFit", Content = "拟合", Padding = new Thickness(5, 1, 5, 1), Margin = new Thickness(4, 0, 0, 0) };
            _liveCalFitButton.Click += LiveCalibrationFit_Click;
            buttons.Children.Add(_liveCalFitButton);
            _liveCalValidateButton = new Button { Name = "ButtonCalValidate", Content = "检验点", Padding = new Thickness(5, 1, 5, 1), Margin = new Thickness(4, 0, 0, 0) };
            _liveCalValidateButton.Click += LiveCalibrationValidate_Click;
            buttons.Children.Add(_liveCalValidateButton);
            _liveCalOdomButton = new Button { Name = "ButtonCalOdom", Content = "起点工装", Padding = new Thickness(5, 1, 5, 1), Margin = new Thickness(4, 0, 0, 0) };
            _liveCalOdomButton.Click += LiveCalibrationOdom_Click;
            buttons.Children.Add(_liveCalOdomButton);
            _liveCalResetButton = new Button { Name = "ButtonCalReset", Content = "清空", Padding = new Thickness(5, 1, 5, 1), Margin = new Thickness(4, 0, 0, 0) };
            _liveCalResetButton.Click += LiveCalibrationReset_Click;
            buttons.Children.Add(_liveCalResetButton);
            stack.Children.Add(buttons);

            _liveCalProgress = new TextBlock { Name = "TextBlockCalProgress", Text = "标定点: 0/6; 检验点: 0", Margin = new Thickness(0, 4, 0, 0) };
            stack.Children.Add(_liveCalProgress);
            _liveCalResult = new TextBlock
            {
                Name = "TextBlockCalResult",
                Text = "等待采集",
                TextWrapping = TextWrapping.Wrap,
                FontFamily = new FontFamily("Consolas"),
                FontSize = 11,
                Margin = new Thickness(0, 2, 0, 0),
            };
            stack.Children.Add(_liveCalResult);

            group.Content = stack;
            panel.Children.Add(group);
        }

        private async void LiveCalibrationCapture_Click(object sender, RoutedEventArgs e)
        {
            if (_liveCalibrationPoints.Count >= 6)
            {
                _liveCalResult.Text = "已有 6 个标定点，先清空再重来";
                return;
            }
            if (!TryReadMapPoint(out var map)) return;
            var average = await CaptureStationaryPoseAsync(20);
            if (average is null)
            {
                _liveCalResult.Text = "采集失败：没有可用的位姿";
                return;
            }
            _liveCalibrationPoints.Add(new CalibrationPoint(map, average.Mean.Position, average.Mean.Yaw));
            _liveCalProgress.Text = $"标定点: {_liveCalibrationPoints.Count}/6; 检验点: {_liveValidationErrors.Count}";
            _liveCalResult.Text = $"已采集 #{_liveCalibrationPoints.Count}: σpos={average.PositionStdDev * 1000.0:F1} mm, σyaw={average.YawStdDev * 180.0 / Math.PI:F2}°";
            LogLine($"[Live] calibration point {_liveCalibrationPoints.Count}: map={map}, chassis={average.Mean.Position}");
        }

        private void LiveCalibrationFit_Click(object sender, RoutedEventArgs e)
        {
            if (_liveCalibrationPoints.Count < 3)
            {
                _liveCalResult.Text = "至少需要 3 个标定点";
                return;
            }
            var sigma = ReadDouble(_liveCalSigmaBox, 0.010);
            try
            {
                var result = CalibrationSolver.FitBest(_liveCalibrationPoints, CalibrationModel.SE2, sigma,
                    yawOffset: 0.0, vehicle: _liveVehicleId, frame: _liveHelloFrame);
                var sim = CalibrationSolver.Fit(_liveCalibrationPoints, CalibrationModel.Sim2, sigma: sigma,
                    yawOffset: 0.0, vehicle: _liveVehicleId, frame: _liveHelloFrame);
                var k = CalibrationSolver.ResidualFactor(_liveCalibrationPoints.Count);
                var maxLoo = result.Points.OrderByDescending(p => p.LooError).First();
                var pass = result.ResidualRms <= k * sigma && Math.Abs(sim.ScaleDiag - 1.0) <= 0.01;
                _liveCalibration = result;
                ApplyFrameGate();
                _liveCalResult.Text =
                    $"SE(2): θ={result.Theta:F4} rad, t=({result.Tx:F3},{result.Ty:F3}) m, RMS={result.ResidualRms * 1000.0:F2} mm\n" +
                    $"k·σ={k * sigma * 1000.0:F2} mm -> {(result.ResidualRms <= k * sigma ? "通过" : "超限")}; " +
                    $"Sim(2) s={sim.ScaleDiag:F5} (|s-1|={(Math.Abs(sim.ScaleDiag - 1.0) * 100.0):F2}%)\n" +
                    $"手性: {(result.Mirror ? "镜像（左手系，请确认）" : "正常")}; LOO 最大={maxLoo.LooError * 1000.0:F2} mm\n" +
                    $"总计: {(pass ? "通过" : "不通过")}";
                LogLine($"[Live] calibration fit: {result.ResidualRms:F5} m RMS, mirror={result.Mirror}, scale={sim.ScaleDiag:F5}");
            }
            catch (Exception ex)
            {
                _liveCalResult.Text = "拟合失败: " + ex.Message;
            }
        }

        private async void LiveCalibrationValidate_Click(object sender, RoutedEventArgs e)
        {
            if (_liveCalibration is null)
            {
                _liveCalResult.Text = "请先拟合标定";
                return;
            }
            if (!TryReadMapPoint(out var map)) return;
            var average = await CaptureStationaryPoseAsync(20);
            if (average is null)
            {
                _liveCalResult.Text = "检验点采集失败：没有可用位姿";
                return;
            }
            var predicted = _liveCalibration.ChassisToMap(average.Mean.Position);
            var error = Point2D.Distance(predicted, map);
            _liveValidationErrors.Add(error);
            var max = _liveValidationErrors.Max();
            var rms = Math.Sqrt(_liveValidationErrors.Sum(x => x * x) / _liveValidationErrors.Count);
            _liveCalProgress.Text = $"标定点: {_liveCalibrationPoints.Count}/6; 检验点: {_liveValidationErrors.Count}";
            _liveCalResult.Text = $"检验点 #{_liveValidationErrors.Count}: 误差={error * 1000.0:F1} mm\n" +
                                  $"最大={max * 1000.0:F1} mm, RMS={rms * 1000.0:F1} mm";
        }

        private void LiveCalibrationOdom_Click(object sender, RoutedEventArgs e)
        {
            if (_liveTracker?.Latest is not { } latest)
            {
                _liveCalResult.Text = "起点工装失败：没有当前位姿";
                return;
            }
            if (!TryReadMapPoint(out var map)) return;
            var mapYaw = ReadDouble(_liveCalMapYawBox, 0.0);
            try
            {
                _liveCalibration = CalibrationSolver.FromOrigin(map, mapYaw, latest.Position, latest.Yaw,
                    vehicle: _liveVehicleId, frame: string.IsNullOrWhiteSpace(_liveHelloFrame) ? "odom" : _liveHelloFrame);
                ApplyFrameGate();
                _liveCalResult.Text = $"起点工装: θ={_liveCalibration.Theta:F4} rad, t=({_liveCalibration.Tx:F3},{_liveCalibration.Ty:F3}) m";
            }
            catch (Exception ex)
            {
                _liveCalResult.Text = "起点工装失败: " + ex.Message;
            }
        }

        private void LiveCalibrationReset_Click(object sender, RoutedEventArgs e)
        {
            _liveCalibrationPoints.Clear();
            _liveValidationErrors.Clear();
            _liveCalibration = null;
            ApplyFrameGate();
            _liveCalProgress.Text = "标定点: 0/6; 检验点: 0";
            _liveCalResult.Text = "已清空";
        }

        private async Task<PoseAverage?> CaptureStationaryPoseAsync(int frameCount)
        {
            var samples = new List<Pose2D>();
            var seen = new HashSet<long>();
            var start = DateTime.UtcNow;
            while (samples.Count < frameCount && (DateTime.UtcNow - start).TotalSeconds < 3.0)
            {
                if (_liveTracker?.Latest is { } latest && seen.Add(latest.Seq))
                    samples.Add(latest.Pose);
                await Task.Delay(50);
            }
            if (samples.Count < Math.Max(3, frameCount / 2)) return null;
            return PoseAverager.Average(samples);
        }

        private bool TryReadMapPoint(out Point2D map)
        {
            map = default;
            if (!double.TryParse(_liveCalMapXBox.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out var x) ||
                !double.TryParse(_liveCalMapYBox.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out var y))
            {
                _liveCalResult.Text = "地图 X/Y 不是有效数字";
                return false;
            }
            map = new Point2D(x, y);
            return true;
        }

        private static double ReadDouble(TextBox box, double fallback) =>
            double.TryParse(box.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out var value) && double.IsFinite(value)
                ? value : fallback;
    }
}
