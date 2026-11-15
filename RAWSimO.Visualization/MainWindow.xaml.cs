using RAWSimO.Core;
using RAWSimO.Core.Generator;
using RAWSimO.Core.Info;
using RAWSimO.Core.Interfaces;
using RAWSimO.Core.IO;
using RAWSimO.Core.Items;
using RAWSimO.Core.Management;
using RAWSimO.Core.Randomization;
using RAWSimO.Core.Control;
using RAWSimO.Playground.Tests;
using RAWSimO.VisualToolbox;
using RAWSimO.Visualization.Rendering;
using HelixToolkit.Wpf;
using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using System.Windows.Navigation;
using System.Windows.Shapes;
using RAWSimO.Core.Configurations;
using System.Threading;
using System.Reflection;
using System.Globalization;
using RAWSimO.Core.Remote;
using RAWSimO.CommFramework;
using RAWSimO.Core.Statistics;
using RAWSimO.DataPreparation;
using RAWSimO.Toolbox;

namespace RAWSimO.Visualization
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            InitializeLiveLayer();

            // Add key event listener
            this.KeyDown += new KeyEventHandler(MainWindow_KeyDown);

            // >>> Init
            // Init lights
            foreach (var light in _lights)
                Viewport3D.Lights.Children.Add(light);
            // Init dynamic button images
            _imageDisableInfoView = ConvertBitmapResource(Properties.Resources.info27);
            _imageEnableInfoView = ConvertBitmapResource(Properties.Resources.info22);
            // Init setting configuration ui-elements
            VisualParameterInputHelper.FillParameterItems(Dispatcher, WrapPanelSettingConfiguration.Children, _baseConfiguration);
            // Init control configuration ui-elements
            VisualParameterInputHelper.FillParameterItems(Dispatcher, WrapPanelControlConfiguration.Children, _controlConfiguration);
            // Init layout configuration ui-elements
            VisualParameterInputHelper.FillParameterItems(Dispatcher, WrapPanelInstanceLayout.Children, _layoutConfig);
            // Init simple item config param list
            VisualParameterInputHelper.FillParameterItems(Dispatcher, WrapPanelSimpleItemConfiguration.Children, _simpleItemGeneratorPreConfig);
            // Init heatmap rendering section ui-elements
            InitHeatmapUIElements();
            // Init some combobox content elements
            InitComboboxContent();
            // Set statistics output directory
            TextStatisicsOutputFolder.Text = Directory.GetCurrentDirectory();
            // Set current active remote button and image
            _remoteButton = ButtonConnectControl;
            _remoteImage = ImageConnectControl;
        }

        /// <summary>
        /// Used to log output to a file.
        /// </summary>
        private StreamWriter _logWriter;

        /// <summary>
        /// Indicates whether a log will be written to disk.
        /// </summary>
        private bool _logWritingEnabled = true;

        /// <summary>
        /// The current view-mode in use (2D/3D).
        /// </summary>
        ViewMode _viewMode = ViewMode.View2D;

        /// <summary>
        /// Denotes whether to draw additional information like a heatmap or not.
        /// </summary>
        bool _heatMode = false;

        /// <summary>
        /// The current mode of bot coloring.
        /// </summary>
        BotColorMode _botColorMode = BotColorMode.DefaultBotDefaultState;

        /// <summary>
        /// The control for the 2D animation.
        /// </summary>
        SimulationAnimation2D _animationControl2D;

        /// <summary>
        /// The control for the 3D animation.
        /// </summary>
        SimulationAnimation3D _animationControl3D;

        /// <summary>
        /// The control for the meta information of the objects.
        /// </summary>
        SimulationInfoManager _infoControl;

        /// <summary>
        /// The simulation wrapper.
        /// </summary>
        SimulationVisualizer _renderer;

        /// <summary>
        /// Indicates whether there is an active paused simulation.
        /// </summary>
        bool _paused = false;

        /// <summary>
        /// Indicates whether there is an active simulation.
        /// </summary>
        bool _running = false;

        /// <summary>
        /// The currently active instance.
        /// </summary>
        Instance _instance = null;

        /// <summary>
        /// Indicates whether the instance was invalidated by an operation such that it cannot execute anymore.
        /// </summary>
        private bool _instanceInvalidated = false;

        /// <summary>
        /// The lights to use in the 3D animation.
        /// </summary>
        List<Light> _lights = new List<Light> { new DirectionalLight(Colors.White, new Vector3D(-4, -5, -6)), new DirectionalLight(Colors.White, new Vector3D(5, 6, 4)), };

        /// <summary>
        /// The image to use for the enable image view button.
        /// </summary>
        BitmapImage _imageEnableInfoView;

        /// <summary>
        /// The image to use for the disable image view button.
        /// </summary>
        BitmapImage _imageDisableInfoView;

        /// <summary>
        /// A directory to save snapshots to.
        /// </summary>
        string _snapshotDir = "snapshots";

        /// <summary>
        /// The list of all tiers currently available for visualization.
        /// </summary>
        List<ITierInfo> _tiers;

        /// <summary>
        /// The index of the tier currently visualized.
        /// </summary>
        int _focusedTierIndex;

        /// <summary>
        /// The configuration to use when simple items shall be generated online. This object is used to automatically generate appropriate ui-elements and parse their content.
        /// </summary>
        private OrderGenerator.SimpleItemGeneratorPreConfiguration _simpleItemGeneratorPreConfig = new OrderGenerator.SimpleItemGeneratorPreConfiguration();
        /// <summary>
        /// The layout configuration to use for generating instances.
        /// </summary>
        private LayoutConfiguration _layoutConfig = new LayoutConfiguration();
        /// <summary>
        /// The configuration to attach to the instance when executing.
        /// </summary>
        private SettingConfiguration _baseConfiguration = new SettingConfiguration();
        /// <summary>
        /// The control configuration to attach to the instance when executing.
        /// </summary>
        private ControlConfiguration _controlConfiguration = new ControlConfiguration();

        #region Instance and configuration parsing

        private void ParseSetting()
        {
            // Parse the configuration from the gui elements
            VisualParameterInputHelper.ParseParameterItems(Dispatcher, WrapPanelSettingConfiguration.Children, _baseConfiguration);
            // Set appropriate default name
            _baseConfiguration.Name = _baseConfiguration.GetMetaInfoBasedConfigName();
        }
        private void ParseConfiguration()
        {
            // Parse the configuration from the gui elements
            VisualParameterInputHelper.ParseParameterItems(Dispatcher, WrapPanelControlConfiguration.Children, _controlConfiguration);
            // Set appropriate default name
            _controlConfiguration.Name = _controlConfiguration.GetMetaInfoBasedConfigName();
        }
        private void ParseLayoutConfiguration()
        {
            // Parse the configuration from the gui elements
            VisualParameterInputHelper.ParseParameterItems(Dispatcher, WrapPanelInstanceLayout.Children, _layoutConfig);
        }

        #endregion

        #region Helper methods

        private bool GetDrawMode3D() { return _viewMode == ViewMode.View3D; }
        private void To2DView()
        {
            // Handle visibility
            scroller.Visibility = Visibility.Visible;
            Viewport3D.Visibility = Visibility.Hidden;
            if (_animationControl3D != null)
                _animationControl3D.StopAnimation();
            _viewMode = ViewMode.View2D;
            // Refresh view
            if (_animationControl2D != null)
                _animationControl2D.Update(true);
        }
        private void To3DView()
        {
            // Handle visibility
            scroller.Visibility = Visibility.Hidden;
            Viewport3D.Visibility = Visibility.Visible;
            if (_animationControl2D != null)
                _animationControl2D.StopAnimation();
            _viewMode = ViewMode.View3D;
            // Refresh view
            if (_animationControl3D != null)
                _animationControl3D.Update(true);
        }
        void Log(string message)
        {
            // Init logger
            if (_logWritingEnabled)
                lock (this)
                    if (_logWriter == null)
                        try { _logWriter = new StreamWriter("controllog.txt", false) { AutoFlush = true }; }
                        catch (Exception) { _logWritingEnabled = false; }
            // Write to file
            _logWriter?.Write(message);
            // Write to GUI
            this.Dispatcher.InvokeAsync(() =>
            {
                TextBoxOutput.AppendText(message);
                TextBoxOutput.ScrollToEnd();
            });
        }
        void LogLine(string message)
        {
            // Init logger
            if (_logWritingEnabled)
                lock (this)
                    if (_logWriter == null)
                        try { _logWriter = new StreamWriter("controllog.txt", false) { AutoFlush = true }; }
                        catch (Exception) { _logWritingEnabled = false; }
            // Write to file
            _logWriter?.WriteLine(message);
            // Write to GUI
            this.Dispatcher.InvokeAsync(() =>
            {
                TextBoxOutput.AppendText(message + "\n");
                TextBoxOutput.ScrollToEnd();
            });
        }
        private void IncreaseSpeed()
        {
            double newSpeed = (double.Parse(TextBlockUpdateSpeed.Text, IOConstants.FORMATTER) + 1);
            if (newSpeed > 0 && newSpeed <= 400)
            {
                TextBlockUpdateSpeed.Text = newSpeed.ToString(IOConstants.FORMATTER);
                if (_renderer != null)
                    _renderer.SetUpdateRate(newSpeed);
            }
        }
        private void DecreaseSpeed()
        {
            double newSpeed = (double.Parse(TextBlockUpdateSpeed.Text, IOConstants.FORMATTER) - 1);
            if (newSpeed > 0 && newSpeed <= 400)
            {
                TextBlockUpdateSpeed.Text = newSpeed.ToString(IOConstants.FORMATTER);
                if (_renderer != null)
                    _renderer.SetUpdateRate(newSpeed);
            }
        }
        private void SetSpeed(int speed)
        {
            // Bound speed
            speed = Math.Max(Math.Min(speed, 400), 1);
            TextBlockUpdateSpeed.Text = speed.ToString(IOConstants.FORMATTER);
            if (_renderer != null)
                _renderer.SetUpdateRate(speed);
        }
        private void UpdateHeatmode()
        {
            if (_instance != null && ComboBoxHeatMode.SelectedIndex >= 0)
            {
                // Check whether heatmode drawing is desired at all
                HeatMode mode = Enum.GetValues(typeof(HeatMode)).Cast<HeatMode>().ElementAt(ComboBoxHeatMode.SelectedIndex);
                if (mode == HeatMode.None)
                {
                    // Disable heat rendering
                    _heatMode = false;
                }
                else
                {
                    // Set the corresponding heatmode
                    _heatMode = true;
                    _instance.SettingConfig.HeatMode = mode;
                }
            }
        }
        private void UpdateBotColoringMode()
        {
            // Set new coloring mode
            if (ComboBoxBotColoringMode.SelectedIndex >= 0)
                _botColorMode = Enum.GetValues(typeof(BotColorMode)).Cast<BotColorMode>().ElementAt(ComboBoxBotColoringMode.SelectedIndex);
        }
        private void UpdateFocusedTier()
        {
            if (_instance != null && _animationControl2D != null)
            {
                // Keep index in bounds
                if (ComboBoxTierSelection2D.SelectedIndex < 0 || ComboBoxTierSelection2D.SelectedIndex >= _tiers.Count)
                    _focusedTierIndex = 0;
                // Update the visual
                _animationControl2D.UpdateCurrentTier(_tiers[_focusedTierIndex]);
                // Update the info panel
                TextBlockTierInFocus.Text = _tiers[_focusedTierIndex].GetInfoID().ToString();
            }
        }
        private void FocusTierAbove() { if (_instance != null && _animationControl2D != null) { _focusedTierIndex = Math.Min(_tiers.Count - 1, _focusedTierIndex + 1); UpdateFocusedTier(); } }
        private void FocusTierBelow() { if (_instance != null && _animationControl2D != null) { _focusedTierIndex = Math.Max(0, _focusedTierIndex - 1); UpdateFocusedTier(); } }
        private void StartSimulation()
        {
            // Check whether there is an active simulation
            if (!_paused)
            {
                if (_instanceInvalidated)
                {
                    // Instance was invalidated
                    MessageBox.Show("Instance was invalidated by an operation and cannot execute anymore.\nIf you modified the instance, please save it to a file and reload it to make sure initialization was done correctly.", "Invalid instance", MessageBoxButton.OK, MessageBoxImage.Error);
                    return;
                }
                if (_instance != null)
                {
                    // Log start
                    LogLine(""); LogLine("");
                    LogLine("<<< Starting simulation >>>");
                    // Handle UI elements
                    DisableButtons();
                    // Execute
                    ExecuteInstance();
                }
            }
            else
            {
                // Handle controls
                FlipPausePlayButton();
                // Resume the simulation
                _renderer.ResumeSimulation();
                // Mark unpaused
                _paused = false;
            }
        }
        private void PauseSimulation()
        {
            _paused = true;
            FlipPausePlayButton();
            _renderer.PauseSimulation();
        }
        private void SetUpdaterate(double rate) { this.Dispatcher.Invoke(() => { TextBlockUpdateSpeed.Text = rate.ToString(IOConstants.FORMATTER); }); }
        private double GetUpdaterate() { double value = 1.0; this.Dispatcher.Invoke(() => { value = double.Parse(TextBlockUpdateSpeed.Text, IOConstants.FORMATTER); }); return value; }
        private void UpdateSimulationTime(double time) { TimeSpan simulationTime = TimeSpan.FromSeconds(time); this.Dispatcher.Invoke(() => { TextBlockSolutionTime.Text = simulationTime.ToString(IOConstants.TIMESPAN_FORMAT_HUMAN_READABLE_DAYS); }); }
        private void InitVisuals()
        {
            // Mark the attached visualization
            _instance.SettingConfig.VisualizationAttached = true;
            // Init info manager
            _infoControl = new SimulationInfoManager(
                TreeViewInfoPanel, // The panel hosting all the infos
                _instance // The instance to manage
                );
            // Update tier info
            _tiers = _instance.GetInfoTiers().ToList();
            ComboBoxTierSelection2D.Items.Clear();
            foreach (var tier in _tiers)
                ComboBoxTierSelection2D.Items.Add(tier.GetInfoID());
            ComboBoxTierSelection2D.SelectedIndex = 0;
            TextBlockTierInFocus.Text = _tiers[_focusedTierIndex].GetInfoID().ToString();
            // Init drawing - 2D
            _animationControl2D = new SimulationAnimation2D(
                _instance, // The instance to visualize
                this.Dispatcher, // The dispatcher handling the calls to the UI thread
                new SimulationAnimationConfig()
                {
                    // The level of detail for drawing
                    DetailLevel = (DetailLevel)Enum.Parse(typeof(DetailLevel), ComboBoxDrawMode2D.Text),
                    // Draw goal marker?
                    DrawGoal = CheckBoxVisDrawGoalMarker.IsChecked == true,
                    // Draw destination marker?
                    DrawDestination = CheckBoxVisDrawDestMarker.IsChecked == true,
                    // Draw path marker?
                    DrawPath = CheckBoxVisDrawPathMarker.IsChecked == true,
                },
                () => { return _heatMode; }, // The function declaring the current heat-mode
                () => { return _botColorMode; }, // The function passing the current bot coloring mode
                content, // The content element containing all the visual objects
                theGrid, // The parent of the content element
                Rectangle_MouseDown, // The event handler to register with all clickable objects
                _infoControl, // The manager of the different info objects
                _tiers[0] // The current tier to draw (begin with the base)
                );
            // Init drawing - 3D
            _animationControl3D = new SimulationAnimation3D(
                _instance, // The instance to visualize
                this.Dispatcher, // The dispatcher handling the calls to the UI thread
                new SimulationAnimationConfig()
                {
                    // The level of detail for drawing
                    DetailLevel = (DetailLevel)Enum.Parse(typeof(DetailLevel), ComboBoxDrawMode3D.Text),
                    // Draw goal marker?
                    DrawGoal = CheckBoxVisDrawGoalMarker.IsChecked == true,
                    // Draw destination marker?
                    DrawDestination = CheckBoxVisDrawDestMarker.IsChecked == true,
                    // Draw path marker?
                    DrawPath = CheckBoxVisDrawPathMarker.IsChecked == true,
                },
                () => { return _heatMode; }, // The function declaring the current heat-mode
                () => { return _botColorMode; }, // The function passing the current bot coloring mode
                Viewport3D, // The content element containing all the visual objects
                _infoControl // The manager of the different info objects
                );
            // Set the initial heatmode
            UpdateHeatmode();
            // Initialize
            _animationControl2D.Init();
            _animationControl3D.Init();
            _infoControl.Init();
        }
        private void InitComboboxContent()
        {
            // Init heat color mode enumeration in combobox
            ComboBoxHeatMode.Items.Clear();
            foreach (var item in Enum.GetNames(typeof(HeatMode)))
                ComboBoxHeatMode.Items.Add(item);
            ComboBoxHeatMode.SelectedIndex = 0;
            // Init bot color mode enumeration in combobox
            ComboBoxBotColoringMode.Items.Clear();
            foreach (var item in Enum.GetNames(typeof(BotColorMode)))
                ComboBoxBotColoringMode.Items.Add(item);
            ComboBoxBotColoringMode.SelectedIndex = 0;
        }
        private void DisableButtons()
        {
            this.Dispatcher.Invoke(() =>
            {
                ButtonStart.IsEnabled = false; ButtonStart.Visibility = Visibility.Collapsed; ImageStart.Visibility = Visibility.Collapsed;
                ButtonPause.IsEnabled = true; ButtonPause.Visibility = Visibility.Visible; ImagePause.Visibility = Visibility.Visible;
                ButtonStop.IsEnabled = true; ImageStop.Visibility = Visibility.Visible;
                ButtonLoadInstance.IsEnabled = false; ImageLoadInstance.Visibility = Visibility.Hidden;
                ButtonGenerateInstance.IsEnabled = false; ImageGenerateInstance.Visibility = Visibility.Hidden;
                ButtonSanityCheck.IsEnabled = false; ImageSanityCheck.Visibility = Visibility.Hidden;
                ButtonSaveInstance.IsEnabled = false; ImageSaveInstance.Visibility = Visibility.Hidden;
                ButtonSaveLayout.IsEnabled = false; ImageSaveLayout.Visibility = Visibility.Hidden;
                ButtonSaveSettingConfiguration.IsEnabled = false; ImageSaveSettingConfiguration.Visibility = Visibility.Hidden;
                ButtonSaveControlConfiguration.IsEnabled = false; ImageSaveControlConfiguration.Visibility = Visibility.Hidden;
                ButtonSaveOrderList.IsEnabled = false; ImageSaveOrderList.Visibility = Visibility.Hidden;
                ButtonLoadHeatInstance.IsEnabled = false; ImageLoadHeatInstance.Visibility = Visibility.Hidden;
                ButtonLoadHeatInfo.IsEnabled = false; ImageLoadHeatInfo.Visibility = Visibility.Hidden;
                ButtonHeatRedraw.IsEnabled = false; ImageHeatRedraw.Visibility = Visibility.Hidden;
                ButtonHeatDrawAll.IsEnabled = false; ImageHeatDrawAll.Visibility = Visibility.Hidden;
                ButtonHeatDrawTimeWindows.IsEnabled = false; ImageHeatDrawTimeWindows.Visibility = Visibility.Hidden;
                _remoteButton.IsEnabled = false; _remoteImage.Visibility = Visibility.Hidden;
            });
        }
        private void EnableButtons()
        {
            this.Dispatcher.Invoke(() =>
            {
                ButtonStart.IsEnabled = true; ButtonStart.Visibility = Visibility.Visible; ImageStart.Visibility = Visibility.Visible;
                ButtonPause.IsEnabled = false; ButtonPause.Visibility = Visibility.Collapsed; ImagePause.Visibility = Visibility.Collapsed;
                ButtonStop.IsEnabled = false; ImageStop.Visibility = Visibility.Hidden;
                ButtonLoadInstance.IsEnabled = true; ImageLoadInstance.Visibility = Visibility.Visible;
                ButtonGenerateInstance.IsEnabled = true; ImageGenerateInstance.Visibility = Visibility.Visible;
                ButtonSanityCheck.IsEnabled = true; ImageSanityCheck.Visibility = Visibility.Visible;
                ButtonSaveInstance.IsEnabled = true; ImageSaveInstance.Visibility = Visibility.Visible;
                ButtonSaveLayout.IsEnabled = true; ImageSaveLayout.Visibility = Visibility.Visible;
                ButtonSaveSettingConfiguration.IsEnabled = true; ImageSaveSettingConfiguration.Visibility = Visibility.Visible;
                ButtonSaveControlConfiguration.IsEnabled = true; ImageSaveControlConfiguration.Visibility = Visibility.Visible;
                ButtonSaveOrderList.IsEnabled = true; ImageSaveOrderList.Visibility = Visibility.Visible;
                ButtonLoadHeatInstance.IsEnabled = true; ImageLoadHeatInstance.Visibility = Visibility.Visible;
                ButtonLoadHeatInfo.IsEnabled = true; ImageLoadHeatInfo.Visibility = Visibility.Visible;
                ButtonHeatRedraw.IsEnabled = true; ImageHeatRedraw.Visibility = Visibility.Visible;
                ButtonHeatDrawAll.IsEnabled = true; ImageHeatDrawAll.Visibility = Visibility.Visible;
                ButtonHeatDrawTimeWindows.IsEnabled = true; ImageHeatDrawTimeWindows.Visibility = Visibility.Visible;
                _remoteButton.IsEnabled = true; _remoteImage.Visibility = Visibility.Visible;
            });
        }
        private void FlipPausePlayButton()
        {
            if (ButtonStart.IsEnabled == true)
            {
                ButtonStart.IsEnabled = false; ButtonStart.Visibility = Visibility.Collapsed; ImageStart.Visibility = Visibility.Collapsed;
                ButtonPause.IsEnabled = true; ButtonPause.Visibility = Visibility.Visible; ImagePause.Visibility = Visibility.Visible;
            }
            else
            {
                ButtonStart.IsEnabled = true; ButtonStart.Visibility = Visibility.Visible; ImageStart.Visibility = Visibility.Visible;
                ButtonPause.IsEnabled = false; ButtonPause.Visibility = Visibility.Collapsed; ImagePause.Visibility = Visibility.Collapsed;
            }
        }
        private void ResetView()
        {
            if (_instance != null)
            {
                // Look at the instance
                CameraHelper.LookAt(
                    Viewport3D.Camera,
                    new Point3D(_instance.GetInfoTiers().Average(t => (t.GetInfoTLX() + t.GetInfoLength()) / 2.0), _instance.GetInfoTiers().Average(t => (t.GetInfoTLY() + t.GetInfoWidth()) / 2.0), 0),
                    50,
                    0);
            }
            else
            {
                // Look at origin
                CameraHelper.LookAt(Viewport3D.Camera, new Point3D(0, 0, 0), 0);
            }
        }
        private void ExecuteInstance()
        {
            // Indicate execution
            _running = true;
            // Find the specified word-list file (just-in-case)
            if (_instance.SettingConfig.InventoryConfiguration.ColoredWordConfiguration != null)
                _instance.SettingConfig.InventoryConfiguration.ColoredWordConfiguration.WordFile =
                    IOHelper.FindResourceFile(_instance.SettingConfig.InventoryConfiguration.ColoredWordConfiguration.WordFile, ".");
            // Find the specified order-list file (just-in-case)
            if (_instance.SettingConfig.InventoryConfiguration.FixedInventoryConfiguration != null)
                _instance.SettingConfig.InventoryConfiguration.FixedInventoryConfiguration.OrderFile =
                    IOHelper.FindResourceFile(_instance.SettingConfig.InventoryConfiguration.FixedInventoryConfiguration.OrderFile, ".");
            // Attach log-action
            _instance.SettingConfig.LogAction = LogLine;
            // Init animation
            InitVisuals();
            // Update stats directory to currently specified one
            if (CheckBoxWriteStatistics.IsChecked == true && Directory.Exists(TextStatisicsOutputFolder.Text))
                _instance.SettingConfig.StatisticsDirectory = TextStatisicsOutputFolder.Text;
            // Execute the simulation
            _renderer = new SimulationVisualizer(_instance, _animationControl2D, _animationControl3D, _infoControl, GetDrawMode3D, SetUpdaterate, GetUpdaterate(), UpdateSimulationTime, LogLine, StopExecution);
            _renderer.VisualizeSimulation(_instance.SettingConfig.SimulationWarmupTime + _instance.SettingConfig.SimulationDuration);
        }
        private void StopExecution()
        {
            this.Dispatcher.Invoke(() =>
            {
                // Handle UI elements
                EnableButtons();
                _renderer.StopSimulation();
                // Stop animation
                if (_viewMode == ViewMode.View2D)
                    _animationControl2D.StopAnimation();
                else
                    _animationControl3D.StopAnimation();
                // Remove any possible pause-marker
                _paused = false;
                //write statistics
                if (CheckBoxWriteStatistics.IsChecked == true && Directory.Exists(TextStatisicsOutputFolder.Text))
                {
                    _instance.SettingConfig.StatisticsDirectory = TextStatisicsOutputFolder.Text;
                    _instance.WriteStatistics();
                }
                // Indicate stop
                _running = false;
            });
        }
        private void SanityCheckAndShowInfo()
        {
            // Check whether we have an instance at all
            if (_instance != null)
            {
                try
                {
                    // Check for obvious errors
                    List<Tuple<Instance.SanityError, string>> errors = _instance.SanityCheck();
                    if (errors.Any())
                    {
                        // Show some info about the recognized errors
                        StringBuilder warningMsg = new StringBuilder("Sanity checking the instance returned negative!\nThe following errors were found:\n");
                        foreach (var errorType in errors.Select(e => e.Item1).Distinct())
                            warningMsg.AppendLine(errorType.ToString() + ": " + _instance.SanityGetDescription(errorType));
                        warningMsg.AppendLine("The detailed error list follows:");
                        foreach (var error in errors)
                            warningMsg.AppendLine(error.Item1.ToString() + ": " + error.Item2);
                        MessageBox.Show(warningMsg.ToString(), "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
                    }
                    else
                    {
                        // Show success
                        MessageBox.Show("No errors encountered!", "Success", MessageBoxButton.OK, MessageBoxImage.Information);
                    }
                }
                catch (Exception ex)
                {
                    string msg =
                        "An exception occurred while sanity-checking - Something went horribly wrong!\n" +
                        "Here are some details:\n" +
                        "Message: " + ex.Message + "\n" +
                        "Stacktrace:\n" + ex.StackTrace + "\n" +
                        "Inner:\n" + ex.InnerException;
                    MessageBox.Show(msg, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }
        private BitmapImage ConvertBitmapResource(System.Drawing.Bitmap bitmap)
        {
            BitmapImage bitmapImage = new BitmapImage();
            using (MemoryStream memory = new MemoryStream())
            {
                bitmap.Save(memory, ImageFormat.Png);
                memory.Position = 0;
                bitmapImage.BeginInit();
                bitmapImage.StreamSource = memory;
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.EndInit();
            }
            return bitmapImage;
        }

        #endregion

        #region Additional events

        void MainWindow_KeyDown(object sender, KeyEventArgs e)
        {
            switch (e.Key)
            {
                case Key.OemMinus:
                case Key.Subtract: DecreaseSpeed(); break;
                case Key.OemPlus:
                case Key.Add: IncreaseSpeed(); break;
                case Key.D0:
                case Key.NumPad0: SetSpeed(1); break;
                case Key.D1:
                case Key.NumPad1: SetSpeed(50); break;
                case Key.D2:
                case Key.NumPad2: SetSpeed(100); break;
                case Key.D3:
                case Key.NumPad3: SetSpeed(200); break;
                case Key.D4:
                case Key.NumPad4: SetSpeed(400); break;
                case Key.Space: if (!_paused && _instance != null && _renderer != null) PauseSimulation(); else StartSimulation(); break;
                case Key.PageUp: FocusTierAbove(); break;
                case Key.PageDown: FocusTierBelow(); break;
                default: break;
            }
        }
        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            TryLoadFromCommandLine();
        }

        /// <summary>
        /// Centres the 2D view on a canvas point at a readable scale.
        /// <para>
        /// Used by the command-line demo path: the live vehicle is at a known node, and a
        /// fitted view of the whole 140x90 m workshop leaves a single 20 px marker lost in it.
        /// Zooming to that point is the difference between "the link works" and "you can see
        /// it working".
        /// </para>
        /// </summary>
        public void CenterMapOn(double canvasX, double canvasY, double scale = 1.2)
        {
            if (zoomAndPanControl == null) return;
            try
            {
                zoomAndPanControl.ContentScale = scale;
                // Aim at the centre of the visible rectangle, not of the control: the control
                // panel covers the left ~364 DIP, so centring on the control hides the vehicle.
                var occlusion = GetMapOcclusion();
                var visW = Math.Max(1.0, zoomAndPanControl.ActualWidth - occlusion.Left - occlusion.Right);
                var visH = Math.Max(1.0, zoomAndPanControl.ActualHeight - occlusion.Top - occlusion.Bottom);
                zoomAndPanControl.ContentOffsetX = canvasX - (occlusion.Left + visW / 2.0) / scale;
                zoomAndPanControl.ContentOffsetY = canvasY - (occlusion.Top + visH / 2.0) / scale;
                LogLine("[load] 2D 视图已居中: canvas=(" + canvasX.ToString("F0") + ","
                    + canvasY.ToString("F0") + ") scale=" + scale.ToString("F3")
                    + " offset=" + zoomAndPanControl.ContentOffsetX.ToString("F0")
                    + "," + zoomAndPanControl.ContentOffsetY.ToString("F0"));
            }
            catch (Exception ex) { LogLine("[load] 居中失败: " + ex.Message); }
        }

        /// <summary>Refits the 2D map whenever the window or the map column changes size.</summary>
        private void FitLiveMapOnce(object sender, SizeChangedEventArgs e)
        {
            FitLiveMapNow();
        }

        /// <summary>
        /// How much of the 2D map is covered by opaque panels drawn on top of it.
        /// <para>
        /// The map is a full-window background (the scroller spans the control, splitter and
        /// map columns), so the control panel and the console are painted over it. Fitting or
        /// centring against the control's full width therefore aims at a strip the user cannot
        /// see: measured, a vehicle centred on the control landed at screen x=448 px while the
        /// control panel covered everything left of x=730 px.
        /// </para>
        /// </summary>
        private Thickness GetMapOcclusion()
        {
            double left = 0.0, right = 0.0, top = 0.0, bottom = 0.0;
            if (zoomAndPanControl == null) return new Thickness(0);
            var w = zoomAndPanControl.ActualWidth;
            var h = zoomAndPanControl.ActualHeight;
            if (w <= 1.0 || h <= 1.0) return new Thickness(0);
            // Measure the overlap instead of assuming a fixed panel width: the layout is what
            // decides how much is hidden, and a hard-coded number silently stops being true the
            // moment a column width or a splitter changes. The window is the common ancestor:
            // the panels are siblings of the map column, not descendants of the map control, so
            // TransformToAncestor(zoomAndPanControl) would throw for exactly the cases that
            // matter. If the layout ever changes so the map no longer spans the panel column,
            // the intersection is empty and this returns zeroes, which is the honest answer.
            Rect mapRect;
            try
            {
                var tl = zoomAndPanControl.TransformToAncestor(this).Transform(new Point(0, 0));
                mapRect = new Rect(tl.X, tl.Y, w, h);
            }
            catch { return new Thickness(0); }
            foreach (var occluder in new System.Windows.FrameworkElement[] { ControlPanelTabs, TextBoxOutput })
            {
                if (occluder == null || !occluder.IsVisible) continue;
                if (occluder.ActualWidth <= 1.0 || occluder.ActualHeight <= 1.0) continue;
                Rect r;
                try
                {
                    var tl = occluder.TransformToAncestor(this).Transform(new Point(0, 0));
                    r = new Rect(tl.X, tl.Y, occluder.ActualWidth, occluder.ActualHeight);
                }
                catch { continue; }
                r.Intersect(mapRect);
                if (r.IsEmpty || r.Width <= 0.5 || r.Height <= 0.5) continue;
                // Only count an occluder that reaches an edge of the map, and only along the axis
                // it does not itself span. The console log box spans the full width, so it touches
                // both the left and the right edge; counting it on both left the visible width at
                // one pixel and the fit collapsed to scale 0.00018. A full-width band can only
                // take height, and a full-height band can only take width.
                var spansWidth = r.Width >= mapRect.Width - 1.0;
                var spansHeight = r.Height >= mapRect.Height - 1.0;
                if (!spansWidth && r.Left <= mapRect.Left + 1.0) left = Math.Max(left, r.Right - mapRect.Left);
                if (!spansWidth && r.Right >= mapRect.Right - 1.0) right = Math.Max(right, mapRect.Right - r.Left);
                if (!spansHeight && r.Top <= mapRect.Top + 1.0) top = Math.Max(top, r.Bottom - mapRect.Top);
                if (!spansHeight && r.Bottom >= mapRect.Bottom - 1.0) bottom = Math.Max(bottom, mapRect.Bottom - r.Top);
            }
            return new Thickness(left, top, right, bottom);
        }

        /// <summary>Fits the 2D map to the viewport and reports the result to the log.</summary>
        private void FitLiveMapNow()
        {
            if (zoomAndPanControl == null) return;
            try
            {
                // Every layout pass is a chance to fit, not just the first one. Switching to the
                // 远程控制 tab makes its panel wider, which squeezes the map column from 783 DIP
                // to about 105; a fit made once at load time then describes a window that no
                // longer exists, and the map is left as a sliver on the right. Refitting on each
                // SizeChanged keeps the picture matched to whatever column it currently has.
                var viewport = zoomAndPanControl.ContentViewportWidth;
                if (viewport <= 1.0) return;   // layout not done yet, wait for the next pass
                // ScaleToFit() derives the scale from ContentViewportHeight, and on this control
                // that value can still equal the content height (3600) after layout. ZoomTo then
                // takes Math.Min(scaleX, 1.0), so the map kept its full height, hanging far off
                // the bottom of a 480-DIP window, and the visible strip was the top-left corner.
                // Compute the fit from the control's measured size instead: it is the thing the
                // user can actually see, whatever the viewport property happens to hold.
                var controlW = zoomAndPanControl.ActualWidth;
                var controlH = zoomAndPanControl.ActualHeight;
                var contentElement = zoomAndPanControl.Content as System.Windows.FrameworkElement;
                var contentW = Math.Max(1.0, contentElement?.ActualWidth ?? 0.0);
                var contentH = Math.Max(1.0, contentElement?.ActualHeight ?? 0.0);
                // The map control spans the whole window, but the control panel and the console
                // are painted over its left and bottom edges. Fitting against the full control
                // therefore aims at a strip the user cannot see: measured, a centred vehicle
                // landed at screen x=448 px while the panel covered everything left of x=730 px.
                // Fit and centre inside the visible sub-rectangle instead. Do not try to express
                // this as a Margin on the control: inside a ScrollViewer a margin only shifts
                // the content, it does not shrink the measured width.
                var occlusion = GetMapOcclusion();
                var visibleLeft = occlusion.Left;
                var visibleTop = 0.0;
                var visW = Math.Max(1.0, controlW - occlusion.Left - occlusion.Right);
                var visH = Math.Max(1.0, controlH - occlusion.Top - occlusion.Bottom);
                if (controlW > 1.0 && controlH > 1.0)
                {
                    var fit = Math.Min(visW / contentW, visH / contentH);
                    zoomAndPanControl.ContentScale = fit;
                    // Screen position of a content coordinate x is (x - offset) * scale. Put the
                    // centre of the content at the centre of the *visible* rectangle.
                    zoomAndPanControl.ContentOffsetX =
                        contentW / 2.0 - (visibleLeft + visW / 2.0) / fit;
                    zoomAndPanControl.ContentOffsetY =
                        contentH / 2.0 - (visibleTop + visH / 2.0) / fit;
                    LogLine("[load] 2D 视图已适配: scale=" + fit.ToString("F5")
                        + " 可见=" + visW.ToString("F0") + "x" + visH.ToString("F0")
                        + " left=" + visibleLeft.ToString("F0") + " bottom=" + occlusion.Bottom.ToString("F0")
                        + " control=" + controlW.ToString("F0") + "x" + controlH.ToString("F0")
                        + " content=" + contentW.ToString("F0") + "x" + contentH.ToString("F0")
                        + " offset=" + zoomAndPanControl.ContentOffsetX.ToString("F0")
                        + "," + zoomAndPanControl.ContentOffsetY.ToString("F0")
                        + " 窗口=" + ActualWidth + "x" + ActualHeight);
                }
                else
                {
                    LogLine("[load] 2D 视图尚未布局, 暂不适配: control=" + controlW + "x" + controlH);
                    _fitOnNextLayout = true;
                    return;
                }
            }
            catch (Exception fitEx) { LogLine("[load] 适配 2D 视图失败: " + fitEx.Message); }
        }

        /// <summary>Whether the pending 2D fit still has to run.</summary>
        private bool _fitOnNextLayout;

        /// <summary>
        /// Loads an instance from a path given on the command line, so the viewer can be
        /// driven without the three OpenFileDialogs behind ButtonLoadInstance. In this mode
        /// the instance is read as a visualization-only layout: this window is a display and
        /// interaction layer, so it must not spin up the inventory/order simulation of
        /// RAWSimO.Core (which requires pods and would fabricate material flow this scene
        /// does not model).
        ///
        /// Usage:
        ///   RAWSimO.Visualization.exe --map &lt;instance.xinst&gt; [--setting &lt;setting.xsett&gt;] [--control &lt;control.xconf&gt;]
        ///                             [--live &lt;host:port&gt;] [--schedule &lt;host:port&gt;]
        ///
        /// --live connects the single-vehicle truth overlay, --schedule connects the fleet
        /// scheduling stream. Both then run without any further interaction, which is what
        /// makes an unattended two-machine demo possible.
        /// </summary>
        public bool TryLoadFromCommandLine(string[] args = null)
        {
            args = args ?? Environment.GetCommandLineArgs();
            string instancePath = null, settingPath = null, controlPath = null;
            string liveHostPort = null, scheduleHostPort = null;
            int? liveRouteWaypoint = null;
            for (int i = 1; i < args.Length; i++)
            {
                switch ((args[i] ?? string.Empty).Trim().ToLowerInvariant())
                {
                    case "--map":
                    case "--instance":
                        if (i + 1 < args.Length) instancePath = args[++i];
                        break;
                    case "--setting":
                    case "--xsett":
                        if (i + 1 < args.Length) settingPath = args[++i];
                        break;
                    case "--control":
                    case "--xconf":
                        if (i + 1 < args.Length) controlPath = args[++i];
                        break;
                    case "--live":
                        if (i + 1 < args.Length) liveHostPort = args[++i];
                        break;
                    case "--schedule":
                        if (i + 1 < args.Length) scheduleHostPort = args[++i];
                        break;
                    case "--live-route":
                        // Ask the bridge for a route to this waypoint as soon as the link is up.
                        // The route is planned in Python; this only triggers the request.
                        if (i + 1 < args.Length && int.TryParse(args[++i], out var liveRouteTarget))
                            liveRouteWaypoint = liveRouteTarget;
                        break;
                }
            }
            if (string.IsNullOrWhiteSpace(instancePath)) return false;
            if (!System.IO.File.Exists(instancePath))
            {
                LogLine("[load] 找不到实例文件: " + instancePath);
                return false;
            }
            LoadInstanceForVisualization(instancePath, settingPath, controlPath);
            // Connect after the map exists, so the first poses land on a fitted canvas
            // instead of being dropped while the scene is still null.
            if (!string.IsNullOrWhiteSpace(liveHostPort))
            {
                // The fitted overview is the honest default now that the marker has a constant
                // screen size: it shows the whole workshop with the vehicle in it. Auto-centring
                // is still available for a close-up, but it must be asked for.
                AutoCenterOnFirstPose =
                    Environment.GetEnvironmentVariable("AMR_AUTOCENTER") == "1";
                Dispatcher.BeginInvoke(new Action(() => AutoConnectLive(liveHostPort)),
                    System.Windows.Threading.DispatcherPriority.ContextIdle);
            }
            if (!string.IsNullOrWhiteSpace(scheduleHostPort))
                Dispatcher.BeginInvoke(new Action(() => AutoConnectSchedule(scheduleHostPort)),
                    System.Windows.Threading.DispatcherPriority.ContextIdle);
            if (liveRouteWaypoint.HasValue)
                Dispatcher.BeginInvoke(new Action(() => RequestLiveRouteWhenConnected(liveRouteWaypoint.Value)),
                    System.Windows.Threading.DispatcherPriority.ContextIdle);
            return true;
        }

        /// <summary>
        /// Reads an instance in visualization-only mode and shows it. The same code path the
        /// manual dialog would take, minus the dialogs and minus the simulation managers.
        /// </summary>
        public bool LoadInstanceForVisualization(string instancePath, string settingPath, string controlPath)
        {
            try
            {
                // An .xsett is optional in visualization-only mode; when one is given it is
                // still parsed by InstanceIO only if visualizationOnly is false, so pass a
                // fallback that keeps the call valid either way.
                var sett = string.IsNullOrWhiteSpace(settingPath) ? instancePath : settingPath;
                var conf = string.IsNullOrWhiteSpace(controlPath) ? instancePath : controlPath;
                _instance = InstanceIO.ReadInstance(instancePath, sett, conf, true, true, logAction: LogLine);
                _instanceInvalidated = false;
                LogLine("[load] 已加载(仅可视化): " + System.IO.Path.GetFileName(instancePath)
                    + " 路点=" + _instance.Waypoints.Count
                    + " 车=" + _instance.Bots.Count
                    + " 上料站=" + _instance.InputStations.Count
                    + " 下料站=" + _instance.OutputStations.Count);
                // The window opens in whatever mode the XAML defaults to and the detail
                // combo defaults to "None", which never draws the waypoint graph. Both would
                // leave the map blank even though the instance loaded, so pick the mode the
                // command-line viewer is for: the 2D map with the full waypoint graph.
                To2DView();
                if (ComboBoxDrawMode2D.Items.Count > 1)
                    ComboBoxDrawMode2D.SelectedIndex = 1;   // Debug = draw the complete waypoint graph
                InitVisuals();
                To2DView();
                ResetView();
                // Shrink into the work area first. The window is 1000x800 DIP while the screen
                // here is 800x500 logical pixels (200% display scaling), so a default-sized
                // window hangs off the right edge and the map looks cropped. Fit the window
                // before fitting the map into it.
                try
                {
                    var work = SystemParameters.WorkArea;
                    var dpi = System.Windows.Media.VisualTreeHelper.GetDpi(this).DpiScaleX;
                    if (dpi <= 0) dpi = 1.0;
                    // WorkArea is already in DIPs. WPF is per-monitor DPI aware, so on this
                    // machine (3200x2000 physical at dpi=2) it reports 1600x952 DIP - the same
                    // unit as Window.Width. Dividing by the DPI scale again halved the window
                    // to 796x480, which left the map column about 304 DIP wide: enough to draw
                    // the 140x90 m network, too narrow to read it.
                    var maxW = Math.Max(640.0, work.Width - 4);
                    var maxH = Math.Max(480.0, work.Height - 4);
                    Width = Math.Min(Width, maxW);
                    Height = Math.Min(Height, maxH);
                    Left = work.Left + 2;
                    Top = work.Top + 2;
                    LogLine("[load] 工作区=" + work.Width + "x" + work.Height + " dpi=" + dpi
                        + " 新窗口=" + Width + "x" + Height + " @ " + Left + "," + Top);
                }
                catch (Exception sizeEx) { LogLine("[load] 调整窗口尺寸失败: " + sizeEx.Message); }

                // Centre the 2D view on the loaded map. ResetView() only moves the 3D camera;
                // without this the 2D canvas stays at scale 1 anchored to its top-left corner,
                // so a 140x90 m map at 40 px/m (5600x3600 px) shows only a corner patch that
                // happens to be empty - the map looks blank although it drew correctly.
                // ScaleToFit() reads the viewport size, which is only correct once layout ran at
                // the final window size. Retry on the first SizeChanged and then once more after
                // the layout pass, so the fit is not computed against a stale 0-sized viewport.
                _fitOnNextLayout = true;
                SizeChanged += FitLiveMapOnce;
                Dispatcher.BeginInvoke(new Action(FitLiveMapNow), System.Windows.Threading.DispatcherPriority.ContextIdle);
                Dispatcher.BeginInvoke(new Action(() =>
                {
                    try
                    {
                        var sb = new System.Text.StringBuilder();
                        sb.Append("[load] 诊断: drawMode2D=").Append(ComboBoxDrawMode2D.Text);
                        sb.Append(" scroller=").Append(scroller.Visibility);
                        sb.Append(" contentChildren=").Append(content.Children.Count);
                        sb.Append(" contentActual=").Append(content.ActualWidth).Append("x").Append(content.ActualHeight);
                        sb.Append(" theGridActual=").Append(theGrid.ActualWidth).Append("x").Append(theGrid.ActualHeight);
                        sb.Append(" scale=").Append(zoomAndPanControl.ContentScale);
                        sb.Append(" offset=").Append(zoomAndPanControl.ContentOffsetX).Append(",").Append(zoomAndPanControl.ContentOffsetY);
                        foreach (var child in content.Children.OfType<System.Windows.Shapes.Shape>().Take(3))
                            sb.Append(" [shape ").Append(child.GetType().Name)
                              .Append(" @").Append(System.Windows.Controls.Canvas.GetLeft(child)).Append(",").Append(System.Windows.Controls.Canvas.GetTop(child))
                              .Append(" ").Append(child.Width).Append("x").Append(child.Height)
                              .Append(" vis=").Append(child.Visibility).Append("]");
                        foreach (var img in content.Children.OfType<System.Windows.Controls.Image>().Take(2))
                            sb.Append(" [img ").Append(img.Width).Append("x").Append(img.Height)
                              .Append(" src=").Append(img.Source != null ? img.Source.Width + "x" + img.Source.Height : "null").Append("]");
                        LogLine(sb.ToString());
                        // Every drawn element in the 2D map area, with its real layout bounds. The
                        // map looked blank while this list said 31 children existed: knowing where
                        // they sit (and at what size) is the only way to tell a clipped/undersized
                        // visual from a missing one.
                        var bounds = new System.Text.StringBuilder("[load] 地图元素: ");
                        foreach (var child in content.Children)
                        {
                            var el = child as System.Windows.FrameworkElement;
                            if (el == null) continue;
                            bounds.Append(el.GetType().Name);
                            if (!string.IsNullOrEmpty(el.Name)) bounds.Append("#").Append(el.Name);
                            bounds.Append(" @").Append(el.Width).Append("x").Append(el.Height)
                                  .Append("(").Append(double.IsNaN(el.ActualWidth) ? "NaN" : el.ActualWidth.ToString("F0"))
                                  .Append("x").Append(double.IsNaN(el.ActualHeight) ? "NaN" : el.ActualHeight.ToString("F0")).Append(") ");
                        }
                        LogLine(bounds.ToString());
                    }
                    catch (Exception diagEx) { LogLine("[load] 诊断失败: " + diagEx.GetType().Name + ": " + diagEx.Message); }
                }), System.Windows.Threading.DispatcherPriority.Loaded);
                return true;
            }
            catch (Exception ex)
            {
                _instance = null;
                LogLine("[load] 加载失败: " + ex.GetType().Name + ": " + ex.Message);
                return false;
            }
        }
        private void Window_Closed(object sender, EventArgs e)
        {
            if (_running)
            {
                StopExecution();
            }
        }
        private void ComboBoxHeatMode_SelectionChanged(object sender, SelectionChangedEventArgs e) { UpdateHeatmode(); UpdateFocusedTier(); }
        private void ComboBoxTierSelection2D_SelectionChanged(object sender, SelectionChangedEventArgs e) { _focusedTierIndex = ComboBoxTierSelection2D.SelectedIndex; UpdateFocusedTier(); }
        private void ComboBoxBotColoringMode_SelectionChanged(object sender, SelectionChangedEventArgs e) { UpdateBotColoringMode(); }

        #endregion

        #region Button clicks

        private void ButtonStart_Click(object sender, RoutedEventArgs e) { StartSimulation(); }

        private void ButtonPause_Click(object sender, RoutedEventArgs e) { PauseSimulation(); }

        private void ButtonSlowDown_Click(object sender, RoutedEventArgs e) { DecreaseSpeed(); }

        private void ButtonSpeedUp_Click(object sender, RoutedEventArgs e) { IncreaseSpeed(); }

        private void ButtonStop_Click(object sender, RoutedEventArgs e) { StopExecution(); }

        private void ButtonViewMode_Click(object sender, RoutedEventArgs e)
        {
            if (_viewMode == ViewMode.View2D) To3DView();
            else To2DView();
        }

        private void ButtonLoadInstance_Click(object sender, RoutedEventArgs e)
        {
            // Create an instance of the open file dialog box.
            OpenFileDialog settingDialog = new OpenFileDialog();

            // Set filter options and filter index.
            settingDialog.Filter = "Setting files (.xsett)|*.xsett";
            settingDialog.FilterIndex = 1;
            settingDialog.Multiselect = false;

            // Call the ShowDialog method to show the dialog box.
            bool? settClick = settingDialog.ShowDialog();

            // Only go on if user clicked OK
            if (settClick == true)
            {
                // Create an instance of the open file dialog box.
                OpenFileDialog configDialog = new OpenFileDialog();

                // Set filter options and filter index.
                configDialog.Filter = "Configuration files (.xconf)|*.xconf";
                configDialog.FilterIndex = 1;
                configDialog.Multiselect = false;

                // Call the ShowDialog method to show the dialog box.
                bool? confClick = configDialog.ShowDialog();

                // Only go on if user clicked OK
                if (confClick == true)
                {
                    // Create an instance of the open file dialog box.
                    OpenFileDialog instanceDialog = new OpenFileDialog();

                    // Set filter options and filter index.
                    instanceDialog.Filter = "Instance files (.xinst,xlayo)|*.xinst;*.xlayo;";
                    instanceDialog.FilterIndex = 1;
                    instanceDialog.Multiselect = false;

                    // Call the ShowDialog method to show the dialog box.
                    bool? instClick = instanceDialog.ShowDialog();

                    // Process input if the user clicked OK.
                    if (instClick == true)
                    {
                        // Parse the instance
                        _instance = InstanceIO.ReadInstance(instanceDialog.FileName, settingDialog.FileName, configDialog.FileName, true, logAction: LogLine);
                        _instanceInvalidated = false;
                        // Show the instance
                        InitVisuals();
                        // Reset the view
                        ResetView();
                    }
                }
            }
        }

        private void ButtonSaveInstance_Click(object sender, RoutedEventArgs e)
        {
            if (_instance != null)
            {
                // Init save dialog
                SaveFileDialog dialog = new SaveFileDialog();
                dialog.FileName = _instance.GetMetaInfoBasedInstanceName(); // Default file name
                dialog.DefaultExt = ".xinst"; // Default file extension
                dialog.Filter = "XINST Files (.xinst)|*.xinst"; // Filter files by extension

                // Show save file dialog box
                bool? userClickedOK = dialog.ShowDialog();

                // Process save file dialog box results
                if (userClickedOK == true)
                {
                    // Save document
                    string filename = dialog.FileName;
                    _instance.Name = System.IO.Path.GetFileNameWithoutExtension(filename);
                    InstanceIO.WriteInstance(filename, _instance);
                }
            }
            else
            {
                MessageBox.Show("Generate an instance first.");
            }
        }

        private void ButtonSaveLayout_Click(object sender, RoutedEventArgs e)
        {
            // Parse the configuration
            ParseLayoutConfiguration();

            // Init save dialog
            SaveFileDialog dialog = new SaveFileDialog();
            dialog.FileName = _layoutConfig.GetMetaInfoBasedLayoutName(); // Default file name
            dialog.DefaultExt = ".xlayo"; // Default file extension
            dialog.Filter = "XLAYO Files (.xlayo)|*.xlayo"; // Filter files by extension

            // Show save file dialog box
            bool? userClickedOK = dialog.ShowDialog();

            // Process save file dialog box results
            if (userClickedOK == true)
            {
                // Save it
                string filename = dialog.FileName;
                _layoutConfig.NameLayout = System.IO.Path.GetFileNameWithoutExtension(filename);
                InstanceIO.WriteLayout(filename, _layoutConfig);
            }
        }

        private void ButtonSaveSettingConfiguration_Click(object sender, RoutedEventArgs e)
        {
            // Parse the configuration
            ParseSetting();

            // Init save dialog
            SaveFileDialog dialog = new SaveFileDialog();
            dialog.FileName = _baseConfiguration.Name; // Default file name
            dialog.DefaultExt = ".xsett"; // Default file extension
            dialog.Filter = "XSETT Files (.xsett)|*.xsett"; // Filter files by extension

            // Remove directory information of word-list file
            if (_baseConfiguration.InventoryConfiguration.ColoredWordConfiguration != null)
                _baseConfiguration.InventoryConfiguration.ColoredWordConfiguration.WordFile =
                    System.IO.Path.GetFileName(_baseConfiguration.InventoryConfiguration.ColoredWordConfiguration.WordFile);
            // Remove directory information of order-list file
            if (_baseConfiguration.InventoryConfiguration.FixedInventoryConfiguration != null)
                _baseConfiguration.InventoryConfiguration.FixedInventoryConfiguration.OrderFile =
                    System.IO.Path.GetFileName(_baseConfiguration.InventoryConfiguration.FixedInventoryConfiguration.OrderFile);
            // Remove directory information of simple-item file
            if (_baseConfiguration.InventoryConfiguration.SimpleItemConfiguration != null)
                _baseConfiguration.InventoryConfiguration.SimpleItemConfiguration.GeneratorConfigFile =
                    System.IO.Path.GetFileName(_baseConfiguration.InventoryConfiguration.SimpleItemConfiguration.GeneratorConfigFile);

            // Show save file dialog box
            bool? userClickedOK = dialog.ShowDialog();

            // Process save file dialog box results
            if (userClickedOK == true)
            {
                // Save it
                string filename = dialog.FileName;
                _baseConfiguration.Name = System.IO.Path.GetFileNameWithoutExtension(filename);
                InstanceIO.WriteSetting(filename, _baseConfiguration);
            }
        }

        private void ButtonSaveControlConfiguration_Click(object sender, RoutedEventArgs e)
        {
            // Parse the configuration
            ParseConfiguration();

            // Init save dialog
            SaveFileDialog dialog = new SaveFileDialog();
            dialog.FileName = _controlConfiguration.Name; // Default file name
            dialog.DefaultExt = ".xconf"; // Default file extension
            dialog.Filter = "XCONF Files (.xconf)|*.xconf"; // Filter files by extension

            // Show save file dialog box
            bool? userClickedOK = dialog.ShowDialog();

            // Process save file dialog box results
            if (userClickedOK == true)
            {
                // Save it
                string filename = dialog.FileName;
                _controlConfiguration.Name = System.IO.Path.GetFileNameWithoutExtension(filename);
                InstanceIO.WriteConfiguration(filename, _controlConfiguration);
            }
        }

        private void ButtonSaveOrderList_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Fetch parameters
                string wordFile; double timeMin; double timeMax; int seed;
                int count; int positionMin; int positionMax;
                double relativeBundleAmount; int bundleSizeMin; int bundleSizeMax;
                double probR; double probG; double probB; double probY; double weightMin; double weightMax;
                wordFile = TextBoxOrderWordFile.Text;
                try { timeMin = double.Parse(TextBoxOrderTimeMin.Text, IOConstants.FORMATTER); }
                catch (FormatException) { throw new FormatException("Error while parsing the minimal time!"); }
                try { timeMax = double.Parse(TextBoxOrderTimeMax.Text, IOConstants.FORMATTER); }
                catch (FormatException) { throw new FormatException("Error while parsing the maximal time!"); }
                try { seed = int.Parse(TextBoxOrderSeed.Text); }
                catch (FormatException) { throw new FormatException("Error while parsing the seed!"); }
                try { count = int.Parse(TextBoxOrderCount.Text); }
                catch (FormatException) { throw new FormatException("Error while parsing the order count!"); }
                try { positionMin = int.Parse(TextBoxOrderPositionMin.Text); }
                catch (FormatException) { throw new FormatException("Error while parsing the minimal position count!"); }
                try { positionMax = int.Parse(TextBoxOrderPositionMax.Text); }
                catch (FormatException) { throw new FormatException("Error while parsing the maximal position count!"); }
                try { relativeBundleAmount = double.Parse(TextBoxOrderBundlesRelativeAmount.Text, IOConstants.FORMATTER); }
                catch (FormatException) { throw new FormatException("Error while parsing the maximal weight!"); }
                try { bundleSizeMin = int.Parse(TextBoxOrderBundleSizeMin.Text, IOConstants.FORMATTER); }
                catch (FormatException) { throw new FormatException("Error while parsing the maximal weight!"); }
                try { bundleSizeMax = int.Parse(TextBoxOrderBundleSizeMax.Text, IOConstants.FORMATTER); }
                catch (FormatException) { throw new FormatException("Error while parsing the maximal weight!"); }
                try { probR = double.Parse(TextBoxOrderR.Text, IOConstants.FORMATTER); }
                catch (FormatException) { throw new FormatException("Error while parsing the probability for red!"); }
                try { probG = double.Parse(TextBoxOrderG.Text, IOConstants.FORMATTER); }
                catch (FormatException) { throw new FormatException("Error while parsing the probability for green!"); }
                try { probB = double.Parse(TextBoxOrderB.Text, IOConstants.FORMATTER); }
                catch (FormatException) { throw new FormatException("Error while parsing the probability for blue!"); }
                try { probY = double.Parse(TextBoxOrderY.Text, IOConstants.FORMATTER); }
                catch (FormatException) { throw new FormatException("Error while parsing the probability for yellow!"); }
                try { weightMin = double.Parse(TextBoxOrderWeightMin.Text, IOConstants.FORMATTER); }
                catch (FormatException) { throw new FormatException("Error while parsing the minimal weight!"); }
                try { weightMax = double.Parse(TextBoxOrderWeightMax.Text, IOConstants.FORMATTER); }
                catch (FormatException) { throw new FormatException("Error while parsing the maximal weight!"); }
                Dictionary<LetterColors, double> colorProbs = new Dictionary<LetterColors, double>();
                if (probR > 0) { colorProbs[LetterColors.Red] = probR; }
                if (probG > 0) { colorProbs[LetterColors.Green] = probG; }
                if (probB > 0) { colorProbs[LetterColors.Blue] = probB; }
                if (probY > 0) { colorProbs[LetterColors.Yellow] = probY; }
                LetterColors[] colors = colorProbs.Keys.OrderBy(c => colorProbs[c]).ToArray();
                // Find the specified word-list file
                wordFile = IOHelper.FindResourceFile(wordFile, ".");
                // Generate list
                OrderList list = OrderGenerator.GenerateOrders(wordFile, colors, colorProbs, seed, timeMin, timeMax, count, weightMin, weightMax, positionMin, positionMax, bundleSizeMin, bundleSizeMax, relativeBundleAmount);

                // Init save dialog
                SaveFileDialog dialog = new SaveFileDialog();
                dialog.FileName = list.GetMetaInfoBasedOrderListName(); // Default file name
                dialog.DefaultExt = ".xitem"; // Default file extension
                dialog.Filter = "XITEM Files (.xitem)|*.xitem"; // Filter files by extension

                // Show save file dialog box
                bool? userClickedOK = dialog.ShowDialog();

                // Process save file dialog box results
                if (userClickedOK == true)
                {
                    // Save document
                    string filename = dialog.FileName;
                    InstanceIO.WriteOrders(filename, list);
                }
            }
            catch (FormatException ex)
            {
                MessageBox.Show(this, "Failure while parsing the parameters:\n" + ex.Message, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void ButtonSaveSimpleItemGeneratorConfig_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Fetch parameters
                VisualParameterInputHelper.ParseParameterItems(Dispatcher, WrapPanelSimpleItemConfiguration.Children, _simpleItemGeneratorPreConfig);
                SimpleItemGeneratorConfiguration config = OrderGenerator.GenerateSimpleItemConfiguration(_simpleItemGeneratorPreConfig);

                // Init save dialog
                SaveFileDialog dialog = new SaveFileDialog
                {
                    FileName = config.GetMetaInfoBasedName(), // Default file name
                    DefaultExt = ".xgenc", // Default file extension
                    Filter = "XGENC Files (.xgenc)|*.xgenc" // Filter files by extension
                };

                // Show save file dialog box
                bool? userClickedOK = dialog.ShowDialog();

                // Process save file dialog box results
                if (userClickedOK == true)
                {
                    // Save document
                    string filename = dialog.FileName;
                    config.Name = System.IO.Path.GetFileNameWithoutExtension(filename);
                    InstanceIO.WriteSimpleItemGeneratorConfigFile(filename, config);
                    // Generate diagram
                    InventoryInfoProcessor.PlotSimpleInventoryFrequencies(filename);
                    // Log
                    LogLine("Saved item config to " + filename + " and generated a diagram for the file in the application directory!");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, "Failure while generating the config:\n" + ex.Message, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void ButtonGenerateInstance_Click(object sender, RoutedEventArgs e)
        {
            TextBoxOutput.Text = "";
            try
            {
                // Parse the base and layout configurations
                ParseSetting();
                ParseConfiguration();
                ParseLayoutConfiguration();

                // Find the specified word-list file
                if (_baseConfiguration.InventoryConfiguration.ColoredWordConfiguration != null)
                    _baseConfiguration.InventoryConfiguration.ColoredWordConfiguration.WordFile =
                        IOHelper.FindResourceFile(_baseConfiguration.InventoryConfiguration.ColoredWordConfiguration.WordFile, ".");
                // Find the specified order-list file
                if (_baseConfiguration.InventoryConfiguration.FixedInventoryConfiguration != null)
                    _baseConfiguration.InventoryConfiguration.FixedInventoryConfiguration.OrderFile =
                        IOHelper.FindResourceFile(_baseConfiguration.InventoryConfiguration.FixedInventoryConfiguration.OrderFile, ".");
                // Find the specified generator-config file
                if (_baseConfiguration.InventoryConfiguration.SimpleItemConfiguration != null)
                    _baseConfiguration.InventoryConfiguration.SimpleItemConfiguration.GeneratorConfigFile =
                        IOHelper.FindResourceFile(_baseConfiguration.InventoryConfiguration.SimpleItemConfiguration.GeneratorConfigFile, ".");
                // Mark the attached visualization
                _baseConfiguration.VisualizationAttached = true;

                // Generate
                _instance = InstanceGenerator.GenerateLayout(_layoutConfig, _baseConfiguration, _controlConfiguration);
                _instanceInvalidated = false;
                _instance.Name = _instance.GetMetaInfoBasedInstanceName();

                // Display
                InitVisuals();
            }
            catch (FormatException ex)
            {
                MessageBox.Show(this, "Failure while parsing the parameters:\n" + ex.Message, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            catch (ArgumentException ex)
            {
                MessageBox.Show(this, "Failure while generating the instance:\n" + ex.Message, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void ButtonSanityCheck_Click(object sender, RoutedEventArgs e)
        {
            if (_instance != null)
                SanityCheckAndShowInfo();
            else
                MessageBox.Show("No instance to check!", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
        }

        private void ButtonShowInfo_Click(object sender, RoutedEventArgs e)
        {
            if (TreeViewInfoPanel.Visibility == Visibility.Hidden)
            {
                TreeViewInfoPanel.Visibility = Visibility.Visible;
                GridSplitterInfoPanel.Visibility = Visibility.Visible;
                TextBoxOutput.Visibility = Visibility.Visible;
                GridSplitterBoxOutput.Visibility = Visibility.Visible;
                ImageShowInfo.Source = _imageDisableInfoView;
            }
            else
            {
                TreeViewInfoPanel.Visibility = Visibility.Hidden;
                GridSplitterInfoPanel.Visibility = Visibility.Hidden;
                TextBoxOutput.Visibility = Visibility.Hidden;
                GridSplitterBoxOutput.Visibility = Visibility.Hidden;
                ImageShowInfo.Source = _imageEnableInfoView;
            }
        }

        private void ButtonSnapshot_Click(object sender, RoutedEventArgs e)
        {
            if (!Directory.Exists(_snapshotDir))
                Directory.CreateDirectory(_snapshotDir);
            if (_viewMode == ViewMode.View2D)
            {
                if (_animationControl2D != null)
                    _animationControl2D.TakeSnapshot(_snapshotDir);
            }
            else
            {
                if (_animationControl3D != null)
                    _animationControl3D.TakeSnapshot(_snapshotDir);
            }
        }

        #endregion

        #region 2D Control

        /// <summary>
        /// Specifies the current state of the mouse handling logic.
        /// </summary>
        private MouseHandlingMode mouseHandlingMode = MouseHandlingMode.None;

        /// <summary>
        /// The point that was clicked relative to the ZoomAndPanControl.
        /// </summary>
        private Point origZoomAndPanControlMouseDownPoint;

        /// <summary>
        /// The point that was clicked relative to the content that is contained within the ZoomAndPanControl.
        /// </summary>
        private Point origContentMouseDownPoint;

        /// <summary>
        /// Records which mouse button clicked during mouse dragging.
        /// </summary>
        private MouseButton mouseButtonDown;

        /// <summary>
        /// Defines the current state of the mouse handling logic.
        /// </summary>
        private enum MouseHandlingMode
        {
            /// <summary>
            /// Not in any special mode.
            /// </summary>
            None,

            /// <summary>
            /// The user is drawing a rectangle to remove waypoints.
            /// </summary>
            WaypointRemoval,

            /// <summary>
            /// The user is left-mouse-button-dragging to pan the viewport.
            /// </summary>
            Panning,

            /// <summary>
            /// The user is holding down shift and left-clicking or right-clicking to zoom in or out.
            /// </summary>
            Zooming,

            /// <summary>
            /// The user is holding down shift and left-mouse-button-dragging to select a region to zoom to.
            /// </summary>
            DragZooming,
        }

        #region Event handlers

        private void ZoomControl_MouseDown(object sender, MouseButtonEventArgs e)
        {
            content.Focus();
            Keyboard.Focus(content);

            mouseButtonDown = e.ChangedButton;
            origZoomAndPanControlMouseDownPoint = e.GetPosition(zoomAndPanControl);
            origContentMouseDownPoint = e.GetPosition(content);

            if ((Keyboard.Modifiers & ModifierKeys.Shift) != 0 &&
                (e.ChangedButton == MouseButton.Left ||
                 e.ChangedButton == MouseButton.Right))
            {
                // Shift + left- or right-down initiates zooming mode.
                mouseHandlingMode = MouseHandlingMode.Zooming;
            }
            else if ((Keyboard.Modifiers & ModifierKeys.Control) != 0 &&
                mouseButtonDown == MouseButton.Left)
            {
                // Only allowed, if waypoints are visible
                if (!((DetailLevel)Enum.Parse(typeof(DetailLevel), ComboBoxDrawMode2D.Text) == DetailLevel.Full))
                {
                    MessageBox.Show("Cannot select invisible waypoints - enable drawing of waypoints first!", "Info", MessageBoxButton.OK, MessageBoxImage.Warning);
                    e.Handled = true;
                    return;
                }
                // Ctrl + left-down initiates waypoint selection
                mouseHandlingMode = MouseHandlingMode.WaypointRemoval;
            }
            else if (mouseButtonDown == MouseButton.Left)
            {
                // Just a plain old left-down initiates panning mode.
                mouseHandlingMode = MouseHandlingMode.Panning;
            }

            if (mouseHandlingMode != MouseHandlingMode.None)
            {
                // Capture the mouse so that we eventually receive the mouse up event.
                zoomAndPanControl.CaptureMouse();
                e.Handled = true;
            }
        }

        private void ZoomControl_MouseUp(object sender, MouseButtonEventArgs e)
        {
            if (mouseHandlingMode != MouseHandlingMode.None)
            {
                if (mouseHandlingMode == MouseHandlingMode.Zooming)
                {
                    if (mouseButtonDown == MouseButton.Left)
                    {
                        // Shift + left-click zooms in on the content.
                        ZoomIn(origContentMouseDownPoint);
                    }
                    else if (mouseButtonDown == MouseButton.Right)
                    {
                        // Shift + left-click zooms out from the content.
                        ZoomOut(origContentMouseDownPoint);
                    }
                }
                else if (mouseHandlingMode == MouseHandlingMode.DragZooming)
                {
                    // When drag-zooming has finished we zoom in on the rectangle that was highlighted by the user.
                    ZoomBySelectionRect();
                }
                else if (mouseHandlingMode == MouseHandlingMode.WaypointRemoval)
                {
                    // When selecting the waypoints has finished we remove them
                    RemoveWaypointsBySelectionRect();
                }

                zoomAndPanControl.ReleaseMouseCapture();
                mouseHandlingMode = MouseHandlingMode.None;
                e.Handled = true;
            }
        }

        private void ZoomControl_MouseMove(object sender, MouseEventArgs e)
        {
            if (mouseHandlingMode == MouseHandlingMode.Panning)
            {
                // The user is left-dragging the mouse.
                // Pan the viewport by the appropriate amount.
                Point curContentMousePoint = e.GetPosition(content);
                Vector dragOffset = curContentMousePoint - origContentMouseDownPoint;

                zoomAndPanControl.ContentOffsetX -= dragOffset.X;
                zoomAndPanControl.ContentOffsetY -= dragOffset.Y;

                e.Handled = true;
            }
            else if (mouseHandlingMode == MouseHandlingMode.Zooming)
            {
                Point curZoomAndPanControlMousePoint = e.GetPosition(zoomAndPanControl);
                Vector dragOffset = curZoomAndPanControlMousePoint - origZoomAndPanControlMouseDownPoint;
                double dragThreshold = 10;
                if (mouseButtonDown == MouseButton.Left &&
                    (Math.Abs(dragOffset.X) > dragThreshold ||
                     Math.Abs(dragOffset.Y) > dragThreshold))
                {
                    // When Shift + left-down zooming mode and the user drags beyond the drag threshold,
                    // initiate drag zooming mode where the user can drag out a rectangle to select the area
                    // to zoom in on.
                    mouseHandlingMode = MouseHandlingMode.DragZooming;
                    Point curContentMousePoint = e.GetPosition(content);
                    UpdateSelectionRect(origContentMouseDownPoint, curContentMousePoint);
                }

                e.Handled = true;
            }
            else if (mouseHandlingMode == MouseHandlingMode.DragZooming || mouseHandlingMode == MouseHandlingMode.WaypointRemoval)
            {
                // When in drag zooming mode continously update the position of the rectangle
                // that the user is dragging out.
                Point curContentMousePoint = e.GetPosition(content);
                UpdateSelectionRect(origContentMouseDownPoint, curContentMousePoint);

                e.Handled = true;
            }
        }

        private void ZoomControl_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            e.Handled = true;

            if (e.Delta > 0)
            {
                Point curContentMousePoint = e.GetPosition(content);
                ZoomIn(curContentMousePoint);
            }
            else if (e.Delta < 0)
            {
                Point curContentMousePoint = e.GetPosition(content);
                ZoomOut(curContentMousePoint);
            }
        }

        private void ZoomControl_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            if ((Keyboard.Modifiers & ModifierKeys.Shift) == 0)
            {
                Point doubleClickPoint = e.GetPosition(content);
                zoomAndPanControl.AnimatedSnapTo(doubleClickPoint);
            }
        }

        private void Rectangle_MouseDown(object sender, MouseButtonEventArgs e)
        {
            content.Focus();
            Keyboard.Focus(content);

            if ((Keyboard.Modifiers & ModifierKeys.Shift) != 0)
            {
                // When the shift key is held down special zooming logic is executed in content_MouseDown,
                // so don't handle mouse input here.
                return;
            }

            if (mouseHandlingMode != MouseHandlingMode.None)
            {
                // We are in some other mouse handling mode, don't do anything.
                return;
            }

            // Get the focussed object
            if (sender is SimulationVisual2D)
            {
                _infoControl.InitInfoObject(sender as SimulationVisual2D);
            }
            if (sender is SimulationVisual3D)
            {
                _infoControl.InitInfoObject(sender as SimulationVisual3D);
            }

            e.Handled = true;
        }

        private void SetStatisticalOutputFolder(object sender, RoutedEventArgs e)
        {
            //open folder browser dialog and set the path
            var folderBrowser = new System.Windows.Forms.FolderBrowserDialog();
            var result = folderBrowser.ShowDialog();
            if (result == System.Windows.Forms.DialogResult.OK)
                TextStatisicsOutputFolder.Text = folderBrowser.SelectedPath;
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Zoom the viewport out, centering on the specified point (in content coordinates).
        /// </summary>
        private void ZoomOut(Point contentZoomCenter)
        {
            zoomAndPanControl.ZoomAboutPoint(zoomAndPanControl.ContentScale - 0.1, contentZoomCenter);
        }

        /// <summary>
        /// Zoom the viewport in, centering on the specified point (in content coordinates).
        /// </summary>
        private void ZoomIn(Point contentZoomCenter)
        {
            zoomAndPanControl.ZoomAboutPoint(zoomAndPanControl.ContentScale + 0.1, contentZoomCenter);
        }

        /// <summary>
        /// Update the position and size of the rectangle that user is dragging out.
        /// </summary>
        private void UpdateSelectionRect(Point pt1, Point pt2)
        {
            double x, y, width, height;

            // Deterine x,y,width and height of the rect inverting the points if necessary.

            if (pt2.X < pt1.X)
            {
                x = pt2.X;
                width = pt1.X - pt2.X;
            }
            else
            {
                x = pt1.X;
                width = pt2.X - pt1.X;
            }

            if (pt2.Y < pt1.Y)
            {
                y = pt2.Y;
                height = pt1.Y - pt2.Y;
            }
            else
            {
                y = pt1.Y;
                height = pt2.Y - pt1.Y;
            }

            // Update the coordinates of the rectangle that is being dragged out by the user.
            // The we offset and rescale to convert from content coordinates.
            System.Windows.Controls.Canvas.SetLeft(dragZoomBorder, x);
            System.Windows.Controls.Canvas.SetTop(dragZoomBorder, y);
            dragZoomBorder.Width = width;
            dragZoomBorder.Height = height;

            // Make the rectangle visible, if not already done
            if (selectionCanvas.Visibility != Visibility.Visible)
            {
                selectionCanvas.Visibility = Visibility.Visible;
                dragZoomBorder.Opacity = 0.5;
            }
        }

        /// <summary>
        /// When the user has finished dragging out the rectangle the zoom operation is applied.
        /// </summary>
        private void ZoomBySelectionRect()
        {
            // Retreive the rectangle that the user draggged out and zoom in on it.
            double contentX = System.Windows.Controls.Canvas.GetLeft(dragZoomBorder);
            double contentY = System.Windows.Controls.Canvas.GetTop(dragZoomBorder);
            double contentWidth = dragZoomBorder.Width;
            double contentHeight = dragZoomBorder.Height;
            zoomAndPanControl.AnimatedZoomTo(new Rect(contentX, contentY, contentWidth, contentHeight));

            FadeOutSelectionRect();
        }

        /// <summary>
        /// When the user has finished dragging out the rectangle for selecting the waypoints to remove the removal is applied.
        /// </summary>
        private void RemoveWaypointsBySelectionRect()
        {
            // Retrieve the rectangle that the user draggged out
            double contentX = System.Windows.Controls.Canvas.GetLeft(dragZoomBorder);
            double contentY = System.Windows.Controls.Canvas.GetTop(dragZoomBorder);
            double contentWidth = dragZoomBorder.Width;
            double contentHeight = dragZoomBorder.Height;

            // If we do not have an instance, just quit here
            if (_instance == null)
            {
                FadeOutSelectionRect();
                return;
            }

            // If the instance is executing, just quit here
            if (_running)
            {
                MessageBox.Show("Cannot modify instance during execution!", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
                FadeOutSelectionRect();
                return;
            }

            // Invalidate instance
            _instanceInvalidated = true;

            // Init a transformer that helps obtaining the corner points of the rectangle in instance coordinates
            Transformation2D lengthTransformer = new Transformation2D(
                _tiers[_focusedTierIndex].GetInfoLength(),
                _tiers[_focusedTierIndex].GetInfoWidth(),
                theGrid.Width,
                theGrid.Height);

            // Obtain removal coordinates
            double selectionXMin = lengthTransformer.RevertX(contentX);
            double selectionXMax = lengthTransformer.RevertX(contentX) + lengthTransformer.RevertXLength(contentWidth);
            // Note: selection rectangle gives us the upper corner point - we have to make the transfer to the lower one here
            double selectionYMin = lengthTransformer.RevertY(contentY) - lengthTransformer.RevertYLength(contentHeight);
            double selectionYMax = lengthTransformer.RevertY(contentY);

            // Remove the waypoints that are contained in the selected area
            _instance.ModRemoveWaypoints(_tiers[_focusedTierIndex], selectionXMin, selectionYMin, selectionXMax, selectionYMax);

            // Update the visual
            _animationControl2D.UpdateCurrentTier(_tiers[_focusedTierIndex]);

            // Small sanity check
            SanityCheckAndShowInfo();

            // Fade out the selection rectangle
            FadeOutSelectionRect();
        }

        /// <summary>
        /// Fade out the drag zoom rectangle.
        /// </summary>
        private void FadeOutSelectionRect()
        {
            AnimationHelper.StartAnimation(dragZoomBorder, Border.OpacityProperty, 0.0, 0.1,
                delegate (object sender, EventArgs e)
                {
                    selectionCanvas.Visibility = Visibility.Collapsed;
                });
        }

        #endregion

        #endregion

        #region Remote features

        private void ButtonRemoteWaypointReached_KeyDown(object sender, KeyEventArgs e) { if (e.Key == Key.Enter) { FireOnWaypointReached(null, null); } }
        private void ButtonRemotePickedUpPod_KeyDown(object sender, KeyEventArgs e) { if (e.Key == Key.Enter) { FireOnPickedUpPodBot(null, null); } }
        private void ButtonRemoteSetDownPod_KeyDown(object sender, KeyEventArgs e) { if (e.Key == Key.Enter) { FireOnSetDownPodBot(null, null); } }
        private void FireOnWaypointReached(object sender, RoutedEventArgs e)
        {
            var bot = _instance.GetBotByID(Int32.Parse(TextBoxEventBot.Text));
            var waypoint = _instance.GetWaypointByID(Int32.Parse(TextBoxEventWaypoint.Text));
            bot.OnReachedWaypoint(waypoint);
        }

        private void FireOnPickedUpPodBot(object sender, RoutedEventArgs e)
        {
            var bot = _instance.GetBotByID(Int32.Parse(TextBoxEventBot.Text));
            bot.OnPickedUpPod();
        }

        private void FireOnSetDownPodBot(object sender, RoutedEventArgs e)
        {
            var bot = _instance.GetBotByID(Int32.Parse(TextBoxEventBot.Text));
            bot.OnSetDownPod();
        }

        private Button _remoteButton;
        private Image _remoteImage;
        private ControlClient _controlClient;
        private void ConnectionErrorHandler(bool connected)
        {
            this.Dispatcher.Invoke(() =>
            {
                if (connected)
                {
                    // Show disconnect button
                    ButtonConnectControl.IsEnabled = false; ButtonConnectControl.Visibility = Visibility.Collapsed; ImageConnectControl.Visibility = Visibility.Collapsed;
                    ButtonDisconnectControl.IsEnabled = true; ButtonDisconnectControl.Visibility = Visibility.Visible; ImageDisconnectControl.Visibility = Visibility.Visible;
                    // Set active remote button and image
                    _remoteButton = ButtonDisconnectControl;
                    _remoteImage = ImageDisconnectControl;
                }
                else
                {
                    // Show connect button
                    ButtonConnectControl.IsEnabled = true; ButtonConnectControl.Visibility = Visibility.Visible; ImageConnectControl.Visibility = Visibility.Visible;
                    ButtonDisconnectControl.IsEnabled = false; ButtonDisconnectControl.Visibility = Visibility.Collapsed; ImageDisconnectControl.Visibility = Visibility.Collapsed;
                    // Set active remote button and image
                    _remoteButton = ButtonConnectControl;
                    _remoteImage = ImageConnectControl;
                    // Deactivate at the instance
                    if (_instance != null)
                        _instance.DeactivateRemoteController();
                }
            });
        }
        private void ButtonConnectControl_Click(object sender, RoutedEventArgs e)
        {
            if (_instance != null)
            {
                // Init both sides and connect all callbacks
                RemoteControlAdapter remoteControlAdapter = _instance.ActivateRemoteController();
                _controlClient = new ControlClient(
                    remoteControlAdapter.RobotLocationUpdateCallback,
                    remoteControlAdapter.RobotPickupFinishedCallback,
                    remoteControlAdapter.RobotSetdownFinishedCallback,
                    LogLine,
                    ConnectionErrorHandler);
                remoteControlAdapter.PathSubmissionCallback = _controlClient.SubmitNewPath;
                remoteControlAdapter.PickupSubmissionCallback = _controlClient.SubmitPickupCommand;
                remoteControlAdapter.SetdownSubmissionCallback = _controlClient.SubmitSetdownCommand;
                remoteControlAdapter.RestSubmissionCallback = _controlClient.SubmitRestCommand;
                remoteControlAdapter.GetItemSubmissionCallback = _controlClient.SubmitGetItemNotification;
                remoteControlAdapter.PutItemSubmissionCallback = _controlClient.SubmitPutItemNotification;
                // Connect with the server
                _controlClient.Connect(TextBoxRemoteIP.Text, TextBoxRemotePort.Text);
                // Show disconnect button
                ButtonConnectControl.IsEnabled = false; ButtonConnectControl.Visibility = Visibility.Collapsed; ImageConnectControl.Visibility = Visibility.Collapsed;
                ButtonDisconnectControl.IsEnabled = true; ButtonDisconnectControl.Visibility = Visibility.Visible; ImageDisconnectControl.Visibility = Visibility.Visible;
                // Set active remote button and image
                _remoteButton = ButtonDisconnectControl;
                _remoteImage = ImageDisconnectControl;
            }
            else
            {
                LogLine("No instance initiated - load the instance first then connect to the server managing communication with the agents on the instance");
            }
        }

        private void ButtonDisconnectControl_Click(object sender, RoutedEventArgs e)
        {
            // Disconnect client
            if (_controlClient != null)
                _controlClient.Disconnect();
            // Show connect button
            ButtonConnectControl.IsEnabled = true; ButtonConnectControl.Visibility = Visibility.Visible; ImageConnectControl.Visibility = Visibility.Visible;
            ButtonDisconnectControl.IsEnabled = false; ButtonDisconnectControl.Visibility = Visibility.Collapsed; ImageDisconnectControl.Visibility = Visibility.Collapsed;
            // Set active remote button and image
            _remoteButton = ButtonConnectControl;
            _remoteImage = ImageConnectControl;
            // Deactivate at the instance
            if (_instance != null)
                _instance.DeactivateRemoteController();
        }

        #endregion

        #region Heatmap rendering

        /// <summary>
        /// The instance for which the heat data shall be visualized.
        /// </summary>
        private string _heatMapInstanceFile = null;
        /// <summary>
        /// The heat data that shall be visualized for the given instance.
        /// </summary>
        private string _heatMapDataFile = null;
        /// <summary>
        /// The index of the sub heat information.
        /// </summary>
        private int _heatMapSubDataIndex = 0;
        /// <summary>
        /// Stores the starting positions of the bots that can be used within heatmap generation.
        /// </summary>
        private List<Tuple<int, double, double>> _heatBotStartingPositions;
        /// <summary>
        /// The renderer used to draw heatmaps.
        /// </summary>
        private HeatMapRenderer _heatmapRenderer = null;

        private void ButtonLoadHeatInstance_Click(object sender, RoutedEventArgs e)
        {
            // Create an instance of the open file dialog box.
            OpenFileDialog instanceDialog = new OpenFileDialog();

            // Set filter options and filter index.
            instanceDialog.Filter = "XINST Files (.xinst)|*.xinst";
            instanceDialog.FilterIndex = 1;
            instanceDialog.Multiselect = false;

            // Call the ShowDialog method to show the dialog box.
            bool? instClick = instanceDialog.ShowDialog();

            // Process input if the user clicked OK.
            if (instClick == true)
            {
                // Save file path (for rendering)
                _heatMapInstanceFile = instanceDialog.FileName;
                TextBlockHeatInstance.Text = "Instance: " + System.IO.Path.GetFileName(_heatMapInstanceFile);
            }
        }

        private void ButtonLoadHeatInfo_Click(object sender, RoutedEventArgs e)
        {
            // Create an instance of the open file dialog box.
            OpenFileDialog heatDataDialog = new OpenFileDialog();

            // Set filter options and filter index.
            heatDataDialog.Filter = "HEAT Files (.heat)|*.heat";
            heatDataDialog.FilterIndex = 1;
            heatDataDialog.Multiselect = false;

            // Call the ShowDialog method to show the dialog box.
            bool? heatClick = heatDataDialog.ShowDialog();

            // Process input if the user clicked OK.
            if (heatClick == true)
            {
                // Extract datatype information
                HeatDataType dataType = HeatMapRenderer.ParseHeatDataType(heatDataDialog.FileName);
                // Get sub data type choices
                string[] choices = HeatMapRenderer.GetSubDataChoices(dataType);
                // Let the user choose
                ComboBoxDialog choiceDialog = new ComboBoxDialog(choices, (int index) => { _heatMapSubDataIndex = index; });
                bool? choiceClick = choiceDialog.ShowDialog();
                // Process input if the user clicked OK
                if (choiceClick == true)
                {
                    // Save file path (for rendering)
                    _heatMapDataFile = heatDataDialog.FileName;
                    TextBlockHeatDataFile.Text = "Data: " + System.IO.Path.GetFileName(_heatMapDataFile) + " (" + choices[_heatMapSubDataIndex] + ")";
                }
            }
        }

        private void ButtonHeatJustDraw_Click(object sender, RoutedEventArgs e)
        {
            if (!string.IsNullOrWhiteSpace(_heatMapInstanceFile) && File.Exists(_heatMapInstanceFile))
            {
                // Read and draw instance
                ReadInstanceForHeatmapRendering();
                // Show the instance
                InitVisuals();
                // Reset the view
                ResetView();
            }
        }

        private void ButtonHeatRedraw_Click(object sender, RoutedEventArgs e)
        {
            if (File.Exists(_heatMapInstanceFile) && File.Exists(_heatMapDataFile))
            {
                // Read instance
                ReadInstanceForHeatmapRendering();
                // Start rendering
                HeatMapRendererConfiguration config = ParseHeatmapConfiguration();
                config.DataFile = _heatMapDataFile;
                config.DataIndex = _heatMapSubDataIndex;
                RenderHeatmap(config, true);
            }
            else
            {
                // Warn the user
                MessageBox.Show("No data files defined - load them first!");
            }
        }

        private void ButtonHeatDrawAll_Click(object sender, RoutedEventArgs e)
        {
            // Init heatmaps to generate
            List<Tuple<string, int>> heatmapFiles = new List<Tuple<string, int>>()
            {
                new Tuple<string, int>(IOConstants.StatFileNames[IOConstants.StatFile.HeatLocationPolling], 0),
                new Tuple<string, int>(IOConstants.StatFileNames[IOConstants.StatFile.HeatTrips], 0),
            };
            // Init parameters
            string instanceDir = null;
            string rootResultDir = null;
            // Select instance resource directory
            using (System.Windows.Forms.FolderBrowserDialog instanceDirDialog = new System.Windows.Forms.FolderBrowserDialog())
            {
                instanceDirDialog.Description = "Select directory containing all necessary instance files";
                if (instanceDirDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    // Save instance dir
                    instanceDir = instanceDirDialog.SelectedPath;
                    // Select result directory
                    using (System.Windows.Forms.FolderBrowserDialog rootResultDirDialog = new System.Windows.Forms.FolderBrowserDialog())
                    {
                        rootResultDirDialog.Description = "Select root directory containing all result directories";
                        if (rootResultDirDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                        {
                            // Save result dir
                            rootResultDir = rootResultDirDialog.SelectedPath;
                        }
                    }
                }
            }
            // Disable buttons
            DisableButtons();
            // Check preliminaries
            if (instanceDir != null && rootResultDir != null)
            {
                // Asynchronously execute
                ThreadPool.QueueUserWorkItem((object unused) =>
                {
                    // --> Generate all heatmaps
                    _heatMapInstanceFile = null;
                    string[] resultDirectories = Directory.EnumerateDirectories(rootResultDir).ToArray();
                    for (int i = 0; i < resultDirectories.Length; i++)
                    {
                        // Fetch instance name
                        string instanceName = null;
                        string path = System.IO.Path.Combine(resultDirectories[i], IOConstants.StatFileNames[IOConstants.StatFile.InstanceName]);
                        if (!File.Exists(path))
                            continue;
                        using (StreamReader sr = new StreamReader(path))
                            instanceName = sr.ReadToEnd().Trim();
                        // Log
                        LogLine("Generating heatmap(s) for instance " + instanceName + " - directory " + i + " / " + resultDirectories.Length);
                        // See whether we have to load a new instance for the current result
                        if (System.IO.Path.GetFileNameWithoutExtension(_heatMapInstanceFile) != instanceName)
                        {
                            _heatMapInstanceFile = System.IO.Path.Combine(instanceDir, instanceName + ".xinst");
                            Dispatcher.Invoke(() =>
                            {
                                ReadInstanceForHeatmapRendering();
                            });
                        }
                        // Generate the heatmaps
                        foreach (var heatFile in heatmapFiles)
                        {
                            // Parse the config to use for rendering
                            HeatMapRendererConfiguration config = null;
                            Dispatcher.Invoke(() =>
                            {
                                config = ParseHeatmapConfiguration();
                            });
                            config.FinishedCallback = null;
                            // Render heatmap
                            config.DataFile = System.IO.Path.Combine(resultDirectories[i], heatFile.Item1);
                            config.DataIndex = heatFile.Item2;
                            Dispatcher.Invoke(() =>
                            {
                                // Show the instance
                                InitVisuals();
                                // Reset the view
                                ResetView();
                                // Init the renderer
                                _heatmapRenderer = new HeatMapRenderer(config);
                            });
                            // Render the heatmap
                            _heatmapRenderer.RenderSync();
                            // Wait a short time for the GUI to update
                            Thread.Sleep(5000);
                            // Save screenshot
                            if (_viewMode == ViewMode.View2D && _animationControl2D != null)
                            {
                                Dispatcher.Invoke(() =>
                                {
                                    _animationControl2D.TakeSnapshot(resultDirectories[i], System.IO.Path.GetFileNameWithoutExtension(heatFile.Item1));
                                });
                            }
                            else
                            {
                                MessageBox.Show("Heatmaps can only be generated in 2D view!", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                                return;
                            }
                        }
                    }
                    // Mark finished
                    HeatRenderingCallback();
                });
            }
        }

        private void ButtonHeatDrawTimeWindows_Click(object sender, RoutedEventArgs e)
        {
            if (File.Exists(_heatMapInstanceFile) && File.Exists(_heatMapDataFile))
            {
                // Read instance
                ReadInstanceForHeatmapRendering();
                // Prepare config
                HeatMapRendererConfiguration config = ParseHeatmapConfiguration();
                config.DataFile = _heatMapDataFile;
                config.DataIndex = _heatMapSubDataIndex;
                config.FinishedCallback = null;

                double stepLength;
                try { stepLength = double.Parse(TextBoxHeatTimeWindowLength.Text, IOConstants.FORMATTER); }
                catch (FormatException) { MessageBox.Show("Cannot parse time-window length!", "Error", MessageBoxButton.OK, MessageBoxImage.Error); return; }
                stepLength = Math.Max(SimulationObserver.STEP_LENGTH_POSITION_POLL * 5, stepLength);
                // --> Start rendering
                // Disable buttons
                DisableButtons();
                // Asynchronously execute
                ThreadPool.QueueUserWorkItem((object unused) =>
                {
                    // --> Generate all heatmaps
                    bool success; double currentFilterLow = -stepLength; double currentFilterHigh = 0;
                    do
                    {
                        // Log
                        LogLine("Generating heatmap for time-window: " + currentFilterLow.ToString(IOConstants.FORMATTER) + " -> " + currentFilterHigh.ToString(IOConstants.FORMATTER));
                        // Set time window
                        config.InitialBotPositions = currentFilterLow < 0 ? _heatBotStartingPositions : null;
                        config.DataTimeFilterLow = currentFilterLow;
                        config.DataTimeFilterHigh = currentFilterHigh;
                        // Render heatmap
                        Dispatcher.Invoke(() =>
                        {
                            // Show the instance
                            InitVisuals();
                            // Reset the view
                            ResetView();
                            // Init the renderer
                            _heatmapRenderer = new HeatMapRenderer(config);
                        });
                        // Render the heatmap
                        success = _heatmapRenderer.RenderSync();
                        // Wait a short time for the GUI to update
                        Thread.Sleep(5000);
                        // Save screenshot
                        if (success)
                        {
                            if (_viewMode == ViewMode.View2D && _animationControl2D != null)
                            {
                                Dispatcher.Invoke(() =>
                                {
                                    if (!Directory.Exists(_snapshotDir))
                                        Directory.CreateDirectory(_snapshotDir);

                                    int day = TimeSpan.FromSeconds(currentFilterLow).Days;
                                    int hour = TimeSpan.FromSeconds(currentFilterLow).Hours;
                                    int minute = TimeSpan.FromSeconds(currentFilterLow).Minutes;
                                    _animationControl2D.TakeSnapshot(_snapshotDir, _instance.Name +
                                        "-D-" + day.ToString(IOConstants.FORMATTER) +
                                        "-H-" + hour.ToString(IOConstants.FORMATTER) +
                                        "-M-" + minute.ToString(IOConstants.FORMATTER) +
                                        "-W-" + (Math.Floor(TimeSpan.FromSeconds(currentFilterHigh - currentFilterLow).TotalMinutes).ToString(IOConstants.EXPORT_FORMAT_SHORTEST_BY_ROUNDING, IOConstants.FORMATTER)));
                                });
                            }
                            else
                            {
                                MessageBox.Show("Heatmaps can only be generated in 2D view!", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                                return;
                            }
                        }
                        // Update time window
                        currentFilterLow += stepLength; currentFilterHigh += stepLength;
                    } while (success);
                    // Mark finished
                    HeatRenderingCallback();
                });
            }
            else
            {
                // Warn the user
                MessageBox.Show("No data files defined - load them first!");
            }
        }

        private void SliderHeatOpacity_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            // Check whether a heatmap currently exists
            if (_heatmapRenderer != null && _heatmapRenderer.ResultImage != null)
            {
                // Set the new opacity value
                _heatmapRenderer.ResultImage.Opacity = SliderHeatOpacity.Value;
            }
        }

        private void CheckBoxHeatTileLengthCountSwitch_Checked(object sender, RoutedEventArgs e)
        {
            // Check whether initialized
            if (IsInitialized)
            {
                // Flip between tile length and count parameter input
                if (CheckBoxHeatTileLengthCountSwitch.IsChecked == true)
                {
                    TextBoxHeatTilesX.Visibility = Visibility.Visible;
                    TextBoxHeatTileLengthX.Visibility = Visibility.Collapsed;
                    CheckBoxHeatTileLengthCountSwitch.Content = "#Tiles (x):";
                }
                else
                {
                    TextBoxHeatTilesX.Visibility = Visibility.Collapsed;
                    TextBoxHeatTileLengthX.Visibility = Visibility.Visible;
                    CheckBoxHeatTileLengthCountSwitch.Content = "TileLength (x):";
                }
            }
        }

        private void CheckBoxHeatTransparentCanvas_Checked(object sender, RoutedEventArgs e)
        {
            if (CheckBoxHeatTransparentCanvas.IsChecked == true) { content.Background = Brushes.Transparent; }
            else { content.Background = Brushes.White; }
        }

        /// <summary>
        /// Initializes the dynamic UI-elements for the heatmap rendering section.
        /// </summary>
        private void InitHeatmapUIElements()
        {
            _heatmapBotTaskFilterCheckboxes = Enum.GetValues(typeof(BotTaskType)).Cast<BotTaskType>().ToDictionary(
                // Bot task type is the key
                k => k,
                // Create a new checkbox for the corresponding task
                v =>
                {
                    CheckBox cb = new CheckBox() { Content = v.ToString(), IsChecked = true };
                    StackPanelHeatmapBotTasks.Children.Add(cb);
                    return cb;
                });
        }
        /// <summary>
        /// Contains the checkboxes used for filtering position heat-data by the corresponding bot task.
        /// </summary>
        private Dictionary<BotTaskType, CheckBox> _heatmapBotTaskFilterCheckboxes;
        /// <summary>
        /// Parses the current heatmap setting from the GUI and returns a new configuration according to these.
        /// </summary>
        /// <returns>The new config.</returns>
        private HeatMapRendererConfiguration ParseHeatmapConfiguration()
        {
            // Init 
            HeatMapRendererConfiguration config = null;
            // Parse the parameters
            bool drawInBackground = CheckBoxHeatDrawInBackground.IsChecked == true;
            bool logarithmic = CheckBoxHeatLogarithmic.IsChecked == true;
            bool weightByDistance = CheckBoxHeatWeightByDistance.IsChecked == true;
            bool manhattanDistance = CheckBoxHeatManhattan.IsChecked == true;
            bool bichromaticColoring = CheckBoxHeatBichromaticColoring.IsChecked == true;
            bool useTileCount = CheckBoxHeatTileLengthCountSwitch.IsChecked == true;
            double radiusRelativeToTileLengthX; int tilesX; double tileLengthX; Color firstColor; Color secondColor;
            HashSet<BotTaskType> botTaskFilter = _heatmapBotTaskFilterCheckboxes.Where(kvp => kvp.Value.IsChecked == true).Select(kvp => kvp.Key).ToHashSet();
            try
            {
                // Parse
                radiusRelativeToTileLengthX = double.Parse(TextBoxHeatRadius.Text, IOConstants.FORMATTER);
                tilesX = int.Parse(TextBoxHeatTilesX.Text);
                tileLengthX = double.Parse(TextBoxHeatTileLengthX.Text, IOConstants.FORMATTER);
                firstColor = (Color)ColorConverter.ConvertFromString(ComboBoxHeatmapColoringFirstColor.SelectedValue.ToString());
                secondColor = (Color)ColorConverter.ConvertFromString(ComboBoxHeatmapColoringSecondColor.SelectedValue.ToString());
                // Create it
                config = new HeatMapRendererConfiguration()
                {
                    ContentControl = content, // The canvas to draw on
                    ContentHost = theGrid, // The host of the canvas (for size information)
                    FinishedCallback = HeatRenderingCallback, // The callback
                    Logger = LogLine, // The logger - this can take a while
                    BotTaskFilter = botTaskFilter, // Only consider data for the given tasks
                    Tier = _instance.GetInfoTiers().First(), // Draw first tier
                    UseTileCount = useTileCount, // Indicates whether to use the count or length info for the tile size
                    TilesX = tilesX, // The number of tiles used in x-Dimension
                    TileLengthX = tileLengthX, // The length of a tile in x-dimension
                    DrawInBackground = drawInBackground, // Indicates whether the heatmap will be drawn behind everything else or in front of it
                    Logarithmic = logarithmic, // Indicates that the values are transformed by applying the logarithm
                    WeightByDistance = weightByDistance, // Specifies whether to weight the value of datapoints by their distance to the tile center
                    ManhattanDistance = manhattanDistance, // Specifies whether to use the manhattan distance metric
                    RadiusRelativeToTileLength = radiusRelativeToTileLengthX, // The radius relative to the tile-length in x-dimension
                    BichromaticColoring = bichromaticColoring, // Specifies whether to use bichromatic coloring
                    BichromaticColorOne = firstColor, // The first color for bichromatic coloring
                    BichromaticColorTwo = secondColor, // The second color for bichromatic coloring
                };
            }
            catch (Exception) { MessageBox.Show("Cannot parse the parameters"); return null; }
            return config;
        }

        private void ReadInstanceForHeatmapRendering()
        {
            // Parse the instance
            _instance = InstanceIO.ReadInstance(_heatMapInstanceFile, null, null, true, true);
            // Invalidate the instance (the instance should not execute in this state)
            _instanceInvalidated = true;
            // Store bot starting positions before removing the bots
            _heatBotStartingPositions = _instance.GetInfoBots().Select(b => new Tuple<int, double, double>(b.GetInfoCurrentTier().GetInfoID(), b.GetInfoCenterX(), b.GetInfoCenterY())).ToList();
            // Remove unecessary objects in case of heatmap visualization
            if (!(CheckBoxHeatDrawBots.IsChecked == true))
                _instance.VisClearBots();
            if (!(CheckBoxHeatDrawPods.IsChecked == true))
                _instance.VisClearPods();
            if (!(CheckBoxHeatDrawStations.IsChecked == true))
                _instance.VisClearStations();
            if (!(CheckBoxHeatDrawSemaphores.IsChecked == true))
                _instance.VisClearSemaphores();
        }

        private void HeatRenderingCallback()
        {
            this.Dispatcher.Invoke(() =>
            {
                // Set meta-info for feedback
                TextBlockHeatDrawingInfoTileLength.Text = "Tile-size: " +
                    _heatmapRenderer.TileLengthX.ToString(IOConstants.EXPORT_FORMAT_SHORT, IOConstants.FORMATTER) + " x " +
                    _heatmapRenderer.TileLengthY.ToString(IOConstants.EXPORT_FORMAT_SHORT, IOConstants.FORMATTER);
                TextBlockHeatDrawingInfoRadius.Text = "Radius: " + _heatmapRenderer.Radius.ToString(IOConstants.EXPORT_FORMAT_SHORT, IOConstants.FORMATTER);
                // Set opacity just in case it is not the default anymore
                if (_heatmapRenderer.ResultImage != null)
                    _heatmapRenderer.ResultImage.Opacity = SliderHeatOpacity.Value;
            });
            // Enable the buttons again
            EnableButtons();
        }

        private void RenderHeatmap(HeatMapRendererConfiguration config, bool asynchronously)
        {

            // Disable the buttons
            DisableButtons();
            // Show the instance
            InitVisuals();
            // Reset the view
            ResetView();
            // Init the renderer
            _heatmapRenderer = new HeatMapRenderer(config);
            // Render the heatmap
            if (asynchronously)
                _heatmapRenderer.RenderAsync();
            else
                _heatmapRenderer.RenderSync();
        }

        #endregion

        #region Flip book debugging

        private void ButtonLoadFlipBookDebuggingMaterial_Click(object sender, RoutedEventArgs e)
        {
            // Create an instance of the open file dialog box.
            OpenFileDialog instanceDialog = new OpenFileDialog();

            // Set filter options and filter index.
            instanceDialog.Filter = "XINST Files (.xinst)|*.xinst";
            instanceDialog.FilterIndex = 1;
            instanceDialog.Multiselect = false;

            // Call the ShowDialog method to show the dialog box.
            bool? instClick = instanceDialog.ShowDialog();

            // Process input if the user clicked OK.
            if (instClick == true)
            {
                // Create an instance of the open file dialog box.
                OpenFileDialog locationDataDialog = new OpenFileDialog();

                // Set filter options and filter index.
                locationDataDialog.Filter = "LP HEAT Files (locationspolled.heat)|locationspolled.heat";
                locationDataDialog.FilterIndex = 1;
                locationDataDialog.Multiselect = false;

                // Call the ShowDialog method to show the dialog box.
                bool? locationDataClick = locationDataDialog.ShowDialog();

                // Process input if the user clicked OK.
                if (locationDataClick == true)
                {
                    // Parse the instance
                    _instance = InstanceIO.ReadInstance(instanceDialog.FileName, null, null, true, true);
                    // TODO what about invalidation?
                    _instanceInvalidated = false;
                    // Remove unecessary objects in case of heatmap visualization
                    _instance.VisClearBots();
                    _instance.VisClearPods();
                    _instance.VisClearSemaphores();

                    // TODO go on
                    throw new NotImplementedException();
                }
            }
        }

        #endregion

        #region Onboard camera

        /// <summary>
        /// The last robot for which the onboard camera was activated.
        /// </summary>
        private SimulationVisualBot3D _onboardCameraBot;

        private void Button3DOnboardCamera_Click(object sender, RoutedEventArgs e)
        {
            if (_onboardCameraBot == null)
            {
                // Choose a new bot and attach the camera to it
                SimulationVisualBot3D[] botVisuals = Viewport3D.Children.OfType<SimulationVisualBot3D>().ToArray();
                if (botVisuals.Any())
                {
                    Random random = new Random();
                    _onboardCameraBot = botVisuals[random.Next(botVisuals.Length)];
                    _onboardCameraBot.StartOnboardCamera(Viewport3D.Camera);
                }
            }
            else
            {
                // Deactivate onboard camera
                _onboardCameraBot.StopOnboardCamera();
                _onboardCameraBot = null;
            }
        }

        #endregion
    }
}
