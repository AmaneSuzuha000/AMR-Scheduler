// StreamPlayer.cs -- plays a .stream.json exported by the Python scheduling core.
//
// Why WinForms and not WPF
// -------------------------
// RAWSim-O's interface is WPF, but this machine has no MSBuild and its csproj does not
// build here. Rather than "tweak the WPF" with no way to verify it, the useful step
// was to get the **data path** working in something that compiles. The player reads the
// same .stream.json, so switching to WPF only replaces the rendering layer.
//
// What it does
// ----------
// - draws the network (nodes coloured by kind, capacity labelled)
// - play / pause / scrub frame by frame, showing vehicle position and state
// - shows task counts and the number of blocked vehicles
// - highlights preemption events
// - labels the pose source prominently (SIM / ODOM / LOCALIZED); **never mixed**

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using System.Web.Script.Serialization;

namespace AmrSched.Player
{
    // ---- Data model, kept in step with AmrSched.Interop but duplicated to avoid a dependency ----
    public class PNode
    {
        public string Id;
        public string Kind;
        public int Capacity = 1;
        public double X, Y;
    }

    public class PEdge { public string U, V; }

    public class PAgv
    {
        public string Id;
        public string Node;
        public string Edge;
        public string State = "idle";
        public string TaskId;
        public string Source = "sim";
        public System.Collections.Generic.List<string> Path = new System.Collections.Generic.List<string>();
    }

    public class PFrame
    {
        public int Step;
        public List<PAgv> Agvs = new List<PAgv>();
        public Dictionary<string, int> Counts = new Dictionary<string, int>();
        public int NBlocked;
        public List<string> PreemptText = new List<string>();
    }

    public class StreamData
    {
        /// <summary>Parses one vehicle's remaining path. Returns an empty list rather than throwing when it is missing or malformed.</summary>
        public static System.Collections.Generic.List<string> ParsePath(
            System.Collections.Generic.Dictionary<string, object> ad)
        {
            var list = new System.Collections.Generic.List<string>();
            if (!ad.ContainsKey("path")) return list;
            var arr = ad["path"] as object[];
            if (arr == null) return list;
            foreach (var o in arr)
                if (o != null) list.Add(Convert.ToString(o));
            return list;
        }

        public string Schema;
        public List<PNode> Nodes = new List<PNode>();
        public List<PEdge> Edges = new List<PEdge>();
        public List<PFrame> Frames = new List<PFrame>();
        public Dictionary<string, object> Meta = new Dictionary<string, object>();

        public static StreamData Load(string path)
        {
            var js = new JavaScriptSerializer();
            js.MaxJsonLength = 512 * 1024 * 1024;
            var root = (Dictionary<string, object>)js.DeserializeObject(File.ReadAllText(path));
            var d = new StreamData();
            d.Schema = root.ContainsKey("schema") ? (string)root["schema"] : null;
            if (d.Schema != "1.0")
                throw new InvalidDataException("schema mismatch: " + (d.Schema ?? "<missing>") + ", expected 1.0");

            if (root.ContainsKey("meta"))
                d.Meta = (Dictionary<string, object>)root["meta"];

            var net = root.ContainsKey("network") ? root["network"] as Dictionary<string, object> : null;
            if (net != null)
            {
                foreach (var o in (object[])net["nodes"])
                {
                    var m = (Dictionary<string, object>)o;
                    d.Nodes.Add(new PNode
                    {
                        Id = (string)m["id"],
                        Kind = m.ContainsKey("kind") ? (string)m["kind"] : "aisle",
                        Capacity = m.ContainsKey("capacity") ? Convert.ToInt32(m["capacity"]) : 1,
                        X = m.ContainsKey("x") && m["x"] != null ? Convert.ToDouble(m["x"]) : 0,
                        Y = m.ContainsKey("y") && m["y"] != null ? Convert.ToDouble(m["y"]) : 0,
                    });
                }
                foreach (var o in (object[])net["edges"])
                {
                    var m = (Dictionary<string, object>)o;
                    d.Edges.Add(new PEdge { U = (string)m["u"], V = (string)m["v"] });
                }
            }

            foreach (var fo in (object[])root["frames"])
            {
                var fd = (Dictionary<string, object>)fo;
                var fr = new PFrame();
                fr.Step = Convert.ToInt32(fd["step"]);
                fr.NBlocked = fd.ContainsKey("n_blocked") ? Convert.ToInt32(fd["n_blocked"]) : 0;
                foreach (var ao in (object[])fd["agvs"])
                {
                    var ad = (Dictionary<string, object>)ao;
                    var pd = (Dictionary<string, object>)ad["pose"];
                    fr.Agvs.Add(new PAgv
                    {
                        Id = (string)ad["agv_id"],
                        Node = pd.ContainsKey("node") && pd["node"] != null ? (string)pd["node"] : null,
                        Edge = pd.ContainsKey("edge") && pd["edge"] != null ? (string)pd["edge"] : null,
                        State = ad.ContainsKey("state") ? (string)ad["state"] : "idle",
                        TaskId = ad.ContainsKey("task_id") && ad["task_id"] != null ? (string)ad["task_id"] : null,
                        Source = pd.ContainsKey("source") ? (string)pd["source"] : "sim",
                        Path = ParsePath(ad),
                    });
                }
                var tc = fd.ContainsKey("task_counts") ? fd["task_counts"] as Dictionary<string, object> : null;
                if (tc != null) foreach (var kv in tc) fr.Counts[kv.Key] = Convert.ToInt32(kv.Value);
                var pre = fd.ContainsKey("preemptions") ? fd["preemptions"] as object[] : null;
                if (pre != null)
                    foreach (var po in pre)
                    {
                        var pm = (Dictionary<string, object>)po;
                        fr.PreemptText.Add(pm["urgent"] + " ← " + pm["victim_agv"]);
                    }
                d.Frames.Add(fr);
            }
            return d;
        }
    }
    // ---- Playback window ----
    public class PlayerForm : Form
    {
        private readonly StreamData _d;
        private int _idx;
        private Timer _timer;
        private Panel _canvas;
        private TrackBar _bar;
        private Button _play;
        private Label _hud;
        private ComboBox _speed;
        private readonly Dictionary<string, PointF> _pos = new Dictionary<string, PointF>();
        private readonly Dictionary<string, PNode> _nodeById = new Dictionary<string, PNode>();
        private float _minX, _maxX, _minY, _maxY;

        public PlayerForm(StreamData d)
        {
            _d = d;
            Text = "AMR schedule playback -- " + Path.GetFileName(_d.Meta.ContainsKey("scenario")
                ? Convert.ToString(_d.Meta["scenario"]) : "stream")
                + "  [" + d.Frames.Count + " frames, schema " + d.Schema + "]";
            Width = 1100; Height = 760;
            BackColor = Color.FromArgb(24, 26, 30);
            ForeColor = Color.Gainsboro;

            foreach (var n in _d.Nodes) _nodeById[n.Id] = n;
            BuildUi();
            ComputeLayout();
        }

        /// <summary>Maps node coordinates onto the canvas rectangle.</summary>
        private void ComputeLayout()
        {
            if (_d.Nodes.Count == 0) return;
            _minX = (float)_d.Nodes.Min(n => n.X); _maxX = (float)_d.Nodes.Max(n => n.X);
            _minY = (float)_d.Nodes.Min(n => n.Y); _maxY = (float)_d.Nodes.Max(n => n.Y);
            float w = Math.Max(1e-6f, _maxX - _minX), h = Math.Max(1e-6f, _maxY - _minY);
            const float pad = 60f;
            float sx = (_canvasW() - 2 * pad) / w, sy = (_canvasH() - 2 * pad) / h;
            float s = Math.Min(sx, sy);
            float ox = pad + ((_canvasW() - 2 * pad) - w * s) / 2f;
            float oy = pad + ((_canvasH() - 2 * pad) - h * s) / 2f;
            foreach (var n in _d.Nodes)
                _pos[n.Id] = new PointF(ox + (float)(n.X - _minX) * s,
                                        oy + (float)(n.Y - _minY) * s);
        }

        private int _canvasW() { return Math.Max(200, Width - 40); }
        private int _canvasH() { return Math.Max(200, Height - 140); }

        private Color KindColor(string kind)
        {
            switch (kind)
            {
                case "workstation": return Color.FromArgb(120, 200, 255);
                case "buffer": return Color.FromArgb(255, 200, 120);
                case "disassembly": return Color.FromArgb(255, 140, 140);
                case "assembly": return Color.FromArgb(160, 255, 160);
                default: return Color.FromArgb(150, 150, 160);
            }
        }

        private static readonly Color[] AgvColors = {
            Color.FromArgb(255, 99, 99), Color.FromArgb(99, 200, 255),
            Color.FromArgb(150, 255, 99), Color.FromArgb(255, 220, 99),
            Color.FromArgb(220, 150, 255), Color.FromArgb(99, 255, 190),
        };

        protected override void OnResize(EventArgs e)
        {
            base.OnResize(e);
            if (_bar != null) {
                _bar.Top = Height - 96; _bar.Width = Width - 240;
                _hud.Left = Width - 230; _hud.Top = Height - 96;
            }
            ComputeLayout();
            if (_canvas != null) _canvas.Invalidate();
        }

        private void BuildUi()
        {
            _canvas = new Panel { Dock = DockStyle.Fill, BackColor = Color.FromArgb(24, 26, 30) };
            _canvas.Paint += CanvasPaint;
            _canvas.Resize += (s, e) => { ComputeLayout(); _canvas.Invalidate(); };
            Controls.Add(_canvas);

            var bar = new Panel { Dock = DockStyle.Bottom, Height = 96,
                                   BackColor = Color.FromArgb(32, 34, 40) };
            Controls.Add(bar);

            _play = new Button { Text = "Play", Left = 12, Top = 12, Width = 90, Height = 32,
                                 BackColor = Color.FromArgb(60, 62, 72),
                                 ForeColor = Color.White, FlatStyle = FlatStyle.Flat };
            _play.Click += (s, e) => Toggle();
            bar.Controls.Add(_play);

            var reset = new Button { Text = "⏮", Left = 110, Top = 12, Width = 44, Height = 32,
                                     BackColor = Color.FromArgb(60, 62, 72),
                                     ForeColor = Color.White, FlatStyle = FlatStyle.Flat };
            reset.Click += (s, e) => { _idx = 0; _bar.Value = 0; _canvas.Invalidate(); Sync(); };
            bar.Controls.Add(reset);

            _speed = new ComboBox { Left = 164, Top = 14, Width = 76, DropDownStyle = ComboBoxStyle.DropDownList };
            foreach (var v in new[] { "1×", "2×", "5×", "10×", "30×" }) _speed.Items.Add(v);
            _speed.SelectedIndex = 1;
            _speed.SelectedIndexChanged += (s, e) => ApplySpeed();
            bar.Controls.Add(_speed);

            _bar = new TrackBar { Left = 250, Top = 8, Width = 600,
                                  Minimum = 0, Maximum = Math.Max(0, _d.Frames.Count - 1),
                                  TickStyle = TickStyle.None };
            _bar.Scroll += (s, e) => { _idx = _bar.Value; _canvas.Invalidate(); Sync(); };
            bar.Controls.Add(_bar);

            _hud = new Label { Left = 860, Top = 12, Width = 220, Height = 70,
                               ForeColor = Color.Gainsboro, Font = new Font("Consolas", 9f) };
            bar.Controls.Add(_hud);

            _timer = new Timer();
            _timer.Interval = 50;
            _timer.Tick += (s, e) =>
            {
                if (_idx < _d.Frames.Count - 1) { _idx++; _bar.Value = Math.Min(_idx, _bar.Maximum); }
                else { _timer.Stop(); _play.Text = "Play"; }
                _canvas.Invalidate();
                Sync();
            };
            ApplySpeed();
            Sync();
        }

        private void ApplySpeed()
        {
            int mul = _speed.SelectedIndex <= 0 ? 1 :
                      _speed.SelectedIndex == 1 ? 2 :
                      _speed.SelectedIndex == 2 ? 5 :
                      _speed.SelectedIndex == 3 ? 10 : 30;
            _timer.Interval = Math.Max(16, 100 / mul);
        }

        private void Toggle()
        {
            if (_d.Frames.Count == 0) return;
            if (_timer.Enabled) { _timer.Stop(); _play.Text = "Play"; }
            else { if (_idx >= _d.Frames.Count - 1) _idx = 0; _timer.Start(); _play.Text = "Pause"; }
        }

        private void Sync()
        {
            if (_d.Frames.Count == 0) { _hud.Text = "no frames"; return; }
            var f = _d.Frames[Math.Min(_idx, _d.Frames.Count - 1)];
            string cnt = string.Join("  ", f.Counts.Select(kv => kv.Key + "=" + kv.Value));
            var srcs = f.Agvs.Select(a => a.Source).Distinct().ToArray();
            _hud.Text = "step " + f.Step + " / " + _d.Frames.Count
                + "\r\n" + cnt
                + "\r\nblocked " + f.NBlocked
                + "\r\npose source: " + string.Join(",", srcs);
        }

        private void CanvasPaint(object sender, PaintEventArgs e)
        {
            var g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.Clear(Color.FromArgb(24, 26, 30));

            // Edges
            using (var pen = new Pen(Color.FromArgb(64, 68, 78), 2f))
                foreach (var ed in _d.Edges)
                    if (_pos.ContainsKey(ed.U) && _pos.ContainsKey(ed.V))
                        g.DrawLine(pen, _pos[ed.U], _pos[ed.V]);

            // Nodes
            foreach (var n in _d.Nodes)
            {
                if (!_pos.ContainsKey(n.Id)) continue;
                var p = _pos[n.Id];
                float r = 12f + n.Capacity * 2f;
                using (var br = new SolidBrush(Color.FromArgb(40, KindColor(n.Kind))))
                    g.FillEllipse(br, p.X - r, p.Y - r, r * 2, r * 2);
                using (var pen = new Pen(KindColor(n.Kind), 2f))
                    g.DrawEllipse(pen, p.X - r, p.Y - r, r * 2, r * 2);
                using (var br = new SolidBrush(Color.Gainsboro))
                using (var fnt = new Font("Consolas", 8.5f))
                {
                    var sz = g.MeasureString(n.Id, fnt);
                    g.DrawString(n.Id, fnt, br, p.X - sz.Width / 2, p.Y - r - sz.Height - 5);
                    if (n.Capacity > 1)
                    {
                        using (var f2 = new Font("Consolas", 7f))
                        using (var b2 = new SolidBrush(Color.FromArgb(160, 255, 200, 120)))
                            g.DrawString("×" + n.Capacity, f2, b2, p.X - 8, p.Y + r + 1);
                    }
                }
            }

            if (_d.Frames.Count == 0) return;
            var frame = _d.Frames[Math.Min(_idx, _d.Frames.Count - 1)];

            // Planned path: drawn before the vehicles so it cannot cover them or their labels.
            //
            // Only the remaining leg is drawn, from the current position through each path
            // node to the goal. Drawing the whole route would bury the picture in history and
            using (var pathPen = new Pen(Color.FromArgb(90, 130, 200, 255), 2.2f))
            {
                pathPen.DashStyle = DashStyle.Dash;
                foreach (var a in frame.Agvs)
                {
                    if (a.Path == null || a.Path.Count == 0) continue;
                    var pts = new List<PointF>();
                    // Start: where the vehicle is now
                    if (a.Node != null && _pos.ContainsKey(a.Node)) pts.Add(_pos[a.Node]);
                    else if (a.Edge != null)
                    {
                        var pr = a.Edge.Split(new[] { "->" }, StringSplitOptions.None);
                        if (pr.Length == 2 && _pos.ContainsKey(pr[0]) && _pos.ContainsKey(pr[1]))
                        {
                            var p0 = _pos[pr[0]]; var p1 = _pos[pr[1]];
                            pts.Add(new PointF(p0.X + 0.5f * (p1.X - p0.X),
                                               p0.Y + 0.5f * (p1.Y - p0.Y)));
                        }
                    }
                    if (pts.Count == 0) continue;
                    foreach (var node in a.Path)
                        if (_pos.ContainsKey(node)) pts.Add(_pos[node]);
                    if (pts.Count >= 2)
                        g.DrawLines(pathPen, pts.ToArray());
                }
            }

            // Vehicles
            int ci = 0;
            foreach (var a in frame.Agvs)
            {
                PointF p;
                if (a.Node != null && _pos.ContainsKey(a.Node)) p = _pos[a.Node];
                else if (a.Edge != null)
                {
                    var pr = a.Edge.Split(new[] { "->" }, StringSplitOptions.None);
                    if (pr.Length == 2 && _pos.ContainsKey(pr[0]) && _pos.ContainsKey(pr[1]))
                    {
                        var p0 = _pos[pr[0]]; var p1 = _pos[pr[1]];
                        p = new PointF(p0.X + 0.5f * (p1.X - p0.X), p0.Y + 0.5f * (p1.Y - p0.Y));
                    }
                    else continue;
                }
                else continue;

                var col = AgvColors[ci % AgvColors.Length];
                ci++;
                // An extra ring when blocked
                if (a.State == "blocked")
                    using (var pen = new Pen(Color.OrangeRed, 2f))
                        g.DrawEllipse(pen, p.X - 13, p.Y - 13, 26, 26);
                using (var br = new SolidBrush(col))
                    g.FillEllipse(br, p.X - 8, p.Y - 8, 16, 16);
                using (var fnt = new Font("Consolas", 7.5f))
                using (var br = new SolidBrush(col))
                {
                    var lbl = a.Id.Replace("AGV", "A") + " " + a.State;
                    g.DrawString(lbl, fnt, br, p.X + 11, p.Y + 9);
                }
            }

            // Preemption events
            if (frame.PreemptText.Count > 0)
            {
                using (var fnt = new Font("Consolas", 12f, FontStyle.Bold))
                using (var br = new SolidBrush(Color.FromArgb(255, 120, 60)))
                    g.DrawString("preempted: " + string.Join(" | ", frame.PreemptText),
                                 fnt, br, 20, 16);
            }

            // Source watermark: the pose source must always stay visible
            var srcSet = frame.Agvs.Select(x => x.Source).Distinct().ToArray();
            string srcText = "pose source: " + string.Join(" + ", srcSet);
            var srcColor = srcSet.Length > 1 ? Color.OrangeRed : Color.FromArgb(140, 160, 200);
            using (var fnt = new Font("Consolas", 9f))
            using (var br = new SolidBrush(srcColor))
            {
                var sz = g.MeasureString(srcText, fnt);
                g.DrawString(srcText, fnt, br, _canvas.Width - sz.Width - 16, 12);
            }
        }
    }

    public static class PlayerMain
    {
        [STAThread]
        public static int Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.Error.WriteLine("usage: player <stream.json>");
                return 2;
            }
            try
            {
                var d = StreamData.Load(args[0]);
                Application.EnableVisualStyles();
                Application.Run(new PlayerForm(d));
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine("ERROR: " + ex.GetType().Name + ": " + ex.Message);
                return 1;
            }
        }
    }
}