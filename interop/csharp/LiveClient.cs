// LiveClient.cs -- subscribes to the Python scheduler's vehicle state in real time.
//
// How this differs from StreamReader.cs
// -----------------------
// StreamReader reads a **finished JSON file**: offline and one-shot.
// This class connects to a **TCP port** and reads one JSON object per line while the run
// is still going: online and continuous. That is the layer "the marker on the map
//
// depends on: only System.Net.Sockets and System.Web.Script.Serialization, both
// shipped with .NET Framework. No third-party dependency.
//
// Usage:
//   var client = new LiveClient("127.0.0.1", 8770);
//   client.Message += msg => {
//       if (msg.Type == "frame") { /* update the map */ }
//   };
//   client.Start();   // receives on a background thread; does not block the UI

using System;
using AmrInterop;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;

namespace AmrSched.Interop
{
    /// <summary>One message received from the live channel.</summary>
    public class LiveMessage
    {
        public string Type;                       // hello / frame / done
        public string Schema;
        public int Step;
        public double T;
        public List<AgvFrame> Agvs = new List<AgvFrame>();
        public Dictionary<string, int> TaskCounts = new Dictionary<string, int>();
        public int NBlocked;
        public NetworkSpec Net;                   // present only on hello
        public Dictionary<string, object> Summary; // present only on done
    }

    public class LiveNode
    {
        public string Id;
        public string Kind;
        public int Capacity;
        public double X, Y;
    }

    public class LiveEdge
    {
        public string Src, Dst;
        public double Length;
        public int Capacity;
        public double[][] Centerline;
    }

    public class NetworkSpec
    {
        public List<LiveNode> Nodes = new List<LiveNode>();
        public List<LiveEdge> Edges = new List<LiveEdge>();
    }

    /// <summary>
    /// Live client. Raises <see cref="Message"/> on a **background thread**, so the UI
    /// side must Invoke back to its own thread.
    /// </summary>
    public class LiveClient : IDisposable
    {
        private readonly string _host;
        private readonly int _port;
        private TcpClient _tcp;
        private Thread _worker;
        private volatile bool _running;
        private readonly JavaScriptSerializer _js = new JavaScriptSerializer();

        /// <summary>Raised once per received message (background thread).</summary>
        public event Action<LiveMessage> Message;
        /// <summary>Raised when the connection drops (background thread).</summary>
        public event Action<string> Disconnected;

        /// <summary>Messages received so far; usable as a heartbeat indicator.</summary>
        public int ReceivedCount { get; private set; }
        /// <summary>Step number of the most recent frame.</summary>
        public int LastStep { get; private set; }

        public LiveClient(string host, int port)
        {
            _host = host;
            _port = port;
            _js.MaxJsonLength = 64 * 1024 * 1024;
        }

        public void Start()
        {
            if (_running) return;
            _running = true;
            _worker = new Thread(Loop);
            _worker.IsBackground = true;   // do not keep the process alive when the window closes
            _worker.Start();
        }

        public void Stop()
        {
            _running = false;
            try { if (_tcp != null) _tcp.Close(); } catch { }
            _tcp = null;
        }

        public void Dispose() { Stop(); }

        private void Loop()
        {
            string reason = "unknown reason";
            try
            {
                _tcp = new TcpClient();
                _tcp.NoDelay = true;      // frames are tiny; disabling Nagle cuts latency
                _tcp.Connect(_host, _port);
                using (var stream = _tcp.GetStream())
                using (var reader = new StreamReader(stream, Encoding.UTF8))
                {
                    while (_running)
                    {
                        string line = reader.ReadLine();
                        if (line == null) { reason = "the server closed the connection"; break; }
                        if (line.Length == 0) continue;
                        LiveMessage msg;
                        try { msg = Parse(line); }
                        catch (Exception ex) { reason = "parse failed: " + ex.Message; break; }
                        ReceivedCount++;
                        if (msg.Type == "frame") LastStep = msg.Step;
                        var h = Message;
                        if (h != null) h(msg);
                        if (msg.Type == "done") { reason = "the run finished"; break; }
                    }
                }
            }
            catch (Exception ex) { reason = ex.Message; }
            finally
            {
                _running = false;
                var h = Disconnected;
                if (h != null) h(reason);
            }
        }

        // ---- Safe access. JavaScriptSerializer maps a JSON null to "the key exists but
        // its value is null", and the key may be missing entirely. A bare d["x"] throws
        private static object G(Dictionary<string, object> d, string k)
        {
            object v;
            return d.TryGetValue(k, out v) ? v : null;
        }

        private static string GS(Dictionary<string, object> d, string k)
        {
            object v = G(d, k);
            return v == null ? null : Convert.ToString(v);
        }

        private static int GI(Dictionary<string, object> d, string k, int def = 0)
        {
            object v = G(d, k);
            return v == null ? def : Convert.ToInt32(v);
        }

        private static double GD(Dictionary<string, object> d, string k, double def = 0.0)
        {
            object v = G(d, k);
            return v == null ? def : Convert.ToDouble(v);
        }

        /// <summary>Parses one JSON line into a <see cref="LiveMessage"/>.</summary>
        public LiveMessage Parse(string line)
        {
            var root = (Dictionary<string, object>)_js.DeserializeObject(line);
            var m = new LiveMessage();
            m.Type = GS(root, "type");
            m.Schema = GS(root, "schema");

            if (m.Type == "hello")
            {
                var net = G(root, "network") as Dictionary<string, object>;
                if (net != null)
                {
                    m.Net = new NetworkSpec();
                    if (G(net, "nodes") != null)
                        foreach (var o in (object[])G(net, "nodes"))
                        {
                            var d = (Dictionary<string, object>)o;
                            m.Net.Nodes.Add(new LiveNode
                            {
                                Id = GS(d, "id"),
                                Kind = GS(d, "kind"),
                                Capacity = GI(d, "capacity", 1),
                                X = GD(d, "x"),
                                Y = GD(d, "y"),
                            });
                        }
                    if (G(net, "edges") != null)
                        foreach (var o in (object[])G(net, "edges"))
                        {
                            var d = (Dictionary<string, object>)o;
                            m.Net.Edges.Add(new LiveEdge
                            {
                                Src = GS(d, "src"),
                                Dst = GS(d, "dst"),
                                Length = GD(d, "length", 1.0),
                                Capacity = GI(d, "capacity", 1),
                                Centerline = ParseCenterline(d),
                            });
                        }
                }
                return m;
            }

            if (m.Type == "done")
            {
                m.Step = GI(root, "step");
                m.Summary = G(root, "summary") as Dictionary<string, object>;
                return m;
            }

            // frame
            m.Step = GI(root, "step");
            m.T = GD(root, "t");
            m.NBlocked = GI(root, "n_blocked");

            var tc = G(root, "task_counts") as Dictionary<string, object>;
            if (tc != null)
                foreach (var kv in tc)
                    m.TaskCounts[kv.Key] = Convert.ToInt32(kv.Value);

            foreach (var o in (object[])G(root, "agvs"))
            {
                var d = (Dictionary<string, object>)o;
                var pd = G(d, "pose") as Dictionary<string, object>;
                if (pd == null) continue;
                var af = new AgvFrame();
                af.AgvId = GS(d, "agv_id");
                af.State = GS(d, "state");
                af.TaskId = GS(d, "task_id");
                af.Pose = new Pose();
                af.Pose.AgvId = af.AgvId;
                af.Pose.Node = GS(pd, "node");
                af.Pose.Edge = GS(pd, "edge");
                af.Pose.Progress = GD(pd, "progress");
                af.Pose.Source = Pose.ParseSource(GS(pd, "source") ?? "sim");
                object ts = G(pd, "timestamp");
                af.Pose.Timestamp = ts == null ? (double?)null : Convert.ToDouble(ts);
                m.Agvs.Add(af);
            }
            return m;
        }

        /// <summary>
        /// Computes a vehicle's continuous position on the canvas.
        ///
        /// On a node it takes the node itself; on an edge it interpolates by progress.
        /// This is the drawing entry point that makes the marker follow the vehicle.
        /// </summary>
        public static void PoseToXY(NetworkSpec net, Pose pose,
                                    out double x, out double y)
        {
            x = y = 0;
            if (net == null || pose == null) return;
            if (!string.IsNullOrEmpty(pose.Node))
            {
                foreach (var n in net.Nodes)
                    if (n.Id == pose.Node) { x = n.X; y = n.Y; return; }
                return;
            }
            if (!string.IsNullOrEmpty(pose.Edge))
            {
                string src, dst;
                if (!PoseMath.TrySplitEdge(pose.Edge, out src, out dst)) return;
                LiveNode a = null, b = null;
                foreach (var n in net.Nodes)
                {
                    if (n.Id == src) a = n;
                    if (n.Id == dst) b = n;
                }
                if (a == null || b == null) return;
                // The interpolation and the clamp live in PoseMath.cs, which RAWSimO.Live
                // also compiles. Do not re-derive them here: that is how the two copies
                // drifted apart in the first place.
                LiveEdge road = net.Edges.Find(e => e.Src == src && e.Dst == dst);
                if (road != null && road.Centerline != null)
                    PoseMath.InterpolatePolyline(road.Centerline, pose.Progress, out x, out y);
                else
                    PoseMath.Interpolate(a.X, a.Y, b.X, b.Y, pose.Progress, out x, out y);
            }
        }

        private static double[][] ParseCenterline(Dictionary<string, object> row)
        {
            if (!row.ContainsKey("centerline")) return null;
            object[] values = row["centerline"] as object[];
            if (values == null || values.Length < 2)
                throw new FormatException("Road centerline needs coordinate pairs.");
            double[][] points = new double[values.Length][];
            for (int i = 0; i < values.Length; i++)
            {
                object[] pair = values[i] as object[];
                if (pair == null || pair.Length != 2)
                    throw new FormatException("Road centerline needs coordinate pairs.");
                points[i] = new double[] { Convert.ToDouble(pair[0]), Convert.ToDouble(pair[1]) };
            }
            double x, y;
            PoseMath.InterpolatePolyline(points, 0.0, out x, out y);
            return points;
        }
    }
}
