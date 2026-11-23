// StreamReader.cs -- reference implementation for reading the amr_sched.stream format.
//
// Why it exists
// ----------
// Before this, the Python scheduling core and the RAWSim-O C# interface had no data
// contract: two programs, each doing its own thing. This class shows the exchange
// format **can actually be consumed by C#**, not merely be a convention in a document.
//
// Depends on System.Web.Script.Serialization, shipped with .NET Framework 4.0,
// and pulls in no third-party JSON library, so it still builds on a bare machine.

using System;
using System.Collections.Generic;
using System.IO;
using System.Web.Script.Serialization;

namespace AmrSched.Interop
{
    /// <summary>Pose source. The three kinds must not be mixed.</summary>
    public enum PoseSource { Sim, Odom, Localized, Unknown }

    public class Pose
    {
        public string AgvId;
        public string Node;      // non-null when parked on a node
        public string Edge;      // non-null in transit, shaped like "R0->R1"
        public double Progress;
        public PoseSource Source;
        public double? Timestamp;

        public static PoseSource ParseSource(string s)
        {
            if (s == "sim") return PoseSource.Sim;
            if (s == "odom") return PoseSource.Odom;
            if (s == "localized") return PoseSource.Localized;
            return PoseSource.Unknown;
        }
    }

    public class AgvFrame
    {
        public string AgvId;
        public Pose Pose;
        public string State;
        public string TaskId;
    }

    public class Preemption
    {
        public string Urgent;
        public string VictimAgv;
        public string VictimTask;
        public bool WasPicked;
    }

    public class Frame
    {
        public int Step;
        public double T;
        public List<AgvFrame> Agvs = new List<AgvFrame>();
        public Dictionary<string, int> TaskCounts = new Dictionary<string, int>();
        public int NBlocked;
        public List<Preemption> Preemptions = new List<Preemption>();
    }

    public class StreamDocument
    {
        public string Schema;
        public Dictionary<string, object> Meta = new Dictionary<string, object>();
        public List<Frame> Frames = new List<Frame>();

        /// <summary>Format version. Must equal SCHEMA_VERSION on the Python side.</summary>
        public const string ExpectedSchema = "1.0";

        public static StreamDocument Load(string path)
        {
            string text = File.ReadAllText(path);
            var js = new JavaScriptSerializer();
            js.MaxJsonLength = 512 * 1024 * 1024;
            var root = (Dictionary<string, object>)js.DeserializeObject(text);

            var doc = new StreamDocument();
            doc.Schema = root.ContainsKey("schema") ? (string)root["schema"] : null;
            if (doc.Schema != ExpectedSchema)
                throw new InvalidDataException(
                    "schema mismatch: file says " + (doc.Schema ?? "<missing>") +
                    ", expected " + ExpectedSchema);

            var metaObj = root.ContainsKey("meta") ? root["meta"] as Dictionary<string, object> : null;
            if (metaObj != null)
                doc.Meta = metaObj;

            if (!root.ContainsKey("frames"))
                throw new InvalidDataException("missing the frames field");

            foreach (var fo in (object[])root["frames"])
            {
                var fd = (Dictionary<string, object>)fo;
                var fr = new Frame();
                fr.Step = Convert.ToInt32(fd["step"]);
                fr.T = fd.ContainsKey("t") ? Convert.ToDouble(fd["t"]) : 0.0;
                fr.NBlocked = fd.ContainsKey("n_blocked") ? Convert.ToInt32(fd["n_blocked"]) : 0;

                foreach (var ao in (object[])fd["agvs"])
                {
                    var ad = (Dictionary<string, object>)ao;
                    var pd = (Dictionary<string, object>)ad["pose"];
                    var p = new Pose();
                    p.AgvId = (string)pd["agv_id"];
                    p.Node = pd.ContainsKey("node") && pd["node"] != null ? (string)pd["node"] : null;
                    p.Edge = pd.ContainsKey("edge") && pd["edge"] != null ? (string)pd["edge"] : null;
                    p.Progress = pd.ContainsKey("progress") ? Convert.ToDouble(pd["progress"]) : 0.0;
                    p.Source = Pose.ParseSource(pd.ContainsKey("source") ? (string)pd["source"] : "sim");
                    p.Timestamp = pd.ContainsKey("timestamp") && pd["timestamp"] != null
                        ? (double?)Convert.ToDouble(pd["timestamp"]) : null;
                    fr.Agvs.Add(new AgvFrame
                    {
                        AgvId = (string)ad["agv_id"],
                        Pose = p,
                        State = ad.ContainsKey("state") ? (string)ad["state"] : "idle",
                        TaskId = ad.ContainsKey("task_id") && ad["task_id"] != null
                            ? (string)ad["task_id"] : null,
                    });
                }

                var tc = fd.ContainsKey("task_counts") ? fd["task_counts"] as Dictionary<string, object> : null;
                if (tc != null)
                    foreach (var kv in tc)
                        fr.TaskCounts[kv.Key] = Convert.ToInt32(kv.Value);

                doc.Frames.Add(fr);
            }
            return doc;
        }

        /// <summary>Checks that every pose source agrees. Mixing them invalidates any consistency claim.</summary>
        public HashSet<PoseSource> DistinctSources()
        {
            var set = new HashSet<PoseSource>();
            foreach (var f in Frames)
                foreach (var a in f.Agvs)
                    set.Add(a.Pose.Source);
            return set;
        }
    }
}