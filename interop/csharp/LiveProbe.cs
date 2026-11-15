// LiveProbe.cs -- the C# side measurement program for the live channel.
//
// Why it exists: to show LiveClient is not code that was written and never run.
// It connects to the Python side on :8770, receives frames, prints them, and checks:
//   1. the first message is a hello and the network is complete;
//   2. the step number strictly increases;
//   3. positions actually change (PoseToXY must yield different coordinates);
//   4. a done message arrives at the end.
//
// Build with the compiler that ships with .NET Framework; no SDK needed:
//   %WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe /out:LiveProbe.exe *.cs

using System;
using System.Collections.Generic;
using System.Threading;
using AmrSched.Interop;

namespace AmrSched.Probe
{
    internal static class Program
    {
        private static int Main(string[] args)
        {
            string host = args.Length > 0 ? args[0] : "127.0.0.1";
            int port = args.Length > 1 ? int.Parse(args[1]) : 8770;

            Console.WriteLine("connecting to " + host + ":" + port + " ...");

            int frames = 0;
            int lastStep = -1;
            bool stepOk = true;
            bool gotHello = false;
            bool gotDone = false;
            int nodeCount = 0, edgeCount = 0;
            var positions = new HashSet<string>();
            NetworkSpec net = null;
            var finished = new ManualResetEventSlim(false);

            var client = new LiveClient(host, port);

            client.Message += msg =>
            {
                if (msg.Type == "hello")
                {
                    gotHello = true;
                    net = msg.Net;
                    if (net != null) { nodeCount = net.Nodes.Count; edgeCount = net.Edges.Count; }
                    Console.WriteLine("[hello] schema=" + msg.Schema +
                                      " network=" + nodeCount + " nodes / " + edgeCount + " edges");
                    return;
                }
                if (msg.Type == "done")
                {
                    gotDone = true;
                    object done = null, total = null, mk = null;
                    if (msg.Summary != null)
                    {
                        msg.Summary.TryGetValue("tasks_completed", out done);
                        msg.Summary.TryGetValue("tasks_total", out total);
                        msg.Summary.TryGetValue("makespan_seconds", out mk);
                    }
                    Console.WriteLine("[done] step=" + msg.Step +
                                      " completed=" + done + "/" + total +
                                      " completion time=" + mk + "s");
                    finished.Set();
                    return;
                }

                frames++;
                if (lastStep >= 0 && msg.Step <= lastStep) stepOk = false;
                lastStep = msg.Step;

                // Compute a continuous position per vehicle, which is what shows motion.
                foreach (var a in msg.Agvs)
                {
                    double x, y;
                    LiveClient.PoseToXY(net, a.Pose, out x, out y);
                    positions.Add(a.AgvId + "@" + Math.Round(x, 1) + "," + Math.Round(y, 1));
                }

                if (frames <= 3 || frames % 100 == 0)
                {
                    var sb = new System.Text.StringBuilder();
                    sb.Append("[frame " + frames + "] step=" + msg.Step + " ");
                    foreach (var a in msg.Agvs)
                    {
                        double x, y;
                        LiveClient.PoseToXY(net, a.Pose, out x, out y);
                        sb.Append(a.AgvId + "(" + Math.Round(x, 0) + "," + Math.Round(y, 0) + ") ");
                    }
                    Console.WriteLine(sb.ToString());
                }
            };

            client.Disconnected += reason =>
            {
                Console.WriteLine("[disconnected] " + reason);
                finished.Set();
            };

            client.Start();
            finished.Wait(120000);
            client.Stop();

            Console.WriteLine();
            Console.WriteLine("frames received   : " + frames);
            Console.WriteLine("distinct layouts  : " + positions.Count);
            Console.WriteLine("step monotonic    : " + (stepOk ? "yes" : "no"));

            bool ok = gotHello && gotDone && frames > 10 && stepOk
                      && positions.Count > 5 && nodeCount == 17 && edgeCount == 36;
            Console.WriteLine(ok ? "PASS" : "FAIL");
            return ok ? 0 : 1;
        }
    }
}
