using System;
using System.IO;
using AmrSched.Interop;

public static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length < 1) { Console.Error.WriteLine("usage: reader <stream.json>"); return 2; }
        StreamDocument doc;
        try
        {
            doc = StreamDocument.Load(args[0]);
        }
        catch (Exception ex)
        {
            // Catch this here on purpose: when stdout is redirected, an unhandled .NET
            // exception can fail even in Exception.ToString(), losing the message.
            Console.Error.WriteLine("ERROR: " + ex.GetType().Name + ": " + ex.Message);
            return 1;
        }
        Console.WriteLine("schema=" + doc.Schema);
        Console.WriteLine("frames=" + doc.Frames.Count);
        if (doc.Frames.Count > 0)
        {
            var f = doc.Frames[0];
            Console.WriteLine("first_step=" + f.Step + " agvs=" + f.Agvs.Count);
            foreach (var a in f.Agvs)
                Console.WriteLine("  " + a.AgvId + " node=" + (a.Pose.Node ?? "-") +
                                  " edge=" + (a.Pose.Edge ?? "-") + " state=" + a.State +
                                  " source=" + a.Pose.Source);
        }
        var srcs = doc.DistinctSources();
        Console.WriteLine("sources=" + string.Join(",", srcs));
        foreach (var s in srcs)
            if (s == PoseSource.Unknown) { Console.Error.WriteLine("unknown pose source present"); return 1; }
        return 0;
    }
}