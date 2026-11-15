using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Xunit;

namespace RAWSimO.Live.Tests;

/// <summary>
/// The display layer must not contain a second decision core. Route planning lives in
/// Python (amr_sched, and tools/live_bridge/route_service.py for the console's own
/// requests); the console asks over the wire and draws what it gets back.
///
/// This test exists because the console *did* plan locally: MainWindow.Live.cs built a
/// waypoint graph from the loaded instance and ran AStarPlanner on it whenever the operator
/// picked a target. That is exactly the split the architecture forbids, and it is invisible
/// in normal use - the drawn route looked right, so nothing failed. A guard is cheaper than
/// remembering.
/// </summary>
public class ConsoleDoesNotPlanTests
{
    private static string RepoRoot()
    {
        var dir = new DirectoryInfo(System.AppContext.BaseDirectory);
        while (dir != null && !Directory.Exists(Path.Combine(dir.FullName, "RAWSimO.Visualization")))
            dir = dir.Parent;
        Assert.NotNull(dir);
        return dir!.FullName;
    }

    [Fact]
    public void VisualizationSourcesDoNotReferenceTheLocalPlanner()
    {
        var root = Path.Combine(RepoRoot(), "RAWSimO.Visualization");
        var offenders = Directory.EnumerateFiles(root, "*.cs", SearchOption.AllDirectories)
            .Where(p => !p.Contains(Path.DirectorySeparatorChar + "obj" + Path.DirectorySeparatorChar))
            .Where(p => !p.Contains(Path.DirectorySeparatorChar + "bin" + Path.DirectorySeparatorChar))
            .Where(p => Regex.IsMatch(File.ReadAllText(p), @"\bAStarPlanner\b|\bMotionParameters\b"))
            .Select(Path.GetFileName)
            .ToList();
        Assert.True(offenders.Count == 0,
            "显示层不得自带规划器，路线必须向 Python 请求。仍在引用本地规划器的文件: " +
            string.Join(", ", offenders));
    }

    [Fact]
    public void VisualizationSourcesAskTheBridgeForRoutes()
    {
        // The positive half of the guard: the console must actually contain the request.
        var live = Path.Combine(RepoRoot(), "RAWSimO.Visualization", "MainWindow.Live.cs");
        Assert.True(File.Exists(live), "找不到 MainWindow.Live.cs");
        var text = File.ReadAllText(live);
        Assert.Contains("RouteRequestMessage", text);
        Assert.Contains("ApplyLiveRoute", text);
    }
}
