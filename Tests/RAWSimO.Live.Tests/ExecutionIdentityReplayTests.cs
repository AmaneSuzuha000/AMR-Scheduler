using System.Text.Json.Nodes;
using RAWSimO.Live.Protocol;
using RAWSimO.Live.Tracking;
using Xunit;

namespace RAWSimO.Live.Tests;

public class ExecutionIdentityReplayTests
{
    private static JsonObject Control(string scene = "车间", string run = "r") => new()
    {
        ["session_id"] = "s", ["scene_id"] = scene, ["run_id"] = run,
        ["paused"] = false, ["operations"] = new JsonArray(),
        ["routes"] = new JsonObject { ["R1"] = run + ":R1:1", ["R2"] = run + ":R2:1" }
    };

    private static JsonObject Hello(bool control) => new()
    {
        ["type"] = "hello", ["network"] = new JsonObject
        {
            ["nodes"] = new JsonArray(), ["edges"] = new JsonArray()
        },
        ["meta"] = new JsonObject(), ["control"] = control ? Control() : null
    };

    private static JsonObject Execution(string robot, string scene, string run) => new()
    {
        ["robot_id"] = robot, ["scene_id"] = scene, ["run_id"] = run,
        ["fault"] = robot == "R1" ? "execution guard: invalid command" : null
    };

    private static JsonObject Frame(int t, bool single = false, string scene = "车间", string run = "r")
    {
        var agvs = new JsonArray();
        foreach (var robot in new[] { "R1", "R2" })
            agvs.Add(new JsonObject
            {
                ["agv_id"] = robot, ["state"] = "blocked",
                ["pose"] = new JsonObject
                {
                    ["agv_id"] = robot, ["source"] = "localized", ["available"] = false
                }
            });
        var frame = new JsonObject
        {
            ["type"] = "frame", ["schema"] = "1.0", ["step"] = t, ["t"] = t, ["agvs"] = agvs
        };
        if (single) frame["execution"] = Execution("R1", scene, run);
        else frame["executions"] = new JsonObject
        {
            ["R1"] = Execution("R1", scene, run), ["R2"] = Execution("R2", scene, run)
        };
        return frame;
    }

    private static string Recording(params JsonObject[] events) =>
        string.Join("\n", events.Select(value => value.ToJsonString()));

    // Exercise both public readers; rejection must also release an indexed file.
    private static void Read(bool indexed, string content, Action<FleetReplay> inspect)
    {
        if (!indexed)
        {
            using var replay = FleetReplay.Load(new StringReader(content));
            inspect(replay);
            return;
        }
        var path = Path.Combine(AppContext.BaseDirectory, "identity-" + Guid.NewGuid().ToString("N") + ".jsonl");
        try
        {
            File.WriteAllText(path, content);
            using (var replay = FleetReplay.Open(path)) inspect(replay);
        }
        finally
        {
            using (var exclusive = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None)) { }
            File.Delete(path);
        }
    }

    [Theory]
    [InlineData(false, false, false)]
    [InlineData(false, false, true)]
    [InlineData(false, true, false)]
    [InlineData(false, true, true)]
    [InlineData(true, false, false)]
    [InlineData(true, false, true)]
    [InlineData(true, true, false)]
    [InlineData(true, true, true)]
    public void MissingControlCannotHideMixedExecutionRunsOrScenes(bool indexed, bool helloControl, bool single)
    {
        foreach (var field in new[] { "scene_id", "run_id" })
        {
            var changed = Frame(1, single);
            if (single) changed["execution"]![field] = "another";
            else foreach (var state in changed["executions"]!.AsObject()) state.Value![field] = "another";
            var content = Recording(Hello(helloControl), Frame(0, single), changed);
            var error = Assert.Throws<FormatException>(() => Read(indexed, content, _ => { }));
            Assert.Contains("scene/run", error.Message);
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void SameExecutionRunSurvivesControlClosureAndPreservesFinalFault(bool indexed, bool single)
    {
        var first = Frame(0, single);
        first["control"] = Control();
        var content = Recording(Hello(true), first, Frame(1, single));
        Read(indexed, content, replay =>
        {
            Assert.Equal(2, replay.Frames.Count);
            Assert.Null(replay.At(1).Frame.Control);
            Assert.Equal("execution guard: invalid command", replay.At(1).Frame.ExecutionFaults!["R1"]);
            Assert.Equal(new FleetExecutionIdentity("车间", "r"), replay.At(1).Frame.ExecutionIdentity);
        });
    }

    [Theory]
    [InlineData("scene_id")]
    [InlineData("run_id")]
    public void OneFrameCannotCombineExecutionsFromDifferentRuns(string field)
    {
        var frame = Frame(0);
        frame["executions"]!["R2"]![field] = "another";
        Assert.False(LiveMessageParser.TryParse(frame.ToJsonString(), out _, out _));
    }

    [Theory]
    [InlineData("scene_id")]
    [InlineData("run_id")]
    public void ExecutionAndControlInOneFrameMustAgree(string field)
    {
        var frame = Frame(0);
        frame["control"] = Control();
        frame["control"]![field] = "another";
        Assert.False(LiveMessageParser.TryParse(frame.ToJsonString(), out _, out _));
    }

    [Theory]
    [InlineData("scene_id", "missing")]
    [InlineData("run_id", "missing")]
    [InlineData("scene_id", "null")]
    [InlineData("run_id", "null")]
    [InlineData("scene_id", "empty")]
    [InlineData("run_id", "empty")]
    [InlineData("scene_id", "number")]
    [InlineData("run_id", "number")]
    public void ExecutionRequiresExplicitNonemptyRunAndScene(string field, string problem)
    {
        var frame = Frame(0, true);
        var state = frame["execution"]!.AsObject();
        if (problem == "missing") state.Remove(field);
        else state[field] = problem switch { "null" => null, "empty" => JsonValue.Create(" "), _ => JsonValue.Create(7) };
        Assert.False(LiveMessageParser.TryParse(frame.ToJsonString(), out _, out _));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LaterControlCannotReplaceIdentityEstablishedByExecution(bool indexed)
    {
        var changed = Frame(1);
        changed.Remove("executions");
        changed["control"] = Control(run: "another");
        var content = Recording(Hello(false), Frame(0), changed);
        Assert.Throws<FormatException>(() => Read(indexed, content, _ => { }));
    }
}
