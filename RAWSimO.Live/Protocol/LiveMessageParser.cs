using System.Text.Json;
using System.Text.Json.Serialization;
using RAWSimO.Live.Geometry;

namespace RAWSimO.Live.Protocol;

/// <summary>
/// Parses and serialises the newline-delimited JSON protocol used by
/// <c>tools/live_bridge</c>. Parsing is deliberately permissive about extra
/// fields and strict about values that would corrupt the pose pipeline.
/// </summary>
public static class LiveMessageParser
{
    /// <summary>Maximum ordinary live message and initial hello size in UTF-8 bytes.</summary>
    public const int MaxLineBytes = 64 * 1024;

    /// <summary>Maximum observation size after a valid fleet hello, in UTF-8 bytes.</summary>
    public const int MaxFleetLineBytes = 4 * 1024 * 1024;

    internal const int MaxRecordedLineBytes = MaxFleetLineBytes;

    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DictionaryKeyPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = false,
    };

    /// <summary>Serialises a message to one JSON line (without the trailing newline).</summary>
    public static string Serialize(LiveMessage message)
    {
        ArgumentNullException.ThrowIfNull(message);
        return JsonSerializer.Serialize(message, message.GetType(), SerializerOptions);
    }

    /// <summary>Serialises a message and appends the protocol's newline terminator.</summary>
    public static string SerializeLine(LiveMessage message) => Serialize(message) + "\n";

    /// <summary>Parses one JSON line. Throws <see cref="FormatException"/> when the message is invalid.</summary>
    public static LiveMessage Parse(string line)
    {
        if (!TryParse(line, out var message, out var error))
            throw new FormatException(error);
        return message!;
    }

    /// <summary>
    /// Tries to parse one JSON line. A syntactically valid message with an
    /// unknown type succeeds and produces <see cref="UnknownMessage"/>; malformed
    /// JSON, a missing type, or an invalid required field fails.
    /// </summary>
    public static bool TryParse(string? line, out LiveMessage? message, out string? error)
        => TryParseCore(line, MaxLineBytes, false, false, out message, out error);

    internal static bool TryParseFleetStream(string line, int ordinaryBytes,
        out LiveMessage? message, out string? error)
        => TryParseCore(line, MaxFleetLineBytes, false, true, out message, out error, ordinaryBytes);

    internal static LiveMessage ParseRecording(string line)
    {
        if (!TryParseCore(line, MaxRecordedLineBytes, true, false, out var message, out var error))
            throw new FormatException(error);
        return message!;
    }

    private static bool TryParseCore(string? line, int maximumBytes, bool recording, bool fleetStream,
        out LiveMessage? message, out string? error, int ordinaryBytes = MaxLineBytes)
    {
        message = null;
        error = null;
        if (string.IsNullOrWhiteSpace(line))
        {
            error = "empty message";
            return false;
        }

        var byteCount = System.Text.Encoding.UTF8.GetByteCount(line);
        if (byteCount > maximumBytes)
        {
            error = recording ? "recorded message exceeds the 4 MiB line limit"
                : fleetStream ? "fleet observation exceeds the 4 MiB line limit"
                : "message exceeds the 64 KiB line limit";
            return false;
        }

        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(line);
        }
        catch (JsonException ex)
        {
            error = "invalid JSON: " + ex.Message;
            return false;
        }

        using (document)
        {
            var root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
            {
                error = "message must be a JSON object";
                return false;
            }

            if (!root.TryGetProperty("type", out var typeElement) ||
                typeElement.ValueKind != JsonValueKind.String)
            {
                error = "message must contain a string type";
                return false;
            }

            var type = typeElement.GetString();
            if (string.IsNullOrWhiteSpace(type))
            {
                error = "message type must not be empty";
                return false;
            }

            // Only fleet observations use the larger budget, after the connection
            // has established its fleet identity. Commands, receipts and unknown
            // message types keep the ordinary contract even on that connection.
            if (fleetStream && byteCount > ordinaryBytes &&
                !((type == "frame" && root.TryGetProperty("agvs", out _)) ||
                  (type == "hello" && root.TryGetProperty("network", out _))))
            {
                error = ordinaryBytes == MaxLineBytes ? "message exceeds the 64 KiB line limit"
                    : $"message exceeds the configured {ordinaryBytes} UTF-8 byte line limit";
                return false;
            }

            try
            {
                switch (type)
                {
                    // A "hello" carrying a "network" object comes from the fleet
                    // scheduler (amr_sched.live); one without it comes from the
                    // single-vehicle bridge. The guarded case must come first: a
                    // later unconditional "hello" would make it unreachable.
                    case "hello" when root.TryGetProperty("network", out _):
                        message = new FleetHelloMessage(
                            OptionalString(root, "schema") ?? "1.0",
                            OptionalString(root, "source") ?? "sim",
                            ParseFleetNodes(root),
                            ParseFleetEdges(root),
                            ParseFleetMeta(root)) { Control = ParseFleetControl(root) };
                        return true;

                    case "hello":
                        message = new HelloMessage(
                            RequiredString(root, "vehicle"),
                            RequiredInt32(root, "proto"),
                            RequiredString(root, "frame"),
                            OptionalStringArray(root, "caps"),
                            OptionalString(root, "scene"));
                        return true;

                    case "pose":
                        message = new PoseMessage(
                            RequiredString(root, "vehicle"),
                            RequiredInt64(root, "seq"),
                            RequiredFiniteDouble(root, "t"),
                            RequiredFiniteDouble(root, "x"),
                            RequiredFiniteDouble(root, "y"),
                            RequiredFiniteDouble(root, "yaw"));
                        return true;

                    case "status":
                        message = new StatusMessage(
                            OptionalString(root, "vehicle"),
                            OptionalFiniteDouble(root, "battery"),
                            OptionalString(root, "mode"),
                            OptionalString(root, "error"));
                        return true;

                    case "nav_result":
                        message = new NavResultMessage(
                            OptionalString(root, "goal"),
                            RequiredString(root, "status"),
                            OptionalString(root, "reason"));
                        return true;

                    case "nav":
                        message = new NavCommandMessage(
                            RequiredString(root, "goal"),
                            RequiredPath(root),
                            OptionalFiniteDouble(root, "max_speed"));
                        return true;

                    case "cancel":
                        message = new CancelCommandMessage(RequiredString(root, "goal"));
                        return true;

                    // The console asks for a route instead of planning one. A result with no
                    // nodes and a reason is a valid answer: the user needs to see *why* a
                    // target was refused, so an empty route must not be a parse error.
                    case "route_request":
                        message = new RouteRequestMessage(
                            RequiredString(root, "target"),
                            OptionalDoubleArray(root, "from"),
                            OptionalBool(root, "direct") ?? false);
                        return true;

                    case "route_result":
                        message = new RouteResultMessage(
                            OptionalString(root, "target"),
                            OptionalStringArray(root, "nodes") ?? Array.Empty<string>(),
                            OptionalPath(root, "map_path") ?? Array.Empty<Point2D>(),
                            OptionalString(root, "start"),
                            OptionalFiniteDouble(root, "meters"),
                            OptionalFiniteDouble(root, "seconds"),
                            OptionalString(root, "reason"),
                            OptionalInt32(root, "waypoint"));
                        return true;

                    case "ping":
                        message = new PingMessage(RequiredFiniteDouble(root, "t1"));
                        return true;

                    case "pong":
                        message = new PongMessage(
                            RequiredFiniteDouble(root, "t1"),
                            RequiredFiniteDouble(root, "t2"),
                            RequiredFiniteDouble(root, "t3"));
                        return true;

                    case "estop":
                        message = new EstopMessage();
                        return true;

                    case "estop_release":
                        message = new EstopReleaseMessage();
                        return true;

                    // --- Fleet scheduling stream (amr_sched.live) ---------------
                    // Read-only channel. A frame that is missing its agv list is a
                    // producer bug, not something to paper over: fail loudly so the
                    // GUI can show a real error instead of an empty map.

                    case "frame":
                        // Parse the vehicle list before the scalar header fields so a
                        // frame without vehicles reports that, which is the failure a
                        // user actually needs to see.
                        var fleetAgvs = ParseFleetAgvs(root);
                        var fleetControl = ParseFleetControl(root);
                        var execution = ParseExecutions(root, fleetAgvs, fleetControl);
                        message = new FleetFrameMessage(
                            RequiredString(root, "schema"),
                            RequiredInt64(root, "step"),
                            RequiredFiniteDouble(root, "t"),
                            fleetAgvs,
                            ParseTaskCounts(root),
                            OptionalInt32(root, "n_blocked") ?? 0,
                            ParsePreemptions(root),
                            ParseHolds(root),
                            ParseCongestion(root),
                            ParseTrails(root),
                            OptionalString(root, "trail_source"),
                            ParseTrailSources(root),
                            ParsePhysical(root)) { Control = fleetControl, PhysicalHolds = ParsePhysicalHolds(root),
                                ExecutionFaults = execution.Faults, ExecutionIdentity = execution.Identity };
                        return true;

                    case "control_result":
                        message = new FleetControlResultMessage(
                            OptionalString(root, "request_id"), OptionalString(root, "operation"),
                            OptionalBool(root, "accepted") ?? throw new FormatException("accepted is required"),
                            OptionalString(root, "reason"), ParseFleetControl(root));
                        return true;

                    default:
                        message = new UnknownMessage(type, root.Clone());
                        return true;
                }
            }
            catch (FormatException ex)
            {
                error = ex.Message;
                return false;
            }
        }
    }

    private static FleetControlState? ParseFleetControl(JsonElement root)
    {
        if (!root.TryGetProperty("control", out var value) || value.ValueKind == JsonValueKind.Null)
            return null;
        if (value.ValueKind != JsonValueKind.Object)
            throw new FormatException("control must be an object");
        if (!value.TryGetProperty("routes", out var routes) || routes.ValueKind != JsonValueKind.Object)
            throw new FormatException("control routes required");
        var versions = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var entry in routes.EnumerateObject())
        {
            if (entry.Value.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(entry.Value.GetString())
                || !versions.TryAdd(entry.Name, entry.Value.GetString()!))
                throw new FormatException("unique named route versions required");
        }
        if (versions.Count == 0) throw new FormatException("fleet routes must not be empty");
        var tasks = new List<FleetControlTask>();
        if (value.TryGetProperty("tasks", out var taskArray))
        {
            if (taskArray.ValueKind != JsonValueKind.Array) throw new FormatException("tasks must be an array");
            foreach (var task in taskArray.EnumerateArray())
                tasks.Add(new FleetControlTask(RequiredString(task, "task_id"), RequiredString(task, "pickup"),
                    RequiredString(task, "dropoff"), RequiredString(task, "status"), OptionalString(task, "assigned_agv")));
        }
        Dictionary<string, string?>? faults = null;
        if (value.TryGetProperty("faults", out var faultValues))
        {
            if (faultValues.ValueKind != JsonValueKind.Object) throw new FormatException("faults must be an object");
            faults = new Dictionary<string, string?>(StringComparer.Ordinal);
            foreach (var fault in faultValues.EnumerateObject())
            {
                if (fault.Value.ValueKind != JsonValueKind.String && fault.Value.ValueKind != JsonValueKind.Null)
                    throw new FormatException("fault reason must be a string or null");
                if (!versions.ContainsKey(fault.Name) || !faults.TryAdd(fault.Name, fault.Value.GetString()))
                    throw new FormatException("fault must identify a unique current fleet robot");
            }
        }
        return new FleetControlState(RequiredString(value, "session_id"), RequiredString(value, "scene_id"),
            RequiredString(value, "run_id"), OptionalBool(value, "paused") ?? throw new FormatException("paused required"),
            OptionalStringArray(value, "operations") ?? Array.Empty<string>(), versions,
            tasks, OptionalInt32(value, "task_total") ?? tasks.Count, faults);
    }

    private static string RequiredString(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var element) || element.ValueKind != JsonValueKind.String)
            throw new FormatException($"missing or invalid string field '{name}'");
        var value = element.GetString();
        if (string.IsNullOrWhiteSpace(value))
            throw new FormatException($"field '{name}' must not be empty");
        return value;
    }

    private static string? OptionalString(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var element) || element.ValueKind == JsonValueKind.Null)
            return null;
        if (element.ValueKind != JsonValueKind.String)
            throw new FormatException($"field '{name}' must be a string");
        return element.GetString();
    }

    private static IReadOnlyList<string>? OptionalStringArray(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var element) || element.ValueKind == JsonValueKind.Null)
            return null;
        if (element.ValueKind != JsonValueKind.Array)
            throw new FormatException($"field '{name}' must be an array");
        var values = new List<string>();
        foreach (var item in element.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.String)
                throw new FormatException($"field '{name}' must contain strings");
            values.Add(item.GetString()!);
        }
        return values;
    }

    private static bool? OptionalBool(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var element) || element.ValueKind == JsonValueKind.Null)
            return null;
        if (element.ValueKind != JsonValueKind.True && element.ValueKind != JsonValueKind.False)
            throw new FormatException($"field '{name}' must be a boolean");
        return element.GetBoolean();
    }

    private static IReadOnlyList<double>? OptionalDoubleArray(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var element) || element.ValueKind == JsonValueKind.Null)
            return null;
        if (element.ValueKind != JsonValueKind.Array)
            throw new FormatException($"field '{name}' must be an array");
        var values = new List<double>();
        foreach (var item in element.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Number || !item.TryGetDouble(out var value) || !double.IsFinite(value))
                throw new FormatException($"field '{name}' must contain finite numbers");
            values.Add(value);
        }
        return values;
    }

    /// <summary>
    /// Reads a polyline sent as a list of [x, y] pairs (the shape the routing stream uses).
    /// A malformed pair is a producer bug and must fail loudly rather than be dropped.
    /// </summary>
    private static IReadOnlyList<Point2D>? OptionalPath(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var element) || element.ValueKind == JsonValueKind.Null)
            return null;
        if (element.ValueKind != JsonValueKind.Array)
            throw new FormatException($"field '{name}' must be an array");
        var points = new List<Point2D>();
        foreach (var item in element.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Array || item.GetArrayLength() != 2)
                throw new FormatException($"field '{name}' must contain [x, y] pairs");
            var pair = item.EnumerateArray().ToArray();
            if (pair[0].ValueKind != JsonValueKind.Number || !pair[0].TryGetDouble(out var x) || !double.IsFinite(x) ||
                pair[1].ValueKind != JsonValueKind.Number || !pair[1].TryGetDouble(out var y) || !double.IsFinite(y))
                throw new FormatException($"field '{name}' must contain finite [x, y] pairs");
            points.Add(new Point2D(x, y));
        }
        return points;
    }

    private static double RequiredFiniteDouble(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var element) || element.ValueKind != JsonValueKind.Number ||
            !element.TryGetDouble(out var value) || !double.IsFinite(value))
            throw new FormatException($"missing or non-finite number field '{name}'");
        return value;
    }

    private static double? OptionalFiniteDouble(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var element) || element.ValueKind == JsonValueKind.Null)
            return null;
        if (element.ValueKind != JsonValueKind.Number || !element.TryGetDouble(out var value) || !double.IsFinite(value))
            throw new FormatException($"field '{name}' must be a finite number");
        return value;
    }


    private static (IReadOnlyDictionary<string, string?>? Faults, FleetExecutionIdentity? Identity)
        ParseExecutions(JsonElement root, IReadOnlyList<FleetAgv> agvs, FleetControlState? control)
    {
        var faults = new Dictionary<string, string?>(StringComparer.Ordinal);
        var known = new HashSet<string>(agvs.Select(a => a.AgvId), StringComparer.Ordinal);
        FleetExecutionIdentity? identity = null;
        void Add(JsonElement state, string? key = null)
        {
            if (state.ValueKind != JsonValueKind.Object)
                throw new FormatException("execution state must be an object");
            var robot = RequiredString(state, "robot_id");
            if (!known.Contains(robot) || key is not null && key != robot || faults.ContainsKey(robot))
                throw new FormatException("execution robot must match a unique observed vehicle");
            var next = new FleetExecutionIdentity(RequiredString(state, "scene_id"), RequiredString(state, "run_id"));
            if (identity is not null && identity != next ||
                control is not null && (control.SceneId != next.SceneId || control.RunId != next.RunId))
                throw new FormatException("Mixed scene/run execution state.");
            identity = next;
            // Absence is unknown, not a reported healthy state.
            if (!state.TryGetProperty("fault", out _))
                throw new FormatException("execution state needs an explicit fault or null");
            faults.Add(robot, OptionalString(state, "fault"));
        }
        var present = false;
        if (root.TryGetProperty("executions", out var states))
        {
            present = true;
            if (states.ValueKind != JsonValueKind.Object)
                throw new FormatException("executions must be an object");
            foreach (var state in states.EnumerateObject()) Add(state.Value, state.Name);
        }
        if (root.TryGetProperty("execution", out var single) && single.ValueKind != JsonValueKind.Null)
        {
            present = true;
            Add(single);
        }
        return (present ? faults : null, identity);
    }

    // -----------------------------------------------------------------------
    // Fleet scheduling stream helpers
    //
    // These are kept next to the single-vehicle helpers above but use their own
    // field names, because the scheduling stream is produced by amr_sched and
    // must not be silently coupled to changes in the single-vehicle bridge.
    // -----------------------------------------------------------------------

    private static IReadOnlyList<FleetNode> ParseFleetNodes(JsonElement root)
    {
        // amr_sched sends {"network": {"nodes": [...], "edges": [...]}}.
        var network = RequiredObject(root, "network");
        var nodes = RequiredArray(network, "nodes");
        var result = new List<FleetNode>();
        foreach (var item in nodes.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Object)
                throw new FormatException("network.nodes must contain objects");
            result.Add(new FleetNode(
                RequiredString(item, "id"),
                RequiredString(item, "kind"),
                RequiredFiniteDouble(item, "x"),
                RequiredFiniteDouble(item, "y"),
                OptionalInt32(item, "capacity") ?? 1));
        }
        return result;
    }

    private static IReadOnlyList<FleetEdge> ParseFleetEdges(JsonElement root)
    {
        var network = RequiredObject(root, "network");
        var edges = RequiredArray(network, "edges");
        var nodes = new Dictionary<string,FleetNode>(StringComparer.Ordinal);
        foreach (var node in ParseFleetNodes(root))
            if (!nodes.TryAdd(node.Id,node))
                throw new FormatException("Network node IDs must be unique.");
        var result = new List<FleetEdge>();
        foreach (var item in edges.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Object)
                throw new FormatException("network.edges must contain objects");
            var centerline = OptionalPath(item, "centerline");
            if (item.TryGetProperty("centerline", out _) && centerline is null)
                throw new FormatException("Road centerline must be an array when present.");
            var edge = new FleetEdge(
                RequiredString(item, "src"),
                RequiredString(item, "dst"),
                RequiredFiniteDouble(item, "length"),
                OptionalInt32(item, "capacity") ?? 1, centerline);
            if (centerline is not null)
            {
                if (!nodes.TryGetValue(edge.Src, out var from) || !nodes.TryGetValue(edge.Dst, out var to))
                    throw new FormatException("Road centerline refers to an unknown endpoint.");
                FleetRoadGeometry.Resolve(edge, from, to);
            }
            result.Add(edge);
        }
        return result;
    }

    private static FleetMeta ParseFleetMeta(JsonElement root)
    {
        var meta = RequiredObject(root, "meta");
        // The wire uses snake_case here because the producer is Python; the C#
        // properties carry explicit [JsonPropertyName] attributes for the same
        // reason. Reading them by name keeps the mapping explicit.
        return new FleetMeta(
            OptionalInt32(meta, "n_tasks") ?? 0,
            OptionalInt32(meta, "n_agvs") ?? 0,
            OptionalFiniteDouble(meta, "dt") ?? 0.0,
            OptionalStringArray(meta, "agv_ids") ?? Array.Empty<string>());
    }

    private static IReadOnlyList<FleetAgv> ParseFleetAgvs(JsonElement root)
    {
        var agvs = RequiredArray(root, "agvs");
        var result = new List<FleetAgv>();
        foreach (var item in agvs.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Object)
                throw new FormatException("agvs must contain objects");

            var poseElement = RequiredObject(item, "pose");
            var pose = new FleetPose(
                RequiredString(poseElement, "agv_id"),
                OptionalString(poseElement, "node"),
                OptionalString(poseElement, "edge"),
                OptionalFiniteDouble(poseElement, "progress") ?? 0.0,
                OptionalString(poseElement, "source") ?? "sim",
                OptionalFiniteDouble(poseElement, "timestamp"),
                OptionalFiniteDouble(poseElement, "x"),
                OptionalFiniteDouble(poseElement, "y"),
                OptionalFiniteDouble(poseElement, "yaw"),
                OptionalBool(poseElement, "available"),
                OptionalString(poseElement, "frame_id"));
            if (pose.Available == true &&
                (pose.X is null || pose.Y is null || pose.FrameId != "network"))
                throw new FormatException("available actual pose requires x/y in network frame");

            result.Add(new FleetAgv(
                RequiredString(item, "agv_id"),
                pose,
                RequiredString(item, "state"),
                OptionalString(item, "task_id"),
                OptionalStringArray(item, "path") ?? Array.Empty<string>()));
        }
        return result;
    }

    private static IReadOnlyDictionary<string, int>? ParseTaskCounts(JsonElement root)
    {
        if (!root.TryGetProperty("task_counts", out var element) || element.ValueKind == JsonValueKind.Null)
            return null;
        if (element.ValueKind != JsonValueKind.Object)
            throw new FormatException("task_counts must be an object");
        var result = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var property in element.EnumerateObject())
        {
            if (property.Value.ValueKind != JsonValueKind.Number || !property.Value.TryGetInt32(out var count))
                throw new FormatException($"task_counts['{property.Name}'] must be an integer");
            result[property.Name] = count;
        }
        return result;
    }

    private static IReadOnlyList<FleetPreemption>? ParsePreemptions(JsonElement root)
    {
        if (!root.TryGetProperty("preemptions", out var element) || element.ValueKind == JsonValueKind.Null)
            return null;
        if (element.ValueKind != JsonValueKind.Array)
            throw new FormatException("preemptions must be an array");
        var result = new List<FleetPreemption>();
        foreach (var item in element.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Object)
                throw new FormatException("preemptions must contain objects");
            result.Add(new FleetPreemption(
                RequiredString(item, "urgent"),
                RequiredString(item, "victim_agv"),
                RequiredString(item, "victim_task"),
                item.TryGetProperty("was_picked", out var picked) && picked.ValueKind == JsonValueKind.True));
        }
        return result;
    }

    /// <summary>Parses node -> holding vehicles. Absent means "this producer does not report holds".</summary>
    private static IReadOnlyDictionary<string, IReadOnlyList<string>>? ParseHolds(JsonElement root)
    {
        if (!root.TryGetProperty("holds", out var element) || element.ValueKind == JsonValueKind.Null)
            return null;
        return ParseResourceOwners(element, "holds");
    }

    private static FleetResourcePins? ParsePhysicalHolds(JsonElement root)
    {
        if (!root.TryGetProperty("physical_holds", out var element) || element.ValueKind == JsonValueKind.Null)
            return null;
        if (element.ValueKind != JsonValueKind.Object) throw new FormatException("physical_holds must be an object");
        return new FleetResourcePins(ParseResourceOwners(RequiredObject(element, "nodes"), "physical_holds.nodes"),
            ParseResourceOwners(RequiredObject(element, "edges"), "physical_holds.edges"));
    }

    private static IReadOnlyDictionary<string, IReadOnlyList<string>> ParseResourceOwners(JsonElement element, string name)
    {
        if (element.ValueKind != JsonValueKind.Object)
            throw new FormatException(name + " must be an object");
        var result = new Dictionary<string, IReadOnlyList<string>>(StringComparer.Ordinal);
        foreach (var property in element.EnumerateObject())
        {
            if (property.Value.ValueKind != JsonValueKind.Array)
                throw new FormatException($"{name}['{property.Name}'] must be an array");
            var owners = new List<string>();
            foreach (var owner in property.Value.EnumerateArray())
            {
                if (owner.ValueKind != JsonValueKind.String)
                    throw new FormatException($"{name}['{property.Name}'] must contain strings");
                owners.Add(owner.GetString()!);
            }
            result[property.Name] = owners;
        }
        return result;
    }




    /// <summary>Parses vehicle -> provenance, e.g. {"AGV00": "sim"}.</summary>
    private static IReadOnlyDictionary<string, string>? ParseTrailSources(JsonElement root)
    {
        if (!root.TryGetProperty("trail_sources", out var element) || element.ValueKind == JsonValueKind.Null)
            return null;
        if (element.ValueKind != JsonValueKind.Object)
            throw new FormatException("trail_sources must be an object");
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var kv in element.EnumerateObject())
        {
            if (kv.Value.ValueKind != JsonValueKind.String)
                throw new FormatException($"trail_sources['{kv.Name}'] must be a string");
            result[kv.Name] = kv.Value.GetString()!;
        }
        return result;
    }

    /// <summary>Parses the optional physical-robot block: id, source, pose, trail.</summary>
    /// <para>
    /// Absent means "no physical robot connected", which is the normal state in 算法研究模式.
    /// Present but malformed is an error: a half-understood pose would be drawn as a robot
    /// standing somewhere it never was.
    /// </para>
    private static FleetPhysical? ParsePhysical(JsonElement root)
    {
        if (!root.TryGetProperty("physical", out var element) || element.ValueKind == JsonValueKind.Null)
            return null;
        if (element.ValueKind != JsonValueKind.Object)
            throw new FormatException("physical must be an object");

        var id = OptionalString(element, "id") ?? "scout_mini";
        var source = OptionalString(element, "source") ?? "gazebo";

        double[]? pose = null;
        if (element.TryGetProperty("pose", out var p) && p.ValueKind == JsonValueKind.Array)
        {
            var values = p.EnumerateArray().Where(v => v.ValueKind == JsonValueKind.Number)
                                          .Select(v => v.GetDouble()).ToArray();
            if (values.Length >= 2) pose = values;
        }

        var trail = new List<FleetPoint>();
        if (element.TryGetProperty("trail", out var t) && t.ValueKind == JsonValueKind.Array)
        {
            foreach (var point in t.EnumerateArray())
            {
                var xy = point.EnumerateArray().ToArray();
                if (xy.Length < 2) throw new FormatException("physical trail point is not a pair");
                trail.Add(new FleetPoint(xy[0].GetDouble(), xy[1].GetDouble()));
            }
        }
        return new FleetPhysical(id, source, pose, trail);
    }

    /// <summary>Parses vehicle -> [{x, y}, ...], the trail the vehicle actually drove.</summary>
    /// <para>
    /// Absent is not the same as malformed: a producer that does not publish trails yields
    /// <c>null</c> and the console simply draws no trail. A present-but-broken field is an
    /// error, so a typo cannot silently degrade to "no trail" and be read as "it never moved".
    /// </para>
    private static IReadOnlyDictionary<string, IReadOnlyList<FleetPoint>>? ParseTrails(JsonElement root)
    {
        if (!root.TryGetProperty("trails", out var element) || element.ValueKind == JsonValueKind.Null)
            return null;
        if (element.ValueKind != JsonValueKind.Object)
            throw new FormatException("trails must be an object");

        var result = new Dictionary<string, IReadOnlyList<FleetPoint>>(StringComparer.Ordinal);
        foreach (var vehicle in element.EnumerateObject())
        {
            if (vehicle.Value.ValueKind != JsonValueKind.Array)
                throw new FormatException($"trails['{vehicle.Name}'] must be an array");
            var points = new List<FleetPoint>();
            foreach (var p in vehicle.Value.EnumerateArray())
            {
                if (p.ValueKind != JsonValueKind.Array)
                    throw new FormatException($"trails['{vehicle.Name}'] points must be [x, y] pairs");
                var xy = p.EnumerateArray().ToArray();
                if (xy.Length < 2 || xy[0].ValueKind != JsonValueKind.Number
                    || xy[1].ValueKind != JsonValueKind.Number)
                    throw new FormatException($"trails['{vehicle.Name}'] point is not two numbers");
                points.Add(new FleetPoint(xy[0].GetDouble(), xy[1].GetDouble()));
            }
            result[vehicle.Name] = points;
        }
        return result;
    }

    /// <summary>Parses node -> {claims, capacity, agvs}, the nodes at or over capacity.</summary>
    private static IReadOnlyDictionary<string, FleetHold>? ParseCongestion(JsonElement root)
    {
        if (!root.TryGetProperty("congestion", out var element) || element.ValueKind == JsonValueKind.Null)
            return null;
        if (element.ValueKind != JsonValueKind.Object)
            throw new FormatException("congestion must be an object");
        var result = new Dictionary<string, FleetHold>(StringComparer.Ordinal);
        foreach (var property in element.EnumerateObject())
        {
            if (property.Value.ValueKind != JsonValueKind.Object)
                throw new FormatException($"congestion['{property.Name}'] must be an object");
            var owners = new List<string>();
            if (property.Value.TryGetProperty("agvs", out var agvs) && agvs.ValueKind == JsonValueKind.Array)
            {
                foreach (var owner in agvs.EnumerateArray())
                    if (owner.ValueKind == JsonValueKind.String) owners.Add(owner.GetString()!);
            }
            result[property.Name] = new FleetHold(
                OptionalInt32(property.Value, "claims") ?? owners.Count,
                OptionalInt32(property.Value, "capacity") ?? 1,
                owners,
                OptionalInt32(property.Value, "blocked") ?? 0,
                OptionalInt32(property.Value, "steps") ?? 0);
        }
        return result;
    }

    private static JsonElement RequiredObject(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var element) || element.ValueKind != JsonValueKind.Object)
            throw new FormatException($"missing or invalid object field '{name}'");
        return element;
    }

    private static JsonElement RequiredArray(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var element) || element.ValueKind != JsonValueKind.Array)
            throw new FormatException($"missing or invalid array field '{name}'");
        return element;
    }

    private static int? OptionalInt32(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var element) || element.ValueKind == JsonValueKind.Null)
            return null;
        if (element.ValueKind != JsonValueKind.Number || !element.TryGetInt32(out var value))
            throw new FormatException($"field '{name}' must be an integer");
        return value;
    }

    private static long RequiredInt64(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var element) || element.ValueKind != JsonValueKind.Number ||
            !element.TryGetInt64(out var value))
            throw new FormatException($"missing or invalid integer field '{name}'");
        return value;
    }

    private static int RequiredInt32(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var element) || element.ValueKind != JsonValueKind.Number ||
            !element.TryGetInt32(out var value))
            throw new FormatException($"missing or invalid integer field '{name}'");
        return value;
    }

    private static IReadOnlyList<Point2D> RequiredPath(JsonElement root)
    {
        if (!root.TryGetProperty("path", out var element) || element.ValueKind != JsonValueKind.Array)
            throw new FormatException("field 'path' must be a non-empty array");
        var path = new List<Point2D>();
        foreach (var item in element.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Array || item.GetArrayLength() != 2)
                throw new FormatException("each path element must be [x, y]");
            var x = item[0];
            var y = item[1];
            if (x.ValueKind != JsonValueKind.Number || y.ValueKind != JsonValueKind.Number ||
                !x.TryGetDouble(out var xv) || !y.TryGetDouble(out var yv) ||
                !double.IsFinite(xv) || !double.IsFinite(yv))
                throw new FormatException("path coordinates must be finite numbers");
            path.Add(new Point2D(xv, yv));
        }
        if (path.Count == 0)
            throw new FormatException("field 'path' must not be empty");
        return path;
    }
}
