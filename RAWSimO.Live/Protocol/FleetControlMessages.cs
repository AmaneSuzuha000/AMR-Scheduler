using System.Text.Json;
using System.Text.Json.Serialization;

namespace RAWSimO.Live.Protocol;

/// <summary>The current durable owner and exact fleet action versions.</summary>
public sealed record FleetControlState(
    [property: JsonPropertyName("session_id")] string SessionId,
    [property: JsonPropertyName("scene_id")] string SceneId,
    [property: JsonPropertyName("run_id")] string RunId,
    bool Paused,
    IReadOnlyList<string> Operations,
    IReadOnlyDictionary<string, string> Routes,
    IReadOnlyList<FleetControlTask>? Tasks = null,
    [property: JsonPropertyName("task_total")] int TaskTotal = 0,
    IReadOnlyDictionary<string, string?>? Faults = null);

/// <summary>Task status comes from Python; the console does not infer completion.</summary>
public sealed record FleetControlTask(
    [property: JsonPropertyName("task_id")] string TaskId,
    string Pickup, string Dropoff, string Status,
    [property: JsonPropertyName("assigned_agv")] string? AssignedAgv);

/// <summary>A request to the Python owner. Payload keys retain exact robot IDs.</summary>
public sealed record FleetControlRequestMessage(
    [property: JsonPropertyName("session_id")] string SessionId,
    [property: JsonPropertyName("scene_id")] string SceneId,
    [property: JsonPropertyName("run_id")] string RunId,
    [property: JsonPropertyName("request_id")] string RequestId,
    string Operation, JsonElement Payload) : LiveMessage("control_request");

/// <summary>Receipt means an operation was durably accepted, not physical arrival.</summary>
public sealed record FleetControlResultMessage(
    [property: JsonPropertyName("request_id")] string? RequestId,
    string? Operation, bool Accepted, string? Reason,
    FleetControlState? Control) : LiveMessage("control_result");
