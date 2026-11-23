using System;

namespace RAWSimO.Live.Planning;

/// <summary>State of one navigation goal sent to the vehicle.</summary>
public enum NavExecutionState
{
    /// <summary>No goal is active.</summary>
    Idle,
    /// <summary>A goal was sent and the vehicle has not answered yet.</summary>
    Executing,
    /// <summary>The vehicle reported success.</summary>
    Succeeded,
    /// <summary>The vehicle reported a failure with a reason.</summary>
    Failed,
    /// <summary>The operator cancelled the goal.</summary>
    Cancelled,
    /// <summary>The connection dropped while the goal was running; the outcome is unknown.</summary>
    Unknown,
    /// <summary>No result arrived within 2 x planned time + 5 s.</summary>
    TimedOut,
    /// <summary>A new goal replaced the previous one before it finished.</summary>
    Replaced,
}

/// <summary>
/// Tracks one navigation goal at a time: start, result, cancel, replacement, disconnect and timeout.
/// The GUI and the tests share this state machine so the transitions are verified without a window.
/// </summary>
public sealed class NavExecution
{
    /// <summary>Extra time granted on top of twice the planned duration.</summary>
    public const double TimeoutSlackSeconds = 5.0;

    /// <summary>Current state.</summary>
    public NavExecutionState State { get; private set; } = NavExecutionState.Idle;

    /// <summary>Goal id of the active or last finished goal.</summary>
    public string? Goal { get; private set; }

    /// <summary>Planned duration of the active goal in seconds.</summary>
    public double PlannedSeconds { get; private set; }

    /// <summary>Absolute deadline of the active goal in the caller's time base.</summary>
    public double Deadline { get; private set; }

    /// <summary>Status text reported by the vehicle for the last finished goal.</summary>
    public string? ResultStatus { get; private set; }

    /// <summary>Reason reported by the vehicle for the last finished goal.</summary>
    public string? ResultReason { get; private set; }

    /// <summary>True while a goal is waiting for its result.</summary>
    public bool IsActive => State == NavExecutionState.Executing;

    /// <summary>
    /// Starts a goal. A running goal is first reported as replaced so the caller can cancel it on the vehicle.
    /// </summary>
    /// <returns>The goal id that the caller still has to cancel, or null.</returns>
    public string? Start(string goal, double plannedSeconds, double now)
    {
        if (string.IsNullOrWhiteSpace(goal)) throw new ArgumentException("goal must not be empty", nameof(goal));
        if (!double.IsFinite(plannedSeconds) || plannedSeconds < 0.0) plannedSeconds = 0.0;
        var replaced = IsActive ? Goal : null;
        Goal = goal;
        PlannedSeconds = plannedSeconds;
        Deadline = now + 2.0 * plannedSeconds + TimeoutSlackSeconds;
        ResultStatus = null;
        ResultReason = null;
        State = NavExecutionState.Executing;
        return replaced;
    }

    /// <summary>Applies a vehicle result. Results for other goals are ignored.</summary>
    /// <returns>True when the result belonged to the active goal.</returns>
    public bool OnResult(string? goal, string? status, string? reason)
    {
        if (!IsActive) return false;
        if (!string.Equals(goal, Goal, StringComparison.Ordinal)) return false;
        ResultStatus = status;
        ResultReason = reason;
        State = IsSuccess(status) ? NavExecutionState.Succeeded : NavExecutionState.Failed;
        return true;
    }

    /// <summary>Marks the active goal as cancelled by the operator.</summary>
    /// <returns>The goal id to cancel on the vehicle, or null when nothing was active.</returns>
    public string? Cancel()
    {
        if (!IsActive) return null;
        var goal = Goal;
        State = NavExecutionState.Cancelled;
        ResultStatus = null;
        ResultReason = null;
        return goal;
    }

    /// <summary>Marks the active goal as unknown because the connection dropped. Never recovers automatically.</summary>
    /// <returns>The goal that was running, or null.</returns>
    public string? OnDisconnected()
    {
        if (!IsActive) return null;
        var goal = Goal;
        State = NavExecutionState.Unknown;
        return goal;
    }

    /// <summary>Times the active goal out once its deadline passed.</summary>
    /// <returns>The goal that timed out, or null.</returns>
    public string? Tick(double now)
    {
        if (!IsActive || now <= Deadline) return null;
        var goal = Goal;
        State = NavExecutionState.TimedOut;
        return goal;
    }

    /// <summary>Clears the state without touching the vehicle.</summary>
    public void Reset()
    {
        State = NavExecutionState.Idle;
        Goal = null;
        PlannedSeconds = 0.0;
        Deadline = 0.0;
        ResultStatus = null;
        ResultReason = null;
    }

    /// <summary>Maps a vehicle status string to success.</summary>
    public static bool IsSuccess(string? status) =>
        string.Equals(status, "success", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(status, "succeeded", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(status, "done", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(status, "reached", StringComparison.OrdinalIgnoreCase);

    /// <summary>Short Chinese label for a state, used in the live panel.</summary>
    public static string Describe(NavExecutionState state) => state switch
    {
        NavExecutionState.Idle => "空闲",
        NavExecutionState.Executing => "执行中",
        NavExecutionState.Succeeded => "已到达",
        NavExecutionState.Failed => "失败",
        NavExecutionState.Cancelled => "已取消",
        NavExecutionState.Unknown => "中断（未知）",
        NavExecutionState.TimedOut => "超时",
        NavExecutionState.Replaced => "被新目标替换",
        _ => state.ToString(),
    };
}
