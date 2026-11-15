using RAWSimO.Live.Planning;
using Xunit;

namespace RAWSimO.Live.Tests;

public class NavExecutionTests
{
    [Fact]
    public void StartsIdleAndBecomesExecuting()
    {
        var nav = new NavExecution();
        Assert.Equal(NavExecutionState.Idle, nav.State);
        Assert.False(nav.IsActive);
        Assert.Null(nav.Start("g1", 10.0, 100.0));
        Assert.Equal(NavExecutionState.Executing, nav.State);
        Assert.Equal("g1", nav.Goal);
        Assert.Equal(100.0 + 2.0 * 10.0 + NavExecution.TimeoutSlackSeconds, nav.Deadline, 9);
    }

    [Fact]
    public void SuccessResultCompletesTheGoal()
    {
        var nav = new NavExecution();
        nav.Start("g1", 5.0, 0.0);
        Assert.True(nav.OnResult("g1", "success", null));
        Assert.Equal(NavExecutionState.Succeeded, nav.State);
        Assert.Equal("success", nav.ResultStatus);
        Assert.False(nav.IsActive);
    }

    [Fact]
    public void FailureResultKeepsTheReason()
    {
        var nav = new NavExecution();
        nav.Start("g1", 5.0, 0.0);
        Assert.True(nav.OnResult("g1", "aborted", "obstacle"));
        Assert.Equal(NavExecutionState.Failed, nav.State);
        Assert.Equal("obstacle", nav.ResultReason);
    }

    [Fact]
    public void ResultForAnotherGoalIsIgnored()
    {
        var nav = new NavExecution();
        nav.Start("g1", 5.0, 0.0);
        Assert.False(nav.OnResult("g0", "success", null));
        Assert.Equal(NavExecutionState.Executing, nav.State);
    }

    [Fact]
    public void CancelReturnsGoalAndStopsExecution()
    {
        var nav = new NavExecution();
        nav.Start("g1", 5.0, 0.0);
        Assert.Equal("g1", nav.Cancel());
        Assert.Equal(NavExecutionState.Cancelled, nav.State);
        Assert.Null(nav.Cancel());
    }

    [Fact]
    public void DisconnectMarksUnknownAndNeverRecovers()
    {
        var nav = new NavExecution();
        nav.Start("g1", 5.0, 0.0);
        Assert.Equal("g1", nav.OnDisconnected());
        Assert.Equal(NavExecutionState.Unknown, nav.State);
        Assert.False(nav.OnResult("g1", "success", null));
        Assert.Equal(NavExecutionState.Unknown, nav.State);
        Assert.Null(nav.Tick(1e6));
    }

    [Fact]
    public void TimeoutFiresOnlyAfterDeadline()
    {
        var nav = new NavExecution();
        nav.Start("g1", 10.0, 100.0);
        Assert.Null(nav.Tick(125.0 - 1e-6));
        Assert.Equal(NavExecutionState.Executing, nav.State);
        Assert.Equal("g1", nav.Tick(125.0 + 1e-6));
        Assert.Equal(NavExecutionState.TimedOut, nav.State);
    }

    [Fact]
    public void NewTargetReplacesRunningGoal()
    {
        var nav = new NavExecution();
        nav.Start("g1", 5.0, 0.0);
        Assert.Equal("g1", nav.Start("g2", 6.0, 1.0));
        Assert.Equal("g2", nav.Goal);
        Assert.Equal(NavExecutionState.Executing, nav.State);
        Assert.False(nav.OnResult("g1", "success", null));
        Assert.True(nav.OnResult("g2", "success", null));
    }

    [Fact]
    public void StatusAliasesCountAsSuccess()
    {
        Assert.True(NavExecution.IsSuccess("SUCCESS"));
        Assert.True(NavExecution.IsSuccess("done"));
        Assert.True(NavExecution.IsSuccess("reached"));
        Assert.False(NavExecution.IsSuccess("aborted"));
        Assert.False(NavExecution.IsSuccess(null));
    }

    [Fact]
    public void ResetClearsEverything()
    {
        var nav = new NavExecution();
        nav.Start("g1", 5.0, 0.0);
        nav.Reset();
        Assert.Equal(NavExecutionState.Idle, nav.State);
        Assert.Null(nav.Goal);
        Assert.Equal(0.0, nav.Deadline);
    }

    [Fact]
    public void NegativePlannedTimeIsClamped()
    {
        var nav = new NavExecution();
        nav.Start("g1", -3.0, 0.0);
        Assert.Equal(NavExecution.TimeoutSlackSeconds, nav.Deadline, 9);
    }
}
