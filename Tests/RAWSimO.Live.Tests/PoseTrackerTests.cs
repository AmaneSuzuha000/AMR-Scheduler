using RAWSimO.Live.Protocol;
using RAWSimO.Live.Tracking;
using Xunit;

namespace RAWSimO.Live.Tests;

public class PoseTrackerTests
{
    [Fact]
    public void FreshnessUsesLocalMonotonicClock()
    {
        var tracker = new PoseTracker(3.0, 3.0);
        tracker.Update(new PoseMessage("agv1", 1, 0, 0.1, 0.2, 0.0), 100.0, DateTimeOffset.UtcNow);
        Assert.Equal(PoseFreshness.Fresh, tracker.GetFreshness(100.10));
        Assert.Equal(PoseFreshness.Stale, tracker.GetFreshness(100.16));
        Assert.Equal(PoseFreshness.Lost, tracker.GetFreshness(102.10));
        Assert.Equal(PoseFreshness.Lost, new PoseTracker(3.0, 3.0).GetFreshness(100.0));
    }

    [Fact]
    public void SequenceGapAndRestartAreDetected()
    {
        var tracker = new PoseTracker(3.0, 3.0);
        Assert.True(tracker.Update(new PoseMessage("agv1", 1, 0, 0.1, 0.1, 0), 1.0, DateTimeOffset.UtcNow).Accepted);
        var gap = tracker.Update(new PoseMessage("agv1", 3, 0, 0.2, 0.2, 0), 1.1, DateTimeOffset.UtcNow);
        Assert.True(gap.SequenceGap);
        Assert.Equal(1, tracker.GapCount);

        var restart = tracker.Update(new PoseMessage("agv1", 2, 0, 0.3, 0.3, 0), 1.2, DateTimeOffset.UtcNow);
        Assert.True(restart.NewSession);
        Assert.Equal(2, tracker.Latest!.Seq);
        Assert.Equal(0, tracker.GapCount);
    }

    [Fact]
    public void InvalidAndExtremePosesAreDiscarded()
    {
        var tracker = new PoseTracker(3.0, 3.0);
        Assert.True(tracker.Update(new PoseMessage("agv1", 1, 0, double.NaN, 0, 0), 1.0, DateTimeOffset.UtcNow).Discarded);
        Assert.True(tracker.Update(new PoseMessage("agv1", 2, 0, 100.0, 0, 0), 1.0, DateTimeOffset.UtcNow).Discarded);
        Assert.Equal(2, tracker.InvalidCount);
    }

    [Fact]
    public void OutsideMapIsAcceptedAndFlagged()
    {
        var tracker = new PoseTracker(3.0, 3.0);
        var result = tracker.Update(new PoseMessage("agv1", 1, 0, 3.1, 0.5, 0), 1.0, DateTimeOffset.UtcNow);
        Assert.True(result.Accepted);
        Assert.True(result.OutsideMap);
        Assert.True(tracker.Latest!.OutsideMap);
    }

    /// <summary>
    /// A frozen pose is indistinguishable from a parked vehicle by age alone.
    /// <para>
    /// Measured 2026-11-05 on the SCOUT Mini: the gazebo bridge's rclpy thread died and the
    /// TCP side kept publishing one unchanged pose at 20 Hz, with the sequence still
    /// advancing. Every age-based freshness rule calls that "fresh". Only the status line
    /// the bridge sends about itself says otherwise.
    /// </para>
    /// </summary>
    [Fact]
    public void SourceReportedStalenessOverridesAFreshArrivalTime()
    {
        var tracker = new PoseTracker(3.0, 3.0);
        tracker.Update(new PoseMessage("agv1", 1, 0, 0.1, 0.2, 0.0), 100.0, DateTimeOffset.UtcNow);
        Assert.Equal(PoseFreshness.Fresh, tracker.GetFreshness(100.01));

        tracker.SetSourceError("odom_stale");

        Assert.True(tracker.SourceStale);
        // The pose itself is 10 ms old, so nothing about its age explains this.
        Assert.Equal(PoseFreshness.Lost, tracker.GetFreshness(100.01));
        // ... and the position is still reported, not invented: the layer may draw it, but
        // only as untrustworthy.
        Assert.Equal(0.1, tracker.Latest!.X, 9);
    }

    [Fact]
    public void ADeliberateStopIsNotMistakenForALostSource()
    {
        // The watchdog and the e-stop hold the vehicle still on purpose; a still pose is then
        // the truth and must keep its normal freshness.
        var tracker = new PoseTracker(3.0, 3.0);
        tracker.Update(new PoseMessage("agv1", 1, 0, 0.1, 0.2, 0.0), 100.0, DateTimeOffset.UtcNow);
        tracker.SetSourceError("watchdog");
        Assert.False(tracker.SourceStale);
        Assert.Equal(PoseFreshness.Fresh, tracker.GetFreshness(100.01));

        tracker.SetSourceError(null);
        Assert.False(tracker.SourceStale);
        Assert.Equal(PoseFreshness.Fresh, tracker.GetFreshness(100.01));
    }

    [Fact]
    public void RecoveringSourceBecomesFreshAgainOnceItReportsNoError()
    {
        var tracker = new PoseTracker(3.0, 3.0);
        tracker.Update(new PoseMessage("agv1", 1, 0, 0.1, 0.2, 0.0), 100.0, DateTimeOffset.UtcNow);
        tracker.SetSourceError("odom_stale");
        Assert.Equal(PoseFreshness.Lost, tracker.GetFreshness(100.01));

        tracker.SetSourceError(null);
        Assert.Equal(PoseFreshness.Fresh, tracker.GetFreshness(100.01));

        tracker.Reset();
        Assert.Null(tracker.SourceError);
    }

    [Fact]
    public void TrailSkipsTinyMovements()
    {
        var tracker = new PoseTracker(3.0, 3.0);
        tracker.Update(new PoseMessage("agv1", 1, 0, 0.1, 0.1, 0), 1.0, DateTimeOffset.UtcNow);
        tracker.Update(new PoseMessage("agv1", 2, 0, 0.105, 0.1, 0), 1.1, DateTimeOffset.UtcNow);
        Assert.Single(tracker.SnapshotTrail());
        tracker.Update(new PoseMessage("agv1", 3, 0, 0.2, 0.1, 0), 1.2, DateTimeOffset.UtcNow);
        Assert.Equal(2, tracker.SnapshotTrail().Count);
    }
}
