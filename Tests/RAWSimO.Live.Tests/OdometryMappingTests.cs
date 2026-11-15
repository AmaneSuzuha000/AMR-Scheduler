using System;
using RAWSimO.Live.Drivers;
using Xunit;

#nullable enable

namespace RAWSimO.Live.Tests;

/// <summary>
/// P3-A anchor, map mapping and drift-reporting tests. These pin the behaviour that decides
/// whether bare wheel odometry is acceptable, so the conclusion cannot be argued away later.
/// </summary>
public class OdometryMappingTests
{
    [Fact]
    public void AnchorMustBeSetBeforeMapping()
    {
        var anchor = new StartPoseAnchor();
        Assert.False(anchor.IsSet);
        Assert.Throws<InvalidOperationException>(() => anchor.CreateOdometry(ScoutMiniGeometry.Default));
        Assert.Throws<InvalidOperationException>(() => OdometryToMap.Apply(anchor, 0, 0, 0));
    }

    [Fact]
    public void AnchorAtOriginIsIdentity()
    {
        var anchor = new StartPoseAnchor();
        anchor.Set(0, 0, 0);

        var (x, y, yaw) = OdometryToMap.Apply(anchor, 1.5, -0.5, 0.25);

        Assert.Equal(1.5, x, 9);
        Assert.Equal(-0.5, y, 9);
        Assert.Equal(0.25, yaw, 9);
    }

    [Fact]
    public void AnchorTranslates()
    {
        var anchor = new StartPoseAnchor();
        anchor.Set(2.0, 3.0, 0.0);

        var (x, y, _) = OdometryToMap.Apply(anchor, 0.5, 0.25, 0.0);

        Assert.Equal(2.5, x, 9);
        Assert.Equal(3.25, y, 9);
    }

    [Fact]
    public void AnchorRotatesRigFrameIntoMapFrame()
    {
        var anchor = new StartPoseAnchor();
        anchor.Set(0, 0, Math.PI / 2.0);   // map frame is rotated 90 degrees from the rig

        // Moving 1 m along the rig's +x must land 1 m along the map's +y.
        var (x, y, yaw) = OdometryToMap.Apply(anchor, 1.0, 0.0, 0.0);

        Assert.Equal(0.0, x, 9);
        Assert.Equal(1.0, y, 9);
        Assert.Equal(Math.PI / 2.0, yaw, 9);
    }

    [Fact]
    public void AnchorYawAddsAndWraps()
    {
        var anchor = new StartPoseAnchor();
        anchor.Set(0, 0, 3.0);

        var (_, _, yaw) = OdometryToMap.Apply(anchor, 0, 0, 3.0);

        Assert.Equal(WheelOdometry.WrapAngle(6.0), yaw, 9);
    }

    [Fact]
    public void ClearRemovesTheAnchor()
    {
        var anchor = new StartPoseAnchor();
        anchor.Set(1, 2, 3);
        anchor.Clear();

        Assert.False(anchor.IsSet);
        Assert.Throws<InvalidOperationException>(() => OdometryToMap.Apply(anchor, 0, 0, 0));
    }

    [Fact]
    public void DriftReportComputesErrors()
    {
        var report = new DriftReport();
        report.Add(new DriftSample(1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0));   // no error yet
        report.Add(new DriftSample(2.0, 1.9, 0.0, 0.0, 2.0, 0.0, 0.0));   // 0.1 m behind

        Assert.Equal(0.1, report.FinalPositionError, 9);
        Assert.Equal(0.1, report.MaxPositionError, 9);
        Assert.Equal(0.05, report.ErrorPerMetre, 9);   // 0.1 m over 2.0 m travelled
        Assert.Equal(2, report.Samples.Count);
        Assert.True(report.ExceedsTolerance(0.05));
        Assert.False(report.ExceedsTolerance(0.2));
    }

    [Fact]
    public void DriftReportWrapsYawError()
    {
        var report = new DriftReport();
        // +179 degrees estimated, -179 degrees actual: the error is 2 degrees across the seam.
        report.Add(new DriftSample(1.0, 0, 0, 179.0 * Math.PI / 180.0, 0, 0, -179.0 * Math.PI / 180.0));

        Assert.Equal(-2.0, report.Samples[0].YawError * 180.0 / Math.PI, 6);
    }

    [Fact]
    public void EmptyReportDoesNotDivideByZero()
    {
        var report = new DriftReport();
        Assert.Equal(0.0, report.FinalPositionError);
        Assert.Equal(0.0, report.ErrorPerMetre);
        Assert.Contains("no samples", report.Summarise());
    }

    [Fact]
    public void ZeroDistanceDoesNotProduceInfiniteError()
    {
        var report = new DriftReport();
        report.Add(new DriftSample(0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0));

        Assert.Equal(0.0, report.ErrorPerMetre);
    }

    [Fact]
    public void SummariseMentionsTheDecidingFigures()
    {
        var report = new DriftReport();
        report.Add(new DriftSample(10.0, 9.9, 0.0, 0.0, 10.0, 0.0, 0.0));

        var text = report.Summarise();
        Assert.Contains("final_error", text);
        Assert.Contains("error_per_metre", text);
        Assert.Contains("max_error", text);
    }

    [Fact]
    public void ReplaySourceFeedsTheReaderAndAdvancesThePose()
    {
        var geometry = ScoutMiniGeometry.Default;
        var counts = geometry.CountsPerRevolution;
        var frames = new (WheelCounters, double)[]
        {
            (new WheelCounters(0, 0), 0.00),
            (new WheelCounters(counts / 2, counts / 2), 0.05),
            (new WheelCounters(counts, counts), 0.10),
        };

        var source = new ReplayCanSource(frames);
        Assert.True(source.TryOpen(out var error));
        Assert.Equal(string.Empty, error);
        Assert.Contains("replay", source.Describe());

        var reader = new ScoutMiniOdometryReader(source, new WheelOdometry(geometry));
        Assert.Equal(3, reader.Pump());
        Assert.Equal(3, reader.FramesRead);

        var expected = 2.0 * Math.PI * geometry.WheelRadius;
        Assert.Equal(expected, reader.Odometry.X, 9);
        Assert.Equal(0.10, reader.LastFrameTimestamp, 9);
        Assert.Equal(0.0, reader.AgeSeconds(0.10), 9);
        Assert.Equal(0.05, reader.AgeSeconds(0.15), 9);
    }

    [Fact]
    public void ReaderReportsStalenessWithNoFrames()
    {
        var reader = new ScoutMiniOdometryReader(
            new ReplayCanSource(Array.Empty<(WheelCounters, double)>()),
            new WheelOdometry(ScoutMiniGeometry.Default));

        Assert.Equal(double.PositiveInfinity, reader.AgeSeconds(1.0));
        Assert.Equal(0, reader.Pump());
    }

    [Fact]
    public void EmptyReplaySourceRefusesToOpen()
    {
        var source = new ReplayCanSource(Array.Empty<(WheelCounters, double)>());
        Assert.False(source.TryOpen(out var error));
        Assert.NotEqual(string.Empty, error.Length == 0 ? "x" : error);
    }
}
