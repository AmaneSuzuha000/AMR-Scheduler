using RAWSimO.Live.Calibration;
using RAWSimO.Live.Geometry;
using Xunit;

namespace RAWSimO.Live.Tests;

public class CalibrationSessionTests
{
    [Fact]
    public void PoseAveragerHandlesYawWrapAround()
    {
        var poses = new[]
        {
            new Pose2D(0.0, 0.0, Math.PI - 0.01),
            new Pose2D(0.02, -0.01, -Math.PI + 0.01),
            new Pose2D(-0.01, 0.02, Math.PI),
        };
        var average = PoseAverager.Average(poses);
        Assert.Equal(3, average.Count);
        Assert.InRange(Math.Abs(AngleMath.ShortestDifference(Math.PI, average.Mean.Yaw)), 0.0, 0.02);
        Assert.True(average.PositionStdDev < 0.02);
    }

    [Fact]
    public void FromOriginProducesIdentityWhenFramesCoincide()
    {
        var result = CalibrationSolver.FromOrigin(
            new Point2D(1.0, 2.0), 0.25,
            new Point2D(1.0, 2.0), 0.25);
        var p = result.ChassisToMap(new Point2D(1.5, 2.25));
        Assert.True(Point2D.Distance(p, new Point2D(1.5, 2.25)) < 1e-12);
        Assert.Equal(0.0, result.Theta, 12);
    }

    [Fact]
    public void ResidualFactorMatchesPlanTable()
    {
        Assert.Equal(1.94, CalibrationSolver.ResidualFactor(4));
        Assert.Equal(1.92, CalibrationSolver.ResidualFactor(5));
        Assert.Equal(1.90, CalibrationSolver.ResidualFactor(6));
    }
}
