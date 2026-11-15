using RAWSimO.Live.Calibration;
using RAWSimO.Live.Geometry;
using Xunit;

namespace RAWSimO.Live.Tests;

public class GeometryAndCalibrationTests
{
    [Fact]
    public void CanvasRoundTripIsAccurate()
    {
        var transform = new CanvasTransform(3.0, 3.0);
        var random = new Random(1234);
        for (var i = 0; i < 1000; i++)
        {
            var map = new Point2D(random.NextDouble() * 3.0, random.NextDouble() * 3.0);
            var canvas = transform.MapToCanvas(map);
            var back = transform.CanvasToMap(canvas);
            Assert.True(Point2D.Distance(map, back) < 1e-9);
        }
    }

    [Fact]
    public void OrientationRoundTripIsAccurate()
    {
        var transform = new CanvasTransform(3.0, 3.0);
        var random = new Random(5678);
        for (var i = 0; i < 1000; i++)
        {
            var yaw = (random.NextDouble() * 2.0 - 1.0) * Math.PI;
            var degrees = transform.MapYawToCanvasDegrees(yaw);
            var back = transform.CanvasDegreesToMapYaw(degrees);
            Assert.True(Math.Abs(AngleMath.ShortestDifference(yaw, back)) < 1e-9);
        }
    }

    [Fact]
    public void CalibrationRecoversNoiselessTransform()
    {
        const double theta = 0.3;
        var tx = 0.45;
        var ty = -0.2;
        var q = System.Numerics.Complex.FromPolarCoordinates(1.0, theta);
        var mapPoints = new[]
        {
            new Point2D(0.375, 0.375),
            new Point2D(2.625, 0.375),
            new Point2D(2.625, 2.625),
            new Point2D(0.375, 2.625),
            new Point2D(1.5, 1.5),
            new Point2D(1.5, 0.375),
        };
        var points = mapPoints.Select(map =>
        {
            var z = new System.Numerics.Complex(map.X, map.Y);
            var w = (z - new System.Numerics.Complex(tx, ty)) / q;
            return new CalibrationPoint(map, new Point2D(w.Real, w.Imaginary));
        }).ToList();

        var result = CalibrationSolver.Fit(points, CalibrationModel.SE2);
        Assert.True(Math.Abs(result.Theta - theta) < 1e-12);
        Assert.True(Math.Abs(result.Tx - tx) < 1e-12);
        Assert.True(Math.Abs(result.Ty - ty) < 1e-12);
        Assert.True(result.ResidualRms < 1e-12);
        foreach (var point in points)
        {
            var predicted = result.ChassisToMap(point.Chassis);
            Assert.True(Point2D.Distance(predicted, point.Map) < 1e-12);
        }
    }

    [Fact]
    public void CalibrationRecoversMirroredTransform()
    {
        const double theta = -0.4;
        var tx = 0.2;
        var ty = 0.7;
        var q = System.Numerics.Complex.FromPolarCoordinates(1.0, theta);
        var mapPoints = new[]
        {
            new Point2D(0.375, 0.375),
            new Point2D(2.625, 0.375),
            new Point2D(2.625, 2.625),
            new Point2D(0.375, 2.625),
            new Point2D(1.5, 1.5),
            new Point2D(1.5, 0.375),
        };
        var points = mapPoints.Select(map =>
        {
            var z = new System.Numerics.Complex(map.X, map.Y);
            var w = (z - new System.Numerics.Complex(tx, ty)) / q;
            return new CalibrationPoint(map, new Point2D(w.Real, -w.Imaginary));
        }).ToList();

        var mirror = CalibrationSolver.Fit(points, CalibrationModel.SE2, mirror: true);
        Assert.True(mirror.Mirror);
        Assert.True(mirror.ResidualRms < 1e-12);
        foreach (var point in points)
            Assert.True(Point2D.Distance(mirror.ChassisToMap(point.Chassis), point.Map) < 1e-12);
    }

    [Fact]
    public void Sim2RecoversScale()
    {
        const double theta = 0.1;
        const double scale = 1.0021;
        var q = System.Numerics.Complex.FromPolarCoordinates(scale, theta);
        var mapPoints = new[]
        {
            new Point2D(0.375, 0.375),
            new Point2D(2.625, 0.375),
            new Point2D(2.625, 2.625),
            new Point2D(0.375, 2.625),
            new Point2D(1.5, 1.5),
            new Point2D(1.5, 0.375),
        };
        var points = mapPoints.Select(map =>
        {
            var z = new System.Numerics.Complex(map.X, map.Y);
            var w = z / q;
            return new CalibrationPoint(map, new Point2D(w.Real, w.Imaginary));
        }).ToList();

        var result = CalibrationSolver.Fit(points, CalibrationModel.Sim2);
        Assert.Equal(scale, result.ScaleDiag, 10);
        Assert.True(result.ResidualRms < 1e-12);
    }

    [Fact]
    public void LeaveOneOutFindsBadPoint()
    {
        var points = SixNoiselessPoints();
        var bad = points[2] with { Chassis = new Point2D(points[2].Chassis.X + 0.05, points[2].Chassis.Y) };
        points[2] = bad;
        var result = CalibrationSolver.Fit(points, CalibrationModel.SE2);
        var max = result.Points.OrderByDescending(p => p.LooError).First();
        Assert.Equal(points[2].Chassis, max.Chassis);
        Assert.True(max.LooError > 0.04);
    }

    [Fact]
    public void NoiseMatchesTechnicalPlanTable()
    {
        var random = new Random(20261031);
        var corners = new[]
        {
            new Point2D(0.375, 0.375),
            new Point2D(2.625, 0.375),
            new Point2D(2.625, 2.625),
            new Point2D(0.375, 2.625),
        };
        const double theta = 0.37;
        const double tx = 0.41, ty = -0.17;
        var q = System.Numerics.Complex.FromPolarCoordinates(1.0, theta);
        var targetPositions = new[]
        {
            new Point2D(1.5, 1.5),
            new Point2D(2.625, 2.625),
            new Point2D(3.0, 3.0),
        };
        var sums = new double[targetPositions.Length];
        const int trials = 600;
        for (var trial = 0; trial < trials; trial++)
        {
            var points = new List<CalibrationPoint>();
            foreach (var map in corners)
            {
                var z = new System.Numerics.Complex(map.X, map.Y);
                var ideal = (z - new System.Numerics.Complex(tx, ty)) / q;
                var noiseX = Gaussian(random) * 0.01;
                var noiseY = Gaussian(random) * 0.01;
                points.Add(new CalibrationPoint(map, new Point2D(ideal.Real + noiseX, ideal.Imaginary + noiseY)));
            }
            var fit = CalibrationSolver.Fit(points, CalibrationModel.SE2, sigma: 0.01);
            for (var i = 0; i < targetPositions.Length; i++)
            {
                var target = targetPositions[i];
                var targetChassis = (new System.Numerics.Complex(target.X, target.Y) - new System.Numerics.Complex(tx, ty)) / q;
                var predicted = fit.ChassisToMap(new Point2D(targetChassis.Real, targetChassis.Imaginary));
                var error = Point2D.Distance(predicted, target);
                sums[i] += error * error;
            }
        }
        var expected = new[] { 0.00703, 0.00864, 0.00971 };
        for (var i = 0; i < expected.Length; i++)
        {
            var rms = Math.Sqrt(sums[i] / trials);
            Assert.InRange(rms, expected[i] * 0.8, expected[i] * 1.2);
        }
    }

    private static List<CalibrationPoint> SixNoiselessPoints()
    {
        const double theta = 0.2;
        const double tx = 0.1, ty = 0.3;
        var q = System.Numerics.Complex.FromPolarCoordinates(1.0, theta);
        var mapPoints = new[]
        {
            new Point2D(0.375, 0.375),
            new Point2D(2.625, 0.375),
            new Point2D(2.625, 2.625),
            new Point2D(0.375, 2.625),
            new Point2D(1.5, 1.5),
            new Point2D(1.5, 0.375),
        };
        return mapPoints.Select(map =>
        {
            var z = new System.Numerics.Complex(map.X, map.Y);
            var w = (z - new System.Numerics.Complex(tx, ty)) / q;
            return new CalibrationPoint(map, new Point2D(w.Real, w.Imaginary));
        }).ToList();
    }

    private static double Gaussian(Random random)
    {
        var u1 = 1.0 - random.NextDouble();
        var u2 = 1.0 - random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
