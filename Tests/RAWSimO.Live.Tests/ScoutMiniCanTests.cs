using System;
using RAWSimO.Live.Drivers;
using Xunit;

#nullable enable

namespace RAWSimO.Live.Tests;

/// <summary>
/// P3-A decoding and integration tests.
///
/// Every expected value here is derived from the published AgileX frame layouts and from
/// closed-form kinematics, never from a recorded run. Nothing in this file is a real-vehicle
/// measurement and none of its output may be presented as one.
/// </summary>
public class ScoutMiniCanTests
{
    [Fact]
    public void OdometryFrameIdMatchesOfficialHeader()
    {
        // agilex_protocol_v2.h line 76
        Assert.Equal(0x311u, ScoutMiniCanFrames.OdometryFrameId);
        Assert.Equal(0x221u, ScoutMiniCanFrames.MotionStateFrameId);
        Assert.Equal(0x211u, ScoutMiniCanFrames.SystemStateFrameId);
        Assert.Equal(500_000, ScoutMiniCanFrames.BaudRate);
    }

    [Fact]
    public void DecodesBigEndianWheelCounters()
    {
        // struct32_t is { msb, high_byte, low_byte, lsb } on the wire.
        var data = new byte[]
        {
            0x00, 0x00, 0x01, 0x00,   // left  = 256
            0xFF, 0xFF, 0xFF, 0xFF,   // right = -1
        };

        var counters = ScoutMiniCanFrames.DecodeOdometry(data);

        Assert.Equal(256, counters.Left);
        Assert.Equal(-1, counters.Right);
    }

    [Fact]
    public void DecodesSignedCountersBothDirections()
    {
        var forward = ScoutMiniCanFrames.DecodeOdometry(new byte[] { 0, 0, 0, 10, 0, 0, 0, 10 });
        var backward = ScoutMiniCanFrames.DecodeOdometry(new byte[] { 0xFF, 0xFF, 0xFF, 0xF6, 0xFF, 0xFF, 0xFF, 0xF6 });

        Assert.Equal(10, forward.Left);
        Assert.Equal(-10, backward.Left);
    }

    [Fact]
    public void RejectsShortOdometryPayload()
    {
        Assert.Throws<ArgumentException>(() => ScoutMiniCanFrames.DecodeOdometry(new byte[4]));
    }

    [Fact]
    public void DecodesMotionStateWithMilliScaling()
    {
        // int16 little-endian-in-struct but big-endian on the wire, scaled by 1000.
        var data = new byte[]
        {
            0x00, 0xC8,   // 200 -> 0.200 m/s
            0xFF, 0x38,   // -200 -> -0.200 rad/s
            0, 0, 0, 0,
        };

        var (linear, angular) = ScoutMiniCanFrames.DecodeMotionState(data);

        Assert.Equal(0.200, linear, 6);
        Assert.Equal(-0.200, angular, 6);
    }

    [Fact]
    public void FirstFrameOnlyEstablishesBaseline()
    {
        var odometry = new WheelOdometry(ScoutMiniGeometry.Default);
        var moved = odometry.Update(new WheelCounters(1000, 1000));

        Assert.Equal(0.0, moved);
        Assert.True(odometry.HasBaseline);
        Assert.Equal(0.0, odometry.X);
        Assert.Equal(0.0, odometry.Y);
    }

    [Fact]
    public void EqualWheelCountsDriveStraightAlongX()
    {
        var geometry = ScoutMiniGeometry.Default;
        var odometry = new WheelOdometry(geometry);
        odometry.Update(new WheelCounters(0, 0));

        var counts = geometry.CountsPerRevolution;   // exactly one wheel revolution
        odometry.Update(new WheelCounters(counts, counts));

        var expected = 2.0 * Math.PI * geometry.WheelRadius;
        Assert.Equal(expected, odometry.X, 9);
        Assert.Equal(0.0, odometry.Y, 9);
        Assert.Equal(0.0, odometry.Yaw, 9);
        Assert.Equal(expected, odometry.TravelledDistance, 9);
    }

    [Fact]
    public void OppositeWheelCountsRotateInPlace()
    {
        var geometry = ScoutMiniGeometry.Default;
        var odometry = new WheelOdometry(geometry);
        odometry.Update(new WheelCounters(0, 0));

        var counts = geometry.CountsPerRevolution;
        odometry.Update(new WheelCounters(-counts, counts));

        // d_yaw = (right - left) * metresPerCount / track
        var metresPerCount = 2.0 * Math.PI * geometry.WheelRadius / geometry.CountsPerRevolution;
        var expectedYaw = 2.0 * counts * metresPerCount / geometry.TrackWidth;
        Assert.Equal(WheelOdometry.WrapAngle(expectedYaw), odometry.Yaw, 9);
        Assert.Equal(0.0, odometry.X, 9);
        Assert.Equal(0.0, odometry.Y, 9);
    }

    [Fact]
    public void QuarterCircleMatchesClosedForm()
    {
        // A differential drive tracing a circle of radius R: v = w * R, and the wheels differ by
        // w * track. Drive exactly a quarter turn and compare against the analytic endpoint.
        var geometry = ScoutMiniGeometry.Default;
        const double radius = 1.0;
        var odometry = new WheelOdometry(geometry);
        odometry.Update(new WheelCounters(0, 0));

        var metresPerCount = 2.0 * Math.PI * geometry.WheelRadius / geometry.CountsPerRevolution;
        var targetYaw = Math.PI / 2.0;
        // w = d_yaw per step; centre travels w*R; wheels travel w*(R -+ track/2)
        var countsCentre = targetYaw * radius / metresPerCount;
        var countsDelta = targetYaw * geometry.TrackWidth / 2.0 / metresPerCount;
        var left = (int)Math.Round(countsCentre - countsDelta);
        var right = (int)Math.Round(countsCentre + countsDelta);

        // Feed it in small steps so the midpoint integration is exercised.
        const int steps = 200;
        for (var i = 1; i <= steps; i++)
            odometry.Update(new WheelCounters(left * i / steps, right * i / steps));

        Assert.Equal(radius, odometry.X, 2);
        Assert.Equal(radius, odometry.Y, 2);
        Assert.Equal(targetYaw, odometry.Yaw, 2);
    }

    [Fact]
    public void CounterWrapIsIntegrated()
    {
        var geometry = ScoutMiniGeometry.Default;
        var odometry = new WheelOdometry(geometry);
        odometry.Update(new WheelCounters(int.MaxValue - 5, int.MaxValue - 5));
        odometry.Update(new WheelCounters(unchecked(int.MaxValue - 5 + 10), unchecked(int.MaxValue - 5 + 10)));
        // int.MaxValue - 5 + 10 wraps past the 32-bit boundary; the increment must still read as 10.

        var expected = 10 * (2.0 * Math.PI * geometry.WheelRadius / geometry.CountsPerRevolution);
        Assert.Equal(expected, odometry.X, 9);
        Assert.Equal(0, odometry.DroppedFrames);
    }

    [Fact]
    public void ImplausibleJumpIsDropped()
    {
        var odometry = new WheelOdometry(ScoutMiniGeometry.Default);
        odometry.Update(new WheelCounters(0, 0));
        odometry.Update(new WheelCounters(1_000_000, -1_000_000));

        Assert.Equal(1, odometry.DroppedFrames);
        Assert.Equal(0.0, odometry.X);
        Assert.Equal(0.0, odometry.Y);
    }

    [Fact]
    public void ResetSeatsTheIntegratorAtTheGivenPose()
    {
        var odometry = new WheelOdometry(ScoutMiniGeometry.Default);
        odometry.Update(new WheelCounters(0, 0));
        odometry.Update(new WheelCounters(1000, 1000));
        odometry.Reset(2.0, 3.0, 0.5);

        Assert.Equal(2.0, odometry.X);
        Assert.Equal(3.0, odometry.Y);
        Assert.Equal(0.5, odometry.Yaw);
        Assert.False(odometry.HasBaseline);
        Assert.Equal(0.0, odometry.TravelledDistance);
    }

    [Theory]
    [InlineData(0.0, 0.0)]
    [InlineData(Math.PI, Math.PI)]
    [InlineData(-Math.PI, -Math.PI)]
    [InlineData(3.0 * Math.PI, Math.PI)]
    [InlineData(-3.0 * Math.PI, -Math.PI)]  // convention is [-pi, pi)
    public void WrapAngleStaysInRange(double input, double expected)
    {
        Assert.Equal(expected, WheelOdometry.WrapAngle(input), 9);
    }

    [Fact]
    public void GeometryRejectsNonPhysicalValues()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new WheelOdometry(new ScoutMiniGeometry(0, 0.3645, 1024)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new WheelOdometry(new ScoutMiniGeometry(0.0955, 0, 1024)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new WheelOdometry(new ScoutMiniGeometry(0.0955, 0.3645, 0)));
    }

    // ---------- 0x211 system state (manual Table 3.1 / 3.2) ----------

    [Fact]
    public void DecodesSystemStateVoltageAsTenthsOfVolt()
    {
        // 29.2 V fully charged -> raw 292 = 0x0124.
        // byte[0]=normal, byte[1]=remote control, byte[2..3]=voltage, byte[5]=no failures.
        var data = new byte[] { 0x00, 0x03, 0x01, 0x24, 0x00, 0x00, 0x00, 0x07 };

        var st = ScoutMiniCanFrames.DecodeSystemState(data);

        Assert.Equal(29.2, st.BatteryVoltage, 3);
        Assert.Equal("remote_control", st.ControlModeName);
        Assert.False(st.HasSystemError);
        Assert.Equal(0, st.Failures);
        Assert.Equal(7, st.Counter);
    }

    [Fact]
    public void DecodesSystemStateErrorAndCanMode()
    {
        var data = new byte[] { 0x02, 0x01, 0x00, 0xCD, 0x00, 0x00, 0x00, 0x00 }; // 20.5 V

        var st = ScoutMiniCanFrames.DecodeSystemState(data);

        Assert.True(st.HasSystemError);
        Assert.Equal("can_command", st.ControlModeName);
        Assert.Equal(20.5, st.BatteryVoltage, 3);
    }

    [Fact]
    public void DecodesFailureBitsPerManualTable32()
    {
        // bit0 low-voltage failure, bit2 remote signal lost, bit3 drive 1 failure -> 0b0000_1101
        var data = new byte[] { 0x00, 0x03, 0x00, 0xE6, 0x00, 0b0000_1101, 0x00, 0x00 };

        var st = ScoutMiniCanFrames.DecodeSystemState(data);
        var f = (SystemFailures)st.Failures;

        Assert.True(f.HasFlag(SystemFailures.LowVoltageFailure));
        Assert.True(f.HasFlag(SystemFailures.RemoteSignalLost));
        Assert.True(f.HasFlag(SystemFailures.Drive1Failure));
        Assert.False(f.HasFlag(SystemFailures.LowVoltageWarning));
        Assert.False(f.HasFlag(SystemFailures.Drive4Failure));
    }

    [Fact]
    public void RejectsShortSystemStatePayload()
    {
        Assert.Throws<ArgumentException>(
            () => ScoutMiniCanFrames.DecodeSystemState(new byte[] { 0, 1, 2 }));
    }
}
