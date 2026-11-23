using System;
using RAWSimO.Live.Calibration;
using Xunit;

namespace RAWSimO.Live.Tests
{
    public class FrameCheckTests
    {
        [Theory]
        [InlineData("map", AnnouncedFrame.Map)]
        [InlineData("MAP", AnnouncedFrame.Map)]
        [InlineData(" odom ", AnnouncedFrame.Odom)]
        [InlineData("chassis", AnnouncedFrame.Chassis)]
        [InlineData("sim", AnnouncedFrame.Sim)]
        [InlineData("", AnnouncedFrame.Unknown)]
        [InlineData(null, AnnouncedFrame.Unknown)]
        [InlineData("world", AnnouncedFrame.Unknown)]
        public void ParsesFrames(string? input, AnnouncedFrame expected) =>
            Assert.Equal(expected, FrameCheck.Parse(input));

        [Theory]
        [InlineData(AnnouncedFrame.Map, true)]
        [InlineData(AnnouncedFrame.Sim, true)]
        [InlineData(AnnouncedFrame.Chassis, false)]
        [InlineData(AnnouncedFrame.Odom, false)]
        [InlineData(AnnouncedFrame.Unknown, false)]
        public void OnlyMapAndSimAreDrawableWithoutCalibration(AnnouncedFrame frame, bool expected) =>
            Assert.Equal(expected, FrameCheck.IsDirectlyDrawable(frame));

        [Fact]
        public void MapFrameNeverNeedsCalibration() =>
            Assert.Null(FrameCheck.Describe(AnnouncedFrame.Map, hasCalibration: false));

        [Fact]
        public void SimFrameNeverNeedsCalibration() =>
            Assert.Null(FrameCheck.Describe(AnnouncedFrame.Sim, hasCalibration: false));

        [Fact]
        public void ChassisWithoutCalibrationIsRejected()
        {
            var msg = FrameCheck.Describe(AnnouncedFrame.Chassis, hasCalibration: false);
            Assert.NotNull(msg);
            Assert.Contains("chassis", msg);
        }

        [Fact]
        public void OdomWithoutCalibrationIsRejected()
        {
            var msg = FrameCheck.Describe(AnnouncedFrame.Odom, hasCalibration: false);
            Assert.NotNull(msg);
            Assert.Contains("odom", msg);
        }

        [Fact]
        public void UnknownWithoutCalibrationIsRejected() =>
            Assert.NotNull(FrameCheck.Describe(AnnouncedFrame.Unknown, hasCalibration: false));

        [Theory]
        [InlineData(AnnouncedFrame.Chassis)]
        [InlineData(AnnouncedFrame.Odom)]
        [InlineData(AnnouncedFrame.Unknown)]
        public void CalibrationMakesEveryFrameDrawable(AnnouncedFrame frame) =>
            Assert.Null(FrameCheck.Describe(frame, hasCalibration: true));
    }
}
