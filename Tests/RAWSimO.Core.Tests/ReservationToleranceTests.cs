using RAWSimO.MultiAgentPathFinding.DataStructures;
using Xunit;

namespace RAWSimO.Core.Tests
{
    /// <summary>
    /// Regression tests for the time tolerance used by the reservation table.
    /// A fixed absolute tolerance stops separating "the same point in time" from "a different point in
    /// time" once the simulation clock is large enough, because the spacing of adjacent doubles grows
    /// with the magnitude while the tolerance does not.
    /// </summary>
    public class ReservationToleranceTests
    {
        /// <summary>
        /// The spacing of adjacent doubles at the given magnitude.
        /// </summary>
        private static double UlpAt(double value) => System.Math.BitIncrement(value) - value;

        /// <summary>
        /// The tolerance has to stay usable at every simulated time up to a long horizon. A tolerance
        /// below one ulp cannot be represented at all: adding it to the time value does not change it.
        /// </summary>
        [Theory]
        [InlineData(0.0)]
        [InlineData(1.0)]
        [InlineData(4500.0)]
        [InlineData(12062.417689081507)]
        [InlineData(28800.0)]
        [InlineData(200000.0)]
        public void ToleranceIsRepresentableAtTheGivenTime(double time)
        {
            var tolerance = ReservationTable.ToleranceAt(time);

            Assert.True(tolerance > 0.0);
            //the whole point: the tolerance must be able to move the time value by at least one step
            Assert.True(time + tolerance > time, "tolerance of " + tolerance + " vanishes at t=" + time);
        }

        /// <summary>
        /// The tolerance must absorb the rounding noise of the interval arithmetic, but it must not grow
        /// so large that it swallows differences a user would call real. The bound is deliberately loose
        /// (a microsecond), the point is that it stays a rounding tolerance and not a scheduling one.
        /// </summary>
        [Theory]
        [InlineData(0.0)]
        [InlineData(12062.417689081507)]
        [InlineData(28800.0)]
        [InlineData(200000.0)]
        public void ToleranceStaysFarBelowAnyMeaningfulTimeDifference(double time)
        {
            var tolerance = ReservationTable.ToleranceAt(time);

            Assert.True(tolerance <= 1e-6, "tolerance grew to " + tolerance + " at t=" + time);
        }

        /// <summary>
        /// The case that killed a CBS run of the planning comparison: a reservation spanning exactly one
        /// ulp at a large simulation time. Such an interval is below the resolution of the simulation
        /// clock, so it has to be dropped as senseless instead of being inserted - inserting it is what
        /// used to raise an exception, because the intersection test could not absorb the step either.
        /// The test asserts the precise thing that matters: adding it must not throw.
        /// </summary>
        [Fact]
        public void SingleUlpIntervalIsDroppedInsteadOfThrowingAtALargeSimulationTime()
        {
            const double start = 12062.417689081507;
            var end = System.Math.BitIncrement(start);

            //the width of such an interval is exactly one ulp, i.e. larger than the old absolute tolerance,
            //which is why the old "no senseless intervals" test let it through
            Assert.Equal(UlpAt(start), end - start);
            Assert.True(end - start > 1e-12, "the regression only reproduces when the width exceeds the old tolerance");

            var tree = new DisjointIntervalTree(false, false);

            //the crash: this call used to throw IntervalIntersectionException
            tree.Add(start, end);

            //and the interval is below the clock resolution, so it must have been dropped
            Assert.Equal(0, tree.Count);
        }

        /// <summary>
        /// Reservations into the indefinite future end at +infinity. The tolerance must stay a usable number
        /// there as well: the ulp of infinity is NaN and would poison every comparison it enters.
        /// </summary>
        [Theory]
        [InlineData(double.PositiveInfinity)]
        [InlineData(double.NegativeInfinity)]
        [InlineData(double.NaN)]
        public void ToleranceIsUsableForNonFiniteTimes(double time)
        {
            var tolerance = ReservationTable.ToleranceAt(time);

            Assert.False(double.IsNaN(tolerance));
            Assert.Equal(ReservationTable.TOLERANCE, tolerance);
        }

        /// <summary>
        /// Neighbouring reservations that only touch must still be accepted - the tolerance exists to let
        /// consecutive intervals share a boundary without being reported as an intersection.
        /// </summary>
        [Fact]
        public void TouchingIntervalsAreAcceptedAtALargeSimulationTime()
        {
            const double boundary = 28800.0;

            var tree = new DisjointIntervalTree(false, false);
            tree.Add(boundary - 10.0, boundary);
            tree.Add(boundary, boundary + 10.0);

            Assert.Equal(2, tree.Count);
        }

        /// <summary>
        /// A genuine overlap has to stay a genuine overlap: the tolerance must not turn a real conflict
        /// into a free interval, otherwise the reservation table would hand out the same node twice.
        /// </summary>
        [Fact]
        public void RealOverlapIsStillDetectedAtALargeSimulationTime()
        {
            const double boundary = 28800.0;

            var tree = new DisjointIntervalTree(false, false);
            tree.Add(boundary, boundary + 10.0);

            Assert.Throws<DisjointIntervalTree.IntervalIntersectionException>(() => tree.Add(boundary + 0.005, boundary + 20.0));
        }
    }
}
