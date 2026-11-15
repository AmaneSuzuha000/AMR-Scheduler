using RAWSimO.Live.Geometry;
using RAWSimO.Live.Planning;
using Xunit;

namespace RAWSimO.Live.Tests;

public class PlanningTests
{
    private static WaypointGraph LineGraph()
    {
        var graph = new WaypointGraph();
        graph.AddWaypoint(0, new Point2D(0.0, 0.0));
        graph.AddWaypoint(1, new Point2D(1.0, 0.0));
        graph.AddWaypoint(2, new Point2D(2.0, 0.0));
        graph.AddEdge(0, 1);
        graph.AddEdge(1, 2);
        return graph;
    }

    [Fact]
    public void FindsReachableGoalAndReportsTimeAndDistance()
    {
        var planner = new AStarPlanner(LineGraph(), new MotionParameters(1.0, 1.0, 1.0, 1.0), ExecutionMode.Stop);
        var result = planner.Plan(0, 0.0, 2);
        Assert.True(result.Success);
        Assert.Equal(2, result.Steps.Count);
        Assert.Equal(2.0, result.TotalDistanceMeters, 9);
        Assert.True(result.TotalTimeSeconds > 0.0);
    }

    [Fact]
    public void UnreachableGoalReturnsFailure()
    {
        var graph = new WaypointGraph();
        graph.AddWaypoint(0, new Point2D(0.0, 0.0));
        graph.AddWaypoint(1, new Point2D(1.0, 0.0));
        var planner = new AStarPlanner(graph);
        var result = planner.Plan(0, 0.0, 1);
        Assert.False(result.Success);
        Assert.NotEmpty(result.Message);
    }

    [Fact]
    public void StartEqualsGoalReturnsEmptySuccess()
    {
        var planner = new AStarPlanner(LineGraph());
        var result = planner.Plan(1, 0.7, 1);
        Assert.True(result.Success);
        Assert.Empty(result.Steps);
        Assert.Equal(0.0, result.TotalTimeSeconds);
    }

    [Fact]
    public void OneWayEdgeIsNotTraversableBackwards()
    {
        var graph = new WaypointGraph();
        graph.AddWaypoint(0, new Point2D(0.0, 0.0));
        graph.AddWaypoint(1, new Point2D(1.0, 0.0));
        graph.AddEdge(0, 1);
        var planner = new AStarPlanner(graph, new MotionParameters(1.0, 1.0, 1.0, 1.0), ExecutionMode.Stop);
        Assert.True(planner.Plan(0, 0.0, 1).Success);
        Assert.False(planner.Plan(1, 0.0, 0).Success);
    }

    [Fact]
    public void StopModeVisitsIntermediateWaypoint()
    {
        var plannerStop = new AStarPlanner(LineGraph(), new MotionParameters(1.0, 1.0, 1.0, 1.0), ExecutionMode.Stop);
        var plannerRun = new AStarPlanner(LineGraph(), new MotionParameters(1.0, 1.0, 1.0, 1.0), ExecutionMode.Run);
        Assert.Equal(2, plannerStop.Plan(0, 0.0, 2).Steps.Count);
        Assert.Single(plannerRun.Plan(0, 0.0, 2).Steps);
    }


    [Fact]
    public void RandomGraphsMatchDijkstraInStopMode()
    {
        var random = new Random(20261031);
        var motion = new MotionParameters(0.21, 0.5, 0.5, 5.5);
        for (var trial = 0; trial < 1000; trial++)
        {
            var count = random.Next(2, 9);
            var graph = new WaypointGraph();
            for (var i = 0; i < count; i++)
                graph.AddWaypoint(i, new Point2D(random.NextDouble() * 10.0, random.NextDouble() * 10.0));
            for (var from = 0; from < count; from++)
            {
                for (var to = 0; to < count; to++)
                {
                    if (from == to || random.NextDouble() > 0.25) continue;
                    graph.AddEdge(from, to);
                }
            }

            var planner = new AStarPlanner(graph, motion, ExecutionMode.Stop);
            var start = random.Next(count);
            var goal = random.Next(count);
            var yaw = random.NextDouble() * 2.0 * Math.PI - Math.PI;
            var plan = planner.Plan(start, yaw, goal, useHeuristic: false);
            var reference = DijkstraTime(graph, planner, start, yaw, goal);
            Assert.Equal(reference.HasValue, plan.Success);
            if (reference.HasValue)
                Assert.Equal(reference.Value, plan.TotalTimeSeconds, 7);
        }
    }

    private static double? DijkstraTime(WaypointGraph graph, AStarPlanner planner, int start, double yaw, int goal)
    {
        if (start == goal) return 0.0;
        var best = new Dictionary<(int Node, long Heading), double>();
        var queue = new PriorityQueue<(int Node, double Heading), double>();
        var startKey = (start, HeadingKey(yaw));
        best[startKey] = 0.0;
        queue.Enqueue((start, AngleMath.Normalize(yaw)), 0.0);
        while (queue.TryDequeue(out var state, out var cost))
        {
            var key = (state.Node, HeadingKey(state.Heading));
            if (best.TryGetValue(key, out var incumbent) && cost > incumbent + 1e-9) continue;
            if (state.Node == goal) return cost;
            foreach (var edge in graph.GetOutgoing(state.Node))
            {
                var turn = Math.Abs(AngleMath.ShortestDifference(state.Heading, edge.Angle));
                var next = cost + planner.TimeForTurn(turn) + planner.TimeForDistance(edge.Length);
                var nextKey = (edge.To, HeadingKey(edge.Angle));
                if (best.TryGetValue(nextKey, out var current) && next >= current - 1e-12) continue;
                best[nextKey] = next;
                queue.Enqueue((edge.To, AngleMath.Normalize(edge.Angle)), next);
            }
        }
        return null;
    }

    private static long HeadingKey(double heading) =>
        (long)Math.Round(AngleMath.Normalize(heading) * 1_000_000_000.0);

    [Fact]
    public void TimeForDistanceUsesTrapezoidalProfileAboveCruiseDistance()
    {
        var motion = new MotionParameters(1.0, 1.0, 1.0, 1.0);
        var planner = new AStarPlanner(LineGraph(), motion);
        const double distance = 10.0;
        var expected = distance / motion.MaxSpeed + motion.MaxSpeed / (2.0 * motion.Acceleration) + motion.MaxSpeed / (2.0 * motion.Deceleration);
        Assert.Equal(expected, planner.TimeForDistance(distance), 9);
    }
}
