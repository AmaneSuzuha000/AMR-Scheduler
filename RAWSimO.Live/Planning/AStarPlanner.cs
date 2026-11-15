using RAWSimO.Live.Geometry;

namespace RAWSimO.Live.Planning;

/// <summary>A waypoint in the map frame.</summary>
public readonly record struct Waypoint(int Id, Point2D Position);

/// <summary>A directed graph edge. <see cref="Angle"/> is the travel direction in radians.</summary>
public readonly record struct GraphEdge(int From, int To, double Length, double Angle);

/// <summary>Directed waypoint graph used by the single-vehicle planner.</summary>
public sealed class WaypointGraph
{
    private readonly Dictionary<int, Waypoint> _waypoints = new();
    private readonly Dictionary<int, List<GraphEdge>> _outgoing = new();

    /// <summary>All waypoints in insertion order.</summary>
    public IReadOnlyList<Waypoint> Waypoints => _waypoints.Values.ToList();

    /// <summary>Number of waypoints.</summary>
    public int Count => _waypoints.Count;

    /// <summary>Adds a waypoint. Throws when the id already exists.</summary>
    public void AddWaypoint(int id, Point2D position)
    {
        if (_waypoints.ContainsKey(id)) throw new ArgumentException($"Waypoint {id} already exists.", nameof(id));
        if (!double.IsFinite(position.X) || !double.IsFinite(position.Y))
            throw new ArgumentException("Waypoint coordinates must be finite.", nameof(position));
        _waypoints.Add(id, new Waypoint(id, position));
        _outgoing.Add(id, new List<GraphEdge>());
    }

    /// <summary>Adds a directed edge, computing its length when omitted.</summary>
    public void AddEdge(int from, int to, double? length = null)
    {
        if (!_waypoints.TryGetValue(from, out var a)) throw new ArgumentException($"Unknown waypoint {from}.", nameof(from));
        if (!_waypoints.TryGetValue(to, out var b)) throw new ArgumentException($"Unknown waypoint {to}.", nameof(to));
        if (from == to) throw new ArgumentException("Self edges are not supported.", nameof(to));
        var dx = b.Position.X - a.Position.X;
        var dy = b.Position.Y - a.Position.Y;
        var actual = Math.Sqrt(dx * dx + dy * dy);
        var value = length ?? actual;
        if (!double.IsFinite(value) || value <= 0) throw new ArgumentOutOfRangeException(nameof(length));
        _outgoing[from].Add(new GraphEdge(from, to, value, Math.Atan2(dy, dx)));
    }

    /// <summary>Returns the outgoing edges of a waypoint (empty when unknown).</summary>
    public IReadOnlyList<GraphEdge> GetOutgoing(int id) =>
        _outgoing.TryGetValue(id, out var edges) ? edges : Array.Empty<GraphEdge>();

    /// <summary>Whether a waypoint id exists.</summary>
    public bool Contains(int id) => _waypoints.ContainsKey(id);

    /// <summary>Returns a waypoint or throws when it does not exist.</summary>
    public Waypoint GetWaypoint(int id) =>
        _waypoints.TryGetValue(id, out var waypoint) ? waypoint : throw new KeyNotFoundException($"Unknown waypoint {id}.");
}

/// <summary>Motion parameters of the vehicle tracker (T3.1 measured values).</summary>
public sealed record MotionParameters(
    double MaxSpeed = 0.21,
    double Acceleration = 0.5,
    double Deceleration = 0.5,
    double TurnSpeedSeconds = 5.5)
{
    /// <summary>Angular speed for an in-place turn, radians per second.</summary>
    public double TurnRateRadiansPerSecond => 2.0 * Math.PI / TurnSpeedSeconds;

    /// <summary>Demo robot parameters from the technical plan.</summary>
    public static MotionParameters Demo => new();

    /// <summary>Validates the parameter set.</summary>
    public void Validate()
    {
        if (!double.IsFinite(MaxSpeed) || MaxSpeed <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSpeed));
        if (!double.IsFinite(Acceleration) || Acceleration <= 0) throw new ArgumentOutOfRangeException(nameof(Acceleration));
        if (!double.IsFinite(Deceleration) || Deceleration <= 0) throw new ArgumentOutOfRangeException(nameof(Deceleration));
        if (!double.IsFinite(TurnSpeedSeconds) || TurnSpeedSeconds <= 0) throw new ArgumentOutOfRangeException(nameof(TurnSpeedSeconds));
    }
}

/// <summary>Whether every waypoint is stopped at, or only direction changes.</summary>
public enum ExecutionMode
{
    /// <summary>Do not stop at collinear intermediate waypoints (the normal tracker behaviour).</summary>
    Run,
    /// <summary>Stop at every waypoint.</summary>
    Stop,
}

/// <summary>One segment of a planned path.</summary>
public sealed record PlanStep(
    int From,
    int To,
    double Heading,
    double TurnRadians,
    double DistanceMeters,
    double TimeSeconds);

/// <summary>Result of a planner query.</summary>
public sealed record PlanResult(
    bool Success,
    IReadOnlyList<PlanStep> Steps,
    double TotalTimeSeconds,
    double TotalDistanceMeters,
    int TurnCount,
    int ExpandedStates,
    string? Message = null)
{
    /// <summary>Creates a failure result.</summary>
    public static PlanResult Failure(string message, int expandedStates = 0) =>
        new(false, Array.Empty<PlanStep>(), 0, 0, 0, expandedStates, message);
}

/// <summary>
/// Time-optimal A* for one vehicle. A state is (waypoint, heading); an
/// expansion is an in-place turn followed by a stop-to-stop straight run.
/// The heuristic is straight-line distance divided by maximum speed.
/// </summary>
/// <remarks>
/// NOT a scheduler. Route planning belongs to Python (amr_sched), which owns task
/// allocation, global routing and spatiotemporal reservation; the console asks for a route
/// over the wire (RouteRequestMessage) and draws the answer. This class therefore has no
/// production caller any more - the GUI's execute button used to build its own route here,
/// which put a second decision core inside the display layer. It is kept only because the
/// single-vehicle motion tests use it, and it must never be wired back into the console.
/// </remarks>
public sealed class AStarPlanner
{
    private readonly WaypointGraph _graph;
    private readonly MotionParameters _motion;
    private readonly ExecutionMode _mode;

    /// <summary>Creates a planner for a graph and motion model.</summary>
    public AStarPlanner(WaypointGraph graph, MotionParameters? motion = null, ExecutionMode mode = ExecutionMode.Run)
    {
        _graph = graph ?? throw new ArgumentNullException(nameof(graph));
        _motion = motion ?? MotionParameters.Demo;
        _motion.Validate();
        _mode = mode;
    }

    /// <summary>Exact stop-to-stop travel time for a straight segment.</summary>
    public double TimeForDistance(double distance)
    {
        if (!double.IsFinite(distance) || distance < 0) throw new ArgumentOutOfRangeException(nameof(distance));
        if (distance == 0) return 0;
        var v = _motion.MaxSpeed;
        var a = _motion.Acceleration;
        var b = _motion.Deceleration;
        var fullAccelerationDistance = v * v / (2.0 * a);
        var fullBrakingDistance = v * v / (2.0 * b);
        if (distance >= fullAccelerationDistance + fullBrakingDistance)
            return distance / v + v / (2.0 * a) + v / (2.0 * b);
        // Triangular speed profile: the vehicle never reaches v.
        return Math.Sqrt(distance / (a / 2.0 + a * a / (2.0 * b))) +
               Math.Sqrt(distance / (b / 2.0 + b * b / (2.0 * a)));
    }

    /// <summary>Time for an in-place turn of the given absolute angle.</summary>
    public double TimeForTurn(double radians) =>
        Math.Abs(radians) / _motion.TurnRateRadiansPerSecond;

    /// <summary>Plans a path. When start equals goal, succeeds with an empty path.</summary>
    public PlanResult Plan(int startNode, double startYaw, int goalNode, bool useHeuristic = true)
    {
        if (!_graph.Contains(startNode)) throw new ArgumentException($"Unknown start waypoint {startNode}.", nameof(startNode));
        if (!_graph.Contains(goalNode)) throw new ArgumentException($"Unknown goal waypoint {goalNode}.", nameof(goalNode));

        var start = new SearchState(startNode, AngleMath.Normalize(startYaw));
        var startKey = Key(start);
        if (startNode == goalNode)
            return new PlanResult(true, Array.Empty<PlanStep>(), 0, 0, 0, 0);

        var best = new Dictionary<StateKey, (double G, int Turns)> { [startKey] = (0.0, 0) };
        var parents = new Dictionary<StateKey, ParentInfo>();
        var queue = new PriorityQueue<SearchNode, (double F, int Turns)>();
        queue.Enqueue(new SearchNode(start, 0.0, 0), (Heuristic(start.Node, goalNode, useHeuristic), 0));

        var expanded = 0;
        while (queue.TryDequeue(out var node, out _))
        {
            var key = Key(node.State);
            if (!best.TryGetValue(key, out var incumbent) ||
                node.G > incumbent.G + 1e-9 ||
                (Math.Abs(node.G - incumbent.G) <= 1e-9 && node.Turns > incumbent.Turns))
                continue;

            expanded++;
            if (node.State.Node == goalNode)
                return Reconstruct(key, best, parents, expanded);

            foreach (var edge in _graph.GetOutgoing(node.State.Node))
            {
                var turn = Math.Abs(AngleMath.ShortestDifference(node.State.Heading, edge.Angle));
                var turnTime = TimeForTurn(turn);
                if (_mode == ExecutionMode.Stop)
                {
                    AddCandidate(edge.To, edge.Angle, turn, edge.Length,
                        turnTime + TimeForDistance(edge.Length), node, best, parents, queue, goalNode, useHeuristic);
                    continue;
                }

                var visited = new HashSet<int> { node.State.Node };
                var cursor = edge;
                var distance = 0.0;
                while (true)
                {
                    if (!visited.Add(cursor.To)) break;
                    distance += cursor.Length;
                    AddCandidate(cursor.To, edge.Angle, turn, distance,
                        turnTime + TimeForDistance(distance), node, best, parents, queue, goalNode, useHeuristic);
                    if (cursor.To == goalNode) break;
                    var collinear = _graph.GetOutgoing(cursor.To)
                        .Where(next => Math.Abs(AngleMath.ShortestDifference(next.Angle, edge.Angle)) < 1e-9)
                        .ToList();
                    if (collinear.Count != 1) break;
                    cursor = collinear[0];
                }
            }
        }

        return PlanResult.Failure("No path to the goal.", expanded);
    }

    private void AddCandidate(
        int toNode,
        double heading,
        double turnRadians,
        double distance,
        double stepTime,
        SearchNode from,
        Dictionary<StateKey, (double G, int Turns)> best,
        Dictionary<StateKey, ParentInfo> parents,
        PriorityQueue<SearchNode, (double F, int Turns)> queue,
        int goalNode,
        bool useHeuristic)
    {
        var next = new SearchState(toNode, AngleMath.Normalize(heading));
        var key = Key(next);
        var newG = from.G + stepTime;
        var newTurns = from.Turns + (turnRadians > 1e-9 ? 1 : 0);
        if (best.TryGetValue(key, out var incumbent))
        {
            if (newG > incumbent.G + 1e-9) return;
            if (Math.Abs(newG - incumbent.G) <= 1e-9 && newTurns >= incumbent.Turns) return;
        }

        best[key] = (newG, newTurns);
        parents[key] = new ParentInfo(Key(from.State),
            new PlanStep(from.State.Node, toNode, next.Heading, turnRadians, distance, stepTime));
        var f = newG + Heuristic(toNode, goalNode, useHeuristic);
        queue.Enqueue(new SearchNode(next, newG, newTurns), (f, newTurns));
    }

    private PlanResult Reconstruct(
        StateKey goalKey,
        Dictionary<StateKey, (double G, int Turns)> best,
        Dictionary<StateKey, ParentInfo> parents,
        int expanded)
    {
        var steps = new List<PlanStep>();
        var key = goalKey;
        while (parents.TryGetValue(key, out var parent))
        {
            steps.Add(parent.Step);
            key = parent.Previous;
        }
        steps.Reverse();
        var totalTime = best[goalKey].G;
        var totalDistance = steps.Sum(step => step.DistanceMeters);
        var turns = steps.Count(step => step.TurnRadians > 1e-9);
        return new PlanResult(true, steps, totalTime, totalDistance, turns, expanded);
    }

    private double Heuristic(int node, int goalNode, bool useHeuristic)
    {
        if (!useHeuristic) return 0.0;
        var a = _graph.GetWaypoint(node).Position;
        var b = _graph.GetWaypoint(goalNode).Position;
        return Point2D.Distance(a, b) / _motion.MaxSpeed;
    }

    private static StateKey Key(SearchState state) =>
        new(state.Node, (long)Math.Round(AngleMath.Normalize(state.Heading) * 1_000_000_000.0));

    private readonly record struct SearchState(int Node, double Heading);

    private readonly record struct StateKey(int Node, long HeadingKey);

    private readonly record struct ParentInfo(StateKey Previous, PlanStep Step);

    private sealed record SearchNode(SearchState State, double G, int Turns);
}
