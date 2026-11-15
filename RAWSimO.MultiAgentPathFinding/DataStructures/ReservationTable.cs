using RAWSimO.MultiAgentPathFinding.Elements;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RAWSimO.MultiAgentPathFinding.DataStructures
{
    /// <summary>
    /// Reservation Table
    /// </summary>
    public class ReservationTable
    {
        /// <summary>
        /// The tolerance of time. This is the absolute floor of the tolerance; for large simulation
        /// times it is superseded by the representable resolution, see <see cref="ToleranceAt"/>.
        /// </summary>
        public static double TOLERANCE = 0.000000000001;

        /// <summary>
        /// The tolerance to use when comparing two points in time of the given magnitude.
        /// <para>
        /// A fixed absolute tolerance silently stops working once the simulation clock grows large
        /// enough, because the spacing of adjacent doubles grows with the magnitude. A double near
        /// 12,000s already has an ulp of 1.8e-12, i.e. larger than <see cref="TOLERANCE"/> itself, so
        /// <c>start + TOLERANCE == start</c> and a difference of a single ulp can neither be absorbed
        /// by the tolerance nor be discarded as zero. That is what made a reservation of exactly one
        /// ulp width pass the "no senseless intervals" test in <see cref="DisjointIntervalTree.Add"/>
        /// and then be rejected by the intersection test, which escalated into an
        /// <see cref="IntervalIntersectionException"/> that killed the whole simulation.
        /// </para>
        /// <para>
        /// Scaling with a few ulps instead keeps the tolerance meaning "the same point in time" at
        /// every magnitude: it absorbs rounding noise of the interval arithmetic while staying far
        /// below any physically meaningful difference (four ulps at 12,000s is about 7ns).
        /// </para>
        /// </summary>
        /// <param name="time">The point in time the comparison happens at.</param>
        /// <returns>The tolerance to use at that magnitude.</returns>
        public static double ToleranceAt(double time)
        {
            var magnitude = Math.Abs(time);
            //reservations into the indefinite future are created with an end of +infinity; there is no meaningful
            //ulp at infinity (it evaluates to NaN, and Math.Max propagates NaN), so fall back to the floor value
            if (double.IsNaN(magnitude) || double.IsInfinity(magnitude))
                return TOLERANCE;
            //spacing of adjacent doubles at this magnitude; BitIncrement of 0 stays denormal, so the floor below wins
            var ulp = Math.BitIncrement(magnitude) - magnitude;
            return Math.Max(TOLERANCE, 4.0 * ulp);
        }

        /// <summary>
        /// The graph
        /// </summary>
        protected Graph _graph;

        /// <summary>
        /// The intervals
        /// </summary>
        protected DisjointIntervalTree[] _intervallTrees;

        /// <summary>
        /// The touched nodes
        /// </summary>
        protected HashSet<int> _touchedNodes;

        /// <summary>
        /// The touched nodes
        /// </summary>
        protected bool _storeAgentIds;

        /// <summary>
        /// The touched nodes
        /// </summary>
        protected bool _storePrios;

        /// <summary>
        /// Initializes a new instance of the <see cref="ReservationTable" /> class.
        /// </summary>
        /// <param name="graph">The graph.</param>
        /// <param name="fastClear">if set to <c>true</c> clear is faster, otherwise memory consumption is lower.</param>
        public ReservationTable(Graph graph, bool fastClear = false, bool storeAgentIds = false, bool storePrios = false)
        {
            if (_storePrios && !_storeAgentIds)
                throw new NotSupportedException("Store priorities implies store agent ids!");

            _graph = graph;
            _storeAgentIds = storeAgentIds;
            _storePrios = storePrios;
            _intervallTrees = new DisjointIntervalTree[graph.NodeCount];
            if (fastClear)
                _touchedNodes = new HashSet<int>();
        }

        /// <summary>
        /// Clears this instance.
        /// </summary>
        public void Clear()
        {
            if (_touchedNodes != null)
            {
                //fast clear
                foreach (var node in _touchedNodes)
                    if (_intervallTrees[node] != null)
                        _intervallTrees[node].Clear();
                _touchedNodes.Clear();
            }
            else
            {
                //slow clear
                foreach (var intervall in _intervallTrees)
                    if (intervall != null)
                        intervall.Clear();
            }
        }

        /// <summary>
        /// Clears this node.
        /// </summary>
        public void Clear(int node)
        {
            _intervallTrees[node].Clear();
            if (_touchedNodes != null && _touchedNodes.Contains(node))
                _touchedNodes.Remove(node);

        }

        /// <summary>
        /// Adds the specified interval.
        /// </summary>
        /// <param name="node">The node.</param>
        /// <param name="Start">The start.</param>
        /// <param name="End">The end.</param>
        public void Add(int node, double Start, double End, int agentId = -1, int prio = -1)
        {
            //lasy initialization
            if (_intervallTrees[node] == null)
                _intervallTrees[node] = new DisjointIntervalTree(_storeAgentIds, _storePrios);

            if (_touchedNodes != null)
                _touchedNodes.Add(node);

            _intervallTrees[node].Add(Start, End, agentId, prio);
        }

        /// <summary>
        /// Adds the specified interval.
        /// </summary>
        /// <param name="interval">The interval.</param>
        public void Add(Interval interval, int agentId = -1, int prio = -1)
        {
            this.Add(interval.Node, interval.Start, interval.End, agentId, prio);
        }

        /// <summary>
        /// Adds the specified intervals.
        /// </summary>
        /// <param name="intervals">The intervals.</param>
        public void Add(List<Interval> intervals, int agentId = -1, int prio = -1)
        {
            for (int i = 0; i < intervals.Count; i++)
                this.Add(intervals[i], agentId, prio);
        }

        /// <summary>
        /// Removes the specified interval.
        /// </summary>
        /// <param name="interval">The interval.</param>
        /// <returns>true, if removed</returns>
        public bool Remove(Interval interval)
        {
            if (double.IsPositiveInfinity(interval.End))
                return RemoveIntersectionWithTime(interval.Node, interval.Start + 1.0);
            else
                return RemoveIntersectionWithTime(interval.Node, interval.Start / 2.0 + interval.End / 2.0);
        }

        /// <summary>
        /// Removes the specified intervals.
        /// </summary>
        /// <param name="intervals">The intervals.</param>
        public void Remove(List<Interval> intervals)
        {
            for (int i = 0; i < intervals.Count; i++)
                this.Remove(intervals[i]);
        }

        /// <summary>
        /// Removes the interval intersects with point in time.
        /// </summary>
        /// <param name="interval">The interval.</param>
        /// <returns>true, if removed</returns>
        public bool RemoveIntersectionWithTime(int node, double time)
        {
            //lasy initialization
            if (_intervallTrees[node] == null)
                return false;

            var removed = _intervallTrees[node].RemoveIntersectionWithTime(time);

            if (_touchedNodes != null && _touchedNodes.Contains(node) && _intervallTrees[node].Count == 0)
                _touchedNodes.Remove(node);

            return removed;
        }

        /// <summary>
        /// Checks weather the given interval intersects with an existing interval.
        /// </summary>
        /// <param name="node">The node.</param>
        /// <param name="start">The start.</param>
        /// <param name="end">The end.</param>
        /// <returns>false, if there is an intersection</returns>
        public bool IntersectionFree(int node, double start, double end)
        {
            //lasy initialization
            if (_intervallTrees[node] == null)
                return true;

            return _intervallTrees[node].IntersectionFree(start, end);
        }

        /// <summary>
        /// Checks weather the given interval intersects with an existing interval.
        /// </summary>
        /// <param name="node">The node.</param>
        /// <param name="start">The start.</param>
        /// <param name="end">The end.</param>
        /// <returns>false, if there is an intersection</returns>
        public bool IntersectionFree(int node, double start, double end, out int agentId)
        {
            agentId = -1;

            //lasy initialization
            if (_intervallTrees[node] == null)
                return true;

            return _intervallTrees[node].IntersectionFree(start, end, out agentId);
        }

        /// <summary>
        /// Checks weather the given interval intersects with an existing interval.
        /// </summary>
        /// <param name="node">The node.</param>
        /// <param name="start">The start.</param>
        /// <param name="end">The end.</param>
        /// <returns>false, if there is an intersection</returns>
        public bool IntersectionFree(int node, double start, double end, out List<Collision> agentsAndPrios)
        {
            agentsAndPrios = null;

            //lasy initialization
            if (_intervallTrees[node] == null)
                return true;

            return _intervallTrees[node].IntersectionFree(start, end, node, out agentsAndPrios);
        }

        /// <summary>
        /// Checks weather the given interval intersects with an existing interval.
        /// </summary>
        /// <param name="interval">The interval.</param>
        /// <returns>false, if there is an intersection</returns>
        public bool IntersectionFree(Interval interval)
        {
            return IntersectionFree(interval.Node, interval.Start, interval.End);
        }

        /// <summary>
        /// Checks weather the given interval intersects with an existing interval.
        /// </summary>
        /// <param name="interval">The interval.</param>
        /// <returns>false, if there is an intersection</returns>
        public bool IntersectionFree(Interval interval, out int agentId)
        {
            return IntersectionFree(interval.Node, interval.Start, interval.End, out agentId);
        }

        /// <summary>
        /// Checks weather the given interval intersects with an existing interval.
        /// </summary>
        /// <param name="interval">The interval.</param>
        /// <returns>false, if there is an intersection</returns>
        public bool IntersectionFree(Interval interval, out List<Collision> agentsAndPrios)
        {
            return IntersectionFree(interval.Node, interval.Start, interval.End, out agentsAndPrios);
        }

        /// <summary>
        /// Checks weather the given interval intersects with an existing interval.
        /// </summary>
        /// <param name="intervals">The list of intervals.</param>
        /// <returns>false, if there is an intersection</returns>
        public bool IntersectionFree(List<Interval> intervals)
        {
            for (int i = 0; i < intervals.Count; i++)
                if (!IntersectionFree(intervals[i].Node, intervals[i].Start, intervals[i].End))
                    return false;
            return true;
        }

        /// <summary>
        /// Checks weather the given interval intersects with an existing interval.
        /// </summary>
        /// <param name="intervals">The list of intervals.</param>
        /// <returns>false, if there is an intersection</returns>
        public bool IntersectionFree(List<Interval> intervals, out int nodeId, out int agentId)
        {
            nodeId = -1;
            agentId = -1;

            for (int i = 0; i < intervals.Count; i++)
            {
                if (!IntersectionFree(intervals[i].Node, intervals[i].Start, intervals[i].End, out agentId))
                {
                    nodeId = intervals[i].Node;
                    return false;
                }
            }
            return true;
        }

        /// <summary>
        /// Checks weather the given interval intersects with an existing interval.
        /// </summary>
        /// <param name="intervals">The list of intervals.</param>
        /// <returns>false, if there is an intersection</returns>
        public bool IntersectionFree(List<Interval> intervals, out List<Collision> agentsAndPrios)
        {
            agentsAndPrios = null;
            List<Collision> agentAndPriosSubset;

            for (int i = 0; i < intervals.Count; i++)
            {
                if (!IntersectionFree(intervals[i].Node, intervals[i].Start, intervals[i].End, out agentAndPriosSubset))
                {
                    //add conflicts
                    if (agentsAndPrios == null)
                        agentsAndPrios = new List<Collision>();
                    agentsAndPrios.AddRange(agentAndPriosSubset);
                }
            }

            //no conflict => no intersection
            return agentsAndPrios == null;
        }

        /// <summary>
        /// Checks weather the given interval intersects with an existing interval.
        /// </summary>
        /// <param name="checkpointNodes">The checkpoint nodes.</param>
        /// <param name="checkpointTimes">The checkpoint times.</param>
        /// <param name="checkIsFreeBeforeFirst">if set to <c>true</c> [check is free before first].</param>
        /// <param name="checkIsFreeAfterLast">if set to <c>true</c> [check is free after last].</param>
        /// <returns>
        /// false, if there is an intersection
        /// </returns>
        public bool IntersectionFree(List<int> checkpointNodes, List<double> checkpointTimes, bool checkIsFreeAfterLast)
        {
            //go through all checkpoints
            for (var i = 0; i < checkpointNodes.Count; i++)
            {
                if (i == 0)
                {
                    //Determine whether the node is reserved from ti-1 -> ti or infinity
                    if (!IntersectionFree(checkpointNodes[i], checkpointTimes[i], checkpointTimes[i + 1]))
                        return false;
                }
                else if (i == checkpointNodes.Count - 1)
                {
                    //Determine whether the node is reserved from ti-1 -> ti or infinity
                    if (!IntersectionFree(checkpointNodes[i], checkpointTimes[i - 1], checkIsFreeAfterLast ? double.PositiveInfinity : checkpointTimes[i]))
                        return false;
                }
                else
                {
                    //Determine whether the node is reserved from ti-1 -> ti+1
                    if (!IntersectionFree(checkpointNodes[i], checkpointTimes[i - 1], checkpointTimes[i + 1]))
                        return false;
                }

            }

            return true;
        }

        /// <summary>
        /// Checks weather the given interval intersects with an existing interval.
        /// </summary>
        /// <param name="checkpointNodes">The checkpoint nodes.</param>
        /// <param name="checkpointTimes">The checkpoint times.</param>
        /// <param name="checkIsFreeBeforeFirst">if set to <c>true</c> [check is free before first].</param>
        /// <param name="checkIsFreeAfterLast">if set to <c>true</c> [check is free after last].</param>
        /// <returns>
        /// false, if there is an intersection
        /// </returns>
        public bool IntersectionFree(List<int> checkpointNodes, List<double> checkpointTimes, bool checkIsFreeAfterLast, out int agentId)
        {
            agentId = -1;

            //go through all checkpoints
            for (var i = 0; i < checkpointNodes.Count; i++)
            {
                if (i == 0)
                {
                    //Determine whether the node is reserved from ti-1 -> ti or infinity
                    if (!IntersectionFree(checkpointNodes[i], checkpointTimes[i], checkpointTimes[i + 1], out agentId))
                        return false;
                }
                else if (i == checkpointNodes.Count - 1)
                {
                    //Determine whether the node is reserved from ti-1 -> ti or infinity
                    if (!IntersectionFree(checkpointNodes[i], checkpointTimes[i - 1], checkIsFreeAfterLast ? double.PositiveInfinity : checkpointTimes[i], out agentId))
                        return false;
                }
                else
                {
                    //Determine whether the node is reserved from ti-1 -> ti+1
                    if (!IntersectionFree(checkpointNodes[i], checkpointTimes[i - 1], checkpointTimes[i + 1], out agentId))
                        return false;
                }

            }

            return true;
        }

        /// <summary>
        /// Checks weather the given interval intersects with an existing interval.
        /// </summary>
        /// <param name="checkpointNodes">The checkpoint nodes.</param>
        /// <param name="checkpointTimes">The checkpoint times.</param>
        /// <param name="checkIsFreeBeforeFirst">if set to <c>true</c> [check is free before first].</param>
        /// <param name="checkIsFreeAfterLast">if set to <c>true</c> [check is free after last].</param>
        /// <returns>
        /// false, if there is an intersection
        /// </returns>
        public bool IntersectionFree(List<int> checkpointNodes, List<double> checkpointTimes, bool checkIsFreeAfterLast, out List<Collision> agentsAndPrios)
        {
            agentsAndPrios = null;
            List<Collision> agentAndPriosSubset;

            //go through all checkpoints
            for (var i = 0; i < checkpointNodes.Count; i++)
            {
                if (i == 0)
                {
                    //Determine whether the node is reserved from ti-1 -> ti or infinity
                    if (!IntersectionFree(checkpointNodes[i], checkpointTimes[i], checkpointTimes[i + 1], out agentAndPriosSubset))
                    {
                        agentsAndPrios = agentsAndPrios ?? new List<Collision>();
                        agentsAndPrios.AddRange(agentAndPriosSubset);
                    }
                }
                else if (i == checkpointNodes.Count - 1)
                {
                    //Determine whether the node is reserved from ti-1 -> ti or infinity
                    if (!IntersectionFree(checkpointNodes[i], checkpointTimes[i - 1], checkIsFreeAfterLast ? double.PositiveInfinity : checkpointTimes[i], out agentAndPriosSubset))
                    {
                        agentsAndPrios = agentsAndPrios ?? new List<Collision>();
                        agentsAndPrios.AddRange(agentAndPriosSubset);
                    }
                }
                else
                {
                    //Determine whether the node is reserved from ti-1 -> ti+1
                    if (!IntersectionFree(checkpointNodes[i], checkpointTimes[i - 1], checkpointTimes[i + 1], out agentAndPriosSubset))
                    {
                        agentsAndPrios = agentsAndPrios ?? new List<Collision>();
                        agentsAndPrios.AddRange(agentAndPriosSubset);
                    }
                }

            }

            return agentsAndPrios == null;
        }

        /// <summary>
        /// Remove all entries up to the current time.
        /// </summary>
        /// <param name="currentTime">The current time.</param>
        public void Reorganize(double currentTime)
        {
            if (_touchedNodes != null)
            {
                //we know the touched nodes => reorganize the touched ones
                var deletableNodes = new HashSet<int>();

                foreach (var touchedNode in _touchedNodes)
                {
                    _intervallTrees[touchedNode].Reorganize(currentTime);

                    if (_intervallTrees[touchedNode].Count == 0)
                        deletableNodes.Add(touchedNode);
                }

                foreach (var node in deletableNodes)
                    _touchedNodes.Remove(node);
            }
            else
            {
                //we don't know the touched nodes => reorganize all
                foreach (var intervall in _intervallTrees)
                    if (intervall != null)
                        intervall.Reorganize(currentTime);

            }
        }

        /// <summary>
        /// Creates the intervals for the given parameters.
        /// </summary>
        /// <param name="startIntervalAt">The point in time where the intervals should start.</param>
        /// <param name="checkpointNodes">The checkpoint nodes.</param>
        /// <param name="checkpointTimes">The checkpoint times.</param>
        /// <param name="addIntervalToInfinity">if set to <c>true</c> to create intervals until double.PositiveInfinity.</param>
        /// <returns>
        /// false, if there is an intersection
        /// </returns>
        public List<Interval> CreateIntervals(double startIntervalAt, List<int> checkpointNodes, List<double> checkpointTimes, bool addIntervalToInfinity)
        {
            var intervals = new List<Interval>();

            //go through all checkpoints
            for (var i = 0; i < checkpointNodes.Count; i++)
            {
                if (i == 0)
                {
                    //Determine whether the node is reserved from ti-1 -> ti or infinity
                    intervals.Add(new Interval(checkpointNodes[i], startIntervalAt, checkpointTimes[i + 1]));
                }
                else if (i == checkpointNodes.Count - 1)
                {
                    //Determine whether the node is reserved from ti-1 -> ti or infinity
                    intervals.Add(new Interval(checkpointNodes[i], checkpointTimes[i - 1], checkpointTimes[i]));
                    if (addIntervalToInfinity)
                        intervals.Add(new Interval(checkpointNodes[i], checkpointTimes[i], double.PositiveInfinity));

                }
                else
                {
                    //Determine whether the node is reserved from ti-1 -> ti+1
                    intervals.Add(new Interval(checkpointNodes[i], checkpointTimes[i - 1], checkpointTimes[i + 1]));
                }

            }

            return intervals;
        }

        /// <summary>
        /// Creates the intervals for the given parameters.
        /// 
        /// Node 1 2 3 4  ____________ startIntervalAt
        ///      |        
        ///      |        
        ///      |        ____________ startDrivingAT
        ///       \
        ///        \
        ///         \ 
        ///          \ 
        ///           \  _____________
        ///            |
        ///            | addIntervallToInfinity == true
        /// 
        /// </summary>
        /// <param name="startIntervalAt">The point in time where the intervals should start.</param>
        /// <param name="startDrivingAt">The point in time the agent starts driving.</param>
        /// <param name="currentSpeed">The current speed.</param>
        /// <param name="physics">The physics.</param>
        /// <param name="startNode">The start node.</param>
        /// <param name="destinationNode">The destination node.</param>
        /// <param name="addIntervalToInfinity">if set to <c>true</c> to create intervals until double.PositiveInfinity.</param>
        /// <returns>
        /// false, if there is an intersection
        /// </returns>
        public List<Interval> CreateIntervals(double startIntervalAt, double startDrivingAt, double currentSpeed, Physic.Physics physics, int startNode, int destinationNode, bool addIntervalToInfinity)
        {
            if (currentSpeed > 0 && startIntervalAt == startDrivingAt)
                throw new ArgumentException("The agent is already driving => startIntervalAt must be equal to startDrivingAt!");

            List<int> checkPointNodes;
            List<double> checkPointTimes;
            var valid = this.GetCheckPointNodes(startDrivingAt, currentSpeed, physics, startNode, destinationNode, out checkPointNodes, out checkPointTimes);

            if (!valid)
                return null;

            return this.CreateIntervals(startIntervalAt, checkPointNodes, checkPointTimes, addIntervalToInfinity);
        }

        /// <summary>
        /// get all passed nodes on the way.
        /// </summary>
        /// <param name="startTime">The current time.</param>
        /// <param name="currentSpeed">The current speed.</param>
        /// <param name="physics">The physics.</param>
        /// <param name="startNode">The start node.</param>
        /// <param name="destinationNode">The destination node.</param>
        /// <param name="checkpointNodes">The checkpoint nodes.</param>
        /// <param name="checkpointTimes">The checkpoint times.</param>
        /// <returns>
        /// check point nodes + times
        /// </returns>
        public bool GetCheckPointNodes(double startTime, double currentSpeed, Physic.Physics physics, int startNode, int destinationNode, out List<int> checkpointNodes, out List<double> checkpointTimes)
        {
            //get checkpoints
            checkpointNodes = _graph.getIntermediateNodes(startNode, destinationNode);
            checkpointTimes = null;

            if (checkpointNodes == null)
                return false; //no valid way point

            checkpointNodes.Insert(0, startNode);
            checkpointNodes.Add(destinationNode);

            var checkpointDistances = new List<double>();
            foreach (var node in checkpointNodes)
                checkpointDistances.Add(_graph.getDistance(startNode, node));

            physics.getTimeNeededToMove(currentSpeed, startTime, checkpointDistances[checkpointDistances.Count - 1], checkpointDistances, out checkpointTimes);

            return true;
        }

        /// <summary>
        /// Gets the interval of the intersection with the intervals of this table and the given interval.
        /// </summary>
        /// <param name="interval">The interval that overlap with an existing one.</param>
        /// <returns>The overlapping interval</returns>
        public ReservationTable.Interval GetOverlappingInterval(ReservationTable.Interval interval)
        {
            double start;
            double end;
            _intervallTrees[interval.Node].GetOverlappingInterval(interval.Start, interval.End, out start, out end);
            return new ReservationTable.Interval(interval.Node, start, end);
        }

        /// <summary>
        /// Interval
        /// </summary>
        public class Interval
        {
            /// <summary>
            /// Node
            /// </summary>
            public int Node;

            /// <summary>
            /// From
            /// </summary>
            public double Start;

            /// <summary>
            /// To
            /// </summary>
            public double End;

            /// <summary>
            /// Initializes a new instance of the <see cref="Interval"/> class.
            /// </summary>
            /// <param name="start">The start.</param>
            /// <param name="end">The end.</param>
            public Interval(int node, double start, double end)
            {
                this.Node = node;
                this.Start = start;
                this.End = end;
            }

            /// <summary>
            /// Returns a <see cref="System.String" /> that represents this instance.
            /// </summary>
            /// <returns>
            /// A <see cref="System.String" /> that represents this instance.
            /// </returns>
            public override string ToString()
            {
                return "Node " + Node + " - [" + Start + " - " + End + "]";
            }
        }
    }
}
