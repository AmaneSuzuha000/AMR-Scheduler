using RAWSimO.MultiAgentPathFinding.Algorithms.AStar;
using RAWSimO.MultiAgentPathFinding.DataStructures;
using RAWSimO.MultiAgentPathFinding.Elements;
using RAWSimO.MultiAgentPathFinding.Physic;
using RAWSimO.MultiAgentPathFinding.Toolbox;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RAWSimO.MultiAgentPathFinding.Methods
{
    /// <summary>
    /// Priority Space-Time A*: resolves multi-agent conflicts by planning agents in a
    /// cost-ordered sequence, so that the agents with the most to lose from waiting get to
    /// pick their routes first and everyone after them routes around those choices.
    ///
    /// The single-agent search itself is delegated to SpaceTimeAStar, the solver this
    /// codebase already ships and has validated. An earlier revision reimplemented the search
    /// from scratch and produced plans that satisfied A* but not the planner contract - paths
    /// made of actions flagged StopAtNode == false, which BotNormal discards while walking the
    /// action list, leaving the fleet stationary with full paths in hand. Reusing the proven
    /// solver keeps that contract in one place; the contribution here is the ordering.
    /// </summary>
    public class PSTAMethod : PathFinder
    {
        /// <summary>
        /// The length of the planning window, in seconds.
        /// </summary>
        public double LengthOfAWindow = 20.0;

        /// <summary>
        /// Indicates whether the method uses a deadlock handler.
        /// </summary>
        public bool UseDeadlockHandler = true;

        /// <summary>
        /// Whether to order agents by cost before planning.
        ///
        /// This exists so the mechanism can be ablated: switching it off makes PSTA plan in
        /// the order the agents arrive in (bot ID order), which is exactly what FAR does.
        /// If the measured gain really comes from the ordering, turning this off must
        /// collapse PSTA back to FAR's numbers. Without that control the improvement is just
        /// a correlation.
        /// </summary>
        public bool UsePriorityOrdering = true;

        /// <summary>
        /// How many times a round may be retried when some agent fails to find a
        /// conflict-free plan. 1 keeps the original behaviour exactly: plan once, let
        /// failures stand. Values above 1 retry only the agents that failed, against a
        /// reservation table holding the routes the successful ones already claimed.
        ///
        /// This mirrors WHCAvStar, which retries and raises the priority of agents it could
        /// not plan so that they yield (WHCAvMethod.cs:84-95, :155).
        /// </summary>
        public int MaximumRetries = 1;

        /// <summary>
        /// When retrying, replan the WHOLE round instead of only the agents that failed,
        /// and demote agents that keep failing so they sort later next time. This is what
        /// WHCAvStar actually does (every retry re-sorts by agentPrios, and a failing agent
        /// gets agentPrios[ID]++). Retrying failures alone, against an unchanged table, does
        /// not help - measured and rejected, see PSTA-重试实验-负结果.md.
        /// </summary>
        public bool DemoteOnRetry = false;

        /// <summary>
        /// Whether a demoted agent also holds back before setting off, instead of only
        /// being sorted later. These are two separable halves of WHCAvStar's mechanism and
        /// they are measured apart: reordering changes who competes for a node, holding back
        /// changes when they compete for it. Conflating the two hides which one works.
        /// </summary>
        public bool WaitBackoffOnRetry = true;

        /// <summary>
        /// Base of the backoff growth. WHCAvStar raises it as Math.Pow(2, prio) / 2, which
        /// is an arbitrary choice inherited from that method's priority schedule. Under a
        /// whole-trip solver the useful shape is not assumed to be the same, so it is a knob.
        /// </summary>
        public double WaitBackoffBase = 2.0;

        /// <summary>
        /// Keep a failure penalty per agent ACROSS planning calls, and use it in the sort
        /// order even when no retry happens. The measured gain of the retry mechanism comes
        /// entirely from reordering (see PSTA-重试机理分解.md), while retrying itself is
        /// neutral-to-harmful and makes the result jump around with the retry count. If the
        /// reordering alone reproduces the gain, the expensive retry loop is unnecessary.
        ///
        /// The penalty decays every call so an agent that recovers is not punished forever.
        /// </summary>
        public bool UsePersistentPriority = false;

        /// <summary>
        /// How fast a persistent failure penalty fades, per planning call. 1.0 never forgets.
        /// </summary>
        public double PersistentPriorityDecay = 0.9;

        /// <summary>
        /// The failure penalty carried between planning calls, keyed by agent id.
        /// </summary>
        private readonly Dictionary<int, double> _persistentPenalty = new Dictionary<int, double>();

        /// <summary>
        /// The reverse resumable A* searches, one per agent.
        /// </summary>
        public Dictionary<int, ReverseResumableAStar> rraStars;

        /// <summary>
        /// The reservation table shared by all agents of a planning round.
        /// </summary>
        public ReservationTable _reservationTable;

        /// <summary>
        /// The deadlock handler.
        /// </summary>
        private DeadlockHandler _deadlockHandler;

        /// <summary>
        /// Creates a new instance of the planner.
        /// </summary>
        /// <param name="graph">The graph to plan on.</param>
        /// <param name="seed">The seed to use for the randomizer.</param>
        /// <param name="logger">The logger to use.</param>
        public PSTAMethod(Graph graph, int seed, PathPlanningCommunicator logger)
            : base(graph, seed, logger)
        {
            if (graph.BackwardEdges == null)
                graph.GenerateBackwardEgdes();
            rraStars = new Dictionary<int, ReverseResumableAStar>();
            _reservationTable = new ReservationTable(graph);
            if (UseDeadlockHandler)
                _deadlockHandler = new DeadlockHandler(graph, seed);
        }

        /// <summary>
        /// Plans a path for every agent that needs one.
        /// </summary>
        /// <param name="currentTime">The current simulation time.</param>
        /// <param name="agents">The agents to plan for.</param>
        public override void FindPaths(double currentTime, List<Agent> agents)
        {
            Stopwatch.Restart();

            _reservationTable.Clear();
            var fixedBlockage = AgentInfoExtractor.getStartBlockage(agents, currentTime);

            // Agents that cannot move are permanent obstacles for this round. Omitting this
            // lets the search route straight through a parked bot and produces a plan that is
            // rejected downstream; measured effect was 4285 -> 4266 items on seed 11.
            foreach (var interval in fixedBlockage.Values.SelectMany(l => l))
            {
                try { _reservationTable.Add(interval); }
                catch (DisjointIntervalTree.IntervalIntersectionException) { }
            }

            // NOTE: An earlier revision additionally reserved every agent's NextNode until
            // t = +infinity, mirroring FAR's setup. That is measurably harmful here and was
            // reverted: because this method hands the whole trip to SpaceTimeAStar (rather
            // than committing one hop at a time like FAR), an infinite claim on every agent's
            // next node removes those nodes from every other agent's search for all time.
            // Seed 11 fell from 4285 to 1419 items. See PSTA-实现记录.md.

            // Order the agents before planning anyone. This is the actual contribution of the
            // method: an agent whose trip is long and whose destination is contested is served
            // first, so it can claim a direct route, while agents with cheap trips absorb the
            // detours. Planning in a fixed or random order instead produces the familiar
            // pattern where one unlucky agent is pushed into a long detour every round.
            var movable = agents.Where(a => !a.FixedPosition);

            // A persistent penalty lets an agent that keeps failing sort later without any
            // retry happening. Measured separately from the retry loop so the two effects do
            // not get conflated again.
            if (UsePersistentPriority && _persistentPenalty.Count > 0)
            {
                foreach (var key in _persistentPenalty.Keys.ToList())
                {
                    var faded = _persistentPenalty[key] * PersistentPriorityDecay;
                    if (faded < 0.5) _persistentPenalty.Remove(key);
                    else _persistentPenalty[key] = faded;
                }
            }

            // The penalty is sorted on FIRST, as WHCAvStar does with agentPrios, rather than
            // added to Priority(). Priority() is measured in seconds of travel and runs into
            // the hundreds; a penalty of one or two added to it would be lost in the noise.
            var ordered = (!UsePriorityOrdering
                ? movable
                : UsePersistentPriority
                    ? movable.OrderByDescending(a => Penalty(a)).ThenByDescending(a => Priority(a))
                    : movable.OrderByDescending(a => Priority(a))).ToList();

            if (UseDeadlockHandler)
            {
                _deadlockHandler.LengthOfAWaitStep = LengthOfAWaitStep;
                _deadlockHandler.MaximumWaitTime = 30;
                _deadlockHandler.Update(agents, currentTime);
            }

            // Round 0 plans every agent exactly as before. Further rounds retry only the
            // agents that failed, against the same reservation table - it still holds the
            // routes the successful agents claimed, so a retried agent has to route around
            // them instead of the whole round being recomputed from scratch.
            // Round 0 plans every agent in priority order. WHCAvStar-style retry, when it is
            // switched on, replans the whole round and pushes persistent failures down the
            // order, so they yield instead of forcing everyone else around them.
            var demotion = new Dictionary<int, int>();
            var failed = new List<Agent>();

            for (int attempt = 0; attempt < Math.Max(1, MaximumRetries); attempt++)
            {
                var round = attempt == 0
                    ? ordered
                    : ordered.OrderByDescending(a => Demotion(demotion, a)).ToList();
                failed = new List<Agent>();

                foreach (var agent in round)
                {
                    if (Stopwatch.ElapsedMilliseconds / 1000.0 > RunTimeLimitOverall)
                    {
                        Communicator.SignalTimeout();
                        return;
                    }

                    // Reuse or rebuild the reverse search towards the agent's destination.
                    if (!rraStars.TryGetValue(agent.ID, out ReverseResumableAStar rraStar) ||
                        rraStar.StartNode != agent.DestinationNode ||
                        UseDeadlockHandler && _deadlockHandler.IsInDeadlock(agent, currentTime))
                    {
                        rraStars[agent.ID] = new ReverseResumableAStar(Graph, agent, agent.Physics, agent.DestinationNode);
                    }

                    var aStar = new SpaceTimeAStar(Graph, LengthOfAWaitStep, currentTime + LengthOfAWindow, _reservationTable, agent, rraStars[agent.ID]);

                    // A demoted agent deliberately holds back before setting off, which frees
                    // the contested node for everyone else - WHCAvStar's WaitStepsBeforeStart.
                    if (attempt > 0 && WaitBackoffOnRetry)
                        aStar.WaitStepsBeforeStart = (int)(Math.Pow(WaitBackoffBase, Demotion(demotion, agent)) / 2.0);

                    var found = aStar.Search();

                    if (found)
                    {
                        List<ReservationTable.Interval> reservations;
                        aStar.GetPathAndReservations(ref agent.Path, out reservations);
                        foreach (var reservation in reservations)
                        {
                            try { _reservationTable.Add(reservation); }
                            catch (DisjointIntervalTree.IntervalIntersectionException) { }
                        }

                        // Re-anchor BEFORE seeding the reverse search: addPath() records
                        // where the agent will be, so it must see the finalized path.
                        if (agent.Path.Count == 0 || agent.Path.NextAction.Node != agent.NextNode || agent.Path.NextAction.StopAtNode == false)
                            agent.Path.AddFirst(agent.NextNode, true, 0);
                        if (aStar.GoalNode >= 0)
                            rraStars[agent.ID].addPath(agent.Path, aStar.NodeTo2D(aStar.GoalNode));
                    }
                    else
                    {
                        failed.Add(agent);
                        if (DemoteOnRetry)
                            demotion[agent.ID] = Demotion(demotion, agent) + 1;
                        if (UsePersistentPriority)
                            _persistentPenalty[agent.ID] = Penalty(agent) + 1.0;

                        // Only an agent that FAILED to find a plan gets a random hop. It is
                        // the agent that could not be placed, so it is the one that gives way.
                        // Same hotfix WHCAvStar uses (WHCAvMethod.cs:198-200).
                        if (UseDeadlockHandler)
                            if (_deadlockHandler.IsInDeadlock(agent, currentTime))
                                _deadlockHandler.RandomHop(agent);
                    }
                }

                if (attempt == 0 && MaximumRetries <= 1) break;
                if (failed.Count == 0) break;

                // A retry has to re-decide the whole round, so the table is rebuilt from the
                // fixed blockage. Reusing the previous round's claims would leave a demoted
                // agent facing exactly the constraints that already defeated it.
                _reservationTable.Clear();
                foreach (var interval in fixedBlockage.Values.SelectMany(l => l))
                {
                    try { _reservationTable.Add(interval); }
                    catch (DisjointIntervalTree.IntervalIntersectionException) { }
                }
            }
        }

        /// <summary>
        /// How often an agent has failed to be planned, zero for the common case.
        /// </summary>
        /// <param name="demotion">The per-agent failure counts.</param>
        /// <param name="agent">The agent to look up.</param>
        /// <returns>The number of failures so far.</returns>
        private static int Demotion(Dictionary<int, int> demotion, Agent agent)
            => demotion.TryGetValue(agent.ID, out var d) ? d : 0;

        /// <summary>
        /// The carried-over failure penalty for an agent, zero when it has not failed.
        /// </summary>
        /// <param name="agent">The agent to look up.</param>
        /// <returns>The accumulated penalty.</returns>
        private double Penalty(Agent agent)
            => _persistentPenalty.TryGetValue(agent.ID, out var p) ? p : 0.0;

        /// <summary>
        /// Ranks an agent by how expensive it is to inconvenience. Longer trips and agents
        /// carrying a pod score higher, so they are planned earlier and keep straighter routes.
        /// </summary>
        /// <param name="agent">The agent to rank.</param>
        /// <returns>The priority; larger means planned earlier.</returns>
        private double Priority(Agent agent)
        {
            double dx = Graph.PositionX[agent.NextNode] - Graph.PositionX[agent.DestinationNode];
            double dy = Graph.PositionY[agent.NextNode] - Graph.PositionY[agent.DestinationNode];
            double distance = Math.Sqrt(dx * dx + dy * dy);

            // Time to cover the remaining distance at full speed; waiting costs an agent with
            // a long remaining trip more than one that is nearly done.
            double travelTime = distance / Math.Max(agent.Physics.MaxSpeed, 1e-6);

            // An agent that has not started moving yet is more flexible than one already in
            // motion, so give the moving agent the stronger claim.
            double momentum = agent.ArrivalTimeAtNextNode > 0 ? 1.0 : 0.0;

            return travelTime + momentum * LengthOfAWaitStep;
        }
    }
}
