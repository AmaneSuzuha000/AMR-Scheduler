using RAWSimO.Core.Bots;
using RAWSimO.Core.Configurations;
using RAWSimO.Core.Elements;
using RAWSimO.Core.Helper;
using RAWSimO.Core.Interfaces;
using RAWSimO.Core.Waypoints;
using RAWSimO.MultiAgentPathFinding;
using RAWSimO.MultiAgentPathFinding.Elements;
using RAWSimO.MultiAgentPathFinding.Methods;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RAWSimO.Core.Control.Defaults.PathPlanning
{
    /// <summary>
    /// Wires the priority space-time A* planner into the simulation.
    /// </summary>
    public class PSTAPathManager : PathManager
    {
        /// <summary>
        /// Creates a new instance of the manager.
        /// </summary>
        /// <param name="instance">The instance this manager belongs to.</param>
        public PSTAPathManager(Instance instance) : base(instance)
        {
            // A failed reservation means the plan must be revised, not retried blindly.
            BotNormal.RequestReoptimizationAfterFailingOfNextWaypointReservation = true;

            var graph = GenerateGraph();
            var config = instance.ControllerConfig.PathPlanningConfig as PSTAPathPlanningConfiguration;

            PathFinder = new PSTAMethod(graph, instance.SettingConfig.Seed, new PathPlanningCommunicator(
                instance.LogSevere,
                instance.LogDefault,
                instance.LogInfo,
                instance.LogVerbose,
                () => { instance.StatOverallPathPlanningTimeouts++; }));

            var method = PathFinder as PSTAMethod;
            method.LengthOfAWaitStep = config.LengthOfAWaitStep;
            method.RuntimeLimitPerAgent = config.RuntimeLimitPerAgent;
            method.RunTimeLimitOverall = config.RunTimeLimitOverall;
            method.UsePriorityOrdering = config.UsePriorityOrdering;
            method.LengthOfAWindow = config.LengthOfAWindow;
            method.MaximumRetries = config.MaximumRetries;
            method.DemoteOnRetry = config.DemoteOnRetry;
            method.WaitBackoffOnRetry = config.WaitBackoffOnRetry;
            method.WaitBackoffBase = config.WaitBackoffBase;
            method.UsePersistentPriority = config.UsePersistentPriority;
            method.PersistentPriorityDecay = config.PersistentPriorityDecay;
        }
    }
}
