using RAWSimO.Core.IO;
using RAWSimO.MultiAgentPathFinding.Methods;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RAWSimO.Core.Configurations
{
    #region Path planning configurations

    /// <summary>
    /// The configuration for the corresponding method.
    /// </summary>
    public class SimplePathPlanningConfiguration : PathPlanningConfiguration
    {
        /// <summary>
        /// Returns the type of the corresponding method this configuration belongs to.
        /// </summary>
        /// <returns>The type of the method.</returns>
        public override PathPlanningMethodType GetMethodType() { return PathPlanningMethodType.Simple; }
        /// <summary>
        /// Returns a name identifying the method.
        /// </summary>
        /// <returns>The name of the method.</returns>
        public override string GetMethodName()
        {
            if (!string.IsNullOrWhiteSpace(Name)) return Name;
            return "ppS" +
                (SimpleWaitingEnabled ? "t" : "f") +
                (SimpleWaitingExtendedEnabled ? "t" : "f") +
                (SimpleWaitingD2Enabled ? "t" : "f") +
                (PingPongWaitingEnabled ? "t" : "f");
        }
        /// <summary>
        /// Enables or disables bots simply waiting for the next waypoint to be clear before advancing.
        /// </summary>
        public bool SimpleWaitingEnabled = true;

        /// <summary>
        /// Enables or disables bots blocking the waypoint they just left.
        /// </summary>
        public bool SimpleWaitingExtendedEnabled = true;

        /// <summary>
        /// Enables or disables waiting for bots approaching the after next waypoint.
        /// </summary>
        public bool SimpleWaitingD2Enabled = true;

        /// <summary>
        /// Enables or disables random waiting after a ping pong (bot moving back and forth between two waypoints) occurred.
        /// </summary>
        public bool PingPongWaitingEnabled = true;

        /// <summary>
        /// Parses the specified arguments.
        /// </summary>
        /// <param name="args">The arguments.</param>
        public override void Parse(string[] args)
        {
            base.Parse(args);
            SimpleWaitingEnabled = bool.Parse(args[2]);
            SimpleWaitingExtendedEnabled = bool.Parse(args[3]);
            SimpleWaitingD2Enabled = bool.Parse(args[4]);
            PingPongWaitingEnabled = bool.Parse(args[5]);
        }
    }
    /// <summary>
    /// The configuration for the corresponding method.
    /// </summary>
    public class DummyPathPlanningConfiguration : PathPlanningConfiguration
    {
        /// <summary>
        /// Returns the type of the corresponding method this configuration belongs to.
        /// </summary>
        /// <returns>The type of the method.</returns>
        public override PathPlanningMethodType GetMethodType() { return PathPlanningMethodType.Dummy; }
        /// <summary>
        /// Returns a name identifying the method.
        /// </summary>
        /// <returns>The name of the method.</returns>
        public override string GetMethodName() { if (!string.IsNullOrWhiteSpace(Name)) return Name; return "ppD"; }
    };
    /// <summary>
    /// The configuration for the corresponding method.
    /// </summary>
    public class WHCAvStarPathPlanningConfiguration : PathPlanningConfiguration
    {
        /// <summary>
        /// Returns the type of the corresponding method this configuration belongs to.
        /// </summary>
        /// <returns>The type of the method.</returns>
        public override PathPlanningMethodType GetMethodType() { return PathPlanningMethodType.WHCAvStar; }
        /// <summary>
        /// Returns a name identifying the method.
        /// </summary>
        /// <returns>The name of the method.</returns>
        public override string GetMethodName() { if (!string.IsNullOrWhiteSpace(Name)) return Name; return "ppWHCAV" + LengthOfAWindow.ToString(IOConstants.EXPORT_FORMAT_SHORTER, IOConstants.FORMATTER) + (AbortAtFirstConflict ? "t" : "f") + (UseDeadlockHandler ? "t" : "f"); }
        /// <summary>
        /// The length of a time space search window
        /// </summary>
        public double LengthOfAWindow = 30.0;
        /// <summary>
        /// Abort the search at first conflict
        /// </summary>
        public bool AbortAtFirstConflict = true;
        /// <summary>
        /// Indicates whether the method uses a deadlock handler.
        /// </summary>
        public bool UseDeadlockHandler = true;
        /// <summary>
        /// Parses the specified arguments.
        /// </summary>
        /// <param name="args">The arguments.</param>
        public override void Parse(string[] args)
        {
            base.Parse(args);
            LengthOfAWindow = double.Parse(args[2], new CultureInfo("en"));
            AbortAtFirstConflict = bool.Parse(args[3]);
        }
        /// <summary>
        /// Checks whether the path planning configuration is valid.
        /// </summary>
        /// <param name="errorMessage">A message describing the error if the configuration is not valid.</param>
        /// <returns>Indicates whether the path planning configuration is valid.</returns>
        public override bool AttributesAreValid(out String errorMessage)
        {
            if (!base.AttributesAreValid(out errorMessage))
            {
                return false;
            }
            if (LengthOfAWindow <= 0)
            {
                errorMessage = "Problem with path planning configuration: LengthOfAWindow has to be > 0.";
                return false;
            }

            errorMessage = "";
            return true;

        }
    }
    /// <summary>
    /// The configuration for the corresponding method.
    /// </summary>
    public class WHCAnStarPathPlanningConfiguration : PathPlanningConfiguration
    {
        /// <summary>
        /// Returns the type of the corresponding method this configuration belongs to.
        /// </summary>
        /// <returns>The type of the method.</returns>
        public override PathPlanningMethodType GetMethodType() { return PathPlanningMethodType.WHCAnStar; }
        /// <summary>
        /// Returns a name identifying the method.
        /// </summary>
        /// <returns>The name of the method.</returns>
        public override string GetMethodName() { if (!string.IsNullOrWhiteSpace(Name)) return Name; return "ppWHCAN" + LengthOfAWindow.ToString(IOConstants.EXPORT_FORMAT_SHORTER, IOConstants.FORMATTER) + (UseBias ? "t" : "f") + (UseDeadlockHandler ? "t" : "f"); }
        /// <summary>
        /// The length of a time space search window
        /// </summary>
        public double LengthOfAWindow = 30.0;
        /// <summary>
        /// Indicates whether the method uses the biased cost pathfinding algorithm
        /// </summary>
        public bool UseBias = false;
        /// <summary>
        /// Indicates whether the method uses a deadlock handler.
        /// </summary>
        public bool UseDeadlockHandler = true;
        /// <summary>
        /// Parses the specified arguments.
        /// </summary>
        /// <param name="args">The arguments.</param>
        public override void Parse(string[] args)
        {
            base.Parse(args);
            LengthOfAWindow = double.Parse(args[2], new CultureInfo("en"));
            UseBias = bool.Parse(args[3]);
        }
        /// <summary>
        /// Checks whether the path planning configuration is valid.
        /// </summary>
        /// <param name="errorMessage">A message describing the error if the configuration is not valid.</param>
        /// <returns>Indicates whether the path planning configuration is valid.</returns>
        public override bool AttributesAreValid(out String errorMessage)
        {
            if (!base.AttributesAreValid(out errorMessage))
            {
                return false;
            }
            if (LengthOfAWindow <= 0)
            {
                errorMessage = "Problem with path planning configuration: LengthOfAWindow has to be > 0.";
                return false;
            }

            errorMessage = "";
            return true;

        }
    }
    /// <summary>
    /// The configuration for the priority space-time A* planner.
    /// </summary>
    public class PSTAPathPlanningConfiguration : PathPlanningConfiguration
    {
        /// <summary>
        /// Returns the type of the corresponding method this configuration belongs to.
        /// </summary>
        /// <returns>The type of the method.</returns>
        public override PathPlanningMethodType GetMethodType() { return PathPlanningMethodType.PSTA; }
        /// <summary>
        /// Returns a name identifying the method.
        /// </summary>
        /// <returns>The name of the method.</returns>
        public override string GetMethodName()
        {
            if (!string.IsNullOrWhiteSpace(Name)) return Name;
            return "ppPSTA" + (UsePriorityOrdering ? "t" : "f");
        }
        /// <summary>
        /// Whether agents are ordered by cost before planning. Turning this off plans in bot
        /// ID order, which reproduces FAR's behaviour and serves as the mechanism control.
        /// </summary>
        public bool UsePriorityOrdering = true;
        /// <summary>
        /// How far into the future the space-time search looks, in seconds. Longer windows let
        /// an agent plan around conflicts that are still far away, at the cost of a bigger
        /// search space per agent. Exposed because it is the main quality/cost dial.
        /// </summary>
        public double LengthOfAWindow = 20.0;
        /// <summary>
        /// How many times a planning round may be retried when agents fail to find a
        /// conflict-free plan. 1 keeps the original single-pass behaviour.
        /// </summary>
        public int MaximumRetries = 1;
        /// <summary>
        /// Whether a retry replans the whole round and demotes agents that keep failing,
        /// instead of only retrying the failures. Mirrors WHCAvStar's agentPrios mechanism.
        /// </summary>
        public bool DemoteOnRetry = false;
        /// <summary>
        /// Whether a demoted agent also holds back before setting off. Separable from the
        /// reordering so the two halves can be measured independently.
        /// </summary>
        public bool WaitBackoffOnRetry = true;
        /// <summary>
        /// Base of the backoff growth applied to a repeatedly failing agent.
        /// </summary>
        public double WaitBackoffBase = 2.0;
        /// <summary>
        /// Whether a failure penalty carries across planning calls and drives the sort order
        /// even without retrying.
        /// </summary>
        public bool UsePersistentPriority = false;
        /// <summary>
        /// How fast the carried penalty fades per planning call.
        /// </summary>
        public double PersistentPriorityDecay = 0.9;
        /// <summary>
        /// Parses the specified arguments.
        /// </summary>
        /// <param name="args">The arguments.</param>
        public override void Parse(string[] args)
        {
            base.Parse(args);
        }
        /// <summary>
        /// Checks whether the path planning configuration is valid.
        /// </summary>
        /// <param name="errorMessage">A message describing the error if the configuration is not valid.</param>
        /// <returns>Indicates whether the path planning configuration is valid.</returns>
        public override bool AttributesAreValid(out String errorMessage)
        {
            if (!base.AttributesAreValid(out errorMessage))
                return false;
            errorMessage = "";
            return true;
        }
    }
    /// <summary>
    /// The configuration for the corresponding method.
    /// </summary>
    public class FARPathPlanningConfiguration : PathPlanningConfiguration
    {
        /// <summary>
        /// Returns the type of the corresponding method this configuration belongs to.
        /// </summary>
        /// <returns>The type of the method.</returns>
        public override PathPlanningMethodType GetMethodType() { return PathPlanningMethodType.FAR; }
        /// <summary>
        /// Returns a name identifying the method.
        /// </summary>
        /// <returns>The name of the method.</returns>
        public override string GetMethodName()
        {
            if (!string.IsNullOrWhiteSpace(Name)) return Name;
            return "ppFAR" +
                (evadingStrategy == FARMethod.EvadingStrategy.EvadeByRerouting ? "P" : evadingStrategy == FARMethod.EvadingStrategy.EvadeToNextNode ? "A" : "?") +
                MaximumNumberOfBreakingManeuverTries.ToString() +
                (NoBackEvading ? "t" : "f") +
                (UseDeadlockHandler ? "t" : "f");
        }
        /// <summary>
        /// The maximum number of breaking maneuver tries
        /// </summary>
        public int MaximumNumberOfBreakingManeuverTries = 2;
        /// <summary>
        /// The evading strategy
        /// </summary>
        public FARMethod.EvadingStrategy evadingStrategy;
        /// <summary>
        /// No evading to a node the bot already evaded from
        /// </summary>
        public bool NoBackEvading = true;
        /// <summary>
        /// Indicates whether the method uses a deadlock handler.
        /// </summary>
        public bool UseDeadlockHandler = true;
        /// <summary>
        /// Whether agents are ordered by cost before planning, using the same ranking as PSTA.
        /// Off by default so existing FAR configurations keep their behaviour; switched on it
        /// becomes the control that isolates the effect of ordering from the effect of the
        /// planning paradigm itself.
        /// </summary>
        public bool UsePriorityOrdering = false;
        /// <summary>
        /// Parses the specified arguments.
        /// </summary>
        /// <param name="args">The arguments.</param>
        public override void Parse(string[] args)
        {
            base.Parse(args);
            MaximumNumberOfBreakingManeuverTries = int.Parse(args[2]);
            evadingStrategy = (FARMethod.EvadingStrategy)Enum.Parse(typeof(FARMethod.EvadingStrategy), (args[3]));
            NoBackEvading = bool.Parse(args[4]);
        }
        /// <summary>
        /// Checks whether the path planning configuration is valid.
        /// </summary>
        /// <param name="errorMessage">A message describing the error if the configuration is not valid.</param>
        /// <returns>Indicates whether the path planning configuration is valid.</returns>
        public override bool AttributesAreValid(out String errorMessage)
        {
            if (!base.AttributesAreValid(out errorMessage))
            {
                return false;
            }
            if (MaximumNumberOfBreakingManeuverTries < 0)
            {
                errorMessage = "Problem with path planning configuration: MaximumNumberOfBreakingManeuverTries has to be >= 0.";
                return false;
            }

            errorMessage = "";
            return true;

        }
    }
    /// <summary>
    /// The configuration for the corresponding method.
    /// </summary>
    public class CBSPathPlanningConfiguration : PathPlanningConfiguration
    {
        /// <summary>
        /// Returns the type of the corresponding method this configuration belongs to.
        /// </summary>
        /// <returns>The type of the method.</returns>
        public override PathPlanningMethodType GetMethodType() { return PathPlanningMethodType.CBS; }
        /// <summary>
        /// Returns a name identifying the method.
        /// </summary>
        /// <returns>The name of the method.</returns>
        public override string GetMethodName()
        {
            if (!string.IsNullOrWhiteSpace(Name)) return Name;
            string name = "ppCBS";
            switch (SearchMethod)
            {
                case CBSMethod.CBSSearchMethod.BestFirst: name += "O"; break;
                case CBSMethod.CBSSearchMethod.DepthFirst: name += "D"; break;
                case CBSMethod.CBSSearchMethod.BreathFirst: name += "B"; break;
                default: throw new ArgumentException("Unexpected argument!");
            }
            return name;
        }
        /// <summary>
        /// The search method
        /// </summary>
        public CBSMethod.CBSSearchMethod SearchMethod = CBSMethod.CBSSearchMethod.BestFirst;
        /// <summary>
        /// Parses the specified arguments.
        /// </summary>
        /// <param name="args">The arguments.</param>
        public override void Parse(string[] args)
        {
            base.Parse(args);
            SearchMethod = (CBSMethod.CBSSearchMethod)Enum.Parse(typeof(CBSMethod.CBSSearchMethod), (args[2]));
        }
    }
    /// <summary>
    /// The configuration for the corresponding method.
    /// </summary>
    public class BCPPathPlanningConfiguration : PathPlanningConfiguration
    {
        /// <summary>
        /// Returns the type of the corresponding method this configuration belongs to.
        /// </summary>
        /// <returns>The type of the method.</returns>
        public override PathPlanningMethodType GetMethodType() { return PathPlanningMethodType.BCP; }
        /// <summary>
        /// Returns a name identifying the method.
        /// </summary>
        /// <returns>The name of the method.</returns>
        public override string GetMethodName() { if (!string.IsNullOrWhiteSpace(Name)) return Name; return "ppBCP" + BiasedCostAmount.ToString(IOConstants.EXPORT_FORMAT_SHORTER, IOConstants.FORMATTER); }
        /// <summary>
        /// amount of biased cost in seconds
        /// </summary>
        public double BiasedCostAmount = 1;
        /// <summary>
        /// Parses the specified arguments.
        /// </summary>
        /// <param name="args">The arguments.</param>
        public override void Parse(string[] args)
        {
            base.Parse(args);
            BiasedCostAmount = double.Parse(args[2], new CultureInfo("en"));
        }
        /// <summary>
        /// Checks whether the path planning configuration is valid.
        /// </summary>
        /// <param name="errorMessage">A message describing the error if the configuration is not valid.</param>
        /// <returns>Indicates whether the path planning configuration is valid.</returns>
        public override bool AttributesAreValid(out String errorMessage)
        {
            if (!base.AttributesAreValid(out errorMessage))
            {
                return false;
            }
            if (BiasedCostAmount < 0)       //<= 0 my be wise as well
            {
                errorMessage = "Problem with path planning configuration: BiasedCostAmount has to be >= 0.";
                return false;
            }
            errorMessage = "";
            return true;

        }

    }
    /// <summary>
    /// The configuration for the corresponding method.
    /// </summary>
    public class PASPathPlanningConfiguration : PathPlanningConfiguration
    {
        /// <summary>
        /// Returns the type of the corresponding method this configuration belongs to.
        /// </summary>
        /// <returns>The type of the method.</returns>
        public override PathPlanningMethodType GetMethodType() { return PathPlanningMethodType.PAS; }
        /// <summary>
        /// Returns a name identifying the method.
        /// </summary>
        /// <returns>The name of the method.</returns>
        public override string GetMethodName() { if (!string.IsNullOrWhiteSpace(Name)) return Name; return "ppPAS" + MaxPriorities.ToString() + LengthOfAWindow.ToString(IOConstants.EXPORT_FORMAT_SHORTER, IOConstants.FORMATTER); }
        /// <summary>
        /// The maximum number of priorities
        /// </summary>
        public int MaxPriorities = 2;
        /// <summary>
        /// The length of a time space search window
        /// </summary>
        public double LengthOfAWindow = 30.0;
        /// <summary>
        /// Indicates whether the method uses a deadlock handler. PAS resolves conflicts purely
        /// by priority and time, so without a handler a ring of mutually waiting agents has no
        /// escape. Exposed rather than hard-coded so the effect can be ablated.
        /// </summary>
        public bool UseDeadlockHandler = true;
        /// <summary>
        /// Parses the specified arguments.
        /// </summary>
        /// <param name="args">The arguments.</param>
        public override void Parse(string[] args)
        {
            base.Parse(args);
            MaxPriorities = int.Parse(args[2]);
            LengthOfAWindow = double.Parse(args[3], new CultureInfo("en"));
        }
        /// <summary>
        /// Checks whether the path planning configuration is valid.
        /// </summary>
        /// <param name="errorMessage">A message describing the error if the configuration is not valid.</param>
        /// <returns>Indicates whether the path planning configuration is valid.</returns>
        public override bool AttributesAreValid(out String errorMessage)
        {
            if (!base.AttributesAreValid(out errorMessage))
            {
                return false;
            }
            if (MaxPriorities < 0)      //<= 0 my be wise as well
            {
                errorMessage = "Problem with path planning configuration: MaxPriorities has to be >= 0.";
                return false;
            }
            if (LengthOfAWindow <= 0)
            {
                errorMessage = "Problem with path planning configuration: LengthOfAWindow has to be > 0";
                return false;
            }

            errorMessage = "";
            return true;

        }
    }
    /// <summary>
    /// The configuration for the corresponding method.
    /// </summary>
    public class ODIDPathPlanningConfiguration : PathPlanningConfiguration
    {
        /// <summary>
        /// Returns the type of the corresponding method this configuration belongs to.
        /// </summary>
        /// <returns>The type of the method.</returns>
        public override PathPlanningMethodType GetMethodType() { return PathPlanningMethodType.OD_ID; }
        /// <summary>
        /// Returns a name identifying the method.
        /// </summary>
        /// <returns>The name of the method.</returns>
        public override string GetMethodName() { if (!string.IsNullOrWhiteSpace(Name)) return Name; return "ppODID" + LengthOfAWindow.ToString(IOConstants.EXPORT_FORMAT_SHORTER, IOConstants.FORMATTER) + (UseFinalReservations ? "t" : "f") + MaxNodeCountPerAgent.ToString(); }
        /// <summary>
        /// The length of a time space search window
        /// </summary>
        public double LengthOfAWindow = 30.0;
        /// <summary>
        /// Use Final Reservations
        /// </summary>
        public bool UseFinalReservations = false;
        /// <summary>
        /// Use Final Reservations
        /// </summary>
        public int MaxNodeCountPerAgent = 100;
        /// <summary>
        /// Parses the specified arguments.
        /// </summary>
        /// <param name="args">The arguments.</param>
        public override void Parse(string[] args)
        {
            base.Parse(args);
            LengthOfAWindow = double.Parse(args[2], new CultureInfo("en"));
            MaxNodeCountPerAgent = int.Parse(args[3]);
            UseFinalReservations = bool.Parse(args[4]);
        }
        /// <summary>
        /// Checks whether the path planning configuration is valid.
        /// </summary>
        /// <param name="errorMessage">A message describing the error if the configuration is not valid.</param>
        /// <returns>Indicates whether the path planning configuration is valid.</returns>
        public override bool AttributesAreValid(out String errorMessage)
        {
            if (!base.AttributesAreValid(out errorMessage))
            {
                return false;
            }
            if (MaxNodeCountPerAgent < 0)       //<= 0 may be wise as well
            {
                errorMessage = "Problem with path planning configuration: MaxPriorities has to be >= 0.";
                return false;
            }
            if (LengthOfAWindow <= 0)
            {
                errorMessage = "LengthOfAWindow has to be > 0";
                return false;
            }

            errorMessage = "";
            return true;

        }
    }

    #endregion
}
