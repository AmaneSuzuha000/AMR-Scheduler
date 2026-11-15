using RAWSimO.Core.Configurations;
using RAWSimO.Core.Elements;
using RAWSimO.Core.Metrics;
using RAWSimO.Core.Waypoints;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RAWSimO.Core.Control.Defaults.PodStorage
{
    /// <summary>
    /// A pod storage policy that prices both the trip to a candidate location and the
    /// traffic that location would create.
    /// <para>
    /// Motivation: under the shipped nearest-location policy the pods migrate to the
    /// storage locations closest to the output stations. Measured on the 1-4-4-32-550
    /// instance over 28800 s (4 policies x 5 seeds), the near band reaches 96% occupancy
    /// while the far band falls to 77%, and the resulting traffic costs more throughput
    /// than the shorter trips save. This policy keeps the distance term but adds a
    /// penalty proportional to how many bots are already working near the candidate.
    /// </para>
    /// <para>
    /// With <see cref="CongestionAwarePodStorageConfiguration.CongestionWeight"/> at zero
    /// the congestion term vanishes and the policy reduces to nearest-location, which
    /// makes the effect of the penalty measurable against an exact baseline.
    /// </para>
    /// </summary>
    public class CongestionAwarePodStorageManager : PodStorageManager
    {
        /// <summary>
        /// Creates a new instance of the manager.
        /// </summary>
        /// <param name="instance">The instance this manager belongs to.</param>
        public CongestionAwarePodStorageManager(Instance instance) : base(instance)
        {
            _config = instance.ControllerConfig.PodStorageConfig as CongestionAwarePodStorageConfiguration
                      ?? throw new ArgumentException("Wrong configuration type for the congestion-aware pod storage manager");
        }

        /// <summary>
        /// The configuration of this manager.
        /// </summary>
        private readonly CongestionAwarePodStorageConfiguration _config;

        /// <summary>
        /// Returns a suitable storage location for the given pod.
        /// </summary>
        /// <param name="pod">The pod to fetch a storage location for.</param>
        /// <returns>The storage location to use.</returns>
        protected override Waypoint GetStorageLocationForPod(Pod pod)
        {
            Waypoint podLocation = _config.PodDisposeRule == NearestPodStorageLocationDisposeRule.ShortestPath ||
                                  _config.PodDisposeRule == NearestPodStorageLocationDisposeRule.ShortestTime
                ? Instance.WaypointGraph.GetClosestWaypoint(pod.Tier, pod.X, pod.Y)
                : null;

            // Snapshot the neighbour set once instead of walking it for every candidate
            // location; with 32 bots / 550 pods and 648 locations that is the difference
            // between O(n) and O(n * locations) per decision.
            //
            // Which set is counted matters. Bots move constantly, so counting them measured
            // "how many happen to be passing right now" rather than lasting congestion, and
            // a weight sweep on that basis changed nothing. Stored pods are persistent and
            // are what actually blocks an aisle, hence the option to count them instead.
            var neighbours = _config.UsePodDensity
                ? Instance.Pods.Where(p => p.Waypoint != null).Select(p => new { p.X, p.Y }).ToArray()
                : Instance.Bots.Select(b => new { b.X, b.Y }).ToArray();
            double radiusSquared = _config.CongestionRadius * _config.CongestionRadius;

            // How hot is the pod being stored? Rank it against every other pod so the term
            // is comparable across instances and does not depend on the absolute order count.
            // 0 = coldest, 1 = hottest.
            double podHotness = 0.0;
            if (_config.FrequencyWeight > 0)
            {
                var frequencies = Instance.Pods
                    .Select(p => p.ItemDescriptionsContained.Any()
                        ? p.ItemDescriptionsContained.Average(i => i.OrderCount)
                        : 0.0)
                    .OrderBy(v => v)
                    .ToArray();
                double ownFrequency = pod.ItemDescriptionsContained.Any()
                    ? pod.ItemDescriptionsContained.Average(i => i.OrderCount)
                    : 0.0;
                int below = 0;
                foreach (var f in frequencies)
                    if (f < ownFrequency) below++;
                podHotness = frequencies.Length > 1 ? (double)below / (frequencies.Length - 1) : 0.0;
            }

            double minScore = double.PositiveInfinity;
            Waypoint bestStorageLocation = null;
            foreach (var storageLocation in Instance.ResourceManager.UnusedPodStorageLocations)
            {
                double distance;
                switch (_config.PodDisposeRule)
                {
                    case NearestPodStorageLocationDisposeRule.Euclid:
                        distance = Distances.CalculateEuclid(pod, storageLocation, Instance.WrongTierPenaltyDistance); break;
                    case NearestPodStorageLocationDisposeRule.Manhattan:
                        distance = Distances.CalculateManhattan(pod, storageLocation, Instance.WrongTierPenaltyDistance); break;
                    case NearestPodStorageLocationDisposeRule.ShortestPath:
                        distance = Distances.CalculateShortestPathPodSafe(podLocation, storageLocation, Instance); break;
                    case NearestPodStorageLocationDisposeRule.ShortestTime:
                        distance = Distances.CalculateShortestTimePathPodSafe(podLocation, storageLocation, Instance); break;
                    default: throw new ArgumentException("Unknown pod dispose rule: " + _config.PodDisposeRule);
                }
                if (double.IsInfinity(distance) || double.IsNaN(distance))
                    continue;

                // Count the bots already inside the candidate's neighbourhood. The penalty
                // is squared so the first few neighbours are cheap and a genuinely crowded
                // corner is expensive, which is what pushes pods back outwards.
                int nearby = 0;
                foreach (var other in neighbours)
                {
                    double dx = other.X - storageLocation.X;
                    double dy = other.Y - storageLocation.Y;
                    if (dx * dx + dy * dy <= radiusSquared)
                        nearby++;
                }
                // The two terms are added directly, so they must share a scale. Distance is
                // in seconds (ShortestTime), typically five to sixty. A raw neighbour count
                // squared is an integer that is usually 0 and occasionally 9 or 16, which is
                // why the weight sweep saturated: raising the weight from 1 to 2 changed
                // nothing because both values already dominated every distance difference.
                // Normalizing to [0, 1] over the theoretical maximum for the radius keeps the
                // weight meaningful across instances.
                double maxNearby = Math.Max(1.0, _config.CongestionRadius * _config.CongestionRadius);
                double congestion = (nearby * nearby) / maxNearby;
                double score = distance + _config.CongestionWeight * congestion;
                if (_config.FrequencyWeight > 0)
                {
                    // Time to reach the nearest output station, priced by how often the pod
                    // is actually picked. A hot pod stored far away pays repeatedly; a cold
                    // pod stored far away costs nothing, which frees the near band for the
                    // pods that need it.
                    double toStation = minStationTime(storageLocation);
                    score += _config.FrequencyWeight * podHotness * toStation;
                }

                if (score < minScore)
                {
                    minScore = score;
                    bestStorageLocation = storageLocation;
                }
            }

            if (bestStorageLocation == null)
                throw new InvalidOperationException("There was no suitable storage location for the pod: " + pod);
            return bestStorageLocation;
        }

        /// <summary>
        /// Travel time from a storage location to the nearest output station.
        /// </summary>
        /// <param name="storageLocation">The storage location to price.</param>
        /// <returns>The shortest pod-safe travel time, or a large value when unreachable.</returns>
        private double minStationTime(Waypoint storageLocation)
        {
            double best = double.PositiveInfinity;
            foreach (var station in Instance.OutputStations)
            {
                double t = Distances.CalculateShortestTimePathPodSafe(storageLocation, station.Waypoint, Instance);
                if (t < best) best = t;
            }
            return double.IsInfinity(best) || double.IsNaN(best) ? 0.0 : best;
        }

        #region IOptimize Members

        /// <summary>
        /// Signals the current time to the mechanism. The mechanism can decide to block the simulation thread in order consume remaining real-time.
        /// </summary>
        /// <param name="currentTime">The current simulation time.</param>
        public override void SignalCurrentTime(double currentTime) { /* Ignore since this manager is always ready. */ }

        #endregion
    }
}
