// Copyright 2026 AMR project. SPDX-License-Identifier: Apache-2.0
// Experimental DWB tie-breaker: retain every upstream collision/path critic.
#include <dwb_core/trajectory_critic.hpp>
#include <dwb_core/exceptions.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <cmath>

namespace amr_execution
{
class ContinuousGoalDistCritic : public dwb_core::TrajectoryCritic
{
public:
  bool prepare(
    const geometry_msgs::msg::Pose2D &, const nav_2d_msgs::msg::Twist2D &,
    const geometry_msgs::msg::Pose2D & goal, const nav_2d_msgs::msg::Path2D &) override
  {
    valid_ = std::isfinite(goal.x) && std::isfinite(goal.y);
    goal_ = goal;
    return valid_;
  }

  double scoreTrajectory(const dwb_msgs::msg::Trajectory2D & trajectory) override
  {
    if (!valid_ || trajectory.poses.empty()) {
      throw dwb_core::IllegalTrajectoryException(name_, "Missing finite goal or predicted poses");
    }
    const auto & end = trajectory.poses.back();
    const double distance = std::hypot(end.x - goal_.x, end.y - goal_.y);
    if (!std::isfinite(distance)) {
      throw dwb_core::IllegalTrajectoryException(name_, "Non-finite predicted goal distance");
    }
    return distance;
  }

private:
  geometry_msgs::msg::Pose2D goal_;
  bool valid_{false};
};
}  // namespace amr_execution
PLUGINLIB_EXPORT_CLASS(amr_execution::ContinuousGoalDistCritic, dwb_core::TrajectoryCritic)
