// Check the installed exported plugin, including the sub-cell deadband case.
#include <dwb_core/trajectory_critic.hpp>
#include <dwb_core/exceptions.hpp>
#include <pluginlib/class_loader.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

int main()
{
  pluginlib::ClassLoader<dwb_core::TrajectoryCritic> loader("dwb_core", "dwb_core::TrajectoryCritic");
  auto critic = loader.createSharedInstance("amr_execution::ContinuousGoalDistCritic");
  geometry_msgs::msg::Pose2D goal, current;
  nav_2d_msgs::msg::Twist2D velocity;
  nav_2d_msgs::msg::Path2D path;
  goal.x = 2.0319755;
  goal.y = 1.4;
  if (!critic->prepare(current, velocity, goal, path)) {throw std::runtime_error("Goal rejected");}
  double previous = std::numeric_limits<double>::infinity();
  // All are in one five-centimetre cell: native cell distance can be tied.
  for (double remaining : {.049, .030, .015, .005, 0.}) {
    dwb_msgs::msg::Trajectory2D trajectory;
    auto end = goal;
    end.x -= remaining;
    trajectory.poses.push_back(end);
    const auto score = critic->scoreTrajectory(trajectory);
    if (!(score < previous)) {throw std::runtime_error("Sub-cell progress has no reward");}
    previous = score;
  }
  dwb_msgs::msg::Trajectory2D empty;
  bool rejected = false;
  try {critic->scoreTrajectory(empty);}
  catch (const dwb_core::IllegalTrajectoryException &) {rejected = true;}
  if (!rejected) {throw std::runtime_error("Empty trajectory accepted");}
  goal.x = std::numeric_limits<double>::quiet_NaN();
  if (critic->prepare(current, velocity, goal, path)) {throw std::runtime_error("NaN goal accepted");}
  std::cout << "PASS installed plugin: sub-cell progress, empty trajectory and non-finite goal\n";
}
