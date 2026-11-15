// Copyright 2026 AMR project. SPDX-License-Identifier: Apache-2.0
#include <nav2_regulated_pure_pursuit_controller/regulated_pure_pursuit_controller.hpp>
#include <nav2_core/exceptions.hpp>
#include <pluginlib/class_list_macros.hpp>
#include "twist_bounds.hpp"

namespace amr_execution
{
// Keep upstream Nav2 tracking/collision checks and Python's original global route.
// Humble RPP's rotation-to-heading speed does not cap its curvature commands.
class BoundedPursuit : public nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController
{
public:
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent, std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap) override
  {
    RegulatedPurePursuitController::configure(parent, name, tf, costmap);
    const auto node = parent.lock();
    if (!node) {throw std::runtime_error("controller node expired");}
    const auto linear_name = name + ".max_command_linear";
    const auto angular_name = name + ".max_command_angular";
    if (!node->has_parameter(linear_name)) {node->declare_parameter(linear_name, .2);}
    if (!node->has_parameter(angular_name)) {node->declare_parameter(angular_name, .8);}
    max_linear_ = node->get_parameter(linear_name).as_double();
    max_angular_ = node->get_parameter(angular_name).as_double();
    BoundTwist(0., 0., max_linear_, max_angular_); // Reject invalid configuration before activation.
    // Limits remain the startup contract. Do not accept parameter changes that would
    // make the reported values disagree with the bounds actually applied.
    bound_parameters_ = node->add_on_set_parameters_callback(
      [linear_name, angular_name](const std::vector<rclcpp::Parameter> & parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto & parameter : parameters) {
          if (parameter.get_name() == linear_name || parameter.get_name() == angular_name) {
            result.successful = false;
            result.reason = "Execution velocity bounds require controller reconfiguration.";
          }
        }
        return result;
      });
  }

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity, nav2_core::GoalChecker * checker) override
  {
    auto command = RegulatedPurePursuitController::computeVelocityCommands(pose, velocity, checker);
    const auto raw_linear = command.twist.linear.x;
    const auto raw_angular = command.twist.angular.z;
    BoundedTwist bounded;
    try {bounded = BoundTwist(raw_linear, raw_angular, max_linear_, max_angular_);}
    catch (const std::invalid_argument & error) {throw nav2_core::PlannerException(error.what());}
    command.twist.linear.x = bounded.linear;
    command.twist.angular.z = bounded.angular;
    if (bounded.scale < 1.) {
      RCLCPP_WARN_THROTTLE(logger_, *clock_, 2000,
        "Execution twist scaled: raw=(%.9f,%.9f) bounded=(%.9f,%.9f) scale=%.9f",
        raw_linear, raw_angular, bounded.linear, bounded.angular, bounded.scale);
    }
    return command;
  }

  void cleanup() override
  {
    bound_parameters_.reset();
    RegulatedPurePursuitController::cleanup();
  }

private:
  double max_linear_{.2};
  double max_angular_{.8};
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr bound_parameters_;
};
}  // namespace amr_execution
PLUGINLIB_EXPORT_CLASS(amr_execution::BoundedPursuit, nav2_core::Controller)
