// Copyright 2026 AMR project. SPDX-License-Identifier: Apache-2.0
// The existing differential-drive plugin remains the wheel controller. Only
// this plugin publishes to its remapped input. No task or route planning here.
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <amr_gazebo_guard/msg/motion_lease.hpp>
#include <amr_gazebo_guard/msg/guard_status.hpp>
#include <amr_gazebo_guard/srv/authorize_motion.hpp>
#include "motion_watchdog.hpp"
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace amr_gazebo_guard
{
class MotionGuard : public gazebo::ModelPlugin
{
  using Steady = std::chrono::steady_clock;
  using Point = Steady::time_point;
  using Lease = msg::MotionLease;
  using Status = msg::GuardStatus;
  using Authorize = srv::AuthorizeMotion;

  static double Age(Point now, Point then)
  {
    return std::chrono::duration<double>(now - then).count();
  }

  static std::string NewId()
  {
    std::random_device random;
    std::ostringstream id;
    id << std::hex << random() << random() << random() << random();
    return id.str();
  }

  void Latch(const std::string & reason)
  {
    armed_ = false;
    reason_ = reason;
    target_ = geometry_msgs::msg::Twist();
    // This publishes immediately as well as at every subsequent physics tick.
    output_->publish(target_);
  }

  void CheckTimeout(Point now)
  {
    if (!armed_) {return;}
    amr_execution::WatchdogInput input;
    input.armed = armed_;
    input.have_command = have_command_;
    input.have_lease = have_lease_;
    input.command_age = Age(now, have_command_ ? last_command_ : armed_at_);
    input.lease_age = Age(now, have_lease_ ? last_lease_ : armed_at_);
    input.park_held_s = park_since_ == Point() ? 0. : Age(now, park_since_);
    const auto decision = amr_execution::EvaluateWatchdog(limits_, input);
    if (decision.latch) {Latch(decision.reason);}
  }

public:
  ~MotionGuard() override
  {
    update_.reset();
    command_.reset();
    lease_.reset();
    authorize_.reset();
    node_.reset();
  }

  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = model;
    node_ = gazebo_ros::Node::Get(sdf);
    limits_.command_timeout = sdf->Get<double>("command_timeout", .25).first;
    limits_.lease_timeout = sdf->Get<double>("lease_timeout", .5).first;
    limits_.initial_command_timeout = sdf->Get<double>("initial_command_timeout", 1.).first;
    limits_.park_speed = sdf->Get<double>("park_speed", .05).first;
    limits_.park_wz = sdf->Get<double>("park_wz", .1).first;
    limits_.park_confirm = sdf->Get<double>("park_confirm", .05).first;
    limits_.park_grace = sdf->Get<double>("park_grace", 1.5).first;
    max_linear_ = sdf->Get<double>("max_linear", 1.).first;
    max_angular_ = sdf->Get<double>("max_angular", 1.).first;
    if (!(std::isfinite(limits_.command_timeout) && std::isfinite(limits_.lease_timeout) &&
      std::isfinite(limits_.initial_command_timeout) && std::isfinite(limits_.park_speed) &&
      std::isfinite(limits_.park_wz) && std::isfinite(limits_.park_confirm) &&
      std::isfinite(limits_.park_grace) && std::isfinite(max_linear_) && std::isfinite(max_angular_) &&
      limits_.command_timeout > 0 && limits_.lease_timeout > 0 &&
      limits_.initial_command_timeout > 0 && limits_.park_speed > 0 && limits_.park_wz > 0 &&
      limits_.park_confirm > 0 && limits_.park_grace > 0 && max_linear_ > 0 && max_angular_ > 0))
    {
      throw std::runtime_error("guard limits must be positive");
    }
    guard_id_ = NewId();
    output_ = node_->create_publisher<geometry_msgs::msg::Twist>("guarded_cmd_vel", rclcpp::QoS(1));
    status_ = node_->create_publisher<Status>("motion_guard/status", rclcpp::QoS(1));
    command_ = node_->create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", rclcpp::QoS(1), [this](geometry_msgs::msg::Twist::ConstSharedPtr cmd) {
        std::lock_guard<std::mutex> lock(mutex_);
        CheckTimeout(Steady::now());
        if (!armed_) {return;}
        if (!std::isfinite(cmd->linear.x) || !std::isfinite(cmd->angular.z) ||
          cmd->linear.y != 0. || cmd->linear.z != 0. || cmd->angular.x != 0. || cmd->angular.y != 0. ||
          std::abs(cmd->linear.x) > max_linear_ || std::abs(cmd->angular.z) > max_angular_)
        {
          Latch("invalid command");
          return;
        }
        target_ = *cmd;
        last_command_ = Steady::now();
        have_command_ = true;
      });
    lease_ = node_->create_subscription<Lease>(
      "motion_guard/lease", rclcpp::QoS(1), [this](Lease::ConstSharedPtr lease) {
        std::lock_guard<std::mutex> lock(mutex_);
        CheckTimeout(Steady::now());
        if (!armed_ || lease->guard_id != guard_id_ || lease->run_id != run_id_ ||
          lease->route_id != route_id_ || lease->revision != revision_ || lease->seq <= lease_seq_)
        {
          return;
        }
        const double stamp = lease->header.stamp.sec + lease->header.stamp.nanosec * 1e-9;
        const double sim_age = sim_time_ - stamp;
        // Queued old heartbeats must not revive a permit after a network stall.
        if (!std::isfinite(stamp) || sim_age < -.1 || sim_age > limits_.lease_timeout) {return;}
        lease_seq_ = lease->seq;
        last_lease_ = Steady::now();
        have_lease_ = true;
      });
    authorize_ = node_->create_service<Authorize>("motion_guard/authorize",
      [this](const Authorize::Request::SharedPtr req, Authorize::Response::SharedPtr response) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto reject = [&](const std::string & reason) {response->accepted = false; response->reason = reason;};
        if (req->guard_id != guard_id_) {reject("guard instance mismatch"); return;}
        if (!req->enable) {
          if (req->run_id != run_id_ || req->route_id != route_id_ || req->revision != revision_) {
            reject("action identity mismatch"); return;
          }
          Latch("explicit stop");
          response->accepted = true;
          response->reason = reason_;
          return;
        }
        if (req->run_id.empty() || req->route_id.empty() || req->revision == 0 ||
          (!run_id_.empty() && req->run_id != run_id_) || req->revision <= revision_)
        {
          reject("new action revision for the same owner required"); return;
        }
        if (armed_ || !stopped_ || Age(Steady::now(), last_update_) > .1) {
          reject("fresh confirmed stop required"); return;
        }
        run_id_ = req->run_id;
        route_id_ = req->route_id;
        revision_ = req->revision;
        lease_seq_ = 0;
        have_command_ = have_lease_ = false;
        target_ = geometry_msgs::msg::Twist();
        armed_at_ = Steady::now();
        armed_ = true;
        stopped_ = false;
        stop_since_ = Point();
        park_since_ = Point();
        reason_.clear();
        response->accepted = true;
        response->reason = "authorized; awaiting fresh lease and command";
      });
    update_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&MotionGuard::Update, this));
  }

  void Reset() override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!output_) {return;}
    Latch("world reset");
    guard_id_ = NewId();
    // The new guard identity invalidates all queued authorizations and leases.
    run_id_.clear();
    route_id_.clear();
    revision_ = 0;
    stopped_ = false;
    stop_since_ = Point();
    park_since_ = Point();
  }

private:
  void Update()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto now = Steady::now();
    last_update_ = now;
    sim_time_ = model_->GetWorld()->SimTime().Double();
    const auto v = model_->WorldLinearVel();
    const double speed = std::hypot(v.X(), v.Y());
    const double wz = model_->WorldAngularVel().Z();
    // Park evidence is refreshed on every physics step, before the watchdog is
    // evaluated, so a silent controller that already commanded zero is not
    // mistaken for a live one when the action result is still in flight.
    const bool parked_now = armed_ && have_command_ && have_lease_ &&
      amr_execution::ParkedTarget(limits_, target_.linear.x, target_.angular.z, speed, wz);
    if (parked_now) {
      if (park_since_ == Point()) {park_since_ = now;}
    } else {
      park_since_ = Point();
    }
    CheckTimeout(now);
    geometry_msgs::msg::Twist output;
    if (armed_ && have_command_ && have_lease_) {output = target_;}
    // Publish on wall intervals. During a paused world no physics advances;
    // the first resumed update checks wall-time expiry before forwarding.
    if (last_publish_ == Point() || Age(now, last_publish_) >= .01) {
      output_->publish(output);
      last_publish_ = now;
    }
    if (output.linear.x == 0. && output.angular.z == 0. && speed < .01 && std::abs(wz) < .02) {
      if (stop_since_ == Point()) {stop_since_ = now; stop_sim_since_ = sim_time_;}
      stopped_ = Age(now, stop_since_) >= .3 && sim_time_ - stop_sim_since_ >= .3;
    } else {
      stop_since_ = Point();
      stopped_ = false;
    }
    if (last_status_ == Point() || Age(now, last_status_) >= .05) {
      Status status;
      status.header.stamp = rclcpp::Time(static_cast<int64_t>(sim_time_ * 1e9));
      status.header.frame_id = "world";
      status.guard_id = guard_id_;
      status.run_id = run_id_;
      status.route_id = route_id_;
      status.revision = revision_;
      status.state = armed_ ? "armed" : "latched";
      status.reason = reason_;
      status.stopped = stopped_;
      status.actual_speed = speed;
      status.actual_wz = wz;
      status.command_age = have_command_ ? Age(now, last_command_) : -1.;
      status.lease_age = have_lease_ ? Age(now, last_lease_) : -1.;
      status.output_vx = output.linear.x;
      status.output_wz = output.angular.z;
      status_->publish(status);
      last_status_ = now;
    }
  }

  gazebo::physics::ModelPtr model_;
  gazebo_ros::Node::SharedPtr node_;
  gazebo::event::ConnectionPtr update_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr output_;
  rclcpp::Publisher<Status>::SharedPtr status_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr command_;
  rclcpp::Subscription<Lease>::SharedPtr lease_;
  rclcpp::Service<Authorize>::SharedPtr authorize_;
  std::mutex mutex_;
  std::string guard_id_, run_id_, route_id_, reason_ = "startup locked";
  uint64_t revision_ = 0, lease_seq_ = 0;
  bool armed_ = false, stopped_ = false, have_command_ = false, have_lease_ = false;
  amr_execution::WatchdogLimits limits_{};
  double max_linear_, max_angular_;
  double sim_time_ = 0.;
  double stop_sim_since_ = 0.;
  geometry_msgs::msg::Twist target_;
  Point armed_at_, last_command_, last_lease_, stop_since_, park_since_, last_publish_, last_status_, last_update_;
};
GZ_REGISTER_MODEL_PLUGIN(MotionGuard)
}  // namespace amr_gazebo_guard
