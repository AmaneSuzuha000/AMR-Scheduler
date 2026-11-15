// Apply bounded forces every physics step, independently of ROS service RTT.
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

namespace wheelset_axis_servo {
class AxisServo : public gazebo::ModelPlugin {
  struct Axis {
    gazebo::physics::JointPtr joint;
    std::string mode;
    double mass, limit, damping, target{0.}, force{0.};
    double integral_gain{0.}, integral_force{0.};
    bool enabled{true};
  };
  gazebo::physics::ModelPtr model_;
  gazebo_ros::Node::SharedPtr node_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
  gazebo::event::ConnectionPtr connection_;
  std::vector<Axis> axes_;
  std::mutex mutex_;
  double sequence_{-1.}, received_{0.}, published_{-1.}, lease_{1.}, updated_{-1.};
  bool pending_{false}, armed_{false}, fault_{false};

  void command(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    if (msg->data.size()!=1+2*axes_.size()) return;
    for (double v:msg->data) if (!std::isfinite(v)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (fault_ || msg->data[0]<=sequence_) return;
    for (size_t i=0;i<axes_.size();++i) {
      const double enable=msg->data[2+2*i];
      if (enable!=0. && enable!=1.) return;
      const auto &axis=axes_[i]; const double target=msg->data[1+2*i];
      if (axis.mode!="velocity" &&
          (target<axis.joint->LowerLimit(0) || target>axis.joint->UpperLimit(0))) return;
      if (axis.mode=="velocity" && std::abs(target)>2.5) return;
    }
    sequence_=msg->data[0];pending_=true;armed_=true;
    for (size_t i=0;i<axes_.size();++i) {
      axes_[i].target=msg->data[1+2*i];axes_[i].enabled=msg->data[2+2*i]!=0.;
    }
  }

  void update(const gazebo::common::UpdateInfo &info) {
    std::lock_guard<std::mutex> lock(mutex_);
    const double now=info.simTime.Double();
    const double dt=updated_<0. ? 0. : std::clamp(now-updated_,0.,.02);
    updated_=now;
    if (pending_) { received_=now;pending_=false; }
    if (armed_ && !fault_ && now-received_>lease_) {
      fault_=true;
      for (auto &a:axes_) {
        a.target=a.mode=="velocity" ? 0. : a.joint->Position(0);
        a.integral_force=0.;
        a.enabled=true;
      }
    }
    for (auto &a:axes_) {
      const double p=a.joint->Position(0),v=a.joint->GetVelocity(0);
      if (a.mode=="velocity") a.force=a.mass*8.*(a.target-v)+a.damping*v;
      else if (a.mode=="probe") {
        const double error=a.target-p;
        const double feedback=a.mass*(100.*error-18.*v)+a.damping*v;
        if ((a.target<=a.joint->LowerLimit(0) && p<=a.joint->LowerLimit(0) && a.integral_force<0.) ||
            (a.target>=a.joint->UpperLimit(0) && p>=a.joint->UpperLimit(0) && a.integral_force>0.))
          a.integral_force=0.;
        if (!a.enabled || fault_) a.integral_force=0.;
        else {
          const double candidate=std::clamp(a.integral_force+a.integral_gain*error*dt,-a.limit,a.limit);
          // Conditional integration prevents further wind-up under saturation,
          // while permitting an opposite error to unwind accumulated force.
          if (std::abs(feedback+candidate)<=a.limit || (feedback+candidate)*error<0.)
            a.integral_force=candidate;
        }
        a.force=feedback+a.integral_force;
      }
      else {
        const double velocity=std::clamp(.8*(a.target-p),-1.2,1.2);
        a.force=a.mass*6.*(velocity-v)+a.damping*v;
      }
      a.force=a.enabled ? std::clamp(a.force,-a.limit,a.limit) : 0.;
      a.joint->SetForce(0,a.force);
    }
    if (now-published_<.025) return;
    sensor_msgs::msg::JointState state;
    state.header.stamp=rclcpp::Time(info.simTime.sec,info.simTime.nsec);
    for (const auto &a:axes_) {
      state.name.push_back(a.joint->GetName());state.position.push_back(a.joint->Position(0));
      state.velocity.push_back(a.joint->GetVelocity(0));state.effort.push_back(a.force);
    }
    state.name.push_back(model_->GetName()+"_servo_sequence");state.position.push_back(sequence_);
    state.velocity.push_back(0.);state.effort.push_back(0.);
    state.name.push_back(model_->GetName()+"_servo_fault");state.position.push_back(fault_ ? 1. : 0.);
    state.velocity.push_back(0.);state.effort.push_back(0.);
    publisher_->publish(state);published_=now;
  }

public:
  void Load(gazebo::physics::ModelPtr model,sdf::ElementPtr sdf) override {
    model_=model;node_=gazebo_ros::Node::Get(sdf);lease_=sdf->Get<double>("lease_s");
    if (!std::isfinite(lease_) || lease_<=0.) throw std::runtime_error("Invalid command lease");
    for (auto element=sdf->GetElement("axis");element;element=element->GetNextElement("axis")) {
      Axis a;a.joint=model_->GetJoint(element->Get<std::string>("joint"));
      if (!a.joint) throw std::runtime_error("Missing servo joint");
      a.mode=element->Get<std::string>("mode");a.mass=element->Get<double>("mass");
      a.limit=element->Get<double>("limit");a.damping=element->Get<double>("damping");
      if (element->HasElement("integral_gain_n_per_m_s"))
        a.integral_gain=element->Get<double>("integral_gain_n_per_m_s");
      if (a.mode!="position" && a.mode!="velocity" && a.mode!="probe") throw std::runtime_error("Invalid axis mode");
      if (!(std::isfinite(a.mass) && std::isfinite(a.limit) && std::isfinite(a.damping) &&
            std::isfinite(a.integral_gain) && a.mass>0. && a.limit>0. && a.damping>=0. && a.integral_gain>=0.))
        throw std::runtime_error("Invalid servo bounds");
      if (a.mode!="probe" && a.integral_gain!=0.) throw std::runtime_error("Integral force is probe-only");
      a.target=a.joint->Position(0);axes_.push_back(a);
    }
    subscription_=node_->create_subscription<std_msgs::msg::Float64MultiArray>("servo_targets",rclcpp::QoS(1),
      std::bind(&AxisServo::command,this,std::placeholders::_1));
    publisher_=node_->create_publisher<sensor_msgs::msg::JointState>("joint_states",rclcpp::SensorDataQoS());
    connection_=gazebo::event::Events::ConnectWorldUpdateBegin(std::bind(&AxisServo::update,this,std::placeholders::_1));
  }
};
GZ_REGISTER_MODEL_PLUGIN(AxisServo)
}  // namespace wheelset_axis_servo
