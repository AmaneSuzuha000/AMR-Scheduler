// Read both actual model bodies at the end of the same physics step.
// No velocity commands, state setters, scheduler or deadline policy here.
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <sensor_msgs/msg/multi_dof_joint_state.hpp>

namespace wheelset_axis_servo {
class FleetTruth : public gazebo::WorldPlugin {
  gazebo::physics::WorldPtr world_;
  gazebo_ros::Node::SharedPtr node_;
  rclcpp::Publisher<sensor_msgs::msg::MultiDOFJointState>::SharedPtr publisher_;
  gazebo::event::ConnectionPtr connection_;
  std::vector<std::string> names_{"scout_a", "scout_b"};
  double published_{-1.};

  void update() {
    const auto now=world_->SimTime();
    if (now.Double()-published_<.025) return;
    std::vector<gazebo::physics::ModelPtr> models;
    for (const auto &name:names_) {
      auto model=world_->ModelByName(name);
      if (!model) return;
      models.push_back(model);
    }
    sensor_msgs::msg::MultiDOFJointState message;
    message.header.stamp=rclcpp::Time(now.sec,now.nsec);
    message.header.frame_id="wheelset_world";
    for (size_t i=0;i<models.size();++i) {
      const auto pose=models[i]->WorldPose();
      const auto linear=models[i]->WorldLinearVel();
      const auto angular=models[i]->WorldAngularVel();
      geometry_msgs::msg::Transform transform;
      transform.translation.x=pose.Pos().X();transform.translation.y=pose.Pos().Y();transform.translation.z=pose.Pos().Z();
      transform.rotation.x=pose.Rot().X();transform.rotation.y=pose.Rot().Y();
      transform.rotation.z=pose.Rot().Z();transform.rotation.w=pose.Rot().W();
      geometry_msgs::msg::Twist twist;
      twist.linear.x=linear.X();twist.linear.y=linear.Y();twist.linear.z=linear.Z();
      twist.angular.x=angular.X();twist.angular.y=angular.Y();twist.angular.z=angular.Z();
      message.joint_names.push_back(names_[i]);
      message.transforms.push_back(transform);message.twist.push_back(twist);
    }
    publisher_->publish(message);published_=now.Double();
  }
public:
  void Load(gazebo::physics::WorldPtr world,sdf::ElementPtr sdf) override {
    world_=world;node_=gazebo_ros::Node::Get(sdf);
    publisher_=node_->create_publisher<sensor_msgs::msg::MultiDOFJointState>("fleet_truth",rclcpp::SensorDataQoS());
    connection_=gazebo::event::Events::ConnectWorldUpdateEnd(std::bind(&FleetTruth::update,this));
  }
};
GZ_REGISTER_WORLD_PLUGIN(FleetTruth)
}  // namespace wheelset_axis_servo
