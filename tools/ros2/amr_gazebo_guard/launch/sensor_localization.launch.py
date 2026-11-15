"""Known-map sensor localization only; no global route planner or truth TF."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def start(context):
    config=LaunchConfiguration("config")
    namespace=LaunchConfiguration('robot_namespace').perform(context)
    remappings=[('/tf','/'+namespace+'/tf'),('/tf_static','/'+namespace+'/tf_static')] if namespace else []
    return [
        Node(package="robot_localization",executable="ekf_node",name="ekf_filter_node",
             namespace=namespace,remappings=remappings,
             parameters=[config],output="screen"),
        Node(package="nav2_map_server",executable="map_server",name="map_server",
             namespace=namespace,
             parameters=[config,{"yaml_filename":LaunchConfiguration("map")}],output="screen"),
        Node(package="nav2_amcl",executable="amcl",name="amcl",
             namespace=namespace,remappings=remappings,
             parameters=[config,{"laser_max_range":LaunchConfiguration("laser_max_range"),
                                 "laser_model_type":LaunchConfiguration("laser_model_type")}],output="screen"),
        Node(package="nav2_lifecycle_manager",executable="lifecycle_manager",name="localization_manager",
             namespace=namespace,
             parameters=[{"use_sim_time":True,"autostart":True,"node_names":["map_server","amcl"]}],output="screen"),
    ]


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument('config'),DeclareLaunchArgument('map'),
        DeclareLaunchArgument('robot_namespace',default_value=''),
        DeclareLaunchArgument('laser_max_range',default_value='8.0'),
        DeclareLaunchArgument('laser_model_type',default_value='likelihood_field'),
        OpaqueFunction(function=start)])
