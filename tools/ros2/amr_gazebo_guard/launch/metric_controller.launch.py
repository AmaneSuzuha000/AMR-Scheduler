"""FollowPath controller and local obstacle map, without another global planner."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def start(context):
    namespace=LaunchConfiguration('robot_namespace').perform(context)
    remappings=[('/tf','/'+namespace+'/tf'),('/tf_static','/'+namespace+'/tf_static')] if namespace else []
    return [
        Node(package='nav2_controller',executable='controller_server',name='controller_server',
             namespace=namespace,remappings=remappings,
             parameters=[LaunchConfiguration('config')],output='screen'),
        Node(package='nav2_lifecycle_manager',executable='lifecycle_manager',name='controller_manager',
             namespace=namespace,
             parameters=[{'use_sim_time':True,'autostart':True,'node_names':['controller_server']}],output='screen'),
    ]


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument('config'),
        DeclareLaunchArgument('robot_namespace',default_value=''),OpaqueFunction(function=start)])
