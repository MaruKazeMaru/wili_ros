import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

param_path = os.path.join(get_package_share_directory("wili_ros"), "sagasu_sim.param.yaml")

def generate_launch_description():
    return LaunchDescription([
        # -----------
        # run nodes
        # -----------

        Node(
            package="ros_tcp_endpoint",
            executable="default_server_endpoint",
            parameters=[
                param_path,
            ],
            output="screen"
        ),
        Node(
            package="wili_ros",
            executable="apriltag",
            parameters=[
                param_path,
            ],
            output="screen"
        ),
    ])
