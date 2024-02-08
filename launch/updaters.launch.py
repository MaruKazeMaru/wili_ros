import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # -----------
        # run nodes
        # -----------

        Node(
            package="wili_ros",
            executable="apriltag",
            parameters=[
                os.path.join(get_package_share_directory("wili_ros"), "apriltag.param.yaml"),
            ],
            output="screen"
        ),
        Node(
            package="wili_ros",
            executable="obs_stacker",
            output="screen"
        ),
        Node(
            package="wili_ros",
            executable="hmm",
            output="screen"
        ),
        Node(
            package="wili_ros",
            executable="suggester",
            output="screen"
        ),
        Node(
            package="wili_ros",
            executable="db_agent",
            parameters=[
                os.path.join(get_package_share_directory('wili_ros'), 'updaters.param.yaml'),
            ],
            output="screen"
        ),
    ])
