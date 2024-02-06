from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
	return LaunchDescription([
        # -----------
		# declare comand line arguments
        # -----------
		DeclareLaunchArgument(
			"db_url",
			description="database URL (The format follows SQLAlchemy.)"
        ),
		DeclareLaunchArgument(
			"area_id",
			description="id of target area in database"
        ),

        # -----------
		# run nodes
        # -----------
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
			parameters=[{
				"db_url": LaunchConfiguration("db_url"),
				"area_id": LaunchConfiguration("area_id")
			}],
	    	output="screen"
    	),
	])
