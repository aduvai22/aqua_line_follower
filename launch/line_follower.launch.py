# =======================================================
# Author : Alakrit Gupta, Adnan Abdullah
# Email: gupta.alankrit@ufl.edu, adnanabdullah@ufl.edu
# =======================================================

from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'reverse_swim',
            default_value='false',
            description='Whether to swim in reverse'
        ),
        DeclareLaunchArgument(
            'fps',
            default_value='2.0',
            description='Frames per second for the velocity estimator'
        ),

        # Launch detector node
        Node(
            package='aqua_line_follower',
            executable='detector',
            name='detector',
            output='screen',
            parameters=[{'reverse_swim': LaunchConfiguration('reverse_swim')
            }]
        ),

        # Launch planner node
        Node(
            package='aqua_line_follower',
            executable='planner',
            name='planner',
            output='screen',
            parameters=[{
                'reverse_swim': LaunchConfiguration('reverse_swim')
            }]
        ),

        # Launch swimmer control node
        # Node(
        #     package='aqua_line_follower',
        #     executable='swimmer',
        #     name='swimmer',
        #     output='screen',
        #     parameters=[{
        #         'reverse_swim': LaunchConfiguration('reverse_swim')
        #     }]
        # ),

        # Launch optical flow-based velocity estimator node
        Node(
            package='aqua_velocity_control',
            executable='velocity_estimator',
            name='velocity_estimator',
            output='screen',
            parameters=[{
                'fps': LaunchConfiguration('fps')
            }]
        ),

        # Launch rosbagger node
        Node(
            package='shape', 
            executable='rosbag_client_node',
            name='rosbag_client_node',
            output='screen',
        ),
    ])
