"""Play the ped_vehicle rosbag in a loop, run 3D object detection and show everything in RViz2.

Usage:
  ros2 launch ~/ros2_humble/src/bag_viz/view_bag.launch.py
  ros2 launch ~/ros2_humble/src/bag_viz/view_bag.launch.py rate:=0.5 loop:=false detect:=false
"""
import glob
import os
import site

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.realpath(__file__))
DEFAULT_BAG = os.path.join(os.path.dirname(HERE), 'rosbag2_2022_04_14-ped_vehicle')
RVIZ_CONFIG = os.path.join(HERE, 'ped_vehicle.rviz')


def cuda_env():
    """LD_LIBRARY_PATH with the pip-installed CUDA/cuDNN libs so onnxruntime-gpu can use the GPU."""
    roots = site.getsitepackages() + [site.getusersitepackages()]
    dirs = sorted({d for r in roots for d in glob.glob(os.path.join(r, 'nvidia', '*', 'lib'))})
    return {'LD_LIBRARY_PATH': ':'.join(dirs + [os.environ.get('LD_LIBRARY_PATH', '')])}


def generate_launch_description():
    bag = LaunchConfiguration('bag')
    rate = LaunchConfiguration('rate')
    loop = LaunchConfiguration('loop')

    play_cmd = ['ros2', 'bag', 'play', bag, '--rate', rate]

    return LaunchDescription([
        DeclareLaunchArgument('bag', default_value=DEFAULT_BAG,
                              description='Path to the rosbag2 folder'),
        DeclareLaunchArgument('rate', default_value='1.0',
                              description='Playback speed multiplier'),
        DeclareLaunchArgument('loop', default_value='true',
                              description='Replay the bag forever'),
        DeclareLaunchArgument('detect', default_value='true',
                              description='Run the 3D object detector'),
        DeclareLaunchArgument('score_threshold', default_value='0.35',
                              description='Minimum CenterPoint confidence'),

        Node(package='rviz2', executable='rviz2', name='rviz2',
             arguments=['-d', RVIZ_CONFIG], output='screen'),

        ExecuteProcess(
            cmd=['python3', os.path.join(HERE, 'lidar_detector_node.py'), '--ros-args',
                 '-p', ['score_threshold:=', LaunchConfiguration('score_threshold')]],
            additional_env=cuda_env(), output='screen', name='lidar_detector',
            condition=IfCondition(LaunchConfiguration('detect'))),

        ExecuteProcess(cmd=play_cmd + ['--loop'], output='screen',
                       condition=IfCondition(loop)),
        ExecuteProcess(cmd=play_cmd, output='screen',
                       condition=UnlessCondition(loop)),
    ])
