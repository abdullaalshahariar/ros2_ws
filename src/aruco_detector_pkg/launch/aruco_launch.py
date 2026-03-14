from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('aruco_detector_pkg')
    params_file = os.path.join(pkg_share, 'config', 'aruco_params.yaml')

    return LaunchDescription([
        Node(
            package='aruco_detector_pkg',
            executable='aruco_detector',
            name='aruco_detector',
            output='screen',
            parameters=[params_file],
        )
    ])
