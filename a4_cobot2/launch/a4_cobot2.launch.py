# ============================================================
# launch/a4_cobot2.launch.py
# 역할:
#   - a4_cobot2 시스템 노드들을 한 번에 실행한다.
# 사용:
#   ros2 launch a4_cobot2 a4_cobot2.launch.py
# 전제 (별도로 먼저 실행):
#   - RealSense 카메라 (예: ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true)
#   - Doosan 로봇 bringup (dsr01 / m0609)
#   위 둘이 떠 있어야 object_detection/robot_arm이 정상 동작한다.
# ============================================================
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package = 'a4_cobot2'

    # 실행할 노드 목록. 순서는 상관없지만(노드가 서로 서비스/토픽을 기다림),
    # 카메라/로봇 bringup은 이 launch 전에 먼저 떠 있어야 한다.
    node_executables = [
        'object_detection_node',
        'workspace_judge_node',
        'robot_arm_node',
        'task_manager_node',
        'safety_node',
        'status_notifier_node',
        'command_input_node',
        'db_node',
    ]

    nodes = [
        Node(
            package=package,
            executable=executable,
            output='screen',
            emulate_tty=True,
        )
        for executable in node_executables
    ]

    return LaunchDescription(nodes)
