# ============================================================
# launch/a4_cobot2.launch.py
# 역할:
#   - a4_cobot2 시스템 노드들을 한 번에 실행한다.
# 사용:
#   ros2 launch a4_cobot2 a4_cobot2.launch.py
# 전제 (별도로 먼저 실행):
#   - RealSense 카메라
#     예: ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
#   - Doosan 로봇 bringup
#     dsr01 / m0609
#   위 둘이 떠 있어야 object_detection/robot_arm이 정상 동작한다.
# ============================================================
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package = 'a4_cobot2'

    # 일반 노드 목록.
    # vlm_report_node는 파라미터가 필요하므로 아래에서 별도로 추가한다.
    node_executables = [
        'object_detection_node',
        'workspace_judge_node',
        'robot_arm_node',
        'task_manager_node',
        'safety_node',
        'status_notifier_node',
        'command_input_node',
        'hmi_interface_node',
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

    nodes.append(
            Node(
                package='a4_cobot2',
                executable='db_node',
                name='db_node',
                output='screen',
                parameters=[{
                    'db_path': '~/a4_cobot2_ws/a4_cobot2_log/cobot2_log.db',
                    'enable_rosbag': True,
                    'bag_path': '~/a4_cobot2_ws/a4_cobot2_log/bags',
                    'bag_storage_id': 'sqlite3',
                    'bag_startup_wait_sec': 0.7,
                    'bag_stop_timeout_sec': 12.0,
                    'bag_flush_delay_sec': 0.25,
                    'bag_topics': [
                        '/task_command',
                        '/task_command_raw',
                        '/task_status',
                        '/user_notice',
                        '/safety_command',
                        '/safety_state',
                        '/emergency_stop',
                        '/workspace_scan_mode',
                        '/scanned_objects_base',
                        '/workspace_judgement',
                        '/rosout',
                        '/camera/camera/color/image_raw',
                        '/camera/camera/aligned_depth_to_color/image_raw',
                        '/camera/camera/color/camera_info',
                    ],
                }],
            ),
        )


    # VLM 최종 보고 노드.
    # TaskManagerNode가 재검증 결과를 받은 뒤 /generate_final_report service를 호출한다.
    nodes.append(
        Node(
            package=package,
            executable='vlm_report_node',
            name='vlm_report_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {
                    # True:
                    #   OpenAI GPT-4o VLM을 사용해서 최종 보고문 생성
                    # False:
                    #   API 호출 없이 기존 fallback 문장만 반환
                    'use_vlm': True,

                    # OpenAI vision-capable model
                    'model': 'gpt-4o',

                    # 최종 작업공간 원본 이미지 topic
                    'image_topic': '/camera/camera/color/image_raw',

                    # YOLO bbox/mask/label이 그려진 이미지 topic
                    'annotated_image_topic': '/yolo_detection_image',

                    # OpenAI로 보낼 이미지 크기/품질 제한
                    'max_image_width': 960,
                    'jpeg_quality': 80,

                    # OpenAI API 응답 대기 시간
                    'openai_timeout_sec': 15.0,
                }
            ],
        )
    )

    return LaunchDescription(nodes)