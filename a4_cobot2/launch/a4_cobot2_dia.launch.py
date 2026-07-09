# ============================================================
# launch/a4_cobot2_dia.launch.py
# 역할:
#   - a4_cobot2 시스템을 "diablo(그리드 배치) 버전"으로 한 번에 실행한다.
#   - 바뀐 3개 노드만 _dia 버전으로 교체하고, 나머지는 원본 노드를 그대로 쓴다.
# 원본과의 관계:
#   - a4_cobot2.launch.py      : 원본 프로그램
#   - a4_cobot2_dia.launch.py  : diablo 그리드 배치 적용 버전 (이 파일)
#   두 launch는 같은 topic/service 이름을 쓰므로 동시에 실행하지 말고 하나만 실행한다.
# 사용:
#   ros2 launch a4_cobot2 a4_cobot2_dia.launch.py
# 전제 (별도로 먼저 실행): RealSense 카메라, Doosan 로봇 bringup
# ============================================================
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package = 'a4_cobot2'

    # diablo 버전으로 교체되는 노드(3개) + 원본 그대로 쓰는 노드들.
    # vlm_report_node는 파라미터가 필요하므로 아래에서 별도로 추가한다.
    node_executables = [
        # --- diablo(_dia) 버전으로 교체 ---
        'object_detection_node_dia',   # footprint width/length 발행
        'workspace_judge_node_dia',    # grid_allocator로 per-object place 좌표
        'robot_arm_node_dia',          # width 벌림 + place y평행 회전
        # --- 원본 그대로 ---
        'task_manager_node',
        'safety_node',
        'status_notifier_node',
        'command_input_node',
        'db_node',
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

    # VLM 최종 보고 노드(원본).
    nodes.append(
        Node(
            package=package,
            executable='vlm_report_node',
            name='vlm_report_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {
                    'use_vlm': True,
                    'model': 'gpt-4o',
                    'image_topic': '/camera/camera/color/image_raw',
                    'annotated_image_topic': '/yolo_detection_image',
                    'max_image_width': 960,
                    'jpeg_quality': 80,
                    'openai_timeout_sec': 15.0,
                }
            ],
        )
    )

    return LaunchDescription(nodes)
