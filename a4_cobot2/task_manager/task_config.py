# task_manager/task_config.py

# 작업공간에서 확인할 대상 물체 이름 목록입니다.
TARGET_OBJECTS = [
    'hammer',
    'screwdriver',
    'wrench',
    'pliers',
    'drill',
]

# ObjectDetectionNode가 반환하는 좌표 frame 이름입니다.
DETECTION_FRAME = 'camera_frame'

# Topic 이름입니다.
TOPIC_TASK_COMMAND = '/task_command'
TOPIC_TASK_STATUS = '/task_status'
TOPIC_SAFETY_COMMAND = '/safety_command'
TOPIC_USER_NOTICE = '/user_notice'

# Service 이름입니다.
# ObjectDetectionNode에서 service를 상대 이름 'get_3d_position'으로 만들었다면
# 같은 namespace 기준으로 맞추기 위해 여기서도 'get_3d_position'을 사용합니다.
SERVICE_GET_3D_POSITION = 'get_3d_position'
SERVICE_JUDGE_WORKSPACE = '/judge_workspace'

# Action 이름입니다.
ACTION_ORGANIZE_OBJECTS = '/organize_objects'

# Service/action server 대기 시간입니다.
SERVICE_WAIT_TIMEOUT_SEC = 2.0
ACTION_WAIT_TIMEOUT_SEC = 2.0
