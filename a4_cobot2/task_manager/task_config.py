# ============================================================
# task_manager/task_config.py
# 역할:
#   - task_manager_node가 사용하는 대상 물체, topic, service, action 이름을 한 곳에서 관리합니다.
# 장점:
#   - topic/service 이름을 여러 파일에 흩뿌리지 않아서 수정이 쉬워집니다.
# 주의:
#   - 실제 server 쪽 이름과 반드시 일치해야 합니다. 예: /judge_workspace, /organize_objects
# ============================================================
# task_manager/task_config.py

# 작업공간에서 확인할 대상 물체 이름 목록
TARGET_OBJECTS = [
    'hammer',
    'screwdriver',
    'bolt',
    'tape',
    'green_apple',
    'pineapple',
    'pocari',
    'gatorade',
]

# ObjectDetectionNode가 반환하는 좌표 frame 이름
# /scan_workspace 응답 payload의 frame과 맞춤
DETECTION_FRAME = 'camera_color_optical_frame'

# Topic 이름
TOPIC_TASK_COMMAND = '/task_command'
TOPIC_TASK_STATUS = '/task_status'
TOPIC_SAFETY_COMMAND = '/safety_command'
TOPIC_USER_NOTICE = '/user_notice'

# WorkspaceJudgeNode의 최신 판단 결과를 HMI에 전달하는 topic
TOPIC_WORKSPACE_JUDGEMENT = '/workspace_judgement'

# ObjectDetectionNode에게 이번 3자세 스캔이 최초 확인인지 재검증인지 알려주는 topic
TOPIC_WORKSPACE_SCAN_MODE = '/workspace_scan_mode'

# Service 이름
SERVICE_GET_3D_POSITION = 'get_3d_position' # get_3d_position은 디버깅용 또는 pick 직전 재확인용으로 남겨둘 수 있음
SERVICE_SCAN_WORKSPACE = 'scan_workspace' # 작업공간 전체를 한 번에 스캔하는 ObjectDetectionNode 서비스
SERVICE_JUDGE_WORKSPACE = '/judge_workspace'

# Action 이름입니다.
ACTION_ORGANIZE_OBJECTS = '/organize_objects'

# Service/action server 대기 시간입니다.
SERVICE_WAIT_TIMEOUT_SEC = 2.0
ACTION_WAIT_TIMEOUT_SEC = 2.0

# ==========================DB==========================================
TOPIC_TASK_COMMAND_RAW = '/task_command_raw'
TOPIC_SAFETY_STATE = '/safety_state'
TOPIC_EMERGENCY_STOP = '/emergency_stop'

# =============================== VLM ===============================
# VLMReportNode가 최종 사용자 보고문을 생성하는 service
SERVICE_GENERATE_FINAL_REPORT = '/generate_final_report'

# OpenAI API 호출은 일반 ROS service보다 시간이 걸릴 수 있으므로 별도 timeout을 둡니다.
# VLM_REPORT_WAIT_TIMEOUT_SEC는 service server가 있는지 확인하는 timeout입니다.
# OpenAI 응답 대기 시간은 vlm_report_node.py의 openai_timeout_sec 파라미터가 담당합니다.
VLM_REPORT_WAIT_TIMEOUT_SEC = 2.0
