# task_manager/status_codes.py

# command_input_node가 /task_command로 보내는 명령어입니다.
COMMAND_CHECK_WORKSPACE = 'check_workspace'
COMMAND_START_ORGANIZE = 'start_organize'
COMMAND_STOP = 'stop'
COMMAND_SHUTDOWN = 'shutdown'

# TaskManagerNode 내부 작업 이름입니다.
TASK_CHECK_WORKSPACE = 'check_workspace'
TASK_RECHECK_WORKSPACE = 'recheck_workspace'
TASK_START_ORGANIZE = 'start_organize'

# TaskManagerNode가 준비되었음을 나타내는 상태입니다.
TASK_MANAGER_READY = 'task_manager_ready'

# 작업공간 확인 흐름 상태입니다.
CHECK_WORKSPACE_REQUESTED = 'check_workspace_requested'
CHECKING_WORKSPACE = 'checking_workspace'
WORKSPACE_DETECTION_FINISHED = 'workspace_detection_finished'
JUDGING_WORKSPACE = 'judging_workspace'
WORKSPACE_ALL_CLEAR = 'workspace_all_clear'
WORKSPACE_MISPLACED_FOUND = 'workspace_misplaced_found'
WORKSPACE_UNKNOWN_RULE_FOUND = 'workspace_unknown_rule_found'
WORKSPACE_JUDGEMENT_FINISHED = 'workspace_judgement_finished'
WORKSPACE_JUDGEMENT_FAILED = 'workspace_judgement_failed'
WORKSPACE_JUDGEMENT_JSON_ERROR = 'workspace_judgement_json_error'
WORKSPACE_JUDGEMENT_RESPONSE_ERROR = 'workspace_judgement_response_error'
WORKSPACE_JUDGEMENT_UNKNOWN_RESULT = 'workspace_judgement_unknown_result'
WORKSPACE_JUDGEMENT_UNEXPECTED_TASK = 'workspace_judgement_unexpected_task'
NO_OBJECTS_DETECTED = 'no_objects_detected'
CHECK_WORKSPACE_STOPPED = 'check_workspace_stopped'
OBJECT_DETECTION_SERVICE_UNAVAILABLE = 'object_detection_service_unavailable'
JUDGE_WORKSPACE_SERVICE_UNAVAILABLE = 'judge_workspace_service_unavailable'

# 정리 후 재검증 흐름 상태입니다.
RECHECK_WORKSPACE_REQUESTED = 'recheck_workspace_requested'
RECHECKING_WORKSPACE = 'rechecking_workspace'
RECHECK_ALL_CLEAR = 'recheck_all_clear'
RECHECK_MISPLACED_REMAINING = 'recheck_misplaced_remaining'
RECHECK_UNKNOWN_RULE_FOUND = 'recheck_unknown_rule_found'
RECHECK_NO_OBJECTS_DETECTED = 'recheck_no_objects_detected'
RECHECK_UNKNOWN_RESULT = 'recheck_unknown_result'

# 로봇 정리 흐름 상태입니다.
START_ORGANIZE_REQUESTED = 'start_organize_requested'
NO_WORKSPACE_JUDGEMENT_AVAILABLE = 'no_workspace_judgement_available'
NOTHING_TO_ORGANIZE = 'nothing_to_organize'
REQUESTING_ROBOT_ORGANIZE = 'requesting_robot_organize'
ROBOT_ARM_ACTION_UNAVAILABLE = 'robot_arm_action_unavailable'
ROBOT_ORGANIZE_GOAL_ACCEPTED = 'robot_organize_goal_accepted'
ROBOT_ORGANIZE_GOAL_REJECTED = 'robot_organize_goal_rejected'
ROBOT_ORGANIZE_FINISHED = 'robot_organize_finished'
ROBOT_ORGANIZE_FAILED = 'robot_organize_failed'
ROBOT_ORGANIZE_RESULT_ERROR = 'robot_organize_result_error'
ROBOT_ORGANIZE_CANCEL_REQUESTED = 'robot_organize_cancel_requested'
ROBOT_ORGANIZE_CANCEL_ACCEPTED = 'robot_organize_cancel_accepted'
ROBOT_ORGANIZE_CANCEL_REJECTED = 'robot_organize_cancel_rejected'
ROBOT_ORGANIZE_CANCEL_ERROR = 'robot_organize_cancel_error'

# 정지/종료/공통 상태입니다.
STOP_REQUESTED = 'stop_requested'
SHUTDOWN_REQUESTED = 'shutdown_requested'
BUSY = 'busy'
UNKNOWN_COMMAND = 'unknown_command'
IDLE = 'idle'


# 로봇 정리 action feedback용 상태 문자열을 만듭니다.
def make_robot_organizing_status(current_index: int, total_count: int) -> str:
    return f'robot_organizing_{current_index}_of_{total_count}'
