# ============================================================
# safety/safety_constants.py
# 역할:
#   - safety_node, task_manager_node, robot_arm_node가 공유하는 안전 명령/상태 문자열 상수입니다.
# 용어:
#   - stop: 로봇 동작 또는 현재 작업 정지
#   - shutdown: 노드/프로세스 종료
# ============================================================
# safety/safety_constants.py

# 비상정지를 활성화하는 안전 명령입니다.
SAFETY_COMMAND_STOP = 'stop'

# 비상정지를 해제하는 안전 명령입니다.
SAFETY_COMMAND_CLEAR = 'clear'

# 정상 안전 상태입니다.
SAFETY_STATE_NORMAL = 'normal'

# 비상정지 안전 상태입니다.
SAFETY_STATE_EMERGENCY_STOP = 'emergency_stop'