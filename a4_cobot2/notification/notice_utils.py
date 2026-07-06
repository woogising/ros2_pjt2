# ============================================================
# notification/notice_utils.py
# 역할:
#   - 내부 상태 코드와 judgement payload를 사용자가 이해할 수 있는 자연어 안내문으로 바꿉니다.
# 사용 위치:
#   - status_notifier_node: /task_status, /safety_state를 자연어로 변환
#   - task_manager_node: 판단 결과 payload를 /user_notice 문장으로 변환
# ============================================================
# notification/notice_utils.py

import re

from collections import Counter
from safety.safety_constants import SAFETY_STATE_NORMAL
from safety.safety_constants import SAFETY_STATE_EMERGENCY_STOP


TASK_STATUS_NOTICE_MAP = {
    'task_manager_ready': '작업 관리자 노드가 준비되었습니다.',
    'check_workspace_requested': '작업공간 확인을 시작합니다.',
    'checking_workspace': '카메라로 작업공간을 확인하고 있습니다.',
    'workspace_detection_finished': '물체 인식이 완료되었습니다.',
    'judging_workspace': '물체가 올바른 구역에 있는지 판단하고 있습니다.',
    'workspace_all_clear': '모든 물체가 올바른 위치에 있습니다.',
    'workspace_misplaced_found': '잘못 배치된 물체가 있습니다.',
    'workspace_unknown_rule_found': '일부 물체의 배치 규칙을 찾을 수 없습니다.',
    'workspace_judgement_finished': '작업공간 판단이 완료되었습니다.',
    'workspace_judgement_failed': '작업공간 판단에 실패했습니다.',
    'workspace_judgement_json_error': '작업공간 판단 결과를 해석하지 못했습니다.',
    'workspace_judgement_response_error': '작업공간 판단 응답 처리 중 오류가 발생했습니다.',
    'workspace_judgement_unknown_result': '작업공간 판단 결과가 명확하지 않습니다.',
    'workspace_judgement_unexpected_task': '예상하지 못한 작업 상태에서 작업공간 판단 결과를 받았습니다.',
    'workspace_detection_stopped': '작업공간 감지가 중단되었습니다.',
    'no_objects_detected': '감지된 물체가 없습니다.',
    'start_organize_requested': '정리 작업을 시작합니다.',
    'no_workspace_judgement_available': '저장된 작업공간 판단 결과가 없습니다. 먼저 작업공간을 확인해주세요.',
    'nothing_to_organize': '정리할 물체가 없습니다.',
    'requesting_robot_organize': '로봇팔에 정리 작업을 요청했습니다.',
    'robot_arm_action_unavailable': '로봇팔 정리 action server를 찾을 수 없습니다.',
    'robot_organize_goal_accepted': '로봇팔이 정리 작업을 수락했습니다.',
    'robot_organize_goal_rejected': '로봇팔이 정리 작업을 거절했습니다.',
    'robot_organize_finished': '로봇 정리 작업이 완료되었습니다.',
    'robot_organize_failed': '로봇 정리 작업이 실패했습니다.',
    'robot_organize_result_error': '로봇 정리 결과 처리 중 오류가 발생했습니다.',
    'robot_organize_cancel_requested': '진행 중인 로봇 정리 작업에 취소 요청을 보냈습니다.',
    'robot_organize_cancel_accepted': '로봇 정리 작업 취소가 수락되었습니다.',
    'robot_organize_cancel_rejected': '취소할 로봇 정리 작업이 없거나 취소 요청이 거절되었습니다.',
    'robot_organize_cancel_error': '로봇 정리 작업 취소 처리 중 오류가 발생했습니다.',
    'check_workspace_stopped': '작업공간 확인이 중단되었습니다.',
    'object_detection_service_unavailable': '물체 위치 인식 서비스를 찾을 수 없습니다.',
    'judge_workspace_service_unavailable': '작업공간 판단 서비스를 찾을 수 없습니다.',
    'recheck_workspace_requested': '정리 결과를 확인하기 위해 작업공간을 다시 검사합니다.',
    'rechecking_workspace': '작업공간을 다시 확인하고 있습니다.',
    'recheck_all_clear': '재검증 결과, 모든 물체가 올바른 위치에 있습니다.',
    'recheck_misplaced_remaining': '재검증 결과, 아직 잘못 배치된 물체가 남아 있습니다.',
    'recheck_unknown_rule_found': '재검증 중 일부 물체의 배치 규칙을 찾을 수 없습니다.',
    'recheck_no_objects_detected': '재검증 중 감지된 물체가 없습니다.',
    'recheck_unknown_result': '재검증 결과가 명확하지 않습니다.',
    'stop_requested': '정지 요청을 보냈습니다.',
    'shutdown_requested': '종료 요청을 받았습니다.',
    'busy': '현재 다른 작업을 처리 중입니다.',
    'unknown_command': '알 수 없는 명령입니다.',
    'idle': None,
}


# task_manager_node의 상태 코드를 사용자가 이해하기 쉬운 문장으로 변환하는 함수
def make_task_status_notice(status: str):
    if status in TASK_STATUS_NOTICE_MAP:
        return TASK_STATUS_NOTICE_MAP[status]

    organize_match = re.match(r'robot_organizing_(\d+)_of_(\d+)', status)
    if organize_match:
        current_index = organize_match.group(1)
        total_count = organize_match.group(2)
        return f'로봇이 {total_count}개 중 {current_index}번째 물체를 정리하고 있습니다.'

    return f'현재 상태: {status}'


# safety_node의 상태 코드를 사용자가 이해하기 쉬운 문장으로 변환하는 함수
def make_safety_state_notice(safety_state: str):
    if safety_state == SAFETY_STATE_NORMAL:
        return '안전 상태가 정상입니다.'

    if safety_state == SAFETY_STATE_EMERGENCY_STOP:
        return '비상정지 상태입니다. 로봇 동작을 중단합니다.'

    if safety_state.startswith('unknown_safety_command'):
        return '알 수 없는 안전 명령이 들어왔습니다.'

    return f'안전 상태: {safety_state}'


# workspace_judge_node의 최초 작업공간 판단 결과를 사용자 안내 문장으로 변환하는 함수
def make_workspace_judgement_notice(judgement_payload):
    result = judgement_payload.get('result', 'unknown')
    misplaced_objects = judgement_payload.get('misplaced_objects', [])
    unknown_rule_objects = judgement_payload.get('unknown_rule_objects', [])
    summary = judgement_payload.get('summary', {})

    if result == 'all_clear':
        return '작업공간 확인 결과, 모든 물건이 지정된 구역에 올바르게 배치되어 있습니다.'

    if result == 'misplaced_found':
        object_text = make_object_count_text(misplaced_objects)
        misplaced_count = summary.get('misplaced_count', len(misplaced_objects))

        return (
            f'현재 잘못 배치된 물건은 {misplaced_count}개입니다. '
            f'{object_text}가 지정 구역 밖에 있습니다. '
            f'정리를 원하면 다시 호출한 뒤 정리해줘라고 말해주세요.'
        )

    if result == 'unknown_rule_found':
        object_names = [
            obj.get('name', '알 수 없는 물체')
            for obj in unknown_rule_objects
        ]

        object_text = ', '.join(object_names)

        if object_text == '':
            object_text = '일부 물체'

        return (
            f'일부 물체의 배치 규칙을 찾을 수 없습니다. '
            f'확인이 필요한 물체는 {object_text}입니다.'
        )

    if result == 'no_objects':
        return '작업공간에서 감지된 물체가 없습니다.'

    return '작업공간 판단 결과를 정확히 해석하지 못했습니다.'


# 재검증 후 남아 있는 오배치 물체 목록을 사용자 안내 문장으로 변환하는 함수
def make_recheck_remaining_notice(judgement_payload):
    misplaced_objects = judgement_payload.get('misplaced_objects', [])
    summary = judgement_payload.get('summary', {})

    object_text = make_object_count_text(misplaced_objects)
    misplaced_count = summary.get('misplaced_count', len(misplaced_objects))

    return (
        f'정리 작업 후 다시 확인했지만, 아직 잘못 배치된 물건이 {misplaced_count}개 남아 있습니다. '
        f'남은 물체는 {object_text}입니다.'
    )


# object dict 목록을 "hammer 1개, wrench 2개" 형태의 문장 조각으로 변환하는 함수
def make_object_count_text(objects):
    object_names = [
        obj.get('name', '알 수 없는 물체')
        for obj in objects
    ]

    object_counter = Counter(object_names)

    object_text = ', '.join(
        [
            f'{name} {count}개'
            for name, count in object_counter.items()
        ]
    )

    if object_text == '':
        object_text = '일부 물체'

    return object_text
