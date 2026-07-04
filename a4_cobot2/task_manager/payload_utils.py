# task_manager/payload_utils.py

import json

from task_manager import status_codes as Status
from task_manager.task_config import DETECTION_FRAME


# 현재 작업명이 작업공간 감지 흐름인지 판단합니다.
def is_workspace_detection_task(task_name: str) -> bool:
    return task_name in [
        Status.TASK_CHECK_WORKSPACE,
        Status.TASK_RECHECK_WORKSPACE,
    ]


# ObjectDetectionNode가 반환한 3D 좌표가 유효한 감지 결과인지 판단합니다.
def is_valid_position(position) -> bool:
    if position is None:
        return False

    if len(position) != 3:
        return False

    x, y, z = position

    if abs(x) < 1e-9 and abs(y) < 1e-9 and abs(z) < 1e-9:
        return False

    return True


# ObjectDetectionNode의 좌표 응답을 task_manager 내부 detected_object dict로 변환합니다.
def make_detected_object(target_name: str, position):
    return {
        'name': target_name,
        'position': {
            'x': float(position[0]),
            'y': float(position[1]),
            'z': float(position[2]),
        },
    }


# workspace_judge_node에 보낼 판단 요청 payload dict를 만듭니다.
def make_workspace_judgement_request_payload(task_name: str, objects: list, frame: str = DETECTION_FRAME):
    return {
        'task': task_name,
        'frame': frame,
        'objects': objects,
    }


# workspace_judge_node에 보낼 판단 요청 payload를 JSON 문자열로 만듭니다.
def make_workspace_judgement_request_json(task_name: str, objects: list, frame: str = DETECTION_FRAME) -> str:
    payload = make_workspace_judgement_request_payload(
        task_name=task_name,
        objects=objects,
        frame=frame,
    )

    return json.dumps(payload, ensure_ascii=False)


# robot_arm_node에 보낼 organize action goal payload dict를 만듭니다.
def make_organize_goal_payload(objects: list):
    return {
        'task': 'organize_objects',
        'objects': objects,
    }


# robot_arm_node에 보낼 organize action goal payload를 JSON 문자열로 만듭니다.
def make_organize_goal_json(objects: list) -> str:
    payload = make_organize_goal_payload(objects)

    return json.dumps(payload, ensure_ascii=False)


# JSON 문자열을 Python dict/list로 변환합니다.
def parse_json_payload(json_text: str):
    return json.loads(json_text)
