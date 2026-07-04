# 물체별 정상 배치 구역 규칙을 정의합니다.
DEFAULT_ZONE_RULES = {
    # TODO:
    # 아래 좌표는 임시 기준입니다.
    # 실제 ObjectDetectionNode의 x, y, z 좌표 범위를 보고 반드시 보정해야 합니다.
    # 현재 position은 camera_frame 기준입니다.

    'hammer': {
        'zone_name': 'hammer_zone',
        'x_min': -300.0,
        'x_max': 0.0,
        'y_min': -300.0,
        'y_max': 300.0,
    },
    'screwdriver': {
        'zone_name': 'screwdriver_zone',
        'x_min': 0.0,
        'x_max': 300.0,
        'y_min': -300.0,
        'y_max': 300.0,
    },
    'wrench': {
        'zone_name': 'wrench_zone',
        'x_min': -300.0,
        'x_max': 0.0,
        'y_min': 300.0,
        'y_max': 600.0,
    },
    'pliers': {
        'zone_name': 'pliers_zone',
        'x_min': 0.0,
        'x_max': 300.0,
        'y_min': 300.0,
        'y_max': 600.0,
    },
    'drill': {
        'zone_name': 'drill_zone',
        'x_min': -150.0,
        'x_max': 150.0,
        'y_min': 600.0,
        'y_max': 900.0,
    },
}


# 기본 작업공간 배치 규칙을 반환하는 함수
def get_default_zone_rules():
    return DEFAULT_ZONE_RULES


# 감지된 물체 목록을 순회하며 정상 배치 물체와 오배치 물체를 구분하는 함수
def judge_workspace(objects, frame: str, zone_rules):
    normal_objects = []
    misplaced_objects = []
    unknown_rule_objects = []

    for detected_object in objects:
        object_name = detected_object.get('name')
        position = detected_object.get('position', {})

        if object_name not in zone_rules:
            unknown_rule_objects.append({
                'name': object_name,
                'position': position,
                'reason': 'no_zone_rule'
            })
            continue

        rule = zone_rules[object_name]

        if is_inside_expected_zone(position, rule):
            normal_objects.append({
                'name': object_name,
                'position': position,
                'zone_name': rule['zone_name']
            })
        else:
            misplaced_objects.append({
                'name': object_name,
                'position': position,
                'expected_zone': rule['zone_name'],
                'reason': 'outside_expected_zone'
            })

    result = make_result_status(
        normal_objects,
        misplaced_objects,
        unknown_rule_objects
    )

    return {
        'task': 'judge_workspace',
        'frame': frame,
        'result': result,
        'normal_objects': normal_objects,
        'misplaced_objects': misplaced_objects,
        'unknown_rule_objects': unknown_rule_objects,
        'summary': {
            'normal_count': len(normal_objects),
            'misplaced_count': len(misplaced_objects),
            'unknown_rule_count': len(unknown_rule_objects),
            'total_detected_count': len(objects),
        }
    }


# 물체의 3D 위치가 해당 물체의 정상 배치 영역 안에 있는지 판단하는 함수
def is_inside_expected_zone(position, rule):
    try:
        x = float(position['x'])
        y = float(position['y'])

    except (KeyError, TypeError, ValueError):
        return False

    is_inside_x = rule['x_min'] <= x <= rule['x_max']
    is_inside_y = rule['y_min'] <= y <= rule['y_max']

    return is_inside_x and is_inside_y


# 정상 물체, 오배치 물체, 규칙 미정 물체 목록을 기준으로 전체 판단 상태를 만드는 함수
def make_result_status(normal_objects, misplaced_objects, unknown_rule_objects):
    if len(misplaced_objects) > 0:
        return 'misplaced_found'

    if len(unknown_rule_objects) > 0:
        return 'unknown_rule_found'

    if len(normal_objects) == 0:
        return 'no_objects'

    return 'all_clear'


# 예외 상황에서 반환할 error judgement payload를 만드는 함수
def make_error_payload(error_message: str):
    return {
        'task': 'judge_workspace',
        'frame': 'unknown_frame',
        'result': 'error',
        'error_message': error_message,
        'normal_objects': [],
        'misplaced_objects': [],
        'unknown_rule_objects': [],
        'summary': {
            'normal_count': 0,
            'misplaced_count': 0,
            'unknown_rule_count': 0,
            'total_detected_count': 0,
        }
    }
