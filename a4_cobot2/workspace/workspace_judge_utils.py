# ============================================================
# workspace/workspace_judge_utils.py
# 역할:
#   - 감지된 물체의 현재 위치가 어느 작업공간 zone에 속하는지 판단하고,
#     클래스별 expected zone과 비교해 정상/오배치/규칙 미정 목록을 만듭니다.
# 핵심 출력:
#   - normal_objects: 이미 올바른 구역에 있는 물체
#   - misplaced_objects: robot_arm_node로 보낼 정리 대상 물체
#   - unknown_rule_objects: CLASS_TO_ZONE에 규칙이 없는 물체
# 주의:
#   - DEFAULT_ZONES의 좌표는 임시값입니다. 실제 /scan_workspace 결과로 보정해야 합니다.
# ============================================================
# workspace/workspace_judge_utils.py

# 작업공간을 robot base 좌표계(mm) 기준 4개 구역(green/yellow/red/blue)으로 나눕니다.
# detection이 3자세 스캔으로 물체 위치를 base 좌표(mm)로 변환해 보내므로 zone도 base 좌표(mm)입니다.
# 각 zone은 네 꼭짓점을 감싸는 축정렬 바운딩박스(x_min~x_max, y_min~y_max)로 판정합니다.
# place_position은 그 zone으로 옮길 때 놓을 대표 위치이며, z는 놓을 높이(mm)입니다.
DEFAULT_ZONES = {
    'green': {
        'label': 'green',
        'x_min': 137.94,
        'x_max': 490.45,
        'y_min': 14.2,
        'y_max': 328.53,
        'place_position': {
            'x': 314.2,
            'y': 171.4,
            'z': 14.0,
        },
    },
    'yellow': {
        'label': 'yellow',
        'x_min': 142.96,
        'x_max': 498.30,
        'y_min': -330.17,
        'y_max': -10.7,
        'place_position': {
            'x': 320.6,
            'y': -170.4,
            'z': 14.0,
        },
    },
    'red': {
        'label': 'red',
        'x_min': 517.10,
        'x_max': 923.67,
        'y_min': 1.64,
        'y_max': 327.05,
        'place_position': {
            'x': 720.4,
            'y': 164.3,
            'z': 14.0,
        },
    },
    'blue': {
        'label': 'blue',
        'x_min': 508.94,
        'x_max': 922.62,
        'y_min': -321.72,
        'y_max': -13.98,
        'place_position': {
            'x': 715.8,
            'y': -167.9,
            'z': 14.0,
        },
    },
}


CLASS_TO_ZONE = {
    # 클래스 → zone 매핑. zone 이름은 green / yellow / red / blue 중 하나여야 합니다.
    # 매핑이 없는 클래스는 unknown_rule_objects로 분류됩니다.
    # key는 YOLO class name과 정확히 일치해야 합니다.
    'hammer': 'red',
    'screwdriver': 'red',
    'bolt': 'blue',
    'tape': 'blue',
    'green_apple': 'green',
    'pineapple': 'green',
    'pocari': 'yellow',
    'gatorade': 'yellow',
}


# 기본 작업공간 배치 규칙을 반환하는 함수
def get_default_zone_rules():
    return {
        'zones': DEFAULT_ZONES,
        'class_to_zone': CLASS_TO_ZONE,
    }


# 감지된 물체 목록을 순회하며 정상 배치 물체와 오배치 물체를 구분하는 함수
def judge_workspace(objects, frame: str, zone_rules):
    zones = zone_rules.get('zones', {})
    class_to_zone = zone_rules.get('class_to_zone', {})

    normal_objects = []
    misplaced_objects = []
    unknown_rule_objects = []

    for detected_object in objects:
        object_name = detected_object.get('name')
        position = detected_object.get('position', {})

        expected_zone_name = class_to_zone.get(object_name)
        current_zone_name = find_current_zone_name(position, zones)

        if expected_zone_name is None:
            unknown_rule_objects.append({
                'name': object_name,
                'position': position,
                'current_zone': current_zone_name,
                'reason': 'no_class_to_zone_rule',
            })
            continue

        expected_zone = zones.get(expected_zone_name)

        if expected_zone is None:
            unknown_rule_objects.append({
                'name': object_name,
                'position': position,
                'current_zone': current_zone_name,
                'expected_zone': expected_zone_name,
                'reason': 'expected_zone_not_defined',
            })
            continue

        if current_zone_name == expected_zone_name:
            normal_objects.append(
                make_normal_object(
                    detected_object=detected_object,
                    current_zone_name=current_zone_name,
                    expected_zone_name=expected_zone_name,
                    expected_zone=expected_zone,
                )
            )
        else:
            misplaced_objects.append(
                make_misplaced_object(
                    detected_object=detected_object,
                    current_zone_name=current_zone_name,
                    expected_zone_name=expected_zone_name,
                    expected_zone=expected_zone,
                    frame=frame,
                )
            )

    result = make_result_status(
        normal_objects,
        misplaced_objects,
        unknown_rule_objects,
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
        },
    }


# 물체의 현재 position이 4개 구역 중 어디에 들어가는지 찾는 함수
def find_current_zone_name(position, zones):
    for zone_name, zone in zones.items():
        if is_inside_zone(position, zone):
            return zone_name

    return 'outside_workspace'


# 물체의 3D position 중 x, y가 특정 zone 안에 있는지 판단하는 함수
def is_inside_zone(position, zone):
    try:
        x = float(position['x'])
        y = float(position['y'])

    except (KeyError, TypeError, ValueError):
        return False

    is_inside_x = zone['x_min'] <= x <= zone['x_max']
    is_inside_y = zone['y_min'] <= y <= zone['y_max']

    return is_inside_x and is_inside_y


# 정상 배치된 물체 정보를 만드는 함수
def make_normal_object(detected_object, current_zone_name, expected_zone_name, expected_zone):
    return {
        'name': detected_object.get('name'),
        'class_id': detected_object.get('class_id'),
        'confidence': detected_object.get('confidence'),
        'box': detected_object.get('box'),
        'center': detected_object.get('center'),
        'depth': detected_object.get('depth'),
        'position': detected_object.get('position'),
        'current_zone': current_zone_name,
        'expected_zone': expected_zone_name,
        'expected_zone_label': expected_zone.get('label'),
    }


# 오배치 물체 정보를 만드는 함수
# robot_arm_node가 바로 사용할 수 있도록 pick_position과 place_position을 함께 넣습니다.
def make_misplaced_object(detected_object, current_zone_name, expected_zone_name, expected_zone, frame):
    return {
        'name': detected_object.get('name'),
        'class_id': detected_object.get('class_id'),
        'confidence': detected_object.get('confidence'),
        'box': detected_object.get('box'),
        'center': detected_object.get('center'),
        'depth': detected_object.get('depth'),

        # 현재 물체 위치입니다. 나중에 로봇 pick 좌표로 변환할 기준값입니다.
        'position': detected_object.get('position'),
        'pick_position': detected_object.get('position'),

        # 파지 시 그리퍼를 돌릴 물체 각도(base 좌표계, deg). detection의 PCA 결과.
        'angle': detected_object.get('angle'),

        # 현재 구역과 원래 있어야 하는 구역입니다.
        'current_zone': current_zone_name,
        'expected_zone': expected_zone_name,
        'expected_zone_label': expected_zone.get('label'),

        # 원래 있어야 하는 zone의 대표 위치입니다. base 좌표계(mm) 기준입니다.
        'place_position': expected_zone.get('place_position'),
        'place_frame': frame,

        'reason': 'outside_expected_zone',
    }


# 정상 물체, 오배치 물체, 규칙 미정 물체 목록을 기준으로 전체 판단 상태를 만드는 함수
def make_result_status(normal_objects, misplaced_objects, unknown_rule_objects):
    # unknown_rule 객체가 하나라도 있으면 전체 결과를 unknown_rule_found로 처리합니다.
    # 이유:
    # - 배치 규칙이 없는 물체가 있으면 현재 시스템이 그 물체의 정상/오배치를 판단할 수 없습니다.
    # - 따라서 오배치 물체가 함께 있어도 우선 사용자 확인이 필요한 상태로 봅니다.
    if len(unknown_rule_objects) > 0:
        return 'unknown_rule_found'

    if len(misplaced_objects) > 0:
        return 'misplaced_found'

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
        },
    }