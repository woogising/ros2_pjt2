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

# 작업공간을 카메라 화면 기준 4개 구역으로 나눕니다.
# 현재 ObjectDetectionNode의 position은 camera_color_optical_frame 기준입니다.
# x < 0 이면 카메라 화면 기준 왼쪽, x > 0 이면 오른쪽입니다.
# y < 0 이면 카메라 화면 기준 위쪽, y > 0 이면 아래쪽입니다.
#
# 주의:
# 아래 좌표는 임시값입니다.
# 실제 /scan_workspace 결과를 보고 x, y 범위를 반드시 보정해야 합니다.
# DEFAULT_ZONES:
#   작업공간을 4개 구역으로 나눈 임시 좌표 규칙입니다.
#   x/y는 ObjectDetectionNode가 계산한 camera_color_optical_frame 기준 position입니다.
#   place_position은 해당 zone으로 옮길 때 사용할 대표 위치입니다.
#   실제 로봇 제어 전에는 robot base frame으로 변환해야 합니다.
DEFAULT_ZONES = {
    'left_top': {
        'label': '왼쪽 위',
        'x_min': -0.45,
        'x_max': 0.0,
        'y_min': -0.35,
        'y_max': 0.0,
        'place_position': {
            'x': -0.225,
            'y': -0.175,
            'z': 0.45,
        },
    },
    'left_bottom': {
        'label': '왼쪽 아래',
        'x_min': -0.45,
        'x_max': 0.0,
        'y_min': 0.0,
        'y_max': 0.35,
        'place_position': {
            'x': -0.225,
            'y': 0.175,
            'z': 0.45,
        },
    },
    'right_top': {
        'label': '오른쪽 위',
        'x_min': 0.0,
        'x_max': 0.45,
        'y_min': -0.35,
        'y_max': 0.0,
        'place_position': {
            'x': 0.225,
            'y': -0.175,
            'z': 0.45,
        },
    },
    'right_bottom': {
        'label': '오른쪽 아래',
        'x_min': 0.0,
        'x_max': 0.45,
        'y_min': 0.0,
        'y_max': 0.35,
        'place_position': {
            'x': 0.225,
            'y': 0.175,
            'z': 0.45,
        },
    },
}


# 클래스 이름별로 원래 있어야 하는 구역을 지정합니다.
# 현재 YOLO 모델이 인식하는 클래스는 class_name_tool.json 기준으로
# drill, hammer, pliers, screwdriver, wrench입니다.
# CLASS_TO_ZONE:
#   물체 클래스 이름별로 원래 있어야 하는 zone 이름을 지정합니다.
#   YOLO class name과 key가 정확히 일치해야 합니다.
#   여기에 없는 클래스는 unknown_rule_objects로 분류됩니다.
CLASS_TO_ZONE = {
    # 예시: 공구류를 오른쪽 위/아래로 나누는 규칙
    'hammer': 'right_top',
    'screwdriver': 'right_top',
    'wrench': 'right_bottom',
    'pliers': 'right_bottom',
    'drill': 'left_bottom',

    # 나중에 과일 모델을 추가하거나 재학습하면 아래처럼 확장하면 됩니다.
    # 'apple': 'left_top',
    # 'banana': 'left_top',
}


# 클래스별로 물체를 놓을 정확한 좌표(슬롯)를 지정합니다.
# 같은 zone에 여러 클래스가 매핑돼도 서로 겹치지 않도록, zone을 나눈 서로 다른 좌표를 줍니다.
# 각 좌표는 그 클래스의 expected zone 범위 안에 있어야 합니다.
# 주의: 아래 좌표는 임시값입니다. 실제 /scan_workspace 결과로 보정해야 합니다.
# x/y는 camera_color_optical_frame 기준이며, 로봇 제어 직전 base frame으로 변환해야 합니다.
CLASS_TO_PLACE_POSITION = {
    # right_top (x 0.0~0.45, y -0.35~0.0)을 좌/우로 나눠 배정
    'hammer': {'x': 0.1125, 'y': -0.175, 'z': 0.45},
    'screwdriver': {'x': 0.3375, 'y': -0.175, 'z': 0.45},

    # right_bottom (x 0.0~0.45, y 0.0~0.35)을 좌/우로 나눠 배정
    'wrench': {'x': 0.1125, 'y': 0.175, 'z': 0.45},
    'pliers': {'x': 0.3375, 'y': 0.175, 'z': 0.45},

    # left_bottom (x -0.45~0.0, y 0.0~0.35) — drill 하나뿐이라 중앙
    'drill': {'x': -0.225, 'y': 0.175, 'z': 0.45},
}


# 기본 작업공간 배치 규칙을 반환하는 함수
def get_default_zone_rules():
    return {
        'zones': DEFAULT_ZONES,
        'class_to_zone': CLASS_TO_ZONE,
        'class_to_place_position': CLASS_TO_PLACE_POSITION,
    }


# 감지된 물체 목록을 순회하며 정상 배치 물체와 오배치 물체를 구분하는 함수
def judge_workspace(objects, frame: str, zone_rules):
    zones = zone_rules.get('zones', {})
    class_to_zone = zone_rules.get('class_to_zone', {})
    class_to_place_position = zone_rules.get('class_to_place_position', {})

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
                    place_position=class_to_place_position.get(object_name),
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
def make_misplaced_object(detected_object, current_zone_name, expected_zone_name, expected_zone, place_position, frame):
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

        # 현재 구역과 원래 있어야 하는 구역입니다.
        'current_zone': current_zone_name,
        'expected_zone': expected_zone_name,
        'expected_zone_label': expected_zone.get('label'),

        # 이 클래스에 배정된 구역 내 고정 슬롯 좌표입니다. (클래스마다 달라 서로 겹치지 않습니다)
        # 지금은 camera frame 기준 좌표이고, 실제 로봇 제어 직전 robot base frame으로 변환해야 합니다.
        'place_position': place_position,
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