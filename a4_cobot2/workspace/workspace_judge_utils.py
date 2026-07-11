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
from .grid_allocator import allocate_placements, ALIGN_ANGLE_DEG
from .grid_allocator import allocate_tidy  # 구역 내 재정렬(tidy) 배정

# footprint 계산이 실패해 width/length가 없을 때 배치에 쓸 기본 물체 크기(mm).
DEFAULT_OBJECT_SIZE_MM = 60.0

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

    # 오배치 물체들의 최종 place 좌표를 그리드로 사전계산해 place_position/place_angle을 채움.
    # 구역 중앙 단일 배치 → 물체끼리 안 겹치는 그리드 줄 세우기
    apply_grid_placement(misplaced_objects, normal_objects)

    # 구역 '안'에 있지만 그리드 슬롯에서 벗어난 정배치 물체도 코너부터 줄 세운다(재정렬).
    # 재정렬 항목은 misplaced_objects '끝'에 붙어 오배치 이동이 끝난 뒤 실행된다.
    apply_zone_tidy(misplaced_objects, normal_objects, unknown_rule_objects, frame)

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
        # 재정렬(tidy)로 다시 집을 때 손목 회전에 쓸 물체 각도(base, deg)
        'angle': detected_object.get('angle'),
        # 그리드 점유 마킹용 크기(mm). 이미 올바른 자리를 차지하므로 그 위엔 안 놓는다.
        'width': detected_object.get('width'),
        'length': detected_object.get('length'),
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

        # 그리드 배치용 물체 크기(mm). detection footprint 결과.
        'width': detected_object.get('width'),
        'length': detected_object.get('length'),

        # 현재 구역과 원래 있어야 하는 구역입니다.
        'current_zone': current_zone_name,
        'expected_zone': expected_zone_name,
        'expected_zone_label': expected_zone.get('label'),

        # 원래 있어야 하는 zone의 대표 위치입니다. base 좌표계(mm) 기준입니다.
        # 기본값은 zone 중앙(place_position)이며, apply_grid_placement가
        # 그리드 셀 좌표로 덮어씁니다. 배치 실패 시 이 중앙값이 fallback으로 남습니다.
        'place_position': expected_zone.get('place_position'),
        # 놓을 때 물체 주축을 y평행으로 두는 목표각(base, deg). 그리드가 채움.
        'place_angle': None,
        'place_frame': frame,

        'reason': 'outside_expected_zone',
    }


# 오배치 물체들의 place 좌표를 그리드로 사전계산해 각 물체 dict에 채운다.
# organize_grid_placement.md §7 국면 1.5.
def apply_grid_placement(misplaced_objects, normal_objects):
    if not misplaced_objects:
        return

    # 오배치 물체 → allocator 입력. name에 인덱스를 넣어 결과를 되짚는다.
    alloc_inputs = []
    for idx, obj in enumerate(misplaced_objects):
        alloc_inputs.append({
            'name': str(idx),
            'zone': obj.get('expected_zone'),
            'width': _size_or_default(obj.get('width')),
            'length': _size_or_default(obj.get('length')),
        })

    # 점유 마킹 → 지금 각 구역에 '실제로 놓여 있는' 모든 물체를 현재 위치/현재 구역으로 막는다.
    # 정배치 물체뿐 아니라, 아직 안 옮겨진 오배치 물체(예: 사과 자리에 있는 망치)도 포함해야
    # 그 위에 다른 물체를 배정하지 않는다. (버그: normal만 막아 오배치 위에 놓이던 문제)
    placed_inputs = []
    for obj in list(normal_objects) + list(misplaced_objects):
        pos = obj.get('position') or {}
        current_zone = obj.get('current_zone')
        # 현재 위치가 4개 구역 안에 있는 것만 점유로 잡는다(밖이면 배치에 영향 없음).
        if current_zone not in DEFAULT_ZONES:
            continue
        if pos.get('x') is None or pos.get('y') is None:
            continue
        placed_inputs.append({
            'zone': current_zone,
            'x': pos.get('x'),
            'y': pos.get('y'),
            'width': _size_or_default(obj.get('width')),
            'length': _size_or_default(obj.get('length')),
        })

    results = allocate_placements(alloc_inputs, placed_objects=placed_inputs)
    by_idx = {r['name']: r for r in results}

    for idx, obj in enumerate(misplaced_objects):
        r = by_idx.get(str(idx))
        if r is not None and r['placed']:
            obj['place_position'] = {
                'x': r['place_x'], 'y': r['place_y'], 'z': r['place_z'],
            }
            obj['place_angle'] = r['place_angle']
            obj['place_cell'] = r['cell']
            obj['place_failed'] = False
        else:
            # 배치 실패: place_position은 zone 중앙 fallback 유지, 표시만 남긴다.
            obj['place_angle'] = ALIGN_ANGLE_DEG
            obj['place_failed'] = True


def _size_or_default(value):
    try:
        v = float(value)
        if v > 0:
            return v
    except (TypeError, ValueError):
        pass
    return DEFAULT_OBJECT_SIZE_MM


# 구역 안 정배치 물체들을 그리드 코너부터 줄 세우는 재정렬(tidy) 항목을 만든다.
# 반드시 apply_grid_placement '뒤'에 호출한다(반입 물체의 새 슬롯을 피해야 하므로).
# 만들어진 재정렬 항목은 misplaced_objects 끝에 append → 로봇 실행 순서가
# [오배치 이동 먼저 → 재정렬 나중]이 되어, 떠날 물체가 비운 코너까지 채울 수 있다.
def apply_zone_tidy(misplaced_objects, normal_objects, unknown_rule_objects, frame):
    if not normal_objects:
        return

    # 재정렬 대상: 정배치 물체(현재 파지점이 곧 pick 위치)
    tidy_inputs = []
    for idx, obj in enumerate(normal_objects):
        pos = obj.get('position') or {}
        if pos.get('x') is None or pos.get('y') is None:
            continue
        tidy_inputs.append({
            'name': str(idx),
            'zone': obj.get('expected_zone'),
            'x': float(pos['x']),
            'y': float(pos['y']),
            'width': _size_or_default(obj.get('width')),
            'length': _size_or_default(obj.get('length')),
        })

    # 재정렬 '실행 시점'(오배치 이동 완료 후)에도 구역을 차지하고 있을 것들:
    #   ① unknown 물체 — 안 움직이므로 현재 자리
    #   ② 오배치 물체가 '새로 놓일' 자리 — place_position (배치 실패 fallback=zone 중앙 포함)
    # 오배치 물체의 '현재' 자리는 그때쯤 비어 있으므로 넣지 않는다.
    occupied = []
    for obj in (unknown_rule_objects or []):
        pos = obj.get('position') or {}
        zone = obj.get('current_zone')
        if zone not in DEFAULT_ZONES or pos.get('x') is None or pos.get('y') is None:
            continue
        occupied.append({
            'zone': zone,
            'x': float(pos['x']),
            'y': float(pos['y']),
            'width': _size_or_default(obj.get('width')),
            'length': _size_or_default(obj.get('length')),
        })
    for obj in misplaced_objects:
        place = obj.get('place_position') or {}
        zone = obj.get('expected_zone')
        if zone not in DEFAULT_ZONES or place.get('x') is None or place.get('y') is None:
            continue
        occupied.append({
            'zone': zone,
            'x': float(place['x']),
            'y': float(place['y']),
            'width': _size_or_default(obj.get('width')),
            'length': _size_or_default(obj.get('length')),
        })

    results = allocate_tidy(tidy_inputs, occupied_footprints=occupied)

    # 반드시 '배정 순서'(results 순서 = 코너 가까운 순)대로 append 한다.
    # 앞 순번이 비운 자리를 뒤 순번이 쓸 수 있는 건 이 실행 순서가 지켜질 때만 안전하다.
    for r in results:
        # 슬롯 배정 성공 + 허용오차 초과(이동 필요)인 물체만 재정렬 항목으로 만든다.
        if not r.get('placed') or not r.get('moved'):
            continue
        obj = normal_objects[int(r['name'])]
        misplaced_objects.append(make_realign_object(obj, r, frame))


# 재정렬 이동 항목을 오배치(misplaced)와 같은 형식으로 만든다.
# dict 모양이 같아 task_manager/robot_arm이 수정 없이 그대로 처리한다.
def make_realign_object(normal_object, tidy_result, frame):
    return {
        'name': normal_object.get('name'),
        'class_id': normal_object.get('class_id'),
        'confidence': normal_object.get('confidence'),
        'box': normal_object.get('box'),
        'center': normal_object.get('center'),
        'depth': normal_object.get('depth'),

        # 현재 위치가 곧 pick 위치
        'position': normal_object.get('position'),
        'pick_position': normal_object.get('position'),

        # 파지 시 손목 회전용 물체 각도(base, deg)
        'angle': normal_object.get('angle'),

        'width': normal_object.get('width'),
        'length': normal_object.get('length'),

        # 같은 구역 안 이동이므로 current == expected
        'current_zone': normal_object.get('current_zone'),
        'expected_zone': normal_object.get('expected_zone'),
        'expected_zone_label': normal_object.get('expected_zone_label'),

        # 재정렬 목표: 그리드 슬롯 중심, 주축 y평행
        'place_position': {
            'x': tidy_result['place_x'],
            'y': tidy_result['place_y'],
            'z': tidy_result['place_z'],
        },
        'place_angle': tidy_result['place_angle'],
        'place_cell': tidy_result['cell'],
        'place_failed': False,
        'place_frame': frame,

        # 올바른 구역이지만 그리드 슬롯에서 벗어나 다시 줄 세우는 물체
        'reason': 'untidy_in_zone',
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