# ============================================================
# workspace/grid_allocator.py
# 역할:
#   - 오배치 물체를 "구역 그리드의 교차점 코너부터 x축 방향으로 한 줄씩"
#     겹치지 않게 배치할 최종 place 좌표를 사전계산한다.
#   - organize_grid_placement.md 의 §3·§4·§6·§7(국면 1.5) 구현.
#
# 입력:
#   misplaced_objects: [{name, zone, width, length, angle?}, ...]
#       zone   : 'green' | 'yellow' | 'red' | 'blue' (배치 목표 구역)
#       width  : 파지 축(x) 방향 물체 폭 (mm)
#       length : 정렬 축(y) 방향 물체 길이 (mm)
#   placed_objects(선택): [{zone, x, y, width, length}, ...]
#       이미 그 구역에 올바르게 놓인 물체. 그 자리를 occupied로 막아
#       오배치 물체가 위에 겹치지 않게 한다(§7-a).
#
# 출력:
#   [{name, zone, placed, place_x, place_y, place_z, place_angle, cell}, ...]
#       placed=False 면 구역이 꽉 차 배치 실패(§9). place_* 는 None.
#
# 좌표계: base, mm. +x=로봇에서 멀어짐, +y=왼쪽.
# 이 모듈은 자체 완결형이라 단독 import / 단독 실행(__main__ 자가테스트)이 가능하다.
# (추후 ZONES 좌표는 workspace_judge_utils.DEFAULT_ZONES 에서 끌어와도 된다.)
# ============================================================
import math

# --- 튜닝 파라미터 (organize_grid_placement.md §10) ---
GRID_DIVISIONS = 20             # 구역당 x·y 분할 수
GRIPPER_CLEARANCE_GRASP = 20.0  # 파지 축(x) clearance (mm)
GRIPPER_CLEARANCE_ALIGN = 12.0  # 정렬 축(y) clearance (mm)
PLACE_Z = 14.0                  # 놓는 높이 (기존 zone place_position.z)
ALIGN_ANGLE_DEG = 90.0          # 물체 주축을 base +y와 평행하게 두는 목표 각(base)

# --- 구역 기하 + 시작 코너(교차점에 가장 가까운 코너) + 배치 방향 (§3) ---
# corner_x/y : 채우기를 시작하는 코너의 base 좌표
# sx : 한 줄을 채워가는 x 방향(+1/-1), sy : 다음 줄로 올라가는 y 방향(+1/-1)
ZONES = {
    "green":  {"x_min": 137.94, "x_max": 490.45, "y_min": 14.20,   "y_max": 328.53,
               "corner_x": 490.45, "corner_y": 14.20,  "sx": -1, "sy": +1},
    "yellow": {"x_min": 142.96, "x_max": 498.30, "y_min": -330.17, "y_max": -10.70,
               "corner_x": 498.30, "corner_y": -10.70, "sx": -1, "sy": -1},
    "red":    {"x_min": 517.10, "x_max": 923.67, "y_min": 1.64,    "y_max": 327.05,
               "corner_x": 517.10, "corner_y": 1.64,   "sx": +1, "sy": +1},
    "blue":   {"x_min": 508.94, "x_max": 922.62, "y_min": -321.72, "y_max": -13.98,
               "corner_x": 508.94, "corner_y": -13.98, "sx": +1, "sy": -1},
}


def _cell_size(zone):
    z = ZONES[zone]
    cell_x = (z["x_max"] - z["x_min"]) / GRID_DIVISIONS
    cell_y = (z["y_max"] - z["y_min"]) / GRID_DIVISIONS
    return cell_x, cell_y


# 물체가 차지하는 셀의 갯수
def _cells_for_size(size_mm, clearance_mm, cell_mm):
    n = math.ceil((size_mm + clearance_mm) / cell_mm)
    return max(1, min(n, GRID_DIVISIONS))


def _block_free(occ, i, j, cx, cy):
    for jj in range(j, j + cy):
        for ii in range(i, i + cx):
            if occ[ii][jj]:
                return False
    return True


def _mark(occ, i, j, cx, cy):
    for jj in range(j, j + cy):
        for ii in range(i, i + cx):
            occ[ii][jj] = True


def _cell_of(coord, corner, s, cell):
    """base 좌표가 속한 셀 인덱스(코너 기준). s=+1/-1 배치 방향."""
    return int(math.floor(s * (coord - corner) / cell))


def _mark_placed_object(occ, zone, obj):
    """이미 정배치된 물체가 (footprint+clearance) 로 덮는 셀을 occupied 로 막는다(§7-a).
    물체 방향을 모르므로 base x/y 축 정렬 박스로 보수적으로 처리한다."""
    z = ZONES[zone]
    cell_x, cell_y = _cell_size(zone)
    hx = (float(obj["width"]) + GRIPPER_CLEARANCE_GRASP) / 2.0
    hy = (float(obj["length"]) + GRIPPER_CLEARANCE_ALIGN) / 2.0
    ox, oy = float(obj["x"]), float(obj["y"])

    i0 = _cell_of(ox - hx, z["corner_x"], z["sx"], cell_x)
    i1 = _cell_of(ox + hx, z["corner_x"], z["sx"], cell_x)
    j0 = _cell_of(oy - hy, z["corner_y"], z["sy"], cell_y)
    j1 = _cell_of(oy + hy, z["corner_y"], z["sy"], cell_y)

    for i in range(max(0, min(i0, i1)), min(GRID_DIVISIONS - 1, max(i0, i1)) + 1):
        for j in range(max(0, min(j0, j1)), min(GRID_DIVISIONS - 1, max(j0, j1)) + 1):
            occ[i][j] = True


def _place_center(zone, i, j, cx, cy):
    """블록 좌하단 셀 (i, j) 크기 (cx, cy) 의 중심 base 좌표(§4)."""
    z = ZONES[zone]
    cell_x, cell_y = _cell_size(zone)
    place_x = z["corner_x"] + z["sx"] * (i + cx / 2.0) * cell_x
    place_y = z["corner_y"] + z["sy"] * (j + cy / 2.0) * cell_y
    return place_x, place_y


def allocate_placements(misplaced_objects, placed_objects=None):
    """오배치 물체들의 최종 place 좌표를 사전계산해 리스트로 반환한다."""
    placed_objects = placed_objects or []

    # 구역별로 묶는다(입력 순서 = 배치 대기열 순서).
    by_zone = {}
    for obj in misplaced_objects:
        by_zone.setdefault(obj["zone"], []).append(obj)

    results = []
    for zone, objs in by_zone.items():
        if zone not in ZONES:
            for obj in objs:
                results.append(_failed(obj, zone))
            continue

        cell_x, cell_y = _cell_size(zone)
        occ = [[False] * GRID_DIVISIONS for _ in range(GRID_DIVISIONS)]

        # (a) 정배치 물체가 덮는 셀 먼저 막기
        for p in placed_objects:
            if p.get("zone") == zone:
                _mark_placed_object(occ, zone, p)

        # (b) x축 줄 세우기(선반형 packing). 커서는 구역 내에서 물체 간 이어진다.
        cursor_i = 0
        cursor_j = 0
        row_height = 0  # 현재 줄의 최대 높이(셀) = 다음 줄로 올라갈 양

        for obj in objs:
            cx = _cells_for_size(float(obj["width"]), GRIPPER_CLEARANCE_GRASP, cell_x)
            cy = _cells_for_size(float(obj["length"]), GRIPPER_CLEARANCE_ALIGN, cell_y)

            spot = None
            while cursor_j + cy <= GRID_DIVISIONS:
                # 이 줄에 x로 안 들어가면 다음 줄로(누적 줄 높이만큼 올라감)
                if cursor_i + cx > GRID_DIVISIONS:
                    cursor_j += row_height if row_height > 0 else 1
                    cursor_i = 0
                    row_height = 0
                    continue
                if _block_free(occ, cursor_i, cursor_j, cx, cy):
                    spot = (cursor_i, cursor_j)
                    break
                cursor_i += 1  # 정배치 등으로 막혔으면 오른쪽으로 밀며 빈칸 탐색

            if spot is None:
                results.append(_failed(obj, zone))  # 구역 초과 → 배치 실패(§9)
                continue

            i, j = spot
            _mark(occ, i, j, cx, cy)
            place_x, place_y = _place_center(zone, i, j, cx, cy)
            results.append({
                "name": obj.get("name"),
                "zone": zone,
                "placed": True,
                "place_x": place_x,
                "place_y": place_y,
                "place_z": PLACE_Z,
                "place_angle": ALIGN_ANGLE_DEG,  # 주축 y평행 목표각(robot_arm이 손목각으로 변환)
                "cell": (i, j, cx, cy),
            })

            cursor_i = i + cx
            row_height = max(row_height, cy)

    return results


def _failed(obj, zone):
    return {
        "name": obj.get("name"),
        "zone": zone,
        "placed": False,
        "place_x": None,
        "place_y": None,
        "place_z": None,
        "place_angle": None,
        "cell": None,
    }


if __name__ == "__main__":
    # 간단 자가테스트: red 구역에 오배치 물체 3개 줄 세우기
    demo = [
        {"name": "hammer",      "zone": "red", "width": 40.0, "length": 180.0},
        {"name": "screwdriver", "zone": "red", "width": 30.0, "length": 200.0},
        {"name": "wrench",      "zone": "red", "width": 60.0, "length": 150.0},
    ]
    for r in allocate_placements(demo):
        if r["placed"]:
            print(f"{r['name']:12s} cell={r['cell']} "
                  f"place=({r['place_x']:.1f}, {r['place_y']:.1f}, {r['place_z']:.1f}) "
                  f"angle={r['place_angle']}")
        else:
            print(f"{r['name']:12s} 배치 실패")
