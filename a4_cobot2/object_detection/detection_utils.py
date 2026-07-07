# ============================================================
# object_detection/detection_utils.py
# 역할:
#   - /scan_workspace에서 쓰는 JSON 파싱, bbox 중심 계산, depth 추출,
#     pixel -> camera 3D 변환, 응답 payload 생성 유틸입니다.
# 데이터 단위:
#   - RealSense depth가 mm처럼 큰 값이면 m 단위로 변환합니다.
#   - 반환 position은 camera_color_optical_frame 기준 m 단위입니다.
# ============================================================
import json
import math

import cv2
import numpy as np


# scan 대상 물체 이름 목록을 JSON 문자열에서 파싱한다.
def parse_target_names_json(target_names_json):
    """
    입력 예:
    ''
    '{"targets": ["hammer", "screwdriver"]}'
    '["hammer", "screwdriver"]'

    반환:
    None 또는 ["hammer", "screwdriver"]
    """
    if not target_names_json:
        return None

    try:
        parsed = json.loads(target_names_json)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        targets = parsed.get("targets")
    elif isinstance(parsed, list):
        targets = parsed
    else:
        return None

    if not targets:
        return None

    return [str(name) for name in targets]


# bbox [x1, y1, x2, y2]의 중심 픽셀 좌표를 계산한다.
def get_box_center(box):
    cx = int((box[0] + box[2]) / 2)
    cy = int((box[1] + box[3]) / 2)
    return cx, cy


# depth frame에서 중심 픽셀 주변 window의 유효 depth median 값을 구한다.
def get_depth_from_frame(frame, x, y, window_size=5):

    if frame is None:
        return None

    height, width = frame.shape[:2]

    if x < 0 or y < 0 or x >= width or y >= height:
        return None

    half = window_size // 2
    x1 = max(0, x - half)
    x2 = min(width, x + half + 1)
    y1 = max(0, y - half)
    y2 = min(height, y + half + 1)

    patch = np.asarray(frame[y1:y2, x1:x2], dtype=np.float32)
    valid = patch[np.isfinite(patch) & (patch > 0)]

    if valid.size == 0:
        return None

    depth = float(np.median(valid))

    if depth > 10.0:
        depth *= 0.001

    if math.isnan(depth) or depth <= 0.0:
        return None

    return depth


# 픽셀 좌표와 camera intrinsics를 이용해 카메라 좌표계 3D 좌표로 변환한다.
def pixel_to_camera_coords(x, y, z, intrinsics):
    fx = intrinsics["fx"] # 카메라의 x축 방향 초점거리 (pixel)
    fy = intrinsics["fy"]
    ppx = intrinsics["ppx"] # 카메라 초점의 중심값 (pixel)
    ppy = intrinsics["ppy"]

    return (
        (x - ppx) * z / fx,
        (y - ppy) * z / fy,
        z,
    )


# YOLO detection 하나에 depth와 3D position 정보를 붙여 object dict로 만든다.
def build_detected_object(detection, depth_frame, intrinsics):
    box = detection.get("box")

    if box is None or len(box) != 4:
        return None

    cx, cy = get_box_center(box)

    depth = get_depth_from_frame(depth_frame, cx, cy)
    if depth is None:
        return None

    position = pixel_to_camera_coords(cx, cy, depth, intrinsics)

    return {
        "name": detection.get("name"),
        "class_id": int(detection.get("class_id", -1)),
        "confidence": float(detection.get("confidence", 0.0)),
        "box": [float(v) for v in box],
        "center": {
            "x": int(cx),
            "y": int(cy),
        },
        "depth": float(depth),
        "position": {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
        },
    }


# 카메라 좌표계(m) position을 base<-camera 4x4 행렬로 base 좌표계(mm) position으로 변환한다.
# camera position은 m, 로봇 posx와 캘리브레이션 행렬은 mm 단위라 1000을 곱해 맞춘다.
def camera_position_to_base(position, base_to_camera_matrix):
    camera_point = np.array([
        float(position["x"]) * 1000.0,
        float(position["y"]) * 1000.0,
        float(position["z"]) * 1000.0,
        1.0,
    ])
    base_point = base_to_camera_matrix @ camera_point
    return {
        "x": float(base_point[0]),
        "y": float(base_point[1]),
        "z": float(base_point[2]),
    }


# segmentation mask에서 PCA로 물체 긴 축 방향(이미지 픽셀 기준 단위벡터)을 구한다.
def mask_pca_axis(binary_mask):
    ys, xs = np.where(binary_mask > 0)
    if len(xs) < 10:
        return None

    points = np.column_stack((xs, ys)).astype(np.float32)
    _, eigenvectors = cv2.PCACompute(points, mean=None)
    vx, vy = eigenvectors[0]
    return float(vx), float(vy)


# 물체 긴 축(이미지 픽셀 벡터)을 base 좌표계 각도(deg)로 변환한다.
# 축 위 두 점을 (중심 depth로) camera->base 변환해 base 평면 각도를 구하므로,
# 카메라 자세(eye-in-hand)와 무관하게 일관된 base 각도가 나온다.
def image_axis_to_base_angle(cx, cy, vx, vy, depth, intrinsics, base_to_camera_matrix, span_px=20.0):
    def to_base_xy(px, py):
        cam = pixel_to_camera_coords(px, py, depth, intrinsics)
        base = camera_position_to_base(
            {"x": cam[0], "y": cam[1], "z": cam[2]}, base_to_camera_matrix
        )
        return base["x"], base["y"]

    x1, y1 = to_base_xy(cx - span_px * vx, cy - span_px * vy)
    x2, y2 = to_base_xy(cx + span_px * vx, cy + span_px * vy)
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


# 여러 관측 자세에서 감지된 같은 이름 물체들을 base 좌표 평균으로 하나로 합친다.
# (물체는 클래스당 1개라는 전제이므로 이름이 같으면 같은 물체로 본다.)
def merge_objects_by_name(objects):
    groups = {}
    for obj in objects:
        groups.setdefault(obj["name"], []).append(obj)

    merged = []
    for group in groups.values():
        best = max(group, key=lambda o: o.get("confidence", 0.0))
        merged_object = dict(best)
        merged_object["position"] = {
            "x": sum(o["position"]["x"] for o in group) / len(group),
            "y": sum(o["position"]["y"] for o in group) / len(group),
            "z": sum(o["position"]["z"] for o in group) / len(group),
        }
        merged_object["detected_pose_count"] = len(group)
        merged.append(merged_object)

    return merged


# scan_workspace 성공 응답용 payload를 생성한다.
def make_scan_workspace_payload(objects, skipped_count, raw_detection_count):
    return {
        "task": "scan_workspace",
        "frame": "camera_color_optical_frame",
        "objects": objects,
        "summary": {
            "detected_count": len(objects),
            "skipped_count": skipped_count,
            "raw_detection_count": raw_detection_count,
        },
    }


# scan_workspace 실패 응답용 payload를 생성한다.
def make_scan_workspace_error_payload(error_message):
    return {
        "task": "scan_workspace",
        "frame": "camera_color_optical_frame",
        "objects": [],
        "summary": {
            "detected_count": 0,
            "skipped_count": 0,
            "raw_detection_count": 0,
        },
        "error_message": str(error_message),
    }