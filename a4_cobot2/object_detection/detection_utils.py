import json
import math
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
    """
    RealSense depth image가 16UC1이면 보통 mm 단위일 수 있고,
    32FC1이면 m 단위일 수 있다.

    여기서는 값이 10보다 크면 mm로 보고 m 단위로 변환한다.
    """
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
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    ppx = intrinsics["ppx"]
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