# ============================================================
# object_detection/detection_utils.py
# 역할:
#   - YOLO 2D 감지 유틸(파싱 / bbox 중심 / depth / pixel→camera / payload)과
#     3D 포인트클라우드 grasp 유틸(mask deproject / 병합 / 윗면 중심 / 각도)을 담는다.
# 데이터 단위:
#   - RealSense depth가 mm처럼 큰 값이면 m로 변환한다.
#   - 3D cloud / grasp 좌표는 base 좌표계(mm) 기준이다.
# 주의:
#   - 3D cloud 부분은 open3d 필요 (pip install open3d).
#   - Doosan base Z축이 "위=+Z"라고 가정 (compute_top_center_grasp).
# ============================================================
import json
import math

import numpy as np
import open3d as o3d


# ============================================================
# YOLO / 2D detection 부분
# ============================================================

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


# ============================================================
# 3D cloud / grasp 부분
# ============================================================

# object mask의 모든 픽셀을 aligned depth로 3D화해 base 좌표계(mm) 포인트클라우드로 변환한다.
def deproject_mask_to_base(mask, depth_frame, intrinsics, base_to_camera_matrix):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    height, width = depth_frame.shape[:2]
    in_bounds = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs, ys = xs[in_bounds], ys[in_bounds]
    if len(xs) == 0:
        return None

    depths = np.asarray(depth_frame[ys, xs], dtype=np.float32)
    # RealSense 16UC1이면 mm 단위 → m로 변환 (get_depth_from_frame과 동일 규칙)
    depths = np.where(depths > 10.0, depths * 0.001, depths)

    good = np.isfinite(depths) & (depths > 0.0)
    xs, ys, depths = xs[good], ys[good], depths[good]
    if len(xs) == 0:
        return None

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    ppx, ppy = intrinsics["ppx"], intrinsics["ppy"]

    # 픽셀 → 카메라 좌표계(m)
    cam_x = (xs - ppx) * depths / fx
    cam_y = (ys - ppy) * depths / fy
    cam_z = depths

    # camera(m) → mm 후 base 변환 (posx / 캘리브레이션 행렬이 mm라 *1000)
    cam_h = np.stack(
        [cam_x * 1000.0, cam_y * 1000.0, cam_z * 1000.0, np.ones_like(cam_x)],
        axis=0,
    )  # (4, N)
    base_h = base_to_camera_matrix @ cam_h  # (4, N)
    return base_h[:3].T  # (N, 3) mm


# 자세별로 모인 물체({name, cloud, ...})들을 이름 기준으로 포인트클라우드 concat한다.
# (좌표를 평균내는 게 아니라 클라우드 자체를 합쳐 윗면 커버리지를 서로 보완한다.)
def merge_clouds_by_name(items):
    groups = {}
    for item in items:
        groups.setdefault(item["name"], []).append(item)

    merged = {}
    for name, group in groups.items():
        best = max(group, key=lambda o: o.get("confidence", 0.0))
        cloud = np.vstack([o["cloud"] for o in group])
        merged[name] = {
            "class_id": best["class_id"],
            "confidence": best["confidence"],
            "box": best["box"],
            "cloud": cloud,
            "count": len(group),
        }
    return merged


# base 포인트클라우드에서 윗면 중심 grasp 좌표(mm)와 윗면 슬라이스 점들을 반환한다.
def compute_top_center_grasp(points_base, voxel_size=3.0, slice_t=18.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_base)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        pts = np.asarray(points_base)  # SOR가 전부 지우면 원본 사용

    # raw max 대신 robust max (flying pixel 방지). base +Z가 위라고 가정.
    z_top = np.percentile(pts[:, 2], 95)

    # 윗면 슬라이스만 선택
    top_points = pts[pts[:, 2] > z_top - slice_t]
    if len(top_points) == 0:
        top_points = pts

    # 밀림 편향 제거: voxel downsample 후 centroid
    top_pcd = o3d.geometry.PointCloud()
    top_pcd.points = o3d.utility.Vector3dVector(top_points)
    top_pcd = top_pcd.voxel_down_sample(voxel_size=voxel_size)
    top_ds = np.asarray(top_pcd.points)
    if len(top_ds) == 0:
        top_ds = top_points

    grasp_x = float(np.mean(top_ds[:, 0]))
    grasp_y = float(np.mean(top_ds[:, 1]))
    grasp_z = float(z_top)  # 접근 목표 높이는 윗면 최상단
    return (grasp_x, grasp_y, grasp_z), top_ds


# 윗면 슬라이스 점들의 base XY 평면 PCA로 물체 긴 축 각도(deg)를 계산한다.
def top_face_angle(top_ds):
    xy = np.asarray(top_ds)[:, :2]
    if len(xy) < 3:
        return None

    xy_centered = xy - xy.mean(axis=0)
    cov = np.cov(xy_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]  # 긴 축

    angle_deg = math.degrees(math.atan2(major[1], major[0]))

    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    return angle_deg



# 전체 포인트클라우드를 주축(angle_deg 방향)/부축에 투영해 물체 폭/길이(mm)를 구한다.
#   width  = 짧은 축(부축=파지 축) 방향 범위
#   length = 긴 축(주축) 방향 범위
#   min-max가 아니라 robust percentile을 써서 flying pixel/노이즈 과대추정을 막는다.
#   angle_deg는 top_face_angle 결과를 그대로 재사용한다.
def footprint_extent(points_base, angle_deg, percentile=(2.0, 98.0)):
    pts = np.asarray(points_base)
    if angle_deg is None or len(pts) < 3:
        return None, None

    xy = pts[:, :2]
    theta = math.radians(angle_deg)
    major = np.array([math.cos(theta), math.sin(theta)])
    minor = np.array([-math.sin(theta), math.cos(theta)])

    centered = xy - xy.mean(axis=0)
    proj_major = centered @ major
    proj_minor = centered @ minor

    lo, hi = percentile
    length = float(np.percentile(proj_major, hi) - np.percentile(proj_major, lo))
    width = float(np.percentile(proj_minor, hi) - np.percentile(proj_minor, lo))
    return width, length
