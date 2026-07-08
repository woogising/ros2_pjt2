# ============================================================
# object_detection/grasp_utils.py
# 역할:
#   - grasp_point_fix.md의 parallax 해결안 구현.
#   - mask 중심 1점 대신, mask 전체 픽셀을 base 포인트클라우드로 만들고
#     윗면(top face) 중심을 grasp 좌표로, 윗면 슬라이스 PCA로 각도를 구한다.
# 사용:
#   - detection_grasp.py(GraspDetectionNode)에서 호출.
# 주의:
#   - open3d 필요 (pip install open3d).
#   - Doosan base Z축이 "위=+Z"라고 가정한다. 반대면 compute_top_center_grasp의
#     percentile/슬라이스 부호를 뒤집어야 한다(문서 5절 참고).
# ============================================================
import math

import numpy as np
import open3d as o3d


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

    # camera(m) → mm 후 base 변환 (camera_position_to_base와 동일하게 *1000)
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
def compute_top_center_grasp(points_base, voxel_size=3.0, slice_t=10.0):
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
    grasp_z = float(z_top)  # 접근 목표 높이는 중측 윗면 최상단
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
    return math.degrees(math.atan2(major[1], major[0]))
