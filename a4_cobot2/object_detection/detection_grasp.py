# ============================================================
# object_detection/detection_grasp.py
# 역할:
#   - detection.py(ObjectDetectionNode)를 상속받아 grasp 좌표 계산을
#     "mask 중심 1점" 대신 "mask 전체 포인트클라우드 → 윗면 중심"으로 바꾼 버전.
#   - grasp_point_fix.md의 parallax 해결안 적용.
# 사용:
#   - setup.py의 object_detection_node entry point를 이 파일의 main으로 지정.
#   - 원래 버전으로 복귀하려면 entry point를 object_detection.detection:main 으로 되돌린다.
# 변경 범위:
#   - _scan_and_transform, handle_scan_pose 두 메서드만 override.
#   - ImgNode/서비스/스캔통신/프리뷰 등 나머지는 detection.py를 그대로 상속.
# ============================================================
import json

import numpy as np
import rclpy
from std_msgs.msg import Int32, String

from object_detection.detection import ObjectDetectionNode
from object_detection.detection_utils import make_scan_workspace_payload
from object_detection.grasp_utils import (
    deproject_mask_to_base,
    merge_clouds_by_name,
    compute_top_center_grasp,
    top_face_angle,
)


class GraspDetectionNode(ObjectDetectionNode):
    # 한 자세에서 감지된 물체마다 mask 전체를 base 포인트클라우드로 만들어 반환한다.
    # (좌표 1점이 아니라 클라우드를 누적해 마지막에 윗면 중심을 계산한다.)
    def _scan_and_transform(self, base_to_camera_matrix):
        detections = self.model.get_all_detections(self.img_node)

        depth_frame = self._wait_for_valid_data(
            self.img_node.get_depth_frame, "depth frame"
        )

        objects = []
        for detection in detections:
            mask = detection.get("mask")
            if mask is None:
                continue

            cloud = deproject_mask_to_base(
                mask, depth_frame, self.intrinsics, base_to_camera_matrix
            )
            if cloud is None or len(cloud) < 10:
                continue

            objects.append({
                "name": detection.get("name"),
                "class_id": detection.get("class_id"),
                "confidence": detection.get("confidence"),
                "box": detection.get("box"),
                "cloud": cloud,  # (N,3) base mm — 내부 누적용 (JSON엔 안 들어감)
            })

        return objects

    # 자세별 클라우드를 누적하고, 마지막 자세에서 윗면 중심 grasp/각도를 계산해 발행한다.
    def handle_scan_pose(self, msg):
        data = list(msg.data)
        if len(data) < 18:
            self.get_logger().error(f"scan_pose_transform 데이터 길이 오류: {len(data)}")
            return

        index = int(data[0])
        total = int(data[1])
        base_to_camera_matrix = np.array(data[2:18]).reshape(4, 4)

        self.get_logger().info(f"scan 자세 {index + 1}/{total} 물체 감지 시작")

        if index == 0:
            self.scan_accumulator = []

        try:
            objects = self._scan_and_transform(base_to_camera_matrix)
            self.scan_accumulator.extend(objects)
            self.get_logger().info(
                f"scan 자세 {index}: {len(objects)}개 감지 (누적 {len(self.scan_accumulator)})"
            )
            for obj in objects:
                centroid = obj["cloud"].mean(axis=0)
                self.get_logger().info(
                    f"  [자세{index}] {obj['name']}: {len(obj['cloud'])}pts, "
                    f"centroid=({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}) mm"
                )
        except Exception as exc:
            self.get_logger().error(f"scan 자세 {index} 처리 실패: {exc}")

        # 이 자세 캡처 완료를 robot_arm에 알려 다음 자세로 이동시킨다.
        self.scan_done_pub.publish(Int32(data=index))

        # 마지막 자세까지 끝나면 클라우드 병합 → 윗면 중심 grasp 계산 후 발행.
        if index >= total - 1:
            result_objects = self._build_grasp_objects(self.scan_accumulator)
            payload = make_scan_workspace_payload(
                objects=result_objects,
                skipped_count=0,
                raw_detection_count=len(self.scan_accumulator),
            )
            payload["frame"] = "base"

            message = String()
            message.data = json.dumps(payload, ensure_ascii=False)
            self.scanned_objects_pub.publish(message)

            self.get_logger().info(
                f"3자세 스캔 완료: {len(result_objects)}개 물체를 task_manager로 전송"
            )

    # 누적된 자세별 클라우드를 이름 기준 병합 → 윗면 중심 grasp 좌표/각도 계산.
    def _build_grasp_objects(self, accumulator):
        merged = merge_clouds_by_name(accumulator)

        result = []
        for name, item in merged.items():
            try:
                (gx, gy, gz), top_ds = compute_top_center_grasp(item["cloud"])
                angle = top_face_angle(top_ds)
            except Exception as exc:
                self.get_logger().error(f"{name} grasp 계산 실패: {exc}")
                continue

            result.append({
                "name": name,
                "class_id": item["class_id"],
                "confidence": item["confidence"],
                "box": item["box"],
                "position": {"x": gx, "y": gy, "z": gz},
                "angle": angle,
                "detected_pose_count": item["count"],
            })

            angle_str = f"{angle:.1f}" if angle is not None else "None"
            self.get_logger().info(
                f"  [grasp] {name}: pos=({gx:.1f}, {gy:.1f}, {gz:.1f}) mm, angle={angle_str}"
            )

        return result


def main(args=None):
    rclpy.init(args=args)
    node = GraspDetectionNode()

    try:
        rclpy.spin(node)
    finally:
        node.img_node.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
