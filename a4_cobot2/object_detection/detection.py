# ============================================================
# object_detection/detection.py
# 역할:
#   - ObjectDetectionNode 서비스 서버입니다.
#   - /scan_workspace 요청을 받으면 YOLO로 작업공간 전체를 한 번 스캔하고,
#     RGB bbox + aligned depth + camera intrinsics를 이용해 물체별 3D 위치를 만듭니다.
# 주요 service:
#   - scan_workspace: target 목록을 받아 감지된 objects JSON 반환
# 현재 구조:
#   - get_3d_position은 주석 처리되어 있고, 전체 스캔은 scan_workspace 중심입니다.
# ============================================================
import json
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Float64MultiArray, Int32, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from od_msg.srv import SrvDepthPosition, ScanWorkspace
from object_detection.realsense import ImgNode
from object_detection.yolo import YoloModel
from object_detection.detection_utils import (
    parse_target_names_json,
    build_detected_object,
    make_scan_workspace_payload,
    make_scan_workspace_error_payload,
    deproject_mask_to_base,
    merge_clouds_by_name,
    compute_top_center_grasp,
    top_face_angle,
)


PACKAGE_NAME = "a4_cobot2"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

# YOLO 인식 프리뷰 이미지를 발행하는 주기(초). 값을 키우면 추론 부하가 줄어든다.
DETECTION_PREVIEW_PERIOD_SEC = 3.0


class ObjectDetectionNode(Node):
    # ObjectDetectionNode를 초기화하고 필요한 서비스 서버를 생성한다.
    def __init__(self, model_name="yolo"):
        super().__init__("object_detection_node")

        # ImgNode는 RealSense topic을 구독해서 최신 RGB/depth/camera_info를 보관합니다.
        # ObjectDetectionNode 자신과 별도 Node로 만들었기 때문에, 데이터 갱신 시 rclpy.spin_once(self.img_node)를 호출합니다.
        self.img_node = ImgNode()

        # YOLO 모델 wrapper입니다. 현재는 model_name='yolo'만 지원합니다.
        self.model = self._load_model(model_name)

        # camera intrinsics는 pixel_to_camera_coords()에 필요합니다.
        # CameraInfo가 들어올 때까지 기다린 뒤 저장합니다.
        self.intrinsics = self._wait_for_valid_data(self.img_node.get_camera_intrinsic, "camera intrinsics")

        # self.create_service(SrvDepthPosition, "get_3d_position", self.handle_get_depth)

        self.create_service(ScanWorkspace, "scan_workspace", self.handle_scan_workspace)

        # 3자세 스캔 흐름:
        #   robot_arm이 각 관측 자세에서 base<-camera 변환 행렬을 /scan_pose_transform으로 보내면
        #   이 노드가 그 순간 물체를 감지해 base 좌표로 변환하고 누적한다.
        #   마지막 자세까지 끝나면 병합 결과를 /scanned_objects_base로 task_manager에 보낸다.
        self.scan_accumulator = []

        # current_scan_mode:
        #   TaskManagerNode가 /workspace_scan_mode로 알려주는 현재 스캔 목적입니다.
        #   check_workspace: 최초 확인 스캔
        #   recheck_workspace: 로봇 정리 후 최종 재검증 스캔
        self.current_scan_mode = "check_workspace"

        # scan_image_records:
        #   최종 재검증 3자세 스캔 중 실제로 저장한 이미지 경로 목록입니다.
        #   최초 확인 스캔에서는 비어 있고, 마지막 재검증 payload에만 포함됩니다.
        self.scan_image_records = []
        self.scan_session_id = None

        # scan_image_dir:
        #   사용자가 확인하기 쉽도록 가능하면 source tree의 notification/scan_images에 저장합니다.
        self.declare_parameter("scan_image_dir", "")
        scan_image_dir_param = self.get_parameter("scan_image_dir").get_parameter_value().string_value
        self.scan_image_dir = self._resolve_scan_image_dir(scan_image_dir_param)
        os.makedirs(self.scan_image_dir, exist_ok=True)

        self.scan_mode_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.scan_mode_sub = self.create_subscription(
            String,
            "/workspace_scan_mode",
            self.workspace_scan_mode_callback,
            self.scan_mode_qos,
        )

        self.scan_pose_sub = self.create_subscription(
            Float64MultiArray, "/scan_pose_transform", self.handle_scan_pose, 10
        )
        self.scan_done_pub = self.create_publisher(Int32, "/scan_capture_done", 10)
        self.scanned_objects_pub = self.create_publisher(String, "/scanned_objects_base", 10)

        # YOLO 인식 결과(박스/마스크/라벨)를 그린 이미지를 발행해
        # rqt_image_view 등으로 실시간 확인할 수 있게 한다.
        self.detection_image_pub = self.create_publisher(Image, "/yolo_detection_image", 10)
        self.bridge = CvBridge()
        self.create_timer(DETECTION_PREVIEW_PERIOD_SEC, self.publish_detection_image)

        self.get_logger().info("ObjectDetectionNode initialized.")

    # 최신 카메라 프레임에 YOLO 인식 결과를 그려 /yolo_detection_image 로 발행한다.
    def publish_detection_image(self):
        rclpy.spin_once(self.img_node, timeout_sec=0.05)  # 최신 컬러 프레임 수신
        frame = self.img_node.get_color_frame()
        if frame is None:
            return

        try:
            results = self.model.model(frame, verbose=False, retina_masks=True)
            annotated = results[0].plot()  # 박스/마스크/라벨이 그려진 BGR 이미지
            self._draw_angle_overlays(annotated, results[0])  # 중심 + 각도 짝대기
            self.detection_image_pub.publish(
                self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            )
        except Exception as exc:
            self.get_logger().warn(f"detection preview 발행 실패: {exc}")

    # 각 물체 mask의 중심과 PCA 긴 축(각도 방향)을 이미지에 짝대기로 그려 디버깅한다.
    # (여기 각도는 "이미지 평면" mask PCA이며, 실제 파지 각도는 base 좌표계라 완전히 같지는 않다.)
    def _draw_angle_overlays(self, image, result):
        if result.masks is None:
            return

        masks = result.masks.data.cpu().numpy()
        for m in masks:
            ys, xs = np.where(m > 0.5)
            if len(xs) < 10:
                continue

            cx, cy = int(xs.mean()), int(ys.mean())
            points = np.column_stack((xs, ys)).astype(np.float32)
            _, eigenvectors = cv2.PCACompute(points, mean=None)
            vx, vy = float(eigenvectors[0][0]), float(eigenvectors[0][1])

            length = 40
            p1 = (int(cx - length * vx), int(cy - length * vy))
            p2 = (int(cx + length * vx), int(cy + length * vy))

            cv2.line(image, p1, p2, (0, 255, 255), 2)        # 노란색: 긴 축(각도 방향)
            cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)  # 빨간색: 중심점

    # scan image 저장 경로를 결정하는 함수
    def _resolve_scan_image_dir(self, scan_image_dir_param: str) -> str:
        if scan_image_dir_param is not None and scan_image_dir_param.strip() != "":
            return os.path.abspath(os.path.expanduser(scan_image_dir_param.strip()))

        env_dir = os.getenv("A4_COBOT2_SCAN_IMAGE_DIR", "").strip()
        if env_dir != "":
            return os.path.abspath(os.path.expanduser(env_dir))

        # 사용자의 현재 workspace 구조를 우선 사용합니다.
        preferred_candidates = [
            os.path.expanduser("~/a4_cobot2_ws/src/a4_cobot2/notification/scan_images"),
            os.path.join(os.getcwd(), "src", "a4_cobot2", "notification", "scan_images"),
        ]

        for candidate in preferred_candidates:
            parent = os.path.dirname(candidate)
            if os.path.isdir(parent):
                return candidate

        # fallback: 현재 Python package 위치 기준 notification/scan_images
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(package_root, "notification", "scan_images")

    # TaskManagerNode가 알려준 현재 스캔 목적을 저장하는 함수
    def workspace_scan_mode_callback(self, msg: String):
        mode = msg.data.strip()

        if mode not in ["check_workspace", "recheck_workspace"]:
            self.get_logger().warn(f"알 수 없는 workspace scan mode: {mode}")
            return

        self.current_scan_mode = mode
        self.get_logger().info(f"workspace scan mode updated: {self.current_scan_mode}")

    # 현재 스캔이 로봇 정리 후 최종 재검증 스캔인지 판단하는 함수
    def should_save_scan_images(self) -> bool:
        return self.current_scan_mode == "recheck_workspace"

    # 모델 이름에 따라 사용할 detection 모델 인스턴스를 반환한다.
    def _load_model(self, name):
        if name.lower() == "yolo":
            return YoloModel()

        raise ValueError(f"Unsupported model: {name}")

    # 작업공간 전체를 스캔해서 탐지된 물체 목록을 JSON 문자열로 반환한다.
    def handle_scan_workspace(self, request, response):
        self.get_logger().info(f"Received scan_workspace request: {request}")

        try:
            target_names = parse_target_names_json(request.target_names_json)

            detections = self.model.get_all_detections(
                self.img_node,
                target_names=target_names,
            )

            depth_frame = self._wait_for_valid_data(
                self.img_node.get_depth_frame,
                "depth frame",
            )

            objects = []
            skipped_count = 0

            for detection in detections:
                obj = build_detected_object(
                    detection=detection,
                    depth_frame=depth_frame,
                    intrinsics=self.intrinsics,
                )

                if obj is None:
                    skipped_count += 1
                    self.get_logger().warn(f"Skipped invalid detection: {detection}")
                    continue

                objects.append(obj)

            payload = make_scan_workspace_payload(
                objects=objects,
                skipped_count=skipped_count,
                raw_detection_count=len(detections),
            )

            response.success = True
            response.detected_objects_json = json.dumps(payload, ensure_ascii=False)
            response.message = f"scan workspace finished: {len(objects)} objects"

            self.get_logger().info(response.message)
            return response

        except Exception as exc:
            self.get_logger().error(f"scan_workspace failed: {exc}")

            payload = make_scan_workspace_error_payload(exc)

            response.success = False
            response.detected_objects_json = json.dumps(payload, ensure_ascii=False)
            response.message = f"scan_workspace failed: {exc}"

            return response

    # robot_arm이 관측 자세에서 보낸 base<-camera 변환 행렬을 받아,
    # 그 자세에서 물체 mask를 base 포인트클라우드로 만들어 누적하는 함수
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
            self.scan_image_records = []

            if self.should_save_scan_images():
                self.scan_session_id = time.strftime("recheck_%Y%m%d_%H%M%S")
                self.get_logger().info(
                    f"최종 재검증 스캔 이미지 저장을 시작합니다: {self.scan_image_dir}"
                )
            else:
                self.scan_session_id = None
                self.get_logger().info("최초 확인 스캔이므로 scan_images를 저장하지 않습니다.")

        try:
            objects = self._scan_and_transform(base_to_camera_matrix)
            self.scan_accumulator.extend(objects)

            if self.should_save_scan_images():
                image_record = self.save_scan_pose_images(index=index, total=total)
                if image_record is not None:
                    self.scan_image_records.append(image_record)

            self.get_logger().info(
                f"scan 자세 {index}: {len(objects)}개 감지 "
                f"(누적 {len(self.scan_accumulator)}), "
                f"scan_images={len(self.scan_image_records)}"
            )
            for obj in objects:
                centroid = obj["cloud"].mean(axis=0)
                self.get_logger().info(
                    f"  [자세{index}] {obj['name']}: {len(obj['cloud'])}pts, "
                    f"centroid=({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}) mm"
                )
        except Exception as exc:
            self.get_logger().error(f"scan 자세 {index} 처리 실패: {exc}")

        # 이 자세 캡처가 끝났음을 robot_arm에 알려 다음 자세로 이동시킨다.
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
            payload["scan_mode"] = self.current_scan_mode
            payload["scan_images"] = self.scan_image_records if self.should_save_scan_images() else []

            message = String()
            message.data = json.dumps(payload, ensure_ascii=False)
            self.scanned_objects_pub.publish(message)

            self.get_logger().info(
                f"3자세 스캔 완료: {len(result_objects)}개 물체를 task_manager로 전송"
            )

    # 현재 scan 자세에서 실제 카메라가 본 원본 이미지와 YOLO 표시 이미지를 notification/scan_images에 저장하는 함수
    def save_scan_pose_images(self, index: int, total: int):
        rclpy.spin_once(self.img_node, timeout_sec=0.05)

        frame = self.img_node.get_color_frame()
        if frame is None:
            self.get_logger().warn(f"scan 자세 {index + 1}/{total} 이미지 저장 실패: color frame 없음")
            return None

        if self.scan_session_id is None:
            self.scan_session_id = time.strftime("recheck_%Y%m%d_%H%M%S")

        base_name = f"{self.scan_session_id}_pose_{index + 1}_of_{total}"
        raw_image_path = os.path.join(self.scan_image_dir, f"{base_name}_raw.jpg")
        annotated_image_path = os.path.join(self.scan_image_dir, f"{base_name}_annotated.jpg")

        try:
            cv2.imwrite(raw_image_path, frame)

            annotated_saved = False
            try:
                results = self.model.model(frame, verbose=False, retina_masks=True)
                annotated = results[0].plot()
                cv2.imwrite(annotated_image_path, annotated)
                annotated_saved = True

            except Exception as exc:
                annotated_image_path = ""
                self.get_logger().warn(
                    f"scan 자세 {index + 1}/{total} annotated 이미지 저장 실패: {exc}"
                )

            record = {
                "index": int(index),
                "total": int(total),
                "scan_mode": self.current_scan_mode,
                "raw_image_path": raw_image_path,
                "annotated_image_path": annotated_image_path if annotated_saved else "",
            }

            self.get_logger().info(
                f"scan 자세 {index + 1}/{total} 이미지 저장 완료: {record}"
            )

            return record

        except Exception as exc:
            self.get_logger().warn(f"scan 자세 {index + 1}/{total} 이미지 저장 실패: {exc}")
            return None

    # 한 자세에서 감지된 물체마다 mask 전체를 base 포인트클라우드로 만들어 반환한다.
    def _scan_and_transform(self, base_to_camera_matrix):
        detections = self.model.get_all_detections(self.img_node)

        depth_frame = self._wait_for_valid_data(
            self.img_node.get_depth_frame,
            "depth frame",
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

    # 누적된 자세별 클라우드를 이름 기준 병합 → 윗면 중심 grasp 좌표/각도를 계산한다.
    def _build_grasp_objects(self, accumulator):
        merged = merge_clouds_by_name(accumulator)

        result = []
        for name, item in merged.items():
            try:
                (gx, gy, gz), top_ds = compute_top_center_grasp(item["cloud"])
                # 각도는 윗면 슬라이스(top_ds)가 아니라 전체 클라우드로 계산한다.
                # 망치 머리/드라이버 손잡이처럼 '가장 높은 부분'이 뭉툭하면
                # 슬라이스가 길쭉하지 않아 PCA 긴 축/짧은 축이 뒤바뀐다(각도 90° 튐).
                # 전체 발자국(손잡이+머리)은 확실히 길쭉해서 긴 축이 안정적으로 잡힌다.
                angle = top_face_angle(item["cloud"])
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


    # # target 물체 이름을 받아 해당 물체의 3D 좌표 [x, y, z]를 반환한다.
    # def handle_get_depth(self, request, response):
    #     self.get_logger().info(f"Received get_3d_position request: {request}")

    #     coords = self._compute_position(request.target)
    #     response.depth_position = [float(x) for x in coords]

    #     return response
    
    # # target 물체 하나의 카메라 좌표계 3D 위치를 계산한다.
    # def _compute_position(self, target):
    #     rclpy.spin_once(self.img_node)

    #     box, score = self.model.get_best_detection(self.img_node, target)
    #     if box is None or score is None:
    #         self.get_logger().warn("No detection found.")
    #         return 0.0, 0.0, 0.0

    #     self.get_logger().info(f"Detection: box={box}, score={score}")

    #     cx, cy = get_box_center(box)

    #     depth_frame = self._wait_for_valid_data(
    #         self.img_node.get_depth_frame,
    #         "depth frame",
    #     )

    #     depth = get_depth_from_frame(depth_frame, cx, cy)
    #     if depth is None:
    #         self.get_logger().warn("Depth out of range or invalid.")
    #         return 0.0, 0.0, 0.0

    #     return pixel_to_camera_coords(cx, cy, depth, self.intrinsics)

    # getter 함수가 유효한 데이터를 반환할 때까지 spin 하며 재시도한다.
    def _wait_for_valid_data(self, getter, description):
        data = getter()

        while data is None or (isinstance(data, np.ndarray) and not data.any()):
            rclpy.spin_once(self.img_node)
            self.get_logger().info(f"Retry getting {description}.")
            data = getter()

        return data


# ROS2 노드를 생성하고 spin한다.
def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()

    try:
        rclpy.spin(node)
    finally:
        node.img_node.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()