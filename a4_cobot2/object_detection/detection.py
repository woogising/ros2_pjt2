# ============================================================
# object_detection/detection.py
# 역할:
#   - ObjectDetectionNode는 카메라 프레임과 detector 결과를 이용해
#     작업공간 물체를 감지하고, 3자세 스캔 결과를 base 좌표계로 변환해 발행합니다.
#
# 사용 detector:
#   - 기본값은 EnsembleDetector입니다.
#   - EnsembleDetector 내부에서 YOLO-seg, RT-DETR, SAM2.1을 조합할 수 있습니다.
#   - model_name="yolo"로 생성하면 기존 YOLO 단독 모드로도 되돌릴 수 있습니다.
#
# 주요 통신:
#   - service  /scan_workspace       : 단일 요청-응답 방식의 카메라 좌표 감지용 호환 서비스
#   - sub      /workspace_scan_mode  : check_workspace / recheck_workspace 모드 수신
#   - sub      /scan_pose_transform  : robot_arm_node가 보낸 base<-camera 변환행렬 수신
#   - pub      /scan_capture_done    : 각 자세 감지 완료 ack
#   - pub      /scanned_objects_base : 3자세 병합 후 base 좌표 물체 목록 발행
#   - pub      /yolo_detection_image : HMI/VLM용 최종 앙상블 annotated 이미지
#
# 현재 메인 흐름:
#   - task_manager_node는 /start_workspace_scan을 robot_arm_node에 요청합니다.
#   - robot_arm_node는 3개의 관측 자세에서 /scan_pose_transform을 발행합니다.
#   - 이 노드는 각 자세에서 detector를 실행하고 mask를 depth와 결합해 point cloud를 만듭니다.
#   - 마지막 자세 후 물체별 cloud를 병합해 position/angle/width/length를 계산합니다.
#
# 주의:
#   - /scan_workspace 서비스는 과거/디버깅 호환용입니다.
#   - 실제 정리 작업에 쓰이는 좌표는 /scanned_objects_base의 base 좌표 결과입니다.
#   - annotated 이미지는 get_all_detections()가 만든 최종 bbox/mask를 그대로 사용합니다.
#   - topic 이름 /yolo_detection_image는 기존 HMI/VLM 호환을 위해 유지합니다.
# ============================================================
import json
import os
import time
import threading
from collections import Counter

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Float64MultiArray, Int32, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from od_msg.srv import ScanWorkspace
from .realsense import ImgNode
from .yolo import YoloModel
from .ensemble_detector import EnsembleDetector
from .detection_utils import (
    parse_target_names_json,
    build_detected_object,
    make_scan_workspace_payload,
    make_scan_workspace_error_payload,
    deproject_mask_to_base,
    merge_clouds_by_name,
    compute_top_center_grasp,
    top_face_angle,
    footprint_extent,
)


PACKAGE_NAME = "a4_cobot2"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

# HMI topic 발행 주기입니다. 무거운 모델 추론은 timer가 아니라 별도 worker에서 수행합니다.
DETECTION_PREVIEW_PERIOD_SEC = 0.10

# HMI 상시 화면은 빠른 YOLO 단독 preview를 사용합니다.
# 실제 3D 스캔과 VLM 저장 이미지는 기존처럼 최종 앙상블 결과를 사용합니다.
LIVE_PREVIEW_IMGSZ = 480
LIVE_PREVIEW_CONF = 0.50
LIVE_PREVIEW_MAX_DET = 20
LIVE_PREVIEW_IDLE_SLEEP_SEC = 0.01

# HMI bbox 라벨 표시 크기입니다.
# HMI 화면에서 축소되어도 클래스명이 잘 보이도록 기본 OpenCV 크기보다 크게 표시합니다.
HMI_CLASS_LABEL_FONT_SCALE = 1.2
HMI_CLASS_LABEL_THICKNESS = 3
HMI_BBOX_THICKNESS = 4

# 최종 mask를 원본 이미지 위에 겹칠 때 사용하는 투명도입니다.
FINAL_MASK_ALPHA = 0.35

# True일 때만 mask 중심/depth/camera/base 좌표를 detection별로 상세 출력합니다.
# 평상시에는 False로 두어 스캔 로그가 과도하게 늘어나는 것을 방지합니다.
DETECTION_COORD_DEBUG = False


class ObjectDetectionNode(Node):
    # ObjectDetectionNode를 초기화하고 카메라 구독 노드, detector, ROS 통신 인터페이스를 준비한다.
    def __init__(self, model_name="yolo"):
        super().__init__("object_detection_node")

        # ImgNode는 RealSense topic을 구독해서 최신 RGB/depth/camera_info를 보관합니다.
        # ObjectDetectionNode 자신과 별도 Node로 만들었기 때문에, 데이터 갱신 시 rclpy.spin_once(self.img_node)를 호출합니다.
        self.img_node = ImgNode()

        # detector wrapper입니다.
        # - ensemble: YOLO-seg + RT-DETR + SAM2.1 조합용 wrapper
        # - yolo    : 기존 YOLO-seg 단독 wrapper
        # 두 wrapper 모두 get_all_detections(img_node, target_names=None)를 제공해야 합니다.
        self.model = self._load_model(model_name)

        # camera intrinsics는 pixel_to_camera_coords()에 필요합니다.
        # CameraInfo가 들어올 때까지 기다린 뒤 저장합니다.
        self.intrinsics = self._wait_for_valid_data(self.img_node.get_camera_intrinsic, "camera intrinsics")

        # /scan_workspace는 카메라 좌표계 기준 단일 스캔 호환 서비스입니다.
        # 현재 정리 작업의 메인 경로는 /scan_pose_transform 기반 3자세 스캔입니다.
        self.create_service(ScanWorkspace, "scan_workspace", self.handle_scan_workspace)


        self.scan_accumulator = []


        #   TaskManagerNode가 /workspace_scan_mode로 알려주는 현재 스캔 목적
        #   check_workspace: 최초 확인 스캔
        #   recheck_workspace: 로봇 정리 후 최종 재검증 스캔
        self.current_scan_mode = "check_workspace"


        #   최종 재검증 3자세 스캔 중 실제로 저장한 이미지 경로 목록
        self.scan_image_records = []
        self.scan_session_id = None

        # 가장 최근 앙상블 추론에서 실제로 사용한 frame과 최종 결과입니다.
        # 재검증 이미지 저장과 /yolo_detection_image 발행이 같은 snapshot을 사용합니다.
        self.last_final_frame = None
        self.last_final_detections = []

        # VLM/재검증 저장용: 기존처럼 bbox + mask + 축 정보를 포함합니다.
        self.last_final_annotated_frame = None

        # HMI 표시용: mask 없이 bbox와 클래스명만 포함합니다.
        self.last_final_hmi_frame = None

        # HMI 상시 화면용 경량 YOLO preview 상태입니다.
        # 모델 추론과 ImgNode spin을 worker thread 하나에서만 수행해 ROS callback을 막지 않습니다.
        self.live_preview_frame = None
        self.preview_frame_lock = threading.Lock()
        self.preview_inference_lock = threading.Lock()
        self.preview_stop_event = threading.Event()
        self.preview_last_stamp = None
        self.preview_thread = None


        # 사용자가 확인하기 쉽도록 가능하면 source tree의 notification/scan_images에 저장
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

        # 이미지가 밀려 오래된 화면을 보여주지 않도록 최신 1장만 유지합니다.
        detection_image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.detection_image_pub = self.create_publisher(
            Image,
            "/yolo_detection_image",
            detection_image_qos,
        )
        self.bridge = CvBridge()
        self.create_timer(DETECTION_PREVIEW_PERIOD_SEC, self.publish_detection_image)

        # 상시 HMI preview 추론은 별도 daemon thread에서 실행합니다.
        # 실제 scan이 같은 detector를 사용할 때는 preview_inference_lock으로 자동 대기합니다.
        self.preview_thread = threading.Thread(
            target=self._live_preview_loop,
            name="hmi_yolo_preview",
            daemon=True,
        )
        self.preview_thread.start()

        self.get_logger().info(
            "ObjectDetectionNode initialized. "
            f"HMI live preview=YOLO-only, imgsz={LIVE_PREVIEW_IMGSZ}, "
            "image_qos=BEST_EFFORT/depth1"
        )

    # worker가 만든 가장 최신 preview 한 장만 /yolo_detection_image로 발행합니다.
    # HMI 상시 화면은 YOLO 단독이며, VLM 저장 이미지는 last_final_annotated_frame을 별도로 사용합니다.
    def publish_detection_image(self):
        with self.preview_frame_lock:
            annotated = None if self.live_preview_frame is None else self.live_preview_frame.copy()

        if annotated is None:
            return

        try:
            self.detection_image_pub.publish(
                self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            )
        except Exception as exc:
            self.get_logger().warn(f"HMI detection preview 발행 실패: {exc}")

    # HMI용 경량 YOLO preview를 ROS callback과 분리해 반복 실행합니다.
    # - YOLO만 사용
    # - 480 입력
    # - retina mask 비활성화
    # - 같은 camera stamp는 다시 추론하지 않음
    def _live_preview_loop(self):
        preview_model = getattr(self.model, "model", None)
        if preview_model is None:
            self.get_logger().warn("HMI preview용 YOLO 모델을 찾지 못했습니다.")
            return

        while not self.preview_stop_event.is_set() and rclpy.ok():
            # 실제 scan/get_all_detections가 모델과 ImgNode를 사용 중이면 이번 preview는 건너뜁니다.
            acquired = self.preview_inference_lock.acquire(timeout=0.01)
            if not acquired:
                self.preview_stop_event.wait(LIVE_PREVIEW_IDLE_SLEEP_SEC)
                continue

            try:
                self.img_node.spin_once(timeout_sec=0.01)
                frame = self.img_node.get_color_frame()
                stamp = self.img_node.get_color_frame_stamp()

                if frame is None or stamp is None or stamp == self.preview_last_stamp:
                    self.preview_stop_event.wait(LIVE_PREVIEW_IDLE_SLEEP_SEC)
                    continue

                self.preview_last_stamp = stamp
                results = preview_model.predict(
                    source=np.asarray(frame).copy(),
                    imgsz=LIVE_PREVIEW_IMGSZ,
                    conf=LIVE_PREVIEW_CONF,
                    max_det=LIVE_PREVIEW_MAX_DET,
                    retina_masks=False,
                    verbose=False,
                )

                if not results:
                    annotated = np.asarray(frame).copy()
                else:
                    # Ultralytics result.plot()은 segmentation mask도 함께 표시할 수 있으므로
                    # HMI 전용 renderer에서 bbox와 클래스명만 직접 그립니다.
                    annotated = self._render_live_yolo_bbox_only(
                        frame,
                        results[0],
                    )

                with self.preview_frame_lock:
                    self.live_preview_frame = annotated

            except Exception as exc:
                self.get_logger().warn(f"HMI YOLO preview 추론 실패: {exc}")
                self.preview_stop_event.wait(0.10)
            finally:
                self.preview_inference_lock.release()

            self.preview_stop_event.wait(LIVE_PREVIEW_IDLE_SLEEP_SEC)

    # detector가 보관한 마지막 frame과 최종 detection을 가져와 같은 snapshot으로 이미지를 만듭니다.
    def _cache_final_detection_snapshot(self, detections):
        frame = None
        snapshot_detections = detections

        get_snapshot = getattr(self.model, "get_last_result_snapshot", None)
        if callable(get_snapshot):
            try:
                frame, snapshot_detections = get_snapshot()
            except Exception as exc:
                self.get_logger().warn(f"앙상블 snapshot 조회 실패: {exc}")

        # YOLO 단독 호환 모드 또는 snapshot이 없는 예외 상황의 fallback입니다.
        if frame is None:
            current_frame = self.img_node.get_color_frame()
            if current_frame is not None:
                frame = np.asarray(current_frame).copy()

        if frame is None:
            self.last_final_frame = None
            self.last_final_detections = []
            self.last_final_annotated_frame = None
            self.last_final_hmi_frame = None
            return

        copied_detections = []
        for detection in snapshot_detections or []:
            item = dict(detection)
            mask = item.get("mask")
            if mask is not None:
                item["mask"] = np.asarray(mask).copy()
            copied_detections.append(item)

        self.last_final_frame = np.asarray(frame).copy()
        self.last_final_detections = copied_detections
        # VLM/재검증 저장 이미지는 기존처럼 mask 정보를 유지합니다.
        self.last_final_annotated_frame = self._render_final_detections(
            self.last_final_frame,
            self.last_final_detections,
        )

        # HMI에는 mask를 제외하고 bbox와 클래스명만 표시합니다.
        self.last_final_hmi_frame = self._render_hmi_detections(
            self.last_final_frame,
            self.last_final_detections,
        )

        # scan 직후에는 HMI에 bbox-only 최종 앙상블 snapshot을 즉시 보여줍니다.
        with self.preview_frame_lock:
            self.live_preview_frame = self.last_final_hmi_frame.copy()
        self.publish_detection_image()

    # HMI 상시 YOLO preview용 renderer입니다.
    # segmentation 결과의 mask는 사용하지 않고 bbox와 클래스명만 표시합니다.
    def _render_live_yolo_bbox_only(self, frame, result):
        image = np.asarray(frame).copy()
        height, width = image.shape[:2]

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return image

        try:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        except Exception:
            return image

        names = getattr(result, "names", {}) or {}
        color = (60, 220, 60)

        for box, class_id in zip(xyxy, class_ids):
            if len(box) != 4:
                continue

            x1, y1, x2, y2 = [int(round(float(value))) for value in box]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            if isinstance(names, dict):
                label = str(names.get(int(class_id), f"class_{int(class_id)}"))
            elif 0 <= int(class_id) < len(names):
                label = str(names[int(class_id)])
            else:
                label = f"class_{int(class_id)}"

            self._draw_bbox_class_label(
                image=image,
                box=(x1, y1, x2, y2),
                label=label,
                color=color,
            )

        return image

    # 스캔 직후 HMI에 보여줄 최종 앙상블 결과입니다.
    # mask, mask contour, 중심점, 주축, confidence/source 문구는 표시하지 않습니다.
    def _render_hmi_detections(self, frame, detections):
        image = np.asarray(frame).copy()
        height, width = image.shape[:2]

        valid_detections = []
        for detection in detections or []:
            box = detection.get("box")
            if box is None or len(box) != 4:
                continue
            valid_detections.append(detection)

        cv2.rectangle(
            image,
            (0, 0),
            (min(width - 1, 310), 34),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            f"DETECTION | objects={len(valid_detections)}",
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        for detection in valid_detections:
            box = detection["box"]
            x1, y1, x2, y2 = [int(round(float(value))) for value in box]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            source = str(detection.get("source") or "unknown")
            color = self._source_color(source)
            label = str(detection.get("name") or "unknown")

            self._draw_bbox_class_label(
                image=image,
                box=(x1, y1, x2, y2),
                label=label,
                color=color,
                min_label_y=35,
            )

        return image

    # bbox와 클래스명 라벨을 공통 형식으로 그립니다.
    def _draw_bbox_class_label(
        self,
        image,
        box,
        label,
        color,
        min_label_y=0,
    ):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            HMI_BBOX_THICKNESS,
        )

        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            HMI_CLASS_LABEL_FONT_SCALE,
            HMI_CLASS_LABEL_THICKNESS,
        )

        text_y = max(min_label_y + text_h + 4, y1 - 5)
        label_top = max(
            min_label_y,
            text_y - text_h - baseline - 4,
        )
        label_right = min(width - 1, x1 + text_w + 12)
        label_bottom = min(height - 1, text_y + baseline + 5)

        cv2.rectangle(
            image,
            (x1, label_top),
            (label_right, label_bottom),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 6, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            HMI_CLASS_LABEL_FONT_SCALE,
            color,
            HMI_CLASS_LABEL_THICKNESS,
            cv2.LINE_AA,
        )

    # YOLO+RT-DETR fusion bbox와 SAM2/Yolo 최종 mask를 원본 frame 위에 직접 그립니다.
    # 이 이미지는 HMI가 아니라 VLM/재검증 저장용으로 유지합니다.
    def _render_final_detections(self, frame, detections):
        image = np.asarray(frame).copy()
        height, width = image.shape[:2]

        overlay = image.copy()
        prepared = []

        for index, detection in enumerate(detections or []):
            source = str(detection.get("source") or "unknown")
            color = self._source_color(source)

            mask = detection.get("mask")
            mask_bool = None
            if mask is not None:
                mask_array = np.asarray(mask)
                if mask_array.ndim == 3:
                    mask_array = np.squeeze(mask_array)
                if mask_array.shape[:2] != (height, width):
                    mask_array = cv2.resize(
                        mask_array.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask_bool = mask_array > 0
                overlay[mask_bool] = color

            prepared.append((index, detection, color, mask_bool))

        image = cv2.addWeighted(
            overlay,
            FINAL_MASK_ALPHA,
            image,
            1.0 - FINAL_MASK_ALPHA,
            0.0,
        )

        cv2.rectangle(image, (0, 0), (min(width - 1, 620), 34), (0, 0, 0), -1)
        cv2.putText(
            image,
            f"FINAL ENSEMBLE | objects={len(prepared)} | bbox=YOLO+RT-DETR | mask=SAM2/fallback",
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        for _, detection, color, mask_bool in prepared:
            box = detection.get("box")
            if box is None or len(box) != 4:
                continue

            x1, y1, x2, y2 = [int(round(float(value))) for value in box]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            if mask_bool is not None:
                contours, _ = cv2.findContours(
                    mask_bool.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(image, contours, -1, color, 2)
                self._draw_mask_axis(image, mask_bool)

            name = str(detection.get("name") or "unknown")
            confidence = float(detection.get("confidence", 0.0) or 0.0)
            source = str(detection.get("source") or "unknown")
            mask_source = str(detection.get("mask_source") or "none")
            label = f"{name} {confidence:.2f} | {source} | mask:{mask_source}"

            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                1,
            )
            text_y = max(36 + text_h, y1 - 5)
            cv2.rectangle(
                image,
                (x1, max(35, text_y - text_h - baseline - 4)),
                (min(width - 1, x1 + text_w + 6), min(height - 1, text_y + baseline + 2)),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                image,
                label,
                (x1 + 3, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

        return image

    # detection source별 bbox/mask 표시 색입니다.
    def _source_color(self, source: str):
        if source == "yolo+rtdetr":
            return (60, 220, 60)
        if source == "rtdetr_only":
            return (0, 165, 255)
        if source == "yolo":
            return (255, 160, 40)
        return (220, 220, 220)

    # 최종 mask의 이미지 평면 중심과 긴 축을 표시합니다.
    # 실제 robot grasp angle은 base point cloud에서 별도로 계산됩니다.
    def _draw_mask_axis(self, image, mask_bool):
        ys, xs = np.where(mask_bool)
        if len(xs) < 10:
            return

        cx, cy = int(xs.mean()), int(ys.mean())
        points = np.column_stack((xs, ys)).astype(np.float32)
        _, eigenvectors = cv2.PCACompute(points, mean=None)
        vx, vy = float(eigenvectors[0][0]), float(eigenvectors[0][1])

        axis_length = 40
        p1 = (int(cx - axis_length * vx), int(cy - axis_length * vy))
        p2 = (int(cx + axis_length * vx), int(cy + axis_length * vy))
        cv2.line(image, p1, p2, (0, 255, 255), 2)
        cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)

    # 최종 재검증 때 저장할 scan image 경로를 결정하는 함수
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

    # model_name에 따라 detector wrapper를 선택하는 함수
    def _load_model(self, name):
        model_name = name.lower()

        if model_name == "yolo":
            return YoloModel()

        if model_name == "ensemble":
            return EnsembleDetector()

        raise ValueError(f"Unsupported model: {name}")

    # /scan_workspace 서비스 요청을 처리해 카메라 좌표계 물체 목록을 JSON 문자열로 반환한다.
    # 현재 메인 정리 흐름은 이 서비스가 아니라 handle_scan_pose()의 3자세 base 좌표 스캔을 사용한다.
    def handle_scan_workspace(self, request, response):
        self.get_logger().info(f"Received scan_workspace request: {request}")

        try:
            target_names = parse_target_names_json(request.target_names_json)

            # HMI preview worker와 같은 YOLO/ImgNode를 동시에 사용하지 않도록 잠급니다.
            with self.preview_inference_lock:
                detections = self.model.get_all_detections(
                    self.img_node,
                    target_names=target_names,
                )
                self._cache_final_detection_snapshot(detections)

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

    # robot_arm_node가 관측 자세마다 보낸 base<-camera 변환 행렬을 받아,
    # 해당 자세의 최종 detection mask를 base point cloud로 변환해 누적하는 함수
    def handle_scan_pose(self, msg):
        data = list(msg.data)

        if len(data) < 18:
            self.get_logger().error(f"scan_pose_transform 데이터 길이 오류: {len(data)}")
            return

        index = int(data[0])
        total = int(data[1])
        base_to_camera_matrix = np.array(data[2:18]).reshape(4, 4)

        self.get_logger().info(f"scan 자세 {index + 1}/{total} 물체 감지 시작")

        # [debug] 좌표 이상 조사용 임시 로그 ①: 카메라 위치(base)와 intrinsics 확인
        cam_pos = base_to_camera_matrix[:3, 3]
        self.get_logger().info(
            f"  [debug] cam_pos(base)=({cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f}) mm, "
            f"intrinsics fx={self.intrinsics['fx']:.1f} ppx={self.intrinsics['ppx']:.1f} "
            f"ppy={self.intrinsics['ppy']:.1f}"
        )

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

            # 기존에는 object마다 centroid를 한 줄씩 출력해,
            # 중복 detection이 발생하면 수십~수백 줄이 출력되었습니다.
            # 이제 자세별 전체 개수와 클래스별 개수만 한 줄로 요약합니다.
            class_counts = Counter(
                str(obj.get("name", "unknown"))
                for obj in objects
            )

            self.get_logger().info(
                f"scan 자세 {index + 1}/{total} 완료: "
                f"objects={len(objects)}, "
                f"classes={dict(class_counts)}, "
                f"누적={len(self.scan_accumulator)}, "
                f"scan_images={len(self.scan_image_records)}"
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

    # 최종 재검증 자세에서 앙상블 추론에 실제로 사용한 원본 frame과
    # 동일 detection 결과로 그린 annotated 이미지를 notification/scan_images에 저장합니다.
    def save_scan_pose_images(self, index: int, total: int):
        frame = self.last_final_frame
        annotated = self.last_final_annotated_frame

        if frame is None:
            self.get_logger().warn(
                f"scan 자세 {index + 1}/{total} 이미지 저장 실패: final ensemble frame 없음"
            )
            return None

        if self.scan_session_id is None:
            self.scan_session_id = time.strftime("recheck_%Y%m%d_%H%M%S")

        base_name = f"{self.scan_session_id}_pose_{index + 1}_of_{total}"
        raw_image_path = os.path.join(self.scan_image_dir, f"{base_name}_raw.jpg")
        annotated_image_path = os.path.join(self.scan_image_dir, f"{base_name}_annotated.jpg")

        try:
            raw_saved = bool(cv2.imwrite(raw_image_path, frame))
            annotated_saved = bool(
                annotated is not None
                and cv2.imwrite(annotated_image_path, annotated)
            )

            if not raw_saved:
                self.get_logger().warn(
                    f"scan 자세 {index + 1}/{total} raw 이미지 파일 기록 실패"
                )
                return None

            if not annotated_saved:
                annotated_image_path = ""
                self.get_logger().warn(
                    f"scan 자세 {index + 1}/{total} final ensemble annotated 이미지 기록 실패"
                )

            record = {
                "index": int(index),
                "total": int(total),
                "scan_mode": self.current_scan_mode,
                "annotation_source": "final_ensemble",
                "raw_image_path": raw_image_path,
                "annotated_image_path": annotated_image_path,
            }

            self.get_logger().info(
                f"scan 자세 {index + 1}/{total} 이미지 저장 완료: "
                f"annotation_source=final_ensemble, detections={len(self.last_final_detections)}"
            )

            return record

        except Exception as exc:
            self.get_logger().warn(f"scan 자세 {index + 1}/{total} 이미지 저장 실패: {exc}")
            return None

    # 한 scan 자세에서 detector 결과의 최종 mask를 depth와 결합해 base point cloud로 변환한다.
    # mask가 없는 detection은 현재 3D grasp 계산에 사용할 수 없으므로 제외한다.
    def _scan_and_transform(self, base_to_camera_matrix):
        # live preview와 실제 scan이 같은 YOLO/ImgNode를 동시에 실행하지 않게 합니다.
        with self.preview_inference_lock:
            detections = self.model.get_all_detections(self.img_node)
            self._cache_final_detection_snapshot(detections)

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

            # 좌표 이상을 조사할 때만 DETECTION_COORD_DEBUG=True로 바꿉니다.
            # 평상시에는 detection별 상세 좌표 로그를 생략합니다.
            if DETECTION_COORD_DEBUG:
                try:
                    ys_d, xs_d = np.where(mask > 0.5)
                    mcx, mcy = int(xs_d.mean()), int(ys_d.mean())
                    d_raw = float(depth_frame[mcy, mcx])
                    d_m = d_raw * 0.001 if d_raw > 10.0 else d_raw
                    fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
                    ppx, ppy = self.intrinsics["ppx"], self.intrinsics["ppy"]
                    cam = np.array([
                        (mcx - ppx) * d_m / fx * 1000.0,
                        (mcy - ppy) * d_m / fy * 1000.0,
                        d_m * 1000.0,
                        1.0,
                    ])
                    base_pt = base_to_camera_matrix @ cam
                    self.get_logger().info(
                        f"  [debug] {detection.get('name')}: "
                        f"mask={mask.shape} depth={depth_frame.shape} "
                        f"center_px=({mcx},{mcy}) depth={d_m:.3f}m "
                        f"cam=({cam[0]:.0f},{cam[1]:.0f},{cam[2]:.0f})mm "
                        f"base=({base_pt[0]:.0f},{base_pt[1]:.0f},{base_pt[2]:.0f})mm"
                    )
                except Exception as exc:
                    self.get_logger().warn(
                        f"  [debug] 좌표 로그 실패: {exc}"
                    )

            objects.append({
                "name": detection.get("name"),
                "class_id": detection.get("class_id"),
                "confidence": detection.get("confidence"),
                "box": detection.get("box"),
                "cloud": cloud,  # (N,3) base mm — 내부 누적용 (JSON엔 안 들어감)
            })

        return objects

    # 3자세에서 누적된 point cloud를 물체 이름 기준으로 병합한 뒤,
    # grasp position, 파지 angle, 그리드 배치용 width/length를 계산한다.
    def _build_grasp_objects(self, accumulator):
        merged = merge_clouds_by_name(accumulator)

        result = []
        for name, item in merged.items():
            try:
                (gx, gy, gz), top_ds = compute_top_center_grasp(item["cloud"])

                # 전체 발자국(손잡이+머리)은 확실히 길쭉해서 긴 축이 안정적으로 잡힌다.
                angle = top_face_angle(item["cloud"])
                # 그리드 배치용 물체 폭/길이(mm)도 전체 cloud 기준으로 계산한다.
                width, length = footprint_extent(item["cloud"], angle)
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
                # 그리드 배치(grid_allocator)용 크기 정보
                "width": width,
                "length": length,
                "major_axis_angle": angle,
                "detected_pose_count": item["count"],
            })

            angle_str = f"{angle:.1f}" if angle is not None else "None"
            size_str = (f"{width:.1f}x{length:.1f}"
                        if width is not None and length is not None else "None")
            self.get_logger().info(
                f"  [grasp] {name}: pos=({gx:.1f}, {gy:.1f}, {gz:.1f}) mm, "
                f"angle={angle_str}, size(wxl)={size_str}"
            )

        return result


    # 아래 get_3d_position 관련 코드는 구버전 단일 물체 조회 흐름입니다.
    # 현재 메인 흐름에서는 사용하지 않지만, pick 직전 재확인 기능을 되살릴 때 참고할 수 있습니다.

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
            self.img_node.spin_once(timeout_sec=0.10)
            self.get_logger().info(f"Retry getting {description}.")
            data = getter()

        return data


# ROS2 노드를 생성하고 spin한다.
    # preview worker를 정리한 뒤 ROS node를 종료합니다.
    def destroy_node(self):
        self.preview_stop_event.set()
        if self.preview_thread is not None and self.preview_thread.is_alive():
            self.preview_thread.join(timeout=1.0)
        try:
            self.img_node.close_executor()
            self.img_node.destroy_node()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()