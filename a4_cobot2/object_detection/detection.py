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
import numpy as np
import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory
from od_msg.srv import SrvDepthPosition, ScanWorkspace
from object_detection.realsense import ImgNode
from object_detection.yolo import YoloModel
from object_detection.detection_utils import (
    parse_target_names_json,
    get_box_center,
    get_depth_from_frame,
    pixel_to_camera_coords,
    build_detected_object,
    make_scan_workspace_payload,
    make_scan_workspace_error_payload,
)


PACKAGE_NAME = "a4_cobot2"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)


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

        self.get_logger().info("ObjectDetectionNode initialized.")

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