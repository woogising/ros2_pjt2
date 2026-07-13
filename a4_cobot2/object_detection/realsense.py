# ============================================================
# object_detection/realsense.py
# 역할:
#   - RealSense 관련 ROS topic을 구독해서 최신 RGB frame, aligned depth frame, camera intrinsics를 보관합니다.
# 구독 topic:
#   - /camera/camera/color/image_raw
#   - /camera/camera/aligned_depth_to_color/image_raw
#   - /camera/camera/color/camera_info
# 주의:
#   - depth는 color에 aligned된 topic을 사용해야 bbox 중심 픽셀로 depth를 읽을 수 있습니다.
# ============================================================
import threading

from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class ImgNode(Node):
    def __init__(self):
        super().__init__('img_node')
        self.bridge = CvBridge()

        # ImgNode는 ObjectDetectionNode와 별도 노드입니다.
        # rclpy.spin()/rclpy.spin_once()의 기본 전역 executor를 여러 스레드에서
        # 동시에 사용하면 ROS2 Humble에서 wait set index 오류가 날 수 있습니다.
        # 따라서 카메라 구독 전용 executor를 하나 만들고 모든 spin_once를
        # 하나의 lock으로 직렬화합니다.
        self._executor_lock = threading.RLock()
        self._executor_closed = False
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        # color_frame:
        #   YOLO 입력으로 사용할 최신 RGB 이미지입니다. OpenCV BGR 형식으로 저장됩니다.
        self.color_frame = None

        # color_frame_stamp:
        #   같은 frame을 여러 번 중복 저장하지 않기 위한 timestamp 문자열입니다.
        self.color_frame_stamp = None

        # depth_frame:
        #   color image에 aligned된 depth image입니다. bbox 중심 픽셀에서 depth를 읽습니다.
        self.depth_frame = None

        # intrinsics:
        #   CameraInfo.K에서 fx, fy, ppx, ppy를 추출한 dict입니다.
        #   pixel 좌표를 camera 3D 좌표로 바꿀 때 사용합니다.
        self.intrinsics = None
        self.color_subscription = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.depth_subscription = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.camera_info_subscription = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        self.get_logger().info("Waiting for client's call...")

    def camera_info_callback(self, msg):
        self.intrinsics = {"fx": msg.k[0], "fy": msg.k[4], "ppx": msg.k[2], "ppy": msg.k[5]}

    def color_callback(self, msg):
        self.color_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.color_frame_stamp = str(msg.header.stamp.sec) + str(msg.header.stamp.nanosec)

    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def get_color_frame(self):
        return self.color_frame

    def get_color_frame_stamp(self):
        return self.color_frame_stamp

    def get_depth_frame(self):
        return self.depth_frame

    def get_camera_intrinsic(self):
        return self.intrinsics


    def spin_once(self, timeout_sec=0.0):
        """카메라 callback만 처리하는 전용 executor를 안전하게 한 번 실행합니다."""
        with self._executor_lock:
            if self._executor_closed:
                return
            self._executor.spin_once(timeout_sec=timeout_sec)

    def close_executor(self):
        """종료 시 ImgNode 전용 executor를 먼저 정리합니다."""
        with self._executor_lock:
            if self._executor_closed:
                return
            self._executor_closed = True
            try:
                self._executor.remove_node(self)
            except Exception:
                pass
            try:
                self._executor.shutdown(timeout_sec=0.5)
            except Exception:
                pass
