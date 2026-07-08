# ============================================================
# hmi/hmi_ros_bridge.py
# 역할:
#   - PyQt HMI와 ROS2 topic을 연결하는 bridge입니다.
#
# HMI -> ROS2
#   - /task_command 로 String command publish
#
# ROS2 -> HMI
#   - /task_status 구독
#   - /user_notice 구독
#   - /safety_state 구독
#
# 현재 팀 구조 기준:
#   command:
#     check_workspace
#     start_organize
#     stop
#     shutdown
# ============================================================

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

try:
    from task_manager.task_config import (
        TOPIC_TASK_COMMAND,
        TOPIC_TASK_STATUS,
        TOPIC_USER_NOTICE,
    )
except Exception:
    TOPIC_TASK_COMMAND = "/task_command"
    TOPIC_TASK_STATUS = "/task_status"
    TOPIC_USER_NOTICE = "/user_notice"


TOPIC_SAFETY_STATE = "/safety_state"
TOPIC_DETECTION_IMAGE = "/yolo_detection_image"


class HmiRosNode(Node):
    def __init__(self, bridge):
        super().__init__("hmi_interface_node")

        self.bridge = bridge

        # HMI -> task_manager_node
        self.task_command_pub = self.create_publisher(
            String,
            TOPIC_TASK_COMMAND,
            10
        )

        # task_manager_node -> HMI
        self.task_status_sub = self.create_subscription(
            String,
            TOPIC_TASK_STATUS,
            self.task_status_callback,
            10
        )

        # task_manager_node/status_notifier 관련 사용자 안내 -> HMI
        self.user_notice_sub = self.create_subscription(
            String,
            TOPIC_USER_NOTICE,
            self.user_notice_callback,
            10
        )

        # safety_node -> HMI
        # safety_node가 /safety_state를 TRANSIENT_LOCAL(래치)로 발행하므로,
        # 시작 시 마지막 안전 상태를 받으려면 구독도 TRANSIENT_LOCAL로 맞춘다.
        safety_state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.safety_state_sub = self.create_subscription(
            String,
            TOPIC_SAFETY_STATE,
            self.safety_state_callback,
            safety_state_qos
        )

        # object_detection_node -> HMI (YOLO 인식 화면)
        self.detection_image_sub = self.create_subscription(
            Image,
            TOPIC_DETECTION_IMAGE,
            self.detection_image_callback,
            10
        )

        self.get_logger().info("HMI ROS bridge node started.")
        self.bridge.log_signal.emit("HMI ROS bridge node started")

    def publish_task_command(self, command: str):
        msg = String()
        msg.data = command
        self.task_command_pub.publish(msg)

        self.get_logger().info(f"Published task command: {command}")
        self.bridge.log_signal.emit(f"Published /task_command: {command}")

    def task_status_callback(self, msg: String):
        status = msg.data.strip()
        self.bridge.task_status_signal.emit(status)

    def user_notice_callback(self, msg: String):
        notice = msg.data.strip()
        if notice:
            self.bridge.user_notice_signal.emit(notice)

    def safety_state_callback(self, msg: String):
        safety_state = msg.data.strip()
        if safety_state:
            self.bridge.safety_state_signal.emit(safety_state)

    def detection_image_callback(self, msg: Image):
        # cv2/cv_bridge를 쓰면 opencv 번들 Qt 플러그인이 PyQt5와 충돌하므로,
        # ROS Image 메시지에서 직접 QImage를 만든다.
        try:
            image = QImage(
                bytes(msg.data), msg.width, msg.height, msg.step, QImage.Format_RGB888
            )
            if msg.encoding == "bgr8":
                qimage = image.rgbSwapped()  # BGR → RGB (복사본 생성)
            else:
                qimage = image.copy()  # 버퍼 소유
            self.bridge.detection_image_signal.emit(qimage)
        except Exception as exc:
            self.get_logger().warn(f"detection image 변환 실패: {exc}")


class HmiRosBridge(QThread):
    task_status_signal = pyqtSignal(str)
    user_notice_signal = pyqtSignal(str)
    safety_state_signal = pyqtSignal(str)
    detection_image_signal = pyqtSignal(QImage)
    log_signal = pyqtSignal(str)
    connected_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.node = None
        self.running = True

    def run(self):
        try:
            rclpy.init(args=None)

            self.node = HmiRosNode(self)
            self.connected_signal.emit(True)

            while self.running and rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.1)

        except Exception as e:
            self.connected_signal.emit(False)
            self.log_signal.emit(f"ROS bridge error: {e}")

        finally:
            try:
                if self.node is not None:
                    self.node.destroy_node()

                if rclpy.ok():
                    rclpy.shutdown()

            except Exception as e:
                self.log_signal.emit(f"ROS shutdown error: {e}")

    def publish_command(self, command: str):
        if self.node is None:
            self.log_signal.emit(f"ROS node is not ready. Command ignored: {command}")
            return

        self.node.publish_task_command(command)

    def stop_bridge(self):
        self.running = False
