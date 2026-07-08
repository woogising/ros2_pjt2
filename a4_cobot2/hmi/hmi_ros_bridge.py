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
from std_msgs.msg import String

from PyQt5.QtCore import QThread, pyqtSignal

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
        self.safety_state_sub = self.create_subscription(
            String,
            TOPIC_SAFETY_STATE,
            self.safety_state_callback,
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


class HmiRosBridge(QThread):
    task_status_signal = pyqtSignal(str)
    user_notice_signal = pyqtSignal(str)
    safety_state_signal = pyqtSignal(str)
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
