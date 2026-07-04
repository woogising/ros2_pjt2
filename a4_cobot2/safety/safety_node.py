import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Bool
from std_msgs.msg import String
from safety.safety_constants import (
    SAFETY_COMMAND_CLEAR,
    SAFETY_COMMAND_STOP,
    SAFETY_STATE_NORMAL,
    SAFETY_STATE_EMERGENCY_STOP,
)

class SafetyNode(Node):
    # safety_node를 초기화하고 stop/clear 명령 구독자와 emergency_stop, safety_state 발행자를 준비하는 함수
    def __init__(self):
        super().__init__('safety_node')

        self.emergency_stop_active = False

        self.command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )

        self.state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.safety_command_sub = self.create_subscription(
            String,
            '/safety_command',
            self.safety_command_callback,
            self.command_qos
        )

        self.emergency_stop_pub = self.create_publisher(
            Bool,
            '/emergency_stop',
            self.state_qos
        )

        self.safety_state_pub = self.create_publisher(
            String,
            '/safety_state',
            self.state_qos
        )

        self.get_logger().info('SafetyNode started.')
        self.publish_safety_state()

    # /safety_command 토픽으로 들어온 stop 또는 clear 명령을 처리하는 함수
    def safety_command_callback(self, msg: String):
        command = msg.data.strip().lower()

        self.get_logger().info(f'Received safety command: {command}')

        if command == SAFETY_COMMAND_STOP:
            self.set_emergency_stop(True, reason='stop_command')

        elif command == SAFETY_COMMAND_CLEAR:
            self.set_emergency_stop(False, reason='clear_command')

        else:
            self.get_logger().warn(f'Unknown safety command: {command}')
            self.publish_safety_state(extra_state=f'unknown_safety_command:{command}')

    # emergency stop 상태를 변경하고 관련 토픽을 발행하는 함수
    def set_emergency_stop(self, is_active: bool, reason: str):
        self.emergency_stop_active = is_active

        if is_active:
            self.get_logger().warn(f'Emergency stop activated. reason={reason}')
        else:
            self.get_logger().info(f'Emergency stop cleared. reason={reason}')

        self.publish_emergency_stop()
        self.publish_safety_state()

    # 현재 emergency stop 상태를 /emergency_stop 토픽으로 발행하는 함수
    def publish_emergency_stop(self):
        msg = Bool()
        msg.data = self.emergency_stop_active

        self.emergency_stop_pub.publish(msg)

        self.get_logger().info(f'Published /emergency_stop: {msg.data}')

    # 현재 safety 상태를 /safety_state 토픽으로 발행하는 함수
    def publish_safety_state(self, extra_state: str = None):
        msg = String()

        if extra_state is not None:
            msg.data = extra_state
        elif self.emergency_stop_active:
            msg.data = SAFETY_STATE_EMERGENCY_STOP
        else:
            msg.data = SAFETY_STATE_NORMAL

        self.safety_state_pub.publish(msg)

        self.get_logger().info(f'Published /safety_state: {msg.data}')


# ROS2 safety_node를 실행하고 safety command callback을 계속 처리하는 메인 함수
def main(args=None):
    rclpy.init(args=args)

    node = SafetyNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt로 종료합니다.')

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()